"""Qwen3-TTS fine-tuning export: one speaker's clips as the folder Qwen's own scripts read.

The layout follows the official ``finetuning/`` code of QwenLM/Qwen3-TTS (see ``qwen3_colab`` for the
commit that was read): a ``train.jsonl`` whose rows hold exactly ``audio``, ``text`` and ``ref_audio``
(one shared reference clip), WAVs at 24 kHz mono, and relative POSIX paths. Qwen's ``prepare_data.py``
adds the ``audio_codes`` itself; nothing here trains or contacts a network.

Audio handling is deliberately plain: mono, 24 kHz, 16-bit PCM and nothing else (no compression, EQ,
denoising or per-clip normalisation). Level matching to one target is applied when the caller passes a target (the dialog and the CLI default to -22 dB).
Every clip that is not exported is listed with its reason in ``rejected_samples.jsonl``.
"""
from __future__ import annotations

import collections
import dataclasses
import hashlib
import json
import logging
import os
import re
import shutil
import statistics
import zipfile
from typing import Optional

from . import dataset, dataset_quality, qwen3_colab

logger = logging.getLogger("omnivoice.fish.qwen3_export")

FORMAT = "qwen3_tts"
RATE = qwen3_colab.SAMPLE_RATE
REFERENCE_PATH = "reference/ref.wav"
SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?…])")
LEFTOVER_MARKUP = re.compile(r"[\[\]<>{}*_|]")
# A clip tagged with any of these (substring match) is not "the normal voice" for the reference.
EXPRESSIVE_HINTS = (
    "shout", "scream", "yell", "whisper", "angry", "furious", "rage", "excit", "sad", "cry", "sob", "laugh",
    "giggle", "chuckl", "gasp", "panic", "terrif", "fear", "moan", "groan", "hysteric", "sarcas", "emotion",
    "tearful", "trembl", "sigh", "breath", "stutter", "scared", "mock",
)
CLIP_FRACTION = 0.0005  # share of samples at full scale above which a clip counts as clipped
SILENT_DB = -55.0
LOUDNESS_SPREAD_DB = 6.0
SHORT_CLIP_S = 2.0


@dataclasses.dataclass
class Options:
    speaker_name: str = ""
    reference_key: Optional[str] = None
    reference_file: Optional[str] = None
    make_zip: bool = False
    overwrite: bool = False
    min_duration: float = 0.8
    max_duration: float = 30.0


def options_from(spec: Optional[dict]) -> Options:
    """Validated options; unknown keys and out-of-range values raise ``ValueError``."""
    data = dict(spec or {})
    unknown = sorted(set(data) - {f.name for f in dataclasses.fields(Options)})
    if unknown:
        raise ValueError("Unknown Qwen3-TTS option(s): " + ", ".join(unknown) + ".")
    opts = Options(**{k: v for k, v in data.items() if v is not None})
    opts.speaker_name = (opts.speaker_name or "").strip()
    if opts.speaker_name and not qwen3_colab.SPEAKER_RE.fullmatch(opts.speaker_name):
        raise ValueError("The speaker name may use letters, digits, '-' and '_' only (1 to 40 characters); "
                         "it goes into a shell command and into the model's config.")
    if not 0.1 <= float(opts.min_duration) < float(opts.max_duration) <= 120:
        raise ValueError("Use 0.1 <= min duration < max duration <= 120 seconds.")
    opts.min_duration, opts.max_duration = float(opts.min_duration), float(opts.max_duration)
    if opts.reference_key and opts.reference_file:
        raise ValueError("Choose a reference clip or a reference file, not both.")
    return opts


def _clean_error(exc: Exception, src: str, rel: str) -> str:
    """An error message with the machine's paths replaced by the root-relative name (no path may leak)."""
    msg = str(exc)
    for variant in (src, os.path.normpath(src), os.path.abspath(src), src.replace("\\", "/"), src.replace("\\", "\\\\")):
        msg = msg.replace(variant, rel)
    msg = re.sub(r"(?<![A-Za-z])[A-Za-z]:[\\/][^\s'\"]*", "<path>", msg)
    return f"{type(exc).__name__}: {msg}"[:200]


# ---- reading one clip --------------------------------------------------------------------------
def _inspect(path: str) -> dict:
    """Decode one clip and measure it. Raises on anything unreadable."""
    import soundfile as sf

    info = sf.info(path)
    data, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    out = {"sample_rate": int(rate), "channels": int(info.channels), "subtype": info.subtype,
           "frames": int(len(mono)), "size": os.path.getsize(path)}
    if len(mono) == 0:
        return out | {"duration": 0.0}
    stats = dataset_quality.measure_data(mono, rate)
    return out | {"duration": len(mono) / rate, "stats": stats,
                  "content_sha": hashlib.sha256(mono.tobytes()).hexdigest()}


def _convert(path: str, target: str, loudness_db: Optional[float]) -> dict:
    """Write ``target`` as 24 kHz mono PCM16. A file that already is that is copied byte for byte.

    With ``loudness_db`` the result also carries the level before and after (``before_db``, ``after_db``)."""
    import numpy as np
    import soundfile as sf

    info = sf.info(path)
    if (loudness_db is None and info.samplerate == RATE and info.channels == 1 and info.subtype == "PCM_16"
            and info.format == "WAV"):
        shutil.copyfile(path, target)
        return {"copied": True, "gain_db": 0.0, "peak_limited": False}
    data, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = dataset.resample_mono(data, rate, RATE)
    level = {"gain_db": 0.0, "peak_limited": False}
    if loudness_db is not None:
        mono, level = dataset.apply_loudness(mono, loudness_db)
    sf.write(target, np.clip(mono, -1.0, 1.0), RATE, subtype="PCM_16")
    return {"copied": False, **level, "gain_db": round(level["gain_db"], 2)}


def spoken_text(raw: str) -> str:
    """The plain transcript: every ``[tag]`` removed (empty dictionary), spaces and punctuation tidied."""
    text = dataset.strip_tags(raw or "", {})
    return SPACE_BEFORE_PUNCT.sub(r"\1", text).strip()


# ---- reference choice --------------------------------------------------------------------------
def _is_clipped(stats: dict) -> bool:
    return stats["clip"] > CLIP_FRACTION or stats["peak"] >= 0.999


def _expressive(tags: list[str]) -> bool:
    return any(h in t for t in tags for h in EXPRESSIVE_HINTS)


# Each tier relaxes the one before it; a clip belongs to the first tier it passes (the tiers are nested).
REFERENCE_TIERS = (
    ("", lambda c: 5 <= c["duration"] <= 15 and c["stats"]["pauses"] <= 2 and c["stats"]["lead_ms"] <= 600
     and c["stats"]["tail_ms"] <= 900 and c["stats"]["noise_floor_db"] <= -40 and not _expressive(c["tags"])
     and not _is_clipped(c["stats"])),
    ("relaxed length and pauses", lambda c: 3 <= c["duration"] <= 20 and c["stats"]["noise_floor_db"] <= -40
     and not _expressive(c["tags"]) and not _is_clipped(c["stats"])),
    ("allowed an expressive tag", lambda c: 3 <= c["duration"] <= 20 and not _is_clipped(c["stats"])),
    ("every clip is clipped or off-length; took the closest", lambda c: True),
)


def _reasons(c: dict, middle: float) -> tuple[list[str], list[str]]:
    """Codes the dialog turns into words: what is good about a candidate and what to be careful about."""
    s = c["stats"]
    good: list[str] = []
    caution: list[str] = []
    (caution if _is_clipped(s) else good).append("clipped" if _is_clipped(s) else "not_clipped")
    (good if s["noise_floor_db"] <= -40 else caution).append("clean_background" if s["noise_floor_db"] <= -40 else "noisy")
    (good if 5 <= c["duration"] <= 15 else caution).append("medium_length" if 5 <= c["duration"] <= 15 else "off_length")
    (caution if _expressive(c["tags"]) else good).append("expressive_tag" if _expressive(c["tags"]) else "neutral")
    (good if s["pauses"] <= 2 else caution).append("few_pauses" if s["pauses"] <= 2 else "many_pauses")
    if abs(s["rms_db"] - middle) <= 3:
        good.append("typical_level")
    return good, caution


def rank_references(clips: list[dict], limit: Optional[int] = None) -> list[dict]:
    """Every candidate reference clip, best first. The ONE place the rules live: the exporter takes the
    first entry when no reference is chosen and the Training data screen lists the top few.

    ``clips`` need ``duration``, ``tags``, ``number`` (a stable order) and ``stats`` (the
    ``dataset_quality`` numbers including ``noise_floor_db``). Ties go to the lower ``number``, so the
    result is the same on every run. Each entry: ``clip``, ``score``, ``tier``, ``note``, ``reasons``,
    ``cautions`` and ``median_db`` (the level of a typical clip of the speaker).
    """
    if not clips:
        return []
    middle = statistics.median(c["stats"]["rms_db"] for c in clips)
    ranked = []
    for c in clips:
        s = c["stats"]
        tier = next(i for i, (_, ok) in enumerate(REFERENCE_TIERS) if ok(c))
        score = (-abs(c["duration"] - 9.0) / 9.0 - 0.5 * abs(s["rms_db"] - middle) / 6.0 - 0.2 * s["pauses"]
                 - (0.3 if c["tags"] else 0.0) - (1.0 if _is_clipped(s) else 0.0))
        good, caution = _reasons(c, middle)
        ranked.append({"clip": c, "score": round(score, 6), "tier": tier, "note": REFERENCE_TIERS[tier][0],
                       "reasons": good, "cautions": caution, "median_db": middle})
    ranked.sort(key=lambda r: (r["tier"], -r["score"], r["clip"]["number"]))
    return ranked if limit is None else ranked[:limit]


def reference_why(entry: dict) -> str:
    """A sentence about one ranked candidate, for the export report."""
    best, s, note = entry["clip"], entry["clip"]["stats"], entry["note"]
    return (f"{best['duration']:.1f} s, peak {s['peak']:.2f}, "
            f"{'no clipping' if not _is_clipped(s) else 'some clipping'}, {s['pauses']} pause(s), "
            f"level {s['rms_db']:.1f} dB (speaker median {entry['median_db']:.1f} dB), "
            f"{'no emotion tags' if not _expressive(best['tags']) else 'has an emotion tag'}"
            f"{'; ' + note if note else ''}")


def choose_reference(clips: list[dict]) -> tuple[dict, str]:
    """The best-looking neutral clip among accepted ``clips`` and a sentence saying why."""
    best = rank_references(clips, 1)[0]
    return best["clip"], reference_why(best)


def speaker_lines(corpus, f: dict) -> tuple[list[dict], str, str]:
    """The selected clips in export order and their one speaker's id and name; refuses an empty or
    multi-speaker selection (the Qwen3-TTS format trains one speaker)."""
    lines = dataset.select(corpus, {**f, "sort": "order"})
    if not lines:
        raise ValueError("Nothing matches the current filters.")
    voices = collections.OrderedDict((l["voice_id"], l["voice_name"]) for l in lines)
    if len(voices) > 1:
        raise ValueError("The Qwen3-TTS format trains one speaker, and the selection holds "
                         f"{len(voices)} ({', '.join(sorted(voices.values()))}). Pick one speaker in the filters.")
    voice_id, voice_name = next(iter(voices.items()))
    lines.sort(key=lambda l: (l["chapter"], l["started_at"], l["index"], l["key"]))
    return lines, voice_id, voice_name


def candidate_clips(lines: list[dict], min_duration: float = 0.8, max_duration: float = 30.0) -> tuple[list[dict], int]:
    """Reference candidates from clips already measured (``lines`` in export order), and how many clips
    could not be judged because they have no cached measurements yet."""
    clips, unmeasured = [], 0
    for number, line in enumerate(lines, 1):
        text = spoken_text(line["text"])
        if not text or not min_duration <= line["duration_s"] <= max_duration:
            continue
        stats = dataset_quality.get(line)
        if not stats or "noise_floor_db" not in stats:
            unmeasured += 1
            continue
        if stats["rms_db"] < SILENT_DB:
            continue
        clips.append({"duration": line["duration_s"], "stats": stats, "tags": line["tags"], "number": number,
                      "line": line, "text": text})
    return clips, unmeasured


# ---- the export --------------------------------------------------------------------------------
def _is_previous_export(path: str) -> bool:
    try:
        with open(os.path.join(path, "dataset_report.json"), encoding="utf-8") as fh:
            return json.load(fh).get("format") == FORMAT
    except (OSError, ValueError):
        return False


def _write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def _write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _file_sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _hms(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 3600}h {s % 3600 // 60:02d}m {s % 60:02d}s"


def make_zip(folder: str, archive: str) -> int:
    """Zip ``folder`` with the hierarchy preserved and the files at the archive root (so
    ``unzip x.zip -d /content/x`` gives ``/content/x/train.jsonl``). Same input, same bytes."""
    count = 0
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for base, dirs, files in os.walk(folder):
            dirs.sort()
            for name in sorted(files):
                full = os.path.join(base, name)
                arc = os.path.relpath(full, folder).replace(os.sep, "/")
                info = zipfile.ZipInfo(arc, date_time=(2020, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                with open(full, "rb") as fh:
                    zf.writestr(info, fh.read())
                count += 1
    return count


def export(corpus, f: dict, dest: str, *, name: str = "fish-dataset", loudness_db: Optional[float] = None,
           options: Optional[Options] = None, settings: Optional[qwen3_colab.TrainSettings] = None) -> dict:
    """Write the selected clips of ONE speaker under ``dest/<name>/`` in Qwen3-TTS's format."""
    opts = options or Options()
    settings = settings or qwen3_colab.TrainSettings()
    lines, voice_id, voice_name = speaker_lines(corpus, f)
    raw = (dest or "").strip().strip('"')
    if not raw or not os.path.isabs(raw):
        raise ValueError("Enter the full path of the folder to export into.")
    folder_name = dataset.slug(name, "fish-dataset")
    out_dir = os.path.join(os.path.normpath(raw), folder_name)
    archive = out_dir + ".zip"
    if os.path.exists(out_dir) and os.listdir(out_dir):
        if not (opts.overwrite and _is_previous_export(out_dir)):
            raise ValueError("That export folder already has files. Choose a new name or folder"
                             + (" (--overwrite only replaces a previous Qwen3-TTS export)." if opts.overwrite else "."))
        shutil.rmtree(out_dir)
    if opts.make_zip and os.path.exists(archive):
        if not opts.overwrite:
            raise ValueError(f"{os.path.basename(archive)} already exists next to the export folder.")
        os.remove(archive)
    speaker = opts.speaker_name or dataset.slug(voice_name, "speaker")[:40]
    if not qwen3_colab.SPEAKER_RE.fullmatch(speaker):
        speaker = "speaker"

    # Pass 1: read and judge every clip. Nothing is dropped without a record.
    accepted: list[dict] = []
    rejected: list[dict] = []
    seen_paths: set[str] = set()
    seen_audio: dict[str, str] = {}

    def reject(line: dict, reason: str, detail: str, duration: Optional[float] = None) -> None:
        rejected.append({"source": line["file"], "reason": reason, "detail": detail,
                         "text": spoken_text(line["text"]), "duration": None if duration is None else round(duration, 3),
                         "key": line["key"]})

    for line in lines:
        src = os.path.join(corpus.root, line["file"])
        text = spoken_text(line["text"])
        if line["file"] in seen_paths:
            reject(line, "duplicate_source_path", "the same file is listed more than once")
            continue
        seen_paths.add(line["file"])
        if not (line["text"] or "").strip():
            reject(line, "missing_transcript", "no transcript for this clip")
            continue
        if not text:
            reject(line, "empty_transcript", "the transcript holds only tags, no spoken words")
            continue
        try:
            clip = _inspect(src)
        except Exception as exc:  # noqa: BLE001 — any decode failure is a recorded rejection
            reject(line, "unreadable_audio", _clean_error(exc, src, line["file"]))
            continue
        if clip["duration"] <= 0:
            reject(line, "empty_audio", "the file has no samples", 0.0)
            continue
        if clip["duration"] < opts.min_duration:
            reject(line, "too_short", f"{clip['duration']:.2f} s is under the {opts.min_duration:g} s minimum", clip["duration"])
            continue
        if clip["duration"] > opts.max_duration:
            reject(line, "too_long", f"{clip['duration']:.2f} s is over the {opts.max_duration:g} s maximum", clip["duration"])
            continue
        if clip["stats"]["rms_db"] < SILENT_DB:
            reject(line, "silent", f"average level {clip['stats']['rms_db']:.0f} dB: no meaningful speech", clip["duration"])
            continue
        first = seen_audio.get(clip["content_sha"])
        if first:
            reject(line, "duplicate_audio", f"identical audio to {first}", clip["duration"])
            continue
        seen_audio[clip["content_sha"]] = line["file"]
        accepted.append({**clip, "line": line, "text": text, "tags": line["tags"], "number": len(accepted) + 1})

    if not accepted:
        reasons = collections.Counter(r["reason"] for r in rejected)
        raise ValueError("No clip passed the checks (" + ", ".join(f"{n} {r}" for r, n in reasons.most_common()) + ").")

    # The reference clip.
    reference = {"mode": "auto"}
    ref_source: Optional[str] = None
    if opts.reference_file:
        ref_source = os.path.normpath(opts.reference_file.strip().strip('"'))
        if not os.path.isfile(ref_source):
            raise ValueError("The reference file does not exist.")
        try:
            ref = _inspect(ref_source)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"The reference file cannot be read ({type(exc).__name__}).") from exc
        if not 2.0 <= ref["duration"] <= 60.0:
            raise ValueError("The reference file should be between 2 and 60 seconds long.")
        reference = {"mode": "file", "source": os.path.basename(ref_source), "duration": round(ref["duration"], 3),
                     "reason": "chosen by you (a file outside the dataset)"}
    else:
        if opts.reference_key:
            chosen = next((l for l in corpus.lines if l["key"] == opts.reference_key), None)
            if chosen is None:
                raise ValueError("The chosen reference clip is not in the corpus (rescan the folder and choose again).")
            if chosen["voice_id"] != voice_id:
                raise ValueError(f"The chosen reference clip is spoken by {chosen['voice_name']}, but this export trains "
                                 f"{voice_name}. Choose a clip of {voice_name} (one speaker per export).")
            picked = next((c for c in accepted if c["line"]["key"] == opts.reference_key), None)
            if picked is None:
                rej = next((r for r in rejected if r["key"] == opts.reference_key), None)
                raise ValueError("The chosen reference clip is not part of the export"
                                 + (f" ({rej['reason']})." if rej else " (it is not in the selection)."))
            why = "chosen by you"
            reference = {"mode": "clip"}
        else:
            picked, why = choose_reference(accepted)
        ref_source = os.path.join(corpus.root, picked["line"]["file"])
        reference |= {"source": picked["line"]["file"], "duration": round(picked["duration"], 3), "reason": why,
                      "exported_as": f"audio/{picked['number']:06d}.wav"}
    # (the reference clip also stays in the training set: the official example does the same)

    # Pass 2: write the files.
    os.makedirs(os.path.join(out_dir, "audio"))
    os.makedirs(os.path.join(out_dir, "reference"))
    converted = copied = 0
    _convert(ref_source, os.path.join(out_dir, "reference", "ref.wav"), loudness_db)
    import soundfile as sf

    ref_info_wav = sf.info(os.path.join(out_dir, "reference", "ref.wav"))
    if ref_info_wav.samplerate != RATE or ref_info_wav.channels != 1:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise ValueError(f"The reference clip could not be converted to {RATE} Hz mono; choose another one.")
    levels = dataset.LevelReport(loudness_db) if loudness_db is not None else None
    mapping: list[dict] = []
    train_rows: list[dict] = []
    for c in accepted:
        rel = f"audio/{c['number']:06d}.wav"
        result = _convert(os.path.join(corpus.root, c["line"]["file"]), os.path.join(out_dir, *rel.split("/")), loudness_db)
        copied += int(result["copied"])
        converted += int(not result["copied"])
        if levels is not None:
            levels.add(c["line"]["chapter"], result)
        train_rows.append({"audio": rel, "text": c["text"], "ref_audio": REFERENCE_PATH})
        mapping.append({"original": c["line"]["file"], "exported": rel, "text": c["text"],
                        "duration_s": round(c["duration"], 3), "chapter": c["line"]["chapter"],
                        "tags_removed": c["tags"], "sha256": _file_sha(os.path.join(out_dir, *rel.split("/"))),
                        "gain_db": result["gain_db"]}
                       | ({"level_after_db": round(result["after_db"], 1), "peak_limited": result["peak_limited"]}
                          if levels is not None else {}))

    # Warnings.
    warnings: list[dict] = []

    def warn(code: str, message: str, count: int = 0, examples: Optional[list] = None) -> None:
        warnings.append({"code": code, "message": message, "count": count, "examples": (examples or [])[:5]})

    n = len(accepted)
    durations = [c["duration"] for c in accepted]
    short = [c for c in accepted if c["duration"] < SHORT_CLIP_S]
    if len(short) > max(3, n * 0.25):
        warn("many_short_clips", f"{len(short)} of {n} clips are under {SHORT_CLIP_S:g} s; very short clips teach little prosody.",
             len(short), [c["line"]["file"] for c in short])
    by_chapter: dict[str, list[float]] = collections.defaultdict(list)
    for c in accepted:
        by_chapter[c["line"]["chapter"]].append(c["stats"]["rms_db"])
    chapter_levels = {ch: round(statistics.median(v), 1) for ch, v in sorted(by_chapter.items())}
    level_summary = levels.summary() if levels is not None else None
    # What the model will actually hear: the matched levels when level matching ran, the source levels otherwise.
    heard = levels.chapter_levels("after_db") if levels is not None else chapter_levels
    if len(heard) > 1:
        centre = statistics.median(heard.values())
        off = {ch: lv for ch, lv in heard.items() if abs(lv - centre) >= LOUDNESS_SPREAD_DB}
        if off:
            advice = ("Those chapters are peak-limited, so they stayed quieter than the target; choose a lower target "
                      "or leave them out." if levels is not None else
                      "Tick level matching in the export or leave those chapters out.")
            warn("inconsistent_loudness",
                 f"Chapter levels differ by {max(heard.values()) - min(heard.values()):.1f} dB "
                 f"(median {centre:.1f} dB). Chapters {', '.join(f'{ch} ({lv} dB)' for ch, lv in off.items())} are "
                 f"{LOUDNESS_SPREAD_DB:g} dB or more away. {advice}", len(off), list(off))
    if level_summary and level_summary["peak_limited"] > max(2, n * 0.1):
        warn("loudness_peak_limited",
             f"{level_summary['peak_limited']} of {n} clip(s) could not reach {loudness_db:g} dB because the peak "
             f"cap ({dataset.PEAK_CEILING:g}) stopped the gain; the worst is {level_summary['worst_shortfall_db']:.1f} dB "
             "short. Choose a lower target level if that matters.", level_summary["peak_limited"],
             [m["original"] for m in mapping if m.get("peak_limited")])
    clipped = [c for c in accepted if _is_clipped(c["stats"])]
    if clipped:
        warn("clipping", f"{len(clipped)} clip(s) touch full scale; the audio was not repaired.", len(clipped),
             [c["line"]["file"] for c in clipped])
    texts = collections.defaultdict(list)
    for c in accepted:
        texts[c["text"].lower()].append(c["line"]["file"])
    dup_text = {t: files for t, files in texts.items() if len(files) > 1}
    if dup_text:
        warn("duplicate_text", f"{sum(len(v) - 1 for v in dup_text.values())} clip(s) repeat a transcript already used "
             "(kept: the audio differs).", sum(len(v) - 1 for v in dup_text.values()), list(dup_text))
    leftover = [c for c in accepted if LEFTOVER_MARKUP.search(c["text"])]
    if leftover:
        warn("leftover_markup", f"{len(leftover)} transcript(s) still contain brackets or markup characters; "
             "check them, they would be read as text.", len(leftover), [c["text"][:60] for c in leftover])
    for code, label in (("duplicate_audio", "identical audio to an earlier clip"),
                        ("missing_transcript", "no transcript"), ("empty_transcript", "only tags, no words")):
        hit = [r for r in rejected if r["reason"] == code]
        if hit:
            warn(code, f"{len(hit)} clip(s) rejected: {label}.", len(hit), [r["source"] for r in hit])
    rates = sorted({c["sample_rate"] for c in accepted})
    if len(rates) > 1:
        warn("mixed_source_rates", f"Source clips use several sample rates ({', '.join(map(str, rates))} Hz); all are exported at {RATE} Hz.", len(rates))
    chapters_used = {c["line"]["chapter"] for c in accepted}
    gone = {"failed": 0, "missing": 0, "no_text": 0}
    for run in corpus.runs:
        if run["chapter"] in chapters_used:
            gone["failed"] += run["failed"]
            gone["missing"] += run["missing"]
            gone["no_text"] += run.get("no_text", 0)
    if any(gone.values()):
        warn("not_indexed", f"{gone['failed']} line(s) failed to render, {gone['missing']} audio file(s) are missing and "
             f"{gone['no_text']} line(s) had no spoken text (tags only) in the chapters used; the corpus index skips them, "
             "so they are counted here only.", sum(gone.values()))
    all_speaker_lines = [l for l in corpus.lines if l["voice_id"] == voice_id]
    in_selection = {l["key"] for l in lines}

    seconds = sum(durations)
    ref_info = {k: v for k, v in reference.items()}
    report = {
        "format": FORMAT,
        "speaker": {"name": voice_name, "voice_id": voice_id, "qwen_speaker_name": speaker},
        "counts": {"source_samples": len(lines), "accepted": n, "rejected": len(rejected), "train": n, "validation": 0,
                   "left_out_by_selection": sum(1 for l in all_speaker_lines if l["key"] not in in_selection and not l["duplicate"]),
                   "older_rerender_copies": sum(1 for l in all_speaker_lines if l["duplicate"])},
        "rejected_by_reason": dict(collections.Counter(r["reason"] for r in rejected).most_common()),
        "duration": {"total_s": round(seconds, 1), "total": _hms(seconds), "average_s": round(seconds / n, 2),
                     "median_s": round(statistics.median(durations), 2), "shortest_s": round(min(durations), 2),
                     "longest_s": round(max(durations), 2)},
        "audio": {"sample_rate": RATE, "channels": 1, "format": "WAV PCM16", "copied_unchanged": copied,
                  "converted": converted, "source_sample_rates": rates},
        "loudness_matching": level_summary,
        "chapter_levels_db": chapter_levels,
        "reference": ref_info | {"path": REFERENCE_PATH},
        "validation": "None. The official Qwen3-TTS fine-tuning scripts have no validation input, so no validation file is written.",
        "text": "Spoken transcript only: every [tag] removed, no emotion or speaker labels.",
        "qwen": qwen3_colab.verified_facts(),
        "training": qwen3_colab.estimate(settings, n) | dataclasses.asdict(settings),
        "source": "Synthetic speech generated by Fish Audio (models: "
                  + ", ".join(sorted({l["model"] or "unknown" for l in lines}))
                  + "). Check Fish's terms before using it to train another model.",
        "warnings": warnings,
        "mapping": mapping,
    }
    _write_jsonl(os.path.join(out_dir, "train.jsonl"), train_rows)
    _write_jsonl(os.path.join(out_dir, "rejected_samples.jsonl"),
                 [{k: v for k, v in r.items() if k != "key"} for r in rejected])
    _write_text(os.path.join(out_dir, "dataset_report.json"), json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    zip_name = folder_name + ".zip"
    cmds = qwen3_colab.commands(settings, speaker, folder_name)
    _write_text(os.path.join(out_dir, "training_settings.json"), json.dumps({
        **dataclasses.asdict(settings), "speaker_name": speaker, "checkpoint_dir": cmds["checkpoint_dir"],
        "fixed_in_the_official_script": {"gradient_accumulation_steps": qwen3_colab.ACCUMULATION, "mixed_precision": "bf16",
                                         "attention": "flash_attention_2", "checkpoint": "once per epoch, no resume"},
        "commands": {"prepare_data": cmds["prepare"], "sft_12hz": cmds["train"]},
        "estimate": qwen3_colab.estimate(settings, n)}, ensure_ascii=False, indent=2) + "\n")
    helper = qwen3_colab.render_files(speaker=speaker, folder_name=folder_name, zip_name=zip_name,
                                      settings=settings, report=report)
    for file_name, body in helper.items():
        _write_text(os.path.join(out_dir, file_name), body)
    zipped = None
    if opts.make_zip:
        make_zip(out_dir, archive)
        zipped = {"path": archive, "size": os.path.getsize(archive)}
    logger.info("Qwen3-TTS export: %d accepted, %d rejected -> %s", n, len(rejected), out_dir)
    return {
        "name": name, "clips": n, "hours": round(seconds / 3600, 3), "formats": [FORMAT], "path": out_dir,
        "speakers": {speaker: {"clips": n, "seconds": round(seconds, 1)}}, "loudness": level_summary,
        "qwen3": {"accepted": n, "rejected": len(rejected), "rejected_by_reason": report["rejected_by_reason"],
                  "reference": ref_info, "warnings": [{"code": w["code"], "message": w["message"]} for w in warnings],
                  "zip": zipped, "files": ["train.jsonl", "dataset_report.json", "rejected_samples.jsonl", "training_settings.json", *helper],
                  "settings": dataclasses.asdict(settings), "estimate": qwen3_colab.estimate(settings, n),
                  "commands": qwen3_colab.start_commands(speaker, folder_name, zip_name, settings) | {"official": cmds}},
    }
