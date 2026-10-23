"""Fish story render jobs: organized output, durable records, request log, resume.

Each render gets its own run folder::

    <output root>/<story title>/run-<NNN>_<YYYYMMDD-HHMMSS>_<job id>/
        001_Shadow_sun_2.wav ...   one untouched Fish response per request
        story.wav                  joined file (WAV/PCM only)
        job.json                   live state: survives restarts, drives the Jobs list
        story.json                 the submitted draft (so a run can be resumed)
        settings.json              the effective options used (never the key)
        requests.jsonl             one record per API call: exact request, timing, hash
        run.log                    human-readable event log
        manifest.json              final summary

Nothing here runs until ``start_render`` / ``resume_render`` is called after an
explicit user confirmation, and at most ``max_concurrency`` requests are in flight.
"""
from __future__ import annotations

import dataclasses
import glob
import hashlib
import io
import json
import logging
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from . import client
from .config import FishConfig, load_config, resolve_api_key, setting_source
from .story import Chunk, Draft, effective_config
from .story import check_request as story_check

logger = logging.getLogger("omnivoice.fish_api.story_jobs")

JOB_FILE = "job.json"
STORY_FILE = "story.json"
SETTINGS_FILE = "settings.json"
REQUESTS_FILE = "requests.jsonl"
LOG_FILE = "run.log"
MANIFEST_FILE = "manifest.json"

JOB_ID_RE = re.compile(r"^[a-f0-9]{12}$")
_EXT = {"wav": "wav", "pcm": "pcm", "mp3": "mp3", "opus": "opus"}
_FATAL = {401, 402, 403}
_MAX_LIVE = 30
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


# ── output location ──
def default_output_root() -> str:
    from core.config import DATA_DIR

    return os.path.join(DATA_DIR, "fish_stories", "renders")


def validate_output_dir(path: str) -> str:
    """An absolute, writable folder (created if needed). Returns the normalised path."""
    raw = (path or "").strip()
    if not raw or len(raw) > 400 or "\x00" in raw:
        raise ValueError("Enter a folder path.")
    if not os.path.isabs(raw):
        raise ValueError("The output folder must be an absolute path, e.g. D:\\Stories.")
    full = os.path.normpath(raw)
    if os.path.dirname(full) == full:
        raise ValueError("Choose a folder, not a drive root.")
    try:
        os.makedirs(full, exist_ok=True)
        probe = os.path.join(full, f".write-test-{uuid.uuid4().hex[:6]}")
        with open(probe, "wb") as fh:
            fh.write(b"ok")
        os.remove(probe)
    except OSError as exc:
        raise ValueError(f"Cannot write to that folder ({type(exc).__name__}).") from exc
    return full


def output_root(cfg: Optional[FishConfig] = None) -> str:
    """Where new renders go: the configured folder, else the data-dir default."""
    cfg = cfg or load_config()
    if cfg.output_dir:
        return validate_output_dir(cfg.output_dir)
    root = default_output_root()
    os.makedirs(root, exist_ok=True)
    return root


def _scan_roots() -> list[str]:
    roots = [default_output_root()]
    try:
        configured = load_config().output_dir
    except Exception:  # noqa: BLE001
        configured = None
    if configured and os.path.isabs(configured):
        roots.insert(0, os.path.normpath(configured))
    seen, out = set(), []
    for r in roots:
        key = os.path.normcase(r)
        if key not in seen and os.path.isdir(r):
            seen.add(key)
            out.append(r)
    return out


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (title or "").lower()).strip("-")[:40] or "untitled"


def _safe(label: str, fallback: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", label).strip("_")[:24] or fallback


STORY_MARKER = ".story-id"


def _marker(folder: str) -> Optional[str]:
    try:
        with open(os.path.join(folder, STORY_MARKER), encoding="utf-8") as fh:
            return fh.read().strip() or None
    except OSError:
        return None


def _story_dir(root: str, title: str, draft_id: Optional[str]) -> str:
    """One readable folder per story (``the-long-night``), kept even if the title changes.

    A hidden marker file records which draft owns the folder, so two stories that
    share a title get ``the-long-night-2`` instead of mixing their runs.
    """
    tag = re.sub(r"[^A-Za-z0-9]", "", draft_id or "")[:6] or "adhoc"
    try:
        names = sorted(n for n in os.listdir(root) if os.path.isdir(os.path.join(root, n)))
    except OSError:
        names = []
    for name in names:  # same draft: reuse (rename-proof); also folders made before markers existed
        folder = os.path.join(root, name)
        if _marker(folder) == tag or (_marker(folder) is None and name.endswith(f"-{tag}")):
            return folder
    base = _slug(title)
    name, n = base, 1
    while os.path.exists(os.path.join(root, name)):
        n += 1
        name = f"{base}-{n}"
    return os.path.join(root, name)


def _claim_story_dir(folder: str, draft_id: Optional[str]) -> None:
    os.makedirs(folder, exist_ok=True)
    tag = re.sub(r"[^A-Za-z0-9]", "", draft_id or "")[:6] or "adhoc"
    if _marker(folder) is None:
        with open(os.path.join(folder, STORY_MARKER), "w", encoding="utf-8") as fh:
            fh.write(tag)


def _next_run_name(story_dir: str, job_id: str) -> str:
    n = 1
    try:
        for name in os.listdir(story_dir):
            m = re.match(r"^run-(\d+)_", name)
            if m:
                n = max(n, int(m.group(1)) + 1)
    except OSError:
        pass
    return f"run-{n:03d}_{time.strftime('%Y%m%d-%H%M%S')}_{job_id}"


# ── job record + logs ──
def _view(job: dict) -> dict:
    with job["lock"]:
        return {k: ([dict(c) for c in v] if k == "chunks" else (list(v) if k == "warnings" else v))
                for k, v in job.items() if k not in {"lock", "cancel", "io"}}


def _write_json(path: str, data) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    for attempt in range(5):  # Windows refuses a replace while a reader has the file open
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.02 * (attempt + 1))


def _persist(job: dict) -> None:
    data = _view(job)
    with job["io"]:
        try:
            _write_json(os.path.join(job["out_dir"], JOB_FILE), data)
        except OSError as exc:
            logger.warning("Fish story: could not save job record (%s)", type(exc).__name__)


def _log(job: dict, level: str, message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {level:<5} {message}\n"
    with job["io"]:
        try:
            with open(os.path.join(job["out_dir"], LOG_FILE), "a", encoding="utf-8") as fh:
                fh.write(line)
        except OSError:
            pass


def _record(job: dict, entry: dict) -> None:
    with job["io"]:
        try:
            with open(os.path.join(job["out_dir"], REQUESTS_FILE), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass


_EXTRA_RUNS_KEY = "tts.fish.extra_runs"


def _extra_runs() -> list[str]:
    """Run folders the user opened by path (outside the output folder), newest first."""
    from services import settings_store

    try:
        folders = json.loads(settings_store.get_text(_EXTRA_RUNS_KEY) or "[]")
    except ValueError:
        return []
    return [f for f in folders if isinstance(f, str) and os.path.isfile(os.path.join(f, JOB_FILE))]


def open_run_folder(path: str) -> dict:
    """Remember a finished run folder so it can be loaded, played and resumed. Reads only."""
    from services import settings_store

    if not isinstance(path, str) or not os.path.isabs(path):
        raise ValueError("Give the full path of a run folder.")
    folder = os.path.normpath(path)
    data = _read_json(os.path.join(folder, JOB_FILE))
    if not data or not JOB_ID_RE.match(str(data.get("id") or "")):
        raise ValueError("That folder has no job.json from a Fish story render.")
    if not os.path.isfile(os.path.join(folder, STORY_FILE)):
        raise ValueError("That run has no story.json, so its lines cannot be loaded.")
    known = [f for f in _extra_runs() if os.path.normcase(f) != os.path.normcase(folder)]
    settings_store.set_text(_EXTRA_RUNS_KEY, json.dumps([folder, *known][:50]))
    return _summary({**data, "out_dir": folder})


def _find_run_dir(job_id: str) -> Optional[str]:
    if not JOB_ID_RE.match(job_id or ""):
        return None
    for root in _scan_roots():
        for hit in glob.glob(os.path.join(glob.escape(root), "*", f"*_{job_id}", JOB_FILE)):
            return os.path.dirname(hit)
    for folder in _extra_runs():
        data = _read_json(os.path.join(folder, JOB_FILE))
        if data and data.get("id") == job_id:
            return folder
    return None


def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _audio_info(raw: bytes, fmt: str, cfg: FishConfig) -> Optional[dict]:
    """Duration / rate of a response, so the log can show what Fish really returned."""
    try:
        if fmt == "pcm":
            return {"duration_s": round(len(raw) // 2 / cfg.sample_rate, 3),
                    "sample_rate": cfg.sample_rate, "channels": 1}
        import soundfile as sf

        info = sf.info(io.BytesIO(raw))
        return {"duration_s": round(info.duration, 3), "sample_rate": info.samplerate,
                "channels": info.channels}
    except Exception:  # noqa: BLE001 — a format libsndfile cannot read is not an error
        return None


def _summary(job: dict) -> dict:
    keys = ("id", "title", "draft_id", "status", "message", "total", "done", "failed", "started_at",
            "finished_at", "out_dir", "joined", "format", "model", "sample_rate", "audio_seconds", "warnings")
    return {k: job.get(k) for k in keys}


# ── public API ──
def get_job(job_id: str) -> Optional[dict]:
    with _jobs_lock:
        live = _jobs.get(job_id)
    if live:
        return _view(live)
    run_dir = _find_run_dir(job_id)
    data = _read_json(os.path.join(run_dir, JOB_FILE)) if run_dir else None
    if data and data.get("status") == "running":
        # No live thread owns it: the app closed mid-render. Resume can finish it.
        data.update(status="interrupted", message="The app closed before this render finished.",
                    finished_at=data.get("finished_at") or time.time())
        try:
            _write_json(os.path.join(run_dir, JOB_FILE), data)
        except OSError:
            pass
    return data


def list_jobs(limit: int = 200) -> list[dict]:
    rows: dict[str, dict] = {}
    for root in _scan_roots():
        for path in glob.glob(os.path.join(glob.escape(root), "*", "*_*", JOB_FILE)):
            data = _read_json(path)
            if data and data.get("id"):
                rows[data["id"]] = data
    for folder in _extra_runs():
        data = _read_json(os.path.join(folder, JOB_FILE))
        if data and data.get("id"):
            rows.setdefault(data["id"], {**data, "out_dir": folder})  # the folder it is in now
    with _jobs_lock:
        live = {k: _view(v) for k, v in _jobs.items()}
    rows.update(live)
    out = []
    for data in rows.values():
        row = _summary(data)
        if row["status"] == "running" and data["id"] not in live:
            row.update(status="interrupted", message="The app closed before this render finished.")
        out.append(row)
    out.sort(key=lambda r: r.get("started_at") or 0, reverse=True)
    return out[:limit]


def load_run(job_id: str) -> Optional[dict]:
    """A finished run's draft (story.json) and per-line audio, to reload and test it again."""
    run_dir = _find_run_dir(job_id)
    snapshot = _read_json(os.path.join(run_dir, STORY_FILE)) if run_dir else None
    if not run_dir or not snapshot:
        return None
    draft = Draft.model_validate(snapshot["draft"])
    data = _read_json(os.path.join(run_dir, JOB_FILE)) or {}
    lines = [{"id": c.get("id"), "index": c.get("index"), "status": c.get("status"),
              "file": c.get("file") if c.get("file") and os.path.isfile(os.path.join(run_dir, c["file"])) else None,
              "voice_name": c.get("voice_name"), "duration_s": c.get("duration_s")}
             for c in data.get("chunks", [])]
    return {"draft": draft.model_dump(), "title": data.get("title"), "lines": lines,
            "joined": data.get("joined")}


def run_audio_path(job_id: str, name: str) -> Optional[str]:
    """Path of one audio file the run itself lists (a line or the joined file); nothing else is served."""
    run_dir = _find_run_dir(job_id)
    data = _read_json(os.path.join(run_dir, JOB_FILE)) if run_dir else None
    if not data:
        return None
    allowed = {c.get("file") for c in data.get("chunks", [])} | {data.get("joined")}
    if name not in allowed or name != os.path.basename(name):
        return None
    path = os.path.join(run_dir, name)
    return path if os.path.isfile(path) else None


def read_log(job_id: str, max_lines: int = 400) -> Optional[str]:
    run_dir = _find_run_dir(job_id)
    if not run_dir:
        return None
    try:
        with open(os.path.join(run_dir, LOG_FILE), encoding="utf-8", errors="replace") as fh:
            return "".join(fh.readlines()[-max_lines:])
    except OSError:
        return ""


def cancel_job(job_id: str) -> bool:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return False
    job["cancel"].set()
    _log(job, "INFO", "Cancel requested; finishing requests already in flight.")
    return True


def _new_job(run_dir: str, draft: Draft, draft_id: Optional[str], cfg: FishConfig,
             chunks: list[dict], job_id: str, started_at: float) -> dict:
    return {
        "id": job_id, "title": draft.title or "Untitled story", "draft_id": draft_id,
        "status": "running", "message": "", "out_dir": run_dir, "joined": None,
        "total": len(chunks), "done": sum(c["status"] == "done" for c in chunks),
        "failed": 0, "started_at": started_at, "finished_at": None,
        "format": cfg.format, "model": cfg.model, "sample_rate": cfg.sample_rate,
        "audio_seconds": 0.0, "warnings": [], "chunks": chunks,
        "lock": threading.Lock(), "io": threading.Lock(), "cancel": threading.Event(),
    }


def _chunk_entry(index: int, chunk: Chunk, default: Optional[str], names: dict) -> dict:
    voice = chunk.voice or default
    return {"id": chunk.id, "index": index, "status": "queued", "file": None, "error": None,
            "speaker": chunk.speaker, "voice": voice, "voice_name": names.get(voice or "", voice),
            "attempts": 0, "elapsed_ms": 0, "bytes": 0, "sha256": None, "duration_s": None,
            "sample_rate": None}


def _audio_name(index: int, speaker: Optional[str], voice_name: Optional[str], ext: str) -> str:
    return (f"{index:03d}_{_safe(speaker or 'narration', 'narration')}"
            f"_{_safe(voice_name or 'default', 'default')}.{ext}")


def _items(draft: Draft, chunk_ids: Optional[list[str]]) -> list[tuple[int, Chunk]]:
    wanted = set(chunk_ids) if chunk_ids else None
    return [(i, c) for i, c in enumerate(draft.chunks, 1)
            if c.text.strip() and (wanted is None or c.id in wanted)]


#: What a bare web-page request contains; anything else in a body is an explicit extra.
MINIMAL_FIELDS = frozenset({"text", "format", "sample_rate", "reference_id"})


@dataclasses.dataclass(frozen=True)
class _RenderRequest:
    """One Fish call and the story lines whose audio it returns."""

    items: tuple[tuple[int, Chunk], ...]
    body: dict
    mode: str
    voices: tuple[Optional[str], ...]
    speakers: tuple[str, ...]


def _voice_for(chunk: Chunk, draft: Draft, default: Optional[str]) -> Optional[str]:
    """Explicit line voice, then saved cast mapping, then narrator/default."""
    return chunk.voice or (draft.cast.get(chunk.speaker) if chunk.speaker else None) or default


def _speaker_key(chunk: Chunk, voice: Optional[str]) -> str:
    # Tagged character names are the stable speaker ids. Untagged lines are
    # narration unless they carry a deliberate per-line voice override.
    return chunk.speaker or (f"voice:{voice}" if chunk.voice and voice else "narrator")


def _request_for_group(items: list[tuple[int, Chunk]], draft: Draft, cfg: FishConfig,
                       default: Optional[str]) -> _RenderRequest:
    voices = tuple(_voice_for(chunk, draft, default) for _, chunk in items)
    speakers = tuple(_speaker_key(chunk, voice) for (_, chunk), voice in zip(items, voices))
    identities: list[tuple[str, Optional[str]]] = []
    for speaker, voice in zip(speakers, voices):
        identity = (speaker, voice)
        if identity not in identities:
            identities.append(identity)

    if len(identities) == 1:
        text = "\n".join(chunk.text.strip() for _, chunk in items)
        body = client.build_body(text, cfg, reference_id=voices[0], speed=draft.settings.speed)
        return _RenderRequest(tuple(items), body, "single", voices, speakers)

    missing = next((index for (index, _), voice in zip(items, voices) if not voice), None)
    if missing is not None:
        raise ValueError(
            f"Line {missing} needs a voice before it can be included in a multi-speaker call. "
            "Assign a character voice or a default narrator voice."
        )
    speaker_index = {identity: i for i, identity in enumerate(identities)}
    tagged = "\n".join(
        f"<|speaker:{speaker_index[(speaker, voice)]}|>{chunk.text.strip()}"
        for (_, chunk), speaker, voice in zip(items, speakers, voices)
    )
    reference_ids = [voice for _, voice in identities if voice is not None]
    body = client.build_body(tagged, cfg, reference_id=reference_ids, speed=draft.settings.speed)
    return _RenderRequest(tuple(items), body, "multi", voices, speakers)


def _render_requests(draft: Draft, cfg: FishConfig,
                     items: list[tuple[int, Chunk]], default: Optional[str]) -> list[_RenderRequest]:
    """Compile small local-stitch calls or large provider-stitched calls."""
    if draft.settings.request_mode != "multi":  # off (single or unset): one call per line
        return [_request_for_group([item], draft, cfg, default) for item in items]

    requests: list[_RenderRequest] = []
    pending: list[tuple[int, Chunk]] = []
    for item in items:
        combined = [*pending, item]
        # Reserve enough room for one speaker tag and newline per line before
        # compiling the group. This can split slightly early, but never creates
        # a body over Fish's 20,000-character text limit.
        tagged_length = sum(len(chunk.text.strip()) + 20 for _, chunk in combined)
        if pending and tagged_length > 20000:
            requests.append(_request_for_group(pending, draft, cfg, default))
            pending = []
            combined = [item]
        candidate = _request_for_group(combined, draft, cfg, default)
        if len(candidate.body["text"]) <= 20000:
            pending.append(item)
            continue
        one = _request_for_group([item], draft, cfg, default)
        if len(one.body["text"]) > 20000:
            raise ValueError(f"Line {item[0]} exceeds Fish Audio's 20,000-character request limit.")
        pending.append(item)
    if pending:
        requests.append(_request_for_group(pending, draft, cfg, default))
    return requests


def _request_id(request: _RenderRequest) -> str:
    first = request.items[0][1].id
    last = request.items[-1][1].id
    return first if len(request.items) == 1 else f"{first}..{last}"


def _request_audio_name(request: _RenderRequest, draft: Draft, ext: str) -> str:
    first, last = request.items[0][0], request.items[-1][0]
    if len(request.items) == 1:
        chunk = request.items[0][1]
        voice = request.voices[0]
        return _audio_name(first, chunk.speaker, draft.voices.get(voice or "", voice), ext)
    span = f"{first:03d}-{last:03d}"
    if request.mode == "multi":
        count = len(dict.fromkeys(zip(request.speakers, request.voices)))
        return f"{span}_multi_{count}-speakers.{ext}"
    voice = request.voices[0]
    label = request.speakers[0] if request.speakers[0] != "narrator" else "narration"
    return f"{span}_{_safe(label, 'narration')}_{_safe(draft.voices.get(voice or '', voice) or 'default', 'default')}.{ext}"


# (setting, where it appears in a request body)
_TABLE = (
    ("model", "header"), ("format", "format"), ("sample_rate", "sample_rate"), ("mp3_bitrate", "mp3_bitrate"),
    ("opus_bitrate", "opus_bitrate"), ("latency", "latency"), ("normalize", "normalize"),
    ("temperature", "temperature"), ("top_p", "top_p"), ("repetition_penalty", "repetition_penalty"),
    ("max_new_tokens", "max_new_tokens"), ("min_chunk_length", "min_chunk_length"),
    ("condition_on_previous_chunks", "condition_on_previous_chunks"),
    ("early_stop_threshold", "early_stop_threshold"), ("chunk_length", "chunk_length"),
    ("language", "language"), ("speed", "prosody.speed"), ("volume", "prosody.volume"),
    ("normalize_loudness", "prosody.normalize_loudness"), ("reference_id", "reference_id"),
)


def settings_in_effect(draft: Draft, cfg: FishConfig, body: dict) -> list[dict]:
    """Every setting with its effective value, where it came from, and whether ``body`` sends it."""
    rows = []
    for name, where in _TABLE:
        value = draft.settings.speed if name == "speed" else getattr(cfg, name, None)
        if name == "reference_id":
            value = draft.settings.default_voice or cfg.reference_id
        story_value = getattr(draft.settings, "default_voice" if name == "reference_id" else name, None)
        if name == "speed":
            story_value = None if abs(draft.settings.speed - 1.0) < 1e-6 else draft.settings.speed
        if story_value is not None:
            source = "story"
        elif name == "model" or name in ("format", "sample_rate", "latency", "normalize"):
            source = setting_source(name) or "default"
        elif name == "reference_id":
            source = setting_source("reference_id") or "none"
        elif name == "speed":
            source = "default"
        else:
            source = setting_source(name) or "fish"
        if where == "header":
            sent = True
        elif where.startswith("prosody."):
            sent = where.split(".", 1)[1] in (body.get("prosody") or {})
        else:
            sent = where in body
        note = "minimal" if (cfg.minimal_request and not sent and value not in (None, "")
                             and name not in ("speed", "volume") and source != "fish") else None
        if (name == "mp3_bitrate" and cfg.format != "mp3") or (name == "opus_bitrate" and cfg.format != "opus"):
            note, source = "na", "-"  # only meaningful for that output format
        rows.append({"name": name, "value": value, "source": source, "sent": sent, "note": note})
    return rows


def preview(draft: Draft, *, chunk_ids: Optional[list[str]] = None, limit: int = 5,
            draft_id: Optional[str] = None) -> dict:
    """Dry run: the exact requests a render would send, checked against the schema.

    Local only. Nothing is sent to Fish, no folder is created, and the API key is
    never included (the Authorization header is shown as a placeholder).
    """
    cfg = effective_config(draft.settings)
    default = draft.settings.default_voice or cfg.reference_id
    ext = _EXT.get(cfg.format, "bin")
    rows, problems = [], 0
    requests = _render_requests(draft, cfg, _items(draft, chunk_ids), default)
    for request in requests:
        index, chunk = request.items[0]
        body = request.body
        issues = story_check(body)
        notes = []
        if not any(request.voices):
            notes.append("No voice set: Fish will use its default voice.")
        problems += len(request.items) if issues else 0
        rows.append({"extras": sorted(set(body) - MINIMAL_FIELDS),
                     "index": index, "indices": [i for i, _ in request.items],
                     "id": _request_id(request), "ids": [c.id for _, c in request.items],
                     "mode": request.mode,
                     "speaker": chunk.speaker, "speakers": list(request.speakers),
                     "voice": request.voices[0] if request.mode == "single" else list(request.voices),
                     "voice_name": (draft.voices.get(request.voices[0] or "", request.voices[0])
                                    if request.mode == "single" else None),
                     "file": _request_audio_name(request, draft, ext),
                     "chars": len(body["text"]), "body": body, "issues": issues, "notes": notes})
    try:
        root = default_output_root() if not cfg.output_dir else os.path.normpath(cfg.output_dir)
        folder = os.path.join(root, os.path.basename(_story_dir(root, draft.title, draft_id)),
                              "run-NNN_<date-time>_<job id>")
    except OSError:
        folder = None
    shown = rows[: max(1, limit)]
    first = shown[0] if shown else None
    curl = None
    if first:
        cont = " \\\n  "
        curl = cont.join([
            f"curl -X POST {cfg.base_url}/v1/tts",
            '-H "Authorization: Bearer $FISH_API_KEY"',
            f'-H "model: {cfg.model}"',
            '-H "Content-Type: application/json"',
            "-d '" + json.dumps(first["body"], ensure_ascii=False).replace("'", "'\\''") + "'",
            "--output line.wav",
        ])
    line_count = sum(len(request.items) for request in requests)
    return {"settings": settings_in_effect(draft, cfg, rows[0]["body"]) if rows else [],
            "extra_fields": sorted({k for r in rows for k in r["extras"]}),
            "minimal": cfg.minimal_request, "ok": bool(rows) and problems == 0, "total": line_count,
            "total_requests": len(rows), "invalid": problems,
            "shown": len(shown), "endpoint": cfg.base_url + "/v1/tts",
            "headers": {"Authorization": "Bearer <hidden>", "model": cfg.model,
                        "Content-Type": "application/json"},
            "folder": folder, "requests": shown, "curl": curl,
            "first_invalid": next((r["index"] for r in rows if r["issues"]), None),
            "lines": line_count,
            "single_requests": sum(request.mode == "single" for request in requests),
            "multi_requests": sum(request.mode == "multi" for request in requests)}


def _register(job: dict) -> None:
    with _jobs_lock:
        for old in [j for j, v in _jobs.items() if v["status"] != "running"][: max(0, len(_jobs) - _MAX_LIVE)]:
            _jobs.pop(old, None)
        _jobs[job["id"]] = job


def start_render(draft: Draft, *, draft_id: Optional[str] = None,
                 chunk_ids: Optional[list[str]] = None) -> dict:
    """Validate and start a background render. Raises ``ValueError`` / ``RuntimeError``."""
    api_key = resolve_api_key()
    if not api_key:
        raise RuntimeError("No Fish Audio API key. Add one in Settings → Models → TTS.")
    cfg = effective_config(draft.settings)
    default = draft.settings.default_voice or cfg.reference_id
    items = _items(draft, chunk_ids)
    if not items:
        raise ValueError("Nothing to render: every chunk is empty or unselected.")
    requests = _render_requests(draft, cfg, items, default)
    for request in requests:  # the same schema check the preview shows, before anything is sent
        bad = story_check(request.body)
        if bad:
            raise ValueError(f"Line {request.items[0][0]} would be rejected: {bad[0]}")

    root = output_root(cfg)
    job_id = uuid.uuid4().hex[:12]
    story_dir = _story_dir(root, draft.title, draft_id)
    _claim_story_dir(story_dir, draft_id)
    run_dir = os.path.join(story_dir, _next_run_name(story_dir, job_id))
    os.makedirs(run_dir, exist_ok=True)
    entries = [_chunk_entry(i, c, _voice_for(c, draft, default), draft.voices) for i, c in items]
    by_id = {entry["id"]: entry for entry in entries}
    for request in requests:
        request_id = _request_id(request)
        indices = [index for index, _ in request.items]
        for _, chunk in request.items:
            by_id[chunk.id].update(request_id=request_id, request_mode=request.mode,
                                   request_indices=indices)
    job = _new_job(run_dir, draft, draft_id, cfg, entries, job_id, time.time())
    try:
        _write_json(os.path.join(run_dir, STORY_FILE), {"draft_id": draft_id, "draft": draft.model_dump()})
        _write_json(os.path.join(run_dir, SETTINGS_FILE), {
            **{k: v for k, v in dataclasses.asdict(cfg).items() if k != "base_url"},
            "speed": draft.settings.speed, "pause_ms": draft.settings.pause_ms,
            "join": draft.settings.join, "request_mode": draft.settings.request_mode,
            "default_voice": default,
        })
    except OSError as exc:
        raise ValueError(f"Cannot write to the output folder ({type(exc).__name__}).") from exc
    _register(job)
    _log(job, "INFO", f"Started: {len(items)} lines in {len(requests)} {draft.settings.request_mode} "
                      f"request(s), model {cfg.model}, {cfg.format} "
                      f"{cfg.sample_rate} Hz, {cfg.max_concurrency} at a time.")
    _persist(job)
    threading.Thread(target=_run, args=(job, requests, draft, cfg, api_key),
                     name=f"fish-story-{job_id}", daemon=True).start()
    return _view(job)


def resume_render(job_id: str) -> dict:
    """Re-run only the lines that did not finish, into the same run folder."""
    api_key = resolve_api_key()
    if not api_key:
        raise RuntimeError("No Fish Audio API key. Add one in Settings → Models → TTS.")
    with _jobs_lock:
        live = _jobs.get(job_id)
    if live and live["status"] == "running":
        raise ValueError("That render is still running.")
    run_dir = _find_run_dir(job_id)
    data = _read_json(os.path.join(run_dir, JOB_FILE)) if run_dir else None
    snapshot = _read_json(os.path.join(run_dir, STORY_FILE)) if run_dir else None
    if not data or not snapshot:
        raise ValueError("That render's record is missing, so it cannot be resumed.")
    draft = Draft.model_validate(snapshot["draft"])
    cfg = effective_config(draft.settings)
    default = draft.settings.default_voice or cfg.reference_id
    by_id = {c.id: c for c in draft.chunks}
    pending = [c for c in data["chunks"] if c["status"] != "done" and c["id"] in by_id]
    if not pending:
        raise ValueError("Nothing to resume: every line already finished.")
    for entry in pending:
        entry.update(status="queued", error=None)
    job = _new_job(run_dir, draft, snapshot.get("draft_id"), cfg, data["chunks"], job_id,
                   data.get("started_at") or time.time())
    job["warnings"] = list(data.get("warnings") or [])
    job["audio_seconds"] = float(data.get("audio_seconds") or 0.0)
    _register(job)
    items = [(c["index"], by_id[c["id"]]) for c in pending]
    requests = _render_requests(draft, cfg, items, default)
    pending_by_id = {entry["id"]: entry for entry in pending}
    for request in requests:
        request_id = _request_id(request)
        indices = [index for index, _ in request.items]
        for _, chunk in request.items:
            pending_by_id[chunk.id].update(request_id=request_id, request_mode=request.mode,
                                           request_indices=indices)
    _log(job, "INFO", f"Resumed: {len(items)} lines left in {len(requests)} request(s).")
    _persist(job)
    threading.Thread(target=_run, args=(job, requests, draft, cfg, api_key),
                     name=f"fish-story-{job_id}", daemon=True).start()
    return _view(job)


def _run(job: dict, requests: list[_RenderRequest], draft: Draft, cfg: FishConfig,
         api_key: str) -> None:
    fatal: dict = {}
    slots = {entry["id"]: n for n, entry in enumerate(job["chunks"])}
    ext = _EXT.get(cfg.format, "bin")

    def one(request: _RenderRequest) -> None:
        entries = [job["chunks"][slots[chunk.id]] for _, chunk in request.items]
        indices = [index for index, _ in request.items]
        chunks = [chunk for _, chunk in request.items]
        label = f"#{indices[0]}" if len(indices) == 1 else f"#{indices[0]}-{indices[-1]}"
        if job["cancel"].is_set() or fatal:
            with job["lock"]:
                for entry in entries:
                    entry["status"] = "skipped"
            _log(job, "INFO", f"{label} skipped.")
            return
        with job["lock"]:
            for entry in entries:
                entry.update(status="running", request_id=_request_id(request),
                             request_mode=request.mode, request_indices=indices)
        meta: dict = {}
        body = request.body
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "index": indices[0], "indices": indices,
            "id": _request_id(request), "ids": [chunk.id for chunk in chunks],
            "mode": request.mode,
            "speaker": chunks[0].speaker if len(chunks) == 1 else None,
            "speakers": list(request.speakers),
            "voice": request.voices[0] if request.mode == "single" else list(request.voices),
            "request": body,
        }
        try:
            raw = client.synthesize(body, api_key=api_key, cfg=cfg, meta=meta)
            name = _request_audio_name(request, draft, ext)
            with open(os.path.join(job["out_dir"], name), "wb") as fh:
                fh.write(raw)
            audio = _audio_info(raw, cfg.format, cfg)
            digest = hashlib.sha256(raw).hexdigest()
            with job["lock"]:
                for entry in entries:
                    entry.update(status="done", file=name, bytes=len(raw), sha256=digest, error=None,
                                 attempts=meta.get("attempts", 1), elapsed_ms=meta.get("elapsed_ms", 0),
                                 duration_s=audio["duration_s"] if audio else None,
                                 sample_rate=audio["sample_rate"] if audio else None)
                job["done"] += len(entries)
                if audio:
                    job["audio_seconds"] = round(job["audio_seconds"] + audio["duration_s"], 3)
                    if cfg.format in ("wav", "pcm") and audio["sample_rate"] != cfg.sample_rate:
                        note = (f"Fish returned {audio['sample_rate']} Hz although {cfg.sample_rate} Hz "
                                "was requested.")
                        if note not in job["warnings"]:
                            job["warnings"].append(note)
                            _log(job, "WARN", note)
            record.update(status=meta.get("status", 200), attempts=meta.get("attempts", 1),
                          elapsed_ms=meta.get("elapsed_ms", 0), bytes=len(raw), file=name,
                          sha256=digest, audio=audio, content_type=meta.get("content_type"))
            _log(job, "INFO", f"{label} ok: {name} [{request.mode}] "
                              f"({len(raw)} bytes, {meta.get('elapsed_ms', 0)} ms, "
                              f"{meta.get('attempts', 1)} attempt(s))")
        except client.FishAPIError as exc:
            if exc.status in _FATAL:
                fatal.update(status=exc.status, message=exc.message)
            with job["lock"]:
                for entry in entries:
                    entry.update(status="failed", error=exc.message, attempts=meta.get("attempts", 0),
                                 elapsed_ms=meta.get("elapsed_ms", 0))
                job["failed"] += len(entries)
            record.update(status=exc.status, attempts=meta.get("attempts", 0),
                          elapsed_ms=meta.get("elapsed_ms", 0), error=exc.message)
            _log(job, "ERROR", f"{label} failed (HTTP {exc.status}): {exc.message}")
        except Exception as exc:  # noqa: BLE001 — one request must not kill the batch
            with job["lock"]:
                for entry in entries:
                    entry.update(status="failed", error=type(exc).__name__)
                job["failed"] += len(entries)
            record.update(status=0, error=type(exc).__name__)
            _log(job, "ERROR", f"{label} failed: {type(exc).__name__}")
        _record(job, record)
        _persist(job)

    try:
        with ThreadPoolExecutor(max_workers=cfg.max_concurrency) as pool:
            for request in requests:
                pool.submit(one, request)
        joined = None
        if draft.settings.join and job["done"] and not job["cancel"].is_set():
            joined = _join(job, draft, cfg)
            _log(job, "INFO" if joined else "WARN", f"Joined file: {joined}" if joined else "Join skipped.")
        if job["cancel"].is_set():
            status, message = "cancelled", "Cancelled; finished lines were kept."
        elif fatal:
            status, message = "stopped", fatal["message"]
        else:
            status, message = ("done" if not job["failed"] else "partial"), ""
        with job["lock"]:
            job["joined"] = joined
            job["finished_at"] = time.time()
        _manifest(job, draft, cfg, status, message)
        _log(job, "INFO", f"Finished: {status} ({job['done']} done, {job['failed']} failed, "
                          f"{job['audio_seconds']:.1f}s of audio).")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Fish story render crashed")
        status, message = "error", type(exc).__name__
        with job["lock"]:
            job["finished_at"] = time.time()
        _log(job, "ERROR", f"Render crashed: {type(exc).__name__}")
    # Publish the final state last, after every file is on disk, so anything that
    # sees "finished" (the UI, a resume) also sees complete records.
    with job["lock"]:
        job["status"], job["message"] = status, message
    _persist(job)


def _resample(data, src: int, dst: int):
    """(n, channels) float32 from ``src`` to ``dst`` Hz. Only used for the convenience join."""
    import numpy as np

    try:
        import torch
        import torchaudio

        out = torchaudio.functional.resample(torch.from_numpy(np.ascontiguousarray(data.T)), src, dst)
        return out.numpy().T.astype("float32")
    except Exception:  # noqa: BLE001 — torch not available here: plain linear interpolation
        n = max(1, int(round(len(data) * dst / src)))
        x = np.linspace(0, len(data) - 1, n)
        return np.stack([np.interp(x, np.arange(len(data)), data[:, c]) for c in range(data.shape[1])], axis=1
                        ).astype("float32")


def _join(job: dict, draft: Draft, cfg: FishConfig) -> Optional[str]:
    """One WAV from the finished lines, in story order (wav/pcm formats only).

    Lines that came back at a different rate are resampled to the most common one
    (the per-line files are never touched). The join is a convenience; the masters
    are the individual files.
    """
    if cfg.format not in ("wav", "pcm"):
        return None
    try:
        import numpy as np
        import soundfile as sf
        from collections import Counter

        clips = []
        seen_files: set[str] = set()
        for entry in sorted(job["chunks"], key=lambda c: c["index"]):
            if entry["status"] != "done":
                continue
            if not entry.get("file") or entry["file"] in seen_files:
                continue
            seen_files.add(entry["file"])
            path = os.path.join(job["out_dir"], entry["file"])
            if cfg.format == "pcm":
                with open(path, "rb") as fh:
                    raw = fh.read()
                data = np.frombuffer(raw[: len(raw) // 2 * 2], dtype="<i2").astype("float32") / 32768.0
                clips.append((data.reshape(-1, 1), cfg.sample_rate))
            else:
                data, sr = sf.read(path, dtype="float32", always_2d=True)
                clips.append((data, sr))
        if not clips:
            return None
        out = os.path.join(job["out_dir"], "story.wav")
        if len(clips) == 1 and cfg.format == "wav":
            # One large Fish request is already the stitched story. Keep its
            # container bytes exact in the convenience output too.
            import shutil

            shutil.copyfile(os.path.join(job["out_dir"], next(iter(seen_files))), out)
            return os.path.basename(out)
        rate = Counter(sr for _, sr in clips).most_common(1)[0][0]
        channels = max(d.shape[1] for d, _ in clips)
        mismatched = sum(1 for _, sr in clips if sr != rate)
        if mismatched:
            _log(job, "WARN", f"Joined file: {mismatched} line(s) resampled to {rate} Hz to match the rest.")
        gap = np.zeros((int(rate * draft.settings.pause_ms / 1000), channels), dtype="float32")
        parts = []
        for data, sr in clips:
            if sr != rate:
                data = _resample(data, sr, rate)
            if data.shape[1] < channels:
                data = np.repeat(data, channels, axis=1)
            if parts and len(gap):
                parts.append(gap)
            parts.append(data)
        sf.write(out, np.concatenate(parts), rate, subtype="PCM_24")
        return os.path.basename(out)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Fish story: join failed (%s)", type(exc).__name__)
        return None


def _manifest(job: dict, draft: Draft, cfg: FishConfig, status: str, message: str) -> None:
    """What was rendered, with which options, and where. No key, no secrets."""
    text = {c.id: c.text for c in draft.chunks}
    view = _view(job)
    info = {
        **_summary(view),
        "status": status, "message": message,
        "settings": {k: v for k, v in dataclasses.asdict(cfg).items() if k != "base_url"},
        "speed": draft.settings.speed, "request_mode": draft.settings.request_mode,
        "pause_ms": draft.settings.pause_ms,
        "chunks": [{**c, "text": text.get(c["id"], "")} for c in sorted(view["chunks"], key=lambda c: c["index"])],
    }
    try:
        _write_json(os.path.join(job["out_dir"], MANIFEST_FILE), info)
    except OSError:
        pass


__all__ = ["start_render", "resume_render", "get_job", "list_jobs", "read_log", "cancel_job",
           "output_root", "validate_output_dir", "default_output_root"]
