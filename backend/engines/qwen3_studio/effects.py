"""Voice effects for Qwen story mode: ffmpeg filter graphs applied to a finished chunk (an inner monologue, a ghost, a demon, a telepathy voice).

An effect is a name plus an ffmpeg graph:

* ``chain``  a plain ``-af`` filter chain (mono or stereo in, whatever the chain makes out);
* ``graph``  a ``-filter_complex`` graph that reads the chunk as ``[0:a]`` and ends in a stream labelled ``[out]``.

``{rate}`` in a graph is replaced by the chunk's sample rate (for pitch shifts without the rubberband filter). Built-in effects ship with
the app; your own live in ``effects.json`` in the studio data folder and are managed from the story page (add, edit, test, delete), so a
new effect never needs a code change. In the text, ``{Internal}`` (an effect's id, label or alias in braces) marks a line: the tag is
stripped before the text reaches the model and the effect is applied to the finished audio. The effect-free take is kept in ``dry/``, and a
failed effect leaves the dry take as the chunk (never a missing line).
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)

NONE = "none"
TAIL = "apad=pad_dur=0.6"  # room for an echo tail; the chunk's own pad follows it
CODECS = ("pcm_s16le", "pcm_s24le", "pcm_f32le")
ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,23}$")
# Filters that read files or open sockets: an effect is a sound recipe, not a way to reach the disk or the network.
FORBIDDEN = re.compile(r"\b(movie|amovie|sendcmd|asendcmd|zmq|azmq|ladspa|lv2|frei0r|sofalizer|subtitles|ass|drawtext|dnn_processing|lua)\s*=?", re.I)
_TAG = re.compile(r"\{\s*([^{}\n]{1,40}?)\s*\}")
_WRAPPED_TAG = re.compile(r"\[\s*(\{[^{}\n]{1,40}\})\s*\]")  # [{Internal}] reads the same as {Internal}


def _pitch(factor: float) -> str:
    """Pitch shift by `factor` without changing the duration (resample the clock, then restore the tempo)."""
    return f"asetrate={{rate}}*{factor},aresample={{rate}},atempo={1 / factor:.4f}"


def _builtin() -> dict[str, dict]:
    def chain(label, graph, aliases=()):
        return {"label": label, "kind": "chain", "graph": graph, "aliases": list(aliases), "codec": "pcm_s16le"}

    return {
        # a thought: a stereo spread of the same voice (a hair sharp left, a hair flat right, each slightly late), a faint echo, and the
        # dry centre; the owner's own recipe, mono in, stereo out.
        "inner_monologue": {
            "label": "Inner monologue", "kind": "graph", "codec": "pcm_s24le", "aliases": ["internal", "monologue", "thought", "telepathy"],
            "graph": ("[0:a]aformat=sample_fmts=fltp:channel_layouts=mono,asplit=4[core][left][right][echo];"
                      "[core]pan=stereo|c0=c0|c1=c0[dry];"
                      "[left]rubberband=pitch=1.003472,adelay=14,volume=0.125893,pan=stereo|c0=c0|c1=0*c0[L];"
                      "[right]rubberband=pitch=0.996540,adelay=20,volume=0.125893,pan=stereo|c0=0*c0|c1=c0[R];"
                      "[echo]adelay=60,volume=0.14,pan=stereo|c0=c0|c1=c0[E];"
                      "[dry][L][R][E]amix=inputs=4:normalize=0,volume=0.8[out]")},
        "slight_echo": chain("Slight echo", f"{TAIL},aecho=0.8:0.7:60:0.3", ["echo"]),
        "ethereal": chain("Ethereal", (f"highpass=f=150,{TAIL},aecho=0.8:0.88:90|180|300:0.4|0.3|0.2,"
                                       "chorus=0.6:0.9:50|60:0.4|0.32:0.25|0.4:2|1.3,lowpass=f=9000,volume=-2dB")),
        "ghost": chain("Ghost", f"{_pitch(1.10)},highpass=f=250,{TAIL},aecho=0.8:0.9:120|240:0.5|0.4,volume=-3dB"),
        "demon": chain("Demon", (f"{_pitch(0.80)},equalizer=f=110:t=q:w=1:g=6,tremolo=f=28:d=0.25,lowpass=f=5500,{TAIL},"
                                 "aecho=0.8:0.9:35|70:0.5|0.3,acompressor=threshold=0.1:ratio=3")),
        "cave": chain("Cave", f"{TAIL},aecho=0.8:0.9:200|400|650:0.5|0.35|0.25"),
        "radio": chain("Radio", "highpass=f=300,lowpass=f=3400,acompressor=threshold=0.1:ratio=4,volume=-1dB"),
        "muffled": chain("Muffled", "lowpass=f=1200,volume=-5dB"),
    }


BUILTIN = _builtin()
PRESET_IDS = (NONE, *BUILTIN)  # the built-in ids, for callers that only need the names


# ── your own effects ─────────────────────────────────────────────────────
def _store_path() -> str:
    from . import registry

    return os.path.join(registry.studio_dir(), "effects.json")


def _read_user() -> dict[str, dict]:
    try:
        with open(_store_path(), encoding="utf-8") as fh:
            rows = json.load(fh).get("effects", [])
    except (OSError, ValueError):
        return {}
    return {r["id"]: r for r in rows if isinstance(r, dict) and ID_RE.match(str(r.get("id", "")))}


def _write_user(rows: dict[str, dict]) -> None:
    path = _store_path()
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"effects": list(rows.values())}, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def all_effects() -> dict[str, dict]:
    """id -> definition for the built-in effects and your own (a saved effect with a built-in's id cannot exist)."""
    out = {k: {**v, "id": k, "builtin": True} for k, v in BUILTIN.items()}
    for k, v in _read_user().items():
        if k not in out:
            out[k] = {"id": k, "builtin": False, "label": v.get("label") or k, "kind": v.get("kind", "chain"), "graph": v.get("graph", ""),
                      "aliases": list(v.get("aliases") or []), "codec": v.get("codec", "pcm_s16le")}
    return out


def listing() -> list[dict]:
    """What the page shows: the picker entries (none first) with their definitions."""
    return [{"id": NONE, "label": "None", "builtin": True, "kind": "chain", "graph": "", "aliases": [], "codec": "pcm_s16le"},
            *all_effects().values()]


def valid(effect: Optional[str]) -> bool:
    return effect in (None, "", NONE) or effect in all_effects()


def validate_definition(body: dict, existing_id: Optional[str] = None) -> dict:
    """The cleaned definition, or ValueError with a plain reason. Does not run ffmpeg (see :func:`test`)."""
    label = str(body.get("label") or "").strip()
    if not 1 <= len(label) <= 40:
        raise ValueError("Give the effect a name of 1 to 40 characters.")
    eid = existing_id or str(body.get("id") or re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")[:24] or "")
    if not ID_RE.match(eid):
        eid = "fx_" + eid if eid and eid[0].isdigit() else eid
    if not ID_RE.match(eid or ""):
        raise ValueError("The effect id must start with a letter and use only a-z, 0-9 and _ (24 characters at most).")
    if eid == NONE or eid in BUILTIN:
        raise ValueError(f"{eid!r} is a built-in effect; choose another name.")
    kind = body.get("kind", "chain")
    if kind not in ("chain", "graph"):
        raise ValueError("The effect type must be 'chain' (an -af chain) or 'graph' (a -filter_complex graph).")
    graph = str(body.get("graph") or "").strip()
    if not graph or len(graph) > 4000 or "\n" in graph or "\x00" in graph:
        raise ValueError("The filter graph must be 1 to 4000 characters on one line.")
    if FORBIDDEN.search(graph):
        raise ValueError("That filter reads files or the network; an effect may only change the sound.")
    if kind == "graph" and "[0:a]" not in graph or kind == "graph" and "[out]" not in graph:
        raise ValueError("A graph must read the chunk as [0:a] and end in a stream labelled [out].")
    codec = body.get("codec", "pcm_s16le")
    if codec not in CODECS:
        raise ValueError(f"The sample format must be one of: {', '.join(CODECS)}.")
    aliases = []
    for a in body.get("aliases") or []:
        a = re.sub(r"\s+", " ", str(a).strip().lower())
        if a and len(a) <= 24 and a not in aliases:
            aliases.append(a)
    if len(aliases) > 8:
        raise ValueError("At most 8 aliases.")
    taken = {x for k, v in all_effects().items() if k != eid for x in (k, v["label"].lower(), *v.get("aliases", []))}
    clash = sorted(set(aliases) & taken)
    if clash:
        raise ValueError(f"The name {clash[0]!r} is already used by another effect.")
    return {"id": eid, "label": label, "kind": kind, "graph": graph, "codec": codec, "aliases": aliases}


def save(body: dict, existing_id: Optional[str] = None) -> dict:
    rows = _read_user()
    if existing_id is not None and existing_id not in rows:
        raise KeyError(existing_id)
    clean = validate_definition(body, existing_id)
    if existing_id is None and clean["id"] in rows:
        raise ValueError(f"An effect with the id {clean['id']!r} already exists.")
    rows[clean["id"]] = clean
    _write_user(rows)
    return {**clean, "builtin": False}


def delete(effect_id: str) -> bool:
    rows = _read_user()
    if effect_id not in rows:
        return False
    del rows[effect_id]
    _write_user(rows)
    return True


# ── tags in the text ─────────────────────────────────────────────────────
def _lookup() -> dict[str, str]:
    names: dict[str, str] = {}
    for eid, d in all_effects().items():
        for n in (eid, d["label"], *d.get("aliases", [])):
            names.setdefault(re.sub(r"[\s_-]+", " ", n.strip().lower()), eid)
    return names


_TAG_WITH_SPACE = re.compile(r"([ \t]?)\{\s*([^{}\n]{1,40}?)\s*\}([ \t]?)")


def extract_tags(text: str) -> tuple[str, Optional[str], list[str]]:
    """``"{Internal} I can't."`` -> ``("I can't.", "inner_monologue", [])``. A recognised effect tag is removed (the model never hears it) and
    the last one wins; nothing else in the text is touched, so a Fish ``[tag]`` or any spacing stays exactly as written. An unknown
    ``{word}`` stays in the text and is returned so the page can warn. The tag may sit anywhere and may be written ``[{Internal}]``."""
    names = _lookup()
    found: Optional[str] = None
    unknown: list[str] = []

    def take(m: "re.Match[str]") -> str:
        nonlocal found
        eid = names.get(re.sub(r"[\s_-]+", " ", m.group(2).strip().lower()))
        if eid:
            found = eid
            return " " if m.group(1) and m.group(3) else ""  # a tag between two spaces leaves one; at an edge it leaves nothing
        unknown.append(m.group(2))
        return m.group(0)

    text = _WRAPPED_TAG.sub(lambda m: m.group(1), text or "")
    out = _TAG_WITH_SPACE.sub(take, text)
    return (out.strip() if found else out), found, unknown


def split_effect_spans(text: str, initial: Optional[str] = None) -> tuple[list[tuple[Optional[str], str]], list[str]]:
    """Cut one line at its effect tags: ``"He stopped. {Internal} Run. Now."`` -> ``[(None, "He stopped."), ("inner_monologue", "Run. Now.")]``.
    A tag switches the effect for the rest of the line (later tags switch again); text before the first tag has no effect of its own.
    Unknown ``{word}`` tags stay in the text and are returned. A piece with no letters or digits (an opening quote, a dash) joins the
    text that follows it instead of becoming a line of its own."""
    names = _lookup()
    text = _WRAPPED_TAG.sub(lambda m: m.group(1), text or "")
    spans: list[tuple[Optional[str], str]] = []
    unknown: list[str] = []
    current: Optional[str] = initial  # the effect already switched on earlier in the same line
    last = 0
    buf = ""
    for m in _TAG.finditer(text):
        eid = names.get(re.sub(r"[\s_-]+", " ", m.group(1).strip().lower()))
        if not eid:
            unknown.append(m.group(1))
            continue
        buf += text[last:m.start()]
        spans.append((current, buf))
        buf, current, last = "", eid, m.end()
    spans.append((current, buf + text[last:]))
    out: list[tuple[Optional[str], str]] = []
    carry = ""
    for fx, piece in spans:
        piece = (carry + re.sub(r"[ 	]{2,}", " ", piece).lstrip()) if carry else re.sub(r"[ 	]{2,}", " ", piece)
        carry = ""
        if not re.search(r"\w", piece):  # only punctuation / quotes: it belongs to what comes next
            carry = piece.strip()
            continue
        out.append((fx, piece.strip()))
    if carry and out:
        out[-1] = (out[-1][0], out[-1][1] + carry)
    return out, unknown


def last_tag_effect(text: str) -> Optional[str]:
    """The effect named by the last recognised ``{tag}`` in `text`, or None."""
    names = _lookup()
    found = None
    for m in _TAG.finditer(_WRAPPED_TAG.sub(lambda m: m.group(1), text or "")):
        found = names.get(re.sub(r"[\s_-]+", " ", m.group(1).strip().lower())) or found
    return found


def resolve(chunk_effect: Optional[str], profile_effect: Optional[str]) -> str:
    """The effect of one chunk: its own choice, else its voice profile's, else none."""
    for value in (chunk_effect, profile_effect):
        if value and value != NONE:
            return value
    return NONE


# ── running ffmpeg ───────────────────────────────────────────────────────
def command(d: dict, src: str, dst: str, rate: int, ffmpeg: str) -> list[str]:
    graph = d["graph"].replace("{rate}", str(int(rate)))
    base = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", src]
    mid = ["-filter_complex", graph, "-map", "[out]"] if d.get("kind") == "graph" else ["-af", graph]
    return base + mid + ["-ar", str(int(rate)), "-c:a", d.get("codec") or "pcm_s16le", dst]


def _run(d: dict, src: str, dst: str, rate: int, ffmpeg: Optional[str], timeout: float) -> None:
    if ffmpeg is None:
        from services import ffmpeg_utils

        ffmpeg = ffmpeg_utils.find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg was not found (Settings -> Media tools), so the effect could not be applied.")
    try:
        done = subprocess.run(command(d, src, dst, rate, ffmpeg), capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ffmpeg took longer than {timeout:g} s on this effect.") from exc
    if done.returncode != 0 or not os.path.isfile(dst):
        lines = [l.strip() for l in (done.stderr or "").strip().splitlines() if l.strip()]
        # ffmpeg names the real problem first ("No such filter: ...") and ends with a generic line; show both.
        reason = "unknown error" if not lines else lines[0] if len(lines) == 1 or lines[0] == lines[-1] else f"{lines[0]} ({lines[-1]})"
        raise RuntimeError(f"ffmpeg could not apply the effect: {reason}")


def apply(src: str, dst: str, effect: str, rate: int, ffmpeg: Optional[str] = None, timeout: float = 120.0) -> None:
    """Write `src` through the effect to `dst`. Raises ValueError/RuntimeError with a plain reason."""
    d = all_effects().get(effect)
    if d is None:
        raise ValueError(f"Unknown effect {effect!r}.")
    _run(d, src, dst, rate, ffmpeg, timeout)


def test(definition: dict, ffmpeg: Optional[str] = None, rate: int = 24000) -> dict:
    """Run a definition (saved or not) on two seconds of a speech-like test tone: ``{"ok", "channels", "error"}``."""
    import numpy as np
    import soundfile as sf

    try:
        eid = definition.get("id")
        if eid in BUILTIN:
            clean = BUILTIN[eid]
        else:
            clean = validate_definition(definition, eid if eid in _read_user() else None)
    except ValueError as exc:
        return {"ok": False, "channels": None, "error": str(exc)}
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = os.path.join(tmp, "in.wav"), os.path.join(tmp, "out.wav")
        t = np.linspace(0, 2.0, int(rate * 2.0), endpoint=False)
        sf.write(src, (0.25 * np.sin(2 * np.pi * 180 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))).astype("float32"), rate)
        try:
            _run(clean, src, dst, rate, ffmpeg, 60.0)
            return {"ok": True, "channels": int(sf.info(dst).channels), "error": None}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "channels": None, "error": str(exc)}
