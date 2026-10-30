"""Qwen3-TTS voice profiles: a saved voice for the book pipeline.

A profile is one JSON file under ``<data dir>/qwen_studio/profiles/`` (like the playground sessions) plus an
optional ``<id>.bin`` speaker embedding: raw little-endian float32[2048], no header, the same layout as the C
engine's x-vector files, so it can be imported and exported. Nothing here loads a model.

Kinds:
  * ``narrator``  a speaker baked into a custom_voice checkpoint (EP09, ``ss_narrator``); no instruct needed;
  * ``stock``     a speaker of a stock CustomVoice model (or a borrowed stock name, with the playground's warning);
  * ``clone``     a Base model cloning a reference clip (clip + text, or the embedding only).

Sampling overrides are optional: ``None`` means the engine's own default; this module invents no defaults.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import struct
import subprocess
import threading
import uuid
from typing import Literal, Optional

from pydantic import BaseModel, Field

from . import emotions, playground, registry

logger = logging.getLogger("omnivoice.qwen_studio.profiles")
CLIP_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".webm", ".aac", ".mp4"}
#: Embedding preparation state of profiles whose file does not exist yet: id -> "preparing" or "failed: <reason>".
_PREP: dict[str, str] = {}
_PREP_LOCK = threading.Lock()

ID_RE = re.compile(r"^[a-f0-9]{12}$")
EMBEDDING_FLOATS = 2048
EMBEDDING_BYTES = EMBEDDING_FLOATS * 4


class ProfileBody(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    kind: Literal["narrator", "clone", "stock"] = "narrator"
    model_id: str = Field(..., pattern=r"^[A-Za-z0-9_.-]{1,160}$")  # an id from the model registry (Settings -> Storage)
    speaker: Optional[str] = Field(None, max_length=64)  # narrator / stock
    ref_audio: Optional[str] = Field(None, max_length=500)  # clone
    ref_text: Optional[str] = Field(None, max_length=5000)  # clone
    xvec_only: bool = False  # clone: speaker embedding only instead of clip + text
    default_instruct: Optional[str] = Field(None, max_length=1000)
    emotion: emotions.EmotionChoice = Field(default_factory=emotions.EmotionChoice)
    temperature: Optional[float] = Field(None, ge=0.05, le=2.0)
    top_k: Optional[int] = Field(None, ge=0, le=500)
    top_p: Optional[float] = Field(None, ge=0.05, le=1.0)
    repetition_penalty: Optional[float] = Field(None, ge=0.5, le=2.0)
    notes: str = Field("", max_length=5000)


def profiles_dir() -> str:
    path = os.path.join(registry.studio_dir(), "profiles")
    os.makedirs(path, exist_ok=True)
    return path


def _json_path(profile_id: str) -> str:
    if not ID_RE.match(profile_id or ""):
        raise ValueError("Bad profile id.")
    return os.path.join(profiles_dir(), profile_id + ".json")


def embedding_path(profile_id: str) -> str:
    return _json_path(profile_id)[:-5] + ".bin"


def _write(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def validate(body: ProfileBody) -> None:
    if body.kind == "clone":
        if not (body.ref_audio or "").strip():
            raise ValueError("A clone profile needs a reference clip.")
        if not body.xvec_only and not (body.ref_text or "").strip():
            raise ValueError("A clone profile needs the reference text (or tick the speaker-embedding-only option).")
    emotions.instruct_for(body.emotion)  # raises on an unknown preset or strength


def _view(data: dict) -> dict:
    out = dict(data)
    out["has_embedding"] = os.path.isfile(embedding_path(data["id"]))
    with _PREP_LOCK:
        state = _PREP.get(data["id"], "")
    out["embedding_state"] = "ready" if out["has_embedding"] else ("preparing" if state == "preparing" else ("failed" if state else "none"))
    out["embedding_error"] = state[len("failed: "):] if state.startswith("failed: ") else ""
    return out


def create(body: ProfileBody) -> dict:
    validate(body)
    data = {"id": uuid.uuid4().hex[:12], **body.model_dump()}
    _write(_json_path(data["id"]), data)
    return _view(data)


def get(profile_id: str) -> Optional[dict]:
    try:
        with open(_json_path(profile_id), encoding="utf-8") as fh:
            return _view(json.load(fh))
    except (OSError, ValueError):
        return None


def update(profile_id: str, body: ProfileBody) -> Optional[dict]:
    old = get(profile_id)
    if old is None:
        return None
    validate(body)
    if (old.get("kind"), old.get("ref_audio")) != (body.kind, body.ref_audio):
        _remove_embedding(profile_id)  # the cached embedding belonged to the old clip
    data = {"id": profile_id, **body.model_dump()}
    _write(_json_path(profile_id), data)
    return _view(data)


def _remove_embedding(profile_id: str) -> None:
    with _PREP_LOCK:
        _PREP.pop(profile_id, None)
    try:
        os.remove(embedding_path(profile_id))
    except OSError:
        pass


def delete(profile_id: str) -> bool:
    try:
        os.remove(_json_path(profile_id))
    except OSError:
        return False
    _remove_embedding(profile_id)
    return True


def list_all() -> list[dict]:
    out = []
    for name in sorted(os.listdir(profiles_dir())):
        if name.endswith(".json") and ID_RE.match(name[:-5]):
            item = get(name[:-5])
            if item:
                out.append(item)
    return sorted(out, key=lambda p: p["name"].lower())


# ── embedding .bin ───────────────────────────────────────────────────────
def check_embedding(raw: bytes) -> None:
    """Refuse anything but exactly float32[2048] of finite numbers."""
    if len(raw) != EMBEDDING_BYTES:
        raise ValueError(f"An embedding file must be exactly {EMBEDDING_BYTES} bytes (float32 x {EMBEDDING_FLOATS}, no header); "
                         f"this one is {len(raw)} bytes.")
    if not all(math.isfinite(v) for v in struct.unpack(f"<{EMBEDDING_FLOATS}f", raw)):
        raise ValueError("The embedding contains values that are not finite numbers.")


def import_embedding(profile_id: str, raw: bytes) -> dict:
    if get(profile_id) is None:
        raise KeyError(profile_id)
    check_embedding(raw)
    tmp = embedding_path(profile_id) + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(raw)
    os.replace(tmp, embedding_path(profile_id))
    return get(profile_id)  # type: ignore[return-value]


def export_embedding(profile_id: str) -> Optional[bytes]:
    try:
        with open(embedding_path(profile_id), "rb") as fh:
            return fh.read()
    except OSError:
        return None


# ── from a playground take ───────────────────────────────────────────────
def from_take(session: str, take_id: str, name: str) -> dict:
    """Save the voice of a finished playground take as a profile."""
    data = playground.read_session(session)
    take = next((t for t in (data or {}).get("takes", []) if t["id"] == take_id), None)
    if take is None:
        raise KeyError(take_id)
    v, p = take["voice"], take.get("params") or {}
    if v.get("kind") == "clone_on_custom":
        raise ValueError("A 'clone on custom voice' take cannot be saved as a profile yet (experimental).")
    kind = "clone" if v.get("kind") == "clone" else "narrator"
    resolved = next((r["voice"] for r in take.get("results", []) if r.get("voice")), {})
    body = ProfileBody(
        name=name, kind=kind, model_id=take["model_id"], speaker=resolved.get("speaker") or v.get("speaker"),
        ref_audio=v.get("ref_audio") or resolved.get("ref_audio"), ref_text=v.get("ref_text") or resolved.get("ref_text"),
        xvec_only=bool(v.get("xvec_only")), default_instruct=v.get("instruct"),
        temperature=p.get("temperature"), top_k=p.get("top_k"), top_p=p.get("top_p"), repetition_penalty=p.get("repetition_penalty"),
        notes=f"Saved from take {take.get('number', '')} of session {session}.")
    return create(body)


# ── resolve for rendering ────────────────────────────────────────────────
def instruct_for(profile: dict, override: Optional[emotions.EmotionChoice] = None) -> tuple[str, list[str]]:
    """The instruct a render uses: a chunk's emotion tag wins, else the profile's default instruct, else its preset."""
    if override is not None:
        return emotions.instruct_for(override)
    if (profile.get("default_instruct") or "").strip():
        return profile["default_instruct"].strip(), []
    return emotions.instruct_for(emotions.EmotionChoice.model_validate(profile.get("emotion") or {}))


def resolve(profile: dict, models: dict, settings: dict, override: Optional[emotions.EmotionChoice] = None) -> dict:
    """What the sidecar needs for one chunk: model, voice dict, sampling options, warnings and the voice info for the manifest.
    Raises ValueError with a plain reason; checks files and the registry only."""
    model = models.get(profile["model_id"])
    if model is None:
        raise ValueError(f"Profile {profile['name']!r}: model {profile['model_id']!r} was not found. Add its folder in "
                         "Settings -> Storage or pick another model.")
    instruct, warnings = instruct_for(profile, override)
    if instruct and profile["kind"] == "narrator":
        warnings.append(emotions.UNVERIFIED_NARRATOR)
    voice = playground.Voice(
        kind="clone" if profile["kind"] == "clone" else "custom_voice", speaker=profile.get("speaker"), instruct=instruct or None,
        ref_audio=profile.get("ref_audio"), ref_text=profile.get("ref_text"), xvec_only=bool(profile.get("xvec_only")))
    take = playground.Take(id="profile", text="x", model_id=profile["model_id"], voice=voice)
    playground._check_pair(model, voice)
    sent, info = playground.resolve_voice(take, model, settings)
    if sent.get("kind") == "clone":
        sent["xvec_path"] = embedding_path(profile["id"])
    if info.get("warning"):
        warnings.append(info["warning"])
    options = {k: profile[k] for k in ("temperature", "top_k", "top_p", "repetition_penalty") if profile.get(k) is not None}
    return {"model": model, "voice": sent, "info": info, "options": options, "warnings": warnings}


# ── reference clips ──────────────────────────────────────────────────────
def clips_dir() -> str:
    path = os.path.join(profiles_dir(), "clips")
    os.makedirs(path, exist_ok=True)
    return path


def _decodes(path: str) -> bool:
    try:
        import soundfile as sf

        with sf.SoundFile(path) as clip:
            return clip.samplerate > 0 and len(clip.read(4096, dtype="float32")) > 0
    except Exception:  # noqa: BLE001 - not a libsndfile format: try ffmpeg
        return False


def store_clip(raw: bytes, filename: str = "") -> str:
    """Save an uploaded or recorded reference clip as a WAV in the app's data folder and return its path.
    Anything libsndfile cannot read is converted with ffmpeg; a file neither can read is refused."""
    ext = os.path.splitext(filename or "")[1].lower()
    if ext not in CLIP_EXTS:
        ext = ".wav"
    if len(raw) < 1000:
        raise ValueError("That recording is empty or too short.")
    name = uuid.uuid4().hex[:12]
    tmp = os.path.join(clips_dir(), name + ".upload" + ext)
    out = os.path.join(clips_dir(), name + ".wav")
    with open(tmp, "wb") as fh:
        fh.write(raw)
    try:
        if _decodes(tmp):
            import soundfile as sf

            data, rate = sf.read(tmp, dtype="float32", always_2d=True)
            sf.write(out, data.mean(axis=1) if data.shape[1] > 1 else data[:, 0], rate)
        else:
            from services.ffmpeg_utils import find_ffmpeg

            ffmpeg = find_ffmpeg()
            if not ffmpeg:
                raise ValueError("This audio format cannot be read here and ffmpeg was not found. Use a WAV file.")
            done = subprocess.run([ffmpeg, "-y", "-v", "error", "-i", tmp, "-ac", "1", "-ar", "24000", out],
                                  capture_output=True, timeout=120, check=False)
            if done.returncode != 0 or not os.path.isfile(out):
                raise ValueError("That audio file could not be read.")
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
    return out


# ── automatic speaker embedding ──────────────────────────────────────────
def needs_embedding(profile: dict) -> bool:
    return profile.get("kind") == "clone" and bool(profile.get("ref_audio")) and not profile.get("has_embedding")


def prepare_embedding(profile_id: str, settings: Optional[dict] = None) -> dict:
    """Make and cache the speaker embedding of a clone profile with the sidecar's own ``_xvector`` (stock Base
    model on the CPU, no TTS model loaded). Caller holds the GPU lock. Returns the updated profile view; a failure
    is recorded on the profile, never raised: voices work without it."""
    settings = settings or registry.load_settings()
    profile = get(profile_id)
    if profile is None:
        raise KeyError(profile_id)
    if not needs_embedding(profile):
        return profile
    with _PREP_LOCK:
        _PREP[profile_id] = "preparing"
    try:
        models = {m["id"]: m for m in registry.discover_models(settings)}
        base = models.get(profile["model_id"])
        if base is None or base["type"] != "base":
            base = registry.first_base(settings)
        if base is None:
            raise ValueError("No stock Base model was found to prepare the voice.")
        if not os.path.isfile(profile["ref_audio"]):
            raise ValueError("The reference clip is missing.")
        voice = {"kind": "prepare_embedding", "ref_audio": os.path.abspath(profile["ref_audio"]),
                 "base_model_path": base["path"], "xvec_path": embedding_path(profile_id)}
        playground._engine(settings).generate("prepare", voice=voice)
        raw = export_embedding(profile_id)
        if raw is None:
            raise ValueError("The embedding was not written.")
        check_embedding(raw)
        with _PREP_LOCK:
            _PREP.pop(profile_id, None)
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not prepare the voice embedding of %s: %s", profile_id, exc)
        _remove_file(embedding_path(profile_id))
        with _PREP_LOCK:
            _PREP[profile_id] = f"failed: {exc}"
    return get(profile_id) or profile


def _remove_file(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def prepare_in_background(profile_id: str) -> str:
    """Prepare now if the sidecar is idle and can start; otherwise leave it for the first render. Returns the state."""
    profile = get(profile_id)
    if profile is None or not needs_embedding(profile):
        return (profile or {}).get("embedding_state", "none")
    settings = registry.load_settings()
    from engines import qwen3_narrator as engine

    engine.set_python_override(registry.effective_python(settings))
    if not engine.availability(ignore_checkpoint=True)[0] or not playground._run_lock.acquire(blocking=False):
        return "none"
    with _PREP_LOCK:
        _PREP[profile_id] = "preparing"

    def work() -> None:
        try:
            prepare_embedding(profile_id, settings)
        finally:
            playground._run_lock.release()

    threading.Thread(target=work, name=f"qwen-embed-{profile_id}", daemon=True).start()
    return "preparing"
