"""Qwen3-TTS story mode: a multi-voice book rendered chunk by chunk through the one resident sidecar.

* A *draft* holds the chunks (text, speaker, optional emotion), the character -> profile mapping, per-character
  emotion defaults, the narrator profile (used for untagged text) and render settings. Drafts are JSON files.
* ``parse_script`` splits pasted text into chunks: ``[Name]`` or ``Name:`` opens a line for that character,
  untagged text belongs to the narrator, ``[sad]`` / ``[joy:strong]`` switch the emotion preset for a span.
* A render is one job at a time (it shares the playground's GPU lock). Chunks are rendered grouped by model so
  a book with a Base clone and a custom_voice narrator loads each model once; the output keeps story order.
  The sidecar picks the speaker per request, so changing profile on the same model never reloads it.
* Output layout (Fish-style): ``<root>/<story>/run-NNN_<time>_<id>/`` with ``NNN_<speaker>_<profile>.wav`` per chunk,
  ``story.wav``, ``manifest.json`` and ``story.json``. A re-render of chunks writes into the same run folder and
  keeps the old file (``..._r2.wav``). Padding after each chunk is the narrator's own (``pad_s``).

Mapping is free: a character with no profile, or a profile on an odd model, is a warning, never a block, except
that a chunk with no profile at all cannot be rendered.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from typing import Optional

from pydantic import BaseModel, Field

from . import emotions, playground, profiles, registry
from .emotions import EmotionChoice

logger = logging.getLogger("omnivoice.qwen_studio.story")

ID_RE = re.compile(r"^[a-f0-9]{12}$")
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,60}$")
RUN_RE = re.compile(r"^run-\d{3}_\d{8}-\d{6}_[a-f0-9]{12}$")
FILE_RE = re.compile(r"^[A-Za-z0-9_.-]{1,200}\.wav$")
MARKER = ".story-id"
_PRESET_NAMES = {k.lower() for k in emotions.PRESETS}
_SPEAKER_TAG = re.compile(r"^\s*\[([^\[\]\n]{1,40})\]\s*")
_SPEAKER_COLON = re.compile(r"^\s*([A-Z][\w'.-]*(?: [A-Z][\w'.-]*){0,2}):\s+(?=\S)")


class StoryChunk(BaseModel):
    id: str = Field(..., pattern=r"^[A-Za-z0-9_-]{1,40}$")
    text: str = Field(..., max_length=20000)
    speaker: Optional[str] = Field(None, max_length=64)  # None = narrator
    emotion: Optional[EmotionChoice] = None  # from an inline tag; wins over the character's emotion


class StorySettings(BaseModel):
    language: str = Field("Auto", max_length=35)
    pad_s: float = Field(0.3, ge=0.0, le=5.0)  # the narrator's silence after each chunk
    seed: Optional[int] = Field(None, ge=0, le=2**31 - 1)
    speaker_change_pause_ms: int = Field(0, ge=0, le=5000)  # extra silence in story.wav when the speaker changes
    warmup: bool = False  # a throwaway render before the first real chunk of each voice; benefit is UNVERIFIED


class Draft(BaseModel):
    title: str = Field("Untitled story", max_length=120)
    chunks: list[StoryChunk] = Field(default_factory=list, max_length=5000)
    narrator_profile_id: Optional[str] = Field(None, max_length=16)
    cast: dict[str, str] = Field(default_factory=dict, max_length=300)  # character -> profile id
    emotions: dict[str, EmotionChoice] = Field(default_factory=dict, max_length=300)  # character -> default
    settings: StorySettings = Field(default_factory=StorySettings)


# ── parsing ──────────────────────────────────────────────────────────────
def _is_emotion_tag(name: str) -> bool:
    return name.strip().lower().split(":")[0] in _PRESET_NAMES


def parse_script(text: str, max_chars: Optional[int] = None) -> list[dict]:
    """Chunks from pasted text. A leading ``[Name]`` or ``Name:`` marks a character; a line that is only a tag
    keeps that character for the following lines until a blank line. Untagged text is narration (speaker None).
    Emotion tags in a line cut it into spans. Text is never rewritten beyond removing the tags."""
    from engines.qwen3_narrator import narrator_logic as logic

    limit = max_chars or logic.default_chunk_chars(1280)
    out: list[dict] = []
    carry: Optional[str] = None
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw.strip()
        if not line:
            carry = None
            continue
        speaker, rest = carry, line
        m = _SPEAKER_TAG.match(line)
        if m and not _is_emotion_tag(m.group(1)):
            speaker, rest = m.group(1).strip(), line[m.end():]
            if not rest.strip():
                carry = speaker
                continue
        else:
            c = _SPEAKER_COLON.match(line)
            if c:
                speaker, rest = c.group(1).strip(), line[c.end():]
        for emotion, span in emotions.split_emotion_tags(rest):
            for piece in logic.split_chunks(span, limit):
                out.append({"id": f"c{len(out) + 1:04d}", "text": piece, "speaker": speaker,
                            "emotion": emotion.model_dump() if emotion else None})
    return out


def characters(chunks: list) -> list[dict]:
    """Speakers in order of first appearance with their chunk and character counts (narration first, if any)."""
    seen: dict[Optional[str], dict] = {}
    for c in chunks:
        sp = c["speaker"] if isinstance(c, dict) else c.speaker
        text = c["text"] if isinstance(c, dict) else c.text
        slot = seen.setdefault(sp, {"name": sp, "chunks": 0, "chars": 0})
        slot["chunks"] += 1
        slot["chars"] += len(text)
    return sorted(seen.values(), key=lambda s: s["name"] is not None)


# ── drafts on disk ───────────────────────────────────────────────────────
def _drafts_dir() -> str:
    path = os.path.join(registry.studio_dir(), "stories")
    os.makedirs(path, exist_ok=True)
    return path


def _draft_path(story_id: str) -> str:
    if not ID_RE.match(story_id or ""):
        raise ValueError("Bad story id.")
    return os.path.join(_drafts_dir(), story_id + ".json")


def save_draft(story_id: Optional[str], draft: Draft) -> dict:
    story_id = story_id or uuid.uuid4().hex[:12]
    data = {"id": story_id, "updated": time.time(), **draft.model_dump()}
    path = _draft_path(story_id)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, path)
    return data


def load_draft(story_id: str) -> Optional[dict]:
    try:
        with open(_draft_path(story_id), encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def delete_draft(story_id: str) -> bool:
    try:
        os.remove(_draft_path(story_id))
        return True
    except OSError:
        return False


def list_drafts() -> list[dict]:
    out = []
    for name in os.listdir(_drafts_dir()):
        if name.endswith(".json") and ID_RE.match(name[:-5]):
            d = load_draft(name[:-5])
            if d:
                out.append({"id": d["id"], "title": d.get("title", ""), "chunks": len(d.get("chunks", [])), "updated": d.get("updated")})
    return sorted(out, key=lambda d: d.get("updated") or 0, reverse=True)


# ── planning ─────────────────────────────────────────────────────────────
def profile_for(chunk: StoryChunk, draft: Draft) -> tuple[Optional[str], list[str]]:
    """(profile id, warnings). A character with no mapping speaks with the narrator profile (warned)."""
    if chunk.speaker and draft.cast.get(chunk.speaker):
        return draft.cast[chunk.speaker], []
    if chunk.speaker:
        return draft.narrator_profile_id, [f"'{chunk.speaker}' has no profile: using the narrator profile."]
    return draft.narrator_profile_id, []


def emotion_for(chunk: StoryChunk, draft: Draft) -> Optional[EmotionChoice]:
    if chunk.emotion is not None:
        return chunk.emotion
    return draft.emotions.get(chunk.speaker) if chunk.speaker else None


def plan(draft: Draft, models: dict, chunk_ids: Optional[list[str]] = None) -> dict:
    """Everything checkable without a GPU. Returns ``{"items": [...], "warnings": [...]}`` in story order; raises
    ValueError for what makes rendering impossible (no profile, missing profile file, a model that is not found)."""
    wanted = set(chunk_ids) if chunk_ids else None
    cache: dict[str, dict] = {}
    items, warnings = [], []
    for index, chunk in enumerate(draft.chunks, 1):
        if not chunk.text.strip() or (wanted is not None and chunk.id not in wanted):
            continue
        pid, warn = profile_for(chunk, draft)
        if not pid:
            raise ValueError(f"Chunk {index} has no voice: set a narrator profile or map {chunk.speaker or 'narration'} to a profile.")
        profile = cache.get(pid) or profiles.get(pid)
        if profile is None:
            raise ValueError(f"Chunk {index}: profile {pid!r} does not exist (deleted?).")
        cache[pid] = profile
        items.append({"index": index, "chunk": chunk, "profile": profile, "warnings": warn, "emotion": emotion_for(chunk, draft)})
    if not items:
        raise ValueError("There is nothing to render.")
    for it in items:
        for w in it["warnings"]:
            if w not in warnings:
                warnings.append(w)
    return {"items": items, "warnings": warnings}


def render_order(items: list[dict]) -> list[dict]:
    """Group by model, then profile, keeping story order inside each group and first appearance between groups."""
    models: dict[str, int] = {}
    profs: dict[str, int] = {}
    for it in items:
        models.setdefault(it["profile"]["model_id"], len(models))
        profs.setdefault(it["profile"]["id"], len(profs))
    return sorted(items, key=lambda it: (models[it["profile"]["model_id"]], profs[it["profile"]["id"]], it["index"]))


# ── output layout ────────────────────────────────────────────────────────
def story_slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (title or "").lower()).strip("-")[:40] or "untitled"


def _safe(label: str, fallback: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", label or "").strip("_")[:24] or fallback


def default_story_root() -> str:
    return os.path.join(registry.studio_dir(), "story")


def story_root(settings: Optional[dict] = None) -> str:
    folder = ((settings or registry.load_settings()).get("output_dir") or "").strip()
    return os.path.join(folder, "stories") if folder else default_story_root()


def _marker(folder: str) -> Optional[str]:
    try:
        with open(os.path.join(folder, MARKER), encoding="utf-8") as fh:
            return fh.read().strip() or None
    except OSError:
        return None


def story_dir(root: str, title: str, story_id: str) -> str:
    """One folder per story, kept even if the title changes; two stories with one title get ``-2``."""
    os.makedirs(root, exist_ok=True)
    for name in sorted(os.listdir(root)):
        if _marker(os.path.join(root, name)) == story_id:
            return os.path.join(root, name)
    base = story_slug(title)
    name, n = base, 1
    while os.path.exists(os.path.join(root, name)):
        n += 1
        name = f"{base}-{n}"
    folder = os.path.join(root, name)
    os.makedirs(folder)
    with open(os.path.join(folder, MARKER), "w", encoding="utf-8") as fh:
        fh.write(story_id)
    return folder


def next_run_name(folder: str, job_id: str) -> str:
    n = 1
    for name in os.listdir(folder):
        m = re.match(r"^run-(\d+)_", name)
        if m:
            n = max(n, int(m.group(1)) + 1)
    return f"run-{n:03d}_{time.strftime('%Y%m%d-%H%M%S')}_{job_id}"


def chunk_filename(index: int, speaker: Optional[str], profile_name: str, version: int = 1) -> str:
    base = f"{index:03d}_{_safe(speaker or 'narration', 'narration')}_{_safe(profile_name, 'voice')}"
    return base + (f"_r{version}" if version > 1 else "") + ".wav"


# ── jobs ─────────────────────────────────────────────────────────────────
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _write_json(path: str, data) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _log(job: dict, level: str, message: str) -> None:
    job["log"].append({"t": time.time(), "level": level, "message": message})
    del job["log"][:-300]
    logger.log(logging.INFO if level == "INFO" else logging.WARNING, "[qwen-story %s] %s", job["id"], message)


def _persist(job: dict) -> None:
    _write_json(os.path.join(job["folder"], "manifest.json"), {k: v for k, v in job.items() if k != "cancel"})


def view(job: dict) -> dict:
    return {k: v for k, v in job.items() if k != "cancel"}


def get_job(job_id: str) -> Optional[dict]:
    with _jobs_lock:
        job = _jobs.get(job_id)
    return view(job) if job else None


def cancel_job(job_id: str) -> bool:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job and job["status"] == "running":
        job["cancel"].set()
        return True
    return False


def start_render(draft: Draft, story_id: Optional[str] = None, chunk_ids: Optional[list[str]] = None,
                 run: Optional[str] = None) -> dict:
    """Validate, then render in the background. ``run`` re-renders ``chunk_ids`` into an existing run folder."""
    settings = registry.load_settings()
    models = {m["id"]: m for m in registry.discover_models(settings)}
    planned = plan(draft, models, chunk_ids)
    for it in planned["items"]:  # raises with a plain reason before anything loads
        it["resolved"] = profiles.resolve(it["profile"], models, settings, it["emotion"])
    from engines import qwen3_narrator as engine

    engine.set_python_override(registry.effective_python(settings))
    ok, reason = engine.availability(ignore_checkpoint=True)
    if not ok:
        raise RuntimeError(reason)
    if not playground._run_lock.acquire(blocking=False):
        raise RuntimeError("A Qwen render is already in progress; wait for it or cancel it.")
    try:
        story_id = story_id or uuid.uuid4().hex[:12]
        folder_root = story_dir(story_root(settings), draft.title, story_id)
        job_id = uuid.uuid4().hex[:12]
        if run:
            if not RUN_RE.match(run) or not os.path.isdir(os.path.join(folder_root, run)):
                raise ValueError("That run folder does not exist.")
            folder = os.path.join(folder_root, run)
            job = _load_manifest(folder) or {}
            job_id = job.get("id") or job_id
        else:
            folder = os.path.join(folder_root, next_run_name(folder_root, job_id))
            os.makedirs(folder)
            job = {}
        previous = {c["id"]: c for c in job.get("chunks", [])}
        entries = []
        for index, chunk in enumerate(draft.chunks, 1):
            if not chunk.text.strip():
                continue
            old = previous.get(chunk.id) or {}
            entries.append({**{"status": "queued", "file": None, "history": [], "error": None}, **old,
                            "id": chunk.id, "index": index, "speaker": chunk.speaker, "text": chunk.text})
        job.update({
            "id": job_id, "story_id": story_id, "story": os.path.basename(folder_root), "run": os.path.basename(folder),
            "folder": folder, "title": draft.title, "status": "running", "created": job.get("created") or time.time(),
            "finished": None, "message": "", "warnings": planned["warnings"], "settings": draft.settings.model_dump(),
            "chunks": entries, "total": len(planned["items"]), "done": 0, "preparing": [], "log": job.get("log", [])[-100:], "cancel": threading.Event(),
        })
        rendering = {it["chunk"].id for it in planned["items"]}
        for e in entries:
            if e["id"] in rendering:
                e["status"] = "queued"
        _write_json(os.path.join(folder, "story.json"), draft.model_dump())
        _persist(job)
    except BaseException:
        playground._run_lock.release()
        raise
    with _jobs_lock:
        for old in [j for j, v in _jobs.items() if v["status"] != "running"][: max(0, len(_jobs) - 20)]:
            _jobs.pop(old, None)
        _jobs[job["id"]] = job
    _log(job, "INFO", f"{job['total']} chunk(s) in {len({it['resolved']['model']['id'] for it in planned['items']})} model group(s).")
    threading.Thread(target=_run, args=(job, draft, planned["items"], settings), name=f"qwen-story-{job['id']}", daemon=True).start()
    return view(job)


def _load_manifest(folder: str) -> Optional[dict]:
    try:
        with open(os.path.join(folder, "manifest.json"), encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _save_wav(path: str, tensor, rate: int) -> float:
    import numpy as np
    import soundfile as sf

    audio = tensor.squeeze(0).numpy() if hasattr(tensor, "numpy") else np.asarray(tensor)
    sf.write(path, audio, rate)
    return len(audio) / rate


def _run(job: dict, draft: Draft, items: list[dict], settings: dict) -> None:
    warmed: set[str] = set()
    entries = {e["id"]: e for e in job["chunks"]}
    s = draft.settings
    try:
        backend = playground._engine(settings)
        # Clone voices get their speaker embedding made and cached first (a visible "preparing" state). A failure is a
        # warning: the render itself does not depend on it.
        need = {it["profile"]["id"]: it["profile"]["name"] for it in items
                if profiles.needs_embedding(profiles.get(it["profile"]["id"]) or {})}
        if need:
            job["preparing"] = list(need.values())
            _log(job, "INFO", "Preparing voice(s): " + ", ".join(need.values()))
            _persist(job)
            for pid in need:
                if job["cancel"].is_set():
                    break
                done = profiles.prepare_embedding(pid, settings)
                if done.get("embedding_state") == "failed":
                    _log(job, "WARN", f"Could not prepare {need[pid]}: {done.get('embedding_error')}")
            job["preparing"] = []
            _persist(job)
        for n, it in enumerate(render_order(items), 1):
            if job["cancel"].is_set():
                job["status"], job["message"] = "cancelled", "Cancelled."
                break
            chunk, entry, res = it["chunk"], entries[it["chunk"].id], it["resolved"]
            profile = it["profile"]
            entry.update(status="running", error=None, profile_id=profile["id"], profile_name=profile["name"],
                         warnings=it["warnings"] + res["warnings"])
            for w in entry["warnings"]:
                _log(job, "WARN", f"Chunk {it['index']}: {w}")
            try:
                common = {"model_path": res["model"]["path"], "voice": res["voice"], "language": s.language, "pad_s": s.pad_s,
                          **res["options"]}
                if s.warmup and profile["id"] not in warmed:
                    _log(job, "INFO", f"Warm-up for {profile['name']}")
                    backend.generate(settings["warmup_text"], seed=1000, **common)
                    warmed.add(profile["id"])
                seed = (s.seed + it["index"]) if s.seed is not None else int(time.time() * 1000) % 2**31
                started = time.time()
                tensor = backend.generate(chunk.text, seed=seed, **common)
                took = time.time() - started
                meta = dict(getattr(backend, "last_reply_meta", {}) or {})
                version = len(entry["history"]) + 1
                name = chunk_filename(it["index"], chunk.speaker, profile["name"], version)
                seconds = _save_wav(os.path.join(job["folder"], name), tensor, backend.sample_rate)
                entry["history"].append(name)
                entry.update(status="done", file=name, seed=seed, seconds=round(seconds, 2), gen_seconds=round(took, 2),
                             complete=meta.get("complete"), chunks=meta.get("chunks"), sample_rate=backend.sample_rate)
                if meta.get("complete") is False:
                    _log(job, "WARN", f"Chunk {it['index']}: hit the length limit (the end-of-speech token was not produced).")
            except Exception as exc:  # noqa: BLE001 - one bad chunk must not stop the book
                entry.update(status="error", error=f"{type(exc).__name__}: {exc}")
                _log(job, "ERROR", f"Chunk {it['index']} failed: {entry['error']}")
            job["done"] = n
            _persist(job)
        if job["status"] == "running":
            job["status"] = "done"
        job["joined"] = stitch(job, s)
    except Exception as exc:  # noqa: BLE001
        job["status"], job["message"] = "error", f"{type(exc).__name__}: {exc}"
        _log(job, "ERROR", job["message"])
    finally:
        job["finished"] = time.time()
        try:
            _persist(job)
        finally:
            playground._run_lock.release()


def stitch(job: dict, s: StorySettings) -> Optional[str]:
    """story.wav: the latest file of each finished chunk in story order. Chunks already end with ``pad_s`` of silence, so
    nothing is added except the optional pause when the speaker changes."""
    import numpy as np
    import soundfile as sf

    parts, rate, last = [], None, object()
    for e in sorted(job["chunks"], key=lambda c: c["index"]):
        if e["status"] != "done" or not e.get("file"):
            continue
        data, sr = sf.read(os.path.join(job["folder"], e["file"]), dtype="float32")
        if rate is None:
            rate = sr
        if sr != rate:
            data = np.interp(np.linspace(0, len(data) - 1, int(len(data) * rate / sr)), np.arange(len(data)), data).astype("float32")
        if parts and s.speaker_change_pause_ms and e["speaker"] != last:
            parts.append(np.zeros(int(rate * s.speaker_change_pause_ms / 1000), dtype="float32"))
        parts.append(data)
        last = e["speaker"]
    if not parts:
        return None
    sf.write(os.path.join(job["folder"], "story.wav"), np.concatenate(parts), rate)
    return "story.wav"


# ── past runs ────────────────────────────────────────────────────────────
def list_runs(slug: str, settings: Optional[dict] = None) -> list[dict]:
    if not SLUG_RE.match(slug):
        return []
    base = os.path.join(story_root(settings), slug)
    out = []
    try:
        names = sorted(os.listdir(base), reverse=True)
    except OSError:
        return out
    for name in names:
        data = _load_manifest(os.path.join(base, name)) if RUN_RE.match(name) else None
        if data:
            out.append({"run": name, "status": data.get("status"), "created": data.get("created"), "chunks": len(data.get("chunks", [])),
                        "done": sum(1 for c in data.get("chunks", []) if c.get("status") == "done")})
    return out


def read_run(slug: str, run: str, settings: Optional[dict] = None) -> Optional[dict]:
    if not (SLUG_RE.match(slug) and RUN_RE.match(run)):
        return None
    return _load_manifest(os.path.join(story_root(settings), slug, run))


def audio_path(slug: str, run: str, name: str, settings: Optional[dict] = None) -> Optional[str]:
    if not (SLUG_RE.match(slug) and RUN_RE.match(run) and FILE_RE.match(name)):
        return None
    path = os.path.join(story_root(settings), slug, run, name)
    return path if os.path.isfile(path) else None
