"""Fish Audio story board: staged drafts and a bounded render job.

A *draft* is local JSON (text chunks, a Fish voice per chunk, output options).
Saving, editing and planning a draft never contact Fish. Only an explicit
``start_render`` sends text to Fish Audio, with at most ``max_concurrency``
requests in flight. Every chunk is written untouched (Fish's own bytes) so the
files can be edited externally; a joined WAV is added when the format allows.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, Optional

from pydantic import BaseModel, Field

from . import client
from .config import FishConfig, load_config, resolve_api_key

logger = logging.getLogger("omnivoice.fish_api.story")

ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
VOICE_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_EXT = {"wav": "wav", "pcm": "pcm", "mp3": "mp3", "opus": "opus"}
_MAX_JOBS = 20


class StorySettings(BaseModel):
    """Output options for one render. ``None`` falls back to Settings / env."""

    model: Optional[str] = Field(None, max_length=64)
    format: Optional[Literal["wav", "pcm", "mp3", "opus"]] = None
    sample_rate: Optional[int] = Field(None, ge=8000, le=96000)
    mp3_bitrate: Optional[Literal[64, 128, 192]] = None
    latency: Optional[Literal["normal", "balanced"]] = None
    normalize: Optional[bool] = None
    temperature: Optional[float] = Field(None, ge=0.0, le=1.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    chunk_length: Optional[int] = Field(None, ge=100, le=300)
    speed: float = Field(1.0, ge=0.5, le=2.0)
    volume: Optional[float] = Field(None, ge=-30.0, le=30.0)
    normalize_loudness: Optional[bool] = None
    language: Optional[str] = Field(None, max_length=35, pattern=r"^[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})*$")
    max_concurrency: Optional[int] = Field(None, ge=1, le=8)
    default_voice: Optional[str] = Field(None, pattern=r"^[A-Za-z0-9_-]{1,128}$")
    pause_ms: int = Field(350, ge=0, le=5000)  # silence between chunks in the joined file
    join: bool = True


class Chunk(BaseModel):
    id: str = Field(..., pattern=r"^[A-Za-z0-9_-]{1,40}$")
    text: str = Field(..., max_length=20000)
    voice: Optional[str] = Field(None, pattern=r"^[A-Za-z0-9_-]{1,128}$")


class Draft(BaseModel):
    title: str = Field("Untitled story", max_length=120)
    chunks: list[Chunk] = Field(default_factory=list, max_length=3000)
    voices: dict[str, str] = Field(default_factory=dict, max_length=300)  # voice id -> display name
    settings: StorySettings = Field(default_factory=StorySettings)


def effective_config(settings: StorySettings) -> FishConfig:
    """Saved/env defaults with this story's non-null options applied on top."""
    over = {k: v for k, v in settings.model_dump().items()
            if v is not None and k in {f.name for f in dataclasses.fields(FishConfig)}}
    over.pop("speed", None)
    return dataclasses.replace(load_config(), **over)


# ── draft store (local files; no network) ──
def _store_dir() -> str:
    from core.config import DATA_DIR

    path = os.path.join(DATA_DIR, "fish_stories")
    os.makedirs(path, exist_ok=True)
    return path


def new_draft_id() -> str:
    return uuid.uuid4().hex[:12]


def _draft_path(draft_id: str) -> str:
    if not ID_RE.match(draft_id):
        raise ValueError("invalid draft id")
    return os.path.join(_store_dir(), f"{draft_id}.json")


def save_draft(draft_id: str, draft: Draft) -> dict:
    payload = draft.model_dump()
    payload["id"] = draft_id
    payload["updated_at"] = time.time()
    tmp = _draft_path(draft_id) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    os.replace(tmp, _draft_path(draft_id))
    return payload


def load_draft(draft_id: str) -> Optional[dict]:
    try:
        with open(_draft_path(draft_id), encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None


def delete_draft(draft_id: str) -> bool:
    try:
        os.remove(_draft_path(draft_id))
        return True
    except FileNotFoundError:
        return False


def list_drafts() -> list[dict]:
    out = []
    for name in os.listdir(_store_dir()):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(_store_dir(), name), encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            continue
        out.append({"id": data.get("id"), "title": data.get("title", ""),
                    "chunks": len(data.get("chunks", [])), "updated_at": data.get("updated_at", 0)})
    return sorted(out, key=lambda d: d["updated_at"], reverse=True)


# ── plan: counts only, never touches the network ──
def plan(draft: Draft) -> dict:
    cfg = effective_config(draft.settings)
    default = draft.settings.default_voice or cfg.reference_id
    per_voice: dict[str, dict] = {}
    unassigned = empty = 0
    chars = nbytes = 0
    for c in draft.chunks:
        text = c.text.strip()
        if not text:
            empty += 1
            continue
        chars += len(text)
        nbytes += len(text.encode("utf-8"))
        voice = c.voice or default
        if not c.voice:
            unassigned += 1
        key = voice or "(Fish default)"
        slot = per_voice.setdefault(key, {"chunks": 0, "chars": 0})
        slot["chunks"] += 1
        slot["chars"] += len(text)
    warnings = []
    if unassigned and not default:
        warnings.append("unassigned_no_default")
    if empty:
        warnings.append("empty_chunks")
    return {
        "chunks": len(draft.chunks) - empty, "empty": empty, "chars": chars, "bytes": nbytes,
        "unassigned": unassigned, "voices": per_voice, "warnings": warnings,
        "model": cfg.model, "format": cfg.format, "sample_rate": cfg.sample_rate,
        "max_concurrency": cfg.max_concurrency,
    }


# ── render job ──
_FATAL = {401, 402, 403}
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _safe(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", label)[:24] or "voice"


def _snapshot(job: dict) -> dict:
    with job["lock"]:
        return {k: ([dict(c) for c in v] if k == "chunks" else v) for k, v in job.items() if k not in {"lock", "cancel"}}


def get_job(job_id: str) -> Optional[dict]:
    with _jobs_lock:
        job = _jobs.get(job_id)
    return _snapshot(job) if job else None


def cancel_job(job_id: str) -> bool:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return False
    job["cancel"].set()
    return True


def start_render(draft: Draft, *, draft_id: Optional[str] = None,
                 chunk_ids: Optional[list[str]] = None) -> dict:
    """Validate and start a background render. Raises ``ValueError`` / ``RuntimeError``."""
    api_key = resolve_api_key()
    if not api_key:
        raise RuntimeError("No Fish Audio API key. Add one in Settings → Models → TTS.")
    cfg = effective_config(draft.settings)
    default = draft.settings.default_voice or cfg.reference_id
    wanted = set(chunk_ids) if chunk_ids else None
    items = [(i, c) for i, c in enumerate(draft.chunks, 1)
             if c.text.strip() and (wanted is None or c.id in wanted)]
    if not items:
        raise ValueError("Nothing to render: every chunk is empty or unselected.")

    with _jobs_lock:
        for old in [j for j, v in _jobs.items() if v["status"] != "running"][: max(0, len(_jobs) - _MAX_JOBS)]:
            _jobs.pop(old, None)
    job_id = uuid.uuid4().hex[:12]
    from core.config import DATA_DIR

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(DATA_DIR, "fish_stories", draft_id or "adhoc", "renders", stamp)
    os.makedirs(out_dir, exist_ok=True)
    job = {
        "id": job_id, "status": "running", "message": "", "out_dir": out_dir, "joined": None,
        "total": len(items), "done": 0, "failed": 0, "started_at": time.time(),
        "chunks": [{"id": c.id, "index": i, "status": "queued", "file": None, "error": None,
                    "voice": c.voice or default} for i, c in items],
        "lock": threading.Lock(), "cancel": threading.Event(),
    }
    with _jobs_lock:
        _jobs[job_id] = job
    threading.Thread(target=_run, args=(job, items, draft, cfg, api_key, default),
                     name=f"fish-story-{job_id}", daemon=True).start()
    return _snapshot(job)


def _run(job, items, draft, cfg, api_key, default) -> None:
    fatal: dict = {}
    slots = {c.id: n for n, (_, c) in enumerate(items)}
    names = draft.voices

    def one(index: int, chunk: Chunk) -> None:
        slot = slots[chunk.id]
        if job["cancel"].is_set() or fatal:
            with job["lock"]:
                job["chunks"][slot]["status"] = "skipped"
            return
        voice = chunk.voice or default
        with job["lock"]:
            job["chunks"][slot]["status"] = "running"
        try:
            body = client.build_body(chunk.text.strip(), cfg, reference_id=voice,
                                     speed=draft.settings.speed)
            raw = client.synthesize(body, api_key=api_key, cfg=cfg)
            label = _safe(names.get(voice or "", voice or "default"))
            path = os.path.join(job["out_dir"], f"{index:03d}_{label}.{_EXT.get(cfg.format, 'bin')}")
            with open(path, "wb") as fh:
                fh.write(raw)
            with job["lock"]:
                job["chunks"][slot].update(status="done", file=os.path.basename(path))
                job["done"] += 1
        except client.FishAPIError as exc:
            if exc.status in _FATAL:
                fatal.update(status=exc.status, message=exc.message)
            with job["lock"]:
                job["chunks"][slot].update(status="failed", error=exc.message)
                job["failed"] += 1
        except Exception as exc:  # noqa: BLE001 — one chunk must not kill the batch
            logger.warning("Fish story chunk %s failed: %s", chunk.id, type(exc).__name__)
            with job["lock"]:
                job["chunks"][slot].update(status="failed", error=f"{type(exc).__name__}")
                job["failed"] += 1

    try:
        with ThreadPoolExecutor(max_workers=cfg.max_concurrency) as pool:
            for index, chunk in items:
                pool.submit(one, index, chunk)
        joined = None
        if draft.settings.join and job["done"] and not job["cancel"].is_set():
            joined = _join(job, items, draft, cfg)
        _manifest(job, items, draft, cfg, joined)
        with job["lock"]:
            job["joined"] = joined
            if job["cancel"].is_set():
                job["status"], job["message"] = "cancelled", "Cancelled; finished chunks were kept."
            elif fatal:
                job["status"], job["message"] = "stopped", fatal["message"]
            else:
                job["status"] = "done" if not job["failed"] else "partial"
    except Exception as exc:  # noqa: BLE001
        logger.exception("Fish story render crashed")
        with job["lock"]:
            job["status"], job["message"] = "error", type(exc).__name__


def _join(job, items, draft, cfg) -> Optional[str]:
    """One WAV from the per-chunk files, in story order (wav/pcm formats only)."""
    if cfg.format not in ("wav", "pcm"):
        return None
    try:
        import numpy as np
        import soundfile as sf

        parts, rate = [], None
        gap = None
        for slot, (_, chunk) in enumerate(items):
            entry = job["chunks"][slot]
            if entry["status"] != "done":
                continue
            path = os.path.join(job["out_dir"], entry["file"])
            if cfg.format == "pcm":
                with open(path, "rb") as fh:
                    raw = fh.read()
                data = np.frombuffer(raw[: len(raw) // 2 * 2], dtype="<i2").astype("float32") / 32768.0
                data, sr = data.reshape(-1, 1), cfg.sample_rate
            else:
                data, sr = sf.read(path, dtype="float32", always_2d=True)
            if rate is None:
                rate = sr
                gap = np.zeros((int(sr * draft.settings.pause_ms / 1000), data.shape[1]), dtype="float32")
            if sr != rate:
                logger.warning("Fish story: chunk rate differs; join skipped")
                return None
            if parts and gap is not None and len(gap):
                parts.append(gap)
            parts.append(data)
        if not parts or rate is None:
            return None
        out = os.path.join(job["out_dir"], "story.wav")
        sf.write(out, np.concatenate(parts), rate, subtype="PCM_24")
        return os.path.basename(out)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Fish story: join failed (%s)", type(exc).__name__)
        return None


def _manifest(job, items, draft, cfg, joined) -> None:
    """What was rendered with which settings. Contains no key and no secrets."""
    try:
        info = {
            "settings": {k: v for k, v in dataclasses.asdict(cfg).items() if k != "base_url"},
            "speed": draft.settings.speed, "pause_ms": draft.settings.pause_ms, "joined": joined,
            "chunks": [{"index": i, "id": c.id, "voice": c.voice, "text": c.text,
                        "file": job["chunks"][n]["file"], "status": job["chunks"][n]["status"]}
                       for n, (i, c) in enumerate(items)],
        }
        with open(os.path.join(job["out_dir"], "manifest.json"), "w", encoding="utf-8") as fh:
            json.dump(info, fh, ensure_ascii=False, indent=2)
    except OSError:
        pass


__all__ = ["Chunk", "Draft", "StorySettings", "plan", "start_render", "get_job", "cancel_job",
           "save_draft", "load_draft", "delete_draft", "list_drafts", "new_draft_id"]
