"""Fish Audio story board: staged drafts and the pre-submit plan.

A *draft* is local JSON (text chunks, a Fish voice per chunk, output options).
Saving, editing and planning a draft never contact Fish; rendering lives in
``story_jobs`` and only runs on an explicit, confirmed request.
"""
from __future__ import annotations

import dataclasses
import json
import os
import re
import time
import uuid
from typing import Literal, Optional

from pydantic import BaseModel, Field

from .config import FishConfig, load_config

ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
VOICE_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


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


class SavedSettings(StorySettings):
    """Defaults the user saves to Settings (every field optional; ``null`` clears it)."""

    output_dir: Optional[str] = Field(None, max_length=400)


class Chunk(BaseModel):
    id: str = Field(..., pattern=r"^[A-Za-z0-9_-]{1,40}$")
    text: str = Field(..., max_length=20000)
    voice: Optional[str] = Field(None, pattern=r"^[A-Za-z0-9_-]{1,128}$")
    speaker: Optional[str] = Field(None, max_length=64)  # [Name] tag this line came from


class Draft(BaseModel):
    title: str = Field("Untitled story", max_length=120)
    chunks: list[Chunk] = Field(default_factory=list, max_length=3000)
    voices: dict[str, str] = Field(default_factory=dict, max_length=300)  # voice id -> display name
    cast: dict[str, str] = Field(default_factory=dict, max_length=300)  # speaker name -> voice id
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


__all__ = ["Chunk", "Draft", "StorySettings", "SavedSettings", "effective_config", "plan",
           "save_draft", "load_draft", "delete_draft", "list_drafts", "new_draft_id"]
