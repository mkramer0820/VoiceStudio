"""The voices you use with Fish, remembered locally so they are easy to find again.

One record per Fish voice id, kept next to the story drafts. ``alias`` is the free name you give
a character ("Captain Reyes"); everything else is what Fish told us about the voice, so the id,
Fish's own title and its details stay visible beside your alias. Nothing here contacts Fish: records
are added from voice lists you already fetched, from voices you add to a story, and from renders.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from typing import Iterable, Optional

logger = logging.getLogger("omnivoice.fish.library")

FILE = "voice_library.json"
VOICE_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_lock = threading.Lock()

_FISH_FIELDS = ("fish_name", "description", "languages", "tags", "visibility", "state", "author")


def _path() -> str:
    from .story import _store_dir

    return os.path.join(_store_dir(), FILE)


def _read() -> dict[str, dict]:
    try:
        with open(_path(), encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in data.items() if isinstance(v, dict) and VOICE_RE.match(k)} if isinstance(data, dict) else {}


def _write(data: dict[str, dict]) -> None:
    path = _path()
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


def _clean(value, limit: int) -> str:
    return str(value or "").strip()[:limit]


def _record(voice_id: str) -> dict:
    return {"voice_id": voice_id, "alias": "", "notes": "", "fish_name": "", "description": "",
            "languages": [], "tags": [], "visibility": None, "state": None, "author": None,
            "fish_url": f"https://fish.audio/m/{voice_id}/", "source": "manual",
            "added_at": time.time(), "last_used_at": None, "uses": 0, "lines": 0}


def _fill_from_fish(rec: dict, voice: dict) -> None:
    """Refresh Fish's own details; never touches the alias or notes."""
    title = voice.get("title") or voice.get("fish_name")
    if title:
        rec["fish_name"] = _clean(title, 200)
    if voice.get("description") is not None:
        rec["description"] = _clean(voice.get("description"), 1000)
    for key in ("languages", "tags"):
        if isinstance(voice.get(key), list):
            rec[key] = [_clean(v, 40) for v in voice[key]][:30]
    for key in ("visibility", "state", "author"):
        if voice.get(key) is not None:
            rec[key] = _clean(voice[key], 120)


def list_all() -> list[dict]:
    rows = list(_read().values())
    rows.sort(key=lambda r: (r.get("last_used_at") or r.get("added_at") or 0), reverse=True)
    return rows


def remember(voices: Iterable[dict], source: str = "manual") -> int:
    """Add voices Fish described (``id``, ``title``, ``description``…); existing aliases are kept."""
    added = 0
    with _lock:
        data = _read()
        for voice in voices:
            vid = str(voice.get("id") or voice.get("voice_id") or "")
            if not VOICE_RE.match(vid):
                continue
            rec = data.get(vid)
            if rec is None:
                rec = data[vid] = _record(vid)
                rec["source"] = source
                added += 1
            _fill_from_fish(rec, voice)
        _write(data)
    return added


def update(voice_id: str, *, alias: Optional[str] = None, notes: Optional[str] = None,
           fish: Optional[dict] = None) -> dict:
    """Name a character, add notes, and/or store Fish's details for a voice (created if new)."""
    if not VOICE_RE.match(voice_id or ""):
        raise ValueError("Not a Fish voice id.")
    with _lock:
        data = _read()
        rec = data.get(voice_id) or _record(voice_id)
        data[voice_id] = rec
        if fish:
            _fill_from_fish(rec, fish)
        if alias is not None:
            rec["alias"] = _clean(alias, 60)
        if notes is not None:
            rec["notes"] = _clean(notes, 1000)
        _write(data)
        return dict(rec)


def forget(voice_id: str) -> bool:
    with _lock:
        data = _read()
        if voice_id not in data:
            return False
        del data[voice_id]
        _write(data)
        return True


def record_use(lines_by_voice: dict[str, int], names: Optional[dict[str, str]] = None) -> None:
    """Called when a render starts: stamp last-used and count lines. A new voice is stored under
    the name the story board showed for it, so a voice used only by id is still findable."""
    names = names or {}
    try:
        with _lock:
            data = _read()
            now = time.time()
            for vid, count in lines_by_voice.items():
                if not VOICE_RE.match(vid or ""):
                    continue
                rec = data.get(vid)
                if rec is None:
                    rec = data[vid] = _record(vid)
                    rec["source"] = "story"
                    if names.get(vid) and names[vid] != vid:
                        rec["alias"] = _clean(names[vid], 60)
                rec["last_used_at"] = now
                rec["uses"] = int(rec.get("uses") or 0) + 1
                rec["lines"] = int(rec.get("lines") or 0) + count
            _write(data)
    except Exception:  # noqa: BLE001 — remembering a voice must never stop a render
        logger.warning("Fish voice library: could not record use", exc_info=True)
