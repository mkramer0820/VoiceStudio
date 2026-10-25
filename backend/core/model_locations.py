"""The user's extra model folders: one shared list, set in Settings -> Storage.

Rows are ``{id, name, path}``. The ``id`` is a slug of the name made unique once and then kept, so a
row keeps its id across restarts and renames. Removing a row never touches files. Engines read the
list (the Qwen registry does today); nothing here scans, loads a model or uses the network.
"""
from __future__ import annotations

import json
import os
import re
import threading
from typing import Optional

FILE = "model_locations.json"
IMPORTED_NAME = "Imported checkpoint folder"
_lock = threading.RLock()


def _data_dir() -> str:
    from core import config

    return config.DATA_DIR


def _path() -> str:
    return os.path.join(_data_dir(), FILE)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48] or "folder"


def _read_rows() -> Optional[list[dict]]:
    try:
        with open(_path(), encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    rows = data.get("locations") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return None
    return [{"id": str(r["id"]), "name": str(r.get("name") or r["id"]), "path": str(r["path"])}
            for r in rows if isinstance(r, dict) and r.get("id") and r.get("path")]


def _write_rows(rows: list[dict]) -> None:
    os.makedirs(_data_dir(), exist_ok=True)
    tmp = _path() + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"locations": rows}, fh, indent=2)
    os.replace(tmp, _path())


def _same(a: str, b: str) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def _migrate_qwen_root(rows: list[dict]) -> bool:
    """Move the old Qwen-only ``checkpoint_root`` into the list once, then blank it so a row the user
    later removes does not come back. Returns True if the list changed."""
    settings_path = os.path.join(_data_dir(), "qwen_studio", "settings.json")
    try:
        with open(settings_path, encoding="utf-8") as fh:
            saved = json.load(fh)
    except (OSError, ValueError):
        return False
    root = str(saved.get("checkpoint_root") or "").strip() if isinstance(saved, dict) else ""
    if not root:
        return False
    added = not any(_same(r["path"], root) for r in rows)
    if added:
        rows.append({"id": _unique_id(rows, IMPORTED_NAME), "name": IMPORTED_NAME, "path": os.path.abspath(root)})
    saved["checkpoint_root"] = ""
    try:
        tmp = settings_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(saved, fh, indent=2)
        os.replace(tmp, settings_path)
    except OSError:
        pass
    return added


def _unique_id(rows: list[dict], name: str) -> str:
    base, taken, n = slug(name), {r["id"] for r in rows}, 1
    candidate = base
    while candidate in taken:
        n += 1
        candidate = f"{base}-{n}"
    return candidate


def load() -> list[dict]:
    with _lock:
        stored = _read_rows()
        rows = stored or []
        if _migrate_qwen_root(rows) or (stored is None and rows):
            _write_rows(rows)
        return rows


def add(name: str, path: str) -> dict:
    """Add a folder, or return the existing row if that folder is already listed."""
    with _lock:
        rows = load()
        for row in rows:
            if _same(row["path"], path):
                return row
        row = {"id": _unique_id(rows, name), "name": name.strip() or os.path.basename(path.rstrip("\\/")) or "Folder",
               "path": os.path.abspath(path)}
        rows.append(row)
        _write_rows(rows)
        return row


def replace(row_id: str, name: Optional[str] = None, path: Optional[str] = None) -> Optional[dict]:
    with _lock:
        rows = load()
        row = next((r for r in rows if r["id"] == row_id), None)
        if row is None:
            return None
        if name is not None and name.strip():
            row["name"] = name.strip()
        if path:
            row["path"] = os.path.abspath(path)
        _write_rows(rows)
        return row


def remove(row_id: str) -> bool:
    """Drop a row from the list. Files on disk are never touched."""
    with _lock:
        rows = load()
        kept = [r for r in rows if r["id"] != row_id]
        if len(kept) == len(rows):
            return False
        _write_rows(kept)
        return True
