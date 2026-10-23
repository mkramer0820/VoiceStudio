"""Per-clip audio measurements for the training-data browser (peak, clipping, level, silences).

Measured from the WAVs themselves, once, then cached by the clip's sha256 so a rescan or a second
export never re-reads audio. Local files only; the measurements are read-only facts about a clip.
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from typing import Optional

logger = logging.getLogger("omnivoice.fish.dataset_quality")

FILE = "clip_stats.json"
FRAME_MS = 10
PAUSE_MS = 300  # a silence this long inside a clip counts as a pause (choppiness)
_lock = threading.Lock()
_stats: dict[str, dict] = {}
_loaded = False
_job: dict = {"running": False, "done": 0, "total": 0, "failed": 0, "started_at": None, "finished_at": None}


def _path() -> str:
    from .story import _store_dir

    return os.path.join(_store_dir(), FILE)


def _load() -> None:
    global _loaded
    if _loaded:
        return
    try:
        with open(_path(), encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            _stats.update({k: v for k, v in data.items() if isinstance(v, dict)})
    except (OSError, ValueError):
        pass
    _loaded = True


def _save() -> None:
    path = _path()
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(_stats, fh)
        os.replace(tmp, path)
    except OSError:
        logger.warning("Fish dataset: could not save clip statistics", exc_info=True)


def reset_for_tests() -> None:
    global _loaded
    with _lock:
        _stats.clear()
        _loaded = False
        _job.update(running=False, done=0, total=0, failed=0, started_at=None, finished_at=None)


def measure(path: str) -> dict:
    """Peak, level, clipping and silence measurements of one WAV."""
    import numpy as np
    import soundfile as sf

    data, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if len(mono) == 0:
        raise ValueError("empty audio")
    peak = float(np.abs(mono).max())
    rms = float(np.sqrt(np.mean(mono ** 2)))
    rms_db = 20 * math.log10(rms + 1e-9)
    clip = float((np.abs(mono) >= 0.999).mean())
    frame = max(1, int(rate * FRAME_MS / 1000))
    n = len(mono) // frame
    out = {"peak": round(peak, 4), "rms_db": round(rms_db, 1), "clip": round(clip, 5),
           "lead_ms": 0, "tail_ms": 0, "pauses": 0, "max_gap_ms": 0}
    if n < 2:
        return out
    env = 20 * np.log10(np.sqrt((mono[: n * frame].reshape(n, frame) ** 2).mean(axis=1)) + 1e-9)
    voiced = env > max(-60.0, rms_db - 30.0)
    idx = np.flatnonzero(voiced)
    if len(idx) == 0:
        out.update(lead_ms=n * FRAME_MS, tail_ms=n * FRAME_MS)
        return out
    out["lead_ms"] = int(idx[0]) * FRAME_MS
    out["tail_ms"] = int(n - 1 - idx[-1]) * FRAME_MS
    run = longest = pauses = 0
    for v in voiced[idx[0]: idx[-1] + 1]:
        if v:
            if run * FRAME_MS >= PAUSE_MS:
                pauses += 1
            longest = max(longest, run)
            run = 0
        else:
            run += 1
    out.update(pauses=pauses, max_gap_ms=longest * FRAME_MS)
    return out


def get(line: dict) -> Optional[dict]:
    with _lock:
        _load()
        return _stats.get(line.get("sha256") or "")


def status() -> dict:
    with _lock:
        _load()
        return {**_job, "cached": len(_stats)}


def start(corpus) -> dict:
    """Measure every clip that has no cached numbers, in the background. One run at a time."""
    with _lock:
        _load()
        if _job["running"]:
            return {**_job, "cached": len(_stats)}
        todo = [l for l in corpus.lines if l.get("sha256") and l["sha256"] not in _stats]
        _job.update(running=True, done=0, total=len(todo), failed=0, started_at=time.time(), finished_at=None)
    threading.Thread(target=_run, args=(corpus, todo), name="fish-dataset-analyze", daemon=True).start()
    return status()


def _run(corpus, todo: list[dict]) -> None:
    try:
        for i, line in enumerate(todo, 1):
            path = os.path.join(corpus.root, line["file"])
            try:
                result = measure(path)
                with _lock:
                    _stats[line["sha256"]] = result
            except Exception:  # noqa: BLE001 — one unreadable clip must not stop the rest
                with _lock:
                    _job["failed"] += 1
            with _lock:
                _job["done"] = i
            if i % 250 == 0:
                with _lock:
                    _save()
    finally:
        with _lock:
            _save()
            _job.update(running=False, finished_at=time.time())
