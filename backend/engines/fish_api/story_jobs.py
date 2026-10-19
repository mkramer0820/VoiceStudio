"""Fish story render jobs: organized output, durable records, request log, resume.

Each render gets its own run folder::

    <output root>/<story title>/run-<NNN>_<YYYYMMDD-HHMMSS>_<job id>/
        001_Shadow_sun_2.wav ...   one untouched Fish response per line
        story.wav                  joined file (WAV/PCM only)
        job.json                   live state: survives restarts, drives the Jobs list
        story.json                 the submitted draft (so a run can be resumed)
        settings.json              the effective options used (never the key)
        requests.jsonl             one record per line: exact request, timing, hash
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
from .config import FishConfig, load_config, resolve_api_key
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


def _find_run_dir(job_id: str) -> Optional[str]:
    if not JOB_ID_RE.match(job_id or ""):
        return None
    for root in _scan_roots():
        for hit in glob.glob(os.path.join(glob.escape(root), "*", f"*_{job_id}", JOB_FILE)):
            return os.path.dirname(hit)
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
    for index, chunk in _items(draft, chunk_ids):
        voice = chunk.voice or default
        body = client.build_body(chunk.text.strip(), cfg, reference_id=voice, speed=draft.settings.speed)
        issues = story_check(body)
        notes = [] if voice else ["No voice set: Fish will use its default voice."]
        problems += bool(issues)
        rows.append({"index": index, "id": chunk.id, "speaker": chunk.speaker, "voice": voice,
                     "voice_name": draft.voices.get(voice or "", voice),
                     "file": _audio_name(index, chunk.speaker, draft.voices.get(voice or "", voice), ext),
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
    return {"ok": bool(rows) and problems == 0, "total": len(rows), "invalid": problems,
            "shown": len(shown), "endpoint": cfg.base_url + "/v1/tts",
            "headers": {"Authorization": "Bearer <hidden>", "model": cfg.model,
                        "Content-Type": "application/json"},
            "folder": folder, "requests": shown, "curl": curl,
            "first_invalid": next((r["index"] for r in rows if r["issues"]), None)}


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
    for index, chunk in items:  # the same schema check the preview shows, before anything is sent
        bad = story_check(client.build_body(chunk.text.strip(), cfg, reference_id=chunk.voice or default,
                                            speed=draft.settings.speed))
        if bad:
            raise ValueError(f"Line {index} would be rejected: {bad[0]}")

    root = output_root(cfg)
    job_id = uuid.uuid4().hex[:12]
    story_dir = _story_dir(root, draft.title, draft_id)
    _claim_story_dir(story_dir, draft_id)
    run_dir = os.path.join(story_dir, _next_run_name(story_dir, job_id))
    os.makedirs(run_dir, exist_ok=True)
    job = _new_job(run_dir, draft, draft_id, cfg,
                   [_chunk_entry(i, c, default, draft.voices) for i, c in items], job_id, time.time())
    try:
        _write_json(os.path.join(run_dir, STORY_FILE), {"draft_id": draft_id, "draft": draft.model_dump()})
        _write_json(os.path.join(run_dir, SETTINGS_FILE), {
            **{k: v for k, v in dataclasses.asdict(cfg).items() if k != "base_url"},
            "speed": draft.settings.speed, "pause_ms": draft.settings.pause_ms,
            "join": draft.settings.join, "default_voice": default,
        })
    except OSError as exc:
        raise ValueError(f"Cannot write to the output folder ({type(exc).__name__}).") from exc
    _register(job)
    _log(job, "INFO", f"Started: {len(items)} lines, model {cfg.model}, {cfg.format} "
                      f"{cfg.sample_rate} Hz, {cfg.max_concurrency} at a time.")
    _persist(job)
    threading.Thread(target=_run, args=(job, items, draft, cfg, api_key, default),
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
    _log(job, "INFO", f"Resumed: {len(items)} lines left.")
    _persist(job)
    threading.Thread(target=_run, args=(job, items, draft, cfg, api_key, default),
                     name=f"fish-story-{job_id}", daemon=True).start()
    return _view(job)


def _run(job: dict, items: list, draft: Draft, cfg: FishConfig, api_key: str,
         default: Optional[str]) -> None:
    fatal: dict = {}
    slots = {entry["id"]: n for n, entry in enumerate(job["chunks"])}
    ext = _EXT.get(cfg.format, "bin")

    def one(index: int, chunk: Chunk) -> None:
        slot = slots[chunk.id]
        entry = job["chunks"][slot]
        if job["cancel"].is_set() or fatal:
            with job["lock"]:
                entry["status"] = "skipped"
            _log(job, "INFO", f"#{index} skipped.")
            return
        voice = chunk.voice or default
        with job["lock"]:
            entry["status"] = "running"
        meta: dict = {}
        body = client.build_body(chunk.text.strip(), cfg, reference_id=voice, speed=draft.settings.speed)
        record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "index": index, "id": chunk.id,
                  "speaker": chunk.speaker, "voice": voice, "request": body}
        try:
            raw = client.synthesize(body, api_key=api_key, cfg=cfg, meta=meta)
            name = _audio_name(index, chunk.speaker, entry["voice_name"], ext)
            with open(os.path.join(job["out_dir"], name), "wb") as fh:
                fh.write(raw)
            audio = _audio_info(raw, cfg.format, cfg)
            digest = hashlib.sha256(raw).hexdigest()
            with job["lock"]:
                entry.update(status="done", file=name, bytes=len(raw), sha256=digest, error=None,
                             attempts=meta.get("attempts", 1), elapsed_ms=meta.get("elapsed_ms", 0),
                             duration_s=audio["duration_s"] if audio else None,
                             sample_rate=audio["sample_rate"] if audio else None)
                job["done"] += 1
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
            _log(job, "INFO", f"#{index} ok: {name} ({len(raw)} bytes, {meta.get('elapsed_ms', 0)} ms, "
                              f"{meta.get('attempts', 1)} attempt(s))")
        except client.FishAPIError as exc:
            if exc.status in _FATAL:
                fatal.update(status=exc.status, message=exc.message)
            with job["lock"]:
                entry.update(status="failed", error=exc.message, attempts=meta.get("attempts", 0),
                             elapsed_ms=meta.get("elapsed_ms", 0))
                job["failed"] += 1
            record.update(status=exc.status, attempts=meta.get("attempts", 0),
                          elapsed_ms=meta.get("elapsed_ms", 0), error=exc.message)
            _log(job, "ERROR", f"#{index} failed (HTTP {exc.status}): {exc.message}")
        except Exception as exc:  # noqa: BLE001 — one line must not kill the batch
            with job["lock"]:
                entry.update(status="failed", error=type(exc).__name__)
                job["failed"] += 1
            record.update(status=0, error=type(exc).__name__)
            _log(job, "ERROR", f"#{index} failed: {type(exc).__name__}")
        _record(job, record)
        _persist(job)

    try:
        with ThreadPoolExecutor(max_workers=cfg.max_concurrency) as pool:
            for index, chunk in items:
                pool.submit(one, index, chunk)
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
        for entry in sorted(job["chunks"], key=lambda c: c["index"]):
            if entry["status"] != "done":
                continue
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
        out = os.path.join(job["out_dir"], "story.wav")
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
        "speed": draft.settings.speed, "pause_ms": draft.settings.pause_ms,
        "chunks": [{**c, "text": text.get(c["id"], "")} for c in sorted(view["chunks"], key=lambda c: c["index"])],
    }
    try:
        _write_json(os.path.join(job["out_dir"], MANIFEST_FILE), info)
    except OSError:
        pass


__all__ = ["start_render", "resume_render", "get_job", "list_jobs", "read_log", "cancel_job",
           "output_root", "validate_output_dir", "default_output_root"]
