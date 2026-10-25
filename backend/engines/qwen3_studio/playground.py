"""Qwen3-TTS playground: several prompts, each with its own model and voice, in one run.

A *take* is one prompt with its own model, voice kind and sampling settings.
Voice kinds:

* ``custom_voice``    a speaker baked into a custom_voice checkpoint (the narrator), optional instruct;
* ``clone``           a Base model cloning a reference clip (clip + its text, or the speaker embedding only);
* ``clone_on_custom`` a custom_voice checkpoint speaking with a reference clip's speaker embedding,
                      made once by a Base model (experimental).

The runner keeps one model on the GPU, optionally orders takes so a model loads once, warms each
model and voice up with throw-away renders before the first real take, and writes every take as its
own WAV next to a ``session.json`` manifest. Nothing here contacts the network.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from typing import Literal, Optional

from pydantic import BaseModel, Field

from . import registry

logger = logging.getLogger("omnivoice.qwen_studio")

MANIFEST = "session.json"
SESSION_RE = re.compile(r"^[A-Za-z0-9_.-]{1,120}$")
FILE_RE = re.compile(r"^[A-Za-z0-9_. -]{1,200}$")
_MAX_LIVE = 20


class Voice(BaseModel):
    kind: Literal["custom_voice", "clone", "clone_on_custom"] = "custom_voice"
    speaker: Optional[str] = Field(None, max_length=64)  # custom_voice / clone_on_custom: row to speak with
    instruct: Optional[str] = Field(None, max_length=1000)
    profile_id: Optional[str] = Field(None, max_length=64)  # an app voice profile: its clip and text
    ref_audio: Optional[str] = Field(None, max_length=500)  # or a clip path
    ref_text: Optional[str] = Field(None, max_length=5000)
    xvec_only: bool = False  # clone: speaker embedding only (free tempo) instead of clip + text


class Params(BaseModel):
    language: str = Field("Auto", max_length=35)
    seed: Optional[int] = Field(None, ge=0, le=2**31 - 1)
    temperature: Optional[float] = Field(None, ge=0.05, le=2.0)
    top_k: Optional[int] = Field(None, ge=0, le=500)
    top_p: Optional[float] = Field(None, ge=0.05, le=1.0)
    repetition_penalty: Optional[float] = Field(None, ge=0.5, le=2.0)
    pad_s: float = Field(0.3, ge=0.0, le=5.0)  # silence after each chunk
    chunk_chars: Optional[int] = Field(None, ge=20, le=2000)
    max_seq_len: Optional[int] = Field(None, ge=512, le=4096)  # KV cache length; changing it reloads the model


class Take(BaseModel):
    id: str = Field(..., pattern=r"^[A-Za-z0-9_-]{1,40}$")
    label: str = Field("", max_length=80)
    text: str = Field(..., min_length=1, max_length=20000)
    model_id: str = Field(..., pattern=r"^[A-Za-z0-9_.-]{1,160}$")
    voice: Voice = Field(default_factory=Voice)
    params: Params = Field(default_factory=Params)
    repeat: int = Field(1, ge=1, le=8)  # same take rendered this many times (new seed each when seed is unset)


class RunRequest(BaseModel):
    session_name: str = Field("playground", max_length=80)
    takes: list[Take] = Field(..., min_length=1, max_length=200)
    group_by_model: bool = True  # run order only; results keep the order you wrote them in
    warmup_runs: Optional[int] = Field(None, ge=0, le=10)
    warmup_text: Optional[str] = Field(None, max_length=500)
    output_dir: Optional[str] = Field(None, max_length=500)


_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_run_lock = threading.Lock()  # the GPU holds one model: one run at a time
_backend = None
_backend_lock = threading.Lock()


# ── naming ────────────────────────────────────────────────────────────────
def slug(text: str, fallback: str = "x", limit: int = 40) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()[:limit] or fallback


def take_filename(number: int, rep: int, repeats: int, model_id: str, voice: Voice, label: str) -> str:
    who = voice.speaker or voice.profile_id or ("clip" if voice.ref_audio else "default")
    parts = [f"take-{number:03d}" + (f"-r{rep}" if repeats > 1 else ""), slug(model_id, "model", 48),
             slug(voice.kind.replace("_", "-")), slug(who, "voice", 24)]
    if label.strip():
        parts.append(slug(label, "label", 28))
    return "_".join(parts) + ".wav"


def session_folder_name(name: str, when: float) -> str:
    return f"{slug(name, 'playground', 50)}_{time.strftime('%Y%m%d-%H%M%S', time.localtime(when))}"


# ── planning ──────────────────────────────────────────────────────────────
def voice_signature(take: Take) -> tuple:
    v = take.voice
    return (take.model_id, take.params.max_seq_len, v.kind, v.speaker, v.profile_id, v.ref_audio, v.xvec_only)


def plan(takes: list[Take], group_by_model: bool) -> list[tuple[int, int]]:
    """(take index, repetition) in run order. Grouping only reorders so the same model and voice
    run back to back; the first appearance of each model keeps its place."""
    items = [(i, r) for i, t in enumerate(takes) for r in range(1, t.repeat + 1)]
    if not group_by_model:
        return items
    order: dict[tuple, int] = {}
    for i, _ in items:
        order.setdefault((takes[i].model_id, takes[i].params.max_seq_len), len(order))
    voices: dict[tuple, int] = {}
    for i, _ in items:
        voices.setdefault(voice_signature(takes[i]), len(voices))
    return sorted(items, key=lambda it: (order[(takes[it[0]].model_id, takes[it[0]].params.max_seq_len)],
                                         voices[voice_signature(takes[it[0]])]))


def swaps(takes: list[Take], order: list[tuple[int, int]]) -> int:
    """How many times the resident model changes in this order (the first load counts)."""
    last, count = None, 0
    for i, _ in order:
        key = (takes[i].model_id, takes[i].params.max_seq_len)
        if key != last:
            count += 1
            last = key
    return count


# ── voice resolution ──────────────────────────────────────────────────────
def _profile(profile_id: str) -> dict:
    from core.config import VOICES_DIR
    from core.db import db_conn

    with db_conn() as conn:
        row = conn.execute("SELECT id, name, ref_audio_path, ref_text FROM voice_profiles WHERE id = ?",
                           (profile_id,)).fetchone()
    if row is None:
        raise ValueError(f"Voice profile {profile_id!r} does not exist.")
    row = dict(row)
    if not row.get("ref_audio_path"):
        raise ValueError(f"Voice profile {row['name']!r} has no reference clip.")
    return {"name": row["name"], "ref_audio": os.path.join(VOICES_DIR, row["ref_audio_path"]),
            "ref_text": row.get("ref_text") or ""}


def resolve_voice(take: Take, model: dict, settings: dict) -> tuple[dict, dict]:
    """(voice dict for the sidecar, what the manifest records about it)."""
    v = take.voice
    sent: dict = {"kind": v.kind}
    if v.instruct and v.instruct.strip():
        sent["instruct"] = v.instruct.strip()
    info: dict = {"kind": v.kind, "instruct": sent.get("instruct")}
    if v.kind in ("clone", "clone_on_custom"):
        ref_audio, ref_text, name = v.ref_audio, v.ref_text or "", None
        if v.profile_id:
            prof = _profile(v.profile_id)
            ref_audio, name = prof["ref_audio"], prof["name"]
            ref_text = v.ref_text if v.ref_text is not None and v.ref_text.strip() else prof["ref_text"]
            info["profile_id"], info["profile_name"] = v.profile_id, name
        if not ref_audio or not os.path.isfile(ref_audio):
            raise ValueError("A clone needs a reference clip (pick a voice profile or give a file path).")
        sent["ref_audio"] = os.path.abspath(ref_audio)
        info["ref_audio"] = sent["ref_audio"]
        if v.kind == "clone":
            sent["xvec_only"] = v.xvec_only
            sent["ref_text"] = ref_text
            info["xvec_only"], info["ref_text"] = v.xvec_only, ref_text
            if not v.xvec_only and not ref_text.strip():
                raise ValueError("Clip-and-text cloning needs the reference text (or tick 'speaker embedding only').")
        else:
            base = registry.first_base(settings)
            if base is None:
                raise ValueError("'Clone on custom voice' needs a stock Base model in the Hugging Face cache.")
            sent["base_model_path"] = base["path"]
            sent["speaker"] = v.speaker or (model.get("speakers") or [None])[0]
            sent["xvec_path"] = os.path.join(registry.studio_dir(), "xvectors",
                                             f"{slug(os.path.basename(ref_audio), 'ref', 60)}_{slug(base['id'], 'b', 30)}.npy")
            info["base_model_id"], info["speaker"] = base["id"], sent["speaker"]
    else:
        sent["speaker"] = v.speaker or (model.get("speakers") or [None])[0]
        info["speaker"] = sent["speaker"]
    return sent, info


def _check_pair(model: dict, voice: Voice) -> None:
    want = {"custom_voice": "custom_voice", "clone_on_custom": "custom_voice", "clone": "base"}[voice.kind]
    if model["type"] != want:
        raise ValueError(f"'{voice.kind}' needs a {want} model, but {model['label']} is {model['type']}.")


# ── engine ────────────────────────────────────────────────────────────────
def _engine(settings: dict):
    global _backend
    from engines import qwen3_narrator as engine

    engine.set_python_override(settings.get("python") or None)
    with _backend_lock:
        if _backend is None:
            _backend = engine.Qwen3NarratorBackend()
        return _backend


def unload_engine() -> None:
    global _backend
    with _backend_lock:
        if _backend is not None:
            _backend.unload()
            _backend = None


# ── jobs ──────────────────────────────────────────────────────────────────
def _write_json(path: str, data) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _log(job: dict, level: str, message: str) -> None:
    job["log"].append({"t": time.time(), "level": level, "message": message})
    del job["log"][:-400]
    logger.log(logging.INFO if level == "INFO" else logging.WARNING, "[qwen-playground %s] %s", job["id"], message)


def _persist(job: dict) -> None:
    saved = {k: v for k, v in job.items() if k not in ("log", "cancel")}
    _write_json(os.path.join(job["folder"], MANIFEST), saved | {"log_tail": job["log"][-40:]})


def view(job: dict) -> dict:
    return {k: v for k, v in job.items() if k != "cancel"}


def is_running() -> bool:
    return _run_lock.locked()


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


def start_run(req: RunRequest) -> dict:
    """Validate everything that can be checked without a GPU, then run in the background."""
    settings = registry.load_settings()
    models = {m["id"]: m for m in registry.discover_models(settings)}
    for take in req.takes:
        model = models.get(take.model_id)
        if model is None:
            raise ValueError(f"Take '{take.label or take.id}': model {take.model_id!r} was not found.")
        _check_pair(model, take.voice)
        resolve_voice(take, model, settings)  # raises with a plain reason before anything loads
    from engines import qwen3_narrator as engine

    engine.set_python_override(settings.get("python") or None)
    ok, reason = engine.availability(ignore_checkpoint=True)
    if not ok:
        raise RuntimeError(reason)
    if not _run_lock.acquire(blocking=False):
        raise RuntimeError("A playground run is already in progress; wait for it or cancel it.")
    try:
        now = time.time()
        root = (req.output_dir or "").strip() or registry.output_dir(settings)
        folder = os.path.join(root, session_folder_name(req.session_name, now))
        os.makedirs(folder, exist_ok=False)
        order = plan(req.takes, req.group_by_model)
        runs = req.warmup_runs if req.warmup_runs is not None else settings["warmup_runs"]
        job = {
            "id": uuid.uuid4().hex[:12], "session": os.path.basename(folder), "folder": folder, "status": "running",
            "name": req.session_name, "created": now, "finished": None, "group_by_model": req.group_by_model,
            "warmup_runs": runs, "warmup_text": req.warmup_text or settings["warmup_text"],
            "model_swaps": swaps(req.takes, order), "total": len(order), "done": 0, "message": "",
            "takes": [], "log": [], "cancel": threading.Event(),
        }
        for i, t in enumerate(req.takes):
            m = models[t.model_id]
            job["takes"].append({
                "id": t.id, "number": i + 1, "label": t.label, "text": t.text, "model_id": t.model_id,
                "model": {"label": m["label"], "path": m["path"], "type": m["type"]},
                "params": t.params.model_dump(), "voice": t.voice.model_dump(), "repeat": t.repeat,
                "results": [], "notes": "", "starred": False,
            })
        _persist(job)
    except BaseException:
        _run_lock.release()
        raise
    with _jobs_lock:
        for old in [j for j, v in _jobs.items() if v["status"] != "running"][: max(0, len(_jobs) - _MAX_LIVE)]:
            _jobs.pop(old, None)
        _jobs[job["id"]] = job
    _log(job, "INFO", f"{len(order)} render(s), {job['model_swaps']} model load(s), warm-up {runs}x per voice.")
    threading.Thread(target=_run, args=(job, req, order, models, settings), name=f"qwen-pg-{job['id']}", daemon=True).start()
    return view(job)


def _save_wav(path: str, tensor, rate: int) -> float:
    import soundfile as sf

    audio = tensor.squeeze(0).numpy()
    sf.write(path, audio, rate)
    return len(audio) / rate


def _run(job: dict, req: RunRequest, order: list[tuple[int, int]], models: dict, settings: dict) -> None:
    warmed: set[tuple] = set()
    try:
        backend = _engine(settings)
        for n, (index, rep) in enumerate(order, 1):
            if job["cancel"].is_set():
                job["status"], job["message"] = "cancelled", "Cancelled."
                break
            take, entry = req.takes[index], job["takes"][index]
            model = models[take.model_id]
            result: dict = {"rep": rep, "status": "running", "file": None}
            entry["results"].append(result)
            try:
                voice, info = resolve_voice(take, model, settings)
                result["voice"] = info
                common = {"model_path": model["path"], "voice": voice, "language": take.params.language,
                          "pad_s": take.params.pad_s}
                for key in ("temperature", "top_k", "top_p", "repetition_penalty", "chunk_chars", "max_seq_len"):
                    if getattr(take.params, key) is not None:
                        common[key] = getattr(take.params, key)
                # Warm-up is per resolved voice: an unset speaker and the model's own speaker are the same voice.
                sig = (take.model_id, take.params.max_seq_len, voice["kind"], voice.get("speaker"),
                       voice.get("ref_audio"), voice.get("xvec_only"))
                if sig not in warmed and job["warmup_runs"] > 0:
                    for w in range(job["warmup_runs"]):
                        _log(job, "INFO", f"Warm-up {w + 1}/{job['warmup_runs']}: {model['label']} / {take.voice.kind}")
                        backend.generate(job["warmup_text"], seed=1000 + w, **common)
                    warmed.add(sig)
                    result["warmed_up"] = job["warmup_runs"]
                seed = take.params.seed if take.params.seed is not None else int(time.time() * 1000) % 2**31
                if take.params.seed is not None and take.repeat > 1:
                    seed += rep - 1
                started = time.time()
                tensor = backend.generate(take.text, seed=seed, **common)
                took = time.time() - started
                meta = dict(getattr(backend, "last_reply_meta", {}) or {})
                name = take_filename(index + 1, rep, take.repeat, take.model_id, take.voice, take.label)
                seconds = _save_wav(os.path.join(job["folder"], name), tensor, backend.sample_rate)
                result.update(status="done", file=name, seed=seed, seconds=round(seconds, 2), gen_seconds=round(took, 2),
                              realtime=round(seconds / took, 2) if took else None, sample_rate=backend.sample_rate,
                              complete=meta.get("complete"), chunks=meta.get("chunks"),
                              vram_allocated_mb=meta.get("vram_allocated_mb"), vram_reserved_mb=meta.get("vram_reserved_mb"))
                if meta.get("complete") is False:
                    _log(job, "WARN", f"Take {index + 1}: a chunk hit the length limit (the end-of-speech token was not produced).")
                _log(job, "INFO", f"[{n}/{len(order)}] take {index + 1} -> {name} ({seconds:.1f}s, {result['realtime']}x)")
            except Exception as exc:  # noqa: BLE001 - one bad take must not stop the others
                result.update(status="error", error=f"{type(exc).__name__}: {exc}")
                _log(job, "ERROR", f"Take {index + 1} failed: {result['error']}")
            job["done"] = n
            _persist(job)
        else:
            job["status"] = "done"
        if job["status"] == "running":
            job["status"] = "done"
    except Exception as exc:  # noqa: BLE001
        job["status"], job["message"] = "error", f"{type(exc).__name__}: {exc}"
        _log(job, "ERROR", job["message"])
    finally:
        job["finished"] = time.time()
        try:
            _persist(job)
        finally:
            _run_lock.release()


# ── sessions on disk ──────────────────────────────────────────────────────
def _session_path(root: str, session: str) -> str:
    if not SESSION_RE.match(session):
        raise ValueError("Bad session name.")
    return os.path.join(root, session)


def list_sessions(root: Optional[str] = None, limit: int = 200) -> list[dict]:
    root = root or registry.output_dir()
    out = []
    try:
        names = sorted(os.listdir(root), reverse=True)
    except OSError:
        return out
    for name in names:
        try:
            data = json.load(open(os.path.join(root, name, MANIFEST), encoding="utf-8"))
        except (OSError, ValueError):
            continue
        takes = data.get("takes") or []
        out.append({"session": name, "name": data.get("name", name), "created": data.get("created"), "status": data.get("status"),
                    "takes": len(takes), "renders": sum(len(t.get("results", [])) for t in takes), "folder": os.path.join(root, name)})
        if len(out) >= limit:
            break
    return out


def read_session(session: str, root: Optional[str] = None) -> Optional[dict]:
    try:
        path = os.path.join(_session_path(root or registry.output_dir(), session), MANIFEST)
        return json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        return None


def annotate_take(session: str, take_id: str, *, notes: Optional[str] = None, starred: Optional[bool] = None,
                  root: Optional[str] = None) -> Optional[dict]:
    """Save the user's notes and star on a take into the manifest (the only edit the manifest takes)."""
    root = root or registry.output_dir()
    path = os.path.join(_session_path(root, session), MANIFEST)
    try:
        data = json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        return None
    for take in data.get("takes", []):
        if take["id"] == take_id:
            if notes is not None:
                take["notes"] = notes[:5000]
            if starred is not None:
                take["starred"] = bool(starred)
            _write_json(path, data)
            with _jobs_lock:
                live = next((j for j in _jobs.values() if j["session"] == session), None)
            if live:  # keep the running job's copy in step so its next save does not undo this
                for t in live["takes"]:
                    if t["id"] == take_id:
                        t["notes"], t["starred"] = take["notes"], take["starred"]
            return take
    return None


def audio_path(session: str, name: str, root: Optional[str] = None) -> Optional[str]:
    if not FILE_RE.match(name) or not name.lower().endswith(".wav"):
        return None
    path = os.path.join(_session_path(root or registry.output_dir(), session), name)
    return path if os.path.isfile(path) else None
