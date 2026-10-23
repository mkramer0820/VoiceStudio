"""Fish training runs: turn an ``omnivoice`` export into tokens and a LoRA adapter, locally.

A run lives in its own folder under ``<data dir>/fish_training/``::

    run_<time>_<id>/
        job.json            live state (survives restarts; drives the runs list)
        run.log             every event plus the output of the child processes
        train_config.json   OmniVoice TrainingConfig (LoRA on)
        data_config.json    points at the tokenised shards
        tokens/<split>/     WebDataset shards + data.lst (tokenising is skipped when already done)
        output/             checkpoints (checkpoint-N/adapter/ holds the small adapter files)
        adapter/            the finished adapter + adapter.json + evaluation.json
        heldout.jsonl       the fixed validation lines a generation check should use
        dry_run.json        only for dry runs: what was validated and what would run

Tokenising and training run as **child processes** (``Launcher``) so a wedged GPU job can be killed
(see docs/engines/omnivoice-subprocess.md). A *dry run* validates the export, builds every config and
command line and probes the hardware without a GPU, a model download or a training step. Nothing here
contacts a network except what the child processes themselves do when ``allow_download`` is on; with it
off they run with ``HF_HUB_OFFLINE=1``. Nothing starts unless training was switched on in Settings and
the caller confirmed.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from typing import Any, Optional, Protocol

from . import training_eval
from .story import _store_dir
from .story_jobs import _write_json

logger = logging.getLogger("omnivoice.fish.training")

ENABLED_KEY = "tts.fish.training_enabled"
JOB_FILE, LOG_FILE, DRY_FILE = "job.json", "run.log", "dry_run.json"
TRAIN_JSONL, VAL_JSONL = "omnivoice_train.jsonl", "omnivoice_val.jsonl"
SAMPLE_RATE = 24000
ACTIVE = "running"
FINISHED = {"done", "failed", "cancelled", "interrupted", "dry_run"}
RESUMABLE = {"failed", "cancelled", "interrupted"}

# Size guidance, not hard rules: owner decision pending (handoff section 9).
MIN_CLIPS_BLOCK = 20
MIN_CLIPS_WARN = 300
MIN_MINUTES_WARN = 30
MIN_VRAM_GB_BLOCK = 6.0
VRAM_GB_COMFORT = 7.5

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


# ── opt-in setting ──
def is_enabled() -> bool:
    from services import settings_store

    return (settings_store.get_text(ENABLED_KEY) or "") == "1"


def set_enabled(on: bool) -> bool:
    from services import settings_store

    if on:
        settings_store.set_text(ENABLED_KEY, "1")
    else:
        settings_store.clear_text(ENABLED_KEY)
    return bool(on)


def training_root() -> str:
    path = os.path.join(os.path.dirname(_store_dir()), "fish_training")
    os.makedirs(path, exist_ok=True)
    return path


# ── options ──
@dataclasses.dataclass
class TrainOptions:
    steps: int = 2000
    learning_rate: float = 1e-4
    batch_tokens: int = 4096
    gradient_accumulation_steps: int = 1
    lora_r: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05
    train_heads: bool = False  # also train audio_heads in full next to the adapter
    seed: int = 42
    base_checkpoint: str = "k2-fsa/OmniVoice"
    tokenizer_path: str = "eustlb/higgs-audio-v2-tokenizer"
    python: Optional[str] = None  # interpreter that has torch + accelerate + omnivoice; default: this one
    allow_download: bool = False  # off = child processes run with HF_HUB_OFFLINE=1

    _RANGES = {"steps": (10, 200000), "learning_rate": (1e-6, 1e-2), "batch_tokens": (512, 65536),
               "gradient_accumulation_steps": (1, 64), "lora_r": (1, 256), "lora_alpha": (1, 512),
               "lora_dropout": (0.0, 0.5), "seed": (0, 2**31 - 1)}

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "TrainOptions":
        data = dict(data or {})
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"Unknown training option(s): {', '.join(unknown)}.")
        opts = cls(**{k: v for k, v in data.items() if v is not None})
        for name, (low, high) in cls._RANGES.items():
            value = getattr(opts, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
                raise ValueError(f"{name} must be a number between {low} and {high}.")
        for name in ("steps", "batch_tokens", "gradient_accumulation_steps", "lora_r", "seed"):
            setattr(opts, name, int(getattr(opts, name)))
        for name in ("base_checkpoint", "tokenizer_path"):
            value = str(getattr(opts, name)).strip()
            if not value or len(value) > 400 or "\x00" in value:
                raise ValueError(f"{name} must be a model id or a folder path.")
            setattr(opts, name, value)
        if opts.python:
            if not os.path.isfile(opts.python):
                raise ValueError("The Python interpreter for training does not exist.")
        else:
            opts.python = None
        opts.train_heads, opts.allow_download = bool(opts.train_heads), bool(opts.allow_download)
        return opts


def defaults() -> dict:
    d = dataclasses.asdict(TrainOptions())
    d["python"] = sys.executable
    return d


# ── export validation ──
def _read_jsonl(path: str) -> tuple[list[dict], int]:
    rows, bad = [], 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                bad += 1
                continue
            if isinstance(row, dict):
                rows.append(row)
            else:
                bad += 1
    return rows, bad


def validate_export(export_dir: str, *, check_audio: bool = True) -> dict:
    """Is this folder a usable ``omnivoice`` export? Reads only; returns errors, warnings and totals."""
    out: dict[str, Any] = {"ok": False, "errors": [], "warnings": [], "train_clips": 0, "val_clips": 0,
                           "seconds": 0.0, "speakers": {}, "card": None, "loudness_matched": None,
                           "train_jsonl": None, "val_jsonl": None}
    errors, warnings = out["errors"], out["warnings"]
    raw = (export_dir or "").strip().strip('"')
    if not raw or not os.path.isabs(raw) or "\x00" in raw:
        errors.append("Give the full path of an export folder made with the OmniVoice format.")
        return out
    folder = os.path.normpath(raw)
    train_path, val_path = os.path.join(folder, TRAIN_JSONL), os.path.join(folder, VAL_JSONL)
    if not os.path.isdir(folder):
        errors.append("That export folder does not exist.")
        return out
    if not os.path.isfile(train_path):
        errors.append(f"{TRAIN_JSONL} is missing. Export again with the 'OmniVoice training files' format.")
        return out
    out["train_jsonl"] = train_path
    try:
        with open(os.path.join(folder, "dataset.json"), encoding="utf-8") as fh:
            out["card"] = card = json.load(fh)
    except (OSError, ValueError):
        card = None
    train, bad = _read_jsonl(train_path)
    val: list[dict] = []
    if os.path.isfile(val_path):
        out["val_jsonl"] = val_path
        val, bad_val = _read_jsonl(val_path)
        bad += bad_val
    if bad:
        errors.append(f"{bad} line(s) in the JSONL files are not valid JSON objects.")
    seen_ids: set = set()
    seen_audio: dict[str, str] = {}
    problems: dict[str, int] = {}
    seconds = 0.0
    sf = None
    if check_audio:
        import soundfile as sf  # noqa: PLC0415 — only needed when the audio is inspected

    def note(kind: str) -> None:
        problems[kind] = problems.get(kind, 0) + 1

    for split, rows in (("train", train), ("val", val)):
        for row in rows:
            if not all(isinstance(row.get(k), str) and row[k].strip() for k in ("id", "audio_path", "text", "language_id")):
                note("a row is missing id, audio_path, text or language_id")
                continue
            if row["id"] in seen_ids:
                note("an id appears twice")
            seen_ids.add(row["id"])
            path = row["audio_path"]
            other = seen_audio.get(path)
            if other and other != split:
                note("a clip is in both train and validation")
            seen_audio[path] = split
            if not os.path.isabs(path) or not os.path.isfile(path):
                note("an audio file is missing")
                continue
            if sf is not None:
                try:
                    info = sf.info(path)
                except Exception:  # noqa: BLE001 — any unreadable file is just a bad clip
                    note("an audio file cannot be read")
                    continue
                if info.samplerate != SAMPLE_RATE or info.channels != 1:
                    note(f"a clip is not {SAMPLE_RATE} Hz mono")
                    continue
                if split == "train":
                    seconds += info.duration
    for kind, n in problems.items():
        errors.append(f"{n} problem(s): {kind}.")
    out.update(train_clips=len(train), val_clips=len(val), seconds=round(seconds, 1))
    if card:
        out["speakers"] = card.get("speakers") or {}
        out["loudness_matched"] = bool(card.get("loudness"))
        if len(out["speakers"]) > 1:
            warnings.append("This export mixes several speakers. A LoRA learns one voice; export a single speaker.")
        if not card.get("loudness"):
            warnings.append("Loudness was not matched on export. Chapters rendered at different levels teach the "
                            "model both; export with 'bring all clips to one loudness level'.")
    if len(train) < MIN_CLIPS_BLOCK:
        errors.append(f"Only {len(train)} training clip(s); at least {MIN_CLIPS_BLOCK} are needed.")
    elif len(train) < MIN_CLIPS_WARN or (check_audio and seconds < MIN_MINUTES_WARN * 60):
        warnings.append(f"This is a small set ({len(train)} clips, {seconds / 60:.0f} min). The adapter may memorise "
                        f"it; about {MIN_CLIPS_WARN}+ clips and {MIN_MINUTES_WARN}+ minutes of one voice is safer.")
    if not val:
        warnings.append("No validation clips: the evaluation gate has nothing held out. Export with a validation share.")
    out["ok"] = not errors
    return out


# ── configs and commands ──
def build_train_config(opts: TrainOptions, *, has_val: bool, resume_from: Optional[str] = None) -> dict:
    """The OmniVoice ``TrainingConfig`` JSON for a LoRA run. Every key is a real ``TrainingConfig`` field."""
    save = max(50, opts.steps // 5)
    return {
        "llm_name_or_path": "Qwen/Qwen3-0.6B",
        "init_from_checkpoint": opts.base_checkpoint,
        "resume_from_checkpoint": resume_from,
        "lora_enabled": True, "lora_r": opts.lora_r, "lora_alpha": opts.lora_alpha, "lora_dropout": opts.lora_dropout,
        "lora_train_modules": ["audio_heads"] if opts.train_heads else [],
        "eval_at_start": has_val,
        "learning_rate": opts.learning_rate, "steps": opts.steps, "seed": opts.seed,
        "warmup_type": "ratio", "warmup_ratio": 0.05,
        "batch_tokens": opts.batch_tokens, "gradient_accumulation_steps": opts.gradient_accumulation_steps,
        "num_workers": 2,
        "mixed_precision": "bf16", "allow_tf32": True,
        # finetune-style sampling (no pinyin / instruct prompts: the export carries plain text only)
        "use_pinyin_ratio": 0.0, "instruct_ratio": 0.0, "only_instruct_ratio": 0.0, "language_ratio": 0.8,
        "logging_steps": min(25, max(1, opts.steps // 10)),
        "eval_steps": save, "save_steps": save, "keep_last_n_checkpoints": 2,
    }


def build_data_config(run_dir: str, *, has_val: bool) -> dict:
    cfg: dict[str, Any] = {"train": [{"manifest_path": [os.path.join(run_dir, "tokens", "train", "data.lst")]}]}
    if has_val:
        cfg["dev"] = [{"manifest_path": [os.path.join(run_dir, "tokens", "dev", "data.lst")]}]
    return cfg


def build_commands(run_dir: str, opts: TrainOptions, *, train_jsonl: str, val_jsonl: Optional[str]) -> dict:
    py = opts.python or sys.executable

    def tokenize(split: str, jsonl: str) -> list[str]:
        base = os.path.join(run_dir, "tokens", split)
        return [py, "-m", "omnivoice.scripts.extract_audio_tokens",
                "--input_jsonl", jsonl,
                "--tar_output_pattern", os.path.join(base, "audios", "shard-%06d.tar"),
                "--jsonl_output_pattern", os.path.join(base, "txts", "shard-%06d.jsonl"),
                "--tokenizer_path", opts.tokenizer_path,
                "--nj_per_gpu", "1", "--loader_workers", "2", "--min_num_shards", "2",
                "--samples_per_shard", "500", "--min_length", "0.5", "--max_length", "30", "--skip_errors"]

    cmds = {"tokenize_train": tokenize("train", train_jsonl)}
    if val_jsonl:
        cmds["tokenize_dev"] = tokenize("dev", val_jsonl)
    cmds["train"] = [py, "-m", "accelerate.commands.launch", "--num_processes", "1", "--num_machines", "1",
                     "-m", "omnivoice.cli.train",
                     "--train_config", os.path.join(run_dir, "train_config.json"),
                     "--data_config", os.path.join(run_dir, "data_config.json"),
                     "--output_dir", os.path.join(run_dir, "output")]
    return cmds


def child_env(opts: TrainOptions) -> dict:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["TQDM_MININTERVAL"] = "5"
    if not opts.allow_download:
        env["HF_HUB_OFFLINE"] = "1"
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    if os.path.isdir(os.path.join(root, "omnivoice")):  # running from a source checkout
        env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    return env


# ── hardware ──
PROBE_CODE = r"""
import importlib.util, json
out = {m: importlib.util.find_spec(m) is not None for m in ("accelerate", "triton", "soundfile", "safetensors", "webdataset", "tensorboard", "tensorboardX")}
try:
    import torch
    out["torch"] = torch.__version__
    out["cuda"] = bool(torch.cuda.is_available())
    out["mps"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    if out["cuda"]:
        props = torch.cuda.get_device_properties(0)
        out["device"], out["vram_gb"] = props.name, round(props.total_memory / 2**30, 2)
        out["hip"] = bool(getattr(torch.version, "hip", None))
        out["bf16"] = bool(torch.cuda.is_bf16_supported())
except Exception as exc:
    out["torch_error"] = str(exc)[:300]
try:
    import omnivoice.training.lora, omnivoice.models.omnivoice
    out["omnivoice"] = True
except Exception as exc:
    out["omnivoice"] = False
    out["omnivoice_error"] = str(exc)[:300]
print("PROBE_JSON " + json.dumps(out))
"""


def assess_hardware(probe: dict, *, system: Optional[str] = None) -> dict:
    """Can the training interpreter really train? ``blockers`` stop a run; ``warnings`` do not."""
    system = system or sys.platform
    blockers: list[str] = []
    warnings: list[str] = []
    if not probe or probe.get("error"):
        blockers.append("The training Python could not be started: " + str((probe or {}).get("error") or "no answer"))
        return {"ok": False, "device": "unknown", "blockers": blockers, "warnings": warnings, "probe": probe or {}}
    if probe.get("torch_error") or "torch" not in probe:
        blockers.append("PyTorch cannot be imported by the training Python: " + str(probe.get("torch_error") or "not installed"))
    elif probe.get("cuda"):
        if probe.get("hip"):
            warnings.append("AMD ROCm training is untested.")
        vram = float(probe.get("vram_gb") or 0)
        if vram < MIN_VRAM_GB_BLOCK:
            blockers.append(f"The GPU has {vram:.1f} GB of memory; training needs about {VRAM_GB_COMFORT:.0f} GB.")
        elif vram < VRAM_GB_COMFORT:
            warnings.append(f"The GPU has {vram:.1f} GB; training is tight. Lower the batch size if it runs out of memory.")
        if probe.get("bf16") is False:
            warnings.append("This GPU lacks bf16; training may be slow or unstable.")
    elif probe.get("mps"):
        blockers.append("Apple Silicon (MPS) cannot run this trainer yet; it needs an NVIDIA GPU. "
                        "Train on a CUDA machine and copy the adapter back.")
    else:
        blockers.append("No GPU was found. Training needs an NVIDIA GPU (CUDA); a CPU run would take days.")
    if not probe.get("omnivoice"):
        blockers.append("The training Python cannot import OmniVoice: " + str(probe.get("omnivoice_error") or "not installed"))
    missing = [m for m in ("accelerate", "webdataset", "safetensors") if not probe.get(m)]
    if not (probe.get("tensorboard") or probe.get("tensorboardX")):
        missing.append("tensorboard")
    if missing:
        blockers.append("Missing from the training Python: " + ", ".join(missing) + ".")
    if not probe.get("triton"):
        hint = ("Install 'triton-windows' into the training Python, or train under WSL/Linux."
                if system.startswith("win") else "It normally ships with CUDA PyTorch; reinstall the CUDA build.")
        blockers.append("Triton is missing, which OmniVoice's flex attention needs. " + hint)
    return {"ok": not blockers, "device": probe.get("device") or ("mps" if probe.get("mps") else "cpu"),
            "vram_gb": probe.get("vram_gb"), "blockers": blockers, "warnings": warnings, "probe": probe}


# ── launching children (replaceable in tests) ──
class Handle(Protocol):
    def poll(self) -> Optional[int]: ...
    def terminate(self) -> None: ...
    def kill(self) -> None: ...
    def wait(self, timeout: Optional[float] = None) -> int: ...


class Launcher(Protocol):
    def probe(self, python: str, env: dict) -> dict: ...
    def spawn(self, argv: list[str], *, cwd: str, env: dict, log_path: str) -> Handle: ...


class SubprocessLauncher:
    """Real children: owned processes that are killed with the app (``core.contained_subprocess``)."""

    def probe(self, python: str, env: dict) -> dict:
        try:
            # Not -I: the child must see PYTHONPATH, which is how a source checkout exposes omnivoice.
            done = subprocess.run([python, "-c", PROBE_CODE], capture_output=True, text=True, timeout=120,
                                  encoding="utf-8", errors="replace", env=env, cwd=training_root())
        except (OSError, subprocess.SubprocessError) as exc:
            return {"error": type(exc).__name__}
        for line in done.stdout.splitlines():
            if line.startswith("PROBE_JSON "):
                try:
                    return json.loads(line[len("PROBE_JSON "):])
                except ValueError:
                    break
        return {"error": (done.stderr or done.stdout or "no output").strip()[-300:]}

    def spawn(self, argv: list[str], *, cwd: str, env: dict, log_path: str) -> Handle:
        from core.contained_subprocess import spawn_owned

        log = open(log_path, "ab")
        try:
            return spawn_owned(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        finally:
            log.close()  # the child holds its own copy of the handle


_launcher: Launcher = SubprocessLauncher()


def set_launcher(launcher: Optional[Launcher]) -> None:
    """Swap the process launcher (tests). ``None`` restores the real one."""
    global _launcher
    _launcher = launcher or SubprocessLauncher()


# ── job records ──
def _view(job: dict) -> dict:
    with job["lock"]:
        return {k: (list(v) if isinstance(v, list) else (dict(v) if isinstance(v, dict) else v))
                for k, v in job.items() if k not in {"lock", "cancel", "handle", "io"}}


def _persist(job: dict) -> None:
    data = _view(job)
    with job["io"]:
        try:
            _write_json(os.path.join(job["run_dir"], JOB_FILE), data)
        except OSError as exc:
            logger.warning("Fish training: could not save job record (%s)", type(exc).__name__)


def _log(job: dict, level: str, message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {level:<5} {message}\n"
    with job["io"]:
        try:
            with open(os.path.join(job["run_dir"], LOG_FILE), "a", encoding="utf-8") as fh:
                fh.write(line)
        except OSError:
            pass


def _set(job: dict, **fields: Any) -> None:
    with job["lock"]:
        job.update(fields)
    _persist(job)


def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _new_job(run_dir: str, job_id: str, name: str, export_dir: str, opts: TrainOptions, *, dry_run: bool,
             created: Optional[float] = None) -> dict:
    return {"id": job_id, "name": name, "status": ACTIVE, "phase": "preparing", "message": "Starting.",
            "created_at": created or time.time(), "started_at": time.time(), "finished_at": None,
            "run_dir": run_dir, "export_dir": export_dir, "options": dataclasses.asdict(opts),
            "dry_run": dry_run, "step": 0, "steps": opts.steps, "loss": None, "eval_loss": None,
            "resumes": 0, "adapter": None, "evaluation": None, "blockers": [], "warnings": [],
            "lock": threading.RLock(), "io": threading.Lock(), "cancel": threading.Event(), "handle": None}


def _find_run_dir(job_id: str) -> Optional[str]:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", job_id or ""):
        return None
    root = training_root()
    for name in os.listdir(root):
        if name.endswith("_" + job_id) and os.path.isfile(os.path.join(root, name, JOB_FILE)):
            return os.path.join(root, name)
    return None


def _summary(data: dict) -> dict:
    keys = ("id", "name", "status", "phase", "message", "created_at", "started_at", "finished_at", "step", "steps",
            "loss", "eval_loss", "dry_run", "resumes", "adapter", "run_dir", "export_dir", "blockers", "warnings")
    row = {k: data.get(k) for k in keys}
    row["ship_ready"] = bool((data.get("evaluation") or {}).get("ship_ready"))
    row["verdict"] = (data.get("evaluation") or {}).get("verdict")
    return row


def get_run(job_id: str) -> Optional[dict]:
    with _jobs_lock:
        live = _jobs.get(job_id)
    if live:
        return _view(live)
    run_dir = _find_run_dir(job_id)
    data = _read_json(os.path.join(run_dir, JOB_FILE)) if run_dir else None
    if data and data.get("status") == ACTIVE:
        # No live thread owns it: the app closed mid-run. Resume continues from the last checkpoint.
        data.update(status="interrupted", message="The app closed before this run finished.",
                    finished_at=data.get("finished_at") or time.time())
        try:
            _write_json(os.path.join(run_dir, JOB_FILE), data)
        except OSError:
            pass
    return data


def list_runs(limit: int = 100) -> list[dict]:
    rows: dict[str, dict] = {}
    root = training_root()
    for name in os.listdir(root):
        data = _read_json(os.path.join(root, name, JOB_FILE))
        if data and data.get("id"):
            rows[data["id"]] = data
    with _jobs_lock:
        live = {k: _view(v) for k, v in _jobs.items()}
    rows.update(live)
    out = []
    for data in rows.values():
        row = _summary(data)
        if row["status"] == ACTIVE and data["id"] not in live:
            row.update(status="interrupted", message="The app closed before this run finished.")
        out.append(row)
    out.sort(key=lambda r: r.get("created_at") or 0, reverse=True)
    return out[:limit]


def read_log(job_id: str, max_lines: int = 400) -> Optional[str]:
    run_dir = _find_run_dir(job_id)
    if not run_dir:
        return None
    try:
        with open(os.path.join(run_dir, LOG_FILE), encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return ""
    return "\n".join(line.rstrip("\r")[:600] for line in lines[-max_lines:])


def cancel_run(job_id: str) -> bool:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job or job["status"] != ACTIVE:
        return False
    job["cancel"].set()
    _log(job, "INFO", "Cancel requested; stopping the training process.")
    return True


# ── starting ──
def start_run(export_dir: str, options: Optional[dict] = None, *, name: str = "", dry_run: bool = False) -> dict:
    """Validate and start a run (background), or do a dry run (synchronous, nothing heavy).

    Raises ``ValueError`` for a bad export/option and ``PermissionError`` while training is off.
    """
    if not dry_run and not is_enabled():
        raise PermissionError("Training is switched off. Turn it on in the Train dialog first.")
    opts = TrainOptions.from_dict(options)
    check = validate_export(export_dir)
    if not check["ok"]:
        raise ValueError(" ".join(check["errors"]))
    if not dry_run:
        with _jobs_lock:
            if any(j["status"] == ACTIVE and not j["dry_run"] for j in _jobs.values()):
                raise ValueError("Another training run is still going. Wait for it or cancel it first.")
    job_id = uuid.uuid4().hex[:12]
    folder = os.path.join(training_root(), f"run_{time.strftime('%Y%m%d-%H%M%S')}_{job_id}")
    os.makedirs(folder, exist_ok=True)
    label = (name or os.path.basename(os.path.normpath(export_dir)) or "run").strip()[:60]
    job = _new_job(folder, job_id, label, os.path.normpath(export_dir), opts, dry_run=dry_run)
    job["warnings"] = list(check["warnings"])
    job["export_summary"] = {k: check[k] for k in ("train_clips", "val_clips", "seconds", "speakers", "loudness_matched")}
    _register(job)
    _log(job, "INFO", f"{'Dry run' if dry_run else 'Run'} created for {job['export_dir']}: "
                      f"{check['train_clips']} training / {check['val_clips']} validation clips.")
    _persist(job)
    if dry_run:
        _run(job, opts, check)
        return _view(job)
    threading.Thread(target=_run, args=(job, opts, check), name=f"fish-train-{job_id}", daemon=True).start()
    return _view(job)


def resume_run(job_id: str) -> dict:
    """Continue a failed, cancelled or interrupted run from its last complete checkpoint."""
    if not is_enabled():
        raise PermissionError("Training is switched off. Turn it on in the Train dialog first.")
    with _jobs_lock:
        live = _jobs.get(job_id)
        if live and live["status"] == ACTIVE:
            raise ValueError("That run is still going.")
        if any(j["status"] == ACTIVE and not j["dry_run"] for j in _jobs.values()):
            raise ValueError("Another training run is still going. Wait for it or cancel it first.")
    run_dir = _find_run_dir(job_id)
    data = _read_json(os.path.join(run_dir, JOB_FILE)) if run_dir else None
    if not data:
        raise ValueError("That run's record is missing, so it cannot be resumed.")
    if data.get("dry_run"):
        raise ValueError("A dry run has nothing to resume. Start a real run instead.")
    if data.get("status") not in RESUMABLE | {ACTIVE}:
        raise ValueError(f"A {data.get('status')} run cannot be resumed.")
    opts = TrainOptions.from_dict(data["options"])
    check = validate_export(data["export_dir"])
    if not check["ok"]:
        raise ValueError("The export this run used has changed or is gone: " + " ".join(check["errors"]))
    job = _new_job(run_dir, job_id, data.get("name") or "run", data["export_dir"], opts, dry_run=False,
                   created=data.get("created_at"))
    job["resumes"] = int(data.get("resumes") or 0) + 1
    job["warnings"] = list(check["warnings"])
    job["export_summary"] = data.get("export_summary")
    _register(job)
    _log(job, "INFO", f"Resuming (attempt {job['resumes'] + 1}).")
    _persist(job)
    threading.Thread(target=_run, args=(job, opts, check), name=f"fish-train-{job_id}", daemon=True).start()
    return _view(job)


def _register(job: dict) -> None:
    with _jobs_lock:
        for old in [j for j, v in _jobs.items() if v["status"] != ACTIVE][: max(0, len(_jobs) - 20)]:
            _jobs.pop(old, None)
        _jobs[job["id"]] = job


# ── the run ──
_STEP_RE = re.compile(r"Step (\d+) \|(.*)")
_LOSS_RE = re.compile(r"train/loss: ([0-9.eE+-]+)")
_EVAL_RE = re.compile(r"Eval Loss: ([0-9.eE+-]+)")


def latest_checkpoint(output_dir: str) -> Optional[str]:
    """Newest *complete* checkpoint (the trainer writes ``train_config.json`` into it last)."""
    best, best_step = None, -1
    try:
        names = os.listdir(output_dir)
    except OSError:
        return None
    for name in names:
        m = re.fullmatch(r"checkpoint-(\d+)", name)
        path = os.path.join(output_dir, name)
        if m and int(m.group(1)) > best_step and os.path.isfile(os.path.join(path, "train_config.json")):
            best, best_step = path, int(m.group(1))
    return best


def _follow(job: dict, offset: int) -> int:
    """Read what the child appended to run.log since ``offset`` and update the job's progress."""
    path = os.path.join(job["run_dir"], LOG_FILE)
    try:
        with open(path, "rb") as fh:
            fh.seek(offset)
            chunk = fh.read()
    except OSError:
        return offset
    text = chunk.decode("utf-8", errors="replace")
    update: dict[str, Any] = {}
    for line in text.replace("\r", "\n").splitlines():
        m = _STEP_RE.search(line)
        if m:
            update["step"] = int(m.group(1))
            lm = _LOSS_RE.search(m.group(2))
            if lm:
                update["loss"] = float(lm.group(1))
        em = _EVAL_RE.search(line)
        if em:
            update["eval_loss"] = float(em.group(1))
    if update:
        with job["lock"]:
            job.update(update)
        _persist(job)
    return offset + len(chunk)


def _supervise(job: dict, argv: list[str], opts: TrainOptions, what: str) -> int:
    """Run one child to the end, following its output; kill it if cancelled. Returns its exit code."""
    _log(job, "INFO", f"{what}: {' '.join(argv)}")
    log_path = os.path.join(job["run_dir"], LOG_FILE)
    offset = os.path.getsize(log_path) if os.path.exists(log_path) else 0
    cwd = job["run_dir"]
    handle = _launcher.spawn(argv, cwd=cwd, env=child_env(opts), log_path=log_path)
    job["handle"] = handle
    try:
        while handle.poll() is None:
            if job["cancel"].wait(1.0):
                _log(job, "INFO", f"Stopping {what}.")
                handle.terminate()
                try:
                    handle.wait(timeout=15)
                except (subprocess.TimeoutExpired, OSError):
                    handle.kill()
                break
            offset = _follow(job, offset)
        _follow(job, offset)
        code = handle.poll()
        return -1 if code is None else code
    finally:
        job["handle"] = None


def _release_gpu(job: dict) -> None:
    """Free the resident TTS model so the trainer gets the whole card. Best effort."""
    try:
        from services import model_manager, subprocess_backend

        model_manager.unload_shared_model()
        subprocess_backend.unload_all_sidecars()
        _log(job, "INFO", "Unloaded the resident speech models to free GPU memory.")
    except Exception as exc:  # noqa: BLE001 — nothing loaded, or the stack is not importable
        _log(job, "INFO", f"Could not unload resident models ({type(exc).__name__}); continuing.")


def _finish(job: dict, status: str, message: str, **extra: Any) -> None:
    _set(job, status=status, message=message, finished_at=time.time(), **extra)
    _log(job, "INFO" if status in {"done", "dry_run", "cancelled"} else "ERROR", message)


def _run(job: dict, opts: TrainOptions, check: dict) -> None:
    run_dir = job["run_dir"]
    try:
        has_val = bool(check["val_jsonl"])
        output = os.path.join(run_dir, "output")
        os.makedirs(output, exist_ok=True)
        resume_from = latest_checkpoint(output) if job["resumes"] else None
        cmds = build_commands(run_dir, opts, train_jsonl=check["train_jsonl"], val_jsonl=check["val_jsonl"])
        train_cfg = build_train_config(opts, has_val=has_val, resume_from=resume_from)
        _write_json(os.path.join(run_dir, "train_config.json"), train_cfg)
        _write_json(os.path.join(run_dir, "data_config.json"), build_data_config(run_dir, has_val=has_val))
        heldout = training_eval.select_heldout(check["val_jsonl"])
        training_eval.write_heldout(os.path.join(run_dir, training_eval.HELDOUT_FILE), heldout)
        if resume_from:
            _log(job, "INFO", f"Resuming from {os.path.basename(resume_from)}.")

        _set(job, phase="checking", message="Checking the training Python and the GPU.")
        hardware = assess_hardware(_launcher.probe(opts.python or sys.executable, child_env(opts)))
        job["hardware"] = {k: hardware[k] for k in ("ok", "device", "blockers", "warnings")}
        job["blockers"], job["warnings"] = hardware["blockers"], list(job["warnings"]) + hardware["warnings"]
        if not opts.allow_download:
            job["warnings"].append("Offline: the base model and tokenizer must already be on disk or in the "
                                   "Hugging Face cache. Allow downloads to fetch them.")
        if job["dry_run"]:
            report = {"export": {k: check[k] for k in ("train_clips", "val_clips", "seconds", "speakers",
                                                         "loudness_matched", "warnings")},
                      "hardware": hardware, "commands": cmds, "train_config": train_cfg,
                      "heldout_clips": len(heldout), "ready": hardware["ok"]}
            _write_json(os.path.join(run_dir, DRY_FILE), report)
            _finish(job, "dry_run", "Everything checks out; a real run would start." if hardware["ok"]
                    else "The data and settings are valid, but this machine cannot train yet: " + hardware["blockers"][0],
                    ready=hardware["ok"])
            return
        if not hardware["ok"]:
            _finish(job, "failed", hardware["blockers"][0])
            return
        _release_gpu(job)

        for split, key in (("train", "tokenize_train"), ("dev", "tokenize_dev")):
            if key not in cmds:
                continue
            lst = os.path.join(run_dir, "tokens", split, "data.lst")
            if os.path.isfile(lst) and os.path.getsize(lst) > 0:
                _log(job, "INFO", f"Tokens for '{split}' already exist; skipping.")
                continue
            _set(job, phase="tokenizing", message=f"Converting the {split} audio to tokens.")
            code = _supervise(job, cmds[key], opts, f"tokenize {split}")
            if job["cancel"].is_set():
                return _finish(job, "cancelled", "Cancelled while tokenizing.")
            if code != 0 or not os.path.isfile(lst):
                return _finish(job, "failed", f"Tokenizing the {split} audio failed (exit code {code}). See the log.")

        _set(job, phase="training", message="Training the adapter.")
        code = _supervise(job, cmds["train"], opts, "train")
        if job["cancel"].is_set():
            return _finish(job, "cancelled", f"Cancelled at step {job['step']}. Resume continues from the last checkpoint.")
        if code != 0:
            return _finish(job, "failed", f"Training stopped (exit code {code}). See the log; Resume continues from the last checkpoint.")

        _set(job, phase="finalizing", message="Saving the adapter and running the evaluation gate.")
        adapter, evaluation = finalize(job, opts, heldout_clips=len(heldout))
        _finish(job, "done", "Adapter ready. " + ("It passed every check." if evaluation["ship_ready"] else
                f"Evaluation is {evaluation['verdict']}: it is not marked ready to use yet."),
                adapter=adapter, evaluation=evaluation)
    except Exception as exc:  # noqa: BLE001 — the record must say what happened, whatever it was
        logger.warning("Fish training run failed", exc_info=True)
        _finish(job, "failed", f"The run failed ({type(exc).__name__}): {str(exc)[:200]}")


def finalize(job: dict, opts: TrainOptions, *, heldout_clips: int) -> tuple[str, dict]:
    """Copy the last checkpoint's adapter out of ``output/`` and store its evaluation record beside it."""
    run_dir = job["run_dir"]
    ckpt = latest_checkpoint(os.path.join(run_dir, "output"))
    src = os.path.join(ckpt, "adapter") if ckpt else None
    if not src or not os.path.isfile(os.path.join(src, "adapter_config.json")):
        raise RuntimeError("Training finished but wrote no adapter checkpoint.")
    dest = os.path.join(run_dir, "adapter")
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    history = training_eval.read_eval_history(os.path.join(run_dir, "output", "eval_history.jsonl"))
    record = training_eval.build_record(
        [training_eval.loss_check(history)], heldout=heldout_clips,
        extra={"checkpoint": os.path.basename(ckpt), "eval_history": history})
    training_eval.save_record(dest, record)
    _write_json(os.path.join(dest, "adapter.json"), {
        "run_id": job["id"], "name": job["name"], "trained_at": time.time(), "base_checkpoint": opts.base_checkpoint,
        "target": "omnivoice", "options": dataclasses.asdict(opts), "export_dir": job["export_dir"],
        "export_summary": job.get("export_summary"), "ship_ready": record["ship_ready"], "verdict": record["verdict"]})
    return dest, record


__all__ = ["TrainOptions", "defaults", "is_enabled", "set_enabled", "validate_export", "start_run", "resume_run",
           "cancel_run", "get_run", "list_runs", "read_log", "set_launcher", "assess_hardware"]
