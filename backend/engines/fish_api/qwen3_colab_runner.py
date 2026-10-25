#!/usr/bin/env python3
"""VoiceStudio: set up and run Qwen3-TTS fine-tuning on Google Colab, with everything reusable kept on Google Drive.

This file is copied unchanged into every Qwen3-TTS export (as ``voicestudio_qwen3_colab.py``). It is the single
implementation behind the notebook, ``colab_setup.sh`` and the ``colab`` command-line route. Standard library only,
so it runs on a bare Colab runtime; the packages it installs come from pip on the runtime.

Commands follow QwenLM/Qwen3-TTS ``finetuning/`` at the pinned commit below. None of this has been run on Colab or on
a GPU by VoiceStudio; every step prints what it does so a first run is easy to read.

    python3 voicestudio_qwen3_colab.py all --data-source /content/drive/MyDrive/VoiceStudio-training/datasets/x.zip
    python3 voicestudio_qwen3_colab.py {gpu,setup,data,train,results,smoke,all} --help
"""
from __future__ import annotations

import argparse
import dataclasses
import contextlib
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
import zipfile
from typing import Callable, Optional

DISCONNECT_CELL = "from google.colab import runtime; runtime.unassign()"

RUNNER_VERSION = "6"
SESSION = "qwen-train"  # the session name used in the colab command-line examples
QWEN_REPO = "https://github.com/QwenLM/Qwen3-TTS.git"
QWEN_COMMIT = "022e286b98fbec7e1e916cb940cdf532cd9f488e"
QWEN_COMMIT_DATE = "2026-03-17"
READ_ON = "2026-10-24"
BASE_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
TOKENIZER = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
SAMPLE_RATE = 24000
DEFAULT_DRIVE_FOLDER = "VoiceStudio-training"
SPEAKER_RE = re.compile(r"[A-Za-z0-9_-]{1,40}")
FOLDER_RE = re.compile(r"[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*")
MODEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")
CHECKPOINT_RE = re.compile(r"checkpoint-epoch-(\d+)")
DRIVE_MOUNT = "/content/drive"
SCRIPT_NAME = "voicestudio_qwen3_colab.py"
OWNED_MANIFEST = ".voicestudio_checkpoints.json"
DONE_MARKER = ".voicestudio_complete"
SOURCE_MARKER = ".voicestudio_source"

# FlashAttention release read on READ_ON from the project's releases page (wheel names via the GitHub API listing).
FLASH_TAG = "v2.8.3.post2"
FLASH_URL = "https://github.com/Dao-AILab/flash-attention/releases/download/" + FLASH_TAG + "/{name}"
FLASH_VERSION = "2.8.3.post2"
FLASH_PREBUILT = {  # torch major.minor -> python tags with a prebuilt Linux x86_64 wheel (cu12 only, both C++ ABIs)
    "2.4": ("cp39", "cp310", "cp311", "cp312"),
    "2.5": ("cp39", "cp310", "cp311", "cp312", "cp313"),
    "2.6": ("cp39", "cp310", "cp311", "cp312", "cp313"),
    "2.7": ("cp39", "cp310", "cp311", "cp312", "cp313"),
    "2.8": ("cp39", "cp310", "cp311", "cp312", "cp313"),
}
ATTN_FALLBACK_IMPL = "sdpa"
FLASH_LINE = 'attn_implementation="flash_attention_2"'  # sft_12hz.py line 51 at the pinned commit

# Packages installed on top of Colab's torch (pyproject.toml of the pinned commit: qwen-tts 0.1.1 pins transformers
# 4.57.3 and accelerate 1.12.0 and lists gradio, librosa, torchaudio, soundfile, sox, onnxruntime, einops), plus what
# the finetuning scripts import directly.
EXTRA_PACKAGES = ["tensorboard", "safetensors", "huggingface_hub", "librosa", "soundfile"]
CHECK_MODULES = ["qwen_tts", "transformers", "accelerate", "torch", "torchaudio", "librosa", "soundfile", "safetensors",
                 "huggingface_hub", "tensorboard"]

# Batch size by GPU memory. Educated guesses (full fine-tune of a 1.7 B model in bf16 with AdamW needs roughly 14 GB
# before activations): NOT measured. The owner's batch size is never changed silently; "auto" picks this one.
BATCH_BY_VRAM = [(12, 0), (20, 1), (30, 1), (60, 2), (10_000, 4)]  # (upper bound in GB, batch size); 0 = too small
# i.e. under 12 GB: too small; T4 (15 GB) 1; L4 (24 GB) 1; A100 40 GB 2; A100 80 GB and H100 80 GB 4. Conservative and UNTESTED.
SMOKE_MIN_BATCHES = 21  # the official script prints "Step N" every 10 batches; 21 batches give steps 0, 10 and 20
CC_BY_NAME = [("H100", (9, 0)), ("H200", (9, 0)), ("A100", (8, 0)), ("L4", (8, 9)), ("A10", (8, 6)), ("T4", (7, 5)),
              ("V100", (7, 0)), ("P100", (6, 0)), ("K80", (3, 7)), ("RTX 40", (8, 9)), ("RTX 30", (8, 6))]
# Compute-unit rates supplied by the owner (NOT verified by this project; Colab can change them). (name hint, min GB, max GB, units/hour)
UNIT_RATES = [("RTX PRO 6000", 60, 10_000, 8.7, "G4 (RTX PRO 6000 Blackwell, 96 GB)"), ("A100", 60, 10_000, 7.5, "A100 80 GB"),
              ("A100", 0, 60, 5.4, "A100 40 GB")]
CENTS_PER_UNIT = 10  # owner's rough figure
MODEL_NEEDS_GB = 12  # local copy of the base model + tokenizer + headroom (a few GB; not measured)


class StepFailed(RuntimeError):
    """A step stopped on purpose, with a message that says what to do."""


def say(msg: str = "") -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------------------------- parameters
OFFICIAL_DEFAULTS = {"epochs": 3, "lr": 2e-5, "batch_size": 2, "base_model": BASE_MODEL, "tokenizer_model": TOKENIZER,
                     "speaker_name": "speaker_test"}  # the argparse defaults of sft_12hz.py / prepare_data.py at the pinned commit
RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,59}")
DEFAULT_SAMPLE_TEXT = "The lighthouse keeper counted seven ships before the fog rolled in."


@dataclasses.dataclass
class Params:
    drive_root: str = f"{DRIVE_MOUNT}/MyDrive/{DEFAULT_DRIVE_FOLDER}"
    data_source: str = ""  # a folder or .zip anywhere on Drive, or an unzipped folder under /content
    speaker_name: str = "speaker"
    epochs: int = 3
    lr: float = 2e-5
    batch_size: object = 2  # an int, or "auto"
    base_model: str = BASE_MODEL
    tokenizer_model: str = TOKENIZER
    run_id: str = ""  # empty: a new timestamp. Re-using an id is explicit (and refused over existing checkpoints unless restart)
    keep_all: bool = True  # default: never delete a checkpoint
    keep_last_n: int = 0  # only with keep_all False: keep the newest N epochs
    keep_first: bool = False  # only with keep_all False: also keep the first epoch
    keep_final: bool = True  # only with keep_all False: also keep the final epoch
    attn_fallback: object = "auto"  # "auto": FlashAttention 2 on Ampere/Ada/Hopper, SDPA on Blackwell; True: force SDPA; False: force FA2
    flash_wheel: str = ""  # optional: a flash_attn .whl on Drive (full path) or a URL, tried before anything automatic
    use_venv: object = "auto"  # "auto": build an isolated Python 3.12 environment when this runtime has no FlashAttention package; True: always; False: never
    build_flash_attn: bool = False  # True: compile FlashAttention from source when no ready-made package exists (an hour or more, may fail)
    restart: bool = False
    local_root: str = "/content"
    cache_models_on_drive: bool = True  # False: download models straight to /content (nothing cached on Drive)
    sample_after_training: bool = True
    sample_text: str = DEFAULT_SAMPLE_TEXT
    sample_seed: int = 1234
    heartbeat_seconds: int = 180
    smoke_run: bool = False  # timing run: one epoch on the first smoke_clips clips, scratch folder
    smoke_clips: int = 40
    units_per_hour: object = None  # compute units per hour of this runtime; looked up for known GPUs when empty
    units_available: float = 100.0
    compare_epoch_seconds: object = None  # seconds per epoch measured on a DIFFERENT GPU, to compare cost per epoch
    compare_units_per_hour: object = None  # that other GPU's rate
    sources: dict = dataclasses.field(default_factory=dict)  # field -> "default" | "export" | "override"

    def validate(self) -> "Params":
        try:
            return self._validate()
        except (ValueError, TypeError) as exc:
            raise StepFailed(f"A parameter has an invalid value ({exc}). Check the numbers and true/false settings.") from exc

    def _validate(self) -> "Params":
        if not SPEAKER_RE.fullmatch(self.speaker_name or ""):
            raise StepFailed("SPEAKER_NAME may use letters, digits, '_' and '-' only (1-40 characters).")
        for key in ("base_model", "tokenizer_model"):
            if not MODEL_RE.fullmatch(getattr(self, key) or ""):
                raise StepFailed(f"{key.upper()} must be a Hugging Face model id such as {BASE_MODEL}.")
        self.run_id = (self.run_id or "").strip() or time.strftime("%Y%m%d-%H%M%S")
        if not RUN_ID_RE.fullmatch(self.run_id):
            raise StepFailed("RUN_ID may use letters, digits, '.', '_' and '-' only (up to 60 characters).")
        self.epochs, self.keep_last_n = int(self.epochs), int(self.keep_last_n or 0)
        self.lr = float(self.lr)
        if not 1 <= self.epochs <= 100 or not 1e-7 <= self.lr <= 1e-3 or self.keep_last_n < 0:
            raise StepFailed("EPOCHS must be 1-100, LR 1e-7 to 1e-3 and KEEP_LAST_N 0 or more.")
        if str(self.batch_size).lower() == "auto":
            self.batch_size = "auto"
        else:
            self.batch_size = int(self.batch_size)
            if not 1 <= self.batch_size <= 256:
                raise StepFailed("BATCH_SIZE must be 1-256 or 'auto'.")
        self.attn_fallback = "auto" if str(self.attn_fallback).strip().lower() in ("auto", "") else _truthy(self.attn_fallback)
        self.flash_wheel = str(self.flash_wheel or "").strip()
        self.use_venv = "auto" if str(self.use_venv).strip().lower() in ("auto", "") else _truthy(self.use_venv)
        for key in ("restart", "build_flash_attn", "smoke_run", "keep_all", "keep_first", "keep_final", "cache_models_on_drive", "sample_after_training"):
            setattr(self, key, _truthy(getattr(self, key)))
        if not self.keep_all and not (self.keep_last_n or self.keep_first or self.keep_final):
            raise StepFailed("KEEP_ALL is off but KEEP_LAST_N, KEEP_FIRST and KEEP_FINAL keep nothing: that would delete every checkpoint.")
        self.smoke_clips, self.sample_seed, self.heartbeat_seconds = int(self.smoke_clips), int(self.sample_seed), int(self.heartbeat_seconds)
        if not 1 <= self.smoke_clips <= 100_000:
            raise StepFailed("SMOKE_CLIPS must be a positive number.")
        if not (self.sample_text or "").strip():
            raise StepFailed("SAMPLE_TEXT must not be empty.")
        for key in ("units_per_hour", "compare_epoch_seconds", "compare_units_per_hour"):
            v = getattr(self, key)
            setattr(self, key, None if v in (None, "") else float(v))
            if getattr(self, key) is not None and getattr(self, key) <= 0:
                raise StepFailed(f"{key.upper()} must be a positive number or empty.")
        self.units_available = float(self.units_available)
        if not self.drive_root or not _is_abs(self.drive_root) or self.drive_root.replace("\\", "/").rstrip("/") in ("", "."):
            raise StepFailed("DRIVE_ROOT must be a full folder path on Drive, e.g. /content/drive/MyDrive/VoiceStudio-training.")
        return self

    def provenance(self) -> dict:
        """Where each resolved setting came from: official script default, our default, the export dialog, or an owner override."""
        out = {}
        for key in ("epochs", "lr", "batch_size", "base_model", "tokenizer_model", "speaker_name", "attn_fallback", "keep_all", "keep_last_n",
                    "keep_first", "keep_final", "sample_text", "sample_seed", "cache_models_on_drive", "restart", "run_id", "drive_root"):
            value, src = getattr(self, key), self.sources.get(key, "default")
            if key in OFFICIAL_DEFAULTS:
                same = value == OFFICIAL_DEFAULTS[key]
                label = ("official default" + (" (set explicitly)" if src == "override" else "")) if same else \
                    ("set in the VoiceStudio export dialog" if src == "export" else "owner override" if src == "override" else "our default")
            else:
                label = "owner override" if src == "override" else "set in the VoiceStudio export dialog" if src == "export" else "our default"
            out[key] = {"value": value, "source": label}
        return out


def _is_abs(path: str) -> bool:
    return path.startswith("/") or os.path.isabs(path)  # Colab is Linux; the tests also run this on Windows


def _truthy(v) -> bool:
    return v if isinstance(v, bool) else str(v).strip().lower() in ("1", "true", "yes", "y", "on")


_ENV_FIELDS = ("drive_root", "data_source", "speaker_name", "epochs", "lr", "batch_size", "base_model", "tokenizer_model", "run_id",
               "keep_all", "keep_last_n", "keep_first", "keep_final", "attn_fallback", "use_venv", "build_flash_attn", "flash_wheel", "restart", "cache_models_on_drive", "sample_after_training",
               "sample_text", "sample_seed", "heartbeat_seconds", "smoke_run", "smoke_clips", "units_per_hour", "units_available",
               "compare_epoch_seconds", "compare_units_per_hour")


def params_from(overrides: Optional[dict] = None, settings_path: Optional[str] = None, env: Optional[dict] = None) -> Params:
    """Precedence: explicit overrides, then environment variables, then training_settings.json, then defaults."""
    env = os.environ if env is None else env
    settings = {}
    if settings_path and os.path.isfile(settings_path):
        with open(settings_path, encoding="utf-8") as fh:
            settings = json.load(fh)
    p = Params()
    if settings:
        folder = settings.get("drive_folder") or DEFAULT_DRIVE_FOLDER
        p.drive_root = f"{DRIVE_MOUNT}/MyDrive/{folder}"
        p.sources["drive_root"] = "export"
        for key, field in (("speaker_name", "speaker_name"), ("epochs", "epochs"), ("learning_rate", "lr"),
                           ("batch_size", "batch_size"), ("base_model", "base_model"), ("tokenizer_model", "tokenizer_model")):
            if settings.get(key) is not None:
                setattr(p, field, settings[key])
                p.sources[field] = "export"
    for name in _ENV_FIELDS:
        if env.get(name.upper()) not in (None, ""):
            setattr(p, name, env[name.upper()])
            p.sources[name] = "override"
    for key, value in (overrides or {}).items():
        if value is not None:
            setattr(p, key, value)
            p.sources[key] = "override"
    return p.validate()


class Layout:
    """Folder layout under the Drive root. Created on demand, never cleaned up by this script.

    Drive:   cache/{models,Qwen3-TTS,wheelhouse/sm<arch>,pip}  datasets/  prepared/<speaker>/  logs/
             runs/<speaker>/<run_id>/{config,logs,checkpoints,samples,environment}/ + status.json + metrics.csv
    /content: models/, Qwen3-TTS/, runs/<speaker>/<run_id>/checkpoints (training writes here first), work/<speaker>/
    """

    def __init__(self, p: Params, run_id: Optional[str] = None):
        r = p.drive_root
        run_id = run_id or p.run_id
        self.root = r
        self.cache = os.path.join(r, "cache")
        self.models = os.path.join(self.cache, "models")
        self.repo = os.path.join(self.cache, "Qwen3-TTS")
        self.wheelhouse = os.path.join(self.cache, "wheelhouse")
        self.pip_cache = os.path.join(self.cache, "pip")
        self.fingerprints = os.path.join(self.cache, "fingerprints.json")
        self.datasets = os.path.join(r, "datasets")
        self.logs = os.path.join(r, "logs")
        self.timings = os.path.join(self.logs, "timings.jsonl")
        self.prepared = os.path.join(r, "prepared", p.speaker_name)
        self.codes = os.path.join(self.prepared, "train_with_codes.jsonl")
        self.codes_meta = os.path.join(self.prepared, "prepared.json")
        self.run_id = run_id
        self.runs = os.path.join(r, "runs", p.speaker_name)
        self.run = os.path.join(self.runs, run_id)
        self.run_config = os.path.join(self.run, "config")
        self.run_logs = os.path.join(self.run, "logs")
        self.checkpoints = os.path.join(self.run, "checkpoints")
        self.ckpt_manifest = os.path.join(self.checkpoints, OWNED_MANIFEST)
        self.samples = os.path.join(self.run, "samples")
        self.environment = os.path.join(self.run, "environment")
        self.status = os.path.join(self.run, "status.json")
        self.metrics = os.path.join(self.run, "metrics.csv")
        self.runlog = os.path.join(self.run_logs, "run.log")
        self.local_models = os.path.join(p.local_root, "models")
        self.local_repo = os.path.join(p.local_root, "Qwen3-TTS")
        self.patched = os.path.join(p.local_root, "qwen3_patched")
        self.work = os.path.join(p.local_root, "work", p.speaker_name)
        self.codes_local = os.path.join(self.work, "train_with_codes.jsonl")
        self.local_checkpoints = os.path.join(p.local_root, "runs", p.speaker_name, run_id, "checkpoints")

    def folders(self) -> list:
        return [self.root, self.cache, self.models, self.wheelhouse, self.pip_cache, self.datasets, self.logs, self.prepared,
                self.run, self.run_config, self.run_logs, self.checkpoints, self.samples, self.environment]

    def timing_layout(self, p: Params) -> "Layout":
        return Layout(p, p.run_id + "-timing")


def similar_drive_folders(root: str) -> list:
    """Names next to ``root`` that contain 'qwen' or 'tts' (nothing else of the owner's Drive is listed)."""
    try:
        names = sorted(os.listdir(os.path.dirname(root.rstrip("/"))))
    except OSError:
        return []
    return [n for n in names if "qwen" in n.lower() or "tts" in n.lower()][:10]


def ensure_layout(p: Params, *, create_run: bool = True) -> Layout:
    lay = Layout(p)
    if p.drive_root.startswith(DRIVE_MOUNT + "/") and not os.path.isdir(f"{DRIVE_MOUNT}/MyDrive"):
        raise StepFailed("Google Drive is not mounted. Run the 'mount Drive' cell first (or drive.mount('/content/drive')).")
    if p.drive_root.startswith(DRIVE_MOUNT + "/") and not os.path.isdir(p.drive_root):
        # Never create the Drive root: if Colab cannot see the real folder yet (sync lag), creating it here makes a SECOND folder
        # with the same name, and Colab can then keep resolving the name to the empty one.
        raise StepFailed(f"The Drive folder {p.drive_root} does not exist, so nothing was created. Create it in Google Drive with exactly "
                         "that name and wait until it has finished syncing, or change DRIVE_ROOT. Drive allows two folders with the same "
                         "name; if an earlier run created an empty copy, delete that copy. "
                         f"Similar folders in My Drive: {similar_drive_folders(p.drive_root) or 'none'}.")
    created = []
    run_only = {lay.run, lay.run_config, lay.run_logs, lay.checkpoints, lay.samples, lay.environment}
    for folder in lay.folders():
        if (folder in run_only and not create_run) or os.path.isdir(folder):
            continue
        os.makedirs(folder)
        created.append(folder)
    say(f"Drive folder: {lay.root}" + (f"  (created {len(created)} missing folder(s))" if created else "  (layout already complete)"))
    if create_run:
        say(f"Run folder: {lay.run}")
    return lay


def write_json_atomic(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False, sort_keys=False)
    os.replace(tmp, path)


def read_json(path: str, default=None):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


# --------------------------------------------------------------------------------------------- processes
def capture(cmd: list, **kw) -> tuple:
    """(returncode, combined output). Never raises for a missing program."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, **kw)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, str(exc)


def stop_process(proc) -> None:
    """Stop a child process (and anything it started) that is still running. An interrupted or failed cell must never
    leave training running on the GPU, because a connected runtime keeps using compute units."""
    if proc is None or proc.poll() is not None:
        return
    posix = os.name == "posix"
    for sig_name, wait in (("SIGTERM", 15), ("SIGKILL", 10)):
        try:
            if posix:
                import signal

                os.killpg(proc.pid, getattr(signal, sig_name))  # the child runs in its own session: signal its whole group
            elif sig_name == "SIGTERM":
                proc.terminate()
            else:
                proc.kill()
        except (OSError, ProcessLookupError):
            pass
        try:
            proc.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


PYTHON = [sys.executable]  # the interpreter every step uses; switched to the isolated environment's once that exists


def py() -> str:
    return PYTHON[0]


def run_stream(cmd: list, *, cwd: Optional[str] = None, env: Optional[dict] = None, log_path: Optional[str] = None,
               on_line: Optional[Callable] = None, max_seconds: Optional[float] = None) -> int:
    """Run a command, echo its output and append it to a log file (on Drive). Returns the exit code. If the caller is
    interrupted (the notebook's stop button) or raises, the command is stopped too."""
    say("$ " + " ".join(cmd))
    log = open(log_path, "a", encoding="utf-8") if log_path else None
    proc = None
    timer = None
    try:
        proc = subprocess.Popen(cmd, cwd=cwd, env={**os.environ, **(env or {})}, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1,
                                **({"start_new_session": True} if os.name == "posix" else {}))
        if max_seconds:
            timer = threading.Timer(max_seconds, stop_process, args=(proc,))  # a step that runs past its limit is stopped, not waited for
            timer.daemon = True
            timer.start()
        for line in proc.stdout:  # type: ignore[union-attr]
            handled = bool(on_line(line)) if on_line else False  # True: the caller printed its own, richer line instead
            if not handled:
                sys.stdout.write(line)
                sys.stdout.flush()
            if log:
                log.write(line)
                log.flush()
        return proc.wait()
    finally:
        if timer is not None:
            timer.cancel()
        stop_process(proc)
        if log:
            log.close()


def new_log(lay: Layout, p: Params, step: str, per_run: bool = False) -> str:
    folder = lay.run_logs if per_run else lay.logs
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{p.speaker_name}-{step}-{time.strftime('%Y%m%d-%H%M%S')}.log")
    say(f"Log: {path}")
    return path


# --------------------------------------------------------------------------------------------- timing, units, reminders
_TIMES: list = []  # this Python session's step timings (the notebook keeps one session for all cells)


def fmt_secs(sec: float) -> str:
    sec = float(sec)
    if sec < 90:
        return f"{sec:.1f} s"
    return f"{int(sec // 60)} min {int(sec % 60):02d} s" if sec < 3600 else f"{int(sec // 3600)} h {int(sec % 3600 // 60):02d} min"


@contextlib.contextmanager
def timed(name: str, phase: str = "setup"):
    """Time a step, print it, and remember it for the summary (setup time is billed in compute units too)."""
    start, ok = time.time(), True
    try:
        yield
    except BaseException:
        ok = False
        raise
    finally:
        dt = time.time() - start
        _TIMES.append({"step": name, "phase": phase, "seconds": round(dt, 1), "ok": ok, "written": False})
        say(f"[{name}: {fmt_secs(dt)}]")


def session_seconds(*phases: str) -> float:
    return sum(t["seconds"] for t in _TIMES if t["phase"] in phases)


def timing_summary(lay: Layout, title: str, *phases: str) -> None:
    """Per-step table for this session, next to the previous run's time for the same step (kept in logs/timings.jsonl on Drive)."""
    rows = [t for t in _TIMES if t["phase"] in phases]
    if not rows:
        return
    previous: dict = {}
    try:
        with open(lay.timings, encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                previous[rec["step"]] = rec["seconds"]
    except (OSError, ValueError, KeyError):
        pass
    say(f"\n{title}: where the time went (earlier run's time for the same step in brackets, so you can see what the cache saved)")
    for t in rows:
        prev = previous.get(t["step"])
        say(f"  {t['step']:<44} {fmt_secs(t['seconds']):>12}" + (f"   [earlier: {fmt_secs(prev)}]" if prev is not None else "   [first time measured]"))
    say(f"  {'total':<44} {fmt_secs(sum(t['seconds'] for t in rows)):>12}")
    try:
        os.makedirs(lay.logs, exist_ok=True)
        with open(lay.timings, "a", encoding="utf-8") as fh:
            for t in rows:
                if not t["written"]:
                    fh.write(json.dumps({"when": time.strftime("%Y-%m-%d %H:%M:%S"), "step": t["step"], "seconds": t["seconds"]}) + "\n")
                    t["written"] = True
    except OSError:
        pass


def default_units_per_hour(gpu: Optional[Gpu]) -> Optional[tuple]:
    """(rate, label) from the owner's table, or None when the GPU is not in it."""
    if gpu is None:
        return None
    for hint, low, high, rate, label in UNIT_RATES:
        if hint in gpu.name and low <= gpu.vram_gb < high:
            return rate, label
    return None


def finish_reminder() -> None:
    """An idle session keeps using compute units; say how to end it."""
    say("")
    say("=" * 78)
    say("REMINDER: disconnect and DELETE the runtime now. An idle session keeps using compute units.")
    say("  In the notebook:  run   " + DISCONNECT_CELL)
    say("     (google.colab.runtime.unassign() was announced by the Colab team in 2022 and is not run here; if it errors, use the")
    say("      menu: Runtime > Disconnect and delete runtime)")
    say(f"  From the colab command line:   colab stop -s {SESSION}")
    say("  Everything on Drive (cache, checkpoints, logs) stays; /content is erased.")
    say("=" * 78)


def breakeven_line(rate_a: float, rate_b: float, name_a: str = "the cheaper-per-hour GPU", name_b: str = "the dearer-per-hour GPU") -> str:
    ratio = rate_b / rate_a
    return (f"Break-even rule: a GPU costing R2 units/hour only saves units over one costing R1 units/hour if it is more than R2/R1 times "
            f"faster on this job. Here {name_b} ({rate_b:g}/h) vs {name_a} ({rate_a:g}/h): {rate_b:g} / {rate_a:g} = {ratio:.1f}x.")


# --------------------------------------------------------------------------------------------- GPU and environment
@dataclasses.dataclass
class Gpu:
    name: str
    vram_gb: float
    cc: Optional[tuple]  # (major, minor) or None when unknown

    @property
    def arch(self) -> str:
        """Architecture family from the compute capability (the name only labels it)."""
        if not self.cc:
            return "unknown"
        major, minor = self.cc
        return {12: "Blackwell (workstation/consumer, sm_120)", 10: "Blackwell (datacenter, sm_100)", 9: "Hopper",
                7: "Volta" if minor == 0 else "Turing", 6: "Pascal", 3: "Kepler"}.get(major, "Ada" if (major, minor) == (8, 9) else "Ampere" if major == 8 else "unknown")

    @property
    def blackwell(self) -> bool:
        return bool(self.cc and self.cc[0] >= 10)

    @property
    def flash_ok(self) -> bool:
        """FlashAttention 2 per its README: Ampere, Ada or Hopper (compute capability 8.x and 9.x) only."""
        return bool(self.cc and self.cc[0] in (8, 9))


# What each runtime is expected to do. Everything here is an UNTESTED estimate or the project's README, never a measurement.
GPU_TABLE = [
    # (name hint, memory shown, architecture, attention path by default, batch size by default)
    ("A100 40 GB", "40 GB", "Ampere sm_80", "FlashAttention 2", 2),
    ("A100 80 GB", "80 GB", "Ampere sm_80", "FlashAttention 2", 4),
    ("H100 80 GB", "80 GB", "Hopper sm_90", "FlashAttention 2", 4),
    ("L4", "24 GB", "Ada sm_89", "FlashAttention 2", 1),
    ("T4", "15 GB", "Turing sm_75", "none: needs ATTN_FALLBACK=True (unverified)", 1),
    ("G4 48 GB (RTX PRO 6000 Blackwell)", "48 GB", "Blackwell sm_120", "SDPA patched copy (default; unverified)", 2),
    ("G4 96 GB (RTX PRO 6000 Blackwell)", "96 GB", "Blackwell sm_120", "SDPA patched copy (default; unverified)", 4),
    ("unknown GPU", "any", "unknown", "refuses unless ATTN_FALLBACK=True", 1),
]


def attention_choice(gpu: Optional[Gpu], mode, *, flash_installed: Optional[bool] = None) -> tuple:
    """(path, why). path is 'flash2', 'sdpa' or 'refuse'. mode is Params.attn_fallback: 'auto', True or False.
    ``flash_installed``: None = not known yet (use the hardware default); False = FlashAttention 2 is not usable in this runtime."""
    if gpu is None:
        return "refuse", "no GPU"
    if mode is True:
        return "sdpa", "ATTN_FALLBACK=True: you chose the patched SDPA copy of the script"
    if mode is False:
        if gpu.flash_ok:
            return "flash2", "ATTN_FALLBACK=False: the original script with FlashAttention 2"
        if gpu.blackwell:
            return "flash2", ("ATTN_FALLBACK=False on a Blackwell GPU: you forced FlashAttention 2, which its README does not list "
                              "(Ampere, Ada, Hopper only) and which is likely to fail")
        return "refuse", "FlashAttention 2 cannot run on this GPU and ATTN_FALLBACK=False forbids the fallback"
    if gpu.flash_ok and flash_installed is False:
        return "sdpa", ("FlashAttention 2 is not installed in this runtime (no ready-made package exists for it, and it is not compiled "
                        "unless BUILD_FLASH_ATTN=True), so the patched copy of the script with PyTorch's built-in SDPA attention is used (unverified)")
    if gpu.flash_ok:
        return "flash2", "Ampere/Ada/Hopper GPU: the original script with FlashAttention 2"
    if gpu.blackwell:
        return "sdpa", ("Blackwell GPU: the FlashAttention 2 README lists Ampere, Ada and Hopper only, and its issue tracker has open "
                        "requests for RTX PRO 6000 / sm_120 support, so the default is a patched copy of the script with PyTorch SDPA "
                        "attention (unverified). Set ATTN_FALLBACK=False to force FlashAttention 2 instead")
    return "refuse", ("this GPU (" + gpu.arch + ") cannot run FlashAttention 2 and its architecture is not one this script knows how to handle by default; "
                      "set ATTN_FALLBACK=True to try the patched SDPA copy (unverified)")


_FLASH = {"ok": None}  # is FlashAttention 2 usable in this runtime? None until setup (or the first probe) has decided


def flash_installed() -> bool:
    """Probed once and remembered; set by ensure_flash_attn."""
    if _FLASH["ok"] is None:
        _FLASH["ok"] = bool(flash_functional()[0])
    return bool(_FLASH["ok"])


def current_attention(gpu: Optional[Gpu], p: Optional[Params], *, probe: bool = False) -> tuple:
    """attention_choice for these parameters. ``probe`` asks the runtime whether FlashAttention 2 really works (use after setup)."""
    mode = p.attn_fallback if p is not None else "auto"
    return attention_choice(gpu, mode, flash_installed=flash_installed() if probe else _FLASH["ok"])


def parse_nvidia_smi(text: str) -> Optional[Gpu]:
    """First line of `nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits`."""
    for line in (text or "").splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 2 and re.fullmatch(r"\d+(\.\d+)?", parts[1]):
            cc = None
            if len(parts) >= 3 and re.fullmatch(r"\d+\.\d+", parts[2]):
                cc = tuple(int(x) for x in parts[2].split("."))
            else:
                cc = next((c for key, c in CC_BY_NAME if key in parts[0]), None)
            return Gpu(parts[0], round(float(parts[1]) / 1024, 1), cc)
    return None


def detect_gpu(capture_fn: Callable = capture) -> Optional[Gpu]:
    code, out = capture_fn(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits"])
    gpu = parse_nvidia_smi(out) if code == 0 else None
    if gpu is None:  # older drivers do not know compute_cap
        code, out = capture_fn(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
        gpu = parse_nvidia_smi(out) if code == 0 else None
    return gpu


def recommended_batch(gpu: Optional[Gpu]) -> int:
    if gpu is None:
        return 0
    for limit, batch in BATCH_BY_VRAM:
        if gpu.vram_gb < limit:
            return batch
    return 1


def gpu_report(p: Optional[Params] = None, gpu: Optional[Gpu] = None, *, detect: Callable = detect_gpu) -> Optional[Gpu]:
    gpu = gpu or detect()
    say("== GPU")
    if gpu is None:
        say("No GPU found. Runtime > Change runtime type > pick a GPU (A100, L4 or H100 for this script).")
        return None
    cc = ".".join(map(str, gpu.cc)) if gpu.cc else "unknown"
    say(f"{gpu.name}   {gpu.vram_gb:g} GB   compute capability {cc}")
    rec = recommended_batch(gpu)
    if rec == 0:
        say(f"Only {gpu.vram_gb:g} GB of GPU memory: too small for full fine-tuning of the 1.7 B model (about 14 GB before activations).")
    else:
        say(f"Suggested batch size for this GPU: {rec} (a guess from the memory size, not measured). Use BATCH_SIZE='auto' to take it.")
    say(f"Architecture: {gpu.arch}")
    path, why = current_attention(gpu, p)
    say(f"Attention path: {path.upper()}. {why}.")
    if path == "sdpa":
        say("The patched copy is written under /content only; Qwen's cached repo is never modified. UNVERIFIED: not run by the author.")
    if path == "refuse":
        say("Options: switch the runtime to an A100, H100, L4 or G4, or set ATTN_FALLBACK=True for the unverified SDPA copy "
            "(a T4 also lacks native bfloat16, which the script uses).")
    say("Expected runtimes (UNTESTED estimates, none run on Colab): " + "; ".join(f"{n}: {a}, batch {b}" for n, _m, _ar, a, b in GPU_TABLE if n.split()[0] in ("A100", "H100", "L4", "G4")))
    if p is not None:
        want = p.batch_size
        if want != "auto" and rec and int(want) > rec:
            say(f"Note: BATCH_SIZE={want} is above the suggestion ({rec}); out-of-memory is possible. Your value is kept.")
    return gpu


def resolve_batch(p: Params, gpu: Optional[Gpu]) -> int:
    if p.batch_size == "auto":
        return max(1, recommended_batch(gpu))
    return int(p.batch_size)


ENV_PROBE_CODE = (
    "import sys, os, json\n"
    "e = {'python': 'cp%d%d' % sys.version_info[:2], 'python_version': sys.version.split()[0], 'platform': sys.platform + '-' + os.uname().machine,\n"
    "     'torch': None, 'torch_mm': None, 'cuda': None, 'cuda_tag': None, 'cxx11abi': None}\n"
    "try:\n"
    "    import torch\n"
    "    e['torch'] = str(torch.__version__).split('+')[0]; e['torch_mm'] = '.'.join(e['torch'].split('.')[:2]); e['cuda'] = torch.version.cuda\n"
    "    e['cuda_tag'] = ('cu' + torch.version.cuda.split('.')[0]) if torch.version.cuda else None; e['cxx11abi'] = bool(torch.compiled_with_cxx11_abi())\n"
    "except Exception:\n"
    "    pass\n"
    "print(json.dumps(e))\n")


def _environment_from(python: str) -> dict:
    code, out = capture([python, "-c", ENV_PROBE_CODE])
    try:
        return json.loads(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        raise StepFailed(f"Could not read Python/torch/CUDA from the isolated environment ({python}): {out[-300:]}")


def detect_environment() -> dict:
    """Python, torch, CUDA and GPU of this runtime. Fields are None where they cannot be read (no torch yet)."""
    if py() != sys.executable:
        env = _environment_from(py())
        gpu = detect_gpu()
        env["gpu"] = gpu.name if gpu else None
        env["gpu_cc"] = ".".join(map(str, gpu.cc)) if gpu and gpu.cc else None
        return env
    env = {"python": f"cp{sys.version_info[0]}{sys.version_info[1]}", "python_version": sys.version.split()[0],
           "platform": f"{sys.platform}-{os.uname().machine}" if hasattr(os, "uname") else sys.platform,
           "torch": None, "torch_mm": None, "cuda": None, "cuda_tag": None, "cxx11abi": None}
    try:
        import torch  # noqa: WPS433 — preinstalled on Colab; absent in unit tests

        env["torch"] = str(torch.__version__).split("+")[0]
        env["torch_mm"] = ".".join(env["torch"].split(".")[:2])
        env["cuda"] = torch.version.cuda
        env["cuda_tag"] = ("cu" + torch.version.cuda.split(".")[0]) if torch.version.cuda else None
        env["cxx11abi"] = bool(torch.compiled_with_cxx11_abi())
    except Exception:  # noqa: BLE001
        pass
    gpu = detect_gpu()
    env["gpu"] = gpu.name if gpu else None
    env["gpu_cc"] = ".".join(map(str, gpu.cc)) if gpu and gpu.cc else None
    return env


def wheel_key(env: dict) -> str:
    """What a compiled FlashAttention wheel is valid for: Python, torch minor, CUDA major, C++ ABI, platform AND the GPU
    architecture. A wheel saved on one GPU type is never reused on another."""
    return "-".join(str(env.get(k)) for k in ("python", "torch_mm", "cuda_tag", "cxx11abi", "platform", "gpu_cc"))


def full_key(env: dict) -> str:
    return wheel_key(env) + f"|{env.get('gpu')}|{env.get('gpu_cc')}"


def record_fingerprint(lay: Layout, env: dict) -> bool:
    """Store this runtime's fingerprint with the cache. Returns True when this exact runtime fingerprint was seen before."""
    known = {}
    if os.path.isfile(lay.fingerprints):
        try:
            with open(lay.fingerprints, encoding="utf-8") as fh:
                known = json.load(fh)
        except (OSError, ValueError):
            known = {}
    key, now = full_key(env), time.strftime("%Y-%m-%d %H:%M:%S")
    seen = key in known
    entry = known.get(key, {"first_seen": now, "env": env})
    entry["last_seen"] = now
    known[key] = entry
    tmp = lay.fingerprints + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(known, fh, indent=2, sort_keys=True)
    os.replace(tmp, lay.fingerprints)
    return seen


# --------------------------------------------------------------------------------------------- dependency checks
def probe_module(name: str) -> tuple:
    """(importable, detail). A failed import reports the ROOT cause (transformers wraps the real error in a generic one)."""
    code, out = capture([py(), "-c", PROBE_CODE.format(name=name)])
    last = (out.strip().splitlines() or [""])[-1]
    return code == 0, (last if code == 0 else last[:220] or "import failed")


PROBE_CODE = """import sys
try:
    import {name} as m
    print(getattr(m, '__version__', 'ok'))
except BaseException as e:
    r = e
    for _ in range(12):
        nxt = r.__cause__ or r.__context__
        if nxt is None:
            break
        r = nxt
    print(type(r).__name__ + ': ' + (str(r).strip().splitlines() or [''])[0])
    sys.exit(1)
"""

COMPANION_CODE = """import json
out = {}
for n in ('torch', 'torchvision', 'torchaudio'):
    try:
        m = __import__(n)
        out[n] = str(m.__version__).split('+')[0]
    except Exception:
        pass
try:
    import torch
    out['cuda'] = torch.version.cuda
except Exception:
    pass
print(json.dumps(out))
"""


def companion_problems(v: dict) -> list:
    """torchaudio must match torch's minor version and torchvision's minor must be torch's + 15 (2.12 pairs with 0.27)."""
    torch_v = v.get("torch")
    if not torch_v:
        return []
    try:
        major, minor = (int(x) for x in torch_v.split(".")[:2])
    except ValueError:
        return []
    bad = []
    ta, tv = v.get("torchaudio"), v.get("torchvision")
    if ta and ta.split(".")[:2] != [str(major), str(minor)]:
        bad.append(f"torchaudio {ta}")
    if tv and tv.split(".")[:2] != ["0", str(minor + 15)]:
        bad.append(f"torchvision {tv}")
    return bad


def repair_torch_companions(p: Params, lay: Layout, log: str, *, stream: Callable = run_stream, run: Callable = capture) -> bool:
    """A torch upgraded in place leaves torchvision/torchaudio behind, and transformers then fails with 'operator torchvision::nms
    does not exist'. Reinstall matching ones with torch pinned to what is installed. True when nothing needed repair or it worked."""
    code, out = run([py(), "-c", COMPANION_CODE])
    try:
        v = json.loads((out.strip().splitlines() or ["{}"])[-1])
    except ValueError:
        return True
    bad = companion_problems(v)
    if not bad:
        return True
    cuda = v.get("cuda")
    say(f"torch {v['torch']} is installed, but {', '.join(bad)} do not belong to it (this makes transformers fail to import). Reinstalling matching ones.")
    cmd = [py(), "-m", "pip", "install", "-q", "--cache-dir", lay.pip_cache, f"torch=={v['torch']}", "torchvision", "torchaudio"]
    if cuda:
        cmd += ["--index-url", f"https://download.pytorch.org/whl/cu{cuda.replace('.', '')}"]
    ok = stream(cmd, log_path=log, max_seconds=PIP_MAX_SECONDS) == 0
    say("Matching torchvision/torchaudio installed." if ok else "Could not install matching torchvision/torchaudio; see the log.")
    return ok


def check_table(rows: list) -> bool:
    """rows: (name, status, detail) with status PASS, FAIL, SKIP or INFO. Prints the table; True when nothing failed."""
    width = max([len(r[0]) for r in rows] + [10])
    say("")
    say(f"{'check'.ljust(width)}  result  detail")
    say("-" * (width + 40))
    for name, status, detail in rows:
        say(f"{name.ljust(width)}  {status.ljust(6)}  {detail}")
    failed = [r[0] for r in rows if r[1] == "FAIL"]
    say("")
    say("ALL CHECKS PASSED" if not failed else "FAILED: " + ", ".join(failed))
    return not failed


def dependency_rows(p: Params, lay: Layout, env: dict, gpu: Optional[Gpu], *, probe: Callable = probe_module,
                    flash_probe: Callable = None, sdpa_probe: Callable = None, arch_probe: Callable = None, conv_probe: Callable = None) -> list:
    rows = [("python / torch", "INFO", f"python {env.get('python_version')}, torch {env.get('torch') or 'not importable'}, CUDA {env.get('cuda')}")]
    rows.append(("gpu", "PASS" if gpu else "FAIL", f"{gpu.name}, {gpu.vram_gb:g} GB" if gpu else "no GPU runtime selected"))
    for mod in CHECK_MODULES:
        ok, detail = probe(mod)
        rows.append((mod, "PASS" if ok else "FAIL", detail))
    path, why = current_attention(gpu, p, probe=True)
    if gpu is not None and gpu.cc:
        ok, detail = (arch_probe or torch_arch_probe)(gpu.cc)
        rows.append(("torch kernels for this GPU", "PASS" if ok else "FAIL", detail))
    if gpu is not None:
        ok, detail = (conv_probe or conv_functional)()
        rows.append(("torch convolution (cuDNN)", "PASS" if ok else "FAIL", detail))
    if path == "flash2":
        ok, detail = (flash_probe or flash_functional)() if gpu else (False, "needs a GPU runtime")
        rows.append(("attention: flash_attn", "PASS" if ok else "FAIL", detail))
    elif path == "sdpa":
        ok, detail = (sdpa_probe or sdpa_functional)()
        rows.append(("attention: PyTorch SDPA (patched copy)", "PASS" if ok else "FAIL",
                     detail + "; the patched training copy itself is UNVERIFIED" if ok else detail))
    else:
        rows.append(("attention", "FAIL", why))
    repo_ok = os.path.isfile(os.path.join(lay.local_repo, "finetuning", "sft_12hz.py"))
    head = _git_head(lay.local_repo) if repo_ok else None
    rows.append(("Qwen3-TTS repo", "PASS" if head == QWEN_COMMIT else "FAIL",
                 f"{lay.local_repo} @ {head[:10]}" if head == QWEN_COMMIT else f"not at the pinned commit {QWEN_COMMIT[:10]}"))
    for label, model in (("base model (local)", p.base_model), ("tokenizer (local)", p.tokenizer_model)):
        path = local_model_dir(lay, model)
        ok = os.path.isfile(os.path.join(path, DONE_MARKER))
        rows.append((label, "PASS" if ok else "FAIL", path if ok else f"{path} is missing or incomplete"))
    free = shutil.disk_usage(p.local_root).free / 2**30 if os.path.isdir(p.local_root) else None
    rows.append(("free disk (/content)", "INFO" if free is None else ("PASS" if free >= MODEL_NEEDS_GB else "FAIL"),
                 "unknown" if free is None else f"{free:.0f} GB free; the local model copy needs about {MODEL_NEEDS_GB} GB or less"))
    return rows


def torch_arch_probe(cc: tuple) -> tuple:
    """Does this torch build ship kernels (or PTX) for the GPU's architecture? Answered by torch itself on the runtime."""
    tag = f"{cc[0]}{cc[1]}"
    code, out = capture([py(), "-c", "import torch; print(' '.join(torch.cuda.get_arch_list()))"])
    if code != 0:
        return False, "torch could not be queried"
    archs = out.split()
    ok = any(a in (f"sm_{tag}", f"compute_{tag}") or (a.startswith("compute_") and int(a[8:]) <= int(tag)) for a in archs if a[a.find("_") + 1:].isdigit())
    return ok, (f"torch lists {' '.join(archs)}" if ok else f"torch's build lists {' '.join(archs)}, nothing for sm_{tag}: it cannot run on this GPU")


def conv_functional() -> tuple:
    """A real convolution on the GPU. Training's audio-code step needs cuDNN; a torch upgraded in place next to Colab's older cuDNN fails
    here with CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH, which otherwise only shows up minutes into preparation."""
    code, out = capture([py(), "-c",
                         "import torch\n"
                         "x = torch.randn(1, 1, 256, device='cuda')\n"
                         "torch.nn.Conv1d(1, 4, 3).cuda()(x)\n"
                         "torch.cuda.synchronize()\n"
                         "print('ok')\n"])
    last = (out.strip().splitlines() or [""])[-1]
    if code == 0:
        return True, "convolution (cuDNN) ran on the GPU"
    hint = (" Torch and cuDNN do not belong together, usually because torch was upgraded inside this runtime: delete the runtime and start a "
            "new one without upgrading torch (the notebook builds a matching isolated environment itself).") if "CUDNN" in out.upper() else ""
    return False, (last[:160] or "convolution failed") + hint


def sdpa_functional() -> tuple:
    code, out = capture([py(), "-c",
                         "import torch\n"
                         "q = torch.randn(1, 2, 16, 64, device='cuda', dtype=torch.bfloat16)\n"
                         "torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)\n"
                         "print('ok')\n"])
    last = (out.strip().splitlines() or [""])[-1]
    return code == 0, ("SDPA ran on the GPU in bfloat16" if code == 0 else last[:160] or "SDPA failed")


def flash_functional() -> tuple:
    code, out = capture([py(), "-c",
                         "import torch, flash_attn\n"
                         "from flash_attn import flash_attn_func\n"
                         "q = torch.randn(1, 16, 2, 64, device='cuda', dtype=torch.bfloat16)\n"
                         "flash_attn_func(q, q, q)\n"
                         "print(flash_attn.__version__)\n"])
    last = (out.strip().splitlines() or [""])[-1]
    return code == 0, (f"{last} (ran on the GPU)" if code == 0 else last[:160] or "not importable")


def _git_head(repo: str) -> Optional[str]:
    code, out = capture(["git", "-C", repo, "rev-parse", "HEAD"])
    return out.strip() if code == 0 else None


# --------------------------------------------------------------------------------------------- repo, models
def ensure_repo(lay: Layout, *, capture_fn: Callable = capture, stream: Callable = run_stream) -> None:
    """Pinned clone kept on Drive; a local copy under /content is what pip installs and the scripts run from."""
    say("== Qwen3-TTS repository (pinned commit " + QWEN_COMMIT[:10] + ")")
    if _git_head(lay.repo) == QWEN_COMMIT:
        say("Drive copy is at the pinned commit (cache hit).")
    else:
        if not os.path.isdir(os.path.join(lay.repo, ".git")):
            if os.path.exists(lay.repo) and os.listdir(lay.repo):
                raise StepFailed(f"{lay.repo} exists but is not a git clone; move it away so the cache can be rebuilt.")
            if stream(["git", "clone", "--quiet", QWEN_REPO, lay.repo]) != 0:
                raise StepFailed("git clone of the Qwen3-TTS repository failed (network?).")
        if stream(["git", "-C", lay.repo, "fetch", "--quiet", "origin", QWEN_COMMIT]) != 0:
            say("fetch of the commit by id failed; trying a plain checkout of what is already cloned")
        if stream(["git", "-C", lay.repo, "checkout", "--quiet", QWEN_COMMIT]) != 0:
            raise StepFailed(f"Could not check out {QWEN_COMMIT}.")
    if _git_head(lay.local_repo) != QWEN_COMMIT:
        if os.path.exists(lay.local_repo):
            if not os.path.isfile(os.path.join(lay.local_repo, SOURCE_MARKER)):
                raise StepFailed(f"{lay.local_repo} exists and was not made by this script; remove it or use another local folder.")
            shutil.rmtree(lay.local_repo)
        shutil.copytree(lay.repo, lay.local_repo)
        with open(os.path.join(lay.local_repo, SOURCE_MARKER), "w", encoding="utf-8") as fh:
            fh.write(QWEN_COMMIT + "\n")
        say(f"Copied to {lay.local_repo} (local disk is faster than Drive).")
    else:
        say("Local copy is at the pinned commit (cache hit).")


def model_slug(model: str) -> str:
    return model.replace("/", "--")


def local_model_dir(lay: Layout, model: str) -> str:
    return os.path.join(lay.local_models, model.split("/")[-1])


def drive_model_dir(lay: Layout, model: str) -> str:
    return os.path.join(lay.models, model_slug(model))


def hf_token() -> tuple:
    """(token or None, where it came from). From the HF_TOKEN environment variable or Colab Secrets; never written anywhere."""
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok, "the HF_TOKEN environment variable"
    try:
        from google.colab import userdata  # only exists on Colab

        tok = userdata.get("HF_TOKEN")
        if tok:
            return tok, "Colab Secrets (HF_TOKEN)"
    except Exception:  # noqa: BLE001 — no secret set, or not on Colab
        pass
    return None, "none (not needed: the models are public)"


def download_model(model: str, dest: str) -> None:
    tok, _src = hf_token()
    if py() != sys.executable:  # huggingface_hub lives in the isolated environment; the token travels in the environment only
        code = ("import os, sys; from huggingface_hub import snapshot_download as d; "
                "d(sys.argv[1], local_dir=sys.argv[2], token=os.environ.get('HF_TOKEN') or None)")
        if run_stream([py(), "-c", code, model, dest], env={"HF_TOKEN": tok} if tok else None, max_seconds=PIP_MAX_SECONDS * 2) != 0:
            raise StepFailed(f"Downloading {model} failed; see the output above.")
        return
    from huggingface_hub import snapshot_download  # installed by ensure_deps; imported late on purpose

    snapshot_download(model, local_dir=dest, **({"token": tok} if tok else {}))


def _mark_done(folder: str) -> None:
    with open(os.path.join(folder, DONE_MARKER), "w", encoding="utf-8") as fh:
        fh.write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")


def ensure_models(p: Params, lay: Layout, *, download: Callable = download_model) -> None:
    """Models are downloaded once into the Drive cache (CACHE_MODELS_ON_DRIVE) and copied to /content, because training needs a local folder."""
    say("== Models")
    _tok, src = hf_token()
    say(f"Hugging Face token: {src}. A token is only ever read from there; it is never written to Drive or to any file.")
    os.makedirs(lay.local_models, exist_ok=True)
    for model in (p.base_model, p.tokenizer_model):
        drive, local = drive_model_dir(lay, model), local_model_dir(lay, model)
        short = model.split("/")[-1]
        if os.path.isfile(os.path.join(local, DONE_MARKER)):
            say(f"{model}: local copy at {local} (already there).")
            continue
        if not p.cache_models_on_drive:
            with timed(f"model download {short} (straight to /content)"):
                os.makedirs(local, exist_ok=True)
                download(model, local)
                _mark_done(local)
            continue
        if os.path.isfile(os.path.join(drive, DONE_MARKER)):
            say(f"{model}: in the Drive cache (cache hit, no download).")
        else:
            say(f"{model}: downloading into the Drive cache {drive} (first time only; a few GB, speed depends on Colab's network).")
            with timed(f"model download {short} (to Drive cache)"):
                os.makedirs(drive, exist_ok=True)
                download(model, drive)
                _mark_done(drive)
        say(f"{model}: copying to {local} (needs a few GB; {shutil.disk_usage(lay.local_models).free / 2**30:.0f} GB free; reading from Drive is the slow part).")
        with timed(f"model copy {short} (Drive to /content)"):
            if os.path.exists(local):
                shutil.rmtree(local)  # only a half-finished copy of ours: no completion marker
            tmp = local + ".partial"
            if os.path.exists(tmp):
                shutil.rmtree(tmp)
            shutil.copytree(drive, tmp, ignore=shutil.ignore_patterns(".cache", DONE_MARKER))
            os.replace(tmp, local)
            _mark_done(local)


# --------------------------------------------------------------------------------------------- FlashAttention
def prebuilt_wheel_name(env: dict) -> Optional[str]:
    """Official wheel name for this runtime, or None when the release has none (names read from the release listing)."""
    if not (env.get("platform", "").startswith("linux-") and env.get("platform", "").endswith(("x86_64", "amd64"))):
        return None
    if env.get("cuda_tag") != "cu12" or env.get("python") not in FLASH_PREBUILT.get(env.get("torch_mm") or "", ()):
        return None
    if env.get("cxx11abi") is None:
        return None
    abi = "TRUE" if env["cxx11abi"] else "FALSE"
    return (f"flash_attn-{FLASH_VERSION}+cu12torch{env['torch_mm']}cxx11abi{abi}-{env['python']}-{env['python']}-linux_x86_64.whl")


def wheel_dir(lay: Layout, env: dict) -> str:
    """Wheels live in a folder per GPU architecture, so a wheel built on one GPU type is never found on another."""
    return os.path.join(lay.wheelhouse, "sm" + str(env.get("gpu_cc") or "unknown").replace(".", ""))


def find_cached_wheel(lay: Layout, env: dict) -> Optional[str]:
    """A wheel in this architecture's wheelhouse folder whose recorded runtime matches this one. Rejected wheels are skipped."""
    folder = wheel_dir(lay, env)
    if not os.path.isdir(folder):
        return None
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".whl.json"):
            continue
        try:
            with open(os.path.join(folder, name), encoding="utf-8") as fh:
                meta = json.load(fh)
        except (OSError, ValueError):
            continue
        wheel = os.path.join(folder, name[:-5])
        if meta.get("wheel_key") == wheel_key(env) and not meta.get("rejected") and os.path.isfile(wheel):
            return wheel
    return None


def _write_wheel_meta(wheel: str, env: dict, origin: str) -> None:
    with open(wheel + ".json", "w", encoding="utf-8") as fh:
        json.dump({"wheel_key": wheel_key(env), "env": env, "origin": origin, "saved": time.strftime("%Y-%m-%d %H:%M:%S")},
                  fh, indent=2)


def _reject_wheel(wheel: str) -> None:
    try:
        with open(wheel + ".json", encoding="utf-8") as fh:
            meta = json.load(fh)
        meta["rejected"] = "failed the GPU check on a later runtime"
        with open(wheel + ".json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
    except (OSError, ValueError):
        pass


def url_exists(url: str) -> bool:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.status < 400
    except (urllib.error.URLError, OSError, ValueError):
        return False


def download_file(url: str, dest: str) -> None:
    tmp = dest + ".partial"
    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as fh:
        shutil.copyfileobj(resp, fh)
    os.replace(tmp, dest)


def build_jobs() -> int:
    mem_gb = 0.0
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            mem_gb = int(fh.readline().split()[1]) / 2**20
    except (OSError, ValueError, IndexError):
        pass
    cpus = os.cpu_count() or 2
    return max(1, min(cpus, int(mem_gb // 3) if mem_gb else cpus))


BUILD_MAX_SECONDS = 90 * 60  # a requested FlashAttention build is stopped after this long


def ensure_flash_attn(*args, **kwargs) -> str:
    """Make FlashAttention 2 usable if a ready-made package exists. Remembers whether it is, so the attention path follows."""
    how = _install_flash_attn(*args, **kwargs)
    _FLASH["ok"] = how in ("installed already", "provided wheel", "cached wheel", "prebuilt wheel", "built")
    return how


def _install_flash_attn(p: Params, lay: Layout, env: dict, gpu: Optional[Gpu], log: str, *, stream: Callable = run_stream,
                      functional: Callable = flash_functional, exists: Callable = url_exists,
                      download: Callable = download_file) -> str:
    """Make flash_attn importable. Returns how: 'skipped', 'installed already', 'cached wheel', 'prebuilt wheel' or 'built'."""
    say("== FlashAttention 2")
    if gpu is None:
        say("No GPU, skipped.")
        return "skipped"
    path, why = attention_choice(gpu, p.attn_fallback)
    if path != "flash2":
        say(f"Not needed on the {path.upper()} path ({why}); skipped.")
        return "skipped"
    ok, detail = functional()
    if ok:
        say(f"Already installed and working in this runtime: {detail}")
        return "installed already"
    pip = [py(), "-m", "pip", "install", "-q", "--no-deps", "--cache-dir", lay.pip_cache]
    if p.flash_wheel:
        say(f"FLASH_WHEEL is set: trying {p.flash_wheel} first.")
        wheel = p.flash_wheel
        try:
            if wheel.lower().startswith(("http://", "https://")):
                os.makedirs(wheel_dir(lay, env), exist_ok=True)
                wheel = os.path.join(wheel_dir(lay, env), wheel_basename(wheel) or "flash_attn_provided.whl")
                download(p.flash_wheel, wheel)
            fits, why_not = wheel_fits(wheel, env)
            if not os.path.isfile(wheel):
                say("That file does not exist, so it is ignored.")
            elif not fits:
                say(f"The wheel you provided does not fit this runtime: {why_not}. Not installed (nothing wasted). "
                    "Use a runtime with the matching Python/torch/CUDA, or the normal options follow.")
            elif stream(pip + [wheel], log_path=log, max_seconds=PIP_MAX_SECONDS) == 0 and functional()[0]:
                return "provided wheel"
            else:
                say("The wheel you provided did not install or does not work on this runtime and GPU (the wrong Python, torch, CUDA or GPU "
                    "architecture is the usual reason); trying the normal options instead.")
        except (urllib.error.URLError, OSError) as exc:
            say(f"Could not fetch the wheel you provided ({exc}); trying the normal options instead.")
    cached = find_cached_wheel(lay, env)
    if cached:
        say(f"Cached wheel for this runtime on Drive: {os.path.basename(cached)}  (expected: seconds to a minute)")
        if stream(pip + [cached], log_path=log) == 0 and functional()[0]:
            return "cached wheel"
        say("The cached wheel did not work on this GPU; marking it rejected and trying the next option.")
        _reject_wheel(cached)
    name = prebuilt_wheel_name(env)
    if name and exists(FLASH_URL.format(name=name.replace("+", "%2B"))):
        say(f"Prebuilt wheel from the FlashAttention release {FLASH_TAG}: {name}  (expected: a minute or two)")
        os.makedirs(wheel_dir(lay, env), exist_ok=True)
        dest = os.path.join(wheel_dir(lay, env), name)
        try:
            download(FLASH_URL.format(name=name.replace("+", "%2B")), dest)
            if stream(pip + [dest], log_path=log) == 0 and functional()[0]:
                _write_wheel_meta(dest, env, f"prebuilt release {FLASH_TAG}")
                return "prebuilt wheel"
        except (urllib.error.URLError, OSError) as exc:
            say(f"Download failed ({exc}); will build instead.")
        say("The prebuilt wheel did not work; falling back to a build.")
    else:
        say("No prebuilt wheel for this runtime "
            f"(python {env.get('python')}, torch {env.get('torch_mm')}, {env.get('cuda_tag')}); the {FLASH_TAG} release covers Linux x86_64, CUDA 12, torch 2.4-2.8.")
    if not p.build_flash_attn:
        here = f"python {env.get('python')}, torch {env.get('torch_mm')}, {env.get('cuda_tag')}"
        if p.attn_fallback is False:
            raise StepFailed(f"ATTN_FALLBACK=False needs FlashAttention 2, but there is no ready-made package for this runtime ({here}). "
                             "Set ATTN_FALLBACK='auto' to use PyTorch's built-in attention instead, or BUILD_FLASH_ATTN=True to compile it "
                             "(an hour or more, it may fail, and compute units are used the whole time).")
        say("FlashAttention 2 has no ready-made package for this runtime, and it is NOT compiled here: that can take an hour or more, may fail "
            "on a new Python/CUDA combination, and every minute uses compute units.")
        say("Training will use PyTorch's built-in SDPA attention through a patched COPY of Qwen's script (the cached repo is not touched). "
            "UNVERIFIED: not run by the author. To compile FlashAttention anyway, set BUILD_FLASH_ATTN = True.")
        return "not available (using SDPA)"
    say(f"BUILD_FLASH_ATTN is on: building FlashAttention, stopped after {BUILD_MAX_SECONDS // 60} minutes if it has not finished.")
    say("Building FlashAttention from source ONCE; the wheel is then saved to the Drive wheelhouse for this runtime type.")
    say("Expected time: commonly tens of minutes, and over an hour on a few CPU cores (an estimate; not measured here).")
    if shutil.which("nvcc") is None and not os.path.exists("/usr/local/cuda/bin/nvcc"):
        raise StepFailed("nvcc (the CUDA compiler) was not found, so FlashAttention cannot be built on this runtime.")
    if stream([py(), "-m", "pip", "install", "-q", "ninja", "packaging", "wheel", "setuptools"], log_path=log) != 0:
        raise StepFailed("Could not install the build tools (ninja, packaging, wheel, setuptools).")
    os.makedirs(wheel_dir(lay, env), exist_ok=True)
    out_dir = os.path.join(wheel_dir(lay, env), f"build-{int(time.time())}")
    os.makedirs(out_dir)
    code = stream([py(), "-m", "pip", "wheel", "flash-attn", "--no-build-isolation", "--no-deps", "-w", out_dir,
                   "--cache-dir", lay.pip_cache], env={"MAX_JOBS": str(build_jobs())}, log_path=log, max_seconds=BUILD_MAX_SECONDS)
    wheels = [f for f in os.listdir(out_dir) if f.endswith(".whl")]
    if code != 0 or not wheels:
        raise StepFailed("The FlashAttention build failed or ran past its time limit; see the log. Set BUILD_FLASH_ATTN = False to use PyTorch's "
                         "built-in attention instead. A T4 or other pre-Ampere GPU cannot use FlashAttention at all.")
    final = os.path.join(wheel_dir(lay, env), wheels[0])
    os.replace(os.path.join(out_dir, wheels[0]), final)
    shutil.rmtree(out_dir, ignore_errors=True)  # the empty build folder we just made
    _write_wheel_meta(final, env, "built here")
    if stream(pip + [final], log_path=log) != 0 or not functional()[0]:
        raise StepFailed("The built FlashAttention wheel was saved but does not work on this GPU; see the log.")
    return "built"


# --------------------------------------------------------------------------------------------- setup
def wheel_basename(path_or_url: str) -> str:
    """The file name of a wheel given as a path or URL, with %2B and friends decoded (pip rejects a wheel named ...%2Bcu130...)."""
    return urllib.parse.unquote(os.path.basename((path_or_url or "").split("?")[0]))


def wheel_fits(name: str, env: dict) -> tuple:
    """(ok, why) for a flash_attn wheel FILE NAME against this runtime: Python, torch, CUDA, platform. Reads the name only."""
    m = re.match(r"flash_attn-[^-]+?(?:\+cu(\d+)torch(\d+\.\d+))?-(cp\d+)-cp\d+-(\w+)\.whl$", wheel_basename(name))
    if not m:
        return True, "unrecognised file name, so it will be tried"
    cu, torch_mm, py, plat = m.groups()
    problems = []
    if py != env.get("python"):
        problems.append(f"built for Python {py[2]}.{py[3:]}, this runtime has {env.get('python_version')}")
    if torch_mm and env.get("torch_mm") and torch_mm != env["torch_mm"]:
        problems.append(f"built for torch {torch_mm}, this runtime has {env['torch']}")
    if cu and env.get("cuda") and cu != env["cuda"].replace(".", "")[:len(cu)]:
        problems.append(f"built for CUDA {cu[:2]}.{cu[2:]}, this runtime has {env['cuda']}")
    if plat.startswith("linux") and env.get("platform", "linux").startswith("win"):
        problems.append("built for Linux")
    return (not problems), "; ".join(problems) or "matches this runtime"


PIP_MAX_SECONDS = 25 * 60  # any single pip step is stopped after this long
TESTED_PYTHON = ("cp310", "cp311", "cp312")  # what the Qwen3-TTS package and FlashAttention wheels were checked against by their authors


VENV_DEFAULT = {"python": "3.12", "torch": "2.8.0", "cuda": "cu126"}  # the combination FlashAttention's official wheels exist for
VENV_READY = ".voicestudio_ready"


def venv_dir(p: Params) -> str:
    return os.path.join(p.local_root, "venv")


def venv_python_path(p: Params) -> str:
    return os.path.join(venv_dir(p), "Scripts" if os.name == "nt" else "bin", "python.exe" if os.name == "nt" else "python")


def use_existing_venv(p: Params) -> bool:
    """Switch every step to the isolated environment when this runtime already has a finished one (survives a kernel restart)."""
    if p.use_venv is not False and os.path.isfile(os.path.join(venv_dir(p), VENV_READY)) and os.path.isfile(venv_python_path(p)):
        PYTHON[0] = venv_python_path(p)
        return True
    PYTHON[0] = sys.executable
    return False


def wheel_tags(name: str) -> Optional[dict]:
    m = re.match(r"flash_attn-[^-]+?(?:\+cu(\d+)torch(\d+\.\d+))?-cp(\d)(\d+)-cp\d+-", wheel_basename(name))
    if not m:
        return None
    cu, torch_mm, major, minor = m.groups()
    return {"python": f"{major}.{minor}", "torch": f"{torch_mm}.*" if torch_mm else VENV_DEFAULT["torch"], "cuda": f"cu{cu}" if cu else VENV_DEFAULT["cuda"]}


def want_venv(p: Params, lay: Layout, env: dict, gpu: Optional[Gpu], *, exists: Callable = None, functional: Callable = None) -> Optional[dict]:
    """The isolated environment to build ({python, torch, cuda}) or None. In auto mode only when FlashAttention 2 would otherwise be
    unavailable on this GPU. A provided wheel that does not fit this runtime decides the versions."""
    exists = exists or url_exists
    functional = functional or flash_functional
    if p.use_venv is False or (gpu is None and p.use_venv != "auto"):
        return None
    tags = wheel_tags(p.flash_wheel) if p.flash_wheel else None
    if p.use_venv is True:
        return tags or dict(VENV_DEFAULT)
    if gpu is None or not gpu.flash_ok or p.attn_fallback is True:
        return None  # Blackwell and pre-Ampere GPUs cannot use FlashAttention 2 whatever the environment
    if functional()[0] or find_cached_wheel(lay, env):
        return None
    if p.flash_wheel and wheel_fits(p.flash_wheel, env)[0]:
        return None
    name = prebuilt_wheel_name(env)
    if name and exists(FLASH_URL.format(name=name.replace("+", "%2B"))):
        return None
    return tags or dict(VENV_DEFAULT)


def ensure_venv(p: Params, lay: Layout, target: dict, log: str, *, stream: Callable = run_stream) -> None:
    """Python + PyTorch in a fresh environment under /content (nothing in the system Python is touched). Needs uv (installed if absent)."""
    say(f"== Isolated environment: Python {target['python']}, torch {target['torch']}, {target['cuda']}")
    if use_existing_venv(p):
        say("An isolated environment from this session is already there; reusing it.")
        return
    shutil.rmtree(venv_dir(p), ignore_errors=True)  # only a half-built one of ours under /content, never anything on Drive
    uv = [shutil.which("uv")] if shutil.which("uv") else None
    if uv is None:
        say("Installing uv (a fast Python installer).")
        if stream([sys.executable, "-m", "pip", "install", "-q", "uv"], log_path=log, max_seconds=PIP_MAX_SECONDS) != 0:
            raise StepFailed("Could not install uv; see the log.")
        uv = [sys.executable, "-m", "uv"]
    with timed("isolated environment: Python"):
        if stream(uv + ["venv", "--seed", "--python", target["python"], venv_dir(p)], log_path=log, max_seconds=PIP_MAX_SECONDS) != 0:
            raise StepFailed(f"Could not create a Python {target['python']} environment; see the log.")
    spec = target["torch"]
    index = f"https://download.pytorch.org/whl/{target['cuda']}"
    with timed("isolated environment: PyTorch"):
        if stream(uv + ["pip", "install", "--python", venv_python_path(p), f"torch=={spec}", f"torchaudio=={spec}", "--index-url", index],
                  log_path=log, max_seconds=PIP_MAX_SECONDS) != 0:
            raise StepFailed(f"Could not install torch {spec} for {target['cuda']} from {index}; see the log.")
    code, out = capture([venv_python_path(p), "-c", "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"])
    say(f"Isolated environment torch check: {out.strip()}")
    if code != 0 or not out.strip().endswith("True"):
        raise StepFailed("The isolated environment's PyTorch cannot see the GPU: " + out.strip()[-300:])
    os.makedirs(venv_dir(p), exist_ok=True)
    with open(os.path.join(venv_dir(p), VENV_READY), "w", encoding="utf-8") as fh:
        fh.write(json.dumps(target) + "\n")
    PYTHON[0] = venv_python_path(p)


def runtime_check(p: Params, lay: Layout, *, env: Optional[dict] = None, gpu: Optional[Gpu] = None,
                  exists: Callable = None, functional: Callable = None) -> bool:
    """The first thing setup does, in seconds: Python, PyTorch and CUDA of this runtime and what they mean for the run, BEFORE anything
    slow starts. False means stop now. Nothing is installed, downloaded or compiled here."""
    exists = exists or url_exists
    functional = functional or flash_functional
    env = env or detect_environment()
    gpu = gpu if gpu is not None else detect_gpu()
    say("== Runtime check (seconds; nothing is installed, downloaded or compiled yet)")
    rows = [("python", "INFO", env.get("python_version") or "unknown")]
    rows.append(("torch / CUDA", "PASS" if env.get("torch") else "FAIL",
                 f"torch {env['torch']}, CUDA {env.get('cuda') or 'none'}" if env.get("torch") else "PyTorch is not installed in this runtime"))
    rows.append(("gpu", "PASS" if gpu else "FAIL", f"{gpu.name}, {gpu.vram_gb:g} GB, compute capability {env.get('gpu_cc')}" if gpu
                 else "no GPU: Runtime > Change runtime type > pick the GPU"))
    if p.flash_wheel:
        fits, why_not = wheel_fits(p.flash_wheel, env)
        rows.append(("your FLASH_WHEEL", "PASS" if fits else "INFO", why_not if fits else "will NOT be used: " + why_not))
    notes = []
    if env.get("python") not in TESTED_PYTHON:
        notes.append(f"Python {env.get('python_version')} is newer than the 3.10-3.12 these packages were built for")
    if env.get("cuda_tag") and env["cuda_tag"] != "cu12":
        notes.append(f"CUDA {env.get('cuda')} is newer than the CUDA 12 that FlashAttention packages exist for")
    if notes:
        rows.append(("newer runtime", "INFO", "; ".join(notes) + ". Expect no ready-made FlashAttention package (PyTorch's built-in attention is used, "
                     "nothing is compiled). If package installs fail below, check Runtime > Change runtime type for an older runtime version."))
    if gpu is not None:
        path, why = attention_choice(gpu, p.attn_fallback)
        if path == "flash2":
            if functional()[0]:
                rows.append(("FlashAttention 2", "PASS", "already installed and working"))
            elif find_cached_wheel(lay, env):
                rows.append(("FlashAttention 2", "PASS", "a cached wheel for this exact runtime is on Drive"))
            elif prebuilt_wheel_name(env) and exists(FLASH_URL.format(name=prebuilt_wheel_name(env).replace("+", "%2B"))):
                rows.append(("FlashAttention 2", "PASS", "a ready-made package exists (a minute or two)"))
            elif p.build_flash_attn:
                rows.append(("FlashAttention 2", "INFO", f"no ready-made package; BUILD_FLASH_ATTN is on, so it WILL be compiled (up to {BUILD_MAX_SECONDS // 60} minutes)"))
            elif p.attn_fallback is False:
                rows.append(("FlashAttention 2", "FAIL", "no ready-made package and ATTN_FALLBACK=False forbids the fallback: set ATTN_FALLBACK='auto' or BUILD_FLASH_ATTN=True"))
            else:
                rows.append(("FlashAttention 2", "INFO", "no ready-made package for this runtime: PyTorch SDPA attention will be used through a patched "
                             "COPY of Qwen's script (unverified). Nothing is compiled."))
        else:
            rows.append(("attention", "INFO" if path == "sdpa" else "FAIL", why))
    for name, url in (("github.com", "https://github.com"), ("huggingface.co", "https://huggingface.co")):
        ok = exists(url)
        rows.append((name, "PASS" if ok else "FAIL", "reachable" if ok else "not reachable from this runtime (needed to fetch code and models)"))
    ok = check_table(rows)
    say(f"Time limits: any pip step stops after {PIP_MAX_SECONDS // 60} minutes; a requested FlashAttention build after {BUILD_MAX_SECONDS // 60}.")
    return ok


def ensure_deps(p: Params, lay: Layout, log: str, *, probe: Callable = probe_module, stream: Callable = run_stream) -> None:
    say("== Python packages")
    base = [py(), "-m", "pip", "install", "-q", "--cache-dir", lay.pip_cache]
    if probe("qwen_tts")[0] and os.path.isfile(os.path.join(lay.local_repo, SOURCE_MARKER)):
        say("qwen_tts is already importable in this runtime (skipping the editable install).")
    elif stream(base + ["-e", lay.local_repo], log_path=log, max_seconds=PIP_MAX_SECONDS) != 0:
        raise StepFailed("pip install -e of the Qwen3-TTS repository failed or ran past its time limit; see the log. If this runtime is much newer "
                         "than Python 3.12 / CUDA 12, try an older runtime version under Runtime > Change runtime type.")
    missing = [m for m in ("accelerate", "tensorboard", "librosa", "soundfile", "safetensors", "huggingface_hub") if not probe(m)[0]]
    if missing:
        say("Installing: " + ", ".join(missing))
        if stream(base + [m.replace("_", "-") for m in missing], log_path=log, max_seconds=PIP_MAX_SECONDS) != 0:
            raise StepFailed("pip install of " + ", ".join(missing) + " failed; see the log.")
    else:
        say("accelerate, tensorboard, librosa, soundfile, safetensors, huggingface_hub: already present.")


def setup(p: Params) -> bool:
    """Layout, environment fingerprint, dependencies, models. Prints a pass/fail table; True when everything passed."""
    say(f"VoiceStudio Qwen3-TTS Colab runner {RUNNER_VERSION}; Qwen3-TTS commit {QWEN_COMMIT[:10]} ({QWEN_COMMIT_DATE}); not run on Colab by its author.")
    say("Setup time is billed like any other runtime time; the table at the end shows where it went.")
    lay = ensure_layout(p, create_run=False)
    log = new_log(lay, p, "setup")
    use_existing_venv(p)
    with timed("runtime check"):
        if not runtime_check(p, lay):
            raise StepFailed("The runtime check failed (see the table above). Nothing slow was started and no compute units were spent on installs.")
    gpu = gpu_report(p)
    target = want_venv(p, lay, detect_environment(), gpu)
    if target and not os.path.isfile(os.path.join(venv_dir(p), VENV_READY)):
        say("This runtime has no FlashAttention package, so training runs in an isolated environment that has one (one-time, a few minutes).")
        try:
            ensure_venv(p, lay, target, log)
        except StepFailed as exc:
            if p.use_venv is True:
                raise
            PYTHON[0] = sys.executable
            say(f"The isolated environment could not be built ({exc}). Continuing in this runtime with PyTorch's built-in attention (patched copy, unverified).")
    with timed("environment fingerprint"):
        env = detect_environment()
        seen = record_fingerprint(lay, env)
    say(f"Runtime fingerprint: python {env['python']}, torch {env['torch']}, {env['cuda_tag']}, abi {env['cxx11abi']}, GPU {env['gpu']} ({env['gpu_cc']})"
        + ("  -> this exact runtime type was seen before." if seen else "  -> new runtime type for this cache."))
    with timed("Qwen3-TTS repository (clone/copy)"):
        ensure_repo(lay)
    with timed("torch companion packages"):
        repair_torch_companions(p, lay, log)
    with timed("pip packages"):
        ensure_deps(p, lay, log)
    with timed("FlashAttention (wheel handling)"):
        how = ensure_flash_attn(p, lay, env, gpu, log)
    say(f"FlashAttention: {how}")
    with timed("models (download + local copy)"):
        ensure_models(p, lay)
    with timed("dependency check"):
        ok = check_table(dependency_rows(p, lay, env, gpu))
    timing_summary(lay, "Setup", "setup")
    return ok


# --------------------------------------------------------------------------------------------- dataset
def _wav_info(path: str) -> tuple:
    """(sample_rate, channels, seconds) for a PCM WAV, using the standard library; soundfile if the format needs it."""
    try:
        with wave.open(path, "rb") as w:
            return w.getframerate(), w.getnchannels(), w.getnframes() / float(w.getframerate() or 1)
    except (wave.Error, EOFError):
        try:
            import soundfile as sf

            i = sf.info(path)
            return i.samplerate, i.channels, i.duration
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"not a readable WAV ({type(exc).__name__})") from exc


def verify_dataset(folder: str) -> dict:
    """Check every audio and ref_audio path of train.jsonl: relative, inside the folder, present, 24 kHz mono."""
    problems: list = []
    counts = {"rows": 0, "audio": 0, "seconds": 0.0, "reference_files": 0}
    jsonl = os.path.join(folder, "train.jsonl")
    if not os.path.isfile(jsonl):
        return {"ok": False, "problems": [f"train.jsonl not found in {folder}"], **counts}
    root = os.path.realpath(folder)
    refs: dict = {}

    def check_path(rel, row_no, field):
        if not isinstance(rel, str) or not rel:
            problems.append(f"row {row_no}: '{field}' is missing")
            return None
        if os.path.isabs(rel) or re.match(r"^[A-Za-z]:[\\/]", rel) or rel.startswith(("\\", "/")):
            problems.append(f"row {row_no}: '{field}' is an absolute path ({rel}); Qwen's scripts open paths relative to the working folder")
            return None
        full = os.path.realpath(os.path.join(folder, *re.split(r"[\\/]", rel)))
        if not (full == root or full.startswith(root + os.sep)):
            problems.append(f"row {row_no}: '{field}' leaves the dataset folder ({rel})")
            return None
        if not os.path.isfile(full):
            problems.append(f"row {row_no}: '{field}' file does not exist ({rel} -> {full})")
            return None
        try:
            rate, channels, seconds = _wav_info(full)
        except ValueError as exc:
            problems.append(f"row {row_no}: '{field}' {rel}: {exc}")
            return None
        if rate != SAMPLE_RATE or channels != 1:
            problems.append(f"row {row_no}: '{field}' {rel} is {rate} Hz, {channels} channel(s); needs {SAMPLE_RATE} Hz mono")
            return None
        return seconds

    with open(jsonl, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                problems.append(f"row {n}: not valid JSON")
                continue
            counts["rows"] += 1
            if not str(row.get("text") or "").strip():
                problems.append(f"row {n}: 'text' is empty")
            seconds = check_path(row.get("audio"), n, "audio")
            if seconds is not None:
                counts["audio"] += 1
                counts["seconds"] += seconds
            ref = row.get("ref_audio")
            if ref not in refs:
                refs[ref] = check_path(ref, n, "ref_audio")
    counts["reference_files"] = sum(1 for v in refs.values() if v is not None)
    if counts["rows"] == 0:
        problems.append("train.jsonl has no rows")
    return {"ok": not problems, "problems": problems, "rows": counts["rows"], "audio": counts["audio"],
            "seconds": round(counts["seconds"], 1), "reference_files": counts["reference_files"]}


def _signature(path: str) -> str:
    st = os.stat(path)
    return f"{os.path.abspath(path)}|{st.st_size}|{int(st.st_mtime)}"


def _find_source(src: str) -> tuple:
    """(kind, path) with kind 'zip' or 'dir'. A folder without train.jsonl may hold exactly one dataset below it."""
    if not src:
        raise StepFailed("DATA_SOURCE is empty. Set it to the dataset folder or .zip (for example "
                         "/content/drive/MyDrive/VoiceStudio-training/datasets/<name>.zip).")
    if os.path.isfile(src):
        if src.lower().endswith(".zip"):
            return "zip", src
        raise StepFailed(f"DATA_SOURCE is a file but not a .zip: {src}")
    if not os.path.isdir(src):
        raise StepFailed(f"DATA_SOURCE does not exist: {src}\nIs Drive mounted, and is the path spelled exactly as in Drive?")
    if os.path.isfile(os.path.join(src, "train.jsonl")):
        return "dir", src
    found = []
    for name in sorted(os.listdir(src)):
        full = os.path.join(src, name)
        if os.path.isdir(full) and os.path.isfile(os.path.join(full, "train.jsonl")):
            found.append(("dir", full))
        elif os.path.isfile(full) and name.lower().endswith(".zip"):
            found.append(("zip", full))
    if len(found) == 1:
        return found[0]
    if not found:
        raise StepFailed(f"No train.jsonl, dataset folder or .zip found in {src}.")
    raise StepFailed("DATA_SOURCE holds several datasets; point it at one:\n  " + "\n  ".join(f[1] for f in found))


def _safe_extract(archive: str, dest: str) -> None:
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            norm = os.path.normpath(member)
            if os.path.isabs(member) or norm.startswith("..") or ".." in norm.split(os.sep):
                raise StepFailed(f"The zip holds an unsafe path ({member}); not extracting it.")
        zf.extractall(dest)


def prepare_dataset(p: Params, lay: Optional[Layout] = None) -> str:
    """Put the dataset on local disk (copy or unzip, skipped when already done) and return the folder."""
    kind, src = _find_source(p.data_source)
    local_root = os.path.realpath(p.local_root)
    if kind == "dir" and os.path.realpath(src).startswith(local_root + os.sep):
        say(f"Dataset: using {src} in place (already on local disk).")
        return src
    name = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.splitext(os.path.basename(src.rstrip("/\\")))[0] if kind == "zip"
                  else os.path.basename(os.path.normpath(src))) or "dataset"
    target = os.path.join(p.local_root, name)
    sig = _signature(src) if kind == "zip" else os.path.abspath(src)
    marker = os.path.join(target, SOURCE_MARKER)
    if os.path.isdir(target):
        if os.path.isfile(marker) and open(marker, encoding="utf-8").read().strip() == sig:
            say(f"Dataset: {target} is already up to date (not copied again).")
            return _descend(target)
        if not os.path.isfile(marker):
            raise StepFailed(f"{target} exists and was not made by this script; choose another DATA_SOURCE name or remove it.")
        shutil.rmtree(target)  # our own earlier copy of a source that has changed
    say(f"Dataset: {'unzipping' if kind == 'zip' else 'copying'} {src} to {target} (local disk is much faster than Drive).")
    tmp = target + ".partial"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(p.local_root, exist_ok=True)
    if kind == "zip":
        os.makedirs(tmp)
        _safe_extract(src, tmp)
    else:
        shutil.copytree(src, tmp)
    os.replace(tmp, target)
    with open(marker, "w", encoding="utf-8") as fh:
        fh.write(sig + "\n")
    return _descend(target)


def _descend(folder: str) -> str:
    if os.path.isfile(os.path.join(folder, "train.jsonl")):
        return folder
    subs = [os.path.join(folder, d) for d in sorted(os.listdir(folder)) if os.path.isfile(os.path.join(folder, d, "train.jsonl"))]
    if len(subs) == 1:
        return subs[0]
    raise StepFailed(f"train.jsonl not found in {folder}.")


def _data_check(p: Params) -> tuple:
    say("== Dataset")
    with timed("dataset copy or unzip", "data"):
        folder = prepare_dataset(p)
    with timed("dataset verification", "data"):
        result = verify_dataset(folder)
    with timed("dataset hash", "data"):
        result["dataset_hash"] = compute_dataset_hash(folder)
    say(f"{result['rows']} rows, {result['audio']} audio clips verified, {result['reference_files']} reference file(s), "
        f"{result['seconds'] / 60:.1f} minutes ({result['seconds']:.0f} s) of training audio.")
    if not result["ok"]:
        shown = result["problems"][:15]
        raise StepFailed(f"{len(result['problems'])} problem(s) in the dataset; nothing was trained:\n  " + "\n  ".join(shown)
                         + ("\n  ..." if len(result["problems"]) > len(shown) else ""))
    say("Every audio and ref_audio path exists and is 24 kHz mono. Dataset OK.")
    return folder, result


def data_check(p: Params) -> str:
    """Prepare the local dataset and verify it before any GPU time is spent. Returns the local folder."""
    folder, _result = _data_check(p)
    timing_summary(Layout(p), "Dataset", "data")
    return folder


# --------------------------------------------------------------------------------------------- dataset manifest
DRIVER_VERSION = "1"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_files(folder: str) -> list:
    """The files the training depends on: train.jsonl, reference/ref.wav and every file under audio/ (relative, sorted)."""
    rels = [r for r in ("train.jsonl", "reference/ref.wav") if os.path.isfile(os.path.join(folder, *r.split("/")))]
    audio = os.path.join(folder, "audio")
    if os.path.isdir(audio):
        rels += ["audio/" + f for f in sorted(os.listdir(audio)) if os.path.isfile(os.path.join(audio, f))]
    return rels


def compute_dataset_hash(folder: str) -> str:
    """One deterministic hash of the data: sha256 over 'relative path NUL file sha256 newline' for the sorted files."""
    h = hashlib.sha256()
    for rel in dataset_files(folder):
        h.update(rel.encode("utf-8") + b"\0" + sha256_file(os.path.join(folder, *rel.split("/"))).encode("ascii") + b"\n")
    return h.hexdigest()


# --------------------------------------------------------------------------------------------- status, logs, metrics
def run_log(lay: Layout, msg: str) -> None:
    """Append a line to the run log on Drive (fatal errors go here before the script stops)."""
    try:
        os.makedirs(lay.run_logs, exist_ok=True)
        with open(lay.runlog, "a", encoding="utf-8") as fh:
            fh.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")
    except OSError:
        pass


class Status:
    """status.json in the run folder on Drive: status, epoch, step, latest loss, latest checkpoint, last update."""

    def __init__(self, lay: Layout, throttle: float = 20.0):
        self.lay, self.throttle, self._last = lay, throttle, 0.0
        self.data = {"status": "created", "epoch": None, "step": None, "loss": None, "latest_checkpoint": None, "error": None,
                     "run_id": lay.run_id, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "updated": None}

    def update(self, force: bool = True, **kw) -> None:
        self.data.update(kw)
        now = time.time()
        if not force and now - self._last < self.throttle:
            return
        self._last = now
        self.data["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            write_json_atomic(self.lay.status, self.data)
        except OSError:
            pass


def fail_run(lay: Layout, status: Optional[Status], message: str) -> None:
    run_log(lay, "FATAL: " + message)
    if status is not None:
        status.update(status="failed", error=message[:2000])


# --------------------------------------------------------------------------------------------- training observability
LOSS_RE = re.compile(r"Epoch\s+(\d+)\s*\|\s*Step\s+(\d+)\s*\|\s*Loss:\s*([-+0-9.eE]+|nan|inf)")


class TrainMonitor:
    """Parses sft_12hz.py's output ('Epoch e | Step n | Loss: x', printed every 10 batches) into a progress line."""

    def __init__(self, *, epochs: int, batches_per_epoch: int, clock: Callable = time.time):
        self.epochs, self.batches, self.clock = epochs, max(1, batches_per_epoch), clock
        self.events: list = []  # (time, epoch, step, loss)
        self.steps: dict = {}  # epoch-0 step -> time, for the timing run
        self.oom, self.start, self.end, self.last_line = False, None, None, clock()
        self.last_step = (0, 0)

    def feed(self, line: str) -> Optional[dict]:
        now = self.clock()
        self.last_line = now
        low = line.lower()
        if "out of memory" in low or "outofmemoryerror" in low:
            self.oom = True
        m = LOSS_RE.search(line)
        if not m:
            return None
        epoch, step = int(m.group(1)), int(m.group(2))
        try:
            loss = float(m.group(3))
        except ValueError:
            loss = float("nan")
        self.events.append((now, epoch, step, loss))
        self.last_step = (epoch, step)
        if epoch == 0:
            self.steps.setdefault(step, now)
        return self.progress(now)

    def progress(self, now: Optional[float] = None) -> dict:
        now, (epoch, step) = (self.clock() if now is None else now), self.last_step
        recent = [e for e in self.events if e[1] == epoch][-10:]
        rate = None
        if len(recent) >= 2 and recent[-1][0] > recent[0][0]:
            rate = (recent[-1][2] - recent[0][2]) / (recent[-1][0] - recent[0][0])
        losses = [e[3] for e in self.events[-10:]]
        done_batches = epoch * self.batches + step
        total = self.epochs * self.batches
        eta_epoch = (self.batches - step) / rate if rate else None
        eta_all = (total - done_batches) / rate if rate else None
        return {"epoch": epoch, "epochs": self.epochs, "step": step, "batches": self.batches, "loss": self.events[-1][3],
                "avg_loss": sum(losses) / len(losses), "steps_per_s": rate, "elapsed": now - (self.start or now),
                "eta_epoch": eta_epoch, "eta_all": eta_all}


def format_progress(pr: dict, mem: Optional[tuple]) -> str:
    rate = f"{pr['steps_per_s']:.2f} step/s" if pr["steps_per_s"] else "-- step/s"
    pct = 100 * min(1.0, (pr["step"] + 1) / pr["batches"])
    text = (f"epoch {pr['epoch'] + 1}/{pr['epochs']}  step {pr['step']}/{pr['batches']} ({pct:.0f}%)  loss {pr['loss']:.4f}  avg10 {pr['avg_loss']:.4f}  "
            f"{rate}  elapsed {fmt_secs(pr['elapsed'])}")
    if pr["eta_epoch"] is not None:
        text += f"  epoch ETA {fmt_secs(pr['eta_epoch'])}  overall ETA {fmt_secs(pr['eta_all'])} (excludes checkpoint saves)"
    if mem and mem[0]:
        text += f"  GPU {mem[0] / 1024:.1f} GB in use (peak {mem[1] / 1024:.1f}; nvidia-smi counts PyTorch's cache as used)"
    return text


class GpuPeak:
    """Polls nvidia-smi for the memory in use on the selected GPU."""

    def __init__(self, sample: Callable, interval: float = 2.0):
        self.sample, self.interval, self.peak_mb, self.last_mb, self._stop = sample, interval, 0.0, 0.0, threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            v = self.sample()
            if v:
                self.last_mb, self.peak_mb = v, max(self.peak_mb, v)
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> float:
        self._stop.set()
        self._thread.join(timeout=5)
        return self.peak_mb


def gpu_memory_used_mb() -> Optional[float]:
    code, out = capture(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    try:
        return float(out.strip().splitlines()[0]) if code == 0 else None
    except (ValueError, IndexError):
        return None


class Heartbeat:
    """Prints a line when the training output has been quiet for a while (no keep-alive tricks: it only reports)."""

    def __init__(self, mon: TrainMonitor, peak: GpuPeak, every: float, clock: Callable = time.time):
        self.mon, self.peak, self.every, self.clock = mon, peak, max(5.0, float(every)), clock
        self._stop, self._thread, self._last_beat = threading.Event(), threading.Thread(target=self._loop, daemon=True), clock()

    def beat(self) -> Optional[str]:
        now = self.clock()
        quiet = now - self.mon.last_line
        if quiet >= self.every and now - self._last_beat >= self.every:
            self._last_beat = now
            e, s = self.mon.last_step
            msg = (f"[heartbeat] still running; no output for {fmt_secs(quiet)}; last seen epoch {e + 1} step {s}; "
                   f"GPU memory {self.peak.last_mb / 1024:.1f} GB in use. (An epoch end copies a full model folder, which is quiet.)")
            say(msg)
            return msg
        return None

    def _loop(self) -> None:
        while not self._stop.wait(min(30.0, self.every)):
            self.beat()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


def append_metrics(lay: Layout, pr: dict, mem_mb: float) -> None:
    new = not os.path.isfile(lay.metrics)
    try:
        with open(lay.metrics, "a", encoding="utf-8") as fh:
            if new:
                fh.write("time,epoch,step,loss,avg_loss10,steps_per_s,elapsed_s,gpu_mem_used_mb\n")
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{pr['epoch']},{pr['step']},{pr['loss']:.6g},{pr['avg_loss']:.6g},"
                     f"{pr['steps_per_s'] or ''},{pr['elapsed']:.1f},{mem_mb:.0f}\n")
    except OSError:
        pass


# --------------------------------------------------------------------------------------------- checkpoints: local first, then safe sync to Drive
def existing_checkpoints(folder: str) -> list:
    if not os.path.isdir(folder):
        return []
    items = [d for d in os.listdir(folder) if CHECKPOINT_RE.fullmatch(d) and os.path.isdir(os.path.join(folder, d))]
    return sorted(items, key=lambda d: int(CHECKPOINT_RE.fullmatch(d).group(1)))


def dir_size(path: str) -> int:
    total = 0
    for base, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(base, f))
            except OSError:
                pass
    return total


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return str(n)


SMALL_FILE = 32 * 2**20  # files below this are compared by sha256 as well as by size when syncing


def copy_verified(src: str, dst: str) -> int:
    """Copy folder src to dst under a temporary name, verify sizes (and hashes of small files), mark it, then rename.
    Never leaves a half-valid dst. Returns the bytes copied. dst must not exist."""
    tmp = dst + ".tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)  # a leftover of an interrupted copy of ours (".tmp" suffix)
    shutil.copytree(src, tmp)
    total = 0
    for base, _dirs, files in os.walk(src):
        for f in files:
            a = os.path.join(base, f)
            b = os.path.join(tmp, os.path.relpath(a, src))
            if not os.path.isfile(b) or os.path.getsize(a) != os.path.getsize(b):
                shutil.rmtree(tmp, ignore_errors=True)
                raise StepFailed(f"Checkpoint copy verification failed: {os.path.relpath(a, src)} differs in size after copying to Drive.")
            if os.path.getsize(a) < SMALL_FILE and sha256_file(a) != sha256_file(b):
                shutil.rmtree(tmp, ignore_errors=True)
                raise StepFailed(f"Checkpoint copy verification failed: {os.path.relpath(a, src)} differs in content after copying to Drive.")
            total += os.path.getsize(a)
    with open(os.path.join(tmp, DONE_MARKER), "w", encoding="utf-8") as fh:
        fh.write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    os.replace(tmp, dst)
    return total


def safetensors_complete(path: str) -> bool:
    """True only when the file's size matches its own header (8-byte length, JSON header, tensor data): a half-written file fails."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            n = int.from_bytes(fh.read(8), "little")
            if n <= 0 or n > size:
                return False
            header = json.loads(fh.read(n))
        end = max([v["data_offsets"][1] for k, v in header.items() if k != "__metadata__"] or [0])
        return size == 8 + n + end
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False


def checkpoint_finished(folder: str) -> bool:
    """The official script copies the BASE model folder first (a complete base model.safetensors and a 'base' config), then rewrites config.json
    with tts_model_type = custom_voice, then writes the trained model.safetensors last. A checkpoint is finished only when all of that
    is true and the weights are newer than the config, so the copied base weights are never mistaken for a trained epoch."""
    cfg, model = os.path.join(folder, "config.json"), os.path.join(folder, "model.safetensors")
    if not (os.path.isfile(cfg) and os.path.isfile(model)):
        return False
    try:
        with open(cfg, encoding="utf-8") as fh:
            if json.load(fh).get("tts_model_type") != "custom_voice":
                return False
        return os.path.getmtime(model) >= os.path.getmtime(cfg) and safetensors_complete(model)
    except (OSError, ValueError, AttributeError):
        return False


class CheckpointSyncer:
    """Training writes checkpoint-epoch-N to local /content; this copies each FINISHED one to Drive (temporary name, verify, rename)."""

    def __init__(self, p: Params, lay: Layout, status: Optional[Status] = None, clock: Callable = time.time, stable_seconds: float = 6.0):
        self.p, self.lay, self.status, self.clock, self.stable = p, lay, status, clock, stable_seconds
        self.seen: dict = {}  # name -> (size of model.safetensors, time first seen at that size)
        self.synced: list = []
        self.not_finished: list = []  # local checkpoint folders that never looked complete (left in /content, not copied)
        self.first_bytes: Optional[int] = None
        self._stop, self._thread, self.error = threading.Event(), threading.Thread(target=self._loop, daemon=True), None
        self._lock = threading.Lock()

    def manifest(self) -> dict:
        return read_json(self.lay.ckpt_manifest, {"checkpoints": {}})

    def _ready(self, name: str, final: bool) -> bool:
        d = os.path.join(self.lay.local_checkpoints, name)
        model = os.path.join(d, "model.safetensors")
        if not checkpoint_finished(d):
            if final and os.path.isdir(d) and name not in self.not_finished:
                self.not_finished.append(name)
            return False  # sft_12hz.py writes model.safetensors last
        if final:
            return True
        size, now = os.path.getsize(model), self.clock()
        prev = self.seen.get(name)
        if prev is None or prev[0] != size:
            self.seen[name] = (size, now)
            return False
        return now - prev[1] >= self.stable

    def scan(self, final: bool = False) -> list:
        done = []
        with self._lock:
            man = self.manifest()
            for name in existing_checkpoints(self.lay.local_checkpoints):
                if name in man["checkpoints"] or not self._ready(name, final):
                    continue
                dst = os.path.join(self.lay.checkpoints, name)
                if os.path.exists(dst):
                    if not self.p.restart or not os.path.isfile(os.path.join(dst, DONE_MARKER)):
                        raise StepFailed(f"{dst} already exists on Drive and is not recorded as written by this run. Nothing was overwritten; "
                                         "use a new RUN_ID or RESTART=True.")
                    shutil.rmtree(dst)  # our own earlier verified copy of the same run id, replaced because RESTART is on
                self._say(f"Syncing {name} to Drive ({human(dir_size(os.path.join(self.lay.local_checkpoints, name)))}) ...")
                t0 = self.clock()
                size = copy_verified(os.path.join(self.lay.local_checkpoints, name), dst)
                man["checkpoints"][name] = {"bytes": size, "synced": time.strftime("%Y-%m-%d %H:%M:%S"), "verified": "sizes of all files, sha256 of files under 32 MB"}
                write_json_atomic(self.lay.ckpt_manifest, man)
                self.synced.append(name)
                done.append(name)
                self._say(f"{name}: on Drive and verified in {fmt_secs(self.clock() - t0)}.")
                if self.status:
                    self.status.update(latest_checkpoint=name)
                if self.first_bytes is None:
                    self.first_bytes = size
                    self._first_report(size)
                self._free_local(size)
        return done

    def _say(self, msg: str) -> None:
        say(msg)

    def _first_report(self, size: int) -> None:
        total = size * self.p.epochs
        say(f"First checkpoint is {human(size)}; all {self.p.epochs} epochs will use about {human(total)} on Drive.")
        try:
            free = shutil.disk_usage(self.lay.checkpoints).free
            if free < total - size:
                say(f"WARNING: Drive reports {human(free)} free, less than the {human(total - size)} still to come. Free space or lower EPOCHS now; "
                    "a late epoch may fail to save (it stays in /content until you disconnect).")
        except OSError:
            pass

    def _free_local(self, size: int) -> None:
        """Keep local copies (fast to sample from) unless /content is getting full; then drop the oldest synced local copy."""
        try:
            free = shutil.disk_usage(self.lay.local_checkpoints).free
        except OSError:
            return
        if free < 3 * size:
            for name in self.synced[:-1]:
                path = os.path.join(self.lay.local_checkpoints, name)
                if os.path.isdir(path):
                    shutil.rmtree(path)  # a local copy that is verified on Drive
                    say(f"/content is low on space: removed the local copy of {name} (it is safe on Drive).")
                    return

    def _loop(self) -> None:
        while not self._stop.wait(10.0):
            try:
                self.scan()
            except StepFailed as exc:
                self.error = exc
                return

    def start(self) -> None:
        self._thread.start()

    def finish(self) -> list:
        self._stop.set()
        self._thread.join(timeout=60)
        if self.error:
            raise self.error
        return self.scan(final=True)


def refuse_overwrite(p: Params, lay: Layout) -> None:
    found = existing_checkpoints(lay.checkpoints)
    if found and not p.restart:
        raise StepFailed(
            f"Run {lay.run_id} already has checkpoints on Drive: {', '.join(found)}.\n"
            "sft_12hz.py cannot resume: a new run starts from the base model. Leave RUN_ID empty for a new timestamped run folder, "
            "or set RESTART=True to deliberately re-run this run id and replace its checkpoints. Nothing was changed.")
    if found and p.restart:
        say(f"RESTART is on: {', '.join(found)} will be replaced as the new run reaches each epoch.")


def apply_retention(p: Params, lay: Layout) -> list:
    """KEEP_ALL=False only: delete checkpoints on Drive that this script verified and recorded, keeping the newest N / first / final."""
    if p.keep_all:
        if p.keep_last_n:
            say(f"Retention: KEEP_ALL is on, so KEEP_LAST_N={p.keep_last_n} is ignored and nothing is deleted.")
        return []
    man = read_json(lay.ckpt_manifest, {"checkpoints": {}})
    names = [d for d in existing_checkpoints(lay.checkpoints) if d in man["checkpoints"] and os.path.isfile(os.path.join(lay.checkpoints, d, DONE_MARKER))]
    foreign = [d for d in existing_checkpoints(lay.checkpoints) if d not in names]
    if foreign:
        say("Not touched (not recorded as written and verified by this script): " + ", ".join(foreign))
    keep = set(names[-p.keep_last_n:]) if p.keep_last_n else set()
    if p.keep_first and names:
        keep.add(names[0])
    if p.keep_final and names:
        keep.add(names[-1])
    real_root, removed = os.path.realpath(lay.checkpoints), []
    for d in names:
        full = os.path.join(lay.checkpoints, d)
        if d in keep:
            continue
        if os.path.islink(full) or os.path.dirname(os.path.realpath(full)) != real_root:
            say(f"Skipping {d}: not a plain folder inside the checkpoint folder.")
            continue
        shutil.rmtree(full)
        man["checkpoints"].pop(d, None)
        removed.append(d)
    write_json_atomic(lay.ckpt_manifest, man)
    say("Retention: " + (f"removed {', '.join(removed)}; kept {', '.join(sorted(keep))}." if removed else "nothing to remove."))
    return removed


# --------------------------------------------------------------------------------------------- reproducibility snapshot
SNAPSHOT_PACKAGES = ["torch", "torchaudio", "transformers", "accelerate", "qwen-tts", "flash-attn", "librosa", "soundfile", "safetensors",
                     "huggingface-hub", "tensorboard", "numpy", "gradio", "onnxruntime"]


def package_versions() -> dict:
    from importlib import metadata

    out = {}
    for name in SNAPSHOT_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except Exception:  # noqa: BLE001
            out[name] = None
    return out


def write_snapshot(p: Params, lay: Layout, env: dict, gpu: Optional[Gpu], path: str, plan: Optional[dict], dataset_hash: str,
                   started: str, ended: Optional[str] = None, outcome: Optional[str] = None) -> None:
    """environment/snapshot.json and config/settings.json in the run folder on Drive."""
    try:
        write_json_atomic(os.path.join(lay.run_config, "settings.json"), p.provenance())
        write_json_atomic(os.path.join(lay.environment, "snapshot.json"), {
            "qwen_repo": QWEN_REPO, "qwen_commit": QWEN_COMMIT, "runner_version": RUNNER_VERSION, "driver_version": DRIVER_VERSION,
            "python": env.get("python_version"), "torch": env.get("torch"), "cuda": env.get("cuda"),
            "gpu": {"name": gpu.name, "vram_gb": gpu.vram_gb, "compute_capability": env.get("gpu_cc")} if gpu else None,
            "attention_path": path, "packages": package_versions(), "settings": p.provenance(), "dataset_hash": dataset_hash,
            "commands": {k: " ".join(shlex_quote(x) for x in plan[k]) for k in ("prepare", "train")} if plan else None,
            "started": started, "ended": ended, "outcome": outcome})
    except OSError as exc:
        say(f"Could not write the environment snapshot ({exc}).")


def shlex_quote(s: str) -> str:
    return s if re.fullmatch(r"[A-Za-z0-9_@%+=:,./-]+", s) else "'" + s.replace("'", "'\"'\"'") + "'"


# --------------------------------------------------------------------------------------------- preprocessing with progress
DRIVER_SOURCE = '''"""VoiceStudio re-drive of Qwen3-TTS finetuning/prepare_data.py: same Qwen3TTSTokenizer calls, batch size 32, but it prints
progress, writes every batch to a partial file and can continue after a crash. Written under /content by voicestudio_qwen3_colab.py."""
import argparse, json, os, sys, time

from qwen_tts import Qwen3TTSTokenizer

BATCH_INFER_NUM = 32  # same as the official script


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tokenizer_model_path", required=True)
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--output_jsonl", required=True)  # the .partial file
    ap.add_argument("--resume_from", type=int, default=0)
    args = ap.parse_args()
    rows = []
    with open(args.input_jsonl, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    sys.exit("invalid JSON in %s at row %d" % (args.input_jsonl, n))
    tok = Qwen3TTSTokenizer.from_pretrained(args.tokenizer_model_path, device_map=args.device)
    total, i, t0 = len(rows), args.resume_from, time.time()
    start = i
    print("PREP start: %d rows%s" % (total, (", continuing after row %d" % i) if i else ""), flush=True)
    with open(args.output_jsonl, "a" if i else "w", encoding="utf-8") as out:
        while i < total:
            batch = rows[i:i + BATCH_INFER_NUM]
            enc = tok.encode([r["audio"] for r in batch])
            for code, row in zip(enc.audio_codes, batch):
                row["audio_codes"] = code.cpu().tolist()
                out.write(json.dumps(row, ensure_ascii=False) + "\\n")
            out.flush()
            os.fsync(out.fileno())
            i += len(batch)
            el = time.time() - t0
            eta = el / max(1, i - start) * (total - i)
            print("PREP %d/%d (%.0f%%) batch %d/%d elapsed %.0fs ETA %.0fs" % (i, total, 100.0 * i / total, (i + BATCH_INFER_NUM - 1) // BATCH_INFER_NUM,
                  (total + BATCH_INFER_NUM - 1) // BATCH_INFER_NUM, el, eta), flush=True)
    print("PREP done", flush=True)


if __name__ == "__main__":
    main()
'''


def validate_codes(path: str, expected_rows: int) -> int:
    """Row count must match and every row must hold audio_codes. Raises with the row number."""
    n = 0
    with open(path, encoding="utf-8") as fh:
        for no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                raise StepFailed(f"{path}: row {no} is not valid JSON.")
            if not row.get("audio_codes"):
                raise StepFailed(f"{path}: row {no} has no audio_codes.")
            n += 1
    if n != expected_rows:
        raise StepFailed(f"{path} has {n} rows but train.jsonl has {expected_rows}.")
    return n


def _complete_lines(path: str) -> int:
    """Number of whole lines in a partial file; a torn final line (crash mid-write) is cut off."""
    keep, n = 0, 0
    with open(path, "rb") as fh:
        for raw in fh:
            if raw.endswith(b"\n"):
                keep += len(raw)
                n += 1
    with open(path, "r+b") as fh:
        fh.truncate(keep)
    return n


def write_driver(lay: Layout) -> str:
    os.makedirs(lay.patched, exist_ok=True)
    path = os.path.join(lay.patched, "prepare_progress.py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(DRIVER_SOURCE)
    return path


def prepare_command(p: Params, lay: Layout, folder: str, input_jsonl: str, partial: str, resume_from: int, driver: str) -> dict:
    ft = os.path.join(lay.local_repo, "finetuning")
    return {"cwd": folder, "env": {"PYTHONPATH": ft + os.pathsep + os.environ.get("PYTHONPATH", "")},
            "cmd": [py(), "-u", driver, "--device", "cuda:0", "--tokenizer_model_path", local_model_dir(lay, p.tokenizer_model),
                    "--input_jsonl", input_jsonl, "--output_jsonl", partial, "--resume_from", str(resume_from)]}


def ensure_codes(p: Params, lay: Layout, folder: str, dataset_hash: str, rows: int, log: str, *, stream: Callable = run_stream) -> str:
    """Audio codes for the whole dataset, cached on Drive and reused ONLY when the dataset hash (and tokenizer, commit) match."""
    want = {"dataset_hash": dataset_hash, "tokenizer": p.tokenizer_model, "qwen_commit": QWEN_COMMIT, "driver_version": DRIVER_VERSION, "rows": rows}
    meta = read_json(lay.codes_meta)
    reason = None
    if not os.path.isfile(lay.codes) or not meta:
        reason = "no prepared codes for this speaker yet"
    else:
        for key, label in (("dataset_hash", "dataset hash"), ("tokenizer", "tokenizer"), ("qwen_commit", "Qwen commit"),
                           ("driver_version", "preprocessing version"), ("rows", "row count")):
            if meta.get(key) != want[key]:
                reason = f"{label} changed ({str(meta.get(key))[:12]} -> {str(want[key])[:12]})"
                break
        if reason is None and meta.get("codes_sha256") != sha256_file(lay.codes):
            reason = "the stored codes file no longer matches its recorded hash"
    os.makedirs(lay.work, exist_ok=True)
    if reason is None:
        say(f"== Preprocessing: reusing the audio codes on Drive (dataset hash {dataset_hash[:12]} matches; cache hit).")
        if not os.path.isfile(lay.codes_local) or sha256_file(lay.codes_local) != meta["codes_sha256"]:
            shutil.copyfile(lay.codes, lay.codes_local)
        return lay.codes_local
    say(f"== Preprocessing: generating audio codes. Why: {reason}.")
    partial, pmeta = lay.codes + ".partial", lay.codes + ".partial.json"
    resume = 0
    pm = read_json(pmeta)
    if os.path.isfile(partial) and pm == want:
        resume = _complete_lines(partial)
        say(f"An interrupted run left {resume} finished rows for this same dataset; continuing after them.")
    elif os.path.isfile(partial):
        os.remove(partial)  # our own leftover for different data or settings: restarting cleanly
        say("A partial file from different data or settings was found (ours); restarting from the first row.")
    os.makedirs(lay.prepared, exist_ok=True)
    write_json_atomic(pmeta, want)
    driver = write_driver(lay)
    cmd = prepare_command(p, lay, folder, "train.jsonl", partial, resume, driver)
    with timed("preprocessing (audio codes)", "train"):
        if stream(cmd["cmd"], cwd=cmd["cwd"], env=cmd["env"], log_path=log) != 0:
            raise StepFailed("Preprocessing failed; see the log. The partial file is kept so the next run continues from it.")
    validate_codes(partial, rows)
    os.replace(partial, lay.codes)
    os.remove(pmeta)
    write_json_atomic(lay.codes_meta, {**want, "codes_sha256": sha256_file(lay.codes), "created": time.strftime("%Y-%m-%d %H:%M:%S")})
    shutil.copyfile(lay.codes, lay.codes_local)
    say(f"Audio codes saved to {lay.codes} (validated: {rows} rows, all with audio_codes).")
    return lay.codes_local


# --------------------------------------------------------------------------------------------- training
def build_patched_script(lay: Layout, impl: str = ATTN_FALLBACK_IMPL) -> str:
    """Copy sft_12hz.py to /content with the attention implementation changed. The cached repo is never touched."""
    src = os.path.join(lay.local_repo, "finetuning", "sft_12hz.py")
    with open(src, encoding="utf-8") as fh:
        text = fh.read()
    if text.count(FLASH_LINE) != 1:
        raise StepFailed("sft_12hz.py does not contain the expected attention line exactly once, so it cannot be patched safely "
                         "(the repo is not the pinned commit?).")
    os.makedirs(lay.patched, exist_ok=True)
    out = os.path.join(lay.patched, "sft_12hz_patched.py")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("# UNVERIFIED copy of Qwen's sft_12hz.py written by VoiceStudio: attention changed from flash_attention_2 to "
                 f"{impl}.\n" + text.replace(FLASH_LINE, f'attn_implementation="{impl}"'))
    return out


def plan_commands(p: Params, lay: Layout, dataset_dir: str, batch: int, *, patched: Optional[str] = None, codes: Optional[str] = None,
                  out_dir: Optional[str] = None, epochs: Optional[int] = None) -> dict:
    """The official training command with this run's values (checkpoints go to local /content first)."""
    ft = os.path.join(lay.local_repo, "finetuning")
    return {
        "cwd": dataset_dir,
        "env": {"PYTHONPATH": ft + os.pathsep + os.environ.get("PYTHONPATH", "")},
        "train": [py(), "-u", patched or os.path.join(ft, "sft_12hz.py"), "--init_model_path", local_model_dir(lay, p.base_model),
                  "--output_model_path", out_dir or lay.local_checkpoints, "--train_jsonl", codes or lay.codes_local, "--batch_size", str(batch),
                  "--lr", f"{p.lr:g}", "--num_epochs", str(epochs or p.epochs), "--speaker_name", p.speaker_name],
        "prepare": [py(), "-u", os.path.join(lay.patched, "prepare_progress.py"), "--tokenizer_model_path", local_model_dir(lay, p.tokenizer_model),
                    "--input_jsonl", "train.jsonl", "--output_jsonl", lay.codes + ".partial"],
    }


def preflight(p: Params, lay: Layout, gpu: Gpu, path: str, batch: int) -> None:
    """Free disk, bf16, attention support: stop with a clear message before any GPU time is spent."""
    base_gb = dir_size(local_model_dir(lay, p.base_model)) / 2**30 or 4.0
    rows = []
    free_local = shutil.disk_usage(p.local_root).free / 2**30 if os.path.isdir(p.local_root) else None
    need_local = 2 * base_gb + 3
    rows.append(("free disk /content", "INFO" if free_local is None else "PASS" if free_local >= need_local else "FAIL",
                 "unknown" if free_local is None else f"{free_local:.0f} GB free; about {need_local:.0f} GB needed for two checkpoints in flight plus data"))
    try:
        free_drive = shutil.disk_usage(lay.root).free / 2**30
        need_drive = base_gb * p.epochs
        rows.append(("free space Drive", "PASS" if free_drive >= need_drive else "FAIL",
                     f"{free_drive:.0f} GB free; about {need_drive:.0f} GB for {p.epochs} epoch checkpoints ({base_gb:.1f} GB each, estimated from the base model)"))
    except OSError:
        rows.append(("free space Drive", "INFO", "could not be read"))
    rows.append(("bfloat16", "PASS" if gpu.cc and gpu.cc[0] >= 8 else "FAIL" if gpu.cc is None else "INFO",
                 "native on this GPU" if gpu.cc and gpu.cc[0] >= 8 else "not native on this GPU (emulated, slow)" if gpu.cc else "unknown GPU architecture"))
    rows.append(("attention", "PASS" if path in ("flash2", "sdpa") else "FAIL", current_attention(gpu, p, probe=True)[1]))
    rows.append(("batch size", "PASS" if batch >= 1 else "FAIL", f"{batch}"))
    say("== Preflight")
    if not check_table(rows):
        raise StepFailed("Preflight failed (see the table). Nothing was trained; no GPU time was spent.")


def _preflight(p: Params, lay: Layout, gpu: Optional[Gpu]) -> tuple:
    """GPU, attention path, batch size and readiness checks shared by the real run and the timing run."""
    gpu_report(p, gpu)
    if gpu is None:
        raise StepFailed("No GPU: select a GPU runtime first.")
    path, why = current_attention(gpu, p, probe=True)
    if path == "refuse":
        raise StepFailed("Qwen's sft_12hz.py hard-codes FlashAttention 2, and " + why + ". Choose an A100, H100, L4 or G4 runtime, or set ATTN_FALLBACK=True.")
    if gpu.vram_gb < BATCH_BY_VRAM[0][0] and p.batch_size != "auto":
        say(f"Warning: {gpu.vram_gb:g} GB is below the roughly 14 GB this fine-tune needs; expect an out-of-memory error.")
    batch = resolve_batch(p, gpu)
    if not os.path.isfile(os.path.join(lay.local_repo, "finetuning", "sft_12hz.py")) \
            or not all(os.path.isfile(os.path.join(local_model_dir(lay, m), DONE_MARKER)) for m in (p.base_model, p.tokenizer_model)):
        raise StepFailed("Setup is not complete on this runtime (repo or local models missing). Run the setup cell first.")
    return path, batch


def _announce_patch(path: str, lay: Layout) -> Optional[str]:
    if path != "sdpa":
        return None
    patched = build_patched_script(lay)
    say("*" * 78)
    say("ATTENTION: training with a PATCHED COPY of sft_12hz.py (PyTorch SDPA attention instead of FlashAttention 2).")
    say(f"Copy: {patched}. Qwen's cached repo is unchanged.")
    say("UNVERIFIED: not run by the author; it may fail or run out of memory.")
    say("*" * 78)
    return patched


def oom_message(gpu: Gpu, batch: int, mon: TrainMonitor, peak_mb: float) -> str:
    e, s = mon.last_step
    return (f"CUDA out of memory on {gpu.name} ({gpu.vram_gb:g} GB) at batch size {batch}, epoch {e + 1}, step {s} "
            f"(peak seen {peak_mb / 1024:.1f} GB). Next: set BATCH_SIZE={max(1, batch // 2)} (or 1) and start a new run; checkpoints of "
            "finished epochs are kept. If batch size 1 still fails, use a GPU with more memory.")


def train(p: Params, *, gpu: Optional[Gpu] = None, stream: Callable = run_stream, detect: Callable = detect_gpu,
          sample_memory: Callable = gpu_memory_used_mb) -> list:
    """Prepare codes, run sft_12hz.py from inside the dataset folder with checkpoints written to /content and synced to Drive after every
    epoch, then sample from each checkpoint. With SMOKE_RUN on, runs the short timing run instead."""
    use_existing_venv(p)
    if p.smoke_run:
        timing_run(p, gpu=gpu, stream=stream, detect=detect, sample_memory=sample_memory)
        return []
    lay = ensure_layout(p)
    status = Status(lay)
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    gpu = gpu or detect()
    try:
        refuse_overwrite(p, lay)
        status.update(status="preflight")
        path, batch = _preflight(p, lay, gpu)
        preflight(p, lay, gpu, path, batch)
        folder, result = _data_check(p)
        patched = _announce_patch(path, lay)
        log = new_log(lay, p, "train", per_run=True)
        env = detect_environment()
        status.update(status="preparing")
        codes = ensure_codes(p, lay, folder, result["dataset_hash"], result["rows"], log, stream=stream)
        plan = plan_commands(p, lay, folder, batch, patched=patched, codes=codes)
        write_snapshot(p, lay, env, gpu, path, plan, result["dataset_hash"], started)
        say(f"== Training: {p.epochs} epochs, batch size {batch}, learning rate {p.lr:g}, speaker {p.speaker_name}, attention {path.upper()}")
        say(f"Checkpoints are written to {lay.local_checkpoints} (local), then each finished epoch is copied to Drive: {lay.checkpoints}")
        say("There is no resume: a disconnect keeps the epochs already synced to Drive. The official script writes no TensorBoard event files, "
            "so there is no TensorBoard to open; progress is shown below and in metrics.csv / status.json in the run folder.")
        os.makedirs(lay.local_checkpoints, exist_ok=True)
        batches = math.ceil(result["audio"] / batch)
        mon = TrainMonitor(epochs=p.epochs, batches_per_epoch=batches)
        peak, beat, syncer = GpuPeak(sample_memory), Heartbeat(mon, GpuPeak(sample_memory), p.heartbeat_seconds), CheckpointSyncer(p, lay, status)
        beat.peak = peak
        status.update(status="training")

        def on_line(line: str) -> bool:
            pr = mon.feed(line)
            if pr is None:
                return False
            say(format_progress(pr, (peak.last_mb, peak.peak_mb)))
            append_metrics(lay, pr, peak.last_mb)
            status.update(force=False, epoch=pr["epoch"], step=pr["step"], loss=pr["loss"])
            return True

        mon.start = time.time()
        peak.start(), beat.start(), syncer.start()
        try:
            with timed("training (sft_12hz.py)", "train"):
                code = stream(plan["train"], cwd=plan["cwd"], env=plan["env"], log_path=os.path.join(lay.run_logs, "train.log"), on_line=on_line)
        except BaseException:  # for example the notebook's interrupt button: still save every epoch that finished
            peak.stop(), beat.stop()
            status.update(status="interrupted")
            try:
                syncer.finish()
            except Exception:  # noqa: BLE001 — the original interruption matters more
                pass
            raise
        finally:
            peak.stop(), beat.stop()
        status.update(status="syncing")
        synced = syncer.finish()
        if syncer.not_finished:
            say("WARNING: not copied to Drive because they do not look complete: " + ", ".join(syncer.not_finished) + f" (left in {lay.local_checkpoints}).")
        say(f"Checkpoints on Drive for this run: {', '.join(existing_checkpoints(lay.checkpoints)) or 'none'}")
        if code != 0:
            if mon.oom:
                raise StepFailed(oom_message(gpu, batch, mon, peak.peak_mb))
            raise StepFailed(f"sft_12hz.py exited with code {code}; see {os.path.join(lay.run_logs, 'train.log')}. Epochs already synced to Drive are kept.")
        if p.sample_after_training:
            status.update(status="sampling")
            generate_samples(p, lay, stream=stream, gpu=gpu)
        apply_retention(p, lay)
        status.update(status="done")
        write_snapshot(p, lay, env, gpu, path, plan, result["dataset_hash"], started, time.strftime("%Y-%m-%d %H:%M:%S"), "done")
        say("Training finished.")
        timing_summary(lay, "Training", "train")
        return synced
    except StepFailed as exc:
        fail_run(lay, status, str(exc))
        raise
    finally:
        finish_reminder()


# --------------------------------------------------------------------------------------------- final sample
SAMPLE_CODE = '''
import sys, torch, soundfile as sf
from qwen_tts import Qwen3TTSModel
ckpt, speaker, text, out, attn, seed = sys.argv[1:7]
torch.manual_seed(int(seed))  # our addition: the README shows no seed argument
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(int(seed))
tts = Qwen3TTSModel.from_pretrained(ckpt, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation=attn)
wavs, sr = tts.generate_custom_voice(text=text, speaker=speaker)
sf.write(out, wavs[0], sr)
print("wrote", out)
'''


def generate_samples(p: Params, lay: Layout, *, stream: Callable = run_stream, gpu: Optional[Gpu] = None,
                     probe: Callable = probe_module, detect: Callable = detect_gpu) -> list:
    """After training (never during it): one WAV, samples/final.wav, from the final checkpoint as persisted on Drive. Usage follows the
    finetuning README at the pinned commit (from_pretrained(path, device_map, dtype, attn_implementation) and
    generate_custom_voice(text, speaker)); NOT run by the author. If it cannot run safely it says why."""
    man = read_json(lay.ckpt_manifest, {"checkpoints": {}})
    names = [d for d in existing_checkpoints(lay.checkpoints) if d in man["checkpoints"]]
    if not names:
        say(f"Sample skipped: no verified checkpoint on Drive in {lay.checkpoints}.")
        return []
    gpu = gpu or detect()
    if gpu is None:
        say("Sample skipped: no GPU is available to load the checkpoint.")
        return []
    if not probe("qwen_tts")[0]:
        say("Sample skipped: qwen_tts cannot be imported in this runtime (run setup).")
        return []
    path, _why = current_attention(gpu, p, probe=True)
    if path == "refuse":
        say("Sample skipped: no usable attention path on this GPU.")
        return []
    attn = "flash_attention_2" if path == "flash2" else ATTN_FALLBACK_IMPL
    os.makedirs(lay.samples, exist_ok=True)
    out, ckpt = os.path.join(lay.samples, "final.wav"), os.path.join(lay.checkpoints, names[-1])
    say(f"== Sample from {names[-1]}: text {p.sample_text!r}, seed {p.sample_seed}, attention {attn}. UNVERIFIED API use (README at the pinned commit).")
    code = stream([py(), "-c", SAMPLE_CODE, ckpt, p.speaker_name, p.sample_text, out, attn, str(p.sample_seed)],
                  log_path=os.path.join(lay.run_logs, "samples.log"))
    if code != 0 or not os.path.isfile(out):
        say("final.wav could not be generated (see samples.log); the checkpoints are unaffected.")
        run_log(lay, f"sample from {names[-1]} failed with code {code}")
        return []
    write_json_atomic(out[:-4] + ".json", {"checkpoint": names[-1], "text": p.sample_text, "seed": p.sample_seed, "speaker": p.speaker_name,
                                           "attention": attn, "generated": time.strftime("%Y-%m-%d %H:%M:%S")})
    say(f"Wrote {out}")
    return [out]


def final_sample(p: Params, *, stream: Callable = run_stream) -> list:
    use_existing_venv(p)
    """Notebook cell: generate samples/final.wav for this run (optional, unverified)."""
    return generate_samples(p, find_run(p), stream=stream)


def find_run(p: Params) -> Layout:
    lay = Layout(p)
    if os.path.isdir(lay.run):
        return lay
    if os.path.isdir(lay.runs):
        runs = sorted(d for d in os.listdir(lay.runs) if os.path.isdir(os.path.join(lay.runs, d)))
        if runs:
            say(f"Run {p.run_id} has no folder yet; showing the newest run {runs[-1]} instead.")
            return Layout(p, runs[-1])
    return lay


# --------------------------------------------------------------------------------------------- timing run
def project(mon: TrainMonitor, *, clips_used: int, total_clips: int, batch: int, epochs: int, ckpt_bytes: int, prep_seconds: float) -> dict:
    """Seconds per batch from the timestamped step lines, then projections to the full data. Estimates from a short sample."""
    out = {"seconds_per_batch": None}
    if mon.end is None or mon.start is None or 0 not in mon.steps:
        return out
    later = [s for s in mon.steps if s > 0]
    if not later:
        return out
    last = max(later)
    spb = (mon.steps[last] - mon.steps[0]) / last
    batches_sample = math.ceil(clips_used / batch)
    batches_epoch = math.ceil(total_clips / batch)
    remaining = max(0, batches_sample - 1 - last)
    save = max(0.0, mon.end - mon.steps[last] - remaining * spb)  # checkpoint copy and save, plus the tail of the epoch
    epoch_s = spb * batches_epoch
    startup = mon.steps[0] - mon.start
    prep_total = prep_seconds * (total_clips / max(1, clips_used))
    out.update(seconds_per_batch=spb, batches_in_sample=batches_sample, batches_per_epoch=batches_epoch, epoch_seconds=epoch_s,
               save_seconds=save, startup_seconds=startup, prep_seconds_full=prep_total, ckpt_bytes=ckpt_bytes,
               train_seconds=startup + epochs * (epoch_s + save), total_seconds=prep_total + startup + epochs * (epoch_s + save))
    return out


def report_projection(p: Params, gpu: Gpu, path: str, batch: int, clips_used: int, total_clips: int, peak_mb: float, proj: dict,
                      setup_seconds: float) -> None:
    say("")
    say("=" * 78)
    say("TIMING RUN RESULT (an ESTIMATE from a short sample; real runs vary, and the first batches include warm-up)")
    say(f"GPU: {gpu.name}, {gpu.vram_gb:g} GB   attention path: {path.upper()}"
        + (" (patched SDPA copy, unverified)" if path == "sdpa" else " (FlashAttention 2, original script)"))
    say(f"Sample: {clips_used} of {total_clips} clips, batch size {batch}, one epoch")
    total_gb = (gpu.vram_gb * 1024)
    say(f"Peak GPU memory seen by nvidia-smi on this GPU: {peak_mb / 1024:.1f} GB of {gpu.vram_gb:g} GB "
        f"({100 * peak_mb / total_gb:.0f}%; includes PyTorch's cache, so it is an upper bound on what is needed)")
    spb = proj.get("seconds_per_batch")
    if spb is None:
        say("Too few timed batches to project (need the 'Step 10' line). Raise SMOKE_CLIPS.")
        return
    say(f"Seconds per batch: {spb:.2f}   (batches per full epoch: {proj['batches_per_epoch']})")
    say(f"Projected ONE full epoch on all {total_clips} clips: {fmt_secs(proj['epoch_seconds'])}")
    say(f"Projected checkpoint save per epoch: about {fmt_secs(proj['save_seconds'])}; each epoch checkpoint adds {human(proj['ckpt_bytes'])} to Drive "
        f"(all {p.epochs}: {human(proj['ckpt_bytes'] * p.epochs)}, base-model copy size)")
    say(f"Projected ALL {p.epochs} EPOCHS (plus start-up {fmt_secs(proj['startup_seconds'])} and preprocessing about "
        f"{fmt_secs(proj['prep_seconds_full'])}, a rough upper bound): {fmt_secs(proj['total_seconds'])}")
    rate_info = default_units_per_hour(gpu)
    rate = p.units_per_hour if p.units_per_hour is not None else (rate_info[0] if rate_info else None)
    if rate is None:
        say("Compute units: UNITS_PER_HOUR is empty and this GPU is not in the built-in table. Read the rate for this runtime in Colab's runtime menu,")
        say("  set UNITS_PER_HOUR to it and run the timing run again (or multiply: hours x rate).")
    else:
        src = "set by you" if p.units_per_hour is not None else f"from the owner's table for {rate_info[1]}"
        say(f"Compute units at {rate:g} units/hour ({src}; the owner's figure, NOT verified, and Colab may change it: check the rate shown in Colab's runtime menu):")
        e_units = proj["epoch_seconds"] / 3600 * rate
        t_units = proj["total_seconds"] / 3600 * rate
        s_units = setup_seconds / 3600 * rate
        say(f"  one epoch: {e_units:.1f} units   all {p.epochs} epochs: {t_units:.1f} units")
        say(f"  setup in this session: {fmt_secs(setup_seconds)} = {s_units:.1f} units" + ("  (not measured in this session: add what your setup took)" if not setup_seconds else ""))
        total = t_units + s_units
        left = p.units_available - total
        say(f"  projected total incl. setup: {total:.1f} units (about ${total * CENTS_PER_UNIT / 100:.2f} at roughly {CENTS_PER_UNIT} cents per unit, the owner's figure)")
        say(f"  against UNITS_AVAILABLE={p.units_available:g}: " + (f"FITS with about {left:.1f} units ({100 * left / p.units_available:.0f}%) to spare"
            if left >= 0 else f"DOES NOT FIT: short by about {-left:.1f} units. Lower EPOCHS or use less data or a cheaper-per-epoch GPU"))
        say("  (this does not include idle time before you disconnect, a failed attempt, or a second setup after a disconnect)")
        if p.compare_epoch_seconds is not None:
            other_rate = p.compare_units_per_hour
            if other_rate is None:
                say(f"COMPARE_EPOCH_SECONDS={p.compare_epoch_seconds:g}: set COMPARE_UNITS_PER_HOUR (the other GPU's rate) to compare cost.")
            else:
                mine, theirs = proj["epoch_seconds"] / 3600 * rate, p.compare_epoch_seconds / 3600 * other_rate
                speedup = max(p.compare_epoch_seconds / proj["epoch_seconds"], proj["epoch_seconds"] / p.compare_epoch_seconds)
                say(f"Per-epoch cost: this GPU {mine:.1f} units ({fmt_secs(proj['epoch_seconds'])}) vs the other {theirs:.1f} units ({fmt_secs(p.compare_epoch_seconds)}): "
                    + ("this GPU is cheaper per epoch." if mine < theirs else "the other GPU is cheaper per epoch." if theirs < mine else "equal."))
                lo, hi = sorted([(rate, "this GPU"), (other_rate, "the other GPU")])
                say(breakeven_line(lo[0], hi[0], lo[1], hi[1]) + f" Measured speed ratio: {speedup:.1f}x.")
        else:
            say("Break-even rule: a GPU costing R2 units/hour only saves units over one costing R1 units/hour if it is more than R2/R1 times faster on this job. "
                "From the owner's table: G4 vs A100 40 GB is 8.7 / 5.4 = 1.6x; A100 80 GB vs A100 40 GB is 7.5 / 5.4 = 1.4x. To compare, run this on the "
                "other GPU and enter its seconds per epoch as COMPARE_EPOCH_SECONDS (and its rate as COMPARE_UNITS_PER_HOUR).")
    say("=" * 78)


def timing_run(p: Params, *, gpu: Optional[Gpu] = None, stream: Callable = run_stream, detect: Callable = detect_gpu,
               sample_memory: Callable = gpu_memory_used_mb) -> dict:
    """SMOKE_RUN: one epoch on the first N clips with the real settings and GPU. Everything goes to a separate '<run id>-timing' run
    folder on Drive (logs, timing.json) and /content (sample checkpoint, deleted afterwards); the real runs are untouched."""
    base = ensure_layout(p, create_run=False)
    lay = Layout(p, p.run_id + "-timing")
    for folder in (lay.run, lay.run_logs):
        os.makedirs(folder, exist_ok=True)
    gpu = gpu or detect()
    status = Status(lay)
    try:
        path, batch = _preflight(p, base, gpu)
        folder, result = _data_check(p)
        total_clips = result["audio"]
        n = min(total_clips, max(p.smoke_clips, batch * SMOKE_MIN_BATCHES))
        say(f"== Timing run: {n} clips (SMOKE_CLIPS={p.smoke_clips}, raised if needed so there are {SMOKE_MIN_BATCHES}+ batches), 1 epoch, run folder {lay.run}")
        say("Nothing is written to your real runs or to the prepared-codes cache.")
        os.makedirs(lay.work, exist_ok=True)
        subset = os.path.join(lay.work, "train_subset.jsonl")
        with open(os.path.join(folder, "train.jsonl"), encoding="utf-8") as src, open(subset, "w", encoding="utf-8") as dst:
            kept = 0
            for line in src:
                if line.strip() and kept < n:
                    dst.write(line if line.endswith("\n") else line + "\n")
                    kept += 1
        smoke_codes = os.path.join(lay.work, "smoke_codes.jsonl")
        smoke_ckpt = os.path.join(p.local_root, "runs", p.speaker_name, p.run_id + "-timing", "checkpoints")
        if os.path.isdir(smoke_ckpt):
            shutil.rmtree(smoke_ckpt)  # leftover of an earlier timing run with this id (ours, under /content/runs)
        patched = _announce_patch(path, lay)
        log = new_log(lay, p, "timing-run", per_run=True)
        driver = write_driver(lay)
        pc = prepare_command(p, lay, folder, subset, smoke_codes + ".partial", 0, driver)
        if os.path.exists(smoke_codes + ".partial"):
            os.remove(smoke_codes + ".partial")
        t = time.time()
        with timed("timing run: preprocessing the sample", "train"):
            if stream(pc["cmd"], cwd=pc["cwd"], env=pc["env"], log_path=log) != 0:
                raise StepFailed("Preprocessing the sample failed; see the log.")
        os.replace(smoke_codes + ".partial", smoke_codes)
        prep_seconds = time.time() - t
        plan = plan_commands(p, lay, folder, batch, patched=patched, codes=smoke_codes, out_dir=smoke_ckpt, epochs=1)
        mon, peak = TrainMonitor(epochs=1, batches_per_epoch=math.ceil(n / batch)), GpuPeak(sample_memory)
        os.makedirs(smoke_ckpt, exist_ok=True)
        peak.start()
        mon.start = time.time()
        try:
            with timed("timing run: one epoch of the sample", "train"):
                code = stream(plan["train"], cwd=plan["cwd"], env=plan["env"], log_path=log, on_line=lambda line: bool(mon.feed(line)) and False)
        finally:
            mon.end = time.time()
            peak_mb = peak.stop()
        ckpt = os.path.join(smoke_ckpt, "checkpoint-epoch-0")
        ckpt_bytes = dir_size(ckpt) if os.path.isdir(ckpt) else int(dir_size(local_model_dir(lay, p.base_model)))
        if os.path.isdir(smoke_ckpt):
            shutil.rmtree(smoke_ckpt)  # the sample checkpoint we just wrote under /content
        if code != 0 or mon.oom:
            if mon.oom or peak_mb >= gpu.vram_gb * 1024 * 0.97:
                raise StepFailed(oom_message(gpu, batch, mon, peak_mb))
            raise StepFailed(f"The sample run exited with code {code}; see the log {log}.")
        proj = project(mon, clips_used=n, total_clips=total_clips, batch=batch, epochs=p.epochs, ckpt_bytes=ckpt_bytes, prep_seconds=prep_seconds)
        report_projection(p, gpu, path, batch, n, total_clips, peak_mb, proj, session_seconds("setup", "data"))
        write_json_atomic(os.path.join(lay.run, "timing.json"), {"gpu": gpu.name, "attention": path, "batch": batch, "clips": n, "total_clips": total_clips,
                                                                 "peak_mb": peak_mb, "projection": proj, "when": time.strftime("%Y-%m-%d %H:%M:%S"),
                                                                 "note": "estimate from a short sample"})
        status.update(status="done")
        timing_summary(lay, "Timing run", "train")
        return {"projection": proj, "peak_mb": peak_mb, "attention": path, "batch": batch, "clips": n}
    except StepFailed as exc:
        fail_run(lay, status, str(exc))
        raise
    finally:
        finish_reminder()


# --------------------------------------------------------------------------------------------- results
def results(p: Params) -> list:
    lay = find_run(p)
    say(f"== Run {lay.run_id}: {lay.run}")
    st = read_json(lay.status)
    if st:
        say(f"status: {st.get('status')}   epoch {st.get('epoch')}   step {st.get('step')}   loss {st.get('loss')}   updated {st.get('updated')}"
            + (f"\nerror: {st['error']}" if st.get("error") else ""))
    man = read_json(lay.ckpt_manifest, {"checkpoints": {}})
    say(f"Checkpoints in {lay.checkpoints}")
    rows = []
    for d in existing_checkpoints(lay.checkpoints):
        size = dir_size(os.path.join(lay.checkpoints, d))
        rows.append((d, size))
        say(f"  {d:<22} {human(size):>10}   {'verified copy written by this script' if d in man['checkpoints'] else 'not recorded by this script'}")
    if not rows:
        say("  none yet")
    else:
        say(f"  total {human(sum(s for _, s in rows))}")
    if os.path.isdir(lay.samples):
        wavs = sorted(f for f in os.listdir(lay.samples) if f.endswith(".wav"))
        if wavs:
            say(f"Samples in {lay.samples}: " + ", ".join(wavs))
    return rows


# --------------------------------------------------------------------------------------------- command line
STEPS = ("gpu", "setup", "data", "train", "results", "sample", "all")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=STEPS, help="all = setup, data, train (incl. samples), results")
    ap.add_argument("--settings", help="training_settings.json from the export (provides the defaults)")
    for flag in ("drive-root", "data-source", "speaker-name", "batch-size", "base-model", "tokenizer-model", "run-id", "sample-text"):
        ap.add_argument("--" + flag)
    for flag, kind in (("epochs", int), ("keep-last-n", int), ("sample-seed", int), ("smoke-clips", int), ("heartbeat-seconds", int),
                       ("lr", float), ("units-per-hour", float), ("units-available", float), ("compare-epoch-seconds", float),
                       ("compare-units-per-hour", float)):
        ap.add_argument("--" + flag, type=kind)
    for flag in ("keep-all", "keep-first", "keep-final", "cache-models-on-drive", "sample-after-training"):
        ap.add_argument("--" + flag, choices=("true", "false"), help="default depends on the setting")
    ap.add_argument("--attn-fallback", choices=("auto", "true", "false"), help="auto (default): FlashAttention 2 on Ampere/Ada/Hopper, patched SDPA "
                    "copy on Blackwell; true: force the SDPA copy (unverified); false: force FlashAttention 2")
    ap.add_argument("--smoke-run", action="store_true", default=None, help="timing run: one epoch on the first --smoke-clips clips, then project time and units")
    ap.add_argument("--restart", action="store_true", default=None, help="allow replacing checkpoints of an existing run id")
    return ap


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    lay = None
    try:
        settings = args.settings
        if not settings:
            guess = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_settings.json")
            settings = guess if os.path.isfile(guess) else None
        p = params_from({k: getattr(args, k) for k in _ENV_FIELDS if k != "data_source" and hasattr(args, k)} | {"data_source": args.data_source},
                        settings)
        if not p.data_source and settings and args.step in ("data", "train", "all"):
            here = os.path.dirname(os.path.abspath(__file__))
            if os.path.isfile(os.path.join(here, "train.jsonl")):
                p.data_source = here
        step = args.step
        if step == "gpu":
            gpu_report(p)
        if step in ("setup", "all") and not setup(p):
            return 1
        if step == "data":
            data_check(p)
        if step in ("train", "all"):
            train(p)
        if step in ("results", "all") and not p.smoke_run:
            results(p)
        if step == "sample":
            final_sample(p)
        return 0
    except StepFailed as exc:
        say("\nSTOPPED: " + str(exc))
        return 2


if __name__ == "__main__":
    sys.exit(main())
