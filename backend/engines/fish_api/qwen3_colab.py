"""Files written next to a Qwen3-TTS export: the Colab runner script, the notebook, README, command sheet and settings.

Everything here was written from the official ``finetuning/`` files of QwenLM/Qwen3-TTS as they stood at
``QWEN_COMMIT`` (README.md, prepare_data.py, sft_12hz.py, dataset.py, pyproject.toml). The scripts and the commands
were NOT run: that needs a GPU runtime. All logic lives in ONE file, ``qwen3_colab_runner.py`` (standard library only),
which is copied into every export as ``voicestudio_qwen3_colab.py``. The notebook, ``colab_setup.sh`` and the
``colab`` command-line route are thin callers of it. Only the speaker name, the folder names and the Drive folder
vary, and they are validated to be shell-safe before they reach a script.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import shlex
from typing import Optional

from . import qwen3_colab_runner as runner
from .qwen3_colab_runner import (BASE_MODEL, DEFAULT_DRIVE_FOLDER, FOLDER_RE, MODEL_RE, QWEN_COMMIT, QWEN_COMMIT_DATE,  # noqa: F401
                                 QWEN_REPO, READ_ON, SAMPLE_RATE, SESSION, SPEAKER_RE, TOKENIZER)

ACCUMULATION = 4  # sft_12hz.py: Accelerator(gradient_accumulation_steps=4), not an option
RUNNER_NAME = "voicestudio_qwen3_colab.py"
NOTEBOOK_NAME = "Qwen3_TTS_Colab_Training.ipynb"
DEFAULT_SAMPLE_TEXT = runner.DEFAULT_SAMPLE_TEXT
DEFAULT_SPEAKER, DEFAULT_FOLDER_NAME = "MySpeaker", "my-dataset"  # the committed notebooks/ copy uses these
LAYOUT = {
    "cache/models": "base model and tokenizer, downloaded once",
    "cache/Qwen3-TTS": "the official repo at the pinned commit",
    "cache/wheelhouse/sm<arch>": "compiled or downloaded FlashAttention wheels, one folder per GPU architecture",
    "cache/pip": "pip download cache",
    "cache/fingerprints.json": "which runtime types (Python, torch, CUDA, GPU) have used this cache",
    "datasets": "your dataset zips or folders (put the export zip here)",
    "prepared/<speaker>": "audio codes (train_with_codes.jsonl), reused only when the dataset hash matches",
    "runs/<speaker>/<run_id>/checkpoints": "checkpoint-epoch-N, copied here from /content under .tmp names and verified",
    "runs/<speaker>/<run_id>/{config,logs,environment,samples}": "settings, logs, reproducibility snapshot, final.wav",
    "runs/<speaker>/<run_id>/status.json": "status, epoch, step, loss, latest checkpoint (metrics.csv beside it)",
    "logs": "setup logs and timings.jsonl",
}


@dataclasses.dataclass
class TrainSettings:
    """What the official scripts take as arguments. Gradient accumulation (4), bf16 precision, FlashAttention 2 and
    saving once per epoch are fixed inside ``sft_12hz.py`` and are not options, so they are not offered here."""

    epochs: int = 3  # sft_12hz.py --num_epochs (default 3)
    learning_rate: float = 2e-5  # --lr (default 2e-5; the README's step 3 example uses 2e-6)
    batch_size: int = 2  # --batch_size (default 2; the README example uses 32)
    base_model: str = BASE_MODEL  # --init_model_path
    tokenizer_model: str = TOKENIZER  # prepare_data.py --tokenizer_model_path
    drive_folder: str = DEFAULT_DRIVE_FOLDER  # everything lives under My Drive/<drive_folder>

    _RANGES = {"epochs": (1, 100), "learning_rate": (1e-7, 1e-3), "batch_size": (1, 256)}


def settings_from(spec: Optional[dict]) -> TrainSettings:
    """Validated settings; unknown keys, bad ranges and unsafe names raise ``ValueError``."""
    data = {k: v for k, v in dict(spec or {}).items() if v is not None}
    unknown = sorted(set(data) - {f.name for f in dataclasses.fields(TrainSettings)})
    if unknown:
        raise ValueError("Unknown Qwen3-TTS training setting(s): " + ", ".join(unknown) + ".")
    out = TrainSettings(**data)
    for key, (low, high) in TrainSettings._RANGES.items():
        value = getattr(out, key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
            raise ValueError(f"{key} must be between {low:g} and {high:g}.")
    if int(out.epochs) != out.epochs or int(out.batch_size) != out.batch_size:
        raise ValueError("epochs and batch_size must be whole numbers.")
    out.epochs, out.batch_size, out.learning_rate = int(out.epochs), int(out.batch_size), float(out.learning_rate)
    for key in ("base_model", "tokenizer_model"):
        if not MODEL_RE.fullmatch(getattr(out, key) or ""):
            raise ValueError(f"{key} must be a Hugging Face model id such as {BASE_MODEL}.")
    folder = (out.drive_folder or "").strip("/")
    if not FOLDER_RE.fullmatch(folder) or len(folder) > 120 or any(p in (".", "..") for p in folder.split("/")):
        raise ValueError("The Drive folder is a folder name under My Drive (letters, digits, '.', '_', '-', and '/' to nest).")
    out.drive_folder = folder
    return out


def estimate(settings: TrainSettings, clips: int) -> dict:
    """Batches and optimizer updates implied by the script's own loop; an estimate (no time or quality claim)."""
    batches = math.ceil(clips / settings.batch_size)
    return {"batches_per_epoch": batches, "updates_per_epoch": math.ceil(batches / ACCUMULATION),
            "total_updates": math.ceil(batches / ACCUMULATION) * settings.epochs,
            "note": f"Estimated from the script's loop: one batch per {settings.batch_size} clips, one optimizer update per "
                    f"{ACCUMULATION} batches (fixed in sft_12hz.py). Not a time or quality prediction."}


def drive_root(settings: TrainSettings) -> str:
    return f"/content/drive/MyDrive/{settings.drive_folder}"


def commands(settings: TrainSettings, speaker: str, folder_name: str) -> dict:
    """The two official commands with this export's values (what the runner executes), for reading and for manual use."""
    drive = f"{drive_root(settings)}/prepared/{speaker}"
    ft = "/content/Qwen3-TTS/finetuning"
    return {
        "checkpoint_dir": f"{drive_root(settings)}/runs/{speaker}/<run_id>/checkpoints",
        "prepare": (f"python {ft}/prepare_data.py --device cuda:0 --tokenizer_model_path "
                    f"/content/models/{settings.tokenizer_model.split('/')[-1]} --input_jsonl train.jsonl "
                    f"--output_jsonl {drive}/train_with_codes.jsonl"),
        "train": (f"python {ft}/sft_12hz.py --init_model_path /content/models/{settings.base_model.split('/')[-1]} "
                  f"--output_model_path /content/runs/{speaker}/<run_id>/checkpoints --train_jsonl /content/work/{speaker}/train_with_codes.jsonl "
                  f"--batch_size {settings.batch_size} --lr {settings.learning_rate:g} --num_epochs {settings.epochs} "
                  f"--speaker_name {speaker}"),
    }


def runner_args(settings: TrainSettings, speaker: str, data_source: str) -> str:
    """The runner's flags for this export (the `colab` CLI route and the shell route pass them explicitly)."""
    return " ".join([
        "--drive-root", shlex.quote(drive_root(settings)), "--data-source", shlex.quote(data_source),
        "--speaker-name", shlex.quote(speaker), "--epochs", str(settings.epochs), "--lr", f"{settings.learning_rate:g}",
        "--batch-size", str(settings.batch_size), "--base-model", shlex.quote(settings.base_model),
        "--tokenizer-model", shlex.quote(settings.tokenizer_model)])


def start_commands(speaker: str, folder_name: str, zip_name: str, settings: TrainSettings) -> dict:
    """How to start the run; shown in the export dialog's success box and written to colab_commands.txt."""
    root = drive_root(settings)
    zip_path = f"{root}/datasets/{zip_name}"
    return {
        "notebook": [
            "from google.colab import drive; drive.mount('/content/drive')",
            f"!unzip -q -o \"{zip_path}\" {RUNNER_NAME} training_settings.json -d /content/vs",
            f"!python3 /content/vs/{RUNNER_NAME} all --settings /content/vs/training_settings.json --data-source \"{zip_path}\"",
            f"# or open {NOTEBOOK_NAME} (inside the export folder and its zip) in Colab",
        ],
        "colab_cli": [
            f"colab new -s {SESSION} --gpu A100", f"colab drivemount -s {SESSION}",
            f"colab upload -s {SESSION} {RUNNER_NAME} /content/{RUNNER_NAME}",
            f"colab exec -s {SESSION} --timeout 60 (starts: nohup python3 /content/{RUNNER_NAME} all "
            f"{runner_args(settings, speaker, zip_path)} > {root}/logs/cli-train.log 2>&1 &)",
            f"colab stop -s {SESSION}   # when finished: release the GPU",
        ],
        "notebook_file": NOTEBOOK_NAME,
        "runner_file": RUNNER_NAME,
        "data_source": zip_path,
        "drive_layout": {f"{root}/{k}": v for k, v in LAYOUT.items()},
    }


def verified_facts() -> dict:
    return {
        "repository": QWEN_REPO.removesuffix(".git"), "commit": QWEN_COMMIT, "commit_date": QWEN_COMMIT_DATE,
        "read_on": READ_ON, "base_model": BASE_MODEL, "tokenizer": TOKENIZER,
        "files_read": ["finetuning/README.md", "finetuning/prepare_data.py", "finetuning/sft_12hz.py",
                       "finetuning/dataset.py", "pyproject.toml"],
        "record_fields": ["audio", "text", "ref_audio"], "audio": "wav, read with librosa; reference must be 24 kHz",
        "scripts_run_by_voicestudio": False,
    }


# ------------------------------------------------------------------------------------------------ text templates
SETUP = r"""#!/usr/bin/env bash
# Qwen3-TTS fine-tuning on Google Colab. Written by VoiceStudio. All the work is done by voicestudio_qwen3_colab.py
# (next to this file): Drive cache layout, dependency checks, FlashAttention wheel cache, models, dataset checks,
# training. Not run by VoiceStudio: it needs a Colab GPU runtime. Mount Google Drive first (colab_commands.txt), then:
#   !bash /content/@@DIR@@/colab_setup.sh
# Settings come from training_settings.json (dialog values); override with flags or environment variables, e.g.
#   !RESTART=1 EPOCHS=5 bash /content/@@DIR@@/colab_setup.sh          or      bash colab_setup.sh --smoke-run
# STEP=setup|data|train|results|sample|all (default all).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$HERE/voicestudio_qwen3_colab.py" "${STEP:-all}" "$@"
"""

COMMANDS = r"""VoiceStudio: Qwen3-TTS fine-tuning on Google Colab, everything reusable on Google Drive
======================================================================================
Verified against QwenLM/Qwen3-TTS commit @@COMMIT@@ (@@COMMIT_DATE@@): finetuning/README.md, prepare_data.py,
sft_12hz.py, dataset.py, pyproject.toml. VoiceStudio did not run them (no GPU here). Treat the first Colab run as the real test.
All logic is in ONE file, voicestudio_qwen3_colab.py; the notebook, colab_setup.sh and the commands below only call it.

Before you start
  1. Copy @@ZIP@@ to Google Drive:  My Drive/@@DRIVE@@/datasets/@@ZIP@@   (or point DATA_SOURCE at the folder or zip)
  2. Colab: Runtime > Change runtime type > a GPU. A100, H100 and L4 use the original script with FlashAttention 2.
     The G4 (RTX PRO 6000 Blackwell) defaults to a patched SDPA copy of the script (UNVERIFIED). A T4 needs ATTN_FALLBACK=True.
  3. First run: set SMOKE_RUN to run a short timing run and see the projected time and compute units before the long run.

Drive layout (My Drive/@@DRIVE@@/)
  cache/models/ cache/Qwen3-TTS/ cache/wheelhouse/sm<arch>/ cache/pip/ cache/fingerprints.json
  datasets/   prepared/@@SPEAKER@@/train_with_codes.jsonl (+ prepared.json: the dataset hash it belongs to)   logs/
  runs/@@SPEAKER@@/<run_id>/{config,logs,checkpoints,samples,environment}/  status.json  metrics.csv   (run_id = timestamp unless you set RUN_ID)
  Nothing is overwritten or deleted by default: every run has its own folder. Checkpoints are written to /content, copied to Drive as
  checkpoint-epoch-N.tmp, verified (file sizes, hashes of small files) and renamed only when complete.

----- Route A: the notebook (recommended) -----
  Open @@NOTEBOOK@@ from Drive in Colab (it is inside this export and its zip: put it anywhere on Drive and open it
  with Google Colab). Pick the GPU runtime, edit the parameters cell (DATA_SOURCE is where Qwen3 finds the training
  data), then run the cells top to bottom.

----- Route B: three notebook cells, no notebook file -----
    from google.colab import drive; drive.mount('/content/drive')
    !unzip -q -o "/content/drive/MyDrive/@@DRIVE@@/datasets/@@ZIP@@" voicestudio_qwen3_colab.py training_settings.json -d /content/vs
    !python3 /content/vs/voicestudio_qwen3_colab.py all --settings /content/vs/training_settings.json --data-source "/content/drive/MyDrive/@@DRIVE@@/datasets/@@ZIP@@"
  Steps one at a time: replace `all` with setup, data, train, results or sample. Timing run: add --smoke-run.
  Other flags: --restart, --keep-last-n N, --attn-fallback auto|true|false, --batch-size N|auto, --units-per-hour R, --units-available U,
  --compare-epoch-seconds S --compare-units-per-hour R.
  Keep the tab open; a disconnect ends the run (see Resume below).

What the runner does, written out (the exact commands it runs for training; paths are the defaults):
    git clone @@REPO@@ /content/drive/MyDrive/@@DRIVE@@/cache/Qwen3-TTS ; git checkout @@COMMIT@@   (then copied to /content/Qwen3-TTS)
    pip install -e /content/Qwen3-TTS ; pip install accelerate tensorboard librosa soundfile safetensors huggingface_hub
    FlashAttention 2 wheel: Drive cache, else the prebuilt release wheel, else one build saved to the Drive wheelhouse
    models downloaded once into Drive cache/models, then copied to /content/models/@@BASE_NAME@@ and /content/models/@@TOKENIZER_NAME@@
    cd /content/@@DIR@@
    (prepare_data.py is re-driven by a small script with the same tokenizer calls so it can show progress and write a partial file;
     the equivalent official command is:)
    python /content/Qwen3-TTS/finetuning/prepare_data.py --device cuda:0 --tokenizer_model_path /content/models/@@TOKENIZER_NAME@@ \
        --input_jsonl train.jsonl --output_jsonl /content/drive/MyDrive/@@DRIVE@@/prepared/@@SPEAKER@@/train_with_codes.jsonl
    python /content/Qwen3-TTS/finetuning/sft_12hz.py --init_model_path /content/models/@@BASE_NAME@@ \
        --output_model_path /content/runs/@@SPEAKER@@/<run_id>/checkpoints \
        --train_jsonl /content/work/@@SPEAKER@@/train_with_codes.jsonl \
        --batch_size @@BATCH@@ --lr @@LR@@ --num_epochs @@EPOCHS@@ --speaker_name @@SPEAKER@@
    (each finished checkpoint-epoch-N is then copied to Drive runs/@@SPEAKER@@/<run_id>/checkpoints as .tmp, verified, renamed)
  (Plain python, not torchrun: sft_12hz.py creates its own accelerate Accelerator and was written for one process.
   On the SDPA path the runner runs a patched COPY of sft_12hz.py from /content/qwen3_patched; the cached repo is never changed.)

Checkpoints: /content/drive/MyDrive/@@DRIVE@@/runs/@@SPEAKER@@/<run_id>/checkpoints/checkpoint-epoch-0, -1, -2 ...
  One folder per finished epoch. Each is a complete model folder (the script copies the whole base model folder into it), so expect
  several GB per epoch; the timing run and the first synced checkpoint print the exact size and the projected Drive use.
  KEEP_ALL is on by default (nothing is deleted). With KEEP_ALL=False, KEEP_LAST_N / KEEP_FIRST / KEEP_FINAL delete the other epochs, and only
  folders this runner verified and recorded.

Resume after a disconnect
  The official sft_12hz.py has NO resume: it saves model weights only (no optimizer state, no epoch counter, no
  resume option) and always starts from --init_model_path. What you can rely on:
    - Every checkpoint-epoch-N that finished before the disconnect is already on Drive and is a usable model.
    - Running the runner again re-uses the Drive cache (models, wheels, repo, codes if the dataset hash matches) and the dataset zip, and starts
      a NEW run folder (a new timestamp), so the earlier run is never touched. Re-using a --run-id that already has checkpoints is refused unless
      --restart is given.
    - True resume (optimizer state, mid-epoch) is NOT built: the official script saves weights only.
  Continuing from a saved checkpoint is not offered: the script rewrites each checkpoint as a custom_voice model and
  drops its speaker encoder, which the training code calls, so passing one as --init_model_path is untested and
  probably fails. Use fewer epochs per session instead, or pick the last finished epoch.

Compute units
  A runtime uses units while it is connected, including when idle. The runner prints a reminder at the end of every training
  or timing run. Disconnect and delete the runtime: in the notebook run
      from google.colab import runtime; runtime.unassign()
  (announced by the Colab team in 2022; not run here) or use the menu Runtime > Disconnect and delete runtime;
  from the colab command line: colab stop -s @@SESSION@@

Copy the final checkpoint elsewhere
  It already is on Drive. From a Colab cell:
    !cp -r "/content/drive/MyDrive/@@DRIVE@@/runs/@@SPEAKER@@/<run_id>/checkpoints/checkpoint-epoch-2" "/content/drive/MyDrive/@@DRIVE@@/@@SPEAKER@@-final"

----- Route C: the `colab` command-line tool -----
Syntax read from `colab readme` and `colab help <command>`. VoiceStudio did not run any of these. Its README says
the tool supports Linux and macOS (Windows is not listed): on Windows run it from WSL if a command misbehaves.
`colab exec` runs PYTHON in the session's kernel and reads code from stdin or -f FILE; its default timeout is 30 s, so
long jobs are started in the background and followed through a log file on Drive.

    colab new -s @@SESSION@@ --gpu A100              # or another GPU your plan offers
    colab drivemount -s @@SESSION@@                  # mounts at /content/drive (a Google sign-in prompt may appear)
    colab upload -s @@SESSION@@ voicestudio_qwen3_colab.py /content/voicestudio_qwen3_colab.py     # a small file
  (Put the zip on Drive first, in drive.google.com or Google Drive for desktop: uploading multi-GB files is slow.)

    colab exec -s @@SESSION@@ --timeout 60 <<'PY'
    import subprocess
    subprocess.Popen("nohup python3 /content/voicestudio_qwen3_colab.py all @@RUNNER_ARGS@@ > /content/drive/MyDrive/@@DRIVE@@/logs/cli-train.log 2>&1 &",
                     shell=True, start_new_session=True)
    PY

  Follow it (the log is on Drive, so it survives too):
    colab exec -s @@SESSION@@ <<'PY'
    print(open('/content/drive/MyDrive/@@DRIVE@@/logs/cli-train.log').read()[-3000:])
    PY
    colab ls -s @@SESSION@@ content/drive/MyDrive/@@DRIVE@@/runs/@@SPEAKER@@
    colab status -s @@SESSION@@

  Fetch a checkpoint file to this computer (not needed, the checkpoints are on Drive; each file is large):
    colab download -s @@SESSION@@ content/drive/MyDrive/@@DRIVE@@/runs/@@SPEAKER@@/<run_id>/checkpoints/checkpoint-epoch-2/model.safetensors ./model.safetensors

  Resume: same limits as above. After a lost session, `colab new` again, `colab drivemount`, and run the same command (a new run folder is made).
    colab stop -s @@SESSION@@                        # release the GPU when you are done (an idle session keeps using units)
"""

README = r"""Qwen3-TTS training set: @@SPEAKER@@
====================================
Made by VoiceStudio for the official Qwen3-TTS fine-tuning scripts (QwenLM/Qwen3-TTS, finetuning/, commit
@@COMMIT@@ of @@COMMIT_DATE@@). Target model: @@BASE@@ (single speaker).
This folder trains nothing by itself and VoiceStudio has not run Qwen's scripts on it.

Contents
  train.jsonl            one row per clip: {"audio": "audio/000001.wav", "text": "...", "ref_audio": "reference/ref.wav"}
  audio/000001.wav ...   the clips, 24 kHz mono 16-bit PCM WAV, numbered in a fixed order
  reference/ref.wav      the one reference clip every row points to (24 kHz, which Qwen's code requires)
  dataset_report.json    counts, durations, original-to-exported mapping, warnings, how the reference was chosen
  rejected_samples.jsonl every clip that was NOT exported, with the reason and its transcript
  voicestudio_qwen3_colab.py   the one script that does the Colab work (Drive cache, dependency checks, data checks, training)
  @@NOTEBOOK@@   Colab notebook with this export's settings; every cell calls the script
  colab_setup.sh         thin wrapper: bash colab_setup.sh [--smoke-run | other flags]
  colab_commands.txt     the same steps for a notebook and for the `colab` command-line tool, plus resume notes

Facts taken from Qwen's code
  - Rows hold audio, text and ref_audio; no ref_text is used. A row may also carry an optional "language" (not set here).
  - Paths are opened relative to the working directory, so the scripts run from inside this folder (the runner does that).
  - prepare_data.py adds "audio_codes" to each row (output: a new JSONL). sft_12hz.py reads that file.
  - There is no validation input and no resume option in the official scripts.
  - sft_12hz.py hard-codes FlashAttention 2 (Ampere, Ada or Hopper GPUs). The runner uses it there, and a patched SDPA copy
    only for GPUs it cannot run on (default on Blackwell / G4; opt-in elsewhere), clearly marked unverified.
  - The text is the spoken transcript only: Fish tags such as [calm] are removed; punctuation is kept.
  - The same ref_audio on every row is what Qwen recommends for a stable voice.

Quick start on Colab (details in colab_commands.txt)
  1. Copy the ZIP to Drive: My Drive/@@DRIVE@@/datasets/@@ZIP@@
  2. Open @@NOTEBOOK@@ in Colab, pick a GPU runtime, edit the parameters cell, run the cells in order.
     Or, in a notebook with Drive mounted:
       !unzip -q -o "/content/drive/MyDrive/@@DRIVE@@/datasets/@@ZIP@@" voicestudio_qwen3_colab.py training_settings.json -d /content/vs
       !python3 /content/vs/voicestudio_qwen3_colab.py all --settings /content/vs/training_settings.json --data-source "/content/drive/MyDrive/@@DRIVE@@/datasets/@@ZIP@@"
  3. Checkpoints appear in My Drive/@@DRIVE@@/runs/@@SPEAKER@@/<run_id>/checkpoints/checkpoint-epoch-N after every epoch (verified copies).
  4. Disconnect and delete the runtime afterwards (an idle session keeps using compute units).

Without the helper, from inside this folder (after cloning and installing Qwen3-TTS):
  python PATH_TO_QWEN/finetuning/prepare_data.py --device cuda:0 --tokenizer_model_path @@TOKENIZER@@ --input_jsonl train.jsonl --output_jsonl train_with_codes.jsonl
  python PATH_TO_QWEN/finetuning/sft_12hz.py --init_model_path LOCAL_COPY_OF_@@BASE_NAME@@ --output_model_path OUTPUT_DIR --train_jsonl train_with_codes.jsonl --batch_size @@BATCH@@ --lr @@LR@@ --num_epochs @@EPOCHS@@ --speaker_name @@SPEAKER@@
  (init_model_path must be a local folder: the script copies it into every checkpoint.)

Training settings (training_settings.json): @@SETTINGS@@

This set
@@SUMMARY@@

Source and terms: the audio is synthetic speech generated by Fish Audio. Check Fish's terms before using it to train a model.
"""


# ------------------------------------------------------------------------------------------------ notebook
def _cell(kind: str, n: int, text: str) -> dict:
    lines = text.strip("\n").split("\n")
    source = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    base = {"cell_type": kind, "id": f"vs-{n:02d}", "metadata": {}, "source": source}
    if kind == "code":
        base |= {"execution_count": None, "outputs": []}
    return base


def _py(value) -> str:
    return json.dumps(value, ensure_ascii=False) if isinstance(value, str) else repr(value)


INTRO_MD = """# Qwen3-TTS fine-tuning (VoiceStudio)

Trains Qwen3-TTS on the dataset folder VoiceStudio exported, using Qwen's official `finetuning/` scripts. Everything reusable is cached on your Google Drive, and every cell below calls one script (`voicestudio_qwen3_colab.py`, shipped with the export). **Nothing here has been run on Colab by its author: treat the first run as the test.**

**Checklist**
1. **Runtime > Change runtime type > pick the GPU.** A100, H100 and L4 use the original script (FlashAttention 2). The **G4** (RTX PRO 6000 Blackwell) uses a patched SDPA copy by default (unverified). A **T4** cannot run FlashAttention 2 and needs `ATTN_FALLBACK = True` (unverified, may run out of memory).
2. **Compute units.** A runtime uses units while connected, idle or not, and each GPU type has its own rate (see Colab's runtime menu). Run once with `SMOKE_RUN = True` for a short timing run that projects time and units, then set it back to `False`.
3. Copy the export zip (or folder) to Drive and set `DATA_SOURCE` below to it.
4. Run the cells top to bottom. The first run on a new runtime type installs packages and downloads models; later runs reuse the Drive cache. FlashAttention is used only when a ready-made package exists for the runtime (newest Colab runtimes, such as Python 3.13 with a new CUDA, often have none): then PyTorch's built-in attention is used and **nothing is compiled** unless you set `BUILD_FLASH_ATTN = True`.
5. When finished, **disconnect and delete the runtime** (last cell, or Runtime > Disconnect and delete runtime).
6. **To cancel fast:** press the stop button on the running cell (it stops the training and first saves the epochs that already finished), then **Runtime > Disconnect and delete runtime**. Compute units are used while the runtime is connected, not while a cell is running, so disconnecting is what stops the spending. Finished epochs stay on Drive; the epoch in progress is lost, because the official script cannot resume. If anything looks stuck, go straight to Disconnect and delete runtime."""


def render_notebook(*, speaker: str = DEFAULT_SPEAKER, folder_name: str = DEFAULT_FOLDER_NAME, zip_name: Optional[str] = None,
                    settings: Optional[TrainSettings] = None) -> str:
    """The Colab notebook for one export (nbformat 4). Same input, same text."""
    st = settings or TrainSettings()
    zip_name = zip_name or folder_name + ".zip"
    root = drive_root(st)
    params = "\n".join([
        "# ---- Edit these, then run the cells below in order. Re-run this cell and the 'mount Drive' cell after changing them. ----",
        f"DRIVE_ROOT = {_py(root)}  # cache, checkpoints and logs live here",
        f"DATA_SOURCE = {_py(f'{root}/datasets/{zip_name}')}  # where the training data is: a .zip or a folder on Drive, or an unzipped folder under /content",
        f"SPEAKER_NAME = {_py(speaker)}  # the name the fine-tuned model answers to",
        f"EPOCHS = {st.epochs}",
        f"LR = {st.learning_rate:g}",
        f"BATCH_SIZE = {st.batch_size}  # a number, or 'auto' for the suggestion for this GPU (untested estimates, see the GPU section of the output)",
        f"BASE_MODEL = {_py(st.base_model)}",
        "RUN_ID = ''  # empty: a new timestamped run folder every time, so a second run never overwrites the first",
        "KEEP_ALL = True  # True keeps every checkpoint (each is several GB). Nothing is ever deleted while this is True",
        "KEEP_LAST_N = 0  # only with KEEP_ALL = False: keep the newest N epochs (plus KEEP_FIRST / KEEP_FINAL)",
        "KEEP_FIRST = False",
        "KEEP_FINAL = True",
        "ATTN_FALLBACK = 'auto'  # 'auto': FlashAttention 2 when a ready-made package exists for this runtime, otherwise PyTorch's built-in attention (patched copy of the script, unverified); True: always the patched copy; False: require FlashAttention 2",
        "FLASH_WHEEL = ''  # optional: the full path of a flash_attn .whl you put on Drive (or a URL) for THIS runtime's Python/torch/CUDA; tried first. Leave empty for the automatic choice.",
        "BUILD_FLASH_ATTN = False  # True: compile FlashAttention from source when no ready-made package exists. Takes an hour or more, may fail on a new Python/CUDA, and uses compute units the whole time. Leave False.",
        "RESTART = False  # True lets a RUN_ID you re-use replace its own checkpoints (the official script cannot resume)",
        "SAMPLE_TEXT = " + _py(DEFAULT_SAMPLE_TEXT) + "  # sentence for final.wav after training",
        "SAMPLE_SEED = 1234",
        "CACHE_MODELS_ON_DRIVE = True  # False: download models straight to /content (nothing cached on Drive)",
        "SMOKE_RUN = False  # True: a short timing run (one epoch on the first SMOKE_CLIPS clips, scratch folder) instead of real training",
        "SMOKE_CLIPS = 40",
        "UNITS_PER_HOUR = None  # this runtime's compute units per hour; empty = looked up for known GPUs (owner's figures, unverified)",
        "UNITS_AVAILABLE = 100",
        "COMPARE_EPOCH_SECONDS = None  # optional: seconds per epoch measured on a different GPU, to compare cost per epoch",
        "COMPARE_UNITS_PER_HOUR = None  # that other GPU's rate",
        f"TOKENIZER_MODEL = {_py(st.tokenizer_model)}",
    ])
    mount = "\n".join([
        "from google.colab import drive",
        "drive.mount('/content/drive')",
        "",
        "# Load the one script that does the work: from the export (DATA_SOURCE) if it is there, else the copy cached on Drive.",
        "import os, sys, zipfile",
        "SCRIPT = 'voicestudio_qwen3_colab.py'",
        "if DRIVE_ROOT.startswith('/content/drive/'):",
        "    # Never create the Drive folder from here: if Colab cannot see the real one yet, creating it makes a second folder with the same name.",
        "    import time",
        "    MY_DRIVE = '/content/drive/MyDrive'",
        "    for _ in range(30):  # the mount can take a moment to fill in",
        "        if os.path.isdir(MY_DRIVE) and os.listdir(MY_DRIVE):",
        "            break",
        "        time.sleep(2)",
        "    if not os.path.isdir(DRIVE_ROOT):",
        "        near = [n for n in sorted(os.listdir(MY_DRIVE)) if 'qwen' in n.lower() or 'tts' in n.lower()][:10] if os.path.isdir(MY_DRIVE) else []",
        "        raise FileNotFoundError(DRIVE_ROOT + ' does not exist, so nothing was created. Create that folder in Google Drive with exactly this name and wait until it has finished syncing, or change DRIVE_ROOT. Similar folders in My Drive: ' + str(near or 'none'))",
        "CACHE = os.path.join(DRIVE_ROOT, 'cache')",
        "os.makedirs(CACHE, exist_ok=True)",
        "cached = os.path.join(CACHE, SCRIPT)",
        "import re",
        "def _version(b):",
        "    m = re.search(rb'^RUNNER_VERSION\\s*=\\s*\"(\\d+)\"', b or b'', re.M)",
        "    return int(m.group(1)) if m else 0",
        "fresh = None",
        "if DATA_SOURCE.lower().endswith('.zip') and os.path.isfile(DATA_SOURCE):",
        "    with zipfile.ZipFile(DATA_SOURCE) as z:",
        "        hits = [n for n in z.namelist() if n == SCRIPT or n.endswith('/' + SCRIPT)]",
        "        fresh = z.read(hits[0]) if hits else None",
        "elif os.path.isfile(os.path.join(DATA_SOURCE, SCRIPT)):",
        "    fresh = open(os.path.join(DATA_SOURCE, SCRIPT), 'rb').read()",
        "have = open(cached, 'rb').read() if os.path.isfile(cached) else None",
        "# The newer runner wins, so a fixed script can be dropped into the cache folder without re-uploading the zip.",
        "if fresh is not None and (have is None or _version(fresh) >= _version(have)):",
        "    if have != fresh:",
        "        open(cached, 'wb').write(fresh)",
        "elif fresh is not None:",
        "    print('Using the newer runner already in', cached, '(version', _version(have), 'over the zip\\'s', str(_version(fresh)) + ')')",
        "if not os.path.isfile(cached):",
        "    raise FileNotFoundError(SCRIPT + ' not found: check DATA_SOURCE (it must point at the export zip or folder, which contains it).')",
        "if CACHE not in sys.path:",
        "    sys.path.insert(0, CACHE)",
        "import importlib, voicestudio_qwen3_colab as vs",
        "importlib.reload(vs)",
        "P = vs.params_from(dict(drive_root=DRIVE_ROOT, data_source=DATA_SOURCE, speaker_name=SPEAKER_NAME, epochs=EPOCHS, lr=LR,",
        "    batch_size=BATCH_SIZE, base_model=BASE_MODEL, tokenizer_model=TOKENIZER_MODEL, run_id=RUN_ID, keep_all=KEEP_ALL, keep_last_n=KEEP_LAST_N,",
        "    keep_first=KEEP_FIRST, keep_final=KEEP_FINAL, sample_text=SAMPLE_TEXT, sample_seed=SAMPLE_SEED, cache_models_on_drive=CACHE_MODELS_ON_DRIVE,",
        "    attn_fallback=ATTN_FALLBACK, build_flash_attn=BUILD_FLASH_ATTN, flash_wheel=FLASH_WHEEL, restart=RESTART, smoke_run=SMOKE_RUN, smoke_clips=SMOKE_CLIPS, units_per_hour=UNITS_PER_HOUR, units_available=UNITS_AVAILABLE,",
        "    compare_epoch_seconds=COMPARE_EPOCH_SECONDS, compare_units_per_hour=COMPARE_UNITS_PER_HOUR), env={})",
        "print('Runner', vs.RUNNER_VERSION, 'loaded from', cached)",
    ])
    cells = [
        _cell("markdown", 1, INTRO_MD),
        _cell("code", 2, "# GPU check: which GPU did Colab give you? (details and the attention path are printed again by the setup cell)\n"
                         "!nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv"),
        _cell("markdown", 3, "## Parameters\nEdit the values in the next cell. `DATA_SOURCE` is where the training data is: the export zip or folder on Drive "
                             "(or an already unzipped folder under `/content`). The cells after it do the work and hold no settings of their own."),
        _cell("code", 4, params),
        _cell("markdown", 5, "## 1. Mount Drive and load the script"),
        _cell("code", 6, mount),
        _cell("markdown", 7, "## 2. Set up and verify\nCreates the Drive layout, installs what is missing, gets FlashAttention (cached wheel, prebuilt wheel or one build "
                             "saved to Drive) when the GPU needs it, downloads the models once, and prints a pass/fail table plus the time each step took."),
        _cell("code", 8, "ok = vs.setup(P)"),
        _cell("markdown", 9, "## 3. Check the data\nCopies or unzips the dataset to local disk and verifies every `audio` and `ref_audio` path (exists, 24 kHz mono) before any GPU time is spent."),
        _cell("code", 10, "folder = vs.data_check(P)"),
        _cell("markdown", 11, "## 4. Train (or run the timing run)\nWith `SMOKE_RUN = True` this runs one epoch on the first few clips and prints the projected "
                              "time and compute units; with `False` it trains for real. Checkpoints are written to `/content`, then each finished epoch is copied to "
                              "`runs/<speaker>/<run_id>/checkpoints` on Drive under a temporary name, verified and renamed, and the output says what is now safe on Drive. "
                              "Progress lines, a heartbeat when quiet, `metrics.csv` and `status.json` show it is alive. Re-running the cell starts a new run unless you re-use a `RUN_ID`."),
        _cell("code", 12, "vs.train(P)"),
        _cell("markdown", 13, "## 5. Results\nLists the checkpoints on Drive with their sizes."),
        _cell("code", 14, "vs.results(P)"),
        _cell("markdown", 15, "## 6. Final sample (optional, unverified)\nTraining already makes `samples/final.wav` when it finishes. Run this to make it again (for example with a new `SAMPLE_TEXT`) "
                              "from the final checkpoint on Drive, using the usage in Qwen's finetuning README."),
        _cell("code", 16, "vs.final_sample(P)"),
        _cell("markdown", 17, "## 7. Disconnect\nAn idle runtime keeps using compute units. Uncomment and run the line below, or use Runtime > Disconnect and delete runtime. "
                              "(`runtime.unassign()` was announced by the Colab team in 2022; it is not run here.)"),
        _cell("code", 18, "# from google.colab import runtime; runtime.unassign()"),
    ]
    nb = {"cells": cells,
          "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                       "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    return json.dumps(nb, ensure_ascii=False, indent=1) + "\n"


def write_default_notebook(path: str) -> None:
    """Refresh the committed copy: ``cd backend && python -c "from engines.fish_api import qwen3_colab as c; c.write_default_notebook('../notebooks/Qwen3_TTS_Colab_Training.ipynb')"``."""
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(render_notebook())


def runner_source() -> str:
    """The Colab script, byte for byte as it is in the repo."""
    with open(runner.__file__, encoding="utf-8", newline="") as fh:
        return fh.read().replace("\r\n", "\n")


def _fill(template: str, values: dict) -> str:
    for key, value in values.items():
        template = template.replace(f"@@{key}@@", value)
    return template


def _summary(report: dict) -> str:
    c, d, ref = report["counts"], report["duration"], report["reference"]
    rows = [
        f"  Speaker: {report['speaker']['name']} (model speaker name: {report['speaker']['qwen_speaker_name']})",
        f"  Source samples: {c['source_samples']}   Accepted: {c['accepted']}   Rejected: {c['rejected']}",
        f"  Accepted duration: {d['total']}   average {d['average_s']} s   median {d['median_s']} s   "
        f"shortest {d['shortest_s']} s   longest {d['longest_s']} s",
        "  Audio: 24000 Hz, mono, WAV PCM16",
        f"  Reference: {ref['path']} from {ref['source']} ({ref['duration']} s): {ref['reason']}",
        f"  Train samples: {c['train']}   Validation samples: 0 (Qwen's scripts have none)",
    ]
    rows += [f"  WARNING: {w['message']}" for w in report["warnings"]]
    return "\n".join(rows)


def render_files(*, speaker: str, folder_name: str, zip_name: str, settings: TrainSettings, report: dict) -> dict:
    """The helper files for one export, as ``{file name: text}``."""
    values = {
        "SPEAKER": speaker, "DIR": folder_name, "ZIP": zip_name, "DRIVE": settings.drive_folder, "COMMIT": QWEN_COMMIT,
        "COMMIT_DATE": QWEN_COMMIT_DATE, "BASE": settings.base_model, "BASE_NAME": settings.base_model.split("/")[-1],
        "TOKENIZER": settings.tokenizer_model, "TOKENIZER_NAME": settings.tokenizer_model.split("/")[-1], "SESSION": SESSION,
        "BATCH": str(settings.batch_size), "LR": f"{settings.learning_rate:g}", "EPOCHS": str(settings.epochs),
        "REPO": QWEN_REPO, "NOTEBOOK": NOTEBOOK_NAME,
        "RUNNER_ARGS": runner_args(settings, speaker, f"{drive_root(settings)}/datasets/{zip_name}"),
        "SETTINGS": f"{settings.epochs} epochs, learning rate {settings.learning_rate:g}, batch size {settings.batch_size}, base model {settings.base_model}",
    }
    return {
        "README.txt": _fill(README, values | {"SUMMARY": _summary(report)}),
        "colab_setup.sh": _fill(SETUP, values),
        "colab_commands.txt": _fill(COMMANDS, values),
        RUNNER_NAME: runner_source(),
        NOTEBOOK_NAME: render_notebook(speaker=speaker, folder_name=folder_name, zip_name=zip_name, settings=settings),
    }
