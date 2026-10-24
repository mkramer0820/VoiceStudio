"""Text written next to a Qwen3-TTS export: README, the Colab setup script and the command sheet.

Everything here was written from the official ``finetuning/`` files of QwenLM/Qwen3-TTS as they stood at
``QWEN_COMMIT`` (README.md, prepare_data.py, sft_12hz.py, dataset.py). The scripts and the commands were
NOT run: that needs a GPU runtime. Only the speaker name, the folder names and the Drive folder vary, and
they are validated to be shell-safe before they reach a script.
"""
from __future__ import annotations

import dataclasses
import math
import re
from typing import Optional

QWEN_REPO = "https://github.com/QwenLM/Qwen3-TTS.git"
QWEN_COMMIT = "022e286b98fbec7e1e916cb940cdf532cd9f488e"
QWEN_COMMIT_DATE = "2026-03-17"
READ_ON = "2026-10-24"
BASE_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
TOKENIZER = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
SAMPLE_RATE = 24000  # dataset.py: extract_mels asserts "Only support 24kHz audio" for ref_audio
DEFAULT_DRIVE_FOLDER = "VoiceStudio-training"
SPEAKER_RE = re.compile(r"[A-Za-z0-9_-]{1,40}")
FOLDER_RE = re.compile(r"[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*")  # a folder under My Drive, may be nested
MODEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")  # a Hugging Face model id
ACCUMULATION = 4  # sft_12hz.py: Accelerator(gradient_accumulation_steps=4), not an option
SESSION = "qwen-train"


@dataclasses.dataclass
class TrainSettings:
    """What the official scripts take as arguments. Gradient accumulation (4), bf16 precision, FlashAttention 2 and
    saving once per epoch are fixed inside ``sft_12hz.py`` and are not options, so they are not offered here."""

    epochs: int = 3  # sft_12hz.py --num_epochs (default 3)
    learning_rate: float = 2e-5  # --lr (default 2e-5; the README's step 3 example uses 2e-6)
    batch_size: int = 2  # --batch_size (default 2; the README example uses 32)
    base_model: str = BASE_MODEL  # --init_model_path
    tokenizer_model: str = TOKENIZER  # prepare_data.py --tokenizer_model_path
    drive_folder: str = DEFAULT_DRIVE_FOLDER  # checkpoints go to My Drive/<drive_folder>/<speaker>/checkpoints

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


def commands(settings: TrainSettings, speaker: str, folder_name: str) -> dict:
    """The two official commands with this export's values (what colab_setup.sh runs)."""
    drive = f"/content/drive/MyDrive/{settings.drive_folder}/{speaker}"
    return {
        "checkpoint_dir": f"{drive}/checkpoints",
        "prepare": (f"python /content/Qwen3-TTS/finetuning/prepare_data.py --device cuda:0 --tokenizer_model_path "
                    f"{settings.tokenizer_model} --input_jsonl train.jsonl --output_jsonl {drive}/train_with_codes.jsonl"),
        "train": (f"python /content/Qwen3-TTS/finetuning/sft_12hz.py --init_model_path /content/models/{settings.base_model.split('/')[-1]} "
                  f"--output_model_path {drive}/checkpoints --train_jsonl {drive}/train_with_codes.jsonl "
                  f"--batch_size {settings.batch_size} --lr {settings.learning_rate:g} --num_epochs {settings.epochs} "
                  f"--speaker_name {speaker}"),
    }


def start_commands(speaker: str, folder_name: str, zip_name: str, settings: TrainSettings) -> dict:
    """How to start the run: notebook cells and the `colab` command-line route (also in colab_commands.txt)."""
    d = f"/content/drive/MyDrive/{settings.drive_folder}/{speaker}"
    return {
        "notebook": ["from google.colab import drive; drive.mount('/content/drive')",
                     f"!mkdir -p /content/{folder_name} && unzip -q -o \"{d}/{zip_name}\" -d /content/{folder_name}",
                     f"!bash /content/{folder_name}/colab_setup.sh"],
        "colab_cli": [f"colab new -s {SESSION} --gpu A100", f"colab drivemount -s {SESSION}",
                      f"# unzip, then start in the background: see colab_commands.txt, Route B",
                      f"colab exec -s {SESSION} --timeout 60 (starts: nohup bash /content/{folder_name}/colab_setup.sh > {d}/train.log &)"],
    }


def verified_facts() -> dict:
    return {
        "repository": QWEN_REPO.removesuffix(".git"), "commit": QWEN_COMMIT, "commit_date": QWEN_COMMIT_DATE,
        "read_on": READ_ON, "base_model": BASE_MODEL, "tokenizer": TOKENIZER,
        "files_read": ["finetuning/README.md", "finetuning/prepare_data.py", "finetuning/sft_12hz.py",
                       "finetuning/dataset.py"],
        "record_fields": ["audio", "text", "ref_audio"], "audio": "wav, read with librosa; reference must be 24 kHz",
        "scripts_run_by_voicestudio": False,
    }


SETUP = r"""#!/usr/bin/env bash
# Qwen3-TTS fine-tuning on Google Colab with the checkpoints written straight to Google Drive.
# Written by VoiceStudio. Commands follow QwenLM/Qwen3-TTS finetuning/ at commit @@COMMIT@@ (@@COMMIT_DATE@@).
# Not run by VoiceStudio: it needs a Colab GPU runtime. Run:  !bash /content/@@DIR@@/colab_setup.sh
# Mount Google Drive first (see colab_commands.txt). Every value below can be overridden by an environment variable.
set -euo pipefail

SPEAKER_NAME="${SPEAKER_NAME:-@@SPEAKER@@}"                  # name the fine-tuned model answers to
DRIVE_DIR="${DRIVE_DIR:-/content/drive/MyDrive/@@DRIVE@@/${SPEAKER_NAME}}"   # survives a disconnect
DATASET_ARCHIVE="${DATASET_ARCHIVE:-${DRIVE_DIR}/@@ZIP@@}"
DATASET_DIR="${DATASET_DIR:-/content/@@DIR@@}"               # local copy: training from /content is faster than from Drive
OUTPUT_DIR="${OUTPUT_DIR:-${DRIVE_DIR}/checkpoints}"         # checkpoint-epoch-N folders are written here, on Drive
CODES_JSONL="${CODES_JSONL:-${DRIVE_DIR}/train_with_codes.jsonl}"   # prepare_data.py output, kept on Drive so it is not redone
QWEN_REPO="${QWEN_REPO:-/content/Qwen3-TTS}"
QWEN_COMMIT="${QWEN_COMMIT:-@@COMMIT@@}"
BASE_MODEL="@@BASE@@"
TOKENIZER_MODEL="@@TOKENIZER@@"
MODEL_DIR="${MODEL_DIR:-/content/models/@@BASE_NAME@@}"
BATCH_SIZE="${BATCH_SIZE:-@@BATCH@@}"      # set in the VoiceStudio export dialog (the script defaults are 2, 2e-5, 3)
LR="${LR:-@@LR@@}"
EPOCHS="${EPOCHS:-@@EPOCHS@@}"

echo "== 0/6 checks"
[ -d /content/drive/MyDrive ] || { echo "Google Drive is not mounted. Mount it first (colab_commands.txt)."; exit 1; }
nvidia-smi -L || { echo "No GPU. Runtime > Change runtime type > pick a GPU."; exit 1; }
nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | awk -F. '$1 < 8 { print "WARNING: this GPU is older than Ampere. sft_12hz.py hard-codes flash_attention_2, which needs Ampere or newer (A100, L4, H100); a T4 will fail." }' || true
if ls "${OUTPUT_DIR}"/checkpoint-epoch-* >/dev/null 2>&1 && [ "${RESTART:-0}" != "1" ]; then
  echo "Checkpoints already exist in ${OUTPUT_DIR}:"; ls -d "${OUTPUT_DIR}"/checkpoint-epoch-*
  echo "sft_12hz.py cannot resume: a new run starts from the base model and overwrites checkpoint-epoch-0, -1, ... in place."
  echo "Pick another OUTPUT_DIR, or set RESTART=1 to start over and overwrite. Nothing was changed."
  exit 2
fi

echo "== 1/6 dataset to ${DATASET_DIR}"
if [ -f "${DATASET_DIR}/train.jsonl" ]; then
  echo "already there"
elif [ -f "${DATASET_ARCHIVE}" ]; then
  mkdir -p "${DATASET_DIR}" && unzip -q -o "${DATASET_ARCHIVE}" -d "${DATASET_DIR}"
elif [ -f "${DRIVE_DIR}/@@DIR@@/train.jsonl" ]; then
  mkdir -p "${DATASET_DIR}" && cp -r "${DRIVE_DIR}/@@DIR@@/." "${DATASET_DIR}/"
else
  echo "Dataset not found. Put @@ZIP@@ in ${DRIVE_DIR} (or set DATASET_ARCHIVE / DATASET_DIR)."; exit 1
fi
[ -f "${DATASET_DIR}/train.jsonl" ] && [ -f "${DATASET_DIR}/reference/ref.wav" ] || { echo "train.jsonl or reference/ref.wav is missing in ${DATASET_DIR}"; exit 1; }
echo "$(wc -l < "${DATASET_DIR}/train.jsonl") training rows"

echo "== 2/6 official Qwen3-TTS repository"
[ -d "${QWEN_REPO}/.git" ] || git clone https://github.com/QwenLM/Qwen3-TTS.git "${QWEN_REPO}"
git -C "${QWEN_REPO}" checkout --quiet "${QWEN_COMMIT}"

echo "== 3/6 dependencies"
pip install -q -e "${QWEN_REPO}"
pip install -q accelerate tensorboard librosa soundfile safetensors huggingface_hub
if [ "${SKIP_FLASH_ATTN:-0}" != "1" ]; then
  python -c "import flash_attn" 2>/dev/null || MAX_JOBS=4 pip install -q -U flash-attn --no-build-isolation
fi

echo "== 4/6 base model ${BASE_MODEL} to ${MODEL_DIR}"
# sft_12hz.py copies the init model FOLDER into every checkpoint, so it needs a local folder, not a hub id.
[ -f "${MODEL_DIR}/config.json" ] || python -c "from huggingface_hub import snapshot_download; snapshot_download('${BASE_MODEL}', local_dir='${MODEL_DIR}')"

echo "== 5/6 official preprocessing (audio_codes)"
# Paths in train.jsonl are relative to the dataset folder, and the official scripts open them relative to the
# working directory, so every step runs from inside DATASET_DIR.
cd "${DATASET_DIR}"
if [ -s "${CODES_JSONL}" ]; then
  echo "using ${CODES_JSONL}"
else
  mkdir -p "$(dirname "${CODES_JSONL}")"
  python "${QWEN_REPO}/finetuning/prepare_data.py" --device cuda:0 --tokenizer_model_path "${TOKENIZER_MODEL}" \
    --input_jsonl train.jsonl --output_jsonl "${CODES_JSONL}.tmp"
  mv "${CODES_JSONL}.tmp" "${CODES_JSONL}"
fi

echo "== 6/6 fine-tuning (checkpoints go to ${OUTPUT_DIR} on Drive after every epoch)"
mkdir -p "${OUTPUT_DIR}"
python "${QWEN_REPO}/finetuning/sft_12hz.py" --init_model_path "${MODEL_DIR}" --output_model_path "${OUTPUT_DIR}" \
  --train_jsonl "${CODES_JSONL}" --batch_size "${BATCH_SIZE}" --lr "${LR}" --num_epochs "${EPOCHS}" \
  --speaker_name "${SPEAKER_NAME}"

echo "Done. Checkpoints:"; ls -d "${OUTPUT_DIR}"/checkpoint-epoch-*
"""

COMMANDS = r"""VoiceStudio: Qwen3-TTS fine-tuning on Google Colab, checkpoints on Google Drive
===============================================================================
Verified against QwenLM/Qwen3-TTS commit @@COMMIT@@ (@@COMMIT_DATE@@): finetuning/README.md, prepare_data.py,
sft_12hz.py, dataset.py. VoiceStudio did not run them (no GPU here). Treat the first Colab run as the real test.

Before you start
  1. Copy @@ZIP@@ to Google Drive:  My Drive/@@DRIVE@@/@@SPEAKER@@/@@ZIP@@
  2. Colab: Runtime > Change runtime type > a GPU with FlashAttention 2 support (A100, L4 or H100).
     sft_12hz.py hard-codes flash_attention_2, so a T4 will not work without editing Qwen's script.

----- Route A: Colab notebook (three cells) -----

Cell 1: mount Drive
    from google.colab import drive
    drive.mount('/content/drive')

Cell 2: unzip the dataset to /content (local disk is faster than Drive)
    !mkdir -p /content/@@DIR@@
    !unzip -q -o "/content/drive/MyDrive/@@DRIVE@@/@@SPEAKER@@/@@ZIP@@" -d /content/@@DIR@@

Cell 3: clone, install, preprocess and train (everything in colab_setup.sh)
    !bash /content/@@DIR@@/colab_setup.sh

  Change the speaker, epochs and so on with variables in front of the command:
    !SPEAKER_NAME=@@SPEAKER@@ EPOCHS=@@EPOCHS@@ BATCH_SIZE=@@BATCH@@ LR=@@LR@@ bash /content/@@DIR@@/colab_setup.sh
  Skip the FlashAttention install if it is already present:  SKIP_FLASH_ATTN=1
  Keep the notebook tab open; a disconnect ends the run (see Resume below).

The steps colab_setup.sh runs, written out (DATASET_DIR=/content/@@DIR@@, OUTPUT_DIR=Drive checkpoints folder):
    git clone https://github.com/QwenLM/Qwen3-TTS.git /content/Qwen3-TTS
    git -C /content/Qwen3-TTS checkout @@COMMIT@@
    pip install -e /content/Qwen3-TTS
    pip install accelerate tensorboard librosa soundfile safetensors huggingface_hub
    MAX_JOBS=4 pip install -U flash-attn --no-build-isolation
    python -c "from huggingface_hub import snapshot_download; snapshot_download('@@BASE@@', local_dir='/content/models/@@BASE_NAME@@')"
    cd /content/@@DIR@@
    python /content/Qwen3-TTS/finetuning/prepare_data.py --device cuda:0 --tokenizer_model_path @@TOKENIZER@@ \
        --input_jsonl train.jsonl --output_jsonl /content/drive/MyDrive/@@DRIVE@@/@@SPEAKER@@/train_with_codes.jsonl
    python /content/Qwen3-TTS/finetuning/sft_12hz.py --init_model_path /content/models/@@BASE_NAME@@ \
        --output_model_path /content/drive/MyDrive/@@DRIVE@@/@@SPEAKER@@/checkpoints \
        --train_jsonl /content/drive/MyDrive/@@DRIVE@@/@@SPEAKER@@/train_with_codes.jsonl \
        --batch_size @@BATCH@@ --lr @@LR@@ --num_epochs @@EPOCHS@@ --speaker_name @@SPEAKER@@
  (Plain python, not torchrun: sft_12hz.py creates its own accelerate Accelerator and was written for one process.)

Checkpoints: /content/drive/MyDrive/@@DRIVE@@/@@SPEAKER@@/checkpoints/checkpoint-epoch-0, -1, -2 ...
  One folder per finished epoch, written directly to Drive. Each is a complete model folder (the script copies the
  whole base model folder into it), so expect several GB per epoch (about the size of the base model); check your
  free Drive space against the number of epochs, and delete older epochs you do not need.

Test a checkpoint (same Colab runtime):
    import torch, soundfile as sf
    from qwen_tts import Qwen3TTSModel
    tts = Qwen3TTSModel.from_pretrained("/content/drive/MyDrive/@@DRIVE@@/@@SPEAKER@@/checkpoints/checkpoint-epoch-2",
        device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    wavs, sr = tts.generate_custom_voice(text="She said she would be here by noon.", speaker="@@SPEAKER@@")
    sf.write("/content/test.wav", wavs[0], sr)

Resume after a disconnect
  The official sft_12hz.py has NO resume: it saves model weights only (no optimizer state, no epoch counter, no
  resume option) and always starts from --init_model_path. What you can rely on:
    - Every checkpoint-epoch-N that finished before the disconnect is already on Drive and is a usable model.
    - Rerunning colab_setup.sh re-uses the dataset zip, the cached train_with_codes.jsonl and the Drive folder, but it
      refuses to run over existing checkpoints (it would restart from the base model and overwrite them). Use another
      OUTPUT_DIR for a new run, or RESTART=1 to deliberately start over.
  Continuing from a saved checkpoint is not offered: the script rewrites each checkpoint as a custom_voice model and
  drops its speaker encoder, which the training code calls, so passing one as --init_model_path is untested and
  probably fails. Use fewer epochs per session instead, or pick the last finished epoch.

Copy the final checkpoint back to your computer
  It already is on Drive:  My Drive/@@DRIVE@@/@@SPEAKER@@/checkpoints/checkpoint-epoch-N  (download it from drive.google.com
  or let Google Drive for desktop sync it). From a Colab cell you can also copy it to another Drive folder:
    !cp -r "/content/drive/MyDrive/@@DRIVE@@/@@SPEAKER@@/checkpoints/checkpoint-epoch-2" "/content/drive/MyDrive/@@DRIVE@@/@@SPEAKER@@/final"

----- Route B: the `colab` command-line tool -----
Syntax read from `colab readme` and `colab help <command>`. VoiceStudio did not run any of these. Its README says
the tool supports Linux and macOS (Windows is not listed): on Windows run it from WSL if a command misbehaves.
`colab exec` runs PYTHON in the session's kernel and reads code from stdin or -f FILE; its default timeout is 30 s, so
long jobs are started in the background and followed through a log file on Drive.

    colab new -s @@SESSION@@ --gpu A100              # or L4; needs a Colab plan that offers it
    colab drivemount -s @@SESSION@@                  # mounts at /content/drive (a Google sign-in prompt may appear)

  Put the zip on Drive first (drive.google.com, or Google Drive for desktop): colab upload of a multi-GB file is slow.
  Small files are fine:  colab upload -s @@SESSION@@ colab_setup.sh /content/colab_setup.sh

    colab exec -s @@SESSION@@ --timeout 900 <<'PY'
    import subprocess
    print(subprocess.run("mkdir -p /content/@@DIR@@ && unzip -q -o '/content/drive/MyDrive/@@DRIVE@@/@@SPEAKER@@/@@ZIP@@' -d /content/@@DIR@@ && ls /content/@@DIR@@",
                         shell=True, capture_output=True, text=True).stdout)
    PY

    colab exec -s @@SESSION@@ --timeout 60 <<'PY'
    import subprocess
    subprocess.Popen("nohup bash /content/@@DIR@@/colab_setup.sh > /content/drive/MyDrive/@@DRIVE@@/@@SPEAKER@@/train.log 2>&1 &",
                     shell=True, start_new_session=True)
    PY

  Follow it (the log is on Drive, so it survives too):
    colab exec -s @@SESSION@@ <<'PY'
    print(open('/content/drive/MyDrive/@@DRIVE@@/@@SPEAKER@@/train.log').read()[-3000:])
    PY
    colab ls -s @@SESSION@@ content/drive/MyDrive/@@DRIVE@@/@@SPEAKER@@/checkpoints
    colab status -s @@SESSION@@

  Fetch a checkpoint file to this computer (not needed, the checkpoints are on Drive; each file is large):
    colab download -s @@SESSION@@ content/drive/MyDrive/@@DRIVE@@/@@SPEAKER@@/checkpoints/checkpoint-epoch-2/model.safetensors ./model.safetensors

  Resume: same limits as above. After a lost session, `colab new` again, `colab drivemount`, and start the setup script
  with a new OUTPUT_DIR (add --env OUTPUT_DIR=... to colab exec, or put it in the command), or RESTART=1 to overwrite.
    colab stop -s @@SESSION@@                        # release the GPU when you are done
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
  colab_setup.sh         Colab script: unzip, clone Qwen3-TTS, install, preprocess, fine-tune, checkpoints to Drive
  colab_commands.txt     the same steps for a notebook and for the `colab` command-line tool, plus resume notes

Facts taken from Qwen's code
  - Rows hold audio, text and ref_audio; no ref_text is used. A row may also carry an optional "language" (not set here).
  - Paths are opened relative to the working directory, so run prepare_data.py and sft_12hz.py from inside this folder.
  - prepare_data.py adds "audio_codes" to each row (output: a new JSONL). sft_12hz.py reads that file.
  - There is no validation input and no resume option in the official scripts.
  - The text is the spoken transcript only: Fish tags such as [calm] are removed; punctuation is kept.
  - The same ref_audio on every row is what Qwen recommends for a stable voice.

Quick start on Colab (details in colab_commands.txt)
  1. Copy the ZIP to Drive: My Drive/@@DRIVE@@/@@SPEAKER@@/@@ZIP@@
  2. Colab, GPU runtime (A100, L4 or H100). Cells:
       from google.colab import drive; drive.mount('/content/drive')
       !mkdir -p /content/@@DIR@@ && unzip -q -o "/content/drive/MyDrive/@@DRIVE@@/@@SPEAKER@@/@@ZIP@@" -d /content/@@DIR@@
       !bash /content/@@DIR@@/colab_setup.sh
  3. Checkpoints appear in My Drive/@@DRIVE@@/@@SPEAKER@@/checkpoints/checkpoint-epoch-N after every epoch.

Without the helper, from inside this folder (after cloning and installing Qwen3-TTS):
  python PATH_TO_QWEN/finetuning/prepare_data.py --device cuda:0 --tokenizer_model_path @@TOKENIZER@@ --input_jsonl train.jsonl --output_jsonl train_with_codes.jsonl
  python PATH_TO_QWEN/finetuning/sft_12hz.py --init_model_path LOCAL_COPY_OF_@@BASE_NAME@@ --output_model_path OUTPUT_DIR --train_jsonl train_with_codes.jsonl --batch_size @@BATCH@@ --lr @@LR@@ --num_epochs @@EPOCHS@@ --speaker_name @@SPEAKER@@
  (init_model_path must be a local folder: the script copies it into every checkpoint.)

Training settings (training_settings.json): @@SETTINGS@@

This set
@@SUMMARY@@

Source and terms: the audio is synthetic speech generated by Fish Audio. Check Fish's terms before using it to train a model.
"""


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
        "TOKENIZER": settings.tokenizer_model, "SESSION": SESSION, "BATCH": str(settings.batch_size),
        "LR": f"{settings.learning_rate:g}", "EPOCHS": str(settings.epochs),
        "SETTINGS": f"{settings.epochs} epochs, learning rate {settings.learning_rate:g}, batch size {settings.batch_size}, base model {settings.base_model}",
    }
    return {
        "README.txt": _fill(README, values | {"SUMMARY": _summary(report)}),
        "colab_setup.sh": _fill(SETUP, values),
        "colab_commands.txt": _fill(COMMANDS, values),
    }
