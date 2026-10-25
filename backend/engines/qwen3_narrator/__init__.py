"""Fine-tuned Qwen3-TTS narrator on faster-qwen3-tts (CUDA graphs), local GPU.

Runs the owner's fine-tuned custom-voice checkpoint in a sidecar inside its own
venv: faster-qwen3-tts needs ``transformers==5.15.1`` and the ``qwen-tts-hf``
fork, which must never share an interpreter with the app's. Opt-in; nothing is
configured by default.

Settings (environment variables, nothing is hard-coded):

  * ``OMNIVOICE_QWEN3_NARRATOR_PYTHON``  python of the engine's venv (required)
  * ``OMNIVOICE_QWEN3_NARRATOR_CKPT``    checkpoint folder (required)
  * ``OMNIVOICE_QWEN3_NARRATOR_MAX_SEQ_LEN``  static KV cache length, default 1280
  * ``OMNIVOICE_QWEN3_NARRATOR_SPEAKER`` speaker name, default the one in config.json

One request at a time: submissions queue on a lock here, and the sidecar runs
the chunks of a long text one after another, never batched.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from services.subprocess_backend import SubprocessBackend

if TYPE_CHECKING:
    import torch  # noqa: F401

logger = logging.getLogger("omnivoice.qwen3_narrator")

PYTHON_ENV = "OMNIVOICE_QWEN3_NARRATOR_PYTHON"
CKPT_ENV = "OMNIVOICE_QWEN3_NARRATOR_CKPT"
SIDECAR_SCRIPT: Path = Path(__file__).parent / "main.py"
#: Request options passed through to the sidecar. ``model_path`` swaps the
#: resident checkpoint, ``voice`` is {"kind": "custom_voice", "speaker", "instruct"}
#: or {"kind": "clone", "ref_audio", "ref_text", "xvec_only", "instruct"}.
_FORWARDED = (
    "speaker", "language", "seed", "chunk_chars", "pad_s", "model_path", "max_seq_len", "voice",
    "temperature", "top_k", "top_p", "repetition_penalty", "slow_s_per_char", "median_s_per_char",
)


_python_override: str | None = None


def set_python_override(path: str | None) -> None:
    """Use this python for the sidecar instead of the environment variable (the studio's setting)."""
    global _python_override
    _python_override = (path or "").strip() or None


def configured_python() -> Path | None:
    value = (_python_override or os.environ.get(PYTHON_ENV, "")).strip()
    return Path(value) if value else None


def configured_checkpoint() -> Path | None:
    value = os.environ.get(CKPT_ENV, "").strip()
    return Path(value) if value else None


def availability(ignore_checkpoint: bool = False) -> tuple[bool, str]:
    """(usable, reason). Looks at files only; it never imports torch or starts the sidecar.
    ``ignore_checkpoint`` is for callers that name the model per request."""
    python = configured_python()
    if python is None:
        return False, f"Set {PYTHON_ENV} to the python of the engine's venv (see docs/engines/qwen3-narrator.md)."
    if not python.is_file():
        return False, f"{PYTHON_ENV} points at {python}, which does not exist."
    ckpt = configured_checkpoint()
    if ckpt is None and not ignore_checkpoint:
        return False, f"Set {CKPT_ENV} to the fine-tuned checkpoint folder."
    for name in ("config.json", "model.safetensors") if ckpt is not None else ():
        if not (ckpt / name).is_file():
            return False, f"{CKPT_ENV} points at {ckpt}, which has no {name}."
    if not SIDECAR_SCRIPT.is_file():
        return False, f"Sidecar script missing at {SIDECAR_SCRIPT}; reinstall VoiceStudio."
    return True, "ok (NVIDIA GPU, CUDA graphs)"


class Qwen3NarratorBackend(SubprocessBackend):
    """Fine-tuned Qwen3-TTS narrator voice via faster-qwen3-tts."""

    id = "qwen3-narrator"
    display_name = "Qwen3-TTS fine-tuned narrator (local GPU, faster-qwen3-tts)"
    supports_voice_design = False
    supports_cloning = False  # a fixed fine-tuned speaker, no reference clip
    _DEFAULT_SAMPLE_RATE = 24000
    gpu_compat = ("cuda",)

    def __init__(self) -> None:
        super().__init__()
        # The GPU holds one model and one CUDA graph: never two requests at once.
        self._job_lock = threading.Lock()

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        return availability()

    @classmethod
    def venv_python(cls) -> Path:
        python = configured_python()
        if python is None:
            raise RuntimeError(f"{PYTHON_ENV} is not set; see docs/engines/qwen3-narrator.md.")
        return python

    @classmethod
    def sidecar_script(cls) -> Path:
        return SIDECAR_SCRIPT

    @property
    def recv_timeout_s(self) -> float:
        # Cold load plus the first CUDA-graph capture; later calls are far quicker.
        try:
            return max(30.0, float(os.environ.get("OMNIVOICE_QWEN3_NARRATOR_RECV_TIMEOUT_S", "600")))
        except (TypeError, ValueError):
            return 600.0

    @property
    def sample_rate(self) -> int:
        return self._DEFAULT_SAMPLE_RATE

    @property
    def supported_languages(self) -> list[str]:
        return ["multi"]

    def generate(self, text: str, **kw) -> "torch.Tensor":
        """Synthesize ``text``; long text is chunked inside the sidecar.

        kwargs forwarded: see ``_FORWARDED``. ``pad_s`` is the silence after each
        chunk (default 0.3 s), ``chunk_chars`` overrides the cache-derived chunk
        size. After the call ``last_reply_meta`` holds the sidecar's per-chunk
        report (``chunks``, ``complete``, VRAM).
        """
        forwarded = {k: kw[k] for k in _FORWARDED if kw.get(k) is not None}
        with self._job_lock:
            return super().generate(text, **forwarded)


__all__ = ["Qwen3NarratorBackend", "availability"]
