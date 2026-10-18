"""Fish Audio cloud TTS (``fish-audio``) — opt-in, key-gated, quality-first.

Selected like any engine (Model Catalogue, or ``OMNIVOICE_TTS_BACKEND=fish-audio``)
once a key exists (``FISH_API_KEY`` or the encrypted ``fish_audio_key`` secret).
With no key the engine reports unavailable and makes no network call; probing
availability never touches the network either.

Text (and, in a later phase, reference audio) is sent to Fish Audio. See
docs/engines/fish-audio.md. Settings live in :mod:`.config`.

Editing workflow: by default the response is requested as lossless WAV at
44.1 kHz and the *untouched* bytes are archived under
``<outputs>/fish_masters/`` before any VoiceStudio processing (watermark,
loudness, resampling) so external edits can start from Fish's own file.
"""
from __future__ import annotations

import hashlib
import io
import logging
import os
import re
import time
from typing import Optional

import torch

from services.tts_backend import TTSBackend, TTSInputError

from . import client
from .config import load_config, resolve_api_key

logger = logging.getLogger("omnivoice.fish_api")

_LANG_RE = re.compile(r"^[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})*$")
_EXT = {"wav": "wav", "pcm": "pcm", "mp3": "mp3", "opus": "opus"}


class FishAudioBackend(TTSBackend):
    """Fish Audio cloud TTS (s2.1-pro, s2-pro, drama-3). Preset/saved voices by id."""

    id = "fish-audio"
    display_name = "Fish Audio (cloud API, s2.1-pro / s2-pro / drama-3)"
    # No local model: never claim a GPU, never needs VRAM.
    gpu_compat = ("cpu",)
    min_vram_gb = 0.0
    supports_cloning = False  # phase 1a: saved voices by reference_id only
    supports_voice_design = False
    instruct_vocabulary = "freeform"

    def __init__(self):
        self._cfg = load_config()

    # ── availability: no network, no key logging ──
    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        if not resolve_api_key():
            return False, (
                "No Fish Audio API key. Set FISH_API_KEY or store one in Settings. "
                "Text you synthesize is sent to Fish Audio."
            )
        return True, "ready (cloud: text is sent to Fish Audio)"

    @property
    def sample_rate(self) -> int:
        cfg = load_config()
        return 44100 if cfg.format in ("mp3", "opus") else cfg.sample_rate

    @property
    def supported_languages(self) -> list[str]:
        return ["multi"]

    def ensure_ready(self) -> None:  # nothing to load
        return None

    # ── generation ──
    def generate(self, text: str, *, speed: float = 1.0, language: Optional[str] = None,
                 ref_audio: Optional[str] = None, **extras) -> torch.Tensor:
        if not text or not text.strip():
            raise TTSInputError("Fish Audio needs non-empty text.")
        api_key = resolve_api_key()
        if not api_key:
            raise RuntimeError(self.is_available()[1])
        if ref_audio:
            logger.info("Fish Audio: ref_audio ignored (use a saved voice id via 'voice').")

        cfg = load_config()
        self._cfg = cfg
        voice = extras.get("voice") or extras.get("voice_id") or extras.get("fish_reference_id")
        lang = language if isinstance(language, str) and _LANG_RE.match(language) else None
        body = client.build_body(text, cfg, reference_id=voice, speed=speed, language=lang)

        try:
            raw = client.synthesize(body, api_key=api_key, cfg=cfg)
        except client.FishAPIError as exc:
            if exc.status == 400:
                raise TTSInputError(exc.message) from exc
            raise RuntimeError(exc.message) from exc

        if cfg.archive_masters:
            self._archive_master(raw, cfg.format, text)
        return self._decode(raw, cfg)

    # ── helpers ──
    @staticmethod
    def _archive_master(raw: bytes, fmt: str, text: str) -> Optional[str]:
        """Keep Fish's response byte-for-byte. Best effort: never fails a render."""
        try:
            from core.config import OUTPUTS_DIR

            day = time.strftime("%Y%m%d")
            folder = os.path.join(OUTPUTS_DIR, "fish_masters", day)
            os.makedirs(folder, exist_ok=True)
            digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
            path = os.path.join(folder, f"{time.strftime('%H%M%S')}_{digest}.{_EXT.get(fmt, 'bin')}")
            with open(path, "wb") as fh:
                fh.write(raw)
            return path
        except Exception as exc:  # noqa: BLE001
            logger.warning("Fish Audio: could not archive master (%s)", type(exc).__name__)
            return None

    @staticmethod
    def _decode(raw: bytes, cfg) -> torch.Tensor:
        """Bytes -> (1, n) float tensor at the response's own rate (no lossy steps)."""
        if cfg.format == "pcm":
            import numpy as np

            samples = np.frombuffer(raw[: len(raw) // 2 * 2], dtype="<i2").astype("float32") / 32768.0
            return torch.from_numpy(samples.copy()).unsqueeze(0)
        try:
            # soundfile reads WAV bit-exactly with no torchaudio/ffmpeg dependency.
            import soundfile as sf

            data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
            wav = torch.from_numpy(data.T.copy())
        except Exception:  # noqa: BLE001 — e.g. a container libsndfile lacks
            from services.audio_io import load_audio

            wav, sr = load_audio(io.BytesIO(raw))
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        elif wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        target = 44100 if cfg.format in ("mp3", "opus") else cfg.sample_rate
        if sr != target:
            # Only when Fish answers at a rate other than the one asked for.
            import torchaudio

            logger.warning("Fish Audio returned %s Hz, expected %s Hz; resampling.", sr, target)
            wav = torchaudio.functional.resample(wav, sr, target)
        return wav
