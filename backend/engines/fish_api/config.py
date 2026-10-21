"""Fish Audio settings: env var > Settings store > default.

Everything is optional except the API key. The key lives in the encrypted
secret store (``fish_audio_key``) or ``FISH_API_KEY``; it is never logged.
Defaults favour *quality for editing*: lossless WAV at 44.1 kHz, the stable
``normal`` latency mode, and an untouched copy of every response archived.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

SECRET_NAME = "fish_audio_key"
DEFAULT_BASE_URL = "https://api.fish.audio"

MODELS = ("s2.1-pro", "s2.1-pro-free", "s2-pro", "drama-3")
FORMATS = ("wav", "pcm", "mp3", "opus")
MP3_BITRATES = (64, 128, 192)
LATENCIES = ("normal", "balanced")

# name -> (settings_store key, env var)
_KEYS = {
    "base_url": ("tts.fish.base_url", "FISH_API_BASE_URL"),
    "model": ("tts.fish.model", "FISH_TTS_MODEL"),
    "format": ("tts.fish.format", "FISH_TTS_FORMAT"),
    "sample_rate": ("tts.fish.sample_rate", "FISH_TTS_SAMPLE_RATE"),
    "mp3_bitrate": ("tts.fish.mp3_bitrate", "FISH_TTS_MP3_BITRATE"),
    "latency": ("tts.fish.latency", "FISH_TTS_LATENCY"),
    "normalize": ("tts.fish.normalize", "FISH_TTS_NORMALIZE"),
    "temperature": ("tts.fish.temperature", "FISH_TTS_TEMPERATURE"),
    "top_p": ("tts.fish.top_p", "FISH_TTS_TOP_P"),
    "chunk_length": ("tts.fish.chunk_length", "FISH_TTS_CHUNK_LENGTH"),
    "reference_id": ("tts.fish.reference_id", "FISH_TTS_REFERENCE_ID"),
    "archive_masters": ("tts.fish.archive_masters", "FISH_TTS_ARCHIVE_MASTERS"),
    "max_concurrency": ("tts.fish.max_concurrency", "FISH_TTS_MAX_CONCURRENCY"),
    "volume": ("tts.fish.volume", "FISH_TTS_VOLUME"),
    "normalize_loudness": ("tts.fish.normalize_loudness", "FISH_TTS_NORMALIZE_LOUDNESS"),
    "language": ("tts.fish.language", "FISH_TTS_LANGUAGE"),
    "output_dir": ("tts.fish.output_dir", "FISH_STORY_OUTPUT_DIR"),
    "opus_bitrate": ("tts.fish.opus_bitrate", "FISH_TTS_OPUS_BITRATE"),
    "repetition_penalty": ("tts.fish.repetition_penalty", "FISH_TTS_REPETITION_PENALTY"),
    "max_new_tokens": ("tts.fish.max_new_tokens", "FISH_TTS_MAX_NEW_TOKENS"),
    "min_chunk_length": ("tts.fish.min_chunk_length", "FISH_TTS_MIN_CHUNK_LENGTH"),
    "condition_on_previous_chunks": ("tts.fish.condition_on_previous_chunks", "FISH_TTS_CONDITION_ON_PREVIOUS"),
    "early_stop_threshold": ("tts.fish.early_stop_threshold", "FISH_TTS_EARLY_STOP_THRESHOLD"),
}


def store_key(name: str) -> str:
    """The Settings-store key for a setting name (used when saving defaults)."""
    return _KEYS[name][0]


def _raw(name: str) -> Optional[str]:
    store_key, env = _KEYS[name]
    value = os.environ.get(env)
    if value is None or value == "":
        try:
            from services import settings_store

            value = settings_store.get_text(store_key)
        except Exception:  # noqa: BLE001 — store unavailable: use the default
            value = None
    return value.strip() if isinstance(value, str) and value.strip() else None


def _bool(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off"}


def _num(value: Optional[str], cast, default):
    if value is None:
        return default
    try:
        return cast(value)
    except ValueError:
        return default


def resolve_api_key() -> Optional[str]:
    key = os.environ.get("FISH_API_KEY")
    if key and key.strip():
        return key.strip()
    try:
        from services import settings_store

        return settings_store.get_secret(SECRET_NAME)
    except Exception:  # noqa: BLE001
        return None


@dataclass(frozen=True)
class FishConfig:
    base_url: str
    model: str
    format: str
    sample_rate: int
    mp3_bitrate: int
    latency: str
    normalize: bool
    temperature: Optional[float]
    top_p: Optional[float]
    chunk_length: Optional[int]
    reference_id: Optional[str]
    archive_masters: bool
    max_concurrency: int
    volume: float = 0.0  # dB, prosody.volume
    normalize_loudness: bool = True  # prosody.normalize_loudness (S2 family default)
    language: Optional[str] = None
    output_dir: Optional[str] = None  # story renders root; None = <data dir>/fish_stories/renders
    # Advanced tuning: only sent when set, so Fish's own tuned defaults stay in charge.
    opus_bitrate: Optional[int] = None  # -1000 = automatic
    repetition_penalty: Optional[float] = None
    max_new_tokens: Optional[int] = None
    min_chunk_length: Optional[int] = None
    condition_on_previous_chunks: Optional[bool] = None
    early_stop_threshold: Optional[float] = None
    # Story option, not a saved default: send only text, format, rate and voice (like the web page).
    minimal_request: bool = False


def load_config() -> FishConfig:
    fmt = (_raw("format") or "wav").lower()
    if fmt not in FORMATS:
        fmt = "wav"
    bitrate = _num(_raw("mp3_bitrate"), int, 192)
    if bitrate not in MP3_BITRATES:
        bitrate = 192
    latency = (_raw("latency") or "normal").lower()
    if latency not in LATENCIES:
        latency = "normal"
    chunk = _num(_raw("chunk_length"), int, None)
    if chunk is not None:
        chunk = max(100, min(300, chunk))
    return FishConfig(
        base_url=(_raw("base_url") or DEFAULT_BASE_URL).rstrip("/"),
        # Unknown names pass through: Fish falls back to s2.1-pro by itself, and
        # new models (drama-3 arrived in Oct 2026) should not need a code change.
        model=_raw("model") or "s2.1-pro",
        format=fmt,
        sample_rate=max(8000, min(96000, _num(_raw("sample_rate"), int, 44100))),
        mp3_bitrate=bitrate,
        latency=latency,
        normalize=_bool(_raw("normalize"), True),
        temperature=_num(_raw("temperature"), float, None),
        top_p=_num(_raw("top_p"), float, None),
        chunk_length=chunk,
        reference_id=_raw("reference_id"),
        archive_masters=_bool(_raw("archive_masters"), True),
        max_concurrency=max(1, _num(_raw("max_concurrency"), int, 2)),
        volume=max(-30.0, min(30.0, _num(_raw("volume"), float, 0.0))),
        normalize_loudness=_bool(_raw("normalize_loudness"), True),
        language=_raw("language"),
        output_dir=_raw("output_dir"),
        opus_bitrate=_num(_raw("opus_bitrate"), int, None),
        repetition_penalty=_num(_raw("repetition_penalty"), float, None),
        max_new_tokens=_num(_raw("max_new_tokens"), int, None),
        min_chunk_length=_num(_raw("min_chunk_length"), int, None),
        condition_on_previous_chunks=(None if _raw("condition_on_previous_chunks") is None
                                      else _bool(_raw("condition_on_previous_chunks"), True)),
        early_stop_threshold=_num(_raw("early_stop_threshold"), float, None),
    )
