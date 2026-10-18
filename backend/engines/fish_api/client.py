"""Thin HTTPS client for Fish Audio's ``POST /v1/tts``.

Plain ``httpx`` + JSON on purpose: the official SDK pins its own model default
and does not expose the ``language`` field. Only called once the user has
selected the engine and set a key; nothing here runs at import or probe time.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional
from urllib.parse import urlsplit

import httpx

from .config import FishConfig

logger = logging.getLogger("omnivoice.fish_api")

_RETRY_STATUSES = {429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 4
_TIMEOUT = httpx.Timeout(connect=10.0, read=180.0, write=60.0, pool=10.0)


class FishAPIError(RuntimeError):
    """A non-success answer from Fish. ``status`` is the HTTP code (0 = transport)."""

    def __init__(self, status: int, message: str):
        self.status = status
        self.message = message
        super().__init__(message)


_FRIENDLY = {
    400: "Fish Audio rejected the request ({m}). Check the model, voice id and format.",
    401: "Fish Audio rejected the API key. Re-enter it in Settings or FISH_API_KEY.",
    402: "Fish Audio account is out of credits ({m}). Top up billing or try model s2.1-pro-free.",
    403: "This Fish Audio key lacks permission for that resource ({m}).",
    404: "Fish Audio could not find that model or voice ({m}).",
    429: "Fish Audio rate/concurrency limit hit ({m}). Lower FISH_TTS_MAX_CONCURRENCY.",
}


def _message(resp: httpx.Response) -> str:
    try:
        data = resp.json()
        if isinstance(data, dict) and data.get("message"):
            return str(data["message"])[:300]
    except ValueError:
        pass
    return (resp.text or "")[:300]


def _validate_base_url(url: str) -> None:
    parts = urlsplit(url)
    loopback = parts.hostname in {"localhost", "127.0.0.1", "::1"}
    if parts.scheme != "https" and not (parts.scheme == "http" and loopback):
        raise FishAPIError(0, "FISH_API_BASE_URL must be https (http only for loopback).")
    if parts.username or parts.password:
        raise FishAPIError(0, "FISH_API_BASE_URL must not embed credentials.")


# Process-wide cap so a batch or dub job cannot trip Fish's account concurrency limit.
_gate_lock = threading.Lock()
_gate: Optional[threading.BoundedSemaphore] = None
_gate_size = 0


def _acquire_gate(size: int) -> threading.BoundedSemaphore:
    global _gate, _gate_size
    with _gate_lock:
        if _gate is None or _gate_size != size:
            _gate, _gate_size = threading.BoundedSemaphore(size), size
        return _gate


def build_body(text: str, cfg: FishConfig, *, reference_id: Optional[str] = None,
               speed: float = 1.0, language: Optional[str] = None) -> dict:
    body: dict = {
        "text": text,
        "format": cfg.format,
        "latency": cfg.latency,
        "normalize": cfg.normalize,
    }
    if cfg.format in ("wav", "pcm", "mp3"):
        body["sample_rate"] = cfg.sample_rate
    if cfg.format == "mp3":
        body["mp3_bitrate"] = cfg.mp3_bitrate
    rid = reference_id or cfg.reference_id
    if rid:
        body["reference_id"] = rid
    if cfg.temperature is not None:
        body["temperature"] = cfg.temperature
    if cfg.top_p is not None:
        body["top_p"] = cfg.top_p
    if cfg.chunk_length is not None:
        body["chunk_length"] = cfg.chunk_length
    if speed and abs(speed - 1.0) > 1e-6:
        body["prosody"] = {"speed": max(0.5, min(2.0, float(speed)))}
    if language and len(language) <= 35:
        body["language"] = language
    return body


def synthesize(body: dict, *, api_key: str, cfg: FishConfig,
               transport: Optional[httpx.BaseTransport] = None,
               sleep=time.sleep) -> bytes:
    """POST the request; return the raw audio bytes exactly as Fish sent them."""
    _validate_base_url(cfg.base_url)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "model": cfg.model,
    }
    gate = _acquire_gate(cfg.max_concurrency)
    delay = 1.0
    with gate, httpx.Client(timeout=_TIMEOUT, transport=transport) as http:
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            resp = None
            try:
                resp = http.post(f"{cfg.base_url}/v1/tts", json=body, headers=headers)
            except httpx.HTTPError as exc:
                if attempt == _MAX_ATTEMPTS:
                    raise FishAPIError(0, f"Could not reach Fish Audio: {type(exc).__name__}") from exc
            else:
                if resp.status_code == 200:
                    return resp.content
                if resp.status_code not in _RETRY_STATUSES or attempt == _MAX_ATTEMPTS:
                    m = _message(resp)
                    tpl = _FRIENDLY.get(resp.status_code)
                    raise FishAPIError(
                        resp.status_code,
                        tpl.format(m=m) if tpl else f"Fish Audio error HTTP {resp.status_code}: {m}",
                    )
            wait = delay
            if resp is not None:
                try:
                    wait = max(wait, float(resp.headers.get("retry-after", 0)))
                except ValueError:
                    pass
            logger.warning("Fish Audio attempt %d failed; retrying in %.1fs", attempt, wait)
            sleep(min(wait, 30.0))
            delay *= 2
    raise FishAPIError(0, "Fish Audio request failed")  # pragma: no cover
