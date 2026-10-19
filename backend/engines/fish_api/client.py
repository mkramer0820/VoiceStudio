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
    if cfg.format == "opus" and cfg.opus_bitrate is not None:
        body["opus_bitrate"] = cfg.opus_bitrate
    for name in ("repetition_penalty", "max_new_tokens", "min_chunk_length",
                 "condition_on_previous_chunks", "early_stop_threshold"):
        value = getattr(cfg, name)
        if value is not None:
            body[name] = value
    prosody: dict = {}
    if speed and abs(speed - 1.0) > 1e-6:
        prosody["speed"] = max(0.5, min(2.0, float(speed)))
    if abs(cfg.volume) > 1e-6:
        prosody["volume"] = cfg.volume
    if not cfg.normalize_loudness:  # Fish's own default is on; only send a change
        prosody["normalize_loudness"] = False
    if prosody:
        body["prosody"] = prosody
    language = language or cfg.language
    if language and len(language) <= 35:
        body["language"] = language
    return body


def _request(method: str, path: str, *, api_key: str, cfg: FishConfig,
             json_body: Optional[dict] = None, params: Optional[dict] = None,
             send_model: bool = False,
             transport: Optional[httpx.BaseTransport] = None,
             sleep=time.sleep, meta: Optional[dict] = None) -> httpx.Response:
    """One authenticated call with the shared retry/error policy.

    ``meta`` (optional) is filled with what a request log needs: ``attempts``,
    ``elapsed_ms`` and the final ``status`` / ``content_type``. It never holds
    the key or the body.
    """
    _validate_base_url(cfg.base_url)
    headers = {"Authorization": f"Bearer {api_key}"}
    if json_body is not None:
        headers["Content-Type"] = "application/json"
    if send_model:
        headers["model"] = cfg.model
    gate = _acquire_gate(cfg.max_concurrency)
    delay = 1.0
    started = time.monotonic()
    if meta is not None:
        meta.update(attempts=0, elapsed_ms=0, status=0, content_type=None)
    with gate, httpx.Client(timeout=_TIMEOUT, transport=transport) as http:
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            resp = None
            if meta is not None:
                meta["attempts"] = attempt
                meta["elapsed_ms"] = int((time.monotonic() - started) * 1000)
            try:
                resp = http.request(method, f"{cfg.base_url}{path}", json=json_body,
                                    params=params, headers=headers)
            except httpx.HTTPError as exc:
                if attempt == _MAX_ATTEMPTS:
                    raise FishAPIError(0, f"Could not reach Fish Audio: {type(exc).__name__}") from exc
            else:
                if meta is not None:
                    meta.update(status=resp.status_code, content_type=resp.headers.get("content-type"),
                                elapsed_ms=int((time.monotonic() - started) * 1000))
                if resp.status_code == 200:
                    return resp
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


def synthesize(body: dict, *, api_key: str, cfg: FishConfig,
               transport: Optional[httpx.BaseTransport] = None,
               sleep=time.sleep, meta: Optional[dict] = None) -> bytes:
    """POST the request; return the raw audio bytes exactly as Fish sent them."""
    return _request("POST", "/v1/tts", api_key=api_key, cfg=cfg, json_body=body,
                    send_model=True, transport=transport, sleep=sleep, meta=meta).content


def _voice(item: dict) -> dict:
    samples = item.get("samples") or []
    first = samples[0] if samples and isinstance(samples[0], dict) else {}
    author = item.get("author") if isinstance(item.get("author"), dict) else {}
    return {
        "id": item.get("_id"),
        "title": item.get("title") or "",
        "description": item.get("description") or "",
        "languages": item.get("languages") or [],
        "tags": item.get("tags") or [],
        "visibility": item.get("visibility"),
        "state": item.get("state"),
        "author": author.get("nickname"),
        "sample_url": first.get("audio"),  # a Fish-hosted URL; never fetched by the backend
        "created_at": item.get("created_at"),
    }


def list_voices(*, api_key: str, cfg: FishConfig, mine: bool = True,
                title: Optional[str] = None, language: Optional[str] = None,
                page_number: int = 1, page_size: int = 100,
                transport: Optional[httpx.BaseTransport] = None,
                sleep=time.sleep) -> dict:
    """One page of voices: your own (``mine``) or the public Voice Library."""
    params: dict = {"page_size": max(1, min(100, page_size)), "page_number": max(1, page_number)}
    if mine:
        params["self"] = "true"
    if title:
        params["title"] = title
    if language:
        params["language"] = language
    data = _request("GET", "/model", api_key=api_key, cfg=cfg, params=params,
                    transport=transport, sleep=sleep).json()
    items = [_voice(i) for i in data.get("items", []) if isinstance(i, dict) and i.get("_id")]
    return {"total": data.get("total", len(items)), "items": items,
            "has_more": bool(data.get("has_more"))}


def list_all_voices(*, api_key: str, cfg: FishConfig, max_pages: int = 20, **kw) -> list[dict]:
    """Every page of ``list_voices`` (bounded so a huge library cannot loop forever)."""
    out: list[dict] = []
    for page in range(1, max_pages + 1):
        res = list_voices(api_key=api_key, cfg=cfg, page_number=page, **kw)
        out.extend(res["items"])
        if not res["items"] or not (res["has_more"] or len(out) < res["total"]):
            break
    return out


def get_balance(*, api_key: str, cfg: FishConfig,
                transport: Optional[httpx.BaseTransport] = None,
                sleep=time.sleep) -> dict:
    """API credit for the key's own account (``GET /wallet/self/api-credit``)."""
    data = _request("GET", "/wallet/self/api-credit", api_key=api_key, cfg=cfg,
                    transport=transport, sleep=sleep).json()
    return {"credit": str(data.get("credit", "")),
            "cumulative_top_up": str(data.get("cumulative_top_up", "")),
            "has_free_credit": data.get("has_free_credit")}
