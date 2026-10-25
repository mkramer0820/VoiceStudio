"""Fine-tuned Qwen3-TTS narrator sidecar (faster-qwen3-tts, CUDA graphs).

Launched as ``<engine venv python> main.py`` by Qwen3NarratorBackend. The venv
holds torch, transformers 5 and ``faster-qwen3-tts``; this file imports nothing
from the app, so it stays importable with the standard library alone (heavy
imports load inside the synthesize handler).

One request at a time. A long text is split into chunks that fit the static KV
cache and each chunk is generated in turn, never batched. Each chunk's length is
predicted from its text, the end-of-speech token is pushed past that length, and
every chunk ends with silence padding. The CUDA graph and its cache are
allocated once at load; nothing grows per request, and the allocator cache is
flushed after each one.

Wire protocol: length-prefixed JSON over stdio, identical to the other
sidecars (engines/voxcpm2_subprocess/main.py).
"""
from __future__ import annotations

import base64
import contextlib
import gc
import json
import os
import struct
import sys
import threading
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import narrator_logic as logic  # noqa: E402

# Mirrors services/subprocess_backend.py::MAX_FRAME_BYTES.
MAX_FRAME_BYTES = 64 * 1024 * 1024
SAMPLE_RATE = 24_000
_HEARTBEAT_S = 5.0
#: How far allocated memory may drift above the post-load figure after a
#: request before the sidecar says so (MiB).
_DRIFT_WARN_MIB = 64

_MODEL = None
_STATE: dict = {"ramp": None, "baseline_mib": 0.0, "speaker": None}

_send_lock = threading.Lock()


def _send(stream, obj: dict) -> None:
    body = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    with _send_lock:
        stream.write(struct.pack("!I", len(body)))
        stream.write(body)
        stream.flush()


def _recv(stream):
    header = stream.read(4)
    if len(header) < 4:
        return None  # EOF
    (n,) = struct.unpack("!I", header)
    if n > MAX_FRAME_BYTES:
        raise IOError(f"frame too large: {n}")
    body = bytearray()
    while len(body) < n:
        chunk = stream.read(n - len(body))
        if not chunk:
            raise IOError("short read")
        body.extend(chunk)
    return json.loads(bytes(body).decode("utf-8"))


def _log(text: str) -> None:
    print(f"[qwen3-narrator] {text}", file=sys.stderr, flush=True)


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def _vram_mb() -> tuple[float, float]:
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            return (torch.cuda.memory_allocated() / 2**20, torch.cuda.memory_reserved() / 2**20)
    except Exception:  # noqa: BLE001 — a probe, never fatal
        pass
    return (0.0, 0.0)


def _flush() -> None:
    gc.collect()
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


@contextlib.contextmanager
def _heartbeating(stdout, stage: str):
    """Progress frames while a long call runs, so the parent's watchdog stays armed."""
    stop = threading.Event()

    def _beat() -> None:
        pct = 1
        while not stop.wait(_HEARTBEAT_S):
            pct = min(pct + 1, 99)
            _send(stdout, {"op": "progress", "stage": stage, "percent": pct})

    thread = threading.Thread(target=_beat, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=_HEARTBEAT_S + 1)


def _speaker_from_config(ckpt: str) -> str:
    try:
        cfg = json.load(open(os.path.join(ckpt, "config.json"), encoding="utf-8"))
        names = list((cfg.get("talker_config") or {}).get("spk_id") or {})
        if names:
            return names[0]
    except Exception:  # noqa: BLE001
        pass
    return "ss_narrator"


def _load_model(stdout):
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    ckpt = os.environ.get("OMNIVOICE_QWEN3_NARRATOR_CKPT", "")
    if not ckpt or not os.path.isfile(os.path.join(ckpt, "config.json")):
        raise FileNotFoundError(
            "OMNIVOICE_QWEN3_NARRATOR_CKPT must point at a checkpoint folder "
            f"(with config.json and model.safetensors); got {ckpt!r}"
        )
    max_seq = _env_int("OMNIVOICE_QWEN3_NARRATOR_MAX_SEQ_LEN", 1280)
    _send(stdout, {"op": "progress", "stage": "loading_model", "percent": 0})
    with _heartbeating(stdout, "loading_model"):
        import torch  # noqa: PLC0415
        import faster_qwen3_tts.generate as G  # noqa: PLC0415
        from faster_qwen3_tts import FasterQwen3TTS  # noqa: PLC0415

        _flush()
        _MODEL = FasterQwen3TTS.from_pretrained(
            ckpt, device="cuda", dtype=torch.bfloat16, attn_implementation="sdpa", max_seq_len=max_seq
        )
        _STATE["eos_id"] = _MODEL.model.model.config.talker_config.codec_eos_token_id
        _STATE["max_seq"] = max_seq
        _STATE["speaker"] = os.environ.get("OMNIVOICE_QWEN3_NARRATOR_SPEAKER") or _speaker_from_config(ckpt)

        # fast_generate() has no logits hook but calls the module-level
        # sample_logits() once per frame; wrap that one name.
        original = G.sample_logits

        def _pushed(logits, **kw):
            ramp = _STATE["ramp"]
            return original(ramp.apply(logits) if ramp is not None else logits, **kw)

        G.sample_logits = _pushed
    _flush()
    _STATE["baseline_mib"] = _vram_mb()[0]
    _send(stdout, {"op": "progress", "stage": "loading_model", "percent": 100})
    return _MODEL


def _generate_chunk(model, text: str, speaker: str, language: str, seed, pad_s: float, slow: float, median: float):
    import torch  # noqa: PLC0415

    start, cap = logic.token_limits(len(text), slow, median)
    cap = min(cap, logic.max_audio_frames(_STATE["max_seq"]))
    ramp = _STATE["ramp"] = logic.EosRamp(_STATE["eos_id"], start)
    if seed is not None:
        torch.manual_seed(int(seed))
    try:
        wavs, sr = model.generate_custom_voice(text=text, speaker=speaker, language=language, max_new_tokens=cap)
        wave = logic.pad_tail(wavs[0], sr, pad_s)
    finally:
        _STATE["ramp"] = None
    how = logic.classify(ramp.steps, start, cap)
    return wave, int(sr), {"chars": len(text), "steps": ramp.steps, "start": start, "cap": cap, "how": how}


def _handle_synthesize(msg: dict, stdout) -> None:
    import numpy as np  # noqa: PLC0415

    text = msg.get("text")
    if not text or not isinstance(text, str):
        raise ValueError("synthesize: missing or non-string 'text'")
    model = _load_model(stdout)
    slow = float(msg.get("slow_s_per_char") or logic.SLOW_S_PER_CHAR)
    median = float(msg.get("median_s_per_char") or logic.MEDIAN_S_PER_CHAR)
    chunk_chars = int(msg.get("chunk_chars") or logic.default_chunk_chars(_STATE["max_seq"], slow))
    pad_s = max(0.0, float(msg.get("pad_s", 0.3)))
    speaker = msg.get("speaker") or _STATE["speaker"]
    language = msg.get("language") or "Auto"
    chunks = logic.split_chunks(text, chunk_chars)
    if not chunks:
        raise ValueError("synthesize: text is empty after cleaning")
    waves, report, rate = [], [], SAMPLE_RATE
    with _heartbeating(stdout, "generating"):
        for i, chunk in enumerate(chunks):
            seed = None if msg.get("seed") is None else int(msg["seed"]) + i
            wave, rate, info = _generate_chunk(model, chunk, speaker, language, seed, pad_s, slow, median)
            waves.append(wave)
            report.append(info)
            if info["how"] == "hit the limit":
                _log(f"chunk {i + 1}/{len(chunks)} hit the length limit ({info['cap']} frames): the end-of-speech token was not produced")
            _flush()  # one job at a time; hand cached blocks back after every chunk
    audio = np.concatenate(waves)
    if rate != SAMPLE_RATE:
        audio = np.interp(np.linspace(0, len(audio) - 1, int(len(audio) * SAMPLE_RATE / rate)), np.arange(len(audio)), audio).astype(np.float32)
    allocated, reserved = _vram_mb()
    if allocated - _STATE["baseline_mib"] > _DRIFT_WARN_MIB:
        _log(f"allocated VRAM drifted {allocated - _STATE['baseline_mib']:.0f} MiB above the post-load figure")
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    _send(stdout, {
        "op": "audio",
        "audio_pcm_b64": base64.b64encode(pcm).decode("ascii"),
        "audio_format": "s16le",
        "sample_rate": SAMPLE_RATE,
        "n_samples": int(audio.shape[-1]),
        "chunks": report,
        "complete": all(r["how"] != "hit the limit" for r in report),
        "vram_allocated_mb": allocated,
        "vram_reserved_mb": reserved,
    })


def main() -> int:
    stdin = sys.stdin.buffer
    # Frames go down a PRIVATE fd and fd 1 is pointed at stderr (#1428): the
    # libraries this loads print to fd 1, which would corrupt the framing.
    _frame_fd = os.dup(1)
    os.dup2(2, 1)
    stdout = os.fdopen(_frame_fd, "wb")

    _send(stdout, {"op": "ready", "engine": "qwen3-narrator", "sample_rate": SAMPLE_RATE})

    while True:
        try:
            msg = _recv(stdin)
        except Exception as exc:  # noqa: BLE001
            _send(stdout, {"op": "error", "stage": "recv", "message": f"{type(exc).__name__}: {exc}",
                           "traceback": traceback.format_exc()})
            return 1
        if msg is None:
            return 0
        op = msg.get("op") if isinstance(msg, dict) else None
        try:
            if op == "ping":
                allocated, reserved = _vram_mb()
                _send(stdout, {"op": "pong", "vram_mb": allocated, "vram_reserved_mb": reserved})
            elif op == "synthesize":
                _handle_synthesize(msg, stdout)
            elif op == "shutdown":
                return 0
            else:
                _send(stdout, {"op": "error", "stage": "dispatch", "message": f"unknown op: {op!r}"})
        except Exception as exc:  # noqa: BLE001
            _send(stdout, {"op": "error", "stage": op or "unknown", "message": f"{type(exc).__name__}: {exc}",
                           "traceback": traceback.format_exc()})
            _flush()


if __name__ == "__main__":
    sys.exit(main())
