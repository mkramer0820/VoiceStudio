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
_STATE: dict = {"ramp": None, "baseline_mib": 0.0, "speaker": None, "key": None, "hooked": False, "type": None}

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


def _read_config(ckpt: str) -> dict:
    try:
        return json.load(open(os.path.join(ckpt, "config.json"), encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _speaker_from_config(ckpt: str) -> str:
    names = list((_read_config(ckpt).get("talker_config") or {}).get("spk_id") or {})
    return names[0] if names else "ss_narrator"


def _unload_model() -> None:
    global _MODEL
    _MODEL = None
    _STATE["key"] = None
    _flush()


def _load_model(stdout, ckpt: str | None = None, max_seq: int | None = None):
    """The resident model for ``ckpt``. Asking for another checkpoint (or cache
    length) unloads the current one first: one model on the GPU at a time."""
    global _MODEL
    ckpt = ckpt or os.environ.get("OMNIVOICE_QWEN3_NARRATOR_CKPT", "")
    max_seq = max_seq or _env_int("OMNIVOICE_QWEN3_NARRATOR_MAX_SEQ_LEN", 1280)
    if _MODEL is not None and _STATE["key"] == (ckpt, max_seq):
        return _MODEL
    if not ckpt or not os.path.isfile(os.path.join(ckpt, "config.json")):
        raise FileNotFoundError(
            "OMNIVOICE_QWEN3_NARRATOR_CKPT (or the request's model_path) must point at a checkpoint "
            f"folder (with config.json and model.safetensors); got {ckpt!r}"
        )
    if _MODEL is not None:
        _log(f"unloading {_STATE['key'][0]} to load {ckpt}")
        _unload_model()
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
        _STATE["type"] = _read_config(ckpt).get("tts_model_type")
        _STATE["speaker"] = os.environ.get("OMNIVOICE_QWEN3_NARRATOR_SPEAKER") or _speaker_from_config(ckpt)
        _STATE["key"] = (ckpt, max_seq)

        if not _STATE["hooked"]:
            # fast_generate() has no logits hook but calls the module-level
            # sample_logits() once per frame; wrap that one name, once.
            original = G.sample_logits

            def _pushed(logits, **kw):
                ramp = _STATE["ramp"]
                return original(ramp.apply(logits) if ramp is not None else logits, **kw)

            G.sample_logits = _pushed
            _STATE["hooked"] = True
    _flush()
    _STATE["baseline_mib"] = _vram_mb()[0]
    _send(stdout, {"op": "progress", "stage": "loading_model", "percent": 100})
    return _MODEL


_XVEC_CACHE: dict = {}


def _xvector(ref_audio: str, base_dir: str, cache_path: str | None):
    """The speaker embedding (x-vector) of ``ref_audio``, made once by the stock
    Base model on the CPU (a custom_voice checkpoint has no speaker encoder),
    then cached in memory and, if ``cache_path`` is given, on disk as .npy."""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    key = (os.path.abspath(ref_audio), os.path.getmtime(ref_audio), os.path.abspath(base_dir))
    if key in _XVEC_CACHE:
        return _XVEC_CACHE[key]
    if cache_path and os.path.isfile(cache_path) and os.path.getmtime(cache_path) >= key[1]:
        vec = torch.from_numpy(np.load(cache_path))
    else:
        from qwen_tts import Qwen3TTSModel  # noqa: PLC0415

        _log(f"extracting x-vector of {os.path.basename(ref_audio)} with {base_dir} on the CPU")
        base = Qwen3TTSModel.from_pretrained(base_dir, device_map="cpu", dtype=torch.bfloat16)
        item = base.create_voice_clone_prompt(ref_audio=ref_audio, ref_text="", x_vector_only_mode=True)[0]
        vec = item.ref_spk_embedding.detach().float().cpu().reshape(-1)
        del base, item
        gc.collect()
        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            np.save(cache_path, vec.numpy())
    _XVEC_CACHE[key] = vec
    return vec


@contextlib.contextmanager
def _as_custom_voice(model, enabled: bool):
    """Let a Base model take the custom-voice call for one generation (the type is a flag the library checks)."""
    inner = model.model.model
    previous = inner.tts_model_type
    if enabled:
        inner.tts_model_type = "custom_voice"
    try:
        yield
    finally:
        inner.tts_model_type = previous


@contextlib.contextmanager
def _speaker_row(model, speaker: str, vector):
    """Write ``vector`` into the codec-embedding row of ``speaker`` for one
    generation and put the trained row back afterwards, so the checkpoint's own
    voice is untouched. The custom_voice path reads the speaker only from that row."""
    import torch  # noqa: PLC0415

    inner = model.model.model
    index = inner.config.talker_config.spk_id[speaker.lower()]
    weight = inner.talker.get_input_embeddings().weight
    with torch.no_grad():
        original = weight[index].clone()
        weight[index] = vector.to(device=weight.device, dtype=weight.dtype)
    try:
        yield
    finally:
        with torch.no_grad():
            weight[index] = original


#: Sampling options a request may set; anything absent keeps the library default.
_SAMPLING = ("temperature", "top_k", "top_p", "repetition_penalty")


def _ref_frames(path: str) -> int:
    import soundfile as sf  # noqa: PLC0415

    info = sf.info(path)
    return int(info.frames / info.samplerate * logic.FRAME_HZ) + 1


def _generate_chunk(model, text: str, voice: dict, language: str, seed, pad_s: float, slow: float, median: float,
                    params: dict):
    import torch  # noqa: PLC0415

    kind = voice.get("kind", "custom_voice")
    loaded = _STATE.get("type")
    borrowed = voice.get("speaker_row")
    if kind == "custom_voice" and loaded != "custom_voice" and not (borrowed is not None and loaded == "base"):
        raise ValueError(f"This model is a {loaded!r} model; a custom-voice speaker needs a custom_voice checkpoint.")
    if kind == "clone_on_custom" and loaded != "custom_voice":
        raise ValueError(f"'Clone on custom voice' needs a custom_voice checkpoint; this is a {loaded!r} model.")
    if kind == "clone" and loaded != "base":
        raise ValueError(
            f"This model is a {loaded!r} model and cannot clone from a reference clip (a fine-tuned custom_voice "
            "checkpoint has no speaker encoder). Use a Base model for clones."
        )
    reserve = logic.PROMPT_RESERVE + (_ref_frames(voice["ref_audio"]) if kind == "clone" else 0)
    start, cap = logic.token_limits(len(text), slow, median)
    cap = min(cap, logic.max_audio_frames(_STATE["max_seq"], reserve))
    if cap < start:
        raise ValueError(
            f"The cache ({_STATE['max_seq']} frames) is too small for this voice's reference plus the text; "
            "use a shorter reference clip or raise max_seq_len."
        )
    ramp = _STATE["ramp"] = logic.EosRamp(_STATE["eos_id"], start)
    if seed is not None:
        torch.manual_seed(int(seed))
    options = {k: params[k] for k in _SAMPLING if params.get(k) is not None}
    try:
        if kind == "clone":
            wavs, sr = model.generate_voice_clone(
                text=text, language=language, ref_audio=voice["ref_audio"], ref_text=voice.get("ref_text", ""),
                xvec_only=bool(voice.get("xvec_only", False)), instruct=voice.get("instruct") or None,
                max_new_tokens=cap, **options)
        else:
            speaker = voice.get("speaker") or _STATE["speaker"]
            if borrowed is not None:  # a stock voice name this model has no entry for: point it at the stock row
                cfg = model.model.model.config.talker_config
                cfg.spk_id[speaker.lower()] = int(borrowed)
                cfg.spk_is_dialect[speaker.lower()] = False
            row = _speaker_row(model, speaker, voice["_xvec"]) if kind == "clone_on_custom" else contextlib.nullcontext()
            with row, _as_custom_voice(model, borrowed is not None):
                wavs, sr = model.generate_custom_voice(
                    text=text, speaker=speaker, language=language,
                    instruct=voice.get("instruct") or None, max_new_tokens=cap, **options)
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
    model = _load_model(stdout, msg.get("model_path"), msg.get("max_seq_len"))
    slow = float(msg.get("slow_s_per_char") or logic.SLOW_S_PER_CHAR)
    median = float(msg.get("median_s_per_char") or logic.MEDIAN_S_PER_CHAR)
    chunk_chars = int(msg.get("chunk_chars") or logic.default_chunk_chars(_STATE["max_seq"], slow))
    pad_s = max(0.0, float(msg.get("pad_s", 0.3)))
    voice = dict(msg.get("voice") or {})
    if msg.get("speaker") and "speaker" not in voice:
        voice["speaker"] = msg["speaker"]
    voice.setdefault("kind", "custom_voice")
    if voice["kind"] == "clone_on_custom":
        for need in ("ref_audio", "base_model_path"):
            if not voice.get(need):
                raise ValueError(f"voice.{need} is required for 'clone_on_custom'")
        voice["_xvec"] = _xvector(voice["ref_audio"], voice["base_model_path"], voice.get("xvec_path"))
    language = msg.get("language") or "Auto"
    chunks = logic.split_chunks(text, chunk_chars)
    if not chunks:
        raise ValueError("synthesize: text is empty after cleaning")
    waves, report, rate = [], [], SAMPLE_RATE
    with _heartbeating(stdout, "generating"):
        for i, chunk in enumerate(chunks):
            seed = None if msg.get("seed") is None else int(msg["seed"]) + i
            wave, rate, info = _generate_chunk(model, chunk, voice, language, seed, pad_s, slow, median, msg)
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
