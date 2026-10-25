"""Tests for the fine-tuned Qwen3-TTS narrator engine (faster-qwen3-tts sidecar).

No GPU, no model, no sidecar spawn: the registry wiring, the availability
reasons, the one-job-at-a-time lock, and the pure logic the sidecar relies on
(chunking to fit the cache, length prediction, end-of-speech push, padding).
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from engines.qwen3_narrator import narrator_logic as logic


@pytest.fixture(autouse=True)
def _no_python_override():
    from engines import qwen3_narrator

    qwen3_narrator.set_python_override(None)
    yield
    qwen3_narrator.set_python_override(None)


# -- registry ---------------------------------------------------------------


def test_registered_and_isolated():
    from services.tts_backend import _ENGINE_DOCS, _INSTALL_HINTS, _REGISTRY

    cls = _REGISTRY["qwen3-narrator"]
    assert cls.__name__ == "Qwen3NarratorBackend"
    assert getattr(cls, "_is_subprocess_isolated", False)
    assert cls.gpu_compat == ("cuda",)
    assert "qwen3-narrator" in _INSTALL_HINTS
    assert _ENGINE_DOCS["qwen3-narrator"] == "docs/engines/qwen3-narrator.md"


# -- availability says why --------------------------------------------------


def test_unavailable_reasons(monkeypatch, tmp_path):
    from engines.qwen3_narrator import PYTHON_ENV, CKPT_ENV, availability

    monkeypatch.delenv(PYTHON_ENV, raising=False)
    monkeypatch.delenv(CKPT_ENV, raising=False)
    ok, reason = availability()
    assert not ok and PYTHON_ENV in reason

    monkeypatch.setenv(PYTHON_ENV, str(tmp_path / "missing" / "python"))
    ok, reason = availability()
    assert not ok and "does not exist" in reason

    python = tmp_path / "python"
    python.write_text("")
    monkeypatch.setenv(PYTHON_ENV, str(python))
    ok, reason = availability()
    assert not ok and CKPT_ENV in reason

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    monkeypatch.setenv(CKPT_ENV, str(ckpt))
    ok, reason = availability()
    assert not ok and "config.json" in reason

    (ckpt / "config.json").write_text("{}")
    ok, reason = availability()
    assert not ok and "model.safetensors" in reason

    (ckpt / "model.safetensors").write_text("")
    ok, _ = availability()
    assert ok


# -- one job at a time ------------------------------------------------------


def test_generate_serialises_submissions(monkeypatch):
    from engines.qwen3_narrator import Qwen3NarratorBackend
    from services.subprocess_backend import SubprocessBackend

    active, peak = [0], [0]
    guard = threading.Lock()

    def fake_generate(self, text, **kw):
        with guard:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.05)
        with guard:
            active[0] -= 1
        return text

    monkeypatch.setattr(SubprocessBackend, "generate", fake_generate)
    backend = Qwen3NarratorBackend.__new__(Qwen3NarratorBackend)
    backend._job_lock = threading.Lock()
    threads = [threading.Thread(target=backend.generate, args=("x",)) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert peak[0] == 1


def test_generate_forwards_only_known_options(monkeypatch):
    from engines.qwen3_narrator import Qwen3NarratorBackend
    from services.subprocess_backend import SubprocessBackend

    seen = {}
    monkeypatch.setattr(SubprocessBackend, "generate", lambda self, text, **kw: seen.update(kw))
    backend = Qwen3NarratorBackend.__new__(Qwen3NarratorBackend)
    backend._job_lock = threading.Lock()
    voice = {"kind": "clone", "ref_audio": "x.wav", "ref_text": "hi"}
    backend.generate("hi", seed=3, pad_s=0.5, voice=voice, model_path="m", ref_audio="x.wav", num_step=9)
    assert seen == {"seed": 3, "pad_s": 0.5, "voice": voice, "model_path": "m"}


# -- chunking to fit the cache ----------------------------------------------


def test_chunks_never_exceed_the_limit_and_keep_all_text():
    text = ("It didn't take Snuggle long to slay the Great Beast. " * 12 +
            "A very long clause, without any full stop, that goes on and on " * 8)
    for limit in (60, 120, 300):
        chunks = logic.split_chunks(text, limit)
        assert all(0 < len(c) <= limit for c in chunks)
        assert " ".join(chunks).split() == text.split()


def test_unbreakable_word_is_hard_cut():
    chunks = logic.split_chunks("x" * 250, 100)
    assert [len(c) for c in chunks] == [100, 100, 50]


def test_short_sentences_share_a_chunk_and_empty_text_gives_none():
    assert logic.split_chunks("One. Two. Three.", 300) == ["One. Two. Three."]
    assert logic.split_chunks("   ", 300) == []


def test_default_chunk_size_fits_the_cache():
    for max_seq in (700, 1280, 2048):
        chars = logic.default_chunk_chars(max_seq)
        _, cap = logic.token_limits(chars)
        assert cap <= logic.max_audio_frames(max_seq)
        assert chars <= logic.MAX_CHUNK_CHARS
    # a smaller cache means smaller chunks
    assert logic.default_chunk_chars(700) < logic.default_chunk_chars(1280)


# -- end-of-speech ----------------------------------------------------------


def test_eos_ramp_is_silent_until_start_then_grows_to_a_ceiling():
    torch = pytest.importorskip("torch")
    ramp = logic.EosRamp(eos_id=2, start=5)
    logits = torch.zeros(1, 4)
    for _ in range(6):  # calls 1..6 = 5 frames: nothing yet
        assert ramp.apply(logits)[0, 2] == 0
    assert ramp.apply(logits)[0, 2] == pytest.approx(1.5)
    for _ in range(100):
        out = ramp.apply(logits)
    assert out[0, 2] == pytest.approx(40.0)
    assert logits[0, 2] == 0  # input is never modified


def test_classify_and_limits():
    start, cap = logic.token_limits(100)
    assert start < cap
    assert logic.classify(start, start, cap) == "stopped by itself"
    assert logic.classify(start + 1, start, cap) == "pushed to stop"
    assert logic.classify(cap, start, cap) == "hit the limit"


# -- padding ----------------------------------------------------------------


def test_pad_tail_trims_trailing_silence_and_adds_exact_padding():
    sr = 1000
    wave = np.concatenate([np.zeros(50), 0.5 * np.ones(100), np.zeros(900)]).astype(np.float32)
    out = logic.pad_tail(wave, sr, pad_s=0.3)
    assert len(out) == 150 + 300
    assert np.all(out[150:] == 0) and out[149] == 0.5


def test_pad_tail_of_pure_silence_is_just_padding():
    out = logic.pad_tail(np.zeros(500, dtype=np.float32), 1000, pad_s=0.2)
    assert len(out) == 200 and not out.any()
