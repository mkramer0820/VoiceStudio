"""Fish Audio engine: offline contract tests (HTTP stubbed, no key, no network)."""
from __future__ import annotations

import io
import json

import httpx
import numpy as np
import pytest
import soundfile as sf

from engines.fish_api import FishAudioBackend, client
from engines.fish_api.config import load_config


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    for name in list(__import__("os").environ):
        if name.startswith("FISH_"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("core.config.OUTPUTS_DIR", str(tmp_path / "outputs"))
    monkeypatch.setattr(client.time, "sleep", lambda *_: None)


def _wav_bytes(sr=44100, seconds=0.5):
    buf = io.BytesIO()
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    sf.write(buf, (0.2 * np.sin(2 * np.pi * 220 * t)).astype("float32"), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _transport(handler):
    return httpx.MockTransport(handler)


def test_unavailable_without_key_and_no_network(monkeypatch):
    def boom(*a, **k):  # any network use fails the test
        raise AssertionError("network used")

    monkeypatch.setattr(httpx.Client, "send", boom)
    monkeypatch.setattr("services.settings_store.get_secret", lambda name: None)
    ok, msg = FishAudioBackend.is_available()
    assert ok is False and "FISH_API_KEY" in msg


def test_available_with_key_does_not_probe(monkeypatch):
    monkeypatch.setenv("FISH_API_KEY", "k")
    monkeypatch.setattr(httpx.Client, "send", lambda *a, **k: (_ for _ in ()).throw(AssertionError("net")))
    assert FishAudioBackend.is_available()[0] is True


def test_defaults_favour_lossless_quality():
    cfg = load_config()
    assert (cfg.format, cfg.sample_rate, cfg.latency, cfg.model) == ("wav", 44100, "normal", "s2.1-pro")
    assert cfg.archive_masters is True
    body = client.build_body("hi", cfg)
    assert body["format"] == "wav" and body["sample_rate"] == 44100
    assert "mp3_bitrate" not in body


def test_mp3_192_is_configurable(monkeypatch):
    monkeypatch.setenv("FISH_TTS_FORMAT", "mp3")
    monkeypatch.setenv("FISH_TTS_MP3_BITRATE", "192")
    body = client.build_body("hi", load_config())
    assert body["format"] == "mp3" and body["mp3_bitrate"] == 192


def test_request_shape_and_model_header(monkeypatch):
    seen = {}

    def handler(request: httpx.Request):
        seen["headers"] = request.headers
        seen["body"] = json.loads(request.content)
        seen["url"] = str(request.url)
        return httpx.Response(200, content=_wav_bytes())

    monkeypatch.setenv("FISH_TTS_MODEL", "drama-3")
    cfg = load_config()
    out = client.synthesize(client.build_body("hello", cfg, reference_id="abc", speed=1.5),
                            api_key="secret", cfg=cfg, transport=_transport(handler))
    assert out[:4] == b"RIFF"
    assert seen["url"] == "https://api.fish.audio/v1/tts"
    assert seen["headers"]["authorization"] == "Bearer secret"
    assert seen["headers"]["model"] == "drama-3"
    assert seen["body"]["reference_id"] == "abc"
    assert seen["body"]["prosody"] == {"speed": 1.5}


@pytest.mark.parametrize("status,needle", [(401, "API key"), (402, "credits"), (404, "could not find")])
def test_error_mapping(status, needle):
    cfg = load_config()
    t = _transport(lambda r: httpx.Response(status, json={"message": "nope", "status": status}))
    with pytest.raises(client.FishAPIError) as err:
        client.synthesize({"text": "x"}, api_key="k", cfg=cfg, transport=t)
    assert err.value.status == status and needle in err.value.message


def test_429_retries_then_succeeds():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(429, json={"message": "slow"}) if calls["n"] < 3 else httpx.Response(200, content=b"ok")

    out = client.synthesize({"text": "x"}, api_key="k", cfg=load_config(), transport=_transport(handler), sleep=lambda *_: None)
    assert out == b"ok" and calls["n"] == 3


def test_non_retryable_4xx_is_not_retried():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(402, json={"message": "no credits"})

    with pytest.raises(client.FishAPIError):
        client.synthesize({"text": "x"}, api_key="k", cfg=load_config(), transport=_transport(handler))
    assert calls["n"] == 1


def test_rejects_insecure_base_url(monkeypatch):
    monkeypatch.setenv("FISH_API_BASE_URL", "http://example.com")
    with pytest.raises(client.FishAPIError):
        client.synthesize({"text": "x"}, api_key="k", cfg=load_config(), transport=_transport(lambda r: httpx.Response(200)))


def test_generate_returns_tensor_and_archives_untouched_master(monkeypatch, tmp_path):
    raw = _wav_bytes()
    monkeypatch.setenv("FISH_API_KEY", "k")
    monkeypatch.setattr(client, "synthesize", lambda body, **kw: raw)
    backend = FishAudioBackend()
    wav = backend.generate("Hello there", voice="v1")
    assert wav.shape[0] == 1 and wav.shape[1] == len(sf.read(io.BytesIO(raw))[0])
    assert backend.sample_rate == 44100
    saved = list((tmp_path / "outputs" / "fish_masters").rglob("*.wav"))
    assert len(saved) == 1 and saved[0].read_bytes() == raw


def test_generate_without_key_raises(monkeypatch):
    monkeypatch.setattr("services.settings_store.get_secret", lambda name: None)
    with pytest.raises(RuntimeError, match="API key"):
        FishAudioBackend().generate("hi")


def test_registered_in_registry():
    from services.tts_backend import _REGISTRY

    assert _REGISTRY["fish-audio"] is FishAudioBackend
