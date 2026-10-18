"""/api/fish routes: list voices, set default voice. Fish is stubbed; no network."""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.dependencies import require_admin
from api.routers import fish_audio
from core.browser_guard import reject_cross_site_get


@pytest.fixture
def api(monkeypatch):
    app = FastAPI()
    app.include_router(fish_audio.router)
    app.dependency_overrides[require_admin] = lambda: None
    app.dependency_overrides[reject_cross_site_get] = lambda: None
    for name in ("FISH_API_KEY", "FISH_TTS_REFERENCE_ID"):
        monkeypatch.delenv(name, raising=False)
    return TestClient(app)


def test_list_voices_returns_voices(api, monkeypatch):
    seen = {}

    def fake(mine, title, language):
        seen.update(mine=mine, title=title, language=language)
        return [{"id": "v1", "title": "Narrator"}]

    monkeypatch.setattr(fish_audio, "_voices", fake)
    res = api.get("/api/fish/voices", params={"scope": "library", "title": "calm"})
    assert res.status_code == 200
    assert res.json()["voices"] == [{"id": "v1", "title": "Narrator"}]
    assert seen == {"mine": False, "title": "calm", "language": None}


def test_list_voices_without_key_is_a_clear_502(api, monkeypatch):
    monkeypatch.setattr("services.settings_store.get_secret", lambda name: None)
    res = api.get("/api/fish/voices")
    assert res.status_code == 502 and "API key" in res.json()["detail"]


def test_rejects_unknown_scope(api):
    assert api.get("/api/fish/voices", params={"scope": "everyone"}).status_code == 422


def test_default_voice_roundtrip(api, monkeypatch):
    store = {}
    monkeypatch.setattr("services.settings_store.set_text", lambda k, v: store.__setitem__(k, v))
    monkeypatch.setattr("services.settings_store.clear_text", lambda k: store.pop(k, None))
    assert api.put("/api/fish/default-voice", json={"voice_id": "abc123"}).json() == {"default_voice_id": "abc123"}
    assert store == {"tts.fish.reference_id": "abc123"}
    assert api.put("/api/fish/default-voice", json={"voice_id": None}).json() == {"default_voice_id": None}
    assert store == {}


def test_default_voice_rejects_odd_ids(api):
    assert api.put("/api/fish/default-voice", json={"voice_id": "../etc"}).status_code == 422
