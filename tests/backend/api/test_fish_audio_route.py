"""/api/fish routes: list voices, set default voice. Fish is stubbed; no network."""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.dependencies import require_admin
from api.routers import fish_audio
from core.browser_guard import reject_cross_site_get


@pytest.fixture
def api(monkeypatch, tmp_path):
    monkeypatch.setattr("engines.fish_api.story._store_dir", lambda: str(tmp_path))
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


def test_key_roundtrip_never_returns_the_key(api, monkeypatch):
    store = {}
    monkeypatch.setattr("services.settings_store.get_secret", lambda n: store.get(n))
    monkeypatch.setattr("services.settings_store.set_secret", lambda n, v: store.__setitem__(n, v))
    monkeypatch.setattr("services.settings_store.clear_secret", lambda n: store.pop(n, None))
    assert api.get("/api/fish/key").json() == {"has_key": False, "stored": False, "from_env": False}
    res = api.put("/api/fish/key", json={"api_key": "sk-secret-123456"})
    assert res.json() == {"has_key": True, "stored": True, "from_env": False}
    assert "sk-secret" not in res.text
    assert store == {"fish_audio_key": "sk-secret-123456"}
    assert api.delete("/api/fish/key").json()["has_key"] is False
    assert store == {}


def test_key_reports_env_and_rejects_bad_input(api, monkeypatch):
    monkeypatch.setattr("services.settings_store.get_secret", lambda n: None)
    monkeypatch.setenv("FISH_API_KEY", "envkey-123456")
    assert api.get("/api/fish/key").json() == {"has_key": True, "stored": False, "from_env": True}
    assert api.put("/api/fish/key", json={"api_key": "short"}).status_code == 422
    assert api.put("/api/fish/key", json={"api_key": "has space inside"}).status_code == 422


def test_import_collection_returns_voices_and_rejects_foreign_links(api, monkeypatch):
    from engines.fish_api import collection

    monkeypatch.setattr(collection, "fetch_collection",
                        lambda url: {"id": "c" * 32, "title": "shadow sl", "voices": [{"id": "a" * 32, "title": "V"}]})
    ok = api.post("/api/fish/collection", json={"url": "https://fish.audio/collections/" + "c" * 32})
    assert ok.status_code == 200 and ok.json()["voices"][0]["title"] == "V"
    monkeypatch.undo()
    bad = api.post("/api/fish/collection", json={"url": "https://evil.example/collections/" + "c" * 32})
    assert bad.status_code == 422 and "collection" in bad.json()["detail"].lower()


def test_library_remembers_account_voices_and_keeps_your_character_names(api, monkeypatch):
    voices = [{"id": "v1", "title": "Fish title", "description": "warm", "languages": ["en"], "author": "me"}]
    monkeypatch.setattr(fish_audio, "_voices", lambda mine, title, language: voices)
    monkeypatch.setattr("engines.fish_api.config.load_config", lambda: type("C", (), {"reference_id": None})())
    api.get("/api/fish/voices", params={"scope": "library"})
    assert api.get("/api/fish/library").json()["voices"] == []  # a public search is not saved
    api.get("/api/fish/voices", params={"scope": "mine"})
    row = api.get("/api/fish/library").json()["voices"][0]
    assert row["voice_id"] == "v1" and row["fish_name"] == "Fish title" and row["languages"] == ["en"]
    assert row["alias"] == "" and row["fish_url"] == "https://fish.audio/m/v1/"

    named = api.put("/api/fish/library/v1", json={"alias": "Captain Reyes", "notes": "gruff"}).json()
    assert named["alias"] == "Captain Reyes" and named["fish_name"] == "Fish title"
    api.get("/api/fish/voices", params={"scope": "mine"})  # a refresh must not wipe the name
    assert api.get("/api/fish/library").json()["voices"][0]["alias"] == "Captain Reyes"

    assert api.put("/api/fish/library/bad id!", json={}).status_code == 422
    assert api.delete("/api/fish/library/v1").status_code == 200
    assert api.delete("/api/fish/library/v1").status_code == 404
