"""/api/fish/story routes. Fish is stubbed; no network."""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.dependencies import require_admin
from api.routers import fish_story
from core.browser_guard import reject_cross_site_get
from engines.fish_api import story


@pytest.fixture
def api(monkeypatch, tmp_path):
    app = FastAPI()
    app.include_router(fish_story.router)
    app.dependency_overrides[require_admin] = lambda: None
    app.dependency_overrides[reject_cross_site_get] = lambda: None
    monkeypatch.delenv("FISH_API_KEY", raising=False)
    monkeypatch.setattr("services.settings_store.get_text", lambda k: None)
    monkeypatch.setattr("services.settings_store.get_secret", lambda n: None)
    monkeypatch.setattr(story, "_store_dir", lambda: str(tmp_path))
    return TestClient(app)


DRAFT = {"title": "s", "chunks": [{"id": "a", "text": "hello", "voice": "v1"}], "voices": {"v1": "sun_2"}}


def test_draft_crud_is_local(api):
    made = api.post("/api/fish/story/drafts", json=DRAFT).json()
    did = made["id"]
    assert api.get(f"/api/fish/story/drafts/{did}").json()["chunks"][0]["voice"] == "v1"
    DRAFT2 = {**DRAFT, "title": "renamed"}
    assert api.put(f"/api/fish/story/drafts/{did}", json=DRAFT2).json()["title"] == "renamed"
    assert [d["title"] for d in api.get("/api/fish/story/drafts").json()["drafts"]] == ["renamed"]
    assert api.delete(f"/api/fish/story/drafts/{did}").status_code == 200
    assert api.get(f"/api/fish/story/drafts/{did}").status_code == 404


def test_bad_drafts_and_ids_are_422(api):
    bad = {"chunks": [{"id": "a", "text": "x", "voice": "../etc"}]}
    assert api.post("/api/fish/story/plan", json=bad).status_code == 422
    assert api.post("/api/fish/story/plan", json={"settings": {"speed": 9}}).status_code == 422
    assert api.get("/api/fish/story/drafts/..%2Fx").status_code in (404, 422)


def test_plan_is_pure_counting(api):
    res = api.post("/api/fish/story/plan", json=DRAFT).json()
    assert res["chunks"] == 1 and res["chars"] == 5 and res["voices"] == {"v1": {"chunks": 1, "chars": 5}}


def test_render_requires_explicit_confirmation(api, monkeypatch):
    monkeypatch.setattr(story, "start_render", lambda *a, **k: pytest.fail("must not start"))
    assert api.post("/api/fish/story/render", json={"confirm": False, "draft": DRAFT}).status_code == 422
    assert api.post("/api/fish/story/render", json={"draft": DRAFT}).status_code == 422


def test_render_without_key_is_a_clear_502(api):
    res = api.post("/api/fish/story/render", json={"confirm": True, "draft": DRAFT})
    assert res.status_code == 502 and "API key" in res.json()["detail"]


def test_render_unknown_job_and_balance_without_key(api):
    assert api.get("/api/fish/story/render/nope").status_code == 404
    assert api.post("/api/fish/story/render/nope/cancel").status_code == 404
    assert api.get("/api/fish/balance").status_code == 502


def test_defaults_never_expose_the_base_url_or_key(api):
    d = api.get("/api/fish/story/defaults").json()
    assert "base_url" not in d and d["format"] == "wav" and d["sample_rate"] == 44100
