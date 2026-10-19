"""/api/fish/story routes. Fish is stubbed; no network."""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.dependencies import require_admin
from api.routers import fish_story
from core.browser_guard import reject_cross_site_get
from engines.fish_api import story, story_jobs


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
    monkeypatch.setattr("core.config.DATA_DIR", str(tmp_path / "data"), raising=False)
    monkeypatch.delenv("FISH_STORY_OUTPUT_DIR", raising=False)
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
    monkeypatch.setattr(story_jobs, "start_render", lambda *a, **k: pytest.fail("must not start"))
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


def test_jobs_list_log_and_resume_for_unknown_or_unconfirmed(api):
    assert api.get("/api/fish/story/jobs").json() == {"jobs": []}
    assert api.get("/api/fish/story/jobs/abcdefabcdef/log").status_code == 404
    assert api.post("/api/fish/story/jobs/abcdefabcdef/resume", json={"confirm": False}).status_code == 422
    assert api.post("/api/fish/story/jobs/abcdefabcdef/resume", json={"confirm": True}).status_code == 502  # no key


def test_saving_defaults_writes_settings_and_null_clears_them(api, monkeypatch):
    store = {}
    monkeypatch.setattr("services.settings_store.get_text", lambda k: store.get(k))
    monkeypatch.setattr("services.settings_store.set_text", lambda k, v: store.__setitem__(k, v))
    monkeypatch.setattr("services.settings_store.clear_text", lambda k: store.pop(k, None))
    res = api.put("/api/fish/settings", json={"format": "mp3", "mp3_bitrate": 192, "sample_rate": 48000,
                                              "normalize_loudness": False, "default_voice": "vnarr",
                                              "max_concurrency": 3, "speed": 1.2})
    assert res.status_code == 200
    assert store == {"tts.fish.format": "mp3", "tts.fish.mp3_bitrate": "192", "tts.fish.sample_rate": "48000",
                     "tts.fish.normalize_loudness": "false", "tts.fish.reference_id": "vnarr",
                     "tts.fish.max_concurrency": "3"}  # speed is story-only: never saved
    eff = res.json()
    assert eff["format"] == "mp3" and eff["reference_id"] == "vnarr" and eff["normalize_loudness"] is False
    api.put("/api/fish/settings", json={"format": None, "default_voice": None})
    assert "tts.fish.format" not in store and "tts.fish.reference_id" not in store


def test_saving_an_output_folder_validates_it(api, monkeypatch, tmp_path):
    store = {}
    monkeypatch.setattr("services.settings_store.get_text", lambda k: store.get(k))
    monkeypatch.setattr("services.settings_store.set_text", lambda k, v: store.__setitem__(k, v))
    monkeypatch.setattr("services.settings_store.clear_text", lambda k: store.pop(k, None))
    bad = api.put("/api/fish/settings", json={"output_dir": "relative/path"})
    assert bad.status_code == 422 and "absolute" in bad.json()["detail"]
    assert store == {}
    good = api.put("/api/fish/settings", json={"output_dir": str(tmp_path / "Stories")})
    assert good.status_code == 200 and (tmp_path / "Stories").is_dir()
    assert good.json()["output_dir_effective"].endswith("Stories")
    assert api.put("/api/fish/settings", json={"format": "flac"}).status_code == 422


def test_defaults_report_where_renders_go(api):
    d = api.get("/api/fish/story/defaults").json()
    assert d["output_dir"] is None and d["output_dir_effective"] == d["output_dir_default"]
    assert d["output_dir_default"].replace("\\", "/").endswith("fish_stories/renders")


def test_preview_returns_checked_requests_and_needs_no_key_or_network(api, monkeypatch):
    monkeypatch.setattr("engines.fish_api.client.synthesize",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no network")))
    r = api.post("/api/fish/story/preview", json={"draft": DRAFT, "limit": 3})
    assert r.status_code == 200
    out = r.json()
    assert out["ok"] and out["total"] == 1 and out["requests"][0]["body"]["reference_id"] == "v1"
    assert out["headers"]["Authorization"] == "Bearer <hidden>"
    assert api.post("/api/fish/story/preview", json={"draft": {"chunks": [{"id": "!", "text": "x"}]}}).status_code == 422
