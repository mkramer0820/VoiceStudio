"""Fish story board: drafts, plan, bounded render. Fish is stubbed; no network."""
from __future__ import annotations

import io
import threading
import time

import numpy as np
import pytest
import soundfile as sf

from engines.fish_api import client, story, story_jobs
from engines.fish_api.config import FishConfig, load_config


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    for name in ("FISH_API_KEY", "FISH_TTS_FORMAT", "FISH_TTS_MAX_CONCURRENCY", "FISH_TTS_REFERENCE_ID",
                 "FISH_TTS_VOLUME", "FISH_TTS_NORMALIZE_LOUDNESS", "FISH_TTS_LANGUAGE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FISH_API_KEY", "test-key-123456")
    monkeypatch.setattr("services.settings_store.get_text", lambda k: None)
    monkeypatch.setattr(story, "_store_dir", lambda: str(tmp_path))
    monkeypatch.setattr("core.config.DATA_DIR", str(tmp_path), raising=False)


def _wav(seconds=0.1, rate=44100) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(rate * seconds), dtype="float32"), rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _draft(n=4, **settings) -> story.Draft:
    return story.Draft(
        title="t",
        chunks=[story.Chunk(id=f"c{i}", text=f"line {i}", voice="vsun" if i < 2 else None) for i in range(n)],
        voices={"vsun": "sun_2"},
        settings=story.StorySettings(**settings),
    )


def _wait(job_id, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        job = story_jobs.get_job(job_id)
        if job["status"] != "running":
            return job
        time.sleep(0.02)
    raise AssertionError("render did not finish")


def test_plan_counts_and_warns_about_unassigned():
    p = story.plan(_draft(4))
    assert p["chunks"] == 4 and p["unassigned"] == 2 and "unassigned_no_default" in p["warnings"]
    assert p["voices"]["vsun"]["chunks"] == 2
    ok = story.plan(_draft(4, default_voice="vnarr"))
    assert ok["warnings"] == [] and ok["voices"]["vnarr"]["chunks"] == 2


def test_effective_config_overlays_non_null_options_only():
    cfg = story.effective_config(story.StorySettings(format="mp3", mp3_bitrate=128, volume=3.0,
                                                    normalize_loudness=False, max_concurrency=1))
    assert (cfg.format, cfg.mp3_bitrate, cfg.volume, cfg.normalize_loudness, cfg.max_concurrency) == \
        ("mp3", 128, 3.0, False, 1)
    assert cfg.sample_rate == load_config().sample_rate  # untouched option keeps its default


@pytest.mark.parametrize("bad", [{"speed": 3.0}, {"format": "flac"}, {"max_concurrency": 99},
                                 {"mp3_bitrate": 320}, {"default_voice": "../x"}])
def test_settings_are_validated(bad):
    with pytest.raises(Exception):
        story.StorySettings(**bad)


def test_prosody_and_language_reach_the_request_body():
    cfg = story.effective_config(story.StorySettings(volume=-3.0, normalize_loudness=False, language="en"))
    body = client.build_body("hi", cfg, speed=1.25)
    assert body["prosody"] == {"speed": 1.25, "volume": -3.0, "normalize_loudness": False}
    assert body["language"] == "en"
    assert "prosody" not in client.build_body("hi", load_config())


def test_draft_roundtrip_and_id_validation():
    saved = story.save_draft("abc123", _draft(2))
    assert saved["id"] == "abc123" and story.load_draft("abc123")["chunks"][0]["text"] == "line 0"
    assert [d["id"] for d in story.list_drafts()] == ["abc123"]
    assert story.delete_draft("abc123") and story.load_draft("abc123") is None
    with pytest.raises(ValueError):
        story.save_draft("../evil", _draft(1))


def test_render_uses_each_chunks_voice_and_writes_masters_and_join(monkeypatch):
    seen = []

    def fake(body, *, api_key, cfg, **_):
        seen.append((body["text"], body.get("reference_id"), body["format"]))
        return _wav()

    monkeypatch.setattr(client, "synthesize", fake)
    job = _wait(story_jobs.start_render(_draft(4, default_voice="vnarr"), draft_id="d1")["id"])
    assert job["status"] == "done" and job["done"] == 4 and job["failed"] == 0
    assert dict((t, v) for t, v, _ in seen) == {"line 0": "vsun", "line 1": "vsun",
                                                "line 2": "vnarr", "line 3": "vnarr"}
    files = [c["file"] for c in job["chunks"]]
    assert files[0].startswith("001_narration_sun_2") and all(f.endswith(".wav") for f in files)
    assert job["joined"] == "story.wav"
    import os
    data, rate = sf.read(os.path.join(job["out_dir"], "story.wav"))
    assert rate == 44100 and len(data) > 4 * 4410  # four clips plus the pauses between them
    assert os.path.exists(os.path.join(job["out_dir"], "manifest.json"))


def test_render_never_exceeds_the_concurrency_cap(monkeypatch):
    live = peak = 0
    lock = threading.Lock()

    def fake(body, **_):
        nonlocal live, peak
        with lock:
            live += 1
            peak = max(peak, live)
        time.sleep(0.05)
        with lock:
            live -= 1
        return _wav()

    monkeypatch.setattr(client, "synthesize", fake)
    job = _wait(story_jobs.start_render(_draft(8, default_voice="v", max_concurrency=2))["id"])
    assert job["done"] == 8 and 1 <= peak <= 2


def test_out_of_credits_stops_the_rest_and_keeps_finished_chunks(monkeypatch):
    calls = []

    def fake(body, **_):
        calls.append(body["text"])
        if body["text"] == "line 1":
            raise client.FishAPIError(402, "out of credits")
        return _wav()

    monkeypatch.setattr(client, "synthesize", fake)
    job = _wait(story_jobs.start_render(_draft(6, default_voice="v", max_concurrency=1))["id"])
    assert job["status"] == "stopped" and "credits" in job["message"]
    states = [c["status"] for c in job["chunks"]]
    assert states[0] == "done" and states[1] == "failed" and "skipped" in states[2:]
    assert len(calls) == 2


def test_one_failed_chunk_does_not_stop_the_others(monkeypatch):
    def fake(body, **_):
        if body["text"] == "line 1":
            raise client.FishAPIError(400, "bad text")
        return _wav()

    monkeypatch.setattr(client, "synthesize", fake)
    job = _wait(story_jobs.start_render(_draft(3, default_voice="v"))["id"])
    assert job["status"] == "partial" and job["done"] == 2 and job["failed"] == 1


def test_render_subset_and_formats_without_a_join(monkeypatch):
    monkeypatch.setattr(client, "synthesize", lambda body, **_: b"ID3fake")
    job = _wait(story_jobs.start_render(_draft(4, default_voice="v", format="mp3"), chunk_ids=["c2", "c3"])["id"])
    assert job["total"] == 2 and job["joined"] is None
    assert all(c["file"].endswith(".mp3") for c in job["chunks"])


def test_render_needs_a_key_and_something_to_render(monkeypatch):
    with pytest.raises(ValueError):
        story_jobs.start_render(story.Draft(chunks=[story.Chunk(id="a", text="  ")]))
    monkeypatch.delenv("FISH_API_KEY")
    monkeypatch.setattr("services.settings_store.get_secret", lambda n: None)
    with pytest.raises(RuntimeError):
        story_jobs.start_render(_draft(1))


def test_get_balance_parses_the_wallet_response():
    import httpx

    def handler(request: httpx.Request):
        assert request.url.path == "/wallet/self/api-credit"
        return httpx.Response(200, json={"credit": "12.50", "cumulative_top_up": "20", "has_free_credit": False})

    out = client.get_balance(api_key="k" * 8, cfg=load_config(), transport=httpx.MockTransport(handler))
    assert out == {"credit": "12.50", "cumulative_top_up": "20", "has_free_credit": False}


def test_fish_config_dataclass_still_constructs_without_new_fields():
    base = load_config()
    assert isinstance(base, FishConfig) and base.volume == 0.0 and base.normalize_loudness is True


def test_speaker_and_cast_survive_a_save_and_reload():
    draft = story.Draft(
        chunks=[story.Chunk(id="a", text="Hello", speaker="Shadow", voice="vsh")],
        voices={"vsh": "sun_2"},
        cast={"Shadow": "vsh"},
    )
    story.save_draft("cast1", draft)
    saved = story.load_draft("cast1")
    assert saved["chunks"][0]["speaker"] == "Shadow" and saved["cast"] == {"Shadow": "vsh"}


def test_manifest_records_the_speaker_for_each_line(monkeypatch):
    import json
    import os

    monkeypatch.setattr(client, "synthesize", lambda body, **_: _wav())
    draft = story.Draft(
        chunks=[story.Chunk(id="a", text="Hello", speaker="Shadow", voice="vsh")],
        voices={"vsh": "sun_2"},
    )
    job = _wait(story_jobs.start_render(draft)["id"])
    with open(os.path.join(job["out_dir"], "manifest.json"), encoding="utf-8") as fh:
        assert json.load(fh)["chunks"][0]["speaker"] == "Shadow"
