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
        settings=story.StorySettings(**{"request_mode": "single", **settings}),
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
    assert client.build_body("hi", load_config())["prosody"] == {"normalize_loudness": True}  # explicit, never left to Fish's default


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


def test_multi_mode_maps_speaker_ids_and_lets_fish_stitch_the_story(monkeypatch):
    seen = []
    response = _wav(seconds=0.3)

    def fake(body, **_):
        seen.append(body)
        return response

    monkeypatch.setattr(client, "synthesize", fake)
    draft = story.Draft(
        title="Dialogue",
        chunks=[
            story.Chunk(id="a", text="Hello.", speaker="Alice"),
            story.Chunk(id="b", text="Hi.", speaker="Bob"),
            story.Chunk(id="c", text="Ready?", speaker="Alice"),
        ],
        voices={"va": "Alice voice", "vb": "Bob voice"},
        cast={"Alice": "va", "Bob": "vb"},
        settings=story.StorySettings(request_mode="multi"),
    )

    preview = story_jobs.preview(draft)
    assert preview["total"] == 3 and preview["total_requests"] == 1
    assert preview["multi_requests"] == 1 and preview["single_requests"] == 0
    assert preview["requests"][0]["body"]["reference_id"] == ["va", "vb"]

    job = _wait(story_jobs.start_render(draft)["id"])

    assert len(seen) == 1
    assert seen[0]["reference_id"] == ["va", "vb"]
    assert seen[0]["text"] == (
        "<|speaker:0|>Hello.\n<|speaker:1|>Hi.\n<|speaker:0|>Ready?"
    )
    assert job["done"] == 3 and {row["request_mode"] for row in job["chunks"]} == {"multi"}
    assert len({row["file"] for row in job["chunks"]}) == 1
    import os

    assert open(os.path.join(job["out_dir"], "story.wav"), "rb").read() == response


def test_multi_mode_uses_large_single_speaker_schema_for_one_mapped_speaker(monkeypatch):
    seen = []
    monkeypatch.setattr(client, "synthesize", lambda body, **_: seen.append(body) or _wav())
    draft = story.Draft(
        chunks=[
            story.Chunk(id="a", text="First.", speaker="Alice"),
            story.Chunk(id="b", text="Second.", speaker="Alice"),
        ],
        cast={"Alice": "va"},
        settings=story.StorySettings(request_mode="multi"),
    )

    job = _wait(story_jobs.start_render(draft)["id"])

    assert job["done"] == 2 and len(seen) == 1
    assert seen[0]["reference_id"] == "va"
    assert seen[0]["text"] == "First.\nSecond."
    assert "<|speaker:" not in seen[0]["text"]


def test_multi_mode_splits_only_at_the_api_limit_then_joins_large_responses(monkeypatch):
    seen = []
    monkeypatch.setattr(client, "synthesize", lambda body, **_: seen.append(body) or _wav())
    draft = story.Draft(
        chunks=[
            story.Chunk(id="a", text="A" * 9000, speaker="Alice"),
            story.Chunk(id="b", text="B" * 9000, speaker="Bob"),
            story.Chunk(id="c", text="C" * 9000, speaker="Alice"),
        ],
        cast={"Alice": "va", "Bob": "vb"},
        settings=story.StorySettings(request_mode="multi", pause_ms=0, max_concurrency=1),
    )

    job = _wait(story_jobs.start_render(draft)["id"])

    assert len(seen) == 2
    assert seen[0]["reference_id"] == ["va", "vb"]
    assert seen[1]["reference_id"] == "va"
    assert len({row["file"] for row in job["chunks"]}) == 2
    import os

    data, rate = sf.read(os.path.join(job["out_dir"], "story.wav"))
    assert rate == 44100 and len(data) == pytest.approx(2 * 0.1 * rate, rel=0.02)


def test_multi_mode_requires_every_speaker_to_resolve_to_a_voice(monkeypatch):
    monkeypatch.setattr(
        client, "synthesize",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("request was sent")),
    )
    draft = story.Draft(
        chunks=[
            story.Chunk(id="a", text="Hello.", speaker="Alice"),
            story.Chunk(id="b", text="Hi.", speaker="Bob"),
        ],
        cast={"Alice": "va"},
        settings=story.StorySettings(request_mode="multi"),
    )

    with pytest.raises(ValueError, match="Line 2 needs a voice"):
        story_jobs.start_render(draft)


def test_single_mode_keeps_one_api_call_per_line(monkeypatch):
    seen = []
    monkeypatch.setattr(client, "synthesize", lambda body, **_: seen.append(body) or _wav())
    draft = story.Draft(
        chunks=[
            story.Chunk(id="a", text="Hello.", speaker="Alice"),
            story.Chunk(id="b", text="Hi.", speaker="Bob"),
        ],
        cast={"Alice": "va", "Bob": "vb"},
        settings=story.StorySettings(request_mode="single", max_concurrency=1),
    )

    job = _wait(story_jobs.start_render(draft)["id"])

    assert job["done"] == 2 and [body["reference_id"] for body in seen] == ["va", "vb"]
    assert all("<|speaker:" not in body["text"] for body in seen)


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


def test_an_unset_request_mode_means_one_call_per_line(monkeypatch):
    draft = story.Draft(chunks=[story.Chunk(id="a", text="Hello", voice="v"),
                                story.Chunk(id="b", text="There", voice="w")])
    out = story_jobs.preview(draft)
    assert out["single_requests"] == 2 and out["multi_requests"] == 0 and out["ok"]


def test_minimal_request_sends_only_the_basics_and_preview_lists_extras(monkeypatch):
    monkeypatch.setattr("services.settings_store.get_text", lambda k: "1.0" if k == "tts.fish.top_p" else None)
    draft = story.Draft(chunks=[story.Chunk(id="a", text="Hello", voice="vvv")],
                        settings=story.StorySettings(top_p=0.9))
    full = story_jobs.preview(draft)
    assert "top_p" in full["extra_fields"] and "latency" in full["extra_fields"]
    mini = story_jobs.preview(draft.model_copy(update={
        "settings": story.StorySettings(top_p=0.9, minimal_request=True, repetition_penalty=1.1)}))
    body = mini["requests"][0]["body"]
    assert set(body) == {"text", "format", "sample_rate", "reference_id"} and mini["extra_fields"] == []
    assert mini["ok"] and mini["invalid"] == 0  # a minimal body is valid, not "missing required fields"


def test_preview_lists_every_setting_with_its_source_and_whether_it_is_sent(monkeypatch):
    monkeypatch.setattr("services.settings_store.get_text", lambda k: "false" if k == "tts.fish.normalize" else None)
    draft = story.Draft(chunks=[story.Chunk(id="a", text="Hello", voice="vvv")],
                        settings=story.StorySettings(temperature=0.7))
    rows = {r["name"]: r for r in story_jobs.preview(draft)["settings"]}
    assert rows["temperature"] == {"name": "temperature", "value": 0.7, "source": "story", "sent": True, "note": None}
    assert rows["normalize"]["source"] == "saved default" and rows["normalize"]["value"] is False
    assert rows["top_p"]["sent"] is False and rows["top_p"]["source"] == "fish"
    assert rows["mp3_bitrate"]["note"] == "na"
    mini = {r["name"]: r for r in story_jobs.preview(draft.model_copy(update={
        "settings": story.StorySettings(temperature=0.7, minimal_request=True)}))["settings"]}
    assert mini["temperature"]["sent"] is False and mini["temperature"]["note"] == "minimal"


def test_multi_speaker_on_a_model_without_support_is_flagged():
    draft = story.Draft(chunks=[story.Chunk(id="a", text="Hi", voice="v")],
                        settings=story.StorySettings(request_mode="multi", model="s1"))
    assert "multi_speaker_model" in story.plan(draft)["warnings"]


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
        settings=story.StorySettings(request_mode="single"),
    )
    job = _wait(story_jobs.start_render(draft)["id"])
    with open(os.path.join(job["out_dir"], "manifest.json"), encoding="utf-8") as fh:
        assert json.load(fh)["chunks"][0]["speaker"] == "Shadow"
