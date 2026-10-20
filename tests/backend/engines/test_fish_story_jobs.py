"""Fish story jobs: organized output, durable records, request log, resume. Fish is stubbed."""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import time

import numpy as np
import pytest
import soundfile as sf

from engines.fish_api import client, story, story_jobs


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    for name in ("FISH_API_KEY", "FISH_TTS_FORMAT", "FISH_TTS_MAX_CONCURRENCY", "FISH_TTS_REFERENCE_ID",
                 "FISH_STORY_OUTPUT_DIR", "FISH_TTS_SAMPLE_RATE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FISH_API_KEY", "sk-super-secret-key-123456")
    monkeypatch.setattr("services.settings_store.get_text", lambda k: None)
    monkeypatch.setattr(story, "_store_dir", lambda: str(tmp_path / "drafts"))
    os.makedirs(tmp_path / "drafts", exist_ok=True)
    monkeypatch.setattr("core.config.DATA_DIR", str(tmp_path / "data"), raising=False)
    yield tmp_path
    story_jobs._jobs.clear()  # other test modules list jobs; leave none behind


def _wav(rate=44100, seconds=0.2) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(rate * seconds), dtype="float32"), rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _draft(title="The Long Night", **settings) -> story.Draft:
    return story.Draft(
        title=title,
        chunks=[
            story.Chunk(id="a", text="Hello there.", speaker="Shadow", voice="vsh"),
            story.Chunk(id="b", text="He looked around.", voice=None),
            story.Chunk(id="c", text="Run!", speaker="Snuggle", voice="vsn"),
        ],
        voices={"vsh": "sun_2", "vsn": "sniggle", "vnar": "Narrator"},
        settings=story.StorySettings(**{"request_mode": "single", "default_voice": "vnar", **settings}),
    )


def _wait(job_id, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        job = story_jobs.get_job(job_id)
        if job and job["status"] != "running":
            return job
        time.sleep(0.02)
    raise AssertionError("render did not finish")


def _ok(monkeypatch, rate=44100):
    def fake(body, *, meta=None, **_):
        if meta is not None:
            meta.update(attempts=1, elapsed_ms=12, status=200, content_type="audio/wav")
        return _wav(rate)

    monkeypatch.setattr(client, "synthesize", fake)


def _read_lines(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_run_folder_is_organised_by_story_and_run(monkeypatch):
    _ok(monkeypatch)
    job = _wait(story_jobs.start_render(_draft(), draft_id="abc123def")["id"])
    run = job["out_dir"]
    story_folder = os.path.basename(os.path.dirname(run))
    assert story_folder == "the-long-night"
    assert re.match(rf"^run-001_\d{{8}}-\d{{6}}_{job['id']}$", os.path.basename(run))
    assert os.path.dirname(os.path.dirname(run)) == story_jobs.default_output_root()
    names = sorted(os.listdir(run))
    for expected in ("job.json", "story.json", "settings.json", "requests.jsonl", "run.log",
                     "manifest.json", "story.wav"):
        assert expected in names
    audio = [n for n in names if n.endswith(".wav") and n != "story.wav"]
    assert audio == ["001_Shadow_sun_2.wav", "002_narration_Narrator.wav", "003_Snuggle_sniggle.wav"]


def test_a_renamed_draft_keeps_using_its_own_folder(monkeypatch):
    _ok(monkeypatch)
    first = _wait(story_jobs.start_render(_draft("First name"), draft_id="zzz999")["id"])
    second = _wait(story_jobs.start_render(_draft("Totally different"), draft_id="zzz999")["id"])
    assert os.path.dirname(first["out_dir"]) == os.path.dirname(second["out_dir"])


def test_request_log_records_exact_request_timing_and_hash_but_never_the_key(monkeypatch):
    _ok(monkeypatch)
    job = _wait(story_jobs.start_render(_draft())["id"])
    rows = _read_lines(os.path.join(job["out_dir"], "requests.jsonl"))
    assert sorted(r["index"] for r in rows) == [1, 2, 3]
    first = next(r for r in rows if r["index"] == 1)
    assert first["request"]["text"] == "Hello there." and first["request"]["reference_id"] == "vsh"
    assert first["status"] == 200 and first["attempts"] == 1 and first["elapsed_ms"] == 12
    assert first["audio"]["sample_rate"] == 44100 and first["audio"]["duration_s"] == pytest.approx(0.2, abs=0.01)
    audio_bytes = open(os.path.join(job["out_dir"], first["file"]), "rb").read()
    assert first["sha256"] == hashlib.sha256(audio_bytes).hexdigest()
    everything = "".join(
        open(os.path.join(job["out_dir"], n), encoding="utf-8").read()
        for n in ("requests.jsonl", "run.log", "job.json", "settings.json", "manifest.json", "story.json")
    )
    assert "sk-super-secret" not in everything and "Authorization" not in everything


def test_run_log_and_manifest_tell_the_story(monkeypatch):
    _ok(monkeypatch)
    job = _wait(story_jobs.start_render(_draft())["id"])
    log = story_jobs.read_log(job["id"])
    assert "Started: 3 lines" in log and "#2 ok: 002_narration_Narrator.wav" in log and "Finished: done" in log
    manifest = json.load(open(os.path.join(job["out_dir"], "manifest.json"), encoding="utf-8"))
    assert manifest["status"] == "done" and manifest["audio_seconds"] == pytest.approx(0.6, abs=0.05)
    assert manifest["chunks"][0]["text"] == "Hello there." and manifest["chunks"][0]["speaker"] == "Shadow"
    settings = json.load(open(os.path.join(job["out_dir"], "settings.json"), encoding="utf-8"))
    assert settings["model"] == "s2.1-pro" and settings["format"] == "wav" and settings["default_voice"] == "vnar"
    assert "base_url" not in settings


def test_job_survives_a_restart_and_shows_in_the_list(monkeypatch):
    _ok(monkeypatch)
    job = _wait(story_jobs.start_render(_draft())["id"])
    story_jobs._jobs.clear()  # a new process knows nothing in memory
    again = story_jobs.get_job(job["id"])
    assert again["status"] == "done" and again["done"] == 3
    rows = story_jobs.list_jobs()
    assert [r["id"] for r in rows] == [job["id"]]
    assert rows[0]["title"] == "The Long Night" and "chunks" not in rows[0]


def test_a_render_cut_off_by_closing_the_app_is_reported_as_interrupted(monkeypatch):
    _ok(monkeypatch)
    job = _wait(story_jobs.start_render(_draft())["id"])
    path = os.path.join(job["out_dir"], "job.json")
    data = json.load(open(path, encoding="utf-8"))
    data["status"] = "running"
    json.dump(data, open(path, "w", encoding="utf-8"))
    story_jobs._jobs.clear()
    assert story_jobs.get_job(job["id"])["status"] == "interrupted"
    assert story_jobs.list_jobs()[0]["status"] == "interrupted"


def test_resume_renders_only_the_unfinished_lines_into_the_same_folder(monkeypatch):
    calls, failed_once = [], []

    def flaky(body, *, meta=None, **_):
        calls.append(body["text"])
        if body["text"] == "He looked around." and not failed_once:
            failed_once.append(True)
            raise client.FishAPIError(400, "bad text")
        if meta is not None:
            meta.update(attempts=1, elapsed_ms=5, status=200)
        return _wav()

    monkeypatch.setattr(client, "synthesize", flaky)
    first = _wait(story_jobs.start_render(_draft())["id"])
    assert first["status"] == "partial" and first["failed"] == 1 and first["joined"] == "story.wav"

    calls.clear()
    story_jobs._jobs.clear()
    resumed = story_jobs.resume_render(first["id"])
    final = _wait(resumed["id"])
    assert calls == ["He looked around."]  # the two finished lines are not sent again
    assert final["status"] == "done" and final["done"] == 3 and final["out_dir"] == first["out_dir"]
    lines = _read_lines(os.path.join(final["out_dir"], "requests.jsonl"))
    assert len(lines) == 4  # the original three plus the retry
    assert "Resumed: 1 lines left" in story_jobs.read_log(first["id"])
    with pytest.raises(ValueError):
        story_jobs.resume_render(first["id"])  # nothing left to do


def test_a_sample_rate_that_differs_from_the_request_is_flagged(monkeypatch):
    _ok(monkeypatch, rate=24000)
    job = _wait(story_jobs.start_render(_draft(sample_rate=44100))["id"])
    assert any("24000 Hz" in w and "44100 Hz" in w for w in job["warnings"])
    assert "WARN" in story_jobs.read_log(job["id"])


def test_failures_are_logged_with_their_http_status(monkeypatch):
    def fake(body, *, meta=None, **_):
        if meta is not None:
            meta.update(attempts=4, elapsed_ms=900, status=429)
        raise client.FishAPIError(429, "rate limited")

    monkeypatch.setattr(client, "synthesize", fake)
    job = _wait(story_jobs.start_render(_draft(max_concurrency=1))["id"])
    row = _read_lines(os.path.join(job["out_dir"], "requests.jsonl"))[0]
    assert row["status"] == 429 and row["attempts"] == 4 and row["error"] == "rate limited"
    assert "failed (HTTP 429)" in story_jobs.read_log(job["id"])


def test_configured_output_folder_is_used_and_listed(monkeypatch, tmp_path):
    _ok(monkeypatch)
    target = tmp_path / "My Stories"
    monkeypatch.setenv("FISH_STORY_OUTPUT_DIR", str(target))
    job = _wait(story_jobs.start_render(_draft())["id"])
    assert job["out_dir"].startswith(str(target))
    story_jobs._jobs.clear()
    assert [r["id"] for r in story_jobs.list_jobs()] == [job["id"]]


@pytest.mark.parametrize("bad", ["", "relative/folder", "   ", "C:\\" if os.name == "nt" else "/"])
def test_output_folder_must_be_a_real_absolute_folder(bad):
    with pytest.raises(ValueError):
        story_jobs.validate_output_dir(bad)


def test_output_folder_is_created_and_checked_writable(tmp_path):
    made = story_jobs.validate_output_dir(str(tmp_path / "a" / "b"))
    assert os.path.isdir(made) and os.listdir(made) == []  # the probe file is removed again
    blocker = tmp_path / "file.txt"
    blocker.write_text("x")
    with pytest.raises(ValueError):
        story_jobs.validate_output_dir(str(blocker / "sub"))


def test_unknown_job_ids_and_path_tricks_find_nothing():
    for bad in ("nope", "../../etc", "ZZZZZZZZZZZZ", ""):
        assert story_jobs.get_job(bad) is None
        assert story_jobs.read_log(bad) is None


def test_mixed_sample_rates_are_resampled_into_the_joined_file_not_skipped(monkeypatch):
    def fake(body, *, meta=None, **_):
        return _wav(48000 if body["text"] == "He looked around." else 44100, seconds=0.5)

    monkeypatch.setattr(client, "synthesize", fake)
    job = _wait(story_jobs.start_render(_draft(pause_ms=0))["id"])
    assert job["joined"] == "story.wav"
    data, rate = sf.read(os.path.join(job["out_dir"], "story.wav"))
    assert rate == 44100 and len(data) == pytest.approx(3 * 0.5 * 44100, rel=0.02)
    assert "resampled to 44100 Hz" in story_jobs.read_log(job["id"])
    # The per-line masters keep the rate Fish returned.
    odd = next(c for c in job["chunks"] if c["index"] == 2)
    assert sf.info(os.path.join(job["out_dir"], odd["file"])).samplerate == 48000


def test_same_44100_rate_join_never_resamples(monkeypatch):
    _ok(monkeypatch, rate=44100)
    monkeypatch.setattr(
        story_jobs, "_resample",
        lambda *_: (_ for _ in ()).throw(AssertionError("same-rate clips were resampled")),
    )
    job = _wait(story_jobs.start_render(_draft(sample_rate=44100, pause_ms=0))["id"])
    assert job["joined"] == "story.wav"
    assert sf.info(os.path.join(job["out_dir"], "story.wav")).samplerate == 44100


def test_two_stories_with_the_same_title_get_separate_folders_and_runs_count_up(monkeypatch):
    _ok(monkeypatch)
    a1 = _wait(story_jobs.start_render(_draft("Same"), draft_id="aaaaaa111")["id"])
    b1 = _wait(story_jobs.start_render(_draft("Same"), draft_id="bbbbbb222")["id"])
    a2 = _wait(story_jobs.start_render(_draft("Renamed"), draft_id="aaaaaa111")["id"])
    assert os.path.basename(os.path.dirname(a1["out_dir"])) == "same"
    assert os.path.basename(os.path.dirname(b1["out_dir"])) == "same-2"
    assert os.path.dirname(a2["out_dir"]) == os.path.dirname(a1["out_dir"])
    assert os.path.basename(a2["out_dir"]).startswith("run-002_")


def test_preview_shows_the_exact_requests_without_any_network_or_files(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("preview must not contact Fish")

    monkeypatch.setattr(client, "synthesize", boom)
    out = story_jobs.preview(_draft(speed=1.2, repetition_penalty=1.1, max_new_tokens=512), limit=2)
    assert out["ok"] and out["total"] == 3 and out["shown"] == 2 and out["invalid"] == 0
    first = out["requests"][0]
    assert first["body"]["reference_id"] == "vsh" and first["body"]["prosody"] == {"speed": 1.2}
    assert first["body"]["repetition_penalty"] == 1.1 and first["body"]["max_new_tokens"] == 512
    assert first["file"] == "001_Shadow_sun_2.wav"
    assert out["headers"]["model"] == "s2.1-pro" and out["headers"]["Authorization"] == "Bearer <hidden>"
    assert "sk-super-secret" not in json.dumps(out) and "$FISH_API_KEY" in out["curl"]
    assert not os.path.exists(story_jobs.default_output_root())  # nothing was created


def test_request_schema_catches_bad_bodies_and_blocks_the_render(monkeypatch):
    assert story.check_request({"text": "hi", "format": "wav", "latency": "normal", "normalize": True}) == []
    bad = story.check_request({"text": "hi", "format": "flac", "latency": "normal", "normalize": True, "oops": 1})
    assert any(b.startswith("format") for b in bad) and any(b.startswith("oops") for b in bad)
    _ok(monkeypatch)
    monkeypatch.setattr(client, "build_body", lambda *a, **k: {"text": "hi", "format": "flac"})
    with pytest.raises(ValueError, match="Line 1 would be rejected"):
        story_jobs.start_render(_draft())
