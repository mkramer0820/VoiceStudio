"""Qwen3-TTS voice profiles, emotion presets and story mode. No GPU, no sidecar: the engine is a fake."""
from __future__ import annotations

import json
import os
import struct
import time
from pathlib import Path

import pytest
import torch

from engines.qwen3_studio import emotions, profiles, registry
from engines.qwen3_studio import playground as pg
from engines.qwen3_studio import story


def _checkpoint(folder, kind="custom_voice", speakers=("ss_narrator",)):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps({
        "tts_model_type": kind, "tts_model_size": "1b7",
        "talker_config": {"spk_id": {s: 3000 + i for i, s in enumerate(speakers)}}}))
    (folder / "model.safetensors").write_bytes(b"x" * 16)


@pytest.fixture
def studio(tmp_path, monkeypatch):
    monkeypatch.setattr("core.config.DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf"))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "home"))
    monkeypatch.delenv("OMNIVOICE_QWEN3_NARRATOR_CKPT", raising=False)
    _checkpoint(tmp_path / "ckpts" / "EP09")
    _checkpoint(tmp_path / "hf" / "models--Qwen--Qwen3-TTS-12Hz-1.7B-Base" / "snapshots" / "abc", kind="base", speakers=())
    from core import model_locations

    model_locations.add("Fine-tunes", str(tmp_path / "ckpts"))
    py = tmp_path / "python.exe"
    py.write_text("")
    registry.save_settings({"output_dir": str(tmp_path / "out"), "python": str(py)})
    yield tmp_path
    from engines import qwen3_narrator

    qwen3_narrator.set_python_override(None)


class FakeEngine:
    sample_rate = 24000

    def __init__(self):
        self.calls = []
        self.last_reply_meta = {}

    def generate(self, text, **kw):
        self.calls.append((text, kw))
        self.last_reply_meta = {"complete": True, "chunks": [{"how": "stopped by itself"}]}
        return torch.full((1, 2400), 0.1)

    def unload(self):
        pass


@pytest.fixture
def fake(studio, monkeypatch):
    engine = FakeEngine()
    monkeypatch.setattr(pg, "_engine", lambda settings: engine)
    return engine


def _models():
    return {m["type"]: m for m in registry.discover_models()}


def _narrator(**kw):
    return profiles.create(profiles.ProfileBody(name="Narrator", kind="narrator", model_id=_models()["custom_voice"]["id"],
                                                speaker="ss_narrator", **kw))


def _clone(clip, name="Shadow", **kw):
    return profiles.create(profiles.ProfileBody(name=name, kind="clone", model_id=_models()["base"]["id"], ref_audio=clip,
                                                xvec_only=True, **kw))


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "ref.wav"
    path.write_bytes(b"RIFF")
    return str(path)


def _renders(fake):
    """Engine calls that render audio (the automatic embedding preparation is a call too)."""
    return [c for c in fake.calls if c[1]["voice"]["kind"] != "prepare_embedding"]


def _wait(job_id, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        data = story.get_job(job_id)
        if data["status"] != "running" and data["finished"] is not None and not pg.is_running():
            return data
        time.sleep(0.02)
    raise AssertionError("render did not finish")


# ── profiles ───────────────────────────────────────────────────────────────
def test_profile_crud_roundtrip_keeps_optional_sampling_null(studio):
    p = _narrator(notes="EP09")
    assert profiles.get(p["id"])["temperature"] is None  # no invented defaults
    updated = profiles.update(p["id"], profiles.ProfileBody(name="Narr", kind="narrator", model_id=p["model_id"], speaker="ss_narrator", top_k=20))
    assert updated["name"] == "Narr" and updated["top_k"] == 20
    assert [x["id"] for x in profiles.list_all()] == [p["id"]]
    assert profiles.delete(p["id"]) and profiles.get(p["id"]) is None and not profiles.delete(p["id"])


def test_clone_profile_needs_clip_and_text_unless_embedding_only(studio, clip):
    model = _models()["base"]["id"]
    with pytest.raises(ValueError, match="reference clip"):
        profiles.create(profiles.ProfileBody(name="x", kind="clone", model_id=model))
    with pytest.raises(ValueError, match="reference text"):
        profiles.create(profiles.ProfileBody(name="x", kind="clone", model_id=model, ref_audio=clip))
    assert _clone(clip)["kind"] == "clone"


def test_embedding_import_validates_length_and_export_is_byte_equal(studio, clip):
    p = _clone(clip)
    raw = struct.pack("<2048f", *[0.5] * 2048)
    assert profiles.import_embedding(p["id"], raw)["has_embedding"] is True
    assert profiles.export_embedding(p["id"]) == raw
    for bad in (raw[:-4], raw + b"\0\0\0\0", b"", struct.pack("<2048f", *[float("nan")] * 2048)):
        with pytest.raises(ValueError):
            profiles.import_embedding(p["id"], bad)
    assert profiles.export_embedding(p["id"]) == raw  # a refused file changes nothing


def test_changing_the_clip_drops_the_stale_embedding(studio, clip, tmp_path):
    p = _clone(clip)
    profiles.import_embedding(p["id"], struct.pack("<2048f", *[1.0] * 2048))
    other = tmp_path / "other.wav"
    other.write_bytes(b"RIFF")
    body = profiles.ProfileBody(name="Shadow", kind="clone", model_id=p["model_id"], ref_audio=str(other), xvec_only=True)
    assert profiles.update(p["id"], body)["has_embedding"] is False


def test_profile_from_a_playground_take(studio):
    model = _models()["custom_voice"]
    session = studio / "out" / "s1"
    session.mkdir(parents=True)
    (session / "session.json").write_text(json.dumps({"takes": [{
        "id": "t1", "number": 1, "model_id": model["id"], "voice": {"kind": "custom_voice", "instruct": "Whisper."},
        "params": {"temperature": 0.7}, "results": [{"voice": {"speaker": "ss_narrator"}}]}]}))
    p = profiles.from_take("s1", "t1", "From take")
    assert (p["kind"], p["speaker"], p["default_instruct"], p["temperature"]) == ("narrator", "ss_narrator", "Whisper.", 0.7)
    with pytest.raises(KeyError):
        profiles.from_take("s1", "nope", "x")


def test_resolve_sends_speaker_and_no_instruct_for_the_narrator(studio):
    p = _narrator()
    models = {m["id"]: m for m in registry.discover_models()}
    r = profiles.resolve(p, models, registry.load_settings())
    assert r["voice"]["speaker"] == "ss_narrator" and "instruct" not in r["voice"] and r["warnings"] == []


def test_resolve_applies_preset_and_flags_it_unverified_on_the_narrator(studio):
    p = _narrator(emotion={"preset": "sad", "strength": "strong"})
    models = {m["id"]: m for m in registry.discover_models()}
    r = profiles.resolve(p, models, registry.load_settings())
    assert r["voice"]["instruct"] == emotions.PRESETS["sad"]["strong"]
    assert any("UNVERIFIED" in w for w in r["warnings"])


def test_resolve_a_clone_uses_the_profile_embedding_path(studio, clip):
    p = _clone(clip)
    models = {m["id"]: m for m in registry.discover_models()}
    r = profiles.resolve(p, models, registry.load_settings())
    assert r["voice"]["kind"] == "clone" and r["voice"]["xvec_path"] == profiles.embedding_path(p["id"])


# ── emotions ───────────────────────────────────────────────────────────────
def test_every_non_neutral_preset_has_three_distinct_strengths():
    for name, texts in emotions.PRESETS.items():
        if name in ("neutral", "custom"):
            continue
        assert len({texts[s] for s in emotions.STRENGTHS}) == 3 and all(texts.values())
    assert emotions.instruct_for(emotions.EmotionChoice())[0] == ""


def test_unknown_preset_and_strength_are_refused():
    with pytest.raises(ValueError):
        emotions.instruct_for(emotions.EmotionChoice(preset="giddy"))
    with pytest.raises(ValueError):
        emotions.instruct_for(emotions.EmotionChoice(preset="sad", strength="loud"))


def test_steer_is_accepted_and_ignored_with_a_message():
    text, warnings = emotions.instruct_for(emotions.EmotionChoice(preset="joy", steer={"vector_id": "ryan_joy", "weight": 12}))
    assert text == emotions.PRESETS["joy"]["normal"] and warnings == [emotions.STEER_UNAVAILABLE]


def test_inline_emotion_tags_cut_spans():
    spans = emotions.split_emotion_tags("Calm start. [sad] Down now. [joy:very strong] Up! [neutral] Back.")
    assert [(e.preset if e else None, e.strength if e else None, t) for e, t in spans] == [
        (None, None, "Calm start."), ("sad", "normal", "Down now."), ("joy", "very_strong", "Up!"), ("neutral", "normal", "Back.")]


# ── parsing ────────────────────────────────────────────────────────────────
def test_parse_detects_tags_colon_lines_carry_and_narration():
    text = "The night fell.\n[Shadow] Hello there.\nMara: Who goes?\n[Shadow]\nOne line.\nTwo line.\n\nBack to narration."
    chunks = story.parse_script(text, 100)
    assert [(c["speaker"], c["text"]) for c in chunks] == [
        (None, "The night fell."), ("Shadow", "Hello there."), ("Mara", "Who goes?"), ("Shadow", "One line."),
        ("Shadow", "Two line."), (None, "Back to narration.")]
    assert [c["name"] for c in story.characters(chunks)] == [None, "Shadow", "Mara"]


def test_parse_does_not_mistake_emotion_tags_for_characters_and_splits_long_text():
    chunks = story.parse_script("[sad] Low. [joy] High.", 100)
    assert [(c["speaker"], c["emotion"]["preset"]) for c in chunks] == [(None, "sad"), (None, "joy")]
    long = story.parse_script("[Ann] " + "Word. " * 60, 60)
    assert len(long) > 3 and all(len(c["text"]) <= 60 and c["speaker"] == "Ann" for c in long)


# ── rendering ──────────────────────────────────────────────────────────────
def _draft(narr, shadow=None, **kw):
    chunks = [{"id": "a", "text": "Night."}, {"id": "b", "text": "Hello.", "speaker": "Shadow"},
              {"id": "c", "text": "Dawn.", "speaker": None}, {"id": "d", "text": "Bye.", "speaker": "Shadow"}]
    return story.Draft.model_validate({"title": "My Book", "chunks": chunks, "narrator_profile_id": narr["id"],
                                       "cast": {"Shadow": shadow["id"]} if shadow else {}, **kw})


def test_render_groups_by_model_keeps_story_order_and_uses_layout_and_names(fake, clip):
    narr, shadow = _narrator(), _clone(clip)
    job = _wait(story.start_render(_draft(narr, shadow))["id"])
    assert job["status"] == "done" and job["total"] == 4
    # custom_voice model first (first appearance), then the Base clone: each model is loaded once
    assert [c[1]["voice"]["kind"] for c in _renders(fake)] == ["custom_voice", "custom_voice", "clone", "clone"]
    names = [c["file"] for c in sorted(job["chunks"], key=lambda c: c["index"])]
    assert names == ["001_narration_Narrator.wav", "002_Shadow_Shadow.wav", "003_narration_Narrator.wav", "004_Shadow_Shadow.wav"]
    folder = Path(job["folder"])
    assert folder.parent.name == "my-book" and folder.name.startswith("run-001_") and (folder / "story.wav").is_file()
    assert (folder / "manifest.json").is_file() and (folder / "story.json").is_file()
    assert job["joined"] == "story.wav"


def test_unmapped_character_falls_back_to_narrator_with_a_warning_not_a_block(fake):
    narr = _narrator()
    job = _wait(story.start_render(_draft(narr))["id"])
    assert job["status"] == "done"
    assert any("Shadow" in w and "no profile" in w for w in job["warnings"])


def test_no_voice_at_all_is_refused_before_anything_runs(fake):
    draft = story.Draft.model_validate({"chunks": [{"id": "a", "text": "Hi."}]})
    with pytest.raises(ValueError, match="no voice"):
        story.start_render(draft)
    assert fake.calls == [] and not pg.is_running()


def test_chunk_emotion_wins_over_character_emotion_and_profile(fake, clip):
    narr = _narrator()
    draft = story.Draft.model_validate({
        "chunks": [{"id": "a", "text": "One.", "speaker": "Ann", "emotion": {"preset": "anger"}}, {"id": "b", "text": "Two.", "speaker": "Ann"}],
        "narrator_profile_id": narr["id"], "cast": {"Ann": narr["id"]}, "emotions": {"Ann": {"preset": "joy", "strength": "strong"}}})
    job = _wait(story.start_render(draft)["id"])
    assert [c[1]["voice"]["instruct"] for c in fake.calls] == [emotions.PRESETS["anger"]["normal"], emotions.PRESETS["joy"]["strong"]]
    assert any("UNVERIFIED" in w for w in job["chunks"][0]["warnings"])


def test_sampling_overrides_are_sent_only_when_set(fake):
    narr = _narrator(top_p=0.8)
    _wait(story.start_render(_draft(narr))["id"])
    kw = fake.calls[0][1]
    assert kw["top_p"] == 0.8 and "temperature" not in kw and "top_k" not in kw


def test_warmup_is_off_by_default_and_once_per_voice_when_on(fake, clip):
    narr, shadow = _narrator(), _clone(clip)
    _wait(story.start_render(_draft(narr, shadow))["id"])
    assert len(_renders(fake)) == 4
    fake.calls.clear()
    _wait(story.start_render(_draft(narr, shadow, settings={"warmup": True}))["id"])
    assert len(_renders(fake)) == 6  # one throwaway per voice


def test_one_failing_chunk_does_not_stop_the_book_and_rerender_keeps_the_old_file(fake):
    narr = _narrator()
    orig = fake.generate
    state = {"fail": True}

    def flaky(text, **kw):
        if text == "Hello." and state["fail"]:
            raise RuntimeError("boom")
        return orig(text, **kw)

    fake.generate = flaky
    job = _wait(story.start_render(_draft(narr), story_id="a" * 12)["id"])
    by_id = {c["id"]: c for c in job["chunks"]}
    assert by_id["b"]["status"] == "error" and by_id["a"]["status"] == "done"
    state["fail"] = False
    again = _wait(story.start_render(_draft(narr), story_id="a" * 12, chunk_ids=["b"], run=job["run"])["id"])
    by_id = {c["id"]: c for c in again["chunks"]}
    assert again["run"] == job["run"] and by_id["b"]["status"] == "done" and by_id["a"]["history"] == ["001_narration_Narrator.wav"]
    # re-render of a finished chunk keeps the old file and adds _r2
    third = _wait(story.start_render(_draft(narr), story_id="a" * 12, chunk_ids=["a"], run=job["run"])["id"])
    a = {c["id"]: c for c in third["chunks"]}["a"]
    assert a["history"] == ["001_narration_Narrator.wav", "001_narration_Narrator_r2.wav"] and a["file"].endswith("_r2.wav")
    assert story.audio_path(third["story"], third["run"], a["file"])


def test_second_render_is_refused_while_one_runs(fake):
    narr = _narrator()
    fake.generate = lambda text, **kw: (time.sleep(0.3), torch.zeros(1, 240))[1]
    first = story.start_render(_draft(narr))
    with pytest.raises(RuntimeError, match="already in progress"):
        story.start_render(_draft(narr))
    story.cancel_job(first["id"])
    _wait(first["id"])


def test_audio_paths_refuse_traversal(fake):
    narr = _narrator()
    job = _wait(story.start_render(_draft(narr))["id"])
    assert story.audio_path(job["story"], job["run"], "001_narration_Narrator.wav")
    assert story.audio_path(job["story"], job["run"], "../manifest.json") is None
    assert story.audio_path("..", job["run"], "story.wav") is None
    assert story.list_runs(job["story"])[0]["done"] == 4


def test_drafts_roundtrip(studio):
    saved = story.save_draft(None, story.Draft(title="T"))
    assert story.load_draft(saved["id"])["title"] == "T" and story.list_drafts()[0]["id"] == saved["id"]
    assert story.delete_draft(saved["id"]) and story.load_draft(saved["id"]) is None


def test_sidecar_reads_and_writes_the_headerless_float32_bin(tmp_path):
    """The C engine's x-vector layout is raw f32[2048]: the sidecar helpers must accept it (and keep .npy)."""
    import importlib.util

    import numpy as np

    spec = importlib.util.spec_from_file_location("qwen_sidecar_main", Path(__file__).parents[1] / "backend/engines/qwen3_narrator/main.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    vec = np.arange(2048, dtype="float32")
    for ext in ("bin", "npy"):
        path = str(tmp_path / ("v." + ext))
        mod._save_vec(path, vec)
        assert np.array_equal(mod._load_vec(path).numpy(), vec)
    assert os.path.getsize(tmp_path / "v.bin") == 2048 * 4


def test_routes_profiles_embedding_parse_and_emotions(studio, clip):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api.dependencies import require_admin
    from api.routers import qwen_studio

    app = FastAPI()
    app.include_router(qwen_studio.router)
    app.dependency_overrides[require_admin] = lambda: None
    c = TestClient(app, headers={"Sec-Fetch-Site": "same-origin"})
    model = _models()["custom_voice"]["id"]
    made = c.post("/api/qwen/profiles", json={"name": "N", "kind": "narrator", "model_id": model, "speaker": "ss_narrator"})
    assert made.status_code == 200
    pid = made.json()["id"]
    assert c.put(f"/api/qwen/profiles/{pid}/embedding", content=b"short").status_code == 422
    assert c.get(f"/api/qwen/profiles/{pid}/embedding").status_code == 404
    raw = struct.pack("<2048f", *[0.25] * 2048)
    assert c.put(f"/api/qwen/profiles/{pid}/embedding", content=raw).json()["has_embedding"] is True
    assert c.get(f"/api/qwen/profiles/{pid}/embedding").content == raw
    assert c.post("/api/qwen/profiles", json={"name": "bad", "kind": "clone", "model_id": model}).status_code == 422
    assert c.post("/api/qwen/story/parse", json={"text": "[Ann] Hi."}).json()["characters"][0]["name"] == "Ann"
    cat = c.get("/api/qwen/emotions").json()
    assert "UNVERIFIED" in cat["unverified_on_narrator"] and {p["name"] for p in cat["presets"]} >= {"sad", "custom"}
    assert c.delete(f"/api/qwen/profiles/{pid}").status_code == 200


# ── clips and automatic embeddings ────────────────────────────────────────
def _wav_bytes(seconds=0.2, rate=16000):
    import io

    import numpy as np
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(seconds * rate), dtype="float32") + 0.01, rate, format="WAV")
    return buf.getvalue()


def _embedding_writer(fake, fail=False):
    """The fake sidecar writes the .bin the way the real ``prepare_embedding`` op does."""
    orig = fake.generate

    def generate(text, **kw):
        voice = kw.get("voice") or {}
        if voice.get("kind") == "prepare_embedding":
            fake.calls.append((text, kw))
            if fail:
                raise RuntimeError("no base model")
            with open(voice["xvec_path"], "wb") as fh:
                fh.write(struct.pack("<2048f", *[0.1] * 2048))
            return torch.zeros(1, 1)
        return orig(text, **kw)

    fake.generate = generate


def test_a_recorded_clip_is_stored_as_wav_and_junk_is_refused(studio):
    path = profiles.store_clip(_wav_bytes(), "recording.webm")  # the name is a hint; the content decides
    import soundfile as sf

    assert path.endswith(".wav") and sf.info(path).frames > 0
    with pytest.raises(ValueError, match="empty or too short"):
        profiles.store_clip(b"x" * 10, "a.wav")


def test_saving_a_clone_prepares_its_embedding_when_the_engine_is_idle(fake, studio):
    _embedding_writer(fake)
    clip = profiles.store_clip(_wav_bytes() * 1, "a.wav")
    p = _clone(clip)
    assert p["embedding_state"] == "none" and p["has_embedding"] is False
    assert profiles.prepare_in_background(p["id"]) == "preparing"
    end = time.time() + 5
    while time.time() < end and profiles.get(p["id"])["embedding_state"] != "ready":
        time.sleep(0.02)
    got = profiles.get(p["id"])
    assert got["embedding_state"] == "ready" and got["has_embedding"] and not pg.is_running()
    assert fake.calls[0][1]["voice"]["kind"] == "prepare_embedding"


def test_a_busy_engine_leaves_the_embedding_for_the_first_render(fake, clip):
    p = _clone(clip)
    assert pg._run_lock.acquire(blocking=False)
    try:
        assert profiles.prepare_in_background(p["id"]) == "none"
    finally:
        pg._run_lock.release()
    assert fake.calls == []


def test_render_prepares_a_missing_embedding_first_and_shows_it(fake, clip):
    _embedding_writer(fake)
    narr, shadow = _narrator(), _clone(clip)
    job = _wait(story.start_render(_draft(narr, shadow))["id"])
    kinds = [c[1]["voice"]["kind"] for c in fake.calls]
    assert kinds[0] == "prepare_embedding" and kinds.count("prepare_embedding") == 1
    assert job["status"] == "done" and job["preparing"] == [] and profiles.get(shadow["id"])["has_embedding"]
    fake.calls.clear()
    _wait(story.start_render(_draft(narr, shadow))["id"])
    assert "prepare_embedding" not in [c[1]["voice"]["kind"] for c in fake.calls]  # cached: never recomputed


def test_a_failed_embedding_is_a_warning_not_a_block(fake, clip):
    _embedding_writer(fake, fail=True)
    narr, shadow = _narrator(), _clone(clip)
    job = _wait(story.start_render(_draft(narr, shadow))["id"])
    assert job["status"] == "done" and all(c["status"] == "done" for c in job["chunks"])
    assert any(m["level"] == "WARN" and "Could not prepare" in m["message"] for m in job["log"])
    got = profiles.get(shadow["id"])
    assert got["embedding_state"] == "failed" and "no base model" in got["embedding_error"]


def test_the_sidecar_prepare_op_loads_no_tts_model(tmp_path, monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location("qwen_sidecar_main2", Path(__file__).parents[1] / "backend/engines/qwen3_narrator/main.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sent = []
    monkeypatch.setattr(mod, "_xvector", lambda ref, base, cache: torch.zeros(2048))
    monkeypatch.setattr(mod, "_load_model", lambda *a, **k: (_ for _ in ()).throw(AssertionError("model loaded")))
    monkeypatch.setattr(mod, "_send", lambda stream, obj: sent.append(obj))
    mod._handle_synthesize({"text": "x", "voice": {"kind": "prepare_embedding", "ref_audio": "a", "base_model_path": "b", "xvec_path": "c.bin"}}, None)
    assert sent[0]["op"] == "audio" and sent[0]["prepared"] == 2048 and sent[0]["n_samples"] == 1
    with pytest.raises(ValueError, match="xvec_path"):
        mod._handle_synthesize({"text": "x", "voice": {"kind": "prepare_embedding", "ref_audio": "a", "base_model_path": "b"}}, None)


def test_a_prepared_embedding_is_keyed_to_the_clip_and_an_imported_one_is_not(studio, clip):
    import json as _json
    import os as _os

    p = _clone(clip)
    raw = struct.pack("<2048f", *[0.5] * 2048)
    # made here: a key file records the clip; the same clip keeps it valid
    with open(profiles.embedding_path(p["id"]), "wb") as fh:
        fh.write(raw)
    with open(profiles.embedding_path(p["id"]) + ".json", "w", encoding="utf-8") as fh:
        _json.dump({"ref": profiles._ref_stamp(clip), "base": "b"}, fh)
    assert profiles.get(p["id"])["has_embedding"] is True
    # the clip is edited in place (new size): the embedding no longer belongs to it, so it is not reported ready
    with open(clip, "ab") as fh:
        fh.write(b"\0" * 64)
    _os.utime(clip, None)
    stale = profiles.get(p["id"])
    assert stale["has_embedding"] is False and profiles.needs_embedding(stale)
    # an imported embedding carries no key and is trusted
    assert profiles.import_embedding(p["id"], raw)["has_embedding"] is True
