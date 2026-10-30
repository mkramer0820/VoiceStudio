"""Qwen3-TTS studio: model registry, run planning, naming, and the playground runner.

No GPU and no sidecar: the engine is replaced by a fake that records what it was asked to render.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pytest
import torch

from engines.qwen3_studio import playground as pg
from engines.qwen3_studio import registry


# ── fixtures ───────────────────────────────────────────────────────────────
def _checkpoint(folder, kind="custom_voice", speakers=("ss_narrator",)):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps({
        "tts_model_type": kind, "tts_model_size": "1b7",
        "talker_config": {"spk_id": {s: 3000 + i for i, s in enumerate(speakers)}}}))
    (folder / "model.safetensors").write_bytes(b"x" * 16)
    return folder


@pytest.fixture
def studio(tmp_path, monkeypatch):
    monkeypatch.setattr("core.config.DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf"))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "home"))  # keep the real default cache out
    monkeypatch.delenv("OMNIVOICE_QWEN3_NARRATOR_CKPT", raising=False)
    root = tmp_path / "ckpts"
    _checkpoint(root / "Run_2_EP09" / "checkpoints" / "checkpoint-epoch-9")
    _checkpoint(tmp_path / "hf" / "models--Qwen--Qwen3-TTS-12Hz-1.7B-Base" / "snapshots" / "abc", kind="base", speakers=())
    from core import model_locations

    model_locations.add("Fine-tunes", str(root))
    registry.save_settings({"output_dir": str(tmp_path / "out"), "python": "py"})
    yield tmp_path
    from engines import qwen3_narrator

    qwen3_narrator.set_python_override(None)


def _models():
    return {m["type"]: m for m in registry.discover_models()}


# ── registry ───────────────────────────────────────────────────────────────
def test_models_are_found_with_stable_ids(studio):
    models = registry.discover_models()
    by_type = {m["type"]: m for m in models}
    assert by_type["custom_voice"]["id"] == "Run_2_EP09-checkpoints-checkpoint-epoch-9"
    assert by_type["custom_voice"]["speakers"] == ["ss_narrator"]
    assert by_type["base"]["id"] == "hf-qwen3-tts-12hz-1.7b-base"
    assert [m["id"] for m in registry.discover_models()] == [m["id"] for m in models]
    assert registry.first_base()["type"] == "base"


def test_folder_without_weights_is_not_a_model(studio, tmp_path):
    (tmp_path / "ckpts" / "junk").mkdir()
    (tmp_path / "ckpts" / "junk" / "config.json").write_text("{}")
    assert all("junk" not in m["id"] for m in registry.discover_models())


def test_stock_models_in_the_users_default_cache_are_found_when_the_app_moves_its_own(studio, tmp_path, monkeypatch):
    default = tmp_path / "home" / ".cache" / "huggingface" / "hub"
    _checkpoint(default / "models--Qwen--Qwen3-TTS-12Hz-0.6B-Base" / "snapshots" / "def", kind="base", speakers=())
    ids = [m["id"] for m in registry.discover_models()]
    assert "hf-qwen3-tts-12hz-0.6b-base" in ids and "hf-qwen3-tts-12hz-1.7b-base" in ids


def test_settings_roundtrip_and_clamp(studio):
    saved = registry.save_settings({"warmup_runs": 99})
    assert saved["warmup_runs"] == 10 and registry.load_settings()["warmup_runs"] == 10


# ── planning and naming ────────────────────────────────────────────────────
def _take(i, model, kind="custom_voice", **kw):
    voice = {"kind": kind, **kw.pop("voice", {})}
    return pg.Take(id=f"t{i}", label=kw.pop("label", ""), text=f"line {i}", model_id=model, voice=voice, **kw)


def test_grouping_loads_each_model_once_but_keeps_first_appearance_order():
    takes = [_take(1, "A"), _take(2, "B"), _take(3, "A"), _take(4, "B")]
    ungrouped = pg.plan(takes, False)
    grouped = pg.plan(takes, True)
    assert [i for i, _ in ungrouped] == [0, 1, 2, 3] and pg.swaps(takes, ungrouped) == 4
    assert [i for i, _ in grouped] == [0, 2, 1, 3] and pg.swaps(takes, grouped) == 2


def test_repeat_expands_and_filenames_are_clean_and_unique():
    takes = [_take(1, "Run_2_EP09-x", label="Angry! / loud", repeat=2, voice={"speaker": "ss_narrator"})]
    order = pg.plan(takes, True)
    assert order == [(0, 1), (0, 2)]
    names = [pg.take_filename(1, r, 2, takes[0].model_id, takes[0].voice, takes[0].label) for _, r in order]
    assert names[0] != names[1] and all(n.endswith(".wav") and "/" not in n and " " not in n for n in names)
    assert names[0].startswith("take-001-r1_")


def test_pair_check_names_the_problem(studio):
    models = _models()
    with pytest.raises(ValueError, match="needs a base model"):
        pg._check_pair(models["custom_voice"], pg.Voice(kind="clone"))
    pg._check_pair(models["custom_voice"], pg.Voice(kind="clone_on_custom"))
    pg._check_pair(models["base"], pg.Voice(kind="clone"))


def test_clone_needs_a_clip_and_its_text(studio, tmp_path):
    model = _models()["base"]
    take = _take(1, model["id"], "clone", voice={"ref_audio": str(tmp_path / "nope.wav")})
    with pytest.raises(ValueError, match="reference clip"):
        pg.resolve_voice(take, model, registry.load_settings())
    clip = tmp_path / "ref.wav"
    clip.write_bytes(b"RIFF")
    take = _take(1, model["id"], "clone", voice={"ref_audio": str(clip)})
    with pytest.raises(ValueError, match="reference text"):
        pg.resolve_voice(take, model, registry.load_settings())
    take = _take(1, model["id"], "clone", voice={"ref_audio": str(clip), "xvec_only": True})
    sent, info = pg.resolve_voice(take, model, registry.load_settings())
    assert sent["xvec_only"] is True and info["kind"] == "clone"


def test_clone_on_custom_gets_the_base_and_a_cache_path(studio, tmp_path):
    custom = _models()["custom_voice"]
    clip = tmp_path / "ref.wav"
    clip.write_bytes(b"RIFF")
    take = _take(1, custom["id"], "clone_on_custom", voice={"ref_audio": str(clip)})
    sent, _ = pg.resolve_voice(take, custom, registry.load_settings())
    assert sent["speaker"] == "ss_narrator" and sent["base_model_path"].endswith("abc")
    assert sent["xvec_path"].endswith(".npy")


def test_speaker_is_never_picked_silently(studio):
    one = _models()["custom_voice"]
    many = {**one, "label": "Stock CustomVoice", "speakers": ["aiden", "ryan", "vivian"]}
    assert pg.pick_speaker(None, one) == ("ss_narrator", None)  # one speaker: unambiguous
    assert pg.pick_speaker("RYAN", many) == ("ryan", None)  # case-insensitive, the model's own spelling
    with pytest.raises(ValueError, match="Pick a speaker.*aiden, ryan, vivian"):
        pg.pick_speaker("", many)
    with pytest.raises(ValueError, match="not a speaker.*stock voices you can borrow"):
        pg.pick_speaker("bob", many)
    with pytest.raises(ValueError, match="no built-in speakers"):
        pg.pick_speaker(None, {**one, "speakers": []})


def test_stock_voice_names_can_be_borrowed_with_a_warning(studio):
    models = _models()
    narrator, base = models["custom_voice"], models["base"]
    assert pg.pick_speaker("ryan", narrator) == ("ryan", registry.STOCK_SPEAKERS["ryan"])  # EP09 has no trained ryan
    assert pg.pick_speaker("ryan", base)[1] == 3061
    assert "ryan" in narrator["borrowable_speakers"] and "ss_narrator" not in narrator["borrowable_speakers"]
    take = _take(1, narrator["id"], voice={"speaker": "ryan"})
    sent, info = pg.resolve_voice(take, narrator, registry.load_settings())
    assert sent["speaker"] == "ryan" and sent["speaker_row"] == 3061 and info["borrowed"] is True
    assert "sound different from the original ryan" in info["warning"]
    own, own_info = pg.resolve_voice(_take(2, narrator["id"]), narrator, registry.load_settings())
    assert "speaker_row" not in own and "warning" not in own_info  # the narrator's own voice carries no warning


def test_a_base_model_takes_a_borrowed_voice_but_not_an_unknown_one(studio):
    base = _models()["base"]
    pg._check_pair(base, pg.Voice(kind="custom_voice", speaker="ryan"))
    with pytest.raises(ValueError, match="needs a custom_voice model"):
        pg._check_pair(base, pg.Voice(kind="custom_voice", speaker="bob"))


# ── the runner, with a fake engine ─────────────────────────────────────────
class FakeEngine:
    sample_rate = 24000

    def __init__(self):
        self.calls = []
        self.last_reply_meta = {}

    def generate(self, text, **kw):
        self.calls.append((text, kw))
        self.last_reply_meta = {"complete": True, "chunks": [{"how": "stopped by itself"}], "vram_allocated_mb": 4000}
        return torch.zeros(1, 2400)

    def unload(self):
        pass


def _wait(job_id, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        data = pg.get_job(job_id)
        if data["status"] != "running" and data["finished"] is not None and not pg.is_running():
            return data
        time.sleep(0.02)
    raise AssertionError("run did not finish")


@pytest.fixture
def fake(studio, monkeypatch, tmp_path):
    engine = FakeEngine()
    monkeypatch.setattr(pg, "_engine", lambda settings: engine)
    py = tmp_path / "python.exe"
    py.write_text("")
    registry.save_settings({"python": str(py)})
    return engine


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "ref.wav"
    path.write_bytes(b"RIFF")
    return str(path)


def _request(clip, **kw):
    models = _models()
    custom, base = models["custom_voice"], models["base"]
    takes = [
        {"id": "a", "label": "narrator", "text": "One.", "model_id": custom["id"], "voice": {"kind": "custom_voice"}},
        {"id": "b", "label": "clone", "text": "Two.", "model_id": base["id"],
         "voice": {"kind": "clone", "ref_audio": clip, "xvec_only": True}},
        {"id": "c", "label": "again", "text": "Three.", "model_id": custom["id"],
         "voice": {"kind": "custom_voice", "speaker": "ss_narrator", "instruct": "Whisper."}},
    ]
    return pg.RunRequest.model_validate({"session_name": "My Test", "takes": takes, **kw})


def test_run_writes_files_and_manifest_and_warms_each_model_and_voice(fake, clip):
    job = _wait(pg.start_run(_request(clip, warmup_runs=2))["id"])
    assert job["status"] == "done" and job["done"] == 3
    real = [t for t in fake.calls if t[0] in ("One.", "Two.", "Three.")]
    assert [t[0] for t in real] == ["One.", "Three.", "Two."]  # grouped: the model loads once
    warm = [t for t in fake.calls if t[0] not in ("One.", "Two.", "Three.")]
    assert len(warm) == 4  # two voice signatures (custom_voice, clone) x 2 warm-up runs
    manifest = json.load(open(os.path.join(job["folder"], "session.json"), encoding="utf-8"))
    assert [t["id"] for t in manifest["takes"]] == ["a", "b", "c"]  # written order is kept
    result = manifest["takes"][0]["results"][0]
    assert result["status"] == "done" and os.path.isfile(os.path.join(job["folder"], result["file"]))
    assert result["complete"] is True and result["seconds"] == 0.1


def test_one_failing_take_does_not_stop_the_rest(fake, clip):
    original = fake.generate

    def boom(text, **kw):
        if text == "Two.":
            raise RuntimeError("sidecar exploded")
        return original(text, **kw)

    fake.generate = boom
    job = _wait(pg.start_run(_request(clip, warmup_runs=0))["id"])
    by_id = {t["id"]: t["results"][0] for t in job["takes"]}
    assert by_id["b"]["status"] == "error" and "exploded" in by_id["b"]["error"]
    assert by_id["a"]["status"] == by_id["c"]["status"] == "done"


def test_second_run_is_refused_while_one_is_running(fake, clip):
    release = threading.Event()
    original = fake.generate
    fake.generate = lambda text, **kw: (release.wait(5), original(text, **kw))[1]
    first = pg.start_run(_request(clip, warmup_runs=0))
    try:
        with pytest.raises(RuntimeError, match="already in progress"):
            pg.start_run(_request(clip, warmup_runs=0))
    finally:
        release.set()
    _wait(first["id"])


def test_cancel_stops_between_takes(fake, clip):
    gate = threading.Event()
    original = fake.generate
    fake.generate = lambda text, **kw: (gate.wait(5), original(text, **kw))[1]
    job = pg.start_run(_request(clip, warmup_runs=0))
    assert pg.cancel_job(job["id"])
    gate.set()
    assert _wait(job["id"])["status"] == "cancelled"


def test_notes_and_star_are_saved_in_the_manifest(fake, clip):
    job = _wait(pg.start_run(_request(clip, warmup_runs=0))["id"])
    take = pg.annotate_take(job["session"], "a", notes="this one is the real narrator", starred=True)
    assert take["starred"] is True
    assert pg.read_session(job["session"])["takes"][0]["notes"] == "this one is the real narrator"
    assert pg.annotate_take(job["session"], "nope", notes="x") is None


def test_audio_and_session_paths_refuse_traversal(studio, tmp_path):
    assert pg.audio_path("x", "../../etc/passwd") is None
    assert pg.audio_path("x", "notes.txt") is None
    with pytest.raises(ValueError):
        pg._session_path(str(tmp_path), "../escape")


def test_unknown_model_is_rejected_before_anything_runs(fake):
    bad = pg.RunRequest.model_validate({"takes": [{"id": "a", "text": "x", "model_id": "nope"}]})
    with pytest.raises(ValueError, match="not found"):
        pg.start_run(bad)
    assert not pg.is_running()


def test_python_setting_falls_back_to_project_venv_only(monkeypatch):
    from engines.qwen3_studio import registry

    monkeypatch.setattr(registry, "default_python", lambda: "/repo/.venv-qwen/python")
    assert registry.effective_python({"python": ""}) == "/repo/.venv-qwen/python"
    assert registry.effective_python({"python": " /mine/python "}) == "/mine/python"
    monkeypatch.setattr(registry, "default_python", lambda: "")
    assert registry.effective_python({"python": ""}) is None  # never PATH, base or the app's venv


def test_scan_finds_a_root_that_is_itself_a_checkpoint(tmp_path):
    from engines.qwen3_studio import registry

    ckpt = tmp_path / "ss_narrator"
    ckpt.mkdir()
    (ckpt / "model.safetensors").write_bytes(b"x")
    (ckpt / "config.json").write_text(
        '{"tts_model_type": "custom_voice", "tts_model_size": "1b7", "talker_config": {"spk_id": {"ss_narrator": 3000}}}',
        encoding="utf-8")
    assert [m["id"] for m in registry._scan(str(ckpt))] == ["ss_narrator"]
    assert [m["id"] for m in registry._scan(str(tmp_path))] == ["ss_narrator"]


# ── LoRA adapters ──────────────────────────────────────────────────────────
def _adapter_setup(studio):
    from safetensors.torch import save_file

    snap = studio / "hf" / "models--Qwen--Qwen3-TTS-12Hz-1.7B-Base" / "snapshots" / "abc"
    key = "talker.model.layers.0.self_attn.q_proj"
    g = torch.Generator().manual_seed(0)
    save_file({key + ".weight": (torch.randn(8, 8, generator=g) * 0.02).to(torch.bfloat16), "talker.model.codec_embedding.weight": torch.zeros(3072, 4, dtype=torch.bfloat16),
               "speaker_encoder.w": torch.ones(2, dtype=torch.bfloat16)}, str(snap / "model.safetensors"))
    folder = studio / "ckpts" / "lora-run" / "adapter-epoch-3"
    folder.mkdir(parents=True)
    save_file({key + ".lora_A.weight": torch.randn(2, 8, generator=g), key + ".lora_B.weight": torch.randn(8, 2, generator=g) * 0.5}, str(folder / "adapter_model.safetensors"))
    (folder / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 4}))
    (folder / "speakers.json").write_text(json.dumps({"amy": 3000, "bob": 3020}))
    (folder / "lora_meta.json").write_text(json.dumps({"base_model": "/content/models/Qwen3-TTS-12Hz-1.7B-Base"}))
    save_file({"amy": torch.ones(4), "bob": torch.full((4,), 2.0)}, str(folder / "speaker_embeddings.safetensors"))
    return folder


def test_an_adapter_folder_is_found_as_its_own_kind_and_is_not_runnable(studio):
    _adapter_setup(studio)
    adapter = next(m for m in registry.discover_models() if m["type"] == "adapter")
    assert adapter["speakers"] == ["amy", "bob"] and adapter["rank"] == 2 and adapter["borrowable_speakers"] == []
    assert registry.resolve_adapter_base(adapter, registry.discover_models())["type"] == "base"
    with pytest.raises(ValueError, match="merge it into its Base model first"):
        pg._check_pair(adapter, pg.Voice(kind="custom_voice", speaker="amy"))
    # a merged checkpoint that also carries its adapter files is listed once, as the checkpoint it is
    both = studio / "ckpts" / "lora-run" / "checkpoint-epoch-3"
    _checkpoint(both, speakers=("amy", "bob"))
    (both / "adapter_model.safetensors").write_bytes(b"x")
    (both / "adapter_config.json").write_text("{}")
    kinds = [(m["type"], m.get("has_adapter")) for m in registry.discover_models() if "lora-run" in m["id"]]
    assert ("custom_voice", True) in kinds and kinds.count(("adapter", None)) == 1


def test_merging_writes_a_standard_multi_speaker_checkpoint_into_a_settings_storage_folder(studio):
    from core import model_locations

    _adapter_setup(studio)
    adapter = next(m for m in registry.discover_models() if m["type"] == "adapter")
    loc = model_locations.load()[0]
    result = registry.merge_adapter_model(adapter["id"], loc["id"], "Duo merged", lora_scale=1.0)
    assert result["speakers"] == ["amy", "bob"] and result["path"].startswith(os.path.abspath(loc["path"])) and result["scale"] == 2.0
    merged = next(m for m in registry.discover_models() if m["path"] == result["path"])
    assert merged["type"] == "custom_voice" and merged["speakers"] == ["amy", "bob"]
    with pytest.raises(ValueError, match="already exists"):
        registry.merge_adapter_model(adapter["id"], loc["id"], "Duo merged")  # nothing is overwritten
    with pytest.raises(ValueError, match="Settings -> Storage"):
        registry.merge_adapter_model(adapter["id"], "not-a-location", "x")  # no free path: only a registered folder
    with pytest.raises(ValueError, match="not a LoRA adapter"):
        registry.merge_adapter_model(merged["id"], loc["id"], "y")
    assert registry.merge_adapter_model(adapter["id"], loc["id"], "scaled", lora_scale=0.3)["scale"] == pytest.approx(0.6)


def test_the_merge_route_returns_422_with_the_reason_and_takes_only_a_registered_location(studio):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api.dependencies import require_admin
    from api.routers import qwen_studio

    app = FastAPI()
    app.include_router(qwen_studio.router)
    app.dependency_overrides[require_admin] = lambda: None
    client = TestClient(app)
    _adapter_setup(studio)
    adapter = next(m for m in registry.discover_models() if m["type"] == "adapter")
    bad = client.post("/api/qwen/models/merge-adapter", json={"model_id": adapter["id"], "location_id": "nope", "name": "x"})
    assert bad.status_code == 422 and "Settings -> Storage" in bad.json()["detail"]
    assert client.post("/api/qwen/models/merge-adapter", json={"model_id": adapter["id"], "location_id": "../..", "name": "x"}).status_code == 422
    from core import model_locations

    ok = client.post("/api/qwen/models/merge-adapter", json={"model_id": adapter["id"], "location_id": model_locations.load()[0]["id"], "name": "viaroute", "lora_scale": 1.0})
    assert ok.status_code == 200 and ok.json()["speakers"] == ["amy", "bob"]
