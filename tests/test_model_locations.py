"""Settings -> Storage -> Model locations: the shared list, its routes and Qwen discovery from it."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi import HTTPException

from api.routers import settings as s
from core import model_locations, path_authorization
from engines.qwen3_studio import registry


def _checkpoint(folder, kind="custom_voice", speakers=("ss_narrator",)):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps({
        "tts_model_type": kind, "tts_model_size": "1b7",
        "talker_config": {"spk_id": {n: 3000 + i for i, n in enumerate(speakers)}}}))
    (folder / "model.safetensors").write_bytes(b"x" * 16)
    return folder


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr("core.config.DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf"))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "home"))
    monkeypatch.delenv("OMNIVOICE_QWEN3_NARRATOR_CKPT", raising=False)
    auth = tmp_path / "auth"
    auth.mkdir()
    monkeypatch.setattr(path_authorization, "_AUTH_DIR", str(auth))
    return tmp_path


_n = iter(range(1, 10_000))


def _token(path, kind="model_location"):
    token = f"{next(_n):064x}"
    with open(os.path.join(path_authorization._AUTH_DIR, token + ".json"), "w", encoding="utf-8") as fh:
        json.dump({"token": token, "kind": kind, "path": str(path)}, fh)
    return token


def test_add_list_status_and_remove_never_delete_files(env):
    ckpt = _checkpoint(env / "tunes" / "a" / "ckpt")
    row = s.add_model_location(s._LocationAddBody(authorization=_token(env / "tunes"), name="My tunes"))
    assert row["id"] == "my-tunes" and row["exists"] and row["models_found"] == 1
    assert row["models"][0]["type"] == "custom_voice" and row["models"][0]["speakers"] == ["ss_narrator"]
    listed = s.get_model_locations()["locations"]
    assert [r["id"] for r in listed] == ["my-tunes"]
    # survives a restart: it is read back from disk
    assert [r["id"] for r in model_locations.load()] == ["my-tunes"]
    assert s.delete_model_location("my-tunes") == {"removed": "my-tunes"}
    assert s.get_model_locations()["locations"] == []
    assert ckpt.is_dir() and (ckpt / "model.safetensors").is_file()
    with pytest.raises(HTTPException) as err:
        s.delete_model_location("my-tunes")
    assert err.value.status_code == 404


def test_add_needs_a_native_authorization_of_the_right_kind(env):
    (env / "x").mkdir()
    for bad in ("f" * 64, _token(env / "x", kind="models_dir")):
        with pytest.raises(HTTPException) as err:
            s.add_model_location(s._LocationAddBody(authorization=bad, name="x"))
        assert err.value.status_code == 403
    assert model_locations.load() == []


def test_a_missing_folder_is_reported_not_raised(env):
    model_locations.add("Gone", str(env / "nope"))
    row = s.get_model_locations()["locations"][0]
    assert row["exists"] is False and row["error"] == "folder missing" and row["models"] == []


def test_same_folder_twice_is_one_row_and_ids_stay_unique(env):
    (env / "a").mkdir()
    (env / "b").mkdir()
    one = model_locations.add("Tunes", str(env / "a"))
    assert model_locations.add("Other name", str(env / "a")) == one
    two = model_locations.add("Tunes", str(env / "b"))
    assert two["id"] == "tunes-2"
    renamed = s.edit_model_location("tunes", s._LocationEditBody(name="Renamed"))
    assert renamed["id"] == "tunes" and renamed["name"] == "Renamed"
    with pytest.raises(HTTPException):
        s.edit_model_location("zzz", s._LocationEditBody(name="x"))


def test_old_checkpoint_root_is_imported_once(env):
    root = env / "old"
    _checkpoint(root / "ck")
    studio = env / "data" / "qwen_studio"
    studio.mkdir(parents=True)
    (studio / "settings.json").write_text(json.dumps({"checkpoint_root": str(root), "python": "p"}))
    rows = model_locations.load()
    assert [(r["name"], r["path"]) for r in rows] == [("Imported checkpoint folder", str(root))]
    # the old field is ignored from now on and a removed row does not come back
    assert registry.load_settings().get("checkpoint_root") is None
    assert registry.load_settings()["python"] == "p"
    model_locations.remove(rows[0]["id"])
    assert model_locations.load() == []


def test_checkpoint_root_on_the_settings_route_is_accepted_and_ignored(env):
    saved = registry.save_settings({"checkpoint_root": str(env), "warmup_runs": 1})
    assert "checkpoint_root" not in saved and saved["warmup_runs"] == 1
    assert model_locations.load() == []


def test_discovery_reads_every_row_env_and_hf_cache_with_a_location(env, monkeypatch):
    _checkpoint(env / "r1" / "deep" / "ep9")
    _checkpoint(env / "r2")  # a root that is itself a checkpoint
    _checkpoint(env / "hf" / "models--Qwen--Qwen3-TTS-12Hz-1.7B-Base" / "snapshots" / "abc", kind="base", speakers=())
    envck = _checkpoint(env / "envck")
    monkeypatch.setenv("OMNIVOICE_QWEN3_NARRATOR_CKPT", str(envck))
    one = model_locations.add("One", str(env / "r1"))
    two = model_locations.add("Two", str(env / "r2"))
    by_id = {m["id"]: m for m in registry.discover_models()}
    assert by_id["deep-ep9"]["location"] == one["id"]
    assert by_id["r2"]["location"] == two["id"] and by_id["r2"]["path"] == os.path.abspath(env / "r2")
    assert by_id["hf-qwen3-tts-12hz-1.7b-base"]["location"] == "huggingface_cache"
    assert by_id["env-envck"]["location"] == "environment"


def test_model_ids_do_not_change_when_the_root_moves_into_the_list(env):
    """The id was derived from the path under the root before the list existed; it must stay that."""
    _checkpoint(env / "ckpts" / "Run_2_EP09" / "checkpoints" / "checkpoint-epoch-9")
    model_locations.add("Fine-tunes", str(env / "ckpts"))
    assert "Run_2_EP09-checkpoints-checkpoint-epoch-9" in [m["id"] for m in registry.discover_models()]
