"""Qwen3-TTS export: record schema, portable paths, rejections, reference choice, ZIP, Colab helper.

Synthetic clips only; no network and none of Qwen's code is imported.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import zipfile

import numpy as np
import pytest
import soundfile as sf
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.dependencies import require_admin
from api.routers import fish_dataset
from core.browser_guard import reject_cross_site_get
from engines.fish_api import dataset, dataset_quality, qwen3_colab, qwen3_export

NARR = "a" * 32
SIDE = "b" * 32


@pytest.fixture(autouse=True)
def _store(monkeypatch, tmp_path):
    os.makedirs(tmp_path / "store", exist_ok=True)
    monkeypatch.setattr("engines.fish_api.story._store_dir", lambda: str(tmp_path / "store"))
    dataset_quality.reset_for_tests()
    yield
    dataset_quality.reset_for_tests()


def _tone(path, seconds, freq, amp=0.2, rate=44100, noise=0.0):
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    data = np.sin(2 * np.pi * freq * t) * amp
    if noise:
        data = data + np.random.default_rng(1).normal(0, noise, len(t))
    sf.write(path, np.clip(data, -1, 1).astype("float32"), rate, subtype="PCM_16")


def _run(root, chapter, started, rows, run="run-001_20261001-000000_aaaa"):
    """rows: (id, voice, name, text, seconds, kind) where kind picks how the audio is made."""
    folder = os.path.join(root, chapter, "untitled", run)
    os.makedirs(folder)
    chunks = []
    for i, (cid, vid, vname, text, secs, kind) in enumerate(rows, 1):
        name = f"{i:03d}_{vname}.wav"
        path = os.path.join(folder, name)
        if kind == "tone":
            _tone(path, secs, 200 + 10 * i)
        elif kind == "loud":
            _tone(path, secs, 200 + 10 * i, amp=0.7)
        elif kind == "clipped":
            _tone(path, secs, 200 + 10 * i, amp=3.0)
        elif kind == "silent":
            sf.write(path, np.zeros(int(44100 * secs), dtype="float32"), 44100, subtype="PCM_16")
        elif kind == "same":
            _tone(path, secs, 777)
        elif kind == "junk":
            with open(path, "wb") as fh:
                fh.write(b"not audio at all")
        chunks.append({"id": cid, "index": i, "status": "done", "file": name, "voice": vid, "voice_name": vname,
                       "duration_s": secs, "sample_rate": 44100, "sha256": f"sha{cid}{chapter}", "text": text})
    with open(os.path.join(folder, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({"id": run[-4:] + chapter[-2:], "status": "done", "started_at": started, "model": "s2.1-pro-free",
                   "sample_rate": 44100, "chunks": chunks}, fh)
    with open(os.path.join(folder, "story.json"), "w", encoding="utf-8") as fh:
        json.dump({"draft": {"chunks": [{"id": c[0], "text": c[3]} for c in rows]}}, fh)


@pytest.fixture
def root(tmp_path):
    r = str(tmp_path / "input")
    _run(r, "3001_one", 100, [
        ("a1", NARR, "Narrator", "[slow][calm] The road was long, and the night was [laugh]cold.", 9.0, "tone"),
        ("a2", NARR, "Narrator", "[shouting] Come out of there!", 8.0, "tone"),
        ("a3", NARR, "Narrator", "Rain fell on the old roof.", 7.0, "tone"),
        ("a4", NARR, "Narrator", "Too short.", 0.3, "tone"),
        ("a5", NARR, "Narrator", "[sigh]", 3.0, "tone"),
        ("a6", NARR, "Narrator", "Nothing but silence here.", 3.0, "silent"),
        ("a7", NARR, "Narrator", "This file cannot be decoded.", 3.0, "junk"),
        ("a8", NARR, "Narrator", "A repeated sound, first take.", 3.0, "same"),
        ("a9", NARR, "Narrator", "A repeated sound, other words.", 3.0, "same"),
        ("b1", SIDE, "Noctis", "I speak for the other voice.", 4.0, "tone"),
    ])
    _run(r, "3002_two", 200, [
        ("c1", NARR, "Narrator", "A clipped line, sadly.", 8.0, "clipped"),
        ("c2", NARR, "Narrator", "Another calm and ordinary sentence.", 10.0, "tone"),
        ("c3", NARR, "Narrator", "Rain fell on the old roof.", 6.0, "tone"),
    ], run="run-001_20261002-000000_bbbb")
    _run(r, "3199_loud", 300, [
        ("d1", NARR, "Narrator", "This chapter was rendered much louder.", 6.0, "loud"),
        ("d2", NARR, "Narrator", "So was this one, for the report.", 6.5, "loud"),
    ], run="run-001_20261003-000000_cccc")
    return r


@pytest.fixture
def corpus(root):
    return dataset.load(root, refresh=True)


def _export(corpus, tmp_path, name="qwen_set", spec=None, training=None, **options):
    spec = {"voice_ids": [NARR], **(spec or {})}
    return dataset.export(corpus, spec, str(tmp_path / "out"), formats=["qwen3_tts"], name=name,
                          qwen3={"speaker_name": "Sunny", **options}, training=training)


def _jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_record_schema_relative_paths_and_audio_format(corpus, tmp_path):
    result = _export(corpus, tmp_path)
    out = result["path"]
    rows = _jsonl(os.path.join(out, "train.jsonl"))
    assert len(rows) == result["clips"] > 0
    for n, row in enumerate(rows, 1):
        assert list(row) == ["audio", "text", "ref_audio"]  # exactly the fields Qwen's loader reads
        assert row["audio"] == f"audio/{n:06d}.wav" and row["ref_audio"] == "reference/ref.wav"
        assert os.path.isfile(os.path.join(out, *row["audio"].split("/")))
        info = sf.info(os.path.join(out, *row["audio"].split("/")))
        assert (info.samplerate, info.channels, info.subtype, info.format) == (24000, 1, "PCM_16", "WAV")
    ref = sf.info(os.path.join(out, "reference", "ref.wav"))
    assert (ref.samplerate, ref.channels, ref.subtype) == (24000, 1, "PCM_16")  # Qwen asserts 24 kHz
    assert {"dataset_report.json", "rejected_samples.jsonl", "README.txt", "colab_setup.sh",
            "colab_commands.txt"} <= set(os.listdir(out))


def test_no_absolute_or_windows_paths_anywhere_in_the_metadata(corpus, tmp_path):
    out = _export(corpus, tmp_path, make_zip=True)["path"]
    bad = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]|\\\\|(?<![\w.])/(?:home|Users|tmp|var|mnt)/")
    for name in ("train.jsonl", "dataset_report.json", "rejected_samples.jsonl"):
        text = open(os.path.join(out, name), encoding="utf-8").read()
        assert not bad.search(text), name
        assert str(tmp_path) not in text and "\\" not in text.replace("\\n", "").replace('\\"', "").replace("\\u", "")
    for row in _jsonl(os.path.join(out, "train.jsonl")):
        assert not os.path.isabs(row["audio"]) and ".." not in row["audio"]


def test_text_is_the_spoken_transcript_only(corpus, tmp_path):
    out = _export(corpus, tmp_path)["path"]
    texts = [r["text"] for r in _jsonl(os.path.join(out, "train.jsonl"))]
    assert "The road was long, and the night was cold." in texts  # tags gone, no space left before punctuation
    assert "Come out of there!" in texts
    assert not any(re.search(r"[\[\]]", t) for t in texts)
    assert qwen3_export.spoken_text("Hi [pause] there [laugh]!") == "Hi there!"


def test_mapping_is_deterministic_and_complete(corpus, tmp_path):
    first = _export(corpus, tmp_path, name="one")["path"]
    second = _export(corpus, tmp_path, name="two")["path"]
    a = json.load(open(os.path.join(first, "dataset_report.json"), encoding="utf-8"))["mapping"]
    b = json.load(open(os.path.join(second, "dataset_report.json"), encoding="utf-8"))["mapping"]
    assert a == b and [m["exported"] for m in a] == [f"audio/{i:06d}.wav" for i in range(1, len(a) + 1)]
    assert open(os.path.join(first, "train.jsonl"), "rb").read() == open(os.path.join(second, "train.jsonl"), "rb").read()
    assert all(set(m) >= {"original", "exported", "text", "duration_s"} for m in a)
    assert a[0]["original"].startswith("3001_one/")  # root-relative, never an absolute path


def test_every_clip_is_exported_or_recorded_as_rejected(corpus, tmp_path):
    result = _export(corpus, tmp_path)
    out = result["path"]
    report = json.load(open(os.path.join(out, "dataset_report.json"), encoding="utf-8"))
    rejected = _jsonl(os.path.join(out, "rejected_samples.jsonl"))
    reasons = {r["reason"] for r in rejected}
    assert {"too_short", "silent", "unreadable_audio", "duplicate_audio"} <= reasons
    assert all(set(r) >= {"source", "reason", "text"} for r in rejected)
    counts = report["counts"]
    assert counts["accepted"] + counts["rejected"] == counts["source_samples"]
    assert counts["accepted"] == result["clips"] and counts["rejected"] == len(rejected)
    assert report["validation"].startswith("None") and counts["validation"] == 0
    assert not os.path.exists(os.path.join(out, "validation.jsonl"))
    codes = {w["code"] for w in report["warnings"]}
    assert {"clipping", "duplicate_audio", "inconsistent_loudness", "not_indexed"} <= codes  # not_indexed: the tag-only line


def test_duration_limits_are_rejections_not_silent_drops(corpus, tmp_path):
    out = _export(corpus, tmp_path, min_duration=5.0, max_duration=8.5)["path"]
    rows = _jsonl(os.path.join(out, "rejected_samples.jsonl"))
    assert {r["reason"] for r in rows if r["duration"] and r["duration"] > 8.5} == {"too_long"}
    assert any(r["reason"] == "too_short" and r["duration"] < 5 for r in rows)


def test_loudness_matching_is_off_by_default_and_silences_the_chapter_warning(corpus, tmp_path):
    plain = _export(corpus, tmp_path, name="plain")["path"]
    plain_report = json.load(open(os.path.join(plain, "dataset_report.json"), encoding="utf-8"))
    assert plain_report["loudness_matching"] is None and plain_report["chapter_levels_db"]["3199_loud"] > \
        plain_report["chapter_levels_db"]["3001_one"] + 6
    matched = dataset.export(corpus, {"voice_ids": [NARR]}, str(tmp_path / "out"), formats=["qwen3_tts"], name="matched",
                             loudness_db=-24.0, qwen3={"speaker_name": "Sunny"})
    rep = json.load(open(os.path.join(matched["path"], "dataset_report.json"), encoding="utf-8"))
    assert rep["loudness_matching"]["target_rms_db"] == -24.0
    assert "inconsistent_loudness" not in {w["code"] for w in rep["warnings"]}


def test_a_24k_mono_pcm16_source_is_copied_byte_for_byte(tmp_path):
    src = tmp_path / "s.wav"
    _tone(str(src), 2.0, 300, rate=24000)
    target = tmp_path / "t.wav"
    assert qwen3_export._convert(str(src), str(target), None)["copied"] is True
    assert target.read_bytes() == src.read_bytes()


def test_reference_is_chosen_automatically_and_avoids_clipped_expressive_and_short_clips(corpus, tmp_path):
    result = _export(corpus, tmp_path)
    ref = result["qwen3"]["reference"]
    assert ref["mode"] == "auto" and ref["reason"]
    assert ref["source"] not in {  # the clipped, shouted, and very short clips
        c["file"] for c in corpus.lines if c["tags"] == ["shouting"] or "clipped" in c["plain"]}
    assert 5 <= ref["duration"] <= 15
    assert sf.info(os.path.join(result["path"], "reference", "ref.wav")).samplerate == 24000
    report = json.load(open(os.path.join(result["path"], "dataset_report.json"), encoding="utf-8"))
    assert report["reference"]["path"] == "reference/ref.wav" and report["reference"]["exported_as"].startswith("audio/")


def test_reference_can_be_named_by_clip_or_file_and_must_be_exportable(corpus, tmp_path):
    key = next(l["key"] for l in corpus.lines if l["plain"].startswith("Rain fell") and l["chapter"] == "3002_two")
    got = _export(corpus, tmp_path, name="chosen", reference_key=key)["qwen3"]["reference"]
    assert got["mode"] == "clip" and got["source"].startswith("3002_two/") and got["reason"] == "chosen by you"
    bad = next(l["key"] for l in corpus.lines if l["plain"] == "Too short.")
    with pytest.raises(ValueError, match="too_short"):
        _export(corpus, tmp_path, name="bad", reference_key=bad)
    external = tmp_path / "mine.wav"
    _tone(str(external), 6.0, 250, rate=48000)
    mine = _export(corpus, tmp_path, name="file", reference_file=str(external))["qwen3"]["reference"]
    assert mine["mode"] == "file" and mine["source"] == "mine.wav"  # the name only, never the full path
    with pytest.raises(ValueError, match="not both"):
        qwen3_export.options_from({"reference_key": "x", "reference_file": "y"})


def test_several_speakers_are_refused_with_a_clear_message(corpus, tmp_path):
    with pytest.raises(ValueError, match="one speaker"):
        dataset.export(corpus, {}, str(tmp_path / "out"), formats=["qwen3_tts"], name="x")


def test_format_cannot_be_mixed_and_has_no_validation_split(corpus, tmp_path):
    base = dict(formats=["qwen3_tts", "jsonl"], name="x")
    with pytest.raises(ValueError, match="complete folder"):
        dataset.export(corpus, {"voice_ids": [NARR]}, str(tmp_path / "out"), **base)
    with pytest.raises(ValueError, match="no validation"):
        dataset.export(corpus, {"voice_ids": [NARR]}, str(tmp_path / "out"), formats=["qwen3_tts"], name="x",
                       val_fraction=0.05)
    with pytest.raises(ValueError, match="24000"):
        dataset.export(corpus, {"voice_ids": [NARR]}, str(tmp_path / "out"), formats=["qwen3_tts"], name="x",
                       sample_rate=16000)


@pytest.mark.parametrize("opts", [{"speaker_name": "bad name; rm -rf"}, {"min_duration": 9, "max_duration": 3}, {"nope": 1}])
def test_options_are_validated_because_they_reach_a_shell_script(opts):
    with pytest.raises(ValueError):
        qwen3_export.options_from(opts)


@pytest.mark.parametrize("bad", [{"drive_folder": "../x"}, {"drive_folder": "$(x)"}, {"drive_folder": "a b"},
                                 {"epochs": 0}, {"epochs": 2.5}, {"batch_size": 1000}, {"learning_rate": 1.0},
                                 {"base_model": "x; ls"}, {"tokenizer_model": "nope"}, {"seed": 1}])
def test_training_settings_accept_only_what_the_official_script_takes(bad):
    with pytest.raises(ValueError):
        qwen3_colab.settings_from(bad)
    ok = qwen3_colab.settings_from({"epochs": 5, "drive_folder": "A/b-c"})
    assert (ok.epochs, ok.batch_size, ok.learning_rate) == (5, 2, 2e-5)  # the script's own defaults


def test_the_estimate_follows_the_scripts_loop_not_an_audio_rate():
    est = qwen3_colab.estimate(qwen3_colab.settings_from({"batch_size": 4, "epochs": 2}), 100)
    assert est["batches_per_epoch"] == 25 and est["updates_per_epoch"] == 7 and est["total_updates"] == 14


def test_zip_keeps_the_hierarchy_and_unzips_to_a_working_dataset(corpus, tmp_path):
    result = _export(corpus, tmp_path, make_zip=True)
    archive = result["qwen3"]["zip"]["path"]
    assert archive == result["path"] + ".zip"
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        assert "train.jsonl" in names and "reference/ref.wav" in names and "audio/000001.wav" in names
        assert all("\\" not in n and not n.startswith("/") for n in names)
        dest = tmp_path / "unzipped"
        zf.extractall(dest)
    for row in _jsonl(str(dest / "train.jsonl")):
        assert (dest / row["audio"]).is_file() and (dest / row["ref_audio"]).is_file()
    shutil.rmtree(result["path"])
    again = _export(corpus, tmp_path, make_zip=True, overwrite=True)
    assert open(again["qwen3"]["zip"]["path"], "rb").read() == open(archive, "rb").read()  # same input, same bytes


def test_export_refuses_a_non_empty_folder_but_overwrite_replaces_only_a_previous_export(corpus, tmp_path):
    first = _export(corpus, tmp_path)["path"]
    with pytest.raises(ValueError, match="already has files"):
        _export(corpus, tmp_path)
    stale = os.path.join(first, "audio", "stale.txt")
    open(stale, "w").write("x")
    _export(corpus, tmp_path, overwrite=True)
    assert not os.path.exists(stale)
    foreign = tmp_path / "out" / "mine"
    foreign.mkdir(parents=True)
    (foreign / "keep.txt").write_text("precious")
    with pytest.raises(ValueError, match="only replaces a previous Qwen3-TTS export"):
        _export(corpus, tmp_path, name="mine", overwrite=True)
    assert (foreign / "keep.txt").read_text() == "precious"


def test_colab_helper_carries_the_verified_commands_and_the_drive_checkpoint_folder(corpus, tmp_path):
    out = _export(corpus, tmp_path, training={"drive_folder": "My-Training/run1", "epochs": 7, "learning_rate": 1e-5,
                                              "batch_size": 4})["path"]
    setup = open(os.path.join(out, "colab_setup.sh"), encoding="utf-8").read()
    commands = open(os.path.join(out, "colab_commands.txt"), encoding="utf-8").read()
    readme = open(os.path.join(out, "README.txt"), encoding="utf-8").read()
    for text in (setup, commands, readme):
        assert qwen3_colab.QWEN_COMMIT in text and "Qwen/Qwen3-TTS-12Hz-1.7B-Base" in text and "@@" not in text
    assert "finetuning/prepare_data.py" in setup and "finetuning/sft_12hz.py" in setup
    for flag in ("--tokenizer_model_path", "--input_jsonl", "--output_jsonl", "--init_model_path", "--output_model_path",
                 "--train_jsonl", "--batch_size", "--lr", "--num_epochs", "--speaker_name"):
        assert flag in setup
    assert "Qwen/Qwen3-TTS-Tokenizer-12Hz" in setup and 'DRIVE_DIR="${DRIVE_DIR:-/content/drive/MyDrive/My-Training/run1/${SPEAKER_NAME}}"' in setup
    # the settings chosen in the dialog reach every helper file, not the defaults
    assert 'BATCH_SIZE="${BATCH_SIZE:-4}"' in setup and 'LR="${LR:-1e-05}"' in setup and 'EPOCHS="${EPOCHS:-7}"' in setup
    assert "--batch_size 4 --lr 1e-05 --num_epochs 7" in commands and "--num_epochs 7" in readme
    assert "--num_epochs 3" not in commands + readme
    settings = json.load(open(os.path.join(out, "training_settings.json"), encoding="utf-8"))
    assert (settings["epochs"], settings["batch_size"], settings["learning_rate"]) == (7, 4, 1e-5)
    assert settings["checkpoint_dir"] == "/content/drive/MyDrive/My-Training/run1/Sunny/checkpoints"
    assert "--num_epochs 7" in settings["commands"]["sft_12hz"] and settings["fixed_in_the_official_script"]["gradient_accumulation_steps"] == 4
    assert settings["estimate"]["updates_per_epoch"] >= 1 and "Not a time or quality prediction" in settings["estimate"]["note"]
    assert 'OUTPUT_DIR="${OUTPUT_DIR:-${DRIVE_DIR}/checkpoints}"' in setup
    assert "/content/drive/MyDrive/My-Training/run1/Sunny/checkpoints" in commands
    assert 'SPEAKER_NAME="${SPEAKER_NAME:-Sunny}"' in setup
    for cmd in ("colab new", "colab drivemount", "colab exec", "colab ls", "colab download", "colab stop"):
        assert cmd in commands
    assert "no resume" in commands.lower() and "RESTART=1" in setup  # the honest statement and the overwrite guard
    assert "checkpoint-epoch-" in commands and "drive.mount" in commands
    bash = shutil.which("bash")
    if bash:
        ok = subprocess.run([bash, "-n", os.path.join(out, "colab_setup.sh")], capture_output=True, text=True)
        assert ok.returncode == 0, ok.stderr


def test_the_report_summarises_the_set(corpus, tmp_path):
    out = _export(corpus, tmp_path)["path"]
    report = json.load(open(os.path.join(out, "dataset_report.json"), encoding="utf-8"))
    assert report["audio"] == {**report["audio"], "sample_rate": 24000, "channels": 1, "format": "WAV PCM16"}
    assert report["speaker"]["qwen_speaker_name"] == "Sunny"
    assert report["qwen"]["commit"] == qwen3_colab.QWEN_COMMIT
    d = report["duration"]
    assert d["shortest_s"] <= d["median_s"] <= d["longest_s"] and d["total_s"] > 0
    assert "Check Fish's terms" in report["source"]


def test_nothing_exportable_leaves_no_folder_behind(tmp_path):
    r = str(tmp_path / "input")
    _run(r, "3001_one", 100, [("z1", NARR, "Narrator", "Quiet.", 3.0, "silent")])
    with pytest.raises(ValueError, match="No clip passed"):
        _export(dataset.load(r, refresh=True), tmp_path)


# ---- API and CLI ----
@pytest.fixture
def client(root, monkeypatch):
    monkeypatch.setattr(dataset, "get_root", lambda: root)
    app = FastAPI()
    app.dependency_overrides[require_admin] = lambda: None
    app.dependency_overrides[reject_cross_site_get] = lambda: None
    app.include_router(fish_dataset.router)
    return TestClient(app)


def test_the_export_route_takes_the_qwen3_options(client, tmp_path):
    body = {"dest": str(tmp_path / "api"), "name": "via_api", "formats": ["qwen3_tts"], "voice_ids": [NARR],
            "qwen3": {"speaker_name": "Sunny", "make_zip": True, "min_duration": 1.0},
            "training": {"epochs": 6, "drive_folder": "Drive-A"}}
    res = client.post("/api/fish/dataset/export", json=body)
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["formats"] == ["qwen3_tts"] and data["qwen3"]["zip"]["path"].endswith("via_api.zip")
    assert data["qwen3"]["reference"]["reason"] and "colab_setup.sh" in data["qwen3"]["files"]
    assert data["qwen3"]["settings"]["epochs"] == 6 and "--num_epochs 6" in data["qwen3"]["commands"]["official"]["train"]
    assert any("colab_setup.sh" in c for c in data["qwen3"]["commands"]["notebook"]) and data["qwen3"]["commands"]["colab_cli"]
    bad = client.post("/api/fish/dataset/export", json={**body, "name": "x2", "voice_ids": []})
    assert bad.status_code == 422 and "one speaker" in bad.json()["detail"]
    mixed = client.post("/api/fish/dataset/export", json={**body, "name": "x3", "formats": ["qwen3_tts", "jsonl"]})
    assert mixed.status_code == 422


def _cli():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "export_qwen3_tts.py")
    spec = importlib.util.spec_from_file_location("export_qwen3_tts", os.path.abspath(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_calls_the_same_exporter(root, tmp_path, capsys):
    cli = _cli()
    out = tmp_path / "cli_out"
    code = cli.run(["--input", root, "--output", str(out), "--voice", "Narrator", "--speaker-name", "Sunny", "--zip",
                    "--min-duration", "1"])
    assert code == 0
    assert (out / "train.jsonl").is_file() and (tmp_path / "cli_out.zip").is_file()
    printed = capsys.readouterr().out
    assert "Reference:" in printed and "WARNING [inconsistent_loudness]" in printed
    assert cli.run(["--input", root, "--output", str(out), "--voice", "Narrator"]) == 2  # exists, no --overwrite
    assert cli.run(["--input", root, "--output", str(tmp_path / "o2")]) == 2  # two speakers, none chosen
    assert cli.run(["--input", root, "--output", str(tmp_path / "o3"), "--voice", NARR, "--validation-ratio", "0.05"]) == 2
