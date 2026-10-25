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
    assert (ok.epochs, ok.batch_size, ok.learning_rate) == (5, 4, 1e-6)  # the recipe that gave the good run: batch 4 (x accumulation 2 = 8 clips per update), lr 1e-6


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


def test_colab_helper_files_are_thin_callers_of_one_runner_and_carry_the_dialog_settings(corpus, tmp_path):
    out = _export(corpus, tmp_path, training={"drive_folder": "My-Training/run1", "epochs": 7, "learning_rate": 1e-5,
                                              "batch_size": 4})["path"]
    read = lambda n: open(os.path.join(out, n), encoding="utf-8").read()  # noqa: E731
    setup, commands, readme = read("colab_setup.sh"), read("colab_commands.txt"), read("README.txt")
    for text in (commands, readme):
        assert qwen3_colab.QWEN_COMMIT in text and "Qwen/Qwen3-TTS-12Hz-1.7B-Base" in text
    for text in (setup, commands, readme):
        assert "@@" not in text
    # one implementation: the shell file only calls the runner, and the runner shipped in the export is the repo's file
    assert "voicestudio_qwen3_colab.py" in setup and "pip install" not in setup and len(setup.splitlines()) < 20
    assert read("voicestudio_qwen3_colab.py") == qwen3_colab.runner_source()
    assert os.path.isfile(os.path.join(out, qwen3_colab.NOTEBOOK_NAME))
    # the official commands and flags are spelled out for reading
    for flag in ("--tokenizer_model_path", "--input_jsonl", "--output_jsonl", "--init_model_path", "--output_model_path",
                 "--train_jsonl", "--batch_size", "--lr", "--num_epochs", "--speaker_name"):
        assert flag in commands
    assert "--batch_size 4 --lr 1e-05 --num_epochs 7" in commands and "--num_epochs 7" in readme
    assert "--num_epochs 3" not in commands + readme
    # the dialog values reach settings, the notebook parameters and the colab CLI command
    settings = json.load(open(os.path.join(out, "training_settings.json"), encoding="utf-8"))
    assert (settings["epochs"], settings["batch_size"], settings["learning_rate"]) == (7, 4, 1e-5)
    assert settings["drive_folder"] == "My-Training/run1" and settings["speaker_name"] == "Sunny"
    assert "--num_epochs 7" in settings["commands"]["sft_12hz"] and settings["fixed_in_the_official_script"]["gradient_accumulation_steps"] == 4
    assert settings["estimate"]["updates_per_epoch"] >= 1 and "Not a time or quality prediction" in settings["estimate"]["note"]
    assert "/content/drive/MyDrive/My-Training/run1/runs/Sunny/<run_id>/checkpoints" in commands
    for cmd in ("colab new", "colab drivemount", "colab exec", "colab ls", "colab download", "colab stop"):
        assert cmd in commands
    assert "--epochs 7" in commands and "--batch-size 4" in commands and "--speaker-name Sunny" in commands
    assert "no resume" in commands.lower() and "--restart" in commands and "runtime.unassign()" in commands
    assert "checkpoint-epoch-" in commands and "drive.mount" in commands
    bash = shutil.which("bash")
    if bash and subprocess.run([bash, "-c", "true"], capture_output=True).returncode == 0:  # a broken WSL launcher is not a bash
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
    assert any("voicestudio_qwen3_colab.py" in c for c in data["qwen3"]["commands"]["notebook"]) and data["qwen3"]["commands"]["colab_cli"]
    assert data["qwen3"]["commands"]["notebook_file"] == "Qwen3_TTS_Colab_Training.ipynb" and data["qwen3"]["commands"]["drive_layout"]
    assert "voicestudio_qwen3_colab.py" in data["qwen3"]["files"] and "Qwen3_TTS_Colab_Training.ipynb" in data["qwen3"]["files"]
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
    assert "Reference:" in printed and "Level matching to -22 dB" in printed
    assert "WARNING [inconsistent_loudness]" not in printed  # matched by default
    raw = tmp_path / "cli_raw"
    assert cli.run(["--input", root, "--output", str(raw), "--voice", "Narrator", "--no-loudness-match", "--min-duration", "1"]) == 0
    assert "WARNING [inconsistent_loudness]" in capsys.readouterr().out
    assert json.load(open(raw / "dataset_report.json", encoding="utf-8"))["loudness_matching"] is None
    other = tmp_path / "cli_m18"
    assert cli.run(["--input", root, "--output", str(other), "--voice", "Narrator", "--loudness-db", "-18", "--min-duration", "1"]) == 0
    assert json.load(open(other / "dataset_report.json", encoding="utf-8"))["loudness_matching"]["target_rms_db"] == -18.0
    assert cli.run(["--input", root, "--output", str(out), "--voice", "Narrator"]) == 2  # exists, no --overwrite
    assert cli.run(["--input", root, "--output", str(tmp_path / "o2")]) == 2  # two speakers, none chosen
    assert cli.run(["--input", root, "--output", str(tmp_path / "o3"), "--voice", NARR, "--validation-ratio", "0.05"]) == 2


# ---- loudness default behaviour, per-chapter report ----
def _spiky(path, seconds, freq, rate=44100):
    """Quiet tone with one loud click: its peak stops the gain long before -22 dB."""
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    data = np.sin(2 * np.pi * freq * t) * 0.02
    data[len(t) // 2] = 0.9
    sf.write(path, data.astype("float32"), rate, subtype="PCM_16")


def _wavs(root, chapter):
    import glob

    return sorted(glob.glob(os.path.join(root, chapter, "*", "run-*", "*.wav")))


def _level_set(tmp_path):
    r = str(tmp_path / "levels")
    _run(r, "3001_quiet", 100, [(f"q{i}", NARR, "Narrator", f"Quiet line number {i}.", 6.0, "tone") for i in range(4)])
    for path in _wavs(r, "3001_quiet"):  # make this chapter quiet (about -25 dB)
        data, rate = sf.read(path, dtype="float32")
        sf.write(path, data * 0.45, rate, subtype="PCM_16")
    _run(r, "3193_hot", 200, [(f"h{i}", NARR, "Narrator", f"Hot line number {i}.", 6.0, "loud") for i in range(4)],
         run="run-001_20261002-000000_bbbb")
    return dataset.load(r, refresh=True)


def test_every_exported_clip_lands_within_a_decibel_of_the_target_and_the_report_says_so(tmp_path):
    corpus = _level_set(tmp_path)
    result = dataset.export(corpus, {"voice_ids": [NARR]}, str(tmp_path / "out"), formats=["qwen3_tts"], name="lv",
                            loudness_db=-22.0, qwen3={"speaker_name": "Sunny", "trailing_silence_s": 0})  # the level is measured on the whole file
    out = result["path"]
    measured = [dataset_quality.measure(os.path.join(out, "audio", f"{n:06d}.wav")) for n in range(1, result["clips"] + 1)]
    assert len(measured) == 8
    assert max(m["rms_db"] for m in measured) - min(m["rms_db"] for m in measured) < 1.0
    assert all(abs(m["rms_db"] + 22.0) < 0.6 for m in measured)
    rep = json.load(open(os.path.join(out, "dataset_report.json"), encoding="utf-8"))
    lm = rep["loudness_matching"]
    assert lm["target_rms_db"] == -22.0 and lm["adjusted"] == 8 and lm["peak_limited"] == 0
    assert lm["chapters"]["3193_hot"]["median_before_db"] > lm["chapters"]["3001_quiet"]["median_before_db"] + 6
    for ch in lm["chapters"].values():
        assert abs(ch["median_after_db"] + 22.0) < 0.6 and ch["worst_shortfall_db"] == 0
    assert "inconsistent_loudness" not in {w["code"] for w in rep["warnings"]}
    assert result["loudness"] == lm and all("level_after_db" in m for m in rep["mapping"])
    raw = dataset.export(corpus, {"voice_ids": [NARR]}, str(tmp_path / "out"), formats=["qwen3_tts"], name="raw",
                         qwen3={"speaker_name": "Sunny"})
    assert raw["loudness"] is None
    assert "inconsistent_loudness" in {w["code"] for w in raw["qwen3"]["warnings"]}


def test_clips_the_peak_cap_stops_are_counted_with_their_shortfall_and_warned_about(tmp_path):
    r = str(tmp_path / "spiky")
    _run(r, "3001_a", 100, [(f"s{i}", NARR, "Narrator", f"Spiky line number {i}.", 6.0, "tone") for i in range(4)])
    for i, path in enumerate(_wavs(r, "3001_a")):
        _spiky(path, 6.0, 200 + 25 * i)
    result = dataset.export(dataset.load(r, refresh=True), {"voice_ids": [NARR]}, str(tmp_path / "out"),
                            formats=["qwen3_tts"], name="sp", loudness_db=-22.0, qwen3={"speaker_name": "Sunny"})
    lm = result["loudness"]
    assert lm["peak_limited"] == 4 and lm["worst_shortfall_db"] > 5
    assert lm["chapters"]["3001_a"]["peak_limited"] == 4
    assert "loudness_peak_limited" in {w["code"] for w in result["qwen3"]["warnings"]}


# ---- reference ranking and the endpoint ----
def _measure_all(corpus):
    for line in corpus.lines:
        try:
            dataset_quality._stats[line["sha256"]] = dataset_quality.measure(os.path.join(corpus.root, line["file"]))
        except Exception:  # the undecodable fixture clip
            pass
    dataset_quality._loaded = True


def test_the_ranking_is_deterministic_and_the_exporter_takes_its_first_entry(corpus, tmp_path):
    lines, _, _ = qwen3_export.speaker_lines(corpus, {"voice_ids": [NARR]})
    _measure_all(corpus)
    clips, unmeasured = qwen3_export.candidate_clips(lines)
    assert unmeasured == 1  # the undecodable fixture clip
    first = qwen3_export.rank_references(clips)
    again = qwen3_export.rank_references(list(reversed(clips)))
    assert [r["clip"]["number"] for r in first] == [r["clip"]["number"] for r in again]
    assert [r["tier"] for r in first] == sorted(r["tier"] for r in first)
    assert "clipped" in {c for r in first if r["clip"]["text"].startswith("A clipped") for c in r["cautions"]}
    assert first[-1]["tier"] >= first[0]["tier"] and len(qwen3_export.rank_references(clips, 2)) == 2
    exported = _export(corpus, tmp_path)["qwen3"]["reference"]
    assert exported["source"] == first[0]["clip"]["line"]["file"]


def test_unmeasured_clips_are_counted_not_guessed(corpus):
    lines, _, _ = qwen3_export.speaker_lines(corpus, {"voice_ids": [NARR]})
    clips, unmeasured = qwen3_export.candidate_clips(lines)
    assert clips == [] and unmeasured > 0


def test_the_candidates_route_ranks_the_current_selection(client, corpus):
    _measure_all(corpus)
    body = {"voice_ids": [NARR], "limit": 3}
    res = client.post("/api/fish/dataset/reference-candidates", json=body)
    assert res.status_code == 200, res.text
    data = res.json()
    assert len(data["candidates"]) == 3 and data["unmeasured"] == 1 and data["speaker"]["name"] == "Narrator"
    top = data["candidates"][0]
    assert {"key", "text", "duration_s", "stats", "score", "reasons", "cautions"} <= set(top)
    assert client.post("/api/fish/dataset/reference-candidates", json=body).json() == data  # same every time
    only = client.post("/api/fish/dataset/reference-candidates", json={**body, "only_keys": [top["key"]]}).json()
    assert [c["key"] for c in only["candidates"]] == [top["key"]]
    assert client.post("/api/fish/dataset/reference-candidates", json={"limit": 3}).status_code == 422  # two speakers
    assert client.post("/api/fish/dataset/reference-candidates", json={**body, "q": "zzzz"}).status_code == 422


def test_the_candidates_route_reports_unmeasured_clips(client):
    data = client.post("/api/fish/dataset/reference-candidates", json={"voice_ids": [NARR]}).json()
    assert data["candidates"] == [] and data["unmeasured"] > 0


def test_a_reference_from_another_speaker_or_outside_the_corpus_is_refused(corpus, tmp_path):
    other = next(l["key"] for l in corpus.lines if l["voice_id"] == SIDE)
    with pytest.raises(ValueError, match="Noctis.*one speaker"):
        _export(corpus, tmp_path, name="other", reference_key=other)
    with pytest.raises(ValueError, match="not in the corpus"):
        _export(corpus, tmp_path, name="ghost", reference_key="nope/none")
    assert not (tmp_path / "out" / "other").exists()


def test_exactly_one_reference_is_written_and_it_is_24k_mono(corpus, tmp_path):
    key = next(l["key"] for l in corpus.lines if l["plain"].startswith("Another calm"))
    out = _export(corpus, tmp_path, name="one", reference_key=key)["path"]
    assert os.listdir(os.path.join(out, "reference")) == ["ref.wav"]
    info = sf.info(os.path.join(out, "reference", "ref.wav"))
    assert (info.samplerate, info.channels) == (24000, 1)
    assert {r["ref_audio"] for r in _jsonl(os.path.join(out, "train.jsonl"))} == {"reference/ref.wav"}


def test_facets_carry_the_level_spread_of_the_selection(client, corpus):
    res = client.post("/api/fish/dataset/facets", json={"voice_ids": [NARR]}).json()
    assert res["levels"]["total"] > 0 and res["levels"]["measured"] == 0 and res["levels"]["median"] is None
    _measure_all(corpus)
    res = client.post("/api/fish/dataset/facets", json={"voice_ids": [NARR]}).json()
    lv = res["levels"]
    assert lv["measured"] == lv["total"] - 1 and lv["min"] <= lv["median"] <= lv["max"]
    assert sum(n for _, n in lv["counts"]) == lv["measured"]


# ------------------------------------------------------------------------------------------------ trailing silence helper
def test_pad_qwen3_dataset_adds_silence_to_clips_only_and_never_touches_the_original(tmp_path):
    import importlib.util
    import wave
    import zipfile

    spec = importlib.util.spec_from_file_location("pad_qwen3_dataset", os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "pad_qwen3_dataset.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    src = tmp_path / "export"
    (src / "audio").mkdir(parents=True)
    (src / "reference").mkdir()

    def make(path, seconds):
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1), w.setsampwidth(2), w.setframerate(24000)
            w.writeframes(b"\x01\x01" * int(24000 * seconds))

    make(src / "audio" / "000001.wav", 2.0)
    make(src / "audio" / "000002.wav", 3.5)
    make(src / "reference" / "ref.wav", 5.0)
    (src / "train.jsonl").write_text('{"audio": "audio/000001.wav", "text": "x", "ref_audio": "reference/ref.wav"}\n')
    (src / "README.txt").write_text("names the old zip")
    (src / "voicestudio_qwen3_colab.py").write_text("old runner")
    before = (src / "audio" / "000001.wav").read_bytes()
    out = mod.pad_dataset(str(src), str(tmp_path / "export_sil"), 1.0, str(tmp_path / "export_sil.zip"))
    assert out["clips"] == 2 and out["zip"]["path"].endswith("export_sil.zip")
    with wave.open(str(tmp_path / "export_sil" / "audio" / "000002.wav")) as w:
        assert abs(w.getnframes() / 24000 - 4.5) < 1e-6  # 3.5 s + exactly 1.000 s
    with wave.open(str(tmp_path / "export_sil" / "reference" / "ref.wav")) as w:
        assert abs(w.getnframes() / 24000 - 5.0) < 1e-6  # the reference clip is not padded
    assert (src / "audio" / "000001.wav").read_bytes() == before  # the original is untouched
    names = zipfile.ZipFile(tmp_path / "export_sil.zip").namelist()
    assert "train.jsonl" in names and "audio/000001.wav" in names and "reference/ref.wav" in names
    assert "README.txt" not in names and "voicestudio_qwen3_colab.py" not in names  # stale helpers stay out
    with pytest.raises(ValueError):
        mod.pad_dataset(str(src), str(src))
    with pytest.raises(FileNotFoundError):
        mod.pad_dataset(str(tmp_path / "nothing"), str(tmp_path / "x"))


# ------------------------------------------------------------------------------------------------ trailing silence, legacy folder, notebook
def test_the_export_appends_one_second_of_silence_to_training_clips_but_not_to_the_reference(tmp_path):
    import soundfile as sf

    corpus = _level_set(tmp_path)
    on = dataset.export(corpus, {"voice_ids": [NARR]}, str(tmp_path / "on"), formats=["qwen3_tts"], name="on", loudness_db=-22.0,
                        qwen3={"speaker_name": "Sunny"})
    off = dataset.export(corpus, {"voice_ids": [NARR]}, str(tmp_path / "off"), formats=["qwen3_tts"], name="off", loudness_db=-22.0,
                         qwen3={"speaker_name": "Sunny", "trailing_silence_s": 0})
    for n in range(1, on["clips"] + 1):
        a, ra = sf.read(os.path.join(on["path"], "audio", f"{n:06d}.wav"), dtype="int16")
        b, rb = sf.read(os.path.join(off["path"], "audio", f"{n:06d}.wav"), dtype="int16")
        assert ra == rb == 24000 and len(a) == len(b) + 24000  # exactly 1.000 s longer
        assert not a[-24000:].any() and (a[:len(b)] == b).all()  # the added second is digital silence, the speech is untouched
    ref_on, _ = sf.read(os.path.join(on["path"], "reference", "ref.wav"), dtype="int16")
    ref_off, _ = sf.read(os.path.join(off["path"], "reference", "ref.wav"), dtype="int16")
    assert len(ref_on) == len(ref_off)  # the reference clip is never padded
    rep = json.load(open(os.path.join(on["path"], "dataset_report.json"), encoding="utf-8"))
    assert rep["trailing_silence_s"] == 1.0
    for bad in (-1, 9, "x"):
        with pytest.raises((ValueError, TypeError)):
            qwen3_export.options_from({"trailing_silence_s": bad})


def test_a_dialog_that_still_holds_the_old_default_drive_folder_cannot_reach_a_notebook():
    assert qwen3_colab.settings_from({"drive_folder": "VoiceStudio-training"}).drive_folder == "Qwen3TTS"
    assert qwen3_colab.settings_from({"drive_folder": "MyOwnFolder"}).drive_folder == "MyOwnFolder"  # a real choice is kept
    assert qwen3_colab.settings_from({}).drive_folder == "Qwen3TTS"


def test_the_notebook_an_export_writes_needs_no_hand_edits(tmp_path):
    settings = qwen3_colab.settings_from({"epochs": 10, "learning_rate": 1e-6, "batch_size": 4})
    files = qwen3_colab.render_files(speaker="SS_Narrator", folder_name="narrator_3h", zip_name="narrator_3h.zip", settings=settings,
                                     report={"counts": {"source_samples": 3, "accepted": 3, "rejected": 0, "train": 3},
                                             "duration": {"total": "0:01", "average_s": 1, "median_s": 1, "shortest_s": 1, "longest_s": 1},
                                             "speaker": {"name": "SS_Narrator", "qwen_speaker_name": "SS_Narrator"},
                                             "reference": {"path": "reference/ref.wav", "source": "x", "duration": 5, "reason": "x"},
                                             "warnings": [], "trailing_silence_s": 1.0})
    nb = files[qwen3_colab.NOTEBOOK_NAME]
    assert "VoiceStudio-training" not in "".join(files.values())
    assert "GRAD_ACCUM = 2" in nb and "BATCH_SIZE = 4" in nb and "EPOCHS = 10" in nb  # 4 x 2 = 8 clips per update
    assert "flash-attention-prebuild-wheels" in nb and "narrator_3h.zip" in nb
    assert qwen3_colab.accum_for(2) == 4 and qwen3_colab.accum_for(4) == 2 and qwen3_colab.accum_for(8) == 1 and qwen3_colab.accum_for(16) == 1
    assert "silence appended to every training clip" in files["README.txt"]


def test_a_backend_running_older_code_than_the_files_on_disk_refuses_to_export(monkeypatch):
    qwen3_colab.assert_backend_is_current()  # in sync: fine
    monkeypatch.setattr(qwen3_colab.runner, "RUNNER_VERSION", "1")
    settings = qwen3_colab.settings_from({})
    with pytest.raises(ValueError, match="Restart the app"):
        qwen3_colab.render_files(speaker="S", folder_name="f", zip_name="f.zip", settings=settings, report={})
