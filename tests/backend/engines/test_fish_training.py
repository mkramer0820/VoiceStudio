"""Fish training runs: export validation, configs, hardware gate, dry run, run lifecycle.

No GPU, no model, no network: the process launcher is replaced by a fake that creates the files a
real tokenizer/trainer would, so the whole lifecycle (progress, cancel, resume, interrupted runs,
adapter + evaluation record) is exercised without starting a training run.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pytest
import soundfile as sf

from engines.fish_api import dataset, dataset_quality, training, training_eval

NARR = "a" * 32
GOOD_PROBE = {"accelerate": True, "webdataset": True, "safetensors": True, "tensorboardX": True, "triton": True, "torch": "2.8.0", "cuda": True, "mps": False,
              "device": "NVIDIA GeForce RTX 3070", "vram_gb": 8.0, "bf16": True, "omnivoice": True}


@pytest.fixture(autouse=True)
def _store(monkeypatch, tmp_path):
    os.makedirs(tmp_path / "store", exist_ok=True)
    monkeypatch.setattr("engines.fish_api.story._store_dir", lambda: str(tmp_path / "store"))
    monkeypatch.setattr(training, "_store_dir", lambda: str(tmp_path / "store"))
    dataset_quality.reset_for_tests()
    with training._jobs_lock:
        training._jobs.clear()
    yield
    training.set_launcher(None)
    dataset_quality.reset_for_tests()


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(training, "is_enabled", lambda: True)


def _make_export(tmp_path, clips=24, val=0.25, loudness=-24.0, name="ov"):
    root = str(tmp_path / "input")
    folder = os.path.join(root, "3001_one", "untitled", "run-001_20261001-000000_aaaa")
    os.makedirs(folder)
    chunks = []
    for i in range(clips):
        wav = f"{i + 1:03d}_Narrator.wav"
        t = np.linspace(0, 440 + i, 22050)
        sf.write(os.path.join(folder, wav), (np.sin(t) * (0.05 + 0.01 * (i % 5))).astype("float32"), 44100, subtype="PCM_16")
        chunks.append({"id": f"c{i}", "index": i + 1, "status": "done", "file": wav, "voice": NARR,
                       "voice_name": "Narrator", "duration_s": 0.5, "sample_rate": 44100, "sha256": f"sha{i}",
                       "text": f"[slow] Line number {i} of the story."})
    json.dump({"id": "aaaa", "status": "done", "started_at": 1, "model": "s2.1-pro-free", "sample_rate": 44100,
               "chunks": chunks}, open(os.path.join(folder, "manifest.json"), "w"))
    json.dump({"draft": {"chunks": [{"id": c["id"], "text": c["text"]} for c in chunks]}},
              open(os.path.join(folder, "story.json"), "w"))
    out = dataset.export(dataset.load(root), {}, str(tmp_path / "exports"), formats=["omnivoice"],
                         val_fraction=val, loudness_db=loudness, name=name)
    return out["path"]


class FakeHandle:
    def __init__(self, code=0, hang=False):
        self.code, self.hang, self.terminated, self.killed = code, hang, False, False

    def poll(self):
        return None if self.hang and not self.terminated else (self.code if not self.terminated else -15)

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return self.poll()


class FakeLauncher:
    def __init__(self, probe=None, train_code=0, tokenize_code=0, hang_train=False, write_adapter=True):
        self._probe = GOOD_PROBE if probe is None else probe
        self.train_code, self.tokenize_code, self.hang_train, self.write_adapter = train_code, tokenize_code, hang_train, write_adapter
        self.calls: list[tuple[list[str], dict]] = []
        self.probes = 0
        self.handles: list[FakeHandle] = []

    def probe(self, python, env=None):
        self.probes += 1
        return self._probe

    @staticmethod
    def _arg(argv, flag):
        return argv[argv.index(flag) + 1]

    def spawn(self, argv, *, cwd, env, log_path):
        self.calls.append((argv, env))
        with open(log_path, "ab") as log:
            if argv[2] == "omnivoice.scripts.extract_audio_tokens":
                pattern = self._arg(argv, "--tar_output_pattern")
                base = os.path.dirname(os.path.dirname(pattern))
                os.makedirs(base, exist_ok=True)
                if self.tokenize_code == 0:
                    open(os.path.join(base, "data.lst"), "w").write("a.tar a.jsonl 10 5.0\n")
                log.write(b"tokenizer output\n")
                handle = FakeHandle(self.tokenize_code)
            else:
                out = self._arg(argv, "--output_dir")
                os.makedirs(out, exist_ok=True)
                log.write(b"Step 25 | train/loss: 3.2500 | train/learning_rate: 1.0e-04\r\n")
                log.write(b"Eval Loss: 3.1000\n")
                log.write(b"Step 50 | train/loss: 2.5000 | train/learning_rate: 5.0e-05\n")
                if self.write_adapter and self.train_code == 0:
                    with open(os.path.join(out, "eval_history.jsonl"), "w") as fh:
                        fh.write(json.dumps({"step": 0, "eval_loss": 3.4, "batches": 4}) + "\n")
                        fh.write(json.dumps({"step": 50, "eval_loss": 3.0, "batches": 4}) + "\n")
                    ck = os.path.join(out, "checkpoint-50")
                    os.makedirs(os.path.join(ck, "adapter"), exist_ok=True)
                    open(os.path.join(ck, "adapter", "adapter_config.json"), "w").write("{}")
                    open(os.path.join(ck, "adapter", "adapter_model.safetensors"), "wb").write(b"x")
                    open(os.path.join(ck, "train_config.json"), "w").write("{}")
                handle = FakeHandle(self.train_code, hang=self.hang_train)
        self.handles.append(handle)
        return handle


def _wait(job_id, states, timeout=20.0):
    end = time.time() + timeout
    while time.time() < end:
        job = training.get_run(job_id)
        if job and job["status"] in states:
            return job
        time.sleep(0.05)
    raise AssertionError(f"run never reached {states}: {training.get_run(job_id)}")


# ── export validation ──
def test_a_good_export_validates_and_reports_its_size(tmp_path):
    check = training.validate_export(_make_export(tmp_path))
    assert check["ok"] and check["train_clips"] + check["val_clips"] == 24 and check["val_clips"] > 0
    assert check["loudness_matched"] is True and check["seconds"] > 0
    assert any("small set" in w for w in check["warnings"])  # 24 clips is small
    assert not any("Loudness was not matched" in w for w in check["warnings"])


def test_an_export_without_loudness_matching_or_validation_gets_warnings(tmp_path):
    check = training.validate_export(_make_export(tmp_path, val=0.0, loudness=None))
    assert check["ok"] and check["val_clips"] == 0
    assert any("Loudness was not matched" in w for w in check["warnings"])
    assert any("No validation clips" in w for w in check["warnings"])


@pytest.mark.parametrize("path,expect", [("", "full path"), ("relative/dir", "full path"), ("Z:/nope/never", "does not exist")])
def test_bad_export_paths_are_explained(path, expect):
    check = training.validate_export(path)
    assert not check["ok"] and expect in check["errors"][0]


def test_a_folder_without_the_omnivoice_files_is_refused(tmp_path):
    (tmp_path / "empty").mkdir()
    check = training.validate_export(str(tmp_path / "empty"))
    assert not check["ok"] and "OmniVoice training files" in check["errors"][0]


def test_wrong_rate_missing_audio_and_duplicate_ids_are_errors(tmp_path):
    folder = _make_export(tmp_path)
    rows = [json.loads(l) for l in open(os.path.join(folder, training.TRAIN_JSONL), encoding="utf-8")]
    sf.write(rows[0]["audio_path"], np.zeros(1600, dtype="float32"), 16000)  # wrong rate
    os.remove(rows[1]["audio_path"])  # gone
    rows[2]["id"] = rows[3]["id"]  # duplicate id
    with open(os.path.join(folder, training.TRAIN_JSONL), "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    check = training.validate_export(folder)
    joined = " ".join(check["errors"])
    assert not check["ok"] and "24000 Hz mono" in joined and "audio file is missing" in joined and "appears twice" in joined


def test_too_few_clips_is_refused(tmp_path):
    check = training.validate_export(_make_export(tmp_path, clips=10, val=0.0))
    assert not check["ok"] and "at least 20" in " ".join(check["errors"])


# ── options, configs, commands ──
def test_options_are_validated_and_unknown_ones_rejected():
    assert training.TrainOptions.from_dict({}).steps == 2000
    assert training.TrainOptions.from_dict({"steps": 300}).steps == 300
    for bad in ({"steps": 1}, {"learning_rate": 5}, {"lora_r": 0}, {"nonsense": 1}, {"base_checkpoint": " "},
                {"python": "Z:/no/python.exe"}, {"steps": True}):
        with pytest.raises(ValueError):
            training.TrainOptions.from_dict(bad)


def test_every_written_config_key_is_a_real_trainingconfig_field():
    from dataclasses import fields

    from omnivoice.training.config import TrainingConfig

    known = {f.name for f in fields(TrainingConfig)}
    cfg = training.build_train_config(training.TrainOptions(), has_val=True, resume_from="x")
    assert set(cfg) <= known, set(cfg) - known
    assert cfg["lora_enabled"] is True and cfg["eval_at_start"] is True and cfg["resume_from_checkpoint"] == "x"
    assert training.build_train_config(training.TrainOptions(train_heads=True), has_val=False)["lora_train_modules"] == ["audio_heads"]


def test_commands_use_only_flags_the_scripts_define_and_data_config_points_at_the_tokens(tmp_path):
    import re

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    source = open(os.path.join(root, "omnivoice", "scripts", "extract_audio_tokens.py"), encoding="utf-8").read()
    defined = set(re.findall(r'add_argument\(\s*"(--[a-z_-]+)"', source))
    train_source = open(os.path.join(root, "omnivoice", "cli", "train.py"), encoding="utf-8").read()
    train_defined = set(re.findall(r'add_argument\(\s*"(--[a-z_-]+)"', train_source))
    opts = training.TrainOptions()
    cmds = training.build_commands(str(tmp_path / "run"), opts, train_jsonl="/x/t.jsonl", val_jsonl="/x/v.jsonl")
    for key in ("tokenize_train", "tokenize_dev"):
        used = {a for a in cmds[key] if a.startswith("--")}
        assert used <= defined, used - defined
    assert {"--train_config", "--data_config", "--output_dir"} <= train_defined
    assert {a for a in cmds["train"] if a.startswith("--") and a in train_defined} == {"--train_config", "--data_config", "--output_dir"}
    assert cmds["train"][2] == "accelerate.commands.launch" and "omnivoice.cli.train" in cmds["train"]
    data = training.build_data_config(str(tmp_path / "run"), has_val=True)
    assert data["train"][0]["manifest_path"][0].endswith(os.path.join("tokens", "train", "data.lst"))
    assert data["dev"][0]["manifest_path"][0].endswith(os.path.join("tokens", "dev", "data.lst"))
    assert "dev" not in training.build_data_config(str(tmp_path / "run"), has_val=False)


def test_children_run_offline_unless_downloads_are_allowed():
    assert training.child_env(training.TrainOptions())["HF_HUB_OFFLINE"] == "1"
    env = training.child_env(training.TrainOptions(allow_download=True))
    assert env.get("HF_HUB_OFFLINE") != "1" or os.environ.get("HF_HUB_OFFLINE") == "1"


# ── hardware ──
def test_hardware_verdicts_explain_what_is_missing():
    assert training.assess_hardware(GOOD_PROBE)["ok"]
    cpu = training.assess_hardware({**GOOD_PROBE, "cuda": False})
    assert not cpu["ok"] and "No GPU" in cpu["blockers"][0]
    mps = training.assess_hardware({**GOOD_PROBE, "cuda": False, "mps": True})
    assert "Apple Silicon" in mps["blockers"][0]
    small = training.assess_hardware({**GOOD_PROBE, "vram_gb": 4.0})
    assert "4.0 GB" in small["blockers"][0]
    tight = training.assess_hardware({**GOOD_PROBE, "vram_gb": 6.5})
    assert tight["ok"] and tight["warnings"]
    win = training.assess_hardware({**GOOD_PROBE, "triton": False}, system="win32")
    assert "triton-windows" in win["blockers"][0]
    lin = training.assess_hardware({**GOOD_PROBE, "triton": False}, system="linux")
    assert "triton-windows" not in lin["blockers"][0]
    gone = training.assess_hardware({**GOOD_PROBE, "webdataset": False, "tensorboardX": False})
    assert "webdataset" in gone["blockers"][0] and "tensorboard" in gone["blockers"][0]
    broken = training.assess_hardware({**GOOD_PROBE, "omnivoice": False, "omnivoice_error": "torchaudio"})
    assert "torchaudio" in broken["blockers"][0]
    assert "could not be started" in training.assess_hardware({"error": "FileNotFoundError"})["blockers"][0]


# ── dry run ──
def test_a_dry_run_validates_everything_and_starts_nothing(tmp_path):
    launcher = FakeLauncher()
    training.set_launcher(launcher)
    job = training.start_run(_make_export(tmp_path), {"steps": 100}, dry_run=True)  # no opt-in needed
    assert job["status"] == "dry_run" and job["ready"] is True and launcher.calls == [] and launcher.probes == 1
    report = json.load(open(os.path.join(job["run_dir"], training.DRY_FILE)))
    assert report["ready"] and set(report["commands"]) == {"tokenize_train", "tokenize_dev", "train"}
    assert report["train_config"]["steps"] == 100 and report["heldout_clips"] > 0
    assert os.path.isfile(os.path.join(job["run_dir"], "train_config.json"))
    assert os.path.isfile(os.path.join(job["run_dir"], "data_config.json"))
    held = [json.loads(l) for l in open(os.path.join(job["run_dir"], training_eval.HELDOUT_FILE), encoding="utf-8")]
    assert held and all(h["audio_path"] for h in held)


def test_a_dry_run_on_a_machine_that_cannot_train_says_so_but_still_succeeds(tmp_path):
    training.set_launcher(FakeLauncher(probe={**GOOD_PROBE, "cuda": False}))
    job = training.start_run(_make_export(tmp_path), dry_run=True)
    assert job["status"] == "dry_run" and job["ready"] is False and "cannot train" in job["message"]
    assert "No GPU" in job["blockers"][0]


def test_a_dry_run_refuses_a_bad_export_up_front(tmp_path):
    training.set_launcher(FakeLauncher())
    with pytest.raises(ValueError, match="does not exist"):
        training.start_run("Z:/never/there", dry_run=True)


# ── real runs (fake processes) ──
def test_a_real_run_needs_the_opt_in(tmp_path):
    training.set_launcher(FakeLauncher())
    with pytest.raises(PermissionError):
        training.start_run(_make_export(tmp_path))


def test_the_opt_in_setting_round_trips(monkeypatch):
    store: dict = {}
    monkeypatch.setattr("services.settings_store.get_text", lambda k, d=None: store.get(k, d))
    monkeypatch.setattr("services.settings_store.set_text", lambda k, v: store.__setitem__(k, v))
    monkeypatch.setattr("services.settings_store.clear_text", lambda k: store.pop(k, None))
    assert training.is_enabled() is False  # off by default
    training.set_enabled(False)
    assert training.is_enabled() is False
    training.set_enabled(True)
    assert training.is_enabled() is True
    training.set_enabled(False)
    assert training.is_enabled() is False


def test_a_run_tokenizes_trains_and_stores_the_adapter_with_an_honest_evaluation(tmp_path, enabled):
    launcher = FakeLauncher()
    training.set_launcher(launcher)
    started = training.start_run(_make_export(tmp_path), {"steps": 50})
    job = _wait(started["id"], {"done", "failed"})
    assert job["status"] == "done", job["message"]
    kinds = [c[0][2] for c in launcher.calls]
    assert kinds == ["omnivoice.scripts.extract_audio_tokens"] * 2 + ["accelerate.commands.launch"]
    assert job["step"] == 50 and job["loss"] == 2.5 and job["eval_loss"] == 3.1
    adapter = job["adapter"]
    assert os.path.isfile(os.path.join(adapter, "adapter_config.json"))
    record = training_eval.load_record(adapter)
    assert record["checks"][0]["check"] == "heldout_loss" and record["checks"][0]["status"] == "passed"
    assert record["verdict"] == "incomplete" and record["ship_ready"] is False
    assert {p["check"] for p in record["pending"]} == {"asr_wer", "speaker_similarity", "listening_test"}
    assert job["evaluation"]["ship_ready"] is False
    assert "not marked ready" in job["message"]
    card = json.load(open(os.path.join(adapter, "adapter.json")))
    assert card["target"] == "omnivoice" and card["base_checkpoint"] == "k2-fsa/OmniVoice"
    log = training.read_log(job["id"])
    assert "Step 25" in log and "tokenizer output" in log and "unload" in log.lower()
    rows = training.list_runs()
    assert rows[0]["id"] == job["id"] and rows[0]["verdict"] == "incomplete" and rows[0]["ship_ready"] is False
    assert all(c[1].get("HF_HUB_OFFLINE") == "1" for c in launcher.calls)


def test_a_machine_that_cannot_train_fails_the_run_before_any_process_starts(tmp_path, enabled):
    launcher = FakeLauncher(probe={"error": "FileNotFoundError"})
    training.set_launcher(launcher)
    job = _wait(training.start_run(_make_export(tmp_path))["id"], {"failed"})
    assert launcher.calls == [] and "could not be started" in job["message"]


def test_a_failed_training_process_is_reported_and_resumable(tmp_path, enabled):
    export = _make_export(tmp_path)
    launcher = FakeLauncher(train_code=1, write_adapter=False)
    training.set_launcher(launcher)
    job = _wait(training.start_run(export, {"steps": 50})["id"], {"failed"})
    assert "exit code 1" in job["message"] and "Resume" in job["message"]
    # fix the "crash" and resume: tokens already exist, so only training runs again
    fixed = FakeLauncher()
    training.set_launcher(fixed)
    resumed = training.resume_run(job["id"])
    assert resumed["resumes"] == 1
    done = _wait(job["id"], {"done"})
    assert [c[0][2] for c in fixed.calls] == ["accelerate.commands.launch"]
    assert done["status"] == "done"


def test_resume_continues_from_the_last_complete_checkpoint(tmp_path, enabled):
    export = _make_export(tmp_path)
    training.set_launcher(FakeLauncher(train_code=1, write_adapter=False))
    job = _wait(training.start_run(export, {"steps": 50})["id"], {"failed"})
    out = os.path.join(job["run_dir"], "output")
    os.makedirs(os.path.join(out, "checkpoint-30"))
    open(os.path.join(out, "checkpoint-30", "train_config.json"), "w").write("{}")
    os.makedirs(os.path.join(out, "checkpoint-40"))  # incomplete: no train_config.json
    training.set_launcher(FakeLauncher())
    training.resume_run(job["id"])
    _wait(job["id"], {"done"})
    cfg = json.load(open(os.path.join(job["run_dir"], "train_config.json")))
    assert cfg["resume_from_checkpoint"].endswith("checkpoint-30")


def test_cancel_kills_the_process_and_the_run_can_be_resumed(tmp_path, enabled):
    launcher = FakeLauncher(hang_train=True, write_adapter=False)
    training.set_launcher(launcher)
    started = training.start_run(_make_export(tmp_path), {"steps": 50})
    end = time.time() + 20
    while len(launcher.calls) < 3 and time.time() < end:
        time.sleep(0.05)
    assert training.cancel_run(started["id"]) is True
    job = _wait(started["id"], {"cancelled"})
    assert launcher.handles[-1].terminated and "Cancelled" in job["message"]
    assert training.cancel_run(started["id"]) is False  # nothing live any more
    assert training.cancel_run("nosuchid") is False


def test_only_one_real_run_at_a_time(tmp_path, enabled):
    launcher = FakeLauncher(hang_train=True, write_adapter=False)
    training.set_launcher(launcher)
    export = _make_export(tmp_path)
    first = training.start_run(export, {"steps": 50})
    try:
        with pytest.raises(ValueError, match="Another training run"):
            training.start_run(export, {"steps": 50})
    finally:
        training.cancel_run(first["id"])
        _wait(first["id"], {"cancelled"})


def test_a_record_left_running_by_a_closed_app_reads_as_interrupted(tmp_path, enabled):
    export = _make_export(tmp_path)
    training.set_launcher(FakeLauncher(train_code=1, write_adapter=False))
    job = _wait(training.start_run(export, {"steps": 50})["id"], {"failed"})
    path = os.path.join(job["run_dir"], training.JOB_FILE)
    data = json.load(open(path))
    data["status"] = "running"
    json.dump(data, open(path, "w"))
    with training._jobs_lock:
        training._jobs.clear()  # as after an app restart
    assert training.get_run(job["id"])["status"] == "interrupted"
    assert [r for r in training.list_runs() if r["id"] == job["id"]][0]["status"] == "interrupted"
    training.set_launcher(FakeLauncher())
    assert training.resume_run(job["id"])["resumes"] == 1
    _wait(job["id"], {"done"})


def test_dry_runs_and_finished_runs_cannot_be_resumed(tmp_path, enabled):
    training.set_launcher(FakeLauncher())
    dry = training.start_run(_make_export(tmp_path), dry_run=True)
    with pytest.raises(ValueError, match="nothing to resume"):
        training.resume_run(dry["id"])
    with pytest.raises(ValueError, match="missing"):
        training.resume_run("nosuchid")


# ── evaluation gate ──
def test_loss_check_needs_a_baseline_and_a_real_improvement():
    hist = [{"step": 0, "eval_loss": 4.0}, {"step": 100, "eval_loss": 3.0}, {"step": 200, "eval_loss": 3.2}]
    check = training_eval.loss_check(hist)
    assert check["status"] == "passed" and check["best_step"] == 100 and check["final_step"] == 200
    assert check["relative_improvement"] == 0.2
    assert training_eval.loss_check([{"step": 0, "eval_loss": 4.0}, {"step": 9, "eval_loss": 3.99}])["status"] == "failed"
    assert training_eval.loss_check([{"step": 5, "eval_loss": 3.0}])["status"] == "not_run"
    assert training_eval.loss_check([{"step": 0, "eval_loss": 3.0}])["status"] == "not_run"
    assert training_eval.loss_check([])["status"] == "not_run"


def test_wer_and_the_other_metric_plumbing():
    assert training_eval.wer("the cat sat", "the cat sat") == 0
    assert training_eval.wer("the cat sat", "the cat") == pytest.approx(1 / 3)
    assert training_eval.wer("the cat sat", "a dog sat down") == pytest.approx(3 / 3)
    assert training_eval.wer("", "") == 0 and training_eval.wer("", "x") == 1
    assert training_eval.wer("Hello, World!", "hello world") == 0
    ok = training_eval.wer_check(["a b c d"], ["a b c d"], ["a b c d"])
    bad = training_eval.wer_check(["a b c d"], ["a b c d"], ["a x c d"])
    assert ok["status"] == "passed" and bad["status"] == "failed" and bad["adapter_wer"] == 0.25
    assert training_eval.similarity_check(0.6, 0.7)["status"] == "passed"
    assert training_eval.similarity_check(0.6, 0.5)["status"] == "failed"
    with pytest.raises(ValueError):
        training_eval.mean_wer(["a"], [])


def test_only_a_fully_passed_record_is_ship_ready():
    passed = [{"check": c, "status": "passed"} for c in training_eval.CHECKS]
    assert training_eval.build_record(passed, heldout=5)["ship_ready"] is True
    one_failed = [*passed[:1], {"check": "asr_wer", "status": "failed"}, *passed[2:]]
    rec = training_eval.build_record(one_failed, heldout=5)
    assert rec["verdict"] == "failed" and rec["failed"] == ["asr_wer"] and not rec["ship_ready"]
    only_loss = training_eval.build_record(passed[:1], heldout=5)
    assert only_loss["verdict"] == "incomplete" and len(only_loss["pending"]) == 3 and not only_loss["ship_ready"]


def test_heldout_selection_is_stable_and_skips_bad_rows(tmp_path):
    rows = [{"id": f"c{i}", "audio_path": f"/a/{i}.wav", "text": f"t{i}", "language_id": "en"} for i in range(30)]
    path = tmp_path / "val.jsonl"
    path.write_text("\n".join([json.dumps(r) for r in rows] + ["not json", json.dumps({"id": "x"})]), encoding="utf-8")
    first = training_eval.select_heldout(str(path), 8)
    shuffled = tmp_path / "val2.jsonl"
    shuffled.write_text("\n".join(json.dumps(r) for r in reversed(rows)), encoding="utf-8")
    assert [r["id"] for r in first] == [r["id"] for r in training_eval.select_heldout(str(shuffled), 8)]
    assert len(first) == 8 and training_eval.select_heldout(None) == [] and training_eval.select_heldout(str(tmp_path / "no")) == []


def test_the_real_probe_describes_this_python_without_needing_a_gpu():
    import sys

    probe = training.SubprocessLauncher().probe(sys.executable, training.child_env(training.TrainOptions()))
    assert "error" not in probe and "accelerate" in probe and "omnivoice" in probe and "cuda" in probe or "torch_error" in probe
    assert training.assess_hardware(probe)["ok"] in (True, False)  # a verdict, never an exception
    assert training.SubprocessLauncher().probe("Z:/no/such/python.exe", {})["error"]
