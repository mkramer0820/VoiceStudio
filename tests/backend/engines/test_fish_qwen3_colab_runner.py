"""The Colab runner and notebook written into every Qwen3-TTS export. No network, no GPU, no Colab: Drive is a temp folder,
the GPU and Qwen's scripts are fakes. The runner itself is standard-library Python that is copied into the export, so it is
imported here from the export folder exactly as Colab would run it."""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import py_compile
import re
import sys
import wave
import zipfile

import pytest

from engines.fish_api import qwen3_colab

RUNNER_PATH = qwen3_colab.runner.__file__
NOTEBOOK_COMMITTED = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "notebooks", qwen3_colab.NOTEBOOK_NAME))


def _load_runner():
    spec = importlib.util.spec_from_file_location("voicestudio_qwen3_colab", RUNNER_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["voicestudio_qwen3_colab"] = mod
    spec.loader.exec_module(mod)
    return mod


vs = _load_runner()


# ------------------------------------------------------------------------------------------------ helpers
def write_wav(path, seconds=0.5, rate=24000, channels=1):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x01\x02" * int(rate * seconds) * channels)


def make_dataset(folder, n=5, rate=24000):
    for i in range(1, n + 1):
        write_wav(os.path.join(folder, "audio", f"{i:06d}.wav"), rate=rate)
    write_wav(os.path.join(folder, "reference", "ref.wav"))
    with open(os.path.join(folder, "train.jsonl"), "w", encoding="utf-8") as fh:
        for i in range(1, n + 1):
            fh.write(json.dumps({"audio": f"audio/{i:06d}.wav", "text": f"clip {i}", "ref_audio": "reference/ref.wav"}) + "\n")
    return str(folder)


def zip_folder(folder, archive, top=None):
    with zipfile.ZipFile(archive, "w") as zf:
        for base, _d, files in os.walk(folder):
            for f in files:
                full = os.path.join(base, f)
                rel = os.path.relpath(full, folder).replace(os.sep, "/")
                zf.write(full, (top + "/" + rel) if top else rel)
    return str(archive)


class Env:
    """A fake Colab: Drive and /content are folders under tmp_path, the repo checkout and models are pre-made."""

    def __init__(self, tmp_path, **overrides):
        self.drive = str(tmp_path / "MyDrive" / "VoiceStudio-training")
        self.content = str(tmp_path / "content")
        os.makedirs(self.content, exist_ok=True)
        base = dict(drive_root=self.drive, local_root=self.content, speaker_name="Sunny", epochs=2, lr=2e-5, batch_size=2,
                    sample_after_training=False, heartbeat_seconds=5)
        self.p = vs.params_from({**base, **overrides}, env={})
        self.lay = vs.Layout(self.p)

    def install_repo_and_models(self):
        ft = os.path.join(self.lay.local_repo, "finetuning")
        os.makedirs(ft)
        open(os.path.join(ft, "sft_12hz.py"), "w").write('x = 1\nattn_implementation="flash_attention_2"\n')
        for m in (self.p.base_model, self.p.tokenizer_model):
            d = vs.local_model_dir(self.lay, m)
            os.makedirs(d)
            open(os.path.join(d, "weights.bin"), "wb").write(b"0" * 2048)
            open(os.path.join(d, vs.DONE_MARKER), "w").write("x")


A100 = vs.Gpu("NVIDIA A100-SXM4-40GB", 40.0, (8, 0))
A100_80 = vs.Gpu("NVIDIA A100-SXM4-80GB", 80.0, (8, 0))
H100 = vs.Gpu("NVIDIA H100 80GB HBM3", 79.6, (9, 0))
L4 = vs.Gpu("NVIDIA L4", 22.5, (8, 9))
T4 = vs.Gpu("Tesla T4", 15.0, (7, 5))
G4_96 = vs.Gpu("NVIDIA RTX PRO 6000 Blackwell Server Edition", 95.6, (12, 0))
G4_48 = vs.Gpu("NVIDIA RTX PRO 6000 Blackwell Server Edition", 47.5, (12, 0))
ENV = {"python": "cp312", "python_version": "3.12.1", "platform": "linux-x86_64", "torch": "2.8.0", "torch_mm": "2.8", "cuda": "12.8",
       "cuda_tag": "cu12", "cxx11abi": True, "gpu": "NVIDIA A100-SXM4-40GB", "gpu_cc": "8.0"}


@pytest.fixture(autouse=True)
def _clean_times():
    vs._TIMES.clear()
    yield
    vs._TIMES.clear()


# ------------------------------------------------------------------------------------------------ the script itself
def test_the_runner_is_standard_library_only_and_compiles(tmp_path):
    py_compile.compile(RUNNER_PATH, cfile=str(tmp_path / "r.pyc"), doraise=True)
    stdlib = set(sys.stdlib_module_names) | {"__future__"}
    tree = ast.parse(open(RUNNER_PATH, encoding="utf-8").read())
    top = {n.names[0].name.split(".")[0] if isinstance(n, ast.Import) else (n.module or "").split(".")[0]
           for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))}
    assert top <= stdlib, top - stdlib
    for code in (vs.DRIVER_SOURCE, vs.SAMPLE_CODE):
        ast.parse(code)


def test_the_export_carries_the_runner_the_notebook_and_a_thin_shell_file(tmp_path):
    base = qwen3_colab.settings_from({"epochs": 4, "learning_rate": 3e-5, "batch_size": 3, "drive_folder": "Tr/a"})
    report = {"counts": {"source_samples": 5, "accepted": 5, "rejected": 0, "train": 5}, "duration": {"total": "0:00:03", "average_s": 0.5,
              "median_s": 0.5, "shortest_s": 0.5, "longest_s": 0.5}, "reference": {"path": "reference/ref.wav", "source": "x", "duration": 5, "reason": "r"},
              "speaker": {"name": "S", "qwen_speaker_name": "S"}, "warnings": []}
    files = qwen3_colab.render_files(speaker="Sunny", folder_name="set", zip_name="set.zip", settings=base, report=report)
    assert {"colab_setup.sh", "colab_commands.txt", "README.txt", qwen3_colab.RUNNER_NAME, qwen3_colab.NOTEBOOK_NAME} == set(files)
    out = tmp_path / "export"
    out.mkdir()
    for name, text in files.items():
        (out / name).write_text(text, encoding="utf-8")
    py_compile.compile(str(out / qwen3_colab.RUNNER_NAME), cfile=str(tmp_path / "e.pyc"), doraise=True)
    assert sum(1 for l in files["colab_setup.sh"].splitlines() if l.startswith("exec python3")) == 1 and "pip " not in files["colab_setup.sh"]


# ------------------------------------------------------------------------------------------------ settings reach commands
def test_settings_from_training_settings_json_reach_the_run_and_the_provenance(tmp_path):
    st = tmp_path / "training_settings.json"
    st.write_text(json.dumps({"epochs": 7, "learning_rate": 1e-5, "batch_size": 4, "speaker_name": "Sunny", "drive_folder": "My/Folder",
                              "base_model": vs.BASE_MODEL, "tokenizer_model": vs.TOKENIZER}), encoding="utf-8")
    p = vs.params_from({"epochs": 9}, str(st), env={"BATCH_SIZE": "auto"})
    assert (p.epochs, p.lr, p.speaker_name, p.drive_root) == (9, 1e-5, "Sunny", "/content/drive/MyDrive/My/Folder")
    assert p.batch_size == "auto"  # environment beats the file; explicit overrides beat both
    prov = p.provenance()
    assert prov["epochs"]["source"] == "owner override" and prov["lr"]["source"] == "set in the VoiceStudio export dialog"
    assert prov["base_model"]["source"] == "official default" and prov["keep_all"]["source"] == "our default"
    env = Env(tmp_path)
    plan = vs.plan_commands(env.p, env.lay, "/content/ds", 4)
    flags = plan["train"]
    assert flags[flags.index("--num_epochs") + 1] == "2" and flags[flags.index("--batch_size") + 1] == "4"
    assert flags[flags.index("--speaker_name") + 1] == "Sunny" and flags[flags.index("--lr") + 1] == "2e-05"
    assert flags[flags.index("--output_model_path") + 1].startswith(env.content)  # local first, never straight to Drive
    assert qwen3_colab.commands(qwen3_colab.settings_from({"epochs": 7}), "Sunny", "x")["train"].count("--num_epochs 7") == 1


@pytest.mark.parametrize("bad", [{"speaker_name": "bad name"}, {"epochs": 0}, {"lr": 5}, {"batch_size": "x"}, {"run_id": "../x"},
                                 {"keep_all": False, "keep_final": False}, {"base_model": "nope"}, {"drive_root": "relative"}])
def test_bad_parameters_are_refused_with_a_message(bad, tmp_path):
    with pytest.raises(vs.StepFailed):
        vs.params_from(bad, env={})


# ------------------------------------------------------------------------------------------------ Drive layout
def test_layout_is_created_idempotently_and_never_touches_foreign_files(tmp_path):
    env = Env(tmp_path)
    os.makedirs(env.drive)
    foreign = os.path.join(env.drive, "my-notes.txt")
    open(foreign, "w").write("precious")
    os.makedirs(os.path.join(env.drive, "runs", "Sunny", "older-run", "checkpoints", "checkpoint-epoch-0"))
    vs.ensure_layout(env.p)
    first = sorted(os.walk(env.drive))
    vs.ensure_layout(env.p)
    assert sorted(os.walk(env.drive)) == first and open(foreign).read() == "precious"
    for sub in ("cache/models", "cache/wheelhouse", "cache/pip", "datasets", "prepared/Sunny", "logs",
                f"runs/Sunny/{env.p.run_id}/checkpoints", f"runs/Sunny/{env.p.run_id}/samples", f"runs/Sunny/{env.p.run_id}/environment"):
        assert os.path.isdir(os.path.join(env.drive, *sub.split("/")))
    assert os.path.isdir(os.path.join(env.drive, "runs", "Sunny", "older-run", "checkpoints", "checkpoint-epoch-0"))


def test_a_second_run_gets_its_own_folder(tmp_path):
    a = Env(tmp_path, run_id="run-a")
    b = vs.params_from(dict(drive_root=a.drive, local_root=a.content, speaker_name="Sunny", run_id="run-b"), env={})
    assert vs.Layout(a.p).run != vs.Layout(b).run and os.path.dirname(vs.Layout(a.p).run) == os.path.dirname(vs.Layout(b).run)
    assert vs.params_from(dict(drive_root=a.drive), env={}).run_id != ""  # a timestamp when none is given


def test_a_missing_drive_folder_is_never_created(tmp_path, monkeypatch):
    """A second folder with the same name made Colab resolve the name to an empty copy; the runner must stop instead."""
    mount = (tmp_path / "drive").as_posix()
    os.makedirs(f"{mount}/MyDrive/Qwen3TTS (1)")  # a look-alike next to the missing folder
    monkeypatch.setattr(vs, "DRIVE_MOUNT", mount)
    p = vs.params_from({"drive_root": f"{mount}/MyDrive/Qwen3TTS", "local_root": str(tmp_path / "content")}, env={})
    with pytest.raises(vs.StepFailed, match=r"does not exist, so nothing was created.*Qwen3TTS \(1\)"):
        vs.ensure_layout(p)
    assert not os.path.exists(f"{mount}/MyDrive/Qwen3TTS")
    os.makedirs(f"{mount}/MyDrive/Qwen3TTS")  # once the folder is really there, the layout is built inside it
    vs.ensure_layout(p)
    assert os.path.isdir(f"{mount}/MyDrive/Qwen3TTS/cache")


def test_the_mount_cell_does_not_create_a_missing_drive_folder(tmp_path, monkeypatch):
    import time as _time

    monkeypatch.setattr(_time, "sleep", lambda s: None)
    nb = _notebook(folder_name="set")
    mount_cell = next("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code" and "drive.mount" in "".join(c["source"]))
    google, colab = type(sys)("google"), type(sys)("google.colab")
    colab.drive = type("D", (), {"mount": staticmethod(lambda p: None)})
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    ns = {"DRIVE_ROOT": "/content/drive/MyDrive/Qwen3TTS_not_there", "DATA_SOURCE": str(tmp_path / "x.zip")}
    with pytest.raises(FileNotFoundError, match="does not exist, so nothing was created"):
        exec(mount_cell, ns)
    assert not os.path.exists("/content/drive/MyDrive/Qwen3TTS_not_there")


def test_unmounted_drive_is_a_clear_stop():
    p = vs.params_from({"drive_root": "/content/drive/MyDrive/X"}, env={})
    if not os.path.isdir("/content/drive/MyDrive"):
        with pytest.raises(vs.StepFailed, match="not mounted"):
            vs.ensure_layout(p)


# ------------------------------------------------------------------------------------------------ environment, GPU, attention
@pytest.mark.parametrize("gpu,batch,path", [(A100, 2, "flash2"), (A100_80, 4, "flash2"), (H100, 4, "flash2"), (L4, 1, "flash2"),
                                            (T4, 1, "refuse"), (G4_96, 4, "sdpa"), (G4_48, 2, "sdpa")])
def test_gpu_defaults_and_attention_path_per_runtime(gpu, batch, path):
    assert vs.recommended_batch(gpu) == batch
    assert vs.attention_choice(gpu, "auto")[0] == path
    assert vs.attention_choice(gpu, True)[0] == "sdpa"
    assert vs.attention_choice(None, "auto")[0] == "refuse"


def test_blackwell_can_be_forced_back_to_flash_attention_and_t4_cannot():
    assert vs.attention_choice(G4_96, False)[0] == "flash2" and "likely to fail" in vs.attention_choice(G4_96, False)[1]
    assert vs.attention_choice(T4, False)[0] == "refuse"
    assert vs.attention_choice(vs.Gpu("mystery", 40, None), "auto")[0] == "refuse"
    assert vs.Gpu("x", 10, None).arch == "unknown" and "Blackwell" in G4_96.arch


def test_nvidia_smi_parsing():
    g = vs.parse_nvidia_smi("NVIDIA A100-SXM4-40GB, 40960, 8.0\n")
    assert (g.name, g.vram_gb, g.cc) == ("NVIDIA A100-SXM4-40GB", 40.0, (8, 0))
    assert vs.parse_nvidia_smi("Tesla T4, 15360").cc == (7, 5)  # older driver without compute_cap: from the name table
    assert vs.parse_nvidia_smi("garbage") is None
    assert vs.detect_gpu(lambda cmd: (127, "")) is None


def test_batch_auto_and_a_too_big_batch_is_warned_but_kept(capsys):
    p = vs.params_from({"batch_size": "auto"}, env={})
    assert vs.resolve_batch(p, L4) == 1 and vs.resolve_batch(p, G4_96) == 4
    p8 = vs.params_from({"batch_size": 8}, env={})
    vs.gpu_report(p8, L4)
    out = capsys.readouterr().out
    assert "above the suggestion" in out and vs.resolve_batch(p8, L4) == 8
    assert "UNTESTED" in out and "FlashAttention" in out


def test_gpu_report_explains_blackwell_default(capsys):
    vs.gpu_report(vs.params_from({}, env={}), G4_96)
    out = capsys.readouterr().out
    assert "SDPA" in out and "open requests" in out and "UNVERIFIED" in out and "ATTN_FALLBACK=False" in out


def test_the_wheel_fingerprint_includes_the_gpu_architecture():
    a, b = dict(ENV), dict(ENV, gpu="NVIDIA L4", gpu_cc="8.9")
    assert vs.wheel_key(a) != vs.wheel_key(b) and vs.full_key(a) != vs.full_key(b)
    assert vs.wheel_key(a) == vs.wheel_key(dict(ENV, python_version="3.12.9"))


def test_fingerprint_is_recorded_and_a_known_runtime_is_recognised(tmp_path):
    env = Env(tmp_path)
    vs.ensure_layout(env.p, create_run=False)
    assert vs.record_fingerprint(env.lay, ENV) is False
    assert vs.record_fingerprint(env.lay, ENV) is True
    assert vs.record_fingerprint(env.lay, dict(ENV, gpu_cc="8.9")) is False
    assert len(json.load(open(env.lay.fingerprints))) == 2


# ------------------------------------------------------------------------------------------------ dependency check table
def test_dependency_table_passes_and_fails_with_fake_probes(tmp_path, capsys):
    env = Env(tmp_path)
    env.install_repo_and_models()
    ok = lambda name: (True, "1.0")  # noqa: E731
    rows = vs.dependency_rows(env.p, env.lay, ENV, A100, probe=ok, flash_probe=lambda: (True, "2.8.3"), arch_probe=lambda cc: (True, "sm_80"))
    by = {r[0]: r[1] for r in rows}
    assert by["qwen_tts"] == by["transformers"] == by["attention: flash_attn"] == by["torch kernels for this GPU"] == "PASS"
    assert by["Qwen3-TTS repo"] == "FAIL"  # not a git checkout of the pinned commit here
    assert vs.check_table(rows) is False
    out = capsys.readouterr().out
    assert "FAILED: Qwen3-TTS repo" in out and "result" in out
    bad = vs.dependency_rows(env.p, env.lay, ENV, A100, probe=lambda n: (n != "librosa", "x"), flash_probe=lambda: (False, "no wheel"),
                             arch_probe=lambda cc: (True, "ok"))
    assert {r[0] for r in bad if r[1] == "FAIL"} >= {"librosa", "attention: flash_attn"}


def test_blackwell_rows_check_sdpa_and_torch_kernels(tmp_path):
    env = Env(tmp_path)
    rows = vs.dependency_rows(env.p, env.lay, dict(ENV, gpu_cc="12.0"), G4_96, probe=lambda n: (True, "1"), sdpa_probe=lambda: (False, "SDPA failed"),
                              arch_probe=lambda cc: (False, "torch lists sm_90, nothing for sm_120"))
    by = {r[0]: r for r in rows}
    assert by["attention: PyTorch SDPA (patched copy)"][1] == "FAIL" and by["torch kernels for this GPU"][1] == "FAIL"
    assert "attention: flash_attn" not in by


# ------------------------------------------------------------------------------------------------ FlashAttention wheel decisions
class FakeStream:
    """Records commands; behaviour per command can be scripted."""

    def __init__(self, handler=None):
        self.calls, self.handler = [], handler

    def __call__(self, cmd, *, cwd=None, env=None, log_path=None, on_line=None):
        self.calls.append(list(cmd))
        return self.handler(cmd, cwd, env, on_line) if self.handler else 0


def test_prebuilt_wheel_names_follow_the_release_listing():
    assert vs.prebuilt_wheel_name(ENV) == "flash_attn-2.8.3.post2+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
    assert vs.prebuilt_wheel_name(dict(ENV, cxx11abi=False, torch_mm="2.4", python="cp310")).endswith("cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl")
    for miss in (dict(ENV, torch_mm="2.4", python="cp313"), dict(ENV, torch_mm="2.9"), dict(ENV, cuda_tag="cu13"), dict(ENV, platform="win32"),
                 dict(ENV, cxx11abi=None)):
        assert vs.prebuilt_wheel_name(miss) is None


def _flash(env, gpu=A100, **kw):
    e = Env(env) if not isinstance(env, Env) else env
    vs.ensure_layout(e.p, create_run=False)
    return e


def test_flash_cache_hit_prebuilt_and_build_decisions(tmp_path):
    e = Env(tmp_path)
    vs.ensure_layout(e.p, create_run=False)
    state = {"works": False}
    functional = lambda: (state["works"], "ok" if state["works"] else "no")  # noqa: E731

    def installs(cmd, *_):
        if "install" in cmd and any(str(c).endswith(".whl") for c in cmd):
            state["works"] = True
        return 0

    # 1. nothing cached, a prebuilt wheel exists: download, install, and keep the wheel in this architecture's folder
    s1 = FakeStream(installs)
    got = vs.ensure_flash_attn(e.p, e.lay, ENV, A100, "log", stream=s1, functional=functional, exists=lambda u: True,
                               download=lambda url, dest: open(dest, "wb").write(b"w"))
    assert got == "prebuilt wheel" and os.path.isdir(os.path.join(e.lay.wheelhouse, "sm80"))
    assert vs.find_cached_wheel(e.lay, ENV).endswith(".whl")
    # 2. a new runtime with the same fingerprint: cache hit, no download
    state["works"] = False
    s2 = FakeStream(installs)
    got = vs.ensure_flash_attn(e.p, e.lay, ENV, A100, "log", stream=s2, functional=functional, exists=lambda u: pytest.fail("no network"),
                               download=lambda *a: pytest.fail("no download"))
    assert got == "cached wheel"
    # 3. already installed in this runtime: nothing to do
    state["works"] = True
    assert vs.ensure_flash_attn(e.p, e.lay, ENV, A100, "log", stream=FakeStream(), functional=functional) == "installed already"
    # 4. a different GPU architecture never reuses it
    assert vs.find_cached_wheel(e.lay, dict(ENV, gpu_cc="8.9")) is None


def test_flash_builds_once_and_saves_the_wheel_when_no_prebuilt_exists(tmp_path, monkeypatch):
    e = Env(tmp_path)
    vs.ensure_layout(e.p, create_run=False)
    state = {"works": False}
    env = dict(ENV, torch_mm="2.9")  # not in the release

    def stream(cmd, *_):
        if "wheel" in cmd and "-w" in cmd:
            out = cmd[cmd.index("-w") + 1]
            open(os.path.join(out, "flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl"), "wb").write(b"built")
        if "install" in cmd and any(str(c).endswith(".whl") for c in cmd):
            state["works"] = True
        return 0

    monkeypatch.setattr(vs.shutil, "which", lambda name: "/usr/local/cuda/bin/nvcc")
    s = FakeStream(stream)
    got = vs.ensure_flash_attn(e.p, e.lay, env, A100, "log", stream=s, functional=lambda: (state["works"], "x"), exists=lambda u: pytest.fail("not offered"))
    assert got == "built"
    assert os.path.isfile(os.path.join(e.lay.wheelhouse, "sm80", "flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl"))
    assert not [d for d in os.listdir(os.path.join(e.lay.wheelhouse, "sm80")) if d.startswith("build-")]
    assert vs.find_cached_wheel(e.lay, env) is not None


def test_flash_is_skipped_on_the_sdpa_path_and_without_a_gpu(tmp_path):
    e = Env(tmp_path)
    vs.ensure_layout(e.p, create_run=False)
    never = FakeStream(lambda *a: pytest.fail("must not run"))
    assert vs.ensure_flash_attn(e.p, e.lay, ENV, G4_96, "log", stream=never) == "skipped"
    assert vs.ensure_flash_attn(e.p, e.lay, ENV, T4, "log", stream=never) == "skipped"
    assert vs.ensure_flash_attn(e.p, e.lay, ENV, None, "log", stream=never) == "skipped"


def test_a_cached_wheel_that_fails_on_the_gpu_is_rejected_not_deleted(tmp_path):
    e = Env(tmp_path)
    vs.ensure_layout(e.p, create_run=False)
    folder = os.path.join(e.lay.wheelhouse, "sm80")
    os.makedirs(folder)
    wheel = os.path.join(folder, "flash_attn-1-cp312-cp312-linux_x86_64.whl")
    open(wheel, "wb").write(b"w")
    vs._write_wheel_meta(wheel, ENV, "test")
    with pytest.raises(vs.StepFailed):  # cached fails, no prebuilt (torch 2.9), then no nvcc
        vs.ensure_flash_attn(e.p, e.lay, dict(ENV, torch_mm="2.9", cxx11abi=True), A100, "log", stream=FakeStream(),
                             functional=lambda: (False, "no"), exists=lambda u: False)
    assert os.path.isfile(wheel)


# ------------------------------------------------------------------------------------------------ models
def test_models_cache_hit_versus_miss_and_the_local_copy(tmp_path):
    e = Env(tmp_path)
    vs.ensure_layout(e.p, create_run=False)
    downloads = []

    def fake_download(model, dest):
        downloads.append(model)
        open(os.path.join(dest, "model.safetensors"), "wb").write(b"0" * 100)

    vs.ensure_models(e.p, e.lay, download=fake_download)
    assert downloads == [e.p.base_model, e.p.tokenizer_model]
    base_local = vs.local_model_dir(e.lay, e.p.base_model)
    assert os.path.isfile(os.path.join(base_local, "model.safetensors")) and os.path.isfile(os.path.join(base_local, vs.DONE_MARKER))
    # new runtime: /content is empty again but the Drive cache is complete -> no download, only the copy
    import shutil

    shutil.rmtree(e.lay.local_models)
    vs.ensure_models(e.p, e.lay, download=lambda *a: pytest.fail("cache hit must not download"))
    assert os.path.isfile(os.path.join(base_local, "model.safetensors"))
    # same runtime again: nothing at all
    vs.ensure_models(e.p, e.lay, download=lambda *a: pytest.fail("no"))
    assert [t["step"] for t in vs._TIMES if "download" in t["step"]] and any("copy" in t["step"] for t in vs._TIMES)


def test_models_can_skip_the_drive_cache_and_never_write_a_token(tmp_path, monkeypatch):
    e = Env(tmp_path, cache_models_on_drive=False)
    vs.ensure_layout(e.p, create_run=False)
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
    seen = []
    vs.ensure_models(e.p, e.lay, download=lambda m, d: (seen.append(d), open(os.path.join(d, "w"), "w").write("x")))
    assert all(d.startswith(e.content) for d in seen) and not os.listdir(e.lay.models)
    for base, _d, files in os.walk(tmp_path):
        for f in files:
            assert b"hf_secret_value" not in open(os.path.join(base, f), "rb").read()
    assert vs.hf_token()[1] == "the HF_TOKEN environment variable"


# ------------------------------------------------------------------------------------------------ dataset checks
def test_dataset_verification_catches_missing_file_wrong_rate_and_absolute_path(tmp_path):
    good = make_dataset(tmp_path / "good")
    r = vs.verify_dataset(good)
    assert r["ok"] and r["rows"] == r["audio"] == 5 and r["seconds"] == 2.5 and r["reference_files"] == 1
    os.remove(os.path.join(good, "audio", "000002.wav"))
    r = vs.verify_dataset(good)
    assert not r["ok"] and any("row 2" in p and "does not exist" in p and "000002.wav" in p for p in r["problems"])
    bad_rate = make_dataset(tmp_path / "rate")
    write_wav(os.path.join(bad_rate, "audio", "000003.wav"), rate=44100)
    assert any("44100 Hz" in p and "row 3" in p for p in vs.verify_dataset(bad_rate)["problems"])
    stereo = make_dataset(tmp_path / "stereo")
    write_wav(os.path.join(stereo, "audio", "000001.wav"), channels=2)
    assert any("2 channel" in p for p in vs.verify_dataset(stereo)["problems"])
    absolute = make_dataset(tmp_path / "abs")
    rows = [json.loads(line) for line in open(os.path.join(absolute, "train.jsonl"))]
    rows[0]["audio"] = os.path.abspath(os.path.join(absolute, "audio", "000001.wav"))
    rows[1]["audio"] = "C:\\clips\\x.wav"
    rows[2]["ref_audio"] = "../outside.wav"
    open(os.path.join(absolute, "train.jsonl"), "w").write("\n".join(json.dumps(r) for r in rows) + "\nnot json\n")
    problems = " | ".join(vs.verify_dataset(absolute)["problems"])
    assert "row 1" in problems and "absolute path" in problems and "row 2" in problems and "leaves the dataset folder" in problems
    assert "row 6: not valid JSON" in problems
    assert vs.verify_dataset(str(tmp_path / "nothing"))["ok"] is False


@pytest.mark.parametrize("top", [None, "my-dataset"])
def test_both_zip_layouts_are_accepted_and_the_right_folder_is_used(tmp_path, top):
    src = make_dataset(tmp_path / "src")
    archive = zip_folder(src, tmp_path / "datasets" / "set.zip", top=top) if (tmp_path / "datasets").mkdir() is None else None
    e = Env(tmp_path, data_source=archive)
    folder = vs.prepare_dataset(e.p)
    assert os.path.isfile(os.path.join(folder, "train.jsonl")) and folder.startswith(e.content)
    assert vs.prepare_dataset(e.p) == folder  # second call: not copied again


def test_data_source_variants_folder_unzipped_and_errors(tmp_path):
    src = make_dataset(tmp_path / "drive" / "set")
    e = Env(tmp_path, data_source=src)
    local = vs.prepare_dataset(e.p)
    assert local == os.path.join(e.content, "set") and os.path.isfile(os.path.join(local, "train.jsonl"))
    in_place = Env(tmp_path, data_source=local)
    assert vs.prepare_dataset(in_place.p) == local  # already under /content: used where it is
    parent = Env(tmp_path, data_source=str(tmp_path / "drive"))
    assert vs.prepare_dataset(parent.p).endswith("set")  # a folder holding exactly one dataset
    with pytest.raises(vs.StepFailed, match="does not exist"):
        vs.prepare_dataset(vs.params_from({"data_source": str(tmp_path / "missing")}, env={}))
    with pytest.raises(vs.StepFailed, match="empty"):
        vs.prepare_dataset(vs.params_from({}, env={}))


def test_unsafe_zip_members_are_refused(tmp_path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape.txt", "x")
    e = Env(tmp_path, data_source=str(archive))
    with pytest.raises(vs.StepFailed, match="unsafe path"):
        vs.prepare_dataset(e.p)
    assert not (tmp_path / "escape.txt").exists()


def test_a_changed_source_replaces_only_our_own_local_copy(tmp_path):
    src = make_dataset(tmp_path / "s")
    archive = zip_folder(src, tmp_path / "a.zip")
    e = Env(tmp_path, data_source=archive)
    local = vs.prepare_dataset(e.p)
    open(os.path.join(src, "train.jsonl"), "a").write("")
    write_wav(os.path.join(src, "audio", "000009.wav"))
    zip_folder(src, tmp_path / "a.zip")
    os.utime(archive, (1, 1))
    assert vs.prepare_dataset(e.p) == local
    foreign = os.path.join(e.content, "b")
    os.makedirs(foreign)
    open(os.path.join(foreign, "mine.txt"), "w").write("x")
    with pytest.raises(vs.StepFailed, match="not made by this script"):
        vs.prepare_dataset(Env(tmp_path, data_source=zip_folder(src, tmp_path / "b.zip")).p)
    assert os.path.isfile(os.path.join(foreign, "mine.txt"))


# ------------------------------------------------------------------------------------------------ prepared codes: hash-keyed cache, progress, crash safety
def fake_prepare(rows_per_call=None, crash_after=None):
    """Fake prepare_progress.py: reads the input jsonl, appends rows with audio_codes to the partial file."""
    def run(cmd, cwd, env, on_line):
        a = {cmd[i]: cmd[i + 1] for i in range(len(cmd) - 1) if str(cmd[i]).startswith("--")}
        rows = [json.loads(line) for line in open(os.path.join(cwd, a["--input_jsonl"]), encoding="utf-8") if line.strip()]
        start = int(a["--resume_from"])
        with open(a["--output_jsonl"], "a" if start else "w", encoding="utf-8") as out:
            for n, row in enumerate(rows[start:], start + 1):
                if crash_after is not None and n > crash_after:
                    out.write('{"audio": "torn')  # a torn final line
                    return 1
                row["audio_codes"] = [[1, 2], [3, 4]]
                out.write(json.dumps(row) + "\n")
        return 0
    return run


def _codes_env(tmp_path, n=5):
    e = Env(tmp_path)
    e.install_repo_and_models()
    vs.ensure_layout(e.p)
    folder = make_dataset(tmp_path / "ds", n=n)
    return e, folder, vs.compute_dataset_hash(folder)


def test_codes_are_generated_validated_promoted_and_reused_only_on_a_matching_hash(tmp_path, capsys):
    e, folder, h = _codes_env(tmp_path)
    s = FakeStream(fake_prepare())
    local = vs.ensure_codes(e.p, e.lay, folder, h, 5, "log", stream=s)
    assert len(s.calls) == 1 and os.path.isfile(e.lay.codes) and not os.path.exists(e.lay.codes + ".partial") and local == e.lay.codes_local
    meta = json.load(open(e.lay.codes_meta))
    assert meta["dataset_hash"] == h and meta["rows"] == 5 and meta["codes_sha256"] == vs.sha256_file(e.lay.codes)
    assert "no prepared codes" in capsys.readouterr().out
    # same hash: reused, the driver is not run
    assert vs.ensure_codes(e.p, e.lay, folder, h, 5, "log", stream=FakeStream(lambda *a: pytest.fail("stale/duplicate work"))) == e.lay.codes_local
    assert "cache hit" in capsys.readouterr().out
    # changed data: regenerated and the reason is said, never silently reused
    write_wav(os.path.join(folder, "audio", "000001.wav"), seconds=0.7)
    h2 = vs.compute_dataset_hash(folder)
    assert h2 != h
    s2 = FakeStream(fake_prepare())
    vs.ensure_codes(e.p, e.lay, folder, h2, 5, "log", stream=s2)
    assert len(s2.calls) == 1 and "dataset hash changed" in capsys.readouterr().out
    # a tampered codes file is not trusted either
    open(e.lay.codes, "a").write("x")
    s3 = FakeStream(fake_prepare())
    vs.ensure_codes(e.p, e.lay, folder, h2, 5, "log", stream=s3)
    assert len(s3.calls) == 1 and "no longer matches" in capsys.readouterr().out


def test_interrupted_preprocessing_resumes_from_whole_lines_and_a_bad_result_is_refused(tmp_path, capsys):
    e, folder, h = _codes_env(tmp_path, n=6)
    with pytest.raises(vs.StepFailed, match="partial file is kept"):
        vs.ensure_codes(e.p, e.lay, folder, h, 6, "log", stream=FakeStream(fake_prepare(crash_after=3)))
    assert os.path.isfile(e.lay.codes + ".partial") and not os.path.exists(e.lay.codes)  # never promoted
    s = FakeStream(fake_prepare())
    vs.ensure_codes(e.p, e.lay, folder, h, 6, "log", stream=s)
    assert "continuing after them" in capsys.readouterr().out and s.calls[0][s.calls[0].index("--resume_from") + 1] == "3"
    assert sum(1 for _ in open(e.lay.codes)) == 6 and vs.validate_codes(e.lay.codes, 6) == 6
    # a different dataset hash with a leftover partial restarts cleanly
    e2, folder2, h2 = _codes_env(tmp_path / "second")
    open(e2.lay.codes + ".partial", "w").write('{"x": 1}\n')
    open(e2.lay.codes + ".partial.json", "w").write(json.dumps({"dataset_hash": "other"}))
    vs.ensure_codes(e2.p, e2.lay, folder2, h2, 5, "log", stream=FakeStream(fake_prepare()))
    assert "restarting from the first row" in capsys.readouterr().out
    # validation: missing audio_codes / wrong row count name the row
    broken = tmp_path / "broken.jsonl"
    broken.write_text('{"audio_codes": [1]}\n{"audio": "x"}\n')
    with pytest.raises(vs.StepFailed, match="row 2 has no audio_codes"):
        vs.validate_codes(str(broken), 2)
    with pytest.raises(vs.StepFailed, match="1 rows but train.jsonl has 2"):
        vs.validate_codes(str(_one(tmp_path)), 2)


def _one(tmp_path):
    p = tmp_path / "one.jsonl"
    p.write_text('{"audio_codes": [1]}\n')
    return p


def test_the_preprocessing_driver_prints_progress_and_writes_each_batch(tmp_path):
    """Run the real driver source with a fake qwen_tts module: counts, percent, ETA and incremental writes."""
    pkg = tmp_path / "fake" / "qwen_tts"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(
        "class _C:\n    def __init__(self, v): self.v = v\n    def cpu(self): return self\n    def tolist(self): return self.v\n"
        "class _E:\n    def __init__(self, n): self.audio_codes = [_C([[i]]) for i in range(n)]\n"
        "class Qwen3TTSTokenizer:\n    @classmethod\n    def from_pretrained(cls, path, device_map=None): return cls()\n"
        "    def encode(self, audios): return _E(len(audios))\n")
    data = tmp_path / "in.jsonl"
    data.write_text("\n".join(json.dumps({"audio": f"a{i}.wav", "text": "t", "ref_audio": "r.wav"}) for i in range(70)) + "\n")
    driver = tmp_path / "driver.py"
    driver.write_text(vs.DRIVER_SOURCE)
    import subprocess

    out = tmp_path / "out.partial"
    r = subprocess.run([sys.executable, str(driver), "--tokenizer_model_path", "x", "--input_jsonl", str(data), "--output_jsonl", str(out)],
                       capture_output=True, text=True, env={"PYTHONPATH": str(tmp_path / "fake"), "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")})
    assert r.returncode == 0, r.stderr
    assert "PREP 32/70 (46%)" in r.stdout and "PREP 70/70 (100%)" in r.stdout and "ETA" in r.stdout and "elapsed" in r.stdout
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 70 and all(row["audio_codes"] for row in rows)


# ------------------------------------------------------------------------------------------------ progress parsing and projection
def test_progress_line_is_parsed_from_the_official_output():
    clock = iter([0.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 100.0])
    mon = vs.TrainMonitor(epochs=3, batches_per_epoch=100, clock=lambda: next(clock))
    mon.start = 0.0
    assert mon.feed("loading model") is None
    pr = mon.feed("Epoch 0 | Step 0 | Loss: 4.5000")
    assert pr["epoch"] == 0 and pr["step"] == 0 and pr["loss"] == 4.5 and pr["steps_per_s"] is None
    pr = mon.feed("Epoch 0 | Step 10 | Loss: 4.0000")
    assert pr["steps_per_s"] == pytest.approx(10 / 2.0) and pr["avg_loss"] == pytest.approx(4.25)
    assert pr["eta_epoch"] == pytest.approx(90 / 5.0) and pr["eta_all"] == pytest.approx((300 - 10) / 5.0)
    text = vs.format_progress(pr, (21000.0, 22000.0))
    for piece in ("epoch 1/3", "step 10/100", "loss 4.0000", "avg10 4.2500", "5.00 step/s", "epoch ETA", "overall ETA", "GPU 20.5 GB"):
        assert piece in text
    assert mon.feed("torch.OutOfMemoryError: CUDA out of memory") is None and mon.oom


def test_heartbeat_only_speaks_when_quiet(capsys):
    t = {"now": 0.0}
    mon = vs.TrainMonitor(epochs=1, batches_per_epoch=10, clock=lambda: t["now"])
    peak = vs.GpuPeak(lambda: 1000.0)
    hb = vs.Heartbeat(mon, peak, every=60, clock=lambda: t["now"])
    t["now"] = 30
    assert hb.beat() is None
    t["now"] = 90
    msg = hb.beat()
    assert msg and "still running" in msg and "no output for" in msg
    t["now"] = 100
    assert hb.beat() is None  # not again straight away


def test_projection_from_a_short_sample():
    clock = {"now": 0.0}
    mon = vs.TrainMonitor(epochs=1, batches_per_epoch=21, clock=lambda: clock["now"])
    mon.start = 0.0
    for step, t in ((0, 30.0), (10, 40.0), (20, 50.0)):
        clock["now"] = t
        mon.feed(f"Epoch 0 | Step {step} | Loss: 1.0")
    mon.end = 100.0
    proj = vs.project(mon, clips_used=42, total_clips=1600, batch=2, epochs=10, ckpt_bytes=4 * 2**30, prep_seconds=60.0)
    assert proj["seconds_per_batch"] == pytest.approx(1.0) and proj["batches_per_epoch"] == 800
    assert proj["epoch_seconds"] == pytest.approx(800.0) and proj["startup_seconds"] == 30.0 and proj["save_seconds"] == pytest.approx(50.0)
    assert proj["total_seconds"] == pytest.approx(60 * 1600 / 42 + 30 + 10 * 850)
    assert vs.project(vs.TrainMonitor(epochs=1, batches_per_epoch=1), clips_used=1, total_clips=1, batch=1, epochs=1, ckpt_bytes=0,
                      prep_seconds=0)["seconds_per_batch"] is None


def _proj(epoch_seconds=3600.0, total=3600.0 * 5):
    return {"seconds_per_batch": 1.0, "batches_per_epoch": 800, "epoch_seconds": epoch_seconds, "save_seconds": 60.0, "startup_seconds": 30.0,
            "prep_seconds_full": 300.0, "ckpt_bytes": 4 * 2**30, "total_seconds": total}


def test_units_projection_fit_budget_and_break_even(capsys):
    p = vs.params_from({"epochs": 5, "units_available": 100}, env={})
    vs.report_projection(p, A100, "flash2", 2, 42, 1600, 30 * 1024, _proj(), 600.0)
    out = capsys.readouterr().out
    assert "ESTIMATE from a short sample" in out and "FLASH2" in out and "30.0 GB of 40 GB" in out
    assert "5.4 units/hour" in out and "A100 40 GB" in out and "NOT verified" in out and "runtime menu" in out
    assert "one epoch: 5.4 units" in out and "all 5 epochs: 27.0 units" in out and "setup in this session" in out
    assert "FITS" in out and "G4 vs A100 40 GB is 8.7 / 5.4 = 1.6x" in out and "7.5 / 5.4 = 1.4x" in out and "4.0 GB" in out
    # does not fit
    p2 = vs.params_from({"epochs": 5, "units_available": 10}, env={})
    vs.report_projection(p2, G4_96, "sdpa", 4, 42, 1600, 1000, _proj(), 0.0)
    out = capsys.readouterr().out
    assert "8.7 units/hour" in out and "DOES NOT FIT" in out and "SDPA" in out and "not measured in this session" in out
    # unknown rate: says how to set it; H100 has none in the table
    vs.report_projection(p, H100, "flash2", 4, 42, 1600, 1000, _proj(), 0.0)
    out = capsys.readouterr().out
    assert "UNITS_PER_HOUR is empty" in out and "units" in out
    p3 = vs.params_from({"units_per_hour": 12.5}, env={})
    vs.report_projection(p3, H100, "flash2", 4, 42, 1600, 1000, _proj(), 0.0)
    assert "12.5 units/hour (set by you" in capsys.readouterr().out


def test_comparing_an_epoch_time_from_another_gpu(capsys):
    p = vs.params_from({"compare_epoch_seconds": 2000, "compare_units_per_hour": 8.7}, env={})
    vs.report_projection(p, A100, "flash2", 2, 42, 1600, 1000, _proj(epoch_seconds=3000.0), 0.0)  # A100 40: 3000 s * 5.4/h = 4.5 units
    out = capsys.readouterr().out
    assert "this GPU 4.5 units" in out and "the other 4.8 units" in out and "this GPU is cheaper per epoch" in out
    assert "Break-even rule" in out and "R2/R1" in out and "8.7 / 5.4 = 1.6x" in out
    p2 = vs.params_from({"compare_epoch_seconds": 1000, "compare_units_per_hour": 8.7}, env={})
    vs.report_projection(p2, A100, "flash2", 2, 42, 1600, 1000, _proj(epoch_seconds=3000.0), 0.0)
    assert "the other GPU is cheaper per epoch" in capsys.readouterr().out
    assert vs.default_units_per_hour(A100)[0] == 5.4 and vs.default_units_per_hour(A100_80)[0] == 7.5
    assert vs.default_units_per_hour(G4_96)[0] == 8.7 and vs.default_units_per_hour(H100) is None and vs.default_units_per_hour(G4_48) is None


def test_the_disconnect_reminder_names_both_ways(capsys):
    vs.finish_reminder()
    out = capsys.readouterr().out
    assert "runtime.unassign()" in out and "Disconnect and delete runtime" in out and "colab stop -s qwen-train" in out and "idle" in out


def test_step_timings_show_the_earlier_time_so_the_cache_saving_is_visible(tmp_path, capsys):
    e = Env(tmp_path)
    vs.ensure_layout(e.p, create_run=False)
    with vs.timed("pip packages"):
        pass
    vs._TIMES[-1]["seconds"] = 100.0
    vs.timing_summary(e.lay, "Setup", "setup")
    assert "first time measured" in capsys.readouterr().out
    vs._TIMES.clear()
    with vs.timed("pip packages"):
        pass
    vs._TIMES[-1]["seconds"] = 3.0
    vs.timing_summary(e.lay, "Setup", "setup")
    out = capsys.readouterr().out
    assert "earlier: 1 min 40 s" in out and "3.0 s" in out
    assert vs.fmt_secs(3700) == "1 h 01 min"


# ------------------------------------------------------------------------------------------------ checkpoints: local first, safe sync, retention
def safetensors_bytes(size=3000):
    header = json.dumps({"w": {"dtype": "U8", "shape": [size], "data_offsets": [0, size]}}).encode()
    return len(header).to_bytes(8, "little") + header + b"m" * size


def make_checkpoint(folder, name, size=3000, config=None):
    """A finished checkpoint as sft_12hz.py leaves it: custom_voice config first, valid weights written last."""
    d = os.path.join(folder, name)
    os.makedirs(d)
    open(os.path.join(d, "tokenizer.json"), "w").write("{}")
    open(os.path.join(d, "config.json"), "w").write(json.dumps(config or {"tts_model_type": "custom_voice"}))
    open(os.path.join(d, "model.safetensors"), "wb").write(safetensors_bytes(size))
    return d


def test_a_checkpoint_is_copied_under_a_temporary_name_verified_then_renamed(tmp_path):
    src = make_checkpoint(str(tmp_path), "checkpoint-epoch-0")
    dst = str(tmp_path / "drive" / "checkpoint-epoch-0")
    os.makedirs(os.path.dirname(dst))
    size = vs.copy_verified(src, dst)
    assert size > 3000 and os.path.isfile(os.path.join(dst, vs.DONE_MARKER)) and not os.path.exists(dst + ".tmp")
    assert open(os.path.join(dst, "model.safetensors"), "rb").read() == safetensors_bytes(3000)


def test_a_failed_verification_leaves_nothing_that_looks_valid(tmp_path, monkeypatch):
    src = make_checkpoint(str(tmp_path), "checkpoint-epoch-0")
    dst = str(tmp_path / "drive" / "checkpoint-epoch-0")
    os.makedirs(os.path.dirname(dst))
    real = vs.shutil.copytree

    def truncating(a, b, *args, **kw):
        real(a, b, *args, **kw)
        with open(os.path.join(b, "model.safetensors"), "r+b") as fh:
            fh.truncate(10)

    monkeypatch.setattr(vs.shutil, "copytree", truncating)
    with pytest.raises(vs.StepFailed, match="differs in size"):
        vs.copy_verified(src, dst)
    assert not os.path.exists(dst) and not os.path.exists(dst + ".tmp")
    # a disconnect mid-copy leaves only a .tmp name, which the checkpoint listing ignores
    os.makedirs(dst + ".tmp")
    assert vs.existing_checkpoints(os.path.dirname(dst)) == []


def test_the_syncer_waits_for_a_finished_checkpoint_and_records_each_one(tmp_path, capsys):
    e = Env(tmp_path)
    vs.ensure_layout(e.p)
    t = {"now": 0.0}
    syncer = vs.CheckpointSyncer(e.p, e.lay, vs.Status(e.lay), clock=lambda: t["now"], stable_seconds=6)
    os.makedirs(e.lay.local_checkpoints)
    d = os.path.join(e.lay.local_checkpoints, "checkpoint-epoch-0")
    os.makedirs(d)
    open(os.path.join(d, "config.json"), "w").write(json.dumps({"tts_model_type": "custom_voice"}))
    assert syncer.scan() == []  # model.safetensors (written last) is not there yet
    open(os.path.join(d, "model.safetensors"), "wb").write(safetensors_bytes(100))
    assert syncer.scan() == []  # size just seen
    t["now"] = 3
    assert syncer.scan() == []  # not stable long enough
    t["now"] = 10
    assert syncer.scan() == ["checkpoint-epoch-0"]
    assert os.path.isfile(os.path.join(e.lay.checkpoints, "checkpoint-epoch-0", vs.DONE_MARKER))
    man = json.load(open(e.lay.ckpt_manifest))
    assert "checkpoint-epoch-0" in man["checkpoints"] and json.load(open(e.lay.status))["latest_checkpoint"] == "checkpoint-epoch-0"
    out = capsys.readouterr().out
    assert "on Drive and verified" in out and "First checkpoint is" in out and "all 2 epochs will use about" in out
    assert syncer.scan() == []  # nothing is copied twice


def test_the_first_checkpoint_warns_when_drive_is_too_small(tmp_path, monkeypatch, capsys):
    e = Env(tmp_path, epochs=10)
    vs.ensure_layout(e.p)
    syncer = vs.CheckpointSyncer(e.p, e.lay)
    monkeypatch.setattr(vs.shutil, "disk_usage", lambda p: type("U", (), {"free": 1000, "total": 10, "used": 0})())
    syncer._first_report(5000)
    assert "WARNING: Drive reports" in capsys.readouterr().out


def test_an_existing_drive_checkpoint_is_never_overwritten_without_restart(tmp_path):
    e = Env(tmp_path)
    vs.ensure_layout(e.p)
    make_checkpoint(e.lay.local_checkpoints if os.makedirs(e.lay.local_checkpoints) is None else "", "checkpoint-epoch-0")
    make_checkpoint(e.lay.checkpoints, "checkpoint-epoch-0")  # something already on Drive under that name
    with pytest.raises(vs.StepFailed, match="Nothing was overwritten"):
        vs.CheckpointSyncer(e.p, e.lay).scan(final=True)
    assert os.path.isfile(os.path.join(e.lay.checkpoints, "checkpoint-epoch-0", "tokenizer.json"))
    with pytest.raises(vs.StepFailed, match="already has checkpoints"):
        vs.refuse_overwrite(e.p, e.lay)
    restart = vs.params_from(dict(drive_root=e.drive, local_root=e.content, speaker_name="Sunny", run_id=e.p.run_id, restart=True), env={})
    vs.refuse_overwrite(restart, e.lay)  # explicit: allowed, and said


def _synced_run(tmp_path, epochs=5, **overrides):
    e = Env(tmp_path, epochs=epochs, **overrides)
    vs.ensure_layout(e.p)
    os.makedirs(e.lay.local_checkpoints)
    for n in range(epochs):
        make_checkpoint(e.lay.local_checkpoints, f"checkpoint-epoch-{n}")
    vs.CheckpointSyncer(e.p, e.lay).scan(final=True)
    return e


def test_retention_keeps_everything_by_default_and_only_deletes_what_it_wrote(tmp_path, capsys):
    e = _synced_run(tmp_path, keep_last_n=2)  # KEEP_ALL is on: KEEP_LAST_N is ignored
    assert vs.apply_retention(e.p, e.lay) == [] and len(vs.existing_checkpoints(e.lay.checkpoints)) == 5
    assert "ignored" in capsys.readouterr().out
    foreign = make_checkpoint(e.lay.checkpoints, "checkpoint-epoch-9")  # a folder this script did not write
    prune = vs.params_from(dict(drive_root=e.drive, local_root=e.content, speaker_name="Sunny", run_id=e.p.run_id, keep_all=False, keep_last_n=2,
                                keep_first=True, keep_final=True), env={})
    removed = vs.apply_retention(prune, e.lay)
    assert removed == ["checkpoint-epoch-1", "checkpoint-epoch-2"]
    assert vs.existing_checkpoints(e.lay.checkpoints) == ["checkpoint-epoch-0", "checkpoint-epoch-3", "checkpoint-epoch-4", "checkpoint-epoch-9"]
    assert os.path.isdir(foreign) and "Not touched" in capsys.readouterr().out


def test_retention_refuses_a_policy_that_would_keep_nothing(tmp_path):
    with pytest.raises(vs.StepFailed, match="delete every checkpoint"):
        vs.params_from({"keep_all": False, "keep_final": False}, env={})


# ------------------------------------------------------------------------------------------------ the whole training flow with fakes
def fake_trainer(epochs=2, batches=3, oom=False, fail=False):
    def run(cmd, cwd, env, on_line):
        if str(cmd[2]).endswith("prepare_progress.py"):
            return fake_prepare()(cmd, cwd, env, on_line)
        a = {cmd[i]: cmd[i + 1] for i in range(len(cmd) - 1) if str(cmd[i]).startswith("--")}
        out = a["--output_model_path"]
        assert os.path.samefile(cwd, cwd) and os.path.isfile(os.path.join(cwd, "train.jsonl"))  # runs from inside the dataset folder
        assert os.path.isfile(a["--train_jsonl"]) and a["--init_model_path"].endswith("Qwen3-TTS-12Hz-1.7B-Base")
        import time as _t

        for epoch in range(int(a["--num_epochs"])):
            for step in range(0, batches + 8, 10):
                _t.sleep(0.01)
                if on_line:
                    on_line(f"Epoch {epoch} | Step {step} | Loss: {3.0 - epoch * 0.5:.4f}\n")
            if oom:
                on_line("torch.OutOfMemoryError: CUDA out of memory.\n")
                return 1
            make_checkpoint(out, f"checkpoint-epoch-{epoch}")
        return 1 if fail else 0
    return run


def _flow_env(tmp_path, **overrides):
    e = Env(tmp_path, **overrides)
    e.install_repo_and_models()
    src = make_dataset(tmp_path / "drive_data" / "set", n=6)
    zip_folder(src, tmp_path / "drive_data" / "set.zip")
    e.p.data_source = str(tmp_path / "drive_data" / "set.zip")
    return e


def run_train(e, handler, gpu=A100, monkeypatch=None):
    monkeypatch.setattr(vs, "detect_environment", lambda: dict(ENV))
    return vs.train(e.p, gpu=gpu, stream=FakeStream(handler), detect=lambda: gpu, sample_memory=lambda: 12345.0)


def test_training_end_to_end_with_progress_status_metrics_snapshot_and_safe_checkpoints(tmp_path, monkeypatch, capsys):
    e = _flow_env(tmp_path)
    synced = run_train(e, fake_trainer(), monkeypatch=monkeypatch)
    assert synced == ["checkpoint-epoch-0", "checkpoint-epoch-1"]
    out = capsys.readouterr().out
    # visible progress that proves it is alive
    assert re.search(r"epoch 1/2 +step 0/3 .*loss 3\.0000", out) and "GPU 12.1 GB in use" in out
    assert "Preflight" in out and "ALL CHECKS PASSED" in out and "Training finished" in out
    assert "no TensorBoard" in out and "REMINDER: disconnect and DELETE the runtime" in out
    assert "on Drive and verified" in out and "First checkpoint is" in out
    # everything on Drive in the run folder
    lay = e.lay
    assert sorted(vs.existing_checkpoints(lay.checkpoints)) == ["checkpoint-epoch-0", "checkpoint-epoch-1"]
    status = json.load(open(lay.status))
    assert status["status"] == "done" and status["latest_checkpoint"] == "checkpoint-epoch-1" and status["loss"] is not None and status["updated"]
    metrics = open(lay.metrics).read().splitlines()
    assert metrics[0].startswith("time,epoch,step,loss") and len(metrics) > 2
    snap = json.load(open(os.path.join(lay.environment, "snapshot.json")))
    assert snap["qwen_commit"] == vs.QWEN_COMMIT and snap["outcome"] == "done" and snap["dataset_hash"] and snap["gpu"]["name"] == A100.name
    assert snap["settings"]["epochs"]["source"] == "owner override" and "sft_12hz.py" in snap["commands"]["train"] and "packages" in snap
    assert json.load(open(os.path.join(lay.run_config, "settings.json")))["batch_size"]["value"] == 2
    # codes cached for the next run with the dataset hash
    assert os.path.isfile(lay.codes) and json.load(open(lay.codes_meta))["dataset_hash"] == snap["dataset_hash"]
    assert not any(n.endswith(".tmp") for n in os.listdir(lay.checkpoints))
    # a second run (new id) is a separate folder, reuses the codes, and the first run is untouched
    p2 = vs.params_from(dict(drive_root=e.drive, local_root=str(tmp_path / "content2"), speaker_name="Sunny", epochs=1, batch_size=2, run_id="second",
                             data_source=e.p.data_source, sample_after_training=False), env={})
    os.makedirs(p2.local_root)
    e3 = Env.__new__(Env)
    e3.p, e3.lay, e3.drive, e3.content = p2, vs.Layout(p2), e.drive, p2.local_root
    e3.install_repo_and_models()
    handler = fake_trainer()
    calls = []
    monkeypatch.setattr(vs, "detect_environment", lambda: dict(ENV))
    vs.train(p2, gpu=A100, stream=FakeStream(lambda *a: (calls.append(a[0][2]), handler(*a))[1]), detect=lambda: A100, sample_memory=lambda: 1.0)
    assert not any(str(c).endswith("prepare_progress.py") for c in calls)  # codes reused: hash matches
    assert e3.lay.run != lay.run and vs.existing_checkpoints(lay.checkpoints) == ["checkpoint-epoch-0", "checkpoint-epoch-1"]


def test_an_oom_names_gpu_batch_step_and_the_next_action_and_is_logged_on_drive(tmp_path, monkeypatch, capsys):
    e = _flow_env(tmp_path, batch_size=4)
    with pytest.raises(vs.StepFailed) as err:
        run_train(e, fake_trainer(oom=True), gpu=L4, monkeypatch=monkeypatch)
    msg = str(err.value)
    assert "out of memory" in msg and L4.name in msg and "batch size 4" in msg and "step" in msg and "BATCH_SIZE=2" in msg
    assert json.load(open(e.lay.status))["status"] == "failed"
    assert "FATAL" in open(e.lay.runlog).read() and "out of memory" in open(e.lay.runlog).read()
    assert "REMINDER" in capsys.readouterr().out  # even a failed run reminds you to disconnect


def test_a_failed_run_keeps_the_epochs_that_reached_drive(tmp_path, monkeypatch):
    e = _flow_env(tmp_path)
    with pytest.raises(vs.StepFailed, match="exited with code 1"):
        run_train(e, fake_trainer(fail=True), monkeypatch=monkeypatch)
    assert vs.existing_checkpoints(e.lay.checkpoints) == ["checkpoint-epoch-0", "checkpoint-epoch-1"]


def test_training_refuses_to_start_over_existing_checkpoints_and_unsupported_setups(tmp_path, monkeypatch):
    e = _flow_env(tmp_path, run_id="same")
    vs.ensure_layout(e.p)
    make_checkpoint(e.lay.checkpoints, "checkpoint-epoch-0")
    with pytest.raises(vs.StepFailed, match="already has checkpoints"):
        run_train(e, fake_trainer(), monkeypatch=monkeypatch)
    assert vs.existing_checkpoints(e.lay.checkpoints) == ["checkpoint-epoch-0"]
    e2 = _flow_env(tmp_path / "t4")
    with pytest.raises(vs.StepFailed, match="hard-codes FlashAttention 2"):
        run_train(e2, fake_trainer(), gpu=T4, monkeypatch=monkeypatch)
    with pytest.raises(vs.StepFailed, match="No GPU"):
        vs.train(e2.p, gpu=None, stream=FakeStream(), detect=lambda: None)


def test_preflight_stops_before_gpu_time_when_disk_is_short(tmp_path, monkeypatch):
    e = _flow_env(tmp_path)
    vs.ensure_layout(e.p)
    real = vs.shutil.disk_usage
    monkeypatch.setattr(vs.shutil, "disk_usage", lambda p: type("U", (), {"free": 1024, "total": 10, "used": 0})() if str(p).startswith(e.content) else real(p))
    with pytest.raises(vs.StepFailed, match="Preflight failed"):
        vs.preflight(e.p, e.lay, A100, "flash2", 2)


def test_the_sdpa_path_trains_with_a_patched_copy_and_leaves_the_cached_repo_alone(tmp_path, monkeypatch, capsys):
    e = _flow_env(tmp_path)
    original = open(os.path.join(e.lay.local_repo, "finetuning", "sft_12hz.py")).read()
    seen = []
    handler = fake_trainer()
    run_train(e, lambda *a: (seen.append(list(a[0])), handler(*a))[1], gpu=G4_96, monkeypatch=monkeypatch)
    train_cmd = next(c for c in seen if "--num_epochs" in c)
    assert train_cmd[2].endswith("sft_12hz_patched.py") and os.path.dirname(train_cmd[2]) == e.lay.patched
    assert 'attn_implementation="sdpa"' in open(train_cmd[2]).read() and "UNVERIFIED" in open(train_cmd[2]).read()
    assert open(os.path.join(e.lay.local_repo, "finetuning", "sft_12hz.py")).read() == original
    assert "PATCHED COPY" in capsys.readouterr().out
    # a repo that does not contain the expected line cannot be patched
    open(os.path.join(e.lay.local_repo, "finetuning", "sft_12hz.py"), "w").write("nothing here")
    with pytest.raises(vs.StepFailed, match="cannot be patched safely"):
        vs.build_patched_script(e.lay)


def test_the_timing_run_works_on_both_attention_paths_and_touches_no_real_run(tmp_path, monkeypatch, capsys):
    for gpu, path, rate in ((A100, "FLASH2", "5.4 units/hour"), (G4_96, "SDPA", "8.7 units/hour")):
        e = _flow_env(tmp_path / path, smoke_run=True, smoke_clips=8, epochs=3)
        monkeypatch.setattr(vs, "detect_environment", lambda: dict(ENV))
        handler = fake_trainer(batches=24)
        result = vs.train(e.p, gpu=gpu, stream=FakeStream(lambda *a: handler(*a)), detect=lambda: gpu, sample_memory=lambda: 20 * 1024.0)
        out = capsys.readouterr().out
        assert result == [] and "TIMING RUN RESULT" in out and f"attention path: {path}" in out and rate in out
        assert "Peak GPU memory seen by nvidia-smi on this GPU: 20.0 GB" in out and "Projected ONE full epoch" in out and "Projected ALL 3 EPOCHS" in out
        assert "REMINDER: disconnect and DELETE" in out and "short sample" in out
        timing = vs.Layout(e.p, e.p.run_id + "-timing")
        assert json.load(open(os.path.join(timing.run, "timing.json")))["attention"] == path.lower()
        assert not os.path.exists(e.lay.checkpoints) and not os.path.exists(e.lay.codes)  # no real run, no codes cache
        assert not os.path.exists(os.path.join(e.content, "runs", "Sunny", e.p.run_id + "-timing", "checkpoints"))  # sample checkpoint removed
    # out of memory in the sample tells the owner to lower the batch before the long run
    e = _flow_env(tmp_path / "oom", smoke_run=True, smoke_clips=8, batch_size=4)
    monkeypatch.setattr(vs, "detect_environment", lambda: dict(ENV))
    with pytest.raises(vs.StepFailed, match="BATCH_SIZE=2"):
        vs.train(e.p, gpu=L4, stream=FakeStream(fake_trainer(oom=True)), detect=lambda: L4, sample_memory=lambda: 1.0)


# ------------------------------------------------------------------------------------------------ final sample
def test_final_sample_uses_the_readme_api_from_the_drive_copy_and_says_when_it_cannot_run(tmp_path, capsys):
    e = _synced_run(tmp_path)
    made = vs.generate_samples(e.p, e.lay, stream=FakeStream(lambda cmd, *a: open(cmd[6], "wb").write(b"RIFF") and 0), gpu=A100, probe=lambda n: (True, "ok"))
    assert made == [os.path.join(e.lay.samples, "final.wav")]
    meta = json.load(open(os.path.join(e.lay.samples, "final.json")))
    assert meta["checkpoint"] == "checkpoint-epoch-4" and meta["seed"] == 1234 and meta["text"] == vs.DEFAULT_SAMPLE_TEXT
    for needle in ("Qwen3TTSModel.from_pretrained", "generate_custom_voice(text=text, speaker=speaker)", 'device_map="cuda:0"', "dtype=torch.bfloat16",
                   "attn_implementation=attn", "sf.write"):
        assert needle in vs.SAMPLE_CODE
    for reason, kw in (("no GPU", dict(gpu=None, detect=lambda: None)), ("qwen_tts cannot be imported", dict(gpu=A100, probe=lambda n: (False, "x")))):
        assert vs.generate_samples(e.p, e.lay, stream=FakeStream(lambda *a: pytest.fail("must not run")), **kw) == []
        assert "Sample skipped" in capsys.readouterr().out


# ------------------------------------------------------------------------------------------------ CLI
def test_cli_reads_the_export_settings_and_runs_a_step(tmp_path, capsys, monkeypatch):
    st = tmp_path / "training_settings.json"
    st.write_text(json.dumps({"epochs": 7, "speaker_name": "Sunny", "drive_folder": "F"}), encoding="utf-8")
    assert vs.main(["gpu", "--settings", str(st), "--drive-root", str(tmp_path / "d"), "--epochs", "8"]) == 0
    assert vs.main(["data", "--settings", str(st), "--drive-root", str(tmp_path / "d"), "--data-source", str(tmp_path / "nope")]) == 2
    assert "STOPPED" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        vs.main(["bogus"])


# ------------------------------------------------------------------------------------------------ the notebook
def _notebook(**kw):
    return json.loads(qwen3_colab.render_notebook(**kw))


def test_notebook_is_valid_nbformat_4_with_the_cells_in_order():
    nb = _notebook()
    assert nb["nbformat"] == 4 and nb["nbformat_minor"] >= 5 and nb["metadata"]["accelerator"] == "GPU"
    ids = [c["id"] for c in nb["cells"]]
    assert len(set(ids)) == len(ids) and all(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", i) for i in ids)
    for c in nb["cells"]:
        assert c["cell_type"] in ("markdown", "code") and isinstance(c["source"], list) and all(isinstance(x, str) for x in c["source"])
        assert "".join(c["source"]).strip()
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None
    text = [("".join(c["source"])) for c in nb["cells"]]
    assert "Runtime > Change runtime type" in text[0] and "compute units" in text[0].lower() and "disconnect and delete" in text[0].lower()
    code = [t for c, t in zip(nb["cells"], text) if c["cell_type"] == "code"]
    assert "nvidia-smi" in code[0]
    assert all(v in code[1] for v in ("DRIVE_ROOT", "DATA_SOURCE", "SPEAKER_NAME", "EPOCHS", "LR", "BATCH_SIZE", "BASE_MODEL", "KEEP_LAST_N", "KEEP_ALL",
                                      "ATTN_FALLBACK", "RESTART", "RUN_ID", "SMOKE_RUN", "UNITS_PER_HOUR", "UNITS_AVAILABLE", "COMPARE_EPOCH_SECONDS",
                                      "SAMPLE_TEXT"))
    order = [("drive.mount" in t, "vs.setup(P)" in t, "vs.data_check(P)" in t, "vs.train(P)" in t, "vs.results(P)" in t, "vs.final_sample(P)" in t) for t in code]
    assert [i for row in order for i, v in enumerate(row) if v] == [0, 1, 2, 3, 4, 5]
    assert len(code) <= 9


def _code_lines(source):
    return "\n".join(line for line in source.split("\n") if not line.lstrip().startswith(("!", "%")))


def test_every_notebook_code_cell_parses_ignoring_magics():
    for variant in (_notebook(), _notebook(speaker="Other", folder_name="other-set", settings=qwen3_colab.settings_from({"epochs": 9, "drive_folder": "A/B"}))):
        for c in variant["cells"]:
            if c["cell_type"] == "code":
                ast.parse(_code_lines("".join(c["source"])))


def test_notebook_prefills_the_exports_defaults_and_only_speaker_and_source_change_for_another_speaker():
    st = qwen3_colab.settings_from({"epochs": 9, "learning_rate": 5e-5, "batch_size": 3, "drive_folder": "Tr/x"})
    params = "".join(_notebook(speaker="Sunny", folder_name="set", zip_name="set.zip", settings=st)["cells"][3]["source"])
    ns: dict = {}
    exec(params, ns)
    assert (ns["EPOCHS"], ns["LR"], ns["BATCH_SIZE"], ns["SPEAKER_NAME"]) == (9, 5e-5, 3, "Sunny")
    assert ns["DRIVE_ROOT"] == "/content/drive/MyDrive/Tr/x" and ns["DATA_SOURCE"] == "/content/drive/MyDrive/Tr/x/datasets/set.zip"
    assert ns["KEEP_ALL"] is True and ns["RESTART"] is False and ns["SMOKE_RUN"] is False and ns["ATTN_FALLBACK"] == "auto" and ns["UNITS_AVAILABLE"] == 100
    other = "".join(_notebook(speaker="Other", folder_name="set2", zip_name="set2.zip", settings=st)["cells"][3]["source"])
    diff = [(a, b) for a, b in zip(params.split("\n"), other.split("\n")) if a != b]
    assert {a.split(" =")[0] for a, _ in diff} == {"SPEAKER_NAME", "DATA_SOURCE"}


def test_the_mount_and_load_cell_runs_against_a_fake_colab(tmp_path, monkeypatch):
    """Execute the notebook's bootstrap and cell calls for real, with google.colab faked and Drive in a temp folder."""
    drive_root = str(tmp_path / "MyDrive" / "VoiceStudio-training")
    src = make_dataset(tmp_path / "ds")
    archive = zip_folder(src, tmp_path / "set.zip")
    with zipfile.ZipFile(archive, "a") as zf:
        zf.write(RUNNER_PATH, qwen3_colab.RUNNER_NAME)
    nb = _notebook(folder_name="set")
    cells = {c["id"]: "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"}
    google, colab = type(sys)("google"), type(sys)("google.colab")
    mounted = []
    colab.drive = type("D", (), {"mount": staticmethod(lambda p: mounted.append(p))})
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.delitem(sys.modules, "voicestudio_qwen3_colab", raising=False)
    ns = {"DRIVE_ROOT": drive_root, "DATA_SOURCE": archive, "SPEAKER_NAME": "Sunny", "EPOCHS": 2, "LR": 2e-5, "BATCH_SIZE": 2, "BASE_MODEL": vs.BASE_MODEL,
          "TOKENIZER_MODEL": vs.TOKENIZER, "RUN_ID": "", "KEEP_ALL": True, "KEEP_LAST_N": 0, "KEEP_FIRST": False, "KEEP_FINAL": True,
          "SAMPLE_TEXT": vs.DEFAULT_SAMPLE_TEXT, "SAMPLE_SEED": 1, "CACHE_MODELS_ON_DRIVE": True, "ATTN_FALLBACK": "auto", "RESTART": False,
          "SMOKE_RUN": False, "SMOKE_CLIPS": 40, "UNITS_PER_HOUR": None, "UNITS_AVAILABLE": 100, "COMPARE_EPOCH_SECONDS": None,
          "COMPARE_UNITS_PER_HOUR": None}
    saved_path = list(sys.path)
    try:
        mount_cell = next(t for t in cells.values() if "drive.mount" in t)
        exec(mount_cell, ns)
        assert mounted == ["/content/drive"] and os.path.isfile(os.path.join(drive_root, "cache", qwen3_colab.RUNNER_NAME))
        P = ns["P"]
        assert P.speaker_name == "Sunny" and P.epochs == 2 and P.data_source == archive and ns["vs"].RUNNER_VERSION == vs.RUNNER_VERSION
        # the script is also found from the Drive cache when the data source no longer holds it
        ns2 = dict(ns, DATA_SOURCE=src)
        exec(mount_cell, ns2)
        bad = dict(ns, DRIVE_ROOT=str(tmp_path / "empty"), DATA_SOURCE=src)
        with pytest.raises(FileNotFoundError, match="not found"):
            exec(mount_cell, bad)
    finally:
        sys.path[:] = saved_path
        sys.modules.pop("voicestudio_qwen3_colab", None)


def test_committed_notebook_matches_what_the_generator_produces():
    expected = qwen3_colab.render_notebook()
    assert os.path.isfile(NOTEBOOK_COMMITTED), "notebooks/Qwen3_TTS_Colab_Training.ipynb is missing"
    actual = open(NOTEBOOK_COMMITTED, encoding="utf-8", newline="").read().replace("\r\n", "\n")
    assert actual == expected, ("The committed notebook is out of date. Refresh it with: cd backend && python -c \"from engines.fish_api import "
                                "qwen3_colab as c; c.write_default_notebook('../notebooks/Qwen3_TTS_Colab_Training.ipynb')\"")


def test_generation_is_deterministic():
    assert qwen3_colab.render_notebook() == qwen3_colab.render_notebook()
    assert qwen3_colab.runner_source() == qwen3_colab.runner_source()


def test_a_half_written_or_copied_base_checkpoint_is_never_synced_as_an_epoch(tmp_path):
    e = Env(tmp_path)
    vs.ensure_layout(e.p)
    os.makedirs(e.lay.local_checkpoints)
    # 1. just the copied base model folder: a complete base weights file but a 'base' config
    make_checkpoint(e.lay.local_checkpoints, "checkpoint-epoch-0", config={"tts_model_type": "base"})
    # 2. custom_voice config written, but the weights are the older base copy (mtime older than the config)
    d1 = make_checkpoint(e.lay.local_checkpoints, "checkpoint-epoch-1")
    os.utime(os.path.join(d1, "model.safetensors"), (1, 1))
    # 3. weights truncated mid-write
    d2 = make_checkpoint(e.lay.local_checkpoints, "checkpoint-epoch-2")
    with open(os.path.join(d2, "model.safetensors"), "r+b") as fh:
        fh.truncate(500)
    # 4. a good one
    make_checkpoint(e.lay.local_checkpoints, "checkpoint-epoch-3")
    syncer = vs.CheckpointSyncer(e.p, e.lay)
    assert syncer.scan(final=True) == ["checkpoint-epoch-3"]
    assert vs.existing_checkpoints(e.lay.checkpoints) == ["checkpoint-epoch-3"]
    assert syncer.not_finished == ["checkpoint-epoch-0", "checkpoint-epoch-1", "checkpoint-epoch-2"]
    assert vs.safetensors_complete(os.path.join(d2, "model.safetensors")) is False
    assert vs.safetensors_complete(os.path.join(e.lay.checkpoints, "checkpoint-epoch-3", "model.safetensors")) is True


def test_an_interrupt_still_syncs_the_epochs_that_finished(tmp_path, monkeypatch):
    e = _flow_env(tmp_path)

    def interrupted(cmd, cwd, env, on_line):
        if str(cmd[2]).endswith("prepare_progress.py"):
            return fake_prepare()(cmd, cwd, env, on_line)
        out = cmd[cmd.index("--output_model_path") + 1]
        make_checkpoint(out, "checkpoint-epoch-0")
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_train(e, interrupted, monkeypatch=monkeypatch)
    assert vs.existing_checkpoints(e.lay.checkpoints) == ["checkpoint-epoch-0"]
    assert json.load(open(e.lay.status))["status"] == "interrupted"


def test_interrupting_a_run_stops_the_child_process_too(tmp_path):
    """The notebook's stop button raises in the cell; the training process must not keep running (and billing) after it."""
    import time

    beat = tmp_path / "beat.txt"
    script = tmp_path / "loop.py"
    script.write_text(
        "import sys, time\n"
        "open(sys.argv[1], 'w').write(str(time.time()))\n"
        "print('started', flush=True)\n"
        "while True:\n"
        "    open(sys.argv[1], 'w').write(str(time.time()))\n"
        "    time.sleep(0.05)\n")

    def stop_button(line):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        vs.run_stream([sys.executable, str(script), str(beat)], log_path=str(tmp_path / "x.log"), on_line=stop_button)
    time.sleep(0.6)
    first = beat.read_text()
    time.sleep(0.6)
    assert beat.read_text() == first  # nothing is writing any more: the child is gone


def test_a_finished_command_is_left_alone_by_the_stop_helper(tmp_path):
    script = tmp_path / "done.py"
    script.write_text("print('ok')\n")
    assert vs.run_stream([sys.executable, str(script)], log_path=str(tmp_path / "y.log")) == 0
    vs.stop_process(None)  # harmless


def test_progress_lines_replace_the_raw_step_lines_on_screen(tmp_path, capsys):
    import io

    script = tmp_path / "noisy.py"
    script.write_text("print('Epoch 0 | Step 0 | Loss: 1.0000', flush=True)\nprint('some warning', flush=True)\n")
    seen = []
    code = vs.run_stream([sys.executable, str(script)], log_path=str(tmp_path / "x.log"), on_line=lambda l: (seen.append(l), "Step" in l)[1])
    out = capsys.readouterr().out
    assert code == 0 and "some warning" in out and "Epoch 0 | Step 0" not in out
    assert "Epoch 0 | Step 0" in open(tmp_path / "x.log").read()  # the raw line is still in the log
