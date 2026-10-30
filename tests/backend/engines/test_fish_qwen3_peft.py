"""LoRA (PEFT) and multi-speaker training for Qwen3-TTS: the pure helpers (qwen3_peft), runner 21's patch variant, parameters and setup.
No GPU, no Colab. The pinned upstream script (commit 022e286) is committed as a fixture so the patches are checked against the real text."""
from __future__ import annotations

import ast
import json
import os
import re
import types

import pytest

from engines.fish_api import qwen3_colab, qwen3_peft as peft_helpers
from engines.fish_api import qwen3_colab_runner as vs
from engines.qwen3_studio import registry

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
torch = pytest.importorskip("torch")
st = pytest.importorskip("safetensors.torch")


def pinned_script() -> str:
    with open(os.path.join(FIXTURES, "sft_12hz_022e286.py"), encoding="utf-8") as fh:
        return fh.read()


# ------------------------------------------------------------------------------------------------ phase 1: speaker rows
def test_speaker_rows_are_lowercase_spaced_by_20_and_never_collide_with_stock_voices():
    assert peft_helpers.speaker_rows(["Zed", "amy"]) == {"zed": 3000, "amy": 3020}
    assert set(peft_helpers.speaker_rows(["a", "b", "c", "d"]).values()).isdisjoint(vs.STOCK_ROWS)
    with pytest.raises(vs.StepFailed, match="listed twice"):
        peft_helpers.speaker_rows(["Amy", "amy"])  # case-insensitive duplicates
    with pytest.raises(vs.StepFailed, match="at most 4"):
        peft_helpers.speaker_rows(["a", "b", "c", "d", "e"])  # row 3080 would be past the 3072-row table
    with pytest.raises(vs.StepFailed, match="stock voice"):
        peft_helpers.speaker_rows(["a", "b"], stock_rows={3020})
    with pytest.raises(vs.StepFailed, match="letters, digits"):
        peft_helpers.speaker_rows(["bad name"])
    with pytest.raises(vs.StepFailed):
        peft_helpers.speaker_rows([])


def test_the_target_regex_hits_the_decoder_layers_of_the_talker_and_code_predictor_only():
    rx = re.compile(vs.LORA_TARGET_REGEX)
    hit = ["talker.model.layers.0.self_attn.q_proj", "talker.model.layers.27.mlp.down_proj",
           "talker.code_predictor.model.layers.3.self_attn.o_proj", "talker.code_predictor.model.layers.4.mlp.gate_proj"]
    miss = ["speaker_encoder.blocks.0.q_proj", "talker.text_projection.linear_fc1", "talker.codec_head", "talker.code_predictor.lm_head.0",
            "talker.code_predictor.small_to_mtp_projection", "talker.model.codec_embedding", "talker.model.layers.0.self_attn.q_proj.base_layer",
            "talker.model.layers.0.input_layernorm"]
    assert all(rx.fullmatch(n) for n in hit) and not any(rx.fullmatch(n) for n in miss)


def test_a_cached_speaker_embedding_is_computed_once_and_never_again(tmp_path):
    path = str(tmp_path / "emb" / "speaker_embeddings.safetensors")
    calls = []

    def compute(name):
        calls.append(name)
        return torch.full((8,), float(len(calls)))

    first = peft_helpers.cached_speaker_embeddings(path, ["Amy", "bob"], compute)
    again = peft_helpers.cached_speaker_embeddings(path, ["amy", "bob"], compute)
    assert calls == ["amy", "bob"] and torch.equal(first["amy"], again["amy"])  # the second call never touched the encoder
    peft_helpers.cached_speaker_embeddings(path, ["amy", "cy"], compute)
    assert calls == ["amy", "bob", "cy"]  # only the missing speaker is computed
    mean = peft_helpers.average_embeddings([torch.zeros(4), torch.full((4,), 2.0)])
    assert mean.dtype == torch.float32 and torch.equal(mean, torch.ones(4))
    assert peft_helpers.embedding_norm_warning("amy", 12.0) is None and "outside" in peft_helpers.embedding_norm_warning("amy", 1.0)


def test_the_config_writer_lowercases_and_keeps_everything_else():
    base = {"tts_model_type": "base", "tts_model_size": "1b7", "talker_config": {"vocab_size": 3072, "spk_id": {}, "x": 1}}
    cfg = peft_helpers.write_checkpoint_config(base, {"Amy": 3000, "bob": 3020})
    assert cfg["tts_model_type"] == "custom_voice" and cfg["talker_config"]["spk_id"] == {"amy": 3000, "bob": 3020}
    assert cfg["talker_config"]["spk_is_dialect"] == {"amy": False, "bob": False} and cfg["talker_config"]["x"] == 1
    assert base["tts_model_type"] == "base"  # the input is not modified
    with pytest.raises(vs.StepFailed, match="differ only in case"):
        peft_helpers.write_checkpoint_config(base, {"Amy": 3000, "amy": 3020})


# ------------------------------------------------------------------------------------------------ phase 1: merge
def _tiny(dtype=torch.bfloat16, out=8, inp=8, rank=2, b_scale=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    base = {"talker.model.layers.0.self_attn.q_proj.weight": (torch.randn(out, inp, generator=g) * 0.02).to(dtype),
            "talker.model.codec_embedding.weight": torch.zeros(3072, 4, dtype=dtype), "speaker_encoder.fc.weight": torch.ones(2, 2, dtype=dtype)}
    mod = "talker.model.layers.0.self_attn.q_proj"
    adapter = {mod + ".lora_A.weight": torch.randn(rank, inp, generator=g), mod + ".lora_B.weight": torch.randn(out, rank, generator=g) * b_scale}
    return base, adapter


def test_the_merge_adds_the_scaled_delta_in_float32_and_rounds_once():
    base, adapter = _tiny(b_scale=0.5)
    merged, report = peft_helpers.merge_state(base, adapter, 2.0)
    key = "talker.model.layers.0.self_attn.q_proj.weight"
    expected = (base[key].float() + 2.0 * (adapter[key[:-7] + ".lora_B.weight"] @ adapter[key[:-7] + ".lora_A.weight"])).to(torch.bfloat16)
    assert torch.equal(merged[key], expected) and merged[key].dtype == torch.bfloat16 and report["adapted"] == 1 and report["changed_pct"] > 50
    assert torch.equal(base[key], _tiny(b_scale=0.5)[0][key])  # the base state is not modified


def test_an_adapter_that_is_all_zeros_fails_loudly():
    base, adapter = _tiny(b_scale=0.0)
    with pytest.raises(vs.StepFailed, match="every lora_B matrix is zero"):
        peft_helpers.merge_state(base, adapter, 2.0)


def test_a_delta_below_bfloat16_resolution_is_caught_instead_of_silently_erased():
    base, adapter = _tiny(b_scale=1e-7)  # delta about 1e-7 on weights of about 0.02: bf16 spacing there is ~1e-4
    with pytest.raises(vs.StepFailed, match="below the resolution"):
        peft_helpers.merge_state(base, adapter, 2.0)
    merged, report = peft_helpers.merge_state(base, adapter, 2.0, min_changed_pct=0.0)  # the same merge in float32 keeps it
    assert report["changed_pct"] < 0.1
    base32, adapter32 = _tiny(dtype=torch.float32, b_scale=1e-7)
    assert peft_helpers.merge_state(base32, adapter32, 2.0)[1]["changed_pct"] > 50


def test_an_adapter_for_another_model_or_with_a_lone_matrix_is_refused():
    base, adapter = _tiny()
    with pytest.raises(vs.StepFailed, match="no tensor"):
        peft_helpers.merge_state({"other.weight": torch.zeros(2, 2)}, adapter, 2.0)
    with pytest.raises(vs.StepFailed, match="only one of"):
        peft_helpers.merge_state(base, {k: v for k, v in adapter.items() if k.endswith("lora_A.weight")}, 2.0)


def test_a_merged_multi_speaker_folder_is_a_standard_custom_voice_checkpoint(tmp_path):
    base, adapter = _tiny(b_scale=0.5)
    bdir, adir, odir = (str(tmp_path / n) for n in ("base", "adapter", "merged"))
    os.makedirs(os.path.join(bdir, "speech_tokenizer"))
    st.save_file(base, os.path.join(bdir, "model.safetensors"))
    open(os.path.join(bdir, "speech_tokenizer", "x.json"), "w").write("{}")
    open(os.path.join(bdir, "tokenizer.json"), "w").write("{}")
    json.dump({"tts_model_type": "base", "tts_model_size": "1b7", "talker_config": {"vocab_size": 3072}}, open(os.path.join(bdir, "config.json"), "w"))
    os.makedirs(adir)
    st.save_file(adapter, os.path.join(adir, peft_helpers.ADAPTER_FILE))
    json.dump({"r": 2, "lora_alpha": 4}, open(os.path.join(adir, peft_helpers.ADAPTER_CONFIG), "w"))
    rows = {"amy": 3000, "bob": 3020}
    json.dump(rows, open(os.path.join(adir, peft_helpers.SPEAKERS_FILE), "w"))
    st.save_file({"amy": torch.ones(4), "bob": torch.full((4,), 2.0)}, os.path.join(adir, peft_helpers.SPEAKER_EMBEDDINGS_FILE))
    info = peft_helpers.read_adapter_info(adir)
    assert info["rank"] == 2 and info["speakers"] == ["amy", "bob"]
    report = peft_helpers.merge_adapter(bdir, adir, odir, lora_scale=1.0)
    assert report["scale"] == 2.0 and report["speakers"] == ["amy", "bob"] and not os.path.exists(odir + ".tmp")
    found = registry.read_checkpoint(odir)
    assert found["type"] == "custom_voice" and found["speakers"] == ["amy", "bob"]
    out = st.load_file(os.path.join(odir, "model.safetensors"))
    assert not any(k.startswith("speaker_encoder") for k in out)
    table = out["talker.model.codec_embedding.weight"]
    assert torch.equal(table[3000].float(), torch.ones(4)) and torch.equal(table[3020].float(), torch.full((4,), 2.0))
    assert os.path.isfile(os.path.join(odir, "speech_tokenizer", "x.json")) and os.path.isfile(os.path.join(odir, "tokenizer.json"))


def test_the_frame_guard_caps_the_micro_batch_under_8192_and_warns():
    assert vs.FRAME_LIMIT == 8192
    assert peft_helpers.cap_micro_batch(4, 2, 250) == (4, 2, None)  # 4 x 20 s is about 1000 frames: far under
    batch, accum, warn = peft_helpers.cap_micro_batch(16, 1, 1000)
    assert (batch, accum) == (8, 2) and batch * 1000 <= 8192 and "8192" in warn and "#378" in warn
    batch, accum, _ = peft_helpers.cap_micro_batch(4, 2, 3000)
    assert batch * 3000 <= 8192 and batch * accum >= 8
    with pytest.raises(vs.StepFailed, match="over the 8192 frame limit"):
        peft_helpers.cap_micro_batch(1, 1, 9000)


# ------------------------------------------------------------------------------------------------ phase 2: the patch
def _layout(tmp_path, **overrides):
    base = dict(drive_root=str(tmp_path / "MyDrive" / "Qwen3TTS"), local_root=str(tmp_path / "content"), speaker_name="Sunny", epochs=2,
                sample_after_training=False, fast_setup=False, drive_skip_first=0)
    os.makedirs(base["local_root"], exist_ok=True)
    p = vs.params_from({**base, **overrides}, env={})
    lay = vs.Layout(p)
    ft = os.path.join(lay.local_repo, "finetuning")
    os.makedirs(ft)
    with open(os.path.join(ft, "sft_12hz.py"), "w", encoding="utf-8") as fh:
        fh.write(pinned_script())
    return p, lay


def test_every_alignment_and_validation_patch_still_applies_to_the_real_pinned_script(tmp_path):
    p, lay = _layout(tmp_path)
    text = open(vs.build_patched_script(lay, None), encoding="utf-8").read()
    ast.parse(text)
    assert "_build_optimizer" not in text and "_SPK_CACHE" not in text
    assert "attn_implementation" in text and "text_projection(" in text


def test_the_lora_patch_applies_once_to_the_real_script_keeps_the_alignment_and_compiles(tmp_path):
    p, lay = _layout(tmp_path, train_mode="lora")
    text = open(vs.build_patched_script(lay, None, lora_or_speakers=True), encoding="utf-8").read()
    ast.parse(text)  # the whole patched real script is valid Python
    for kept in (vs.ALIGN_A_NEW, vs.ALIGN_B_NEW, vs.ALIGN_C_NEW, vs.VAL_EPOCH_NEW, vs.VAL_SAVE_NEW, vs.VAL_DATA_NEW):
        assert kept in text  # ALIGN_A/B/C and VAL_* are intact
    assert "_build_optimizer(qwen3tts, args)" in text and "AdamW(qwen3tts.model.parameters()" not in text.split("def _build_optimizer")[1].split("def ")[0].replace(
        'return AdamW(qwen3tts.model.parameters(), lr=args.lr, weight_decay=0.01)', "")
    assert "_lora_merge_for_save(state_dict, output_dir, args)" in text and "_SPK_ROWS or {args.speaker_name: 3000}" in text
    assert "batch['speaker_idx']" in text and text.index("_lora_merge_for_save(state_dict") < text.index("save_file(state_dict, save_path)")
    # an upstream script that differs is refused, not guessed at
    path = os.path.join(lay.local_repo, "finetuning", "sft_12hz.py")
    open(path, "w", encoding="utf-8").write(pinned_script().replace("weight_decay=0.01", "weight_decay=0.02"))
    with pytest.raises(vs.StepFailed, match="cannot be applied safely"):
        vs.build_patched_script(lay, None, lora_or_speakers=True)
    assert "_build_optimizer" not in open(vs.build_patched_script(lay, None), encoding="utf-8").read()  # the default full script does not need that line


def test_the_default_full_mode_script_is_unchanged_by_the_lora_machinery(tmp_path):
    p, lay = _layout(tmp_path)
    full = open(vs.build_patched_script(lay, None), encoding="utf-8").read()
    for marker in ("_SPK_", "_lora_merge_for_save", "speaker_idx", "peft"):
        assert marker not in full
    assert "AdamW(qwen3tts.model.parameters(), lr=args.lr, weight_decay=0.01)" in full


def _helper_namespace(**env):
    """LORA_HELPER executed with the names the real script has at that point (a stand-in dataset class and AdamW)."""
    for k, v in env.items():
        os.environ[k] = v
    ns = {"torch": torch, "json": json, "os": os, "TTSDataset": type("TTSDataset", (), {}), "AdamW": torch.optim.AdamW}
    exec(vs.LORA_HELPER.replace("target_speaker_embedding = None", ""), ns)
    return ns


@pytest.fixture
def lora_env(monkeypatch):
    for k, v in {"VS_TRAIN_MODE": "lora", "VS_LORA_RANK": "2", "VS_LORA_ALPHA": "4", "VS_LORA_DROPOUT": "0", "VS_LORA_TARGETS": vs.LORA_TARGET_REGEX,
                 "VS_RUNNER_VERSION": vs.RUNNER_VERSION}.items():
        monkeypatch.setenv(k, v)


def _wrapped_state(base, adapter):
    """A state dict as accelerate hands it over after peft injection: base_layer.* and lora_A.default.* names, adapters in float32."""
    state = {}
    for k, v in base.items():
        state[k[:-7] + ".base_layer.weight" if k == "talker.model.layers.0.self_attn.q_proj.weight" else k] = v
    for k, v in adapter.items():
        mod, rest = k.split(".lora_")
        state[f"{mod}.lora_{rest[0]}.default.weight"] = v
    return state


def test_the_script_side_merge_equals_the_tested_helper_and_writes_the_adapter(tmp_path, lora_env):
    ns = _helper_namespace()
    base, adapter = _tiny(b_scale=0.5)
    state = _wrapped_state(base, adapter)
    args = types.SimpleNamespace(init_model_path="/m/base", lr=2e-5)
    merged = ns["_lora_merge_for_save"](dict(state), str(tmp_path), args)
    reference, _ = peft_helpers.merge_state(base, adapter, 2.0)
    assert set(merged) == set(reference) and all(torch.equal(merged[k], reference[k]) for k in reference)  # same maths in both copies
    assert not any("lora_" in k or "base_layer" in k for k in merged)  # the merged tensors have the standard names
    saved = st.load_file(str(tmp_path / "adapter_model.safetensors"))
    assert set(saved) == set(adapter) and all(v.dtype == torch.float32 for v in saved.values())
    cfg = json.load(open(tmp_path / "adapter_config.json"))
    assert (cfg["r"], cfg["lora_alpha"]) == (2, 4.0) and os.path.isfile(tmp_path / "lora_meta.json") and os.path.isfile(tmp_path / "speakers.json")
    # an untrained adapter is refused before anything is promoted
    zero_base, zero_adapter = _tiny(b_scale=0.0)
    with pytest.raises(RuntimeError, match="all zeros"):
        ns["_lora_merge_for_save"](_wrapped_state(zero_base, zero_adapter), str(tmp_path / "z"), args)


def test_full_mode_leaves_the_state_dict_alone(monkeypatch):
    monkeypatch.setenv("VS_TRAIN_MODE", "full")
    ns = _helper_namespace()
    state = {"a.weight": torch.zeros(1)}
    assert ns["_lora_merge_for_save"](state, "/nowhere", None) is state
    model = torch.nn.Linear(2, 2)
    opt = ns["_build_optimizer"](types.SimpleNamespace(model=model), types.SimpleNamespace(lr=1e-3))
    assert [g["weight_decay"] for g in opt.param_groups] == [0.01] and len(opt.param_groups[0]["params"]) == 2  # upstream optimizer, whole model


def test_speaker_embeddings_are_cached_per_speaker_and_the_encoder_runs_once(tmp_path, monkeypatch):
    monkeypatch.setenv("VS_SPEAKERS", json.dumps(["amy", "bob"]))
    monkeypatch.setenv("VS_SPEAKER_ROWS", json.dumps({"amy": 3000, "bob": 3020}))
    monkeypatch.setenv("VS_SPEAKER_CACHE_DIR", str(tmp_path / "cache"))
    ns = _helper_namespace()
    calls = []

    class Enc(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

        def forward(self, mel):
            calls.append(float(mel.mean()))
            return torch.full((1, 6), float(mel.mean()))

    class DS:
        def _load_audio_to_np(self, ref):
            return torch.full((4,), float(ref[-1])).numpy(), 24000

        def extract_mels(self, audio, sr):
            return torch.as_tensor(audio).reshape(1, 1, -1)

    model = types.SimpleNamespace(model=types.SimpleNamespace(speaker_encoder=Enc()))
    rows = [{"speaker": "amy", "ref_audio": "a1"}, {"speaker": "amy", "ref_audio": "a3"}, {"speaker": "bob", "ref_audio": "b5"}]
    ns["_setup_speakers"](model, rows, DS(), None)
    cache = ns["_SPK_CACHE"]
    assert cache.shape == (2, 6) and cache.dtype == torch.float32 and calls and ns["_SPK_ROWS"] == {"amy": 3000, "bob": 3020}
    assert cache[0, 0] != cache[1, 0]  # different references, different vectors; amy is the mean of her two clips
    n_calls = len(calls)
    ns2 = _helper_namespace()
    ns2["_setup_speakers"](model, rows, DS(), None)
    assert len(calls) == n_calls and torch.equal(ns2["_SPK_CACHE"], cache)  # the second start reads the cache file, no encoder run
    # the dataset hands each sample's speaker index to the collate function
    base_cls = type("Base", (), {"__init__": lambda self, rows: setattr(self, "data_list", rows),
                                 "__getitem__": lambda self, i: {"x": i}, "collate_fn": lambda self, b: {"n": len(b)}})
    ns3 = {"torch": torch, "json": json, "os": os, "TTSDataset": base_cls, "AdamW": torch.optim.AdamW}
    exec(vs.LORA_HELPER.replace("target_speaker_embedding = None", ""), ns3)
    ns3["_SPK_INDEX"].update({"amy": 0, "bob": 1})
    ds = ns3["TTSDataset"](rows)
    out = ds.collate_fn([ds[0], ds[2]])
    assert out["speaker_idx"].tolist() == [0, 1] and out["n"] == 2


def test_lora_adapters_are_float32_trained_alone_and_leave_the_speaker_encoder_and_heads_frozen(lora_env):
    pytest.importorskip("peft")  # the Colab stack (transformers 4.57.3 + peft 0.17.1); skipped where peft cannot import
    nn = torch.nn

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj, self.up_proj = nn.Linear(8, 8), nn.Linear(8, 8)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.talker = nn.Module()
            self.talker.model = nn.Module()
            self.talker.model.layers = nn.ModuleList([nn.ModuleDict({"self_attn": nn.ModuleDict({"q_proj": nn.Linear(8, 8)}),
                                                                       "mlp": nn.ModuleDict({"up_proj": nn.Linear(8, 8)})})])
            self.talker.codec_head = nn.Linear(8, 8)
            self.speaker_encoder = nn.Linear(8, 8)

    model = M().to(torch.bfloat16)
    ns = _helper_namespace()
    opt = ns["_build_optimizer"](types.SimpleNamespace(model=model), types.SimpleNamespace(lr=2e-5))
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n in trainable) and all(p.dtype == torch.float32 for n, p in model.named_parameters() if "lora_" in n)
    assert model.speaker_encoder.weight.requires_grad is False and model.talker.codec_head.weight.requires_grad is False
    assert opt.param_groups[0]["weight_decay"] == 0.0


# ------------------------------------------------------------------------------------------------ phase 2: parameters and setup
def test_lora_has_its_own_learning_rate_and_full_keeps_2e6():
    assert vs.DEFAULT_LR == 2e-6 and vs.Params().lr == 2e-6 and vs.Params().train_mode == "full"
    assert vs.LORA_DEFAULT_LR == 2e-6
    assert vs.params_from({"train_mode": "lora"}, env={}).lr == vs.LORA_DEFAULT_LR
    assert vs.params_from({"train_mode": "lora", "lr": 5e-5}, env={}).lr == 5e-5  # an explicit value always wins
    assert vs.params_from({"train_mode": "full"}, env={}).lr == 2e-6
    assert vs.params_from({"train_mode": "lora"}, env={"LR": "3e-5"}).lr == 3e-5
    with pytest.raises(vs.StepFailed, match="TRAIN_MODE"):
        vs.params_from({"train_mode": "qlora"}, env={})
    with pytest.raises(vs.StepFailed, match="LORA_RANK"):
        vs.params_from({"lora_rank": 0}, env={})
    with pytest.raises(vs.StepFailed, match="at most 4"):
        vs.params_from({"speakers": "a,b,c,d,e"}, env={})


def test_peft_is_installed_without_dependencies_only_for_lora_and_probed_in_the_setup_table(tmp_path):
    for mode, expect in (("lora", True), ("full", False)):
        p, lay = _layout(tmp_path / mode, train_mode=mode)
        cmds = []
        vs.ensure_deps(p, lay, "log", probe=lambda m: (m != "peft", "x"), stream=lambda cmd, **kw: cmds.append(cmd) or 0)
        peft_cmds = [c for c in cmds if vs.PEFT_PIN in c]
        assert bool(peft_cmds) is expect
        if expect:
            assert "--no-deps" in peft_cmds[0] and vs.PEFT_PIN == "peft==0.17.1"
        rows = vs.dependency_rows(p, lay, {"python_version": "3.13", "torch": "2.11"}, None, probe=lambda m: (True, "ok"), peft_probe=lambda: (True, "peft 0.17.1"))
        assert ("peft (LoRA)" in [r[0] for r in rows]) is expect
    p, lay = _layout(tmp_path / "bad", train_mode="lora")
    rows = vs.dependency_rows(p, lay, {"python_version": "3.13", "torch": "2.11"}, None, probe=lambda m: (True, "ok"), peft_probe=lambda: (False, "HybridCache"))
    assert ("peft (LoRA)", "FAIL", "HybridCache") in rows


def test_speakers_come_from_the_data_and_the_training_environment_carries_the_rows(tmp_path):
    p, lay = _layout(tmp_path, train_mode="lora", speaker_name="Narr")
    jsonl = tmp_path / "train.jsonl"
    jsonl.write_text("\n".join(json.dumps({"audio": "a", "text": "t", "ref_audio": "r", "speaker": s}) for s in ("Amy", "bob", "Amy")))
    assert vs.dataset_speakers(str(jsonl)) == (["amy", "bob"], True)
    assert vs.resolve_speakers(p, str(jsonl)) == ["amy", "bob"]
    plain = tmp_path / "plain.jsonl"
    plain.write_text(json.dumps({"audio": "a", "text": "t", "ref_audio": "r"}))
    assert vs.resolve_speakers(p, str(plain)) == ["narr"]  # LoRA with one speaker: the SPEAKER_NAME at row 3000
    full, _ = _layout(tmp_path / "f", train_mode="full")
    assert vs.resolve_speakers(full, str(plain)) == []  # full + one speaker: the upstream path, unchanged
    mixed = tmp_path / "mixed.jsonl"
    mixed.write_text(json.dumps({"speaker": "a"}) + "\n" + json.dumps({"speaker": "b"}) + "\n" + json.dumps({"x": 1}))
    with pytest.raises(vs.StepFailed, match="every row"):
        vs.resolve_speakers(p, str(mixed))
    p.speakers = "amy,bob"
    env = vs.plan_commands(p, lay, "/d", 2)["env"]
    assert env["VS_TRAIN_MODE"] == "lora" and json.loads(env["VS_SPEAKER_ROWS"]) == {"amy": 3000, "bob": 3020} and env["VS_LORA_RANK"] == "16"
    assert env["VS_LORA_TARGETS"] == vs.LORA_TARGET_REGEX
    full_env = vs.plan_commands(full, vs.Layout(full), "/d", 2)["env"]
    assert "VS_SPEAKERS" not in full_env and "VS_LORA_RANK" not in full_env
    codes = tmp_path / "codes.jsonl"
    codes.write_text(json.dumps({"audio_codes": [[0] * 16] * 30}) + "\n" + json.dumps({"audio_codes": [[0] * 16] * 50}) + "\n")
    assert vs.max_code_frames(str(codes)) == 50
    assert vs.model_speaker_name(p) == "amy"  # a multi-speaker model is sampled with one of its own names


def test_the_notebook_parameters_cell_roundtrips_the_lora_fields_and_pins_peft_in_the_runner():
    cell = next(c for c in json.loads(qwen3_colab.render_notebook())["cells"] if c["cell_type"] == "code" and "TRAIN_MODE =" in "".join(c["source"]))
    ns: dict = {}
    exec("".join(cell["source"]), ns)
    assert (ns["TRAIN_MODE"], ns["LR"], ns["LORA_LR"], ns["LORA_RANK"], ns["LORA_ALPHA"], ns["LORA_DROPOUT"], ns["SPEAKERS"]) == \
           ("full", vs.DEFAULT_LR, vs.LORA_DEFAULT_LR, 16, 32.0, 0.05, "")
    lora = qwen3_colab.settings_from({"train_mode": "lora"})
    assert lora.learning_rate == vs.LORA_DEFAULT_LR and qwen3_colab.settings_from({}).learning_rate == vs.DEFAULT_LR
    nb = qwen3_colab.render_notebook(settings=lora)
    ns2: dict = {}
    exec("".join(next(c for c in json.loads(nb)["cells"] if c["cell_type"] == "code" and "TRAIN_MODE =" in "".join(c["source"]))["source"]), ns2)
    assert ns2["TRAIN_MODE"] == "lora" and ns2["LORA_LR"] == vs.LORA_DEFAULT_LR and ns2["LR"] == vs.DEFAULT_LR
    assert "lr=LORA_LR if TRAIN_MODE == 'lora' else LR" in nb and "--no-deps" in open(vs.__file__, encoding="utf-8").read() and "peft==0.17.1" in open(vs.__file__, encoding="utf-8").read()
    assert "--train-mode lora" in qwen3_colab.runner_args(lora, "x", "/d")


def test_the_merge_scale_multiplies_alpha_over_r_and_the_rounding_check_runs_at_the_chosen_scale(tmp_path, lora_env, monkeypatch):
    assert vs.LORA_DEFAULT_SCALE == 0.3 and 0.2 in vs.LORA_SCALE_SWEEP and vs.Params().lora_scale == 0.3
    base, adapter = _tiny(b_scale=1e-3)  # delta about 2e-3 x scale on weights of about 0.02
    key = "talker.model.layers.0.self_attn.q_proj.weight"
    full, _ = peft_helpers.merge_state(base, adapter, 2.0 * 1.0)
    cut, _ = peft_helpers.merge_state(base, adapter, 2.0 * 0.3, min_changed_pct=0.0)
    assert (full[key].float() - base[key].float()).abs().mean() > (cut[key].float() - base[key].float()).abs().mean()
    base2, adapter2 = _tiny(b_scale=1e-5)  # survives at scale 1.0 only barely, not at 0.3: the check must use the chosen scale
    ok = peft_helpers.merge_state(base2, adapter2, 2.0, min_changed_pct=0.0)[1]["changed_pct"]
    low = peft_helpers.merge_state(base2, adapter2, 2.0 * 0.3, min_changed_pct=0.0)[1]["changed_pct"]
    assert low < ok
    # the script-side merge reads the same scale from VS_LORA_SCALE
    monkeypatch.setenv("VS_LORA_SCALE", "0.3")
    ns = _helper_namespace()
    base3, adapter3 = _tiny(b_scale=0.5)
    merged = ns["_lora_merge_for_save"](_wrapped_state(base3, adapter3), str(tmp_path), types.SimpleNamespace(init_model_path="b", lr=1e-5))
    ref, _ = peft_helpers.merge_state(base3, adapter3, 2.0 * 0.3)
    assert torch.equal(merged[key], ref[key]) and json.load(open(tmp_path / "adapter_config.json"))["merge_scale"] == 0.3
    assert vs.params_from({"train_mode": "lora", "lora_scale": 0.3}, env={}).lora_scale == 0.3
    assert vs.plan_commands(vs.params_from({"train_mode": "lora", "lora_scale": 0.3}, env={}), vs.Layout(vs.Params(drive_root="/content/drive/MyDrive/Q")), "/d", 2)["env"]["VS_LORA_SCALE"] == "0.3"
    with pytest.raises(vs.StepFailed, match="LORA_SCALE"):
        vs.params_from({"lora_scale": 1.5}, env={})


# ------------------------------------------------------------------------------------------------ export mode switch and manifest guard
def test_manifest_guard_refuses_the_other_mode_in_both_directions(tmp_path):
    (tmp_path / "manifest.json").write_text('{"training_mode": "lora"}')
    vs.check_manifest(str(tmp_path), "lora")
    with pytest.raises(vs.StepFailed, match="training_mode=lora"):
        vs.check_manifest(str(tmp_path), "full")
    (tmp_path / "manifest.json").write_text('{"training_mode": "full"}')
    vs.check_manifest(str(tmp_path), "full")
    with pytest.raises(vs.StepFailed, match="training_mode=full"):
        vs.check_manifest(str(tmp_path), "lora")
    (tmp_path / "manifest.json").write_text("{nope")
    with pytest.raises(vs.StepFailed, match="unreadable"):
        vs.check_manifest(str(tmp_path), "full")
    (tmp_path / "manifest.json").unlink()
    vs.check_manifest(str(tmp_path), "full")  # an older export has no manifest


def test_training_mode_alias_and_merge_scale_limits():
    from engines.fish_api import qwen3_colab
    assert qwen3_colab.settings_from({"training_mode": "lora"}).train_mode == "lora"
    assert qwen3_colab.settings_from({"training_mode": "lora"}).learning_rate == 2e-6
    assert qwen3_colab.settings_from({}).train_mode == "full"
    with pytest.raises(ValueError):
        qwen3_colab.settings_from({"training_mode": "both"})
    with pytest.raises(ValueError):
        qwen3_colab.settings_from({"training_mode": "lora", "train_mode": "full"})
