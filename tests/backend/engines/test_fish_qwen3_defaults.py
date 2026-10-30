"""One set of Qwen3-TTS training defaults across every layer, so they cannot drift:
export dialog (TypeScript) -> qwen3_colab.TrainSettings -> notebook parameters cell -> runner Params; plus the SFT patch's dimension handling."""
from __future__ import annotations

import os
import re

import pytest

from engines.fish_api import qwen3_colab, qwen3_export
from engines.fish_api import qwen3_colab_runner as runner

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OPTIONS_TS = os.path.join(ROOT, "electron", "src", "renderer", "src", "features", "fish-dataset", "export-options.ts")


def _ui_defaults() -> dict:
    text = open(OPTIONS_TS, encoding="utf-8").read()
    block = text[text.index("export const QWEN_NUMBERS"):text.index("export const QWEN_TEXT")]
    nums = {k: float(v) for k, v in re.findall(r"(\w+):\s*\{\s*def:\s*([0-9.eE+-]+)", block)}
    texts = text[text.index("export const QWEN_TEXT"):]
    folder = re.search(r"drive_folder:\s*\{\s*def:\s*'([^']*)'", texts)[1]
    return {**nums, "drive_folder": folder}


def _notebook_params() -> dict:
    nb = qwen3_colab.render_notebook()
    import json

    cell = next(c for c in json.loads(nb)["cells"] if c["cell_type"] == "code" and "EPOCHS =" in "".join(c["source"]))
    ns: dict = {}
    exec("".join(cell["source"]), ns)
    return ns


def test_every_layer_agrees_on_the_training_defaults():
    ui, st, ns, params = _ui_defaults(), qwen3_colab.TrainSettings(), _notebook_params(), runner.Params()
    # UI -> settings (the dialog only sends what the user changed, so its numbers must equal the backend's)
    assert (ui["epochs"], ui["learning_rate"], ui["batch_size"], ui["drive_folder"]) == (st.epochs, st.learning_rate, st.batch_size, st.drive_folder)
    # settings -> notebook parameters cell
    assert (ns["EPOCHS"], ns["LR"], ns["BATCH_SIZE"], ns["SPEAKER_NAME"]) == (st.epochs, st.learning_rate, st.batch_size, qwen3_colab.DEFAULT_SPEAKER)
    assert ns["DRIVE_ROOT"] == f"/content/drive/MyDrive/{st.drive_folder}"
    assert ns["GRAD_ACCUM"] == qwen3_colab.accum_for(st.batch_size)
    # notebook -> runner Params
    assert (params.epochs, params.lr, params.batch_size, params.grad_accum, params.speaker_name) == \
           (ns["EPOCHS"], ns["LR"], ns["BATCH_SIZE"], ns["GRAD_ACCUM"], ns["SPEAKER_NAME"])
    assert params.drive_root.endswith("/" + runner.DEFAULT_DRIVE_FOLDER) and runner.DEFAULT_DRIVE_FOLDER == st.drive_folder
    assert (params.val_fraction, params.drive_skip_first) == (ns["VAL_FRACTION"], ns["DRIVE_SKIP_FIRST"])
    assert (params.keep_all, params.keep_last_n, params.keep_first, params.keep_final) == (ns["KEEP_ALL"], ns["KEEP_LAST_N"], ns["KEEP_FIRST"], ns["KEEP_FINAL"])
    assert params.cache_models_on_drive == ns["CACHE_MODELS_ON_DRIVE"] and params.sample_seed == ns["SAMPLE_SEED"]
    # the batch x accumulation recipe holds, and the learning rate is the corrected one (the script's own 2e-5 makes noise, issue #39)
    assert st.batch_size * qwen3_colab.accum_for(st.batch_size) == qwen3_colab.CLIPS_PER_UPDATE and st.learning_rate == 2e-6
    assert qwen3_export.Options().trailing_silence_s == 1.0  # 1 s of silence on every training clip, on by default


# ------------------------------------------------------------------ the SFT patch, run for real on small fake tensors
torch = pytest.importorskip("torch")


def _namespace():
    ns = {"torch": torch}
    exec(runner.ALIGN_HELPER.replace("target_speaker_embedding = None", "").replace("def _aligned", "def _aligned", 1), ns)
    exec(runner.VAL_HELPER.replace("target_speaker_embedding = None", ""), ns)
    return ns


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _fake_model(text_dim, codec_dim, vocab=11, groups=16, labels_seen=None):
    nn = torch.nn

    class CodePredictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.ModuleList(nn.Embedding(vocab, codec_dim) for _ in range(groups - 1))
            self.lm = nn.Linear(codec_dim, vocab)

        def get_input_embeddings(self):
            return self.emb

        def forward_finetune(self, inputs_embeds, labels=None):
            assert inputs_embeds.shape[1] == groups and labels is None  # hidden + embeddings of codes 0..14
            return _Obj(logits=self.lm(inputs_embeds[:, 1:]))  # position i+1 (holds code i) predicts code i+1: 15 logits

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.text_embedding, self.codec_embedding = nn.Embedding(vocab, text_dim), nn.Embedding(vocab, codec_dim)

    class Talker(nn.Module):
        def __init__(self):
            super().__init__()
            self.model, self.code_predictor = Inner(), CodePredictor()
            self.text_projection = nn.Linear(text_dim, codec_dim)  # 0.6B: 2048 -> 1024; the 1.7B model: equal sizes, still a real layer
            self.config = _Obj(num_code_groups=groups)
            self.head = nn.Linear(codec_dim, vocab)

        def get_input_embeddings(self):
            return self.model.codec_embedding

        def forward(self, inputs_embeds, attention_mask, labels, output_hidden_states):
            if labels_seen is not None:
                labels_seen.append(labels)
            return _Obj(loss=self.head(inputs_embeds).mean(), hidden_states=[[inputs_embeds]])

    class Model(nn.Module):
        device, dtype = torch.device("cpu"), torch.float32

        def __init__(self):
            super().__init__()
            self.talker = Talker()
            self.speaker_encoder = lambda mels: torch.zeros(mels.shape[0], codec_dim)

    return Model()


def _batch(codec_dim, vocab=11, B=2, T=12, groups=16):
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, vocab, (B, T, 2), generator=g)
    codec_mask = torch.zeros(B, T, dtype=torch.bool)
    codec_mask[:, 5:] = True
    return {"input_ids": ids, "codec_ids": torch.randint(0, vocab, (B, T, groups), generator=g), "ref_mels": torch.zeros(B, 4, 3),
            "text_embedding_mask": torch.ones(B, T, 1), "codec_embedding_mask": torch.ones(B, T, 1), "attention_mask": torch.ones(B, T, dtype=torch.long),
            "codec_0_labels": torch.zeros(B, T, dtype=torch.long), "codec_mask": codec_mask}


@pytest.mark.parametrize("text_dim,codec_dim", [(16, 16), (32, 16)])  # equal (1.7B-like) and 2x wider text embedding (0.6B: 2048 vs 1024)
def test_the_validation_loss_projects_the_text_before_adding_codec_embeddings_at_any_width(text_dim, codec_dim):
    ns = _namespace()
    loss = ns["_val_loss"](_fake_model(text_dim, codec_dim), _batch(codec_dim))
    assert torch.isfinite(loss)  # without text_projection the (T, 32) text embedding cannot be added to the (T, 16) codec embedding
    assert "text_projection(" in runner.VAL_HELPER and "text_projection(" in runner.ALIGN_A_NEW


def test_the_patch_projects_every_text_embedding_and_hard_codes_no_width():
    patched = runner.ALIGN_A_NEW + runner.VAL_HELPER
    assert len(re.findall(r"text_projection\(model\.talker\.model\.text_embedding\(", patched)) == len(re.findall(r"text_embedding\(", patched))
    assert not re.search(r"\b(1024|2048|1536|3072)\b", runner.ALIGN_A_NEW + runner.ALIGN_B_NEW + runner.ALIGN_HELPER + runner.VAL_HELPER)


def test_the_sub_talker_gets_15_labels_for_15_logits_and_no_second_shift():
    ns = _namespace()
    seen: list = []
    batch = _batch(16)
    model = _fake_model(16, 16, labels_seen=seen)
    ns["_val_loss"](model, batch)
    assert torch.equal(seen[0], batch["codec_0_labels"])  # the talker gets the labels unshifted: ForCausalLMLoss shifts once
    # the sub-talker: hidden of frame p-1 and the 16 codes of frame p, for every codec frame p; labels are codes 1..15 against 15 logits, as they are
    ids, hid = batch["codec_ids"], torch.randn(2, 12, 16)
    mask = batch["codec_mask"][:, 1:]
    targets = ids[:, 1:][mask]
    captured = {}
    real = torch.nn.functional.cross_entropy
    torch.nn.functional.cross_entropy = lambda logits, labels, **kw: (captured.update(shape=tuple(logits.shape), labels=labels), real(logits, labels, **kw))[1]
    try:
        logits, loss = ns["_aligned_sub_talker_loss"](model.talker, targets, hid[:, :-1][mask])
    finally:
        torch.nn.functional.cross_entropy = real
    assert logits.shape == (targets.shape[0], 15, 11) and captured["labels"].shape == (targets.shape[0] * 15,) and torch.isfinite(loss)
    assert torch.equal(captured["labels"].reshape(-1, 15), targets[:, 1:])  # head i is scored on code i+1: codes 1..15, none dropped, none repeated


def test_every_layer_agrees_on_the_defaults_of_each_training_mode():
    """Full fine-tune and LoRA have separate constants; neither layer may share or drift. The LoRA learning rate and merge scale are
    UNCONFIRMED placeholders: this test only pins that every layer shows the same number, not that the number is right."""
    import json

    ui, params = _ui_defaults(), runner.Params()
    # full: 2e-6 everywhere, never 1e-6, and never the script's own 2e-5 (the endless-generation cause, issue #39)
    assert runner.DEFAULT_LR == params.lr == ui["learning_rate"] == qwen3_colab.TrainSettings().learning_rate == 2e-6
    assert runner.DEFAULT_TRAIN_MODE == "full" and params.train_mode == "full" and params.speakers == ""
    # lora: its own constant in the runner, the settings, the notebook cell, Params and the UI
    assert runner.LORA_DEFAULT_LR == 2e-6  # owner-confirmed; its own constant even though it equals the full one today
    assert runner.LORA_DEFAULT_SCALE == 0.3 and runner.LORA_SCALE_RANGE == (0.0, 1.0) and runner.LORA_SCALE_WARN_ABOVE == 0.5
    assert runner.params_from({"train_mode": "lora"}, env={}).lr == runner.LORA_DEFAULT_LR == ui["lora_learning_rate"]
    assert qwen3_colab.settings_from({"train_mode": "lora"}).learning_rate == runner.LORA_DEFAULT_LR
    assert (ui["lora_rank"], ui["lora_alpha"], ui["lora_dropout"]) == (runner.DEFAULT_LORA_RANK, runner.DEFAULT_LORA_ALPHA, runner.DEFAULT_LORA_DROPOUT)
    assert (params.lora_rank, params.lora_alpha, params.lora_dropout, params.lora_scale) == (16, 32, 0.05, runner.LORA_DEFAULT_SCALE)
    cell = next(c for c in json.loads(qwen3_colab.render_notebook())["cells"] if c["cell_type"] == "code" and "TRAIN_MODE =" in "".join(c["source"]))
    ns: dict = {}
    exec("".join(cell["source"]), ns)
    assert (ns["TRAIN_MODE"], ns["LR"], ns["LORA_LR"], ns["LORA_RANK"], ns["LORA_ALPHA"], ns["LORA_DROPOUT"], ns["LORA_SCALE"]) == \
           ("full", runner.DEFAULT_LR, runner.LORA_DEFAULT_LR, ui["lora_rank"], ui["lora_alpha"], ui["lora_dropout"], runner.LORA_DEFAULT_SCALE)
    # no sampling defaults are set per mode yet (owner has not chosen): nothing in the runner defaults names temperature, top_k or top_p
    assert not [f for f in runner.Params.__dataclass_fields__ if f in ("temperature", "top_k", "top_p")]
