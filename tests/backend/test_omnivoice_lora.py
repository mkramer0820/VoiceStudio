"""The in-repo LoRA used by the OmniVoice trainer: CPU only, tiny models, no downloads."""
from __future__ import annotations

import json
import os

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from omnivoice.training import lora  # noqa: E402
from omnivoice.training.config import TrainingConfig  # noqa: E402


class _Block(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        return x + self.v_proj(torch.tanh(self.q_proj(self.norm(x))))


class _Llm(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.layers = nn.ModuleList([_Block(d), _Block(d)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class _Tiny(nn.Module):
    """Same shape as OmniVoice: an ``llm`` backbone plus an ``audio_heads`` projection."""

    def __init__(self, d=16):
        super().__init__()
        torch.manual_seed(0)
        self.llm = _Llm(d)
        self.audio_heads = nn.Linear(d, 8, bias=False)

    def forward(self, x):
        return self.audio_heads(self.llm(x))


def _data():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(32, 5, 16, generator=g)
    y = torch.randn(32, 5, 8, generator=g)
    return x, y


def test_a_fresh_adapter_changes_nothing_and_only_the_adapter_trains():
    model = _Tiny()
    x, _ = _data()
    before = model(x)
    report = lora.apply_lora(model, lora.LoraSpec(r=4, alpha=8, dropout=0.0, target_modules=["q_proj", "v_proj"]))
    torch.testing.assert_close(model(x), before)  # B starts at zero
    assert report.wrapped == [f"llm.layers.{i}.{n}" for i in (0, 1) for n in ("q_proj", "v_proj")]
    assert report.lora_params == 4 * (4 * 16 + 16 * 4) and report.extra_trainable_params == 0
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(n.endswith((".lora_A", ".lora_B")) for n in trainable)
    assert not model.audio_heads.weight.requires_grad
    with pytest.raises(ValueError, match="already applied"):
        lora.apply_lora(model, lora.LoraSpec())


def test_it_refuses_targets_that_match_nothing():
    with pytest.raises(ValueError, match="No layers matched"):
        lora.apply_lora(_Tiny(), lora.LoraSpec(target_modules=["nope_proj"]))
    with pytest.raises(ValueError, match="train_modules"):
        lora.apply_lora(_Tiny(), lora.LoraSpec(target_modules=["q_proj"], train_modules=["ghost"]))


def _train(model, steps=60):
    x, y = _data()
    opt = torch.optim.AdamW(lora.trainable_parameters(model), lr=5e-3)
    first = last = None
    for _ in range(steps):
        loss = nn.functional.mse_loss(model(x), y)
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
        last = loss.item()
    return first, last


def test_training_moves_only_adapter_weights_and_lowers_the_loss():
    model = _Tiny()
    frozen = {n: p.detach().clone() for n, p in model.named_parameters()}
    lora.apply_lora(model, lora.LoraSpec(r=4, alpha=8, dropout=0.0, target_modules=["q_proj", "v_proj"],
                                         train_modules=["audio_heads"]))
    first, last = _train(model)
    assert last < first * 0.9
    for n, p in model.named_parameters():
        if n.endswith((".lora_A", ".lora_B")) or n.startswith("audio_heads"):
            continue
        key = n.replace(".base.", ".")
        assert torch.equal(p, frozen[key]), f"{n} changed"
    assert not torch.equal(model.audio_heads.weight, frozen["audio_heads.weight"])


def test_adapter_round_trip_reproduces_the_trained_model(tmp_path):
    trained = _Tiny()
    lora.apply_lora(trained, lora.LoraSpec(r=4, alpha=8, dropout=0.0, target_modules=["q_proj"],
                                           train_modules=["audio_heads"]))
    _train(trained, steps=20)
    out = lora.save_adapter(trained, str(tmp_path / "adapter"), {"note": "x"})
    meta = json.load(open(os.path.join(out, lora.ADAPTER_CONFIG)))
    assert meta["r"] == 4 and meta["note"] == "x" and meta["tensors"] == 2 * 2 + 1
    fresh = _Tiny()  # same base weights (seeded), no adapter yet
    report = lora.load_adapter(fresh, out)
    assert report.lora_params == 2 * 4 * (16 + 16)
    x, _ = _data()
    trained.eval(), fresh.eval()
    torch.testing.assert_close(fresh(x), trained(x))
    # the file holds only the changed tensors, not the base model
    assert set(lora.adapter_state_dict(trained)) == {
        k for k in trained.state_dict() if k.endswith((".lora_A", ".lora_B")) or k.startswith("audio_heads")}


def test_loading_into_a_different_model_fails_loudly(tmp_path):
    model = _Tiny()
    lora.apply_lora(model, lora.LoraSpec(r=2, target_modules=["q_proj"]))
    out = lora.save_adapter(model, str(tmp_path / "a"))
    meta_path = os.path.join(out, lora.ADAPTER_CONFIG)
    meta = json.load(open(meta_path))
    meta["target_modules"] = ["v_proj"]  # config says v_proj, weights are for q_proj
    json.dump(meta, open(meta_path, "w"))
    with pytest.raises(ValueError, match="does not match"):
        lora.load_adapter(_Tiny(), out)
    meta["format"] = 99
    json.dump(meta, open(meta_path, "w"))
    with pytest.raises(ValueError, match="Unsupported adapter format"):
        lora.load_adapter(_Tiny(), out)


def test_merge_gives_the_same_outputs_as_the_wrapped_model():
    model = _Tiny()
    lora.apply_lora(model, lora.LoraSpec(r=4, alpha=8, dropout=0.0, target_modules=["q_proj", "v_proj"]))
    _train(model, steps=20)
    x, _ = _data()
    model.eval()
    wrapped = model(x).detach()
    assert lora.merge_adapter(model) == 4
    assert not any(isinstance(m, lora.LoRALinear) for m in model.modules())
    assert all(p.requires_grad for p in model.parameters())
    torch.testing.assert_close(model(x).detach(), wrapped, rtol=1e-4, atol=1e-5)
    assert lora.merge_adapter(model) == 0


def test_works_on_a_real_tiny_qwen3_backbone():
    try:
        import transformers

        cfg = transformers.Qwen3Config(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                       num_attention_heads=4, num_key_value_heads=2, head_dim=8, vocab_size=50,
                                       max_position_embeddings=64)
        llm = transformers.AutoModel.from_config(cfg, attn_implementation="eager")
    except (ImportError, OSError) as exc:  # e.g. torchaudio fails to load on some hosts
        pytest.skip(f"transformers is not usable here: {exc}")

    class Wrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.llm = llm
            self.audio_heads = nn.Linear(32, 8, bias=False)

    model = Wrap()
    report = lora.apply_lora(model, lora.LoraSpec(r=4, alpha=8))
    assert len(report.wrapped) == 2 * 7  # q k v o + gate up down per layer
    assert report.trainable_params == report.lora_params < report.total_params
    emb = torch.randn(2, 6, 32)
    out = model.llm(inputs_embeds=emb)[0]
    assert out.shape == (2, 6, 32)
    out.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in model.named_parameters() if n.endswith("lora_B"))


def test_training_config_keeps_working_with_and_without_the_new_fields(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"steps": 10, "something_unknown": 1}))
    cfg = TrainingConfig.from_json(str(path))
    assert cfg.steps == 10 and cfg.lora_enabled is False and cfg.lora_r == 16
    path.write_text(json.dumps({"lora_enabled": True, "lora_r": 8, "lora_train_modules": ["audio_heads"]}))
    cfg = TrainingConfig.from_json(str(path))
    assert cfg.lora_enabled and cfg.lora_r == 8 and cfg.lora_train_modules == ["audio_heads"]
    assert cfg.lora_target_modules[0] == "q_proj"
