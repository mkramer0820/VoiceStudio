#!/usr/bin/env python3
# Copyright    2026  VoiceStudio contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A small, dependency-free LoRA for the OmniVoice backbone.

``peft`` is not a dependency of this repo, so the adapter path lives here in plain torch:

- ``apply_lora(model, ...)`` replaces the matching ``nn.Linear`` layers of ``model.llm`` with
  ``LoRALinear`` (frozen base weight plus a low-rank update) and freezes everything else.
  ``B`` starts at zero, so a freshly wrapped model computes exactly what the base model did.
- ``save_adapter`` / ``load_adapter`` store only the trainable tensors (the LoRA matrices and
  any extra modules you chose to train) in ``adapter_model.safetensors`` plus
  ``adapter_config.json``; the base checkpoint is never copied.
- ``merge_adapter`` folds the update into the base weights and unwraps the layers, giving an
  ordinary OmniVoice that ``OmniVoice.from_pretrained`` can save and load.

Key names are module paths relative to the model passed in (``llm.layers.0.self_attn.q_proj``),
so an adapter saved from one run loads into any model built the same way.
"""

import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from torch import nn

ADAPTER_WEIGHTS = "adapter_model.safetensors"
ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_FORMAT = 1
DEFAULT_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


class LoRALinear(nn.Module):
    """``y = base(x) + (dropout(x) @ A^T @ B^T) * alpha / r`` with ``base`` frozen."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.r = r
        self.alpha = float(alpha)
        self.scaling = self.alpha / r
        self.lora_A = nn.Parameter(
            torch.empty(r, base.in_features, dtype=base.weight.dtype, device=base.weight.device)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(base.out_features, r, dtype=base.weight.dtype, device=base.weight.device)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.merged:
            return out
        update = nn.functional.linear(self.dropout(x), self.lora_A.to(x.dtype))
        update = nn.functional.linear(update, self.lora_B.to(x.dtype))
        return out + update * self.scaling

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        """Fold the update into the base weight and return the plain layer."""
        delta = (self.lora_B @ self.lora_A) * self.scaling
        self.base.weight.add_(delta.to(self.base.weight.dtype))
        self.merged = True
        return self.base


@dataclass
class LoraSpec:
    r: int = 16
    alpha: float = 32.0
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: list(DEFAULT_TARGETS))
    # Extra top-level modules trained in full next to the adapter (e.g. "audio_heads").
    train_modules: List[str] = field(default_factory=list)
    root: str = "llm"  # attribute of the model whose Linear layers are wrapped

    def to_json(self) -> dict:
        return {"format": ADAPTER_FORMAT, **asdict(self)}


@dataclass
class LoraReport:
    wrapped: List[str]
    lora_params: int
    extra_trainable_params: int
    total_params: int

    @property
    def trainable_params(self) -> int:
        return self.lora_params + self.extra_trainable_params


def _children_to_wrap(root: nn.Module, targets: Sequence[str]) -> List[str]:
    wanted = set(targets)
    return [
        name
        for name, module in root.named_modules()
        if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in wanted
    ]


def _swap(root: nn.Module, dotted: str, new: nn.Module) -> None:
    parent = root
    *path, leaf = dotted.split(".")
    for part in path:
        parent = getattr(parent, part)
    setattr(parent, leaf, new)


def apply_lora(model: nn.Module, spec: LoraSpec) -> LoraReport:
    """Wrap the target layers, freeze the rest, and report what will train."""
    if getattr(model, "_lora_spec", None) is not None:
        raise ValueError("LoRA is already applied to this model")
    root = getattr(model, spec.root, None)
    if not isinstance(root, nn.Module):
        raise ValueError(f"The model has no module called {spec.root!r} to attach LoRA to")
    names = _children_to_wrap(root, spec.target_modules)
    if not names:
        raise ValueError(
            "No layers matched the LoRA target modules "
            f"{list(spec.target_modules)}; check the names against the model."
        )
    missing = [m for m in spec.train_modules if not isinstance(getattr(model, m, None), nn.Module)]
    if missing:
        raise ValueError(f"train_modules not found on the model: {missing}")
    for p in model.parameters():
        p.requires_grad_(False)
    for dotted in names:
        base = root.get_submodule(dotted)
        _swap(root, dotted, LoRALinear(base, spec.r, spec.alpha, spec.dropout))
    for m in spec.train_modules:
        for p in getattr(model, m).parameters():
            p.requires_grad_(True)
    model._lora_spec = spec
    lora_params = sum(
        p.numel() for n, p in model.named_parameters() if n.endswith((".lora_A", ".lora_B"))
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return LoraReport(
        wrapped=[f"{spec.root}.{n}" for n in names],
        lora_params=lora_params,
        extra_trainable_params=trainable - lora_params,
        total_params=sum(p.numel() for p in model.parameters()),
    )


def adapter_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Only what training changed: LoRA matrices plus any fully trained extra modules."""
    spec: Optional[LoraSpec] = getattr(model, "_lora_spec", None)
    if spec is None:
        raise ValueError("LoRA is not applied to this model")
    extras = tuple(f"{m}." for m in spec.train_modules)
    return {
        k: v.detach().cpu().contiguous()
        for k, v in model.state_dict().items()
        if k.endswith((".lora_A", ".lora_B")) or k.startswith(extras)
    }


def save_adapter(model: nn.Module, out_dir: str, extra_meta: Optional[dict] = None) -> str:
    from safetensors.torch import save_file

    spec: LoraSpec = model._lora_spec
    os.makedirs(out_dir, exist_ok=True)
    state = adapter_state_dict(model)
    save_file(state, os.path.join(out_dir, ADAPTER_WEIGHTS))
    meta = {**spec.to_json(), "tensors": len(state), **(extra_meta or {})}
    tmp = os.path.join(out_dir, ADAPTER_CONFIG + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    os.replace(tmp, os.path.join(out_dir, ADAPTER_CONFIG))
    return out_dir


def read_adapter_config(adapter_dir: str) -> dict:
    with open(os.path.join(adapter_dir, ADAPTER_CONFIG), encoding="utf-8") as fh:
        meta = json.load(fh)
    if meta.get("format") != ADAPTER_FORMAT:
        raise ValueError(f"Unsupported adapter format {meta.get('format')!r}")
    return meta


def spec_from_config(meta: dict) -> LoraSpec:
    keys = LoraSpec.__dataclass_fields__.keys()
    return LoraSpec(**{k: v for k, v in meta.items() if k in keys})


def load_adapter(model: nn.Module, adapter_dir: str) -> LoraReport:
    """Wrap ``model`` the way the adapter was trained, then load its tensors (strictly)."""
    from safetensors.torch import load_file

    spec = spec_from_config(read_adapter_config(adapter_dir))
    report = apply_lora(model, spec)
    state = load_file(os.path.join(adapter_dir, ADAPTER_WEIGHTS))
    expected = set(adapter_state_dict(model))
    got = set(state)
    if expected != got:
        raise ValueError(
            "The adapter does not match this model: "
            f"missing {sorted(expected - got)[:3]}, unexpected {sorted(got - expected)[:3]}"
        )
    model.load_state_dict(state, strict=False)
    return report


@torch.no_grad()
def merge_adapter(model: nn.Module) -> int:
    """Fold every LoRA layer into its base weight and unwrap it. Returns layers merged."""
    spec: Optional[LoraSpec] = getattr(model, "_lora_spec", None)
    if spec is None:
        return 0
    root = getattr(model, spec.root)
    wrapped = [n for n, m in root.named_modules() if isinstance(m, LoRALinear)]
    for dotted in wrapped:
        _swap(root, dotted, root.get_submodule(dotted).merge())
    del model._lora_spec
    for p in model.parameters():
        p.requires_grad_(True)
    return len(wrapped)


def trainable_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    return (p for p in model.parameters() if p.requires_grad)
