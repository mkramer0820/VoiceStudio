"""Pure helpers for LoRA (PEFT) fine-tuning of Qwen3-TTS and for multi-speaker custom_voice checkpoints.

Nothing here needs a GPU. torch and safetensors are imported only inside the functions that touch tensors, so the constants,
speaker-row logic, config writer and frame guard import anywhere. The training copy of ``sft_12hz.py`` that runs on Colab
(``qwen3_colab_runner.LORA_HELPER``) carries its own small copy of the merge maths because the runner is a single standard-library
file; ``tests/backend/engines/test_fish_qwen3_peft.py`` runs both on the same tensors and fails if they ever disagree.

Facts checked against the pinned Qwen3-TTS commit 022e286 (2026-10-28): the adapted linear layers are named
``talker.model.layers.N.{self_attn.{q,k,v,o}_proj, mlp.{gate,up,down}_proj}`` and
``talker.code_predictor.model.layers.N.<same>``; the custom_voice lookup is ``speaker.lower() in talker_config.spk_id``; the talker
codec embedding table has ``talker_config.vocab_size`` (3072) rows, so rows ``3000 + 20 * idx`` fit four speakers only.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from typing import Callable, Iterable, Optional

from .qwen3_colab_runner import (CODEC_VOCAB_ROWS, FRAME_LIMIT, LORA_TARGET_REGEX, SPK_ROW_BASE, SPK_ROW_STEP,  # noqa: F401
                                 STOCK_ROWS, StepFailed, cap_micro_batch, speaker_rows, LORA_DEFAULT_SCALE)  # one definition: the runner is a single standard-library file

PeftError = StepFailed  # the runner's "stopped on purpose, with a message" error
#: Plausible L2 norm range of a speaker embedding, taken from the stock rows (about 9 to 17). INFERRED, so it is a warning, not a stop.
EMBEDDING_NORM_RANGE = (9.0, 17.0)

LORA_FORMAT = "voicestudio-lora-1"
ADAPTER_FILE = "adapter_model.safetensors"
ADAPTER_CONFIG = "adapter_config.json"
LORA_META = "lora_meta.json"
SPEAKER_EMBEDDINGS_FILE = "speaker_embeddings.safetensors"
SPEAKERS_FILE = "speakers.json"

#: Adapted tensors that must differ from the base after the bf16 merge, in percent of elements (average over the adapted tensors).
#: PLACEHOLDER, not measured on a GPU: below it the merge probably rounded the adapter away.
MERGE_MIN_CHANGED_PCT = 0.1


# ------------------------------------------------------------------------------------------------ speakers
def embedding_norm_warning(name: str, norm: float, lo_hi: tuple = EMBEDDING_NORM_RANGE) -> Optional[str]:
    """A warning when a speaker embedding's norm is outside the range the stock rows have, else None."""
    lo, hi = lo_hi
    if lo <= norm <= hi:
        return None
    return (f"Speaker {name}: embedding norm {norm:.2f} is outside {lo:g} to {hi:g}, the range of the stock rows "
            "(an inferred range, not a rule). Check the reference clips.")


def average_embeddings(vectors: list):
    """Mean of several 1-D speaker embeddings, computed in float32 (a list of torch tensors or lists)."""
    import torch

    if not vectors:
        raise PeftError("No reference clip produced an embedding.")
    stacked = torch.stack([torch.as_tensor(v, dtype=torch.float32).reshape(-1) for v in vectors])
    return stacked.mean(dim=0)


def cached_speaker_embeddings(path: str, names: list, compute: Callable[[str], object]) -> dict:
    """``{name: float32 vector}``. Vectors found in the safetensors file at ``path`` are reused; missing ones come from
    ``compute(name)`` and the file is rewritten. A cache hit never calls ``compute`` (the encoder is not even loaded)."""
    import torch
    from safetensors.torch import load_file, save_file

    have = {k: v.float() for k, v in load_file(path).items()} if os.path.isfile(path) else {}
    out, changed = {}, False
    for name in names:
        low = name.lower()
        if low not in have:
            have[low] = torch.as_tensor(compute(low), dtype=torch.float32).reshape(-1).cpu().clone()
            changed = True
        out[low] = have[low]
    if changed:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp"
        save_file({k: v.contiguous() for k, v in have.items()}, tmp)
        os.replace(tmp, path)
    return out


# ------------------------------------------------------------------------------------------------ config
def write_checkpoint_config(base_config: dict, rows: dict, out_path: Optional[str] = None) -> dict:
    """The ``config.json`` of a custom_voice checkpoint: ``tts_model_type``, ``spk_id`` and ``spk_is_dialect`` for every speaker
    (lowercase names, every other key of the base config untouched). Returns the dict and writes it when ``out_path`` is given."""
    cfg = json.loads(json.dumps(base_config))  # deep copy
    cfg["tts_model_type"] = "custom_voice"
    talker = cfg.get("talker_config") or {}
    clean = {str(n).lower(): int(r) for n, r in rows.items()}
    if len(clean) != len(rows):
        raise PeftError("Speaker names differ only in case.")
    talker["spk_id"] = clean
    talker["spk_is_dialect"] = {n: False for n in clean}
    cfg["talker_config"] = talker
    if out_path:
        tmp = out_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, out_path)
    return cfg


def write_speaker_rows(state: dict, rows: dict, embeddings: dict, table_key: str = "talker.model.codec_embedding.weight") -> None:
    """Write each speaker's cached vector into its row of the codec embedding table in a state dict (in place)."""
    table = state[table_key]
    for name, row in rows.items():
        vec = embeddings[name]
        if vec.numel() != table.shape[1]:
            raise PeftError(f"Speaker {name}: embedding has {vec.numel()} values, the table rows have {table.shape[1]}.")
        table[row] = vec.detach().to(table.device).to(table.dtype)


# ------------------------------------------------------------------------------------------------ merge
def merged_weight(weight, lora_a, lora_b, scale: float):
    """``weight + scale * (B @ A)`` computed in float32, returned in float32 (round once, at the very end)."""
    import torch

    return weight.float() + float(scale) * (lora_b.float() @ lora_a.float())


def _pairs(adapter: dict) -> dict:
    """``{module path: (A, B)}`` from adapter keys ``<module>.lora_A.weight`` / ``<module>.lora_B.weight``."""
    pairs: dict = {}
    for key, tensor in adapter.items():
        m = re.fullmatch(r"(.+)\.lora_([AB])\.weight", key)
        if not m:
            raise PeftError(f"Unexpected adapter tensor name {key!r}.")
        pairs.setdefault(m.group(1), {})[m.group(2)] = tensor
    for module, ab in pairs.items():
        if set(ab) != {"A", "B"}:
            raise PeftError(f"Adapter has only one of lora_A / lora_B for {module}.")
    return {k: (v["A"], v["B"]) for k, v in pairs.items()}


def merge_state(base_state: dict, adapter: dict, scale: float, *, min_changed_pct: float = MERGE_MIN_CHANGED_PCT) -> tuple:
    """Merge an adapter into a copy of ``base_state`` (bf16 or float32 tensors). The add is done in float32 and rounded once.

    Returns ``(merged_state, report)``; ``report`` has ``adapted`` (tensor count) and ``changed_pct`` (mean share of elements that differ
    from the base after rounding, in percent). Raises when the adapter is all zeros in every B matrix (nothing trained), names a tensor
    the base does not have, or the rounding erased the delta (changed share under ``min_changed_pct``)."""
    pairs = _pairs(adapter)
    if not pairs:
        raise PeftError("The adapter holds no LoRA tensors.")
    merged = dict(base_state)
    shares, nonzero = [], 0
    for module, (a, b) in pairs.items():
        key = module + ".weight"
        if key not in base_state:
            raise PeftError(f"The base model has no tensor {key}: this adapter was trained on another model.")
        base = base_state[key]
        if float(b.float().abs().max()) > 0.0:
            nonzero += 1
        new = merged_weight(base, a, b, scale).to(base.dtype)
        shares.append(float((new != base).float().mean()) * 100.0)
        merged[key] = new
    if nonzero == 0:
        raise PeftError("MERGE FAILED: every lora_B matrix is zero, so nothing was trained (B starts at zero).")
    pct = sum(shares) / len(shares)
    if pct < min_changed_pct:
        raise PeftError(f"MERGE FAILED: after rounding to {next(iter(base_state.values())).dtype} only {pct:.4f}% of the adapted elements differ "
                        f"from the base (limit {min_changed_pct}%): the adapter delta is below the resolution of the checkpoint dtype. "
                        "Train longer or with a higher rank/learning rate, or keep the adapter unmerged.")
    return merged, {"adapted": len(pairs), "changed_pct": pct, "nonzero_b": nonzero}


def merge_adapter(base_dir: str, adapter_dir: str, out_dir: str, *, rows: Optional[dict] = None, embeddings: Optional[dict] = None,
                  scale: Optional[float] = None, lora_scale: float = LORA_DEFAULT_SCALE, min_changed_pct: float = MERGE_MIN_CHANGED_PCT) -> dict:
    """Merge a LoRA adapter folder into a copy of the base model folder and write a standard custom_voice checkpoint (same layout as a
    full fine-tune: ``model.safetensors`` without ``speaker_encoder.*``, ``config.json``, tokenizer files, ``speech_tokenizer/``).

    ``rows``/``embeddings`` default to the adapter folder's ``speakers.json`` and ``speaker_embeddings.safetensors``. ``scale`` defaults to
    ``alpha / r * lora_scale`` (``lora_scale`` default 0.3, range 0.0-1.0); the bf16 survival check runs at that scale. Written to a temporary folder and renamed last. Runs on CPU. Returns the merge report."""
    from safetensors.torch import load_file, save_file

    adapter_cfg = _read_json(os.path.join(adapter_dir, ADAPTER_CONFIG))
    if scale is None:
        scale = float(adapter_cfg["lora_alpha"]) / float(adapter_cfg["r"]) * float(lora_scale)
    if rows is None:
        rows = _read_json(os.path.join(adapter_dir, SPEAKERS_FILE))
    if embeddings is None:
        embeddings = {k: v for k, v in load_file(os.path.join(adapter_dir, SPEAKER_EMBEDDINGS_FILE)).items()}
    missing = [n for n in rows if n not in embeddings]
    if missing:
        raise PeftError("No cached embedding for speaker(s): " + ", ".join(missing))
    base_state = load_file(os.path.join(base_dir, "model.safetensors"))
    state, report = merge_state(base_state, load_file(os.path.join(adapter_dir, ADAPTER_FILE)), scale, min_changed_pct=min_changed_pct)
    for key in [k for k in state if k.startswith("speaker_encoder")]:
        del state[key]
    write_speaker_rows(state, rows, embeddings)
    tmp = out_dir.rstrip("/\\") + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(base_dir, tmp, ignore=shutil.ignore_patterns("model.safetensors", "config.json", ".cache"))
    write_checkpoint_config(_read_json(os.path.join(base_dir, "config.json")), rows, os.path.join(tmp, "config.json"))
    save_file({k: v.contiguous() for k, v in state.items()}, os.path.join(tmp, "model.safetensors"), metadata={"format": "pt"})
    shutil.rmtree(out_dir, ignore_errors=True)
    os.replace(tmp, out_dir)
    return {**report, "scale": scale, "lora_scale": float(lora_scale), "speakers": sorted(rows), "out_dir": out_dir}


def is_adapter_folder(folder: str) -> bool:
    return os.path.isfile(os.path.join(folder, ADAPTER_CONFIG)) and os.path.isfile(os.path.join(folder, ADAPTER_FILE))


def read_adapter_info(folder: str) -> Optional[dict]:
    """What an adapter folder says about itself, or None if it is not one."""
    if not is_adapter_folder(folder):
        return None
    try:
        cfg = _read_json(os.path.join(folder, ADAPTER_CONFIG))
        meta = _read_json(os.path.join(folder, LORA_META)) if os.path.isfile(os.path.join(folder, LORA_META)) else {}
        speakers = _read_json(os.path.join(folder, SPEAKERS_FILE)) if os.path.isfile(os.path.join(folder, SPEAKERS_FILE)) else {}
    except (OSError, ValueError):
        return None
    return {"rank": cfg.get("r"), "alpha": cfg.get("lora_alpha"), "base_model": meta.get("base_model"), "speakers": sorted(speakers),
            "bytes": os.path.getsize(os.path.join(folder, ADAPTER_FILE))}


def _read_json(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
