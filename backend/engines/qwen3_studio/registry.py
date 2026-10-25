"""Qwen3-TTS studio: settings and the model registry.

Models are found, never hard-coded: the checkpoint named by the engine's
environment variable, every checkpoint folder under a root the user sets, and
the stock Qwen3-TTS models already in the Hugging Face cache. Each gets a stable
id derived from its location, so a saved session keeps pointing at the same
model after a restart. Nothing here loads a model or touches the network.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

SETTINGS_FILE = "settings.json"
ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,160}$")
#: Folders never searched for checkpoints (they hold parts of one, not a checkpoint).
_SKIP = {"speech_tokenizer", "samples", "logs", "config", "environment", "__pycache__", ".git"}
_MAX_DEPTH = 6
#: The built-in voices of the stock CustomVoice models, with their rows in the codec embedding table
#: (read from the 0.6B CustomVoice config; assumed identical in the 1.7B one, which was not available to check).
#: Only the stock CustomVoice models carry a trained embedding in these rows. Other models, including
#: fine-tunes, have empty rows there, so a voice "borrowed" from this table sounds different from the original.
STOCK_SPEAKERS = {"serena": 3066, "vivian": 3065, "uncle_fu": 3010, "ryan": 3061, "aiden": 2861,
                  "ono_anna": 2873, "sohee": 2864, "eric": 2875, "dylan": 2878}

_DEFAULTS = {"python": "", "checkpoint_root": "", "output_dir": "", "warmup_runs": 2,
             "warmup_text": "This is a short warm up sentence to settle the voice."}


def studio_dir() -> str:
    from core.config import DATA_DIR

    path = os.path.join(DATA_DIR, "qwen_studio")
    os.makedirs(path, exist_ok=True)
    return path


def load_settings() -> dict:
    try:
        with open(os.path.join(studio_dir(), SETTINGS_FILE), encoding="utf-8") as fh:
            saved = json.load(fh)
    except (OSError, ValueError):
        saved = {}
    return {**_DEFAULTS, **{k: v for k, v in saved.items() if k in _DEFAULTS}}


def save_settings(patch: dict) -> dict:
    merged = {**load_settings(), **{k: v for k, v in patch.items() if k in _DEFAULTS and v is not None}}
    merged["warmup_runs"] = max(0, min(10, int(merged["warmup_runs"])))
    for key in ("python", "checkpoint_root", "output_dir"):
        merged[key] = str(merged[key]).strip()
    path = os.path.join(studio_dir(), SETTINGS_FILE)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=2)
    os.replace(tmp, path)
    return merged


def default_output_dir() -> str:
    from core.config import DATA_DIR

    return os.path.join(DATA_DIR, "qwen_studio", "playground")


def output_dir(settings: Optional[dict] = None) -> str:
    return ((settings or load_settings()).get("output_dir") or "").strip() or default_output_dir()


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-.")[:120] or "model"


def read_checkpoint(folder: str) -> Optional[dict]:
    """What a checkpoint folder says about itself, or None if it is not one."""
    cfg_path = os.path.join(folder, "config.json")
    if not (os.path.isfile(cfg_path) and os.path.isfile(os.path.join(folder, "model.safetensors"))):
        return None
    try:
        cfg = json.load(open(cfg_path, encoding="utf-8"))
    except (OSError, ValueError):
        return None
    kind = cfg.get("tts_model_type")
    if kind not in ("base", "custom_voice", "voice_design"):
        return None
    return {
        "type": kind,
        "size": cfg.get("tts_model_size"),
        "speakers": sorted((cfg.get("talker_config") or {}).get("spk_id") or {}),
        "bytes": os.path.getsize(os.path.join(folder, "model.safetensors")),
    }


def _hf_caches() -> list[Path]:
    """Hugging Face caches to look in: the app's (it moves the cache on Windows to keep paths short)
    first, then the user's default one, where other tools usually downloaded the stock models."""
    found: list[Path] = []
    explicit = os.environ.get("HF_HUB_CACHE")
    home = os.environ.get("HF_HOME")
    if explicit:
        found.append(Path(explicit))
    elif home:
        found.append(Path(home) / "hub")
    found.append(Path.home() / ".cache" / "huggingface" / "hub")
    unique: list[Path] = []
    for path in found:
        if path.is_dir() and path not in unique:
            unique.append(path)
    return unique


def _entry(folder: str, model_id: str, label: str, source: str) -> Optional[dict]:
    info = read_checkpoint(folder)
    if info is None:
        return None
    borrowable = [] if info["type"] == "voice_design" else [n for n in STOCK_SPEAKERS if n not in info["speakers"]]
    return {"id": model_id, "label": label, "path": os.path.abspath(folder), "source": source,
            "borrowable_speakers": borrowable, **info}


def _scan(root: str) -> list[dict]:
    found: list[dict] = []
    base = os.path.abspath(root)
    if not os.path.isdir(base):
        return found
    for current, dirs, _files in os.walk(base):
        depth = os.path.relpath(current, base).count(os.sep)
        dirs[:] = [d for d in dirs if d not in _SKIP] if depth < _MAX_DEPTH else []
        rel = os.path.relpath(current, base)
        if rel == ".":
            continue
        entry = _entry(current, _slug(rel.replace(os.sep, "-")), " / ".join(rel.split(os.sep)[-3:]), "checkpoint_root")
        if entry:
            found.append(entry)
            dirs[:] = []
    return found


def _stock_models() -> list[dict]:
    found: list[dict] = []
    for hub in _hf_caches():
        for repo in sorted(hub.glob("models--Qwen--Qwen3-TTS-*")):
            snaps = sorted((repo / "snapshots").glob("*")) if (repo / "snapshots").is_dir() else []
            for snap in snaps[-1:]:
                name = repo.name.replace("models--Qwen--", "")
                entry = _entry(str(snap), "hf-" + _slug(name.lower()), "Qwen/" + name, "huggingface_cache")
                if entry:
                    found.append(entry)
    return found


def discover_models(settings: Optional[dict] = None) -> list[dict]:
    """All usable Qwen3-TTS models, with stable ids. Unreadable folders are skipped."""
    settings = settings or load_settings()
    models: dict[str, dict] = {}
    env = os.environ.get("OMNIVOICE_QWEN3_NARRATOR_CKPT", "").strip()
    if env:
        entry = _entry(env, "env-" + _slug(os.path.basename(env.rstrip("\\/"))), "Engine checkpoint (environment)", "environment")
        if entry:
            models[entry["id"]] = entry
    root = (settings.get("checkpoint_root") or "").strip()
    if root:
        for entry in _scan(root):
            models.setdefault(entry["id"], entry)
    for entry in _stock_models():
        models.setdefault(entry["id"], entry)
    return sorted(models.values(), key=lambda m: (m["type"] != "custom_voice", m["label"]))


def get_model(model_id: str, settings: Optional[dict] = None) -> Optional[dict]:
    return next((m for m in discover_models(settings) if m["id"] == model_id), None)


def first_base(settings: Optional[dict] = None) -> Optional[dict]:
    """The Base model used to make x-vectors, preferring the biggest stock one."""
    bases = [m for m in discover_models(settings) if m["type"] == "base"]
    return max(bases, key=lambda m: m["bytes"], default=None)
