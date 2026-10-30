"""Qwen3-TTS studio: settings and the model registry.

Models are found, never hard-coded: the checkpoint named by the engine's
environment variable, every checkpoint under the folders listed in Settings ->
Storage -> Model locations (``core.model_locations``), and the stock Qwen3-TTS
models already in the Hugging Face cache. Each gets a stable
id derived from its location, so a saved session keeps pointing at the same
model after a restart. Nothing here loads a model or touches the network.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

from core import model_locations

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

_DEFAULTS = {"python": "", "output_dir": "", "warmup_runs": 2,
             "warmup_text": "This is a short warm up sentence to settle the voice."}


def default_python() -> str:
    """The project's own Qwen environment (``.venv-qwen`` in the repo root), or "" if it has not been built.
    The sidecar never falls back to any other python: not PATH, not the app's venv, not a system install."""
    root = Path(__file__).resolve().parents[3]
    python = root / ".venv-qwen" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return str(python) if python.is_file() else ""


def effective_python(settings: dict) -> Optional[str]:
    """The python the sidecar runs with: the user's setting, else the project's ``.venv-qwen``."""
    return (settings.get("python") or "").strip() or default_python() or None


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
    for key in ("python", "output_dir"):
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
        "has_adapter": os.path.isfile(os.path.join(folder, "adapter_model.safetensors")),
    }


def read_adapter(folder: str) -> Optional[dict]:
    """What a LoRA adapter folder (``adapter_config.json`` + ``adapter_model.safetensors``, no full weights) says about itself, or None.
    An adapter is not runnable: it must be merged into its Base model first (``merge_adapter_model``)."""
    if os.path.isfile(os.path.join(folder, "model.safetensors")):
        return None  # a merged checkpoint (it may also carry its adapter files); it is listed as the checkpoint it is
    from engines.fish_api import qwen3_peft

    info = qwen3_peft.read_adapter_info(folder)
    if info is None:
        return None
    return {"type": "adapter", "size": None, "speakers": info["speakers"], "bytes": info["bytes"], "rank": info["rank"],
            "alpha": info["alpha"], "base_model": info["base_model"]}


def _hf_caches() -> list[Path]:
    """Hugging Face caches to look in: the app's (it moves the cache on Windows to keep paths short)
    first, then the user's default one, where other tools usually downloaded the stock models."""
    found: list[Path] = []
    explicit = os.environ.get("HF_HUB_CACHE")
    home = os.environ.get("HF_HOME")
    legacy = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if explicit:
        found.append(Path(explicit))
    elif legacy:
        found.append(Path(legacy))
    elif home:
        found.append(Path(home) / "hub")
    found.append(Path.home() / ".cache" / "huggingface" / "hub")
    unique: list[Path] = []
    for path in found:
        if path.is_dir() and path not in unique:
            unique.append(path)
    return unique


def _entry(folder: str, model_id: str, label: str, source: str) -> Optional[dict]:
    info = read_checkpoint(folder) or read_adapter(folder)
    if info is None:
        return None
    borrowable = [] if info["type"] in ("voice_design", "adapter") else [n for n in STOCK_SPEAKERS if n not in info["speakers"]]
    return {"id": model_id, "label": label, "path": os.path.abspath(folder), "source": source,
            "location": source, "borrowable_speakers": borrowable, **info}


def _scan(root: str, location: str = "checkpoint_root") -> list[dict]:
    found: list[dict] = []
    base = os.path.abspath(root)
    if not os.path.isdir(base):
        return found
    # The root may itself be a checkpoint (the user pointed at one folder, not a parent of several).
    own = _entry(base, _slug(os.path.basename(base)), os.path.basename(base), location)
    if own:
        return [own]
    for current, dirs, _files in os.walk(base):
        depth = os.path.relpath(current, base).count(os.sep)
        dirs[:] = [d for d in dirs if d not in _SKIP] if depth < _MAX_DEPTH else []
        rel = os.path.relpath(current, base)
        if rel == ".":
            continue
        entry = _entry(current, _slug(rel.replace(os.sep, "-")), " / ".join(rel.split(os.sep)[-3:]), location)
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
    for row in model_locations.load():
        for entry in _scan(row["path"], row["id"]):
            entry["location_name"] = row["name"]
            models.setdefault(entry["id"], entry)
    for entry in _stock_models():
        models.setdefault(entry["id"], entry)
    return sorted(models.values(), key=lambda m: (m["type"] != "custom_voice", m["label"]))


def get_model(model_id: str, settings: Optional[dict] = None) -> Optional[dict]:
    return next((m for m in discover_models(settings) if m["id"] == model_id), None)


def resolve_adapter_base(adapter: dict, models: list[dict]) -> Optional[dict]:
    """The Base model an adapter was trained on, found among the models the app already knows (Hugging Face cache and the folders of
    Settings -> Storage), by the model name recorded in ``lora_meta.json``. Never searches a private location."""
    recorded = str(adapter.get("base_model") or "").replace("\\", "/").rstrip("/")
    name = recorded.split("/")[-1].lower()
    bases = [m for m in models if m["type"] == "base"]
    if name:
        for m in bases:
            if m["label"].split("/")[-1].lower() == name or os.path.basename(m["path"].rstrip("/\\")).lower() == name:
                return m
    return None


def merge_adapter_model(model_id: str, location_id: str, name: str, lora_scale: float = 0.3, settings: Optional[dict] = None,
                       confirm: bool = False) -> dict:
    """Merge an adapter into its Base model and write the standard custom_voice checkpoint into a folder of Settings -> Storage
    (``location_id``). No new path is accepted: the output is ``<location>/<name>``. Refuses to overwrite and never touches the adapter (it stays the
    source of truth). Needs ``confirm=True``: a merge bakes the scale into the weights. CPU only."""
    from engines.fish_api import qwen3_peft

    models = discover_models(settings)
    adapter = next((m for m in models if m["id"] == model_id), None)
    if adapter is None or adapter["type"] != "adapter":
        raise ValueError("That model is not a LoRA adapter.")
    base = resolve_adapter_base(adapter, models)
    if base is None:
        raise ValueError(f"The Base model this adapter was trained on ({adapter.get('base_model') or 'unknown'}) was not found. Download it or add its "
                         "folder under Settings -> Storage -> Model locations, then try again.")
    loc = next((r for r in model_locations.load() if r["id"] == location_id), None)
    if loc is None:
        raise ValueError("Choose one of the model folders listed in Settings -> Storage to write the merged model into.")
    if not 0.0 <= float(lora_scale) <= 1.0:
        raise ValueError("The adapter scale must be between 0.0 (the base model) and 1.0.")
    if not confirm:
        raise ValueError(f"Merging bakes scale {float(lora_scale):g} into a new model; confirm it explicitly. The adapter itself is kept.")
    folder = _slug(name or adapter["label"] + "-merged")
    out = os.path.join(os.path.abspath(loc["path"]), folder)
    if os.path.exists(out):
        raise ValueError(f"{folder} already exists in {loc['name']}; choose another name (nothing is overwritten).")
    os.makedirs(loc["path"], exist_ok=True)
    report = qwen3_peft.merge_adapter(base["path"], adapter["path"], out, lora_scale=float(lora_scale))
    return {"path": out, "location": loc["name"], "folder": folder, "base": base["label"], "speakers": report["speakers"],
            "changed_pct": report["changed_pct"], "scale": report["scale"], "lora_scale": report["lora_scale"],
            "warnings": ([f"Scale {float(lora_scale):g} is above 0.5; voices often get noisy or lose the base's clarity."] if float(lora_scale) > 0.5 else [])}


def first_base(settings: Optional[dict] = None) -> Optional[dict]:
    """The Base model used to make x-vectors, preferring the biggest stock one."""
    bases = [m for m in discover_models(settings) if m["type"] == "base"]
    return max(bases, key=lambda m: m["bytes"], default=None)
