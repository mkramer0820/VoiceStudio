# Fish training data: handoff for the LoRA phase

State of `ext/fish-training-data` and what the next phase (training LoRAs) can build on.

## What exists

- **Corpus index:** `backend/engines/fish_api/dataset.py` reads story run folders (`<root>/<chapter>/<story>/run-NNN_…/manifest.json` + `story.json` + WAVs). Read-only, no network. Root is the `tts.fish.dataset_dir` setting.
- **API:** `backend/api/routers/fish_dataset.py` under `/api/fish/dataset` (overview, lines, audio, export, root).
- **UI:** `electron/src/renderer/src/features/fish-dataset/` (route `/fish-dataset`).
- **Export layout:** `<dest>/<name>/<speaker>/<speaker>_00001.wav` (+ `.lab`), `<speaker>/metadata.csv`, top-level `metadata.jsonl` (`file_name`, `text`, `text_raw`, `text_plain`, `speaker`, `voice_id`, `duration`, `sample_rate`, `tags`, `chapter`, `run_id`), optional `train.jsonl` / `val.jsonl`, and `dataset.json` (card with filters and source).
- **Voice library:** `backend/engines/fish_api/library.py` stores character names, Fish details and usage per voice id.

## Facts about the data (2026-10-23, `D:\fish_tts\input`)

- 32 runs, 31 chapters, ~3,370 usable clips, all `s2.1-pro-free`, 44.1 kHz mono WAV.
- Only **SS_Narrator** has hours of audio (~4 h). Noctis ~34 min and sun_2 ~26 min are borderline; every other speaker is under ~8 min.
- `[slow]` is on ~99% of narrator clips, so a narrator-only model will always speak slowly unless that is changed in the next renders or controlled at generation time.
- Fish's non-verbal tags are `[sighing]`, `[laughing]`, `[chuckling]`; OmniVoice expects its own spellings. Use the export's `name=other` tag mapping.
- **Loudness differs by chapter.** Chapters 3193–3199 were rendered with `normalize_loudness: false`, `volume: 0` and `top_p: 1.0`: median level about -14 dB RMS with peaks near 0.95 (hot, not clipped). From 3200 on the runs used loudness normalization and `volume -4`: about -26 dB RMS, peaks near 0.27. Do not mix the two in one training set without choosing one level (filter by level or chapter, or normalize on export).
- Audio measurements are cached by clip sha256 in `<data dir>/fish_stories/clip_stats.json`.
- 140 clips are listed in manifests but their WAVs were deleted (mostly chapter 3218); they are skipped.
- 303 clips are older copies of re-rendered lines and hidden by default.

## Decisions and caveats

- Speakers are grouped by Fish **voice id**, not name (two names can share one voice).
- The clips are synthetic Fish output. The owner uses them privately; the export dialog and `dataset.json` still record the source.
- Loudness: requests now always send `prosody.normalize_loudness` explicitly (it used to be omitted when on). Renders from 3200 on were made without the field and still measure about -26 dB, so Fish treats a missing field like `true`; the explicit field changes nothing for them.
- Locale JSON files are committed with CRLF line endings. With `core.autocrlf=true` a plain `git add` converts them to LF and produces a ~79k-line diff. Stage them with `git hash-object -w --no-filters` + `git update-index --cacheinfo`, or set `core.autocrlf=false` for this repo.

## Where to start for the LoRA phase

Read these in order. Everything below was checked in the repo on 2026-10-23.

### 1. The trainer that already exists (OmniVoice)

- Code: `omnivoice/training/` (`trainer.py`, `builder.py`, `config.py`, `checkpoint.py`) and `omnivoice/cli/train.py`. Docs: `docs/training.md` (launch with `accelerate launch -m omnivoice.cli.train --train_config … --data_config … --output_dir …`) and `docs/data_preparation.md`. Example configs: `examples/config/`.
- It does **full fine-tuning** from a checkpoint (`init_from_checkpoint`, `resume_from_checkpoint`; backbone `Qwen/Qwen3-0.6B` via `llm_name_or_path`). A repo-wide search finds **no LoRA or `peft` code**, so LoRA means adding an adapter path to the trainer (and checking `peft` against `pyproject.toml` / `uv tree` before adding a dependency).
- Input it expects: a JSONL with `{"id", "audio_path", "text", "language_id"}` per line, audio resampled to **24 kHz**. `omnivoice.scripts.extract_audio_tokens` (needs `eustlb/higgs-audio-v2-tokenizer`) turns that into WebDataset tar shards plus a `data.lst`, which the data config points at.
- **Quick win to build first:** an `omnivoice` export format in `backend/engines/fish_api/dataset.py` that writes exactly that JSONL (24 kHz mono WAVs, tags stripped except the OmniVoice tags you keep, `language_id: "en"`), so the existing export dialog feeds the existing tokenizer script with no manual conversion.

### 2. Other targets

- **Fish S2 local:** not in the repo yet (only the cloud API engine exists). The `fish_lab` export (WAV + `.lab`) matches Fish's own fine-tuning layout. A local engine has to come before a Fish LoRA. The S2 Pro config pasted during design shows a 36-layer text model plus a 4-layer audio decoder with 10 codebooks; which modules to target with LoRA was not investigated.
- **Qwen3-TTS:** nothing in the repo. Its transcript and tag format, sample rate and trainer are unresearched.

### 3. Infrastructure to reuse

- **Long jobs with progress, logs and resume:** `backend/engines/fish_api/story_jobs.py` (a background thread, `job.json` + `run.log` per run, interrupted-run detection, resume) and the smaller `dataset_quality.py` job. Copy the pattern for a `training_runs` module.
- **Run training as a subprocess, not in the backend process:** `backend/services/subprocess_backend.py` (long-lived sidecar with its own venv and a hard kill), `backend/core/contained_subprocess.py`, and `docs/engines/omnivoice-subprocess.md` for why a wedged GPU job must be killable. Training will need its own venv for torch, accelerate and peft.
- **GPU and VRAM gating:** `backend/services/model_manager.py` (the GPU pool, execution budgets and the "abandoned worker" behaviour) and `backend/services/gpu_sandbox.py`. Unload the resident TTS model before a run.
- **API and UI pattern:** `backend/api/routers/fish_dataset.py` and `electron/src/renderer/src/features/fish-dataset/` (hook + page + dialogs + tests). A natural home is a *Train* step reached from the export dialog, with a runs list like the Jobs dialog.

### 4. Constraints from the repo rules

- `docs/ROADMAP.md` (lines about on-device learning) says LoRA **ships only with an automatic evaluation gate**. Plan the before/after listening test and a metric (for example a speaker-similarity or ASR word-error check on held-out `val.jsonl` clips) as part of the first version, not a follow-up.
- `CLAUDE.md`: default features must behave the same on macOS, Windows and Linux. GPU training will differ by host, so keep it opt-in (a setting or flag), skip it where it cannot work (for example no CUDA) with a clear message, and never call out to the network for it. Docs, changelog, locale parity (all 22 locale files) and a fail-before/pass-after test come with the change.
- Hold-out: `val.jsonl` already uses a stable hash split, so re-exporting never leaks a validation clip into training.

### 5. Data decisions to make before training

- Pick **one loudness level** (see above) or add normalization on export; filter or normalize chapters 3193–3199.
- Only SS_Narrator has hours of audio. Noctis (~34 min) and sun_2 (~26 min) are the next candidates; the rest are too small. Decide whether `[slow]` stays in the narrator's text or the clips are re-rendered without it.
- Decide the tag policy per target: Fish keeps its tags; OmniVoice takes only its non-verbal tags (use the `name=other` mapping, e.g. `laughing=laughter`); verify OmniVoice's exact tag list first.

## Not built yet (small items)

- A tag-to-`instruct` mapping for OmniVoice style attributes (whisper, pitch, gender).
- Loudness normalization on export (bring all chapters to one level).
- An `omnivoice` export format (see above).
