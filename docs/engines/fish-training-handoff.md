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
- 140 clips are listed in manifests but their WAVs were deleted (mostly chapter 3218); they are skipped.
- 303 clips are older copies of re-rendered lines and hidden by default.

## Decisions and caveats

- Speakers are grouped by Fish **voice id**, not name (two names can share one voice).
- The clips are synthetic Fish output. The owner uses them privately; the export dialog and `dataset.json` still record the source.
- Loudness: requests now always send `prosody.normalize_loudness` explicitly (it used to be omitted when on). Existing renders were made without the field, so their level may differ slightly from new ones; mixing old and new in one set is worth a loudness check.
- Locale JSON files are committed with CRLF line endings. With `core.autocrlf=true` a plain `git add` converts them to LF and produces a ~79k-line diff. Stage them with `git hash-object -w --no-filters` + `git update-index --cacheinfo`, or set `core.autocrlf=false` for this repo.

## Not built yet (candidates for the training phase)

- Audio-quality filters that read the WAVs (peak, clipping, silence, RMS outliers); the index uses manifest data only.
- A tag-to-`instruct` mapping for OmniVoice style attributes (whisper, pitch, gender).
- Training itself: choose the target (Fish S2 local, Qwen3-TTS, OmniVoice), its sample rate and transcript format, where training runs (subprocess/venv, GPU detection), progress and logs, resume, adapter storage and loading into the matching engine, and a before/after listening test.
- Open questions: which base model first, minimum clips per speaker to allow a run, and whether adapters are managed per speaker in the voice library.
