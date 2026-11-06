# Fork handoff: what this project is and where things are

Read this first. It is for the next person or agent who picks the fork up. How to use the pages as a user: [Story studios guide](../story-studios-guide.md).

## 1. What the fork is for

A **Windows + NVIDIA CUDA** studio for building and using multi-voice stories with **Qwen3-TTS** (your own fine-tunes and LoRAs) and **Fish Audio** (cloud API), plus the tools that make training data for the next Qwen fine-tune. It started as a fork of `debpalash/VoiceStudio` (an open-source voice cloning, dubbing and dictation app). Much of upstream is still in the repo and is **not part of this pipeline**; see section 5.

Owner preferences that shape every decision:

- Windows and CUDA only. **AMD/ROCm, Mac (MPS/MLX) and Linux-server work is out of scope**; it stays in the code as a later fallback and must not be made impossible, but nothing here is built, tested or documented for it.
- **No model loads unless a feature is used.** Nothing loads at startup (the TTS model, dictation ASR and watermark generator are opt-in: `OMNIVOICE_PRELOAD_TTS=1`, `OMNIVOICE_PRELOAD_CAPTURE_ASR=1`, `OMNIVOICE_PRELOAD_WATERMARK=1`). The owner does not use OmniVoice (the upstream default engine) and does not want it on the GPU.
- Models come only from **Settings → Storage → Model locations** (plus the Hugging Face cache). The pages never add a model-path box. The owner's fine-tunes are in `E:\Qwen3TTS_Checkpoints` (registered as "Qwen3 fine-tunes").
- **Do not touch the training-export code** (section 4.3) unless asked; it feeds the owner's real fine-tuning runs.
- The UI offers choices; it never decides which model or engine runs something.
- Licence-registry governance and upstream's cross-platform rules are not the owner's concern in this fork ("my project, my fork"), but new features must not make other platforms impossible later.

## 2. Branches and safety

| Branch | What it is |
|---|---|
| `main` | The fork's main line (everything in this document). `origin` = `mkramer0820/VoiceStudio`. |
| `v1` | The old `main` (commit `f47221a1`, upstream's state when the fork started), kept as-is. |
| `ext/qwen3-story` | The branch `main` was fast-forwarded from; same commits as `main`, kept for now. |
| `archive/upstream-full-before-debloat` | A snapshot taken before the UI was slimmed down: every upstream page still offered. |
| `ext/fish-api` | An older fork branch that `ext/qwen3-story` was built on. |

Commit identity: the owner's git identity only; **no Co-Authored-By or "Generated with" lines** (CLAUDE.md, enforced by `scripts/check_commit_identities.py`). Do not push or open PRs unless asked.

## 3. The studios (what exists)

| Page | Route | Backend | What it does |
|---|---|---|---|
| Qwen playground | `/qwen-playground` | `backend/engines/qwen3_studio/playground.py`, router `backend/api/routers/qwen_studio.py` | Run "takes" on any Qwen3-TTS model: custom voice, clone on Base, LoRA adapter at a chosen scale (merged on the CPU into a cache), instruct, experiments. Sessions are saved. Will keep growing. |
| Qwen story | `/qwen-story` | `qwen3_studio/{story,profiles,emotions,effects}.py` | Multi-voice book: voice profiles (narrator / stock / clone with a cached speaker embedding), `[Name]` and emotion tags, per-line and per-voice ffmpeg **voice effects** (`{Internal}` tags, your own recipes), per-chunk render and redo, `story.wav`. |
| Fish Story | `/fish-story` | `backend/engines/fish_api/{story,story_jobs,client,config}.py` | Story board for Fish Audio: voices per character, **a Fish model per narrator/character**, `{Internal}` effects applied to the returned audio, preview of the exact requests, resume. |
| Fish dataset | `/fish-dataset` | `fish_api/{dataset,qwen3_export,...}.py`, router `fish_dataset.py` | Browse the audio Fish Story produced, select clips of one speaker, **export a training set** (section 4). |
| Transcriptions | `/transcriptions` | upstream ASR (`services/asr_backend.py`) | Transcribe, search, export text and **segments with timestamps** (SRT, VTT, timed text, JSON). |
| Integrations / API | `/integrations`, OpenAI-compatible routes | `backend/api/routers/openai_compat.py` | Kept: the OpenAI-compatible API is useful. |

Shared pieces: `qwen3_narrator` is the sidecar process that runs Qwen3-TTS (CUDA graphs, one model at a time, `backend/engines/qwen3_narrator/main.py`, install pins in `requirements.txt`, one-click install in `services/sidecar_install.py`); `services/pronunciation.py` (one dictionary used by Generate, Fish Story, Qwen playground and Qwen story); `core/model_locations.py`.

## 4. Training: the fork's main output

### 4.1 Export (`fish_api/qwen3_export.py`, `qwen3_colab.py`)
Export dialog on `/fish-dataset` writes a folder or zip with `train.jsonl`, `audio/` (24 kHz mono PCM16, 1 s of silence appended), `reference/ref.wav`, `manifest.json` (training mode, speaker, counts, parameters, **bundle id**), `training_settings.json`, the Colab notebook, the runner `voicestudio_qwen3_colab.py`, `colab_setup.sh`, `train_local.sh`/`.ps1`, `README.txt`. Options: full fine-tune or LoRA (own `_lora` folder), `--instruct` conditioning (an `instruct` field from the clip `[tags]`), up to four speakers in one model.

### 4.2 Runner (`fish_api/qwen3_colab_runner.py`, **runner 23**)
One file does everything; the notebook and scripts only call it. Defaults: full fine-tune lr **2e-6**, LoRA lr **2e-6**, rank 16, alpha 32, adapter/merge scale **0.3** (0.0-1.0, warn above 0.5). The upstream training script (pinned commit `022e286`) is patched in a copy for text projection, label shift, sub-talker alignment, validation loss, atomic checkpoint saves, LoRA, multi-speaker and the opt-in instruct conditioning. **Colab needs an exact stack** (Python 3.13, torch 2.11.0+cu130, community flash-attn wheel); never upgrade torch in the runtime (details: [qwen3-tts-colab-runtime.md](../engines/qwen3-tts-colab-runtime.md)). Every shipped file prints the same bundle id; the runner warns when it differs from the manifest.

### 4.3 Hands off
Do not change the export format, the runner's patches or the notebook generator without the owner asking. They have a cross-layer defaults test (`tests/backend/engines/test_fish_qwen3_defaults.py`) and a committed notebook that must equal the generator's output.

### 4.4 What is **not** built (the next big feature)
**Audio in, training set out.** The Fish dataset page works from audio Fish Story produced. There is no flow that takes *your own recordings* → transcribes → cuts clips by segment → writes a Qwen training set. The pieces exist separately: Transcriptions now exports segments with start/end times, and `qwen3_export.py` knows the target format. Suggested shape: a page or button that takes a transcript's segments plus the source audio, slices with ffmpeg (inputs through `services.ffmpeg_utils.local_inputs_only`), lets the owner accept/reject clips, and feeds the existing exporter. Upstream's own data preparation (`docs/data_preparation.md`, `omnivoice/`) is for OmniVoice tokens and is not this.

## 5. What upstream left in, and what the fork hides

The UI hides these workspaces from the menu, command palette and home page: `/clone`, `/design`, `/stories`, `/audiobook`, `/dub`, `/batch`, `/calls`, `/gallery`, `/projects`, `/tools`. The list is one file, `electron/src/renderer/src/lib/fork-profile.ts` (`HIDDEN_ROUTES`); delete a path to bring a page back. The code, routes and APIs of hidden pages are untouched and still work.

Not removed (deliberately, to keep a later merge or revert easy): Mac/MLX/MPS code in `backend/` and `electron/`, AMD/ROCm code, Docker/ROCm images, the engines the owner does not use (OmniVoice, IndexTTS, CosyVoice, MOSS, VoxCPM, Supertonic, PocketTTS, dots.tts, Confucius4, and so on), the store/installer workflows under `.github/workflows` (Mac App Store, Microsoft Store, Snap) and about 10,000 upstream tests. **Removing code with no use at all is allowed** when it is separable and tests stay green, but it is a separate, reviewable commit each time; nothing in the training export may move.

## 6. Running things

```powershell
cd electron; bun run dev                      # the app (the owner runs it)
.venv\Scripts\python.exe -m pytest <scoped tests below>
cd electron; bun run typecheck; bun run locale:check; bunx vitest run <folder>
```

Scoped tests for this fork's work (all green at the last run):
`tests/backend/engines` (Fish story, jobs, dataset, Qwen export, runner, LoRA, instruct), `tests/test_qwen3_narrator.py`, `tests/test_qwen_studio.py`, `tests/test_qwen_story.py`, `backend/tests/test_model_load_shutdown.py`, `tests/test_changelog_style.py`, `tests/test_locale_parity.py`, `tests/test_structure_doc.py`; frontend: `qwen-story`, `qwen-playground`, `fish-story`, `fish-dataset`, `transcriptions`, `settings`, `components`, `features/home`. The full upstream suite has about 80 failing tests that fail the same way before this fork's work (Windows/ffmpeg/AMD/Mac environment problems); they are out of scope.

Conventions that bite: edit CRLF files carefully (`.json` locale files keep CRLF; no `sed -i`); every new UI string needs a key in all 21 locale files; new user-visible work gets a one-line `[Unreleased]` CHANGELOG entry ending with `— thanks @mkramer0820!` (a test enforces the style); docs change in the same commit as the behaviour.

## 7. Not verified (never run on real hardware)

The whole Qwen playground and story pages in the real app; LoRA training and the adapter merge on a real adapter; `--instruct` conditioning; local training (`train_local.sh`); the other built-in effects besides the inner-monologue recipe (tune by ear); Fish drama-3 against the real API; the one-click Qwen3-TTS install; emotion presets on the fine-tuned narrator (an instruction moved the voice in the owner's tests).

## 8. Open items, roughly in order

1. Audio → transcript → clips → training set (section 4.4).
2. A per-line Fish model picker and "re-apply an effect to a finished run" (both backends already record what is needed in `manifest.json`).
3. More debloat only if asked: the non-CUDA/Mac/AMD code paths, unused engines, upstream tests.
4. Merge `ext/qwen3-story` into `ext/fish-api` when the owner says.
