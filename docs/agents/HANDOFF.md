# Fork handoff: what this project is and where things are

Read this first. It is for the next person or agent who picks the fork up. How to use the pages as a user: [Story studios guide](../story-studios-guide.md). Last updated 2026-11-06 (after the Fish replace-in-place work and the first look at `sound_effects/`).

## 0. If you are a new session: do this before anything else

1. Read this file, then [STORY_MIX_BRIEF.md](STORY_MIX_BRIEF.md) and, if the owner picks the sound-effects work, section 9 below and the script it describes.
2. Check the state: `git status` (the folder `sound_effects/` was untracked at the time of writing and is the owner's work in progress: do not commit or delete it unasked), `git log --oneline -10`, and that the scoped tests in section 6 pass.
3. **Ask the owner which part to work on next** with the question tool, using the menu in section 8 (four options per question: put the main parts in the first question and the smaller ones in a second). Do not start building before they answer.
4. After the answer, **restate the objectives of the chosen part** (copy them from section 8, adjust if the owner changed anything) and get a yes before you write code. Then work in small commits, say what is unverified, and update this file in the same commit as the behaviour.

## 1. What the fork is for

A **Windows + NVIDIA CUDA** studio for building and using multi-voice stories with **Fish Audio** (cloud API; **the go-to studio right now: put Fish first when choosing what to work on**) and **Qwen3-TTS** (your own fine-tunes and LoRAs, the test bench and the next voice), plus the tools that make training data for the next Qwen fine-tune. It started as a fork of `debpalash/VoiceStudio` (an open-source voice cloning, dubbing and dictation app). Much of upstream is still in the repo and is **not part of this pipeline**; see section 5.

Owner preferences that shape every decision:

- Windows and CUDA only. **AMD/ROCm, Mac (MPS/MLX) and Linux-server work is out of scope**; it stays in the code as a later fallback and must not be made impossible, but nothing here is built, tested or documented for it.
- **No model loads unless a feature is used.** Nothing loads at startup (the TTS model, dictation ASR and watermark generator are opt-in: `OMNIVOICE_PRELOAD_TTS=1`, `OMNIVOICE_PRELOAD_CAPTURE_ASR=1`, `OMNIVOICE_PRELOAD_WATERMARK=1`). The owner does not use OmniVoice (the upstream default engine) and does not want it on the GPU.
- Models come only from **Settings → Storage → Model locations** (plus the Hugging Face cache). The pages never add a model-path box. The owner's fine-tunes are in `E:\Qwen3TTS_Checkpoints` (registered as "Qwen3 fine-tunes").
- **Do not touch the training-export code** (section 4.3) unless asked; it feeds the owner's real fine-tuning runs.
- **Fish tags are the owner's and have always worked: never rewrite or "fix" a Fish `[tag]`.** Only the app's own `{Effect}` tags are removed before text goes to Fish.
- The UI offers choices; it never decides which model or engine runs something.
- Licence-registry governance and upstream's cross-platform rules are not the owner's concern in this fork ("my project, my fork"), but new features must not make other platforms impossible later.

## 2. Branches and safety

| Branch | What it is |
|---|---|
| `main` | The fork's main line (everything in this document). `origin` = `mkramer0820/VoiceStudio`. |
| `v1` | The old `main` (commit `f47221a1`, upstream's state when the fork started), kept as-is. |
| `ext/qwen3-story` | The branch `main` was fast-forwarded from; same commits as `main` at the time, kept for now. |
| `archive/upstream-full-before-debloat` | A snapshot taken before the UI was slimmed down: every upstream page still offered. |
| `ext/fish-api` | An older fork branch that `ext/qwen3-story` was built on. |

Commit identity: the owner's git identity only; **no Co-Authored-By or "Generated with" lines** (CLAUDE.md, enforced by `scripts/check_commit_identities.py`). The owner has asked for pushes to `main` as work finishes; never force-push. Do not open PRs unless asked.

## 3. The studios (what exists)

| Page | Route | Backend | What it does |
|---|---|---|---|
| **Fish Story** (primary) | `/fish-story` | `backend/engines/fish_api/{story,story_jobs,client,config}.py`, router `fish_story.py` | Story board for Fish Audio: voices per character, **a Fish model per narrator/character**, `{Internal}`-style **voice effects** (tag stripped before Fish, effect put on the returned audio; per character too), Voice effects editor with **Hear it**, preview of the exact requests, resume, and **fixing a finished run in place** (below). |
| Fish dataset | `/fish-dataset` | `fish_api/{dataset,qwen3_export,...}.py`, router `fish_dataset.py` | Browse the audio Fish Story produced, select clips of one speaker, **export a training set** (section 4). |
| Qwen story | `/qwen-story` | `qwen3_studio/{story,profiles,emotions,effects}.py` | Multi-voice book: voice profiles (narrator / stock / clone with a cached speaker embedding), `[Name]` and emotion tags, per-line and per-voice voice effects, per-chunk render and redo, `story.wav`. |
| Qwen playground | `/qwen-playground` | `backend/engines/qwen3_studio/playground.py`, router `qwen_studio.py` | Test bench: "takes" on any Qwen3-TTS model (custom voice, clone on Base, LoRA adapter at a chosen scale merged on the CPU into a cache, instruct). Will keep growing. |
| Transcriptions | `/transcriptions` | upstream ASR (`services/asr_backend.py`) | Transcribe, search, export text and **segments with timestamps** (SRT, VTT, timed text, JSON). |
| Integrations / API | `/integrations`, OpenAI-compatible routes | `backend/api/routers/openai_compat.py` | Kept: the OpenAI-compatible API is useful. |

Shared pieces: **voice effects** (`backend/engines/qwen3_studio/effects.py`: ffmpeg recipes as an `-af` chain or a `-filter_complex` graph, your own effects in `effects.json`, built-ins editable with reset, `{Name}` tags, `dry_folder`, `preview`; UI `features/qwen-story/effects-dialog.tsx`, used by both Fish Story and Qwen story); `qwen3_narrator` (the sidecar that runs Qwen3-TTS, CUDA graphs, one model at a time; one-click install in `services/sidecar_install.py`); `services/pronunciation.py` (one dictionary for Generate, Fish Story, Qwen playground and story); `core/model_locations.py`; the fork's menu/home filter `electron/src/renderer/src/lib/fork-profile.ts`.

### 3.1 Anatomy of a Fish run (what every later tool reads)
`<fish output root>\<story folder>\run-NNN_<yyyymmdd-hhmmss>_<job id>\` (default root `…\OmniVoice\fish_stories\renders`, overridable in Fish settings; a `.story-id` marker sits in the story folder). Inside: `NNN_<speaker or narration>_<voice name>.<wav|mp3|opus>` per Fish call, `story.wav` (plain join, wav only), `manifest.json` (per line: `index id status file speaker voice voice_name model effect effect_error dry_file duration_s sample_rate request_id request_mode request_indices text`, plus `replaced_at`/`replace_error` after a replacement), `job.json`, `story.json` (the draft the audio matches), `requests.jsonl` (one record per call, `replacement: true` for re-renders), `run.log`, `settings.json`. **Outside the run folder, next to it** (so tools that read a run folder see only finished files): `<story folder>\dry\<run>\` = untouched Fish responses of lines that carry an effect; `<story folder>\replaced\<run>\` = audio that a replacement overwrote. The owner copies run folders into `D:\fish_tts\input\<chapter>\…` for his stitcher; those copies keep the manifest.

### 3.2 Fixing a finished run in place (done)
Fish Story → Jobs → a run's lines: tick lines → **Replace** (sends them to Fish again as single calls with the board's text, voice, model, effect; new audio goes into the same run folder; manifest, job, story, request log, run log and `story.wav` are updated; old audio kept in `replaced\`; a failed line keeps its old audio) or **Re-apply effects** (no Fish call; starts from the untouched response in `dry\`). Code: `story_jobs.replace_lines`, `story_jobs.reapply_effects`; routes `POST /api/fish/story/jobs/{id}/replace` and `/reapply-effects`. Qwen story already re-renders a chunk into its run (older take kept as `_r2`).

## 4. Training: the fork's main output

### 4.1 Export (`fish_api/qwen3_export.py`, `qwen3_colab.py`)
Export dialog on `/fish-dataset` writes a folder or zip with `train.jsonl`, `audio/` (24 kHz mono PCM16, 1 s of silence appended), `reference/ref.wav`, `manifest.json` (training mode, speaker, counts, parameters, **bundle id**), `training_settings.json`, the Colab notebook, the runner `voicestudio_qwen3_colab.py`, `colab_setup.sh`, `train_local.sh`/`.ps1`, `README.txt`. Options: full fine-tune or LoRA (own `_lora` folder), `--instruct` conditioning (an `instruct` field from the clip `[tags]`), up to four speakers in one model.

### 4.2 Runner (`fish_api/qwen3_colab_runner.py`, **runner 23**)
One file does everything; the notebook and scripts only call it. Defaults: full fine-tune lr **2e-6**, LoRA lr **2e-6**, rank 16, alpha 32, adapter/merge scale **0.3** (0.0-1.0, warn above 0.5). The upstream training script (pinned commit `022e286`) is patched in a copy for text projection, label shift, sub-talker alignment, validation loss, atomic checkpoint saves, LoRA, multi-speaker and the opt-in instruct conditioning. **Colab needs an exact stack** (Python 3.13, torch 2.11.0+cu130, community flash-attn wheel); never upgrade torch in the runtime (details: [qwen3-tts-colab-runtime.md](../engines/qwen3-tts-colab-runtime.md)). Every shipped file prints the same bundle id; the runner warns when it differs from the manifest.

### 4.3 Hands off
Do not change the export format, the runner's patches or the notebook generator without the owner asking. They have a cross-layer defaults test (`tests/backend/engines/test_fish_qwen3_defaults.py`) and a committed notebook that must equal the generator's output.

### 4.4 What is **not** built
**Audio in, training set out.** The Fish dataset page works from audio Fish Story produced. There is no flow that takes *your own recordings* → transcribes → cuts clips by segment → writes a Qwen training set. The pieces exist separately: Transcriptions exports segments with start/end times, and `qwen3_export.py` knows the target format. Upstream's own data preparation (`docs/data_preparation.md`, `omnivoice/`) is for OmniVoice tokens and is not this.

## 5. What upstream left in, and what the fork hides

The UI hides these workspaces from the menu, command palette and home page: `/clone`, `/design`, `/stories`, `/audiobook`, `/dub`, `/batch`, `/calls`, `/gallery`, `/projects`, `/tools`. The list is one file, `electron/src/renderer/src/lib/fork-profile.ts` (`HIDDEN_ROUTES`); delete a path to bring a page back. The code, routes and APIs of hidden pages are untouched and still work. The home page leads with Fish Story, Fish dataset and Qwen story.

Not removed (deliberately, to keep a later merge or revert easy): Mac/MLX/MPS code, AMD/ROCm code, Docker/ROCm images, the engines the owner does not use (OmniVoice, IndexTTS, CosyVoice, MOSS, VoxCPM, Supertonic, PocketTTS, dots.tts, Confucius4, and so on), the store/installer workflows under `.github/workflows` and about 10,000 upstream tests. **Removing code with no use at all is allowed** when it is separable and tests stay green, but it is a separate, reviewable commit each time; nothing in the training export may move.

## 6. Running things

**Python environments** (do not mix them):

| Environment | Used by | Notes |
|---|---|---|
| `E:\VoiceStudio\.venv` (Python 3.11.17) | `bun run dev` and every backend test | Built by `bun run setup:api` (`uv sync` from `pyproject.toml` / `uv.lock`). Anything installed into it by hand is removed by the next sync. Contains `pedalboard` 0.9.20 (pinned `>=0.9.14,<0.9.21`: newer wheels crash on some CPUs, issue #2052) and numpy 2.2.6. |
| `E:\VoiceStudio\.venv-qwen` | the Qwen3-TTS sidecar only | torch 2.8.0+cu128, transformers 5.15.1; also created by the one-click install under the app data folder. Do not add audio-tool packages here. |
| a throwaway venv | experiments | `uv venv E:\vst_test --python 3.11`, then `uv pip install --python E:\vst_test\Scripts\python.exe pedalboard soundfile numpy`. |
| packaged app's own `VoiceStudio\project\.venv` | an installed build | independent of the checkout. |

After pulling an update, **quit the app completely so the backend restarts**: a stale backend answers new routes with "Method Not Allowed" (the effects dialog now says so).

```powershell
cd electron; bun run dev                      # the app (the owner runs it)
.venv\Scripts\python.exe -m pytest <scoped tests below>
cd electron; bun run typecheck; bun run locale:check; bunx vitest run <folder>
```

Scoped tests for this fork's work (all green at the last run): `tests/backend/engines` (Fish story, jobs incl. replace and effects, dataset, Qwen export, runner, LoRA, instruct), `tests/test_qwen3_narrator.py`, `tests/test_qwen_studio.py`, `tests/test_qwen_story.py` (also the shared effects module), `tests/test_sidecar_install.py` (the Qwen3 install cases), `tests/test_ffmpeg_local_inputs.py` (the `local_inputs_only` rule; two `test_dub_upload…` cases fail before and after this work), `backend/tests/test_model_load_shutdown.py`, `tests/test_changelog_style.py`, `tests/test_locale_parity.py`, `tests/test_structure_doc.py`, `tests/test_model_licenses.py`, `tests/test_model_license_workflow.py`; frontend: `qwen-story`, `qwen-playground`, `fish-story`, `fish-dataset`, `transcriptions`, `settings`, `components`, `features/home`. The full upstream suite had 77 failures and 11 errors out of about 10,600 tests at the last full run (2026-11-06); the ones in files this work touches fail the same way on the commit before this fork's work (Windows/ffmpeg/AMD/Mac environment problems, two ROCm install cases in `tests/test_sidecar_install.py`), a few more only appear in a full run (test-order effects). They are out of scope.

Conventions that bite: edit CRLF files carefully (`.json` locale files keep CRLF; no `sed -i`; several source files have mixed line endings, so check before a multi-line edit); every new UI string needs a key in all 21 locale files; new user-visible work gets a one-line `[Unreleased]` CHANGELOG entry ending with `— thanks @mkramer0820!` (a test enforces the style); docs change in the same commit as the behaviour; every ffmpeg command goes through `services.ffmpeg_utils.local_inputs_only()`.

## 7. Not verified (never run on real hardware or in the real app)

The Qwen playground and story pages; LoRA training and the adapter merge on a real adapter; `--instruct` conditioning; local training (`train_local.sh`); every built-in effect except the owner's inner-monologue recipe (tune by ear); Fish drama-3 against the real API; **Replace and Re-apply effects against the real Fish API and a real run folder**; the Hear it preview and the Fish line picker in the running app; the one-click Qwen3-TTS install; emotion presets on the fine-tuned narrator (an instruction moved the voice in the owner's tests); everything in section 9 (the sound-effects script was read, not run).

## 8. Next-part menu (ask the owner; objectives to restate after the answer)

Fish first. Ask these with the question tool (first question: the four main parts; second: the smaller ones).

### Main parts
1. **Stitch & mix tab** (brief: [STORY_MIX_BRIEF.md](STORY_MIX_BRIEF.md)). Objectives: (a) a paths strip with Open folder / Copy path for source, project, music, output; (b) pick a Fish run (or any folder), read it by manifest with file-name fallback, skip `dry/` and `replaced/`; (c) segments table with role, pause, trim, mute and playback; (d) stitch with his tool's behaviour (edge fades, pauses, speed, internal pauses, limiter) to FLAC/M4A/MP3/WAV; (e) per-role level and EQ (internal thoughts quieter and darker), per-segment loudness normalisation; (f) background music at a LUFS offset with fades, pause boost, optional ducking; (g) a portable `mix.json` project with relative paths, Collect files, bundle export, relink list; (h) import of his `config.json`/`settings.json`; parity check against `D:\fish_tts` on the `3210_2210_fear_of_hope` run.
2. **Sound-effects addon** (section 9). Objectives: (a) turn `reverse_reverb.py` into a tested module (plugin discovery with a clear error and a built-in fallback, no hard-coded path, non-destructive output names, explicit bit depth); (b) add a `pedalboard` effect kind to the voice-effects system so these effects show up in the editor, **Hear it**, `{Tag}` use and Fish Story's effect pipeline, with the ffmpeg kinds untouched; (c) handle effects that change a line's length (the reverse-reverb lead-in) in Fish replace/re-apply and in the stitcher (overlap into the previous pause); (d) document the VST requirement and that it is optional and Windows-first.
3. **Fish Story polish.** Objectives: a per-line Fish model picker in the chunk list; effect and replaced indicators on lines and in the run list (with a way to open `replaced/` and `dry/`); a check of `[tags]` against the S2.1-Pro and Drama 3 tag sets before a render (warn only, never rewrite); a `replaced/` clean-up and "restore previous audio" action.
4. **Audio → transcript → clips → training set.** Objectives: take a recording plus its transcript segments (with timestamps), slice with ffmpeg, accept/reject clips with playback, write them into the existing Qwen export format without changing the exporter, and keep the source/clip folders visible.

### Smaller parts
5. **Run the unverified list in the real app** (section 7) and fix what breaks, starting with Fish replace/re-apply.
6. **Qwen follow-ups** (lower priority): re-apply effects for Qwen story, playground growth, emotion/LoRA verification.
7. **Further debloat** (only if asked): separable Mac/AMD/unused-engine code in its own commits; refresh the docs.
8. **Housekeeping:** commit the `sound_effects/` folder when the owner says it is ready; keep this file and the briefs current.

## 9. Sound-effects addon: `sound_effects/` (owner's work in progress)

`sound_effects/vst_scripts/reverse_reverb.py` (untracked at the time of writing; Windows; runs in the project `.venv`, which already has `pedalboard`). It makes a **reverse-reverb "swoosh" that swells before a spoken line**:

1. read the clip, sum it to mono and centre it in stereo;
2. reverse it and pad `--lead` seconds (default 4) of silence on the end so the reverb tail has room;
3. run a reverb: `--engine nectar` loads the iZotope **Nectar 4 Reverb** VST3 (`C:\Program Files\Common Files\VST3\iZotope\Nectar 4 Reverb.vst3`, hard-coded) through `pedalboard.load_plugin` and sets `reverb_wet_mix=100`, `reverb_width=100`, `reverb_pre_delay_ms=32`, `reverb_low_cutoff_hz=70`, `reverb_high_cutoff_hz=3500`; `--engine freeverb` uses pedalboard's built-in `Reverb(room_size=0.9, damping=0.5, wet_level=1, dry_level=0, width=1)`;
4. reverse the result back so the wash builds up into the word, scale it by `--wet-db` (default -6 dB), put the dry voice `lead` seconds later, sum them and cap the peak at 0.95;
5. write `<engine>_swoosh.wav` (the wind alone) and `<engine>_combined.wav` into `--out` (default `reverb_tests` in the current folder).

```powershell
.venv\Scripts\python.exe sound_effects\vst_scripts\reverse_reverb.py line.wav --engine nectar --lead 4 --wet-db -6 --out reverb_tests
```

Things to know before productising it (read from the code, not run): the plugin path and the parameter names are not checked (use `plugin.parameters` to confirm them on his installed Nectar version, and fail with a clear message when the file or a parameter is missing, falling back to the built-in reverb); outputs overwrite each other per engine; the file is written with `AudioFile(..., "w", sr, channels)` without a bit depth, so check the written depth against the float32 data; the combined clip is **longer than the line by `lead` seconds with the voice starting `lead` seconds in**, which matters for Fish replace/re-apply (duration changes) and for stitching (the swoosh should overlap the previous pause rather than add dead time); VST3 loading is Windows/macOS-specific and must stay optional.

Suggested integration: a new effect kind (`pedalboard`) next to `chain` and `graph` in `effects.py`, executed in-process in the backend with the project `.venv`'s `pedalboard`, validated with the same **Test / Hear it** flow, stored in `effects.json`, selectable by `{Tag}` like any effect. Keep `local_inputs_only` for any ffmpeg call it makes and never load a plugin path the user did not choose.
