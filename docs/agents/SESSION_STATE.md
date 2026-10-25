# Session state: Fish pipeline, training data and the Qwen3-TTS Colab runs

**Read [QWEN3_TRAINING_STATE.md](QWEN3_TRAINING_STATE.md) first for anything about Colab or Qwen3 training, and [../engines/qwen3-tts-colab-runtime.md](../engines/qwen3-tts-colab-runtime.md) before setting up a runtime (Python, torch, CUDA, the FlashAttention wheel, every trap).** Details of the Fish side live in [../engines/fish-audio-handoff.md](../engines/fish-audio-handoff.md). If a line here is wrong, correct it when you find out.

## The project in five lines
- VoiceStudio is a local-first voice app. This work is a pipeline around Fish Audio's cloud TTS for the owner's private audiobooks.
- Stage a story, render it through the Fish API, keep every response untouched in a run folder, and remember voices (the voice library).
- The **Training data** screen indexes those run folders (about 2,500 narrator clips in a folder of chapters), lets the owner filter, listen, measure and choose clips, and exports a training set.
- Training happens **on Google Colab** (the owner's GPU is too small). The export writes the dataset, a notebook and the runner; the notebook alone (the runner is inside it) plus the dataset zip is all Colab needs.
- Target: fine-tune **Qwen3-TTS 1.7B Base** on the narrator voice, then other speakers with the same notebook.

## Where the code is
- Branch **`ext/fish-api`** holds everything (the Qwen3 export, loudness matching, reference chooser, staged Explorer, the Colab runner and notebook, the docs). It was merged from `ext/fish-qwen3-export` and pushed to origin on 2026-10-25. The older branches `ext/fish-qwen3-export` and `ext/fish-training-data` were fully contained in it and were deleted on 2026-10-25 (locally and on origin). `main` has none of it. No pull request exists; do not open one or push `main` unless the owner asks.
- Colab does not need our branch: the export carries the runner, and Colab clones only the public Qwen3-TTS repository.

## Project board (update your row when you finish)

Status words: **Not started**, **In progress**, **Blocked** (say why), **Done** (tests pass), **Done, unverified** (built and tested with fakes or component tests, never run on real hardware or viewed on screen), **Done, verified on Colab**. Do not write Done for anything only partly covered.

| Project | Status | Where / proof | What is left |
|---|---|---|---|
| Fish engine, story board, preview, jobs, voice library, apply-settings | Done | `fish_api/`, `fish-story/`; backend and vitest tests | nothing |
| Training data screen, measure audio, Explorer with staged Apply and chapter checklist | Done, used by the owner for two exports | `fish-dataset/`, `dataset.py`; tests | nothing known |
| Export formats (jsonl, `.lab`, LJSpeech, OmniVoice bundle, Qwen3-TTS folder) | Done | `dataset.py`, `qwen3_export.py`; tests | nothing |
| Qwen3 export: loudness match, reference chooser, **1 s trailing silence by default**, validated defaults (10 epochs, batch 4, lr 1e-6), notebook with the FlashAttention wheel pre-filled, refuses to export on an outdated backend | Done | `qwen3_export.py`, `qwen3_colab.py`; tests | a checkbox for the silence in the dialog (the payload field `trailing_silence_s` works) |
| Colab runner and notebook (runtime check, wheel handling, patched training script, fast setup, per-epoch samples) | **Done, verified on an A100** (runs of 2026-10-24/25) | `qwen3_colab_runner.py`, `notebooks/`, [runtime guide](../engines/qwen3-tts-colab-runtime.md) | other GPUs, `USE_VENV` |
| **Colab training runs** | Run 1 (script defaults) gave noise; run 2 (patched script, lr 1e-6) gave a narrator-like voice, epochs 4 to 8 best; run 3 (3 hours, 10 epochs) done or in progress | [QWEN3_TRAINING_STATE.md](QWEN3_TRAINING_STATE.md) | compare run 3 by ear, then one change at a time |
| Fast per-epoch sample test (faster-qwen3-tts, CUDA graphs) | Done, verified on an A100 (about 0.9x real time per worker, 3 to 6 workers) | owner's `epoch_test_fast.py` (see the state doc) | not in the repo; a stable copy could move to `scripts/` |
| FlashAttention wheel finder | Done | `scripts/find_flash_wheel.py`; test | nothing |
| Trailing-silence padder for older exports | Done | `scripts/pad_qwen3_dataset.py`; test | nothing |
| OmniVoice LoRA path and local Train dialog | Done, unverified | `omnivoice/training/lora.py` | not part of the Qwen plan |
| Agent docs (this folder) | Done | `docs/agents/` | keep this file current |
| Push branch / pull request | Branch pushed 2026-10-25; no PR | n/a | PR only if the owner asks |
| Model locations in Settings -> Storage (shared list, `/api/settings/storage/model-locations`, native folder picker, playground reads it, "model not found" warning on takes) | Done, unverified (never seen rendered) | `backend/core/model_locations.py`, `model-locations-settings.tsx`, `authorizeModelLocation` IPC; `tests/test_model_locations.py`, vitest, IPC test; [brief](MODEL_LOCATIONS_BRIEF.md) | owner check on screen; other engines can read the list the same way; rows cannot be renamed in the UI (API supports it); no Browse button on the output folder |
| Qwen playground page (`/qwen-playground`) | Done, unverified (never seen rendered; no real run) | `electron/src/renderer/src/features/qwen-playground/`, [user page](../engines/qwen3-playground.md); vitest (model + page), typecheck, locale:check | manager's real GPU session; translations of the `qwenPlayground` namespace (English placeholders in 20 locales); no folder-browse button (needs a new IPC); story mode (prompt 2) |
| Load a trained Qwen model into the app | Not built | n/a | after the best epoch is chosen |

## Verified on real hardware, and not
- **Verified on a Colab A100 40 GB (Python 3.13, torch 2.11 cu130):** the runtime check, the FlashAttention wheel install and GPU test, model download, code preparation, the timing run, training with progress lines, per-epoch checkpoint copy to Drive with verification, sampling, the fast sample test.
- **Not verified:** the isolated Python 3.12 environment (`USE_VENV`), Blackwell (G4) and T4 paths, the other GPUs' batch sizes, interrupting a run on Colab, the Explorer/export dialog by eye beyond what the owner used, compute-unit rates (the owner's figures).
- **Unproven claims to avoid:** why one epoch sounds better than another; that the patched training script alone (vs. lr and padding) fixed the noise: run 2 changed all three together.

## Not built (on purpose)
True resume with optimizer state (the official script saves weights only), TensorBoard (the official script writes no events), a per-file hash manifest, loading a trained Qwen model into the app, Fish S2 local, a Qwen validation split (the official scripts have none).

## Owner decisions already made
- Colab, not local; target `Qwen/Qwen3-TTS-12Hz-1.7B-Base`; narrator first. Clips 4 to 20 s, 24 kHz mono 16-bit, all Fish tags removed, loudness matched (default -22 dB), speaker `SS_Narrator`, one shared reference clip.
- Everything cached and saved on Google Drive, in the folder **`Qwen3TTS`** (the owner's Drive is mounted locally as `G:\`). Never lose data: no overwriting, no deleting by default. "Local" means Colab's `/content`, which is wiped when the runtime is deleted; Drive is the backup, and models are loaded from `/content`.
- Wants progress logs that prove the job is alive, and every epoch sampled and listened to. Colab budget: 100 compute units; the A100 40 GB is the runtime in use.
- Never start Colab sessions, upload, train or spend units without the owner's explicit go; give scripts and cells as text first, before any other work, when the owner is sitting in a live session.

## Still open (ask the owner)
1. Which epochs of the 3-hour run sound best, and whether more data helped.
2. Whether to try lr 5e-6, freezing the speaker encoder, or more epochs (see the state doc's next steps).
3. Whether `KEEP_ALL=True` (every epoch kept, about 4.2 GB each) is fine for their Drive space.

## What to do next
Part 1 of the local backend on the owner's RTX 3070 is done (engine `qwen3-narrator`; results in item 00 of [QWEN3_TRAINING_STATE.md](QWEN3_TRAINING_STATE.md), plan in [NEXT_SESSION_LOCAL_BACKEND.md](NEXT_SESSION_LOCAL_BACKEND.md)). Next: the owner listens to the samples, then Part 2 (the UI). For further training work use the "Next steps" in [QWEN3_TRAINING_STATE.md](QWEN3_TRAINING_STATE.md). Fix whatever breaks in `qwen3_colab_runner.py` with a failing-then-passing test in `tests/backend/engines/test_fish_qwen3_colab_runner.py`, regenerate the committed notebook (`write_default_notebook`), and keep the docs in step.

## Traps already hit
- Windows + Git Bash: a plain `git add` can rewrite the locale JSON files' CRLF line endings (a 79,000-line diff). Stage them with `git hash-object -w --no-filters <f>` then `git update-index --cacheinfo 100644,<hash>,<f>`. Read and write files with `newline=""`.
- Large shell heredocs fail; write a script file instead. `npx tsc` is not the compiler; use `bun run typecheck`.
- A worktree's `node_modules` may be a junction to the real one. Create and remove junctions with absolute paths; unlink with `cmd /c rmdir` before removing the worktree. Never delete through a junction.
- The shell's working directory can stay inside a worktree between commands: use absolute paths.
- Google Drive can hold two folders with the same name. Colab then resolves the name to one of them, possibly an empty copy. The notebook and runner used to create the Drive root folder when they could not see it (sync lag), which made that second copy; they now stop with a clear message and create nothing. If you see `Qwen3TTS (1)` in `G:\My Drive`, an old run made it.
- The owner's Drive holds personal documents. Never list, read or search `G:\` beyond the one folder the owner names.
- Known failing tests that are not ours: one torchaudio-load test and the hardcoded-CJK test (`story-model.ts`).
- Quirk: choosing no chapters sends `chapters: [""]` (an empty list means "all"). Explorer counts include clips unticked in the clip list.

## Checks before you hand back
See [fish-project-organization.md](fish-project-organization.md#checks). Always: backend Fish tests, locale/changelog/structure/docs-drift tests, `bun run typecheck`, `bun run locale:check`, and the Fish vitest folders.
