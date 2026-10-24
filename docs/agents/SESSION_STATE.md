# Session state: Fish pipeline, training data and the Qwen3-TTS Colab run

Short and current. If a line here is wrong, correct it when you find out. Details live in [../engines/fish-audio-handoff.md](../engines/fish-audio-handoff.md).

## The project in five lines
- VoiceStudio is a local-first voice app. This work is a pipeline around Fish Audio's cloud TTS for the owner's private audiobooks.
- Stage a story, render it through the Fish API, keep every response untouched in a run folder, and remember voices (the voice library).
- The **Training data** screen indexes those run folders (a folder of chapters, about 2,500 narrator clips), lets the owner filter, listen, measure and choose clips, and exports a training set.
- Training happens **outside this repo's machine, on Google Colab** (the owner's GPU is too small). The export writes everything Colab needs.
- First target: fine-tune **Qwen3-TTS 1.7B Base** on the narrator voice, then other speakers with the same notebook.

## Where the code is
- Branch to use: **`ext/fish-qwen3-export`**. It holds everything below. It is **local only (not pushed)**. `origin/ext/fish-api` and `origin/ext/fish-training-data` are older and lack the Qwen3, Colab, reference-clip and Explorer work. `main` has none of it. No pull request exists. Check with `git status` and `git log --oneline -5`.
- Colab does not need our branch: the export carries a standard-library runner and a notebook, and Colab clones only the public Qwen3-TTS repository.

## Project board (update your row when you finish)

Status words: **Not started**, **In progress**, **Blocked** (say why), **Done** (tests pass), **Done, unverified** (built and tested with fakes or component tests, never run on real hardware or viewed on screen). Do not write Done for anything only partly covered. When you finish a project, change its row in the same commit as your work: status, the branch or commit that proves it, and what is left. Add a row for new work. Move anything unfinished to "Still open".

| Project | Status | Where / proof | What is left |
|---|---|---|---|
| Fish engine, story board, preview, jobs, voice library, apply-settings | Done | `fish_api/`, `fish-story/`; backend and vitest tests | nothing |
| Training data screen, measure audio, Explorer with staged Apply and chapter checklist | Done, unverified (never viewed) | `fish-dataset/`, `dataset.py`; tests | owner to look at it |
| Export formats (jsonl, `.lab`, LJSpeech, OmniVoice bundle, Qwen3-TTS folder) | Done | `dataset.py`, `qwen3_export.py`; tests | nothing |
| Loudness matching on export (default -22 dB) | Done | `dataset.py` `apply_loudness`; real-sample trial | nothing |
| Reference clip chooser (shortlist, Suggest, one reference) | Done, unverified (never viewed) | `export-reference.tsx`, `rank_references` | owner to try it |
| Colab runner and notebook (Drive cache, GPU-aware, safe checkpoint copy, timing run) | Done, unverified (fakes only) | `qwen3_colab_runner.py`, `notebooks/` | first real Colab run |
| OmniVoice LoRA path and local Train dialog | Done, unverified | `omnivoice/training/lora.py` | not part of the Qwen plan |
| Agent docs (this folder) | Done | `docs/agents/` | keep this file current |
| **First real Colab run** (timing run, then narrator pilot) | **Not started** | [NEXT_SESSION_PROMPT.md](NEXT_SESSION_PROMPT.md) | everything |
| Push branch / pull request | Not started (owner decision) | n/a | owner's call |
| Load a trained Qwen model into the app | Not built | n/a | after a trained model exists |

## Done, and how it was checked
Checked means automated tests pass (backend pytest, frontend vitest, typecheck, locale parity, docs drift). It does **not** mean seen on screen or run on a GPU.

| Area | State |
|---|---|
| Fish engine, story board, preview, jobs, resume, load past runs, voice library, apply-settings | Done, tested |
| Training data screen: index, speakers, filters, play, measure audio (peak, level, pauses, silences), select all/none | Done, tested |
| Explorer: staged multi-select with **Apply** (chapters checklist with level columns, tags with Any/All, several length bars), facets that ignore their own filter | Done, tested |
| Export formats: jsonl, Fish `.lab`, LJSpeech, OmniVoice (+ training bundle), **Qwen3-TTS folder** | Done, tested |
| Loudness matching on export (default -22 dB, editable, per-chapter before/after report) | Done, tested on a real-data sample |
| Reference clip: shortlist by star, listen, ranked Suggest, one chosen reference | Done, tested |
| Colab runner (`voicestudio_qwen3_colab.py`) and notebook: Drive cache, GPU-aware attention path, safe checkpoint copy, progress, timing run with compute-unit projection | Built, tested **with fakes only** |
| In-repo LoRA for the OmniVoice trainer and a local Train dialog | Built, tested on tiny CPU models; not used for the Qwen plan |

## Not verified (nothing below has happened yet)
- **No run on Colab or any GPU.** Every Colab path (install next to Colab's torch, FlashAttention wheels, the patched SDPA copy for Blackwell, checkpoint copy, the notebook cells) is untested.
- **The UI has never been viewed.** Layout, wording and the Explorer/export dialog were checked by component tests with mocked text.
- Compute-unit rates in the notebook are the owner's figures, not confirmed. Batch sizes and memory numbers are estimates.
- Qwen's inference call used for the final sample is taken from its README and unconfirmed.

## Not built (on purpose)
True resume with optimizer state (the official script saves weights only), per-epoch sample audio, TensorBoard, a per-file hash manifest, loading a trained Qwen model into the app, Fish S2 local, Qwen validation split (the official scripts have none).

## Owner decisions already made
- Colab, not local; target `Qwen/Qwen3-TTS-12Hz-1.7B-Base`; narrator first (pilot of about 60 minutes of audio, later more). Export audio is 24 kHz mono 16-bit, all Fish tags removed, loudness matched (owner's good range is -19 to -23 dB, default -22).
- Everything cached and saved on Google Drive (the owner's Drive is mounted locally as `G:\`; default folder `My Drive\VoiceStudio-training\`). Never lose data: no overwriting, no deleting by default.
- Wants progress logs that prove the job is alive. A fast first look: run the notebook with `SMOKE_RUN = True` before the real run.
- Colab budget: 100 compute units. The A100 is the planned first runtime; the G4 (Blackwell) is optional and uses a fallback attention path.

## Still open (ask the owner)
1. Clip length range for the pilot (the owner was weighing 3-20 s vs 8-15 s; 4 s lower bound suggested). Not decided.
2. Which GPU runtime they actually pick, and whether the timing run projection fits their 100 units.
3. Whether to push `ext/fish-qwen3-export` (and open a pull request). Do not push without being told.
4. Whether `KEEP_ALL=True` (every epoch kept, several GB each) is fine for their Drive space.

## What to do next: the first real Colab run
Use [NEXT_SESSION_PROMPT.md](NEXT_SESSION_PROMPT.md). In short: confirm the export exists and the ZIP is on Drive, open the notebook, do the **timing run first**, review its projection with the owner, then the real run, then listen to the epoch checkpoints. Fix whatever breaks (most likely install, FlashAttention or memory) in `qwen3_colab_runner.py` with a test, and report. **Never start a Colab session, upload data or spend credits without the owner's go.**

## Traps already hit
- Windows + Git Bash: a plain `git add` can rewrite the locale JSON files' CRLF line endings (a 79,000-line diff). Stage them with `git hash-object -w --no-filters <f>` then `git update-index --cacheinfo 100644,<hash>,<f>`. Read and write files with `newline=""`.
- Large shell heredocs fail; write a script file instead. `npx tsc` is not the compiler; use `bun run typecheck`.
- A worktree's `node_modules` may be a junction to the real one. Create and remove junctions with absolute paths; unlink with `cmd /c rmdir` before removing the worktree. Never delete through a junction.
- The shell's working directory can stay inside a worktree between commands: use absolute paths.
- The owner's Drive holds personal documents. Never list, read or search `G:\` beyond the one folder the owner names.
- Known failing tests that are not ours: one torchaudio-load test and the hardcoded-CJK test (`story-model.ts`).
- Quirk: choosing no chapters sends `chapters: [""]` (an empty list means "all"). Explorer counts include clips unticked in the clip list.

## Checks before you hand back
See [fish-project-organization.md](fish-project-organization.md#checks). Always: backend Fish tests, locale/changelog/structure/docs-drift tests, `bun run typecheck`, `bun run locale:check`, and the Fish vitest folders.
