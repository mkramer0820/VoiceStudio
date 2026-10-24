# Worker briefing (read this instead of exploring)

For any agent or person doing a task on the Fish pipeline, whether delegated or not. It replaces re-reading the repo rules for each task. If something here contradicts the code or `AGENTS.md`, tell the owner and fix this file.

## Ground rules
- The repo contract is `AGENTS.md` (mirrored in `CLAUDE.md`). Docs and `CHANGELOG.md` change in the same commit as behavior; changelog lines are one line each (a test enforces style, about 400 characters at most). UI text goes through i18n in **all 21** locale files. A change has a test that fails before and passes after.
- Commits carry only the configured git identity: no `Co-authored-by`, no "Generated with" line, no session links, no agent author. Commit locally in small commits; **do not push** unless told.
- Never commit personal folders (`_gpt_rewrite_chapter_prompt/`, `_prompt_non_repo_related/`, `.claude/`). They are ignored locally.
- Do not change: how Fish audio is saved (byte-for-byte), request modes, the run-folder layout and manifest fields, existing export file names, the request contracts of the dataset API. Ask the owner before adding a dependency, changing training defaults, or anything that costs money or sends data to a service.

## Hard limits
- `G:\` is the owner's Google Drive with personal and financial documents. Do not list, open, search, read, write or delete anything there unless the owner names a path for this task. Use temp folders for trial exports.
- `D:\fish_tts\input` holds the owner's Fish renders: read-only, small samples only, never modify.
- No Colab sessions, uploads, training runs, model downloads or network calls unless the task says so. Reading public documentation to verify a fact is fine; record what you read.

## Commands
Backend (repo root): `python -m pytest tests/backend/engines tests/backend/api/test_fish_audio_route.py -q -k "fish or qwen3 or lora"`, plus `python -m pytest tests/test_structure_doc.py tests/test_changelog_style.py tests/test_changelog_parse.py tests/test_locale_parity.py -q` and `python scripts/check-docs-drift.py`.
Frontend (`electron/`): `bun run typecheck`, `bun run locale:check`, `npx vitest run src/renderer/src/features/fish-dataset src/renderer/src/features/fish-story`. (`npx tsc` is not the compiler.)
Known failures that are not yours: one torchaudio-load test (`test_finished_provider_audio_skips_route_dsp`) and the hardcoded-CJK test (`story-model.ts`).

## Environment traps (Windows, Git Bash or PowerShell)
- Locale JSON files are CRLF. A plain `git add` can rewrite them (a ~79,000-line diff). Stage with `git hash-object -w --no-filters <file>` then `git update-index --cacheinfo 100644,<hash>,<file>`, and check `git diff --cached --stat` is small. Read and write files with `newline=""`.
- Large or quote-heavy shell heredocs fail: write the script to a file and run it.
- A new git worktree has no `node_modules`. Make junctions with **absolute paths** (`cmd //c mklink //J "<worktree>\node_modules" "E:\VoiceStudio\node_modules"`, same for `electron\node_modules`). Before you finish, unlink them with `cmd //c rmdir <absolute path>`. Never delete through a junction; confirm the real `node_modules` still has its contents.
- The shell may stay in the worktree folder between commands: use absolute paths.
- Tools may move a long command to the background; wait for it instead of rerunning.

## Code map (open the named function; do not read whole large files)
Backend `backend/engines/fish_api/`:
- `client.py` `build_body` (the Fish request), `config.py`, `story.py` (draft schema), `story_jobs.py` (`preview`, `start_render`, `resume_render`, `load_run`), `library.py` (voice library), `collection.py`.
- `dataset.py` (~770 lines): `load` (index), `select` (all filters), `facets` + `select_ignoring` (counts that ignore their own dimension), `export` (formats), `normalize_tags`/`strip_tags`, `apply_loudness`, `LevelReport`.
- `dataset_quality.py`: `measure` (peak, level, pauses, silences), analysis job.
- `qwen3_export.py` (~570): `export` (the Qwen3-TTS folder), `rank_references` (one reference ranking).
- `qwen3_colab.py`: writes the helper files and the notebook into each export (`write_default_notebook`); `qwen3_colab_runner.py` (**2,300 lines, standard library only**): the single Colab implementation. Entry points: `params_from`, `setup`, `data_check`, `train`, `timing_run`, `results`, `final_sample`; `ensure_flash_attn`, `ensure_codes`, `copy_verified`/`checkpoint_finished` (safe checkpoint copy), `attention_choice`/`detect_gpu` (GPU table), `report_projection` (time and units).
- `training_bundle.py` (OmniVoice bundle), `training.py` + `training_eval.py` (local OmniVoice runner, evaluation record). In-repo LoRA: `omnivoice/training/lora.py`.
Routers `backend/api/routers/`: `fish_audio.py`, `fish_story.py`, `fish_dataset.py` (lines, facets, export, reference-candidates, analyze), `fish_training.py`.
Frontend `electron/src/renderer/src/features/`:
- `fish-story/`: `fish-story-page.tsx`, `use-fish-story.ts`, `settings-panel.tsx`, `jobs-dialog.tsx`, `voice-library.tsx`, `voices-dialog.tsx`.
- `fish-dataset/`: `fish-dataset-page.tsx`, `use-fish-dataset.ts` (builds the request specs: `baseSpec`, `selectionSpec`, `spec`), `dataset-explorer.tsx` + `chapter-picker.tsx` + `explorer-staging.ts` (staged choices with Apply), `export-dialog.tsx` with `export-options.ts`, `export-qwen.tsx`, `export-reference.tsx`, `export-loudness.tsx`, `export-tags.tsx`, `export-training.tsx`, `clip-player.ts`, `train-dialog.tsx`.
Other: `scripts/export_qwen3_tts.py` (CLI over the same export), `notebooks/Qwen3_TTS_Colab_Training.ipynb` (default notebook; a drift test keeps it equal to the generator), strings in `electron/src/renderer/src/i18n/locales/*.json`.
Tests: `tests/backend/engines/test_fish_*.py` (dataset, qwen3 export, qwen3 colab runner, training, story...), `tests/backend/test_omnivoice_lora.py`, vitest files beside the features.

## When you finish
Update your row in the project board in [SESSION_STATE.md](SESSION_STATE.md) in the same commit as your work: set the status (Done only if fully covered; otherwise "Done, unverified" or "In progress" and say what is left), name the branch or commit, and move anything unfinished to "Still open". Add a row for new work. Keep that file under about 160 lines.

## How to report
Under about **500 words**: what changed (files and commit hashes), the tests you ran and results, what you could not verify, decisions needed from the owner. Put detail in `docs/` (handoff or SESSION_STATE.md), not in the reply. Say plainly anything incomplete.
