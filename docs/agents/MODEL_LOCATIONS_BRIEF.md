# Model locations: one place in Settings to say where local models live

Owner decision (2026-10-25): where the app looks for local models and custom checkpoints must be set in **Settings → Storage**, matching how the Model Library and the Models directory already work. It is a project rule (see "Local models" in `CLAUDE.md` / `AGENTS.md`). Work on the current branch `ext/qwen3-story`; do not create a branch, do not push.

## Why

The Qwen playground has its own "Checkpoint root" and Python text boxes. The owner keeps stock models in the normal Hugging Face cache (`C:\Users\User\.cache\huggingface\hub`) and fine-tunes in several places (for example `C:\Users\User\.cache\custom-fine-tunes\Qwen3TTS\ss_narrator` and `E:\Qwen3TTS_Checkpoints`). The playground could not find a checkpoint when the root pointed straight at it (fixed in `b49b64b0`), and nothing tells the owner where the app is looking. Fine-tunes "may be all over the place but it should be neat".

## Read first

1. `CLAUDE.md` / `AGENTS.md` (rules, attribution, localization, docs-sync, keep main green).
2. `docs/engines/qwen3-finetune-findings.md` (Qwen facts; do not contradict it).
3. `docs/agents/QWEN3_STUDIO_UI_BRIEF.md` (the playground, its API, plain recipe names).
4. How Settings does this today: `electron/src/renderer/src/features/settings/models-directory-settings.tsx`, `storage-settings.tsx`, `model-library.tsx`; backend `backend/api/routers/settings.py` (`/api/settings/storage/models-dir`, the `authorization` token consumed with `consume(..., "models_dir")`); the preload/main code behind `bridge.files.authorizeModelsDirectory`. Copy their patterns; do not invent a new style.
5. `backend/engines/qwen3_studio/registry.py` (current discovery: `checkpoint_root`, `OMNIVOICE_QWEN3_NARRATOR_CKPT`, `_hf_caches()`, `_scan`, `read_checkpoint`).

## What to build

### 1. Backend: a shared list of model locations
- Store a list of `{id, name, path}` rows ("locations") in the app's settings (follow how the models-dir setting is stored, not a Qwen-only file). `id` is a stable slug that survives restarts.
- Routes under `/api/settings/storage/model-locations`: `GET` (rows plus per-row status: exists, number of models found, error text), `PUT`/`POST` add or replace a row, `DELETE` remove a row. Adding takes the same one-time `authorization` token as the Models directory (the folder came from the native picker). Removing a row never deletes files.
- Status per row: a quick scan with the existing discovery code (a folder that is itself a checkpoint counts; otherwise up to 6 levels deep, same skip list). Cheap, no model load, no network.
- Migration: on first read, if the Qwen studio settings has a non-empty `checkpoint_root` and the list has no row for it, add it as a row named "Imported checkpoint folder". Then stop reading `checkpoint_root` from the Qwen settings (keep the field accepted on `PUT /api/qwen/settings` for old clients but ignore it, and say so in the docstring).
- `backend/engines/qwen3_studio/registry.py`: `discover_models` reads the shared list (every row), the env checkpoint, and the Hugging Face caches (the effective models directory, plus the user-level `~/.cache/huggingface/hub` as it does now). Keep model ids stable: an id derived from the folder must not change for an existing saved session. Add a `location` field (row id or "huggingface_cache" / "environment") to each model.
- `GET /api/qwen/state` keeps working; models now carry `location` and `path`.

### 2. Electron: a folder picker for locations
- Add the IPC in `electron/src/main/` and `electron/src/preload/` next to `authorizeModelsDirectory` (same security model: the renderer only gets an opaque authorization token, never an arbitrary path write). Follow its tests.

### 3. UI: Settings → Storage → "Model locations"
- A section under the existing Models directory section, in the same visual style (`SettingsSection` / `SettingsRow`). First row: the Hugging Face cache (the existing control stays; show it here read-only with a link to it). Then the user's rows: name, path (monospace), status line ("3 models found", "folder missing"), Browse/Change, Remove. An "Add folder" button opens the native picker.
- A "Models found here" expander per row listing each model: plain label, type (`custom_voice` / `base` / `voice_design`), size (0.6B / 1.7B), speakers, path. No install or delete actions for custom locations, only "Remove from list".
- Use plain names: say "narrator model (custom voice)" and "voice-clone model (base)", not library terms. See the plain-names table in the Qwen brief.
- All text through i18n: a new `modelLocations` namespace in all 21 locales (English placeholders in the 20 others, staged the way the playground did it; keep each locale diff small). `bun run locale:check` must pass.

### 4. Qwen playground: use it
- Remove the "Checkpoint root" box from the playground setup bar. Show the models' `location` in the model dropdown labels (for example "ss_narrator, Production") and add a short link "Manage model folders" to Settings → Storage. Keep the Python box, the output folder box and the warm-up fields (they are not model locations); put the Browse button on the output folder only if the new IPC makes it cheap, otherwise leave it.
- A saved session or take whose model is no longer found shows a clear "model not found: <label>, <path>" warning on the take (warn, do not crash or block opening the session).

### 5. Docs
- `docs/engines/qwen3-playground.md` and `docs/engines/qwen3-narrator.md`: where models are set now. A short user doc section for Settings → Storage. One line in `CHANGELOG.md` under [Unreleased] (quiet one-liner, ref style as the other lines). `docs/STRUCTURE.md` / `docs/features.yaml` if a router or inventory entry changes (`scripts/check-docs-drift.py` must pass). A row in `docs/agents/SESSION_STATE.md`.

## Hard rules
- Python: use the repo `.venv` for the backend and tests (`.venv/Scripts/python.exe -m pytest ...`), never a base or system Python. Do not install anything.
- Do not use the GPU, do not load any model, no network, no downloads.
- Do not touch Fish files. Do not edit `docs/engines/qwen3-finetune-findings.md` unless you learn a new fact.
- Commits: only the configured git identity, no `Co-authored-by`, no "Generated with" lines, no session links (repo rule). Commit locally in small commits; do not push.
- Line endings: locale files and most docs are CRLF. Edit with CRLF-aware tools; do not rewrite whole files.
- Tests: fail-before/pass-after for the backend (list add/remove/status, migration, discovery from several rows, a root that is itself a checkpoint, stable ids, missing-model warning), vitest for the new settings section and the playground change, an IPC test next to the existing one. Run `bun run typecheck`, `bun run locale:check`, vitest for the touched folders, `pytest tests/test_qwen_studio.py tests/test_qwen3_narrator.py tests/test_locale_parity.py tests/test_structure_doc.py tests/test_changelog_style.py` (and any settings-router tests), and `scripts/check-docs-drift.py`.

## Report (under 400 words)
Commits, files changed, test results, what you could not verify (you cannot see the UI), the English-placeholder locale keys, and decisions you made that the owner should know about.
