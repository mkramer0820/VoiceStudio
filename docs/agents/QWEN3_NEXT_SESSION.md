# Qwen3 in Story Studio: state at end of session (2026-10-27) and what to do next

Branch **`ext/qwen3-story`** (on top of `ext/fish-api`). All work is committed locally; **nothing is pushed**. Commits carry only the owner's identity (repo rule: no agent attribution lines).

## Read first

1. `docs/engines/qwen3-finetune-findings.md` (what is verified about EP09 and the stock voices, what the owner heard, what is unknown).
2. `docs/agents/QWEN3_STUDIO_UI_BRIEF.md` (API contract, plain recipe names, Prompt 2 = story mode).
3. `docs/agents/MODEL_LOCATIONS_BRIEF.md` and the "Local models" rule in `CLAUDE.md` / `AGENTS.md`.

## Done

| Piece | Where | Status |
|---|---|---|
| `qwen3-narrator` backend (sidecar, one job at a time, padding, end-of-speech capture, chunking, GPU flush) | `backend/engines/qwen3_narrator/` | Done, measured on the RTX 3070: about 2.25x real time warm, flat VRAM |
| Playground backend (registry, runner, session manifests, borrowed stock voices) | `backend/engines/qwen3_studio/`, `/api/qwen/*` | Done, tested, run for real once |
| Playground page `/qwen-playground` | `electron/src/renderer/src/features/qwen-playground/` | Built by a worker agent, tests pass, **never seen rendered, no real GPU run through the UI** |
| Project venv `.venv-qwen` (git-ignored), pins in `backend/engines/qwen3_narrator/requirements.txt` | repo root | Built, imports OK, torch sees the GPU. The sidecar never uses PATH, a base Python or the app's `.venv` |
| Model locations: Settings -> Storage list of model folders, folder-picker IPC, Qwen discovery from it | `backend/core/model_locations.py`, `settings.py`, `model-locations-settings.tsx` | Built, tests pass, **not seen rendered**. Owner's cache dir is set to `C:\Users\User\.cache\huggingface\hub` and their fine-tune folder is registered; owner believes it works |
| Fix: a checkpoint root that is itself a checkpoint is found | `b49b64b0` | Done |

## Owner decisions that stand

- Narrator = EP09, custom_voice model, speaker `ss_narrator`, no instruct. Clones = stock Base with clip and text. Hybrid dropped.
- Mapping is free (warn, never block). Borrowed stock voices (ryan, ...) are allowed with a warning. Real stock voices need the stock CustomVoice model.
- Model locations are configured only in Settings -> Storage (project rule). Engines read that list; no per-engine model path boxes.
- The owner runs the app (`bun run dev`); give commands as text, do not run the app or the GPU without being asked.

## Do next, in order

1. **Real test of the playground on the 3070** (owner, or the agent with the owner's yes; check the GPU is free: another agent is testing an emotion-control checkpoint). A Narrator take (EP09 + `ss_narrator`) and a Base clone. Fix what the owner sees.
2. **Story mode** (Prompt 2 in the Qwen brief): characters mapped to voices, `[Tag]` clone variants per character, per-character warm-up, Fish-style output layout and names, persisted assignments. Uses the model-locations list. Start it after the playground test.
3. **Feed in the emotion-control checkpoint results** (another agent is testing it) to the findings doc; it may change the voice and instruct rules.

## Training side (2026-10-27): Colab runner 19, committed as `74cfc628`, not yet run on a GPU

- An AI-written audit brief said the training script was an unpatched copy. It is not: runner 18 already applied the text-projection, double-shift and sub-talker alignment fixes (upstream #371, #278, #39). Not proven in isolation (run 3 changed lr, padding and the patch together).
- Runner 19 adds: a 5% held-out validation loss (`val_loss` column in `metrics.csv`, `Epoch N | Val loss` line), a checkpoint save written to `.tmp-checkpoint-epoch-N`, checked against the base weights (`SAVE VERIFIED` / `SAVE FAILED`, plus the share of weights changed) and renamed last, models downloaded straight to `/content` by default, and `DRIVE_SKIP_FIRST = 3` so only epochs 4 and later go to Drive (the final epoch always does).
- No clip re-export is needed: the notebook uses the newest runner (zip, notebook or Drive cache). Use the regenerated `notebooks/Qwen3_TTS_Colab_Training.ipynb`, not the older `Qwen3_TTS_Colab_Training_3h.ipynb`.
- **Next:** a short first run (`SMOKE_RUN`, 2 to 3 epochs) on the A100 to confirm `_val_loss` and `_verify_saved` work, then a full run at `LR = 2e-6` against the 1e-6 runs. Read `SAVE CHECK` first: a tiny changed share at 1e-6 would support the (inferred, untested) idea that bf16 rounds small updates away.
- Details: `docs/engines/qwen3-tts-colab-runtime.md` section 6b.

## Needs the owner's yes

- Control test for the "Ryan-like" outputs: render another empty speaker row (2900 or 3071) on EP09 and Base.
- Check whether the cached stock Base snapshot (`fd4b2543...`) is stale (a network metadata call); redownload if so.
- Download the 1.7B CustomVoice model (about 4 GB) for the real stock voices.
- Delete my leftovers: `C:\Users\User\fq_env\narrate.py`, `C:\Users\User\fq_env\out\`, and the old `fq_env` venv (now replaced by `.venv-qwen`).

## Rough edges

- About 170 new UI strings (`qwenPlayground.*`, `modelLocations.*`) are English placeholders in the 20 non-English locales.
- Model location rows cannot be renamed in the UI (the route supports it). The playground output folder has no Browse button. The old Models directory page is still confusing.
- The app does not log which Hugging Face cache directory it uses at startup (one log line would make cache problems checkable).
- Windows: `.gitignore` and some new files show LF/CRLF warnings; locale files must stay CRLF (do not use `sed -i` on them).
