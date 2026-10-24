# Prompt for the next session: first real Colab training run

Copy everything inside the box into a new agent session (any agent). It is written to need no other context.

```
You are continuing work on the VoiceStudio repository (a local-first voice app). The owner is going to fine-tune Qwen3-TTS on Google Colab using audio they generated with Fish Audio. The tooling is built and tested but has never run on Colab. Your job in this session is the FIRST REAL RUN, step by step with the owner, and fixing what breaks.

READ ONLY THESE, in order, and nothing else until a step needs it:
1. AGENTS.md (the repo contract; CLAUDE.md is a mirror) - skim the rules.
2. docs/agents/README.md, then docs/agents/SESSION_STATE.md (current state, done/not done/unverified, open questions).
3. docs/agents/SUBAGENT_BRIEF.md (limits, check commands, code map, git rules).
Only when you must change or debug something: the section of docs/engines/fish-audio-handoff.md named in SESSION_STATE.md (10b = Qwen3 export format facts, 10c = Colab runner), and the specific function in backend/engines/fish_api/qwen3_colab_runner.py (a 2,300-line file: search for the function name, do not read it whole).

STATE IN BRIEF: branch ext/fish-qwen3-export (local only, not pushed). The export dialog on the "Training data" screen writes a Qwen3-TTS folder plus a ZIP, a Colab notebook (Qwen3_TTS_Colab_Training.ipynb) and one runner script (voicestudio_qwen3_colab.py). The notebook caches models/wheels/audio codes on Google Drive (My Drive/VoiceStudio-training/), checks dependencies and data, shows live progress, copies each finished epoch to Drive only after verifying it, and has a SMOKE_RUN timing mode that projects time and compute units. Training runs on a 100-compute-unit Colab budget; A100 is the planned first runtime.

HARD RULES (more in SUBAGENT_BRIEF.md):
- Do NOT start a Colab session, upload data, run training, or spend compute units without the owner's explicit go for that exact step.
- Do NOT list, open, search, read, write or delete anything on G:\ (the owner's Google Drive, with personal documents) except a path the owner names for this task.
- Do NOT push, open a pull request, or delete branches unless told. No agent attribution in commits.
- Never overwrite or delete data on Drive by default.
- If you are unsure what the owner wants, or two docs disagree, ask. Do not guess about money, data or irreversible actions.

START BY asking the owner these (skip any answered in SESSION_STATE.md "Owner decisions"):
1. Has the export been made? Where is the ZIP (name, folder), and what speaker name did they choose?
2. Which Colab runtime did they pick (A100 40/80 GB, G4, other), and how many compute units are left?
3. Clip length range and how many minutes of audio did they export (pilot ~60 min?)
4. Do they prefer driving Colab from the browser notebook or from the `colab` command-line tool (its README lists Linux/macOS only)?

THEN, with the owner running Colab (you guide and read their pasted output):
1. Confirm the ZIP and the notebook are in My Drive/VoiceStudio-training/ (datasets/ for the ZIP).
2. Open the notebook, set the parameters cell (DATA_SOURCE, SPEAKER_NAME, EPOCHS, BATCH_SIZE...), and run it with SMOKE_RUN = True first. Read the printed GPU table, dependency table, data check, seconds per batch, peak memory, and the projected epoch time and compute units.
3. Decide with the owner whether the projection fits the budget; adjust batch size/epochs; then run for real (SMOKE_RUN = False).
4. While training, check the progress lines, the heartbeat, and that each finished epoch is copied to Drive and verified. Remind them to disconnect and delete the runtime when done.
5. Help them listen to the final sample and epoch checkpoints; report what needs changing.

WHEN SOMETHING BREAKS: reproduce from the pasted log, find the function in qwen3_colab_runner.py, fix it with a failing-then-passing test (tests/backend/engines/test_fish_qwen3_colab_runner.py), keep the notebook in sync (a drift test enforces it), update docs/agents/SESSION_STATE.md and the handoff section 10c, run the checks listed in SUBAGENT_BRIEF.md, and commit locally.

FINISH BY updating docs/agents/SESSION_STATE.md (set the status of the "First real Colab run" row and any row you touched; Done only if it really ran) so the next session can start from it: what ran, what worked, what failed, what is still unverified, the next step. Keep it under about 140 lines. Your final report to the owner: under 300 words.
```
