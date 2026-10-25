# Prompt for the next session

Copy everything inside the box into a new agent session (any agent). It needs no other context.

```
You are continuing work on the VoiceStudio repository (a local-first voice app). The owner fine-tunes Qwen3-TTS on Google Colab with narrator audio generated through Fish Audio. The tooling works; two real training runs and a 3-hour run exist.

READ, in order, and nothing else until a step needs it:
1. AGENTS.md (the repo contract; CLAUDE.md mirrors it) - skim the rules.
2. docs/agents/QWEN3_TRAINING_STATE.md (what happened on Colab, root causes, the recipe, next steps).
3. docs/engines/qwen3-tts-colab-runtime.md (runtime, FlashAttention wheel, every trap) - before setting up or changing any runtime.
4. docs/agents/SESSION_STATE.md (board, decisions, open questions) and docs/agents/SUBAGENT_BRIEF.md (limits, commands, code map, git rules).

BRANCH: ext/fish-api (pushed). Work there; commit with the owner's git identity only (no agent attribution lines). Do not open a PR or push main unless asked.

HARD RULES:
- Before advising any training setting, read the model's code and its GitHub issues. Never call a script default "the official setting".
- Do NOT start Colab sessions, upload, train or spend compute units without the owner's explicit go for that exact step.
- Do NOT list, open, search, read or write anything on G:\ (the owner's Google Drive with personal documents) except a path the owner names.
- When the owner is in a live Colab session and asks for a script or cell, give the full text FIRST; do other work afterwards.
- Tell the owner at the start of every change whether it affects a notebook they have running.
- Never overwrite or delete data on Drive by default. "Local" means Colab's /content (wiped on disconnect); models are loaded from /content, Drive is the backup.

START BY asking the owner: which epochs of the latest run sound best and what is wrong with them. Then make ONE targeted change at a time (see "Next steps" in QWEN3_TRAINING_STATE.md).

WHEN SOMETHING BREAKS: reproduce from the pasted log, find the function in backend/engines/fish_api/qwen3_colab_runner.py, fix it with a failing-then-passing test, regenerate the committed notebook, and update the docs in the same commit.

FINISH BY updating docs/agents/SESSION_STATE.md and QWEN3_TRAINING_STATE.md (what ran, what worked, what failed, what is next).
```
