# Agent docs: start here

Works for any coding agent or human. Read in this order and stop when you have what you need.

| Need | Read | Size |
|---|---|---|
| Where the work stands, what is done, what is not ready, what to do next | [SESSION_STATE.md](SESSION_STATE.md) | ~2 min |
| **Qwen3 fine-tuning: what happened on Colab, what we learned, current recipe (read before touching the runner or advising on settings)** | [QWEN3_TRAINING_STATE.md](QWEN3_TRAINING_STATE.md) | ~5 min |
| **Setting up a Colab runtime: Python, torch, CUDA, the FlashAttention wheel, and every trap we hit (read before any new training phase)** | [../engines/qwen3-tts-colab-runtime.md](../engines/qwen3-tts-colab-runtime.md) | ~5 min |
| **Next job: run the fine-tuned voice locally on the RTX 3070 with faster-qwen3-tts as a backend (ready-to-paste prompt, facts, plan). Default weights: the owner's Run 2, epoch 09 (the 3-hour run, `checkpoint-epoch-9`)** | [NEXT_SESSION_LOCAL_BACKEND.md](NEXT_SESSION_LOCAL_BACKEND.md) | ~3 min |
| A ready prompt to start the next session | [NEXT_SESSION_PROMPT.md](NEXT_SESSION_PROMPT.md) | ~1 min |
| Rules, limits, check commands, a code map (for delegated workers; everyone benefits) | [SUBAGENT_BRIEF.md](SUBAGENT_BRIEF.md) | ~3 min |
| Writing a prompt for a worker you delegate to | [SUBAGENT_PROMPT_TEMPLATE.md](SUBAGENT_PROMPT_TEMPLATE.md) | ~1 min |
| Layout, git rules, definition of done, when to ask the owner | [fish-project-organization.md](fish-project-organization.md) | ~5 min |
| The deep reference: invariants, data facts, Qwen3 format, Colab runner design | [../engines/fish-audio-handoff.md](../engines/fish-audio-handoff.md) | read only the section you need |
| How the features behave for a user | [../engines/fish-audio.md](../engines/fish-audio.md) | on demand |

The repo-wide contract is `AGENTS.md` (mirrored in `CLAUDE.md`; the two are kept in sync). It outranks everything here.

Older agent docs in this folder (`issue-tracker.md`, `triage-labels.md`, `domain.md`) are about GitHub issues and domain terms and are unrelated to the Fish work.

Rules of thumb that save the most effort:
- Do not read whole files over about 500 lines. Open the function the code map names (search for it).
- If something here disagrees with the code, the code wins; fix the doc in the same change.
- If you are unsure what the owner wants, ask before building. SESSION_STATE.md lists the open questions.
