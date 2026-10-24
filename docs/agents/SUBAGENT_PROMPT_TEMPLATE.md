# Prompt template for delegating a task

Fill in the angle brackets and paste. The shared rules live in [SUBAGENT_BRIEF.md](SUBAGENT_BRIEF.md), so do **not** paste them again. The aim is a small prompt and a small search: delegated tasks so far used roughly 115k to 380k tokens each (about 170k when the prompt was focused). A good target is under about 150k.

```
You are working on the VoiceStudio repository (<path>). Work in your own git worktree or copy, on a new branch <name> created from <branch>@<commit>.

READ FIRST, in this order, and nothing else until you need it:
1. docs/agents/SUBAGENT_BRIEF.md (rules, limits, commands, code map). It replaces reading AGENTS.md/CLAUDE.md in full.
2. docs/agents/SESSION_STATE.md (current state, 2 minutes).
3. <at most 3-6 specific files or functions for this task, each with the function name to search for>. Do not read files over ~500 lines in full.

GOAL: <one paragraph: the user-visible outcome, and why it matters to the owner.>

CHANGE: <numbered list of concrete changes, each naming the file/function it lives in.>
CONTRACTS TO KEEP: <request fields, file names or behaviors that must not change.>
TESTS: <the tests that must exist and pass, named by behavior.>
DOCS: <which doc sections to update; one changelog line.>

LIMITS: as in SUBAGENT_BRIEF.md, plus: <anything task-specific, e.g. "do not edit export-dialog.tsx; another worker owns it".>
OWNER QUESTIONS: if <ambiguity>, stop that part, write the options and your recommendation, and continue with the rest.

WHEN DONE: update your row in the project board of docs/agents/SESSION_STATE.md (status, proof, what is left) in the same commit.

REPORT (under 500 words): files changed and commit hashes; tests run and results; what you could not verify; decisions needed. Details go in docs, not in the reply.
```

## Tips that keep it cheap
- Name the exact function or component to open (use the code map in the brief); say "search for it, do not read the file".
- Split by file ownership. Two workers on the same files means merge conflicts in locale files and docs.
- Re-use the staged-edit, facet and export patterns already in the code instead of describing new ones.
- State the finish line (the tests that prove it) so the worker stops exploring.
- Make the worker mark its project row done (or partly done) in SESSION_STATE.md, so the next session starts from an accurate board.
- Ask for the report in the shape above; long reports fill your own context.
- When a worker finishes, verify its branch yourself (author, no attribution lines, file list, checks) before relaying it.
