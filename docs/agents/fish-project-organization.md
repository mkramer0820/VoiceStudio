# Fish pipeline and training data: project organization and agent guidance

How the Fish work is laid out, how to work on it safely, and what a new agent session should do first. The current state of the code (what exists, what is not built, what is unverified) is in [../engines/fish-audio-handoff.md](../engines/fish-audio-handoff.md); user-facing behavior is in [../engines/fish-audio.md](../engines/fish-audio.md). `CLAUDE.md` and `AGENTS.md` still rule; this file adds the Fish-specific parts.

## What this area is

A local pipeline around Fish Audio's cloud TTS, for the owner's private audiobook workflow:

1. **Generate:** stage a multi-voice story, render it through the Fish API, keep every response untouched in a run folder.
2. **Remember:** a voice library of the owner's characters, with Fish's own details beside them.
3. **Explore:** the Training data screen indexes those run folders, groups clips by speaker, filters, plays and measures them.
4. **Export:** write a training set (several formats) and, for OmniVoice, a portable bundle with configs, a run script and a Colab guide.
5. **Train:** mostly outside this repo's machine (Colab). An in-repo LoRA path and an opt-in local runner exist for a capable GPU.

Everything except step 1's render and the explicit Fish lookups is local. Nothing leaves the machine without an explicit button.

## Where things live

| Concern | Backend | Frontend | Tests |
|---|---|---|---|
| Fish engine, request building | `backend/engines/fish_api/{client,config,__init__}.py` | n/a | `tests/backend/engines/test_fish_api.py` |
| Story board, render jobs, preview, past runs | `fish_api/{story,story_jobs}.py`, `api/routers/fish_story.py` | `electron/src/renderer/src/features/fish-story/` | `test_fish_story*.py`, `fish-story-page.test.tsx` |
| Voices and the voice library | `fish_api/{collection,library}.py`, `api/routers/fish_audio.py` | `voice-library.tsx`, `voices-dialog.tsx` | `test_fish_collection.py`, `test_fish_audio_route.py` |
| Training-data index, filters, export | `fish_api/{dataset,dataset_quality}.py`, `api/routers/fish_dataset.py` | `features/fish-dataset/` | `test_fish_dataset.py`, `fish-dataset-page.test.tsx`, `export-dialog.test.tsx` |
| Export training bundle | `fish_api/training_bundle.py` | `export-training.tsx`, `export-tags.tsx`, `export-options.ts` | `test_fish_dataset.py` |
| Local training runs and the evaluation record | `fish_api/{training,training_eval}.py`, `api/routers/fish_training.py` | `train-dialog.tsx`, `use-fish-training.ts` | `test_fish_training.py` |
| LoRA in the OmniVoice trainer | `omnivoice/training/{lora,trainer,checkpoint,config}.py`, `omnivoice/cli/train.py` | n/a | `tests/backend/test_omnivoice_lora.py` |
| Navigation entries | n/a | `components/app-shell/{workspace-menu,command-palette}`, `use-workspace-sidebar.ts`, `routes/index.ts` | route and component tests |
| Strings | n/a | `electron/src/renderer/src/i18n/locales/*.json` (21 files, CRLF) | `tests/test_locale_parity.py`, `bun run locale:check` |

Local data (never committed): drafts, `voice_library.json`, `clip_stats.json` under `<data dir>/fish_stories/`; training runs under `<data dir>/fish_training/`; the owner's renders in their configured output folder. The repo's Colab notebook (`notebooks/OmniVoice_Studio_Colab.ipynb`) runs the whole app on Colab and is unrelated to the exported training bundle.

## Branches and git

- Work continues from `ext/fish-api`. Create a branch from it for anything sizeable; keep commits small and named for the behavior.
- Do **not** work on `main`, and do not open or merge a PR unless asked.
- Commit and push only when the owner says so. A push publishes the work; say what will go out and wait for the word.
- Commits carry only the configured git identity: no `Co-authored-by`, no "Generated with" line, no session links, regardless of any default your tooling suggests.
- Never commit personal material. The owner keeps local-only folders out of git with `.git/info/exclude` (and their own `.gitignore` edit); do not stage `.gitignore` unless asked, and do not add folder names to a public file.
- Locale JSON is stored with CRLF. With `core.autocrlf=true` a plain `git add` rewrites those files and creates a ~79,000-line diff. Stage them with `git hash-object -w --no-filters <file>` then `git update-index --cacheinfo 100644,<hash>,<file>`, and check `git diff --cached --stat` before committing. Read and write files with `newline=""`.

## Your first ten minutes

1. Read `AGENTS.md` (`CLAUDE.md` mirrors it), then `docs/agents/README.md` and `docs/agents/SESSION_STATE.md`. Open the sections of `docs/engines/fish-audio-handoff.md` you need (the "do not change" section first).
2. `git status`, `git log --oneline -10`, and confirm which branch you are on and that it matches what the handoff says.
3. Run the quick checks below once, so you know the baseline before you change anything. The failures listed in the handoff (the torchaudio test and the CJK test) are known; anything else is yours.
4. Look at the real shape of the data before designing for it: `D:\fish_tts\input` holds the owner's renders. Read it; never write to it, rename or delete anything in it.

## Checks

From the repo root:

```
python -m pytest tests/backend/engines tests/backend/api/test_fish_audio_route.py tests/backend/test_omnivoice_lora.py -q -k "fish or lora"
python -m pytest tests/test_locale_parity.py tests/test_changelog_style.py tests/test_changelog_parse.py tests/test_structure_doc.py -q
python scripts/check-docs-drift.py
```

From `electron/` (use these exact commands; `npx tsc` is not the compiler here):

```
bun run typecheck
bun run locale:check
npx vitest run src/renderer/src/features/fish-dataset src/renderer/src/features/fish-story
```

## Definition of done

- A test that fails before and passes after, for the behavior you changed (backend pytest, or a vitest component test). No live Fish call and no real key in any test.
- Every user-facing string through i18n, in all 21 locale files; `locale:check` shows no missing keys and no orphans.
- Docs updated in the same change: `docs/engines/fish-audio.md` for behavior, the handoff for state, and `docs/STRUCTURE.md` when you add a router, a service or an engine adapter (`tests/test_structure_doc.py` pins those counts and fails if they drift).
- A one-line `CHANGELOG.md` entry per user-visible change (about 400 characters at most; a test enforces the style).
- You have said plainly what you could not verify. The UI has only ever been tested with component tests, not driven by hand; GPU training has never run; the Colab bundle has never been run on Colab.

## Decide, or ask the owner

Decide yourself: structure inside the established patterns, names, tests, copy, defaults that are reversible.

Ask first (write the options and your recommendation, then continue with work that does not depend on the answer):

- Adding a dependency (for example `peft`), or changing the lockfile.
- Anything that sends data to a network service, or spends Fish credits (a render, a resume, an A/B test). Say how many calls and wait.
- Changing the audio path, the request shape or modes, the run-folder layout or manifest fields, or the export formats' file names; real data depends on them.
- Training defaults, minimum data sizes, which base model, tag policy for a target model.
- Pushing, merging into `main`, deleting branches, version bumps.

## Delegating to a subagent

- Use [SUBAGENT_PROMPT_TEMPLATE.md](SUBAGENT_PROMPT_TEMPLATE.md) and point the worker at [SUBAGENT_BRIEF.md](SUBAGENT_BRIEF.md) instead of pasting the rules again. Give each agent its own git worktree and a branch name, and a written brief that stands alone: what to read first, the exact job, the limits (no push, no downloads, no network, no real training), the commands to run, and what the report must contain. Agents start with no memory of this session.
- A worktree's `node_modules` may be junctions to the main checkout's. Unlink them (`cmd /c rmdir <path>` on each junction) **before** removing the worktree, then confirm the real `node_modules` still has its contents. Removing a worktree with live junctions can delete the real packages.
- Verify a report before relaying it: check the branch, the commits (author, no attribution lines), the diff size, and re-run the checks yourself. Treat what the agent says it did as a claim.
- Merge in a straight line where you can (fast-forward) and delete merged branches and worktrees afterwards.

## Shell and tooling pitfalls seen here

- Large or quote-heavy shell heredocs fail in this environment; write the script to a file and run that.
- Some agent tools move a long command to the background after a time limit; wait for it to finish instead of polling or rerunning it.
- Do not run two things that rewrite the same files at once (for example a script and an editor), and do not rely on line numbers across edits.
- `npx tsc` is not the TypeScript compiler in this repo; use `bun run typecheck`.

## Safety rules specific to this data

- The renders are the owner's, kept for private listening. They are synthetic Fish output; the export dialog and `dataset.json` record that, and anyone planning to train a model on them should check Fish's terms.
- Dataset code only reads WAVs. Only an explicit export writes, and only into a new folder it creates; it refuses a non-empty one.
- Never put an API key, token or personal path into code, tests, fixtures, logs, commits or docs.
- Refreshing voices from Fish must never overwrite a name the owner chose.

## Vocabulary

- **Run folder:** `<root>/<chapter>/<story>/run-NNN_<time>_<id>/`, one render, with its WAVs and JSON records.
- **Chunk / line:** one story line and its audio file; a clip in the training set.
- **Voice id:** Fish's 32-character id. **Alias:** the owner's own character name for it.
- **Selection:** the matching clips that are ticked; only those are exported.
- **Bundle:** the training configs, script and guide written next to an OmniVoice export.
- **Pass:** one run over every training clip; steps are converted to passes with an estimate that assumes 25 audio frames per second.
