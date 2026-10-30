# Qwen3-TTS studio: UI and story mode (delegation brief)

Written 2026-10-25 for the manager agent, who hands the two prompts below to workers. The backend is built and verified; this brief covers what is left. Rules, limits, commands and the code map are in [SUBAGENT_BRIEF.md](SUBAGENT_BRIEF.md) (do not paste them again). **Read [../engines/qwen3-finetune-findings.md](../engines/qwen3-finetune-findings.md) first** (what is verified, heard or unknown about the fine-tuned narrator and the stock voices; it overrides any assumption you bring from other TTS work), then [../engines/qwen3-narrator.md](../engines/qwen3-narrator.md).

## State of play (branch `ext/qwen3-story`, from `ext/fish-api`; all backend work below is committed)

**Built and tested (162 tests pass in the related files; one real GPU session ran):**

| Piece | Where |
|---|---|
| Sidecar: model and voice per request, one model on the GPU at a time (swaps by itself), custom_voice, Base clone (clip + text, or speaker embedding only), "clone on custom voice" (a clip's x-vector written into the speaker row for one render, then restored), per-chunk end-of-speech report, flush after each chunk | `backend/engines/qwen3_narrator/{main.py,narrator_logic.py,__init__.py}` |
| Model registry: stable ids, finds checkpoints under a root the user sets, the engine env checkpoint, and stock Qwen3-TTS models in **both** the app's Hugging Face cache and the user's default one (the app moves its cache on Windows) | `backend/engines/qwen3_studio/registry.py` |
| Speaker rules: a blank speaker is accepted only on a model with exactly one speaker; the nine stock voice names (`ryan`, `vivian`, ...) can be **borrowed** on any model, with a warning (section 4b of the findings doc) | `playground.py` (`pick_speaker`, `borrowed_warning`) |
| Playground runner: many takes, each with its own model, voice and settings; warm-up per voice; "group by model" ordering; cancel; per-take WAV plus `session.json` manifest; notes and stars; one run at a time | `backend/engines/qwen3_studio/playground.py` |
| HTTP API | `backend/api/routers/qwen_studio.py`, mounted in `backend/main.py` |
| Tests (fake engine, no GPU) | `tests/test_qwen_studio.py`, `tests/test_qwen3_narrator.py` |

**Built since this brief was written (2026-10-25/27, unverified in a real window):** the playground page, its nav entry, the changelog line and the user docs page (`docs/engines/qwen3-playground.md`), and Settings -> Storage -> Model locations. Translations are English placeholders in the 20 non-English locales. **Story mode built 2026-10-28 (unverified, never run):** `/qwen-story`, see `docs/engines/qwen3-story.md` (profiles replace the voice catalogs; Prompt 2 below is the original plan). Prompt 1 is done.

**Decided by listening (owner, 2026-10-25), do not reopen:** the narrator is EP09 as **custom_voice**, speaker `ss_narrator`, **no instruct** (instruct moves the voice away from the narrator; offer it as an experimental control only). Clones use **stock Base**, reference clip plus its text by default; "speaker embedding only" is an option with free tempo. The "hybrid" (EP09 weights inside Base) was **dropped**. **Not yet judged by ear:** "clone on custom voice" (the new kind). Label it experimental in the UI.

## Plain names the UI must use (owner: "your descriptions are hard to tell what is doing what")

Label every take with one of three recipes **and** a one-line summary of what it really runs, in this form: `Model: <model label> (<type>) · Voice: <speaker or reference> · Mode: <what>`.

| Recipe name | Model | What picks the voice | Notes |
|---|---|---|---|
| **Narrator** | EP09, type custom_voice | speaker `ss_narrator` | the intended narrator output: no instruct, no reference clip |
| **Clone** | stock Base | a reference clip and its text | the best clone mode so far; any other character |
| **Experiment** | EP09 or Base | instruct on the narrator; "speaker embedding only" clone; "clone on custom voice" | label as experimental, never a default |

Never describe a take with the library's internal terms alone (ICL, x-vector, graft). If a term is needed, put it in a tooltip next to the plain name.

## API contract (`/api/qwen`, admin-guarded; GETs are cross-site guarded like Fish)

- `GET /state` also returns `stock_speakers` (the nine names) and each model has `borrowable_speakers` (stock names it has no trained voice for). `GET /state` returns `settings` (`python`, `checkpoint_root`, `output_dir`, `warmup_runs`, `warmup_text`), `output_dir_default`, `engine` (`available`, `reason`), `models[]` (`id`, `label`, `path`, `source`, `type` = `custom_voice`|`base`|`voice_design`, `size`, `speakers[]`, `bytes`) and `profiles[]` (the app's clone-capable voice profiles: `id`, `name`, `ref_text`, `language`, `kind`).
- `PUT /settings` takes any of those settings (partial). Changing `python` restarts the sidecar on the next run.
- `POST /playground/run` takes `{session_name, takes[], group_by_model, warmup_runs?, warmup_text?, output_dir?}`. Each take: `{id, label, text, model_id, voice, params, repeat}`. `voice`: `{kind: custom_voice|clone|clone_on_custom, speaker?, instruct?, profile_id? | ref_audio? + ref_text?, xvec_only?}`. `params`: `{language, seed, temperature, top_k, top_p, repetition_penalty, pad_s (default 0.3), chunk_chars, max_seq_len}`, all optional but `language`/`pad_s`. Returns the job view; **422** with a plain-language reason (unknown model, wrong model type for the voice kind, missing clip or reference text), **409** if a run is in progress.
- `GET /playground/jobs/{id}` is the job view: `status` (`running|done|cancelled|error`), `total`, `done`, `model_swaps`, `folder`, `session`, `takes[]` (each with `results[]`: `rep`, `status`, `file`, `seconds`, `gen_seconds`, `realtime`, `seed`, `complete`, `chunks[]`, `vram_allocated_mb`, `warmed_up`, `error`, `voice` info), `log[]`. Poll it; there is no websocket.
- `POST /playground/jobs/{id}/cancel`, `POST /playground/unload` (stop the sidecar, free the GPU), `GET /playground/sessions`, `GET /playground/sessions/{session}` (the manifest), `PATCH /playground/sessions/{session}/takes/{take_id}` with `{notes?, starred?}`, `GET /playground/sessions/{session}/audio/{name}` (WAV).
- Adding a clone voice reuses the **existing** `POST /profiles` (multipart: `kind=clone`, `ref_audio` file, optional `ref_text`, which is auto-transcribed when blank). Do not create a second voice store.

Valid pairs: `custom_voice` and `clone_on_custom` need a `custom_voice` model; `clone` needs a `base` model; a `custom_voice` take on a Base model is accepted only with a borrowed stock voice name.

**Speakers.** `models[].speakers` are the model's own trained speakers (EP09: `ss_narrator`; stock 0.6B CustomVoice: nine). For a custom-voice take the UI must show a speaker picker: the model's own speakers first, then its `borrowable_speakers` under a clearly marked "borrowed stock voice (will sound different)" group. Pre-fill the single speaker of a one-speaker model (EP09); never leave a multi-speaker model blank (the backend answers 422 "Pick a speaker ..."). A borrowed take's result carries `warnings[]` and `voice.borrowed: true`: show the warning on the take, plainly, next to the player. The page should still **let the user pick any combination** and show the backend's reason (the owner wants to try things); disable nothing silently.

## Prompt 1: the playground page

```
You are working on the VoiceStudio repository (E:\VoiceStudio). Work in the main checkout (E:\VoiceStudio) directly on the existing branch ext/qwen3-story: do not create a branch or a worktree (the owner's instruction). The manager will not edit files while you work. Commit locally in small commits; do not push.

READ FIRST, in this order, and nothing else until you need it:
1. docs/agents/SUBAGENT_BRIEF.md (rules, limits, commands, code map).
2. docs/agents/QWEN3_STUDIO_UI_BRIEF.md (this brief: the API contract, what is decided, what is not built).
2b. docs/engines/qwen3-finetune-findings.md (what is verified about the models and voices; sections 1, 4b, 5, 6 matter most).
3. electron/src/renderer/src/features/fish-story/fish-story-page.tsx and use-fish-story.ts (search for the page layout and the hook's fetch/poll pattern; do not read whole files), jobs-dialog.tsx (the past-runs list).
4. How a page is wired: search for 'fish-story' in electron/src/renderer/src/routes/index.ts, components/app-shell/workspace-menu.tsx, use-workspace-sidebar.ts, components/command-palette.tsx.

GOAL: a "Qwen playground" page next to the Fish Story board where the owner can try every Qwen3-TTS variant fast: narrator on the fine-tuned custom_voice, clones on Base (clip + text, or embedding only), clone on custom voice (experimental), narrator-as-reference on custom voice, with or without instruct. Several prompts in one run, like the story board, so a model that does not change does not reload. Lots of knobs (the owner: "the more configs the better"), a fun toy, but every run is manifested to a folder the owner chooses and every take can carry notes.

CHANGE (new folder electron/src/renderer/src/features/qwen-playground/; do not edit any fish-story or fish-dataset file):
1. Route /qwen-playground, a nav entry in the same group as Fish Story (icon of your choice from the existing icon set), a command-palette entry.
2. Setup bar: output folder (text + browse, default from GET /state, shows the resolved path), session name, python path and checkpoint-root settings (PUT /settings) with the engine's `available`/`reason` shown plainly, warm-up runs and warm-up text, a "group by model" switch showing how many model loads the current order needs, Run all / Run selected / Cancel / Unload GPU.
3. Takes list (the main area), add/duplicate/delete/reorder/collapse. Each take card: label, prompt text, model picker (from models[], showing type and size), voice-kind switch (custom voice / clone on Base / clone on custom voice), then the fields that kind needs: speaker (from the model's speakers[]), instruct, clone source (a voice profile picker from profiles[] with its ref text editable, or a file path + ref text, plus "speaker embedding only"), and a collapsible Advanced group with every param (language, seed with a dice button and "new seed per repeat", temperature, top_k, top_p, repetition penalty, tail padding pad_s, chunk_chars, max_seq_len), and repeat count (1-8). Show a pair warning (not a block) when the voice kind and model type do not match. "Add clone voice" opens a small dialog that posts to the existing /profiles route.
4. Results under each take as the job runs: progress, a player per render, seconds, real-time factor, "complete" flag (a false one means the end-of-speech token was never produced: show a warning), error text, A/B: select two renders and play them back to back. Notes box and star per take, saved with the PATCH route.
5. Past sessions list (GET /playground/sessions) that opens a manifest read-only with the same players and notes.
6. A pure-logic module (take defaults, validation messages, model-load counting that mirrors the backend's grouping, the request builder) with unit tests; a component test for the page's main flows with the fetch layer mocked.
7. i18n: every string through t(), a new top-level `qwenPlayground` namespace in en.json and, as English placeholders, in all 20 other locale files (list the placeholders in your report so they can be translated). Mind the CRLF rule in SUBAGENT_BRIEF.md.
8. Docs and changelog: a user page docs/engines/qwen3-playground.md (what each voice kind is, in the owner's words: custom voice is a baked-in speaker, Base is the clone model, clone on custom voice is experimental; where files and manifests go), one CHANGELOG.md line under [Unreleased], link the page from docs/agents/README.md only if needed.

CONTRACTS TO KEEP: the API above (if it is missing something you need, stop and say so; do not change backend files except to fix a bug you can prove with a failing test); Fish pages and their tests unchanged.
TESTS: pure-logic tests (grouping count, request builder, validation), component test (add take, run, results appear, notes saved), `bun run typecheck`, `bun run locale:check`, vitest for the new folder plus fish-story and fish-dataset (must stay green), tests/test_locale_parity.py.
LIMITS: as in SUBAGENT_BRIEF.md, plus: **do not start the real sidecar, load a model or use the GPU**; the manager does the real runs. Do not push. Commit with the configured git identity only.
OWNER QUESTIONS: if a control is ambiguous, pick the more configurable option and note it in your report; do not stop for it.

WHEN DONE: update your row in the project board of docs/agents/SESSION_STATE.md in the same commit.

REPORT (under 500 words): files changed and commit hashes; tests run and results; the English placeholder keys needing translation; what you could not verify (anything not seen rendered); decisions needed.
```

## Prompt 2: story mode (start after prompt 1 merges; both touch the locale files)

Backend first, then UI. The owner's decisions, in their words where it matters:

- A **separate Qwen story page in the same place as the Fish one** (`features/qwen-story/`). It may import the pure logic of `fish-story` (script splitting, `[Name]` speaker detection in `tag-tools.ts`, the chunk list) **read-only**; it must not edit any Fish file. Do not copy Fish-specific assumptions (one remote voice id per speaker, request modes, the 20,000-character grouping, byte-preserved remote audio).
- Drop in the current Story Studio script format, detect characters, **assign a voice to each character, plus a narrator**. The UI changes for character mapping as needed.
- **Voices come from a per-character catalog.** A character can have several voices (clones with a reference clip + text, or a custom_voice speaker), optionally tagged (`[Angry]`, `[Whisper]`): a line whose tag matches a voice's tag uses that voice, otherwise the character's default. The tag is stripped from the spoken text; tags do nothing else for now.
- Different characters may use different models and checkpoints. Nothing hard-coded to EP09. The user maps any voice to any model (warn, do not block).
- **Render one character at a time with a warm model**, with warm-up runs (a throw-away sentence rendered several times) before a character's first real line, because short lines on raspy voices are poor when cold. Track every line's index, character, voice, model, and where its file lands.
- **Organisation and manifests are critical.** Stable IDs for every model, checkpoint, voice, reference asset and assignment; JSON manifests, never inferred from file names. Persist the story's character-to-voice assignments so a project reopens and regenerates the same way. Clone voices stay in the app's existing `voice_profiles` table (reuse `POST /profiles`); models come from `registry.py`; the new story JSON (`<data dir>/qwen_studio/stories/<id>.json`: chunks, cast, voice catalogs, settings) is the only new store.
- **Output like the Fish board, with production-grade backend work:** a folder the user sets, `<root>/<chapter>/<story>/run-NNN_<time>_<id>/` with `job.json`, `manifest.json`, `story.json`, `settings.json`, `run.log` and the WAVs, named like `story_jobs._audio_name` (index, speaker, voice). Each line is its own WAV; a joined file is optional with `pause_ms` between lines; tail padding `pad_s` is a UI setting. Resume and "load a past run" can follow Fish's `story_jobs` (`load_run`, `resume_render`) but are lower priority than a correct render.
- Reuse `playground.py`'s runner pieces (`resolve_voice`, `_check_pair`, `_engine`, `_save_wav`, the job and manifest helpers) instead of duplicating them; move shared helpers into a common module if that is cleaner, with the playground tests staying green.
- Preview before render (the lines grouped by character and model, the number of model loads, warnings) like the Fish preview dialog, but no network.

Ask the owner before: adding a table or migration, changing the playground API shape, or adding a dependency.

## Things already learned (do not relearn)

- On Windows the app sets `HF_HOME`/`HF_HUB_CACHE` to `%LOCALAPPDATA%\OmniVoice\hf_cache`; stock models downloaded by other tools sit in `~/.cache/huggingface/hub`. The registry reads both.
- Tests that call `playground.start_run` set a module-level python override in the engine; reset it (`set_python_override(None)`), or the engine's own tests see it.
- A manifest must not contain the job's cancel `threading.Event` (it once broke `json.dump`).
- Engine tests never need torchaudio; keep it that way. The known torchaudio and CJK failures are listed in SUBAGENT_BRIEF.md.
- `clone_on_custom` extracts the x-vector once per reference on the CPU with the stock Base model (about 3.7 GB RAM briefly) and caches it as `.npy` under `<data dir>/qwen_studio/xvectors/`.
- Instruct on the narrator changes the voice a lot; it is a creative control, not a default. EP09 itself cannot clone (no speaker encoder); clone-on-custom works around that.

## For the manager: before relaying a worker's branch

Check the author is the owner's identity, there are no attribution lines, the file list stays inside the folder named in the prompt (no Fish files, and the locale diffs are small), `git diff --cached --stat` is sane for locale files, and that `bun run typecheck`, `bun run locale:check` and the listed tests pass. Do a real GPU session yourself (the playground script pattern is in the session notes: scratch data dir, `OMNIVOICE_DATA_DIR`, the project's `.venv-qwen` python) before telling the owner it works.
