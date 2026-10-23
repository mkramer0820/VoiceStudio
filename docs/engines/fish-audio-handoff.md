# Fish Audio branch handoff

This document records the integration decisions on `ext/fish-api` so future work preserves the intended audio path and Story Studio behavior.

## Audio preservation

- Fish Audio and GPT-SoVITS are remote backends. Their returned audio must not pass through a local model backend or VoiceStudio's normalization/export pipeline.
- A single remote WAV response is preserved byte-for-byte, including when it becomes Story Studio's convenience `story.wav` output.
- Story Studio only decodes and joins audio when Fish's 20,000-character request limit requires multiple responses. That join does not apply loudness normalization. It may resample mismatched responses so a valid combined WAV can be written.
- The `normalize` field in a Fish request is Fish's server-side option. It is separate from VoiceStudio's local normalization pipeline.
- Keep the response's real format and sample rate. In particular, a requested WAV at 44.1 kHz must not be silently transcoded or routed through another engine.

## Story Studio request modes

Multi-speaker calls are a simple on/off switch. Off (the default; also a draft with no mode saved) means one call per line. On groups lines into speaker-tagged calls. The choice is saved with the draft and kept for new stories.

- **Small single-speaker calls (`single`)**: one Fish request per story line. VoiceStudio joins the returned files locally when Join is enabled.
- **Large multi-speaker calls (`multi`)**: Story Studio groups lines into requests below Fish's 20,000-character text limit. It sends speaker-tagged text (`<|speaker:N|>`) and an ordered `reference_id` array. Fish performs the speaker transitions.
- A large-mode group containing only one speaker uses Fish's single-speaker schema: combined plain text plus one string `reference_id`.
- If a large story needs more than one Fish request, VoiceStudio joins only those large response files. Shared response filenames are deduplicated before joining.

Speaker voices resolve in this order:

1. Explicit voice on the line.
2. The draft cast mapping for the line's speaker ID.
3. The story/default narrator voice.

A mixed-speaker request is rejected locally if any speaker lacks a resolved voice. Preview and render compile the same request groups and schemas, so preview is the authoritative dry run.

## Code map

- `backend/engines/fish_api/client.py`: accepts a string or array `reference_id` and sends the Fish request.
- `backend/engines/fish_api/story.py`: draft schema, explicit nullable `request_mode`, request validation, and planning.
- `backend/engines/fish_api/story_jobs.py`: request grouping, speaker mapping, preview, render/resume logging, and output joining.
- `backend/api/routers/fish_story.py`: Story API validation and preview errors.
- `electron/src/renderer/src/features/fish-story/`: required mode controls, disabled preview/render state, and request preview.
- `docs/engines/fish-audio.md`: user-facing setup and behavior reference.

## Regression coverage

The backend tests cover exact WAV preservation, one-call multi-speaker rendering, one-speaker large calls, missing voice rejection, 20,000-character splitting, local joining, an unset mode meaning single, minimal requests, and the multi-speaker model warning. The renderer test covers the on/off switch. Keep all 21 renderer locales in sync when changing labels.

No live Fish call is required for these tests, and no API key belongs in fixtures, logs, commits, or documentation.

## Current state (2026-10-23) and what not to touch

Read this section first if you are picking the work up cold. The two sections above are still correct; this one adds what was built after them.

### Branches

- `main` has none of the Fish work. `ext/fish-api` is the Fish API engine and story board, pushed. `ext/fish-training-data` branches from it and adds the training-data screen and the voice library, pushed. Both are unmerged and no PR exists; there is no release or version bump involved.
- The training branch's own plan and next steps are in [fish-training-handoff.md](fish-training-handoff.md).

### What exists, by area (all local unless stated)

| Area | What it does | Where |
|---|---|---|
| Engine and story board | Fish cloud TTS, story staging, preview, render, join | `backend/engines/fish_api/{client,story,story_jobs}.py`, `backend/api/routers/fish_story.py`, `features/fish-story/` |
| Request preview | Exact schema-checked requests, with each setting's source and whether it was sent | `story_jobs.preview`, `preview-dialog.tsx` |
| Apply settings | Setting edits are staged in the panel and only used by preview and render after **Apply settings** | `settings-panel.tsx` (staged state), page wiring in `fish-story-page.tsx` |
| Past runs | Jobs dialog: list, log, resume, **Load into board** (story, voices and settings), **Listen** per line, open a run folder by path | `story_jobs.{load_run,run_audio_path,open_run_folder}`, `jobs-dialog.tsx` |
| Voice library | Saved voices with your character name, Fish's name, id, details, notes and usage | `backend/engines/fish_api/library.py`, routes in `backend/api/routers/fish_audio.py` (`/api/fish/library`), `voice-library.tsx`, `voices-dialog.tsx` |
| Loudness | `prosody.normalize_loudness` is now always sent explicitly | `client.build_body` |
| Training data | Index of the render folders, speakers by voice id, filters, character zoom, audio measurements, select all/none, export (jsonl, Fish `.lab`, LJSpeech) with tag strip or keep list | `backend/engines/fish_api/{dataset,dataset_quality}.py`, `backend/api/routers/fish_dataset.py`, `features/fish-dataset/` |

Local data written by these features (all under `<data dir>/fish_stories/`): drafts `*.json`, `voice_library.json`, `clip_stats.json`, and renders in the configured output folder. Run folders also keep `job.json`, `manifest.json`, `story.json`, `settings.json`, `requests.jsonl` and `run.log`.

### Do not change without a plan

- **Audio preservation:** a single remote WAV is saved byte-for-byte; no local normalization or transcoding of Fish audio. The dataset code only reads the WAVs, and only an explicit export resamples (to a copy).
- **Request shape:** keep preview and render compiling the same requests, keep `request_mode` meaning "unset = single", and keep minimal-request mode sending only text, format, sample rate and reference id. `prosody.normalize_loudness` is sent on purpose; do not go back to omitting it when true.
- **The run-folder layout and manifest fields** (`<root>/<chapter>/<story>/run-NNN_<time>_<id>/…`). The training-data index and Load/Resume read them, and real folders (about 3,400 clips under `D:\fish_tts\input`) already depend on them. Changes need to stay readable for old runs: 728 of the 3,510 chunks have no `request_mode`, and the dataset index falls back to `story.json` when a chunk has no `text`.
- **Never write to, rename or delete the user's render folders** from the dataset code. It is read-only; exports go to a new folder and refuse a non-empty one. `fish_non_api` is deliberately ignored.
- **Local-first:** only listing the account's voices, searching the public library, importing a collection page, checking the balance, and rendering contact Fish or fish.audio, each from an explicit button. Dataset browsing, measuring and exporting make no network call. Do not add telemetry, keys in fixtures, or background Fish calls.
- **Voice names:** refreshing the account's voices must never overwrite a name the user chose (`library.update` keeps `alias` and `notes`; the board keeps names already on it).
- **Do not "fix" these deliberately-kept details:** speakers are grouped by voice id, not name; older re-renders of the same words are hidden by default rather than deleted; `exclude_keys` leaves clips listed but out of the selection and export.

### Rules from the repo that apply here

- Per `CLAUDE.md`: no agent credit in commits or PRs (no `Co-authored-by` or "Generated with" lines), update docs and `CHANGELOG.md` in the same change, keep changelog entries one line (a test enforces about 400 characters), put UI text through i18n in all 21 locale files, and read bot reviews before merging. Do not bump versions or create release tags.
- Locale JSON is committed with **CRLF** endings. With `core.autocrlf=true` a plain `git add` rewrites them to LF (a ~79k-line diff). Stage them with `git hash-object -w --no-filters <file>` and `git update-index --cacheinfo 100644,<hash>,<file>`.
- Write files with an explicit newline convention (read and write with `newline=""` and keep what the file already uses) so edits do not change line endings.

### Known environment problems (not caused by this work)

- Many backend tests error at import because `torchaudio` cannot load on the author's Windows machine (`libtorchaudio.pyd`). `test_finished_provider_audio_skips_route_dsp` fails for that reason with or without these changes. Run the Fish tests with `python -m pytest tests/backend/engines tests/backend/api/test_fish_audio_route.py -q -k fish`.
- `tests/test_no_hardcoded_cjk.py` fails on `features/fish-story/story-model.ts` (a sentence-splitting regex with CJK punctuation) since an earlier commit. Adding that file to the test's `_ALLOWED_FILES` with a justification is the fix; it was left alone.
- Local OmniVoice generation can time out on long text (default chunk size 800 characters; the budget only extends while chunks finish). That is the local engine, separate from Fish.

### Checks before handing back

`bun run typecheck` and `bun run locale:check` (from `electron/`), `python -m pytest tests/test_locale_parity.py tests/test_changelog_style.py tests/test_changelog_parse.py`, the Fish backend tests above, and `npx vitest run src/renderer/src/features/fish-story src/renderer/src/features/fish-dataset` (use `bun run typecheck`, not `npx tsc`). The folder `_gpt_rewrite_chapter_prompt/` in the repo root is the owner's own untracked material; leave it out of commits.

### Not verified

No live Fish call was made while building the later features, so the voice-library details (author, languages, tags) rely on the fields the existing voice listing already returns. The screens were tested with component tests, not driven by hand in the running app. OmniVoice's exact non-verbal tag list and anything about LoRA on Fish S2 or Qwen3-TTS are unresearched.
