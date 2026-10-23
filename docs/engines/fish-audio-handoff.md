# Fish Audio: handoff (read this first)

The single state-of-the-work document for everything Fish in VoiceStudio: the API engine, the story board, the voice library, the training-data screen, and where the LoRA/training phase should start. User-facing behavior lives in [fish-audio.md](fish-audio.md); this file is for whoever works on the code next. Last updated 2026-10-23.

## 1. Branches and status

| Branch | Contents | Status |
|---|---|---|
| `main` | none of the Fish work | untouched |
| `ext/fish-api` | Fish cloud engine, story board, jobs, preview, voices, apply settings, load past runs | pushed, no PR |
| `ext/fish-training-data` | branches from `ext/fish-api`; voice library, training-data screen, audio measurements, exports, these docs | pushed, no PR |

No release, version bump or tag is involved. Per `CLAUDE.md`, versions only change when the owner asks.

## 2. What exists

All local unless the Contact column says otherwise.

| Area | What it does | Code | Contacts Fish |
|---|---|---|---|
| Engine | Fish cloud TTS, lossless WAV defaults, masters archive | `backend/engines/fish_api/{__init__,client,config}.py` | on synthesis, voice list, balance |
| Story board | Stage a multi-voice story, tags and cast, review, render | `fish_api/{story,story_jobs}.py`, `backend/api/routers/fish_story.py`, `features/fish-story/` | on render only |
| Request preview | Exact schema-checked requests; each setting's source and whether it is sent | `story_jobs.preview`, `preview-dialog.tsx` | no |
| Apply settings | Panel edits are staged; preview and render use only applied settings | `settings-panel.tsx` (staged state), `fish-story-page.tsx` | no |
| Past runs | Jobs dialog: list, log, resume, **Load into board**, **Listen** per line, open a run folder by path | `story_jobs.{load_run,run_audio_path,open_run_folder}`, `jobs-dialog.tsx` | resume only |
| Voice library | Saved voices with your character name, Fish name, id, details, notes, usage | `fish_api/library.py`, `/api/fish/library` in `routers/fish_audio.py`, `voice-library.tsx`, `voices-dialog.tsx` | no |
| Loudness | `prosody.normalize_loudness` always sent explicitly | `client.build_body` | n/a |
| Training data | Index of render folders, speakers by voice id, filters, character zoom, audio measurements, select all/none, export | `fish_api/{dataset,dataset_quality}.py`, `routers/fish_dataset.py`, `features/fish-dataset/` | never |

Local data under `<data dir>/fish_stories/`: drafts `*.json`, `voice_library.json`, `clip_stats.json`, and renders in the configured output folder. Each run folder holds `job.json`, `manifest.json`, `story.json`, `settings.json`, `requests.jsonl`, `run.log` and the WAVs.

## 3. Do not change without a plan

### Audio path

- Fish Audio and GPT-SoVITS are remote backends. Their returned audio must not pass through a local model backend or VoiceStudio's normalization/export pipeline.
- A single remote WAV response is preserved byte-for-byte, including when it becomes the convenience `story.wav`.
- Story Studio only decodes and joins audio when Fish's 20,000-character request limit needs multiple responses, or when Join stitches single-speaker lines. The join applies no loudness normalization; it may resample mismatched responses so a valid WAV can be written.
- The `normalize` field in a request is Fish's server-side option, separate from VoiceStudio's local normalization.
- Keep the response's real format and sample rate: 44.1 kHz WAV must never be silently transcoded or routed through another engine.
- The dataset code only reads WAVs. Only an explicit export writes anything, and only to a new copy (optionally resampled).

### Request shape and modes

- **Multi-speaker is an on/off switch.** Off (default, and a draft with no saved mode) means one request per line (`single`). On groups lines into speaker-tagged calls (`multi`) below the 20,000-character limit, sending `<|speaker:N|>` tags and an ordered `reference_id` array. A multi group with one speaker uses the single-speaker schema. If a large story needs more than one request, only those response files are joined, with shared filenames deduplicated.
- **Voice resolution order:** explicit voice on the line, then the draft cast mapping for the line's speaker, then the story/default narrator. A mixed-speaker request is rejected locally if any speaker has no voice.
- **Preview and render compile the same request groups and schemas**, so preview is the authoritative dry run. Keep it that way.
- **Minimal request** sends only text, format, sample rate and reference id (plus speed if it is not 1.0).
- **`prosody.normalize_loudness` is sent on purpose.** Runs from chapter 3200 on were made without the field and still measure about -26 dB, so Fish treats a missing field like `true`; sending it explicitly changes nothing for them but makes it visible in preview. Do not go back to omitting it when true.

### Run folders and manifests

The layout `<root>/<chapter>/<story>/run-NNN_<time>_<id>/` and its JSON files are read by Resume, Load into board and the training-data index, and about 3,400 real clips under `D:\fish_tts\input` depend on them. Keep changes readable for old runs: 728 of 3,510 chunks have no `request_mode`, and the index falls back to `story.json` when a chunk has no `text`.

### Local-first and data safety

- Only these contact Fish or fish.audio, each from an explicit button: listing the account's voices, searching the public library, importing a collection page, checking the balance, rendering and resuming. Dataset browsing, measuring and exporting make no network call. No telemetry, no keys in fixtures, no background Fish calls.
- The dataset code never writes to, renames or deletes the user's render folders. Exports refuse a non-empty folder. `fish_non_api` is deliberately ignored.
- Refreshing the account's voices must never overwrite a name the user chose (`library.update` keeps `alias` and `notes`; the board keeps names already on it).

### Deliberate details that look like bugs

Speakers are grouped by voice id, not name (two names can share one voice). Older re-renders of the same words are hidden by default, not deleted. `exclude_keys` leaves clips listed but out of the selection and the export; the list shows every match.

## 4. Code map

- `backend/engines/fish_api/client.py`: request body (`reference_id` string or array), sending, retries.
- `fish_api/story.py`: draft schema, explicit nullable `request_mode`, request validation, planning.
- `fish_api/story_jobs.py`: request grouping, speaker mapping, preview, render/resume, logging, joining, past-run loading and audio, voice-use recording.
- `fish_api/library.py`: the voice library store.
- `fish_api/dataset.py`: corpus index, filters, facets, export. `fish_api/dataset_quality.py`: per-clip audio measurements, cached by sha256.
- `backend/api/routers/{fish_audio,fish_story,fish_dataset}.py`: the HTTP surface.
- `electron/src/renderer/src/features/{fish-story,fish-dataset}/`: UI, hooks and tests. Navigation entries are in `components/app-shell/workspace-menu.tsx`, `use-workspace-sidebar.ts`, `command-palette.tsx` and `routes/index.ts`.
- Tests: `tests/backend/engines/test_fish_{api,story,story_jobs,collection,dataset}.py`, `tests/backend/api/test_fish_audio_route.py`, and the two feature folders' `*.test.ts(x)`. No live Fish call is needed and no key belongs in fixtures, logs, commits or docs.

## 5. Rules from the repo that apply

- No agent credit in commits or PRs (`CLAUDE.md`). Update docs and `CHANGELOG.md` in the same change; changelog entries are one line (a test enforces about 400 characters).
- All UI text goes through i18n in all **21** locale files; `bun run locale:check` and `tests/test_locale_parity.py` enforce it.
- Locale JSON is committed with **CRLF** endings. With `core.autocrlf=true` a plain `git add` rewrites them to LF (a ~79k-line diff). Stage them with `git hash-object -w --no-filters <file>` then `git update-index --cacheinfo 100644,<hash>,<file>`. Read and write files with `newline=""` and keep each file's existing endings.
- Default features must behave the same on macOS, Windows and Linux. GPU training will vary by host, so keep it opt-in and skip it, with a clear message, where it cannot work.

## 6. Known environment problems (not caused by this work)

- Many backend tests error at import because `torchaudio` cannot load on the author's Windows machine (`libtorchaudio.pyd`). `test_finished_provider_audio_skips_route_dsp` fails for that reason with or without these changes.
- `tests/test_no_hardcoded_cjk.py` fails on `features/fish-story/story-model.ts` (a sentence-splitting regex with CJK punctuation) since an earlier commit. The fix is to add that file to the test's `_ALLOWED_FILES` with a justification; it was left alone.
- Local OmniVoice generation can time out on long text (default chunk size 800 characters; the budget only extends while chunks finish). That is the local engine, not Fish.
- The folder `_gpt_rewrite_chapter_prompt/` in the repo root is the owner's own untracked material. Never commit it.

## 7. Before handing back

`bun run typecheck` and `bun run locale:check` (from `electron/`; use these, not `npx tsc`), `python -m pytest tests/test_locale_parity.py tests/test_changelog_style.py tests/test_changelog_parse.py`, `python -m pytest tests/backend/engines tests/backend/api/test_fish_audio_route.py -q -k fish`, and `npx vitest run src/renderer/src/features/fish-story src/renderer/src/features/fish-dataset`.

## 8. Facts about the data (2026-10-23, `D:\fish_tts\input`)

- 32 runs, 31 chapters, about 3,370 usable clips, all `s2.1-pro-free`, 44.1 kHz mono WAV.
- Only **SS_Narrator** has hours of audio (about 4 h). Noctis (about 34 min) and sun_2 (about 26 min) are borderline; every other speaker is under about 8 min.
- `[slow]` is on about 99% of narrator clips, so a model trained on the narrator will always speak slowly unless that changes in future renders.
- Fish's non-verbal tags are `[sighing]` (33), `[laughing]` (17), `[chuckling]` (10); OmniVoice uses its own spellings. Use the export's `name=other` tag mapping (for example `laughing=laughter`).
- **Loudness differs by chapter.** Chapters 3193–3199 were rendered with `normalize_loudness: false`, `volume: 0`, `top_p: 1.0`: median about -14 dB RMS with peaks near 0.95 (hot, not clipped). From 3200 on, loudness normalization and `volume -4`: about -26 dB RMS, peaks near 0.27. Do not mix them in one training set without choosing a level.
- 140 clips are listed in manifests but their WAVs were deleted (mostly chapter 3218); 303 clips are older copies of re-rendered lines.
- Choppiness: pauses of 300 ms or more inside a clip are similar across 3203 and 3206–3208 (settings are identical), so no settings explanation was found for the "choppy" 3206–3208 audio. A/B renders (loudness field omitted, true, false, basics only) were proposed but not run, to avoid spending credits.

## 9. Where to start for the LoRA phase

### The trainer that already exists (OmniVoice)

- Code: `omnivoice/training/` (`trainer.py`, `builder.py`, `config.py`, `checkpoint.py`) and `omnivoice/cli/train.py`. Docs: `docs/training.md` (`accelerate launch -m omnivoice.cli.train --train_config … --data_config … --output_dir …`) and `docs/data_preparation.md`. Example configs: `examples/config/`.
- It does **full fine-tuning** from a checkpoint (`init_from_checkpoint`, `resume_from_checkpoint`; backbone `Qwen/Qwen3-0.6B` via `llm_name_or_path`). A repo-wide search finds **no LoRA or `peft` code**, so LoRA means adding an adapter path to the trainer, and checking `peft` against `pyproject.toml` / `uv tree` before adding a dependency.
- Input it expects: JSONL with `{"id", "audio_path", "text", "language_id"}` per line, audio resampled to **24 kHz**. `omnivoice.scripts.extract_audio_tokens` (needs `eustlb/higgs-audio-v2-tokenizer`) turns that into WebDataset tar shards plus a `data.lst`, which the data config points at.
- **Quick win to build first:** an `omnivoice` export format in `dataset.py` that writes exactly that JSONL (24 kHz mono WAVs, tags stripped except the OmniVoice tags you keep, `language_id: "en"`), so the export dialog feeds the existing tokenizer with no manual conversion.

### Research findings (read from the code, 2026-10-23; nothing was downloaded or run)

**Model structure** (`omnivoice/models/omnivoice.py`, `training/builder.py`):

- `OmniVoice` = `self.llm` (a bare `AutoModel` backbone, Qwen3-0.6B: hidden 1024, 28 layers, no LM head) + `self.audio_embeddings` (`nn.Embedding(8 * 1025, hidden)`, new, no pretrained init) + `self.audio_heads` (`nn.Linear(hidden, 8 * 1025, bias=False)`, new). `forward()` builds `inputs_embeds` itself (text ids through the LLM's own embedding, audio ids through `audio_embeddings`, summed over the 8 codebooks), calls `self.llm(inputs_embeds=...)`, then `audio_heads`. The loss is a per-codebook weighted cross-entropy (`audio_codebook_weights`).
- Attention is `flex_attention` with a packed block mask built with `create_block_mask(..., _compile=True)`. That needs a working compile toolchain (Triton on CUDA). Plain Windows PyTorch has no Triton, so on Windows the trainer realistically needs `triton-windows` or WSL; this is a host capability limit, not a behavior difference, and the training step must say so up front instead of failing deep inside the subprocess.
- **Where a LoRA adapter attaches:** wrap `model.llm` only (`q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` in each of the 28 blocks). `OmniVoice.forward` only ever calls `self.llm(...)` and `self.llm.get_input_embeddings()`, so a wrapped `llm` leaves the rest untouched. `audio_embeddings` / `audio_heads` are the "new" task-specific layers: the choice to keep them frozen or train them fully (`modules_to_save`) is a quality question that needs a real run.
- **Checkpoints:** `checkpoint.save_checkpoint` calls `accelerator.save_state` (full weights + optimizer) and `unwrap_model.save_pretrained`. With an adapter that is wasteful (the whole 0.6B again every `save_steps`), so the adapter path writes `adapter/` (the LoRA tensors plus any trained non-LoRA modules plus a small `adapter_config.json`) and `resume` loads the base checkpoint plus the adapter.
- `create_optimizer_and_scheduler` hands `self.model.parameters()` to AdamW; with frozen base weights only the trainable ones should be passed.

**Config fields** (`training/config.py`, `TrainingConfig`; `from_json` silently drops unknown keys, so new `lora_*` fields are backward compatible): paths (`output_dir`, `data_config` come from the CLI), model (`llm_name_or_path`, `audio_vocab_size` 1025, `audio_mask_id` 1024, `num_audio_codebook` 8, `audio_codebook_weights`), sample processing (`drop_cond_ratio`, `prompt_ratio_range`, `mask_ratio_range`, `language_ratio`, `use_pinyin_ratio`, `instruct_ratio`, `only_instruct_ratio`), init (`init_from_checkpoint`, `resume_from_checkpoint`), optimisation (`learning_rate`, `weight_decay`, `max_grad_norm`, `steps`, `seed`, `lr_scheduler_type`, `warmup_*`), data (`batch_tokens`, `gradient_accumulation_steps`, `num_workers`), system (`mixed_precision`, `allow_tf32`, DeepSpeed), logging (`logging_steps`, `eval_steps`, `save_steps`, `keep_last_n_checkpoints`). `examples/config/train_config_finetune.json` is the fine-tuning template: `init_from_checkpoint: k2-fsa/OmniVoice`, 5000 steps, lr 5e-5, `batch_tokens` 8192, instruct ratios 0.

**Memory (estimates from parameter counts, not measured):** about 0.62 B parameters. The builder loads fp32 (about 2.5 GB), and full fine-tuning adds fp32 gradients and two AdamW moments, so about 10 GB before activations at `batch_tokens` 8192: more than the 8 GB RTX 3070 on the owner's machine. LoRA (r=16 on all linear layers, roughly 10 M trainable parameters) keeps the optimizer state under 0.2 GB, so the cost is dominated by the fp32 base (about 2.5 GB) plus activations; with `batch_tokens` 2048 to 4096 it should fit in 8 GB, which is why the LoRA defaults are small. Loading the base in bf16 would halve that again, but the builder hard-codes fp32 and was left alone. CPU and MPS training are not realistic here (flex attention, bf16 autocast); the module reports them as unavailable rather than guessing.

**Dependencies:** `accelerate` is a direct dependency (`pyproject.toml`, locked at 1.14.0); `transformers>=5.10.0`, `torch>=2.4` too. **`peft` is neither in `pyproject.toml` nor in `uv.lock`** and is not installed here. Adding it is an owner decision (see below); this phase therefore uses a small in-repo LoRA (`omnivoice/training/lora.py`, plain torch) so nothing is added to the lockfile.

**Tokenizer step:** `python -m omnivoice.scripts.extract_audio_tokens --input_jsonl X.jsonl --tar_output_pattern <dir>/audios/shard-%06d.tar --jsonl_output_pattern <dir>/txts/shard-%06d.jsonl --tokenizer_path eustlb/higgs-audio-v2-tokenizer [--nj_per_gpu 3] [--shuffle True] [--min_length S] [--max_length S] [--skip_errors] [--samples_per_shard N] [--min_num_shards N]`, one run per split. It writes `<dir>/data.lst` (lines of `tar jsonl num_items seconds`). The data config then lists `data.lst` under `train` / `dev`. `examples/run_finetune.sh` is the reference pipeline. The tokenizer reads at 24 kHz (`HIGGS_INPUT_SAMPLE_RATE`); `--min_num_shards` defaults to 32, which must be lowered for a small set so shards are not empty. The tokenizer repo is a model download, so this step is skipped in dry runs.

**Non-verbal tags OmniVoice knows** (`_NONVERBAL_PATTERN`): `[laughter] [sigh] [confirmation-en] [question-en] [question-ah] [question-oh] [question-ei] [question-yi] [surprise-ah] [surprise-oh] [surprise-wa] [surprise-yo] [dissatisfaction-hnn]`. So for the narrator, `laughing=laughter` and `sighing=sigh` are the only Fish tags with a direct equivalent; `chuckling` has none and is best dropped.

### Other targets

- **Fish S2 local:** not in the repo (only the cloud engine exists). The `fish_lab` export (WAV + `.lab`) matches Fish's own fine-tuning layout, but a local engine has to come first. The S2 Pro config shared during design shows a 36-layer text model plus a 4-layer audio decoder with 10 codebooks; which modules to target with LoRA was not investigated.
- **Qwen3-TTS:** nothing in the repo; its tag format, sample rate and trainer are unresearched.

### Infrastructure to reuse

- **Long jobs with progress, logs and resume:** `story_jobs.py` (background thread, `job.json` + `run.log`, interrupted-run detection, resume) and the smaller `dataset_quality.py` job.
- **Run training as a killable subprocess:** `backend/services/subprocess_backend.py`, `backend/core/contained_subprocess.py`, and [omnivoice-subprocess.md](omnivoice-subprocess.md) for why a wedged GPU job must be killable. Training will need its own venv for torch, accelerate and peft.
- **GPU and VRAM gating:** `backend/services/model_manager.py` (GPU pool, execution budgets, "abandoned worker") and `backend/services/gpu_sandbox.py`. Unload the resident TTS model before a run.
- **API and UI pattern:** `routers/fish_dataset.py` and `features/fish-dataset/`. A natural home is a *Train* step reached from the export dialog, with a runs list like the Jobs dialog.

### Constraints

- `docs/ROADMAP.md` (the on-device learning rows) says LoRA **ships only with an automatic evaluation gate**. Plan a before/after listening test and a metric (for example speaker similarity or an ASR word-error check on held-out `val.jsonl` clips) in the first version.
- `val.jsonl` uses a stable hash split, so re-exporting never leaks a validation clip into training.
- Local-first: no network for training.

### Data decisions needed before training

- One loudness level (filter chapters 3193–3199 or add normalization on export).
- Whether `[slow]` stays in the narrator's text or those clips are re-rendered without it.
- Tag policy per target: Fish keeps its tags; OmniVoice only its non-verbal tags (verify its exact list first).

## 10. Not built yet

A tag-to-`instruct` mapping for OmniVoice style attributes (whisper, pitch, gender); loudness normalization on export; an `omnivoice` export format; the training step itself.

## 11. Not verified

No live Fish call was made while building the later features, so voice-library details rely on the fields the existing voice listing already returns. The screens were tested with component tests, not driven by hand in the running app. OmniVoice's exact non-verbal tag list, and LoRA on Fish S2 or Qwen3-TTS, are unresearched.
