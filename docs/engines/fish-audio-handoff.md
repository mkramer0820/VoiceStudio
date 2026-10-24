# Fish Audio: handoff (read this first)

The single state-of-the-work document for everything Fish in VoiceStudio: the API engine, the story board, the voice library, the training-data screen, and where the LoRA/training phase should start. User-facing behavior lives in [fish-audio.md](fish-audio.md); this file is for whoever works on the code next. How to work on it (layout, git, checks, delegating) is in [../agents/fish-project-organization.md](../agents/fish-project-organization.md).

## 1. Branches and status

| Branch | Contents | Status |
|---|---|---|
| `main` | none of the Fish work | untouched |
| `ext/fish-api` | the line that holds all of it: engine, story board, jobs, preview, voices and library, past runs, training-data screen, exports and training bundle, in-repo LoRA and the Train dialog | continue from here; no PR |
| `ext/fish-training-data` | an earlier name for the same line, kept on the remote at the same commit | do not develop on it |

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
| Training data | Index of render folders, speakers by voice id, filters, character zoom, audio measurements, select all/none, export with a tag dictionary | `fish_api/{dataset,dataset_quality}.py`, `routers/fish_dataset.py`, `features/fish-dataset/` | never |
| Training bundle | OmniVoice export also writes portable rows, `train_config.json`, `data_config.json`, `train.sh` and `COLAB.md` for training elsewhere (Colab), with every trainer setting configurable and a passes estimate | `fish_api/training_bundle.py`, `export-tags.tsx`, `export-training.tsx` | never |
| Training (LoRA) | OmniVoice export, loudness matching, opt-in Train dialog, dry run, cancellable and resumable runs, evaluation record | `fish_api/{training,training_eval}.py`, `omnivoice/training/lora.py`, `routers/fish_training.py`, `features/fish-dataset/train-dialog.tsx` | never |

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
- `fish_api/dataset.py`: corpus index, filters, facets, tag dictionary, export. `fish_api/dataset_quality.py`: per-clip audio measurements, cached by sha256.
- `fish_api/training_bundle.py`: the config, script and guide written with an OmniVoice export. `fish_api/training.py` and `training_eval.py`: the opt-in local training runner and its evaluation record. `omnivoice/training/lora.py`: the in-repo LoRA.
- `backend/api/routers/{fish_audio,fish_story,fish_dataset,fish_training}.py`: the HTTP surface.
- `electron/src/renderer/src/features/{fish-story,fish-dataset}/`: UI, hooks and tests. Navigation entries are in `components/app-shell/workspace-menu.tsx`, `use-workspace-sidebar.ts`, `command-palette.tsx` and `routes/index.ts`.
- Tests: `tests/backend/engines/test_fish_{api,story,story_jobs,collection,dataset,training}.py`, `tests/backend/test_omnivoice_lora.py`, `tests/backend/api/test_fish_audio_route.py`, and the two feature folders' `*.test.ts(x)`. No live Fish call is needed and no key belongs in fixtures, logs, commits or docs.

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

## 8. Facts about the data (`D:\fish_tts\input` when this was written)

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
- **Quick win (now built):** the `omnivoice` export format, so the export dialog feeds the existing tokenizer with no manual conversion. See "What the LoRA phase built" below.

### Research findings (read from the code; nothing was downloaded or run)

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

### What the LoRA phase built

| Piece | Where | Notes |
|---|---|---|
| `omnivoice` export format + loudness matching | `fish_api/dataset.py` (`export`, `OMNIVOICE_TAGS`, `loudness_gain`), export dialog | Always 24 kHz mono, plain text plus only OmniVoice tags you keep (unknown target tags are refused), `language_id` default `en`. Loudness target is a whole-clip RMS in dBFS (the number `dataset_quality.measure` reports), peaks capped at 0.97, so a spiky clip can land a little below target. |
| In-repo LoRA | `omnivoice/training/lora.py`; `lora_*` and `eval_at_start` fields in `TrainingConfig`; wiring in `cli/train.py`, `checkpoint.py`, `trainer.py` | Plain torch, no `peft`. Wraps `model.llm` linears, `B` starts at zero, adapter saved as `checkpoint-N/adapter/` (safetensors + json), `merge_adapter` folds it back. Full fine-tuning is unchanged when `lora_enabled` is false. |
| Training-run module | `fish_api/training.py` | Background run with `job.json` + `run.log` under `<data dir>/fish_training/run_*/`; tokenise and train are child processes through a `Launcher` interface (`spawn_owned`, killed on cancel); dry run; hardware gate from a probe of the training Python; resume from the last complete checkpoint; interrupted detection; one real run at a time; unloads resident speech models first. |
| Evaluation gate | `fish_api/training_eval.py` | See below. |
| API | `routers/fish_training.py` (`/api/fish/training`, admin-protected, GETs behind `reject_cross_site_get`) | `GET /`, `PUT /enabled`, `POST /dry-run`, `POST /runs` (needs `confirm`), `GET /runs[/id[/log]]`, `POST /runs/id/cancel`, `POST /runs/id/resume`. |
| UI | `features/fish-dataset/{train-dialog,use-fish-training}`; button in the sidebar and in the export dialog after an OmniVoice export | Off until the user turns it on; dry-run report; runs list with progress, log, cancel, resume. i18n namespace `fishTraining`. |

**Evaluation gate (what exists, what is missing).** `select_heldout` writes a fixed sample of `omnivoice_val.jsonl` to `heldout.jsonl` in the run. The trainer (with `eval_at_start`) appends `{step, eval_loss}` to `output/eval_history.jsonl`, step 0 being the untouched base model (LoRA `B` is zero). `loss_check` requires at least a 2% relative drop in held-out loss. `wer`, `wer_check` and `similarity_check` are the arithmetic for the generation-based checks. `build_record` stores `evaluation.json` next to the adapter; **`ship_ready` is true only when all four checks (`heldout_loss`, `asr_wer`, `speaker_similarity`, `listening_test`) ran and passed**, so today a finished run is `incomplete` by design. Still needs a real GPU run: (1) generate the `heldout.jsonl` lines with the base model and with the adapter (needs adapter loading in the synthesis engine, not built), (2) transcribe both with an ASR model and feed `wer_check`, (3) embed generated and real clips with a speaker-embedding model and feed `similarity_check`, (4) a person listens. The thresholds (2% loss drop, 0.02 WER regression, 0 similarity gain) are placeholders to tune on a real run. Caveat: the trainer's eval loss is a masked-token cross-entropy with random masking, so compare with several evaluations, not one number.

**Run folder layout** (`fish_training/run_<time>_<id>/`): `job.json`, `run.log`, `train_config.json`, `data_config.json`, `tokens/{train,dev}/` (skipped on resume when `data.lst` exists), `output/` (checkpoints, `eval_history.jsonl`), `adapter/` (final adapter, `adapter.json`, `evaluation.json`), `heldout.jsonl`, `dry_run.json` for dry runs.

### Owner decisions (answered)

- **Where training runs:** mostly outside this machine (Google Colab), because of GPU limits. The dataset is transferred there, so the export must be self-contained: relative paths, the configs and a script. This is the `training` option of the export (`fish_api/training_bundle.py`, `train.sh`, `COLAB.md`). The in-app Train button stays for a machine that can train.
- **`peft` vs in-repo LoRA:** depends on the model that gets trained and will be settled out of this repo. The exported config uses the in-repo `lora_*` fields; the guide says how to carry the hyper-parameters over to another library.
- **Tags:** `[slow]` exists only in Fish output, so the owner wants a **user-built tag dictionary** (`[frustrated]:(sad)`: convert to whatever the target model understands; everything not listed is removed) plus plain removal as the default. Most sets have no tags to convert. The export dialog builds the dictionary from the unique tags actually present in the selection, so nothing has to be typed from memory.
- **Training defaults:** configurable per export (steps, learning rate, batch tokens, rank, precision, checkpoint cadence, base model, LoRA or full). The defaults below are still unmeasured. Steps should be set by passes over the data, not by a fixed number: 2000 steps is about 19 passes of the narrator's 4 hours but over 100 passes of a 30-minute voice (estimate, assuming 25 frames per second); watch the held-out loss and keep the best checkpoint.

### Owner decisions needed (original list)

1. **`peft` dependency.** Options: (a) keep the in-repo LoRA (~250 lines, no lockfile change, adapter format is this repo's own); (b) add `peft` to the training environment (standard adapter format, `merge_and_unload`, DoRA/rsLoRA for free, but a new dependency in `pyproject.toml`/`uv.lock` and a CI/Docker check, and a version pin against `transformers>=5.10`). Recommendation: stay with (a) until a second adapter consumer needs the PEFT format; switching later only touches `lora.py` and the adapter loader.
2. **Which base checkpoint.** The default is `k2-fsa/OmniVoice` (what `examples/config/train_config_finetune.json` uses). It is a Hugging Face download and the licence/terms are the model's. Options: that one, or the checkpoint the app's own OmniVoice engine loads (so the adapter matches what synthesis runs). Recommendation: use whatever the engine loads, once adapter loading in synthesis exists; confirm they are the same revision first.
3. **Tag policy.** The export keeps only tags the user lists and refuses ones OmniVoice does not know. Open: whether `[slow]` (on 99% of narrator clips) should be dropped (current behavior: always dropped from the text, so the model will not learn to be told "slow" but will still speak slowly) or re-rendered without it. Recommendation: drop it, and warn in the UI that a narrator trained on slow clips speaks slowly.
4. **Minimum data per speaker.** Current guidance: block under 20 clips, warn under 300 clips or 30 minutes (constants `MIN_CLIPS_BLOCK`, `MIN_CLIPS_WARN`, `MIN_MINUTES_WARN` in `training.py`). These are guesses. Only SS_Narrator (about 4 h) clears them; Noctis (34 min) and sun_2 (26 min) are borderline.
5. **Defaults.** 2000 steps, lr 1e-4, rank 16, alpha 32, `batch_tokens` 4096, backbone-only LoRA (`audio_heads` frozen). All unmeasured; a real run decides. `lora_train_modules: ["audio_heads"]` is a UI checkbox.
6. **Platform reach.** The trainer needs Triton for flex attention: native Windows needs `triton-windows` in the training Python (or WSL), and Apple Silicon/CPU are reported as unable to train. Per the CLAUDE.md parity clarification this is a host-capability limit, not a behavior divergence, but it is the owner's call whether to ship the Train button visible on such hosts (it is, with the reason shown) or hide it.

### Other targets

- **Fish S2 local:** not in the repo (only the cloud engine exists). The `fish_lab` export (WAV + `.lab`) matches Fish's own fine-tuning layout, but a local engine has to come first. The S2 Pro config shared during design shows a 36-layer text model plus a 4-layer audio decoder with 10 codebooks; which modules to target with LoRA was not investigated.
- **Qwen3-TTS:** nothing in the repo; its tag format, sample rate and trainer are unresearched.

### Infrastructure to reuse

- **Long jobs with progress, logs and resume:** `story_jobs.py` (background thread, `job.json` + `run.log`, interrupted-run detection, resume) and the smaller `dataset_quality.py` job.
- **Run training as a killable subprocess:** `backend/services/subprocess_backend.py`, `backend/core/contained_subprocess.py`, and [omnivoice-subprocess.md](omnivoice-subprocess.md) for why a wedged GPU job must be killable. Training needs a Python with CUDA torch, accelerate, webdataset, tensorboard, Triton and OmniVoice; the run's *Python for training* option points at it (default: the interpreter VoiceStudio runs with).
- **GPU and VRAM gating:** `backend/services/model_manager.py` (GPU pool, execution budgets, "abandoned worker") and `backend/services/gpu_sandbox.py`. Unload the resident TTS model before a run.
- **API and UI pattern:** `routers/fish_dataset.py` and `features/fish-dataset/`. A natural home is a *Train* step reached from the export dialog, with a runs list like the Jobs dialog.

### Constraints

- `docs/ROADMAP.md` (the on-device learning rows) says LoRA **ships only with an automatic evaluation gate**. The gate's record and loss check exist; the generation-based checks do not (see above), so nothing is marked ship-ready yet.
- `val.jsonl` uses a stable hash split, so re-exporting never leaks a validation clip into training.
- Local-first: no network for training.

### Data decisions needed before training

- One loudness level: now available as the export's loudness option (use it for any set that includes chapters 3193–3199).
- Whether `[slow]` stays in the narrator's text or those clips are re-rendered without it.
- Tag policy per target: Fish keeps its tags; OmniVoice only its non-verbal tags (list verified in the code, see the research findings).

## 10. Not built yet

A tag-to-`instruct` mapping for OmniVoice style attributes (whisper, pitch, gender); loading a finished adapter into the synthesis engine; the generation-based evaluation checks (ASR word error rate, speaker similarity, listening test); Fish S2 local and Qwen3-TTS trainers; deleting a training run from the UI; a `peft`-format adapter export.

## 11. Not verified

No live Fish call was made while building the later features, so voice-library details rely on the fields the existing voice listing already returns. The screens were tested with component tests, not driven by hand in the running app. LoRA on Fish S2 or Qwen3-TTS is unresearched.

**LoRA phase, sanity-checked on real data:** a 50-clip OmniVoice export from `D:\fish_tts\input` (25 hot-chapter and 25 quiet-chapter narrator clips) came out at 24 kHz mono with every clip measuring -24.0 dB RMS after loudness matching, and `validate_export` accepted it (read-only on the source, written to a temp folder and deleted).

**LoRA phase, not verified (needs a GPU or a model, none was available or allowed):** a real training run end to end; that `OmniVoice.from_pretrained(..., train=True)` followed by `apply_lora` trains under `accelerate` with flex attention and bf16 (the LoRA module is tested on tiny CPU models and a fake process launcher, the trainer edits only by compilation); the `accelerate.commands.launch` command line and the tokenizer flags (checked against the scripts' argument definitions, not run); memory use (estimates only); whether `accelerator.save_state`/`load_state` round-trips the wrapped model on resume; the Qwen3 layer names against a real Qwen3 backbone (that test is skipped on this machine because `torchaudio` fails to load; it runs where torchaudio works); the Train dialog in the running Electron app (component tests only); and the real `SubprocessLauncher.spawn` through `spawn_owned` (the probe is tested for real, the spawn is not). On this machine the training Python probe reports no GPU, no Triton, no `webdataset`/`tensorboard` and a broken `torchaudio`.
