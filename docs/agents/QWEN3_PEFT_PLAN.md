# Qwen3-TTS 1.7B: issue audit and LoRA (PEFT) plan

Written 2026-10-28. Focus: `Qwen/Qwen3-TTS-12Hz-1.7B-Base` (0.6B noted where it differs). Phases 1-4 were implemented on 2026-10-28 (see QWEN3_TRAINING_STATE.md, "Runner 21"); phases 5-7 are not started. Assumes **runner v20 lands first** (the working tree already shows `RUNNER_VERSION = "20"`; line numbers below are from that dirty tree and will move). Background: [QWEN3_TRAINING_STATE.md](QWEN3_TRAINING_STATE.md), [../engines/qwen3-finetune-findings.md](../engines/qwen3-finetune-findings.md), [../engines/qwen3-tts-colab-runtime.md](../engines/qwen3-tts-colab-runtime.md).

Tags: **V** = verified by reading code or measured by the reporter with numbers; **C** = claimed (reporter's word, no evidence shown); **I** = my inference, untested. Issue numbers are `QwenLM/Qwen3-TTS` unless prefixed `ez#` (mozi1924/Qwen3-TTS-EasyFinetuning) or `ins` (instavar/qwen3-tts-lora-finetuning).

## Part 1. Issue audit

"Ours" = `qwen3_colab_runner.py` (R), `qwen3_colab.py`, `qwen3_export.py`, `qwen3_narrator/` (N).

| # | Issue | Symptom | Root cause | Users' fix | Ours |
|---|---|---|---|---|---|
| 1 | 39, 286(lr) | Noise / never emits EOS; loss falls | lr 2e-5 too high (C, several agree, no A/B on aligned code); also missing text_projection (V) | lr 2e-6; 1e-6 | Handled: `DEFAULT_LR=2e-6` (R:50), 1e-6 used in the runs |
| 2 | 371, 337, 278, 384, 179, ez (not fixed) | Speech gets faster each epoch; EOS learned one frame late | Double label shift: manual `[:, :-1]`/`[:, 1:]` plus `ForCausalLMLoss` shift. V: reporter teacher-forced real codes: CE 1.87 vs j+2, 6.29 vs j+1 (371); 9.07 double vs 0.55 single (337) | PR 278/384, fumyou13 (179) | Handled: `ALIGN_B_*` (R:84-85). **Not proven by listening** (changed together with lr/silence) |
| 3 | 371, 337 | Sub-talker gets leaked hidden (position p instead of p-1) | V by same measurements (sub-talker CE 10.08 > ln2048) | `hidden[:, :-1][codec_mask[:, 1:]]` | Handled: `ALIGN_B_NEW` |
| 4 | 371 item 3, 278 | Sub-talker labels shifted again inside `forward_finetune`; head i learns codebook i+2, codebook 15 never | V (371: CE 7.24 as-is vs 10.03 shifted, stock model) | `shift_labels` / plain CE | Handled: `_aligned_sub_talker_loss` (R:88-89). `ins` patch leaves `modeling` untouched (so likely still double shifts, **I**) |
| 5 | 39, 198, 335, 120, 174, 297, 188, 336 | 0.6B crash `2048 vs 1024`; 1.7B silently wrong | Missing `talker.text_projection` (V; identity-shaped only on 1.7B) | project text embeds | Handled: `ALIGN_A_*` (R:82-83), val path too. 0.6B code-predictor dims in `_aligned_sub_talker_loss`: **unverified** |
| 6 | 286, 295 | `Logging with tensorboard requires a logging_dir` | accelerate 1.12 | `project_dir=` | Handled (R:188) |
| 7 | 372, 373 | `flash_attention_2` hard-coded; fails on ROCm/Blackwell/T4 | V | `sdpa` | Handled: `ATTN_FALLBACK_IMPL`, `build_patched_script(impl)` |
| 8 | 378, 379 | NaN grads / illegal memory above 8,192 frames per micro-batch (FA2 bwd in code predictor; sdpa on torch 2.8-2.11 too) | Reporter V with standalone repro; scoped to FA2 2.8.3 and torch bundled flash | run `code_predictor` eager; sdpa on torch>=2.12; cap 8,192 frames/batch | **Gap, latent.** Batch 4 x 20 s = ~1,000 frames, so far under. No guard or eager override. Risk rises with bigger batch on H100/80 GB |
| 9 | 365 | Non-atomic save leaves untrained base weights under final name; host RAM 2x | V (code read) | tmp dir + rename; `del state_dict` | Handled: tmp+rename, `_verify_saved`, `del state_dict` (R:140, 165) |
| 10 | 351 (open PR) | Speaker encoder "silently decays" under AdamW weight decay | **C.** AdamW skips params whose `.grad` is None, and the encoder runs under `.detach()`, so decay likely never applies (**I**, 5-line test would settle it) | freeze encoder, exclude from optimizer, reuse locked embedding | Not patched. Impact at lr 2e-6 x wd 0.01 would be ~6e-5 relative over 3k steps even if it applied (**I**). LoRA design removes the question |
| 11 | 72, 70, 314, 217 | Fine-tune sounds less like the speaker than zero-shot clone; less emotion | custom_voice has no ICL (V: type check in `qwen3_tts_model.py`, findings doc section 4); lr too high (C, #314: 4e-6 worse, 1e-8..1e-7 "better", loss stayed 11-14) | ICL on custom_voice (C, EthanLifeGreat); lower lr | Partly. "Clone on custom voice" built, **not yet listened to** (findings 9.1). lr claim conflicts with 2e-6 reports (see C3 below) |
| 12 | 343 | Short/opening utterances unstable (wrong timbre, gender flip ~15%), long form fine. LoRA r16, lr 2e-6, 20 ep | Unknown. Reporter ruled out serving path. Candidate: row 3000 not baked, or embedding mismatch (**I**) | Bake x-vector from un-FT'd Base into a registry and pass `speaker_embeddings` per call (PR 351 area); trailing silence; more short data | Partly: 1 s trailing silence (export, `Options.trailing_silence_s`); N has `_speaker_row` override (main.py ~238) used for clone_on_custom. No short-utterance eval set |
| 13 | 217, ez#6 | Cannot train several speakers in one model; cannot add a speaker to existing custom_voice | Script hard-codes row 3000 and one name (V) | ez: `3000 + idx*20`, cached embeddings; "cannot add to existing CustomVoice" (ez, C) | **Gap.** Single speaker only |
| 14 | 27, 323, ez (mozi) | New language: noise or accent bias (Bengali sounds Hindi, Turkish like American) | `dataset.py` hard-codes `nothink` prefill and drops the `language` field (mozi: "inferred from source", **C**); Turkish also fixed by lower lr (C, 3 users) | add language id to `config.json`, language-conditioned prefill (PR 361) | Not applicable to English narrator; trained and inferred language-neutral (QWEN3_TRAINING_STATE item 7). Gap only if non-English speakers are added |
| 15 | 179 | Which epoch? 19 closest to voice but unintelligibly fast; 5-7 right pace | Double shift (item 2) | fix 278; fumyou13: 20 epochs then improves (C, own numbers) | Same as item 2 |
| 16 | 39 | Short training clips -> audio decays after ~30 s; need long clips | C (one user, fixed by adding long samples) | add 30 s+ clips | Not handled: export clips are 4-20 s. Open question for long-form narration |
| 17 | 39 | Model speaks too fast after FT; 1 s trailing silence fixes | C (dariox1337); partly explained by item 2 | 1 s silence on training clips | Handled (export default `trailing_silence_s=1.0`) |
| 18 | 118 | ~0.5% of generations never emit EOS (base models too) | Sampling tail risk; `max_new_tokens` default 8192 | cap tokens; "add EOS ids [2150, 2157, 151670, 151673, ...]" (**C**, rebutted: talker vocab is 3072) | Handled: `max_tokens_for` (R:~2777), N `EosRamp` (`narrator_logic.py:94`), flagged "did not stop by itself". EOS-list claim not adopted |
| 19 | 5 (prepare_data) | VRAM leak in `prepare_data.py` | V, fixed upstream | clear lists | n/a (pinned commit post-fix; runner validates codes, `validate_codes`) |
| 20 | 89, 171, 43, 333 | Slow generation; fp16 assert on Turing; NaN logits with FA2 on inference | launch-bound loop (V); FA2 NaN (V, repro) | faster-qwen3-tts; sdpa | Handled: N uses faster-qwen3-tts + sdpa |
| 21 | 5 mozi (`ez#5`) | `Speaker X not implemented` | Name case: lookup is lowercase (V) | lowercase name | Handled: `model_speaker_name`, `fix_speaker_names` (R:~2553-2579) |
| 22 | 181, 55, 341, 386 | Last words swallowed; reference tail echoed; 40 ms codec echo | Inference/decoder side, clone path (C); not FT-specific | trim reference tail | Partly: `trim_tail` / `pad_tail` in N; not investigated |
| 23 | ez#14, ez#4 | Noise after FT on CustomVoice base; "what loss is normal?" | Same as 1/2 (C); no answer given | lower lr | Our loss plateaus ~2.0 with +/-0.1 noise; val loss added in v19 |
| 24 | 14 | Cannot combine custom speaker with instruct | Base/CustomVoice split (V); maintainers: future `25Hz-1.7B-VoiceEditing` | wait | Known (findings section 6): instruct moves the voice away from the narrator |

### Claims that conflict

| Id | Claim A | Claim B | Status |
|---|---|---|---|
| C1 | Sub-talker slice and label handling are "upstream-correct" (Baseten ml-cookbook) | They double-shift and leak (371, 337, 278, fumyou13) | A is not tested with numbers; B has teacher-forcing numbers. Ours follows B. Only listening to an unfixed A/B baseline settles audio impact |
| C2 | Drop codec groups 1-15 summing in the talker input (dragonfyre13, #179) | Keep it: generation feeds the sum of all 16 levels (humblenginr, fumyou13, cites modeling L1671) | A reverted. Ours keeps it |
| C3 | lr 2e-6 is right (many) / 1e-6 (fumyou13, ours) | 1e-8..1e-7 (#314, ez default 1e-7) / 5e-6 (Baseten) / 2e-5 works after alignment fix (Zahgrom34, 337) | Unresolved. Confounded by the alignment bug and bf16 rounding (I). Do a fixed-seed sweep on the **aligned** script only |
| C4 | More epochs help (fumyou13: 20) | Overfit after ~10 (ins, **C**); epoch 9 worse than 19 (343); owner: ep 9 of 10 best, 4-7 had pacing problems | Unresolved; our loss does not separate epochs. Use val loss + listening |
| C5 | Sub-talker loss weight 0.3 (upstream, ours, ez) | 1.0 (ins patch `codec_0_loss + sub_talker_loss`) | Untested. Ours matches upstream (`_val_loss`, R:~162) |
| C6 | Fine-tune carries ICL if `tts_model_type` is flipped (EthanLifeGreat) | custom_voice rows have no speaker-encoder; hybrid test drifted voice (findings 5) | Only the C-side claim is untested |
| C7 | Speaker encoder is wrongly decayed (351) | AdamW skips grad-less params (I) | Needs a test before any work |
| C8 | Row embedding should be averaged over the run (ez multi-speaker note, 351 text) | Lock first embedding (upstream) / cache once (ez embed step) | 351 says averaging collapses norm ~1/(epoch+1) (**C**). Plan caches once |
| C9 | Language id token fixes accent (mozi) | Plain lr fix was enough for Turkish (3 users) | Both C |

### Gaps in our handling, most important first

1. **No multi-speaker training** (row 3000 hard-coded; item 13).
2. **Full fine-tune in bf16 without fp32 master weights**: low-lr updates may round away (docs 6b, **I**). No LoRA or fp32 path exists.
3. **No guard for the 8,192-frame micro-batch limit** (item 8) and no eager code-predictor override.
4. **Short/onset instability (item 12) has no test set and no eval**; per-epoch sample is one long sentence.
5. **Speaker row is the first-batch embedding** of the shared `ref.wav`; with several speakers or references this must be a cached, averaged, fixed vector (item 10 / C8).
6. Clips are 4-20 s only (item 16); no long-form check.
7. Language field untrained (item 14), acceptable for English.

## Part 2. Plan: LoRA for talker + code predictor (sub-talker)

### What mozi's EasyFinetuning actually does (read at `03c53fa`, 2026-05-30)

| Feature | In `src/sft_12hz.py` / `embed_speaker.py` |
|---|---|
| Cached speaker embedding | `embed` step: Base `speaker_encoder` on CPU, `torch.no_grad`, optional average over several reference clips, saved `final-dataset/<spk>/speaker_emb.safetensors` (norm logged). Training loads these and never runs the encoder |
| Multi-speaker | `--speaker_name a,b`; `spk_id_map = {name: 3000 + idx*20}`; per-sample `batch["speaker_ids"]` picks the cached vector; `input_codec_embedding[:, 6, :] = speaker_embedding`; save writes each vector into `codec_embedding.weight[spk_idx]`, `talker_config.spk_id`, `spk_is_dialect` |
| Resume | `resume_from_checkpoint="latest"`, trainer state (epoch, step_in_epoch, global_step), accelerate state dir, optimizer/scheduler, speaker embeddings stored in `training_state.pt`; TensorBoard `purge_step` |
| Step checkpoints | `save_steps=200`, `keep_last_n_checkpoints=3`, `save_strategy` step/epoch/both |
| Defaults | lr **1e-7** (CLI and function), batch 2, grad accum 4, 2 epochs, `AdamW(model.parameters(), wd=0.01)` (so encoder and everything else is in the optimizer), `DummyLRScheduler` (no warmup/cosine) |
| **Not fixed there** | Still `inputs_embeds[:, :-1]` + `labels[:, 1:]` (double shift) and `hidden_states[codec_mask[:, :-1]]` (leak). It does apply `text_projection`. No LoRA. Its own maintainer says it has not run many full trainings (ez#4) and multi-speaker consistency was reported poor (ez#15, "root cause identified", unreleased) |

Take its **plumbing** (cached embeddings, id map, resume, step saves). Do **not** take its loss path or lr.

### Community LoRA attempts (what they used)

| Source | Adapters | r / alpha / dropout | lr | Notes |
|---|---|---|---|---|
| `instavar/qwen3-tts-lora-finetuning` (patch on upstream `0c6a7cb`) | `get_peft_model(whole model)`, names `q,k,v,o,gate,up,down_proj` (suffix match, so talker **and** code predictor) | 16 / 32 / 0.05, bias none | 2e-6 (!) | single shift manual CE, `hidden[codec_mask[:, 1:]]`, sub weight 1.0, `text_projection`; saves `save_pretrained` + `speaker_embedding.safetensors`, row 3000 **not** written into weights; infer `lora_scale` 0.3-0.35 "best", merge via `merge_and_unload` |
| `6z6z6z6z/qwen3-tts-personalized` | `get_peft_model(model.talker)` (talker + nested code predictor), all base params frozen first | 16 / 32 / 0.05, ~0.99% trainable | script default 2e-5, README example 2e-5 | optional fixed `--speaker_embedding_path`; mixed-context M3 inference beats embedding-only LoRA (their MOS table) |
| #343 reporter | LoRA r16, merged at scale 0.4 | 16 | 2e-6, 20 epochs | onset instability, baked embedding |

No source gives a Qwen-specific rank/lr sweep; the scale-0.3 finding with lr 2e-6 suggests nobody has a calibrated recipe. **Defaults below are starting points, not facts.**

### Design

| Topic | Decision |
|---|---|
| Base | Frozen, bf16 (as today). Adapters injected with `peft.inject_adapter_in_model(cfg, model.talker)` so the model class and `forward_sub_talker_finetune` / `get_input_embeddings` stay intact (a `PeftModel` wrapper hides them; 6z6z wraps `model.talker` and works, **I** that inject is cleaner) |
| Adapted | `q,k,v,o,gate,up,down_proj` in `talker.model.layers.*` (28 layers, hidden 2048) and `talker.code_predictor.model.layers.*` (16 heads x 128). Use an explicit regex so nothing outside `talker.` matches |
| Frozen | `speaker_encoder` (not even loaded: embeddings cached), `text_projection`, `text_embedding`, `codec_head`, code-predictor `lm_head`s and per-group embeddings, `codec_embedding` (including EOS row 2150). Rows 3000+ written from cached vectors. v2 option: `modules_to_save` for `codec_head`/EOS if end-of-speech proves hard to learn with attention/MLP adapters only (**I**) |
| rank / alpha / dropout | 16 / 32 / 0.05 (both community repos). Try 32/64 only if val loss stalls |
| lr | **Updated 2026-10-28: the range below is contradicted by instavar/qwen3-tts-lora-finetuning, whose LoRA default is 2e-6 (constant, no warmup, wd 0.01, clip 1.0, batch 4 x 4, 10 epochs), with a merge-time LoRA scale of 0.3 (1.0 "almost always wrong"). The sweep must also include 2e-6 and 1e-5, and the merge scale (1.0 and the 0.3 candidate; 0.2-0.5) is a sweep axis. Implemented as the placeholders `LORA_DEFAULT_LR 2e-6 = 6z6z; 1e-4 = general LoRA practice, **not Qwen-specific**; 2e-6 (instavar) is the full-FT value and is almost certainly too low for adapters (**I**). wd 0, grad clip 1.0 |
| Alignment fixes | Carry over unchanged: `ALIGN_A/B/C` act on loss computation, independent of which params train. Adapters sit inside the layers, so `hidden_states`, text projection, `_aligned_sub_talker_loss`, 0.3 weight, `VAL_*` all apply as is. Only the optimizer line (`AdamW(model.parameters())`) and save block change |
| Precision | Base bf16. peft casts adapter weights to fp32 when the base is bf16 (`autocast_adapter_dtype=True`, default in recent peft; **verify on the pinned version** and assert `lora_A.weight.dtype == float32` at start). Forward under accelerate bf16 autocast; optimizer state fp32 on adapters only. This removes the docs 6b concern for **training** |
| Merge rounding (new risk) | `W_bf16 + (alpha/r) B A` rounded once to bf16 can erase a small delta, the same effect as 6b. Do the add in fp32 and report `changed %` per tensor like `_verify_saved`; if the changed share of adapted tensors is below a threshold, **stop and say so**; fall back to saving the merged checkpoint in fp32 (7.3 GB) or shipping adapter + unmerged apply. Compare logits merged vs unmerged on a fixed batch (GPU) |
| Adapter-only checkpoint | `checkpoint-epoch-N/` with `adapter_model.safetensors`, `adapter_config.json`, `speakers.json` (`name -> row`, lowercase), `speaker_embeddings.safetensors` (name -> fp32 2048), `lora_meta.json` (base repo id + snapshot hash, tokenizer id, runner version, ALIGN flags, r/alpha/lr, lora dtype). ~100-200 MB vs 4.2 GB per epoch, so keep all epochs on Drive |
| Merge to standard checkpoint | New `merge_adapter(base_dir, adapter_dir, out_dir)`: copy base tree, apply deltas in fp32 -> bf16, drop `speaker_encoder.*`, write rows into `talker.model.codec_embedding.weight`, set `tts_model_type=custom_voice`, `talker_config.spk_id`/`spk_is_dialect`, copy `speech_tokenizer/`. Same layout as EP09, so `read_checkpoint` (registry.py) and the sidecar load it unchanged. Runs on Colab (after training) and locally (CPU, no GPU needed) |
| Atomic save | Reuse the runner's tmp-dir + rename; verify that adapter tensors are non-zero (B matrices start at 0, so "B all zero" = nothing trained) |
| Resume / step saves | Adapter + optimizer + step + RNG in the checkpoint (small), `SAVE_STEPS` option, mirrors ez. Skip accelerate full state |

### Quick speaker switching

| Part | Approach |
|---|---|
| (a) Multi-speaker, one run | Export writes `speaker` per clip and per-speaker reference clip. Prepare step caches one vector per speaker (mean of up to N clips, Base encoder on CPU, saved `speaker_embeddings.safetensors`, norm logged; reject norm outside the stock-row range ~9-17 as a sanity check, **I**). Row map `3000 + idx*20`, lowercase, validated against `STOCK_SPEAKERS` ids (registry.py) so a trained row never equals a stock id. Patched `TTSDataset` returns `speaker`; the loop gathers the cached vector per sample (ez `sft_12hz.py` ~692-712). Balance clips per speaker; per-speaker val loss |
| (b) No reload to switch | A merged multi-speaker checkpoint already switches by the `speaker` argument: N looks the id up per call (`generate_custom_voice(speaker=...)`, `_speaker_row` in `main.py`). Nothing to reload. For **separately trained per-speaker adapters** (v2): do not use `set_adapter` under faster-qwen3-tts (CUDA graphs and static cache are captured on fixed kernels). Instead swap weight deltas **in place** (`param.copy_`) from a CPU copy of the base, so pointers stay valid; this must be verified on the GPU (graph replay after swap, drift after many swaps, peak VRAM). Default recommendation: one multi-speaker adapter, merged |
| (c) Settings -> Storage | No new path boxes. Merged and adapter folders live in any location from `model_locations.load()`; `discover_models` -> `_scan` already finds merged checkpoints. Add an `adapter` kind to `read_checkpoint` (folder has `adapter_config.json`, base resolved from `lora_meta.json` via the same model list: HF cache + locations, never a private root). Merge output goes to a user-chosen location row (needs the folder authorization token, `backend/core/path_authorization.py`). The Colab side writes to Drive; the user copies into a registered folder |

### File-by-file change list (after runner v20 lands)

| File | Change |
|---|---|
| `backend/engines/fish_api/qwen3_peft.py` (new, pure Python) | LoRA patch constants (optimizer + save-block replacements for the pinned `sft_12hz.py`), target regex, `speaker_rows()`, `write_checkpoint_config()`, `merge_adapter()`, `verify_merge()`, adapter dtype assert. Importable without torch where possible |
| `qwen3_colab_runner.py` | Version 21. Params `TRAIN_MODE` (`full`/`lora`), `LORA_RANK/ALPHA/DROPOUT`, `SAVE_STEPS`, `SPEAKERS`. `build_patched_script` gets a LoRA variant (still refuses other scripts). Setup installs `peft` with `--no-deps` and probes it (`dependency_rows`). Frame-budget guard (item 8). Retention counts adapter checkpoints |
| `qwen3_colab.py` | Parameters cell fields, notebook text, regenerated `notebooks/Qwen3_TTS_Colab_Training.ipynb` |
| `qwen3_export.py` | `speaker` per clip, per-speaker reference, balance report |
| `backend/engines/qwen3_studio/registry.py`, `playground.py` | `adapter` kind, multi-speaker `speakers` list already supported, "merge adapter" action |
| `backend/engines/qwen3_narrator/main.py` | No reload logic needed; add test coverage for sequential speakers. Optional v2 in-place delta swap |
| Docs | This file -> status in QWEN3_TRAINING_STATE.md, colab-runtime guide section 6c, `notebooks/README.md` (docs-sync rule) |

### Tests (fail before, pass after; no GPU)

1. LoRA patch applies to the real pinned `sft_12hz.py`, result compiles, any other script is refused (mirror existing ALIGN tests in `tests/backend/engines/test_fish_qwen3_colab_runner.py`); unpatched script fails the test.
2. `speaker_rows(["B","a"])` -> lowercase, `3000 + 20*idx`, no stock-id collision, duplicate/case-duplicate names rejected.
3. Merge math: tiny random weights, `merge_adapter` equals `W + (alpha/r) B A` in fp32; zero-B adapter -> verify step fails loudly (analogue of `SAVE FAILED`).
4. bf16 rounding: a delta below bf16 resolution reports `changed %` near 0 and trips the threshold.
5. Config writer output passes `registry.read_checkpoint` with all speakers, `speaker_encoder.*` absent.
6. Index-math helper (extracted pure function) reproduces single-shift alignment on a toy sequence; double-shift version fails (guards ALIGN carry-over).
7. Frame-budget guard refuses batch x frames > 8,192.
8. Notebook generator embeds the `peft` pin with `--no-deps`; parameters cell round-trips the new fields.
9. Sidecar: two speakers in sequence with a fake model, `_load_model` called once.
10. Registry: adapter folder detected; location scan finds a merged multi-speaker folder.

### GPU-only (cannot be verified here)

Adapter grads actually reach the code predictor through `forward_sub_talker_finetune`; peft 0.17.1 + transformers 4.57.3 import and `inject_adapter_in_model` on this model; adapter dtype fp32; VRAM and speed with frozen base (should drop; unmeasured); lr/rank choice; EOS learned with attention/MLP adapters only; merged vs unmerged logits and `changed %`; whether the merged checkpoint sounds like the unmerged; onset stability on short lines (343); multi-speaker separation and row spacing; in-place delta swap under CUDA graphs; whether LoRA beats the epoch-9 narrator.

### Colab 3.13 constraints

- `transformers==4.57.3`, `accelerate==1.12.0`, `huggingface-hub==0.36.2` come from Qwen's package; never upgrade them (runtime guide section 4).
- PyPI metadata (checked 2026-10-28): `peft` 0.17.1 requires Python >=3.9, 0.19.1 and 0.21.2 require >=3.10; all are pure `py3-none-any` wheels with **unpinned** `transformers`, and need `accelerate>=0.21`, `huggingface_hub>=0.25`, `safetensors`, `torch>=1.13`. So nothing blocks 3.13, but the metadata does not prove newer peft works with transformers 4.57.3 (it may assume 5.x APIs). Pin **`peft==0.17.1`** (nearest to the 4.57-era), install with `--no-deps`, then probe the import and a one-layer injection in the setup table.
- The local sidecar venv uses transformers 5.15.1 (`requirements.txt`); it needs **no** peft if it loads merged checkpoints only. Keep peft out of the sidecar.
- Checkpoints trained under 4.57.3 already load under the sidecar (EP09), so merged output keeps that property.

### Ordering (avoid conflicts with runner v20 work)

1. Now (no conflict): this doc; new `qwen3_peft.py` + its tests only; pure helpers (rows, merge, config writer, frame guard).
2. After v20 lands: runner v21 integration in one commit (params, patch variant, setup probe), notebook regen.
3. Export: per-speaker clips and cached embeddings step (touches `qwen3_export.py`, no runner overlap).
4. Registry/playground `adapter` kind and merge action; sidecar tests.
5. Owner-approved GPU smoke: single-speaker LoRA, 2 epochs, read `changed %` and listen; then a two-speaker run.
6. Lr/rank sweep with fixed seeds on 10-20 unseen sentences, three seeds (QWEN3_TRAINING_STATE next step 3).
7. Only then v2: delta-swap hot adapters, `modules_to_save` for the heads, language ids for non-English speakers.
