# Qwen3-TTS 1.7B: emotion on the narrator (research and plan)

Written 2026-10-28. Nothing here was run: the C engine was read, not built (clone at `C:\Users\User\AppData\Local\Temp\claude\qwen3-research\c-engine`, HEAD `ef339be`, 2026-09-25; `faster-qwen3-tts` beside it in `fq`). Tags: **R** = read in their source (file:line), **I** = my inference, **U** = unknown until run. Paths below are inside the C repo unless prefixed `VS:`. Background: [QWEN3_TRAINING_STATE.md](QWEN3_TRAINING_STATE.md), [QWEN3_PEFT_PLAN.md](QWEN3_PEFT_PLAN.md), [../engines/qwen3-narrator.md](../engines/qwen3-narrator.md).

## 0. Summary

| Question | Answer |
|---|---|
| Does the C engine load our custom_voice checkpoint? | Very likely, with **no conversion** and the speaker resolved by name from `config.json` (R). Never executed (U). |
| How are `.qlsteer` made? | Mean residual-stream activation (emotion instruct) minus (neutral instruct), per layer, cleaned. **No training.** We can make them for our speaker with HF hooks (R). |
| Do the `ryan_*` vectors transfer? | Ear-validated by the author on **stock presets** across languages; **not** validated on a fine-tune. They say a clone needs more (R). |
| How are `.expr` made? | PEFT LoRA r32/r64 on talker L16-26 (q,k,v,o,gate), instruct-conditioned, random preset speaker, **unaligned loss**, lr 2e-4 (R). We would redo it with our runner. |
| Ship the C engine? | **No, not now.** Port steering to the Python sidecar, train an emotion LoRA with our runner, and run one optional 30-minute WSL check of the C engine (section 4). |
| Biggest new risk | Our narrator is trained from **Base**, which has no instruct training (I); the C emotion recipe depends on CustomVoice's instruct competence. See section 2 and 6. |

## 1. Can the C engine load our checkpoint? (R, not executed)

| Item | Evidence | Verdict |
|---|---|---|
| Weights | `ingot_st_open_dir` takes `model.safetensors.index.json`, else single `model.safetensors`, else any `*.safetensors` (`third_party/ingot/amalgam/ingot.c:2095-2140`). Weights are mmapped. | Our single `model.safetensors` is fine |
| Tensor names | `talker.model.text_embedding.weight`, `talker.text_projection.linear_fc{1,2}.{weight,bias}`, `talker.codec_head.weight`, `talker.model.codec_embedding.weight` (`qwen_tts_talker.c:487-502`) | Same names as the HF state dict we save (I) |
| dtype | `get_bf16` returns NULL unless the tensor is BF16 (`qwen_tts_talker.c:92-96`) -> load error. Norm tensors may be any dtype (`get_f32`, `:98-107`). | Our full-FT and merged-LoRA saves are bf16; an fp32 fallback save (PEFT plan, merge rounding) would **not load** |
| Speech decoder | `speech_tokenizer/model.safetensors` required (`qwen_tts.c:793-800`) | Our export copies the folder |
| Model type | `tts_model_type` starting with `base` sets `is_base_model` (`qwen_tts.c:736-742`); anything else is treated as CustomVoice | `custom_voice` OK |
| Speaker table | `parse_spk_id_table` reads `talker_config.spk_id` `{"name": slot}` from `config.json` (`qwen_tts.c:164-206`, called `:763-764`). An **empty** `spk_id` makes it VoiceDesign (`:750-761`). | `{"ss_narrator": 3000}` is registered by the file we already write |
| `-s <name>` | `qwen_tts_resolve_speaker`: model table first, case-insensitive, then the hardcoded preset table (`qwen_tts.c:106-113`; `main.c:1868-1878`); `--list-speakers` prints the table (`qwen_tts.c:148-162`); `--speaker-id N` bypasses names (`main.c:1151`) | Custom name works; **no 4 KB voice asset needed** |
| Row lookup | Slot goes into the codec prefix, embedding read from `codec_embedding` row `speaker_id`, bound `codec_vocab_size` (`qwen_tts.c:1172-1176, 1062-1068`) | Row 3000 baked into the weights is used as in HF |
| Language | No `-l` -> `NO_THINK` prefix (`qwen_tts.c:1160-1166`) | Matches our language-neutral training |
| `spk_is_dialect` | Not referenced anywhere in the C source (grep) | Ignored; harmless for an English narrator |
| Conversion | None. Their `.qvoice`, `.bin`, `.expr` are add-ons, not model formats. Optional quantization at load (`--int8`, `--quant-mixed`) | The 1.7B CUDA RTF 0.44 figure is **`--quant-mixed` (int4 talker)** (`docs/cuda-performance.md:24-31`). bf16 fidelity on our fine-tune is untested (U) |

Gaps: the engine was never shown a fine-tune; EOS behaviour differs from ours (their ramp `QWEN_EOS_V1` vs our `max_tokens_for` + `EosRamp`), so "did not stop" cases need a re-check (U).

## 2. `.qlsteer` steering vectors

### How they are made (R)
1. Run the engine twice with `QWEN_ACT_MAP=<file>` and `-I "<instruct>"`: once with a neutral instruct, once with an emotion instruct (`presets/steer/emotion/MANIFEST.md`, "How to rebuild").
2. Capture (`qwen_tts_talker.c:20-62, 797-824`): after each talker layer's residual add, the 2048-d vector is summed per layer; the dump is the **mean over all decode frames** of the run (29 x 2048 floats: 28 layer outputs + the final normed hidden). One utterance per file; no dataset.
3. `tests/act_map_steer.py` (all 62 lines): per layer `d = emo - neutral`; `--clean` = subtract the mean across the 2048 dims, then project out the neutral activation direction; `--unit-per-layer` = L2-normalise. Writes magic `QLST`, L, D, floats.
4. Dyads are weighted sums (`tools/steer/dyad_mix.py`).

### How they are applied (R)
- Only on **decode** steps, layers 21-25 (`--ml-range 21-25`), weight 12 for presets, decayed `0.985^frame` (`qwen_tts.c:1883-1886`). At each layer: `x += w*v`, then `x *= sqrt(|x_old|^2/|x_new|^2)` (norm-preserving rotation, `qwen_tts_talker.c:803-818`). Prefill is not steered (`:1956`).
- **Performance catch:** whenever steering is on, the fused CUDA and Metal talker step is bypassed (`qwen_tts_talker.c:690-700`, `qwen_tts.c:1601-1612`); no `.cu`/`.m` file mentions steering (grep). So steered decode probably runs the talker on the CPU path (I). The published GPU RTFs do not apply with emotion on.
- Vectors are hardcoded to `presets/steer/emotion/ryan_<tok>.qlsteer`, a relative path (`qwen_tts_emotion.c:46-50`).

### Could we make them for our speaker?
Yes, and cheaply (I): hook `talker.model.layers[i]` outputs in HF (`output_hidden_states`, index offset +1 vs their layer numbering), run the narrator row 3000 with a neutral instruct and an emotion instruct, average over decode frames, same clean/normalise math. Needs: the checkpoint, ~20-50 neutral/emotion sentence pairs (their method uses one pair; more pairs and averaging is our improvement, U), no training. Minutes on a 3070.

### Do `ryan_*` transfer?
| Claim | Evidence | Strength |
|---|---|---|
| Ryan's palette works on other presets and languages | `docs/emotion-THE-recipe.md` "The recipe": "used for ALL voices (it transfers cross-voice/cross-language)"; ear-validated, seed 42, T1.1 | Author's ear only; no blind test |
| Clones need more | Recipe: clone -> COMBINE (`.expr` + steer + English instruct). `docs/expressivity-lora.md` section 4: a clone "emotes ~3-4x weaker"; x-vector clone needs weight 1.6-2.0 and T1.3 (`docs/csp-ft-emotion.md:33`); `MANIFEST.md`: voice-native vectors `galatea_*_ft` captured **with** `.expr` | Strong hint that identity-conditioned voices damp steering |
| Fine-tuned voices | Nothing. The only fine-tunes in their repo are their own expressivity bands on CustomVoice | **U** |
| Automatic scoring exists | `tests/emo_score.py` uses `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` (arousal/dominance/valence) | Usable as a sanity check, not as truth |

Verdict: do **not** reuse `ryan_*` blindly. Capture our own on the narrator and compare against the transferred ones at fixed seeds. Our narrator row is baked into the weights like a preset (no x-vector injected), so it should behave more like a preset than a clone (I).

### Which base: the Base-derived narrator is the catch
Their whole emotion stack presumes **CustomVoice**: instruct and `.expr` live in talker L16-26 (`docs/expressivity-lora.md` section 2, Base vs CustomVoice weight diff). Our narrator is a full fine-tune of **Base** with no instruct text, so instruct competence is probably weak or absent (I). Evidence from our side: instructions changed length and moved the voice away from the narrator on EP09 (`../engines/qwen3-finetune-findings.md` section 6). Their own docs disagree on how much Base and CustomVoice differ ("99.98% identical", `docs/custom-voices.md` graft section, versus a concentrated L16-26 diff in `expressivity-lora.md` section 2): U. Steering vectors captured from an instruct pair therefore depend on the narrator following the instruct at all; if it does not, capture from stock CustomVoice and test transfer instead.

## 3. `.expr` and emotion LoRA

### How `.expr` is produced (R)
| Step | Source | Detail |
|---|---|---|
| Data | `training/expressivity-lora/prepare_manifest.py:1-40`, `prepare_cremad.py`, `prepare_esd.py` | wav 24 kHz + `text` + `instruct` (vivid English per emotion, e.g. "Speak with a sad, sorrowful, downcast tone...") + `emotion`; neutral has empty instruct |
| Codes | upstream `prepare_data.py` | adds `audio_codes` (12 Hz tokenizer) |
| Dataset | `dataset.py:98-165` | ChatML: instruct as `<|im_start|>user\n{instruct}<|im_end|>\n` tokens first, codec pad, mask on; then the usual text/codec layout; labels only on codec frames and EOS; `nothink` prefill; no language tag (mixing languages corrupted Italian, `MULTISPEAKER_PIPELINE.md` TL;DR 2-4) |
| Train | `train_lora.py:12-76` | `get_peft_model(whole model)`, targets `q,k,v,o,gate_proj`, layers 16-26, r32/alpha64 (preset) or r64/alpha128 (clone), dropout 0.05, **lr 2e-4**, wd 0, batch 8 x accum 4, 8 epochs, eager attention, `PRESET_IDS=[3066,3065,3010,3061,2861,2873,2864,2875,2878]` -> a **random stock speaker row per sample** (`:58-61`) so the adapter does not bake in one timbre |
| Loss | `train_lora.py:65-70` | `inputs_embeds[:, :-1]` + `labels[:, 1:]` through the library loss (double shift), no `text_projection`, hidden taken from `[:-1]` mask: the **unaligned** pattern we patched (`VS:backend/engines/fish_api/qwen3_colab_runner.py` `ALIGN_A/B/C`) (I: same shape as the upstream bugs, not run) |
| Export | `export_expr.py:36-113` | per target weight writes A, B (f32) and `alpha/r` as dtype-5 tensors, `QEXP` header; engine reconstructs `W += scale*(B@A)` at load (`main.c:551+`, `docs/expressivity-lora.md` section 5). Optional dense dtype-4 path, plus a `text_embedding` delta |
| Loader limits | `main.c:575-675` | only `talker.model.layers.N.*` (q,k,v,o,gate,up,down, norms), text projection, embeddings, head; **code-predictor LoRA tensors are not applied** |

Their data: EMOVO 588 clips (6 actors, 7 emotions) for the Italian pack; r128 overfit it. A later English-dominant mix of ESD (10 speakers) + CREMA-D (91 actors, 12 fixed sentences) + EMOVO gave better English emotion but wrecked untagged Italian.

### What must change to train an emotion LoRA on OUR checkpoint
| Topic | Their trainer | Ours |
|---|---|---|
| Base weights | CustomVoice | our EP09-class custom_voice checkpoint, frozen bf16 |
| Loss | unaligned | `ALIGN_A/B/C` + `_aligned_sub_talker_loss` (already in the runner, applies to any trainable set) |
| Adapter dtype/optimizer | peft `get_peft_model`, bf16 | runner's `inject_adapter_in_model` + fp32 adapters (`LORA_*`, runner 21) |
| Targets | talker L16-26 q,k,v,o,gate | start with talker L16-26 same set (so an `.expr` stays possible); never the code predictor (the C loader drops it); do not use runner's default regex that includes code predictor + all 28 layers without a reason |
| lr | 2e-4 | placeholders `LORA_DEFAULT_LR=2e-5`, scale 1.0/0.3 are unconfirmed; the sweep must include 2e-6, 1e-5, 2e-5 and merge scale 0.3-1.0. 2e-4 is not a safe copy: it is 100x the narrator's full-FT value |
| Speaker conditioning | random stock row per sample | our checkpoint has **no stock rows** (Base-derived, Ryan does not live in Base, findings section 4) and **no speaker encoder**. Inject per-sample vectors at codec position 6: (a) the **narrator row 3000 read from the checkpoint** for narrator samples, (b) cached embeddings of the emotion-corpus speakers made once with the stock Base encoder (runner already has `_setup_speakers`, rows not saved), (c) stock CustomVoice rows 3066 etc. copied from the stock CV checkpoint (Base and CV share the embedding space per `docs/custom-voices.md` graft section, I). Needs a new patch variant "speaker vector from the checkpoint row"; the pinned script loads the encoder from Base |
| Identity preservation | none (voice-agnostic only by randomisation) | (1) frozen `codec_embedding` incl. row 3000, frozen heads and text projection (already in the LoRA design); (2) **replay**: 30-50% of every batch is the narrator's own clips with empty instruct and row 3000, same loss; (3) low merge scale; (4) the speaker-similarity gate in section 7 |
| Data text | EMOVO/CREMA-D fixed short sentences | vary text; narrator-style long sentences; fixed sentence corpora teach text, not emotion |
| Instruct | English prose, strong | same style, 6 primaries x 2 strengths from `docs/emotion-instruct-control.md` (R) |
| Output | `.expr` | merged checkpoint (HF) **and** adapter; `.expr` export only if the C route is adopted |

## 4. Integration options

Facts for the platform rule (R): C engine runs on macOS (Metal or CPU), Linux and "Windows/WSL2 - Beta" (`docs/building.md:74-94`); no native Windows target (`_WIN32` appears only in thread/kernel guards, `qwen_tts_thread.c:119`, `qwen_tts_kernels.c:457`; POSIX `mmap`, `dirent`, `strcasecmp` elsewhere); CUDA via `make cuda`, sm_80/86/89/120 in one binary (`Makefile:305-309`, covers the 3070 = sm_86) with the CUDA server "work in progress, never qualified" (`README.md:581`); MIT licence (`LICENSE`), vendored `ingot` MIT, `kleidiai` Apache-2.0 + NOTICE (`third_party/`), so shipping needs a third-party notice entry; model weights stay Qwen's Apache licence (U: verify exact text).

| | (a) Python only: port steering + emotion LoRA | (b) C engine sidecar | (c) both |
|---|---|---|---|
| New code | hook module (5 layers x add, rescale) captured **inside** the existing CUDA graph: `TalkerGraph._decode_step` calls the model's own forward (`fq/faster_qwen3_tts/talker_graph.py:99-110`), so forward hooks on `talker.model.layers[21..25]` that read static buffers (vector, weight scalar) are baked into the graph at capture; per-frame decay = copy a scalar into the static buffer before `replay()` (I) | build + packaging + IPC + a second stack; native Windows impossible (WSL2 only) | both costs |
| Platform parity | same as today: narrator is CUDA-only, opt-in (`qwen3-narrator.md` "Platform support"); no regression | Metal gives macOS a path the narrator lacks; Windows needs WSL2: an opt-in only (CLAUDE.md parity rule) | opt-in on top |
| Speed on the 3070 | cost: 5 elementwise ops per step, negligible (I). Measured baseline 2.25x real time | with steering: CPU talker path (above); without: their claim RTF 0.44 only with int4 `--quant-mixed` | |
| Fidelity risk | none beyond the checkpoint | int4/int8 on a fine-tune untested | |
| Colab | capture vectors and train LoRA on the A100 (existing runner) | not needed | |
| macOS | stays unsupported (U until a CPU/MPS path exists) | **only route to macOS**, CPU or Metal, never tried with our checkpoint | |

**Recommendation: (a) now; (b) only as an owner-run feasibility gate, not product work.** Reasons: all emotion value (vectors, LoRA) is produced by Python tools we already own; steering in the C engine forfeits GPU decode; native Windows, the owner's platform, is unsupported there; our checkpoint's EOS and quality behaviour are unvalidated in C. Gate for (b), owner runs it in WSL2/Linux/macOS (commands as text, nothing run by me): build `make blas` (or `make cuda`), `./qwen_tts -d <checkpoint> --list-speakers`, then `-s ss_narrator --text <fixed sentence> --seed 42`, compare by ear to the Python render. If it loads and sounds the same, macOS support through (b) becomes a decision (section 9).

## 5. Base + clone/reference vs emotion on the full-tuned custom voice

Owner finding (by ear): cloning from a full-FT custom_voice checkpoint works but sounds different; clone works best on an **unmodified Base**. Compare:

| | A. Full-tuned custom voice (EP09 class) + emotion adapter/steer | B. Stock Base + clone (x-vector or ICL) of the narrator + emotion adapter/steer | C. Hybrid: Base + narrator **LoRA** (identity) + emotion LoRA/steer |
|---|---|---|---|
| Identity | best fidelity to the trained narrator (owner's chosen EP09); emotion LoRA/steer may drift it | clone identity from a reference; owner: Base+clone sounds right for cloning, but it is a clone of the voice, not the fine-tuned voice (U how close to EP09; findings sections 5, 9.1 never listened) | narrator LoRA on Base: identity close, but LoRA-vs-full-FT narrator quality not yet compared (PEFT plan phases 5-6) |
| Instruct competence | probably weak (trained from Base with no instruct, I) | Base has no instruct (HF library: instruct needs CustomVoice for custom-voice mode; findings 6, 63) -> no instruct-based capture | same as B unless the emotion LoRA is instruct-conditioned (then it supplies it) |
| What the C engine expects | custom_voice + `spk_id` (section 1) | the C clone path is **Base-only**: `--ref-audio` errors on a non-Base model (`main.c:2141-2147`); `.bin`/graft are made on Base and loaded on CustomVoice (`docs/custom-voices.md`, `docs/voice-cloning.md:3-6`). Emotion steering and the `--emotion` router are documented/validated for **CustomVoice** only (`emotion-THE-recipe.md`: "1.7B CustomVoice only") | needs Base + `.expr`-style delta; untested |
| Steering vectors apply? | yes; apply code is model-agnostic (R); validity U | the hook math runs on Base too (I), but vectors are validated only on CustomVoice presets; on clones the author needed COMBINE and higher weights (R) -> expect weaker/unstable (I) | U |
| Extra cost | none (checkpoint exists) | none to train; clone quality depends on reference (we already cache x-vectors from stock Base, `main.py _xvector`) | needs the PEFT plan phases 5-7 |
| Python sidecar | works today | partly built: "clone on custom voice" experiment exists; a Base clone path would need Base loaded (second 4 GB model or a load swap) | merged checkpoint, one model |

Reading (I): B and C are the only ways to avoid the "different voice" the owner hears when cloning from the full-tune, but B ties the narrator to a reference clip and gives up instruct/steering validity; A keeps the narrator exactly and takes the risk on emotion strength. **Make the first experiment decide it**, one fixed-seed grid of 10 unseen sentences x 3 seeds: {A, B} x {no emotion, steer from own capture, emotion LoRA} scored by ear plus the speaker-similarity and emotion metrics in section 7. For the narrator specifically: keep EP09 as the identity reference, and judge any Base-based variant by how close it gets to it, not the reverse.

## 5b. `.qvoice`, instruct, and their best combination (R unless tagged)

### (1) `.qvoice` and `.bin`
| Item | Evidence |
|---|---|
| Container | `QVCE` magic, `u32 version` (current writer = 3; reader accepts 1-3, rejects others) `main.c:2763-2766, 2182-2189`; reader also in the list tool `main.c:59-119` |
| Fields in order | f32 speaker embedding (enc_dim: 2048 on 1.7B, 1024 on 0.6B), `ref_text` (u32 len + bytes), `n_ref_frames` + `int[n*16]` **ref codes** (ICL; zero in x-vector mode), `META` (lang id, 16-byte language, model hidden size, ref seconds, 64-byte voice name, flags: 1 = x-vector only, 2 = has ref codes, 4 = made on Base), optional `TPAD` (three source `tts_pad/bos/eos` embeds, f32), optional `WOVR` (source model's text_projection + first 2048 codec_embedding rows, bf16), optional `WDELTA` (deprecated multi-GB weight delta, `--target-cv`) `main.c:2767-2830` |
| Sizes | x-vector `.bin` 8 KB; graft `.qvoice` about 25 MB (1.7B) or 16 MB (0.6B); the old heavy file 0.8-2.9 GB (`docs/icl-graft-portability.md` section 1, `docs/custom-voices.md`). ICL ref codes would add about 24 KB per 30 s (same doc) |
| `.bin` | raw `f32[enc_dim]`, no header (`main.c:2999-3003`) |
| Made from | **Base model only**: `--ref-audio` on a non-Base model is an error (`main.c:2141-2147`); the speaker encoder lives in Base, and our custom_voice checkpoints dropped it. So: yes from our Base + a reference clip; **no** from the full-tuned custom_voice checkpoint by the engine. Our narrator's row 3000 is itself a 2048-float vector, so we could write a `.bin` from it ourselves (I; whether it behaves as an x-vector in their loader is U, because the baked row was trained) |
| Loaded by | C engine: a `.bin`/graft is loaded onto CustomVoice with `--load-voice` + `--xvector-only` / `--icl-only` (cross-model load message `main.c:2149-2153`). A graft carries `WOVR`, so it overrides the target's text projection and codec embedding with **the source Base's**: loading it on a fine-tuned checkpoint would replace the fine-tune's text projection/codec rows (I, from `main.c:2805-2821, 2469-2470`); prefer the 8 KB `.bin` for our checkpoint |
| Our Python stack | No code reads `.qvoice` today. The format is flat and documented, so a ~40-line reader for embedding + META is easy (I); `main.py _xvector` already makes and caches the same kind of vector with the stock Base encoder, but the C engine uses its own C port of the encoder (`qwen_tts_voice_clone.c`), so vectors from the two stacks are close but not byte-equal (I). Do not depend on cross-loading |

### (2) Instruct
| Item | Evidence |
|---|---|
| Mechanism | `--instruct/-I` text is wrapped as `<|im_start|>user\n{text}<|im_end|>\n`, tokenised, and prepended in the prefill before the role/text tokens (`qwen_tts.c:1087-1096, 1262`). Same layout as our HF path and as `dataset.py:98-102` for training |
| Sizes | 1.7B only; 0.6B warns "ignored" (`main.c:3011-3013`; `--emotion` also gated by `hidden_size >= 2048`, `main.c:1822`) |
| Custom voice (CustomVoice) | supported, the intended path |
| Base + clone | allowed but "not officially supported"; the tool's own advice is to extract the voice on Base and use it on CustomVoice (`main.c:3020-3025`) |
| Does our fine-tune keep instruct? | **Repo says nothing about fine-tunes.** From our side: on EP09 instructions changed length and pushed the voice away from the narrator (`../engines/qwen3-finetune-findings.md` section 6, verified by listening). Inference: full FT on no-instruct data with a language-neutral prefix, from **Base** (which has no instruct training per the CustomVoice weight-diff claim, `docs/expressivity-lora.md` section 2), gives an instruct that is partly responded to but not controlled as on stock CustomVoice. Two unknowns remain: how much of CustomVoice's instruct competence Base already has, and how much our 8-10 epochs removed. Test: same text, 6 emotion instructs, stock CV vs Base vs EP09, arousal/valence deltas (section 7) |

### (3) What their docs say works best
| Setting | Best combination (R) |
|---|---|
| Stock preset | **pure steer**: `ryan_<emo>.qlsteer`, layers 21-25, weight 12, decay 0.985, no expr, no instruct, seed 42, T1.1 (`docs/emotion-THE-recipe.md` "The recipe") |
| Cloned voice | **COMBINE**: language `.expr` at 1.0 + steer w12 + a default English instruct; x-vector clone at T1.3, weight 1.6-2.0 (`emotion-THE-recipe.md`, `docs/csp-ft-emotion.md:33`) |
| Instruct alone | secondary flavour on top; vivid prose, `strong` default, `very-strong` for anger; speed and pitch hints work mildly; slot templates do not (`docs/emotion-instruct-control.md`) |
| Graft / x-vector on CustomVoice | graft `--icl-only` is their recommended clone because it keeps CustomVoice weights so instruct, `.expr` and steer all work; the heavy Base-weight swap is called "emotion-dead" (`docs/custom-voices.md`) |
| 0.6B | no steer/instruct; emotion is a property of the reference audio (`docs/emotion-06b-recipe.md`) |

Implication for us (I): their best case is always **CustomVoice weights + a voice injected as a vector + emotion levers**, never Base weights. That matches the owner's observation about Base clones only for identity, not for emotion. For the narrator the nearest equivalents are: (i) EP09 + own-capture steer (+ instruct if it responds), or (ii) EP09-class weights + an instruct-conditioned emotion LoRA that restores instruct control, which is the plan in section 3.

## 6. Phased plan (nothing needs a GPU until phase 3)

| Phase | Work | Tests (CPU, fail before / pass after) | Needs GPU |
|---|---|---|---|
| 0 | Owner decisions (section 9); owner-run C engine load check (section 4) | none | optional |
| 1 | `qwen3_steer.py` (pure Python): `.qlsteer` reader/writer (magic `QLST` = 0x54534C51, L=29, D=2048, f32), `build_steer(neutral_mean, emotion_mean, clean, unit)` identical to `act_map_steer.py`, `apply_steer(x, v, w)` with norm restore, decay schedule, layer range 21-25, dyad mix | round-trip with their format; golden numbers computed from `act_map_steer.py` on random arrays; norm preserved; w=0 is identity; decay math equals `0.985**frame` | no |
| 2 | Capture script for a checkpoint (HF hooks, mean over decode frames only, instruct pairs list, N pairs averaged), runnable on Colab and the 3070 | hook index offset test with a tiny random Qwen3 config (layer i output = `hidden_states[i+1]`); mean excludes prefill | smoke on 3070 |
| 3 | Sidecar: static-buffer hook installed before graph capture; request fields `emotion`, `emotion_weight`; off by default; CPU tests with a fake graph object; check no recapture needed when changing weight | weight 0 = bit-identical output to no hook (GPU); changing emotion between requests does not re-capture; VRAM delta < 10 MiB | yes (3070) |
| 4 | Emotion dataset (section 8) + runner variant: instruct-conditioned dataset, speaker vector from checkpoint row or cached embedding, replay mix, no code-predictor targets; patch must apply once to the pinned `sft_12hz.py` | patch applies/refuses like ALIGN tests; dataset builder: instruct tokens masked, labels only on codec; replay fraction honoured | no |
| 5 | Short Colab smoke (2 epochs, owner go): read loss, `MERGE CHECK` changed %, listen | | A100 |
| 6 | Fixed-seed sweep: lr {2e-6, 1e-5, 2e-5} x merge scale {0.3, 0.6, 1.0}, steer weight {4, 8, 12}; metrics below | metric scripts have unit tests on synthetic audio | A100 + 3070 |
| 7 | Only if the gate passes: `.expr` exporter from our adapter (talker-only), docs/notice for the C engine, macOS decision | exporter vs `export_expr.py` byte layout test | no |

Docs-sync: update `qwen3-narrator.md` (settings, "Alternative implementation idea"), runtime guide section 6c, CHANGELOG `[Unreleased]` with each shipped phase. Locale parity: any new UI string through i18n.

## 7. Evaluation

| Axis | Method | Gate |
|---|---|---|
| Identity | cosine similarity of speaker embeddings (stock Base encoder, CPU) between each render and EP09 neutral renders of the same text; plus by-ear A/B against EP09 | similarity not below EP09's own seed-to-seed spread (measure it first) |
| Emotion | by ear (owner) first; `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` arousal/valence/dominance deltas vs neutral as a monotonic sanity check (their `emo_score.py` does this; the model's licence must be checked before we depend on it, U); an emotion classifier trained on acted data may mislabel TTS | arousal/valence moves in the expected direction for >= 5 of 6 emotions, ear agrees |
| Intelligibility | Whisper (or the repo's ASR) WER on the render | not above narrator baseline + a margin to be set |
| Stability | "did not stop by itself" count, length vs `max_tokens_for` | not above baseline |
| Protocol | 10-20 sentences **not** in training, 3 seeds each (as in TRAINING_STATE next step 3), same seeds for every arm, order randomised for listening | |

## 8. Emotion data

| Source | Size | Language, licence (from their `DATASETS.md` unless marked) | Use |
|---|---|---|---|
| EMOVO | 588 clips, 6 actors | Italian | their pack only; not for an English narrator |
| RAVDESS (`confit/ravdess-parquet`) | 2880, 24 actors, 2 fixed sentences | English, **NC-SA** (my recall, U) | too little text variety |
| CREMA-D | 7442, 91 actors, 12 fixed sentences, 16 kHz | English, licence U | variety of speakers, not text |
| ESD | 10 EN speakers, 5 emotions | English, research licence (U) | cleanest acted English |
| Expresso | studio 48 kHz | English, **CC BY-NC** | style clips, not full emotion sets |
| LibriTTS-R | large, 24 kHz | English, **CC BY 4.0**, no emotion labels | neutral replay text, auto-label pitch |
| **Synthetic from stock CustomVoice 1.7B** | unlimited | Qwen weights licence (verify, U); no acted-dataset licence issue | generate (text, instruct, codes) over varied narrator-style text with the 9 stock voices x 6 emotions x 2 strengths. Self-distillation transfers CustomVoice's instruct->emotion behaviour into the Base-derived narrator; quality ceiling is the stock model's |
| Narrator's own clips | the existing 1378 | owner's | replay, empty instruct |

Counts: their evidence is hundreds of clips per language for a LoRA (r16-r64), thousands for richer expressivity; r128 overfit 588. Plan: 300-500 clips per emotion per strength tier is the first target for a 6-emotion set (U), with at least 20 text templates per emotion that do **not** repeat across emotions. Fixed-sentence corpora alone will teach the sentences. Licence rule (CLAUDE.md local-first): no dataset download without the owner's yes; training corpora never ship in the repo; adapters trained on NC data may carry NC terms (U).

## 9. Decisions needed from the owner

1. Is the C engine worth a one-off load check in WSL2/Linux/macOS (section 4 gate), or skip it entirely?
2. Which arm first: A (full-tune + emotion), B (Base + clone + emotion), or the grid in section 5? Is "narrator = EP09 fixed" the reference?
3. Emotion data: synthetic from stock CustomVoice (needs the ~4 GB 1.7B CustomVoice download, findings section 4) vs acted corpora (licences) vs both.
4. lr and merge scale placeholders (`LORA_DEFAULT_LR`, `LORA_DEFAULT_SCALE`) and permission for the Colab smoke (phase 5).
5. If steering ships: opt-in setting name and default off (parity rule), plus whether the narrator remains CUDA-only.

## 10. Risks

| Risk | Mitigation |
|---|---|
| Base-derived narrator ignores instruct, so captures are meaningless | capture on stock CV and test transfer; or instruct-condition the LoRA (phase 4) |
| Steering weakens or drifts identity | own-capture vectors, weight sweep, identity gate |
| Emotion LoRA erases the narrator | replay mix, frozen row/heads, scale 0.3 start, similarity gate |
| bf16 merge rounding erases a small delta | existing `MERGE CHECK` (changed %) |
| Hooks inside the CUDA graph misbehave | check weight 0 bit-identical, no recapture, VRAM flat |
| Their numbers are one author's ear at seed 42 | our fixed-seed multi-arm protocol |
| C engine route splits maintenance | gate first, no commitment |

## 11. Only verifiable on a GPU

Static-buffer hook inside the captured graph (weight changes between replays); hook layer-index offset against real hidden states; steering vector strength on EP09 and on stock CV; instruct response of the Base-derived narrator; adapter gradients under the instruct-conditioned dataset; LoRA lr/scale/identity trade-off; merged vs unmerged logits; C engine load, bf16 vs `--quant-mixed` fidelity, EOS behaviour and steered-decode speed on the RTX 3070.
