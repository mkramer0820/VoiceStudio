# Qwen3-TTS fine-tune: what we found (research notes)

Written 2026-10-25 for follow-up research. Every claim is tagged: **Verified** (read from files or code, or measured here), **Heard** (the owner's listening notes, a judgement by ear), **Inferred** (a reasoned guess, not tested) or **Unknown**. Training history and Colab traps are in [qwen3-tts-colab-runtime.md](qwen3-tts-colab-runtime.md) and [../agents/QWEN3_TRAINING_STATE.md](../agents/QWEN3_TRAINING_STATE.md); running it locally is in [qwen3-narrator.md](qwen3-narrator.md).

## 1. What the checkpoint is

| | |
|---|---|
| Folder | `E:\Qwen3TTS_Checkpoints\SS_Narrator\Run_2_EP09\checkpoints\checkpoint-epoch-9` (Drive run `20261025-133200`, epoch file 9, the 10th and last epoch). Also kept: `Run_1_EP07` (the 105-minute run, epoch file 7) |
| Type, size | `tts_model_type: custom_voice`, `tts_model_size: 1b7`, architecture `Qwen3TTSForConditionalGeneration`, 12 Hz codec tokenizer. **Verified** |
| Speaker | one speaker, `ss_narrator`, codec-embedding row **3000** (`talker_config.spk_id`). The training name was `SS_Narrator`; Qwen looks speakers up in lowercase. **Verified** |
| Weights | `model.safetensors`, 404 tensors, 3656 MiB, plus a separate `speech_tokenizer/` (651 MiB). **Verified** |
| Talker | hidden 2048, 28 layers, 16 codec groups, codec vocab 3072, codec end-of-speech token 2150. **Verified** |

## 2. How it was trained (from `config/settings.json`, `metrics.csv`, `status.json`)

Base model `Qwen/Qwen3-TTS-12Hz-1.7B-Base`, tokenizer `Qwen/Qwen3-TTS-Tokenizer-12Hz`, 10 epochs, learning rate 1e-6, batch 4, speaker name `SS_Narrator`, all checkpoints kept, FlashAttention 2 on an A100. `status.json` ends at step 340; loss 5.20 at step 0 and 1.74 at the last status update (single-step values; `metrics.csv` also logs a 10-step average). Sample sentence for the per-epoch samples: "The lighthouse keeper counted seven ships before the fog rolled in." (seed 1234). **Verified**

## 3. How the weights relate to Base

Compared the checkpoint's tensors with the stock `Qwen3-TTS-12Hz-1.7B-Base` (480 tensors):

- All 404 EP09 tensors exist in Base with **identical shapes**; none are new. **Verified**
- Base has **76 extra tensors, all `speaker_encoder.*`**; EP09 has none. **Verified**
- EP09's `config.json` still contains a `speaker_encoder_config` block, but the library builds the encoder **only when `tts_model_type == "base"`** (`qwen_tts/core/models/modeling_qwen3_tts.py`, in `__init__`). **Verified**
- So the fine-tune is the Base model with the talker (and related) weights updated, the speaker encoder dropped, and the type switched to `custom_voice` with one new speaker row. That last sentence is **Inferred** from the three facts above plus the official fine-tuning script's behaviour; the script itself was not re-read here.

## 4. What the libraries allow

- `generate_custom_voice` works only on `custom_voice` models, `generate_voice_clone` and `create_voice_clone_prompt` only on `base`, `generate_voice_design` only on `voice_design` (type checks in `qwen_tts/inference/qwen3_tts_model.py`). **Verified**
- In the custom-voice path the speaker is **one row of the codec embedding table**, looked up by `spk_id`, and inserted into the prompt (`faster_qwen3_tts/model.py`, `_build_talker_inputs_local`). **Verified**
- Clone modes on Base: **ICL** (default) feeds the reference clip's codec tokens plus its transcript as context; **`xvec_only`** feeds only the speaker embedding. The library's own description: x-vector-only "prevents the model from continuing the reference audio's last phoneme". **Verified**
- `instruct` is dropped on the 0.6B model; the library marks instruct with x-vector-only cloning as experimental. **Verified**
- Static KV cache: prompt tokens plus generated frames must fit `max_seq_len`; a clone's reference clip adds its own frames to the prompt. **Verified**

## 5. Experiments and results

All on the owner's RTX 3070, seed 7 unless noted, files in the session scratchpad (temporary; copy anything worth keeping). Types were printed by the script where marked.

| Test | What | Result |
|---|---|---|
| Hybrid | EP09's 404 tensors overlaid on stock Base (type `base`, encoder from Base) | Loads and generates, no errors. **Heard:** cloning the Shadow voice, it "started in a random voice then moved to Shadow". **Dropped.** |
| Narrator reference, 4 files | stock Base and hybrid, each ICL and x-vector only, reference = a narrator training clip | **Heard:** all four sounded OK, but stock-Base x-vector-only had a very different tempo, more "robotic or off" |
| Noctis reference | stock Base, hybrid and EP09's own voice on a Noctis line | **Heard:** stock Base sounded like "narrator and Noctis". **Unknown** why (Inferred: the Fish-rendered Noctis may already resemble the narrator) |
| Shadow reference, same line as the reference text | stock Base, hybrid, and EP09 custom_voice `ss_narrator` | **Heard:** stock Base clone "sounded good"; EP09 narrator "sounds great"; hybrid as above |
| Instruct on EP09 custom_voice (type printed `custom_voice`) | `ss_narrator` with six instructions | see section 6 |
| Narrator on Base | stock Base and hybrid with a narrator training clip as reference | **Heard:** both sound like the narrator, because the narrator was the reference (cloning, not the fine-tune) |

Caveat for the Shadow test: the reference text and the target text were the same line, which can make a clone sound closer than it would on new text. **Verified** (it was designed that way at the owner's request).

## 6. Instruct on the fine-tuned narrator

Instructions tried, same sentence: none, "Speak slowly and calmly, in a soft, steady voice.", "Speak in an angry, harsh tone with a raised voice.", "Whisper quietly, as if sharing a secret.", "Speak cheerfully, with a bright, energetic tone.", "Speak in a tired, sad, weary tone, with low energy."

**Heard:** none = the narrator (good, correct parody and timbre). slow_calm = a higher-pitched narrator, voice combo. angry and cheerful = "sounds like nobody I know". tired_sad and whisper = "a very nice character voice, the narrator but not the narrator". Lengths changed with the instruction (11.3 s to 14.7 s for the same text), so the model did respond. **Verified**

Reading: **Inferred**. The fine-tune saw no instruction text, so an instruction changes what the model is conditioned on and moves the voice away from the narrator. This acts somewhat like voice design from a base character. The narrator itself should always be rendered with no instruction. Whether a fine-tune on instruction-labelled data would make it controllable is **Unknown**.

## 7. Measurements

RTX 3070 8 GB, `max_seq_len` 1280, bf16, SDPA, CUDA graphs (see [qwen3-narrator.md](qwen3-narrator.md)): 4158 MiB allocated and 4240 MiB reserved, flat over 24 calls; peak 4.65 GiB allocated, 5.10 GiB reserved; first call 10.6 s, then 2.25x real time. **Verified**

## 8. External idea: the C implementation

[gabriele-mastrapasqua/qwen3-tts](https://github.com/gabriele-mastrapasqua/qwen3-tts) (MIT) runs CustomVoice models with a precomputed x-vector or a small "graft" `.qvoice` for the clone, keeping instruct and emotion controls working, and has an optional CUDA backend. It is the closest published precedent for "clone on a custom_voice model". Its claims are unverified here, and whether it loads a fine-tuned checkpoint is **Unknown**. Details in [qwen3-narrator.md](qwen3-narrator.md#alternative-implementation-idea-qwen3-tts-in-c).

## 9. Open questions worth researching

1. **Clone on the fine-tuned custom_voice model.** Make the x-vector with stock Base, write it into the `ss_narrator` row for one generation (restoring the row afterwards), and compare with the narrator and with stock Base cloning. Built as the playground's "clone on custom voice" kind; **not yet listened to**.
2. Does a clone on the narrator model keep **instruct** working better than the narrator voice itself does?
3. **Run 2 epoch 9 against `Run_1_EP07`** on text that is not in the training data, three seeds each, including pacing (the loss log does not explain the dip in epochs 4 to 7).
4. **Why the stock-Base x-vector-only mode has a different tempo.** Try a fixed pacing prompt or ICL with a short reference.
5. **Narrator-on-Base without a reference:** is there any way to get the fine-tuned timbre on a Base model other than cloning a narrator clip? The hybrid suggests the weights alone are not enough.
6. Speech pace values (characters per second) behind the end-of-speech push were the script's fallbacks, not derived from the dataset; 4 of 24 test lines were pushed to stop.
7. A fine-tune on **several speakers** or with instruction text, so characters come from one custom_voice checkpoint.

## 10. Files

Test scripts used for the numbers above (session scratchpad, temporary): `fq_measure.py`, `clone_test.py` to `clone_test4.py`, the hybrid builder inline, and the WAVs `t1` to `t4_*`. The reproducible pieces live in the repo: `backend/engines/qwen3_narrator/` (sidecar and engine), `backend/engines/qwen3_studio/` (registry and playground), `tests/test_qwen3_narrator.py`.
