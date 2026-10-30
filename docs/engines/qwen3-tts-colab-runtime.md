# Qwen3-TTS on Google Colab: the runtime, the FlashAttention wheel, and every trap we hit

Written from the first real training runs (an A100 40 GB). Read this before setting up a Colab runtime for any training phase,
before changing Python or torch, and before touching the FlashAttention settings. "Verified" means it ran on a real Colab; "not verified"
means it was built and tested with fakes only. The code lives in `backend/engines/fish_api/qwen3_colab_runner.py` (the runner) and
`qwen3_colab.py` (the notebook generator); the training story is in [../agents/QWEN3_TRAINING_STATE.md](../agents/QWEN3_TRAINING_STATE.md).

## 1. The runtime that worked (verified)

| | |
|---|---|
| GPU | NVIDIA A100-SXM4-40GB, compute capability 8.0 (Ampere) |
| Python | 3.13.16 (`cp313`) |
| torch | 2.11.0+cu130, CUDA 13.0, C++ ABI `cxx11abi` TRUE |
| torchaudio | 2.11.0+cu130 |
| Driver | 580.82.07 |
| transformers / accelerate / huggingface-hub | 4.57.3 / 1.12.0 / 0.36.2 (installed by Qwen's package; do not upgrade) |
| FlashAttention | 2.8.3 community wheel `flash_attn-2.8.3+cu130torch2.11-cp313-cp313-linux_x86_64.whl` (section 2) |
| Local disk (`/content`) | about 180 GB free; wiped when the runtime is deleted |
| Cost | A100 40 GB about 5.4 compute units per hour (the owner's figure, not verified) |

Colab changes its default Python and torch without notice. Before anything else on a new runtime, print what you got:

```python
import sys, torch
print(sys.version.split()[0], '| torch', torch.__version__, '| CUDA', torch.version.cuda, '| cxx11abi', torch.compiled_with_cxx11_abi())
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
!nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
```

The notebook's setup cell prints the same facts in its "Runtime check" table in a few seconds, before it installs or downloads anything.

## 2. FlashAttention: the wheel must match exactly

A FlashAttention wheel loads only against the **exact** Python, torch and CUDA it was built for. The file name tells you:

`flash_attn-2.8.3+cu130torch2.11-cp313-cp313-linux_x86_64.whl` = FlashAttention 2.8.3, CUDA 13.0, torch 2.11, Python 3.13, Linux x86_64.

- **Official wheels** (Dao-AILab release v2.8.3.post2): Linux x86_64, **CUDA 12 only**, torch 2.4 to 2.8, `cp313` only for torch 2.5 and newer. **None exist for CUDA 13.**
- **Community wheels:** [mjun0812/flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels) builds CUDA 13 and newer torch for `cp313`.
  The one verified on Colab (A100, Python 3.13, torch 2.11, CUDA 13.0) is pinned as `qwen3_colab_runner.COMMUNITY_PINNED_WHEEL` (also `qwen3_colab.COLAB_FLASH_WHEEL`, pre-filled into every notebook):
  `https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.4/flash_attn-2.8.3%2Bcu130torch2.11-cp313-cp313-linux_x86_64.whl`
  (the `+` is `%2B` in the URL; the runner decodes it, pip needs the real name).
- **The runner finds the right wheel itself (runner 20).** Setup asks the GitHub releases API of that repo (no token, up to 5 pages, once per runtime) for the Linux x86_64 wheel whose Python, torch major.minor and CUDA match the runtime exactly (`cp313t` free-threaded builds never match; the newest FlashAttention version and release win, tags compared as numbers). If the lookup fails (offline, rate limit) it is not an error: the pinned wheel is used when its file name fits this runtime, otherwise nothing. The runtime check row says which one it found. It is **not verified on Colab** that a looked-up wheel other than the pinned one installs and passes the GPU test; the pinned one is.
- **Find a file by hand** with `python scripts/find_flash_wheel.py` (same lookup as the runner; reads the runtime's Python, torch and CUDA, or `--python cp313 --torch 2.11 --cuda cu130`) and put it in `FLASH_WHEEL`.
- **Never compile it on Colab.** An early runner built FlashAttention from source automatically because no wheel existed for CUDA 13: it ran for about an hour and wasted compute. The runner compiles only with `BUILD_FLASH_ATTN = True` (capped at 90 minutes), and the runtime check says up front whether a package exists.
- **How an install is decided**, in this order, stopping at the first that installs *and* passes a real GPU forward pass (`attention: flash_attn PASS (ran on the GPU)`): already working > the `FLASH_WHEEL` you set (name-checked first; a wheel that does not fit is skipped with the reason, never installed) > a wheel cached on Drive for this exact runtime and GPU architecture > the community wheel (looked up, else pinned) > the official CUDA 12 wheel > no wheel (below). FlashAttention is never installed on a GPU that cannot use it.

### Decision table

| Runtime and GPU | What happens | Training result |
|---|---|---|
| Ampere / Ada / Hopper (A100, L4, H100) and an exact-match wheel exists (community or official) | The wheel is installed and tested; the patched copy of `sft_12hz.py` runs with FlashAttention 2 | The reference path |
| Same GPU, **no** wheel for this Python / torch / CUDA | The patched copy trains with PyTorch SDPA attention. Nothing is compiled. **This is the supported fallback**: FlashAttention changes speed and memory, not what is learned | Same training quality; slower and more memory (not measured here) |
| Blackwell (G4, RTX 50xx) | SDPA by default; FlashAttention 2 does not support it (`ATTN_FALLBACK = False` forces it and will likely fail) | Same quality, SDPA speed |
| Turing (T4) | Refused in `auto`; `ATTN_FALLBACK = True` trains with SDPA (may run out of memory on a 15 GB card) | Same quality if it fits |
| No wheel, and you want FlashAttention anyway | `USE_VENV = 'auto'` builds an isolated Python 3.12 + torch 2.8 (cu126) environment with the official wheel. **Last resort**: minutes of setup, not verified on Colab. `BUILD_FLASH_ATTN = True` compiles instead (an hour or more, may fail) | Same quality |

The SDPA copy has not been verified end to end on Colab. When no wheel exists, prefer SDPA over the isolated environment unless speed matters; an older runtime version (Runtime > Change runtime type) also works.

## 3. Setting up a fresh runtime (the order that worked)

1. Runtime > Change runtime type > **A100**. Do not `pip install` or upgrade torch.
2. Open the notebook, fill the parameters cell (the export writes it), run the cells top to bottom: mount Drive, setup, check the data, train.
3. Setup, in order: runtime check (seconds) > Qwen repo > torch companion check > pip packages > FlashAttention wheel > models > dependency check (`torch convolution (cuDNN)`, `attention: flash_attn`, `qwen_tts` must all say PASS).
4. Setup takes about 7 minutes on a new runtime (the log: repo copy 65 s, pip 156 s, models 150 s, check 36 s, FlashAttention 22 s). `FAST_SETUP = True` (default) clones the repo and downloads the models straight into `/content` instead of copying them back from Drive, which was slow and erratic (the same 3.9 GB copy took 9 s once and 131 s another time).
5. Keep on Drive: the dataset zip, the prepared audio codes, checkpoints, logs. Everything under `/content` is scratch.
6. When finished: Runtime > Disconnect and delete runtime (an idle runtime keeps using units).

Drive layout (`Qwen3TTS` is the folder name; the old default `VoiceStudio-training` was removed everywhere): `datasets/`, `prepared/<speaker>/`, `runs/<speaker>/<run id>/{checkpoints,samples,logs}`, `logs/`, `cache/`.

## 4. The traps we hit, and what fixed them

| What you see | Cause | Fix / rule |
|---|---|---|
| Setup sits for an hour, then fails | FlashAttention compiled from source (no CUDA 13 wheel) | Section 2. Never compile. Runtime check first |
| `operator torchvision::nms does not exist` on `import qwen_tts` | torch was upgraded in place; torchvision/torchaudio stayed old | **Never upgrade torch in the runtime.** The setup repairs mismatched companions |
| `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH` in the audio-code step | torch upgraded next to Colab's older cuDNN | Same rule. Setup now runs a real GPU convolution and says so |
| `ValueError: Logging with tensorboard requires a logging_dir` | accelerate 1.12 vs Qwen's `sft_12hz.py` | The runner trains from a patched copy of the script (adds `project_dir`) |
| Noise from every epoch, loss still falls | lr 2e-5 (script default) and bugs in the training loop (text projection missing, labels shifted twice) | lr 1e-6 to 2e-6 and the patched script. Details: QWEN3_TRAINING_STATE.md |
| `NotImplementedError: Speaker X not implemented` | Qwen looks the speaker name up in lowercase; the script stores it as typed | The runner trains with a lowercase name |
| `FileNotFoundError ... does not exist, so nothing was created` / two `Qwen3TTS` folders on Drive | Colab could not see the Drive folder yet and the code created a second one | The notebook and runner never create the Drive root. Create the folder in Drive first and wait for it to sync |
| The stop button left training running | the child process was not in the cell's process group | The runner starts it in its own group and kills the group on interrupt |
| `ModuleNotFoundError: qwen_tts` in a notebook cell | the notebook's Python started before the package was installed | `import sys; sys.path.insert(0, '/content/Qwen3-TTS')` |
| Only 16 GB GPU memory free | earlier test cells still hold models in the notebook | Delete the variables, or Runtime > Restart session (keeps `/content` and the Drive mount) |
| An export's notebook names the old Drive folder | the app's backend was not restarted after an update | The export now refuses to run on an outdated backend. Restart the app |
| Generation at 0.36x real time | Qwen's stock loop is launch-bound (upstream issues #89, #171) | `faster-qwen3-tts` (section 5) |

Harmless messages, ignore: `diffusers requires huggingface-hub<2.0,>=1.32` (Qwen pins an older hub); `sox: not found`; `torch_dtype is deprecated`; `You are attempting to use Flash Attention 2 without specifying a torch dtype`; `Could not find cuda drivers on your machine` from TensorFlow; `Setting pad_token_id to eos_token_id`; `Warning: flash-attn is not installed...` from the fast test (that path uses SDPA by design).

## 5. Generating test samples fast (faster-qwen3-tts)

- Qwen's stock generation measured **0.36x real time** on the A100. [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) records the decode step as a CUDA graph; one worker measured about 0.9x here (7 s for 6.2 s of audio), and it runs several checkpoints in parallel.
- It needs transformers 5 and the `qwen-tts-hf` fork, which **conflict with the training stack**, so the test script builds a separate uv environment (`/content/fq_venv`) with **its own copy of torch** (about 3 GB, one time per runtime) and checks `pip freeze` before and after.
- It can only use SDPA (a static KV cache for the graph); that costs nothing because it is launch-bound, not attention-bound.
- One constraint rule: pin `torch==2.11.0`, **never** `torch==2.11.0+cu130` (a local-version label cannot be found on PyPI; the plain version is satisfied by the installed build).
- Memory: about 5.7 GB per worker, so 5 to 6 fit on a 40 GB card. More workers are faster only until the GPU and the VM's CPU cores saturate; measure the per-line "seconds to generate".

## 6. Settings that worked

epochs 10 (the last epochs of the first good run were the best), batch 4 with `GRAD_ACCUM` 2 (8 clips per update, the same recipe as 2 x 4), lr 1e-6 for the good runs (the shipped default is 2e-6, being tried against it), `USE_VENV` auto, `FAST_SETUP` True, clips 4 to 20 s at 24 kHz mono with 1 s of silence at the end (the export adds it). Training speed at batch 2: 3.7 steps per second, about 1 min 46 s per epoch of 683 clips, 21 GB peak GPU memory. Batch 4 with `GRAD_ACCUM` 2 (1378 clips) ran on the same card at about 25 GB.

**One set of defaults.** `qwen3_colab_runner.DEFAULT_EPOCHS / DEFAULT_LR / DEFAULT_BATCH_SIZE / DEFAULT_GRAD_ACCUM / DEFAULT_SPEAKER` feed `qwen3_colab.TrainSettings`, the notebook's parameters cell and the runner's `Params`; the export dialog (`export-options.ts`, `QWEN_NUMBERS`) repeats the numbers and `tests/backend/engines/test_fish_qwen3_defaults.py` fails if any layer drifts. The notebook has one parameters cell, near the top.

## 6b. Validation loss, safe save, Drive only for epochs

- **Validation loss.** `VAL_FRACTION` (default 0.05, `0` turns it off) holds out that share of the clips with a fixed seed. After every epoch the patched script prints `Epoch N | Val loss: x` and the runner writes it to its own `val_loss` column of `metrics.csv` (the training rows leave it empty). While it falls, more epochs help; once it stops falling while the training loss keeps falling, they do not. Fewer than 20 clips: skipped, and it says so.
- **Safe save.** The checkpoint is written to `.tmp-checkpoint-epoch-N` and renamed to `checkpoint-epoch-N` only after it is verified, so a crash can no longer leave the untrained base weights under a finished-looking name (upstream issue #365). The check compares a sample of tensors with the base model; `SAVE VERIFIED` appears when it passes and `SAVE FAILED` (training stops) when none differ. It also prints `SAVE CHECK: x% of elements changed vs the base model`.
- **Drive only for epochs.** Models download straight to `/content` (`CACHE_MODELS_ON_DRIVE` defaults to `False`). `DRIVE_SKIP_FIRST` (default 3) keeps the first N epochs on `/content` and does not copy them to Drive; the final epoch is always copied.
- **The training copy of `sft_12hz.py`** (pinned commit 022e286) carries these fixes, none tied to a model size, and tests run the patched helpers on small tensors: the text goes through `text_projection` before the codec embeddings are added (the 0.6B model's text embedding is wider than its codec embedding, so without it that model crashes, and the 1.7B one would train on unprojected text); the talker gets unshifted labels and tensors (the loss shifts once); the sub-talker scores its 15 logits against codes 1 to 15 with plain cross entropy (no second shift). The mozi1924 EasyFinetuning script does not fix the double shift; ours follows the issue #179 analysis.
- **Not run on a GPU yet:** `_val_loss` and `_verify_saved` have run only against fake tensors and small files. The first real run should be a short one.
- **Possible reason low learning rates change little (inferred, not tested):** the model is loaded in bf16 and trained in bf16 without fp32 master weights, so an update of about 1e-6 on a weight of about 0.02 is smaller than bf16's resolution and rounds away for most weights. `SAVE CHECK` shows how much actually changes; if it is tiny at 1e-6 and larger at 2e-6, that is the reason.

## 6c. LoRA and multi-speaker training (runner 21; built and tested with fakes, NOT run on a GPU)

- **`TRAIN_MODE`** `'full'` (default, unchanged: the whole model, `LR` = `DEFAULT_LR` = 2e-6) or `'lora'` (adapters on a frozen model). LoRA has its **own** constants: `LORA_LR` (`qwen3_colab_runner.LORA_DEFAULT_LR`), `LORA_RANK` 16, `LORA_ALPHA` 32, `LORA_DROPOUT` 0.05, `LORA_SCALE` (`LORA_DEFAULT_SCALE`). The LoRA learning rate (2e-6) and the adapter scale (0.3, range 0.0-1.0, warn above 0.5) are owner-confirmed defaults (2026-10-29), not yet swept on our aligned loss. The scale is applied at generation or merge time, not in training. A dataset's `manifest.json` names its `training_mode`; the runner refuses the other mode. No sampling defaults (temperature, top_k, top_p) are set per mode.
- **What trains.** `peft==0.17.1` (installed with `--no-deps`, probed in the setup table as `peft (LoRA)`) injects adapters with `inject_adapter_in_model` into the Linear layers `q,k,v,o,gate,up,down` of `talker.model.layers.*` and `talker.code_predictor.model.layers.*` only (a full-match regex; the speaker encoder, `text_projection`, heads and embeddings stay frozen). Checked on peft 0.17.1 + transformers 4.57.3 (CPU): `inject_adapter_in_model` leaves adapters in the base's **bfloat16** (unlike `get_peft_model`), so the patched script casts them to float32 itself and asserts it. peft 0.17.1 does not import under transformers 5, so keep it out of the local sidecar environment.
- **The patch** is the same pinned `sft_12hz.py` (commit 022e286) with the ALIGN_A/B/C and VAL_* patches intact plus `LORA_PAIRS` (optimizer, speaker embeddings, save block); it refuses any script that does not contain each line exactly once. A full single-speaker run does not get these patches and is byte-for-byte runner 20's script.
- **Checkpoints.** Each epoch is still a normal merged `checkpoint-epoch-N` (so the save check, the Drive sync, the samples and the playground work unchanged) and also holds `adapter_model.safetensors` (float32), `adapter_config.json`, `speakers.json`, `speaker_embeddings.safetensors` and `lora_meta.json`. The merge adds `scale * B A` in float32 and rounds once to bfloat16; `MERGE CHECK` prints the share of adapted elements that changed and warns below `MERGE_MIN_CHANGED_PCT` (a placeholder). An all-zero adapter stops the run (`LORA FAILED`).
- **Several speakers in one model.** The export (`multi_speaker`) writes a `speaker` per clip and a reference clip per speaker; the runner reads them (or `SPEAKERS`), caches one fixed float32 embedding per speaker (mean of up to 8 reference clips, encoder on the GPU once, file named after the references so a rerun skips it) and writes speaker `i` into codec row `3000 + 20 * i`. The table has 3072 rows, so **at most 4 speakers fit** (the plan's scheme would reach row 3080 for a fifth). Names are lowercase and checked against the stock rows. Either mode can train several speakers.
- **Frame guard** (Qwen3-TTS issue #378): `batch x longest clip` in codec frames is kept under 8192 by lowering the batch and raising `GRAD_ACCUM`, with a warning. Not tested on a GPU; the eager code-predictor override from the issue is not built.
- **Merging after the fact** (CPU, no GPU): `qwen3_peft.merge_adapter(base_dir, adapter_dir, out_dir, lora_scale=...)`, or the app's *Merge adapter* route (`POST /api/qwen/models/merge-adapter`), which writes into a folder listed in Settings -> Storage -> Model locations.
- **Not verified on a GPU:** adapter gradients through `forward_sub_talker_finetune`, peft 0.17.1 with transformers 4.57.3 on Colab (probed on the CPU only), VRAM and speed, the learning rate, rank and merge scale, that attention/MLP adapters can learn the end-of-speech token, merged versus unmerged logits, the cached embedding versus the per-sample one, multi-speaker separation.

## 7. Not verified

LoRA and multi-speaker training (section 6c); a looked-up community wheel other than the pinned one; the SDPA fallback copy end to end on Colab; the isolated Python 3.12 environment (`USE_VENV`) on Colab; Blackwell (G4) and Turing (T4) fallbacks; other GPUs' batch sizes; interrupting a run on Colab; the validation loss and safe-save helpers on a GPU; the narrator's `lead_s` start trim as a cold-start remedy (it only shapes the start of the output; a silence-token warm-up prefix needs library internals and is not built). When one of these is run for real, change this file in the same commit.

## 8. When you change the setup

Update this file together with the code: a new pinned wheel (`COMMUNITY_PINNED_WHEEL` in the runner), a new runtime, a new trap. Run `python scripts/find_flash_wheel.py` to check which wheels exist for a new runtime.
