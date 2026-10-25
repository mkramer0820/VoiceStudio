# Qwen3-TTS on Google Colab: the runtime, the FlashAttention wheel, and every trap we hit

Written 2026-10-25 from the first real training runs (an A100 40 GB). Read this before setting up a Colab runtime for any training phase,
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
- **Community wheels:** [mjun0812/flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels) builds CUDA 13 and newer torch. As of 2026-10-25 it has `cp313` Linux x86_64 FlashAttention 2.8.3 wheels for:

  | CUDA | torch | release tags |
  |---|---|---|
  | cu130 | 2.9 | v0.5.2, v0.6.8, v0.7.16, v0.9.0, v0.9.21 |
  | cu130 | 2.10 | v0.7.12, v0.7.16, v0.9.0, v0.9.21 |
  | **cu130** | **2.11** | **v0.9.4** (the one in use), v0.9.31 |
  | cu130 | 2.12 | v0.9.17 |
  | cu128 | 2.9, 2.10, 2.11 | several |

  The one in use:
  `https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.4/flash_attn-2.8.3%2Bcu130torch2.11-cp313-cp313-linux_x86_64.whl`
  (the `+` is `%2B` in the URL; the runner decodes it, pip needs the real name). It is `qwen3_colab.COLAB_FLASH_WHEEL`, pre-filled into every
  notebook the export writes.
- **Find the right file for any runtime** with `python scripts/find_flash_wheel.py` (reads the runtime's Python, torch and CUDA) or
  `--python cp313 --torch 2.11 --cuda cu130`. It ignores `cp313t` (the free-threaded build; it does not load on a normal Python 3.13).
- **Which GPUs can use it:** FlashAttention 2 supports Ampere (A100, RTX 30), Ada (L4, RTX 40) and Hopper (H100). **Not** Blackwell (G4, RTX 50xx) and **not** Turing (T4): those use PyTorch's built-in attention (SDPA) through a patched copy of the training script (not verified on Colab).
- **Never compile it on Colab.** An early runner built FlashAttention from source automatically because no wheel existed for CUDA 13: it ran for about an hour and wasted the owner's compute. The runner now only compiles with `BUILD_FLASH_ATTN = True` (capped at 90 minutes), and the runtime check says up front whether a package exists.
- **How the notebook uses it:** `FLASH_WHEEL = '<url or Drive path>'`. The runner reads the file name, compares it to the runtime (`wheel_fits`: Python, torch, CUDA), and only then installs it, then runs a real GPU forward pass (`attention: flash_attn PASS (ran on the GPU)`). A wheel that does not fit is skipped with the reason, never installed.
- **The check that counts:** the setup table row `attention: flash_attn  PASS  2.8.3 (ran on the GPU)`.

If the runtime changes and no wheel exists: (a) Runtime > Change runtime type > an older runtime version; (b) `USE_VENV = 'auto'` builds an isolated Python 3.12 + torch 2.8 (cu126) environment with the official wheel (**not verified on Colab**); (c) `ATTN_FALLBACK = True` trains with PyTorch attention (not verified).

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
| Noise from every epoch, loss still falls | lr 2e-5 (script default) and bugs in the training loop (text projection missing, labels shifted twice) | lr 1e-6 and the patched script. Details: QWEN3_TRAINING_STATE.md |
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

epochs 10 (the last epochs of the first good run were the best), batch 4 with `GRAD_ACCUM` 2 (8 clips per update, the same recipe as 2 x 4), lr 1e-6, `USE_VENV` auto, `FAST_SETUP` True, clips 4 to 20 s at 24 kHz mono with 1 s of silence at the end (the export adds it). Training speed at batch 2: 3.7 steps per second, about 1 min 46 s per epoch of 683 clips, 21 GB peak GPU memory. Batch 4 with `GRAD_ACCUM` 2 (1378 clips) ran on the same card at about 25 GB.

## 7. Not verified

The isolated Python 3.12 environment (`USE_VENV`) on Colab; Blackwell (G4) and Turing (T4) fallbacks; other GPUs' batch sizes; interrupting a run on Colab; how the 3-hour run's voice compares with the 105-minute run's. When one of these is run for real, change this file in the same commit.

## 8. When you change the setup

Update this file together with the code: a new wheel (`COLAB_FLASH_WHEEL` in `qwen3_colab.py`), a new runtime, a new trap. Re-run `python scripts/find_flash_wheel.py` to refresh the table in section 2.
