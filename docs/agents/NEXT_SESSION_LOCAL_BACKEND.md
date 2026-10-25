# Next session: run the fine-tuned narrator on the owner's RTX 3070 with faster-qwen3-tts (a local backend, no UI yet)

Written 2026-10-25. Part 1 (this session): prove that `faster-qwen3-tts` (CUDA graphs) runs the owner's fine-tuned checkpoint on their own PC and wrap it as a backend of the app. Part 2 (a later session): hook that backend up to the UI. **Do not touch the UI in Part 1.** The owner has used faster-qwen3-tts before, so it is known to be possible on this card; what is new is our fine-tuned checkpoint and the integration.

## Paste this into the new session

```
You are continuing work on the VoiceStudio repository (a local-first voice app), branch ext/fish-api.

TASK (Part 1 only): get the owner's fine-tuned Qwen3-TTS narrator voice running on THEIR PC (RTX 3070, 8 GB) through faster-qwen3-tts, measured, and wrapped as a local TTS backend. No UI work.

READ FIRST, in order, nothing else until a step needs it:
1. docs/agents/NEXT_SESSION_LOCAL_BACKEND.md (this plan, facts, acceptance, rules).
2. docs/engines/qwen3-tts-colab-runtime.md section 5 (how faster-qwen3-tts was installed and measured on Colab, the pins, the traps).
3. AGENTS.md (repo contract). Only when you reach the integration step: backend/services/tts_backend.py (class TTSBackend, the registries near the end) and backend/engines/voxcpm2_subprocess/ (the isolated-engine pattern), tests/test_engine_docs.py.

RULES: read the library source and its GitHub issues before advising settings. Give any script or command as text FIRST when the owner is waiting. Never touch G:\ (the owner's Google Drive) or install into the conda env "tts" or C:\python3. Commit with the owner's git identity only (no agent attribution). Do not push unless the owner says so. Ask before downloading several GB (torch is about 2.5 GB and is expected).

START BY looking at the real folder tree under E:\Qwen3TTS_Checkpoints\SS_Narrator\ (the owner renames folders; do not trust the doc), then follow the plan in the doc.
```

## Facts (checked 2026-10-25)

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3070, 8192 MiB, driver 610.88. **About 2.7 GB was already in use** by Windows and other programs, so only about 5.4 GB is free for the model. Close browsers and other GPU programs for measurements |
| **The app's own environment** | a `uv`-managed venv named **`.venv`** (Python pinned to 3.11 by `.python-version`, locked by `uv.lock`, torch from the `pytorch-cuda` index in `pyproject.toml`). From the repo it is `E:\VoiceStudio\.venv`; the installed desktop app uses `%APPDATA%\VoiceStudio\runtime\project\.venv`. **Checked 2026-10-25 in the repo's `.venv`:** Python 3.11.17, **torch 2.8.0+cu128 (CUDA 12.8), the 3070 is visible, bf16 works**, torchaudio 2.8.0+cu128, **transformers 5.15.1**, accelerate 1.14.0, huggingface_hub 1.28.0, onnxruntime 1.29.0. No `qwen_tts`, no `faster_qwen3_tts`, no `flash_attn`. **Do not install into it**: it is the app's locked environment. |
| The owner's `tts` conda env | Python 3.10.20, torch 2.6.0+cu124 (CUDA works, bf16 works), transformers **4.30.0 (too old)**, no `qwen_tts`, no `faster_qwen3_tts`, no `flash_attn`. **Do not install into it** |
| `C:\python3` | torch 2.13.0 **CPU-only**, and its torchaudio DLL does not load. Do not use it |
| Weights | on Drive: `G:\My Drive\Qwen3TTS\runs\SS_Narrator\20261025-035854\checkpoints\checkpoint-epoch-7` (the best of the 105-minute run) and `...\20261025-133200\checkpoints\checkpoint-epoch-0` to `-9` (the 3-hour run). **Local copies:** `E:\Qwen3TTS_Checkpoints\SS_Narrator\`, naming `Run_<n>_EP<NN>` (`Run_1_EP07` = the first), still being copied and renamed: check the real tree |
| Checkpoint | about 4.3 GB, self-contained: `model.safetensors` (3.8 GB), `config.json` (`tts_model_type` custom_voice, speaker `ss_narrator`), tokenizer files, the speech tokenizer |
| Speed to expect | Colab A100: about 0.9x real time per worker (stock Qwen loop: 0.36x). The library's own table lists an 8 GB RTX 4060 on Windows at 1.83x for the 1.7B model. **Not measured on the 3070** |
| Attention | The fast path uses PyTorch SDPA with a static cache, **not FlashAttention**, by design. Nothing to install for it. (Do not spend time on a Windows FlashAttention wheel for this job) |

## Plan and acceptance

1. **A fresh environment that mirrors the app's known-good versions,** outside the repo and not the app's `.venv`, not the conda env `tts`: Python 3.11, **torch 2.8.0+cu128 and torchaudio 2.8.0+cu128** (the exact build already running on this 3070 in the app's `.venv`), **transformers 5.15.1** (the version the app already uses and the one faster-qwen3-tts is pinned to), then `faster-qwen3-tts==0.5.4`. Because the CUDA and torch combination is the same one the app runs, nothing needs downgrading. The pins and traps are in `epoch_test_fast.py` (Colab) and the runtime guide section 5: keep the torch constraint **without** a `+cuXXX` label. *Done when:* `import torch, transformers, qwen_tts, faster_qwen3_tts` works and `torch.cuda.is_available()` is true.
2. **Load `Run_1_EP07`** with `FasterQwen3TTS.from_pretrained(<local checkpoint folder>, device='cuda', dtype=torch.bfloat16, attn_implementation='sdpa', max_seq_len=...)`. Record peak VRAM (`torch.cuda.max_memory_allocated` and `nvidia-smi`). If it does not fit in about 5.4 GB, lower `max_seq_len` first (Colab used 1280), then say plainly that it does not fit. *Done when:* a sentence is generated, or a measured out-of-memory result is reported.
3. **Generate the owner's test sentences** (the 24 in `epoch_test_fast.py`; start with the 5 short ones): `generate_custom_voice(text, speaker='ss_narrator', language='Auto', max_new_tokens=<limit from the text length>)`. Use the length limit and tail trim from `epoch_test_fast.py` (the end-of-speech push is applied there by wrapping the library's `sample_logits`). *Done when:* WAVs exist and the owner has listened. Compare with the Colab sample of the same epoch.
4. **Measure** seconds of audio per second of generation after warm-up (the first call records the CUDA graph), VRAM, and the first-audio delay. Report the numbers, not adjectives.
5. **Wrap it as a backend**, following the isolated-engine pattern: a `TTSBackend` subclass registered in `backend/services/tts_backend.py` (the id/class registry, the install-hint dict, the docs-page dict and the subprocess registry, see how `voxcpm2` appears in all four) with a worker under `backend/engines/<name>_subprocess/`, run by the separate environment's python so its transformers 5 never meets the app's. The checkpoint folder and the environment path must be settings, not hard-coded. `is_available()` must say why when something is missing. Add `docs/engines/<name>.md` (required by `tests/test_engine_docs.py`) and tests that run without a GPU. *Done when:* the backend generates audio from the app's backend code path in a script, with tests green.
6. **Stop before the UI.** Write what Part 2 needs into the state doc.

## Open questions for the owner

- Which epochs to keep locally for comparison (each is 4.3 GB), and whether the 3-hour run is better than `Run_1_EP07`.
- If the model does not fit in about 5.4 GB with the CUDA graph: accept a shorter `max_seq_len`, or fall back to the stock loop (slower, smaller)?

## After the session

Update `docs/agents/QWEN3_TRAINING_STATE.md` (item 00) and `docs/agents/SESSION_STATE.md` with what was measured, what failed, and what is next.
