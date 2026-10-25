# VoiceStudio — Qwen3-TTS fine-tuned narrator (local GPU)

Runs a fine-tuned **Qwen3-TTS custom-voice** checkpoint (the narrator trained with the
Colab notebook, see [qwen3-tts-colab-runtime.md](qwen3-tts-colab-runtime.md)) on your own
NVIDIA GPU through [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts)
(CUDA graphs). Output is 24 kHz mono.

> **Opt-in, and never a default.** Select it in **Model Catalogue** or with
> `OMNIVOICE_TTS_BACKEND=qwen3-narrator`. It needs your own checkpoint; nothing is downloaded.

## Platform support

NVIDIA GPU with CUDA only (`gpu_compat = ("cuda",)`). Measured on an RTX 3070 8 GB, Windows.
macOS, Linux without CUDA and CPU are not supported by this engine.

## Why it has its own venv

faster-qwen3-tts pins `qwen-tts-hf`, which wants `accelerate==1.12.0`; the app's lockfile has 1.14.0.
Installing it into the app's `.venv` would drift from `uv.lock`, so the engine gets its own venv inside
the project: **`.venv-qwen`** in the repo root (git-ignored). The sidecar runs under that venv's Python;
the app only talks to it over stdio. **It never uses `PATH`, a system or base Python, or the app's `.venv`.**
With the Python setting blank, the playground uses `.venv-qwen` if it exists; otherwise the engine reports
"not available".

## Install

```powershell
uv venv .venv-qwen --python 3.11
uv pip install --python .venv-qwen\Scripts\python.exe --index-strategy unsafe-best-match -r backend/engines/qwen3_narrator/requirements.txt
```

`backend/engines/qwen3_narrator/requirements.txt` is the exact package set the narrator was measured in
(torch 2.8.0+cu128, transformers 5.15.1, faster-qwen3-tts 0.5.4). Do not install into the app's own `.venv`.

## Settings (environment variables)

| Variable | Meaning | Default |
|---|---|---|
| `OMNIVOICE_QWEN3_NARRATOR_PYTHON` | python of the venv above | the playground's Python setting, else `.venv-qwen` |
| `OMNIVOICE_QWEN3_NARRATOR_CKPT` | (engine only; the playground finds checkpoints in **Settings -> Storage -> Model locations**) checkpoint folder (`config.json`, `model.safetensors`, tokenizer, `speech_tokenizer/`) | required |
| `OMNIVOICE_QWEN3_NARRATOR_MAX_SEQ_LEN` | static KV cache length of the CUDA graph (prompt + audio frames) | 1280 |
| `OMNIVOICE_QWEN3_NARRATOR_SPEAKER` | speaker name | first speaker in `config.json` |
| `OMNIVOICE_QWEN3_NARRATOR_RECV_TIMEOUT_S` | sidecar reply timeout | 600 |

Per request: `chunk_chars` (override the chunk size), `pad_s` (silence after each chunk, default
0.3 s), `seed`, `language` (default `Auto`).

## How it behaves

- **One job at a time.** Submissions queue on a lock; the sidecar never batches. The chunks of
  a long text run one after another.
- **Chunking.** Text is split at sentence ends, then commas/dashes, then spaces, into chunks
  that fit the cache (at most 300 characters, about the 20 s clips the model was trained on).
  A smaller `MAX_SEQ_LEN` gives smaller chunks.
- **End of speech.** Each chunk's length is predicted from its text; past that length the codec
  end-of-speech logit is pushed up. A chunk is reported as *stopped by itself*, *pushed to
  stop* or *hit the limit*. The reply carries `complete: false` and the sidecar logs a warning
  when any chunk hit the limit.
- **Padding.** Silence after the last audible sample is replaced by exactly `pad_s` of silence,
  so endings are not clipped and chunks join evenly.
- **GPU memory.** The cache and graph are allocated once at load; allocated memory stayed
  constant over 24 consecutive requests, and the allocator cache is flushed after every chunk.
  The app's idle reaper stops the sidecar, which releases the GPU completely.

## Measured (RTX 3070 8 GB, Run 2 epoch 09, `MAX_SEQ_LEN` 1280, 2026-10-25)

Free VRAM before load 6256 MiB. Peak 4.65 GiB allocated / 5.10 GiB reserved (the longest
clip, 18.6 s); steady 4158 MiB allocated / 4240 MiB reserved. First call 10.6 s (graph
capture), then 2.25x real time (183.6 s of audio in 81.7 s over 24 lines). Close other GPU
programs before measuring: browsers and video players hold 1–3 GB.

What was found about the fine-tuned model itself (weights, model types, instruct, clone experiments, open questions): [qwen3-finetune-findings.md](qwen3-finetune-findings.md).

## Troubleshooting

`is_available()` names the missing piece: unset or wrong python path, unset checkpoint, or a
checkpoint folder without `config.json` / `model.safetensors`. Out-of-memory at load: lower
`MAX_SEQ_LEN` (smaller cache, smaller chunks) and close other GPU programs.

## Alternative implementation idea: Qwen3-TTS in C

[gabriele-mastrapasqua/qwen3-tts](https://github.com/gabriele-mastrapasqua/qwen3-tts) (MIT) is a pure C inference engine for Qwen3-TTS: no Python or PyTorch, CPU by default, with optional Apple Metal and NVIDIA CUDA backends (`make cuda`). Read from its README on 2026-10-25; none of it has been run here.

- **Speed claim:** about 0.44 real-time factor for the 1.7B model on an RTX 4060-class GPU. Ours measured 2.25x real time (a factor of about 0.44) on an RTX 3070, so the two are comparable.
- **Clones as small files:** a reference clip is processed once into a `.qvoice` (a 25 MB "graft") or an 8 KB x-vector `.bin`, then reused on the **CustomVoice** model, where `--instruct` and emotion controls keep working. This is the same idea as the "clone on custom voice" experiment in the Qwen playground.
- **Caveats:** its CUDA server mode is marked work in progress and never qualified; Windows is supported through WSL2; it needs the model converted to its own format, and it is **not known** whether a fine-tuned checkpoint such as the narrator's loads in it.
- **Not adopted.** Revisit if the Python sidecar becomes a bottleneck (install size, startup time or memory), and test first that the fine-tuned narrator loads and sounds the same.
