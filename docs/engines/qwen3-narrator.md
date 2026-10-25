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

faster-qwen3-tts is pinned to `transformers==5.15.1` and the `qwen-tts-hf` fork, which must not
share an interpreter with the app. The sidecar runs under that venv's Python; the app only
talks to it over stdio.

## Install

```powershell
uv venv --python 3.11 C:\path\to\fq_env\venv
echo "torch==2.8.0`ntorchaudio==2.8.0" > C:\path\to\fq_env\constraints.txt
uv pip install --python C:\path\to\fq_env\venv\Scripts\python.exe torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python C:\path\to\fq_env\venv\Scripts\python.exe -c C:\path\to\fq_env\constraints.txt faster-qwen3-tts==0.5.4 transformers==5.15.1
```

Keep the torch constraint **without** a `+cuXXX` label (a local-version label cannot be found
on PyPI). Do not install into the app's own `.venv`.

## Settings (environment variables)

| Variable | Meaning | Default |
|---|---|---|
| `OMNIVOICE_QWEN3_NARRATOR_PYTHON` | python of the venv above | required |
| `OMNIVOICE_QWEN3_NARRATOR_CKPT` | checkpoint folder (`config.json`, `model.safetensors`, tokenizer, `speech_tokenizer/`) | required |
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

## Troubleshooting

`is_available()` names the missing piece: unset or wrong python path, unset checkpoint, or a
checkpoint folder without `config.json` / `model.safetensors`. Out-of-memory at load: lower
`MAX_SEQ_LEN` (smaller cache, smaller chunks) and close other GPU programs.
