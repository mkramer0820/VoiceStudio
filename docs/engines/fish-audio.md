# VoiceStudio — Fish Audio (cloud API)

Fish Audio's hosted TTS (`s2.1-pro`, `s2-pro`, and the new `drama-3`) as an
opt-in engine. **Text you synthesize is sent to Fish Audio.** The engine is
unavailable until you add a key, makes no network call while idle or when
checking availability, and the app works fully without it.

## Setup

1. Create an API key at fish.audio.
2. Set it before launching: PowerShell
   `[Environment]::SetEnvironmentVariable('FISH_API_KEY','<key>','User')`
   (or store it in the encrypted secret store as `fish_audio_key`).
3. Pick **Fish Audio** in **Model Catalogue**, or pin it with
   `OMNIVOICE_TTS_BACKEND=fish-audio`.

An account with no credits answers HTTP 402; VoiceStudio reports it as
"out of credits". `s2.1-pro-free` (a $0 tier without latency guarantees) can be
set as the model.

## Output quality (defaults suit editing)

| Setting | Env var | Default | Notes |
|---|---|---|---|
| Model | `FISH_TTS_MODEL` | `s2.1-pro` | `s2-pro`, `drama-3`; unknown names fall back to `s2.1-pro` at Fish |
| Format | `FISH_TTS_FORMAT` | `wav` | `wav`, `pcm` (16-bit mono), `mp3`, `opus` |
| Sample rate | `FISH_TTS_SAMPLE_RATE` | `44100` | wav / pcm / mp3 |
| MP3 bitrate | `FISH_TTS_MP3_BITRATE` | `192` | 64 / 128 / 192; mp3 only. WAV is lossless, so no bitrate applies |
| Latency | `FISH_TTS_LATENCY` | `normal` | `normal` is the most stable; `balanced` is faster |
| Normalize | `FISH_TTS_NORMALIZE` | `true` | expands numbers and dates |
| Temperature / top-p | `FISH_TTS_TEMPERATURE`, `FISH_TTS_TOP_P` | unset | |
| Chunk length | `FISH_TTS_CHUNK_LENGTH` | unset | 100–300 |
| Voice (`reference_id`) | `FISH_TTS_REFERENCE_ID` | unset | default saved voice; per-request `voice` overrides |
| Archive masters | `FISH_TTS_ARCHIVE_MASTERS` | `true` | see below |
| Concurrency cap | `FISH_TTS_MAX_CONCURRENCY` | `2` | 429 backs off and retries |
| Base URL | `FISH_API_BASE_URL` | `https://api.fish.audio` | must be https (http for loopback only) |

Each setting can also be stored under `tts.fish.<name>` in the settings store.
The environment wins.

### Masters for external editing

Every response is saved byte-for-byte to
`<data dir>/outputs/fish_masters/<yyyymmdd>/<HHMMSS>_<hash>.<ext>` **before**
VoiceStudio's own processing (provenance watermark, loudness, resampling). Edit
those files when you want Fish's exact output. The file your project exports
goes through the normal pipeline.

## Limits (phase 1)

- Saved voices by `reference_id` only; no inline cloning or voice upload yet.
- No voice design, no streaming.
- Licensing and pricing are Fish's; check their terms for commercial use.
