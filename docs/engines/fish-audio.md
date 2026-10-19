# VoiceStudio — Fish Audio (cloud API)

Fish Audio's hosted TTS (`s2.1-pro`, `s2-pro`, and the new `drama-3`) as an
opt-in engine. **Text you synthesize is sent to Fish Audio.** The engine is
unavailable until you add a key, makes no network call while idle or when
checking availability, and the app works fully without it.

## Setup

1. Create an API key at fish.audio.
2. Open **Settings → Models → TTS → Fish Audio voices**, paste the key into **Fish Audio API key** and press **Save**.
   It is stored encrypted in VoiceStudio's local secret store, shown as a password field, and never sent back to the UI.
   (Alternatively set the `FISH_API_KEY` environment variable; it takes precedence over the stored key.)
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
| Volume (dB) | `FISH_TTS_VOLUME` | `0` | -30 to 30 |
| Loudness normalization | `FISH_TTS_NORMALIZE_LOUDNESS` | `true` | S2 family; turn off to keep raw levels |
| Language | `FISH_TTS_LANGUAGE` | unset | e.g. `en`; unset lets Fish detect it |
| Base URL | `FISH_API_BASE_URL` | `https://api.fish.audio` | must be https (http for loopback only) |

Each setting can also be stored under `tts.fish.<name>` in the settings store.
The environment wins.

### Masters for external editing

Every response is saved byte-for-byte to
`<data dir>/outputs/fish_masters/<yyyymmdd>/<HHMMSS>_<hash>.<ext>` **before**
VoiceStudio's own processing (provenance watermark, loudness, resampling). Edit
those files when you want Fish's exact output. The file your project exports
goes through the normal pipeline.

## Your voices

In the app: **Settings → Models → TTS → Fish Audio voices**. Choose *My voices* or *Public library*, optionally type a title, press **Load voices**, then click a voice to make it the default. Nothing is requested until you press Load. The same thing over HTTP:

```bash
curl "http://127.0.0.1:<port>/api/fish/voices"                      # your saved voices
curl "http://127.0.0.1:<port>/api/fish/voices?scope=library&title=calm&language=en"
curl -X PUT http://127.0.0.1:<port>/api/fish/default-voice -H "Content-Type: application/json" -d '{"voice_id":"<id>"}'
```

Each entry has `id`, `title`, `languages`, `tags`, `visibility`, `state`, `author`
and a Fish-hosted `sample_url` (VoiceStudio does not fetch it). `id` is the value
for `FISH_TTS_REFERENCE_ID` / the default voice, or the per-request `voice`.
Listing is an explicit request that contacts Fish; it never runs at startup or
while checking availability. From Python: `FishAudioBackend().list_voices(mine=True)`.

## Story board (stage first, then submit)

**Stories → Fish story board** is a staging editor for multi-voice stories. Everything
on it is local: text, voice assignments and options autosave as a draft in
`<data dir>/fish_stories/`, and **nothing is sent to Fish Audio until you press
Submit and confirm** in the review dialog.

1. **Add text**: paste a story and split it into lines, paragraphs or sentences (optional max characters per line).
2. **Load my voices** (contacts Fish) lists your saved voices.
3. **Select lines** by shift/ctrl-click, by checkbox, or by number (*Lines 1 to 15* → **Select**), then choose a voice and **Apply**. **Apply & select next block** moves on to the following lines with the same count, so a long story goes quickly. Used voices appear as colored chips: click one to apply it to the current selection.
4. **Render settings** (left panel): format (WAV/PCM/MP3/Opus), sample rate, MP3 bitrate, speed, volume, language, model, latency mode, text and loudness normalization, temperature, top-p, chunk length, default voice, simultaneous requests, and the pause used in the joined file. Empty fields follow Settings/env.
5. **Review & submit to Fish** shows lines, characters, model, format and per-voice counts. **Check balance** shows your remaining credit (explicit request).

Output lands in `<data dir>/fish_stories/<draft>/renders/<timestamp>/`: one untouched file per line (`001_sun_2.wav`, …), a `manifest.json` (settings and text, no key) and, for WAV/PCM, a joined `story.wav`. Failed lines can be retried; running out of credits stops the job and keeps what finished. **Cancel** stops after the requests in flight.

## Limits (phase 1)

- Saved voices by `reference_id` only (list them as above); no inline cloning or voice upload yet.
- No voice design, no streaming.
- Licensing and pricing are Fish's; check their terms for commercial use.
