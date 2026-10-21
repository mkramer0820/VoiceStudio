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
| Model | `FISH_TTS_MODEL` | `s2.1-pro` | `s2.1-pro-free` (free, for testing; no latency guarantees), `s2-pro`, `drama-3`; unknown names fall back to `s2.1-pro` at Fish |
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
| Repetition penalty | `FISH_TTS_REPETITION_PENALTY` | unset | above 1.0 reduces repeated sounds |
| Max audio tokens | `FISH_TTS_MAX_NEW_TOKENS` | unset | caps audio length per chunk |
| Min chunk length | `FISH_TTS_MIN_CHUNK_LENGTH` | unset | |
| Condition on previous chunks | `FISH_TTS_CONDITION_ON_PREVIOUS` | unset | keeps the voice consistent across chunks |
| Early stop threshold | `FISH_TTS_EARLY_STOP_THRESHOLD` | unset | |
| Opus bitrate | `FISH_TTS_OPUS_BITRATE` | unset | `-1000` = automatic; opus only |
| Language | `FISH_TTS_LANGUAGE` | unset | e.g. `en`; unset lets Fish detect it |
| Story output folder | `FISH_STORY_OUTPUT_DIR` | `<data dir>/fish_stories/renders` | absolute path; one folder per story and run is created inside |
| Base URL | `FISH_API_BASE_URL` | `https://api.fish.audio` | must be https (http for loopback only) |

Each setting can also be stored under `tts.fish.<name>` in the settings store.
The environment wins.

### Masters for external editing

Every response is saved byte-for-byte to
`<data dir>/outputs/fish_masters/<yyyymmdd>/<HHMMSS>_<hash>.<ext>` **before**
VoiceStudio's standard-output watermark and container encoding. Edit those
files when you want Fish's exact response bytes. The file your project exports
keeps Fish's finished waveform: VoiceStudio does not apply its local-engine
broadcast mastering, effect preset, or peak normalization to Fish output.
Provenance watermarking and the selected WAV container encoding still run.

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
2. **Load my voices** (contacts Fish) lists your saved voices. **More voices** adds any other voice: paste a voice id (or a `fish.audio/m/<id>` link) with an optional name, or search the public voice library by title and language (searching contacts Fish; adding is local). Added voices show up everywhere a voice can be chosen, including the character mapping.
   - **Collections and bookmarks:** Fish's public API cannot list voices you bookmarked or collected on fish.audio (`GET /model` only filters by ownership, title, tag, language, author and sort order). Open the voice on fish.audio and paste its link into *By voice id*; voices you created are already under *Load my voices*.
3. **Select lines** by shift/ctrl-click, by checkbox, or by number (*Lines 1 to 15* → **Select**), then choose a voice and **Apply**. **Apply & select next block** moves on to the following lines with the same count, so a long story goes quickly. Used voices appear as colored chips: click one to apply it to the current selection.
4. **Tags & cast** handles AI-directed scripts such as `[Shadow] [slow][soft] It feels strange, [pause] to become a [emphasis] human.` It opens by itself when pasted text has `[Name]` tags at the start of lines.
   - **Narrator (default voice):** used for narration and every line with no speaker tag, so characters only need their own lines tagged.
   - **Characters:** a capitalised tag that starts a line (`[Shadow]`, `[Snuggle]`) is a character. Map each to a voice and press **Apply characters and voices**; the name is removed from the text and recorded on the line. A tag in the middle of a line starts a new line. Untagged lines stay narration unless you tick *keep the previous character*. Delivery words such as `[Pause]` are never treated as characters, and each tag can be flipped with *Not a character* / *Is a character*.
   - **Safety check:** Preview and Review warn when a line still opens with a character tag such as `[Shadow]` that was never applied, since Fish would receive it as text. Only a tag at the start of a line counts; character tags are never expected mid-line.
   - **Delivery tags** (`[slow]`, `[soft]`, `[pause]`, `[emphasis]`…) are kept for Fish by default; remove them or replace them with other tags.
   - **Find and replace** works on plain text across the whole story (include the brackets to change a tag).
5. **Render settings** (left panel). **Multi-speaker calls** is a plain on/off switch, off by default.
   - **Off (default), single-speaker calls:** sends one ordinary request per story line with a string `reference_id`. VoiceStudio keeps every response and stitches the joined WAV locally with the configured pause. This is the resumable, line-by-line workflow.
   - **On, multi-speaker calls** (`s2-pro`, `s2.1-pro`, `s2.1-pro-free` and `drama-3` support it; `s1` does not and the review warns): combines as much of the story as Fish's 20,000-character request limit permits. Character IDs are mapped in first-appearance order to `<|speaker:N|>` tags and the corresponding voice IDs are sent as a `reference_id` array. Fish performs the speaker transitions and stitching inside each response. If only one mapped speaker is present, the same large-call mode uses Fish's single-speaker schema (one string ID and no speaker tags). A one-response WAV is copied unchanged to `story.wav`; VoiceStudio only stitches locally when the story needs multiple large calls.
   - **Send only the basics** (Requests section) sends just text, voice, format and sample rate, like the web page, and leaves everything else to Fish's defaults. **Use Fish defaults** clears the tuning fields and any saved tuning defaults. A blank tuning field is never sent, and *Save these as my defaults* clears a stored value when the field is blank. **Preview requests** lists every field a request sends beyond the basics.
   - Multi-speaker calls require every character and narrator in a mixed-speaker request to resolve to a voice. Resolution order is a line's explicit voice, its saved character mapping, then the default narrator voice. Preview and render stop before contacting Fish if a voice is missing.
   Other settings include format (WAV/PCM/MP3/Opus), sample rate, MP3 bitrate, speed, volume, language, model, latency mode, text and loudness normalization, temperature, top-p, chunk length, default voice, simultaneous requests, and the pause used when VoiceStudio joins responses. Empty fields follow Settings/env.
6. **Preview requests** is a dry run: it shows the exact JSON bodies a render would send (with the endpoint, the `model` header, the planned file name and the output folder), checks each one against the request schema, and offers the first request as a `curl` command. It sends nothing, creates no files and never shows your key. A render also runs the same check and refuses to start if a line would be rejected.
7. **Review & submit to Fish** shows lines, characters, model, format and per-voice counts. **Check balance** shows your remaining credit (explicit request).

### Output, job tracking and logs

Each render gets its own folder, so nothing is ever overwritten:

```text
<output folder>/<story title>/run-<NNN>_<YYYYMMDD-HHMMSS>_<job id>/
  001_Shadow_sun_2.wav …   one untouched Fish response per API call
  001-020_multi_3-speakers.wav  example large multi-speaker response
  story.wav                one Fish WAV copied unchanged, or multiple responses joined locally
  job.json                 live state; survives closing the app
  story.json               the submitted story (used by Resume)
  settings.json            the options used (never the key)
  requests.jsonl           one record per API call: covered lines, mode, exact request, HTTP status, attempts, timing, size, SHA-256 and audio details
  run.log                  readable event log (started, each request, retries, warnings, finish)
  manifest.json            final summary with the text, speaker and voice for every line
```

- **Output folder:** the default is `<data dir>/fish_stories/renders`. Set another absolute path in the board's **Output folder** field (or `FISH_STORY_OUTPUT_DIR`). The folder is created and checked for write access when you save it. Each story gets a readable folder named after its title (`the-long-night`; a second story with the same title becomes `the-long-night-2`) and runs are numbered `run-001`, `run-002`… A renamed story keeps using its own folder.
- **Render jobs:** the **Render jobs** button lists every run, newest first, with status, progress, model and format, minutes of audio, folder, and the log. A render cut off by closing the app shows as *Interrupted*.
- **Resume:** *Resume N unfinished* (after a confirmation) re-sends only unfinished lines, grouping them again according to the multi-speaker switch, and rebuilds the joined file. Running out of credits stops the job and keeps what finished, so top up and resume.
- **Warnings:** if Fish returns a different sample rate than requested, the job and log say so.
- **Defaults:** **Save these as my defaults** stores the render options in Settings for every new story. Environment variables still win.

## Limits (phase 1)

- Saved voices by `reference_id` only (list them as above); no inline cloning or voice upload yet.
- No voice design, no streaming.
- Licensing and pricing are Fish's; check their terms for commercial use.
