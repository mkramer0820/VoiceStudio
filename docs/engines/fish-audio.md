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
| Loudness normalization | `FISH_TTS_NORMALIZE_LOUDNESS` | `true` | S2 family; turn off to keep raw levels. Always sent explicitly (true or false), so the preview shows it. Off gives much hotter audio (about 11 dB louder in measured renders) |
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

## Voice library (name your characters)

**More voices** opens your voice library first. Every voice you list with **Load my voices**, add to a story (by id, library search or collection) or render with is saved locally in `<data dir>/fish_stories/voice_library.json`. Each record keeps your **character name**, the Fish name, the voice id (copy button and a link to fish.audio), Fish's details (author, languages, tags, description, visibility), your notes, and how many lines you have rendered with it and when. Renaming a character there also renames it on the board; refreshing your Fish voices never overwrites a name you chose. Search matches the character name, Fish name, id, notes and tags. Removing a voice only removes it from this list. Saving and renaming are local; only listing your account's voices or searching the public library contacts Fish.

## Story board (stage first, then submit)

**Stories → Fish story board** is a staging editor for multi-voice stories. Everything
on it is local: text, voice assignments and options autosave as a draft in
`<data dir>/fish_stories/`, and **nothing is sent to Fish Audio until you press
Submit and confirm** in the review dialog.

1. **Add text**: paste a story and split it into lines, paragraphs or sentences (optional max characters per line).
2. **Load my voices** (contacts Fish) lists your saved voices. **More voices** adds any other voice: paste a voice id (or a `fish.audio/m/<id>` link) with an optional name, or search the public voice library by title and language (searching contacts Fish; adding is local). **Public collection:** paste a `fish.audio/collections/<id>` link and press **Import** to list its voices, then add them one by one or all at once. Fish's API cannot list collections, so this reads the collection's public web page (an explicit request to fish.audio; no key or cookie is sent, only that one https page, size-capped). Private pages and bookmarks need a login and cannot be read, and a very long collection page may show only the voices rendered on its first page. Added voices show up everywhere a voice can be chosen, including the character mapping.
   - **Collections and bookmarks:** Fish's public API cannot list voices you bookmarked or collected on fish.audio (`GET /model` only filters by ownership, title, tag, language, author and sort order). Public collections can be imported by link (above); bookmarks are private and cannot be read, so open the voice on fish.audio and paste its link into *By voice id*. Voices you created are already under *Load my voices*.
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
   **Edits are staged:** changes in the settings panel only take effect for Preview and Render after you press **Apply settings** at the top of the panel, which also says how many changes are waiting (*Discard changes* drops them). Other settings include format (WAV/PCM/MP3/Opus), sample rate, MP3 bitrate, speed, volume, language, model, latency mode, text and loudness normalization, temperature, top-p, chunk length, default voice, simultaneous requests, and the pause used when VoiceStudio joins responses. Empty fields follow Settings/env.
6. **Preview requests** is a dry run: it shows the exact JSON bodies a render would send (with the endpoint, the `model` header, the planned file name and the output folder), checks each one against the request schema, and offers the first request as a `curl` command. It sends nothing, creates no files and never shows your key. The dialog also lists **Settings in effect**: every setting with its value, whether it comes from this story, the environment, a saved default or Fish's own default, and whether it is actually sent (for example, tuning fields show *No (basics only)* while **Send only the basics** is on). It reflects the applied settings (see *Apply settings* above), refreshes live, and shows the `curl` command open. A render also runs the same check and refuses to start if a line would be rejected.
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
- **Go back to an old run:** on a finished run, **Load into board** puts its story (lines, voices, cast) and its settings on the board as a new draft so you can change something and render again (it asks first if the board has lines; your current draft stays saved). **Listen** lists the run's lines with a play button each, to compare takes. **Open a run folder** at the top of the dialog takes the full path of any run folder (for example one you moved or archived), and the run then appears in the list. The folder is read in place; its own `job.json` may still name the old location.
- **Resume:** *Resume N unfinished* (after a confirmation) re-sends only unfinished lines, grouping them again according to the multi-speaker switch, and rebuilds the joined file. Running out of credits stops the job and keeps what finished, so top up and resume.
- **Warnings:** if Fish returns a different sample rate than requested, the job and log say so.
- **Defaults:** **Save these as my defaults** stores the render options in Settings for every new story. Environment variables still win.

## Training data (your renders as a dataset)

**Training data** in the sidebar indexes the story renders already on disk. It reads local files only and never contacts Fish.

- **Folder:** set the folder that holds your chapter folders (`<folder>/<chapter>/<story>/run-NNN_…/`, each with `manifest.json`, `story.json` and the WAVs). It is only read. A folder named `fish_non_api` is ignored.
- **Speakers:** clips are grouped by Fish voice id, so two names for one voice count as one speaker. Lines whose audio file is gone are skipped and counted; an older render of the same words by the same voice is hidden unless you include older copies.
- **Zoom in:** the magnifier beside a speaker opens a character view with that speaker's stats; *All speakers* returns. Under it, a length chart, an emotion-tag cloud and a chapter list describe whatever the filters currently match, and clicking them narrows the filters: a bar keeps that length range, a tag cycles *required* → *excluded* → off, a chapter toggles. Each active filter is a removable chip.
- **Filters:** text, length, word count, tags (required or excluded), chapters, older copies, sort order (story order, longest, shortest, most words, random sample) and a view of only the clips you left out. *Select all matching* and *Select none* act on every match, not only the visible page; unticked clips stay in the list (dimmed) so you can tick them again, and the header shows how many of the matches are selected. Everything starts selected; only selected clips are exported.
- **Audio quality:** *Measure audio* reads each WAV once (about 15 s for 3,000 clips) and remembers the numbers. Then you can limit peak, loudness range, leading and trailing silence, the number of pauses of 300 ms or more inside a clip, and the longest gap, or hide clipped clips. Chapters whose median loudness differs by 6 dB or more from the others are flagged, since mixing levels teaches a model two different volumes.
- **Listening:** play any clip, *Play from the top* with automatic advance to the next clip (and the next page), *Shuffle a sample* for spot checks, and keyboard control: click the list, then ↑ ↓ to move, Space to play or pause, X to leave a clip out.
- **Export:** writes a new folder (`<speaker>/<speaker>_00001.wav`) plus any of `metadata.jsonl` (Hugging Face audiofolder layout), a Fish `.lab` transcript beside each WAV, and an LJSpeech `metadata.csv` per speaker, and the **OmniVoice** training files (`omnivoice_train.jsonl` and `omnivoice_val.jsonl`, one `{id, audio_path, text, language_id}` line per clip), with an optional fixed validation split, optional resampling and either the `[emotion]` tags kept or removed. The OmniVoice format always writes 24000 Hz mono WAVs, plain text (only the tags you keep survive, and only OmniVoice's own tags are accepted) and `language_id: "en"`. **Bring all clips to one loudness level** (any format) rewrites each exported clip to the same average level, with peaks held below 0.97, so loud and quiet chapters can share a training set; your source files are never changed. A `dataset.json` card records the filters and source. An existing non-empty folder is never overwritten.
- **Tags:** the transcript can keep every `[emotion]` tag, or strip them all (the default, and all most sets need). To keep or convert some, build a **tag dictionary**: one entry per tag, separated by commas or new lines. `calm` keeps `[calm]`; `laughing=laughter` or `[laughing]:[laughter]` writes `[laughter]` instead; `[frustrated]:(sad)` writes exactly what follows the colon, so a model that marks emotion as `(sad)` gets that. Every tag not in the dictionary is removed. The OmniVoice format accepts only tags OmniVoice knows unless you untick *only tags OmniVoice knows*.
- **Portable rows:** the OmniVoice format writes each row twice: `omnivoice_train.jsonl` with absolute paths for the local runner, and `omnivoice_train.relative.jsonl` (plus `omnivoice_val.relative.jsonl`) with paths relative to the export folder, which is what you upload to Colab.
- **Training bundle:** with the OmniVoice format you can also write the full training setup next to the clips: `train_config.json` (the trainer's own settings: steps, learning rate, batch tokens, LoRA rank/alpha/dropout or a full fine-tune, precision, warm-up, checkpoint cadence, base model), `data_config.json`, `train.sh` (the exact setup, tokenise and train commands, all relative) and `COLAB.md` (a step-by-step guide with a table of the settings and a rough number of passes over your data). The guide and script have not been run on Colab by the generator, so treat the first run as a test. The pass estimate assumes the audio tokenizer runs at 25 frames per second; the real number comes from the tokenised data.
- **Terms:** the clips are synthetic speech from Fish. Check Fish's terms before using them to train another model.

## Training a voice (LoRA on OmniVoice)

The **Training** button in the Training data sidebar (and **Train a LoRA from this export** in the export dialog after an *OmniVoice training files* export) trains a small LoRA adapter for the local OmniVoice model from your clips. Nothing is uploaded and Fish is never contacted.

- **Opt in first.** Training is off until you turn it on in the dialog. It uses the GPU for a long time, writes several GB under the app's data folder (`fish_training/`), and unloads the other speech models while it runs. Starting a run also needs a confirmation tick.
- **What to export first.** One speaker only, *OmniVoice training files* format, *bring all clips to one loudness level* on, and a validation share (5-10%) so the result can be checked. The dialog warns about mixed speakers, unmatched loudness, no validation clips and small sets (under about 300 clips or 30 minutes).
- **Check (dry run)** validates the export (files, 24 kHz mono, ids, no train/validation overlap), builds the training and data configs and the exact commands, and probes the training Python and GPU, without starting anything. It tells you precisely what is missing: no NVIDIA GPU (Apple Silicon and CPU cannot run this trainer yet), too little GPU memory (about 8 GB is needed), Triton (OmniVoice's flex attention needs it; on Windows install `triton-windows` into the training Python, or train under WSL/Linux), `accelerate`, `webdataset`, `tensorboard` or OmniVoice itself.
- **A run** converts the audio to tokens (skipped if already done), trains, then stores the adapter in `fish_training/run_*/adapter/` with `adapter.json` and `evaluation.json`. Both steps are child processes, so **Cancel** really stops them. **Resume** continues from the last complete checkpoint after a cancel, a failure or closing the app (those runs show as *Interrupted*). One run at a time.
- **Offline by default.** The child processes run with `HF_HUB_OFFLINE=1`; the base model (`k2-fsa/OmniVoice`) and audio tokenizer (`eustlb/higgs-audio-v2-tokenizer`) must already be on disk or in the Hugging Face cache unless you tick *Allow downloading models from Hugging Face*.
- **Evaluation gate.** Each run records a held-out check: the validation loss before training (the untouched base model) against the finished adapter. The record is stored with the adapter and a run is **only marked ready to use when every check passes**. Today only the held-out-loss check can run automatically, so a finished run reads *Not fully checked*; the checks that need generated audio (speech-recognition word error rate, speaker similarity to your real clips, and a listening test on `heldout.jsonl`) are listed as pending in `evaluation.json`.
- **Not supported yet.** Fish S2 local and Qwen3-TTS adapters (those trainers are not in the repo), and loading a finished adapter into the synthesis engine.

## Limits (phase 1)

- Saved voices by `reference_id` only (list them as above); no inline cloning or voice upload yet.
- No voice design, no streaming.
- Licensing and pricing are Fish's; check their terms for commercial use.
