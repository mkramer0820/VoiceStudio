# Qwen story mode: a multi-voice book

A page (**Qwen story**, `/qwen-story`) that renders a book or script chunk by chunk with several Qwen3-TTS voices. Everything runs on this computer through the same one-at-a-time sidecar as the [playground](qwen3-playground.md). Built for the 1.7B models; not the C engine, not Colab, no training.

## Voice profiles

A profile is a saved voice: JSON under `<data dir>/qwen_studio/profiles/`, optional `<id>.bin` speaker embedding.

| Kind | Runs on | Voice |
|---|---|---|
| Narrator | a custom voice model (the fine-tuned checkpoint) | its speaker row, for example `ss_narrator`; no instruct needed |
| Stock speaker | a custom voice model | a speaker of a stock CustomVoice model; a borrowed stock name is allowed and warned |
| Clone | a stock **Base** model | a reference clip and what it says, or the speaker embedding only |

Fields: name, model (a model found through **Settings, Storage, Model locations** and the Hugging Face cache: there is no model path box on this page), speaker or voice sample, default instruct, emotion preset, optional sampling overrides (temperature, top K, top P, repetition penalty; empty means the engine's own default) and notes. A profile can also be made from a finished playground take (**Save as profile** under the take).

### Making a clone voice (the normal flow)

New profile, kind *Clone*: upload or record a sample (the same upload and record controls as the Clone page), check the words (the app transcribes the sample with its existing transcription and fills them in; you can edit them), press **Save voice**. The sample is stored as a WAV in the app's data folder and the voice is prepared automatically: right after saving if the Qwen engine is idle, otherwise just before the first render that uses it (the render panel shows "Preparing voice"). Preparing uses the sidecar's own x-vector step (the stock Base model on the CPU, no TTS model loaded) and caches the result next to the profile. A failed preparation is shown on the voice with a retry button and never blocks a render. You never see a `.bin` in this flow.

Under **Advanced** (collapsed): import or export that cached embedding as a `.bin` file, and the option to use only the sound of the voice without the sample's words. The `.bin` is raw little-endian float32[2048] with no header, the same layout as the C engine's x-vector file. Import refuses any other length or non-finite values; export returns the same bytes. Note: a normal Base clone computes its embedding inside the model call, so today the cached file is not read by story renders; it is kept for export and for the playground's *clone on custom voice* experiment (which stays playground-only). Whether a `.bin` made by the C engine behaves the same as one made by this stack is unverified (they are close, not byte-equal).

API: `GET/POST /api/qwen/profiles`, `GET/PUT/DELETE /api/qwen/profiles/{id}`, `PUT/GET /api/qwen/profiles/{id}/embedding`, `POST /api/qwen/profiles/{id}/prepare`, `POST /api/qwen/clips` (store a sample), `POST /api/qwen/profiles/from-take`.

## Emotion presets

`neutral, sad, joy, anger, fear, disgust, surprise, custom`, each in three strengths (normal, strong, very strong) as plain-English instruct text (`GET /api/qwen/emotions`). A preset is applied as `instruct`.

**Unverified on the fine-tuned narrator.** On EP09 an instruction changed the length and moved the voice away from the narrator ([findings section 6](qwen3-finetune-findings.md)). The page and the API say so; leave narrator profiles on *neutral* unless you are testing. The preset schema also reserves `steer` (a steering vector id and weight); it is accepted and ignored with the message "Emotion steering is not available yet".

Precedence for a chunk: an inline tag, then the character's emotion, then the profile's default instruct, then the profile's preset.

## Writing the script

- `[Name] text` or `Name: text` starts a line for a character. A line holding only `[Name]` keeps that character until a blank line. Untagged text is the narrator.
- `[sad]`, `[joy:strong]`, `[anger:very strong]`, `[neutral]` switch the emotion for the rest of the line. Names equal to a preset are never taken for characters.
- Text is cut into chunks the model can finish inside its cache (about 300 characters, sentence ends first). The chunk list is editable: text, speaker, emotion. Tick chunks (shift-click for a range, or **Select lines** on a character) to set an emotion or speaker for all of them, render only them, or remove them; **Tags and find** renames or merges characters, removes emotion tags story-wide and does plain-text find and replace.

## Rendering

Assign a profile to the narrator and to each character (free mapping: a character without a profile uses the narrator profile with a warning; a chunk with no voice at all is refused with the reason). Mappings and emotions are saved per story (`<data dir>/qwen_studio/stories/`).

One render at a time (shared with the playground). Chunks are rendered **grouped by model** so the narrator model and a Base clone model each load once, then the audio is put back in story order. Switching profile on the same model never reloads it: the sidecar chooses the speaker per request. *Warm up each voice first* (off by default) renders one throwaway sentence before the first real chunk of each voice; its benefit is unverified.

Output, as the Fish story pipeline does: `<output folder>/stories/<story>/run-NNN_<time>_<id>/` with `NNN_<speaker>_<profile>.wav` per chunk, `story.wav`, `manifest.json` (per chunk: file, profile, seed, length, whether the end of speech was produced, warnings) and `story.json`. Without an output folder set, the folder is `<data dir>/qwen_studio/story/`. Each chunk ends with the narrator's tail silence (**Pause after each chunk**, default 0.3 s); `story.wav` joins them in order, with an optional extra pause when the speaker changes. **Render chunk again** (and **Render failed or missing**) writes into the same run folder and keeps the old file (`..._r2.wav`).

Differences from Fish: the Fish board chunks and tags in the page and sends one request per speaker group; here parsing is done by the backend (`POST /api/qwen/story/parse`), each request is one chunk with one speaker, and the render order is by model, not story order.

API: `POST /api/qwen/story/parse`, `GET/POST/PUT/DELETE /api/qwen/story/drafts`, `POST /api/qwen/story/render`, `GET /api/qwen/story/jobs/{id}`, `POST .../cancel`, `GET /api/qwen/story/runs/{story}[/{run}[/audio/{file}]]`.

## Not verified

Never run through the real app or on a GPU: the whole page (component tests only), a Base clone and the narrator in one book, model swap timing, emotion presets on any model, warm-up benefit.

**Speaker embeddings and consistency.** While a Base model is loaded, the engine (faster-qwen3-tts) already builds a clone voice's prompt once and reuses it for every chunk, so a voice does not drift from re-encoding between chunks. The saved `.bin` embedding is for reuse across sessions and for import/export. An embedding prepared by the app is keyed to the reference clip (modification time and size): edit or replace the clip and the embedding is treated as missing and prepared again; an imported `.bin` is trusted as it is. A failed preparation is logged as a warning in the render log and never silently replaced. For repeatable renders set a seed in the story settings (each chunk then uses seed + its number).
