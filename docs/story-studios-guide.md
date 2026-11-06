# Story studios guide: Qwen playground, Qwen story, Fish Story, training data

How to use the pages for multi-voice books and for your own Qwen3-TTS fine-tunes. Everything here runs on your computer except Fish Story (which sends text to Fish Audio) and Colab training (which is yours to start). Things that were written but never run on a real GPU are marked **unverified**.

Reference pages for developers: [Qwen playground](engines/qwen3-playground.md), [Qwen story](engines/qwen3-story.md), [Qwen engine and install](engines/qwen3-narrator.md), [Colab training](engines/qwen3-tts-colab-runtime.md), [Fish Audio](engines/fish-audio.md).

## 1. One-time setup

1. **Models.** Open **Settings → Storage → Model locations** and add the folder that holds your fine-tuned checkpoints and LoRA adapters (for example `E:\Qwen3TTS_Checkpoints`). The pages search that folder, so a run folder such as `Run_2_EP09/checkpoints/checkpoint-epoch-9` is found by itself. They also list any stock Qwen3-TTS models already in your Hugging Face cache. There is no model path box on the Qwen pages: this list is the only place models are added.
2. **Engine.** The Qwen pages need the Qwen3-TTS Python environment (NVIDIA GPU only). Press **Install** on *Qwen3-TTS (faster-qwen3-tts)* in **Settings → Models**, or build `.venv-qwen` by hand ([steps](engines/qwen3-narrator.md#install)). The install adds no models.
3. **Nothing loads at startup.** The app loads a model only when you press a render, transcribe or generate button, so the first press waits a little longer. **Unload GPU** on the playground gives the memory back. To load a model at launch anyway, set `OMNIVOICE_PRELOAD_TTS=1` (or `OMNIVOICE_PRELOAD_CAPTURE_ASR=1`, `OMNIVOICE_PRELOAD_WATERMARK=1`).
4. **Keys.** The Fish Audio key lives in **Settings → Credentials** (the same key Fish Story uses). Qwen needs none.

## 2. Qwen playground (`/qwen-playground`): try a model

The playground renders short *takes* so you can compare models, voices and settings before using them in a book.

1. **Add take.** Each take has a prompt (the text to speak), a **Model**, a **Voice** kind and a repeat count.
   - **Custom voice**: a speaker baked into the model. Your fine-tuned narrator is one of these (speaker `ss_narrator`). Leave **Instruction** empty for the plain voice.
   - **Clone (on Base)**: copies a voice from a reference clip. Pick a saved **Voice profile**, or give a clip path and what it says. *Speaker embedding only* keeps the timbre but not the clip's pacing.
   - **Clone on custom voice** is an experiment.
2. **A LoRA adapter as the model.** Pick the adapter folder as the model. An **Adapter scale** field appears: 0.0 is the base model, **0.3 is the default**, above 0.5 may sound noisy. The first run merges the adapter into the Base model it was trained on (a minute on the CPU, then cached); the adapter itself is never changed. Add several takes at 0.2, 0.3, 0.35 and 0.5 to compare. The Base the adapter was trained on must be on this computer (in a Model location or the Hugging Face cache), otherwise the take is refused and says so.
3. **Run all** or **Run selected**. **Group by model** loads each model once. Results appear under each take with a player; star or note the ones you like.
4. **Save as profile** under a finished take stores its voice for Qwen story.
5. **Past sessions** reopens earlier runs. Each run writes its WAV files and a `manifest.json` into the output folder you set under *Settings* on this page.

## 3. Qwen story (`/qwen-story`): a multi-voice book

### 3.1 Voice profiles
A profile is a saved voice. **Profiles → New profile**:

| Kind | Use it for |
|---|---|
| Narrator (fine-tuned speaker) | your fine-tune; pick the model and its speaker, no instruction needed |
| Stock speaker | a speaker of a stock CustomVoice model (borrowed stock names are allowed but sound different) |
| Clone (reference clip) | upload or record a clean 5 to 15 s clip of one person; the words are transcribed for you (fix mistakes), then **Save voice** |

A profile can also carry a default instruct, an emotion preset, sampling values and a **Voice effect** (section 3.4). The voice file (`.bin`) is made automatically for clones; import or export one under **Advanced** only to share a voice with another tool.

### 3.2 Write or paste the script
- `[Anna] text` or `Anna: text` marks a character; a line that is only `[Anna]` keeps her for the following lines until a blank line. Untagged text is the narrator.
- `[sad]`, `[joy:strong]`, `[anger:very strong]` change the emotion for the rest of that line. Emotion on the fine-tuned narrator is **unverified** and may move the voice.
- `{Internal}` marks an effect (section 3.4). It switches the effect on for the rest of that line, so `He stopped. {Internal} Run. Now.` becomes two chunks and only the second has the effect; `{Internal}` on a line of its own covers the lines below it until a blank line. It combines with emotion tags in any order (`[sad] {Internal} Please no.`).
- Press **Split into chunks**. Chunks are sized for the model; the text is otherwise kept.

### 3.3 Assign voices, edit, render
1. Under **Characters**, choose a **Profile** (and optionally an emotion) for the narrator and for each character. A character with no profile uses the narrator's, with a warning.
2. In **Chunks**, edit text, speaker, emotion or effect per line. Tick lines (shift-click for a range) and use the toolbar to set the speaker, emotion or effect of the whole selection, **Render selected**, or **Remove selected**. **Tags** does story-wide rename/merge of characters, removal of emotion tags and find/replace.
3. **Render all** (or **Render failed or missing**). Lines are rendered grouped by model so each model loads once, then put back in story order. One render runs at a time (shared with the playground). **Render chunk again** (the redo arrow on its row) redoes a single line and keeps the older take as `..._r2.wav`.
4. Output: `<output folder>/stories/<story>/run-NNN_<time>_<id>/` with one WAV per line, `story.wav`, `manifest.json`, (the lines before their effect are kept outside the run folder, in `<story>/dry/<run>/`). **Full story** plays `story.wav`.
5. **Save** keeps the draft; **Open a saved story** reopens it.

Pause settings (in *Render*): *Pause after each chunk* and *Extra pause when the speaker changes*. *Warm up each voice first* is optional; its benefit is unverified.

### 3.4 Voice effects (inner monologue, telepathy, demon)
An effect is an ffmpeg recipe applied to a finished line.

- **Use one.** Set **Voice effect** on a profile (every line of that voice), on a single line, or on a selection from the toolbar. Or write the tag in the text: `{Internal} I should not be here.` (also `[{Internal}]`). The tag is stripped before the voice reads the line and the effect is applied to the audio afterwards. A line that is only `{Internal}` covers the lines below it until a blank line. Precedence: tag, then the line's setting, then the profile's.
- **Built in:** Inner monologue (alias `internal`, `thought`, `monologue`, `telepathy`: a stereo spread with a faint echo; needs an ffmpeg with the *rubberband* filter), Slight echo, Ethereal, Ghost, Demon, Cave, Radio, Muffled. They are starting points to tune by ear.
- **Judge them by ear.** **Voice effects → Edit** opens any effect, built-in ones included (your version is stored under the same id; **Reset to the shipped recipe** brings the original back). Under **Listen to it**, choose a clip (any audio file; on Fish Story you can also pick a line from your last render) and press **Hear it**: the first 20 seconds are run through the recipe as it is in the form, even unsaved, and the original and the processed audio play side by side. Change the recipe and press it again until it sounds right, then Save. **Test on a tone** only checks that ffmpeg accepts the recipe.
- **Add your own.** **Voice effects → New effect** (or **Copy** a built-in to start from it). Give it a name, the tags it answers to (comma separated), a **Type** and the recipe:
  - *Chain (-af)*: a filter chain such as `volume=-3dB,aecho=0.8:0.6:40:0.3`.
  - *Graph (-filter_complex)*: reads the line as `[0:a]` and ends in a stream labelled `[out]`, and may output stereo.
  - `{rate}` in a recipe becomes the sample rate. Choose the sample format (16/24-bit or float).
  - Press **Test on a tone**: it runs the recipe on a short test sound and shows ffmpeg's reason if it fails. Then **Save**. Your effects are kept in `effects.json` in the studio data folder and appear in every effect picker.
  - Filters that read files or the network (`movie`, `amovie`, `sendcmd`, `zmq`, `ladspa`, ...) are refused.
- If ffmpeg is missing or a recipe fails, the plain line is kept and the chunk shows why. `story.wav` is stereo when any line is (mono lines are centred).

## 4. Fish Story (`/fish-story`): a Fish model and an effect per role

Fish Story sends your text to Fish Audio, so it needs the key from **Settings → Credentials**. **Tags** (the Tags and cast dialog) holds the narrator and the characters:

- **Narrator (default voice)** and **Narrator model**: for example voice `Narrator` on model `s2.1-pro`.
- Under **Characters**, each `[Name]` has a voice and a **Model** (for example `drama-3`). Empty follows the story's model (**Settings panel → Model**).
- A line can carry its own model in the draft file; there is no per-line picker yet.
- **Voice effects work here too.** Write `{Internal}` anywhere in a line (the board keeps showing it): the tag is removed from the text sent to Fish, and when that line's audio comes back the effect is put on it. Fish's own `[tags]` such as `[whispering]` are sent exactly as written. The narrator and each character also have an **effect** choice in **Tags**, and **Voice effects** (next to Tags) is the same effect manager as in Qwen story, so your own recipes work in both. The returned file is kept untouched in `<story>/dry/<run>/` (next to the run folder, not inside it), the processed file is a `.wav`, and `manifest.json` records each line's text (with the tag) and its effect. One Fish call returns one file, so lines with a different effect go in separate calls. Effects need wav, mp3 or opus output (not raw pcm); if one fails the Fish audio stays as the line and the reason is in the manifest and log.
- **One Fish call has one model.** In multi-speaker mode, lines for different models are sent as separate calls; **Preview requests** lists every call with its model (and shows the first as a `curl` command), and the render log and manifest record it. Nothing is sent until you confirm the render.
- Fish Story and the Qwen pages share one **Pronunciation dictionary** (Settings → Pronunciation); `[[word|respelling]]` works inline in both.

### Fixing one line of a finished run (replace in place)

You do not need a new run folder to fix a line. In **Jobs**, open the run's lines and tick the ones to change:

1. If the board does not hold this story, press **Load** on the run first (that puts the run's story on the board). Change what was wrong *on the board*: the line's text, its voice, the character's voice in **Tags**, its model or its effect.
2. Tick the line in the run's list, then **Replace N line(s) with the board's version**, and confirm. The chosen lines are sent to Fish again (one call each, credits are used) and the new audio goes **into the same run folder**.
3. What is updated: the audio file (named for the new voice, for example `031_Shadow_right_voice.wav`), `manifest.json`, `job.json`, `story.json` (so it still describes the audio that is there), `requests.jsonl` (a new record marked `replacement`), `run.log`, and the joined `story.wav`. Other lines are not touched.
4. **Nothing is lost:** the audio that was replaced is copied to `<story folder>/replaced/<run folder name>/` (outside the run folder, like `dry/`). If a replacement fails, the line keeps its old audio and shows why.

For an **effect** only (add one, change it, or take it off), tick the lines and press **Re-apply effects (no Fish call)**: it starts from the untouched Fish response in `dry/`, so nothing is sent to Fish and nothing is paid. Put `{Internal}` in the line on the board (or change the character's effect) first. Lines that came back in one shared multi-speaker response must end up with the same effect.

## 5. Transcriptions: export with timestamps

On **Transcriptions**, open a transcript. Under **Segments**, **Export SRT**, **Export VTT**, **Export timed text** (`[hh:mm:ss]` per segment) and **Export JSON** (every segment with its start and end). If the speech engine returned no timings, only JSON is offered and the page says why; nothing is invented. The top-level export still writes plain text.

## 6. Training data: export for a fine-tune or a LoRA

1. Open **Fish dataset** (`/fish-dataset`), select the clips of **one** speaker, and press **Export … clips**.
2. Choose the **Qwen3-TTS fine-tuning folder** format. Set the speaker name (letters, digits, `-`, `_`), the reference clip (chosen for you unless you pick one), and the shortest and longest clip. Rejected clips are listed with a reason in `rejected_samples.jsonl`.
3. **Training mode**: *Full fine-tune (the whole model)* or *LoRA (small adapters)*. Full uses learning rate 2e-6; LoRA also defaults to 2e-6 with rank 16 and alpha 32. A LoRA export goes to its own `..._lora` folder and never overwrites a full export.
4. **Train with --instruct conditioning** (optional, **unverified on a GPU**): the `[tags]` of each clip (whispering, sigh) are written to an `instruct` field and trained as the instruction block, left out 30% of the time. It exists so a fine-tune keeps its voice when an instruction is used.
5. **Train several speakers in one model** (up to four) is optional.
6. The export folder holds `train.jsonl`, `audio/`, `reference/`, `manifest.json` (training mode, speaker, counts, parameters and a **bundle id**), `training_settings.json`, the Colab notebook, `voicestudio_qwen3_colab.py` (the runner), `colab_setup.sh`, `train_local.sh` / `train_local.ps1` and `README.txt`. The notebook and scripts only call the runner, and all of them print the same bundle id (`runner 23 (…)`); if two copies disagree, the older one is stale and the runner warns you.
7. **Run it.**
   - **Colab:** copy the export to Google Drive, open the notebook, pick an A100/H100/L4 GPU runtime, run the cells top to bottom. First do a `SMOKE_RUN` for a timing check. The notebook's first cell explains the exact Python/torch/FlashAttention versions it needs; **never upgrade torch in the Colab runtime**.
   - **Your own NVIDIA GPU (Linux or WSL2):** `bash train_local.sh` in the export folder (`--smoke-run` first). Unverified.
8. **Use the result.** Copy the checkpoint (or LoRA adapter) folder into a Model location. A full checkpoint is a normal custom-voice model in the playground and story. A LoRA adapter is used through the **Adapter scale** field (section 2) or merged into a permanent model with *Merge adapter* (asks for confirmation, never overwrites, keeps the adapter).

## 7. If something does not work

| You see | Do this |
|---|---|
| Your fine-tune is not in the model list | Add its parent folder under **Settings → Storage → Model locations**, then reload the page |
| "The Qwen engine cannot start" | Install *Qwen3-TTS (faster-qwen3-tts)* in **Settings → Models**, or set the Python path on the playground; it needs an NVIDIA GPU |
| A LoRA take is refused: Base not found | Put the Base model the adapter was trained on in a Model location or the Hugging Face cache |
| A clone profile says it could not prepare | Check the clip and that a stock **Base** model is available; the voice is then prepared at its first render |
| An effect is not applied | Open the chunk's message; press **Test on a tone** in *Voice effects* to see ffmpeg's error (the *rubberband* filter needs a full ffmpeg build) |
| A render stops at a length limit | Use a shorter chunk; the end-of-speech token was not produced |
| First press is slow | The model loads on first use by design; see section 1 to preload |
| A different model than expected in Fish Story | Check **Tags → Narrator model / character Model**, and **Preview requests** |
