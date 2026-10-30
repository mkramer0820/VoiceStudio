# Qwen playground (try every Qwen3-TTS variant)

A page in the app (**Qwen playground**, next to the Fish story board) for trying the local Qwen3-TTS models quickly: several prompts ("takes") in one run, many settings, every render saved with a manifest, and notes and a star on each take. It needs the [Qwen3-TTS engine](qwen3-narrator.md) (NVIDIA GPU, its own Python environment). Everything runs on your computer; nothing is downloaded or sent anywhere. What the models and voices are known (and not known) to do is in [qwen3-finetune-findings.md](qwen3-finetune-findings.md).

## The three recipes

Each take shows its recipe and a one-line summary of what it really runs: `Model: ... (type) · Voice: ... · Mode: ...`.

| Recipe | What it runs | Use it for |
|---|---|---|
| **Narrator** | A model with a speaker built in (the fine-tuned narrator, speaker `ss_narrator`), no instruction, no reference clip | The intended narrator output |
| **Clone** | The stock **Base** model plus a reference clip and the words the clip says | Any other character; the best clone mode so far |
| **Experiment** | An instruction on the narrator, "speaker embedding only" cloning, or "clone on custom voice" | Trying things. Never a default; the narrator moves away from itself when given an instruction |

Voice choices on a take:

- **Custom voice**: a speaker baked into the model. Pick it from the model's own speakers. A one-speaker model (the narrator) is pre-filled; a model with several makes you choose.
- **Clone (on Base)**: pick a saved voice profile (its text is filled in and editable) or give a WAV path and its text. "Speaker embedding only" copies just the timbre; tempo is freer.
- **Clone on custom voice (experimental)**: the clip's speaker embedding is written into the custom model's speaker for one render and restored afterwards. Not yet judged by ear.

**Borrowed stock voices.** The nine stock names (`ryan`, `vivian`, ...) can be picked on any model, under "Borrowed stock voice (will sound different)". A model that was not trained with that voice does not have it, so the result will not sound like the real one, and the take shows that warning next to the player. For the real stock voices use the stock Hugging Face CustomVoice model.

The page warns when a voice type and model type do not match, but never blocks it: you can try any combination, and if the engine refuses, its reason is shown on the page.

## Running

Set the **output folder**, a **session name**, and (if needed) the **Python path** (saved with *Save settings*; changing the Python path restarts the engine on the next run). Where models live is not set here: use *Manage model folders* (or open **Settings -> Storage -> Model locations**). The model list shows where each model comes from, for example `ss_narrator, Production`. A saved session whose model is no longer found still opens; its take shows "Model not found" with the old path. *Warm-up runs* render throw-away text before the first real take of each voice so a cold start does not spoil it. *Group by model* reorders the run so the same model and voice run back to back; the page shows how many model loads the current order needs. Results keep the order you wrote the takes in.

*Run all*, *Run selected*, *Cancel* (after the current render) and *Unload GPU* (stops the engine and frees the memory). Only one run goes at a time.

Under each take you get a player per render, its length, speed (times real time), the seed used, and a warning when the model never produced its end-of-speech token (audio may be cut off or run on; try another seed). Pick **A** and **B** on two renders and use *Play A then B* to compare. *Past sessions* lists earlier sessions in the output folder and opens them with the same players and notes.

The advanced group has every setting: language, seed (or a new seed for every render; repeats use the seed, then the next numbers), temperature, top-k, top-p, repetition penalty, tail silence, chunk size, and max sequence length (changing it reloads the model). Empty boxes use the engine's own defaults.

## Where files go

`<output folder>/<session name>_<date-time>/` holds one WAV per render (named with the take number, repeat, model, voice kind and label) and `session.json`, the manifest: the run's settings, each take, each result (file, seconds, seed, whether it finished, warnings) and your notes and stars. Notes and stars are the only thing that edits a manifest afterwards. The default output folder is shown on the page; the folder you type is saved as the engine setting so past sessions and notes use the same place.

Voices you add with *Add clone voice* are saved as ordinary voice profiles through the app's existing voices, so they also appear in the rest of the app. Your takes on the page are remembered in the browser for next time.

Under a finished take, **Save as profile** stores its voice as a story-mode profile; see [qwen3-story.md](qwen3-story.md).

**LoRA adapters.** An adapter folder (found in Settings -> Storage like any model) can be picked as the model of a take. Pick the *Adapter scale* (0.0 is the Base model, default 0.3, above 0.5 warns); the playground merges the adapter into its Base model on the CPU the first time (about a minute, cached under the app's `qwen_studio/merged-cache`, the newest three kept) and runs that. The adapter folder is never changed. Add several takes with different scales (0.2, 0.3, 0.35, 0.5) to compare them. To keep a merged model as a normal checkpoint use *Merge adapter* (it asks for confirmation and never overwrites).
