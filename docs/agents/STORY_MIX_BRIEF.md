# Brief: Stitch & mix tab for the story studios

Next phase of the fork. Read [HANDOFF.md](HANDOFF.md) first (what the fork is, branches, conventions, scoped tests), then this. It is written so a fresh session can start without asking the owner what they meant; section 9 lists the few questions worth asking first.

## 1. Goal

Today the owner finishes a story in Fish Story or Qwen story, **copies the segment folder by hand** into `D:\fish_tts\input\<chapter>\...`, and runs their own command-line tool (`D:\fish_tts`, section 3) to stitch the segments into a chapter with pauses, speed, background music and tags. Bring that into the app as a **new tab** so the whole path from script to finished chapter is in one place, and add what the CLI cannot do: **per-role loudness and EQ** (inner thoughts a little quieter and darker than spoken parts, narration balanced against dialogue), background music that sits under the voices, and a **project that can be moved to another computer** with every folder easy to find.

The owner's words to keep in mind: "transferable and easy to find the folder locations", "works well with the current segmenting structure", "balance optimized audio", "add background music", "different EQs for audio with sound effects, aka internal thoughts should be slightly lower volume than spoken parts".

## 2. What the studios already write (the input)

Both studios write one folder per render. Use these facts, do not guess.

- **Fish Story**: `<fish output root>/<story folder>/run-NNN_<yyyymmdd-hhmmss>_<job id>/`. Default root `…\OmniVoice\fish_stories\renders` (`story_jobs.default_output_root()`; the Fish settings `output_dir` overrides it). A `.story-id` marker sits in the story folder. Files: `NNN_<speaker or narration>_<voice name>.<wav|mp3|opus|…>` per call, `story.wav` (a plain join, only for wav), `manifest.json`, `story.json` (the draft), `requests.jsonl`, `run.log`. One call can cover several lines (multi-speaker mode): then one file maps to several chunks; the manifest's chunk entries say which (`request_id`, `request_indices`).
- **Qwen story**: `<qwen story root>/<story>/run-NNN_<time>_<id>/` (`story.default_story_root()` = `…\qwen_studio\story`, or the playground output setting). Files: `NNN_<speaker>_<profile>.wav` per chunk, `…_r2.wav` for re-renders (the manifest's `file` is the current one; older ones are in `history`), `story.wav`, `manifest.json`, `story.json`.
- **Effect-free takes** are kept in `<story folder>/dry/<run folder name>/`, outside the run folder on purpose (an earlier layout confused the owner's stitcher). Never read `dry/` as segments.
- **manifest.json** chunk entries carry what a mixer needs: `index`, `id`, `status`, `file`, `speaker`, `voice`/`profile_name`, `effect` (for example `inner_monologue`, set by a `{Internal}` tag, a line setting or a character setting), `model`, `duration_s`, `text` (Fish; Qwen has it in `story.json`), `effect_error`. Prefer the manifest. **The owner's copied run folders keep it** (checked on `D:\fish_tts\input\3210_2210_fear_of_hope\untitled\run-001_…`: `manifest.json`, `job.json`, `story.json`, `requests.jsonl`, `run.log`, `settings.json` and 73 WAV lines at 44.1 kHz, the text still carrying Fish tags such as `[slow][calm]`), but a folder with only WAVs must still work, so fall back to the file name: `NNN_<speaker>_<voice>` with `narration` meaning no speaker, natural order, durations from ffprobe.
- The simple joins already in the app (`story_jobs._join`, `qwen3_studio/story.py stitch`) stay as quick previews; the new tab does not replace them.
- Effects: `backend/engines/qwen3_studio/effects.py` (ffmpeg recipes, `{Internal}` tags, edit/hear-it dialog, `dry_folder`). Reuse it; do not build a second effect system. **Do not apply an effect twice**: if a chunk's manifest `effect` is set, the audio already has it.

## 3. The owner's existing tool: port its behaviour, not its code layout

`D:\fish_tts` is outside the repo (the owner's, pure Python 3.10+ stdlib plus ffmpeg/ffprobe on PATH). Read these before designing anything:

| File | What to take from it |
|---|---|
| `src/audiobook_stitcher/narration.py` | per-segment decode to float PCM, **5 ms edge fades** (kills boundary clicks), optional per-segment `loudnorm` (`normalize_lufs`), pause files via `anullsrc`, concat, **pitch-preserving `atempo`** (chained when out of range), `silencedetect` and **internal-pause elongation** (cut at silence midpoints, splice extra silence; long filtergraphs go in a `-filter_complex_script` file because of the Windows command-line limit) |
| `src/audiobook_stitcher/background.py` | music loudness measured with `loudnorm` JSON, **gain = narration LUFS − `music_under_db` − music LUFS**, loop with `-stream_loop`, trim to the narration length, fade in/out, **pause boost** (music swells by N dB, with a ramp, inside detected pauses; one small `volume` expression per window), `amix duration=first normalize=0` |
| `src/audiobook_stitcher/pipeline.py`, `metadata.py` | one final encode with `alimiter`, staged output published atomically (hard link when not overwriting), FLAC/Ogg tags need Vorbis names (`TRACKNUMBER`, `DISCNUMBER`, `ALBUMARTIST`), M4A `media_type=2` + `+faststart`, MP3 ID3v2.3, cover art for M4A/MP3 |
| `src/audiobook_stitcher/config.py` | natural file ordering (`2.wav` before `10.wav`), `pause_after` keyed by zero-based position, last segment never gets a pause, **`speed` also stretches the pauses**, validation ranges |
| `src/audiobook_stitcher/batch.py`, `BATCH.md`, `settings.json` | a root folder where each child folder is a chapter, shared `defaults` plus per-chapter overrides, chapter filter, whole batch validated before any audio work |
| `config.json`, `settings.json` | the owner's real values: pause 1000-1150 ms, speed 0.97, music 20-28 dB under the voice, fades 2000/3000 ms, pause boost 6 dB with a 150 ms ramp, internal pause +350 ms below -72 dB for 150 ms, 44.1 kHz, FLAC, metadata template (artist/album/album_artist/composer/genre/date/disc/comment/copyright) |
| `voice_effect_te/effects_test.ps1`, `echo1.txt` | his voice-design experiments: dozens of ffmpeg chains (`rubberband=pitch=0.84:formant=0.88`, `highpass`, `equalizer`, `lowshelf`, `acompressor`, `aecho`). Seed **narrator EQ presets** from these (the pitch-free parts), and offer a few of his deity/whisper chains as built-in voice effects after he confirms which he likes. |

The tool has a known open item in its README: an unreproduced stutter in one recording; it needs an excerpt to diagnose, so do not claim it is fixed.

Compatibility goal: the new tab must be able to **import his `config.json` / `settings.json`** (map the fields above) so he can try it on a chapter he already stitched and compare, and keep working in his CLI meanwhile.

## 4. What to build

Route `/story-mix`, menu entry in the stories group of `workspace-menu.tsx`, a command-palette entry, a home tile (see `lib/fork-profile.ts`: the page is visible by default; nothing to hide). Name it "Stitch & mix" (all UI text through i18n, 21 locales).

### 4.1 Where things are (build this first; it is the "easy to find" part)
A header strip on the page with four rows: **Source** (the run or folder), **Project** (the mix project folder), **Music**, **Output**. Each row shows the full path, an **Open folder** button, a **Copy path** button, and a status (found / missing). A "Where are my files?" panel lists the Fish and Qwen output roots from their settings with the same two buttons. Open-in-Explorer already exists (`POST /export/reveal`, `POST /api/fish/story/jobs/open`, the bridge); reuse it. Never make the owner type a path when a folder picker exists.

### 4.2 Source
Pick a render run from either studio (newest first, with story, date, line count, status, path) or any folder. For a folder, use the same discovery as his tool (natural order, optional recursion), skip `dry/`, skip `story.wav` and the output file, and use the manifest if present, else the file names. If the same story has several runs, show them and default to the newest finished one. Show a missing-line warning when `manifest.json` lists lines whose files are gone.

### 4.3 Segments table
One row per line: index, role, speaker, text start (from the manifest or `story.json`), duration, play button, mute, **pause after (ms)**, **gain trim (dB)**. Bulk select (shift-click) for role, pause and gain, like the Qwen story chunk list. Roles: **narration**, **dialogue** (spoken by a character), **internal** (thoughts), **other**; chosen automatically (effect `inner_monologue` or an `{Internal}` tag means internal; a speaker means dialogue; no speaker means narration) and overridable per line. Hearing a region: play from this line for 30 s using the same render chain (4.6).

### 4.4 The mix model (the new part)
A project-level settings block plus a table per role:

| Per role | Meaning |
|---|---|
| level offset (dB) | relative to the narration target, for example internal -3, dialogue +1 |
| EQ | a preset or custom bands (high-pass, low-pass, low shelf, high shelf, up to four peaking bands) built into one ffmpeg `equalizer`/`highpass`/`lowpass`/`bass`/`treble` chain |
| compressor (optional) | `acompressor` threshold/ratio, off by default |

Project level: speed, default pause, internal-pause elongation, **per-segment loudness normalisation to a target LUFS** (his `normalize_lufs`), the order of operations below, output loudness target (optional, `loudnorm` two-pass on the final mix), peak limiter.

Order per segment: decode to float PCM at the project rate → 5 ms edge fades → mute/skip → **loudness normalise to the narration target** → role level offset + line trim → role EQ/compressor → into the stitch. Then pauses, speed, internal-pause elongation, music mix, final limiter and one encode, exactly as his tool does after the segment stage. Use one ffmpeg process per segment stage and a concat, not one giant graph (long chapters).

Judge EQ by ear the same way as effects: reuse the **Hear it** pattern (pick a line, original vs processed side by side) for any role preset. Ship a few starting presets and say plainly they are untested starting points: *Voice clarity* (high-pass 70 Hz, +2 dB around 3 kHz), *Warm narrator* (his low-shelf/low-mid chain without the pitch shift), *Thought* (low-pass about 7 kHz, -3 dB), *Dialogue* (+1.5 dB, gentle presence).

### 4.5 Background music
Everything his tool does (section 3), plus a better default for dialogue: optional **ducking** (`sidechaincompress` keyed by the voice bus) in addition to his fixed `music_under_db`. Keep the LUFS-relative gain as the default because it is what he trusts. Music file picker (copy-in option, section 4.7), per-chapter override, fades, pause boost. Show the computed music gain after a dry run.

### 4.6 Render
A background job like `fish_api/story_jobs.py`: start, progress per stage, cancel, log, `keep temp` option, atomic publish, overwrite protection. Also a **dry run** that prints the plan the way his `--dry-run` does (files, pauses, gains, music) without touching audio, and an **excerpt render** (a range of lines, about 60 s) for quick listening. All ffmpeg through `services.ffmpeg_utils.find_ffmpeg()`, and every command through `local_inputs_only()` (the repo has a test that fails otherwise). Temp files go to a job temp folder and are removed unless `keep temp`.

### 4.7 Project and portability
A mix **project** is one folder with `mix.json` and optional subfolders (`segments/`, `music/`, `cover/`, `output/`). Rules:
- **All paths in `mix.json` are relative to the project folder**, except explicitly "linked" sources, which store an absolute path and are flagged in the UI.
- **Collect files** copies the used segments, the music and the cover into the project so the folder is self-contained; **Export bundle** zips it. **Open project** on another machine works with no edits; missing linked files appear in a "Relink" list instead of an error wall.
- Settings that are the owner's defaults (metadata template, music, pauses, role table) live in a separate "defaults" file so a new chapter starts from them.
- `mix.json` carries a `version`; loading an older version migrates, an unknown newer one refuses clearly.

### 4.8 Output and tags
Format FLAC (default, his), M4A, MP3, WAV; sample rate 44.1 kHz; channels mono or stereo (default: stereo if any line is stereo, because the inner-monologue effect is stereo, otherwise his default); metadata template with per-chapter title/track; cover for M4A/MP3. Output default `<project>/output`, file name from the chapter title, never overwrite without the setting.

## 5. Where to put it in this codebase

- Backend: `backend/services/story_mix/` (`sources.py` discovery and role detection, `plan.py` the validated plan and dry run, `chain.py` the per-segment ffmpeg chain and EQ builder, `music.py`, `render.py` the job, `project.py` load/save/collect/migrate) and a router `backend/api/routers/story_mix.py` under `/api/mix/…` registered where the other routers are. No new Python dependency (his tool has none; keep it so).
- Frontend: `electron/src/renderer/src/features/story-mix/` (page, `use-story-mix.ts`, `mix-model.ts` pure logic, segment table, role/EQ panel, music panel, paths strip, render panel), route in `routes/index.ts`, menu in `components/app-shell/workspace-menu.tsx`, palette in `components/command-palette.tsx`.
- Reuse: `effects.py` (Hear it, `dry_folder`, recipe validation), the Qwen story chunk-list selection helpers (`story-model.ts`: `idsBetween`, `applyEffect`-style bulk helpers), `describeError`, `apiFetch`.
- Docs: a "Stitch & mix" section in `docs/story-studios-guide.md`, an engine-neutral page `docs/story-mix.md`, HANDOFF.md updated, CHANGELOG `[Unreleased]` lines.

## 6. Hard rules (from the owner and the repo)

1. **Do not touch the training export or the Colab runner** (HANDOFF.md section 4.3).
2. Windows + CUDA only is the supported target; ffmpeg work is platform-neutral anyway. Nothing loads a model: this feature is pure ffmpeg.
3. No model path boxes; the output/music/source folders are job inputs, chosen with pickers and shown in the paths strip.
4. Commits: owner's git identity only, no Co-Authored-By or "Generated with" lines, small reviewable commits, `main` is the working branch, do not push unless asked.
5. CRLF files and the 21 locale files: edit carefully (no `sed -i` on locale JSON; keys in every locale; no hard-coded CJK). A CHANGELOG line per user-visible change in the repo's style (one line, ends `— thanks @mkramer0820!`).
6. Every ffmpeg command goes through `local_inputs_only()`; long filtergraphs go in script files.
7. Say what is **unverified** (never listened to) in the docs and the UI, as the rest of the fork does.

## 7. Phases

1. **MVP, paths and stitch.** Paths strip, source picker, segments table (play, mute, pause), stitch with pauses/speed/edge fades/limiter, FLAC/M4A/MP3/WAV, dry run, `mix.json` with relative paths, Open/Copy buttons. Acceptance: on `D:\fish_tts\input\3210_2210_fear_of_hope\untitled\run-001_…` it produces the same duration as his tool with the same settings (pauses and speed), and the project folder can be moved and still renders.
2. **Balance.** Roles, per-segment loudness normalisation, role offsets, EQ presets and editor with Hear it, excerpt render.
3. **Music and polish.** Background music (his algorithm), pause boost, internal-pause elongation, optional ducking, metadata and cover, import of his `config.json`/`settings.json`.
4. **Transfer and batch.** Collect files, export bundle, relink list, chapter batch with shared defaults.

Each phase ends with scoped tests green, docs and changelog updated, a commit.

## 8. Tests (write them with the code)

- Pure logic (frontend `mix-model.test.ts`, backend `tests/test_story_mix_*.py`): natural ordering, role detection from manifest and from file names, pause indexing (zero-based, last never padded), speed scaling pauses, gain arithmetic (music gain equals narration LUFS minus offset minus music LUFS), path relativisation and migration.
- ffmpeg integration with synthetic tones (skip when no ffmpeg): stitched duration equals the sum of segments plus pauses divided by speed within a few samples; a quieter role really measures lower (loudnorm/ebur128); EQ chain builds and runs; stereo effect line plus mono lines stitch to stereo; edge fades remove an abrupt opposite-polarity boundary (his regression test).
- Portability: build a project, move the folder, render again; delete a linked file and expect a Relink entry, not an exception.
- `tests/test_ffmpeg_local_inputs.py` stays green (every command wrapped).
- Parity check (manual, recorded in the PR text): same chapter through his tool and the tab, compare duration, music gain log line and a listening note.

## 9. Ask the owner first (two minutes, saves a rewrite)

1. Default output: FLAC mono like `settings.json`, or stereo? (Inner-monologue lines are stereo.)
2. Should the tab **replace** copying folders into `D:\fish_tts\input`, or sit beside his CLI for a while? (Plan assumes beside, with config import.)
3. Roles: are narration / dialogue / internal enough, or does he want per-character mixes (each character its own level and EQ)?
4. Which of his `voice_effect_te` chains (god voice, freeman narrator) should become built-in presets?
5. Ducking: wanted, or is the fixed `music_under_db` plus pause boost enough?

## 10. Not in scope

Dubbing, the upstream audiobook builder, anything that loads a TTS model, changes to Fish/Qwen render output formats (the mixer adapts to them), loudness standards compliance for retail platforms (mention ACX/Audible limits in docs only if asked).
