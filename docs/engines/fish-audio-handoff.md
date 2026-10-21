# Fish Audio branch handoff

This document records the integration decisions on `ext/fish-api` so future work preserves the intended audio path and Story Studio behavior.

## Audio preservation

- Fish Audio and GPT-SoVITS are remote backends. Their returned audio must not pass through a local model backend or VoiceStudio's normalization/export pipeline.
- A single remote WAV response is preserved byte-for-byte, including when it becomes Story Studio's convenience `story.wav` output.
- Story Studio only decodes and joins audio when Fish's 20,000-character request limit requires multiple responses. That join does not apply loudness normalization. It may resample mismatched responses so a valid combined WAV can be written.
- The `normalize` field in a Fish request is Fish's server-side option. It is separate from VoiceStudio's local normalization pipeline.
- Keep the response's real format and sample rate. In particular, a requested WAV at 44.1 kHz must not be silently transcoded or routed through another engine.

## Story Studio request modes

Multi-speaker calls are a simple on/off switch. Off (the default; also a draft with no mode saved) means one call per line. On groups lines into speaker-tagged calls. The choice is saved with the draft and kept for new stories.

- **Small single-speaker calls (`single`)**: one Fish request per story line. VoiceStudio joins the returned files locally when Join is enabled.
- **Large multi-speaker calls (`multi`)**: Story Studio groups lines into requests below Fish's 20,000-character text limit. It sends speaker-tagged text (`<|speaker:N|>`) and an ordered `reference_id` array. Fish performs the speaker transitions.
- A large-mode group containing only one speaker uses Fish's single-speaker schema: combined plain text plus one string `reference_id`.
- If a large story needs more than one Fish request, VoiceStudio joins only those large response files. Shared response filenames are deduplicated before joining.

Speaker voices resolve in this order:

1. Explicit voice on the line.
2. The draft cast mapping for the line's speaker ID.
3. The story/default narrator voice.

A mixed-speaker request is rejected locally if any speaker lacks a resolved voice. Preview and render compile the same request groups and schemas, so preview is the authoritative dry run.

## Code map

- `backend/engines/fish_api/client.py`: accepts a string or array `reference_id` and sends the Fish request.
- `backend/engines/fish_api/story.py`: draft schema, explicit nullable `request_mode`, request validation, and planning.
- `backend/engines/fish_api/story_jobs.py`: request grouping, speaker mapping, preview, render/resume logging, and output joining.
- `backend/api/routers/fish_story.py`: Story API validation and preview errors.
- `electron/src/renderer/src/features/fish-story/`: required mode controls, disabled preview/render state, and request preview.
- `docs/engines/fish-audio.md`: user-facing setup and behavior reference.

## Regression coverage

The backend tests cover exact WAV preservation, one-call multi-speaker rendering, one-speaker large calls, missing voice rejection, 20,000-character splitting, local joining, an unset mode meaning single, minimal requests, and the multi-speaker model warning. The renderer test covers the on/off switch. Keep all 21 renderer locales in sync when changing labels.

No live Fish call is required for these tests, and no API key belongs in fixtures, logs, commits, or documentation.
