"""Pure logic for the fine-tuned Qwen3-TTS narrator sidecar: chunking, length
prediction, end-of-speech push, tail padding.

Stdlib only at import time (numpy loads lazily inside the audio helpers) so the
parent's tests can import it without torch or a GPU, and ``main.py`` can import
it from inside the engine's own venv. Frame rate is Qwen3-TTS's 12.5 Hz.
"""
from __future__ import annotations

import math
import re

FRAME_HZ = 12.5
#: Frames of the static KV cache the prompt itself uses; the rest is for audio.
PROMPT_RESERVE = 400
#: The model was trained on 4-20 s clips, so chunks aim at about that length
#: however much cache is left.
MAX_CHUNK_CHARS = 300
#: Seconds per character: slowest 5% of the narrator's training clips, and the
#: median. The Colab test derives them from the dataset; these are its fallbacks.
SLOW_S_PER_CHAR = 1 / 10.0
MEDIAN_S_PER_CHAR = 1 / 15.0

_SENTENCE_END = re.compile(r"(?<=[.!?…])[\"'”’)\]]*\s+")
_SOFT_BREAK = re.compile(r"(?<=[,;:—–])\s+")


def token_limits(chars: int, slow: float = SLOW_S_PER_CHAR, median: float = MEDIAN_S_PER_CHAR) -> tuple[int, int]:
    """(start, cap) in frames: from ``start`` the end-of-speech logit is pushed
    up; ``cap`` is the hard backstop."""
    cap = int(min(700, max(40, math.ceil((chars * slow * 1.15 + 0.5) * FRAME_HZ))))
    start = int(max(20, math.ceil(chars * median * 1.1 * FRAME_HZ)))
    return start, cap


def classify(steps: int, start: int, cap: int) -> str:
    return "hit the limit" if steps >= cap else ("pushed to stop" if steps > start else "stopped by itself")


def max_audio_frames(max_seq_len: int, reserve: int = PROMPT_RESERVE) -> int:
    """Frames of audio one generation may use: the cache minus the prompt
    (``reserve``; a clone's reference clip adds its own frames to it)."""
    return max(0, min(700, int(max_seq_len) - int(reserve)))


def default_chunk_chars(max_seq_len: int, slow: float = SLOW_S_PER_CHAR) -> int:
    """Largest chunk whose capped length still fits the static cache, never
    above the clip length the model was trained on."""
    seconds = max_audio_frames(max_seq_len) / FRAME_HZ
    fit = int((seconds - 0.5) / (slow * 1.15))
    return max(40, min(MAX_CHUNK_CHARS, fit))


def split_chunks(text: str, max_chars: int) -> list[str]:
    """Split ``text`` into chunks of at most ``max_chars``, on sentence ends
    first, then soft breaks (, ; : dashes), then whitespace, then hard cuts.
    Neighbouring pieces are packed greedily so short sentences share a chunk."""
    max_chars = max(1, int(max_chars))
    text = " ".join(text.split())
    if not text:
        return []
    pieces: list[str] = []
    for sentence in _SENTENCE_END.split(text):
        pieces.extend(_shrink(sentence.strip(), max_chars))
    chunks: list[str] = []
    for piece in pieces:
        if chunks and len(chunks[-1]) + 1 + len(piece) <= max_chars:
            chunks[-1] += " " + piece
        else:
            chunks.append(piece)
    return chunks


def _shrink(piece: str, max_chars: int) -> list[str]:
    if not piece:
        return []
    if len(piece) <= max_chars:
        return [piece]
    for splitter in (_SOFT_BREAK, re.compile(r"\s+")):
        parts = [p for p in splitter.split(piece) if p]
        if len(parts) > 1:
            out: list[str] = []
            for part in parts:
                # Re-join small parts up to the limit so a long sentence is not
                # cut into word-sized chunks.
                if out and len(out[-1]) + 1 + len(part) <= max_chars:
                    out[-1] += " " + part
                else:
                    out.extend(_shrink(part, max_chars) if len(part) > max_chars else [part])
            return out
    return [piece[i:i + max_chars] for i in range(0, len(piece), max_chars)]


class EosRamp:
    """Counts the sampling calls of one generation and, once more than ``start``
    frames have been produced, adds a growing bonus to the end-of-speech logit.
    A model that stops by itself is never touched. ``calls`` = frames + 1."""

    def __init__(self, eos_id: int, start: int, per_step: float = 1.5, max_bonus: float = 40.0):
        self.eos_id, self.start, self.per_step, self.max_bonus, self.calls = eos_id, start, per_step, max_bonus, 0

    @property
    def steps(self) -> int:
        return max(0, self.calls - 1)

    def bonus(self) -> float:
        over = self.steps - self.start
        return min(self.max_bonus, over * self.per_step) if over > 0 else 0.0

    def apply(self, logits):
        self.calls += 1
        b = self.bonus()
        if b > 0:
            logits = logits.clone()
            logits[..., self.eos_id] = logits[..., self.eos_id] + b
        return logits


def pad_tail(wave, sr: int, pad_s: float = 0.3, floor: float = 0.01):
    """Cut the silence after the last audible sample, then end the clip with
    exactly ``pad_s`` seconds of silence, so no ending is clipped and chunks
    join with an even gap."""
    import numpy as np  # noqa: PLC0415

    w = np.asarray(wave, dtype=np.float32).reshape(-1)
    loud = np.nonzero(np.abs(w) > floor)[0]
    body = w[: int(loud[-1]) + 1] if len(loud) else w[:0]
    return np.concatenate([body, np.zeros(int(round(pad_s * sr)), dtype=np.float32)])


def pad_head(wave, sr: int, lead_s: float | None = None, floor: float = 0.01):
    """Start of the clip: cut the silence before the first audible sample, then
    begin with exactly ``lead_s`` seconds of silence. ``lead_s=None`` (the
    default) returns the wave untouched. This only controls what the listener
    hears at the start; it does not change what the model generates (UNVERIFIED
    on a GPU as a cold-start remedy: a warm-up prefix of silence codec tokens
    needs library internals and is not implemented)."""
    import numpy as np  # noqa: PLC0415

    w = np.asarray(wave, dtype=np.float32).reshape(-1)
    if lead_s is None:
        return w
    loud = np.nonzero(np.abs(w) > floor)[0]
    body = w[int(loud[0]):] if len(loud) else w[:0]
    return np.concatenate([np.zeros(int(round(max(0.0, lead_s) * sr)), dtype=np.float32), body])
