"""Emotion presets for Qwen3-TTS custom_voice profiles.

A preset is plain-English instruct text in three strengths (normal, strong, very strong), written in the
vivid-prose style the CustomVoice documentation recommends. It is applied as ``instruct``.

UNVERIFIED on the fine-tuned narrator: on EP09 an instruction changed the length of the take and moved the
voice away from the narrator (findings section 6, heard). Presets are meant for stock CustomVoice
models and for experiments; the narrator itself is normally rendered with no instruct.

``steer`` (a steering-vector id and weight) is reserved so the later steering hook needs no schema change.
It is accepted and ignored with a clear message; nothing here loads or applies a vector.
"""
from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel, Field

STRENGTHS = ("normal", "strong", "very_strong")
UNVERIFIED_NARRATOR = ("Instruct strength is UNVERIFIED on the fine-tuned narrator: instructions changed its length and moved the "
                       "voice away from the narrator. Leave the preset on 'neutral' for narrator profiles unless you are testing.")
STEER_UNAVAILABLE = "Emotion steering is not available yet; the steer field is accepted but ignored."

PRESETS: dict[str, dict[str, str]] = {
    "neutral": {s: "" for s in STRENGTHS},
    "sad": {
        "normal": "Speak in a sad, subdued tone, with low energy and a slight heaviness in the voice.",
        "strong": "Speak in a strongly sad voice: slow, heavy, with a trembling softness, as if holding back tears.",
        "very_strong": "Speak in a devastated, grief-stricken voice: very slow and broken, thick with sorrow, the words barely held together.",
    },
    "joy": {
        "normal": "Speak cheerfully, with a bright, warm and lively tone.",
        "strong": "Speak with strong delight: a bright, bouncy, smiling voice full of energy and warmth.",
        "very_strong": "Speak with overflowing joy: an elated, laughing, breathless voice, rising and fast with excitement.",
    },
    "anger": {
        "normal": "Speak in an irritated, firm tone, with sharp emphasis.",
        "strong": "Speak in an angry, harsh voice with a raised volume and clipped, forceful words.",
        "very_strong": "Speak in a furious, shouting voice: explosive, harsh and shaking with rage, every word spat out.",
    },
    "fear": {
        "normal": "Speak nervously, with a tight, uneasy tone and a little tension in the voice.",
        "strong": "Speak in a frightened voice: quick, shaky and hushed, breath catching between words.",
        "very_strong": "Speak in a terrified, panicked voice: trembling, high and breathless, words tumbling out in dread.",
    },
    "disgust": {
        "normal": "Speak with mild distaste, in a flat, slightly curled tone.",
        "strong": "Speak with clear disgust: a sneering, repulsed voice that drags on the harsh sounds.",
        "very_strong": "Speak with revulsion: a gagging, contemptuous voice, every word dripping with loathing.",
    },
    "surprise": {
        "normal": "Speak with mild surprise, with a lifted pitch and an open, curious tone.",
        "strong": "Speak in a startled voice: a sudden rise in pitch, wide-eyed and breathless.",
        "very_strong": "Speak in utter astonishment: a shocked, gasping voice jumping high, hardly believing what it says.",
    },
    "custom": {s: "" for s in STRENGTHS},
}

#: Inline tags in book text, e.g. ``[sad]`` or ``[joy:strong]``, switch the preset for that span.
_TAG = re.compile(r"\[(" + "|".join(k for k in PRESETS if k != "custom") + r")(?::(strong|very[ _-]?strong))?\]", re.I)


class Steer(BaseModel):
    """Reserved for steering vectors. Accepted, ignored."""
    vector_id: Optional[str] = Field(None, max_length=120)
    weight: Optional[float] = Field(None, ge=0.0, le=100.0)


class EmotionChoice(BaseModel):
    preset: str = Field("neutral", max_length=20)
    strength: str = "normal"
    custom_text: Optional[str] = Field(None, max_length=1000)  # used when preset == 'custom'
    steer: Optional[Steer] = None


def normalize_strength(value: Optional[str]) -> str:
    key = re.sub(r"[ -]+", "_", (value or "normal").strip().lower())
    if key not in STRENGTHS:
        raise ValueError(f"Unknown strength {value!r}; use normal, strong or very_strong.")
    return key


def instruct_for(choice: Optional[EmotionChoice]) -> tuple[str, list[str]]:
    """(instruct text, warnings). Neutral gives an empty instruct."""
    if choice is None:
        return "", []
    warnings: list[str] = []
    if choice.steer is not None and (choice.steer.vector_id or choice.steer.weight is not None):
        warnings.append(STEER_UNAVAILABLE)
    preset = (choice.preset or "neutral").lower()
    if preset not in PRESETS:
        raise ValueError(f"Unknown emotion preset {choice.preset!r}; use one of {', '.join(PRESETS)}.")
    if preset == "custom":
        return (choice.custom_text or "").strip(), warnings
    return PRESETS[preset][normalize_strength(choice.strength)], warnings


def catalog() -> dict:
    return {"presets": [{"name": n, "texts": t} for n, t in PRESETS.items()], "strengths": list(STRENGTHS),
            "unverified_on_narrator": UNVERIFIED_NARRATOR, "steer_message": STEER_UNAVAILABLE}


def split_emotion_tags(text: str) -> list[tuple[Optional[EmotionChoice], str]]:
    """Cut ``text`` at inline emotion tags. Each piece carries the preset that was switched on before it
    (None before the first tag). A tag stays active until the next tag; ``[neutral]`` switches back. Empty
    pieces are dropped, so a tag at the end of text does nothing."""
    out: list[tuple[Optional[EmotionChoice], str]] = []
    current: Optional[EmotionChoice] = None
    cursor = 0
    for m in _TAG.finditer(text):
        piece = text[cursor:m.start()].strip()
        if piece:
            out.append((current, piece))
        current = EmotionChoice(preset=m.group(1).lower(), strength=normalize_strength(m.group(2)))
        cursor = m.end()
    tail = text[cursor:].strip()
    if tail:
        out.append((current, tail))
    return out
