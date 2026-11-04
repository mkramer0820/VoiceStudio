"""Voice effects for Qwen story mode: named ffmpeg filter presets applied to a finished chunk (an inner monologue, a ghost, a demon).

Every preset is a plain ffmpeg ``-af`` chain built from filters every ffmpeg build ships (aecho, chorus, volume, lowpass, highpass,
equalizer, tremolo, acompressor, asetrate/atempo). The dry render is kept in ``dry/`` next to the story so an effect can be changed
without rendering again, and a failed effect leaves the dry take in place with a warning (never a missing chunk).
"""
from __future__ import annotations

import logging
import os
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)

NONE = "none"
TAIL = "apad=pad_dur=0.6"  # room for an echo tail; the chunk's own pad follows it


def _pitch(rate: int, factor: float) -> str:
    """Pitch shift by `factor` (0.8 = lower) without changing the duration: resample the clock, then restore the tempo."""
    return f"asetrate={int(rate * factor)},aresample={rate},atempo={1 / factor:.4f}"


def _presets(rate: int) -> dict[str, str]:
    return {
        # a thought: a little darker, a little quieter, a short soft echo
        "inner_monologue": f"highpass=f=120,lowpass=f=6500,volume=-4dB,{TAIL},aecho=0.8:0.6:45|90:0.25|0.15",
        "slight_echo": f"{TAIL},aecho=0.8:0.7:60:0.3",
        # floating: long fading echoes, a touch of chorus, thin low end
        "ethereal": (f"highpass=f=150,{TAIL},aecho=0.8:0.88:90|180|300:0.4|0.3|0.2,"
                     "chorus=0.6:0.9:50|60:0.4|0.32:0.25|0.4:2|1.3,lowpass=f=9000,volume=-2dB"),
        # a presence behind you: higher, hollow, long echo
        "ghost": f"{_pitch(rate, 1.10)},highpass=f=250,{TAIL},aecho=0.8:0.9:120|240:0.5|0.4,volume=-3dB",
        # low, heavy, a growl of tremolo and bass, a metallic double
        "demon": (f"{_pitch(rate, 0.80)},equalizer=f=110:t=q:w=1:g=6,tremolo=f=28:d=0.25,lowpass=f=5500,{TAIL},"
                  "aecho=0.8:0.9:35|70:0.5|0.3,acompressor=threshold=0.1:ratio=3"),
        "cave": f"{TAIL},aecho=0.8:0.9:200|400|650:0.5|0.35|0.25",
        "radio": "highpass=f=300,lowpass=f=3400,acompressor=threshold=0.1:ratio=4,volume=-1dB",
        "muffled": "lowpass=f=1200,volume=-5dB",
    }


PRESET_IDS = (NONE, "inner_monologue", "slight_echo", "ethereal", "ghost", "demon", "cave", "radio", "muffled")


def valid(effect: Optional[str]) -> bool:
    return effect in (None, "") or effect in PRESET_IDS


def filter_chain(effect: str, rate: int) -> str:
    chain = _presets(rate).get(effect)
    if chain is None:
        raise ValueError(f"Unknown effect {effect!r}. Choose one of: {', '.join(PRESET_IDS)}.")
    return chain


def resolve(chunk_effect: Optional[str], profile_effect: Optional[str]) -> str:
    """The effect of one chunk: its own choice, else its voice profile's, else none."""
    for value in (chunk_effect, profile_effect):
        if value and value != NONE:
            return value
    return NONE


def apply(src: str, dst: str, effect: str, rate: int, ffmpeg: Optional[str] = None, timeout: float = 120.0) -> None:
    """Write `src` through the preset to `dst` (PCM16 WAV, same rate, mono). Raises ValueError/RuntimeError with a plain reason."""
    chain = filter_chain(effect, rate)
    if ffmpeg is None:
        from services import ffmpeg_utils

        ffmpeg = ffmpeg_utils.find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg was not found (Settings -> Media tools), so the effect could not be applied.")
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", src, "-af", chain, "-ar", str(rate), "-ac", "1", "-c:a", "pcm_s16le", dst]
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ffmpeg took longer than {timeout:g} s on the effect {effect}.") from exc
    if done.returncode != 0 or not os.path.isfile(dst):
        reason = (done.stderr or "").strip().splitlines()
        raise RuntimeError(f"ffmpeg could not apply the effect {effect}: {reason[-1] if reason else 'unknown error'}")
