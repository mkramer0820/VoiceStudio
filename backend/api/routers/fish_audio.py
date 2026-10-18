"""Fish Audio voices: list the account's saved voices and choose a default.

Both routes are explicit user actions. Listing contacts Fish with the stored key;
nothing here runs on startup or while checking engine availability.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from api.dependencies import require_admin
from core.browser_guard import reject_cross_site_get

logger = logging.getLogger("omnivoice.api.fish_audio")

router = APIRouter(
    prefix="/api/fish",
    tags=["fish-audio"],
    dependencies=[Depends(require_admin)],
)

_DEFAULT_VOICE_KEY = "tts.fish.reference_id"


class _DefaultVoice(BaseModel):
    voice_id: Optional[str] = Field(None, max_length=128, pattern=r"^[A-Za-z0-9_-]+$",
                                    description="Fish voice id; null clears the default")


def _voices(mine: bool, title: Optional[str], language: Optional[str]) -> list[dict]:
    from engines.fish_api import FishAudioBackend

    return FishAudioBackend().list_voices(mine=mine, title=title, language=language)


@router.get("/voices", dependencies=[Depends(reject_cross_site_get)])
async def list_voices(
    scope: str = Query("mine", pattern="^(mine|library)$"),
    title: Optional[str] = Query(None, max_length=100),
    language: Optional[str] = Query(None, max_length=35),
):
    """``scope=mine`` lists your saved voices; ``scope=library`` searches the public library."""
    from engines.fish_api.config import load_config

    try:
        items = await run_in_threadpool(_voices, scope == "mine", title, language)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return {"voices": items, "default_voice_id": load_config().reference_id,
            "count": len(items), "scope": scope}


@router.put("/default-voice")
def set_default_voice(body: _DefaultVoice):
    """Store the voice used when a request names none (env FISH_TTS_REFERENCE_ID still wins)."""
    from services import settings_store

    if body.voice_id:
        settings_store.set_text(_DEFAULT_VOICE_KEY, body.voice_id)
    else:
        settings_store.clear_text(_DEFAULT_VOICE_KEY)
    return {"default_voice_id": body.voice_id or None}
