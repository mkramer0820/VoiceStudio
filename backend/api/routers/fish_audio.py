"""Fish Audio voices: list the account's saved voices and choose a default.

Both routes are explicit user actions. Listing contacts Fish with the stored key;
nothing here runs on startup or while checking engine availability.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query
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


class _KeyBody(BaseModel):
    api_key: str = Field(..., min_length=8, max_length=512, pattern=r"^\S+$",
                         description="Fish Audio API key; stored encrypted, never returned")


def _key_status() -> dict:
    import os

    from engines.fish_api import config as fish_config
    from services import settings_store

    env = bool((os.environ.get("FISH_API_KEY") or "").strip())
    stored = bool(settings_store.get_secret(fish_config.SECRET_NAME))
    return {"has_key": env or stored, "stored": stored, "from_env": env}


@router.get("/key", dependencies=[Depends(reject_cross_site_get)])
def get_key_status():
    """Whether a key is set. The key itself is never returned."""
    return _key_status()


@router.put("/key")
def set_key(body: _KeyBody):
    """Store the key in the encrypted secret store (FISH_API_KEY env still wins)."""
    from engines.fish_api import config as fish_config
    from services import settings_store

    settings_store.set_secret(fish_config.SECRET_NAME, body.api_key.strip())
    return _key_status()


@router.delete("/key")
def delete_key():
    from engines.fish_api import config as fish_config
    from services import settings_store

    settings_store.clear_secret(fish_config.SECRET_NAME)
    return _key_status()


def _voices(mine: bool, title: Optional[str], language: Optional[str]) -> list[dict]:
    from engines.fish_api import FishAudioBackend

    return FishAudioBackend().list_voices(mine=mine, title=title, language=language,
                                          one_page=not mine)


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
    if scope == "mine":  # your own account's voices are always worth remembering
        _remember(items, "mine")
    return {"voices": items, "default_voice_id": load_config().reference_id,
            "count": len(items), "scope": scope}


def _remember(voices: list[dict], source: str) -> None:
    from engines.fish_api import library

    try:
        library.remember(voices, source)
    except OSError:
        logger.warning("Fish voice library: could not save", exc_info=True)


class _LibraryEntry(BaseModel):
    alias: Optional[str] = Field(None, max_length=60, description="Your name for the character")
    notes: Optional[str] = Field(None, max_length=1000)
    fish: Optional[dict] = Field(None, description="Details Fish gave for the voice (title, description, languages…)")


@router.get("/library", dependencies=[Depends(reject_cross_site_get)])
def library_list():
    """Every voice you have saved, used or named, newest use first. Local only."""
    from engines.fish_api import library

    return {"voices": library.list_all()}


@router.put("/library/{voice_id}")
def library_put(body: _LibraryEntry, voice_id: str = Path(..., pattern=r"^[A-Za-z0-9_-]{1,128}$")):
    from engines.fish_api import library

    return library.update(voice_id, alias=body.alias, notes=body.notes, fish=body.fish)


@router.delete("/library/{voice_id}")
def library_delete(voice_id: str = Path(..., pattern=r"^[A-Za-z0-9_-]{1,128}$")):
    from engines.fish_api import library

    if not library.forget(voice_id):
        raise HTTPException(status_code=404, detail="Voice not in your library")
    return {"deleted": voice_id}


class _CollectionBody(BaseModel):
    url: str = Field(..., min_length=8, max_length=300,
                     description="A public fish.audio/collections/<id> link, or the 32-character id")


@router.post("/collection")
async def import_collection(body: _CollectionBody):
    """Read a public collection's voices from its fish.audio web page (explicit action).

    Fish's API cannot list collections or bookmarks, so this fetches the public page the
    user pasted. No API key or cookie is sent; private pages cannot be read.
    """
    from engines.fish_api import collection

    try:
        return await run_in_threadpool(collection.fetch_collection, body.url)
    except collection.CollectionError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.put("/default-voice")
def set_default_voice(body: _DefaultVoice):
    """Store the voice used when a request names none (env FISH_TTS_REFERENCE_ID still wins)."""
    from services import settings_store

    if body.voice_id:
        settings_store.set_text(_DEFAULT_VOICE_KEY, body.voice_id)
    else:
        settings_store.clear_text(_DEFAULT_VOICE_KEY)
    return {"default_voice_id": body.voice_id or None}
