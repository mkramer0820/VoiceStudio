"""Fish Audio story board routes.

Drafts, plans and the balance-free listing are local and never contact Fish.
``POST /render`` (explicit ``confirm``) and ``GET /balance`` are the only calls
that do, and both need the user's key.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Path
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from api.dependencies import require_admin
from core.browser_guard import reject_cross_site_get

logger = logging.getLogger("omnivoice.api.fish_story")

router = APIRouter(prefix="/api/fish", tags=["fish-audio"], dependencies=[Depends(require_admin)])

def _id():
    # A fresh Path per parameter: FastAPI binds the alias onto the instance.
    return Path(..., pattern=r"^[A-Za-z0-9_-]{1,64}$")


def _story():
    from engines.fish_api import story

    return story


def _draft(data):
    """Validate a draft dict; a bad one is a 422, not a 500."""
    from pydantic import ValidationError

    try:
        return _story().Draft.model_validate(data)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors(
            include_url=False, include_context=False, include_input=False))


class _RenderBody(BaseModel):
    confirm: bool = Field(..., description="Must be true: this sends text to Fish Audio and uses credits")
    draft_id: Optional[str] = Field(None, pattern=r"^[A-Za-z0-9_-]{1,64}$")
    chunk_ids: Optional[list[str]] = Field(None, max_length=3000)
    draft: Optional[dict] = Field(None, description="Inline draft; defaults to the saved draft_id")


@router.get("/balance", dependencies=[Depends(reject_cross_site_get)])
async def balance():
    """Fish API credit for the stored key. Explicit user action; contacts Fish."""
    from engines.fish_api import client
    from engines.fish_api.config import load_config, resolve_api_key

    key = resolve_api_key()
    if not key:
        raise HTTPException(status_code=502, detail="No Fish Audio API key.")
    try:
        return await run_in_threadpool(lambda: client.get_balance(api_key=key, cfg=load_config()))
    except client.FishAPIError as exc:
        raise HTTPException(status_code=502, detail=exc.message)


@router.get("/story/defaults", dependencies=[Depends(reject_cross_site_get)])
def defaults():
    """The effective settings (Settings/env) the board starts from."""
    import dataclasses

    from engines.fish_api.config import load_config

    cfg = dataclasses.asdict(load_config())
    cfg.pop("base_url", None)
    return cfg


@router.get("/story/drafts", dependencies=[Depends(reject_cross_site_get)])
def list_drafts():
    return {"drafts": _story().list_drafts()}


@router.post("/story/drafts")
def create_draft(draft: dict):
    story = _story()
    parsed = _draft(draft)
    return story.save_draft(story.new_draft_id(), parsed)


@router.get("/story/drafts/{draft_id}", dependencies=[Depends(reject_cross_site_get)])
def get_draft(draft_id: str = _id()):
    data = _story().load_draft(draft_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Draft not found")
    return data


@router.put("/story/drafts/{draft_id}")
def put_draft(draft: dict, draft_id: str = _id()):
    story = _story()
    return story.save_draft(draft_id, _draft(draft))


@router.delete("/story/drafts/{draft_id}")
def delete_draft(draft_id: str = _id()):
    if not _story().delete_draft(draft_id):
        raise HTTPException(status_code=404, detail="Draft not found")
    return {"deleted": draft_id}


@router.post("/story/plan")
def plan(draft: dict):
    """Counts and warnings for the staging summary. No network."""
    story = _story()
    return story.plan(_draft(draft))


@router.post("/story/render")
async def render(body: _RenderBody):
    story = _story()
    if not body.confirm:
        raise HTTPException(status_code=422, detail="confirm must be true")
    if body.draft is not None:
        draft = _draft(body.draft)
    elif body.draft_id:
        data = story.load_draft(body.draft_id)
        if data is None:
            raise HTTPException(status_code=404, detail="Draft not found")
        draft = _draft(data)
    else:
        raise HTTPException(status_code=422, detail="draft or draft_id required")
    try:
        return await run_in_threadpool(
            lambda: story.start_render(draft, draft_id=body.draft_id, chunk_ids=body.chunk_ids))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@router.get("/story/render/{job_id}", dependencies=[Depends(reject_cross_site_get)])
def render_status(job_id: str = _id()):
    job = _story().get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Render job not found")
    return job


@router.post("/story/render/{job_id}/cancel")
def render_cancel(job_id: str = _id()):
    if not _story().cancel_job(job_id):
        raise HTTPException(status_code=404, detail="Render job not found")
    return {"cancelling": job_id}
