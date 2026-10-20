"""Fish Audio story board routes.

Drafts, plans and the balance-free listing are local and never contact Fish.
``POST /render`` (explicit ``confirm``) and ``GET /balance`` are the only calls
that do, and both need the user's key.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query
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


def _jobs():
    from engines.fish_api import story_jobs

    return story_jobs


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


class _ResumeBody(BaseModel):
    confirm: bool = Field(..., description="Must be true: the unfinished lines are sent to Fish Audio")


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
    """The effective settings (Settings/env) the board starts from, plus where renders go."""
    import dataclasses

    from engines.fish_api.config import load_config

    cfg = load_config()
    out = dataclasses.asdict(cfg)
    out.pop("base_url", None)
    out["output_dir_default"] = _jobs().default_output_root()
    out["output_dir_effective"] = cfg.output_dir or out["output_dir_default"]
    return out


@router.put("/settings")
def save_settings(body: dict):
    """Save render options as the defaults for every story (``null`` clears one).

    Environment variables still override what is saved here.
    """
    from pydantic import ValidationError

    from engines.fish_api.config import store_key
    from services import settings_store

    try:
        parsed = _story().SavedSettings.model_validate(body)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors(
            include_url=False, include_context=False, include_input=False))
    names = {"default_voice": "reference_id"}
    skip = {"speed", "request_mode", "pause_ms", "join"}  # story-only options
    for field in parsed.model_fields_set - skip:
        value = getattr(parsed, field)
        if field == "output_dir" and value:
            try:
                value = _jobs().validate_output_dir(value)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc))
        key = store_key(names.get(field, field))
        if value is None or value == "":
            settings_store.clear_text(key)
        else:
            settings_store.set_text(key, str(value).lower() if isinstance(value, bool) else str(value))
    return defaults()


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


class _PreviewBody(BaseModel):
    draft: dict
    draft_id: Optional[str] = Field(None, max_length=64)
    chunk_ids: Optional[list[str]] = Field(None, max_length=3000)
    limit: int = Field(5, ge=1, le=50)


@router.post("/story/preview")
def preview(body: _PreviewBody):
    """Dry run: the exact requests a render would send, schema-checked. No network, no files."""
    try:
        return _jobs().preview(_draft(body.draft), chunk_ids=body.chunk_ids, limit=body.limit,
                               draft_id=body.draft_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


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
            lambda: _jobs().start_render(draft, draft_id=body.draft_id, chunk_ids=body.chunk_ids))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@router.get("/story/render/{job_id}", dependencies=[Depends(reject_cross_site_get)])
def render_status(job_id: str = _id()):
    job = _jobs().get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Render job not found")
    return job


@router.post("/story/render/{job_id}/cancel")
def render_cancel(job_id: str = _id()):
    if not _jobs().cancel_job(job_id):
        raise HTTPException(status_code=404, detail="Render job not found")
    return {"cancelling": job_id}


@router.get("/story/jobs", dependencies=[Depends(reject_cross_site_get)])
def list_jobs(limit: int = Query(200, ge=1, le=1000)):
    """Past and running renders, newest first. Reads local run folders only."""
    return {"jobs": _jobs().list_jobs(limit)}


@router.get("/story/jobs/{job_id}/log", dependencies=[Depends(reject_cross_site_get)])
def job_log(job_id: str = _id(), lines: int = Query(400, ge=1, le=5000)):
    text = _jobs().read_log(job_id, lines)
    if text is None:
        raise HTTPException(status_code=404, detail="Render job not found")
    return {"log": text}


@router.post("/story/jobs/{job_id}/resume")
async def resume_job(body: _ResumeBody, job_id: str = _id()):
    """Re-run only the lines that did not finish, in the same folder. Contacts Fish."""
    if not body.confirm:
        raise HTTPException(status_code=422, detail="confirm must be true")
    try:
        return await run_in_threadpool(lambda: _jobs().resume_render(job_id))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
