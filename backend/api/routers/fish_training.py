"""Fish training: turn an OmniVoice export into a LoRA adapter, locally.

Nothing here contacts Fish or any service. Training is opt-in (``PUT /enabled``), a real run also needs
``confirm: true``, and a *dry run* (``POST /dry-run``) validates the export, builds every config and
command and probes the hardware without a GPU, a model download or a training step. The work itself
runs in child processes (see ``engines/fish_api/training.py``) so a stuck GPU job can be cancelled.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from api.dependencies import require_admin
from core.browser_guard import reject_cross_site_get

logger = logging.getLogger("omnivoice.api.fish_training")

router = APIRouter(prefix="/api/fish/training", tags=["fish-audio"], dependencies=[Depends(require_admin)])


def _t():
    from engines.fish_api import training

    return training


def _id():
    return Path(..., pattern=r"^[A-Za-z0-9_-]{1,64}$")


class _Enabled(BaseModel):
    enabled: bool


class _Start(BaseModel):
    export_dir: str = Field(..., min_length=3, max_length=400, description="Folder made by the OmniVoice export format")
    name: str = Field("", max_length=60)
    options: dict = Field(default_factory=dict, description="TrainOptions overrides; see GET / for the defaults")


class _Run(_Start):
    confirm: bool = Field(..., description="Must be true: this uses the GPU for a long time and writes to the data folder")


class _Resume(BaseModel):
    confirm: bool = Field(..., description="Must be true: training continues from the last checkpoint")


def _fail(exc: Exception) -> HTTPException:
    if isinstance(exc, PermissionError):
        return HTTPException(status_code=403, detail=str(exc))
    return HTTPException(status_code=422, detail=str(exc))


@router.get("", dependencies=[Depends(reject_cross_site_get)])
def overview():
    """Whether training is switched on, the default options and the recent runs."""
    t = _t()
    return {"enabled": t.is_enabled(), "defaults": t.defaults(), "runs": t.list_runs(),
            "limits": {"min_clips": t.MIN_CLIPS_BLOCK, "comfortable_clips": t.MIN_CLIPS_WARN,
                       "comfortable_minutes": t.MIN_MINUTES_WARN, "vram_gb": t.VRAM_GB_COMFORT}}


@router.put("/enabled")
def set_enabled(body: _Enabled):
    return {"enabled": _t().set_enabled(body.enabled)}


@router.post("/dry-run")
async def dry_run(body: _Start):
    """Validate everything a run needs and show what would happen. Starts no training."""
    t = _t()
    try:
        job = await run_in_threadpool(lambda: t.start_run(body.export_dir, body.options, name=body.name, dry_run=True))
    except (ValueError, PermissionError) as exc:
        raise _fail(exc)
    return {**job, "report": t.dry_run_report(job["id"])}


@router.post("/runs")
async def start(body: _Run):
    if not body.confirm:
        raise HTTPException(status_code=422, detail="Confirm that you want to start training.")
    t = _t()
    try:
        return await run_in_threadpool(lambda: t.start_run(body.export_dir, body.options, name=body.name))
    except (ValueError, PermissionError) as exc:
        raise _fail(exc)


@router.get("/runs", dependencies=[Depends(reject_cross_site_get)])
def runs(limit: int = Query(100, ge=1, le=500)):
    return {"runs": _t().list_runs(limit)}


@router.get("/runs/{job_id}", dependencies=[Depends(reject_cross_site_get)])
def run(job_id: str = _id()):
    t = _t()
    job = t.get_run(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return {**job, "report": t.dry_run_report(job_id) if job.get("dry_run") else None}


@router.get("/runs/{job_id}/log", dependencies=[Depends(reject_cross_site_get)])
def log(job_id: str = _id(), lines: int = Query(400, ge=1, le=2000)):
    text: Optional[str] = _t().read_log(job_id, lines)
    if text is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"log": text}


@router.post("/runs/{job_id}/cancel")
def cancel(job_id: str = _id()):
    if not _t().cancel_run(job_id):
        raise HTTPException(status_code=409, detail="That run is not running.")
    return {"cancelling": True}


@router.post("/runs/{job_id}/resume")
async def resume(body: _Resume, job_id: str = _id()):
    if not body.confirm:
        raise HTTPException(status_code=422, detail="Confirm that you want to resume training.")
    try:
        return await run_in_threadpool(_t().resume_run, job_id)
    except (ValueError, PermissionError) as exc:
        raise _fail(exc)
