"""Qwen3-TTS studio routes: models, voices and the playground.

Everything is local: the sidecar runs on the user's GPU and nothing here contacts
the network. ``GET`` routes are cross-site guarded like the Fish routes.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pydantic import BaseModel, Field, ValidationError
from starlette.concurrency import run_in_threadpool

from api.dependencies import require_admin
from core.browser_guard import reject_cross_site_get

logger = logging.getLogger("omnivoice.api.qwen_studio")

router = APIRouter(prefix="/api/qwen", tags=["qwen3-tts"], dependencies=[Depends(require_admin)])


def _sid():
    # A fresh Path per parameter: FastAPI binds the alias onto the instance.
    return Path(..., pattern=r"^[A-Za-z0-9_.-]{1,120}$")


def _reg():
    from engines.qwen3_studio import registry

    return registry


def _pg():
    from engines.qwen3_studio import playground

    return playground


def _profiles() -> list[dict]:
    """Voice profiles that have a reference clip, i.e. what a clone can use."""
    from core.db import db_conn

    with db_conn() as conn:
        rows = conn.execute(
            "SELECT id, name, ref_text, language, kind FROM voice_profiles "
            "WHERE ref_audio_path IS NOT NULL AND ref_audio_path != '' ORDER BY created_at DESC"
        ).fetchall()
    return [{"id": r["id"], "name": r["name"], "ref_text": r["ref_text"] or "", "language": r["language"], "kind": r["kind"]}
            for r in rows]


@router.get("/state", dependencies=[Depends(reject_cross_site_get)])
def state():
    """Settings, whether the engine can start (and why not), models, and clone-capable voice profiles."""
    from engines import qwen3_narrator as engine

    reg = _reg()
    settings = reg.load_settings()
    engine.set_python_override(settings.get("python") or None)
    ok, reason = engine.availability(ignore_checkpoint=True)
    return {
        "settings": settings,
        "output_dir_default": reg.default_output_dir(),
        "engine": {"available": ok, "reason": reason},
        "models": reg.discover_models(settings),
        "profiles": _profiles(),
    }


class _SettingsBody(BaseModel):
    python: Optional[str] = Field(None, max_length=500)
    checkpoint_root: Optional[str] = Field(None, max_length=500)
    output_dir: Optional[str] = Field(None, max_length=500)
    warmup_runs: Optional[int] = Field(None, ge=0, le=10)
    warmup_text: Optional[str] = Field(None, max_length=500)


@router.put("/settings")
def save_settings(body: _SettingsBody):
    saved = _reg().save_settings(body.model_dump(exclude_unset=True))
    if "python" in body.model_fields_set and not _pg().is_running():
        _pg().unload_engine()  # the next run starts the sidecar with the new python
    return saved


@router.post("/playground/run")
async def run(body: dict):
    pg = _pg()
    try:
        req = pg.RunRequest.model_validate(body)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors(include_url=False, include_context=False, include_input=False))
    try:
        return await run_in_threadpool(pg.start_run, req)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except OSError as exc:
        raise HTTPException(status_code=422, detail=f"Cannot write to the output folder ({type(exc).__name__}).")


@router.get("/playground/jobs/{job_id}", dependencies=[Depends(reject_cross_site_get)])
def job(job_id: str = _sid()):
    data = _pg().get_job(job_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return data


@router.post("/playground/jobs/{job_id}/cancel")
def cancel(job_id: str = _sid()):
    if not _pg().cancel_job(job_id):
        raise HTTPException(status_code=404, detail="No running job with that id")
    return {"cancelled": job_id}


@router.post("/playground/unload")
def unload():
    """Stop the sidecar and give the GPU back."""
    _pg().unload_engine()
    return {"unloaded": True}


@router.get("/playground/sessions", dependencies=[Depends(reject_cross_site_get)])
def sessions(limit: int = Query(100, ge=1, le=500)):
    return {"sessions": _pg().list_sessions(limit=limit)}


@router.get("/playground/sessions/{session}", dependencies=[Depends(reject_cross_site_get)])
def session(session: str = _sid()):
    data = _pg().read_session(session)
    if data is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return data


class _AnnotateBody(BaseModel):
    notes: Optional[str] = Field(None, max_length=5000)
    starred: Optional[bool] = None


@router.patch("/playground/sessions/{session}/takes/{take_id}")
def annotate(body: _AnnotateBody, session: str = _sid(), take_id: str = Path(..., pattern=r"^[A-Za-z0-9_-]{1,40}$")):
    take = _pg().annotate_take(session, take_id, notes=body.notes, starred=body.starred)
    if take is None:
        raise HTTPException(status_code=404, detail="Take not found")
    return take


@router.get("/playground/sessions/{session}/audio/{name}", dependencies=[Depends(reject_cross_site_get)])
def audio(session: str = _sid(), name: str = Path(..., pattern=r"^[A-Za-z0-9._ -]{1,200}$")):
    from fastapi.responses import FileResponse

    path = _pg().audio_path(session, name)
    if path is None:
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(path, media_type="audio/wav")
