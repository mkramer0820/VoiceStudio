"""Qwen3-TTS studio routes: models, voices and the playground.

Everything is local: the sidecar runs on the user's GPU and nothing here contacts
the network. ``GET`` routes are cross-site guarded like the Fish routes.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request
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
    engine.set_python_override(reg.effective_python(settings))
    ok, reason = engine.availability(ignore_checkpoint=True)
    return {
        "settings": settings,
        "output_dir_default": reg.default_output_dir(),
        "python_default": reg.default_python(),
        "engine": {"available": ok, "reason": reason},
        "models": reg.discover_models(settings),
        "stock_speakers": list(reg.STOCK_SPEAKERS),
        "profiles": _profiles(),
    }


class _SettingsBody(BaseModel):
    python: Optional[str] = Field(None, max_length=500)
    #: Accepted for old clients and ignored: model folders live in Settings -> Storage -> Model locations.
    checkpoint_root: Optional[str] = Field(None, max_length=500)
    output_dir: Optional[str] = Field(None, max_length=500)
    warmup_runs: Optional[int] = Field(None, ge=0, le=10)
    warmup_text: Optional[str] = Field(None, max_length=500)


@router.put("/settings")
def save_settings(body: _SettingsBody):
    """Save python, output folder and warm-up settings. ``checkpoint_root`` is accepted but ignored (see
    ``/api/settings/storage/model-locations``)."""
    saved = _reg().save_settings(body.model_dump(exclude_unset=True))
    if "python" in body.model_fields_set and not _pg().is_running():
        _pg().unload_engine()  # the next run starts the sidecar with the new python
    return saved


class _MergeBody(BaseModel):
    model_id: str = Field(..., pattern=r"^[A-Za-z0-9_.-]{1,160}$")
    #: A row of Settings -> Storage -> Model locations. No free path is accepted here.
    location_id: str = Field(..., pattern=r"^[A-Za-z0-9_.-]{1,80}$")
    name: str = Field("", max_length=80)
    #: Multiplier on alpha/r at merge time. 1.0 = as trained; 0.3 is an unconfirmed candidate.
    lora_scale: float = Field(1.0, gt=0, le=2)


@router.post("/models/merge-adapter")
async def merge_adapter(body: _MergeBody):
    """Merge a LoRA adapter into its Base model (CPU, a minute or two) and write a normal custom_voice checkpoint."""
    from engines.fish_api.qwen3_colab_runner import StepFailed

    try:
        return await run_in_threadpool(_reg().merge_adapter_model, body.model_id, body.location_id, body.name, body.lora_scale)
    except (ValueError, StepFailed) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except OSError as exc:
        raise HTTPException(status_code=422, detail=f"Cannot write the merged model ({type(exc).__name__}).")


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


# ── voice profiles, emotions, story mode ─────────────────────────────────
def _prof():
    from engines.qwen3_studio import profiles

    return profiles


def _story():
    from engines.qwen3_studio import story

    return story


def _validation(exc: ValidationError):
    return HTTPException(status_code=422, detail=exc.errors(include_url=False, include_context=False, include_input=False))


@router.get("/emotions", dependencies=[Depends(reject_cross_site_get)])
def emotion_presets():
    """Named emotion presets (instruct text per strength). Instruct strength is UNVERIFIED on the fine-tuned narrator."""
    from engines.qwen3_studio import emotions

    return emotions.catalog()


@router.get("/profiles", dependencies=[Depends(reject_cross_site_get)])
def list_profiles():
    return {"profiles": _prof().list_all()}


@router.post("/clips")
async def upload_clip(request: Request, name: str = Query("", max_length=200)):
    """Store a reference clip (uploaded or recorded; the body is the audio file) and return its path for a clone profile."""
    try:
        return {"path": await run_in_threadpool(_prof().store_clip, await request.body(), name)}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.post("/profiles/{profile_id}/prepare")
def prepare_profile(profile_id: str = Path(..., pattern=r"^[a-f0-9]{12}$")):
    """Prepare the voice now if the engine is idle; otherwise it is prepared before the first render that uses it."""
    prof = _prof()
    if prof.get(profile_id) is None:
        raise HTTPException(status_code=404, detail="Profile not found")
    prof.prepare_in_background(profile_id)
    return prof.get(profile_id)


@router.post("/profiles")
def create_profile(body: dict):
    prof = _prof()
    try:
        made = prof.create(prof.ProfileBody.model_validate(body))
        prof.prepare_in_background(made["id"])
        return prof.get(made["id"])
    except ValidationError as exc:
        raise _validation(exc)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


class _FromTakeBody(BaseModel):
    session: str = Field(..., pattern=r"^[A-Za-z0-9_.-]{1,120}$")
    take_id: str = Field(..., pattern=r"^[A-Za-z0-9_-]{1,40}$")
    name: str = Field(..., min_length=1, max_length=80)


@router.post("/profiles/from-take")
def profile_from_take(body: _FromTakeBody):
    try:
        return _prof().from_take(body.session, body.take_id, body.name)
    except KeyError:
        raise HTTPException(status_code=404, detail="Take not found")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


def _pid():
    return Path(..., pattern=r"^[a-f0-9]{12}$")


@router.get("/profiles/{profile_id}", dependencies=[Depends(reject_cross_site_get)])
def get_profile(profile_id: str = _pid()):
    data = _prof().get(profile_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Profile not found")
    return data


@router.put("/profiles/{profile_id}")
def update_profile(body: dict, profile_id: str = _pid()):
    prof = _prof()
    try:
        data = prof.update(profile_id, prof.ProfileBody.model_validate(body))
    except ValidationError as exc:
        raise _validation(exc)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    if data is None:
        raise HTTPException(status_code=404, detail="Profile not found")
    prof.prepare_in_background(profile_id)
    return prof.get(profile_id)


@router.delete("/profiles/{profile_id}")
def delete_profile(profile_id: str = _pid()):
    if not _prof().delete(profile_id):
        raise HTTPException(status_code=404, detail="Profile not found")
    return {"deleted": profile_id}


@router.put("/profiles/{profile_id}/embedding")
async def import_embedding(request: Request, profile_id: str = _pid()):
    """Import a raw float32[2048] speaker embedding (the body is the .bin file). Anything else is refused."""
    raw = await request.body()
    try:
        return _prof().import_embedding(profile_id, raw)
    except KeyError:
        raise HTTPException(status_code=404, detail="Profile not found")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.get("/profiles/{profile_id}/embedding", dependencies=[Depends(reject_cross_site_get)])
def export_embedding(profile_id: str = _pid()):
    from fastapi.responses import Response

    raw = _prof().export_embedding(profile_id)
    if raw is None:
        raise HTTPException(status_code=404, detail="This profile has no cached embedding yet")
    return Response(raw, media_type="application/octet-stream",
                    headers={"Content-Disposition": "attachment; filename=" + profile_id + ".bin"})


class _ParseBody(BaseModel):
    text: str = Field(..., max_length=2_000_000)
    max_chars: Optional[int] = Field(None, ge=20, le=2000)


@router.post("/story/parse")
def story_parse(body: _ParseBody):
    """Split pasted text into chunks, detect speaker and emotion tags. Nothing is saved."""
    story = _story()
    chunks = story.parse_script(body.text, body.max_chars)
    return {"chunks": chunks, "characters": story.characters(chunks)}


@router.get("/story/drafts", dependencies=[Depends(reject_cross_site_get)])
def story_drafts():
    return {"stories": _story().list_drafts()}


@router.post("/story/drafts")
def story_create(body: dict):
    story = _story()
    try:
        return story.save_draft(None, story.Draft.model_validate(body))
    except ValidationError as exc:
        raise _validation(exc)


@router.put("/story/drafts/{story_id}")
def story_save(body: dict, story_id: str = _pid()):
    story = _story()
    try:
        return story.save_draft(story_id, story.Draft.model_validate(body))
    except ValidationError as exc:
        raise _validation(exc)


@router.get("/story/drafts/{story_id}", dependencies=[Depends(reject_cross_site_get)])
def story_load(story_id: str = _pid()):
    data = _story().load_draft(story_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Story not found")
    return data


@router.delete("/story/drafts/{story_id}")
def story_delete(story_id: str = _pid()):
    if not _story().delete_draft(story_id):
        raise HTTPException(status_code=404, detail="Story not found")
    return {"deleted": story_id}


class _RenderBody(BaseModel):
    story_id: Optional[str] = Field(None, pattern=r"^[a-f0-9]{12}$")
    draft: dict
    chunk_ids: Optional[list[str]] = Field(None, max_length=5000)
    run: Optional[str] = Field(None, max_length=80)  # re-render chunk_ids into this existing run folder


@router.post("/story/render")
async def story_render(body: _RenderBody):
    story = _story()
    try:
        draft = story.Draft.model_validate(body.draft)
    except ValidationError as exc:
        raise _validation(exc)
    try:
        return await run_in_threadpool(story.start_render, draft, body.story_id, body.chunk_ids, body.run)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except OSError as exc:
        raise HTTPException(status_code=422, detail=f"Cannot write to the output folder ({type(exc).__name__}).")


@router.get("/story/jobs/{job_id}", dependencies=[Depends(reject_cross_site_get)])
def story_job(job_id: str = _pid()):
    data = _story().get_job(job_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Render not found")
    return data


@router.post("/story/jobs/{job_id}/cancel")
def story_cancel(job_id: str = _pid()):
    if not _story().cancel_job(job_id):
        raise HTTPException(status_code=404, detail="No running render with that id")
    return {"cancelled": job_id}


_SLUG = r"^[a-z0-9][a-z0-9-]{0,60}$"
_RUN = r"^run-\d{3}_\d{8}-\d{6}_[a-f0-9]{12}$"


@router.get("/story/runs/{slug}", dependencies=[Depends(reject_cross_site_get)])
def story_runs(slug: str = Path(..., pattern=_SLUG)):
    return {"runs": _story().list_runs(slug)}


@router.get("/story/runs/{slug}/{run}", dependencies=[Depends(reject_cross_site_get)])
def story_run(slug: str = Path(..., pattern=_SLUG), run: str = Path(..., pattern=_RUN)):
    data = _story().read_run(slug, run)
    if data is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return data


@router.get("/story/runs/{slug}/{run}/audio/{name}", dependencies=[Depends(reject_cross_site_get)])
def story_audio(slug: str = Path(..., pattern=_SLUG), run: str = Path(..., pattern=_RUN),
                name: str = Path(..., pattern=r"^[A-Za-z0-9_.-]{1,200}$")):
    from fastapi.responses import FileResponse

    path = _story().audio_path(slug, run, name)
    if path is None:
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(path, media_type="audio/wav")
