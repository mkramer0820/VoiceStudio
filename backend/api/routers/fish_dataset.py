"""Fish training-data browser: index the story renders already on disk, play them, export a corpus.

Local files only. Nothing here contacts Fish. ``POST /export`` writes a new folder and refuses
to touch an existing non-empty one.
"""
from __future__ import annotations

import collections
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from api.dependencies import require_admin
from core.browser_guard import reject_cross_site_get

logger = logging.getLogger("omnivoice.api.fish_dataset")

router = APIRouter(prefix="/api/fish/dataset", tags=["fish-audio"], dependencies=[Depends(require_admin)])


def _ds():
    from engines.fish_api import dataset

    return dataset


class _Root(BaseModel):
    path: Optional[str] = Field(None, max_length=400, description="Folder that holds the chapter folders; null clears")


class _Filter(BaseModel):
    voice_ids: list[str] = Field(default_factory=list, max_length=300)
    chapters: list[str] = Field(default_factory=list, max_length=500)
    q: str = Field("", max_length=200)
    tag: str = Field("", max_length=40)
    min_dur: Optional[float] = Field(None, ge=0, le=600)
    max_dur: Optional[float] = Field(None, ge=0, le=600)
    min_words: Optional[int] = Field(None, ge=0, le=5000)
    duplicates: bool = False
    exclude_keys: list[str] = Field(default_factory=list, max_length=20000)


class _Export(_Filter):
    dest: str = Field(..., min_length=3, max_length=400)
    name: str = Field("fish-dataset", min_length=1, max_length=60)
    formats: list[str] = Field(default_factory=lambda: ["jsonl"], max_length=3)
    text_mode: str = Field("keep", pattern="^(keep|strip)$")
    sample_rate: Optional[int] = Field(None, ge=8000, le=96000)
    val_fraction: float = Field(0.0, ge=0.0, lt=0.5)
    keep_tags: list[str] = Field(default_factory=list, max_length=100, description="Tags left in the text when text_mode is strip")


def _corpus(refresh: bool = False):
    corpus = _ds().load(refresh=refresh)
    if corpus is None:
        raise HTTPException(status_code=404, detail="Set the folder that holds your chapter folders first.")
    return corpus


@router.get("", dependencies=[Depends(reject_cross_site_get)])
async def overview(refresh: bool = False):
    """Saved folder, speakers (grouped by voice id), runs and totals."""
    ds = _ds()
    root = ds.get_root()
    if not root:
        return {"root": None, "exists": False, "speakers": [], "runs": [], "chapters": [], "totals": None}
    corpus = await run_in_threadpool(ds.load, root, refresh=refresh)
    if corpus is None:
        return {"root": root, "exists": False, "speakers": [], "runs": [], "chapters": [], "totals": None}
    usable = [l for l in corpus.lines if not l["duplicate"]]
    tag_counts = collections.Counter(t for l in usable for t in l["tags"])
    return {
        "root": root, "exists": True, "speakers": corpus.speakers(), "runs": corpus.runs,
        "tags": tag_counts.most_common(80),
        "chapters": sorted({l["chapter"] for l in corpus.lines}),
        "totals": {"lines": len(usable), "duplicates": len(corpus.lines) - len(usable),
                   "seconds": round(sum(l["duration_s"] for l in usable), 1), "runs": len(corpus.runs)},
    }


@router.put("/root")
def set_root(body: _Root):
    try:
        return {"root": _ds().set_root(body.path)}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.post("/lines")
async def lines(body: _Filter, offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=500)):
    """A page of matching clips. POST because the filter can carry thousands of excluded keys."""
    ds = _ds()
    corpus = _corpus()
    matched = await run_in_threadpool(ds.select, corpus, body.model_dump())
    keep = ("key", "chapter", "run", "index", "voice_id", "voice_name", "text", "plain", "tags",
            "words", "duration_s", "duplicate")
    return {"total": len(matched), "seconds": round(sum(l["duration_s"] for l in matched), 1),
            "offset": offset, "lines": [{k: l[k] for k in keep} for l in matched[offset:offset + limit]]}


@router.get("/audio", dependencies=[Depends(reject_cross_site_get)])
def audio(key: str = Query(..., min_length=3, max_length=120)):
    path = _ds().audio_path(_corpus(), key)
    if path is None:
        raise HTTPException(status_code=404, detail="Clip not found")
    return FileResponse(path, media_type="audio/wav")


@router.post("/export")
async def export(body: _Export):
    """Write the selected clips and metadata into a new folder (explicit action; local only)."""
    ds = _ds()
    corpus = _corpus()
    spec = body.model_dump()
    try:
        return await run_in_threadpool(
            lambda: ds.export(corpus, spec, body.dest, formats=body.formats, text_mode=body.text_mode,
                              sample_rate=body.sample_rate, val_fraction=body.val_fraction, name=body.name,
                              keep_tags=body.keep_tags))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not write the export ({type(exc).__name__}).")
