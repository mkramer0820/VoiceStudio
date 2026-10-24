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
from engines.fish_api import dataset_quality

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
    max_words: Optional[int] = Field(None, ge=0, le=5000)
    duplicates: bool = False
    exclude_keys: list[str] = Field(default_factory=list, max_length=20000)
    only_keys: Optional[list[str]] = Field(None, max_length=20000, description="Restrict to these clips (e.g. the excluded ones)")
    tags_any: list[str] = Field(default_factory=list, max_length=100)
    tags_all: list[str] = Field(default_factory=list, max_length=100)
    tags_none: list[str] = Field(default_factory=list, max_length=100)
    sort: str = Field("order", pattern="^(order|duration_desc|duration_asc|words_desc|words_asc|random)$")
    seed: int = Field(0, ge=0, le=2**31)
    # Audio-quality limits (need the clips analysed first)
    max_peak: Optional[float] = Field(None, ge=0, le=1)
    min_peak: Optional[float] = Field(None, ge=0, le=1)
    max_clip: Optional[float] = Field(None, ge=0, le=1)
    min_rms_db: Optional[float] = Field(None, ge=-120, le=0)
    max_rms_db: Optional[float] = Field(None, ge=-120, le=0)
    max_lead_ms: Optional[int] = Field(None, ge=0, le=60000)
    max_tail_ms: Optional[int] = Field(None, ge=0, le=60000)
    max_pauses: Optional[int] = Field(None, ge=0, le=500)
    max_gap_ms: Optional[int] = Field(None, ge=0, le=60000)


class _Qwen3(BaseModel):
    speaker_name: str = Field("", max_length=40, description="Name the fine-tuned model answers to; empty uses the voice name")
    reference_key: Optional[str] = Field(None, max_length=120, description="Clip to use as the reference; null chooses automatically")
    reference_file: Optional[str] = Field(None, max_length=400, description="A local audio file to use as the reference instead")
    make_zip: bool = False
    overwrite: bool = False
    min_duration: float = Field(0.8, ge=0.1, le=120)
    max_duration: float = Field(30.0, ge=0.2, le=120)


class _Export(_Filter):
    dest: str = Field(..., min_length=3, max_length=400)
    name: str = Field("fish-dataset", min_length=1, max_length=60)
    formats: list[str] = Field(default_factory=lambda: ["jsonl"], max_length=5)
    text_mode: str = Field("keep", pattern="^(keep|strip)$")
    sample_rate: Optional[int] = Field(None, ge=8000, le=96000)
    val_fraction: float = Field(0.0, ge=0.0, lt=0.5)
    keep_tags: list[str] = Field(default_factory=list, max_length=100, description="Tags left in the text when text_mode is strip")
    loudness_db: Optional[float] = Field(None, ge=-60, le=-3, description="Bring every clip to this RMS level (dBFS); null leaves levels alone")
    language_id: str = Field("en", min_length=2, max_length=12, description="language_id written by the omnivoice format")
    strict_tags: bool = Field(True, description="omnivoice format: refuse tags OmniVoice does not know")
    training: Optional[dict] = Field(None, description="omnivoice format: also write train_config.json, data_config.json, train.sh and COLAB.md using these settings. qwen3_tts format: epochs, learning_rate, batch_size, base_model, tokenizer_model, drive_folder for the Colab run")
    qwen3: Optional[_Qwen3] = Field(None, description="qwen3_tts format: speaker name, reference clip, ZIP and duration limits")


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
        "tags": tag_counts.most_common(80), "omnivoice_tags": list(ds.OMNIVOICE_TAGS),
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
    spec = body.model_dump()
    left_out = set(spec["exclude_keys"])
    # The list shows every match, ticked or not; leaving a clip out only affects the selection count
    # and the export, so an unticked clip stays in view and can be ticked again.
    matched = await run_in_threadpool(ds.select, corpus, {**spec, "exclude_keys": []})
    picked = [l for l in matched if l["key"] not in left_out]
    keep = ("key", "chapter", "run", "index", "voice_id", "voice_name", "text", "plain", "tags",
            "words", "duration_s", "duplicate")
    page = [{**{k: l[k] for k in keep}, "stats": dataset_quality.get(l)} for l in matched[offset:offset + limit]]
    return {"total": len(matched), "seconds": round(sum(l["duration_s"] for l in matched), 1),
            "selected": len(picked), "selected_seconds": round(sum(l["duration_s"] for l in picked), 1),
            "offset": offset, "lines": page}


@router.post("/facets")
async def facets(body: _Filter):
    """Length, tag, chapter and audio-quality breakdown of what the current filters match."""
    ds = _ds()
    corpus = _corpus()
    matched = await run_in_threadpool(ds.select, corpus, {**body.model_dump(), "exclude_keys": []})
    return await run_in_threadpool(ds.facets, matched)


@router.get("/analyze", dependencies=[Depends(reject_cross_site_get)])
def analyze_status():
    return dataset_quality.status()


@router.post("/analyze")
def analyze_start():
    """Measure peak, level, clipping and silences of every clip not yet measured (background, local)."""
    return dataset_quality.start(_corpus())


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
    spec = {k: v for k, v in body.model_dump().items() if k in _Filter.model_fields}
    try:
        return await run_in_threadpool(
            lambda: ds.export(corpus, spec, body.dest, formats=body.formats, text_mode=body.text_mode,
                              sample_rate=body.sample_rate, val_fraction=body.val_fraction, name=body.name,
                              keep_tags=body.keep_tags, loudness_db=body.loudness_db, language_id=body.language_id,
                              strict_tags=body.strict_tags, training=body.training,
                              qwen3=body.qwen3.model_dump() if body.qwen3 else None))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not write the export ({type(exc).__name__}).")
