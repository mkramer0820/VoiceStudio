"""Fish training-data corpus: index story run folders, filter, play, export. Local files only."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest
import soundfile as sf
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.dependencies import require_admin
from api.routers import fish_dataset
from core.browser_guard import reject_cross_site_get
from engines.fish_api import dataset, dataset_quality

@pytest.fixture(autouse=True)
def _store(monkeypatch, tmp_path):
    os.makedirs(tmp_path / "store", exist_ok=True)
    monkeypatch.setattr("engines.fish_api.story._store_dir", lambda: str(tmp_path / "store"))
    dataset_quality.reset_for_tests()
    yield
    dataset_quality.reset_for_tests()


NARR = "a" * 32
SIDE = "b" * 32


def _wav(path, seconds=2.0, rate=44100):
    sf.write(path, (np.sin(np.linspace(0, 440, int(rate * seconds))) * 0.2).astype("float32"), rate, subtype="PCM_16")


def _run(root, chapter, run, started, lines, story="untitled"):
    """lines: (chunk id, voice id, voice name, text, seconds, status)"""
    folder = os.path.join(root, chapter, story, run)
    os.makedirs(folder)
    chunks = []
    for i, (cid, vid, vname, text, secs, status) in enumerate(lines, 1):
        name = f"{i:03d}_{vname}.wav"
        if status == "done" and secs:
            _wav(os.path.join(folder, name), secs)
        chunks.append({"id": cid, "index": i, "status": status, "file": name, "voice": vid,
                       "voice_name": vname, "duration_s": secs, "sample_rate": 44100,
                       "sha256": f"sha{cid}{run}", "text": text})
    with open(os.path.join(folder, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({"id": run[-4:], "status": "done", "started_at": started, "model": "s2.1-pro-free",
                   "sample_rate": 44100, "chunks": chunks}, fh)
    with open(os.path.join(folder, "story.json"), "w", encoding="utf-8") as fh:
        json.dump({"draft": {"chunks": [{"id": c[0], "text": c[3]} for c in lines]}}, fh)
    return folder


@pytest.fixture
def root(tmp_path):
    r = str(tmp_path / "input")
    _run(r, "3001_one", "run-001_20261001-000000_aaaa", 100, [
        ("c1", NARR, "Narrator", "[slow][calm] The road was long.", 3.0, "done"),
        ("c2", NARR, "Narrator", "[slow] Rain fell.", 1.0, "done"),
        ("c3", SIDE, "Noctis", "[shouting] Come out!", 2.0, "done"),
        ("c4", SIDE, "Noctis", "Lost line.", 2.0, "failed"),
    ])
    # a re-render of the same first line, later: only this copy should count
    _run(r, "3001_one", "run-002_20261002-000000_bbbb", 200, [
        ("c1", NARR, "Narrator", "[slow][calm] The road was long.", 3.5, "done"),
    ])
    _run(r, "3002_two", "run-001_20261003-000000_cccc", 300, [
        ("d1", NARR, "Narrator", "[slow] Another chapter begins.", 4.0, "done"),
    ])
    _run(r, "fish_non_api", "run-001_20261003-000000_dddd", 300, [("x", NARR, "Narrator", "Ignored.", 1.0, "done")])
    return r


def test_indexes_runs_groups_speakers_and_drops_stale_duplicates(root):
    c = dataset.load(root)
    assert {r["chapter"] for r in c.runs} == {"3001_one", "3002_two"}  # fish_non_api is ignored
    usable = [l for l in c.lines if not l["duplicate"]]
    assert len(c.lines) == 5 and len(usable) == 4  # failed line skipped, older copy flagged
    old = next(l for l in c.lines if l["duplicate"])
    assert old["duration_s"] == 3.0 and old["run"].startswith("run-001")
    narr = next(s for s in c.speakers() if s["voice_id"] == NARR)
    assert narr["lines"] == 3 and narr["duplicates"] == 1 and narr["seconds"] == pytest.approx(8.5)
    assert dict(narr["top_tags"])["slow"] == 3
    assert c.lines[0]["plain"] == "The road was long." and c.lines[0]["tags"] == ["slow", "calm"]


def test_a_missing_audio_file_is_counted_not_indexed(root):
    folder = os.path.join(root, "3002_two", "untitled", "run-001_20261003-000000_cccc")
    os.remove(os.path.join(folder, "001_Narrator.wav"))
    c = dataset.load(root, refresh=True)
    run = next(r for r in c.runs if r["chapter"] == "3002_two")
    assert run["missing"] == 1 and run["lines"] == 0


def test_filters(root):
    c = dataset.load(root)
    assert {l["voice_name"] for l in dataset.select(c, {"voice_ids": [SIDE]})} == {"Noctis"}
    assert len(dataset.select(c, {"min_dur": 2.5})) == 2
    assert len(dataset.select(c, {"max_dur": 1.5})) == 1
    assert len(dataset.select(c, {"tag": "shouting"})) == 1
    assert len(dataset.select(c, {"q": "chapter"})) == 1
    assert len(dataset.select(c, {"duplicates": True})) == 5
    key = c.lines[0]["key"]
    assert key not in {l["key"] for l in dataset.select(c, {"exclude_keys": [key]})}


def test_text_helpers():
    assert dataset.strip_tags("[slow][calm]  Hello, [pause] world [laugh]!") == "Hello, world !"
    assert dataset.tags_of("[Slow] a [break] b") == ["slow", "break"]


def test_audio_is_served_only_for_indexed_clips(root):
    c = dataset.load(root)
    assert dataset.audio_path(c, c.lines[0]["key"]).endswith(".wav")
    for bad in ("../../x", "nope:nope", ""):
        assert dataset.audio_path(c, bad) is None


def test_export_writes_audio_labels_metadata_and_a_stable_split(root, tmp_path):
    c = dataset.load(root)
    out = dataset.export(c, {}, str(tmp_path / "out"), formats=["jsonl", "ljspeech", "fish_lab"],
                         text_mode="strip", sample_rate=24000, val_fraction=0.25, name="demo set")
    assert out["clips"] == 4 and out["speakers"]["Narrator"]["clips"] == 3
    folder = out["path"]
    assert os.path.basename(folder) == "demo_set"
    assert sf.info(os.path.join(folder, "Narrator", "Narrator_00001.wav")).samplerate == 24000
    assert open(os.path.join(folder, "Narrator", "Narrator_00002.lab"), encoding="utf-8").read() == "The road was long.\n"
    assert "Narrator_00002|The road was long.|" in open(os.path.join(folder, "Narrator", "metadata.csv"), encoding="utf-8").read()
    rows = [json.loads(l) for l in open(os.path.join(folder, "metadata.jsonl"), encoding="utf-8")]
    road = next(r for r in rows if r["text"] == "The road was long.")
    assert road["text_raw"].startswith("[slow][calm]") and road["file_name"] == "Narrator/Narrator_00002.wav"
    train = [json.loads(l) for l in open(os.path.join(folder, "train.jsonl"), encoding="utf-8")]
    val = [json.loads(l) for l in open(os.path.join(folder, "val.jsonl"), encoding="utf-8")]
    assert len(train) + len(val) == 4
    card = json.load(open(os.path.join(folder, "dataset.json"), encoding="utf-8"))
    assert "Fish Audio" in card["source"] and card["clips"] == 4


def test_export_keeps_tags_by_default_and_never_overwrites(root, tmp_path):
    c = dataset.load(root)
    out = dataset.export(c, {"voice_ids": [SIDE]}, str(tmp_path / "o"), formats=["fish_lab"])
    assert open(os.path.join(out["path"], "Noctis", "Noctis_00001.lab"), encoding="utf-8").read() == "[shouting] Come out!\n"
    with pytest.raises(ValueError, match="already has files"):
        dataset.export(c, {"voice_ids": [SIDE]}, str(tmp_path / "o"), formats=["fish_lab"])


@pytest.mark.parametrize("kwargs", [
    {"formats": []}, {"formats": ["mp4"]}, {"formats": ["jsonl"], "text_mode": "x"},
    {"formats": ["jsonl"], "val_fraction": 0.9}, {"formats": ["jsonl"], "sample_rate": 100},
])
def test_export_rejects_bad_options(root, tmp_path, kwargs):
    with pytest.raises(ValueError):
        dataset.export(dataset.load(root), {}, str(tmp_path / "o"), **kwargs)


def test_export_with_nothing_selected_or_a_relative_folder_is_refused(root, tmp_path):
    c = dataset.load(root)
    with pytest.raises(ValueError, match="Nothing matches"):
        dataset.export(c, {"q": "zzz"}, str(tmp_path / "o"), formats=["jsonl"])
    with pytest.raises(ValueError, match="full path"):
        dataset.export(c, {}, "relative/dir", formats=["jsonl"])


@pytest.fixture
def api(monkeypatch, root):
    store = {}
    monkeypatch.setattr("services.settings_store.get_text", lambda k, d=None: store.get(k, d))
    monkeypatch.setattr("services.settings_store.set_text", lambda k, v: store.__setitem__(k, v))
    monkeypatch.setattr("services.settings_store.clear_text", lambda k: store.pop(k, None))
    app = FastAPI()
    app.include_router(fish_dataset.router)
    app.dependency_overrides[require_admin] = lambda: None
    app.dependency_overrides[reject_cross_site_get] = lambda: None
    return TestClient(app)


def test_routes_end_to_end(api, root, tmp_path):
    assert api.get("/api/fish/dataset").json()["root"] is None
    assert api.post("/api/fish/dataset/lines", json={}).status_code == 404
    assert api.put("/api/fish/dataset/root", json={"path": "relative"}).status_code == 422
    assert api.put("/api/fish/dataset/root", json={"path": str(tmp_path / "nope")}).status_code == 422
    assert api.put("/api/fish/dataset/root", json={"path": root}).json()["root"] == os.path.normpath(root)

    overview = api.get("/api/fish/dataset").json()
    assert overview["totals"]["lines"] == 4 and overview["totals"]["duplicates"] == 1
    assert overview["chapters"] == ["3001_one", "3002_two"]

    page = api.post("/api/fish/dataset/lines", params={"limit": 2}, json={"voice_ids": [NARR]}).json()
    assert page["total"] == 3 and len(page["lines"]) == 2 and page["seconds"] == pytest.approx(8.5)
    key = page["lines"][0]["key"]
    assert api.get("/api/fish/dataset/audio", params={"key": key}).headers["content-type"] == "audio/wav"
    assert api.get("/api/fish/dataset/audio", params={"key": "x:y"}).status_code == 404

    res = api.post("/api/fish/dataset/export", json={"dest": str(tmp_path / "ex"), "voice_ids": [SIDE],
                                                   "formats": ["jsonl"]})
    assert res.status_code == 200 and res.json()["clips"] == 1
    again = api.post("/api/fish/dataset/export", json={"dest": str(tmp_path / "ex"), "voice_ids": [SIDE],
                                                     "formats": ["jsonl"]})
    assert again.status_code == 422


def test_strip_everything_or_keep_only_the_tags_you_list():
    text = "[slow][Laughter] Ha, [pause] well [sigh] ok [whispering] shh"
    assert dataset.strip_tags(text) == "Ha, well ok shh"
    assert dataset.strip_tags(text, ["laughter", "[Sigh]"]) == "[Laughter] Ha, well [sigh] ok shh"
    assert dataset.strip_tags(text, "laughter, sigh".split(",")) == "[Laughter] Ha, well [sigh] ok shh"
    assert dataset.strip_tags(text, ["nothing-here"]) == "Ha, well ok shh"
    assert dataset.normalize_tags(["[Laughter], sigh", " "]) == {"laughter": "laughter", "sigh": "sigh"}
    # a tag can be kept under another spelling, e.g. Fish's [laughing] for a model that wants [laughter]
    assert dataset.strip_tags("[slow][laughing] Ha [sighing] oh", ["laughing=laughter", "sighing=sigh"]) == "[laughter] Ha [sigh] oh"


def test_export_can_keep_listed_tags_while_stripping_the_rest(root, tmp_path):
    c = dataset.load(root)
    out = dataset.export(c, {"voice_ids": [NARR]}, str(tmp_path / "k"), formats=["jsonl", "fish_lab"],
                         text_mode="strip", keep_tags=["calm"])
    rows = {r["text_plain"]: r for r in (json.loads(l) for l in open(os.path.join(out["path"], "metadata.jsonl"), encoding="utf-8"))}
    assert rows["The road was long."]["text"] == "[calm] The road was long."  # [slow] dropped, [calm] kept
    assert rows["Rain fell."]["text"] == "Rain fell." and rows["Rain fell."]["text_raw"] == "[slow] Rain fell."
    assert out["kept_tags"] == {"calm": "calm"}
    # keeping tags has no effect when the transcript keeps everything anyway
    both = dataset.export(c, {"voice_ids": [NARR]}, str(tmp_path / "k2"), formats=["jsonl"], text_mode="keep", keep_tags=["calm"])
    assert both["kept_tags"] == {}


def _signal(path, parts, rate=44100):
    """parts: (seconds, amplitude) pieces joined together; amplitude 0 is silence."""
    chunks = [np.sin(np.linspace(0, 600, int(rate * sec))) * amp for sec, amp in parts]
    sf.write(path, np.concatenate(chunks).astype("float32"), rate, subtype="PCM_16")


def test_measure_reports_clipping_level_edges_and_pauses(tmp_path):
    clean = str(tmp_path / "clean.wav")
    _signal(clean, [(2.0, 0.2)])
    m = dataset_quality.measure(clean)
    assert m["peak"] == pytest.approx(0.2, abs=0.01) and m["clip"] == 0 and m["pauses"] == 0
    assert m["rms_db"] == pytest.approx(-17, abs=1.5) and m["lead_ms"] == 0

    hot = str(tmp_path / "hot.wav")
    _signal(hot, [(1.0, 1.0)])
    assert dataset_quality.measure(hot)["clip"] > 0.001

    chopped = str(tmp_path / "chopped.wav")
    _signal(chopped, [(0.6, 0.0), (0.5, 0.2), (0.4, 0.0), (0.5, 0.2), (0.35, 0.0), (0.5, 0.2), (0.9, 0.0)])
    m = dataset_quality.measure(chopped)
    assert 500 <= m["lead_ms"] <= 650 and 800 <= m["tail_ms"] <= 950
    assert m["pauses"] == 2 and 350 <= m["max_gap_ms"] <= 450


def _analyze(corpus, timeout=15):
    import time

    dataset_quality.start(corpus)
    end = time.time() + timeout
    while dataset_quality.status()["running"] and time.time() < end:
        time.sleep(0.02)
    return dataset_quality.status()


def test_analysis_fills_the_cache_once_and_quality_filters_use_it(root):
    c = dataset.load(root)
    # make the Noctis clip clip and the 4 s narrator clip choppy
    hot = os.path.join(root, "3001_one", "untitled", "run-001_20261001-000000_aaaa", "003_Noctis.wav")
    _signal(hot, [(2.0, 1.0)])
    chop = os.path.join(root, "3002_two", "untitled", "run-001_20261003-000000_cccc", "001_Narrator.wav")
    _signal(chop, [(0.5, 0.2), (0.4, 0.0), (0.5, 0.2), (0.4, 0.0), (0.5, 0.2), (0.4, 0.0), (0.5, 0.2)])

    assert len(dataset.select(c, {"max_clip": 0.0})) == 0  # a quality limit with no measurements matches nothing
    status = _analyze(c)
    assert status["done"] == status["total"] == 5 and status["failed"] == 0 and not status["running"]
    keys = lambda f: {l["voice_name"] + str(l["index"]) for l in dataset.select(c, f)}
    assert len(dataset.select(c, {"max_clip": 0.0005})) == 3  # four clips, one of them clipping
    assert "Noctis3" not in keys({"max_clip": 0.0005}) and "Noctis3" in keys({"max_peak": 1.0})
    chopped = [l for l in dataset.select(c, {}) if (dataset_quality.get(l) or {}).get("pauses", 0) >= 3]
    assert len(chopped) == 1 and chopped[0]["chapter"] == "3002_two"
    assert all(l["chapter"] != "3002_two" for l in dataset.select(c, {"max_pauses": 1}))

    again = dataset_quality.start(c)  # nothing left to measure
    assert again["total"] == 0
    assert dataset.facets(dataset.select(c, {}))["quality"]["clipping"] == 1
    dataset_quality.reset_for_tests()  # a fresh process reads the saved cache from disk
    assert dataset.select(c, {"max_peak": 0.5})


def test_tag_chapter_word_and_sort_filters(root):
    c = dataset.load(root)
    names = lambda f: [l["plain"] for l in dataset.select(c, f)]
    assert names({"tags_any": ["calm", "shouting"]}) == ["Come out!", "The road was long."]
    assert names({"tags_all": ["slow", "calm"]}) == ["The road was long."]
    assert "Rain fell." not in names({"tags_none": ["[slow]"]}) and len(names({"tags_none": ["slow"]})) == 1
    assert names({"max_words": 2}) == ["Rain fell.", "Come out!"]
    assert names({"chapters": ["3002_two"]}) == ["Another chapter begins."]
    assert names({"sort": "duration_desc"})[0] == "Another chapter begins."
    assert names({"sort": "duration_asc"})[0] == "Rain fell."
    assert names({"sort": "random", "seed": 3}) == names({"sort": "random", "seed": 3})
    assert sorted(names({"sort": "random", "seed": 3})) == sorted(names({}))
    keys = [l["key"] for l in c.lines if not l["duplicate"]][:2]
    assert [l["key"] for l in dataset.select(c, {"only_keys": keys})] == keys
    assert dataset.select(c, {"only_keys": []}) == []


def test_facets_describe_the_selection(root):
    c = dataset.load(root)
    f = dataset.facets(dataset.select(c, {"voice_ids": [NARR]}))
    assert f["clips"] == 3 and f["seconds"] == pytest.approx(8.5)
    assert sum(b["clips"] for b in f["duration"]) == 3
    by_from = {b["from"]: b["clips"] for b in f["duration"]}
    assert by_from[1] == 1 and by_from[3] == 1 and by_from[4] == 1  # 1.0 s, 3.5 s, 4.0 s
    assert dict(f["tags"])["slow"] == 3 and dict(f["tags"])["calm"] == 1
    assert [(x["chapter"], x["clips"]) for x in f["chapters"]] == [("3001_one", 2), ("3002_two", 1)]
    assert f["words"]["min"] == 2 and f["quality"]["analyzed"] == 0
    assert dataset.facets([])["words"] is None


def test_facets_analyze_and_new_filters_over_the_api(api, root):
    import time

    api.put("/api/fish/dataset/root", json={"path": root})
    facets = api.post("/api/fish/dataset/facets", json={"voice_ids": [NARR], "tags_none": ["calm"]}).json()
    assert facets["clips"] == 2 and dict(facets["tags"]).get("calm") is None
    assert api.post("/api/fish/dataset/lines", json={"sort": "sideways"}).status_code == 422

    assert api.post("/api/fish/dataset/analyze").status_code == 200
    end = time.time() + 15
    while api.get("/api/fish/dataset/analyze").json()["running"] and time.time() < end:
        time.sleep(0.02)
    assert api.get("/api/fish/dataset/analyze").json()["done"] == 5
    page = api.post("/api/fish/dataset/lines", json={"max_clip": 0.01, "sort": "duration_desc"}).json()
    assert page["total"] == 4 and page["lines"][0]["stats"]["peak"] > 0
    assert page["lines"][0]["duration_s"] >= page["lines"][-1]["duration_s"]


def test_ticked_off_clips_stay_listed_but_leave_the_selection_and_export(api, root, tmp_path):
    api.put("/api/fish/dataset/root", json={"path": root})
    everything = api.post("/api/fish/dataset/lines", json={}).json()
    assert everything["total"] == 4 and everything["selected"] == 4
    left_out = [l["key"] for l in everything["lines"][:3]]
    page = api.post("/api/fish/dataset/lines", json={"exclude_keys": left_out}).json()
    assert page["total"] == 4 and len(page["lines"]) == 4  # still listed, so they can be ticked again
    assert page["selected"] == 1 and page["selected_seconds"] == pytest.approx(everything["lines"][3]["duration_s"])
    assert api.post("/api/fish/dataset/facets", json={"exclude_keys": left_out}).json()["clips"] == 4
    only = api.post("/api/fish/dataset/lines", json={"only_keys": left_out, "exclude_keys": []}).json()
    assert only["total"] == 3
    out = api.post("/api/fish/dataset/export", json={"dest": str(tmp_path / "sel"), "exclude_keys": left_out,
                                                    "formats": ["jsonl"]}).json()
    assert out["clips"] == 1
    none = api.post("/api/fish/dataset/export", json={"dest": str(tmp_path / "none"),
                                                     "exclude_keys": [l["key"] for l in everything["lines"]],
                                                     "formats": ["jsonl"]})
    assert none.status_code == 422  # nothing selected
