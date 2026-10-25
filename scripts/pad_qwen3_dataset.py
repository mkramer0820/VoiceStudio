#!/usr/bin/env python3
"""Append silence to the end of every training clip of a Qwen3-TTS export and zip the result.

Why: people fine-tuning Qwen3-TTS report that about one second of trailing silence on every clip stops rushed speech and noise hiccups
(Qwen3-TTS issue #39). The exporter does not add it, so run this on an export folder before uploading it to Drive.

    python scripts/pad_qwen3_dataset.py E:/qwen3TTS/narrator_3h --name narrator_3h_sil

Writes <export folder>_sil (or --out) and <name>.zip next to it. The original folder is never changed. The padded copy holds only what
training needs (train.jsonl, audio/, reference/, the report); the helper notebook, runner and text files are left out because they name
the old zip. The reference clip is NOT padded.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import wave
import zipfile

KEEP_FILES = ("train.jsonl", "dataset_report.json", "rejected_samples.jsonl")


def pad_wav(path: str, seconds: float) -> None:
    with wave.open(path, "rb") as w:
        params, data = w.getparams(), w.readframes(w.getnframes())
    if params.sampwidth != 2 or params.nchannels != 1:
        raise ValueError(f"{path}: expected 16-bit mono WAV (the Qwen3-TTS export), got {params.sampwidth * 8}-bit x{params.nchannels}")
    silence = b"\x00\x00" * int(round(params.framerate * seconds))
    with wave.open(path, "wb") as o:
        o.setparams(params)
        o.writeframes(data + silence)


def pad_dataset(src: str, dst: str, seconds: float = 1.0, zip_path: str | None = None) -> dict:
    """Copy ``src`` to ``dst`` (training files only), pad every clip under audio/, optionally zip. Returns counts."""
    if not os.path.isfile(os.path.join(src, "train.jsonl")) or not os.path.isdir(os.path.join(src, "audio")):
        raise FileNotFoundError(f"{src} is not a Qwen3-TTS export (needs train.jsonl and audio/)")
    if os.path.abspath(src) == os.path.abspath(dst):
        raise ValueError("the output folder must differ from the export folder: the original is never modified")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    for name in KEEP_FILES:
        if os.path.isfile(os.path.join(src, name)):
            shutil.copyfile(os.path.join(src, name), os.path.join(dst, name))
    if os.path.isdir(os.path.join(src, "reference")):
        shutil.copytree(os.path.join(src, "reference"), os.path.join(dst, "reference"))
    os.makedirs(os.path.join(dst, "audio"))
    clips = 0
    for name in sorted(os.listdir(os.path.join(src, "audio"))):
        if name.lower().endswith(".wav"):
            target = os.path.join(dst, "audio", name)
            shutil.copyfile(os.path.join(src, "audio", name), target)
            pad_wav(target, seconds)
            clips += 1
    result = {"clips": clips, "seconds_added_each": seconds, "folder": dst, "zip": None}
    if zip_path:
        if os.path.exists(zip_path):
            os.remove(zip_path)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk(dst):
                for f in sorted(files):
                    full = os.path.join(root, f)
                    z.write(full, os.path.relpath(full, dst).replace(os.sep, "/"))
        if zipfile.ZipFile(zip_path).testzip() is not None:
            raise RuntimeError(f"{zip_path} failed its integrity check")
        result["zip"] = {"path": zip_path, "mb": round(os.path.getsize(zip_path) / 2**20)}
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("export_folder")
    ap.add_argument("--seconds", type=float, default=1.0, help="silence to append to every clip (default 1.0)")
    ap.add_argument("--out", help="output folder (default <export folder>_sil)")
    ap.add_argument("--name", help="zip name without .zip (default: the output folder's name)")
    args = ap.parse_args(argv)
    src = os.path.abspath(args.export_folder)
    dst = os.path.abspath(args.out or src.rstrip("/\\") + "_sil")
    name = args.name or os.path.basename(dst)
    res = pad_dataset(src, dst, args.seconds, os.path.join(os.path.dirname(dst), name + ".zip"))
    print(f"padded {res['clips']} clips with {res['seconds_added_each']:g} s of silence -> {res['folder']}")
    print(f"zip: {res['zip']['path']} ({res['zip']['mb']} MB). Upload it to Drive: Qwen3TTS/datasets/ and set DATA_SOURCE in the notebook to it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
