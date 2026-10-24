#!/usr/bin/env python3
"""Export one speaker's Fish renders as a Qwen3-TTS fine-tuning folder (optionally zipped).

A thin wrapper over ``backend/engines/fish_api/qwen3_export.py``, the code the Training data screen
uses, so the two cannot drift apart. It trains nothing and contacts no network.

    python scripts/export_qwen3_tts.py --input <fish render root> --output <new folder> \
        --speaker-name MySpeaker --zip

Run it with the interpreter that runs the backend (it imports the backend package).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))


DEFAULT_LOUDNESS_DB = -22.0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="the Fish render root (the folder that holds the chapter folders)")
    p.add_argument("--output", required=True, help="the new export folder to create (its parent must exist)")
    p.add_argument("--voice", help="Fish voice id or exact voice name; optional when the input holds one speaker")
    p.add_argument("--speaker-name", default="", help="name the fine-tuned model answers to (default: the voice name)")
    p.add_argument("--reference", help="reference audio file, or a clip key; default: chosen automatically")
    p.add_argument("--chapter", action="append", default=[], help="only this chapter folder (repeat for several)")
    p.add_argument("--zip", action="store_true", help="also write <output>.zip next to the folder")
    p.add_argument("--overwrite", action="store_true", help="replace a previous Qwen3-TTS export at --output")
    p.add_argument("--min-duration", type=float, default=0.8, help="reject clips shorter than this (seconds)")
    p.add_argument("--max-duration", type=float, default=30.0, help="reject clips longer than this (seconds)")
    p.add_argument("--validation-ratio", type=float, default=0.0,
                   help="not supported: Qwen's scripts have no validation input (only 0 is accepted)")
    p.add_argument("--loudness-db", type=float, default=DEFAULT_LOUDNESS_DB,
                   help=f"match every clip to this whole-clip RMS level in dBFS (default {DEFAULT_LOUDNESS_DB:g}; "
                        "-19 to -23 is the usual range)")
    p.add_argument("--no-loudness-match", action="store_true", help="leave every clip at its own level")
    p.add_argument("--drive-folder", default="VoiceStudio-training",
                   help="Google Drive folder under My Drive; checkpoints go to <folder>/<speaker>/checkpoints")
    p.add_argument("--epochs", type=int, help="sft_12hz.py --num_epochs (default 3)")
    p.add_argument("--learning-rate", type=float, help="sft_12hz.py --lr (default 2e-5)")
    p.add_argument("--batch-size", type=int, help="sft_12hz.py --batch_size (default 2)")
    p.add_argument("--base-model", help="sft_12hz.py --init_model_path (default Qwen/Qwen3-TTS-12Hz-1.7B-Base)")
    return p


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from engines.fish_api import dataset, qwen3_export

    if args.validation_ratio:
        print("error: the official Qwen3-TTS fine-tuning scripts have no validation input, so --validation-ratio "
              "must be 0.", file=sys.stderr)
        return 2
    corpus = dataset.load(os.path.abspath(args.input))
    if corpus is None:
        print(f"error: {args.input} is not a folder.", file=sys.stderr)
        return 2
    speakers = corpus.speakers()
    if not speakers:
        print("error: no finished clips were found under the input folder.", file=sys.stderr)
        return 2
    wanted = None
    if args.voice:
        wanted = next((s for s in speakers if args.voice in (s["voice_id"], s["name"])), None)
        if wanted is None:
            print("error: no such voice. Voices found: " + ", ".join(f"{s['name']} ({s['voice_id']})" for s in speakers),
                  file=sys.stderr)
            return 2
    elif len(speakers) > 1:
        print("error: the input holds several speakers; pick one with --voice. Voices found: "
              + ", ".join(f"{s['name']} ({s['voice_id']}, {s['lines']} clips)" for s in speakers), file=sys.stderr)
        return 2
    else:
        wanted = speakers[0]
    spec: dict = {"voice_ids": [wanted["voice_id"]]}
    if args.chapter:
        spec["chapters"] = args.chapter
    options: dict = {"speaker_name": args.speaker_name, "make_zip": args.zip, "overwrite": args.overwrite,
                     "min_duration": args.min_duration, "max_duration": args.max_duration}
    training = {"drive_folder": args.drive_folder, "epochs": args.epochs, "learning_rate": args.learning_rate,
                "batch_size": args.batch_size, "base_model": args.base_model}
    if args.reference:
        if os.path.isfile(args.reference):
            options["reference_file"] = os.path.abspath(args.reference)
        else:
            options["reference_key"] = args.reference
    output = os.path.abspath(args.output)
    try:
        result = dataset.export(
            corpus, spec, os.path.dirname(output), formats=["qwen3_tts"], name=os.path.basename(output),
            loudness_db=None if args.no_loudness_match else args.loudness_db, qwen3=options, training=training)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    q = result["qwen3"]
    ref = q["reference"]
    print(f"Exported {q['accepted']} clips ({result['hours']} h), rejected {q['rejected']} {q['rejected_by_reason'] or ''}")
    print(f"Folder: {result['path']}")
    lv = result.get("loudness")
    if lv:
        print(f"Level matching to {lv['target_rms_db']:g} dB: {lv['adjusted']} of {lv['clips']} clips adjusted, "
              f"{lv['peak_limited']} limited by the peak cap (worst {lv['worst_shortfall_db']:.1f} dB short)")
    print(f"Reference: {ref.get('source')} ({ref.get('duration')} s): {ref.get('reason')}")
    print("Checkpoints on Drive: " + q["commands"]["official"]["checkpoint_dir"])
    if q["zip"]:
        print(f"ZIP: {q['zip']['path']}")
    for w in q["warnings"]:
        print(f"WARNING [{w['code']}]: {w['message']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
