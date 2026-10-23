"""Evaluation gate for trained adapters (the part that needs no GPU).

``docs/ROADMAP.md`` says on-device LoRA ships only with an automatic evaluation gate. This module
holds everything about that gate that can be built and tested without a model:

- the held-out clips (``select_heldout``): a fixed, reproducible sample of ``omnivoice_val.jsonl``,
  written into the run as ``heldout.jsonl`` so a later generation check uses the same lines;
- the metrics that are plain arithmetic: held-out loss before/after (``loss_check``) and word
  error rate (``wer``) between what was asked for and what an ASR model heard;
- the record stored beside the adapter (``build_record``) with an honest ``ship_ready``.

An adapter is ``ship_ready`` only when every required check ran and passed. The checks that need a
GPU and models (generate the held-out lines with the base model and with the adapter, transcribe both
with an ASR model, compare speaker similarity to the real clips, listen to a sample) are listed under
``pending`` until a real run supplies them. Nothing here contacts a network.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from typing import Iterable, Optional

RECORD_FILE = "evaluation.json"
HELDOUT_FILE = "heldout.jsonl"

# Thresholds are deliberately conservative placeholders, recorded in every record so a verdict can
# be re-read against the numbers it used. They need tuning against a real run (handoff section 9).
MIN_LOSS_IMPROVEMENT = 0.02  # held-out loss must fall by at least 2% relative to the base model
MAX_WER_REGRESSION = 0.02  # adapter's word error rate may not exceed the base model's by more than this
MIN_SPEAKER_GAIN = 0.0  # adapter must be at least as close to the real voice as the base model

CHECKS = ("heldout_loss", "asr_wer", "speaker_similarity", "listening_test")
NEEDS_GPU = {"asr_wer": "generate the held-out lines with the base model and the adapter, then transcribe both with an ASR model",
             "speaker_similarity": "embed the generated and the real clips with a speaker-embedding model and compare",
             "listening_test": "a person listens to the base and adapter versions of the held-out lines"}


def select_heldout(val_jsonl: Optional[str], limit: int = 20) -> list[dict]:
    """A fixed sample of the validation clips: the same rows every time, whatever order the file has."""
    if not val_jsonl or not os.path.isfile(val_jsonl):
        return []
    rows = []
    with open(val_jsonl, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("id") and row.get("text") and row.get("audio_path"):
                rows.append(row)
    rows.sort(key=lambda r: hashlib.sha256(str(r["id"]).encode()).hexdigest())
    return rows[: max(0, limit)]


def write_heldout(path: str, rows: Iterable[dict]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_eval_history(path: str) -> list[dict]:
    """``eval_history.jsonl`` as written by the trainer: one ``{step, eval_loss}`` per evaluation."""
    out = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row.get("step"), int) and isinstance(row.get("eval_loss"), (int, float)):
                    out.append(row)
    except OSError:
        pass
    return sorted(out, key=lambda r: r["step"])


def loss_check(history: list[dict], min_improvement: float = MIN_LOSS_IMPROVEMENT) -> dict:
    """Held-out loss of the finished adapter against the untouched base model (the step-0 entry).

    ``status`` is ``passed`` / ``failed`` / ``not_run`` (no step-0 baseline or no later evaluation).
    The loss is the trainer's masked-token cross-entropy on the validation shards, so it measures how
    well the model predicts held-out speech tokens, not how it sounds.
    """
    base = next((r for r in history if r["step"] == 0), None)
    later = [r for r in history if r["step"] > 0]
    if base is None or not later:
        why = "no baseline evaluation before training" if base is None else "no evaluation after training started"
        return {"check": "heldout_loss", "status": "not_run", "reason": why,
                "threshold": min_improvement}
    final = later[-1]
    best = min(later, key=lambda r: r["eval_loss"])
    improvement = (base["eval_loss"] - final["eval_loss"]) / max(abs(base["eval_loss"]), 1e-9)
    return {"check": "heldout_loss", "status": "passed" if improvement >= min_improvement else "failed",
            "base_loss": base["eval_loss"], "final_loss": final["eval_loss"], "final_step": final["step"],
            "best_loss": best["eval_loss"], "best_step": best["step"],
            "relative_improvement": round(improvement, 4), "threshold": min_improvement,
            "reason": None if improvement >= min_improvement else
            f"held-out loss improved {improvement:.1%}, below the required {min_improvement:.0%}"}


_WORD_RE = re.compile(r"[\w']+", re.UNICODE)


def _words(text: str) -> list[str]:
    return _WORD_RE.findall((text or "").lower())


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate: word-level edit distance / reference length (0 = identical)."""
    ref, hyp = _words(reference), _words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1] / len(ref)


def mean_wer(references: list[str], hypotheses: list[str]) -> float:
    if len(references) != len(hypotheses) or not references:
        raise ValueError("Need the same, non-zero number of references and hypotheses.")
    return sum(wer(r, h) for r, h in zip(references, hypotheses)) / len(references)


def wer_check(references: list[str], base_hyps: list[str], tuned_hyps: list[str],
              max_regression: float = MAX_WER_REGRESSION) -> dict:
    """Does the adapter keep the speech as intelligible as the base model? Inputs are ASR transcripts."""
    base, tuned = mean_wer(references, base_hyps), mean_wer(references, tuned_hyps)
    ok = tuned <= base + max_regression
    return {"check": "asr_wer", "status": "passed" if ok else "failed", "base_wer": round(base, 4),
            "adapter_wer": round(tuned, 4), "threshold": max_regression, "clips": len(references),
            "reason": None if ok else f"adapter word error rate {tuned:.1%} vs base {base:.1%}"}


def similarity_check(base_score: float, adapter_score: float, min_gain: float = MIN_SPEAKER_GAIN) -> dict:
    """Mean cosine similarity to the real voice, from a speaker-embedding model, base vs adapter."""
    gain = adapter_score - base_score
    ok = gain >= min_gain
    return {"check": "speaker_similarity", "status": "passed" if ok else "failed",
            "base_score": round(base_score, 4), "adapter_score": round(adapter_score, 4),
            "gain": round(gain, 4), "threshold": min_gain,
            "reason": None if ok else f"adapter is {abs(gain):.3f} less similar to the real voice than the base"}


def build_record(results: list[dict], *, heldout: int, extra: Optional[dict] = None) -> dict:
    """The pass/fail record stored beside the adapter.

    ``ship_ready`` is true only if every check in ``CHECKS`` ran and passed. Checks that have not run
    are listed under ``pending`` with what they need, so a loss-only record can never read as a pass.
    """
    by_name = {r["check"]: r for r in results}
    pending = [{"check": c, "needs": NEEDS_GPU.get(c, "a training run with a baseline evaluation")}
               for c in CHECKS if by_name.get(c, {}).get("status", "not_run") == "not_run"]
    failed = [c for c in CHECKS if by_name.get(c, {}).get("status") == "failed"]
    if failed:
        verdict = "failed"
    elif pending:
        verdict = "incomplete"
    else:
        verdict = "passed"
    return {"verdict": verdict, "ship_ready": verdict == "passed", "failed": failed, "pending": pending,
            "checks": [by_name.get(c, {"check": c, "status": "not_run"}) for c in CHECKS],
            "heldout_clips": heldout, "evaluated_at": time.time(), **(extra or {})}


def save_record(directory: str, record: dict) -> str:
    path = os.path.join(directory, RECORD_FILE)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    os.replace(tmp, path)
    return path


def load_record(directory: str) -> Optional[dict]:
    try:
        with open(os.path.join(directory, RECORD_FILE), encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None
