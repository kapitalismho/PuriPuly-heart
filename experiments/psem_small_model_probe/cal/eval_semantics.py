"""Shared evaluator semantics: any-speech gate, transition-aware CUT validity,
decoder-dependent (current-segment) contamination (V2 speaker-change repair).

Single location for GT gate semantics; imported by ``cal/run_cal.py``,
``main/run_main.py`` and ``vadreplay/run_replay.py``. No duplication: the two
runners re-export (not redefine) these helpers so read-only consumers
(``compare/``) keep working.

Frame gate (P0): ``speech_gt`` := any-speech — an unmasked GT interval
covering the sample with non-empty ``active_speakers`` — regardless of who
speaks. The previous anchor-only gate starved the persistence decoder during
B-only speech, so a clean A->B handoff could never accumulate evidence.
``anchor_speech_gt`` (anchor active) is retained as a frame-level diagnostic
only and MUST NOT gate the decoder.

CUT validity (P0/P1): a committed CUT is valid iff its
``source_boundary_time >= authoritative_transition_ms - CUT_TOLERANCE_MS``.
Tolerance provenance: 50 ms reuses ``constants_ms.annotation_boundary_jitter``
from ``experiments/psem_training_strategy_gate/data/v2/``
``operational_label_contract.json`` (contract_version ``psem-handoff-v1``;
psem-handoff-v0 carries the same 50 ms value). A committed source boundary
within one annotation-jitter quantum before the authoritative transition is
timing noise, not a premature decision. Episode success := first valid CUT
exists; premature-only := missed + premature_cut (+ n_premature_cuts
diagnostic). KEEP-episode false-cut usage is unchanged (any CUT = false cut).

Contamination (P1): decoder-dependent current-segment numerator. Per episode
the span is ``[evaluation_start, first valid CUT source_boundary)`` or the
full window when no valid CUT exists; the numerator sums GT unmasked
anchor-absent active seconds inside that span. The denominator
(active-speech hour over the full window) is unchanged. Callers MUST pass
``source_boundary_time``, never ``decision_time``.
"""

from __future__ import annotations

from bisect import bisect_right
from typing import Any

from experiments.psem_small_model_probe.cal.audio_resolve import SAMPLES_PER_MS

# Provenance: V2 operational_label_contract.json (psem-handoff-v1)
# constants_ms.annotation_boundary_jitter. See module docstring.
CUT_TOLERANCE_MS = 50


def _covering_interval(gt: dict, sample: int) -> dict | None:
    i = bisect_right(gt["starts"], sample) - 1
    if i < 0:
        return None
    interval = gt["intervals"][i]
    if sample >= interval["end_sample"]:
        return None
    return interval


def gt_any_speech(gt: dict | None, sample: int) -> bool:
    """Decoder gate: any unmasked GT speech covering sample."""
    if gt is None:
        return False
    interval = _covering_interval(gt, sample)
    return bool(
        interval is not None
        and not interval.get("masked", False)
        and interval.get("active_speakers", [])
    )


def gt_anchor_speech(gt: dict | None, anchor: str, sample: int) -> bool:
    """Diagnostic only: anchor active in an unmasked interval covering sample.

    MUST NOT gate the decoder; stored as ``anchor_speech_gt`` alongside the
    any-speech ``speech_gt`` frame record.
    """
    if gt is None:
        return False
    interval = _covering_interval(gt, sample)
    return bool(
        interval is not None
        and not interval.get("masked", False)
        and anchor in interval.get("active_speakers", [])
    )


def gt_window_stats(
    gt: dict, anchor: str, start_ms: int, end_ms: int
) -> tuple[float, float]:
    """(exclusive non-anchor seconds, active-speech seconds) in eval window."""
    start, end = start_ms * SAMPLES_PER_MS, end_ms * SAMPLES_PER_MS
    contam = active = 0
    for iv in gt["intervals"]:
        if iv["end_sample"] <= start:
            continue
        if iv["start_sample"] >= end:
            break
        if iv.get("masked", False):
            continue
        speakers = iv.get("active_speakers", [])
        if not speakers:
            continue
        overlap = min(iv["end_sample"], end) - max(iv["start_sample"], start)
        if overlap <= 0:
            continue
        seconds = overlap / 16000.0
        active += seconds
        if anchor not in speakers:
            contam += seconds
    return contam, active


def split_cuts(
    cuts: list[dict[str, Any]],
    transition_ms: int,
    tolerance_ms: int = CUT_TOLERANCE_MS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition committed CUTs into (valid, premature) by source boundary.

    Valid iff ``source_boundary_time >= transition_ms - tolerance_ms``.
    """
    valid: list[dict[str, Any]] = []
    premature: list[dict[str, Any]] = []
    for cut in cuts:
        boundary = cut.get("source_boundary_time")
        if boundary is not None and boundary >= transition_ms - tolerance_ms:
            valid.append(cut)
        else:
            premature.append(cut)
    return valid, premature


def compact_gt(
    gt: dict | None, anchor: str, start_ms: int, end_ms: int
) -> dict | None:
    """Sample-exact GT slice for one episode record (window-clipped spans).

    Stored on the record so tau-dependent current-segment contamination can
    be computed at scoring time, when the first valid CUT is known.
    """
    if gt is None:
        return None
    start, end = start_ms * SAMPLES_PER_MS, end_ms * SAMPLES_PER_MS
    spans = []
    for iv in gt["intervals"]:
        if iv["end_sample"] <= start:
            continue
        if iv["start_sample"] >= end:
            break
        spans.append(
            [
                max(iv["start_sample"], start),
                min(iv["end_sample"], end),
                list(iv.get("active_speakers", [])),
                bool(iv.get("masked", False)),
            ]
        )
    return {
        "anchor": anchor,
        "eval_start_ms": start_ms,
        "eval_end_ms": end_ms,
        "spans": spans,
    }


def current_segment_contam_s(
    gt_eval: dict, first_valid_source_boundary_ms: int | float | None
) -> float:
    """Decoder-dependent contamination numerator for one episode at one tau.

    Sums unmasked anchor-absent active seconds over
    ``[eval_start, first_valid_source_boundary)`` (full window when None).
    ``first_valid_source_boundary_ms`` MUST be a ``source_boundary_time``,
    never a ``decision_time``.
    """
    start = gt_eval["eval_start_ms"] * SAMPLES_PER_MS
    end_ms = (
        first_valid_source_boundary_ms
        if first_valid_source_boundary_ms is not None
        else gt_eval["eval_end_ms"]
    )
    end = int(end_ms * SAMPLES_PER_MS)
    anchor = gt_eval["anchor"]
    total = 0
    for span_start, span_end, speakers, masked in gt_eval["spans"]:
        if masked or not speakers or anchor in speakers:
            continue
        overlap = min(span_end, end) - max(span_start, start)
        if overlap > 0:
            total += overlap
    return total / 16000.0
