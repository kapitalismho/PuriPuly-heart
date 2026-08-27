from __future__ import annotations

import json
from typing import Any

import numpy as np

from experiments.psem_frozen_ceiling_gate.build_ceiling_examples import (
    PACKAGE_ROOT,
    config,
    load_sessions,
)
from experiments.psem_frozen_ceiling_gate.evaluate_ceiling import (
    RESULTS_ROOT,
    aggregate_rows,
    session_row,
)
from experiments.psem_relative_occupancy_gate.io_utils import sha256_file, write_json, write_jsonl

REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]
ISSUE98_RESULTS = REPOSITORY_ROOT / "experiments" / "psem_ontology_simplification_gate" / "results"


def read_gate(role: str) -> dict[str, dict[str, Any]]:
    path = ISSUE98_RESULTS / role / "production_vad_speech_gate.jsonl"
    receipt = json.loads(
        (ISSUE98_RESULTS / role / "production_vad_replay_receipt.json").read_text(encoding="utf-8")
    )
    if receipt["speech_gate_sha256"] != sha256_file(path):
        raise ValueError("production VAD gate differs from its pinned receipt")
    result = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            result[str(row["source_id"])] = row
    if len(result) != int(receipt["source_count"]):
        raise ValueError("production VAD source coverage differs from its pinned receipt")
    return result


def support_intervals(
    session: Any, gate_row: dict[str, Any], pre_roll_samples: int
) -> list[dict[str, Any]]:
    boundaries = {
        int(session.manifest["scored_start_sample"]),
        int(session.manifest["scored_end_sample"]),
    }
    spans = []
    for value in gate_row["speech_spans"]:
        start = max(int(value["start_sample"]), int(session.manifest["scored_start_sample"]))
        end = min(int(value["end_sample"]), int(session.manifest["scored_end_sample"]))
        if end <= start:
            continue
        committed_start = min(end, start + pre_roll_samples)
        spans.append((start, committed_start, end))
        boundaries.update((start, committed_start, end))
    for interval in session.manifest["intervals"]:
        boundaries.update((int(interval["start_sample"]), int(interval["end_sample"])))
    ordered = sorted(boundaries)
    result = []
    span_index = 0
    interval_index = 0
    gt = session.manifest["intervals"]
    for start, end in zip(ordered, ordered[1:]):
        while span_index < len(spans) and spans[span_index][2] <= start:
            span_index += 1
        while interval_index < len(gt) and int(gt[interval_index]["end_sample"]) <= start:
            interval_index += 1
        in_model = span_index < len(spans) and spans[span_index][0] <= start < spans[span_index][2]
        pre_roll = in_model and start < spans[span_index][1]
        committed = in_model and start >= spans[span_index][1]
        active = (
            interval_index < len(gt)
            and int(gt[interval_index]["start_sample"]) <= start
            and bool(gt[interval_index]["active_speakers"])
            and not bool(gt[interval_index]["masked"])
        )
        overlapping = np.logical_and(session.starts < end, session.ends > start)
        state_valid = bool(np.any(np.logical_and(overlapping, session.valid)))
        result.append(
            {
                "source_id": session.source_id,
                "start_sample": start,
                "end_sample": end,
                "acoustic_speech_present": active,
                "model_input_present": in_model,
                "live_confirmation_support": committed,
                "pre_roll_only": pre_roll,
                "padding_only": False,
                "model_state_valid": state_valid,
            }
        )
    return result


def run() -> dict[str, Any]:
    cfg = config()
    sessions = [value for value in load_sessions((500,)) if value.role == "eval"]
    gate = read_gate("eval")
    all_intervals = []
    rows = []
    support_totals = {
        "pre_roll_only_seconds": 0.0,
        "live_confirmation_support_seconds": 0.0,
        "model_input_seconds": 0.0,
    }
    for session in sessions:
        intervals = support_intervals(
            session, gate[session.source_id], int(cfg["vad_pre_roll_ms"] * 16)
        )
        all_intervals.extend(intervals)
        for interval in intervals:
            seconds = (interval["end_sample"] - interval["start_sample"]) / 16000.0
            support_totals["pre_roll_only_seconds"] += seconds * interval["pre_roll_only"]
            support_totals["live_confirmation_support_seconds"] += (
                seconds * interval["live_confirmation_support"]
            )
            support_totals["model_input_seconds"] += seconds * interval["model_input_present"]
        scores = (session.probabilities[:, 0] < float(cfg["current_anchor_threshold"])).astype(
            np.float32
        )
        rows.append(
            session_row(
                session,
                scores,
                condition="VAD-hygiene",
                probe_class="S-current-corrected-support",
                threshold=0.5,
                confirmation_ms=500,
                time_condition="causal",
                confirmation_support=[
                    (int(value["start_sample"]), int(value["end_sample"]))
                    for value in intervals
                    if value["live_confirmation_support"]
                ],
            )
        )
    interval_path = RESULTS_ROOT / "vad_support_intervals.jsonl"
    write_jsonl(interval_path, all_intervals)
    prior_path = ISSUE98_RESULTS / "eval" / "production_vad_sensitivity.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    prior_row = next(
        value
        for value in prior["rows"]
        if value["family"] == "streaming_sortformer"
        and value["arm"] == "s1_oracle_anchor"
        and value["candidate"] == "simple_anchor"
        and value["replacement_confirm_ms"] == 500
    )
    result = {
        "schema_version": "psem.frozen_ceiling.vad_support_hygiene.v1",
        "speaker_model_inference_performed": False,
        "vad_inference_performed": False,
        "vad_retuned": False,
        "pre_roll_ms": cfg["vad_pre_roll_ms"],
        "support_rule": "first 500ms of each recorded issue98 VAD span is context-only; remainder is committed-live confirmation support",
        "support_totals": support_totals,
        "support_intervals": {
            "path": str(interval_path.relative_to(REPOSITORY_ROOT)).replace("\\", "/"),
            "sha256": sha256_file(interval_path),
            "row_count": len(all_intervals),
        },
        "corrected_cached_posterior_replay": aggregate_rows(rows),
        "issue98_mixed_support_reference": prior_row["production_vad"],
        "issue98_reference_sha256": sha256_file(prior_path),
        "interpretation_scope": "integration hygiene only; not teacher-path selection or a VAD-gating verdict",
    }
    write_json(RESULTS_ROOT / "vad_support_hygiene.json", result)
    return result


if __name__ == "__main__":
    print(run()["support_totals"])
