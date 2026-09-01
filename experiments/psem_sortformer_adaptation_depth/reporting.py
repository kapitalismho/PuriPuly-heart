from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.psem_sortformer_adaptation_depth.authority_registry import (
    require_registered_execution,
)
from experiments.psem_sortformer_adaptation_depth.evaluation import (
    evaluate_prediction_set,
    source_family_frontiers,
    validate_prediction_set,
)
from experiments.psem_sortformer_adaptation_depth.protocol import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    LOWER_IS_BETTER,
    bind_payload,
    require_bound,
    validate_eval_authorization,
)
from experiments.psem_sortformer_adaptation_depth.receipts import (
    evaluator_reconstruction_contract,
    paired_source_bootstrap_v1,
)
from experiments.psem_training_strategy_gate.sampling import EVAL_ROLE

DEPTH_ORDER = ("F0-FROZEN-FLOAT", "H-HEAD", "T2-TOP", "TA-ALL-TEMPORAL")
DECISION_METRIC_DIRECTIONS = {
    "contamination": "lower",
    "speaker_induced_cuts": "lower",
    "false_cuts": "lower",
    "missed_replacements": "lower",
    "replacement_delay_p50": "lower",
    "replacement_delay_p90": "lower",
    "backdated_boundary_error_p50": "lower",
    "backdated_boundary_error_p90": "lower",
    "overlap_return_preservation": "higher",
    "overlap_takeover_success": "higher",
}
CORE_GAIN_METRICS = ("contamination", "missed_replacements")
TOPOLOGY_METRICS = ("overlap_return_preservation", "overlap_takeover_success")
MANDATORY_TOPOLOGIES = (
    "clean_direct_different_speaker_handoff",
    "silence_gap_different_speaker_handoff",
    "same_speaker_silence_gap_resume",
    "overlap_return",
    "overlap_takeover",
    "short_backchannel_return",
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ISSUE99_RESULTS = (
    REPOSITORY_ROOT / "experiments" / "psem_frozen_ceiling_gate" / "results" / "frozen_ceiling_1"
)


class ReportingError(RuntimeError):
    pass


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _legacy_validate_eval_result(value: Mapping[str, Any]) -> dict[str, Any]:
    require_registered_execution("evaluation-result", value)
    payload = require_bound(value, "psem_sortformer_eval_result")
    if (
        payload.get("split_role") != EVAL_ROLE
        or payload.get("evaluation_roles") != [EVAL_ROLE]
        or payload.get("eval_open_count") != 1
        or type(payload.get("slot_mapping_coverage_passed")) is not bool
        or type(payload.get("timing_gate_passed")) is not bool
        or payload.get("passed")
        is not (payload["slot_mapping_coverage_passed"] and payload["timing_gate_passed"])
        or payload.get("arm") not in DEPTH_ORDER
    ):
        raise ReportingError("EVAL result identity or integrity gate is invalid")
    frontier = payload.get("frontier")
    if not isinstance(frontier, list) or len(frontier) != 9:
        raise ReportingError("EVAL result lacks the complete fixed frontier")
    cells = {(row.get("threshold"), row.get("confirmation_ms")) for row in frontier}
    expected = {
        (threshold, confirmation)
        for threshold in (0.35, 0.5, 0.65)
        for confirmation in (100, 300, 500)
    }
    if cells != expected:
        raise ReportingError("EVAL result frontier cells differ from the fixed grid")
    return dict(value)


def _primary(result: Mapping[str, Any]) -> Mapping[str, Any]:
    return next(
        row
        for row in result["frontier"]
        if row["threshold"] == 0.5 and row["confirmation_ms"] == 500
    )


def _result_id(value: Mapping[str, Any]) -> tuple[str, int | None]:
    return str(value["arm"]), value.get("seed")


def _candidate_results(
    eval_authorization: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    prediction_sets: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    validate_eval_authorization(eval_authorization)
    authorization = require_bound(eval_authorization, "psem_sortformer_eval_open_authorization")
    validated = [validate_eval_result(value) for value in results]
    expected = [(row["arm"], row.get("seed")) for row in authorization["candidate_set"]]
    observed = [_result_id(value) for value in validated]
    if observed != expected or len(set(observed)) != len(observed):
        raise ReportingError("EVAL results differ from the frozen candidate set")
    if any(
        value.get("prediction_set_sha256") in {None, ""}
        or value.get("eval_evidence_sha256") in {None, ""}
        for value in validated
    ):
        raise ReportingError("EVAL results are not bound to predictions and evidence")
    predictions = []
    for value in prediction_sets:
        try:
            validate_prediction_set(value, eval_authorization)
        except Exception as exc:
            raise ReportingError("EVAL prediction set is not reproducible") from exc
        predictions.append(value)
    if [_result_id(value) for value in predictions] != expected:
        raise ReportingError("EVAL prediction identities differ from the candidate set")
    for candidate, result, prediction in zip(
        authorization["candidate_set"], validated, predictions, strict=True
    ):
        try:
            recomputed = evaluate_prediction_set(prediction, eval_authorization)
        except Exception as exc:
            raise ReportingError(
                "EVAL result cannot be recomputed from prediction evidence"
            ) from exc
        if (
            result != recomputed
            or result.get("prediction_set_sha256") != prediction.get("payload_sha256")
            or result.get("prediction_set") != prediction
            or prediction.get("trained_checkpoint_sha256") != candidate.get("checkpoint_sha256")
            or prediction.get("trained_checkpoint_receipt_sha256")
            != candidate.get("checkpoint_receipt_sha256")
        ):
            raise ReportingError("EVAL result used a prediction or checkpoint outside the freeze")
    return validated


def _full_metric_vector(full: Mapping[str, Any]) -> dict[str, float | None]:
    hours = full.get("active_speech_hours")
    if not _finite(hours) or float(hours) <= 0:
        raise ReportingError("per-source active-speech exposure is invalid")
    replacement_delay = full.get("replacement_emit_delay_ms")
    boundary_error = full.get("backdated_boundary_error_ms")
    return {
        "contamination": full.get("exclusive_other_contamination_seconds_per_active_speech_hour"),
        "speaker_induced_cuts": full.get("speaker_induced_cut_count_per_active_speech_hour"),
        "false_cuts": float(full["false_cut_count"]) / float(hours),
        "missed_replacements": float(full["missed_replacement_count"]) / float(hours),
        "replacement_delay_p50": (
            replacement_delay.get("p50") if isinstance(replacement_delay, Mapping) else None
        ),
        "replacement_delay_p90": (
            replacement_delay.get("p90") if isinstance(replacement_delay, Mapping) else None
        ),
        "backdated_boundary_error_p50": (
            boundary_error.get("p50") if isinstance(boundary_error, Mapping) else None
        ),
        "backdated_boundary_error_p90": (
            boundary_error.get("p90") if isinstance(boundary_error, Mapping) else None
        ),
        "overlap_return_preservation": full.get("overlap_return_preservation_rate"),
        "overlap_takeover_success": full.get("overlap_takeover_success_rate"),
    }


def _result_per_source(result: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    def topology_values(full: Mapping[str, Any]) -> dict[str, float | None]:
        topology = full.get("topology")
        if not isinstance(topology, Mapping):
            raise ReportingError("per-source topology evidence is absent")
        values = {}
        for name in MANDATORY_TOPOLOGIES:
            row = topology.get(name)
            if not isinstance(row, Mapping):
                raise ReportingError(f"mandatory topology evidence is absent: {name}")
            eligible = row.get("eligible_episode_count")
            if not _finite(eligible) or float(eligible) < 0:
                raise ReportingError(f"topology exposure is invalid: {name}")
            if name == "overlap_return":
                value = row.get("overlap_return_preservation_rate")
            elif name == "overlap_takeover":
                value = row.get("overlap_takeover_success_rate")
            elif name in {
                "clean_direct_different_speaker_handoff",
                "silence_gap_different_speaker_handoff",
            }:
                denominator = row.get("episodes_with_reference_replacement")
                numerator = row.get("episodes_with_aligned_cut")
                value = (
                    float(numerator) / float(denominator)
                    if _finite(numerator) and _finite(denominator) and float(denominator) > 0
                    else None
                )
            else:
                numerator = row.get("episodes_with_predicted_cut")
                value = (
                    -float(numerator) / float(eligible)
                    if _finite(numerator) and float(eligible) > 0
                    else None
                )
            values[name] = float(value) if _finite(value) else None
        return values

    return {
        row["source_id"]: {
            "source_id": row["source_id"],
            "corpus": row["corpus"],
            "metrics": _full_metric_vector(row["full_metrics"]),
            "topology_values": topology_values(row["full_metrics"]),
        }
        for row in result["per_source_primary"]
    }


def _arm_average_per_source(
    results: Sequence[Mapping[str, Any]], arm: str
) -> dict[str, dict[str, Any]]:
    chosen = [value for value in results if value["arm"] == arm]
    if not chosen:
        raise ReportingError(f"arm has no EVAL result: {arm}")
    by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for result in chosen:
        for row in _result_per_source(result).values():
            by_source[row["source_id"]].append(row)
    if any(len(rows) != len(chosen) for rows in by_source.values()):
        raise ReportingError(f"seed source coverage differs for arm: {arm}")
    averaged = {}
    for source_id, rows in sorted(by_source.items()):
        if len({row["corpus"] for row in rows}) != 1:
            raise ReportingError("source corpus identity changes across seeds")
        metrics = {}
        support = {}
        for metric in DECISION_METRIC_DIRECTIONS:
            raw_values = [row["metrics"][metric] for row in rows]
            values = [float(value) for value in raw_values if _finite(value)]
            metrics[metric] = sum(values) / len(values) if len(values) == len(rows) else None
            support[metric] = len(values)
        averaged[source_id] = {
            "source_id": source_id,
            "corpus": rows[0]["corpus"],
            "metrics": metrics,
            "seed_support_by_metric": support,
            "topology_values": {
                topology: (sum(values) / len(values) if len(values) == len(rows) else None)
                for topology in MANDATORY_TOPOLOGIES
                for values in [
                    [
                        float(row["topology_values"][topology])
                        for row in rows
                        if _finite(row["topology_values"][topology])
                    ]
                ]
            },
        }
    return averaged


def _unsupported_interval(
    deltas: Mapping[str, float],
    *,
    invalid_source_ids: Sequence[str],
    seed: int,
    reason: str = "missing_or_nonfinite_paired_source_metric",
) -> dict[str, Any]:
    return {
        "status": "unsupported",
        "reason": reason,
        "lower": None,
        "upper": None,
        "replicate_estimates_sha256": None,
        "seed": seed,
        "resamples": BOOTSTRAP_RESAMPLES,
        "algorithm": "paired_source_bootstrap_v1",
        "unit": "source_or_meeting",
        "paired_source_deltas": dict(deltas),
        "point_estimate": None,
        "support_source_count": len(deltas),
        "invalid_source_ids": list(invalid_source_ids),
    }


def _supported_interval(deltas: Mapping[str, float], *, seed: int) -> dict[str, Any]:
    return {
        **_bootstrap_delta(deltas, seed=seed),
        "status": "supported",
        "invalid_source_ids": [],
    }


def _interval_for_sources(
    left: Mapping[str, Mapping[str, Any]],
    right: Mapping[str, Mapping[str, Any]],
    source_ids: Sequence[str],
    metric: str,
    direction: str,
    *,
    seed: int,
) -> dict[str, Any]:
    if not source_ids:
        return _unsupported_interval(
            {},
            invalid_source_ids=[],
            seed=seed,
            reason="no_paired_source_metric_support",
        )
    invalid = []
    deltas = {}
    for source_id in source_ids:
        left_value = left[source_id]["metrics"][metric]
        right_value = right[source_id]["metrics"][metric]
        if not _finite(left_value) or not _finite(right_value):
            invalid.append(source_id)
            continue
        delta = float(left_value) - float(right_value)
        deltas[source_id] = delta if direction == "lower" else -delta
    if invalid:
        return _unsupported_interval(deltas, invalid_source_ids=invalid, seed=seed)
    return _supported_interval(deltas, seed=seed)


def _bootstrap_delta(deltas: Mapping[str, float], *, seed: int) -> dict[str, Any]:
    if not deltas:
        raise ReportingError("paired source bootstrap has no defined metric support")
    interval = paired_source_bootstrap_v1(
        deltas,
        seed=seed,
        resamples=BOOTSTRAP_RESAMPLES,
    )
    return {
        **interval,
        "seed": seed,
        "resamples": BOOTSTRAP_RESAMPLES,
        "algorithm": "paired_source_bootstrap_v1",
        "unit": "source_or_meeting",
        "paired_source_deltas": dict(deltas),
        "point_estimate": sum(deltas.values()) / len(deltas),
        "support_source_count": len(deltas),
    }


def _paired_source_maps(
    left: Mapping[str, Mapping[str, Any]],
    right: Mapping[str, Mapping[str, Any]],
    left_arm: str,
    right_arm: str,
    seed_offset: int,
) -> dict[str, Any]:
    if set(left) != set(right):
        raise ReportingError("paired EVAL arm source coverage differs")
    source_ids = sorted(left)
    metrics = {}
    corpus_metrics = {"AMI": {}, "AliMeeting": {}}
    for metric_index, (metric, direction) in enumerate(DECISION_METRIC_DIRECTIONS.items()):
        metrics[metric] = _interval_for_sources(
            left,
            right,
            source_ids,
            metric,
            direction,
            seed=BOOTSTRAP_SEED + seed_offset + metric_index,
        )
        for corpus_index, corpus in enumerate(("AMI", "AliMeeting")):
            corpus_source_ids = [
                source_id for source_id in source_ids if left[source_id]["corpus"] == corpus
            ]
            corpus_metrics[corpus][metric] = _interval_for_sources(
                left,
                right,
                corpus_source_ids,
                metric,
                direction,
                seed=BOOTSTRAP_SEED + seed_offset + 20 + corpus_index * 10 + metric_index,
            )
    return {
        "left_arm": left_arm,
        "right_arm": right_arm,
        "delta_direction": "negative_is_favorable",
        "metric_directions": DECISION_METRIC_DIRECTIONS,
        "metrics": metrics,
        "corpus_specific": corpus_metrics,
    }


def _paired_comparison(
    results: Sequence[Mapping[str, Any]],
    left_arm: str,
    right_arm: str,
    seed_offset: int,
) -> dict[str, Any]:
    left = _arm_average_per_source(results, left_arm)
    right = _arm_average_per_source(results, right_arm)
    return _paired_source_maps(left, right, left_arm, right_arm, seed_offset)


def _legacy_build_bootstrap_report(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    arms = [arm for arm in DEPTH_ORDER if any(value["arm"] == arm for value in results)]
    comparisons = []
    offset = 0
    for left_index, left in enumerate(arms):
        for right in arms[:left_index]:
            comparisons.append(_paired_comparison(results, left, right, offset))
            offset += 100
    return bind_payload(
        {
            "schema_version": 1,
            "artifact_role": "psem_sortformer_bootstrap_intervals",
            "resamples": BOOTSTRAP_RESAMPLES,
            "unit": "source_or_meeting",
            "frame_bootstrap_used": False,
            "decision_metric_directions": DECISION_METRIC_DIRECTIONS,
            "comparisons": comparisons,
        }
    )


def _comparison(bootstrap: Mapping[str, Any], left: str, right: str) -> Mapping[str, Any]:
    return next(
        row
        for row in bootstrap["comparisons"]
        if row["left_arm"] == left and row["right_arm"] == right
    )


def _stable_gain(comparison: Mapping[str, Any]) -> bool:
    def supported(interval: Mapping[str, Any]) -> bool:
        return interval.get("status", "supported") == "supported"

    def favorable(interval: Mapping[str, Any]) -> bool:
        return (
            supported(interval) and _finite(interval.get("upper")) and float(interval["upper"]) < 0
        )

    def harmful(interval: Mapping[str, Any]) -> bool:
        return (
            supported(interval) and _finite(interval.get("lower")) and float(interval["lower"]) > 0
        )

    complete_support = all(
        supported(comparison["metrics"][metric])
        and all(
            supported(comparison["corpus_specific"][corpus][metric])
            for corpus in ("AMI", "AliMeeting")
        )
        for metric in DECISION_METRIC_DIRECTIONS
    )

    pooled_core_gain = any(favorable(comparison["metrics"][metric]) for metric in CORE_GAIN_METRICS)
    both_corpora_gain = all(
        any(
            favorable(comparison["corpus_specific"][corpus][metric]) for metric in CORE_GAIN_METRICS
        )
        for corpus in ("AMI", "AliMeeting")
    )
    full_metric_harm = sum(
        harmful(comparison["metrics"][metric]) for metric in DECISION_METRIC_DIRECTIONS
    )
    corpus_metric_harm = {
        corpus: sum(
            harmful(comparison["corpus_specific"][corpus][metric])
            for metric in DECISION_METRIC_DIRECTIONS
        )
        for corpus in ("AMI", "AliMeeting")
    }
    return bool(
        complete_support
        and pooled_core_gain
        and both_corpora_gain
        and full_metric_harm == 0
        and all(value == 0 for value in corpus_metric_harm.values())
    )


def _meaningfully_dominated(bootstrap: Mapping[str, Any], shallow: str, deep: str) -> bool:
    comparison = _comparison(bootstrap, deep, shallow)
    favorable = sum(
        _finite(comparison["metrics"][metric].get("upper"))
        and comparison["metrics"][metric]["upper"] < 0
        for metric in DECISION_METRIC_DIRECTIONS
    )
    core_favorable = any(
        _finite(comparison["metrics"][metric].get("upper"))
        and comparison["metrics"][metric]["upper"] < 0
        for metric in CORE_GAIN_METRICS
    )
    return bool(core_favorable and favorable >= 2 and _stable_gain(comparison))


def _seed_stability_evidence(results: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    chosen = [value for value in results if value["arm"] == arm]
    f0 = next(value for value in results if value["arm"] == "F0-FROZEN-FLOAT")
    rows = []
    for index, value in enumerate(chosen):
        comparison = _paired_source_maps(
            _result_per_source(value),
            _result_per_source(f0),
            f"{arm}:{value.get('seed')}",
            "F0-FROZEN-FLOAT",
            1000 + index * 100,
        )
        rows.append(
            {
                "seed": value.get("seed"),
                "stable_gain_vs_f0": _stable_gain(comparison),
                "comparison": comparison,
            }
        )
    return {
        "confirmation_seed_count": len(chosen),
        "all_seeds_stable_vs_f0": len(chosen) >= 2
        and all(row["stable_gain_vs_f0"] for row in rows),
        "seeds": rows,
    }


def _seed_stable(results: Sequence[Mapping[str, Any]], arm: str) -> bool:
    return _seed_stability_evidence(results, arm)["all_seeds_stable_vs_f0"]


def _ta_concentration_evidence(
    results: Sequence[Mapping[str, Any]],
    bootstrap: Mapping[str, Any],
    shallower: Sequence[str],
) -> dict[str, Any]:
    topology = {}
    for arm in shallower:
        comparison = _comparison(bootstrap, "TA-ALL-TEMPORAL", arm)
        topology[arm] = {
            metric: bool(
                _finite(comparison["metrics"][metric].get("upper"))
                and comparison["metrics"][metric]["upper"] < 0
            )
            for metric in TOPOLOGY_METRICS
        }
    ta = _arm_average_per_source(results, "TA-ALL-TEMPORAL")
    mandatory_topology = {}
    for arm_index, arm in enumerate(shallower):
        baseline = _arm_average_per_source(results, arm)
        if set(ta) != set(baseline):
            raise ReportingError("TA topology comparison source coverage differs")
        topology_intervals = {}
        for topology_index, topology_name in enumerate(MANDATORY_TOPOLOGIES):
            invalid = []
            deltas = {}
            for source_id in sorted(ta):
                ta_value = ta[source_id]["topology_values"][topology_name]
                baseline_value = baseline[source_id]["topology_values"][topology_name]
                if not _finite(ta_value) or not _finite(baseline_value):
                    invalid.append(source_id)
                    continue
                deltas[source_id] = float(ta_value) - float(baseline_value)
            normalized = {source_id: -value for source_id, value in deltas.items()}
            seed = BOOTSTRAP_SEED + 12000 + arm_index * 1000 + topology_index
            topology_intervals[topology_name] = (
                _unsupported_interval(normalized, invalid_source_ids=invalid, seed=seed)
                if invalid
                else _supported_interval(normalized, seed=seed)
            )
        mandatory_topology[arm] = topology_intervals
    leave_one_out = {}
    for arm_index, arm in enumerate(shallower):
        baseline = _arm_average_per_source(results, arm)
        arm_rows = {}
        for source_index, source_id in enumerate(sorted(ta)):
            left = {key: value for key, value in ta.items() if key != source_id}
            right = {key: value for key, value in baseline.items() if key != source_id}
            comparison = _paired_source_maps(
                left,
                right,
                "TA-ALL-TEMPORAL",
                arm,
                2000 + arm_index * 3000 + source_index * 100,
            )
            arm_rows[source_id] = _stable_gain(comparison)
        leave_one_out[arm] = arm_rows
    topology_concentration_pass = all(
        sum(
            interval.get("status") == "supported"
            and _finite(interval.get("upper"))
            and interval["upper"] < 0
            for interval in intervals.values()
        )
        >= 2
        and all(
            interval.get("status") == "supported"
            and _finite(interval.get("lower"))
            and interval["lower"] <= 0
            for interval in intervals.values()
        )
        for intervals in mandatory_topology.values()
    )
    return {
        "both_overlap_topologies_favorable_vs_each_shallower": all(
            all(values.values()) for values in topology.values()
        ),
        "stable_after_leaving_out_each_source_vs_each_shallower": all(
            all(values.values()) for values in leave_one_out.values()
        ),
        "mandatory_topology_concentration_pass": topology_concentration_pass,
        "mandatory_topology_bootstrap_by_shallower_arm": mandatory_topology,
        "topology_bootstrap_pass_by_shallower_arm": topology,
        "leave_one_source_out_pass_by_shallower_arm": leave_one_out,
    }


def _legacy_decide_outcome(
    results: Sequence[Mapping[str, Any]],
    bootstrap: Mapping[str, Any],
    candidate_set: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    arms = [arm for arm in DEPTH_ORDER if any(value["arm"] == arm for value in results)]
    gains = {
        (left, right): _stable_gain(_comparison(bootstrap, left, right))
        for left in arms[1:]
        for right in arms[: arms.index(left)]
    }
    supported = []
    seed_evidence = {
        arm: _seed_stability_evidence(results, arm) for arm in arms if arm != "F0-FROZEN-FLOAT"
    }
    if (
        "H-HEAD" in arms
        and gains[("H-HEAD", "F0-FROZEN-FLOAT")]
        and seed_evidence["H-HEAD"]["all_seeds_stable_vs_f0"]
        and not any(
            deeper in arms
            and (
                gains.get((deeper, "H-HEAD"), False)
                or _meaningfully_dominated(bootstrap, "H-HEAD", deeper)
            )
            for deeper in ("T2-TOP", "TA-ALL-TEMPORAL")
        )
    ):
        supported.append("H-HEAD")
    if (
        "T2-TOP" in arms
        and gains[("T2-TOP", "F0-FROZEN-FLOAT")]
        and gains[("T2-TOP", "H-HEAD")]
        and seed_evidence["T2-TOP"]["all_seeds_stable_vs_f0"]
        and not (
            "TA-ALL-TEMPORAL" in arms
            and (
                gains.get(("TA-ALL-TEMPORAL", "T2-TOP"), False)
                or _meaningfully_dominated(bootstrap, "T2-TOP", "TA-ALL-TEMPORAL")
            )
        )
    ):
        supported.append("T2-TOP")
    if (
        "TA-ALL-TEMPORAL" in arms
        and all(
            gains[("TA-ALL-TEMPORAL", shallower)]
            for shallower in ("F0-FROZEN-FLOAT", "H-HEAD", "T2-TOP")
        )
        and seed_evidence["TA-ALL-TEMPORAL"]["all_seeds_stable_vs_f0"]
    ):
        ta_concentration = _ta_concentration_evidence(
            results,
            bootstrap,
            ("H-HEAD", "T2-TOP"),
        )
        if ta_concentration["both_overlap_topologies_favorable_vs_each_shallower"]:
            supported.append("TA-ALL-TEMPORAL")
    else:
        ta_concentration = None
    ta_present = "TA-ALL-TEMPORAL" in arms
    integrity_damage = any(
        value.get("slot_mapping_coverage_passed") is not True
        or value.get("timing_gate_passed") is not True
        for value in results
    )
    if candidate_set is not None:
        integrity_damage = integrity_damage or any(
            row.get("training_summary", {}).get("native_diarization_contract_passed") is not True
            for row in candidate_set
        )
    if integrity_damage:
        supported = []
    selected = supported[0] if supported else None
    outcome = {
        "H-HEAD": "A",
        "T2-TOP": "B",
        "TA-ALL-TEMPORAL": "C",
        None: "D",
    }[selected]
    selected_comparison = (
        _comparison(bootstrap, selected, "F0-FROZEN-FLOAT") if selected is not None else None
    )
    both_corpora = bool(selected_comparison and _stable_gain(selected_comparison))
    overlap_return = bool(
        selected_comparison
        and all(
            _finite(interval.get("upper")) and interval["upper"] < 0
            for interval in (
                selected_comparison["metrics"]["overlap_return_preservation"],
                selected_comparison["corpus_specific"]["AMI"]["overlap_return_preservation"],
                selected_comparison["corpus_specific"]["AliMeeting"]["overlap_return_preservation"],
            )
        )
    )
    overlap_takeover = bool(
        selected_comparison
        and all(
            _finite(interval.get("upper")) and interval["upper"] < 0
            for interval in (
                selected_comparison["metrics"]["overlap_takeover_success"],
                selected_comparison["corpus_specific"]["AMI"]["overlap_takeover_success"],
                selected_comparison["corpus_specific"]["AliMeeting"]["overlap_takeover_success"],
            )
        )
    )
    return {
        "evidence_level": "engineering",
        "outcome": outcome,
        "selected_minimum_adaptation_depth": selected,
        "supported_arms": supported,
        "head_materially_improves_f0": (
            "H-HEAD" in arms and _stable_gain(_comparison(bootstrap, "H-HEAD", "F0-FROZEN-FLOAT"))
        ),
        "top2_stably_improves_head": (
            "T2-TOP" in arms and _stable_gain(_comparison(bootstrap, "T2-TOP", "H-HEAD"))
        ),
        "ta_was_opened": ta_present,
        "seed_stability_by_arm": {
            arm: value["all_seeds_stable_vs_f0"] for arm, value in seed_evidence.items()
        },
        "seed_stability_evidence_by_arm": seed_evidence,
        "ta_concentration_evidence": ta_concentration,
        "improvement_holds_in_both_corpora": both_corpora,
        "improvement_holds_in_overlap_return": overlap_return,
        "improvement_holds_in_overlap_takeover": overlap_takeover,
        "slot_timing_or_original_diarization_contract_damage": integrity_damage,
        "adapted_teacher_ready_for_native_causal_binding_gate": outcome in {"A", "B", "C"}
        and not integrity_damage,
        "adapted_teacher_readiness_scope": "engineering_next-stage_experiment_only",
        "stop_training": outcome == "D",
    }


def _arm_metrics(results: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    chosen = [value for value in results if value["arm"] == arm]
    return {
        "schema_version": 1,
        "artifact_role": "psem_sortformer_arm_metrics",
        "arm": arm,
        "seeds": [value.get("seed") for value in chosen],
        "results": chosen,
    }


def _topology_report(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = []
    required_episode_slices = {
        "clean_direct_different_speaker_handoff",
        "silence_gap_different_speaker_handoff",
        "same_speaker_silence_gap_resume",
        "overlap_return",
        "overlap_takeover",
        "short_backchannel_return",
    }
    for result in results:
        primary = _primary(result)
        pooled_topology = primary["views"]["pooled"]["full_metrics"]["topology"]
        frame_diagnostics = result["frame_diagnostics"]
        if not required_episode_slices <= set(pooled_topology) or any(
            not {
                "anchor_only",
                "anchor_with_overlap",
                "active_anchor_absent",
                "gt_overlap_anchor_dropout",
            }
            <= set(source)
            for source in frame_diagnostics.values()
        ):
            raise ReportingError("mandatory topology or frame-diagnostic slices are incomplete")
        rows.append(
            {
                "arm": result["arm"],
                "seed": result.get("seed"),
                "pooled_topology": pooled_topology,
                "equal_corpus_topology": primary["views"]["equal_corpus"]["full_metrics"][
                    "topology"
                ],
                "corpus_specific_topology": {
                    corpus: value["full_metrics"]["topology"]
                    for corpus, value in primary["views"]["corpus_specific"].items()
                },
                "frame_diagnostics": frame_diagnostics,
                "frame_diagnostic_views": result["frame_diagnostic_views"],
                "oracle_mapping_and_reset_diagnostics": result["mapping_diagnostics"],
                "oracle_mapping_and_reset_views": result["mapping_diagnostic_views"],
                "continuation_slice_mapping": {
                    "stable_anchor_only_continuation": "frame_diagnostics.anchor_only",
                    "ongoing_anchor_with_overlap_continuation": "frame_diagnostics.anchor_with_overlap",
                },
            }
        )
    return {
        "schema_version": 1,
        "artifact_role": "psem_sortformer_topology_slices",
        "required_slices": [
            "clean_direct_different_speaker_handoff",
            "silence_gap_different_speaker_handoff",
            "same_speaker_silence_gap_resume",
            "overlap_return",
            "overlap_takeover",
            "short_backchannel_return",
            "stable_anchor_only_continuation",
            "ongoing_anchor_with_overlap_continuation",
        ],
        "rows": rows,
    }


def _timing_compute_report(
    results: Sequence[Mapping[str, Any]],
    training_results: Sequence[Mapping[str, Any]],
    candidate_set: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    training_by_id = {(value.get("arm"), value.get("seed")): value for value in training_results}
    candidates = {(value.get("arm"), value.get("seed")): value for value in candidate_set}
    expected_training_ids = {key for key in candidates if key[0] != "F0-FROZEN-FLOAT"}
    if set(training_by_id) != expected_training_ids or len(training_by_id) != len(training_results):
        raise ReportingError("training result coverage differs from the frozen candidate set")
    rows = []
    for result in results:
        key = (result["arm"], result.get("seed"))
        candidate = candidates.get(key)
        training = training_by_id.get(key)
        summary = candidate.get("training_summary") if isinstance(candidate, Mapping) else None
        if not isinstance(summary, Mapping):
            raise ReportingError("candidate training summary is absent")
        if key[0] == "F0-FROZEN-FLOAT":
            if training is not None or summary.get("not_trained") is not True:
                raise ReportingError("F0 compute evidence is not the frozen no-training path")
        else:
            payload = (
                {name: item for name, item in training.items() if name != "payload_sha256"}
                if isinstance(training, Mapping)
                else None
            )
            if (
                payload is None
                or training.get("artifact_role") != "psem_sortformer_training_result"
                or training.get("payload_sha256") != bind_payload(payload)["payload_sha256"]
                or training.get("payload_sha256") != candidate.get("training_result_sha256")
                or training.get("checkpoint_sha256") != candidate.get("checkpoint_sha256")
                or training.get("training_summary") != summary
                or type(training.get("peak_training_memory_bytes")) is not int
                or training["peak_training_memory_bytes"] <= 0
                or any(
                    training.get(field) != summary.get(field)
                    for field in (
                        "training_wall_clock_seconds",
                        "peak_training_memory_bytes",
                        "total_parameters",
                        "trainable_parameters",
                    )
                )
            ):
                raise ReportingError("training compute evidence is not bound to its checkpoint")
            require_registered_execution("training-result", training)
        rows.append(
            {
                "arm": key[0],
                "seed": key[1],
                "algorithmic_evidence_delay_samples": 16640,
                "native_frame_samples": 1280,
                "timing_gate_passed": result["timing_gate_passed"],
                "total_parameters": summary["total_parameters"],
                "trainable_parameters": summary["trainable_parameters"],
                "peak_training_memory_bytes": summary["peak_training_memory_bytes"],
                "training_wall_clock_seconds": summary["training_wall_clock_seconds"],
                "native_diarization_contract_passed": summary["native_diarization_contract_passed"],
                "native_diarization_contract_evidence_sha256": summary[
                    "native_diarization_contract_evidence_sha256"
                ],
                "checkpoint_receipt_sha256": candidate.get("checkpoint_receipt_sha256"),
            }
        )
    return {
        "schema_version": 1,
        "artifact_role": "psem_sortformer_timing_and_compute",
        "rows": rows,
    }


def _historical_metric_vector(metrics: Mapping[str, Any]) -> dict[str, float]:
    hours = float(metrics["active_speech_hours"])
    if hours <= 0:
        raise ReportingError("historical baseline has no active-speech exposure")
    return {
        "contamination": float(
            metrics["exclusive_other_contamination_seconds_per_active_speech_hour"]
        ),
        "false_cuts": float(metrics["false_cut_count"]) / hours,
        "missed_replacements": float(metrics["missed_replacement_count"]) / hours,
    }


def _historical_baseline_comparisons(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    evaluator_reconstruction_contract()
    g_path = ISSUE99_RESULTS / "gt_causal_action_frontier.json"
    q8_path = ISSUE99_RESULTS / "scalar_current_metrics.json"
    g_value = json.loads(g_path.read_text(encoding="utf-8"))
    q8_value = json.loads(q8_path.read_text(encoding="utf-8"))
    g_row = next(
        row for row in g_value["rows"] if row["condition"] == "G" and row["confirmation_ms"] == 500
    )
    q8_row = next(
        row
        for row in q8_value["rows"]
        if row["condition"] == "S-current" and row["confirmation_ms"] == 500
    )
    g_metrics = _historical_metric_vector(g_row["metrics"])
    q8_metrics = _historical_metric_vector(q8_row["metrics"])
    f0_result = next(value for value in results if value["arm"] == "F0-FROZEN-FLOAT")
    f0_metrics = {
        key: float(value)
        for key, value in _primary(f0_result)["views"]["pooled"]["metrics"].items()
    }
    rows = []
    for result in results:
        metrics = {
            key: float(value)
            for key, value in _primary(result)["views"]["pooled"]["metrics"].items()
        }
        closure = {}
        for metric in LOWER_IS_BETTER:
            denominator = f0_metrics[metric] - g_metrics[metric]
            closure[metric] = (
                (f0_metrics[metric] - metrics[metric]) / denominator if denominator != 0 else None
            )
        rows.append(
            {
                "arm": result["arm"],
                "seed": result.get("seed"),
                "absolute_metrics": metrics,
                "delta_vs_issue99_G": {
                    metric: metrics[metric] - g_metrics[metric] for metric in LOWER_IS_BETTER
                },
                "delta_vs_issue99_Q8_S_current": {
                    metric: metrics[metric] - q8_metrics[metric] for metric in LOWER_IS_BETTER
                },
                "delta_vs_F0_FROZEN_FLOAT": {
                    metric: metrics[metric] - f0_metrics[metric] for metric in LOWER_IS_BETTER
                },
                "normalized_closure_of_F0_to_G_gap": closure,
            }
        )
    return {
        "schema_version": 1,
        "artifact_role": "psem_sortformer_historical_baseline_comparisons",
        "metric_units": {
            "contamination": "seconds_per_active_speech_hour",
            "false_cuts": "count_per_active_speech_hour",
            "missed_replacements": "count_per_active_speech_hour",
        },
        "issue99_G_primary": g_metrics,
        "issue99_Q8_S_current_primary": q8_metrics,
        "F0_FROZEN_FLOAT_primary": f0_metrics,
        "gap_closure_is_descriptive_only": True,
        "rows": rows,
    }


def render_decision_markdown(
    decision: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    bootstrap: Mapping[str, Any],
) -> str:
    selected = decision["selected_minimum_adaptation_depth"] or "none"
    ta_escalation = decision["ta_escalation"]
    favorable_metrics = sorted(ta_escalation.get("favorable_intervals", {}))
    ta_reason = (
        "The all-temporal arm was opened by the frozen DEV rule and completed; favorable "
        f"paired intervals were observed for {', '.join(favorable_metrics)} and the per-corpus "
        f"harm counts were {ta_escalation.get('wholly_harmful_metric_counts_by_corpus')}."
        if decision["ta_was_opened"]
        else "The all-temporal arm was not opened because the frozen DEV escalation receipt "
        "did not satisfy every Pareto, paired-interval, corpus-harm, mapping, and timing condition."
    )
    outcome_text = {
        "A": "A realistic causal head on the frozen Sortformer backbone is the shallowest supported depth.",
        "B": "Final-two-block temporal adaptation is the shallowest supported depth.",
        "C": "All temporal Transformer blocks are required by the observed evidence.",
        "D": "No validated adaptation path was established; training stops before KD or acoustic unfreezing.",
    }[decision["outcome"]]
    return "\n".join(
        [
            "# Sortformer adaptation-depth decision",
            "",
            f"Outcome: **{decision['outcome']}**",
            "",
            outcome_text,
            "",
            "## Required answers",
            "",
            f"1. A realistic head solved a material part of the frozen-float gap: **{'yes' if decision['head_materially_improves_f0'] else 'no'}**.",
            f"2. Final-two-block adaptation added a stable gain over the head: **{'yes' if decision['top2_stably_improves_head'] else 'no'}**.",
            f"3. {ta_reason}",
            f"4. The minimum supported adaptation depth is **{selected}**.",
            f"5. Improvement held in both corpora: **{'yes' if decision['improvement_holds_in_both_corpora'] else 'no'}**. Improvement held in overlap-return for both corpora: **{'yes' if decision['improvement_holds_in_overlap_return'] else 'no'}**. Improvement held in overlap-takeover for both corpora: **{'yes' if decision['improvement_holds_in_overlap_takeover'] else 'no'}**.",
            f"6. Slot mapping, streaming timing, or the original diarization contract was damaged: **{'yes' if decision['slot_timing_or_original_diarization_contract_damage'] else 'no'}**.",
            f"7. An adapted teacher is **{'ready' if decision['adapted_teacher_ready_for_native_causal_binding_gate'] else 'not ready'}** for a separately scoped native causal-binding engineering experiment.",
            "",
            "## Scope and limitations",
            "",
            "This decision is engineering evidence from a trusted sequential execution workflow, not publication-grade research evidence.",
            "",
            "EVAL was development-known from issues #97–#99 and supports only this within-program path decision. It is not a fresh generalization, deployment, or production-readiness claim.",
            "",
            "The experiment did not perform KD, native causal enrollment/lifecycle, production VAD gating, NEST/acoustic-encoder unfreezing, adapters, quantization, export, or deployment benchmarking.",
            "",
            f"Candidate result count: {len(results)}. Bootstrap comparisons: {len(bootstrap['comparisons'])}. Source/meeting resamples per interval: {BOOTSTRAP_RESAMPLES}.",
            "",
        ]
    )


def _legacy_build_final_artifacts(
    *,
    eval_authorization: Mapping[str, Any],
    eval_results: Sequence[Mapping[str, Any]],
    eval_prediction_sets: Sequence[Mapping[str, Any]],
    training_results: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], str]:
    results = _candidate_results(eval_authorization, eval_results, eval_prediction_sets)
    authorization = require_bound(eval_authorization, "psem_sortformer_eval_open_authorization")
    from experiments.psem_sortformer_adaptation_depth.execution import (
        validate_current_candidate_identity,
    )

    validate_current_candidate_identity(
        authorization["candidate_freeze"]["candidate_code_identity"]
    )
    bootstrap = _legacy_build_bootstrap_report(results)
    decision = _legacy_decide_outcome(results, bootstrap, authorization["candidate_set"])
    staged_state = authorization["candidate_freeze"]["staged_execution_state"]
    ta_escalation = staged_state.get("ta_escalation")
    if (
        not isinstance(ta_escalation, Mapping)
        or ta_escalation.get("decision") not in {"opened", "closed"}
        or decision["ta_was_opened"] is not (ta_escalation["decision"] == "opened")
    ):
        raise ReportingError("final TA execution differs from the frozen DEV escalation")
    decision = {**decision, "ta_escalation": dict(ta_escalation)}
    artifacts = {
        "frozen_float_metrics.json": _arm_metrics(results, "F0-FROZEN-FLOAT"),
        "head_metrics.json": _arm_metrics(results, "H-HEAD"),
        "top2_metrics.json": _arm_metrics(results, "T2-TOP"),
        "per_source_metrics.jsonl": [
            row for result in results for row in result["per_source_rows"]
        ],
        "source_family_frontiers.json": {
            "schema_version": 1,
            "artifact_role": "psem_sortformer_source_family_frontiers",
            "results": [source_family_frontiers(result) for result in results],
        },
        "topology_slices.json": _topology_report(results),
        "bootstrap_intervals.json": bootstrap,
        "timing_and_compute.json": _timing_compute_report(
            results, training_results, authorization["candidate_set"]
        ),
        "historical_baseline_comparisons.json": _historical_baseline_comparisons(results),
        "decision_receipt.json": bind_payload(
            {
                "schema_version": 1,
                "artifact_role": "psem_sortformer_adaptation_decision",
                **decision,
                "eval_authorization_sha256": eval_authorization["payload_sha256"],
                "eval_result_sha256s": [value["payload_sha256"] for value in eval_results],
                "bootstrap_sha256": bootstrap["payload_sha256"],
                "explicit_non_goals_performed": [],
            }
        ),
    }
    if any(value["arm"] == "TA-ALL-TEMPORAL" for value in results):
        artifacts["all_temporal_metrics.json"] = _arm_metrics(results, "TA-ALL-TEMPORAL")
    return artifacts, render_decision_markdown(decision, results, bootstrap)


def validate_eval_result(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = require_bound(value, "psem_sortformer_eval_result")
    frontier = payload.get("frontier")
    if (
        payload.get("split_role") != EVAL_ROLE
        or payload.get("evaluation_roles") != [EVAL_ROLE]
        or payload.get("eval_open_count") != 1
        or type(payload.get("slot_mapping_coverage_passed")) is not bool
        or type(payload.get("timing_gate_passed")) is not bool
        or type(payload.get("passed")) is not bool
        or payload.get("passed")
        is not (
            payload["slot_mapping_coverage_passed"] and payload["timing_gate_passed"]
        )
        or not isinstance(frontier, list)
        or len(frontier) != 1
        or payload.get("arm") not in DEPTH_ORDER
    ):
        raise ReportingError("EVAL result identity or lean integrity gate is invalid")
    cell = frontier[0]
    if (
        cell.get("threshold") != 0.5
        or cell.get("confirmation_ms") != 500
        or set(cell.get("views", {})) != {"pooled", "AMI", "AliMeeting"}
    ):
        raise ReportingError("EVAL result must contain only the singleton required views")
    return dict(value)


_LEAN_OUTCOME_BY_ARM = {
    "H-HEAD": "A",
    "T2-TOP": "B",
    "TA-ALL-TEMPORAL": "C",
}


def _render_lean_decision_markdown(
    *,
    outcome: str,
    selected_arm: str | None,
    rationale: str,
    eval_opened: bool,
) -> str:
    selected = selected_arm or "none"
    evidence_lines = (
        [
            "Operating point: replacement cell 0.50 / 500 ms.",
            "Required views: pooled, AMI, AliMeeting.",
            "EVAL contains exactly F0 and the operator-selected seed-7301 candidate.",
        ]
        if eval_opened
        else [
            "EVAL was not opened or used; this STOP decision uses frozen DEV evidence only.",
        ]
    )
    return (
        "\n".join(
            [
                "# PSEM Sortformer adaptation engineering decision",
                "",
                f"Outcome: **{outcome}**",
                f"Selected arm: `{selected}`",
                "",
                rationale,
                "",
                *evidence_lines,
                "",
                "This is trusted single-operator engineering evidence. It makes no significance or seed-stability claim.",
                "",
                "No student KD was performed or authorized; this outcome does not start KD automatically.",
                "All NEST/acoustic-encoder parameters remained frozen; this outcome does not authorize acoustic/NEST unfreezing.",
            ]
        )
        + "\n"
    )


def build_stop_artifacts(*, candidate_freeze: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    from experiments.psem_sortformer_adaptation_depth.protocol import (
        validate_candidate_freeze,
    )

    validate_candidate_freeze(candidate_freeze)
    frozen = require_bound(candidate_freeze, "psem_sortformer_candidate_freeze")
    decision = frozen.get("operator_dev_decision")
    if not isinstance(decision, Mapping):
        raise ReportingError("STOP report lacks its operator DEV decision")
    decision_payload = require_bound(decision, "psem_sortformer_operator_dev_decision")
    if (
        frozen.get("candidate_set") != []
        or frozen.get("eval_open_count") != 0
        or frozen.get("eval_used_for_development") is not False
        or decision_payload.get("decision") != "stop"
        or decision_payload.get("selected_arm") is not None
        or frozen.get("operator_dev_decision_sha256") != decision.get("payload_sha256")
    ):
        raise ReportingError("STOP report requires an exact empty pre-EVAL candidate freeze")
    receipt = bind_payload(
        {
            "schema_version": 1,
            "artifact_role": "psem_sortformer_adaptation_decision",
            "outcome": "D",
            "selected_arm": None,
            "operator_dev_decision_sha256": decision["payload_sha256"],
            "candidate_freeze_sha256": candidate_freeze["payload_sha256"],
            "rationale": decision_payload["rationale"],
            "eval_open_count": 0,
            "eval_result_sha256s": [],
            "significance_claim": False,
            "seed_stability_claim": False,
            "evidence_level": "engineering",
            "stop_training": True,
            "student_kd_performed": False,
            "student_kd_authorized": False,
            "acoustic_or_nest_unfreezing_performed": False,
            "acoustic_or_nest_unfreezing_authorized": False,
        }
    )
    return {"decision_receipt.json": receipt}, _render_lean_decision_markdown(
        outcome="D",
        selected_arm=None,
        rationale=decision_payload["rationale"],
        eval_opened=False,
    )


def build_final_artifacts(
    *,
    eval_authorization: Mapping[str, Any],
    eval_results: Sequence[Mapping[str, Any]],
    eval_prediction_sets: Sequence[Mapping[str, Any]],
    training_results: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], str]:
    validate_eval_authorization(eval_authorization)
    authorization = require_bound(eval_authorization, "psem_sortformer_eval_open_authorization")
    expected = [(row["arm"], row.get("seed")) for row in authorization["candidate_set"]]
    if len(expected) != 2 or expected[0] != ("F0-FROZEN-FLOAT", None):
        raise ReportingError("final engineering report requires F0 plus one selected candidate")
    results = _candidate_results(eval_authorization, eval_results, eval_prediction_sets)
    decision = authorization["candidate_freeze"].get("operator_dev_decision")
    if not isinstance(decision, Mapping):
        raise ReportingError("operator DEV decision is absent from the final report")
    decision_payload = require_bound(decision, "psem_sortformer_operator_dev_decision")
    selected_arm = decision_payload["selected_arm"]
    if decision_payload.get("decision") != "select_candidate" or expected[1] != (
        selected_arm,
        7301,
    ):
        raise ReportingError("operator decision differs from the frozen EVAL winner")
    selected_candidate = authorization["candidate_set"][1]
    training = training_results[0] if len(training_results) == 1 else None
    training_payload = (
        {key: value for key, value in training.items() if key != "payload_sha256"}
        if isinstance(training, Mapping)
        else None
    )
    if (
        training_payload is None
        or training.get("artifact_role") != "psem_sortformer_training_result"
        or training.get("payload_sha256") != bind_payload(training_payload)["payload_sha256"]
        or training.get("payload_sha256") != selected_candidate.get("training_result_sha256")
        or training.get("arm") != selected_arm
        or training.get("seed") != 7301
        or training.get("checkpoint_sha256") != selected_candidate.get("checkpoint_sha256")
        or training.get("candidate_code_identity_sha256")
        != authorization.get("candidate_code_identity_sha256")
        or training.get("training_summary") != selected_candidate.get("training_summary")
        or not isinstance(selected_candidate.get("checkpoint_receipt_sha256"), str)
        or len(selected_candidate["checkpoint_receipt_sha256"]) != 64
    ):
        raise ReportingError("selected training result differs from the frozen candidate")
    outcome = _LEAN_OUTCOME_BY_ARM[selected_arm]
    artifacts = {
        "frozen_float_metrics.json": _arm_metrics(results, "F0-FROZEN-FLOAT"),
        "selected_arm_metrics.json": _arm_metrics(results, selected_arm),
        "singleton_metrics.json": {
            "schema_version": 1,
            "artifact_role": "psem_sortformer_singleton_metrics",
            "operating_point": {"threshold": 0.5, "confirmation_ms": 500},
            "views": {
                f"{value['arm']}:{value.get('seed')}": value["frontier"][0]["views"]
                for value in results
            },
        },
        "per_source_metrics.jsonl": [
            row
            for result in results
            for row in result.get("per_source_rows", [])
            if row.get("threshold") == 0.5 and row.get("confirmation_ms") == 500
        ],
        "timing_and_compute.json": {
            "schema_version": 1,
            "artifact_role": "psem_sortformer_timing_and_compute",
            "training_results": [dict(training)],
        },
        "decision_receipt.json": bind_payload(
            {
                "schema_version": 1,
                "artifact_role": "psem_sortformer_adaptation_decision",
                "outcome": outcome,
                "selected_arm": selected_arm,
                "operator_dev_decision_sha256": decision["payload_sha256"],
                "rationale": decision_payload["rationale"],
                "eval_authorization_sha256": eval_authorization["payload_sha256"],
                "eval_result_sha256s": [value["payload_sha256"] for value in results],
                "eval_open_count": 1,
                "significance_claim": False,
                "seed_stability_claim": False,
                "evidence_level": "engineering",
                "stop_training": False,
                "student_kd_performed": False,
                "student_kd_authorized": False,
                "acoustic_or_nest_unfreezing_performed": False,
                "acoustic_or_nest_unfreezing_authorized": False,
            }
        ),
    }
    return artifacts, _render_lean_decision_markdown(
        outcome=outcome,
        selected_arm=selected_arm,
        rationale=decision_payload["rationale"],
        eval_opened=True,
    )
