from __future__ import annotations

import hashlib
import io
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.psem_sortformer_adaptation_depth.authority_registry import (
    authority_registry_root,
    require_registered_execution,
)
from experiments.psem_sortformer_adaptation_depth.preflight import (
    _safe_external_output_root,
    canonical_sha256,
)
from experiments.psem_sortformer_adaptation_depth.receipts import (
    build_data_split_receipt,
    paired_source_bootstrap_v1,
    revalidate_material_training_gate_from_bundle,
)
from experiments.psem_training_strategy_gate.sampling import DEV_ROLE, EVAL_ROLE

PRIMARY_THRESHOLD = 0.5
PRIMARY_CONFIRMATION_MS = 500
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 1077301
REQUIRED_PRIMARY_ARMS = ("F0-FROZEN-FLOAT", "H-HEAD", "T2-TOP")
TRAINABLE_ARMS = ("H-HEAD", "T2-TOP", "TA-ALL-TEMPORAL")
LOWER_IS_BETTER = ("contamination", "false_cuts", "missed_replacements")
EVAL_REGISTRY_MARKER = (
    "issue-107-eba82c5a39421b7c8d619cfd971720d8b35b19c8d198605e6e5c0dd09fcd0a97.json"
)


class ProtocolError(RuntimeError):
    pass


def bind_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    if "payload_sha256" in value:
        raise ProtocolError("unbound payload must not already contain a digest")
    return {**value, "payload_sha256": canonical_sha256(value)}


def require_bound(value: Mapping[str, Any], role: str) -> dict[str, Any]:
    payload = {key: item for key, item in value.items() if key != "payload_sha256"}
    if value.get("artifact_role") != role or value.get("payload_sha256") != canonical_sha256(
        payload
    ):
        raise ProtocolError(f"artifact is absent or not content-bound: {role}")
    return payload


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _metric_vector(view: Mapping[str, Any]) -> tuple[float, ...]:
    metrics = view.get("metrics")
    if not isinstance(metrics, Mapping) or any(
        not _finite(metrics.get(metric)) for metric in LOWER_IS_BETTER
    ):
        raise ProtocolError("primary comparison metrics are incomplete")
    return tuple(float(metrics[metric]) for metric in LOWER_IS_BETTER)


def _dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_values = _metric_vector(left)
    right_values = _metric_vector(right)
    return all(a <= b for a, b in zip(left_values, right_values, strict=True)) and any(
        a < b for a, b in zip(left_values, right_values, strict=True)
    )


def validate_dev_result(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = require_bound(value, "psem_sortformer_dev_result")
    arm = payload.get("arm")
    seed = payload.get("seed")
    if (
        arm not in REQUIRED_PRIMARY_ARMS + ("TA-ALL-TEMPORAL",)
        or (arm == "F0-FROZEN-FLOAT" and seed is not None)
        or (arm != "F0-FROZEN-FLOAT" and seed not in {7301, 7302})
        or payload.get("split_role") != DEV_ROLE
        or payload.get("evaluation_roles") != [DEV_ROLE]
        or payload.get("eval_open_count") != 0
        or payload.get("passed") is not True
        or payload.get("slot_mapping_coverage_passed") is not True
        or payload.get("timing_gate_passed") is not True
    ):
        raise ProtocolError("DEV result identity or fail-closed gates are invalid")
    frontier = payload.get("frontier")
    if not isinstance(frontier, list) or len(frontier) != 9:
        raise ProtocolError("DEV result does not contain the complete fixed frontier")
    cells = []
    primary = None
    for row in frontier:
        if not isinstance(row, Mapping):
            raise ProtocolError("DEV frontier row is invalid")
        threshold = row.get("threshold")
        confirmation = row.get("confirmation_ms")
        views = row.get("views")
        if (
            threshold not in {0.35, 0.5, 0.65}
            or confirmation not in {100, 300, 500}
            or not isinstance(views, Mapping)
            or set(views) != {"pooled", "equal_corpus", "corpus_specific"}
            or not isinstance(views.get("corpus_specific"), Mapping)
            or set(views["corpus_specific"]) != {"AMI", "AliMeeting"}
        ):
            raise ProtocolError("DEV frontier coverage or views are invalid")
        _metric_vector(views["pooled"])
        _metric_vector(views["equal_corpus"])
        _metric_vector(views["corpus_specific"]["AMI"])
        _metric_vector(views["corpus_specific"]["AliMeeting"])
        cells.append((float(threshold), int(confirmation)))
        if threshold == PRIMARY_THRESHOLD and confirmation == PRIMARY_CONFIRMATION_MS:
            primary = row
    expected_cells = [
        (threshold, confirmation)
        for threshold in (0.35, 0.5, 0.65)
        for confirmation in (100, 300, 500)
    ]
    if sorted(cells) != sorted(expected_cells) or primary is None:
        raise ProtocolError("DEV fixed frontier is duplicated or missing its primary cell")
    per_source = payload.get("per_source_primary")
    if not isinstance(per_source, list) or not per_source:
        raise ProtocolError("DEV paired-source evidence is absent")
    source_ids = []
    for row in per_source:
        metrics = row.get("metrics") if isinstance(row, Mapping) else None
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("source_id"), str)
            or row.get("corpus") not in {"AMI", "AliMeeting"}
            or not isinstance(metrics, Mapping)
            or set(metrics) != set(LOWER_IS_BETTER)
            or any(not _finite(metrics[metric]) for metric in LOWER_IS_BETTER)
        ):
            raise ProtocolError("DEV per-source primary evidence is invalid")
        source_ids.append(row["source_id"])
    if source_ids != sorted(source_ids) or len(set(source_ids)) != len(source_ids):
        raise ProtocolError("DEV per-source identities are unordered or duplicated")
    if payload.get("dev_evidence_sha256") != canonical_sha256(
        {"frontier": frontier, "per_source_primary": per_source}
    ):
        raise ProtocolError("DEV evidence digest is not reproducible")
    from experiments.psem_sortformer_adaptation_depth.evaluation import (
        evaluate_prediction_set,
    )

    prediction_set = payload.get("prediction_set")
    if not isinstance(prediction_set, Mapping):
        raise ProtocolError("DEV result does not embed its prediction evidence")
    try:
        recomputed = evaluate_prediction_set(prediction_set, historical_replay=True)
    except Exception as exc:
        raise ProtocolError("DEV result prediction evidence is not reproducible") from exc
    if recomputed != dict(value):
        raise ProtocolError("DEV result differs from an exact prediction-set reevaluation")
    split = build_data_split_receipt()
    if source_ids != split["source_ids_by_role"][DEV_ROLE]:
        raise ProtocolError("DEV result does not cover the complete frozen DEV split")
    output_root = prediction_set.get("experiment_output_root")
    registry_root = prediction_set.get("protocol_registry_root")
    root = Path(output_root).resolve() if isinstance(output_root, str) else None
    registry = Path(registry_root).resolve() if isinstance(registry_root, str) else None
    if (
        root is None
        or registry is None
        or not _safe_external_output_root(root)
        or not _safe_external_output_root(registry)
        or root == registry
    ):
        raise ProtocolError("DEV result output root is not an authorized external root")
    return dict(value)


def _run_id(result: Mapping[str, Any]) -> tuple[str, int | None]:
    return str(result["arm"]), result.get("seed")


def _completed_row(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "arm": result["arm"],
        "seed": result.get("seed"),
        "passed": True,
        "evaluation_roles": [DEV_ROLE],
        "dev_evidence_sha256": result["dev_evidence_sha256"],
        "result_payload_sha256": result["payload_sha256"],
    }


def _result_output_root(result: Mapping[str, Any]) -> str:
    prediction = result.get("prediction_set")
    root = prediction.get("experiment_output_root") if isinstance(prediction, Mapping) else None
    if not isinstance(root, str):
        raise ProtocolError("DEV result output root is absent")
    return str(Path(root).resolve())


def _result_protocol_registry_root(result: Mapping[str, Any]) -> str:
    prediction = result.get("prediction_set")
    root = prediction.get("protocol_registry_root") if isinstance(prediction, Mapping) else None
    if not isinstance(root, str):
        raise ProtocolError("DEV result protocol registry root is absent")
    resolved = Path(root).resolve()
    if not _safe_external_output_root(resolved):
        raise ProtocolError("DEV result protocol registry root is not external")
    return str(resolved)


def _eval_registry_marker() -> Path:
    return authority_registry_root() / EVAL_REGISTRY_MARKER


def _initial_staged_state(
    f0_result: Mapping[str, Any], *, enforce_eval_seal: bool
) -> dict[str, Any]:
    f0 = validate_dev_result(f0_result)
    if _run_id(f0) != ("F0-FROZEN-FLOAT", None):
        raise ProtocolError("the staged protocol must begin with F0-FROZEN-FLOAT")
    registry_root = _result_protocol_registry_root(f0)
    if enforce_eval_seal and _eval_registry_marker().exists():
        raise ProtocolError("staged execution cannot restart after EVAL opened")
    return bind_payload(
        {
            "schema_version": 1,
            "artifact_role": "staged_execution_state",
            "eval_open_count": 0,
            "eval_used_for_development": False,
            "experiment_output_root": _result_output_root(f0),
            "protocol_registry_root": registry_root,
            "completed_runs": [_completed_row(f0_result)],
        }
    )


def initial_staged_state(f0_result: Mapping[str, Any]) -> dict[str, Any]:
    return _initial_staged_state(f0_result, enforce_eval_seal=True)


def _append_dev_result(
    state: Mapping[str, Any],
    result: Mapping[str, Any],
    prior_results: Sequence[Mapping[str, Any]],
    *,
    enforce_eval_seal: bool,
) -> dict[str, Any]:
    payload = require_bound(state, "staged_execution_state")
    current = validate_dev_result(result)
    validated = [validate_dev_result(value) for value in prior_results]
    result_by_id = {_run_id(value): value for value in validated}
    completed = payload.get("completed_runs")
    if not isinstance(completed, list) or [
        (row.get("arm"), row.get("seed")) for row in completed
    ] != [_run_id(value) for value in validated]:
        raise ProtocolError("staged state and supplied DEV evidence history differ")
    if any(
        row.get("result_payload_sha256") != value.get("payload_sha256")
        for row, value in zip(completed, prior_results, strict=True)
    ):
        raise ProtocolError("staged state does not bind the supplied DEV result history")
    roots = {_result_output_root(value) for value in [*validated, current]}
    registry_roots = {_result_protocol_registry_root(value) for value in [*validated, current]}
    if roots != {payload.get("experiment_output_root")}:
        raise ProtocolError("staged DEV evidence crosses experiment output roots")
    if registry_roots != {payload.get("protocol_registry_root")}:
        raise ProtocolError("staged DEV evidence crosses protocol registries")
    if enforce_eval_seal and _eval_registry_marker().exists():
        raise ProtocolError("staged execution cannot change after EVAL opened")
    current_id = _run_id(current)
    if current_id in result_by_id:
        raise ProtocolError("a staged run cannot be completed twice")
    expected = _next_allowed_ids(payload)
    if current_id not in expected:
        raise ProtocolError(f"DEV result is out of staged order: {current_id}")
    next_payload = {**payload, "completed_runs": [*completed, _completed_row(result)]}
    merged = [*validated, current]
    ids = {_run_id(value) for value in merged}
    if {("H-HEAD", 7301), ("T2-TOP", 7301)} <= ids:
        next_payload["ta_escalation"] = build_ta_escalation(merged)
    escalation = next_payload.get("ta_escalation")
    primary_complete = escalation is not None and (
        escalation.get("decision") == "closed" or ("TA-ALL-TEMPORAL", 7301) in ids
    )
    if primary_complete:
        next_payload["confirmation_seed_authorization"] = build_confirmation_authorization(merged)
    return bind_payload(next_payload)


def append_dev_result(
    state: Mapping[str, Any],
    result: Mapping[str, Any],
    prior_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return _append_dev_result(state, result, prior_results, enforce_eval_seal=True)


def validate_staged_execution_state(
    state: Mapping[str, Any], results: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if not results:
        raise ProtocolError("staged execution has no DEV evidence")
    validated = [validate_dev_result(value) for value in results]
    replayed = _initial_staged_state(validated[0], enforce_eval_seal=False)
    history = [validated[0]]
    for result in validated[1:]:
        replayed = _append_dev_result(replayed, result, history, enforce_eval_seal=False)
        history.append(result)
    if replayed != dict(state):
        raise ProtocolError("staged state differs from an exact DEV evidence replay")
    return dict(state)


def authorize_conditional_arm_audit(
    arm: str,
    state: Mapping[str, Any] | None,
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    if arm != "TA-ALL-TEMPORAL":
        if state is not None or results:
            raise ProtocolError("non-conditional canaries must not consume staged DEV evidence")
        return None
    if state is None:
        raise ProtocolError("TA implementation audits require the staged DEV authorization")
    validated_state = validate_staged_execution_state(state, results)
    payload = require_bound(validated_state, "staged_execution_state")
    completed = payload.get("completed_runs")
    completed_ids = (
        [(row.get("arm"), row.get("seed")) for row in completed]
        if isinstance(completed, list)
        else []
    )
    escalation = payload.get("ta_escalation")
    root_value = payload.get("protocol_registry_root")
    root = Path(root_value) if isinstance(root_value, str) else None
    if (
        ("H-HEAD", 7301) not in completed_ids
        or ("T2-TOP", 7301) not in completed_ids
        or ("TA-ALL-TEMPORAL", 7301) in completed_ids
        or not isinstance(escalation, Mapping)
        or escalation.get("decision") != "opened"
        or root is None
        or _eval_registry_marker().exists()
    ):
        raise ProtocolError("TA implementation audit precedes its frozen DEV escalation gate")
    evidence = {
        "staged_execution_state_sha256": state["payload_sha256"],
        "dev_result_sha256s": [value["payload_sha256"] for value in results],
        "ta_escalation_dev_evidence_sha256": escalation["dev_evidence_sha256"],
    }
    return bind_payload(
        {
            "schema_version": 1,
            "artifact_role": "conditional_arm_audit_authorization",
            "arm": arm,
            "experiment_output_root": payload["experiment_output_root"],
            "protocol_registry_root": str(root.resolve()),
            **evidence,
        }
    )


def _next_allowed_ids(payload: Mapping[str, Any]) -> set[tuple[str, int | None]]:
    completed = [(row.get("arm"), row.get("seed")) for row in payload.get("completed_runs", [])]
    ids = set(completed)
    if completed == [("F0-FROZEN-FLOAT", None)]:
        return {("H-HEAD", 7301)}
    if ("H-HEAD", 7301) in ids and ("T2-TOP", 7301) not in ids:
        return {("T2-TOP", 7301)}
    escalation = payload.get("ta_escalation")
    if (
        isinstance(escalation, Mapping)
        and escalation.get("decision") == "opened"
        and ("TA-ALL-TEMPORAL", 7301) not in ids
    ):
        return {("TA-ALL-TEMPORAL", 7301)}
    confirmation = payload.get("confirmation_seed_authorization")
    arms = confirmation.get("arms") if isinstance(confirmation, Mapping) else []
    return {(str(arm), 7302) for arm in arms if (str(arm), 7302) not in ids}


def _result_index(
    results: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int | None], dict[str, Any]]:
    validated = [validate_dev_result(value) for value in results]
    index = {_run_id(value): value for value in validated}
    if len(index) != len(validated):
        raise ProtocolError("DEV result set contains duplicate arm/seed identities")
    return index


def _primary(result: Mapping[str, Any]) -> Mapping[str, Any]:
    return next(
        row
        for row in result["frontier"]
        if row["threshold"] == PRIMARY_THRESHOLD
        and row["confirmation_ms"] == PRIMARY_CONFIRMATION_MS
    )


def _per_source(result: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {row["source_id"]: row for row in result["per_source_primary"]}


def _frontier_dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_views = [row["views"]["equal_corpus"] for row in left["frontier"]]
    right_views = [row["views"]["equal_corpus"] for row in right["frontier"]]
    return all(
        any(_dominates(left_view, right_view) for left_view in left_views)
        for right_view in right_views
    )


def _paired_interval(
    left: Mapping[str, Any], right: Mapping[str, Any], metric: str, seed_offset: int
) -> dict[str, Any]:
    left_rows = _per_source(left)
    right_rows = _per_source(right)
    if set(left_rows) != set(right_rows):
        raise ProtocolError("paired DEV result source identities differ")
    deltas = {
        source_id: float(left_rows[source_id]["metrics"][metric])
        - float(right_rows[source_id]["metrics"][metric])
        for source_id in sorted(left_rows)
    }
    receipt = paired_source_bootstrap_v1(
        deltas,
        seed=BOOTSTRAP_SEED + seed_offset,
        resamples=BOOTSTRAP_RESAMPLES,
    )
    return {
        **receipt,
        "unit": "source_or_meeting",
        "algorithm": "paired_source_bootstrap_v1",
        "paired_source_deltas": deltas,
    }


def build_ta_escalation(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    index = _result_index(results)
    head = index.get(("H-HEAD", 7301))
    top = index.get(("T2-TOP", 7301))
    if head is None or top is None:
        raise ProtocolError("TA escalation requires both mandatory primary DEV arms")
    intervals = {
        metric: _paired_interval(top, head, metric, offset)
        for offset, metric in enumerate(LOWER_IS_BETTER)
    }
    favorable = {
        metric: interval
        for metric, interval in intervals.items()
        if metric in {"contamination", "missed_replacements"} and interval["upper"] < 0
    }
    harmful_counts = {}
    for corpus in ("AMI", "AliMeeting"):
        top_rows = {key: row for key, row in _per_source(top).items() if row["corpus"] == corpus}
        head_rows = {key: row for key, row in _per_source(head).items() if row["corpus"] == corpus}
        harmful = 0
        for offset, metric in enumerate(LOWER_IS_BETTER, start=10 if corpus == "AMI" else 20):
            interval = _paired_interval(
                {**top, "per_source_primary": list(top_rows.values())},
                {**head, "per_source_primary": list(head_rows.values())},
                metric,
                offset,
            )
            harmful += int(interval["lower"] > 0)
        harmful_counts[corpus] = harmful
    not_dominated = not _frontier_dominates(head, top)
    opened = (
        not_dominated
        and bool(favorable)
        and all(value < 2 for value in harmful_counts.values())
        and top.get("slot_mapping_coverage_passed") is True
        and top.get("timing_gate_passed") is True
    )
    evidence = {
        "head_result_sha256": head["payload_sha256"],
        "top2_result_sha256": top["payload_sha256"],
        "intervals": intervals,
    }
    return {
        "decision": "opened" if opened else "closed",
        "equal_corpus_not_pareto_dominated": not_dominated,
        "favorable_intervals": favorable,
        "wholly_harmful_metric_counts_by_corpus": harmful_counts,
        "slot_mapping_coverage_passed": top["slot_mapping_coverage_passed"],
        "timing_gate_passed": top["timing_gate_passed"],
        "dev_evidence_sha256": canonical_sha256(evidence),
    }


def build_confirmation_authorization(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    index = _result_index(results)
    primaries = {
        arm: result
        for (arm, seed), result in index.items()
        if seed == 7301 and arm in TRAINABLE_ARMS
    }
    if not {"H-HEAD", "T2-TOP"} <= set(primaries):
        raise ProtocolError("confirmation authorization requires mandatory primary arms")
    non_dominated = {
        arm
        for arm, result in primaries.items()
        if not any(
            other_arm != arm and _frontier_dominates(other_result, result)
            for other_arm, other_result in primaries.items()
        )
    }
    leader = min(non_dominated, key=TRAINABLE_ARMS.index)
    authorized = []
    evidence_by_arm = {}
    for offset, arm in enumerate(TRAINABLE_ARMS):
        result = primaries.get(arm)
        if result is None:
            continue
        interval = _paired_interval(result, primaries[leader], "contamination", 100 + offset)
        allowed = arm in non_dominated or interval["lower"] <= 0 <= interval["upper"]
        if allowed:
            authorized.append(arm)
            evidence_by_arm[arm] = {
                "non_dominated": arm in non_dominated,
                "leader": leader,
                "leader_difference_metric": "contamination",
                "leader_difference_bootstrap_95": interval,
            }
    evidence = {
        "primary_result_sha256_by_arm": {
            arm: result["payload_sha256"] for arm, result in sorted(primaries.items())
        },
        "equal_corpus_frontier_sha256_by_arm": {
            arm: canonical_sha256([row["views"]["equal_corpus"] for row in result["frontier"]])
            for arm, result in sorted(primaries.items())
        },
        "leader": leader,
        "arms": authorized,
    }
    return {
        "arms": authorized,
        "rule": "dev_non_dominated_or_difference_within_paired_bootstrap_uncertainty",
        "dev_evidence_sha256": canonical_sha256(evidence),
        "evidence_by_arm": evidence_by_arm,
    }


def _validate_checkpoint_receipt(
    receipt: Mapping[str, Any],
    arm: str,
    seed: int,
    code_identity_sha256: str,
    *,
    revalidate_material_gate: bool = True,
) -> dict[str, Any]:
    payload = {key: value for key, value in receipt.items() if key != "payload_sha256"}
    path_value = payload.get("checkpoint_path")
    path = Path(path_value) if isinstance(path_value, str) else None
    material_gate = payload.get("material_training_authorization")
    training_summary = payload.get("training_summary")
    training_result = payload.get("training_result")
    runtime_identity = payload.get("runtime_identity")
    parameter_policy = payload.get("parameter_policy")
    validation_bundle = (
        material_gate.get("validation_bundle") if isinstance(material_gate, Mapping) else None
    )
    authorized_root_value = payload.get("authorized_output_root")
    authorized_root = (
        Path(authorized_root_value).resolve() if isinstance(authorized_root_value, str) else None
    )
    checkpoint_bytes = path.read_bytes() if path is not None and path.is_file() else None
    if (
        receipt.get("artifact_role") != "psem_sortformer_checkpoint"
        or receipt.get("payload_sha256") != canonical_sha256(payload)
        or receipt.get("arm") != arm
        or receipt.get("seed") != seed
        or path is None
        or not path.is_absolute()
        or not path.is_file()
        or authorized_root is None
        or not path.resolve().is_relative_to(authorized_root)
        or checkpoint_bytes is None
        or receipt.get("checkpoint_sha256") != hashlib.sha256(checkpoint_bytes).hexdigest()
        or receipt.get("checkpoint_size_bytes") != len(checkpoint_bytes)
        or not isinstance(material_gate, Mapping)
        or material_gate.get("artifact_role") != "material_training_authorization"
        or material_gate.get("payload_sha256") != receipt.get("material_gate_sha256")
        or material_gate.get("arm") != arm
        or material_gate.get("seed") != seed
        or material_gate.get("candidate_code_identity_sha256") != code_identity_sha256
        or receipt.get("candidate_code_identity_sha256") != code_identity_sha256
        or not isinstance(validation_bundle, Mapping)
        or not isinstance(runtime_identity, Mapping)
        or runtime_identity != validation_bundle.get("runtime_identity")
        or receipt.get("runtime_identity_sha256") != canonical_sha256(runtime_identity)
        or not isinstance(parameter_policy, Mapping)
        or parameter_policy.get("arm") != arm
        or receipt.get("parameter_policy_sha256") != canonical_sha256(parameter_policy)
        or receipt.get("split_roles") != ["PSEM-STRATEGY-TRAIN", DEV_ROLE]
        or receipt.get("eval_source_count") != 0
        or receipt.get("dev_source_ids_sha256") != material_gate.get("dev_source_ids_sha256")
        or receipt.get("authorized_output_root") != material_gate.get("authorized_output_root")
        or not isinstance(training_summary, Mapping)
        or not isinstance(receipt.get("selected_metrics"), Mapping)
        or training_summary.get("selected_epoch") != receipt.get("selected_epoch")
        or training_summary.get("selected_metrics") != receipt.get("selected_metrics")
        or type(training_summary.get("total_parameters")) is not int
        or training_summary["total_parameters"] <= 0
        or type(training_summary.get("trainable_parameters")) is not int
        or not 0 < training_summary["trainable_parameters"] <= training_summary["total_parameters"]
        or not _finite(training_summary.get("training_wall_clock_seconds"))
        or training_summary["training_wall_clock_seconds"] <= 0
        or type(training_summary.get("peak_training_memory_bytes")) is not int
        or training_summary["peak_training_memory_bytes"] <= 0
        or training_summary.get("native_diarization_contract_passed") is not True
        or not isinstance(training_summary.get("native_diarization_contract_evidence_sha256"), str)
        or len(training_summary["native_diarization_contract_evidence_sha256"]) != 64
        or training_summary["native_diarization_contract_evidence_sha256"]
        != canonical_sha256(
            {
                "overfit_receipt_sha256": material_gate.get("overfit_receipt_sha256"),
                "gradient_receipt_sha256": material_gate.get("gradient_receipt_sha256"),
                "timing_receipt_sha256": material_gate.get("timing_receipt_sha256"),
            }
        )
    ):
        raise ProtocolError(f"checkpoint receipt is not reproducible: {arm}:{seed}")
    require_registered_execution("checkpoint-receipt", receipt)
    training_payload = (
        {key: value for key, value in training_result.items() if key != "payload_sha256"}
        if isinstance(training_result, Mapping)
        else None
    )
    selected_checkpoint = (
        training_result.get("selected_checkpoint") if isinstance(training_result, Mapping) else None
    )
    if (
        training_payload is None
        or training_result.get("artifact_role") != "psem_sortformer_training_result"
        or training_result.get("payload_sha256") != canonical_sha256(training_payload)
        or receipt.get("training_result_sha256") != training_result.get("payload_sha256")
        or training_result.get("arm") != arm
        or training_result.get("seed") != seed
        or training_result.get("authorization_sha256") != material_gate.get("payload_sha256")
        or training_result.get("candidate_code_identity_sha256") != code_identity_sha256
        or training_result.get("runtime_identity") != runtime_identity
        or training_result.get("runtime_identity_sha256") != receipt.get("runtime_identity_sha256")
        or training_result.get("parameter_policy") != parameter_policy
        or training_result.get("parameter_policy_sha256") != receipt.get("parameter_policy_sha256")
        or training_result.get("split_roles") != receipt.get("split_roles")
        or training_result.get("eval_source_count") != 0
        or training_result.get("dev_source_ids_sha256") != receipt.get("dev_source_ids_sha256")
        or training_result.get("checkpoint_path") != str(path.resolve())
        or training_result.get("checkpoint_sha256") != receipt.get("checkpoint_sha256")
        or training_result.get("checkpoint_size_bytes") != receipt.get("checkpoint_size_bytes")
        or training_result.get("training_summary") != training_summary
        or not isinstance(selected_checkpoint, Mapping)
        or selected_checkpoint.get("epoch") != receipt.get("selected_epoch")
        or selected_checkpoint.get("dev_total_loss")
        != receipt.get("selected_metrics", {}).get("dev_total_loss")
        or selected_checkpoint.get("dev_replacement_average_precision")
        != receipt.get("selected_metrics", {}).get("dev_replacement_average_precision")
        or selected_checkpoint.get("selection_roles") != [DEV_ROLE]
    ):
        raise ProtocolError(f"checkpoint training provenance is invalid: {arm}:{seed}")
    require_registered_execution("training-result", training_result)
    try:
        if revalidate_material_gate:
            revalidate_material_training_gate_from_bundle(material_gate)
        import torch

        checkpoint_payload = torch.load(
            io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True
        )
    except Exception as exc:
        raise ProtocolError(
            f"checkpoint material gate is not currently valid: {arm}:{seed}"
        ) from exc
    state_dict = (
        checkpoint_payload.get("model_state_dict")
        if isinstance(checkpoint_payload, Mapping)
        else None
    )
    inventory = material_gate.get("validation_bundle", {}).get("parameter_inventory")
    inventory_rows = inventory.get("parameters") if isinstance(inventory, Mapping) else None
    model_graph = (
        material_gate.get("validation_bundle", {}).get("runtime_identity", {}).get("model_graph")
    )
    state_dict_rows = (
        [
            {"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in state_dict.items()
        ]
        if isinstance(state_dict, Mapping)
        and all(
            hasattr(value, "shape") and hasattr(value, "dtype") for value in state_dict.values()
        )
        else None
    )
    if (
        not isinstance(checkpoint_payload, Mapping)
        or set(checkpoint_payload) != {"schema_version", "arm", "seed", "model_state_dict"}
        or checkpoint_payload.get("schema_version") != 1
        or checkpoint_payload.get("arm") != arm
        or checkpoint_payload.get("seed") != seed
        or not isinstance(state_dict, Mapping)
        or not state_dict
        or not isinstance(inventory_rows, list)
        or not isinstance(model_graph, Mapping)
        or not isinstance(state_dict_rows, list)
        or model_graph.get("executable_state_entry_count") != len(state_dict_rows)
        or model_graph.get("state_dict_schema_sha256") != canonical_sha256(state_dict_rows)
        or any(
            not isinstance(row, Mapping)
            or row.get("name") not in state_dict
            or not hasattr(state_dict[row["name"]], "shape")
            or not hasattr(state_dict[row["name"]], "dtype")
            or list(state_dict[row["name"]].shape) != row.get("shape")
            or str(state_dict[row["name"]].dtype) != row.get("dtype")
            for row in inventory_rows
        )
    ):
        raise ProtocolError(f"checkpoint payload differs from the authorized graph: {arm}:{seed}")
    return dict(receipt)


def _freeze_candidate_set(
    state: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    checkpoint_receipts: Mapping[str, Mapping[str, Any]],
    prediction_sets: Mapping[str, Mapping[str, Any]],
    code_identity: Mapping[str, Any],
    *,
    enforce_eval_seal: bool,
) -> dict[str, Any]:
    from experiments.psem_sortformer_adaptation_depth.evaluation import (
        validate_prediction_set,
    )

    payload = require_bound(state, "staged_execution_state")
    validated = [validate_dev_result(value) for value in results]
    completed_ids = [(row["arm"], row.get("seed")) for row in payload["completed_runs"]]
    if completed_ids != [_run_id(value) for value in validated]:
        raise ProtocolError("candidate freeze history differs from staged state")
    if not set(REQUIRED_PRIMARY_ARMS) <= {arm for arm, _ in completed_ids}:
        raise ProtocolError("candidate freeze precedes mandatory primary arms")
    escalation = payload.get("ta_escalation")
    if not isinstance(escalation, Mapping) or escalation.get("decision") not in {
        "opened",
        "closed",
    }:
        raise ProtocolError("candidate freeze precedes the TA decision")
    if escalation["decision"] == "opened" and ("TA-ALL-TEMPORAL", 7301) not in completed_ids:
        raise ProtocolError("candidate freeze precedes the opened TA primary run")
    confirmation = payload.get("confirmation_seed_authorization")
    required_confirmations = (
        {(arm, 7302) for arm in confirmation.get("arms", [])}
        if isinstance(confirmation, Mapping)
        else set()
    )
    if not required_confirmations <= set(completed_ids):
        raise ProtocolError("candidate freeze precedes authorized confirmation seeds")
    code = require_bound(code_identity, "psem_sortformer_candidate_code_identity")
    if (
        not isinstance(code.get("git_head"), str)
        or len(code["git_head"]) != 40
        or not isinstance(code.get("artifact_sha256s"), Mapping)
        or code.get("worktree_clean") is not True
    ):
        raise ProtocolError("candidate code identity is incomplete")
    roots = {_result_output_root(value) for value in validated}
    if roots != {payload.get("experiment_output_root")}:
        raise ProtocolError("candidate freeze crosses external experiment roots")
    experiment_output_root = next(iter(roots))
    registry_roots = {_result_protocol_registry_root(value) for value in validated}
    if registry_roots != {payload.get("protocol_registry_root")}:
        raise ProtocolError("candidate freeze crosses protocol registries")
    protocol_registry_root = next(iter(registry_roots))
    if enforce_eval_seal and _eval_registry_marker().exists():
        raise ProtocolError("candidate set cannot be refrozen after EVAL opened")
    selected = []
    for result in validated:
        arm = result["arm"]
        seed = result.get("seed")
        key = f"{arm}:{seed if seed is not None else 'f0'}"
        prediction = prediction_sets.get(key)
        prediction_payload = (
            validate_prediction_set(prediction) if isinstance(prediction, Mapping) else None
        )
        if (
            prediction_payload is None
            or prediction_payload.get("arm") != arm
            or prediction_payload.get("seed") != seed
            or prediction_payload.get("split_role") != DEV_ROLE
            or prediction_payload.get("eval_authorization_sha256") is not None
            or prediction_payload.get("candidate_git_head") != code["git_head"]
            or prediction_payload.get("candidate_code_identity_sha256")
            != code_identity.get("payload_sha256")
            or result.get("prediction_set_sha256") != prediction.get("payload_sha256")
        ):
            raise ProtocolError(f"DEV prediction/result lineage is absent: {key}")
        if arm == "F0-FROZEN-FLOAT":
            if prediction_payload.get("trained_checkpoint_sha256") is not None:
                raise ProtocolError("F0 DEV evidence unexpectedly binds a trained checkpoint")
            selected.append(
                {
                    "arm": arm,
                    "seed": None,
                    "dev_result_sha256": result["payload_sha256"],
                    "dev_prediction_set_sha256": prediction["payload_sha256"],
                    "checkpoint_sha256": None,
                    "checkpoint_receipt_sha256": None,
                    "training_summary": {
                        "total_parameters": prediction_payload["total_parameters"],
                        "trainable_parameters": 0,
                        "peak_training_memory_bytes": 0,
                        "training_wall_clock_seconds": 0.0,
                        "not_trained": True,
                        "native_diarization_contract_passed": True,
                        "native_diarization_contract_evidence_sha256": prediction["payload_sha256"],
                    },
                }
            )
            continue
        checkpoint_key = f"{arm}:{seed}"
        checkpoint = checkpoint_receipts.get(checkpoint_key)
        if not isinstance(checkpoint, Mapping):
            raise ProtocolError(f"candidate checkpoint receipt is absent: {checkpoint_key}")
        _validate_checkpoint_receipt(
            checkpoint,
            arm,
            int(seed),
            str(code_identity["payload_sha256"]),
            revalidate_material_gate=enforce_eval_seal,
        )
        if (
            checkpoint.get("authorized_output_root") != experiment_output_root
            or checkpoint["material_training_authorization"].get("authorized_output_root")
            != experiment_output_root
            or checkpoint["material_training_authorization"].get(
                "authorized_protocol_registry_root"
            )
            != protocol_registry_root
            or prediction_payload.get("trained_checkpoint_sha256")
            != checkpoint.get("checkpoint_sha256")
            or prediction_payload.get("trained_checkpoint_receipt_sha256")
            != checkpoint.get("payload_sha256")
        ):
            raise ProtocolError(f"DEV prediction used another checkpoint: {checkpoint_key}")
        selected.append(
            {
                "arm": arm,
                "seed": seed,
                "dev_result_sha256": result["payload_sha256"],
                "dev_prediction_set_sha256": prediction["payload_sha256"],
                "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "checkpoint_receipt_sha256": checkpoint["payload_sha256"],
                "training_result_sha256": checkpoint["training_result_sha256"],
                "training_summary": checkpoint["training_summary"],
            }
        )
    frozen_payload = {
        "schema_version": 1,
        "artifact_role": "psem_sortformer_candidate_freeze",
        "candidate_set": selected,
        "staged_execution_state_sha256": state["payload_sha256"],
        "thresholds": [0.35, 0.5, 0.65],
        "confirmation_ms": [100, 300, 500],
        "decision_rule": "issue-107-shallowest-nondominated-v1",
        "report_schema": "issue-107-required-artifacts-v1",
        "candidate_code_identity_sha256": code_identity["payload_sha256"],
        "candidate_git_head": code["git_head"],
        "candidate_artifact_sha256s": code["artifact_sha256s"],
        "experiment_output_root": experiment_output_root,
        "protocol_registry_root": protocol_registry_root,
        "eval_open_count": 0,
        "eval_used_for_development": False,
        "staged_execution_state": dict(state),
        "dev_results": [dict(value) for value in results],
        "checkpoint_receipts": {
            key: dict(value) for key, value in sorted(checkpoint_receipts.items())
        },
        "prediction_sets": {key: dict(value) for key, value in sorted(prediction_sets.items())},
        "candidate_code_identity": dict(code_identity),
    }
    return bind_payload(frozen_payload)


def freeze_candidate_set(
    state: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    checkpoint_receipts: Mapping[str, Mapping[str, Any]],
    prediction_sets: Mapping[str, Mapping[str, Any]],
    code_identity: Mapping[str, Any],
) -> dict[str, Any]:
    return _freeze_candidate_set(
        state,
        results,
        checkpoint_receipts,
        prediction_sets,
        code_identity,
        enforce_eval_seal=True,
    )


def validate_candidate_freeze(candidate_freeze: Mapping[str, Any]) -> dict[str, Any]:
    frozen = require_bound(candidate_freeze, "psem_sortformer_candidate_freeze")
    state = frozen.get("staged_execution_state")
    results = frozen.get("dev_results")
    checkpoints = frozen.get("checkpoint_receipts")
    predictions = frozen.get("prediction_sets")
    code_identity = frozen.get("candidate_code_identity")
    if (
        not isinstance(state, Mapping)
        or not isinstance(results, list)
        or not isinstance(checkpoints, Mapping)
        or not isinstance(predictions, Mapping)
        or not isinstance(code_identity, Mapping)
    ):
        raise ProtocolError("candidate freeze lacks its reproducible evidence bundle")
    validate_staged_execution_state(state, results)
    recomputed = _freeze_candidate_set(
        state,
        results,
        checkpoints,
        predictions,
        code_identity,
        enforce_eval_seal=False,
    )
    if recomputed != dict(candidate_freeze):
        raise ProtocolError("candidate freeze differs from an exact evidence replay")
    return dict(candidate_freeze)


def open_eval_once(
    candidate_freeze: Mapping[str, Any], experiment_output_root: str
) -> dict[str, Any]:
    from experiments.psem_sortformer_adaptation_depth.execution import (
        validate_current_candidate_identity,
    )

    validate_candidate_freeze(candidate_freeze)
    frozen = require_bound(candidate_freeze, "psem_sortformer_candidate_freeze")
    root = Path(experiment_output_root).resolve()
    if (
        frozen.get("eval_open_count") != 0
        or frozen.get("eval_used_for_development") is not False
        or not isinstance(experiment_output_root, str)
        or not experiment_output_root
        or not _safe_external_output_root(root)
        or str(root) != frozen.get("experiment_output_root")
    ):
        raise ProtocolError("candidate freeze does not preserve the sealed EVAL state")
    validate_current_candidate_identity(frozen["candidate_code_identity"])
    receipt = bind_payload(
        {
            "schema_version": 1,
            "artifact_role": "psem_sortformer_eval_open_authorization",
            "candidate_freeze_sha256": candidate_freeze["payload_sha256"],
            "evaluation_roles": [EVAL_ROLE],
            "eval_open_count": 1,
            "eval_used_for_development": False,
            "candidate_set": frozen["candidate_set"],
            "candidate_code_identity_sha256": frozen["candidate_code_identity_sha256"],
            "candidate_git_head": frozen["candidate_git_head"],
            "candidate_artifact_sha256s": frozen["candidate_artifact_sha256s"],
            "experiment_output_root": str(root),
            "protocol_registry_root": frozen["protocol_registry_root"],
            "candidate_freeze": dict(candidate_freeze),
        }
    )
    marker = _eval_registry_marker()
    marker.parent.mkdir(parents=True, exist_ok=True)
    try:
        with marker.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(receipt, ensure_ascii=False, sort_keys=True) + "\n")
    except FileExistsError as exc:
        raise ProtocolError("EVAL has already been opened for the pinned authority") from exc
    return receipt


def validate_eval_authorization(authorization: Mapping[str, Any]) -> dict[str, Any]:
    payload = require_bound(authorization, "psem_sortformer_eval_open_authorization")
    candidate_freeze = payload.get("candidate_freeze")
    root_value = payload.get("experiment_output_root")
    root = Path(root_value).resolve() if isinstance(root_value, str) else None
    registry_value = payload.get("protocol_registry_root")
    registry_root = Path(registry_value).resolve() if isinstance(registry_value, str) else None
    if (
        not isinstance(candidate_freeze, Mapping)
        or root is None
        or registry_root is None
        or not _safe_external_output_root(registry_root)
        or not _safe_external_output_root(root)
        or payload.get("candidate_freeze_sha256") != candidate_freeze.get("payload_sha256")
        or payload.get("candidate_set") != candidate_freeze.get("candidate_set")
        or payload.get("candidate_code_identity_sha256")
        != candidate_freeze.get("candidate_code_identity_sha256")
        or payload.get("candidate_git_head") != candidate_freeze.get("candidate_git_head")
        or payload.get("candidate_artifact_sha256s")
        != candidate_freeze.get("candidate_artifact_sha256s")
        or str(root) != candidate_freeze.get("experiment_output_root")
        or str(registry_root) != candidate_freeze.get("protocol_registry_root")
        or payload.get("evaluation_roles") != [EVAL_ROLE]
        or payload.get("eval_open_count") != 1
        or payload.get("eval_used_for_development") is not False
    ):
        raise ProtocolError("EVAL authorization identity is invalid")
    validate_candidate_freeze(candidate_freeze)
    marker = _eval_registry_marker()
    if not marker.is_file():
        raise ProtocolError("canonical EVAL open marker is absent")
    try:
        persisted = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("canonical EVAL open marker is unreadable") from exc
    if persisted != dict(authorization):
        raise ProtocolError("EVAL authorization differs from the canonical single-open marker")
    return dict(authorization)
