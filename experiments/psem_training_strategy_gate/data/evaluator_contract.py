from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from experiments.psem_training_strategy_gate.data.dataset_context import (
    resolve_dataset_context,
)
from experiments.psem_training_strategy_gate.data.provenance import (
    canonical_sha256,
    sha256_file,
)
from experiments.psem_training_strategy_gate.data.split_assignment import (
    EVAL_ROLE,
    validate_checked_split_package,
)
from experiments.psem_training_strategy_gate.data.split_feasibility import (
    TOPOLOGY_REQUIREMENTS,
)

CORPORA = ("AMI", "AliMeeting")
REQUIRED_VIEW_IDS = (
    "pooled_exposure_weighted",
    "equal_corpus_macro",
    "corpus_specific",
    "topology_by_corpus",
    "corpus_stratified_meeting_bootstrap",
)
ALLOWED_OUTCOMES = (
    "winner",
    "close_runner_up",
    "corpus_dependent",
    "inconclusive",
)


class EvaluatorContractError(RuntimeError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluatorContractError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise EvaluatorContractError(f"JSON artifact must be an object: {path}")
    return value


def build_evaluator_contract(
    data_dir: Path,
    registry_path: Path,
    source_registry_path: Path,
) -> dict[str, Any]:
    context = resolve_dataset_context(data_dir)
    if not context.is_v2:
        raise EvaluatorContractError("the corpus-balanced evaluator requires the v2 dataset")
    split, feasibility = validate_checked_split_package(
        data_dir,
        registry_path,
        source_registry_path,
    )
    if split.get("hard_gate_status") != "pass" or feasibility.get("hard_gate_status") != "pass":
        raise EvaluatorContractError("the accepted v2 split does not pass its hard gates")
    eval_rows = [row for row in split["assignments"]["sources"] if row.get("role") == EVAL_ROLE]
    eval_by_corpus = {
        corpus: sorted(row["source_id"] for row in eval_rows if row.get("corpus") == corpus)
        for corpus in CORPORA
    }
    if any(not source_ids for source_ids in eval_by_corpus.values()) or sum(
        len(source_ids) for source_ids in eval_by_corpus.values()
    ) != len(eval_rows):
        raise EvaluatorContractError("EVAL source coverage is not exactly two-corpus")
    role_summary = split["role_summaries"][EVAL_ROLE]
    topology_rows = [
        json.loads(line)
        for line in (data_dir / "topology_manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    topology_by_source = {row.get("source_id"): row for row in topology_rows}
    if set(topology_by_source) != {row["source_id"] for row in split["assignments"]["sources"]}:
        raise EvaluatorContractError("topology and split source coverage mismatch")
    topology_exposure = {
        corpus: {
            topology: sum(
                topology_by_source[source_id]["primary_topology_counts"][topology]
                for source_id in source_ids
            )
            for topology in TOPOLOGY_REQUIREMENTS
        }
        for corpus, source_ids in eval_by_corpus.items()
    }
    if any(count <= 0 for counts in topology_exposure.values() for count in counts.values()):
        raise EvaluatorContractError("EVAL lacks a required topology-by-corpus slice")
    return {
        "schema_version": 1,
        "artifact_role": "psem_shared_threshold_evaluator_contract",
        "authority_ref": context.authority_ref,
        "authority_pin": context.authority_pin,
        "contract_version": context.label_contract.contract_version,
        "dataset_freeze_id": context.freeze_id,
        "split_binding": {
            "split_manifest_sha256": sha256_file(data_dir / "split_manifest.json"),
            "split_manifest_canonical_sha256": canonical_sha256(split),
            "split_assignment_sha256": split["assignment_sha256"],
            "split_feasibility_sha256": sha256_file(data_dir / "split_feasibility.json"),
            "eval_source_ids_by_corpus": eval_by_corpus,
            "eval_source_count_by_corpus": {
                corpus: len(source_ids) for corpus, source_ids in eval_by_corpus.items()
            },
            "eval_scored_samples_by_corpus": role_summary["corpus_scored_samples"],
            "topology_counts_by_corpus": topology_exposure,
        },
        "threshold_policy": {
            "score_threshold_domain": {"minimum": 0.0, "maximum": 1.0},
            "strictly_increasing_unique_thresholds_required": True,
            "same_threshold_vector_required_for_every_output": True,
            "per_corpus_thresholds_allowed": False,
            "threshold_selection_from_eval_allowed": False,
        },
        "required_outputs": {
            "pooled_exposure_weighted": {
                "aggregation": "weighted arithmetic mean of corpus-specific metric values using bound EVAL scored samples"
            },
            "equal_corpus_macro": {
                "aggregation": "unweighted arithmetic mean of AMI and AliMeeting metric values"
            },
            "corpus_specific": {"corpora": list(CORPORA)},
            "topology_by_corpus": {
                "corpora": list(CORPORA),
                "topologies": sorted(TOPOLOGY_REQUIREMENTS),
            },
            "corpus_stratified_meeting_bootstrap": {
                "resampling_unit": "whole_meeting",
                "strata": list(CORPORA),
                "with_replacement_within_each_corpus": True,
                "same_resample_indices_required_across_arms_and_thresholds": True,
                "resample_plan_sha256_required": True,
                "intervals_required_for_all_frontier_outputs": True,
            },
        },
        "result_schema": {
            "score_thresholds": "at least two strictly increasing finite numbers in [0,1]",
            "frontier_row": {
                "score_threshold": "one value from score_thresholds",
                "metrics": "non-empty object of finite numeric metrics",
            },
            "bootstrap_interval_metric": {
                "lower": "finite number",
                "upper": "finite number not below lower",
            },
        },
        "decision_policy": {
            "allowed_outcomes": list(ALLOWED_OUTCOMES),
            "pooled_frontier_alone_may_declare_universal_winner": False,
            "universal_winner_requires_all_required_views": True,
            "material_equal_corpus_or_corpus_specific_reversal_outcomes": [
                "corpus_dependent",
                "inconclusive",
            ],
        },
        "model_policy": {
            "model_predictions_consulted_for_contract": False,
            "model_scores_consulted_for_contract": False,
            "official_model_results_inspected_for_contract": False,
            "official_model_training_performed_for_contract": False,
        },
    }


def _thresholds(value: Any) -> list[float | int]:
    if not isinstance(value, list) or len(value) < 2:
        raise EvaluatorContractError("score_thresholds must contain at least two values")
    if any(
        not isinstance(item, (int, float))
        or isinstance(item, bool)
        or not math.isfinite(item)
        or item < 0
        or item > 1
        for item in value
    ):
        raise EvaluatorContractError("score_thresholds are outside the finite [0,1] domain")
    if any(left >= right for left, right in zip(value, value[1:], strict=False)):
        raise EvaluatorContractError("score_thresholds are not strictly increasing and unique")
    return value


def _require_exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise EvaluatorContractError(f"{label} field inventory is invalid")
    return value


def _finite_metrics(value: Any, label: str) -> dict[str, float | int]:
    if not isinstance(value, dict) or not value:
        raise EvaluatorContractError(f"{label} metrics are missing")
    if any(not isinstance(name, str) or not name for name in value):
        raise EvaluatorContractError(f"{label} metric names are invalid")
    if any(
        not isinstance(item, (int, float)) or isinstance(item, bool) or not math.isfinite(item)
        for item in value.values()
    ):
        raise EvaluatorContractError(f"{label} metrics are not finite numeric values")
    return value


def _validate_frontier_rows(
    rows: Any,
    thresholds: list[float | int],
    label: str,
) -> set[str]:
    if not isinstance(rows, list) or len(rows) != len(thresholds):
        raise EvaluatorContractError(f"{label} does not cover every shared threshold")
    metric_names: set[str] | None = None
    for threshold, row in zip(thresholds, rows, strict=True):
        row = _require_exact_keys(row, {"score_threshold", "metrics"}, f"{label} row")
        if row["score_threshold"] != threshold:
            raise EvaluatorContractError(f"{label} threshold vector differs from the shared vector")
        names = set(_finite_metrics(row.get("metrics"), label))
        if metric_names is None:
            metric_names = names
        elif names != metric_names:
            raise EvaluatorContractError(f"{label} metric inventory changes across thresholds")
    return metric_names or set()


def _validate_frontier_tree(
    frontiers: dict[str, Any],
    contract: dict[str, Any],
    thresholds: list[float | int],
) -> dict[str, set[str]]:
    _require_exact_keys(
        frontiers,
        {
            "pooled_exposure_weighted",
            "equal_corpus_macro",
            "corpus_specific",
            "topology_by_corpus",
        },
        "frontiers",
    )
    inventories = {}
    inventories["pooled_exposure_weighted"] = _validate_frontier_rows(
        frontiers.get("pooled_exposure_weighted"), thresholds, "pooled_exposure_weighted"
    )
    inventories["equal_corpus_macro"] = _validate_frontier_rows(
        frontiers.get("equal_corpus_macro"), thresholds, "equal_corpus_macro"
    )
    corpus_specific = frontiers.get("corpus_specific")
    topology_by_corpus = frontiers.get("topology_by_corpus")
    if not isinstance(corpus_specific, dict) or set(corpus_specific) != set(CORPORA):
        raise EvaluatorContractError("corpus_specific frontier coverage is invalid")
    if not isinstance(topology_by_corpus, dict) or set(topology_by_corpus) != set(CORPORA):
        raise EvaluatorContractError("topology_by_corpus frontier coverage is invalid")
    topologies = set(contract["required_outputs"]["topology_by_corpus"]["topologies"])
    for corpus in CORPORA:
        label = f"corpus_specific.{corpus}"
        inventories[label] = _validate_frontier_rows(corpus_specific[corpus], thresholds, label)
        if (
            not isinstance(topology_by_corpus[corpus], dict)
            or set(topology_by_corpus[corpus]) != topologies
        ):
            raise EvaluatorContractError(f"topology_by_corpus.{corpus} coverage is invalid")
        for topology in sorted(topologies):
            label = f"topology_by_corpus.{corpus}.{topology}"
            inventories[label] = _validate_frontier_rows(
                topology_by_corpus[corpus][topology],
                thresholds,
                label,
            )
    required_metric_names = inventories["pooled_exposure_weighted"]
    if any(names != required_metric_names for names in inventories.values()):
        raise EvaluatorContractError("frontier metric inventory differs across required views")
    return inventories


def _validate_cross_view_aggregation(
    frontiers: dict[str, Any],
    contract: dict[str, Any],
    thresholds: list[float | int],
    metric_names: set[str],
) -> None:
    corpus_rows = frontiers["corpus_specific"]
    weights = contract["split_binding"]["eval_scored_samples_by_corpus"]
    total_weight = sum(weights.values())
    if (
        set(weights) != set(CORPORA)
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in weights.values()
        )
        or total_weight <= 0
    ):
        raise EvaluatorContractError("contract corpus exposure weights are invalid")
    for index, _ in enumerate(thresholds):
        pooled = frontiers["pooled_exposure_weighted"][index]["metrics"]
        macro = frontiers["equal_corpus_macro"][index]["metrics"]
        ami = corpus_rows["AMI"][index]["metrics"]
        alimeeting = corpus_rows["AliMeeting"][index]["metrics"]
        for metric in metric_names:
            expected_macro = (ami[metric] + alimeeting[metric]) / 2
            expected_pooled = (
                ami[metric] * weights["AMI"]
                + alimeeting[metric] * weights["AliMeeting"]
            ) / total_weight
            if not math.isclose(
                macro[metric], expected_macro, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise EvaluatorContractError(
                    "equal_corpus_macro is not the arithmetic mean of corpus frontiers"
                )
            if not math.isclose(
                pooled[metric], expected_pooled, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise EvaluatorContractError(
                    "pooled_exposure_weighted does not use bound corpus exposure"
                )


def _validate_interval_rows(
    rows: Any,
    thresholds: list[float | int],
    label: str,
    metric_names: set[str],
) -> None:
    if not isinstance(rows, list) or len(rows) != len(thresholds):
        raise EvaluatorContractError(f"{label} intervals do not cover every threshold")
    for threshold, row in zip(thresholds, rows, strict=True):
        row = _require_exact_keys(
            row, {"score_threshold", "metrics"}, f"{label} interval row"
        )
        if row["score_threshold"] != threshold:
            raise EvaluatorContractError(
                f"{label} interval thresholds differ from the shared vector"
            )
        metrics = row.get("metrics")
        if not isinstance(metrics, dict) or not metrics:
            raise EvaluatorContractError(f"{label} interval metrics are missing")
        if set(metrics) != metric_names:
            raise EvaluatorContractError(
                f"{label} interval metric inventory differs from its frontier"
            )
        for interval in metrics.values():
            if not isinstance(interval, dict) or set(interval) != {"lower", "upper"}:
                raise EvaluatorContractError(f"{label} interval shape is invalid")
            lower = interval["lower"]
            upper = interval["upper"]
            if (
                not isinstance(lower, (int, float))
                or isinstance(lower, bool)
                or not isinstance(upper, (int, float))
                or isinstance(upper, bool)
                or not math.isfinite(lower)
                or not math.isfinite(upper)
                or lower > upper
            ):
                raise EvaluatorContractError(f"{label} interval bounds are invalid")


def _validate_interval_tree(
    intervals: Any,
    contract: dict[str, Any],
    thresholds: list[float | int],
    metric_inventories: dict[str, set[str]],
) -> None:
    if not isinstance(intervals, dict):
        raise EvaluatorContractError("bootstrap intervals are missing")
    _require_exact_keys(
        intervals,
        {
            "pooled_exposure_weighted",
            "equal_corpus_macro",
            "corpus_specific",
            "topology_by_corpus",
        },
        "bootstrap intervals",
    )
    _validate_interval_rows(
        intervals.get("pooled_exposure_weighted"),
        thresholds,
        "pooled_exposure_weighted",
        metric_inventories["pooled_exposure_weighted"],
    )
    _validate_interval_rows(
        intervals.get("equal_corpus_macro"),
        thresholds,
        "equal_corpus_macro",
        metric_inventories["equal_corpus_macro"],
    )
    corpus_specific = intervals.get("corpus_specific")
    topology_by_corpus = intervals.get("topology_by_corpus")
    if not isinstance(corpus_specific, dict) or set(corpus_specific) != set(CORPORA):
        raise EvaluatorContractError("bootstrap corpus_specific coverage is invalid")
    if not isinstance(topology_by_corpus, dict) or set(topology_by_corpus) != set(CORPORA):
        raise EvaluatorContractError("bootstrap topology_by_corpus coverage is invalid")
    topologies = set(contract["required_outputs"]["topology_by_corpus"]["topologies"])
    for corpus in CORPORA:
        label = f"corpus_specific.{corpus}"
        _validate_interval_rows(
            corpus_specific[corpus], thresholds, label, metric_inventories[label]
        )
        if (
            not isinstance(topology_by_corpus[corpus], dict)
            or set(topology_by_corpus[corpus]) != topologies
        ):
            raise EvaluatorContractError(
                f"bootstrap topology_by_corpus.{corpus} coverage is invalid"
            )
        for topology in sorted(topologies):
            label = f"topology_by_corpus.{corpus}.{topology}"
            _validate_interval_rows(
                topology_by_corpus[corpus][topology],
                thresholds,
                label,
                metric_inventories[label],
            )


def _validate_resample_plan(
    bootstrap: dict[str, Any],
    expected_sources: dict[str, list[str]],
) -> None:
    plan = bootstrap["resample_indices_by_corpus"]
    count = bootstrap["resample_count"]
    if not isinstance(plan, dict) or set(plan) != set(CORPORA):
        raise EvaluatorContractError("bootstrap resample plan corpus coverage is invalid")
    for corpus in CORPORA:
        replicates = plan[corpus]
        allowed = set(expected_sources[corpus])
        if not isinstance(replicates, list) or len(replicates) != count:
            raise EvaluatorContractError("bootstrap resample count does not match its plan")
        if any(
            not isinstance(replicate, list)
            or len(replicate) != len(expected_sources[corpus])
            or any(source_id not in allowed for source_id in replicate)
            for replicate in replicates
        ):
            raise EvaluatorContractError(
                "bootstrap resample plan is not whole-meeting within corpus strata"
            )
    if bootstrap["resample_plan_sha256"] != canonical_sha256(plan):
        raise EvaluatorContractError("bootstrap resample plan receipt mismatch")


def validate_evaluator_result(
    result: dict[str, Any],
    contract: dict[str, Any],
    data_dir: Path,
    registry_path: Path,
    source_registry_path: Path,
) -> None:
    expected_contract = build_evaluator_contract(
        data_dir,
        registry_path,
        source_registry_path,
    )
    if canonical_sha256(contract) != canonical_sha256(expected_contract):
        raise EvaluatorContractError("evaluator contract is not current")
    _require_exact_keys(
        result,
        {
            "evaluator_contract_canonical_sha256",
            "score_thresholds",
            "frontiers",
            "bootstrap",
            "decision",
        },
        "evaluator result",
    )
    if result.get("evaluator_contract_canonical_sha256") != canonical_sha256(contract):
        raise EvaluatorContractError("result is not bound to the evaluator contract")
    thresholds = _thresholds(result.get("score_thresholds"))
    frontiers = result.get("frontiers")
    if not isinstance(frontiers, dict):
        raise EvaluatorContractError("frontiers are missing")
    metric_inventories = _validate_frontier_tree(frontiers, contract, thresholds)
    _validate_cross_view_aggregation(
        frontiers,
        contract,
        thresholds,
        metric_inventories["pooled_exposure_weighted"],
    )
    bootstrap = result.get("bootstrap")
    expected_sources = contract["split_binding"]["eval_source_ids_by_corpus"]
    if (
        not isinstance(bootstrap, dict)
        or set(bootstrap)
        != {
            "method",
            "resampling_unit",
            "strata",
            "meeting_ids_by_corpus",
            "resample_count",
            "seed",
            "with_replacement_within_each_corpus",
            "same_resample_indices_across_arms_and_thresholds",
            "resample_indices_by_corpus",
            "resample_plan_sha256",
            "intervals",
        }
        or bootstrap.get("method") != "corpus_stratified_meeting_bootstrap"
        or bootstrap.get("resampling_unit") != "whole_meeting"
        or bootstrap.get("strata") != list(CORPORA)
        or bootstrap.get("meeting_ids_by_corpus") != expected_sources
        or bootstrap.get("with_replacement_within_each_corpus") is not True
        or not isinstance(bootstrap.get("resample_count"), int)
        or isinstance(bootstrap.get("resample_count"), bool)
        or bootstrap["resample_count"] <= 0
        or not isinstance(bootstrap.get("seed"), int)
        or isinstance(bootstrap.get("seed"), bool)
        or bootstrap["seed"] < 0
        or bootstrap.get("same_resample_indices_across_arms_and_thresholds") is not True
    ):
        raise EvaluatorContractError("bootstrap is not corpus-stratified over exact EVAL meetings")
    _validate_resample_plan(bootstrap, expected_sources)
    _validate_interval_tree(
        bootstrap.get("intervals"),
        contract,
        thresholds,
        metric_inventories,
    )
    decision = result.get("decision")
    if not isinstance(decision, dict):
        raise EvaluatorContractError("decision summary is missing")
    _require_exact_keys(
        decision,
        {
            "evidence_views_consulted",
            "universal_winner_claimed",
            "material_equal_corpus_or_corpus_specific_reversal_detected",
            "outcome",
        },
        "decision",
    )
    consulted = decision.get("evidence_views_consulted")
    if (
        not isinstance(consulted, list)
        or any(not isinstance(view, str) for view in consulted)
        or len(consulted) != len(set(consulted))
        or any(view not in REQUIRED_VIEW_IDS for view in consulted)
    ):
        raise EvaluatorContractError("decision evidence view inventory is invalid")
    universal = decision.get("universal_winner_claimed")
    reversal = decision.get("material_equal_corpus_or_corpus_specific_reversal_detected")
    outcome = decision.get("outcome")
    if (
        not isinstance(universal, bool)
        or not isinstance(reversal, bool)
        or outcome not in ALLOWED_OUTCOMES
    ):
        raise EvaluatorContractError("decision policy fields are invalid")
    all_views = set(consulted) == set(REQUIRED_VIEW_IDS)
    if universal and not all_views:
        raise EvaluatorContractError(
            "a universal winner cannot be declared from pooled evidence alone"
        )
    if not all_views:
        raise EvaluatorContractError(
            "every decision outcome requires every required evidence view"
        )
    if outcome == "winner" and (universal is not True or not all_views):
        raise EvaluatorContractError(
            "a winner outcome requires a universal claim supported by every required view"
        )
    if outcome != "winner" and universal:
        raise EvaluatorContractError("only a winner outcome may claim a universal winner")
    if reversal and (
        universal
        or not all_views
        or outcome not in {"corpus_dependent", "inconclusive"}
    ):
        raise EvaluatorContractError(
            "material corpus reversal must be corpus-dependent or inconclusive"
        )
    if outcome == "corpus_dependent" and not all_views:
        raise EvaluatorContractError(
            "a corpus-dependent outcome requires every required evidence view"
        )


def write_evaluator_contract(
    data_dir: Path,
    registry_path: Path,
    source_registry_path: Path,
    output_path: Path,
) -> None:
    contract = build_evaluator_contract(data_dir, registry_path, source_registry_path)
    output_path.write_text(
        json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--source-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_evaluator_contract(
        args.data_dir.resolve(),
        args.registry.resolve(),
        args.source_registry.resolve(),
        args.output.resolve(),
    )


if __name__ == "__main__":
    main()
