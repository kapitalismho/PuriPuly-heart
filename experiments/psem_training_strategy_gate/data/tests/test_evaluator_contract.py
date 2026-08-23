from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.psem_training_strategy_gate.data.evaluator_contract import (
    CORPORA,
    REQUIRED_VIEW_IDS,
    EvaluatorContractError,
    build_evaluator_contract,
    validate_evaluator_result,
    write_evaluator_contract,
)
from experiments.psem_training_strategy_gate.data.provenance import canonical_sha256

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = Path(__file__).resolve().parents[1] / "v2"
REGISTRY_PATH = REPO_ROOT / "experiments/speaker_representation_scd/models/registry.json"
SOURCE_REGISTRY_PATH = (
    REPO_ROOT / "experiments/speaker_representation_scd/models/source_registry.json"
)


def _frontier_rows(thresholds: list[float]) -> list[dict[str, object]]:
    return [
        {
            "score_threshold": threshold,
            "metrics": {"precision": 0.8, "recall": 0.7},
        }
        for threshold in thresholds
    ]


def _interval_rows(thresholds: list[float]) -> list[dict[str, object]]:
    return [
        {
            "score_threshold": threshold,
            "metrics": {
                "precision": {"lower": 0.75, "upper": 0.85},
                "recall": {"lower": 0.65, "upper": 0.75},
            },
        }
        for threshold in thresholds
    ]


def _result(contract: dict[str, object]) -> dict[str, object]:
    thresholds = [0.25, 0.5, 0.75]
    topologies = contract["required_outputs"]["topology_by_corpus"]["topologies"]
    frontiers = {
        "pooled_exposure_weighted": _frontier_rows(thresholds),
        "equal_corpus_macro": _frontier_rows(thresholds),
        "corpus_specific": {corpus: _frontier_rows(thresholds) for corpus in CORPORA},
        "topology_by_corpus": {
            corpus: {topology: _frontier_rows(thresholds) for topology in topologies}
            for corpus in CORPORA
        },
    }
    intervals = {
        "pooled_exposure_weighted": _interval_rows(thresholds),
        "equal_corpus_macro": _interval_rows(thresholds),
        "corpus_specific": {corpus: _interval_rows(thresholds) for corpus in CORPORA},
        "topology_by_corpus": {
            corpus: {topology: _interval_rows(thresholds) for topology in topologies}
            for corpus in CORPORA
        },
    }
    resample_count = 3
    resample_plan = {
        corpus: [
            list(contract["split_binding"]["eval_source_ids_by_corpus"][corpus])
            for _ in range(resample_count)
        ]
        for corpus in CORPORA
    }
    return {
        "evaluator_contract_canonical_sha256": canonical_sha256(contract),
        "score_thresholds": thresholds,
        "frontiers": frontiers,
        "bootstrap": {
            "method": "corpus_stratified_meeting_bootstrap",
            "resampling_unit": "whole_meeting",
            "strata": list(CORPORA),
            "meeting_ids_by_corpus": contract["split_binding"]["eval_source_ids_by_corpus"],
            "resample_count": resample_count,
            "seed": 760086,
            "with_replacement_within_each_corpus": True,
            "same_resample_indices_across_arms_and_thresholds": True,
            "resample_indices_by_corpus": resample_plan,
            "resample_plan_sha256": canonical_sha256(resample_plan),
            "intervals": intervals,
        },
        "decision": {
            "evidence_views_consulted": list(REQUIRED_VIEW_IDS),
            "universal_winner_claimed": True,
            "material_equal_corpus_or_corpus_specific_reversal_detected": False,
            "outcome": "winner",
        },
    }


def _validate(result: dict[str, object], contract: dict[str, object]) -> None:
    validate_evaluator_result(
        result,
        contract,
        DATA_DIR,
        REGISTRY_PATH,
        SOURCE_REGISTRY_PATH,
    )


def test_checked_evaluator_contract_is_current_and_complete() -> None:
    checked = json.loads((DATA_DIR / "evaluator_contract.json").read_text(encoding="utf-8"))
    assert checked == build_evaluator_contract(DATA_DIR, REGISTRY_PATH, SOURCE_REGISTRY_PATH)
    assert set(checked["required_outputs"]) == set(REQUIRED_VIEW_IDS)
    assert checked["threshold_policy"]["per_corpus_thresholds_allowed"] is False
    assert checked["threshold_policy"]["same_threshold_vector_required_for_every_output"] is True
    assert checked["decision_policy"]["pooled_frontier_alone_may_declare_universal_winner"] is False
    assert checked["split_binding"]["eval_source_count_by_corpus"] == {
        "AMI": 11,
        "AliMeeting": 8,
    }
    assert all(
        count > 0
        for counts in checked["split_binding"]["topology_counts_by_corpus"].values()
        for count in counts.values()
    )


def test_evaluator_contract_output_is_deterministic(tmp_path: Path) -> None:
    output = tmp_path / "evaluator_contract.json"
    write_evaluator_contract(
        DATA_DIR,
        REGISTRY_PATH,
        SOURCE_REGISTRY_PATH,
        output,
    )
    first = output.read_bytes()
    write_evaluator_contract(
        DATA_DIR,
        REGISTRY_PATH,
        SOURCE_REGISTRY_PATH,
        output,
    )
    assert output.read_bytes() == first


def test_complete_shared_threshold_result_passes() -> None:
    contract = json.loads((DATA_DIR / "evaluator_contract.json").read_text(encoding="utf-8"))
    _validate(_result(contract), contract)


def test_per_corpus_threshold_vector_fails_closed() -> None:
    contract = json.loads((DATA_DIR / "evaluator_contract.json").read_text(encoding="utf-8"))
    result = _result(contract)
    result["frontiers"]["corpus_specific"]["AliMeeting"][1]["score_threshold"] = 0.6
    with pytest.raises(EvaluatorContractError, match="shared vector"):
        _validate(result, contract)


def test_pooled_only_universal_winner_fails_closed() -> None:
    contract = json.loads((DATA_DIR / "evaluator_contract.json").read_text(encoding="utf-8"))
    result = _result(contract)
    result["decision"]["evidence_views_consulted"] = ["pooled_exposure_weighted"]
    with pytest.raises(EvaluatorContractError, match="pooled evidence alone"):
        _validate(result, contract)


def test_material_corpus_reversal_cannot_report_universal_winner() -> None:
    contract = json.loads((DATA_DIR / "evaluator_contract.json").read_text(encoding="utf-8"))
    result = _result(contract)
    result["decision"]["material_equal_corpus_or_corpus_specific_reversal_detected"] = True
    with pytest.raises(EvaluatorContractError, match="corpus-dependent or inconclusive"):
        _validate(result, contract)
    result = _result(contract)
    result["decision"] = {
        "evidence_views_consulted": ["pooled_exposure_weighted"],
        "universal_winner_claimed": False,
        "material_equal_corpus_or_corpus_specific_reversal_detected": True,
        "outcome": "corpus_dependent",
    }
    with pytest.raises(EvaluatorContractError, match="every required evidence view"):
        _validate(result, contract)


def test_bootstrap_must_cover_exact_corpus_stratified_eval_meetings() -> None:
    contract = json.loads((DATA_DIR / "evaluator_contract.json").read_text(encoding="utf-8"))
    result = _result(contract)
    result["bootstrap"]["meeting_ids_by_corpus"] = copy.deepcopy(
        result["bootstrap"]["meeting_ids_by_corpus"]
    )
    result["bootstrap"]["meeting_ids_by_corpus"]["AMI"].pop()
    with pytest.raises(EvaluatorContractError, match="exact EVAL meetings"):
        _validate(result, contract)


def test_bootstrap_interval_metrics_must_match_frontier_metrics() -> None:
    contract = json.loads((DATA_DIR / "evaluator_contract.json").read_text(encoding="utf-8"))
    result = _result(contract)
    result["bootstrap"]["intervals"]["pooled_exposure_weighted"][0]["metrics"].pop("recall")
    with pytest.raises(EvaluatorContractError, match="differs from its frontier"):
        _validate(result, contract)


def test_tampered_contract_fails_even_when_result_hash_is_recomputed() -> None:
    contract = json.loads(
        (DATA_DIR / "evaluator_contract.json").read_text(encoding="utf-8")
    )
    changed = copy.deepcopy(contract)
    changed["dataset_freeze_id"] = "PSEM-STRATEGY-DATA-v1"
    result = _result(changed)
    with pytest.raises(EvaluatorContractError, match="contract is not current"):
        _validate(result, changed)


@pytest.mark.parametrize("schema_version", [1.0, True])
def test_numeric_type_mutated_contract_fails_closed(
    schema_version: float | bool,
) -> None:
    contract = json.loads(
        (DATA_DIR / "evaluator_contract.json").read_text(encoding="utf-8")
    )
    changed = copy.deepcopy(contract)
    changed["schema_version"] = schema_version
    result = _result(changed)
    with pytest.raises(EvaluatorContractError, match="contract is not current"):
        _validate(result, changed)


def test_equal_macro_and_pooled_aggregation_are_recomputed() -> None:
    contract = json.loads(
        (DATA_DIR / "evaluator_contract.json").read_text(encoding="utf-8")
    )
    result = _result(contract)
    result["frontiers"]["equal_corpus_macro"][0]["metrics"]["precision"] = 0.9
    with pytest.raises(EvaluatorContractError, match="arithmetic mean"):
        _validate(result, contract)


def test_bootstrap_requires_bound_with_replacement_resample_plan() -> None:
    contract = json.loads(
        (DATA_DIR / "evaluator_contract.json").read_text(encoding="utf-8")
    )
    result = _result(contract)
    result["bootstrap"]["with_replacement_within_each_corpus"] = False
    with pytest.raises(EvaluatorContractError, match="corpus-stratified"):
        _validate(result, contract)
    result = _result(contract)
    result["bootstrap"]["resample_indices_by_corpus"]["AMI"][0][0] = "not-eval"
    result["bootstrap"]["resample_plan_sha256"] = canonical_sha256(
        result["bootstrap"]["resample_indices_by_corpus"]
    )
    with pytest.raises(EvaluatorContractError, match="whole-meeting"):
        _validate(result, contract)


def test_unknown_result_fields_and_duplicate_decision_views_fail_closed() -> None:
    contract = json.loads(
        (DATA_DIR / "evaluator_contract.json").read_text(encoding="utf-8")
    )
    result = _result(contract)
    result["frontiers"]["unexpected"] = []
    with pytest.raises(EvaluatorContractError, match="frontiers field inventory"):
        _validate(result, contract)
    result = _result(contract)
    result["decision"]["evidence_views_consulted"].append(REQUIRED_VIEW_IDS[0])
    with pytest.raises(EvaluatorContractError, match="evidence view inventory"):
        _validate(result, contract)


def test_every_outcome_requires_complete_decision_evidence() -> None:
    contract = json.loads(
        (DATA_DIR / "evaluator_contract.json").read_text(encoding="utf-8")
    )
    result = _result(contract)
    result["decision"] = {
        "evidence_views_consulted": ["pooled_exposure_weighted"],
        "universal_winner_claimed": False,
        "material_equal_corpus_or_corpus_specific_reversal_detected": False,
        "outcome": "close_runner_up",
    }
    with pytest.raises(EvaluatorContractError, match="every required evidence view"):
        _validate(result, contract)
