from __future__ import annotations

from pathlib import Path

import pytest

import tools.eot_experiment.policy_analysis as policy_analysis
from tools.eot_experiment.policy_analysis import (
    MATCH_TOLERANCE,
    METRIC_FIELDS,
    _candidate_is_valid,
    _candidate_thresholds,
    _final_operating_points,
    _hierarchical_pair_samples,
    _language_decisions,
    _matched_candidate,
    _outer_bootstrap_evidence,
    _outer_splits,
    _pair_bootstrap_records,
    _paired_row,
    _policy_gate_by_language,
    _prepare_inner_selection,
    _select_candidate,
    _select_p3_prepared_inner,
    _select_prepared_inner,
    _split_groups,
    cross_validate,
    simulate_policy,
)


def make_row(
    group: str,
    label: str,
    duration_ms: float,
    *,
    score_224: float = 0.1,
    score_512: float = 0.1,
    turn_id: str | None = None,
) -> dict:
    return {
        "language": "en",
        "conversation_id": group,
        "turn_id": turn_id or group,
        "span_id": f"{group}-{label}-{duration_ms}",
        "label": label,
        "span_duration_ms": duration_ms,
        "score_224": score_224,
        "inference_latency_224_ms": 0.0,
        "score_512": score_512,
        "inference_latency_512_ms": 0.0,
    }


def metric(false_rate: float, endpoint: float, timeout_rate: float = 0.0) -> dict:
    return {
        "false_cutoff_rate": false_rate,
        "mean_endpoint_latency_ms": endpoint,
        "p50_endpoint_latency_ms": endpoint,
        "eot_timeout_rate": timeout_rate,
    }


def test_hold_resuming_before_timeout_is_not_an_eot_timeout() -> None:
    metrics = simulate_policy([make_row("g1", "hold", 300.0)], "P1", 0.5)
    assert metrics["eot_timeout_rate"] == 0.0
    assert metrics["unresolved_span_rate"] == 1.0


def test_rejected_eot_ending_at_timeout_is_an_eot_timeout() -> None:
    metrics = simulate_policy([make_row("g1", "eot", 900.0)], "P1", 0.5)
    assert metrics["eot_timeout_rate"] == 1.0
    assert metrics["unresolved_span_rate"] == 1.0


def test_timeout_and_unresolved_metrics_are_independent() -> None:
    rows = [
        make_row("g1", "hold", 300.0),
        make_row("g2", "eot", 900.0),
    ]
    metrics = simulate_policy(rows, "P1", 0.5)
    assert metrics["eot_timeout_rate"] == 1.0
    assert metrics["unresolved_span_rate"] == 1.0


def test_selection_gate_uses_eot_timeout_rate() -> None:
    baseline = metric(1.0, 512.0)
    candidate = metric(0.5, 400.0, timeout_rate=0.26)
    assert not _candidate_is_valid(candidate, baseline, 0.2)


def test_matched_candidate_is_unavailable_when_difference_exceeds_tolerance() -> None:
    candidate = {"metrics": metric(0.506, 400.0)}
    assert _matched_candidate([candidate], 0.5) is None


def test_matched_candidate_accepts_only_the_explicit_tolerance() -> None:
    candidate = {"metrics": metric(0.5 + MATCH_TOLERANCE, 400.0)}
    selected = _matched_candidate([candidate], 0.5)
    assert selected is not None
    assert selected["matched_status"] == "matched"


def test_observed_threshold_candidates_do_not_use_a_uniform_grid() -> None:
    rows = [
        {"score_224": 0.23},
        {"score_224": 0.87},
    ]
    candidates, source = _candidate_thresholds(rows, "score_224")
    assert candidates == [0.0, 0.23, 0.87, 1.0]
    assert source == "observed_inner_training_scores"


def test_group_split_keeps_conversations_together() -> None:
    rows = [make_row(f"g{index}", "hold", 300.0) for index in range(12)]
    for _, train, test in _split_groups(rows, seed=17, n_folds=3):
        assert {row["conversation_id"] for row in train}.isdisjoint(
            {row["conversation_id"] for row in test}
        )


def test_p3_searches_all_candidates_for_false_cutoff_match() -> None:
    prepared = {
        "baseline": metric(0.8, 512.0),
        "threshold_source": "observed_inner_training_scores",
        "inner_folds": 5,
        "inner_validation_rows": 10,
        "inner_training_rows": 10,
        "candidate_count": 2,
        "candidates": [
            {
                "threshold224": 0.4,
                "threshold512": 0.5,
                "metrics": metric(0.6, 500.0),
            },
            {
                "threshold224": 0.6,
                "threshold512": 0.7,
                "metrics": metric(0.6, 480.0),
            },
        ],
    }
    selected = _select_p3_prepared_inner(
        prepared,
        "low_latency",
        {"inner_false_cutoff_rate": 0.6, "inner_mean_endpoint_latency_ms": 500.0},
    )
    assert selected is not None
    assert selected["threshold224"] == 0.6
    assert selected["selection_mode"] == "false_cutoff_matched"


def test_p3_can_use_mean_latency_match_for_false_cutoff_gain() -> None:
    prepared = {
        "baseline": metric(0.8, 512.0),
        "threshold_source": "observed_inner_training_scores",
        "inner_folds": 5,
        "inner_validation_rows": 10,
        "inner_training_rows": 10,
        "candidate_count": 1,
        "candidates": [
            {
                "threshold224": 0.6,
                "threshold512": 0.7,
                "metrics": metric(0.48, 500.5),
            }
        ],
    }
    selected = _select_p3_prepared_inner(
        prepared,
        "low_latency",
        {"inner_false_cutoff_rate": 0.5, "inner_mean_endpoint_latency_ms": 500.0},
    )
    assert selected is not None
    assert selected["selection_mode"] == "mean_latency_matched"


def test_selection_records_deterministic_tie_reason() -> None:
    candidates = [
        {"threshold224": 0.6, "threshold512": 0.4, "metrics": metric(0.4, 400.0)},
        {"threshold224": 0.6, "threshold512": 0.7, "metrics": metric(0.4, 400.0)},
    ]
    selected = _select_candidate(candidates, metric(0.8, 512.0), 0.2)
    assert selected is not None
    assert selected["selection_tie_count"] == 2
    assert "deterministic threshold ordering" in selected["selection_tie_reason"]


def make_bootstrap_cv() -> dict:
    rows = [
        make_row("g1", "hold", 300.0, score_224=0.9),
        make_row("g1", "eot", 600.0),
        make_row("g2", "hold", 900.0, score_512=0.9),
        make_row("g2", "eot", 900.0),
    ]
    splits = {"en": []}
    cv_rows = []
    paired_rows = []
    for repeat, group in enumerate(("g1", "g2")):
        test_rows = [row for row in rows if row["conversation_id"] == group]
        splits["en"].append({"repeat": repeat, "fold": 0, "test_rows": test_rows})
        for policy, threshold512 in (("P1", None), ("P2", 0.5)):
            cv_rows.append(
                {
                    "language": "en",
                    "repeat": repeat,
                    "outer_fold": 0,
                    "policy": policy,
                    "target": "low_latency",
                    "status": "available",
                    "threshold224": 0.5,
                    "threshold512": threshold512,
                }
            )
        paired_rows.append(
            {
                "language": "en",
                "repeat": repeat,
                "outer_fold": 0,
                "target": "low_latency",
                "status": "available",
                "p1_threshold224": 0.5,
                "p2_threshold224": 0.5,
                "p2_threshold512": 0.5,
            }
        )
    return {"cv_rows": cv_rows, "paired_rows": paired_rows, "outer_splits": splits}


def test_outer_partition_bootstrap_is_reproducible_and_fixed_size() -> None:
    cv = make_bootstrap_cv()
    first = _outer_bootstrap_evidence(cv, ("en",), resamples=11, seed=123)
    second = _outer_bootstrap_evidence(cv, ("en",), resamples=11, seed=123)
    assert first == second
    assert first
    assert {row["resamples"] for row in first} == {11}
    assert {row["bootstrap_unit"] for row in first} == {"conversation_within_outer_test_partition"}
    assert {row["outer_partitions"] for row in first} == {2}


def test_p2_gate_uses_bootstrap_ci_instead_of_outer_percentile() -> None:
    paired = [
        {
            "language": "en",
            "target": "low_latency",
            "status": "available",
            "value_condition": True,
            "p2_false_cutoff_minus_p1": 0.0,
            "false_cutoff_regression_ok": True,
        }
        for _ in range(5)
    ]
    evidence = [
        {
            "language": "en",
            "target": "low_latency",
            "comparison": "P2_vs_P1",
            "metric": "false_cutoff_delta",
            "ci_low": -0.01,
            "ci_high": 0.006,
            "resamples": 10000,
        }
    ]
    gate = _policy_gate_by_language(paired, ("en",), evidence)["en"]
    assert not gate["passes_false_cutoff_ci"]
    assert not gate["passes"]


def test_hierarchical_bootstrap_resamples_outer_repeats_and_conversations() -> None:
    cv = make_bootstrap_cv()
    records = []
    for row in cv["paired_rows"]:
        split = cv["outer_splits"]["en"][row["repeat"]]
        records.append(
            {
                "rows": split["test_rows"],
                "repeat": row["repeat"],
                "reference_threshold224": row["p1_threshold224"],
                "reference_threshold512": None,
                "candidate_threshold224": row["p2_threshold224"],
                "candidate_threshold512": row["p2_threshold512"],
            }
        )
    reference, candidate = _hierarchical_pair_samples(records, "P1", "P2", resamples=13, seed=99)
    assert len(reference["false_cutoff_rate"]) == 13
    assert len(candidate["eot_timeout_rate"]) == 13


def test_language_decision_cannot_shadow_when_p1_heldout_reduction_fails() -> None:
    rows = []
    for index in range(200):
        row = {
            "language": "en",
            "policy": "P1",
            "target": "low_latency",
            "status": "available",
            "threshold224": 0.5,
            "threshold512": None,
            "relative_false_cutoff_reduction": -0.1,
            "false_cutoff_rate": 0.8,
            "mean_endpoint_latency_ms": 400.0,
            "p50_endpoint_latency_ms": 400.0,
            "eot_timeout_rate": 0.1,
        }
        row.update({field: row.get(field) for field in METRIC_FIELDS})
        row.update(
            {
                "repeat": index // 5,
                "outer_seed": index,
                "outer_fold": index % 5,
                "test_rows": 1,
            }
        )
        rows.append(row)
    decisions, _ = _language_decisions(
        {"en": []},
        rows,
        {
            "en": {
                "passes": False,
                "target": "low_latency",
            }
        },
        {"en": {"passes": False, "target": "low_latency"}},
        [
            {
                "language": "en",
                "policy": "P1",
                "target": "low_latency",
                "comparison": "P1_vs_B0",
                "metric": "relative_false_cutoff_reduction",
                "ci_low": -0.2,
                "ci_high": 0.0,
                "resamples": 10000,
            }
        ],
    )
    assert decisions["en"]["decision"] == "BASELINE_ONLY"


def test_no_matched_candidate_does_not_fallback_to_boundary_candidate() -> None:
    prepared = {
        "baseline": metric(0.8, 512.0),
        "threshold_source": "observed_inner_training_scores",
        "inner_folds": 5,
        "inner_validation_rows": 10,
        "inner_training_rows": 10,
        "candidate_count": 1,
        "candidates": [
            {
                "threshold224": 1.0,
                "threshold512": 1.0,
                "metrics": metric(0.6, 400.0),
            }
        ],
    }
    assert (
        _select_p3_prepared_inner(
            prepared,
            "low_latency",
            {
                "inner_false_cutoff_rate": 0.5,
                "inner_mean_endpoint_latency_ms": 500.0,
            },
        )
        is None
    )


def test_inner_selection_excludes_outer_test_scores() -> None:
    training_rows = [
        make_row(
            f"train-{index}",
            "hold",
            600.0,
            score_224=0.1 + index * 0.05,
        )
        for index in range(5)
    ]
    outer_test_row = make_row("outer-test", "hold", 600.0, score_224=0.99)
    prepared = _prepare_inner_selection(training_rows, "P1", seed=17)
    assert prepared is not None
    thresholds = {candidate["threshold224"] for candidate in prepared["candidates"]}
    assert outer_test_row["score_224"] not in thresholds
    assert prepared["inner_validation_rows"] == len(training_rows)
    assert prepared["inner_training_rows"] == len(training_rows) * 4
    assert prepared["threshold_source"] == "observed_inner_training_scores"


def test_inner_validation_metrics_select_the_threshold() -> None:
    prepared = {
        "baseline": metric(0.8, 512.0),
        "threshold_source": "observed_inner_training_scores",
        "inner_folds": 5,
        "inner_validation_rows": 10,
        "inner_training_rows": 40,
        "candidate_count": 2,
        "candidates": [
            {
                "threshold224": 0.2,
                "threshold512": None,
                "metrics": metric(0.5, 500.0),
            },
            {
                "threshold224": 0.8,
                "threshold512": None,
                "metrics": metric(0.6, 300.0),
            },
        ],
    }
    selected = _select_prepared_inner(prepared, "low_latency")
    assert selected is not None
    assert selected["threshold224"] == 0.8
    assert selected["inner_false_cutoff_rate"] == 0.6
    assert selected["inner_mean_endpoint_latency_ms"] == 300.0


def make_repeated_split_rows(group_count: int = 100) -> list[dict]:
    rows = []
    for index in range(group_count):
        group = f"group-{index}"
        rows.extend(
            [
                make_row(group, "hold", 600.0, score_224=0.9, score_512=0.9),
                make_row(group, "eot", 900.0, score_224=0.9, score_512=0.9),
            ]
        )
    return rows


def test_outer_splits_have_50_repeats_and_are_seed_deterministic() -> None:
    rows = make_repeated_split_rows()
    first_rejected = []
    second_rejected = []
    first = _outer_splits(rows, "en", first_rejected)
    second = _outer_splits(rows, "en", second_rejected)
    assert len(first) == 50 * 5
    assert {split["repeat"] for split in first} == set(range(50))
    assert all(sum(split["repeat"] == repeat for split in first) == 5 for repeat in range(50))
    assert first == second
    assert first_rejected == second_rejected == []


def test_invalid_outer_folds_are_rejected_with_deterministic_reasons(monkeypatch) -> None:
    rows = [make_row("group-0", "hold", 600.0)]
    rows.extend(make_row("group-0", "eot", 900.0, turn_id=f"eot-{index}") for index in range(5))
    rows.extend(make_row(f"group-{index}", "hold", 600.0) for index in range(1, 5))
    monkeypatch.setattr(policy_analysis, "CV_SEEDS", (17,))
    monkeypatch.setattr(policy_analysis, "OUTER_ATTEMPTS_PER_REPEAT", 2)
    monkeypatch.setattr(policy_analysis, "MIN_OUTER_EOT", 1)
    monkeypatch.setattr(policy_analysis, "MIN_OUTER_ELIGIBLE_HOLD", 1)
    first_rejected = []
    second_rejected = []
    assert _outer_splits(rows, "en", first_rejected) == []
    assert _outer_splits(rows, "en", second_rejected) == []
    assert first_rejected == second_rejected
    assert any(item["status"] == "rejected" for item in first_rejected)
    assert any(item["status"] == "unavailable" for item in first_rejected)
    assert all(item.get("reason") for item in first_rejected)


def test_cross_validate_evaluates_each_selected_outer_candidate_once(
    monkeypatch, tmp_path: Path
) -> None:
    train_rows = [make_row("train", "hold", 600.0, score_224=0.9, score_512=0.9)]
    test_rows = [
        make_row("test", "hold", 600.0, score_224=0.9, score_512=0.9),
        make_row("test", "eot", 900.0, score_224=0.9, score_512=0.9),
    ]
    split = {
        "language": "en",
        "repeat": 0,
        "seed": 17,
        "fold": 0,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "train_groups": 1,
        "test_groups": 1,
        "test_eot": 1,
        "test_eligible_hold": 1,
        "folds_requested": 5,
        "folds_used": 1,
        "fold_reduced": True,
    }
    calls = []
    real_simulate_policy = policy_analysis.simulate_policy

    def spy_simulate_policy(rows, policy, *args, **kwargs):
        if rows is test_rows and policy in {"P1", "P2"}:
            calls.append(policy)
        return real_simulate_policy(rows, policy, *args, **kwargs)

    def fake_prepare(rows, policy, seed):
        return {"policy": policy}

    def fake_select(prepared, target):
        return {
            "threshold224": 0.0,
            "threshold512": 0.0 if prepared["policy"] == "P2" else None,
            "threshold_source": "observed_inner_training_scores",
            "inner_folds": 2,
            "inner_validation_rows": 1,
            "inner_training_rows": 1,
            "candidate_count": 1,
            "inner_false_cutoff_rate": 0.0,
            "inner_mean_endpoint_latency_ms": 224.0,
            "inner_eot_timeout_rate": 0.0,
        }

    monkeypatch.setattr(policy_analysis, "_outer_splits", lambda rows, language, rejected: [split])
    monkeypatch.setattr(policy_analysis, "_prepare_inner_selection", fake_prepare)
    monkeypatch.setattr(policy_analysis, "_select_prepared_inner", fake_select)
    monkeypatch.setattr(policy_analysis, "_outer_bootstrap_evidence", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        policy_analysis,
        "_policy_gate_by_language",
        lambda paired, languages, evidence: {"en": {"passes": False}},
    )
    monkeypatch.setattr(policy_analysis, "simulate_policy", spy_simulate_policy)

    result = cross_validate(
        {"en": train_rows + test_rows},
        output_dir=tmp_path,
        bootstrap_resamples=3,
    )
    candidate_rows = [row for row in result["cv_rows"] if row["policy"] in {"P1", "P2"}]
    keys = [
        (row["policy"], row["target"], row["repeat"], row["outer_fold"]) for row in candidate_rows
    ]
    assert len(candidate_rows) == 4
    assert len(keys) == len(set(keys))
    assert calls.count("P1") == 2
    assert calls.count("P2") == 2


def make_cv_row(
    policy: str,
    *,
    repeat: int,
    status: str = "available",
    target: str = "low_latency",
) -> dict:
    row = {field: 0.0 for field in METRIC_FIELDS}
    row.update(
        {
            "language": "en",
            "policy": policy,
            "target": target,
            "repeat": repeat,
            "outer_seed": 100 + repeat,
            "outer_fold": 0,
            "status": status,
            "threshold224": 0.4 if status == "available" else None,
            "threshold512": 0.5 if policy != "P1" and status == "available" else None,
            "false_cutoff_rate": 0.4,
            "relative_false_cutoff_reduction": 0.5,
            "mean_endpoint_latency_ms": 400.0,
            "p50_endpoint_latency_ms": 400.0,
            "eot_timeout_rate": 0.1,
        }
    )
    return row


def test_paired_rows_keep_the_same_outer_partition_and_preserve_missing_values() -> None:
    split = {"language": "en", "repeat": 4, "seed": 101, "fold": 2}
    p1 = {
        "threshold224": 0.4,
        "mean_endpoint_latency_ms": 500.0,
        "false_cutoff_rate": 0.4,
        "eot_timeout_rate": 0.1,
        "turn_fragmentation_rate": 0.2,
    }
    p2 = {
        "threshold224": 0.5,
        "threshold512": 0.5,
        "mean_endpoint_latency_ms": 470.0,
        "false_cutoff_rate": 0.402,
        "eot_timeout_rate": 0.04,
        "turn_fragmentation_rate": 0.1,
    }
    paired = _paired_row(p1=p1, p2=p2, split=split, target="low_latency")
    assert (paired["repeat"], paired["outer_seed"], paired["outer_fold"]) == (4, 101, 2)
    assert paired["p2_mean_endpoint_minus_p1_ms"] == -30.0
    assert paired["p2_false_cutoff_minus_p1"] == pytest.approx(0.002)
    assert paired["p2_eot_timeout_minus_p1"] == pytest.approx(-0.06)
    missing = _paired_row(p1=p1, p2=None, split=split, target="low_latency")
    assert missing["status"] == "unavailable_candidate"
    assert missing["p2_mean_endpoint_minus_p1_ms"] is None
    assert missing["value_condition"] is None


def test_missing_paired_candidates_reduce_availability_without_zero_imputation() -> None:
    rows = [
        make_cv_row("P2", repeat=0, status="available"),
        make_cv_row("P2", repeat=1, status="unavailable"),
    ]
    point = _final_operating_points(rows)[("en", "P2", "low_latency")]
    assert point["folds_expected"] == 2
    assert point["folds_available"] == 1
    assert point["availability_rate"] == 0.5
    assert point["heldout_mean"]["false_cutoff_rate"] == 0.4


def test_pair_bootstrap_records_match_reference_and_candidate_on_each_fold() -> None:
    cv = make_bootstrap_cv()
    records = _pair_bootstrap_records(cv, "en", "P1", "P2", "low_latency")
    assert {(record["repeat"], record["outer_fold"]) for record in records} == {
        (0, 0),
        (1, 0),
    }
    assert [record["rows"][0]["conversation_id"] for record in records] == ["g1", "g2"]
    assert all(record["reference_threshold224"] == 0.5 for record in records)
    assert all(record["candidate_threshold512"] == 0.5 for record in records)


def test_conversation_bootstrap_generates_exact_10000_reproducible_intervals() -> None:
    cv = make_bootstrap_cv()
    first = _outer_bootstrap_evidence(
        cv,
        ("en",),
        resamples=10_000,
        seed=20260802,
        policies=("P1",),
        comparisons=(),
    )
    second = _outer_bootstrap_evidence(
        cv,
        ("en",),
        resamples=10_000,
        seed=20260802,
        policies=("P1",),
        comparisons=(),
    )
    assert first == second
    assert first
    assert {row["resamples"] for row in first} == {10_000}
    assert {row["bootstrap_unit"] for row in first} == {"conversation_within_outer_test_partition"}
