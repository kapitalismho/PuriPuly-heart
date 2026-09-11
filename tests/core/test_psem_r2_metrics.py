from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.psem_r2_policy.metrics import (
    MIN_ELIGIBLE_CLUSTERS,
    attribute_tokens,
    confirmatory_decision,
    conservation_record,
    fragmentation_record,
    paired_cluster_bootstrap,
    policy_delta_rows,
    sequential_merge_contamination,
)


def test_conservation_prefix_reports_lost_suffix() -> None:
    record = conservation_record(
        parent_text="Hello there",
        unit_texts=["Hello "],
        token_texts=["Hello ", "there"],
        token_ids=[0, 1],
        unit_token_ids=[[0]],
    )
    assert record["missing"] == "there"
    assert record["missing_text"] == "there"
    assert record["duplicate_text"] == ""
    assert record["missing_token_ids"] == [1]
    assert record["duplicate_token_ids"] == []
    assert record["order_preserved"] is False
    assert record["conserved_units_to_parent"] is False
    assert record["conserved_token_ids"] is False


def test_conservation_token_ids_independent_of_text() -> None:
    record = conservation_record(
        parent_text="ab",
        unit_texts=["ab"],
        token_texts=["a", "b"],
        token_ids=["t0", "t1"],
        unit_token_ids=[["t0", "t0"]],
    )
    assert record["conserved_units_to_parent"] is True
    assert record["missing_token_ids"] == ["t1"]
    assert record["duplicate_token_ids"] == ["t0"]
    assert record["order_preserved"] is False
    assert record["conserved_token_ids"] is False


def test_fragmentation_singleton_is_lexical_not_group_name() -> None:
    record = fragmentation_record(
        ["OTHER-1", "CURRENT-0"],
        unit_texts=["Hi", "Hello there"],
        unit_token_counts=[1, 2],
    )
    assert record["singleton_fragments"] == 1
    assert record["children_per_parent"] == 2


def test_mixed_ami_overlap_is_retained_not_forced() -> None:
    tokens = [
        {
            "token_id": 0,
            "text": "hi",
            "source_start_sample": 0,
            "source_end_sample": 1600,
        }
    ]
    words = [
        {"role": "A", "start_src": 0, "end_src": 1200, "text": "a"},
        {"role": "B", "start_src": 800, "end_src": 2000, "text": "b"},
    ]
    rows = attribute_tokens(tokens, words)
    assert rows[0]["status"] == "mixed"
    assert rows[0]["ambiguous"] is True
    assert rows[0]["roles"] == ["A", "B"]


def test_no_attributable_text_is_ineligible_not_zero() -> None:
    units = [
        {
            "group_id": "CURRENT-0",
            "relation": "CURRENT",
            "token_indexes": [0],
            "start_source_sample": 0,
            "end_source_sample": 1600,
        }
    ]
    attributed = [
        {
            "token_id": 0,
            "text": "hi",
            "status": "unaligned",
            "roles": [],
            "start_src": 0,
            "end_src": 1600,
            "ambiguous": False,
        }
    ]
    scored = sequential_merge_contamination(units=units, attributed=attributed, words=[])
    assert scored["eligible"] is False
    assert scored["proportion"] is None
    assert scored["coverage"] == "none"
    assert scored["attributable_chars"] == 0


def test_sequential_merge_counts_crossed_speaker_text() -> None:
    words = [
        {"role": "A", "start_src": 0, "end_src": 1600, "text": "aa"},
        {"role": "B", "start_src": 1600, "end_src": 3200, "text": "bb"},
    ]
    attributed = [
        {
            "token_id": 0,
            "text": "aa",
            "status": "attributable",
            "roles": ["A"],
            "start_src": 0,
            "end_src": 1600,
            "ambiguous": False,
        },
        {
            "token_id": 1,
            "text": "bb",
            "status": "attributable",
            "roles": ["B"],
            "start_src": 1600,
            "end_src": 3200,
            "ambiguous": False,
        },
    ]
    merged = sequential_merge_contamination(
        units=[
            {
                "group_id": "CURRENT-0",
                "relation": "CURRENT",
                "token_indexes": [0, 1],
                "start_source_sample": 0,
                "end_source_sample": 3200,
            }
        ],
        attributed=attributed,
        words=words,
    )
    split = sequential_merge_contamination(
        units=[
            {
                "group_id": "CURRENT-0",
                "relation": "CURRENT",
                "token_indexes": [0],
                "start_source_sample": 0,
                "end_source_sample": 1600,
            },
            {
                "group_id": "OTHER-1",
                "relation": "OTHER",
                "token_indexes": [1],
                "start_source_sample": 1600,
                "end_source_sample": 3200,
            },
        ],
        attributed=attributed,
        words=words,
    )
    assert merged["eligible"] is True
    assert merged["proportion"] == 0.5
    assert split["eligible"] is True
    assert split["proportion"] == 0.0
    rows = policy_delta_rows(
        per_cluster={
            "ES2009": {
                "R0": {"contamination": merged},
                "R2": {"contamination": split},
            }
        }
    )
    assert rows[0]["delta"] == -0.5
    assert rows[0]["eligible"] is True


def test_confirmatory_under_eight_clusters_is_inconclusive() -> None:
    rows = [
        {"cluster_id": "ES2009", "eligible": True, "delta": -0.2},
        {"cluster_id": "ES2002", "eligible": True, "delta": -0.1},
        {"cluster_id": "EN2009", "eligible": True, "delta": -0.3},
    ]
    decision = confirmatory_decision(cluster_rows=rows)
    assert decision["n_eligible_clusters"] == 3
    assert decision["n_eligible_clusters"] < MIN_ELIGIBLE_CLUSTERS
    assert decision["pass"] is False
    assert decision["result"].startswith("Inconclusive")
    assert decision["cluster_mean_delta"] == pytest_approx_mean([-0.2, -0.1, -0.3])
    boot = paired_cluster_bootstrap([-0.2, -0.1, -0.3])
    assert boot["seed"] == 156
    assert boot["resamples"] == 10000
    assert boot["n_clusters"] == 3


def pytest_approx_mean(values: list[float]) -> float:
    return sum(values) / len(values)


def test_confirmatory_eight_clusters_can_support() -> None:
    rows = [{"cluster_id": str(index), "eligible": True, "delta": -0.1} for index in range(8)]
    decision = confirmatory_decision(cluster_rows=rows)
    assert decision["n_eligible_clusters"] == 8
    assert decision["pass"] is True
    assert decision["ci95"][1] < 0
    assert decision["result"].startswith("Supported")
