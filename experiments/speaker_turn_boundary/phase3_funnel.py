from __future__ import annotations

from typing import Any

from experiments.speaker_turn_boundary.metrics import RECALL_DEADLINES_MS
from experiments.speaker_turn_boundary.phase3_metrics import (
    PRIMARY_DEADLINE_MS,
    REFERENCE_EXTRA_FALSE_CUT_COUNTS,
    REFERENCE_FALSE_CUT_RATES_PER_SOURCE_HOUR,
)
from experiments.speaker_turn_boundary.schemas import sha256_hex

FRONTIER_SCHEMA = "experiments.speaker_turn_boundary.phase3.frontier.v2"
FROZEN_SCHEMA = "experiments.speaker_turn_boundary.phase3.frozen_panel.v2"


def _cost(row: dict[str, Any]) -> int:
    return int(row["extra_false_cuts"])


def _recovered(row: dict[str, Any], deadline: int) -> int:
    return int(row["recovered_b0_misses_at_ms"][str(deadline)])


def _causal_delay(row: dict[str, Any]) -> float:
    summary = row.get("timing", {}).get("incremental_recoveries_at_2000ms", {})
    value = summary.get("causal_audio_delay_ms", {}).get("mean")
    return float(value) if value is not None else float("inf")


def dominates(candidate: dict[str, Any], row: dict[str, Any]) -> bool:
    if candidate["profile_id"] == row["profile_id"]:
        return False
    if _cost(candidate) > _cost(row):
        return False
    no_worse = all(
        _recovered(candidate, deadline) >= _recovered(row, deadline)
        for deadline in RECALL_DEADLINES_MS
    )
    if not no_worse:
        return False
    return _cost(candidate) < _cost(row) or any(
        _recovered(candidate, deadline) > _recovered(row, deadline)
        for deadline in RECALL_DEADLINES_MS
    )


def pareto_frontier(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    frontier: list[dict[str, Any]] = []
    dominated_ids: list[str] = []
    for row in rows:
        if any(dominates(other, row) for other in rows if other is not row):
            dominated_ids.append(str(row["profile_id"]))
        else:
            frontier.append(row)
    frontier.sort(key=_frontier_order)
    dominated_ids.sort()
    return frontier, dominated_ids


def _frontier_order(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _cost(row),
        -_recovered(row, PRIMARY_DEADLINE_MS),
        -_recovered(row, 250),
        -_recovered(row, 1000),
        _causal_delay(row),
        str(row["profile_id"]),
    )


def best_within_cost(
    rows: list[dict[str, Any]],
    allowance: int,
) -> dict[str, Any] | None:
    eligible = [row for row in rows if _cost(row) <= allowance]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda row: (
            -_recovered(row, PRIMARY_DEADLINE_MS),
            -_recovered(row, 250),
            -_recovered(row, 1000),
            -_recovered(row, 1500),
            -_recovered(row, 2000),
            _cost(row),
            _causal_delay(row),
            str(row["profile_id"]),
        ),
    )


def integer_operating_curve(
    rows: list[dict[str, Any]],
    allowances: tuple[int, ...] = REFERENCE_EXTRA_FALSE_CUT_COUNTS,
) -> dict[str, Any]:
    reference: dict[str, Any] = {}
    for allowance in allowances:
        row = best_within_cost(rows, allowance)
        reference[str(allowance)] = None if row is None else _curve_point(row, allowance)
    frontier, _ = pareto_frontier(rows)
    achieved: list[dict[str, Any]] = []
    previous_id: str | None = None
    for allowance in sorted({_cost(row) for row in frontier}):
        row = best_within_cost(rows, allowance)
        if row is None or row["profile_id"] == previous_id:
            continue
        achieved.append(_curve_point(row, allowance))
        previous_id = str(row["profile_id"])
    return {
        "cost_unit": "actual_incremental_false_cuts_on_this_split",
        "reference_allowances": reference,
        "all_achieved_breakpoints": achieved,
    }


def _curve_point(row: dict[str, Any], allowance: int) -> dict[str, Any]:
    return {
        "allowance": allowance,
        "profile_id": row["profile_id"],
        "actual_extra_false_cuts": _cost(row),
        "recovered_b0_misses_at_ms": row["recovered_b0_misses_at_ms"],
        "added_logical_cuts": row["added_logical_cuts"],
    }


def rate_reference_slices(rows: list[dict[str, Any]]) -> dict[str, Any]:
    slices: dict[str, Any] = {}
    for rate in REFERENCE_FALSE_CUT_RATES_PER_SOURCE_HOUR:
        eligible = [
            row for row in rows if float(row["incremental_false_per_source_hour"]["rate"]) <= rate
        ]
        if not eligible:
            slices[f"{rate:g}"] = None
            continue
        row = min(
            eligible,
            key=lambda item: (
                -_recovered(item, PRIMARY_DEADLINE_MS),
                -_recovered(item, 250),
                _cost(item),
                str(item["profile_id"]),
            ),
        )
        slices[f"{rate:g}"] = {
            "reference_rate_per_source_hour": rate,
            "profile_id": row["profile_id"],
            "actual_extra_false_cuts": _cost(row),
            "actual_rate_per_source_hour": row["incremental_false_per_source_hour"],
            "recovered_b0_misses_at_ms": row["recovered_b0_misses_at_ms"],
        }
    return {
        "role": "descriptive_reference_only_not_an_elimination_cap",
        "slices": slices,
    }


def freeze_panel(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frontier, _ = pareto_frontier(rows)
    if not frontier:
        return []
    gains = [row for row in frontier if _recovered(row, PRIMARY_DEADLINE_MS) > 0]
    if not gains:
        return []
    selected = min(
        gains,
        key=lambda row: (
            (
                float("-inf")
                if _cost(row) == 0
                else -_recovered(row, PRIMARY_DEADLINE_MS) / _cost(row)
            ),
            -_recovered(row, 250),
            -_recovered(row, PRIMARY_DEADLINE_MS),
            _cost(row),
            _causal_delay(row),
            str(row["profile_id"]),
        ),
    )
    return [
        {
            "profile_id": str(selected["profile_id"]),
            "family": selected["family"],
            "checkpoint": selected["checkpoint"],
            "profile_kind": selected["profile_kind"],
            "params": selected["params"],
            "selection_roles": ["maximum_500ms_recovery_per_extra_false_cut"],
            "dev_extra_false_cuts": _cost(selected),
            "dev_recovered_b0_misses_at_ms": selected["recovered_b0_misses_at_ms"],
            "dev_recovery_efficiency_at_500ms": (
                None
                if _cost(selected) == 0
                else _recovered(selected, PRIMARY_DEADLINE_MS) / _cost(selected)
            ),
        }
    ]


def _frontier_knee(frontier: list[dict[str, Any]]) -> dict[str, Any] | None:
    collapsed: dict[int, dict[str, Any]] = {}
    for cost in sorted({_cost(row) for row in frontier}):
        best = best_within_cost(frontier, cost)
        if best is not None:
            collapsed[cost] = best
    points = list(collapsed.values())
    if not points:
        return None
    min_cost = min(_cost(row) for row in points)
    max_cost = max(_cost(row) for row in points)
    min_recovery = min(_recovered(row, PRIMARY_DEADLINE_MS) for row in points)
    max_recovery = max(_recovered(row, PRIMARY_DEADLINE_MS) for row in points)
    if min_cost == max_cost or min_recovery == max_recovery:
        return min(points, key=_frontier_order)
    return max(
        points,
        key=lambda row: (
            (
                (_recovered(row, PRIMARY_DEADLINE_MS) - min_recovery)
                / (max_recovery - min_recovery)
                - (_cost(row) - min_cost) / (max_cost - min_cost)
            ),
            _recovered(row, 250),
            -_cost(row),
            -_causal_delay(row),
            str(row["profile_id"]),
        ),
    )


def build_frontier_summary(
    rows: list[dict[str, Any]],
    *,
    family: str,
) -> dict[str, Any]:
    frontier, dominated = pareto_frontier(rows)
    return {
        "schema_version": FRONTIER_SCHEMA,
        "family": family,
        "profile_count": len(rows),
        "frontier_count": len(frontier),
        "frontier_profile_ids": [row["profile_id"] for row in frontier],
        "dominated_count": len(dominated),
        "integer_operating_curve": integer_operating_curve(rows),
        "rate_reference_slices": rate_reference_slices(rows),
        "frozen_panel": freeze_panel(rows),
    }


def build_frozen_artifact(
    *,
    panels: dict[str, list[dict[str, Any]]],
    manifest_sha256: str,
    model_hashes: dict[str, str],
    frontend_contracts: dict[str, str],
    rows_sha256: str,
    metric_contract: dict[str, Any],
) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "schema_version": FROZEN_SCHEMA,
        "selection_policy": {
            "source": "full integer-count Pareto frontier",
            "roles": ["maximum_500ms_recovery_per_extra_false_cut"],
            "rate_budgets_are_elimination_caps": False,
        },
        "dev_manifest_sha256": manifest_sha256,
        "dev_rows_sha256": rows_sha256,
        "model_hashes": model_hashes,
        "frontend_contracts": frontend_contracts,
        "metric_contract": metric_contract,
        "panels": panels,
    }
    artifact["frozen_sha256"] = sha256_hex(artifact)
    return artifact
