from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from experiments.speaker_representation_scd.provenance import with_self_sha256
from experiments.speaker_representation_scd.r4_continuous import (
    _anchor_coordinates,
    _load_panel_sources,
)
from experiments.speaker_representation_scd.r5_data import (
    R4_POOLED_DIR,
    load_config,
    read_jsonl,
)
from experiments.speaker_representation_scd.r5_models import CausalTCN
from experiments.speaker_representation_scd.r5_scoring import (
    causal_match_events,
    detect_probability_events,
    event_metrics,
)
from experiments.speaker_representation_scd.r5_train import _predict

CHECKPOINT_DIR = Path("data/r5/checkpoints/b0")
SEQUENCE_DIR = Path("data/r5/legacy_common_gt/sequences")
OUTPUT_DIR = Path("manifests/r5/r5_b0_capability")
PROFILE_RULES = {
    "fast": {"tolerance_ms": 500, "false_events_per_hour_budget": 20.0},
    "balanced": {"tolerance_ms": 1000, "false_events_per_hour_budget": 5.0},
    "stable": {"tolerance_ms": 1500, "false_events_per_hour_budget": 1.0},
}


def _model(cache_root: Path, model_id: str, seed: int) -> CausalTCN:
    config = load_config()
    checkpoint = torch.load(
        cache_root / CHECKPOINT_DIR / model_id / f"seed_{seed}.pt",
        map_location="cpu",
        weights_only=False,
    )
    settings = config["tcn"]
    model = CausalTCN(
        int(checkpoint["input_dimension"]),
        int(settings["hidden_dimension"]),
        int(settings["kernel_size"]),
        [int(value) for value in settings["dilations"]],
        float(settings["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    return model.eval()


def _dev_records(cache_root: Path, model_id: str, model: CausalTCN) -> list[dict[str, Any]]:
    root = cache_root / SEQUENCE_DIR / model_id
    vectors = np.load(root / "vectors.npy", mmap_mode="r")
    records = []
    for row in read_jsonl(root / "index.jsonl"):
        if row["split"] != "dev":
            continue
        values = vectors[np.asarray(row["vector_rows"], dtype=np.int64)]
        records.append(
            {
                "frontiers": row["frontier_samples"],
                "ground_truth": [int(row["coordinate"])] if row["class"] == "positive" else [],
                "probabilities": _predict(model, values),
                "source_seconds": (
                    int(row["frontier_samples"][-1])
                    - int(row["frontier_samples"][0])
                    + 1600
                )
                / 16000,
            }
        )
    return records


def _r4_records(cache_root: Path, model_id: str, model: CausalTCN) -> list[dict[str, Any]]:
    vectors = np.load(
        cache_root / R4_POOLED_DIR / model_id / "vectors_300.npy",
        mmap_mode="r",
    )
    ground_truth = _anchor_coordinates(cache_root)
    sources, _ = _load_panel_sources(cache_root)
    source_map = {str(row["session_id"]): row for row in sources}
    records = []
    for row in read_jsonl(cache_root / R4_POOLED_DIR / model_id / "index_300.jsonl"):
        start = int(row["row_start"])
        count = int(row["row_count"])
        session_id = str(row["session_id"])
        source = source_map[session_id]
        records.append(
            {
                "frontiers": row["frontier_samples"],
                "ground_truth": [
                    int(value["coordinate"]) for value in ground_truth.get(session_id, [])
                ],
                "probabilities": _predict(model, vectors[start : start + count]),
                "source_seconds": (
                    int(source["eligible_end_sample"])
                    - int(source["eligible_start_sample"])
                )
                / 16000,
            }
        )
    return records


def _score_grid(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    config = load_config()
    rows = []
    source_hours = sum(float(record["source_seconds"]) for record in records) / 3600
    ground_truth_count = sum(len(record["ground_truth"]) for record in records)
    for threshold in config["sequence"]["thresholds"]:
        for confirmation in config["sequence"]["confirmation_hops"]:
            total_events = 0
            matched = []
            for record in records:
                events = detect_probability_events(
                    record["probabilities"],
                    record["frontiers"],
                    float(threshold),
                    int(confirmation),
                )
                matched.extend(
                    causal_match_events(
                        record["ground_truth"],
                        events,
                        tolerance_ms=1500,
                    )
                )
                total_events += len(events)
            rows.append(
                {
                    "config_id": f"threshold={float(threshold):.2f}|confirmation={int(confirmation)}",
                    "threshold": float(threshold),
                    "confirmation_hops": int(confirmation),
                    "metrics": event_metrics(
                        matched,
                        total_events,
                        ground_truth_count,
                        source_hours,
                    ),
                }
            )
    return rows


def _profile(rows: Sequence[dict[str, Any]], name: str) -> dict[str, Any]:
    rule = PROFILE_RULES[name]
    tolerance = int(rule["tolerance_ms"])
    budget = float(rule["false_events_per_hour_budget"])
    feasible = [
        row
        for row in rows
        if row["metrics"]["false_events_per_hour"] is not None
        and float(row["metrics"]["false_events_per_hour"]) <= budget
    ]

    def recall(row: dict[str, Any]) -> float:
        value = row["metrics"]["boundary_f1"][f"at_{tolerance}ms"]["recall"]
        return float(value or 0.0)

    def f1(row: dict[str, Any]) -> float:
        value = row["metrics"]["boundary_f1"][f"at_{tolerance}ms"]["f1"]
        return float(value or 0.0)

    def false_rate(row: dict[str, Any]) -> float:
        value = row["metrics"]["false_events_per_hour"]
        return float(value) if value is not None else 1e18

    candidates = feasible or list(rows)
    selected = min(
        candidates,
        key=lambda row: (
            -recall(row) if feasible else false_rate(row),
            -f1(row),
            float(row["metrics"]["availability_latency_ms"]["median_ms"] or 1e18),
            str(row["config_id"]),
        ),
    )
    return {
        "profile": name,
        "rule": rule,
        "budget_met": bool(feasible),
        "selected": selected,
    }


def run_capability(cache_root: Path, model_id: str, seed: int) -> Path:
    model = _model(cache_root, model_id, seed)
    dev_grid = _score_grid(_dev_records(cache_root, model_id, model))
    r4_grid = _score_grid(_r4_records(cache_root, model_id, model))
    dev_profiles = {name: _profile(dev_grid, name) for name in PROFILE_RULES}
    r4_by_id = {row["config_id"]: row for row in r4_grid}
    fixed_profiles = {
        name: {
            **profile,
            "r4_fixed_metrics": r4_by_id[profile["selected"]["config_id"]]["metrics"],
        }
        for name, profile in dev_profiles.items()
    }
    exploratory_profiles = {name: _profile(r4_grid, name) for name in PROFILE_RULES}
    document = with_self_sha256(
        {
            "schema_version": 1,
            "artifact_role": "r5_b0_product_capability_map",
            "claim_level": "internal_exploratory_policy_surface",
            "model_id": model_id,
            "seed": seed,
            "reporting_tolerances_ms": [100, 250, 500, 750, 1000, 1500],
            "profile_rules": PROFILE_RULES,
            "dev_selected_r4_fixed_profiles": fixed_profiles,
            "r4_exploratory_profiles": exploratory_profiles,
            "dev_grid": dev_grid,
            "r4_grid": r4_grid,
        }
    )
    path = cache_root / OUTPUT_DIR / f"{model_id}.seed_{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    cache_root = Path(os.environ["SRSCD_CACHE_ROOT"]).resolve()
    print(run_capability(cache_root, args.model_id, args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
