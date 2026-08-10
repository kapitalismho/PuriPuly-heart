from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from experiments.speaker_representation_scd.provenance import (
    load_json,
    sha256_file,
    with_self_sha256,
)
from experiments.speaker_representation_scd.r1_forecast import (
    MODEL_IDS,
    TECHNICAL_VALIDITY_PATH,
    _cache_root_size,
    external_validation_binding,
    validate_technical_validity,
)
from experiments.speaker_representation_scd.r1_gate import EXPERIMENT_ROOT
from experiments.speaker_representation_scd.r2l_gate import AUTHORITY
from experiments.speaker_representation_scd.run_provenance import run_provenance

FORECAST_CONTRACT_PATH = Path("configs/r1/full_job_forecast_contract.json")
LEDGER_RELATIVE_PATH = Path("manifests/r2/legacy_common_gt/coordinate_ledger.json")
FORECAST_RELATIVE_PATH = Path("manifests/r3/legacy_common_gt/r3_probe_forecast.json")
R3_TOTAL_EXPECTED = 33946


def forecast_provenance(requested_argv: tuple[str, ...]) -> dict[str, Any]:
    contract_path = EXPERIMENT_ROOT / FORECAST_CONTRACT_PATH
    contract = load_json(contract_path)
    calculator_path = Path(__file__).resolve()
    return {
        "authority": AUTHORITY,
        "forecast_contract": {
            "path": FORECAST_CONTRACT_PATH.as_posix(),
            "sha256": sha256_file(contract_path),
            "self_sha256": contract["self_sha256"],
        },
        "calculator": {
            "path": "r3_forecast.py",
            "sha256": sha256_file(calculator_path),
        },
        "execution_identity": {
            "run_id": uuid4().hex,
            "process_id": os.getpid(),
            "started_at_utc": datetime.now(UTC).isoformat(),
        },
        "run_provenance": run_provenance(
            EXPERIMENT_ROOT.parents[1],
            requested_argv,
            deterministic_seed=0,
            deterministic_kernels=False,
        ),
    }


def build_r3_forecast(
    technical: dict[str, Any],
    contract: dict[str, Any],
    ledger: dict[str, Any],
    cache_root: Path,
    provenance: dict[str, Any],
    input_errors: list[str] | None = None,
    supervision_binding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    blockers = list(input_errors or [])
    if ledger.get("schema_version") != 1 or ledger.get("artifact_role") != "r2l_legacy_common_gt_coordinate_ledger":
        blockers.append("coordinate_ledger: unexpected identity")
    r3 = ledger.get("r3")
    if not isinstance(r3, dict):
        blockers.append("coordinate_ledger: r3 missing")
    else:
        primary = int(r3.get("primary_window_count", 0))
        trajectory = int(r3.get("trajectory_window_count", 0))
        if primary + trajectory != R3_TOTAL_EXPECTED:
            blockers.append(f"coordinate_ledger: R3 window total differs: {primary + trajectory}")
    if r3 and int(r3.get("positive_anchor_count", 0)) != 450 or r3 and int(r3.get("negative_anchor_count", 0)) != 360:
        blockers.append("coordinate_ledger: R3 anchor counts differ")
    blockers.extend(validate_technical_validity(technical, cache_root))
    current_root_bytes, root_errors = _cache_root_size(cache_root)
    blockers.extend(root_errors)
    if blockers:
        return with_self_sha256(
            {
                "schema_version": 1,
                "artifact_role": "r3_probe_forecast",
                "experiment_id": "speaker_representation_scd_v1",
                "status": "not_ready",
                "blockers": sorted(set(blockers)),
                "forecast_provenance": provenance,
                "forecast_approved": False,
            }
        )
    method = contract["forecast_method"]
    upper_factor = int(method["runtime_upper_bound_factor"])
    safety = float(method["runtime_safety_multiplier"])
    ceilings = contract["ceilings"]
    storage = {row["model_id"]: row for row in contract["model_storage_contracts"]}
    smoke = {row["model_id"]: row for row in technical["model_smoke_reports"]}
    r3_primary = int(r3["primary_window_count"])
    r3_trajectory = int(r3["trajectory_window_count"])
    r3_total = r3_primary + r3_trajectory
    layer_count = len(storage["mhubert-147"]["retained_layer_ids"])
    models: list[dict[str, Any]] = []
    for model_id in MODEL_IDS:
        single = smoke[model_id]["single_seconds_per_window"]
        batched = smoke[model_id]["batch_seconds_per_window"]
        cold = smoke[model_id]["cold_load_seconds"]
        rows = {
            "model_id": model_id,
            "inference_window_count": r3_total,
            "layer_or_tap_count": layer_count,
            "measured_balanced_seconds_per_window": single,
            "measured_batch_seconds_per_window": batched,
            "conservative_wall_hours": (r3_total * single * upper_factor * safety + cold) / 3600,
            "batched_wall_hours": (r3_total * batched * upper_factor * safety + cold) / 3600,
            "cold_load_seconds": cold,
            "authoritative_peak_job_memory_bytes": smoke[model_id][
                "authoritative_peak_job_memory_bytes"
            ],
            "pooled_cache_bytes": r3_total
            * layer_count
            * storage[model_id]["pooled_dimension_per_layer"]
            * storage[model_id]["dtype_bytes"],
        }
        models.append(rows)
    total_batched = sum(row["batched_wall_hours"] for row in models)
    total_conservative = sum(row["conservative_wall_hours"] for row in models)
    derived = sum(row["pooled_cache_bytes"] for row in models)
    checks = {
        "per_model_wall_hours_batched": all(
            row["batched_wall_hours"] <= ceilings["max_per_model_wall_hours"] for row in models
        ),
        "per_model_wall_hours_conservative": all(
            row["conservative_wall_hours"] <= ceilings["max_per_model_wall_hours"]
            for row in models
        ),
        "total_wall_hours_batched": total_batched <= ceilings["max_total_wall_hours"],
        "total_wall_hours_conservative": total_conservative <= ceilings["max_total_wall_hours"],
        "derived_cache_gib": derived / 1024**3 <= ceilings["max_derived_cache_gib"],
        "peak_memory": all(
            row["authoritative_peak_job_memory_bytes"]
            <= ceilings["max_resident_ram_gib"] * 1024**3
            for row in models
        ),
        "external_storage_gib": (current_root_bytes + derived) / 1024**3
        <= ceilings["max_external_storage_gib"],
    }
    return with_self_sha256(
        {
            "schema_version": 1,
            "artifact_role": "r3_probe_forecast",
            "experiment_id": "speaker_representation_scd_v1",
            "authority": AUTHORITY,
            "scope": "legacy-common-gt-v1",
            "status": "ceiling_pass_candidate" if all(checks.values()) else "ceiling_failed",
            "forecast_provenance": provenance,
            "supervision_binding": supervision_binding,
            "external_validation_binding": external_validation_binding(technical),
            "coordinate_ledger": {
                "relative_to_cache_root": LEDGER_RELATIVE_PATH.as_posix(),
                "sha256": sha256_file(cache_root / LEDGER_RELATIVE_PATH),
                "self_sha256": ledger["self_sha256"],
            },
            "current_external_root_bytes": current_root_bytes,
            "r3": {
                "primary_window_count": r3_primary,
                "trajectory_window_count": r3_trajectory,
                "total_window_count": r3_total,
                "models": models,
                "total_batched_wall_hours": total_batched,
                "total_conservative_wall_hours": total_conservative,
                "derived_pooled_cache_gib": derived / 1024**3,
            },
            "ceiling_checks": checks,
            "forecast_approved": False,
        }
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    args = parser.parse_args(argv)
    cache_root = args.cache_root.resolve()
    technical = load_json(EXPERIMENT_ROOT / TECHNICAL_VALIDITY_PATH)
    contract = load_json(EXPERIMENT_ROOT / FORECAST_CONTRACT_PATH)
    ledger_path = cache_root / LEDGER_RELATIVE_PATH
    if not ledger_path.is_file():
        ledger = {}
        ledger_errors = [f"coordinate_ledger: missing: {LEDGER_RELATIVE_PATH}"]
    else:
        try:
            ledger = load_json(ledger_path)
            ledger_errors = []
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            ledger = {}
            ledger_errors = [f"coordinate_ledger: unreadable: {exc}"]
    forecast = build_r3_forecast(
        technical,
        contract,
        ledger,
        cache_root,
        forecast_provenance(tuple(sys.argv) if argv is None else ("r3_forecast", *argv)),
        ledger_errors,
    )
    output = cache_root / FORECAST_RELATIVE_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(forecast, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(forecast, indent=2, sort_keys=True))
    return 0 if forecast["status"].startswith("ceiling_pass_candidate") else 2


if __name__ == "__main__":
    raise SystemExit(main())
