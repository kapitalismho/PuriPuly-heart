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
    self_sha256_valid,
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
from experiments.speaker_representation_scd.run_provenance import run_provenance

AUTHORITY = {
    "path": "experiments/speaker_representation_scd/EXPERIMENT_PLAN.en.md",
    "sha256": "bf1727a62dcd9c8c28cc095c46ebeaaab8e3f723a0c1c6440ed19a9968d590be",
}

FORECAST_CONTRACT_PATH = Path("configs/r1/full_job_forecast_contract.json")
LEDGER_RELATIVE_PATH = Path("manifests/r2/legacy_common_gt/coordinate_ledger.json")
REDUCED_FORECAST_RELATIVE_PATH = Path("manifests/r2/legacy_common_gt/reduced_r3_r4_forecast.json")
VALIDATION_RECEIPT_RELATIVE_PATH = Path(
    "manifests/r2/legacy_common_gt/validation_receipt.json"
)
CONTEXTS_MS = (100, 300, 500)
PRIMARY_CONTEXT_MS = 300
HOP_SAMPLES = 1600
SENSITIVITY_HOP_SAMPLES = 800
TRAJECTORY_OFFSETS_MS = (
    -1000, -750, -500, -300, -200, -100, 0, 100, 200, 300, 500, 750, 1000, 1500, 2000,
)


def _coordinate_windows(start: int, end: int, context_ms: int, hop: int) -> int:
    first = start + context_ms * 16
    if end < first:
        return 0
    return (end - first) // hop + 1


def _ledger_errors(ledger: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if ledger.get("schema_version") != 1:
        errors.append("coordinate_ledger: unexpected schema version")
    if ledger.get("artifact_role") != "r2l_legacy_common_gt_coordinate_ledger":
        errors.append("coordinate_ledger: unexpected artifact role")
    if ledger.get("experiment_id") != "speaker_representation_scd_v1":
        errors.append("coordinate_ledger: unexpected experiment")
    if ledger.get("authority") != AUTHORITY:
        errors.append("coordinate_ledger: authority differs")
    if ledger.get("scope") != "legacy-common-gt-v1":
        errors.append("coordinate_ledger: scope differs")
    if not self_sha256_valid(ledger):
        errors.append("coordinate_ledger: invalid self hash")
    r3 = ledger.get("r3")
    if not isinstance(r3, dict):
        errors.append("coordinate_ledger: r3 missing")
        r3 = {}
    if r3.get("positive_anchor_count") != 450 or r3.get("negative_anchor_count") != 360:
        errors.append("coordinate_ledger: r3 anchor counts differ")
    if not isinstance(r3.get("primary_window_count"), int) or r3["primary_window_count"] <= 0:
        errors.append("coordinate_ledger: r3 primary windows invalid")
    if not isinstance(r3.get("trajectory_window_count"), int) or r3[
        "trajectory_window_count"
    ] < 0:
        errors.append("coordinate_ledger: r3 trajectory windows invalid")
    r4 = ledger.get("r4")
    if not isinstance(r4, dict):
        errors.append("coordinate_ledger: r4 missing")
        r4 = {}
    if not isinstance(r4.get("panel_source_count"), int) or r4["panel_source_count"] <= 0:
        errors.append("coordinate_ledger: r4 panel sources invalid")
    if not isinstance(r4.get("panel_total_source_hours"), (int, float)) or r4[
        "panel_total_source_hours"
    ] <= 0:
        errors.append("coordinate_ledger: r4 panel hours invalid")
    if r4.get("panel_total_source_hours") > 6.0:
        errors.append("coordinate_ledger: r4 panel exceeds six source hours")
    sources = r4.get("panel_sources")
    if not isinstance(sources, list) or not sources:
        errors.append("coordinate_ledger: r4 panel sources missing")
        sources = []
    for index, row in enumerate(sources):
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("session_id"), str)
            or not isinstance(row.get("eligible_start_sample"), int)
            or not isinstance(row.get("eligible_end_sample"), int)
            or row["eligible_end_sample"] <= row["eligible_start_sample"]
            or "synthetic_manifest" not in row
            or "stratum" not in row
        ):
            errors.append(f"coordinate_ledger: r4.panel_sources[{index}] invalid")
    counts = r4.get("windows_by_context_ms")
    if (
        not isinstance(counts, dict)
        or set(counts) != {str(value) for value in CONTEXTS_MS}
        or any(not isinstance(value, int) or value <= 0 for value in counts.values())
    ):
        errors.append("coordinate_ledger: r4 window counts invalid")
    return errors


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
            "path": "r2l_forecast.py",
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


def _forecast_provenance_errors(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return ["forecast_provenance: missing"]
    errors: list[str] = []
    contract_path = EXPERIMENT_ROOT / FORECAST_CONTRACT_PATH
    contract = load_json(contract_path)
    if value.get("authority") != AUTHORITY:
        errors.append("forecast_provenance: authority differs")
    if value.get("forecast_contract") != {
        "path": FORECAST_CONTRACT_PATH.as_posix(),
        "sha256": sha256_file(contract_path),
        "self_sha256": contract["self_sha256"],
    }:
        errors.append("forecast_provenance: contract identity differs")
    if value.get("calculator") != {
        "path": "r2l_forecast.py",
        "sha256": sha256_file(Path(__file__).resolve()),
    }:
        errors.append("forecast_provenance: calculator identity differs")
    execution = value.get("execution_identity")
    if (
        not isinstance(execution, dict)
        or not isinstance(execution.get("run_id"), str)
        or len(execution["run_id"]) != 32
        or any(character not in "0123456789abcdef" for character in execution["run_id"])
        or not isinstance(execution.get("process_id"), int)
        or not isinstance(execution.get("started_at_utc"), str)
    ):
        errors.append("forecast_provenance: execution identity invalid")
    run = value.get("run_provenance")
    if (
        not isinstance(run, dict)
        or not isinstance(run.get("git_commit"), str)
        or len(run["git_commit"]) != 40
        or not isinstance(run.get("git_dirty"), bool)
        or not isinstance(run.get("git_status_porcelain"), list)
        or not isinstance(run.get("requested_argv"), list)
    ):
        errors.append("forecast_provenance: Git/run identity invalid")
    return errors


def build_reduced_forecast(
    technical: dict[str, Any],
    contract: dict[str, Any],
    ledger: dict[str, Any],
    cache_root: Path,
    provenance: dict[str, Any],
    input_errors: list[str] | None = None,
    supervision_binding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    blockers = list(input_errors or [])
    blockers.extend(_ledger_errors(ledger))
    blockers.extend(_forecast_provenance_errors(provenance))
    technical_errors = validate_technical_validity(technical, cache_root)
    blockers.extend(technical_errors)
    current_root_bytes, root_errors = _cache_root_size(cache_root)
    blockers.extend(root_errors)
    if blockers:
        return with_self_sha256(
            {
                "schema_version": 1,
                "artifact_role": "r2l_reduced_r3_r4_forecast",
                "experiment_id": "speaker_representation_scd_v1",
                "status": "not_ready",
                "blockers": sorted(set(blockers)),
                "forecast_provenance": provenance,
                "external_validation_binding": external_validation_binding(technical)
                if not technical_errors
                else None,
                "current_external_root_bytes": current_root_bytes,
                "forecast_approved": False,
                "full_extraction_enabled": False,
            }
        )
    method = contract["forecast_method"]
    upper_factor = int(method["runtime_upper_bound_factor"])
    safety = float(method["runtime_safety_multiplier"])
    ceilings = contract["ceilings"]
    storage = {row["model_id"]: row for row in contract["model_storage_contracts"]}
    smoke = {row["model_id"]: row for row in technical["model_smoke_reports"]}

    r3_primary = int(ledger["r3"]["primary_window_count"])
    r3_trajectory = int(ledger["r3"]["trajectory_window_count"])
    r3_total = r3_primary + r3_trajectory
    r4_panel = ledger["r4"]
    r4_windows_primary_context = int(r4_panel["windows_by_context_ms"][str(PRIMARY_CONTEXT_MS)])
    sources = r4_panel["panel_sources"]
    event_stratum = [row for row in sources if row.get("synthetic_manifest") is not None]

    def source_windows(row: dict[str, Any], context_ms: int, hop: int) -> int:
        return _coordinate_windows(
            int(row["eligible_start_sample"]), int(row["eligible_end_sample"]), context_ms, hop
        )

    sensitivity_full = sum(
        source_windows(row, PRIMARY_CONTEXT_MS, SENSITIVITY_HOP_SAMPLES) for row in sources
    )
    sensitivity_event = sum(
        source_windows(row, PRIMARY_CONTEXT_MS, SENSITIVITY_HOP_SAMPLES)
        for row in event_stratum
    )

    def rows(
        model_id: str, windows: int, prefix: str, layer_count: int
    ) -> dict[str, Any]:
        single = smoke[model_id]["single_seconds_per_window"]
        batched = smoke[model_id]["batch_seconds_per_window"]
        cold = smoke[model_id]["cold_load_seconds"]
        worst = single * upper_factor * safety
        worst_batch = batched * upper_factor * safety
        return {
            "model_id": model_id,
            "inference_window_count": windows,
            "measured_balanced_seconds_per_window": single,
            "measured_batch_seconds_per_window": batched,
            "conservative_seconds_per_window": worst,
            "batched_seconds_per_window": worst_batch,
            "conservative_wall_hours": (windows * worst + cold) / 3600,
            "batched_wall_hours": (windows * worst_batch + cold) / 3600,
            "cold_load_seconds": cold,
            "authoritative_peak_job_memory_bytes": smoke[model_id][
                "authoritative_peak_job_memory_bytes"
            ],
            "pooled_cache_bytes": windows
            * layer_count
            * storage[model_id]["pooled_dimension_per_layer"]
            * storage[model_id]["dtype_bytes"],
        }

    r3_layer_count = len(storage["mhubert-147"]["retained_layer_ids"])
    r3_models = [rows(model_id, r3_total, "r3", r3_layer_count) for model_id in MODEL_IDS]
    r4_models = [rows(model_id, r4_windows_primary_context, "r4", 1) for model_id in MODEL_IDS]
    top_two_models = sorted(
        MODEL_IDS,
        key=lambda model_id: smoke[model_id]["single_seconds_per_window"],
        reverse=True,
    )[:2]
    sensitivity_models = [
        rows(model_id, sensitivity_full, "sensitivity", 1) for model_id in top_two_models
    ]
    sensitivity_event_models = [
        rows(model_id, sensitivity_event, "sensitivity_event", 1)
        for model_id in top_two_models
    ]

    def total_wall(model_rows: list[dict[str, Any]], basis: str) -> float:
        return sum(
            row["conservative_wall_hours"] if basis == "conservative" else row["batched_wall_hours"]
            for row in model_rows
        )

    def peak_check(model_rows: list[dict[str, Any]]) -> bool:
        return all(
            row["authoritative_peak_job_memory_bytes"]
            <= ceilings["max_resident_ram_gib"] * 1024**3
            for row in model_rows
        )

    def ceiling_checks(basis: str) -> dict[str, Any]:
        per_model = all(
            (
                row["conservative_wall_hours"]
                if basis == "conservative"
                else row["batched_wall_hours"]
            )
            <= ceilings["max_per_model_wall_hours"]
            for row in r3_models + r4_models + sensitivity_models
        )
        total = (
            total_wall(r3_models, basis)
            + total_wall(r4_models, basis)
            + total_wall(sensitivity_models, basis)
        )
        derived = (
            sum(row["pooled_cache_bytes"] for row in r3_models + r4_models)
            + sum(row["pooled_cache_bytes"] for row in sensitivity_models)
        )
        return {
            "checks": {
                "per_model_wall_hours": per_model,
                "total_wall_hours": total <= ceilings["max_total_wall_hours"],
                "derived_cache_gib": derived / 1024**3 <= ceilings["max_derived_cache_gib"],
                "peak_memory": peak_check(r3_models + r4_models + sensitivity_models),
                "external_storage_gib": (current_root_bytes + derived) / 1024**3
                <= ceilings["max_external_storage_gib"],
            },
            "values": {
                "total_wall_hours": total,
                "derived_cache_gib": derived / 1024**3,
                "external_storage_gib": (current_root_bytes + derived) / 1024**3,
            },
        }

    checks_conservative = ceiling_checks("conservative")
    checks_batched = ceiling_checks("batched")

    def bounded_panel(budget_seconds: float, basis_worst: float) -> tuple[int, list[str], float, int]:
        bounded_windows = (
            max(0, int(budget_seconds / basis_worst)) if basis_worst else 0
        )
        bounded_sources: list[str] = []
        bounded_seconds = 0.0
        bounded_window_count = 0
        for row in sources:
            seconds = (int(row["eligible_end_sample"]) - int(row["eligible_start_sample"])) / 16000
            window_count = _coordinate_windows(
                int(row["eligible_start_sample"]),
                int(row["eligible_end_sample"]),
                PRIMARY_CONTEXT_MS,
                HOP_SAMPLES,
            )
            if bounded_window_count + window_count > bounded_windows:
                break
            bounded_sources.append(str(row["session_id"]))
            bounded_seconds += seconds
            bounded_window_count += window_count
        return bounded_windows, bounded_sources, bounded_seconds, bounded_window_count

    r3_batched_total = total_wall(r3_models, "batched")
    r3_conservative_total = total_wall(r3_models, "conservative")
    sensitivity_event_batched_total = total_wall(sensitivity_event_models, "batched")
    sensitivity_event_conservative_total = total_wall(sensitivity_event_models, "conservative")
    sum_batched_worst = sum(
        smoke[model_id]["batch_seconds_per_window"] * upper_factor * safety
        for model_id in MODEL_IDS
    )
    sum_conservative_worst = sum(
        smoke[model_id]["single_seconds_per_window"] * upper_factor * safety
        for model_id in MODEL_IDS
    )
    budget_batched = (
        ceilings["max_total_wall_hours"] * 3600
        - r3_batched_total * 3600
        - sensitivity_event_batched_total * 3600
    )
    budget_conservative = (
        ceilings["max_total_wall_hours"] * 3600
        - r3_conservative_total * 3600
        - sensitivity_event_conservative_total * 3600
    )
    (
        bounded_windows_batched,
        bounded_sources_batched,
        bounded_seconds_batched,
        bounded_window_count_batched,
    ) = bounded_panel(budget_batched, sum_batched_worst)
    (
        bounded_windows_conservative,
        bounded_sources_conservative,
        bounded_seconds_conservative,
        bounded_window_count_conservative,
    ) = bounded_panel(budget_conservative, sum_conservative_worst)
    bounded_models_batched = [
        rows(model_id, bounded_window_count_batched, "r4_bounded", 1)
        for model_id in MODEL_IDS
    ]
    bounded_models_conservative = [
        rows(model_id, bounded_window_count_conservative, "r4_bounded", 1)
        for model_id in MODEL_IDS
    ]
    bounded_total_batched = (
        total_wall(bounded_models_batched, "batched")
        + r3_batched_total
        + sensitivity_event_batched_total
    )
    bounded_total_conservative = (
        total_wall(bounded_models_conservative, "conservative")
        + r3_conservative_total
        + sensitivity_event_conservative_total
    )

    conservative_pass = all(checks_conservative["checks"].values())
    batched_pass = all(checks_batched["checks"].values())
    status = (
        "ceiling_pass_candidate_conservative"
        if conservative_pass
        else "ceiling_pass_candidate_batched"
        if batched_pass
        else "ceiling_failed_with_bounded_variant"
    )
    return with_self_sha256(
        {
            "schema_version": 1,
            "artifact_role": "r2l_reduced_r3_r4_forecast",
            "experiment_id": "speaker_representation_scd_v1",
            "authority": AUTHORITY,
            "status": status,
            "scope": "legacy-common-gt-v1",
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
                "primary_only_batched_wall_hours": sum(
                    rows(model_id, r3_primary, "r3", r3_layer_count)["batched_wall_hours"]
                    for model_id in MODEL_IDS
                ),
                "primary_only_conservative_wall_hours": sum(
                    rows(model_id, r3_primary, "r3", r3_layer_count)["conservative_wall_hours"]
                    for model_id in MODEL_IDS
                ),
                "models": r3_models,
            },
            "r4": {
                "primary_context_ms": PRIMARY_CONTEXT_MS,
                "primary_hop_ms": 100,
                "panel_source_count": r4_panel["panel_source_count"],
                "panel_total_source_hours": r4_panel["panel_total_source_hours"],
                "windows_by_context_ms": {
                    str(context): int(r4_panel["windows_by_context_ms"][str(context)])
                    for context in CONTEXTS_MS
                },
                "primary_window_count": r4_windows_primary_context,
                "models": r4_models,
            },
            "sensitivity": {
                "hop_ms": 50,
                "top_two_model_ids": list(top_two_models),
                "full_panel_window_count": sensitivity_full,
                "event_stratum_window_count": sensitivity_event,
                "full_panel_models": sensitivity_models,
                "event_stratum_models": sensitivity_event_models,
            },
            "ceiling_checks_conservative": checks_conservative,
            "ceiling_checks_batched": checks_batched,
            "bounded_variant": {
                "basis": "batched",
                "r4_windows": bounded_window_count_batched,
                "r4_source_count": len(bounded_sources_batched),
                "r4_source_hours": round(bounded_seconds_batched / 3600, 6),
                "r4_batched_wall_hours": total_wall(bounded_models_batched, "batched"),
                "r4_conservative_wall_hours": total_wall(bounded_models_batched, "conservative"),
                "r3_batched_wall_hours": r3_batched_total,
                "sensitivity_event_stratum_batched_wall_hours": sensitivity_event_batched_total,
                "total_batched_wall_hours_with_r3_and_sensitivity": bounded_total_batched,
                "included_session_ids": bounded_sources_batched,
                "rule": "prefix_of_frozen_panel_order_fitting_batched_24h_budget",
                "r4_windows_conservative_basis": bounded_window_count_conservative,
                "r4_source_hours_conservative_basis": round(
                    bounded_seconds_conservative / 3600, 6
                ),
                "total_conservative_wall_hours_with_r3_and_sensitivity": bounded_total_conservative,
                "included_session_ids_conservative_basis": bounded_sources_conservative,
            },
            "forecast_approved": False,
            "full_extraction_enabled": False,
        }
    )


def _load_ledger(cache_root: Path) -> tuple[dict[str, Any], list[str]]:
    path = cache_root / LEDGER_RELATIVE_PATH
    if not path.is_file():
        return {}, [f"coordinate_ledger: missing: {LEDGER_RELATIVE_PATH}"]
    try:
        ledger = load_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {}, [f"coordinate_ledger: unreadable: {exc}"]
    return ledger, []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    args = parser.parse_args(argv)
    cache_root = args.cache_root.resolve()
    technical = load_json(EXPERIMENT_ROOT / TECHNICAL_VALIDITY_PATH)
    contract = load_json(EXPERIMENT_ROOT / FORECAST_CONTRACT_PATH)
    ledger, ledger_errors = _load_ledger(cache_root)
    forecast = build_reduced_forecast(
        technical,
        contract,
        ledger,
        cache_root,
        forecast_provenance(tuple(sys.argv) if argv is None else ("r2l_forecast", *argv)),
        ledger_errors,
    )
    print(json.dumps(forecast, indent=2, sort_keys=True))
    return 0 if forecast["status"].startswith("ceiling_pass_candidate") else 2


if __name__ == "__main__":
    raise SystemExit(main())
