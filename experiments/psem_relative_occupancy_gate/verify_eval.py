from __future__ import annotations

import argparse
import tempfile
from argparse import Namespace
from pathlib import Path
from typing import Any

from experiments.psem_relative_occupancy_gate.eval_access import (
    load_frozen_selection,
    validate_opened_eval_manifest,
)
from experiments.psem_relative_occupancy_gate.io_utils import (
    config,
    load_json,
    load_jsonl,
    safe_output_path,
    sha256_file,
    write_json,
)
from experiments.psem_relative_occupancy_gate.run_eval import (
    EVENT_COMPUTE_LAG_UNAVAILABLE_REASON,
)
from experiments.psem_relative_occupancy_gate.run_eval import (
    run as run_eval,
)
from experiments.psem_relative_occupancy_gate.verify_model_gates import (
    ModelGateVerificationError,
    _group_event_ledger,
    _validate_aggregate_rows,
)


class EvalVerificationError(RuntimeError):
    pass


def _load_object(path: Path) -> dict[str, Any]:
    value = load_json(path)
    if not isinstance(value, dict):
        raise EvalVerificationError(f"EVAL artifact is not an object: {path}")
    return value


def _validate_semantics(
    metrics: dict[str, Any],
    product: dict[str, Any],
    topology: dict[str, Any],
    latency: dict[str, Any],
    selection: dict[str, Any],
    ledger_paths: dict[str, Path],
) -> None:
    expected_families = {"streaming_sortformer", "ls_eend"}
    if set(metrics.get("families", {})) != expected_families:
        raise EvalVerificationError("EVAL family coverage mismatch")
    for family in expected_families:
        if metrics["families"][family]["selected_settings"] != selection[
            "selected_settings"
        ][family]:
            raise EvalVerificationError(f"EVAL settings drifted: {family}")
    rows = product.get("rows", [])
    gates = [str(value.get("gate")) for value in rows]
    expected_counts = {
        "vad_only_no_speaker_cut": 1,
        "gate0_oracle": 4,
        "gate1_oracle_anchor": 8,
        "gate2_causal_anchor": 8,
    }
    if {key: gates.count(key) for key in expected_counts} != expected_counts:
        raise EvalVerificationError("EVAL product frontier coverage mismatch")
    if len(topology.get("rows", [])) != 16:
        raise EvalVerificationError("EVAL topology coverage mismatch")
    if set(latency.get("families", {})) != expected_families:
        raise EvalVerificationError("EVAL latency coverage mismatch")
    for artifact in (metrics, product, topology, latency):
        if (
            artifact.get("role") != "PSEM-STRATEGY-EVAL"
            or artifact.get("eval_status") != "opened_once"
            or artifact.get("selection_sha256") != selection["selection_sha256"]
        ):
            raise EvalVerificationError("EVAL artifact role/selection binding mismatch")
        if (
            artifact.get("gate1_event_ledger_sha256")
            != sha256_file(ledger_paths["gate1_events"])
            or artifact.get("gate2_event_ledger_sha256")
            != sha256_file(ledger_paths["gate2_events"])
        ):
            raise EvalVerificationError("EVAL event-ledger hash binding mismatch")


def _validate_event_ledgers(
    *,
    metrics: dict[str, Any],
    product: dict[str, Any],
    topology: dict[str, Any],
    manifest_rows: list[dict[str, Any]],
    gate1_rows: list[dict[str, Any]],
    gate2_rows: list[dict[str, Any]],
    selection: dict[str, Any],
) -> None:
    manifest = {str(row["source_id"]): row for row in manifest_rows}
    manifest_sha256 = str(metrics["manifest_sha256"])
    receipt_hashes = {
        family: str(value["model_receipt"]["receipt_sha256"])
        for family, value in metrics["families"].items()
    }
    for row in [*gate1_rows, *gate2_rows]:
        family = str(row.get("family", ""))
        lag = row.get("event_compute_lag")
        exposure = row.get("fail_closed_exposure")
        if (
            row.get("role") != "PSEM-STRATEGY-EVAL"
            or row.get("eval_status") != "opened_once"
            or row.get("manifest_sha256") != manifest_sha256
            or row.get("selection_sha256") != selection["selection_sha256"]
            or row.get("model_receipt_sha256") != receipt_hashes.get(family)
            or not isinstance(exposure, dict)
            or "exclusive_other_contamination_seconds" not in exposure
            or not isinstance(lag, dict)
            or lag
            != {
                "availability": "unavailable_at_event_level",
                "reason": EVENT_COMPUTE_LAG_UNAVAILABLE_REASON,
                "model_receipt_sha256": receipt_hashes.get(family),
                "aggregate_runtime_artifact": "latency_breakdown.json",
            }
        ):
            raise EvalVerificationError("EVAL event-ledger provenance mismatch")
    if any("expected_opportunities" not in row for row in gate2_rows):
        raise EvalVerificationError("EVAL causal opportunity ledger is missing")
    cfg = config()
    try:
        gate1_grouped = _group_event_ledger(
            gate1_rows,
            gate="gate1_oracle_anchor",
            manifest=manifest,
            cfg=cfg,
        )
        gate2_grouped = _group_event_ledger(
            gate2_rows,
            gate="gate2_causal_anchor",
            manifest=manifest,
            cfg=cfg,
        )
        tolerance_samples = int(cfg["product_event_alignment_tolerance_ms"]) * 16
        _validate_aggregate_rows(
            grouped=gate1_grouped,
            product_rows=product["rows"],
            topology_rows=topology["rows"],
            manifest=manifest,
            tolerance_samples=tolerance_samples,
        )
        _validate_aggregate_rows(
            grouped=gate2_grouped,
            product_rows=product["rows"],
            topology_rows=topology["rows"],
            manifest=manifest,
            tolerance_samples=tolerance_samples,
        )
    except ModelGateVerificationError as exc:
        raise EvalVerificationError(str(exc)) from exc


def run(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest).resolve()
    paths = {
        "metrics": Path(args.metrics).resolve(),
        "product": Path(args.product).resolve(),
        "topology": Path(args.topology).resolve(),
        "latency": Path(args.latency).resolve(),
        "gate1_events": Path(args.gate1_events).resolve(),
        "gate2_events": Path(args.gate2_events).resolve(),
    }
    expected_paths = {
        "metrics": manifest_path.parent / "eval_metrics.json",
        "product": manifest_path.parent / "product_frontiers.json",
        "topology": manifest_path.parent / "topology_slices.json",
        "latency": manifest_path.parent / "latency_breakdown.json",
        "gate1_events": manifest_path.parent / "gate1_event_ledger.jsonl",
        "gate2_events": manifest_path.parent / "gate2_event_ledger.jsonl",
    }
    output_path = Path(args.output).resolve()
    if paths != expected_paths or output_path != manifest_path.parent / "eval_verification.json":
        raise EvalVerificationError("EVAL verification paths are not canonical")
    artifacts = {
        name: _load_object(path)
        for name, path in paths.items()
        if name not in {"gate1_events", "gate2_events"}
    }
    gate1_rows = load_jsonl(paths["gate1_events"])
    gate2_rows = load_jsonl(paths["gate2_events"])
    selection = load_frozen_selection(Path(args.selection).resolve())
    manifest_rows, _ = validate_opened_eval_manifest(
        manifest_path=Path(args.manifest).resolve(),
        access_path=Path(args.access_receipt).resolve(),
        selection_path=Path(args.selection).resolve(),
        authorization_path=Path(args.eval_authorization).resolve(),
    )
    _validate_semantics(
        artifacts["metrics"],
        artifacts["product"],
        artifacts["topology"],
        artifacts["latency"],
        selection,
        {
            "gate1_events": paths["gate1_events"],
            "gate2_events": paths["gate2_events"],
        },
    )
    _validate_event_ledgers(
        metrics=artifacts["metrics"],
        product=artifacts["product"],
        topology=artifacts["topology"],
        manifest_rows=manifest_rows,
        gate1_rows=gate1_rows,
        gate2_rows=gate2_rows,
        selection=selection,
    )
    with tempfile.TemporaryDirectory(prefix="psem-issue97-eval-verify-") as temporary:
        root = Path(temporary)
        regenerated = {
            "metrics": root / "eval_metrics.json",
            "product": root / "product_frontiers.json",
            "topology": root / "topology_slices.json",
            "latency": root / "latency_breakdown.json",
            "gate1_events": root / "gate1_event_ledger.jsonl",
            "gate2_events": root / "gate2_event_ledger.jsonl",
        }
        run_eval(
            Namespace(
                manifest=args.manifest,
                access_receipt=args.access_receipt,
                selection=args.selection,
                eval_authorization=args.eval_authorization,
                sortformer_receipt=args.sortformer_receipt,
                lseend_receipt=args.lseend_receipt,
                output=str(regenerated["metrics"]),
                product_output=str(regenerated["product"]),
                topology_output=str(regenerated["topology"]),
                latency_output=str(regenerated["latency"]),
                gate1_event_output=str(regenerated["gate1_events"]),
                gate2_event_output=str(regenerated["gate2_events"]),
                independent_verification=True,
            )
        )
        for name in {"metrics", "product", "topology", "latency"}:
            if artifacts[name] != _load_object(regenerated[name]):
                raise EvalVerificationError(
                    f"independent deterministic EVAL regeneration mismatch: {name}"
                )
        for name, rows in (
            ("gate1_events", gate1_rows),
            ("gate2_events", gate2_rows),
        ):
            if rows != load_jsonl(regenerated[name]):
                raise EvalVerificationError(
                    f"independent deterministic EVAL regeneration mismatch: {name}"
                )
    receipt = {
        "schema_version": "psem.relative_occupancy.eval_verification.v2",
        "role": "PSEM-STRATEGY-EVAL",
        "eval_status": "opened_once",
        "passed": True,
        "independent_regeneration": True,
        "selection_sha256": selection["selection_sha256"],
        "artifact_sha256": {
            name: sha256_file(path) for name, path in sorted(paths.items())
        },
    }
    write_json(safe_output_path(output_path), receipt)
    print({"output": str(output_path), "passed": True})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--access-receipt", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--eval-authorization", required=True)
    parser.add_argument("--sortformer-receipt", required=True)
    parser.add_argument("--lseend-receipt", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--product", required=True)
    parser.add_argument("--topology", required=True)
    parser.add_argument("--latency", required=True)
    parser.add_argument("--gate1-events", required=True)
    parser.add_argument("--gate2-events", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
