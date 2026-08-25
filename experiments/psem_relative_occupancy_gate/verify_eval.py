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
    load_json,
    safe_output_path,
    sha256_file,
    write_json,
)
from experiments.psem_relative_occupancy_gate.run_eval import run as run_eval


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


def run(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest).resolve()
    paths = {
        "metrics": Path(args.metrics).resolve(),
        "product": Path(args.product).resolve(),
        "topology": Path(args.topology).resolve(),
        "latency": Path(args.latency).resolve(),
    }
    expected_paths = {
        "metrics": manifest_path.parent / "eval_metrics.json",
        "product": manifest_path.parent / "product_frontiers.json",
        "topology": manifest_path.parent / "topology_slices.json",
        "latency": manifest_path.parent / "latency_breakdown.json",
    }
    output_path = Path(args.output).resolve()
    if paths != expected_paths or output_path != manifest_path.parent / "eval_verification.json":
        raise EvalVerificationError("EVAL verification paths are not canonical")
    artifacts = {name: _load_object(path) for name, path in paths.items()}
    selection = load_frozen_selection(Path(args.selection).resolve())
    validate_opened_eval_manifest(
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
    )
    with tempfile.TemporaryDirectory(prefix="psem-issue97-eval-verify-") as temporary:
        root = Path(temporary)
        regenerated = {
            "metrics": root / "eval_metrics.json",
            "product": root / "product_frontiers.json",
            "topology": root / "topology_slices.json",
            "latency": root / "latency_breakdown.json",
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
                independent_verification=True,
            )
        )
        for name in paths:
            if artifacts[name] != _load_object(regenerated[name]):
                raise EvalVerificationError(
                    f"independent deterministic EVAL regeneration mismatch: {name}"
                )
    receipt = {
        "schema_version": "psem.relative_occupancy.eval_verification.v1",
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
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
