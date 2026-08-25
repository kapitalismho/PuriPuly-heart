from __future__ import annotations

import argparse
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.psem_relative_occupancy_gate.contracts import ActivityInterval
from experiments.psem_relative_occupancy_gate.derive_relative_occupancy import derive_rows
from experiments.psem_relative_occupancy_gate.io_utils import (
    CONFIG_PATH,
    PACKAGE_ROOT,
    canonical_sha256,
    config,
    load_json,
    load_jsonl,
    safe_output_path,
    sha256_file,
    write_json,
    write_jsonl,
)
from experiments.psem_relative_occupancy_gate.provenance import load_frozen_dataset
from experiments.psem_relative_occupancy_gate.run_gate0 import (
    CONTRACT_ARTIFACTS,
    PRIMARY_TOPOLOGIES,
    _validate_manifest_bindings,
    run_gate0,
)


class Gate0VerificationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Gate0VerificationError(message)


def _semantic_metrics(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    for field in (
        "content_sha256",
        "manifest_path",
        "preflight_path",
        "topology_examples_path",
        "oracle_events_path",
    ):
        result.pop(field, None)
    return result


def verify_gate0(output_dir: Path) -> dict[str, Any]:
    output_dir = safe_output_path(output_dir)
    cfg = config()
    metrics_path = output_dir / "gate0_oracle_metrics.json"
    events_path = output_dir / "gate0_oracle_events.jsonl"
    examples_path = output_dir / "gate0_topology_examples.jsonl"
    preflight_path = output_dir / "preflight_receipt.json"
    manifest_path = output_dir / "relative_occupancy_manifest.jsonl"
    result_path = output_dir / "GATE0_ONTOLOGY_RESULT.md"
    for path in (
        metrics_path,
        events_path,
        examples_path,
        preflight_path,
        manifest_path,
        result_path,
    ):
        _require(path.is_file(), f"missing Gate 0 artifact: {path}")
    preflight = load_json(preflight_path)
    manifest = load_jsonl(manifest_path)
    dataset_binding = _validate_manifest_bindings(manifest, preflight)
    dataset = load_frozen_dataset()
    expected_source_ids = set(dataset.source_ids("PSEM-STRATEGY-DEV"))
    _require(
        {str(row["source_id"]) for row in manifest} == expected_source_ids,
        "Gate 0 manifest source identities differ from frozen DEV",
    )
    regenerated_rows = derive_rows(
        corpus=Path(str(preflight["paths"]["corpus_root"])),
        reference=Path(str(preflight["paths"]["reference_root"])),
        roles=["PSEM-STRATEGY-DEV"],
        frozen_selection=None,
    )
    _require(
        canonical_sha256(regenerated_rows) == canonical_sha256(manifest),
        "derived manifest is not reproducible from frozen V2 inputs",
    )
    metrics = load_json(metrics_path)
    stored_content_sha256 = str(metrics.get("content_sha256"))
    metrics_payload = dict(metrics)
    metrics_payload.pop("content_sha256", None)
    _require(
        stored_content_sha256 == canonical_sha256(metrics_payload),
        "metrics self-hash mismatch",
    )
    _require(metrics.get("passed") is True, "Gate 0 metrics did not pass")
    _require(metrics.get("role") == "PSEM-STRATEGY-DEV", "Gate 0 metrics role changed")
    _require(metrics.get("eval_status") == "sealed", "Gate 0 metrics did not seal EVAL")
    _require(
        metrics.get("frozen_dataset") == dataset_binding,
        "Gate 0 metrics dataset binding changed",
    )
    _require(
        metrics.get("manifest_sha256") == sha256_file(manifest_path),
        "manifest hash mismatch",
    )
    _require(
        metrics.get("preflight_sha256") == sha256_file(preflight_path),
        "preflight hash mismatch",
    )
    _require(
        metrics.get("oracle_events_sha256") == sha256_file(events_path),
        "event hash mismatch",
    )
    _require(
        metrics.get("topology_examples_sha256") == sha256_file(examples_path),
        "topology example hash mismatch",
    )
    _require(
        set(metrics.get("contract_artifacts", {})) == set(CONTRACT_ARTIFACTS),
        "load-bearing contract artifact set changed",
    )
    for name in CONTRACT_ARTIFACTS:
        _require(
            metrics["contract_artifacts"][name] == sha256_file(PACKAGE_ROOT / name),
            f"contract artifact changed: {name}",
        )
    expected_ms = [int(value) for value in cfg["replacement_confirm_ms"]]
    settings = metrics.get("settings")
    _require(isinstance(settings, list), "Gate 0 settings are missing")
    _require(
        [int(row["confirmation_ms"]) for row in settings] == expected_ms,
        "duration grid changed",
    )
    events = load_jsonl(events_path)
    event_counts = Counter(int(row["confirmation_ms"]) for row in events)
    intervals_by_source = {
        str(row["source_id"]): tuple(
            ActivityInterval.from_dict(value) for value in row["intervals"]
        )
        for row in manifest
    }
    masked_intervals = [
        interval
        for intervals in intervals_by_source.values()
        for interval in intervals
        if interval.masked
    ]
    event_boundaries_inside_mask = 0
    for row in events:
        _require(
            str(row.get("source_id")) in expected_source_ids,
            "event source is outside frozen DEV",
        )
        _require(
            int(row["boundary_source_sample"])
            < int(row["model_evidence_frontier_sample"])
            == int(row["decoder_emit_sample"]),
            "Gate 0 event timing is not exact",
        )
        _require(
            row.get("schema_version") == "psem.relative_occupancy.gate0_event.v1",
            "event schema changed",
        )
        if any(
            interval.start_sample <= int(row["boundary_source_sample"]) < interval.end_sample
            and interval.masked
            for interval in intervals_by_source[str(row["source_id"])]
        ):
            event_boundaries_inside_mask += 1
    _require(
        event_boundaries_inside_mask == 0,
        "a replacement boundary falls inside a frozen V2 mask",
    )
    masked_seconds = 0.0
    for setting in settings:
        duration = int(setting["confirmation_ms"])
        aggregate = setting["aggregate"]
        sources = setting["sources"]
        _require(
            {str(row["source_id"]) for row in sources} == expected_source_ids,
            f"setting source set differs from frozen DEV: {duration}",
        )
        _require(
            event_counts[duration] == int(aggregate["speaker_induced_cut_count"]),
            f"event count does not match aggregate: {duration}",
        )
        _require(
            aggregate["boundary_backdating_exact"] is True,
            f"independent backdating audit failed: {duration}",
        )
        for source in sources:
            _require(
                source["boundary_audit"]["passed"] is True
                and source["boundary_audit"]["errors"] == [],
                f"source boundary audit failed: {source['source_id']}:{duration}",
            )
            _require(
                float(source["exclusive_other_contamination_upper_bound_seconds"])
                >= float(source["exclusive_other_contamination_seconds"]),
                f"fail-closed exposure bound regressed: {source['source_id']}:{duration}",
            )
            masked_seconds += float(source["exposure"]["masked_seconds"])
    _require(masked_seconds > 0.0, "frozen V2 masks were not represented in Gate 0")
    examples = load_jsonl(examples_path)
    synthetic_counts = Counter(
        (str(row["name"]), int(row["confirmation_samples"]))
        for row in examples
        if row.get("kind") == "synthetic_fixture"
    )
    natural_rows = [row for row in examples if row.get("kind") == "natural_v2_dev"]
    natural_coverage = {str(row["primary_topology"]) for row in natural_rows}
    _require(
        all(value == 1 for value in synthetic_counts.values()),
        "synthetic fixtures are duplicated",
    )
    _require(
        len(synthetic_counts) == 10 * len(expected_ms),
        "synthetic fixture grid is incomplete",
    )
    _require(
        all(
            row["boundary_audit"]["passed"] is True
            for row in examples
            if row.get("kind") == "synthetic_fixture"
        ),
        "synthetic boundary audit failed",
    )
    _require(
        natural_coverage == set(PRIMARY_TOPOLOGIES),
        "natural topology coverage is incomplete",
    )
    _require(
        all(
            row["exact_replacement_timing"] is True and row["boundary_audit"]["passed"] is True
            for row in natural_rows
        ),
        "natural topology timing or boundary audit failed",
    )
    _require(
        metrics.get("synthetic_fixture_failures") == [],
        "synthetic fixture failures remain",
    )
    _require(
        metrics.get("natural_topology_failures") == [],
        "natural topology failures remain",
    )
    _require(
        "Result: **PASS**" in result_path.read_text(encoding="utf-8"),
        "result statement did not pass",
    )
    with tempfile.TemporaryDirectory(prefix="issue97-gate0-replay-") as directory:
        replay_dir = Path(directory)
        replay_manifest = replay_dir / "relative_occupancy_manifest.jsonl"
        write_jsonl(replay_manifest, regenerated_rows)
        replay_metrics = run_gate0(replay_manifest, preflight_path, replay_dir)
        _require(
            events_path.read_bytes() == (replay_dir / "gate0_oracle_events.jsonl").read_bytes(),
            "independent Gate 0 replay changed the event ledger",
        )
        _require(
            examples_path.read_bytes()
            == (replay_dir / "gate0_topology_examples.jsonl").read_bytes(),
            "independent Gate 0 replay changed the topology ledger",
        )
        _require(
            result_path.read_bytes() == (replay_dir / "GATE0_ONTOLOGY_RESULT.md").read_bytes(),
            "independent Gate 0 replay changed the result statement",
        )
        _require(
            canonical_sha256(_semantic_metrics(metrics))
            == canonical_sha256(_semantic_metrics(replay_metrics)),
            "independent Gate 0 replay changed semantic metrics",
        )
        replay = {
            "manifest_sha256": sha256_file(replay_manifest),
            "events_sha256": sha256_file(replay_dir / "gate0_oracle_events.jsonl"),
            "examples_sha256": sha256_file(replay_dir / "gate0_topology_examples.jsonl"),
            "result_sha256": sha256_file(replay_dir / "GATE0_ONTOLOGY_RESULT.md"),
        }
    receipt = {
        "schema_version": "psem.relative_occupancy.gate0_verification.v1",
        "config_sha256": sha256_file(CONFIG_PATH),
        "contract_artifacts": {
            name: sha256_file(PACKAGE_ROOT / name) for name in CONTRACT_ARTIFACTS
        },
        "frozen_dataset": dataset_binding,
        "metrics_sha256": sha256_file(metrics_path),
        "events_sha256": sha256_file(events_path),
        "examples_sha256": sha256_file(examples_path),
        "manifest_sha256": sha256_file(manifest_path),
        "preflight_sha256": sha256_file(preflight_path),
        "source_count": len(manifest),
        "source_ids": sorted(expected_source_ids),
        "event_count": len(events),
        "decoder_settings_ms": expected_ms,
        "natural_topologies": sorted(natural_coverage),
        "masked_seconds_across_settings": masked_seconds,
        "masked_interval_count": len(masked_intervals),
        "masked_source_seconds": sum(
            interval.end_sample - interval.start_sample for interval in masked_intervals
        )
        / int(cfg["dataset"]["sample_rate_hz"]),
        "event_boundaries_inside_mask": event_boundaries_inside_mask,
        "independent_replay": replay,
        "eval_status": "sealed",
        "passed": True,
    }
    receipt["content_sha256"] = canonical_sha256(receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    receipt = verify_gate0(args.output_dir)
    if args.receipt:
        write_json(safe_output_path(args.receipt), receipt)
    print(f"Gate 0 verification passed for {receipt['source_count']} DEV sources with EVAL sealed")


if __name__ == "__main__":
    main()
