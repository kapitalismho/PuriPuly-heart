from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from experiments.psem_frozen_ceiling_gate.build_ceiling_examples import (
    PACKAGE_ROOT,
    SessionExamples,
    config,
    load_sessions,
)
from experiments.psem_frozen_ceiling_gate.evaluate_ceiling import (
    RESULTS_ROOT,
    aggregate_conditions,
    session_row,
)
from experiments.psem_frozen_ceiling_gate.posterior_features import (
    TemporalContract,
    temporal_features,
)
from experiments.psem_frozen_ceiling_gate.run_posterior_probe import (
    LinearProbe,
    TinyMLPProbe,
    fit_linear,
    fit_mlp,
    fit_sanity,
    training_data,
)
from experiments.psem_relative_occupancy_gate.io_utils import (
    canonical_sha256,
    load_json,
    sha256_file,
    write_json,
)

REPRESENTATION_RECEIPT_PATH = PACKAGE_ROOT / "hidden_representation_receipt.json"
HIDDEN_CONFIG_PATH = PACKAGE_ROOT / "hidden_config.json"
EXTRACTION_RECEIPT_PATH = RESULTS_ROOT / "hidden_extraction_receipt.json"
SPLIT_PATH = PACKAGE_ROOT / "split_manifest.json"
CONFIG_PATH = PACKAGE_ROOT / "config.json"
MAPPING_PATH = PACKAGE_ROOT / "oracle_mapping_ledger.jsonl"
ACTION_REFERENCE_PATH = PACKAGE_ROOT / "action_reference_ledger.jsonl"
EXTRACTOR_PATH = PACKAGE_ROOT / "extract_hidden_features.py"


def _extracted_sources() -> dict[str, dict[str, Any]]:
    extraction = load_json(EXTRACTION_RECEIPT_PATH)
    representation = load_json(REPRESENTATION_RECEIPT_PATH)
    split = load_json(SPLIT_PATH)
    expected_sources = {str(value["source_id"]) for value in split["sources"]}
    expected_top_level = {
        "schema_version": "psem.hidden_ceiling.extraction_receipt.v1",
        "status": "complete",
        "representation_receipt_sha256": sha256_file(REPRESENTATION_RECEIPT_PATH),
        "hidden_config_sha256": sha256_file(HIDDEN_CONFIG_PATH),
        "split_manifest_sha256": sha256_file(SPLIT_PATH),
        "extractor_sha256": sha256_file(EXTRACTOR_PATH),
        "instrumented_bench_sha256": representation["runtime"]["instrumented_bench_sha256"],
        "model_sha256": representation["runtime"]["model_sha256"],
        "hidden_export_patch_sha256": representation["runtime"]["hidden_export_patch_sha256"],
    }
    if any(extraction.get(key) != value for key, value in expected_top_level.items()):
        raise ValueError("hidden extraction receipt contract differs")
    if extraction.get("posterior_equivalence", {}).get("status") != "equivalent":
        raise ValueError("hidden extraction posterior equivalence is not proven")
    result = {}
    for value in extraction["source_receipts"]:
        source_id = str(value["source_id"])
        if source_id in result:
            raise ValueError(f"duplicate hidden extraction source: {source_id}")
        if value.get("posterior_equivalence", {}).get("status") != "equivalent":
            raise ValueError(f"hidden source equivalence is not proven: {source_id}")
        receipt_path = Path(str(value["receipt_path"]))
        if sha256_file(receipt_path) != value["receipt_sha256"]:
            raise ValueError(f"hidden source receipt differs: {source_id}")
        receipt = load_json(receipt_path)
        contract = receipt.get("extraction_contract", {})
        if (
            receipt.get("schema_version") != "psem.hidden_ceiling.source_extraction.v1"
            or receipt.get("status") != "complete"
            or receipt.get("source_id") != source_id
            or receipt.get("extraction_contract_sha256") != value.get("extraction_contract_sha256")
            or canonical_sha256(contract) != receipt.get("extraction_contract_sha256")
            or contract.get("source_id") != source_id
            or contract.get("representation_receipt_sha256")
            != expected_top_level["representation_receipt_sha256"]
            or contract.get("hidden_config_sha256") != expected_top_level["hidden_config_sha256"]
            or contract.get("split_manifest_sha256") != expected_top_level["split_manifest_sha256"]
            or contract.get("extractor_sha256") != expected_top_level["extractor_sha256"]
            or contract.get("instrumented_bench_sha256")
            != expected_top_level["instrumented_bench_sha256"]
            or contract.get("model_sha256") != expected_top_level["model_sha256"]
            or contract.get("hidden_export_patch_sha256")
            != expected_top_level["hidden_export_patch_sha256"]
            or receipt.get("posterior_equivalence", {}).get("status") != "equivalent"
        ):
            raise ValueError(f"hidden source receipt contract differs: {source_id}")
        feature_path = Path(str(receipt["hidden_features_path"]))
        if sha256_file(feature_path) != receipt["hidden_features_sha256"]:
            raise ValueError(f"hidden feature identity differs: {source_id}")
        result[source_id] = receipt
    if set(result) != expected_sources or int(extraction.get("source_count", -1)) != len(
        expected_sources
    ):
        raise ValueError("hidden extraction source coverage differs from frozen split")
    return result


def _anchor_slot_one_hot(session: SessionExamples) -> np.ndarray:
    mapping = {
        str(value["anchor_episode_id"]): int(value["slot_index"])
        for value in session.mapping_records
        if value["status"] == "mapped"
    }
    slots = np.asarray([mapping[str(value)] for value in session.episode_ids], dtype=np.int64)
    if np.any((slots < 0) | (slots >= session.probabilities.shape[1])):
        raise ValueError(f"oracle slot index is invalid: {session.source_id}")
    one_hot = np.zeros((len(slots), session.probabilities.shape[1]), dtype=np.float32)
    one_hot[np.arange(len(slots)), slots] = 1.0
    return one_hot


def hidden_base(session: SessionExamples, receipt: dict[str, Any]) -> np.ndarray:
    feature_path = Path(str(receipt["hidden_features_path"]))
    with np.load(feature_path, allow_pickle=False) as extracted:
        hidden = np.asarray(extracted["hidden"], dtype=np.float32)
        starts = np.asarray(extracted["frame_start_samples"], dtype=np.int64)
        ends = np.asarray(extracted["frame_end_samples"], dtype=np.int64)
        frontiers = np.asarray(extracted["evidence_frontier_samples"], dtype=np.int64)
    centers = session.posterior_centers
    indices = np.searchsorted(ends, centers, side="right")
    if (
        np.any(indices >= len(ends))
        or np.any(starts[indices] > centers)
        or np.any(centers >= ends[indices])
        or np.any(session.frontiers != np.maximum(frontiers[indices], session.ends))
    ):
        raise ValueError(f"hidden frames do not align with posterior cells: {session.source_id}")
    selected = hidden[indices]
    return np.column_stack(
        (
            selected,
            _anchor_slot_one_hot(session),
            session.evidence_delay_ms / 1000.0,
            session.reset.astype(np.float32),
        )
    ).astype(np.float32, copy=False)


def _compact_source_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        metrics = row["metrics"]
        result.append(
            {
                **{
                    key: value
                    for key, value in row.items()
                    if key not in ("metrics", "diagnostics")
                },
                "metrics": {
                    key: metrics[key]
                    for key in (
                        "active_speech_seconds",
                        "predicted_cut_count",
                        "reference_replacement_count",
                        "matched_replacement_count",
                        "false_cut_count",
                        "missed_replacement_count",
                        "exclusive_other_contamination_seconds",
                    )
                },
                "diagnostics": row["diagnostics"],
            }
        )
    return result


def run() -> dict[str, Any]:
    cfg = config()
    hidden_cfg = load_json(HIDDEN_CONFIG_PATH)
    split_path = SPLIT_PATH
    split = load_json(split_path)
    folds = {str(value["held_out_family"]): value for value in split["folds"]}
    frozen_sources = {str(value["source_id"]): value for value in split["sources"]}
    receipts = _extracted_sources()
    sessions = load_sessions((500,))
    if {value.source_id for value in sessions} != set(receipts):
        raise ValueError("hidden extraction does not cover every frozen source")
    for session in sessions:
        frozen = frozen_sources[session.source_id]
        if frozen["source_family"] != session.source_family:
            raise ValueError(f"hidden split family differs: {session.source_id}")
    bases = {
        session.source_id: hidden_base(session, receipts[session.source_id]) for session in sessions
    }
    contract = TemporalContract(
        tuple(map(int, cfg["causal_lag_frames"])),
        tuple(map(int, cfg["noncausal_future_frames"])),
    )
    provenance = {
        "config_sha256": sha256_file(CONFIG_PATH),
        "hidden_config_sha256": sha256_file(HIDDEN_CONFIG_PATH),
        "split_manifest_sha256": sha256_file(split_path),
        "hidden_representation_receipt_sha256": sha256_file(REPRESENTATION_RECEIPT_PATH),
        "hidden_extraction_receipt_sha256": sha256_file(EXTRACTION_RECEIPT_PATH),
        "oracle_mapping_ledger_sha256": sha256_file(MAPPING_PATH),
        "action_reference_ledger_sha256": sha256_file(ACTION_REFERENCE_PATH),
    }
    models: dict[tuple[str, str, str], LinearProbe | TinyMLPProbe] = {}
    training_receipts = []
    source_rows = []
    scoring_sessions = [value for value in sessions if value.role == "eval"]
    conditions = (("H-C", False), ("H-NC", True))
    for condition_index, (condition, noncausal) in enumerate(conditions):
        matrices = {
            session.source_id: temporal_features(
                bases[session.source_id],
                session.episode_ids,
                contract,
                noncausal=noncausal,
            )
            for session in sessions
        }
        for fold_index, held_family in enumerate(cfg["split"]["families"]):
            fold = folds[held_family]
            train_ids = set(map(str, fold["training_sources"]))
            eval_ids = set(map(str, fold["evaluation_sources"]))
            train = [value for value in sessions if value.source_id in train_ids]
            if {value.source_id for value in train} != train_ids or train_ids & eval_ids:
                raise ValueError(f"hidden frozen split differs: {held_family}")
            seed = (
                int(cfg["training_seed"])
                + int(hidden_cfg["training_seed_offset"])
                + condition_index * 10
                + fold_index
            )
            train_x, train_y, train_w = training_data(train, matrices, cfg, seed)
            for probe_name in cfg["probe_classes"]:
                probe = (
                    fit_linear(train_x, train_y, train_w, cfg=cfg, seed=seed)
                    if probe_name == "linear"
                    else fit_mlp(train_x, train_y, train_w, cfg=cfg, seed=seed)
                )
                models[(condition, probe_name, held_family)] = probe
                training_receipts.append(
                    {
                        "condition": condition,
                        "probe_class": probe_name,
                        "held_out_family": held_family,
                        "train_source_ids": sorted(train_ids),
                        "eval_source_ids": sorted(eval_ids),
                        "future_context_ms": cfg["noncausal_horizon_ms"] if noncausal else 0,
                        "train_fit_sanity": fit_sanity(probe, train_x, train_y, train_w),
                    }
                )
        for persistence in map(int, cfg["probe_confirmation_ms"]):
            for session in scoring_sessions:
                if session.source_id not in set(folds[session.source_family]["evaluation_sources"]):
                    raise ValueError(
                        f"hidden source is outside its frozen fold: {session.source_id}"
                    )
                for probe_name in cfg["probe_classes"]:
                    scores = models[(condition, probe_name, session.source_family)].predict(
                        matrices[session.source_id]
                    )
                    for threshold in cfg["probe_thresholds"]:
                        source_rows.append(
                            session_row(
                                session,
                                scores,
                                condition=condition,
                                probe_class=probe_name,
                                threshold=float(threshold),
                                confirmation_ms=persistence,
                                time_condition="bounded_noncausal" if noncausal else "causal",
                                future_context_frames=max(contract.future_lags) if noncausal else 0,
                            )
                        )
    for condition, filename in (
        ("H-C", "hidden_causal_metrics.json"),
        ("H-NC", "hidden_noncausal_metrics.json"),
    ):
        chosen = [value for value in source_rows if value["condition"] == condition]
        write_json(
            RESULTS_ROOT / filename,
            {
                "schema_version": "psem.frozen_ceiling.hidden_metrics.v1",
                "condition": condition,
                "provenance": provenance,
                "rows": aggregate_conditions(chosen),
                "training_receipts": [
                    value for value in training_receipts if value["condition"] == condition
                ],
            },
        )
    write_json(
        RESULTS_ROOT / "hidden_source_family_results.json",
        {
            "schema_version": "psem.frozen_ceiling.hidden_source_results.v1",
            "provenance": provenance,
            "rows": _compact_source_rows(source_rows),
        },
    )
    return {
        "source_row_count": len(source_rows),
        "training_receipt_count": len(training_receipts),
    }


def main() -> None:
    print(run())


if __name__ == "__main__":
    main()
