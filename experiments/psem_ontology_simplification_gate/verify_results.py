from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Sequence

from experiments.psem_relative_occupancy_gate.io_utils import (
    canonical_sha256,
    load_json,
    load_jsonl,
    sha256_file,
    write_json,
)

PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]
PREDECESSOR_ROOT = REPOSITORY_ROOT / "experiments" / "psem_relative_occupancy_gate"
AUTHORITY_SNAPSHOT_PATH = REPOSITORY_ROOT / ".agents" / "goals" / "goal-issue-98" / "authority.snapshot.md"
CONFIG_PATH = PACKAGE_ROOT / "config.json"
EXPECTED_RESULTS_PATH = PACKAGE_ROOT / "expected_results_manifest.json"
MODEL_PATH = REPOSITORY_ROOT / "src" / "puripuly_heart" / "data" / "vad" / "silero_vad.onnx"
VAD_RUNNER_PATH = PACKAGE_ROOT / "run_production_vad.py"
VERIFIER_PATH = Path(__file__).resolve()
SAMPLE_RATE_HZ = 16000
RECEIPT_NAMES = {
    "streaming_sortformer": "sortformer_model_receipt.json",
    "ls_eend": "lseend_model_receipt.json",
}
FAMILIES = ("streaming_sortformer", "ls_eend")
ARMS = {
    "s1_oracle_anchor": "gate1_oracle_anchor",
    "s2_fixed_issue97_lifecycle": "gate2_causal_anchor",
}


class ResultVerificationError(RuntimeError):
    pass


def _close(left: Any, right: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(_close(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _close(left_value, right_value)
            for left_value, right_value in zip(left, right, strict=True)
        )
    if isinstance(left, float) or isinstance(right, float):
        return math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-8)
    return left == right


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _expected_result_hashes(role: str) -> dict[str, str]:
    manifest = load_json(EXPECTED_RESULTS_PATH)
    cfg = load_json(CONFIG_PATH)
    if (
        manifest.get("schema_version")
        != "psem.ontology_simplification.expected_results.v1"
        or manifest.get("authority_pin") != cfg["authority"]["sha256"]
        or manifest.get("config_sha256") != sha256_file(CONFIG_PATH)
        or manifest.get("evaluator_source_sha256")
        != sha256_file(PACKAGE_ROOT / "evaluate_simplified_ontologies.py")
        or manifest.get("production_vad_runner_source_sha256")
        != sha256_file(VAD_RUNNER_PATH)
        or manifest.get("verifier_source_sha256") != sha256_file(VERIFIER_PATH)
        or set(manifest.get("roles", {})) != {"dev", "eval"}
    ):
        raise ResultVerificationError("expected result manifest provenance mismatch")
    role_manifest = manifest["roles"][role]
    expected_role = "PSEM-STRATEGY-DEV" if role == "dev" else "PSEM-STRATEGY-EVAL"
    hashes = role_manifest.get("artifact_sha256")
    if (
        role_manifest.get("role") != expected_role
        or not isinstance(hashes, dict)
        or not hashes
        or any(not isinstance(name, str) or not _is_sha256(value) for name, value in hashes.items())
        or role_manifest.get("artifact_set_sha256") != canonical_sha256(hashes)
    ):
        raise ResultVerificationError("expected result manifest role contract mismatch")
    return dict(hashes)


def _verify_expected_artifact_hashes(
    *,
    result_dir: Path,
    paths: Sequence[Path],
    expected_hashes: dict[str, str],
    require_complete: bool,
) -> dict[str, str]:
    names = [value.relative_to(result_dir).as_posix() for value in paths]
    if len(names) != len(set(names)):
        raise ResultVerificationError("result artifact path coverage contains duplicates")
    if not set(names).issubset(expected_hashes) or (
        require_complete and set(names) != set(expected_hashes)
    ):
        raise ResultVerificationError("expected result artifact coverage mismatch")
    actual = {name: sha256_file(result_dir / name) for name in names}
    mismatched = [name for name, value in actual.items() if expected_hashes[name] != value]
    if mismatched:
        raise ResultVerificationError(
            f"expected result artifact digest mismatch: {sorted(mismatched)}"
        )
    return actual


def _report_number(value: float | int | None, precision: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):,.{precision}f}".rstrip("0").rstrip(".")


def _report_metric(value: float | int) -> str:
    return f"{float(value):,.1f}"


def _report_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.1f}%"


def _report_signed(value: float | int, precision: int = 1) -> str:
    return f"{float(value):+,.{precision}f}"


def _primary_frontier_row(
    rows: list[dict[str, Any]],
    cfg: dict[str, Any],
    family: str,
    arm: str,
    candidate: str,
    persistence: int,
) -> dict[str, Any]:
    primary = cfg["candidate_b"]["threshold_grids"][family]["primary"]
    matches = [
        row
        for row in rows
        if row["family"] == family
        and row["arm"] == arm
        and row["candidate"] == candidate
        and row["variant"] == "primary"
        and int(row["replacement_confirm_ms"]) == persistence
        and (
            candidate != "anchor_overlap"
            or (
                float(row["anchor_threshold"]) == float(primary[0])
                and float(row["overlap_threshold"]) == float(primary[1])
            )
        )
    ]
    if len(matches) != 1:
        raise ResultVerificationError("primary frontier report row coverage mismatch")
    return matches[0]


def _expected_provenance(role: str) -> dict[str, Any]:
    return {
        "authority_snapshot_sha256": sha256_file(AUTHORITY_SNAPSHOT_PATH),
        "config_sha256": sha256_file(CONFIG_PATH),
        "trace_reuse_receipt_sha256": sha256_file(PACKAGE_ROOT / "trace_reuse_receipt.json"),
        "causal_dependency_audit_sha256": sha256_file(
            PACKAGE_ROOT / "causal_lifecycle_dependency_audit.json"
        ),
        "predecessor_manifest_sha256": sha256_file(
            PREDECESSOR_ROOT / "results" / role / "relative_occupancy_manifest.jsonl"
        ),
        "predecessor_gate1_ledger_sha256": sha256_file(
            PREDECESSOR_ROOT / "results" / role / "gate1_event_ledger.jsonl"
        ),
        "predecessor_gate2_ledger_sha256": sha256_file(
            PREDECESSOR_ROOT / "results" / role / "gate2_event_ledger.jsonl"
        ),
        "predecessor_product_frontiers_sha256": sha256_file(
            PREDECESSOR_ROOT / "results" / role / "product_frontiers.json"
        ),
        "predecessor_model_receipt_sha256": {
            family: sha256_file(PREDECESSOR_ROOT / "results" / role / name)
            for family, name in RECEIPT_NAMES.items()
        },
        "evaluator_source_sha256": sha256_file(
            PACKAGE_ROOT / "evaluate_simplified_ontologies.py"
        ),
        "simple_anchor_source_sha256": sha256_file(PACKAGE_ROOT / "derive_simple_anchor.py"),
        "anchor_overlap_source_sha256": sha256_file(
            PACKAGE_ROOT / "derive_anchor_overlap.py"
        ),
    }


def verify(role: str, require_production_vad: bool) -> dict[str, Any]:
    result_dir = PACKAGE_ROOT / "results" / role
    required = [
        result_dir / "ontology_sufficiency.json" if role == "dev" else None,
        result_dir / "PATH_DECISION.md" if role == "eval" else None,
        result_dir / "anchor_dropout_slices.json",
        result_dir / "global_overlap_diagnostic.json",
        result_dir / "product_frontiers.json",
        result_dir / "paired_session_deltas.json",
        result_dir / "bootstrap_intervals.json",
        result_dir / "sortformer_simple_anchor_metrics.json",
        result_dir / "lseend_simple_anchor_metrics.json",
        result_dir / "sortformer_anchor_overlap_metrics.json",
        result_dir / "lseend_anchor_overlap_metrics.json",
    ]
    if require_production_vad:
        required.extend(
            [
                result_dir / "production_vad_speech_gate.jsonl",
                result_dir / "production_vad_replay_receipt.json",
                result_dir / "production_vad_sensitivity.json",
            ]
        )
    paths = [value for value in required if value is not None]
    missing = [str(value) for value in paths if not value.is_file()]
    if missing:
        raise ResultVerificationError(f"required result artifacts are missing: {missing}")
    expected_result_hashes = _expected_result_hashes(role)
    trace_receipt = load_json(PACKAGE_ROOT / "trace_reuse_receipt.json")
    if (
        trace_receipt["missing_required_neutral_fields"]
        or trace_receipt["new_model_inference_required"] is not False
        or trace_receipt["new_model_inference_performed"] is not False
    ):
        raise ResultVerificationError("trace reuse receipt contract mismatch")
    role_receipt = trace_receipt["roles"][role]
    for artifact in role_receipt["predecessor_artifacts"].values():
        path = REPOSITORY_ROOT / artifact["path"]
        if artifact["sha256"] != sha256_file(path):
            raise ResultVerificationError("trace reuse predecessor artifact mismatch")
    for family in FAMILIES:
        family_receipt = role_receipt["families"][family]
        if family_receipt["model_receipt_sha256"] != sha256_file(
            REPOSITORY_ROOT / family_receipt["model_receipt_path"]
        ):
            raise ResultVerificationError("trace reuse model receipt mismatch")
        for source in family_receipt["sources"]:
            if (
                source["trace_sha256"] != sha256_file(Path(source["trace_path"]))
                or source["trace_schema_version"] != "psem.relative_occupancy.trace.v1"
            ):
                raise ResultVerificationError("trace reuse source binding mismatch")
    audit = load_json(PACKAGE_ROOT / "causal_lifecycle_dependency_audit.json")
    if (
        audit["conclusion"] != "material_dependency_present"
        or audit["s2_label"] != "fixed-issue-97-lifecycle-counterfactual-ablation"
        or audit["native_simplified_ontology_runtime_claim_allowed"] is not False
    ):
        raise ResultVerificationError("causal dependency audit conclusion mismatch")
    if any(
        source["sha256"] != sha256_file(REPOSITORY_ROOT / source["path"])
        for source in audit["audited_sources"]
    ):
        raise ResultVerificationError("causal dependency audit source mismatch")
    if role == "dev":
        s0 = load_json(result_dir / "ontology_sufficiency.json")
        if not (
            s0["candidate_a_exact_action_equivalence"]
            and s0["candidate_b_exact_action_equivalence"]
            and all(int(value["mismatch_count"]) == 0 for value in s0["cells"])
        ):
            raise ResultVerificationError("S0 exact-action equivalence failed")
        expected_s0_provenance = {
            "authority_snapshot_sha256": sha256_file(AUTHORITY_SNAPSHOT_PATH),
            "config_sha256": sha256_file(CONFIG_PATH),
            "predecessor_manifest_sha256": sha256_file(
                PREDECESSOR_ROOT / "results" / "dev" / "relative_occupancy_manifest.jsonl"
            ),
            "predecessor_gate0_oracle_events_sha256": sha256_file(
                PREDECESSOR_ROOT / "results" / "dev" / "gate0_oracle_events.jsonl"
            ),
            "predecessor_gate0_oracle_metrics_sha256": sha256_file(
                PREDECESSOR_ROOT / "results" / "dev" / "gate0_oracle_metrics.json"
            ),
            "trace_reuse_receipt_sha256": sha256_file(
                PACKAGE_ROOT / "trace_reuse_receipt.json"
            ),
            "causal_dependency_audit_sha256": sha256_file(
                PACKAGE_ROOT / "causal_lifecycle_dependency_audit.json"
            ),
            "evaluator_source_sha256": sha256_file(
                PACKAGE_ROOT / "evaluate_simplified_ontologies.py"
            ),
            "simple_anchor_source_sha256": sha256_file(
                PACKAGE_ROOT / "derive_simple_anchor.py"
            ),
            "anchor_overlap_source_sha256": sha256_file(
                PACKAGE_ROOT / "derive_anchor_overlap.py"
            ),
        }
        if s0.get("provenance") != expected_s0_provenance:
            raise ResultVerificationError("S0 provenance mismatch")
        if any(
            int(value["candidate_a_mismatch_count"]) != 0
            or int(value["candidate_b_mismatch_count"]) != 0
            or int(value["candidate_a_event_count"]) != int(value["candidate_b_event_count"])
            for value in s0["cells"]
        ):
            raise ResultVerificationError("S0 independent challenger evidence mismatch")
    result_payload_paths = [
        result_dir / "anchor_dropout_slices.json",
        result_dir / "global_overlap_diagnostic.json",
        result_dir / "product_frontiers.json",
        result_dir / "paired_session_deltas.json",
        result_dir / "bootstrap_intervals.json",
        result_dir / "sortformer_simple_anchor_metrics.json",
        result_dir / "lseend_simple_anchor_metrics.json",
        result_dir / "sortformer_anchor_overlap_metrics.json",
        result_dir / "lseend_anchor_overlap_metrics.json",
    ]
    expected_provenance = _expected_provenance(role)
    for path in result_payload_paths:
        payload = load_json(path)
        if any(payload.get(key) != value for key, value in expected_provenance.items()):
            raise ResultVerificationError(f"result provenance mismatch: {path.name}")
    frontier = load_json(result_dir / "product_frontiers.json")["rows"]
    expected_sources = 10 if role == "dev" else 19
    if len(frontier) != 192:
        raise ResultVerificationError(f"unexpected product frontier row count: {len(frontier)}")
    if any(int(value["source_count"]) != expected_sources for value in frontier):
        raise ResultVerificationError("product frontier source coverage mismatch")
    predecessor = load_json(PREDECESSOR_ROOT / "results" / role / "product_frontiers.json")["rows"]
    comparison_count = 0
    for family in FAMILIES:
        for arm, predecessor_gate in ARMS.items():
            for persistence in (100, 200, 300, 500):
                current = next(
                    value
                    for value in frontier
                    if value["family"] == family
                    and value["arm"] == arm
                    and value["candidate"] == "r0_relative_occupancy"
                    and value["replacement_confirm_ms"] == persistence
                )
                previous = next(
                    value
                    for value in predecessor
                    if value["family"] == family
                    and value["gate"] == predecessor_gate
                    and value["replacement_confirm_ms"] == persistence
                )
                compared_fields = sorted(
                    set(current)
                    & set(previous)
                    - {"family", "arm", "gate", "candidate", "variant"}
                )
                for field in compared_fields:
                    if not _close(current[field], previous[field]):
                        raise ResultVerificationError(
                            f"R0 reconstruction mismatch: {family} {arm} {persistence} {field}"
                        )
                comparison_count += 1
    dropout = load_json(result_dir / "anchor_dropout_slices.json")
    overlap = load_json(result_dir / "global_overlap_diagnostic.json")
    for family in FAMILIES:
        if family not in dropout["families"] or family not in overlap["families"]:
            raise ResultVerificationError(f"diagnostic family coverage mismatch: {family}")
        for context in ("gt_anchor_only", "gt_anchor_overlap"):
            horizons = dropout["families"][family]["s1_oracle_anchor"][context]["horizons_ms"]
            if set(horizons) != {"100", "300", "500"}:
                raise ResultVerificationError("anchor dropout horizon coverage mismatch")
        mapping = dropout["families"][family]["oracle_anchor_mapping_coverage"]
        if (
            int(mapping["episode_count"])
            != int(mapping["mapped_episode_count"]) + int(mapping["unmapped_episode_count"])
            or int(mapping["unmapped_episode_count"]) <= 0
        ):
            raise ResultVerificationError("oracle anchor mapping coverage mismatch")
        coverage = overlap["families"][family]["coverage"]
        if (
            int(coverage["total_unmasked_cell_count"])
            != int(coverage["scored_cell_count"]) + int(coverage["invalid_cell_count"])
        ):
            raise ResultVerificationError("global-overlap invalid-cell coverage mismatch")
    deltas = load_json(result_dir / "paired_session_deltas.json")["rows"]
    intervals = load_json(result_dir / "bootstrap_intervals.json")["rows"]
    if len(deltas) != 2 * 4 * 9 * expected_sources:
        raise ResultVerificationError("paired session delta coverage mismatch")
    if len(intervals) != 2 * 4 * 9:
        raise ResultVerificationError("bootstrap interval coverage mismatch")
    cross_candidates = {
        value["candidate"]
        for value in intervals
        if value["comparison"] == "lseend_minus_streaming_sortformer"
    }
    if cross_candidates != {"r0_relative_occupancy", "simple_anchor", "anchor_overlap"}:
        raise ResultVerificationError("cross-family paired comparison coverage mismatch")
    production_rows = 0
    if require_production_vad:
        cfg = load_json(CONFIG_PATH)
        vad_cfg = cfg["speech_gate"]["production_vad"]
        receipt = load_json(result_dir / "production_vad_replay_receipt.json")
        gate_path = result_dir / "production_vad_speech_gate.jsonl"
        manifest_path = (
            PREDECESSOR_ROOT / "results" / role / "relative_occupancy_manifest.jsonl"
        )
        manifest_values = load_jsonl(manifest_path)
        manifest_ids = [str(value["source_id"]) for value in manifest_values]
        if len(manifest_ids) != len(set(manifest_ids)):
            raise ResultVerificationError("production VAD manifest contains duplicates")
        manifest_rows = {str(value["source_id"]): value for value in manifest_values}
        gate_rows = load_jsonl(gate_path)
        gate_ids = [str(value["source_id"]) for value in gate_rows]
        expected_role = "PSEM-STRATEGY-DEV" if role == "dev" else "PSEM-STRATEGY-EVAL"
        expected_model = {
            "version": vad_cfg["model_version"],
            "path": str(MODEL_PATH.relative_to(REPOSITORY_ROOT)),
            "sha256": vad_cfg["model_sha256"],
            "backend": vad_cfg["backend"],
        }
        if (
            receipt["schema_version"]
            != "psem.ontology_simplification.production_vad_receipt.v1"
            or receipt["role"] != expected_role
            or receipt["config_sha256"] != sha256_file(CONFIG_PATH)
            or Path(receipt["manifest_path"]) != manifest_path.relative_to(REPOSITORY_ROOT)
            or receipt["manifest_sha256"] != sha256_file(manifest_path)
            or int(receipt["source_count"]) != len(manifest_values)
            or receipt["source_ids_sha256"] != canonical_sha256(manifest_ids)
            or Path(receipt["speech_gate_path"])
            != gate_path.relative_to(REPOSITORY_ROOT)
            or receipt["speech_gate_sha256"] != sha256_file(gate_path)
            or receipt["row_payload_sha256"] != canonical_sha256(gate_rows)
            or receipt["runner_source_sha256"] != sha256_file(VAD_RUNNER_PATH)
            or receipt["profile"] != vad_cfg
            or any(receipt["model"].get(key) != value for key, value in expected_model.items())
            or receipt["model"].get("onnxruntime_version") in (None, "")
            or sha256_file(MODEL_PATH) != vad_cfg["model_sha256"]
            or "elapsed_seconds" in receipt
        ):
            raise ResultVerificationError("production VAD receipt contract mismatch")
        if gate_ids != manifest_ids or len(gate_ids) != len(set(gate_ids)):
            raise ResultVerificationError("production VAD row coverage mismatch")
        total_audio_samples = 0
        total_speech_samples = 0
        for row in gate_rows:
            manifest_row = manifest_rows[str(row["source_id"])]
            audio_path = Path(str(row["audio_path"]))
            spans = row["speech_spans"]
            previous_end = 0
            speech_samples = 0
            for span in spans:
                start = int(span["start_sample"])
                end = int(span["end_sample"])
                if start < previous_end or end <= start or end > int(row["audio_length_samples"]):
                    raise ResultVerificationError("production VAD speech span accounting mismatch")
                previous_end = end
                speech_samples += end - start
            if (
                row["schema_version"]
                != "psem.ontology_simplification.production_vad_source.v1"
                or audio_path.resolve() != Path(str(manifest_row["audio_path"])).resolve()
                or row["audio_sha256"] != manifest_row["waveform_sha256"]
                or row["audio_sha256"] != sha256_file(audio_path)
                or int(row["audio_size_bytes"]) != int(manifest_row["waveform_size_bytes"])
                or int(row["audio_size_bytes"]) != audio_path.stat().st_size
                or int(row["audio_length_samples"])
                != int(manifest_row["source_duration_samples"])
                or int(row["scored_start_sample"]) != int(manifest_row["scored_start_sample"])
                or int(row["scored_end_sample"]) != int(manifest_row["scored_end_sample"])
                or not 0
                <= int(row["scored_start_sample"])
                <= int(row["scored_end_sample"])
                <= int(row["audio_length_samples"])
                or int(row["processed_samples"]) != int(row["audio_length_samples"])
                or int(row["ignored_tail_samples"]) != 0
                or int(row["speech_span_count"]) != len(spans)
                or not math.isclose(
                    float(row["speech_seconds"]),
                    speech_samples / SAMPLE_RATE_HZ,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            ):
                raise ResultVerificationError("production VAD exact-audio replay mismatch")
            total_audio_samples += int(row["audio_length_samples"])
            total_speech_samples += speech_samples
        if not math.isclose(
            float(receipt["total_audio_seconds"]),
            total_audio_samples / SAMPLE_RATE_HZ,
            rel_tol=0.0,
            abs_tol=1e-9,
        ) or not math.isclose(
            float(receipt["total_vad_speech_seconds"]),
            total_speech_samples / SAMPLE_RATE_HZ,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ResultVerificationError("production VAD receipt totals mismatch")
        production = load_json(result_dir / "production_vad_sensitivity.json")
        production_rows = len(production["rows"])
        sensitivity_keys = {
            (
                row["family"],
                row["arm"],
                row["candidate"],
                int(row["replacement_confirm_ms"]),
            )
            for row in production["rows"]
        }
        expected_sensitivity_keys = {
            (family, arm, candidate, persistence)
            for family in FAMILIES
            for arm in ARMS
            for candidate in ("simple_anchor", "anchor_overlap")
            for persistence in (100, 200, 300, 500)
        }
        if (
            production["schema_version"]
            != "psem.ontology_simplification.production_vad_sensitivity.v1"
            or production["role"] != expected_role
            or production["config_sha256"] != sha256_file(CONFIG_PATH)
            or production["production_vad_receipt_sha256"]
            != sha256_file(result_dir / "production_vad_replay_receipt.json")
            or production["production_vad_speech_gate_sha256"] != sha256_file(gate_path)
            or production["production_readiness_claim"] is not False
            or sensitivity_keys != expected_sensitivity_keys
            or production_rows != len(expected_sensitivity_keys)
        ):
            raise ResultVerificationError("production VAD sensitivity contract mismatch")
    if role == "eval":
        report = (result_dir / "PATH_DECISION.md").read_text(encoding="utf-8")
        if "0.112" not in report or "0.108" in report:
            raise ResultVerificationError("PATH_DECISION global-overlap AUPRC mismatch")
        if any(f"\n{number}. **" not in report for number in range(1, 15)):
            raise ResultVerificationError("PATH_DECISION ordered answer coverage mismatch")
        sortformer_a0_300 = {
            row["arm"]: row["intervals"]
            for row in intervals
            if row["family"] == "streaming_sortformer"
            and row["comparison"] == "simple_anchor_minus_r0"
            and int(row["replacement_confirm_ms"]) == 300
        }
        if set(sortformer_a0_300) != set(ARMS):
            raise ResultVerificationError("PATH_DECISION bootstrap source coverage mismatch")
        s1_intervals = sortformer_a0_300["s1_oracle_anchor"]
        s2_intervals = sortformer_a0_300["s2_fixed_issue97_lifecycle"]
        report_intervals = (
            (
                s1_intervals["contamination_seconds_per_active_speech_hour_delta"],
                1,
            ),
            (s1_intervals["false_cut_count_per_session_delta"], 1),
            (s1_intervals["missed_replacement_count_per_session_delta"], 1),
            (s1_intervals["overlap_takeover_success_rate_delta"], 3),
            (
                s2_intervals["contamination_seconds_per_active_speech_hour_delta"],
                1,
            ),
            (s2_intervals["missed_replacement_count_per_session_delta"], 1),
            (s2_intervals["false_cut_count_per_session_delta"], 1),
        )
        expected_report_intervals = [
            f"[{interval['lower']:+.{precision}f}, {interval['upper']:+.{precision}f}]"
            for interval, precision in report_intervals
        ]
        if any(value not in report for value in expected_report_intervals):
            raise ResultVerificationError("PATH_DECISION bootstrap interval mismatch")
        family_labels = {
            "streaming_sortformer": "Sortformer",
            "ls_eend": "LS-EEND",
        }
        arm_labels = {
            "s1_oracle_anchor": "S1",
            "s2_fixed_issue97_lifecycle": "S2",
        }
        persistence_arm_labels = {
            "s1_oracle_anchor": "S1",
            "s2_fixed_issue97_lifecycle": "S2 fixed",
        }
        candidate_labels = {
            "r0_relative_occupancy": "R0",
            "simple_anchor": "A0",
            "anchor_overlap": "B0",
        }
        expected_product_rows = []
        for family, family_label in family_labels.items():
            for arm, arm_label in arm_labels.items():
                for candidate, candidate_label in candidate_labels.items():
                    row = _primary_frontier_row(
                        frontier, cfg, family, arm, candidate, 300
                    )
                    active_seconds = float(row["active_speech_hours"]) * 3600.0
                    unknown_denominator = active_seconds - float(
                        row["masked_active_speech_seconds"]
                    )
                    unknown_fraction = (
                        float(row["unanchored_active_speech_seconds"])
                        + float(row.get("anchor_uncertain_active_speech_seconds", 0.0))
                    ) / unknown_denominator
                    delay = row["replacement_emit_delay_ms"]
                    boundary = row["backdated_boundary_error_ms"]
                    takeover = row["topology"]["overlap_takeover"][
                        "overlap_takeover_success_rate"
                    ]
                    overlap_return = row["topology"]["overlap_return"][
                        "overlap_return_preservation_rate"
                    ]
                    wrong_anchor = "n/a"
                    if "wrong_anchor_rate" in row:
                        wrong_anchor = (
                            f"{_report_percent(row['wrong_anchor_rate'])} / "
                            f"{int(row['anchor_error_cascade_length']['maximum'])}"
                        )
                    cells = (
                        _report_metric(
                            row[
                                "exclusive_other_contamination_seconds_per_active_speech_hour"
                            ]
                        ),
                        _report_metric(
                            float(row["exclusive_other_contamination_upper_bound_seconds"])
                            / float(row["active_speech_hours"])
                        ),
                        _report_metric(row["speaker_induced_cut_count_per_active_speech_hour"]),
                        f"{int(row['false_cut_count']):,}",
                        f"{int(row['missed_replacement_count']):,}",
                        f"{_report_number(delay['p50'])}/{_report_number(delay['p90'])}",
                        f"{_report_number(boundary['p50'])}/{_report_number(boundary['p90'])}",
                        _report_percent(takeover),
                        _report_percent(overlap_return),
                        _report_percent(unknown_fraction),
                        wrong_anchor,
                    )
                    expected_product_rows.append(
                        f"| {family_label} {arm_label} {candidate_label} | "
                        + " | ".join(cells)
                        + " |"
                    )
                    persistence_cells = []
                    for persistence in (100, 200, 300, 500):
                        persistence_row = _primary_frontier_row(
                            frontier, cfg, family, arm, candidate, persistence
                        )
                        persistence_cells.append(
                            " / ".join(
                                (
                                    _report_metric(
                                        persistence_row[
                                            "exclusive_other_contamination_seconds_per_active_speech_hour"
                                        ]
                                    ),
                                    _report_metric(
                                        persistence_row[
                                            "speaker_induced_cut_count_per_active_speech_hour"
                                        ]
                                    ),
                                    f"{int(persistence_row['false_cut_count']):,}",
                                    f"{int(persistence_row['missed_replacement_count']):,}",
                                )
                            )
                        )
                    expected_product_rows.append(
                        f"| {family_label} {persistence_arm_labels[arm]} | "
                        f"{candidate_label} | "
                        + " | ".join(persistence_cells)
                        + " |"
                    )
        if any(value not in report for value in expected_product_rows):
            raise ResultVerificationError("PATH_DECISION product table mismatch")
        if require_production_vad:
            production_labels = (
                ("streaming_sortformer", "s1_oracle_anchor", "simple_anchor", "Sortformer S1 A0"),
                ("streaming_sortformer", "s1_oracle_anchor", "anchor_overlap", "Sortformer S1 B0"),
                (
                    "streaming_sortformer",
                    "s2_fixed_issue97_lifecycle",
                    "simple_anchor",
                    "Sortformer S2 fixed A0",
                ),
                (
                    "streaming_sortformer",
                    "s2_fixed_issue97_lifecycle",
                    "anchor_overlap",
                    "Sortformer S2 fixed B0",
                ),
                ("ls_eend", "s1_oracle_anchor", "simple_anchor", "LS-EEND S1 A0"),
                ("ls_eend", "s1_oracle_anchor", "anchor_overlap", "LS-EEND S1 B0"),
            )
            expected_production_rows = []
            for family, arm, candidate, label in production_labels:
                matches = [
                    row
                    for row in production["rows"]
                    if row["family"] == family
                    and row["arm"] == arm
                    and row["candidate"] == candidate
                    and int(row["replacement_confirm_ms"]) == 300
                ]
                if len(matches) != 1:
                    raise ResultVerificationError("production report row coverage mismatch")
                row = matches[0]
                gt = row["gt_speech_gate"]
                vad = row["production_vad"]
                delta = row["production_vad_minus_gt_speech_gate"]
                cells = (
                    _report_metric(
                        gt["exclusive_other_contamination_seconds_per_active_speech_hour"]
                    ),
                    _report_metric(
                        vad["exclusive_other_contamination_seconds_per_active_speech_hour"]
                    ),
                    _report_signed(
                        delta[
                            "exclusive_other_contamination_seconds_per_active_speech_hour"
                        ]
                    ),
                    _report_metric(vad["speaker_induced_cut_count_per_active_speech_hour"]),
                    _report_signed(
                        delta["speaker_induced_cut_count_per_active_speech_hour"]
                    ),
                    f"{int(delta['false_cut_count']):+,}",
                    f"{int(delta['missed_replacement_count']):+,}",
                )
                expected_production_rows.append(
                    f"| {label} | " + " | ".join(cells) + " |"
                )
            if any(value not in report for value in expected_production_rows):
                raise ResultVerificationError("PATH_DECISION production VAD table mismatch")
        path_dispositions = (
            "Path A, frozen Sortformer to scratch compact-student KD: **RED**",
            "Path B, Sortformer task adaptation before KD: **GREEN**",
            "Path C1, reuse this frozen LS-EEND checkpoint: **RED**",
            "Path C2, future attractor-family ideas: **YELLOW**",
        )
        if any(value not in report for value in path_dispositions):
            raise ResultVerificationError("PATH_DECISION path disposition mismatch")
    actual_artifact_hashes = _verify_expected_artifact_hashes(
        result_dir=result_dir,
        paths=paths,
        expected_hashes=expected_result_hashes,
        require_complete=require_production_vad,
    )
    output = {
        "schema_version": "psem.ontology_simplification.verification.v1",
        "role": "PSEM-STRATEGY-DEV" if role == "dev" else "PSEM-STRATEGY-EVAL",
        "passed": True,
        "source_count": expected_sources,
        "product_frontier_row_count": len(frontier),
        "r0_reconstruction_comparison_count": comparison_count,
        "paired_delta_row_count": len(deltas),
        "bootstrap_interval_row_count": len(intervals),
        "production_vad_sensitivity_row_count": production_rows,
        "expected_results_manifest_sha256": sha256_file(EXPECTED_RESULTS_PATH),
        "artifact_sha256": actual_artifact_hashes,
    }
    write_json(result_dir / "verification.json", output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("dev", "eval"), required=True)
    parser.add_argument("--require-production-vad", action="store_true")
    args = parser.parse_args()
    verify(args.role, args.require_production_vad)


if __name__ == "__main__":
    main()
