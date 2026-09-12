from __future__ import annotations

import hashlib
import json
import mmap
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
BASE = ROOT / "experiments/psem_r2_policy"
ATTEMPT = BASE / "artifacts/dev-supervised/cases/ES2009d/attempt-1"
PREFLIGHT_PATH = BASE / "artifacts/dev-supervised/preflight/20260912T074459Z/preflight.json"
MANIFEST_PATH = Path("C:/tmp/psem-u8-inputs/input-integrity-manifest.json")
RAW_PATH = BASE / "artifacts/dev/ES2009d/20260912T084839024097Z.json"
PARENT_ID = "c06fe9cd-e419-47bd-8427-066896b59e18"
OUTPUT = HERE / "safety-stop-projection.json"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_parent() -> dict:
    marker = f'  {{\n   "index": 340,\n   "parent_id": "{PARENT_ID}"'.encode()
    with RAW_PATH.open("rb") as handle:
        mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            start = mapped.find(marker)
            if start < 0:
                raise RuntimeError("pinned safety parent record not found")
            text = mapped[start + 2 : start + 15_000_000].decode("utf-8")
            record, _ = json.JSONDecoder().raw_decode(text)
            if record.get("parent_id") != PARENT_ID:
                raise RuntimeError("decoded the wrong parent record")
            return record
        finally:
            mapped.close()


def slim_arm(arm: dict) -> dict:
    fields = (
        "meeting",
        "cluster_id",
        "conservation",
        "fragmentation",
        "contamination",
        "guard",
        "strata",
        "primary_stratum",
        "sequential_target",
        "latency",
        "late_operations",
        "n_late_rejected",
        "eligible",
        "translated",
        "child_ids",
        "child_groups",
        "child_texts",
        "skipped_reason",
    )
    return {field: arm[field] for field in fields if field in arm}


def main() -> None:
    summary = load(ATTEMPT / "summary.json")
    pacing = load(ATTEMPT / "pacing.json")
    outcome = load(ATTEMPT / "outcome.json")
    preflight = load(PREFLIGHT_PATH)
    paid_gate = load(ATTEMPT / "gate.json")
    capsule_precheck = load(ATTEMPT / "capsule-precheck.json")
    prelaunch = load(ATTEMPT / "prelaunch.json")
    manifest = load(MANIFEST_PATH)
    meeting_input = next(row for row in manifest["dev"] if row["meeting"] == "ES2009d")
    parent = decode_parent()
    r0_attribution = parent["r0"]["attribution"]
    r2_attribution = parent["r2"]["attribution"]
    token_24 = next(row for row in parent["tokens"] if row["token_id"] == 24)
    attribution_24 = next(row for row in r0_attribution if row["token_id"] == 24)
    r0_reference_unit = parent["guard"]["arm_witnesses"]["r0"]["units"][0]
    r2_reference_unit = next(
        row for row in parent["guard"]["arm_witnesses"]["r2"]["units"] if 24 in row["wrong_token_ids"]
    )
    derived_names = (
        "summary.json",
        "summary-print.json",
        "pacing.json",
        "outcome.json",
        "canonical_case.json",
        "gate.json",
        "capsule-precheck.json",
        "prelaunch.json",
    )
    attribution_bytes = json.dumps(
        r0_attribution, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    projection = {
        "schema": "PSEM-R2-DEV-SAFETY-STOP-PROJECTION-1",
        "status": "STOPPED_ON_GENUINE_SAFETY_GATE",
        "projection_builder": {
            "path": str(Path(__file__).resolve()),
            "sha256": file_sha256(Path(__file__).resolve()),
        },
        "policy": {
            "protocol_revision": summary["protocol_revision"],
            "guard_revision": "U10-GUARD-3",
            "decision": "No retry, no later DEV calls, no aggregation, no freeze, no HOLDOUT.",
            "reason": summary["u8_safety_failures"][0],
            "u10_overlap_exception_applicable": False,
            "basis": "The failing parent has checked lexical_tokens=15 and coverage_status=assessed; it is not no_attributable_lexical_tokens.",
        },
        "acquisition_identity": {
            "meeting": "ES2009d",
            "phase": "dev",
            "attempt": 1,
            "started_utc": outcome["started_utc"],
            "finished_utc": outcome["finished_utc"],
            "wall_s": outcome["wall_s"],
            "exit_code": outcome["exit_code"],
            "retried": outcome["retried"],
            "canonical_command": outcome["canonical_command"],
            "canonical_case": {
                "path": summary["case_output"]["path"],
                "sha256": summary["case_output"]["sha256"],
                "bytes": RAW_PATH.stat().st_size,
            },
            "stdout": summary["stdout"],
            "derived_files": {
                name: {
                    "path": str(ATTEMPT / name),
                    "sha256": file_sha256(ATTEMPT / name),
                    "bytes": (ATTEMPT / name).stat().st_size,
                }
                for name in derived_names
            },
            "ledger_after": outcome["ledger_after"],
        },
        "runtime_and_input_pins": {
            "input_manifest": {"path": str(MANIFEST_PATH), "sha256": preflight["manifest"]["sha256"]},
            "native_executable": manifest["native"]["paced_exe"],
            "sortformer_model": manifest["native"]["sortformer_model"],
            "runtime_archive": {
                "path": preflight["capsule_reuse"]["archive"],
                "sha256": preflight["capsule_reuse"]["archive_sha256"],
            },
            "capsule": {
                "fingerprint": preflight["capsule_reuse"]["capsule_fingerprint"],
                "root": preflight["capsule_reuse"]["capsule_root"],
                "overlay_files": preflight["capsule_reuse"]["overlay_files"],
                "overlay_mismatches": preflight["capsule_reuse"]["overlay_mismatches"],
                "reuse_expected": preflight["capsule_reuse"]["reuse_expected"],
                "expected_tests": preflight["capsule_reuse"]["expected_tests"],
            },
            "prompt": summary["launch"]["prompt"],
            "driver": {
                "path": "experiments/psem_r2_policy/artifacts/dev-supervised/supervised_exec.py",
                "sha256": "707a04a4e6644226f9965702e42865beadc6fd25afa554044f801b2c5165c3e8",
            },
            "audio": meeting_input["audio"],
            "word_annotations": meeting_input["words"],
            "ground_truth": meeting_input["gt"],
        },
        "gates": {
            "activated_preflight": {
                "path": str(PREFLIGHT_PATH),
                "gates_all_ok": preflight["gates_all_ok"],
                "providers_started": preflight["providers_started"],
                "paid_calls": preflight["paid_calls"],
                "cases": preflight["gates"],
                "capsule_reuse": preflight["capsule_reuse"],
            },
            "case_input_gate": paid_gate,
            "capsule_precheck": capsule_precheck,
            "execution": {
                "completed": summary["completed"],
                "execution_completed": summary["execution_completed"],
                "operational_clean": summary["operational_clean"],
                "evaluation_valid": summary["evaluation_valid"],
                "clean_completion": summary["clean_completion"],
                "incomplete_reasons": summary["u8_execution_incomplete_reasons"],
                "invalid_reasons": summary["u8_evaluation_invalid_reasons"],
                "safety_failures": summary["u8_safety_failures"],
                "timing_failures": summary["timing_failures"],
                "task_failures": summary["task_failures"],
                "provider_fault": summary["provider_fault"],
                "budget_truncated": summary["budget_truncated"],
            },
        },
        "source_accounting_and_pacing": {
            "whole_audio": {
                "frames": meeting_input["audio"]["nframes"],
                "sample_rate_hz": meeting_input["audio"]["framerate_hz"],
                "duration_s": meeting_input["audio"]["duration_s"],
            },
            "feed": {
                "fed_source_samples": pacing["fed_source_samples"],
                "fed_source_duration_s": pacing["fed_source_samples"] / meeting_input["audio"]["framerate_hz"],
                "unprocessed_source_samples": pacing["unprocessed_source_samples"],
                "capture_frame_seconds": pacing["capture_frame_seconds"],
                "feed_progress_n_samples": pacing["n_samples"],
                "feed_progress_first_sample": pacing["first_sample"],
                "feed_progress_last_sample": pacing["last_sample"],
                "feed_progress_source_span_s": pacing["source_span_s"],
                "feed_progress_wall_span_s": pacing["wall_span_s"],
                "overall_rate": pacing["overall_rate"],
                "window_min_span_s": pacing["window_min_span_s"],
                "window_rate_min": pacing["window_rate_min"],
                "window_rate_median": pacing["window_rate_median"],
                "window_rate_max": pacing["window_rate_max"],
                "source_eof_monotonic_s": pacing["source_eof_monotonic_s"],
                "native_chunk_arrival_span_s": pacing["native_chunk_arrival_span_s"],
                "native_chunk_arrival_distinct_stamps": pacing["native_chunk_arrival_distinct_stamps"],
                "native_chunk_arrival_batched": pacing["native_chunk_arrival_batched"],
            },
            "duration_note": "Whole-audio duration is nframes/sample_rate (2114.944 s). feed_progress source_span (2112.992 s) is last recorded source position minus first recorded source position, not total fed duration; fed_source_samples equals all 33,839,104 declared frames and unprocessed_source_samples is zero.",
            "case_counts": summary["counts"],
            "dispatch": summary["dispatch"],
            "seal_lateness_summary": summary["seal_lateness_summary"],
        },
        "safety_witness": {
            "parent_record": {
                field: parent[field]
                for field in (
                    "index", "parent_id", "meeting", "cluster_id", "outcome", "terminal_outcome",
                    "seal_reason", "status", "clean_completion", "degraded",
                    "unsuccessful_source_processing", "accounted", "provenance_valid",
                    "text_authority", "failure_reason", "incomplete", "outage", "text", "n_timed",
                    "timed_start_ms", "timed_timings", "span", "assignment", "conserved",
                    "unknown_reasons", "group_ids", "reconstructed", "children", "marks", "latency",
                    "receipt", "sequential_target",
                )
            },
            "guard": parent["guard"],
            "shared_token_attribution": {
                "r0_equals_r2": r0_attribution == r2_attribution,
                "sha256": hashlib.sha256(attribution_bytes).hexdigest(),
                "records": r0_attribution,
            },
            "token_24": {"token": token_24, "attribution": attribution_24},
            "arms": {"r0": slim_arm(parent["r0"]), "r2": slim_arm(parent["r2"])},
            "factual_interpretation": {
                "token_24_text": token_24["text"],
                "unique_gt_role": attribution_24["roles"],
                "r0_reference_unit": r0_reference_unit,
                "r2_reference_unit": r2_reference_unit,
                "statement": "Token 24 is uniquely attributed to GT role B. R0-0 has evaluation reference role B and token 24 is absent from the R0 wrong-token set. R2 UNKNOWN-4 has evaluation reference role D, crosses a verified boundary, and includes token 24 in the R2 wrong-token set. This is a factual projection of recorded scorer output, not a new policy interpretation.",
            },
        },
        "budget": {
            "ledger_before": prelaunch["ledger_before"],
            "ledger_after": outcome["ledger_after"],
            "canonical_summary": summary["ledger"],
            "openrouter_accounted_usd_cumulative": 0.67445175,
            "openrouter_accounted_usd_added_by_case": 0.205427384,
            "note": "Deepgram entries/credit are informational under U7; OpenRouter retained reservations are the cash-cap evidence.",
        },
        "remaining": {
            "ES2002b": "NOT_STARTED",
            "EN2009d": "NOT_STARTED",
            "aggregate_5_dev": "NOT_PRODUCED",
            "holdout": "UNTOUCHED",
            "retry": "NONE",
        },
    }
    OUTPUT.write_text(json.dumps(projection, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(OUTPUT), "bytes": OUTPUT.stat().st_size, "sha256": file_sha256(OUTPUT)}))


if __name__ == "__main__":
    main()
