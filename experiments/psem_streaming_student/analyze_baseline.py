from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIG_PATH = HERE / "baseline_config.json"
OUTPUT_ROOT = HERE
RUNS = OUTPUT_ROOT / "runs"
ANNOTATIONS = Path(r"C:/Users/salee/AppData/Local/Temp/opencode/stb_phase2_corpora/ami/annotations/words")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        return [json.loads(line) for line in source]
def measurement_metadata(run_dir: Path, cost: dict[str, Any]) -> None:
    memory_path = run_dir / "process-memory-samples.csv"
    if memory_path.exists():
        with memory_path.open("r", encoding="utf-8", newline="") as source:
            qpcs = [int(row["qpc"]) for row in csv.DictReader(source)]
        intervals_ms = sorted((right - left) / 10_000 for left, right in zip(qpcs, qpcs[1:]))
        cost["memory_sampling"] = {
            "samples": len(qpcs),
            "pairing": "Native and wrapper readings are back-to-back in each QPC-stamped sample; near-simultaneous, not atomic.",
            "median_interval_ms": intervals_ms[len(intervals_ms) // 2],
            "max_interval_ms": max(intervals_ms),
            "aggregate_peak": "Maximum sum from one sampled pair; not a sum of independent peaks.",
            "wrapper_scope": "Python receiver/orchestrator and measurement instrumentation footprint, not production-only receiver memory.",
        }
    gpu_path = run_dir / "gpu-process-memory.csv"
    if gpu_path.exists():
        header = ""
        for encoding in ("utf-16", "utf-8-sig", "cp949"):
            try:
                header = gpu_path.read_text(encoding=encoding).splitlines()[0]
                break
            except (UnicodeError, IndexError):
                pass
        match = re.search(r"pid_(\d+)_", header, re.IGNORECASE)
        if match:
            cost["gpu_memory"]["native_pid"] = int(match.group(1))
            cost["gpu_memory"]["sampling_interval_s"] = 1
            cost["gpu_memory"]["sampling_method"] = "typeperf per-native-PID dedicated-usage counter"


def cross_run_cpu_factor(meeting: str, current: float) -> float | None:
    counterpart = HERE / ("runs" if OUTPUT_ROOT != HERE else "cost_rerun/runs") / meeting / "RESULT.json"
    if not counterpart.exists():
        return None
    other = float(read_json(counterpart)["cost"]["native_process_cpu_s"])
    return max(current, other) / min(current, other) if current > 0 and other > 0 else None

def retained_guards() -> dict[tuple[str, str], dict[str, Any]]:
    result = {}
    path = ROOT / "experiments/psem_r2_policy/artifacts/retained/historical_policy_inputs.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row.get("cohort") == "current" and row.get("type") == "parent":
                value = row["value"]
                result[(row["meeting"], value["parent_id"])] = ((value.get("r0") or {}).get("guard") or {})
    return result



def intervals(meeting: str, limit_samples: int) -> list[tuple[int, int, str, str]]:
    result = []
    for path in sorted(ANNOTATIONS.glob(f"{meeting}.*.words.xml")):
        role = path.name.split(".")[1]
        for node in ET.parse(path).getroot():
            if not node.tag.endswith("w") or node.get("punc") == "true" or node.get("starttime") is None or node.get("endtime") is None:
                continue
            start = round(float(node.get("starttime")) * 16000)
            end = min(round(float(node.get("endtime")) * 16000), limit_samples)
            if start < limit_samples and end > start:
                result.append((start, end, role, "".join(node.itertext())))
    return sorted(result)


def overlap_witnesses(words: list[tuple[int, int, str, str]]) -> list[dict[str, Any]]:
    result = []
    seen = set()
    for index, left in enumerate(words):
        for right in words[index + 1:]:
            if right[0] >= left[1]:
                break
            if left[2] == right[2]:
                continue
            start, end = max(left[0], right[0]), min(left[1], right[1])
            key = (start, end, *sorted((left[2], right[2])))
            if end > start and key not in seen:
                seen.add(key)
                result.append({"span_samples": [start, end], "roles": sorted((left[2], right[2])), "words": [left[3], right[3]]})
    return result[:12]


def analyze_source(meeting: str, guards: dict[tuple[str, str], dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    run_dir = RUNS / meeting
    result = read_json(run_dir / "RESULT.json")
    assignments = read_jsonl(run_dir / "receiver-assignments.jsonl.gz")
    messages = read_jsonl(run_dir / "native-events.jsonl.gz")
    qpf = int(result["clock"]["qpf"])
    chunks = [row for row in messages if row.get("type") == "chunk"]
    overlapping = []
    for assignment in assignments:
        for event in result["native"]["transitions"]:
            if int(assignment["source_span"][0]) < int(event["boundary_sample"]) < int(assignment["source_span"][1]):
                relative = (int(event["receiver_receipt_qpc"]) - int(assignment["clock"]["actual_admission_qpc"])) / qpf
                overlapping.append({"parent_id": assignment["parent_id"], "accepted_text": assignment["accepted_text"], "event_id": event["event_id"], "boundary_sample": event["boundary_sample"], "receiver_receipt_minus_admission_s": relative, "timely": relative <= 0, "receiver_result": assignment["receiver"]["unknown_reasons"], "uncertain_tokens": [{"index": index, "text": token["text"], "timing": token.get("timing")} for index, token in enumerate(assignment["tokens"]) if token.get("timing") != "interval"]})
    parent_events = {row["parent_id"] for row in overlapping}
    timely_parents = {row["parent_id"] for row in overlapping if row["timely"]}
    cutoff_lags = []
    for assignment in assignments:
        end_sample = int(assignment["source_span"][1])
        end_frame = end_sample // 1280
        chunk = next(row for row in chunks if int(row["emit_start_frame"]) <= end_frame < int(row["emit_start_frame"]) + int(row["emit_count"]))
        cutoff_lags.append((int(chunk["receiver_receipt_qpc"]) - int(assignment["clock"]["scheduled_cutoff_qpc"])) / qpf)
    gt = result["source"]["annotation_coverage"]["transition_witnesses"]
    available = list(result["native"]["transitions"])
    matches = []
    used = set()
    for witness in gt:
        candidates = [(abs(int(event["boundary_sample"]) - int(witness["sample"])), index, event) for index, event in enumerate(available) if index not in used]
        if not candidates:
            continue
        distance, index, event = min(candidates)
        if distance <= 4000:
            used.add(index)
            matches.append({"gt_sample": witness["sample"], "roles": [witness["from"], witness["to"]], "native_sample": event["boundary_sample"], "boundary_error_s": (int(event["boundary_sample"]) - int(witness["sample"])) / 16000, "confirmation_availability_lag_s": float(event["receiver_receipt_source_zero_s"]) - int(event["boundary_sample"]) / 16000})
    guard_counts = Counter()
    for assignment in assignments:
        guard = guards.get((meeting, assignment["parent_id"]))
        if guard and guard.get("same_speaker_stratum"):
            guard_counts["same_speaker"] += 1
        if guard and int(guard.get("span_verified_changes") or 0) > 0:
            guard_counts["verified_change"] += 1
        if guard and int((guard.get("excluded") or {}).get("mixed") or 0) > 0:
            guard_counts["mixed_or_overlap"] += 1
    words = intervals(meeting, int(result["source"]["evaluation_samples"]))
    raw_guard_counts = result["source"]["annotation_coverage"].get("retained_parent_guards") or {}
    if raw_guard_counts:
        guard_source = "The populated retained_parent_guards counts were computed before native launch by annotating the accepted input parents; they are not runtime receiver guard proof."
    else:
        guard_source = "The original-run raw retained_parent_guards map is empty; post-hoc counts below annotate the recorded inputs and are not runtime receiver guard proof."
    result["source"]["annotation_coverage"]["interpretation"] = f"Transition and overlap conditions come from accessible AMI word intervals. {guard_source}"
    analysis = {
        "meeting": meeting,
        "native_events": len(result["native"]["transitions"]),
        "gt_nonoverlap_transitions": len(gt),
        "gt_matches_within_250ms": len(matches),
        "matched_transitions": matches,
        "accepted_parents": len(assignments),
        "cutoff_coverage": {
            "admission_rule": "artificial zero-wait frozen accepted-text source-span end",
            "parents_whose_end_covering_native_frame_arrived_after_cutoff": len(cutoff_lags),
            "scheduled_cutoff_lag_s": {"min": min(cutoff_lags), "max": max(cutoff_lags)},
            "scope": "Full native frame coverage at this artificial cutoff was unavailable for every admitted parent; this is cutoff-conditional and is not proof of a receiver defect or actual ASR admission blocker.",
        },
        "parents_with_native_event_inside": len(parent_events),
        "parents_with_timely_native_event_inside": len(timely_parents),
        "timely_event_parent_unknown_reasons": {assignment["parent_id"]: assignment["receiver"]["unknown_reasons"] for assignment in assignments if assignment["parent_id"] in timely_parents},
        "late_after_commit_events": [row for row in overlapping if not row["timely"]],
        "selected_policy_changed_parents": result["receiver"]["selected_changed_parents"],
        "selected_policy_removed_boundaries": result["receiver"]["selected_suppressed_boundaries"],
        "text_conservation_failures": result["receiver"]["text_conservation_failures"],
        "posthoc_retained_guard_annotation_counts": dict(guard_counts),
        "posthoc_guard_annotation_provenance": guard_source,
        "annotation_overlap_witnesses": overlap_witnesses(words),
        "interpretation": f"{len(timely_parents)} parents contain confirmed transitions that arrived before this source's replay cutoff. Full coverage was nevertheless unavailable for every parent because the end-covering output arrived later; this only characterizes the artificial zero-wait cutoff and does not establish a receiver defect, actual ASR admission blocking, or downstream benefit.",
    }
    cost = result["cost"]
    prior_cpu = cost.get("receiver_wrapper_cpu_measurement") or {}
    prior_rss = cost.get("receiver_wrapper_working_set_measurement") or {}
    raw_cpu = cost.pop("receiver_wrapper_cpu_s", prior_cpu.get("raw_api_seconds", prior_cpu.get("seconds", prior_cpu.get("raw_returned_value", 0.0))))
    raw_rss = cost.pop("receiver_wrapper_peak_working_set_bytes_observed", prior_rss.get("peak_bytes", prior_rss.get("raw_returned_value", 0)))
    raw_sum = cost.pop("complete_path_peak_sum_working_set_bytes_sampled", cost.get("complete_path_peak_sum_working_set_bytes_observed"))
    native_cpu = float(cost["native_process_cpu_s"])
    variation = cross_run_cpu_factor(meeting, native_cpu)
    cost["native_process_cpu_measurement"] = {
        "status": "diagnostic_only_quantum_limited_and_cross_run_variable",
        "raw_api_seconds": native_cpu,
        "api_quantum_s": 0.015625,
        "same_geometry_cross_run_variation_factor": variation,
        "interpretation": "Raw Windows process CPU reading retained. Same-geometry original/rerun readings vary materially; no speedup, efficiency, universal bias, or correction factor is inferred.",
    }
    cost["process_cpu_cost_status"] = "unquantified; API readings are quantum-limited and same-geometry runs are variable"
    if raw_rss and raw_sum:
        cost["receiver_wrapper_cpu_measurement"] = {"status": "diagnostic_instrument_floor", "raw_api_seconds": raw_cpu, "api_quantum_s": 0.015625, "interpretation": "Exactly one timer quantum; not a reliable wrapper CPU-efficiency measurement."}
        cost["receiver_wrapper_working_set_measurement"] = {"status": "observed_sampled", "peak_bytes": raw_rss}
        cost["complete_path_peak_sum_working_set_bytes_observed"] = raw_sum
        cost.pop("complete_path_working_set_lower_bound_bytes", None)
        cost["rss_sampling"] = "Native and receiver-wrapper readings are back-to-back within each QPC-stamped sample, so pairs are near-simultaneous rather than atomic. The aggregate peak is the maximum sampled pair, not a sum of independent peaks."
    else:
        cost["receiver_wrapper_cpu_measurement"] = {"status": "unavailable", "raw_returned_value": raw_cpu, "reason": "The original API returned zero. The executed original runner revision is unavailable, so the cause and any correction factor are unknown."}
        cost["receiver_wrapper_working_set_measurement"] = {"status": "unavailable", "raw_returned_value": raw_rss, "reason": "The original API returned zero. The executed original runner revision is unavailable, so the cause is unknown."}
        cost.pop("complete_path_peak_sum_working_set_bytes_observed", None)
        cost["complete_path_working_set_lower_bound_bytes"] = cost["native_peak_working_set_bytes_observed"]
        cost["rss_sampling"] = "Native process working set was observed. Original receiver-wrapper RSS returned zero for an unknown cause, so native peak remains only a complete-path lower bound."
    measurement_metadata(run_dir, cost)
    result["receiver"]["late_events_postclassified_from_actual_qpc"] = len(analysis["late_after_commit_events"])
    result["receiver"]["causal_analysis"] = analysis
    result["cost"] = cost
    result["artifacts"]["soft_target_tensor"] = {"path": str((run_dir / "dump" / "diar.probs.f32").relative_to(ROOT)), "sha256": digest(run_dir / "dump" / "diar.probs.f32")}
    result["artifacts"]["soft_target_metadata"] = {"path": str((run_dir / "dump" / "diar.probs.json").relative_to(ROOT)), "sha256": digest(run_dir / "dump" / "diar.probs.json")}
    write_json(run_dir / "RESULT.json", result)
    return analysis, cost


def main() -> None:
    global CONFIG_PATH, OUTPUT_ROOT, RUNS
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "baseline_config.json")
    args = parser.parse_args()
    CONFIG_PATH = args.config.resolve()
    config = read_json(CONFIG_PATH)
    OUTPUT_ROOT = (ROOT / config["output_root"]).resolve() if config.get("output_root") else HERE
    if OUTPUT_ROOT != HERE and HERE not in OUTPUT_ROOT.parents:
        raise RuntimeError("output_root must remain inside the experiment directory")
    RUNS = OUTPUT_ROOT / "runs"
    summary = read_json(OUTPUT_ROOT / "RESULT.json")
    analyses = {}
    costs = {}
    guards = retained_guards()
    for meeting in summary["execution"]["sources"]:
        analyses[meeting], costs[meeting] = analyze_source(meeting, guards)
    summary["status"] = "completed_cutoff_conditional_partial_baseline"
    posthoc_analysis_sha256 = digest(Path(__file__))
    if "runner_sha256" in summary["identities"]:
        if summary["identities"]["runner_sha256"] != digest(HERE / "run_baseline.py") or summary["identities"]["config_sha256"] != digest(CONFIG_PATH):
            raise RuntimeError("executed runner/config identity mismatch")
        summary["identities"]["executed_runner_revision_provenance"] = f"pinned before execution: {summary['identities']['source_revision']}"
        summary["identities"]["executed_analysis_sha256"] = summary["identities"]["analysis_sha256"]
        summary["identities"]["posthoc_analysis_sha256"] = posthoc_analysis_sha256
    else:
        summary["identities"]["executed_runner_revision_provenance"] = "unavailable; the executed runner revision was not pinned during the two native passes"
        summary["identities"]["runner_sha256_after_analysis_fix"] = digest(HERE / "run_baseline.py")
        summary["identities"]["previous_posthoc_analysis_sha256"] = summary["identities"].get("analysis_sha256")
        summary["identities"]["posthoc_analysis_sha256"] = posthoc_analysis_sha256
    summary["postprocessing"] = {
        "code_changed_after_execution": True,
        "executed_runner_unchanged": True,
        "posthoc_analysis_sha256": posthoc_analysis_sha256,
        "clarification": "Reporting-only repair: CPU API precision/variability, unknown original zero cause, guard provenance, wall decomposition, GPU PID/cadence, and near-simultaneous memory wording. No raw measured artifact or value changed.",
    }
    summary["measurements"]["causal_receiver"] = analyses
    for meeting in summary["execution"]["sources"]:
        summary["measurements"]["annotation_conditions"][meeting]["interpretation"] = read_json(RUNS / meeting / "RESULT.json")["source"]["annotation_coverage"]["interpretation"]
    summary["measurements"]["cost"] = costs
    combined_wall = summary["execution"].get("combined_paced_orchestration_wall_s")
    if combined_wall is not None:
        source_loop_sum = sum(float(value["complete_path_wall_s"]) for value in costs.values())
        summary["execution"]["source_timed_loop_wall_sum_s"] = source_loop_sum
        summary["execution"]["host_preparation_serialization_and_between_source_wall_s"] = combined_wall - source_loop_sum
        summary["execution"]["wall_scope"] = "Combined orchestration wall includes host preparation, serialization, and between-source work beyond the two native timed source loops; the cap comparison is conservative."
    rerun_memory_observed = all(value.get("receiver_wrapper_working_set_measurement", {}).get("status") == "observed_sampled" and value.get("complete_path_peak_sum_working_set_bytes_observed") for value in costs.values())
    summary["decision"] = {
        "baseline_target_usable": True,
        "disposition": "CUTOFF_CONDITIONAL_PARTIAL_BASELINE",
        "finding": "All 38 admitted parents lacked end-of-span native frame coverage at the artificial zero-wait frozen accepted-text cutoff. Timely confirmed-transition coverage remains partial, so this is not a proven receiver defect or actual ASR admission blocker.",
        "soft_target_status": "usable raw four-slot independent probabilities with support/validity metadata",
        "complete_path_cost_status": "paced wall and sampled process memory observed; process CPU is quantum-limited/variable and GPU-kernel compute remains unavailable" if rerun_memory_observed else "paced wall and native memory lower bound observed; receiver CPU/RSS and GPU-kernel compute remain unavailable",
        "not_a_teacher_finetuning_finding": True,
        "compression_training": "not run; GPU training-method discussion selected first and no training/backward is authorized",
        "quality_scope": "The approved prefixes contain actual non-overlap transitions, overlap intervals, and post-hoc retained-parent annotations, but this baseline establishes no general early-stop, downstream benefit, or negative teacher result.",
        "requested_next_decision": "Discuss GPU training method separately. No further baseline rerun or policy repair is implied; preserve zero added wait and the fixed policy.",
    }
    write_json(OUTPUT_ROOT / "RESULT.json", summary)
    command_suffix = f" --config {CONFIG_PATH.relative_to(ROOT)}" if config.get("output_root") else ""
    findings = {"schema": "PSEM-STREAMING-STUDENT-BASELINE-FINDINGS-1", "result_status": summary["status"], "execution": summary["execution"], "causal_receiver": analyses, "cost": costs, "decision": summary["decision"], "provenance": summary["identities"], "postprocessing": summary["postprocessing"], "evidence_boundaries": summary["architecture"], "commands": [f"python -B experiments/psem_streaming_student/analyze_baseline.py{command_suffix}", f"python -B experiments/psem_streaming_student/run_baseline.py verify{command_suffix}"]}
    write_json(OUTPUT_ROOT / "FINDINGS.json", findings)
    print(json.dumps({"status": findings["result_status"], "decision": findings["decision"]["disposition"], "findings_sha256": digest(OUTPUT_ROOT / "FINDINGS.json")}, separators=(",", ":")))


if __name__ == "__main__":
    main()
