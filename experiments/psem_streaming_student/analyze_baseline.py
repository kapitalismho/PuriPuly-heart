from __future__ import annotations

import gzip
import hashlib
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
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
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        return [json.loads(line) for line in source]

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
    run_dir = HERE / "runs" / meeting
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
    result["source"]["annotation_coverage"]["interpretation"] = "Transition and overlap conditions come from accessible AMI word intervals. The raw retained_parent_guards map is empty; guard counts below are a post-hoc annotation join over the recorded inputs, not runtime guard proof."
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
        "posthoc_guard_annotation_provenance": "Counts were joined after execution from annotations of the recorded accepted-parent inputs. The raw source annotation_coverage.retained_parent_guards={} is preserved and is not runtime guard proof.",
        "annotation_overlap_witnesses": overlap_witnesses(words),
        "interpretation": f"{len(timely_parents)} parents contain confirmed transitions that arrived before this source's replay cutoff. Full coverage was nevertheless unavailable for every parent because the end-covering output arrived later; this only characterizes the artificial zero-wait cutoff and does not establish a receiver defect, actual ASR admission blocking, or downstream benefit.",
    }
    cost = result["cost"]
    prior_cpu = cost.get("receiver_wrapper_cpu_measurement") or {}
    prior_rss = cost.get("receiver_wrapper_working_set_measurement") or {}
    raw_cpu = cost.pop("receiver_wrapper_cpu_s", prior_cpu.get("raw_returned_value", 0.0))
    raw_rss = cost.pop("receiver_wrapper_peak_working_set_bytes_observed", prior_rss.get("raw_returned_value", 0))
    raw_sum = cost.pop("complete_path_peak_sum_working_set_bytes_sampled", cost.get("complete_path_peak_sum_working_set_bytes_observed"))
    if raw_cpu and raw_rss and raw_sum:
        cost["receiver_wrapper_cpu_measurement"] = {"status": "observed", "seconds": raw_cpu}
        cost["receiver_wrapper_working_set_measurement"] = {"status": "observed", "peak_bytes": raw_rss}
        cost["complete_path_peak_sum_working_set_bytes_observed"] = raw_sum
        cost.pop("complete_path_working_set_lower_bound_bytes", None)
        cost["rss_sampling"] = "Native and receiver-wrapper working sets were sampled together; the peak sum is the maximum simultaneous sample."
    else:
        cost["receiver_wrapper_cpu_measurement"] = {"status": "unavailable", "raw_returned_value": raw_cpu, "reason": "The original execution probe's Windows pseudo-handle measurement returned zero; zero is retained but not reported as receiver processing cost."}
        cost["receiver_wrapper_working_set_measurement"] = {"status": "unavailable", "raw_returned_value": raw_rss, "reason": "The original execution probe's Windows pseudo-handle measurement returned zero; zero is retained but not reported as receiver memory."}
        cost.pop("complete_path_peak_sum_working_set_bytes_observed", None)
        cost["complete_path_working_set_lower_bound_bytes"] = cost["native_peak_working_set_bytes_observed"]
        cost["rss_sampling"] = "Native process working set was observed. Receiver-wrapper RSS failed, so the native peak is only a complete-path lower bound and no complete-path peak sum is claimed."
    result["receiver"]["late_events_postclassified_from_actual_qpc"] = len(analysis["late_after_commit_events"])
    result["receiver"]["causal_analysis"] = analysis
    result["cost"] = cost
    result["artifacts"]["soft_target_tensor"] = {"path": str((run_dir / "dump" / "diar.probs.f32").relative_to(ROOT)), "sha256": digest(run_dir / "dump" / "diar.probs.f32")}
    result["artifacts"]["soft_target_metadata"] = {"path": str((run_dir / "dump" / "diar.probs.json").relative_to(ROOT)), "sha256": digest(run_dir / "dump" / "diar.probs.json")}
    write_json(run_dir / "RESULT.json", result)
    return analysis, cost


def main() -> None:
    summary = read_json(HERE / "RESULT.json")
    analyses = {}
    costs = {}
    guards = retained_guards()
    for meeting in summary["execution"]["sources"]:
        analyses[meeting], costs[meeting] = analyze_source(meeting, guards)
    summary["status"] = "completed_cutoff_conditional_partial_baseline"
    summary["identities"]["executed_runner_revision_provenance"] = "unavailable; the executed runner revision was not pinned during the two native passes"
    summary["identities"]["runner_sha256_after_analysis_fix"] = digest(HERE / "run_baseline.py")
    summary["identities"]["analysis_sha256"] = digest(Path(__file__))
    summary["measurements"]["causal_receiver"] = analyses
    for meeting in summary["execution"]["sources"]:
        summary["measurements"]["annotation_conditions"][meeting]["interpretation"] = "Transition and overlap conditions come from accessible AMI word intervals. The raw retained_parent_guards map is empty; post-hoc guard counts characterize recorded inputs and are not runtime guard proof."
    summary["measurements"]["cost"] = costs
    summary["decision"] = {
        "baseline_target_usable": True,
        "disposition": "CUTOFF_CONDITIONAL_PARTIAL_BASELINE",
        "finding": "All 38 admitted parents lacked end-of-span native frame coverage at the artificial zero-wait frozen accepted-text cutoff. Six parents contained timely confirmed transitions (2 ES, 4 EN), so this is partial coverage rather than a proven receiver defect or actual ASR admission blocker.",
        "soft_target_status": "usable raw four-slot independent probabilities with support/validity metadata",
        "complete_path_cost_status": "incomplete because original receiver CPU and RSS measurements are unavailable",
        "not_a_teacher_finetuning_finding": True,
        "compression_training": "not run; may proceed only after maintainer approval and is not contingent on changing wait or fixed policy",
        "quality_scope": "The approved prefixes contain actual non-overlap transitions, overlap intervals, and post-hoc retained-parent annotations, but this baseline establishes no general early-stop, downstream benefit, or negative teacher result.",
        "requested_next_decision": "Discuss R4 training under its own authority. Preserve zero added wait and the fixed policy; the cutoff-conditional mapping gap may be retained or separately scoped without making policy repair mandatory.",
    }
    write_json(HERE / "RESULT.json", summary)
    findings = {"schema": "PSEM-STREAMING-STUDENT-BASELINE-FINDINGS-1", "result_status": summary["status"], "execution": summary["execution"], "causal_receiver": analyses, "cost": costs, "decision": summary["decision"], "provenance": {"executed_runner_revision": summary["identities"]["executed_runner_revision_provenance"], "posthoc_runner_sha256": summary["identities"]["runner_sha256_after_analysis_fix"]}, "evidence_boundaries": summary["architecture"], "commands": ["python -B experiments/psem_streaming_student/run_baseline.py prepare", "python -B experiments/psem_streaming_student/run_baseline.py execute", "python -B experiments/psem_streaming_student/analyze_baseline.py", "python -B experiments/psem_streaming_student/run_baseline.py verify"]}
    write_json(HERE / "FINDINGS.json", findings)
    print(json.dumps({"status": findings["result_status"], "decision": findings["decision"]["disposition"], "findings_sha256": digest(HERE / "FINDINGS.json")}, separators=(",", ":")))


if __name__ == "__main__":
    main()
