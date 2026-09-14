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
    qpf = int(result["clock"]["qpf"])
    overlapping = []
    for assignment in assignments:
        for event in result["native"]["transitions"]:
            if int(assignment["source_span"][0]) < int(event["boundary_sample"]) < int(assignment["source_span"][1]):
                relative = (int(event["receiver_receipt_qpc"]) - int(assignment["clock"]["actual_admission_qpc"])) / qpf
                overlapping.append({"parent_id": assignment["parent_id"], "accepted_text": assignment["accepted_text"], "event_id": event["event_id"], "boundary_sample": event["boundary_sample"], "receiver_receipt_minus_admission_s": relative, "timely": relative <= 0, "receiver_result": assignment["receiver"]["unknown_reasons"], "uncertain_tokens": [{"index": index, "text": token["text"], "timing": token.get("timing")} for index, token in enumerate(assignment["tokens"]) if token.get("timing") != "interval"]})
    parent_events = {row["parent_id"] for row in overlapping}
    timely_parents = {row["parent_id"] for row in overlapping if row["timely"]}
    blocked = {assignment["parent_id"]: assignment["receiver"]["unknown_reasons"] for assignment in assignments if assignment["parent_id"] in timely_parents and assignment["receiver"]["unknown_reasons"]}
    late_post_commit = [row for row in overlapping if not row["timely"]]
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
        parent = assignment["parent_id"]
        guard = guards.get((meeting, parent))
        if guard and guard.get("same_speaker_stratum"):
            guard_counts["same_speaker"] += 1
        if guard and int(guard.get("span_verified_changes") or 0) > 0:
            guard_counts["verified_change"] += 1
        if guard and int((guard.get("excluded") or {}).get("mixed") or 0) > 0:
            guard_counts["mixed_or_overlap"] += 1
    words = intervals(meeting, int(result["source"]["evaluation_samples"]))
    analysis = {"meeting": meeting, "native_events": len(result["native"]["transitions"]), "gt_nonoverlap_transitions": len(gt), "gt_matches_within_250ms": len(matches), "matched_transitions": matches, "accepted_parents": len(assignments), "parents_with_native_event_inside": len(parent_events), "parents_with_timely_native_event_inside": len(timely_parents), "timely_event_parents_blocked_by_receiver": blocked, "late_after_commit_events": late_post_commit, "selected_policy_changed_parents": result["receiver"]["selected_changed_parents"], "selected_policy_removed_boundaries": result["receiver"]["selected_suppressed_boundaries"], "text_conservation_failures": result["receiver"]["text_conservation_failures"], "retained_guard_counts": dict(guard_counts), "annotation_overlap_witnesses": overlap_witnesses(words), "interpretation": "Actual receiver consumption reached confirmed native events before admission in every listed timely parent, but unchanged whole-parent coverage/uncertain-timing gates returned a whole-parent unit. Later events are post-commit observations only and were not retrospectively applied."}
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
    result["receiver"]["late_events_postclassified_from_actual_qpc"] = len(late_post_commit)
    result["receiver"]["causal_analysis"] = analysis
    result["cost"] = cost
    write_json(run_dir / "RESULT.json", result)
    return analysis, cost


def main() -> None:
    summary = read_json(HERE / "RESULT.json")
    analyses = {}
    costs = {}
    guards = retained_guards()
    for meeting in summary["execution"]["sources"]:
        analyses[meeting], costs[meeting] = analyze_source(meeting, guards)
    summary["status"] = "completed_with_supported_receiver_blocker"
    summary["identities"]["runner_sha256_after_analysis_fix"] = digest(HERE / "run_baseline.py")
    summary["identities"]["analysis_sha256"] = digest(Path(__file__))
    summary["measurements"]["causal_receiver"] = analyses
    summary["measurements"]["cost"] = costs
    summary["decision"] = {"baseline_target_usable": False, "disposition": "SUPPORTED_NAMED_TIMING_MAPPING_FAILURE", "failure": "Confirmed native events were consumed before the no-wait admission cutoff for 2/2 event-bearing ES parents and 4/4 event-bearing EN parents, but whole-parent eligibility did not produce a partition. All six failed complete evidence coverage at that cutoff; one ES parent additionally has an unmapped accepted token. The fixed selected UNKNOWN-only suppression therefore had no eligible boundary to change.", "not_a_teacher_finetuning_finding": True, "compression_training": "not run; awaiting later discussion", "quality_scope": "The approved prefixes contain actual non-overlap transitions, overlap intervals, and retained same-speaker guards, but this baseline does not establish translation effect or general teacher quality.", "requested_next_decision": "Discuss R4 training independently under its own authority; retain, separately repair, or scope out the receiver coverage/admission gap without changing the fixed policy by implication. No extra source/model pass is authorized here."}
    write_json(HERE / "RESULT.json", summary)
    findings = {"schema": "PSEM-STREAMING-STUDENT-BASELINE-FINDINGS-1", "result_status": summary["status"], "execution": summary["execution"], "causal_receiver": analyses, "cost": costs, "decision": summary["decision"], "evidence_boundaries": summary["architecture"], "commands": ["python -B experiments/psem_streaming_student/run_baseline.py prepare", "python -B experiments/psem_streaming_student/run_baseline.py execute", "python -B experiments/psem_streaming_student/analyze_baseline.py", "python -B experiments/psem_streaming_student/run_baseline.py verify"]}
    write_json(HERE / "FINDINGS.json", findings)
    print(json.dumps({"status": findings["result_status"], "decision": findings["decision"]["disposition"], "findings_sha256": digest(HERE / "FINDINGS.json")}, separators=(",", ":")))


if __name__ == "__main__":
    main()
