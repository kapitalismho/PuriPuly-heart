from __future__ import annotations

import json
import re
import statistics
from collections import Counter, defaultdict

from prepare import HERE, save


def quantile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def spread(values):
    return {"n": len(values), "median": statistics.median(values), "p10": quantile(values, 0.1), "p90": quantile(values, 0.9), "min": min(values), "max": max(values)} if values else None


def aggregate(rows):
    complete = [row for row in rows if row["status"] in {"completed", "quality_flag"}]
    result = {"attempted": len(rows), "status_counts": dict(Counter(row["status"] for row in rows))}
    for key in ("wall_ms", "ttft_ms", "cpu_seconds", "cpu_percent_one_core"):
        result[key] = spread([row[key] for row in complete if row.get(key) is not None])
    for key in ("prompt_n", "cache_n", "predicted_n", "prompt_ms", "predicted_ms", "prompt_per_second", "predicted_per_second", "draft_n", "draft_n_accepted"):
        values = [row["timings"][key] for row in complete if key in row.get("timings", {})]
        result[key] = spread(values)
        if key in {"prompt_n", "predicted_n", "draft_n", "draft_n_accepted"}:
            result[key + "_sum"] = sum(values) if values else None
    for key in ("private_commit", "working_set"):
        result[key + "_bytes"] = spread([row["memory_after"][key] for row in complete if row.get("memory_after")])
    return result


def comparison_records(groups):
    comparisons = {}
    for backend in ("cpu", "vulkan"):
        for profile, middle in (("update_only", ""), ("cache0", "cache0-")):
            entries = {}
            paired = {}
            for version in ("old", "new"):
                prefix = f"{version}-{backend}-{middle}final-"
                rows = [row for label, values in groups.items() if label.startswith(prefix) for row in values if row["request_index"] >= 3]
                entries[version] = aggregate(rows)
                paired[version] = {(row["arm"].rsplit("-", 1)[1], row["request_index"], row["case"]): row for row in rows if row["status"] == "completed"}
            keys = sorted(paired["old"].keys() & paired["new"].keys())
            entries["matched_pairs"] = len(keys)
            entries["paired_wall_delta_ms"] = spread([paired["new"][key]["wall_ms"] - paired["old"][key]["wall_ms"] for key in keys])
            entries["paired_wall_ratio_new_over_old"] = spread([paired["new"][key]["wall_ms"] / paired["old"][key]["wall_ms"] for key in keys])
            comparisons[f"{backend}-{profile}"] = entries
    for backend in ("cpu", "vulkan"):
        for profile in ("default", "cache0"):
            entries = {}
            paired = {}
            for version in ("old", "new"):
                rows = [row for row in groups.get(f"literal-{version}-{backend}-{profile}", []) if row["request_index"] >= 3]
                entries[version] = aggregate(rows)
                paired[version] = {row["request_index"]: row for row in rows if row["status"] == "completed"}
            keys = sorted(paired["old"].keys() & paired["new"].keys())
            entries["matched_pairs"] = len(keys)
            entries["paired_wall_delta_ms"] = spread([paired["new"][key]["wall_ms"] - paired["old"][key]["wall_ms"] for key in keys])
            comparisons[f"literal-{backend}-{profile}"] = entries
    save("comparisons.json", comparisons)
    bins = {}
    for label, rows in groups.items():
        if not label.endswith("-soak"):
            continue
        entries = []
        for start in range(0, len(rows), 20):
            block = rows[start:start + 20]
            entries.append({"first_request": block[0]["request_index"], "last_request": block[-1]["request_index"], **aggregate(block)})
        bins[label] = entries
    save("memory-soak-bins.json", bins)
    quality = {label: {row["case"]: row["output"] for row in rows if row["status"] == "completed"} for label, rows in groups.items() if label.endswith("-quality")}
    cases = sorted({case for outputs in quality.values() for case in outputs})
    deterministic = {}
    on = {row["case"]: row for row in groups.get("new-cpu-greedy-on", [])}
    off = {row["case"]: row for row in groups.get("new-cpu-greedy-off-clean", [])}
    for case in sorted(on.keys() | off.keys()):
        deterministic[case] = {"both_completed": on.get(case, {}).get("status") == "completed" and off.get(case, {}).get("status") == "completed", "output_equal": on.get(case, {}).get("output") == off.get(case, {}).get("output")}
    save("quality-comparisons.json", {"compatibility_arms": sorted(quality), "cases": {case: {"arm_count": sum(case in outputs for outputs in quality.values()), "unique_outputs": sorted({outputs[case] for outputs in quality.values() if case in outputs})} for case in cases}, "deterministic_mtp_on_off": deterministic})
    all_rows = [row for rows in groups.values() for row in rows]
    save("execution-counts.json", {"standalone_requests": len(all_rows), "request_statuses": dict(Counter(row["status"] for row in all_rows)), "phase_counts": dict(Counter(row["phase"] for row in all_rows)), "arms_with_requests": len(groups), "preflight_failure_records": [path.name for path in sorted((HERE / "results").glob("*-failure.json"))]})


def main():
    groups = defaultdict(list)
    for line in (HERE / "results/requests.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        groups[row["arm"]].append(row)
    summary = {}
    for label, all_rows in groups.items():
        rows = [row for row in all_rows if row["status"] in {"completed", "quality_flag"} and not (row["phase"] in {"final", "tuning", "soak", "contention", "literal"} and row["request_index"] < 3 and len(all_rows) > 3)]
        entry = {"attempted": len(all_rows), "status_counts": dict(Counter(row["status"] for row in all_rows)), "excluded_warmup_count": len(all_rows) - len(rows), "wall_ms": spread([r["wall_ms"] for r in rows]), "ttft_ms": spread([r["ttft_ms"] for r in rows if r.get("ttft_ms") is not None]), "cpu_seconds": spread([r["cpu_seconds"] for r in rows if r.get("cpu_seconds") is not None]), "cpu_percent_one_core": spread([r["cpu_percent_one_core"] for r in rows if r.get("cpu_percent_one_core") is not None]), "private_commit_bytes": spread([r["memory_after"]["private_commit"] for r in rows if r.get("memory_after")]), "working_set_bytes": spread([r["memory_after"]["working_set"] for r in rows if r.get("memory_after")]), "sampled_peak_private_commit_bytes": max((r["sampled_request_peaks"]["private_commit"] for r in rows), default=None), "sampled_peak_working_set_bytes": max((r["sampled_request_peaks"]["working_set"] for r in rows), default=None), "per_case": {}, "errors_and_flags": [r for r in all_rows if r["status"] != "completed"]}
        for key in ("prompt_n", "cache_n", "predicted_n", "prompt_ms", "predicted_ms", "prompt_per_second", "predicted_per_second", "draft_n", "draft_n_accepted"):
            values = [r["timings"][key] for r in rows if r.get("timings") and key in r["timings"]]
            entry[key] = spread(values)
            if key in {"prompt_n", "cache_n", "predicted_n", "draft_n", "draft_n_accepted"}:
                entry[key + "_sum"] = sum(values) if values else None
        for case in sorted({r["case"] for r in rows}):
            case_rows = [r for r in rows if r["case"] == case]
            entry["per_case"][case] = {"wall_ms": spread([r["wall_ms"] for r in case_rows]), "ttft_ms": spread([r["ttft_ms"] for r in case_rows]), "outputs": sorted({r["output"] for r in case_rows})}
        metadata = HERE / "results" / f"{label}-run.json"
        if metadata.exists():
            run = json.loads(metadata.read_text(encoding="utf-8"))
            entry["run_observations"] = {k: run.get(k) for k in ("readiness_ms", "M1_model_ready", "M5_before_shutdown", "gpu_M1", "gpu_M5", "host_cpu_before_percent", "host_cpu_after_percent", "exit_code", "owned_process_exited", "resource_stop")}
        producer = HERE / "results" / f"{label}-producer.txt"
        if producer.exists():
            text = producer.read_text(encoding="utf-8")
            entry["producer_buffers"] = re.findall(r"[^\n]*(?:model buffer size|compute buffer size|KV buffer size|lazy read enabled|enabling prefetch|offloaded \d+/\d+ layers)[^\n]*", text)
            stats = re.findall(r"[^\n]*statistics\s+draft-mtp:[^\n]*", text)
            entry["mtp_last_cumulative_statistics"] = stats[-1] if stats else None
            entry["warnings_errors"] = re.findall(r"[^\n]*(?: E | W |assert|NaN|nan detected|error:)[^\n]*", text)
            cache_states = re.findall(r"cache state: (\d+) prompts, ([\d.]+) MiB", text)
            entry["native_prompt_cache"] = {"observed_states": len(cache_states), "maximum_prompts": max((int(count) for count, _ in cache_states), default=None), "maximum_reported_mib": max((float(size) for _, size in cache_states), default=None), "last_state": cache_states[-1] if cache_states else None}
        summary[label] = entry
    save("summary.json", summary)
    comparison_records(groups)
    for label, entry in summary.items():
        wall, ttft = entry["wall_ms"], entry["ttft_ms"]
        print(label, entry["status_counts"], "wall", round(wall["median"], 2) if wall else None, "ttft", round(ttft["median"], 2) if ttft else None, "privateMiB", round(entry["private_commit_bytes"]["median"] / 1048576, 1) if entry["private_commit_bytes"] else None)


if __name__ == "__main__":
    main()
