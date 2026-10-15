from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


def distribution(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "median": None, "p95": None, "min": None, "max": None}
    values = sorted(values)
    return {"n": len(values), "median": statistics.median(values), "p95": values[max(0, math.ceil(0.95 * len(values))-1)], "min": values[0], "max": values[-1]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    unique_outputs = {}
    for path in sorted(args.results.glob("*.json")):
        if path.name in {"summary.json", "identity.json", "app.json", "checks.json"}:
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        if "profile" not in result or "requests" not in result:
            continue
        requests = result.get("requests", [])
        for request in requests:
            key = (request["input"], request["output"])
            unique_outputs.setdefault(key, []).append({"profile": result["profile"], "phase": request["phase"], "index": request["index"]})
        row = {key: result.get(key) for key in ("profile", "status", "error", "build", "command", "cold_readiness_seconds", "sample_count", "cpu_seconds", "cpu_one_core_percent", "process_exit_code", "shutdown", "counter_errors", "memory_peaks", "phase_memory_peaks", "mapped_resident_at_ready", "mapped_resident_after_repeated", "placement_cache_log")}
        row["gpu_adapter_luid"] = result.get("gpu_adapter_luid")
        row["counter_instances"] = result.get("counter_instances")
        row["elapsed_seconds"] = result.get("elapsed_seconds")
        row["cpu_all_16_logical_percent"] = result.get("cpu_one_core_percent", 0) / 16
        row["requests_completed"] = len(requests)
        row["quality_check_failures"] = [{"index": request["index"], "phase": request["phase"], "checks": request["quality_checks"]} for request in requests if not all(request["quality_checks"].values())]
        row["phases"] = {}
        for phase in sorted({request["phase"] for request in requests}):
            cases = [request for request in requests if request["phase"] == phase]
            row["phases"][phase] = {"wall_seconds": distribution([r["wall_seconds"] for r in cases]), "ttft_seconds": distribution([r["ttft_seconds"] for r in cases if r["ttft_seconds"] is not None]), "decode_tokens_per_second": distribution([r["timings"]["predicted_per_second"] for r in cases if r.get("timings") and "predicted_per_second" in r["timings"]]), "prompt_tokens": distribution([r["timings"]["prompt_n"] for r in cases if r.get("timings") and "prompt_n" in r["timings"]]), "output_tokens": distribution([r["timings"]["predicted_n"] for r in cases if r.get("timings") and "predicted_n" in r["timings"]])}
        row["memory_units"] = {key: {"bytes": value, "GB": value / 1e9, "GiB": value / 2**30} for key, value in result.get("memory_peaks", {}).items() if key.endswith("bytes") or key in {"WorkingSetSize", "PeakWorkingSetSize", "PrivateUsage", "PeakPagefileUsage"}}
        sample_path = args.raw_root / "runs" / result["profile"] / "samples.json"
        if sample_path.exists():
            samples = json.loads(sample_path.read_text(encoding="utf-8"))
            row["sample_cadence_seconds"] = distribution([b["time"]-a["time"] for a,b in zip(samples, samples[1:])])
            if requests and samples:
                start = min(r["started_seconds"] for r in requests)
                end = max(r["ended_seconds"] for r in requests)
                first = min(samples, key=lambda s: abs(s["time"]-start))
                last = min(samples, key=lambda s: abs(s["time"]-end))
                row["serving_cpu_seconds_sampled"] = last.get("cpu_seconds", 0)-first.get("cpu_seconds", 0)
                row["serving_wall_seconds"] = end-start
                row["serving_cpu_one_core_percent_sampled"] = row["serving_cpu_seconds_sampled"] / (end-start) * 100
            compact = []
            bins = {}
            for sample in samples:
                bins.setdefault((int(sample["time"]), sample.get("phase")), []).append(sample)
            for (second, phase), group in bins.items():
                item = {"second": second, "phase": phase, "n": len(group)}
                for key in group[0]:
                    if key in {"time", "phase"}:
                        continue
                    values = [s[key] for s in group if s.get(key) is not None]
                    item[key + "_max"] = max(values) if values else None
                    if key == "system_available_physical_bytes":
                        item[key + "_min"] = min(values) if values else None
                compact.append(item)
            (args.results / (result["profile"] + ".samples.jsonl")).write_text("".join(json.dumps(sample, separators=(",", ":")) + "\n" for sample in compact), encoding="utf-8")
            row["minimum_available_physical_bytes"] = min(s["system_available_physical_bytes"] for s in samples) if samples else None
        rows.append(row)
    (args.results / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.results / "quality-unique.json").write_text(json.dumps([{"input": key[0], "output": key[1], "occurrences": occurrences} for key, occurrences in sorted(unique_outputs.items())], ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps([{k: r[k] for k in ("profile", "status", "requests_completed", "cold_readiness_seconds", "quality_check_failures")} for r in rows], ensure_ascii=False))


if __name__ == "__main__":
    main()
