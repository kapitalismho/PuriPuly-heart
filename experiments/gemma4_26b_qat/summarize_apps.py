from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from summarize import distribution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.results.glob("app-*.json")):
        if path.name == "app-summary.json":
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        requests = record.get("translations", [])
        row = {key: record.get(key) for key in ("profile", "attempt", "status", "error", "scope", "controlled_boundary", "source_import_identity", "server_command", "host_release", "server_shutdown", "server_exit_code", "resource_sample_count", "resource_peaks", "udp_sink", "placement_cache_log")}
        row["completed_translations"] = len(requests)
        row["resource_units"] = {owner: {key: {"bytes": value, "GB": value/1e9, "GiB": value/2**30} for key,value in peak.items() if key.endswith("bytes") or key in {"WorkingSetSize", "PeakWorkingSetSize", "PrivateUsage", "PeakPagefileUsage"}} for owner,peak in record.get("resource_peaks", {}).items()}
        row["completion_wall_seconds"] = distribution([request["completion_wall_seconds"] for request in requests])
        row["cycles"] = {str(cycle): distribution([request["completion_wall_seconds"] for request in requests if request["cycle"] == cycle]) for cycle in sorted({request["cycle"] for request in requests})}
        row["degraded_settings_count"] = len(record.get("controlled_degraded_operations", []))
        row["degraded_settings_codes"] = sorted({operation.get("transaction", {}).get("diagnostics", {}).get("code") for operation in record.get("controlled_degraded_operations", [])})
        row["initial_settings_by_fresh_instance"] = []
        row["instances"] = [{key: instance.get(key) for key in ("instance_id", "host_pid", "translation_enabled", "translation_runtime_ready")} for instance in record.get("instances", [])]
        waiting = False
        for command in record.get("commands", []):
            if command["arguments"][:2] == ["app", "start"]:
                waiting = True
            if waiting and command["arguments"] == ["settings", "current"]:
                snapshot = command["result"]
                row["initial_settings_by_fresh_instance"].append({"instance_id": snapshot["instance_id"], "revision": snapshot["revision"], "languages": snapshot["settings"]["intent"]["languages"], "translation": snapshot["settings"]["intent"]["translation"], "endpoint": snapshot["settings"]["intent"]["local_llm"]})
                waiting = False
        profile = record["profile"]
        attempt = record.get("attempt", 1)
        raw = args.raw_root / "app" / f"{profile}-attempt{attempt}"
        if (raw / "server.log").exists():
            log = (raw / "server.log").read_text(encoding="utf-8", errors="replace")
            timings = []
            pending = None
            for line in log.splitlines():
                prompt = re.search(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens", line)
                generation = re.search(r"(?<!prompt )eval time =\s*([\d.]+) ms /\s*(\d+) tokens.*?([\d.]+) tokens per second", line)
                if prompt:
                    pending = {"prompt_ms": float(prompt[1]), "evaluated_prompt_tokens": int(prompt[2])}
                elif generation and pending is not None:
                    pending.update(generation_ms=float(generation[1]), generation_eval_tokens=int(generation[2]), generation_tokens_per_second=float(generation[3]))
                    timings.append(pending)
                    pending = None
            row["server_timing_records"] = timings
            row["server_timing_count_matches_completions"] = len(timings) == len(requests)
            row["timing_attribution"] = "Ordered native-server timing records for sequential synthetic requests; no direct native task-ID to app turn-UUID link is claimed."
            row["evaluated_prompt_tokens_total"] = sum(t["evaluated_prompt_tokens"] for t in timings)
            row["generation_eval_tokens_total"] = sum(t["generation_eval_tokens"] for t in timings)
            row["generation_tokens_per_second"] = distribution([t["generation_tokens_per_second"] for t in timings])
        recovery_path = args.results / "managed-e4b-recovery.json"
        if profile == "managed-e4b-stock" and recovery_path.exists():
            recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
            native = recovery["native_observations"]
            row.update(original_capture_status=record["status"], status=recovery["status"], managed_recovery_file=recovery_path.name, managed_servers=record.get("managed_servers"), preparation_windows=recovery["preparation_windows"])
            row["server_timing_records"] = native
            row["server_timing_count_matches_completions"] = len(native) == len(requests)
            row["timing_attribution"] = "Structured native-response diagnostics correlated by host PID/request ID to completed utterance UUID; original legacy-performance capture classification retained. Stock native stderr unavailable."
            row["evaluated_prompt_tokens_total"] = sum(t["prompt_evaluated_tokens"] for t in native)
            row["generation_eval_tokens_total"] = None
            row["generation_counter_unavailable_reason"] = "Stock managed stderr DEVNULL; API predicted-output count is distinct from a physical decode-evaluation counter."
            row["generation_tokens_per_second"] = distribution([t["decode_tps"] for t in native])
        sample_path = raw / "samples.json"
        if sample_path.exists():
            samples = json.loads(sample_path.read_text(encoding="utf-8"))
            first = samples[0]["time"] if samples else 0
            app_cpu_segments = []
            segment = None
            for sample in samples:
                if sample.get("app") is not None:
                    segment = max(segment or 0, sample["app"].get("cpu_seconds", 0))
                elif segment is not None:
                    app_cpu_segments.append(segment)
                    segment = None
            if segment is not None:
                app_cpu_segments.append(segment)
            row["app_cpu_seconds_per_observed_instance"] = app_cpu_segments
            row["app_cpu_seconds_total_sampled"] = sum(app_cpu_segments)
            row["observed_wall_seconds"] = samples[-1]["time"]-first if samples else None
            row["server_cpu_seconds_sampled"] = max(s["server"].get("cpu_seconds", 0) for s in samples if s.get("server")) if any(s.get("server") for s in samples) else None
            server_segments, server_segment = [], None
            for sample in samples:
                if sample.get("server") is not None:
                    server_segment = max(server_segment or 0, sample["server"].get("cpu_seconds", 0))
                elif server_segment is not None:
                    server_segments.append(server_segment)
                    server_segment = None
            if server_segment is not None:
                server_segments.append(server_segment)
            row["server_cpu_seconds_per_observed_instance"] = server_segments
            row["server_cpu_seconds_total_sampled"] = sum(server_segments)
            bins = {}
            for sample in samples:
                bins.setdefault(int(sample["time"]-first), []).append(sample)
            compact = []
            for second, group in bins.items():
                item = {"second": second, "n": len(group)}
                for owner in ("server", "app"):
                    keys = {key for sample in group if sample.get(owner) for key in sample[owner] if key != "time"}
                    item[owner] = {}
                    for key in keys:
                        values = [sample[owner][key] for sample in group if sample.get(owner) and sample[owner].get(key) is not None]
                        item[owner][key + "_max"] = max(values) if values else None
                        if key == "system_available_physical_bytes":
                            item[owner][key + "_min"] = min(values) if values else None
                compact.append(item)
            compact_path = args.results / f"app-{profile}-attempt{attempt}.samples.jsonl"
            if not compact_path.exists():
                compact_path.write_text("".join(json.dumps(sample, separators=(",", ":")) + "\n" for sample in compact), encoding="utf-8")
            row["sample_cadence_seconds"] = distribution([b["time"]-a["time"] for a,b in zip(samples, samples[1:])])
            row["minimum_available_physical_bytes"] = min(s["server"]["system_available_physical_bytes"] for s in samples if s.get("server")) if any(s.get("server") for s in samples) else None
        rows.append(row)
    (args.results / "app-summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps([{key: row.get(key) for key in ("profile", "attempt", "status", "completed_translations", "completion_wall_seconds", "server_timing_count_matches_completions", "degraded_settings_count")} for row in rows]))


if __name__ == "__main__":
    main()
