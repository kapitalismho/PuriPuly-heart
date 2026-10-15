from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

MIB = 1024**2


def distribution(values):
    values = sorted(value for value in values if isinstance(value, (int, float)))
    return (
        None
        if not values
        else {
            "n": len(values),
            "median": statistics.median(values),
            "p95": values[max(0, math.ceil(0.95 * len(values)) - 1)],
            "max": max(values),
        }
    )


def stage(phase):
    if phase == "model-load":
        return "loading"
    if phase == "startup:first-prefix":
        return "startup_first_prefix"
    if "first-prefix" in phase:
        return "language_pair_first_prefix"
    if "transition" in phase:
        return "transition"
    if phase == "shutdown":
        return "shutdown"
    return "active_or_settled_request_group"


def intervals(samples, predicate, start, end):
    groups, group = [], []
    previous_above = start
    for row in samples:
        stamp = row["sample"]["time"]
        if predicate(row["sample"]):
            group.append(row)
        else:
            if group:
                groups.append((group, previous_above, stamp))
                group = []
            previous_above = stamp
    if group:
        groups.append((group, previous_above, end))
    runs = []
    for group, preceding, following in groups:
        times = [row["sample"]["time"] for row in group]
        runs.append(
            {
                "first_low_sample_s": times[0],
                "last_low_sample_s": times[-1],
                "sample_supported_span_s": times[-1] - times[0],
                "low_samples": len(times),
                "largest_within_run_gap_s": max(
                    (b - a for a, b in zip(times, times[1:])), default=0
                ),
                "preceding_above_or_lifetime_boundary_s": preceding,
                "following_above_or_lifetime_boundary_s": following,
                "outer_bracket_s": following - preceding,
                "phases": sorted(set(row["phase"] for row in group)),
                "stages": sorted(set(stage(row["phase"]) for row in group)),
            }
        )
    above = (
        [start] + [row["sample"]["time"] for row in samples if not predicate(row["sample"])] + [end]
    )
    return {
        "low_samples": sum(run["low_samples"] for run in runs),
        "longest_sample_supported_run": max(
            runs, key=lambda run: run["sample_supported_span_s"], default=None
        ),
        "maximum_possible_continuous_low_window_between_above_observations_s": max(
            (b - a for a, b in zip(above, above[1:])), default=0
        ),
        "runs": runs,
    }


def analyze(path):
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    processes = []
    for spawn in (row for row in rows if row["kind"] == "managed_spawn"):
        pid = spawn["pid"]
        end = next(row["t"] for row in rows if row["kind"] == "owned_exit" and row["pid"] == pid)
        samples = sorted(
            (
                row
                for row in rows
                if row["kind"] == "resource"
                and row["pid"] == pid
                and spawn["t"] <= row["sample"]["time"] <= end
            ),
            key=lambda row: row["sample"]["time"],
        )
        command = spawn["effective_command"]
        if not samples:
            processes.append({"pid": pid, "missing_live_samples": True})
            continue
        times = [row["sample"]["time"] for row in samples]
        minimum = min(samples, key=lambda row: row["sample"]["system_available_physical_bytes"])
        first = samples[0]
        prelaunch = next(
            (
                row
                for row in reversed(rows)
                if row["kind"] == "prelaunch_system_baseline" and row["t"] < spawn["t"]
            ),
            None,
        )
        physical = {
            str(limit): intervals(
                samples,
                lambda sample, limit=limit: sample["system_available_physical_bytes"] < limit * MIB,
                spawn["t"],
                end,
            )
            for limit in (256, 512)
        }
        watchdog = intervals(
            samples,
            lambda sample: (
                sample["system_available_physical_bytes"] < 256 * MIB
                or sample["system_commit_headroom_bytes"] < 512 * MIB
            ),
            spawn["t"],
            end,
        )
        phases = {}
        for phase in sorted(set(row["phase"] for row in samples)):
            values = [row for row in samples if row["phase"] == phase]
            phases[phase] = {
                "stage": stage(phase),
                "sample_count": len(values),
                "minimum_available_physical_bytes": min(
                    row["sample"]["system_available_physical_bytes"] for row in values
                ),
                "minimum_commit_headroom_bytes": min(
                    row["sample"]["system_commit_headroom_bytes"] for row in values
                ),
            }
        accounting = {}
        for key in (
            "WorkingSetSize",
            "PrivateUsage",
            "process_gpu_dedicated_bytes",
            "process_gpu_shared_bytes",
            "device_gpu_dedicated_bytes",
            "device_gpu_shared_bytes",
            "system_pages_input_per_second",
            "system_page_reads_per_second",
        ):
            accounting[key] = distribution([row["sample"].get(key) for row in samples])
        processes.append(
            {
                "pid": pid,
                "model_filename": Path(command[command.index("--model") + 1]).name,
                "backend": command[command.index("--device") + 1],
                "spawn_s": spawn["t"],
                "exit_s": end,
                "samples": len(samples),
                "live_window_sample_gaps_s": distribution(
                    [b - a for a, b in zip(times, times[1:])]
                ),
                "first_sample_delay_after_spawn_s": times[0] - spawn["t"],
                "last_sample_to_exit_gap_s": end - times[-1],
                "prelaunch_baseline": {
                    "captured": prelaunch is not None,
                    "observation": prelaunch,
                    "reason": (
                        None
                        if prelaunch is not None
                        else "Monitor starts after native spawn; no prelaunch observation is reconstructed."
                    ),
                },
                "earliest_postspawn_not_prelaunch": {
                    "time_s": times[0],
                    "phase": first["phase"],
                    "available_physical_bytes": first["sample"]["system_available_physical_bytes"],
                    "commit_headroom_bytes": first["sample"]["system_commit_headroom_bytes"],
                    "committed_bytes": first["sample"]["system_committed_bytes"],
                    "commit_limit_bytes": first["sample"]["system_commit_limit_bytes"],
                },
                "physical_minimum": {
                    "time_s": minimum["sample"]["time"],
                    "phase": minimum["phase"],
                    "stage": stage(minimum["phase"]),
                    "sample": minimum["sample"],
                },
                "below_physical_mib": physical,
                "configured_watchdog_predicate": watchdog,
                "watchdog_cutoff_events": [row for row in rows if row["kind"] == "watchdog_cutoff"],
                "accounting_and_global_paging": accounting,
                "by_phase": phases,
            }
        )
    return {
        "raw_journal_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "processes": processes,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("journals", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "schema": "puripuly-heart/managed-gemma-251-pressure/v1",
        "runs": {path.parent.name: analyze(path) for path in args.journals},
        "method": [
            "Only samples inside each recorded native PID spawn-to-exit window are attributed to that PID; postexit monitor samples are excluded.",
            "Sample-supported spans are last-low minus first-low with every recorded intervening sample low; they are not proof of continuous pressure between discrete observations.",
            "Outer brackets and maximum possible low windows use surrounding above-threshold observations or process-lifetime boundaries. Sampling gaps are explicit; threshold crossings can occur unseen.",
            "The configured watchdog is physical available<256MiB OR commit headroom<512MiB for8s. Physical<512MiB is an additional analysis threshold, not that watchdog's physical threshold.",
            "Global paging/device counters include other workloads. Private commitment is not physical residency; sharedGPU accounting does not diagnose spill/OOM/RAM safety.",
        ],
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                key: [
                    {
                        "pid": p["pid"],
                        "minimum": p.get("physical_minimum", {})
                        .get("sample", {})
                        .get("system_available_physical_bytes"),
                        "longest_low256_s": (
                            p.get("below_physical_mib", {})
                            .get("256", {})
                            .get("longest_sample_supported_run")
                            or {}
                        ).get("sample_supported_span_s"),
                        "longest_low512_s": (
                            p.get("below_physical_mib", {})
                            .get("512", {})
                            .get("longest_sample_supported_run")
                            or {}
                        ).get("sample_supported_span_s"),
                    }
                    for p in value["processes"]
                ]
                for key, value in result["runs"].items()
            }
        )
    )
