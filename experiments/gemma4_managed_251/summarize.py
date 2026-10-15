from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


def distribution(values):
    values = sorted(
        value for value in values if isinstance(value, (int, float)) and math.isfinite(value)
    )
    if not values:
        return {"n": 0, "median": None, "p95": None, "max": None}
    return {
        "n": len(values),
        "median": statistics.median(values),
        "p95": values[max(0, math.ceil(len(values) * 0.95) - 1)],
        "max": values[-1],
        "min": values[0],
    }


def safe_command(command):
    result = []
    previous = None
    for argument in command:
        if previous in {"--model", "--spec-draft-model"}:
            result.append("<isolated-models>/" + Path(argument).name)
        elif previous == "--slot-save-path":
            result.append("<isolated-prefix-root>")
        elif not result:
            result.append(
                "<source-runtime>/" + Path(argument).parent.name + "/" + Path(argument).name
            )
        else:
            result.append(argument)
        previous = argument
    return result


def native_intervals(rows):
    active = {}
    intervals = []
    for row in rows:
        if row["kind"] != "native":
            continue
        line = row["line"]
        match = re.search(
            r"id\s+(\d+)\s*\|\s*task\s+(\d+)\s*\|\s*(processing task|stop processing)", line
        )
        if not match:
            continue
        stamp = re.match(r"\s*(\d+)\.(\d+)\.(\d+)\.(\d+)\s", line)
        native_t = None
        if stamp:
            minute, second, millis, micros = map(int, stamp.groups())
            native_t = minute * 60 + second + millis / 1000 + micros / 1000000
        key = (row["pid"], int(match[1]), int(match[2]))
        if match[3] == "processing task":
            active[key] = {
                "received_start": row["t"],
                "native_start": native_t,
                "pid": row["pid"],
                "slot": key[1],
                "task": key[2],
            }
        elif key in active:
            launch = active.pop(key)
            intervals.append({**launch, "received_end": row["t"], "native_end": native_t})
    return intervals, list(active.values())


def maximum_active(intervals):
    by_pid = defaultdict(list)
    for interval in intervals:
        if interval["native_start"] is not None and interval["native_end"] is not None:
            by_pid[interval["pid"]].extend(
                ((interval["native_start"], 1), (interval["native_end"], -1))
            )
    maxima = []
    for events in by_pid.values():
        active = maximum = 0
        for _stamp, delta in sorted(events):
            active += delta
            maximum = max(maximum, active)
        maxima.append(maximum)
    return max(maxima) if maxima else None


def summarize_arm(path: Path):
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    intervals, unreleased = native_intervals(rows)
    results = [row for row in rows if row["kind"] == "request_complete"]
    completions = [
        row
        for row in rows
        if row["kind"] == "native_completion" and row.get("purpose") == "translation"
    ]
    native_by_result = defaultdict(list)
    translation_intervals = []
    for completion in completions:
        identity = completion.get("identity") or {}
        key = identity.get("utterance_id")
        possible = [
            interval
            for interval in intervals
            if interval["slot"] == completion["slot_id"]
            and (completion.get("pid") is None or interval["pid"] == completion["pid"])
            and completion["started"] <= interval["received_start"] <= completion["completed"]
            and interval["received_end"] <= completion["completed"] + 0.1
        ]
        interval = max(possible, key=lambda item: item["received_start"]) if possible else None
        completion["matched_interval"] = interval
        if interval is not None:
            translation_intervals.append(interval)
        native_by_result[key].append(completion)
    request_rows = []
    for result in results:
        native = next(iter(native_by_result.get(result["result_id"], [])), None)
        payload = native["payload"] if native is not None else {}
        interval = native["matched_interval"] if native is not None else None
        http_payload = next(
            (
                row
                for row in rows
                if row["kind"] == "native_http_payload"
                and (row.get("context") or {}).get("purpose") == "translation"
                and ((row.get("context") or {}).get("identity") or {}).get("utterance_id")
                == result["result_id"]
            ),
            None,
        )
        request_rows.append(
            {
                "label": result["label"],
                "phase": result["phase"],
                "channel": result["channel"],
                "unit_id": result["unit_id"],
                "result_id": result["result_id"],
                "started": result["started"],
                "completed": result["completed"],
                "latency_ms": result["latency_ms"],
                "translation": result["translation"],
                "quality": result["quality"],
                "native_timings": payload.get("timings"),
                "native_generation_settings": payload.get("generation_settings"),
                "effective_completion_parameters": (
                    None if http_payload is None else http_payload["effective_parameters"]
                ),
                "original_completion_parameters": (
                    None if http_payload is None else http_payload["original_parameters"]
                ),
                "native_request_prompt_sha256": (
                    None if http_payload is None else http_payload["native_prompt_sha256"]
                ),
                "native_prompt_echo_matches": (
                    None if native is None else native.get("native_prompt_echo_matches")
                ),
                "native_prompt_echo_representation": (
                    None if native is None else native.get("native_prompt_echo_representation")
                ),
                "actual_model": payload.get("model"),
                "tokens_evaluated": payload.get("tokens_evaluated"),
                "tokens_cached": payload.get("tokens_cached"),
                "tokens_predicted": payload.get("tokens_predicted"),
                "native_template_prompt_sha256": (
                    hashlib.sha256(payload["prompt"].encode()).hexdigest()
                    if isinstance(payload.get("prompt"), str)
                    else None
                ),
                "stop": {
                    key: payload.get(key)
                    for key in (
                        "stop",
                        "stopped_eos",
                        "stopped_limit",
                        "stopped_word",
                        "stop_type",
                        "stopping_word",
                        "truncated",
                    )
                },
                "rendered_messages_sha256": (
                    None if native is None else native["rendered_messages_sha256"]
                ),
                "slot": None if native is None else native["slot_id"],
                "native_interval": interval,
            }
        )
    pairs = []
    for pair in (row for row in rows if row["kind"] == "pair_complete"):
        members = [
            next((request for request in request_rows if request["result_id"] == identity), None)
            for identity in pair["results"]
        ]
        overlap = None
        if all(member is not None and member["native_interval"] is not None for member in members):
            left, right = [member["native_interval"] for member in members]
            if all(
                item["native_start"] is not None and item["native_end"] is not None
                for item in (left, right)
            ):
                overlap = max(
                    0,
                    min(left["native_end"], right["native_end"])
                    - max(left["native_start"], right["native_start"]),
                )
        pairs.append(
            {
                "phase": pair["phase"],
                "label": pair["label"],
                "concurrent": pair["concurrent"],
                "makespan_ms": pair["makespan_ms"],
                "server_active_overlap_s": overlap,
                "result_ids": pair["results"],
            }
        )
    memory = defaultdict(list)
    for row in rows:
        if row["kind"] == "resource":
            memory[row["phase"]].append(row)
    memory_summary = {}
    for phase, samples in memory.items():
        keys = sorted(set(key for sample in samples for key in sample["sample"]) - {"time"})
        memory_summary[phase] = {
            key: distribution([sample["sample"].get(key) for sample in samples]) for key in keys
        }
        memory_summary[phase]["sample_count"] = len(samples)
        memory_summary[phase]["observed_sample_gaps_s"] = distribution(
            [right["t"] - left["t"] for left, right in zip(samples, samples[1:])]
        )
    instance = next((row for row in rows if row["kind"] == "instance"), {})
    candidate = instance.get("candidate", {})
    spawns = [row for row in rows if row["kind"] == "managed_spawn"]
    route_rows = [row for row in rows if row["kind"] == "routing_snapshot"]
    routes = route_rows[-1]["decisions"] if route_rows else []
    metric_summary = {}
    for phase in sorted(set(row["phase"] for row in request_rows)):
        requests = [row for row in request_rows if row["phase"] == phase]
        metric_summary[phase] = {
            "latency_ms": distribution([row["latency_ms"] for row in requests]),
            "native": {
                key: distribution([(row.get("native_timings") or {}).get(key) for row in requests])
                for key in (
                    "prompt_n",
                    "cache_n",
                    "predicted_n",
                    "prompt_ms",
                    "predicted_ms",
                    "prompt_per_second",
                    "predicted_per_second",
                    "draft_n",
                    "draft_n_accepted",
                )
            },
        }
    event_counts = defaultdict(int)
    for row in rows:
        if row["kind"] == "app_event":
            event_counts[row["event"].get("topic", "unknown")] += 1
    return {
        "arm": path.parent.name,
        "raw_journal_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "candidate": {
            key: candidate.get(key) for key in ("head", "python_version", "source_tree_sha256")
        },
        "source_hashes": candidate.get("source_sha256"),
        "harness_hashes": candidate.get("harness_sha256"),
        "asset_manifest_sha256": instance.get("assets_manifest_sha256"),
        "profile_contracts": [row for row in rows if row["kind"] == "profile_contract"],
        "prefix_snapshots": [row for row in rows if row["kind"] == "prefix_snapshot"],
        "native_http_payloads": [row for row in rows if row["kind"] == "native_http_payload"],
        "native_identity": [
            row["line"]
            for row in rows
            if row["kind"] == "native"
            and re.search(
                r"build 11537|using device|ggml_vulkan:|offloaded \d+/\d+|MoE cache size|I impl:.*layers|buffer size =|llama_context:|using full-size SWA|prompt cache is disabled|threadpool init|add_bos_token",
                row["line"],
                re.I,
            )
        ],
        "processes": [
            {
                "pid": row["pid"],
                "parent_pid": row["parent_pid"],
                "original_command": safe_command(row["original_command"]),
                "effective_command": safe_command(row["effective_command"]),
                "executable_sha256": row["executable_sha256"],
                "live_owned_model_processes": row["live_owned_model_processes"],
            }
            for row in spawns
        ],
        "completed": len(results),
        "requests": request_rows,
        "phase_metrics": metric_summary,
        "pairs": pairs,
        "native_max_active_all_tasks": maximum_active(intervals),
        "native_max_active_translation_tasks": maximum_active(translation_intervals),
        "native_interval_count": len(intervals),
        "native_unreleased_intervals": unreleased,
        "process_shutdown_events": [
            row for row in rows if row["kind"] in {"owned_terminate", "owned_kill", "owned_exit"}
        ],
        "sampled_max_active_slots": max(
            (
                sum(bool(slot.get("is_processing")) for slot in row["slots"])
                for row in rows
                if row["kind"] == "slots"
            ),
            default=None,
        ),
        "memory_by_phase": memory_summary,
        "prelaunch_system_baselines": [
            row for row in rows if row["kind"] == "prelaunch_system_baseline"
        ],
        "resource_samples": [
            {key: row.get(key) for key in ("t", "phase", "pid", "adapter", "sample")}
            for row in rows
            if row["kind"] == "resource"
        ],
        "adapters": [row for row in rows if row["kind"] == "adapter_discovered"],
        "counter_failures": [
            row for row in rows if row["kind"] in {"counter_failure", "counter_setup"}
        ],
        "app_event_counts": dict(event_counts),
        "controlled_outcomes": [
            row
            for row in rows
            if row["kind"]
            in {
                "cancellation_requested",
                "cancellation_outcome",
                "response_failure_outcome",
                "manual_capacity_outcome",
                "profile_contract",
                "ordering_outcome",
                "no_fallback_assertion",
                "explicit_retry_outcome",
                "supported_backend_recovery_requested",
                "supported_backend_recovery_prepared",
                "backend_return_outcome",
                "shutdown_waiting_manual_admission",
                "stock_cleanup",
            }
        ],
        "inactive_stt_degradations": sum(
            row["kind"] == "known_inactive_stt_degradation" for row in rows
        ),
        "routing_summary": {
            "delivery_health": route_rows[-1].get("delivery_health") if route_rows else None,
            "peer_chatbox_denied": sum(row["reason"] == "peer_chatbox_denied" for row in routes),
            "by_decision": dict(
                (decision, sum(row["decision"] == decision for row in routes))
                for decision in sorted(set(row["decision"] for row in routes))
            ),
        },
        "cleanup": [row for row in rows if row["kind"] in {"cleanup", "shutdown_ports"}],
        "destination_evidence": {
            "routing_decisions": routes,
            "translation_events": [
                row
                for row in rows
                if row["kind"] == "app_event" and row["event"].get("topic") == "translation"
            ],
            "osc_sent_events": [
                row
                for row in rows
                if row["kind"] == "app_event" and row["event"].get("topic") == "osc_sent"
            ],
            "sink_messages": [row for row in rows if row["kind"] == "osc_received"],
        },
        "failures": [
            {"kind": row["kind"], "error": row.get("error"), "stage": row.get("stage")}
            for row in rows
            if row["kind"] in {"failure", "fatal", "cleanup_failure", "watchdog_cutoff"}
        ],
        "injected_failures": [
            {
                "kind": row["kind"],
                "boundary": row.get("failure_boundary"),
                "identity": row.get("identity"),
            }
            for row in rows
            if row["kind"] in {"injected_spawn_failure", "injected_response_failure"}
        ],
    }


def summarize(root: Path, output: Path | None = None):
    journals = sorted(root.rglob("events.jsonl"))
    report = {
        "schema": "puripuly-heart/managed-gemma-251/v1",
        "arms": [summarize_arm(path) for path in journals],
        "execution_outcomes": (
            json.loads((root / "outcomes.json").read_text(encoding="utf-8"))
            if (root / "outcomes.json").is_file()
            else None
        ),
        "limits": [
            "Controlled synthetic recognition ingress; no live audio, capture activation, ASR or consent proof.",
            "Peer effective-state projection is a disclosed capture-admission substitution.",
            "Comparison uses fixed synthetic source history seeded only between settled groups.",
            "Ordinary smoke retains product random sampling. ABBA alone fixes seed2512026; all other product sampling/output limits are retained and effective native parameters recorded.",
            "Memory maxima are sampled highwater lower bounds. Private commitment is not physical RAM.",
            "Process and device-wide GPU counters are separate. Shared GPU bytes do not by themselves establish spill.",
            "Headless event completion/OSC receiver observations do not prove GUI/VRChat presentation.",
        ],
    }
    target = output or root / "summary.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = summarize(args.root, args.output)
    print(
        json.dumps(
            {
                "arms": [
                    {"arm": arm["arm"], "completed": arm["completed"], "failures": arm["failures"]}
                    for arm in report["arms"]
                ]
            }
        )
    )
