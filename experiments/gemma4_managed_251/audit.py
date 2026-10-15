from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from summarize import maximum_active, native_intervals, safe_command, summarize_arm


def audit(path: Path):
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    events = [row for row in rows if row["kind"] in {"app_event", "supported_manual_event"}]
    translations = [row for row in events if row["event"].get("topic") == "translation"]
    published = {row["event"]["utterance_id"] for row in translations}
    intervals, unfinished = native_intervals(rows)
    observed_intervals = intervals + [
        {**interval, "native_end": float("inf")} for interval in unfinished
    ]
    native_requests = [row for row in rows if row["kind"] == "native_request"]
    completions = [
        row
        for row in rows
        if row["kind"] == "native_completion" and row["purpose"] == "translation"
    ]
    outputs = []
    for completion in completions:
        identity = completion["identity"]
        request = next(
            row
            for row in native_requests
            if row["started"] == completion["started"] and row["pid"] == completion["pid"]
        )
        outputs.append(
            {
                "identity": identity,
                "phase": completion["phase"],
                "pid": completion["pid"],
                "slot": completion["slot_id"],
                "rendered_user_message": request["messages"][-1]["content"],
                "content": completion["payload"]["content"],
                "published_translation": identity["utterance_id"] in published,
                "native_timings": completion["payload"]["timings"],
                "native_generation_settings": completion["payload"]["generation_settings"],
                "stop_type": completion["payload"].get("stop_type"),
                "truncated": completion["payload"].get("truncated"),
            }
        )
    cancellations = []
    for row in rows:
        if row["kind"] == "cancellation_requested":
            late = set(row["request_ids"]) & published
            cancellations.append({**row, "published_anywhere_in_final_journal": sorted(late)})
            assert not late, "Cancelled input published later in the completed journal"
    faults = [row for row in rows if row["kind"] == "injected_response_failure"]
    for fault in faults:
        assert fault["native_completed"]
        assert fault["identity"]["utterance_id"] not in published
    manual = []
    for row in rows:
        if row["kind"] == "cli" and row["arguments"][:2] == ["text", "submit"]:
            receipt = row["output"][-1]
            identities = set(receipt.get("child_turn_ids", [])) | {receipt.get("turn_id")}
            matches = [
                event["event"]
                for event in translations
                if event["event"]["utterance_id"] in identities
            ]
            manual.append(
                {
                    "receipt_status": receipt.get("status"),
                    "admission": receipt.get("admission"),
                    "turn_id": receipt.get("turn_id"),
                    "child_turn_ids": receipt.get("child_turn_ids"),
                    "receipt_output_delivery": receipt.get("output_delivery"),
                    "actual_translation_events": matches,
                }
            )
            assert (
                matches
            ), "Manual admission was not independently followed by translation completion"
    cpu_pids = {
        row["pid"]
        for row in rows
        if row["kind"] == "managed_spawn"
        and row["effective_command"][row["effective_command"].index("--device") + 1] == "none"
    }
    maxima = {
        str(pid): maximum_active(
            [interval for interval in observed_intervals if interval["pid"] == pid]
        )
        for pid in {interval["pid"] for interval in observed_intervals}
    }
    for pid, value in maxima.items():
        assert value <= (1 if int(pid) in cpu_pids else 2)
    for row in rows:
        if row["kind"] == "managed_spawn":
            assert row["live_owned_model_processes"] == 1
    readiness = [
        {
            "gemma": row["models"].get("gemma"),
            "translation_enabled": row["app"].get("translation_enabled"),
            "translation_runtime_ready": row["app"].get("translation_runtime_ready"),
            "native_readiness_captured": "native_readiness" in row,
            "native_readiness": row.get("native_readiness"),
        }
        for row in rows
        if row["kind"] == "no_fallback_state"
    ]
    candidate = next(
        (row["candidate"] for row in rows if row["kind"] in {"instance", "stock_candidate"}), {}
    )
    backend_returns = [row for row in rows if row["kind"] == "backend_return_outcome"]
    return_pairs = []
    if backend_returns:
        return_pairs = [
            pair
            for pair in summarize_arm(path)["pairs"]
            if pair["label"] in {"after-return-to-gpu", "after-return-to-gpu-repeat"}
        ]
        assert len(return_pairs) == 2
        assert all(
            (pair["server_active_overlap_s"] or 0) > 0 for pair in return_pairs
        ), "Repeated GPU-return requests did not overlap at the native server"
        shutdown = [row for row in rows if row["kind"] == "shutdown_waiting_manual_admission"]
        assert len(shutdown) == 1
        assert shutdown[0]["admission"]["status"] == "admitted"
        assert shutdown[0]["active_native_slots"] == 2
        assert shutdown[0]["pid"] == backend_returns[-1]["pid"]
    return {
        "raw_journal_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "candidate": {
            key: candidate.get(key) for key in ("head", "source_tree_sha256", "harness_sha256")
        },
        "record_counts": dict(Counter(row["kind"] for row in rows)),
        "native_active_maximum_by_pid": maxima,
        "native_unfinished_intervals_at_owned_shutdown": unfinished,
        "native_draft_tokens_by_cpu_pid": {
            str(pid): sum(
                row["payload"]["timings"].get("draft_n", 0)
                for row in completions
                if row["pid"] == pid
            )
            for pid in cpu_pids
        },
        "global_no_stale_cancelled_publication": cancellations,
        "manual_admission_and_completion": manual,
        "no_fallback_readiness": readiness,
        "gpu_return_native_pairs": return_pairs,
        "managed_spawn_attempts": [
            {
                "kind": row["kind"],
                "t": row["t"],
                "phase": row.get("phase"),
                "boundary": row.get("failure_boundary"),
                "original_command": safe_command(row["original_command"]),
                "effective_command": (
                    safe_command(row["effective_command"]) if "effective_command" in row else None
                ),
            }
            for row in rows
            if row["kind"] in {"managed_spawn_attempt", "injected_spawn_failure"}
        ],
        "controlled_outcomes": [
            {key: value for key, value in row.items() if key not in {"models", "survivor"}}
            for row in rows
            if row["kind"]
            in {
                "ordering_outcome",
                "cancellation_outcome",
                "response_failure_outcome",
                "manual_capacity_outcome",
                "shutdown_waiting_manual_admission",
                "supported_backend_recovery_requested",
                "supported_backend_recovery_prepared",
                "no_fallback_assertion",
                "explicit_retry_outcome",
                "backend_return_outcome",
            }
        ],
        "processes": [
            {
                "pid": row["pid"],
                "command": safe_command(row.get("effective_command", row.get("command"))),
                "executable_sha256": row["executable_sha256"],
            }
            for row in rows
            if row["kind"] in {"managed_spawn", "stock_managed_identity"}
        ],
        "cleanup": [
            row for row in rows if row["kind"] in {"cleanup", "shutdown_ports", "stock_cleanup"}
        ],
        "failures": [
            {key: row.get(key) for key in ("kind", "stage", "error", "detail")}
            for row in rows
            if row["kind"] in {"failure", "cleanup_failure"}
        ],
        "all_native_translation_outputs_including_rejected_response": outputs,
        "supported_translation_events": [
            row["event"] for row in rows if row["kind"] == "supported_manual_event"
        ],
        "destination_packets": [row for row in rows if row["kind"] == "osc_received"],
        "stock_hosts": [
            {
                "pid": row["pid"],
                "create_time": row["create_time"],
                "instance_id": row["instance_id"],
                "executable_matches_observed_source_python": Path(row["executable"]).resolve()
                == Path(candidate["python"]).resolve(),
                "launch_argv0_matches_observed_source_python": Path(row["command"][0]).resolve()
                == Path(candidate["python"]).resolve(),
                "os_executable_basename": Path(row["executable"]).name,
                "import_root_matches_observed_source": Path(row["source_import_root"]).resolve()
                == Path(candidate["import"]).resolve().parent.parent,
                "cwd_matches_candidate": Path(row["cwd"]).resolve()
                == Path(candidate["import"]).resolve().parents[2],
                "source_main_module_in_command": "puripuly_heart.main" in row["command"],
            }
            for row in rows
            if row["kind"] == "stock_host_identity"
        ],
        "stock_persisted_selection_reads": [
            {
                "t": row["t"],
                "model": row["output"][-1]["settings"]["intent"]["translation"]["model"],
                "connection": row["output"][-1]["settings"]["intent"]["translation"]["connection"],
            }
            for row in rows
            if row["kind"] == "cli" and row["arguments"] == ["settings", "current"]
        ],
        "limits": [
            "Native maxima include started-but-unfinished tasks as active until owned process termination; active shutdown also requires fresh native slot observations.",
            "Manual terminal admission alone is not destination delivery; actual translation events and separate OSC routing records remain distinct.",
            "No automated semantic quality verdict is assigned. Original synthetic native outputs, including injected rejected responses, are retained.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("journal", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.journal)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "counts": result["record_counts"],
                "failures": result["failures"],
            }
        )
    )
