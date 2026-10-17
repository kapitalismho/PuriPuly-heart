from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from evidence_support import HERE, digest, save


def summarize_native(trace):
    events = trace["events"]
    rows = []
    for turn_id in dict.fromkeys(
        event["snapshot"]["utterance_id"]
        for event in events
        if event["event"] == "presentation_snapshot"
    ):
        snapshots = [
            event
            for event in events
            if event["event"] == "presentation_snapshot"
            and event["snapshot"]["utterance_id"] == turn_id
        ]
        closed = next((event for event in snapshots if event["snapshot"]["source_closed"]), None)
        ready = next(
            (
                event
                for event in snapshots
                if any(target["status"] == "ready" for target in event["snapshot"]["targets"])
            ),
            None,
        )
        commits = [
            event
            for event in events
            if event["event"] in ("commit_requested", "source_committed")
            and event["state"]["merge_id"] == turn_id
        ]
        entry = next((event for event in commits if event["event"] == "commit_requested"), None)
        returned = next((event for event in commits if event["event"] == "source_committed"), None)
        pages = [
            event
            for event in events
            if event["event"] == "chatbox_stage"
            and event["stage"] == "page_send"
            and event["utterance_id"] == turn_id
        ]
        packet_receipts = []
        for page in pages:
            sent = next(
                (
                    event
                    for event in reversed(events[: events.index(page)])
                    if event["event"] == "udp_send_returned"
                    and event["address"] == "/chatbox/input"
                ),
                None,
            )
            if sent is not None:
                receipt = next(
                    (
                        event
                        for event in events
                        if event["event"] == "udp_receiver_packet"
                        and event["at_s"] >= sent["at_s"]
                        and any(
                            message["address"] == sent["address"]
                            and message["params"] == sent["values"]
                            for message in event["messages"]
                        )
                    ),
                    None,
                )
                if receipt is not None:
                    packet_receipts.append(receipt)
        terminal = next(
            (event for event in reversed(snapshots) if event["snapshot"]["status"] != "active"),
            None,
        )
        active_calls = []
        if entry:
            for call in (
                event
                for event in events
                if event["event"] == "llm_invoked"
                and event["utterance_id"] == turn_id
                and event["at_s"] < entry["at_s"]
            ):
                end = next(
                    (
                        event
                        for event in events
                        if event["event"] in ("llm_result_ready", "llm_cancelled")
                        and event["invocation"] == call["invocation"]
                    ),
                    None,
                )
                if end is None or end["at_s"] > entry["at_s"]:
                    active_calls.append(
                        {
                            "invocation": call["invocation"],
                            "invoked_at_s": call["at_s"],
                            "terminal_event": None if end is None else end["event"],
                            "terminal_at_s": None if end is None else end["at_s"],
                        }
                    )
        final_target_ids = (
            []
            if terminal is None
            else [
                target["translation"]["update_id"]
                for target in terminal["snapshot"]["targets"]
                if target.get("translation")
            ]
        )
        ready_before_close_ids = (
            []
            if entry is None
            else [
                target["translation"]["update_id"]
                for event in snapshots
                if event["at_s"] < entry["at_s"]
                for target in event["snapshot"]["targets"]
                if target.get("translation")
            ]
        )
        rows.append(
            {
                "turn_id": turn_id,
                "turn_order": snapshots[0]["snapshot"]["turn_order"],
                "source_text": snapshots[-1]["snapshot"]["source_text"],
                "first_usable_snapshot_at_s": None if ready is None else ready["at_s"],
                "source_close_request_at_s": None if entry is None else entry["at_s"],
                "source_close_return_at_s": None if returned is None else returned["at_s"],
                "source_closed_snapshot_at_s": None if closed is None else closed["at_s"],
                "first_local_page_transmission_at_s": None if not pages else pages[0]["at_s"],
                "local_page_transmissions": len(pages),
                "local_send_before_source_close_request": bool(
                    entry and pages and pages[0]["at_s"] < entry["at_s"]
                ),
                "llm_calls_running_at_source_close_request": active_calls,
                "terminal_presentation_status": (
                    None if terminal is None else terminal["snapshot"]["status"]
                ),
                "final_result_update_ids": final_target_ids,
                "final_promotes_prior_result_identity": bool(final_target_ids)
                and all(identity in ready_before_close_ids for identity in final_target_ids),
            }
        )
        rows[-1].update(
            {
                "first_loopback_packet_received_at_s": (
                    None if not packet_receipts else packet_receipts[0]["at_s"]
                ),
                "loopback_packet_before_source_close_request": bool(
                    entry and packet_receipts and packet_receipts[0]["at_s"] < entry["at_s"]
                ),
                "receiver_correlation": "Delegated synchronous successful UDP send values preceding each paginator page-send receipt matched to actual received address/params; not a mock sender",
            }
        )
        rows[-1].update(
            {
                "page_transmissions_marked_preview": sum(
                    event.get("self_turn_final") is False for event in pages
                ),
                "page_transmissions_marked_final": sum(
                    event.get("self_turn_final") is True for event in pages
                ),
                "page_transmissions_without_finality_metadata": sum(
                    event.get("self_turn_final") is None for event in pages
                ),
                "final_transmission_metric_boundary": "Identical final promotion may retain its earlier preview send relationship; zero fresh final packets is not fabricated as a new final transmission",
            }
        )
    received = [
        event
        for event in events
        if event["event"] == "udp_receiver_packet"
        and any(message["address"] == "/chatbox/input" for message in event["messages"])
    ]
    return {
        "counts": trace["counts"],
        "observations": trace["observations"],
        "source_accounting": trace["source_accounting"],
        "overlay_enabled": trace["isolation"]["overlay_enabled"],
        "effective_grace_ms": trace["effective_translation_profile"]["value"][
            "low_latency_finalize_wait_ms"
        ],
        "turns": rows,
        "loopback_chatbox_packets": [
            {
                "received_at_s": event["at_s"],
                "packet_sha256": event["packet_sha256"],
                "messages": event["messages"],
            }
            for event in received
        ],
        "limits": [
            "Source close is bracketed by delegated commit entry/return, not a fabricated exact internal timestamp",
            "Local transmission and owned UDP reception are separate observations; neither is physical VRChat/HMD display",
            "Fresh native schedules are not controlled same-ASR causal timing comparisons",
            "Zero stable updates means final-only local ASR, not stable-contribution route coverage",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if any(Path(name).name != name for name in (*args.reports, args.output)):
        parser.error("Reports must be task-local filenames")
    target = HERE / args.output
    if target.exists():
        parser.error("Retained summary exists; choose a new filename")
    result = {"schema": "self-stream-233/runtime-summary/v1", "reports": []}
    for name in args.reports:
        path = HERE / name
        trace = json.loads(path.read_text(encoding="utf-8"))
        row = {
            "file": name,
            "sha256": digest(path),
            "status": trace["status"],
            "revision": trace["revision"],
            "candidate_stable_during_run": trace["candidate_stable_during_run"],
            "source_file_count": len(trace["source_files"]),
            "source_files": trace["source_files"],
            "trace_export": trace["trace_export"],
        }
        row["product_python_manifest_sha256"] = hashlib.sha256(
            json.dumps(trace["source_files"], sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if "production-synthetic-audio" in trace["schema"]:
            row.update(summarize_native(trace))
        else:
            row.update(
                {
                    "instance_id": trace.get("readiness", {}).get("instance_id"),
                    "host_identity": trace.get("host_identity"),
                    "operations": [
                        {
                            "arguments": operation["arguments"],
                            "exit_code": operation["exit_code"],
                            "terminal": (operation.get("payload") or {}).get("terminal"),
                            "status": (operation.get("payload") or {}).get("status"),
                        }
                        for operation in trace["operations"]
                    ],
                    "manual_text_admission": trace.get("source_only_text_admission"),
                    "loopback_packets": trace.get("loopback_packets"),
                    "host_exit_code": trace.get("host_exit_code"),
                    "cleanup": trace.get("cleanup"),
                }
            )
        result["reports"].append(row)
    result["same_product_python_fingerprint_across_reports"] = (
        len({row["product_python_manifest_sha256"] for row in result["reports"]}) == 1
    )
    result["fingerprint_definition"] = (
        "SHA-256 of UTF-8 JSON source_files map with sort_keys=True and separators=(',', ':'); each value is the executed product .py file SHA-256"
    )
    save(target, result)
    print(json.dumps({"output": args.output, "reports": len(result["reports"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
