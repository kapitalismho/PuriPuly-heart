from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def distribution(values):
    ordered = sorted(values)
    if not ordered:
        return {"samples": 0}
    return {
        "samples": len(ordered),
        "min_ms": min(ordered),
        "median_ms": statistics.median(ordered),
        "mean_ms": statistics.mean(ordered),
        "max_ms": max(ordered),
        "values_ms": ordered,
    }


def main():
    baseline = json.loads((HERE / "baseline_translation.json").read_text(encoding="utf-8"))
    comparison = json.loads(
        (HERE / "recorded_schedule_comparison.json").read_text(encoding="utf-8")
    )
    events = baseline["events"]
    origin = baseline["clock"]["origin_monotonic_s"]
    starts = {event["label"]: event for event in events if event["event"] == "vad_start"}
    ends = {event["label"]: event for event in events if event["event"] == "vad_end"}
    terminals = {event["label"]: event for event in events if event["event"] == "asr_terminal"}
    native_starts = {
        event["label"]: event for event in events if event["event"] == "native_asr_invoked"
    }
    native_results = {
        event["label"]: event for event in events if event["event"] == "native_asr_result_ready"
    }
    commits = [event for event in events if event["event"] == "source_committed"]
    requests = {
        event["state"]["merge_id"]: event
        for event in events
        if event["event"] == "commit_requested"
    }
    source_accounting = Counter(
        label for commit in commits for label in commit["state"]["acoustic_labels"]
    )
    episode_rows = []
    for entry in baseline["inputs"]["episodes"]:
        label = entry["label"]
        commit = next(item for item in commits if label in item["state"]["acoustic_labels"])
        native_start = native_starts[label]
        native_result = native_results[label]
        seal_at = ends[label]["source"]["sealed_at_monotonic_s"] - origin
        row = {
            "label": label,
            "input_sha256": entry["sha256"],
            "frozen_synthetic_text": entry["phrase"],
            "asr_text": terminals[label]["payload"]["text"],
            "audio_samples": entry["samples"],
            "audio_seconds": entry["seconds"],
            "source_opened_at_s": starts[label]["source"]["opened_at_monotonic_s"] - origin,
            "vad_start_receipt_at_s": starts[label]["at_s"],
            "vad_seal_at_s": seal_at,
            "vad_end_receipt_at_s": ends[label]["at_s"],
            "native_invoked_at_s": native_start["at_s"],
            "native_result_ready_at_s": native_result["at_s"],
            "scoped_terminal_receipt_at_s": terminals[label]["at_s"],
            "identity": terminals[label]["payload"]["identity"],
            "merge_id": commit["state"]["merge_id"],
            "publication_ids": commit["state"]["publication_ids"],
            "source_closure_observation_bracket_at_s": [
                requests[commit["state"]["merge_id"]]["at_s"],
                commit["at_s"],
            ],
            "native_decode_ms": (native_result["at_s"] - native_start["at_s"]) * 1000,
            "seal_to_terminal_receipt_ms": (terminals[label]["at_s"] - seal_at) * 1000,
            "seal_to_commit_return_ms": (commit["at_s"] - seal_at) * 1000,
            "accounted_publications": source_accounting[label],
        }
        episode_rows.append(row)
    commit_rows = []
    for commit in commits:
        state = commit["state"]
        other_pending = [
            label
            for label in starts
            if label not in state["acoustic_labels"]
            and starts[label]["at_s"]
            < requests[state["merge_id"]]["at_s"]
            < terminals[label]["at_s"]
        ]
        other_active = [label for label in other_pending if commit["at_s"] < ends[label]["at_s"]]
        commit_rows.append(
            {
                "at_s": commit["at_s"],
                "merge_id": state["merge_id"],
                "acoustic_labels": state["acoustic_labels"],
                "text": state["text"],
                "attempt": state["attempt"],
                "other_pending_acoustics": other_pending,
                "other_active_acoustics": other_active,
                "continuation_invariant_status": "failed" if other_pending else "passed",
            }
        )
    replay_rows = []
    for row in comparison["results"]:
        replay_rows.append(
            {
                "case": row["case"],
                "policy": row["policy"],
                "llm_delay_s": row["llm_delay_s"],
                "source_inputs": row["source_inputs"],
                "commits": row["commits"],
                "continuation_invariant_status": row["required_continuation_status"],
                "premature_commit_count": len(row["premature_commits"]),
                "grace_only_wait_ms": row["grace_only_wait_ms"],
                "llm_invocations": len(row["llm_invocations"]),
                "physical_active_at_observation_end": row["physical_active_at_observation_end"],
                "metrics": row["metrics"],
            }
        )
    changes = []
    for row in replay_rows:
        if row["policy"] not in ("elapsed", "combined"):
            continue
        unchanged = next(
            item
            for item in replay_rows
            if item["case"] == row["case"] and item["policy"] == "unchanged"
        )
        same_grouping = [item["acoustics"] for item in row["commits"]] == [
            item["acoustics"] for item in unchanged["commits"]
        ]
        changes.append(
            {
                "case": row["case"],
                "policy": row["policy"],
                "same_grouping": same_grouping,
                "grace_only_wait_reduction_ms": unchanged["grace_only_wait_ms"]
                - row["grace_only_wait_ms"],
                "commit_advance_ms": (
                    [
                        (old["at_s"] - new["at_s"]) * 1000
                        for old, new in zip(unchanged["commits"], row["commits"], strict=True)
                    ]
                    if same_grouping
                    else None
                ),
            }
        )
    report = {
        "schema": "self-grace-233/synthetic-audio-summary/v1",
        "revision": baseline["revision"],
        "artifacts": {
            name: sha256(HERE / name)
            for name in (
                "inputs.json",
                "baseline_translation.json",
                "baseline_translation_asr_schedule.json",
                "recorded_schedule_comparison.json",
                "run_production_audio.py",
                "replay_recorded_schedule.py",
            )
        },
        "fresh_asr_counts": baseline["counts"],
        "source_accounting": baseline["source_accounting"],
        "input_audio_seconds": sum(entry["seconds"] for entry in baseline["inputs"]["episodes"]),
        "episode_timelines": episode_rows,
        "source_commits": commit_rows,
        "fresh_asr_distributions": {
            key: distribution([row[key] for row in episode_rows])
            for key in (
                "native_decode_ms",
                "seal_to_terminal_receipt_ms",
                "seal_to_commit_return_ms",
            )
        },
        "real_grace_only_wait": "Not inferred from commit minus LLM readiness. Use the clock-coordinated shared fixture replay accounting, which excludes active or unresolved successors.",
        "replay_samples": len(replay_rows),
        "replay_rows": replay_rows,
        "replay_candidate_deltas": changes,
        "semantic_review": [
            {
                "case": "continuation",
                "status": "failed",
                "observation": "ASR inserts 'our' in after sunset; unchanged production also splits A from B and commits A while B is still active. Recognition quality and continuation-ordering limits are explicit.",
            },
            {
                "case": "number_negation",
                "status": "passed",
                "observation": "15 versus 50 and Not are retained; both contributions occur exactly once in one baseline unit. Translation meaning is not evaluated.",
            },
            {
                "case": "question",
                "status": "passed",
                "observation": "Statement then question text and question punctuation are retained; both contributions occur exactly once in one baseline unit. Translation meaning is not evaluated.",
            },
        ],
        "comparison_status": {
            "production_real_audio_final_only_ingress": "passed",
            "native_cpu_decode_and_model_hashes": "passed",
            "unchanged_continuation_obligation": "failed",
            "G1_frozen_real_ASR_schedule_replay": "passed",
            "G1_candidate_invariant_promotion": "blocked",
            "G2_fixed_real_ASR_schedule_LLM_timing": "passed",
            "fresh_candidate_ASR_runs": "not run",
            "stable_contribution_live_provider": "not run",
            "microphone_capture_and_VAD_detector": "not run",
            "private_audio_remote_provider_spending": "not run",
            "translation_semantic_quality": "not run",
            "physical_chatbox_GUI_HMD_delivery": "not run",
            "integrated_selected_candidate": "not run",
        },
        "limits": baseline["limits"] + comparison["limits"],
        "decision": "This worker supplies evidence, not policy adoption. Elapsed/combined replay advances the completed suffix/group commits with no grouping difference on this small frozen set; no-op has no effect here. All fast-policy replays retain the continuation failure. Main owns correctness repair and policy decision; maintainer owns material segmentation/quality/presentation/architecture tradeoffs. No optimization is implemented by this worker.",
    }
    report["baseline_source_files"] = baseline["source_files"]
    report["experimental_candidate_source"] = {
        "file": "tests/fixtures/self_grace_233/experiment.py",
        "sha256_at_execution": comparison["fixture_sha256"],
        "product_delta": "none; opt-in shared fixture policies only",
    }
    report["G2_recorded_schedule"] = {
        "status": "consequential grouping observed; adoption deferred to Main/maintainer structural gate",
        "continuation_fast_groups": next(
            [item["acoustics"] for item in row["commits"]]
            for row in replay_rows
            if row["case"] == "recorded_audio_continuation_0.05" and row["policy"] == "unchanged"
        ),
        "continuation_slow_groups": next(
            [item["acoustics"] for item in row["commits"]]
            for row in replay_rows
            if row["case"] == "recorded_audio_continuation_1" and row["policy"] == "unchanged"
        ),
        "slow_llm_logical_cancels": sum(
            event["event"] == "llm_logical_cancel"
            for row in comparison["results"]
            if row["policy"] == "unchanged" and row["llm_delay_s"] == 1.0
            for event in row["timeline"]
        ),
        "contract_or_prototype_implemented": False,
    }
    report["indirect_log_side_effect"] = {
        "earlier_setup_attempts": "[INFERENCE] Default SessionRuntimeLoggingService may have appended synthetic-only session entries to the default user-config log before isolation was repaired; no canonical log contents were read or altered directly",
        "retained_final_run_logging": baseline["logging"],
        "director_notified": True,
    }
    (HERE / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "summary": "summary.json",
                "fresh_asr_counts": report["fresh_asr_counts"],
                "replay_samples": report["replay_samples"],
                "source_commits": commit_rows,
                "fresh_asr_distributions": report["fresh_asr_distributions"],
                "candidate_deltas": changes,
            }
        )
    )


if __name__ == "__main__":
    main()
