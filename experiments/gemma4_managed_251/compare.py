from __future__ import annotations

import argparse
import json
from pathlib import Path

from summarize import distribution

SEED = 2512026
ORDER = ("A1", "B1", "B2", "A2")
MODES = {"A1": "mmap", "B1": "none", "B2": "none", "A2": "mmap"}
MEMORY_KEYS = (
    "process_gpu_dedicated_bytes",
    "process_gpu_shared_bytes",
    "WorkingSetSize",
    "PrivateUsage",
    "PeakWorkingSetSize",
    "PeakPagefileUsage",
    "device_gpu_dedicated_bytes",
    "device_gpu_shared_bytes",
    "system_available_physical_bytes",
    "system_commit_headroom_bytes",
    "system_committed_bytes",
    "system_pages_input_per_second",
    "system_page_reads_per_second",
    "PageFaultCount",
    "cpu_seconds",
)
REQUIRED_MEMORY = MEMORY_KEYS[:4] + MEMORY_KEYS[6:8] + MEMORY_KEYS[8:10] + MEMORY_KEYS[11:13]


def request_key(row):
    return (row["phase"], row["label"], row["channel"], row["translation"]["source_text"])


def metric_group(arm, selector):
    requests = [row for row in arm["requests"] if selector(row)]
    result = {
        "completed": len(requests),
        "latency_ms": distribution([row["latency_ms"] for row in requests]),
    }
    for key in (
        "prompt_n",
        "cache_n",
        "predicted_n",
        "prompt_ms",
        "predicted_ms",
        "prompt_per_second",
        "predicted_per_second",
    ):
        result[key] = distribution([(row.get("native_timings") or {}).get(key) for row in requests])
    return result


def quality_flags(row):
    text = row["translation"]["translated_text"]
    lowered = text.lower()
    flags = []
    if not text.strip():
        flags.append("empty")
    if any(marker in lowered for marker in ("<think>", "<analysis>", "<|channel>", "analysis:")):
        flags.append("reasoning_marker")
    stop = row.get("stop") or {}
    if stop.get("truncated") is not False or stop.get("stop_type") != "eos":
        flags.append("not_observed_eos_without_truncation")
    source = row["translation"]["source_text"].lower()
    if any(marker in source for marker in ("mina", "미나")):
        requirements = {
            "name_mina": ("mina", "미나"),
            "seven": ("7", "일곱"),
            "nine": ("9", "아홉"),
            "gate_three": ("3", "세 번", "셋"),
            "lunch": ("lunch", "점심"),
            "negation": ("not", "아니", "아님", "않"),
        }
    elif any(marker in source for marker in ("nori", "노리")):
        requirements = {
            "name_nori": ("nori", "노리"),
            "forty_two": ("42", "사십이"),
            "negation": ("not", "don't", "끄지", "꺼지지", "않"),
        }
    elif any(marker in source for marker in ("tuesday", "화요일")):
        requirements = {
            "tuesday": ("tuesday", "화요일"),
            "thursday": ("thursday", "목요일"),
            "corrected_time": ("14:30", "14시 30", "2:30", "2시 30", "두 시 반", "두시 반"),
            "rejected_time": ("13:40", "13시 40", "1:40", "1시 40", "한 시 사십"),
        }
    else:
        requirements = {
            "red": ("red", "빨간", "빨강", "붉"),
            "green": ("green", "초록", "녹색"),
            "twelve": ("12", "열두", "열둘"),
            "door": ("door", "문"),
        }
    for label, variants in requirements.items():
        if not any(variant in lowered for variant in variants):
            flags.append("review_surface_" + label)
    return flags


def compare(summary):
    by_label = {arm["arm"]: arm for arm in summary["arms"]}
    errors, warnings = [], []
    arms = [by_label[label] for label in ORDER if label in by_label]
    if len(arms) != 4:
        errors.append({"missing_arms": [label for label in ORDER if label not in by_label]})
    trees = {arm["candidate"].get("source_tree_sha256") for arm in arms}
    if len(trees) != 1:
        errors.append("candidate source content changed across arms")
    harnesses = {json.dumps(arm.get("harness_hashes"), sort_keys=True) for arm in arms}
    if len(harnesses) != 1 or any(not arm.get("harness_hashes") for arm in arms):
        errors.append("harness identity missing or changed across arms")
    normalized_profiles = []
    for arm in arms:
        label = arm["arm"]
        if arm["failures"] or arm["completed"] != 44:
            errors.append(
                {
                    "arm": label,
                    "completion_or_lifecycle_failures": arm["failures"],
                    "completed": arm["completed"],
                }
            )
        if len(arm["processes"]) != 1:
            errors.append({"arm": label, "model_process_count": len(arm["processes"])})
            continue
        command = arm["processes"][0]["effective_command"]
        normalized = list(command)
        normalized[normalized.index("--port") + 1] = "<dynamic-port>"
        normalized[normalized.index("--load-mode") + 1] = "<comparison-mode>"
        normalized_profiles.append(normalized)
        if (
            command[command.index("--load-mode") + 1] != MODES[label]
            or command[command.index("--device") + 1] == "none"
        ):
            errors.append({"arm": label, "effective_profile_mismatch": True})
        if (
            arm["native_max_active_all_tasks"] != 2
            or arm["native_max_active_translation_tasks"] != 2
        ):
            errors.append(
                {"arm": label, "native_active_capacity": arm["native_max_active_all_tasks"]}
            )
        overlap = [pair for pair in arm["pairs"] if pair["concurrent"]]
        if len(overlap) != 12 or any(
            pair["server_active_overlap_s"] is None or pair["server_active_overlap_s"] <= 0
            for pair in overlap
        ):
            errors.append({"arm": label, "server_overlap_incomplete": True})
        if arm["routing_summary"]["peer_chatbox_denied"] < 22:
            errors.append(
                {
                    "arm": label,
                    "peer_chatbox_denials": arm["routing_summary"]["peer_chatbox_denied"],
                }
            )
        if arm["app_event_counts"].get("gap", 0) or (
            arm["routing_summary"].get("delivery_health") or {}
        ).get("owner_history_evictions", 0):
            errors.append({"arm": label, "destination_absence_observation_gap": True})
        self_ids = {row["result_id"] for row in arm["requests"] if row["channel"] == "self"}
        peer_ids = {row["result_id"] for row in arm["requests"] if row["channel"] == "peer"}
        destinations = arm.get("destination_evidence") or {}
        if any(
            row["event"].get("utterance_id") in peer_ids
            for row in destinations.get("osc_sent_events", [])
        ):
            errors.append({"arm": label, "peer_osc_producer_event": True})
        if any(
            row["route"] == "self_chatbox"
            and row["decision"] == "published"
            and row["publication_id"] not in self_ids
            for row in destinations.get("routing_decisions", [])
        ):
            errors.append(
                {"arm": label, "chatbox_publication_not_associated_with_completed_self": True}
            )
        allowed_texts = {
            row["translation"]["translated_text"]
            for row in arm["requests"]
            if row["channel"] == "self"
        }
        sink_texts = [
            message["parameters"][0]
            for row in destinations.get("sink_messages", [])
            for message in row["messages"]
            if message["address"] == "/chatbox/input"
            and message["parameters"]
            and str(message["parameters"][0]).strip()
        ]
        if not sink_texts or any(text not in allowed_texts for text in sink_texts):
            errors.append(
                {"arm": label, "chatbox_sink_missing_positive_self_or_unassociated_payload": True}
            )
        resources = [
            row
            for row in arm.get("resource_samples", [])
            if ":" in row["phase"] and "first-prefix" not in row["phase"]
        ]
        for key in REQUIRED_MEMORY:
            if not any(isinstance(row["sample"].get(key), (int, float)) for row in resources):
                errors.append({"arm": label, "missing_required_resource_counter": key})
        for row in arm["requests"]:
            native = row.get("native_generation_settings") or {}
            if native.get("seed") != SEED:
                errors.append({"arm": label, "unfrozen_native_seed": row["result_id"]})
            effective, original = (
                row.get("effective_completion_parameters"),
                row.get("original_completion_parameters"),
            )
            if effective is None or original is None:
                errors.append({"arm": label, "missing_effective_payload": row["result_id"]})
                continue
            if effective.get("seed") != SEED or {
                key: value for key, value in effective.items() if key != "seed"
            } != {key: value for key, value in original.items() if key != "seed"}:
                errors.append(
                    {"arm": label, "sampling_control_changed_more_than_seed": row["result_id"]}
                )
            if (
                row.get("native_prompt_echo_matches") is not True
                or row.get("native_interval") is None
            ):
                errors.append(
                    {
                        "arm": label,
                        "missing_native_prompt_or_interval_confirmation": row["result_id"],
                    }
                )
            timing = row.get("native_timings") or {}
            if row["tokens_evaluated"] != timing.get("prompt_n", 0) + timing.get(
                "cache_n", 0
            ) or row["tokens_predicted"] != timing.get("predicted_n"):
                errors.append({"arm": label, "native_token_accounting_mismatch": row["result_id"]})
        for payload in arm.get("native_http_payloads", []):
            if payload["effective_parameters"].get("seed") != SEED:
                errors.append({"arm": label, "prefix_or_translation_seed_unfrozen": True})
    if normalized_profiles and any(
        profile != normalized_profiles[0] for profile in normalized_profiles
    ):
        errors.append(
            "effective native profiles differ beyond load mode, owned paths and dynamic port"
        )
    maps = {arm["arm"]: {request_key(row): row for row in arm["requests"]} for arm in arms}
    if arms and any(set(maps[arm["arm"]]) != set(maps[arms[0]["arm"]]) for arm in arms):
        errors.append("input/phase/channel request keys differ across arms")
    matched = []
    if len(arms) == 4 and all(set(maps[label]) == set(maps["A1"]) for label in ORDER):
        for key in maps["A1"]:
            rows = {label: maps[label][key] for label in ORDER}
            prompt_hashes = {row.get("native_request_prompt_sha256") for row in rows.values()}
            if len(prompt_hashes) != 1 or None in prompt_hashes:
                errors.append({"request_key": key, "native_prompt_hashes_differ": True})
            echoed_hashes = {row.get("native_template_prompt_sha256") for row in rows.values()}
            if len(echoed_hashes) != 1 or None in echoed_hashes:
                errors.append({"request_key": key, "raw_native_echo_prompt_hashes_differ": True})
            sampling = {
                json.dumps(row["native_generation_settings"], sort_keys=True)
                for row in rows.values()
            }
            if len(sampling) != 1:
                errors.append({"request_key": key, "native_sampling_settings_differ": True})
            output_groups = {row["translation"]["translated_text"] for row in rows.values()}
            if len(output_groups) != 1:
                warnings.append({"request_key": key, "output_varies_with_fixed_seed": True})
            matched.append(
                {
                    "phase": key[0],
                    "label": key[1],
                    "channel": key[2],
                    "source_text": key[3],
                    "native_prompt_hashes": {
                        label: row.get("native_request_prompt_sha256")
                        for label, row in rows.items()
                    },
                    "raw_native_echo_prompt_hashes": {
                        label: row.get("native_template_prompt_sha256")
                        for label, row in rows.items()
                    },
                    "latency_ms": {label: row["latency_ms"] for label, row in rows.items()},
                    "output_tokens": {
                        label: row["tokens_predicted"] for label, row in rows.items()
                    },
                    "cached_input_tokens": {
                        label: row["native_timings"]["cache_n"] for label, row in rows.items()
                    },
                    "outputs_identical": len(output_groups) == 1,
                }
            )
    phase_metrics, memory = {}, {}
    for arm in arms:
        label = arm["arm"]
        phase_metrics[label] = {
            "startup_and_first_prefix_excluded": True,
            "single": metric_group(arm, lambda row: row["label"].startswith("single-")),
            "serial_request": metric_group(arm, lambda row: row["label"].startswith("serial-")),
            "overlap_request": metric_group(arm, lambda row: row["label"].startswith("overlap-")),
            "serial_pair_makespan_ms": distribution(
                [pair["makespan_ms"] for pair in arm["pairs"] if not pair["concurrent"]]
            ),
            "overlap_pair_makespan_ms": distribution(
                [pair["makespan_ms"] for pair in arm["pairs"] if pair["concurrent"]]
            ),
        }
        raw = arm.get("resource_samples", [])
        memory[label] = {}
        for phase, selector in (
            ("lifecycle", lambda row: True),
            ("steady", lambda row: ":" in row["phase"] and "first-prefix" not in row["phase"]),
            ("load", lambda row: row["phase"] == "model-load"),
            ("prefix", lambda row: "first-prefix" in row["phase"]),
            ("shutdown", lambda row: row["phase"] == "shutdown"),
        ):
            selected = [row for row in raw if selector(row)]
            memory[label][phase] = {
                key: distribution([row["sample"].get(key) for row in selected])
                for key in MEMORY_KEYS
            }
            memory[label][phase]["samples"] = len(selected)
            memory[label][phase]["selected_observation_gaps_s"] = distribution(
                [right["t"] - left["t"] for left, right in zip(selected, selected[1:])]
            )
    mode_metrics = {}
    for mode in ("mmap", "none"):
        pooled = {
            "requests": [
                row for arm in arms if MODES[arm["arm"]] == mode for row in arm["requests"]
            ]
        }
        mode_metrics[mode] = {}
        for pair_kind in ("same", "opposite"):
            mode_metrics[mode][pair_kind] = {
                category: metric_group(
                    pooled,
                    lambda row, prefix=prefix, pair_kind=pair_kind: (
                        row["phase"].startswith(pair_kind + ":") and row["label"].startswith(prefix)
                    ),
                )
                for category, prefix in (
                    ("single", "single-"),
                    ("serial_request", "serial-"),
                    ("overlap_request", "overlap-"),
                )
            }
            selected_pairs = [
                pair
                for arm in arms
                if MODES[arm["arm"]] == mode
                for pair in arm["pairs"]
                if pair["phase"].startswith(pair_kind + ":")
            ]
            mode_metrics[mode][pair_kind]["overlap_pair_makespan_ms"] = distribution(
                [pair["makespan_ms"] for pair in selected_pairs if pair["concurrent"]]
            )
            mode_metrics[mode][pair_kind]["serial_pair_makespan_ms"] = distribution(
                [pair["makespan_ms"] for pair in selected_pairs if not pair["concurrent"]]
            )
    brackets = []
    for mmap, none in (("A1", "B1"), ("A2", "B2")):
        if mmap not in phase_metrics or none not in phase_metrics:
            continue
        delta = {}
        for metric in ("single", "serial_request", "overlap_request"):
            a = phase_metrics[mmap][metric]["latency_ms"]["median"]
            b = phase_metrics[none][metric]["latency_ms"]["median"]
            delta[metric + "_none_minus_mmap_median_ms"] = (
                b - a if a is not None and b is not None else None
            )
            delta[metric + "_none_vs_mmap_percent"] = (
                (b / a - 1) * 100 if a and b is not None else None
            )
        for metric in ("serial_pair_makespan_ms", "overlap_pair_makespan_ms"):
            a = phase_metrics[mmap][metric]["median"]
            b = phase_metrics[none][metric]["median"]
            delta[metric + "_none_minus_mmap_median_ms"] = (
                b - a if a is not None and b is not None else None
            )
        brackets.append({"mmap": mmap, "none": none, "differences": delta})
    cards = {}
    for arm in arms:
        for row in arm["requests"]:
            key = (
                row["translation"]["source_text"],
                row["translation"]["target_language"],
                row["translation"]["translated_text"],
            )
            if key not in cards:
                cards[key] = {
                    "source_text": key[0],
                    "target_language": key[1],
                    "translated_text": key[2],
                    "automatic_review_flags": quality_flags(row),
                    "instances": [],
                }
            cards[key]["instances"].append(
                {"arm": arm["arm"], "phase": row["phase"], "result_id": row["result_id"]}
            )
    return {
        "schema": "puripuly-heart/managed-gemma-251-comparison/v1",
        "order": list(ORDER),
        "controlled_seed": SEED,
        "complete_comparison": len(arms) == 4 and not errors,
        "completed_outputs": sum(arm["completed"] for arm in arms),
        "validation_errors": errors,
        "warnings": warnings,
        "phase_metrics": phase_metrics,
        "mode_metrics": mode_metrics,
        "memory": memory,
        "bracket_differences": brackets,
        "matched_requests": matched,
        "quality_cards": list(cards.values()),
        "human_quality_review_required": True,
        "quality_summary": {
            "unique_cards": len(cards),
            "flagged_cards": sum(bool(card["automatic_review_flags"]) for card in cards.values()),
            "flagged_output_instances": sum(
                len(card["instances"]) for card in cards.values() if card["automatic_review_flags"]
            ),
            "automatic_flags_are_not_quality_verdicts": True,
        },
        "decision": "Director-owned adopt/retain decision after reviewing repeatability, all outputs, latency tails and resource tradeoffs; no automatic speedup SLA.",
        "limits": summary["limits"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(json.loads(args.summary.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "complete_comparison": report["complete_comparison"],
                "completed_outputs": report["completed_outputs"],
                "validation_error_count": len(report["validation_errors"]),
                "quality_cards": len(report["quality_cards"]),
            }
        )
    )
