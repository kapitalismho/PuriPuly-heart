from __future__ import annotations

import argparse
import json
from pathlib import Path


def merged(ranges):
    result = []
    for left, right in sorted(ranges):
        if right <= left:
            continue
        if result and left <= result[-1][1]:
            result[-1][1] = max(right, result[-1][1])
        else:
            result.append([left, right])
    return result


def size(ranges):
    return sum(right - left for left, right in merged(ranges))


def frames_at(case, timestamp):
    delivered = sum(
        event["count"]
        for event in case["trace"]
        if event["event"] == "evidence_delivery" and event["at"] <= timestamp
    )
    frames = []
    epoch = None
    immutable = 0
    for item in case["evidence"][:delivered]:
        if epoch is not None and item["speaker_epoch"] < epoch:
            continue
        if item["speaker_epoch"] != epoch:
            frames, epoch, immutable = [], item["speaker_epoch"], 0
        prior = immutable
        if item["frame"] is None:
            immutable = max(immutable, item["immutable_through_sample"])
        if item["reason"] not in (None, "vad_pause", "source_eof"):
            frames = []
            continue
        if item["frame"] is None:
            continue
        probabilities = item["frame"]["probabilities"]
        slots = tuple(i for i, value in enumerate(probabilities) if value >= 0.5)
        ranked = sorted(probabilities, reverse=True)
        competing = len(slots) < 2 and ranked[1] >= 0.35
        for piece in item["source_map"]:
            capture = piece["capture"]
            if capture is None or piece["kind"] != "original":
                continue
            left, right = capture["normalized_start_sample"], capture["normalized_end_sample"]
            cap_epoch = capture["capture_epoch"]

            def overlaps(f):
                return f["cap_epoch"] == cap_epoch and f["left"] < right and left < f["right"]

            frames = [
                f
                for f in frames
                if not (overlaps(f) and not f["closed"] and f["compact_end"] > prior)
            ]
            if any(overlaps(f) and (f["closed"] or f["compact_end"] <= prior) for f in frames):
                continue
            frames.append(
                {
                    "left": left,
                    "right": right,
                    "cap_epoch": cap_epoch,
                    "epoch": epoch,
                    "slots": slots,
                    "compact_end": piece["compact_end_sample"],
                    "closed": item["range_state"] == "closed",
                    "clear": len(slots) == 1 and ranked[0] >= 0.8 and ranked[1] <= 0.2,
                    "uncertain": (not slots and ranked[0] >= 0.2) or competing,
                    "competing": competing,
                    "top": ranked[0],
                    "second": ranked[1],
                }
            )
        frames.sort(key=lambda f: (f["cap_epoch"], f["left"]))
        frames = frames[-2048:]
    return frames, immutable, delivered


def diagnose(report):
    rows = []
    for case in report["cases"]:
        if case["mode"] != "new":
            continue
        for source in case["sources"]:
            for origin in source["source"]["origins"]:
                submit = next(
                    (
                        e
                        for e in case["trace"]
                        if e["event"] == "native_submit"
                        and e["sealed_content_ranges"] == origin["source_ranges"]
                    ),
                    None,
                )
                if submit is None:
                    continue
                frames, immutable, delivered = frames_at(case, submit["at"])
                selected = [*origin["source_ranges"], *origin["context_ranges"]]
                coverage = []
                uncertain = []
                competing = []
                clear = []
                slots, epochs = set(), set()
                probability = []
                for span in selected:
                    left, right, epoch = (
                        span["normalized_start_sample"],
                        span["normalized_end_sample"],
                        span["capture_epoch"],
                    )
                    for frame in frames:
                        if frame["cap_epoch"] != epoch or not (
                            frame["closed"] or frame["compact_end"] <= immutable
                        ):
                            continue
                        a, b = max(left, frame["left"]), min(right, frame["right"])
                        if a >= b:
                            continue
                        coverage.append((a, b))
                        slots.update(frame["slots"])
                        epochs.add(frame["epoch"])
                        probability.append(frame["top"])
                        if frame["uncertain"]:
                            uncertain.append((a, b))
                        if frame["competing"]:
                            competing.append((a, b))
                        if frame["clear"]:
                            clear.append((a, b))
                expected = size(
                    (s["normalized_start_sample"], s["normalized_end_sample"]) for s in selected
                )
                missing = expected - size(coverage)
                state = (
                    "mixed"
                    if len(slots) > 1 or len(epochs) > 1
                    else (
                        "unavailable"
                        if missing or not slots
                        else "uncertain" if uncertain else "identified"
                    )
                )
                rows.append(
                    {
                        "route": report["route"],
                        "case": case["case"],
                        "profile": case["profile"],
                        "repetition": case["repetition"],
                        "text": source["source"]["raw_text"],
                        "parent_request_id": origin["request_id"],
                        "native_slice_id": submit["identity"]["provider_turn_id"],
                        "actual_frozen_attribution": origin["speaker_attribution"],
                        "actual_submit_at": submit["at"],
                        "source_available_at": origin["source_available_at_monotonic_s"],
                        "seal_reason": submit["seal_reason"],
                        "delivered_evidence_items_at_submit": delivered,
                        "selected_source_plus_context_unique_samples": expected,
                        "eligible_coverage_unique_samples": size(coverage),
                        "missing_eligible_samples": missing,
                        "uncertain_unique_samples": size(uncertain),
                        "competing_unique_samples": size(competing),
                        "clear_unique_samples": size(clear),
                        "observed_active_slots": sorted(slots),
                        "observed_epochs": sorted(epochs),
                        "top_probability_min": min(probability, default=None),
                        "top_probability_max": max(probability, default=None),
                        "reconstructed_purity_state": state,
                        "limit": "Offline inspection of delivered recorded evidence at submission; reproduces revision/immutability and MAX_FRAMES but not per-request retained-frame floor pruning. Frozen actual attribution is authoritative. No thresholds changed, future evidence used, or decisions rerun.",
                    }
                )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in args.trace:
        rows.extend(diagnose(json.loads(path.read_text(encoding="utf-8"))))
    result = {
        "schema": "issue218-recorded-attribution-diagnosis-1",
        "command": __import__("sys").argv,
        "thresholds": {
            "active": 0.5,
            "clear": 0.8,
            "quiet": 0.2,
            "competing": 0.35,
            "MAX_FRAMES": 2048,
        },
        "source": "Frozen CausalBatchSession._attribute considers every supplied source/context span, not majority duration; attribution frozen at _submit, before ASR final/admission; downstream allocator cannot invent keys for uncertain/unavailable/mixed.",
        "rows": rows,
    }
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
