from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path


def plain(value):
    if isinstance(value, list):
        return [plain(item) for item in value]
    if not isinstance(value, dict):
        return value
    if "$type" in value:
        return {key: plain(item) for key, item in value["fields"].items()}
    if "$tuple" in value:
        return [plain(item) for item in value["$tuple"]]
    if "$uuid" in value:
        return value["$uuid"]
    return {key: plain(item) for key, item in value.items()}


def load(path):
    return plain(json.loads(Path(path).read_text(encoding="utf-8")))


def timing(values):
    if not values:
        return {"n": 0, "median_s": None, "p95_s": None}
    ordered = sorted(values)

    def percentile(fraction):
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {"n": len(values), "median_s": percentile(0.5), "p95_s": percentile(0.95)}


def source_units(trace, manifest, references):
    cursors = defaultdict(int)
    timeline = defaultdict(list)
    pending = defaultdict(list)
    completed = []
    sessions = {}
    for row in trace["rows"]:
        kind = row["kind"]
        if kind == "session":
            sessions[row["epoch"]] = row["scope"]
        elif kind == "write":
            epoch = row["identity"]["provider_epoch_id"]
            for span in row["source_ranges"]:
                count = span["normalized_end_sample"] - span["normalized_start_sample"]
                timeline[epoch].append(
                    (cursors[epoch], cursors[epoch] + count, span["source_start_sample"])
                )
                cursors[epoch] += count
            assert (
                sum(
                    s["normalized_end_sample"] - s["normalized_start_sample"]
                    for s in row["source_ranges"]
                )
                == row["bytes"] // 2
            )
        elif kind == "seal":
            epoch = row["identity"]["provider_epoch_id"]
            count = sum(
                s["normalized_end_sample"] - s["normalized_start_sample"]
                for s in row["sealed_content_ranges"]
            )
            padded = row["seal_reason"] == "delivery_deadline" or (
                row["seal_reason"] == "delivery_pause" and 64000 <= count < 112000
            )
            if padded:
                cursors[epoch] += 3200
        elif kind == "native":
            for token in row["payload"].get("tokens", []):
                if not token.get("is_final"):
                    continue
                if token["text"] == "<fin>":
                    completed.append((row["session"], pending.pop(row["session"], [])))
                elif not token["text"].startswith("<"):
                    pending[row["session"]].append(token)
    finals = [row for row in trace["rows"] if row["kind"] == "final"]
    assert len(completed) == len(finals), (len(completed), len(finals))
    coherent = {
        annotation: item["id"]
        for item in references["coherent"][trace["language"]]
        for annotation in item["annotations"]
    }
    units, mismatches = [], []
    for final_index, (final, (session, tokens)) in enumerate(zip(finals, completed, strict=True)):
        text = final["event"]["text"]
        raw_text = "".join(token["text"] for token in tokens)
        exact = raw_text.strip() == text
        if not exact:
            mismatches.append(final_index)
        offset = -(len(raw_text) - len(raw_text.lstrip()))
        epoch = final["event"]["identity"]["provider_epoch_id"]
        assert sessions[epoch] == session
        for token_index, token in enumerate(tokens):
            raw_right = offset + len(token["text"])
            left, right = max(0, offset), min(len(text), raw_right)
            offset = raw_right
            if not any(char.isalnum() for char in token["text"]):
                continue
            start, end = token.get("start_ms"), token.get("end_ms")
            source = []
            if start is not None and end is not None:
                for a, b, original in timeline[epoch]:
                    lo, hi = max(a, start * 16), min(b, end * 16)
                    if hi > lo:
                        source.append((original + lo - a, original + hi - a))
            covered = sum(b - a for a, b in source)
            contiguous = all(a[1] == b[0] for a, b in zip(source, source[1:]))
            valid = (
                exact and source and end > start and covered == (end - start) * 16 and contiguous
            )
            ref = None
            exclusion = "time_or_correspondence"
            if valid:
                a, b = source[0][0] / 16000, source[-1][1] / 16000
                midpoint = (a + b) / 2
                active = [
                    turn
                    for turn in manifest["turns"]
                    if turn["offset_s"] <= midpoint < turn["offset_s"] + turn["duration_s"]
                ]
                overlap = [
                    turn
                    for turn in manifest["turns"]
                    if max(a, turn["offset_s"]) < min(b, turn["offset_s"] + turn["duration_s"])
                ]
                if (
                    len(active) == 1
                    and active[0]["reference_available"]
                    and all(t["speaker_id"] == active[0]["speaker_id"] for t in overlap)
                ):
                    ref = active[0]
                    exclusion = None
                else:
                    exclusion = (
                        "overlap"
                        if len({t["speaker_id"] for t in overlap}) > 1
                        else "reference_unavailable_or_gap"
                    )
            units.append(
                {
                    "id": f"{final_index}:{token_index}",
                    "final": final_index,
                    "token_index": token_index,
                    "text": token["text"],
                    "left": left,
                    "right": right,
                    "session": session,
                    "native": str(token["speaker"]) if token.get("speaker") is not None else None,
                    "reference": ref["speaker_id"] if ref else None,
                    "annotation": ref["source_annotation_id"] if ref else None,
                    "coherent": coherent.get(ref["source_annotation_id"]) if ref else None,
                    "source_s": [source[0][0] / 16000, source[-1][1] / 16000] if source else None,
                    "weight_s": (end - start) / 1000 if valid else 0,
                    "excluded": exclusion,
                }
            )
    mapping = {}
    for session in set(unit["session"] for unit in units):
        scored = [u for u in units if u["session"] == session and u["reference"]]
        ids = sorted({u["native"] for u in scored if u["native"]})
        refs = sorted({u["reference"] for u in scored})
        weights = Counter()
        for unit in scored:
            if unit["native"]:
                weights[unit["native"], unit["reference"]] += unit["weight_s"]
        best, best_score = {}, -1
        for chosen in itertools.permutations(ids + [None] * len(refs), len(refs)):
            actual = [item for item in chosen if item is not None]
            if len(set(actual)) != len(actual):
                continue
            score = sum(weights[native, ref] for native, ref in zip(chosen, refs) if native)
            if score > best_score:
                best, best_score = {
                    native: ref for native, ref in zip(chosen, refs) if native
                }, score
        mapping[session] = best
    overlap_ms = sum(
        1
        for ms in range(120000)
        if len(
            {
                t["speaker_id"]
                for t in manifest["turns"]
                if t["offset_s"] * 1000 <= ms + 0.5 < (t["offset_s"] + t["duration_s"]) * 1000
            }
        )
        > 1
    )
    native = attribution(units, {u["id"]: (u["session"], u["native"]) for u in units}, mapping)
    recognized_annotations = {u["annotation"] for u in units if u["annotation"]}
    return (
        units,
        finals,
        mapping,
        {
            "native": native,
            "excluded": dict(Counter(u["excluded"] for u in units if u["excluded"])),
            "lexical_tokens": len(units),
            "correspondence_mismatched_finals": mismatches,
            "overlap_s": overlap_ms / 1000,
            "complete_reference_intervals": sum(
                t["reference_available"] for t in manifest["turns"]
            ),
            "reference_intervals_with_scorable_recognized_token": len(recognized_annotations),
            "reference_intervals_without_scorable_recognized_token": [
                t["source_annotation_id"]
                for t in manifest["turns"]
                if t["reference_available"]
                and t["source_annotation_id"] not in recognized_annotations
            ],
            "sessions": len(sessions),
            "seals": sum(r["kind"] == "seal" for r in trace["rows"]),
            "sent_audio_s_including_product_padding": sum(cursors.values()) / 16000,
            "sent_source_audio_s": sum(r["bytes"] for r in trace["rows"] if r["kind"] == "write")
            / 32000,
        },
    )


def attribution(units, keys, mapping):
    scored = [u for u in units if u["reference"]]
    assigned = correct = 0
    for unit in scored:
        scope, key = keys.get(unit["id"], (None, None))
        if key is not None:
            assigned += 1
            correct += mapping.get(scope, {}).get(key) == unit["reference"]
    return {"assigned": assigned, "correct": correct, "scorable": len(scored)}


def score_arm(arm, units, finals, mapping, references, presentation):
    origin = arm["clock_origin_estimate"]
    represented = defaultdict(list)
    flows = {}
    group_records = []
    accepted = {}
    for row in arm["events"]:
        event = row["event"]
        if event.get("is_final"):
            accepted[event["utterance_id"]] = event
    for group in arm["groups"]:
        candidates = [
            i for i, final in enumerate(finals) if abs(group["t"] - origin - final["t"]) < 0.0001
        ]
        assert len(candidates) == 1
        final_index = candidates[0]
        transcript = group["transcript"]
        left, right = transcript["source_text_range"]
        expected = finals[final_index]["event"]["text"][left:right]
        assert expected == transcript["text"]
        target = group["accepted_caption"]["utterance_id"]
        previous = group["previous"]
        if group["kind"] == "continuation" and previous:
            flows[target] = flows[previous]
        elif target not in flows:
            flows[target] = target
        covered = [
            u
            for u in units
            if u["final"] == final_index and left <= u["left"] and u["right"] <= right
        ]
        for unit in covered:
            represented[unit["id"]].append(target)
        group_records.append(
            {
                "final": final_index,
                "target": target,
                "kind": group["kind"],
                "units": [u["id"] for u in covered],
                "text": transcript["text"],
            }
        )
    keys = {}
    for unit in units:
        targets = represented.get(unit["id"], [])
        event = accepted.get(targets[-1]) if targets else None
        assignment = event.get("speaker_assignment") if event else None
        key = assignment["attribution"].get("key") if assignment else None
        keys[unit["id"]] = (
            (key["speaker_scope_id"], key["local_speaker_id"]) if key else (None, None)
        )
    coherent_counts = []
    for reference in references["coherent"][arm["language"]]:
        eligible = [u for u in units if u["coherent"] == reference["id"]]
        if eligible:
            connected = {
                flows[target]
                for u in eligible
                for target in represented.get(u["id"], [])
                if target in accepted
            }
            coherent_counts.append({"id": reference["id"], "flows": len(connected)})
    boundaries, within = [], []
    lexical = [u for u in units if u["reference"]]

    def boundary(left, right, within_final):
        left_targets, right_targets = represented.get(left["id"], []), represented.get(
            right["id"], []
        )
        joined = bool(
            left_targets and right_targets and flows[left_targets[-1]] == flows[right_targets[-1]]
        )
        label = (
            "separate"
            if left["reference"] != right["reference"]
            else (
                "continue"
                if left["coherent"] and left["coherent"] == right["coherent"]
                else "uncertain"
            )
        )
        if label == "uncertain" and left["coherent"] and right["coherent"]:
            complete = {
                r["id"]
                for r in references["coherent"][arm["language"]]
                if r["status"] == "complete"
            }
            if left["coherent"] in complete:
                label = "separate"
        right_group = next((g for g in group_records if right["id"] in g["units"]), None)
        cause = (
            "composition"
            if left_targets
            and right_targets
            and left_targets[-1] == right_targets[-1]
            and within_final
            else right_group["kind"] if right_group else "missing"
        )
        return {
            "left": left["id"],
            "right": right["id"],
            "label": label,
            "joined": joined,
            "speaker_change": left["reference"] != right["reference"],
            "cause": cause,
            "native_id_error": left["native"] == right["native"],
            "left_text": left["text"],
            "right_text": right["text"],
        }

    for index in range(len(finals) - 1):
        left = [u for u in lexical if u["final"] == index]
        right = [u for u in lexical if u["final"] == index + 1]
        if left and right:
            boundaries.append(boundary(left[-1], right[0], False))
    for left, right in zip(units, units[1:]):
        if (
            left["final"] == right["final"]
            and left["reference"]
            and right["reference"]
            and left["reference"] != right["reference"]
        ):
            within.append(boundary(left, right, True))
    scored = [b for b in boundaries if b["label"] != "uncertain"]
    changes = [b for b in boundaries + within if b["speaker_change"]]
    presentation_finals = [row for row in presentation["records"] if row["is_final"]]
    last_presentation = {row["utterance_id"]: row for row in presentation_finals}
    colored = 0
    presentation_gray = Counter()
    presentation_missing = []
    style_mismatches = []
    for unit in lexical:
        targets = represented.get(unit["id"], [])
        entry = last_presentation.get(targets[-1], {}).get("entry") if targets else None
        if not entry or entry.get("speaker_style") is None:
            presentation_missing.append(unit["id"])
        elif entry["speaker_style"] != "gray":
            colored += 1
        else:
            presentation_gray[entry.get("speaker_gray_reason")] += 1
    applied = [row for row in presentation_finals if row["receipt"]["outcome"] == "applied"]
    for row in applied:
        entry = row["entry"]
        assignment = entry.get("speaker_assignment") if entry else None
        if assignment:
            expected_gray = (
                assignment["attribution"]["state"] != "identified"
                or assignment.get("palette_index") is None
            )
            if expected_gray != (entry.get("speaker_style") == "gray"):
                style_mismatches.append(
                    {"utterance_id": row["utterance_id"], "revision": row["revision"]}
                )
    readable, settled, provider_settled = [], [], []
    for final in finals:
        if not final["event"]["text"]:
            continue
        segment = final["receipt"]["segment"]
        source_id = segment["identity"]["segment_id"]
        siblings = [
            group for group in arm["groups"] if abs(group["t"] - origin - final["t"]) < 0.0001
        ]
        visible = [
            row["t"]
            for row in arm["events"]
            if row["event"]["utterance_id"] == source_id and not row["event"].get("is_final")
        ]
        final_times = []
        for group in siblings:
            caption = group["accepted_caption"]
            matched = [
                row["t"]
                for row in arm["events"]
                if row["event"]["utterance_id"] == caption["utterance_id"]
                and row["event"]["peer_caption_revision"] >= caption["revision"]
                and row["event"].get("is_final")
            ]
            if matched:
                visible.append(min(matched))
                final_times.extend(matched)
        if visible:
            readable.append(min(visible) - segment["opened_at_monotonic_s"])
        if final_times:
            settled.append(max(final_times) - segment["sealed_at_monotonic_s"])
        provider_settled.append(origin + final["t"] - segment["sealed_at_monotonic_s"])
    return {
        "attribution": attribution(units, keys, mapping),
        "counts": arm["counts"],
        "logical_requests": len(arm["requests"]),
        "fragmented": sum(c["flows"] > 1 for c in coherent_counts),
        "coherent_scored": len(coherent_counts),
        "flow_sum": sum(c["flows"] for c in coherent_counts),
        "coherent": coherent_counts,
        "boundaries": boundaries,
        "within_final_speaker_changes": within,
        "joined": sum(b["joined"] for b in scored),
        "scored_boundaries": len(scored),
        "correctly_joined": sum(b["joined"] and b["label"] == "continue" for b in scored),
        "continue_boundaries": sum(b["label"] == "continue" for b in scored),
        "joined_kinds": dict(Counter(b["cause"] for b in boundaries if b["joined"])),
        "harmful": [b for b in changes if b["joined"]],
        "speaker_change_boundaries": len(changes),
        "same_speaker_unrelated_joins": [
            b
            for b in scored
            if b["joined"] and not b["speaker_change"] and b["label"] == "separate"
        ],
        "lost": [u["id"] for u in units if not represented.get(u["id"])],
        "duplicates": [u["id"] for u in units if len(represented.get(u["id"], [])) > 1],
        "missing_output": [
            u["id"] for u in units if not any(t in accepted for t in represented.get(u["id"], []))
        ],
        "presentation": {
            "colored": colored,
            "scorable": len(lexical),
            "gray": dict(presentation_gray),
            "missing_style": presentation_missing,
            "style_mismatches": style_mismatches,
            "applied_final_revisions": len(applied),
            "final_revisions": len(presentation_finals),
        },
        "timing": {
            "controlled_open_to_first_readable": timing(readable),
            "controlled_seal_to_last_settled": timing(settled),
            "actual_seal_to_provider_terminal": timing(provider_settled),
        },
        "group_records": group_records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", default=".tmp/issue373")
    args = parser.parse_args()
    directory = Path(args.directory)
    references = load(Path(__file__).with_name("issue373-references.json"))
    results = {}
    for language in ("en", "ja"):
        trace = load(directory / f"{language}-trace.json")
        manifest = load(Path(__file__).with_name("inputs") / f"replay-{language}.json")
        units, finals, mapping, shared = source_units(trace, manifest, references)
        arms = {
            arm: score_arm(
                load(directory / f"{language}-{arm}.json"),
                units,
                finals,
                mapping,
                references,
                load(directory / f"{language}-{arm}-presentation.json"),
            )
            for arm in ("B0", "B1")
        }
        results[language] = {
            "shared": shared,
            "mapping": mapping,
            "units": units,
            "arms": arms,
            "trace_sha256": hashlib.sha256(
                (directory / f"{language}-trace.json").read_bytes()
            ).hexdigest(),
        }
    (directory / "scores.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                lang: {
                    "shared": result["shared"],
                    "arms": {
                        name: {
                            k: v
                            for k, v in arm.items()
                            if k
                            not in (
                                "boundaries",
                                "within_final_speaker_changes",
                                "group_records",
                                "coherent",
                            )
                        }
                        for name, arm in result["arms"].items()
                    },
                }
                for lang, result in results.items()
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
