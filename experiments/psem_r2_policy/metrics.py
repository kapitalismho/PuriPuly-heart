from __future__ import annotations

import hashlib
import json
import math
import random
import xml.etree.ElementTree as ET
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Mapping, Sequence

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
AMI_WORDS = Path(r"C:/Users/salee/AppData/Local/Temp/opencode/stb_phase2_corpora/ami/annotations/words")
NITE_NS = "{http://nite.sourceforge.net/}"
HZ = 16000
BOOTSTRAP_SEED = 156
BOOTSTRAP_RESAMPLES = 10000
MIN_ELIGIBLE_CLUSTERS = 8
ROLES = ("A", "B", "C", "D")
STRATA = (
    "sequential",
    "short intervening speaker and return",
    "overlap return",
    "overlap takeover",
    "multiple consecutive OTHER segments",
    "same speaker",
    "initial or invalid anchor",
    "capture discontinuity",
    "C5 rollover",
    "late evidence",
)
DEV_CLUSTERS: dict[str, tuple[str, ...]] = {
    "ES2009": ("ES2009a", "ES2009c", "ES2009d"),
    "ES2002": ("ES2002b",),
    "EN2009": ("EN2009d",),
}
HOLDOUT_CLUSTERS: dict[str, tuple[str, ...]] = {
    meeting: (meeting,)
    for meeting in (
        "ES2004a",
        "ES2014a",
        "IS1009a",
        "TS3003b",
        "TS3007a",
        "ES2003a",
        "ES2011a",
        "IS1008a",
        "TS3004a",
        "TS3006a",
    )
}
_MEETING_TO_CLUSTER = {
    meeting: cluster
    for cluster, meetings in {**DEV_CLUSTERS, **HOLDOUT_CLUSTERS}.items()
    for meeting in meetings
}


def write_artifact(name: str, payload: dict[str, Any]) -> dict[str, str]:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / name
    encoded = json.dumps(payload, indent=1, ensure_ascii=False)
    path.write_text(encoded, encoding="utf-8")
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return {"path": str(path), "sha256": digest}


def cluster_id_for_meeting(meeting: str) -> str:
    return _MEETING_TO_CLUSTER.get(meeting, meeting)


def _independent_text_spans(parent_text: str, reconstructed: str) -> tuple[str, str]:
    if parent_text == reconstructed:
        return "", ""
    missing_parts: list[str] = []
    extra_parts: list[str] = []
    matcher = SequenceMatcher(a=parent_text, b=reconstructed, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in {"delete", "replace"} and i1 < i2:
            missing_parts.append(parent_text[i1:i2])
        if tag in {"insert", "replace"} and j1 < j2:
            extra_parts.append(reconstructed[j1:j2])
    return "".join(missing_parts), "".join(extra_parts)


def conservation_record(
    *,
    parent_text: str,
    unit_texts: list[str],
    token_texts: list[str],
    token_ids: Sequence[Any] | None = None,
    unit_token_ids: Sequence[Sequence[Any]] | None = None,
) -> dict[str, Any]:
    reconstructed = "".join(unit_texts)
    token_join = "".join(token_texts)
    missing_text, extra_text = _independent_text_spans(parent_text, reconstructed)
    ids = list(token_ids) if token_ids is not None else list(range(len(token_texts)))
    used: list[Any] = []
    if unit_token_ids is not None:
        for group in unit_token_ids:
            used.extend(group)
    else:
        cursor = 0
        for text in unit_texts:
            width = 0
            while cursor + width < len(token_texts) and "".join(token_texts[cursor : cursor + width + 1]) != text:
                width += 1
            if cursor + width < len(token_texts) and "".join(token_texts[cursor : cursor + width + 1]) == text:
                used.extend(ids[cursor : cursor + width + 1])
                cursor += width + 1
            else:
                cursor += 1
    counts = Counter(used)
    missing_ids = [item for item in ids if counts[item] == 0]
    duplicate_ids = [item for item in ids if counts[item] > 1]
    unknown_ids = [item for item in used if item not in ids]
    order_ok = used == ids
    return {
        "parent_text": parent_text,
        "reconstructed_units": reconstructed,
        "reconstructed_tokens": token_join,
        "conserved_units_to_parent": reconstructed == parent_text,
        "conserved_units_to_tokens": reconstructed == token_join,
        "missing": missing_text,
        "duplicate": extra_text,
        "missing_text": missing_text,
        "duplicate_text": extra_text,
        "n_units": len(unit_texts),
        "n_tokens": len(token_texts),
        "token_ids": ids,
        "used_token_ids": used,
        "missing_token_ids": missing_ids,
        "duplicate_token_ids": duplicate_ids,
        "unknown_token_ids": unknown_ids,
        "order_preserved": order_ok,
        "conserved_token_ids": not missing_ids and not duplicate_ids and not unknown_ids and order_ok,
    }


def fragmentation_record(
    group_ids: list[str],
    unit_texts: Sequence[str] | None = None,
    unit_token_counts: Sequence[int] | None = None,
    same_speaker_guard_splits: int = 0,
) -> dict[str, Any]:
    singletons = 0
    if unit_token_counts is not None:
        singletons = sum(1 for count in unit_token_counts if count == 1)
    elif unit_texts is not None:
        singletons = sum(1 for text in unit_texts if text.strip() and len(text.split()) == 1)
    return {
        "children_per_parent": len(group_ids),
        "group_ids": group_ids,
        "singleton_fragments": singletons,
        "extra_same_speaker_splits": same_speaker_guard_splits,
    }


def latency_record(marks: dict[str, float | None]) -> dict[str, Any]:
    def _delta(start: str, end: str) -> float | None:
        left = marks.get(start)
        right = marks.get(end)
        if left is None or right is None:
            return None
        return right - left

    delays = {
        "partition_delay_s": _delta("recognition_terminal", "partition"),
        "admission_delay_s": _delta("recognition_terminal", "translation_admission"),
        "receipt_to_terminal_s": _delta("producer_receipt", "recognition_terminal"),
        "source_to_receipt_s": _delta("source_support", "producer_receipt"),
        "admission_to_dispatch_s": _delta("translation_admission", "translation_completion"),
    }
    numeric = [value for value in delays.values() if value is not None]
    return {
        **dict(marks),
        **delays,
        "added_delay_s": numeric,
        "p50_added_delay_s": _percentile(numeric, 50) if numeric else None,
        "p95_added_delay_s": _percentile(numeric, 95) if numeric else None,
        "max_added_delay_s": max(numeric) if numeric else None,
        "c5_deadline_violations": int(bool(marks.get("c5_deadline_violation"))),
    }


def latency_distribution(values: Sequence[float]) -> dict[str, float | None]:
    numeric = [float(item) for item in values]
    if not numeric:
        return {"n": 0, "p50": None, "p95": None, "max": None}
    return {
        "n": len(numeric),
        "p50": _percentile(numeric, 50),
        "p95": _percentile(numeric, 95),
        "max": max(numeric),
    }


def _percentile(values: Sequence[float], percent: float) -> float:
    ordered = sorted(float(item) for item in values)
    if not ordered:
        raise ValueError("percentile of empty sample")
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (percent / 100.0)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def load_ami_words(meeting: str, words_dir: Path | None = None) -> list[dict[str, Any]]:
    root = words_dir or AMI_WORDS
    out: list[dict[str, Any]] = []
    for role in ROLES:
        path = root / f"{meeting}.{role}.words.xml"
        if not path.exists():
            continue
        xml = ET.fromstring(path.read_bytes())
        for node in xml.iter():
            if node.tag.split("}")[-1] != "w":
                continue
            attrib = node.attrib
            if attrib.get("punc", "") == "true":
                continue
            text = (node.text or "").strip()
            if not text:
                continue
            try:
                start = float(attrib.get("starttime", -1))
                end = float(attrib.get("endtime", -1))
            except ValueError:
                continue
            if start < 0 or end < 0:
                continue
            word_id = attrib.get(NITE_NS + "id") or attrib.get("nite:id") or ""
            out.append(
                {
                    "id": word_id,
                    "role": role,
                    "start": start,
                    "end": end,
                    "start_src": int(round(start * HZ)),
                    "end_src": int(round(end * HZ)),
                    "text": text,
                }
            )
    out.sort(key=lambda item: (item["start_src"], item["end_src"], item["role"]))
    return out


def _token_interval(token: Mapping[str, Any]) -> tuple[int | None, int | None]:
    start = token.get("source_start_sample")
    end = token.get("source_end_sample")
    if start is not None and end is not None:
        return int(start), int(end)
    start_ms = token.get("start_ms")
    end_ms = token.get("end_ms")
    if start_ms is None or end_ms is None:
        return None, None
    return int(round(float(start_ms) * HZ / 1000.0)), int(round(float(end_ms) * HZ / 1000.0))


def attribute_tokens(
    tokens: Sequence[Mapping[str, Any]],
    words: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    attributed: list[dict[str, Any]] = []
    for index, token in enumerate(tokens):
        start, end = _token_interval(token)
        text = str(token.get("text") or "")
        if start is None or end is None:
            attributed.append(
                {
                    "token_id": token.get("token_id", index),
                    "text": text,
                    "status": "unaligned",
                    "roles": [],
                    "start_src": start,
                    "end_src": end,
                    "ambiguous": False,
                }
            )
            continue
        roles: list[str] = []
        overlapping: list[dict[str, Any]] = []
        for word in words:
            if word["end_src"] > start and word["start_src"] < end:
                overlapping.append(word)
                if word["role"] not in roles:
                    roles.append(word["role"])
        if not roles:
            status = "unaligned"
        elif len(roles) > 1:
            status = "mixed"
        else:
            status = "attributable"
        attributed.append(
            {
                "token_id": token.get("token_id", index),
                "text": text,
                "status": status,
                "roles": roles,
                "start_src": start,
                "end_src": end,
                "ambiguous": len(roles) > 1,
                "n_overlapping_words": len(overlapping),
            }
        )
    return attributed


def _gt_events(words: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for word in words:
        if previous is None:
            previous = dict(word)
            continue
        overlap = word["start_src"] < previous["end_src"]
        gap_src = word["start_src"] - previous["end_src"]
        changed = word["role"] != previous["role"]
        events.append(
            {
                "from_role": previous["role"],
                "to_role": word["role"],
                "at_src": word["start_src"],
                "prev_end_src": previous["end_src"],
                "overlap": overlap,
                "gap_src": gap_src,
                "changed": changed,
            }
        )
        if word["end_src"] >= previous["end_src"]:
            previous = dict(word)
    return events


def classify_stratum(
    *,
    gt_events: Sequence[Mapping[str, Any]],
    unit_relations: Sequence[str],
    lifecycle: Mapping[str, Any] | None = None,
) -> list[str]:
    flags = lifecycle or {}
    labels: list[str] = []
    if flags.get("capture_discontinuity"):
        labels.append("capture discontinuity")
    if flags.get("c5_rollover"):
        labels.append("C5 rollover")
    if flags.get("late_evidence"):
        labels.append("late evidence")
    if flags.get("invalid_anchor") or flags.get("initial_anchor"):
        labels.append("initial or invalid anchor")
    others = [item for item in unit_relations if item == "OTHER"]
    if len(others) >= 2:
        labels.append("multiple consecutive OTHER segments")
    roles_in_order = [event["to_role"] for event in gt_events if event.get("changed")]
    if any(event.get("overlap") and event.get("changed") for event in gt_events):
        if len(roles_in_order) >= 2 and roles_in_order[-1] == (gt_events[0].get("from_role") if gt_events else None):
            labels.append("overlap return")
        else:
            labels.append("overlap takeover")
    changed = [event for event in gt_events if event.get("changed") and not event.get("overlap")]
    if not changed and not labels:
        labels.append("same speaker")
    if changed:
        labels.append("sequential")
        if len(changed) >= 2:
            first = changed[0]
            second = changed[1]
            if first["to_role"] != first["from_role"] and second["to_role"] == first["from_role"]:
                width = second["at_src"] - first["at_src"]
                if width <= int(2.0 * HZ):
                    labels.append("short intervening speaker and return")
    return list(dict.fromkeys(item for item in labels if item in STRATA)) or ["same speaker"]


def sequential_merge_contamination(
    *,
    units: Sequence[Mapping[str, Any]],
    attributed: Sequence[Mapping[str, Any]],
    words: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_id = {item["token_id"]: item for item in attributed}
    events = [event for event in _gt_events(words) if event.get("changed") and not event.get("overlap")]
    attributable_chars = 0
    contaminated_chars = 0
    eligible_units = 0
    mixed_chars = 0
    unaligned_chars = 0
    unknown_chars = 0
    unit_rows: list[dict[str, Any]] = []
    for unit in units:
        token_ids = list(unit.get("token_indexes") or unit.get("token_ids") or ())
        relation = unit.get("relation")
        texts = []
        roles: list[str] = []
        statuses: list[str] = []
        for token_id in token_ids:
            row = by_id.get(token_id)
            if row is None:
                continue
            text = str(row.get("text") or "")
            texts.append(text)
            statuses.append(str(row.get("status")))
            if row.get("status") == "attributable":
                roles.extend(row.get("roles") or [])
                attributable_chars += len(text)
            elif row.get("status") == "mixed":
                mixed_chars += len(text)
            else:
                unaligned_chars += len(text)
        if relation == "UNKNOWN":
            unknown_chars += sum(len(text) for text in texts)
        unique_roles = list(dict.fromkeys(roles))
        start = unit.get("start_source_sample")
        end = unit.get("end_source_sample")
        crossed = []
        if start is not None and end is not None:
            crossed = [
                event
                for event in events
                if start < event["at_src"] < end
            ]
        merged = bool(crossed) and len(unique_roles) >= 2
        contaminated = 0
        if merged:
            first_role = unique_roles[0]
            for token_id in token_ids:
                row = by_id.get(token_id)
                if row is None or row.get("status") != "attributable":
                    continue
                token_roles = row.get("roles") or []
                if token_roles and token_roles[0] != first_role:
                    contaminated += len(str(row.get("text") or ""))
        if any(status == "attributable" for status in statuses):
            eligible_units += 1
        contaminated_chars += contaminated
        unit_rows.append(
            {
                "group_id": unit.get("group_id"),
                "relation": relation,
                "roles": unique_roles,
                "merged_sequential": merged,
                "contaminated_chars": contaminated,
                "n_tokens": len(token_ids),
            }
        )
    starts = [unit.get("start_source_sample") for unit in units if unit.get("start_source_sample") is not None]
    ends = [unit.get("end_source_sample") for unit in units if unit.get("end_source_sample") is not None]
    for row in attributed:
        if row.get("start_src") is not None:
            starts.append(row["start_src"])
        if row.get("end_src") is not None:
            ends.append(row["end_src"])
    span_start = min(starts) if starts else None
    span_end = max(ends) if ends else None
    sequential_hits = []
    if span_start is not None and span_end is not None:
        sequential_hits = [
            event for event in events if span_start < event["at_src"] < span_end
        ]
    sequential_target = bool(sequential_hits)
    if not sequential_target:
        return {
            "eligible": False,
            "sequential_target": False,
            "reason": "no_sequential_target",
            "proportion": None,
            "attributable_chars": attributable_chars,
            "contaminated_chars": contaminated_chars,
            "mixed_chars": mixed_chars,
            "unaligned_chars": unaligned_chars,
            "unknown_chars": unknown_chars,
            "eligible_units": eligible_units,
            "units": unit_rows,
            "coverage": "none" if attributable_chars == 0 else "guard_only",
        }
    if attributable_chars == 0:
        return {
            "eligible": False,
            "sequential_target": True,
            "reason": "no_attributable_accepted_text",
            "proportion": None,
            "attributable_chars": 0,
            "contaminated_chars": 0,
            "mixed_chars": mixed_chars,
            "unaligned_chars": unaligned_chars,
            "unknown_chars": unknown_chars,
            "eligible_units": eligible_units,
            "units": unit_rows,
            "coverage": "none",
        }
    return {
        "eligible": True,
        "sequential_target": True,
        "reason": None,
        "proportion": contaminated_chars / attributable_chars,
        "attributable_chars": attributable_chars,
        "contaminated_chars": contaminated_chars,
        "unknown_chars": unknown_chars,
        "mixed_chars": mixed_chars,
        "unaligned_chars": unaligned_chars,
        "eligible_units": eligible_units,
        "units": unit_rows,
        "coverage": "partial" if (mixed_chars or unaligned_chars or unknown_chars) else "full",
    }


def coverage_record(
    *,
    r0: Mapping[str, Any],
    r2: Mapping[str, Any],
) -> dict[str, Any]:
    r0_attr = int(r0.get("attributable_chars") or 0)
    r2_attr = int(r2.get("attributable_chars") or 0)
    r2_unknown = int(r2.get("unknown_chars") or 0)
    newly_unassigned = max(0, r0_attr - r2_attr)
    worst_r2 = dict(r2)
    if r2.get("eligible") and newly_unassigned:
        contaminated = int(r2.get("contaminated_chars") or 0) + newly_unassigned
        attributable = r0_attr
        worst_r2 = {
            **r2,
            "contaminated_chars": contaminated,
            "attributable_chars": attributable,
            "proportion": contaminated / attributable if attributable else None,
            "newly_unassigned_treated_as_harm": newly_unassigned,
        }
    return {
        "r0_attributable_chars": r0_attr,
        "r2_attributable_chars": r2_attr,
        "r0_unaligned_chars": r0.get("unaligned_chars") or 0,
        "r2_unaligned_chars": r2.get("unaligned_chars") or 0,
        "r0_mixed_chars": r0.get("mixed_chars") or 0,
        "r2_mixed_chars": r2.get("mixed_chars") or 0,
        "r2_unknown_chars": r2_unknown,
        "newly_unassigned_chars": newly_unassigned,
        "worst_case_r2": worst_r2,
        "benefit_explained_only_by_unassigned": bool(newly_unassigned)
        and (r2.get("contaminated_chars") or 0) == 0
        and (r0.get("contaminated_chars") or 0) > 0,
    }


def same_speaker_extra_splits(
    units: Sequence[Mapping[str, Any]],
    attributed: Sequence[Mapping[str, Any]],
) -> int:
    by_id = {item["token_id"]: item for item in attributed}
    roles: list[str | None] = []
    for unit in units:
        found: list[str] = []
        for token_id in unit.get("token_indexes") or unit.get("token_ids") or ():
            row = by_id.get(token_id)
            if row is None or row.get("status") != "attributable":
                continue
            token_roles = row.get("roles") or []
            if token_roles:
                found.append(token_roles[0])
        unique = list(dict.fromkeys(found))
        roles.append(unique[0] if len(unique) == 1 else None)
    extra = 0
    for left, right in zip(roles, roles[1:]):
        if left is not None and left == right:
            extra += 1
    return extra


def score_parent(
    *,
    parent_text: str,
    tokens: Sequence[Mapping[str, Any]],
    units: Sequence[Mapping[str, Any]],
    words: Sequence[Mapping[str, Any]] | None = None,
    receipts: Sequence[Mapping[str, Any]] | None = None,
    marks: Mapping[str, float | None] | None = None,
    lifecycle: Mapping[str, Any] | None = None,
    meeting: str | None = None,
) -> dict[str, Any]:
    token_texts = [str(token.get("text") or "") for token in tokens]
    token_ids = [token.get("token_id", index) for index, token in enumerate(tokens)]
    unit_texts = [str(unit.get("text") or "") for unit in units]
    unit_ids = [list(unit.get("token_indexes") or unit.get("token_ids") or ()) for unit in units]
    conservation = conservation_record(
        parent_text=parent_text,
        unit_texts=unit_texts,
        token_texts=token_texts,
        token_ids=token_ids,
        unit_token_ids=unit_ids,
    )
    attributed = attribute_tokens(tokens, words or ())
    contamination = sequential_merge_contamination(units=units, attributed=attributed, words=words or ())
    extra_splits = same_speaker_extra_splits(units, attributed)
    fragmentation = fragmentation_record(
        [str(unit.get("group_id") or "") for unit in units],
        unit_texts=unit_texts,
        unit_token_counts=[len(item) for item in unit_ids],
        same_speaker_guard_splits=extra_splits,
    )
    gt_events = _gt_events(words or ())
    strata = classify_stratum(
        gt_events=gt_events,
        unit_relations=[str(unit.get("relation") or "") for unit in units],
        lifecycle=lifecycle,
    )
    late = [
        item
        for item in receipts or ()
        if item.get("disposition") in {"ignored_late", "late", "rejected_late"}
    ]
    return {
        "meeting": meeting,
        "cluster_id": cluster_id_for_meeting(meeting) if meeting else None,
        "conservation": conservation,
        "fragmentation": fragmentation,
        "contamination": contamination,
        "attribution": attributed,
        "strata": strata,
        "primary_stratum": "sequential" if contamination.get("sequential_target") else "same speaker",
        "sequential_target": bool(contamination.get("sequential_target")),
        "latency": latency_record(dict(marks or {})),
        "late_operations": late,
        "n_late_rejected": len(late),
        "eligible": bool(contamination.get("eligible")),
    }


def paired_cluster_bootstrap(
    deltas: Sequence[float],
    *,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    values = [float(item) for item in deltas]
    n = len(values)
    if n == 0:
        return {
            "n_clusters": 0,
            "mean": None,
            "ci95": [None, None],
            "seed": seed,
            "resamples": resamples,
            "leave_one_out": [],
        }
    mean = sum(values) / n
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = _percentile(means, 2.5)
    hi = _percentile(means, 97.5)
    leave = []
    if n >= 2:
        for index, dropped in enumerate(values):
            rest = values[:index] + values[index + 1 :]
            leave.append({"dropped_index": index, "dropped_delta": dropped, "mean": sum(rest) / len(rest)})
    return {
        "n_clusters": n,
        "mean": mean,
        "ci95": [lo, hi],
        "seed": seed,
        "resamples": resamples,
        "leave_one_out": leave,
        "deltas": values,
    }


def confirmatory_decision(
    *,
    cluster_rows: Sequence[Mapping[str, Any]],
    safety_failures: Sequence[str] | None = None,
    coverage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    eligible = [row for row in cluster_rows if row.get("eligible")]
    deltas = [float(row["delta"]) for row in eligible if row.get("delta") is not None]
    boot = paired_cluster_bootstrap(deltas)
    n = len(deltas)
    improved = sum(1 for delta in deltas if delta < 0)
    share = (improved / n) if n else None
    mean = boot["mean"]
    hi = boot["ci95"][1]
    explained = bool((coverage or {}).get("benefit_explained_only_by_unassigned"))
    failures = list(safety_failures or ())
    if failures:
        result = "Safety failure"
        passed = False
    elif n < MIN_ELIGIBLE_CLUSTERS:
        result = "Inconclusive due to sample, timing, alignment, runtime or budget gap"
        passed = False
    elif (
        mean is not None
        and mean < 0
        and hi is not None
        and hi < 0
        and share is not None
        and share >= 0.7
        and not explained
    ):
        result = "Supported only within declared AMI/provider scope"
        passed = True
    elif mean is not None and mean >= 0:
        result = "No useful effect"
        passed = False
    else:
        result = "Inconclusive due to sample, timing, alignment, runtime or budget gap"
        passed = False
    return {
        "result": result,
        "pass": passed,
        "n_eligible_clusters": n,
        "min_eligible_clusters": MIN_ELIGIBLE_CLUSTERS,
        "cluster_mean_delta": mean,
        "ci95": boot["ci95"],
        "improved_cluster_share": share,
        "absolute_effect": mean,
        "bootstrap": boot,
        "safety_failures": failures,
        "benefit_explained_only_by_unassigned": explained,
        "cluster_rows": list(cluster_rows),
    }


def latency_by_operation(
    runs: Sequence[Mapping[str, float | None]],
) -> dict[str, dict[str, float | int | None]]:
    keys: set[str] = set()
    for row in runs:
        keys.update(str(key) for key in row)
    out: dict[str, dict[str, float | int | None]] = {}
    for key in sorted(keys):
        values = [float(row[key]) for row in runs if row.get(key) is not None]
        dist = latency_distribution(values)
        out[key] = {
            "n": int(dist["n"] or 0),
            "p50": dist["p50"],
            "p95": dist["p95"],
            "max": dist["max"],
        }
    return out


def aggregate_cluster_parents(
    parents: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    excluded = 0
    incomplete = 0
    for parent in parents:
        if parent.get("incomplete") or parent.get("outage"):
            incomplete += 1
            continue
        if not parent.get("sequential_target"):
            excluded += 1
            continue
        cluster_id = str(parent.get("cluster_id") or parent.get("meeting") or "unknown")
        grouped.setdefault(cluster_id, []).append(parent)
    rows: list[dict[str, Any]] = []
    coverage_parts: list[dict[str, Any]] = []
    for cluster_id, items in grouped.items():
        r0_c = 0
        r0_a = 0
        r2_c = 0
        r2_a = 0
        r0_unknown = 0
        r2_unknown = 0
        for item in items:
            r0 = item.get("r0") or item.get("R0") or {}
            r2 = item.get("r2") or item.get("R2") or {}
            r0_cont = r0.get("contamination") or r0
            r2_cont = r2.get("contamination") or r2
            r0_c += int(r0_cont.get("contaminated_chars") or 0)
            r0_a += int(r0_cont.get("attributable_chars") or 0)
            r2_c += int(r2_cont.get("contaminated_chars") or 0)
            r2_a += int(r2_cont.get("attributable_chars") or 0)
            r0_unknown += int(r0_cont.get("unknown_chars") or 0)
            r2_unknown += int(r2_cont.get("unknown_chars") or 0)
        r0_p = (r0_c / r0_a) if r0_a else None
        r2_p = (r2_c / r2_a) if r2_a else None
        eligible = r0_p is not None and r2_p is not None
        delta = (r2_p - r0_p) if eligible else None
        r0_row = {
            "eligible": r0_a > 0,
            "attributable_chars": r0_a,
            "contaminated_chars": r0_c,
            "unknown_chars": r0_unknown,
            "proportion": r0_p,
        }
        r2_row = {
            "eligible": r2_a > 0,
            "attributable_chars": r2_a,
            "contaminated_chars": r2_c,
            "unknown_chars": r2_unknown,
            "proportion": r2_p,
        }
        coverage_parts.append(coverage_record(r0=r0_row, r2=r2_row))
        rows.append(
            {
                "cluster_id": cluster_id,
                "eligible": eligible,
                "n_parents": len(items),
                "r0_proportion": r0_p,
                "r2_proportion": r2_p,
                "delta": delta,
                "r0_chars": r0_a,
                "r2_chars": r2_a,
                "r0_contaminated": r0_c,
                "r2_contaminated": r2_c,
            }
        )
    explained = any(part.get("benefit_explained_only_by_unassigned") for part in coverage_parts)
    return {
        "cluster_rows": rows,
        "n_sequential_parents": sum(len(items) for items in grouped.values()),
        "n_non_sequential_excluded": excluded,
        "n_incomplete_preserved": incomplete,
        "coverage": {"benefit_explained_only_by_unassigned": explained},
    }

def policy_delta_rows(
    *,
    per_cluster: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cluster_id, policies in per_cluster.items():
        r0 = (policies.get("R0") or {}).get("contamination") or {}
        r2 = (policies.get("R2") or {}).get("contamination") or {}
        r1 = (policies.get("R1") or {}).get("contamination") or {}
        control = (policies.get("control") or {}).get("contamination") or {}
        eligible = bool(r0.get("eligible")) and bool(r2.get("eligible"))
        delta = None
        if eligible and r0.get("proportion") is not None and r2.get("proportion") is not None:
            delta = float(r2["proportion"]) - float(r0["proportion"])
        rows.append(
            {
                "cluster_id": cluster_id,
                "eligible": eligible,
                "r0_proportion": r0.get("proportion"),
                "r2_proportion": r2.get("proportion"),
                "r1_proportion": r1.get("proportion"),
                "control_proportion": control.get("proportion"),
                "delta": delta,
                "r0_chars": r0.get("attributable_chars"),
                "r2_chars": r2.get("attributable_chars"),
                "coverage_r0": r0.get("coverage"),
                "coverage_r2": r2.get("coverage"),
            }
        )
    return rows


def live_parent_ledger(
    *,
    parent_text: str,
    tokens: Sequence[Any],
    units: Sequence[Any],
    receipts: Sequence[Any] | None = None,
    marks: Mapping[str, float | None] | None = None,
    meeting: str | None = None,
    seal_reasons: Sequence[str] | None = None,
    speech_chunks: int | None = None,
    silence_chunks: int | None = None,
) -> dict[str, Any]:
    token_rows = []
    for index, token in enumerate(tokens):
        if isinstance(token, Mapping):
            row = dict(token)
            row.setdefault("token_id", index)
            token_rows.append(row)
            continue
        provenance = getattr(token, "provenance", None)
        token_rows.append(
            {
                "token_id": index,
                "text": getattr(token, "text", ""),
                "start_ms": getattr(token, "start_ms", None),
                "end_ms": getattr(token, "end_ms", None),
                "source_start_sample": getattr(token, "source_start_sample", None),
                "source_end_sample": getattr(token, "source_end_sample", None),
                "timing": getattr(token, "timing", None),
                "provenance": {
                    "native_event_id": getattr(provenance, "native_event_id", None),
                    "native_request_id": getattr(provenance, "native_request_id", None),
                    "native_item_id": getattr(provenance, "native_item_id", None),
                    "barrier": getattr(provenance, "barrier", None),
                    "from_finalize": getattr(provenance, "from_finalize", None),
                },
            }
        )
    unit_rows = []
    for unit in units:
        if isinstance(unit, Mapping):
            unit_rows.append(dict(unit))
            continue
        unit_rows.append(
            {
                "group_id": getattr(unit, "group_id", ""),
                "relation": getattr(unit, "relation", ""),
                "text": getattr(unit, "text", ""),
                "token_indexes": list(getattr(unit, "token_indexes", ())),
                "start_source_sample": getattr(unit, "start_source_sample", None),
                "end_source_sample": getattr(unit, "end_source_sample", None),
            }
        )
    receipt_rows = []
    for item in receipts or ():
        if isinstance(item, Mapping):
            receipt_rows.append(dict(item))
            continue
        receipt_rows.append(
            {
                "hypothesis_id": getattr(item, "hypothesis_id", None),
                "disposition": getattr(item, "disposition", None),
                "available_at_monotonic_s": getattr(item, "available_at_monotonic_s", None),
                "applied_at_monotonic_s": getattr(item, "applied_at_monotonic_s", None),
                "receipt_kind": "native_arrival",
            }
        )
    return {
        "meeting": meeting,
        "parent_text": parent_text,
        "tokens": token_rows,
        "units": unit_rows,
        "receipts": receipt_rows,
        "marks": dict(marks or {}),
        "c5_seal_reasons": list(seal_reasons or ()),
        "vad_speech_chunks": speech_chunks,
        "vad_silence_chunks": silence_chunks,
    }


def score_live_ledger(
    ledger: Mapping[str, Any],
    *,
    words: Sequence[Mapping[str, Any]] | None = None,
    lifecycle: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    meeting = ledger.get("meeting")
    loaded = words
    if loaded is None and meeting:
        loaded = load_ami_words(str(meeting))
    return score_parent(
        parent_text=str(ledger.get("parent_text") or ""),
        tokens=list(ledger.get("tokens") or ()),
        units=list(ledger.get("units") or ()),
        words=loaded,
        receipts=list(ledger.get("receipts") or ()),
        marks=dict(ledger.get("marks") or {}),
        lifecycle=lifecycle,
        meeting=None if meeting is None else str(meeting),
    )

