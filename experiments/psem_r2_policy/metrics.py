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
AMI_WORDS = Path(
    r"C:/Users/salee/AppData/Local/Temp/opencode/stb_phase2_corpora/ami/annotations/words"
)
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


def write_artifact(
    name: str,
    payload: dict[str, Any],
    *,
    directory: Path | None = None,
) -> dict[str, str]:
    target = directory or ARTIFACTS
    target.mkdir(parents=True, exist_ok=True)
    path = target / name
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
            while (
                cursor + width < len(token_texts)
                and "".join(token_texts[cursor : cursor + width + 1]) != text
            ):
                width += 1
            if (
                cursor + width < len(token_texts)
                and "".join(token_texts[cursor : cursor + width + 1]) == text
            ):
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
        "conserved_token_ids": not missing_ids
        and not duplicate_ids
        and not unknown_ids
        and order_ok,
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


LATENCY_DURATION_MARKS = {
    "partition_delay_s": ("recognition_terminal", "partition"),
    "admission_delay_s": ("recognition_terminal", "translation_admission"),
    "source_to_receipt_s": ("source_support", "producer_receipt"),
    "receipt_to_terminal_s": ("producer_receipt", "recognition_terminal"),
    "admission_to_dispatch_s": ("translation_admission", "translation_dispatch"),
    "admission_to_completion_s": (
        "translation_admission",
        "translation_completion",
    ),
}


def latency_record(marks: dict[str, float | None]) -> dict[str, Any]:
    def _delta(start: str, end: str) -> float | None:
        left = marks.get(start)
        right = marks.get(end)
        if left is None or right is None:
            return None
        return right - left

    return {
        **dict(marks),
        **{name: _delta(start, end) for name, (start, end) in LATENCY_DURATION_MARKS.items()},
        "c5_deadline_violations": (
            int(bool(marks["c5_deadline_violation"])) if "c5_deadline_violation" in marks else None
        ),
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
        if len(roles_in_order) >= 2 and roles_in_order[-1] == (
            gt_events[0].get("from_role") if gt_events else None
        ):
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
    events = [
        event for event in _gt_events(words) if event.get("changed") and not event.get("overlap")
    ]
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
            crossed = [event for event in events if start < event["at_src"] < end]
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
    starts = [
        unit.get("start_source_sample")
        for unit in units
        if unit.get("start_source_sample") is not None
    ]
    ends = [
        unit.get("end_source_sample") for unit in units if unit.get("end_source_sample") is not None
    ]
    for row in attributed:
        if row.get("start_src") is not None:
            starts.append(row["start_src"])
        if row.get("end_src") is not None:
            ends.append(row["end_src"])
    span_start = min(starts) if starts else None
    span_end = max(ends) if ends else None
    sequential_hits = []
    if span_start is not None and span_end is not None:
        sequential_hits = [event for event in events if span_start < event["at_src"] < span_end]
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


CONCRETE_RELATIONS = ("CURRENT", "OTHER")


def _is_lexical_text(text: Any) -> bool:
    return any(char.isalnum() for char in str(text or ""))


def arm_guard_record(
    *,
    units: Sequence[Mapping[str, Any]],
    attributed: Sequence[Mapping[str, Any]],
    words: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_id = {row["token_id"]: row for row in attributed}
    events = [
        event for event in _gt_events(words) if event.get("changed") and not event.get("overlap")
    ]
    spans: list[tuple[int, int]] = []
    for unit in units:
        start = unit.get("start_source_sample")
        end = unit.get("end_source_sample")
        if start is not None and end is not None:
            spans.append((int(start), int(end)))
    for row in attributed:
        if row.get("start_src") is not None and row.get("end_src") is not None:
            spans.append((int(row["start_src"]), int(row["end_src"])))
    span = (min(start for start, _ in spans), max(end for _, end in spans)) if spans else None
    span_events = (
        [event for event in events if span[0] < int(event["at_src"]) < span[1]] if span else []
    )
    excluded = {"punctuation_only": 0, "mixed": 0, "unaligned": 0}
    lexical_ids: list[Any] = []
    lexical_roles: list[str] = []
    lexical_chars: dict[Any, int] = {}
    wrong_token_ids: list[Any] = []
    wrong_chars: list[int] = []
    unit_witnesses: list[dict[str, Any]] = []
    relations: dict[Any, list[str]] = {}
    for unit in units:
        token_ids = list(unit.get("token_indexes") or unit.get("token_ids") or ())
        relation = str(unit.get("relation") or "")
        start = unit.get("start_source_sample")
        end = unit.get("end_source_sample")
        crossed = bool(
            start is not None
            and end is not None
            and any(int(start) < int(event["at_src"]) < int(end) for event in events)
        )
        reference: str | None = None
        unit_tokens: list[tuple[Any, str]] = []
        for token_id in token_ids:
            row = by_id.get(token_id)
            if row is None:
                continue
            text = str(row.get("text") or "")
            status = str(row.get("status") or "")
            if status != "attributable":
                excluded["mixed" if status == "mixed" else "unaligned"] += 1
                continue
            if not _is_lexical_text(text):
                excluded["punctuation_only"] += 1
                continue
            roles = [str(role) for role in (row.get("roles") or ())]
            if not roles:
                excluded["unaligned"] += 1
                continue
            if reference is None:
                reference = roles[0]
            if token_id not in lexical_ids:
                lexical_ids.append(token_id)
                lexical_roles.append(roles[0])
                lexical_chars[token_id] = len(text)
            if relation:
                bucket = relations.setdefault(token_id, [])
                if relation not in bucket:
                    bucket.append(relation)
            unit_tokens.append((token_id, roles[0]))
        wrong = [
            token_id
            for token_id, role in unit_tokens
            if crossed and reference is not None and role != reference
        ]
        wrong_token_ids.extend(wrong)
        wrong_chars.extend(lexical_chars.get(token_id, 0) for token_id in wrong)
        unit_witnesses.append(
            {
                "group_id": unit.get("group_id"),
                "relation": relation,
                "crossed_verified_boundary": crossed,
                "reference_role": reference,
                "lexical_tokens": len(unit_tokens),
                "wrong_token_ids": list(wrong),
                "wrong_chars": sum(lexical_chars.get(token_id, 0) for token_id in wrong),
            }
        )
    return {
        "span": None if span is None else [span[0], span[1]],
        "span_verified_changes": len(span_events),
        "same_speaker_stratum": span is not None and not span_events,
        "lexical_token_ids": lexical_ids,
        "lexical_roles": lexical_roles,
        "lexical_tokens": len(lexical_ids),
        "excluded": excluded,
        "annotation_tokens": len(words),
        "annotation_source_scope": "meeting_annotation_source",
        "wrong_token_ids": wrong_token_ids,
        "wrong_chars": sum(wrong_chars),
        "lexical_token_chars": [[token_id, lexical_chars[token_id]] for token_id in lexical_ids],
        "unit_witnesses": unit_witnesses,
        "relations_by_token": [[token_id, relations[token_id]] for token_id in relations],
    }


def _guard_relations(record: Mapping[str, Any]) -> dict[Any, list[str]]:
    mapping: dict[Any, list[str]] = {}
    for token_id, relations in record.get("relations_by_token") or ():
        bucket = mapping.setdefault(token_id, [])
        for relation in relations:
            if relation not in bucket:
                bucket.append(relation)
    return mapping


def pair_parent_guard(
    r0_guard: Mapping[str, Any] | None,
    r2_guard: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not r0_guard or not r2_guard:
        return {
            "assessed": False,
            "reason": "missing_arm_guard_records",
            "failures": [],
            "severe": False,
            "wrong_merge": {
                "new_wrong_token_ids": [],
                "new_wrong_tokens": 0,
                "new_wrong_chars": 0,
            },
            "same_speaker": {"new_same_speaker_splits": 0},
            "checked": {
                "lexical_tokens": 0,
                "annotation_tokens": 0,
                "annotation_source_scope": "meeting_annotation_source",
                "concrete_relation_claims": 0,
                "grouping_safety_assessed": False,
                "coverage_status": "missing_arm_guard_records",
            },
        }
    r0_wrong = list(dict.fromkeys(r0_guard.get("wrong_token_ids") or ()))
    r2_wrong = list(r2_guard.get("wrong_token_ids") or ())
    new_wrong = [token_id for token_id in r2_wrong if token_id not in set(r0_wrong)]
    r0_chars = {
        token_id: int(chars) for token_id, chars in r0_guard.get("lexical_token_chars") or ()
    }
    r2_chars = {
        token_id: int(chars) for token_id, chars in r2_guard.get("lexical_token_chars") or ()
    }
    new_wrong_chars = sum(r2_chars.get(token_id, 0) for token_id in new_wrong)
    r0_relations = _guard_relations(r0_guard)
    r2_relations = _guard_relations(r2_guard)
    roles = dict(
        zip(
            r2_guard.get("lexical_token_ids") or (),
            r2_guard.get("lexical_roles") or (),
        )
    )
    offenders: list[Any] = []
    witnesses: list[dict[str, Any]] = []
    if r2_guard.get("same_speaker_stratum"):
        by_role: dict[str, dict[str, list[Any]]] = {}
        for token_id, role in roles.items():
            entry = by_role.setdefault(role, {"CURRENT": [], "OTHER": []})
            for relation in r2_relations.get(token_id, []):
                if relation in CONCRETE_RELATIONS:
                    entry[relation].append(token_id)
        for role, entry in by_role.items():
            if not entry["CURRENT"] or not entry["OTHER"]:
                continue
            prior = {"CURRENT": [], "OTHER": []}
            for token_id in entry["CURRENT"] + entry["OTHER"]:
                for relation in r0_relations.get(token_id, []):
                    if relation in CONCRETE_RELATIONS:
                        prior[relation].append(token_id)
            if prior["CURRENT"] and prior["OTHER"]:
                continue
            offenders.extend(entry["CURRENT"] + entry["OTHER"])
            witnesses.append(
                {
                    "role": role,
                    "current_token_ids": entry["CURRENT"],
                    "other_token_ids": entry["OTHER"],
                }
            )
    failures: list[str] = []
    if new_wrong:
        failures.append(f"wrong_merge:{len(new_wrong)}")
    if offenders:
        failures.append(f"same_speaker_cross_owner:{len(offenders)}")
    concrete_claims = [
        token_id
        for token_id, _role in roles.items()
        if any(relation in CONCRETE_RELATIONS for relation in r2_relations.get(token_id, []))
    ]
    return {
        "assessed": True,
        "failures": failures,
        "severe": bool(failures),
        "wrong_merge": {
            "new_wrong_token_ids": new_wrong,
            "new_wrong_tokens": len(new_wrong),
            "new_wrong_chars": new_wrong_chars,
            "r0_wrong_tokens": len(r0_wrong),
            "r0_wrong_chars": sum(r0_chars.get(token_id, 0) for token_id in r0_wrong),
            "r2_wrong_tokens": len(r2_wrong),
            "r2_wrong_chars": sum(r2_chars.get(token_id, 0) for token_id in r2_wrong),
        },
        "same_speaker": {
            "stratum": bool(r2_guard.get("same_speaker_stratum")),
            "new_same_speaker_splits": len(offenders),
            "offender_token_ids": offenders,
            "witnesses": witnesses,
        },
        "checked": {
            "lexical_tokens": len(roles),
            "annotation_tokens": _annotation_tokens(r0_guard, r2_guard),
            "excluded": dict(r2_guard.get("excluded") or {}),
            "concrete_relation_claims": len(concrete_claims),
            "grouping_safety_assessed": bool(roles),
            "coverage_status": (
                "assessed"
                if roles
                else (
                    "no_attributable_lexical_tokens"
                    if _annotation_tokens(r0_guard, r2_guard)
                    else "no_annotation_source"
                )
            ),
        },
        "arm_witnesses": {
            "r0": {
                "wrong_token_ids": [item for item in r0_wrong],
                "units": list(r0_guard.get("unit_witnesses") or ()),
            },
            "r2": {
                "wrong_token_ids": r2_wrong,
                "units": list(r2_guard.get("unit_witnesses") or ()),
            },
        },
    }


def guard_coverage_required(parent: Mapping[str, Any]) -> bool:
    """True when a parent carries accepted text whose grouping safety must be computed."""
    if parent.get("incomplete") or parent.get("outage"):
        return False
    if not str(parent.get("text") or ""):
        return False
    return operational_outcome(parent) in {"final_nonempty", "degraded_prefix"}


def _annotation_tokens(*guards: Mapping[str, Any]) -> int:
    return max(int(guard.get("annotation_tokens") or 0) for guard in guards)


ZERO_COVERAGE_STATUSES = ("no_attributable_lexical_tokens", "no_annotation_source")
OVERLAP_UNASSESSABLE = "overlap_unassessable"


def guard_coverage_status(parent: Mapping[str, Any]) -> tuple[str, bool]:
    guard = parent.get("guard")
    if not isinstance(guard, Mapping) or not guard.get("assessed"):
        return "missing_guard", False
    checked = guard.get("checked") or {}
    return str(checked.get("coverage_status") or "missing_coverage_status"), True


def overlap_unassessable(parent: Mapping[str, Any]) -> tuple[bool, str | None]:
    """Measured multi-role overlap: computed guard, zero unique lexical attribution.

    Single predicate behind primary-pool membership, case/phase validity, coverage
    reporting and the formed-parent selection bounds subset.
    """
    if not guard_coverage_required(parent):
        return False, "no_accepted_text"
    if parent.get("provenance_valid") is False:
        return False, "invalid_provenance"
    status, computed = guard_coverage_status(parent)
    if not computed:
        return False, "missing_guard"
    checked = dict((parent.get("guard") or {}).get("checked") or {})
    if status != "no_attributable_lexical_tokens":
        return False, f"coverage_status:{status}"
    if int(checked.get("lexical_tokens") or 0) != 0:
        return False, "unique_lexical_attribution_present"
    if int(checked.get("annotation_tokens") or 0) <= 0:
        return False, "no_annotation_tokens"
    excluded = dict(checked.get("excluded") or {})
    if int(excluded.get("mixed") or 0) <= 0:
        return False, "no_mixed_overlap_tokens"
    for arm in ("r0", "r2"):
        payload = parent.get(arm) or parent.get(arm.upper())
        if not isinstance(payload, Mapping):
            return False, f"missing_{arm}_record"
        contamination = payload.get("contamination") or payload
        if not isinstance(contamination, Mapping):
            return False, f"missing_{arm}_contamination"
        if contamination.get("eligible") is not False:
            return False, f"{arm}_score_present"
        if str(contamination.get("reason") or "") != "no_attributable_accepted_text":
            return False, f"{arm}_contamination_reason:{contamination.get('reason')}"
    return True, None


def case_guard_summary(parents: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    assessed = 0
    unassessed = 0
    coverage_reasons: list[str] = []
    missing_guard_parents: list[str] = []
    unassessed_coverage_parents: list[str] = []
    unassessed_alignment_parents: list[str] = []
    no_annotation_source_parents: list[str] = []
    overlap_parents: list[str] = []
    for parent in parents:
        parent_id = str(parent.get("parent_id") or parent.get("index") or "parent")
        in_pool, pool_reason = primary_pool_membership(parent)
        guard = parent.get("guard")
        coverage_status, guard_computed = guard_coverage_status(parent)
        if pool_reason == OVERLAP_UNASSESSABLE:
            overlap_parents.append(parent_id)
        if not guard_computed:
            unassessed += 1
            if guard_coverage_required(parent):
                missing_guard_parents.append(parent_id)
                coverage_reasons.append(f"missing_guard:{parent_id}")
            continue
        assessed += 1
        if coverage_status == "no_attributable_lexical_tokens":
            unassessed_alignment_parents.append(parent_id)
        elif coverage_status == "no_annotation_source":
            no_annotation_source_parents.append(parent_id)
        if (
            coverage_status in ZERO_COVERAGE_STATUSES
            and in_pool
            and pool_reason != OVERLAP_UNASSESSABLE
        ):
            unassessed_coverage_parents.append(parent_id)
            coverage_reasons.append(f"grouping_safety_unassessed:{parent_id}")
        for reason in guard.get("failures") or ():
            failures.append(f"{parent_id}:{reason}")
        rows.append(
            {
                "parent_id": parent_id,
                "failures": list(guard.get("failures") or ()),
                "new_wrong_tokens": int(
                    (guard.get("wrong_merge") or {}).get("new_wrong_tokens") or 0
                ),
                "new_wrong_chars": int(
                    (guard.get("wrong_merge") or {}).get("new_wrong_chars") or 0
                ),
                "new_same_speaker_splits": int(
                    (guard.get("same_speaker") or {}).get("new_same_speaker_splits") or 0
                ),
                "lexical_tokens": int((guard.get("checked") or {}).get("lexical_tokens") or 0),
                "annotation_tokens": (
                    int((guard.get("checked") or {})["annotation_tokens"])
                    if "annotation_tokens" in (guard.get("checked") or {})
                    else None
                ),
                "annotation_source_scope": "meeting_annotation_source",
                "guard_computed": True,
                "concrete_relation_claims": int(
                    (guard.get("checked") or {}).get("concrete_relation_claims") or 0
                ),
                "excluded": dict((guard.get("checked") or {}).get("excluded") or {}),
                "coverage_status": str((guard.get("checked") or {}).get("coverage_status") or ""),
                "grouping_safety_assessed": bool(
                    (guard.get("checked") or {}).get("grouping_safety_assessed")
                ),
            }
        )
    excluded_totals = {"punctuation_only": 0, "mixed": 0, "unaligned": 0}
    for row in rows:
        for key, value in row["excluded"].items():
            excluded_totals[key] = excluded_totals.get(key, 0) + int(value or 0)
    return {
        "assessed_parents": assessed,
        "unassessed_parents": unassessed,
        "failures": failures,
        "severe": bool(failures),
        "wrong_merge_tokens": sum(row["new_wrong_tokens"] for row in rows),
        "wrong_merge_chars": sum(row["new_wrong_chars"] for row in rows),
        "same_speaker_splits": sum(row["new_same_speaker_splits"] for row in rows),
        "coverage_reasons": coverage_reasons,
        "overlap_unassessable_parents": overlap_parents,
        "coverage": {
            "grouping_safety_assessed": bool(
                sum(row["lexical_tokens"] for row in rows)
                and not missing_guard_parents
                and not unassessed_coverage_parents
            ),
            "missing_guard_parents": missing_guard_parents,
            "unassessed_coverage_parents": unassessed_coverage_parents,
            "unassessed_alignment_parents": unassessed_alignment_parents,
            "no_annotation_source_parents": no_annotation_source_parents,
        },
        "checked": {
            "lexical_tokens": sum(row["lexical_tokens"] for row in rows),
            "concrete_relation_claims": sum(row["concrete_relation_claims"] for row in rows),
            "excluded": excluded_totals,
        },
        "parents": rows,
    }


def guard_aggregate(
    cases: Sequence[Mapping[str, Any]],
    parents: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    per_case = [
        {
            "meeting": case.get("meeting"),
            "phase": case.get("phase"),
            **case_guard_summary(list(case.get("parents") or ())),
        }
        for case in cases
    ]
    failures = [f"{row.get('meeting')}:{item}" for row in per_case for item in row["failures"]]
    coverage_reasons = [
        f"{row.get('meeting')}:{item}" for row in per_case for item in row["coverage_reasons"]
    ]
    return {
        "failures": failures,
        "coverage_reasons": coverage_reasons,
        "severe": bool(failures),
        "assessed_parents": sum(row["assessed_parents"] for row in per_case),
        "unassessed_parents": sum(row["unassessed_parents"] for row in per_case),
        "wrong_merge_tokens": sum(row["wrong_merge_tokens"] for row in per_case),
        "wrong_merge_chars": sum(row["wrong_merge_chars"] for row in per_case),
        "same_speaker_splits": sum(row["same_speaker_splits"] for row in per_case),
        "coverage": {
            "grouping_safety_assessed": bool(
                sum(row["checked"]["lexical_tokens"] for row in per_case)
                and not any(row["coverage"]["missing_guard_parents"] for row in per_case)
                and not any(row["coverage"]["unassessed_coverage_parents"] for row in per_case)
            ),
            "missing_guard_parents": sum(
                len(row["coverage"]["missing_guard_parents"]) for row in per_case
            ),
            "unassessed_coverage_parents": sum(
                len(row["coverage"]["unassessed_coverage_parents"]) for row in per_case
            ),
            "unassessed_alignment_parents": sum(
                len(row["coverage"]["unassessed_alignment_parents"]) for row in per_case
            ),
            "no_annotation_source_parents": sum(
                len(row["coverage"]["no_annotation_source_parents"]) for row in per_case
            ),
        },
        "formed_parents": len(list(parents)),
        "overlap_unassessable_parents": sum(
            len(row["overlap_unassessable_parents"]) for row in per_case
        ),
        "cases": per_case,
    }


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
    contamination = sequential_merge_contamination(
        units=units, attributed=attributed, words=words or ()
    )
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
        "guard": arm_guard_record(units=units, attributed=attributed, words=words or ()),
        "strata": strata,
        "primary_stratum": (
            "sequential" if contamination.get("sequential_target") else "same speaker"
        ),
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
            leave.append(
                {"dropped_index": index, "dropped_delta": dropped, "mean": sum(rest) / len(rest)}
            )
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
    evaluation: Mapping[str, Any] | None = None,
    sensitivity: Mapping[str, Any] | None = None,
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
    evidence = dict(evaluation or {})
    operational_clean = bool((coverage or {}).get("operational_clean", True))
    evaluation_valid = bool(evidence.get("evaluation_valid", True))
    execution_completed = bool(evidence.get("execution_completed", True))
    invalid_reasons = list(evidence.get("evaluation_invalid_reasons") or ())
    if failures:
        result = "Safety failure"
        passed = False
    elif not evaluation_valid:
        result = "Inconclusive due to sample, timing, alignment, runtime or budget gap"
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
        "conditional_support": passed,
        "n_eligible_clusters": n,
        "min_eligible_clusters": MIN_ELIGIBLE_CLUSTERS,
        "cluster_mean_delta": mean,
        "ci95": boot["ci95"],
        "improved_cluster_share": share,
        "absolute_effect": mean,
        "bootstrap": boot,
        "safety_failures": failures,
        "benefit_explained_only_by_unassigned": explained,
        "operational_clean": operational_clean,
        "execution_completed": execution_completed,
        "evaluation_valid": evaluation_valid,
        "evaluation_invalid_reasons": invalid_reasons,
        "n_operationally_unsuccessful": int(
            (coverage or {}).get("n_operationally_unsuccessful") or 0
        ),
        "n_degraded_conditional": int((coverage or {}).get("n_degraded_conditional") or 0),
        "sensitivity": dict(sensitivity or {}),
        "cluster_rows": list(cluster_rows),
    }


def latency_by_operation(runs: Sequence[Mapping[str, float | None]]) -> dict[str, Any]:
    rows = [
        row if any(name in row for name in LATENCY_DURATION_MARKS) else latency_record(dict(row))
        for row in runs
    ]
    durations: dict[str, dict[str, float | int | None]] = {}
    for name, endpoints in LATENCY_DURATION_MARKS.items():
        values: list[float] = []
        invalid = 0
        for row in rows:
            start, end = endpoints
            left, right = row.get(start), row.get(end)
            if left is not None and right is not None:
                value = float(right) - float(left)
            elif name == "admission_to_dispatch_s":
                value = None
            else:
                value = row.get(name)
            if value is None:
                continue
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0:
                invalid += 1
                continue
            values.append(numeric)
        dist = latency_distribution(values)
        durations[name] = {
            "n_available": int(dist["n"] or 0),
            "n_missing": len(rows) - len(values) - invalid,
            "n_invalid": invalid,
            "p50_s": dist["p50"],
            "p95_s": dist["p95"],
            "max_s": dist["max"],
        }
    violations = [
        int(bool(row["c5_deadline_violation"]))
        for row in rows
        if row.get("c5_deadline_violation") is not None
    ]
    empty_markers = [
        bool(row["_nontranslating_empty_parent"])
        for row in rows
        if "_nontranslating_empty_parent" in row
    ]
    all_empty_markers_available = len(empty_markers) == len(rows)
    return {
        "unit": "within_record_seconds",
        "n_records": len(rows),
        "denominator_scope": (
            "all_formed_parents" if all_empty_markers_available else "input_records"
        ),
        "n_nontranslating_empty_records": (
            sum(empty_markers) if all_empty_markers_available else None
        ),
        "durations": durations,
        "c5_deadline_violations": {
            "scope": "parent_marks",
            "n_available": len(violations),
            "n_missing": len(rows) - len(violations),
            "count": sum(violations) if violations else None,
        },
    }


def _coverage_parent_row(parent: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "meeting": parent.get("meeting"),
        "cluster_id": parent.get("cluster_id") or parent.get("meeting"),
        "parent_id": parent.get("parent_id"),
        "status": parent.get("status") or "unsuccessful",
        "outcome": parent.get("outcome"),
        "seal_reason": parent.get("seal_reason"),
        "text_authority": parent.get("text_authority"),
        "failure_reason": parent.get("failure_reason"),
        "accepted_text": parent.get("text") or "",
        "conserved": parent.get("conserved"),
    }


def _conditional_parent_row(parent: Mapping[str, Any]) -> dict[str, Any]:
    row = _coverage_parent_row(parent)
    row["status"] = "degraded"
    for arm_key in ("r0", "r2"):
        arm = parent.get(arm_key) or {}
        contamination = arm.get("contamination") or {}
        row[f"{arm_key}_proportion"] = contamination.get("proportion")
        row[f"{arm_key}_attributable_chars"] = contamination.get("attributable_chars")
    row["conditional_ownership"] = True
    return row


def cluster_rows_from_pool(
    parents: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for parent in parents:
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
        coverage = coverage_record(r0=r0_row, r2=r2_row)
        coverage_parts.append(coverage)
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
                "newly_unassigned_chars": coverage["newly_unassigned_chars"],
                "r0_unaligned_chars": coverage["r0_unaligned_chars"],
                "r2_unaligned_chars": coverage["r2_unaligned_chars"],
                "r0_mixed_chars": coverage["r0_mixed_chars"],
                "r2_mixed_chars": coverage["r2_mixed_chars"],
                "r2_unknown_chars": coverage["r2_unknown_chars"],
                "worst_case_r2": coverage["worst_case_r2"],
                "benefit_explained_only_by_unassigned": coverage[
                    "benefit_explained_only_by_unassigned"
                ],
            }
        )
    return rows, coverage_parts


def aggregate_cluster_parents(
    parents: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    pool: list[Mapping[str, Any]] = []
    excluded = 0
    incomplete_source: list[dict[str, Any]] = []
    unsuccessful_parents: list[dict[str, Any]] = []
    degraded_parents: list[dict[str, Any]] = []
    pool_exclusions: list[dict[str, Any]] = []
    for parent in parents:
        outcome = operational_outcome(parent)
        if parent.get("incomplete") or parent.get("outage"):
            incomplete_source.append(
                {
                    **_coverage_parent_row(parent),
                    "operational_outcome": outcome,
                    "accounted": bool(parent.get("accounted", True)),
                }
            )
            continue
        if outcome == "degraded_prefix":
            degraded_parents.append(
                {**_conditional_parent_row(parent), "operational_outcome": outcome}
            )
        if outcome in {"failed", "expired", "cancelled"} or outcome.startswith("unknown"):
            unsuccessful_parents.append(
                {**_coverage_parent_row(parent), "operational_outcome": outcome}
            )
        member, reason = primary_pool_membership(parent)
        if not member:
            if reason == "non_sequential":
                excluded += 1
            pool_exclusions.append(
                {
                    **_coverage_parent_row(parent),
                    "operational_outcome": outcome,
                    "pool_exclusion": reason,
                }
            )
            continue
        pool.append(parent)
    rows, coverage_parts = cluster_rows_from_pool(pool)
    explained = any(part.get("benefit_explained_only_by_unassigned") for part in coverage_parts)
    census = operational_census(parents)
    operational_clean = not unsuccessful_parents and not degraded_parents and not incomplete_source
    coverage = {
        "benefit_explained_only_by_unassigned": explained,
        "n_operationally_unsuccessful": len(unsuccessful_parents),
        "n_degraded_conditional": len(degraded_parents),
        "n_incomplete_source_parents": len(incomplete_source),
        "operational_clean": operational_clean,
        "unsuccessful_parents": unsuccessful_parents,
        "degraded_parents": degraded_parents,
        "incomplete_source_parents": incomplete_source,
    }
    return {
        "cluster_rows": rows,
        "n_sequential_parents": len(pool),
        "n_non_sequential_excluded": excluded,
        "n_incomplete_source_parents": len(incomplete_source),
        "n_operationally_unsuccessful": len(unsuccessful_parents),
        "n_degraded_conditional": len(degraded_parents),
        "unsuccessful_parents": unsuccessful_parents,
        "degraded_parents": degraded_parents,
        "pool_exclusions": pool_exclusions,
        "operational_census": census,
        "coverage": coverage,
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


OPERATIONAL_OUTCOMES = (
    "final_nonempty",
    "empty",
    "degraded_prefix",
    "failed",
    "expired",
    "cancelled",
)
SOURCE_SAMPLE_FIELDS = (
    "input_source_samples",
    "fed_source_samples",
    "synthetic_hangover_samples",
    "fed_total_source_samples",
    "chunked_source_samples",
    "unprocessed_source_samples",
    "buffered_source_samples",
    "dropped_tail_source_samples",
    "flush_pad_source_samples",
)


def operational_outcome(parent: Mapping[str, Any]) -> str:
    outcome = str(parent.get("outcome") or "")
    text = str(parent.get("text") or "")
    authority = str(parent.get("text_authority") or "")
    if text and authority == "degraded":
        return "degraded_prefix"
    if outcome in {"failed", "expired", "cancelled"}:
        return outcome
    if outcome == "empty" or not text:
        return "empty"
    if outcome == "final":
        if authority == "authoritative":
            return "final_nonempty"
        return f"unknown:authority:{authority or 'missing'}"
    return f"unknown:outcome:{outcome or 'missing'}"


def _arm_contamination(parent: Mapping[str, Any], arm: str) -> Mapping[str, Any]:
    payload = parent.get(arm) or parent.get(arm.upper()) or {}
    return payload.get("contamination") or payload


def _paired_delta(parent: Mapping[str, Any]) -> float | None:
    r0 = _arm_contamination(parent, "r0")
    r2 = _arm_contamination(parent, "r2")
    if not (r0.get("eligible") and r2.get("eligible")):
        return None
    if r0.get("proportion") is None or r2.get("proportion") is None:
        return None
    return float(r2["proportion"]) - float(r0["proportion"])


def primary_pool_membership(parent: Mapping[str, Any]) -> tuple[bool, str | None]:
    if parent.get("incomplete") or parent.get("outage"):
        return False, "incomplete_source_or_missing_parent"
    if not parent.get("sequential_target"):
        return False, "non_sequential"
    if not str(parent.get("text") or ""):
        return False, "empty_text"
    outcome = operational_outcome(parent)
    if outcome not in {"final_nonempty", "degraded_prefix"}:
        return False, f"operational_outcome:{outcome}"
    if _paired_delta(parent) is None:
        if overlap_unassessable(parent)[0]:
            return False, OVERLAP_UNASSESSABLE
        return False, "missing_paired_score"
    return True, None


def _census_bucket(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = {name: 0 for name in OPERATIONAL_OUTCOMES}
    unknown = 0
    for row in rows:
        outcome = str(row.get("operational_outcome") or "")
        if outcome in counts:
            counts[outcome] += 1
        else:
            unknown += 1
    total = len(rows)
    return {
        "denominator": total,
        "counts": counts,
        "rates": {name: (counts[name] / total if total else None) for name in OPERATIONAL_OUTCOMES},
        "n_unknown_outcome": unknown,
    }


def _census_group(
    rows: Sequence[Mapping[str, Any]],
    key_name: str,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(key_name) or "unknown"), []).append(row)
    return {key: _census_bucket(items) for key, items in sorted(grouped.items())}


def operational_census(
    parents: Sequence[Mapping[str, Any]],
    *,
    phase: str | None = None,
    source_rows: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    rows = [
        {
            "meeting": parent.get("meeting"),
            "cluster_id": parent.get("cluster_id") or parent.get("meeting"),
            "parent_id": parent.get("parent_id"),
            "operational_outcome": operational_outcome(parent),
            "outcome": parent.get("outcome"),
            "terminal_outcome": parent.get("terminal_outcome"),
            "text_authority": parent.get("text_authority"),
            "failure_reason": parent.get("failure_reason"),
            "status": parent.get("status"),
            "phase": parent.get("phase"),
            "degraded": bool(parent.get("degraded")),
            "clean_completion": bool(parent.get("clean_completion")),
            "accounted": bool(parent.get("accounted", True)),
            "accepted_chars": len(str(parent.get("text") or "")),
        }
        for parent in parents
    ]
    return {
        "unit": "formed_parent",
        "denominator_policy": (
            "all formed parents; operationally failed, expired, empty or cancelled parents are "
            "counted and never dropped from the denominator"
        ),
        "n_formed_parents": len(rows),
        "overall": _census_bucket(rows),
        "by_meeting": _census_group(rows, "meeting"),
        "by_cluster": _census_group(rows, "cluster_id"),
        "by_phase": (
            _census_group(rows, "phase")
            if any(row["phase"] for row in rows)
            else ({phase: _census_bucket(rows)} if phase else {})
        ),
        "unknown_outcome_rows": [
            row for row in rows if str(row["operational_outcome"]).startswith("unknown")
        ],
        "outcome_rows": rows,
        "source_accounting": dict(source_rows or {}),
        "source_denominator_policy": (
            "source that never forms a parent keeps its own accounting row and is not folded "
            "into a parent denominator"
        ),
    }


def source_accounting_rows(
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for case in cases:
        meeting = str(case.get("meeting") or "unknown")
        capture = dict(case.get("capture_timing") or {})
        rows[meeting] = {
            "meeting": meeting,
            "formed_parents": len(list(case.get("parents") or ())),
            **{name: capture.get(name) for name in SOURCE_SAMPLE_FIELDS},
        }
    return rows


def source_samples_supplied(capture: Mapping[str, Any]) -> int | None:
    """Real fed source plus explicit EOF hangover and the final-frame pad."""
    fed = capture.get("fed_source_samples")
    pad = capture.get("flush_pad_source_samples")
    if fed is None or pad is None:
        return None
    hangover = int(capture.get("synthetic_hangover_samples") or 0)
    return int(fed) + hangover + int(pad)


def source_samples_accounted(capture: Mapping[str, Any]) -> int | None:
    """Samples either chunked downstream or explicitly retained as loss/debt."""
    names = (
        "chunked_source_samples",
        "unprocessed_source_samples",
        "dropped_tail_source_samples",
        "buffered_source_samples",
    )
    if any(capture.get(name) is None for name in names):
        return None
    return sum(int(capture[name]) for name in names)


def case_execution_record(case: Mapping[str, Any]) -> dict[str, Any]:
    capture = dict(case.get("capture_timing") or {})
    parents = list(case.get("parents") or ())
    reasons: list[str] = []
    unprocessed = capture.get("unprocessed_source_samples")
    dropped = capture.get("dropped_tail_source_samples")
    buffered = capture.get("buffered_source_samples")
    if unprocessed:
        reasons.append(f"unprocessed_source_samples:{int(unprocessed)}")
    if dropped:
        reasons.append(f"dropped_tail_source_samples:{int(dropped)}")
    if buffered:
        reasons.append(f"buffered_source_samples:{int(buffered)}")
    declared = case.get("declared_source_samples")
    fed = capture.get("fed_source_samples")
    if declared is not None and fed is not None and int(fed) != int(declared):
        delta = int(declared) - int(fed)
        reasons.append(
            f"unconsumed_source_samples:{delta}"
            if delta > 0
            else f"overfed_source_samples:{-delta}"
        )
    supplied = source_samples_supplied(capture)
    accounted = source_samples_accounted(capture)
    if supplied is not None and accounted is not None and supplied != accounted:
        reasons.append(f"source_ledger_unreconciled:{supplied - accounted}")
    if case.get("budget_truncated"):
        reasons.append("budget_truncated")
    if case.get("provider_fault"):
        reasons.append("aborted_recording:provider_fault")
    sealed = case.get("sealed_segments")
    if sealed is not None and len(parents) < int(sealed):
        reasons.append(f"missing_parents:{int(sealed) - len(parents)}")
    unaccounted = [row for row in parents if not row.get("accounted", True)]
    if unaccounted:
        reasons.append(f"unaccounted_parents:{len(unaccounted)}")
    return {
        "execution_completed": not reasons,
        "execution_incomplete_reasons": reasons,
        "declared_source_samples": declared,
        "formed_parents": len(parents),
        "sealed_segments": sealed,
        "source_accounting": {name: capture.get(name) for name in SOURCE_SAMPLE_FIELDS},
    }


def case_evaluation_record(
    case: Mapping[str, Any],
    *,
    execution: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    execution = execution or case_execution_record(case)
    parents = list(case.get("parents") or ())
    reasons = list(execution.get("execution_incomplete_reasons") or ())
    conservation = [
        row for row in parents if str(row.get("text") or "") and row.get("conserved") is False
    ]
    if conservation:
        reasons.append(f"conservation_failure:{len(conservation)}")
    unaccounted = [row for row in parents if not row.get("accounted", True)]
    if unaccounted:
        reasons.append(f"missing_outcome_records:{len(unaccounted)}")
    unscorable = [
        row for row in parents if primary_pool_membership(row) == (False, "missing_paired_score")
    ]
    if unscorable:
        reasons.append(f"missing_paired_score:{len(unscorable)}")
    provenance = [row for row in parents if row.get("provenance_valid") is False]
    if provenance:
        reasons.append(f"invalid_provenance:{len(provenance)}")
    reasons.extend(str(item) for item in (case.get("timing_failures") or ()))
    tasks = list(case.get("task_failures") or ())
    if tasks:
        reasons.append(f"unhandled_task_failure:{len(tasks)}")
    guard = list(case.get("severe_guard_failures") or ())
    if guard:
        reasons.append(f"severe_guard:{len(guard)}")
    guard_summary = case_guard_summary(parents)
    reasons.extend(guard_summary["failures"])
    reasons.extend(guard_summary["coverage_reasons"])
    unknown = [row for row in parents if str(operational_outcome(row)).startswith("unknown")]
    if unknown:
        reasons.append(f"unknown_outcome_records:{len(unknown)}")
    overlap = list(guard_summary["overlap_unassessable_parents"])
    return {
        "evaluation_valid": not reasons,
        "evaluation_invalid_reasons": reasons,
        "overlap_unassessable_parents": overlap,
        "n_overlap_unassessable": len(overlap),
    }


def case_safety_failures(case: Mapping[str, Any]) -> list[str]:
    failures = [str(item) for item in (case.get("safety_failures") or ())]
    for row in case.get("parents") or ():
        if str(row.get("text") or "") and row.get("conserved") is False:
            failures.append(f"conservation_failure:{row.get('parent_id')}")
    failures.extend(case_guard_summary(list(case.get("parents") or ()))["failures"])
    return failures


def u8_case_report(case: Mapping[str, Any]) -> dict[str, Any]:
    execution = case_execution_record(case)
    evaluation = case_evaluation_record(case, execution=execution)
    parents = list(case.get("parents") or ())
    outcomes = [operational_outcome(row) for row in parents]
    unsuccessful = sum(1 for item in outcomes if item in {"failed", "expired", "cancelled"})
    degraded = sum(1 for item in outcomes if item == "degraded_prefix")
    return {
        "execution_completed": execution["execution_completed"],
        "execution_incomplete_reasons": execution["execution_incomplete_reasons"],
        "evaluation_valid": evaluation["evaluation_valid"],
        "evaluation_invalid_reasons": evaluation["evaluation_invalid_reasons"],
        "safety_failures": case_safety_failures(case),
        "guard": case_guard_summary(parents),
        "operational_clean": unsuccessful == 0 and degraded == 0,
        "n_operationally_unsuccessful": unsuccessful,
        "n_degraded_prefix": degraded,
        "formed_parents": len(parents),
        "operational_census": operational_census(
            parents,
            phase=case.get("phase"),
            source_rows=source_accounting_rows([case]),
        ),
        "source_accounting": execution["source_accounting"],
        "overlap_coverage": overlap_coverage_report([case], parents),
    }


def formed_parent_selection_bounds(
    parents: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    clusters: dict[str, dict[str, Any]] = {}
    for parent in parents:
        cluster = str(parent.get("cluster_id") or parent.get("meeting") or "unknown")
        row = clusters.setdefault(
            cluster,
            {
                "cluster_id": cluster,
                "N": 0,
                "M": 0,
                "S": 0.0,
                "M_empty": 0,
                "M_unavailable": 0,
                "M_unscorable": 0,
                "M_overlap_unassessable": 0,
            },
        )
        outcome = operational_outcome(parent)
        if parent.get("incomplete") or parent.get("outage"):
            row["M"] += 1
            row["M_unavailable"] += 1
            continue
        if outcome == "empty":
            row["M"] += 1
            row["M_empty"] += 1
            continue
        if outcome in {"failed", "expired", "cancelled"} or outcome.startswith("unknown"):
            row["M"] += 1
            row["M_unavailable"] += 1
            continue
        if not parent.get("sequential_target"):
            continue
        if not str(parent.get("text") or ""):
            row["M"] += 1
            row["M_empty"] += 1
            continue
        delta = _paired_delta(parent)
        if delta is None:
            row["M"] += 1
            row["M_unscorable"] += 1
            if overlap_unassessable(parent)[0]:
                row["M_overlap_unassessable"] += 1
            continue
        row["S"] += delta
        row["N"] += 1
    per_cluster: list[dict[str, Any]] = []
    for cluster in sorted(clusters):
        row = dict(clusters[cluster])
        total = row["N"] + row["M"]
        row["denominator"] = total
        row["lower"] = ((row["S"] - row["M"]) / total) if total else None
        row["upper"] = ((row["S"] + row["M"]) / total) if total else None
        per_cluster.append(row)
    lowers = [row["lower"] for row in per_cluster if row["lower"] is not None]
    uppers = [row["upper"] for row in per_cluster if row["upper"] is not None]
    mean_lower = (sum(lowers) / len(lowers)) if lowers else None
    mean_upper = (sum(uppers) / len(uppers)) if uppers else None
    if mean_lower is None or mean_upper is None:
        fragility = "unavailable"
    elif mean_lower <= 0 <= mean_upper:
        fragility = "bounds_cross_zero"
    else:
        fragility = "bounds_exclude_zero"
    return {
        "estimand": "formed_parent_selection_bounds",
        "note": (
            "Distinct diagnostic estimand on formed-parent selection; not a lexical-weighted "
            "primary confidence interval."
        ),
        "per_cluster": per_cluster,
        "equal_cluster_mean_lower": mean_lower,
        "equal_cluster_mean_upper": mean_upper,
        "n_clusters": len(per_cluster),
        "n_clusters_included": len(lowers),
        "N_total": sum(row["N"] for row in per_cluster),
        "M_total": sum(row["M"] for row in per_cluster),
        "M_unscorable_total": sum(row["M_unscorable"] for row in per_cluster),
        "M_overlap_unassessable_total": sum(row["M_overlap_unassessable"] for row in per_cluster),
        "overlap_unassessable_subset": [
            {"cluster_id": row["cluster_id"], "count": row["M_overlap_unassessable"]}
            for row in per_cluster
            if row["M_overlap_unassessable"]
        ],
        "fragility": fragility,
    }


def parent_content_ranges(parent: Mapping[str, Any]) -> list[list[int]]:
    source = parent.get("content_ranges")
    if not isinstance(source, (list, tuple)):
        receipt = parent.get("receipt")
        source = receipt.get("content_ranges") if isinstance(receipt, Mapping) else None
    ranges: list[list[int]] = []
    for item in source or ():
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        start, end = item
        if start is None or end is None:
            continue
        start, end = int(start), int(end)
        if end < start:
            continue
        ranges.append([start, end])
    return ranges


def _source_interval(ranges: Sequence[Sequence[int]]) -> list[int] | None:
    if not ranges:
        return None
    return [min(int(row[0]) for row in ranges), max(int(row[1]) for row in ranges)]


def _annotation_tokens_by_meeting(
    parents: Sequence[Mapping[str, Any]],
) -> dict[str, list[int]]:
    counts: dict[str, set[int]] = {}
    for parent in parents:
        guard = parent.get("guard")
        checked = dict(guard.get("checked") or {}) if isinstance(guard, Mapping) else {}
        if "annotation_tokens" not in checked:
            continue
        meeting = str(parent.get("meeting") or "unknown")
        counts.setdefault(meeting, set()).add(int(checked["annotation_tokens"]))
    return {meeting: sorted(values) for meeting, values in sorted(counts.items())}


def _overlap_parent_row(parent: Mapping[str, Any]) -> dict[str, Any]:
    checked = {}
    guard = parent.get("guard")
    if isinstance(guard, Mapping):
        checked = dict(guard.get("checked") or {})
    excluded = dict(checked.get("excluded") or {})
    r0 = _arm_contamination(parent, "r0")
    r2 = _arm_contamination(parent, "r2")
    status, computed = guard_coverage_status(parent)
    qualified, reason = overlap_unassessable(parent)
    span = parent.get("span")
    token_envelope = None
    if (
        isinstance(span, (list, tuple))
        and len(span) == 2
        and all(item is not None for item in span)
    ):
        token_envelope = [int(span[0]), int(span[1])]
    source_ranges = parent_content_ranges(parent)
    return {
        "parent_id": str(parent.get("parent_id") or parent.get("index") or "parent"),
        "meeting": parent.get("meeting"),
        "cluster_id": parent.get("cluster_id") or parent.get("meeting"),
        "guard_computed": computed,
        "coverage_status": status,
        "qualification_reason": reason,
        "overlap_unassessable": qualified,
        "token_envelope": token_envelope,
        "source_ranges": [list(row) for row in source_ranges],
        "source_interval": _source_interval(source_ranges),
        "source_scope": "optional_segment_content_ranges",
        "accepted_chars": len(str(parent.get("text") or "")),
        "checked_lexical_tokens": int(checked.get("lexical_tokens") or 0),
        "annotation_tokens": (
            int(checked["annotation_tokens"]) if "annotation_tokens" in checked else None
        ),
        "annotation_source": parent.get("meeting"),
        "annotation_source_scope": "meeting_annotation_source",
        "excluded_tokens": {
            "punctuation_only": int(excluded.get("punctuation_only") or 0),
            "mixed": int(excluded.get("mixed") or 0),
            "unaligned": int(excluded.get("unaligned") or 0),
        },
        "r0_mixed_chars": int(r0.get("mixed_chars") or 0),
        "r0_unaligned_chars": int(r0.get("unaligned_chars") or 0),
        "r2_mixed_chars": int(r2.get("mixed_chars") or 0),
        "r2_unaligned_chars": int(r2.get("unaligned_chars") or 0),
        "r0_contamination_reason": r0.get("reason"),
        "r2_contamination_reason": r2.get("reason"),
    }


def _zero_coverage_candidate(parent: Mapping[str, Any]) -> bool:
    if not guard_coverage_required(parent):
        return False
    status, computed = guard_coverage_status(parent)
    return (not computed) or status in ZERO_COVERAGE_STATUSES


def _overlap_scope(parents: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    formed = list(parents)
    accepted = [row for row in formed if guard_coverage_required(row)]
    qualified = [row for row in formed if overlap_unassessable(row)[0]]
    missing_guard = [row for row in accepted if not guard_coverage_status(row)[1]]
    accepted_chars = sum(len(str(row.get("text") or "")) for row in accepted)
    qualified_chars = sum(len(str(row.get("text") or "")) for row in qualified)
    return {
        "formed_parents": len(formed),
        "accepted_nonempty_parents": len(accepted),
        "accepted_chars": accepted_chars,
        "qualified_parents": len(qualified),
        "qualified_accepted_chars": qualified_chars,
        "qualified_parent_rate_of_formed": ((len(qualified) / len(formed)) if formed else None),
        "qualified_parent_rate_of_accepted_nonempty": (
            (len(qualified) / len(accepted)) if accepted else None
        ),
        "accepted_char_coverage": (qualified_chars / accepted_chars) if accepted_chars else None,
        "missing_guard_parents": len(missing_guard),
        "annotation_source_scope": "meeting_annotation_source",
        "annotation_tokens": _annotation_tokens_by_meeting(formed),
        "source_interval": _source_interval(
            [row for parent in formed for row in parent_content_ranges(parent)]
        ),
        "source_ranges_parents": sum(1 for parent in formed if parent_content_ranges(parent)),
        "accepted_parents_missing_source_ranges": sum(
            1 for parent in accepted if not parent_content_ranges(parent)
        ),
        "qualified_source_ranges": {
            str(parent.get("parent_id") or parent.get("index")): parent_content_ranges(parent)
            for parent in qualified
        },
        "zero_coverage_parents": len([row for row in formed if _zero_coverage_candidate(row)]),
    }


def overlap_coverage_report(
    cases: Sequence[Mapping[str, Any]],
    parents: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    case_list = list(cases)
    formed = list(parents)
    case_rows = [
        {
            "meeting": case.get("meeting"),
            "phase": case.get("phase"),
            **_overlap_scope(list(case.get("parents") or ())),
        }
        for case in case_list
    ]
    if not formed:
        formed = [row for case in case_list for row in (case.get("parents") or ())]
    phase_by_meeting = {
        str(case.get("meeting")): str(case.get("phase"))
        for case in case_list
        if case.get("phase") is not None
    }
    cluster_groups: dict[str, list[Mapping[str, Any]]] = {}
    phase_groups: dict[str, list[Mapping[str, Any]]] = {}
    for parent in formed:
        cluster = str(parent.get("cluster_id") or parent.get("meeting") or "unknown")
        cluster_groups.setdefault(cluster, []).append(parent)
        phase = parent.get("phase")
        if phase is None:
            phase = phase_by_meeting.get(str(parent.get("meeting")))
        if phase is not None:
            phase_groups.setdefault(str(phase), []).append(parent)
    row_source = [row for case in case_list for row in (case.get("parents") or ())] or formed
    rows = [_overlap_parent_row(row) for row in row_source if _zero_coverage_candidate(row)]
    return {
        "unit": "formed_parent",
        "claim_scope": (
            "Uniquely GT-attributable accepted lexical text. Measured overlap-unassessable parents "
            "keep their accepted text, authoritative source ranges and translations, contribute no invented "
            "score or characters to the point estimate, stay in coverage and in the conservative "
            "formed-parent selection bounds, and carry no safety claim for overlapping text."
        ),
        "denominators": (
            "formed parents and accepted nonempty parents per case, cluster and phase; "
            "accepted-character coverage at the same scopes"
        ),
        "source_note": (
            "source_ranges lists the authoritative optional-segment content ranges recorded with "
            "the accepted text, preserving recorded order and gaps; source_interval is only the "
            "envelope of those ranges and implies no continuity between them. token_envelope is "
            "the lexical GT token envelope, not the accepted-text extent. annotation_tokens "
            "carries annotation_source_scope=meeting_annotation_source: it counts the meeting "
            "annotation source, not per-parent GT coverage."
        ),
        "overall": _overlap_scope(formed),
        "by_case": case_rows,
        "by_cluster": {
            cluster: _overlap_scope(items) for cluster, items in sorted(cluster_groups.items())
        },
        "by_phase": {phase: _overlap_scope(items) for phase, items in sorted(phase_groups.items())},
        "parents": rows,
    }


def complete_only_sensitivity(
    parents: Sequence[Mapping[str, Any]],
    *,
    primary_mean: float | None,
) -> dict[str, Any]:
    complete = [
        parent
        for parent in parents
        if primary_pool_membership(parent)[0] and operational_outcome(parent) == "final_nonempty"
    ]
    rows, _coverage = cluster_rows_from_pool(complete)
    deltas = [float(row["delta"]) for row in rows if row.get("delta") is not None]
    boot = paired_cluster_bootstrap(deltas)
    mean = boot["mean"]
    return {
        "n_clusters": len(deltas),
        "n_parents": len(complete),
        "cluster_mean_delta": mean,
        "ci95": boot["ci95"],
        "change_from_primary": (
            None if mean is None or primary_mean is None else mean - primary_mean
        ),
    }


def u8_phase_report(
    *,
    parents: Sequence[Mapping[str, Any]],
    cases: Sequence[Mapping[str, Any]] = (),
    phase: str | None = None,
    primary_mean: float | None = None,
) -> dict[str, Any]:
    case_rows = list(cases)
    source_rows = source_accounting_rows(case_rows)
    execution_reasons: list[str] = []
    evaluation_reasons: list[str] = []
    safety: list[str] = []
    for case in case_rows:
        report = u8_case_report(case)
        meeting = str(case.get("meeting") or "unknown")
        execution_reasons.extend(
            f"{meeting}:{item}" for item in report["execution_incomplete_reasons"]
        )
        evaluation_reasons.extend(
            f"{meeting}:{item}" for item in report["evaluation_invalid_reasons"]
        )
        safety.extend(f"{meeting}:{item}" for item in report["safety_failures"])
    execution_completed = not execution_reasons
    evaluation_valid = not evaluation_reasons
    bounds = formed_parent_selection_bounds(parents)
    sensitivity = {
        "complete_only": complete_only_sensitivity(parents, primary_mean=primary_mean),
        "formed_parent_selection_bounds": bounds,
        "leave_one_cluster_out": None,
    }
    phased_parents = [
        {**parent, "phase": case.get("phase")}
        for case in case_rows
        for parent in (case.get("parents") or ())
    ]
    return {
        "execution_completed": execution_completed,
        "execution_incomplete_reasons": execution_reasons,
        "evaluation_valid": evaluation_valid,
        "evaluation_invalid_reasons": evaluation_reasons,
        "safety_failures": safety,
        "overlap_coverage": overlap_coverage_report(case_rows, parents),
        "operational_census": operational_census(
            phased_parents or parents,
            phase=phase,
            source_rows=source_rows,
        ),
        "guard": guard_aggregate(case_rows, parents),
        "source_accounting": source_rows,
        "sensitivity": sensitivity,
    }
