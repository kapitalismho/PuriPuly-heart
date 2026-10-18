from __future__ import annotations

from collections import Counter, defaultdict

from bench.align import Item, align, edit_counts
from bench.reference import Reference
from bench.text import tokens as tokenize
from bench.units import CaseRun, Unit

PARAMETERS = {
    "overlap_margin_s": 0.1,
    "short_turn_s": 1.0,
    "alignment_tolerance_s": 0.5,
    "same_speaker_split_clearance_s": 0.3,
    "burst_window_s": 0.5,
    "burst_explained_s": 0.5,
    "mixed_purity": 0.8,
    "return_gap_s": 10.0,
    "edge_words": 3,
    "duplicate_scan_tokens": 5,
    "late_window_s": 1.0,
    "preferred_window_s": 0.5,
}


class _Hyp:
    __slots__ = ("token", "unit")

    def __init__(self, token: str, unit: int) -> None:
        self.token = token
        self.unit = unit


def _hypothesis(run: CaseRun, language: str) -> tuple[list[_Hyp], list[Item]]:
    hyps: list[_Hyp] = []
    items: list[Item] = []
    for index, unit in enumerate(run.units):
        lo, hi = (unit.lo, unit.hi) if unit.spans else (float("-inf"), float("inf"))
        for token in tokenize(unit.text, language):
            hyps.append(_Hyp(token, index))
            items.append(Item(token, lo, hi))
    return hyps, items


def _hungarian(weights: list[list[float]]) -> list[int | None]:
    rows = len(weights)
    cols = len(weights[0]) if rows else 0
    size = max(rows, cols)
    if size == 0:
        return []
    peak = max((w for row in weights for w in row), default=0.0)
    cost = [
        [peak - (weights[i][j] if i < rows and j < cols else 0.0) for j in range(size)]
        for i in range(size)
    ]
    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    p = [0] * (size + 1)
    way = [0] * (size + 1)
    for i in range(1, size + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], float("inf"), 0
            for j in range(1, size + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    result: list[int | None] = [None] * rows
    for j in range(1, size + 1):
        i = p[j] - 1
        if i < rows and j - 1 < cols and weights[i][j - 1] > 0:
            result[i] = j - 1
    return result


def _mapping(ref: Reference, run: CaseRun, carried: list[int | None], hyps: list[_Hyp]) -> dict:
    counts: Counter = Counter()
    for token, hyp in zip(ref.tokens, carried):
        if hyp is None:
            continue
        final = run.units[hyps[hyp].unit].final
        if final is not None and final.identified:
            counts[(final.key, token.speaker)] += 1
    keys = sorted({k for k, _ in counts})
    speakers = sorted({t.speaker for t in ref.tokens})
    weights = [[float(counts[(k, s)]) for s in speakers] for k in keys]
    assigned = _hungarian(weights)
    return {k: (speakers[a] if a is not None else None) for k, a in zip(keys, assigned)}


def _unit_label(unit: Unit, mapping: dict, speaker: str) -> str:
    final = unit.final
    if final is None or not final.identified:
        return "gray"
    if mapping.get(final.key) == speaker:
        return "correct"
    return "wrong"


def score_case(ref: Reference, run: CaseRun) -> dict:
    p = PARAMETERS
    language = ref.language
    hyps, items = _hypothesis(run, language)
    ref_items = [Item(t.token, t.window[0], t.window[1]) for t in ref.tokens]
    carried = align(ref_items, items, tolerance=p["alignment_tolerance_s"])
    unit_of = [None if h is None else hyps[h].unit for h in carried]
    mapping = _mapping(ref, run, carried, hyps)
    units = run.units

    unit_speakers: dict[int, Counter] = defaultdict(Counter)
    for token, u in zip(ref.tokens, unit_of):
        if u is not None:
            unit_speakers[u][token.speaker] += 1

    def majority(u: int) -> str | None:
        counter = unit_speakers.get(u)
        return counter.most_common(1)[0][0] if counter else None

    # Attribution (word-weighted, final presented assignment).
    attribution = Counter()
    for token, u in zip(ref.tokens, unit_of):
        if u is None:
            attribution["not_recognized"] += 1
            continue
        final = units[u].final
        displayed = "unknown" if final is None else final.displayed
        if final is not None and final.identified:
            correct = mapping.get(final.key) == token.speaker
            if displayed == "colored":
                attribution["colored_correct" if correct else "colored_wrong"] += 1
            else:
                attribution[f"{displayed}_{'correct' if correct else 'wrong'}"] += 1
        else:
            attribution[f"gray_{displayed}"] += 1
    recognized = sum(v for k, v in attribution.items() if k != "not_recognized")

    key_speakers: dict[str, Counter] = defaultdict(Counter)
    speaker_keys: dict[str, Counter] = defaultdict(Counter)
    for token, u in zip(ref.tokens, unit_of):
        if u is None:
            continue
        final = units[u].final
        if final is not None and final.identified:
            key_speakers[final.key][token.speaker] += 1
            speaker_keys[token.speaker][final.key] += 1
    fragmentation = {
        s: sum(1 for c in keys.values() if c >= 2) for s, keys in sorted(speaker_keys.items())
    }
    collision = {
        k: sum(1 for c in speakers.values() if c >= 2)
        for k, speakers in sorted(key_speakers.items())
    }

    mixed_units = colored_mixed_units = scored_units = 0
    for u, counter in unit_speakers.items():
        total = sum(counter.values())
        if total == 0:
            continue
        scored_units += 1
        purity = counter.most_common(1)[0][1] / total
        if purity < p["mixed_purity"]:
            mixed_units += 1
            if units[u].final is not None and units[u].final.colored:
                colored_mixed_units += 1

    # Return continuity.
    returns = []
    for segment in ref.segments:
        previous = [
            s for s in ref.segments if s.speaker == segment.speaker and s.end <= segment.start
        ]
        if not previous:
            continue
        last = previous[-1]
        if segment.start - last.end < p["return_gap_s"]:
            continue
        if not any(
            s.speaker != segment.speaker and s.start >= last.end and s.end <= segment.start
            for s in ref.segments
        ):
            continue

        def dominant(seg):
            keys = Counter()
            for t in seg.tokens:
                u = unit_of[t]
                if u is not None and units[u].final is not None and units[u].final.identified:
                    keys[units[u].final.key] += 1
            return keys.most_common(1)[0][0] if keys else None

        before, after = dominant(last), dominant(segment)
        outcome = (
            "gray"
            if before is None or after is None
            else ("same" if before == after else "different")
        )
        returns.append(
            {"segment": segment.index, "gap_s": segment.start - last.end, "outcome": outcome}
        )

    # Transitions and early boundaries.
    transitions = []
    for tr in ref.transitions:
        before, after = ref.segments[tr.before], ref.segments[tr.after]
        a_tail = before.tokens[-p["edge_words"] :]
        b_head = after.tokens[: p["edge_words"]]
        ua = next((unit_of[t] for t in reversed(a_tail) if unit_of[t] is not None), None)
        ub = next((unit_of[t] for t in b_head if unit_of[t] is not None), None)
        row = {
            "id": tr.key,
            "kind": tr.kind,
            "gap_s": round(tr.gap, 4),
            "short_successor": tr.short_successor,
            "returning": tr.returning,
            "before_speaker": before.speaker,
            "after_speaker": after.speaker,
            "a_end_s": before.end,
            "b_start_s": after.start,
            "region_end_s": ref.region_end(before.end),
        }
        if ua is None or ub is None:
            row["outcome"] = "content_missing"
            transitions.append(row)
            continue
        separated = ua != ub
        unit_a = units[ua]
        release = unit_a.first_readable
        nearby = sorted({unit_of[t] for t in a_tail + b_head if unit_of[t] is not None})
        if separated:
            outcome = "separated"
        elif len(nearby) > 1:
            outcome = "misplaced_boundary"
        else:
            outcome = "merged"
        row.update(
            {
                "outcome": outcome,
                "separated": separated,
                "predecessor_release_s": release,
                "predecessor_latency_s": None if release is None else release - before.end,
                "early_release": bool(
                    separated and release is not None and release < row["region_end_s"]
                ),
                "boundary_error_s": None,
                "a_unit": ua,
                "b_unit": ub,
                "a_label": _unit_label(unit_a, mapping, before.speaker),
                "b_label": _unit_label(units[ub], mapping, after.speaker),
            }
        )
        cut_unit = ua if separated else (nearby[0] if len(nearby) > 1 else None)
        if cut_unit is not None and units[cut_unit].spans:
            boundary = units[cut_unit].hi
            if boundary > after.start:
                error = boundary - after.start
            elif boundary < before.end:
                error = boundary - before.end
            else:
                error = 0.0
            row["boundary_error_s"] = round(error, 4)
        deleted = sum(1 for t in a_tail + b_head if unit_of[t] is None)
        misplaced = sum(
            1
            for t in a_tail
            if unit_of[t] is not None and majority(unit_of[t]) not in (None, before.speaker)
        ) + sum(
            1
            for t in b_head
            if unit_of[t] is not None and majority(unit_of[t]) not in (None, after.speaker)
        )
        aligned_hyps = {h for h in carried if h is not None}
        a_words = {ref.tokens[t].token for t in a_tail}
        b_words = {ref.tokens[t].token for t in b_head}
        duplicated = 0
        if cut_unit is not None:
            head_unit = ub if separated else cut_unit + 1
            head = [i for i, h in enumerate(hyps) if h.unit == head_unit][
                : p["duplicate_scan_tokens"]
            ]
            tail = [i for i, h in enumerate(hyps) if h.unit == cut_unit][
                -p["duplicate_scan_tokens"] :
            ]
            duplicated += sum(1 for i in head if i not in aligned_hyps and hyps[i].token in a_words)
            duplicated += sum(1 for i in tail if i not in aligned_hyps and hyps[i].token in b_words)
        row["edge"] = {"deleted": deleted, "misplaced": misplaced, "duplicated": duplicated}
        transitions.append(row)

    # Speaker-triggered cuts reported by the batch route.
    points = [(t["id"], t["a_end_s"], t["b_start_s"]) for t in transitions]
    speaker_cuts = []
    for item in run.slices:
        if item.reason != "speaker":
            continue
        nearest = min(
            points,
            key=lambda p_: (
                0.0
                if p_[1] <= item.boundary_s <= p_[2]
                else min(abs(item.boundary_s - p_[1]), abs(item.boundary_s - p_[2]))
            ),
            default=None,
        )
        distance = None
        if nearest is not None:
            distance = (
                0.0
                if nearest[1] <= item.boundary_s <= nearest[2]
                else min(abs(item.boundary_s - nearest[1]), abs(item.boundary_s - nearest[2]))
            )
        matched = nearest is not None and distance <= p["burst_explained_s"]
        speaker_cuts.append(
            {
                "boundary_s": round(item.boundary_s, 4),
                "evidence_ready_s": item.evidence_ready_s,
                "submitted_s": item.submitted_s,
                "transition": nearest[0] if matched else None,
                "distance_s": None if distance is None else round(distance, 4),
            }
        )
    by_transition = {c["transition"]: c for c in speaker_cuts if c["transition"] is not None}
    for row in transitions:
        cut = by_transition.get(row["id"])
        row["speaker_cut"] = cut is not None
        if cut is not None and cut["evidence_ready_s"] is not None:
            row["cut_detection_s"] = round(cut["evidence_ready_s"] - row["b_start_s"], 4)
            row["cut_submission_s"] = round(cut["submitted_s"] - row["b_start_s"], 4)

    # Unit boundaries: same-speaker splits and bursts.
    boundaries = []
    for i in range(len(units) - 1):
        before_tokens = [t for t, u in enumerate(unit_of) if u == i]
        after_tokens = [t for t, u in enumerate(unit_of) if u == i + 1]
        boundary_time = units[i].hi if units[i].spans else None
        kind = "unscored"
        if before_tokens and after_tokens:
            w1 = max(before_tokens, key=lambda t: ref.tokens[t].end)
            w2 = min(after_tokens, key=lambda t: ref.tokens[t].start)
            t1, t2 = ref.tokens[w1], ref.tokens[w2]
            interposed = any(
                tok.speaker != t1.speaker
                and tok.end > t1.end - p["same_speaker_split_clearance_s"]
                and tok.start < t2.start + p["same_speaker_split_clearance_s"]
                for tok in ref.tokens
            )
            if t1.speaker != t2.speaker:
                kind = "speaker_change"
            elif t2.start - t1.end >= ref.gap_s:
                kind = "pause"
            elif interposed:
                kind = "near_other_speaker"
            else:
                kind = "same_speaker_split"
        boundaries.append({"after_unit": i, "time_s": boundary_time, "kind": kind})
    transition_points = [(t["id"], (t["a_end_s"] + t["b_start_s"]) / 2) for t in transitions]
    bursts = 0
    timed = [b for b in boundaries if b["time_s"] is not None]
    for first, second in zip(timed, timed[1:]):
        if second["time_s"] - first["time_s"] >= p["burst_window_s"]:
            continue
        near_first = {
            tid
            for tid, tp in transition_points
            if abs(tp - first["time_s"]) <= p["burst_explained_s"]
        }
        near_second = {
            tid
            for tid, tp in transition_points
            if abs(tp - second["time_s"]) <= p["burst_explained_s"]
        }
        if not near_first or not near_second or near_first == near_second and len(near_first) < 2:
            bursts += 1

    # Short segments (interjections).
    short = []
    for segment in ref.segments:
        if segment.duration > p["short_turn_s"]:
            continue
        aligned = [unit_of[t] for t in segment.tokens if unit_of[t] is not None]
        owned = set(aligned)
        separate = bool(owned) and all(set(unit_speakers[u]) == {segment.speaker} for u in owned)
        labels = {_unit_label(units[u], mapping, segment.speaker) for u in owned}
        short.append(
            {
                "segment": segment.index,
                "speaker": segment.speaker,
                "duration_s": round(segment.duration, 3),
                "retained": len(aligned) / max(1, len(segment.tokens)),
                "separate": separate,
                "identity": (
                    "correct"
                    if labels == {"correct"}
                    else ("wrong" if "wrong" in labels else "gray")
                ),
            }
        )

    # Presentation timing.
    presentation = Counter()
    violations = Counter()
    delays = []
    for unit in units:
        if unit.first_readable is None:
            presentation["never_readable"] += 1
            continue
        at = unit.readable_assignment
        final = unit.final
        if (
            at is not None
            and at.identified
            and (at.palette_index is not None or at.palette_overflow)
        ):
            presentation["identified_before_readable"] += 1
            if final is not None and final.key != at.key:
                violations["identity_changed"] += 1
            continue
        if final is not None and final.identified and unit.settled_at is not None:
            delay = unit.settled_at - unit.first_readable
            delays.append(delay)
            if delay <= p["preferred_window_s"]:
                presentation["late_within_500ms"] += 1
            elif delay <= p["late_window_s"] + 1e-6:
                presentation["late_within_1s"] += 1
            else:
                violations["late_after_deadline"] += 1
        else:
            presentation["stayed_gray"] += 1
        if unit.settle_events > 1:
            violations["multiple_settlements"] += 1

    # Turn completion latency (per speaker segment).
    completion = []
    for segment in ref.segments:
        aligned = [unit_of[t] for t in segment.tokens if unit_of[t] is not None]
        times = [units[u].first_readable for u in aligned]
        if not aligned or any(t is None for t in times):
            continue
        completion.append(
            {
                "segment": segment.index,
                "duration_s": round(segment.duration, 3),
                "complete": len(aligned) == len(segment.tokens),
                "latency_s": max(times) - segment.end,
            }
        )

    content = edit_counts([t.token for t in ref.tokens], [h.token for h in hyps])

    return {
        "case_id": ref.case_id,
        "language": language,
        "category": ref.category,
        "synthetic": ref.synthetic,
        "gap_s": ref.gap_s,
        "speech_s": ref.speech_seconds(),
        "reference_tokens": len(ref.tokens),
        "units": len(units),
        "invalid": run.invalid,
        "mapping": mapping,
        "attribution": dict(attribution),
        "recognized_tokens": recognized,
        "key_fragmentation": fragmentation,
        "key_collision": collision,
        "mixed_units": mixed_units,
        "colored_mixed_units": colored_mixed_units,
        "scored_units": scored_units,
        "returns": returns,
        "transitions": transitions,
        "boundaries": boundaries,
        "bursts": bursts,
        "speaker_cuts": speaker_cuts,
        "batch_slices": [
            {
                "reason": x.reason,
                "boundary_s": round(x.boundary_s, 4),
                "submitted_s": round(x.submitted_s, 4),
            }
            for x in run.slices
        ],
        "short_segments": short,
        "presentation": dict(presentation),
        "presentation_violations": dict(violations),
        "late_delays_s": delays,
        "completion": completion,
        "content": content,
        "unit_detail": [
            {
                "index": i,
                "text": u.text,
                "span_s": [u.lo, u.hi] if u.spans else None,
                "admitted_s": u.admitted_at,
                "first_readable_s": u.first_readable,
                "settled_s": u.settled_at,
                "readable": (
                    None if u.readable_assignment is None else u.readable_assignment.displayed
                ),
                "final": None if u.final is None else u.final.displayed,
                "key": None if u.final is None else u.final.key,
                "majority_speaker": majority(i),
            }
            for i, u in enumerate(units)
        ],
    }
