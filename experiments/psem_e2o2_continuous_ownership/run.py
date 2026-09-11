from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from experiments.psem_product_translation.capture import corrected_groups_for_capture

FREEZE = json.loads((EXP / "FREEZE.json").read_text(encoding="utf-8"))
STAGE2 = ROOT / "experiments" / "psem_repeatability_stage2"
OBSDIR = ROOT / "experiments" / "psem_phase_a_headroom" / "observations"
NS = "{http://nite.sourceforge.net/}"
TAU = float(FREEZE["profile"]["tau"])
FRAME = int(FREEZE["profile"]["frame_samples"])
CONFIRMATION = int(FREEZE["profile"]["confirmation_samples"])
HZ = int(FREEZE["profile"]["source_clock_hz"])
FAILURE_TYPES = list(FREEZE["failure_types"])
CASE_ORDER = ["NP1", "NP2", "NP3", "R1", "R2", "T1", "COMBINED", "BC1", "SINGLE_ES2009c", "SINGLE_ES2009d"]
MANIFEST = json.loads((EXP / "inputs" / "CAPTURE_MANIFEST.json").read_text(encoding="utf-8"))


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def write_json(p, obj):
    Path(p).write_text(json.dumps(obj, indent=1), encoding="utf-8")
    return sha256_file(p)


def conservation_check(accepted_tokens, groups):
    ids = [t.get("o") for t in accepted_tokens]
    ref_ids = [r.get("o") for g in groups for r in g.get("token_refs", [])]
    gidx = [g.get("idx") for g in groups]
    dup_groups = sorted(i for i, c in Counter(gidx).items() if c > 1)
    missing = sorted(set(ids) - set(ref_ids))
    unknown = sorted(set(ref_ids) - set(ids))
    text_equal = "".join(t.get("text", "") for t in accepted_tokens) == "".join(g.get("text", "") for g in groups)
    conserved = (not dup_groups and not missing and not unknown and text_equal)
    return {
        "conserved": conserved,
        "text_equal": text_equal,
        "n_accepted_tokens": len(ids),
        "n_groups": len(groups),
        "missing_token_refs": missing,
        "duplicate_group_ids": dup_groups,
        "unknown_ref_ids": unknown,
    }


def throwaway_controls(accepted_tokens, groups):
    drop = copy.deepcopy(groups)
    if drop and drop[0].get("token_refs"):
        drop[0]["token_refs"] = drop[0]["token_refs"][1:]
    dup = copy.deepcopy(groups)
    if dup:
        dup.append(copy.deepcopy(dup[-1]))
    txt = copy.deepcopy(groups)
    if txt:
        txt[0]["text"] = (txt[0].get("text") or "") + "X"
    return {
        "drop_detected": not conservation_check(accepted_tokens, drop)["conserved"],
        "dup_detected": not conservation_check(accepted_tokens, dup)["conserved"],
        "text_detected": not conservation_check(accepted_tokens, txt)["conserved"],
    }


def classify_masked(prow):
    a = []
    for s, p in enumerate(prow):
        if float(p) >= TAU:
            a.append(s)
    if len(a) >= 2:
        return "OVERLAP"
    if len(a) == 0:
        return "NONE"
    return a[0]


def full_inside_range(lo, hi):
    if hi <= lo:
        return None, None
    low = (lo + FRAME - 1) // FRAME
    high = hi // FRAME - 1
    if high < low:
        return None, None
    return low, high


def normalize_capture(cap):
    cap = copy.deepcopy(cap)
    for ch in cap.get("chunk_ledger") or []:
        if ch.get("flush_end_wall") is None and ch.get("send_end") is not None:
            ch["flush_end_wall"] = ch["send_end"]
            ch["flush_clock"] = "send_end"
        elif ch.get("flush_end_wall") is not None:
            ch["flush_clock"] = "flush_end_wall"
        else:
            ch["flush_clock"] = "missing"
    return cap


def terminal_of(case, cap):
    spec = FREEZE["cases"][case]
    sess = cap.get("session") or {}
    if sess.get("seal_wall") is not None:
        return float(sess["seal_wall"]), "session.seal_wall"
    ent = (MANIFEST.get("cases") or {}).get(case) or {}
    if ent.get("terminal_relative_wall") is not None:
        return float(ent["terminal_relative_wall"]), "CAPTURE_MANIFEST.terminal_relative_wall"
    if sess.get("fin_wall") is not None:
        return float(sess["fin_wall"]), "session.fin_wall"
    if sess.get("finalize_wall") is not None:
        return float(sess["finalize_wall"]), "session.finalize_wall"
    pay = spec["payload_samples"]
    return (pay[1] - pay[0]) / float(HZ), "payload_duration_fallback"


def flush_frontier_at(cap, avail, pay):
    if avail is None:
        return None
    best = None
    for ch in cap.get("chunk_ledger") or []:
        fe = ch.get("flush_end_wall")
        r = ch.get("src_range", [0, 0])
        if fe is not None and fe <= avail:
            if best is None or r[1] > best:
                best = r[1]
    return best


def chained_prev_end_uncertainty(groups):
    n = 0
    notes = []
    for g in groups:
        note = g.get("start_note") or ""
        if note == "chained-prev-end":
            n += 1
            notes.append(g.get("idx"))
    return {
        "n_chained_prev_end": n,
        "n_groups": len(groups),
        "note": "start_src for chained-prev-end groups is previous end_src, not an independent provider start",
        "example_idxs": notes[:8],
    }


def assemble_units(groups, labels, case_id, arm):
    ordered = sorted(groups, key=lambda g: g.get("idx", 0))
    units = []
    if not ordered:
        return units
    def lab(g):
        return labels.get(g["idx"], {"relation": "UNKNOWN", "segment_id": "none"})
    run = [ordered[0]]
    cur = lab(ordered[0])
    for g in ordered[1:]:
        lg = lab(g)
        if lg.get("segment_id") == cur.get("segment_id") and lg.get("relation") == cur.get("relation"):
            run.append(g)
        else:
            rel = cur.get("relation", "UNKNOWN")
            units.append({
                "unit_id": f"{case_id}.{arm}.{len(units)}",
                "arm": arm,
                "relation": rel,
                "segment_id": cur.get("segment_id"),
                "text": "".join(x.get("text", "") for x in run),
                "group_idxs": [x.get("idx") for x in run],
            })
            run = [g]
            cur = lg
    rel = cur.get("relation", "UNKNOWN")
    units.append({
        "unit_id": f"{case_id}.{arm}.{len(units)}",
        "arm": arm,
        "relation": rel,
        "segment_id": cur.get("segment_id"),
        "text": "".join(x.get("text", "") for x in run),
        "group_idxs": [x.get("idx") for x in run],
    })
    return units


def unit_conservation(groups, units, final_text):
    ordered = sorted(groups, key=lambda g: g.get("idx", 0))
    all_idx = [g.get("idx") for g in ordered]
    flat = [gi for u in units for gi in u["group_idxs"]]
    concat = "".join(u["text"] for u in units)
    return {
        "concat_equal_final": concat == (final_text or ""),
        "groups_exact_once": sorted(flat) == sorted(all_idx),
        "n_units": len(units),
        "n_other_segments": len({u["segment_id"] for u in units if u.get("relation") == "OTHER"}),
    }


def synthetic_table(n):
    starts = [i * FRAME for i in range(n)]
    ends = [(i + 1) * FRAME for i in range(n)]
    frontiers = list(ends)
    chunk_of = [0] * n
    valid = [True] * n
    return {
        "n": n,
        "starts": starts,
        "ends": ends,
        "frontiers": frontiers,
        "chunk_of": chunk_of,
        "valid_tail": valid,
        "chunks": [{"index": 0, "emit_start_frame": 0, "emit_count": n, "raw_support_end_sample": n * FRAME, "service_us": 0}],
    }


def decode_events(table, probs, lo, hi, avail_by_chunk, clock):
    low, high = full_inside_range(lo, hi)
    events = []
    gaps = []
    last = None
    pending = None
    pend_start = None
    pend_n = 0
    prev_end = None
    seg_n = 0
    if low is None:
        return {"events": events, "gap_spans": gaps, "final_confirmed": last, "anchor_slot": None}
    n = table["n"]
    if low < 0:
        low = 0
    if high >= n:
        high = n - 1
    anchor_slot = None
    for i in range(low, high + 1):
        if i < 0 or i >= n or not table["valid_tail"][i]:
            if pending is not None:
                gaps.append({"kind": "ineligible-reset", "frame": i})
            pending, pend_n, prev_end = None, 0, None
            continue
        lab = classify_masked(probs[i])
        if lab in ("OVERLAP", "NONE"):
            if pending is not None:
                gaps.append({"kind": "overlap-none-reset", "frame": i, "lab": lab})
            pending, pend_n, prev_end = None, 0, None
            continue
        if last is None:
            last = lab
            pending, pend_n, prev_end = None, 0, None
            if anchor_slot is None:
                if i + 1 <= high and table["valid_tail"][i + 1] and classify_masked(probs[i + 1]) == lab:
                    anchor_slot = lab
            continue
        if lab == last:
            pending, pend_n, prev_end = None, 0, None
            continue
        s = i * FRAME
        e = (i + 1) * FRAME
        if prev_end is not None and s != prev_end:
            pending, pend_n = None, 0
            prev_end = None
        if pending is None or pending != lab:
            pending = lab
            pend_start = s
            pend_n = 0
            prev_end = s
        dur = e - s
        need = CONFIRMATION - pend_n
        if dur >= need:
            ci = table["chunk_of"][i]
            av = avail_by_chunk.get(ci) if isinstance(avail_by_chunk, dict) else None
            seg_n += 1
            relation = "CURRENT" if (anchor_slot is not None and lab == anchor_slot) else "OTHER"
            semantic = "CONTINUE_CURRENT" if relation == "CURRENT" else "SEPARATE_OTHER"
            events.append({
                "event_id": f"e.{seg_n}",
                "boundary": int(pend_start),
                "confirm_frame": int(i),
                "frontier": int(table["frontiers"][i]),
                "candidate_slot": int(lab),
                "relation": relation,
                "segment_id": f"{relation}-{seg_n}",
                "semantic": semantic,
                "uncertainty_samples": FRAME,
                "avail_record": av,
                "clock": clock,
            })
            last = lab
            pending, pend_n, prev_end = None, 0, None
            continue
        pend_n += dur
        prev_end = e
    return {"events": events, "gap_spans": gaps, "final_confirmed": last, "anchor_slot": anchor_slot}


def attach_avail(events, payload, terminal, clock_for_deadline):
    out = []
    for e in events:
        rec = e.get("avail_record") or {}
        src_s = rec.get("source_support_s")
        clip_s = None
        if rec.get("source_support_sample") is not None:
            clip_s = (rec["source_support_sample"] - payload[0]) / float(HZ)
        file_hi = rec.get("poll_hi_s")
        file_lo = rec.get("poll_lo_s")
        reconstructed = rec.get("reconstructed_finish_s")
        deadline_value = None
        timely = None
        too_late_reason = None
        if clock_for_deadline == "source_clip_relative":
            deadline_value = clip_s
            if clip_s is None:
                timely = None
                too_late_reason = "missing_source_support"
            elif terminal is None:
                timely = None
                too_late_reason = "missing_terminal"
            else:
                timely = clip_s <= terminal
                if not timely:
                    too_late_reason = "source_support_after_research_terminal"
        elif clock_for_deadline == "file_mode_poll":
            deadline_value = file_hi
            timely = None
            too_late_reason = "file_mode_poll_not_comparable_to_asr_seal_wall"
        e2 = dict(e)
        e2["source_support_s"] = src_s
        e2["clip_relative_source_s"] = clip_s
        e2["file_mode_poll_lo_s"] = file_lo
        e2["file_mode_poll_hi_s"] = file_hi
        e2["poll_uncertainty_s"] = None if (file_hi is None or file_lo is None) else (file_hi - file_lo)
        e2["reconstructed_finish_s"] = reconstructed
        e2["reconstructed_is_causal"] = False
        e2["deadline_clock"] = clock_for_deadline
        e2["deadline_value"] = deadline_value
        e2["research_timely"] = timely
        e2["too_late_reason"] = too_late_reason
        e2["causal_gpu_availability"] = False
        if clock_for_deadline == "source_clip_relative":
            e2["avail"] = clip_s
        else:
            e2["avail"] = None
        out.append(e2)
    return out


def r1_project(groups, cap, pay, terminal, requests_ordered):
    seals = []
    history = []
    seen = set()
    for req in requests_ordered:
        b = req["boundary"]
        sem = req["semantic"]
        avail = req.get("avail")
        eid = req["event_id"]
        last_Z = seals[-1]["sealed_frontier_Z"] if seals else pay[0]
        max_Z = max((s["sealed_frontier_Z"] for s in seals), default=None)
        applied_bs = [s["boundary"] for s in seals]
        if not (pay[0] <= b < pay[1]):
            history.append({"event_id": eid, "outcome": "invalid_scope", "applied_boundaries": list(applied_bs), "n_seals": len(seals), "reason": "boundary outside text object"})
            continue
        if eid in seen:
            if max_Z is not None and max_Z >= b:
                history.append({"event_id": eid, "outcome": "already_separated", "applied_boundaries": list(applied_bs), "n_seals": len(seals), "reason": "duplicate"})
            else:
                history.append({"event_id": eid, "outcome": "unsupported", "applied_boundaries": list(applied_bs), "n_seals": len(seals), "reason": "duplicate rev not cut"})
            continue
        if sem == "CONTINUE_CURRENT":
            if seals:
                history.append({"event_id": eid, "outcome": "unsupported_operation", "applied_boundaries": list(applied_bs), "n_seals": len(seals), "reason": "return-requires-past unsupported; future OTHER still legal"})
            else:
                history.append({"event_id": eid, "outcome": "noop-continue-no-cut", "applied_boundaries": [], "n_seals": 0, "reason": "continue current no cut needed"})
            seen.add(eid)
            continue
        if sem in ("UNRESOLVED", "NOOP"):
            history.append({"event_id": eid, "outcome": "unsupported", "applied_boundaries": list(applied_bs), "n_seals": len(seals), "reason": "overlap-unresolved-no-cut" if sem == "UNRESOLVED" else "noop-no-cut"})
            seen.add(eid)
            continue
        if avail is None or terminal is None:
            history.append({"event_id": eid, "outcome": "UNKNOWN", "applied_boundaries": list(applied_bs), "n_seals": len(seals), "reason": "missing availability or terminal"})
            seen.add(eid)
            continue
        if avail > terminal:
            history.append({"event_id": eid, "outcome": "tooLate", "applied_boundaries": list(applied_bs), "n_seals": len(seals), "availability": avail, "deadline": terminal, "reason": "late; no open content"})
            seen.add(eid)
            continue
        Z = flush_frontier_at(cap, avail, pay)
        if Z is None or Z < pay[0] or Z >= pay[1]:
            history.append({"event_id": eid, "outcome": "tooLate", "applied_boundaries": list(applied_bs), "n_seals": len(seals), "availability": avail, "deadline": terminal, "requestedX": b, "sealedZ": Z, "reason": "sealed no open range; Z outside object"})
            seen.add(eid)
            continue
        covering = [s for s in seals if s["sealed_frontier_Z"] >= b]
        if covering:
            history.append({"event_id": eid, "outcome": "already_separated", "applied_boundaries": list(applied_bs), "n_seals": len(seals), "requestedX": b, "reason": "existing seal >= estimated transition"})
            seen.add(eid)
            continue
        if not (last_Z < b < Z):
            history.append({"event_id": eid, "outcome": "tooLate", "applied_boundaries": list(applied_bs), "n_seals": len(seals), "requestedX": b, "sealedZ": Z, "last_Z": last_Z, "reason": "transition outside current open range or no flush progress"})
            seen.add(eid)
            continue
        seg_id = len(seals) + 1
        seals.append({"boundary": b, "sealed_frontier_Z": Z, "availability": avail, "event_id": eid, "segment": f"OTHER-{seg_id}"})
        history.append({"event_id": eid, "outcome": "applied", "applied_boundaries": [s["boundary"] for s in seals], "n_seals": len(seals), "availability": avail, "deadline": terminal, "requestedX": b, "sealedZ": Z, "segment": f"OTHER-{seg_id}", "reason": "C13 prospective seal at accepted frontier Z; no X..Z repair claim"})
        seen.add(eid)
    labels = {}
    if not seals:
        for g in groups:
            labels[g["idx"]] = {"relation": "CURRENT", "segment_id": "CURRENT-0"}
    else:
        ordered = sorted(seals, key=lambda s: s["sealed_frontier_Z"])
        bounds_Z = [s["sealed_frontier_Z"] for s in ordered]
        for g in groups:
            es = g.get("end_src")
            if es is None:
                labels[g["idx"]] = {"relation": "UNKNOWN", "segment_id": "unknown"}
            elif es <= bounds_Z[0]:
                labels[g["idx"]] = {"relation": "CURRENT", "segment_id": "CURRENT-0"}
            else:
                k = 0
                while k + 1 < len(bounds_Z) and es > bounds_Z[k + 1]:
                    k += 1
                labels[g["idx"]] = {"relation": "OTHER", "segment_id": ordered[k]["segment"]}
    return {"seals": seals, "history": history, "labels": labels, "n_seals": len(seals)}


def r2_partition(groups, obj_span, terminal, events_ordered):
    applic = [e for e in events_ordered if obj_span[0] <= e["boundary"] < obj_span[1] and e.get("avail") is not None and terminal is not None and e["avail"] <= terminal]
    applic = sorted(applic, key=lambda e: (e["boundary"], e.get("confirm_frame", 0)))
    labels = {}
    for g in groups:
        s = g.get("start_src")
        e = g.get("end_src")
        if s is None or e is None:
            labels[g["idx"]] = {"relation": "UNKNOWN", "segment_id": "unknown"}
            continue
        straddle = any((s < b["boundary"] < e) for b in applic)
        if straddle:
            labels[g["idx"]] = {"relation": "UNKNOWN", "segment_id": "straddle"}
            continue
        cur_rel = "CURRENT"
        cur_seg = "CURRENT-0"
        seg_n = 0
        for ev in applic:
            if ev["boundary"] < e:
                if ev.get("semantic") == "SEPARATE_OTHER" or ev.get("relation") == "OTHER":
                    seg_n += 1
                    cur_rel = "OTHER"
                    cur_seg = f"OTHER-{seg_n}"
                elif ev.get("semantic") == "CONTINUE_CURRENT" or ev.get("relation") == "CURRENT":
                    cur_rel = "CURRENT"
                    cur_seg = "CURRENT-return"
                elif ev.get("semantic") in ("UNRESOLVED", "NOOP"):
                    cur_rel = "UNKNOWN"
                    cur_seg = "unresolved"
            else:
                break
        labels[g["idx"]] = {"relation": cur_rel, "segment_id": cur_seg}
    return {"labels": labels, "applicable": applic}


def r0_labels(groups):
    return {g["idx"]: {"relation": "CURRENT", "segment_id": "CURRENT-0"} for g in groups}


def make_groups_from_spans(spans, text_prefix="w"):
    groups = []
    acc = []
    for i, (s, e, txt) in enumerate(spans):
        tok = {"o": i, "text": txt}
        acc.append(tok)
        groups.append({"idx": i, "text": txt, "word": txt.strip(), "start_src": s, "end_src": e, "token_refs": [{"o": i, "slice": txt}]})
    return groups, acc


def synthetic_probs(rows):
    return rows


def run_synthetics():
    results = {}
    n = 30
    table = synthetic_table(n)
    lo, hi = 0, n * FRAME
    avail = {0: {"source_support_s": 0.0, "source_support_sample": hi, "poll_lo_s": 0.0, "poll_hi_s": 0.0, "reconstructed_finish_s": 0.0}}
    pay = [0, hi]
    terminal = 10.0
    cap = {"chunk_ledger": [{"src_range": [0, hi], "flush_end_wall": 1.0}]}

    rows = [[1, 0, 0, 0]] * 10 + [[0, 1, 0, 0]] * 10 + [[0, 0, 1, 0]] * 10
    dec = decode_events(table, rows, lo, hi, avail, "synthetic")
    ev = attach_avail(dec["events"], pay, terminal, "source_clip_relative")
    spans = [(0, 10 * FRAME, "A "), (10 * FRAME, 20 * FRAME, "B "), (20 * FRAME, 30 * FRAME, "C")]
    groups, acc = make_groups_from_spans(spans)
    r2 = r2_partition(groups, pay, terminal, ev)
    units = assemble_units(groups, r2["labels"], "SYN_ABC", "R2")
    other_ids = [u["segment_id"] for u in units if u["relation"] == "OTHER"]
    results["A_to_B_to_C"] = {
        "n_events": len(ev),
        "n_units": len(units),
        "other_segment_ids": other_ids,
        "preserved_other_other": len(set(other_ids)) >= 2,
        "relations": [u["relation"] for u in units],
        "pass": len(ev) >= 2 and len(set(other_ids)) >= 2,
    }

    rows = [[1, 0, 0, 0]] * 10 + [[0, 1, 0, 0]] * 1 + [[1, 0, 0, 0]] * 10
    table2 = synthetic_table(21)
    dec2 = decode_events(table2, rows, 0, 21 * FRAME, avail, "synthetic")
    results["A_shortB_A"] = {
        "n_events": len(dec2["events"]),
        "final_confirmed": dec2["final_confirmed"],
        "pass": len(dec2["events"]) == 0 and dec2["final_confirmed"] == 0,
    }

    rows = [[1, 0, 0, 0]] * 10 + [[1, 1, 0, 0]] * 10 + [[1, 0, 0, 0]] * 10
    dec3 = decode_events(table, rows, lo, hi, avail, "synthetic")
    ev3 = attach_avail(dec3["events"], pay, terminal, "source_clip_relative")
    for e in ev3:
        if e["semantic"] != "SEPARATE_OTHER":
            e["semantic"] = e.get("semantic") or "NOOP"
    overlap_ev = list(ev3)
    overlap_ev.insert(0, {
        "event_id": "ov.0",
        "boundary": 10 * FRAME,
        "semantic": "UNRESOLVED",
        "avail": 0.5,
        "relation": "UNKNOWN",
        "segment_id": "unresolved",
        "confirm_frame": 10,
    })
    groups3, _ = make_groups_from_spans([(0, 10 * FRAME, "A "), (10 * FRAME, 20 * FRAME, "AB "), (20 * FRAME, 30 * FRAME, "A2")])
    r1_3 = r1_project(groups3, cap, pay, terminal, overlap_ev)
    r2_3 = r2_partition(groups3, pay, terminal, overlap_ev)
    results["A_overlap_A"] = {
        "n_native_events": len(dec3["events"]),
        "r1_n_seals": r1_3["n_seals"],
        "r1_outcomes": [h["outcome"] for h in r1_3["history"]],
        "r2_mid_relation": r2_3["labels"][1]["relation"],
        "pass": r1_3["n_seals"] == 0 and r2_3["labels"][1]["relation"] == "UNKNOWN",
    }

    rows = [[1, 0, 0, 0]] * 10 + [[1, 1, 0, 0]] * 10 + [[0, 1, 0, 0]] * 10
    dec4 = decode_events(table, rows, lo, hi, avail, "synthetic")
    ev4 = attach_avail(dec4["events"], pay, terminal, "source_clip_relative")
    groups4, _ = make_groups_from_spans([(0, 10 * FRAME, "A "), (10 * FRAME, 20 * FRAME, "AB "), (20 * FRAME, 30 * FRAME, "B")])
    r1_4 = r1_project(groups4, cap, pay, terminal, ev4)
    r2_4 = r2_partition(groups4, pay, terminal, ev4)
    units4 = assemble_units(groups4, r2_4["labels"], "SYN_AB", "R2")
    results["A_overlap_B"] = {
        "n_events": len(ev4),
        "r1_n_seals": r1_4["n_seals"],
        "r2_relations": [r2_4["labels"][i]["relation"] for i in range(3)],
        "r2_segment_ids": [r2_4["labels"][i]["segment_id"] for i in range(3)],
        "n_units": len(units4),
        "pass": len(ev4) >= 1 and r2_4["labels"][2]["relation"] == "OTHER",
    }
    old_merge = assemble_units(
        groups,
        {0: {"relation": "CURRENT", "segment_id": "CURRENT-0"}, 1: {"relation": "OTHER", "segment_id": "OTHER-merged"}, 2: {"relation": "OTHER", "segment_id": "OTHER-merged"}},
        "SYN_MERGE",
        "OLD",
    )
    new_keep = assemble_units(
        groups,
        {0: {"relation": "CURRENT", "segment_id": "CURRENT-0"}, 1: {"relation": "OTHER", "segment_id": "OTHER-1"}, 2: {"relation": "OTHER", "segment_id": "OTHER-2"}},
        "SYN_KEEP",
        "NEW",
    )
    results["other_other_not_merged"] = {
        "old_n_units": len(old_merge),
        "new_n_units": len(new_keep),
        "new_ids": [u["segment_id"] for u in new_keep],
        "pass": len(old_merge) == 2 and len(new_keep) == 3,
    }
    n_pass = sum(1 for k, v in results.items() if v.get("pass"))
    return {"scenarios": results, "n_pass": n_pass, "n_total": len(results)}


def load_gt_words(meet, role):
    path = STAGE2 / "annotations" / "words" / f"{meet}.{role}.words.xml"
    if not path.exists():
        return []
    root = ET.fromstring(path.read_bytes())
    out = []
    for w in root.iter():
        if w.tag.split("}")[-1] != "w":
            continue
        a = w.attrib
        if a.get("punc", "") == "true":
            continue
        txt = (w.text or "").strip()
        if not txt:
            continue
        try:
            s = float(a.get("starttime", -1))
            e = float(a.get("endtime", -1))
        except ValueError:
            continue
        if s < 0 or e < 0:
            continue
        wid = a.get(NS + "id") or a.get("nite:id") or ""
        out.append({"id": wid, "role": role, "start": s, "end": e, "start_src": int(round(s * HZ)), "end_src": int(round(e * HZ)), "text": txt})
    return out


def watcher_smoke():
    d = EXP / "_tmp_watch"
    d.mkdir(exist_ok=True)
    p = d / "diar.trace.json"
    if p.exists():
        p.unlink()
    polls = []
    t0 = time.perf_counter()
    def snap():
        try:
            sz = p.stat().st_size
        except OSError:
            sz = 0
        polls.append({"t": time.perf_counter() - t0, "size": sz})
    snap()
    with open(p, "w", encoding="utf-8") as f:
        f.write('{"feat_len":8,"chunks":[\n')
        f.flush()
        os.fsync(f.fileno())
        snap()
        f.write('{"chunk":0,"raw_support_end_sample":1280,"service_us":100}')
        f.flush()
        os.fsync(f.fileno())
        snap()
        time.sleep(0.01)
        f.write(',\n{"chunk":1,"raw_support_end_sample":2560,"service_us":100}')
        f.flush()
        os.fsync(f.fileno())
        snap()
        f.write('],"total_n":2}')
        f.flush()
        os.fsync(f.fileno())
        snap()
    text = p.read_text(encoding="utf-8")
    assigned = assign_polls_to_chunks(text, polls)
    ok = len(assigned) >= 2 and assigned[1]["poll_hi_s"] >= assigned[0]["poll_hi_s"]
    try:
        p.unlink()
        d.rmdir()
    except OSError:
        pass
    return {"pass": ok, "n_assigned": len(assigned), "assigned": assigned, "n_polls": len(polls)}


def assign_polls_to_chunks(text, polls):
    offs = []
    start = 0
    while True:
        i = text.find('{"chunk":', start)
        if i < 0:
            break
        j = text.find("}", i)
        if j < 0:
            break
        offs.append(j + 1)
        start = j + 1
    out = []
    for ci, end_off in enumerate(offs):
        hi = None
        lo = 0.0
        prev = 0.0
        for rec in polls:
            if rec["size"] >= end_off:
                hi = rec["t"]
                lo = prev
                break
            prev = rec["t"]
        if hi is None and polls:
            hi = polls[-1]["t"]
            lo = polls[-1]["t"]
        out.append({"chunk": ci, "end_offset": end_off, "poll_lo_s": lo, "poll_hi_s": hi, "poll_uncertainty_s": None if hi is None else (hi - lo)})
    return out


def lifetimes_pack(case, cap, events, r1, source_id):
    sess = cap.get("session") or {}
    return {
        "physical_asr_session": {
            "case": case,
            "provider_model": sess.get("model") or sess.get("model_requested"),
            "seal_wall": sess.get("seal_wall"),
            "finalize_wall": sess.get("finalize_wall"),
            "fin_wall": sess.get("fin_wall"),
            "payload": (cap.get("audio") or {}).get("payload_samples"),
        },
        "logical_ownership_segment": {
            "r1_n_seals": r1.get("n_seals"),
            "r1_segments": [s.get("segment") for s in r1.get("seals") or []],
        },
        "psem_state_reference": {
            "source_id": source_id,
            "continuous_prefix": source_id in FREEZE["native_runs"]["sources"],
            "asr_rollover_does_not_reset_psem": True,
        },
    }




def score_against_gt(meet, anchor_role, groups, labels, span):
    if not meet or not anchor_role:
        return {"available": False, "reason": "no_anchor_or_meet"}
    words = {}
    for role in ("A", "B", "C", "D"):
        words[role] = load_gt_words(meet, role)
    if not any(words.values()):
        return {"available": False, "reason": "no_gt_xml"}
    lo, hi = span
    correct, wrong, unresolved, mixed = [], [], [], []
    for g in groups:
        s = g.get("start_src")
        e = g.get("end_src")
        if s is None or e is None:
            continue
        if e < lo or s > hi:
            continue
        present = set()
        for role, lst in words.items():
            for w in lst:
                if w["end_src"] > s and w["start_src"] < e:
                    present.add(role)
        lab = labels.get(g["idx"], {})
        rel = lab.get("relation")
        if rel in ("UNKNOWN", None):
            unresolved.append({"idx": g["idx"], "gt": sorted(present), "rel": rel})
            continue
        if len(present) >= 2:
            mixed.append({"idx": g["idx"], "gt": sorted(present), "rel": rel})
            continue
        if len(present) == 0:
            unresolved.append({"idx": g["idx"], "gt": [], "rel": rel})
            continue
        role = next(iter(present))
        want = "CURRENT" if role == anchor_role else "OTHER"
        rec = {"idx": g["idx"], "gt_role": role, "want": want, "rel": rel, "word": g.get("word")}
        if rel == want:
            correct.append(rec)
        else:
            wrong.append(rec)
    return {
        "available": True,
        "n_correct": len(correct),
        "n_wrong": len(wrong),
        "n_unresolved": len(unresolved),
        "n_mixed": len(mixed),
        "correct": correct,
        "wrong": wrong,
        "unresolved": unresolved,
        "mixed": mixed,
        "denominator": len(correct) + len(wrong) + len(unresolved) + len(mixed),
    }


def build_table_from_trace(trace, prefix_end):
    chunks = trace.get("chunks") or []
    n = int(trace.get("used_n") or trace.get("total_n") or 0)
    frontiers = [0] * n
    chunk_of = [None] * n
    valid_tail = [False] * n
    fi = 0
    for ci, c in enumerate(chunks):
        rs = c.get("raw_support_end_sample")
        ec = c.get("emit_count") or 0
        ok = rs is not None and rs <= prefix_end
        for _ in range(ec):
            if fi >= n:
                break
            frontiers[fi] = rs
            chunk_of[fi] = ci
            valid_tail[fi] = ok
            fi += 1
    return {"n": n, "frontiers": frontiers, "chunk_of": chunk_of, "valid_tail": valid_tail, "chunks": chunks, "prefix": prefix_end}


def load_probs_f32(path, n, nspk=4):
    import numpy as np
    raw = np.fromfile(str(path), dtype=np.float32)
    return raw.reshape(n, nspk)


def gt_control_events(meet, anchor_role, payload, table, avail_by_chunk, terminal):
    if not meet or not anchor_role:
        return []
    words = []
    for role in ("A", "B", "C", "D"):
        words.extend(load_gt_words(meet, role))
    words.sort(key=lambda w: (w["start_src"], w["end_src"]))
    events = []
    last_role = None
    seg_n = 0
    for w in words:
        if w["end_src"] <= payload[0] or w["start_src"] >= payload[1]:
            continue
        if last_role is None:
            last_role = w["role"]
            continue
        if w["role"] == last_role:
            continue
        b = w["start_src"]
        fi = min(table["n"] - 1, max(0, b // FRAME))
        ci = table["chunk_of"][fi] if 0 <= fi < table["n"] else None
        rec = avail_by_chunk.get(ci) if ci is not None else None
        seg_n += 1
        relation = "CURRENT" if w["role"] == anchor_role else "OTHER"
        semantic = "CONTINUE_CURRENT" if relation == "CURRENT" else "SEPARATE_OTHER"
        ev = {
            "event_id": f"gt.{seg_n}",
            "boundary": int(b),
            "confirm_frame": int(fi),
            "frontier": int(table["frontiers"][fi]) if 0 <= fi < table["n"] else None,
            "relation": relation,
            "segment_id": f"{relation}-gt-{seg_n}",
            "semantic": semantic,
            "avail_record": rec,
            "clock": "charged_control",
            "oracle": True,
        }
        events.append(ev)
        last_role = w["role"]
    return attach_avail(events, payload, terminal, "source_clip_relative")


def classify_target_transition(native_final_slots, events, r1, r2_labels, groups, payload):
    classes = []
    n_sep = sum(1 for e in events if e.get("semantic") == "SEPARATE_OTHER")
    if not native_final_slots and n_sep == 0:
        classes.append({"type": "not_present_in_native_final", "detail": "no confirmed slot change in full-inside decode"})
        return classes
    if events and all(e.get("research_timely") is False for e in events if e.get("semantic") == "SEPARATE_OTHER"):
        classes.append({"type": "present_but_available_too_late", "detail": "source-clip research clock only; not causal GPU", "causal": False})
    n_sep = sum(1 for e in events if e.get("semantic") == "SEPARATE_OTHER")
    if native_final_slots and n_sep < len(native_final_slots):
        classes.append({"type": "lost_changed_in_event_conversion", "detail": f"native_slot_changes={len(native_final_slots)} events={n_sep}"})
    other_ids = {lab.get("segment_id") for lab in r2_labels.values() if lab.get("relation") == "OTHER"}
    if len(native_final_slots) >= 2 and len(other_ids) < 2:
        classes.append({"type": "lost_changed_in_ownership_assembly", "detail": "OTHER-OTHER collapsed"})
    n_unknown = sum(1 for g in groups if (r2_labels.get(g["idx"]) or {}).get("relation") == "UNKNOWN")
    if n_unknown:
        classes.append({"type": "invalid_unknown_by_contract", "detail": f"n_unknown={n_unknown}"})
    if not classes:
        classes.append({"type": None, "detail": "no_classified_failure"})
    return classes


def native_slot_changes(table, probs, lo, hi):
    low, high = full_inside_range(lo, hi)
    if low is None:
        return []
    last = None
    changes = []
    n = table["n"]
    high = min(high, n - 1)
    low = max(low, 0)
    for i in range(low, high + 1):
        if not table["valid_tail"][i]:
            last = None
            continue
        lab = classify_masked(probs[i])
        if lab == "NONE":
            continue
        if lab == "OVERLAP":
            last = None
            continue
        if last is None:
            last = lab
            continue
        if lab != last:
            changes.append({"frame": i, "from": last, "to": lab, "boundary": i * FRAME})
            last = lab
    return changes


def run_native_one(source_id, force=False):
    pin = FREEZE["prefixes"][source_id]
    wav = Path(pin["wav"])
    want = pin["sha256"]
    got = sha256_file(wav)
    if got != want:
        return {"ok": False, "error": f"prefix sha mismatch {source_id} {got} {want}"}
    exe = Path(FREEZE["profile"]["exe"])
    model = Path(FREEZE["profile"]["model"])
    if sha256_file(exe) != FREEZE["profile"]["exe_sha256"]:
        return {"ok": False, "error": "exe sha mismatch"}
    if sha256_file(model) != FREEZE["profile"]["model_sha256"]:
        return {"ok": False, "error": "model sha mismatch"}
    dump = Path(FREEZE["native_runs"]["dump_root"]) / source_id
    dump.mkdir(parents=True, exist_ok=True)
    owned = EXP / "native" / source_id
    owned.mkdir(parents=True, exist_ok=True)
    meta_path = owned / "meta.json"
    if meta_path.exists() and not force:
        prev = load_json(meta_path)
        if prev.get("ok") and (dump / "diar.trace.json").exists():
            return prev
    for name in ("diar.trace.json", "diar.probs.f32", "diar.hidden.f32", "diar.logits.f32"):
        p = dump / name
        if p.exists():
            p.unlink()
    env = os.environ.copy()
    for k, v in FREEZE["profile"]["env"].items():
        env[k] = v
    env["TRANSCRIBE_DUMP_DIR"] = str(dump)
    polls = []
    t0 = time.perf_counter()
    utc0 = utc_now()
    interval = float(FREEZE["native_runs"]["poll_interval_s"])
    cmd = [str(exe), "-m", str(model), "--backend", "vulkan", str(wav)]
    stdout_p = dump / "stdout.txt"
    stderr_p = dump / "stderr.txt"
    so = open(stdout_p, "w", encoding="utf-8", errors="replace")
    se = open(stderr_p, "w", encoding="utf-8", errors="replace")
    try:
        proc = subprocess.Popen(cmd, env=env, stdout=so, stderr=se, cwd=str(dump))
        trace_p = dump / "diar.trace.json"
        while True:
            try:
                sz = trace_p.stat().st_size
            except OSError:
                sz = 0
            polls.append({"t": time.perf_counter() - t0, "size": sz})
            if proc.poll() is not None:
                try:
                    sz = trace_p.stat().st_size
                except OSError:
                    sz = 0
                polls.append({"t": time.perf_counter() - t0, "size": sz})
                break
            time.sleep(interval)
        rc = proc.returncode
    finally:
        so.close()
        se.close()
    wall = time.perf_counter() - t0
    if rc != 0 or not (dump / "diar.trace.json").exists():
        meta = {"ok": False, "error": f"cli rc={rc}", "wall_s": wall, "utc0": utc0, "cmd": cmd, "n_polls": len(polls)}
        write_json(meta_path, meta)
        return meta
    text = (dump / "diar.trace.json").read_text(encoding="utf-8")
    assigned = assign_polls_to_chunks(text, polls)
    trace = json.loads(text)
    chunks = trace.get("chunks") or []
    for a in assigned:
        ci = a["chunk"]
        if 0 <= ci < len(chunks):
            c = chunks[ci]
            a["raw_support_end_sample"] = c.get("raw_support_end_sample")
            a["source_support_s"] = None if c.get("raw_support_end_sample") is None else c["raw_support_end_sample"] / float(HZ)
            a["service_us"] = c.get("service_us")
            a["emit_start_frame"] = c.get("emit_start_frame")
            a["emit_count"] = c.get("emit_count")
    owned_trace = owned / "diar.trace.json"
    owned_trace.write_text(text, encoding="utf-8")
    write_json(owned / "polls.json", {"polls": polls, "assigned": assigned, "poll_interval_s": interval})
    first_hi = assigned[0]["poll_hi_s"] if assigned else None
    src_first = assigned[0].get("source_support_s") if assigned else None
    faster = None
    if first_hi is not None and src_first is not None:
        faster = first_hi < src_first
    n_ahead = 0
    n_behind = 0
    for a in assigned:
        ss = a.get("source_support_s")
        hi = a.get("poll_hi_s")
        if ss is None or hi is None:
            continue
        if hi < ss:
            n_ahead += 1
        else:
            n_behind += 1
    meta = {
        "ok": True,
        "source_id": source_id,
        "utc0": utc0,
        "wall_s": wall,
        "rc": rc,
        "cmd": cmd,
        "dump": str(dump),
        "exe_sha256": FREEZE["profile"]["exe_sha256"],
        "model_sha256": FREEZE["profile"]["model_sha256"],
        "wav_sha256": got,
        "trace_sha256": sha256_file(dump / "diar.trace.json"),
        "probs_sha256": sha256_file(dump / "diar.probs.f32") if (dump / "diar.probs.f32").exists() else None,
        "n_chunks": len(chunks),
        "n_polls": len(polls),
        "n_assigned": len(assigned),
        "time_to_first_chunk_poll_hi_s": first_hi,
        "first_chunk_source_support_s": src_first,
        "file_mode_faster_than_first_source_support": faster,
        "n_chunks_poll_hi_before_source_support": n_ahead,
        "n_chunks_poll_hi_after_source_support": n_behind,
        "initialization_us": trace.get("initialization_us"),
        "mel_full_diagnostic_us": trace.get("mel_full_diagnostic_us"),
        "causal_frontend": trace.get("causal_frontend"),
        "used_n": trace.get("used_n"),
        "clocks_distinct": True,
        "causal_gpu_availability": False,
        "note": "poll bounds are file-mode dump visibility, not live capture-tied availability",
    }
    write_json(meta_path, meta)
    return meta


def avail_map_from_assigned(assigned, chunks, init_s):
    out = {}
    prev = None
    for a in assigned:
        ci = a["chunk"]
        rec = {
            "poll_lo_s": a.get("poll_lo_s"),
            "poll_hi_s": a.get("poll_hi_s"),
            "source_support_sample": a.get("raw_support_end_sample"),
            "source_support_s": a.get("source_support_s"),
            "service_us": a.get("service_us"),
            "reconstructed_finish_s": None,
        }
        if a.get("service_us") is not None:
            if prev is None:
                prev = (init_s or 0.0) + a["service_us"] / 1e6
            else:
                prev = prev + a["service_us"] / 1e6
            rec["reconstructed_finish_s"] = prev
        out[ci] = rec
    if not assigned:
        prev = init_s or 0.0
        for ci, c in enumerate(chunks):
            prev = prev + (c.get("service_us") or 0) / 1e6
            out[ci] = {
                "poll_lo_s": None,
                "poll_hi_s": None,
                "source_support_sample": c.get("raw_support_end_sample"),
                "source_support_s": None if c.get("raw_support_end_sample") is None else c["raw_support_end_sample"] / float(HZ),
                "service_us": c.get("service_us"),
                "reconstructed_finish_s": prev,
            }
    return out


def load_source_bundle(source_id, native_meta):
    prefix = None
    if source_id in FREEZE["prefixes"]:
        prefix = FREEZE["prefixes"][source_id]["prefix_end_sample"]
    obs = load_json(OBSDIR / "OBSERVATIONS.json")
    src_entry = next(s for s in obs["sources"] if s["source_id"] == source_id)
    if prefix is None:
        prefix = src_entry["prefix_end_sample"]
    dump = Path(FREEZE["native_runs"]["dump_root"]) / source_id
    owned = EXP / "native" / source_id
    if native_meta is None and (owned / "meta.json").exists():
        native_meta = load_json(owned / "meta.json")
    instrumented = bool(native_meta and native_meta.get("ok") and (dump / "diar.trace.json").exists() and (owned / "polls.json").exists())
    if instrumented:
        trace = load_json(dump / "diar.trace.json")
        table = build_table_from_trace(trace, prefix)
        probs = load_probs_f32(dump / "diar.probs.f32", table["n"])
        polls = load_json(owned / "polls.json")
        assigned = polls["assigned"]
        init_s = ((trace.get("load_us") or 0) + (trace.get("sched_setup_us") or 0)) / 1e6
        avail = avail_map_from_assigned(assigned, table["chunks"], init_s)
        return {
            "source_id": source_id,
            "table": table,
            "probs": probs,
            "avail": avail,
            "instrumented": True,
            "init_s": init_s,
            "trace": trace,
            "src_entry": src_entry,
        }
    trace = load_json(ROOT / src_entry["trace_path"])
    table = build_table_from_trace(trace, prefix)
    feat = src_entry["feature_files"]["probs"]
    probs = load_probs_f32(ROOT / feat["path"], feat["shape"][0], feat["shape"][1])
    init_s = ((trace.get("load_us") or 0) + (trace.get("sched_setup_us") or 0)) / 1e6
    avail = avail_map_from_assigned([], table["chunks"], init_s)
    return {
        "source_id": source_id,
        "table": table,
        "probs": probs,
        "avail": avail,
        "instrumented": False,
        "init_s": init_s,
        "trace": trace,
        "src_entry": src_entry,
    }


def run_case(case, bundles):
    spec = FREEZE["cases"][case]
    cap_path = Path(spec["capture"])
    if not cap_path.is_absolute():
        cap_path = ROOT / spec["capture"]
    cap_raw = load_json(cap_path)
    cap = normalize_capture(cap_raw)
    groups = corrected_groups_for_capture(cap)
    acc = (cap.get("accepted") or {}).get("tokens") or []
    final_text = (cap.get("accepted") or {}).get("final_text") or ""
    pay = spec["payload_samples"]
    terminal, terminal_src = terminal_of(case, cap)
    bundle = bundles[spec["source"]]
    table = bundle["table"]
    probs = bundle["probs"]
    native_changes = native_slot_changes(table, probs, pay[0], pay[1])
    dec = decode_events(table, probs, pay[0], pay[1], bundle["avail"], "native")
    events = attach_avail(dec["events"], pay, terminal, "source_clip_relative")
    file_mode_events = events
    paced_path = EXP / "paced" / case / "arrivals.json"
    paced = load_json(paced_path) if paced_path.exists() else None
    if paced and paced.get("ok") and paced.get("events") is not None:
        events = paced["events"]
    r0l = r0_labels(groups)
    r1 = r1_project(groups, cap, pay, terminal, events)
    r2 = r2_partition(groups, pay, terminal, events)
    u0 = assemble_units(groups, r0l, case, "R0")
    u1 = assemble_units(groups, r1["labels"], case, "R1")
    u2 = assemble_units(groups, r2["labels"], case, "R2")
    chained = chained_prev_end_uncertainty(cap_raw.get("groups") or groups)
    cons = conservation_check(acc, groups)
    ctrls = throwaway_controls(acc, groups)
    uc0 = unit_conservation(groups, u0, final_text)
    uc1 = unit_conservation(groups, u1, final_text)
    uc2 = unit_conservation(groups, u2, final_text)
    span = spec.get("scored_span_samples") or pay
    sc0 = score_against_gt(spec.get("meet"), spec.get("anchor_role"), groups, r0l, span)
    sc1 = score_against_gt(spec.get("meet"), spec.get("anchor_role"), groups, r1["labels"], span)
    sc2 = score_against_gt(spec.get("meet"), spec.get("anchor_role"), groups, r2["labels"], span)
    control_ev = gt_control_events(spec.get("meet"), spec.get("anchor_role"), pay, table, bundle["avail"], terminal)
    r1c = r1_project(groups, cap, pay, terminal, control_ev)
    r2c = r2_partition(groups, pay, terminal, control_ev)
    sc1c = score_against_gt(spec.get("meet"), spec.get("anchor_role"), groups, r1c["labels"], span)
    sc2c = score_against_gt(spec.get("meet"), spec.get("anchor_role"), groups, r2c["labels"], span)
    fails = classify_target_transition(native_changes, events, r1, r2["labels"], groups, pay)
    n_unknown1 = sum(1 for g in groups if r1["labels"][g["idx"]]["relation"] == "UNKNOWN")
    n_unknown2 = sum(1 for g in groups if r2["labels"][g["idx"]]["relation"] == "UNKNOWN")
    credited = {
        "r1_causal_credited": False,
        "r2_causal_credited": False,
        "reason": "no causal GPU availability on Audio/ASR clock; research scores are not credited improvements",
    }
    if paced and paced.get("ok"):
        credited = {
            "r1_causal_credited": False,
            "r2_causal_credited": False,
            "reason": "paced QPC receipt is experiment-A causal vs frozen ASR terminal; not a production PCM seal and not C5",
        }
    return {
        "case": case,
        "kind": spec["kind"],
        "source": spec["source"],
        "independent_episode": spec.get("independent_episode", True),
        "instrumented_native": bundle["instrumented"],
        "paced_ok": bool(paced and paced.get("ok")),
        "paced_n_chunks": (paced or {}).get("n_paced_chunks"),
        "paced_excluded_startup": (paced or {}).get("excluded_startup_scope"),
        "c5_legal_production_scope": False,
        "payload": pay,
        "terminal": terminal,
        "terminal_source": terminal_src,
        "lifetimes": lifetimes_pack(case, cap, events, r1, spec["source"]),
        "chained_prev_end": chained,
        "conservation": cons,
        "conservation_controls": ctrls,
        "n_groups": len(groups),
        "final_text_len": len(final_text),
        "native_slot_changes": native_changes,
        "n_native_slot_changes": len(native_changes),
        "events": events,
        "n_events": len(events),
        "gap_spans": dec["gap_spans"],
        "anchor_slot": dec["anchor_slot"],
        "r0": {"n_units": len(u0), "units": u0, "unit_conservation": uc0, "score": sc0},
        "r1": {
            "n_seals": r1["n_seals"],
            "history": r1["history"],
            "seals": r1["seals"],
            "n_units": len(u1),
            "units": u1,
            "unit_conservation": uc1,
            "n_unknown": n_unknown1,
            "score": sc1,
            "n_other_segments": uc1["n_other_segments"],
        },
        "r2": {
            "n_units": len(u2),
            "units": u2,
            "unit_conservation": uc2,
            "n_unknown": n_unknown2,
            "score": sc2,
            "n_other_segments": uc2["n_other_segments"],
            "research_only": True,
        },
        "control_charged": {
            "n_events": len(control_ev),
            "events": control_ev,
            "r1_n_seals": r1c["n_seals"],
            "r1_score": sc1c,
            "r2_score": sc2c,
            "zero_delay_hindsight": False,
        },
        "failure_classes": fails,
        "credited": credited,
        "fragmentation": {"r0": len(u0), "r1": len(u1), "r2": len(u2)},
    }


def smoke():
    checks = []
    def ck(name, ok, detail=""):
        checks.append({"name": name, "pass": bool(ok), "detail": str(detail)[:500]})
    syn = run_synthetics()
    for k, v in syn["scenarios"].items():
        ck("synthetic-" + k, v.get("pass"), json.dumps({x: v[x] for x in v if x != "pass"}))
    w = watcher_smoke()
    ck("watcher-fake-append", w["pass"], json.dumps({k: w[k] for k in w if k != "assigned"}))
    cap = load_json(ROOT / FREEZE["inputs"]["np_captures"]["NP1"]["path"])
    groups = corrected_groups_for_capture(cap)
    acc = cap["accepted"]["tokens"]
    cons = conservation_check(acc, groups)
    ck("np1-conservation", cons["conserved"], cons)
    ctrls = throwaway_controls(acc, groups)
    ck("drop-dup-text-controls", ctrls["drop_detected"] and ctrls["dup_detected"] and ctrls["text_detected"], ctrls)
    ck("failure-types-five", FAILURE_TYPES == ["not_present_in_native_final", "present_but_available_too_late", "lost_changed_in_event_conversion", "lost_changed_in_ownership_assembly", "invalid_unknown_by_contract"], FAILURE_TYPES)
    ck("c5-not-auto", FREEZE["c5_applicability"]["psem_cannot_extend_c5"] and all(not FREEZE["cases"][c].get("c5_legal") for c in FREEZE["cases"]), "7s clips not C5")
    ck("three-lifetimes", len(FREEZE["lifetimes"]) == 3, FREEZE["lifetimes"])
    groups3, _ = make_groups_from_spans([(0, 1000, "A "), (1000, 2000, "B "), (2000, 3000, "C "), (3000, 4000, "D")])
    capf = {"chunk_ledger": [
        {"src_range": [0, 1000], "flush_end_wall": 1.0},
        {"src_range": [1000, 2000], "flush_end_wall": 2.0},
        {"src_range": [2000, 3000], "flush_end_wall": 2.5},
        {"src_range": [3000, 4000], "flush_end_wall": 3.0},
    ]}
    ev_two = [
        {"event_id": "p.0", "boundary": 1200, "semantic": "SEPARATE_OTHER", "avail": 2.2, "confirm_frame": 0},
        {"event_id": "p.1", "boundary": 2500, "semantic": "SEPARATE_OTHER", "avail": 2.8, "confirm_frame": 1},
    ]
    rp = r1_project(groups3, capf, [0, 4000], 7.0, ev_two)
    ck("c13-two-distinct-X", rp["n_seals"] == 2, rp["history"])
    ev_late = [{"event_id": "s.1", "boundary": 1500, "semantic": "SEPARATE_OTHER", "avail": 7.000001, "confirm_frame": 1}]
    rl = r1_project(groups3, capf, [0, 4000], 7.0, ev_late)
    ck("c13-1eps-late", rl["n_seals"] == 0 and any(h["outcome"] == "tooLate" for h in rl["history"]), rl["history"])
    from experiments.psem_e2o2_continuous_ownership.paced import lifetime_exercises
    lt = lifetime_exercises()
    ck("lifetimes-invalid-discontinuity-rollover", lt["n_pass"] == lt["n_total"], lt)
    n_pass = sum(1 for c in checks if c["pass"])
    return {"checks": checks, "n_pass": n_pass, "n_total": len(checks), "synthetics": syn, "watcher": w, "lifetimes": lt}


def c_blockers():
    src_hits = []
    src_root = ROOT / "src" / "puripuly_heart"
    if src_root.exists():
        for p in src_root.rglob("*.py"):
            try:
                t = p.read_text(encoding="utf-8")
            except Exception:
                continue
            if "too_late_for_current_scope" in t or "already_separated" in t:
                src_hits.append(str(p.relative_to(ROOT)))
    return [
        {
            "id": "C13_receiver_absent_in_HEAD_src",
            "blocks": "Experiment C and production R1/R2 application: HEAD src has no C13 receiver adapter",
            "evidence": {"src_hits": src_hits, "n": len(src_hits)},
        },
        {
            "id": "C13_retrospective_R2_not_authorized",
            "blocks": "production pre-translation ownership; remaining Audio adapter/capability, not a producer gap",
            "evidence": "issue 134 C13 defers retrospective PCM/text partition; paced producer/receiver exist in this experiment only",
        },
        {
            "id": "no_paid_asr_or_translation_this_assignment",
            "blocks": "Experiment C translation comparison",
            "evidence": "translate.py exists under experiments/psem_product_translation; keys not used; 0 calls",
        },
        {
            "id": "c5_7s_clips_not_production_scope",
            "blocks": "claiming C5 4s/6s legality",
            "evidence": FREEZE["c5_applicability"],
        },
        {
            "id": "r1_on_frozen_text_is_projection_only",
            "blocks": "end-to-end prospective PCM seal effect on ASR output",
            "evidence": "C13 applied Z is accepted frontier; real PCM seal would change subsequent ASR",
        },
    ]


def decide(ledger):
    cases = ledger["cases"]
    n_ind = sum(1 for c in cases.values() if c.get("independent_episode"))
    n_inst = sum(1 for c in cases.values() if c.get("instrumented_native"))
    n_cons = sum(1 for c in cases.values() if c.get("conservation", {}).get("conserved"))
    n_paced = sum(1 for c in cases.values() if c.get("paced_ok"))
    r2_better = 0
    r1_better = 0
    control_helps = 0
    guard_new_severe = []
    r1_pos = []
    r2_pos = []
    r1_worse = []
    r2_worse = []
    score_rows = []
    for cid, c in cases.items():
        s0 = (c.get("r0") or {}).get("score") or {}
        s1 = (c.get("r1") or {}).get("score") or {}
        s2 = (c.get("r2") or {}).get("score") or {}
        sc = (c.get("control_charged") or {}).get("r2_score") or {}
        row = {
            "id": cid,
            "independent": bool(c.get("independent_episode")),
            "r0_wrong": s0.get("n_wrong"),
            "r0_correct": s0.get("n_correct"),
            "r0_mixed": s0.get("n_mixed"),
            "r1_wrong": s1.get("n_wrong"),
            "r1_correct": s1.get("n_correct"),
            "r2_wrong": s2.get("n_wrong"),
            "r2_correct": s2.get("n_correct"),
            "control_r2_wrong": sc.get("n_wrong"),
        }
        score_rows.append(row)
        if s0.get("available") and s1.get("available"):
            if (s1.get("n_wrong") or 0) < (s0.get("n_wrong") or 0):
                r1_better += 1
                r1_pos.append(row)
            elif (s1.get("n_wrong") or 0) > (s0.get("n_wrong") or 0):
                r1_worse.append(row)
        if s0.get("available") and s2.get("available"):
            if (s2.get("n_wrong") or 0) < (s0.get("n_wrong") or 0):
                r2_better += 1
                r2_pos.append(row)
            elif (s2.get("n_wrong") or 0) > (s0.get("n_wrong") or 0):
                r2_worse.append(row)
        if sc.get("available") and s0.get("available"):
            if (sc.get("n_wrong") or 0) < (s0.get("n_wrong") or 0):
                control_helps += 1
        if cid in ("R1", "R2", "T1", "SINGLE_ES2009c", "SINGLE_ES2009d", "BC1"):
            if s1.get("available") and s0.get("available"):
                if (s1.get("n_wrong") or 0) > (s0.get("n_wrong") or 0) + 1:
                    guard_new_severe.append(cid)
    r2_ind = [r for r in r2_pos if r.get("independent")]
    r1_ind = [r for r in r1_pos if r.get("independent")]
    causal_feasible = n_paced > 0
    if n_paced == 0:
        disposition = "BLOCKED"
        reason = "Paced live producer/receiver did not complete."
        next_step = "Complete paced TCP runs. No Audio amendment. No training. No wait sweep."
    elif guard_new_severe:
        disposition = "BLOCKED"
        reason = f"Guard harm {guard_new_severe}; cannot select an integration path."
        next_step = "Do not adopt R1/R2. No Audio amendment. No training."
    elif len(r2_ind) >= 2 and causal_feasible and not r2_worse:
        disposition = "Audio decision for R2"
        reason = (
            "R2 is materially better on two independent sequential positives under paced QPC receipt before research terminals; "
            "R1 is not sufficient on those NPs. This is not production adoption. Audio must decide pre-translation ownership."
        )
        next_step = (
            "Open one explicit Audio capability decision for preserved timing + pre-translation ownership. "
            "Do not implement R2 in production Audio in this issue. No training. No wait sweep."
        )
    elif len(r1_ind) >= 2 and not r1_worse:
        disposition = "narrow R1 adoption/integration"
        reason = "R1 reduced wrong-owner groups on two independent episodes; R2 not required for that effect. Not production C5/Audio adoption."
        next_step = "Keep the narrow prospective path as a research integration recommendation. Do not build retrospective ownership. No training."
    elif control_helps and r1_better == 0 and r2_better == 0:
        disposition = "one named bottleneck repair"
        reason = "Correct-transition control helps; native R1/R2 did not reduce wrong-owner groups."
        next_step = "Do not train. Name one bottleneck only after this record; no wait sweep."
    elif control_helps == 0:
        disposition = "stop"
        reason = "Correct-transition control does not help this ownership/application target."
        next_step = "Stop this ownership/application target. Do not optimize speaker AP for it. No training."
    else:
        disposition = "BLOCKED"
        reason = (
            "Insufficient independent positives for R1, and R2 is not a production-authorized success path without an Audio decision. "
            "Not an invented fifth success disposition."
        )
        next_step = "Do not amend Audio in this issue without an explicit Audio decision. No training. No wait sweep."
    return {
        "disposition": disposition,
        "reason": reason,
        "next_step": next_step,
        "causal_feasible": causal_feasible,
        "n_independent_episodes": n_ind,
        "n_instrumented": n_inst,
        "n_paced": n_paced,
        "n_conserved": n_cons,
        "research_r2_fewer_wrong_than_r0": r2_better,
        "research_r1_fewer_wrong_than_r0": r1_better,
        "charged_control_helps_r2": control_helps,
        "guard_new_severe": guard_new_severe,
        "credited_improvements": 0,
        "r1_positive_cases": r1_pos,
        "r2_positive_cases": r2_pos,
        "r1_wrong_increased": r1_worse,
        "r2_wrong_increased": r2_worse,
        "score_rows": score_rows,
        "c_blockers": [b["id"] for b in c_blockers()],
        "production_adapter_gap": "HEAD src has no C13 receiver; #134 does not authorize retrospective R2. Paced experiment producer/receiver is not that production adapter.",
    }




def run_all(native=True, force_native=False):
    started = utc_now()
    t0 = time.perf_counter()
    sm = smoke()
    native_results = {}
    if native:
        for sid in FREEZE["native_runs"]["sources"]:
            native_results[sid] = run_native_one(sid, force=force_native)
    else:
        for sid in FREEZE["native_runs"]["sources"]:
            mp = EXP / "native" / sid / "meta.json"
            if mp.exists():
                native_results[sid] = load_json(mp)
    bundles = {}
    needed = sorted({FREEZE["cases"][c]["source"] for c in CASE_ORDER})
    for sid in needed:
        meta = native_results.get(sid)
        bundles[sid] = load_source_bundle(sid, meta)
    cases = {}
    for cid in CASE_ORDER:
        cases[cid] = run_case(cid, bundles)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()
    blockers = c_blockers()
    ledger = {
        "freeze_id": FREEZE["freeze_id"],
        "started_utc": started,
        "finished_utc": utc_now(),
        "wall_s": time.perf_counter() - t0,
        "head": head,
        "director_named_baseline": FREEZE["authority"]["director_named_baseline"],
        "exe_sha256": FREEZE["profile"]["exe_sha256"],
        "model_sha256": FREEZE["profile"]["model_sha256"],
        "smoke": {"n_pass": sm["n_pass"], "n_total": sm["n_total"], "checks": sm["checks"]},
        "native": native_results,
        "cases": cases,
        "c_blockers": blockers,
        "comparison_counts": {
            "n_cases": len(cases),
            "n_independent": sum(1 for c in cases.values() if c.get("independent_episode")),
            "n_instrumented": sum(1 for c in cases.values() if c.get("instrumented_native")),
            "n_events_total": sum(c.get("n_events") or 0 for c in cases.values()),
            "n_r1_seals_total": sum((c.get("r1") or {}).get("n_seals") or 0 for c in cases.values()),
        },
        "event_timing_validity": {
            "file_mode_poll_is_causal_gpu": False,
            "paced_qpc_receipt_is_experiment_A_causal": True,
            "shared_epoch": "QueryPerformanceCounter",
            "source_support_distinct_from_file_mode": True,
            "chained_service_us_is_causal": False,
            "asr_seal_wall_mixes_with_file_mode": False,
        },
    }
    decision = decide(ledger)
    ledger["decision"] = decision
    ledger_sha = write_json(EXP / "ledger.json", ledger)
    dec_sha = write_json(EXP / "DECISION.json", {
        "freeze_id": FREEZE["freeze_id"],
        "head": head,
        "decision": decision,
        "smoke": {"n_pass": sm["n_pass"], "n_total": sm["n_total"]},
        "native_ok": {k: v.get("ok") for k, v in native_results.items()},
        "comparison_counts": ledger["comparison_counts"],
        "event_timing_validity": ledger["event_timing_validity"],
        "c_blockers": blockers,
        "ledger_sha256": ledger_sha,
        "unverified": [
            "live microphone capture (paced prefix is 16 kHz simulated capture)",
            "production C13 receiver adapter in HEAD src",
            "Experiment C translation path",
            "C5 4s/6s production scopes",
            "production Audio R2 pre-translation ownership",
        ],
    })
    manifest = {
        "freeze_id": FREEZE["freeze_id"],
        "freeze_sha256": sha256_file(EXP / "FREEZE.json"),
        "ledger_sha256": ledger_sha,
        "decision_sha256": dec_sha,
        "head": head,
        "director_named_baseline": FREEZE["authority"]["director_named_baseline"],
        "exe": FREEZE["profile"]["exe"],
        "exe_sha256": FREEZE["profile"]["exe_sha256"],
        "model": FREEZE["profile"]["model"],
        "model_sha256": FREEZE["profile"]["model_sha256"],
        "native": {
            sid: {
                "ok": (native_results.get(sid) or {}).get("ok"),
                "wall_s": (native_results.get(sid) or {}).get("wall_s"),
                "n_chunks": (native_results.get(sid) or {}).get("n_chunks"),
                "trace_sha256": (native_results.get(sid) or {}).get("trace_sha256"),
                "probs_sha256": (native_results.get(sid) or {}).get("probs_sha256"),
                "wav_sha256": (native_results.get(sid) or {}).get("wav_sha256"),
                "causal_gpu_availability": (native_results.get(sid) or {}).get("causal_gpu_availability"),
                "time_to_first_chunk_poll_hi_s": (native_results.get(sid) or {}).get("time_to_first_chunk_poll_hi_s"),
                "first_chunk_source_support_s": (native_results.get(sid) or {}).get("first_chunk_source_support_s"),
                "n_chunks_poll_hi_before_source_support": (native_results.get(sid) or {}).get("n_chunks_poll_hi_before_source_support"),
                "n_chunks_poll_hi_after_source_support": (native_results.get(sid) or {}).get("n_chunks_poll_hi_after_source_support"),
            }
            for sid in FREEZE["native_runs"]["sources"]
        },
        "prefixes": FREEZE["prefixes"],
        "inputs": FREEZE["inputs"],
        "disposition": decision["disposition"],
        "comparison_counts": ledger["comparison_counts"],
        "event_timing_validity": ledger["event_timing_validity"],
        "owned_only": str(EXP.relative_to(ROOT)),
        "paced_exe_sha256": FREEZE["profile"].get("paced_exe_sha256"),
        "commit_include": [
            "experiments/psem_e2o2_continuous_ownership/run.py",
            "experiments/psem_e2o2_continuous_ownership/paced.py",
            "experiments/psem_e2o2_continuous_ownership/FREEZE.json",
            "experiments/psem_e2o2_continuous_ownership/DECISION.json",
            "experiments/psem_e2o2_continuous_ownership/ledger.json",
            "experiments/psem_e2o2_continuous_ownership/MANIFEST.json",
            "experiments/psem_e2o2_continuous_ownership/smoke.json",
            "experiments/psem_e2o2_continuous_ownership/paced_summary.json",
            "experiments/psem_e2o2_continuous_ownership/native_summary.json",
            "experiments/psem_e2o2_continuous_ownership/native_src/",
            "experiments/psem_e2o2_continuous_ownership/inputs/",
            "experiments/psem_e2o2_continuous_ownership/paced/*/arrivals.json",
            "experiments/psem_e2o2_continuous_ownership/native/*/meta.json",
        ],
        "commit_exclude": [
            "experiments/psem_e2o2_continuous_ownership/native/*/polls.json",
            "experiments/psem_e2o2_continuous_ownership/native/*/diar.trace.json",
            "experiments/psem_e2o2_continuous_ownership/paced/*/tcp_lines.json",
            "**/__pycache__/",
            "C:/tmp/psem-e2o2-native/",
            "C:/tmp/psem-e2o2-paced-dumps/",
        ],
    }
    write_json(EXP / "MANIFEST.json", manifest)
    write_json(EXP / "smoke.json", sm)
    return {"smoke": sm, "native": native_results, "decision": decision, "ledger_sha256": ledger_sha, "decision_sha256": dec_sha}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["smoke", "synthetic", "native", "paced", "compare", "all"])
    ap.add_argument("--force-native", action="store_true")
    a = ap.parse_args()
    if a.cmd == "smoke":
        sm = smoke()
        write_json(EXP / "smoke.json", sm)
        print(json.dumps({"n_pass": sm["n_pass"], "n_total": sm["n_total"]}, indent=1))
        return 0 if sm["n_pass"] == sm["n_total"] else 1
    if a.cmd == "synthetic":
        syn = run_synthetics()
        print(json.dumps(syn, indent=1))
        return 0 if syn["n_pass"] == syn["n_total"] else 1
    if a.cmd == "native":
        out = {}
        for sid in FREEZE["native_runs"]["sources"]:
            out[sid] = run_native_one(sid, force=a.force_native)
            slim = {k: out[sid].get(k) for k in ("ok", "source_id", "wall_s", "n_chunks", "error")}
            print(json.dumps(slim, indent=1))
        write_json(EXP / "native_summary.json", out)
        return 0 if all(v.get("ok") for v in out.values()) else 1
    if a.cmd == "paced":
        from experiments.psem_e2o2_continuous_ownership.paced import run_paced_all
        out = run_paced_all(force=a.force_native)
        return 0 if all(v.get("ok") for v in out.values()) else 1
    if a.cmd == "all":
        from experiments.psem_e2o2_continuous_ownership.paced import run_paced_all
        run_paced_all(force=a.force_native)
        r = run_all(native=False, force_native=False)
        print(json.dumps({"decision": r["decision"]["disposition"], "smoke": r["smoke"]["n_pass"], "ledger": r["ledger_sha256"]}, indent=1))
        return 0 if r["smoke"]["n_pass"] == r["smoke"]["n_total"] else 1
    if a.cmd == "compare":
        r = run_all(native=False, force_native=False)
        print(json.dumps({"decision": r["decision"]["disposition"], "smoke": r["smoke"]["n_pass"], "ledger": r["ledger_sha256"]}, indent=1))
        return 0 if r["smoke"]["n_pass"] == r["smoke"]["n_total"] else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
