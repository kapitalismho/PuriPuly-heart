"""Return capacity probe on frozen native observations."""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
EXP = ROOT / "experiments" / "psem_return_capacity"
PHASEA = ROOT / "experiments" / "psem_phase_a_headroom"
DEC = ROOT / "experiments" / "psem_decision_sufficiency"
STAGE2 = ROOT / "experiments" / "psem_repeatability_stage2"
FREEZE = json.loads((EXP / "FREEZE.json").read_text(encoding="utf-8"))
PHASEA_FREEZE = json.loads((PHASEA / "FREEZE.json").read_text(encoding="utf-8"))
from experiments.psem_decision_sufficiency.replay import align_window as ds_align_window
from experiments.psem_decision_sufficiency.replay import partition_ownership as ds_partition_ownership
from experiments.psem_decision_sufficiency.replay import REGION_PAD
import experiments.psem_phase_a_headroom.live_headroom as LH
TAU = float(FREEZE["f0_rule"]["tau"])
CONFIRMATION = int(FREEZE["f0_rule"]["confirmation"])
SENS = 1280
CASES = ("NP1", "NP2", "NP3")
SID_OF = {"NP1": "ami_ES2009c", "NP2": "ami_ES2009d", "NP3": "ami_ES2002b"}
MEET_OF = {"NP1": "ES2009c", "NP2": "ES2009d", "NP3": "ES2002b"}
ANCHOR_ROLE = {"NP1": "B", "NP2": "B", "NP3": "D"}
BARRIER_SHA = str(FREEZE["authority"]["obs_sha256"])
ORACLE_RETURN = 680320
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
def load_np(path, shape):
    import numpy as np
    raw = np.fromfile(str(path), dtype=np.float32)
    return raw.reshape(shape[0], shape[1])
def verify_inputs():
    errs = []
    for rel, want in FREEZE["inputs"].items():
        p = ROOT / rel
        if not p.exists():
            errs.append(rel)
            continue
        if sha256_file(p) != want:
            errs.append(rel)
    return errs
def positive_scope_of(emit, pay):
    if emit is None:
        return "no-event"
    if emit < pay[0] or emit >= pay[1]:
        return "invalid_scope"
    return "valid"
def guard_scope_of(boundary, obj):
    if boundary is None:
        return "no-event"
    if boundary < obj[0] or boundary >= obj[1]:
        return "invalid_scope"
    return "valid"
def receive_current_positive(requests, seal):
    applied = None
    history = []
    for req in requests:
        slim = {"boundary": req["boundary"], "emit": req["emit"], "frontier": req["frontier"], "kind": req["kind"], "op": req["op"]}
        if req["scope"] != "valid":
            history.append({"request": slim, "outcome": req["scope"], "applied_boundary": applied["boundary"] if applied else None})
            continue
        if applied is not None:
            history.append({"request": slim, "outcome": "already_separated", "applied_boundary": applied["boundary"]})
            continue
        if req["op"] != "split":
            history.append({"request": slim, "outcome": "unsupported_operation", "applied_boundary": None})
            continue
        if req["avail"] is None or seal is None:
            history.append({"request": slim, "outcome": "UNKNOWN", "applied_boundary": None})
            continue
        if req["avail"] > seal:
            history.append({"request": slim, "outcome": "too_late", "applied_boundary": None, "availability": req["avail"], "deadline": seal})
            continue
        applied = {"boundary": req["boundary"], "availability": req["avail"], "request": slim}
        history.append({"request": slim, "outcome": "applied", "applied_boundary": req["boundary"], "availability": req["avail"], "deadline": seal})
    return applied, history
def receive_current_guard(requests, obj):
    applied = None
    history = []
    for req in requests:
        slim = {"boundary": req["boundary"], "emit": req["emit"], "frontier": req["frontier"], "kind": req["kind"], "op": req["op"]}
        scope = guard_scope_of(req["boundary"], obj)
        if scope != "valid":
            history.append({"request": slim, "outcome": "invalid_scope", "applied_boundary": applied["boundary"] if applied else None})
            continue
        if applied is not None:
            history.append({"request": slim, "outcome": "already_separated", "applied_boundary": applied["boundary"]})
            continue
        if req["op"] != "split":
            history.append({"request": slim, "outcome": "unsupported_operation", "applied_boundary": None})
            continue
        applied = {"boundary": req["boundary"], "emit": req["emit"]}
        history.append({"request": slim, "outcome": "applied", "applied_boundary": req["boundary"]})
    return applied, history
def receive_counterfactual_positive(requests, pay, seal, initial_state):
    states = []
    history = []
    cur = initial_state
    applied_bounds = []
    for req in requests:
        slim = {"boundary": req["boundary"], "emit": req["emit"], "frontier": req["frontier"], "kind": req["kind"], "op": req["op"], "state": req["state"]}
        if req["scope"] != "valid":
            history.append({"request": slim, "outcome": req["scope"], "current_state": cur, "applied_bounds": list(applied_bounds)})
            continue
        if req["avail"] is None or seal is None:
            history.append({"request": slim, "outcome": "UNKNOWN", "current_state": cur, "applied_bounds": list(applied_bounds)})
            continue
        if req["avail"] > seal:
            history.append({"request": slim, "outcome": "too_late", "current_state": cur, "applied_bounds": list(applied_bounds), "availability": req["avail"], "deadline": seal})
            continue
        if req["state"] == cur:
            history.append({"request": slim, "outcome": "redundant_same_state", "current_state": cur, "applied_bounds": list(applied_bounds)})
            continue
        cur = req["state"]
        applied_bounds.append(req["boundary"])
        states.append({"boundary": req["boundary"], "state": req["state"]})
        history.append({"request": slim, "outcome": "applied_state", "current_state": cur, "applied_bounds": list(applied_bounds), "availability": req["avail"], "deadline": seal})
    return {"initial_state": initial_state, "final_state": cur, "applied_bounds": applied_bounds, "spans": states}, history
def receive_counterfactual_guard(requests, obj, initial_state):
    history = []
    cur = initial_state
    applied_bounds = []
    spans = []
    for req in requests:
        slim = {"boundary": req["boundary"], "emit": req["emit"], "frontier": req["frontier"], "kind": req["kind"], "op": req["op"], "state": req["state"]}
        scope = guard_scope_of(req["boundary"], obj)
        past_end = None
        if req.get("avail_source") is not None:
            past_end = req["avail_source"] - obj[1]
        if scope != "valid":
            history.append({"request": slim, "outcome": "invalid_scope", "current_state": cur, "applied_bounds": list(applied_bounds), "avail_source": req.get("avail_source"), "synthetic_cap": obj[1], "past_end": past_end})
            continue
        if req["state"] == cur:
            history.append({"request": slim, "outcome": "redundant_same_state", "current_state": cur, "applied_bounds": list(applied_bounds), "avail_source": req.get("avail_source"), "synthetic_cap": obj[1], "past_end": past_end})
            continue
        cur = req["state"]
        applied_bounds.append(req["boundary"])
        spans.append({"boundary": req["boundary"], "state": req["state"]})
        history.append({"request": slim, "outcome": "applied_state", "current_state": cur, "applied_bounds": list(applied_bounds), "avail_source": req.get("avail_source"), "synthetic_cap": obj[1], "past_end": past_end})
    return {"initial_state": initial_state, "final_state": cur, "applied_bounds": applied_bounds, "spans": spans}, history
def counterfactual_partition_proxy(gt_proxy, initial_state, ordered_bounds, return_spans=None):
    wrong = []
    false_left = []
    right_fixed = []
    right_unc = []
    ambiguous = []
    bounds = sorted(ordered_bounds, key=lambda x: x["boundary"])
    spans = return_spans if return_spans is not None else []
    for w in gt_proxy:
        st = initial_state
        for b in bounds:
            if w["end"] > b["boundary"]:
                st = b["state"]
            else:
                break
        assigned = "left" if st == "A" else "right"
        strad = any(w["start"] < b["boundary"] < w["end"] for b in bounds)
        if strad:
            ambiguous.append({"gt": w, "assigned": assigned})
            if assigned == "right" and w["side"] == "right":
                right_unc.append({"gt": w, "assigned": assigned})
            continue

        if assigned != w["side"]:
            wrong.append({"gt": w, "assigned": assigned})
            if w["side"] == "left":
                false_left.append({"gt": w, "assigned": assigned})
        elif w["side"] == "right":
            right_fixed.append({"gt": w, "assigned": assigned})
    n_def = len(wrong)
    n_unc = len(ambiguous)
    return {"n_wrong_definite": n_def, "n_uncertain_straddle": n_unc, "error_interval": [n_def, n_def + n_unc], "wrong_definite": wrong, "n_false_moved_left": len(false_left), "false_moved_left": false_left, "n_right_assigned_new": len(right_fixed), "ambiguous": ambiguous, "n_right_assigned_new_uncertain": len(right_unc)}
def counterfactual_states_proxy(gt_proxy, initial_state, ordered_bounds, return_spans=None):
    states = {}
    bounds = sorted(ordered_bounds, key=lambda x: x["boundary"])
    spans = return_spans if return_spans is not None else []
    amb = set()
    for w in gt_proxy:
        if any(w["start"] < b["boundary"] < w["end"] for b in bounds):
            amb.add(w["id"])
    part = counterfactual_partition_proxy(gt_proxy, initial_state, ordered_bounds, return_spans)
    wrong_ids = {r["gt"]["id"] for r in part["wrong_definite"]}
    for w in gt_proxy:
        gid = w["id"]
        if gid in amb:
            states[gid] = "uncertain"
        elif gid in wrong_ids:
            states[gid] = "wrong"
        else:
            states[gid] = "correct"
    return states, part
def counterfactual_partition_aligned(alignment, initial_state, ordered_bounds, return_spans=None):
    wrong = []
    false_left = []
    right_fixed = []
    right_unc = []
    ambiguous = []
    bounds = sorted(ordered_bounds, key=lambda x: x["boundary"])
    spans = return_spans if return_spans is not None else []
    for m in alignment.get("matched", []):
        g = m["gt"]
        key = m.get("group_end_src")
        if key is None:
            continue
        st = initial_state
        for b in bounds:
            if key > b["boundary"]:
                st = b["state"]
            else:
                break
        assigned = "left" if st == "A" else "right"
        strad = any(g["start"] < b["boundary"] < g["end"] for b in bounds)
        rec = {**m, "assigned": assigned, "straddling": bool(strad)}
        if strad:
            ambiguous.append(rec)
            if assigned == "right" and g["side"] == "right":
                right_unc.append(rec)
            continue

        if assigned != g["side"]:
            wrong.append(rec)
            if g["side"] == "left":
                false_left.append(rec)
        elif g["side"] == "right":
            right_fixed.append(rec)
    n_def = len(wrong)
    n_unc = len(ambiguous)
    return {"boundary_list": [b["boundary"] for b in bounds], "n_wrong_definite": n_def, "n_uncertain_straddle": n_unc, "error_interval": [n_def, n_def + n_unc], "wrong_definite": wrong, "n_false_moved_left": len(false_left), "false_moved_left": false_left, "n_right_assigned_new": len(right_fixed), "ambiguous": ambiguous, "n_right_assigned_new_uncertain": len(right_unc)}
def counterfactual_states_aligned(alignment, initial_state, ordered_bounds, return_spans=None):
    part = counterfactual_partition_aligned(alignment, initial_state, ordered_bounds, return_spans)
    amb_ids = {m["gt"]["id"] for m in part.get("ambiguous", [])}
    wrong_ids = {m["gt"]["id"] for m in part.get("wrong_definite", [])}
    states = {}
    for m in alignment.get("matched", []):
        gid = m["gt"]["id"]
        if gid in amb_ids:
            states[gid] = "uncertain"
        elif gid in wrong_ids:
            states[gid] = "wrong"
        else:
            states[gid] = "correct"
    return states, part
def partition_states(alignment, boundary):
    part = ds_partition_ownership(alignment, boundary)
    by_id = {}
    for m in alignment.get("matched", []):
        gid = m["gt"]["id"]
        by_id[gid] = {"gt": m["gt"], "group_end_src": m.get("group_end_src")}
    amb_ids = {m["gt"]["id"] for m in part.get("ambiguous", [])}
    wrong_ids = {m["gt"]["id"] for m in part.get("wrong_definite", [])}
    states = {}
    for gid, rec in by_id.items():
        if gid in amb_ids:
            states[gid] = "uncertain"
        elif gid in wrong_ids:
            states[gid] = "wrong"
        else:
            states[gid] = "correct"
    return states, part
def paired_transitions(gt_list, states_base, states_new):
    side_of = {w["id"]: w.get("side") for w in gt_list}
    counts = {"stillwrong": 0, "fixed": 0, "new_wrong": 0, "new_uncertain_pure": 0, "new_uncertain_right": 0, "wrong_to_uncertain": 0, "uncertain_other": 0}
    new_wrong_ids = []
    new_unc_pure_ids = []
    for gid, b in states_base.items():
        n = states_new.get(gid)
        if n is None:
            continue
        side = side_of.get(gid)
        if b == "wrong" and n == "wrong":
            counts["stillwrong"] += 1
        elif b == "wrong" and n == "correct":
            counts["fixed"] += 1
        elif b == "wrong" and n == "uncertain":
            counts["wrong_to_uncertain"] += 1
        elif b == "correct" and n == "wrong":
            counts["new_wrong"] += 1
            new_wrong_ids.append(gid)
        elif b == "correct" and n == "uncertain":
            if side == "left":
                counts["new_uncertain_pure"] += 1
                new_unc_pure_ids.append(gid)
            else:
                counts["new_uncertain_right"] += 1
        elif b == "uncertain" or n == "uncertain":
            counts["uncertain_other"] += 1
    return {"counts": counts, "new_wrong_ids": sorted(new_wrong_ids), "new_uncertain_pure_ids": sorted(new_unc_pure_ids)}
def gt8_for_case(case, groups):
    spec = PHASEA_FREEZE["cases"][case]
    gt8 = LH.gt_window_samples(case)
    wlo = min(w["start"] for w in gt8) - REGION_PAD
    whi = max(w["end"] for w in gt8) + REGION_PAD
    al8 = ds_align_window(gt8, groups, wlo, whi)
    return gt8, wlo, whi, al8
def scalar_ordered_events(frames, starts, ends, valid, masked, speech, p_anchor, frontiers, ready_sample):
    sc = [float(1.0 - v) for v in p_anchor]
    sr = [float(v) for v in p_anchor]
    t0 = time.perf_counter()
    ev_c = LH.f0_fire_events(frames, starts, ends, valid, masked, speech, sc, frontiers, TAU, CONFIRMATION, True)
    t1 = time.perf_counter()
    ev_r = LH.f0_fire_events(frames, starts, ends, valid, masked, speech, sr, frontiers, TAU, CONFIRMATION, True)
    t2 = time.perf_counter()
    cand = []
    for e in ev_c:
        cand.append({"boundary": int(e["boundary"]), "emit": int(e["emit"]), "frontier": int(e["frontier"]), "frame": int(e["frame"]), "op": "split", "state": "UNKNOWN", "kind": "scalar-change"})
    for e in ev_r:
        cand.append({"boundary": int(e["boundary"]), "emit": int(e["emit"]), "frontier": int(e["frontier"]), "frame": int(e["frame"]), "op": "set_state", "state": "A", "kind": "scalar-return"})
    cand = [c for c in cand if c["boundary"] >= ready_sample]
    cand = sorted(cand, key=lambda x: (x["boundary"], 0 if x["op"] == "split" else 1))
    ordered = []
    cur = "A"
    for c in cand:
        if c["state"] == cur:
            continue
        cur = c["state"]
        ordered.append(c)
    return ordered, (t1 - t0), (t2 - t1)
def oracle_ordered_events(change_events, oracle_boundary, oracle_emit, oracle_frontier, oracle_frame, ready_sample):
    cand = []
    for e in change_events:
        if int(e["boundary"]) < ready_sample:
            continue
        cand.append({"boundary": int(e["boundary"]), "emit": int(e["emit"]), "frontier": int(e["frontier"]), "frame": int(e["frame"]), "op": "split", "state": "UNKNOWN", "kind": "oracle-change"})
    cand.append({"boundary": int(oracle_boundary), "emit": int(oracle_emit), "frontier": int(oracle_frontier), "frame": int(oracle_frame), "op": "set_state", "state": "A", "kind": "oracle-return"})
    cand = sorted(cand, key=lambda x: (x["boundary"], 0 if x["op"] == "split" else 1))
    ordered = []
    cur = "A"
    for c in cand:
        if c["state"] == cur:
            continue
        cur = c["state"]
        ordered.append(c)
    return ordered
def oracle_availability(oracle_boundary, starts, ends, valid, masked, speech, frontiers):
    need_at = int(oracle_boundary) + CONFIRMATION
    best = None
    for i in range(len(starts)):
        if starts[i] < need_at:
            continue
        if not valid[i]:
            continue
        if masked[i]:
            continue
        if not speech[i]:
            continue
        best = i
        break
    if best is None:
        return None, None, None, None
    emit = need_at if need_at >= frontiers[best] else frontiers[best]
    return int(emit), int(frontiers[best]), int(best), int(best)
def return_spans_of(ordered):
    spans = []
    seq = sorted(ordered, key=lambda x: x["boundary"])
    for idx, e in enumerate(seq):
        if e.get("state") != "A":
            continue
        nxt = seq[idx + 1]["boundary"] if idx + 1 < len(seq) else 9223372036854775807
        spans.append([int(e["boundary"]), int(nxt)])
    return spans
def full4_ordered_events(frames, starts, ends, valid, masked, speech, p_anchor, maxothers, frontiers, ready_sample):
    sc = [float(1.0 - v) for v in p_anchor]
    sr = [float(v) for v in p_anchor]
    valid_ret = [bool(valid[i] and maxothers[i] < 0.5) for i in range(len(valid))]
    ev_c = LH.f0_fire_events(frames, starts, ends, valid, masked, speech, sc, frontiers, TAU, CONFIRMATION, True)
    ev_r = LH.f0_fire_events(frames, starts, ends, valid_ret, masked, speech, sr, frontiers, TAU, CONFIRMATION, True)
    cand = []
    for e in ev_c:
        cand.append({"boundary": int(e["boundary"]), "emit": int(e["emit"]), "frontier": int(e["frontier"]), "frame": int(e["frame"]), "op": "split", "state": "UNKNOWN", "kind": "full4-change"})
    for e in ev_r:
        cand.append({"boundary": int(e["boundary"]), "emit": int(e["emit"]), "frontier": int(e["frontier"]), "frame": int(e["frame"]), "op": "set_state", "state": "A", "kind": "full4-return"})
    cand = [c for c in cand if c["boundary"] >= ready_sample]
    cand = sorted(cand, key=lambda x: (x["boundary"], 0 if x["op"] == "split" else 1))
    ordered = []
    cur = "A"
    for c in cand:
        if c["state"] == cur:
            continue
        cur = c["state"]
        ordered.append(c)
    return ordered
def overlap_diagnostic(gt_proxy, spans):
    n_r = 0
    ids = []
    for w in gt_proxy:
        if w.get("side") != "right":
            continue
        if any(w["start"] < sp[1] and w["end"] > sp[0] for sp in spans):
            n_r += 1
            ids.append(w["id"])
    return {"n_right_overlapping": n_r, "ids": sorted(ids), "spans": spans}
def run_positive(case, obs, cache, probs_by_sid):
    spec = PHASEA_FREEZE["cases"][case]
    fz = FREEZE["cases"][case]
    sid = SID_OF[case]
    meet = MEET_OF[case]
    arole = ANCHOR_ROLE[case]
    src_entry = next(s for s in obs["sources"] if s["source_id"] == sid)
    probs = probs_by_sid[sid]
    table = LH.build_native_table(src_entry, cache["sources"][sid])
    n = src_entry["valid_native_frames"]
    ep = spec["episode_span_samples"]
    pay = spec["payload_samples"]
    b = spec["boundary_samples"]
    if case in ("NP1", "NP3"):
        mlo, mhi = ep[0], pay[0]
    else:
        mlo, mhi = ep[0], b
    mf0, mf1 = int(mlo // 1280), int((mhi - 1) // 1280) + 1
    support = LH.anchor_support_frames(meet, arole, mf0, mf1)
    mapping = LH.map_anchor_slot(probs, table["valid_native"], support)
    if mapping is None:
        return {"case": case, "status": "mapping-failure-separable-P3R-no-events-admitted"}
    slot = mapping["slot"]
    p_anchor = [float(probs[i, slot]) for i in range(n)]
    valid = [bool(table["valid_native"][i] and table["valid_old"][i]) for i in range(n)]
    f0, f1 = int(ep[0] // 1280), int((ep[1] - 1) // 1280) + 1
    ep_frames = list(range(max(f0, 0), min(f1, n)))
    t_dec0 = time.perf_counter()
    import numpy as _np
    sc_list = [float(1.0 - v) for v in p_anchor]
    ev_orig = LH.f0_fire_events(ep_frames, table["starts"], table["ends"], valid, table["masked"], table["speech"], sc_list, table["frontiers"], TAU, CONFIRMATION, False)
    t_dec1 = time.perf_counter()
    ev_rearm = LH.f0_fire_events(ep_frames, table["starts"], table["ends"], valid, table["masked"], table["speech"], sc_list, table["frontiers"], TAU, CONFIRMATION, True)
    t_dec2 = time.perf_counter()
    ordered_scalar, cpu_c, cpu_r = scalar_ordered_events(ep_frames, table["starts"], table["ends"], valid, table["masked"], table["speech"], p_anchor, table["frontiers"], mapping["ready_sample"])
    import numpy as _np2
    _mo = [float(max(probs[i, j] for j in range(probs.shape[1]) if j != slot)) for i in range(n)]
    ordered_full4 = full4_ordered_events(ep_frames, table["starts"], table["ends"], valid, table["masked"], table["speech"], p_anchor, _mo, table["frontiers"], mapping["ready_sample"])
    t_dec3 = time.perf_counter()
    decision_cpu_orig = t_dec1 - t_dec0
    decision_cpu_rearm = t_dec2 - t_dec1
    decision_cpu_scalar = (t_dec3 - t_dec2) + cpu_c + cpu_r
    decision_cpu_full4 = decision_cpu_scalar
    first = ev_orig[0] if ev_orig else None
    cap = json.loads((STAGE2 / "captures" / (case + ".json")).read_text(encoding="utf-8"))
    init_s = LH.trace_init_s(src_entry)
    sched = LH.build_schedule(src_entry, cap, pay, init_s)
    backlog = sched["zero_finish"]
    ch_of = sched["chunk_of_frame"]
    fin = sched["finish"]
    def event_avail(frame_idx, cpu):
        ci = ch_of[frame_idx] if 0 <= frame_idx < len(ch_of) else None
        if ci is None:
            return None, None, "UNKNOWN-no-chunk"
        if fin[ci] is None:
            return None, ci, "UNKNOWN-support-past-payload"
        return fin[ci] + cpu, ci, "measured"
    seal = cap.get("session", {}).get("seal_wall")
    groups = cap.get("groups", [])
    gt8, wlo, whi, al8 = gt8_for_case(case, groups)
    full_gt = LH.gt_words_in_span(meet, pay[0] / 16000.0, pay[1] / 16000.0)
    full_al = ds_align_window([{"id": str(i), "text": w["text"], "side": "payload", "in_span": True, "start": w["start"], "end": w["end"]} for i, w in enumerate(full_gt)], groups, pay[0] - REGION_PAD, pay[1] + REGION_PAD)
    t_cpu = time.perf_counter()
    rearm_reqs = []
    for e in ev_rearm:
        av, ci, rel = event_avail(e["frame"], decision_cpu_rearm)
        rearm_reqs.append({"boundary": e["boundary"], "emit": e["emit"], "frontier": e["frontier"], "frame": e["frame"], "scope": positive_scope_of(e["emit"], pay), "avail": av, "cpu": decision_cpu_rearm, "kind": "f0-rearm", "op": "split", "state": "UNKNOWN", "release": rel})
    orig_reqs = []
    if first is not None:
        av0, _, rel0 = event_avail(first["frame"], decision_cpu_orig)
        orig_reqs.append({"boundary": first["boundary"], "emit": first["emit"], "frontier": first["frontier"], "frame": first["frame"], "scope": positive_scope_of(first["emit"], pay), "avail": av0, "cpu": decision_cpu_orig, "kind": "f0-single-fire", "op": "split", "state": "UNKNOWN", "release": rel0})
    scalar_reqs_current = []
    scalar_reqs_counter = []
    for e in ordered_scalar:
        av, ci, rel = event_avail(e["frame"], decision_cpu_scalar)
        base = {"boundary": e["boundary"], "emit": e["emit"], "frontier": e["frontier"], "frame": e["frame"], "scope": positive_scope_of(e["emit"], pay), "avail": av, "cpu": decision_cpu_scalar, "kind": e["kind"], "op": e["op"], "state": e["state"], "release": rel}
        scalar_reqs_current.append(dict(base))
        scalar_reqs_counter.append(dict(base))
    full4_reqs_current = []
    full4_reqs_counter = []
    for e in ordered_full4:
        av, ci, rel = event_avail(e["frame"], decision_cpu_full4)
        base = {"boundary": e["boundary"], "emit": e["emit"], "frontier": e["frontier"], "frame": e["frame"], "scope": positive_scope_of(e["emit"], pay), "avail": av, "cpu": decision_cpu_full4, "kind": e["kind"], "op": e["op"], "state": e["state"], "release": rel}
        full4_reqs_current.append(dict(base))
        full4_reqs_counter.append(dict(base))
    oracle_reqs_current = [dict(r) for r in scalar_reqs_current]
    oracle_reqs_counter = [dict(r) for r in scalar_reqs_counter]
    for r in oracle_reqs_current:
        r["kind"] = r["kind"].replace("scalar", "oracle-held-scalar")
    for r in oracle_reqs_counter:
        r["kind"] = r["kind"].replace("scalar", "oracle-held-scalar")
    arms_current = {}
    for arm, reqs in (("none", []), ("f0_original", orig_reqs), ("f0_rearm", rearm_reqs), ("s_scalar", scalar_reqs_current), ("s_oracle", oracle_reqs_current), ("s_full4", full4_reqs_current)):
        applied, history = receive_current_positive(reqs, seal)
        ab = applied["boundary"] if applied else None
        cur_part = ds_partition_ownership(ds_align_window(gt8, groups, wlo, whi), ab)
        cur_states, _ = partition_states(ds_align_window(gt8, groups, wlo, whi), ab)
        arms_current[arm] = {"requests": history, "applied_boundary": ab, "availability": applied["availability"] if applied else None, "deadline": seal, "ownership_8": cur_part, "states": cur_states}
    pay_start = pay[0]
    def initial_for(reqs):
        return "A"
    arms_counter = {}
    for arm, reqs in (("s_scalar", scalar_reqs_counter), ("s_oracle", oracle_reqs_counter), ("s_full4", full4_reqs_counter)):
        init = initial_for(reqs)
        res, hist = receive_counterfactual_positive(reqs, pay, seal, init)
        bounds = [{"boundary": bb, "state": st["state"]} for bb, st in zip(res["applied_bounds"], res["spans"])]
        if not bounds:
            for r in sorted(reqs, key=lambda x: x["boundary"]):
                if r["scope"] != "valid":
                    continue
                if r["avail"] is None or seal is None:
                    continue
                if r["avail"] > seal:
                    continue
                bounds.append({"boundary": r["boundary"], "state": r["state"]})
                break
            ordered_all = []
            cur2 = init
            for r in sorted(reqs, key=lambda x: x["boundary"]):
                if r["scope"] != "valid":
                    continue
                if r["avail"] is None or seal is None:
                    continue
                if r["avail"] > seal:
                    continue
                if r["state"] == cur2:
                    continue
                cur2 = r["state"]
                ordered_all.append({"boundary": r["boundary"], "state": r["state"]})
            bounds = ordered_all
        full_ordered = ordered_scalar if arm in ("s_scalar", "s_oracle") else ordered_full4
        spans = return_spans_of(full_ordered)
        cstates, cpart = counterfactual_states_aligned(ds_align_window(gt8, groups, wlo, whi), init, bounds, spans)
        arms_counter[arm] = {"requests": hist, "initial_state": init, "applied_bounds": res["applied_bounds"], "final_state": res["final_state"], "ownership_8": cpart, "states": cstates, "return_spans": spans}
    recv_cpu = time.perf_counter() - t_cpu
    order_single = ["none", "f0_original", "f0_rearm", "s_scalar", "s_oracle", "s_full4"]
    parts = {}
    states = {}
    for a in order_single:
        parts[a] = arms_current[a]["ownership_8"]
        states[a] = arms_current[a]["states"]
    for a in ("s_scalar_counterfactual", "s_oracle_counterfactual", "s_full4_counterfactual"):
        src_arm = "s_scalar" if a.startswith("s_scalar") else ("s_full4" if a.startswith("s_full4") else "s_oracle")
        parts[a] = arms_counter[src_arm]["ownership_8"]
        states[a] = arms_counter[src_arm]["states"]
    gt_ids = [{"id": m["gt"]["id"], "side": m["gt"]["side"]} for m in al8["matched"]]
    deltas = {}
    for pair in (("f0_rearm", "none"), ("f0_rearm", "f0_original"), ("s_scalar", "f0_original"), ("s_oracle", "f0_original"), ("s_full4", "f0_original"), ("s_scalar_counterfactual", "none"), ("s_oracle_counterfactual", "none"), ("s_full4_counterfactual", "none")):
        new, base = pair
        nb = parts[new]["error_interval"]
        bb = parts[base]["error_interval"]
        deltas[new + "_vs_" + base] = {"definite": nb[0] - bb[0], "pessimistic": nb[1] - bb[1], "new": nb, "base": bb}
    transitions = {}
    for pair in (("f0_rearm", "none"), ("f0_rearm", "f0_original"), ("s_scalar", "f0_original"), ("s_oracle", "f0_original"), ("s_full4", "f0_original"), ("s_scalar_counterfactual", "none"), ("s_oracle_counterfactual", "none"), ("s_full4_counterfactual", "none"), ("s_scalar_counterfactual", "f0_original"), ("s_oracle_counterfactual", "f0_original"), ("s_full4_counterfactual", "f0_original")):
        new, base = pair
        transitions[new + "_vs_" + base] = paired_transitions(gt_ids, states[base], states[new])
    rframe = mapping["ready_frames"][1]
    rci = ch_of[rframe] if 0 <= rframe < len(ch_of) else None
    ready_session = fin[rci] if (rci is not None and fin[rci] is not None) else None
    acc_toks = cap.get("accepted", {}).get("tokens", [])
    cons = LH.conservation_check(acc_toks, groups)
    ctrls = LH.throwaway_controls(acc_toks, groups)
    n_applied = sum(1 for h in arms_current["f0_rearm"]["requests"] if h["outcome"] == "applied")
    n_red = sum(1 for h in arms_current["f0_rearm"]["requests"] if h["outcome"] == "already_separated")
    frag = {"applied_new_bounds": n_applied, "redundant_repeated": n_red, "n_requests": len(arms_current["f0_rearm"]["requests"])}
    qa = LH.qa_merge_for_case(case, groups)
    full_detail = {"n_gt": len(full_gt), "n_matched": len(full_al["matched"]), "n_unmatched": len(full_al["unmatched"]), "n_mixed": len(full_al["mixed"])}
    sens = {}
    ab_r = arms_current["f0_rearm"]["applied_boundary"]
    for delta in (-SENS, SENS):
        bb = None if ab_r is None else ab_r + delta
        sens[str(delta)] = ds_partition_ownership(ds_align_window(gt8, groups, wlo, whi), bb)["error_interval"]
    return {"case": case, "source": sid, "mapping": mapping, "mapping_scope": [mlo, mhi], "mapping_support_n": len(support), "f0_original_events": ev_orig, "f0_rearm_events": ev_rearm, "scalar_ordered_events": ordered_scalar, "full4_ordered_events": ordered_full4, "decision_cpu_s": {"f0_original": decision_cpu_orig, "f0_rearm": decision_cpu_rearm, "scalar": decision_cpu_scalar, "receiver": recv_cpu}, "warm_backlog_at_capture_entry_s": backlog, "ready_session_clock": ready_session, "arms_current": arms_current, "arms_counterfactual": arms_counter, "ownership_sens_pm1280_rearm": sens, "deltas_fixed_cohort": deltas, "paired_transitions": transitions, "full_payload_alignment": full_detail, "conservation": cons, "conservation_controls": ctrls, "fragmentation": frag, "qa_merge": qa}
def run_guards(obs, cache, probs_by_sid):
    t0 = time.perf_counter()
    out = {}
    en_entry = next(s for s in obs["sources"] if s["source_id"] == "ami_EN2009d")
    en_probs = probs_by_sid["ami_EN2009d"]
    en_table = LH.build_native_table(en_entry, cache["sources"]["ami_EN2009d"])
    en_n = en_entry["valid_native_frames"]
    gdefs = FREEZE["guards"]
    en_map_scope = gdefs["R2"]["map_scope_samples"]
    mf0, mf1 = int(en_map_scope[0] // 1280), int((en_map_scope[1] - 1) // 1280) + 1
    mf0, mf1 = max(mf0, 0), min(mf1, en_n)
    en_support = LH.anchor_support_frames("EN2009d", "A", mf0, mf1)
    en_mapping = LH.map_anchor_slot(en_probs, en_table["valid_native"], en_support)
    en_slot = en_mapping["slot"]
    en_p_anchor = [float(en_probs[i, en_slot]) for i in range(en_n)]
    en_valid = [bool(en_table["valid_native"][i] and en_table["valid_old"][i]) for i in range(en_n)]
    ep = FREEZE["ref_epoch_EN2009d"]["full_epoch_samples"]
    f0e, f1e = int(ep[0] // 1280), int((ep[1] - 1) // 1280) + 1
    ep_frames = list(range(max(f0e, 0), min(f1e, en_n)))
    t_dec0 = time.perf_counter()
    en_sc = [float(1.0 - v) for v in en_p_anchor]
    en_orig = LH.f0_fire_events(ep_frames, en_table["starts"], en_table["ends"], en_valid, en_table["masked"], en_table["speech"], en_sc, en_table["frontiers"], TAU, CONFIRMATION, False)
    t_dec1 = time.perf_counter()
    en_rearm = LH.f0_fire_events(ep_frames, en_table["starts"], en_table["ends"], en_valid, en_table["masked"], en_table["speech"], en_sc, en_table["frontiers"], TAU, CONFIRMATION, True)
    t_dec2 = time.perf_counter()
    ordered_scalar, cpu_c, cpu_r = scalar_ordered_events(ep_frames, en_table["starts"], en_table["ends"], en_valid, en_table["masked"], en_table["speech"], en_p_anchor, en_table["frontiers"], en_mapping["ready_sample"])
    _en_mo = [float(max(en_probs[i, j] for j in range(en_probs.shape[1]) if j != en_slot)) for i in range(en_n)]
    ordered_full4 = full4_ordered_events(ep_frames, en_table["starts"], en_table["ends"], en_valid, en_table["masked"], en_table["speech"], en_p_anchor, _en_mo, en_table["frontiers"], en_mapping["ready_sample"])
    t_dec3 = time.perf_counter()
    en_cpu_orig = t_dec1 - t_dec0
    en_cpu_rearm = t_dec2 - t_dec1
    en_cpu_scalar = (t_dec3 - t_dec2) + cpu_c + cpu_r
    en_cpu_full4 = en_cpu_scalar
    old_frame = 374400 // 1280
    old_valid_check = {"frame": int(old_frame), "valid_old": bool(en_table["valid_old"][old_frame]), "masked": bool(en_table["masked"][old_frame]), "speech": bool(en_table["speech"][old_frame]), "reason": str(en_table["old_reason"][old_frame])}
    oracle_emit, oracle_front, oracle_frame, oracle_first_valid = oracle_availability(ORACLE_RETURN, en_table["starts"], en_table["ends"], en_valid, en_table["masked"], en_table["speech"], en_table["frontiers"])
    oracle_ordered = oracle_ordered_events(en_rearm, ORACLE_RETURN, oracle_emit, oracle_front, oracle_frame if oracle_frame is not None else 538, en_mapping["ready_sample"])
    mask_gap = {"frames": [530, 531, 532, 533, 534, 535, 536, 537], "span": [678400, 688640], "valid_old_all_false": all(not en_table["valid_old"][i] for i in [530, 531, 532, 533, 534, 535, 536, 537])}
    mask_delay = {"gt_confirm": ORACLE_RETURN + CONFIRMATION, "first_valid_start": en_table["starts"][oracle_first_valid] if oracle_first_valid is not None else None, "first_valid_frame": oracle_first_valid, "source_wait": (en_table["starts"][oracle_first_valid] - (ORACLE_RETURN + CONFIRMATION)) if oracle_first_valid is not None else None, "frontier": oracle_front, "frontier_delay": (oracle_front - (ORACLE_RETURN + CONFIRMATION)) if oracle_front is not None else None, "emit": oracle_emit}
    scalar_return_rec = next((e for e in ordered_scalar if e["kind"] == "scalar-return"), None)
    scalar_delay = {"gt_return": ORACLE_RETURN, "scalar_boundary": scalar_return_rec["boundary"] if scalar_return_rec else None, "scalar_emit": scalar_return_rec["emit"] if scalar_return_rec else None, "gap_samples": (scalar_return_rec["boundary"] - ORACLE_RETURN) if scalar_return_rec else None}
    ea_entry = next(s for s in obs["sources"] if s["source_id"] == "ami_ES2009a")
    ea_probs = probs_by_sid["ami_ES2009a"]
    ea_table = LH.build_native_table(ea_entry, cache["sources"]["ami_ES2009a"])
    ea_n = ea_entry["valid_native_frames"]
    r1_scope = gdefs["R1"]["map_scope_samples"]
    q0, q1 = int(r1_scope[0] // 1280), int((r1_scope[1] - 1) // 1280) + 1
    q0, q1 = max(q0, 0), min(q1, ea_n)
    r1_support = LH.anchor_support_frames("ES2009a", "A", q0, q1)
    r1_mapping = LH.map_anchor_slot(ea_probs, ea_table["valid_native"], r1_support)
    r1_slot = r1_mapping["slot"]
    r1_p = [float(ea_probs[i, r1_slot]) for i in range(ea_n)]
    r1_valid = [bool(ea_table["valid_native"][i] and ea_table["valid_old"][i]) for i in range(ea_n)]
    carried = {"slot": r1_mapping["slot"], "ready_frames": r1_mapping["ready_frames"], "ready_sample": r1_mapping["ready_sample"]}
    single_c = next(s for s in obs["sources"] if s["source_id"] == "ami_ES2009c")
    single_d = next(s for s in obs["sources"] if s["source_id"] == "ami_ES2009d")
    meet_of_guard = {"R1": "ES2009a", "R2": "EN2009d", "T1": "EN2009d", "BC1": "ES2009a", "SINGLE_ES2009c": "ES2009c", "SINGLE_ES2009d": "ES2009d", "COMBINED_R2T1": "EN2009d"}
    anchor_of = {"R1": "A", "R2": "A", "T1": "A", "BC1": "A", "SINGLE_ES2009c": "A", "SINGLE_ES2009d": "A", "COMBINED_R2T1": "A"}
    for gid in ("R1", "R2", "T1", "BC1", "SINGLE_ES2009c", "SINGLE_ES2009d", "COMBINED_R2T1"):
        gd = gdefs[gid]
        obj = gd["text_object_samples"]
        span = gd["span_samples"]
        meet = meet_of_guard[gid]
        arole = anchor_of[gid]
        if gid in ("R2", "T1", "COMBINED_R2T1"):
            mapping = en_mapping
            table = en_table
            support_n = len(en_support)
            oracle = False
            scalar_sig = ordered_scalar
            oracle_sig = oracle_ordered
            cpu_o = en_cpu_orig
            cpu_r2 = en_cpu_rearm
            cpu_s = en_cpu_scalar
            full_orig = en_orig
            full_rearm = en_rearm
        elif gid == "R1":
            mapping = r1_mapping
            table = ea_table
            support_n = len(r1_support)
            f0s, f1s = int(span[0] // 1280), int((span[1] - 1) // 1280) + 1
            w_frames = list(range(max(f0s, 0), min(f1s, ea_n)))
            t_a = time.perf_counter()
            r1_sc = [float(1.0 - v) for v in r1_p]
            w_o = LH.f0_fire_events(w_frames, table["starts"], table["ends"], r1_valid, table["masked"], table["speech"], r1_sc, table["frontiers"], TAU, CONFIRMATION, False)
            t_b = time.perf_counter()
            w_r = LH.f0_fire_events(w_frames, table["starts"], table["ends"], r1_valid, table["masked"], table["speech"], r1_sc, table["frontiers"], TAU, CONFIRMATION, True)
            t_c = time.perf_counter()
            scalar_sig, _, _ = scalar_ordered_events(w_frames, table["starts"], table["ends"], r1_valid, table["masked"], table["speech"], r1_p, table["frontiers"], mapping["ready_sample"])
            oracle_sig = [{"boundary": int(e["boundary"]), "emit": int(e["emit"]), "frontier": int(e["frontier"]), "frame": int(e["frame"]), "op": "split", "state": "UNKNOWN", "kind": "oracle-held-f0-rearm"} for e in w_r]
            full_orig, full_rearm, cpu_o, cpu_r2, cpu_s = w_o, w_r, t_b - t_a, t_c - t_b, 0.0
            oracle = False
        elif gid == "BC1":
            mapping = carried
            support_n = 0
            oracle = True
            table = ea_table
            full_orig, full_rearm, scalar_sig, oracle_sig, cpu_o, cpu_r2, cpu_s = [], [], [], [], 0.0, 0.0, 0.0
        else:
            sentry = single_c if gid == "SINGLE_ES2009c" else single_d
            sprobs = probs_by_sid[sentry["source_id"]]
            stable = LH.build_native_table(sentry, cache["sources"][sentry["source_id"]])
            sn = sentry["valid_native_frames"]
            ms = gd["map_scope_samples"]
            s0, s1 = int(ms[0] // 1280), int((ms[1] - 1) // 1280) + 1
            s0, s1 = max(s0, 0), min(s1, sn)
            sup = LH.anchor_support_frames(meet, arole, s0, s1)
            mp = LH.map_anchor_slot(sprobs, stable["valid_native"], sup)
            mapping = mp
            support_n = len(sup)
            slot = mp["slot"]
            sp_anchor = [float(sprobs[i, slot]) for i in range(sn)]
            vv = [bool(stable["valid_native"][i] and stable["valid_old"][i]) for i in range(sn)]
            f0s, f1s = int(span[0] // 1280), int((span[1] - 1) // 1280) + 1
            w_frames = list(range(max(f0s, 0), min(f1s, sn)))
            t_a = time.perf_counter()
            scw = [float(1.0 - v) for v in sp_anchor]
            w_o = LH.f0_fire_events(w_frames, stable["starts"], stable["ends"], vv, stable["masked"], stable["speech"], scw, stable["frontiers"], TAU, CONFIRMATION, False)
            t_b = time.perf_counter()
            w_r = LH.f0_fire_events(w_frames, stable["starts"], stable["ends"], vv, stable["masked"], stable["speech"], scw, stable["frontiers"], TAU, CONFIRMATION, True)
            t_c = time.perf_counter()
            scalar_sig, _, _ = scalar_ordered_events(w_frames, stable["starts"], stable["ends"], vv, stable["masked"], stable["speech"], sp_anchor, stable["frontiers"], mapping["ready_sample"])
            oracle_sig = [{"boundary": int(e["boundary"]), "emit": int(e["emit"]), "frontier": int(e["frontier"]), "frame": int(e["frame"]), "op": "split", "state": "UNKNOWN", "kind": "oracle-held-f0-rearm"} for e in w_r]
            full_orig, full_rearm, cpu_o, cpu_r2, cpu_s = w_o, w_r, t_b - t_a, t_c - t_b, 0.0
            table = stable
        def to_current_reqs(sig, cpu):
            reqs = []
            for e in sig:
                if e["op"] == "split":
                    reqs.append({"boundary": e["boundary"], "emit": e["emit"], "frontier": e["frontier"], "kind": e["kind"], "op": "split", "state": "UNKNOWN"})
                else:
                    reqs.append({"boundary": e["boundary"], "emit": e["emit"], "frontier": e["frontier"], "kind": e["kind"], "op": "set_state", "state": "A"})
            return reqs
        def to_counter_reqs(sig):
            reqs = []
            for e in sig:
                if gid in ("R2", "T1", "COMBINED_R2T1") and e["kind"] == "oracle-return":
                    reqs.append({"boundary": e["boundary"], "emit": e["emit"], "frontier": e["frontier"], "kind": e["kind"], "op": "set_state", "state": "A", "avail_source": e["emit"]})
                elif gid in ("R2", "T1", "COMBINED_R2T1") and e["kind"] == "scalar-return":
                    reqs.append({"boundary": e["boundary"], "emit": e["emit"], "frontier": e["frontier"], "kind": e["kind"], "op": "set_state", "state": "A", "avail_source": e["emit"]})
                elif e["op"] == "split":
                    reqs.append({"boundary": e["boundary"], "emit": e.get("emit"), "frontier": e.get("frontier"), "kind": e["kind"], "op": "split", "state": "UNKNOWN", "avail_source": e.get("emit")})
                else:
                    reqs.append({"boundary": e["boundary"], "emit": e.get("emit"), "frontier": e.get("frontier"), "kind": e["kind"], "op": "set_state", "state": "A", "avail_source": e.get("emit")})
            return reqs
        orig_sig = [{"boundary": int(e["boundary"]), "emit": int(e["emit"]), "frontier": int(e["frontier"]), "frame": int(e["frame"]), "op": "split", "state": "UNKNOWN", "kind": "f0-single-fire"} for e in (full_orig[:1] if full_orig else [])]
        rearm_sig = [{"boundary": int(e["boundary"]), "emit": int(e["emit"]), "frontier": int(e["frontier"]), "frame": int(e["frame"]), "op": "split", "state": "UNKNOWN", "kind": "f0-rearm"} for e in full_rearm]
        if gid in ("R2", "T1", "COMBINED_R2T1"):
            scalar_sig_full = scalar_sig
            oracle_sig_full = oracle_sig
            full4_sig_full = ordered_full4
        else:
            scalar_sig_full = scalar_sig
            oracle_sig_full = oracle_sig
            full4_sig_full = scalar_sig
        orig_cur = to_current_reqs(orig_sig, cpu_o)
        rearm_cur = to_current_reqs(rearm_sig, cpu_r2)
        scalar_cur = to_current_reqs(scalar_sig_full, cpu_s)
        oracle_cur = to_current_reqs(oracle_sig_full, cpu_s)
        full4_cur = to_current_reqs(full4_sig_full, cpu_s)
        none_applied, none_hist = receive_current_guard([], obj)
        orig_applied, orig_hist = receive_current_guard(orig_cur, obj)
        rearm_applied, rearm_hist = receive_current_guard(rearm_cur, obj)
        scalar_cur_applied, scalar_cur_hist = receive_current_guard(scalar_cur, obj)
        oracle_cur_applied, oracle_cur_hist = receive_current_guard(oracle_cur, obj)
        full4_cur_applied, full4_cur_hist = receive_current_guard(full4_cur, obj)
        def init_state_for(sig):
            return "A"
        scalar_cf_reqs = to_counter_reqs(scalar_sig_full)
        oracle_cf_reqs = to_counter_reqs(oracle_sig_full)
        full4_cf_reqs = to_counter_reqs(full4_sig_full)
        scalar_init = init_state_for(scalar_sig_full)
        oracle_init = init_state_for(oracle_sig_full)
        scalar_cf_res, scalar_cf_hist = receive_counterfactual_guard(scalar_cf_reqs, obj, scalar_init)
        oracle_cf_res, oracle_cf_hist = receive_counterfactual_guard(oracle_cf_reqs, obj, oracle_init)
        full4_init = "A"
        full4_cf_res, full4_cf_hist = receive_counterfactual_guard(full4_cf_reqs, obj, full4_init)
        words = LH.gt_words_in_span(meet, span[0] / 16000.0, span[1] / 16000.0)
        gt_proxy = [{"id": w["id"], "text": w["text"], "side": "left" if w["role"] == arole else "right", "in_span": True, "start": w["start"], "end": w["end"]} for w in words]
        def proxy_single(b):
            matched = []
            for w in gt_proxy:
                matched.append({"gt": w, "group_idx": -1, "group_word": w["text"], "group_end_src": w["end"], "dt_samples": 0})
            al = {"matched": matched, "unmatched": [], "mixed": [], "region_group_idxs": [], "unresolved_group_idxs": []}
            return ds_partition_ownership(al, b)
        def states_single(b):
            al = {"matched": [{"gt": w, "group_idx": -1, "group_word": w["text"], "group_end_src": w["end"], "dt_samples": 0} for w in gt_proxy], "unmatched": [], "mixed": [], "region_group_idxs": [], "unresolved_group_idxs": []}
            by = {}
            part = ds_partition_ownership(al, b)
            amb = {m["gt"]["id"] for m in part.get("ambiguous", [])}
            wid = {m["gt"]["id"] for m in part.get("wrong_definite", [])}
            for w in gt_proxy:
                if w["id"] in amb:
                    by[w["id"]] = "uncertain"
                elif w["id"] in wid:
                    by[w["id"]] = "wrong"
                else:
                    by[w["id"]] = "correct"
            return by, part
        p_none = proxy_single(none_applied["boundary"] if none_applied else None)
        p_orig = proxy_single(orig_applied["boundary"] if orig_applied else None)
        p_rearm = proxy_single(rearm_applied["boundary"] if rearm_applied else None)
        p_scur = proxy_single(scalar_cur_applied["boundary"] if scalar_cur_applied else None)
        p_ocur = proxy_single(oracle_cur_applied["boundary"] if oracle_cur_applied else None)
        s_none, _ = states_single(none_applied["boundary"] if none_applied else None)
        s_orig, _ = states_single(orig_applied["boundary"] if orig_applied else None)
        s_rearm, _ = states_single(rearm_applied["boundary"] if rearm_applied else None)
        s_scur, _ = states_single(scalar_cur_applied["boundary"] if scalar_cur_applied else None)
        s_ocur, _ = states_single(oracle_cur_applied["boundary"] if oracle_cur_applied else None)
        scalar_bounds = [{"boundary": bb, "state": st["state"]} for bb, st in zip(scalar_cf_res["applied_bounds"], scalar_cf_res["spans"])]
        oracle_bounds = [{"boundary": bb, "state": st["state"]} for bb, st in zip(oracle_cf_res["applied_bounds"], oracle_cf_res["spans"])]
        scalar_spans = return_spans_of(scalar_sig_full)
        oracle_spans = return_spans_of(oracle_sig_full)
        s_scf, p_scf = counterfactual_states_proxy(gt_proxy, scalar_init, scalar_bounds, scalar_spans)
        s_ocf, p_ocf = counterfactual_states_proxy(gt_proxy, oracle_init, oracle_bounds, oracle_spans)
        full4_bounds = [{"boundary": bb, "state": st["state"]} for bb, st in zip(full4_cf_res["applied_bounds"], full4_cf_res["spans"])]
        full4_spans = return_spans_of(full4_sig_full)
        s_fcf, p_fcf = counterfactual_states_proxy(gt_proxy, full4_init, full4_bounds, full4_spans)
        p_fcur, s_fcur = proxy_single(full4_cur_applied["boundary"] if full4_cur_applied else None), states_single(full4_cur_applied["boundary"] if full4_cur_applied else None)[0]
        _od = {"scalar_spans": return_spans_of(scalar_sig_full), "oracle_spans": return_spans_of(oracle_sig_full), "full4_spans": return_spans_of(full4_sig_full), "scalar_overlap": overlap_diagnostic(gt_proxy, return_spans_of(scalar_sig_full)), "oracle_overlap": overlap_diagnostic(gt_proxy, return_spans_of(oracle_sig_full)), "full4_overlap": overlap_diagnostic(gt_proxy, return_spans_of(full4_sig_full))}
        tr_ro_n = paired_transitions(gt_proxy, s_none, s_rearm)
        tr_ro_o = paired_transitions(gt_proxy, s_orig, s_rearm)
        tr_o_n = paired_transitions(gt_proxy, s_none, s_orig)
        tr_scur_n = paired_transitions(gt_proxy, s_none, s_scur)
        tr_scf_n = paired_transitions(gt_proxy, s_none, s_scf)
        tr_ocur_n = paired_transitions(gt_proxy, s_none, s_ocur)
        tr_ocf_n = paired_transitions(gt_proxy, s_none, s_ocf)
        tr_scf_orig = paired_transitions(gt_proxy, s_orig, s_scf)
        tr_ocf_orig = paired_transitions(gt_proxy, s_orig, s_ocf)
        tr_scf_rearm = paired_transitions(gt_proxy, s_rearm, s_scf)
        tr_ocf_rearm = paired_transitions(gt_proxy, s_rearm, s_ocf)
        tr_fcur_n = paired_transitions(gt_proxy, s_none, s_fcur)
        tr_fcf_n = paired_transitions(gt_proxy, s_none, s_fcf)
        tr_fcf_orig = paired_transitions(gt_proxy, s_orig, s_fcf)
        tr_fcf_rearm = paired_transitions(gt_proxy, s_rearm, s_fcf)
        out[gid] = {"id": gid, "text_object_samples": obj, "span_samples": span, "map_scope_samples": gd["map_scope_samples"], "anchor_role": arole, "mapping": mapping, "mapping_support_n": support_n, "oracle_diagnostic": bool(oracle), "full_epoch_single_events": full_orig, "full_epoch_rearm_events": full_rearm, "scalar_ordered": scalar_sig_full, "oracle_ordered": oracle_sig_full, "decision_cpu_s": {"f0_original": cpu_o, "f0_rearm": cpu_r2, "scalar": cpu_s}, "arms": {"none": {"requests": none_hist, "applied_boundary": none_applied["boundary"] if none_applied else None, "partition": p_none}, "f0_original_current": {"requests": orig_hist, "applied_boundary": orig_applied["boundary"] if orig_applied else None, "partition": p_orig}, "f0_rearm_current": {"requests": rearm_hist, "applied_boundary": rearm_applied["boundary"] if rearm_applied else None, "partition": p_rearm}, "s_scalar_current": {"requests": scalar_cur_hist, "applied_boundary": scalar_cur_applied["boundary"] if scalar_cur_applied else None, "partition": p_scur}, "s_oracle_current": {"requests": oracle_cur_hist, "applied_boundary": oracle_cur_applied["boundary"] if oracle_cur_applied else None, "partition": p_ocur}, "s_scalar_counterfactual": {"requests": scalar_cf_hist, "initial_state": scalar_init, "applied_bounds": scalar_cf_res["applied_bounds"], "final_state": scalar_cf_res["final_state"], "partition": p_scf}, "s_oracle_counterfactual": {"requests": oracle_cf_hist, "initial_state": oracle_init, "applied_bounds": oracle_cf_res["applied_bounds"], "final_state": oracle_cf_res["final_state"], "partition": p_ocf}, "s_full4_current": {"requests": full4_cur_hist, "applied_boundary": full4_cur_applied["boundary"] if full4_cur_applied else None, "partition": p_fcur}, "s_full4_counterfactual": {"requests": full4_cf_hist, "initial_state": full4_init, "applied_bounds": full4_cf_res["applied_bounds"], "final_state": full4_cf_res["final_state"], "partition": p_fcf}, "overlap_diagnostic": _od, "full4_ordered": full4_sig_full}, "paired": {"rearm_vs_none": tr_ro_n, "rearm_vs_orig": tr_ro_o, "orig_vs_none": tr_o_n, "scalar_current_vs_none": tr_scur_n, "scalar_cf_vs_none": tr_scf_n, "oracle_current_vs_none": tr_ocur_n, "oracle_cf_vs_none": tr_ocf_n, "scalar_cf_vs_orig": tr_scf_orig, "oracle_cf_vs_orig": tr_ocf_orig, "scalar_cf_vs_rearm": tr_scf_rearm, "oracle_cf_vs_rearm": tr_ocf_rearm, "full4_current_vs_none": tr_fcur_n, "full4_cf_vs_none": tr_fcf_n, "full4_cf_vs_orig": tr_fcf_orig, "full4_cf_vs_rearm": tr_fcf_rearm}, "n_words": len(gt_proxy)}
    guard_wall = time.perf_counter() - t0
    epoch_record = {"full_epoch_samples": ep, "single_events": en_orig, "rearm_events": en_rearm, "scalar_ordered": ordered_scalar, "full4_ordered": ordered_full4, "oracle_ordered": oracle_ordered, "oracle_emit": oracle_emit, "oracle_frontier": oracle_front, "oracle_first_valid_frame": oracle_first_valid, "mask_gap": mask_gap, "mask_delay": mask_delay, "scalar_delay": scalar_delay, "decision_cpu_s": {"single": en_cpu_orig, "rearm": en_cpu_rearm, "scalar": en_cpu_scalar}, "old_374400_check": old_valid_check, "mapping": en_mapping, "mapping_support_n": len(en_support)}
    return out, epoch_record, guard_wall
def smoke_checks():
    traces = []
    a1, h1 = receive_current_guard([{"boundary": 672000, "emit": 684896, "frontier": 684896, "kind": "f0-single-fire", "op": "split", "state": "UNKNOWN"}, {"boundary": 733440, "emit": 746336, "frontier": 746336, "kind": "f0-rearm", "op": "split", "state": "UNKNOWN"}], [701760, 755520])
    traces.append({"name": "outscope-firstwin", "pass": bool(a1 is not None and a1["boundary"] == 733440 and h1[0]["outcome"] == "invalid_scope" and h1[1]["outcome"] == "applied")})
    a2, h2 = receive_current_guard([{"boundary": 672000, "emit": 684896, "frontier": 684896, "kind": "f0-single-fire", "op": "split", "state": "UNKNOWN"}, {"boundary": 733440, "emit": 746336, "frontier": 746336, "kind": "f0-rearm", "op": "split", "state": "UNKNOWN"}], [670592, 755520])
    traces.append({"name": "combined-firstwins", "pass": bool(a2 is not None and a2["boundary"] == 672000 and h2[1]["outcome"] == "already_separated")})
    r1, rh1 = receive_current_guard([{"boundary": 672000, "emit": 684896, "frontier": 684896, "kind": "scalar-change", "op": "split", "state": "UNKNOWN"}, {"boundary": 680320, "emit": 700256, "frontier": 700256, "kind": "oracle-return", "op": "set_state", "state": "A"}], [670592, 701312])
    traces.append({"name": "current-rejects-return", "pass": bool(r1 is not None and r1["boundary"] == 672000 and rh1[1]["outcome"] == "already_separated")})
    r2, rh2 = receive_current_guard([{"boundary": 680320, "emit": 700256, "frontier": 700256, "kind": "oracle-return", "op": "set_state", "state": "A"}], [670592, 701312])
    traces.append({"name": "current-rejects-lone-return", "pass": bool(r2 is None and rh2[0]["outcome"] == "unsupported_operation")})
    c1, ch1 = receive_counterfactual_guard([{"boundary": 672000, "emit": 684896, "frontier": 684896, "kind": "scalar-change", "op": "split", "state": "UNKNOWN", "avail_source": 684896}, {"boundary": 680320, "emit": 700256, "frontier": 700256, "kind": "oracle-return", "op": "set_state", "state": "A", "avail_source": 700256}, {"boundary": 733440, "emit": 746336, "frontier": 746336, "kind": "oracle-change", "op": "split", "state": "UNKNOWN", "avail_source": 746336}], [670592, 701312], "A")
    traces.append({"name": "counterfactual-r2-two-bounds", "pass": bool(c1["applied_bounds"] == [672000, 680320] and c1["final_state"] == "A")})
    c2, ch2 = receive_counterfactual_guard([{"boundary": 680320, "emit": 700256, "frontier": 700256, "kind": "oracle-return", "op": "set_state", "state": "A", "avail_source": 700256}], [701760, 755520], "A")
    traces.append({"name": "counterfactual-t1-outscope-initial", "pass": bool(c2["applied_bounds"] == [] and c2["initial_state"] == "A")})
    fake_base = {"w1": "wrong", "w2": "wrong", "w3": "correct"}
    fake_new = {"w1": "wrong", "w2": "correct", "w3": "wrong"}
    fake_gt = [{"id": "w1", "side": "right"}, {"id": "w2", "side": "right"}, {"id": "w3", "side": "left"}]
    tr = paired_transitions(fake_gt, fake_base, fake_new)
    traces.append({"name": "netmask-new-wrong", "pass": bool(tr["counts"]["new_wrong"] == 1 and tr["counts"]["fixed"] == 1)})
    base2 = {"w1": "correct", "w2": "wrong"}
    new2 = {"w1": "uncertain", "w2": "wrong"}
    gt2 = [{"id": "w1", "side": "left"}, {"id": "w2", "side": "right"}]
    tr2 = paired_transitions(gt2, base2, new2)
    traces.append({"name": "pure-unknown-vs-right", "pass": bool(tr2["counts"]["new_uncertain_pure"] == 1 and tr2["counts"]["new_uncertain_right"] == 0)})
    obs_path = PHASEA / "observations" / "OBSERVATIONS.json"
    obs = json.loads(obs_path.read_text(encoding="utf-8"))
    cache = json.loads((PHASEA / "old_grid_cache.json").read_text(encoding="utf-8"))
    probs_by_sid = {}
    for s in obs["sources"]:
        ff = s["feature_files"]["probs"]
        probs_by_sid[s["source_id"]] = load_np(ROOT / ff["path"], ff["shape"])
    guards, epoch, _ = run_guards(obs, cache, probs_by_sid)
    traces.append({"name": "fresh-epoch-single", "pass": bool([e["boundary"] for e in epoch["single_events"]] == [672000] and [e["emit"] for e in epoch["single_events"]] == [684896])})
    traces.append({"name": "fresh-epoch-rearm", "pass": bool([e["boundary"] for e in epoch["rearm_events"]] == [672000, 733440] and [e["emit"] for e in epoch["rearm_events"]] == [684896, 746336])})
    traces.append({"name": "scalar-ordered-three", "pass": bool([e["boundary"] for e in epoch["scalar_ordered"]] == [672000, 702720, 733440])})
    traces.append({"name": "oracle-ordered-three", "pass": bool([e["boundary"] for e in epoch["oracle_ordered"]] == [672000, 680320, 733440] and epoch["oracle_emit"] == 700256)})
    traces.append({"name": "old-fire-not-fresh", "pass": bool(epoch["old_374400_check"]["valid_old"] is False and epoch["old_374400_check"]["reason"] == "no-old-support")})
    traces.append({"name": "r2-same-first", "pass": bool(guards["R2"]["arms"]["f0_original_current"]["applied_boundary"] == 672000 and guards["R2"]["arms"]["f0_rearm_current"]["applied_boundary"] == 672000)})
    traces.append({"name": "t1-orig-scoped-none", "pass": bool(guards["T1"]["arms"]["f0_original_current"]["applied_boundary"] is None and guards["T1"]["arms"]["f0_rearm_current"]["applied_boundary"] == 733440 and guards["T1"]["arms"]["none"]["partition"]["error_interval"] == [11, 11])})
    traces.append({"name": "r2-cf-bounds-deterministic", "pass": bool(guards["R2"]["arms"]["s_oracle_counterfactual"]["applied_bounds"] == [672000, 680320] and guards["R2"]["arms"]["s_scalar_counterfactual"]["applied_bounds"] == [672000])})
    traces.append({"name": "states-only-A-UNKNOWN", "pass": bool(all(s in ("A", "UNKNOWN") for g in guards for a in ("s_scalar_counterfactual", "s_oracle_counterfactual", "s_full4_counterfactual") for s in [guards[g]["arms"][a]["initial_state"], guards[g]["arms"][a]["final_state"]]))})
    traces.append({"name": "full4-subset-deterministic", "pass": bool([e["boundary"] for e in epoch["full4_ordered"]] == [x for x in [e["boundary"] for e in epoch["scalar_ordered"]] if True][:len([e["boundary"] for e in epoch["full4_ordered"]])] and len(epoch["full4_ordered"]) <= len(epoch["scalar_ordered"]))})
    ok = sum(1 for t in traces if t["pass"])
    return traces, ok
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="run", choices=["run", "smoke"])
    a = ap.parse_args()
    if a.cmd == "smoke":
        traces, ok = smoke_checks()
        print(json.dumps({"traces": traces, "pass": [ok, len(traces)]}))
        return 0 if ok == len(traces) else 1
    t_all = time.perf_counter()
    errs = verify_inputs()
    if errs:
        print(json.dumps({"error": "input-hash-mismatch", "details": errs}))
        return 2
    obs_path = PHASEA / "observations" / "OBSERVATIONS.json"
    obs = json.loads(obs_path.read_text(encoding="utf-8"))
    cache = json.loads((PHASEA / "old_grid_cache.json").read_text(encoding="utf-8"))
    probs_by_sid = {}
    for s in obs["sources"]:
        ff = s["feature_files"]["probs"]
        probs_by_sid[s["source_id"]] = load_np(ROOT / ff["path"], ff["shape"])
    cases = {}
    for case in CASES:
        cases[case] = run_positive(case, obs, cache, probs_by_sid)
    guards, epoch, guard_wall = run_guards(obs, cache, probs_by_sid)
    checks = []
    c = cases["NP1"]
    checks.append(["NP1-rearm-applied-19808000", c["arms_current"]["f0_rearm"]["applied_boundary"] == 19808000])
    checks.append(["NP1-rearm-interval-00", c["arms_current"]["f0_rearm"]["ownership_8"]["error_interval"] == [0, 0]])
    checks.append(["NP1-scalar-no-new-vs-rearm", c["paired_transitions"]["s_scalar_vs_f0_original"]["counts"]["new_wrong"] == 0])
    checks.append(["NP1-mapping-pre-event", c["mapping"] is not None and c["mapping"]["ready_sample"] < PHASEA_FREEZE["cases"]["NP1"]["boundary_samples"]])
    c2 = cases["NP2"]
    checks.append(["NP2-rearm-preserved-00", c2["arms_current"]["f0_rearm"]["ownership_8"]["error_interval"] == [0, 0]])
    checks.append(["NP2-scalar-no-new-vs-orig", c2["paired_transitions"]["s_scalar_vs_f0_original"]["counts"]["new_wrong"] == 0])
    c3 = cases["NP3"]
    checks.append(["NP3-rearm-within-01", c3["arms_current"]["f0_rearm"]["ownership_8"]["error_interval"] in ([0, 0], [0, 1])])
    checks.append(["NP3-scalar-no-new-vs-orig", c3["paired_transitions"]["s_scalar_vs_f0_original"]["counts"]["new_wrong"] == 0])
    for case in CASES:
        cc = cases[case]
        checks.append([case + "-conserved", bool(cc["conservation"]["conserved"] and cc["conservation"]["text_equal"])])
        checks.append([case + "-controls", bool(cc["conservation_controls"]["drop_detected"] and cc["conservation_controls"]["dup_detected"] and cc["conservation_controls"]["text_detected"])])
        checks.append([case + "-none-no-request", cc["arms_current"]["none"]["requests"] == []])
    checks.append(["epoch-single-672000", [e["boundary"] for e in epoch["single_events"]] == [672000]])
    checks.append(["epoch-rearm-672000-733440", [e["boundary"] for e in epoch["rearm_events"]] == [672000, 733440]])
    checks.append(["epoch-scalar-672000-702720-733440", [e["boundary"] for e in epoch["scalar_ordered"]] == [672000, 702720, 733440]])
    checks.append(["epoch-oracle-672000-680320-733440", [e["boundary"] for e in epoch["oracle_ordered"]] == [672000, 680320, 733440]])
    checks.append(["oracle-emit-700256", epoch["oracle_emit"] == 700256])
    checks.append(["old-374400-not-fresh", epoch["old_374400_check"]["valid_old"] is False])
    checks.append(["R2-same-first-no-increment", guards["R2"]["arms"]["f0_original_current"]["applied_boundary"] == 672000 and guards["R2"]["arms"]["f0_rearm_current"]["applied_boundary"] == 672000])
    checks.append(["R2-scalar-current-same-first", guards["R2"]["arms"]["s_scalar_current"]["applied_boundary"] == 672000])
    checks.append(["R2-oracle-current-rejects-return", any(h["outcome"] in ("already_separated", "unsupported_operation") for h in guards["R2"]["arms"]["s_oracle_current"]["requests"] if h["request"]["kind"] == "oracle-return")])
    checks.append(["T1-orig-scoped-none-11", guards["T1"]["arms"]["f0_original_current"]["applied_boundary"] is None and guards["T1"]["arms"]["none"]["partition"]["error_interval"] == [11, 11]])
    checks.append(["COMBINED-firstwins-current", guards["COMBINED_R2T1"]["arms"]["f0_rearm_current"]["applied_boundary"] == 672000 and len([h for h in guards["COMBINED_R2T1"]["arms"]["f0_rearm_current"]["requests"] if h["outcome"] == "already_separated"]) == 1])
    checks.append(["R1-no-fire", guards["R1"]["arms"]["f0_rearm_current"]["applied_boundary"] is None])
    checks.append(["BC1-oracle-no-inspan", guards["BC1"]["oracle_diagnostic"] is True and guards["BC1"]["arms"]["f0_rearm_current"]["applied_boundary"] is None])
    checks.append(["SINGLEs-no-fire", guards["SINGLE_ES2009c"]["arms"]["f0_rearm_current"]["applied_boundary"] is None and guards["SINGLE_ES2009d"]["arms"]["f0_rearm_current"]["applied_boundary"] is None])
    wall = time.perf_counter() - t_all
    ledger = {"freeze_id": FREEZE["freeze_id"], "frozen_at_utc": FREEZE["frozen_at_utc"], "obs_sha256": BARRIER_SHA, "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "cases": cases, "guards": guards, "ref_epoch_EN2009d": epoch, "guard_compute_wall_s": guard_wall, "live_compute_wall_s": wall, "profile_note": "fresh native F0 FP16 Vulkan NO_MUL_MAT_VEC F32_HEAD LOWLATENCY loaded local, no H restore, service costs observed offline virtual stream not live API", "integration_checks": [{"name": k, "pass": bool(v)} for k, v in checks]}
    (EXP / "ledger.json").write_text(json.dumps(ledger, indent=1), encoding="utf-8")
    npass = sum(1 for _, v in checks if v)
    print(json.dumps({"cases": list(cases.keys()), "guards": list(guards.keys()), "checks_pass": [npass, len(checks)], "wall_s": round(wall, 1)}))
    return 0 if npass == len(checks) else 1
if __name__ == "__main__":
    raise SystemExit(main())
