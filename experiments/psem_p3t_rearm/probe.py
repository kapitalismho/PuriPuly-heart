"""P3T fixed F0 excursion rearm probe on frozen native observations."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from datetime import timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
EXP = ROOT / "experiments" / "psem_p3t_rearm"
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
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
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
            errs.append({"path": rel, "error": "missing"})
            continue
        got = sha256_file(p)
        if got != want:
            errs.append({"path": rel, "want": want, "got": got})
    obs_path = PHASEA / "observations" / "OBSERVATIONS.json"
    if sha256_file(obs_path) != BARRIER_SHA:
        errs.append({"path": "observations-bundle", "error": "barrier-sha-mismatch"})
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
def receive_positive(requests, seal):
    applied = None
    history = []
    for req in requests:
        slim = {"boundary": req["boundary"], "emit": req["emit"], "frontier": req["frontier"], "kind": req["kind"]}
        if req["scope"] != "valid":
            history.append({"request": slim, "outcome": req["scope"], "applied_boundary": applied["boundary"] if applied else None})
            continue
        if req["avail"] is None or seal is None:
            history.append({"request": slim, "outcome": "UNKNOWN", "applied_boundary": applied["boundary"] if applied else None})
            continue
        if req["avail"] > seal:
            history.append({"request": slim, "outcome": "too_late", "applied_boundary": applied["boundary"] if applied else None, "availability": req["avail"], "deadline": seal})
            continue
        if applied is None:
            applied = {"boundary": req["boundary"], "availability": req["avail"], "request": slim}
            history.append({"request": slim, "outcome": "applied", "applied_boundary": req["boundary"], "availability": req["avail"], "deadline": seal})
        else:
            history.append({"request": slim, "outcome": "already_separated", "applied_boundary": applied["boundary"]})
    return applied, history
def receive_guard(requests, obj):
    applied = None
    history = []
    for req in requests:
        slim = {"boundary": req["boundary"], "emit": req["emit"], "frontier": req["frontier"], "kind": req["kind"]}
        scope = guard_scope_of(req["boundary"], obj)
        if scope != "valid":
            history.append({"request": slim, "outcome": "invalid_scope", "applied_boundary": applied["boundary"] if applied else None})
            continue
        if applied is None:
            applied = {"boundary": req["boundary"], "emit": req["emit"]}
            history.append({"request": slim, "outcome": "applied", "applied_boundary": req["boundary"]})
        else:
            history.append({"request": slim, "outcome": "already_separated", "applied_boundary": applied["boundary"]})
    return applied, history
def word_states(part):
    states = {}
    amb = {m["gt"]["id"]: m for m in part.get("ambiguous", [])}
    wrong_ids = {m["gt"]["id"]: m for m in part.get("wrong_definite", m if False else [])} if False else {}
    return states
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
    scores = [float(1 - probs[i, slot]) for i in range(n)]
    valid = [bool(table["valid_native"][i] and table["valid_old"][i]) for i in range(n)]
    f0, f1 = int(ep[0] // 1280), int((ep[1] - 1) // 1280) + 1
    ep_frames = list(range(f0, f1))
    t_dec0 = time.perf_counter()
    ev_orig = LH.f0_fire_events(ep_frames, table["starts"], table["ends"], valid, table["masked"], table["speech"], scores, table["frontiers"], TAU, CONFIRMATION, False)
    t_dec1 = time.perf_counter()
    ev_rearm = LH.f0_fire_events(ep_frames, table["starts"], table["ends"], valid, table["masked"], table["speech"], scores, table["frontiers"], TAU, CONFIRMATION, True)
    t_dec2 = time.perf_counter()
    decision_cpu_orig = t_dec1 - t_dec0
    decision_cpu_rearm = t_dec2 - t_dec1
    first = ev_orig[0] if ev_orig else None
    cap = json.loads((STAGE2 / "captures" / (case + ".json")).read_text(encoding="utf-8"))
    init_s = LH.trace_init_s(src_entry)
    sched = LH.build_schedule(src_entry, cap, pay, init_s)
    backlog = sched["zero_finish"]
    ch_of = sched["chunk_of_frame"]
    fin = sched["finish"]
    ctail = sched["tail_invalid"]
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
    f0_avail = None
    f0_scope = "no-event"
    f0_rel = None
    if first is not None:
        f0_avail, f0_ci, f0_rel = event_avail(first["frame"], decision_cpu_orig)
        f0_scope = positive_scope_of(first["emit"], pay)
    if f0_avail is not None:
        f0_avail = f0_avail
    ctrl_frame = (b + CONFIRMATION) // 1280
    ctrl_ci = ch_of[int(ctrl_frame)] if 0 <= int(ctrl_frame) < len(ch_of) else None
    ctrl_base, ctrl_rel = None, None
    if ctrl_ci is not None:
        if fin[ctrl_ci] is None:
            ctrl_rel = "UNKNOWN-support-past-payload"
        else:
            ctrl_base = fin[ctrl_ci]
            ctrl_rel = "measured"
    part_probe = ds_partition_ownership(ds_align_window(gt8, groups, wlo, whi), b)
    recv_cpu = time.perf_counter() - t_cpu
    ctrl_avail = (ctrl_base + recv_cpu) if ctrl_base is not None else None
    rearm_reqs = []
    for e in ev_rearm:
        av, ci, rel = event_avail(e["frame"], decision_cpu_rearm)
        rearm_reqs.append({"boundary": e["boundary"], "emit": e["emit"], "frontier": e["frontier"], "frame": e["frame"], "scope": positive_scope_of(e["emit"], pay), "avail": av, "cpu": decision_cpu_rearm, "kind": "f0-rearm", "release": rel})
    orig_reqs = []
    if first is not None:
        orig_reqs.append({"boundary": first["boundary"], "emit": first["emit"], "frontier": first["frontier"], "frame": first["frame"], "scope": f0_scope, "avail": f0_avail, "cpu": decision_cpu_orig, "kind": "f0-single-fire", "release": f0_rel})
    ctrl_reqs = [{"boundary": b, "emit": b + CONFIRMATION, "frontier": int(ctrl_frame) * 1280, "scope": "valid", "avail": ctrl_avail, "cpu": recv_cpu, "kind": "control-capacity", "release": ctrl_rel}]
    arms = {}
    for arm, reqs in (("none", []), ("f0_original", orig_reqs), ("f0_rearm", rearm_reqs)):
        applied, history = receive_positive(reqs, seal)
        ab = applied["boundary"] if applied else None
        states_none_ref, part_none = partition_states(al8, None)
        cur_part = ds_partition_ownership(ds_align_window(gt8, groups, wlo, whi), ab)
        cur_states, _ = partition_states(ds_align_window(gt8, groups, wlo, whi), ab)
        sens = {}
        for delta in (-SENS, SENS):
            bb = None if ab is None else ab + delta
            sens[str(delta)] = ds_partition_ownership(ds_align_window(gt8, groups, wlo, whi), bb)["error_interval"]
        if ab is None:
            acc_part = None
        else:
            left_groups = [g for g in groups if g.get("end_src") is not None and g["end_src"] <= ab]
            right_groups = [g for g in groups if g.get("end_src") is not None and g["end_src"] > ab]
            acc_part = {"n_left": len(left_groups), "n_right": len(right_groups), "left_text": "".join(g.get("text", "") for g in left_groups), "right_text": "".join(g.get("text", "") for g in right_groups)}
        arms[arm] = {"requests": history, "applied_boundary": ab, "availability": applied["availability"] if applied else None, "deadline": seal, "accepted_partition": acc_part, "ownership_8": cur_part, "ownership_8_sens_pm1280": sens, "n_alignment_matched": len(al8["matched"]), "n_alignment_unmatched": len(al8["unmatched"]), "n_alignment_mixed": len(al8["mixed"])}
    ctrl_applied, ctrl_history = receive_positive(ctrl_reqs, seal)
    ctrl_ab = ctrl_applied["boundary"] if ctrl_applied else None
    ctrl_part = ds_partition_ownership(ds_align_window(gt8, groups, wlo, whi), ctrl_ab)
    reference = {"requests": ctrl_history, "applied_boundary": ctrl_ab, "availability": ctrl_applied["availability"] if ctrl_applied else None, "deadline": seal, "ownership_8": ctrl_part, "label": "carried-GT-capacity-REFERENCE-nonbinding"}
    order = ["none", "f0_original", "f0_rearm"]
    parts = {a: ds_partition_ownership(ds_align_window(gt8, groups, wlo, whi), arms[a]["applied_boundary"]) for a in order}
    states = {}
    for a in order:
        s, _ = partition_states(ds_align_window(gt8, groups, wlo, whi), arms[a]["applied_boundary"])
        states[a] = s
    gt_ids = [{"id": m["gt"]["id"], "side": m["gt"]["side"]} for m in al8["matched"]]
    deltas = {}
    for pair in (("f0_rearm", "none"), ("f0_rearm", "f0_original"), ("f0_original", "none")):
        new, base = pair
        nb = parts[new]["error_interval"]
        bb = parts[base]["error_interval"]
        deltas[new + "_vs_" + base] = {"definite": nb[0] - bb[0], "pessimistic": nb[1] - bb[1], "new": nb, "base": bb}
    transitions = {}
    for pair in (("f0_rearm", "none"), ("f0_rearm", "f0_original"), ("f0_original", "none")):
        new, base = pair
        transitions[new + "_vs_" + base] = paired_transitions(gt_ids, states[base], states[new])
    rframe = mapping["ready_frames"][1]
    rci = ch_of[rframe] if 0 <= rframe < len(ch_of) else None
    ready_session = fin[rci] if (rci is not None and fin[rci] is not None) else None
    acc_toks = cap.get("accepted", {}).get("tokens", [])
    cons = LH.conservation_check(acc_toks, groups)
    ctrls = LH.throwaway_controls(acc_toks, groups)
    n_applied = sum(1 for h in arms["f0_rearm"]["requests"] if h["outcome"] == "applied")
    n_red = sum(1 for h in arms["f0_rearm"]["requests"] if h["outcome"] == "already_separated")
    frag = {"applied_new_bounds": n_applied, "redundant_repeated": n_red, "n_requests": len(arms["f0_rearm"]["requests"])}
    qa = LH.qa_merge_for_case(case, groups)
    full_detail = {"n_gt": len(full_gt), "n_matched": len(full_al["matched"]), "n_unmatched": len(full_al["unmatched"]), "n_mixed": len(full_al["mixed"])}
    return {"case": case, "source": sid, "mapping": mapping, "mapping_scope": [mlo, mhi], "mapping_support_n": len(support), "f0_original_events": ev_orig, "f0_rearm_events": ev_rearm, "decision_cpu_s": {"f0_original": decision_cpu_orig, "f0_rearm": decision_cpu_rearm, "receiver": recv_cpu}, "warm_backlog_at_capture_entry_s": backlog, "ready_session_clock": ready_session, "arms": arms, "reference_control_capacity": reference, "deltas_fixed_cohort": deltas, "paired_transitions": transitions, "full_payload_alignment": full_detail, "conservation": cons, "conservation_controls": ctrls, "fragmentation": frag, "qa_merge": qa}
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
    en_scores = [float(1 - en_probs[i, en_slot]) for i in range(en_n)]
    en_valid = [bool(en_table["valid_native"][i] and en_table["valid_old"][i]) for i in range(en_n)]
    ep = FREEZE["ref_epoch_EN2009d"]["full_epoch_samples"]
    f0e, f1e = int(ep[0] // 1280), int((ep[1] - 1) // 1280) + 1
    ep_frames = list(range(max(f0e, 0), min(f1e, en_n)))
    t_dec0 = time.perf_counter()
    en_orig = LH.f0_fire_events(ep_frames, en_table["starts"], en_table["ends"], en_valid, en_table["masked"], en_table["speech"], en_scores, en_table["frontiers"], TAU, CONFIRMATION, False)
    t_dec1 = time.perf_counter()
    en_rearm = LH.f0_fire_events(ep_frames, en_table["starts"], en_table["ends"], en_valid, en_table["masked"], en_table["speech"], en_scores, en_table["frontiers"], TAU, CONFIRMATION, True)
    t_dec2 = time.perf_counter()
    en_cpu_orig = t_dec1 - t_dec0
    en_cpu_rearm = t_dec2 - t_dec1
    old_frame = 374400 // 1280
    old_valid_check = {"frame": int(old_frame), "valid_old": bool(en_table["valid_old"][old_frame]), "masked": bool(en_table["masked"][old_frame]), "speech": bool(en_table["speech"][old_frame]), "reason": str(en_table["old_reason"][old_frame])}
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
    r1_scores = [float(1 - ea_probs[i, r1_slot]) for i in range(ea_n)]
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
            full_orig = en_orig
            full_rearm = en_rearm
            cpu_o = en_cpu_orig
            cpu_r = en_cpu_rearm
            oracle = False
            support_n = len(en_support)
        elif gid == "R1":
            mapping = r1_mapping
            table = ea_table
            support_n = len(r1_support)
            f0s, f1s = int(span[0] // 1280), int((span[1] - 1) // 1280) + 1
            t_a = time.perf_counter()
            w_o = LH.f0_fire_events(list(range(max(f0s, 0), min(f1s, ea_n))), table["starts"], table["ends"], r1_valid, table["masked"], table["speech"], r1_scores, table["frontiers"], TAU, CONFIRMATION, False)
            t_b = time.perf_counter()
            w_r = LH.f0_fire_events(list(range(max(f0s, 0), min(f1s, ea_n))), table["starts"], table["ends"], r1_valid, table["masked"], table["speech"], r1_scores, table["frontiers"], TAU, CONFIRMATION, True)
            t_c = time.perf_counter()
            full_orig, full_rearm, cpu_o, cpu_r = w_o, w_r, t_b - t_a, t_c - t_b
            oracle = False
        elif gid == "BC1":
            mapping = carried
            support_n = 0
            oracle = True
            f0s, f1s = int(gd["map_scope_samples"][0] // 1280), int((span[1] - 1) // 1280) + 1
            bc_scores = r1_scores
            bc_valid = r1_valid
            table = ea_table
            t_a = time.perf_counter()
            w_o = LH.f0_fire_events(list(range(max(f0s, 0), min(f1s, ea_n))), table["starts"], table["ends"], bc_valid, table["masked"], table["speech"], bc_scores, table["frontiers"], TAU, CONFIRMATION, False)
            t_b = time.perf_counter()
            w_r = LH.f0_fire_events(list(range(max(f0s, 0), min(f1s, ea_n))), table["starts"], table["ends"], bc_valid, table["masked"], table["speech"], bc_scores, table["frontiers"], TAU, CONFIRMATION, True)
            t_c = time.perf_counter()
            full_orig, full_rearm, cpu_o, cpu_r = w_o, w_r, t_b - t_a, t_c - t_b
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
            sc = [float(1 - sprobs[i, slot]) for i in range(sn)]
            vv = [bool(stable["valid_native"][i] and stable["valid_old"][i]) for i in range(sn)]
            f0s, f1s = int(span[0] // 1280), int((span[1] - 1) // 1280) + 1
            t_a = time.perf_counter()
            w_o = LH.f0_fire_events(list(range(max(f0s, 0), min(f1s, sn))), stable["starts"], stable["ends"], vv, stable["masked"], stable["speech"], sc, stable["frontiers"], TAU, CONFIRMATION, False)
            t_b = time.perf_counter()
            w_r = LH.f0_fire_events(list(range(max(f0s, 0), min(f1s, sn))), stable["starts"], stable["ends"], vv, stable["masked"], stable["speech"], sc, stable["frontiers"], TAU, CONFIRMATION, True)
            t_c = time.perf_counter()
            full_orig, full_rearm, cpu_o, cpu_r = w_o, w_r, t_b - t_a, t_c - t_b
            table = stable
        orig_reqs = [{"boundary": e["boundary"], "emit": e["emit"], "frontier": e["frontier"], "kind": "f0-single-fire"} for e in (full_orig[:1] if full_orig else [])]
        rearm_reqs = [{"boundary": e["boundary"], "emit": e["emit"], "frontier": e["frontier"], "kind": "f0-rearm"} for e in full_rearm]
        none_applied, none_hist = receive_guard([], obj)
        orig_applied, orig_hist = receive_guard(orig_reqs, obj)
        rearm_applied, rearm_hist = receive_guard(rearm_reqs, obj)
        words = LH.gt_words_in_span(meet, span[0] / 16000.0, span[1] / 16000.0)
        gt_proxy = [{"id": w["id"], "text": w["text"], "side": "left" if w["role"] == arole else "right", "in_span": True, "start": w["start"], "end": w["end"]} for w in words]
        def proxy_of(b):
            matched = []
            for w in gt_proxy:
                matched.append({"gt": w, "group_idx": -1, "group_word": w["text"], "group_end_src": w["end"], "dt_samples": 0})
            al = {"matched": matched, "unmatched": [], "mixed": [], "region_group_idxs": [], "unresolved_group_idxs": []}
            return ds_partition_ownership(al, b)
        p_none = proxy_of(none_applied["boundary"] if none_applied else None)
        p_orig = proxy_of(orig_applied["boundary"] if orig_applied else None)
        p_rearm = proxy_of(rearm_applied["boundary"] if rearm_applied else None)
        s_none, _ = partition_states({"matched": [{"gt": w, "group_idx": -1, "group_word": w["text"], "group_end_src": w["end"], "dt_samples": 0} for w in gt_proxy], "unmatched": [], "mixed": [], "region_group_idxs": [], "unresolved_group_idxs": []}, none_applied["boundary"] if none_applied else None)
        s_orig, _ = partition_states({"matched": [{"gt": w, "group_idx": -1, "group_word": w["text"], "group_end_src": w["end"], "dt_samples": 0} for w in gt_proxy], "unmatched": [], "mixed": [], "region_group_idxs": [], "unresolved_group_idxs": []}, orig_applied["boundary"] if orig_applied else None)
        s_rearm, _ = partition_states({"matched": [{"gt": w, "group_idx": -1, "group_word": w["text"], "group_end_src": w["end"], "dt_samples": 0} for w in gt_proxy], "unmatched": [], "mixed": [], "region_group_idxs": [], "unresolved_group_idxs": []}, rearm_applied["boundary"] if rearm_applied else None)
        tr_ro_n = paired_transitions(gt_proxy, s_none, s_rearm)
        tr_ro_o = paired_transitions(gt_proxy, s_orig, s_rearm)
        tr_o_n = paired_transitions(gt_proxy, s_none, s_orig)
        out[gid] = {"id": gid, "text_object_samples": obj, "span_samples": span, "map_scope_samples": gd["map_scope_samples"], "anchor_role": arole, "mapping": mapping, "mapping_support_n": support_n, "oracle_diagnostic": bool(oracle), "full_epoch_single_events": full_orig, "full_epoch_rearm_events": full_rearm, "decision_cpu_s": {"f0_original": cpu_o, "f0_rearm": cpu_r}, "arms": {"none": {"requests": none_hist, "applied_boundary": none_applied["boundary"] if none_applied else None, "partition": p_none}, "f0_original": {"requests": orig_hist, "applied_boundary": orig_applied["boundary"] if orig_applied else None, "partition": p_orig}, "f0_rearm": {"requests": rearm_hist, "applied_boundary": rearm_applied["boundary"] if rearm_applied else None, "partition": p_rearm}}, "paired_transitions": {"rearm_vs_none": tr_ro_n, "rearm_vs_orig": tr_ro_o, "orig_vs_none": tr_o_n}, "actual_asr_quality": "UNKNOWN-no-ASR-on-guard", "deadline": None, "availability_scope": "SOURCE-samples-from-zero-plus-measured-cpu", "proxy_seal_samples": span[1], "n_span_gt_words": len(words)}
    guard_wall = time.perf_counter() - t0
    epoch_record = {"full_epoch_samples": ep, "single_events": en_orig, "rearm_events": en_rearm, "decision_cpu_s": {"single": en_cpu_orig, "rearm": en_cpu_rearm}, "old_374400_check": old_valid_check, "mapping": en_mapping, "mapping_support_n": len(en_support)}
    return out, epoch_record, guard_wall
def smoke_checks():
    traces = []
    a1, h1 = receive_guard([{"boundary": 672000, "emit": 684896, "frontier": 684896, "kind": "f0-single-fire"}, {"boundary": 733440, "emit": 746336, "frontier": 746336, "kind": "f0-rearm"}], [701760, 755520])
    traces.append({"name": "outscope-firstwin", "pass": bool(a1 is not None and a1["boundary"] == 733440 and h1[0]["outcome"] == "invalid_scope" and h1[1]["outcome"] == "applied")})
    a2, h2 = receive_guard([{"boundary": 672000, "emit": 684896, "frontier": 684896, "kind": "f0-single-fire"}, {"boundary": 733440, "emit": 746336, "frontier": 746336, "kind": "f0-rearm"}], [670592, 755520])
    traces.append({"name": "combined-firstwins", "pass": bool(a2 is not None and a2["boundary"] == 672000 and h2[1]["outcome"] == "already_separated")})
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
    traces.append({"name": "old-fire-not-fresh", "pass": bool(epoch["old_374400_check"]["valid_old"] is False and epoch["old_374400_check"]["reason"] == "no-old-support")})
    traces.append({"name": "r2-same-first", "pass": bool(guards["R2"]["arms"]["f0_original"]["applied_boundary"] == 672000 and guards["R2"]["arms"]["f0_rearm"]["applied_boundary"] == 672000)})
    traces.append({"name": "t1-orig-scoped-none", "pass": bool(guards["T1"]["arms"]["f0_original"]["applied_boundary"] is None and guards["T1"]["arms"]["f0_rearm"]["applied_boundary"] == 733440 and guards["T1"]["arms"]["none"]["partition"]["error_interval"] == [11, 11])})
    traces.append({"name": "t1-rearm-partition", "pass": bool(guards["T1"]["arms"]["f0_rearm"]["partition"]["error_interval"] == [6, 7])})
    traces.append({"name": "r2-partition", "pass": bool(guards["R2"]["arms"]["f0_rearm"]["partition"]["error_interval"] == [3, 5] and guards["R2"]["arms"]["none"]["partition"]["error_interval"] == [6, 6])})
    traces.append({"name": "combined-diagnostic", "pass": bool(guards["COMBINED_R2T1"]["arms"]["f0_rearm"]["applied_boundary"] == 672000 and len([h for h in guards["COMBINED_R2T1"]["arms"]["f0_rearm"]["requests"] if h["outcome"] == "already_separated"]) == 1)})
    traces.append({"name": "no-new-rearm-vs-orig-r2t1", "pass": bool(guards["R2"]["paired_transitions"]["rearm_vs_orig"]["counts"]["new_wrong"] == 0 and guards["T1"]["paired_transitions"]["rearm_vs_orig"]["counts"]["new_wrong"] == 0 and guards["R2"]["paired_transitions"]["rearm_vs_orig"]["counts"]["new_uncertain_pure"] == 0 and guards["T1"]["paired_transitions"]["rearm_vs_orig"]["counts"]["new_uncertain_pure"] == 0)})
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
    checks.append(["NP1-rearm-applied-19808000", c["arms"]["f0_rearm"]["applied_boundary"] == 19808000])
    checks.append(["NP1-rearm-interval-00", c["arms"]["f0_rearm"]["ownership_8"]["error_interval"] == [0, 0]])
    checks.append(["NP1-gain-strict-vs-none", c["deltas_fixed_cohort"]["f0_rearm_vs_none"]["definite"] < 0 and c["deltas_fixed_cohort"]["f0_rearm_vs_none"]["pessimistic"] < 0])
    checks.append(["NP1-gain-strict-vs-orig", c["deltas_fixed_cohort"]["f0_rearm_vs_f0_original"]["definite"] < 0 and c["deltas_fixed_cohort"]["f0_rearm_vs_f0_original"]["pessimistic"] < 0])
    checks.append(["NP1-no-new-wrong-vs-orig", c["paired_transitions"]["f0_rearm_vs_f0_original"]["counts"]["new_wrong"] == 0])
    checks.append(["NP1-mapping-pre-event", c["mapping"] is not None and c["mapping"]["ready_sample"] < PHASEA_FREEZE["cases"]["NP1"]["boundary_samples"]])
    c2 = cases["NP2"]
    checks.append(["NP2-rearm-preserved-00", c2["arms"]["f0_rearm"]["ownership_8"]["error_interval"] == [0, 0]])
    checks.append(["NP2-no-loss-vs-orig", c2["deltas_fixed_cohort"]["f0_rearm_vs_f0_original"]["definite"] == 0 and c2["deltas_fixed_cohort"]["f0_rearm_vs_f0_original"]["pessimistic"] == 0])
    checks.append(["NP2-no-new-wrong-vs-orig", c2["paired_transitions"]["f0_rearm_vs_f0_original"]["counts"]["new_wrong"] == 0])
    c3 = cases["NP3"]
    checks.append(["NP3-rearm-within-01", c3["arms"]["f0_rearm"]["ownership_8"]["error_interval"] in ([0, 0], [0, 1])])
    checks.append(["NP3-no-additional-unc-vs-orig", c3["arms"]["f0_rearm"]["ownership_8"]["error_interval"][1] <= c3["arms"]["f0_original"]["ownership_8"]["error_interval"][1]])
    checks.append(["NP3-no-new-wrong-vs-orig", c3["paired_transitions"]["f0_rearm_vs_f0_original"]["counts"]["new_wrong"] == 0])
    for case in CASES:
        cc = cases[case]
        checks.append([case + "-conserved", bool(cc["conservation"]["conserved"] and cc["conservation"]["text_equal"])])
        checks.append([case + "-controls", bool(cc["conservation_controls"]["drop_detected"] and cc["conservation_controls"]["dup_detected"] and cc["conservation_controls"]["text_detected"])])
        checks.append([case + "-none-no-request", cc["arms"]["none"]["requests"] == []])
        checks.append([case + "-rearm-measured-cpu", cc["decision_cpu_s"]["f0_rearm"] > 0])
    checks.append(["epoch-single-672000", [e["boundary"] for e in epoch["single_events"]] == [672000]])
    checks.append(["epoch-rearm-672000-733440", [e["boundary"] for e in epoch["rearm_events"]] == [672000, 733440]])
    checks.append(["old-374400-not-fresh", epoch["old_374400_check"]["valid_old"] is False])
    checks.append(["R2-same-first-no-increment", guards["R2"]["arms"]["f0_original"]["applied_boundary"] == 672000 and guards["R2"]["arms"]["f0_rearm"]["applied_boundary"] == 672000])
    checks.append(["R2-no-new-rearm-vs-orig", guards["R2"]["paired_transitions"]["rearm_vs_orig"]["counts"]["new_wrong"] == 0 and guards["R2"]["paired_transitions"]["rearm_vs_orig"]["counts"]["new_uncertain_pure"] == 0])
    checks.append(["T1-orig-scoped-none-11", guards["T1"]["arms"]["f0_original"]["applied_boundary"] is None and guards["T1"]["arms"]["none"]["partition"]["error_interval"] == [11, 11]])
    checks.append(["T1-rearm-67", guards["T1"]["arms"]["f0_rearm"]["partition"]["error_interval"] == [6, 7]])
    checks.append(["T1-no-new-pure-vs-orig", guards["T1"]["paired_transitions"]["rearm_vs_orig"]["counts"]["new_wrong"] == 0 and guards["T1"]["paired_transitions"]["rearm_vs_orig"]["counts"]["new_uncertain_pure"] == 0])
    checks.append(["COMBINED-firstwins", guards["COMBINED_R2T1"]["arms"]["f0_rearm"]["applied_boundary"] == 672000 and len([h for h in guards["COMBINED_R2T1"]["arms"]["f0_rearm"]["requests"] if h["outcome"] == "already_separated"]) == 1])
    checks.append(["R1-no-fire", guards["R1"]["arms"]["f0_rearm"]["applied_boundary"] is None])
    checks.append(["BC1-oracle-no-inspan", guards["BC1"]["oracle_diagnostic"] is True and guards["BC1"]["arms"]["f0_rearm"]["applied_boundary"] is None])
    checks.append(["SINGLEs-no-fire", guards["SINGLE_ES2009c"]["arms"]["f0_rearm"]["applied_boundary"] is None and guards["SINGLE_ES2009d"]["arms"]["f0_rearm"]["applied_boundary"] is None])
    wall = time.perf_counter() - t_all
    ledger = {"freeze_id": FREEZE["freeze_id"], "frozen_at_utc": FREEZE["frozen_at_utc"], "obs_sha256": BARRIER_SHA, "generated_at_utc": datetime.now(timezone.utc).isoformat(), "cases": cases, "guards": guards, "ref_epoch_EN2009d": epoch, "guard_compute_wall_s": guard_wall, "live_compute_wall_s": wall, "profile_note": "fresh native F0 FP16 Vulkan NO_MUL_MAT_VEC F32_HEAD LOWLATENCY loaded local, no H restore, service costs observed offline virtual stream not live API", "integration_checks": [{"name": k, "pass": bool(v)} for k, v in checks]}
    (EXP / "per_case_ledger.json").write_text(json.dumps(ledger, indent=1), encoding="utf-8")
    npass = sum(1 for _, v in checks if v)
    print(json.dumps({"cases": list(cases.keys()), "guards": list(guards.keys()), "checks_pass": [npass, len(checks)], "wall_s": round(wall, 1)}))
    return 0 if npass == len(checks) else 1
if __name__ == "__main__":
    raise SystemExit(main())
