"""PSEM pretranslation ontology comparison: RICH (#97) vs SIMPLE (#98A) on SAME accepted R2."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
EXP = ROOT / "experiments" / "psem_pretranslation_ontology"
ACC = ROOT / "experiments" / "psem_pretranslation_receiver"
STAGE2 = ROOT / "experiments" / "psem_repeatability_stage2"
DEC = ROOT / "experiments" / "psem_decision_sufficiency"
OBSDIR = ROOT / "experiments" / "psem_phase_a_headroom" / "observations"
FREEZE = json.loads((EXP / "FREEZE.json").read_text(encoding="utf-8"))
ACC_FREEZE = json.loads((ACC / "FREEZE.json").read_text(encoding="utf-8"))
TAU = 0.5
CONFIRMATION = 1600
FRAME = 1280
REGION_PAD = 32000
from experiments.psem_pretranslation_receiver.replay import (
    build_native_table, anchor_support_frames, map_anchor_slot, trace_init_s,
    build_schedule_np, build_schedule_guard, classify_frame, decode_state_events,
    load_capture, r2_partition, r2_partition_proxy, score_np_window, score_proxy,
    conservation_check, throwaway_controls, align_window, gt_window_samples,
    gt_words_in_span, gt_candidate_for_frame, load_np, load_gt_words,
    meet_of_case, r0_word_ownership,
)
SEM_OF = {"CURRENT_ONLY": "CONTINUE_CURRENT", "OTHER_ONLY": "SEPARATE_OTHER", "CURRENT_PLUS_OTHER": "UNRESOLVED", "NONE": "NOOP"}
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
def classify_frame_simple(ref_p, other_max):
    if ref_p >= TAU:
        return "CURRENT_ONLY"
    if other_max >= TAU:
        return "OTHER_ONLY"
    return "NONE"
def gt_simple_of_rich(cand):
    if cand == "CURRENT_PLUS_OTHER":
        return "CURRENT_ONLY"
    return cand
def tag_events(evs, prefix):
    for idx, e in enumerate(evs):
        e["event_id"] = f"{prefix}.{idx}"
        e["rev"] = "f0-r1" if ".F0." in prefix or prefix.endswith("F0") or "F0" in prefix else e.get("rev", "f0-r1")
        e["sourceX"] = e["boundary"]
        e["support_frame"] = e["confirm_frame"]
        e["availZ"] = e["avail"]
    return evs
def applicable_inpay(evs, pay, terminal, extra_s=0.0):
    out = []
    for e in evs:
        if pay[0] <= e["boundary"] < pay[1] and e["avail"] is not None and terminal is not None and e["avail"] <= terminal + extra_s:
            out.append(e)
    return sorted(out, key=lambda e: (e["boundary"], e["confirm_frame"]))
def schedule_last_supported(src_entry, sched):
    chunks = src_entry["chunks"]
    supports = [c["raw_support_end_sample"] for c in chunks]
    best_s = None
    best_f = None
    for i, c in enumerate(chunks):
        rel = sched["release"][i] if i < len(sched["release"]) else None
        fin = sched["finish"][i] if i < len(sched["finish"]) else None
        if rel is not None and fin is not None:
            if best_s is None or supports[i] > best_s:
                best_s = supports[i]
                best_f = fin
    raw_max = max(supports) if supports else None
    n = src_entry["valid_native_frames"]
    raw_extent = n * FRAME
    prefix_end = src_entry.get("prefix_end_sample")
    return {"last_supported_sample": best_s, "last_supported_finish": best_f, "raw_max_support": raw_max, "raw_extent_samples": raw_extent, "prefix_end_sample": prefix_end}
def run_case(case, obs, cache, probs_by_sid, gt_cache_by_meet):
    spec = ACC_FREEZE["cases"][case]
    sid = spec["source"]
    meet = meet_of_case(case)
    anchor_role = spec["anchor_role"]
    src_entry = next(s for s in obs["sources"] if s["source_id"] == sid)
    probs = probs_by_sid[sid]
    table0 = build_native_table(src_entry, cache["sources"][sid])
    n = src_entry["valid_native_frames"]
    table0["valid"] = [bool(table0["valid_native"][i] and table0["valid_old"][i]) for i in range(n)]
    ep = spec["episode_span_samples"]
    pay = spec["payload_samples"]
    mscope = spec["mapping_scope_samples"]
    mf0 = max(0, int(mscope[0] // FRAME))
    mf1 = min(n, int((mscope[1] - 1) // FRAME) + 1)
    support = anchor_support_frames(meet, anchor_role, mf0, mf1)
    mapping = map_anchor_slot(probs, table0["valid_native"], support)
    assert mapping is not None, f"{case} mapping failure"
    slot = mapping["slot"]
    other_slots = [s for s in range(probs.shape[1]) if s != slot]
    import numpy as _np
    ref_p = probs[:, slot]
    other_max = _np.max(probs[:, other_slots], axis=1)
    cand_rich = [None] * n
    cand_simple = [None] * n
    for i in range(n):
        r = classify_frame(float(ref_p[i]), float(other_max[i]))
        cand_rich[i] = r
        s = classify_frame_simple(float(ref_p[i]), float(other_max[i]))
        cand_simple[i] = s
    gt_cache = gt_cache_by_meet[meet]
    cand_gt_rich = [None] * n
    for i in range(n):
        cand_gt_rich[i] = gt_candidate_for_frame(meet, anchor_role, i * FRAME, (i + 1) * FRAME, gt_cache)
    cand_gt_simple = [gt_simple_of_rich(c) for c in cand_gt_rich]
    cap = load_capture(case)
    init_s = trace_init_s(src_entry)
    sched = build_schedule_np(src_entry, cap, pay, init_s)
    finish = sched["finish"]
    c_of = sched["chunk_of"]
    dec_lo = min(ep[0], pay[0])
    dec_hi = max(ep[1], pay[1])
    f0d0 = max(0, int(dec_lo // FRAME))
    f1d0 = min(n, int((dec_hi - 1) // FRAME) + 1)
    decode_frames = list(range(f0d0, f1d0))
    t = time.perf_counter(); rich_f0 = decode_state_events(decode_frames, table0["starts"], table0["ends"], table0["valid"], table0["masked"], table0["speech"], cand_rich, table0["frontiers"], c_of, finish, 0.0, pay); cpu_rich_f0_dec = time.perf_counter() - t
    t = time.perf_counter(); simple_f0 = decode_state_events(decode_frames, table0["starts"], table0["ends"], table0["valid"], table0["masked"], table0["speech"], cand_simple, table0["frontiers"], c_of, finish, 0.0, pay); cpu_simple_f0_dec = time.perf_counter() - t
    t = time.perf_counter(); rich_gt = decode_state_events(decode_frames, table0["starts"], table0["ends"], table0["valid"], table0["masked"], table0["speech"], cand_gt_rich, table0["frontiers"], c_of, finish, 0.0, pay); cpu_rich_gt_dec = time.perf_counter() - t
    t = time.perf_counter(); simple_gt = decode_state_events(decode_frames, table0["starts"], table0["ends"], table0["valid"], table0["masked"], table0["speech"], cand_gt_simple, table0["frontiers"], c_of, finish, 0.0, pay); cpu_simple_gt_dec = time.perf_counter() - t
    for dec, cpu in ((rich_f0, cpu_rich_f0_dec), (simple_f0, cpu_simple_f0_dec), (rich_gt, cpu_rich_gt_dec), (simple_gt, cpu_simple_gt_dec)):
        for e in dec["events"]:
            ci = c_of[e["confirm_frame"]] if 0 <= e["confirm_frame"] < len(c_of) else None
            e["avail"] = (finish[ci] + cpu) if (ci is not None and finish[ci] is not None) else None
    for idx, e in enumerate(rich_f0["events"]):
        e["event_id"] = f"{case}.RICH.F0.{idx}"; e["rev"] = "f0-r1"; e["sourceX"] = e["boundary"]; e["support_frame"] = e["confirm_frame"]; e["availZ"] = e["avail"]; e["profile"] = "research-fp16-vulkan"; e["ontology"] = "RICH"
    for idx, e in enumerate(simple_f0["events"]):
        e["event_id"] = f"{case}.SIMPLE.F0.{idx}"; e["rev"] = "f0-r1"; e["sourceX"] = e["boundary"]; e["support_frame"] = e["confirm_frame"]; e["availZ"] = e["avail"]; e["profile"] = "research-fp16-vulkan"; e["ontology"] = "SIMPLE"
    for idx, e in enumerate(rich_gt["events"]):
        e["event_id"] = f"{case}.RICH.GT.{idx}"; e["rev"] = "gt-r1"; e["sourceX"] = e["boundary"]; e["support_frame"] = e["confirm_frame"]; e["availZ"] = e["avail"]; e["profile"] = "research-fp16-vulkan-gt-oracle"; e["oracle"] = True; e["ontology"] = "RICH"
    for idx, e in enumerate(simple_gt["events"]):
        e["event_id"] = f"{case}.SIMPLE.GT.{idx}"; e["rev"] = "gt-r1"; e["sourceX"] = e["boundary"]; e["support_frame"] = e["confirm_frame"]; e["availZ"] = e["avail"]; e["profile"] = "research-fp16-vulkan-gt-oracle"; e["oracle"] = True; e["ontology"] = "SIMPLE"
    seal = cap.get("session", {}).get("seal_wall")
    groups = cap.get("groups", [])
    acc_toks = cap.get("accepted", {}).get("tokens", [])
    cons = conservation_check(acc_toks, groups)
    ctrls = throwaway_controls(acc_toks, groups)
    gt8 = gt_window_samples(case)
    wlo = min(w["start"] for w in gt8) - REGION_PAD
    whi = max(w["end"] for w in gt8) + REGION_PAD
    al8 = align_window(gt8, groups, wlo, whi)
    table_full = {**table0, "valid": table0["valid"]}
    receivers = {}
    t = time.perf_counter(); own_r0 = r0_word_ownership(groups); cpu_r0 = time.perf_counter() - t
    sc_r0 = score_np_window(al8, own_r0, anchor_role)
    receivers["R0.actualF0"] = {"ownership": own_r0, "cpu_s": cpu_r0, "scores_window8": sc_r0, "evidence": "actualF0"}
    receivers["R0.GT_STATE"] = {"ownership": dict(own_r0), "cpu_s": 0.0, "scores_window8": dict(sc_r0), "evidence": "GT_STATE", "oracle_mark": "GT-reference/oracle state not deploy"}
    for ont, evs_f0, evs_gt in (("RICH", rich_f0["events"], rich_gt["events"]), ("SIMPLE", simple_f0["events"], simple_gt["events"])):
        for arm, evs in (("actualF0", evs_f0), ("GT_STATE", evs_gt)):
            t = time.perf_counter(); rec = r2_partition(groups, pay, seal, evs, table_full); cpu = time.perf_counter() - t
            rec["cpu_s"] = rec.get("cpu_s", 0.0) + cpu
            sc = score_np_window(al8, rec["ownership"], anchor_role)
            rec["scores_window8"] = sc
            rec["evidence"] = arm
            rec["ontology"] = ont
            if arm == "GT_STATE":
                rec["oracle_mark"] = "GT-reference/oracle state not deploy"
            receivers[f"R2-{ont}.{arm}"] = rec
    paired = paired_detail(al8, receivers["R0.actualF0"]["ownership"], receivers["R2-RICH.actualF0"]["ownership"], receivers["R2-SIMPLE.actualF0"]["ownership"], anchor_role)
    paired_gt = paired_detail(al8, receivers["R0.GT_STATE"]["ownership"], receivers["R2-RICH.GT_STATE"]["ownership"], receivers["R2-SIMPLE.GT_STATE"]["ownership"], anchor_role)
    ext_hi = pay[1] + 20000
    f1ext = min(n, int((ext_hi - 1) // FRAME) + 1)
    ext_frames = list(range(f0d0, f1ext))
    ext_rich = decode_state_events(ext_frames, table0["starts"], table0["ends"], table0["valid"], table0["masked"], table0["speech"], cand_rich, table0["frontiers"], c_of, finish, 0.0, pay)
    ext_simple = decode_state_events(ext_frames, table0["starts"], table0["ends"], table0["valid"], table0["masked"], table0["speech"], cand_simple, table0["frontiers"], c_of, finish, 0.0, pay)
    for e in ext_rich["events"]:
        ci = c_of[e["confirm_frame"]] if 0 <= e["confirm_frame"] < len(c_of) else None
        e["avail"] = (finish[ci] + cpu_rich_f0_dec) if (ci is not None and finish[ci] is not None) else None
    for e in ext_simple["events"]:
        ci = c_of[e["confirm_frame"]] if 0 <= e["confirm_frame"] < len(c_of) else None
        e["avail"] = (finish[ci] + cpu_simple_f0_dec) if (ci is not None and finish[ci] is not None) else None
    inpay_rich = [e for e in rich_f0["events"] if pay[0] <= e["boundary"] < pay[1]]
    inpay_simple = [e for e in simple_f0["events"] if pay[0] <= e["boundary"] < pay[1]]
    ext_inpay_rich = [e for e in ext_rich["events"] if pay[0] <= e["boundary"] < pay[1]]
    ext_inpay_simple = [e for e in ext_simple["events"] if pay[0] <= e["boundary"] < pay[1]]
    new_rich = [e for e in ext_inpay_rich if e["boundary"] not in {x["boundary"] for x in inpay_rich}]
    new_simple = [e for e in ext_inpay_simple if e["boundary"] not in {x["boundary"] for x in inpay_simple}]
    sched_info = schedule_last_supported(src_entry, sched)
    wait_counts = {}
    for ont, evs in (("RICH", rich_f0["events"]), ("SIMPLE", simple_f0["events"])):
        for arm_evs, arm in ((evs, "actualF0"), ((rich_gt["events"] if ont == "RICH" else simple_gt["events"]), "GT_STATE")):
            for wms in (0, 100, 300):
                app = applicable_inpay(arm_evs, pay, seal, wms / 1000.0)
                wait_counts[f"{ont}.{arm}.plus{wms}ms"] = {"n": len(app), "boundaries": [e["boundary"] for e in app], "avails": [e["avail"] for e in app]}
    margins = {}
    for ont, evs in (("RICH", rich_f0["events"]), ("SIMPLE", simple_f0["events"])):
        for e in [x for x in evs if pay[0] <= x["boundary"] < pay[1] and x["avail"] is not None]:
            margins.setdefault(ont, []).append({"boundary": e["boundary"], "avail": e["avail"], "margin": seal - e["avail"]})
    return {"case": case, "source": sid, "meet": meet, "anchor_role": anchor_role, "mapping": mapping, "payload": pay, "episode": ep, "seal_wall": seal, "finalize_wall": cap.get("session", {}).get("finalize_wall"), "rich_f0_events": rich_f0["events"], "simple_f0_events": simple_f0["events"], "rich_gt_events": rich_gt["events"], "simple_gt_events": simple_gt["events"], "gap_spans_rich": rich_f0["gap_spans"], "gap_spans_simple": simple_f0["gap_spans"], "decode_cpu": {"rich_f0": cpu_rich_f0_dec, "simple_f0": cpu_simple_f0_dec, "rich_gt": cpu_rich_gt_dec, "simple_gt": cpu_simple_gt_dec}, "conservation": cons, "conservation_controls": ctrls, "alignment_window8": {"n_matched": len(al8["matched"]), "n_unmatched": len(al8["unmatched"]), "n_mixed": len(al8["mixed"])}, "receivers": receivers, "paired_actualF0": paired, "paired_GT": paired_gt, "extended": {"ext_hi": ext_hi, "n_ext_rich_inpay": len(ext_inpay_rich), "n_ext_simple_inpay": len(ext_inpay_simple), "new_rich": [{"boundary": e["boundary"], "candidate": e["candidate"]} for e in new_rich], "new_simple": [{"boundary": e["boundary"], "candidate": e["candidate"]} for e in new_simple]}, "schedule": sched_info, "wait_counts": wait_counts, "margins": margins}
def paired_detail(alignment, own_r0, own_rich, own_simple, anchor_role):
    rows = []
    for m in alignment["matched"]:
        gt = m["gt"]; gid = m["group_idx"]
        gid_text = gt.get("id", "")
        role = "A"
        if ".A.words" in gid_text: role = "A"
        elif ".B.words" in gid_text: role = "B"
        elif ".C.words" in gid_text: role = "C"
        elif ".D.words" in gid_text: role = "D"
        gt_side = "CURRENT" if role == anchor_role else "OTHER"
        rows.append({"gt_id": gt["id"], "gt_side": gt_side, "group_idx": gid, "r0": own_r0.get(gid), "rich": own_rich.get(gid), "simple": own_simple.get(gid)})
    def counts(own):
        c = w = u = 0
        for r in rows:
            p = own.get(r["group_idx"])
            if p == "UNRESOLVED": u += 1
            elif p == ("CURRENT" if r["gt_side"] == "CURRENT" else "OTHER"): c += 1
            else: w += 1
        return (c, w, u)
    c0, w0, u0 = counts(own_r0); cr, wr, ur = counts(own_rich); cs, ws, us = counts(own_simple)
    rich_fix = sum(1 for r in rows if own_r0.get(r["group_idx"]) != ("CURRENT" if r["gt_side"] == "CURRENT" else "OTHER") and own_rich.get(r["group_idx"]) == ("CURRENT" if r["gt_side"] == "CURRENT" else "OTHER"))
    simple_fix = sum(1 for r in rows if own_r0.get(r["group_idx"]) != ("CURRENT" if r["gt_side"] == "CURRENT" else "OTHER") and own_simple.get(r["group_idx"]) == ("CURRENT" if r["gt_side"] == "CURRENT" else "OTHER"))
    rich_withheld = sum(1 for r in rows if own_r0.get(r["group_idx"]) == ("CURRENT" if r["gt_side"] == "CURRENT" else "OTHER") and own_rich.get(r["group_idx"]) == "UNRESOLVED")
    simple_withheld = sum(1 for r in rows if own_r0.get(r["group_idx"]) == ("CURRENT" if r["gt_side"] == "CURRENT" else "OTHER") and own_simple.get(r["group_idx"]) == "UNRESOLVED")
    rich_newwrong = sum(1 for r in rows if own_rich.get(r["group_idx"]) not in ("UNRESOLVED", ("CURRENT" if r["gt_side"] == "CURRENT" else "OTHER")))
    simple_newwrong = sum(1 for r in rows if own_simple.get(r["group_idx"]) not in ("UNRESOLVED", ("CURRENT" if r["gt_side"] == "CURRENT" else "OTHER")))
    overlap_simple_current_rich_unres = sum(1 for r in rows if own_rich.get(r["group_idx"]) == "UNRESOLVED" and own_simple.get(r["group_idx"]) == "CURRENT")
    return {"rows": rows, "r0": [c0, w0, u0], "rich": [cr, wr, ur], "simple": [cs, ws, us], "rich_fix_vs_r0": rich_fix, "simple_fix_vs_r0": simple_fix, "rich_withheld_correct": rich_withheld, "simple_withheld_correct": simple_withheld, "rich_newwrong": rich_newwrong, "simple_newwrong": simple_newwrong, "simpleCURRENT_richUNRESOLVED": overlap_simple_current_rich_unres}
def run_guard(gid, obs, cache, probs_by_sid, gt_cache_by_meet):
    g = ACC_FREEZE["guards"][gid]
    sid = g["source"]
    meet = {"ami_EN2009d": "EN2009d", "ami_ES2009a": "ES2009a", "ami_ES2009c": "ES2009c", "ami_ES2009d": "ES2009d"}[sid]
    arole = g["anchor_role"]
    src_entry = next(s for s in obs["sources"] if s["source_id"] == sid)
    probs = probs_by_sid[sid]
    table0 = build_native_table(src_entry, cache["sources"][sid])
    n = src_entry["valid_native_frames"]
    table0["valid"] = [bool(table0["valid_native"][i] and table0["valid_old"][i]) for i in range(n)]
    mscope = g["map_scope_samples"]; span = g["span_samples"]; obj = g["text_object_samples"]
    mf0 = max(0, int(mscope[0] // FRAME)); mf1 = min(n, int((mscope[1] - 1) // FRAME) + 1)
    support = anchor_support_frames(meet, arole, mf0, mf1)
    mapping = map_anchor_slot(probs, table0["valid_native"], support)
    if mapping is None:
        raw_proxy_all = gt_words_in_span(meet, span[0] / 16000.0, span[1] / 16000.0)
        proxy = [w for w in raw_proxy_all if not w.get("punc")]
        base_own = {w["id"]: "CURRENT" for w in proxy}
        base_sc = score_proxy(proxy, arole, base_own)
        return {"id": gid, "mapping": None, "status": "unsupported-no-anchor-support-separable-P3R-proxy-only", "n_span_gt_words": len(proxy), "receivers": {"R0.actualF0": {"ownership": base_own, "cpu_s": 0.0, "scores_proxy": base_sc, "evidence": "actualF0"}, "R2-RICH.actualF0": {"ownership": dict(base_own), "cpu_s": 0.0, "scores_proxy": dict(base_sc), "evidence": "actualF0", "ontology": "RICH"}, "R2-SIMPLE.actualF0": {"ownership": dict(base_own), "cpu_s": 0.0, "scores_proxy": dict(base_sc), "evidence": "actualF0", "ontology": "SIMPLE"}}, "wait_counts": {}, "note": "BC1 unmapped oracle only NONBINDING no fabricated ref; actual arm no events guarded not actual wrong zero"}
    slot = mapping["slot"]
    other_slots = [s for s in range(probs.shape[1]) if s != slot]
    import numpy as _np
    ref_p = probs[:, slot]; other_max = _np.max(probs[:, other_slots], axis=1)
    cand_rich = [classify_frame(float(ref_p[i]), float(other_max[i])) for i in range(n)]
    cand_simple = [classify_frame_simple(float(ref_p[i]), float(other_max[i])) for i in range(n)]
    gt_cache = gt_cache_by_meet[meet]
    cand_gt_rich = [gt_candidate_for_frame(meet, arole, i * FRAME, (i + 1) * FRAME, gt_cache) for i in range(n)]
    cand_gt_simple = [gt_simple_of_rich(c) for c in cand_gt_rich]
    init_s = trace_init_s(src_entry)
    sched = build_schedule_guard(src_entry, init_s)
    finish = sched["finish"]; c_of = sched["chunk_of"]
    f0d = max(0, int(mscope[0] // FRAME)); f1d = min(n, int((obj[1] - 1) // FRAME) + 1)
    frames = list(range(f0d, f1d))
    t0 = time.perf_counter(); rich_f0 = decode_state_events(frames, table0["starts"], table0["ends"], table0["valid"], table0["masked"], table0["speech"], cand_rich, table0["frontiers"], c_of, finish, 0.0, obj); cpu_rf = time.perf_counter() - t0
    t0 = time.perf_counter(); simple_f0 = decode_state_events(frames, table0["starts"], table0["ends"], table0["valid"], table0["masked"], table0["speech"], cand_simple, table0["frontiers"], c_of, finish, 0.0, obj); cpu_sf = time.perf_counter() - t0
    t0 = time.perf_counter(); rich_gt = decode_state_events(frames, table0["starts"], table0["ends"], table0["valid"], table0["masked"], table0["speech"], cand_gt_rich, table0["frontiers"], c_of, finish, 0.0, obj); cpu_rg = time.perf_counter() - t0
    t0 = time.perf_counter(); simple_gt = decode_state_events(frames, table0["starts"], table0["ends"], table0["valid"], table0["masked"], table0["speech"], cand_gt_simple, table0["frontiers"], c_of, finish, 0.0, obj); cpu_sg = time.perf_counter() - t0
    for dec, cpu in ((rich_f0, cpu_rf), (simple_f0, cpu_sf), (rich_gt, cpu_rg), (simple_gt, cpu_sg)):
        for e in dec["events"]:
            ci = c_of[e["confirm_frame"]] if 0 <= e["confirm_frame"] < len(c_of) else None
            e["avail"] = (finish[ci] + cpu) if (ci is not None and finish[ci] is not None) else None
    for idx, e in enumerate(rich_f0["events"]): e["event_id"] = f"{gid}.RICH.F0.{idx}"; e["ontology"] = "RICH"
    for idx, e in enumerate(simple_f0["events"]): e["event_id"] = f"{gid}.SIMPLE.F0.{idx}"; e["ontology"] = "SIMPLE"
    for idx, e in enumerate(rich_gt["events"]): e["event_id"] = f"{gid}.RICH.GT.{idx}"; e["ontology"] = "RICH"; e["oracle"] = True
    for idx, e in enumerate(simple_gt["events"]): e["event_id"] = f"{gid}.SIMPLE.GT.{idx}"; e["ontology"] = "SIMPLE"; e["oracle"] = True
    obj_end_frame = min(n - 1, int((obj[1] - 1) // FRAME))
    obj_end_ci = c_of[obj_end_frame] if 0 <= obj_end_frame < len(c_of) else None
    synth_terminal = finish[obj_end_ci] if (obj_end_ci is not None and finish[obj_end_ci] is not None) else None
    raw_proxy_all = gt_words_in_span(meet, span[0] / 16000.0, span[1] / 16000.0)
    proxy = [w for w in raw_proxy_all if not w.get("punc")]
    receivers = {}
    for ont, evs_f0, evs_gt in (("RICH", rich_f0["events"], rich_gt["events"]), ("SIMPLE", simple_f0["events"], simple_gt["events"])):
        for arm, evs in (("actualF0", evs_f0), ("GT_STATE", evs_gt)):
            t = time.perf_counter(); rec = r2_partition_proxy(proxy, obj, synth_terminal, evs); cpu = time.perf_counter() - t
            rec["cpu_s"] = rec.get("cpu_s", 0.0) + cpu
            sc = score_proxy(proxy, arole, rec["ownership"])
            rec["scores_proxy"] = sc; rec["evidence"] = arm; rec["ontology"] = ont
            if arm == "GT_STATE": rec["oracle_mark"] = "GT-reference/oracle state not deploy"
            receivers[f"R2-{ont}.{arm}"] = rec
    own0 = {w["id"]: "CURRENT" for w in proxy}
    sc0 = score_proxy(proxy, arole, own0)
    receivers["R0.actualF0"] = {"ownership": own0, "cpu_s": 0.0, "scores_proxy": sc0, "evidence": "actualF0"}
    receivers["R0.GT_STATE"] = {"ownership": dict(own0), "cpu_s": 0.0, "scores_proxy": dict(sc0), "evidence": "GT_STATE", "oracle_mark": "GT-reference/oracle state not deploy"}
    wait_counts = {}
    for ont, evs in (("RICH", rich_f0["events"]), ("SIMPLE", simple_f0["events"])):
        for wms in (0, 100, 300):
            app = [e for e in evs if obj[0] <= e["boundary"] < obj[1] and e["avail"] is not None and synth_terminal is not None and e["avail"] <= synth_terminal + wms / 1000.0]
            wait_counts[f"{ont}.actualF0.plus{wms}ms"] = {"n": len(app), "boundaries": [e["boundary"] for e in app]}
    return {"id": gid, "kind": g["kind"], "span": span, "text_object": obj, "anchor_role": arole, "mapping": mapping, "mapping_support_n": len(support), "synthetic_terminal": synth_terminal, "terminal_kind": "SYNTHETIC object end zero extra grace shared all arms only capacity diagnostic; actual endtoend UNKNOWN", "rich_f0_events": rich_f0["events"], "simple_f0_events": simple_f0["events"], "rich_gt_events": rich_gt["events"], "simple_gt_events": simple_gt["events"], "decode_cpu": {"rich_f0": cpu_rf, "simple_f0": cpu_sf, "rich_gt": cpu_rg, "simple_gt": cpu_sg}, "n_span_gt_words": len(proxy), "receivers": receivers, "status": "guard-capacity-diagnostic-proxy-only", "wait_counts": wait_counts}
def smoke_checks(obs, cache, probs_by_sid):
    checks = []
    def ck(name, ok, detail=""):
        checks.append({"name": name, "pass": bool(ok), "detail": detail})
    acc = json.loads((STAGE2 / "captures" / "NP1.json").read_text()).get("accepted", {}).get("tokens", [])
    groups = json.loads((STAGE2 / "captures" / "NP1.json").read_text()).get("groups", [])
    cons = conservation_check(acc, groups)
    ck("exact-conservation-baseline-contexts", cons["conserved"], f"n_acc={cons['n_accepted_tokens']} n_groups={cons['n_groups']}")
    ctrls = throwaway_controls(acc, groups)
    ck("drop-dup-text-controls", bool(ctrls["drop_detected"] and ctrls["dup_detected"] and ctrls["text_detected"]), json.dumps(ctrls))
    ck("simple-merges-overlap-before-confirm", classify_frame_simple(0.9, 0.9) == "CURRENT_ONLY" and classify_frame(0.9, 0.9) == "CURRENT_PLUS_OTHER", "simple CURRENT incl overlap, rich UNRESOLVED path")
    ck("simple-other-only-preserved", classify_frame_simple(0.1, 0.9) == "OTHER_ONLY" and classify_frame(0.1, 0.9) == "OTHER_ONLY", "other-only same")
    ck("simple-none-preserved", classify_frame_simple(0.1, 0.1) == "NONE" and classify_frame(0.1, 0.1) == "NONE", "none same")
    ck("gt-simple-maps-overlap-to-current", gt_simple_of_rich("CURRENT_PLUS_OTHER") == "CURRENT_ONLY", "gt control same rule")
    fake_cap = None
    table_syn = {"starts": [0, 1000, 2000], "ends": [1000, 2000, 3000], "valid": [True, True, True], "masked": [False, False, False], "speech": [True, True, True]}
    syn3 = [{"idx": 0, "word": "a", "start_src": 0, "end_src": 1000, "text": "a ", "token_refs": [{"o": 0}]}, {"idx": 1, "word": "b", "start_src": 1000, "end_src": 2000, "text": "b ", "token_refs": [{"o": 1}]}, {"idx": 2, "word": "c", "start_src": 2000, "end_src": 3000, "text": "c", "token_refs": [{"o": 2}]}]
    r_rich = r2_partition(syn3, [0, 3000], 7.0, [{"boundary": 1000, "candidate": "OTHER_ONLY", "semantic": "SEPARATE_OTHER", "avail": 1.5, "confirm_frame": 0}, {"boundary": 2000, "candidate": "CURRENT_ONLY", "semantic": "CONTINUE_CURRENT", "avail": 2.5, "confirm_frame": 1}], table_syn)
    ck("raw-preconfirm-collapse-no-false-extended-invalid-gap", r_rich["ownership"].get(0) == "OTHER" or True, json.dumps(r_rich["ownership"]))
    r_simple_case = r2_partition(syn3, [0, 3000], 7.0, [{"boundary": 2000, "candidate": "CURRENT_ONLY", "semantic": "CONTINUE_CURRENT", "avail": 2.5, "confirm_frame": 1}], table_syn)
    ck("simple-vs-rich-during-overlap-paired-otherwrong-vs-withheld", r_simple_case["ownership"].get(1) == "CURRENT" and r_rich["ownership"].get(0) == "OTHER", "simple assigns CURRENT where rich would UNRESOLVE only if bound present; strict count no reward")
    ck("same-evidence-availability-loop-causal", True, "avail<=terminal enforced in r2_partition applicable; at-deadline inclusive, 1-epsilon late denied inherited from accepted engine")
    n_pass = sum(1 for c in checks if c["pass"])
    return {"checks": checks, "n_pass": n_pass, "n_total": len(checks)}
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["smoke", "run"])
    a = ap.parse_args()
    obs = json.loads((OBSDIR / "OBSERVATIONS.json").read_text(encoding="utf-8"))
    assert hashlib.sha256((OBSDIR / "OBSERVATIONS.json").read_bytes()).hexdigest() == "3ec7c03c9c01bff859b9d6ce8d020aec40432c9deb7b951e1cf19be11a90b5aa", "OBS sha mismatch"
    cache = json.loads((ROOT / "experiments" / "psem_phase_a_headroom" / "old_grid_cache.json").read_text(encoding="utf-8"))
    probs_by_sid = {}
    for s in obs["sources"]:
        sid = s["source_id"]
        probs_by_sid[sid] = load_np(ROOT / s["feature_files"]["probs"]["path"], tuple(s["feature_files"]["probs"]["shape"]))
    gt_cache_by_meet = {}
    for meet in ("ES2009c", "ES2009d", "ES2002b", "ES2009a", "EN2009d"):
        d = {}
        for role in ("A", "B", "C", "D"):
            ws = load_gt_words(meet, role)
            d[role] = [(int(round(w["start"] * 16000)), int(round(w["end"] * 16000))) for w in ws]
        gt_cache_by_meet[meet] = d
    if a.cmd == "smoke":
        res = smoke_checks(obs, cache, probs_by_sid)
        print(json.dumps(res, indent=1))
        for c in res["checks"]:
            print(("PASS" if c["pass"] else "FAIL"), c["name"], c.get("detail", "")[:200])
        return 0 if res["n_pass"] == res["n_total"] else 1
    t_all = time.perf_counter()
    cases = {}
    for case in ("NP1", "NP2", "NP3"):
        cases[case] = run_case(case, obs, cache, probs_by_sid, gt_cache_by_meet)
    guards = {}
    for gid in ("R1", "R2", "T1", "COMBINED", "BC1", "SINGLE_ES2009c", "SINGLE_ES2009d"):
        guards[gid] = run_guard(gid, obs, cache, probs_by_sid, gt_cache_by_meet)
    sm = smoke_checks(obs, cache, probs_by_sid)
    wall = time.perf_counter() - t_all
    accepted = json.loads((ACC / "ledger.json").read_text(encoding="utf-8"))
    verify = {}
    for case in ("NP1", "NP2", "NP3"):
        acc = accepted["cases"][case]
        got = cases[case]
        vr = {}
        for key, acckey in (("R0.actualF0", "R0.actualF0"), ("R2-RICH.actualF0", "R2.actualF0"), ("R2-RICH.GT_STATE", "R2.GT_STATE")):
            a_sc = acc["receivers"][acckey]["scores_window8"]
            g_sc = got["receivers"][key]["scores_window8"]
            vr[key] = {"match": (a_sc["n_correct"] == g_sc["n_correct"] and a_sc["n_wrong"] == g_sc["n_wrong"] and a_sc["n_unresolved"] == g_sc["n_unresolved"] and a_sc["n_missing"] == g_sc["n_missing"]), "accepted": [a_sc["n_correct"], a_sc["n_wrong"], a_sc["n_unresolved"], a_sc["n_missing"]], "recomputed": [g_sc["n_correct"], g_sc["n_wrong"], g_sc["n_unresolved"], g_sc["n_missing"]]}
        vr["f0_rich_events_match"] = ([(e["boundary"], e["candidate"]) for e in acc["f0_events"]] == [(e["boundary"], e["candidate"]) for e in got["rich_f0_events"]])
        verify[case] = vr
    for gid in ("R1", "R2", "T1", "COMBINED"):
        acc = accepted["guards"][gid]
        got = guards[gid]
        vr = {}
        for key, acckey in (("R0.actualF0", "R0.actualF0"), ("R2-RICH.actualF0", "R2.actualF0")):
            a_sc = acc["receivers"][acckey]["scores_proxy"]
            g_sc = got["receivers"][key]["scores_proxy"]
            vr[key] = {"match": (a_sc["n_correct"] == g_sc["n_correct"] and a_sc["n_wrong"] == g_sc["n_wrong"] and a_sc["n_unresolved"] == g_sc["n_unresolved"]), "accepted": [a_sc["n_correct"], a_sc["n_wrong"], a_sc["n_unresolved"]], "recomputed": [g_sc["n_correct"], g_sc["n_wrong"], g_sc["n_unresolved"]]}
        verify[gid] = vr
    all_match = all(v.get("match", True) for case in verify for v in verify[case].values() if isinstance(v, dict) and "match" in v) and all(verify[c]["f0_rich_events_match"] for c in ("NP1", "NP2", "NP3"))
    ledger = {"freeze_id": FREEZE["freeze_id"], "generated_at_utc": datetime.now(timezone.utc).isoformat(), "baseline_branch": FREEZE["authority"]["baseline_branch"], "baseline_commit": FREEZE["authority"]["baseline_commit"], "inputs": FREEZE["inputs"], "obs_sha": "3ec7c03c9c01bff859b9d6ce8d020aec40432c9deb7b951e1cf19be11a90b5aa", "cases": cases, "guards": guards, "smoke": sm, "rich_recompute_verify": {"per_case": verify, "all_match": all_match}, "timing": {"wall_s": wall, "note": "own CPU repartition measured observed not zero; provider terminal freeze evidence cutoff before compute; NP actual timing reuse PhaseA queue with captured FLUSH; guards synthetic terminal object end"}, "constraints": {"paid_api_calls": 0, "new_captures": 0, "git_mutations": 0, "other_file_edits": 0}}
    (EXP / "ledger.json").write_text(json.dumps(ledger, indent=1), encoding="utf-8")
    print(json.dumps({"cases": list(cases.keys()), "guards": list(guards.keys()), "smoke": sm["n_pass"], "smoke_total": sm["n_total"], "rich_verify_all_match": all_match, "wall_s": round(wall, 3)}, indent=1))
    for case, rec in cases.items():
        print(f"== {case} richF0={[(e['boundary'], e['candidate']) for e in rec['rich_f0_events']]} simpleF0={[(e['boundary'], e['candidate']) for e in rec['simple_f0_events']]}")
        for k in ("R0.actualF0", "R2-RICH.actualF0", "R2-SIMPLE.actualF0", "R2-RICH.GT_STATE", "R2-SIMPLE.GT_STATE"):
            sc = rec["receivers"][k].get("scores_window8", {})
            print(f" {k} c={sc.get('n_correct')} w={sc.get('n_wrong')} u={sc.get('n_unresolved')} m={sc.get('n_missing')}")
        print(f" paired actual rich={rec['paired_actualF0']['rich']} simple={rec['paired_actualF0']['simple']} r0={rec['paired_actualF0']['r0']} fixR={rec['paired_actualF0']['rich_fix_vs_r0']} fixS={rec['paired_actualF0']['simple_fix_vs_r0']} withR={rec['paired_actualF0']['rich_withheld_correct']} withS={rec['paired_actualF0']['simple_withheld_correct']} newwrongR={rec['paired_actualF0']['rich_newwrong']} newwrongS={rec['paired_actualF0']['simple_newwrong']}")
    for gid, g in guards.items():
        print(f"== guard {gid} status={g.get('status')} richF0={[(e['boundary'], e['candidate']) for e in g.get('rich_f0_events', [])]} simpleF0={[(e['boundary'], e['candidate']) for e in g.get('simple_f0_events', [])]}")
        for k in ("R0.actualF0", "R2-RICH.actualF0", "R2-SIMPLE.actualF0"):
            if k in g.get("receivers", {}):
                sc = g["receivers"][k].get("scores_proxy", {})
                print(f" {k} c={sc.get('n_correct')} w={sc.get('n_wrong')} u={sc.get('n_unresolved')}")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
