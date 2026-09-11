"""PSEM ownership residual causes probe: ASR versus GT interval crossed with actualF0 versus GT_STATE."""
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
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
EXP = ROOT / "experiments" / "psem_ownership_residual_causes"
ACC = ROOT / "experiments" / "psem_pretranslation_receiver"
STAGE2 = ROOT / "experiments" / "psem_repeatability_stage2"
DEC = ROOT / "experiments" / "psem_decision_sufficiency"
FREEZE = json.loads((EXP / "FREEZE.json").read_text(encoding="utf-8"))
ACC_FREEZE = json.loads((ACC / "FREEZE.json").read_text(encoding="utf-8"))
ACC_LEDGER = json.loads((ACC / "ledger.json").read_text(encoding="utf-8"))
TAU = 0.5
CONFIRMATION = 1600
FRAME = 1280
REGION_PAD = 32000
from experiments.psem_pretranslation_receiver.replay import (
    build_native_table, anchor_support_frames, map_anchor_slot, trace_init_s,
    build_schedule_np, decode_state_events, load_capture, r2_partition,
    score_np_window, conservation_check, throwaway_controls, align_window,
    gt_window_samples, gt_candidate_for_frame, load_np, load_gt_words,
    meet_of_case, sid_of_case, r0_word_ownership, run_guards,
)
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
def load_all():
    obs = json.loads((ROOT / "experiments" / "psem_phase_a_headroom" / "observations" / "OBSERVATIONS.json").read_text(encoding="utf-8"))
    cache = json.loads((ROOT / "experiments" / "psem_phase_a_headroom" / "old_grid_cache.json").read_text(encoding="utf-8"))
    probs_by_sid = {}
    for s in obs["sources"]:
        sid = s["source_id"]
        probs_by_sid[sid] = load_np(ROOT / s["feature_files"]["probs"]["path"], tuple(s["feature_files"]["probs"]["shape"]))
    gt_cache_by_meet = {}
    for meet in ("ES2009c", "ES2009d", "ES2002b", "ES2009a", "EN2009d"):
        d = {}
        for role in ("A", "B", "C", "D"):
            d[role] = []
            for w in load_gt_words(meet, role):
                d[role].append((int(round(w["start"] * 16000)), int(round(w["end"] * 16000))))
        gt_cache_by_meet[meet] = d
    return obs, cache, probs_by_sid, gt_cache_by_meet
def frame_report(lo, hi, table):
    f0 = max(0, int(lo // FRAME))
    f1 = int((hi - 1) // FRAME)
    frames = []
    valid_n = 0
    invalid_n = 0
    valid_dur = 0
    invalid_dur = 0
    for fi in range(f0, f1 + 1):
        fs = fi * FRAME
        fe = (fi + 1) * FRAME
        ov = min(fe, hi) - max(fs, lo)
        v = bool(table["valid"][fi]) if 0 <= fi < len(table["valid"]) else False
        vo = bool(table["valid_old"][fi]) if 0 <= fi < len(table["valid_old"]) else False
        vn = bool(table["valid_native"][fi]) if 0 <= fi < len(table["valid_native"]) else False
        sp = bool(table["speech"][fi]) if 0 <= fi < len(table["speech"]) else False
        mk = bool(table["masked"][fi]) if 0 <= fi < len(table["masked"]) else False
        frames.append({"frame": fi, "span": [fs, fe], "overlap": ov, "valid": v, "valid_old": vo, "valid_native": vn, "speech": sp, "masked": mk})
        if v:
            valid_n += 1
            valid_dur += ov
        else:
            invalid_n += 1
            invalid_dur += ov
    return {"lo": lo, "hi": hi, "f0": f0, "f1": f1, "n_frames": f1 - f0 + 1, "n_valid": valid_n, "n_invalid": invalid_n, "dur_total": hi - lo, "dur_valid": valid_dur, "dur_invalid": invalid_dur, "frames": frames}
def old_rows_for(sid, lo, hi, cache):
    out = []
    for r in cache["sources"][sid]:
        if r["e"] > lo and r["s"] < hi:
            out.append({"s": r["s"], "e": r["e"], "v": bool(r["v"]), "m": bool(r["m"]), "sp": bool(r["sp"]), "ep": r["ep"], "overlap": min(hi, r["e"]) - max(lo, r["s"])})
    return sorted(out, key=lambda r: r["s"])
def old_gap_note(sid, lo, hi, cache):
    rows = sorted(cache["sources"][sid], key=lambda r: r["s"])
    cover = []
    for r in rows:
        if r["e"] > lo and r["s"] < hi:
            cover.append(r)
    if len(cover) == 0:
        before = [r for r in rows if r["e"] <= lo]
        after = [r for r in rows if r["s"] >= hi]
        b = max(before, key=lambda r: r["e"]) if before else None
        a = min(after, key=lambda r: r["s"]) if after else None
        return {"kind": "no-old-rows", "before": ({"s": b["s"], "e": b["e"], "ep": b["ep"]} if b else None), "after": ({"s": a["s"], "e": a["e"], "ep": a["ep"]} if a else None)}
    return {"kind": "rows-present", "n": len(cover)}
def run_np_case(case, obs, cache, probs_by_sid, gt_cache_by_meet):
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
    slot = mapping["slot"]
    other_slots = [s for s in range(probs.shape[1]) if s != slot]
    import numpy as _np
    ref_p = probs[:, slot]
    other_max = _np.max(probs[:, other_slots], axis=1)
    from experiments.psem_pretranslation_receiver.replay import classify_frame
    cand_f0 = [classify_frame(float(ref_p[i]), float(other_max[i])) for i in range(n)]
    gt_cache = gt_cache_by_meet[meet]
    cand_gt = [gt_candidate_for_frame(meet, anchor_role, i * FRAME, (i + 1) * FRAME, gt_cache) for i in range(n)]
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
    t0 = time.perf_counter()
    f0_dec = decode_state_events(decode_frames, table0["starts"], table0["ends"], table0["valid"], table0["masked"], table0["speech"], cand_f0, table0["frontiers"], c_of, finish, 0.0, pay)
    cpu_f0 = time.perf_counter() - t0
    for e in f0_dec["events"]:
        ci = c_of[e["confirm_frame"]] if 0 <= e["confirm_frame"] < len(c_of) else None
        e["avail"] = (finish[ci] + cpu_f0) if (ci is not None and finish[ci] is not None) else None
    t1 = time.perf_counter()
    gt_dec = decode_state_events(decode_frames, table0["starts"], table0["ends"], table0["valid"], table0["masked"], table0["speech"], cand_gt, table0["frontiers"], c_of, finish, 0.0, pay)
    cpu_gt = time.perf_counter() - t1
    for e in gt_dec["events"]:
        ci = c_of[e["confirm_frame"]] if 0 <= e["confirm_frame"] < len(c_of) else None
        e["avail"] = (finish[ci] + cpu_gt) if (ci is not None and finish[ci] is not None) else None
    for idx, e in enumerate(f0_dec["events"]):
        e["event_id"] = case + ".F0." + str(idx)
        e["rev"] = "f0-r1"
        e["sourceX"] = e["boundary"]
        e["support_frame"] = e["confirm_frame"]
        e["availZ"] = e["avail"]
        e["profile"] = "research-fp16-vulkan"
    for idx, e in enumerate(gt_dec["events"]):
        e["event_id"] = case + ".GT." + str(idx)
        e["rev"] = "gt-r1"
        e["sourceX"] = e["boundary"]
        e["support_frame"] = e["confirm_frame"]
        e["availZ"] = e["avail"]
        e["profile"] = "research-fp16-vulkan-gt-oracle"
        e["oracle"] = True
    seal = cap.get("session", {}).get("seal_wall")
    groups = cap.get("groups", [])
    acc_toks = cap.get("accepted", {}).get("tokens", [])
    cons = conservation_check(acc_toks, groups)
    ctrls = throwaway_controls(acc_toks, groups)
    gt8 = gt_window_samples(case)
    wlo = min(w["start"] for w in gt8) - REGION_PAD
    whi = max(w["end"] for w in gt8) + REGION_PAD
    al8 = align_window(gt8, groups, wlo, whi)
    by_group = {}
    for m in al8["matched"]:
        by_group.setdefault(m["group_idx"], []).append(m["gt"]["id"])
    ambiguous = set()
    for gid, ids in by_group.items():
        if len(ids) > 1:
            for gid2 in ids:
                ambiguous.add(gid2)
    gt_groups = copy.deepcopy(groups)
    gmap = {g["idx"]: g for g in gt_groups}
    sub_map = {}
    for m in al8["matched"]:
        gid = m["group_idx"]
        gt_id = m["gt"]["id"]
        if gt_id in ambiguous:
            sub_map[gt_id] = {"status": "not_identifiable", "reason": "ambiguous multi-group map, never forced"}
            continue
        g = gmap[gid]
        asr_iv = [g["start_src"], g["end_src"]]
        g["start_src"] = m["gt"]["start"]
        g["end_src"] = m["gt"]["end"]
        sub_map[gt_id] = {"status": "substituted", "asr": asr_iv, "gt": [m["gt"]["start"], m["gt"]["end"]]}
    cons_gt = conservation_check(acc_toks, gt_groups)
    r0_own = r0_word_ownership(groups)
    r0_sc = score_np_window(al8, r0_own, anchor_role)
    aa = r2_partition(groups, pay, seal, f0_dec["events"], table0)
    ag = r2_partition(groups, pay, seal, gt_dec["events"], table0)
    ga = r2_partition(gt_groups, pay, seal, f0_dec["events"], table0)
    gg = r2_partition(gt_groups, pay, seal, gt_dec["events"], table0)
    sc_aa = score_np_window(al8, aa["ownership"], anchor_role)
    sc_ag = score_np_window(al8, ag["ownership"], anchor_role)
    sc_ga = score_np_window(al8, ga["ownership"], anchor_role)
    sc_gg = score_np_window(al8, gg["ownership"], anchor_role)
    acc_r2f0 = ACC_LEDGER["cases"][case]["receivers"]["R2.actualF0"]["scores_window8"]
    acc_r2gt = ACC_LEDGER["cases"][case]["receivers"]["R2.GT_STATE"]["scores_window8"]
    def rows_of(sc):
        rows = {}
        for lst, pred in ((sc.get("correct", []), "match"), (sc.get("wrong", []), "match"), (sc.get("unresolved", []), "UNRESOLVED"), (sc.get("missing", []), "MISSING")):
            for r in lst:
                gid = r.get("gt_id")
                if pred == "match":
                    rows[gid] = r.get("pred")
                elif pred == "UNRESOLVED":
                    rows[gid] = "UNRESOLVED"
                else:
                    rows[gid] = "MISSING"
        return rows
    base_rows = rows_of(sc_aa)
    acc_rows = rows_of(acc_r2f0)
    gt_rows = rows_of(acc_r2gt)
    ag_rows = rows_of(sc_ag)
    per_word = []
    matched_by_id = {m["gt"]["id"]: m for m in al8["matched"]}
    for w in gt8:
        gid = w["id"]
        m = matched_by_id.get(gid)
        if m is not None:
            gidx = m["group_idx"]
            g0 = next(g for g in groups if g["idx"] == gidx)
            asr_iv = [g0["start_src"], g0["end_src"]]
            gt_iv = [w["start"], w["end"]]
            entry = {
                "gt_id": gid, "gt_side": ("CURRENT" if w["side"] == "left" and anchor_role in ("B", "D") and ((anchor_role == "B" and w["id"].split(".")[1] == "B") or (anchor_role == "D" and w["id"].split(".")[1] == "D")) else ("CURRENT" if (w["id"].split(".")[1] == anchor_role) else "OTHER")),
                "group_idx": gidx, "text": w["text"], "asr_interval": asr_iv, "gt_interval": gt_iv,
                "substitution": sub_map.get(gid, {}).get("status", "substituted"),
                "AA_pred": base_rows.get(gid), "AG_pred": ag_rows.get(gid), "GA_pred": rows_of(sc_ga).get(gid), "GG_pred": rows_of(sc_gg).get(gid),
                "AA_reason": aa["detail"].get(str(gidx), aa["detail"].get(gidx)),
                "AG_reason": ag["detail"].get(str(gidx), ag["detail"].get(gidx)),
                "GA_reason": ga["detail"].get(str(gidx), ga["detail"].get(gidx)),
                "GG_reason": gg["detail"].get(str(gidx), gg["detail"].get(gidx)),
                "asr_validity": frame_report(asr_iv[0], asr_iv[1], table0),
                "gt_validity": frame_report(gt_iv[0], gt_iv[1], table0),
            }
            per_word.append(entry)
        else:
            per_word.append({"gt_id": gid, "gt_side": "OTHER" if "C.words" in gid and case == "NP2" else ("CURRENT" if gid.split(".")[1] == anchor_role else "OTHER"), "group_idx": None, "text": w["text"], "asr_interval": None, "gt_interval": [w["start"], w["end"]], "substitution": "keep_unchanged", "AA_pred": base_rows.get(gid), "AG_pred": ag_rows.get(gid), "GA_pred": rows_of(sc_ga).get(gid), "GG_pred": rows_of(sc_gg).get(gid)})
    inpay = [e for e in f0_dec["events"] if pay[0] <= e["boundary"] < pay[1] and e["avail"] is not None and seal is not None and e["avail"] <= seal]
    margin = (seal - inpay[-1]["avail"]) if len(inpay) > 0 else None
    return {"case": case, "source": sid, "meet": meet, "anchor_role": anchor_role, "mapping": mapping, "mapping_support_n": len(support), "payload": pay, "episode": ep, "seal_wall": seal, "finalize_wall": cap.get("session", {}).get("finalize_wall"), "f0_events": f0_dec["events"], "gt_events": gt_dec["events"], "conservation": cons, "conservation_gt_groups": cons_gt, "conservation_controls": ctrls, "alignment": {"n_matched": len(al8["matched"]), "n_unmatched": len(al8["unmatched"]), "n_mixed": len(al8["mixed"]), "matched": [{"gt_id": m["gt"]["id"], "group_idx": m["group_idx"], "group_word": m["group_word"], "gt_start": m["gt"]["start"], "gt_end": m["gt"]["end"]} for m in al8["matched"]], "unmatched": [{"gt_id": u["gt"]["id"]} for u in al8["unmatched"]], "mixed": [{"gt_id": m["gt"]["id"], "candidate_group_idxs": m.get("candidate_group_idxs")} for m in al8["mixed"]]}, "ambiguous": sorted(list(ambiguous)), "r0_scores": {"n_correct": r0_sc["n_correct"], "n_wrong": r0_sc["n_wrong"], "n_unresolved": r0_sc["n_unresolved"], "n_missing": r0_sc["n_missing"]}, "arms": {"AA_ASR_actualF0": {"n_correct": sc_aa["n_correct"], "n_wrong": sc_aa["n_wrong"], "n_unresolved": sc_aa["n_unresolved"], "n_missing": sc_aa["n_missing"]}, "AG_ASR_GT_STATE": {"n_correct": sc_ag["n_correct"], "n_wrong": sc_ag["n_wrong"], "n_unresolved": sc_ag["n_unresolved"], "n_missing": sc_ag["n_missing"]}, "GA_GT_actualF0": {"n_correct": sc_ga["n_correct"], "n_wrong": sc_ga["n_wrong"], "n_unresolved": sc_ga["n_unresolved"], "n_missing": sc_ga["n_missing"]}, "GG_GT_GT_STATE": {"n_correct": sc_gg["n_correct"], "n_wrong": sc_gg["n_wrong"], "n_unresolved": sc_gg["n_unresolved"], "n_missing": sc_gg["n_missing"]}}, "baseline_match_acc": {"AA_equals_acc_R2F0": base_rows == acc_rows, "AG_equals_acc_R2GT": ag_rows == gt_rows, "acc_R2F0": {"n_correct": acc_r2f0["n_correct"], "n_wrong": acc_r2f0["n_wrong"], "n_unresolved": acc_r2f0["n_unresolved"], "n_missing": acc_r2f0["n_missing"]}}, "per_word": per_word, "margin_seal_minus_avail": margin, "decode_cpu": {"f0": cpu_f0, "gt": cpu_gt}}
def smoke_all(obs, cache, probs_by_sid, gt_cache_by_meet):
    checks = []
    def ck(name, ok, detail=""):
        checks.append({"name": name, "pass": bool(ok), "detail": detail})
    acc_toks = json.loads((STAGE2 / "captures" / "NP1.json").read_text(encoding="utf-8")).get("accepted", {}).get("tokens", [])
    groups = json.loads((STAGE2 / "captures" / "NP1.json").read_text(encoding="utf-8")).get("groups", [])
    from experiments.psem_pretranslation_receiver.replay import conservation_check
    c0 = conservation_check(acc_toks, groups)
    ck("conservation-exact", c0["conserved"] and c0["text_equal"], str(c0["n_accepted_tokens"]) + " tokens")
    import copy as _cp
    drop = _cp.deepcopy(groups)
    drop = drop[1:]
    ck("drop-detected", not conservation_check(acc_toks, drop)["conserved"], "throwaway")
    dup = _cp.deepcopy(groups)
    dup.append(_cp.deepcopy(groups[0]))
    ck("dup-detected", not conservation_check(acc_toks, dup)["conserved"], "throwaway")
    txt = _cp.deepcopy(groups)
    txt[0] = dict(txt[0])
    txt[0]["text"] = txt[0]["text"] + "X"
    ck("text-detected", not conservation_check(acc_toks, txt)["conserved"], "throwaway")
    ck("tau-fixed", TAU == 0.5, "0.5")
    ck("confirmation-fixed", CONFIRMATION == 1600, "100ms")
    ck("frame-fixed", FRAME == 1280, "80ms")
    total_c = sum(ACC_LEDGER["cases"][c]["receivers"]["R2.actualF0"]["scores_window8"]["n_correct"] for c in ("NP1", "NP2", "NP3"))
    total_u = sum(ACC_LEDGER["cases"][c]["receivers"]["R2.actualF0"]["scores_window8"]["n_unresolved"] for c in ("NP1", "NP2", "NP3"))
    total_m = sum(ACC_LEDGER["cases"][c]["receivers"]["R2.actualF0"]["scores_window8"]["n_missing"] for c in ("NP1", "NP2", "NP3"))
    ck("baseline-17c6u1m", total_c == 17 and total_u == 6 and total_m == 1, str(total_c) + "c " + str(total_u) + "u " + str(total_m) + "m")
    n_pass = sum(1 for c in checks if c["pass"])
    return {"checks": checks, "n_pass": n_pass, "n_total": len(checks)}
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["smoke", "run"])
    a = ap.parse_args()
    obs, cache, probs_by_sid, gt_cache_by_meet = load_all()
    if a.cmd == "smoke":
        res = smoke_all(obs, cache, probs_by_sid, gt_cache_by_meet)
        print(json.dumps(res, indent=1))
        for c in res["checks"]:
            print(("PASS" if c["pass"] else "FAIL") + " " + c["name"] + " " + c.get("detail", "")[:200])
        return 0
    t_all = time.perf_counter()
    cases = {}
    for case in ("NP1", "NP2", "NP3"):
        cases[case] = run_np_case(case, obs, cache, probs_by_sid, gt_cache_by_meet)
    guards_recomputed = run_guards(obs, cache, probs_by_sid, gt_cache_by_meet)
    guard_check = {}
    for gid in ("R1", "R2", "T1", "COMBINED", "BC1", "SINGLE_ES2009c", "SINGLE_ES2009d"):
        prev = ACC_LEDGER["guards"][gid]
        now = guards_recomputed[gid]
        if gid == "BC1":
            ps = prev["receivers"]["R0.actualF0"]["scores_proxy"] if "scores_proxy" in prev["receivers"]["R0.actualF0"] else prev["receivers"]["R0.actualF0"].get("scores_window8")
            ns = now["receivers"]["R0.actualF0"]["scores_proxy"] if "scores_proxy" in now["receivers"]["R0.actualF0"] else None
            guard_check[gid] = {"match": ps == ns, "prev": ps, "now": ns}
        else:
            ps = prev["receivers"]["R2.actualF0"]["scores_proxy"]
            ns = now["receivers"]["R2.actualF0"]["scores_proxy"]
            ps2 = prev["receivers"]["R2.GT_STATE"]["scores_proxy"]
            ns2 = now["receivers"]["R2.GT_STATE"]["scores_proxy"]
            guard_check[gid] = {"match_actualF0": ps == ns, "match_GT_STATE": ps2 == ns2, "prev_actualF0": ps, "now_actualF0": ns}
    sm = smoke_all(obs, cache, probs_by_sid, gt_cache_by_meet)
    wall = time.perf_counter() - t_all
    inputs = dict(FREEZE["inputs"])
    ledger = {"freeze_id": FREEZE["freeze_id"], "generated_at_utc": datetime.now(timezone.utc).isoformat(), "baseline_branch": FREEZE["authority"]["baseline_branch"], "baseline_commit": FREEZE["authority"]["baseline_commit"], "inputs": inputs, "obs_sha": "3ec7c03c9c01bff859b9d6ce8d020aec40432c9deb7b951e1cf19be11a90b5aa", "cases": cases, "guard_recompute_match": guard_check, "guards_note": "proxy already GT times so timing control N/A, no new ASR evidence; proxy helper has no gap check so cross-guard failure cannot condemn NP gap repair", "smoke": sm, "timing": {"wall_s": wall, "note": "own CPU repartition measured observed not zero"}, "constraints": {"paid_api_calls": 0, "new_captures": 0, "git_mutations": 0, "other_file_edits": 0}, "constants": {"TAU": TAU, "CONFIRMATION": CONFIRMATION, "FRAME": FRAME, "REGION_PAD": REGION_PAD}}
    (EXP / "ledger.json").write_text(json.dumps(ledger, indent=1), encoding="utf-8")
    print(json.dumps({"cases": {k: v["arms"] for k, v in cases.items()}, "guard_check": guard_check, "smoke": sm, "wall_s": wall}, indent=1))
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
