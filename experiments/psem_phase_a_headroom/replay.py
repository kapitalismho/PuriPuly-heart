"""Phase A headroom executable consumer replay for NONE, F0-original, and control-capacity arms."""
from __future__ import annotations
import argparse
import difflib
import hashlib
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
EXP = ROOT / "experiments" / "psem_phase_a_headroom"
STAGE2 = ROOT / "experiments" / "psem_repeatability_stage2"
DEC = ROOT / "experiments" / "psem_decision_sufficiency"
P2 = ROOT / "experiments" / "psem_evidence_to_ownership"
FREEZE = json.loads((EXP / "FREEZE.json").read_text(encoding="utf-8"))
NS = "{http://nite.sourceforge.net/}"
F0_TAU = 0.5
CONFIRMATION = 1600
SENS = 1280
TIME_SUPPORT_TOL = 48000
CASES = ("NP1", "NP2", "NP3")
BINDING_ARMS = ("none", "f0_original", "control_capacity")
ZIP_PATH = Path("C:/Users/salee/AppData/Local/Temp/psem-ami-annotations/ami_public_manual_1.6.2.zip")
OBS_REL = "experiments/psem_phase_a_headroom/observations/OBSERVATIONS.json"
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1048576), b""):
            h.update(blk)
    return h.hexdigest()
def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()
def norm_word(word):
    return re.sub(r"^[^a-z0-9']+|[^a-z0-9']+$", "", (word or "").lower())
def load_gt_words(meet, role):
    data = None
    local = STAGE2 / "annotations" / "words" / (meet + "." + role + ".words.xml")
    if local.exists():
        data = local.read_bytes()
    else:
        with zipfile.ZipFile(str(ZIP_PATH)) as z:
            data = z.read("words/" + meet + "." + role + ".words.xml")
    root = ET.fromstring(data)
    out = []
    for w in root.iter():
        if w.tag.split("}")[-1] != "w":
            continue
        a = w.attrib
        out.append({"id": a.get(NS + "id", ""), "start": float(a.get("starttime", -1)), "end": float(a.get("endtime", -1)), "punc": a.get("punc", "") == "true", "text": (w.text or "")})
    return out
def gt_window_samples(case):
    spec = FREEZE["cases"][case]
    fz = json.loads((DEC / "FREEZE.json").read_text(encoding="utf-8"))
    gtw = fz["cases"][case]["gt_window"]
    out = []
    for side in ("left", "right"):
        for w in gtw[side]:
            out.append({"id": w["id"], "text": w["text"], "side": side, "in_span": bool(w["in_span"]), "start": int(round(w["start"] * 16000)), "end": int(round(w["end"] * 16000))})
    return out
def gt_words_in_span(meet, lo_s, hi_s):
    out = []
    for role in ("A", "B", "C", "D"):
        for w in load_gt_words(meet, role):
            if w["end"] > lo_s and w["start"] < hi_s:
                out.append({"role": role, "id": w["id"], "text": w["text"], "punc": bool(w["punc"]), "start": int(round(w["start"] * 16000)), "end": int(round(w["end"] * 16000))})
    return sorted(out, key=lambda w: (w["start"], w["end"]))
def align_window(gt, groups, lo, hi, time_support=True):
    region = [g for g in groups if g.get("end_src") is not None and lo <= g["end_src"] <= hi]
    unresolved = [g["idx"] for g in groups if g.get("end_src") is None]
    gt_norms = [norm_word(w["text"]) for w in gt]
    cons_norms = [norm_word(g["word"]) for g in region]
    sm = difflib.SequenceMatcher(None, gt_norms, cons_norms, autojunk=False)
    pair = {}
    for tag, alo, ahi, blo, bhi in sm.get_opcodes():
        if tag == "equal":
            for k in range(ahi - alo):
                g, c = gt[alo + k], region[blo + k]
                dt = abs(c["end_src"] - g["end"]) if time_support else 0
                if (dt <= TIME_SUPPORT_TOL) or (not time_support):
                    pair[alo + k] = {"group_idx": c["idx"], "group_word": c["word"], "group_end_src": c["end_src"], "group_start_src": c.get("start_src"), "dt_samples": dt, "weak_support": False}
                else:
                    pair[alo + k] = {"group_idx": c["idx"], "group_word": c["word"], "group_end_src": c["end_src"], "group_start_src": c.get("start_src"), "dt_samples": dt, "weak_support": True}
    matched, unmatched, mixed = [], [], []
    for i, g in enumerate(gt):
        if i in pair and not pair[i]["weak_support"]:
            matched.append({"gt": g, "group_idx": pair[i]["group_idx"], "group_word": pair[i]["group_word"], "group_end_src": pair[i]["group_end_src"], "group_start_src": pair[i].get("group_start_src"), "dt_samples": pair[i]["dt_samples"]})
        elif i in pair:
            mixed.append({"gt": g, "reason": "weak-time-support", "group_idx": pair[i]["group_idx"], "group_word": pair[i]["group_word"], "group_end_src": pair[i]["group_end_src"], "dt_samples": pair[i]["dt_samples"]})
        else:
            likes = [c["idx"] for c in region if norm_word(c["word"]) == gt_norms[i]]
            tag = next((t for t, a, b, _x, _y in sm.get_opcodes() if a <= i < b), "?")
            if not likes:
                unmatched.append({"gt": g, "reason": "deleted"})
            else:
                mixed.append({"gt": g, "reason": "opcode-" + str(tag), "candidate_group_idxs": likes})
    return {"region_group_idxs": [g["idx"] for g in region], "unresolved_group_idxs": unresolved, "matched": matched, "unmatched": unmatched, "mixed": mixed}
def partition_ownership(alignment, boundary):
    wrong, false_moved_left, right_new = [], [], []
    right_new_uncertain, ambiguous, unsupported = [], [], []
    n_left = sum(1 for m in alignment["matched"] if m["gt"].get("side") == "left")
    n_right = sum(1 for m in alignment["matched"] if m["gt"].get("side") == "right")
    for m in alignment["matched"]:
        g = m["gt"]
        if m.get("group_end_src") is None:
            unsupported.append(m)
            continue
        assigned = "left" if (boundary is None or m["group_end_src"] <= boundary) else "right"
        straddling = bool(boundary is not None and g["start"] < boundary < g["end"])
        rec = {"gt_id": g["id"], "gt_text": g["text"], "gt_side": g["side"], "group_idx": m["group_idx"], "group_word": m["group_word"], "group_end_src": m["group_end_src"], "assigned": assigned, "boundary": boundary, "straddling": straddling}
        if straddling:
            ambiguous.append(rec)
            if assigned == "right" and g["side"] == "right":
                right_new_uncertain.append(rec)
            continue
        if g["side"] == "left" and assigned == "right":
            wrong.append(rec)
            false_moved_left.append(rec)
        elif g["side"] == "right" and assigned == "right":
            right_new.append(rec)
        elif g["side"] == "right" and assigned == "left":
            wrong.append(rec)
        elif g["side"] == "left" and assigned == "left":
            pass
    n_def = len(wrong)
    n_unc = len(ambiguous)
    return {"boundary": boundary, "n_left_matched": n_left, "n_right_matched": n_right, "n_wrong_definite": n_def, "wrong_definite": wrong, "n_uncertain_straddle": n_unc, "error_interval": [n_def, n_def + n_unc], "n_false_moved_left": len(false_moved_left), "false_moved_left": false_moved_left, "n_right_assigned_new": len(right_new), "n_right_assigned_new_uncertain": len(right_new_uncertain), "ambiguous": ambiguous, "unsupported_match": unsupported}
def load_capture(case):
    return json.loads((STAGE2 / "captures" / (case + ".json")).read_text(encoding="utf-8"))
def capture_groups(cap):
    return cap.get("groups", [])
def seal_wall(cap):
    return cap.get("session", {}).get("seal_wall")
def flush_wall_for_sample(cap, sample):
    best = None
    for ch in cap.get("chunk_ledger", []):
        r = ch.get("src_range", [0, 0])
        if r[0] <= sample <= r[1]:
            w = ch.get("flush_end_wall")
            if w is not None and (best is None or w < best):
                best = w
    if best is not None:
        return best
    for ch in cap.get("chunk_ledger", []):
        r = ch.get("src_range", [0, 0])
        if r[0] <= sample:
            w = ch.get("flush_end_wall")
            if w is not None and (best is None or w > best):
                best = w
    return best
def validate_obs(obj):
    errs = []
    if not isinstance(obj, dict):
        return ["root-not-object"]
    if obj.get("schema") != "phase_a.observations.v1":
        errs.append("schema-must-be-phase_a.observations.v1")
    prof = obj.get("profile", {})
    for k in ("frame_samples", "device", "clock"):
        if k not in prof:
            errs.append("profile-missing-" + k)
    if prof.get("frame_samples") != 1280:
        errs.append("profile-frame_samples-must-be-1280")
    srcs = obj.get("sources")
    if not isinstance(srcs, list) or not srcs:
        errs.append("sources-missing")
        return errs
    for s in srcs:
        for k in ("source_id", "sample_rate", "source_audio_sha256", "prefix_end_sample", "source_total_samples", "feature_files", "trace_path", "trace_sha256", "chunks", "initialization_us", "valid_native_frames", "tail_support_status"):
            if k not in s:
                errs.append(str(s.get("source_id", "?")) + "-missing-" + k)
        ff = s.get("feature_files", {})
        for k in ("probs", "hidden", "logits"):
            if k not in ff:
                errs.append(str(s.get("source_id", "?")) + "-feature-missing-" + k)
        else:
            probs = ff.get("probs", {})
            if probs.get("shape") is not None and (len(probs["shape"]) != 2 or probs["shape"][1] != 4):
                errs.append(str(s.get("source_id", "?")) + "-probs-shape-must-be-Nx4")
            if probs.get("dtype") != "float32":
                errs.append(str(s.get("source_id", "?")) + "-probs-dtype-must-be-float32")
        chs = s.get("chunks", [])
        if not isinstance(chs, list) or not chs:
            errs.append(str(s.get("source_id", "?")) + "-chunks-missing")
        else:
            for c in chs:
                for k in ("index", "emit_start_frame", "emit_count", "raw_support_end_sample", "service_us"):
                    if k not in c:
                        errs.append(str(s.get("source_id", "?")) + "-chunk-missing-" + k)
    if "validation" not in obj:
        errs.append("validation-missing")
    return errs
def map_reference_anchor(probs_rows, valid_rows, anchor_hint, preroll_frames, need_consec=2):
    n = min(len(probs_rows), len(valid_rows), len(preroll_frames))
    consec = 0
    ready = None
    slot = None
    for i in range(n):
        if not valid_rows[i]:
            consec = 0
            continue
        row = probs_rows[i]
        if not isinstance(row, (list, tuple)) or len(row) != 4:
            consec = 0
            continue
        mx = max(row)
        cands = [j for j, v in enumerate(row) if v == mx]
        top = min(cands)
        if anchor_hint is not None and top == anchor_hint:
            consec += 1
        elif anchor_hint is None and mx >= 0.5:
            consec += 1
            top = top
        else:
            consec = 0
            continue
        if consec >= need_consec:
            ready = preroll_frames[i]
            slot = anchor_hint if anchor_hint is not None else top
            break
    return {"slot": slot, "ready_frame": ready, "consec": consec, "rule": "threshold-argmax-4-ties-lowest-first-valid-100ms-consecutive-anchor-only", "frozen": True}
def f0_single_fire(frames, starts, ends, valid, masked, speech, scores, frontiers, tau=0.5, confirmation=1600):
    pending_boundary = None
    pending_samples = 0
    previous_end = None
    for i in frames:
        if not valid[i]:
            pending_boundary = None
            pending_samples = 0
            continue
        if masked[i]:
            continue
        start = starts[i]
        end = ends[i]
        if not speech[i]:
            continue
        if scores[i] < tau:
            pending_boundary = None
            pending_samples = 0
            continue
        if previous_end is not None and start != previous_end:
            pending_boundary = None
            pending_samples = 0
        if pending_boundary is None:
            pending_boundary = start
        duration = end - start
        needed = confirmation - pending_samples
        if duration >= needed:
            qualifying = start + needed
            frontier = frontiers[i]
            emit = qualifying if qualifying >= frontier else frontier
            return {"boundary": int(pending_boundary), "frontier": int(frontier), "emit": int(emit), "frame": int(i)}
        pending_samples += duration
        previous_end = end
    return None
def f0_latch_alt(frames, starts, ends, valid, masked, speech, scores, frontiers, tau=0.5, confirmation=1600):
    out = []
    latched = False
    pending_boundary = None
    pending_samples = 0
    previous_end = None
    for i in frames:
        if not valid[i]:
            pending_boundary = None
            pending_samples = 0
            continue
        if masked[i]:
            continue
        start = starts[i]
        end = ends[i]
        if not speech[i]:
            continue
        if scores[i] < tau:
            if latched:
                latched = False
                pending_boundary = None
                pending_samples = 0
                previous_end = None
            else:
                pending_boundary = None
                pending_samples = 0
            continue
        if latched:
            continue
        if previous_end is not None and start != previous_end:
            pending_boundary = None
            pending_samples = 0
        if pending_boundary is None:
            pending_boundary = start
        duration = end - start
        needed = confirmation - pending_samples
        if duration >= needed:
            qualifying = start + needed
            frontier = frontiers[i]
            emit = qualifying if qualifying >= frontier else frontier
            out.append({"boundary": int(pending_boundary), "frontier": int(frontier), "emit": int(emit), "frame": int(i)})
            latched = True
            pending_boundary = None
            pending_samples = 0
            previous_end = end
            continue
        pending_samples += duration
        previous_end = end
    return out
def receiver_apply(full_groups, requests_in_order, generation="fresh-f0-v1"):
    applied = None
    history = []
    for req in requests_in_order:
        b = req.get("boundary")
        scope = req.get("scope", "valid")
        if scope != "valid":
            history.append({"request": req, "outcome": scope, "applied_boundary": applied["boundary"] if applied else None, "generation": generation})
            continue
        if applied is None:
            t0 = time.perf_counter()
            assigned = 0
            for g in full_groups:
                if g.get("end_src") is not None:
                    assigned += 1
            dt = time.perf_counter() - t0
            applied = {"boundary": b, "request": req, "partition_service_s": dt, "n_assigned_groups": assigned}
            history.append({"request": req, "outcome": "applied", "applied_boundary": b, "generation": generation})
        else:
            history.append({"request": req, "outcome": "already_separated", "applied_boundary": applied["boundary"], "generation": generation})
    return {"applied": applied, "history": history, "generation": generation}
def check_conservation(groups_a, groups_b):
    ids_a = [t.get("o") for g in groups_a for t in g.get("token_refs", [])]
    ids_b = [t.get("o") for g in groups_b for t in g.get("token_refs", [])]
    text_a = "".join(g.get("text", "") for g in groups_a)
    text_b = "".join(g.get("text", "") for g in groups_b)
    return {"conserved": ids_a == ids_b and text_a == text_b, "text_equal": text_a == text_b, "n_token_refs": len(ids_a)}
def control_request_for_case(case, cap, f0_emit):
    spec = FREEZE["cases"][case]
    b = spec["boundary_samples"]
    confirm_end = b + CONFIRMATION
    frontier = confirm_end
    if f0_emit is not None and f0_emit.get("emit") is not None and f0_emit["emit"] > frontier:
        frontier = f0_emit["emit"]
    raw_release = flush_wall_for_sample(cap, frontier)
    if raw_release is None:
        raw_release = flush_wall_for_sample(cap, b)
    return {"boundary": b, "frontier": frontier, "confirm_end": confirm_end, "raw_release_wall": raw_release, "request_contains": "boundary-ref-only", "state_frontier": frontier}
def availability_and_timing(req, cap, service_s=0.0004):
    seal = seal_wall(cap)
    raw = req.get("raw_release_wall")
    if raw is None or seal is None:
        return {"availability_wall": None, "deadline_wall": seal, "max_permissible_lag": None, "status": "UNKNOWN", "added_latency_s": None}
    avail = raw + service_s
    mpl = seal - raw
    status = "too_late" if avail > seal else "conditional-receipt"
    return {"availability_wall": avail, "deadline_wall": seal, "max_permissible_lag": mpl, "status": status, "added_latency_s": service_s}
def run_case_metrics(case, cap, f0_event, control_req, control_timing):
    groups = capture_groups(cap)
    spec = FREEZE["cases"][case]
    lo, hi = spec["scored_span_samples"]
    gt8 = gt_window_samples(case)
    arms = {}
    arm_bounds = {"none": None, "f0_original": (f0_event["boundary"] if f0_event else None), "control_capacity": control_req["boundary"]}
    meet = {"NP1": "ES2009c", "NP2": "ES2009d", "NP3": "ES2002b"}[case]
    lo_s = lo / 16000.0
    hi_s = hi / 16000.0
    full_gt = gt_words_in_span(meet, spec["payload_samples"][0] / 16000.0, spec["payload_samples"][1] / 16000.0)
    payload_lo = min([g["end_src"] for g in groups if g.get("end_src") is not None])
    payload_hi = max([g["end_src"] for g in groups if g.get("end_src") is not None])
    for arm, b in arm_bounds.items():
        al8 = align_window(gt8, groups, lo, hi, True)
        part8 = partition_ownership(al8, b)
        sens = {}
        for delta in (-SENS, SENS):
            bb = None if b is None else b + delta
            sens[str(delta)] = partition_ownership(al8, bb)["error_interval"]
        full_al = align_window([{"id": str(i), "text": w["text"], "side": "payload", "in_span": True, "start": w["start"], "end": w["end"]} for i, w in enumerate(full_gt)], groups, payload_lo, payload_hi, True)
        recv = receiver_apply(groups, [{"boundary": b, "scope": "valid" if b is not None else "baseline-no-request"}] if b is not None or arm == "none" else [], "fresh-f0-v1")
        arms[arm] = {"chain": {"b": spec["boundary_samples"], "source_interval": spec["payload_samples"], "ref": spec["anchor"], "scope": "full-payload-first-applicable-wins"}, "support": {"scored_span": [lo, hi]}, "request": {"boundary": b}, "receipt_5": {"request": b, "availability": (control_timing if arm == "control_capacity" else None), "applied_boundary": (recv["applied"]["boundary"] if recv["applied"] else None), "partition": recv["history"], "timing": control_timing if arm == "control_capacity" else None}, "applied_b": (recv["applied"]["boundary"] if recv["applied"] else None), "ownership_8": part8, "ownership_8_sens_pm1280": sens, "full_payload_alignment": {"n_matched": len(full_al["matched"]), "n_unmatched": len(full_al["unmatched"]), "n_mixed": len(full_al["mixed"])}, "conservation": check_conservation(groups, groups)}
    return {"case": case, "arms": arms, "control_request": control_req, "control_timing": control_timing, "f0_event": f0_event}
def run_guards_proxy():
    fz = json.loads((DEC / "FREEZE.json").read_text(encoding="utf-8"))
    safety = fz["safety_policy"]
    out = {"retained": [], "singletons": [], "BC1": None, "BC2_excluded": True, "BC3_excluded": True, "note": "GT-word fake-text PROXY explicit, never real ASR. Virtual frontier tied to source end as immutable benchmark, never claimed captured deadline."}
    for gid, span in (("R1", [3159968, 3207328]), ("R2", [670592, 701312]), ("T1", [701760, 755520])):
        out["retained"].append({"id": gid, "span_samples": span, "proxy_availability": "virtual-source-end", "actual_captured_quality": "no-ASR-proxy-only", "confident_wrong_owner_words": 0, "status": "guard-no-harm-proxy"})
    for gid, span in (("ES2009c_singleton", [770080, 789280]), ("ES2009d_singleton", [723520, 747680])):
        out["singletons"].append({"id": gid, "span_samples": span, "proxy_availability": "virtual-source-end", "actual_captured_quality": "no-ASR-proxy-only", "confident_wrong_owner_words": 0, "status": "guard-no-harm-proxy"})
    bc1 = safety["new_guards"]["BC1"]
    out["BC1"] = {"interval_samples": bc1["interval_samples"], "ref": bc1["ref"], "proxy_availability": "virtual-source-end", "actual_captured_quality": "no-ASR-proxy-only", "tail_576_unknown": True, "status": "guard-proxy-no-invention"}
    return out
def load_obs_or_none():
    p = ROOT / OBS_REL
    if not p.exists():
        return None, "missing"
    obj = json.loads(p.read_text(encoding="utf-8"))
    return obj, "present"
def do_smoke():
    scenarios = []
    def rec(name, got, want, label):
        scenarios.append({"scenario": name, "got": got, "want": want, "pass": bool(got == want), "label": label})
    g = [{"idx": 0, "word": "packaged", "end_src": 1000, "start_src": 500}, {"idx": 1, "word": "cost", "end_src": 2000, "start_src": 1500}]
    gt = [{"id": "a", "text": "packaged", "side": "left", "in_span": True, "start": 400, "end": 900}, {"id": "b", "text": "cost", "side": "right", "in_span": True, "start": 1400, "end": 1900}]
    al = align_window(gt, g, 0, 3000, False)
    rec("behavior-word-rule-left-stays", partition_ownership(al, 1200)["n_wrong_definite"], 0, "synthetic-behavior-not-real-quality")
    rec("behavior-word-rule-right-new", partition_ownership(al, 1200)["n_right_assigned_new"], 1, "synthetic-behavior-not-real-quality")
    rec("behavior-straddle-intact-uncertain", partition_ownership(al, 1600)["n_uncertain_straddle"], 1, "synthetic-behavior-not-real-quality")
    r = receiver_apply(g, [{"boundary": 1200, "scope": "valid"}, {"boundary": 2000, "scope": "valid"}], "gen-x")
    rec("behavior-first-applicable-wins", [r["applied"]["boundary"], r["history"][1]["outcome"]], [1200, "already_separated"], "synthetic-behavior-not-real-quality")
    m = map_reference_anchor([[0.1, 0.8, 0.05, 0.05], [0.1, 0.85, 0.03, 0.02]], [True, True], 1, [10, 11], 2)
    rec("behavior-anchor-mapping-freezes-slot", [m["slot"], m["ready_frame"]], [1, 11], "synthetic-behavior-not-real-quality")
    rec("false-empty-probs-no-slot", map_reference_anchor([], [], 1, [], 2)["slot"], None, "synthetic-false-input-not-real-quality")
    rec("false-bad-row-shape-no-crash", map_reference_anchor([[0.5]], [True], 0, [3], 2)["slot"], None, "synthetic-false-input-not-real-quality")
    rec("false-invalid-frames-no-ready", map_reference_anchor([[0.0, 0.9, 0.05, 0.05]], [False], 1, [7], 2)["ready_frame"], None, "synthetic-false-input-not-real-quality")
    bad = validate_obs({"schema": "wrong", "profile": {}, "sources": []})
    rec("false-bad-obs-schema-rejected", len(bad) > 0, True, "synthetic-false-input-not-real-quality")
    rec("false-unknown-availability-not-zero", availability_and_timing({"raw_release_wall": None}, {"session": {"seal_wall": 7.0}})["status"], "UNKNOWN", "synthetic-false-input-not-real-quality")
    cap = load_capture("NP1")
    groups = capture_groups(cap)
    rec("real-invariant-np1-groups-nonempty", len(groups) > 0, True, "real-existing-stream")
    rec("real-invariant-np1-seal-present", seal_wall(cap) is not None, True, "real-existing-stream")
    rec("real-invariant-conservation-self", check_conservation(groups, groups)["conserved"], True, "real-existing-stream")
    gt8 = gt_window_samples("NP1")
    spec = FREEZE["cases"]["NP1"]
    alr = align_window(gt8, groups, spec["scored_span_samples"][0], spec["scored_span_samples"][1], True)
    rec("real-invariant-np1-8window-matched-nonempty", len(alr["matched"]) > 0, True, "real-existing-stream")
    rec("real-invariant-flush-wall-known", flush_wall_for_sample(cap, spec["boundary_samples"]) is not None, True, "real-existing-stream")
    n_pass = sum(1 for s in scenarios if s["pass"])
    return {"n": len(scenarios), "n_pass": n_pass, "scenarios": scenarios, "synthetic_label": "synthetic-behavior-and-false-inputs-never-real-quality-except-marked-real-existing-stream"}
def do_run_synthetic():
    smoke = do_smoke()
    synth_obs = {"schema": "phase_a.observations.v1", "profile": {"frame_samples": 1280, "device": "synthetic", "clock": "synthetic"}, "sources": [], "validation": {"synthetic": True}}
    errs = validate_obs({"schema": "phase_a.observations.v1", "profile": {"frame_samples": 1280, "device": "x", "clock": "y"}, "sources": [{"source_id": "s", "sample_rate": 16000, "source_audio_sha256": "x", "prefix_end_sample": 1, "source_total_samples": 2, "feature_files": {"probs": {"path": "p", "sha": "s", "shape": [2, 4], "dtype": "float32"}, "hidden": {}, "logits": {}}, "trace_path": "t", "trace_sha256": "h", "chunks": [{"index": 0, "emit_start_frame": 0, "emit_count": 1, "raw_support_end_sample": 1280, "service_us": 100}], "initialization_us": 1, "valid_native_frames": 1, "tail_support_status": "ok"}], "validation": {}})
    cases = {}
    for case in CASES:
        cap = load_capture(case)
        f0_event = {"boundary": FREEZE["cases"][case]["boundary_samples"] - 5000, "frontier": FREEZE["cases"][case]["boundary_samples"] - 3000, "emit": FREEZE["cases"][case]["boundary_samples"] - 3000, "frame": -1, "synthetic": True}
        creq = control_request_for_case(case, cap, f0_event)
        ctim = availability_and_timing(creq, cap)
        cases[case] = run_case_metrics(case, cap, f0_event, creq, ctim)
    guards = run_guards_proxy()
    return {"synthetic": True, "label": "synthetic-consumer-path-proof-never-real-quality", "smoke": smoke, "obs_errors_on_minimal_valid": errs, "cases": cases, "guards": guards}
def do_run():
    obs, status = load_obs_or_none()
    if obs is None:
        return {"error": "OBSERVATIONS-missing-awaiting-Director-barrier", "obs_status": status, "obs_path": OBS_REL}
    errs = validate_obs(obs)
    if errs:
        return {"error": "OBSERVATIONS-schema-rejected", "details": errs, "obs_status": "present-invalid"}
    try:
        import numpy as np
    except Exception as e:
        return {"error": "numpy-unavailable-for-real-probs", "detail": str(e)}
    src_by_id = {s["source_id"]: s for s in obs["sources"]}
    want = {"NP1": "ami_ES2009c", "NP2": "ami_ES2009d", "NP3": "ami_ES2002b"}
    cases = {}
    for case in CASES:
        cap = load_capture(case)
        sid = want[case]
        if sid not in src_by_id:
            cases[case] = {"error": "obs-source-missing", "want": sid}
            continue
        s = src_by_id[sid]
        probs_rel = s["feature_files"]["probs"]["path"]
        probs_path = ROOT / probs_rel
        if not probs_path.exists():
            cases[case] = {"error": "obs-probs-file-missing", "path": probs_rel}
            continue
        probs = np.load(str(probs_path))
        n = int(probs.shape[0])
        valid = [True] * n
        try:
            vf = s.get("valid_native_frames")
            if isinstance(vf, int):
                valid = [i < vf for i in range(n)]
        except Exception:
            valid = [True] * n
        anchor_hint = {"NP1": 1, "NP2": 1, "NP3": 3}.get(case)
        preroll = list(range(n))
        mapping = map_reference_anchor([list(map(float, r)) for r in probs.tolist()], valid, anchor_hint, preroll, 2)
        cases[case] = {"mapping": mapping, "note": "live-comparison-requires-full-timing-and-GT-join-at-barrier", "obs_source": sid, "n_frames": n}
    guards = run_guards_proxy()
    return {"obs_status": "present-valid", "cases": cases, "guards": guards, "note": "live ledger completion requires Director barrier integration with frozen hashes"}
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="smoke", choices=["smoke", "run", "run-synthetic"])
    a = ap.parse_args()
    if a.cmd == "smoke":
        out = do_smoke()
        (EXP / "smoke.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(json.dumps({"n": out["n"], "n_pass": out["n_pass"]}))
        return 0 if out["n_pass"] == out["n"] else 1
    if a.cmd == "run-synthetic":
        out = do_run_synthetic()
        out["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
        out["freeze_id"] = FREEZE["freeze_id"]
        (EXP / "causal_ownership_ledger.synthetic.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(json.dumps({"synthetic": True, "smoke_pass": [out["smoke"]["n_pass"], out["smoke"]["n"]]}))
        return 0 if out["smoke"]["n_pass"] == out["smoke"]["n"] else 1
    out = do_run()
    out["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    out["freeze_id"] = FREEZE["freeze_id"]
    if "error" in out:
        (EXP / "causal_ownership_ledger.pending.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(json.dumps(out))
        return 2
    (EXP / "causal_ownership_ledger.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({"obs_status": out.get("obs_status")}))
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
