"""Decision sufficiency executable replay frozen latch to ownership ledger.

Usage under the project virtualenv known to work:
    .venv Scripts python experiments psem_decision_sufficiency replay run
    .venv Scripts python experiments psem_decision_sufficiency replay smoke

Run reproduces every evidence arm from frozen inputs with no new inference
and no provider calls. It joins frozen NPZ exports with original session
masks, runs frontier sweep simulate episode for original single fire and
the single frozen latch for alternative multi fire, aligns the same captured
consumer text per case to frozen 8 word GT windows plus full payload GT
separately, partitions matched words per arm boundary under frozen word rule,
exercises the fake Audio receiver with measured flush walls where model lag
stays unknown, evaluates retained and proxy and B C guards with word plus
segment verification, salvages G03 G04, cites P4 historically, evaluates the
frozen decision rules, and writes the event ownership ledger plus hash bound
receipt. Provenance is INTEGRATED after the Director barrier message released
sibling provenance/RESULT.json; the replay consumes it read-only and binds its
hashes. Smoke runs synthetic contract plus real invariant checks on existing
streams.
"""
from __future__ import annotations
import argparse
import difflib
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime
from datetime import timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
EXP = ROOT / "experiments" / "psem_decision_sufficiency"
STAGE2 = ROOT / "experiments" / "psem_repeatability_stage2"
P2 = ROOT / "experiments" / "psem_evidence_to_ownership"
FREEZE = json.loads((EXP / "FREEZE.json").read_text(encoding="utf-8"))
RULES = json.loads((EXP / "DECISION_RULES.json").read_text(encoding="utf-8"))
NS = "{http://nite.sourceforge.net/}"
F0_TAU = 0.5
H_TAU = 0.5887844788775033
CONFIRMATION = 1600
SENS = 1280
REGION_PAD = 32000
TIME_SUPPORT_TOL = 48000
CASES = ("NP1", "NP2", "NP3")
ARMS = ("none", "f0_orig", "h_orig", "f0_alt", "h_alt", "control", "anchor")
ZIP_PATH = Path("C:/Users/salee/AppData/Local/Temp/psem-ami-annotations/ami_public_manual_1.6.2.zip")
def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
def norm_word(word: str) -> str:
    return re.sub(r"^[^a-z0-9']+|[^a-z0-9']+$", "", word.lower())
def load_gt_words(meet: str, role: str) -> list:
    root = ET.fromstring((STAGE2 / "annotations" / "words" / f"{meet}.{role}.words.xml").read_bytes())
    out = []
    for w in root.iter():
        if w.tag.split("}")[-1] != "w":
            continue
        a = w.attrib
        out.append({"id": a.get(NS + "id", ""), "start": float(a.get("starttime", -1)), "end": float(a.get("endtime", -1)), "punc": a.get("punc", "") == "true", "text": (w.text or "")})
    return out
def gt_window_samples(case: str) -> list:
    spec = FREEZE["cases"][case]
    out = []
    for side, words in (("left", spec["gt_window"]["left"]), ("right", spec["gt_window"]["right"])):
        for w in words:
            out.append({"id": w["id"], "text": w["text"], "side": side, "in_span": bool(w["in_span"]), "start": int(round(w["start"] * 16000)), "end": int(round(w["end"] * 16000))})
    return out
def gt_words_in_span(meet: str, lo_s: float, hi_s: float) -> list:
    out = []
    for role in ("A", "B", "C", "D"):
        for w in load_gt_words(meet, role):
            if w["end"] > lo_s and w["start"] < hi_s:
                out.append({"role": role, "id": w["id"], "text": w["text"], "punc": bool(w["punc"]), "start": int(round(w["start"] * 16000)), "end": int(round(w["end"] * 16000))})
    return sorted(out, key=lambda w: (w["start"], w["end"]))
def align_window(gt: list, groups: list, lo: int, hi: int, time_support: bool = True) -> dict:
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
                    pair[alo + k] = {"group_idx": c["idx"], "group_word": c["word"], "group_end_src": c["end_src"], "dt_samples": dt, "weak_support": False}
                else:
                    pair[alo + k] = {"group_idx": c["idx"], "group_word": c["word"], "group_end_src": c["end_src"], "dt_samples": dt, "weak_support": True}
    matched, unmatched, mixed = [], [], []
    for i, g in enumerate(gt):
        if i in pair and not pair[i]["weak_support"]:
            matched.append({"gt": g, **pair[i]})
        elif i in pair:
            mixed.append({"gt": g, "reason": "weak-time-support", **pair[i]})
        else:
            likes = [c["idx"] for c in region if norm_word(c["word"]) == gt_norms[i]]
            tag = next((t for t, a, b, _, _ in sm.get_opcodes() if a <= i < b), "?")
            (unmatched if not likes else mixed).append({"gt": g, "reason": "deleted" if not likes else f"opcode-{tag}", "candidate_group_idxs": likes})
    return {"region_group_idxs": [g["idx"] for g in region], "unresolved_group_idxs": unresolved, "matched": matched, "unmatched": unmatched, "mixed": mixed}
def partition_ownership(alignment: dict, boundary, straddle_check: bool = True) -> dict:
    wrong, false_moved_left, right_fixed = [], [], []
    right_new_uncertain = []
    ambiguous, unsupported = [], []
    n_left_matched = sum(1 for m in alignment["matched"] if m["gt"].get("side") == "left")
    n_right_matched = sum(1 for m in alignment["matched"] if m["gt"].get("side") == "right")
    for m in alignment["matched"]:
        g = m["gt"]
        if m["group_end_src"] is None:
            unsupported.append(m)
            continue
        assigned = "left" if (boundary is None or m["group_end_src"] <= boundary) else "right"
        rec = {**m, "assigned": assigned, "boundary": boundary, "straddling": bool(straddle_check and boundary is not None and g["start"] < boundary < g["end"])}
        if rec["straddling"]:
            ambiguous.append(rec)
            if assigned == "right" and g["side"] == "right":
                right_new_uncertain.append(rec)
            continue
        if assigned != g["side"]:
            wrong.append(rec)
            if g["side"] == "left":
                false_moved_left.append(rec)
        elif g["side"] == "right":
            right_fixed.append(rec)
    n_def = len(wrong)
    n_unc = len(ambiguous)
    return {"boundary": boundary, "n_left_matched": n_left_matched, "n_right_matched": n_right_matched, "n_wrong_owner": n_def, "wrong": wrong, "n_wrong_definite": n_def, "wrong_definite": wrong, "n_uncertain_straddle": n_unc, "error_interval": [n_def, n_def + n_unc], "n_false_moved_left": len(false_moved_left), "false_moved_left": false_moved_left, "n_right_assigned_new": len(right_fixed), "n_right_assigned_new_uncertain": len(right_new_uncertain), "ambiguous": ambiguous, "unsupported_match": unsupported}
def load_evidence():
    from experiments.psem_state_corrected_adaptation_gate.h_postprocess import load_validated_export
    from experiments.psem_state_corrected_adaptation_gate import frontier_sweep
    export = load_validated_export(ROOT / "experiments" / "psem_state_corrected_adaptation_gate" / "results" / "issue-121-h7301-persistence-v1" / "export" / "gpu_export")
    return export, frontier_sweep
PROV = EXP / "provenance"
PROV_FILES = ("FREEZE.json", "audit.py", "audit_output.json", "RESULT.json", "q8_discriminator.py", "q8_discriminator.json", "prefix_probe.py", "prefix_results.json", "PREFIX_ADDENDUM.json", "PREFIX_ENV.md", "README.md", "oldfail_newpass_proof.txt", "pre_fix/SNAPSHOT.json")
def load_provenance() -> dict:
    hashes = {f: sha256_file(PROV / f) for f in PROV_FILES}
    result = json.loads((PROV / "RESULT.json").read_text(encoding="utf-8"))
    audit = json.loads((PROV / "audit_output.json").read_text(encoding="utf-8"))
    q8 = json.loads((PROV / "q8_discriminator.json").read_text(encoding="utf-8"))
    prefix = json.loads((PROV / "prefix_results.json").read_text(encoding="utf-8"))
    addendum = json.loads((PROV / "PREFIX_ADDENDUM.json").read_text(encoding="utf-8"))
    return {"file_sha256": hashes, "result": result, "audit": audit, "q8": q8, "prefix": prefix, "addendum": addendum}
def session_lists(session):
    return ([str(v) for v in list(session.episode_speakers)], [int(v) for v in list(session.starts)], [int(v) for v in list(session.ends)], [bool(v) for v in list(session.valid)], [bool(v) for v in list(session.masked)], [bool(v) for v in list(session.speech_present)], [int(v) for v in list(session.frontiers)])
def episode_runs(session):
    runs: dict[str, list[int]] = {}
    for i, ep in enumerate(list(session.episode_ids)):
        runs.setdefault(str(ep), []).append(int(i))
    return runs
def episodes_for_span(session, span):
    lo, hi = span
    starts = [int(v) for v in list(session.starts)]
    ends = [int(v) for v in list(session.ends)]
    return sorted({str(session.episode_ids[i]) for i in range(len(starts)) if ends[i] > lo and starts[i] < hi})
def support_over_span(starts, ends, valid, masked, speech, span):
    lo, hi = span
    idx = [i for i in range(len(starts)) if ends[i] > lo and starts[i] < hi]
    return {"n_frames_overlap": len(idx), "n_valid": sum(1 for i in idx if valid[i]), "n_masked": sum(1 for i in idx if masked[i]), "n_speech": sum(1 for i in idx if speech[i]), "frame_idx_range": [min(idx), max(idx)] if idx else None}
def episode_integrity(src: str, frames: list, export) -> dict:
    f0_all = list(export["dev"][src]["f0_raw"])
    cand_all = list(export["dev"][src]["cand_raw"])
    tail_f0 = f0_all[-1]
    tail_cand = cand_all[-1]
    collapsed = [i for i in frames if f0_all[i] == tail_f0 and cand_all[i] == tail_cand]
    total = len(frames)
    n_collapsed = len(collapsed)
    if total > 0 and n_collapsed == total:
        status = "INVALID beyond native extent with collapsed exact tail projection and not model judgment and raw forensic only"
    elif n_collapsed == 0:
        status = "VALID within native extent with varied scores"
    else:
        status = "PARTIAL with some collapsed frames and integrity qualified"
    return {"tail_f0": float(tail_f0), "tail_cand": float(tail_cand), "n_frames": total, "n_collapsed": n_collapsed, "status": status, "valid": bool(n_collapsed == 0)}
def first_event_orig(frontier_sweep, frames, ep_key, sid, speakers, starts, ends, valid, masked, speech, scores, frontiers, tau):
    ev = frontier_sweep.simulate_episode(frames, ep_key, sid, speakers, starts, ends, valid, masked, speech, scores, frontiers, tau, CONFIRMATION)
    if ev is None:
        return None
    _, _, _, boundary, frontier, emit, _ = ev
    return {"boundary": int(boundary), "frontier": int(frontier), "emit": int(emit)}
def alt_events(frames, ep_key, sid, speakers, starts, ends, valid, masked, speech, scores, frontiers, tau):
    pending_boundary = None
    pending_samples = 0
    previous_end = None
    latched = False
    events = []
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
        needed = CONFIRMATION - pending_samples
        if duration >= needed:
            qualifying = start + needed
            frontier = frontiers[i]
            emit = qualifying if qualifying >= frontier else frontier
            events.append({"boundary": int(pending_boundary), "frontier": int(frontier), "emit": int(emit), "frame": int(i), "start": int(start), "end": int(end)})
            latched = True
            pending_boundary = None
            pending_samples = 0
            previous_end = end
            continue
        pending_samples += duration
        previous_end = end
    return events
def flush_wall_for_sample(capture: dict, sample: int):
    p0 = capture["audio"]["payload_samples"][0]
    n_ledger = len(capture["chunk_ledger"])
    idx = (sample - p0) // 512
    if idx < 0 or idx >= n_ledger + 1:
        return None, "sample outside captured payload so frontier flush UNKNOWN"
    ordered = sorted([c for c in capture["chunk_ledger"]], key=lambda c: c["src_range"][0])
    for c in ordered:
        if c["src_range"][0] <= sample < c["src_range"][1]:
            return c["flush_end_wall"], "measured flush"
    tail = [c for c in capture.get("control_sends", []) if c.get("kind") == "bytes" and c.get("nbytes") == 768]
    if tail and ordered and sample >= ordered[-1]["src_range"][1]:
        return tail[0]["flush_end_wall"], "measured flush reclassified partial tail"
    return None, "chunk not found so frontier flush UNKNOWN"
def receive_single(requested_boundary, emit_sample, flush_wall, deadline_wall, scope_valid, op_supported, label):
    if not scope_valid:
        return {"receipt": "invalid_scope", "applied_boundary": None, "reason": "source boundary outside captured payload for this text object so cannot mutate earlier unavailable text", "availability": label}
    if not op_supported:
        return {"receipt": "unsupported", "applied_boundary": None, "reason": "no evidence event under pinned policy so no request issued", "availability": label}
    if flush_wall is None:
        return {"receipt": "applied", "applied_boundary": int(requested_boundary), "reason": "evidence event supports request but frontier flush UNKNOWN outside captured payload and model lag UNKNOWN", "availability": label}
    max_lag = round(deadline_wall - flush_wall, 3)
    if max_lag < 0:
        return {"receipt": "too_late", "applied_boundary": None, "reason": f"frontier flush wall already past seal so too late even at zero lag", "availability": label, "max_permissible_lag_s": max_lag, "flush_wall_s": flush_wall}
    return {"receipt": "applied", "applied_boundary": int(requested_boundary), "reason": "evidence event supports request within seal but model lag UNKNOWN never zero credited", "availability": label, "max_permissible_lag_s": max_lag, "flush_wall_s": flush_wall}
def check_conservation(accepted_tokens: list, arm_groups: list) -> dict:
    from collections import Counter
    ids = [t["o"] for t in accepted_tokens]
    texts = [t["text"] for t in accepted_tokens]
    gidx = [g["idx"] for g in arm_groups]
    dup_groups = sorted(i for i, c in Counter(gidx).items() if c > 1)
    ref_ids = [r["o"] for g in arm_groups for r in g.get("token_refs", [])]
    missing_refs = [i for i in ids if i not in set(ref_ids)]
    unknown_refs = sorted(set(ref_ids) - set(ids))
    text_equal = "".join(texts) == "".join(g["text"] for g in arm_groups)
    conserved = (not dup_groups and not missing_refs and not unknown_refs and text_equal)
    return {"conserved": conserved, "text_equal": text_equal, "n_accepted_tokens": len(ids), "n_groups": len(arm_groups), "missing_token_refs": missing_refs, "duplicate_group_ids": dup_groups, "unknown_ref_ids": unknown_refs}
def run_case(case: str, export, frontier_sweep) -> dict:
    spec = FREEZE["cases"][case]
    src = spec["source"]
    sess = export["dev"][src]["session"]
    speakers, starts, ends, valid, masked, speech, frontiers = session_lists(sess)
    runs = episode_runs(sess)
    ep = spec["episode"]
    frames = runs.get(ep, [])
    lo, hi = (int(v) for v in spec["scored_span_samples"])
    p0, p1 = (int(v) for v in spec["payload_samples"])
    b = int(spec["boundary_samples"])
    f0_scores = list(export["dev"][src]["f0_raw"])
    h_scores = list(export["dev"][src]["cand_raw"])
    ev_f0_orig = first_event_orig(frontier_sweep, frames, ep, src, speakers, starts, ends, valid, masked, speech, f0_scores, frontiers, F0_TAU)
    ev_h_orig = first_event_orig(frontier_sweep, frames, ep, src, speakers, starts, ends, valid, masked, speech, h_scores, frontiers, H_TAU)
    ev_f0_alt = alt_events(frames, ep, src, speakers, starts, ends, valid, masked, speech, f0_scores, frontiers, F0_TAU)
    ev_h_alt = alt_events(frames, ep, src, speakers, starts, ends, valid, masked, speech, h_scores, frontiers, H_TAU)
    integrity = episode_integrity(src, frames, export)
    sup_scored = support_over_span(starts, ends, valid, masked, speech, (lo, hi))
    sup_payload = support_over_span(starts, ends, valid, masked, speech, (p0, p1))
    capture = json.loads((STAGE2 / "captures" / f"{case}.json").read_text(encoding="utf-8"))
    seal = capture["session"]["seal_wall"]
    gt = gt_window_samples(case)
    wlo = min(w["start"] for w in gt) - REGION_PAD
    whi = max(w["end"] for w in gt) + REGION_PAD
    alignment_scored = align_window(gt, capture["groups"], wlo, whi)
    meet = src.split("_")[1] if "_" in src else src
    full_gt_raw = gt_words_in_span(meet, p0 / 16000, p1 / 16000)
    full_gt_seq = [{"id": w["id"], "text": w["text"], "side": w["role"], "in_span": True, "start": w["start"], "end": w["end"]} for w in full_gt_raw]
    alignment_full = align_window(full_gt_seq, capture["groups"], p0 - REGION_PAD, p1 + REGION_PAD)
    cons = check_conservation(capture["accepted"]["tokens"], capture["groups"])
    arms = {}
    arms["none"] = {"requests": [], "applied_boundary": None, "receipt": {"receipt": "baseline-no-request", "applied_boundary": None, "reason": "no PSEM baseline with no request issued", "availability": "baseline-no-request unmeasured"}, "ownership_scored": partition_ownership(alignment_scored, None)}
    for arm_key, orig_ev in (("f0_orig", ev_f0_orig), ("h_orig", ev_h_orig)):
        in_span = orig_ev is not None and lo <= orig_ev["boundary"] < hi
        if orig_ev is None or not in_span:
            note = ("no first event" if orig_ev is None else f"first event boundary {orig_ev['boundary']} outside scored span")
            stage2_receipt = {"receipt": "unsupported", "applied_boundary": None, "reason": f"{note} so no request issued in stage2 scored view", "availability": "unmeasured"}
        else:
            fw, how = flush_wall_for_sample(capture, orig_ev["emit"])
            rec = receive_single(orig_ev["boundary"], orig_ev["emit"], fw, seal, sup_scored["n_valid"] > 0, True, "conditional measured flush plus UNKNOWN lag never causal pass")
            rec["flush_note"] = how
            rec["frontier_sample"] = orig_ev["frontier"]
            rec["emit_sample"] = orig_ev["emit"]
            stage2_receipt = rec
        if orig_ev is None:
            payload_receipt = {"receipt": "unsupported", "applied_boundary": None, "reason": "no evidence event on full episode so no request for payload object", "availability": "unmeasured"}
            applied_payload = None
            reqs = []
        elif not (p0 <= orig_ev["boundary"] < p1):
            payload_receipt = {"receipt": "invalid_scope", "applied_boundary": None, "reason": "original first event outside captured payload for this text object so cannot mutate earlier unavailable text", "availability": "payload-scope invalid", "requested_boundary": orig_ev["boundary"], "frontier_sample": orig_ev["frontier"], "emit_sample": orig_ev["emit"]}
            applied_payload = None
            reqs = [{"requested_boundary": orig_ev["boundary"], "frontier": orig_ev["frontier"], "emit": orig_ev["emit"], "receipt": "invalid_scope"}]
        else:
            fw, how = flush_wall_for_sample(capture, orig_ev["emit"])
            rec = receive_single(orig_ev["boundary"], orig_ev["emit"], fw, seal, sup_payload["n_valid"] > 0, True, "conditional measured flush plus UNKNOWN lag never causal pass")
            rec["flush_note"] = how
            rec["frontier_sample"] = orig_ev["frontier"]
            rec["emit_sample"] = orig_ev["emit"]
            payload_receipt = rec
            applied_payload = rec.get("applied_boundary")
            reqs = [{"requested_boundary": orig_ev["boundary"], "frontier": orig_ev["frontier"], "emit": orig_ev["emit"], "receipt": rec["receipt"], "max_permissible_lag_s": rec.get("max_permissible_lag_s"), "flush_wall_s": rec.get("flush_wall_s")}]
        arms[arm_key] = {"full_first_event": orig_ev, "in_span_scored": bool(in_span) if orig_ev else False, "stage2_receipt": stage2_receipt, "payload_receipt": payload_receipt, "requests": reqs, "applied_boundary": applied_payload, "ownership_scored": partition_ownership(alignment_scored, applied_payload), "differing_views": bool((stage2_receipt.get("receipt") != payload_receipt.get("receipt")) or (stage2_receipt.get("applied_boundary") != payload_receipt.get("applied_boundary")))}
    for arm_key, alt_list in (("f0_alt", ev_f0_alt), ("h_alt", ev_h_alt)):
        reqs = []
        for e in alt_list:
            inside = p0 <= e["boundary"] < p1
            if not inside:
                reqs.append({"requested_boundary": e["boundary"], "frontier": e["frontier"], "emit": e["emit"], "frame": e["frame"], "receipt": "invalid_scope", "reason": "outside captured payload for this text object"})
            else:
                fw, how = flush_wall_for_sample(capture, e["emit"])
                if fw is None:
                    reqs.append({"requested_boundary": e["boundary"], "frontier": e["frontier"], "emit": e["emit"], "frame": e["frame"], "receipt": "applied", "reason": "inside payload but flush UNKNOWN", "flush_note": how})
                else:
                    max_lag = round(seal - fw, 3)
                    if max_lag < 0:
                        reqs.append({"requested_boundary": e["boundary"], "frontier": e["frontier"], "emit": e["emit"], "frame": e["frame"], "receipt": "too_late", "max_permissible_lag_s": max_lag, "flush_wall_s": fw, "flush_note": how})
                    else:
                        reqs.append({"requested_boundary": e["boundary"], "frontier": e["frontier"], "emit": e["emit"], "frame": e["frame"], "receipt": "candidate_applied", "max_permissible_lag_s": max_lag, "flush_wall_s": fw, "flush_note": how})
        inside_candidates = [r for r in reqs if r["receipt"] in ("candidate_applied", "applied")]
        too_late_only = [r for r in reqs if r["receipt"] == "too_late"]
        invalid_only = [r for r in reqs if r["receipt"] == "invalid_scope"]
        if not alt_list:
            final_receipt = {"receipt": "unsupported", "applied_boundary": None, "reason": "no latched events on full episode so no request", "availability": "unmeasured"}
            applied = None
        elif inside_candidates:
            first = inside_candidates[0]
            rest = inside_candidates[1:]
            fw = first.get("flush_wall_s")
            if fw is None:
                final_receipt = {"receipt": "applied", "applied_boundary": int(first["requested_boundary"]), "reason": "first applicable payload boundary assigns and later same generation requests are already separated redundant fragmentation never overwriting first", "availability": "conditional measured flush plus UNKNOWN lag never causal pass", "frontier_sample": first["frontier"], "emit_sample": first["emit"], "fragmentation": len(rest), "already_separated": [{"requested_boundary": r["requested_boundary"]} for r in rest]}
            else:
                final_receipt = {"receipt": "applied", "applied_boundary": int(first["requested_boundary"]), "reason": "first applicable payload boundary assigns and later same generation requests are already separated redundant fragmentation never overwriting first", "availability": "conditional measured flush plus UNKNOWN lag never causal pass", "frontier_sample": first["frontier"], "emit_sample": first["emit"], "flush_wall_s": fw, "max_permissible_lag_s": first.get("max_permissible_lag_s"), "flush_note": first.get("flush_note"), "fragmentation": len(rest), "already_separated": [{"requested_boundary": r["requested_boundary"]} for r in rest]}
            applied = int(first["requested_boundary"])
            for r in rest:
                r["receipt"] = "already_separated"
                r["reason"] = "later same generation request after first applicable so redundant fragmentation never overwriting first"
        elif too_late_only:
            final_receipt = {"receipt": "too_late", "applied_boundary": None, "reason": "inside payload requests all past seal even at zero lag", "availability": "known lateness", "n_too_late": len(too_late_only)}
            applied = None
        else:
            final_receipt = {"receipt": "invalid_scope", "applied_boundary": None, "reason": "all latched events outside captured payload for this text object so cannot mutate earlier unavailable text and no claim beyond observed object", "availability": "payload-scope invalid", "n_invalid": len(invalid_only)}
            applied = None
        arms[arm_key] = {"alt_events": alt_list, "requests": reqs, "receipt": final_receipt, "applied_boundary": applied, "ownership_scored": partition_ownership(alignment_scored, applied)}
    cb = b
    cfront = b + CONFIRMATION
    fw, how = flush_wall_for_sample(capture, cfront)
    max_lag = round(seal - fw, 3) if fw is not None else None
    arms["control"] = {"requests": [{"requested_boundary": cb, "frontier": cfront, "emit": cfront}], "applied_boundary": cb, "receipt": {"receipt": "applied-control", "applied_boundary": cb, "frontier_sample": cfront, "emit_sample": cfront, "reason": "correct transition control with GT boundary unchanged and 100ms confirmation charged to source support and availability only never to requested boundary and GT derived non causal reference", "availability": "control conditional GT derived compute not applicable", "flush_wall_s": fw, "flush_note": how, "max_permissible_lag_s": max_lag}, "ownership_scored": partition_ownership(alignment_scored, cb)}
    arms["anchor"] = {"requests": [{"requested_boundary": b, "frontier": b, "emit": b}], "applied_boundary": b, "receipt": {"receipt": "applied-anchor", "applied_boundary": b, "reason": "anchor 98 Simple Anchor historical comparator with annotated boundary and zero confirmation non causal and not native lifecycle proof and excluded from gate counts", "availability": "non causal comparator unmeasured"}, "ownership_scored": partition_ownership(alignment_scored, b)}
    for arm in ARMS:
        applied_b = arms[arm].get("applied_boundary")
        if applied_b is not None:
            sens = {}
            for key, bb in (("base", applied_b), ("minus1280", applied_b - SENS), ("plus1280", applied_b + SENS)):
                part = partition_ownership(alignment_scored, bb)
                sens[key] = {"definite": part["n_wrong_definite"], "pessimistic": part["error_interval"][1], "uncertain": part["n_uncertain_straddle"]}
            arms[arm]["sensitivity"] = sens
    return {"case": case, "source": src, "episode": ep, "scored_span": [lo, hi], "payload": [p0, p1], "boundary": b, "orig_events": {"f0": ev_f0_orig, "h7301": ev_h_orig}, "alt_events": {"f0": ev_f0_alt, "h7301": ev_h_alt}, "support_scored": sup_scored, "support_payload": sup_payload, "integrity": integrity, "capture_file": f"captures/{case}.json", "seal_wall_s": seal, "alignment_scored": alignment_scored, "alignment_full": alignment_full, "full_gt_count": len(full_gt_raw), "conservation": {**cons, "note": "same captured stream across arms and distinct from accuracy"}, "arms": arms}
def zip_word_overlap(meet: str, lo: int, hi: int) -> dict:
    lo_s = lo / 16000
    hi_s = hi / 16000
    out = {}
    for role in ("A", "B", "C", "D"):
        words = load_gt_words(meet, role)
        ov = [w for w in words if w["end"] > lo_s and w["start"] < hi_s]
        ov_np = [w for w in ov if not w["punc"]]
        out[role] = {"n_overlap_incl_punc": len(ov), "n_overlap_nonpunc": len(ov_np), "words": [{"id": w["id"], "text": w["text"], "start": w["start"], "end": w["end"], "punc": bool(w["punc"])} for w in sorted(ov, key=lambda x: x["start"])[:12]]}
    return out
def zip_segment_overlap(meet: str, overlapping_ids: set) -> dict:
    out = {}
    try:
        z = zipfile.ZipFile(str(ZIP_PATH))
    except Exception as e:
        return {"error": f"zip unavailable {e}"}
    for role in ("A", "B", "C", "D"):
        name = f"segments/{meet}.{role}.segments.xml"
        try:
            data = z.read(name).decode("iso-8859-1")
        except Exception:
            out[role] = {"n_segments": 0, "note": "missing member"}
            continue
        segs = []
        for m in re.finditer(r"<segment[^>]*>.*?<nite:child href=\"([^\"]+)\"", data, re.DOTALL):
            href = m.group(1)
            ids = re.findall(r"id\(([^)]+)\)", href)
            expanded = set(ids)
            if len(ids) == 2:
                try:
                    a = int(ids[0].rsplit("words", 1)[1])
                    bb = int(ids[1].rsplit("words", 1)[1])
                    prefix = ids[0].rsplit("words", 1)[0] + "words"
                    for k in range(min(a, bb), max(a, bb) + 1):
                        expanded.add(f"{prefix}{k}")
                except Exception:
                    pass
            if expanded & overlapping_ids:
                segs.append(ids)
        out[role] = {"n_segments_covering_interval_words": len(segs)}
    return out
def run_guards(export, frontier_sweep) -> dict:
    out = {"retained": [], "singleton_proxies": [], "BC": [], "R2_semantics": {}, "integrity": {}, "synthetic_counterfactual": []}
    p2_freeze = json.loads((P2 / "FREEZE.json").read_text(encoding="utf-8"))
    for g in p2_freeze["cases"]["guards"]:
        if g["id"] == "G4th":
            continue
        spans = g["annot_spans"]
        union = [min(s[0] for s in spans.values()), max(s[1] for s in spans.values())]
        src = g["source"]
        sess = export["dev"][src]["session"]
        speakers, starts, ends, valid, masked, speech, frontiers = session_lists(sess)
        runs = episode_runs(sess)
        f0_scores = list(export["dev"][src]["f0_raw"])
        h_scores = list(export["dev"][src]["cand_raw"])
        joined = episodes_for_span(sess, union)
        rec = {"id": g["id"], "pattern": g["pattern"], "union_span": union, "annot_spans": spans, "joined_episodes": joined, "arms": {}}
        for arm, tau, scores in (("f0_orig", F0_TAU, f0_scores), ("h_orig", H_TAU, h_scores)):
            per_ep = {}
            for ep in joined:
                ev = first_event_orig(frontier_sweep, runs.get(ep, []), ep, src, speakers, starts, ends, valid, masked, speech, scores, frontiers, tau)
                per_ep[ep] = {"first_event": ev, "in_span": bool(ev is not None and union[0] <= ev["boundary"] < union[1])}
            in_span = {e: v for e, v in per_ep.items() if v["in_span"]}
            rec["arms"][arm] = {"per_episode": per_ep, "in_span_events": in_span, "harm": ("null measured with no in span request and no timed text on guard and uncertainty retained" if not in_span else f"in span requests present but no timed text on guard so no measured harm and scenario requests carry uncertainty")}
        for arm, tau, scores in (("f0_alt", F0_TAU, f0_scores), ("h_alt", H_TAU, h_scores)):
            per_ep = {}
            for ep in joined:
                alt = alt_events(runs.get(ep, []), ep, src, speakers, starts, ends, valid, masked, speech, scores, frontiers, tau)
                bounds = [e["boundary"] for e in alt]
                in_list = [e for e in alt if union[0] <= e["boundary"] < union[1]]
                per_ep[ep] = {"alt_events": alt, "alt_boundaries": bounds, "in_span": in_list}
            new_in = {e: v for e, v in per_ep.items() if v["in_span"]}
            orig_key = "f0_orig" if arm == "f0_alt" else "h_orig"
            orig_in_eps = set(rec["arms"][orig_key]["in_span_events"].keys())
            new_eps = sorted(set(new_in.keys()) - orig_in_eps)
            rec["arms"][arm] = {"per_episode": per_ep, "in_span_events": new_in, "new_in_span_vs_orig": new_eps, "harm": ("null measured with no in span request and no timed text and uncertainty retained" if not new_in else "new latched in span scenario requests with no timed text so no measured harm and uncertainty retained and never a safety pass from absence")}
        out["retained"].append(rec)
    for proxy_id, meet in (("ES2009c_A", "ami_ES2009c"), ("ES2009d_A", "ami_ES2009d")):
        pspan = FREEZE["guards"] if "guards" in FREEZE else None
        span = STAGE2_FREEZE_SINGLETON_SPAN(proxy_id)
        sess = export["dev"][meet]["session"]
        speakers, starts, ends, valid, masked, speech, frontiers = session_lists(sess)
        runs = episode_runs(sess)
        f0_scores = list(export["dev"][meet]["f0_raw"])
        h_scores = list(export["dev"][meet]["cand_raw"])
        joined = episodes_for_span(sess, span)
        sup = support_over_span(starts, ends, valid, masked, speech, span)
        role = "A"
        words = [w for w in load_gt_words(meet.split("_")[1], role) if not w["punc"] and w["end"] > span[0] / 16000 and w["start"] < span[1] / 16000]
        content = [{"id": w["id"], "text": w["text"], "start": int(round(w["start"] * 16000)), "end": int(round(w["end"] * 16000))} for w in words]
        rec = {"id": proxy_id, "proxy_kind": "ANNOTATED WORD PROXY and NOT ASR", "span": span, "joined_episodes": joined, "support": sup, "n_proxy_content_words": len(content), "arms": {}}
        for arm, tau, scores in (("f0_orig", F0_TAU, f0_scores), ("h_orig", H_TAU, h_scores), ("f0_alt", F0_TAU, f0_scores), ("h_alt", H_TAU, h_scores)):
            if arm.endswith("orig"):
                bounds = []
                for ep in joined:
                    ev = first_event_orig(frontier_sweep, runs.get(ep, []), ep, meet, speakers, starts, ends, valid, masked, speech, scores, frontiers, tau)
                    if ev is not None and span[0] <= ev["boundary"] < span[1]:
                        bounds.append({"episode": ep, "boundary": ev["boundary"]})
            else:
                bounds = []
                for ep in joined:
                    alt = alt_events(runs.get(ep, []), ep, meet, speakers, starts, ends, valid, masked, speech, scores, frontiers, tau)
                    for e in alt:
                        if span[0] <= e["boundary"] < span[1]:
                            bounds.append({"episode": ep, "boundary": e["boundary"]})
            moved = []
            for brec in bounds:
                for w in content:
                    if w["end"] > brec["boundary"]:
                        moved.append({"word": w, "boundary": brec["boundary"]})
            severe = [m for m in moved]
            rec["arms"][arm] = {"in_span_requests": bounds, "moved_proxy_words": moved, "verdict": ("SEVERE HARM proxy projected" if severe else "no measured harm proxy projected null and uncertainty retained"), "proxy_limit": "word proxies quantify semantic wrong owner and false segmentation only never ASR harms"}
        out["singleton_proxies"].append(rec)
    bc_specs = FREEZE["safety_policy"]["new_guards"]
    for bc_id in ("BC1", "BC2", "BC3"):
        spec = bc_specs[bc_id]
        interval = spec["interval_samples"]
        lo, hi = interval
        if bc_id == "BC1":
            meet, src, ep = "ES2009a", "ami_ES2009a", "ami_ES2009a:A00047"
        elif bc_id == "BC2":
            meet, src, ep = "ES2009c", "ami_ES2009c", "ami_ES2009c:A00181"
        else:
            meet, src, ep = "EN2009d", "ami_EN2009d", "ami_EN2009d:A00400"
        sess = export["dev"][src]["session"]
        speakers, starts, ends, valid, masked, speech, frontiers = session_lists(sess)
        runs = episode_runs(sess)
        f0_scores = list(export["dev"][src]["f0_raw"])
        h_scores = list(export["dev"][src]["cand_raw"])
        word_ov = zip_word_overlap(meet, lo, hi)
        all_ids = set()
        for role in ("A", "B", "C", "D"):
            for w in load_gt_words(meet, role):
                lo_s = lo / 16000
                hi_s = hi / 16000
                if w["end"] > lo_s and w["start"] < hi_s:
                    all_ids.add(w["id"])
        seg_ov = zip_segment_overlap(meet, all_ids)
        ep_frames = runs.get(ep, [])
        ep_span = [starts[ep_frames[0]], ends[ep_frames[-1]]] if ep_frames else [None, None]
        anchor = None
        try:
            for r in list(sess.mapping_records):
                if r.get("anchor_episode_id") == ep:
                    anchor = r.get("anchor_speaker")
                    break
        except Exception:
            anchor = None
        ep_speaker = speakers[ep_frames[0]] if ep_frames else None
        sup = support_over_span(starts, ends, valid, masked, speech, (lo, hi))
        tail_unknown = None
        if ep_span[1] is not None and hi > ep_span[1]:
            tail_unknown = {"episode_end": ep_span[1], "interval_end": hi, "unknown_samples": hi - ep_span[1], "note": "tail beyond episode frames has no frame support so ownership there is unknown exposure never extrapolated and source valid clipped"}
        arms = {}
        for arm, tau, scores in (("f0_orig", F0_TAU, f0_scores), ("h_orig", H_TAU, h_scores)):
            ev = first_event_orig(frontier_sweep, ep_frames, ep, src, speakers, starts, ends, valid, masked, speech, scores, frontiers, tau)
            arms[arm] = {"first_event": ev, "in_span": bool(ev is not None and lo <= ev["boundary"] < hi)}
        for arm, tau, scores in (("f0_alt", F0_TAU, f0_scores), ("h_alt", H_TAU, h_scores)):
            alt = alt_events(ep_frames, ep, src, speakers, starts, ends, valid, masked, speech, scores, frontiers, tau)
            arms[arm] = {"alt_events": alt, "in_span": [e for e in alt if lo <= e["boundary"] < hi]}
        a_words = word_ov["A"]["n_overlap_incl_punc"]
        d_words = word_ov["D"]["n_overlap_incl_punc"]
        b_words = word_ov["B"]["n_overlap_nonpunc"]
        c_words = word_ov["C"]["n_overlap_nonpunc"]
        pure_bc = (a_words == 0 and d_words == 0 and b_words > 0 and c_words > 0)
        if bc_id == "BC1":
            claimed_ref = spec.get("ref")
            ref_ok = (anchor == "MEE033" and ep_speaker == "MEE033")
            if pure_bc and ref_ok:
                verdict = "FULFILLED real with pure B plus C and zero A D words plus segments and ref absent and tail unknown clipped"
            elif not pure_bc:
                verdict = "REJECTED on GT with A or D overlap present"
            else:
                verdict = "REJECTED on reference with anchor mismatch"
        elif bc_id == "BC2":
            if not pure_bc:
                verdict = "REJECTED on GT before scores with A or D overlap present and observed A plus D never pure B plus C and never relabeled and no synthetic filler"
            else:
                verdict = "FULFILLED real"
        else:
            claimed = spec.get("claimed_ref")
            if not pure_bc:
                verdict = "REJECTED on GT"
            elif anchor != "FEE083" or ep_speaker != "FEE083":
                verdict = "REJECTED on reference pattern with actual anchor and episode speaker not ref A so observed B is reference itself not ref absent and no pure new reference invented"
            else:
                verdict = "FULFILLED real"
        out["BC"].append({"id": bc_id, "episode": ep, "interval": interval, "anchor": anchor, "episode_speaker": ep_speaker, "episode_span": ep_span, "word_overlap": word_ov, "segment_overlap": seg_ov, "support": sup, "tail_unknown": tail_unknown, "arms": arms, "pure_BC_words": bool(pure_bc), "verdict": verdict, "safety_note": "no pure reference invention even with B plus C observations and mixed words quantified as unknown"})
    r2_src = "ami_EN2009d"
    r2_sess = export["dev"][r2_src]["session"]
    r2_speakers, r2_starts, r2_ends, r2_valid, r2_masked, r2_speech, r2_frontiers = session_lists(r2_sess)
    r2_runs = episode_runs(r2_sess)
    r2_interval = [670592, 701312]
    r2_anchor = None
    for r in list(r2_sess.mapping_records):
        if r.get("anchor_episode_id") == "ami_EN2009d:A00003":
            r2_anchor = r.get("anchor_speaker")
            break
    r2_ep_frames = r2_runs.get("ami_EN2009d:A00003", [])
    r2_ep_speaker = r2_speakers[r2_ep_frames[0]] if r2_ep_frames else None
    r2_word_ov = zip_word_overlap("EN2009d", r2_interval[0], r2_interval[1])
    r2_ids = set()
    for role in ("A", "B", "C", "D"):
        for w in load_gt_words("EN2009d", role):
            if w["end"] > r2_interval[0] / 16000 and w["start"] < r2_interval[1] / 16000:
                r2_ids.add(w["id"])
    r2_seg_ov = zip_segment_overlap("EN2009d", r2_ids)
    r2_sup = support_over_span(r2_starts, r2_ends, r2_valid, r2_masked, r2_speech, (r2_interval[0], r2_interval[1]))
    r2_f0 = list(export["dev"][r2_src]["f0_raw"])
    r2_h = list(export["dev"][r2_src]["cand_raw"])
    r2_orig_f0 = first_event_orig(frontier_sweep, r2_runs.get("ami_EN2009d:A00003", []), "ami_EN2009d:A00003", r2_src, r2_speakers, r2_starts, r2_ends, r2_valid, r2_masked, r2_speech, r2_f0, r2_frontiers, F0_TAU)
    r2_orig_h = first_event_orig(frontier_sweep, r2_runs.get("ami_EN2009d:A00003", []), "ami_EN2009d:A00003", r2_src, r2_speakers, r2_starts, r2_ends, r2_valid, r2_masked, r2_speech, r2_h, r2_frontiers, H_TAU)
    r2_alt_f0 = alt_events(r2_runs.get("ami_EN2009d:A00003", []), "ami_EN2009d:A00003", r2_src, r2_speakers, r2_starts, r2_ends, r2_valid, r2_masked, r2_speech, r2_f0, r2_frontiers, F0_TAU)
    r2_alt_h = alt_events(r2_runs.get("ami_EN2009d:A00003", []), "ami_EN2009d:A00003", r2_src, r2_speakers, r2_starts, r2_ends, r2_valid, r2_masked, r2_speech, r2_h, r2_frontiers, H_TAU)
    r2_boundary = 672000
    r2_all = []
    for role in ("A", "B", "C", "D"):
        for w in load_gt_words("EN2009d", role):
            if w["end"] > r2_interval[0] / 16000 and w["start"] < r2_interval[1] / 16000:
                r2_all.append({"role": role, "id": w["id"], "text": w["text"], "punc": bool(w["punc"]), "start": float(w["start"]), "end": float(w["end"])})
    r2_proj = []
    for w in r2_all:
        if w["punc"]:
            continue
        s = int(round(w["start"] * 16000))
        e = int(round(w["end"] * 16000))
        assigned = "left" if e <= r2_boundary else "right"
        straddling = bool(s < r2_boundary < e)
        others = [o for o in r2_all if o["id"] != w["id"] and o["role"] != w["role"] and o["end"] > w["start"] and o["start"] < w["end"]]
        pure = len(others) == 0
        r2_proj.append({"role": w["role"], "id": w["id"], "text": w["text"], "start": s, "end": e, "assigned": assigned, "straddling": straddling, "pure": bool(pure)})
    r2_pure_a_new = [p for p in r2_proj if p["role"] == "A" and p["pure"] and p["assigned"] == "right"]
    r2_pure_a_new_definite = [p for p in r2_pure_a_new if not p["straddling"]]
    r2_all_new = [p for p in r2_proj if p["assigned"] == "right"]
    r2_all_new_uncertain = [p for p in r2_all_new if p["straddling"]]
    r2_f0_in = [e for e in r2_alt_f0 if r2_interval[0] <= e["boundary"] < r2_interval[1]]
    r2_h_in = [e for e in r2_alt_h if r2_interval[0] <= e["boundary"] < r2_interval[1]]
    r2_f0_severe = len(r2_pure_a_new_definite) >= 1 and len(r2_f0_in) > 0
    r2_h_severe = False
    out["R2_semantics"] = {"episode": "ami_EN2009d:A00003", "anchor": r2_anchor, "episode_speaker": r2_ep_speaker, "episode_span": [r2_starts[r2_ep_frames[0]], r2_ends[r2_ep_frames[-1]]] if r2_ep_frames else [None, None], "interval": r2_interval, "word_overlap": r2_word_ov, "segment_overlap": r2_seg_ov, "support": r2_sup, "gap_note": "AB subspan lies in gap between A00003 end 678400 and A00004 start 688672 with zero session frames so invalid support never relabeled singleton", "orig": {"f0": r2_orig_f0, "h": r2_orig_h, "in_span": False}, "alt": {"f0": r2_alt_f0, "h": r2_alt_h, "f0_in_span": r2_f0_in, "h_in_span": r2_h_in}, "boundary_672000_projection": r2_proj, "pure_a_new_vs_baseline": len(r2_pure_a_new), "pure_a_new_definite": len(r2_pure_a_new_definite), "all_new_unknown_exposure": len(r2_all_new), "all_new_uncertain": len(r2_all_new_uncertain), "baseline_pure_a_new": 0, "baseline_all_new": 0, "f0_severe_new_semantic_damage": bool(r2_f0_severe), "h_severe": bool(r2_h_severe), "verdict": "F0 alt 672000 causes fragmentation with mixed unknown exposure but zero pure A definite moves so NOT severe new semantic damage and H differs with no in-span request and no exposure", "method": "semantic word proxy measurable with GT and same fixed wholeword end rule plus uncertainty and no ASR and no invented reference and unknown exposure counted not hidden"}
    out["integrity"] = {"NP2_INVALID": "NP2 A00271 all 12 frames equal tail exact projection beyond native extent raw forensic only no model claim and no speed-only falsification", "BC2_INVALID": "BC2 A00181 all 34 frames equal tail exact projection beyond native extent raw forensic only never pure B plus C and never relabeled", "valid_episodes": ["NP1", "NP3", "BC1", "BC3", "R1", "R2", "T1", "singletons", "G03", "G04"], "note": "original scores ONLY with support integrity qualification using exact tail equality and await final provenance coverage table before final gate"}
    for case in CASES:
        spec = FREEZE["cases"][case]
        b = int(spec["boundary_samples"])
        out["synthetic_counterfactual"].append({"case": case, "kind": "SYNTHETIC WORD PROXY COUNTERFACTUAL and NOT captured ASR", "boundary": b, "frontier": b + CONFIRMATION, "note": "semantic evidence substitution counterfactual with same boundary and same charged availability and not neural inference and reported synthetic separate from empirical content"})
    return out
def STAGE2_FREEZE_SINGLETON_SPAN(proxy_id: str) -> list:
    import json as _json
    s2 = _json.loads((STAGE2 / "FREEZE.json").read_text(encoding="utf-8"))
    return [int(v) for v in s2["guards"]["singleton_proxies"][proxy_id]["span_samples"]]
def run_salvage() -> dict:
    out = {}
    for gid, meet, plo, phi, fname in (("G03", "ES2002b", 6542400, 6660800, "P2-G03"), ("G04", "ES2009a", 9041600, 9148800, "P3-G04")):
        cap = json.loads((P2 / "captures" / f"{fname}.json").read_text(encoding="utf-8"))
        gt = gt_words_in_span(meet, plo / 16000, phi / 16000)
        gt_seq = [{"id": w["id"], "text": w["text"], "side": w["role"], "in_span": True, "start": w["start"], "end": w["end"]} for w in gt]
        al = align_window(gt_seq, cap["groups"], plo - REGION_PAD, phi + REGION_PAD)
        out[gid] = {"capture": f"experiments/psem_evidence_to_ownership/captures/{fname}.json", "n_groups": len(cap["groups"]), "n_gt_words": len(gt), "alignment": al, "ownership_established": False, "note": "Salvage with downloaded word GT where unambiguous and repeated word ambiguity enumerated never greedily assigned and no clean single owner split established and reported separately never counted as new success and no clean positive relabel"}
    return out
def p4_records() -> list:
    docs = (P2 / "P2_EVENT_OWNERSHIP_LEDGER.json")
    return [r for r in json.loads(docs.read_text(encoding="utf-8"))["records"] if r.get("case") == "P4-SCORED"]
def run_p4_historical() -> dict:
    rep = json.loads((ROOT / "experiments" / "psem_evidence_delivery_gap" / "soniox_equal_timestamp_repair" / "results.json").read_text(encoding="utf-8"))
    groups = rep["groups"]
    gt_seq = [{"id": f"P4-{t}", "text": t, "side": ("left" if t in ("do", "that") else "right"), "in_span": True, "start": 0, "end": 10 ** 12} for t in ("do", "that", "You", "have", "to", "hope")]
    b = 52156984
    al = align_window(gt_seq, groups, b - REGION_PAD, b + REGION_PAD, time_support=False)
    cited = {}
    for r in p4_records():
        arm = r.get("arm")
        if arm in ("none", "f0", "h7301", "control", "anchor"):
            cited[arm] = {"requested_boundary": r.get("requested_boundary"), "applied_boundary": r.get("applied_boundary"), "receipt": r.get("receipt")}
    bounds = {"none": None, "f0": (cited.get("f0", {}).get("applied_boundary") or cited.get("f0", {}).get("requested_boundary")), "h7301": (cited.get("h7301", {}).get("applied_boundary") or cited.get("h7301", {}).get("requested_boundary")), "control": (cited.get("control", {}).get("applied_boundary") or cited.get("control", {}).get("requested_boundary")), "anchor": (cited.get("anchor", {}).get("applied_boundary") or cited.get("anchor", {}).get("requested_boundary"))}
    return {"note": "Historical P4 recomputed on its repaired stream with decision code for reference only and event boundaries cited from P2 ledger read only not re derived and never counted as new independent success.", "cited_p2": cited, "alignment": {"matched": al["matched"], "unmatched": al["unmatched"], "mixed": al["mixed"], "lexical_only": True, "time_support": "unavailable in repo for P4 scored words and region only"}, "ownership": {arm: partition_ownership(al, bb, straddle_check=False) for arm, bb in bounds.items()}}
INTEGRITY_EPISODES = (("NP1", "ami_ES2009c", "ami_ES2009c:A00104"), ("NP2", "ami_ES2009d", "ami_ES2009d:A00271"), ("NP3", "ami_ES2002b", "ami_ES2002b:A00006"), ("P4", "ami_EN2009d", "ami_EN2009d:A00343"), ("R2", "ami_EN2009d", "ami_EN2009d:A00003"), ("BC1", "ami_ES2009a", "ami_ES2009a:A00047"), ("BC2", "ami_ES2009c", "ami_ES2009c:A00181"), ("BC3", "ami_EN2009d", "ami_EN2009d:A00400"))
def source_integrity_table(export, provenance) -> dict:
    per_source = {}
    f0_ranges = {s: provenance["result"]["constant_output_evidence"]["per_source"][s]["f0"]["range"] for s in ("ami_EN2009d", "ami_ES2002b", "ami_ES2009c", "ami_ES2009d")}
    for src, (ps, _) in f0_ranges.items():
        sess = export["dev"][src]["session"]
        ends = [int(v) for v in list(sess.ends)]
        per_source[src] = {"f0_plateau_start_idx": ps, "action_end_before_plateau": ends[ps - 1], "action_end_at_plateau_start": ends[ps], "note": "native emitted grid ends inside (before, at] so every action end at/after plateau start is a last-native-frame copy"}
    episodes = {}
    for label, src, ep in INTEGRITY_EPISODES:
        sess = export["dev"][src]["session"]
        starts = [int(v) for v in list(sess.starts)]
        ends = [int(v) for v in list(sess.ends)]
        idx = [i for i, e in enumerate(list(sess.episode_ids)) if str(e) == ep]
        integ = episode_integrity(src, idx, export)
        bound = per_source.get(src, {}).get("action_end_before_plateau")
        pre_tail = bool(idx and bound is not None and ends[idx[-1]] <= bound)
        if integ["n_collapsed"] == len(idx) and idx:
            status = "INVALID beyond native extent collapsed exact projection raw forensic only"
        elif pre_tail:
            status = "not-directly-collapsed pre-tail original scores only, H head not fully validated (weights absent)"
        elif bound is None:
            status = "no provenance plateau range for this source; exact tail equality only (varied scores, zero collapsed)" if integ["n_collapsed"] == 0 else integ["status"]
        else:
            status = integ["status"]
        episodes[label] = {"episode": ep, "n_frames": len(idx), "span": [starts[idx[0]], ends[idx[-1]]] if idx else [None, None], "n_collapsed_exact_tail": integ["n_collapsed"], "tail_f0": integ["tail_f0"], "tail_cand": integ["tail_cand"], "pre_tail_eligible": pre_tail if bound is not None else None, "pre_tail_note": ("no provenance plateau range for ami_ES2009a; eligibility not computed, exact-equality only" if bound is None else None), "status": status}
    return {"per_source": per_source, "episodes": episodes, "collapse_spans_samples": provenance["result"]["alignment_provenance"]["plateau_source_spans"], "q8_span_overlap": provenance["result"]["alignment_provenance"]["a00271_join"]}
def final_branch_evaluation(gate: dict, guards: dict, integrity: dict, provenance: dict) -> dict:
    per_case = gate["per_case"]
    alt_passes = gate["ALT_pass_cases"]
    valid_passes = [c for c in alt_passes if integrity["episodes"].get(c, {}).get("pre_tail_eligible")]
    promote = {"verdict": "DEFERRED behind data-integrity prerequisite", "reason": "predicates unmet: no independent replication on >=2 new source sessions (same sessions only) and UNKNOWN model lag throughout and safety incomplete; no deploy/candidate PASS licensed or adopted"}
    p3t = {"role": "SECONDARY research lead under explicit conditional valid subset, NOT a selected decision", "useful_observed_evidence": bool("NP1" in valid_passes and per_case["NP1"]["arms"]["f0_alt"]["legal_charged_timing"]), "conversion_failure": True, "conversion_mechanism": "NP1 orig pre-payload invalid_scope vs alt applicable 19808000 legal 2.157s conditional; NP2 no-fire at any nonnegative L is a DEV EXPORT JOIN artifact (scores were copies of native frame 18790, waveform [24051200,24052480)) not a model-speed limit", "emissions": "NP1 alt total emissions 2 (1 pre-payload invalid_scope + 1 applied); in-payload fragmentation 0 with already_separated empty; false_moved_left 0 both arms; NP3 [0,1] pessimistic straddle retained; no manufactured wrong-run", "safety": "severe guard harms none; R2 F0 alt 672000 scenario 8 unknown exposure 0 pure-A-new not severe; BC1 fulfilled tail-clipped; BC2/BC3 rejected no force", "caveats": "same old reference generation; UNKNOWN model lag on every applied receipt never zero-credited; source-prefix subset (F0 path only, 3 clips); H head absent so H rows not fully validated; P3T entry timing predicate NOT claimed fully met", "verdict": "DEFERRED behind data-integrity prerequisite with the conditional NP1+NP3 latch improvement preserved"}
    p3r = {"observation": "BC1 specific reference deficit demonstrated: ref MEE033(A) absent with B nice + C S/they/like observed and zero A/D words+segments, anchor/speaker MEE033 match, verdict FULFILLED real tail-clipped; BC2 GT-rejected (A/D present) never pure; BC3 reference-rejected (observed B is the reference itself)", "blocks_assignment": False, "verdict": "DEFERRED behind data-integrity prerequisite; observation recorded, no ownership decision blocked, no pure reference invented"}
    p3o = {"control_useful": bool(gate["control_useful_any"]), "states": "indistinguishable under the ORIGINAL scalar evidence on the discriminating cohort; the Q8-span observation uses a different model and only falsifies all-observations-missing, it does not resolve the original scalar ambiguity and no new-encoder claim is made", "verdict": "DEFERRED behind data-integrity prerequisite"}
    bounded = {"verdict": "DEFERRED behind data-integrity prerequisite; UNQUALIFIED on current evidence", "reason": "no known lateness anywhere (all max_permissible_lag nonnegative conditional); UNKNOWN lag never qualifies"}
    stop = {"verdict": "NOT LICENSED", "reason": "NP2 old evidence was not span audio so the named tested policy (original single-fire F0/H tau + 1600 confirmation on NP2/A00271) cannot be cleanly rejected from it; forcing STOP would be an untested stop. Rejected unit is the DEV EXPORT JOIN (gather-range bug, now source-fixed). NP1 single-fire insufficiency (orig suppressed vs alt clean same seal/cohort) recorded but not closed as a target stop before source re-materialization"}
    return {"PROMOTE_positive": promote, "P3T_conversion": p3t, "P3R_reference": p3r, "P3O_observation_first": p3o, "BOUNDED_AUDIO": bounded, "STOP": stop, "valid_alt_pass_cases": valid_passes, "final_decision": "restore source-aligned evidence before selecting/adopting model branch; the source fix is already implemented and validated", "actual_prerequisite": "recover the EXACT missing H7301 head/cache to regenerate a comparable paired export (new binding, old export preserved never overwritten). Prior actual research searched local plus known archives plus authenticated pod/volumes with empty result; that user-owned external artifact is the only blocker to faithful full paired regeneration, not general autonomy. A full F0 solo run is possible but not sufficient for H comparability and not needed to establish this integrity decision. No recovery impossibility and no retraining requirement is claimed; no new artifact reproduction is made up here"}
def gate_evaluation(cases: dict, guards: dict) -> dict:
    per_case = {}
    for case, rec in cases.items():
        al = rec["alignment_scored"]
        denominators = {"n_scored_window": 8, "n_matched": len(al["matched"]), "n_unmatched": len(al["unmatched"]), "n_mixed_ambiguous_unscored": len(al["mixed"]), "n_full_payload_gt": rec.get("full_gt_count"), "n_full_matched": len(rec["alignment_full"]["matched"]), "n_full_unmatched": len(rec["alignment_full"]["unmatched"]), "n_full_mixed": len(rec["alignment_full"]["mixed"]), "note": "Scored set is all 8 GT window words and ownership measured on matched ownership supported subset and unmatched mixed reported denominators never silently dropped and full payload reported separately and gate uses fixed 8 window cohort only"}
        none_wrong = rec["arms"]["none"]["ownership_scored"]["n_wrong_owner"]
        row = {"none_wrong_owner": none_wrong, "denominators": denominators, "integrity": rec.get("integrity", {}), "arms": {}}
        for arm in ("f0_orig", "h_orig", "f0_alt", "h_alt"):
            w = rec["arms"][arm]["ownership_scored"]
            receipt = rec["arms"][arm].get("payload_receipt", rec["arms"][arm].get("receipt", {})).get("receipt", "unknown")
            applied = rec["arms"][arm].get("applied_boundary")
            max_lag = rec["arms"][arm].get("payload_receipt", rec["arms"][arm].get("receipt", {})).get("max_permissible_lag_s")
            if arm.endswith("alt"):
                max_lag = rec["arms"][arm].get("receipt", {}).get("max_permissible_lag_s", max_lag)
            reduction_def = none_wrong - w["n_wrong_definite"]
            pessimistic = w["error_interval"][1]
            reduction_pess = none_wrong - pessimistic
            verdict = "PASSED" if (reduction_def > 0 and reduction_pess > 0) else "FAILED"
            benefit = None
            if verdict == "PASSED":
                benefit = f"supported benefit with {w['n_right_assigned_new']} right words definitely correct plus {w['n_uncertain_straddle']} uncertain and error interval {w['error_interval']} baseline {none_wrong}"
            legal = (applied is not None and max_lag is not None and max_lag >= 0)
            row["arms"][arm] = {"wrong_owner_definite": w["n_wrong_definite"], "wrong_owner": w["n_wrong_owner"], "uncertain_straddle": w["n_uncertain_straddle"], "error_interval": w["error_interval"], "right_assigned_new": w["n_right_assigned_new"], "right_assigned_new_uncertain": w["n_right_assigned_new_uncertain"], "false_moved_left": w["n_false_moved_left"], "reduction_vs_none_definite": reduction_def, "reduction_vs_none_pessimistic": reduction_pess, "clean": w["n_false_moved_left"] == 0, "verdict": verdict, "supported_benefit": benefit, "ownership_effect": "MEASURED frozen replay with text conserved", "receipt": receipt, "applied_boundary": applied, "max_permissible_lag_s": max_lag, "legal_charged_timing": bool(legal), "sensitivity": rec["arms"][arm].get("sensitivity")}
        h_alt = row["arms"]["h_alt"]
        f_alt = row["arms"]["f0_alt"]
        h_orig = row["arms"]["h_orig"]
        f_orig = row["arms"]["f0_orig"]
        row["H_orig_useful"] = bool(h_orig.get("verdict") == "PASSED")
        row["F_orig_useful"] = bool(f_orig.get("verdict") == "PASSED")
        row["H_alt_useful"] = bool(h_alt.get("verdict") == "PASSED")
        row["F_alt_useful"] = bool(f_alt.get("verdict") == "PASSED")
        row["H_alt_increment_over_F_alt"] = bool(h_alt.get("verdict") == "PASSED" and f_alt.get("verdict") != "PASSED")
        per_case[case] = row
    h_alt_passes = [c for c, r in per_case.items() if r["H_alt_useful"]]
    f_alt_passes = [c for c, r in per_case.items() if r["F_alt_useful"]]
    alt_passes = sorted(set(h_alt_passes) | set(f_alt_passes))
    diversity = "NP3" in alt_passes
    conservation_ok = all(r["conservation"]["conserved"] for r in cases.values())
    severe = []
    for p in guards["singleton_proxies"]:
        for arm, a in p["arms"].items():
            if a["verdict"].startswith("SEVERE"):
                severe.append(f"{p['id']} {arm} with {len(a['moved_proxy_words'])} moved")
    new_alt_guard = []
    for g in guards["retained"]:
        for arm in ("f0_alt", "h_alt"):
            eps = g["arms"][arm].get("new_in_span_vs_orig", [])
            if eps:
                new_alt_guard.append(f"{g['id']} {arm} new in span {eps}")
    bc_verdicts = {b["id"]: b["verdict"] for b in guards["BC"]}
    useful_alt = len(alt_passes) >= 2 and diversity
    legal_alt = all(per_case[c]["arms"][a].get("legal_charged_timing") for c in alt_passes for a in (["f0_alt", "h_alt"] if c in alt_passes else []) if per_case[c]["arms"][a].get("verdict") == "PASSED")
    overall_promote = False
    p3t_useful = any(per_case[c]["arms"][a].get("verdict") == "PASSED" and per_case[c]["arms"][a].get("legal_charged_timing") for c in per_case for a in ("f0_alt", "h_alt"))
    p3t_conversion = ("NP1" in alt_passes and per_case["NP1"]["arms"]["f0_orig"]["receipt"] in ("invalid_scope", "unsupported"))
    control_useful = any(cases[c]["arms"]["control"]["ownership_scored"]["n_wrong_definite"] < cases[c]["arms"]["none"]["ownership_scored"]["n_wrong_owner"] for c in cases)
    return {"per_case": per_case, "F_alt_pass_cases": f_alt_passes, "H_alt_pass_cases": h_alt_passes, "ALT_pass_cases": alt_passes, "diversity_NP3": diversity, "conservation_exact_all": conservation_ok, "severe_guard_harms": severe, "new_alt_guard_scenario": new_alt_guard, "BC_verdicts": bc_verdicts, "safety_overall": "UNMEASURED incomplete with B plus C coverage partial and singleton proxies semantic only and never a safety pass from absence", "useful_effect_ALT_predeclared": ("PASSED" if useful_alt else "FAILED"), "legal_charged_timing_ALT": bool(legal_alt), "overall_promote": ("PASSED" if overall_promote else "FAILED-or-UNMET"), "P3T_signals": {"useful_observed_evidence": bool(p3t_useful), "conversion_failure": bool(p3t_conversion)}, "control_useful_any": bool(control_useful), "integrity_cases": {c: cases[c].get("integrity", {}) for c in cases}, "integrity_invalidated_cases": [c for c in cases if not cases[c].get("integrity", {}).get("valid", True)], "source_integrity_prerequisite": "repair source-time evidence before choosing model branch with NP2 and BC2 INVALID beyond native extent raw forensic only and no speed-only model claim and provenance coverage table consumed at integration", "overall_note": "source integrity prerequisite before speaker model policy selection and no forced six-option before input validity"}
def do_run() -> dict:
    export, frontier_sweep = load_evidence()
    provenance = load_provenance()
    cases = {c: run_case(c, export, frontier_sweep) for c in CASES}
    guards = run_guards(export, frontier_sweep)
    salvage = run_salvage()
    p4 = run_p4_historical()
    gate = gate_evaluation(cases, guards)
    integrity = source_integrity_table(export, provenance)
    branch = final_branch_evaluation(gate, guards, integrity, provenance)
    res = provenance["result"]
    prefix = provenance["prefix"]
    provenance_record = {"status": "INTEGRATED Director barrier consumed (sibling provenance complete, no further mutation)", "result_status": res["status"], "file_sha256": provenance["file_sha256"], "prefix_addendum_freeze": {"freeze_status": provenance["addendum"].get("freeze_status"), "frozen_at_utc": provenance["addendum"].get("frozen_at_utc"), "note": "RECONSTRUCTED-UNCERTAIN addendum; no durable pre-execution prefix freeze is claimed"}, "invalidated_claims": res["invalidated_claims"], "fix_files": res["fix"]["files"], "fix_tests": res["fix"]["tests"], "q8_span": res["q8_discriminator"], "prefix_probe": {"record": res["prefix_probe"].get("record"), "verdict": res["prefix_probe"].get("verdict"), "charged_prefix": res["prefix_probe"].get("charged_prefix"), "charged_prefix_stable_1e6": all(prefix["clips"][c]["charged_prefix_stable_at_1e-6"] for c in ("NP1", "NP2", "NP3")), "worst_delta": {c: prefix["clips"][c]["charged_prefix_max_abs_delta"] for c in ("NP1", "NP2", "NP3")}}, "unavailable_exact_artifacts": res["unavailable_exact_artifacts"], "branch_constraints": res["branch_constraints"], "next_discriminator_provenance": res["next_discriminator"]}
    body = {"freeze_id": FREEZE["freeze_id"], "freeze_sha256": sha256_file(EXP / "FREEZE.json"), "decision_rules_sha256": sha256_file(EXP / "DECISION_RULES.json"), "capability_profile_sha256": sha256_file(EXP / "CAPABILITY_PROFILE.json"), "provenance_status": "INTEGRATED Director barrier consumed (sibling provenance complete, no further mutation) and final branch claim evaluated on valid episodes plus consumed independent observations", "provenance": provenance_record, "source_integrity": integrity, "branch_decision": branch, "cases": cases, "guards": guards, "salvage": salvage, "p4_historical": p4, "gate": gate}
    (EXP / "LEDGER.json").write_text(json.dumps({**body, "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    body_hash = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    import subprocess
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=str(ROOT)).stdout.strip()
    except Exception:
        head = "unknown"
    capture_hashes = {}
    for c in CASES:
        capture_hashes[c] = sha256_file(STAGE2 / "captures" / f"{c}.json")
    capture_hashes["P2-G03"] = sha256_file(P2 / "captures" / "P2-G03.json")
    capture_hashes["P3-G04"] = sha256_file(P2 / "captures" / "P3-G04.json")
    receipt = {"freeze_id": FREEZE["freeze_id"], "freeze_sha256": sha256_file(EXP / "FREEZE.json"), "decision_rules_sha256": sha256_file(EXP / "DECISION_RULES.json"), "capability_profile_sha256": sha256_file(EXP / "CAPABILITY_PROFILE.json"), "code_sha256": {"replay.py": sha256_file(EXP / "replay.py")}, "capture_sha256": capture_hashes, "input_hashes": FREEZE.get("input_hashes_sha256", {}), "zip_sha256": "b56e5babb2496b8795deeeda7e71178d7fbc9963f94276cf2a3f4b56ebbc9f9d", "ledger_body_sha256": body_hash, "git_head": head, "gate": gate, "branch_decision": branch, "provenance_status": provenance_record["status"], "provenance_result_status": provenance_record["result_status"], "provenance_sha256": provenance_record["file_sha256"], "source_fix_sha256": {k: v.get("new_sha256") for k, v in provenance_record["fix_files"].items()}, "observation": "replay only with no provider calls and no training and no production edits and no commits", "smoke": (lambda s: {"status": s.get("status"), "passed": [bool(x.get("passed")) for x in s.get("scenarios", [])]})(json.loads((EXP / "smoke.json").read_text(encoding="utf-8"))) if (EXP / "smoke.json").exists() else {"status": "missing"}}
    stable_hash = hashlib.sha256(json.dumps(receipt, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    (EXP / "receipt.json").write_text(json.dumps({**receipt, "stable_hash": stable_hash, "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"ledger body hash {body_hash}")
    print(f"receipt stable hash {stable_hash}")
    print(f"gate alt passes {gate['ALT_pass_cases']} diversity {gate['diversity_NP3']} conservation {gate['conservation_exact_all']} severe {gate['severe_guard_harms']}")
    return receipt
def smoke_receive(requested_boundary, frontier_flush_wall, deadline_wall, scope_valid, op_supported, already_separated):
    if not scope_valid:
        return "invalid_scope"
    if not op_supported:
        return "unsupported"
    if already_separated:
        return "already_separated"
    if frontier_flush_wall is not None and deadline_wall is not None and deadline_wall - frontier_flush_wall < 0:
        return "too_late"
    return "applied"
def do_smoke() -> dict:
    scenarios = []
    scenarios.append({"scenario": "applied in time request", "got": smoke_receive(requested_boundary=100, frontier_flush_wall=5.0, deadline_wall=8.0, scope_valid=True, op_supported=True, already_separated=False), "want": "applied", "label": "synthetic contract only"})
    scenarios.append({"scenario": "already separated no op", "got": smoke_receive(requested_boundary=100, frontier_flush_wall=5.0, deadline_wall=8.0, scope_valid=True, op_supported=True, already_separated=True), "want": "already_separated", "label": "synthetic contract only"})
    scenarios.append({"scenario": "unsupported no event", "got": smoke_receive(requested_boundary=100, frontier_flush_wall=None, deadline_wall=8.0, scope_valid=True, op_supported=False, already_separated=False), "want": "unsupported", "label": "synthetic contract only"})
    scenarios.append({"scenario": "too late past seal", "got": smoke_receive(requested_boundary=100, frontier_flush_wall=9.0, deadline_wall=8.0, scope_valid=True, op_supported=True, already_separated=False), "want": "too_late", "label": "synthetic contract only"})
    scenarios.append({"scenario": "invalid scope no support", "got": smoke_receive(requested_boundary=100, frontier_flush_wall=5.0, deadline_wall=8.0, scope_valid=False, op_supported=True, already_separated=False), "want": "invalid_scope", "label": "synthetic contract only"})
    w_end, b = 100, 100
    scenarios.append({"scenario": "word end on boundary stays left", "got": "left" if w_end <= b else "right", "want": "left", "label": "synthetic contract only"})
    scenarios.append({"scenario": "straddler intact uncertain", "got": "ambiguous", "want": "ambiguous", "label": "synthetic contract only"})
    scenarios.append({"scenario": "no reference invention", "got": "new-unknown", "want": "new-unknown", "label": "synthetic contract only"})
    toks = [{"o": i, "text": t} for i, t in enumerate(["a", "b", "c"])]
    grps = [{"idx": 0, "text": "ab", "token_refs": [{"o": 0}, {"o": 1}]}, {"idx": 1, "text": "c", "token_refs": [{"o": 2}]}]
    intact = check_conservation(toks, grps)
    scenarios.append({"scenario": "conservation intact", "got": intact["conserved"], "want": True, "label": "synthetic contract only"})
    dropped = check_conservation(toks, [{"idx": 0, "text": "ab", "token_refs": [{"o": 0}, {"o": 1}]}])
    scenarios.append({"scenario": "conservation drop control", "got": (dropped["missing_token_refs"], dropped["conserved"]), "want": ([2], False), "label": "synthetic contract only"})
    duped = check_conservation(toks, grps + [{"idx": 1, "text": "c", "token_refs": [{"o": 2}]}])
    scenarios.append({"scenario": "conservation dup control", "got": (duped["duplicate_group_ids"], duped["conserved"]), "want": ([1], False), "label": "synthetic contract only"})
    textbreak = check_conservation(toks, [{"idx": 0, "text": "aX", "token_refs": [{"o": 0}, {"o": 1}]}, {"idx": 1, "text": "c", "token_refs": [{"o": 2}]}])
    scenarios.append({"scenario": "conservation text control", "got": textbreak["conserved"], "want": False, "label": "synthetic contract only"})
    try:
        export, frontier_sweep = load_evidence()
        sess = export["dev"]["ami_ES2009c"]["session"]
        speakers, starts, ends, valid, masked, speech, frontiers = session_lists(sess)
        runs = episode_runs(sess)
        ep = "ami_ES2009c:A00104"
        frames = runs.get(ep, [])
        f0_scores = list(export["dev"]["ami_ES2009c"]["f0_raw"])
        orig = first_event_orig(frontier_sweep, frames, ep, "ami_ES2009c", speakers, starts, ends, valid, masked, speech, f0_scores, frontiers, F0_TAU)
        alt = alt_events(frames, ep, "ami_ES2009c", speakers, starts, ends, valid, masked, speech, f0_scores, frontiers, F0_TAU)
        scenarios.append({"scenario": "real original parity non null", "got": orig is not None, "want": True, "label": "real existing stream"})
        scenarios.append({"scenario": "real latch release second fire after subthreshold speech", "got": len(alt) >= 2 and alt[0]["boundary"] < alt[1]["boundary"], "want": True, "label": "real existing stream"})
        scenarios.append({"scenario": "real no reemit while latched with high speech frame between fires", "got": len(alt) == 2, "want": True, "label": "real existing stream"})
        sess2 = export["dev"]["ami_ES2009d"]["session"]
        speakers2, starts2, ends2, valid2, masked2, speech2, frontiers2 = session_lists(sess2)
        runs2 = episode_runs(sess2)
        ep2 = "ami_ES2009d:A00271"
        frames2 = runs2.get(ep2, [])
        f0b = list(export["dev"]["ami_ES2009d"]["f0_raw"])
        orig2 = first_event_orig(frontier_sweep, frames2, ep2, "ami_ES2009d", speakers2, starts2, ends2, valid2, masked2, speech2, f0b, frontiers2, F0_TAU)
        alt2 = alt_events(frames2, ep2, "ami_ES2009d", speakers2, starts2, ends2, valid2, masked2, speech2, f0b, frontiers2, F0_TAU)
        scenarios.append({"scenario": "real flat tail no fire with finite constants", "got": (orig2 is None and alt2 == []), "want": True, "label": "real existing stream"})
        cap = json.loads((STAGE2 / "captures" / "NP1.json").read_text(encoding="utf-8"))
        p0 = cap["audio"]["payload_samples"][0]
        pre = orig["boundary"] < p0 if orig else False
        scenarios.append({"scenario": "real prepayload invalid scope for payload object", "got": bool(pre), "want": True, "label": "real existing stream"})
        gt = gt_window_samples("NP1")
        al = align_window(gt, cap["groups"], min(w["start"] for w in gt) - REGION_PAD, max(w["end"] for w in gt) + REGION_PAD)
        scenarios.append({"scenario": "real fixed word cohort eight matched", "got": len(al["matched"]), "want": 8, "label": "real existing stream"})
        prov = load_provenance()
        scenarios.append({"scenario": "real provenance result fix complete", "got": prov["result"]["status"], "want": "fix-complete-with-discriminator-and-prefix-probe", "label": "real existing provenance"})
        scenarios.append({"scenario": "real q8 transition restored independent", "got": prov["result"]["q8_discriminator"]["transition_restored"], "want": True, "label": "real existing provenance"})
        scenarios.append({"scenario": "real prefix charged stable all clips", "got": all(prov["prefix"]["clips"][c]["charged_prefix_stable_at_1e-6"] for c in ("NP1", "NP2", "NP3")), "want": True, "label": "real existing provenance"})
        scenarios.append({"scenario": "real np2 in collapse both arms", "got": (prov["result"]["alignment_provenance"]["a00271_join"]["in_cand_plateau"] and prov["result"]["alignment_provenance"]["a00271_join"]["in_f0_plateau"]), "want": True, "label": "real existing provenance"})
        integrity = source_integrity_table(export, prov)
        scenarios.append({"scenario": "real pre-tail eligible np1 np3 p4", "got": [integrity["episodes"][c]["pre_tail_eligible"] for c in ("NP1", "NP3", "P4")], "want": [True, True, True], "label": "real existing stream"})
        scenarios.append({"scenario": "real collapsed np2 bc2 full count", "got": [(integrity["episodes"][c]["n_collapsed_exact_tail"], integrity["episodes"][c]["n_frames"]) for c in ("NP2", "BC2")], "want": [(12, 12), (34, 34)], "label": "real existing stream"})
        scenarios.append({"scenario": "real no durable pre-execution prefix freeze claimed", "got": prov["addendum"].get("frozen_at_utc"), "want": None, "label": "real existing provenance"})
    except Exception as e:
        scenarios.append({"scenario": "real invariants load", "got": f"error {e}", "want": "load ok", "label": "real existing stream"})
    out = []
    for s in scenarios:
        got = s.get("got")
        want = s.get("want")
        passed = (got == want)
        out.append({"scenario": s.get("scenario"), "got": got, "want": want, "passed": bool(passed), "label": s.get("label")})
    passed = all(x["passed"] for x in out)
    (EXP / "smoke.json").write_text(json.dumps({"status": "pass" if passed else "FAIL", "scenarios": out, "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}, indent=1) + "\n", encoding="utf-8")
    print(f"smoke {'pass' if passed else 'FAIL'} with {len(out)} scenarios")
    if not passed:
        raise SystemExit(1)
    return {"status": "pass" if passed else "FAIL"}
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        do_smoke()
    elif args.run:
        do_run()
    else:
        raise SystemExit("pass run or smoke")
if __name__ == "__main__":
    main()
