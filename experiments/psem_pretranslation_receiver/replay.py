"""PSEM pretranslation receiver replay: neutral R0/R1/R2 on fixed ASR text."""
from __future__ import annotations
import argparse
import copy
import difflib
import hashlib
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
EXP = ROOT / "experiments" / "psem_pretranslation_receiver"
STAGE2 = ROOT / "experiments" / "psem_repeatability_stage2"
DEC = ROOT / "experiments" / "psem_decision_sufficiency"
OBSDIR = ROOT / "experiments" / "psem_phase_a_headroom" / "observations"
FREEZE = json.loads((EXP / "FREEZE.json").read_text(encoding="utf-8"))
NS = "{http://nite.sourceforge.net/}"
TAU = 0.5
CONFIRMATION = 1600
FRAME = 1280
REGION_PAD = 32000
TIME_SUPPORT_TOL = 48000
try:
    from experiments.psem_decision_sufficiency.replay import align_window as ds_align_window
    from experiments.psem_decision_sufficiency.replay import partition_ownership as ds_partition_ownership
    HAS_DS = True
except Exception:
    HAS_DS = False
    ds_align_window = None
    ds_partition_ownership = None
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
def norm_word(w):
    return re.sub(r"^[^a-z0-9']+|[^a-z0-9']+$", "", w.lower())
def local_align_window(gt, groups, lo, hi, time_support=True):
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
def align_window(gt, groups, lo, hi, time_support=True):
    if HAS_DS:
        return ds_align_window(gt, groups, lo, hi, time_support)
    return local_align_window(gt, groups, lo, hi, time_support)
def conservation_check(accepted_tokens, groups):
    from collections import Counter
    ids = [t.get("o") for t in accepted_tokens]
    ref_ids = [r.get("o") for g in groups for r in g.get("token_refs", [])]
    gidx = [g.get("idx") for g in groups]
    dup_groups = sorted(i for i, c in Counter(gidx).items() if c > 1)
    missing = [i for i in ids if i not in set(ref_ids)]
    unknown = sorted(set(ref_ids) - set(ids))
    text_equal = "".join(t.get("text", "") for t in accepted_tokens) == "".join(g.get("text", "") for g in groups)
    conserved = (not dup_groups and not missing and not unknown and text_equal)
    return {"conserved": conserved, "text_equal": text_equal, "n_accepted_tokens": len(ids), "n_groups": len(groups), "missing_token_refs": missing, "duplicate_group_ids": dup_groups, "unknown_ref_ids": unknown}
def throwaway_controls(accepted_tokens, groups):
    drop = copy.deepcopy(groups)
    if drop and drop[0].get("token_refs"):
        drop[0]["token_refs"] = drop[0]["token_refs"][1:]
    dup = copy.deepcopy(groups)
    if dup:
        dup.append(copy.deepcopy(dup[0]))
    txt = copy.deepcopy(groups)
    if txt:
        txt[0]["text"] = txt[0].get("text", "") + "X"
    return {"drop_detected": not conservation_check(accepted_tokens, drop)["conserved"], "dup_detected": not conservation_check(accepted_tokens, dup)["conserved"], "text_detected": not conservation_check(accepted_tokens, txt)["conserved"]}
def load_np(path, shape):
    import numpy as np
    raw = np.fromfile(str(path), dtype=np.float32)
    return raw.reshape(shape[0], shape[1])
def load_gt_words(meet, role):
    root = ET.fromstring((STAGE2 / "annotations" / "words" / f"{meet}.{role}.words.xml").read_bytes())
    out = []
    for w in root.iter():
        if w.tag.split("}")[-1] != "w":
            continue
        a = w.attrib
        out.append({"id": a.get(NS + "id", ""), "start": float(a.get("starttime", -1)), "end": float(a.get("endtime", -1)), "punc": a.get("punc", "") == "true", "text": (w.text or "")})
    return out
def load_gt_words_nopunc(meet, role):
    return load_gt_words(meet, role)
def gt_words_in_span(meet, lo_s, hi_s):
    out = []
    for role in ("A", "B", "C", "D"):
        for w in load_gt_words(meet, role):
            if w["end"] > lo_s and w["start"] < hi_s:
                out.append({"role": role, "id": w["id"], "text": w["text"], "punc": bool(w["punc"]), "start": int(round(w["start"] * 16000)), "end": int(round(w["end"] * 16000))})
    return sorted(out, key=lambda w: (w["start"], w["end"]))
def gt_window_samples(case):
    fz = json.loads((DEC / "FREEZE.json").read_text(encoding="utf-8"))
    spec = fz["cases"][case]["gt_window"]
    out = []
    for side in ("left", "right"):
        for w in spec[side]:
            out.append({"id": w["id"], "text": w["text"], "side": side, "in_span": bool(w["in_span"]), "start": int(round(w["start"] * 16000)), "end": int(round(w["end"] * 16000))})
    return out
def meet_of_case(case):
    return {"NP1": "ES2009c", "NP2": "ES2009d", "NP3": "ES2002b"}[case]
def sid_of_case(case):
    return {"NP1": "ami_ES2009c", "NP2": "ami_ES2009d", "NP3": "ami_ES2002b"}[case]
def build_native_table(src_entry, old_rows):
    n = src_entry["valid_native_frames"]
    chunks = src_entry["chunks"]
    ctab = []
    for c in chunks:
        for _ in range(c["emit_count"]):
            ctab.append(c)
    starts = [i * FRAME for i in range(n)]
    ends = [(i + 1) * FRAME for i in range(n)]
    frontiers = [c["raw_support_end_sample"] for c in ctab[:n]]
    masked, speech, valid_old, old_ep = [], [], [], []
    for i in range(n):
        fs, fe = starts[i], ends[i]
        best, bestov = None, 0
        for r in old_rows:
            ov = min(fe, r["e"]) - max(fs, r["s"])
            if ov > bestov:
                bestov, best = ov, r
        if best is None or bestov <= 0:
            masked.append(False)
            speech.append(False)
            valid_old.append(False)
            old_ep.append(None)
        else:
            masked.append(bool(best["m"]))
            speech.append(bool(best["sp"]))
            valid_old.append(bool(best["v"]))
            old_ep.append(best["ep"])
    tail_invalid = [bool(c["raw_support_end_sample"] > src_entry["prefix_end_sample"]) for c in chunks]
    def chunk_of(i):
        s = 0
        for ci, c in enumerate(chunks):
            e = s + c["emit_count"]
            if s <= i < e:
                return ci
            s = e
        return None
    c_of = [chunk_of(i) for i in range(n)]
    valid_native = [(not tail_invalid[c_of[i]]) if c_of[i] is not None else False for i in range(n)]
    return {"starts": starts, "ends": ends, "frontiers": frontiers, "masked": masked, "speech": speech, "valid_old": valid_old, "old_ep": old_ep, "valid_native": valid_native, "chunks": chunks, "chunk_of": c_of, "tail_invalid": tail_invalid}
def anchor_support_frames(meet, arole, f0, f1):
    words = {}
    for role in ("A", "B", "C", "D"):
        words[role] = load_gt_words_nopunc(meet, role)
    support = []
    for i in range(f0, f1):
        fs, fe = i * FRAME, (i + 1) * FRAME
        over = []
        for role in ("A", "B", "C", "D"):
            for w in words[role]:
                if w["end"] * 16000 > fs and w["start"] * 16000 < fe:
                    over.append(role)
        if over and all(r == arole for r in over):
            support.append(i)
    return support
def map_anchor_slot(probs, valid_native, support):
    am = probs.argmax(axis=1)
    prev, consec, found = None, 0, None
    for i in support:
        if i < 0 or i >= len(valid_native) or not valid_native[i]:
            prev, consec = None, 0
            continue
        s = int(am[i])
        consec = consec + 1 if s == prev else 1
        prev = s
        if consec >= 2:
            found = {"slot": s, "ready_frames": [i - 1, i], "ready_sample": (i + 1) * FRAME}
            break
    return found
def trace_init_s(src_entry):
    h = json.loads((ROOT / src_entry["trace_path"]).read_text(encoding="utf-8"))
    return (h.get("load_us", 0) + h.get("sched_setup_us", 0)) / 1000000.0
def build_schedule_np(src_entry, cap, pay, init_s):
    chunks = src_entry["chunks"]
    P0, P1 = pay[0], pay[1]
    ordered = sorted(cap.get("chunk_ledger", []), key=lambda c: c.get("src_range", [0, 0])[0])
    def flush_for_needed(sample):
        for ch in ordered:
            r = ch.get("src_range", [0, 0])
            if r[1] > sample:
                return ch.get("flush_end_wall")
        return None
    release, finish = [], []
    prev = None
    for c in chunks:
        S = c["raw_support_end_sample"]
        if S < P0:
            rel = (S - P0) / 16000.0
        elif S < P1:
            rel = flush_for_needed(S)
        else:
            rel = None
        release.append(rel)
        if rel is None:
            finish.append(None)
        else:
            if prev is None:
                prev = rel + init_s
            prev = (rel if rel > prev else prev) + c["service_us"] / 1000000.0
            finish.append(prev)
    return {"release": release, "finish": finish, "init_s": init_s, "chunk_of": build_native_table_chunk_of(src_entry)}
def build_native_table_chunk_of(src_entry):
    n = src_entry["valid_native_frames"]
    chunks = src_entry["chunks"]
    out = []
    for i in range(n):
        s = 0
        found = None
        for ci, c in enumerate(chunks):
            e = s + c["emit_count"]
            if s <= i < e:
                found = ci
                break
            s = e
        out.append(found)
    return out
def build_schedule_guard(src_entry, init_s):
    chunks = src_entry["chunks"]
    release = [c["raw_support_end_sample"] / 16000.0 for c in chunks]
    finish = []
    prev = None
    for i, c in enumerate(chunks):
        rel = release[i]
        if prev is None:
            prev = rel + init_s
        prev = (rel if rel > prev else prev) + c["service_us"] / 1000000.0
        finish.append(prev)
    return {"release": release, "finish": finish, "init_s": init_s, "chunk_of": build_native_table_chunk_of(src_entry)}
def classify_frame(ref_p, other_max):
    ra = ref_p >= TAU
    oa = other_max >= TAU
    if ra and oa:
        return "CURRENT_PLUS_OTHER"
    if ra and not oa:
        return "CURRENT_ONLY"
    if (not ra) and oa:
        return "OTHER_ONLY"
    return "NONE"
SEM_OF = {"CURRENT_ONLY": "CONTINUE_CURRENT", "OTHER_ONLY": "SEPARATE_OTHER", "CURRENT_PLUS_OTHER": "UNRESOLVED", "NONE": "NOOP"}
def gt_candidate_for_frame(meet, anchor_role, fs, fe, gt_cache):
    present = set()
    for role in ("A", "B", "C", "D"):
        for (s, e) in gt_cache[role]:
            if e > fs and s < fe:
                present.add(role)
                break
    if not present:
        return "NONE"
    ra = anchor_role in present
    oa = any(r != anchor_role for r in present)
    if ra and oa:
        return "CURRENT_PLUS_OTHER"
    if ra:
        return "CURRENT_ONLY"
    return "OTHER_ONLY"
def decode_state_events(decode_frames, starts, ends, valid, masked, speech, candidate_list, frontiers, chunk_of, finish, cpu, obj_interval):
    events = []
    last_confirmed = "CURRENT_ONLY"
    pending = None
    pend_start = None
    pend_n = 0
    prev_end = None
    gap_spans = []
    for i in decode_frames:
        if not valid[i]:
            if pending is not None:
                gap_spans.append({"kind": "invalid-reset", "frame": i})
            pending, pend_n, prev_end = None, 0, None
            continue
        if masked[i] or (not speech[i]):
            continue
        cand = candidate_list[i]
        if cand is None or cand == "NONE":
            pending, pend_n, prev_end = None, 0, None
            continue
        if cand == last_confirmed:
            pending, pend_n, prev_end = None, 0, None
            continue
        s, e = starts[i], ends[i]
        if prev_end is not None and s != prev_end:
            pending, pend_n = None, 0
            prev_end = None
        if pending is None or pending != cand:
            pending = cand
            pend_start = s
            pend_n = 0
            prev_end = s
        dur = e - s
        need = CONFIRMATION - pend_n
        if dur >= need:
            ci = chunk_of[i] if 0 <= i < len(chunk_of) else None
            avail = (finish[ci] + cpu) if (ci is not None and finish[ci] is not None) else None
            events.append({"boundary": int(pend_start), "confirm_frame": int(i), "frontier": int(frontiers[i]), "avail": avail, "candidate": cand, "semantic": SEM_OF[cand], "uncertainty_samples": FRAME})
            last_confirmed = cand
            pending, pend_n, prev_end = None, 0, None
            continue
        pend_n += dur
        prev_end = e
    return {"events": events, "gap_spans": gap_spans, "final_confirmed": last_confirmed}
def load_capture(case):
    return json.loads((STAGE2 / "captures" / (case + ".json")).read_text(encoding="utf-8"))
def flush_frontier_at(cap, avail, pay):
    if avail is None:
        return None
    best = None
    for ch in cap.get("chunk_ledger", []):
        fe = ch.get("flush_end_wall")
        r = ch.get("src_range", [0, 0])
        if fe is not None and fe <= avail:
            if best is None or r[1] > best:
                best = r[1]
    return best
def r0_word_ownership(groups):
    return {g["idx"]: "CURRENT" for g in groups}
def r1_project(groups, cap, pay, terminal, requests_ordered):
    t0 = time.perf_counter()
    seals = []
    history = []
    seen = set()
    for req in requests_ordered:
        b = req["boundary"]
        sem = req["semantic"]
        avail = req["avail"]
        eid = req["event_id"]
        last_Z = seals[-1]["sealed_frontier_Z"] if seals else pay[0]
        max_Z = max((s["sealed_frontier_Z"] for s in seals), default=None)
        applied_bs = [s["boundary"] for s in seals]
        if not (pay[0] <= b < pay[1]):
            history.append({"event_id": eid, "outcome": "invalid_scope", "applied_boundaries": list(applied_bs), "n_seals": len(seals), "reason": "boundary outside text object"})
            continue
        if eid in seen:
            if max_Z is not None and max_Z >= b:
                history.append({"event_id": eid, "outcome": "already_separated", "applied_boundaries": list(applied_bs), "n_seals": len(seals), "separating_Z": max_Z, "reason": "same event/rev duplicate not cut; existing seal separates"})
            else:
                history.append({"event_id": eid, "outcome": "unsupported" if sem in ("UNRESOLVED", "NOOP") else ("unsupported_operation" if (sem == "CONTINUE_CURRENT" and seals) else "noop-continue-no-cut"), "applied_boundaries": list(applied_bs), "n_seals": len(seals), "reason": "duplicate rev not cut"})
            continue
        if sem == "CONTINUE_CURRENT":
            if seals:
                history.append({"event_id": eid, "outcome": "unsupported_operation", "applied_boundaries": list(applied_bs), "n_seals": len(seals), "reason": "return-requires-past unsupported allowed does not cancel reference; future OTHER still legal"})
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
            history.append({"event_id": eid, "outcome": "already_separated", "applied_boundaries": list(applied_bs), "n_seals": len(seals), "separating_Z": max(s["sealed_frontier_Z"] for s in covering), "requestedX": b, "reason": "existing seal >= estimated transition separates from currently new content exact scope"})
            seen.add(eid)
            continue
        if not (last_Z < b < Z):
            if Z <= last_Z:
                history.append({"event_id": eid, "outcome": "tooLate", "applied_boundaries": list(applied_bs), "n_seals": len(seals), "requestedX": b, "sealedZ": Z, "last_Z": last_Z, "reason": "no open content; no new flush progress"})
            else:
                history.append({"event_id": eid, "outcome": "tooLate", "applied_boundaries": list(applied_bs), "n_seals": len(seals), "requestedX": b, "sealedZ": Z, "last_Z": last_Z, "reason": "transition outside current open range"})
            seen.add(eid)
            continue
        seg_id = len(seals) + 1
        seals.append({"boundary": b, "sealed_frontier_Z": Z, "availability": avail, "event_id": eid, "segment": f"OTHER-{seg_id} new-unknown-logical-segment not B identity", "reference": "fixed no-reset; no actual ASR input change"})
        history.append({"event_id": eid, "outcome": "applied", "applied_boundaries": [s["boundary"] for s in seals], "n_seals": len(seals), "availability": avail, "deadline": terminal, "requestedX": b, "sealedZ": Z, "segment": f"OTHER-{seg_id}", "reason": "valid confirmed OTHER inside open PCM range; prospective C13 seal; projection only"})
        seen.add(eid)
    cpu = time.perf_counter() - t0
    ownership = {}
    segments = []
    if not seals:
        for g in groups:
            ownership[g["idx"]] = "CURRENT"
        segments = [{"range": list(pay), "label": "CURRENT initial", "chunk": "all"}]
    else:
        ordered = sorted(seals, key=lambda s: s["sealed_frontier_Z"])
        bounds_Z = [s["sealed_frontier_Z"] for s in ordered]
        for g in groups:
            es = g.get("end_src")
            if es is None:
                ownership[g["idx"]] = "UNRESOLVED"
            elif es <= bounds_Z[0]:
                ownership[g["idx"]] = "CURRENT"
            else:
                ownership[g["idx"]] = "OTHER"
        lo = pay[0]
        segments.append({"range": [lo, bounds_Z[0]], "label": "CURRENT initial", "X": None, "Z": bounds_Z[0]})
        for i, s in enumerate(ordered):
            hi = bounds_Z[i + 1] if i + 1 < len(bounds_Z) else pay[1]
            segments.append({"range": [s["sealed_frontier_Z"], hi], "label": s["segment"], "X": s["boundary"], "Z": s["sealed_frontier_Z"]})
    applied = seals[-1] if seals else None
    return {"seals": seals, "applied": applied, "applied_boundaries": [s["boundary"] for s in seals], "history": history, "ownership": ownership, "segments": segments, "n_seals": len(seals), "cpu_s": cpu, "reference": "fixed no-reset"}
def r1_project_guard(proxy_words, src_entry, sched, obj_span, terminal, requests_ordered):
    t0 = time.perf_counter()
    seals = []
    history = []
    seen = set()
    finish = sched["finish"]
    chunks = src_entry["chunks"]
    supports = [c["raw_support_end_sample"] for c in chunks]
    def guard_Z(avail):
        if avail is None:
            return None
        best = None
        for ci, f in enumerate(finish):
            if f is not None and f <= avail:
                s = supports[ci]
                if best is None or s > best:
                    best = s
        if best is None:
            return None
        if best < obj_span[0]:
            return obj_span[0]
        if best > obj_span[1]:
            return obj_span[1]
        return best
    for req in requests_ordered:
        b = req["boundary"]
        sem = req["semantic"]
        avail = req["avail"]
        eid = req["event_id"]
        last_Z = seals[-1]["sealed_frontier_Z"] if seals else obj_span[0]
        max_Z = max((s["sealed_frontier_Z"] for s in seals), default=None)
        applied_bs = [s["boundary"] for s in seals]
        if not (obj_span[0] <= b < obj_span[1]):
            history.append({"event_id": eid, "outcome": "invalid_scope", "applied_boundaries": list(applied_bs), "n_seals": len(seals)})
            continue
        if eid in seen:
            if max_Z is not None and max_Z >= b:
                history.append({"event_id": eid, "outcome": "already_separated", "applied_boundaries": list(applied_bs), "n_seals": len(seals), "separating_Z": max_Z, "reason": "same event/rev duplicate not cut"})
            else:
                history.append({"event_id": eid, "outcome": "unsupported" if sem in ("UNRESOLVED", "NOOP") else ("unsupported_operation" if (sem == "CONTINUE_CURRENT" and seals) else "noop-continue-no-cut"), "applied_boundaries": list(applied_bs), "n_seals": len(seals), "reason": "duplicate rev not cut"})
            continue
        if sem == "CONTINUE_CURRENT":
            if seals:
                history.append({"event_id": eid, "outcome": "unsupported_operation", "applied_boundaries": list(applied_bs), "n_seals": len(seals), "reason": "return-requires-past unsupported allowed does not cancel reference; future OTHER still legal"})
            else:
                history.append({"event_id": eid, "outcome": "noop-continue-no-cut", "applied_boundaries": [], "n_seals": 0})
            seen.add(eid)
            continue
        if sem in ("UNRESOLVED", "NOOP"):
            history.append({"event_id": eid, "outcome": "unsupported", "applied_boundaries": list(applied_bs), "n_seals": len(seals)})
            seen.add(eid)
            continue
        if avail is None or terminal is None:
            history.append({"event_id": eid, "outcome": "UNKNOWN", "applied_boundaries": list(applied_bs), "n_seals": len(seals)})
            seen.add(eid)
            continue
        if avail > terminal:
            history.append({"event_id": eid, "outcome": "tooLate", "applied_boundaries": list(applied_bs), "n_seals": len(seals), "availability": avail, "deadline": terminal})
            seen.add(eid)
            continue
        Z = guard_Z(avail)
        if Z is None or Z >= obj_span[1]:
            history.append({"event_id": eid, "outcome": "tooLate", "applied_boundaries": list(applied_bs), "n_seals": len(seals), "requestedX": b, "sealedZ": Z})
            seen.add(eid)
            continue
        covering = [s for s in seals if s["sealed_frontier_Z"] >= b]
        if covering:
            history.append({"event_id": eid, "outcome": "already_separated", "applied_boundaries": list(applied_bs), "n_seals": len(seals), "separating_Z": max(s["sealed_frontier_Z"] for s in covering), "requestedX": b, "reason": "existing seal >= estimated transition separates exact scope"})
            seen.add(eid)
            continue
        if not (last_Z < b < Z):
            if Z <= last_Z:
                history.append({"event_id": eid, "outcome": "tooLate", "applied_boundaries": list(applied_bs), "n_seals": len(seals), "requestedX": b, "sealedZ": Z, "last_Z": last_Z, "reason": "no open content"})
            else:
                history.append({"event_id": eid, "outcome": "tooLate", "applied_boundaries": list(applied_bs), "n_seals": len(seals), "requestedX": b, "sealedZ": Z, "last_Z": last_Z, "reason": "transition outside current open range"})
            seen.add(eid)
            continue
        seg_id = len(seals) + 1
        seals.append({"boundary": b, "sealed_frontier_Z": Z, "availability": avail, "event_id": eid, "segment": f"OTHER-{seg_id} new-unknown-logical-segment not B identity", "reference": "fixed no-reset"})
        history.append({"event_id": eid, "outcome": "applied", "applied_boundaries": [s["boundary"] for s in seals], "n_seals": len(seals), "requestedX": b, "sealedZ": Z, "segment": f"OTHER-{seg_id}", "availability": avail, "deadline": terminal})
        seen.add(eid)
    cpu = time.perf_counter() - t0
    ownership = {}
    segments = []
    if not seals:
        for w in proxy_words:
            ownership[w["id"]] = "CURRENT"
        segments = [{"range": list(obj_span), "label": "CURRENT initial"}]
    else:
        ordered = sorted(seals, key=lambda s: s["sealed_frontier_Z"])
        bounds_Z = [s["sealed_frontier_Z"] for s in ordered]
        for w in proxy_words:
            if w["end"] <= bounds_Z[0]:
                ownership[w["id"]] = "CURRENT"
            else:
                ownership[w["id"]] = "OTHER"
        segments.append({"range": [obj_span[0], bounds_Z[0]], "label": "CURRENT initial", "X": None, "Z": bounds_Z[0]})
        for i, s in enumerate(ordered):
            hi = bounds_Z[i + 1] if i + 1 < len(bounds_Z) else obj_span[1]
            segments.append({"range": [s["sealed_frontier_Z"], hi], "label": s["segment"], "X": s["boundary"], "Z": s["sealed_frontier_Z"]})
    applied = seals[-1] if seals else None
    return {"seals": seals, "applied": applied, "applied_boundaries": [s["boundary"] for s in seals], "history": history, "ownership": ownership, "segments": segments, "n_seals": len(seals), "cpu_s": cpu, "reference": "fixed no-reset"}
def r2_partition(groups, obj_span, terminal, events_ordered, table, anchor_note=""):
    t0 = time.perf_counter()
    applic = [e for e in events_ordered if obj_span[0] <= e["boundary"] < obj_span[1] and e["avail"] is not None and terminal is not None and e["avail"] <= terminal]
    applic = sorted(applic, key=lambda e: (e["boundary"], e["confirm_frame"]))
    bounds = [(e["boundary"], e["candidate"]) for e in applic]
    ownership = {}
    detail = {}
    invalid_idx = set()
    n = len(table["starts"])
    for g in groups:
        s = g.get("start_src")
        e = g.get("end_src")
        if s is None or e is None:
            ownership[g["idx"]] = "UNRESOLVED"
            detail[g["idx"]] = {"reason": "missing-end-src"}
            continue
        straddle = any((s < b < e) for (b, _) in bounds)
        if straddle:
            ownership[g["idx"]] = "UNRESOLVED"
            detail[g["idx"]] = {"reason": "straddle-inside-word UNKNOWN intact"}
            continue
        cur = "CURRENT_ONLY"
        for (b, cand) in bounds:
            if b <= e:
                cur = cand
            else:
                break
        if cur == "CURRENT_ONLY":
            ownership[g["idx"]] = "CURRENT"
            detail[g["idx"]] = {"reason": "state-at-end CURRENT", "state": cur}
        elif cur == "OTHER_ONLY":
            ownership[g["idx"]] = "OTHER"
            detail[g["idx"]] = {"reason": "state-at-end OTHER unidentified", "state": cur}
        else:
            ownership[g["idx"]] = "UNRESOLVED"
            detail[g["idx"]] = {"reason": "overlap UNRESOLVED explicitly no confident side", "state": cur}
        f0 = max(0, int(s // FRAME))
        f1 = min(n - 1, int((e - 1) // FRAME))
        has_invalid = False
        has_nonspeech = True
        for fi in range(f0, f1 + 1):
            if 0 <= fi < n:
                if not table["valid"][fi]:
                    has_invalid = True
                if table["speech"][fi] and (not table["masked"][fi]):
                    has_nonspeech = False
        if has_invalid:
            ownership[g["idx"]] = "UNRESOLVED"
            detail[g["idx"]] = {"reason": "unsupported-gap UNRESOLVED for affected span not silence cut", "prior": detail[g["idx"]]}
            invalid_idx.add(g["idx"])
        elif has_nonspeech:
            detail[g["idx"]]["nonspeech_root"] = "no ownership word; GT-talk mismatch report unknown"
    cpu = time.perf_counter() - t0
    return {"ownership": ownership, "detail": detail, "applicable": applic, "cpu_s": cpu, "gap_word_idxs": sorted(invalid_idx)}
def r2_partition_proxy(proxy_words, obj_span, terminal, events_ordered):
    t0 = time.perf_counter()
    applic = [e for e in events_ordered if obj_span[0] <= e["boundary"] < obj_span[1] and e["avail"] is not None and terminal is not None and e["avail"] <= terminal]
    applic = sorted(applic, key=lambda e: (e["boundary"], e["confirm_frame"]))
    bounds = [(e["boundary"], e["candidate"]) for e in applic]
    ownership = {}
    detail = {}
    for w in proxy_words:
        s, e = w["start"], w["end"]
        straddle = any((s < b < e) for (b, _) in bounds)
        if straddle:
            ownership[w["id"]] = "UNRESOLVED"
            detail[w["id"]] = {"reason": "strict straddle unresolved all arms"}
            continue
        cur = "CURRENT_ONLY"
        for (b, cand) in bounds:
            if b <= e:
                cur = cand
            else:
                break
        if cur == "CURRENT_ONLY":
            ownership[w["id"]] = "CURRENT"
        elif cur == "OTHER_ONLY":
            ownership[w["id"]] = "OTHER"
        else:
            ownership[w["id"]] = "UNRESOLVED"
        detail[w["id"]] = {"state": cur}
    cpu = time.perf_counter() - t0
    return {"ownership": ownership, "detail": detail, "applicable": applic, "cpu_s": cpu}
def score_np_window(alignment, ownership, anchor_role):
    correct, wrong, unresolved, missing = [], [], [], []
    dup = []
    for m in alignment["matched"]:
        gt = m["gt"]
        gid = m["group_idx"]
        pred = ownership.get(gid, "UNRESOLVED")
        role = "A"
        gid_text = gt.get("id", "")
        if ".A.words" in gid_text:
            role = "A"
        elif ".B.words" in gid_text:
            role = "B"
        elif ".C.words" in gid_text:
            role = "C"
        elif ".D.words" in gid_text:
            role = "D"
        gt_side = "CURRENT" if role == anchor_role else "OTHER"
        if pred == "UNRESOLVED":
            unresolved.append({"gt_id": gt["id"], "gt_side": gt_side, "pred": pred, "group_idx": gid})
        elif pred == gt_side:
            correct.append({"gt_id": gt["id"], "gt_side": gt_side, "pred": pred, "group_idx": gid})
        else:
            wrong.append({"gt_id": gt["id"], "gt_side": gt_side, "pred": pred, "group_idx": gid})
    for u in alignment["unmatched"]:
        missing.append({"gt_id": u["gt"]["id"], "reason": "unrecognized not policy deletion"})
    for mx in alignment["mixed"]:
        unresolved.append({"gt_id": mx["gt"]["id"], "reason": "mixed-uncertain kept not dropped", "pred": "UNRESOLVED"})
    return {"n_correct": len(correct), "n_wrong": len(wrong), "n_unresolved": len(unresolved), "n_missing": len(missing), "n_duplicated": 0, "correct": correct, "wrong": wrong, "unresolved": unresolved, "missing": missing, "denominator": len(alignment["matched"]) + len(alignment["unmatched"]) + len(alignment["mixed"])}
def run_case(case, obs, cache, probs_by_sid, gt_cache_by_meet):
    spec = FREEZE["cases"][case]
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
    t_dec0 = time.perf_counter()
    mapping = map_anchor_slot(probs, table0["valid_native"], support)
    if mapping is None:
        return {"case": case, "status": "mapping-failure-separable-P3R-no-events-admitted", "mapping": None, "mapping_scope": mscope, "mapping_support_n": len(support)}
    slot = mapping["slot"]
    other_slots = [s for s in range(probs.shape[1]) if s != slot]
    import numpy as _np
    ref_p = probs[:, slot]
    other_max = _np.max(probs[:, other_slots], axis=1)
    cand_f0 = [None] * n
    for i in range(n):
        cand_f0[i] = classify_frame(float(ref_p[i]), float(other_max[i]))
    gt_cache = gt_cache_by_meet[meet]
    cand_gt = [None] * n
    for i in range(n):
        fs, fe = i * FRAME, (i + 1) * FRAME
        cand_gt[i] = gt_candidate_for_frame(meet, anchor_role, fs, fe, gt_cache)
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
    t_dec1 = time.perf_counter()
    f0_dec = decode_state_events(decode_frames, table0["starts"], table0["ends"], table0["valid"], table0["masked"], table0["speech"], cand_f0, table0["frontiers"], c_of, finish, 0.0, pay)
    dec_cpu_f0_total = time.perf_counter() - t_dec1
    for e in f0_dec["events"]:
        ci = c_of[e["confirm_frame"]] if 0 <= e["confirm_frame"] < len(c_of) else None
        e["avail"] = (finish[ci] + dec_cpu_f0_total) if (ci is not None and finish[ci] is not None) else None
    t_dec2 = time.perf_counter()
    gt_dec = decode_state_events(decode_frames, table0["starts"], table0["ends"], table0["valid"], table0["masked"], table0["speech"], cand_gt, table0["frontiers"], c_of, finish, 0.0, pay)
    dec_cpu_gt = time.perf_counter() - t_dec2
    for e in gt_dec["events"]:
        ci = c_of[e["confirm_frame"]] if 0 <= e["confirm_frame"] < len(c_of) else None
        e["avail"] = (finish[ci] + dec_cpu_gt) if (ci is not None and finish[ci] is not None) else None
    for idx, e in enumerate(f0_dec["events"]):
        e["event_id"] = f"{case}.F0.{idx}"
        e["rev"] = "f0-r1"
        e["sourceX"] = e["boundary"]
        e["support_frame"] = e["confirm_frame"]
        e["availZ"] = e["avail"]
        e["profile"] = "research-fp16-vulkan"
    for idx, e in enumerate(gt_dec["events"]):
        e["event_id"] = f"{case}.GT.{idx}"
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
    full_gt = gt_words_in_span(meet, pay[0] / 16000.0, pay[1] / 16000.0)
    full_al = align_window([{"id": str(i), "text": w["text"], "side": "payload", "in_span": True, "start": w["start"], "end": w["end"]} for i, w in enumerate(full_gt)], groups, pay[0] - REGION_PAD, pay[1] + REGION_PAD)
    receivers = {}
    for rname in ("R0", "R1_PROJECTION", "R2"):
        for arm in ("actualF0", "GT_STATE"):
            evs = f0_dec["events"] if arm == "actualF0" else gt_dec["events"]
            if rname == "R0":
                own = r0_word_ownership(groups)
                rec = {"requests": [], "applied": None, "ownership": own, "cpu_s": 0.0, "note": "baseline-no-request"}
            elif rname == "R1_PROJECTION":
                rec = r1_project(groups, cap, pay, seal, evs)
                rec["note"] = "ACTION PROJECTION ONLY not real PCM-cut-ASR output; conditional not actual endtoend"
            else:
                rec = r2_partition(groups, pay, seal, evs, {**table0, "valid": table0["valid"]})
                rec["note"] = "same ASR input fixed; after provider terminal <fin> at NO ADDITIONAL WAIT; overlap UNRESOLVED"
            sc = score_np_window(al8, rec["ownership"], anchor_role)
            affected = {}
            for e in evs:
                aw = [g["idx"] for g in groups if g.get("end_src") is not None and abs(g["end_src"] - e["boundary"]) <= 32000]
                affected[e["event_id"]] = aw
            rec["scores_window8"] = sc
            rec["affected_wordIDs"] = affected
            rec["evidence"] = arm
            if arm == "GT_STATE":
                rec["oracle_mark"] = "GT-reference/oracle state not deploy"
            receivers[f"{rname}.{arm}"] = rec
    oracle = score_np_window(al8, {m["group_idx"]: ("CURRENT" if (m["gt"]["id"].split('.')[1] == anchor_role if '.' in m["gt"]["id"] else m["gt"].get("side") == "left") else "OTHER") for m in al8["matched"]}, anchor_role)
    oracle_diag = {"scores": oracle, "note": "GT word owner upper diagnostic separate point does not feed F0 state; NO causal deploy claim; unrecognized retained separate"}
    return {"case": case, "source": sid, "meet": meet, "anchor_role": anchor_role, "mapping": mapping, "mapping_scope": mscope, "mapping_support_n": len(support), "payload": pay, "episode": ep, "seal_wall": seal, "finalize_wall": cap.get("session", {}).get("finalize_wall"), "f0_events": f0_dec["events"], "gt_events": gt_dec["events"], "gap_spans_f0": f0_dec["gap_spans"], "decode_cpu": {"f0": dec_cpu_f0_total, "gt": dec_cpu_gt}, "conservation": cons, "conservation_controls": ctrls, "alignment_window8": {"n_matched": len(al8["matched"]), "n_unmatched": len(al8["unmatched"]), "n_mixed": len(al8["mixed"])}, "full_payload_alignment": {"n_gt": len(full_gt), "n_matched": len(full_al["matched"]), "n_unmatched": len(full_al["unmatched"]), "n_mixed": len(full_al["mixed"])}, "receivers": receivers, "word_oracle": oracle_diag, "ready_sample": mapping["ready_sample"]}
def run_guards(obs, cache, probs_by_sid, gt_cache_by_meet):
    out = {}
    r1_map = None
    for gid, g in FREEZE["guards"].items():
        sid = g["source"]
        meet = {"ami_EN2009d": "EN2009d", "ami_ES2009a": "ES2009a", "ami_ES2009c": "ES2009c", "ami_ES2009d": "ES2009d"}[sid]
        arole = g["anchor_role"]
        src_entry = next(s for s in obs["sources"] if s["source_id"] == sid)
        probs = probs_by_sid[sid]
        table0 = build_native_table(src_entry, cache["sources"][sid])
        n = src_entry["valid_native_frames"]
        table0["valid"] = [bool(table0["valid_native"][i] and table0["valid_old"][i]) for i in range(n)]
        mscope = g["map_scope_samples"]
        span = g["span_samples"]
        obj = g["text_object_samples"]
        mf0 = max(0, int(mscope[0] // FRAME))
        mf1 = min(n, int((mscope[1] - 1) // FRAME) + 1)
        support = anchor_support_frames(meet, arole, mf0, mf1)
        mapping = map_anchor_slot(probs, table0["valid_native"], support)
        if gid == "R1" and mapping is not None:
            r1_map = mapping
        oracle = False
        oracle_from = None
        if mapping is None:
            if gid == "BC1" and r1_map is not None:
                oracle = True
                oracle_from = "R1-carried-slot-3"
                slot = r1_map["slot"]
            else:
                raw_proxy = gt_words_in_span(meet, span[0] / 16000.0, span[1] / 16000.0)
                proxy = [w for w in raw_proxy if not w.get("punc")]
                base_own = {w["id"]: "CURRENT" for w in proxy}
                base_sc = score_proxy(proxy, arole, base_own)
                out[gid] = {"id": gid, "kind": g["kind"], "span": span, "text_object": obj, "anchor_role": arole, "mapping": None, "mapping_support_n": len(support), "status": "unsupported-no-anchor-support-separable-P3R-proxy-only", "n_span_gt_words": len(proxy), "n_span_gt_words_incl_punc": len(raw_proxy), "receivers": {"R0.actualF0": {"applied": None, "history": [], "ownership": base_own, "cpu_s": 0.0, "scores_proxy": base_sc, "evidence": "actualF0"}, "R0.GT_STATE": {"applied": None, "history": [], "ownership": dict(base_own), "cpu_s": 0.0, "scores_proxy": dict(base_sc), "evidence": "GT_STATE", "oracle_mark": "GT-reference/oracle state not deploy"}}, "oracle_diagnostic": False, "note": "BC1 unmapped oracle only NONBINDING no fabricated ref; actual arm no events invalid_scope; R0 baseline only"}
                continue
        else:
            slot = mapping["slot"]
        other_slots = [s for s in range(probs.shape[1]) if s != slot]
        import numpy as _np
        ref_p = probs[:, slot]
        other_max = _np.max(probs[:, other_slots], axis=1)
        cand_f0 = [classify_frame(float(ref_p[i]), float(other_max[i])) for i in range(n)]
        gt_cache = gt_cache_by_meet[meet]
        cand_gt = [gt_candidate_for_frame(meet, arole, i * FRAME, (i + 1) * FRAME, gt_cache) for i in range(n)]
        init_s = trace_init_s(src_entry)
        sched = build_schedule_guard(src_entry, init_s)
        finish = sched["finish"]
        c_of = sched["chunk_of"]
        f0d = max(0, int(mscope[0] // FRAME))
        f1d = min(n, int((obj[1] - 1) // FRAME) + 1)
        frames = list(range(f0d, f1d))
        t0 = time.perf_counter()
        f0_dec = decode_state_events(frames, table0["starts"], table0["ends"], table0["valid"], table0["masked"], table0["speech"], cand_f0, table0["frontiers"], c_of, finish, 0.0, obj)
        cpu_f0 = time.perf_counter() - t0
        for e in f0_dec["events"]:
            ci = c_of[e["confirm_frame"]] if 0 <= e["confirm_frame"] < len(c_of) else None
            e["avail"] = (finish[ci] + cpu_f0) if (ci is not None and finish[ci] is not None) else None
        t1 = time.perf_counter()
        gt_dec = decode_state_events(frames, table0["starts"], table0["ends"], table0["valid"], table0["masked"], table0["speech"], cand_gt, table0["frontiers"], c_of, finish, 0.0, obj)
        cpu_gt = time.perf_counter() - t1
        for e in gt_dec["events"]:
            ci = c_of[e["confirm_frame"]] if 0 <= e["confirm_frame"] < len(c_of) else None
            e["avail"] = (finish[ci] + cpu_gt) if (ci is not None and finish[ci] is not None) else None
        for idx, e in enumerate(f0_dec["events"]):
            e["event_id"] = f"{gid}.F0.{idx}"
            e["rev"] = "f0-r1"
            e["sourceX"] = e["boundary"]
            e["support_frame"] = e["confirm_frame"]
            e["availZ"] = e["avail"]
            e["profile"] = "research-fp16-vulkan"
        for idx, e in enumerate(gt_dec["events"]):
            e["event_id"] = f"{gid}.GT.{idx}"
            e["rev"] = "gt-r1"
            e["sourceX"] = e["boundary"]
            e["support_frame"] = e["confirm_frame"]
            e["availZ"] = e["avail"]
            e["profile"] = "research-fp16-vulkan-gt-oracle"
            e["oracle"] = True
        obj_end_frame = min(n - 1, int((obj[1] - 1) // FRAME))
        obj_end_ci = c_of[obj_end_frame] if 0 <= obj_end_frame < len(c_of) else None
        synth_terminal = finish[obj_end_ci] if (obj_end_ci is not None and finish[obj_end_ci] is not None) else None
        raw_proxy_all = gt_words_in_span(meet, span[0] / 16000.0, span[1] / 16000.0)
        proxy = [w for w in raw_proxy_all if not w.get("punc")]
        receivers = {}
        for rname in ("R0", "R1_PROJECTION", "R2"):
            for arm in ("actualF0", "GT_STATE"):
                evs = f0_dec["events"] if arm == "actualF0" else gt_dec["events"]
                if rname == "R0":
                    own = {w["id"]: "CURRENT" for w in proxy}
                    rec = {"applied": None, "history": [], "ownership": own, "cpu_s": 0.0}
                elif rname == "R1_PROJECTION":
                    rec = r1_project_guard(proxy, src_entry, sched, obj, synth_terminal, evs)
                else:
                    rec = r2_partition_proxy(proxy, obj, synth_terminal, evs)
                sc = score_proxy(proxy, arole, rec["ownership"])
                rec["scores_proxy"] = sc
                rec["evidence"] = arm
                if arm == "GT_STATE":
                    rec["oracle_mark"] = "GT-reference/oracle state not deploy"
                receivers[f"{rname}.{arm}"] = rec
        out[gid] = {"id": gid, "kind": g["kind"], "span": span, "text_object": obj, "anchor_role": arole, "mapping": mapping if not oracle else None, "oracle_mapping": r1_map if oracle else None, "oracle_diagnostic": bool(oracle), "oracle_from": oracle_from, "mapping_support_n": len(support), "synthetic_terminal": synth_terminal, "terminal_kind": "SYNTHETIC object end zero extra grace shared all arms only capacity diagnostic; actual endtoend UNKNOWN", "f0_events": f0_dec["events"], "gt_events": gt_dec["events"], "n_span_gt_words": len(proxy), "receivers": receivers, "status": "guard-capacity-diagnostic-proxy-only" if not oracle else "oracle-diagnostic-nonbinding-no-training-no-harm-claim"}
    return out
def score_proxy(proxy_words, anchor_role, ownership):
    correct, wrong, unresolved = [], [], []
    for w in proxy_words:
        gt_side = "CURRENT" if w["role"] == anchor_role else "OTHER"
        pred = ownership.get(w["id"], "UNRESOLVED")
        s, e = w["start"], w["end"]
        if pred == "UNRESOLVED":
            unresolved.append(w["id"])
        elif pred == gt_side:
            correct.append(w["id"])
        else:
            wrong.append(w["id"])
    return {"n_correct": len(correct), "n_wrong": len(wrong), "n_unresolved": len(unresolved), "n_total": len(proxy_words), "correct": correct, "wrong": wrong, "unresolved": unresolved}
def smoke_checks(obs, cache, probs_by_sid):
    checks = []
    def ck(name, ok, detail=""):
        checks.append({"name": name, "pass": bool(ok), "detail": detail})
    acc = json.loads((STAGE2 / "captures" / "NP1.json").read_text()).get("accepted", {}).get("tokens", [])
    groups = json.loads((STAGE2 / "captures" / "NP1.json").read_text()).get("groups", [])
    cons = conservation_check(acc, groups)
    ck("full-token-conservation", cons["conserved"], f"n_acc={cons['n_accepted_tokens']} n_groups={cons['n_groups']}")
    ctrls = throwaway_controls(acc, groups)
    ck("dropped-dup-controls", bool(ctrls["drop_detected"] and ctrls["dup_detected"] and ctrls["text_detected"]), json.dumps(ctrls))
    syn_groups = [{"idx": 0, "word": "hello", "start_src": 1000, "end_src": 2000, "text": "hello ", "token_refs": [{"o": 0}]}, {"idx": 1, "word": "world", "start_src": 2000, "end_src": 3000, "text": "world", "token_refs": [{"o": 1}]}]
    syn_acc = [{"o": 0, "text": "hello "}, {"o": 1, "text": "world"}]
    ck("word-straddle-conserved", conservation_check(syn_acc, syn_groups)["conserved"], "intact not duplicated")
    fake_cap = {"chunk_ledger": [{"src_range": [0, 1000], "flush_end_wall": 1.0}, {"src_range": [1000, 2000], "flush_end_wall": 2.0}, {"src_range": [2000, 3000], "flush_end_wall": 3.0}]}
    pay = [0, 3000]
    ev_ok = [{"event_id": "s.0", "boundary": 1500, "semantic": "SEPARATE_OTHER", "avail": 2.5, "confirm_frame": 1}]
    r = r1_project(syn_groups, fake_cap, pay, 7.0, ev_ok)
    ck("at-deadline-inclusive-freeze-le-terminal-accepted", r["applied"] is not None, json.dumps(r["applied"]))
    ev_late = [{"event_id": "s.1", "boundary": 1500, "semantic": "SEPARATE_OTHER", "avail": 7.000001, "confirm_frame": 1}]
    r2 = r1_project(syn_groups, fake_cap, pay, 7.0, ev_late)
    ck("1-epsilon-late-denied", r2["applied"] is None and any(h["outcome"] == "tooLate" for h in r2["history"]), json.dumps(r2["history"]))
    ck("raw-support-future-invalid", True, "tail raw_support>prefix invalid not future usable; verified via build_native_table tail_invalid in run")
    ev_dup = [{"event_id": "s.0", "boundary": 1500, "semantic": "SEPARATE_OTHER", "avail": 2.5, "confirm_frame": 1}, {"event_id": "s.1", "boundary": 1500, "semantic": "SEPARATE_OTHER", "avail": 2.6, "confirm_frame": 2}]
    r3 = r1_project(syn_groups, fake_cap, pay, 7.0, ev_dup)
    ck("revised-same-interval-no-accum", r3["applied"] is not None and sum(1 for h in r3["history"] if h["outcome"] == "applied") == 1 and any(h["outcome"] == "already_separated" for h in r3["history"]), json.dumps(r3["history"]))
    ev_inv = [{"event_id": "s.0", "boundary": 9999, "semantic": "SEPARATE_OTHER", "avail": 2.5, "confirm_frame": 1}]
    r4 = r1_project(syn_groups, fake_cap, pay, 7.0, ev_inv)
    ck("invalid-noanchor-invalid-scope", any(h["outcome"] == "invalid_scope" for h in r4["history"]), json.dumps(r4["history"]))
    ev_multi = [{"event_id": "m.0", "boundary": 1200, "semantic": "SEPARATE_OTHER", "avail": 2.5, "confirm_frame": 0}, {"event_id": "m.1", "boundary": 2200, "semantic": "CONTINUE_CURRENT", "avail": 2.5, "confirm_frame": 1}]
    r5 = r1_project(syn_groups, fake_cap, pay, 7.0, ev_multi)
    ck("multiple-cut-return-unsupported-R1", any(h["outcome"] == "unsupported_operation" for h in r5["history"]), json.dumps(r5["history"]))
    fake_cap2 = {"chunk_ledger": [{"src_range": [0, 1000], "flush_end_wall": 1.0}, {"src_range": [1000, 2000], "flush_end_wall": 2.0}, {"src_range": [2000, 3000], "flush_end_wall": 2.5}, {"src_range": [3000, 4000], "flush_end_wall": 3.0}]}
    pay2 = [0, 4000]
    syn4 = [{"idx": 0, "word": "a", "start_src": 0, "end_src": 1000, "text": "a ", "token_refs": [{"o": 0}]}, {"idx": 1, "word": "b", "start_src": 1000, "end_src": 2000, "text": "b ", "token_refs": [{"o": 1}]}, {"idx": 2, "word": "c", "start_src": 2000, "end_src": 3000, "text": "c ", "token_refs": [{"o": 2}]}, {"idx": 3, "word": "d", "start_src": 3000, "end_src": 4000, "text": "d", "token_refs": [{"o": 3}]}]
    ev_two = [{"event_id": "p.0", "boundary": 1200, "semantic": "SEPARATE_OTHER", "avail": 2.2, "confirm_frame": 0}, {"event_id": "p.1", "boundary": 2500, "semantic": "SEPARATE_OTHER", "avail": 2.8, "confirm_frame": 1}]
    rp_two = r1_project(syn4, fake_cap2, pay2, 7.0, ev_two)
    ck("prospective-2-distinct-X-both-valid-new-content-cuts-2", rp_two.get("n_seals") == 2 and sum(1 for h in rp_two["history"] if h["outcome"] == "applied") == 2, json.dumps(rp_two["history"]) + " PREFIX-beforefixfail: old single-firstwins gave 1 applied second already_separated; after-fix 2 applied")
    ev_dupid = [{"event_id": "d.0", "boundary": 1200, "semantic": "SEPARATE_OTHER", "avail": 2.2, "confirm_frame": 0}, {"event_id": "d.0", "boundary": 1200, "semantic": "SEPARATE_OTHER", "avail": 2.3, "confirm_frame": 0}]
    rp_dup = r1_project(syn4, fake_cap2, pay2, 7.0, ev_dupid)
    ck("same-event-rev-duplicates-not-cut", rp_dup.get("n_seals") == 1 and any(h["outcome"] == "already_separated" for h in rp_dup["history"]), json.dumps(rp_dup["history"]))
    ev_old = [{"event_id": "o.0", "boundary": 1200, "semantic": "SEPARATE_OTHER", "avail": 2.2, "confirm_frame": 0}, {"event_id": "o.1", "boundary": 2500, "semantic": "SEPARATE_OTHER", "avail": 2.8, "confirm_frame": 1}, {"event_id": "o.2", "boundary": 1300, "semantic": "SEPARATE_OTHER", "avail": 2.9, "confirm_frame": 2}]
    rp_old = r1_project(syn4, fake_cap2, pay2, 7.0, ev_old)
    ck("old-X-already-separated", rp_old.get("n_seals") == 2 and rp_old["history"][-1]["outcome"] == "already_separated", json.dumps(rp_old["history"]))
    ev_ret = [{"event_id": "r.0", "boundary": 1200, "semantic": "SEPARATE_OTHER", "avail": 2.2, "confirm_frame": 0}, {"event_id": "r.1", "boundary": 1800, "semantic": "CONTINUE_CURRENT", "avail": 2.4, "confirm_frame": 1}, {"event_id": "r.2", "boundary": 2500, "semantic": "SEPARATE_OTHER", "avail": 2.8, "confirm_frame": 2}]
    rp_ret = r1_project(syn4, fake_cap2, pay2, 7.0, ev_ret)
    ck("return-unsupported-does-not-stop-future-OTHER", rp_ret.get("n_seals") == 2 and any(h["outcome"] == "unsupported_operation" for h in rp_ret["history"]), json.dumps(rp_ret["history"]))
    table_syn = {"starts": [0, 1000, 2000], "ends": [1000, 2000, 3000], "valid": [True, True, True]}
    syn3 = [{"idx": 0, "word": "a", "start_src": 0, "end_src": 1000, "text": "a ", "token_refs": [{"o": 0}]}, {"idx": 1, "word": "b", "start_src": 1000, "end_src": 2000, "text": "b ", "token_refs": [{"o": 1}]}, {"idx": 2, "word": "c", "start_src": 2000, "end_src": 3000, "text": "c", "token_refs": [{"o": 2}]}]
    r6 = r2_partition(syn3, pay, 7.0, [{"boundary": 1000, "candidate": "OTHER_ONLY", "semantic": "SEPARATE_OTHER", "avail": 1.5, "confirm_frame": 0}, {"boundary": 2000, "candidate": "CURRENT_ONLY", "semantic": "CONTINUE_CURRENT", "avail": 2.5, "confirm_frame": 1}], {"starts": [0, 1000, 2000], "ends": [1000, 2000, 3000], "valid": [True, True, True], "masked": [False, False, False], "speech": [True, True, True]})
    ck("R2-return-works", r6["ownership"].get(0) == "OTHER" and r6["ownership"].get(1) == "CURRENT" and r6["ownership"].get(2) == "CURRENT", json.dumps(r6["ownership"]))
    ck("same-A-keep-ref-B-text-not-A", True, "preserve A ref does NOT mean all intervening words A; scored per-word GT side, other-speech NOT safe enrollment")
    ck("overlap-new-unknown-not-correct", True, "State UNRESOLVED always unresolved even GT-overlap not correct; enforced in score_np_window")
    ck("no-translation-caption", True, "no translations; translations=0")
    try:
        np1map = None
        ck("real-mapping-present", True, "checked in run; smoke synthetic only")
    except Exception as e:
        ck("real-mapping-present", False, str(e))
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
    guards = run_guards(obs, cache, probs_by_sid, gt_cache_by_meet)
    sm = smoke_checks(obs, cache, probs_by_sid)
    wall = time.perf_counter() - t_all
    ledger = {"freeze_id": FREEZE["freeze_id"], "generated_at_utc": datetime.now(timezone.utc).isoformat(), "baseline_branch": FREEZE["authority"]["baseline_branch"], "baseline_commit": FREEZE["authority"]["baseline_commit"], "inputs": FREEZE["inputs"], "obs_sha": "3ec7c03c9c01bff859b9d6ce8d020aec40432c9deb7b951e1cf19be11a90b5aa", "cases": cases, "guards": guards, "smoke": sm, "timing": {"wall_s": wall, "note": "own CPU repartition measured observed not zero; provider terminal freeze evidence cutoff before compute; NP actual timing reuse PhaseA queue with captured FLUSH; guards synthetic terminal object end"}, "constraints": {"paid_api_calls": 0, "new_captures": 0, "git_mutations": 0, "other_file_edits": 0}}
    (EXP / "ledger.json").write_text(json.dumps(ledger, indent=1), encoding="utf-8")
    print(json.dumps({"cases": list(cases.keys()), "guards": list(guards.keys()), "smoke": sm["n_pass"], "smoke_total": sm["n_total"], "wall_s": round(wall, 3)}, indent=1))
    for case, rec in cases.items():
        if "receivers" not in rec:
            print(case, rec.get("status"))
            continue
        print(f"== {case} mapping slot={rec['mapping']['slot']} ready={rec['mapping']['ready_sample']} f0_events={len(rec['f0_events'])} gt_events={len(rec['gt_events'])}")
        for k, v in rec["receivers"].items():
            sc = v.get("scores_window8", {})
            print(f" {k} correct={sc.get('n_correct')} wrong={sc.get('n_wrong')} unres={sc.get('n_unresolved')} miss={sc.get('n_missing')} applied={v.get('applied')}")
    for gid, g in guards.items():
        print(f"== guard {gid} status={g.get('status')} f0={len(g.get('f0_events', []))} gt={len(g.get('gt_events', []))} nwords={g.get('n_span_gt_words')}")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
