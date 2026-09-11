"""Phase A live headroom consumer on frozen real observations."""
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
OBSID = "phase_a.observations.v1"
BARRIER_SHA = "3ec7c03c9c01bff859b9d6ce8d020aec40432c9deb7b951e1cf19be11a90b5aa"
NS = "{http://nite.sourceforge.net/}"
F0_TAU = 0.5
CONFIRMATION = 1600
SENS = 1280
CASES = ("NP1", "NP2", "NP3")
SID_OF = {"NP1": "ami_ES2009c", "NP2": "ami_ES2009d", "NP3": "ami_ES2002b"}
MEET_OF = {"NP1": "ES2009c", "NP2": "ES2009d", "NP3": "ES2002b"}
ANCHOR_ROLE = {"NP1": "B", "NP2": "B", "NP3": "D"}
ZIP_PATH = Path("C:/Users/salee/AppData/Local/Temp/psem-ami-annotations/ami_public_manual_1.6.2.zip")
from experiments.psem_decision_sufficiency.replay import align_window as ds_align_window, partition_ownership as ds_partition_ownership, REGION_PAD
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1048576), b""):
            h.update(blk)
    return h.hexdigest()
def load_np(path, shape):
    import numpy as np
    raw = np.fromfile(str(path), dtype=np.float32)
    return raw.reshape(shape[0], shape[1])
def validate_obs_bundle(obs):
    errs = []
    if obs.get("schema") != OBSID:
        errs.append("schema-mismatch")
    for s in obs.get("sources", []):
        for kind in ("probs", "hidden", "logits"):
            ff = s["feature_files"][kind]
            p = ROOT / ff["path"]
            if not p.exists():
                errs.append(s["source_id"] + "-" + kind + "-missing")
                continue
            if sha256_file(p) != ff["sha256"]:
                errs.append(s["source_id"] + "-" + kind + "-sha-mismatch")
        if sha256_file(ROOT / s["trace_path"]) != s["trace_sha256"]:
            errs.append(s["source_id"] + "-trace-sha-mismatch")
        n = s["feature_files"]["probs"]["shape"][0]
        if s["valid_native_frames"] != n:
            errs.append(s["source_id"] + "-valid-frames-mismatch")
        if s["prefix_end_sample"] % 1280 != 0:
            errs.append(s["source_id"] + "-prefix-not-multiple-1280")
        if not s["prefix_end_sample"] < s["source_total_samples"]:
            errs.append(s["source_id"] + "-prefix-tail-not-artificial")
        tot = 0
        for c in s["chunks"]:
            if c["emit_start_frame"] != tot:
                errs.append(s["source_id"] + "-continuity-break")
            tot += c["emit_count"]
            if c.get("mel_parity_maxabs", 0) != 0:
                errs.append(s["source_id"] + "-parity-nonzero")
        if tot != n:
            errs.append(s["source_id"] + "-emit-total-mismatch")
        if s["chunks"][-1]["raw_support_end_sample"] != s["prefix_end_sample"] + 96:
            errs.append(s["source_id"] + "-tail-not-N-plus-96")
    return errs
def load_gt_words_nopunc(meet, role):
    local = STAGE2 / "annotations" / "words" / (meet + "." + role + ".words.xml")
    data = local.read_bytes() if local.exists() else zipfile.ZipFile(str(ZIP_PATH)).read("words/" + meet + "." + role + ".words.xml")
    root = ET.fromstring(data)
    out = []
    for w in root.iter():
        if w.tag.split("}")[-1] != "w":
            continue
        a = w.attrib
        if a.get("punc", "") == "true":
            continue
        out.append({"id": a.get(NS + "id", ""), "start": float(a.get("starttime", -1)), "end": float(a.get("endtime", -1)), "text": (w.text or ""), "role": role})
    return out
def gt_window_samples(case):
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
        for w in load_gt_words_nopunc(meet, role):
            if w["end"] > lo_s and w["start"] < hi_s:
                out.append({"role": role, "id": w["id"], "text": w["text"], "start": int(round(w["start"] * 16000)), "end": int(round(w["end"] * 16000))})
    return sorted(out, key=lambda w: (w["start"], w["end"]))
align_window = ds_align_window
partition_ownership = ds_partition_ownership
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
    import copy
    drop = copy.deepcopy(groups)
    if drop and drop[0].get("token_refs"):
        drop[0]["token_refs"] = drop[0]["token_refs"][1:]
    dup = copy.deepcopy(groups)
    if dup:
        dup.append(copy.deepcopy(dup[0]))
    txt = copy.deepcopy(groups)
    if txt:
        txt[0]["text"] = txt[0].get("text", "") + "X"
    return {"drop_detected": not conservation_check(accepted_tokens, drop)["conserved"], "dup_detected": not conservation_check(accepted_tokens, dup)["conserved"], "text_detected": not conservation_check(accepted_tokens, txt)["conserved"], "note": "throwaway behavior copies discarded after check never permanent load"}
def build_native_table(src_entry, old_rows):
    n = src_entry["valid_native_frames"]
    chunks = src_entry["chunks"]
    ctab = []
    for c in chunks:
        for k in range(c["emit_count"]):
            ctab.append(c)
    starts = [i * 1280 for i in range(n)]
    ends = [(i + 1) * 1280 for i in range(n)]
    frontiers = [c["raw_support_end_sample"] for c in ctab[:n]]
    masked, speech, valid_old, old_reason, old_ep = [], [], [], [], []
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
            old_reason.append("no-old-support")
            old_ep.append(None)
        else:
            masked.append(bool(best["m"]))
            speech.append(bool(best["sp"]))
            valid_old.append(bool(best["v"]))
            old_reason.append("old-join" if best["v"] else "old-invalid")
            old_ep.append(best["ep"])
    tail_unsupported = [(i == n - 1) for i in range(n)]
    valid_native = [(i < n - 1) for i in range(n)]
    return {"starts": starts, "ends": ends, "frontiers": frontiers, "masked": masked, "speech": speech, "valid_old": valid_old, "old_reason": old_reason, "old_ep": old_ep, "tail_unsupported": tail_unsupported, "valid_native": valid_native, "chunks": chunks}
def anchor_support_frames(meet, arole, f0, f1):
    words = {}
    for role in ("A", "B", "C", "D"):
        words[role] = load_gt_words_nopunc(meet, role)
    support = []
    for i in range(f0, f1):
        fs, fe = i * 1280, (i + 1) * 1280
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
        if not valid_native[i]:
            prev, consec = None, 0
            continue
        s = int(am[i])
        consec = consec + 1 if s == prev else 1
        prev = s
        if consec >= 2:
            found = {"slot": s, "ready_frames": [i - 1, i], "ready_sample": (i + 1) * 1280}
            break
    return found
def f0_fire_events(frames, starts, ends, valid, masked, speech, scores, frontiers, tau, confirmation, latch):
    out = []
    latched, pending, pend_n, prev_end = False, None, 0, None
    for i in frames:
        if not valid[i]:
            pending, pend_n = None, 0
            continue
        if masked[i]:
            continue
        s, e = starts[i], ends[i]
        if not speech[i]:
            continue
        if scores[i] < tau:
            if latched:
                latched, pending, pend_n, prev_end = False, None, 0, None
            else:
                pending, pend_n = None, 0
            continue
        if latched:
            continue
        if prev_end is not None and s != prev_end:
            pending, pend_n = None, 0
        if pending is None:
            pending = s
        dur = e - s
        need = confirmation - pend_n
        if dur >= need:
            q = s + need
            fr = frontiers[i]
            emit = q if q >= fr else fr
            out.append({"boundary": int(pending), "frontier": int(fr), "emit": int(emit), "frame": int(i)})
            if not latch:
                return out
            latched, pending, pend_n, prev_end = True, None, 0, e
            continue
        pend_n += dur
        prev_end = e
    return out
def chunk_index_of_frame(src_entry, frame):
    for idx, c in enumerate(src_entry["chunks"]):
        if c["emit_start_frame"] <= frame < c["emit_start_frame"] + c["emit_count"]:
            return idx
    return None
def trace_init_s(src_entry):
    h = json.loads((ROOT / src_entry["trace_path"]).read_text(encoding="utf-8"))
    return (h.get("load_us", 0) + h.get("sched_setup_us", 0)) / 1000000.0
def build_schedule(src_entry, cap, pay, init_s):
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
    zero_ci = chunk_index_of_frame(src_entry, int(P0 // 1280))
    zero_finish = finish[zero_ci]
    c_of = []
    for i in range(src_entry["valid_native_frames"]):
        c_of.append(chunk_index_of_frame(src_entry, i))
    tail_invalid = [bool(c["raw_support_end_sample"] > src_entry["prefix_end_sample"]) for c in chunks]
    return {"release": release, "finish": finish, "zero_ci": zero_ci, "zero_finish": zero_finish, "init_s": init_s, "chunk_of_frame": c_of, "tail_invalid": tail_invalid}
def load_capture(case):
    return json.loads((STAGE2 / "captures" / (case + ".json")).read_text(encoding="utf-8"))
def word_id_time_map(meet):
    table = {}
    for role in ("A", "B", "C", "D"):
        local = STAGE2 / "annotations" / "words" / (meet + "." + role + ".words.xml")
        data = local.read_bytes() if local.exists() else zipfile.ZipFile(str(ZIP_PATH)).read("words/" + meet + "." + role + ".words.xml")
        root = ET.fromstring(data)
        for w in root.iter():
            if w.tag.split("}")[-1] != "w":
                continue
            a = w.attrib
            if a.get("punc", "") == "true":
                continue
            nid = a.get(NS + "id", "")
            table[meet + "." + role + ".words.xml#" + nid] = (float(a.get("starttime", -1)), float(a.get("endtime", -1)), (w.text or ""), role)
    return table
def qa_merge_for_case(case, groups):
    meet = MEET_OF[case]
    try:
        data = zipfile.ZipFile(str(ZIP_PATH)).read("dialogueActs/" + meet + ".adjacency-pairs.xml")
    except KeyError:
        return {"status": "null-no-dialogueacts-members", "reason": "archive holds no dialogueActs members for " + meet, "n_pairs": 0, "n_resolved": 0, "grounded": []}
    root = ET.fromstring(data)
    pairs = []
    for ap in root.iter():
        if ap.tag.split("}")[-1] != "adjacency-pair":
            continue
        src, tgt = None, None
        for p in ap:
            if p.tag.split("}")[-1] != "pointer":
                continue
            href = p.attrib.get("href", "")
            if p.attrib.get("role") == "source":
                src = href
            elif p.attrib.get("role") == "target":
                tgt = href
        if src is not None and tgt is not None:
            pairs.append((src, tgt))
    if not pairs:
        return {"status": "null-no-sourced-target-pairs", "reason": "member present but no adjacency pair carries both source and target pointers", "n_pairs": 0, "n_resolved": 0, "grounded": []}
    wmap = word_id_time_map(meet)
    dact_cache = {}
    def dact_span(href):
        if "#" not in href:
            return None
        fname, frag = href.split("#", 1)
        if not frag.startswith("id(") or not frag.endswith(")"):
            return None
        did = frag[3:-1]
        if fname not in dact_cache:
            try:
                ddata = zipfile.ZipFile(str(ZIP_PATH)).read("dialogueActs/" + fname)
            except KeyError:
                return None
            droot = ET.fromstring(ddata)
            table = {}
            for dact in droot.iter():
                if dact.tag.split("}")[-1] != "dact":
                    continue
                key = dact.attrib.get(NS + "id", dact.attrib.get("id", ""))
                kids = []
                for ch in dact:
                    if ch.tag.split("}")[-1] == "child":
                        kids.append(ch.attrib.get("href", ""))
                table[key] = kids
                table[key.split(".dialog-act.")[-1]] = kids
            dact_cache[fname] = table
        table = dact_cache[fname]
        kids = table.get(did)
        if not kids:
            for k, v in table.items():
                if k.endswith(did.split(".")[-1]):
                    kids = v
                    break
        if not kids:
            return None
        times = []
        for k in kids:
            if ".." in k:
                a0, a1 = k.split("..", 1)
                f0, p0 = a0.split("#id(")[0], a0.split("#id(")[-1].rstrip(")")
                f1, p1 = a1.split("id(")[0].rstrip("#"), a1.split("id(")[-1].rstrip(")")
                d0 = ""
                for ch in reversed(p0):
                    if ch.isdigit():
                        d0 = ch + d0
                    else:
                        break
                d1 = ""
                for ch in reversed(p1):
                    if ch.isdigit():
                        d1 = ch + d1
                    else:
                        break
                if not d1:
                    d1 = d0
                pre = p0[:len(p0) - len(d0)] if d0 else p0
                for nn in range(min(int(d0 or -1), int(d1 or -1)), max(int(d0 or -1), int(d1 or -1)) + 1):
                    t = wmap.get(f0 + "#" + pre + str(nn))
                    if t is not None:
                        times.append(t)
            else:
                if "#id(" in k:
                    f2, p2 = k.split("#id(")
                    t = wmap.get(f2 + "#" + p2.rstrip(")"))
                else:
                    t = wmap.get(k)
                if t is not None:
                    times.append(t)
        if not times:
            return None
        ss = [t[0] for t in times]
        ee = [t[1] for t in times]
        return (min(ss), max(ee))
    spec = FREEZE["cases"][case]
    lo, hi = spec["scored_span_samples"]
    gt8 = gt_window_samples(case)
    wlo = min(w["start"] for w in gt8) - REGION_PAD
    whi = max(w["end"] for w in gt8) + REGION_PAD
    left_lo = min(w["start"] for w in gt8 if w["side"] == "left")
    left_hi = max(w["end"] for w in gt8 if w["side"] == "left")
    right_lo = min(w["start"] for w in gt8 if w["side"] == "right")
    right_hi = max(w["end"] for w in gt8 if w["side"] == "right")
    al8 = align_window(gt8, groups, wlo, whi)
    gmap = {}
    for m in al8["matched"]:
        gmap[m["gt"]["id"]] = m
    resolved, grounded = 0, []
    for src, tgt in pairs:
        s0, t0 = dact_span(src), dact_span(tgt)
        if s0 is None or t0 is None:
            continue
        resolved += 1
        covers = (s0[0] < left_hi and s0[1] > left_lo and t0[0] < right_hi and t0[1] > right_lo) or (t0[0] < left_hi and t0[1] > left_lo and s0[0] < right_hi and s0[1] > right_lo)
        if covers:
            grounded.append({"source": src, "target": tgt, "source_span_s": list(s0), "target_span_s": list(t0)})
    return {"status": "measured" if grounded else "null-no-grounded-pair-in-window", "reason": "resolved both ends against word times; grounded only where one part overlaps the left GT window and the other overlaps the right GT window" if not grounded else "grounded pairs joined on mapped accepted texts", "n_pairs": len(pairs), "n_resolved": resolved, "grounded": grounded}
def support_in_span(cache_rows, lo, hi):
    idx = [r for r in cache_rows if r["e"] > lo and r["s"] < hi]
    return {"n_frames_overlap": len(idx), "n_valid": sum(1 for r in idx if r["v"]), "n_masked": sum(1 for r in idx if r["m"]), "n_speech": sum(1 for r in idx if r["sp"])}
def run_positive(case, obs, cache, probs_by_sid):
    spec = FREEZE["cases"][case]
    sid = SID_OF[case]
    meet = MEET_OF[case]
    arole = ANCHOR_ROLE[case]
    src_entry = next(s for s in obs["sources"] if s["source_id"] == sid)
    probs = probs_by_sid[sid]
    table = build_native_table(src_entry, cache["sources"][sid])
    n = src_entry["valid_native_frames"]
    ep = spec["episode_span_samples"]
    pay = spec["payload_samples"]
    b = spec["boundary_samples"]
    if case in ("NP1", "NP3"):
        mlo, mhi = ep[0], pay[0]
    else:
        mlo, mhi = ep[0], b
    mf0, mf1 = int(mlo // 1280), int((mhi - 1) // 1280) + 1
    support = anchor_support_frames(meet, arole, mf0, mf1)
    mapping = map_anchor_slot(probs, table["valid_native"], support)
    if mapping is None:
        return {"case": case, "source": sid, "mapping": None, "mapping_scope": [mlo, mhi], "mapping_support_n": len(support), "status": "mapping-failure-separable-P3R-no-events-admitted"}
    slot = mapping["slot"]
    scores = [float(1 - probs[i, slot]) for i in range(n)]
    speakers = ["anchor"] * n
    valid = [bool(table["valid_native"][i] and table["valid_old"][i]) for i in range(n)]
    f0, f1 = int(ep[0] // 1280), int((ep[1] - 1) // 1280) + 1
    ep_frames = list(range(f0, f1))
    ev_orig = f0_fire_events(ep_frames, table["starts"], table["ends"], valid, table["masked"], table["speech"], scores, table["frontiers"], F0_TAU, CONFIRMATION, False)
    ev_alt = f0_fire_events(ep_frames, table["starts"], table["ends"], valid, table["masked"], table["speech"], scores, table["frontiers"], F0_TAU, CONFIRMATION, True)
    first = ev_orig[0] if ev_orig else None
    cap = load_capture(case)
    init_s = trace_init_s(src_entry)
    sched = build_schedule(src_entry, cap, pay, init_s)
    backlog = sched["zero_finish"]
    ch_of = sched["chunk_of_frame"]
    fin = sched["finish"]
    ctail = sched["tail_invalid"]
    table["tail_unsupported"] = [bool(ctail[ch_of[i]]) if ch_of[i] is not None else True for i in range(n)]
    table["valid_native"] = [not x for x in table["tail_unsupported"]]
    table["n_tail_invalid"] = sum(1 for x in table["tail_unsupported"] if x)
    def event_avail(frame_idx, cpu):
        ci = ch_of[frame_idx] if 0 <= frame_idx < len(ch_of) else None
        if ci is None:
            return None, None, "UNKNOWN-no-chunk"
        if fin[ci] is None:
            return None, ci, "UNKNOWN-support-past-payload"
        return fin[ci] + cpu, ci, "measured"
    seal = cap.get("session", {}).get("seal_wall")
    groups = cap.get("groups", [])
    lo, hi = spec["scored_span_samples"]
    gt8 = gt_window_samples(case)
    wlo = min(w["start"] for w in gt8) - REGION_PAD
    whi = max(w["end"] for w in gt8) + REGION_PAD
    full_gt = gt_words_in_span(meet, pay[0] / 16000.0, pay[1] / 16000.0)
    full_al = align_window([{"id": str(i), "text": w["text"], "side": "payload", "in_span": True, "start": w["start"], "end": w["end"]} for i, w in enumerate(full_gt)], groups, pay[0] - REGION_PAD, pay[1] + REGION_PAD)
    right_exposure = [w for w in full_gt if w["start"] >= b and not any(m["gt"]["start"] == w["start"] and m["gt"]["end"] == w["end"] for m in full_al["matched"])]
    def scope_of(emit):
        if emit is None:
            return "no-event"
        if emit < pay[0] or emit >= pay[1]:
            return "invalid_scope"
        return "valid"
    t_cpu0 = time.perf_counter()
    f0_avail, f0_scope, f0_ci, f0_rel = None, "no-event", None, None
    if first is not None:
        f0_avail, f0_ci, f0_rel = event_avail(first["frame"], 0.0)
        f0_scope = scope_of(first["emit"])
    f0_decision_cpu = time.perf_counter() - t_cpu0
    if f0_avail is not None:
        f0_avail = f0_avail + f0_decision_cpu
    ctrl_frame = (b + CONFIRMATION) // 1280
    ctrl_ci = ch_of[int(ctrl_frame)] if 0 <= int(ctrl_frame) < len(ch_of) else None
    ctrl_base, ctrl_rel = None, None
    if ctrl_ci is not None:
        if fin[ctrl_ci] is None:
            ctrl_rel = "UNKNOWN-support-past-payload"
        else:
            ctrl_base = fin[ctrl_ci]
            ctrl_rel = "measured"
    t_cpu1 = time.perf_counter()
    part_probe = partition_ownership(align_window(gt8, groups, wlo, whi), b)
    ctrl_cpu = time.perf_counter() - t_cpu1
    ctrl_avail = (ctrl_base + ctrl_cpu) if ctrl_base is not None else None
    ctrl_status = "UNKNOWN" if ctrl_avail is None or seal is None else ("too_late" if ctrl_avail > seal else "legal-measured-virtual-profile")
    ctrl_mpl = (seal - ctrl_base) if (ctrl_base is not None and seal is not None) else None
    arms = {}
    req_lists = {"none": [], "f0_original": ([{"boundary": first["boundary"], "emit": first["emit"], "frontier": first["frontier"], "scope": f0_scope, "avail": f0_avail, "cpu": f0_decision_cpu, "kind": "f0-single-fire", "release": f0_rel}] if first is not None else []), "control_capacity": [{"boundary": b, "emit": b + CONFIRMATION, "frontier": int(ctrl_frame) * 1280, "scope": "valid", "avail": ctrl_avail, "cpu": ctrl_cpu, "kind": "control-capacity", "release": ctrl_rel}]}
    for arm in ("none", "f0_original", "control_capacity"):
        applied, history = None, []
        for req in req_lists[arm]:
            if req["scope"] != "valid":
                history.append({"request": {k: req[k] for k in ("boundary", "emit", "frontier", "kind")}, "outcome": req["scope"], "applied_boundary": applied["boundary"] if applied else None})
                continue
            if req["avail"] is None or seal is None:
                history.append({"request": {k: req[k] for k in ("boundary", "emit", "frontier", "kind")}, "outcome": "UNKNOWN", "applied_boundary": applied["boundary"] if applied else None})
                continue
            if req["avail"] > seal:
                history.append({"request": {k: req[k] for k in ("boundary", "emit", "frontier", "kind")}, "outcome": "too_late", "applied_boundary": applied["boundary"] if applied else None, "availability": req["avail"], "deadline": seal})
                continue
            if applied is None:
                applied = {"boundary": req["boundary"], "availability": req["avail"], "request": {k: req[k] for k in ("boundary", "emit", "frontier", "kind")}}
                history.append({"request": {k: req[k] for k in ("boundary", "emit", "frontier", "kind")}, "outcome": "applied", "applied_boundary": req["boundary"], "availability": req["avail"], "deadline": seal})
            else:
                history.append({"request": {k: req[k] for k in ("boundary", "emit", "frontier", "kind")}, "outcome": "already_separated", "applied_boundary": applied["boundary"]})
        ab = applied["boundary"] if applied else None
        al8 = align_window(gt8, groups, wlo, whi)
        part8 = partition_ownership(al8, ab)
        sens = {}
        for delta in (-SENS, SENS):
            bb = None if ab is None else ab + delta
            sens[str(delta)] = partition_ownership(al8, bb)["error_interval"]
        for w in part8["wrong_definite"]:
            w["boundary_overshoot"] = w["group_end_src"] - ab if ab is not None else None
            w["asr_gt_diff"] = w["group_end_src"] - w["gt"]["end"]
        if ab is None:
            acc_part = None
        else:
            left_groups = [g for g in groups if g.get("end_src") is not None and g["end_src"] <= ab]
            right_groups = [g for g in groups if g.get("end_src") is not None and g["end_src"] > ab]
            acc_part = {"n_left": len(left_groups), "n_right": len(right_groups), "left_text": "".join(g.get("text", "") for g in left_groups), "right_text": "".join(g.get("text", "") for g in right_groups)}
        arms[arm] = {"requests": history, "applied_boundary": ab, "receipt_5": {"request": req_lists[arm][0] if req_lists[arm] else None, "availability": applied["availability"] if applied else None, "applied_boundary": ab, "partition": history, "timing": {"added_latency_cpu": (f0_decision_cpu if arm == "f0_original" else ctrl_cpu) if arm != "none" else None, "max_permissible_lag": (ctrl_mpl if arm == "control_capacity" else ((seal - f0_avail) if (arm == "f0_original" and f0_scope == "valid" and f0_avail is not None and seal is not None) else None))}}, "deadline": seal, "accepted_partition": acc_part, "ownership_8": part8, "ownership_8_sens_pm1280": sens, "n_alignment_matched": len(al8["matched"]), "n_alignment_unmatched": len(al8["unmatched"]), "n_alignment_mixed": len(al8["mixed"]), "unmatched_detail": al8["unmatched"], "mixed_detail": al8["mixed"]}
    alt_reqs = [{"boundary": e["boundary"], "emit": e["emit"], "frontier": e["frontier"], "scope": scope_of(e["emit"]), "avail": (fin[ch_of[e["frame"]]] + f0_decision_cpu) if (0 <= e["frame"] < len(ch_of) and ch_of[e["frame"]] is not None and fin[ch_of[e["frame"]]] is not None) else None, "cpu": f0_decision_cpu, "kind": "f0-latch-alt"} for e in ev_alt]
    alt_applied, alt_history = None, []
    for req in alt_reqs:
        if req["scope"] != "valid":
            alt_history.append({"request": {k: req[k] for k in ("boundary", "emit", "frontier", "kind")}, "outcome": req["scope"], "applied_boundary": alt_applied["boundary"] if alt_applied else None})
            continue
        if req["avail"] is None or seal is None:
            alt_history.append({"request": {k: req[k] for k in ("boundary", "emit", "frontier", "kind")}, "outcome": "UNKNOWN", "applied_boundary": alt_applied["boundary"] if alt_applied else None})
            continue
        if req["avail"] > seal:
            alt_history.append({"request": {k: req[k] for k in ("boundary", "emit", "frontier", "kind")}, "outcome": "too_late", "applied_boundary": alt_applied["boundary"] if alt_applied else None})
            continue
        if alt_applied is None:
            alt_applied = {"boundary": req["boundary"]}
            alt_history.append({"request": {k: req[k] for k in ("boundary", "emit", "frontier", "kind")}, "outcome": "applied", "applied_boundary": req["boundary"]})
        else:
            alt_history.append({"request": {k: req[k] for k in ("boundary", "emit", "frontier", "kind")}, "outcome": "already_separated", "applied_boundary": alt_applied["boundary"]})
    alt_part = partition_ownership(align_window(gt8, groups, wlo, whi), alt_applied["boundary"] if alt_applied else None)
    rframe = mapping["ready_frames"][1]
    rci = ch_of[rframe] if 0 <= rframe < len(ch_of) else None
    ready_session = fin[rci] if (rci is not None and fin[rci] is not None) else None
    all_measured = ((first is None) or (f0_rel == "measured")) and all(h["outcome"] != "UNKNOWN" for h in alt_history) and (ctrl_rel == "measured")
    qa = qa_merge_for_case(case, groups)
    acc_toks = cap.get("accepted", {}).get("tokens", [])
    cons = conservation_check(acc_toks, groups)
    ctrls = throwaway_controls(acc_toks, groups)
    frag = {"binding_single_request": True, "unnecessary_fragment": 0, "alt_applied_total": sum(1 for h in alt_history if h["outcome"] == "applied"), "alt_redundant_already_separated": sum(1 for h in alt_history if h["outcome"] == "already_separated")}
    sup = support_in_span(cache["sources"][sid], lo, hi)
    return {"case": case, "source": sid, "mapping": mapping, "mapping_scope": [mlo, mhi], "mapping_support_n": len(support), "scored_support": sup, "full_payload_alignment": {"n_gt": len(full_gt), "n_matched": len(full_al["matched"]), "n_unmatched": len(full_al["unmatched"]), "n_mixed": len(full_al["mixed"]), "right_exposure_unmatched": [{"text": w["text"], "start": w["start"], "end": w["end"], "role": w["role"]} for w in right_exposure]}, "f0_original_event": first, "f0_original_scope": f0_scope, "f0_original_release": f0_rel, "f0_original_avail_session": f0_avail, "f0_original_frame": first["frame"] if first else None, "f0_latch_alt_events": ev_alt, "f0_latch_alt_history": alt_history, "f0_latch_alt_applied": alt_applied, "f0_latch_alt_ownership_8": alt_part, "warm_backlog_at_capture_entry_s": backlog, "control": {"boundary": b, "confirm_end": b + CONFIRMATION, "confirm_frame": int(ctrl_frame), "release_kind": ctrl_rel, "chunk_finish_session": ctrl_base, "availability": ctrl_avail, "deadline": seal, "max_permissible_lag": ctrl_mpl, "status": ctrl_status, "added_latency_cpu": ctrl_cpu}, "arms": arms, "qa_merge": qa, "ready_session_clock": ready_session, "all_events_measured": all_measured, "conservation": cons, "conservation_controls": ctrls, "fragmentation": frag, "clocks": {"seal": seal, "asr_final_arrival": seal, "pcm_seal": cap.get("session", {}).get("finalize_wall"), "request_receipt": "per-arm-availability", "translation_dispatch": None, "visible_commit": None, "clock_note": "virtual captured clock honest assumption not live API"}}
SRC_REF = {"ami_EN2009d": {"meet": "EN2009d", "arole": "A", "scope": [168000, 670592], "global": "FEE083", "episode": "ami_EN2009d:A00003"}, "ami_ES2009a": {"meet": "ES2009a", "arole": "A", "scope": [3101568, 3159968], "global": "MEE033", "episode": "ami_ES2009a:A00018"}}
GUARDS = {"R1": {"sid": "ami_ES2009a", "meet": "ES2009a", "arole": "A", "span": [3159968, 3207328], "map_scope": [3101568, 3159968], "kind": "overlap-return", "carried": True}, "R2": {"sid": "ami_EN2009d", "meet": "EN2009d", "arole": "A", "span": [670592, 701312], "map_scope": [168000, 670592], "kind": "overlap-return", "carried": True, "unsupported_reason": "no anchor-only 100ms run in the full eligible reference prior from actual A00003 epoch start 168000; recorded explicit observed UNSUPPORTED with no invented failure"}, "T1": {"sid": "ami_EN2009d", "meet": "EN2009d", "arole": "A", "span": [701760, 755520], "map_scope": [168000, 670592], "kind": "overlap-takeover", "carried": True}, "BC1": {"sid": "ami_ES2009a", "meet": "ES2009a", "arole": "A", "span": [9119360, 9125280], "map_scope": [9091552, 9119360], "kind": "reference-scope", "oracle_fallback": True, "tail_unknown_576": [9124704, 9125280], "unsupported_reason": "anchor absent by design with verified pure B+C at word level and no anchor-only support in genuine A00047 preroll; carried-slot oracle diagnostic only, non-binding, no pure new reference invented"}, "SINGLE_ES2009c": {"sid": "ami_ES2009c", "meet": "ES2009c", "arole": "A", "span": [770080, 789280], "map_scope": [610080, 770080], "kind": "singleton"}, "SINGLE_ES2009d": {"sid": "ami_ES2009d", "meet": "ES2009d", "arole": "A", "span": [723520, 747680], "map_scope": [563520, 723520], "kind": "singleton"}}
def proxy_partition_words(meet, anchor_role, span, boundary):
    words = gt_words_in_span(meet, span[0] / 16000.0, span[1] / 16000.0)
    matched = []
    for w in words:
        matched.append({"gt": {"id": w["id"], "text": w["text"], "side": "left" if w["role"] == anchor_role else "right", "in_span": True, "start": w["start"], "end": w["end"]}, "group_idx": -1, "group_word": w["text"], "group_end_src": w["end"], "dt_samples": 0})
    al = {"matched": matched, "unmatched": [], "mixed": [], "region_group_idxs": [], "unresolved_group_idxs": []}
    return {"n_span_gt_words": len(words), "partition": partition_ownership(al, boundary), "actual_asr_quality": "UNKNOWN-no-ASR-on-guard"}
def run_guards_live(obs, cache, probs_by_sid):
    out = []
    carried = {}
    for sid, ref in SRC_REF.items():
        sentry = next(s for s in obs["sources"] if s["source_id"] == sid)
        pp = probs_by_sid[sid]
        f0r = max(ref["scope"][0] // 1280, 0)
        f1r = min(int((ref["scope"][1] - 1) // 1280) + 1, sentry["valid_native_frames"])
        sup = anchor_support_frames(ref["meet"], ref["arole"], f0r, f1r)
        carried[sid] = {"ref": ref, "mapping": map_anchor_slot(pp, [True] * sentry["valid_native_frames"], sup), "support_n": len(sup)}
    for gid, g in GUARDS.items():
        src_entry = next(s for s in obs["sources"] if s["source_id"] == g["sid"])
        probs = probs_by_sid[g["sid"]]
        table = build_native_table(src_entry, cache["sources"][g["sid"]])
        n = src_entry["valid_native_frames"]
        mf0, mf1 = int(g["map_scope"][0] // 1280), int((g["map_scope"][1] - 1) // 1280) + 1
        mf0, mf1 = max(mf0, 0), min(mf1, n)
        support = anchor_support_frames(g["meet"], g["arole"], mf0, mf1)
        mapping = map_anchor_slot(probs, table["valid_native"], support)
        rec = {"id": gid, "kind": g["kind"], "span": g["span"], "map_scope": g["map_scope"], "anchor_role": g["arole"], "proxy_availability": "virtual-source-end", "actual_captured_quality": "no-ASR-proxy-only", "mapping": mapping, "mapping_support_n": len(support)}
        oracle = False
        if mapping is None:
            if g.get("oracle_fallback") and carried.get(g["sid"], {}).get("mapping") is not None:
                oracle = True
                mapping = carried[g["sid"]]["mapping"]
                rec["oracle_ref"] = {"carried_mapping": mapping, "carried_from": carried[g["sid"]]["ref"], "label": "oracle-diagnostic-non-binding-no-training-no-harm-claim"}
            else:
                rec["status"] = "unsupported-no-anchor-support-separable-P3R-proxy-only"
                rec["unsupported_reason"] = g.get("unsupported_reason", "no anchor-only 100ms run in causal preroll scope; proxy-only")
                rec["confident_wrong_owner_words"] = None
                out.append(rec)
                continue
        rec["oracle_diagnostic"] = bool(oracle)
        slot = mapping["slot"]
        scores = [float(1 - probs[i, slot]) for i in range(n)]
        speakers = ["anchor"] * n
        valid = [bool(table["valid_native"][i] and table["valid_old"][i]) for i in range(n)]
        f0, f1 = int(g["span"][0] // 1280), int((g["span"][1] - 1) // 1280) + 1
        f0, f1 = max(f0, 0), min(f1, n)
        evs = f0_fire_events(list(range(f0, f1)), table["starts"], table["ends"], valid, table["masked"], table["speech"], scores, table["frontiers"], F0_TAU, CONFIRMATION, True)
        in_span = [e for e in evs if g["span"][0] <= e["boundary"] < g["span"][1]]
        rec["f0_events_in_scope"] = len(evs)
        rec["f0_events_in_span"] = [{"boundary": e["boundary"], "emit": e["emit"]} for e in in_span]
        if not in_span:
            rec["confident_wrong_owner_words"] = 0
            rec["proxy_exposure"] = {"n_span_gt_words": len(gt_words_in_span(g["meet"], g["span"][0] / 16000.0, g["span"][1] / 16000.0)), "intervention": "zero-no-applied-boundary", "actual_safety": "not-claimed-from-proxy"}
            rec["status"] = "guard-no-intervention-exposure-proxy-only"
        else:
            first_b = in_span[0]["boundary"]
            proxy = proxy_partition_words(g["meet"], g["arole"], g["span"], first_b)
            proxy["fragmentation_extra_events"] = len(in_span) - 1
            rec["proxy_partition"] = proxy
            rec["confident_wrong_owner_words"] = proxy["partition"]["n_wrong_definite"]
            rec["proxy_uncertain"] = proxy["partition"]["n_uncertain_straddle"]
            rec["status"] = "guard-direction-supported-proxy-no-ASR-harm-claim" if g["kind"] == "overlap-takeover" else "guard-in-span-events-proxy-partitioned"
        out.append(rec)
    return out, carried
def run_salvage():
    out = {}
    for gid, path in (("G03", P2 / "captures" / "P2-G03.json"), ("G04", P2 / "captures" / "P3-G04.json")):
        cap = json.loads(path.read_text(encoding="utf-8"))
        groups = cap.get("groups", [])
        toks = cap.get("accepted", {}).get("tokens", [])
        out[gid] = {"capture": str(path.relative_to(ROOT)), "n_groups": len(groups), "n_accepted_tokens": len(toks), "status": "mixed-salvage-nonbinding-no-fresh-inference"}
    rp = ROOT / "experiments" / "psem_evidence_delivery_gap" / "soniox_equal_timestamp_repair" / "results.json"
    r = json.loads(rp.read_text(encoding="utf-8"))
    corr = r.get("corrected_accepted", {})
    out["P4"] = {"source": "experiments/psem_evidence_delivery_gap/soniox_equal_timestamp_repair/results.json", "n_corrected_tokens": len(corr.get("tokens", corr) if isinstance(corr, dict) else corr), "status": "historical-nonbinding-no-fresh-inference"}
    return out
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="run", choices=["run", "smoke"])
    a = ap.parse_args()
    if a.cmd == "smoke":
        import subprocess
        r = subprocess.run([sys.executable, str(EXP / "replay.py"), "smoke"], capture_output=True, text=True)
        print(r.stdout)
        return r.returncode
    t_all = time.perf_counter()
    obs_path = EXP / "observations" / "OBSERVATIONS.json"
    if sha256_file(obs_path) != BARRIER_SHA:
        print(json.dumps({"error": "OBSERVATIONS-sha-mismatch-barrier"}))
        return 2
    obs = json.loads(obs_path.read_text(encoding="utf-8"))
    errs = validate_obs_bundle(obs)
    if errs:
        print(json.dumps({"error": "bundle-validation-failed", "details": errs}))
        return 2
    cache = json.loads((EXP / "old_grid_cache.json").read_text(encoding="utf-8"))
    probs_by_sid = {}
    for s in obs["sources"]:
        ff = s["feature_files"]["probs"]
        probs_by_sid[s["source_id"]] = load_np(ROOT / ff["path"], ff["shape"])
    cases = {}
    for case in CASES:
        cases[case] = run_positive(case, obs, cache, probs_by_sid)
    guards = run_guards_live(obs, cache, probs_by_sid)
    carried_refs = guards[1]
    guards = guards[0]
    salvage = run_salvage()
    checks = []
    for case in CASES:
        c = cases[case]
        checks.append([case + "-mapping-pre-event", c["mapping"] is not None and c["mapping"]["ready_sample"] < FREEZE["cases"][case]["boundary_samples"]])
        src_entry = next(s for s in obs["sources"] if s["source_id"] == c["source"])
        checks.append([case + "-frontiers-in-prefix", all(e["frontier"] <= src_entry["prefix_end_sample"] for e in ([c["f0_original_event"]] if c["f0_original_event"] else []) + c["f0_latch_alt_events"])])
        checks.append([case + "-none-no-request", c["arms"]["none"]["requests"] == []])
        checks.append([case + "-events-measured-support", bool(c["all_events_measured"])])
        accs = [h["availability"] for arm in ("f0_original", "control_capacity") for h in c["arms"][arm]["requests"] if h["outcome"] == "applied" and h.get("availability") is not None]
        checks.append([case + "-ready-before-accept", c["ready_session_clock"] is not None and all(c["ready_session_clock"] <= a for a in accs)])
    wall = time.perf_counter() - t_all
    ledger = {"freeze_id": FREEZE["freeze_id"], "obs_sha256": BARRIER_SHA, "generated_at_utc": datetime.now(timezone.utc).isoformat(), "cases": cases, "guards": guards, "carried_refs": carried_refs, "salvage": salvage, "future_head_latency": "UNKNOWN-not-measured-no-H-restore", "branch_class": "legal-measured-virtual-profile-scoped-benchmark-actual-service-capture-trace-not-production-live-proof", "integration_checks": [{"name": k, "pass": bool(v)} for k, v in checks], "live_compute_wall_s": wall, "profile_note": "fresh native F0 FP16 Vulkan NO_MUL_MAT_VEC=1 F32_HEAD=1 LOWLATENCY exe 3706db55 loaded local; no H restore; service costs observed offline virtual stream not live API"}
    (EXP / "causal_ownership_ledger.json").write_text(json.dumps(ledger, indent=1), encoding="utf-8")
    npass = sum(1 for _, v in checks if v)
    print(json.dumps({"cases": list(cases.keys()), "checks_pass": [npass, len(checks)], "wall_s": round(wall, 1)}))
    return 0 if npass == len(checks) else 1
if __name__ == "__main__":
    raise SystemExit(main())
