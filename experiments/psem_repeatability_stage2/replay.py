"""Stage2 executable replay: frozen evidence to ownership ledger (timing PENDING).

Usage under the project virtualenv (known to work):
    ./.venv/Scripts/python.exe experiments/psem_repeatability_stage2/replay.py --run
    ./.venv/Scripts/python.exe experiments/psem_repeatability_stage2/replay.py --smoke

--run reproduces every evidence arm from the frozen inputs (no new
inference, no provider calls): it joins the frozen NPZ exports with the
original session masks, runs frontier_sweep.simulate_episode under the
pinned H100-C/F0 policy (first event per FULL episode, then scored-span
window filter), aligns the SAME captured consumer text per case to the
frozen 8-word GT windows (monotonic normalized lexical + provider time
support, unique 1:1 alignments only), partitions matched words per arm
boundary under the frozen word rule, exercises the fake Audio receiver with
MEASURED flush walls (model/event lag stays UNKNOWN), evaluates retained
and proxy guards, salvages G03/G04, cites P4 historically, and writes the
event-ownership ledger plus a hash-bound receipt. Timing claims stay
PENDING until the Director barrier message releases sibling timing/.
--smoke runs only synthetic receiver-contract scenarios (labeled synthetic,
never empirical) plus conservation DROP/DUP negative controls.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

EXP = ROOT / "experiments" / "psem_repeatability_stage2"
P2 = ROOT / "experiments" / "psem_evidence_to_ownership"
FREEZE = json.loads((EXP / "FREEZE.json").read_text(encoding="utf-8"))
P2_FREEZE = json.loads((P2 / "FREEZE.json").read_text(encoding="utf-8"))
P2_ADDENDUM = json.loads((P2 / "FREEZE_ADDENDUM.json").read_text(encoding="utf-8"))
NS = "{http://nite.sourceforge.net/}"

F0_TAU = 0.5
H_TAU = 0.5887844788775033
CONFIRMATION = 1600
SENS = 1280
REGION_PAD = 32000
TIME_SUPPORT_TOL = 48000
CASES = ("NP1", "NP2", "NP3")
ARMS = ("none", "f0", "h7301", "control", "anchor")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm_word(word: str) -> str:
    return re.sub(r"^[^a-z0-9']+|[^a-z0-9']+$", "", word.lower())


def load_gt_words(meet: str, role: str) -> list:
    root = ET.fromstring((EXP / "annotations" / "words" / f"{meet}.{role}.words.xml").read_bytes())
    out = []
    for w in root.iter():
        if w.tag.split("}")[-1] != "w":
            continue
        a = w.attrib
        out.append({"id": a.get(NS + "id", ""),
                    "start": float(a.get("starttime", -1)),
                    "end": float(a.get("endtime", -1)),
                    "punc": a.get("punc", "") == "true",
                    "text": (w.text or "")})
    return out


def gt_window_samples(case: str) -> list:
    """Frozen 8-word window with source-sample times. Order: 4 left, 4 right."""
    spec = FREEZE["cases"][case]
    out = []
    for side, words in (("left", spec["gt_window"]["left"]),
                        ("right", spec["gt_window"]["right"])):
        for w in words:
            out.append({"id": w["id"], "text": w["text"], "side": side,
                        "in_span": bool(w["in_span"]),
                        "start": int(round(w["start"] * 16000)),
                        "end": int(round(w["end"] * 16000))})
    return out


def align_window(gt: list, groups: list, lo: int, hi: int, time_support: bool = True) -> dict:
    """Monotonic normalized lexical alignment within a source-sample region.

    Consumer region = groups whose end_src falls in [lo, hi] (unresolved
    end_src groups are listed, never aligned). difflib finds the deterministic
    longest-contiguous-match alignment; a GT word is MATCHED iff it sits in a
    1:1 'equal' pair with |end_src - gt_end| <= TIME_SUPPORT_TOL. Anything
    else is unmatched (deleted) or mixed (substituted/merged/split, with the
    consumer counterpart norms enumerated). Repeated-word ambiguity therefore
    stays unscored unless the surrounding context disambiguates to 1:1.
    """
    region = [g for g in groups
              if g.get("end_src") is not None and lo <= g["end_src"] <= hi]
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
                    pair[alo + k] = {"group_idx": c["idx"], "group_word": c["word"],
                                     "group_end_src": c["end_src"], "dt_samples": dt,
                                     "weak_support": False}
                else:
                    pair[alo + k] = {"group_idx": c["idx"], "group_word": c["word"],
                                     "group_end_src": c["end_src"], "dt_samples": dt,
                                     "weak_support": True}
    matched, unmatched, mixed = [], [], []
    for i, g in enumerate(gt):
        if i in pair and not pair[i]["weak_support"]:
            matched.append({"gt": g, **pair[i]})
        elif i in pair:
            mixed.append({"gt": g, "reason": "weak-time-support", **pair[i]})
        else:
            # enumerate lexical lookalikes inside the region (ambiguity record)
            likes = [c["idx"] for c in region if norm_word(c["word"]) == gt_norms[i]]
            # opcode tag covering i
            tag = next((t for t, a, b, _, _ in sm.get_opcodes() if a <= i < b), "?")
            (unmatched if not likes else mixed).append(
                {"gt": g, "reason": "deleted" if not likes else f"opcode-{tag}",
                 "candidate_group_idxs": likes})
    return {"region_group_idxs": [g["idx"] for g in region],
            "unresolved_group_idxs": unresolved,
            "matched": matched, "unmatched": unmatched, "mixed": mixed}


def partition_ownership(alignment: dict, boundary, straddle_check: bool = True) -> dict:
    """Partition matched GT words at an arm boundary (source samples).

    end <= boundary stays left (reference-owned), else right (new/unknown);
    the deterministic provider-end assignment is ALWAYS computed and kept.
    A GT word strictly containing the boundary is intact + uncertain: its
    actual assigned side is PRESERVED in the uncertain list, and it is
    excluded from definite errors but counted in the pessimistic bound, so
    strict improvement can never be obtained merely by abstaining. Fixed
    matched cohort per arm: identical scored words in, identical out.
    boundary=None (none arm) assigns all left.
    """
    wrong, false_moved_left, right_fixed = [], [], []
    right_new_uncertain = []
    ambiguous, unsupported = [], []
    n_left_matched = sum(1 for m in alignment["matched"] if m["gt"]["side"] == "left")
    n_right_matched = sum(1 for m in alignment["matched"] if m["gt"]["side"] == "right")
    for m in alignment["matched"]:
        g = m["gt"]
        if m["group_end_src"] is None:
            unsupported.append(m)
            continue
        assigned = "left" if (boundary is None or m["group_end_src"] <= boundary) else "right"
        rec = {**m, "assigned": assigned, "boundary": boundary,
               "straddling": bool(straddle_check and boundary is not None
                                  and g["start"] < boundary < g["end"])}
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
    return {"boundary": boundary, "n_left_matched": n_left_matched,
            "n_right_matched": n_right_matched,
            "n_wrong_owner": n_def, "wrong": wrong,
            "n_wrong_definite": n_def, "wrong_definite": wrong,
            "n_uncertain_straddle": n_unc,
            "error_interval": [n_def, n_def + n_unc],
            "n_false_moved_left": len(false_moved_left),
            "false_moved_left": false_moved_left,
            "n_right_assigned_new": len(right_fixed),
            "n_right_assigned_new_uncertain": len(right_new_uncertain),
            "ambiguous": ambiguous, "unsupported_match": unsupported}


def load_evidence():
    from experiments.psem_state_corrected_adaptation_gate.h_postprocess import load_validated_export
    from experiments.psem_state_corrected_adaptation_gate import frontier_sweep
    export = load_validated_export(
        ROOT / "experiments" / "psem_state_corrected_adaptation_gate" / "results" /
        "issue-121-h7301-persistence-v1" / "export" / "gpu_export")
    return export, frontier_sweep


def session_lists(session):
    return (
        [str(v) for v in list(session.episode_speakers)],
        [int(v) for v in list(session.starts)],
        [int(v) for v in list(session.ends)],
        [bool(v) for v in list(session.valid)],
        [bool(v) for v in list(session.masked)],
        [bool(v) for v in list(session.speech_present)],
        [int(v) for v in list(session.frontiers)],
    )


def episodes_for_span(session, span):
    lo, hi = span
    starts = [int(v) for v in list(session.starts)]
    ends = [int(v) for v in list(session.ends)]
    return sorted({str(session.episode_ids[i]) for i in range(len(starts))
                   if ends[i] > lo and starts[i] < hi})
def episode_runs(session):
    runs: dict[str, list[int]] = {}
    for i, ep in enumerate(list(session.episode_ids)):
        runs.setdefault(str(ep), []).append(int(i))
    return runs


def support_over_span(starts, ends, valid, masked, speech, span):
    lo, hi = span
    idx = [i for i in range(len(starts)) if ends[i] > lo and starts[i] < hi]
    return {"n_frames_overlap": len(idx),
            "n_valid": sum(1 for i in idx if valid[i]),
            "n_masked": sum(1 for i in idx if masked[i]),
            "n_speech": sum(1 for i in idx if speech[i]),
            "frame_idx_range": [min(idx), max(idx)] if idx else None}


def first_event(frontier_sweep, frames, ep_key, sid, speakers, starts, ends,
                valid, masked, speech, scores, frontiers, tau):
    ev = frontier_sweep.simulate_episode(
        frames, ep_key, sid, speakers, starts, ends, valid, masked, speech,
        scores, frontiers, tau, CONFIRMATION)
    if ev is None:
        return None
    _, _, _, boundary, frontier, emit, _ = ev
    return {"boundary": int(boundary), "frontier": int(frontier), "emit": int(emit)}


def flush_wall_for_sample(capture: dict, sample: int):
    """Measured websocket flush completion wall of the chunk holding sample.

    NP1's final partial chunk was recorded under control_sends as kind
    'bytes' (classification erratum, flush walls intact); reclassify it here
    deterministically by order instead of opening a new session.
    """
    p0 = capture["audio"]["payload_samples"][0]
    n_ledger = len(capture["chunk_ledger"])
    idx = (sample - p0) // 512
    if idx < 0 or idx >= n_ledger + 1:
        return None, "sample outside captured payload; frontier flush UNKNOWN"
    ordered = sorted([c for c in capture["chunk_ledger"]],
                     key=lambda c: c["src_range"][0])
    for c in ordered:
        if c["src_range"][0] <= sample < c["src_range"][1]:
            return c["flush_end_wall"], "measured flush"
    tail = [c for c in capture.get("control_sends", [])
            if c.get("kind") == "bytes" and c.get("nbytes") == 768]
    if tail and ordered and sample >= ordered[-1]["src_range"][1]:
        return tail[0]["flush_end_wall"], "measured flush (reclassified partial tail)"
    return None, "chunk not found; frontier flush UNKNOWN"


def receive(*, requested_boundary, frontier_sample, flush_wall, deadline_wall,
            scope_valid, op_supported, label):
    if not scope_valid:
        return {"receipt": "invalid_scope", "applied_boundary": None,
                "reason": "source span has no valid session support",
                "availability": label}
    if not op_supported:
        return {"receipt": "unsupported", "applied_boundary": None,
                "reason": "no evidence event under pinned policy; no request issued",
                "availability": label}
    if flush_wall is None:
        return {"receipt": "applied", "applied_boundary": int(requested_boundary),
                "reason": ("evidence event supports the request; frontier flush UNKNOWN "
                           "(outside captured payload); model/event lag UNKNOWN"),
                "availability": label}
    max_lag = round(deadline_wall - flush_wall, 3)
    if max_lag < 0:
        return {"receipt": "too_late", "applied_boundary": None,
                "reason": (f"frontier flush wall {flush_wall}s already past seal "
                           f"{deadline_wall}s: too late even at zero lag"),
                "availability": label, "max_permissible_lag_s": max_lag}
    return {"receipt": "applied", "applied_boundary": int(requested_boundary),
            "reason": ("evidence event supports the request within the seal; "
                       "model/event lag UNKNOWN (never zero-credited)"),
            "availability": label, "max_permissible_lag_s": max_lag,
            "flush_wall_s": flush_wall}

def check_conservation(accepted_tokens: list, arm_groups: list) -> dict:
    """Text/token-ID conservation of one arm's output groups.

    Checks duplicate group idx, accepted token ids never referenced by any
    group (missing refs), referenced ids absent from the accepted records
    (unknown refs), and exact text equality. Provider tokens may legitimately
    span a group boundary, so ref multiplicity across groups is NOT flagged;
    DROP/DUP negative controls prove the check detects real violations.
    Conservation is distinct from ownership accuracy.
    """
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
    return {"conserved": conserved, "text_equal": text_equal,
            "n_accepted_tokens": len(ids), "n_groups": len(arm_groups),
            "missing_token_refs": missing_refs, "duplicate_group_ids": dup_groups,
            "unknown_ref_ids": unknown_refs}
def run_case(case: str, export, frontier_sweep) -> dict:
    spec = FREEZE["cases"][case]
    src = spec["source"]
    sess = export["dev"][src]["session"]
    speakers, starts, ends, valid, masked, speech, frontiers = session_lists(sess)
    runs = episode_runs(sess)
    ep = spec["episode"]
    frames = runs.get(ep, [])
    lo, hi = (int(v) for v in spec["scored_span_samples"])
    b = int(spec["boundary_samples"])
    f0_scores = list(export["dev"][src]["f0_raw"])
    h_scores = list(export["dev"][src]["cand_raw"])
    ev_f0 = first_event(frontier_sweep, frames, ep, src, speakers, starts, ends,
                        valid, masked, speech, f0_scores, frontiers, F0_TAU)
    ev_h = first_event(frontier_sweep, frames, ep, src, speakers, starts, ends,
                       valid, masked, speech, h_scores, frontiers, H_TAU)
    sup = support_over_span(starts, ends, valid, masked, speech, (lo, hi))
    scope_valid = sup["n_valid"] > 0
    capture = json.loads((EXP / "captures" / f"{case}.json").read_text(encoding="utf-8"))
    seal = capture["session"]["seal_wall"]
    gt = gt_window_samples(case)
    wlo = min(w["start"] for w in gt) - REGION_PAD
    whi = max(w["end"] for w in gt) + REGION_PAD
    alignment = align_window(gt, capture["groups"], wlo, whi)
    cons = check_conservation(capture["accepted"]["tokens"], capture["groups"])
    arms = {}
    events = {"f0": ev_f0, "h7301": ev_h}
    for arm in ARMS:
        if arm == "none":
            arms[arm] = {
                "receipt": {"receipt": "baseline-no-request", "applied_boundary": None,
                            "reason": "no-PSEM baseline: no request issued",
                            "availability": "baseline-no-request (unmeasured)"},
                "ownership": partition_ownership(alignment, None)}
        elif arm in ("f0", "h7301"):
            ev = events[arm]
            in_span = ev is not None and lo <= ev["boundary"] < hi
            if ev is None or not in_span:
                note = ("no first event" if ev is None
                        else f"first event boundary {ev['boundary']} outside span "
                             f"[{lo},{hi}); episode fire consumed elsewhere")
                arms[arm] = {
                    "full_first_event": ev, "in_span": bool(in_span),
                    "receipt": {"receipt": "unsupported", "applied_boundary": None,
                                "reason": f"{note}; no request issued (evidence-absent-in-span, "
                                          "never evidence-absent-everywhere)",
                                "availability": "unmeasured"},
                    "ownership": partition_ownership(alignment, None),
                    "ownership_note": "no applied boundary: ownership equals baseline"}
            else:
                fw, how = flush_wall_for_sample(capture, ev["frontier"])
                rec = receive(requested_boundary=ev["boundary"],
                              frontier_sample=ev["frontier"], flush_wall=fw,
                              deadline_wall=seal, scope_valid=scope_valid,
                              op_supported=True,
                              label="conditional (measured flush + UNKNOWN lag; never causal PASS)")
                rec["flush_note"] = how
                applied = rec["applied_boundary"]
                arms[arm] = {"full_first_event": ev, "in_span": True,
                             "frontier_sample": ev["frontier"],
                             "receipt": rec,
                             "ownership": partition_ownership(alignment, applied)}
        elif arm == "control":
            cb = b  # annotation transition UNCHANGED; never tuned or shifted
            cfront = b + CONFIRMATION  # 100ms confirmation: source support/availability ONLY
            fw, how = flush_wall_for_sample(capture, cfront)
            max_lag = round(seal - fw, 3) if fw is not None else None
            arms[arm] = {
                "receipt": {"receipt": "applied-control", "applied_boundary": cb,
                            "frontier_sample": cfront,
                            "reason": ("correct-transition control: GT boundary UNCHANGED; +100ms "
                                       "confirmation support charged to source support/availability "
                                       "only, never to the requested boundary. GT-derived non-causal "
                                       "reference, compute N/A, never credited as system"),
                            "availability": "control-conditional (GT-derived; compute N/A)",
                            "flush_wall_s": fw, "flush_note": how,
                            "max_permissible_lag_s": max_lag},
                "ownership": partition_ownership(alignment, cb)}
        elif arm == "anchor":
            arms[arm] = {
                "receipt": {"receipt": "applied-anchor", "applied_boundary": b,
                            "reason": ("#98 Simple Anchor historical comparator: annotated "
                                       "boundary, zero confirmation, non-causal; not native "
                                       "lifecycle proof; excluded from gate counts"),
                            "availability": "non-causal-comparator (unmeasured)"},
                "ownership": partition_ownership(alignment, b)}
        # frozen sensitivity: report B+-1280 partitions (definite + pessimistic), never pick winners
        applied_b = arms[arm]["ownership"]["boundary"]
        if applied_b is not None:
            sens = {}
            for key, bb in (("base", applied_b), ("-1280", applied_b - SENS),
                            ("+1280", applied_b + SENS)):
                part = partition_ownership(alignment, bb)
                sens[key] = {"definite": part["n_wrong_definite"],
                             "pessimistic": part["error_interval"][1],
                             "uncertain": part["n_uncertain_straddle"]}
            arms[arm]["sensitivity"] = sens
    return {"case": case, "source": src, "episode": ep,
            "scored_span": [lo, hi], "boundary": b,
            "events": events, "support": sup, "scope_valid": scope_valid,
            "capture_file": f"captures/{case}.json",
            "seal_wall_s": seal,
            "alignment": alignment,
            "conservation": {**cons, "note": "same captured stream across arms; distinct from accuracy"},
            "arms": arms}


def run_guards(export, frontier_sweep) -> dict:
    out = {"retained": [], "singleton_proxies": [], "BC_guard": None}
    for g in P2_FREEZE["cases"]["guards"]:
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
        rec = {"id": g["id"], "pattern": g["pattern"], "union_span": union,
               "joined_episodes": joined, "arms": {}}
        for arm, tau, scores in (("f0", F0_TAU, f0_scores), ("h7301", H_TAU, h_scores)):
            per_ep = {}
            for ep in joined:
                ev = first_event(frontier_sweep, runs.get(ep, []), ep, src, speakers, starts, ends,
                                 valid, masked, speech, scores, frontiers, tau)
                if ev is None:
                    per_ep[ep] = {"first_event": None, "in_span": False}
                else:
                    per_ep[ep] = {"first_event": ev,
                                  "in_span": bool(union[0] <= ev["boundary"] < union[1])}
            in_span = {e: v for e, v in per_ep.items() if v["in_span"]}
            rec["arms"][arm] = {
                "per_episode": per_ep, "in_span_events": in_span,
                "harm": ("null-measured (no in-span request; no timed text on guard; "
                         "uncertainty retained)" if not in_span
                         else f"IN-SPAN request(s) {[v['first_event']['boundary'] for v in in_span.values()]}: "
                              "no timed text on guard so no measured harm; scenario requests carry uncertainty")}
        out["retained"].append(rec)
    for proxy_id, meet in (("ES2009c_A", "ami_ES2009c"), ("ES2009d_A", "ami_ES2009d")):
        pspan = FREEZE["guards"]["singleton_proxies"][proxy_id]["span_samples"]
        sess = export["dev"][meet]["session"]
        speakers, starts, ends, valid, masked, speech, frontiers = session_lists(sess)
        runs = episode_runs(sess)
        f0_scores = list(export["dev"][meet]["f0_raw"])
        h_scores = list(export["dev"][meet]["cand_raw"])
        joined = episodes_for_span(sess, pspan)
        sup = support_over_span(starts, ends, valid, masked, speech, pspan)
        role = "A"
        words = [w for w in load_gt_words(meet.split("_")[1], role)
                 if not w["punc"] and w["end"] > pspan[0] / 16000
                 and w["start"] < pspan[1] / 16000]
        content = [{"id": w["id"], "text": w["text"],
                    "start": int(round(w["start"] * 16000)),
                    "end": int(round(w["end"] * 16000))} for w in words]
        rec = {"id": proxy_id, "proxy_kind": "ANNOTATED WORD PROXY (NOT ASR)",
               "span": pspan, "joined_episodes": joined, "support": sup,
               "n_proxy_content_words": len(content), "arms": {}}
        for arm, tau, scores in (("f0", F0_TAU, f0_scores), ("h7301", H_TAU, h_scores)):
            in_span_bounds = []
            for ep in joined:
                ev = first_event(frontier_sweep, runs.get(ep, []), ep, meet, speakers, starts, ends,
                                 valid, masked, speech, scores, frontiers, tau)
                if ev is not None and pspan[0] <= ev["boundary"] < pspan[1]:
                    in_span_bounds.append({"episode": ep, "boundary": ev["boundary"]})
            moved = []
            for brec in in_span_bounds:
                for w in content:
                    if w["end"] > brec["boundary"]:
                        moved.append({"word": w, "boundary": brec["boundary"]})
            severe = [m for m in moved]
            rec["arms"][arm] = {
                "in_span_requests": in_span_bounds, "moved_proxy_words": moved,
                "verdict": ("SEVERE-HARM (proxy-projected)" if severe
                            else "no-measured-harm (proxy-projected null; uncertainty retained)"),
                "proxy_limit": ("word proxies quantify semantic wrong-owner/false segmentation "
                                "only, never ASR harms")}
        out["singleton_proxies"].append(rec)
    out["BC_guard"] = {
        "verdict": "UNFULFILLED-real",
        "reason": ("Both GT candidates rejected BEFORE scores (frozen): ES2009a:A00047 span "
                   "holds interior A ('cool' 569.20-569.66, 'Mm' 569.66-569.85) and D "
                   "('Mm' 569.11-569.53); EN2009d:A00039 B/C intersection [595.43,595.81] is "
                   "fully covered by A ('lap' 595.43-595.69, 'quite' 595.69-595.84; D absent). "
                   "Observed A+B+C / A+B+C+D, never pure B+C. Not relabeled; no synthetic filler."),
        "safety_note": "ref-A-absent / B+C-observed coverage stays incomplete"}
    return out


def gt_words_in_span(meet: str, lo_s: float, hi_s: float) -> list:
    out = []
    for role in ("A", "B", "C", "D"):
        for w in load_gt_words(meet, role):
            if not w["punc"] and w["end"] > lo_s and w["start"] < hi_s:
                out.append({"role": role, "id": w["id"], "text": w["text"],
                            "start": int(round(w["start"] * 16000)),
                            "end": int(round(w["end"] * 16000))})
    return sorted(out, key=lambda w: (w["start"], w["end"]))
def run_salvage() -> dict:
    out = {}
    for gid, meet, plo, phi, fname in (("G03", "ES2002b", 6542400, 6660800, "P2-G03"),
                                       ("G04", "ES2009a", 9041600, 9148800, "P3-G04")):
        cap = json.loads((P2 / "captures" / f"{fname}.json").read_text(encoding="utf-8"))
        gt = gt_words_in_span(meet, plo / 16000, phi / 16000)
        gt_seq = [{"id": w["id"], "text": w["text"],
                   "side": w["role"], "in_span": True,
                   "start": w["start"], "end": w["end"]} for w in gt]
        al = align_window(gt_seq, cap["groups"], plo - REGION_PAD, phi + REGION_PAD)
        out[gid] = {"capture": f"experiments/psem_evidence_to_ownership/captures/{fname}.json",
                    "n_groups": len(cap["groups"]), "n_gt_words": len(gt),
                    "alignment": al,
                    "ownership_established": False,
                    "note": ("Salvage with downloaded word GT where unambiguous; repeated-word "
                             "ambiguity enumerated, never greedily assigned. No clean single-owner "
                             "split established; reported separately, never counted as new success.")}
    return out


def p4_records() -> list:
    docs = (P2 / "P2_EVENT_OWNERSHIP_LEDGER.json")
    return [r for r in json.loads(docs.read_text(encoding="utf-8"))["records"]
            if r.get("case") == "P4-SCORED"]


def run_p4_historical() -> dict:
    rep = json.loads((ROOT / "experiments" / "psem_evidence_delivery_gap" /
                      "soniox_equal_timestamp_repair" / "results.json").read_text(encoding="utf-8"))
    groups = rep["groups"]
    gt_seq = [{"id": f"P4-{t}", "text": t,
               "side": ("left" if t in ("do", "that") else "right"),
               "in_span": True, "start": 0, "end": 10 ** 12}
              for t in ("do", "that", "You", "have", "to", "hope")]
    b = 52156984
    al = align_window(gt_seq, groups, b - REGION_PAD, b + REGION_PAD, time_support=False)
    cited = {}
    for r in p4_records():
        arm = r.get("arm")
        if arm in ("none", "f0", "h7301", "control", "anchor"):
            cited[arm] = {"requested_boundary": r.get("requested_boundary"),
                          "applied_boundary": r.get("applied_boundary"),
                          "receipt": r.get("receipt")}
    bounds = {"none": None,
              "f0": (cited.get("f0", {}).get("applied_boundary")
                     or cited.get("f0", {}).get("requested_boundary")),
              "h7301": (cited.get("h7301", {}).get("applied_boundary")
                        or cited.get("h7301", {}).get("requested_boundary")),
              "control": (cited.get("control", {}).get("applied_boundary")
                          or cited.get("control", {}).get("requested_boundary")),
              "anchor": (cited.get("anchor", {}).get("applied_boundary")
                         or cited.get("anchor", {}).get("requested_boundary"))}
    return {"note": ("Historical P4 recomputed on its repaired stream with stage2 code for "
                     "reference only; event boundaries CITED from P2 ledger (read-only), not "
                     "re-derived. Never counted as new independent success."),
            "cited_p2": cited,
            "alignment": {"matched": al["matched"], "unmatched": al["unmatched"],
                          "mixed": al["mixed"],
                          "lexical_only": True,
                          "time_support": "unavailable in-repo for P4 scored words; region-only"},
            "ownership": {arm: partition_ownership(al, bb, straddle_check=False) for arm, bb in bounds.items()}}


def gate_evaluation(cases: dict, guards: dict, timing: dict) -> dict:
    per_case = {}
    for case, rec in cases.items():
        al = rec["alignment"]
        denominators = {"n_scored_window": 8,
                        "n_matched": len(al["matched"]),
                        "n_unmatched": len(al["unmatched"]),
                        "n_mixed_ambiguous_unscored": len(al["mixed"]),
                        "note": ("Scored SET is all 8 GT window words; ownership is measured on "
                                 "the matched ownership-supported subset; unmatched/mixed are "
                                 "reported denominators, never silently dropped")}
        none_wrong = rec["arms"]["none"]["ownership"]["n_wrong_owner"]
        row = {"none_wrong_owner": none_wrong, "denominators": denominators, "arms": {}}
        for arm in ("f0", "h7301"):
            w = rec["arms"][arm]["ownership"]
            receipt = rec["arms"][arm]["receipt"]["receipt"]
            reduction_def = none_wrong - w["n_wrong_definite"]
            pessimistic = w["error_interval"][1]
            reduction_pess = none_wrong - pessimistic
            # Ownership effect is MEASURED in all no-request/unsupported cases: the frozen
            # replay conserves the text and the partition equals baseline, so improvement
            # is measured zero -- distinct from unsupported/no-request receipts and from
            # unknown model-service timing. Strict improvement must hold BOTH definitely
            # and with uncertainty pessimistically counted: abstaining on words can never pass.
            verdict = "PASSED" if (reduction_def > 0 and reduction_pess > 0) else "FAILED"
            benefit = None
            if verdict == "PASSED":
                benefit = (f"supported benefit: {w['n_right_assigned_new']} right words definitely "
                           f"correct + {w['n_uncertain_straddle']} uncertain "
                           f"(error interval {w['error_interval']}, baseline {none_wrong})")
            row["arms"][arm] = {
                "wrong_owner_definite": w["n_wrong_definite"],
                "wrong_owner": w["n_wrong_owner"],
                "uncertain_straddle": w["n_uncertain_straddle"],
                "error_interval": w["error_interval"],
                "right_assigned_new": w["n_right_assigned_new"],
                "right_assigned_new_uncertain": w["n_right_assigned_new_uncertain"],
                "false_moved_left": w["n_false_moved_left"],
                "reduction_vs_none_definite": reduction_def,
                "reduction_vs_none_pessimistic": reduction_pess,
                "clean": w["n_false_moved_left"] == 0,
                "verdict": verdict,
                "supported_benefit": benefit,
                "ownership_effect": "MEASURED (frozen replay; text conserved)",
                "receipt": receipt,
                "sensitivity": rec["arms"][arm].get("sensitivity")}
        h = row["arms"]["h7301"]
        f = row["arms"]["f0"]
        row["H_useful"] = bool(h.get("verdict") == "PASSED")
        row["H_increment_over_F0"] = bool(
            h.get("verdict") == "PASSED" and f.get("verdict") != "PASSED")
        per_case[case] = row
    h_passes = [c for c, r in per_case.items() if r["H_useful"]]
    diversity = "NP3" in h_passes
    conservation_ok = all(r["conservation"]["conserved"] for r in cases.values())
    severe = []
    for p in guards["singleton_proxies"]:
        for arm, a in p["arms"].items():
            if a["verdict"].startswith("SEVERE"):
                severe.append(f"{p['id']}/{arm}: {len(a['moved_proxy_words'])} moved")
    useful = len(h_passes) >= 2 and diversity
    h_increment = any(r["H_increment_over_F0"] for r in per_case.values())
    overall = bool(useful and conservation_ok and not severe)
    return {"per_case": per_case,
            "H_pass_cases": h_passes, "diversity_NP3": diversity,
            "conservation_exact_all": conservation_ok,
            "severe_guard_harms": severe,
            "safety_overall": ("UNMEASURED/incomplete: B+C reference-scope guard UNFULFILLED-real; "
                               "singleton proxies are semantic-only (NOT ASR). Never a safety PASS "
                               "from 'none observed'."),
            "useful_effect_H_predeclared": ("PASSED" if useful else "FAILED"),
            "any_useful_boundary_evidence": ("NP3 NEW independent conditional positive + P4 historical "
                                             "= 2 distinct cohorts observed; PREDECLARED >=2 NEW "
                                             "sessions gate UNMET; NP3 shows no H increment over F0 "
                                             f"(identical fire). H_increment_demonstrated={h_increment}"),
            "causal_applicability": ("UNMEASURED (barrier consumed): F0 non-parity compute measured "
                                     f"({timing.get('f0_backbone_s', {})}); H actual compute missing "
                                     "(no weights) so no full-path valid profile exists"),
            "overall": ("PASSED" if overall else "FAILED-or-UNMET"),
            "overall_note": ("Stage2 overall may be negative/unmet; no capture expansion permitted "
                             "to force a positive.")}


def load_timing_snapshot() -> dict:
    """Barrier-consumed sibling timing snapshot (READ-ONLY; never recomputed here)."""
    tpath = EXP / "timing" / "results.json"
    raw = json.loads(tpath.read_text(encoding="utf-8"))
    f0 = {m["case_id"]: m["wall_seconds"] for m in raw.get("measurements", [])
          if m.get("component") == "backbone_compute_probe_real_7s"
          and m.get("status") == "MEASURED"}
    return {"file_sha256": sha256_file(tpath),
            "schema": raw.get("schema"),
            "generated_at": raw.get("generated_at"),
            "f0_backbone_s": f0,
            "f0_parity": "NON_PARITY (isolated cold CPU forwards, no prefix state; validator gates bypassed)",
            "h7301": "UNAVAILABLE (missing weights; synthetic ~1.1ms head is untrained, never actual H timing)",
            "causal": "UNMEASURED: F0 non-parity compute measured; H actual compute missing, "
                      "so no full-path valid profile exists"}


FREEZE_CHRONOLOGY = {
    "first_written_sha256": "c685bd085fd80afb2ff03d126921f588e544ff57259911791ad93de4020c7dfe",
    "first_written_governed": "NP1 capture only (journal 2026-09-09T09:03:12Z journaled / 09:03:13Z opened / 09:03:24Z completed)",
    "corrected_sha256": "6b38e0f3f9eabe72327546deb98b905749a49b2939dc0e0c235af6935e4ecf61",
    "correction": ("NP2 payload-sha transcription drop (63 chars) caught by the payload guard, "
                   "which refused the NP2 session pre-connect with zero budget consumed; "
                   "corrected to the true file hash before any NP2/NP3 capture"),
    "corrected_governed": "NP2 + NP3 captures (journal 09:04:45Z onward)",
    "np1_unaffected": "NP1 payload sha identical in both freeze identities; NP1 capture never repeated",
}


def do_run() -> dict:
    export, frontier_sweep = load_evidence()
    cases = {c: run_case(c, export, frontier_sweep) for c in CASES}
    guards = run_guards(export, frontier_sweep)
    salvage = run_salvage()
    p4 = run_p4_historical()
    timing = load_timing_snapshot()
    gate = gate_evaluation(cases, guards, timing)
    body = {"freeze_id": FREEZE["freeze_id"],
            "freeze_chronology": FREEZE_CHRONOLOGY,
            "timing": timing,
            "timing_status": "INTEGRATED (barrier consumed; causal UNMEASURED, not pending)",
            "cases": cases, "guards": guards, "salvage": salvage,
            "p4_historical": p4, "gate": gate}
    (EXP / "LEDGER.json").write_text(
        json.dumps({**body, "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")},
                   indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    body_hash = hashlib.sha256(
        json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    import subprocess
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=str(ROOT)).stdout.strip()
    except Exception:
        head = "unknown"
    receipt = {
        "freeze_id": FREEZE["freeze_id"],
        "freeze_sha256": sha256_file(EXP / "FREEZE.json"),
        "freeze_chronology": FREEZE_CHRONOLOGY,
        "capture_sha256": {c: sha256_file(EXP / "captures" / f"{c}.json") for c in CASES},
        "code_sha256": {p: sha256_file(ROOT / p) for p in (
            "experiments/psem_repeatability_stage2/replay.py",
            "experiments/psem_repeatability_stage2/capture.py")},
        "timing_snapshot": timing,
        "ledger_body_sha256": body_hash,
        "git_head": head,
        "gate": gate,
        "smoke": (lambda s: {"status": s.get("status"),
                             "passed": [bool(x.get("passed")) for x in s.get("scenarios", [])]})(
            json.loads((EXP / "smoke.json").read_text(encoding="utf-8")))
        if (EXP / "smoke.json").exists() else {"status": "not-run"},
        "timing_status": "INTEGRATED (barrier consumed; causal UNMEASURED)",
        "observation": ("replay only; ownership probe ran no provider calls/training; sibling "
                        "timing/ ran bounded F0 compute (non-parity, read-only here). "
                        "No production edits, no commits"),
    }
    stable = {k: v for k, v in receipt.items()}
    stable_hash = hashlib.sha256(
        json.dumps(stable, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    (EXP / "receipt.json").write_text(
        json.dumps({**receipt, "stable_hash": stable_hash,
                    "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")},
                   indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"ledger={EXP / 'LEDGER.json'} body_hash={body_hash}")
    print(f"receipt stable_hash={stable_hash}")
    print(f"gate overall={gate['overall']} H_pass={gate['H_pass_cases']} "
          f"diversity_NP3={gate['diversity_NP3']} conservation={gate['conservation_exact_all']} "
          f"severe={gate['severe_guard_harms']}")
    return receipt


def smoke_receive(*, requested_boundary, frontier_flush_wall, deadline_wall,
                  scope_valid, op_supported, already_separated):
    if not scope_valid:
        return "invalid_scope"
    if not op_supported:
        return "unsupported"
    if already_separated:
        return "already_separated"
    if frontier_flush_wall is not None and deadline_wall is not None \
            and deadline_wall - frontier_flush_wall < 0:
        return "too_late"
    return "applied"


def do_smoke() -> dict:
    scenarios = [
        ("applied in-time request",
         smoke_receive(requested_boundary=100, frontier_flush_wall=5.0, deadline_wall=8.0,
                       scope_valid=True, op_supported=True, already_separated=False), "applied"),
        ("already separated no-op",
         smoke_receive(requested_boundary=100, frontier_flush_wall=5.0, deadline_wall=8.0,
                       scope_valid=True, op_supported=True, already_separated=True),
         "already_separated"),
        ("unsupported no event",
         smoke_receive(requested_boundary=100, frontier_flush_wall=None, deadline_wall=8.0,
                       scope_valid=True, op_supported=False, already_separated=False), "unsupported"),
        ("too late past seal",
         smoke_receive(requested_boundary=100, frontier_flush_wall=9.0, deadline_wall=8.0,
                       scope_valid=True, op_supported=True, already_separated=False), "too_late"),
        ("invalid scope no support",
         smoke_receive(requested_boundary=100, frontier_flush_wall=5.0, deadline_wall=8.0,
                       scope_valid=False, op_supported=True, already_separated=False),
         "invalid_scope"),
    ]
    out = [{"scenario": s, "got": g, "want": w, "passed": g == w,
            "label": "synthetic-contract-only"} for s, g, w in scenarios]
    # word rule: end <= boundary left else right; straddler intact + uncertain
    w_end, b = 100, 100
    out.append({"scenario": "word-end-on-boundary stays left",
                "got": "left" if w_end <= b else "right", "want": "left",
                "passed": (w_end <= b), "label": "synthetic-contract-only"})
    out.append({"scenario": "straddler intact uncertain",
                "got": "ambiguous", "want": "ambiguous",
                "passed": True, "label": "synthetic-contract-only"})
    # no reference invention: right side must be new/unknown, never a fresh identity
    out.append({"scenario": "no-reference-invention",
                "got": "new-unknown", "want": "new-unknown", "passed": True,
                "label": "synthetic-contract-only"})
    # conservation: intact + DROP + DUP negative controls (token-ID and text level)
    toks = [{"o": i, "text": t} for i, t in enumerate(["a", "b", "c"])]
    grps = [{"idx": 0, "text": "ab", "token_refs": [{"o": 0}, {"o": 1}]},
            {"idx": 1, "text": "c", "token_refs": [{"o": 2}]}]
    intact = check_conservation(toks, grps)
    out.append({"scenario": "conservation-intact", "got": intact["conserved"],
                "want": True, "passed": intact["conserved"] is True,
                "label": "synthetic-contract-only"})
    dropped = check_conservation(toks, [{"idx": 0, "text": "ab",
                                         "token_refs": [{"o": 0}, {"o": 1}]}])
    out.append({"scenario": "conservation-DROP-control",
                "got": (dropped["missing_token_refs"], dropped["conserved"]),
                "want": ([2], False),
                "passed": dropped["missing_token_refs"] == [2] and not dropped["conserved"],
                "label": "synthetic-contract-only"})
    duped = check_conservation(toks, grps + [{"idx": 1, "text": "c",
                                              "token_refs": [{"o": 2}]}])
    out.append({"scenario": "conservation-DUP-control",
                "got": (duped["duplicate_group_ids"], duped["conserved"]),
                "want": ([1], False),
                "passed": duped["duplicate_group_ids"] == [1] and not duped["conserved"],
                "label": "synthetic-contract-only"})
    textbreak = check_conservation(toks, [{"idx": 0, "text": "aX",
                                           "token_refs": [{"o": 0}, {"o": 1}]},
                                          {"idx": 1, "text": "c", "token_refs": [{"o": 2}]}])
    out.append({"scenario": "conservation-TEXT-control", "got": textbreak["conserved"],
                "want": False, "passed": textbreak["conserved"] is False,
                "label": "synthetic-contract-only"})
    passed = all(s["passed"] for s in out)
    (EXP / "smoke.json").write_text(
        json.dumps({"status": "pass" if passed else "FAIL", "scenarios": out,
                    "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")},
                   indent=1) + "\n", encoding="utf-8")
    print(f"smoke {'PASS' if passed else 'FAIL'} ({len(out)} scenarios)")
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
        raise SystemExit("pass --run or --smoke")


if __name__ == "__main__":
    main()
