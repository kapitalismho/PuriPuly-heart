"""Soniox reference scoring v2: delta finals, occurrence prefix deadlines, WER.

Repairs v1: no retract-on-absent (official finals are deltas); committed
append-once keyed on full (start,end,text,speaker) occurrence order;
availability by sequence prefix alignment to snapshots at or before the
deadline (never bag membership, never final text); one per-session
Hungarian over pure A-D roles applied to all scopes; NP L/R cohort kept
as a separate explicit binary metric; WER S/D/I on fixed ref vs scoped
hyp; truncated edge words counted once. Reuses existing helpers only.
"""
from __future__ import annotations

import difflib
import importlib.util
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "soniox_realtime_reference_benchmark"
STAGE2 = ROOT / "experiments" / "psem_repeatability_stage2"
FREEZE = json.loads((EXP / "FREEZE.json").read_text(encoding="utf-8"))
S2FREEZE = json.loads((STAGE2 / "FREEZE.json").read_text(encoding="utf-8"))
DEADLINES = (0.2, 0.5, 1.0, 2.0)
TIME_SUPPORT_TOL = 48000
REGION_PAD = 32000


def load_helpers():
    spec = importlib.util.spec_from_file_location(
        "ds_replay_helpers",
        ROOT / "experiments" / "psem_decision_sufficiency" / "replay.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


H = load_helpers()


def committed_stream(messages: list) -> dict:
    """Delta discipline: finals sent ONCE; absent-in-later-batch is normal.

    Committed appends each unseen final identity once, keyed on full
    (start_ms, end_ms, text, speaker) in arrival occurrence order. Exact
    repeats dedupe. Same end with legitimately different words coexists.
    Nothing is ever retracted. No revision is hallucinated from absence.
    """
    committed: list = []
    seen: set = set()
    n_exact_dup = 0
    first_prov: dict = {}
    final_seen_wall: dict = {}
    for m in messages:
        toks = m.get("tokens") or []
        wall = m.get("arrival_wall")
        for t in toks:
            key = (t.get("start_ms"), t.get("end_ms"), str(t.get("text", "") or ""),
                   t.get("speaker"))
            if key not in first_prov and wall is not None:
                first_prov[key] = wall
        for t in toks:
            if not bool(t.get("is_final")):
                continue
            if str(t.get("text", "") or "") in ("<fin>", "<end>"):
                continue
            key = (t.get("start_ms"), t.get("end_ms"), str(t.get("text", "") or ""),
                   t.get("speaker"))
            if wall is not None and key not in final_seen_wall:
                final_seen_wall[key] = wall
            if key in seen:
                n_exact_dup += 1
                continue
            seen.add(key)
            committed.append(dict(t))
    return {"committed": committed, "first_prov": first_prov,
            "final_seen_wall": final_seen_wall,
            "n_exact_duplicate_finals": n_exact_dup,
            "violations": []}


def aggregate_words(committed: list, first_prov: dict, final_seen_wall: dict,
                    p0: int) -> list:
    chars: list = []
    owners: list = []
    for t in committed:
        s = str(t.get("text", "") or "")
        for ch in s:
            chars.append(ch)
            owners.append(t)
    words: list = []
    i, n = 0, len(chars)
    idx = 0
    while i < n:
        if chars[i].isspace():
            i += 1
            continue
        j = i
        while j < n and not chars[j].isspace():
            j += 1
        toks = owners[i:j]
        lasts = toks[-1]
        speakers = [t.get("speaker") for t in toks]
        end_ms = lasts.get("end_ms")
        start_ms = toks[0].get("start_ms")
        keys = [(t.get("start_ms"), t.get("end_ms"), str(t.get("text", "") or ""),
                 t.get("speaker")) for t in toks]
        provs = [first_prov[k] for k in keys if k in first_prov]
        fins = [final_seen_wall[k] for k in keys if k in final_seen_wall]
        wo = "".join(chars[i:j])
        words.append({"idx": idx, "word": wo, "norm": H.norm_word(wo),
                      "start_ms": start_ms, "end_ms": end_ms,
                      "end_src": (p0 + int(end_ms) * 16) if isinstance(end_ms, (int, float)) else None,
                      "start_src": (p0 + int(start_ms) * 16) if isinstance(start_ms, (int, float)) else None,
                      "speaker": lasts.get("speaker"), "speakers": speakers,
                      "mixed_speaker": len({s for s in speakers if s is not None}) > 1,
                      "first_prov_wall": min(provs) if provs else None,
                      "final_wall": max(fins) if fins else None})
        idx += 1
        i = j
    return words
TIME_SUPPORT_S = TIME_SUPPORT_TOL / 16000.0
def time_admissible(snap_wall: float, word_end_wall: float) -> bool:
    """Predeclared TIME_SUPPORT tolerance (48000 samples = 3.0s), untuned.
    A late word can never map to an early repeated word and vice versa."""
    return abs(snap_wall - word_end_wall) <= TIME_SUPPORT_S


def snapshots(messages: list) -> list:
    """Ordered display snapshots: what the server showed, in order, with walls."""
    out = []
    for m in messages:
        toks = m.get("tokens") or []
        if not toks or m.get("arrival_wall") is None:
            continue
        out.append({"wall": m["arrival_wall"],
                    "norms": [H.norm_word(str(t.get("text", "") or "")) for t in toks],
                    "speakers": [t.get("speaker") for t in toks]})
    return out


def align_full_table(full_norms: list, full_walls: list, snaps: list,
                      eos_wall: object) -> list:
    """ONE generic full-list occurrence alignment over cumulative snapshots.
    Full GT norms vs full snapshot norms via difflib (existing align_window
    pattern, autojunk off) give a monotone one-to-one occurrence map; each
    pair then needs predeclared 3.0s time admissibility. Snapshots capped at
    EOS (no EOS future). Ambiguous pairs (substituted, unmatched, or
    time-inadmissible) are unknown, never false true. Negative receipt is
    never credited: availability needs snap wall at or past word end;
    earlier admissible hits are flagged separately."""
    import difflib as _dl
    capped = [s for s in snaps if eos_wall is None or s["wall"] <= eos_wall]
    maps = []
    for s in capped:
        sm = _dl.SequenceMatcher(None, full_norms, s["norms"], autojunk=False)
        m = {}
        for tag, alo, ahi, blo, bhi in sm.get_opcodes():
            if tag == "equal":
                for k in range(ahi - alo):
                    m[alo + k] = blo + k
        maps.append(m)
    rows = []
    for i, W in enumerate(full_walls):
        first = None
        first_pos = None
        credited = None
        credited_pos = None
        for s, m in zip(capped, maps):
            if i not in m:
                continue
            if not time_admissible(s["wall"], W):
                continue
            if first is None:
                first = s["wall"]
                first_pos = m[i]
            if s["wall"] >= W:
                credited = s["wall"]
                credited_pos = m[i]
                break
        avail = {}
        for d in DEADLINES:
            cap = W + d if eos_wall is None else min(W + d, eos_wall)
            cand = [(s, m) for s, m in zip(capped, maps) if s["wall"] <= cap]
            hit = False
            if cand:
                s, m = cand[-1]
                if (i in m and time_admissible(s["wall"], W)
                        and s["wall"] >= W):
                    hit = True
            avail[str(d)] = hit
        rows.append({"first_wall": first, "snap_pos": first_pos,
                     "neg_flag": bool(first is not None and first < W),
                     "first_credited_wall": credited,
                     "credited_snap_pos": credited_pos, "avail": avail})
    return rows


def enrich(ali: dict, words: list) -> dict:
    by_idx = {w["idx"]: w for w in words}
    for m in ali["matched"]:
        w = by_idx.get(m["group_idx"], {})
        m["speaker"] = w.get("speaker")
        m["first_prov_wall"] = w.get("first_prov_wall")
        m["final_wall"] = w.get("final_wall")
    return ali


def hungarian_map(pairs: list) -> dict:
    lefts = sorted({a for a, _ in pairs if a is not None}, key=str)
    rights = sorted({b for _, b in pairs}, key=str)
    count = {(a, b): sum(1 for x, y in pairs if x == a and y == b)
             for a in lefts for b in rights}
    best: dict = {}
    best_score = -1
    for perm in itertools.permutations(rights, min(len(lefts), len(rights))):
        score = sum(count.get((a, b), 0) for a, b in zip(lefts, perm))
        if score > best_score:
            best_score = score
            best = dict(zip(lefts, perm))
    return {"map": best, "score": best_score, "n_pairs": len(pairs),
            "labels_soniox": lefts, "labels_gt": rights}


def gt8_cohort(case: str) -> list:
    spec = S2FREEZE["cases"][case]
    out = []
    for side in ("left", "right"):
        for w in spec["gt_window"][side]:
            out.append({"id": w["id"], "text": w["text"], "side": side,
                        "in_span": bool(w["in_span"]),
                        "start": int(round(w["start"] * 16000)),
                        "end": int(round(w["end"] * 16000))})
    return out


def prior_groups(case: str) -> list:
    d = json.loads((STAGE2 / "captures" / f"{case}.json").read_text(encoding="utf-8"))
    return d["groups"]


def wer(ref_norms: list, hyp_norms: list) -> dict:
    """Standard Levenshtein edit distance, unit S/D/I costs, deterministic
    backtrace with stable tie priority diagonal then deletion then insertion."""
    n, m = len(ref_norms), len(hyp_norms)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_norms[i - 1] == hyp_norms[j - 1] else 1
            best = dp[i - 1][j - 1] + cost
            if dp[i - 1][j] + 1 < best:
                best = dp[i - 1][j] + 1
            if dp[i][j - 1] + 1 < best:
                best = dp[i][j - 1] + 1
            dp[i][j] = best
    i, j = n, m
    s = d = ins = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if ref_norms[i - 1] == hyp_norms[j - 1] else 1):
            if ref_norms[i - 1] != hyp_norms[j - 1]:
                s += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            d += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return {"S": s, "D": d, "I": ins, "N": n,
            "WER": round((s + d + ins) / n, 4) if n else None,
            "method": "levenshtein-dp-unit-cost-tie-diagonal-deletion-insertion"}


def mark_truncated(gt: list, lo: int, hi: int) -> list:
    for g in gt:
        g["truncated"] = bool(g["start"] < lo < g["end"] or g["start"] < hi < g["end"])
    return gt


def recognition_view(gt: list, gt_norms: list, words: list, lo: int, hi: int) -> dict:
    """Recognition attribution on a fixed GT set: align_window matched sets
    plus Levenshtein WER of the fixed ordered ref vs scoped hyp sequence."""
    scored = [w for w in words if w["norm"]]
    hyp = [w["norm"] for w in words
           if w["norm"] and w.get("end_src") is not None and lo <= w["end_src"] <= hi]
    ali = enrich(H.align_window(gt, scored, lo, hi), words)
    matched = ali["matched"]
    return {"n_matched": len(matched), "n_unmatched": len(ali["unmatched"]),
            "n_mixed": len(ali["mixed"]),
            "unmatched_ids": [m["gt"]["id"] for m in ali["unmatched"]],
            "mixed_ids": [m["gt"]["id"] for m in ali["mixed"]],
            "strict_recall": round(len(matched) / len(gt), 4) if gt else None,
            "wer": wer(gt_norms, hyp),
            "matched": [{"id": m["gt"]["id"], "role": m["gt"].get("role"),
                         "side": m["gt"].get("side"), "speaker": m.get("speaker"),
                         "group_idx": m["group_idx"], "end": m["gt"]["end"],
                         "final_wall": m.get("final_wall")} for m in matched]}


def timing_view(gt: list, p0: int, id2row: dict) -> dict:
    """Project full-table occurrence rows onto a scoped ID set. Negative
    receipt is flagged separately and never credited in medians or counts."""
    avail = {d: 0 for d in DEADLINES}
    lat_credited, neg_ids = [], []
    per_word = []
    for g in gt:
        r = id2row.get(g["id"])
        W = (g["end"] - p0) / 16000.0
        row: dict = {"id": g["id"], "first_wall": None, "neg_flag": False}
        if r is None:
            per_word.append(row)
            continue
        row["first_wall"] = r["first_wall"]
        if r["neg_flag"]:
            row["neg_flag"] = True
            neg_ids.append(g["id"])
        if r["first_credited_wall"] is not None:
            lat_credited.append(round(r["first_credited_wall"] - W, 3))
        for d in DEADLINES:
            hit = bool(r["avail"].get(str(d)))
            row[f"avail_{d}"] = hit
            avail[d] += hit
        per_word.append(row)
    return {"deadline_avail": {str(d): v for d, v in avail.items()},
            "deadline_note": ("full-list occurrence ordinal plus predeclared 3.0s "
                              "admissibility, latest snapshot at or before "
                              "min(deadline, EOS); negatives never credited"),
            "first_prov_lat": stats(lat_credited),
            "n_neg_first_prov": len(neg_ids), "neg_ids": neg_ids,
            "neg_note": ("flagged receipt before word end is reported, not counted, "
                         "not in medians"),
            "per_word": per_word}


def final_latency(matched: list, p0: int) -> dict:
    lat = []
    for m in matched:
        if m.get("final_wall") is not None:
            lat.append(round(m["final_wall"] - (m["end"] - p0) / 16000.0, 3))
    return stats(lat)


def summarize_scope(gt: list, gt_norms: list, words: list, pre_words: list,
                    lo: int, hi: int, p0: int, id2row: dict) -> dict:
    comp = recognition_view(gt, gt_norms, words, lo, hi)
    pre = recognition_view(gt, gt_norms, pre_words, lo, hi)
    timing = timing_view(gt, p0, id2row)
    return {"n_gt": len(gt), "n_truncated": sum(1 for g in gt if g.get("truncated")),
            "truncated_ids": [g["id"] for g in gt if g.get("truncated")],
            "wer_note": ("ref fixed GT time order vs scoped hyp; overlap regions "
                         "serialized in time order which bounds WER interpretation"),
            **{k: v for k, v in comp.items() if k != "matched"},
            "wer": comp["wer"], "strict_recall": comp["strict_recall"],
            "matched": comp["matched"],
            "pre_eos_final_only": {
                "n_matched": pre["n_matched"], "strict_recall": pre["strict_recall"],
                "wer": pre["wer"], "unmatched_ids": pre["unmatched_ids"],
                "note": ("recognition from finals received at or before EOS only; "
                         "gap to complete-final quantifies EOS-tail assistance")},
            **timing,
            "final_lat": final_latency(comp["matched"], p0)}


def stats(xs: list) -> dict:
    if not xs:
        return {"n": 0}
    s = sorted(xs)
    q = lambda p: s[min(len(s) - 1, int(p * len(s)))]
    return {"n": len(s), "median": q(0.5), "p95": q(0.95), "min": s[0], "max": s[-1]}


def score_session(sess: dict) -> dict:
    case = sess["case"]
    profile = sess.get("profile")
    p0 = sess["audio"]["src0"]
    stream = committed_stream(sess["messages"])
    words = aggregate_words(stream["committed"], stream["first_prov"],
                            stream["final_seen_wall"], p0)
    snaps = snapshots(sess["messages"])
    scored_words = [w for w in words if w["norm"]]
    out: dict = {
        "case": case, "profile": profile, "capture_status": sess.get("status"),
        "n_committed_tokens": len(stream["committed"]),
        "n_exact_duplicate_finals": stream["n_exact_duplicate_finals"],
        "violations": stream["violations"],
        "n_words": len(words), "n_scored_words": len(scored_words),
        "n_punc_excluded": len(words) - len(scored_words),
        "speaker_labels": sorted({w["speaker"] for w in words
                                  if w.get("speaker") is not None}, key=str),
        "n_unknown_speaker": sum(1 for w in words if w.get("speaker") is None),
        "n_mixed_speaker": sum(1 for w in words if w.get("mixed_speaker")),
        "eos_send": {"start": (sess.get("session") or {}).get("eos_send_start"),
                     "end": (sess.get("session") or {}).get("eos_send_end")},
        "finished": {"wall": (sess.get("session") or {}).get("finished_wall"),
                     "value": (sess.get("session") or {}).get("finished_value")}}

    eos_wall = (sess.get("session") or {}).get("eos_send_end")
    n_fin_pre = n_fin_post = 0
    for m in sess["messages"]:
        toks = m.get("tokens") or []
        nf = sum(1 for t in toks if bool(t.get("is_final")))
        if not nf or m.get("arrival_wall") is None:
            continue
        if eos_wall is not None and m["arrival_wall"] > eos_wall:
            n_fin_post += nf
        else:
            n_fin_pre += nf
    pre_words = [w for w in words
                 if w.get("final_wall") is not None
                 and (eos_wall is None or w["final_wall"] <= eos_wall)]
    post_words = [w for w in words
                  if w.get("final_wall") is not None
                  and eos_wall is not None and w["final_wall"] > eos_wall]
    out["eos_final_split"] = {
        "eos_wall": eos_wall,
        "n_emitted_final_tokens_pre_eos": n_fin_pre,
        "n_emitted_final_tokens_post_eos": n_fin_post,
        "n_unique_committed_words_pre_eos": len(pre_words),
        "n_unique_committed_words_post_eos": len(post_words),
        "n_never_final": sum(1 for w in words if w.get("final_wall") is None),
        "clock": "arrival_wall origin audio first send, same clock both sides"}

    if case in ("NP1", "NP2", "NP3"):
        gt8 = gt8_cohort(case)
        lo, hi = S2FREEZE["cases"][case]["scored_span_samples"]
        prior = H.align_window(gt8, prior_groups(case), lo - REGION_PAD, hi + REGION_PAD)
        out["np_frozen_cohort"] = {
            "matched": [m["gt"]["id"] for m in prior["matched"]],
            "unmatched": [m["gt"]["id"] for m in prior["unmatched"]],
            "mixed": [m["gt"]["id"] for m in prior["mixed"]]}
        b = S2FREEZE["cases"][case]["boundary_samples"]
        plo, phi = S2FREEZE["cases"][case]["payload_samples"]
        meet = S2FREEZE["cases"][case]["source"].split("_", 1)[1]
        gt_all = mark_truncated(
            [g for g in H.gt_words_in_span(meet, plo / 16000, phi / 16000)
             if H.norm_word(g["text"])], plo, phi)
        gt_alln = [H.norm_word(g["text"]) for g in gt_all]
        full_walls = [(g["end"] - p0) / 16000.0 for g in gt_all]
        table = align_full_table(gt_alln, full_walls, snaps, eos_wall)
        id2row = {g["id"]: r for g, r in zip(gt_all, table)}
        gt8n = [H.norm_word(g["text"]) for g in gt8]
        out["np_cohort"] = summarize_scope(gt8, gt8n, words, pre_words,
                                           plo, phi, p0, id2row)
        out["np_cohort"]["cohort_binary_speaker"] = cohort_binary_speaker(
            out["np_cohort"], gt8)
        out["np_cohort"]["note"] = ("binary L/R cohort metric on frozen 8 IDs with "
                                    "whole-payload hyp aligned once, explicit, "
                                    "not full speaker diarization")
        full = summarize_scope(gt_all, gt_alln, words, pre_words,
                               plo, phi, p0, id2row)
        out["np_full"] = full
        out["session_speaker_map"] = session_map(full, by="role")
        out["session_speaker_map"]["scope"] = "full payload matched, pure A-D roles"
    else:
        whole = mark_truncated(
            [g for g in H.gt_words_in_span("EN2009d", 0, 755520 / 16000)
             if H.norm_word(g["text"])], 0, 755520)
        whole_n = [H.norm_word(g["text"]) for g in whole]
        whole_walls = [(g["end"] - p0) / 16000.0 for g in whole]
        table = align_full_table(whole_n, whole_walls, snaps, eos_wall)
        id2row = {g["id"]: r for g, r in zip(whole, table)}
        scopes = {}
        session_pairs = []
        for name, (lo, hi) in FREEZE["cases"]["EN2009d"]["scopes"].items():
            gt = mark_truncated(
                [g for g in H.gt_words_in_span("EN2009d", lo / 16000, hi / 16000)
                 if H.norm_word(g["text"])], lo, hi)
            for g in gt:
                g["overlap"] = any(o["role"] != g["role"] and o["start"] < g["end"]
                                   and g["start"] < o["end"] for o in gt)
            gtn = [H.norm_word(g["text"]) for g in gt]
            s = summarize_scope(gt, gtn, words, pre_words, lo, hi, p0, id2row)
            s["n_overlap_gt"] = sum(1 for g in gt if g["overlap"])
            s["n_nonoverlap_gt"] = sum(1 for g in gt if not g["overlap"])
            scopes[name] = s
            if name == "COMBINED":
                session_pairs = [(m["speaker"], m["role"]) for m in s["matched"]]
        out["scopes"] = scopes
        mp = hungarian_map([(a if a is not None else "__unknown__", b)
                            for a, b in session_pairs])
        out["session_speaker_map"] = {
            "hungarian": mp["map"], "n_pairs": len(session_pairs),
            "n_correct_mapped": sum(1 for a, b in session_pairs
                                    if mp["map"].get(a if a is not None else "__unknown__") == b),
            "n_unmapped_speaker": sum(1 for a, _ in session_pairs if a is None),
            "scope": ("one per-session map over COMBINED matched pure A-D roles, "
                      "applied consistently to R2 T1 COMBINED, never scope independent"),
            "note": "retrospective permutation optimistic, disclosed, not live identity"}
        whole_gt = [g for g in H.gt_words_in_span("EN2009d", 0, 755520 / 16000)
                    if H.norm_word(g["text"])]
        out["whole_session_denom"] = {
            "n_gt_all": len(whole_gt),
            "n_gt_scored_scopes_unique": len({g["id"] for s in scopes.values()
                                              for g in ([{"id": m["id"]} for m in s["matched"]]
                                                        + [{"id": i} for i in s["unmatched_ids"]])}),
            "note": ("denominator reported once per session; R2 T1 COMBINED rows "
                     "overlap and are never summed")}
    out["words"] = words
    return out
def provenance_correction() -> dict:
    import os
    from datetime import datetime, timezone
    fz = EXP / "FREEZE.json"
    mtime = datetime.fromtimestamp(os.path.getmtime(fz), timezone.utc).isoformat()
    first = None
    jp = EXP / "captures" / "attempt_journal.jsonl"
    if jp.exists():
        for line in jp.read_text(encoding="utf-8").splitlines():
            e = json.loads(line) if line.strip() else {}
            if e.get("freeze_id") == FREEZE["freeze_id"] and e.get("event") == "attempt-journaled":
                first = e.get("utc")
                break
    return {"corrected_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "v3_embedded_frozen_at": FREEZE.get("frozen_at_utc"),
            "v3_freeze_file_mtime_utc": mtime,
            "v3_first_connect_utc_journal": first,
            "finding": ("embedded frozen_at_utc is a false future string and is "
                        "withdrawn as provenance; readonly mtime plus journal show "
                        "the freeze file was written before the first connect; "
                        "no preregistration-time claim is made; no backdating "
                        "performed, FREEZE.json bytes unchanged")}


def cohort_binary_speaker(cohort: dict, gt8: list) -> dict:
    by_id = {g["id"]: g for g in gt8}
    pairs = []
    for m in cohort["matched"]:
        side = by_id.get(m["id"], {}).get("side")
        owner = {"left": "L", "right": "R"}.get(side, "?")
        pairs.append((m["speaker"], owner))
    mp = hungarian_map([(a if a is not None else "__unknown__", b) for a, b in pairs])
    return {"hungarian": mp["map"], "n_pairs": len(pairs),
            "n_correct_mapped": sum(1 for a, b in pairs
                                    if mp["map"].get(a if a is not None else "__unknown__") == b),
            "n_unmapped_speaker": sum(1 for a, _ in pairs if a is None),
            "note": "binary cohort sides only, retrospective, not diarization"}


def session_map(full: dict, by: str) -> dict:
    pairs = [(m["speaker"], m.get(by)) for m in full["matched"]]
    mp = hungarian_map([(a if a is not None else "__unknown__", b) for a, b in pairs])
    return {"hungarian": mp["map"], "n_pairs": len(pairs),
            "n_correct_mapped": sum(1 for a, b in pairs
                                    if mp["map"].get(a if a is not None else "__unknown__") == b),
            "n_unmapped_speaker": sum(1 for a, _ in pairs if a is None),
            "note": "retrospective permutation optimistic, disclosed, not live identity"}


def main() -> None:
    scored = []
    for s in FREEZE["sessions"]:
        p = EXP / "captures" / f"{s['case']}-{s['profile']}.json"
        if not p.exists():
            scored.append({"case": s["case"], "profile": s["profile"],
                           "status": "missing-not-pass"})
            continue
        sess = json.loads(p.read_text(encoding="utf-8"))
        try:
            r = score_session(sess)
            r["status"] = "ok"
        except Exception as exc:
            r = {"case": s["case"], "profile": s["profile"],
                 "status": "scoring-error", "detail": f"{type(exc).__name__}: {exc}"}
        scored.append(r)
    (EXP / "ledger.json").write_text(
        json.dumps({"freeze_id": FREEZE["freeze_id"],
                    "provenance_correction": provenance_correction(),
                    "sessions": scored},
                   indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for r in scored:
        extra = ""
        if r.get("status") == "ok":
            if "np_cohort" in r:
                extra = (f" coh={r['np_cohort']['n_matched']}/{r['np_cohort']['n_gt']}"
                         f" full={r['np_full']['n_matched']}/{r['np_full']['n_gt']}"
                         f" wer={r['np_full']['wer']['WER']}")
            else:
                extra = " " + " ".join(
                    f"{k}={v['n_matched']}/{v['n_gt']},wer={v['wer']['WER']}"
                    for k, v in r["scopes"].items())
        print(f"{r['case']}-{r['profile']} {r['status']} cap={r.get('capture_status')}{extra}")


if __name__ == "__main__":
    main()
