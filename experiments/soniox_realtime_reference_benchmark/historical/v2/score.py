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
def prefix_aligned(gt_norms: list, snap_norms: list, target: int) -> bool:
    """True iff GT occurrence <target> aligns within the sequence alignment
    of the GT prefix GT[0..target] against the snapshot display, with
    occurrence attribution: the covering equal block must have length >= 2
    (a run; isolated singleton surface matches cannot attribute occurrence),
    except when the GT prefix itself is a single word. Frozen rule."""
    sm = difflib.SequenceMatcher(None, gt_norms[:target + 1], snap_norms, autojunk=False)
    for tag, alo, ahi, _, _ in sm.get_opcodes():
        if tag == "equal" and alo <= target < ahi:
            if target == 0 or (ahi - alo) >= 2:
                return True
    return False


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




def occurrence_availability(gt_norms: list, snaps: list, target: int,
                            deadline_wall: float) -> tuple[bool, object]:
    """Latest snapshot at or before the deadline only; past never future."""
    cand = [s for s in snaps if s["wall"] <= deadline_wall]
    if not cand:
        return False, None
    snap = cand[-1]
    if prefix_aligned(gt_norms, snap["norms"], target):
        return True, snap["wall"]
    return False, None


def occurrence_first_prov(gt_norms: list, snaps: list, target: int) -> object:
    for s in snaps:
        if prefix_aligned(gt_norms, s["norms"], target):
            return s["wall"]
    return None


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
    sm = difflib.SequenceMatcher(None, ref_norms, hyp_norms, autojunk=False)
    s = d = ins = 0
    for tag, alo, ahi, blo, bhi in sm.get_opcodes():
        if tag == "replace":
            s += max(ahi - alo, bhi - blo)
        elif tag == "delete":
            d += ahi - alo
        elif tag == "insert":
            ins += bhi - blo
    n = len(ref_norms)
    return {"S": s, "D": d, "I": ins, "N": n,
            "WER": round((s + d + ins) / n, 4) if n else None}


def mark_truncated(gt: list, lo: int, hi: int) -> list:
    for g in gt:
        g["truncated"] = bool(g["start"] < lo < g["end"] or g["start"] < hi < g["end"])
    return gt


def summarize_scope(gt: list, words: list, scored_words: list, lo: int, hi: int,
                    p0: int, snaps: list, gt_norms: list) -> dict:
    ali = enrich(H.align_window(gt, scored_words, lo, hi), scored_words)
    matched = ali["matched"]
    avail = {d: 0 for d in DEADLINES}
    lat_first, lat_final = [], []
    n_neg = 0
    per_word = []
    for pos, g in enumerate(gt):
        wend_wall = (g["end"] - p0) / 16000.0
        row: dict = {"id": g["id"]}
        fp = occurrence_first_prov(gt_norms, snaps, pos)
        row["first_prov_wall"] = fp
        if fp is not None:
            lat_first.append(round(fp - wend_wall, 3))
            if fp < wend_wall:
                n_neg += 1
                row["neg_flag"] = True
        for d in DEADLINES:
            hit, _ = occurrence_availability(gt_norms, snaps, pos, wend_wall + d)
            row[f"avail_{d}"] = hit
            avail[d] += bool(hit)
        per_word.append(row)
    for m in matched:
        if m.get("final_wall") is not None:
            lat_final.append(round(m["final_wall"] - (m["gt"]["end"] - p0) / 16000.0, 3))
    hyp = [w["norm"] for w in words
           if w["norm"] and w.get("end_src") is not None and lo <= w["end_src"] <= hi]
    return {"n_gt": len(gt), "n_truncated": sum(1 for g in gt if g.get("truncated")),
            "truncated_ids": [g["id"] for g in gt if g.get("truncated")],
            "n_matched": len(matched), "n_unmatched": len(ali["unmatched"]),
            "n_mixed": len(ali["mixed"]),
            "unmatched_ids": [m["gt"]["id"] for m in ali["unmatched"]],
            "mixed_ids": [m["gt"]["id"] for m in ali["mixed"]],
            "strict_recall": round(len(matched) / len(gt), 4) if gt else None,
            "wer": wer(gt_norms, hyp),
            "wer_note": ("ref fixed GT time order vs scoped hyp; overlap regions "
                         "serialized in time order which bounds WER interpretation"),
            "deadline_avail": {str(d): v for d, v in avail.items()},
            "deadline_note": ("occurrence prefix alignment to latest snapshot at or "
                              "before deadline; never bag membership, never final text"),
            "first_prov_lat": stats(lat_first), "n_neg_first_prov": n_neg,
            "neg_note": "negative flags time alignment, not proof the provider predicts",
            "final_lat": stats(lat_final), "per_word": per_word,
            "matched": [{"id": m["gt"]["id"], "role": m["gt"].get("role"),
                         "side": m["gt"].get("side"), "speaker": m.get("speaker"),
                         "group_idx": m["group_idx"]} for m in matched]}


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

    if case in ("NP1", "NP2", "NP3"):
        gt8 = gt8_cohort(case)
        lo, hi = S2FREEZE["cases"][case]["scored_span_samples"]
        prior = H.align_window(gt8, prior_groups(case), lo - REGION_PAD, hi + REGION_PAD)
        out["np_frozen_cohort"] = {
            "matched": [m["gt"]["id"] for m in prior["matched"]],
            "unmatched": [m["gt"]["id"] for m in prior["unmatched"]],
            "mixed": [m["gt"]["id"] for m in prior["mixed"]]}
        gt8n = [H.norm_word(g["text"]) for g in gt8]
        out["np_cohort"] = summarize_scope(gt8, words, scored_words, lo, hi, p0,
                                           snaps, gt8n)
        out["np_cohort"]["cohort_binary_speaker"] = cohort_binary_speaker(
            out["np_cohort"], gt8)
        out["np_cohort"]["note"] = ("binary L/R cohort metric, explicit, "
                                    "not full speaker diarization")
        b = S2FREEZE["cases"][case]["boundary_samples"]
        plo, phi = S2FREEZE["cases"][case]["payload_samples"]
        meet = S2FREEZE["cases"][case]["source"].split("_", 1)[1]
        gt_all = mark_truncated(
            [g for g in H.gt_words_in_span(meet, plo / 16000, phi / 16000)
             if H.norm_word(g["text"])], plo, phi)
        gt_alln = [H.norm_word(g["text"]) for g in gt_all]
        full = summarize_scope(gt_all, words, scored_words, plo, phi, p0, snaps, gt_alln)
        out["np_full"] = full
        out["session_speaker_map"] = session_map(full, by="role")
        out["session_speaker_map"]["scope"] = "full payload matched, pure A-D roles"
    else:
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
            s = summarize_scope(gt, words, scored_words, lo, hi, p0, snaps, gtn)
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
        json.dumps({"freeze_id": FREEZE["freeze_id"], "sessions": scored},
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
