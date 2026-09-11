"""Soniox reference scoring: reconstruct committed finals, align to human GT only.

Reuses existing word helpers (norm_word, load_gt_words, gt_words_in_span,
align_window) via read-only import. No Soniox-as-GT. No PSEM mutation.
"""
from __future__ import annotations

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


def load_helpers():
    spec = importlib.util.spec_from_file_location(
        "ds_replay_helpers",
        ROOT / "experiments" / "psem_decision_sufficiency" / "replay.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


H = load_helpers()


def committed_stream(messages: list) -> dict:
    """Mirror provider merge on final-flagged tokens, carrying speaker.

    Returns committed token list plus lifecycle diagnostics. Final revisions
    are protocol violations: reported, never silent.
    """
    committed: list = []
    first_prov: dict = {}
    final_seen_wall: dict = {}
    violations: list = []
    ignored_batches = 0
    pending_replace_snapshots = 0
    for m in messages:
        toks = m.get("tokens") or []
        wall = m.get("arrival_wall")
        if toks and not any(bool(t.get("is_final")) for t in toks):
            pending_replace_snapshots += 1
        for t in toks:
            key = (str(t.get("text", "")), t.get("end_ms"))
            if key not in first_prov and wall is not None:
                first_prov[key] = wall
            if bool(t.get("is_final")) and wall is not None:
                final_seen_wall[key] = wall
        finals = [t for t in toks if bool(t.get("is_final"))
                  and str(t.get("text", "") or "") not in ("<fin>", "<end>")]
        if not finals:
            continue
        if not committed:
            committed = [dict(t) for t in finals]
            continue
        new_max = max((t.get("end_ms") if isinstance(t.get("end_ms"), (int, float)) else -1)
                      for t in finals)
        old_max = max((t.get("end_ms") if isinstance(t.get("end_ms"), (int, float)) else -1)
                      for t in committed)
        if new_max < old_max:
            ignored_batches += 1
            continue
        new_first = min((t.get("end_ms") if isinstance(t.get("end_ms"), (int, float)) else 10**18)
                        for t in finals)
        cut = None
        for i, t in enumerate(committed):
            e = t.get("end_ms")
            if isinstance(e, (int, float)) and e >= new_first:
                cut = i
                break
        if cut is None:
            before = {(t.get("text"), t.get("end_ms")) for t in committed}
            committed = committed + [dict(t) for t in finals]
        elif cut == 0:
            old = {(t.get("text"), t.get("end_ms")) for t in committed}
            new = {(t.get("text"), t.get("end_ms")) for t in finals}
            if old != new and len(committed) and len(finals):
                overlap_old = [t for t in committed if t.get("end_ms") in
                               {f.get("end_ms") for f in finals}]
                overlap_new = [t for t in finals if t.get("end_ms") in
                               {c.get("end_ms") for c in committed}]
                if {(t.get("text")) for t in overlap_old} != {(t.get("text")) for t in overlap_new}:
                    violations.append({"kind": "final-revised",
                                       "old": overlap_old[:4], "new": overlap_new[:4]})
            committed = [dict(t) for t in finals]
        else:
            dropped = committed[cut:]
            kept_ends = {t.get("end_ms") for t in committed[:cut]}
            repl = [t for t in finals if t.get("end_ms") not in kept_ends]
            for d in dropped:
                for r in repl:
                    if d.get("end_ms") == r.get("end_ms") and d.get("text") != r.get("text"):
                        violations.append({"kind": "final-revised",
                                           "old": [d], "new": [r]})
                        break
            committed = committed[:cut] + [dict(t) for t in finals]
    return {"committed": committed, "first_prov": first_prov,
            "final_seen_wall": final_seen_wall, "violations": violations,
            "ignored_batches": ignored_batches,
            "pending_replace_snapshots": pending_replace_snapshots}


def aggregate_words(committed: list, first_prov: dict, final_seen_wall: dict,
                    p0: int) -> list:
    chars: list = []
    owners: list = []
    for t in committed:
        s = str(t.get("text", "") or "")
        for _ in s:
            chars.append(s[:1])
            owners.append(t)
        # rebuild per char correctly
        for j, ch in enumerate(s):
            chars[len(chars) - len(s) + j] = ch
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
        keys = [(t.get("text"), t.get("end_ms")) for t in toks]
        provs = [first_prov[k] for k in keys if k in first_prov]
        fins = [final_seen_wall[k] for k in keys if k in final_seen_wall]
        words.append({"idx": idx, "word": "".join(chars[i:j]),
                      "norm": H.norm_word("".join(chars[i:j])),
                      "start_ms": start_ms, "end_ms": end_ms,
                      "end_src": (p0 + int(end_ms) * 16) if isinstance(end_ms, (int, float)) else None,
                      "start_src": (p0 + int(start_ms) * 16) if isinstance(start_ms, (int, float)) else None,
                      "speaker": lasts.get("speaker"),
                      "speakers": speakers,
                      "mixed_speaker": len({s for s in speakers if s is not None}) > 1,
                      "first_prov_wall": min(provs) if provs else None,
                      "final_wall": max(fins) if fins else None})
        idx += 1
        i = j
    return words


def enrich(ali: dict, words: list) -> dict:
    by_idx = {w["idx"]: w for w in words}
    for m in ali["matched"]:
        w = by_idx.get(m["group_idx"], {})
        m["speaker"] = w.get("speaker")
        m["first_prov_wall"] = w.get("first_prov_wall")
        m["final_wall"] = w.get("final_wall")
        m["word_start_src"] = w.get("start_src")
    return ali


def prov_timeline(messages: list) -> list:
    out = []
    for m in messages:
        toks = m.get("tokens") or []
        if not toks or m.get("arrival_wall") is None:
            continue
        out.append({"wall": m["arrival_wall"],
                    "norms": [H.norm_word(str(t.get("text", "") or "")) for t in toks],
                    "speakers": [t.get("speaker") for t in toks]})
    return out


def prov_set_at(timeline: list, deadline_wall: float) -> set:
    cur: set = set()
    for snap in timeline:
        if snap["wall"] <= deadline_wall:
            cur = set(snap["norms"])
        else:
            break
    return cur


def hungarian_map(pairs: list) -> dict:
    """Max-count map from soniox speaker id to GT role. Retrospective optimistic."""
    lefts = sorted({a for a, _ in pairs if a is not None})
    rights = sorted({b for _, b in pairs})
    count = {(a, b): sum(1 for x, y in pairs if x == a and y == b) for a in lefts for b in rights}
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


def score_session(sess: dict) -> dict:
    case, profile = sess["case"], sess["profile"]
    p0 = sess["audio"]["src0"]
    stream = committed_stream(sess["messages"])
    words = aggregate_words(stream["committed"], stream["first_prov"],
                            stream["final_seen_wall"], p0)
    timeline = prov_timeline(sess["messages"])
    scored_words = [w for w in words if w["norm"]]
    n_punc_excluded = len(words) - len(scored_words)

    out: dict = {"case": case, "profile": profile,
                 "n_committed_tokens": len(stream["committed"]),
                 "n_words": len(words), "n_scored_words": len(scored_words),
                 "n_punc_excluded": n_punc_excluded,
                 "violations": stream["violations"],
                 "ignored_batches": stream["ignored_batches"],
                 "pending_replace_snapshots": stream["pending_replace_snapshots"],
                 "speaker_labels": sorted({w["speaker"] for w in words if w.get("speaker") is not None},
                                          key=str),
                 "n_unknown_speaker": sum(1 for w in words if w.get("speaker") is None),
                 "n_mixed_speaker": sum(1 for w in words if w.get("mixed_speaker"))}

    if case in ("NP1", "NP2", "NP3"):
        gt8 = gt8_cohort(case)
        lo, hi = S2FREEZE["cases"][case]["scored_span_samples"]
        prior = H.align_window(gt8, prior_groups(case), lo - 32000, hi + 32000)
        frozen = {"matched": [m["gt"]["id"] for m in prior["matched"]],
                  "unmatched": [m["gt"]["id"] for m in prior["unmatched"]],
                  "mixed": [m["gt"]["id"] for m in prior["mixed"]]}
        ali = enrich(H.align_window(gt8, scored_words, lo - 32000, hi + 32000), scored_words)
        b = S2FREEZE["cases"][case]["boundary_samples"]
        plo, phi = S2FREEZE["cases"][case]["payload_samples"]
        meet = S2FREEZE["cases"][case]["source"].split("_", 1)[1]
        gt_all = H.gt_words_in_span(meet, plo / 16000, phi / 16000)
        for g in gt_all:
            g["side"] = "left" if g["end"] <= b else "right"
            g["crossing"] = bool(g["start"] < b < g["end"])
        gt_all_np = [g for g in gt_all if H.norm_word(g["text"])]
        ali_full = enrich(H.align_window(gt_all_np, scored_words, plo, phi), scored_words)
        out["np_frozen_cohort"] = frozen
        out["np_cohort"] = summarize_alignment(ali, p0, timeline, gt8)
        out["np_cohort"]["speaker"] = speaker_score(ali, gt8)
        out["np_full"] = summarize_alignment(ali_full, p0, timeline, gt_all_np)
        out["np_full"]["speaker"] = None
    else:
        scopes = {}
        for name, (lo, hi) in FREEZE["cases"]["EN2009d"]["scopes"].items():
            gt = H.gt_words_in_span("EN2009d", lo / 16000, hi / 16000)
            gt = [g for g in gt if H.norm_word(g["text"])]
            for g in gt:
                g["overlap"] = any(o["role"] != g["role"] and o["start"] < g["end"]
                                   and g["start"] < o["end"] for o in gt)
            ali = enrich(H.align_window(gt, scored_words, lo, hi), scored_words)
            s = summarize_alignment(ali, p0, timeline, gt)
            s["speaker"] = speaker_score_roles(ali, gt)
            s["n_overlap_gt"] = sum(1 for g in gt if g["overlap"])
            s["n_nonoverlap_gt"] = sum(1 for g in gt if not g["overlap"])
            scopes[name] = s
        out["scopes"] = scopes
    eos = (sess.get("session") or {}).get("eos_wall")
    if eos is not None:
        pre = [w for w in words if w.get("final_wall") is not None and w["final_wall"] <= eos]
        post = [w for w in words if w.get("final_wall") is not None and w["final_wall"] > eos]
        out["eos_split"] = {"eos_wall": eos, "n_final_pre_eos": len(pre),
                            "n_final_post_eos": len(post),
                            "n_never_final": sum(1 for w in words if w.get("final_wall") is None)}
    out["words"] = words
    return out


def summarize_alignment(ali: dict, p0: int, timeline: list, gt: list) -> dict:
    matched = ali["matched"]
    lat_first, lat_final = [], []
    n_neg_first = 0
    avail = {d: 0 for d in DEADLINES}
    per_word = []
    for m in matched:
        g = m["gt"]
        wend_wall = (g["end"] - p0) / 16000.0
        fp, fw = m.get("first_prov_wall"), m.get("final_wall")
        if fp is not None:
            lat_first.append(round(fp - wend_wall, 3))
            if fp < wend_wall:
                n_neg_first += 1
        if fw is not None:
            lat_final.append(round(fw - wend_wall, 3))
        row = {"id": g["id"], "first_prov_lat": (round(fp - wend_wall, 3) if fp is not None else None),
               "final_lat": (round(fw - wend_wall, 3) if fw is not None else None)}
        for d in DEADLINES:
            hit = bool(m["group_word"] and H.norm_word(m["group_word"]) in prov_set_at(timeline, wend_wall + d))
            row[f"avail_{d}"] = hit
            avail[d] += bool(hit)
        per_word.append(row)
    n_gt = len(gt)
    return {"n_gt": n_gt, "n_matched": len(matched),
            "n_unmatched": len(ali["unmatched"]), "n_mixed": len(ali["mixed"]),
            "unmatched_ids": [m["gt"]["id"] for m in ali["unmatched"]],
            "mixed_ids": [m["gt"]["id"] for m in ali["mixed"]],
            "deadline_avail": {str(d): v for d, v in avail.items()},
            "first_prov_lat": stats(lat_first), "n_neg_first_prov": n_neg_first,
            "final_lat": stats(lat_final), "per_word": per_word}


def stats(xs: list) -> dict:
    if not xs:
        return {"n": 0}
    s = sorted(xs)
    q = lambda p: s[min(len(s) - 1, int(p * len(s)))]
    return {"n": len(s), "median": q(0.5), "p95": q(0.95),
            "min": s[0], "max": s[-1]}


def speaker_score(ali: dict, gt: list) -> dict:
    by_id = {g["id"]: g for g in gt}
    pairs = []
    for m in ali["matched"]:
        owner = {"left": "L", "right": "R"}.get(m["gt"].get("side"), "?")
        sp = m.get("speaker")
        pairs.append((sp, owner))
    m = hungarian_map([(a if a is not None else "__unknown__", b) for a, b in pairs])
    mp = m["map"]
    correct = sum(1 for a, b in pairs
                  if mp.get(a if a is not None else "__unknown__") == b)
    return {"hungarian": mp, "n_pairs": len(pairs),
            "n_correct_mapped": correct,
            "n_unmapped_speaker": sum(1 for a, _ in pairs if a is None),
            "note": "retrospective permutation optimistic per-session global map"}


def speaker_score_roles(ali: dict, gt: list) -> dict:
    by_id = {g["id"]: g for g in gt}
    pairs = []
    for m in ali["matched"]:
        g = by_id.get(m["gt"]["id"], {})
        pairs.append((m.get("speaker"), g.get("role", "?")))
    m = hungarian_map([(a if a is not None else "__unknown__", b) for a, b in pairs])
    mp = m["map"]
    correct = sum(1 for a, b in pairs
                  if mp.get(a if a is not None else "__unknown__") == b)
    return {"hungarian": mp, "n_pairs": len(pairs),
            "n_correct_mapped": correct,
            "n_unmapped_speaker": sum(1 for a, _ in pairs if a is None),
            "note": "retrospective permutation optimistic per-session global map"}


def main() -> None:
    scored = []
    for s in FREEZE["sessions"]:
        p = EXP / "captures" / f"{s['case']}-{s['profile']}.json"
        if not p.exists():
            scored.append({"case": s["case"], "profile": s["profile"],
                           "status": "missing-not-pass"})
            continue
        sess = json.loads(p.read_text(encoding="utf-8"))
        if sess.get("session", {}).get("failure"):
            scored.append({"case": s["case"], "profile": s["profile"],
                           "status": "failed", "detail": sess["session"]["failure"]})
            continue
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
                c, f = r["np_cohort"], r["np_full"]
                extra = (f" cohort={c['n_matched']}/{c['n_gt']} full={f['n_matched']}/{f['n_gt']}"
                         f" spk={c['speaker']['n_correct_mapped']}/{c['speaker']['n_pairs']}")
            else:
                extra = " " + " ".join(
                    f"{k}={v['n_matched']}/{v['n_gt']}" for k, v in r["scopes"].items())
        print(f"{r['case']}-{r['profile']} {r['status']}{extra}")


if __name__ == "__main__":
    main()
