#!/usr/bin/env python3
"""Gate 0 builder: freeze PSEM-SMALL-MODEL-PROBE-v1 manifest (issue #117).

Derives episodes from V2 GT intervals only (no model outputs, no thresholds,
no training). GT interval source: cached relative-occupancy manifests
(dev + eval), which embed GT `intervals / transitions / topology_episodes`
regenerated from the frozen V2 reference checkout. Waveform bytes are NOT
read (spans are sample/time metadata only); audio resolution via `audio_ref`
against PSEM_CORPUS_ROOT is a documented follow-up.

Strata mapping (V2 primary_topology -> C1..C6):
  C1 continuity:          same_speaker_silence_gap_resume,
                          same_speaker_direct_continuation (valid only)
  C2 clean transfer:      silence_gap_different_speaker_handoff,
                          clean_direct_different_speaker_handoff,
                          micro_gap_different_speaker_handoff (valid only)
  C3 overlap return:      overlap_return, micro_overlap_return,
                          overlap_gap_return (valid only)
  C4 overlap takeover:    overlap_takeover, micro_overlap_takeover (valid only)
  C5 short interruption:  short_backchannel_return episodes (eligible only)
  C6 binding/uncertain:   masked transitions (any ambiguous topology) or
                          ambiguous_nonlexical / continuity_unknown / mixed /
                          complex / overlap_to_silence_unresolved (any mask)

CAL12 topology groups (schema enum) <- source:
  A            <- C1   A+B          <- C6 (overlap at/near boundary)
  A+A+B        <- C5 with overlap entry
  A->A+B->A    <- C3 micro/overlap_gap_return
  A->A+B->B    <- C4
  overlap_return <- C3 primary_topology == overlap_return
"""
from __future__ import annotations

import hashlib
from bisect import bisect_right
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OCC = {
    "dev": REPO / "experiments/psem_relative_occupancy_gate/results/dev/relative_occupancy_manifest.jsonl",
    "eval": REPO / "experiments/psem_relative_occupancy_gate/results/eval/relative_occupancy_manifest.jsonl",
}
V2_FREEZE = REPO / "experiments/psem_training_strategy_gate/data/v2/dataset_freeze.json"
OUT_DIR = Path(__file__).resolve().parent
MANIFEST = OUT_DIR / "manifest.jsonl"
FREEZE = OUT_DIR / "dataset_freeze.json"

SCHEMA = "psem.small_probe.manifest.v1"
SR = 16000
NATIVE_SAMPLES = 5 * SR
CAUSAL_SAMPLES = 1 * SR
ACTIVITY_MIN = 0.95
CAUSAL_ACTIVITY_MIN = 0.50
EVAL_PRE_MS, EVAL_POST_MS = 2000, 3000
MS = SR // 1000  # samples per ms
C1 = {"same_speaker_silence_gap_resume", "same_speaker_direct_continuation"}
C2 = {"silence_gap_different_speaker_handoff", "clean_direct_different_speaker_handoff",
      "micro_gap_different_speaker_handoff"}
C3 = {"overlap_return", "micro_overlap_return", "overlap_gap_return"}
C4 = {"overlap_takeover", "micro_overlap_takeover"}
C6 = {"ambiguous_nonlexical_vocalization_region", "ambiguous_nonlexical_vocalization_crossing",
      "continuity_unknown", "mixed_unresolved_transition", "complex_overlap_region",
      "complex_overlap_transition", "overlap_to_silence_unresolved"}


class FailClosed(RuntimeError):
    pass


def load_gt():
    sessions = {}
    for role, path in OCC.items():
        if not path.is_file():
            raise FailClosed(f"missing GT interval source: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("intervals"):
                raise FailClosed(f"no GT intervals for {r.get('source_id')}")
            key = (r["corpus"], r["session_id"])
            if key in sessions:
                raise FailClosed(f"duplicate session {key}")
            corpus = {"AMI": "ami", "AliMeeting": "alimeeting"}.get(r["corpus"])
            if corpus is None:
                raise FailClosed(f"unknown corpus {r['corpus']}")
            sessions[key] = {
                "corpus": corpus, "session_id": r["session_id"],
                "audio_ref": r["audio_ref"], "duration": int(r["source_duration_samples"]),
                "intervals": r["intervals"], "transitions": r["transitions"],
                "episodes": r.get("topology_episodes", []),
                "starts": [iv["start_sample"] for iv in r["intervals"]],
            }
    return sessions


def iter_ov(sess, a, b):
    ivs, starts = sess["intervals"], sess["starts"]
    i = max(0, bisect_right(starts, a) - 1)
    n = len(ivs)
    while i < n and ivs[i]["start_sample"] < b:
        if ivs[i]["end_sample"] > a:
            yield ivs[i]
        i += 1


def active_at(sess, s):
    for iv in iter_ov(sess, s, s + 1):
        return iv
    return None


def run_bounds(sess, anchor, t_end):
    """Maximal A-only (A or silence), unmasked runs ending <= t_end."""
    runs = []
    cur = None
    for iv in sess["intervals"]:
        if iv["start_sample"] >= t_end:
            break
        ok = (not iv["masked"]) and all(sp == anchor for sp in iv["active_speakers"])
        if ok:
            cur = (cur[0], iv["end_sample"]) if cur else (iv["start_sample"], iv["end_sample"])
        else:
            if cur:
                runs.append(cur)
                cur = None
    if cur:
        runs.append(cur)
    return [(a, min(b, t_end)) for a, b in runs if min(b, t_end) > a]
def single_before(sess, s):
    """Last single-speaker interval near/before s, skipping silence (<=4 s lookback)."""
    ivs, starts = sess["intervals"], sess["starts"]
    i = bisect_right(starts, s) - 1
    while i >= 0:
        iv = ivs[i]
        if iv["end_sample"] <= s - 4 * SR:
            return None
        if iv["masked"] or len(iv["active_speakers"]) > 1:
            return None
        if len(iv["active_speakers"]) == 1:
            return iv
        i -= 1
    return None


def single_after(sess, s):
    """First single-speaker interval near/after s, skipping silence (<=4 s lookahead)."""
    ivs = sess["intervals"]
    i = bisect_right(sess["starts"], s)
    if i > 0 and ivs[i - 1]["end_sample"] > s:
        i -= 1
    while i < len(ivs):
        iv = ivs[i]
        if iv["start_sample"] >= s + 4 * SR:
            return None
        if iv["masked"] or len(iv["active_speakers"]) > 1:
            return None
        if len(iv["active_speakers"]) == 1:
            return iv
        i += 1
    return None


def activity(sess, anchor, a, b):
    hit = 0
    for iv in iter_ov(sess, a, b):
        if anchor in iv["active_speakers"]:
            hit += min(iv["end_sample"], b) - max(iv["start_sample"], a)
    return hit / (b - a) if b > a else 0.0


def other_free(sess, anchor, a, b):
    for iv in iter_ov(sess, a, b):
        if iv["masked"]:
            return False
        for sp in iv["active_speakers"]:
            if sp != anchor:
                return False
    return True


def native_span(sess, anchor, t_end):
    cands = [r for r in run_bounds(sess, anchor, t_end) if r[1] - r[0] >= NATIVE_SAMPLES]
    cands.sort(key=lambda r: (r[1] - r[0], r[1]), reverse=True)  # longest run first
    for rs, re in cands:
        # latest qualifying 5 s ms-aligned window, 100 ms steps
        m = (re - NATIVE_SAMPLES) // 1600 * 1600
        while m >= rs and m + NATIVE_SAMPLES <= re:
            if activity(sess, anchor, m, m + NATIVE_SAMPLES) > ACTIVITY_MIN:
                return m // 16, m // 16 + 5000
            m -= 1600
    return None


def causal_span(sess, anchor, t_end):
    m = (t_end - CAUSAL_SAMPLES) // 1600 * 1600
    while m >= 0 and m + CAUSAL_SAMPLES <= t_end:
        if other_free(sess, anchor, m, m + CAUSAL_SAMPLES) and \
                activity(sess, anchor, m, m + CAUSAL_SAMPLES) >= CAUSAL_ACTIVITY_MIN:
            return m // 16, m // 16 + 1000
        m -= 1600
    return None


def overlap_span(intervals, ia, ib):
    los = [iv for iv in intervals[ia:ib + 1] if len(iv["active_speakers"]) > 1]
    if not los:
        return None
    return (min(iv["start_sample"] for iv in los), max(iv["end_sample"] for iv in los))


def gen_candidates(sess):
    ivs = sess["intervals"]
    by_id = {t["transition_id"]: t for t in sess["transitions"]}
    out = []
    for t in sess["transitions"]:
        topo, mask = t["primary_topology"], t["mask_state"]
        ia, ib = t.get("from_interval_index"), t.get("to_interval_index")
        if topo in C1 or topo in C2:
            if mask != "valid" or ia is None or ib is None:
                continue
            a = t["to_speaker"] if topo in C1 else t["from_speaker"]
            if not a:
                continue
            trans = ivs[ib]["start_sample"]
            stratum = "C1" if topo in C1 else "C2"
            out.append(mk(sess, stratum, None, a, trans, None))
        elif topo in C3 or topo in C4:
            if mask != "valid" or ia is None or ib is None:
                continue
            ov = overlap_span(ivs, min(ia, ib), max(ia, ib))
            if not ov:
                continue
            a = t["from_speaker"]
            if not a:
                continue
            ov_a = activity(sess, a, ov[0], ov[1])
            out.append(mk(sess, "C3" if topo in C3 else "C4", topo, a, ov[1], {
                "overlap_samples": ov[1] - ov[0],
                "overlap_a_active_samples": int(round(ov_a * (ov[1] - ov[0]))),
                "has_overlap": True}))
        elif topo in C6 or mask != "valid":
            if topo in C1 or topo in C2 or topo in C3 or topo in C4:
                continue
            if topo == "short_backchannel_return" or topo == "initial_start":
                continue
            a = t.get("from_speaker") or t.get("to_speaker")
            if not a:
                continue
            if ib is not None:
                trans = ivs[ib]["start_sample"]
            elif ia is not None:
                trans = ivs[ia]["end_sample"]
            else:
                continue
            # overlap at/near boundary?
            w0, w1 = max(0, trans - 2 * SR), trans + 2 * SR
            has_ov = any(len(iv["active_speakers"]) > 1 for iv in iter_ov(sess, w0, w1))
            out.append(mk(sess, "C6", topo, a, trans, {"has_overlap": has_ov}))
    for e in sess["episodes"]:
        if e["primary_topology"] != "short_backchannel_return" or not e["coverage_gate_eligible"]:
            continue
        tids = e.get("transition_ids", [])
        if len(tids) != 2 or not all(tid in by_id for tid in tids):
            continue
        s, en = e["start_sample"], e["end_sample"]
        pre = single_before(sess, s - 1) if s > 0 else None
        post = single_after(sess, en)
        pre_sp = (pre["active_speakers"] if pre and len(pre["active_speakers"]) == 1 else [])
        post_sp = (post["active_speakers"] if post and len(post["active_speakers"]) == 1 else [])
        if not pre_sp or pre_sp != post_sp:
            continue
        entry = by_id[tids[0]]
        entry_tags = [t for t in (e.get("secondary_tags", []) + entry.get("secondary_tags", []))]
        has_ov_entry = any("overlap" in str(x) for x in (entry_tags + [entry.get("primary_topology")]))
        out.append(mk(sess, "C5", e["primary_topology"], pre_sp[0], s,
                      {"has_overlap": has_ov_entry, "episode_ref": e["episode_id"]}))
    return out


def mk(sess, stratum, v2_topo, anchor, trans_sample, extra):
    c = {"sess": sess, "stratum": stratum, "v2_topo": v2_topo, "anchor": anchor,
         "trans_sample": trans_sample, "extra": extra or {}}
    return c


def finalize(c):
    sess, anchor, ts = c["sess"], c["anchor"], c["trans_sample"]
    trans_ms = ts // MS
    nat = native_span(sess, anchor, ts)
    if nat is None:
        return None
    cau = causal_span(sess, anchor, ts)
    ev_s = max(0, trans_ms - EVAL_PRE_MS)
    ev_e = trans_ms + EVAL_POST_MS
    if ev_e * MS > sess["duration"] or ev_e <= ev_s:
        return None
    row_extra = {
        "native_reference_start_ms": nat[0], "native_reference_end_ms": nat[1],
        "causal_reference_start_ms": cau[0] if cau else None,
        "causal_reference_end_ms": cau[1] if cau else None,
        "causal_bindable": cau is not None,
        "authoritative_transition_time_ms": trans_ms,
        "evaluation_start_ms": ev_s, "evaluation_end_ms": ev_e,
    }
    if cau and cau[1] > trans_ms:
        return None
    c.update(row_extra)
    return c


TOPO_OF = {"C1": "A", "C2": "A", "C3": "A->A+B->A", "C4": "A->A+B->B"}


def topo_label(c):
    s = c["stratum"]
    if s in ("C1", "C2", "C4"):
        return TOPO_OF[s]
    if s == "C3":
        return "overlap_return" if c["v2_topo"] == "overlap_return" else "A->A+B->A"
    if s == "C5":
        return "A+A+B" if c["extra"].get("has_overlap") else "overlap_return"
    return "A+B"  # C6


def main():
    sessions = load_gt()
    v2fr = json.loads(V2_FREEZE.read_text(encoding="utf-8"))
    by = {}
    for key in sorted(sessions):
        sess = sessions[key]
        for c in gen_candidates(sess):
            c["key"] = key
            by.setdefault((key, c["stratum"]), []).append(c)
    for (key, stratum), v in by.items():
        if stratum in ("C3", "C4"):
            v.sort(key=lambda c: (-c["extra"].get("overlap_a_active_samples", 0), c["trans_sample"]))
        else:
            v.sort(key=lambda c: c["trans_sample"])

    cache = {}
    n_finalized = [0]

    def ensure(c):
        k = id(c)
        if k not in cache:
            n_finalized[0] += 1
            cache[k] = finalize(c)
        return cache[k]

    def fits(key, f, extra=()):
        for _, x in list(picked) + [("x", g) for g in extra]:
            if x["key"] == key and not (f["evaluation_end_ms"] <= x["evaluation_start_ms"]
                                        or x["evaluation_end_ms"] <= f["evaluation_start_ms"]):
                return False
        return True

    used = set()
    picked = []  # (split, finalized candidate)

    def take(pool_keys, stratum, corpus, n, cap, split):
        got = []
        for key in pool_keys:
            if len(got) >= n:
                break
            if key[0] != corpus:
                continue
            for c in by.get((key, stratum), []):
                if len(got) >= n:
                    break
                if id(c) in used:
                    continue
                if sum(1 for s, x in picked if s == split and x["key"] == key) >= cap:
                    break
                f = ensure(c)
                if f is None or not fits(key, f, got):
                    continue
                used.add(id(c))
                got.append(f)
        if len(got) < n:
            raise FailClosed(f"{split}/{stratum}/{corpus}: {len(got)}/{n}")
        picked.extend((split, f) for f in got)
        return got

    keys = sorted(sessions)
    # CAL pools: first 2 AMI + first 2 Ali sessions with C1/C3/C4/C5/C6 coverage
    cal_groups = [("A", "C1", False), ("A+B", "C6", True), ("A+A+B", "C5", True),
                  ("A->A+B->A", "C3", False), ("A->A+B->B", "C4", False),
                  ("overlap_return", "C3", True)]
    cal_pools = {"ami": [k for k in keys if k[0] == "AMI"][:4],
                 "alimeeting": [k for k in keys if k[0] == "AliMeeting"][:4]}
    cal_main_ext = []
    for grp, stratum, want_ov in cal_groups:
        for corpus in ("ami", "alimeeting"):
            pool = cal_pools[corpus]
            done = False
            for key in pool:
                for c in by.get((key, stratum), []):
                    if id(c) in used:
                        continue
                    if stratum == "C3" and want_ov != (c["v2_topo"] == "overlap_return"):
                        continue
                    if stratum == "C5" and want_ov != bool(c["extra"].get("has_overlap")):
                        continue
                    if stratum == "C6" and not c["extra"].get("has_overlap"):
                        continue
                    if sum(1 for s, x in picked if s == "CAL12" and x["key"] == key) >= 4:
                        continue
                    f = ensure(c)
                    if f is None or not fits(key, f):
                        continue
                    used.add(id(c))
                    f["cal_group"] = grp
                    picked.append(("CAL12", f))
                    done = True
                    break
                if done:
                    break
            if not done:
                raise FailClosed(f"CAL12/{grp}/{corpus}: no candidate")
    cal_keys = {x["key"] for s, x in picked if s == "CAL12"}
    rest_ami = [k for k in keys if k[0] == "AMI" and k not in cal_keys]
    rest_ali = [k for k in keys if k[0] == "AliMeeting" and k not in cal_keys]
    main_pools = {"ami": rest_ami[:6], "alimeeting": rest_ali[:6]}
    for stratum in ("C1", "C2", "C3", "C4", "C5", "C6"):
        take(main_pools["ami"], stratum, "AMI", 4, 6, "MAIN48")
        take(main_pools["alimeeting"], stratum, "AliMeeting", 4, 6, "MAIN48")
    main_keys = {x["key"] for s, x in picked if s == "MAIN48"}
    ext_pools = {"ami": [k for k in rest_ami if k not in main_keys][:3],
                 "alimeeting": [k for k in rest_ali if k not in main_keys][:3]}
    for stratum in ("C1", "C2", "C3", "C4", "C5", "C6"):
        take(ext_pools["ami"], stratum, "AMI", 2, 6, "EXT24")
        take(ext_pools["alimeeting"], stratum, "AliMeeting", 2, 6, "EXT24")

    main = [(s, c) for s, c in picked if s == "MAIN48"]
    tier1 = [c for _, c in main if c["stratum"] in ("C3", "C4")
             and c["extra"].get("overlap_a_active_samples", 0) >= int(0.3 * SR)]
    tier2 = [c for _, c in main if c["stratum"] == "C5" and c["extra"].get("has_overlap")]
    tier3 = [c for _, c in main if c["stratum"] == "C6" and c["extra"].get("has_overlap")]
    for t in (tier1, tier2, tier3):
        t.sort(key=lambda c: (c["key"], c["trans_sample"]))
    onto_pool, seen = [], set()
    for t in (tier1, tier2, tier3):
        for c in t:
            if id(c) not in seen:
                seen.add(id(c))
                onto_pool.append(c)
    if len(onto_pool) < 16:
        raise FailClosed(f"ONTOLOGY pool {len(onto_pool)}/16")
    onto = set(id(c) for c in onto_pool[:16])
    ctrl = {(c["key"], c["trans_sample"]) for _, c in main if c["stratum"] in ("C1", "C2", "C5")}
    if len(ctrl) != 24:
        raise FailClosed(f"CONTROL pool {len(ctrl)}/24")

    # emit rows
    counters = {}
    rows = []
    for split, c in picked:
        key = c["key"]
        counters[key] = counters.get(key, 0) + 1
        sess = c["sess"]
        topo = c.get("cal_group") if split == "CAL12" else topo_label(c)
        rows.append({
            "schema_version": SCHEMA,
            "episode_id": f"{sess['session_id']}:A{counters[key]:05d}",
            "corpus": sess["corpus"],
            "session_id": sess["session_id"],
            "topology": topo,
            "split": split,
            "evaluation_start_ms": c["evaluation_start_ms"],
            "evaluation_end_ms": c["evaluation_end_ms"],
            "anchor_speaker": c["anchor"],
            "native_reference_start_ms": c["native_reference_start_ms"],
            "native_reference_end_ms": c["native_reference_end_ms"],
            "causal_reference_start_ms": c["causal_reference_start_ms"],
            "causal_reference_end_ms": c["causal_reference_end_ms"],
            "causal_bindable": c["causal_bindable"],
            "authoritative_transition_time_ms": c["authoritative_transition_time_ms"],
            "ontology_subset": split == "MAIN48" and id(c) in onto,
            "control_subset": split == "MAIN48" and (c["key"], c["trans_sample"]) in ctrl
                            and c["stratum"] in ("C1", "C2", "C5"),
        })
    rows.sort(key=lambda r: r["episode_id"])

    # fail-closed asserts
    assert len(rows) == 84, len(rows)
    assert sum(1 for r in rows if r["split"] == "CAL12") == 12
    assert sum(1 for r in rows if r["split"] == "MAIN48") == 48
    assert sum(1 for r in rows if r["split"] == "EXT24") == 24
    assert sum(1 for r in rows if r["ontology_subset"]) == 16
    assert sum(1 for r in rows if r["control_subset"]) == 24
    assert all(not r["ontology_subset"] and not r["control_subset"] or r["split"] == "MAIN48"
               for r in rows)
    assert len({r["episode_id"] for r in rows}) == 84
    cal = {(r["corpus"], r["session_id"]) for r in rows if r["split"] == "CAL12"}
    mai = {(r["corpus"], r["session_id"]) for r in rows if r["split"] == "MAIN48"}
    ext = {(r["corpus"], r["session_id"]) for r in rows if r["split"] == "EXT24"}
    assert not (cal & mai) and not (ext & (cal | mai)), (cal & mai, ext & (cal | mai))
    for r in rows:
        assert r["native_reference_end_ms"] - r["native_reference_start_ms"] == 5000
        if r["causal_bindable"]:
            assert r["causal_reference_end_ms"] - r["causal_reference_start_ms"] == 1000
            assert r["causal_reference_end_ms"] <= r["authoritative_transition_time_ms"]
        else:
            assert r["causal_reference_start_ms"] is None and r["causal_reference_end_ms"] is None
        assert r["evaluation_end_ms"] > r["evaluation_start_ms"]
    bysess = {}
    for r in rows:
        bysess.setdefault((r["corpus"], r["session_id"]), []).append(r)
    for k, v in bysess.items():
        v.sort(key=lambda r: r["evaluation_start_ms"])
        for a, b in zip(v, v[1:]):
            assert a["evaluation_end_ms"] <= b["evaluation_start_ms"], k

    raw = "".join(json.dumps(r, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True) + "\n" for r in rows)
    MANIFEST.write_text(raw, encoding="utf-8", newline="\n")
    file_sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    counts = {"CAL12": 12, "MAIN48": 48, "EXT24": 24, "ONTOLOGY16": 16,
              "CONTROL24": 24, "total": 84}
    freeze_payload = {"rows": rows, "counts": counts}
    freeze_sha = hashlib.sha256(json.dumps(
        freeze_payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True).encode("utf-8")).hexdigest()
    v2_file_sha = hashlib.sha256(V2_FREEZE.read_bytes()).hexdigest()
    FREEZE.write_text(json.dumps({
        "artifact_role": "psem_small_probe_manifest_freeze",
        "schema_version": SCHEMA,
        "counts": counts,
        "file_sha256": file_sha,
        "freeze_sha256": freeze_sha,
        "v2_dataset_freeze_id": v2fr.get("dataset_freeze_id"),
        "v2_freeze_file_sha256": v2_file_sha,
        "v2_freeze_payload_sha256": v2fr.get("freeze_payload_sha256"),
        "v2_freeze_core_payload_sha256": v2fr.get("freeze_core_payload_sha256"),
        "v2_freeze": {k: v for k, v in v2fr.items() if "sha256" in k.lower()},
        "gt_interval_sources": sorted(p.relative_to(REPO).as_posix() for p in OCC.values()),
        "session_sets": {s: sorted(f"{c}/{i}" for c, i in xs)
                         for s, xs in (("CAL12", cal), ("MAIN48", mai), ("EXT24", ext))},
    }, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(MANIFEST), "rows": len(rows), "counts": counts,
                      "file_sha256": file_sha, "freeze_sha256": freeze_sha,
                      "sessions": {s: len(xs) for s, xs in
                                   (("CAL12", cal), ("MAIN48", mai), ("EXT24", ext))}},
                     indent=1))


if __name__ == "__main__":
    main()
