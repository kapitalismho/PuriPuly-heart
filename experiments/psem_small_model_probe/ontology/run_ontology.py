#!/usr/bin/env python3
"""Gate 6 — ONTOLOGY16 downstream challenge (issue #117).

For each of the 16 ontology episodes (audio/GT only, independent of model
scores) build TWO boundary renderings over the SAME evaluation-window audio:

  K (Simple Anchor) : old segment = [win_start, T_trans) attributed to the
                      anchor A (keep current segment while A active);
                      new segment = [T_trans, win_end) attributed to NEXT
                      (first single speaker at/after T_trans, else NONE).
  T (takeover)      : cut at candidate takeover frontier F (first ms in
                      [T_trans-2000, T_trans] with non-anchor activity =
                      overlap onset, GT-derived; backdated source boundary);
                      old = [win_start, F) -> A, new = [F, win_end) -> B
                      (dominant non-anchor speaker at F). F == T_trans => tie.

Both renderings pass through the SAME deterministic scoring pass (substitution
for the ASR+translation pipeline, see README header) under blind X/Y masking
(seeded shuffle, reveal only at decision). Evaluated ONLY:
  (a) cross-speaker lexical contamination proxy: non-attributed-speaker
      active ms inside the old segment;
  (b) boundary word loss/duplication proxy: utterance-straddle flag at the
      cut (speech of any speaker active on both sides within +/-200 ms) +
      orphaned/duplicated ms (0 by partition construction, reported);
  (c) coherence/attribution proxy: new-segment attribution precision =
      fraction of speech-active ms where the attributed speaker is active.

Predeclared T-better(ep) <=> (contamK - contamT >= 50 ms)
  AND (lossriskT <= lossriskK) AND (precT > precK).
Decision rule: n>=4 -> reopen ownership; n<=1 -> retain Simple Anchor;
  2<=n<=3 -> diagnostic/HOLD-only, no new primitive.

Writes results/k_render.jsonl, results/t_render.jsonl (gitignored),
results/pairwise.md (blind X/Y notes), results/decision.md (verdict).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from experiments.psem_small_model_probe.cal.audio_resolve import (
    load_span,
    resolve_audio,
    sha256_pcm,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MANIFEST = REPO / "experiments/psem_small_model_probe/manifest/manifest.jsonl"
OCC = {
    "dev": REPO / "experiments/psem_relative_occupancy_gate/results/dev/relative_occupancy_manifest.jsonl",
    "eval": REPO / "experiments/psem_relative_occupancy_gate/results/eval/relative_occupancy_manifest.jsonl",
}
RESULTS = HERE / "results"
MS = 16  # samples per ms at 16 kHz
SEED = 117
CONTAM_MIN_MS = 50
STRADDLE_MS = 200


def load_rows() -> list[dict]:
    rows = [json.loads(l) for l in MANIFEST.read_text(encoding="utf-8").splitlines() if l.strip()]
    onto = [r for r in rows if r.get("ontology_subset")]
    assert len(onto) == 16, f"ontology rows {len(onto)}/16"
    return sorted(onto, key=lambda r: r["episode_id"])


def load_intervals() -> dict[tuple[str, str], list[dict]]:
    out: dict[tuple[str, str], list[dict]] = {}
    for path in OCC.values():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            corpus = {"AMI": "ami", "AliMeeting": "alimeeting"}[r["corpus"]]
            out[(corpus, r["session_id"])] = sorted(r["intervals"], key=lambda iv: iv["start_sample"])
    return out


def active_at(ivs: list[dict], sample: int) -> frozenset[str]:
    return frozenset(s for iv in ivs if iv["start_sample"] <= sample < iv["end_sample"] for s in iv["active_speakers"])


def active_ms(ivs: list[dict], ms: int) -> frozenset[str]:
    return active_at(ivs, ms * MS)


def stratum_of(row: dict) -> str:
    return "C4" if row["topology"] == "A->A+B->B" else "C3"


def frontier(ivs: list[dict], anchor: str, trans_ms: int, win_s: int) -> tuple[int, str]:
    """(F, B): first ms in [trans-2000, trans] with non-anchor activity; B = dominant other."""
    lo = max(win_s, trans_ms - 2000)
    f = trans_ms
    for ms in range(lo, trans_ms):
        if active_ms(ivs, ms) - {anchor}:
            f = ms
            break
    others: dict[str, int] = {}
    for ms in range(f, min(f + 1000, trans_ms + 3000)):
        for s in active_ms(ivs, ms) - {anchor}:
            others[s] = others.get(s, 0) + 1
    b = max(others, key=lambda s: others[s]) if others else ""
    return f, b


def next_speaker(ivs: list[dict], trans_ms: int) -> str:
    for iv in ivs:
        if iv["end_sample"] <= trans_ms * MS:
            continue
        sp = iv["active_speakers"]
        if len(sp) == 1 and iv["start_sample"] >= trans_ms * MS:
            return sp[0]
    return "NONE"


def contam_old(ivs: list[dict], s_ms: int, e_ms: int, attr: str) -> int:
    return sum(1 for ms in range(s_ms, e_ms) if active_ms(ivs, ms) - {attr})


def straddle_risk(ivs: list[dict], cut_ms: int, win_s: int, win_e: int) -> int:
    b = max(win_s, cut_ms - STRADDLE_MS)
    a = min(win_e, cut_ms + STRADDLE_MS)
    if b >= cut_ms or a <= cut_ms:
        return 0
    before = set().union(*(active_ms(ivs, ms) for ms in range(b, cut_ms))) if b < cut_ms else set()
    after = set().union(*(active_ms(ivs, ms) for ms in range(cut_ms, a))) if cut_ms < a else set()
    return int(bool(before & after))


def attribution_precision(ivs: list[dict], s_ms: int, e_ms: int, attr: str) -> float:
    num = den = 0
    for ms in range(s_ms, e_ms):
        act = active_ms(ivs, ms)
        if not act:
            continue
        den += 1
        if attr in act:
            num += 1
    return num / den if den else 1.0


def score(ivs: list[dict], old: tuple[int, int], new: tuple[int, int],
          attr_old: str, attr_new: str, cut: int, win_s: int, win_e: int) -> dict:
    return {
        "contam_old_ms": contam_old(ivs, *old, attr_old),
        "loss_risk": straddle_risk(ivs, cut, win_s, win_e),
        "attr_prec_new": round(attribution_precision(ivs, *new, attr_new), 4),
        "orphaned_ms": 0, "duplicated_ms": 0,  # partition covers window exactly
    }


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    rows = load_rows()
    ivs_by_sess = load_intervals()
    rng = random.Random(SEED)
    k_rows, t_rows, pairs = [], [], []
    for row in rows:
        ep, anchor = row["episode_id"], row["anchor_speaker"]
        ws, we, tt = row["evaluation_start_ms"], row["evaluation_end_ms"], row["authoritative_transition_time_ms"]
        ivs = ivs_by_sess[(row["corpus"], row["session_id"])]
        pcm = load_span(resolve_audio(row), ws, we)
        assert len(pcm) == (we - ws) * MS * 2, f"{ep}: span {len(pcm)}"
        audio = {"sha256": sha256_pcm(pcm), "frames": len(pcm) // 2}
        f, b = frontier(ivs, anchor, tt, ws)
        nxt = next_speaker(ivs, tt)
        k = {"old": (ws, tt), "new": (tt, we), "attr_old": anchor,
             "attr_new": nxt if nxt != "NONE" else anchor, "attr_new_none": nxt == "NONE", "cut": tt}
        t = {"old": (ws, f), "new": (f, we), "attr_old": anchor,
             "attr_new": b if b else k["attr_new"], "cut": f}
        km = score(ivs, k["old"], k["new"], anchor, k["attr_new"], tt, ws, we)
        tm = score(ivs, t["old"], t["new"], anchor, t["attr_new"], f, ws, we)
        if k["attr_new_none"]:
            k["attr_new"] = "NONE"
            km["attr_prec_new"] = round(attribution_precision(ivs, tt, we, "\x00"), 4)
        base = {"episode_id": ep, "stratum": stratum_of(row), "topology": row["topology"],
                "anchor_speaker": anchor, "trans_ms": tt, "frontier_ms": f,
                "takeover_speaker": b or nxt, "audio": audio}
        k_rows.append({**base, "rendering": "K", "old_ms": list(k["old"]), "new_ms": list(k["new"]),
                       "attr_old": anchor, "attr_new": k["attr_new"], "metrics": km})
        t_rows.append({**base, "rendering": "T", "old_ms": list(t["old"]), "new_ms": list(t["new"]),
                       "attr_old": anchor, "attr_new": t["attr_new"], "metrics": tm})
        order = ["K", "T"]
        rng.shuffle(order)
        blind = {tag: ("X" if i == 0 else "Y") for i, tag in enumerate(order)}
        m = {"K": km, "T": tm}
        # direction-agnostic blind comparator: P better than Q on (a)+(c) with (b) guard
        pq = {blind["K"]: "K", blind["T"]: "T"}
        mx, my = m[pq["X"]], m[pq["Y"]]
        x_better = (my["contam_old_ms"] - mx["contam_old_ms"] >= CONTAM_MIN_MS
                    and mx["loss_risk"] <= my["loss_risk"] and mx["attr_prec_new"] > my["attr_prec_new"])
        y_better = (mx["contam_old_ms"] - my["contam_old_ms"] >= CONTAM_MIN_MS
                    and my["loss_risk"] <= mx["loss_risk"] and my["attr_prec_new"] > mx["attr_prec_new"])
        blind_verdict = "X better" if x_better and not y_better else ("Y better" if y_better and not x_better else "tie")
        t_better = bool((tm["contam_old_ms"] - km["contam_old_ms"] <= -CONTAM_MIN_MS)
                        and tm["loss_risk"] <= km["loss_risk"] and tm["attr_prec_new"] > km["attr_prec_new"])
        pairs.append({"base": base, "blind": blind, "blind_verdict": blind_verdict, "t_better": t_better,
                      "loss_up": int(tm["loss_risk"] > km["loss_risk"]),
                      "both_poor": int(km["contam_old_ms"] >= CONTAM_MIN_MS and tm["attr_prec_new"] <= km["attr_prec_new"])})
        k_rows[-1]["blind_id"] = blind["K"]
        t_rows[-1]["blind_id"] = blind["T"]

    (RESULTS / "k_render.jsonl").write_text("\n".join(json.dumps(r) for r in k_rows) + "\n", encoding="utf-8")
    (RESULTS / "t_render.jsonl").write_text("\n".join(json.dumps(r) for r in t_rows) + "\n", encoding="utf-8")

    L = ["# ONTOLOGY16 K-vs-T pairwise notes (blind X/Y; seed 117)",
         "", "Metric key: contam_old_ms (a, lower better) | loss_risk (b, lower better) |",
         "attr_prec_new (c, higher better). Tie => no candidate clearly better.", ""]
    for p, kr, tr in zip(pairs, k_rows, t_rows):
        b = p["base"]
        m = {"X": kr["metrics"] if p["blind"]["K"] == "X" else tr["metrics"],
             "Y": tr["metrics"] if p["blind"]["K"] == "X" else kr["metrics"]}
        L += [f"## {b['episode_id']} ({b['stratum']}/{b['topology']}, anchor {b['anchor_speaker']})",
              f"- frontier F={b['frontier_ms']} vs T_trans={b['trans_ms']} (delta {b['trans_ms'] - b['frontier_ms']} ms); takeover/NEXT spk {b['takeover_speaker']}",
              f"- X: contam={m['X']['contam_old_ms']} loss={m['X']['loss_risk']} prec={m['X']['attr_prec_new']}",
              f"- Y: contam={m['Y']['contam_old_ms']} loss={m['Y']['loss_risk']} prec={m['Y']['attr_prec_new']}",
              f"- blind verdict: **{p['blind_verdict']}**", ""]
    (RESULTS / "pairwise.md").write_text("\n".join(L), encoding="utf-8")

    n = sum(p["t_better"] for p in pairs)
    n_loss = sum(p["loss_up"] for p in pairs)
    n_poor = sum(p["both_poor"] for p in pairs)
    winners = [p["base"]["episode_id"] for p in pairs if p["t_better"]]
    verdict = ("REOPEN ownership" if n >= 4 else ("RETAIN Simple Anchor" if n <= 1 else "diagnostic HOLD-only"))
    D = ["# ONTOLOGY16 decision", "", f"T-better count: **{n}/16** ({', '.join(winners) or 'none'})",
         f"Word-loss increase episodes (loss_risk_T > loss_risk_K): **{n_loss}/16**",
         f"Both-poor episodes (K contam + T no precision gain): **{n_poor}/16**",
         f"Predeclared rule (>=4 reopen; <=1 retain; 2-3 HOLD): **{verdict}**", "",
         "Mapping (reveal): " + "; ".join(f"{p['base']['episode_id']}: K={p['blind']['K']}, T={p['blind']['T']}" for p in pairs),
         ""]
    if n_poor >= 4:
        D += ["Note: KEEP and CUT both repeatedly poor (high kept contamination with no",
              "takeover-side precision gain) -> extraction / conditioned-ASR is a",
              "separate-issue candidate (NOT implemented here).", ""]
    (RESULTS / "decision.md").write_text("\n".join(D), encoding="utf-8")
    print(f"T-better {n}/16 | loss-up {n_loss}/16 | both-poor {n_poor}/16 -> {verdict}")


if __name__ == "__main__":
    main()
