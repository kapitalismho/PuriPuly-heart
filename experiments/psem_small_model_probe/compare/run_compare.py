#!/usr/bin/env python3
"""F0 (frozen Sortformer) scored on MAIN48 with the main/ rules (issue #117).

Head-to-head: frozen Sortformer posteriors vs FireRed/NeoVAD on the SAME 48
MAIN48 episodes. No new Sortformer inference — cached posteriors only.

Mapping: each MAIN48 episode -> Sortformer posterior cells by
(corpus, session_id) + source-sample time math (16 kHz, half-open
[start_sample, end_sample)). Episodes/sessions with no F0 coverage are
reported as missing, never inferred.

Scoring reuses main/ exactly (read-only imports, no copies):
  GT speech gate (cal.run_cal.gt_anchor_speech / gt_window_stats),
  500 ms confirmation via cal.metrics.replay_decisions
  (CommonPersistenceDecoder 500/300), headline metrics via
  cal.metrics.score_episode / aggregate at a single predeclared tau.

Anchor-slot assumption (explicit): F0 is 4-slot diarization with no
production anchor binding. Per episode the anchor slot is chosen by the
oracle_anchor_mapping pattern from model_decode.py — duration-weighted mean
of (probabilities * alive) over trace-valid, unmasked, GT anchor-active
support cells — restricted here to the episode evaluation window
(F0 episodes use emit-to-end support; MAIN48 has no emit spans, so the eval
window is the support). support_scores/weight are recorded per episode.
This is oracle (uses GT), i.e. it favours F0; the small-model binders use
audio enrollment instead.

Operating point: TAU_F0 = 0.5 — the predeclared F0 anchor threshold
(config.json current_anchor_threshold 0.5; probe grid 0.35/0.5/0.65).
Single point, no sweep, no tuning. main/ taus are untouched; small-model
numbers are reused from main/results/summary.md, never rescored.

Writes (owned namespace only): results/f0_main.jsonl, results/summary.md.
"""

from __future__ import annotations

import argparse
import json
from bisect import bisect_right
from pathlib import Path

import numpy as np

from experiments.psem_small_model_probe.cal import run_cal
from experiments.psem_small_model_probe.cal.metrics import aggregate, score_episode

COMPARE_DIR = Path(__file__).resolve().parent
MANIFEST = COMPARE_DIR.parent / "manifest" / "manifest.jsonl"
FROZEN_NPZ = Path("experiments/psem_frozen_ceiling_gate/frozen_inputs/posterior_sessions.npz")
RESULTS = COMPARE_DIR / "results"

SAMPLES_PER_MS = 16
FRAME_MS = 10
FRAME_SAMPLES = FRAME_MS * SAMPLES_PER_MS  # 160
TAU_F0 = 0.5  # predeclared, not tuned (see module docstring)

# main/ headline rows reused verbatim from main/results/summary.md (frozen taus).
MAIN_ROWS = {
    ("firered", "O"): {"contam": 877.7, "false": 12, "n_keep": 32,
                       "missed": 5, "n_cut": 8, "src_p50": -820.0,
                       "src_p90": 1068.0, "dec_p50": 500.0, "dec_p90": 500.0},
    ("neovad", "O"): {"contam": 877.7, "false": 0, "n_keep": 32,
                      "missed": 8, "n_cut": 8, "src_p50": None,
                      "src_p90": None, "dec_p50": None, "dec_p90": None},
    ("firered", "C"): {"contam": 877.7, "false": 13, "n_keep": 32,
                       "missed": 5, "n_cut": 8, "src_p50": -670.0,
                       "src_p90": 1098.0, "dec_p50": 500.0, "dec_p90": 500.0},
    ("neovad", "C"): {"contam": 877.7, "false": 0, "n_keep": 32,
                      "missed": 8, "n_cut": 8, "src_p50": None,
                      "src_p90": None, "dec_p50": None, "dec_p90": None},
}


def load_main48() -> list[dict]:
    return [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines()
            if line.strip() and json.loads(line).get("split") == "MAIN48"]


def load_sessions() -> dict[str, dict]:
    """source_id (e.g. ami_ES2002b) -> cached posterior arrays."""
    data = np.load(str(FROZEN_NPZ))
    indices = sorted({int(name.split("_")[0][1:]) for name in data.files})
    sessions: dict[str, dict] = {}
    for i in indices:
        p = f"s{i:03d}_"
        ids = [str(x) for x in data[p + "episode_ids"]]
        source = ids[0].split(":")[0] if ids else None
        if source is None:
            continue
        sessions[source] = {
            "starts": np.asarray(data[p + "starts"]),
            "ends": np.asarray(data[p + "ends"]),
            "probabilities": np.asarray(data[p + "probabilities"]),
            "alive": np.asarray(data[p + "alive"]),
            "valid": np.asarray(data[p + "valid"]),
            "masked": np.asarray(data[p + "masked"]),
        }
    return sessions


def oracle_slot(sess: dict, gt: dict, anchor: str, ws: int, we: int) -> dict | None:
    """Duration-weighted oracle slot pick over eval-window anchor-active cells."""
    starts, ends = sess["starts"], sess["ends"]
    lo = int(np.searchsorted(ends, ws + 1, side="left"))
    hi = int(np.searchsorted(starts, we, side="left"))
    weighted = np.zeros(sess["probabilities"].shape[1])
    denom = 0
    for idx in range(lo, hi):
        cs, ce = int(starts[idx]), int(ends[idx])
        if ce <= ws or cs >= we or not bool(sess["valid"][idx]) or bool(sess["masked"][idx]):
            continue
        center = (max(cs, ws) + min(ce, we)) // 2
        if not run_cal.gt_anchor_speech(gt, anchor, center):
            continue
        w = min(ce, we) - max(cs, ws)
        denom += w
        weighted += np.asarray(sess["probabilities"][idx]) * np.asarray(sess["alive"][idx]) * w
    if denom <= 0:
        return None
    return {"slot": int(np.argmax(weighted / denom)),
            "support_scores": [float(v) for v in weighted / denom],
            "support_weight_samples": denom, "cell_lo": lo, "cell_hi": hi}


def score_f0(rows: list[dict], gt_index: dict, sessions: dict) -> tuple[list[dict], list[dict], list[str]]:
    records, episodes, missing = [], [], []
    for row in rows:
        episode_id = row["episode_id"]
        key = (str(row["corpus"]).lower(), row["session_id"])
        gt = gt_index.get(key)
        source = f"{str(row['corpus']).lower()}_{row['session_id']}"
        sess = sessions.get(source)
        eval_start, eval_end = int(row["evaluation_start_ms"]), int(row["evaluation_end_ms"])
        ws, we = eval_start * SAMPLES_PER_MS, eval_end * SAMPLES_PER_MS
        n_frames = (eval_end - eval_start) // FRAME_MS
        status = "BOUND"
        reason = ""
        mapping = None
        if gt is None or sess is None:
            status = "MISSING"
            reason = f"no {'GT' if gt is None else 'F0 coverage'} for {key}"
        else:
            mapping = oracle_slot(sess, gt, row["anchor_speaker"], ws, we)
            if mapping is None:
                status = "MISSING"
                reason = "no valid anchor-active support cells in eval window"
            # Partial trace gaps (~200 ms dropped cells in cached posteriors)
            # are NOT episode-missing: gap frames fall through to UNCERTAIN
            # (no model evidence -> HOLD + persistence reset) in the frame
            # loop below. Only a wholly unmappable episode is MISSING.
        frames: list[dict] = []
        if status == "BOUND":
            assert mapping is not None
            slot = mapping["slot"]
            starts, ends = sess["starts"], sess["ends"]  # type: ignore[union-attr]
            for i in range(n_frames):
                fs = ws + i * FRAME_SAMPLES
                j = int(np.searchsorted(ends, fs + 1, side="left"))
                ok = (j < len(starts) and int(starts[j]) <= fs < int(ends[j])
                      and bool(sess["valid"][j]) and bool(sess["alive"][j][slot]))  # type: ignore[union-attr]
                center = ws + i * FRAME_SAMPLES + FRAME_SAMPLES // 2
                frames.append({
                    "speech_gt": run_cal.gt_anchor_speech(gt, row["anchor_speaker"], center),
                    "anchor": float(sess["probabilities"][j][slot]) if ok else 0.0,  # type: ignore[union-attr]
                    # invalid/dead model cells carry no evidence -> UNCERTAIN
                    # (resets persistence, HOLD): mirrors relative_probabilities None.
                    "lifecycle": "BOUND" if ok else "UNCERTAIN",
                    "source_time_ms": eval_start + (i + 1) * FRAME_MS,
                })
        contam, active = run_cal.gt_window_stats(gt, row["anchor_speaker"], eval_start, eval_end) \
            if gt is not None else (0.0, 0.0)
        if status == "BOUND":
            records.append({
                "episode_id": episode_id, "topology": str(row["topology"]),
                "authoritative_transition_ms": int(row["authoritative_transition_time_ms"]),
                "frames": frames, "contam_s": contam, "active_speech_s": active,
                "lifecycle": "BOUND",
            })
            ep = score_episode(records[-1], TAU_F0, FRAME_MS)
            ep["anchor_slot"] = mapping["slot"]  # type: ignore[index]
            ep["support_scores"] = mapping["support_scores"]  # type: ignore[index]
            ep["support_weight_samples"] = mapping["support_weight_samples"]  # type: ignore[index]
            ep["contam_s"] = contam
            ep["covered_frac"] = (sum(1 for f in frames if f["lifecycle"] == "BOUND")
                                 / len(frames)) if frames else 0.0
            episodes.append(ep)
        else:
            missing.append(episode_id)
            episodes.append({"episode_id": episode_id, "topology": str(row["topology"]),
                             "status": status, "reason": reason})
    return records, episodes, missing


def promotion_check(f0: dict, name: str, cand: dict) -> tuple[str, str]:
    """#117 fallback rule (F0/G exact-gap unavailable on this subset).

    PASS iff contamination AND misses both strictly improve over F0 with
    >=20% relative on at least one, false cuts <= F0 x 1.10, and
    dec_p90 <= F0 dec_p90 + 100 ms.
    """
    legs: list[str] = []

    def improves(c: float | None, f: float | None) -> tuple[bool, float | None]:
        if c is None or f is None or f <= 0 or not c < f:
            return False, None
        return True, (f - c) / f

    ok_c, rel_c = improves(cand["contam"], f0["contamination_s_per_speech_h"])
    ok_m, rel_m = improves(cand["missed"], f0["missed"])
    legs.append(f"contam {cand['contam']:.1f} vs F0 {f0['contamination_s_per_speech_h']:.1f}: "
                + ("improves" if ok_c else "NO improvement"))
    legs.append(f"missed {cand['missed']}/{cand['n_cut']} vs F0 {f0['missed']}/{f0['n_cut']}: "
                + ("improves" if ok_m else "NO improvement"))
    rel_ok = (rel_c is not None and rel_c >= 0.20) or (rel_m is not None and rel_m >= 0.20)
    fmt = lambda v: f"{v:.1%}" if v is not None else "n/a"
    legs.append(f">=20% on one: {'yes' if rel_ok else 'NO'} "
                f"(contam {fmt(rel_c)}, missed {fmt(rel_m)})")
    limit = f0["false_cuts"] * 1.10
    ok_f = bool(cand["false"] <= limit)
    legs.append(f"false cuts {cand['false']}/{cand['n_keep']} vs F0 {f0['false_cuts']} "
                f"(limit {limit:.2f}): {'ok' if ok_f else 'EXCEEDS'}")
    f_p90, c_p90 = f0["delay_dec_p90"], cand["dec_p90"]
    if f_p90 is None:
        ok_p = False
        legs.append("p90: F0 has no CUT detections — no reference; cannot pass")
    elif c_p90 is None:
        ok_p = False
        legs.append(f"p90: {name} has no CUT detections (missed all) — cannot pass")
    else:
        ok_p = bool(c_p90 <= f_p90 + 100)
        legs.append(f"dec p90 {c_p90} vs F0 {f_p90}+100: {'ok' if ok_p else 'EXCEEDS'}")
    verdict = "PASS" if (ok_c and ok_m and rel_ok and ok_f and ok_p) else "FAIL"
    return verdict, "; ".join(legs)


def main() -> None:
    parser = argparse.ArgumentParser(description="F0-on-MAIN48 head-to-head scorer")
    parser.parse_args()
    rows = load_main48()
    gt_index = run_cal.load_gt_index()
    sessions = load_sessions()
    records, episodes, missing = score_f0(rows, gt_index, sessions)
    agg = aggregate(records, TAU_F0, FRAME_MS)
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "f0_main.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"type": "coverage", "n_main48": len(rows),
                             "n_scored": len(records), "n_missing": len(missing),
                             "missing": missing, "tau_f0": TAU_F0,
                             "frame_ms": FRAME_MS, "confirmation_ms": 500}) + "\n")
        for ep in episodes:
            fh.write(json.dumps({"type": "episode", **ep}) + "\n")
        fh.write(json.dumps({"type": "aggregate", **agg}) + "\n")

    from experiments.psem_small_model_probe.cal.metrics import KEEP_TOPOLOGIES, CUT_TOPOLOGIES
    keep = [e for e in episodes if e.get("status", "BOUND") == "BOUND"
            and e["topology"] in KEEP_TOPOLOGIES]
    cut = [e for e in episodes if e.get("status", "BOUND") == "BOUND"
           and e["topology"] in CUT_TOPOLOGIES]
    aba = [e for e in cut if e["topology"] == "A->A+B->B"]
    lines = [
        "# F0 on MAIN48 (frozen Sortformer, cached posteriors only)",
        "",
        f"coverage: {len(records)}/{len(rows)} scored, missing={missing or 'none'}",
        f"tau_f0={TAU_F0} (predeclared), frame {FRAME_MS}ms, confirmation 500ms, "
        "GT speech gate, oracle anchor-slot per episode",
        "",
        "| model | contam s/h | false cuts (KEEP-n) | missed (CUT-n) | "
        "src_err p50/p90 (ms) | dec p50/p90 (ms) |",
        "|---|---|---|---|---|---|",
        f"| F0-sortformer | {agg['contamination_s_per_speech_h']:.1f} | "
        f"{agg['false_cuts']}/{agg['n_keep']} | {agg['missed']}/{agg['n_cut']} | "
        f"{agg['delay_src_err_p50']}/{agg['delay_src_err_p90']} | "
        f"{agg['delay_dec_p50']}/{agg['delay_dec_p90']} |",
    ]
    for (m, r), v in MAIN_ROWS.items():
        lines.append(f"| {m} {r} (main/) | {v['contam']:.1f} | {v['false']}/{v['n_keep']} | "
                     f"{v['missed']}/{v['n_cut']} | {v['src_p50']}/{v['src_p90']} | "
                     f"{v['dec_p50']}/{v['dec_p90']} |")
    lines += ["",
              f"A->A+B->A KEEP: n/a (MAIN48 carries zero rows) | "
              f"A->A+B->B CUT: F0 {sum(1 for e in aba if not e['missed'])}/{len(aba)} "
              f"(missed {sum(1 for e in aba if e['missed'])})"]
    lines.append("F0 KEEP by topology (same KEEP shape as main/):")
    for topo in ("A", "overlap_return", "A+A+B"):
        sub = [e for e in keep if e["topology"] == topo]
        lines.append(f"- {topo}: {sum(1 for e in sub if not e['false_cut'])}/{len(sub)} "
                     f"(false {sum(1 for e in sub if e['false_cut'])})")
    cov = [e.get("covered_frac", 1.0) for e in episodes if e.get("status", "BOUND") == "BOUND"]
    lines.append(f"frame coverage: min {min(cov):.3f}, "
                 f"{sum(1 for c in cov if c < 1.0)}/{len(cov)} episodes with trace gaps "
                 f"(gap frames UNCERTAIN/HOLD)")
    lines += ["",
              "## #117 fallback promotion check vs F0 (contam+misses both improve, "
              "one >=20%, false <= F0x1.10, dec p90 <= F0+100ms)",
              ""]
    for (m, r), v in MAIN_ROWS.items():
        verdict, detail = promotion_check(agg, f"{m}-{r}", v)
        lines.append(f"- {m} {r}: {verdict} — {detail}")
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
