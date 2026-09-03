#!/usr/bin/env python3
"""MAIN48 re-score under the repaired V2 evaluator (evaluator_revision=2).

Raw-inference REUSE (no adapter inference, no audio reload):
  per-frame raw anchor outputs (episode_id/source_time_ms/anchor/lifecycle/
  model/regime + archived step_ms timings) reused verbatim from the frozen
  ``main/results/{model}_{regime}_main.jsonl`` archives (native weights,
  stub_fallback=false, verified per header). Old anchor-gated speech_gt
  discarded. Fresh inference NOT needed: archives are frame-complete
  (fail-closed grid/count/lifecycle checks per episode).

Freshly regenerated under frozen inputs:
  manifest hash verified + MAIN48==48 rows; GT index via frozen loader;
  frame speech_gt recomputed with the any-speech gate (+ anchor_speech_gt
  diagnostic); gt_eval attached; NEW frozen taus from
  ``cal/results_repaired/thresholds.json`` applied as-is (no MAIN48
  retuning); 500 ms primary, 300 ms sensitivity diagnostic only.

Writes ``main/results_repaired/`` (broken ``main/results/`` untouched):
  thresholds_frozen.json, {model}_{regime}_main.jsonl (repaired steps),
  {model}_{regime}_calibration.jsonl (per-tau + tau_frozen),
  summary.md (existing renderer + premature addendum), cpu.json
  (step timing recomputed from archived step_ms; bind/reset/RSS not
  re-timed under reuse), provenance.json.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.psem_small_model_probe.cal import audio_resolve  # noqa: E402
from experiments.psem_small_model_probe.cal.eval_semantics import (  # noqa: E402
    compact_gt,
    gt_anchor_speech,
    gt_any_speech,
    gt_window_stats,
)
from experiments.psem_small_model_probe.cal.metrics import (  # noqa: E402
    TAU_GRID,
    aggregate,
    replay_decisions,
    score_episode,
)
from experiments.psem_small_model_probe.main.run_main import (  # noqa: E402
    MODELS,
    REGIMES,
    gate3_supported,
    load_main_rows,
    render_summary,
    topology_view,
)
from experiments.psem_small_model_probe.cal.run_cal import (  # noqa: E402
    load_gt_index,
    verify_freeze,
)

MAIN_DIR = Path(__file__).resolve().parent
RESULTS = MAIN_DIR / "results"
CAL_REPAIRED = MAIN_DIR.parent / "cal" / "results_repaired"
OUT = MAIN_DIR / "results_repaired"


class FailClosed(RuntimeError):
    pass


def load_new_taus() -> dict[tuple[str, str], float]:
    frozen = json.loads((CAL_REPAIRED / "thresholds.json").read_text())
    taus = {}
    for entry in frozen:
        if entry.get("evaluator_revision") != 2:
            raise FailClosed(f"non-rev2 tau entry: {entry}")
        taus[(entry["model"], entry["regime"])] = entry["tau"]
    for m in MODELS:
        for r in REGIMES:
            if (m, r) not in taus:
                raise FailClosed(f"missing repaired tau for {m} {r}")
    return taus


def _pct(values, q):
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def rebuild_cell(model, regime, rows, gt_index):
    path = RESULTS / f"{model}_{regime}_main.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    headers = [json.loads(l) for l in lines if '"episode_header"' in l]
    steps = [json.loads(l) for l in lines if '"type": "step"' in l]
    if len(headers) != 48:
        raise FailClosed(f"{model} {regime}: {len(headers)} headers != 48")
    for h in headers:
        if h.get("stub_fallback") is not False:
            raise FailClosed(f"{model} {regime}: stub fallback present")
        if h.get("lifecycle") != "BOUND":
            raise FailClosed(f"{model} {regime}: lifecycle={h.get('lifecycle')}")
    by_ep: dict[str, list[dict]] = defaultdict(list)
    for s in steps:
        if s.get("model") != model or s.get("regime") != regime:
            raise FailClosed(f"{model} {regime}: step cell mismatch")
        by_ep[s["episode_id"]].append(s)
    row_by_ep = {r["episode_id"]: r for r in rows}
    if set(by_ep) != set(row_by_ep):
        raise FailClosed(f"{model} {regime}: episode set mismatch")
    records, out_lines, step_ms = [], [], []
    frame_ms: int | None = None
    for header in headers:
        ep = header["episode_id"]
        row = row_by_ep[ep]
        key = (str(row["corpus"]).lower(), row["session_id"])
        gt = gt_index.get(key)
        if gt is None:
            raise FailClosed(f"{ep}: no GT intervals")
        eval_start = int(row["evaluation_start_ms"])
        eval_end = int(row["evaluation_end_ms"])
        anchor = row["anchor_speaker"]
        ep_steps = sorted(by_ep[ep], key=lambda s: s["source_time_ms"])
        gaps = {b["source_time_ms"] - a["source_time_ms"]
                for a, b in zip(ep_steps, ep_steps[1:])}
        if len(gaps) != 1:
            raise FailClosed(f"{ep}: irregular grid {sorted(gaps)}")
        fms = gaps.pop()
        frame_ms = fms if frame_ms is None else frame_ms
        if fms != frame_ms:
            raise FailClosed(f"{ep}: frame_ms drift")
        window_ms = eval_end - eval_start
        if window_ms % fms != 0 or len(ep_steps) != window_ms // fms:
            raise FailClosed(f"{ep}: count vs window mismatch")
        unit = audio_resolve.SAMPLES_PER_MS * 2 * fms
        frames = []
        for i, s in enumerate(ep_steps):
            t = eval_start + (i + 1) * fms
            if s["source_time_ms"] != t or s.get("lifecycle") != "BOUND":
                raise FailClosed(f"{ep}: grid/lifecycle mismatch at {i}")
            center = eval_start * audio_resolve.SAMPLES_PER_MS + (i * unit) // 2 + unit // 4
            any_speech = gt_any_speech(gt, center)
            anchor_speech = gt_anchor_speech(gt, anchor, center)
            frames.append({
                "speech_gt": any_speech, "anchor_speech_gt": anchor_speech,
                "anchor": float(s["anchor"]), "adapter_speech": None,
                "lifecycle": "BOUND", "source_time_ms": t,
            })
            out_lines.append(json.dumps({
                "type": "step", "episode_id": ep, "model": model,
                "regime": regime, "source_time_ms": t,
                "speech_gt": any_speech, "anchor_speech_gt": anchor_speech,
                "anchor": float(s["anchor"]), "lifecycle": "BOUND",
                "step_ms": s.get("step_ms"),
            }))
            if s.get("step_ms") is not None:
                step_ms.append(float(s["step_ms"]))
        contam, active = gt_window_stats(gt, anchor, eval_start, eval_end)
        out_lines.append(json.dumps({**header, "evaluator_revision": 2}))
        records.append({
            "episode_id": ep, "topology": str(row["topology"]),
            "authoritative_transition_ms": int(row["authoritative_transition_time_ms"]),
            "frames": frames, "contam_s": contam, "active_speech_s": active,
            "lifecycle": "BOUND",
            "gt_eval": compact_gt(gt, anchor, eval_start, eval_end),
        })
    assert frame_ms is not None
    return records, out_lines, frame_ms, step_ms

def main() -> None:
    freeze = verify_freeze()
    rows = load_main_rows()
    if len(rows) != 48:
        raise FailClosed(f"MAIN48 rows: {len(rows)} != 48")
    gt_index = load_gt_index()
    taus = load_new_taus()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "thresholds_frozen.json").write_text(
        json.dumps([{"model": m, "regime": r, "tau": taus[(m, r)],
                     "evaluator_revision": 2} for m in MODELS for r in REGIMES],
                   indent=2, sort_keys=True) + "\n", encoding="utf-8")

    cells, cpu_cells = [], {}
    for regime in ("O", "C"):
        for name in MODELS:
            tau = taus[(name, regime)]
            records, out_lines, frame_ms, step_ms = rebuild_cell(
                name, regime, rows, gt_index)
            bound = [r for r in records if r["lifecycle"] == "BOUND"]
            per_tau = [aggregate(bound, t, frame_ms) for t in TAU_GRID]
            agg = next(r for r in per_tau if r["tau"] == tau)
            cuts_at_tau = sum(len(replay_decisions(r["frames"], tau, frame_ms)[0])
                              for r in bound)
            sens_at_tau = sum(replay_decisions(r["frames"], tau, frame_ms)[1]
                              for r in bound)
            eps = [score_episode(r, tau, frame_ms) for r in bound]
            premature = sum(1 for e in eps
                            if e["topology"] in ("A->A+B->B",) and e["premature_cut"])
            (OUT / f"{name}_{regime}_main.jsonl").write_text(
                "\n".join(out_lines) + "\n", encoding="utf-8")
            (OUT / f"{name}_{regime}_calibration.jsonl").write_text(
                "\n".join(json.dumps({**r, "tau_frozen": tau}) for r in per_tau)
                + "\n", encoding="utf-8")
            cells.append({
                "model": name, "regime": regime, "tau": tau,
                "stub_fallback": False, "agg": agg,
                "view": topology_view(bound, tau, frame_ms),
                "cuts_at_tau": cuts_at_tau, "sens_at_tau": sens_at_tau,
                "premature_cut_episodes": premature,
                "n_episodes": len(rows), "n_unbound": 0, "unbound_fraction": 0.0,
            })
            audio_s = sum(len(r["frames"]) * frame_ms for r in bound) / 1000.0
            total_step_s = sum(step_ms) / 1000.0
            cpu_cells[f"{name}_{regime}"] = {
                "model": name, "regime": regime, "tau": tau,
                "stub_fallback": False, "evaluator_revision": 2,
                "timing": "reused archived step_ms (live run); bind/reset/RSS not re-timed",
                "n_steps": len(step_ms), "audio_s": audio_s,
                "step_ms_p50": _pct(step_ms, 0.5), "step_ms_p95": _pct(step_ms, 0.95),
                "step_ms_p99": _pct(step_ms, 0.99),
                "step_ms_max": max(step_ms) if step_ms else None,
                "rtf": (total_step_s / audio_s) if audio_s > 0 else None,
                "frame_ms": frame_ms, "reset_ms_p50": None, "bind_ms_p50": None,
                "bind_ms_p95": None, "rss_note": "reused archive: no RSS sampling",
            }
    verdict = {c["model"]: gate3_supported(c["agg"])
               for c in cells if c["regime"] == "O"}
    summary = render_summary(cells, verdict, False)
    summary += ("## Repaired-evaluator addendum (evaluator_revision=2)\n\n"
                "Transition-aware CUT validity (50 ms tolerance); contamination is "
                "decoder-dependent (current-segment numerator).\n\n"
                "| model | regime | premature-cut CUT episodes |\n"
                "|---|---|---|\n")
    for cell in cells:
        summary += (f"| {cell['model']} | {cell['regime']} | "
                    f"{cell['premature_cut_episodes']} |\n")
    (OUT / "summary.md").write_text(summary, encoding="utf-8")
    cpu_report = {"cells": cpu_cells, "gate": {},
                  "note": "step timings reused from frozen live-run archives; "
                          "bind/reset/RSS unavailable under reuse."}
    for key, cell in cpu_cells.items():
        p99, rtf, chunk = cell.get("step_ms_p99"), cell.get("rtf"), cell.get("frame_ms")
        cpu_report["gate"][key] = {
            "p99_lt_chunk": (p99 < chunk) if p99 is not None else None,
            "rtf_le_025": (rtf <= 0.25) if rtf is not None else None}
    (OUT / "cpu.json").write_text(
        json.dumps(cpu_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "provenance.json").write_text(json.dumps({
        "evaluator_revision": 2,
        "frozen_taus_from": "cal/results_repaired/thresholds.json (applied as-is, no MAIN48 retuning)",
        "raw_inference": "REUSED verbatim from main/results/{model}_{regime}_main.jsonl "
                         "(native weights, stub_fallback=false, all BOUND, 48 eps/cell). "
                         "Fresh inference NOT needed (frame-complete).",
        "manifest_sha256": freeze["file_sha256"],
        "supersedes_without_overwriting": "main/results/* (tau 0.05) VOID, preserved in place.",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
