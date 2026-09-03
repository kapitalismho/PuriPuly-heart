#!/usr/bin/env python3
"""CAL12 re-freeze under the repaired V2 evaluator (evaluator_revision=2).

Raw-inference REUSE (no adapter inference, no audio reload):
  per-frame raw anchor scores (episode_id/source_time_ms/anchor/lifecycle/
  model/regime) are reused verbatim from the frozen
  ``cal/results/{model}_{regime}_steps.jsonl`` files (native weights,
  stub_fallback=false, dry_run=false, verified per header).

Freshly regenerated under frozen inputs:
  manifest hash verified + CAL12==12 rows from the frozen manifest; GT index
  regenerated with the frozen loader; frame ``speech_gt`` recomputed with
  the repaired any-speech gate (+ ``anchor_speech_gt`` diagnostic);
  ``gt_eval`` compact spans attached; decoder replayed over the EXISTING
  TAU_GRID with the EXISTING fixed-priority rule. Fail-closed on any grid/
  count/lifecycle mismatch.

Writes ``cal/results_repaired/`` (broken ``cal/results/`` untouched):
  {model}_{regime}_steps.jsonl (repaired frames), {model}_{regime}_
  calibration.jsonl (per-tau aggregates), thresholds.json, summary.md,
  provenance.json.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.psem_small_model_probe.cal import audio_resolve  # noqa: E402
from experiments.psem_small_model_probe.cal.eval_semantics import (  # noqa: E402
    CUT_TOLERANCE_MS,
    compact_gt,
    gt_anchor_speech,
    gt_any_speech,
    gt_window_stats,
)
from experiments.psem_small_model_probe.cal.metrics import (  # noqa: E402
    TAU_GRID,
    aggregate,
    replay_decisions,
    select_threshold,
)
from experiments.psem_small_model_probe.cal.run_cal import (  # noqa: E402
    _totals,
    load_cal_rows,
    load_gt_index,
    render_summary,
    topology_of,
    verify_freeze,
)

CAL_DIR = Path(__file__).resolve().parent
RESULTS = CAL_DIR / "results"
OUT = CAL_DIR / "results_repaired"
CELLS = [("firered", "O"), ("firered", "C"), ("neovad", "O"), ("neovad", "C")]


class FailClosed(RuntimeError):
    pass


def load_reused_steps(model: str, regime: str) -> tuple[list[dict], list[dict]]:
    path = RESULTS / f"{model}_{regime}_steps.jsonl"
    headers = [json.loads(l) for l in path.read_text().splitlines()
               if '"episode_header"' in l]
    steps = [json.loads(l) for l in path.read_text().splitlines()
             if '"type": "step"' in l]
    if not headers:
        raise FailClosed(f"{path}: no episode headers")
    for h in headers:
        if h.get("stub_fallback") is not False:
            raise FailClosed(f"{model} {regime}: stub_fallback={h.get('stub_fallback')}")
        if h.get("dry_run") is not False:
            raise FailClosed(f"{model} {regime}: dry_run flag set")
        if h.get("lifecycle") != "BOUND":
            raise FailClosed(f"{model} {regime}: lifecycle={h.get('lifecycle')}")
        if h.get("model") != model or h.get("regime") != regime:
            raise FailClosed(f"{model} {regime}: header cell mismatch")
    return headers, steps


def rebuild_records(model, regime, rows, gt_index, headers, steps):
    by_ep: dict[str, list[dict]] = defaultdict(list)
    for s in steps:
        if s.get("model") != model or s.get("regime") != regime:
            raise FailClosed(f"{model} {regime}: step cell mismatch")
        by_ep[s["episode_id"]].append(s)
    row_by_ep = {r["episode_id"]: r for r in rows}
    if set(by_ep) != set(row_by_ep):
        raise FailClosed(
            f"{model} {regime}: episode set mismatch "
            f"(steps={sorted(set(by_ep))} manifest={sorted(set(row_by_ep))})")
    records = []
    out_lines: list[str] = []
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
            raise FailClosed(f"{ep}: irregular step grid {sorted(gaps)}")
        fms = gaps.pop()
        frame_ms = fms if frame_ms is None else frame_ms
        if fms != frame_ms:
            raise FailClosed(f"{ep}: frame_ms drift {fms} vs {frame_ms}")
        window_ms = eval_end - eval_start
        if window_ms % fms != 0 or len(ep_steps) != window_ms // fms:
            raise FailClosed(f"{ep}: {len(ep_steps)} steps vs window {window_ms}ms")
        unit = audio_resolve.SAMPLES_PER_MS * 2 * fms  # int16 mono @16kHz
        frames = []
        for i, s in enumerate(ep_steps):
            t = eval_start + (i + 1) * fms
            if s["source_time_ms"] != t or s.get("lifecycle") != "BOUND":
                raise FailClosed(f"{ep}: grid/lifecycle mismatch at frame {i}")
            center = eval_start * audio_resolve.SAMPLES_PER_MS + (i * unit) // 2 + unit // 4
            any_speech = gt_any_speech(gt, center)
            anchor_speech = gt_anchor_speech(gt, anchor, center)
            frames.append({
                "speech_gt": any_speech,
                "anchor_speech_gt": anchor_speech,
                "anchor": float(s["anchor"]),
                "adapter_speech": None,  # raw reuse carries anchor scores only
                "lifecycle": "BOUND",
                "source_time_ms": t,
            })
            out_lines.append(json.dumps({
                "type": "step", "episode_id": ep, "model": model,
                "regime": regime, "source_time_ms": t,
                "speech_gt": any_speech, "anchor_speech_gt": anchor_speech,
                "anchor": float(s["anchor"]), "lifecycle": "BOUND",
            }))
        contam, active = gt_window_stats(gt, anchor, eval_start, eval_end)
        out_lines.append(json.dumps({**header, "evaluator_revision": 2}))
        records.append({
            "episode_id": ep,
            "topology": topology_of(row),
            "authoritative_transition_ms": int(row["authoritative_transition_time_ms"]),
            "frames": frames,
            "contam_s": contam,
            "active_speech_s": active,
            "lifecycle": "BOUND",
            "gt_eval": compact_gt(gt, anchor, eval_start, eval_end),
        })
    assert frame_ms is not None
    return records, out_lines, frame_ms


def main() -> None:
    freeze = verify_freeze()
    rows = load_cal_rows()
    if len(rows) != 12:
        raise FailClosed(f"CAL12 rows: {len(rows)} != 12")
    gt_index = load_gt_index()
    OUT.mkdir(parents=True, exist_ok=True)

    thresholds: list[dict] = []
    summary_cells: list[dict] = []
    for model, regime in CELLS:
        headers, steps = load_reused_steps(model, regime)
        records, out_lines, frame_ms = rebuild_records(
            model, regime, rows, gt_index, headers, steps)
        bound = [r for r in records if r["lifecycle"] == "BOUND"]
        per_tau = [aggregate(bound, tau, frame_ms) for tau in TAU_GRID]
        tau, why = select_threshold(per_tau)
        (OUT / f"{model}_{regime}_steps.jsonl").write_text(
            "\n".join(out_lines) + "\n", encoding="utf-8")
        (OUT / f"{model}_{regime}_calibration.jsonl").write_text(
            "\n".join(json.dumps(r) for r in per_tau) + "\n", encoding="utf-8")
        cuts_at_tau, sens_at_tau = _totals(bound, tau, frame_ms)
        thresholds.append({"model": model, "regime": regime, "tau": tau,
                           "evaluator_revision": 2, "selection_reason": why})
        summary_cells.append({
            "model": model, "regime": regime, "tau": tau,
            "stub_fallback": False, "dry_run": False,
            "agg": next(r for r in per_tau if r["tau"] == tau),
            "cuts_at_tau": cuts_at_tau, "sens_at_tau": sens_at_tau,
            "n_episodes": len(rows), "n_unbound": 0, "unbound_fraction": 0.0,
        })
        agg = summary_cells[-1]["agg"]
        print(f"{model} {regime}: tau={tau} false={agg['false_cuts']}/{agg['n_keep']} "
              f"missed={agg['missed']}/{agg['n_cut']} "
              f"src_p50={agg['delay_src_err_p50']} dec_p50={agg['delay_dec_p50']} "
              f"contam={agg['contamination_s_per_speech_h']} frames={frame_ms}ms")

    (OUT / "thresholds.json").write_text(
        json.dumps(thresholds, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "summary.md").write_text(
        render_summary(summary_cells), encoding="utf-8")
    provenance = {
        "evaluator_revision": 2,
        "repair": "any-speech decoder gate; transition-aware CUT validity "
                  f"(tolerance {CUT_TOLERANCE_MS}ms = annotation_boundary_jitter, "
                  "V2 operational_label_contract.json psem-handoff-v1); "
                  "decoder-dependent current-segment contamination "
                  "(source_boundary_time, never decision_time)",
        "raw_inference": "REUSED verbatim from cal/results/{model}_{regime}_steps.jsonl "
                         "(native weights, stub_fallback=false, dry_run=false, "
                         "all BOUND; anchor scores + source_time grid only; "
                         "old anchor-gated speech_gt discarded). "
                         "No adapter inference, no audio reload, no GT reselect.",
        "regenerated_frozen": "manifest sha verified; CAL12==12 rows; GT index via "
                              "frozen loader; speech_gt/anchor_speech_gt/gt_eval rebuilt; "
                              "existing TAU_GRID 0.05..0.95 + existing fixed-priority rule.",
        "manifest_sha256": freeze["file_sha256"],
        "supersedes_without_overwriting": "cal/results/thresholds.json + "
            "main/results/summary.md (tau 0.05 everywhere) are VOID, preserved in place.",
    }
    (OUT / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(thresholds, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
