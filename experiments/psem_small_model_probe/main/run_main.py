#!/usr/bin/env python3
"""Gate 3 (MAIN48 native-ceiling O) + Gate 4 (causal C) runner (issue #117).

Models firered/neovad x regimes O/C on all 48 MAIN48 rows with the EXACT
frozen Gate 2 taus (cal/results/thresholds.json) — no retuning, no MAIN48
viewing for tuning. Reuses the Gate 2 pattern (freeze sha check, lazy
adapters, GT speech gate, UNBOUND regime-C, per-step JSONL, 500 ms primary
+ 300 ms sensitivity stream via metrics.replay_decisions).

Gate 3 fail-fast: O runs first. If BOTH models collapse on O (missed_rate
== 1.0 on the CUT set at the frozen tau), the runner reports the verdict
and STOPS before C (no causal spin).

CPU gate is measured on the real path: per-step wall time around
adapter.step() only, bind/reset splits, RSS sampling (psutil when
available), thread counts. Hard gate: p99 step < frame chunk,
RTF <= 0.25, no backlog.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from experiments.psem_small_model_probe.adapter.protocol import frame_bytes
from experiments.psem_small_model_probe.adapter.stub import StubAdapter
from experiments.psem_small_model_probe.cal import audio_resolve
from experiments.psem_small_model_probe.cal.audio_resolve import SAMPLES_PER_MS
from experiments.psem_small_model_probe.cal.metrics import (
    CUT_TOPOLOGIES,
    TAU_GRID,
    aggregate,
    replay_decisions,
    score_episode,
)
from experiments.psem_small_model_probe.cal.run_cal import (
    FailClosed,
    gt_anchor_speech,
    gt_window_stats,
    load_adapter,
    load_gt_index,
    verify_freeze,
)

MAIN_DIR = Path(__file__).resolve().parent
CAL_RESULTS = MAIN_DIR.parent / "cal" / "results"
FROZEN_THRESHOLDS = CAL_RESULTS / "thresholds.json"
MANIFEST = MAIN_DIR.parent / "manifest" / "manifest.jsonl"
RESULTS = MAIN_DIR / "results"

DEV_ONLY_NOTE = (
    "V2 EVAL sessions reused as dev-only probe per program approval; "
    "no unbiased generalization claim; V3 fresh holdout required for selection claims."
)

MODELS = ("firered", "neovad")
REGIMES = ("O", "C")
RSS_EVERY_STEPS = 100


def load_main_rows() -> list[dict]:
    rows = [
        json.loads(line)
        for line in MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    main = [r for r in rows if r.get("split") == "MAIN48"]
    if len(main) != 48:
        raise FailClosed(f"expected 48 MAIN48 rows, got {len(main)}")
    return main


def load_frozen_taus() -> dict[tuple[str, str], float]:
    frozen = json.loads(FROZEN_THRESHOLDS.read_text(encoding="utf-8"))
    taus = {(r["model"], r["regime"]): float(r["tau"]) for r in frozen}
    for model in MODELS:
        for regime in REGIMES:
            if (model, regime) not in taus:
                raise FailClosed(f"no frozen tau for {model} x {regime}")
    return taus


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def _rss_sampler():
    try:
        import psutil

        proc = psutil.Process()
        return lambda: proc.memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def run_cell(
    adapter,
    adapter_name: str,
    stub_fallback: bool,
    fallback_reason: str,
    regime: str,
    rows: list[dict],
    gt_index: dict,
    tau: float,
    out_lines: list[str],
    cpu: dict,
) -> tuple[list[dict], dict]:
    """Run one adapter x regime cell at the frozen tau; times the real path."""
    frame_ms = int(adapter.frame_ms)
    unit = frame_bytes(frame_ms)
    rss = _rss_sampler()
    records: list[dict] = []
    n_unbound = 0
    step_ms: list[float] = []
    reset_ms: list[float] = []
    bind_ms: list[float] = []
    rss_peak = rss() if rss else None
    rss_start = rss_peak
    audio_s = 0.0
    step_audio_s = frame_ms / 1000.0
    n_steps = 0
    for row in rows:
        episode_id = row["episode_id"]
        header: dict = {
            "type": "episode_header",
            "episode_id": episode_id,
            "model": adapter_name,
            "regime": regime,
            "topology": row["topology"],
            "tau_frozen": tau,
            "stub_fallback": stub_fallback,
            "fallback_reason": fallback_reason,
            "lifecycle": "BOUND",
        }
        key = (str(row["corpus"]).lower(), row["session_id"])
        gt = gt_index.get(key)
        if gt is None:
            raise FailClosed(f"no GT intervals for {key} (episode {episode_id})")
        eval_start = int(row["evaluation_start_ms"])
        eval_end = int(row["evaluation_end_ms"])
        window_ms = eval_end - eval_start
        if window_ms % frame_ms != 0:
            raise FailClosed(
                f"episode {episode_id}: window {window_ms}ms not a "
                f"{frame_ms}ms-frame multiple"
            )
        n_frames = window_ms // frame_ms

        if regime == "C" and not row.get("causal_bindable", False):
            header["lifecycle"] = "UNBOUND"
            n_unbound += 1
            out_lines.append(json.dumps(header))
            contam, active = gt_window_stats(
                gt, row["anchor_speaker"], eval_start, eval_end
            )
            records.append(
                {
                    "episode_id": episode_id,
                    "topology": str(row["topology"]),
                    "authoritative_transition_ms": int(
                        row["authoritative_transition_time_ms"]
                    ),
                    "frames": [],
                    "contam_s": contam,
                    "active_speech_s": active,
                    "lifecycle": "UNBOUND",
                }
            )
            continue

        span_start, span_end = audio_resolve.span_for_regime(row, regime)
        wav = audio_resolve.resolve_audio(row)
        ref_pcm = audio_resolve.load_span(wav, span_start, span_end)
        ref_sha = audio_resolve.sha256_pcm(ref_pcm)
        eval_pcm = audio_resolve.load_span(wav, eval_start, eval_end)
        if len(ref_pcm) % unit != 0:
            raise FailClosed(f"episode {episode_id}: bind span not frame-aligned")
        if len(eval_pcm) != n_frames * unit:
            raise FailClosed(f"episode {episode_id}: eval window byte mismatch")

        header["bind_span_sha256"] = ref_sha
        t0 = time.perf_counter()
        adapter.reset()
        reset_ms.append((time.perf_counter() - t0) * 1000.0)
        t0 = time.perf_counter()
        try:
            adapter.bind(ref_pcm)
        except Exception as exc:
            header["lifecycle"] = "BIND_FAILED"
            header["bind_error"] = str(exc)
            out_lines.append(json.dumps(header))
            contam, active = gt_window_stats(
                gt, row["anchor_speaker"], eval_start, eval_end
            )
            records.append(
                {
                    "episode_id": episode_id,
                    "topology": str(row["topology"]),
                    "authoritative_transition_ms": int(
                        row["authoritative_transition_time_ms"]
                    ),
                    "frames": [],
                    "contam_s": contam,
                    "active_speech_s": active,
                    "lifecycle": "BIND_FAILED",
                }
            )
            continue
        bind_ms.append((time.perf_counter() - t0) * 1000.0)

        frames: list[dict] = []
        for i in range(n_frames):
            chunk = eval_pcm[i * unit:(i + 1) * unit]
            t0 = time.perf_counter()
            out = adapter.step(chunk)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            step_ms.append(dt_ms)
            n_steps += 1
            if rss is not None and n_steps % RSS_EVERY_STEPS == 0:
                rss_peak = max(rss_peak, rss())
            t = eval_start + (i + 1) * frame_ms
            center = eval_start * SAMPLES_PER_MS + (i * unit) // 2 + unit // 4
            speech_gt = gt_anchor_speech(gt, row["anchor_speaker"], center)
            frames.append(
                {
                    "speech_gt": speech_gt,
                    "anchor": float(out.anchor),
                    "adapter_speech": out.speech,
                    "lifecycle": "BOUND",
                    "source_time_ms": t,
                }
            )
            out_lines.append(
                json.dumps(
                    {
                        "type": "step",
                        "episode_id": episode_id,
                        "model": adapter_name,
                        "regime": regime,
                        "source_time_ms": t,
                        "speech_gt": speech_gt,
                        "anchor": frames[-1]["anchor"],
                        "lifecycle": "BOUND",
                        "step_ms": dt_ms,
                    }
                )
            )
        audio_s += window_ms / 1000.0
        contam, active = gt_window_stats(gt, row["anchor_speaker"], eval_start, eval_end)
        out_lines.append(json.dumps(header))
        records.append(
            {
                "episode_id": episode_id,
                "topology": str(row["topology"]),
                "authoritative_transition_ms": int(
                    row["authoritative_transition_time_ms"]
                ),
                "frames": frames,
                "contam_s": contam,
                "active_speech_s": active,
                "lifecycle": "BOUND",
            }
        )
    rss_end = rss() if rss else None
    if rss is not None:
        rss_peak = max(rss_peak, rss_end)
    total_step_s = sum(step_ms) / 1000.0
    cpu.update(
        {
            "n_steps": len(step_ms),
            "audio_s": audio_s,
            "step_ms_p50": _pct(step_ms, 0.5),
            "step_ms_p95": _pct(step_ms, 0.95),
            "step_ms_p99": _pct(step_ms, 0.99),
            "step_ms_max": max(step_ms) if step_ms else None,
            "rtf": (total_step_s / audio_s) if audio_s > 0 else None,
            "frame_ms": frame_ms,
            "reset_ms_p50": _pct(reset_ms, 0.5),
            "bind_ms_p50": _pct(bind_ms, 0.5),
            "bind_ms_p95": _pct(bind_ms, 0.95),
            "rss_start_mb": rss_start,
            "rss_peak_mb": rss_peak,
            "rss_end_mb": rss_end,
            "rss_note": None if rss else "psutil unavailable: wall-time only",
        }
    )
    stats = {
        "n_episodes": len(rows),
        "n_unbound": n_unbound,
        "unbound_fraction": (n_unbound / len(rows)) if rows else 0.0,
    }
    return records, stats


def topology_view(records: list[dict], tau: float, frame_ms: int) -> dict:
    """Mandatory A->A+B->A KEEP vs A->A+B->B CUT split + sens stream.

    Remaining KEEP topologies (A, overlap_return, A+A+B) follow the same
    KEEP shape. MAIN48 may carry zero A->A+B->A rows — still reported
    as n=0, never dropped.
    """
    view: dict = {}
    for topo in ("A->A+B->A", "A->A+B->B", "A", "overlap_return", "A+A+B"):
        subset = [r for r in records if r["topology"] == topo and r["lifecycle"] == "BOUND"]
        if not subset and topo not in ("A->A+B->A", "A->A+B->B"):
            continue
        eps = [score_episode(r, tau, frame_ms) for r in subset]
        cuts = sum(e["n_cuts"] for e in eps)
        sens = sum(e["sens_hits"] for e in eps)
        if topo in CUT_TOPOLOGIES:
            view[topo] = {
                "n": len(eps),
                "cut_success": sum(1 for e in eps if not e["missed"]),
                "missed": sum(1 for e in eps if e["missed"]),
                "total_cuts": cuts,
                "sens_hits": sens,
            }
        else:
            view[topo] = {
                "n": len(eps),
                "keep_success": sum(1 for e in eps if not e["false_cut"]),
                "false_cuts": sum(1 for e in eps if e["false_cut"]),
                "sens_hits": sens,
            }
    return view


def gate3_supported(agg: dict) -> bool:
    """A formulation is supported iff it detects at least one CUT episode."""
    return (agg.get("missed_rate") is not None) and agg["missed_rate"] < 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate 3+4 MAIN48 runner")
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--adapters", nargs="+", default=list(MODELS))
    parser.add_argument("--regimes", nargs="+", default=list(REGIMES),
                        help="subset of O C (default runs O, gate-checks, then C)")
    args = parser.parse_args()

    verify_freeze()
    rows = load_main_rows()
    gt_index = load_gt_index()
    taus = load_frozen_taus()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "thresholds_frozen.json").write_text(
        json.dumps(
            [
                {"model": m, "regime": r, "tau": taus[(m, r)]}
                for m in MODELS
                for r in REGIMES
            ],
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    cells: list[dict] = []
    cpu_cells: dict = {}
    gate3_verdict: dict = {}
    run_order = [r for r in ("O", "C") if r in args.regimes]
    stop_before_c = False
    for regime in run_order:
        if regime == "C" and stop_before_c:
            break
        for name in args.adapters:
            if name not in MODELS:
                raise FailClosed(f"unknown adapter {name!r}")
            adapter, fallback, reason = load_adapter(name)
            tau = taus[(name, regime)]
            out_lines: list[str] = []
            cpu: dict = {}
            try:
                records, stats = run_cell(
                    adapter, name, fallback, reason, regime,
                    rows, gt_index, tau, out_lines, cpu,
                )
            except Exception as exc:
                if fallback:
                    raise
                adapter = StubAdapter()
                fallback, reason = True, f"runtime failed, stub fallback: {exc}"
                out_lines, cpu = [], {}
                records, stats = run_cell(
                    adapter, name, fallback, reason, regime,
                    rows, gt_index, tau, out_lines, cpu,
                )
            bound = [r for r in records if r["lifecycle"] == "BOUND"]
            frame_ms = int(adapter.frame_ms)
            per_tau = [aggregate(bound, t, frame_ms) for t in TAU_GRID]
            agg = next(r for r in per_tau if r["tau"] == tau)
            cuts_at_tau = sum(
                len(replay_decisions(r["frames"], tau, frame_ms)[0]) for r in bound
            )
            sens_at_tau = sum(
                replay_decisions(r["frames"], tau, frame_ms)[1] for r in bound
            )
            (args.results_dir / f"{name}_{regime}_main.jsonl").write_text(
                "\n".join(out_lines) + "\n", encoding="utf-8"
            )
            (args.results_dir / f"{name}_{regime}_calibration.jsonl").write_text(
                "\n".join(
                    json.dumps({**r, "tau_frozen": tau}) for r in per_tau
                )
                + "\n",
                encoding="utf-8",
            )
            cells.append(
                {
                    "model": name,
                    "regime": regime,
                    "tau": tau,
                    "stub_fallback": fallback,
                    "agg": agg,
                    "view": topology_view(bound, tau, frame_ms),
                    "cuts_at_tau": cuts_at_tau,
                    "sens_at_tau": sens_at_tau,
                    **stats,
                }
            )
            cpu_cells[f"{name}_{regime}"] = {
                "model": name,
                "regime": regime,
                "tau": tau,
                "stub_fallback": fallback,
                **cpu,
            }
        if regime == "O":
            o_cells = [c for c in cells if c["regime"] == "O"]
            supported = {c["model"]: gate3_supported(c["agg"]) for c in o_cells}
            gate3_verdict = supported
            if o_cells and "C" in run_order and not any(supported.values()):
                stop_before_c = True

    try:
        import psutil

        threads = psutil.Process().num_threads()
        threads_note = None
    except Exception:
        threads, threads_note = None, "psutil unavailable"
    try:
        import torch

        torch_threads = {
            "intraop": torch.get_num_threads(),
            "interop": torch.get_num_interop_threads(),
        }
    except Exception:
        torch_threads = None
    cpu_report = {
        "cells": cpu_cells,
        "process_threads": threads,
        "threads_note": threads_note,
        "torch_threads": torch_threads,
        "cpu_count": os.cpu_count(),
        "gate": {},
    }
    for key, cell in cpu_cells.items():
        p99 = cell.get("step_ms_p99")
        rtf = cell.get("rtf")
        chunk = cell.get("frame_ms")
        cpu_report["gate"][key] = {
            "p99_lt_chunk": (p99 < chunk) if p99 is not None else None,
            "rtf_le_025": (rtf <= 0.25) if rtf is not None else None,
        }
    (args.results_dir / "cpu.json").write_text(
        json.dumps(cpu_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.results_dir / "summary.md").write_text(
        render_summary(cells, gate3_verdict, stop_before_c), encoding="utf-8"
    )
    print(render_summary(cells, gate3_verdict, stop_before_c))


def render_summary(cells: list[dict], verdict: dict, stopped: bool) -> str:
    lines = [
        "# MAIN48 native-ceiling (O) + causal (C) scoring — Gates 3+4",
        "",
        f"> {DEV_ONLY_NOTE}",
        "",
        "> Frozen Gate 2 taus applied as-is (firered/neovad x O/C); "
        "no retuning on MAIN48.",
        "",
        "| model | regime | tau | contam s/h | false cuts (KEEP-n) | "
        "missed (CUT-n) | src_err p50/p90 (ms) | dec p50/p90 (ms) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for cell in cells:
        agg = cell["agg"]
        lines.append(
            f"| {cell['model']} | {cell['regime']} | {cell['tau']} | "
            f"{_fmt(agg['contamination_s_per_speech_h'])} | "
            f"{agg['false_cuts']}/{agg['n_keep']} | "
            f"{agg['missed']}/{agg['n_cut']} | "
            f"{_fmt(agg['delay_src_err_p50'])}/{_fmt(agg['delay_src_err_p90'])} | "
            f"{_fmt(agg['delay_dec_p50'])}/{_fmt(agg['delay_dec_p90'])} |"
        )
    lines += [
        "",
        "## Mandatory topology views (frozen tau)",
        "",
        "| model | regime | A->A+B->A KEEP ok/n (false) | "
        "A->A+B->B CUT ok/n (missed) | sens hits (KEEP/CUT) |",
        "|---|---|---|---|---|",
    ]
    for cell in cells:
        keep = cell["view"].get("A->A+B->A", {})
        cut = cell["view"].get("A->A+B->B", {})
        lines.append(
            f"| {cell['model']} | {cell['regime']} | "
            f"{keep.get('keep_success')}/{keep.get('n')} "
            f"({keep.get('false_cuts')}) | "
            f"{cut.get('cut_success')}/{cut.get('n')} "
            f"({cut.get('missed')}) | "
            f"{keep.get('sens_hits')}/{cut.get('sens_hits')} |"
        )
    lines += [
        "",
        "KEEP breakdown by topology (frozen tau):",
        "",
        "| model | regime | topology | KEEP ok/n (false) | sens hits |",
        "|---|---|---|---|---|",
    ]
    for cell in cells:
        for topo in ("A", "overlap_return", "A+A+B"):
            v = cell["view"].get(topo)
            if v is None:
                continue
            lines.append(
                f"| {cell['model']} | {cell['regime']} | {topo} | "
                f"{v.get('keep_success')}/{v.get('n')} ({v.get('false_cuts')}) | "
                f"{v.get('sens_hits')} |"
            )
    lines += [
        "",
        "Diagnostics only: frame AUPRC/F1 "
        "(`frame_auprc_diag`, `frame_f1_diag` in `*_calibration.jsonl`), "
        "unbound fraction, role-flip agreement.",
        "",
        "## Gate 3 verdict (native O)",
    ]
    if verdict:
        for model, ok in verdict.items():
            status = "SUPPORTED" if ok else "COLLAPSED (missed_rate=1.0 at frozen tau)"
            lines.append(f"- {model} O: {status}")
        if stopped:
            lines.append(
                "- GATE 3 FAIL-FAST: both formulations unsupported under "
                "native O — STOPPED before causal (no Gate 4 spin)."
            )
        else:
            lines.append(
                "- Gate 3: at least one formulation supported under native O "
                "— proceeded to Gate 4 causal."
            )
    else:
        lines.append("- Gate 3 verdict pending (O not run).")
    lines += [
        "",
        "## Gate 4 causal gap (O vs C at frozen tau)",
        "",
        "See `*_calibration.jsonl` per-tau replay rows and the headline "
        "table above; 300 ms sensitivity stream (`sens_hits`) is reversal "
        "detection only, no frontier.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _fmt(value) -> str:
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


if __name__ == "__main__":
    main()
