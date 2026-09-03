#!/usr/bin/env python3
"""Gate 2 CAL12 threshold runner scaffold (issue #117).

Adapters x regimes x CAL12 episodes -> single-scalar thresholds
(see metrics.py for the fixed-priority rule).

Regimes: O = 5 s native bind span; C = 1 s causal bind span.
A row with ``causal_bindable=false`` stays UNBOUND in regime C (HOLD,
no inference). GT any-speech gates every frame before the decoder; the
500 ms confirmation decoder emits headline CUTs while a 300 ms
sensitivity stream is recorded alongside.

Adapters are imported lazily; missing weights fall back to StubAdapter
and are flagged ``stub_fallback`` in every header. ``--dry-run``
synthesizes deterministic zero PCM (flagged ``dry_run``) so the
pipeline, metrics, and threshold rule run without corpus audio.

No training, no MAIN48/EXT24 scoring, no dependency changes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path

from experiments.psem_small_model_probe.adapter.protocol import frame_bytes
from experiments.psem_small_model_probe.adapter.stub import StubAdapter
from experiments.psem_small_model_probe.cal import audio_resolve
from experiments.psem_small_model_probe.cal.audio_resolve import SAMPLES_PER_MS
from experiments.psem_small_model_probe.cal.eval_semantics import (
    compact_gt,
    gt_anchor_speech,
    gt_any_speech,
    gt_window_stats,
)
from experiments.psem_small_model_probe.cal.metrics import (
    TAU_GRID,
    aggregate,
    replay_decisions,
    select_threshold,
)

CAL_DIR = Path(__file__).resolve().parent
MANIFEST = CAL_DIR.parent / "manifest" / "manifest.jsonl"
FREEZE = CAL_DIR.parent / "manifest" / "dataset_freeze.json"
RESULTS = CAL_DIR / "results"

DEV_ONLY_NOTE = (
    "V2 EVAL sessions reused as dev-only probe per program approval; "
    "no unbiased generalization claim; V3 fresh holdout required for selection claims."
)


class FailClosed(RuntimeError):
    pass


def verify_freeze() -> dict:
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    actual = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    if actual != freeze["file_sha256"]:
        raise FailClosed(
            f"manifest hash mismatch: file={actual} "
            f"freeze={freeze['file_sha256']} (refusing to run)"
        )
    return freeze


def load_cal_rows() -> list[dict]:
    rows = [
        json.loads(line)
        for line in MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cal = [r for r in rows if r.get("split") == "CAL12"]
    if not cal:
        raise FailClosed("no CAL12 rows in frozen manifest")
    return cal


def load_gt_index() -> dict[tuple[str, str], dict]:
    """(corpus.lower(), session_id) -> {audio_ref, intervals sorted by start}."""
    index: dict[tuple[str, str], dict] = {}
    for path in audio_resolve.OCC_MANIFESTS:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["corpus"]).lower(), row["session_id"])
            intervals = sorted(row["intervals"], key=lambda iv: iv["start_sample"])
            index[key] = {
                "audio_ref": row.get("audio_ref"),
                "intervals": intervals,
                "starts": [iv["start_sample"] for iv in intervals],
            }
    return index


def load_adapter(name: str):
    """Lazy adapter import; missing weights -> StubAdapter fallback (flagged)."""
    if name == "stub":
        return StubAdapter(), False, "native stub"
    try:
        module = importlib.import_module(
            f"experiments.psem_small_model_probe.adapter.{name}"
        )
    except Exception as exc:
        return StubAdapter(), True, f"import failed, stub fallback: {exc}"
    for attr in dir(module):
        if attr.startswith("_"):
            continue
        candidate = getattr(module, attr)
        if (
            isinstance(candidate, type)
            and hasattr(candidate, "reset")
            and hasattr(candidate, "bind")
            and hasattr(candidate, "step")
        ):
            try:
                return candidate(), False, "native weights"
            except Exception as exc:
                return StubAdapter(), True, f"construct failed, stub fallback: {exc}"
    return StubAdapter(), True, "no adapter class found, stub fallback"


def run_cell(
    adapter,
    adapter_name: str,
    stub_fallback: bool,
    fallback_reason: str,
    regime: str,
    rows: list[dict],
    gt_index: dict,
    dry_run: bool,
    out_lines: list[str],
) -> tuple[list[dict], dict]:
    """Run one adapter x regime cell; returns (episode records, cell stats)."""
    frame_ms = int(adapter.frame_ms)
    records: list[dict] = []
    n_unbound = 0
    for row in rows:
        episode_id = row["episode_id"]
        header: dict = {
            "type": "episode_header",
            "episode_id": episode_id,
            "model": adapter_name,
            "regime": regime,
            "topology": row["topology"],
            "stub_fallback": stub_fallback,
            "fallback_reason": fallback_reason,
            "dry_run": dry_run,
            "lifecycle": "BOUND",
        }
        key = (str(row["corpus"]).lower(), row["session_id"])
        gt = gt_index.get(key)
        if gt is None and not dry_run:
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
        unit = frame_bytes(frame_ms)

        if regime == "C" and not row.get("causal_bindable", False):
            header["lifecycle"] = "UNBOUND"
            n_unbound += 1
            out_lines.append(json.dumps(header))
            records.append(
                _empty_record(row, lifecycle="UNBOUND",
                              contam=(0.0, 0.0) if gt is None else None,
                              gt=gt)
            )
            continue

        span_start, span_end = audio_resolve.span_for_regime(row, regime)
        if dry_run:
            ref_pcm = b"\x00" * ((span_end - span_start) * SAMPLES_PER_MS * 2)
            eval_pcm = b"\x00" * (window_ms * SAMPLES_PER_MS * 2)
            ref_sha = "dry_run"
        else:
            wav = audio_resolve.resolve_audio(row)
            ref_pcm = audio_resolve.load_span(wav, span_start, span_end)
            ref_sha = audio_resolve.sha256_pcm(ref_pcm)
            eval_pcm = audio_resolve.load_span(wav, eval_start, eval_end)
        if len(ref_pcm) % unit != 0:
            raise FailClosed(f"episode {episode_id}: bind span not frame-aligned")
        if len(eval_pcm) != n_frames * unit:
            raise FailClosed(f"episode {episode_id}: eval window byte mismatch")

        header["bind_span_sha256"] = ref_sha
        adapter.reset()
        try:
            adapter.bind(ref_pcm)
        except Exception as exc:
            header["lifecycle"] = "BIND_FAILED"
            header["bind_error"] = str(exc)
            out_lines.append(json.dumps(header))
            records.append(_empty_record(row, lifecycle="BIND_FAILED",
                                         contam=None, gt=gt))
            continue

        frames: list[dict] = []
        for i in range(n_frames):
            chunk = eval_pcm[i * unit:(i + 1) * unit]
            out = adapter.step(chunk)
            t = eval_start + (i + 1) * frame_ms
            center = eval_start * SAMPLES_PER_MS + (i * unit) // 2 + unit // 4
            any_speech = gt_any_speech(gt, center) if gt else False
            anchor_speech = (
                gt_anchor_speech(gt, row["anchor_speaker"], center) if gt else False
            )
            frames.append(
                {
                    "speech_gt": any_speech,
                    "anchor_speech_gt": anchor_speech,
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
                        "speech_gt": any_speech,
                        "anchor_speech_gt": anchor_speech,
                        "anchor": frames[-1]["anchor"],
                        "lifecycle": "BOUND",
                    }
                )
            )
        contam, active = (
            gt_window_stats(gt, row["anchor_speaker"], eval_start, eval_end)
            if gt
            else (0.0, 0.0)
        )
        out_lines.append(json.dumps(header))
        records.append(
            {
                "episode_id": episode_id,
                "topology": topology_of(row),
                "authoritative_transition_ms": int(
                    row["authoritative_transition_time_ms"]
                ),
                "frames": frames,
                "contam_s": contam,
                "active_speech_s": active,
                "lifecycle": "BOUND",
                "gt_eval": compact_gt(
                    gt, row["anchor_speaker"], eval_start, eval_end
                ),
            }
        )
    stats = {"n_episodes": len(rows), "n_unbound": n_unbound,
             "unbound_fraction": (n_unbound / len(rows)) if rows else 0.0}
    return records, stats


def _empty_record(row: dict, lifecycle: str, contam, gt) -> dict:
    if contam is None and gt is not None:
        contam, active = gt_window_stats(
            gt, row["anchor_speaker"],
            int(row["evaluation_start_ms"]), int(row["evaluation_end_ms"]),
        )
    elif contam is None:
        contam, active = 0.0, 0.0
    else:
        active = 0.0
    return {
        "episode_id": row["episode_id"],
        "topology": topology_of(row),
        "authoritative_transition_ms": int(row["authoritative_transition_time_ms"]),
        "frames": [],
        "contam_s": contam,
        "active_speech_s": active,
        "lifecycle": lifecycle,
        "gt_eval": compact_gt(
            gt,
            row["anchor_speaker"],
            int(row["evaluation_start_ms"]),
            int(row["evaluation_end_ms"]),
        ),
    }


def topology_of(row: dict) -> str:
    return str(row["topology"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate 2 CAL12 threshold runner")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--adapters", nargs="+", default=["stub", "firered", "neovad"])
    args = parser.parse_args()

    verify_freeze()
    rows = load_cal_rows()
    gt_index = load_gt_index()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    thresholds: list[dict] = []
    summary_cells: list[dict] = []
    for name in args.adapters:
        adapter, fallback, reason = load_adapter(name)
        for regime in ("O", "C"):
            out_lines: list[str] = []
            try:
                records, stats = run_cell(
                    adapter, name, fallback, reason, regime,
                    rows, gt_index, args.dry_run, out_lines,
                )
            except Exception as exc:
                # Adapter runtime blew up mid-cell (e.g. torch missing at
                # step time despite lazy construct): rerun cell on the stub,
                # flagged. Real weights stay a sibling-worker concern.
                if fallback:
                    raise
                adapter = StubAdapter()
                fallback, reason = True, f"runtime failed, stub fallback: {exc}"
                out_lines = []
                records, stats = run_cell(
                    adapter, name, fallback, reason, regime,
                    rows, gt_index, args.dry_run, out_lines,
                )
            bound = [r for r in records if r["lifecycle"] == "BOUND"]
            frame_ms = int(adapter.frame_ms)
            per_tau = [aggregate(bound, tau, frame_ms) for tau in TAU_GRID]
            tau, why = select_threshold(per_tau)
            (args.results_dir / f"{name}_{regime}_steps.jsonl").write_text(
                "\n".join(out_lines) + "\n", encoding="utf-8"
            )
            (args.results_dir / f"{name}_{regime}_calibration.jsonl").write_text(
                "\n".join(json.dumps(r) for r in per_tau) + "\n", encoding="utf-8"
            )
            cuts_at_tau, sens_at_tau = _totals(bound, tau, frame_ms)
            thresholds.append(
                {"model": name, "regime": regime, "tau": tau,
                 "selection_reason": why}
            )
            summary_cells.append(
                {
                    "model": name, "regime": regime, "tau": tau,
                    "stub_fallback": fallback, "dry_run": args.dry_run,
                    "agg": next(r for r in per_tau if r["tau"] == tau),
                    "cuts_at_tau": cuts_at_tau, "sens_at_tau": sens_at_tau,
                    **stats,
                }
            )

    (args.results_dir / "thresholds.json").write_text(
        json.dumps(thresholds, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.results_dir / "summary.md").write_text(
        render_summary(summary_cells), encoding="utf-8"
    )
    print(json.dumps(thresholds, indent=2, sort_keys=True))


def _totals(records: list[dict], tau: float, frame_ms: int) -> tuple[int, int]:
    cuts = sens = 0
    for record in records:
        cell_cuts, cell_sens = replay_decisions(record["frames"], tau, frame_ms)
        cuts += len(cell_cuts)
        sens += cell_sens
    return cuts, sens


def render_summary(cells: list[dict]) -> str:
    lines = [
        "# CAL12 threshold calibration (Gate 2 scaffold)",
        "",
        f"> {DEV_ONLY_NOTE}",
        "",
        "| model | regime | tau | false_cuts | missed | "
        "src_err p50/p90 (ms) | dec p50/p90 (ms) | contam s/h | stub | dry |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cell in cells:
        agg = cell["agg"]
        lines.append(
            f"| {cell['model']} | {cell['regime']} | {cell['tau']} | "
            f"{agg['false_cuts']}/{agg['n_keep']} | "
            f"{agg['missed']}/{agg['n_cut']} | "
            f"{agg['delay_src_err_p50']}/{agg['delay_src_err_p90']} | "
            f"{agg['delay_dec_p50']}/{agg['delay_dec_p90']} | "
            f"{agg['contamination_s_per_speech_h']} | "
            f"{cell['stub_fallback']} | {cell['dry_run']} |"
        )
    lines += [
        "",
        "Topology views: `A->A+B->A` KEEP vs `A->A+B->B` CUT reported "
        "separately in per-tau calibration rows "
        "(`*_calibration.jsonl`: `n_keep`, `n_cut`, `false_cuts`, `missed`).",
        "Frame AUPRC/F1 (`frame_*_diag`), unbound fraction, and role-flip "
        "agreement are diagnostics only.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
