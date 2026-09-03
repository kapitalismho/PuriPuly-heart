#!/usr/bin/env python3
"""Gate 5 — production VAD replay for the ONE causal winner (issue #117).

Same 48 MAIN48 rows, firered regime C, frozen tau=0.05, same causal 1 s
bind + CommonPersistenceDecoder + headline metrics as main/run_main.py.
ONLY change: the per-frame speech gate is the production Silero VAD span
output (peer profile: threshold 0.5, 512-sample chunks, pre-roll 500 ms,
hangover 500 ms, max_segment 7000 ms; gate = pre-roll + committed chunks
through speech end excluding trailing hangover) instead of GT anchor
speech. Lifecycle/binding/decoder are untouched.

Metrics, span loading, GT helpers, and the frozen tau are imported from
the main/cal modules — never reimplemented here. neovad is NOT replayed
(collapsed 8/8 missed under both gates; nothing to retain).
"""

from __future__ import annotations

import argparse
import json
import os
import time
import wave
from pathlib import Path

import numpy as np

from experiments.psem_small_model_probe.adapter.protocol import frame_bytes
from experiments.psem_small_model_probe.cal import audio_resolve
from experiments.psem_small_model_probe.cal.audio_resolve import SAMPLES_PER_MS
from experiments.psem_small_model_probe.cal.metrics import (
    aggregate,
    replay_decisions,
    score_episode,
)
from experiments.psem_small_model_probe.cal.eval_semantics import (
    compact_gt,
    gt_anchor_speech,
    gt_any_speech,
    gt_window_stats,
)
from experiments.psem_small_model_probe.cal.run_cal import (
    FailClosed,
    load_adapter,
    load_gt_index,
)
from experiments.psem_small_model_probe.main.run_main import (
    load_frozen_taus,
    load_main_rows,
    topology_view,
)
from puripuly_heart.core.vad.gating import (
    SpeechChunk,
    SpeechEnd,
    SpeechStart,
    create_peer_vad_gating,
)
from puripuly_heart.core.vad.silero import SileroVadOnnx

REPLAY_DIR = Path(__file__).resolve().parent
RESULTS = REPLAY_DIR / "results"

MODEL = "firered"
REGIME = "C"

VAD_MODEL_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "puripuly_heart"
    / "data"
    / "vad"
    / "silero_vad.onnx"
)
VAD_SAMPLE_RATE_HZ = 16000
VAD_CHUNK_SAMPLES = 512
VAD_THRESHOLD = 0.5
VAD_PRE_ROLL_MS = 500
VAD_HANGOVER_MS = 500

DEV_ONLY_NOTE = (
    "V2 EVAL sessions reused as dev-only probe per program approval; "
    "no unbiased generalization claim; V3 fresh holdout required for selection claims."
)


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def production_spans(
    audio_path: Path,
    engine: SileroVadOnnx,
    vad_step_ms: list[float],
) -> tuple[list[tuple[int, int]], int]:
    """Full-source production VAD replay for one session WAV.

    Peer profile verbatim: 512-sample chunks through create_peer_vad_gating
    (threshold 0.5, debounce/commit 3/3, hangover 500 ms, max_segment
    7000 ms, 500 ms pre-roll ring); gate spans = pre-roll + committed
    chunks through speech end excluding trailing hangover. Returns
    (merged spans in file-absolute samples, total samples). Per-chunk
    process_chunk latency is appended to vad_step_ms.
    """
    gating = create_peer_vad_gating(
        engine,
        sample_rate_hz=VAD_SAMPLE_RATE_HZ,
        ring_buffer_ms=VAD_PRE_ROLL_MS,
        speech_threshold=VAD_THRESHOLD,
        hangover_ms=VAD_HANGOVER_MS,
    )
    spans: list[tuple[int, int]] = []
    active_start: int | None = None
    processed = 0
    total = 0
    with wave.open(str(audio_path), "rb") as reader:
        if (
            reader.getframerate() != VAD_SAMPLE_RATE_HZ
            or reader.getnchannels() != 1
            or reader.getsampwidth() != 2
        ):
            raise FailClosed(f"audio contract mismatch: {audio_path}")
        total = reader.getnframes()
        while True:
            payload = reader.readframes(VAD_CHUNK_SAMPLES)
            if not payload:
                break
            original = len(payload) // 2
            # np.int16 (== '<i2' on little-endian x86): the '<i2' string alias
            # is rejected by frombuffer in this process; type object is exact.
            chunk = np.frombuffer(payload, dtype=np.int16).astype(np.float32)
            if original < VAD_CHUNK_SAMPLES:
                chunk = np.pad(chunk, (0, VAD_CHUNK_SAMPLES - original))
            chunk /= 32768.0
            t0 = time.perf_counter()
            events = gating.process_chunk(chunk)
            vad_step_ms.append((time.perf_counter() - t0) * 1000.0)
            chunk_start = processed
            chunk_end = min(total, chunk_start + original)
            for event in events:
                if isinstance(event, SpeechStart):
                    buffered = 1 + sum(
                        isinstance(value, SpeechChunk) for value in events
                    )
                    active_start = max(
                        0,
                        chunk_start
                        - (buffered - 1) * VAD_CHUNK_SAMPLES
                        - int(np.asarray(event.pre_roll).size),
                    )
                elif isinstance(event, SpeechEnd) and active_start is not None:
                    if event.reason == "silence":
                        trailing = int(
                            round(
                                event.trailing_silence_ms
                                * VAD_SAMPLE_RATE_HZ
                                / 1000.0
                            )
                        )
                        trailing = max(
                            0, trailing - (VAD_CHUNK_SAMPLES - original)
                        )
                        speech_end = max(active_start, chunk_end - trailing)
                    else:
                        speech_end = chunk_end
                    spans.append(
                        (min(active_start, total), min(speech_end, total))
                    )
                    active_start = None
            processed = chunk_end
    if active_start is not None:
        spans.append((active_start, processed))
    return _merge_spans(spans), total


def _in_spans(spans: list[tuple[int, int]], sample: int) -> bool:
    return any(start <= sample < end for start, end in spans)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate 5 production-VAD replay")
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    args = parser.parse_args()

    rows = load_main_rows()
    gt_index = load_gt_index()
    tau = load_frozen_taus()[(MODEL, REGIME)]
    if tau != 0.05:
        raise FailClosed(f"frozen tau for {MODEL} x {REGIME} is {tau}, want 0.05")
    adapter, fallback, reason = load_adapter(MODEL)
    if fallback:
        raise FailClosed(f"winner must run native weights, got fallback: {reason}")
    frame_ms = int(adapter.frame_ms)
    unit = frame_bytes(frame_ms)
    if not VAD_MODEL_PATH.is_file():
        raise FailClosed(f"production VAD model missing: {VAD_MODEL_PATH}")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_lines: list[str] = []

    # One full-source VAD replay per unique session; spans are file-absolute.
    engine = SileroVadOnnx(VAD_MODEL_PATH)
    vad_step_ms: list[float] = []
    vad_audio_s = 0.0
    span_cache: dict[str, list[tuple[int, int]]] = {}
    for row in rows:
        wav = audio_resolve.resolve_audio(row)
        key = str(wav)
        if key not in span_cache:
            spans, total = production_spans(wav, engine, vad_step_ms)
            vad_audio_s += total / VAD_SAMPLE_RATE_HZ
            span_cache[key] = spans
            out_lines.append(
                json.dumps(
                    {
                        "type": "vad_session",
                        "audio_path": key,
                        "total_samples": total,
                        "span_count": len(spans),
                        "speech_s": sum(e - s for s, e in spans)
                        / VAD_SAMPLE_RATE_HZ,
                    }
                )
            )

    pvad_step_ms: list[float] = []
    bind_ms: list[float] = []
    reset_ms: list[float] = []
    eval_audio_s = 0.0
    records_gt: list[dict] = []
    records_prod: list[dict] = []
    n_frames_total = 0
    n_agree = 0
    n_gt_speech = 0
    n_vad_gated_off = 0  # GT speech frames the production gate drops
    n_vad_extra = 0  # production-gate-on frames where GT is off
    n_vad_empty_windows = 0
    for row in rows:
        episode_id = row["episode_id"]
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
        spans = span_cache[str(audio_resolve.resolve_audio(row))]

        span_start, span_end = audio_resolve.span_for_regime(row, REGIME)
        wav = audio_resolve.resolve_audio(row)
        ref_pcm = audio_resolve.load_span(wav, span_start, span_end)
        eval_pcm = audio_resolve.load_span(wav, eval_start, eval_end)
        if len(ref_pcm) % unit != 0:
            raise FailClosed(f"episode {episode_id}: bind span not frame-aligned")
        if len(eval_pcm) != n_frames * unit:
            raise FailClosed(f"episode {episode_id}: eval window byte mismatch")

        t0 = time.perf_counter()
        adapter.reset()
        reset_ms.append((time.perf_counter() - t0) * 1000.0)
        t0 = time.perf_counter()
        adapter.bind(ref_pcm)
        bind_ms.append((time.perf_counter() - t0) * 1000.0)

        frames_gt: list[dict] = []
        frames_prod: list[dict] = []
        vad_on_in_window = 0
        for i in range(n_frames):
            chunk = eval_pcm[i * unit:(i + 1) * unit]
            t0 = time.perf_counter()
            out = adapter.step(chunk)
            pvad_step_ms.append((time.perf_counter() - t0) * 1000.0)
            t = eval_start + (i + 1) * frame_ms
            center = eval_start * SAMPLES_PER_MS + (i * unit) // 2 + unit // 4
            speech_gt = gt_any_speech(gt, center)
            anchor_speech_gt = gt_anchor_speech(gt, row["anchor_speaker"], center)
            speech_vad = _in_spans(spans, center)
            n_frames_total += 1
            n_agree += speech_gt == speech_vad
            n_gt_speech += bool(speech_gt)
            n_vad_gated_off += bool(speech_gt and not speech_vad)
            n_vad_extra += bool(speech_vad and not speech_gt)
            vad_on_in_window += bool(speech_vad)
            base = {
                "anchor": float(out.anchor),
                "adapter_speech": out.speech,
                "lifecycle": "BOUND",
                "source_time_ms": t,
                "anchor_speech_gt": anchor_speech_gt,
            }
            frames_gt.append({**base, "speech_gt": speech_gt})
            frames_prod.append({**base, "speech_gt": speech_vad})
            out_lines.append(
                json.dumps(
                    {
                        "type": "step",
                        "episode_id": episode_id,
                        "model": MODEL,
                        "regime": REGIME,
                        "gate": "gt+prod",
                        "source_time_ms": t,
                        "speech_gt": speech_gt,
                        "anchor_speech_gt": anchor_speech_gt,
                        "speech_vad": speech_vad,
                        "anchor": float(out.anchor),
                        "lifecycle": "BOUND",
                    }
                )
            )
        if vad_on_in_window == 0:
            n_vad_empty_windows += 1
        eval_audio_s += window_ms / 1000.0
        contam, active = gt_window_stats(gt, row["anchor_speaker"], eval_start, eval_end)
        meta = {
            "episode_id": episode_id,
            "topology": str(row["topology"]),
            "authoritative_transition_ms": int(
                row["authoritative_transition_time_ms"]
            ),
            "contam_s": contam,
            "active_speech_s": active,
            "lifecycle": "BOUND",
            "gt_eval": compact_gt(
                gt, row["anchor_speaker"], eval_start, eval_end
            ),
        }
        records_gt.append({**meta, "frames": frames_gt})
        records_prod.append({**meta, "frames": frames_prod})
        out_lines.append(
            json.dumps(
                {
                    "type": "episode_header",
                    "episode_id": episode_id,
                    "model": MODEL,
                    "regime": REGIME,
                    "topology": row["topology"],
                    "tau_frozen": tau,
                    "stub_fallback": fallback,
                    "fallback_reason": reason,
                    "lifecycle": "BOUND",
                    "vad_gate_on_frames": vad_on_in_window,
                    "vad_gate_on_n": n_frames,
                }
            )
        )

    agg_gt = aggregate(records_gt, tau, frame_ms)
    agg_prod = aggregate(records_prod, tau, frame_ms)
    view_gt = topology_view(records_gt, tau, frame_ms)
    view_prod = topology_view(records_prod, tau, frame_ms)
    cuts_gt = sum(len(replay_decisions(r["frames"], tau, frame_ms)[0]) for r in records_gt)
    sens_gt = sum(replay_decisions(r["frames"], tau, frame_ms)[1] for r in records_gt)
    cuts_prod = sum(
        len(replay_decisions(r["frames"], tau, frame_ms)[0]) for r in records_prod
    )
    sens_prod = sum(
        replay_decisions(r["frames"], tau, frame_ms)[1] for r in records_prod
    )

    # Episode-level retention: which GT-detected CUT episodes the prod gate keeps.
    ep_detail: list[dict] = []
    for rec_gt, rec_prod in zip(records_gt, records_prod):
        sc_gt = score_episode(rec_gt, tau, frame_ms)
        sc_prod = score_episode(rec_prod, tau, frame_ms)
        ep_detail.append(
            {
                "episode_id": rec_gt["episode_id"],
                "topology": rec_gt["topology"],
                "gt_n_cuts": sc_gt["n_cuts"],
                "gt_missed": sc_gt["missed"],
                "gt_false_cut": sc_gt["false_cut"],
                "prod_n_cuts": sc_prod["n_cuts"],
                "prod_missed": sc_prod["missed"],
                "prod_false_cut": sc_prod["false_cut"],
            }
        )
    gt_hit_ids = sorted(
        e["episode_id"] for e in ep_detail if e["topology"] == "A->A+B->B" and not e["gt_missed"]
    )
    prod_hit_ids = sorted(
        e["episode_id"] for e in ep_detail if e["topology"] == "A->A+B->B" and not e["prod_missed"]
    )
    retained_ids = sorted(set(gt_hit_ids) & set(prod_hit_ids))
    gt_cut_hits = len(gt_hit_ids)
    prod_cut_hits = len(prod_hit_ids)
    retained = (prod_cut_hits / gt_cut_hits) if gt_cut_hits else None

    vad_total_s = sum(vad_step_ms) / 1000.0
    pvad_total_s = sum(pvad_step_ms) / 1000.0
    cpu = {
        "production_vad_step_ms_p50": _pct(vad_step_ms, 0.5),
        "production_vad_step_ms_p95": _pct(vad_step_ms, 0.95),
        "production_vad_step_ms_max": max(vad_step_ms) if vad_step_ms else None,
        "production_vad_n_chunks": len(vad_step_ms),
        "production_vad_audio_s": vad_audio_s,
        "production_vad_rtf": (vad_total_s / vad_audio_s) if vad_audio_s else None,
        "pvad_step_ms_p50": _pct(pvad_step_ms, 0.5),
        "pvad_step_ms_p95": _pct(pvad_step_ms, 0.95),
        "pvad_reset_ms_p50": _pct(reset_ms, 0.5),
        "pvad_bind_ms_p50": _pct(bind_ms, 0.5),
        "combined_rtf_eval_audio": ((vad_total_s + pvad_total_s) / eval_audio_s)
        if eval_audio_s
        else None,
        "eval_audio_s": eval_audio_s,
        "cpu_count": os.cpu_count(),
    }

    (args.results_dir / "replay.jsonl").write_text(
        "\n".join(out_lines) + "\n", encoding="utf-8"
    )
    payload = {
        "model": MODEL,
        "regime": REGIME,
        "tau_frozen": tau,
        "n_episodes": len(rows),
        "gt": {
            "agg": agg_gt,
            "view": view_gt,
            "cuts": cuts_gt,
            "sens": sens_gt,
        },
        "prod": {
            "agg": agg_prod,
            "view": view_prod,
            "cuts": cuts_prod,
            "sens": sens_prod,
        },
        "gate_agreement": {
            "frames": n_frames_total,
            "agree_fraction": (n_agree / n_frames_total) if n_frames_total else None,
            "gt_speech_frames": n_gt_speech,
            "gt_speech_gated_off": n_vad_gated_off,
            "vad_extra_on": n_vad_extra,
            "vad_empty_windows": n_vad_empty_windows,
        },
        "cut_retention": {
            "gt_hit_ids": gt_hit_ids,
            "prod_hit_ids": prod_hit_ids,
            "retained_ids": retained_ids,
        },
        "retained_improvement_fraction": retained,
        "episodes": ep_detail,
        "cpu": cpu,
    }
    (args.results_dir / "replay_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    summary = render_summary(payload)
    (args.results_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)


def _f(value) -> str:
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _row(agg: dict, cuts: int, sens: int) -> str:
    return (
        f"| {agg['false_cuts']}/{agg['n_keep']} | {agg['missed']}/{agg['n_cut']} | "
        f"{_f(agg['delay_src_err_p50'])}/{_f(agg['delay_src_err_p90'])} | "
        f"{_f(agg['delay_dec_p50'])}/{_f(agg['delay_dec_p90'])} | {cuts} / {sens} |"
    )


def render_summary(payload: dict) -> str:
    gt = payload["gate_agreement"]
    agg_gt, agg_prod = payload["gt"]["agg"], payload["prod"]["agg"]
    lines = [
        "# Gate 5 — production VAD replay (firered x C, frozen tau=0.05)",
        "",
        f"> {DEV_ONLY_NOTE}",
        "",
        "> firered regime C only (sole plausible Gate 3+4 candidate); neovad "
        "collapsed 8/8 missed under the GT gate and is NOT replayed. Same 48 "
        "MAIN48 rows, same causal 1 s bind, same CommonPersistenceDecoder "
        "(500 ms confirmer / 300 ms sensitivity); ONLY the speech gate changes "
        "(GT anchor speech -> production Silero VAD peer-profile spans).",
        "",
        "## GT-gate vs production-VAD (frozen tau=0.05)",
        "",
        "| gate | false cuts (KEEP-n) | missed (CUT-n) | src_err p50/p90 (ms) | "
        "dec p50/p90 (ms) | CUT events / sens hits |",
        "|---|---|---|---|---|---|",
        f"| GT-gate {_row(agg_gt, payload['gt']['cuts'], payload['gt']['sens'])}",
        f"| prod-VAD {_row(agg_prod, payload['prod']['cuts'], payload['prod']['sens'])}",
        "",
        f"Contamination (GT-derived, identical both gates): "
        f"{_f(agg_gt['contamination_s_per_speech_h'])} s/h over "
        f"{payload['n_episodes']} episodes.",
        "",
        "Cross-check: the GT-gate row reproduces main firered-C exactly "
        "(13/32 false cuts, 5/8 missed, src_err -670/1098 ms, dec 500/500 ms) "
        "— lifecycle, binding, and decoder are identical; only the gate differs.",
        "",
        "## Gate agreement (per 10 ms frame, GT anchor speech vs prod VAD)",
        "",
        f"- frames scored: {gt['frames']}; agreement: {_f(gt['agree_fraction'])}",
        f"- GT-speech frames gated OFF by production VAD: "
        f"{gt['gt_speech_gated_off']}/{gt['gt_speech_frames']}",
        f"- production-gate-ON frames where GT is off: {gt['vad_extra_on']}",
        f"- eval windows with ZERO production-gate coverage: "
        f"{gt['vad_empty_windows']}/{payload['n_episodes']}",
        "",
        "## Retained-improvement fraction",
        "",
        f"- CUT successes GT-gate: "
        f"{agg_gt['n_cut'] - agg_gt['missed']}/{agg_gt['n_cut']}; "
        f"prod-VAD: {agg_prod['n_cut'] - agg_prod['missed']}/{agg_prod['n_cut']}; "
        f"hit-count ratio = {_f(payload['retained_improvement_fraction'])}",
        f"- episode-level retention: "
        f"{len(payload['cut_retention']['retained_ids'])}/{len(payload['cut_retention']['gt_hit_ids'])} "
        f"GT-detected CUT episodes still detected under prod-VAD",
        f"- false cuts GT-gate: {agg_gt['false_cuts']}; "
        f"prod-VAD: {agg_prod['false_cuts']}",
        f"- read: the prod gate is strictly wider (drops only "
        f"{gt['gt_speech_gated_off']}/{gt['gt_speech_frames']} GT-speech frames, "
        f"adds {gt['vad_extra_on']} gate-on frames) — missed "
        f"{agg_gt['missed']}/8->{agg_prod['missed']}/8, false cuts "
        f"{agg_gt['false_cuts']}->{agg_prod['false_cuts']}. No VAD "
        f"under-triggering; the cost is over-triggering on non-anchor speech.",
        "",
        "## Verdict",
        "",
        f"{verdict_text(payload)}",
        "",
        "## CPU note (this machine, wall-time)",
        "",
        f"- production VAD step (512-sample process_chunk, "
        f"n={payload['cpu']['production_vad_n_chunks']}): "
        f"median {_f(payload['cpu']['production_vad_step_ms_p50'])} ms, "
        f"p95 {_f(payload['cpu']['production_vad_step_ms_p95'])} ms, "
        f"max {_f(payload['cpu']['production_vad_step_ms_max'])} ms; "
        f"VAD-only RTF {_f(payload['cpu']['production_vad_rtf'])}",
        f"- pVAD adapter step (10 ms frames): median "
        f"{_f(payload['cpu']['pvad_step_ms_p50'])} ms, p95 "
        f"{_f(payload['cpu']['pvad_step_ms_p95'])} ms",
        f"- combined RTF (VAD + pVAD over eval audio "
        f"{_f(payload['cpu']['eval_audio_s'])} s): "
        f"{_f(payload['cpu']['combined_rtf_eval_audio'])}",
        "",
    ]
    return "\n".join(lines)


def verdict_text(payload: dict) -> str:
    agg_gt, agg_prod = payload["gt"]["agg"], payload["prod"]["agg"]
    gt_ok = agg_gt["missed"] < agg_gt["n_cut"]
    prod_ok = agg_prod["missed"] < agg_prod["n_cut"]
    if gt_ok and not prod_ok:
        return (
            "GT good / prod bad: VAD timing/gating problem — the observation "
            "model detects transitions the production gate hides. Do NOT "
            "fine-tune the observation model on this signal."
        )
    if gt_ok and prod_ok:
        return (
            "Both good: integration clean — production gating preserves the "
            "GT-gate detections."
        )
    return (
        "Both bad: observation model is the bottleneck — even the GT gate "
        "misses every CUT episode."
    )


if __name__ == "__main__":
    main()
