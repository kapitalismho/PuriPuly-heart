#!/usr/bin/env python3
"""V1 evaluator-semantics unit tests (V2 speaker-change repair, no pytest).

Runnable via ``python experiments/psem_small_model_probe/cal/test_eval_semantics.py``.

Covers: (A) any-speech KEEP; (B) A->B B-only accumulation to a 500 ms CUT;
(C) silence reset; (D) premature-only CUT episode scores missed=true +
premature_cut=true; (E) decoder-dependent contamination differs across two
behaviors; plus the tolerance-boundary edge (transition-50 valid,
transition-51 premature).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.psem_small_model_probe.adapter.decoder import (  # noqa: E402
    CommonPersistenceDecoder,
)
from experiments.psem_small_model_probe.cal.eval_semantics import (  # noqa: E402
    CUT_TOLERANCE_MS,
    compact_gt,
    gt_anchor_speech,
    gt_any_speech,
    split_cuts,
)
from experiments.psem_small_model_probe.cal.metrics import score_episode  # noqa: E402

FRAME_MS = 100
TAU = 0.5


def make_gt(specs):
    """specs: [(start_ms, end_ms, speakers, masked)] -> gt index dict."""
    intervals = [
        {
            "start_sample": s * 16,
            "end_sample": e * 16,
            "active_speakers": list(sp),
            "masked": m,
        }
        for s, e, sp, m in specs
    ]
    intervals.sort(key=lambda iv: iv["start_sample"])
    return {
        "intervals": intervals,
        "starts": [iv["start_sample"] for iv in intervals],
    }


def make_frames(n, speech, anchors, eval_start=0):
    assert len(speech) == len(anchors) == n
    return [
        {
            "speech_gt": speech[i],
            "anchor_speech_gt": speech[i] and anchors[i] >= TAU,
            "anchor": anchors[i],
            "adapter_speech": True,
            "lifecycle": "BOUND",
            "source_time_ms": eval_start + (i + 1) * FRAME_MS,
        }
        for i in range(n)
    ]


def make_record(episode_id, topology, transition, frames, gt, anchor,
                eval_start=0, eval_end=None):
    eval_end = eval_start + len(frames) * FRAME_MS if eval_end is None else eval_end
    contam = 0.0
    active = sum(1 for f in frames if f["speech_gt"]) * FRAME_MS / 1000.0
    return {
        "episode_id": episode_id,
        "topology": topology,
        "authoritative_transition_ms": transition,
        "frames": frames,
        "contam_s": contam,
        "active_speech_s": active,
        "lifecycle": "BOUND",
        "gt_eval": compact_gt(gt, anchor, eval_start, eval_end),
    }


def check(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"{name}: {detail}")
    print(f"PASS {name}" + (f" — {detail}" if detail else ""))


def test_gate_separation():
    gt = make_gt([(0, 1000, ["A"], False), (1000, 2000, ["B"], False)])
    b_only = 1500 * 16
    check("gate-separation/any", gt_any_speech(gt, b_only) is True)
    check("gate-separation/anchor", gt_anchor_speech(gt, "A", b_only) is False)
    check("gate-separation/masked",
          gt_any_speech(make_gt([(0, 1000, ["A"], True)]), 500 * 16) is False)
    check("gate-separation/silence",
          gt_any_speech(make_gt([(0, 1000, [], False)]), 500 * 16) is False)


def test_a_speech_keep():
    gt = make_gt([(0, 1000, ["A"], False)])
    frames = make_frames(10, [True] * 10, [0.9] * 10)
    rec = make_record("keep-a", "A", 500, frames, gt, "A")
    out = score_episode(rec, TAU, FRAME_MS)
    check("keep-a/no-cuts", out["n_cuts"] == 0, f"n_cuts={out['n_cuts']}")
    check("keep-a/false-cut", out["false_cut"] is False)


def test_b_only_accumulates_cut():
    gt = make_gt([(0, 300, ["A"], False), (200, 1000, ["B"], False)])
    anchors = [0.9, 0.9] + [0.0] * 5 + [0.9] * 3
    frames = make_frames(10, [True] * 10, anchors)
    rec = make_record("cut-ab", "A->A+B->B", 200, frames, gt, "A")
    out = score_episode(rec, TAU, FRAME_MS)
    check("cut-ab/one-cut", out["n_cuts"] == 1, f"n_cuts={out['n_cuts']}")
    check("cut-ab/detected", out["missed"] is False)
    check("cut-ab/src-err", out["src_err_ms"] == 0, f"src_err={out['src_err_ms']}")
    check("cut-ab/dec-delay", out["dec_delay_ms"] == 500,
          f"dec_delay={out['dec_delay_ms']}")
    check("cut-ab/no-premature", out["premature_cut"] is False)


def test_silence_resets():
    gt = make_gt([(0, 1000, ["B"], False)])
    speech = [True] * 10
    speech[4] = False  # any-speech gap splits the 500 ms run 2+2
    anchors = [0.9, 0.9] + [0.0] * 5 + [0.9] * 3
    frames = make_frames(10, speech, anchors)
    rec = make_record("gap", "A->A+B->B", 200, frames, gt, "A")
    out = score_episode(rec, TAU, FRAME_MS)
    check("gap/no-cut", out["n_cuts"] == 0, f"n_cuts={out['n_cuts']}")
    check("gap/missed", out["missed"] is True)


def test_premature_only():
    gt = make_gt([(0, 6000, ["A"], False)])
    anchors = [0.0] * 5 + [0.9] * 15
    frames = make_frames(20, [True] * 20, anchors)
    rec = make_record("prem", "A->A+B->B", 5000, frames, gt, "A")
    out = score_episode(rec, TAU, FRAME_MS)
    check("premature/n-cuts", out["n_cuts"] == 1, f"n_cuts={out['n_cuts']}")
    check("premature/missed", out["missed"] is True)
    check("premature/flag", out["premature_cut"] is True
          and out["n_premature_cuts"] == 1,
          f"premature_cut={out['premature_cut']} n={out['n_premature_cuts']}")
    check("premature/no-delays", out["src_err_ms"] is None
          and out["dec_delay_ms"] is None and out["total_delay_ms"] is None)


def test_contamination_differs():
    gt = make_gt([(0, 1000, ["A"], False), (1000, 2000, ["B"], False)])
    cut_frames = make_frames(20, [True] * 20,
                             [0.9] * 10 + [0.0] * 5 + [0.9] * 5)
    keep_frames = make_frames(20, [True] * 20, [0.9] * 20)
    cut_rec = make_record("contam-cut", "A->A+B->B", 1000, cut_frames, gt, "A",
                          eval_end=2000)
    keep_rec = make_record("contam-nocut", "A->A+B->B", 1000, keep_frames, gt,
                           "A", eval_end=2000)
    cut_out = score_episode(cut_rec, TAU, FRAME_MS)
    keep_out = score_episode(keep_rec, TAU, FRAME_MS)
    check("contam/cut-detected", cut_out["missed"] is False)
    check("contam/cut-zero", cut_out["contam_s"] == 0.0,
          f"contam={cut_out['contam_s']}")
    check("contam/nocut-full", keep_out["contam_s"] == 1.0,
          f"contam={keep_out['contam_s']}")
    check("contam/differs", cut_out["contam_s"] != keep_out["contam_s"])


def test_tolerance_edge():
    cuts = [{"source_boundary_time": 950, "decision_time": 1450}]
    valid, premature = split_cuts(cuts, 1000)
    check("tolerance/edge-valid", len(valid) == 1 and not premature,
          f"CUT_TOLERANCE_MS={CUT_TOLERANCE_MS}")
    cuts = [{"source_boundary_time": 949, "decision_time": 1449}]
    valid, premature = split_cuts(cuts, 1000)
    check("tolerance/edge-premature", not valid and len(premature) == 1)


def test_decoder_uses_any_speech_gate():
    dec = CommonPersistenceDecoder(FRAME_MS)
    outs = [
        dec.update(
            {"speech_gt": True, "anchor": 0.0, "lifecycle": "BOUND",
             "source_time_ms": (i + 1) * FRAME_MS},
            tau=TAU,
        )
        for i in range(5)
    ]
    check("decoder/b-only-cut", outs[-1]["action"] == "CUT",
          f"last={outs[-1]['action']}")


def main():
    test_gate_separation()
    test_a_speech_keep()
    test_b_only_accumulates_cut()
    test_silence_resets()
    test_premature_only()
    test_contamination_differs()
    test_tolerance_edge()
    test_decoder_uses_any_speech_gate()
    print("ALL EVAL-SEMANTICS TESTS PASS")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"FAIL {exc}")
        sys.exit(1)
