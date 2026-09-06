from __future__ import annotations

import unittest

from experiments.psem_state_corrected_adaptation_gate.frontier import (
    FrontierPoint,
    build_frontier_from_scores,
    reference_budget,
    select_envelopes,
    unique_thresholds,
)
from experiments.psem_state_corrected_adaptation_gate.gates import (
    CANDIDATE_FROZEN,
    EVAL_OPEN,
    FINAL_R_H_SC,
    H_CONFIRM,
    H_SCREEN,
    H_STOP,
    OPEN_T2,
    P0_FAIL,
    P0_PASS,
    GateError,
    GateTracker,
)


class FrontierTest(unittest.TestCase):
    def test_unique_thresholds_and_budget_envelopes(self):
        self.assertEqual(unique_thresholds([0.5, 0.9, 0.5, 0.2]), [0.9, 0.5, 0.2])
        f0 = FrontierPoint(0.5, 10.0, 0.20, 0.30)
        cands = [
            FrontierPoint(0.9, 12.0, 0.05, 0.05),
            FrontierPoint(0.5, 10.0, 0.20, 0.30),
            FrontierPoint(0.3, 9.0, 0.15, 0.25),
            FrontierPoint(0.2, 8.0, 0.15, 0.28),
        ]
        result = select_envelopes(f0, cands)
        self.assertEqual(result["c_envelope"].threshold, 0.3)
        self.assertEqual(result["m_envelope"].threshold, 0.3)
        self.assertTrue(result["useful"])
        over_budget = select_envelopes(f0, [FrontierPoint(0.9, 12.0, 0.01, 0.01)])
        self.assertFalse(over_budget["useful"])
        self.assertIsNone(over_budget["c_envelope"])
        trade = select_envelopes(
            f0,
            [FrontierPoint(0.3, 9.0, 0.10, 0.40), FrontierPoint(0.2, 8.0, 0.25, 0.20)],
        )
        self.assertFalse(trade["useful"])

    def test_tie_breaks_and_reference(self):
        f0 = FrontierPoint(0.5, 10.0, 0.20, 0.30)
        tied = [
            FrontierPoint(0.4, 9.0, 0.15, 0.25),
            FrontierPoint(0.3, 9.0, 0.15, 0.20),
        ]
        result = select_envelopes(f0, tied)
        self.assertEqual(result["c_envelope"].threshold, 0.3)
        self.assertEqual(result["m_envelope"].threshold, 0.3)
        points = [f0, FrontierPoint(0.3, 9.0, 0.15, 0.25)]
        self.assertEqual(reference_budget(points), f0)
        built = build_frontier_from_scores(
            [0.9, 0.5, 0.5, 0.2], lambda t: (1.0, 0.1, 0.2)
        )
        self.assertEqual(len(built), 3)


class GateSequencingTest(unittest.TestCase):
    def test_branch_gate_edges_and_illegal_jump(self):
        tracker = GateTracker()
        opened = tracker.advance("R-H-SC", P0_PASS)
        self.assertTrue(opened["edge"])
        repeated = tracker.advance("R-H-SC", P0_PASS)
        self.assertFalse(repeated["edge"])
        screened = tracker.advance("R-H-SC", H_SCREEN)
        self.assertTrue(screened["edge"])
        self.assertEqual(screened["prior"], P0_PASS)
        with self.assertRaises(GateError):
            tracker.advance("R-H-SC", EVAL_OPEN)
        tracker.advance("R-H-SC", H_CONFIRM)
        with self.assertRaises(GateError):
            tracker.advance("R-H-SC", OPEN_T2)
        tracker.advance("R-H-SC", H_STOP)
        with self.assertRaises(GateError):
            tracker.advance("other", H_SCREEN)
        failed = tracker.advance("other", P0_FAIL)
        self.assertTrue(failed["edge"])
        second = GateTracker()
        second.advance("R-H-SC", P0_PASS)
        second.advance("R-H-SC", H_SCREEN)
        second.advance("R-H-SC", OPEN_T2)
        self.assertEqual(second.current("R-H-SC"), OPEN_T2)
        _ = (CANDIDATE_FROZEN, FINAL_R_H_SC)


if __name__ == "__main__":
    unittest.main()
