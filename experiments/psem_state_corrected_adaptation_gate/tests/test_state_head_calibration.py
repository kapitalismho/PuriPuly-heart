from __future__ import annotations

import unittest

from experiments.psem_state_corrected_adaptation_gate import head, streaming
from experiments.psem_state_corrected_adaptation_gate.calibrate import (
    CalibrationError,
    apply_affine,
    brier_score,
    fit_affine_calibrator,
    nll_loss,
)


class StreamingTest(unittest.TestCase):
    def test_chunk_carry_detach_and_boundaries(self):
        bounds = streaming.chunk_boundaries(800)
        self.assertEqual(bounds[0], (0, 375))
        self.assertEqual(bounds[-1][1], 800)
        carrier = streaming.StateCarrier([1.0, 2.0], "src-a")
        carried = carrier.carry()
        self.assertEqual(carried, [1.0, 2.0])
        carrier.detach()
        self.assertEqual(carrier.detached_steps, 1)
        carrier.reset([0.0], "src-b")
        self.assertEqual(carrier.source_id, "src-b")
        self.assertEqual(carrier.detached_steps, 0)

    def test_equivalence_pass_and_fail(self):
        ok = streaming.chunk_equivalence([1.0, 2.0], [1.0, 2.0], 1e-9)
        self.assertTrue(ok["passed"])
        bad = streaming.chunk_equivalence([1.0, 2.0], [1.0, 2.5], 1e-9)
        self.assertFalse(bad["passed"])


class ResidualIdentityTest(unittest.TestCase):
    def test_zero_residual_is_f0(self):
        report = head.check_zero_residual_identity([0.99, 0.7, 0.5, 0.3, 0.01])
        self.assertTrue(report["passed"])
        for posterior in (0.9, 0.5, 0.2):
            self.assertAlmostEqual(head.product_logit(posterior, 0.0), head.f0_logit(posterior))
        self.assertNotAlmostEqual(head.product_logit(0.5, 1.0), head.f0_logit(0.5))

    def test_selective_modes_and_update_audit(self):
        receipt = head.selective_mode_receipt(True, True)
        self.assertTrue(receipt["frozen_representation_ok"])
        self.assertFalse(head.selective_mode_receipt(False, True)["frozen_representation_ok"])
        audit = head.audit_update(
            [[1.0, 2.0]], [[1.0, 2.0]], [[0.1, -0.2]], [[0.5, 0.5]], [[0.6, 0.4]]
        )
        self.assertTrue(audit["passed"])
        frozen_moved = head.audit_update(
            [[1.0]], [[1.1]], [[0.1]], [[0.5]], [[0.6]]
        )
        self.assertFalse(frozen_moved["passed"])


class CalibrationTest(unittest.TestCase):
    def test_positive_slope_and_train_calib_only(self):
        z_raw = [-2.0, -1.0, 0.0, 1.0, 2.0, -1.5, 0.5, 1.5]
        targets = [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
        result = fit_affine_calibrator(z_raw, targets, "TRAIN-CALIB")
        self.assertGreater(float(result["slope"]), 0)
        self.assertLessEqual(float(result["nll"]), float(result["raw_nll"]))
        with self.assertRaises(CalibrationError):
            fit_affine_calibrator(z_raw, targets, "DEV")
        with self.assertRaises(CalibrationError):
            apply_affine(z_raw, 0.0, 0.0)
        self.assertGreaterEqual(nll_loss(z_raw, targets), 0)
        self.assertGreaterEqual(brier_score(z_raw, targets), 0)


if __name__ == "__main__":
    unittest.main()
