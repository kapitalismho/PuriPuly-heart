from __future__ import annotations

import time
import unittest

from experiments.psem_state_corrected_adaptation_gate import calibrate as calibrate_mod
from experiments.psem_state_corrected_adaptation_gate.calibrate import (
    CalibrationError,
    fit_affine_calibrator,
    nll_loss,
)
from experiments.psem_state_corrected_adaptation_gate.multiplicity import (
    MultiplicityError,
    build_multiplicity,
    crop_frame_range,
)


def _reference_multiplicity(num_frames, crops, valid=None):
    if num_frames <= 0:
        raise MultiplicityError("num_frames must be positive")
    if valid is not None and len(valid) != num_frames:
        raise MultiplicityError("validity geometry differs from source frames")
    multiplicity = [0] * num_frames
    for crop_start_s, crop_end_s in crops:
        for frame in crop_frame_range(num_frames, crop_start_s, crop_end_s):
            if valid is None or valid[frame]:
                multiplicity[frame] += 1
    return multiplicity


def _reference_fit(z_raw, targets, role):
    if role != "TRAIN-CALIB":
        raise CalibrationError("calibration fits TRAIN-CALIB only")
    z_list = list(z_raw)
    y_list = [float(v) for v in targets]
    if len(z_list) != len(y_list) or not z_list:
        raise CalibrationError("logit/target geometry differs")
    if all(v == y_list[0] for v in y_list):
        raise CalibrationError("calibration needs positive and negative support")
    best = {"slope": 1.0, "intercept": 0.0, "nll": nll_loss(z_list, y_list)}
    for slope in (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0):
        for intercept in (-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0):
            calibrated = [slope * z + intercept for z in z_list]
            nll = nll_loss(calibrated, y_list)
            if nll < float(best["nll"]):
                best = {"slope": slope, "intercept": intercept, "nll": nll}
    calibrated = calibrate_mod.apply_affine(
        z_list, float(best["slope"]), float(best["intercept"])
    )
    best["brier"] = calibrate_mod.brier_score(calibrated, y_list)
    best["raw_nll"] = nll_loss(z_list, y_list)
    best["raw_brier"] = calibrate_mod.brier_score(z_list, y_list)
    best["role"] = role
    return best


class MultiplicityParityTest(unittest.TestCase):
    def test_matches_reference_with_mask_and_zero_regions(self):
        valid = [True] * 40
        crops = [(0.0, 3.0), (0.5, 5.0), (4.0, 8.0), (10.0, 20.0)]
        expected = _reference_multiplicity(40, crops, valid)
        self.assertEqual(build_multiplicity(40, crops, valid), expected)
        self.assertEqual(build_multiplicity(40, crops, None), _reference_multiplicity(40, crops))
        self.assertEqual(build_multiplicity(40, [], valid), [0] * 40)

    def test_error_contracts(self):
        with self.assertRaises(MultiplicityError):
            build_multiplicity(0, [])
        with self.assertRaises(MultiplicityError):
            build_multiplicity(4, [(0.0, 5.0)], [True] * 3)
        with self.assertRaises(MultiplicityError):
            build_multiplicity(100, [(1.0, 2.0)])

    def test_bounded_for_dense_overlapping_crops(self):
        num_frames = 400000
        crops = [(0.0, 30000.0)] * 400
        start = time.perf_counter()
        result = build_multiplicity(num_frames, crops)
        self.assertLess(time.perf_counter() - start, 5.0)
        self.assertEqual(result[300000], 400)
        self.assertEqual(result[375000], 0)


class CalibrationParityTest(unittest.TestCase):
    def test_exact_reference_equality(self):
        z_raw = [-2.0, -1.0, 0.0, 1.0, 2.0, -1.5, 0.5, 1.5, -80.0, 80.0]
        targets = [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
        expected = _reference_fit(z_raw, targets, "TRAIN-CALIB")
        result = fit_affine_calibrator(z_raw, targets, "TRAIN-CALIB")
        self.assertEqual(result, expected)
        for key in ("nll", "brier", "raw_nll", "raw_brier"):
            self.assertEqual(repr(result[key]), repr(expected[key]))

    def test_error_contracts(self):
        with self.assertRaises(CalibrationError):
            fit_affine_calibrator([0.0, 1.0], [0.0, 1.0], "DEV")
        with self.assertRaises(CalibrationError):
            fit_affine_calibrator([0.0, 1.0], [1.0, 1.0], "TRAIN-CALIB")
        with self.assertRaises(CalibrationError):
            fit_affine_calibrator([], [], "TRAIN-CALIB")

    def test_bounded_single_data_pass(self):
        n = 20000
        z_raw = [(i % 80 - 40) / 10.0 for i in range(n)]
        targets = [1.0 if z > 0 else 0.0 for z in z_raw]
        start = time.perf_counter()
        result = fit_affine_calibrator(z_raw, targets, "TRAIN-CALIB")
        self.assertLess(time.perf_counter() - start, 10.0)
        self.assertLessEqual(float(result["nll"]), float(result["raw_nll"]))


class SpoolShaTest(unittest.TestCase):
    def test_spool_sha_matches_file_bytes(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace

        from experiments.psem_state_corrected_adaptation_gate.lifecycle import AnchorEpisode
        from experiments.psem_state_corrected_adaptation_gate.stages import (
            sha256_file,
            write_spooled_target,
        )

        entry = {
            "authority": SimpleNamespace(
                num_frames=4,
                episodes=(AnchorEpisode("ep1", "spkA", 0, 4),),
                y_anchor=tuple([1.0] * 4),
                y_replace=tuple([0.0] * 4),
                valid=tuple([True] * 4),
                ledger={"opportunities": []},
            ),
            "multiplicity": [1] * 4,
            "episode_ids": ["ep1"] * 4,
            "intervals": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = SimpleNamespace(audio_ref="b.wav", waveform_sha256="0" * 64)
            record = write_spooled_target(root, "b", entry, session)
            self.assertEqual(record["sha256"], sha256_file(root / "b.json"))
            self.assertEqual(
                set(record), {"file", "sha256", "num_frames", "audio_ref", "waveform_sha256"}
            )


if __name__ == "__main__":
    unittest.main()
