from pathlib import Path

import pytest

from experiments.psem_sortformer_adaptation_depth.execution import (
    run_canary_arm,
    run_overfit_arm_result,
    run_training_arm,
)
from experiments.psem_sortformer_adaptation_depth.preflight import PreflightError
from experiments.psem_sortformer_adaptation_depth.runtime_audit import run_gradient_update_canary
from experiments.psem_sortformer_adaptation_depth.training import fit_arm, run_overfit_arm

ABSENT = Path("absent")


@pytest.mark.parametrize(
    ("runner", "kwargs"),
    (
        (
            run_canary_arm,
            {
                "checkpoint_path": ABSENT,
                "nemo_checkout": ABSENT,
                "dependency_lock": ABSENT,
                "corpus_root": ABSENT,
                "reference_root": ABSENT,
                "sampling_manifest": ABSENT,
                "arm": "H-HEAD",
                "device": "cuda",
            },
        ),
        (
            run_overfit_arm_result,
            {
                "checkpoint_path": ABSENT,
                "nemo_checkout": ABSENT,
                "dependency_lock": ABSENT,
                "corpus_root": ABSENT,
                "reference_root": ABSENT,
                "sampling_manifest": ABSENT,
                "class_weight_receipt": {},
                "arm": "H-HEAD",
                "device": "cuda",
            },
        ),
        (
            run_training_arm,
            {
                "checkpoint_path": ABSENT,
                "nemo_checkout": ABSENT,
                "dependency_lock": ABSENT,
                "corpus_root": ABSENT,
                "reference_root": ABSENT,
                "sampling_manifest": ABSENT,
                "class_weight_receipt": {},
                "material_gate": {},
                "output_root": ABSENT,
                "device": "cuda",
            },
        ),
    ),
)
def test_legacy_material_routes_fail_before_touching_runtime_inputs(runner, kwargs) -> None:
    with pytest.raises(PreflightError, match="blocked_pending_lean_runner_alignment"):
        runner(**kwargs)


@pytest.mark.parametrize(
    ("runner", "args", "kwargs"),
    (
        (run_gradient_update_canary, (None, "H-HEAD", None), {}),
        (run_overfit_arm, (None, "H-HEAD", (), None), {"authorization": None}),
        (
            fit_arm,
            (None, "H-HEAD", None, None, 0, None, None),
            {"authorization": None},
        ),
    ),
)
def test_lower_level_material_apis_fail_before_any_work(runner, args, kwargs) -> None:
    with pytest.raises(PreflightError, match="blocked_pending_lean_runner_alignment"):
        runner(*args, **kwargs)
