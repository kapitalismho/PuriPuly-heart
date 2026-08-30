import json

import pytest

from experiments.psem_sortformer_adaptation_depth.execution import (
    ExecutionError,
    build_cost_receipt,
    run_overfit_arm_result,
)
from experiments.psem_sortformer_adaptation_depth.receipts import validate_cost_receipt
from experiments.psem_sortformer_adaptation_depth.run import main


def test_cost_receipt_uses_exact_arithmetic_and_allows_the_hard_stop_boundary() -> None:
    receipt = build_cost_receipt(
        hourly_price_usd=36.0,
        hourly_price_source="trusted operator quote",
        actual_gpu_seconds=1800.0,
        projected_remaining_gpu_seconds=1200.0,
        command="train-arm H-HEAD",
    )
    assert receipt["actual_cost_usd"] == 18.0
    assert receipt["projected_remaining_cost_usd"] == 12.0
    assert receipt["projected_total_cost_usd"] == 30.0
    assert receipt["target_is_informational"] is True
    assert validate_cost_receipt(receipt) == receipt


def test_cost_receipt_rejects_a_projection_above_usd_30() -> None:
    with pytest.raises(ExecutionError, match="USD-30 hard stop"):
        build_cost_receipt(
            hourly_price_usd=36.0,
            hourly_price_source="trusted operator quote",
            actual_gpu_seconds=1800.0,
            projected_remaining_gpu_seconds=1201.0,
            command="train-arm H-HEAD",
        )


def test_cost_receipt_cli_preserves_the_executed_command(tmp_path) -> None:
    output = tmp_path / "cost.json"
    assert (
        main(
            [
                "cost-receipt",
                "--hourly-price-usd",
                "1",
                "--hourly-price-source",
                "operator quote",
                "--actual-gpu-seconds",
                "3600",
                "--projected-remaining-gpu-seconds",
                "3600",
                "--command",
                "train-arm H-HEAD",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text(encoding="utf-8"))["command"] == "train-arm H-HEAD"


def test_legacy_overfit_runner_is_a_fail_closed_tombstone() -> None:
    with pytest.raises(ExecutionError, match="legacy overfit arm is not supported; use smoke-arm"):
        run_overfit_arm_result()
