from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMPOSITION_PATH = ROOT / "src" / "puripuly_heart" / "composition" / "application_runtime.py"
PROVIDER_RUNTIME_PATH = (
    ROOT / "src" / "puripuly_heart" / "app" / "wiring" / "wiring_provider_runtime.py"
)


def test_provider_runtime_owns_gpu_recovery_and_composition_has_no_algorithm() -> None:
    source = COMPOSITION_PATH.read_text(encoding="utf-8")
    provider_runtime = PROVIDER_RUNTIME_PATH.read_text(encoding="utf-8")

    assert source.count("create_gpu_provider_recovery_application_owner(") == 1
    assert "await require_gpu_recovery().recover(" in source
    assert "lambda: gpu_recovery_request(" in source
    assert "recover_gpu=effects.gpu_recovery" in provider_runtime
    for retired_name in (
        "_gpu_provider_recovery_lock",
        "_get_gpu_provider_recovery_lock",
        "_apply_gpu_runtime_owner_recovery_locked",
        "_execute_gpu_provider_recovery_retry",
        "_build_gpu_recovery_request",
        "_abort_provider_recoveries",
        "_resume_gpu_provider_consumers",
        "_gpu_provider_recovery_execution",
        "_complete_gpu_provider_recovery",
        "_desired_gpu_channels",
        "_gpu_provider_recovery_channel_plans",
    ):
        assert retired_name not in source
