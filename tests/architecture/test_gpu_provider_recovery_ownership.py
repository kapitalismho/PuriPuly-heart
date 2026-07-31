from __future__ import annotations

from tests.helpers.paths import SOURCE_ROOT

PROVIDER_RUNTIME_PATH = SOURCE_ROOT / "app" / "wiring" / "wiring_provider_runtime.py"


def test_provider_runtime_wires_gpu_recovery_effect() -> None:
    provider_runtime = PROVIDER_RUNTIME_PATH.read_text(encoding="utf-8")
    assert "recover_gpu=effects.gpu_recovery" in provider_runtime
