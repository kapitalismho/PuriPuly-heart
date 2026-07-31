from __future__ import annotations

from tests.helpers.ast_sources import method_source_unscoped as _method_source
from tests.helpers.paths import SOURCE_ROOT

UI_RUNTIME_PATH = SOURCE_ROOT / "app" / "adapters" / "ui_runtime.py"
INTERACTION_PATH = SOURCE_ROOT / "app" / "services" / "gpu_runtime_interaction.py"
COMPOSITION_PATH = SOURCE_ROOT / "app" / "wiring" / "wiring_composition.py"


def test_ui_gpu_provisioning_command_delegates_to_composed_interaction_owner() -> None:
    selected = _method_source(UI_RUNTIME_PATH, "install_selected_gpu_model_if_needed")
    composition = COMPOSITION_PATH.read_text(encoding="utf-8")

    assert "self.gpu.install_selected_model_if_needed()" in selected
    assert "LocalASRInstallRequest" not in selected
    assert composition.count("create_gpu_runtime_interaction_owner(") == 1
    assert "LocalASRGpuProvisioningOwner(" not in composition
    assert "LocalASRGpuProvisioningOwner(" in INTERACTION_PATH.read_text(encoding="utf-8")
