from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTROLLER_PATH = ROOT / "src" / "puripuly_heart" / "ui" / "controller.py"
REPOSITORY_PATH = (
    ROOT / "src" / "puripuly_heart" / "app" / "services" / "canonical_settings_persistence.py"
)


def test_controller_composes_application_settings_patch_repository() -> None:
    source = CONTROLLER_PATH.read_text(encoding="utf-8")

    assert "_ControllerSettingsPatchRepository" not in source
    assert "def _settings_snapshot_values" not in source
    assert source.count("self._legacy_settings_patch_repository(") == 8
    assert "create_legacy_patch_repository(" in source
    assert "persist_settings=self._persist_settings_at_controller_boundary" in source


def test_settings_patch_repository_has_no_ui_or_controller_dependency() -> None:
    source = REPOSITORY_PATH.read_text(encoding="utf-8")

    assert "puripuly_heart.ui" not in source
    assert "GuiController" not in source
    assert "asyncio.to_thread(self.callbacks.persist_settings" in source
    assert "self.callbacks.rollback_canonical_mutation()" in source
