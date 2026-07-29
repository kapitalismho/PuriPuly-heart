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
    assert source.count("self._legacy_settings_patch_repository(") == 7
    assert "create_legacy_patch_repository(" in source
    assert "persist_settings=self._persist_settings_at_controller_boundary" not in source
    assert "begin_canonical_mutation=" not in source
    assert "_persist_settings_at_controller_boundary" not in source
    assert "_update_canonical_settings_from_legacy_delta" not in source
    assert "_canonical_vnext_after_legacy_delta" not in source
    assert "_remember_canonical_legacy_projection" not in source
    assert "_begin_canonical_mutation" not in source
    assert "_rollback_canonical_mutation" not in source
    assert "_complete_canonical_mutation" not in source


def test_settings_patch_repository_has_no_ui_or_controller_dependency() -> None:
    source = REPOSITORY_PATH.read_text(encoding="utf-8")

    assert "puripuly_heart.ui" not in source
    assert "GuiController" not in source
    assert "asyncio.to_thread(self.owner.persist)" in source
    assert "self.owner.rollback()" in source
