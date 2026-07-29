from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTROLLER_PATH = ROOT / "src" / "puripuly_heart" / "ui" / "controller.py"
PROVIDER_SETTINGS_PATH = (
    ROOT / "src" / "puripuly_heart" / "app" / "services" / "provider_settings.py"
)


def test_controller_composes_application_settings_patch_repository() -> None:
    source = CONTROLLER_PATH.read_text(encoding="utf-8")
    provider_settings_source = PROVIDER_SETTINGS_PATH.read_text(encoding="utf-8")

    assert "_ControllerSettingsPatchRepository" not in source
    assert "def _settings_snapshot_values" not in source
    assert source.count("self._legacy_settings_patch_repository(") == 5
    assert provider_settings_source.count("self.settings.create_legacy_patch_repository(") == 3
    assert "class ProviderApplicationOwner" in provider_settings_source
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
