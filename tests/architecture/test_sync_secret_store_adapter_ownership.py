from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTROLLER_PATH = ROOT / "src" / "puripuly_heart" / "ui" / "controller.py"
UI_APPLICATION_COMPOSITION_PATH = (
    ROOT / "src" / "puripuly_heart" / "composition" / "ui_application.py"
)
MANAGED_AUTH_WIRING_PATH = (
    ROOT / "src" / "puripuly_heart" / "app" / "wiring_managed_auth_factory.py"
)


def test_controller_uses_explicit_sync_secret_store_adapter() -> None:
    source = CONTROLLER_PATH.read_text(encoding="utf-8")
    composition_source = UI_APPLICATION_COMPOSITION_PATH.read_text(encoding="utf-8")
    managed_auth_source = MANAGED_AUTH_WIRING_PATH.read_text(encoding="utf-8")

    assert "_ControllerSecretStorePortAdapter" not in source
    assert "puripuly_heart.app.adapters.sync_secret_store" not in source
    assert source.count("create_sync_secret_store_adapter(") == 2
    assert managed_auth_source.count("SyncSecretStoreAdapter(secret_store)") == 3
    assert composition_source.count("create_sync_secret_store_adapter(") == 1
    assert "provider_settings_owner=provider_settings_owner" in composition_source
