from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTROLLER_PATH = ROOT / "src" / "puripuly_heart" / "ui" / "controller.py"
ADAPTER_PATH = ROOT / "src" / "puripuly_heart" / "app" / "adapters" / "sync_secret_store.py"


def test_controller_uses_explicit_sync_secret_store_adapter() -> None:
    source = CONTROLLER_PATH.read_text(encoding="utf-8")

    assert "_ControllerSecretStorePortAdapter" not in source
    assert "puripuly_heart.app.adapters.sync_secret_store" not in source
    assert source.count("create_sync_secret_store_adapter(secret_store)") == 5


def test_sync_secret_store_adapter_has_no_ui_or_controller_dependency() -> None:
    source = ADAPTER_PATH.read_text(encoding="utf-8")

    assert "puripuly_heart.ui" not in source
    assert "GuiController" not in source
    assert "asyncio.to_thread" in source
