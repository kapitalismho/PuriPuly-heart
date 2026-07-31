from __future__ import annotations

from tests.helpers.ast_sources import method_source
from tests.helpers.paths import REPO_ROOT as ROOT

UI_RUNTIME_PATH = ROOT / "src" / "puripuly_heart" / "app" / "adapters" / "ui_runtime.py"


def _adapter_method_source(method_name: str) -> str:
    return method_source(UI_RUNTIME_PATH, "UiProviderRuntimeAdapter", method_name)


def test_ui_delegates_complete_provider_secret_transaction_owner() -> None:
    source = UI_RUNTIME_PATH.read_text(encoding="utf-8")
    method = _adapter_method_source("persist_provider_secret_change")

    assert "_provider_secret_change_serialization_owner" not in source
    assert "_persist_provider_secret_change_serialized" not in source
    assert "self.provider_settings.change_secret(key, value)" in method
    assert "_provider_secret_change_execution" not in source
    assert "_apply_provider_secret_change_result" not in source
    assert "LifecycleScope" not in source
    assert "start_lifecycle_task" not in source
    assert "asyncio.shield" not in source
