from __future__ import annotations

import ast
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from puripuly_heart.app.adapters.ui_runtime import UiProviderRuntimeAdapter
from tests.helpers.ast_sources import call_name
from tests.helpers.paths import SOURCE_ROOT

UI_RUNTIME_PATH = SOURCE_ROOT / "app" / "adapters" / "ui_runtime.py"


@pytest.mark.asyncio
@pytest.mark.parametrize("succeeded", [True, False], ids=["committed", "rejected"])
async def test_ui_secret_change_waits_for_transaction_owner_before_rebinding(succeeded) -> None:
    events = []

    async def change_secret(key, value):
        events.append(("transaction", key, value))
        return succeeded

    rebind = Mock(side_effect=lambda key, value: events.append(("rebind", key, value)))
    settings = SimpleNamespace(change_secret=AsyncMock(side_effect=change_secret))
    adapter = UiProviderRuntimeAdapter(
        settings=SimpleNamespace(canonical=None),
        provider_application=SimpleNamespace(rebind_rolling_stt_secret=rebind),
        gpu=None,
        managed=None,
        credential_verification=None,
        provider_settings=settings,
        build_byok_target_settings=lambda _settings: None,
    )

    assert (
        await adapter.persist_provider_secret_change("soniox_api_key", "synthetic-secret")
        is succeeded
    )
    settings.change_secret.assert_awaited_once_with("soniox_api_key", "synthetic-secret")
    assert events == [
        (phase, "soniox_api_key", "synthetic-secret")
        for phase in (("transaction", "rebind") if succeeded else ("transaction",))
    ]


def test_ui_secret_adapter_does_not_own_transaction_serialization_or_lifetime() -> None:
    tree = ast.parse(UI_RUNTIME_PATH.read_text(encoding="utf-8"))
    names = (
        {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        | {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        | {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        }
    )
    assert {
        "_provider_secret_change_serialization_owner",
        "_persist_provider_secret_change_serialized",
        "_provider_secret_change_execution",
        "_apply_provider_secret_change_result",
        "LifecycleScope",
        "start_lifecycle_task",
    }.isdisjoint(names)
    assert "shield" not in {
        call_name(node) for node in ast.walk(tree) if isinstance(node, ast.Call)
    }
