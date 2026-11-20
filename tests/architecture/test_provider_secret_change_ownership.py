from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from puripuly_heart.app.adapters.ui_runtime import UiProviderRuntimeAdapter


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
