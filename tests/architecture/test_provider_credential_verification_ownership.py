from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from puripuly_heart.app.adapters.ui_runtime import UiProviderRuntimeAdapter
from puripuly_heart.app.ports.provider_verifier import ProviderCredentialVerificationOutcome


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome",
    [
        ProviderCredentialVerificationOutcome(status="verified", provider="openai"),
        ProviderCredentialVerificationOutcome(
            status="error", provider="openai", error_text="provider.failure"
        ),
    ],
)
async def test_ui_interactive_verification_returns_the_owner_result(outcome) -> None:
    verification = SimpleNamespace(verify=AsyncMock(return_value=outcome))
    adapter = UiProviderRuntimeAdapter(
        settings=None,
        provider_application=None,
        gpu=None,
        managed=None,
        credential_verification=verification,
        provider_settings=None,
        build_byok_target_settings=lambda _settings: None,
    )

    assert await adapter.verify_api_key("openai", "synthetic-secret") is outcome
    verification.verify.assert_awaited_once_with("openai", "synthetic-secret")
