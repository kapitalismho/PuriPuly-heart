from __future__ import annotations

import pytest

from puripuly_heart.app.services.provider.provider_verification_binding import (
    ProviderVerificationBindingOwner,
)


def test_owner_builds_credential_redacted_binding_with_current_context() -> None:
    owner = ProviderVerificationBindingOwner(
        context_provider=lambda provider: (
            {"model": "gemini-current"} if provider == "gemini" else {}
        ),
    )
    raw_key = "raw-provider-secret"

    binding = owner.binding(
        "gemini",
        raw_key,
        flow="settings_api_key_verification",
        context_values={"origin": "settings"},
    )

    assert binding.provider == "gemini"
    assert binding.secret_key == "gemini_api_key"
    assert binding.secret_fingerprint.startswith("sha256:")
    assert binding.verifier_context["flow"] == "settings_api_key_verification"
    assert binding.verifier_context["model"] == "gemini-current"
    assert binding.verifier_context["origin"] == "settings"
    assert binding.verifier_evidence == {"source": "provider_verifier"}
    assert raw_key not in repr(binding)


def test_owner_rejects_unsupported_provider_and_secret_key() -> None:
    owner = ProviderVerificationBindingOwner(
        context_provider=lambda _provider: {},
    )

    with pytest.raises(ValueError, match="unsupported provider verification binding"):
        owner.binding("unknown", "secret", flow="settings")
    with pytest.raises(ValueError, match="unsupported provider secret key"):
        owner.provider_for_secret_key("unknown_key")
