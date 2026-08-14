from __future__ import annotations

import pytest
from puripuly_heart.app.services.provider_verification_binding import (
    ProviderVerificationBindingOwner,
)


def test_owner_builds_credential_redacted_binding_with_current_context() -> None:
    owner = ProviderVerificationBindingOwner(
        context_provider=lambda provider: (
            {"model": "gemini-current"} if provider == "google" else {}
        ),
    )
    raw_key = "raw-provider-secret"

    binding = owner.binding(
        "google",
        raw_key,
        flow="settings_api_key_verification",
        context_values={"origin": "settings"},
    )

    assert binding.provider == "google"
    assert binding.secret_key == "google_api_key"
    assert binding.secret_fingerprint.startswith("sha256:")
    assert binding.verifier_context["flow"] == "settings_api_key_verification"
    assert binding.verifier_context["model"] == "gemini-current"
    assert binding.verifier_context["origin"] == "settings"
    assert binding.verifier_evidence == {"source": "provider_verifier"}
    assert raw_key not in repr(binding)


@pytest.mark.parametrize(
    ("secret_key", "provider"),
    (
        ("google_api_key", "google"),
        ("openrouter_api_key", "openrouter"),
        ("deepseek_api_key", "deepseek"),
        ("cerebras_api_key", "cerebras"),
        ("alibaba_api_key_beijing", "alibaba_beijing"),
        ("alibaba_api_key_singapore", "alibaba_singapore"),
        ("deepgram_api_key", "deepgram"),
        ("soniox_api_key", "soniox"),
    ),
)
def test_owner_resolves_compatible_secret_keys(secret_key: str, provider: str) -> None:
    assert ProviderVerificationBindingOwner.provider_for_secret_key(secret_key) == provider


def test_owner_rejects_unsupported_provider_and_secret_key() -> None:
    owner = ProviderVerificationBindingOwner(
        context_provider=lambda _provider: {},
    )

    with pytest.raises(ValueError, match="unsupported provider verification binding"):
        owner.binding("unknown", "secret", flow="settings")
    with pytest.raises(ValueError, match="unsupported provider secret key"):
        owner.provider_for_secret_key("unknown_key")
