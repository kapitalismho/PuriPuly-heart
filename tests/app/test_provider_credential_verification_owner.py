from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from puripuly_heart.app.services.provider_credential_verification import (
    PROVIDER_CREDENTIAL_EMPTY,
    PROVIDER_CREDENTIAL_ERROR,
    PROVIDER_CREDENTIAL_FAILED,
    PROVIDER_CREDENTIAL_MODEL_UNAVAILABLE,
    PROVIDER_CREDENTIAL_UNKNOWN,
    PROVIDER_CREDENTIAL_VERIFIED,
    ProviderCredentialVerificationInteractionOwner,
    ProviderCredentialVerificationOwner,
    ProviderCredentialVerificationRequest,
)
from puripuly_heart.app.wiring_composition import (
    create_provider_credential_verification_interaction_owner,
)


@dataclass
class RecordingVerifier:
    outcomes: dict[tuple[str, str | None], bool] = field(default_factory=dict)
    calls: list[tuple[str, str, str | None, str | None, bool]] = field(default_factory=list)
    failure: Exception | None = None

    async def verify_api_key(
        self,
        provider: str,
        api_key: str,
        *,
        model: str | None = None,
        base_url: str | None = None,
        low_latency: bool = False,
    ) -> bool:
        self.calls.append((provider, api_key, model, base_url, low_latency))
        if self.failure is not None:
            raise self.failure
        return self.outcomes.get((provider, model), False)

    async def verify_qwen_llm_api_key(
        self,
        api_key: str,
        *,
        base_url: str,
        model: str | None,
        low_latency: bool,
    ) -> bool:
        self.calls.append(("qwen", api_key, model, base_url, low_latency))
        if self.failure is not None:
            raise self.failure
        return self.outcomes.get(("qwen", model), False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "selected_model"),
    (
        ("google", "gemini-model"),
        ("cerebras", "cerebras-model"),
        ("openrouter", None),
        ("deepseek", None),
        ("deepgram", None),
        ("soniox", None),
    ),
)
async def test_owner_routes_direct_provider_verification(
    provider: str,
    selected_model: str | None,
) -> None:
    verifier = RecordingVerifier(outcomes={(provider, selected_model): True})
    owner = ProviderCredentialVerificationOwner(verifier=verifier)

    outcome = await owner.verify(
        ProviderCredentialVerificationRequest(
            provider=provider,
            api_key="secret",
            selected_model=selected_model,
        )
    )

    assert outcome.status == PROVIDER_CREDENTIAL_VERIFIED
    assert verifier.calls == [(provider, "secret", selected_model, None, False)]


@pytest.mark.asyncio
async def test_owner_reports_selected_qwen_model_unavailable_when_fallback_works() -> None:
    verifier = RecordingVerifier(outcomes={("qwen", "fallback"): True})
    owner = ProviderCredentialVerificationOwner(verifier=verifier)

    outcome = await owner.verify(
        ProviderCredentialVerificationRequest(
            provider="alibaba_singapore",
            api_key="secret",
            selected_model="selected",
            fallback_models=("selected", "fallback", "unused"),
            low_latency=True,
        )
    )

    assert outcome.status == PROVIDER_CREDENTIAL_MODEL_UNAVAILABLE
    assert outcome.unavailable_model == "selected"
    assert verifier.calls == [
        (
            "qwen",
            "secret",
            "selected",
            "https://dashscope-intl.aliyuncs.com/api/v1",
            True,
        ),
        (
            "qwen",
            "secret",
            "fallback",
            "https://dashscope-intl.aliyuncs.com/api/v1",
            True,
        ),
    ]


@pytest.mark.asyncio
async def test_owner_preserves_qwen_success_and_failure_outcomes() -> None:
    verifier = RecordingVerifier(outcomes={("qwen", "selected"): True})
    owner = ProviderCredentialVerificationOwner(verifier=verifier)

    verified = await owner.verify(
        ProviderCredentialVerificationRequest(
            provider="alibaba_beijing",
            api_key="secret",
            selected_model="selected",
            fallback_models=("fallback",),
            low_latency=True,
        )
    )
    missing_model = await owner.verify(
        ProviderCredentialVerificationRequest(
            provider="alibaba_beijing",
            api_key="secret",
        )
    )

    assert verified.status == PROVIDER_CREDENTIAL_VERIFIED
    assert missing_model.status == PROVIDER_CREDENTIAL_FAILED


@pytest.mark.asyncio
async def test_owner_rejects_empty_and_unknown_credentials_without_verifier_calls() -> None:
    verifier = RecordingVerifier()
    owner = ProviderCredentialVerificationOwner(verifier=verifier)

    empty = await owner.verify(
        ProviderCredentialVerificationRequest(
            provider="google",
            api_key="",
        )
    )
    unknown = await owner.verify(
        ProviderCredentialVerificationRequest(
            provider="unknown",
            api_key="secret",
        )
    )

    assert empty.status == PROVIDER_CREDENTIAL_EMPTY
    assert unknown.status == PROVIDER_CREDENTIAL_UNKNOWN
    assert verifier.calls == []


@pytest.mark.asyncio
async def test_owner_contains_verifier_failure_and_emits_safe_diagnostics() -> None:
    verifier = RecordingVerifier(failure=RuntimeError("private provider detail"))
    diagnostics: list[tuple[str, dict[str, object], BaseException | None]] = []
    owner = ProviderCredentialVerificationOwner(
        verifier=verifier,
        diagnostics_sink=lambda event, metadata, exception: diagnostics.append(
            (event, dict(metadata), exception)
        ),
    )

    outcome = await owner.verify(
        ProviderCredentialVerificationRequest(
            provider="google",
            api_key="secret",
            selected_model="gemini-model",
        )
    )

    assert outcome.status == PROVIDER_CREDENTIAL_ERROR
    assert outcome.error_text == "private provider detail"
    assert diagnostics[0][0] == "provider_credential_verification_failed"
    assert diagnostics[0][1] == {
        "provider": "google",
        "error_type": "RuntimeError",
    }
    assert isinstance(diagnostics[0][2], RuntimeError)
    assert "private provider detail" not in str(diagnostics[0][1])


@pytest.mark.asyncio
async def test_interaction_owner_resolves_current_model_and_maps_qwen_unavailable() -> None:
    selected = ["selected"]
    verifier = RecordingVerifier(outcomes={("qwen", "fallback"): True})
    owner = ProviderCredentialVerificationInteractionOwner(
        verification_owner=ProviderCredentialVerificationOwner(verifier=verifier),
        selected_model_provider=lambda _provider: selected[0],
        fallback_models=("selected", "fallback"),
        low_latency=True,
    )

    first = await owner.verify("alibaba_beijing", "secret")
    selected[0] = "fallback"
    second = await owner.verify("alibaba_beijing", "secret")

    assert first == (False, "qwen_model_unavailable:selected")
    assert second == (True, "Verification successful")


@pytest.mark.asyncio
async def test_interaction_owner_maps_empty_unknown_failed_and_error_results() -> None:
    error_calls: list[tuple[str, str]] = []
    verifier = RecordingVerifier()
    owner = ProviderCredentialVerificationInteractionOwner(
        verification_owner=ProviderCredentialVerificationOwner(verifier=verifier),
        selected_model_provider=lambda _provider: None,
        error_sink=lambda provider, error: error_calls.append((provider, error)),
    )

    assert await owner.verify("google", "") == (False, "API Key is empty")
    assert await owner.verify("unknown", "secret") == (
        False,
        "Unknown provider: unknown",
    )
    assert await owner.verify("deepseek", "secret") == (
        False,
        "Verification failed (check logs/console for details)",
    )
    verifier.failure = RuntimeError("provider failure")
    assert await owner.verify("deepseek", "secret") == (False, "provider failure")
    assert error_calls == [("deepseek", "provider failure")]


def test_interaction_owner_factory_composes_verifier_and_callbacks() -> None:
    verifier = RecordingVerifier()
    owner = create_provider_credential_verification_interaction_owner(
        verifier=verifier,
        selected_model_provider=lambda provider: f"{provider}-model",
        fallback_models=("fallback",),
        low_latency=True,
    )

    assert isinstance(owner, ProviderCredentialVerificationInteractionOwner)
    assert owner.verification_owner.verifier is verifier
    assert owner.selected_model_provider("google") == "google-model"
    assert owner.fallback_models == ("fallback",)
    assert owner.low_latency is True
