from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Final, Literal

from puripuly_heart.app.ports.provider_verifier import ProviderVerifierPort

ProviderCredentialVerificationStatus = Literal[
    "verified",
    "failed",
    "empty",
    "unknown",
    "model_unavailable",
    "error",
]
ProviderCredentialVerificationDiagnosticsSink = Callable[
    [str, Mapping[str, object], BaseException | None],
    None,
]
ProviderCredentialSelectedModelProvider = Callable[[str], str | None]
ProviderCredentialVerificationErrorSink = Callable[[str, str], None]

PROVIDER_CREDENTIAL_VERIFIED: Final[ProviderCredentialVerificationStatus] = "verified"
PROVIDER_CREDENTIAL_FAILED: Final[ProviderCredentialVerificationStatus] = "failed"
PROVIDER_CREDENTIAL_EMPTY: Final[ProviderCredentialVerificationStatus] = "empty"
PROVIDER_CREDENTIAL_UNKNOWN: Final[ProviderCredentialVerificationStatus] = "unknown"
PROVIDER_CREDENTIAL_MODEL_UNAVAILABLE: Final[ProviderCredentialVerificationStatus] = (
    "model_unavailable"
)
PROVIDER_CREDENTIAL_ERROR: Final[ProviderCredentialVerificationStatus] = "error"

_ALIBABA_BASE_URLS = {
    "alibaba_beijing": "https://dashscope.aliyuncs.com/api/v1",
    "alibaba_singapore": "https://dashscope-intl.aliyuncs.com/api/v1",
}
_MODEL_AWARE_PROVIDERS = frozenset({"google", "cerebras"})
_DIRECT_PROVIDERS = frozenset(
    {
        "google",
        "openrouter",
        "deepseek",
        "cerebras",
        "deepgram",
        "soniox",
    }
)


@dataclass(frozen=True, slots=True)
class ProviderCredentialVerificationRequest:
    provider: str
    api_key: str = field(repr=False)
    selected_model: str | None = None
    fallback_models: tuple[str, ...] = ()
    low_latency: bool = False


@dataclass(frozen=True, slots=True)
class ProviderCredentialVerificationOutcome:
    status: ProviderCredentialVerificationStatus
    provider: str
    unavailable_model: str | None = None
    error_text: str | None = None


@dataclass(slots=True)
class ProviderCredentialVerificationOwner:
    verifier: ProviderVerifierPort
    diagnostics_sink: ProviderCredentialVerificationDiagnosticsSink | None = None

    @property
    def owner_name(self) -> str:
        return "ProviderCredentialVerificationOwner"

    async def verify(
        self,
        request: ProviderCredentialVerificationRequest,
    ) -> ProviderCredentialVerificationOutcome:
        provider = request.provider
        if not request.api_key:
            return ProviderCredentialVerificationOutcome(
                status=PROVIDER_CREDENTIAL_EMPTY,
                provider=request.provider,
            )
        if provider in _ALIBABA_BASE_URLS:
            return await self._verify_qwen(request, provider=provider)
        if provider not in _DIRECT_PROVIDERS:
            return ProviderCredentialVerificationOutcome(
                status=PROVIDER_CREDENTIAL_UNKNOWN,
                provider=request.provider,
            )
        try:
            verified = await self.verifier.verify_api_key(
                provider,
                request.api_key,
                model=(request.selected_model if provider in _MODEL_AWARE_PROVIDERS else None),
            )
        except Exception as exc:
            return self._error_outcome(request.provider, exc)
        return ProviderCredentialVerificationOutcome(
            status=(PROVIDER_CREDENTIAL_VERIFIED if verified else PROVIDER_CREDENTIAL_FAILED),
            provider=request.provider,
        )

    async def _verify_qwen(
        self,
        request: ProviderCredentialVerificationRequest,
        *,
        provider: str,
    ) -> ProviderCredentialVerificationOutcome:
        selected_model = request.selected_model
        if selected_model is None:
            return ProviderCredentialVerificationOutcome(
                status=PROVIDER_CREDENTIAL_FAILED,
                provider=request.provider,
            )
        try:
            if await self.verifier.verify_qwen_llm_api_key(
                request.api_key,
                base_url=_ALIBABA_BASE_URLS[provider],
                model=selected_model,
                low_latency=request.low_latency,
            ):
                return ProviderCredentialVerificationOutcome(
                    status=PROVIDER_CREDENTIAL_VERIFIED,
                    provider=request.provider,
                )
            for fallback_model in request.fallback_models:
                if fallback_model == selected_model:
                    continue
                if await self.verifier.verify_qwen_llm_api_key(
                    request.api_key,
                    base_url=_ALIBABA_BASE_URLS[provider],
                    model=fallback_model,
                    low_latency=request.low_latency,
                ):
                    return ProviderCredentialVerificationOutcome(
                        status=PROVIDER_CREDENTIAL_MODEL_UNAVAILABLE,
                        provider=request.provider,
                        unavailable_model=selected_model,
                    )
        except Exception as exc:
            return self._error_outcome(request.provider, exc)
        return ProviderCredentialVerificationOutcome(
            status=PROVIDER_CREDENTIAL_FAILED,
            provider=request.provider,
        )

    def _error_outcome(
        self,
        provider: str,
        exception: BaseException,
    ) -> ProviderCredentialVerificationOutcome:
        self._emit(
            "provider_credential_verification_failed",
            {
                "provider": provider,
                "error_type": type(exception).__name__,
            },
            exception,
        )
        return ProviderCredentialVerificationOutcome(
            status=PROVIDER_CREDENTIAL_ERROR,
            provider=provider,
            error_text=str(exception),
        )

    def _emit(
        self,
        event: str,
        metadata: Mapping[str, object],
        exception: BaseException | None = None,
    ) -> None:
        if self.diagnostics_sink is None:
            return
        try:
            self.diagnostics_sink(event, metadata, exception)
        except Exception:
            return


@dataclass(slots=True)
class ProviderCredentialVerificationInteractionOwner:
    verification_owner: ProviderCredentialVerificationOwner
    selected_model_provider: ProviderCredentialSelectedModelProvider
    fallback_models: tuple[str, ...] = ()
    low_latency: bool = False
    error_sink: ProviderCredentialVerificationErrorSink | None = None

    @property
    def owner_name(self) -> str:
        return "ProviderCredentialVerificationInteractionOwner"

    async def verify(self, provider: str, api_key: str) -> tuple[bool, str]:
        outcome = await self.verification_owner.verify(
            ProviderCredentialVerificationRequest(
                provider=provider,
                api_key=api_key,
                selected_model=self.selected_model_provider(provider),
                fallback_models=self.fallback_models,
                low_latency=self.low_latency,
            )
        )
        if outcome.status == PROVIDER_CREDENTIAL_VERIFIED:
            return True, "Verification successful"
        if outcome.status == PROVIDER_CREDENTIAL_EMPTY:
            return False, "API Key is empty"
        if outcome.status == PROVIDER_CREDENTIAL_UNKNOWN:
            return False, f"Unknown provider: {provider}"
        if outcome.status == PROVIDER_CREDENTIAL_MODEL_UNAVAILABLE:
            return False, f"qwen_model_unavailable:{outcome.unavailable_model}"
        if outcome.status == PROVIDER_CREDENTIAL_ERROR:
            error_text = outcome.error_text or ""
            if self.error_sink is not None:
                self.error_sink(provider, error_text)
            return False, error_text
        return False, "Verification failed (check logs/console for details)"


__all__ = [
    "PROVIDER_CREDENTIAL_EMPTY",
    "PROVIDER_CREDENTIAL_ERROR",
    "PROVIDER_CREDENTIAL_FAILED",
    "PROVIDER_CREDENTIAL_MODEL_UNAVAILABLE",
    "PROVIDER_CREDENTIAL_UNKNOWN",
    "PROVIDER_CREDENTIAL_VERIFIED",
    "ProviderCredentialVerificationOutcome",
    "ProviderCredentialVerificationInteractionOwner",
    "ProviderCredentialVerificationOwner",
    "ProviderCredentialVerificationRequest",
    "ProviderCredentialVerificationStatus",
]
