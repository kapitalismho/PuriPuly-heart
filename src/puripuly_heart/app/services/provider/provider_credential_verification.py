from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Final, Literal

from puripuly_heart.app.ports.provider_verifier import ProviderVerifierPort
from puripuly_heart.config.alibaba_connection import (
    AlibabaConnection,
    AlibabaRegionalSettings,
    resolve_alibaba_connection,
)
from puripuly_heart.core.diagnostics import emit_failure
from puripuly_heart.core.error_messages import (
    openrouter_auth_failure_report,
    provider_failure_report,
    stt_failure_report,
)
from puripuly_heart.core.messages import ErrorDiagnostics

logger = logging.getLogger(__name__)

ProviderCredentialVerificationStatus = Literal[
    "verified",
    "failed",
    "empty",
    "unknown",
    "error",
]
ProviderCredentialSelectedModelProvider = Callable[[str], str | None]
ProviderCredentialConnectionProvider = Callable[[str], AlibabaConnection]
ProviderCredentialVerificationErrorSink = Callable[[str, str], None]

PROVIDER_CREDENTIAL_VERIFIED: Final[ProviderCredentialVerificationStatus] = "verified"
PROVIDER_CREDENTIAL_FAILED: Final[ProviderCredentialVerificationStatus] = "failed"
PROVIDER_CREDENTIAL_EMPTY: Final[ProviderCredentialVerificationStatus] = "empty"
PROVIDER_CREDENTIAL_UNKNOWN: Final[ProviderCredentialVerificationStatus] = "unknown"
PROVIDER_CREDENTIAL_ERROR: Final[ProviderCredentialVerificationStatus] = "error"

_MODEL_AWARE_PROVIDERS = frozenset({"google", "openai"})
_DIRECT_PROVIDERS = frozenset(
    {
        "google",
        "openrouter",
        "openai",
        "deepseek",
        "deepgram",
        "gemini_transcribe",
        "elevenlabs_scribe",
        "soniox",
    }
)


@dataclass(frozen=True, slots=True)
class ProviderCredentialVerificationRequest:
    provider: str
    api_key: str = field(repr=False)
    connection: AlibabaConnection | None = None
    selected_model: str | None = None
    low_latency: bool = False


@dataclass(frozen=True, slots=True)
class ProviderCredentialVerificationOutcome:
    status: ProviderCredentialVerificationStatus
    provider: str
    error_text: str | None = None
    diagnostics: ErrorDiagnostics | None = None
    verification_scope: str = "key_probe"
    recognition_readiness: str = "unobserved"


@dataclass(slots=True)
class ProviderCredentialVerificationOwner:
    verifier: ProviderVerifierPort

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
        if provider in {"alibaba_beijing", "alibaba_singapore"}:
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
            connection = request.connection or resolve_alibaba_connection(
                "beijing" if provider == "alibaba_beijing" else "singapore",
                AlibabaRegionalSettings(),
            )
            if await self.verifier.verify_qwen_llm_api_key(
                request.api_key,
                base_url=connection.native_url,
                model=selected_model,
                low_latency=request.low_latency,
            ):
                return ProviderCredentialVerificationOutcome(
                    status=PROVIDER_CREDENTIAL_VERIFIED,
                    provider=request.provider,
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
        if provider == "openrouter" and isinstance(exception, Exception):
            report = openrouter_auth_failure_report(exception, stage="key_verification")
        elif provider in {"deepgram", "gemini_transcribe", "elevenlabs_scribe", "soniox"}:
            report = stt_failure_report(
                exception, provider=provider, operation="key_verification", channel=""
            )
        else:
            report = provider_failure_report(
                exception, provider=provider, operation="key_verification"
            )
        diagnostics = replace(
            report.diagnostics,
            channel=None,
            fields={
                **{
                    key: value
                    for key, value in report.diagnostics.fields.items()
                    if key != "channel"
                },
                "verification_scope": "key_probe",
                "recognition_readiness": "unobserved",
            },
        )
        emit_failure(logger, diagnostics, stage="key_verification")
        return ProviderCredentialVerificationOutcome(
            status=PROVIDER_CREDENTIAL_ERROR,
            provider=provider,
            error_text=report.message.key,
            diagnostics=diagnostics,
        )


@dataclass(slots=True)
class ProviderCredentialVerificationInteractionOwner:
    verification_owner: ProviderCredentialVerificationOwner
    selected_model_provider: ProviderCredentialSelectedModelProvider
    low_latency: bool = False
    connection_provider: ProviderCredentialConnectionProvider | None = None
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
                low_latency=self.low_latency,
                connection=(
                    self.connection_provider(provider)
                    if self.connection_provider is not None and provider.startswith("alibaba_")
                    else None
                ),
            )
        )
        if outcome.status == PROVIDER_CREDENTIAL_VERIFIED:
            return True, "Verification successful"
        if outcome.status == PROVIDER_CREDENTIAL_EMPTY:
            return False, "API Key is empty"
        if outcome.status == PROVIDER_CREDENTIAL_UNKNOWN:
            return False, f"Unknown provider: {provider}"
        if outcome.status == PROVIDER_CREDENTIAL_ERROR:
            error_text = outcome.error_text or ""
            if self.error_sink is not None and provider != "openrouter":
                self.error_sink(provider, error_text)
            return False, error_text
        if provider == "openrouter":
            return False, "error.openrouter_auth.credential_rejected"
        return False, "Verification failed (check logs/console for details)"


__all__ = [
    "PROVIDER_CREDENTIAL_EMPTY",
    "PROVIDER_CREDENTIAL_ERROR",
    "PROVIDER_CREDENTIAL_FAILED",
    "PROVIDER_CREDENTIAL_UNKNOWN",
    "PROVIDER_CREDENTIAL_VERIFIED",
    "ProviderCredentialVerificationOutcome",
    "ProviderCredentialVerificationInteractionOwner",
    "ProviderCredentialVerificationOwner",
    "ProviderCredentialVerificationRequest",
    "ProviderCredentialVerificationStatus",
]
