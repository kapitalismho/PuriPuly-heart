from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Final, Literal

from puripuly_heart.app.ports.provider_verifier import ProviderVerifierPort
from puripuly_heart.config.alibaba_connection import (
    AlibabaConnection,
    AlibabaRegionalSettings,
    resolve_alibaba_connection,
)
from puripuly_heart.core.error_messages import (
    format_error_report_for_log,
    openrouter_auth_failure_report,
)
from puripuly_heart.core.openrouter.authentication import OpenRouterAuthenticationError

ProviderCredentialVerificationStatus = Literal[
    "verified",
    "failed",
    "empty",
    "unknown",
    "error",
]
ProviderCredentialVerificationDiagnosticsSink = Callable[
    [str, Mapping[str, object], BaseException | None],
    None,
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
        if provider == "openrouter" and not verified:
            self._openrouter_report(
                OpenRouterAuthenticationError.from_status(401, stage="key_verification")
            )
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
            return ProviderCredentialVerificationOutcome(
                status=PROVIDER_CREDENTIAL_ERROR,
                provider=provider,
                error_text=self._openrouter_report(exception),
            )
        alibaba = provider.startswith("alibaba_")
        self._emit(
            "provider_credential_verification_failed",
            {"provider": provider, "error_type": type(exception).__name__},
            None if alibaba else exception,
        )
        return ProviderCredentialVerificationOutcome(
            status=PROVIDER_CREDENTIAL_ERROR,
            provider=provider,
            error_text=("Alibaba verification failed" if alibaba else str(exception)),
        )

    def _openrouter_report(self, exception: Exception) -> str:
        report = openrouter_auth_failure_report(exception, stage="key_verification")
        self._emit(
            "provider_credential_verification_failed",
            {
                "provider": "openrouter",
                "error_type": report.diagnostics.fields["exception_type"],
                "report": "[OpenRouterAuth] failed " + format_error_report_for_log(report),
            },
        )
        return report.message.key

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
