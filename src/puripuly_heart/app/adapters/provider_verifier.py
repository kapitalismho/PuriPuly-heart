from __future__ import annotations

import logging
from dataclasses import dataclass

from puripuly_heart.app.ports.provider_verifier import (
    PROVIDER_VERIFICATION_STATUS_FAILED,
    PROVIDER_VERIFICATION_STATUS_VERIFIED,
    ProviderVerificationRequest,
    ProviderVerificationResult,
    ProviderVerifierPort,
)
from puripuly_heart.config.alibaba_connection import (
    AlibabaRegionalSettings,
    resolve_alibaba_connection,
    validated_native_url,
    validated_websocket_url,
)
from puripuly_heart.core.error_messages import (
    format_error_report_for_log,
    openrouter_auth_failure_report,
)
from puripuly_heart.core.messages import (
    CONTENT_POLICY_METADATA_ONLY,
    DIAGNOSTIC_CATEGORY_AUTH,
    DIAGNOSTIC_VISIBILITY_BASIC,
    ErrorDiagnostics,
)
from puripuly_heart.core.openrouter.authentication import OpenRouterAuthenticationError
from puripuly_heart.core.openrouter_metadata import OpenRouterKeyMetadata
from puripuly_heart.core.translation_policy import FIXED_TRANSLATION_POLICY
from puripuly_heart.providers.llm.deepseek import DeepSeekLLMProvider
from puripuly_heart.providers.llm.gemini import GeminiLLMProvider
from puripuly_heart.providers.llm.openai import OpenAILLMProvider
from puripuly_heart.providers.llm.openrouter import OpenRouterLLMProvider
from puripuly_heart.providers.llm.qwen_async import AsyncQwenLLMProvider
from puripuly_heart.providers.stt.deepgram import DeepgramRealtimeSTTBackend
from puripuly_heart.providers.stt.elevenlabs_scribe import ElevenLabsScribeSTTBackend
from puripuly_heart.providers.stt.gemini_transcribe import GeminiTranscribeSTTBackend
from puripuly_heart.providers.stt.qwen_audio import QwenAudioStreamingSTTBackend
from puripuly_heart.providers.stt.soniox import SonioxRealtimeSTTBackend

logger = logging.getLogger(__name__)


def _validated_compatible_url(base_url: str) -> str:
    for region in ("beijing", "singapore"):
        try:
            native_url = validated_native_url(base_url, region)
            return native_url[: -len("/api/v1")] + "/compatible-mode/v1"
        except ValueError:
            continue
    raise ValueError("Invalid Alibaba native endpoint")


def _optional_context_str(request: ProviderVerificationRequest, key: str) -> str | None:
    value = request.context.get(key)
    if isinstance(value, str) and value:
        return value
    return None


def _optional_context_bool(request: ProviderVerificationRequest, key: str) -> bool:
    value = request.context.get(key)
    return bool(value) if isinstance(value, bool) else False


def _verification_diagnostics(
    *,
    provider: str,
    code: str,
    error_type: str | None = None,
) -> ErrorDiagnostics:
    fields: dict[str, str] = {"provider": provider}
    if error_type:
        fields["error_type"] = error_type
    return ErrorDiagnostics(
        component="provider_verifier",
        operation="verify_api_key",
        code=code,
        category=DIAGNOSTIC_CATEGORY_AUTH,
        visibility=DIAGNOSTIC_VISIBILITY_BASIC,
        content_policy=CONTENT_POLICY_METADATA_ONLY,
        status_code=None,
        retry_after_ms=None,
        fields=fields,
    )


@dataclass(frozen=True, slots=True)
class ProviderVerifierAdapter(ProviderVerifierPort):
    async def verify_api_key(
        self,
        provider: str,
        api_key: str,
        *,
        model: str | None = None,
        base_url: str | None = None,
        low_latency: bool = False,
    ) -> bool:
        normalized_provider = provider.strip().lower()
        if normalized_provider == "google":
            return await GeminiLLMProvider.verify_api_key(
                api_key,
                **({"model": model} if model is not None else {}),
            )
        if normalized_provider == "openrouter":
            return await OpenRouterLLMProvider.verify_api_key(api_key)
        if normalized_provider == "openai":
            if model not in (None, "gpt-6-luna"):
                return False
            return await OpenAILLMProvider.verify_api_key(
                api_key,
                model="gpt-6-luna",
            )
        if normalized_provider == "deepseek":
            kwargs: dict[str, str] = {}
            if base_url is not None:
                kwargs["base_url"] = base_url
            if model is not None:
                kwargs["model"] = model
            return await DeepSeekLLMProvider.verify_api_key(api_key, **kwargs)
        if normalized_provider in {"alibaba_beijing", "alibaba_singapore", "qwen"}:
            region = "singapore" if normalized_provider == "alibaba_singapore" else "beijing"
            qwen_base_url = validated_native_url(
                base_url
                or resolve_alibaba_connection(region, AlibabaRegionalSettings()).native_url,
                region,
            )
            return await self.verify_qwen_llm_api_key(
                api_key,
                base_url=qwen_base_url,
                model=model,
                low_latency=low_latency,
            )
        if normalized_provider == "deepgram":
            return await DeepgramRealtimeSTTBackend.verify_api_key(api_key)
        if normalized_provider == "gemini_transcribe":
            return await GeminiTranscribeSTTBackend.verify_api_key(api_key)
        if normalized_provider == "elevenlabs_scribe":
            return await ElevenLabsScribeSTTBackend.verify_api_key(api_key)
        if normalized_provider == "soniox":
            return await SonioxRealtimeSTTBackend.verify_api_key(api_key)
        raise ValueError(f"Unknown provider: {provider}")

    async def verify_qwen_llm_api_key(
        self,
        api_key: str,
        *,
        base_url: str,
        model: str | None,
        low_latency: bool,
    ) -> bool:
        _ = low_latency
        async_base_url = _validated_compatible_url(base_url)
        kwargs = {"base_url": async_base_url}
        if model is not None:
            kwargs["model"] = model
        if not FIXED_TRANSLATION_POLICY.fast_translation_enabled:
            raise RuntimeError("Fast Translation policy is disabled")
        return await AsyncQwenLLMProvider.verify_api_key(api_key, **kwargs)

    async def probe_qwen_llm_api_key(self, api_key: str, *, base_url: str, model: str) -> bool:
        if not FIXED_TRANSLATION_POLICY.fast_translation_enabled:
            raise RuntimeError("Fast Translation policy is disabled")
        return await AsyncQwenLLMProvider.probe_api_key(
            api_key, base_url=_validated_compatible_url(base_url), model=model
        )

    async def verify_qwen_audio_api_key(self, api_key: str, *, endpoint: str, model: str) -> bool:
        for region in ("beijing", "singapore"):
            try:
                endpoint = validated_websocket_url(endpoint, region)
                break
            except ValueError:
                continue
        else:
            raise ValueError("Invalid Alibaba WebSocket endpoint")
        return await QwenAudioStreamingSTTBackend.verify_api_key(
            api_key, endpoint=endpoint, model=model
        )

    async def fetch_openrouter_key_metadata(
        self,
        api_key: str,
    ) -> OpenRouterKeyMetadata | None:
        return await OpenRouterLLMProvider.fetch_key_metadata(api_key)

    async def verify_provider_secret(
        self,
        request: ProviderVerificationRequest,
    ) -> ProviderVerificationResult:
        try:
            verified = await self.verify_api_key(
                request.provider,
                request.secret_value,
                model=_optional_context_str(request, "model"),
                base_url=_optional_context_str(request, "base_url"),
                low_latency=_optional_context_bool(request, "low_latency"),
            )
        except Exception as exc:
            if request.provider == "openrouter":
                report = openrouter_auth_failure_report(exc, stage="key_verification")
                logger.error("[OpenRouterAuth] failed " + format_error_report_for_log(report))
                return ProviderVerificationResult(
                    status=PROVIDER_VERIFICATION_STATUS_FAILED,
                    provider=request.provider,
                    secret_key=request.secret_key,
                    secret_revision=request.secret_revision,
                    evidence={"verifier": "provider_adapter", "provider": request.provider},
                    message=report.message,
                    diagnostics=report.diagnostics,
                )
            return ProviderVerificationResult(
                status=PROVIDER_VERIFICATION_STATUS_FAILED,
                provider=request.provider,
                secret_key=request.secret_key,
                secret_revision=request.secret_revision,
                evidence={
                    "verifier": "provider_adapter",
                    "provider": request.provider,
                    "context_count": len(request.context),
                    "error_type": type(exc).__name__,
                },
                message=None,
                diagnostics=_verification_diagnostics(
                    provider=request.provider,
                    code="provider_verifier_exception",
                    error_type=type(exc).__name__,
                ),
            )

        if verified:
            return ProviderVerificationResult(
                status=PROVIDER_VERIFICATION_STATUS_VERIFIED,
                provider=request.provider,
                secret_key=request.secret_key,
                secret_revision=request.secret_revision,
                evidence={
                    "verifier": "provider_adapter",
                    "provider": request.provider,
                    "context_count": len(request.context),
                },
                message=None,
                diagnostics=None,
            )

        if request.provider == "openrouter":
            report = openrouter_auth_failure_report(
                OpenRouterAuthenticationError.from_status(401, stage="key_verification"),
                stage="key_verification",
            )
            logger.error("[OpenRouterAuth] failed " + format_error_report_for_log(report))
            return ProviderVerificationResult(
                status=PROVIDER_VERIFICATION_STATUS_FAILED,
                provider=request.provider,
                secret_key=request.secret_key,
                secret_revision=request.secret_revision,
                evidence={"verifier": "provider_adapter", "provider": request.provider},
                message=report.message,
                diagnostics=report.diagnostics,
            )
        return ProviderVerificationResult(
            status=PROVIDER_VERIFICATION_STATUS_FAILED,
            provider=request.provider,
            secret_key=request.secret_key,
            secret_revision=request.secret_revision,
            evidence={
                "verifier": "provider_adapter",
                "provider": request.provider,
                "context_count": len(request.context),
            },
            message=None,
            diagnostics=_verification_diagnostics(
                provider=request.provider,
                code="provider_verification_failed",
            ),
        )


__all__ = ["ProviderVerifierAdapter"]
