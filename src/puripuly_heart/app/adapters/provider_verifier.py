from __future__ import annotations

from dataclasses import dataclass

from puripuly_heart.app.ports.provider_verifier import (
    PROVIDER_VERIFICATION_STATUS_FAILED,
    PROVIDER_VERIFICATION_STATUS_VERIFIED,
    ProviderVerificationRequest,
    ProviderVerificationResult,
    ProviderVerifierPort,
)
from puripuly_heart.core.messages import (
    CONTENT_POLICY_METADATA_ONLY,
    DIAGNOSTIC_CATEGORY_AUTH,
    DIAGNOSTIC_VISIBILITY_BASIC,
    ErrorDiagnostics,
)
from puripuly_heart.core.openrouter_metadata import OpenRouterKeyMetadata
from puripuly_heart.core.translation_policy import FIXED_TRANSLATION_POLICY
from puripuly_heart.providers.llm.cerebras import CerebrasLLMProvider
from puripuly_heart.providers.llm.deepseek import DeepSeekLLMProvider
from puripuly_heart.providers.llm.gemini import GeminiLLMProvider
from puripuly_heart.providers.llm.openrouter import OpenRouterLLMProvider
from puripuly_heart.providers.llm.qwen_async import AsyncQwenLLMProvider
from puripuly_heart.providers.stt.deepgram import DeepgramRealtimeSTTBackend
from puripuly_heart.providers.stt.elevenlabs_scribe import ElevenLabsScribeSTTBackend
from puripuly_heart.providers.stt.gemini_transcribe import GeminiTranscribeSTTBackend
from puripuly_heart.providers.stt.soniox import SonioxRealtimeSTTBackend

_ALIBABA_BEIJING_BASE_URL = "https://dashscope.aliyuncs.com/api/v1"
_ALIBABA_SINGAPORE_BASE_URL = "https://dashscope-intl.aliyuncs.com/api/v1"


def _compatible_qwen_base_url(base_url: str) -> str:
    return base_url.replace("/api/v1", "/compatible-mode/v1")


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
        if normalized_provider == "deepseek":
            kwargs: dict[str, str] = {}
            if base_url is not None:
                kwargs["base_url"] = base_url
            if model is not None:
                kwargs["model"] = model
            return await DeepSeekLLMProvider.verify_api_key(api_key, **kwargs)
        if normalized_provider == "cerebras":
            kwargs: dict[str, str] = {}
            if base_url is not None:
                kwargs["base_url"] = base_url
            if model is not None:
                kwargs["model"] = model
            return await CerebrasLLMProvider.verify_api_key(api_key, **kwargs)
        if normalized_provider in {"alibaba_beijing", "alibaba_singapore", "qwen"}:
            qwen_base_url = base_url or (
                _ALIBABA_SINGAPORE_BASE_URL
                if normalized_provider == "alibaba_singapore"
                else _ALIBABA_BEIJING_BASE_URL
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
        async_base_url = _compatible_qwen_base_url(base_url)
        kwargs = {"base_url": async_base_url}
        if model is not None:
            kwargs["model"] = model
        if not FIXED_TRANSLATION_POLICY.fast_translation_enabled:
            raise RuntimeError("Fast Translation policy is disabled")
        return await AsyncQwenLLMProvider.verify_api_key(api_key, **kwargs)

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
