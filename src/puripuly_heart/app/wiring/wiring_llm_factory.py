from __future__ import annotations

import asyncio
import contextlib
import os
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace

from puripuly_heart.app.ports.secret_store import SecretReadResult
from puripuly_heart.app.services.managed_auth_claims import (
    MANAGED_AUTH_CLAIM_SOURCE_DISCORD,
    MANAGED_AUTH_CLAIM_SOURCE_QQ,
    ManagedAuthClaimGuard,
    local_managed_auth_blocking_source,
)
from puripuly_heart.config.llm_profiles import openrouter_alias_for_fields
from puripuly_heart.config.resolved import (
    CREDENTIAL_SOURCE_MANAGED,
    CREDENTIAL_SOURCE_NONE,
    CREDENTIAL_SOURCE_SECRET_STORE,
    ResolvedCredentialRequirement,
    ResolvedLLMConfig,
    ResolvedLLMTarget,
)
from puripuly_heart.config.runtime_resolution import (
    CREDENTIAL_REF_CEREBRAS_BYOK,
    CREDENTIAL_REF_OPENROUTER_BYOK,
    CREDENTIAL_REF_OPENROUTER_MANAGED,
    CREDENTIAL_REF_OPENROUTER_MANAGED_QQ,
    CREDENTIAL_REF_QWEN_BEIJING,
    CREDENTIAL_REF_QWEN_SINGAPORE,
    PROVIDER_CEREBRAS,
    PROVIDER_DEEPSEEK,
    PROVIDER_GEMINI,
    PROVIDER_LOCAL_LLM,
    PROVIDER_MANAGED_GEMMA,
    PROVIDER_OPENROUTER,
    PROVIDER_QWEN,
    TRANSLATION_CONNECTION_OFFICIAL_BYOK,
    TRANSLATION_MODEL_MANAGED_GEMMA,
    TRANSLATION_MODEL_QWEN_35_PLUS,
    DirectProviderRuntimeIntent,
    RuntimeResolutionInput,
    TranslationFallbackRuntimeIntent,
    derive_translation_runtime_intent_from_compatibility,
    normalize_openrouter_runtime_intent,
    normalize_translation_runtime_intent,
    resolve_llm_config,
)
from puripuly_heart.config.settings import (
    AppSettings,
    LLMProviderName,
    OpenRouterCredentialSource,
    OpenRouterLLMModel,
    OpenRouterProviderRouting,
    OpenRouterRoutingMode,
    OpenRouterSelectionAlias,
    QwenRegion,
    TranslationConnection,
    TranslationModel,
)
from puripuly_heart.core.llm import FallbackRacingLLMProvider
from puripuly_heart.core.llm.fallback_racing import LLMProviderAttempt
from puripuly_heart.core.llm.provider import LLMProvider, SemaphoreLLMProvider
from puripuly_heart.core.local_translation.runtime import ManagedGemmaRuntimeOwner
from puripuly_heart.core.openrouter_credentials import (
    OPENROUTER_BYOK_API_KEY_ENV,
    OPENROUTER_BYOK_API_KEY_SECRET,
    OPENROUTER_MANAGED_API_KEY_SECRET,
    OPENROUTER_MANAGED_QQ_API_KEY_SECRET,
    load_managed_openrouter_user_identifier,
)
from puripuly_heart.core.runtime_logging import SessionRuntimeLoggingService
from puripuly_heart.core.storage.secrets import SecretStore
from puripuly_heart.core.translation_policy import FIXED_TRANSLATION_POLICY
from puripuly_heart.domain.models import Translation
from puripuly_heart.providers.llm.cerebras import CerebrasLLMProvider
from puripuly_heart.providers.llm.deepseek import DeepSeekLLMProvider
from puripuly_heart.providers.llm.gemini import GeminiLLMProvider
from puripuly_heart.providers.llm.local_openai import LocalOpenAICompatibleLLMProvider
from puripuly_heart.providers.llm.managed_gemma import ManagedGemmaLLMProvider
from puripuly_heart.providers.llm.openrouter import OpenRouterLLMProvider
from puripuly_heart.providers.llm.qwen_async import AsyncQwenLLMProvider

from .wiring_managed_auth_factory import (
    _managed_release_service_for_alias,
    build_openrouter_credential_runtime_config,
)
from .wiring_secrets_factory import require_secret, require_secret_any

MANAGED_OPENROUTER_RELEASE_SERVICE_REQUIRED_ERROR = (
    "OpenRouter managed mode requires a managed release service; "
    "non-GUI paths are not wired for managed OpenRouter mode yet"
)


@dataclass(slots=True)
class _LazyFactoryLLMProvider(LLMProvider):
    factory: Callable[[], LLMProvider]
    _delegate: LLMProvider | None = field(init=False, default=None, repr=False)
    _delegate_lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock, repr=False)

    async def _ensure_delegate(self) -> LLMProvider:
        if self._delegate is not None:
            return self._delegate

        async with self._delegate_lock:
            if self._delegate is None:
                self._delegate = self.factory()
            return self._delegate

    async def translate(
        self,
        *,
        utterance_id,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
    ) -> Translation:
        delegate = await self._ensure_delegate()
        return await delegate.translate(
            utterance_id=utterance_id,
            text=text,
            system_prompt=system_prompt,
            source_language=source_language,
            target_language=target_language,
            context=context,
        )

    async def close(self) -> None:
        if self._delegate is not None:
            await self._delegate.close()


@dataclass(slots=True)
class _SyncSecretStoreReadPort:
    secrets: SecretStore

    async def get_secret(self, key: str) -> SecretReadResult:
        return SecretReadResult(
            key=key,
            value=self.secrets.get(key),
            revision=None,
            message=None,
            diagnostics=None,
        )


@dataclass(slots=True)
class _ClaimGuardedManagedReleaseService:
    release_service: object
    claim_guard: ManagedAuthClaimGuard
    claim_source: str

    def __getattr__(self, name: str) -> object:
        return getattr(self.release_service, name)

    async def ensure_key_for_llm_start(self) -> object:
        claim_result = await self.claim_guard.preflight(self.claim_source)
        if claim_result is not None:
            return _managed_release_result_from_claim_result(claim_result)
        ensure_key = getattr(self.release_service, "ensure_key_for_llm_start")
        result = await ensure_key()
        if _managed_release_result_is_ready_with_local_key(result):
            with contextlib.suppress(Exception):
                self.claim_guard.record_success(self.claim_source)
                self.claim_guard.managed_state.persist()
        return result


def _shared_managed_release_service_for_fallback(
    primary: LLMProvider,
    managed_release_service: object | None,
) -> object | None:
    from puripuly_heart.core.managed_openrouter_release import ManagedOpenRouterLLMProvider

    if isinstance(primary, ManagedOpenRouterLLMProvider):
        return _unwrap_claim_guarded_managed_release_service(primary.release_service)
    return _unwrap_claim_guarded_managed_release_service(managed_release_service)


def _unwrap_claim_guarded_managed_release_service(release_service: object | None) -> object | None:
    while isinstance(release_service, _ClaimGuardedManagedReleaseService):
        release_service = release_service.release_service
    return release_service


def _runtime_resolution_input_from_compatibility_settings(
    settings: AppSettings,
) -> RuntimeResolutionInput:
    openrouter_intent = normalize_openrouter_runtime_intent(
        provider_llm=settings.provider.llm,
        model=settings.openrouter.llm_model,
        selected_source=settings.openrouter.selected_source,
        selection_alias=settings.openrouter.selection_alias,
        routing_mode=settings.openrouter.routing_mode,
        provider_routing=settings.openrouter.provider_routing,
        broker_base_url=settings.openrouter.broker_base_url,
    )
    translation_intent = derive_translation_runtime_intent_from_compatibility(
        provider_llm=settings.provider.llm,
        openrouter_model=openrouter_intent.model,
        openrouter_selected_source=openrouter_intent.selected_source,
        openrouter_provider_routing=openrouter_intent.provider_routing,
        gemini_model=settings.gemini.llm_model,
        qwen_model=settings.qwen.llm_model,
        concurrency_limit=settings.llm.concurrency_limit,
    )
    if settings.translation.model == TranslationModel.CUSTOM_HTTP:
        translation_intent = normalize_translation_runtime_intent(
            model="custom_http",
            connection="custom_http",
            concurrency_limit=settings.llm.concurrency_limit,
        )
    elif settings.translation.model == TranslationModel.MANAGED_GEMMA:
        translation_intent = normalize_translation_runtime_intent(
            model=TRANSLATION_MODEL_MANAGED_GEMMA,
            connection=settings.translation.connection.value,
            concurrency_limit=settings.llm.concurrency_limit,
        )
    elif settings.provider.llm == LLMProviderName.QWEN:
        translation_intent = normalize_translation_runtime_intent(
            model=TRANSLATION_MODEL_QWEN_35_PLUS,
            connection=TRANSLATION_CONNECTION_OFFICIAL_BYOK,
            concurrency_limit=settings.llm.concurrency_limit,
        )
    elif (
        settings.provider.llm == LLMProviderName.OPENROUTER
        and openrouter_intent.selected_source == OpenRouterCredentialSource.NONE.value
    ):
        raise ValueError("OpenRouter selected source must not be `none` for execution")
    direct_intent = DirectProviderRuntimeIntent(
        qwen_35_plus_model=settings.qwen.llm_model.value,
        qwen_region=settings.qwen.region.value,
        local_llm_backend=settings.local_llm.backend.value,
        local_llm_base_url=settings.local_llm.base_url,
        local_llm_model=settings.local_llm.model,
        local_llm_extra_body=settings.local_llm.extra_body,
    )
    return RuntimeResolutionInput(
        translation=translation_intent,
        translation_fallback=TranslationFallbackRuntimeIntent(
            enabled=settings.translation.fallback.enabled,
            model=settings.translation.fallback.model.value,
            connection=settings.translation.fallback.connection.value,
        ),
        openrouter=openrouter_intent,
        direct=direct_intent,
    )


def _plain_resolved_option_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain_resolved_option_value(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_plain_resolved_option_value(child) for child in value]
    return value


def _resolved_option_mapping(
    values: Mapping[str, object],
    key: str,
) -> dict[str, object]:
    value = values.get(key)
    if not isinstance(value, Mapping):
        return {}
    return {
        str(option_key): _plain_resolved_option_value(option_value)
        for option_key, option_value in value.items()
    }


def _normalize_secret_value(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _require_openrouter_byok_api_key(secrets: SecretStore) -> str:
    value = _normalize_secret_value(secrets.get(OPENROUTER_BYOK_API_KEY_SECRET))
    if value is not None:
        return value
    value = _normalize_secret_value(os.getenv(OPENROUTER_BYOK_API_KEY_ENV))
    if value is not None:
        return value
    raise ValueError(
        f"Missing secret `{OPENROUTER_BYOK_API_KEY_SECRET}` "
        f"(or env var {OPENROUTER_BYOK_API_KEY_ENV})"
    )


def _openrouter_managed_api_key(
    secrets: SecretStore,
    credential: ResolvedCredentialRequirement,
    settings: AppSettings,
) -> str | None:
    requested_secret_key = _openrouter_managed_api_key_secret_for_resolved_credential(credential)
    opposite_secret_key = _opposite_openrouter_managed_api_key_secret(requested_secret_key)
    requested_value = _normalize_secret_value(secrets.get(requested_secret_key))
    opposite_value = _normalize_secret_value(secrets.get(opposite_secret_key))
    if opposite_value is not None:
        raise ValueError("OpenRouter managed local claim conflict")
    if (
        credential.reference == CREDENTIAL_REF_OPENROUTER_MANAGED_QQ
        and _normalize_secret_value(settings.managed_identity.active_managed_credential_ref) is None
    ):
        return None
    return requested_value


def _openrouter_managed_api_key_secret_for_resolved_credential(
    credential: ResolvedCredentialRequirement,
) -> str:
    if credential.reference == CREDENTIAL_REF_OPENROUTER_MANAGED_QQ:
        return OPENROUTER_MANAGED_QQ_API_KEY_SECRET
    return OPENROUTER_MANAGED_API_KEY_SECRET


def _managed_claim_source_for_resolved_credential(
    credential: ResolvedCredentialRequirement,
) -> str:
    if credential.reference == CREDENTIAL_REF_OPENROUTER_MANAGED_QQ:
        return MANAGED_AUTH_CLAIM_SOURCE_QQ
    return MANAGED_AUTH_CLAIM_SOURCE_DISCORD


def _settings_have_opposite_managed_claim_source(
    settings: AppSettings | None,
    credential: ResolvedCredentialRequirement,
) -> bool:
    if settings is None:
        return False
    return (
        local_managed_auth_blocking_source(
            settings.managed_identity.local_managed_claim_sources,
            _managed_claim_source_for_resolved_credential(credential),
        )
        is not None
    )


def _opposite_openrouter_managed_api_key_secret(secret_key: str) -> str:
    if secret_key == OPENROUTER_MANAGED_QQ_API_KEY_SECRET:
        return OPENROUTER_MANAGED_API_KEY_SECRET
    return OPENROUTER_MANAGED_QQ_API_KEY_SECRET


def _guarded_managed_release_service_for_claim(
    release_service: object,
    *,
    secrets: SecretStore,
    credential: ResolvedCredentialRequirement,
) -> object:
    managed_state = getattr(release_service, "managed_state", None)
    if managed_state is None:
        return release_service
    return _ClaimGuardedManagedReleaseService(
        release_service=release_service,
        claim_guard=ManagedAuthClaimGuard(
            managed_state=managed_state,
            secret_store=_SyncSecretStoreReadPort(secrets),
        ),
        claim_source=_managed_claim_source_for_resolved_credential(credential),
    )


def _managed_release_result_from_claim_result(result: object) -> object:
    from puripuly_heart.core.managed_openrouter_release import (
        ManagedOpenRouterReleaseBehavior,
        ManagedOpenRouterReleaseDiagnostics,
        ManagedOpenRouterReleaseResult,
    )

    message = getattr(result, "message", None)
    diagnostics = getattr(result, "diagnostics", None)
    return ManagedOpenRouterReleaseResult(
        behavior=ManagedOpenRouterReleaseBehavior.STOP,
        message_key=getattr(message, "key", None) or "managed_auth.error.claim_conflict",
        message_kwargs=dict(getattr(message, "params", {}) or {}),
        diagnostics=ManagedOpenRouterReleaseDiagnostics(
            operation=getattr(diagnostics, "operation", None),
            code=getattr(diagnostics, "code", None) or "managed_auth_claim_source_blocked",
            error_class="terminal",
        ),
    )


def _managed_release_result_is_ready_with_local_key(result: object) -> bool:
    return getattr(result, "behavior", None) == "ready" and bool(
        getattr(result, "local_key_available", False)
    )


def _openrouter_source_for_resolved_credential(
    credential: ResolvedCredentialRequirement,
) -> OpenRouterCredentialSource:
    if (
        credential.source == CREDENTIAL_SOURCE_MANAGED
        or credential.reference == CREDENTIAL_REF_OPENROUTER_MANAGED
        or credential.reference == CREDENTIAL_REF_OPENROUTER_MANAGED_QQ
    ):
        return OpenRouterCredentialSource.MANAGED
    if (
        credential.source == CREDENTIAL_SOURCE_SECRET_STORE
        and credential.reference == CREDENTIAL_REF_OPENROUTER_BYOK
    ):
        return OpenRouterCredentialSource.BYOK
    if credential.source == CREDENTIAL_SOURCE_NONE:
        return OpenRouterCredentialSource.NONE
    raise ValueError("Unsupported OpenRouter resolved credential reference")


def _settings_for_resolved_openrouter_fields(
    settings: AppSettings | None,
    *,
    model: str,
    models: tuple[str, ...] = (),
    service_endpoint: str | None,
    selected_source: OpenRouterCredentialSource,
    provider_routing: OpenRouterProviderRouting,
    routing_mode: OpenRouterRoutingMode,
    include_selection_alias: bool,
) -> AppSettings:
    resolved_settings = replace(settings) if settings is not None else AppSettings()
    resolved_settings.openrouter = replace(resolved_settings.openrouter)
    resolved_settings.openrouter.llm_model = OpenRouterLLMModel(model)
    resolved_settings.openrouter.selected_source = selected_source
    resolved_settings.openrouter.routing_mode = routing_mode
    resolved_settings.openrouter.provider_routing = provider_routing
    resolved_settings.openrouter.broker_base_url = service_endpoint or ""
    selection_alias = None
    if include_selection_alias:
        alias_value = openrouter_alias_for_fields(
            model=model,
            source=selected_source.value,
            models=models,
        )
        if alias_value is not None:
            selection_alias = OpenRouterSelectionAlias(alias_value)
    resolved_settings.openrouter.selection_alias = selection_alias
    return resolved_settings


def _settings_for_resolved_managed_credential(
    settings: AppSettings,
    credential: ResolvedCredentialRequirement,
) -> AppSettings:
    connection = (
        TranslationConnection.MANAGED_CHINA
        if credential.reference == CREDENTIAL_REF_OPENROUTER_MANAGED_QQ
        else TranslationConnection.MANAGED
    )
    resolved_settings = replace(settings)
    resolved_settings.translation = replace(resolved_settings.translation)
    resolved_settings.translation.connection = connection
    return resolved_settings


def _openrouter_routing_mode(value: str | None) -> OpenRouterRoutingMode:
    if value is None:
        return OpenRouterRoutingMode.LATENCY
    return OpenRouterRoutingMode(value)


def _openrouter_provider_routing(value: str | None) -> OpenRouterProviderRouting:
    if value is None:
        return OpenRouterProviderRouting.DEFAULT
    return OpenRouterProviderRouting(value)


def _qwen_api_key_for_resolved_credential(
    credential: ResolvedCredentialRequirement,
    *,
    secrets: SecretStore,
) -> str:
    if credential.reference == CREDENTIAL_REF_QWEN_SINGAPORE:
        return require_secret_any(
            secrets,
            key="alibaba_api_key_singapore",
            env_vars=("ALIBABA_API_KEY_SINGAPORE", "ALIBABA_API_KEY", "DASHSCOPE_API_KEY"),
            legacy_keys=("alibaba_api_key",),
        )
    if credential.reference in (CREDENTIAL_REF_QWEN_BEIJING, None):
        return require_secret_any(
            secrets,
            key="alibaba_api_key_beijing",
            env_vars=("ALIBABA_API_KEY_BEIJING", "ALIBABA_API_KEY", "DASHSCOPE_API_KEY"),
            legacy_keys=("alibaba_api_key",),
        )
    raise ValueError("Unsupported Qwen resolved credential reference")


def _cerebras_api_key_for_resolved_credential(
    credential: ResolvedCredentialRequirement,
    *,
    secrets: SecretStore,
) -> str:
    if credential.reference in (CREDENTIAL_REF_CEREBRAS_BYOK, None):
        return require_secret(secrets, key="cerebras_api_key", env_var="CEREBRAS_API_KEY")
    raise ValueError("Unsupported Cerebras resolved credential reference")


def _qwen_sync_base_url(target: ResolvedLLMTarget) -> str:
    if target.service_endpoint:
        return target.service_endpoint
    if target.region == QwenRegion.SINGAPORE.value:
        return "https://dashscope-intl.aliyuncs.com/api/v1"
    return "https://dashscope.aliyuncs.com/api/v1"


def _qwen_async_base_url(target: ResolvedLLMTarget) -> str:
    sync_base_url = _qwen_sync_base_url(target).rstrip("/")
    if sync_base_url.endswith("/compatible-mode/v1"):
        return sync_base_url
    if sync_base_url.endswith("/api/v1"):
        return sync_base_url[: -len("/api/v1")] + "/compatible-mode/v1"
    return sync_base_url + "/compatible-mode/v1"


def _openrouter_provider_from_resolved_config(
    config: ResolvedLLMConfig,
    *,
    secrets: SecretStore,
    managed_release_service: object | None,
    managed_delegate_ready: Callable[[], object] | None,
    runtime_logging: SessionRuntimeLoggingService | None,
    compatibility_settings: AppSettings | None,
    force_managed_wrapper: bool = False,
    include_selection_alias: bool = True,
) -> LLMProvider:
    return _openrouter_provider_from_resolved_target(
        config.primary,
        secrets=secrets,
        managed_release_service=managed_release_service,
        managed_delegate_ready=managed_delegate_ready,
        runtime_logging=runtime_logging,
        compatibility_settings=compatibility_settings,
        force_managed_wrapper=force_managed_wrapper,
        include_selection_alias=include_selection_alias,
    )


def _openrouter_provider_from_resolved_target(
    target: ResolvedLLMTarget,
    *,
    secrets: SecretStore,
    managed_release_service: object | None,
    managed_delegate_ready: Callable[[], object] | None,
    runtime_logging: SessionRuntimeLoggingService | None,
    compatibility_settings: AppSettings | None,
    force_managed_wrapper: bool = False,
    include_selection_alias: bool = True,
) -> LLMProvider:
    return _openrouter_provider_from_resolved_fields(
        model=target.model,
        models=target.models,
        credential=target.credential,
        service_endpoint=target.service_endpoint,
        routing_mode_value=target.routing_mode,
        provider_routing_value=target.provider_routing,
        secrets=secrets,
        managed_release_service=managed_release_service,
        managed_delegate_ready=managed_delegate_ready,
        runtime_logging=runtime_logging,
        compatibility_settings=compatibility_settings,
        force_managed_wrapper=force_managed_wrapper,
        include_selection_alias=include_selection_alias,
    )


def _openrouter_provider_from_resolved_fields(
    *,
    model: str,
    models: tuple[str, ...] = (),
    credential: ResolvedCredentialRequirement,
    service_endpoint: str | None,
    routing_mode_value: str | None,
    provider_routing_value: str | None,
    secrets: SecretStore,
    managed_release_service: object | None,
    managed_delegate_ready: Callable[[], object] | None,
    runtime_logging: SessionRuntimeLoggingService | None,
    compatibility_settings: AppSettings | None,
    force_managed_wrapper: bool = False,
    include_selection_alias: bool = True,
) -> LLMProvider:
    selected_source = _openrouter_source_for_resolved_credential(credential)
    if selected_source == OpenRouterCredentialSource.NONE:
        raise ValueError("OpenRouter selected source must not be `none` for execution")

    routing_mode = _openrouter_routing_mode(routing_mode_value)
    provider_routing = _openrouter_provider_routing(provider_routing_value)
    openrouter_settings = _settings_for_resolved_openrouter_fields(
        compatibility_settings,
        model=model,
        models=models,
        service_endpoint=service_endpoint,
        selected_source=selected_source,
        provider_routing=provider_routing,
        routing_mode=routing_mode,
        include_selection_alias=include_selection_alias,
    )

    if selected_source == OpenRouterCredentialSource.MANAGED:
        if managed_release_service is None:
            raise ValueError(MANAGED_OPENROUTER_RELEASE_SERVICE_REQUIRED_ERROR)
        openrouter_settings = _settings_for_resolved_managed_credential(
            openrouter_settings,
            credential,
        )
        if _settings_have_opposite_managed_claim_source(compatibility_settings, credential):
            raise ValueError("OpenRouter managed local claim conflict")
        alias_managed_release_service = _managed_release_service_for_alias(
            _unwrap_claim_guarded_managed_release_service(managed_release_service),
            alias_settings=openrouter_settings,
        )
        guarded_managed_release_service = _guarded_managed_release_service_for_claim(
            alias_managed_release_service,
            secrets=secrets,
            credential=credential,
        )
        managed_api_key = _openrouter_managed_api_key(secrets, credential, openrouter_settings)
        if force_managed_wrapper or managed_api_key is None:
            from puripuly_heart.core.managed_openrouter_release import ManagedOpenRouterLLMProvider

            return ManagedOpenRouterLLMProvider(
                release_service=guarded_managed_release_service,
                delegate_factory=lambda api_key: OpenRouterLLMProvider(
                    api_key=api_key,
                    user_identifier=load_managed_openrouter_user_identifier(
                        build_openrouter_credential_runtime_config(openrouter_settings),
                        secrets=secrets,
                    ),
                    model=model,
                    models=models,
                    routing_mode=routing_mode,
                    provider_routing=provider_routing,
                    runtime_logging=runtime_logging,
                ),
                on_delegate_ready=managed_delegate_ready,
            )
        return OpenRouterLLMProvider(
            api_key=managed_api_key,
            user_identifier=load_managed_openrouter_user_identifier(
                build_openrouter_credential_runtime_config(openrouter_settings),
                secrets=secrets,
            ),
            model=model,
            models=models,
            routing_mode=routing_mode,
            provider_routing=provider_routing,
            runtime_logging=runtime_logging,
        )

    api_key = _require_openrouter_byok_api_key(secrets)
    return OpenRouterLLMProvider(
        api_key=api_key,
        model=model,
        models=models,
        routing_mode=routing_mode,
        provider_routing=provider_routing,
        runtime_logging=runtime_logging,
    )


def _provider_from_resolved_target(
    target: ResolvedLLMTarget,
    *,
    secrets: SecretStore,
    managed_release_service: object | None,
    managed_delegate_ready: Callable[[], object] | None,
    runtime_logging: SessionRuntimeLoggingService | None,
    compatibility_settings: AppSettings | None,
    managed_gemma_runtime: ManagedGemmaRuntimeOwner | None,
    managed_gemma_release: Callable[[], Awaitable[None]] | None,
    qwen_low_latency_mode: bool,
    force_managed_wrapper: bool = False,
    include_selection_alias: bool = True,
) -> LLMProvider:
    if target.provider == PROVIDER_MANAGED_GEMMA:
        if managed_gemma_runtime is None:
            raise RuntimeError("managed Gemma runtime is unavailable")
        backend = target.provider_options.get("backend")
        if backend not in {"cpu", "gpu"}:
            raise ValueError("managed Gemma backend must be cpu or gpu")
        return ManagedGemmaLLMProvider(
            runtime=managed_gemma_runtime,
            backend=backend,
            release_runtime=managed_gemma_release,
            runtime_logging=runtime_logging,
        )

    if target.provider == PROVIDER_GEMINI:
        api_key = require_secret(secrets, key="google_api_key", env_var="GOOGLE_API_KEY")
        return GeminiLLMProvider(
            api_key=api_key,
            model=target.model,
            runtime_logging=runtime_logging,
        )

    if target.provider == PROVIDER_OPENROUTER:
        return _openrouter_provider_from_resolved_target(
            target,
            secrets=secrets,
            managed_release_service=managed_release_service,
            managed_delegate_ready=managed_delegate_ready,
            runtime_logging=runtime_logging,
            compatibility_settings=compatibility_settings,
            force_managed_wrapper=force_managed_wrapper,
            include_selection_alias=include_selection_alias,
        )

    if target.provider == PROVIDER_QWEN:
        api_key = _qwen_api_key_for_resolved_credential(target.credential, secrets=secrets)
        _ = qwen_low_latency_mode
        return AsyncQwenLLMProvider(
            api_key=api_key,
            base_url=_qwen_async_base_url(target),
            model=target.model,
            runtime_logging=runtime_logging,
        )

    if target.provider == PROVIDER_DEEPSEEK:
        api_key = require_secret(
            secrets,
            key="deepseek_api_key",
            env_var="DEEPSEEK_API_KEY",
        )
        return DeepSeekLLMProvider(
            api_key=api_key,
            model=target.model,
            runtime_logging=runtime_logging,
        )

    if target.provider == PROVIDER_CEREBRAS:
        api_key = _cerebras_api_key_for_resolved_credential(target.credential, secrets=secrets)
        return CerebrasLLMProvider(
            api_key=api_key,
            model=target.model,
            runtime_logging=runtime_logging,
        )

    if target.provider == PROVIDER_LOCAL_LLM:
        api_key = (secrets.get("local_llm_api_key") or "").strip()
        return LocalOpenAICompatibleLLMProvider(
            base_url=target.base_url or "http://127.0.0.1:11434/v1",
            model=target.model,
            extra_body=_resolved_option_mapping(target.provider_options, "extra_body"),
            api_key=api_key,
            runtime_logging=runtime_logging,
        )

    raise ValueError(f"Unsupported LLM provider: {target.provider}")


def _base_llm_provider_from_resolved_config(
    config: ResolvedLLMConfig,
    *,
    secrets: SecretStore,
    managed_release_service: object | None,
    managed_delegate_ready: Callable[[], object] | None,
    runtime_logging: SessionRuntimeLoggingService | None,
    compatibility_settings: AppSettings | None,
    managed_gemma_runtime: ManagedGemmaRuntimeOwner | None,
    managed_gemma_release: Callable[[], Awaitable[None]] | None,
    qwen_low_latency_mode: bool,
) -> LLMProvider:
    return _provider_from_resolved_target(
        config.primary,
        secrets=secrets,
        managed_release_service=managed_release_service,
        managed_delegate_ready=managed_delegate_ready,
        runtime_logging=runtime_logging,
        compatibility_settings=compatibility_settings,
        managed_gemma_runtime=managed_gemma_runtime,
        managed_gemma_release=managed_gemma_release,
        qwen_low_latency_mode=qwen_low_latency_mode,
    )


def _fallback_attempt_log_summary(
    target: ResolvedLLMTarget,
    *,
    start_after_ms: int,
) -> str:
    fields = [f"provider={target.provider}"]
    if len(target.models) == 1:
        fields.append(f"model={target.models[0]}")
    else:
        fields.append(f"models=[{','.join(target.models)}]")
    if target.routing_mode:
        fields.append(f"mode={target.routing_mode}")
    if target.provider_routing:
        fields.append(f"route={target.provider_routing}")
    fields.append(f"delay={start_after_ms}ms")
    return ", ".join(fields)


def create_llm_provider_from_resolved_config(
    config: ResolvedLLMConfig,
    *,
    secrets: SecretStore,
    managed_release_service: object | None = None,
    managed_delegate_ready: Callable[[], object] | None = None,
    runtime_logging: SessionRuntimeLoggingService | None = None,
    compatibility_settings: AppSettings | None = None,
    managed_gemma_runtime: ManagedGemmaRuntimeOwner | None = None,
    managed_gemma_release: Callable[[], Awaitable[None]] | None = None,
    qwen_low_latency_mode: bool = True,
) -> LLMProvider:
    base = _base_llm_provider_from_resolved_config(
        config,
        secrets=secrets,
        managed_release_service=managed_release_service,
        managed_delegate_ready=managed_delegate_ready,
        runtime_logging=runtime_logging,
        compatibility_settings=compatibility_settings,
        managed_gemma_runtime=managed_gemma_runtime,
        managed_gemma_release=managed_gemma_release,
        qwen_low_latency_mode=qwen_low_latency_mode,
    )
    if len(config.attempts) > 1:
        fallback_managed_release_service = _shared_managed_release_service_for_fallback(
            base,
            managed_release_service,
        )
        attempt_providers: list[LLMProviderAttempt] = [
            LLMProviderAttempt(provider=base, start_after_ms=config.attempts[0].start_after_ms)
        ]
        for index, attempt_plan in enumerate(config.attempts[1:], start=1):
            force_managed_wrapper = (
                attempt_plan.target.provider == PROVIDER_OPENROUTER
                and attempt_plan.target.credential.source == CREDENTIAL_SOURCE_MANAGED
            )
            if index == 1 and config.fallback is not None:
                force_managed_wrapper = config.fallback.force_managed_wrapper
            attempt_providers.append(
                LLMProviderAttempt(
                    provider=_LazyFactoryLLMProvider(
                        factory=lambda attempt_plan=attempt_plan, force_managed_wrapper=force_managed_wrapper: _provider_from_resolved_target(
                            attempt_plan.target,
                            secrets=secrets,
                            managed_release_service=fallback_managed_release_service,
                            managed_delegate_ready=managed_delegate_ready,
                            runtime_logging=runtime_logging,
                            compatibility_settings=compatibility_settings,
                            managed_gemma_runtime=managed_gemma_runtime,
                            managed_gemma_release=managed_gemma_release,
                            qwen_low_latency_mode=qwen_low_latency_mode,
                            force_managed_wrapper=force_managed_wrapper,
                            include_selection_alias=False,
                        )
                    ),
                    start_after_ms=attempt_plan.start_after_ms,
                    start_on_primary_error=attempt_plan.start_on_primary_error,
                    log_summary=_fallback_attempt_log_summary(
                        attempt_plan.target,
                        start_after_ms=attempt_plan.start_after_ms,
                    ),
                )
            )
        base = FallbackRacingLLMProvider(
            primary=base,
            fallback=attempt_providers[1].provider,
            attempts=tuple(attempt_providers),
            fallback_timeout_ms=config.attempts[1].start_after_ms,
            loser_grace_ms=config.loser_grace_ms,
            runtime_logging=runtime_logging,
        )
    return SemaphoreLLMProvider(
        inner=base,
        semaphore=asyncio.Semaphore(config.concurrency_limit),
    )


def create_llm_provider(
    settings: AppSettings,
    *,
    secrets: SecretStore,
    managed_release_service: object | None = None,
    managed_delegate_ready: Callable[[], object] | None = None,
    runtime_logging: SessionRuntimeLoggingService | None = None,
    managed_gemma_runtime: ManagedGemmaRuntimeOwner | None = None,
    managed_gemma_release: Callable[[], Awaitable[None]] | None = None,
) -> LLMProvider:
    runtime_input = _runtime_resolution_input_from_compatibility_settings(settings)
    resolved = resolve_llm_config(runtime_input)
    return create_llm_provider_from_resolved_config(
        resolved,
        secrets=secrets,
        managed_release_service=managed_release_service,
        managed_delegate_ready=managed_delegate_ready,
        runtime_logging=runtime_logging,
        compatibility_settings=settings,
        managed_gemma_runtime=managed_gemma_runtime,
        managed_gemma_release=managed_gemma_release,
        qwen_low_latency_mode=FIXED_TRANSLATION_POLICY.fast_translation_enabled,
    )
