from __future__ import annotations

from collections.abc import Awaitable, Callable

from puripuly_heart.app.wiring.wiring_llm_factory import (
    LlmFactoryResolvedExtras,
    create_llm_provider,
)
from puripuly_heart.config.runtime_resolution import RuntimeResolutionInput
from puripuly_heart.core.http_extensions import (
    HttpExtensionConfigurationError,
    HttpExtensionRegistry,
)
from puripuly_heart.core.local_translation.runtime import ManagedGemmaRuntimeOwner
from puripuly_heart.core.observability import ProviderObservationPort
from puripuly_heart.core.storage.secrets import SecretStore
from puripuly_heart.core.translation_backend import LlmTranslationBackend, TranslationBackend
from puripuly_heart.providers.extensions.http_extension_backend import (
    HttpExtensionTranslationBackend,
)


def create_translation_backend(
    *,
    translation_model: object,
    secrets: SecretStore,
    http_extensions: HttpExtensionRegistry,
    runtime_input: RuntimeResolutionInput | None = None,
    extras: LlmFactoryResolvedExtras | None = None,
    http_extension_id: str | None = None,
    concurrency_limit: int = 5,
    managed_release_service: object | None = None,
    managed_delegate_ready: Callable[[], object] | None = None,
    runtime_logging: ProviderObservationPort | None = None,
    managed_gemma_runtime: ManagedGemmaRuntimeOwner | None = None,
    managed_gemma_release: Callable[[], Awaitable[None]] | None = None,
) -> TranslationBackend:
    model = getattr(translation_model, "value", translation_model)
    if model != "custom_http":
        if runtime_input is None:
            raise ValueError("LLM translation backend requires runtime_input")
        return LlmTranslationBackend(
            create_llm_provider(
                runtime_input,
                secrets=secrets,
                extras=extras,
                managed_release_service=managed_release_service,
                managed_delegate_ready=managed_delegate_ready,
                runtime_logging=runtime_logging,
                managed_gemma_runtime=managed_gemma_runtime,
                managed_gemma_release=managed_gemma_release,
            )
        )

    loaded = http_extensions.snapshot.get(http_extension_id)
    if loaded is None:
        raise HttpExtensionConfigurationError("selected HTTP extension is unavailable")
    return HttpExtensionTranslationBackend(
        extension=loaded.definition,
        secret_store=secrets,
        concurrency_limit=concurrency_limit,
    )


__all__ = ["create_translation_backend"]
