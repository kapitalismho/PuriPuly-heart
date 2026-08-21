from __future__ import annotations

from collections.abc import Awaitable, Callable

from puripuly_heart.app.wiring.wiring_llm_factory import create_llm_provider
from puripuly_heart.core.http_extensions import (
    HttpExtensionConfigurationError,
    HttpExtensionRegistry,
)
from puripuly_heart.core.local_translation.runtime import ManagedGemmaRuntimeOwner
from puripuly_heart.core.storage.secrets import SecretStore
from puripuly_heart.core.translation_backend import LlmTranslationBackend, TranslationBackend
from puripuly_heart.providers.extensions.http_extension_backend import (
    HttpExtensionTranslationBackend,
)


def create_translation_backend(
    settings: object,
    *,
    secrets: SecretStore,
    http_extensions: HttpExtensionRegistry,
    managed_release_service: object | None = None,
    managed_delegate_ready: Callable[[], object] | None = None,
    runtime_logging: object | None = None,
    managed_gemma_runtime: ManagedGemmaRuntimeOwner | None = None,
    managed_gemma_release: Callable[[], Awaitable[None]] | None = None,
) -> TranslationBackend:
    translation = getattr(settings, "translation")
    model = getattr(translation, "model")
    if getattr(model, "value", model) != "custom_http":
        return LlmTranslationBackend(
            create_llm_provider(
                settings,
                secrets=secrets,
                managed_release_service=managed_release_service,
                managed_delegate_ready=managed_delegate_ready,
                runtime_logging=runtime_logging,
                managed_gemma_runtime=managed_gemma_runtime,
                managed_gemma_release=managed_gemma_release,
            )
        )

    http_extension_id = getattr(translation, "http_extension_id", None)
    loaded = http_extensions.snapshot.get(http_extension_id)
    if loaded is None:
        raise HttpExtensionConfigurationError("selected HTTP extension is unavailable")
    llm_settings = getattr(settings, "llm", None)
    return HttpExtensionTranslationBackend(
        extension=loaded.definition,
        secret_store=secrets,
        concurrency_limit=getattr(llm_settings, "concurrency_limit", 5),
    )


__all__ = ["create_translation_backend"]
