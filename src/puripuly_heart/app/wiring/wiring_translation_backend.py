from __future__ import annotations

from collections.abc import Callable

from puripuly_heart.app.wiring.wiring_llm_factory import create_llm_provider
from puripuly_heart.core.storage.secrets import SecretStore
from puripuly_heart.core.translation_backend import TranslationBackend
from puripuly_heart.core.translation_extensions import (
    HttpExtensionTranslationBackend,
    TranslationExtensionRegistry,
)


def create_translation_backend(
    settings: object,
    *,
    secrets: SecretStore,
    translation_extensions: TranslationExtensionRegistry,
    managed_release_service: object | None = None,
    managed_delegate_ready: Callable[[], object] | None = None,
    runtime_logging: object | None = None,
) -> TranslationBackend | object:
    translation = getattr(settings, "translation")
    model = getattr(translation, "model")
    if getattr(model, "value", model) != "custom_http":
        return create_llm_provider(
            settings,
            secrets=secrets,
            managed_release_service=managed_release_service,
            managed_delegate_ready=managed_delegate_ready,
            runtime_logging=runtime_logging,
        )

    extension_id = getattr(translation, "extension_id", None)
    loaded = translation_extensions.snapshot.get(extension_id)
    if loaded is None:
        raise ValueError("selected translation extension is unavailable")
    llm_settings = getattr(settings, "llm", None)
    return HttpExtensionTranslationBackend(
        extension=loaded.definition,
        secret_store=secrets,
        concurrency_limit=getattr(llm_settings, "concurrency_limit", 5),
    )


__all__ = ["create_translation_backend"]
