from __future__ import annotations

from collections.abc import Callable

from puripuly_heart.app.ports.managed_gemma_translation import (
    ManagedGemmaTranslationSelection,
)
from puripuly_heart.core.local_translation.runtime import ManagedGemmaRuntimeOwner
from puripuly_heart.core.orchestrator.translation_request import (
    render_translation_system_prompt,
)
from puripuly_heart.providers.llm.managed_gemma import HttpxManagedGemmaTransport


def managed_gemma_selection(
    settings: object,
) -> ManagedGemmaTranslationSelection:
    translation = getattr(settings, "translation")
    model = getattr(translation, "model")
    if getattr(model, "value", model) != "managed_gemma":
        raise ValueError("managed Gemma selection requires the managed Gemma model")
    connection = getattr(translation, "connection")
    backend = getattr(connection, "value", connection)
    if backend not in {"cpu", "gpu"}:
        raise ValueError("managed Gemma connection must be CPU or GPU")
    languages = getattr(settings, "languages")
    source_language = getattr(languages, "source_language")
    target_language = getattr(languages, "target_language")
    system_prompt = render_translation_system_prompt(
        getattr(settings, "system_prompt"),
        source_language=source_language,
        target_language=target_language,
    )
    return ManagedGemmaTranslationSelection(
        backend=backend,
        source_language=source_language,
        target_language=target_language,
        system_prompt=system_prompt,
    )


def create_managed_gemma_runtime(
    *,
    log_sink: Callable[[str, int], None] | None = None,
) -> ManagedGemmaRuntimeOwner:
    return ManagedGemmaRuntimeOwner(
        transport_factory=lambda base_url: HttpxManagedGemmaTransport(base_url),
        log_sink=log_sink,
    )


__all__ = ["create_managed_gemma_runtime", "managed_gemma_selection"]
