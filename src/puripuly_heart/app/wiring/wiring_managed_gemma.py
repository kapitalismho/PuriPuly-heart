from __future__ import annotations

from collections.abc import Callable

from puripuly_heart.app.ports.managed_gemma_translation import (
    ManagedGemmaTranslationSelection,
)
from puripuly_heart.app.services.managed_gemma_translation import (
    ManagedGemmaTranslationOwner,
)
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.core.local_translation.assets import GEMMA_12B_MODEL_ID, GEMMA_MODEL_ID
from puripuly_heart.core.local_translation.devices import resolve_llama_vulkan_device
from puripuly_heart.core.local_translation.prefix_cache import (
    GemmaPrefixCache,
    default_gemma_prefix_cache_dir,
)
from puripuly_heart.core.local_translation.runtime import ManagedGemmaRuntimeOwner
from puripuly_heart.core.orchestrator.translation_request import (
    render_translation_system_prompt,
)
from puripuly_heart.providers.llm.managed_gemma import HttpxManagedGemmaTransport


def managed_gemma_selection(
    settings: AppSettingsVNext,
) -> ManagedGemmaTranslationSelection:
    translation = settings.intent.translation
    model_value = translation.model
    if model_value == "managed_gemma_12b":
        backend = "gpu"
        model_id = GEMMA_12B_MODEL_ID
    elif model_value == "managed_gemma":
        backend = translation.connection
        if backend not in {"cpu", "gpu"}:
            raise ValueError("managed Gemma connection must be CPU or GPU")
        model_id = GEMMA_MODEL_ID
    else:
        raise ValueError("managed Gemma selection requires the managed Gemma model")
    languages = settings.intent.languages
    source_language = languages.source_language
    target_language = languages.target_language
    system_prompt = render_translation_system_prompt(
        settings.intent.prompts.system_prompt,
        source_language=source_language,
        target_language=target_language,
    )
    return ManagedGemmaTranslationSelection(
        backend=backend,
        source_language=source_language,
        target_language=target_language,
        system_prompt=system_prompt,
        model_id=model_id,
        vulkan_device=resolve_llama_vulkan_device(translation.gpu_device_id),
    )


def managed_gemma_translation_desired(
    *,
    translation_enabled: bool,
    peer_translation_enabled: bool,
) -> bool:
    return bool(translation_enabled or peer_translation_enabled)


async def sync_managed_gemma_demand(
    *,
    managed_gemma: ManagedGemmaTranslationOwner | None,
    settings: AppSettingsVNext | None,
    desired: bool,
) -> None:
    if managed_gemma is None:
        return
    if not desired:
        await managed_gemma.deactivate(linger=True)
        return
    if settings is None:
        return
    model = settings.intent.translation.model
    if model not in {"managed_gemma", "managed_gemma_12b"}:
        return
    await managed_gemma.prepare(managed_gemma_selection(settings))


async def noop_managed_gemma_release() -> None:
    return None


def create_managed_gemma_runtime(
    *,
    log_sink: Callable[[str, int], None] | None = None,
) -> ManagedGemmaRuntimeOwner:
    return ManagedGemmaRuntimeOwner(
        transport_factory=lambda base_url: HttpxManagedGemmaTransport(base_url),
        log_sink=log_sink,
        prefix_cache=GemmaPrefixCache(default_gemma_prefix_cache_dir()),
    )


__all__ = [
    "create_managed_gemma_runtime",
    "managed_gemma_selection",
    "managed_gemma_translation_desired",
    "noop_managed_gemma_release",
    "sync_managed_gemma_demand",
]
