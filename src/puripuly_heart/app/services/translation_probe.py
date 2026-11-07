from __future__ import annotations

import asyncio
from collections.abc import Callable
from uuid import uuid4

from puripuly_heart.core.error_messages import provider_failure_report
from puripuly_heart.core.runtime.provider_handle import ProviderRuntimeHandle
from puripuly_heart.core.translation_backend import (
    TranslationBackend,
    TranslationBackendRequest,
    TranslationProbePreparationRequired,
    translation_probe_scope,
)
from puripuly_heart.domain.models import Translation

TRANSLATION_PROBE_METADATA = {
    "translation_result": "not_started",
    "captured_generation": None,
    "current_generation": None,
    "source_language": "en",
    "target_language": "es",
    "output_delivery": "not_exercised",
    "display": "not_exercised",
}


async def run_translation_probe(
    runtime_provider: Callable[[], ProviderRuntimeHandle | None],
) -> dict:
    runtime = runtime_provider()
    result = {
        **TRANSLATION_PROBE_METADATA,
        "status": "action_required",
        "translation_result": "unavailable",
    }
    if runtime is None:
        return {**result, "error": {"code": "translation_backend_unavailable"}}
    with runtime.retain_current_provider() as (backend, generation):
        result.update(captured_generation=generation, current_generation=runtime.generation)
        if not isinstance(backend, TranslationBackend):
            return {**result, "error": {"code": "translation_backend_unavailable"}}
        request = TranslationBackendRequest(
            utterance_id=uuid4(),
            text="Hello, have a nice day.",
            system_prompt="Translate from English to Spanish. Return only the translation.",
            source_language="en",
            target_language="es",
            context="",
            scene_participant_count=None,
        )
        failure = None
        try:
            with translation_probe_scope():
                translation = await backend.translate(request)
        except asyncio.CancelledError:
            current = runtime_provider()
            return {
                **result,
                "status": "cancelled",
                "translation_result": "cancelled",
                "current_generation": current.generation if current is not None else None,
            }
        except Exception as exc:
            failure = exc
        current = runtime_provider()
        result["current_generation"] = current.generation if current is not None else None
        task = asyncio.current_task()
        if task is not None and task.cancelling():
            return {**result, "status": "cancelled", "translation_result": "cancelled"}
        if current is not runtime or not runtime.is_current_provider_generation(
            provider=backend, generation=generation
        ):
            return {
                **result,
                "status": "failed",
                "translation_result": "stale",
                "error": {"code": "stale_provider_completion"},
            }
        if isinstance(failure, TranslationProbePreparationRequired):
            return {**result, "error": {"code": "translation_preparation_required"}}
        if failure is not None:
            diagnostics = provider_failure_report(
                failure, provider="translation", operation="translation_probe"
            ).diagnostics
            return {
                **result,
                "status": "failed",
                "translation_result": "failed",
                "error": {"code": "translation_probe_failed"},
                "failure_category": diagnostics.category,
            }
        if (
            not isinstance(translation, Translation)
            or translation.utterance_id != request.utterance_id
            or not isinstance(translation.text, str)
            or not translation.text.strip()
        ):
            return {
                **result,
                "status": "failed",
                "translation_result": "failed",
                "error": {"code": "translation_probe_invalid_response"},
            }
        return {**result, "status": "applied", "translation_result": "succeeded"}
