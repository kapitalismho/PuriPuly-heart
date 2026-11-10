from __future__ import annotations

import asyncio
from collections.abc import Callable
from uuid import uuid4

from puripuly_heart.core.error_messages import provider_failure_report
from puripuly_heart.core.language import get_language_info, get_llm_language_name
from puripuly_heart.core.runtime.provider_handle import ProviderRuntimeHandle
from puripuly_heart.core.translation_backend import (
    TranslationBackend,
    TranslationBackendRequest,
    TranslationProbePreparationRequired,
    translation_probe_scope,
)
from puripuly_heart.domain.models import Translation

TRANSLATION_PROBE_SAMPLES = {
    "ar": "لحظة واحدة، سأحضر كوبًا من الماء.",
    "bg": "Един момент, ще донеса чаша вода.",
    "ca": "Un moment, aniré a buscar un got d'aigua.",
    "cs": "Moment, přinesu si sklenici vody.",
    "da": "Et øjeblik, jeg henter et glas vand.",
    "de": "Einen Moment, ich hole ein Glas Wasser.",
    "el": "Μια στιγμή, θα φέρω ένα ποτήρι νερό.",
    "en": "One moment, I'll get a glass of water.",
    "es": "Un momento, voy a buscar un vaso de agua.",
    "et": "Üks hetk, toon klaasi vett.",
    "fi": "Hetkinen, haen lasin vettä.",
    "fr": "Un instant, je vais chercher un verre d'eau.",
    "hi": "एक पल, मैं एक गिलास पानी लेकर आता हूँ।",
    "hu": "Egy pillanat, hozok egy pohár vizet.",
    "id": "Sebentar, saya akan mengambil segelas air.",
    "it": "Un momento, vado a prendere un bicchiere d'acqua.",
    "ja": "ちょっと待って、水を一杯取ってきます。",
    "ko": "잠깐만요, 물 한 잔 가져올게요.",
    "lt": "Akimirką, atsinešiu stiklinę vandens.",
    "lv": "Vienu mirkli, atnesīšu glāzi ūdens.",
    "ms": "Sebentar, saya akan mengambil segelas air.",
    "nl": "Een moment, ik haal een glas water.",
    "no": "Et øyeblikk, jeg henter et glass vann.",
    "pl": "Chwileczkę, przyniosę szklankę wody.",
    "pt": "Um momento, vou buscar um copo de água.",
    "ro": "O clipă, aduc un pahar cu apă.",
    "ru": "Подождите, я принесу стакан воды.",
    "sk": "Moment, prinesiem si pohár vody.",
    "sv": "Ett ögonblick, jag hämtar ett glas vatten.",
    "th": "รอสักครู่ ฉันจะไปเอาน้ำมาสักแก้ว",
    "tr": "Bir dakika, bir bardak su alacağım.",
    "uk": "Зачекайте, я принесу склянку води.",
    "vi": "Chờ một chút, tôi sẽ lấy một cốc nước.",
    "zh": "等一下，我去拿杯水。",
    "zh-CN": "等一下，我去拿杯水。",
    "zh-TW": "等一下，我去拿杯水。",
}

TRANSLATION_PROBE_METADATA = {
    "translation_result": "not_started",
    "captured_generation": None,
    "current_generation": None,
    "source_language": None,
    "target_language": None,
    "target_languages": None,
    "output_delivery": "not_exercised",
    "display": "not_exercised",
}


def translation_probe_direction(
    source_language: str | None, target_languages: tuple[str, ...]
) -> dict:
    return {
        "source_language": source_language,
        "target_language": target_languages[0] if target_languages else None,
        "target_languages": list(target_languages),
    }


async def run_translation_probe(
    runtime_provider: Callable[[], ProviderRuntimeHandle | None],
    *,
    source_language: str | None,
    target_languages: tuple[str, ...],
) -> dict:
    runtime = runtime_provider()
    result = {
        **TRANSLATION_PROBE_METADATA,
        **translation_probe_direction(source_language, target_languages),
        "status": "action_required",
        "translation_result": "unavailable",
        "target_results": [
            {"target_language": target, "translation_result": "not_started"}
            for target in target_languages
        ],
    }
    if runtime is None:
        return {**result, "error": {"code": "translation_backend_unavailable"}}
    with runtime.retain_current_provider() as (backend, generation):
        result.update(captured_generation=generation, current_generation=runtime.generation)
        if not isinstance(backend, TranslationBackend):
            return {**result, "error": {"code": "translation_backend_unavailable"}}
        language = get_language_info(source_language) if source_language else None
        if language is None or not target_languages:
            return {**result, "error": {"code": "translation_direction_unavailable"}}
        for target, target_result in zip(target_languages, result["target_results"], strict=True):
            request = TranslationBackendRequest(
                utterance_id=uuid4(),
                text=TRANSLATION_PROBE_SAMPLES[language.code],
                system_prompt=(
                    f"Translate from {language.name} to {get_llm_language_name(target)}. "
                    "Return only the translation."
                ),
                source_language=source_language,
                target_language=target,
                context="",
                scene_participant_count=None,
            )
            outcome = await _run_translation_target_probe(
                runtime_provider, runtime, backend, generation, request
            )
            target_result["translation_result"] = outcome["translation_result"]
            result.update(outcome)
            if outcome["status"] != "applied":
                return result
        return result


async def _run_translation_target_probe(
    runtime_provider: Callable[[], ProviderRuntimeHandle | None],
    runtime: ProviderRuntimeHandle,
    backend: TranslationBackend,
    generation: int,
    request: TranslationBackendRequest,
) -> dict:
    result = {"status": "action_required", "translation_result": "unavailable"}
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
