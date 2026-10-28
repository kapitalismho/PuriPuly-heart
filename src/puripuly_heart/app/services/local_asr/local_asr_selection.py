from __future__ import annotations

from dataclasses import dataclass

from puripuly_heart.core.local_asr.local_asr_provisioning import LocalASRProvisioningSnapshot
from puripuly_heart.core.local_asr.local_stt_assets import (
    FUN_ASR_MLT_NANO_MODEL_ID,
    FUN_ASR_NANO_MODEL_ID,
    ORUKEET_MODEL_ID,
    PARAKEET_JAPANESE_MODEL_ID,
    REQUIRED_CPU_LOCAL_STT_MODEL_IDS,
)
from puripuly_heart.core.local_asr.local_stt_catalog import (
    LocalSTTUnsupportedLanguageError,
    local_cpu_model_supports_language,
    resolve_cpu_auto_model,
)

LOCAL_CPU_AUTO_PROVIDER = "local_cpu_auto"
LOCAL_CPU_DIRECT_MODEL_BY_PROVIDER = {
    "local_orukeet": ORUKEET_MODEL_ID,
    "local_fun_asr_nano": FUN_ASR_NANO_MODEL_ID,
    "local_fun_asr_mlt_nano": FUN_ASR_MLT_NANO_MODEL_ID,
    "local_parakeet_ja": PARAKEET_JAPANESE_MODEL_ID,
}
LOCAL_CPU_PROVIDERS = frozenset({LOCAL_CPU_AUTO_PROVIDER, *LOCAL_CPU_DIRECT_MODEL_BY_PROVIDER})


@dataclass(frozen=True, slots=True)
class LocalASRSelectionDecision:
    requested_provider: str
    effective_provider: str
    source_language: str
    model_id: str | None
    supported: bool


def required_local_asr_model_ids(provider: str) -> tuple[str, ...]:
    if provider == LOCAL_CPU_AUTO_PROVIDER:
        return REQUIRED_CPU_LOCAL_STT_MODEL_IDS
    model_id = LOCAL_CPU_DIRECT_MODEL_BY_PROVIDER.get(provider)
    return (model_id,) if model_id is not None else ()


def local_asr_status_for_provider(
    snapshot: LocalASRProvisioningSnapshot,
    provider: str,
) -> str:
    model_ids = required_local_asr_model_ids(provider)
    return snapshot.status_for(model_ids) if model_ids else "ready"


def resolve_local_asr_selection(
    provider: str,
    source_language: str,
) -> LocalASRSelectionDecision:
    if provider == LOCAL_CPU_AUTO_PROVIDER:
        try:
            model_id = resolve_cpu_auto_model(source_language)
        except LocalSTTUnsupportedLanguageError:
            model_id = None
        return LocalASRSelectionDecision(
            requested_provider=provider,
            effective_provider=provider,
            source_language=source_language,
            model_id=model_id,
            supported=model_id is not None,
        )
    model_id = LOCAL_CPU_DIRECT_MODEL_BY_PROVIDER.get(provider)
    if model_id is None:
        return LocalASRSelectionDecision(
            requested_provider=provider,
            effective_provider=provider,
            source_language=source_language,
            model_id=None,
            supported=True,
        )
    return LocalASRSelectionDecision(
        requested_provider=provider,
        effective_provider=provider,
        source_language=source_language,
        model_id=model_id,
        supported=local_cpu_model_supports_language(model_id, source_language),
    )
