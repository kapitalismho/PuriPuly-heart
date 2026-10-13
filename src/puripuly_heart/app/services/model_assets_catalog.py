from dataclasses import dataclass

from puripuly_heart.core.local_stt_assets import (
    LOCAL_QWEN_GPU_MODEL_ID,
    LOCAL_STT_MODEL_ID,
    PARAKEET_JAPANESE_MODEL_ID,
    PARAKEET_V3_MODEL_ID,
    load_local_stt_asset_manifest,
)
from puripuly_heart.core.local_translation.assets import GEMMA_MODEL_ID, e4b_gemma_spec


@dataclass(frozen=True, slots=True)
class ModelAssetChoice:
    model_id: str
    name: str
    description_key: str
    purpose: str
    size_bytes: int
    sources: tuple[str, ...]
    default_source: str


def model_asset_choices() -> tuple[ModelAssetChoice, ...]:
    choices = []
    for model_id, name, key in (
        (LOCAL_STT_MODEL_ID, "Qwen3 ASR 0.6B · CPU", "qwen_cpu"),
        (PARAKEET_V3_MODEL_ID, "Parakeet TDT 0.6B v3 · CPU", "parakeet"),
        (PARAKEET_JAPANESE_MODEL_ID, "Parakeet TDT/CTC 0.6B · CPU", "parakeet_ja"),
        (LOCAL_QWEN_GPU_MODEL_ID, "Qwen3 ASR 1.7B Q6_K · GPU", "qwen_gpu"),
    ):
        manifest = load_local_stt_asset_manifest(model_id)
        sources = ("official", "mirror")
        if "modelscope" in manifest.sources:
            sources = ("modelscope_only",) + sources
        choices.append(
            ModelAssetChoice(
                model_id,
                name,
                "model_assets.description." + key,
                "asr",
                sum(a.size_bytes or 0 for a in manifest.files),
                sources,
                sources[0],
            )
        )
    spec = e4b_gemma_spec()
    choices.append(
        ModelAssetChoice(
            GEMMA_MODEL_ID,
            "Gemma E4B + MTP · GGUF",
            "model_assets.description.gemma",
            "translation",
            sum(a.size_bytes for a in spec.assets),
            ("official", "mirror", "local"),
            "official",
        )
    )
    return tuple(choices)
