from dataclasses import dataclass

from puripuly_heart.core.local_asr.local_stt_assets import (
    FUN_ASR_MLT_NANO_MODEL_ID,
    FUN_ASR_NANO_MODEL_ID,
    LOCAL_QWEN_GPU_MODEL_ID,
    ORUKEET_MODEL_ID,
    PARAKEET_JAPANESE_MODEL_ID,
    load_local_stt_asset_manifest,
)
from puripuly_heart.core.local_translation.assets import e4b_gemma_spec, gemma_26b_spec


@dataclass(frozen=True, slots=True)
class ModelAssetChoice:
    model_id: str
    name: str
    description_key: str
    purpose: str
    size_bytes: int
    importable: bool = False
    required_files: tuple[str, ...] = ()


def model_asset_choices() -> tuple[ModelAssetChoice, ...]:
    choices = []
    for model_id, name, key in (
        (ORUKEET_MODEL_ID, "Orukeet Q8_0", "orukeet"),
        (FUN_ASR_NANO_MODEL_ID, "Fun-ASR-Nano-2512 Q8_0", "fun_asr_nano"),
        (FUN_ASR_MLT_NANO_MODEL_ID, "Fun-ASR-MLT-Nano-2512 Q8_0", "fun_asr_mlt_nano"),
        (LOCAL_QWEN_GPU_MODEL_ID, "Qwen3 ASR 1.7B Q6_K", "qwen_gpu"),
        (PARAKEET_JAPANESE_MODEL_ID, "Parakeet TDT/CTC 0.6B Japanese INT8", "parakeet_ja"),
    ):
        manifest = load_local_stt_asset_manifest(model_id)
        choices.append(
            ModelAssetChoice(
                model_id,
                name,
                "model_assets.description." + key,
                "asr",
                sum(a.size_bytes or 0 for a in manifest.files),
            )
        )
    for spec, name, key in (
        (e4b_gemma_spec(), "Gemma 4 E4B QAT UD Q4 K XL", "gemma_e4b"),
        (gemma_26b_spec(), "Gemma 4 26B A4B QAT UD Q4 K XL", "gemma_26b"),
    ):
        choices.append(
            ModelAssetChoice(
                spec.model_id,
                name,
                "model_assets.description." + key,
                "translation",
                sum(a.size_bytes for a in spec.assets),
                importable=True,
                required_files=tuple(asset.filename for asset in spec.assets),
            )
        )
    return tuple(choices)
