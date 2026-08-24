from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import torch
import torchaudio

from experiments.psem_training_strategy_gate.targets import SAMPLE_RATE_HZ, WINDOW_SAMPLES

AUGMENTATION_FAMILIES = (
    "global_gain",
    "additive_non_speech_noise",
    "light_reverberation",
    "band_limitation",
    "codec_simulation",
)
AUGMENTATION_RECIPE_VERSION = "psem-waveform-augmentation-v1"


class AugmentationContractError(RuntimeError):
    pass


def _digest(key: str, field: str) -> bytes:
    return hashlib.sha256(f"{AUGMENTATION_RECIPE_VERSION}\0{key}\0{field}".encode()).digest()


def _unit(key: str, field: str) -> float:
    return int.from_bytes(_digest(key, field)[:8], "big") / float(2**64)


def _uniform(key: str, field: str, low: float, high: float) -> float:
    return low + (high - low) * _unit(key, field)


def augmentation_decision(row_key: str) -> dict[str, Any]:
    if not row_key:
        raise AugmentationContractError("augmentation row key is required")
    return {
        "recipe_version": AUGMENTATION_RECIPE_VERSION,
        "decision_key": row_key,
        "global_gain": {
            "enabled": True,
            "gain_db": round(_uniform(row_key, "gain_db", -6.0, 6.0), 6),
        },
        "additive_non_speech_noise": {
            "enabled": _unit(row_key, "noise_enabled") < 0.5,
            "snr_db": round(_uniform(row_key, "snr_db", 10.0, 30.0), 6),
            "noise_seed": int.from_bytes(_digest(row_key, "noise_seed")[:8], "big"),
        },
        "light_reverberation": {
            "enabled": _unit(row_key, "reverb_enabled") < 0.3,
            "decay_seconds": round(_uniform(row_key, "reverb_decay_seconds", 0.02, 0.12), 6),
            "reverb_seed": int.from_bytes(_digest(row_key, "reverb_seed")[:8], "big"),
        },
        "band_limitation": {
            "enabled": _unit(row_key, "band_enabled") < 0.3,
            "cutoff_hz": round(_uniform(row_key, "cutoff_hz", 3000.0, 7000.0), 3),
        },
        "codec_simulation": {
            "enabled": _unit(row_key, "codec_enabled") < 0.3,
            "quantization_bits": 8 + int(_unit(row_key, "codec_bits") * 5),
        },
    }


def validate_augmentation_decision(decision: Mapping[str, Any]) -> None:
    key = decision.get("decision_key")
    if not isinstance(key, str) or decision != augmentation_decision(key):
        raise AugmentationContractError(
            "augmentation decision is not canonical or label-independent"
        )


def _add_noise(waveform: torch.Tensor, decision: Mapping[str, Any]) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(decision["noise_seed"]) % (2**63 - 1))
    noise = torch.randn(waveform.shape, generator=generator, dtype=torch.float32).to(
        waveform.device,
        waveform.dtype,
    )
    signal_rms = waveform.square().mean().sqrt().clamp_min(1e-6)
    noise_rms = noise.square().mean().sqrt().clamp_min(1e-6)
    target_noise_rms = signal_rms / (10.0 ** (float(decision["snr_db"]) / 20.0))
    return waveform + noise * (target_noise_rms / noise_rms)


def _reverberate(waveform: torch.Tensor, decision: Mapping[str, Any]) -> torch.Tensor:
    length = SAMPLE_RATE_HZ // 10
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(decision["reverb_seed"]) % (2**63 - 1))
    noise = torch.randn(length, generator=generator, dtype=torch.float32)
    time = torch.arange(length, dtype=torch.float32) / SAMPLE_RATE_HZ
    impulse = noise * torch.exp(-time / float(decision["decay_seconds"]))
    impulse[0] += 4.0
    impulse = impulse / impulse.abs().sum().clamp_min(1e-6)
    values = torch.nn.functional.conv1d(
        waveform[None, None],
        impulse.flip(0).to(waveform.device, waveform.dtype)[None, None],
        padding=length - 1,
    )[0, 0, : waveform.numel()]
    return values


def apply_augmentation(
    waveform: torch.Tensor,
    decision: Mapping[str, Any],
) -> torch.Tensor:
    validate_augmentation_decision(decision)
    if waveform.shape != (WINDOW_SAMPLES,):
        raise AugmentationContractError("augmentation requires one complete mono window")
    result = waveform * (10.0 ** (float(decision["global_gain"]["gain_db"]) / 20.0))
    if decision["additive_non_speech_noise"]["enabled"]:
        result = _add_noise(result, decision["additive_non_speech_noise"])
    if decision["light_reverberation"]["enabled"]:
        result = _reverberate(result, decision["light_reverberation"])
    if decision["band_limitation"]["enabled"]:
        result = torchaudio.functional.lowpass_biquad(
            result,
            SAMPLE_RATE_HZ,
            float(decision["band_limitation"]["cutoff_hz"]),
        )
    if decision["codec_simulation"]["enabled"]:
        levels = 2 ** int(decision["codec_simulation"]["quantization_bits"])
        result = torch.round((result.clamp(-1.0, 1.0) + 1.0) * (levels - 1) / 2.0)
        result = result * 2.0 / (levels - 1) - 1.0
    if not bool(torch.isfinite(result).all()):
        raise AugmentationContractError("augmentation produced non-finite waveform samples")
    return result.clamp(-1.0, 1.0)


def augmentation_manifest_summary(decisions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not decisions:
        raise AugmentationContractError("augmentation manifest must not be empty")
    for decision in decisions:
        validate_augmentation_decision(decision)
    enabled_counts = {
        family: sum(bool(decision[family]["enabled"]) for decision in decisions)
        for family in AUGMENTATION_FAMILIES
    }
    return {
        "recipe_version": AUGMENTATION_RECIPE_VERSION,
        "families": list(AUGMENTATION_FAMILIES),
        "decision_count": len(decisions),
        "enabled_counts": enabled_counts,
        "whole_window_consistency": True,
        "label_fields_consulted": [],
        "synthetic_manifest": None,
        "synthetic_optimizer_batch_fraction": 0.0,
    }
