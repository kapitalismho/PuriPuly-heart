from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from puripuly_heart.core.diagnostics import exception_diagnostics
from .local_stt_assets import (
    FUN_ASR_MLT_NANO_MODEL_ID,
    FUN_ASR_NANO_MODEL_ID,
    GIGAAM_MODEL_ID,
    GRANITE_MODEL_ID,
    PARAKEET_JAPANESE_MODEL_ID,
    REQUIRED_CPU_LOCAL_STT_MODEL_IDS,
    LocalSTTAssetManifest,
    LocalSTTInstallState,
    default_local_stt_model_root,
    inspect_local_stt_install_state,
    load_local_stt_asset_manifest,
)

GRANITE_SUPPORTED_LANGUAGE_CODES = frozenset({"en", "fr", "de", "es", "pt", "it", "ja"})
GIGAAM_SUPPORTED_LANGUAGE_CODES = frozenset({"ru"})
FUN_ASR_NANO_SUPPORTED_LANGUAGE_CODES = frozenset({"zh"})
FUN_ASR_MLT_NANO_SUPPORTED_LANGUAGE_CODES = frozenset(
    {
        "zh",
        "en",
        "yue",
        "ja",
        "ko",
        "vi",
        "id",
        "th",
        "ms",
        "tl",
        "ar",
        "hi",
        "bg",
        "hr",
        "cs",
        "da",
        "nl",
        "et",
        "fi",
        "el",
        "hu",
        "ga",
        "lv",
        "lt",
        "mt",
        "pl",
        "pt",
        "ro",
        "sk",
        "sl",
        "sv",
    }
)
PARAKEET_JAPANESE_SUPPORTED_LANGUAGE_CODES = frozenset({"ja"})

_CPU_AUTO_MODEL_PRIORITY = (
    FUN_ASR_NANO_MODEL_ID,
    PARAKEET_JAPANESE_MODEL_ID,
    GRANITE_MODEL_ID,
    GIGAAM_MODEL_ID,
    FUN_ASR_MLT_NANO_MODEL_ID,
)


class LocalSTTUnsupportedLanguageError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class LocalCPUModelContract:
    model_id: str
    supported_language_codes: frozenset[str]

    def supports_language(self, language_code: str) -> bool:
        return canonical_local_stt_language_code(language_code) in self.supported_language_codes


@dataclass(frozen=True, slots=True)
class LocalCPUModelInstall:
    model_id: str
    state: LocalSTTInstallState


@dataclass(frozen=True, slots=True)
class LocalCPUInstallSnapshot:
    models: tuple[LocalCPUModelInstall, ...]

    @property
    def cpu_auto_available(self) -> bool:
        required_model_ids = frozenset(REQUIRED_CPU_LOCAL_STT_MODEL_IDS)
        actual_model_ids = tuple(model.model_id for model in self.models)
        return (
            len(actual_model_ids) == len(required_model_ids)
            and frozenset(actual_model_ids) == required_model_ids
            and all(
                model.state.status == "ready"
                and model.state.installed_manifest is not None
                and getattr(model.state.installed_manifest, "model_id", None) == model.model_id
                for model in self.models
            )
        )

    def state_for(self, model_id: str) -> LocalSTTInstallState:
        for model in self.models:
            if model.model_id == model_id:
                return model.state
        raise KeyError(model_id)


class LocalCPUAutoUnavailableError(RuntimeError):
    def __init__(self, snapshot: LocalCPUInstallSnapshot) -> None:
        self.snapshot = snapshot
        states = ", ".join(f"{model.model_id}={model.state.status}" for model in snapshot.models)
        super().__init__(f"CPU Auto requires all local CPU models to be valid: {states}")


LOCAL_CPU_MODEL_CONTRACTS = {
    GRANITE_MODEL_ID: LocalCPUModelContract(
        model_id=GRANITE_MODEL_ID,
        supported_language_codes=GRANITE_SUPPORTED_LANGUAGE_CODES,
    ),
    GIGAAM_MODEL_ID: LocalCPUModelContract(
        model_id=GIGAAM_MODEL_ID,
        supported_language_codes=GIGAAM_SUPPORTED_LANGUAGE_CODES,
    ),
    FUN_ASR_NANO_MODEL_ID: LocalCPUModelContract(
        model_id=FUN_ASR_NANO_MODEL_ID,
        supported_language_codes=FUN_ASR_NANO_SUPPORTED_LANGUAGE_CODES,
    ),
    FUN_ASR_MLT_NANO_MODEL_ID: LocalCPUModelContract(
        model_id=FUN_ASR_MLT_NANO_MODEL_ID,
        supported_language_codes=FUN_ASR_MLT_NANO_SUPPORTED_LANGUAGE_CODES,
    ),
    PARAKEET_JAPANESE_MODEL_ID: LocalCPUModelContract(
        model_id=PARAKEET_JAPANESE_MODEL_ID,
        supported_language_codes=PARAKEET_JAPANESE_SUPPORTED_LANGUAGE_CODES,
    ),
}


def canonical_local_stt_language_code(language_code: str) -> str:
    normalized = language_code.strip().replace("_", "-").lower()
    if not normalized:
        return ""
    return normalized.split("-", 1)[0]


def local_cpu_model_supports_language(model_id: str, language_code: str) -> bool:
    try:
        contract = LOCAL_CPU_MODEL_CONTRACTS[model_id]
    except KeyError as exc:
        raise KeyError(model_id) from exc
    return contract.supports_language(language_code)


def resolve_cpu_auto_model(language_code: str) -> str:
    canonical_code = canonical_local_stt_language_code(language_code)
    for model_id in _CPU_AUTO_MODEL_PRIORITY:
        if canonical_code in LOCAL_CPU_MODEL_CONTRACTS[model_id].supported_language_codes:
            return model_id
    raise LocalSTTUnsupportedLanguageError(
        f"CPU Auto does not support source language: {language_code}"
    )


def inspect_local_cpu_model_installs(
    model_ids: tuple[str, ...],
    model_root: Path | None = None,
    *,
    manifests: Mapping[str, LocalSTTAssetManifest] | None = None,
    verify_checksums: bool = True,
) -> LocalCPUInstallSnapshot:
    resolved_root = model_root or default_local_stt_model_root()
    installs: list[LocalCPUModelInstall] = []
    for model_id in model_ids:
        if model_id not in LOCAL_CPU_MODEL_CONTRACTS:
            raise KeyError(model_id)
        manifest = (
            manifests[model_id]
            if manifests is not None
            else load_local_stt_asset_manifest(model_id)
        )
        if manifest.model_id != model_id:
            installs.append(
                LocalCPUModelInstall(
                    model_id=model_id,
                    state=LocalSTTInstallState(
                        status="invalid",
                        failure_diagnostics=exception_diagnostics(
                            ValueError("Local STT manifest identity mismatch"),
                            component="model_assets",
                            operation="inspection",
                            code="local_asr.invalid_assets",
                            fields={"model_id": model_id},
                        ),
                    ),
                )
            )
            continue
        installs.append(
            LocalCPUModelInstall(
                model_id=model_id,
                state=inspect_local_stt_install_state(
                    resolved_root / manifest.install_dirname,
                    manifest=manifest,
                    verify_checksums=verify_checksums,
                ),
            )
        )
    return LocalCPUInstallSnapshot(models=tuple(installs))


def inspect_required_cpu_model_installs(
    model_root: Path | None = None,
    *,
    manifests: Mapping[str, LocalSTTAssetManifest] | None = None,
    verify_checksums: bool = True,
) -> LocalCPUInstallSnapshot:
    return inspect_local_cpu_model_installs(
        REQUIRED_CPU_LOCAL_STT_MODEL_IDS,
        model_root,
        manifests=manifests,
        verify_checksums=verify_checksums,
    )
