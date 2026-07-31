from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .local_stt_assets import (
    LOCAL_STT_MODEL_ID,
    PARAKEET_JAPANESE_MODEL_ID,
    PARAKEET_V3_MODEL_ID,
    REQUIRED_CPU_LOCAL_STT_MODEL_IDS,
    LocalSTTAssetManifest,
    LocalSTTInstallState,
    default_local_stt_model_root,
    inspect_local_stt_install_state,
    load_local_stt_asset_manifest,
)

PARAKEET_V3_SUPPORTED_LANGUAGE_CODES = frozenset(
    {
        "bg",
        "hr",
        "cs",
        "da",
        "nl",
        "en",
        "et",
        "fi",
        "fr",
        "de",
        "el",
        "hu",
        "it",
        "lv",
        "lt",
        "mt",
        "pl",
        "pt",
        "ro",
        "sk",
        "sl",
        "es",
        "sv",
        "ru",
        "uk",
    }
)
PARAKEET_JAPANESE_SUPPORTED_LANGUAGE_CODES = frozenset({"ja"})
QWEN_06B_SUPPORTED_LANGUAGE_CODES = frozenset(
    {
        "zh",
        "en",
        "yue",
        "ar",
        "de",
        "fr",
        "es",
        "pt",
        "id",
        "it",
        "ko",
        "ru",
        "th",
        "vi",
        "ja",
        "tr",
        "hi",
        "ms",
        "nl",
        "sv",
        "da",
        "fi",
        "pl",
        "cs",
        "fil",
        "fa",
        "el",
        "hu",
        "mk",
        "ro",
    }
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
    PARAKEET_V3_MODEL_ID: LocalCPUModelContract(
        model_id=PARAKEET_V3_MODEL_ID,
        supported_language_codes=PARAKEET_V3_SUPPORTED_LANGUAGE_CODES,
    ),
    PARAKEET_JAPANESE_MODEL_ID: LocalCPUModelContract(
        model_id=PARAKEET_JAPANESE_MODEL_ID,
        supported_language_codes=PARAKEET_JAPANESE_SUPPORTED_LANGUAGE_CODES,
    ),
    LOCAL_STT_MODEL_ID: LocalCPUModelContract(
        model_id=LOCAL_STT_MODEL_ID,
        supported_language_codes=QWEN_06B_SUPPORTED_LANGUAGE_CODES,
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
    if canonical_code == "ja":
        return PARAKEET_JAPANESE_MODEL_ID
    if canonical_code in PARAKEET_V3_SUPPORTED_LANGUAGE_CODES:
        return PARAKEET_V3_MODEL_ID
    if canonical_code in QWEN_06B_SUPPORTED_LANGUAGE_CODES:
        return LOCAL_STT_MODEL_ID
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
                        error_message="local STT manifest model_id does not match required model identity",
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
