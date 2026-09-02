from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Generic, Protocol, TypeVar


class SettingsSecretKey(StrEnum):
    GOOGLE_API_KEY = "google_api_key"
    OPENROUTER_API_KEY = "openrouter_api_key"
    DEEPSEEK_API_KEY = "deepseek_api_key"
    CEREBRAS_API_KEY = "cerebras_api_key"
    DEEPGRAM_API_KEY = "deepgram_api_key"
    GEMINI_TRANSCRIBE_API_KEY = "gemini_transcribe_api_key"
    ELEVENLABS_SCRIBE_API_KEY = "elevenlabs_scribe_api_key"
    SONIOX_API_KEY = "soniox_api_key"
    LOCAL_LLM_API_KEY = "local_llm_api_key"
    CUSTOM_STT_API_KEY = "custom_stt_api_key"
    ALIBABA_API_KEY_BEIJING = "alibaba_api_key_beijing"
    ALIBABA_API_KEY_SINGAPORE = "alibaba_api_key_singapore"


@dataclass(frozen=True, slots=True, repr=False)
class SettingsSecretSnapshot:
    google_api_key: str | None = None
    openrouter_api_key: str | None = None
    deepseek_api_key: str | None = None
    cerebras_api_key: str | None = None
    deepgram_api_key: str | None = None
    gemini_transcribe_api_key: str | None = None
    elevenlabs_scribe_api_key: str | None = None
    soniox_api_key: str | None = None
    local_llm_api_key: str | None = None
    custom_stt_api_key: str | None = None
    alibaba_api_key_beijing: str | None = None
    alibaba_api_key_singapore: str | None = None


@dataclass(frozen=True, slots=True, repr=False)
class SettingsSecretValuesSnapshot:
    values: tuple[tuple[str, str], ...] = ()

    def get(self, key: str) -> str | None:
        return next((value for candidate, value in self.values if candidate == key), None)


@dataclass(frozen=True, slots=True, repr=False)
class OpenRouterPkceSecretSnapshot:
    openrouter_api_key: str | None = None
    deepseek_api_key: str | None = None
    cerebras_api_key: str | None = None


SecretSnapshotT = TypeVar("SecretSnapshotT")


@dataclass(frozen=True, slots=True)
class SettingsSecretLoadResult(Generic[SecretSnapshotT]):
    snapshot: SecretSnapshotT | None
    error_message: str | None = None
    read_error: Exception | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True, repr=False)
class SettingsSecretMutation:
    key: SettingsSecretKey
    value: str


@dataclass(frozen=True, slots=True)
class SettingsSecretMutationResult:
    succeeded: bool
    error_type: str | None = None


class SettingsSecretsPort(Protocol):
    def load(self) -> SettingsSecretLoadResult[SettingsSecretSnapshot]: ...

    def load_values(
        self,
        keys: tuple[str, ...],
    ) -> SettingsSecretLoadResult[SettingsSecretValuesSnapshot]: ...

    def load_openrouter_pkce(
        self,
    ) -> SettingsSecretLoadResult[OpenRouterPkceSecretSnapshot]: ...

    def mutate(self, mutation: SettingsSecretMutation) -> SettingsSecretMutationResult: ...


class SettingsSecretStorePort(Protocol):
    def get(self, key: str) -> str | None: ...

    def set(self, key: str, value: str) -> None: ...

    def delete(self, key: str) -> None: ...


__all__ = [
    "OpenRouterPkceSecretSnapshot",
    "SettingsSecretLoadResult",
    "SettingsSecretKey",
    "SettingsSecretMutation",
    "SettingsSecretMutationResult",
    "SettingsSecretSnapshot",
    "SettingsSecretStorePort",
    "SettingsSecretValuesSnapshot",
    "SettingsSecretsPort",
]
