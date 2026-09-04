from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from threading import Lock
from typing import Callable, Literal, Protocol, cast

from puripuly_heart.config.vad_defaults import DEFAULT_STABLE_VAD_HANGOVER_MS

TranslationRuntimeConfigField = Literal[
    "source_language",
    "target_language",
    "self_target_languages",
    "peer_source_language",
    "peer_target_language",
    "peer_source_mode",
    "system_prompt",
    "chatbox_include_source",
    "fallback_transcript_only",
    "translation_enabled",
    "peer_translation_enabled",
    "integrated_context_enabled",
    "hangover_s",
    "peer_hangover_s",
    "context_time_window_s",
    "context_max_entries",
    "integrated_context_time_window_s",
    "integrated_context_max_entries",
    "low_latency_mode",
    "low_latency_merge_gap_ms",
    "low_latency_spec_retry_max",
    "low_latency_finalize_wait_ms",
    "low_latency_awaiting_vad_timeout_s",
]


class TranslationRuntimeConfigCategory(StrEnum):
    LANGUAGES = "languages"
    PROMPT = "prompt"
    OUTPUT = "output"
    ENABLEMENT = "enablement"
    LATENCY = "latency"
    CONTEXT = "context"
    LOW_LATENCY = "low_latency"


@dataclass(frozen=True, slots=True)
class TranslationRuntimeConfig:
    source_language: str = "ko"
    target_language: str = "en"
    self_target_languages: tuple[str, ...] = ()
    peer_source_language: str = ""
    peer_target_language: str = ""
    peer_source_mode: str = "manual"
    system_prompt: str = ""
    chatbox_include_source: bool = True
    fallback_transcript_only: bool = False
    translation_enabled: bool = True
    peer_translation_enabled: bool = False
    integrated_context_enabled: bool = False
    hangover_s: float = DEFAULT_STABLE_VAD_HANGOVER_MS / 1000.0
    peer_hangover_s: float = 0.6
    context_time_window_s: float = 30.0
    context_max_entries: int = 3
    integrated_context_time_window_s: float = 40.0
    integrated_context_max_entries: int = 4
    low_latency_mode: bool = False
    low_latency_merge_gap_ms: int = 600
    low_latency_spec_retry_max: int = 1
    low_latency_finalize_wait_ms: int = 400
    low_latency_awaiting_vad_timeout_s: float = 3.0

    def __post_init__(self) -> None:
        target_language = self.target_language.strip()
        if not target_language:
            raise ValueError("target_language must be non-empty")
        provided_targets = tuple(
            dict.fromkeys(
                language.strip()
                for language in self.self_target_languages
                if isinstance(language, str) and language.strip()
            )
        )
        if len(provided_targets) > 2:
            raise ValueError("self_target_languages supports at most two targets")
        secondary_targets = tuple(
            language for language in provided_targets[1:] if language != target_language
        )
        normalized_targets = (target_language, *secondary_targets[:1])
        object.__setattr__(self, "target_language", target_language)
        object.__setattr__(self, "self_target_languages", normalized_targets)


@dataclass(frozen=True, slots=True)
class TranslationRuntimeConfigSnapshot:
    revision: int
    value: TranslationRuntimeConfig


@dataclass(frozen=True, slots=True)
class TranslationRuntimeConfigChange:
    before: TranslationRuntimeConfigSnapshot
    after: TranslationRuntimeConfigSnapshot
    changed_fields: frozenset[TranslationRuntimeConfigField]
    categories: frozenset[TranslationRuntimeConfigCategory]

    @property
    def self_language_changed(self) -> bool:
        return (
            self.before.value.source_language,
            self.before.value.self_target_languages,
        ) != (
            self.after.value.source_language,
            self.after.value.self_target_languages,
        )

    @property
    def peer_language_changed(self) -> bool:
        before_source = self.before.value.peer_source_language or self.before.value.source_language
        before_target = self.before.value.peer_target_language or self.before.value.target_language
        after_source = self.after.value.peer_source_language or self.after.value.source_language
        after_target = self.after.value.peer_target_language or self.after.value.target_language
        return (
            before_source,
            before_target,
            self.before.value.peer_source_mode,
        ) != (
            after_source,
            after_target,
            self.after.value.peer_source_mode,
        )


class TranslationRuntimeConfigSnapshotPort(Protocol):
    def __call__(self) -> TranslationRuntimeConfigSnapshot: ...


class TranslationRuntimeConfigurationPort(Protocol):
    def snapshot(self) -> TranslationRuntimeConfigSnapshot: ...

    def replace(
        self,
        value: TranslationRuntimeConfig,
    ) -> TranslationRuntimeConfigChange: ...

    def transform(
        self,
        transformer: Callable[[TranslationRuntimeConfig], TranslationRuntimeConfig],
    ) -> TranslationRuntimeConfigChange: ...


_FIELDS_BY_CATEGORY: dict[
    TranslationRuntimeConfigCategory,
    frozenset[TranslationRuntimeConfigField],
] = {
    TranslationRuntimeConfigCategory.LANGUAGES: frozenset(
        {
            "source_language",
            "target_language",
            "self_target_languages",
            "peer_source_language",
            "peer_target_language",
            "peer_source_mode",
        }
    ),
    TranslationRuntimeConfigCategory.PROMPT: frozenset({"system_prompt"}),
    TranslationRuntimeConfigCategory.OUTPUT: frozenset(
        {"chatbox_include_source", "fallback_transcript_only"}
    ),
    TranslationRuntimeConfigCategory.ENABLEMENT: frozenset(
        {
            "translation_enabled",
            "peer_translation_enabled",
            "integrated_context_enabled",
        }
    ),
    TranslationRuntimeConfigCategory.LATENCY: frozenset({"hangover_s", "peer_hangover_s"}),
    TranslationRuntimeConfigCategory.CONTEXT: frozenset(
        {
            "context_time_window_s",
            "context_max_entries",
            "integrated_context_time_window_s",
            "integrated_context_max_entries",
        }
    ),
    TranslationRuntimeConfigCategory.LOW_LATENCY: frozenset(
        {
            "low_latency_mode",
            "low_latency_merge_gap_ms",
            "low_latency_spec_retry_max",
            "low_latency_finalize_wait_ms",
            "low_latency_awaiting_vad_timeout_s",
        }
    ),
}
_CONFIG_FIELDS = tuple(
    field_name for category_fields in _FIELDS_BY_CATEGORY.values() for field_name in category_fields
)


class TranslationRuntimeConfigurationOwner:
    __slots__ = ("_lock", "_snapshot")

    def __init__(self, initial: TranslationRuntimeConfig | None = None) -> None:
        self._lock = Lock()
        self._snapshot = TranslationRuntimeConfigSnapshot(
            revision=0,
            value=initial or TranslationRuntimeConfig(),
        )

    def snapshot(self) -> TranslationRuntimeConfigSnapshot:
        with self._lock:
            return self._snapshot

    def replace(
        self,
        value: TranslationRuntimeConfig,
    ) -> TranslationRuntimeConfigChange:
        if not isinstance(value, TranslationRuntimeConfig):
            raise TypeError("translation runtime configuration must use TranslationRuntimeConfig")
        with self._lock:
            return self._replace_locked(value)

    def transform(
        self,
        transformer: Callable[[TranslationRuntimeConfig], TranslationRuntimeConfig],
    ) -> TranslationRuntimeConfigChange:
        with self._lock:
            value = transformer(self._snapshot.value)
            if not isinstance(value, TranslationRuntimeConfig):
                raise TypeError(
                    "translation runtime configuration must use TranslationRuntimeConfig"
                )
            return self._replace_locked(value)

    def _replace_locked(
        self,
        value: TranslationRuntimeConfig,
    ) -> TranslationRuntimeConfigChange:
        before = self._snapshot
        changed_fields = frozenset(
            cast(TranslationRuntimeConfigField, field_name)
            for field_name in _CONFIG_FIELDS
            if getattr(before.value, field_name) != getattr(value, field_name)
        )
        categories = frozenset(
            category
            for category, category_fields in _FIELDS_BY_CATEGORY.items()
            if changed_fields & category_fields
        )
        after = TranslationRuntimeConfigSnapshot(
            revision=before.revision + 1,
            value=value,
        )
        self._snapshot = after
        return TranslationRuntimeConfigChange(
            before=before,
            after=after,
            changed_fields=changed_fields,
            categories=categories,
        )
