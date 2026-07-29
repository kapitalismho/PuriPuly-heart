from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

SettingsT = TypeVar("SettingsT")


@dataclass(frozen=True, slots=True)
class SettingsRuntimeState:
    runtime_available: bool
    self_stt_desired: bool
    self_stt_available: bool
    peer_stt_desired: bool
    peer_stt_available: bool
    qwen_llm_desired: bool
    llm_available: bool


@dataclass(frozen=True, slots=True)
class SettingsRuntimeTransition(Generic[SettingsT]):
    settings: SettingsT
    previous_settings: SettingsT | None
    previous_locale: str
    previous_overlay_enabled: bool
    previous_self_signature: object | None
    previous_peer_signature: object | None
    previous_peer_translation_enabled: bool
    previous_peer_activation_requested: bool
    source_language_changed: bool
    target_language_changed: bool
    effective_peer_source_changed: bool
    effective_peer_target_changed: bool
    peer_source_language_changed: bool
    peer_target_language_changed: bool
    peer_source_mode_changed: bool
    desktop_runtime_controls: tuple[dict[str, object], ...]


class SettingsRuntimeEffectsPort(Protocol[SettingsT]):
    async def preserve_before_replace(self, settings: SettingsT) -> None: ...

    def capture_runtime_signatures(self) -> None: ...

    async def prepare(
        self,
        current_settings: SettingsT | None,
        next_settings: SettingsT,
    ) -> SettingsRuntimeTransition[SettingsT]: ...

    def activate_before_persist(
        self,
        transition: SettingsRuntimeTransition[SettingsT],
    ) -> None: ...

    async def prepare_overlay_persistence(
        self,
        previous_settings: SettingsT,
        next_settings: SettingsT,
    ) -> None: ...

    def restore_memory(self, settings: SettingsT) -> None: ...

    def sync_signatures(self, settings: SettingsT) -> None: ...

    def state(self, settings: SettingsT) -> SettingsRuntimeState: ...

    async def apply_after_persist(
        self,
        transition: SettingsRuntimeTransition[SettingsT],
        *,
        strict_runtime_errors: bool,
        reload_settings_view: bool,
    ) -> None: ...


__all__ = [
    "SettingsRuntimeEffectsPort",
    "SettingsRuntimeState",
    "SettingsRuntimeTransition",
]
