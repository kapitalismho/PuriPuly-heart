from __future__ import annotations

import copy
from dataclasses import dataclass

from puripuly_heart.app.services.local_asr_selection import (
    LOCAL_CPU_AUTO_PROVIDER,
    LOCAL_CPU_DIRECT_MODEL_BY_PROVIDER,
    LOCAL_CPU_PROVIDERS,
    LOCAL_QWEN_PROVIDER,
    resolve_local_asr_selection,
)
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext


@dataclass(frozen=True, slots=True)
class ManualLocalASRFallbackState:
    self_provider: str
    peer_provider: str
    self_source_language: str
    peer_source_language: str
    cpu_auto_available: bool = True


@dataclass(frozen=True, slots=True)
class ManualLocalASRFallbackPlan:
    self_provider: str
    peer_provider: str
    fallback_channels: tuple[str, ...]
    installation_fallback: bool

    @property
    def changed(self) -> bool:
        return bool(self.fallback_channels)


class ManualLocalASRFallbackOwner:
    @staticmethod
    def state(
        settings: AppSettingsVNext,
        *,
        cpu_auto_available: bool,
    ) -> ManualLocalASRFallbackState:
        intent = settings.intent
        return ManualLocalASRFallbackState(
            self_provider=str(intent.stt.provider),
            peer_provider=str(intent.peer_stt.provider),
            self_source_language=intent.languages.source_language,
            peer_source_language=intent.languages.effective_peer_source,
            cpu_auto_available=cpu_auto_available,
        )

    @staticmethod
    def apply(
        settings: AppSettingsVNext,
        plan: ManualLocalASRFallbackPlan,
    ) -> AppSettingsVNext:
        from dataclasses import replace

        normalized = copy.deepcopy(settings)
        return replace(
            normalized,
            intent=replace(
                normalized.intent,
                stt=replace(normalized.intent.stt, provider=plan.self_provider),
                peer_stt=replace(normalized.intent.peer_stt, provider=plan.peer_provider),
            ),
        )

    def plan(
        self,
        state: ManualLocalASRFallbackState,
        *,
        channel: str | None = None,
    ) -> ManualLocalASRFallbackPlan:
        if channel not in {None, "self", "peer"}:
            raise ValueError("channel must be 'self' or 'peer'")

        self_provider = state.self_provider
        peer_provider = state.peer_provider
        scoped_self_provider = LOCAL_QWEN_PROVIDER if channel == "peer" else self_provider
        scoped_peer_provider = LOCAL_QWEN_PROVIDER if channel == "self" else peer_provider
        self_decision = resolve_local_asr_selection(
            scoped_self_provider,
            state.self_source_language,
            cpu_auto_available=state.cpu_auto_available,
        )
        peer_decision = resolve_local_asr_selection(
            scoped_peer_provider,
            state.peer_source_language,
            cpu_auto_available=state.cpu_auto_available,
        )

        fallback_channels: list[str] = []
        if channel in {None, "self"} and self_decision.fallback_applied:
            self_provider = self_decision.effective_provider
            fallback_channels.append("self")
        if channel in {None, "peer"} and peer_decision.fallback_applied:
            peer_provider = peer_decision.effective_provider
            fallback_channels.append("peer")

        installation_fallback = not state.cpu_auto_available and (
            ("self" in fallback_channels and state.self_provider == LOCAL_CPU_AUTO_PROVIDER)
            or ("peer" in fallback_channels and state.peer_provider == LOCAL_CPU_AUTO_PROVIDER)
        )
        return ManualLocalASRFallbackPlan(
            self_provider=self_provider,
            peer_provider=peer_provider,
            fallback_channels=tuple(fallback_channels),
            installation_fallback=installation_fallback,
        )

    def normalization_channels(
        self,
        *,
        current: ManualLocalASRFallbackState | None,
        pending: ManualLocalASRFallbackState,
    ) -> frozenset[str]:
        if current is None:
            return frozenset({"self", "peer"})
        channels: set[str] = set()
        if (
            current.self_provider != pending.self_provider
            and pending.self_provider in LOCAL_CPU_PROVIDERS
        ) or (
            pending.self_provider in LOCAL_CPU_DIRECT_MODEL_BY_PROVIDER
            and pending.self_provider != LOCAL_QWEN_PROVIDER
            and current.self_source_language != pending.self_source_language
        ):
            channels.add("self")
        if (
            current.peer_provider != pending.peer_provider
            and pending.peer_provider in LOCAL_CPU_PROVIDERS
        ) or (
            pending.peer_provider in LOCAL_CPU_DIRECT_MODEL_BY_PROVIDER
            and pending.peer_provider != LOCAL_QWEN_PROVIDER
            and current.peer_source_language != pending.peer_source_language
        ):
            channels.add("peer")
        return frozenset(channels)


__all__ = [
    "ManualLocalASRFallbackOwner",
    "ManualLocalASRFallbackPlan",
    "ManualLocalASRFallbackState",
]
