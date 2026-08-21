from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from puripuly_heart.core.clock import Clock, SystemClock
from puripuly_heart.core.orchestrator.channel_runtime import ChannelRuntime, ContextEntry
from puripuly_heart.core.orchestrator.configuration import (
    TranslationRuntimeConfig,
    TranslationRuntimeConfigSnapshotPort,
    TranslationRuntimeConfigurationOwner,
)

ContextMode = Literal["local", "integrated"]


def _default_config_snapshot_port() -> TranslationRuntimeConfigSnapshotPort:
    return TranslationRuntimeConfigurationOwner().snapshot


@dataclass(slots=True)
class ContextResolver:
    clock: Clock = SystemClock()
    config_snapshot: TranslationRuntimeConfigSnapshotPort = field(
        default_factory=_default_config_snapshot_port,
        repr=False,
    )

    def _configuration(
        self,
        configuration: TranslationRuntimeConfig | None = None,
    ) -> TranslationRuntimeConfig:
        if configuration is not None:
            return configuration
        return self.config_snapshot().value

    def get_local_entries(
        self,
        *,
        runtime: ChannelRuntime,
        source_language: str,
        target_language: str,
        configuration: TranslationRuntimeConfig | None = None,
    ) -> list[ContextEntry]:
        configuration = self._configuration(configuration)
        return runtime.get_valid_context(
            now=self.clock.now(),
            source_language=source_language,
            target_language=target_language,
            time_window_s=configuration.context_time_window_s,
            max_entries=configuration.context_max_entries,
        )

    def format_local(self, entries: list[ContextEntry]) -> str:
        return self._format_entries(entries)

    def resolve_local(
        self,
        *,
        runtime: ChannelRuntime,
        source_language: str,
        target_language: str,
        configuration: TranslationRuntimeConfig | None = None,
    ) -> tuple[str, ContextMode]:
        entries = self.get_local_entries(
            runtime=runtime,
            source_language=source_language,
            target_language=target_language,
            configuration=configuration,
        )
        return self.format_local(entries), "local"

    def resolve_for_request(
        self,
        *,
        runtime: ChannelRuntime,
        other_runtime: ChannelRuntime,
        requested_mode: ContextMode,
        peer_translation_enabled: bool,
        source_language: str,
        target_language: str,
        other_source_language: str | None = None,
        other_target_language: str | None = None,
        configuration: TranslationRuntimeConfig | None = None,
    ) -> tuple[str, ContextMode]:
        configuration = self._configuration(configuration)
        if requested_mode != "integrated" or not peer_translation_enabled:
            return self.resolve_local(
                runtime=runtime,
                source_language=source_language,
                target_language=target_language,
                configuration=configuration,
            )
        integrated_entries = self._get_integrated_entries(
            runtime=runtime,
            other_runtime=other_runtime,
            source_language=source_language,
            target_language=target_language,
            other_source_language=other_source_language,
            other_target_language=other_target_language,
            configuration=configuration,
        )
        if not any(channel_runtime.channel == "peer" for channel_runtime, _ in integrated_entries):
            return self.resolve_local(
                runtime=runtime,
                source_language=source_language,
                target_language=target_language,
                configuration=configuration,
            )
        return self.format_integrated(integrated_entries), "integrated"

    def _format_entries(self, entries: list[ContextEntry]) -> str:
        if not entries:
            return ""
        return "\n".join(self._format_entry(entry) for entry in entries)

    def _format_entry(self, entry: ContextEntry) -> str:
        label = "peer" if entry.channel == "peer" else "self"
        return f'- [{label}] "{entry.text}"'

    def format_integrated(self, entries: list[tuple[ChannelRuntime, ContextEntry]]) -> str:
        if not entries:
            return ""
        return "\n".join(self._format_entry(entry) for _, entry in entries)

    def _get_integrated_entries(
        self,
        *,
        runtime: ChannelRuntime,
        other_runtime: ChannelRuntime,
        source_language: str,
        target_language: str,
        other_source_language: str | None = None,
        other_target_language: str | None = None,
        configuration: TranslationRuntimeConfig | None = None,
    ) -> list[tuple[ChannelRuntime, ContextEntry]]:
        configuration = self._configuration(configuration)
        time_window_s = configuration.integrated_context_time_window_s
        max_entries = configuration.integrated_context_max_entries
        combined: list[tuple[ChannelRuntime, ContextEntry]] = []
        other_source_language = (
            source_language if other_source_language is None else other_source_language
        )
        other_target_language = (
            target_language if other_target_language is None else other_target_language
        )
        for channel_runtime, entry_source_language, entry_target_language in (
            (runtime, source_language, target_language),
            (other_runtime, other_source_language, other_target_language),
        ):
            for entry in channel_runtime.get_valid_context(
                now=self.clock.now(),
                source_language=entry_source_language,
                target_language=entry_target_language,
                time_window_s=time_window_s,
                max_entries=max_entries,
            ):
                combined.append((channel_runtime, entry))
        combined.sort(key=lambda item: item[1].timestamp)
        if max_entries > 0:
            return combined[-max_entries:]
        return combined
