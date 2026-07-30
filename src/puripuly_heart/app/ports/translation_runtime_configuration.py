from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TranslationRuntimeSettingsValues:
    source_language: str
    target_language: str
    peer_source_language: str
    peer_target_language: str
    system_prompt: str
    chatbox_include_source: bool
    hangover_s: float
    peer_hangover_s: float
    low_latency_mode: bool
    low_latency_merge_gap_ms: int
    low_latency_spec_retry_max: int
