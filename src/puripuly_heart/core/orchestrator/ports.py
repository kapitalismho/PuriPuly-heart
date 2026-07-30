from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol


class HubRuntimeLoggingPort(Protocol):
    @property
    def mode(self) -> object: ...

    def emit_basic(self, message: str, *, level: int = ...) -> None: ...

    def emit_detailed(self, message: str, *, level: int = ...) -> bool: ...

    def emit_detailed_lazy(
        self,
        build_message: Callable[[], str],
        *,
        level: int = ...,
    ) -> bool: ...


def runtime_logging_mode_is_detailed(mode: object) -> bool:
    return str(getattr(mode, "value", mode)) == "detailed"


def format_basic_latency_summary(
    *,
    channel: str,
    e2e_ms: int,
) -> str:
    parts = [
        f"channel={channel}",
        f"e2e_ms={e2e_ms}",
    ]
    return f"[Basic][Latency] {' '.join(parts)}"


def format_detailed_latency_trace(
    *,
    channel: str,
    utterance_id: str,
    stage: str,
    elapsed_ms: int,
) -> str:
    return (
        f"[Detailed][Latency] channel={channel} utterance_id={utterance_id} "
        f"stage={stage} elapsed_ms={elapsed_ms}"
    )


def format_detailed_latency_breakdown(
    *,
    channel: str,
    e2e_ms: int,
    speech_end_to_stt_final_ms: int | None = None,
    stt_final_to_final_output_ms: int | None = None,
) -> str:
    parts = [
        f"channel={channel}",
        f"e2e_ms={e2e_ms}",
    ]
    if speech_end_to_stt_final_ms is not None:
        parts.append(f"speech_end_to_stt_final_ms={speech_end_to_stt_final_ms}")
    if stt_final_to_final_output_ms is not None:
        parts.append(f"stt_final_to_final_output_ms={stt_final_to_final_output_ms}")
    return f"[Detailed][LatencyBreakdown] {' '.join(parts)}"


def compute_latency_dominant_stage(
    stage_durations_ms: Mapping[str, int | None],
) -> str | None:
    safe_durations = {
        str(stage): int(duration)
        for stage, duration in stage_durations_ms.items()
        if duration is not None and int(duration) >= 0
    }
    if not safe_durations:
        return None
    return max(safe_durations, key=lambda stage: (safe_durations[stage], stage))


def format_latency_cause_metric(
    *,
    channel: str,
    provider: str,
    utterance_id: str,
    stage_durations_ms: Mapping[str, int | None],
) -> str | None:
    dominant_stage = compute_latency_dominant_stage(stage_durations_ms)
    if dominant_stage is None:
        return None
    parts = [
        "[Metric] latency_cause",
        f"channel={channel}",
        f"provider={provider}",
        f"utterance_id={utterance_id}",
        f"dominant_stage={dominant_stage}",
    ]
    for stage in sorted(stage_durations_ms):
        duration = stage_durations_ms[stage]
        if duration is None:
            continue
        parts.append(f"{stage}_ms={max(0, int(duration))}")
    return " ".join(parts)


def format_translation_ready_for_output(
    *,
    channel: str,
    utterance_id: str,
    update_id: str,
    origin_wall_clock_ms: int | None,
    session_scope: str | None,
    source_text_hash: str | None,
    source_text_len: int | None,
    logical_turn_key: str | None,
    translation_len: int,
    elapsed_ms: int | None,
) -> str:
    parts = [
        "[Detailed][Hub] translation_ready_for_output",
        f"channel={channel}",
        f"utterance_id={utterance_id}",
        f"update_id={update_id}",
        f"origin_wall_clock_ms={origin_wall_clock_ms}",
        f"session_scope={session_scope}",
        f"source_text_hash={source_text_hash}",
        f"source_text_len={source_text_len}",
        f"logical_turn_key={logical_turn_key}",
        f"translation_len={translation_len}",
    ]
    if elapsed_ms is not None:
        parts.append(f"elapsed_ms={elapsed_ms}")
    return " ".join(parts)


__all__ = [
    "HubRuntimeLoggingPort",
    "compute_latency_dominant_stage",
    "format_basic_latency_summary",
    "format_detailed_latency_breakdown",
    "format_detailed_latency_trace",
    "format_latency_cause_metric",
    "format_translation_ready_for_output",
    "runtime_logging_mode_is_detailed",
]
