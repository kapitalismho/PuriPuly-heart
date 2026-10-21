from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

from puripuly_heart.core.messages import ErrorDiagnostics
from puripuly_heart.core.output.models import OutputRoutingDecision


class TranslationRuntimeLoggingPort(Protocol):

    def emit_basic(self, message: str, *, level: int = ...) -> None: ...

    def emit_failure(
        self,
        diagnostics: ErrorDiagnostics,
        *,
        severity: str = "error",
        disposition: str = "failed",
        stage: str | None = None,
        exception: BaseException | None = None,
    ) -> dict: ...

    def emit_diagnostic(self, message: str, *, level: int = ...) -> bool: ...

    def emit_translation_latency(self, message: str) -> bool: ...

    def emit_diagnostic_lazy(
        self,
        build_message: Callable[[], str],
        *,
        level: int = ...,
    ) -> bool: ...
    def record_output_routing_decision(self, decision: OutputRoutingDecision) -> None: ...

    def record_conversation_observation(
        self,
        *,
        utterance_id: str,
        speaker_channel: str,
        transcript_text: str | None,
        translation_text: str | None,
        source_language: str | None,
        target_language: str | None,
        metadata: Mapping[str, str | int | float | bool | None] | None = None,
        correlation_id: str | None = None,
    ) -> None: ...

    def record_request_context(
        self,
        *,
        utterance_id: str,
        context_texts: Sequence[str],
        segment_index: int | None = None,
    ) -> None: ...


def format_basic_latency_summary(
    *,
    channel: str,
    endpoint: str,
    elapsed_ms: int,
    estimated: bool = False,
) -> str:
    metric_name = f"last_speech_to_{endpoint}_ms"
    parts = [
        f"channel={channel}",
        f"endpoint={endpoint}",
        f"{metric_name}={elapsed_ms}",
    ]
    if estimated:
        parts.append("estimated=true")
    return f"[Basic][Latency] {' '.join(parts)}"


__all__ = [
    "TranslationRuntimeLoggingPort",
    "format_basic_latency_summary",
]
