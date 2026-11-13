from __future__ import annotations

from dataclasses import replace

from puripuly_heart.core.asr_worker import ASRWorkerRequestError
from puripuly_heart.core.diagnostics import exception_diagnostics, sanitize_diagnostic_fields
from puripuly_heart.core.error_messages import stt_failure_report
from puripuly_heart.core.messages import ErrorDiagnostics
from puripuly_heart.core.network_diagnostics import (
    classify_transport_error,
    transport_exception_chain,
)
from puripuly_heart.core.stt.backend import STTProviderTurnIdentity

_RECOGNITION_CAUSES = frozenset(
    {
        "buffer_exhausted",
        "cancelled",
        "closed",
        "provider_epoch_ended",
        "provider_event_stream_ended",
        "provider_idle_timeout",
        "retired",
        "pending_capacity",
        "channel_disabled",
        "dispatcher_cancelled",
        "channel_cancel_timeout",
        "gemini_receive_failed",
        "gemini_write_failed",
        "gemini_connection_ended",
        "gemini_go_away",
        "gemini_write_cancelled",
        "gemini_recognition_buffer_overflow",
        "provider_final_timeout",
        "provider_result_too_large",
        "provider_retirement_drain_timeout",
        "provider_session_lifetime_exceeded",
        "provider_stable_prefix_inconsistent",
        "provider_turn_failed_before_terminal",
        "provider_turn_identity_mismatch",
        "provider_update_sequence_disorder",
        "provider_not_ready",
        "provider_event_stream_failed",
        "provider_begin_timeout",
        "provider_send_timeout",
        "provider_seal_timeout",
        "provider_abort_timeout",
        "provider_begin_failed",
        "provider_send_failed",
        "provider_seal_failed",
        "provider_abort_failed",
        "soniox_request_failed",
        "soniox_protocol_ambiguity",
        "soniox_write_failed",
        "soniox_receive_failed",
        "soniox_keepalive_failed",
        "soniox_connection_ended",
        "soniox_stream_finished",
        "soniox_idle_authoritative_text",
        "soniox_token_buffer_overflow",
        "deepgram_protocol_failed",
        "deepgram_transport_error",
        "deepgram_connection_closed",
        "deepgram_writer_ended",
        "deepgram_close_stream_failed",
        "deepgram_finalize_ack_missing",
        "deepgram_finalize_ack_before_seal",
        "deepgram_idle_authoritative_text",
        "provider_stream_write_failed",
        "provider_stream_write_timeout",
        "provider_input_submission_failed",
        "provider_event_buffer_overflow",
        "provider_recovery_exhausted",
        "provider_readiness_timeout",
        "local_decode_timeout",
        "local_decode_unavailable",
        "local_decode_failed",
        "offline_total_timeout",
        "auth_failure",
        "compatibility_mismatch",
        "model_unavailable",
        "unreachable",
        "audio_write_failed",
        "commit_write_failed",
        "provider_protocol_error",
        "session_closed",
        "scribe_committed_transcript_missing_text",
        "scribe_connection_event_overflow",
        "scribe_keepalive_failed",
        "scribe_write_failed",
        "scribe_commit_failed",
        "scribe_connection_ended",
        "deepgram_write_failed",
        "deepgram_keepalive_failed",
        "source_discontinuity",
        "stream_input_ownership_changed",
        "stream_input_capacity_exhausted",
        "expired_before_recognition",
        "overload",
        "final_timeout",
        "connection_eof",
        "native_transcription_failed",
        "ambiguous_unkeyed_terminal",
        "qwen_write_failed",
        "qwen_keepalive_failed",
        "connect_failed",
        "receive_failed",
        "receive_eof",
        "task_start_timeout",
        "run_task_send_failed",
        "session_failed",
        "provider_task_failed",
        "task_finish_timeout",
        "scribe_auth_error",
        "scribe_quota_exceeded",
        "scribe_rate_limited",
        "scribe_transcriber_error",
        "scribe_invalid_request",
        "scribe_queue_overflow",
        "scribe_resource_exhausted",
        "scribe_session_time_limit_exceeded",
        "scribe_input_error",
        "scribe_chunk_size_exceeded",
        "scribe_insufficient_audio_activity",
        "scribe_connection_closed",
        "provider_event_delivery_failed",
        "provider_event_drain_timeout",
        "provider_cleanup_failed",
        "provider_cleanup_timeout",
        "stopped",
        "toggle_off",
    }
)


def recognition_cause(reason: str | None) -> str:
    if reason is None:
        return "none"
    category = reason.partition(":")[0]
    if category in _RECOGNITION_CAUSES:
        return category
    return "unclassified"


def retryable_transport_failure(exception: BaseException | None) -> bool:
    if exception is None:
        return False
    for cause in transport_exception_chain(exception):
        if isinstance(
            cause, (FileNotFoundError, IsADirectoryError, NotADirectoryError, PermissionError)
        ):
            return False
        diagnostics = getattr(cause, "failure_diagnostics", None)
        if isinstance(diagnostics, ErrorDiagnostics) and diagnostics.category in {
            "auth",
            "quota",
            "configuration",
            "invalid_request",
        }:
            return False
        response = getattr(cause, "response", None)
        status = getattr(cause, "status_code", None) or getattr(response, "status_code", None)
        if type(status) is int:
            return status in {408, 429, 500, 502, 503, 504}
        close = getattr(cause, "rcvd", None)
        code = getattr(close, "code", None)
        if type(code) is int:
            return code in {1001, 1006, 1011, 1012, 1013, 1014}
    return classify_transport_error(exception) in {"network", "timeout"}


def recognition_failure_diagnostics(
    exception: BaseException | ErrorDiagnostics | None,
    *,
    provider: str,
    channel: str,
    operation: str,
    reason: str,
    identity: STTProviderTurnIdentity | None = None,
) -> ErrorDiagnostics:
    cause = recognition_cause(reason)
    diagnostics = stt_failure_report(
        exception, provider=provider, channel=channel, operation=operation
    ).diagnostics
    category = diagnostics.category
    if cause in {"pending_capacity", "channel_disabled", "dispatcher_cancelled"}:
        category = "lifecycle"
    if exception is None and cause.endswith("_timeout"):
        category = "timeout"
    elif cause in {
        "deepgram_protocol_failed",
        "soniox_protocol_ambiguity",
        "scribe_committed_transcript_missing_text",
        "provider_protocol_error",
        "compatibility_mismatch",
    }:
        category = "invalid_response"
    if exception is None:
        category = {
            "scribe_auth_error": "auth",
            "scribe_quota_exceeded": "quota",
            "scribe_rate_limited": "rate_limit",
        }.get(cause, category)
    fields = {**diagnostics.fields, "recognition_cause": cause}
    fields.pop("channel", None)
    if isinstance(exception, BaseException):
        for item in transport_exception_chain(exception):
            if isinstance(item, ASRWorkerRequestError):
                native_fields, _ = sanitize_diagnostic_fields(
                    {
                        "native_classification": item.code,
                        **{
                            key: item.fields[key]
                            for key in ("native_reason", "native_status")
                            if key in item.fields
                        },
                    }
                )
                native_code = native_fields.get("native_classification")
                if native_code is not None:
                    fields.update(native_fields)
                    category = (
                        "timeout"
                        if native_code in {"heartbeat_timeout", "channel_cancel_timeout"}
                        else "lifecycle"
                    )
                break
    if identity is not None:
        fields.update(
            provider_epoch_id=identity.provider_epoch_id,
            provider_turn_id=identity.provider_turn_id,
            capture_epoch=identity.segment.capture_epoch,
        )
    return replace(
        diagnostics,
        code=(
            diagnostics.code
            if isinstance(exception, ErrorDiagnostics)
            or isinstance(getattr(exception, "failure_diagnostics", None), ErrorDiagnostics)
            else f"stt.{cause}"
        ),
        category=category,
        channel=channel if channel in {"self", "peer"} else None,
        turn_id=str(identity.segment.segment_id) if identity is not None else diagnostics.turn_id,
        generation=(
            identity.segment.activation_generation
            if identity is not None
            else diagnostics.generation
        ),
        fields=fields,
    )


def recognition_cleanup_failure(
    exception: BaseException,
    *,
    provider: str,
    channel: str | None,
    primary: ErrorDiagnostics | None = None,
) -> ErrorDiagnostics:
    diagnostics = exception_diagnostics(
        exception,
        component="provider.stt",
        operation="cleanup",
        code="stt.provider_cleanup_failed",
        fields={
            "provider": provider,
            "recognition_cause": "provider_cleanup_failed",
        },
    )
    diagnostics = replace(
        diagnostics,
        channel=channel if channel in {"self", "peer"} else None,
        cause_id=primary.diagnostic_id if primary is not None else diagnostics.cause_id,
        causes=(*diagnostics.causes, *((primary,) if primary is not None else ())),
        fields={**diagnostics.fields, "cleanup_confirmed": False},
    )
    exception.failure_diagnostics = diagnostics
    return diagnostics
