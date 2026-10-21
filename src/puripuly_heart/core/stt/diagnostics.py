from __future__ import annotations

from dataclasses import replace

from puripuly_heart.core.diagnostic_validation import stt_startup_failure_code
from puripuly_heart.core.error_messages import stt_failure_report
from puripuly_heart.core.gpu_worker import GpuWorkerRequestError
from puripuly_heart.core.messages import ErrorDiagnostics
from puripuly_heart.core.network_diagnostics import transport_exception_chain
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
            if isinstance(item, GpuWorkerRequestError):
                native_code = stt_startup_failure_code(item.code)
                if native_code not in {"none", "unclassified"}:
                    fields["native_classification"] = native_code
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
