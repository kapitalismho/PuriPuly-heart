from __future__ import annotations

import asyncio
import builtins
import json
import logging
import re
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from datetime import datetime
from itertools import islice
from threading import RLock
from uuid import uuid4

from puripuly_heart.config.provider_values import STTProviderName
from puripuly_heart.core.diagnostic_codes import is_supported_diagnostic_code
from puripuly_heart.core.diagnostic_validation import stt_startup_failure_code
from puripuly_heart.core.lifecycle import LIFECYCLE_SHUTDOWN_PHASE_ORDER
from puripuly_heart.core.local_stt_assets import (
    LOCAL_QWEN_GPU_MODEL_ID,
    REQUIRED_CPU_LOCAL_STT_MODEL_IDS,
)
from puripuly_heart.core.messages import DIAGNOSTIC_CATEGORIES, SEVERITIES, ErrorDiagnostics
from puripuly_heart.core.network_diagnostics import TRANSPORT_ENUM_FIELDS, TRANSPORT_NUMERIC_FIELDS
from puripuly_heart.core.openrouter.managed_operation_identity import is_valid_operation_id
from puripuly_heart.core.qq_auth_diagnostics import ENUM_FIELDS as QQ_ENUM_FIELDS
from puripuly_heart.core.qq_auth_diagnostics import STAGES as QQ_STAGES

DIAGNOSTIC_SCHEMA_VERSION = 1
DIAGNOSTIC_RECORD_MAX_BYTES = 4096
DIAGNOSTIC_HISTORY_MAX_RECORDS = 256
DIAGNOSTIC_HISTORY_MAX_BYTES = 512 * 1024
DIAGNOSTIC_QUERY_MAX_RECORDS = 100
DIAGNOSTIC_RECORD_ATTRIBUTE = "_puripuly_heart_failure"
DIAGNOSTIC_PROJECTION_ATTRIBUTE = "_puripuly_heart_failure_projection"
DISPOSITIONS = frozenset(
    {
        "failed",
        "degraded",
        "recovered",
        "cancelled",
        "superseded",
        "disabled",
        "unknown",
        "action_required",
    }
)
_KNOWN_ID_SENTINELS = {"provider_epoch_id": frozenset({"admission"})}
_ID = re.compile(
    r"(?:[a-z][a-z0-9_-]{0,31}-)?"
    r"(?:[0-9a-fA-F]{8,64}|[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12})\Z"
)
_EXCEPTION_CLASSES = frozenset(
    name
    for name in dir(builtins)
    if isinstance(getattr(builtins, name), type)
    and issubclass(getattr(builtins, name), BaseException)
)
_NUMERIC_FIELDS = frozenset(
    {
        "errno",
        "winerror",
        "exit_code",
        "attempts",
        "count",
        "generation",
        "runtime_generation",
        "activation_generation",
        "turn_generation",
        "target_index",
        "provider_epoch",
        "process_generation",
        "diagnostic_dropped",
        "reliable_delivery_failures",
        "rejected_records",
        "local_send_completed",
        "local_send_requested",
        "status_code",
        "retry_after_ms",
        "cleanup_failure_count",
        "secondary_count",
        "task_sequence",
    }
) | frozenset(TRANSPORT_NUMERIC_FIELDS)
_NUMERIC_FIELDS = _NUMERIC_FIELDS | frozenset({"pending_startups", "process_id", "monotonic_ns"})
_NUMERIC_FIELDS = _NUMERIC_FIELDS | frozenset({"queue_capacity", "dropped_count"})
_NUMERIC_FIELDS = _NUMERIC_FIELDS | frozenset({"status_polls"})
_NUMERIC_FIELDS = _NUMERIC_FIELDS | frozenset(
    {
        "logging_delivery_dropped",
        "logging_delivery_failures",
        "delivery_failures",
        "oversized_rejected",
        "failed_records",
        "listener_failures",
        "abandoned_records",
    }
)
_NUMERIC_FIELDS = _NUMERIC_FIELDS | frozenset({"logging_oversized_rejected"})
_NUMERIC_FIELDS = _NUMERIC_FIELDS | frozenset({"dropped"})
_NUMERIC_FIELDS = _NUMERIC_FIELDS | frozenset({"pending_temporary_directories"})
_BOOLEAN_FIELDS = frozenset(
    {
        "settings_commit_succeeded",
        "secret_write_succeeded",
        "secret_restore_succeeded",
        "previous_secret_existed",
        "remote_revoked",
        "local_credentials_removed",
        "fallback_persisted",
        "observation_succeeded",
        "process_present",
        "notification_delivered",
        "cleanup_confirmed",
        "cleanup_complete",
        "timed_out",
        "abandoned",
        "close_complete",
        "recovery_pending",
        "warmup_usable",
        "rollback_succeeded",
        "managed_secret_present",
        "token_present",
    }
)
_BOOLEAN_FIELDS = _BOOLEAN_FIELDS | frozenset(
    {"remote_active", "broker_issue_succeeded", "settings_values_accepted", "diagnostics_present"}
)
_BOOLEAN_FIELDS = _BOOLEAN_FIELDS | frozenset({"drain_complete"})
_ID_FIELDS = frozenset(
    {
        "publication_id",
        "overlay_instance_id",
        "provider_epoch_id",
        "provider_turn_id",
        "parent_utterance_id",
        "capture_epoch",
        "worker_session_id",
        "attempt_id",
        "temporary_cleanup_diagnostic_id",
    }
)
_ENUM_FIELDS = {
    **TRANSPORT_ENUM_FIELDS,
    **{key: frozenset(values) for key, values in QQ_ENUM_FIELDS.items()},
    "channel": frozenset({"self", "peer"}),
    "model_id": frozenset((*REQUIRED_CPU_LOCAL_STT_MODEL_IDS, LOCAL_QWEN_GPU_MODEL_ID)),
    "destination": frozenset(
        {
            "conversation_feed",
            "self_chatbox",
            "subtitle_overlay",
            "system_disclosure_chatbox",
            "dashboard",
        }
    ),
    "remote_delivery": frozenset({"unacknowledged"}),
    "evidence_state": frozenset({"retained", "incomplete", "unavailable", "unknown"}),
    "action": frozenset({"set", "clear", "sign_in", "refresh", "sign_out"}),
    "cleanup_stage": frozenset(
        {
            "process_terminate",
            "process_wait",
            "writer_close",
            "server_close",
            "task_close",
            "temp_cleanup",
            "reader_cleanup",
            "window_close",
            "diagnostic_close",
            "parent_monitor_close",
        }
    ),
    "outcome": frozenset(
        {
            "final",
            "empty",
            "suppressed",
            "failed",
            "degraded",
            "cancelled",
            "expired",
            "submitted",
            "applied",
            "recovered",
            "superseded",
            "stale",
            "disabled",
            "completed",
            "interrupted",
        }
    ),
    "provider": frozenset(item.value for item in STTProviderName)
    | frozenset(
        {
            "soniox",
            "deepgram",
            "qwen",
            "qwen_asr",
            "alibaba",
            "openai",
            "openrouter",
            "gemini",
            "deepseek",
            "local",
            "local_llm",
            "local_asr",
            "cpu",
            "gpu",
            "whisper",
            "faster_whisper",
            "parakeet",
            "sherpa",
            "sherpa_onnx",
            "custom",
            "managed",
            "managed_gemma",
            "chatgpt",
            "custom_http",
            "llm",
            "stt",
            "unknown",
        }
    ),
    "publication_kind": frozenset(
        {"self_utterance", "peer_subtitle", "system_disclosure", "conversation_feed"}
    ),
    "observed_state": frozenset(
        {
            "running",
            "ready",
            "building",
            "failed",
            "dormant",
            "inactive",
            "stopped",
            "admission_pending",
            "target_resolving",
            "provider_pending",
            "starting",
            "stopping",
            "faulted",
        }
    ),
    "native_classification": frozenset(
        {
            "terminal_failed",
            "pose_unavailable",
            "recovering",
            "due",
            "intentional_hidden",
            "no_drawable_content",
            "healthy_idle",
        }
    ),
    "target": frozenset({"desktop", "steamvr"}),
    "surface": frozenset(
        {
            "translation_provider",
            "stt_language_audio",
            "overlay_osc_output",
            "ui_prompt_clipboard_state",
        }
    ),
    "verification_scope": frozenset({"key_probe"}),
    "recognition_readiness": frozenset({"unobserved"}),
    "policy_disposition": frozenset({"excluded", "transient"}),
    "limit_source": frozenset(
        {"openrouter_credits", "openrouter_key_limit", "openrouter_in_flight_budget"}
    ),
    "limit_reason": frozenset({"in_flight_budget_exhausted", "weight_exceeds_budget"}),
    "owner_name": frozenset(
        {
            "ApplicationRuntime",
            "GithubStarPromptOwner",
            "ManualTypingOwner",
            "ClipboardRuntime",
            "VrchatOscPresenceProbeOwner",
            "SelfCaptureSessionOwner",
            "VrcMicReceiverRuntime",
            "OverlayApplicationOwner",
            "PeerApplicationOwner",
            "OpenRouterPkceFlowOwner",
            "LocalASRProvisioningRuntime",
            "AudioTestSessionOwner",
            "PeerCaptureSessionOwner",
            "VrchatSceneService",
            "ApplicationRuntimeLoggingOwner",
            "ManagedAuthOwner",
            "TranslationEnableOwner",
            "ManagedUsageOwner",
            "RuntimePipelineLauncher",
            "SelfTranslationIngress",
            "PeerTranslationIngress",
            "TranslationTurnLifecycleOwner",
            "OutputRuntime",
            "SelfChannelRuntime",
            "PeerChannelRuntime",
            "LocalASRProviderRuntimeOwner",
            "ProviderRuntimeHandle:llm",
            "ManagedGemmaTranslationOwner",
            "VrchatOscUdpSender",
            "ManagedOpenRouterReleaseService",
            "RuntimeLoggingService",
            "ApplicationShutdownCoordinator",
            "ApplicationControlHost",
        }
    ),
    "callback_name": frozenset(
        {
            "freeze_ingress",
            "stop_ingress",
            "release",
            "close",
            "cancel",
            "close_background_tasks",
            "close_failed_resources",
            "emit_final_shutdown_diagnostics",
            "close_after_producers_stop",
            "emit_diagnostic",
            "freeze",
            "drain_and_close",
        }
    ),
    "delivery_ack_status": frozenset(QQ_ENUM_FIELDS["ack_status"]),
    "operation_status": frozenset(QQ_ENUM_FIELDS["managed_operation_state"])
    | {"UNKNOWN_OPERATION"},
    "client_action": frozenset(
        {"wait", "retry_authorized", "acknowledge_delivery", "action_required"}
    ),
    "other_source": frozenset({"qq", "discord"}),
    "pending_kind": frozenset({"operation", "delivery_ack"}),
    "local_secret_key": frozenset(
        {
            "openrouter_managed_api_key",
            "openrouter_managed_china_api_key",
            "openrouter_managed_discord_api_key",
        }
    ),
    "refresh_kind": frozenset({"status", "trial_usage"}),
    "api_status": frozenset(
        {
            "OK",
            "CANCELLED",
            "UNKNOWN",
            "INVALID_ARGUMENT",
            "DEADLINE_EXCEEDED",
            "NOT_FOUND",
            "ALREADY_EXISTS",
            "PERMISSION_DENIED",
            "UNAUTHENTICATED",
            "RESOURCE_EXHAUSTED",
            "FAILED_PRECONDITION",
            "ABORTED",
            "OUT_OF_RANGE",
            "UNIMPLEMENTED",
            "INTERNAL",
            "UNAVAILABLE",
            "DATA_LOSS",
            "GO_AWAY",
            "none",
            "unclassified",
        }
    ),
    "message_kind": frozenset(
        {"none", "go_away", "connection_closed", "timeout", "tls", "validation", "other"}
    ),
}
_COMPONENTS = frozenset(
    {
        "unclassified",
        "application_control",
        "application_runtime_logging",
        "provider.stt",
        "provider.llm",
        "provider.openrouter.auth",
        "provider.runtime",
        "capture.self",
        "capture.peer",
        "settings_repository",
        "settings_mutation",
        "settings_application",
        "provider_application",
        "secret_settings_transaction",
        "settings_secrets",
        "chatgpt_account",
        "chatgpt_session",
        "vrchat_scene",
        "application_startup",
        "application_shutdown",
        "gui_settings_queue",
        "application_notification",
        "output_runtime",
        "osc_control",
        "gpu_worker",
        "overlay_process",
        "overlay_bridge",
        "provider_verifier",
        "openrouter_pkce_handoff",
        "runtime_logging",
        "managed_key_delivery_ack",
        "managed_openrouter_broker_client",
        "managed_account",
        "managed_status_refresh",
        "managed_usage",
        "managed_auth",
        "translation.turn",
        "managed_connection_auth",
        "qq_managed_auth",
        "manual_typing",
        "local_asr_provisioning",
    }
)
_BOUNDARIES = (
    frozenset(
        {
            "emit",
            "dispatch",
            "diagnostic",
            "python_log",
            "readiness",
            "begin",
            "send",
            "seal",
            "abort",
            "receive",
            "protocol",
            "decode",
            "event_loop",
            "turn_terminal",
            "recovery",
            "provider_build",
            "provider_warmup",
            "provider_handoff",
            "provider_replace",
            "ingress",
            "source_open",
            "vad",
            "source_read",
            "probe",
            "translate",
            "installation",
            "save",
            "settings_save",
            "rollback",
            "runtime_apply",
            "runtime_apply_interrupted",
            "snapshot_notification",
            "runtime_notification",
            "commit_notification",
            "failure_notification",
            "dashboard_notification",
            "compensate_local_asr",
            "resync_committed_runtime",
            "snapshot_secret",
            "set_secret",
            "clear_secret",
            "restore_secret",
            "verify_provider_secret",
            "verify_transient_key",
            "commit_provider_verification",
            "store_creation",
            "secret_read",
            "secret_mutation",
            "legacy_secret_migration",
            "loopback_bind",
            "authorization_request",
            "browser_open",
            "authorization_notification",
            "authorization_wait",
            "token_exchange",
            "identity_verification",
            "credential_store",
            "sign_in",
            "refresh",
            "refresh_request",
            "refresh_response",
            "refresh_store",
            "refresh_rejected",
            "local_credential_delete",
            "remote_revoke",
            "sign_out",
            "secret_observation",
            "process_enumeration",
            "process_identity",
            "process_watch",
            "scene_log_read",
            "watch_close",
            "settings_fallback",
            "startup_presentation",
            "startup_prepare",
            "queued_mutation",
            "osc_state_notification",
            "publish",
            "close",
            "observe",
            "discovery",
            "publish_state",
            "process_lifecycle",
            "native_presentation",
            "process_failure",
            "reader_cleanup",
            "diagnostic_delivery",
            "transport",
            "destination_delivery",
            "routing_observer",
            "ui_delivery_close",
            "ui_bridge_close",
            "chatbox_backlog_close",
            "chatbox_typing_close",
            "automatic_discovery",
            "startup_state",
            "delta_state",
            "full_state",
            "snapshot_state",
            "startup",
            "connected",
            "runtime",
            "cleanup",
            "process_terminate",
            "process_wait",
            "writer_close",
            "server_close",
            "task_close",
            "temp_cleanup",
            "native_terminal",
            "native_health",
            "reverse_ingress",
            "native_diagnostic_delivery",
            "failed",
            "shutdown",
            "off",
            "apply_translation_provider",
            "apply_stt_language_audio",
            "apply_overlay_osc_output",
            "apply_ui_prompt_clipboard_state",
            "app.stop",
            "capture.set",
            "translation.set",
            "text.submit",
            "audio.test",
            "audio.target.set",
            "audio.retry",
            "settings.apply",
            "provider.apply",
            "models.install",
            "models.prepare",
            "models.cancel",
            "gemma.prepare",
            "gemma.cancel",
            "models.retry",
            "gpu.discover",
            "secrets.set",
            "secrets.delete",
            "secrets.verify",
            "auth.login",
            "auth.logout",
            "overlay.set",
            "overlay.lock",
            "overlay.size",
            "overlay.position.reset",
            "overlay.calibrate",
            "typing_state",
            "typing_reason",
            "typing_clear",
            "chatbox_page",
            "ui_event_delivery",
            "settings_replace",
            "settings_write",
            "settings_validate",
            "settings_restore",
            "temporary_cleanup",
            "load",
            "parse_failed",
            "migration_failed",
            "backup_failed",
            "verify_api_key",
            "key_verification",
            "child_execution",
            "provider_generation",
            "restore_memory",
            "startup_runtime",
            "startup_events",
            "apply_provider_runtime",
            "apply_translation_provider_runtime",
            "apply_order21_order22_order24_provider_runtime",
            "apply_stt_language_audio_runtime",
            "apply_stt_language_audio_provider_runtime",
            "apply_overlay_osc_output_runtime",
            "apply_ui_prompt_clipboard_state_runtime",
            "apply_order22_order23_order24_full_draft_save",
            "apply_stt_language_audio_full_draft_save",
            "apply_overlay_osc_output_full_draft_save",
            "apply_ui_prompt_clipboard_state_full_draft_save",
            "managed_authentication",
            "qq_authentication",
            "managed_recovery",
            "managed_operation_status",
            "managed_operation_resume",
            "managed_operation_clear",
            "local_settings_commit",
            "persist_managed_operation",
            "delivery_ack_pending_persist",
            "auth_notification",
            "local_identity_preflight",
            "discord_auth",
            "broker_issue",
            "local_secret_write",
            "local_claim_persist",
            "manual_submit",
            "manual_completion_wait",
            "manual_typing_output",
            "manual_typing_idle_schedule",
            "manual_typing_idle_timeout",
            "listener_start",
            "browser_launch",
            "callback_wait",
            "code_exchange",
            "settings_commit",
            "flow",
            "broker_assertion",
            "delivery_ack_token_store",
            "remote_delivery_ack_recovery",
            "remote_managed_operation_recovery",
            "read_delivery_ack_token",
            "recover_managed_key_delivery",
            "acknowledge_managed_key_delivery",
            "clear_delivery_ack_token",
            "restore_delivery_ack_token",
            "clear_delivery_ack_metadata",
            "store_delivery_ack_token",
            "write",
            "enqueue",
            "file_write",
            "file_delivery",
            "queue_overflow",
            "acknowledge_qq_managed_key_delivery",
            "assert_qq_managed_identity",
            "authenticate_qq_managed_identity",
            "authorize_managed_connection",
            "commit_settings",
            "discord_callback",
            "discord_issue",
            "discord_start",
            "ensure_managed_operation",
            "ensure_qq_managed_operation",
            "issue",
            "issue_managed_connection",
            "managed_key_delivery_ack",
            "managed_operation_recovery",
            "persist_managed_claim_source",
            "persist_qq_managed_auth",
            "persist_qq_pending_delivery_ack",
            "persist_recovered_managed_claim_source",
            "preflight_managed_identity",
            "qq_assert",
            "recover_managed_operation",
            "recover_qq_managed_key_delivery",
            "recover_qq_managed_operation",
            "resolve_openrouter_credentials",
            "set_managed_secret",
            "set_qq_managed_secret",
            "snapshot_qq_managed_secret",
            "start_discord_auth",
            "store_qq_delivery_ack_token",
            "validate_settings_values",
            "other_source_pending",
            "remote_delivery_ack",
            "secret_snapshot",
            "secret_write",
            "validate_settings_values_after_broker",
            "writer_close",
            "create_store",
            "managed_release_close",
            "managed_release_create",
            "managed_status_refresh_failed",
            "managed_status_refresh_schedule_failed",
            "managed_usage_refresh",
            "managed_authentication",
            "admission",
            "child_creation",
            "rejection",
            "parent_execution",
            "output_submit",
            "turn_generation",
            "retirement",
            "activation",
        }
    )
    | frozenset(LIFECYCLE_SHUTDOWN_PHASE_ORDER)
    | QQ_STAGES
)
_NATIVE_REASONS = frozenset(
    {
        "manifest_invalid",
        "contract_mismatch",
        "bridge_auth_failed",
        "steamvr_not_installed",
        "steamvr_not_running",
        "hmd_not_found",
        "openvr_init_failed",
        "renderer_init_failed",
        "gpu_readiness_late",
        "gpu_readiness_cancelled",
        "gpu_query_failed",
        "gpu_stalled",
        "bridge_failed",
        "render_failed",
        "openvr_failed",
        "runtime_disconnected",
        "stopped",
        "unknown",
        "startup_timeout",
        "spawn_failed",
        "unsupported_binary",
        "shutdown_cleanup_failed",
        "termination_unconfirmed",
        "reverse_control_capacity",
        "native_owner_unresponsive",
        "native_acceptance_timeout",
        "missing_executable",
        "invalid_manifest",
        "window_configuration_failed",
        "runtime_control_invalid",
        "runtime_crashed",
        "startup_cancelled",
        "shutdown_terminal_unavailable",
        "shutdown_forced",
        "shutdown_not_acknowledged",
        "runtime_exit_nonzero",
        "listener_bind_failed",
        "window_reveal_lost",
        "window_observation_failed",
        "window_visibility_unstable",
        "window_bounds_failed",
        "window_native_ready_failed",
        "window_identity_failed",
    }
)
_ENUM_FIELDS.update(
    {
        "stage": _BOUNDARIES,
        "phase": _BOUNDARIES,
        "primary_failure_reason": _NATIVE_REASONS,
        "cleanup_failure_reason": _NATIVE_REASONS,
        "cleanup_stage": frozenset({"process_terminate", "cleanup"}),
    }
)
_NULLABLE_FIELDS = (
    _NUMERIC_FIELDS
    | _BOOLEAN_FIELDS
    | _ID_FIELDS
    | _ENUM_FIELDS.keys()
    | {"provider_code", "ws_close_code", "api_code", "managed_operation_id"}
)
_operation_context: ContextVar[tuple[str, str, str, list[bool], object] | None] = ContextVar(
    "diagnostic_operation", default=None
)


def emit_failure(
    logger: logging.Logger,
    diagnostics: ErrorDiagnostics,
    *,
    severity: str = "error",
    disposition: str = "failed",
    stage: str | None = None,
    exception: BaseException | None = None,
) -> None:
    """Emit owned typed meaning; the installed application ingress normalizes it."""
    logger.log(
        (
            logging.INFO
            if severity == "info"
            else logging.WARNING if severity == "warning" else logging.ERROR
        ),
        "[Failure] structured_failure",
        extra={DIAGNOSTIC_RECORD_ATTRIBUTE: (diagnostics, severity, disposition, stage, exception)},
    )


def safe_exception_fields(exception: BaseException) -> dict[str, str | int | bool]:
    """Only builtin class relationships and native numeric codes; never inspect messages."""
    fields: dict[str, str | int | bool] = {}
    current: BaseException | None = exception
    seen: set[int] = set()
    for index in range(5):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        name = type(current).__name__
        fields["exception_class" if index == 0 else f"cause_class_{index}"] = (
            name
            if type(current).__module__ == "builtins" and name in _EXCEPTION_CLASSES
            else "unclassified"
        )
        if isinstance(current, OSError):
            for key in ("errno", "winerror"):
                value = getattr(current, key, None)
                if type(value) is int and -(2**31) <= value < 2**31 and key not in fields:
                    fields[key] = value
        current = current.__cause__ or (
            None if current.__suppress_context__ else current.__context__
        )
    if current is not None and len(seen) == 5:
        fields["cause_chain_truncated"] = True
    return fields


@contextmanager
def diagnostic_operation(instance_id: str, operation_id: str, request_id: str):
    active = [True]
    try:
        owner_task = asyncio.current_task()
    except RuntimeError:
        owner_task = None
    token = _operation_context.set((instance_id, operation_id, request_id, active, owner_task))
    try:
        yield
    finally:
        _operation_context.reset(token)
        active[0] = False


def _active_operation(context) -> bool:
    if context is None or not context[3][0]:
        return False
    try:
        current_task = asyncio.current_task()
    except RuntimeError:
        current_task = None
    return current_task is None or current_task is context[4]


def current_diagnostic_operation() -> dict[str, str | None]:
    """Capture at owner admission, then bind explicitly to that source generation."""
    context = _operation_context.get()
    if not _active_operation(context):
        return {"operation_id": None, "request_id": None}
    return {"operation_id": _identity(context[1]), "request_id": _identity(context[2])}


def _identity(value: object) -> str | None:
    return value if type(value) is str and _ID.fullmatch(value) else None


def _label(value: object, default: str | None = None) -> str | None:
    if type(value) is str and value in _BOUNDARIES:
        return value
    return default


def sanitize_diagnostic_fields(
    values: Mapping[str, object],
    *,
    exception: BaseException | None = None,
) -> tuple[dict[str, object], int]:
    fields: dict[str, object] = {}
    candidate = dict(islice(values.items(), 32))
    rejected = max(0, len(values) - 32)
    if isinstance(exception, BaseException):
        candidate.update(safe_exception_fields(exception))
    elif exception is not None:
        rejected += 1
    for key, value in candidate.items():
        accepted = False
        if value is None and key in _NULLABLE_FIELDS:
            fields[key] = None
            continue
        if key in {"provider_code", "ws_close_code", "api_code"}:
            accepted = type(value) is int and 0 <= value <= (
                999999 if key == "provider_code" else 4999
            )
        elif key in _NUMERIC_FIELDS:
            accepted = type(value) is int and (
                -(2**31) <= value < 2**32
                if key in {"errno", "os_errno", "winerror", "exit_code"}
                else 0 <= value < 2**64
            )
        elif key in _BOOLEAN_FIELDS or key == "cause_chain_truncated":
            accepted = value is None or type(value) is bool
        elif key in _ID_FIELDS:
            accepted = _identity(value) is not None or (
                key == "capture_epoch" and type(value) is int and 0 <= value < 2**31
            )
        elif key == "managed_operation_id":
            accepted = is_valid_operation_id(value)
        elif key in _ENUM_FIELDS:
            accepted = type(value) is str and value in _ENUM_FIELDS[key]
        elif key == "recognition_cause" and type(value) is str:
            from puripuly_heart.core.stt.diagnostics import recognition_cause

            accepted = value in {"none", "unclassified"} or recognition_cause(value) == value
        elif key in {
            "exception_class",
            "exception_type",
            "cause_class_1",
            "cause_class_2",
            "cause_class_3",
            "cause_class_4",
        }:
            accepted = value == "unclassified" or (
                type(value) is str and value in _EXCEPTION_CLASSES
            )
        if (
            key == "native_classification"
            and type(value) is str
            and stt_startup_failure_code(value) == value
        ):
            accepted = True
        if key in _KNOWN_ID_SENTINELS and type(value) is str and value in _KNOWN_ID_SENTINELS[key]:
            accepted = True
        if accepted:
            fields[key] = value
        else:
            rejected += 1
    return fields, rejected


def normalize_diagnostic(
    diagnostics: ErrorDiagnostics,
    *,
    instance_id: str | None,
    severity: str,
    disposition: str,
    stage: str | None = None,
    exception: BaseException | None = None,
    file_evidence: str = "unknown",
) -> dict:
    context = _operation_context.get()
    source_bound = (
        diagnostics.channel is not None
        or diagnostics.turn_id is not None
        or diagnostics.generation is not None
        or any(
            key in diagnostics.fields
            for key in (
                "provider_epoch_id",
                "provider_turn_id",
                "publication_id",
                "capture_epoch",
                "activation_generation",
            )
        )
    )
    inherit = (
        context
        if not source_bound and _active_operation(context) and context[0] == instance_id
        else None
    )
    fields, rejected = sanitize_diagnostic_fields(diagnostics.fields, exception=exception)
    rejected += (
        diagnostics.rejected_fields
        if type(diagnostics.rejected_fields) is int and 0 <= diagnostics.rejected_fields < 2**31
        else 1
    )
    diagnostic_id = _identity(diagnostics.diagnostic_id)
    if diagnostic_id is None:
        diagnostic_id = uuid4().hex
        rejected += 1
    try:
        occurred = datetime.fromisoformat(diagnostics.occurred_at)
        if occurred.utcoffset() is None or occurred.utcoffset().total_seconds() != 0:
            raise ValueError
        occurred_at = occurred.isoformat(timespec="milliseconds")
    except TypeError, ValueError, AttributeError:
        occurred_at = None
        rejected += int(diagnostics.occurred_at is not None)
    component = (
        diagnostics.component
        if type(diagnostics.component) is str and diagnostics.component in _COMPONENTS
        else "unclassified"
    )
    operation = _label(diagnostics.operation)
    boundary = _label(stage or diagnostics.fields.get("stage") or diagnostics.fields.get("phase"))
    rejected += int(component != diagnostics.component)
    rejected += int(diagnostics.operation is not None and operation is None)
    rejected += int(stage is not None and boundary is None)
    code_allowed = is_supported_diagnostic_code(diagnostics.code, _NATIVE_REASONS)
    rejected += int(diagnostics.code is not None and not code_allowed)
    identities = {
        key: _identity(getattr(diagnostics, key))
        for key in ("cause_id", "operation_id", "request_id", "turn_id")
    }
    rejected += sum(
        getattr(diagnostics, key) is not None and value is None for key, value in identities.items()
    )
    channel = (
        diagnostics.channel
        if type(diagnostics.channel) is str and diagnostics.channel in {"self", "peer"}
        else None
    )
    generation = (
        diagnostics.generation
        if type(diagnostics.generation) is int and 0 <= diagnostics.generation < 2**31
        else None
    )
    status_code = (
        diagnostics.status_code
        if type(diagnostics.status_code) is int and 100 <= diagnostics.status_code <= 599
        else None
    )
    retry_after_ms = (
        diagnostics.retry_after_ms
        if type(diagnostics.retry_after_ms) is int and 0 <= diagnostics.retry_after_ms < 2**31
        else None
    )
    rejected += sum(
        raw is not None and safe is None
        for raw, safe in (
            (diagnostics.channel, channel),
            (diagnostics.generation, generation),
            (diagnostics.status_code, status_code),
            (diagnostics.retry_after_ms, retry_after_ms),
        )
    )
    record = {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "diagnostic_id": diagnostic_id,
        "instance_id": instance_id,
        "occurred_at": occurred_at,
        "component": component,
        "operation": operation,
        "stage": boundary,
        "severity": severity if type(severity) is str and severity in SEVERITIES else "error",
        "disposition": (
            disposition if type(disposition) is str and disposition in DISPOSITIONS else "unknown"
        ),
        "code": diagnostics.code if code_allowed else "unclassified_failure",
        "category": (
            diagnostics.category
            if type(diagnostics.category) is str and diagnostics.category in DIAGNOSTIC_CATEGORIES
            else "unknown"
        ),
        "cause_id": identities["cause_id"],
        "operation_id": identities["operation_id"]
        or (inherit[1] if inherit and diagnostics.operation_id is None else None),
        "request_id": identities["request_id"]
        or (_identity(inherit[2]) if inherit and diagnostics.request_id is None else None),
        "channel": channel if channel is not None else fields.get("channel"),
        "turn_id": identities["turn_id"],
        "generation": (
            generation
            if generation is not None
            else fields.get("activation_generation", fields.get("runtime_generation"))
        ),
        "status_code": status_code,
        "retry_after_ms": retry_after_ms,
        "fields": fields,
        "file_evidence": (
            file_evidence if file_evidence in {"unknown", "unavailable"} else "unknown"
        ),
        "detail": {
            "state": (
                "truncated"
                if diagnostics.detail_truncated
                else (
                    "rejected"
                    if rejected
                    else (
                        "available"
                        if any(value is not None for value in fields.values())
                        or diagnostics.status_code is not None
                        else "unavailable"
                    )
                )
            ),
            "rejected_fields": rejected,
            "truncated": bool(diagnostics.detail_truncated),
        },
    }
    if len(json.dumps(record, separators=(",", ":")).encode()) + 64 > DIAGNOSTIC_RECORD_MAX_BYTES:
        record["fields"] = {}
        record["detail"] = {"state": "truncated", "rejected_fields": rejected, "truncated": True}
    return record


def sanitize_diagnostics(diagnostics: ErrorDiagnostics) -> ErrorDiagnostics:
    """Use the same ingress policy for typed owner results before their projection."""
    record = normalize_diagnostic(
        diagnostics,
        instance_id=None,
        severity="error",
        disposition="unknown",
    )
    return replace(
        diagnostics,
        component=record["component"],
        operation=record["operation"],
        code=record["code"],
        category=record["category"],
        content_policy="metadata_only",
        diagnostic_id=record["diagnostic_id"],
        occurred_at=record["occurred_at"],
        cause_id=record["cause_id"],
        operation_id=record["operation_id"],
        request_id=record["request_id"],
        channel=record["channel"],
        turn_id=record["turn_id"],
        generation=record["generation"],
        status_code=record["status_code"],
        retry_after_ms=record["retry_after_ms"],
        fields=record["fields"],
        rejected_fields=record["detail"]["rejected_fields"],
        detail_truncated=record["detail"]["truncated"],
    )


class DiagnosticQueryArgumentsError(ValueError):
    """Deliberate diagnostic query argument rejection, never an execution defect."""


class DiagnosticHistory:
    """Current-run metadata only, independent of file delivery and event subscribers."""

    def __init__(
        self,
        *,
        max_records: int = DIAGNOSTIC_HISTORY_MAX_RECORDS,
        max_bytes: int = DIAGNOSTIC_HISTORY_MAX_BYTES,
    ):
        if max_records < 1 or max_bytes < DIAGNOSTIC_RECORD_MAX_BYTES:
            raise ValueError("diagnostic history must hold at least one bounded record")
        self.instance_id: str | None = None
        self.max_records = max_records
        self.max_bytes = max_bytes
        self.records: OrderedDict[str, tuple[dict, int]] = OrderedDict()
        self.sequence = 0
        self.retained_bytes = 0
        self.evicted = 0
        self.rejected_fields = 0
        self.rejected_records = 0
        self.duplicates = 0
        self.subscriber_delivery_failures = 0
        self.file_projection_failures = 0
        self._lock = RLock()

    def bind_instance(self, instance_id: str) -> None:
        if (
            type(instance_id) is not str
            or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", instance_id) is None
        ):
            raise ValueError("invalid diagnostic instance identity")
        if self.instance_id not in (None, instance_id):
            raise ValueError("diagnostic instance cannot change")
        self.instance_id = instance_id

    def retain(self, diagnostics: ErrorDiagnostics, **kwargs) -> tuple[dict, bool]:
        record = normalize_diagnostic(diagnostics, instance_id=self.instance_id, **kwargs)
        with self._lock:
            existing = self.records.get(record["diagnostic_id"])
            if existing is not None:
                self.duplicates += 1
                return json.loads(json.dumps(existing[0])), False
            self.sequence += 1
            record["diagnostic_sequence"] = self.sequence
            size = len(json.dumps(record, separators=(",", ":")).encode())
            while self.records and (
                len(self.records) >= self.max_records or self.retained_bytes + size > self.max_bytes
            ):
                _, (_, removed_size) = self.records.popitem(last=False)
                self.retained_bytes -= removed_size
                self.evicted += 1
            self.records[record["diagnostic_id"]] = (record, size)
            self.retained_bytes += size
            self.rejected_fields += record["detail"]["rejected_fields"]
            return json.loads(json.dumps(record)), True

    def query(self, arguments: dict) -> dict:
        allowed = {"operation_id", "diagnostic_id", "instance_id", "after", "limit"}
        if arguments.keys() - allowed:
            raise DiagnosticQueryArgumentsError("unknown diagnostic query argument")
        limit = arguments.get("limit", 50)
        after = arguments.get("after", 0)
        if (
            type(limit) is not int
            or not 1 <= limit <= DIAGNOSTIC_QUERY_MAX_RECORDS
            or type(after) is not int
            or after < 0
        ):
            raise DiagnosticQueryArgumentsError("invalid diagnostic query bounds")
        for key in ("operation_id", "diagnostic_id", "instance_id"):
            identity = arguments[key] if key in arguments else None
            valid = _identity(identity) is not None
            if key == "instance_id":
                valid = (
                    type(identity) is str
                    and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", identity) is not None
                )
            if key in arguments and not valid:
                raise DiagnosticQueryArgumentsError("invalid diagnostic identity")
        with self._lock:
            oldest = (
                next(iter(self.records.values()))[0]["diagnostic_sequence"]
                if self.records
                else self.sequence + 1
            )
            foreign = arguments.get("instance_id", self.instance_id) != self.instance_id
            invalid = after > self.sequence
            matched = (
                []
                if foreign or invalid
                else [
                    record
                    for record, _ in self.records.values()
                    if record["diagnostic_sequence"] > after
                    and all(
                        record.get(key) == arguments[key]
                        for key in ("operation_id", "diagnostic_id")
                        if key in arguments
                    )
                ]
            )
            selected = matched[:limit]
            result = {
                "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
                "instance_id": self.instance_id,
                "availability": (
                    "unavailable" if self.instance_id is None else "retained_current_run"
                ),
                "lookup": (
                    "foreign_instance"
                    if foreign
                    else "invalid_cursor" if invalid else "found" if matched else "not_retained"
                ),
                "records": selected,
                "oldest_sequence": oldest,
                "latest_sequence": self.sequence,
                "next_after": selected[-1]["diagnostic_sequence"] if selected else after,
                "gap": foreign or invalid or after < oldest - 1,
                "evicted_records": self.evicted,
                "truncated": len(matched) > limit,
                "retention": {
                    "max_records": self.max_records,
                    "max_bytes": self.max_bytes,
                    "record_max_bytes": DIAGNOSTIC_RECORD_MAX_BYTES,
                    "query_max_records": DIAGNOSTIC_QUERY_MAX_RECORDS,
                    "durable": False,
                    "restart_resets_history": True,
                },
            }
            return json.loads(json.dumps(result))

    def health(self) -> dict:
        with self._lock:
            return {
                "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
                "instance_id": self.instance_id,
                "history": {
                    "availability": "retained_current_run" if self.instance_id else "unbound",
                    "retained_records": len(self.records),
                    "retained_bytes": self.retained_bytes,
                    "evicted_records": self.evicted,
                    "rejected_fields": self.rejected_fields,
                    "rejected_records": self.rejected_records,
                    "duplicate_projections": self.duplicates,
                    "subscriber_delivery_failures": self.subscriber_delivery_failures,
                    "file_projection_failures": self.file_projection_failures,
                    "durable": False,
                    "restart_resets_history": True,
                },
            }
