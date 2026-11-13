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
from dataclasses import fields as dataclass_fields
from datetime import datetime
from itertools import islice
from threading import RLock
from uuid import uuid4

import httpx

from puripuly_heart.config.provider_values import STTProviderName
from puripuly_heart.core.diagnostic_codes import (
    AUDIO_DIAGNOSE_FAILURE_CODES,
    is_supported_diagnostic_code,
)
from puripuly_heart.core.diagnostic_validation import (
    OPENAI_FAILURE_ENUM_FIELDS,
    QWEN_FAILURE_ENUM_FIELDS,
    QWEN_FAILURE_NUMERIC_FIELDS,
    stt_startup_failure_code,
)
from puripuly_heart.core.lifecycle import LIFECYCLE_SHUTDOWN_PHASE_ORDER
from puripuly_heart.core.local_asr.local_stt_assets import (
    LOCAL_QWEN_GPU_MODEL_ID,
    REQUIRED_CPU_LOCAL_STT_MODEL_IDS,
)
from puripuly_heart.core.messages import (
    DIAGNOSTIC_CATEGORIES,
    DIAGNOSTIC_VISIBILITIES,
    SEVERITIES,
    DiagnosticCategory,
    ErrorDiagnostics,
)
from puripuly_heart.core.network_diagnostics import (
    TRANSPORT_ENUM_FIELDS,
    TRANSPORT_NUMERIC_FIELDS,
    classify_transport_error,
    safe_transport_fields,
    transport_exception_chain,
)
from puripuly_heart.core.openrouter.managed_operation_identity import is_valid_operation_id
from puripuly_heart.core.qq_auth_diagnostics import ENUM_FIELDS as QQ_ENUM_FIELDS
from puripuly_heart.core.qq_auth_diagnostics import STAGES as QQ_STAGES

DIAGNOSTIC_SCHEMA_VERSION = 1
DIAGNOSTIC_RECORD_MAX_BYTES = 4096
DIAGNOSTIC_HISTORY_MAX_RECORDS = 256
DIAGNOSTIC_HISTORY_MAX_BYTES = 512 * 1024
DIAGNOSTIC_QUERY_MAX_RECORDS = 100
DIAGNOSTIC_CAUSE_MAX_COUNT = 8
DIAGNOSTIC_CAUSE_MAX_DEPTH = 3
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
_NUMERIC_FIELDS = (
    frozenset(
        {
            "errno",
            "winerror",
            "exit_code",
            "attempts",
            "count",
            "frame_count",
            "vad_sample_count",
            "level_ppm",
            "speech_probability_ppm",
            "elapsed_ms",
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
    )
    | frozenset(TRANSPORT_NUMERIC_FIELDS)
    | frozenset(QWEN_FAILURE_NUMERIC_FIELDS)
)
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
_NUMERIC_FIELDS = _NUMERIC_FIELDS | frozenset({"source_attempt"})
_UNIT_INTERVAL_FIELDS = frozenset(
    {"speech_probability_min", "speech_probability_max", "speech_threshold"}
)
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
        "opened",
        "frames_received",
        "silence_observed",
        "speech_detected",
        "candidate_current_route",
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
        "route_id",
    }
)
_ENUM_FIELDS = {
    **TRANSPORT_ENUM_FIELDS,
    **OPENAI_FAILURE_ENUM_FIELDS,
    **QWEN_FAILURE_ENUM_FIELDS,
    **{key: frozenset(values) for key, values in QQ_ENUM_FIELDS.items()},
    "service_code": frozenset(
        {
            "service_authentication_failed",
            "service_authentication_unavailable",
            "service_rate_limited",
            "service_turn_exhausted",
            "service_busy",
            "request_too_large",
            "service_allowance_exhausted",
            "service_shutdown",
            "unsupported_protocol",
        }
    ),
    "source_kind": frozenset({"huggingface", "huggingface_mirror", "modelscope", "local"}),
    "backend": frozenset({"cpu", "gpu"}),
    "channel": frozenset({"self", "peer"}),
    "channel_availability": frozenset({"continue", "recovering", "unavailable"}),
    "audio_failure_code": AUDIO_DIAGNOSE_FAILURE_CODES | {"none"},
    "audio_restore_self": frozenset(
        {"not_needed", "restored", "skipped", "failed", "cleanup_pending"}
    ),
    "audio_restore_peer": frozenset(
        {"not_needed", "restored", "skipped", "failed", "cleanup_pending"}
    ),
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
        "audio_diagnosis",
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
        "asr_worker",
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
        "translation_request",
        "managed_connection_auth",
        "qq_managed_auth",
        "manual_typing",
        "local_asr_provisioning",
        "clipboard",
        "updater",
        "ui.foundation",
        "headless.presentation",
        "control.transport",
        "lifecycle.scope",
        "lifecycle.shutdown",
        "index_selection",
        "ui_audio_settings",
        "ui_i18n",
        "ui_model_assets",
        "ui_application",
        "gui_launcher",
        "api_key_verification",
        "model_assets",
        "settings_projection",
        "telemetry",
        "managed_openrouter_release",
    }
)
_BOUNDARIES = (
    frozenset(
        {
            "emit",
            "dispatch",
            "diagnostic",
            "python_log",
            "audio_diagnosis",
            "audio.diagnose",
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
            "translation_probe",
            "agent_tool",
            "support_report",
            "agent_chat",
            "tool_dispatch",
            "bundle",
            "upload",
            "response",
            "watcher_start",
            "watcher_stop",
            "clipboard_read",
            "clipboard_callback",
            "clipboard_cleanup",
            "clipboard_loop",
            "clipboard_submit",
            "submit_drain",
            "update_check",
            "page_task",
            "shutdown_task",
            "control_request",
            "fallback_commit",
            "task_done",
            "close_callback",
            "host_api_enumeration",
            "microphone_enumeration",
            "desktop_output_enumeration",
            "desktop_output_cleanup",
            "bundle_load",
            "locale_discovery",
            "message_format",
            "asset_operation",
            "model_list_scroll",
            "star_prompt",
            "usage_refresh",
            "runtime_preparation",
            "discord_auth",
            "discord_translation_start",
            "managed_auth_admission",
            "window_center",
            "window_hide",
            "ui_task_cleanup",
            "startup_cleanup",
            "update_notification",
            "gui_startup",
            "gui_runtime",
            "gui_cleanup",
            "secret_save",
            "cpu_install",
            "gpu_install",
            "gemma_install",
            "download",
            "remove",
            "render_settings",
            "refresh_settings_after_openrouter_pkce_success",
            "telemetry_send",
            "telemetry_persist",
            "telemetry_notification",
            "managed_delivery_store",
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
    | TRANSPORT_ENUM_FIELDS["stage"]
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
        "cleanup_stage": _ENUM_FIELDS["cleanup_stage"] | {"cleanup"},
    }
)
_NUMERIC_FIELDS = _NUMERIC_FIELDS | frozenset(
    {"native_status", "native_code", "requested_channels"}
)
_COMPONENTS = _COMPONENTS | frozenset(
    {
        "github_star_prompt",
        "control.host",
        "recognition_probe",
        "secret_store",
        "overlay_application",
    }
)
_BOUNDARIES = _BOUNDARIES | frozenset(
    {
        "state_delivery",
        "diagnostic_delivery",
        "inspection",
        "error_presented",
        "launch_prompt",
        "translation_success",
        "prompt_persist",
        "prompt_rollback",
        "endpoint_close",
        "host_start",
        "host_close",
        "readiness_cleanup",
        "capture_observer",
        "capture_background",
        "routing_observer",
        "observe",
        "recognition_probe",
        "process_lifecycle",
        "diagnostic_sink",
        "preflight_managed_auth_claim",
        "avatar_query",
        "host_info",
        "discovery_monitor",
        "unadvertise_receiver",
        "receiver_stop",
        "service_stop",
        "control_dispatch",
        "transport_write",
        "replacement_start",
        "recovery_state",
    }
)
_ENUM_FIELDS.update(
    {
        "asset_failure_code": frozenset(
            {
                "unknown",
                "install_failed",
                "download_failed",
                "download_timeout",
                "proxy_failed",
                "connection_failed",
                "source_unavailable",
                "disk_full",
                "checksum_failed",
                "file_size_mismatch",
                "mirror_unavailable",
                "mirror_redirect_external",
                "range_invalid",
                "network_configuration_failed",
                "worker_spawn_failed",
                "worker_protocol_failed",
                "worker_reported_failure",
                "worker_exited",
                "worker_import_failed",
                "status_callback_failed",
                "unsupported_source",
                "missing_file",
                "import_failed",
                "task_busy",
                "model_in_use",
                "model_owner_unavailable",
                "unknown_model",
                "cancelled",
                "inspect_failed",
            }
        ),
        "capture_attempt": frozenset(
            {
                "device_resolution",
                "primary",
                "primary_mono_retry",
                "name_fallback",
                "name_fallback_mono_retry",
                "system_default",
                "system_default_mono_retry",
            }
        ),
        "native_reason": frozenset(
            {
                "invalid_argument",
                "not_implemented",
                "model_file_not_found",
                "model_load",
                "out_of_memory",
                "backend",
                "unsupported",
                "bad_struct_size",
                "input_too_long",
                "aborted",
                "output_truncated",
                "output_repetition",
                "unsupported_role",
                "input_too_short",
                "version_mismatch",
                "nul",
                "busy",
                "other",
            }
        ),
        "native_classification": frozenset(
            {
                "decode_failure",
                "audio_invalid",
                "cancelled",
                "worker_busy",
                "protocol_invalid",
                "mode_rejected",
                "request_rejected",
                "worker_failure",
            }
        ),
    }
)
_OVERLAY_NATIVE_STAGES = frozenset(
    {
        "unknown",
        "renderer_init",
        "hardware_device",
        "draw",
        "openvr_init",
        "adapter_selection",
        "submit",
        "calibration",
        "render_frame",
        "reanchor",
        "submit_frame",
        "show_overlay",
        "hide_overlay",
        "reconcile_visibility",
        "cleanup_hide",
        "cleanup_release",
        "acknowledge_quit",
        "event_process_quit",
    }
)
_ENUM_FIELDS.update({"native_stage": _OVERLAY_NATIVE_STAGES})
_NUMERIC_FIELDS = _NUMERIC_FIELDS | frozenset({"runtime_build"})
_BOOLEAN_FIELDS = _BOOLEAN_FIELDS | frozenset({"build_match", "commit_match"})
_COMPONENTS = _COMPONENTS | frozenset({"audio_test", "stt.provider_runtime"})
_BOUNDARIES = _BOUNDARIES | frozenset(
    {
        "audio.test",
        "device_metadata",
        "device_resolution",
        "smart_turn_prepare",
        "smart_turn_predict",
        "smart_turn_completion",
        "segmentation",
        "runtime_version",
        "runtime_files",
        "runtime_startup",
        "runtime",
        "prefix",
        "prefix_restore",
        "prefix_save",
        "slot_cleanup",
        "state_changed",
    }
)
_ENUM_FIELDS["native_stage"] = _ENUM_FIELDS["native_stage"] | {"gpu_query_create", "gpu_query_poll"}
_ENUM_FIELDS["native_classification"] = _ENUM_FIELDS["native_classification"] | {
    "worker_operation_panicked",
    "worker_operation_cancelled",
    "worker_unload_panicked",
    "worker_unload_cancelled",
    "worker_reader_failed",
    "worker_writer_failed",
    "worker_unload_failed",
    "worker_connection_closed",
}
_ENUM_FIELDS["reason"] = frozenset(
    {
        "transport",
        "metadata_unavailable",
        "credential_unavailable",
        "observation_unavailable",
        "owned_work_failed",
        "settings_unavailable",
        "credential_not_current",
        "metadata_invalid",
    }
)
_ENUM_FIELDS["deepgram_code"] = frozenset({"DATA-0000", "NET-0000", "NET-0001", "NET-0002"})
_ENUM_FIELDS["owner_name"] = _ENUM_FIELDS["owner_name"] | {
    "self_audio",
    "provider_runtime",
    "runtime_logging",
}
_ENUM_FIELDS["callback_name"] = _ENUM_FIELDS["callback_name"] | {
    "drain_cancel",
    "close_provider",
    "final_summary",
}
_BOUNDARIES = _BOUNDARIES | {"accounts.status"}
_ID_FIELDS = _ID_FIELDS | {"provider_request_id"}
_NULLABLE_FIELDS = (
    _NUMERIC_FIELDS
    | _BOOLEAN_FIELDS
    | _ID_FIELDS
    | _ENUM_FIELDS.keys()
    | _UNIT_INTERVAL_FIELDS
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
    try:
        logger.log(
            (
                logging.INFO
                if severity == "info"
                else logging.WARNING if severity == "warning" else logging.ERROR
            ),
            "[Failure] structured_failure",
            extra={
                DIAGNOSTIC_RECORD_ATTRIBUTE: (diagnostics, severity, disposition, stage, exception)
            },
        )
    except Exception:
        return


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


def exception_diagnostics(
    exception: BaseException,
    *,
    component: str,
    operation: str,
    code: str,
    category: DiagnosticCategory = "lifecycle",
    fields: Mapping[str, object] | None = None,
) -> ErrorDiagnostics:
    seen: set[int] = set()
    remaining = [DIAGNOSTIC_CAUSE_MAX_COUNT]

    def build(current: BaseException, depth: int) -> ErrorDiagnostics:
        seen.add(id(current))
        attached = getattr(current, "failure_diagnostics", None)
        if not isinstance(attached, ErrorDiagnostics):
            attached = getattr(current, "diagnostics", None)
        causes = [attached] if isinstance(attached, ErrorDiagnostics) else []
        truncated = False
        children = (
            current.exceptions
            if isinstance(current, BaseExceptionGroup)
            else (
                (
                    current.__cause__
                    or (None if current.__suppress_context__ else current.__context__)
                ),
            )
        )
        for child in children:
            if child is None:
                continue
            if depth >= DIAGNOSTIC_CAUSE_MAX_DEPTH or remaining[0] == 0:
                truncated = True
                break
            if id(child) in seen:
                truncated = True
                continue
            remaining[0] -= 1
            causes.append(build(child, depth + 1))
        transport_kind = classify_transport_error(current)
        effective_category = category
        if (
            category == "lifecycle"
            and transport_kind is not None
            and (
                transport_kind in {"timeout", "tls", "proxy"}
                or isinstance(current, (httpx.HTTPError, ConnectionError))
            )
        ):
            effective_category = "timeout" if transport_kind == "timeout" else "network"
        status_code = next(
            (
                item.response.status_code
                for item in transport_exception_chain(current)
                if isinstance(item, httpx.HTTPStatusError)
                and type(item.response.status_code) is int
                and 100 <= item.response.status_code <= 599
            ),
            None,
        )
        return ErrorDiagnostics(
            component=component,
            operation=operation,
            code=code,
            category=effective_category,
            visibility="basic",
            content_policy="metadata_only",
            status_code=status_code,
            retry_after_ms=None,
            fields={
                **safe_exception_fields(current),
                **{
                    key: value
                    for key, value in safe_transport_fields(current).items()
                    if value != "unknown"
                },
                **(fields if depth == 0 and fields is not None else {}),
            },
            **(current_diagnostic_operation() if depth == 0 else {}),
            causes=tuple(causes),
            causes_truncated=truncated,
        )

    return sanitize_diagnostics(build(exception, 0))


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
        elif key in {"level_ppm", "speech_probability_ppm"}:
            accepted = type(value) is int and 0 <= value <= 1_000_000
        elif key in {"native_status", "native_code"}:
            accepted = type(value) is int and -(2**31) <= value < 2**31
        elif key in _NUMERIC_FIELDS:
            accepted = type(value) is int and (
                -(2**31) <= value < 2**32
                if key in {"errno", "os_errno", "winerror", "exit_code"}
                else 0 <= value < 2**64
            )
        elif key in _UNIT_INTERVAL_FIELDS:
            accepted = type(value) in {int, float} and 0 <= value <= 1
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
        if key == "model_id" and type(value) is str and not accepted:
            from puripuly_heart.core.local_translation.assets import (
                GEMMA_26B_MODEL_ID,
                GEMMA_MODEL_ID,
            )

            accepted = value in {GEMMA_MODEL_ID, GEMMA_26B_MODEL_ID}
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
    return _normalize_diagnostic(
        diagnostics,
        instance_id=instance_id,
        severity=severity,
        disposition=disposition,
        stage=stage,
        exception=exception,
        file_evidence=file_evidence,
        depth=0,
        ancestors=frozenset(),
        remaining=[DIAGNOSTIC_CAUSE_MAX_COUNT],
    )


def _normalize_diagnostic(
    diagnostics: ErrorDiagnostics,
    *,
    instance_id: str | None,
    severity: str,
    disposition: str,
    stage: str | None = None,
    exception: BaseException | None = None,
    file_evidence: str = "unknown",
    depth: int,
    ancestors: frozenset[int],
    remaining: list[int],
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
    causes = []
    causes_truncated = diagnostics.causes_truncated is True
    rejected += int(type(diagnostics.causes_truncated) is not bool)
    ancestors = ancestors | {id(diagnostics)}
    for index, cause in enumerate(diagnostics.causes):
        if (
            depth >= DIAGNOSTIC_CAUSE_MAX_DEPTH
            or remaining[0] == 0
            or index >= DIAGNOSTIC_CAUSE_MAX_COUNT
        ):
            causes_truncated = True
            break
        if not isinstance(cause, ErrorDiagnostics):
            rejected += 1
            continue
        if id(cause) in ancestors:
            causes_truncated = True
            continue
        remaining[0] -= 1
        cause_record = _normalize_diagnostic(
            cause,
            instance_id=None,
            severity="error",
            disposition="failed",
            depth=depth + 1,
            ancestors=ancestors,
            remaining=remaining,
        )
        causes.append(_diagnostic_value(cause_record))
    detail_truncated = (
        bool(diagnostics.detail_truncated)
        or causes_truncated
        or any(cause["detail_truncated"] for cause in causes)
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
        "visibility": (
            diagnostics.visibility
            if type(diagnostics.visibility) is str
            and diagnostics.visibility in DIAGNOSTIC_VISIBILITIES
            else "basic"
        ),
        "content_policy": "metadata_only",
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
        "causes": causes,
        "causes_truncated": causes_truncated,
        "file_evidence": (
            file_evidence
            if type(file_evidence) is str and file_evidence in {"unknown", "unavailable"}
            else "unknown"
        ),
        "detail": {
            "state": (
                "truncated"
                if detail_truncated
                else (
                    "rejected"
                    if rejected
                    else (
                        "available"
                        if any(value is not None for value in fields.values())
                        or causes
                        or status_code is not None
                        else "unavailable"
                    )
                )
            ),
            "rejected_fields": rejected,
            "truncated": detail_truncated,
        },
    }
    _bound_diagnostic_record(record)
    return record


def _diagnostic_value(record: Mapping[str, object]) -> dict:
    detail = record["detail"]
    values = {item.name: record.get(item.name) for item in dataclass_fields(ErrorDiagnostics)}
    values.update(
        content_policy="metadata_only",
        rejected_fields=detail["rejected_fields"],
        detail_truncated=detail["truncated"],
    )
    return values


def _bound_diagnostic_record(record: dict) -> None:
    def truncate(node: dict) -> None:
        if "detail" in node:
            node["detail"]["state"] = "truncated"
            node["detail"]["truncated"] = True
        else:
            node["detail_truncated"] = True

    def prune_causes(node: dict) -> bool:
        for cause in reversed(node["causes"]):
            if cause["causes"] and prune_causes(cause):
                truncate(node)
                return True
        if len(node["causes"]) > 1:
            node["causes"].pop()
            node["causes_truncated"] = True
            truncate(node)
            return True
        return False

    def prune_fields(node: dict) -> bool:
        for cause in reversed(node["causes"]):
            if prune_fields(cause):
                truncate(node)
                return True
        if node["fields"]:
            node["fields"].pop(next(reversed(node["fields"])))
            truncate(node)
            return True
        return False

    while (
        len(json.dumps(record, separators=(",", ":")).encode()) + 64 > DIAGNOSTIC_RECORD_MAX_BYTES
    ):
        if prune_causes(record) or prune_fields(record):
            continue
        if record["causes"]:
            record["causes"].pop()
            record["causes_truncated"] = True
            truncate(record)
            continue
        break


def _diagnostics_from_mapping(
    value: Mapping[str, object], *, depth: int = 0, remaining: list[int] | None = None
) -> ErrorDiagnostics:
    if remaining is None:
        remaining = [DIAGNOSTIC_CAUSE_MAX_COUNT]
    detail = value.get("detail")
    detail = detail if isinstance(detail, Mapping) else {}
    fields = value.get("fields", {})
    rejected = value.get("rejected_fields", detail.get("rejected_fields", 0))
    rejected = rejected if type(rejected) is int and 0 <= rejected < 2**31 else 1
    if not isinstance(fields, Mapping):
        fields = {}
        rejected += 1
    raw_causes = value.get("causes", ())
    causes = []
    truncated = value.get("causes_truncated") is True
    if not isinstance(raw_causes, (tuple, list)):
        rejected += 1
    else:
        for index, cause in enumerate(raw_causes):
            if (
                depth >= DIAGNOSTIC_CAUSE_MAX_DEPTH
                or remaining[0] == 0
                or index >= DIAGNOSTIC_CAUSE_MAX_COUNT
            ):
                truncated = True
                break
            if not isinstance(cause, Mapping):
                rejected += 1
                continue
            remaining[0] -= 1
            causes.append(_diagnostics_from_mapping(cause, depth=depth + 1, remaining=remaining))
    values = {
        item.name: value[item.name]
        for item in dataclass_fields(ErrorDiagnostics)
        if item.name in value
    }
    values.update(
        component=value.get("component", "unclassified"),
        operation=value.get("operation"),
        code=value.get("code"),
        category=value.get("category", "unknown"),
        visibility=value.get("visibility", "basic"),
        content_policy="metadata_only",
        status_code=value.get("status_code"),
        retry_after_ms=value.get("retry_after_ms"),
        fields=fields,
        diagnostic_id=value.get("diagnostic_id"),
        occurred_at=value.get("occurred_at"),
        rejected_fields=rejected,
        detail_truncated=value.get("detail_truncated", detail.get("truncated")) is True,
        causes=tuple(causes),
        causes_truncated=truncated,
    )
    return ErrorDiagnostics(**values)


def serialize_diagnostics(diagnostics: ErrorDiagnostics) -> dict:
    return _diagnostic_value(
        normalize_diagnostic(diagnostics, instance_id=None, severity="error", disposition="unknown")
    )


def deserialize_diagnostics(value: Mapping[str, object]) -> ErrorDiagnostics:
    return sanitize_diagnostics(_diagnostics_from_mapping(value))


def project_diagnostic_record(value: Mapping[str, object]) -> dict:
    instance_id = value.get("instance_id")
    if type(instance_id) is not str or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", instance_id) is None:
        instance_id = None
    record = normalize_diagnostic(
        _diagnostics_from_mapping(value),
        instance_id=instance_id,
        severity=value.get("severity", "error"),
        disposition=value.get("disposition", "unknown"),
        stage=value.get("stage"),
        file_evidence=value.get("file_evidence", "unknown"),
    )
    sequence = value.get("diagnostic_sequence")
    if type(sequence) is int and 0 <= sequence < 2**64:
        record["diagnostic_sequence"] = sequence
    return record


def sanitize_diagnostics(diagnostics: ErrorDiagnostics) -> ErrorDiagnostics:
    """Use the same ingress policy for typed owner results before their projection."""
    record = normalize_diagnostic(
        diagnostics,
        instance_id=None,
        severity="error",
        disposition="unknown",
    )
    return _diagnostics_from_mapping(_diagnostic_value(record))


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
                    else (
                        "invalid_cursor"
                        if invalid
                        else (
                            "found"
                            if matched
                            else (
                                "no_new_records"
                                if self.instance_id is not None
                                and "after" in arguments
                                and after == self.sequence
                                and not any(
                                    key in arguments for key in ("operation_id", "diagnostic_id")
                                )
                                else "not_retained"
                            )
                        )
                    )
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
