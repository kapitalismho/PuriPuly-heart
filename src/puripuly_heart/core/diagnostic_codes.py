"""Finite repository-owned diagnostic codes; unknown external labels are not evidence."""

from __future__ import annotations

from typing import get_args

from puripuly_heart.core.diagnostic_validation import stt_startup_failure_code
from puripuly_heart.core.messages import DIAGNOSTIC_CATEGORIES, TRANSACTION_RESULT_STATUSES
from puripuly_heart.core.qq_auth_diagnostics import CODES as QQ_AUTH_CODES

DOMAIN_CODES = (
    frozenset(
        {
            "unclassified_failure",
            "diagnostic_builder_failed",
            "application_operation_failed",
            "presentation_sync_failed",
            "provider_receive_failed",
            "output_cleanup_failed",
            "output_observer_failed",
            "oscquery_unavailable",
            "osc_state_send_failed",
            "worker_cleanup_failed",
            "startup_cleanup_failed",
            "worker_startup_failed",
            "worker_activation_failed",
            "worker_startup_cancelled",
            "worker_executable_missing",
            "native_diagnostic_loss",
            "native_terminal_evidence_incomplete",
            "child_detail_rejected",
            "overlay_reader_cleanup_failed",
            "osc_local_send_failed",
            "shutdown_cleanup_failed",
            "destination_write_failed",
            "destination_write_timeout",
            "ui_event_delivery_failed",
            "settings_memory_restore_failed",
            "chatgpt_browser_open_failed",
            "delivery_ack_failed",
            "log_file_delivery_failed",
            "log_queue_overflow",
            "diagnostic_builder_failed",
            "managed_release_cleanup_failed",
            "managed_release_create_failed",
            "managed_account_observation_failed",
            "managed_status_refresh_failed",
            "managed_status_refresh_schedule_failed",
            "managed_usage_refresh_failed",
            "managed_status_notification_failed",
            "managed_usage_notification_failed",
            "managed_auth_notification_failed",
            "managed_authentication_failed",
            "worker_transport_closed_with_error",
            "secret_store_passphrase_missing",
            "secret_store_backend_unsupported",
            "destination_publish_failed",
            "osc_state_send_recovered",
            "desktop_view_job_assignment_failed",
            "broker_issue_exception",
            "broker_issue_failed",
            "broker_issue_unknown_outcome",
            "delivery_ack_recovered",
            "delivery_ack_recovery_pending",
            "discord_auth_exception",
            "discord_auth_failed",
            "discord_issue_transport_failure",
            "discord_loopback_unavailable",
            "discord_oauth_callback_closed",
            "discord_oauth_callback_error",
            "discord_oauth_timeout",
            "discord_redirect_mismatch",
            "local_identity_preflight_exception",
            "local_identity_preflight_failed",
            "managed_key_delivery_ack_error",
            "managed_key_delivery_ack_failed",
            "managed_operation_action_required",
            "managed_operation_authorization_expired",
            "managed_operation_delivery_reconciling",
            "managed_operation_error",
            "managed_operation_gone_malformed",
            "managed_operation_malformed",
            "managed_operation_recovery_pending",
            "managed_operation_resume_pending",
            "managed_operation_status_failed",
            "managed_operation_status_unavailable",
            "managed_operation_other_source_pending_operation",
            "managed_operation_other_source_pending_delivery_ack",
            "qq_managed_auth_resumed_credential_acknowledged",
            "qq_managed_key_required",
            "qq_managed_operation_delivery_ack_recovered",
            "recovered_delivery_ack_claim_persist_failed",
            "remote_active_local_claim_persist_failed",
            "remote_active_local_secret_write_failed",
            "remote_active_local_settings_commit_failed",
            "remote_active_unsafe_settings_values",
            "unsafe_settings_values",
            "translation.parent_admission_failed",
            "translation.child_creation_failed",
            "translation.parent_rejection_failed",
            "translation.parent_execution_failed",
            "translation.turn_generation_observer_failed",
            "translation.child_retirement_failed",
            "translation.parent_retirement_failed",
            "translation.output_submission_failed",
            "desktop_view_terminate_failed",
            "desktop_view_kill_failed",
            "desktop_view_job_close_failed",
            "desktop_view_launch_cleanup_pending",
            "desktop_view_cleanup_failed",
            "desktop_view_cleanup_recovered",
            "desktop_view_graceful_close_failed",
            "desktop_view_rejected_cleanup_failed",
            "desktop_view_process_wait_failed",
            "desktop_view_pid_file_remove_failed",
            "settings_save_failed",
            "settings_rollback_failed",
            "settings_commit_notification_failed",
            "settings_failure_notification_failed",
            "settings_parse_failed",
            "settings_migration_failed",
            "settings_backup_failed",
            "settings_restore_failed",
            "settings_apply_result_missing",
            "provider_verifier_missing",
            "provider_verification_context_missing",
            "provider_verifier_exception",
            "provider_verifier_result_mismatch",
            "provider_verification_failed",
            "secret_restore_failed",
            "dashboard_notification_failed",
            "secret_write_failed",
            "settings_commit_failed",
            "settings_commit_failed_secret_restored",
            "settings_commit_failed_secret_restore_failed",
            "settings_secret_failed",
            "provider_runtime_apply_unavailable",
            "stt_runtime_apply_unavailable",
            "peer_stt_runtime_apply_unavailable",
            "stt_language_audio_runtime_unavailable",
            "peer_stt_language_audio_runtime_unavailable",
            "llm_stt_language_audio_runtime_unavailable",
            "stt_language_audio_runtime_apply_exception",
            "overlay_osc_output_runtime_apply_exception",
            "ui_prompt_clipboard_state_runtime_apply_exception",
            "provider_runtime_apply_exception",
            "settings_runtime_apply_exception",
            "self_stt_runtime_apply_not_converged",
            "stt_runtime_apply_not_converged",
            "runtime_apply_exception",
            "runtime_apply_degraded",
            "runtime_apply_interrupted",
            "settings_snapshot_notification_failed",
            "runtime_result_notification_failed",
            "settings_compensation_failed",
            "settings_runtime_resync_failed",
            "settings_background_task_failed",
            "application_state_notification_failed",
            "application_shutdown_timeout",
            "application_shutdown_failed",
            "application_shutdown_diagnostic_delivery_failed",
            "startup_fallback_not_persisted",
            "startup_presentation_failed",
            "startup_prepare_failed",
            "managed_application_failed",
            "qq_application_failed",
            "manual_submission_failed",
            "manual_typing_degraded",
            "vrchat_process_observation_failed",
            "vrchat_process_watch_failed",
            "vrchat_scene_log_read_failed",
            "vrchat_process_watch_close_failed",
            "remote_active_managed_secret_missing",
            "delivery_ack_token_store_failed_before_local_key_write",
            "managed_operation_state_unavailable",
            "managed_operation_installation_unavailable",
            "managed_operation_token_store_failed_before_issue",
            "managed_operation_persist_failed_before_issue",
            "managed_operation_active_key_missing",
            "managed_operation_active_recovered",
            "managed_operation_delivery_ack_recovered",
            "delivery_ack_pending_metadata_incomplete",
            "delivery_ack_managed_secret_missing",
            "delivery_ack_token_read_failed",
            "delivery_ack_token_missing",
            "delivery_ack_exception",
            "delivery_ack_token_clear_failed",
            "delivery_ack_token_restore_failed",
            "delivery_ack_clear_persist_failed",
            "provider.openrouter.insufficient_credits",
            "provider.openrouter.key_limit",
            "provider.openrouter.temporary_limit",
            "provider.openrouter.payment_required",
            "translation.child_timeout",
            "translation.child_execution_failed",
            "translation.stale_provider_completion",
            "stt.recovered",
            "stt.decode_callback_failed",
            "capture.peer.source_failure",
            "capture.peer.target_exited",
            "chatgpt_failed",
            "chatgpt_timeout",
            "chatgpt_loopback_unavailable",
            "chatgpt_network_error",
            "chatgpt_state_mismatch",
            "chatgpt_access_denied",
            "chatgpt_authorization_failed",
            "chatgpt_registration_incomplete",
            "chatgpt_client_mismatch",
            "chatgpt_invalid_id_token",
            "chatgpt_plan_scope_missing",
            "chatgpt_token_exchange_failed",
            "chatgpt_token_response_invalid",
            "chatgpt_jwks_unavailable",
            "chatgpt_credential_store_failed",
            "chatgpt_credential_restore_failed",
            "chatgpt_loopback_close_failed",
            "chatgpt_remote_revocation_unconfirmed",
            "chatgpt_local_credential_delete_failed",
            "chatgpt_secret_read_failed",
            "chatgpt_refresh_failed",
        }
    )
    | frozenset(TRANSACTION_RESULT_STATUSES)
    | QQ_AUTH_CODES
    | frozenset(
        f"{surface}.{category}"
        for surface in ("provider", "stt")
        for category in DIAGNOSTIC_CATEGORIES
    )
    | frozenset(
        f"chatgpt_{code}"
        for code in {
            "signed_out",
            "reauth_required",
            "invalid_grant",
            "invalid_refresh_token",
            "token_expired",
            "refresh_token_expired",
            "refresh_token_invalidated",
            "refresh_token_reused",
            "invalid_client",
            "refresh_failed",
            "refresh_network_error",
            "plan_scope_missing",
            "invalid_request",
            "invalid_scope",
            "unauthorized_client",
            "server_error",
            "temporarily_unavailable",
        }
    )
)


def is_supported_diagnostic_code(code: object, native_codes: frozenset[str]) -> bool:
    if type(code) is not str or len(code) > 96:
        return False
    if code in DOMAIN_CODES or code in native_codes or stt_startup_failure_code(code) == code:
        return True
    if code.startswith("stt."):
        from puripuly_heart.core.stt.diagnostics import recognition_cause

        reason = code[4:]
        return reason in {"none", "unclassified"} or recognition_cause(reason) == reason
    if code.startswith("capture.self."):
        from puripuly_heart.core.self_capture import SelfCaptureFailureReason

        return code[len("capture.self.") :] in {
            reason.value for reason in SelfCaptureFailureReason
        } | {"unknown"}
    if code.startswith("capture.peer."):
        from puripuly_heart.core.peer_capture import PeerCaptureFailureReason

        return code[len("capture.peer.") :] in {reason.value for reason in PeerCaptureFailureReason}
    if code.startswith("openrouter.auth."):
        from puripuly_heart.core.openrouter.authentication import OpenRouterAuthReason

        return code[len("openrouter.auth.") :] in get_args(OpenRouterAuthReason)
    if code.startswith("qq."):
        return code[3:] in QQ_AUTH_CODES
    return False
