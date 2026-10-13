from __future__ import annotations

import logging
import socket
from collections.abc import Mapping

from puripuly_heart.core.messages import ErrorDiagnostics
from puripuly_heart.core.network_diagnostics import (
    classify_transport_error,
    safe_transport_fields,
    transport_exception_chain,
)

logger = logging.getLogger(__name__)

STAGES = frozenset(
    {
        "preflight",
        "broker_request",
        "key_issuance",
        "secret_storage",
        "settings_storage",
        "delivery_ack",
        "recovery",
        "authentication",
        "translation_start",
    }
)
CODES = frozenset(
    {
        "invalid_input",
        "configuration_unavailable",
        "flow_not_ready",
        "unknown_error",
        "authentication_succeeded",
        "qq_managed_auth_succeeded",
        "translation_init_failed",
        "broker_exception",
        "broker_timeout",
        "broker_proxy",
        "broker_tls",
        "broker_dns",
        "broker_network",
        "invalid_request",
        "rate_limited",
        "internal_error",
        "trial_unavailable",
        "qq_credential_invalid",
        "qq_lifetime_used",
        "qq_already_issuing",
        "qq_assertion_failed",
        "qq_assertion_exception",
        "qq_key_unavailable",
        "qq_malformed_success_payload",
        "qq_assertion_missing_managed_key",
        "qq_managed_operation_installation_unavailable",
        "qq_managed_operation_token_store_failed_before_assert",
        "qq_managed_operation_persist_failed_before_assert",
        "qq_secret_snapshot_failed",
        "qq_secret_write_exception",
        "qq_secret_write_failed",
        "qq_settings_commit_failed_rolled_back",
        "qq_settings_commit_failed_rollback_failed",
        "qq_delivery_ack_pending",
        "qq_delivery_ack_managed_secret_missing",
        "qq_delivery_ack_recovered",
        "qq_delivery_ack_token_store_failed_before_local_key_write",
        "qq_pending_delivery_ack_persist_failed_before_local_key_write",
        "qq_delivery_ack_token_store_failed_local_key_rolled_back",
        "qq_delivery_ack_token_store_failed_local_key_rollback_failed",
        "qq_managed_operation_recovery_pending",
        "qq_managed_operation_active_recovered",
        "qq_managed_operation_active_key_missing",
        "qq_managed_operation_action_required",
        "qq_managed_operation_authorization_expired",
        "qq_other_source_pending_delivery_ack",
        "qq_other_source_pending_operation",
        "operation_status_failed",
        "operation_resume_failed",
        "status_auth_storage_failed",
        "operation_cleanup_failed",
        "network_timeout",
        "network_proxy",
        "network_tls",
        "network_dns",
        "network_network",
        "local_storage_unavailable",
        "configuration_unavailable",
        "delivery_ack_pending",
        "managed_secret_missing",
        "invalid_credential",
        "credential_mismatch",
        "mismatch",
        "lifetime_used",
        "key_unavailable",
        "broker_unavailable",
        "secret_write_failed",
        "settings_commit_failed",
        "recovery_pending",
        "action_required",
        "authorization_expired",
        "other_source_pending",
    }
)
ENUM_FIELDS = {
    "ack_status": {
        "acknowledged",
        "already_acknowledged",
        "retryable",
        "pending",
        "expired",
        "malformed",
        "missing",
        "cleanup_required",
        "invalid_token",
        "managed_secret_missing",
    },
    "managed_operation_state": {
        "AUTHENTICATED",
        "ISSUE_READY",
        "CREATING",
        "CREATE_UNKNOWN",
        "RECONCILING",
        "CLEANUP_REQUIRED",
        "CLEAN",
        "RETRY_READY",
        "DELIVERY_PENDING",
        "ACTIVE",
        "FAILED",
    },
    "proxy_source": {"env", "system", "direct", "bypass", "explicit", "unknown"},
    "tls_source": {
        "windows",
        "explicit_file",
        "explicit_directory",
        "requests_bundle",
        "curl_bundle",
        "none",
        "unknown",
    },
    "tls_backend": {"native", "openssl", "none", "unknown"},
    "network_error_kind": {"timeout", "proxy", "tls", "dns", "network", "unknown"},
    "qq_failure_subcode": {
        "invalid_credential",
        "mismatch",
        "lifetime_used",
        "rate_limited",
        "key_unavailable",
        "broker_unavailable",
    },
    "broker_class": {"retryable", "terminal", "security_fail"},
    "exception_type": {
        "TimeoutException",
        "ProxyError",
        "SSLError",
        "gaierror",
        "TransportError",
        "PermissionError",
        "FileNotFoundError",
        "ValueError",
        "RuntimeError",
        "Exception",
    },
}
NUMBER_FIELDS = {"http_status", "retry_after_ms", "os_errno", "winerror", "tls_verify_code"}


def exception_fields(exc: BaseException) -> dict[str, object]:
    fields = dict(safe_transport_fields(exc))
    chain = tuple(transport_exception_chain(exc))
    kind = (
        "dns"
        if any(isinstance(item, socket.gaierror) for item in chain)
        else (
            None
            if isinstance(exc, (PermissionError, FileNotFoundError))
            else classify_transport_error(exc)
        )
    )
    fields["network_error_kind"] = kind or "unknown"
    kinds = {
        "timeout": "TimeoutException",
        "proxy": "ProxyError",
        "tls": "SSLError",
        "dns": "gaierror",
        "network": "TransportError",
    }
    name = kinds.get(kind, type(exc).__name__)
    fields["exception_type"] = name if name in ENUM_FIELDS["exception_type"] else "Exception"
    return fields


def safe_fields(fields: Mapping[str, object]) -> dict[str, object]:
    result = {}
    for key, allowed in ENUM_FIELDS.items():
        value = fields.get(key)
        if type(value) is str and value in allowed:
            result[key] = value
    for key in ("broker_code", "broker_subcode"):
        value = fields.get(key)
        if type(value) is str and value in CODES:
            result[key] = value
    for key in NUMBER_FIELDS:
        value = fields.get(key)
        if type(value) is int and -100000 <= value <= 86400000:
            result[key] = value
    return result


def log_qq_auth(
    stage: str,
    code: str,
    *,
    fields: Mapping[str, object] | None = None,
    status_code: int | None = None,
    level: int = logging.ERROR,
) -> None:
    stage = stage if stage in STAGES else "authentication"
    code = code if code in CODES else "unknown_error"
    metadata = safe_fields(fields or {})
    if type(status_code) is int and 100 <= status_code <= 599:
        metadata["http_status"] = status_code
    event = "success" if level == logging.INFO else "failure"
    category = (
        "network"
        if metadata.get("network_error_kind") not in {None, "unknown"}
        or code.startswith("network_")
        else (
            "auth"
            if code
            in {
                "qq_credential_invalid",
                "invalid_credential",
                "credential_mismatch",
                "mismatch",
                "invalid_input",
            }
            or metadata.get("qq_failure_subcode") in {"invalid_credential", "mismatch"}
            else (
                "rate_limit"
                if code == "rate_limited" or metadata.get("qq_failure_subcode") == "rate_limited"
                else (
                    "quota"
                    if code in {"lifetime_used", "qq_lifetime_used"}
                    or metadata.get("qq_failure_subcode") == "lifetime_used"
                    else "storage" if stage in {"secret_storage", "settings_storage"} else stage
                )
            )
        )
    )
    text = f"[QQAuth] {event} stage={stage} code={code} category={category}"
    text += "".join(f" {key}={value}" for key, value in sorted(metadata.items()))
    from puripuly_heart.core.runtime_logging import emit_basic_log

    emit_basic_log(logger, text, level=level)


def log_qq_diagnostics(
    diagnostics: ErrorDiagnostics | None,
    *,
    stage: str | None = None,
    fallback_code: str = "unknown_error",
    level: int = logging.ERROR,
) -> None:
    fields = diagnostics.fields if diagnostics is not None else {}
    phase = fields.get("phase", "")
    if stage is None:
        if phase in {"persist_managed_operation", "other_source_pending"}:
            stage = "preflight"
        elif phase in {"secret_snapshot", "secret_write", "delivery_ack_token_store"}:
            stage = "secret_storage"
        elif phase in {"settings_commit", "delivery_ack_pending_persist"}:
            stage = "settings_storage"
        elif phase in {"remote_delivery_ack", "remote_delivery_ack_recovery"}:
            stage = "delivery_ack"
        elif phase == "broker_assertion":
            stage = (
                "key_issuance"
                if fields.get("qq_failure_subcode") == "key_unavailable"
                and fields.get("broker_code") != "internal_error"
                else "broker_request"
            )
        else:
            stage = "recovery"
    code = (
        diagnostics.code if diagnostics is not None and diagnostics.code in CODES else fallback_code
    )
    metadata = dict(fields)
    if diagnostics is not None and diagnostics.retry_after_ms is not None:
        metadata["retry_after_ms"] = diagnostics.retry_after_ms
    log_qq_auth(
        stage,
        code,
        fields=metadata,
        status_code=diagnostics.status_code if diagnostics is not None else None,
        level=level,
    )
