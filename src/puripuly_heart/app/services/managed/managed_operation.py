from __future__ import annotations

import logging
import secrets
from base64 import urlsafe_b64encode
from collections.abc import Callable
from dataclasses import dataclass

from puripuly_heart.app.ports.managed_identity_state import ManagedIdentityStatePort
from puripuly_heart.app.ports.secret_store import SecretStorePort
from puripuly_heart.core.diagnostics import emit_failure, safe_exception_fields
from puripuly_heart.core.messages import ErrorDiagnostics
from puripuly_heart.core.openrouter import managed_operation_identity

MANAGED_OPERATION_SOURCE_DISCORD = "discord"
MANAGED_OPERATION_SOURCE_QQ = "qq"
MANAGED_OPERATION_RESUME_TOKEN_SECRET = "openrouter_managed_operation_resume_token"

MANAGED_OPERATION_STATUSES = frozenset(
    {
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
    }
)

MANAGED_OPERATION_UNKNOWN_OPERATION_STATUS = "UNKNOWN_OPERATION"

OPERATION_WAIT_STATUSES = frozenset(
    {
        "AUTHENTICATED",
        "ISSUE_READY",
        "CREATING",
        "CREATE_UNKNOWN",
        "RECONCILING",
        "CLEANUP_REQUIRED",
        "CLEAN",
    }
)

OPERATION_ACTIVE_STATUSES = frozenset({"DELIVERY_PENDING", "ACTIVE"})

DEFAULT_MAX_STATUS_POLLS = 60
MANAGED_OPERATION_MAX_CONSECUTIVE_PROBE_FAILURES = 5
DEFAULT_STATUS_POLL_INTERVAL_MS = 1000
DEFAULT_MAX_STATUS_POLL_INTERVAL_MS = 5000


@dataclass(frozen=True, slots=True)
class ManagedOperationIdentity:
    operation_id: str
    source: str
    installation_id: str
    last_known_state: str | None = None


class ManagedOperationTokenStoreError(Exception):
    def __init__(self, message: str, diagnostics: ErrorDiagnostics | None = None) -> None:
        self.diagnostics = diagnostics
        super().__init__(message)


class ManagedOperationTokenClearError(Exception):
    def __init__(self, message: str, diagnostics: ErrorDiagnostics | None = None) -> None:
        self.diagnostics = diagnostics
        super().__init__(message)


def new_managed_operation_id() -> str:
    return managed_operation_identity.MANAGED_OPERATION_ID_PREFIX + _unpadded_b64url(
        secrets.token_bytes(24)
    )


def new_managed_operation_resume_token() -> str:
    return _unpadded_b64url(secrets.token_bytes(32))


def _unpadded_b64url(raw: bytes) -> str:
    return urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def is_valid_resume_token(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 43:
        return False
    return all(character.isalnum() or character in "-_" for character in value)


def read_pending_operation(
    managed_state: ManagedIdentityStatePort,
    *,
    source: str = MANAGED_OPERATION_SOURCE_DISCORD,
) -> ManagedOperationIdentity | None:
    operation_id = getattr(managed_state, "pending_managed_operation_id", None)
    pending_source = getattr(managed_state, "pending_managed_operation_source", None)
    installation_id = getattr(managed_state, "pending_managed_operation_installation_id", None)
    if (
        not managed_operation_identity.is_valid_operation_id(operation_id)
        or pending_source != source
        or not isinstance(installation_id, str)
        or not installation_id.strip()
    ):
        return None
    last_known_state = getattr(managed_state, "pending_managed_operation_state", None)
    if last_known_state not in MANAGED_OPERATION_STATUSES and (
        last_known_state != MANAGED_OPERATION_UNKNOWN_OPERATION_STATUS
    ):
        last_known_state = None
    return ManagedOperationIdentity(
        operation_id=operation_id,
        source=source,
        installation_id=installation_id,
        last_known_state=last_known_state,
    )


def read_any_pending_operation(
    managed_state: ManagedIdentityStatePort,
) -> ManagedOperationIdentity | None:
    operation_id = getattr(managed_state, "pending_managed_operation_id", None)
    pending_source = getattr(managed_state, "pending_managed_operation_source", None)
    installation_id = getattr(managed_state, "pending_managed_operation_installation_id", None)
    if (
        not managed_operation_identity.is_valid_operation_id(operation_id)
        or pending_source not in (MANAGED_OPERATION_SOURCE_DISCORD, MANAGED_OPERATION_SOURCE_QQ)
        or not isinstance(installation_id, str)
        or not installation_id.strip()
    ):
        return None
    last_known_state = getattr(managed_state, "pending_managed_operation_state", None)
    if last_known_state not in MANAGED_OPERATION_STATUSES and (
        last_known_state != MANAGED_OPERATION_UNKNOWN_OPERATION_STATUS
    ):
        last_known_state = None
    return ManagedOperationIdentity(
        operation_id=operation_id,
        source=pending_source,
        installation_id=installation_id,
        last_known_state=last_known_state,
    )


def other_source_pending_operation(
    managed_state: ManagedIdentityStatePort,
    *,
    source: str,
) -> ManagedOperationIdentity | None:
    pending = read_any_pending_operation(managed_state)
    if pending is None or pending.source == source:
        return None
    return pending


def clear_pending_operation_if_source(
    managed_state: ManagedIdentityStatePort,
    source: str,
) -> bool:
    pending_source = getattr(managed_state, "pending_managed_operation_source", None)
    if pending_source is not None and pending_source != source:
        return False
    clear_pending_operation(managed_state)
    return True


def write_pending_operation(
    managed_state: ManagedIdentityStatePort,
    identity: ManagedOperationIdentity,
) -> None:
    managed_state.pending_managed_operation_id = identity.operation_id
    managed_state.pending_managed_operation_source = identity.source
    managed_state.pending_managed_operation_installation_id = identity.installation_id
    managed_state.pending_managed_operation_state = identity.last_known_state


def update_pending_operation_state(
    managed_state: ManagedIdentityStatePort,
    operation_status: str | None,
) -> None:
    if operation_status in MANAGED_OPERATION_STATUSES:
        managed_state.pending_managed_operation_state = operation_status


def clear_pending_operation(managed_state: ManagedIdentityStatePort) -> None:
    managed_state.pending_managed_operation_id = None
    managed_state.pending_managed_operation_source = None
    managed_state.pending_managed_operation_installation_id = None
    managed_state.pending_managed_operation_state = None


async def store_resume_token(
    secret_store: SecretStorePort,
    resume_token: str,
) -> None:
    try:
        result = await secret_store.set_secret(MANAGED_OPERATION_RESUME_TOKEN_SECRET, resume_token)
    except Exception as exc:
        raise ManagedOperationTokenStoreError("managed operation token store failed") from exc
    if not result.succeeded:
        raise ManagedOperationTokenStoreError(
            "managed operation token store failed", result.diagnostics
        )


async def read_resume_token(secret_store: SecretStorePort) -> str | None:
    try:
        result = await secret_store.get_secret(MANAGED_OPERATION_RESUME_TOKEN_SECRET)
    except Exception as exc:
        _observe_operation_failure("secret_observation", exc)
        return None
    if result.diagnostics is not None:
        emit_failure(
            logging.getLogger(__name__),
            result.diagnostics,
            severity="warning",
            disposition="degraded",
        )
    value = result.value
    return value if is_valid_resume_token(value) else None


async def clear_resume_token(secret_store: SecretStorePort) -> None:
    try:
        result = await secret_store.clear_secret(MANAGED_OPERATION_RESUME_TOKEN_SECRET)
    except Exception as exc:
        raise ManagedOperationTokenClearError("managed operation token clear failed") from exc
    if not result.succeeded:
        raise ManagedOperationTokenClearError(
            "managed operation token clear failed", result.diagnostics
        )


def status_poll_delay_ms(poll_index: int, base_interval_ms: int, max_interval_ms: int) -> int:
    delay = base_interval_ms * (2 ** min(poll_index, 4))
    return min(delay, max_interval_ms)


def managed_operation_ui_phase(
    operation_status: str | None,
    client_action: str | None,
) -> str:
    if operation_status == "FAILED":
        return "action_required"
    if operation_status == "ACTIVE":
        return "ready"
    if client_action == "action_required":
        return "action_required"
    if operation_status in OPERATION_ACTIVE_STATUSES:
        return "ready"
    if operation_status in OPERATION_WAIT_STATUSES or client_action in {
        "wait",
        "retry_authorized",
        "acknowledge_delivery",
    }:
        return "recovering"
    return "preparing"


ProgressSink = Callable[[str], None]


def emit_progress(sink: ProgressSink | None, phase: str) -> None:
    if sink is None:
        return
    try:
        sink(phase)
    except Exception as exc:
        _observe_operation_failure("auth_notification", exc)


def _observe_operation_failure(phase: str, exception: BaseException) -> None:
    diagnostics = getattr(exception, "diagnostics", None)
    if not isinstance(diagnostics, ErrorDiagnostics):
        diagnostics = ErrorDiagnostics(
            component="managed_connection_auth",
            operation=phase,
            code="managed_application_failed",
            category="transaction",
            visibility="basic",
            content_policy="metadata_only",
            status_code=None,
            retry_after_ms=None,
            fields={"phase": phase, **safe_exception_fields(exception)},
        )
    emit_failure(
        logging.getLogger(__name__), diagnostics, severity="warning", disposition="degraded"
    )


__all__ = [
    "DEFAULT_MAX_STATUS_POLLS",
    "DEFAULT_MAX_STATUS_POLL_INTERVAL_MS",
    "DEFAULT_STATUS_POLL_INTERVAL_MS",
    "MANAGED_OPERATION_MAX_CONSECUTIVE_PROBE_FAILURES",
    "MANAGED_OPERATION_RESUME_TOKEN_SECRET",
    "MANAGED_OPERATION_SOURCE_DISCORD",
    "MANAGED_OPERATION_SOURCE_QQ",
    "MANAGED_OPERATION_STATUSES",
    "MANAGED_OPERATION_UNKNOWN_OPERATION_STATUS",
    "OPERATION_ACTIVE_STATUSES",
    "OPERATION_WAIT_STATUSES",
    "ManagedOperationIdentity",
    "ManagedOperationTokenClearError",
    "ManagedOperationTokenStoreError",
    "ProgressSink",
    "clear_pending_operation",
    "clear_pending_operation_if_source",
    "clear_resume_token",
    "emit_progress",
    "is_valid_resume_token",
    "managed_operation_ui_phase",
    "new_managed_operation_id",
    "new_managed_operation_resume_token",
    "other_source_pending_operation",
    "read_any_pending_operation",
    "read_pending_operation",
    "read_resume_token",
    "status_poll_delay_ms",
    "store_resume_token",
    "update_pending_operation_state",
    "write_pending_operation",
]
