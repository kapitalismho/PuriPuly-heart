from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, is_dataclass
from typing import get_args, get_type_hints

import pytest

from puripuly_heart.core import messages


def test_message_and_diagnostic_dtos_are_frozen_slotted_and_metadata_only() -> None:
    message = messages.UserMessageRef(
        key="settings.save.failed",
        params={"profile": "default", "attempt": 2, "retryable": False, "ratio": 0.5, "hint": None},
        severity=messages.SEVERITY_ERROR,
    )
    diagnostics = messages.ErrorDiagnostics(
        component="settings.apply",
        operation="commit",
        code="settings_commit_failed",
        category=messages.DIAGNOSTIC_CATEGORY_TRANSACTION,
        visibility=messages.DIAGNOSTIC_VISIBILITY_DETAILED,
        content_policy=messages.CONTENT_POLICY_METADATA_ONLY,
        status_code=None,
        retry_after_ms=None,
        fields={"phase": "settings_commit", "attempt": 2, "retryable": False, "latency_ms": 1.25},
    )

    for instance in (message, diagnostics):
        assert is_dataclass(instance)
        assert not hasattr(instance, "__dict__")

    with pytest.raises(FrozenInstanceError):
        message.key = "settings.save.ok"  # type: ignore[misc]
    with pytest.raises(TypeError):
        message.params["profile"] = "other"  # type: ignore[index]
    with pytest.raises(TypeError):
        diagnostics.fields["phase"] = "runtime_apply"  # type: ignore[index]


def test_result_statuses_cover_settings_runtime_secret_and_compensation_flows() -> None:
    assert set(messages.RUNTIME_APPLY_RESULT_STATUSES) == {
        messages.RUNTIME_APPLY_STATUS_APPLIED,
        messages.RUNTIME_APPLY_STATUS_DEGRADED,
        messages.RUNTIME_APPLY_STATUS_FAILED,
    }
    assert set(messages.TRANSACTION_RESULT_STATUSES) == {
        messages.TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED,
        messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED,
        messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED,
        messages.TRANSACTION_STATUS_SECRET_WRITE_FAILED,
        messages.TRANSACTION_STATUS_PROVIDER_VERIFICATION_FAILED,
        messages.TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED_SECRET_RESTORED,
        messages.TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED_SECRET_RESTORE_FAILED,
        messages.TRANSACTION_STATUS_REMOTE_ACTIVE_LOCAL_MISSING,
        messages.TRANSACTION_STATUS_REMOTE_DELIVERY_ACK_PENDING,
    }


def test_result_dtos_carry_message_refs_and_diagnostics_not_localized_text() -> None:
    forbidden_localized_fields = {
        "text",
        "title",
        "description",
        "localized_text",
        "localized_title",
    }

    for result_type in (messages.RuntimeApplyResult, messages.TransactionResult):
        field_names = {field.name for field in fields(result_type)}
        assert field_names == {"status", "message", "diagnostics"}
        assert field_names.isdisjoint(forbidden_localized_fields)

        hints = get_type_hints(result_type)
        assert hints["message"] == messages.UserMessageRef | None
        assert hints["diagnostics"] == messages.ErrorDiagnostics | None


def test_safe_message_and_diagnostic_field_aliases_expose_downstream_limits() -> None:
    allowed_scalar_types = {str, int, float, bool, type(None)}

    assert set(get_args(messages.SafeMessageParam)) == allowed_scalar_types
    assert set(get_args(messages.DiagnosticFieldValue)) == allowed_scalar_types

    for limit_name in (
        "SAFE_MESSAGE_PARAM_KEY_MAX_LENGTH",
        "SAFE_MESSAGE_PARAM_VALUE_MAX_LENGTH",
        "SAFE_MESSAGE_PARAM_MAX_ITEMS",
        "DIAGNOSTIC_FIELD_KEY_MAX_LENGTH",
        "DIAGNOSTIC_FIELD_VALUE_MAX_LENGTH",
        "DIAGNOSTIC_FIELD_MAX_ITEMS",
    ):
        limit_value = getattr(messages, limit_name)
        assert isinstance(limit_value, int)
        assert limit_value > 0

    assert (
        messages.DIAGNOSTIC_FIELD_VALUE_MAX_LENGTH >= messages.SAFE_MESSAGE_PARAM_VALUE_MAX_LENGTH
    )
