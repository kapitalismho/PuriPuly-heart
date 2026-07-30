from __future__ import annotations

import hashlib
import importlib
from collections.abc import Mapping
from typing import Any

import pytest

from puripuly_heart.app.ports import provider_verifier, secret_store, settings_repository
from puripuly_heart.core import messages

SERVICE_MODULE = "puripuly_heart.app.services.secret_settings_transaction"

RAW_SET_SECRET = "sk-test-order26-new-secret-must-not-leak"
RAW_PREVIOUS_SECRET = "sk-test-order26-previous-secret-must-not-leak"
RAW_RESTORE_DIAGNOSTIC = "sk-test-order26-restore-diagnostic-must-not-leak"
RAW_VERIFY_SECRET = "sk-test-order27-verify-secret-must-not-leak"
RAW_VERIFY_KEY_MATERIAL = "sk-order27-material-must-not-leak"


class RecordingSecretStore:
    def __init__(
        self,
        secrets: dict[str, str] | None = None,
        *,
        events: list[tuple[str, str]] | None = None,
        fail_set: bool = False,
        fail_clear: bool = False,
        fail_restore: bool = False,
    ) -> None:
        self.secrets = dict(secrets or {})
        self.events = events if events is not None else []
        self.fail_set = fail_set
        self.fail_clear = fail_clear
        self.fail_restore = fail_restore
        self.snapshots: list[secret_store.SecretSnapshot] = []
        self.restores: list[secret_store.SecretSnapshot] = []

    async def get_secret(self, key: str) -> secret_store.SecretReadResult:
        value = self.secrets.get(key)
        return secret_store.SecretReadResult(
            key=key,
            value=value,
            revision="secret-current" if value is not None else None,
            message=None,
            diagnostics=None,
        )

    async def set_secret(self, key: str, value: str) -> secret_store.SecretWriteResult:
        self.events.append(("set", key))
        if self.fail_set:
            return secret_store.SecretWriteResult(
                succeeded=False,
                key=key,
                revision=None,
                message=None,
                diagnostics=_secret_store_diagnostics(
                    "set_failed",
                    operation="set_secret",
                    raw_value=value,
                ),
            )
        self.secrets[key] = value
        return secret_store.SecretWriteResult(
            succeeded=True,
            key=key,
            revision="secret-written",
            message=None,
            diagnostics=None,
        )

    async def clear_secret(self, key: str) -> secret_store.SecretWriteResult:
        self.events.append(("clear", key))
        if self.fail_clear:
            return secret_store.SecretWriteResult(
                succeeded=False,
                key=key,
                revision=None,
                message=None,
                diagnostics=_secret_store_diagnostics(
                    "clear_failed",
                    operation="clear_secret",
                    raw_value=self.secrets.get(key) or RAW_RESTORE_DIAGNOSTIC,
                ),
            )
        self.secrets.pop(key, None)
        return secret_store.SecretWriteResult(
            succeeded=True,
            key=key,
            revision="secret-cleared",
            message=None,
            diagnostics=None,
        )

    async def snapshot_secret(self, key: str) -> secret_store.SecretSnapshot:
        self.events.append(("snapshot", key))
        value = self.secrets.get(key)
        snapshot = secret_store.SecretSnapshot(
            key=key,
            value=value,
            revision="secret-before" if value is not None else None,
            existed=value is not None,
        )
        self.snapshots.append(snapshot)
        return snapshot

    async def restore_secret(
        self,
        snapshot: secret_store.SecretSnapshot,
    ) -> secret_store.SecretWriteResult:
        self.events.append(("restore", snapshot.key))
        self.restores.append(snapshot)
        if self.fail_restore:
            return secret_store.SecretWriteResult(
                succeeded=False,
                key=snapshot.key,
                revision=None,
                message=None,
                diagnostics=_secret_store_diagnostics(
                    "restore_failed",
                    operation="restore_secret",
                    raw_value=RAW_RESTORE_DIAGNOSTIC,
                ),
            )
        if snapshot.existed:
            assert snapshot.value is not None
            self.secrets[snapshot.key] = snapshot.value
        else:
            self.secrets.pop(snapshot.key, None)
        return secret_store.SecretWriteResult(
            succeeded=True,
            key=snapshot.key,
            revision="secret-restored",
            message=None,
            diagnostics=None,
        )


class RecordingSettingsRepository:
    def __init__(
        self,
        result: settings_repository.SettingsCommitResult,
        *,
        events: list[tuple[str, str]] | None = None,
        raise_on_save: bool = False,
    ) -> None:
        self.result = result
        self.events = events if events is not None else []
        self.raise_on_save = raise_on_save
        self.saved_requests: list[settings_repository.SettingsCommitRequest] = []

    async def load(self) -> settings_repository.SettingsSnapshot:
        raise AssertionError("SecretSettingsTransaction should not load settings here")

    async def save(
        self,
        request: settings_repository.SettingsCommitRequest,
    ) -> settings_repository.SettingsCommitResult:
        self.events.append(("save", request.reason or ""))
        self.saved_requests.append(request)
        if self.raise_on_save:
            raise RuntimeError("simulated settings save failure")
        return self.result


class RecordingDashboardNeedsKeyPublisher:
    def __init__(
        self,
        *,
        events: list[tuple[str, str]] | None = None,
        fail_publish: bool = False,
    ) -> None:
        self.events = events if events is not None else []
        self.fail_publish = fail_publish
        self.publications: list[tuple[Any, str | None]] = []

    async def publish_dashboard_needs_key_snapshot(
        self,
        snapshot: Any,
        *,
        correlation_id: str | None,
    ) -> None:
        self.events.append(("publish", correlation_id or ""))
        if self.fail_publish:
            raise RuntimeError("simulated dashboard needs-key publish failure")
        self.publications.append((snapshot, correlation_id))


class RecordingProviderVerifier:
    def __init__(
        self,
        result: provider_verifier.ProviderVerificationResult | None = None,
        *,
        events: list[tuple[str, str]] | None = None,
        raise_on_verify: bool = False,
    ) -> None:
        self.result = result
        self.events = events if events is not None else []
        self.raise_on_verify = raise_on_verify
        self.requests: list[provider_verifier.ProviderVerificationRequest] = []

    async def verify_provider_secret(
        self,
        request: provider_verifier.ProviderVerificationRequest,
    ) -> provider_verifier.ProviderVerificationResult:
        self.events.append(("verify", request.provider))
        self.requests.append(request)
        if self.raise_on_verify:
            raise RuntimeError("simulated verifier failure with redacted secret")
        if self.result is None:
            pytest.fail("RecordingProviderVerifier requires a configured result")
        return self.result


def _service_module():
    return importlib.import_module(SERVICE_MODULE)


def _commit_success(
    values: dict[str, object],
    *,
    revision: str = "settings-r2",
) -> settings_repository.SettingsCommitResult:
    return settings_repository.SettingsCommitResult(
        succeeded=True,
        snapshot=settings_repository.SettingsSnapshot(values=values, revision=revision),
        message=None,
        diagnostics=None,
    )


def _commit_failure() -> settings_repository.SettingsCommitResult:
    return settings_repository.SettingsCommitResult(
        succeeded=False,
        snapshot=None,
        message=None,
        diagnostics=messages.ErrorDiagnostics(
            component="settings_repository",
            operation="save",
            code="settings_commit_failed",
            category=messages.DIAGNOSTIC_CATEGORY_TRANSACTION,
            visibility=messages.DIAGNOSTIC_VISIBILITY_BASIC,
            content_policy=messages.CONTENT_POLICY_METADATA_ONLY,
            status_code=None,
            retry_after_ms=None,
            fields={"phase": "settings_commit"},
        ),
    )


def _secret_store_diagnostics(
    code: str,
    *,
    operation: str,
    raw_value: str,
) -> messages.ErrorDiagnostics:
    return messages.ErrorDiagnostics(
        component="fake_secret_store",
        operation=operation,
        code=code,
        category=messages.DIAGNOSTIC_CATEGORY_TRANSACTION,
        visibility=messages.DIAGNOSTIC_VISIBILITY_BASIC,
        content_policy=messages.CONTENT_POLICY_METADATA_ONLY,
        status_code=None,
        retry_after_ms=None,
        fields={"raw_value_that_service_must_not_return": raw_value},
    )


FORBIDDEN_RAW_SECRET_VALUES = (
    RAW_SET_SECRET,
    RAW_PREVIOUS_SECRET,
    RAW_RESTORE_DIAGNOSTIC,
    RAW_VERIFY_SECRET,
    RAW_VERIFY_KEY_MATERIAL,
)


def _assert_no_raw_secret_values(value: object, *, label: str = "value") -> None:
    rendered = repr(value)
    for index, raw in enumerate(FORBIDDEN_RAW_SECRET_VALUES, start=1):
        if raw in rendered:
            pytest.fail(f"{label} repr exposed forbidden raw secret sentinel #{index}")


def _redacted_repr(value: object) -> str:
    rendered = repr(value)
    for raw in FORBIDDEN_RAW_SECRET_VALUES:
        rendered = rendered.replace(raw, "<raw-secret-redacted>")
    return rendered


def _secret_fingerprint(value: str | None) -> tuple[str] | tuple[str, int, str]:
    if value is None:
        return ("absent",)
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return ("present", len(value), digest)


def _assert_secret_value_matches(
    actual: str | None,
    expected: str | None,
    *,
    label: str,
) -> None:
    actual_fingerprint = _secret_fingerprint(actual)
    expected_fingerprint = _secret_fingerprint(expected)
    if actual_fingerprint != expected_fingerprint:
        pytest.fail(
            f"{label} secret fingerprint mismatch: "
            f"actual={actual_fingerprint!r}, expected={expected_fingerprint!r}"
        )


def _assert_secret_key_absent(
    secrets: Mapping[str, str],
    key: str,
    *,
    label: str,
) -> None:
    if key in secrets:
        pytest.fail(
            f"{label} unexpectedly retained secret key {key!r} with "
            f"fingerprint={_secret_fingerprint(secrets.get(key))!r}"
        )


def _values_match(actual: object, expected: object) -> bool:
    if isinstance(expected, bool):
        return actual is expected
    return actual == expected


def _assert_diagnostics_field_matches(
    actual_fields: Mapping[str, Any],
    key: str,
    expected_value: object,
    *,
    label: str,
) -> None:
    if key not in actual_fields:
        pytest.fail(f"{label} missing field {key!r}; actual keys={sorted(actual_fields)!r}")

    actual_value = actual_fields[key]
    if not _values_match(actual_value, expected_value):
        pytest.fail(
            f"{label} field {key!r} mismatch: "
            f"actual={_redacted_repr(actual_value)}, "
            f"expected={_redacted_repr(expected_value)}"
        )


def _assert_diagnostics_fields_match(
    actual_fields: Mapping[str, Any],
    expected_fields: Mapping[str, Any],
    *,
    label: str,
) -> None:
    actual_keys = set(actual_fields)
    expected_keys = set(expected_fields)
    if actual_keys != expected_keys:
        pytest.fail(
            f"{label} field keys mismatch: "
            f"actual={sorted(actual_keys)!r}, expected={sorted(expected_keys)!r}"
        )

    for key, expected_value in expected_fields.items():
        actual_value = actual_fields[key]
        if not _values_match(actual_value, expected_value):
            pytest.fail(
                f"{label} field {key!r} mismatch: "
                f"actual={_redacted_repr(actual_value)}, "
                f"expected={_redacted_repr(expected_value)}"
            )


def _only_item(items: list[Any], *, label: str) -> Any:
    if len(items) != 1:
        pytest.fail(f"{label} count mismatch: actual={len(items)}, expected=1")
    return items[0]


def _provider_verification_entry(
    values: Mapping[str, object],
    provider: str,
    *,
    label: str = "values",
) -> Mapping[str, object]:
    state = values.get("state")
    if not isinstance(state, Mapping):
        pytest.fail(f"{label} missing nested 'state' mapping")
    provider_verification = state.get("provider_verification")
    if not isinstance(provider_verification, Mapping):
        pytest.fail(f"{label} missing nested 'state.provider_verification' mapping")
    entry = provider_verification.get(provider)
    if not isinstance(entry, Mapping):
        pytest.fail(f"{label} missing nested 'state.provider_verification.{provider}' mapping")
    return entry


def _assert_secret_snapshot_matches(
    snapshot: secret_store.SecretSnapshot,
    *,
    expected_key: str,
    expected_value: str | None,
    expected_revision: str | None,
    expected_existed: bool,
    label: str,
) -> None:
    if snapshot.key != expected_key:
        pytest.fail(
            f"{label} key mismatch: "
            f"actual={_redacted_repr(snapshot.key)}, expected={_redacted_repr(expected_key)}"
        )
    if snapshot.revision != expected_revision:
        pytest.fail(
            f"{label} revision mismatch: "
            f"actual={_redacted_repr(snapshot.revision)}, "
            f"expected={_redacted_repr(expected_revision)}"
        )
    if snapshot.existed is not expected_existed:
        pytest.fail(
            f"{label} existed mismatch: "
            f"actual={snapshot.existed!r}, expected={expected_existed!r}"
        )
    _assert_secret_value_matches(
        snapshot.value,
        expected_value,
        label=f"{label} value",
    )


@pytest.mark.asyncio
async def test_verify_provider_secret_commits_bound_evidence_after_verifier_success() -> None:
    secret_settings = _service_module()
    events: list[tuple[str, str]] = []
    verifier = RecordingProviderVerifier(
        provider_verifier.ProviderVerificationResult(
            status="verified",
            provider="openrouter",
            secret_key="openrouter_api_key",
            secret_revision="secret-r1",
            evidence={
                "verifier": "openrouter",
                "latency_ms": 12.5,
                "raw_provider_payload": RAW_VERIFY_SECRET,
                "api_key": RAW_VERIFY_SECRET,
            },
            message=None,
            diagnostics=None,
        ),
        events=events,
    )
    repository = RecordingSettingsRepository(
        _commit_success({"state.provider_verification.openrouter": {}}),
        events=events,
    )
    request = secret_settings.ProviderSecretVerificationRequest(
        provider="openrouter",
        secret_key="openrouter_api_key",
        secret_value=RAW_VERIFY_SECRET,
        secret_revision="secret-r1",
        verifier_context={"flow": "settings.verify_api_key"},
        expected_settings_revision="settings-r1",
        reason="api_key_verify",
        correlation_id="corr-verify-success",
    )
    _assert_no_raw_secret_values(request, label="ProviderSecretVerificationRequest")

    result = await secret_settings.SecretSettingsTransaction(
        secret_store=RecordingSecretStore(events=events),
        settings_repository=repository,
        provider_verifier=verifier,
    ).verify_provider_secret(request)

    _assert_no_raw_secret_values(result, label="verification success result")
    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED
    assert events == [("verify", "openrouter"), ("save", "api_key_verify")]
    verifier_request = _only_item(verifier.requests, label="provider verifier requests")
    _assert_no_raw_secret_values(verifier_request, label="provider verifier request")
    _assert_secret_value_matches(
        verifier_request.secret_value,
        RAW_VERIFY_SECRET,
        label="provider verifier secret value",
    )
    assert len(repository.saved_requests) == 1
    saved_request = repository.saved_requests[0]
    _assert_no_raw_secret_values(saved_request, label="verification settings commit request")
    entry = _provider_verification_entry(
        saved_request.values, "openrouter", label="verification settings commit request"
    )
    assert entry["status"] == "verified"
    assert entry["provider"] == "openrouter"
    assert entry["secret_key"] == "openrouter_api_key"
    assert entry["secret_revision"] == "secret-r1"
    assert str(entry["secret_fingerprint"]).startswith("sha256:")
    assert entry["verifier_context"]["flow"] == "settings.verify_api_key"  # type: ignore[index]
    evidence = entry["verifier_evidence"]
    if not isinstance(evidence, Mapping):
        pytest.fail("verification evidence details should be a mapping")
    assert evidence["verifier"] == "openrouter"
    assert evidence["latency_ms"] == 12.5
    assert "raw_provider_payload" not in evidence
    assert "api_key" not in evidence


def test_provider_secret_verification_request_freezes_sanitized_context() -> None:
    secret_settings = _service_module()
    request = secret_settings.ProviderSecretVerificationRequest(
        provider="openrouter",
        secret_key="openrouter_api_key",
        secret_value=RAW_VERIFY_SECRET,
        secret_revision="secret-r1",
        verifier_context={"flow": "settings.verify_api_key"},
        expected_settings_revision="settings-r1",
        reason="api_key_verify",
        correlation_id="corr-verify-freeze",
    )

    with pytest.raises(TypeError):
        request.verifier_context["flow"] = "mutated"  # type: ignore[index]


@pytest.mark.asyncio
async def test_verify_provider_secret_strips_raw_secret_material_from_context_and_evidence_keys() -> (
    None
):
    secret_settings = _service_module()
    events: list[tuple[str, str]] = []
    raw_context_key = f"context-{RAW_VERIFY_KEY_MATERIAL}"
    raw_evidence_key = f"evidence-{RAW_VERIFY_KEY_MATERIAL}"
    verifier = RecordingProviderVerifier(
        provider_verifier.ProviderVerificationResult(
            status="verified",
            provider="openrouter",
            secret_key="openrouter_api_key",
            secret_revision="secret-r1",
            evidence={"verifier": "openrouter", raw_evidence_key: "drop-me"},
            message=None,
            diagnostics=None,
        ),
        events=events,
    )
    repository = RecordingSettingsRepository(
        _commit_success({"state.provider_verification.openrouter": {}}),
        events=events,
    )
    request = secret_settings.ProviderSecretVerificationRequest(
        provider="openrouter",
        secret_key="openrouter_api_key",
        secret_value=RAW_VERIFY_KEY_MATERIAL,
        secret_revision="secret-r1",
        verifier_context={"flow": "settings.verify_api_key", raw_context_key: "drop-me"},
        expected_settings_revision="settings-r1",
        reason="api_key_verify",
        correlation_id="corr-verify-raw-key",
    )

    _assert_no_raw_secret_values(request, label="raw-key verification request")
    result = await secret_settings.SecretSettingsTransaction(
        secret_store=RecordingSecretStore(events=events),
        settings_repository=repository,
        provider_verifier=verifier,
    ).verify_provider_secret(request)

    _assert_no_raw_secret_values(result, label="raw-key verification result")
    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED
    verifier_request = _only_item(verifier.requests, label="provider verifier requests")
    _assert_no_raw_secret_values(verifier_request, label="raw-key provider verifier request")
    assert set(verifier_request.context) == {"flow"}
    saved_request = _only_item(repository.saved_requests, label="verification settings saves")
    _assert_no_raw_secret_values(saved_request, label="raw-key settings commit request")
    entry = _provider_verification_entry(
        saved_request.values, "openrouter", label="raw-key verification settings commit request"
    )
    _assert_no_raw_secret_values(entry, label="raw-key verification evidence entry")
    assert set(entry["verifier_context"]) == {"flow"}  # type: ignore[arg-type]
    assert set(entry["verifier_evidence"]) == {"verifier"}  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_verify_provider_secret_rejects_empty_sanitized_context_before_verifier() -> None:
    secret_settings = _service_module()
    events: list[tuple[str, str]] = []
    verifier = RecordingProviderVerifier(events=events)
    repository = RecordingSettingsRepository(_commit_success({}), events=events)

    result = await secret_settings.SecretSettingsTransaction(
        secret_store=RecordingSecretStore(events=events),
        settings_repository=repository,
        provider_verifier=verifier,
    ).verify_provider_secret(
        secret_settings.ProviderSecretVerificationRequest(
            provider="openrouter",
            secret_key="openrouter_api_key",
            secret_value=RAW_VERIFY_SECRET,
            secret_revision="secret-r1",
            verifier_context={"api_key": "dropped-by-policy"},
            expected_settings_revision="settings-r1",
            reason="api_key_verify",
            correlation_id="corr-verify-empty-context",
        )
    )

    _assert_no_raw_secret_values(result, label="empty-context verification result")
    assert result.status == messages.TRANSACTION_STATUS_PROVIDER_VERIFICATION_FAILED
    assert events == []
    assert repository.saved_requests == []
    assert result.diagnostics is not None
    assert result.diagnostics.code == "provider_verification_context_missing"
    assert result.diagnostics.content_policy == messages.CONTENT_POLICY_METADATA_ONLY


@pytest.mark.asyncio
async def test_verify_provider_secret_failure_commits_failed_evidence_without_raw_diagnostics() -> (
    None
):
    secret_settings = _service_module()
    events: list[tuple[str, str]] = []
    verifier = RecordingProviderVerifier(
        provider_verifier.ProviderVerificationResult(
            status="failed",
            provider="openrouter",
            secret_key="openrouter_api_key",
            secret_revision="secret-r1",
            evidence={"error_code": "unauthorized", "response_body": RAW_VERIFY_SECRET},
            message=None,
            diagnostics=_secret_store_diagnostics(
                "provider_rejected_secret",
                operation="verify_provider_secret",
                raw_value=RAW_VERIFY_SECRET,
            ),
        ),
        events=events,
    )
    repository = RecordingSettingsRepository(
        _commit_success({"state.provider_verification.openrouter": {}}),
        events=events,
    )

    result = await secret_settings.SecretSettingsTransaction(
        secret_store=RecordingSecretStore(events=events),
        settings_repository=repository,
        provider_verifier=verifier,
    ).verify_provider_secret(
        secret_settings.ProviderSecretVerificationRequest(
            provider="openrouter",
            secret_key="openrouter_api_key",
            secret_value=RAW_VERIFY_SECRET,
            secret_revision="secret-r1",
            verifier_context={"flow": "settings.verify_api_key"},
            expected_settings_revision="settings-r1",
            reason="api_key_verify",
            correlation_id="corr-verify-failed",
        )
    )

    _assert_no_raw_secret_values(result, label="verification failed result")
    assert result.status == messages.TRANSACTION_STATUS_PROVIDER_VERIFICATION_FAILED
    assert events == [("verify", "openrouter"), ("save", "api_key_verify")]
    saved_request = _only_item(repository.saved_requests, label="verification settings saves")
    _assert_no_raw_secret_values(saved_request, label="failed verification settings commit request")
    entry = _provider_verification_entry(
        saved_request.values, "openrouter", label="failed verification settings commit request"
    )
    assert entry["status"] == "failed"
    evidence = entry["verifier_evidence"]
    if not isinstance(evidence, Mapping):
        pytest.fail("failed verification evidence details should be a mapping")
    assert evidence["error_code"] == "unauthorized"
    assert "response_body" not in evidence
    assert result.diagnostics is not None
    assert result.diagnostics.component == "secret_settings_transaction"
    assert result.diagnostics.operation == "verify_provider_secret"
    assert result.diagnostics.content_policy == messages.CONTENT_POLICY_METADATA_ONLY
    _assert_diagnostics_field_matches(
        result.diagnostics.fields,
        "provider",
        "openrouter",
        label="verification failed diagnostics",
    )


@pytest.mark.asyncio
async def test_verify_provider_secret_exception_returns_safe_failure_without_settings_commit() -> (
    None
):
    secret_settings = _service_module()
    events: list[tuple[str, str]] = []
    verifier = RecordingProviderVerifier(events=events, raise_on_verify=True)
    repository = RecordingSettingsRepository(_commit_success({}), events=events)

    result = await secret_settings.SecretSettingsTransaction(
        secret_store=RecordingSecretStore(events=events),
        settings_repository=repository,
        provider_verifier=verifier,
    ).verify_provider_secret(
        secret_settings.ProviderSecretVerificationRequest(
            provider="openrouter",
            secret_key="openrouter_api_key",
            secret_value=RAW_VERIFY_SECRET,
            secret_revision="secret-r1",
            verifier_context={"flow": "settings.verify_api_key"},
            expected_settings_revision="settings-r1",
            reason="api_key_verify",
            correlation_id="corr-verify-exception",
        )
    )

    _assert_no_raw_secret_values(result, label="verification exception result")
    assert result.status == messages.TRANSACTION_STATUS_PROVIDER_VERIFICATION_FAILED
    assert events == [("verify", "openrouter")]
    assert repository.saved_requests == []
    assert result.diagnostics is not None
    assert result.diagnostics.component == "secret_settings_transaction"
    assert result.diagnostics.operation == "verify_provider_secret"
    assert result.diagnostics.content_policy == messages.CONTENT_POLICY_METADATA_ONLY


@pytest.mark.asyncio
async def test_verify_provider_secret_settings_commit_failure_returns_safe_settings_failure() -> (
    None
):
    secret_settings = _service_module()
    events: list[tuple[str, str]] = []
    verifier = RecordingProviderVerifier(
        provider_verifier.ProviderVerificationResult(
            status="verified",
            provider="openrouter",
            secret_key="openrouter_api_key",
            secret_revision="secret-r1",
            evidence={"verifier": "openrouter"},
            message=None,
            diagnostics=None,
        ),
        events=events,
    )
    repository = RecordingSettingsRepository(_commit_failure(), events=events)

    result = await secret_settings.SecretSettingsTransaction(
        secret_store=RecordingSecretStore(events=events),
        settings_repository=repository,
        provider_verifier=verifier,
    ).verify_provider_secret(
        secret_settings.ProviderSecretVerificationRequest(
            provider="openrouter",
            secret_key="openrouter_api_key",
            secret_value=RAW_VERIFY_SECRET,
            secret_revision="secret-r1",
            verifier_context={"flow": "settings.verify_api_key"},
            expected_settings_revision="settings-r1",
            reason="api_key_verify",
            correlation_id="corr-verify-commit-failed",
        )
    )

    _assert_no_raw_secret_values(result, label="verification settings failure result")
    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED
    assert events == [("verify", "openrouter"), ("save", "api_key_verify")]
    saved_request = _only_item(repository.saved_requests, label="verification settings saves")
    _assert_no_raw_secret_values(
        saved_request, label="settings failure verification commit request"
    )
    assert result.diagnostics is not None
    assert result.diagnostics.content_policy == messages.CONTENT_POLICY_METADATA_ONLY


@pytest.mark.asyncio
async def test_successful_set_snapshots_writes_commits_then_publishes_dashboard_snapshot() -> None:
    secret_settings = _service_module()
    events: list[tuple[str, str]] = []
    store = RecordingSecretStore(
        {"openrouter_api_key": RAW_PREVIOUS_SECRET},
        events=events,
    )
    repository = RecordingSettingsRepository(
        _commit_success({"caller_marker": {"openrouter": False}}),
        events=events,
    )
    publisher = RecordingDashboardNeedsKeyPublisher(events=events)
    request = secret_settings.SecretSetRequest(
        secret_key="openrouter_api_key",
        secret_value=RAW_SET_SECRET,
        settings_values={"caller_marker": {"openrouter": False}},
        expected_settings_revision="settings-r1",
        reason="secret_set",
        correlation_id="corr-set-success",
        dashboard_needs_key=secret_settings.DashboardNeedsKeySnapshot(
            translation_needs_key=True,
            stt_needs_key=None,
        ),
    )
    _assert_no_raw_secret_values(request, label="SecretSetRequest")

    result = await secret_settings.SecretSettingsTransaction(
        secret_store=store,
        settings_repository=repository,
        dashboard_needs_key_publisher=publisher,
    ).set_provider_secret(request)

    _assert_no_raw_secret_values(result, label="set success result")
    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED
    assert result.message is None
    assert result.diagnostics is None
    assert events == [
        ("snapshot", "openrouter_api_key"),
        ("set", "openrouter_api_key"),
        ("save", "secret_set"),
        ("publish", "corr-set-success"),
    ]
    _assert_secret_snapshot_matches(
        _only_item(store.snapshots, label="secret snapshots"),
        expected_key="openrouter_api_key",
        expected_value=RAW_PREVIOUS_SECRET,
        expected_revision="secret-before",
        expected_existed=True,
        label="set snapshot",
    )
    _assert_secret_value_matches(
        store.secrets.get("openrouter_api_key"),
        RAW_SET_SECRET,
        label="stored openrouter secret",
    )
    assert len(repository.saved_requests) == 1
    saved_request = repository.saved_requests[0]
    _assert_no_raw_secret_values(saved_request, label="settings commit request")
    assert saved_request.expected_revision == "settings-r1"
    assert saved_request.reason == "secret_set"
    assert saved_request.values["caller_marker"]["openrouter"] is False  # type: ignore[index]
    assert publisher.publications == [
        (
            secret_settings.DashboardNeedsKeySnapshot(
                translation_needs_key=True,
                stt_needs_key=None,
                settings_revision="settings-r2",
            ),
            "corr-set-success",
        )
    ]


@pytest.mark.asyncio
async def test_dashboard_publish_failure_does_not_change_successful_transaction_result() -> None:
    secret_settings = _service_module()
    events: list[tuple[str, str]] = []
    store = RecordingSecretStore(events=events)
    repository = RecordingSettingsRepository(
        _commit_success({"caller_marker": {"openrouter": False}}),
        events=events,
    )
    publisher = RecordingDashboardNeedsKeyPublisher(events=events, fail_publish=True)

    result = await secret_settings.SecretSettingsTransaction(
        secret_store=store,
        settings_repository=repository,
        dashboard_needs_key_publisher=publisher,
    ).set_provider_secret(
        secret_settings.SecretSetRequest(
            secret_key="openrouter_api_key",
            secret_value=RAW_SET_SECRET,
            settings_values={"caller_marker": {"openrouter": False}},
            expected_settings_revision="settings-r1",
            reason="secret_set",
            correlation_id="corr-set-publish-failed",
            dashboard_needs_key=secret_settings.DashboardNeedsKeySnapshot(
                translation_needs_key=True,
                stt_needs_key=None,
            ),
        )
    )

    _assert_no_raw_secret_values(result, label="publish failure result")
    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED
    assert events == [
        ("snapshot", "openrouter_api_key"),
        ("set", "openrouter_api_key"),
        ("save", "secret_set"),
        ("publish", "corr-set-publish-failed"),
    ]
    _assert_secret_value_matches(
        store.secrets.get("openrouter_api_key"),
        RAW_SET_SECRET,
        label="stored openrouter secret after publish failure",
    )
    assert len(repository.saved_requests) == 1
    assert publisher.publications == []


@pytest.mark.asyncio
async def test_settings_commit_failure_after_set_restores_previous_secret_without_dashboard_publish() -> (
    None
):
    secret_settings = _service_module()
    events: list[tuple[str, str]] = []
    store = RecordingSecretStore(
        {"openrouter_api_key": RAW_PREVIOUS_SECRET},
        events=events,
    )
    repository = RecordingSettingsRepository(_commit_failure(), events=events)
    publisher = RecordingDashboardNeedsKeyPublisher(events=events)

    result = await secret_settings.SecretSettingsTransaction(
        secret_store=store,
        settings_repository=repository,
        dashboard_needs_key_publisher=publisher,
    ).set_provider_secret(
        secret_settings.SecretSetRequest(
            secret_key="openrouter_api_key",
            secret_value=RAW_SET_SECRET,
            settings_values={"caller_marker": {"openrouter": False}},
            expected_settings_revision="settings-r1",
            reason="secret_set",
            correlation_id="corr-set-commit-failed",
            dashboard_needs_key=secret_settings.DashboardNeedsKeySnapshot(
                translation_needs_key=True,
                stt_needs_key=None,
            ),
        )
    )

    _assert_no_raw_secret_values(result, label="set commit failure result")
    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED_SECRET_RESTORED
    assert events == [
        ("snapshot", "openrouter_api_key"),
        ("set", "openrouter_api_key"),
        ("save", "secret_set"),
        ("restore", "openrouter_api_key"),
    ]
    _assert_secret_value_matches(
        store.secrets.get("openrouter_api_key"),
        RAW_PREVIOUS_SECRET,
        label="restored openrouter secret",
    )
    assert len(repository.saved_requests) == 1
    assert publisher.publications == []
    assert result.diagnostics is not None
    assert result.diagnostics.content_policy == messages.CONTENT_POLICY_METADATA_ONLY
    _assert_diagnostics_field_matches(
        result.diagnostics.fields,
        "secret_key",
        "openrouter_api_key",
        label="set commit failure diagnostics",
    )
    _assert_diagnostics_field_matches(
        result.diagnostics.fields,
        "previous_secret_existed",
        True,
        label="set commit failure diagnostics",
    )


@pytest.mark.asyncio
async def test_settings_save_exception_after_set_restores_previous_secret() -> None:
    secret_settings = _service_module()
    events: list[tuple[str, str]] = []
    store = RecordingSecretStore(
        {"openrouter_api_key": RAW_PREVIOUS_SECRET},
        events=events,
    )
    repository = RecordingSettingsRepository(
        _commit_success({"caller_marker": {"openrouter": False}}),
        events=events,
        raise_on_save=True,
    )
    publisher = RecordingDashboardNeedsKeyPublisher(events=events)

    result = await secret_settings.SecretSettingsTransaction(
        secret_store=store,
        settings_repository=repository,
        dashboard_needs_key_publisher=publisher,
    ).set_provider_secret(
        secret_settings.SecretSetRequest(
            secret_key="openrouter_api_key",
            secret_value=RAW_SET_SECRET,
            settings_values={"caller_marker": {"openrouter": False}},
            expected_settings_revision="settings-r1",
            reason="secret_set",
            correlation_id="corr-set-save-raised",
            dashboard_needs_key=secret_settings.DashboardNeedsKeySnapshot(
                translation_needs_key=True,
                stt_needs_key=None,
            ),
        )
    )

    _assert_no_raw_secret_values(result, label="save exception restore result")
    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED_SECRET_RESTORED
    assert events == [
        ("snapshot", "openrouter_api_key"),
        ("set", "openrouter_api_key"),
        ("save", "secret_set"),
        ("restore", "openrouter_api_key"),
    ]
    _assert_secret_value_matches(
        store.secrets.get("openrouter_api_key"),
        RAW_PREVIOUS_SECRET,
        label="restored openrouter secret after save exception",
    )
    _assert_secret_snapshot_matches(
        _only_item(store.restores, label="secret restores"),
        expected_key="openrouter_api_key",
        expected_value=RAW_PREVIOUS_SECRET,
        expected_revision="secret-before",
        expected_existed=True,
        label="save exception restore snapshot",
    )
    assert len(repository.saved_requests) == 1
    assert publisher.publications == []
    assert result.diagnostics is not None
    _assert_diagnostics_field_matches(
        result.diagnostics.fields,
        "previous_secret_existed",
        True,
        label="save exception restore diagnostics",
    )


@pytest.mark.asyncio
async def test_absent_secret_set_commit_failure_clears_newly_written_secret_on_restore() -> None:
    secret_settings = _service_module()
    events: list[tuple[str, str]] = []
    store = RecordingSecretStore(events=events)
    repository = RecordingSettingsRepository(_commit_failure(), events=events)
    publisher = RecordingDashboardNeedsKeyPublisher(events=events)

    result = await secret_settings.SecretSettingsTransaction(
        secret_store=store,
        settings_repository=repository,
        dashboard_needs_key_publisher=publisher,
    ).set_provider_secret(
        secret_settings.SecretSetRequest(
            secret_key="openrouter_api_key",
            secret_value=RAW_SET_SECRET,
            settings_values={"caller_marker": {"openrouter": False}},
            expected_settings_revision="settings-r1",
            reason="secret_set",
            correlation_id="corr-set-absent-commit-failed",
            dashboard_needs_key=secret_settings.DashboardNeedsKeySnapshot(
                translation_needs_key=True,
                stt_needs_key=None,
            ),
        )
    )

    _assert_no_raw_secret_values(result, label="absent set restore result")
    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED_SECRET_RESTORED
    assert events == [
        ("snapshot", "openrouter_api_key"),
        ("set", "openrouter_api_key"),
        ("save", "secret_set"),
        ("restore", "openrouter_api_key"),
    ]
    _assert_secret_snapshot_matches(
        _only_item(store.restores, label="secret restores"),
        expected_key="openrouter_api_key",
        expected_value=None,
        expected_revision=None,
        expected_existed=False,
        label="absent set restore snapshot",
    )
    _assert_secret_key_absent(
        store.secrets,
        "openrouter_api_key",
        label="absent set restore",
    )
    assert len(repository.saved_requests) == 1
    assert publisher.publications == []
    assert result.diagnostics is not None
    _assert_diagnostics_field_matches(
        result.diagnostics.fields,
        "previous_secret_existed",
        False,
        label="absent set restore diagnostics",
    )


@pytest.mark.asyncio
async def test_restore_failure_after_settings_commit_failure_reports_compensation_failure_only() -> (
    None
):
    secret_settings = _service_module()
    events: list[tuple[str, str]] = []
    store = RecordingSecretStore(
        {"openrouter_api_key": RAW_PREVIOUS_SECRET},
        events=events,
        fail_restore=True,
    )
    repository = RecordingSettingsRepository(_commit_failure(), events=events)
    publisher = RecordingDashboardNeedsKeyPublisher(events=events)

    result = await secret_settings.SecretSettingsTransaction(
        secret_store=store,
        settings_repository=repository,
        dashboard_needs_key_publisher=publisher,
    ).set_provider_secret(
        secret_settings.SecretSetRequest(
            secret_key="openrouter_api_key",
            secret_value=RAW_SET_SECRET,
            settings_values={"caller_marker": {"openrouter": False}},
            expected_settings_revision="settings-r1",
            reason="secret_set",
            correlation_id="corr-set-restore-failed",
            dashboard_needs_key=secret_settings.DashboardNeedsKeySnapshot(
                translation_needs_key=True,
                stt_needs_key=None,
            ),
        )
    )

    _assert_no_raw_secret_values(result, label="restore failure result")
    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED_SECRET_RESTORE_FAILED
    assert events == [
        ("snapshot", "openrouter_api_key"),
        ("set", "openrouter_api_key"),
        ("save", "secret_set"),
        ("restore", "openrouter_api_key"),
    ]
    _assert_secret_value_matches(
        store.secrets.get("openrouter_api_key"),
        RAW_SET_SECRET,
        label="unrestored openrouter secret",
    )
    assert publisher.publications == []
    assert result.diagnostics is not None
    assert result.diagnostics.component == "secret_settings_transaction"
    assert result.diagnostics.operation == "restore_secret"
    assert result.diagnostics.code == "settings_commit_failed_secret_restore_failed"
    assert result.diagnostics.content_policy == messages.CONTENT_POLICY_METADATA_ONLY
    _assert_diagnostics_fields_match(
        result.diagnostics.fields,
        {
            "secret_key": "openrouter_api_key",
            "action": "set",
            "previous_secret_existed": True,
            "settings_commit_succeeded": False,
            "secret_restore_succeeded": False,
        },
        label="restore failure diagnostics",
    )


@pytest.mark.asyncio
async def test_successful_clear_snapshots_clears_commits_then_publishes_dashboard_snapshot() -> (
    None
):
    secret_settings = _service_module()
    events: list[tuple[str, str]] = []
    store = RecordingSecretStore(
        {"deepgram_api_key": RAW_PREVIOUS_SECRET},
        events=events,
    )
    repository = RecordingSettingsRepository(
        _commit_success({"caller_marker": {"deepgram": False}}, revision="settings-r-clear"),
        events=events,
    )
    publisher = RecordingDashboardNeedsKeyPublisher(events=events)

    result = await secret_settings.SecretSettingsTransaction(
        secret_store=store,
        settings_repository=repository,
        dashboard_needs_key_publisher=publisher,
    ).clear_provider_secret(
        secret_settings.SecretClearRequest(
            secret_key="deepgram_api_key",
            settings_values={"caller_marker": {"deepgram": False}},
            expected_settings_revision="settings-r-before-clear",
            reason="secret_clear",
            correlation_id="corr-clear-success",
            dashboard_needs_key=secret_settings.DashboardNeedsKeySnapshot(
                translation_needs_key=None,
                stt_needs_key=True,
            ),
        )
    )

    _assert_no_raw_secret_values(result, label="clear success result")
    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED
    assert events == [
        ("snapshot", "deepgram_api_key"),
        ("clear", "deepgram_api_key"),
        ("save", "secret_clear"),
        ("publish", "corr-clear-success"),
    ]
    _assert_secret_key_absent(
        store.secrets,
        "deepgram_api_key",
        label="successful clear",
    )
    _assert_no_raw_secret_values(
        repository.saved_requests[0],
        label="clear settings commit request",
    )
    assert repository.saved_requests[0].values["caller_marker"]["deepgram"] is False  # type: ignore[index]
    assert publisher.publications == [
        (
            secret_settings.DashboardNeedsKeySnapshot(
                translation_needs_key=None,
                stt_needs_key=True,
                settings_revision="settings-r-clear",
            ),
            "corr-clear-success",
        )
    ]


@pytest.mark.asyncio
async def test_clear_absent_secret_commit_failure_restores_absent_snapshot_by_clearing() -> None:
    secret_settings = _service_module()
    events: list[tuple[str, str]] = []
    store = RecordingSecretStore(events=events)
    repository = RecordingSettingsRepository(_commit_failure(), events=events)
    publisher = RecordingDashboardNeedsKeyPublisher(events=events)

    result = await secret_settings.SecretSettingsTransaction(
        secret_store=store,
        settings_repository=repository,
        dashboard_needs_key_publisher=publisher,
    ).clear_provider_secret(
        secret_settings.SecretClearRequest(
            secret_key="soniox_api_key",
            settings_values={"caller_marker": {"soniox": False}},
            expected_settings_revision="settings-r-before-clear",
            reason="secret_clear",
            correlation_id="corr-clear-absent-failed",
            dashboard_needs_key=secret_settings.DashboardNeedsKeySnapshot(
                translation_needs_key=None,
                stt_needs_key=True,
            ),
        )
    )

    _assert_no_raw_secret_values(result, label="absent clear restore result")
    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED_SECRET_RESTORED
    assert events == [
        ("snapshot", "soniox_api_key"),
        ("clear", "soniox_api_key"),
        ("save", "secret_clear"),
        ("restore", "soniox_api_key"),
    ]
    _assert_secret_snapshot_matches(
        _only_item(store.restores, label="secret restores"),
        expected_key="soniox_api_key",
        expected_value=None,
        expected_revision=None,
        expected_existed=False,
        label="absent clear restore snapshot",
    )
    _assert_secret_key_absent(
        store.secrets,
        "soniox_api_key",
        label="absent clear restore",
    )
    assert publisher.publications == []
    assert result.diagnostics is not None
    _assert_diagnostics_field_matches(
        result.diagnostics.fields,
        "previous_secret_existed",
        False,
        label="absent clear restore diagnostics",
    )


@pytest.mark.asyncio
async def test_clear_write_failure_returns_secret_write_failed_without_settings_or_dashboard() -> (
    None
):
    secret_settings = _service_module()
    events: list[tuple[str, str]] = []
    store = RecordingSecretStore(
        {"deepgram_api_key": RAW_PREVIOUS_SECRET},
        events=events,
        fail_clear=True,
    )
    repository = RecordingSettingsRepository(_commit_success({}), events=events)
    publisher = RecordingDashboardNeedsKeyPublisher(events=events)

    result = await secret_settings.SecretSettingsTransaction(
        secret_store=store,
        settings_repository=repository,
        dashboard_needs_key_publisher=publisher,
    ).clear_provider_secret(
        secret_settings.SecretClearRequest(
            secret_key="deepgram_api_key",
            settings_values={"caller_marker": {"deepgram": False}},
            expected_settings_revision="settings-r-before-clear",
            reason="secret_clear",
            correlation_id="corr-clear-write-failed",
            dashboard_needs_key=secret_settings.DashboardNeedsKeySnapshot(
                translation_needs_key=None,
                stt_needs_key=True,
            ),
        )
    )

    _assert_no_raw_secret_values(result, label="clear write failure result")
    assert result.status == messages.TRANSACTION_STATUS_SECRET_WRITE_FAILED
    assert events == [("snapshot", "deepgram_api_key"), ("clear", "deepgram_api_key")]
    _assert_secret_value_matches(
        store.secrets.get("deepgram_api_key"),
        RAW_PREVIOUS_SECRET,
        label="uncleared deepgram secret after clear failure",
    )
    assert repository.saved_requests == []
    assert publisher.publications == []
    assert result.diagnostics is not None
    assert result.diagnostics.component == "secret_settings_transaction"
    assert result.diagnostics.operation == "clear_secret"
    assert result.diagnostics.content_policy == messages.CONTENT_POLICY_METADATA_ONLY


@pytest.mark.asyncio
async def test_secret_write_failure_returns_secret_write_failed_without_settings_or_dashboard() -> (
    None
):
    secret_settings = _service_module()
    events: list[tuple[str, str]] = []
    store = RecordingSecretStore(events=events, fail_set=True)
    repository = RecordingSettingsRepository(_commit_success({}), events=events)
    publisher = RecordingDashboardNeedsKeyPublisher(events=events)

    result = await secret_settings.SecretSettingsTransaction(
        secret_store=store,
        settings_repository=repository,
        dashboard_needs_key_publisher=publisher,
    ).set_provider_secret(
        secret_settings.SecretSetRequest(
            secret_key="openrouter_api_key",
            secret_value=RAW_SET_SECRET,
            settings_values={"caller_marker": {"openrouter": False}},
            expected_settings_revision="settings-r1",
            reason="secret_set",
            correlation_id="corr-set-write-failed",
            dashboard_needs_key=secret_settings.DashboardNeedsKeySnapshot(
                translation_needs_key=True,
                stt_needs_key=None,
            ),
        )
    )

    _assert_no_raw_secret_values(result, label="set write failure result")
    assert result.status == messages.TRANSACTION_STATUS_SECRET_WRITE_FAILED
    assert events == [("snapshot", "openrouter_api_key"), ("set", "openrouter_api_key")]
    assert repository.saved_requests == []
    assert publisher.publications == []
    assert result.diagnostics is not None
    assert result.diagnostics.component == "secret_settings_transaction"
    assert result.diagnostics.operation == "set_secret"
    assert result.diagnostics.content_policy == messages.CONTENT_POLICY_METADATA_ONLY
