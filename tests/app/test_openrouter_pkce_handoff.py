from __future__ import annotations

import hashlib
import importlib
from collections.abc import Mapping
from typing import Any

import pytest

from puripuly_heart.app.ports import (
    provider_verifier,
    runtime_apply,
    secret_store,
    settings_repository,
)
from puripuly_heart.core import messages

SERVICE_MODULE = "puripuly_heart.app.services.openrouter_pkce_handoff"

RAW_PKCE_SECRET = "sk-test-order29-pkce-secret-must-not-leak"
RAW_PREVIOUS_SECRET = "sk-test-order29-previous-secret-must-not-leak"
RAW_PROVIDER_PAYLOAD = "sk-test-order29-provider-payload-must-not-leak"

FORBIDDEN_RAW_SECRET_VALUES = (
    RAW_PKCE_SECRET,
    RAW_PREVIOUS_SECRET,
    RAW_PROVIDER_PAYLOAD,
)


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


class RecordingSecretStore:
    def __init__(
        self,
        secrets: dict[str, str] | None = None,
        *,
        events: list[tuple[str, str]] | None = None,
        fail_restore: bool = False,
    ) -> None:
        self.secrets = dict(secrets or {})
        self.events = events if events is not None else []
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
                diagnostics=_safe_diagnostics("restore_failed"),
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
    ) -> None:
        self.result = result
        self.events = events if events is not None else []
        self.saved_requests: list[settings_repository.SettingsCommitRequest] = []

    async def load(self) -> settings_repository.SettingsSnapshot:
        raise AssertionError("OpenRouterPkceHandoffService should not load settings here")

    async def save(
        self,
        request: settings_repository.SettingsCommitRequest,
    ) -> settings_repository.SettingsCommitResult:
        self.events.append(("save", request.reason or ""))
        self.saved_requests.append(request)
        return self.result


class RecordingRuntimeApply:
    def __init__(
        self,
        result: messages.RuntimeApplyResult | None = None,
        *,
        events: list[tuple[str, str]] | None = None,
        raise_on_apply: bool = False,
    ) -> None:
        self.result = result or messages.RuntimeApplyResult(
            status=messages.RUNTIME_APPLY_STATUS_APPLIED,
            message=None,
            diagnostics=None,
        )
        self.events = events if events is not None else []
        self.raise_on_apply = raise_on_apply
        self.requests: list[runtime_apply.RuntimeApplyRequest] = []

    async def apply_runtime(
        self,
        request: runtime_apply.RuntimeApplyRequest,
    ) -> messages.RuntimeApplyResult:
        self.events.append(("runtime_apply", request.reason or ""))
        self.requests.append(request)
        if self.raise_on_apply:
            raise RuntimeError("runtime failed after local commit")
        return self.result


def _service_module():
    return importlib.import_module(SERVICE_MODULE)


def _verification_result(
    *,
    status: provider_verifier.ProviderVerificationStatus = "verified",
    provider: str = "openrouter",
    secret_key: str = "openrouter_api_key",
) -> provider_verifier.ProviderVerificationResult:
    return provider_verifier.ProviderVerificationResult(
        status=status,
        provider=provider,
        secret_key=secret_key,
        secret_revision="secret-r1",
        evidence={
            "verifier": "openrouter",
            "latency_ms": 7.5,
            "raw_provider_payload": RAW_PROVIDER_PAYLOAD,
            "api_key": RAW_PKCE_SECRET,
        },
        message=None,
        diagnostics=_safe_diagnostics("provider_rejected_secret") if status != "verified" else None,
    )


def _commit_success() -> settings_repository.SettingsCommitResult:
    return settings_repository.SettingsCommitResult(
        succeeded=True,
        snapshot=settings_repository.SettingsSnapshot(
            values={"intent": {"translation": {"connection": "byok"}}},
            revision="settings-r2",
        ),
        message=None,
        diagnostics=None,
    )


def _commit_failure() -> settings_repository.SettingsCommitResult:
    return settings_repository.SettingsCommitResult(
        succeeded=False,
        snapshot=None,
        message=None,
        diagnostics=_safe_diagnostics("settings_commit_failed"),
    )


def _safe_diagnostics(code: str) -> messages.ErrorDiagnostics:
    return messages.ErrorDiagnostics(
        component="test_double",
        operation="test_operation",
        code=code,
        category=messages.DIAGNOSTIC_CATEGORY_TRANSACTION,
        visibility=messages.DIAGNOSTIC_VISIBILITY_BASIC,
        content_policy=messages.CONTENT_POLICY_METADATA_ONLY,
        status_code=None,
        retry_after_ms=None,
        fields={"phase": "test"},
    )


def _runtime_result(status: messages.RuntimeApplyStatus) -> messages.RuntimeApplyResult:
    return messages.RuntimeApplyResult(
        status=status,
        message=None,
        diagnostics=messages.ErrorDiagnostics(
            component="runtime_apply_adapter",
            operation="apply_runtime",
            code="runtime_apply_degraded",
            category=messages.DIAGNOSTIC_CATEGORY_LIFECYCLE,
            visibility=messages.DIAGNOSTIC_VISIBILITY_BASIC,
            content_policy=messages.CONTENT_POLICY_METADATA_ONLY,
            status_code=None,
            retry_after_ms=None,
            fields={"runtime_status": status},
        ),
    )


def _request() -> Any:
    return _request_with_settings(
        {
            "intent": {
                "translation": {
                    "connection": "byok",
                    "model": "gemma4",
                }
            }
        }
    )


def _request_with_settings(settings_values: Mapping[str, object]) -> Any:
    handoff = _service_module()
    request = handoff.OpenRouterPkceHandoffRequest(
        provider="openrouter",
        secret_key="openrouter_api_key",
        transient_api_key=RAW_PKCE_SECRET,
        settings_values=settings_values,
        expected_settings_revision="settings-r1",
        reason="openrouter_pkce_handoff",
        correlation_id="corr-pkce",
        verifier_context={
            "flow": "openrouter.pkce",
            "launch_source": "settings",
            "raw_context": RAW_PROVIDER_PAYLOAD,
        },
    )
    _assert_no_raw_secret_values(request, label="OpenRouterPkceHandoffRequest")
    return request


def _service(
    *,
    verifier: RecordingProviderVerifier,
    store: RecordingSecretStore,
    repository: RecordingSettingsRepository,
    runtime: RecordingRuntimeApply,
) -> Any:
    handoff = _service_module()
    secret_settings = importlib.import_module(
        "puripuly_heart.app.services.secret_settings_transaction"
    )
    return handoff.OpenRouterPkceHandoffService(
        provider_verifier=verifier,
        secret_transaction=secret_settings.SecretSettingsTransaction(
            secret_store=store,
            settings_repository=repository,
        ),
        runtime_apply=runtime,
    )


def _assert_no_raw_secret_values(value: object, *, label: str = "value") -> None:
    rendered = repr(value)
    for index, raw in enumerate(FORBIDDEN_RAW_SECRET_VALUES, start=1):
        if raw in rendered:
            pytest.fail(f"{label} repr exposed forbidden raw secret sentinel #{index}")


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


def _only_item(items: list[Any], *, label: str) -> Any:
    if len(items) != 1:
        pytest.fail(f"{label} count mismatch: actual={len(items)}, expected=1")
    return items[0]


def _assert_no_items(items: list[Any], *, label: str) -> None:
    if items:
        pytest.fail(f"{label} count mismatch: actual={len(items)}, expected=0")


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


def _assert_rejected_unsafe_settings_before_side_effects(
    *,
    result: messages.TransactionResult,
    events: list[tuple[str, str]],
    store: RecordingSecretStore,
    repository: RecordingSettingsRepository,
    runtime: RecordingRuntimeApply,
) -> None:
    _assert_no_raw_secret_values(result, label="unsafe settings rejection result")
    assert result.status == messages.TRANSACTION_STATUS_PROVIDER_VERIFICATION_FAILED
    assert events == []
    _assert_secret_value_matches(
        store.secrets.get("openrouter_api_key"),
        RAW_PREVIOUS_SECRET,
        label="unchanged openrouter secret after unsafe settings rejection",
    )
    _assert_no_items(repository.saved_requests, label="settings saves")
    _assert_no_items(runtime.requests, label="runtime apply requests")
    assert result.diagnostics is not None
    assert result.diagnostics.component == "openrouter_pkce_handoff"
    assert result.diagnostics.operation == "validate_settings_values"
    assert result.diagnostics.code == "unsafe_settings_values"
    assert result.diagnostics.content_policy == messages.CONTENT_POLICY_METADATA_ONLY
    _assert_no_raw_secret_values(result.diagnostics, label="unsafe settings diagnostics")


def test_request_repr_does_not_expose_raw_bad_settings_values() -> None:
    request = _request_with_settings(
        {
            "selection": "openrouter",
            "nested": {"value": ["safe", (RAW_PKCE_SECRET,)]},
            f"models.{RAW_PKCE_SECRET}.id": "safe-value",
        }
    )

    _assert_no_raw_secret_values(request, label="bad settings handoff request")


@pytest.mark.asyncio
async def test_raw_transient_key_in_settings_value_is_rejected_before_side_effects() -> None:
    events: list[tuple[str, str]] = []
    verifier = RecordingProviderVerifier(_verification_result(), events=events)
    store = RecordingSecretStore(
        {"openrouter_api_key": RAW_PREVIOUS_SECRET},
        events=events,
    )
    repository = RecordingSettingsRepository(_commit_success(), events=events)
    runtime = RecordingRuntimeApply(events=events)

    result = await _service(
        verifier=verifier,
        store=store,
        repository=repository,
        runtime=runtime,
    ).complete_handoff(
        _request_with_settings(
            {
                "selection": "openrouter",
                "routing": {"fallbacks": ["safe", (RAW_PKCE_SECRET,)]},
            }
        )
    )

    _assert_rejected_unsafe_settings_before_side_effects(
        result=result,
        events=events,
        store=store,
        repository=repository,
        runtime=runtime,
    )
    _assert_no_items(verifier.requests, label="provider verifier requests")


@pytest.mark.asyncio
async def test_raw_transient_key_in_settings_key_is_rejected_before_side_effects() -> None:
    events: list[tuple[str, str]] = []
    verifier = RecordingProviderVerifier(_verification_result(), events=events)
    store = RecordingSecretStore(
        {"openrouter_api_key": RAW_PREVIOUS_SECRET},
        events=events,
    )
    repository = RecordingSettingsRepository(_commit_success(), events=events)
    runtime = RecordingRuntimeApply(events=events)

    result = await _service(
        verifier=verifier,
        store=store,
        repository=repository,
        runtime=runtime,
    ).complete_handoff(
        _request_with_settings(
            {
                "selection": "openrouter",
                f"models.{RAW_PKCE_SECRET}.id": "safe-value",
            }
        )
    )

    _assert_rejected_unsafe_settings_before_side_effects(
        result=result,
        events=events,
        store=store,
        repository=repository,
        runtime=runtime,
    )
    _assert_no_items(verifier.requests, label="provider verifier requests")


@pytest.mark.asyncio
async def test_secret_bearing_settings_path_is_rejected_before_side_effects() -> None:
    events: list[tuple[str, str]] = []
    verifier = RecordingProviderVerifier(_verification_result(), events=events)
    store = RecordingSecretStore(
        {"openrouter_api_key": RAW_PREVIOUS_SECRET},
        events=events,
    )
    repository = RecordingSettingsRepository(_commit_success(), events=events)
    runtime = RecordingRuntimeApply(events=events)

    result = await _service(
        verifier=verifier,
        store=store,
        repository=repository,
        runtime=runtime,
    ).complete_handoff(
        _request_with_settings(
            {
                "selection": "openrouter",
                "credentials.api_key": "configured-by-service-boundary",
            }
        )
    )

    _assert_rejected_unsafe_settings_before_side_effects(
        result=result,
        events=events,
        store=store,
        repository=repository,
        runtime=runtime,
    )
    _assert_no_items(verifier.requests, label="provider verifier requests")


@pytest.mark.asyncio
async def test_success_verifies_saves_secret_and_settings_then_applies_runtime() -> None:
    events: list[tuple[str, str]] = []
    verifier = RecordingProviderVerifier(_verification_result(), events=events)
    store = RecordingSecretStore(
        {"openrouter_api_key": RAW_PREVIOUS_SECRET},
        events=events,
    )
    repository = RecordingSettingsRepository(_commit_success(), events=events)
    runtime = RecordingRuntimeApply(events=events)

    result = await _service(
        verifier=verifier,
        store=store,
        repository=repository,
        runtime=runtime,
    ).complete_handoff(_request())

    _assert_no_raw_secret_values(result, label="handoff success result")
    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED
    assert events == [
        ("verify", "openrouter"),
        ("snapshot", "openrouter_api_key"),
        ("set", "openrouter_api_key"),
        ("save", "openrouter_pkce_handoff"),
        ("runtime_apply", "openrouter_pkce_handoff"),
    ]
    verifier_request = _only_item(verifier.requests, label="provider verifier requests")
    _assert_no_raw_secret_values(verifier_request, label="provider verifier request")
    _assert_secret_value_matches(
        verifier_request.secret_value,
        RAW_PKCE_SECRET,
        label="provider verifier secret value",
    )
    assert set(verifier_request.context) == {"flow", "launch_source"}
    _assert_secret_value_matches(
        store.secrets.get("openrouter_api_key"),
        RAW_PKCE_SECRET,
        label="stored openrouter secret",
    )
    saved_request = _only_item(repository.saved_requests, label="settings saves")
    _assert_no_raw_secret_values(saved_request, label="handoff settings commit request")
    assert saved_request.expected_revision == "settings-r1"
    entry = _provider_verification_entry(
        saved_request.values, "openrouter", label="handoff settings commit request"
    )
    _assert_no_raw_secret_values(entry, label="handoff verification evidence entry")
    assert entry["status"] == "verified"
    assert entry["provider"] == "openrouter"
    assert entry["secret_key"] == "openrouter_api_key"
    assert entry["secret_revision"] == "secret-r1"
    assert str(entry["secret_fingerprint"]).startswith("sha256:")
    assert set(entry["verifier_context"]) == {"flow", "launch_source"}  # type: ignore[arg-type]
    evidence = entry["verifier_evidence"]
    if not isinstance(evidence, Mapping):
        pytest.fail("verification evidence details should be a mapping")
    assert evidence["verifier"] == "openrouter"
    assert evidence["latency_ms"] == 7.5
    assert "raw_provider_payload" not in evidence
    assert "api_key" not in evidence
    runtime_request = _only_item(runtime.requests, label="runtime apply requests")
    _assert_no_raw_secret_values(runtime_request, label="runtime apply request")
    assert runtime_request.settings_values["intent"]["translation"]["connection"] == "byok"  # type: ignore[index]
    assert (
        _provider_verification_entry(
            runtime_request.settings_values, "openrouter", label="runtime apply request"
        )
        == entry
    )


@pytest.mark.asyncio
async def test_success_preserves_existing_nested_operational_state_payloads() -> None:
    events: list[tuple[str, str]] = []
    verifier = RecordingProviderVerifier(_verification_result(), events=events)
    store = RecordingSecretStore(
        {"openrouter_api_key": RAW_PREVIOUS_SECRET},
        events=events,
    )
    repository = RecordingSettingsRepository(_commit_success(), events=events)
    runtime = RecordingRuntimeApply(events=events)
    existing_settings_values = {
        "intent": {"translation": {"connection": "byok"}},
        "state": {
            "managed_connection": {"enabled": True},
            "provider_verification": {
                "google": {
                    "status": "verified",
                    "provider": "google",
                }
            },
        },
    }

    result = await _service(
        verifier=verifier,
        store=store,
        repository=repository,
        runtime=runtime,
    ).complete_handoff(_request_with_settings(existing_settings_values))

    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED
    saved_request = _only_item(repository.saved_requests, label="settings saves")
    state = saved_request.values["state"]
    assert isinstance(state, Mapping)
    assert state["managed_connection"] == {"enabled": True}
    provider_verification = state["provider_verification"]
    assert isinstance(provider_verification, Mapping)
    assert provider_verification["google"] == {
        "status": "verified",
        "provider": "google",
    }
    assert _provider_verification_entry(saved_request.values, "openrouter")["status"] == "verified"
    runtime_request = _only_item(runtime.requests, label="runtime apply requests")
    assert runtime_request.settings_values == saved_request.values
    assert "openrouter" not in existing_settings_values["state"]["provider_verification"]


@pytest.mark.asyncio
async def test_success_replaces_existing_same_provider_verification_entry() -> None:
    events: list[tuple[str, str]] = []
    verifier = RecordingProviderVerifier(_verification_result(), events=events)
    store = RecordingSecretStore(
        {"openrouter_api_key": RAW_PREVIOUS_SECRET},
        events=events,
    )
    repository = RecordingSettingsRepository(_commit_success(), events=events)
    runtime = RecordingRuntimeApply(events=events)
    existing_settings_values = {
        "intent": {"translation": {"connection": "byok"}},
        "state": {
            "managed_connection": {"enabled": True},
            "provider_verification": {
                "openrouter": {
                    "status": "failed",
                    "provider": "openrouter",
                    "stale_top_level": "removed",
                    "verifier_context": {"stale_context": "removed"},
                    "verifier_evidence": {"stale_evidence": "removed"},
                },
                "google": {"status": "verified", "provider": "google"},
            },
        },
    }

    result = await _service(
        verifier=verifier,
        store=store,
        repository=repository,
        runtime=runtime,
    ).complete_handoff(_request_with_settings(existing_settings_values))

    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED
    saved_request = _only_item(repository.saved_requests, label="settings saves")
    state = saved_request.values["state"]
    assert isinstance(state, Mapping)
    assert state["managed_connection"] == {"enabled": True}
    provider_verification = state["provider_verification"]
    assert isinstance(provider_verification, Mapping)
    assert provider_verification["google"] == {"status": "verified", "provider": "google"}
    entry = _provider_verification_entry(saved_request.values, "openrouter")
    assert "stale_top_level" not in entry
    assert entry["status"] == "verified"
    assert set(entry["verifier_context"]) == {"flow", "launch_source"}  # type: ignore[arg-type]
    evidence = entry["verifier_evidence"]
    if not isinstance(evidence, Mapping):
        pytest.fail("verification evidence details should be a mapping")
    assert "stale_evidence" not in evidence
    runtime_request = _only_item(runtime.requests, label="runtime apply requests")
    assert runtime_request.settings_values == saved_request.values


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("verification_result", "raise_on_verify", "expected_code"),
    (
        (_verification_result(status="failed"), False, "provider_verification_failed"),
        (
            _verification_result(provider="unexpected", secret_key="openrouter_api_key"),
            False,
            "provider_verifier_result_mismatch",
        ),
        (None, True, "provider_verifier_exception"),
    ),
)
async def test_verification_failures_short_circuit_secret_settings_and_runtime(
    verification_result: provider_verifier.ProviderVerificationResult | None,
    raise_on_verify: bool,
    expected_code: str,
) -> None:
    events: list[tuple[str, str]] = []
    verifier = RecordingProviderVerifier(
        verification_result,
        events=events,
        raise_on_verify=raise_on_verify,
    )
    store = RecordingSecretStore(
        {"openrouter_api_key": RAW_PREVIOUS_SECRET},
        events=events,
    )
    repository = RecordingSettingsRepository(_commit_success(), events=events)
    runtime = RecordingRuntimeApply(events=events)

    result = await _service(
        verifier=verifier,
        store=store,
        repository=repository,
        runtime=runtime,
    ).complete_handoff(_request())

    _assert_no_raw_secret_values(result, label="verification failure result")
    assert result.status == messages.TRANSACTION_STATUS_PROVIDER_VERIFICATION_FAILED
    assert events == [("verify", "openrouter")]
    _assert_secret_value_matches(
        store.secrets.get("openrouter_api_key"),
        RAW_PREVIOUS_SECRET,
        label="unchanged openrouter secret",
    )
    _assert_no_items(repository.saved_requests, label="settings saves")
    _assert_no_items(runtime.requests, label="runtime apply requests")
    assert result.diagnostics is not None
    assert result.diagnostics.component == "openrouter_pkce_handoff"
    assert result.diagnostics.operation == "verify_transient_key"
    assert result.diagnostics.code == expected_code
    assert result.diagnostics.content_policy == messages.CONTENT_POLICY_METADATA_ONLY


@pytest.mark.asyncio
async def test_settings_commit_failure_delegates_secret_restore_and_skips_runtime() -> None:
    events: list[tuple[str, str]] = []
    verifier = RecordingProviderVerifier(_verification_result(), events=events)
    store = RecordingSecretStore(
        {"openrouter_api_key": RAW_PREVIOUS_SECRET},
        events=events,
    )
    repository = RecordingSettingsRepository(_commit_failure(), events=events)
    runtime = RecordingRuntimeApply(events=events)

    result = await _service(
        verifier=verifier,
        store=store,
        repository=repository,
        runtime=runtime,
    ).complete_handoff(_request())

    _assert_no_raw_secret_values(result, label="settings failure result")
    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED_SECRET_RESTORED
    assert events == [
        ("verify", "openrouter"),
        ("snapshot", "openrouter_api_key"),
        ("set", "openrouter_api_key"),
        ("save", "openrouter_pkce_handoff"),
        ("restore", "openrouter_api_key"),
    ]
    _assert_secret_value_matches(
        store.secrets.get("openrouter_api_key"),
        RAW_PREVIOUS_SECRET,
        label="restored openrouter secret",
    )
    _assert_no_items(runtime.requests, label="runtime apply requests")
    saved_request = _only_item(repository.saved_requests, label="settings saves")
    _assert_no_raw_secret_values(saved_request, label="failed handoff settings commit request")
    assert result.diagnostics is not None
    assert result.diagnostics.content_policy == messages.CONTENT_POLICY_METADATA_ONLY


@pytest.mark.asyncio
async def test_restore_failure_after_settings_commit_failure_reports_compensation_failure() -> None:
    events: list[tuple[str, str]] = []
    verifier = RecordingProviderVerifier(_verification_result(), events=events)
    store = RecordingSecretStore(
        {"openrouter_api_key": RAW_PREVIOUS_SECRET},
        events=events,
        fail_restore=True,
    )
    repository = RecordingSettingsRepository(_commit_failure(), events=events)
    runtime = RecordingRuntimeApply(events=events)

    result = await _service(
        verifier=verifier,
        store=store,
        repository=repository,
        runtime=runtime,
    ).complete_handoff(_request())

    _assert_no_raw_secret_values(result, label="restore failure result")
    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED_SECRET_RESTORE_FAILED
    assert events == [
        ("verify", "openrouter"),
        ("snapshot", "openrouter_api_key"),
        ("set", "openrouter_api_key"),
        ("save", "openrouter_pkce_handoff"),
        ("restore", "openrouter_api_key"),
    ]
    _assert_secret_value_matches(
        store.secrets.get("openrouter_api_key"),
        RAW_PKCE_SECRET,
        label="unrestored openrouter secret",
    )
    _assert_no_items(runtime.requests, label="runtime apply requests")
    assert result.diagnostics is not None
    assert result.diagnostics.content_policy == messages.CONTENT_POLICY_METADATA_ONLY


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "runtime_status",
    (messages.RUNTIME_APPLY_STATUS_DEGRADED, messages.RUNTIME_APPLY_STATUS_FAILED),
)
async def test_runtime_apply_non_applied_status_is_degraded_without_rollback(
    runtime_status: messages.RuntimeApplyStatus,
) -> None:
    events: list[tuple[str, str]] = []
    verifier = RecordingProviderVerifier(_verification_result(), events=events)
    store = RecordingSecretStore(
        {"openrouter_api_key": RAW_PREVIOUS_SECRET},
        events=events,
    )
    repository = RecordingSettingsRepository(_commit_success(), events=events)
    runtime = RecordingRuntimeApply(_runtime_result(runtime_status), events=events)

    result = await _service(
        verifier=verifier,
        store=store,
        repository=repository,
        runtime=runtime,
    ).complete_handoff(_request())

    _assert_no_raw_secret_values(result, label="runtime degraded result")
    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED
    assert events == [
        ("verify", "openrouter"),
        ("snapshot", "openrouter_api_key"),
        ("set", "openrouter_api_key"),
        ("save", "openrouter_pkce_handoff"),
        ("runtime_apply", "openrouter_pkce_handoff"),
    ]
    _assert_secret_value_matches(
        store.secrets.get("openrouter_api_key"),
        RAW_PKCE_SECRET,
        label="committed openrouter secret after runtime degradation",
    )
    _assert_no_items(store.restores, label="secret restores")
    assert result.diagnostics is not None
    assert result.diagnostics.component == "openrouter_pkce_handoff"
    assert result.diagnostics.operation == "runtime_apply"
    assert result.diagnostics.code == "runtime_apply_degraded"
    assert result.diagnostics.content_policy == messages.CONTENT_POLICY_METADATA_ONLY
    assert result.diagnostics.fields["runtime_status"] == runtime_status


@pytest.mark.asyncio
async def test_runtime_apply_exception_is_degraded_without_rollback_or_raw_exception() -> None:
    events: list[tuple[str, str]] = []
    verifier = RecordingProviderVerifier(_verification_result(), events=events)
    store = RecordingSecretStore(
        {"openrouter_api_key": RAW_PREVIOUS_SECRET},
        events=events,
    )
    repository = RecordingSettingsRepository(_commit_success(), events=events)
    runtime = RecordingRuntimeApply(events=events, raise_on_apply=True)

    result = await _service(
        verifier=verifier,
        store=store,
        repository=repository,
        runtime=runtime,
    ).complete_handoff(_request())

    _assert_no_raw_secret_values(result, label="runtime exception result")
    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_DEGRADED
    assert events == [
        ("verify", "openrouter"),
        ("snapshot", "openrouter_api_key"),
        ("set", "openrouter_api_key"),
        ("save", "openrouter_pkce_handoff"),
        ("runtime_apply", "openrouter_pkce_handoff"),
    ]
    _assert_secret_value_matches(
        store.secrets.get("openrouter_api_key"),
        RAW_PKCE_SECRET,
        label="committed openrouter secret after runtime exception",
    )
    _assert_no_items(store.restores, label="secret restores")
    assert result.diagnostics is not None
    assert result.diagnostics.component == "openrouter_pkce_handoff"
    assert result.diagnostics.operation == "runtime_apply"
    assert result.diagnostics.code == "runtime_apply_exception"
    assert result.diagnostics.content_policy == messages.CONTENT_POLICY_METADATA_ONLY
    assert "runtime failed after local commit" not in repr(result)
