from __future__ import annotations

import importlib
import inspect
from collections.abc import Mapping
from dataclasses import FrozenInstanceError, fields, is_dataclass
from pathlib import Path
from typing import get_type_hints

import pytest

from puripuly_heart.core import messages
from tests.helpers.ast_sources import assert_no_forbidden_imports

FORBIDDEN_IMPORT_PREFIXES = (
    "flet",
    "keyring",
    "puripuly_heart.app.adapters",
    "puripuly_heart.config.settings",
    "puripuly_heart.core.managed_openrouter_broker_client",
    "puripuly_heart.core.storage",
    "puripuly_heart.providers",
    "puripuly_heart.ui",
)


def _assert_no_forbidden_imports(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert_no_forbidden_imports(Path(module.__file__ or ""), FORBIDDEN_IMPORT_PREFIXES)


def _assert_protocol_method_is_async(protocol: type[object], method_name: str) -> None:
    method = getattr(protocol, method_name)
    assert inspect.iscoroutinefunction(method)


def test_app_service_port_modules_are_import_safe_protocol_modules() -> None:
    modules = (
        "puripuly_heart.app.ports.settings_repository",
        "puripuly_heart.app.ports.secret_store",
        "puripuly_heart.app.ports.broker_client",
        "puripuly_heart.app.ports.discord_auth",
        "puripuly_heart.app.ports.managed_identity",
        "puripuly_heart.app.ports.provider_verifier",
        "puripuly_heart.app.ports.runtime_apply",
    )

    for module_name in modules:
        _assert_no_forbidden_imports(module_name)

    expectations = {
        "puripuly_heart.app.ports.settings_repository": "SettingsRepositoryPort",
        "puripuly_heart.app.ports.secret_store": "SecretStorePort",
        "puripuly_heart.app.ports.broker_client": "BrokerClientPort",
        "puripuly_heart.app.ports.discord_auth": "DiscordAuthPort",
        "puripuly_heart.app.ports.managed_identity": "ManagedIdentityPort",
        "puripuly_heart.app.ports.provider_verifier": "ProviderVerifierPort",
        "puripuly_heart.app.ports.runtime_apply": "RuntimeApplyPort",
    }
    for module_name, port_name in expectations.items():
        port = getattr(importlib.import_module(module_name), port_name)
        assert getattr(port, "_is_protocol", False), f"{port_name} must stay abstract"


def test_settings_repository_port_uses_neutral_snapshot_dtos() -> None:
    settings_repository = importlib.import_module("puripuly_heart.app.ports.settings_repository")

    snapshot = settings_repository.SettingsSnapshot(
        values={"provider": "openrouter", "enabled": True},
        revision="settings-r1",
    )
    request = settings_repository.SettingsCommitRequest(
        values=snapshot.values,
        expected_revision="settings-r1",
        reason="user_patch",
    )
    result = settings_repository.SettingsCommitResult(
        succeeded=True,
        snapshot=snapshot,
        message=None,
        diagnostics=None,
    )

    for dto in (snapshot, request, result):
        assert is_dataclass(dto)
        assert not hasattr(dto, "__dict__")

    with pytest.raises(FrozenInstanceError):
        snapshot.revision = "settings-r2"  # type: ignore[misc]
    with pytest.raises(TypeError):
        snapshot.values["provider"] = "qwen"  # type: ignore[index]

    hints = get_type_hints(settings_repository.SettingsSnapshot)
    assert hints["values"] == Mapping[str, object]
    assert result.snapshot is snapshot

    port = settings_repository.SettingsRepositoryPort
    _assert_protocol_method_is_async(port, "load")
    _assert_protocol_method_is_async(port, "save")


def test_settings_value_dtos_deep_freeze_nested_payloads() -> None:
    settings_repository = importlib.import_module("puripuly_heart.app.ports.settings_repository")
    runtime_apply = importlib.import_module("puripuly_heart.app.ports.runtime_apply")

    values = {
        "provider": {
            "aliases": ["openrouter"],
            "options": {"streaming": True},
        }
    }
    snapshot = settings_repository.SettingsSnapshot(values=values, revision="settings-r1")
    request = settings_repository.SettingsCommitRequest(
        values=values,
        expected_revision="settings-r1",
        reason="user_patch",
    )
    apply_request = runtime_apply.RuntimeApplyRequest(
        settings_values=values,
        reason="settings_commit",
        correlation_id="corr-1",
    )

    values["provider"]["aliases"].append("qwen")
    values["provider"]["options"]["streaming"] = False

    for frozen_values in (
        snapshot.values,
        request.values,
        apply_request.settings_values,
    ):
        provider = frozen_values["provider"]
        assert isinstance(provider, Mapping)
        assert provider["aliases"] == ("openrouter",)

        options = provider["options"]
        assert isinstance(options, Mapping)
        assert options["streaming"] is True

        with pytest.raises(TypeError):
            provider["options"] = {"streaming": False}  # type: ignore[index]
        with pytest.raises(AttributeError):
            provider["aliases"].append("deepseek")  # type: ignore[attr-defined]
        with pytest.raises(TypeError):
            options["streaming"] = False  # type: ignore[index]


def test_secret_broker_provider_and_runtime_ports_expose_service_result_seams() -> None:
    secret_store = importlib.import_module("puripuly_heart.app.ports.secret_store")
    broker_client = importlib.import_module("puripuly_heart.app.ports.broker_client")
    discord_auth = importlib.import_module("puripuly_heart.app.ports.discord_auth")
    managed_identity = importlib.import_module("puripuly_heart.app.ports.managed_identity")
    provider_verifier = importlib.import_module("puripuly_heart.app.ports.provider_verifier")
    runtime_apply = importlib.import_module("puripuly_heart.app.ports.runtime_apply")

    secret_result = secret_store.SecretWriteResult(
        succeeded=True,
        key="providers.openrouter.api_key",
        revision="secret-r1",
        message=None,
        diagnostics=None,
    )
    broker_result = broker_client.BrokerIssueResult(
        succeeded=True,
        broker_connection_id="conn-1",
        managed_secret_key="managed.openrouter.local_key",
        remote_key_revision="remote-r1",
        message=None,
        diagnostics=None,
    )
    qq_assertion_request = broker_client.QqManagedAssertionRequest(
        qq_identity="qq-user-1",
        credential="a" * 64,
        asserted_at="2026-07-03T06:00:00.000Z",
        metadata={"flow": "qq_managed"},
    )
    qq_entitlement = broker_client.QqManagedEntitlementSnapshot(
        qq_subject_ref="ph-qq-subject-v1_subject",
        managed_credential_ref="managed-ref-qq",
        expires_at="2026-08-03T06:00:00.000Z",
        openrouter_user_id="qq-user-openrouter",
    )
    qq_assertion_result = broker_client.QqManagedAssertionResult(
        succeeded=True,
        managed_secret_key="qq-managed-secret",
        entitlement=qq_entitlement,
        failure_subcode=None,
        retry_after_ms=None,
        message=None,
        diagnostics=None,
    )
    discord_request = discord_auth.DiscordAuthRequest(
        correlation_id="corr-1",
        metadata={"flow": "managed_connection"},
    )
    discord_result = discord_auth.DiscordAuthResult(
        succeeded=True,
        discord_user_id="discord-user-1",
        message=None,
        diagnostics=None,
    )
    identity_request = managed_identity.ManagedIdentityPreflightRequest(
        local_secret_key="openrouter_managed_api_key",
        correlation_id="corr-1",
        metadata={"flow": "managed_connection"},
    )
    identity_result = managed_identity.ManagedIdentityPreflightResult(
        succeeded=True,
        local_public_key="local-public-key-1",
        local_identity_revision="identity-r1",
        message=None,
        diagnostics=None,
    )
    verification_request = provider_verifier.ProviderVerificationRequest(
        provider="openrouter",
        secret_key="providers.openrouter.api_key",
        secret_value="not-logged",
        secret_revision="secret-r1",
        context={"verifier_context": "settings_commit"},
    )
    verification_result = provider_verifier.ProviderVerificationResult(
        status="verified",
        provider="openrouter",
        secret_key="providers.openrouter.api_key",
        secret_revision="secret-r1",
        evidence={"verifier": "openrouter", "latency_ms": 12.5},
        message=None,
        diagnostics=None,
    )
    apply_request = runtime_apply.RuntimeApplyRequest(
        settings_values={"provider": "openrouter"},
        reason="settings_commit",
        correlation_id="corr-1",
    )

    for dto in (
        secret_result,
        broker_result,
        qq_assertion_request,
        qq_entitlement,
        qq_assertion_result,
        discord_request,
        discord_result,
        identity_request,
        identity_result,
        verification_request,
        verification_result,
        apply_request,
    ):
        assert is_dataclass(dto)
        assert not hasattr(dto, "__dict__")

    with pytest.raises(TypeError):
        verification_request.context["verifier_context"] = "other"  # type: ignore[index]
    with pytest.raises(TypeError):
        verification_result.evidence["verifier"] = "other"  # type: ignore[index]
    with pytest.raises(TypeError):
        apply_request.settings_values["provider"] = "qwen"  # type: ignore[index]
    with pytest.raises(TypeError):
        discord_request.metadata["flow"] = "other"  # type: ignore[index]
    with pytest.raises(TypeError):
        identity_request.metadata["flow"] = "other"  # type: ignore[index]
    with pytest.raises(TypeError):
        qq_assertion_request.metadata["flow"] = "other"  # type: ignore[index]

    assert "qq-user-1" not in repr(qq_assertion_request)
    assert "credential" not in repr(qq_assertion_request)
    assert "qq-managed-secret" not in repr(qq_assertion_result)

    request_hints = get_type_hints(provider_verifier.ProviderVerificationRequest)
    assert request_hints["secret_value"] is str
    assert "not-logged" not in repr(verification_request)
    assert "secret_value" not in {
        field.name for field in fields(provider_verifier.ProviderVerificationResult)
    }

    verifier_hints = get_type_hints(provider_verifier.ProviderVerificationResult)
    assert verifier_hints["evidence"] == Mapping[str, messages.DiagnosticFieldValue]

    runtime_hints = get_type_hints(runtime_apply.RuntimeApplyPort.apply_runtime)
    assert runtime_hints["return"] == messages.RuntimeApplyResult

    for port, methods in {
        secret_store.SecretStorePort: (
            "get_secret",
            "set_secret",
            "clear_secret",
            "snapshot_secret",
            "restore_secret",
        ),
        broker_client.BrokerClientPort: (
            "issue_managed_connection",
            "assert_qq_managed_identity",
        ),
        discord_auth.DiscordAuthPort: ("start_discord_auth",),
        managed_identity.ManagedIdentityPort: ("preflight_managed_identity",),
        provider_verifier.ProviderVerifierPort: ("verify_provider_secret",),
        runtime_apply.RuntimeApplyPort: ("apply_runtime",),
    }.items():
        for method_name in methods:
            _assert_protocol_method_is_async(port, method_name)
