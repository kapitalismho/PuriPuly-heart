from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("flet")

from puripuly_heart.app.ports.provider_verifier import ProviderCredentialVerificationOutcome
from puripuly_heart.ui.components.settings.api_key_field import ApiKeyField
from puripuly_heart.ui.components.settings.api_key_verification_controller import (
    ApiKeyVerificationController,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "code", "param", "category", "message_key"),
    [
        (400, "unsupported_value", "max_completion_tokens", "invalid_response", "request_rejected"),
        (400, "model_not_found", "model", "invalid_response", "model_unavailable"),
        (401, "invalid_api_key", None, "auth", "authentication_rejected"),
        (401, "model_not_found", "model", "auth", "authentication_rejected"),
        (403, "permission_denied", "model", "auth", "permission_denied"),
        (404, "model_not_found", "model", "invalid_response", "model_unavailable"),
        (429, "insufficient_quota", None, "quota", "quota"),
        (429, "rate_limit_exceeded", None, "rate_limit", "rate_limit"),
        (503, "server_error", None, "service_unavailable", "service_unavailable"),
    ],
)
async def test_openai_verification_displays_typed_failure_and_retains_only_safe_details(
    monkeypatch, caplog, status, code, param, category, message_key
) -> None:
    import httpx

    from puripuly_heart.app.adapters.provider_verifier import ProviderVerifierAdapter
    from puripuly_heart.app.services.provider.provider_credential_verification import (
        ProviderCredentialVerificationInteractionOwner,
        ProviderCredentialVerificationOwner,
    )
    from puripuly_heart.core import network_clients
    from puripuly_heart.core.diagnostic_validation import validate_diagnostics_for_sink
    from puripuly_heart.core.diagnostics import DIAGNOSTIC_RECORD_ATTRIBUTE, normalize_diagnostic
    from puripuly_heart.ui.i18n import t

    private = "sk-synthetic-private-key-not-for-output"
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            status,
            json={
                "error": {
                    "code": code,
                    "type": "invalid_request_error",
                    "param": param,
                    "message": f"Private upstream message containing {private}",
                }
            },
        )
    )
    monkeypatch.setattr(
        network_clients,
        "external_async_client",
        lambda **kwargs: httpx.AsyncClient(transport=transport, **kwargs),
    )
    interaction = ProviderCredentialVerificationInteractionOwner(
        verification_owner=ProviderCredentialVerificationOwner(verifier=ProviderVerifierAdapter()),
        selected_model_provider=lambda _provider: "gpt-6-luna",
    )
    shown = []
    field = ApiKeyField(
        "settings.openai_api_key",
        "openai_api_key",
        "openai",
        on_verify=interaction.verify,
        show_snackbar=lambda message, _color: shown.append(message),
    )
    field.value = private
    field._handle_blur(None)
    await field._run_verification()

    assert field.controller.status == "error"
    assert field.controller.last_verified_hash == ""
    assert shown == [t("snackbar.verification_failed", message=t(f"error.openai.{message_key}"))]
    failures = [
        getattr(record, DIAGNOSTIC_RECORD_ATTRIBUTE)[0]
        for record in caplog.records
        if hasattr(record, DIAGNOSTIC_RECORD_ATTRIBUTE)
    ]
    assert len(failures) == 1
    failure = failures[0]
    assert failure.status_code == status
    assert failure.category == category
    assert failure.operation == "key_verification"
    assert failure.fields["openai_error_code"] == code
    assert failure.fields["openai_error_param"] == (param or "unknown")
    assert failure.fields["openai_error_type"] == "invalid_request_error"
    assert validate_diagnostics_for_sink(failure, "persisted_logs").status == "accepted"
    retained = normalize_diagnostic(
        failure, instance_id=None, severity="error", disposition="failed"
    )
    assert retained["fields"]["openai_error_code"] == code
    assert retained["fields"]["openai_error_param"] == (param or "unknown")
    assert retained["detail"]["rejected_fields"] == 0
    assert private not in repr(retained) + repr(shown) + caplog.text
    assert "Private upstream message" not in repr(retained) + repr(shown) + caplog.text


@pytest.mark.asyncio
async def test_openai_verification_can_recover_after_a_rejected_request(monkeypatch) -> None:
    import httpx

    from puripuly_heart.app.adapters.provider_verifier import ProviderVerifierAdapter
    from puripuly_heart.app.services.provider.provider_credential_verification import (
        ProviderCredentialVerificationInteractionOwner,
        ProviderCredentialVerificationOwner,
    )
    from puripuly_heart.core import network_clients

    responses = iter(
        [
            httpx.Response(400, json={"error": {"code": "invalid_value", "param": "temperature"}}),
            httpx.Response(200, json={"choices": [{"finish_reason": "length"}]}),
        ]
    )
    monkeypatch.setattr(
        network_clients,
        "external_async_client",
        lambda **kwargs: httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _request: next(responses)), **kwargs
        ),
    )
    interaction = ProviderCredentialVerificationInteractionOwner(
        verification_owner=ProviderCredentialVerificationOwner(verifier=ProviderVerifierAdapter()),
        selected_model_provider=lambda _provider: "gpt-6-luna",
    )
    field = ApiKeyField(
        "settings.openai_api_key", "openai_api_key", "openai", on_verify=interaction.verify
    )
    field.value = "synthetic-key"

    field._handle_blur(None)
    await field._run_verification()
    assert field.controller.status == "error"
    assert field.controller.last_verified_hash == ""

    field._handle_blur(None)
    await field._run_verification()
    assert field.controller.status == "success"
    assert field.controller.last_verified_hash == ApiKeyVerificationController.key_hash(
        "synthetic-key"
    )


@pytest.mark.asyncio
async def test_api_key_field_default_status_mode_verifies_saved_value() -> None:
    saved: list[tuple[str, str]] = []
    verified: list[tuple[str, str]] = []

    async def verify(provider: str, key: str) -> ProviderCredentialVerificationOutcome:
        verified.append((provider, key))
        return ProviderCredentialVerificationOutcome(
            status="verified", provider=provider, error_text="ok"
        )

    field = ApiKeyField(
        "settings.api_keys.openrouter",
        "openrouter_api_key",
        "openrouter",
        on_verify=verify,
        on_save=lambda key, value: saved.append((key, value)),
    )

    field._text_field.value = "provider-secret"
    field._handle_change(None)
    field._handle_blur(None)
    await field._run_verification()

    assert len(field.controls) == 2
    assert saved == [("openrouter_api_key", "provider-secret")]
    assert verified == [("openrouter", "provider-secret")]
    assert field._current_status == "success"
    assert field.controller.last_verified_hash == ApiKeyVerificationController.key_hash(
        "provider-secret"
    )


@pytest.mark.asyncio
async def test_api_key_field_default_status_mode_verifies_unchanged_loaded_value() -> None:
    saved: list[tuple[str, str]] = []
    verified: list[tuple[str, str]] = []

    async def verify(provider: str, key: str) -> ProviderCredentialVerificationOutcome:
        verified.append((provider, key))
        return ProviderCredentialVerificationOutcome(
            status="verified", provider=provider, error_text="ok"
        )

    field = ApiKeyField(
        "settings.api_keys.openrouter",
        "openrouter_api_key",
        "openrouter",
        on_verify=verify,
        on_save=lambda key, value: saved.append((key, value)),
    )

    field.value = "loaded-secret"
    field._handle_blur(None)
    await field._run_verification()

    assert saved == []
    assert verified == [("openrouter", "loaded-secret")]
    assert field._current_status == "success"
    assert field.controller.last_verified_hash == ApiKeyVerificationController.key_hash(
        "loaded-secret"
    )


@pytest.mark.asyncio
async def test_api_key_field_verifies_latest_edit_after_blur_during_inflight_verification() -> None:
    saved: list[tuple[str, str]] = []
    verified: list[tuple[str, str]] = []
    first_started = asyncio.Event()
    release_first = asyncio.Event()

    async def verify(provider: str, key: str) -> ProviderCredentialVerificationOutcome:
        verified.append((provider, key))
        if key == "first-secret":
            first_started.set()
            await release_first.wait()
        return ProviderCredentialVerificationOutcome(
            status="verified", provider=provider, error_text="ok"
        )

    field = ApiKeyField(
        "settings.api_keys.openrouter",
        "openrouter_api_key",
        "openrouter",
        on_verify=verify,
        on_save=lambda key, value: saved.append((key, value)),
    )

    field._text_field.value = "first-secret"
    field._handle_change(None)
    field._handle_blur(None)
    verification_task = asyncio.create_task(field._run_verification())
    await first_started.wait()

    field._text_field.value = "second-secret"
    field._handle_change(None)
    field._handle_blur(None)

    release_first.set()
    await verification_task

    assert saved == [
        ("openrouter_api_key", "first-secret"),
        ("openrouter_api_key", "second-secret"),
    ]
    assert verified == [
        ("openrouter", "first-secret"),
        ("openrouter", "second-secret"),
    ]
    assert field._current_status == "success"
    assert field.controller.last_verified_hash == ApiKeyVerificationController.key_hash(
        "second-secret"
    )


@pytest.mark.asyncio
async def test_api_key_field_can_hide_status_and_skip_verification() -> None:
    saved: list[tuple[str, str]] = []
    verified: list[tuple[str, str]] = []

    async def verify(provider: str, key: str) -> ProviderCredentialVerificationOutcome:
        verified.append((provider, key))
        return ProviderCredentialVerificationOutcome(
            status="verified", provider=provider, error_text="ok"
        )

    field = ApiKeyField(
        "settings.local_llm.api_key",
        "local_llm_api_key",
        "local_llm",
        on_verify=verify,
        on_save=lambda key, value: saved.append((key, value)),
        show_status=False,
    )

    field._text_field.value = "local-secret"
    field._handle_change(None)
    field._handle_blur(None)
    await field._run_verification()

    assert len(field.controls) == 1
    assert saved == [("local_llm_api_key", "local-secret")]
    assert verified == []
    assert not field.controller.has_pending


@pytest.mark.asyncio
async def test_api_key_field_does_not_save_unchanged_loaded_value() -> None:
    saved: list[tuple[str, str]] = []
    field = ApiKeyField(
        "settings.local_llm.api_key",
        "local_llm_api_key",
        "local_llm",
        on_save=lambda key, value: saved.append((key, value)),
        show_status=False,
    )

    field.value = "loaded-secret"
    field._handle_blur(None)
    await field._run_verification()

    assert saved == []


@pytest.mark.asyncio
async def test_api_key_field_does_not_verify_when_secret_save_fails() -> None:
    verified: list[tuple[str, str]] = []

    async def verify(provider: str, key: str) -> ProviderCredentialVerificationOutcome:
        verified.append((provider, key))
        return ProviderCredentialVerificationOutcome(
            status="verified", provider=provider, error_text="ok"
        )

    field = ApiKeyField(
        "settings.api_keys.openrouter",
        "openrouter_api_key",
        "openrouter",
        on_verify=verify,
        on_save=lambda _key, _value: False,
    )
    field._text_field.value = "unsaved-secret"
    field._handle_change(None)

    field._handle_blur(None)
    await field._run_verification()

    assert verified == []
    assert field._current_status == "error"
    assert field.controller.last_verified_hash == ""
    assert not field.controller.has_pending


@pytest.mark.asyncio
async def test_api_key_field_awaits_secret_transaction_before_verification() -> None:
    events: list[str] = []
    release_save = asyncio.Event()

    async def save(_key: str, _value: str) -> bool:
        events.append("save-start")
        await release_save.wait()
        events.append("save-complete")
        return True

    async def verify(_provider: str, _key: str) -> ProviderCredentialVerificationOutcome:
        events.append("verify")
        return ProviderCredentialVerificationOutcome(
            status="verified", provider=_provider, error_text="ok"
        )

    field = ApiKeyField(
        "settings.api_keys.openrouter",
        "openrouter_api_key",
        "openrouter",
        on_verify=verify,
        on_save=save,
    )
    field._text_field.value = "new-secret"
    field._handle_change(None)
    field._handle_blur(None)
    task = asyncio.create_task(field._run_verification())
    await asyncio.sleep(0)

    assert events == ["save-start"]
    release_save.set()
    await task

    assert events == ["save-start", "save-complete", "verify"]


@pytest.mark.asyncio
async def test_controller_verifies_and_reports_success() -> None:
    saved: list[tuple[str, str]] = []
    verified: list[tuple[str, str]] = []
    statuses: list[str] = []
    messages: list[tuple[str, str]] = []
    current = {"value": "provider-secret"}

    async def verify(provider: str, key: str) -> ProviderCredentialVerificationOutcome:
        verified.append((provider, key))
        return ProviderCredentialVerificationOutcome(
            status="verified", provider=provider, error_text="ok"
        )

    controller = ApiKeyVerificationController(
        secret_key="openrouter_api_key",
        provider="openrouter",
        on_verify=verify,
        on_save=lambda key, value: saved.append((key, value)),
        on_status=statuses.append,
        on_message=lambda key, msg: messages.append((key, msg)),
    )
    controller.set_value_getter(lambda: current["value"])

    controller.notify_edit()
    controller.handle_blur("provider-secret")
    drain = controller.run_pending()
    assert drain is not None
    await drain

    assert saved == [("openrouter_api_key", "provider-secret")]
    assert verified == [("openrouter", "provider-secret")]
    assert controller.status == "success"
    assert controller.last_verified_hash == controller.key_hash("provider-secret")
    assert statuses[-1] == "success"
    assert messages[0][0] == "snackbar.verification_ok"


@pytest.mark.asyncio
async def test_controller_ignores_stale_result_when_value_changed_midflight() -> None:
    verified: list[tuple[str, str]] = []
    statuses: list[str] = []
    current = {"value": "stale-secret"}

    async def verify(provider: str, key: str) -> ProviderCredentialVerificationOutcome:
        verified.append((provider, key))
        current["value"] = "edited-secret"
        return ProviderCredentialVerificationOutcome(
            status="failed", provider=provider, error_text="401 unauthorized"
        )

    controller = ApiKeyVerificationController(
        secret_key="openrouter_api_key",
        provider="openrouter",
        on_verify=verify,
        on_status=statuses.append,
    )
    controller.set_value_getter(lambda: current["value"])

    controller.handle_blur("stale-secret")
    drain = controller.run_pending()
    assert drain is not None
    await drain

    assert verified == [("openrouter", "stale-secret")]
    assert controller.status == "verifying"
    assert controller.last_verified_hash == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("success", [True, False])
async def test_completion_reports_actual_verification_not_restored_status(success: bool) -> None:
    completed = []
    secret = "synthetic-private-value"

    async def verify(_provider: str, _key: str) -> ProviderCredentialVerificationOutcome:
        return ProviderCredentialVerificationOutcome(
            status="verified" if success else "failed",
            provider=_provider,
            error_text=f"401 unauthorized {secret}",
        )

    field = ApiKeyField(
        "settings.openrouter_api_key",
        "openrouter_api_key",
        "openrouter",
        on_verify=verify,
        on_complete=lambda *args: completed.append(args),
    )
    field.value = secret
    field._set_status("success")
    field._last_verified_hash = field._get_key_hash(secret)
    field._handle_blur(None)
    await field._run_verification()
    assert completed == []

    field._last_verified_hash = ""
    field._handle_blur(None)
    await field._run_verification()

    assert completed == [("openrouter", success, None if success else "verification_failed", None)]
    assert secret not in repr(completed)
    assert field._get_key_hash(secret) not in repr(completed)


@pytest.mark.asyncio
@pytest.mark.parametrize("raises", [True, False])
async def test_completion_ignores_stale_verification(raises: bool) -> None:
    completed = []

    async def verify(_provider: str, _key: str) -> ProviderCredentialVerificationOutcome:
        field.value = "replacement"
        if raises:
            raise RuntimeError("401 unauthorized")
        return ProviderCredentialVerificationOutcome(
            status="verified", provider=_provider, error_text="ok"
        )

    field = ApiKeyField(
        "settings.openrouter_api_key",
        "openrouter_api_key",
        "openrouter",
        on_verify=verify,
        on_complete=lambda *args: completed.append(args),
    )
    field.value = "old"
    field._handle_blur(None)
    await field._run_verification()

    assert completed == []
    assert field.controller.last_verified_hash == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("stale", [True, False])
@pytest.mark.parametrize("raises", [True, False])
async def test_completion_rejects_stale_async_save_failure_and_allows_retry(
    stale: bool, raises: bool
) -> None:
    completed = []
    verified = []
    started = asyncio.Event()
    release = asyncio.Event()
    saves = []

    async def save(_name: str, value: str) -> bool:
        saves.append(value)
        if len(saves) > 1:
            return True
        started.set()
        await release.wait()
        if raises:
            raise OSError("synthetic-private-value")
        return False

    async def verify(provider: str, _key: str) -> ProviderCredentialVerificationOutcome:
        verified.append(provider)
        return ProviderCredentialVerificationOutcome(
            status="verified", provider=provider, error_text="ok"
        )

    field = ApiKeyField(
        "settings.openrouter_api_key",
        "openrouter_api_key",
        "openrouter",
        on_save=save,
        on_verify=verify,
        on_complete=lambda *args: completed.append(args),
    )
    field._text_field.value = "synthetic-private-value"
    field._handle_change(None)
    field._handle_blur(None)
    task = asyncio.create_task(field._run_verification())
    await started.wait()
    if stale:
        field.value = "replacement"
        field._set_status("success")
        field._last_verified_hash = field._get_key_hash(field.value)
    release.set()
    await task

    assert verified == []
    if stale:
        assert completed == []
        assert field.controller.status == "success"
        assert field.controller.last_verified_hash == field._get_key_hash("replacement")
    else:
        assert [item[:3] for item in completed] == [("openrouter", False, "verification_failed")]
        assert completed[0][3].code == "verification_save_failed"
        assert completed[0][3].fields["exception_class"] == (
            "OSError" if raises else "RuntimeError"
        )
        assert field.controller.status == "error"
        field._handle_blur(None)
        await field._run_verification()
        assert len(saves) == 2
        assert completed[-1] == ("openrouter", True, None, None)
        assert verified == ["openrouter"]
    assert "synthetic-private-value" not in repr(completed)


@pytest.mark.asyncio
async def test_completion_forwards_typed_error_but_not_exception_prose() -> None:
    from puripuly_heart.core.openrouter.authentication import OpenRouterAuthenticationError

    completed = []
    failures = iter(
        [
            OpenRouterAuthenticationError(stage="key_verification", reason="credential_rejected"),
            RuntimeError("401 unauthorized"),
        ]
    )

    async def verify(_provider: str, _key: str) -> ProviderCredentialVerificationOutcome:
        raise next(failures)

    field = ApiKeyField(
        "settings.openrouter_api_key",
        "openrouter_api_key",
        "openrouter",
        on_verify=verify,
        on_complete=lambda *args: completed.append(args),
    )
    field.value = "synthetic"
    for _ in range(2):
        field._handle_blur(None)
        await field._run_verification()

    assert [item[:3] for item in completed] == [
        ("openrouter", False, "openrouter.auth.credential_rejected"),
        ("openrouter", False, "verification_failed"),
    ]
    assert [item[3].code for item in completed] == [
        "openrouter.auth.credential_rejected",
        "openrouter.auth.unknown",
    ]
    assert "401 unauthorized" not in repr(completed)


@pytest.mark.asyncio
async def test_completion_unbind_discards_inflight_result() -> None:
    completed = []

    async def verify(_provider: str, _key: str) -> ProviderCredentialVerificationOutcome:
        field.bind_verification_completion(None)
        return ProviderCredentialVerificationOutcome(
            status="verified", provider=_provider, error_text="ok"
        )

    field = ApiKeyField(
        "settings.openrouter_api_key",
        "openrouter_api_key",
        "openrouter",
        on_verify=verify,
        on_complete=lambda *args: completed.append(args),
    )
    field.value = "synthetic"
    field._handle_blur(None)
    await field._run_verification()

    assert field.controller.status == "success"
    assert completed == []


@pytest.mark.asyncio
async def test_real_verification_preserves_typed_completion_recovery_and_generic_fallback(
    monkeypatch, caplog
) -> None:
    import socket

    import httpx

    from puripuly_heart.app.adapters.provider_verifier import ProviderVerifierAdapter
    from puripuly_heart.app.adapters.ui_runtime import UiProviderRuntimeAdapter
    from puripuly_heart.app.services.provider.provider_credential_verification import (
        ProviderCredentialVerificationInteractionOwner,
        ProviderCredentialVerificationOwner,
    )
    from puripuly_heart.core import network_clients

    secret = "sk-synthetic-private-completion-key"
    body = f"Private upstream prose containing {secret}"
    statuses = iter([429, 200, 401])
    requests = []
    sockets = []

    def reject_socket(*args):
        sockets.append(True)
        raise AssertionError("unexpected socket connection")

    def respond(request):
        status = next(statuses)
        requests.append((request.method, request.url.path, status))
        return httpx.Response(status, text=body)

    monkeypatch.setattr(socket.socket, "connect", reject_socket)
    monkeypatch.setattr(
        network_clients,
        "external_async_client",
        lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(respond), **kwargs),
    )
    interaction = ProviderCredentialVerificationInteractionOwner(
        verification_owner=ProviderCredentialVerificationOwner(verifier=ProviderVerifierAdapter()),
        selected_model_provider=lambda _provider: None,
    )
    adapter = UiProviderRuntimeAdapter(
        settings=None,
        provider_application=None,
        gpu=None,
        managed=None,
        credential_verification=interaction,
        provider_settings=None,
        build_byok_target_settings=lambda _settings: None,
    )
    completed = []
    messages = []
    current = {"value": secret}
    controller = ApiKeyVerificationController(
        secret_key="openrouter_api_key",
        provider="openrouter",
        on_verify=adapter.verify_api_key,
        on_complete=lambda *args: completed.append(args),
        on_message=lambda *args: messages.append(args),
    )
    controller.set_value_getter(lambda: current["value"])

    controller.handle_blur(current["value"])
    await controller.run_pending()
    assert controller.status == "error"
    assert [item[:3] for item in completed] == [("openrouter", False, "openrouter.auth.rate_limit")]
    assert completed[0][3].status_code == 429

    controller.handle_blur(current["value"])
    await controller.run_pending()
    assert controller.status == "success"
    assert controller.last_verified_hash == controller.key_hash(secret)
    assert completed[-1] == ("openrouter", True, None, None)

    current["value"] = "synthetic-rejected-value"
    controller.handle_blur(current["value"])
    await controller.run_pending()
    assert controller.status == "error"
    assert controller.last_verified_hash == ""
    assert completed[-1][:3] == ("openrouter", False, "openrouter.auth.credential_rejected")
    assert completed[-1][3].status_code == 401
    assert messages == [
        ("snackbar.verification_failed", "error.openrouter_auth.rate_limit"),
        ("snackbar.verification_ok", ""),
        ("snackbar.verification_failed", "error.openrouter_auth.credential_rejected"),
    ]
    assert requests == [
        ("GET", "/api/v1/key", 429),
        ("GET", "/api/v1/key", 200),
        ("GET", "/api/v1/key", 401),
    ]
    assert sockets == []
    retained = repr(completed) + repr(messages) + repr(requests) + caplog.text
    assert secret not in retained
    assert controller.key_hash(secret) not in retained
    assert "Private upstream prose" not in retained
