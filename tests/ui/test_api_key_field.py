from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("flet")

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

    async def verify(provider: str, key: str) -> tuple[bool, str]:
        verified.append((provider, key))
        return True, "ok"

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

    async def verify(provider: str, key: str) -> tuple[bool, str]:
        verified.append((provider, key))
        return True, "ok"

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

    async def verify(provider: str, key: str) -> tuple[bool, str]:
        verified.append((provider, key))
        if key == "first-secret":
            first_started.set()
            await release_first.wait()
        return True, "ok"

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

    async def verify(provider: str, key: str) -> tuple[bool, str]:
        verified.append((provider, key))
        return True, "ok"

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

    async def verify(provider: str, key: str) -> tuple[bool, str]:
        verified.append((provider, key))
        return True, "ok"

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

    async def verify(_provider: str, _key: str) -> tuple[bool, str]:
        events.append("verify")
        return True, "ok"

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

    async def verify(provider: str, key: str) -> tuple[bool, str]:
        verified.append((provider, key))
        return True, "ok"

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

    async def verify(provider: str, key: str) -> tuple[bool, str]:
        verified.append((provider, key))
        current["value"] = "edited-secret"
        return False, "401 unauthorized"

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
