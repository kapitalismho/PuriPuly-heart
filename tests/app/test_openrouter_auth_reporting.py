from __future__ import annotations

import asyncio
import socket
import ssl
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import httpx
import pytest
from puripuly_heart.app.services.provider_credential_verification import (
    ProviderCredentialVerificationInteractionOwner,
    ProviderCredentialVerificationOwner,
)
from puripuly_heart.app.services.settings_transaction_result import SettingsTransactionResultOwner
from puripuly_heart.core.openrouter_pkce import OpenRouterPKCEClient, OpenRouterPKCEExchangeResult

from puripuly_heart.app.adapters.provider_verifier import ProviderVerifierAdapter
from puripuly_heart.app.adapters.settings_vnext_canonical_persistence import (
    SettingsVNextCanonicalPersistenceAdapter,
)
from puripuly_heart.app.ports.provider_verifier import ProviderVerificationRequest
from puripuly_heart.app.ports.settings_view import OpenRouterPkceTarget, ProviderApplyIntent
from puripuly_heart.app.services.canonical_settings_persistence import SettingsOwner
from puripuly_heart.app.services.openrouter_pkce_flow import (
    OpenRouterPkceApplicationOwner,
    OpenRouterPkceFlowOwner,
)
from puripuly_heart.config.provider_values import OpenRouterSelectionAlias
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.core.diagnostic_validation import validate_diagnostics_for_sink
from puripuly_heart.core.error_messages import (
    format_error_report_for_log,
    openrouter_auth_failure_report,
)
from puripuly_heart.core.openrouter.authentication import OpenRouterAuthenticationError
from puripuly_heart.providers.llm.openrouter import OpenRouterLLMProvider
from puripuly_heart.ui.components.settings.api_key_verification_controller import (
    ApiKeyVerificationController,
)
from puripuly_heart.ui.i18n import get_locale, set_locale, t

SECRET = "arbitrary-user-input-without-a-token-pattern"
UNSAFE = f"https://user:proxy-password@host/path?code={SECRET} body={SECRET}"


def install_response(monkeypatch: pytest.MonkeyPatch, response: httpx.Response) -> None:
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: real_client(
            **kwargs, transport=httpx.MockTransport(lambda _request: response), trust_env=False
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("status", "expected"), [(200, True), (401, False)])
async def test_manual_verification_accepts_success_and_only_explicit_rejection(
    monkeypatch: pytest.MonkeyPatch, status: int, expected: bool
) -> None:
    install_response(monkeypatch, httpx.Response(status, text=UNSAFE))
    assert await OpenRouterLLMProvider.verify_api_key(SECRET) is expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "reason"),
    [(403, "access_denied"), (429, "rate_limit"), (500, "service_unavailable"),
     (503, "service_unavailable"), (504, "timeout"), (302, "http"), (400, "http")],
)
async def test_manual_noncredential_http_failures_reach_ui_and_safe_metadata(
    monkeypatch: pytest.MonkeyPatch, status: int, reason: str
) -> None:
    install_response(monkeypatch, httpx.Response(status, text=UNSAFE))
    diagnostics: list[dict[str, object]] = []
    errors: list[object] = []
    owner = ProviderCredentialVerificationInteractionOwner(
        verification_owner=ProviderCredentialVerificationOwner(
            verifier=ProviderVerifierAdapter(),
            diagnostics_sink=lambda _event, metadata, exc: (
                diagnostics.append(dict(metadata)), errors.append(exc)
            ),
        ),
        selected_model_provider=lambda _provider: None,
    )
    messages: list[tuple[str, str]] = []
    controller = ApiKeyVerificationController(
        secret_key="openrouter_api_key", provider="openrouter", on_verify=owner.verify,
        on_message=lambda key, message: messages.append((key, message)),
    )
    controller.set_value_getter(lambda: SECRET)
    await controller.verify_direct(SECRET)
    assert controller.status == "error"
    assert controller.last_verified_hash == ""
    assert messages == [("snackbar.verification_failed", f"error.openrouter_auth.{reason}")]
    assert errors == [None]
    assert f"status={status}" in str(diagnostics)
    assert "operation=key_verification" in str(diagnostics)
    assert SECRET not in str(diagnostics) + str(messages)
    assert "proxy-password" not in str(diagnostics) + str(messages)


def tls_error() -> httpx.ConnectError:
    error = httpx.ConnectError(UNSAFE)
    error.__cause__ = ssl.SSLCertVerificationError(UNSAFE)
    return error


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("exception", "reason", "exception_type"),
    [(tls_error(), "tls", "SSLCertVerificationError"),
     (httpx.ProxyError(UNSAFE), "proxy", "ProxyError"),
     (httpx.ConnectError(UNSAFE), "network", "ConnectError"),
     (httpx.ReadTimeout(UNSAFE), "timeout", "ReadTimeout"),
     (ImportError(UNSAFE), "setup", "ImportError"),
     (FileNotFoundError(UNSAFE), "setup", "FileNotFoundError"),
     (PermissionError(UNSAFE), "setup", "PermissionError"),
     (ValueError(UNSAFE), "setup", "ValueError")],
)
async def test_transport_and_client_setup_failures_are_not_credential_rejection(
    monkeypatch: pytest.MonkeyPatch, exception: Exception, reason: str, exception_type: str
) -> None:
    def fail(**_kwargs: object) -> None:
        raise exception

    monkeypatch.setattr(httpx, "AsyncClient", fail)
    with pytest.raises(OpenRouterAuthenticationError) as caught:
        await OpenRouterLLMProvider.verify_api_key(SECRET)
    failure = caught.value
    assert failure.reason == reason
    report = openrouter_auth_failure_report(failure, stage="key_verification")
    rendered = format_error_report_for_log(report)
    assert report.message.key == f"error.openrouter_auth.{reason}"
    assert f"exception_type={exception_type}" in rendered
    assert SECRET not in rendered + str(failure) + repr(report)
    assert "proxy-password" not in rendered + str(failure) + repr(report)
    assert validate_diagnostics_for_sink(report.diagnostics, "persisted_logs").status == "accepted"
    assert validate_diagnostics_for_sink(report.diagnostics, "snackbar").status == "accepted"


@pytest.mark.asyncio
async def test_verification_adapter_preserves_noncredential_failure_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_response(monkeypatch, httpx.Response(503, text=UNSAFE))
    result = await ProviderVerifierAdapter().verify_provider_secret(
        ProviderVerificationRequest(
            provider="openrouter", secret_key="openrouter_api_key", secret_value=SECRET,
            secret_revision="revision", context={"flow": "manual"},
        )
    )
    assert result.status == "failed"
    assert result.message is not None
    assert result.message.key == "error.openrouter_auth.service_unavailable"
    assert result.diagnostics is not None
    assert result.diagnostics.category == "service_unavailable"
    assert result.diagnostics.status_code == 503
    assert SECRET not in repr(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403, 429, 500, 503])
async def test_pkce_exchange_preserves_status_without_response_or_code(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    install_response(monkeypatch, httpx.Response(status, text=UNSAFE))
    with pytest.raises(OpenRouterAuthenticationError) as caught:
        await OpenRouterPKCEClient(callback_origin="http://localhost:3000").exchange_code(
            code=SECRET, code_verifier=SECRET, code_challenge_method="S256"
        )
    report = openrouter_auth_failure_report(caught.value, stage="flow")
    assert report.diagnostics.status_code == status
    assert report.diagnostics.operation == "code_exchange"
    assert SECRET not in repr(report) + str(caught.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [UNSAFE, '[]', '{}', '{"key": null}', '{"key": ""}'])
async def test_pkce_unusable_success_payload_cannot_become_a_key(
    monkeypatch: pytest.MonkeyPatch, body: str
) -> None:
    install_response(monkeypatch, httpx.Response(200, text=body))
    with pytest.raises(OpenRouterAuthenticationError) as caught:
        await OpenRouterPKCEClient(callback_origin="http://localhost:3000").exchange_code(
            code=SECRET, code_verifier=SECRET, code_challenge_method="S256"
        )
    assert caught.value.reason == "invalid_response"
    assert SECRET not in str(caught.value)


@pytest.mark.asyncio
async def test_pkce_listener_bind_failure_is_classified_without_os_message() -> None:
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        port = occupied.getsockname()[1]
        client = OpenRouterPKCEClient(callback_origin=f"http://127.0.0.1:{port}", open_browser=False)
        with pytest.raises(OpenRouterAuthenticationError) as caught:
            await client.run_desktop_flow()
    assert caught.value.stage == "listener_start"
    assert caught.value.reason == "setup"
    assert caught.value.exception_type in {"OSError", "PermissionError"}


def failure_application(tmp_path: Path, flow: object, verifier: object):
    settings = SettingsOwner(
        path=tmp_path / "settings.json",
        canonical=AppSettingsVNext(),
        persistence=SettingsVNextCanonicalPersistenceAdapter(),
    )
    messages: list[str] = []
    diagnostics: list[str] = []
    routes: list[str] = []
    owner = OpenRouterPkceApplicationOwner(
        flow=cast(OpenRouterPkceFlowOwner, flow), verifier=verifier, settings=settings,
        provider_settings=cast(object, None), provider_runtime=cast(object, None),
        secret_store_factory=lambda _settings: pytest.fail("failure must not write credentials"),
        failure_message_sink=messages.append, failure_diagnostics_sink=diagnostics.append,
        failure_route=routes.append, results=SettingsTransactionResultOwner(),
    )
    target = OpenRouterPkceTarget(
        selection_alias=OpenRouterSelectionAlias.GEMMA4_26B_31B_BYOK,
        provider_intent=ProviderApplyIntent(()),
    )
    return owner, target, messages, diagnostics, routes


@pytest.mark.asyncio
@pytest.mark.parametrize("verification_failure", [False, True])
async def test_pkce_application_retains_failure_stage_and_does_not_commit(
    tmp_path: Path, verification_failure: bool
) -> None:
    async def run_flow():
        if not verification_failure:
            raise OpenRouterAuthenticationError(
                stage="callback_wait", reason="callback_timeout", exception_type="TimeoutError"
            )
        return OpenRouterPKCEExchangeResult(api_key=SECRET, user_id=None)

    async def verify_api_key(*_args):
        raise tls_error()

    owner, target, messages, diagnostics, routes = failure_application(
        tmp_path, SimpleNamespace(run_flow=run_flow), SimpleNamespace(verify_api_key=verify_api_key)
    )
    assert await owner.connect(target=target, launch_source="settings") is False
    reason = "tls" if verification_failure else "callback_timeout"
    stage = "key_verification" if verification_failure else "callback_wait"
    assert messages == [f"error.openrouter_auth.{reason}"]
    assert f"operation={stage}" in diagnostics[0]
    assert routes == ["settings"]
    assert SECRET not in str(diagnostics) + str(messages)
    assert not (tmp_path / "settings.json").exists()


@pytest.mark.asyncio
async def test_pkce_application_cancel_is_not_failure(tmp_path: Path) -> None:
    async def run_flow():
        raise asyncio.CancelledError

    owner, target, messages, diagnostics, routes = failure_application(
        tmp_path, SimpleNamespace(run_flow=run_flow), object()
    )
    with pytest.raises(asyncio.CancelledError):
        await owner.connect(target=target, launch_source="settings")
    assert messages == diagnostics == routes == []


def test_auth_failure_keys_resolve_in_every_supported_locale() -> None:
    previous = get_locale()
    try:
        for locale in ("en", "ko", "ja", "zh-CN", "ru"):
            set_locale(locale)
            for reason in ("credential_rejected", "access_denied", "http", "rate_limit",
                           "service_unavailable", "tls", "proxy", "network", "timeout",
                           "callback_timeout", "setup", "invalid_response", "unknown"):
                key = f"error.openrouter_auth.{reason}"
                assert t(key) != key
    finally:
        set_locale(previous)


@pytest.mark.asyncio
async def test_pkce_explicit_rejection_cannot_write_secret_or_settings(tmp_path: Path) -> None:
    async def run_flow():
        return OpenRouterPKCEExchangeResult(api_key=SECRET, user_id=None)

    async def verify_api_key(*_args):
        return False

    owner, target, messages, diagnostics, routes = failure_application(
        tmp_path, SimpleNamespace(run_flow=run_flow), SimpleNamespace(verify_api_key=verify_api_key)
    )
    assert await owner.connect(target=target, launch_source="onboarding") is False
    assert messages == ["error.openrouter_auth.credential_rejected"]
    assert "status=401" in diagnostics[0]
    assert routes == ["onboarding"]
    assert not (tmp_path / "settings.json").exists()


@pytest.mark.asyncio
async def test_ui_unexpected_exception_does_not_expose_secret_or_reject_key() -> None:
    async def verify(*_args):
        raise tls_error()

    messages: list[tuple[str, str]] = []
    controller = ApiKeyVerificationController(
        secret_key="openrouter_api_key", provider="openrouter", on_verify=verify,
        on_message=lambda key, message: messages.append((key, message)),
    )
    controller.set_value_getter(lambda: SECRET)
    await controller.verify_direct(SECRET)
    assert messages == [("snackbar.verification_error", "error.openrouter_auth.tls")]
    assert controller.status == "error"
    assert SECRET not in str(messages)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [200, 201])
async def test_pkce_exchange_accepts_successful_key_responses(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    install_response(monkeypatch, httpx.Response(status, json={"key": SECRET}))
    result = await OpenRouterPKCEClient(callback_origin="http://localhost:3000").exchange_code(
        code="synthetic-code", code_verifier="synthetic-verifier", code_challenge_method="S256"
    )
    assert result.api_key == SECRET


@pytest.mark.asyncio
async def test_missing_callback_times_out_and_closes_real_listener(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with socket.socket() as available:
        available.bind(("127.0.0.1", 0))
        port = available.getsockname()[1]
    monkeypatch.setattr("puripuly_heart.core.openrouter_pkce.CALLBACK_TIMEOUT_SECONDS", 0.01)
    client = OpenRouterPKCEClient(callback_origin=f"http://127.0.0.1:{port}", open_browser=False)
    with pytest.raises(OpenRouterAuthenticationError) as caught:
        await client.run_desktop_flow()
    assert caught.value.stage == "callback_wait"
    assert caught.value.reason == "callback_timeout"
    with socket.socket() as released:
        released.bind(("127.0.0.1", port))


@pytest.mark.asyncio
async def test_unsuccessful_browser_launch_is_reported_without_callback_wait_or_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    client = OpenRouterPKCEClient(callback_origin="http://127.0.0.1:43123")

    class Listener:
        def wait_for_code(self) -> str:
            calls.append("wait")
            pytest.fail("unsuccessful browser launch must not wait for a callback")

        def close(self) -> None:
            calls.append("closed")

    def unsuccessful_browser(_url: str) -> bool:
        calls.append("browser")
        return False

    monkeypatch.setattr(client, "_create_callback_listener", lambda _session: Listener())
    monkeypatch.setattr(
        "puripuly_heart.core.openrouter_pkce.webbrowser.open", unsuccessful_browser,
    )
    flow = OpenRouterPkceFlowOwner(client_factory=lambda: client)
    owner, target, messages, diagnostics, routes = failure_application(tmp_path, flow, object())

    assert await owner.connect(target=target, launch_source="settings") is False
    assert messages == ["error.openrouter_auth.setup"]
    assert "operation=browser_launch" in diagnostics[0]
    assert "category=lifecycle" in diagnostics[0]
    assert calls == ["browser", "closed"]
    assert routes == ["settings"]
    assert flow.active_client is None
    assert flow.get_runtime().active_task_names == ()
    assert not (tmp_path / "settings.json").exists()
