from __future__ import annotations

import hashlib
import io
import json
import logging
import socket
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from uuid import uuid4

import httpx
import pytest
from puripuly_heart.app.services.provider_runtime_apply import ProviderRuntimeApplyPlan

from puripuly_heart.app.wiring import wiring_provider_runtime
from puripuly_heart.app.wiring.wiring_secrets_factory import create_secret_store
from puripuly_heart.composition import application_runtime
from puripuly_heart.composition.headless_application import compose_headless_application
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext, SecretsIntent
from puripuly_heart.config.settings_vnext.serialization import to_dict
from puripuly_heart.core import network_clients
from puripuly_heart.core.llm.fallback_racing import FallbackRacingLLMProvider
from puripuly_heart.core.llm.provider import SemaphoreLLMProvider
from puripuly_heart.core.messages import ErrorDiagnostics, TransactionResult
from puripuly_heart.core.runtime_logging import RuntimeLoggingSinks
from puripuly_heart.core.translation_backend import LlmTranslationBackend, TranslationBackendRequest
from puripuly_heart.providers.llm.deepseek import DeepSeekLLMProvider
from puripuly_heart.providers.llm.gemini import GEMINI_FLASH_GA_MODEL, GeminiLLMProvider
from puripuly_heart.providers.llm.openai import OpenAILLMProvider
from puripuly_heart.providers.llm.openrouter import OpenRouterLLMProvider
from puripuly_heart.ui.components.settings.api_key_verification_controller import (
    ApiKeyVerificationController,
)


@dataclass(frozen=True)
class Route:
    model: str
    connection: str
    provider: str
    secret_key: str
    concrete_type: type
    host: str


ROUTES = (
    Route(
        "deepseek_v4_flash_41",
        "official_byok",
        "deepseek",
        "deepseek_api_key",
        DeepSeekLLMProvider,
        "api.deepseek.com",
    ),
    Route(
        "gpt_6_luna",
        "official_byok",
        "openai",
        "openai_api_key",
        OpenAILLMProvider,
        "api.openai.com",
    ),
    Route(
        "gemini_flash",
        "official_byok",
        "google",
        "google_api_key",
        GeminiLLMProvider,
        "generativelanguage.googleapis.com",
    ),
    Route(
        "gemma4_26b_31b",
        "openrouter",
        "openrouter",
        "openrouter_api_key",
        OpenRouterLLMProvider,
        "openrouter.ai",
    ),
)
ROUTE_IDS = tuple(route.provider for route in ROUTES)
OLD_CREDENTIAL = "synthetic-active-byok-before"
NEW_CREDENTIAL = "synthetic-active-byok-after"
STT_CREDENTIAL = "synthetic-inactive-soniox"


def _fingerprint(value: str | None) -> str | None:
    return hashlib.sha256(value.encode()).hexdigest() if value is not None else None


def _assert_credential(actual: str | None, expected: str | None) -> None:
    if _fingerprint(actual) != _fingerprint(expected):
        pytest.fail("credential fingerprint mismatch")


def _assert_saved_credential(app, key, expected) -> None:
    store = create_secret_store(
        app.compatibility_settings().intent.secrets,
        config_path=app.control().settings.path,
    )
    _assert_credential(store.get(key), expected)


def _assert_backend_identity(runtime, expected, *, changed=False) -> None:
    same = runtime.provider is expected
    if same == changed:
        pytest.fail("live translation backend identity did not match the expected handoff")


def _assert_private(value: object) -> None:
    rendered = repr(value)
    if any(secret in rendered for secret in (OLD_CREDENTIAL, NEW_CREDENTIAL, STT_CREDENTIAL)):
        pytest.fail("consumer-visible result exposed a synthetic credential")


def _runtime(app):
    runtime = app.control().pipeline.llm_runtime
    assert runtime is not None
    return runtime


def _concrete(app, route):
    backend = _runtime(app).provider
    if not isinstance(backend, LlmTranslationBackend):
        pytest.fail("live translation backend is unavailable")
    provider = backend.provider
    if not isinstance(provider, SemaphoreLLMProvider):
        pytest.fail("live translation provider is missing its concurrency wrapper")
    provider = provider.inner
    if isinstance(provider, FallbackRacingLLMProvider):
        provider = provider.primary
    if not isinstance(provider, route.concrete_type):
        pytest.fail("live translation backend has the wrong concrete provider")
    return provider


def _presentation(app):
    return app._startup._presentation.presentation


def _verification(app, route):
    return getattr(app.compatibility_settings().state.provider_verification, route.provider)


async def _assert_disabled(app) -> None:
    status = await app.control().query("app.status", {})
    assert status["translation_enabled"] is False
    assert status["channels"]["self"]["desired_active"] is False
    assert status["channels"]["peer"]["desired_active"] is False
    assert status["channels"]["self"]["effective_active"] is False
    assert status["channels"]["peer"]["effective_active"] is False


async def _submit(app, command, arguments):
    control = app.control()
    receipt = await control.submit(command, arguments, request_id=str(uuid4()))
    result = await control.wait(receipt["operation_id"], timeout=5)
    _assert_private(result)
    return result


async def _verify_explicitly(app, route, credential):
    verified, _reason = await app.verify_api_key(route.provider, credential)
    assert verified is True
    app.persist_api_key_verification(route.provider, credential, verified)
    assert _verification(app, route).status == "verified"
    assert await app.apply_providers(force_rebuild_llm=True, persist_settings=False) is True
    assert _presentation(app).projections["translation_needs_key"] is False


async def _assert_live_request(app, route, credential, requests):
    concrete = _concrete(app, route)
    _assert_credential(concrete.api_key, credential)
    start = len(requests)
    translated = await _runtime(app).provider.translate(
        TranslationBackendRequest(
            utterance_id=uuid4(),
            text="local translation input",
            system_prompt="Translate the supplied text.",
            source_language="en",
            target_language="ko",
        )
    )
    assert translated.text == "translated locally"
    sent = requests[start:]
    assert sent
    assert all(host == route.host for host, _method, _path, _credential in sent)
    expected_fingerprint = _fingerprint(credential)
    assert all(fingerprint == expected_fingerprint for _host, _method, _path, fingerprint in sent)


@pytest.fixture
async def composed(tmp_path, monkeypatch):
    root = logging.getLogger()
    level = root.level
    monkeypatch.setattr(root, "handlers", list(root.handlers))
    monkeypatch.setattr(root, "filters", list(root.filters))
    for handler in root.handlers:
        monkeypatch.setattr(handler, "filters", list(handler.filters))
    monkeypatch.setenv("PURIPULY_HEART_SECRETS_PASSPHRASE", "isolated-test-passphrase")
    for name in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        application_runtime, "default_http_extensions_dir", lambda: tmp_path / "extensions"
    )
    requests = []

    def local_response(request):
        host = request.url.host
        credential = request.headers.get("authorization", "").removeprefix("Bearer ")
        if host == "generativelanguage.googleapis.com":
            credential = request.headers.get("x-goog-api-key") or request.url.params.get("key")
        requests.append((host, request.method, request.url.path, _fingerprint(credential)))
        if host == "generativelanguage.googleapis.com":
            if request.method == "GET" and request.url.path.endswith("/models"):
                return httpx.Response(
                    200,
                    json={"models": [{"name": f"models/{GEMINI_FLASH_GA_MODEL}"}]},
                )
            if request.method == "POST" and request.url.path.endswith(":generateContent"):
                return httpx.Response(
                    200,
                    json={
                        "candidates": [
                            {
                                "content": {
                                    "role": "model",
                                    "parts": [{"text": "translated locally"}],
                                },
                                "finishReason": "STOP",
                            }
                        ]
                    },
                )
        elif host in {"api.deepseek.com", "api.openai.com", "openrouter.ai"}:
            if request.method == "POST" and request.url.path.endswith("/chat/completions"):
                return httpx.Response(
                    200,
                    json={
                        "choices": [
                            {
                                "message": {"content": "translated locally"},
                                "finish_reason": "stop",
                            }
                        ]
                    },
                )
            if (
                host == "openrouter.ai"
                and request.method == "GET"
                and request.url.path.endswith("/key")
            ):
                return httpx.Response(
                    200, json={"data": {"limit": 10, "usage": 0, "is_free_tier": False}}
                )
        pytest.fail("unexpected external request in isolated BYOK regression")

    def async_client(**kwargs):
        kwargs.pop("tls", None)
        kwargs.pop("policy", None)
        kwargs["trust_env"] = False
        return httpx.AsyncClient(transport=httpx.MockTransport(local_response), **kwargs)

    def sync_client(**kwargs):
        kwargs.pop("tls", None)
        kwargs.pop("policy", None)
        kwargs["trust_env"] = False
        return httpx.Client(transport=httpx.MockTransport(local_response), **kwargs)

    def genai_options():
        return {"httpx_client": sync_client(), "httpx_async_client": async_client()}

    def forbidden_network(*_args, **_kwargs):
        pytest.fail("isolated BYOK regression attempted a real network connection")

    monkeypatch.setattr(network_clients, "external_async_client", async_client)
    monkeypatch.setattr(network_clients, "external_client", sync_client)
    monkeypatch.setattr(network_clients, "genai_http_options", genai_options)
    monkeypatch.setattr(network_clients, "external_websocket_connect", forbidden_network)
    monkeypatch.setattr(socket.socket, "connect", forbidden_network)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden_network)

    @asynccontextmanager
    async def open_application(route, *, initial=None, connection=None):
        path = tmp_path / "settings.json"
        settings = AppSettingsVNext()
        settings = replace(
            settings,
            intent=replace(
                settings.intent,
                translation=replace(
                    settings.intent.translation,
                    model=route.model,
                    connection=connection or route.connection,
                    openrouter_selected_source="managed" if connection else "byok",
                    openrouter_selection_alias=None,
                ),
                secrets=SecretsIntent(backend="encrypted_file", encrypted_file_path="secrets.json"),
                stt=replace(settings.intent.stt, provider="soniox"),
                peer_stt=replace(settings.intent.peer_stt, provider="soniox"),
                osc=replace(settings.intent.osc, connection_mode="off"),
                telemetry=replace(settings.intent.telemetry, enabled=False),
            ),
        )
        path.write_text(json.dumps(to_dict(settings)), encoding="utf-8")
        store = create_secret_store(settings.intent.secrets, config_path=path)
        store.set("soniox_api_key", STT_CREDENTIAL)
        if initial is not None:
            store.set(route.secret_key, initial)
        log_path = tmp_path / "headless.log"
        log_output = io.StringIO()
        sinks = RuntimeLoggingSinks(
            logging.StreamHandler(log_output),
            logging.FileHandler(log_path, encoding="utf-8"),
            log_path,
        )
        app = None
        try:
            app = compose_headless_application(path, runtime_logging_sinks=sinks)
            await app.start()
            app.control().bind_instance("active-byok-regression")
            await _assert_disabled(app)
            yield app, store, requests, path
        finally:
            try:
                if app is not None:
                    await app.stop()
            finally:
                sinks.close()
                sinks.stream_handler.close()
                root.setLevel(level)
                _assert_private(log_output.getvalue())
                _assert_private(log_path.read_text(encoding="utf-8"))

    yield open_application
    root.setLevel(level)


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ROUTES, ids=ROUTE_IDS)
@pytest.mark.parametrize(
    "initial", (None, OLD_CREDENTIAL), ids=("first-registration", "replacement")
)
async def test_active_byok_save_reaches_live_concrete_request_without_restart(
    composed, route, initial
):
    async with composed(route, initial=initial) as (app, _store, requests, _path):
        runtime = _runtime(app)
        before = runtime.provider
        if initial is None:
            _assert_backend_identity(runtime, None)
        else:
            await _verify_explicitly(app, route, initial)
            await _assert_live_request(app, route, initial, requests)
            before = runtime.provider
        selected = app.compatibility_settings().intent.translation
        assert (selected.model, selected.connection) == (route.model, route.connection)
        if route.provider == "openrouter":
            assert selected.openrouter_selected_source == "byok"
            assert "managed" not in (selected.openrouter_selection_alias or "")
        start = len(requests)

        result = await _submit(
            app, "secrets.set", {"name": route.secret_key, "value": NEW_CREDENTIAL}
        )

        assert result["status"] == "applied"
        assert _runtime(app) is runtime
        _assert_backend_identity(runtime, before, changed=True)
        _assert_saved_credential(app, route.secret_key, NEW_CREDENTIAL)
        assert _verification(app, route).status == "unknown"
        assert _presentation(app).projections["translation_needs_key"] is True
        assert len(requests) == start
        await _assert_disabled(app)
        await _assert_live_request(app, route, NEW_CREDENTIAL, requests)
        assert _verification(app, route).status == "unknown"
        await _verify_explicitly(app, route, NEW_CREDENTIAL)
        await _assert_disabled(app)


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ROUTES, ids=ROUTE_IDS)
@pytest.mark.parametrize("failure", ("secret-write", "settings-commit"))
async def test_failed_or_rolled_back_byok_save_keeps_old_storage_and_runtime(
    composed, monkeypatch, route, failure
):
    async with composed(route, initial=OLD_CREDENTIAL) as (app, store, requests, path):
        await _verify_explicitly(app, route, OLD_CREDENTIAL)
        runtime = _runtime(app)
        backend = runtime.provider
        generation = runtime.generation
        original_settings = path.read_bytes()
        before_verification = _verification(app, route)
        if failure == "secret-write":
            original_set = type(store).set

            def fail_set(self, key, value):
                if key == route.secret_key:
                    raise OSError("isolated credential write failure")
                return original_set(self, key, value)

            monkeypatch.setattr(type(store), "set", fail_set)
        else:

            def fail_persist(_self, *_args):
                raise OSError("isolated settings commit failure")

            monkeypatch.setattr(type(app.control().settings.persistence), "persist", fail_persist)
        result = await _submit(
            app, "secrets.set", {"name": route.secret_key, "value": NEW_CREDENTIAL}
        )

        assert result["status"] == "persistence_failed"
        transaction = result["transaction"]
        assert transaction["status"] == (
            "secret_write_failed"
            if failure == "secret-write"
            else "settings_commit_failed_secret_restored"
        )
        assert transaction["diagnostics"] is not None
        _assert_backend_identity(runtime, backend)
        assert runtime.generation == generation
        _assert_saved_credential(app, route.secret_key, OLD_CREDENTIAL)
        assert path.read_bytes() == original_settings
        assert _verification(app, route) == before_verification
        await _assert_live_request(app, route, OLD_CREDENTIAL, requests)
        await _assert_disabled(app)


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ROUTES, ids=ROUTE_IDS)
async def test_inactive_byok_key_saves_do_not_rebuild_selected_provider(composed, route):
    async with composed(route, initial=OLD_CREDENTIAL) as (app, _store, requests, _path):
        runtime = _runtime(app)
        backend = runtime.provider
        generation = runtime.generation
        for inactive in ROUTES:
            if inactive is route:
                continue
            start = len(requests)
            result = await _submit(
                app, "secrets.set", {"name": inactive.secret_key, "value": NEW_CREDENTIAL}
            )
            assert result["status"] == "applied"
            _assert_saved_credential(app, inactive.secret_key, NEW_CREDENTIAL)
            _assert_backend_identity(runtime, backend)
            assert runtime.generation == generation
            assert len(requests) == start
        await _assert_live_request(app, route, OLD_CREDENTIAL, requests)
        await _assert_disabled(app)


@pytest.mark.asyncio
@pytest.mark.parametrize("connection", ("managed", "managed_china"))
async def test_managed_route_excludes_all_ordinary_byok_key_saves(composed, connection):
    route = ROUTES[3] if connection == "managed" else ROUTES[0]
    async with composed(route, connection=connection) as (app, _store, requests, _path):
        runtime = _runtime(app)
        backend = runtime.provider
        generation = runtime.generation
        selected = app.compatibility_settings().intent.translation
        assert selected.connection == connection
        assert selected.openrouter_selected_source == "managed"
        for ordinary in ROUTES:
            start = len(requests)
            result = await _submit(
                app, "secrets.set", {"name": ordinary.secret_key, "value": NEW_CREDENTIAL}
            )
            assert result["status"] == "applied"
            _assert_saved_credential(app, ordinary.secret_key, NEW_CREDENTIAL)
            _assert_backend_identity(runtime, backend)
            assert runtime.generation == generation
            assert len(requests) == start
        assert app.compatibility_settings().intent.translation == selected
        await _assert_disabled(app)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ("unknown", "known", "wrapped-known"))
async def test_stored_key_runtime_failure_is_degraded_and_preserves_known_cause(
    composed, monkeypatch, failure
):
    route = ROUTES[0]
    async with composed(route, initial=OLD_CREDENTIAL) as (app, _store, requests, _path):
        diagnostic = ErrorDiagnostics(
            component="deepseek",
            operation="provider_construction",
            code="provider.auth",
            category="auth",
            visibility="basic",
            content_policy="metadata_only",
            status_code=401,
            retry_after_ms=250,
            fields={"provider": "deepseek"},
        )
        original_backend_factory = wiring_provider_runtime.create_translation_backend

        def fail_backend(**_kwargs):
            error = RuntimeError("isolated provider construction failure")
            if failure != "unknown":
                error.failure_diagnostics = diagnostic
            if failure == "wrapped-known":
                raise RuntimeError("isolated factory wrapper failure") from error
            raise error

        monkeypatch.setattr(wiring_provider_runtime, "create_translation_backend", fail_backend)
        result = await _submit(
            app, "secrets.set", {"name": route.secret_key, "value": NEW_CREDENTIAL}
        )

        assert result["status"] == "degraded"
        transaction = result["transaction"]
        assert transaction["status"] == "settings_commit_success_runtime_degraded"
        _assert_saved_credential(app, route.secret_key, NEW_CREDENTIAL)
        assert _verification(app, route).status == "unknown"
        assert _presentation(app).projections["translation_needs_key"] is True
        actual = transaction["diagnostics"]
        assert actual is not None
        assert actual["code"] != "provider_runtime_apply_unavailable"
        if failure == "unknown":
            assert actual["code"] == "provider_runtime_apply_exception"
            assert actual["fields"]["exception_class"] == "RuntimeError"
        else:
            assert actual["diagnostic_id"] == diagnostic.diagnostic_id
            assert actual["code"] == diagnostic.code
            assert actual["category"] == diagnostic.category
            assert actual["status_code"] == diagnostic.status_code
            assert actual["retry_after_ms"] == diagnostic.retry_after_ms
        await _assert_disabled(app)
        monkeypatch.setattr(
            wiring_provider_runtime, "create_translation_backend", original_backend_factory
        )
        assert await app.apply_providers(persist_settings=False) is True
        await _assert_live_request(app, route, NEW_CREDENTIAL, requests)
        assert _verification(app, route).status == "unknown"
        assert _presentation(app).projections["translation_needs_key"] is True
        await _assert_disabled(app)


@pytest.mark.asyncio
async def test_gui_key_flow_verifies_persisted_key_after_runtime_degradation(composed, monkeypatch):
    route = ROUTES[0]
    async with composed(route, initial=OLD_CREDENTIAL) as (app, _store, requests, _path):
        diagnostic = ErrorDiagnostics(
            component="provider.llm",
            operation="provider_construction",
            code="provider.auth",
            category="auth",
            visibility="basic",
            content_policy="metadata_only",
            status_code=401,
            retry_after_ms=None,
            fields={},
        )

        def fail_backend(**_kwargs):
            error = RuntimeError("controlled provider construction failure")
            error.failure_diagnostics = diagnostic
            raise error

        async def save(key, value):
            persisted = await app.persist_provider_secret_change(key, value)
            assert persisted is True
            return persisted

        async def verify(provider, value):
            verified, message = await app.verify_api_key(provider, value)
            app.persist_api_key_verification(provider, value, verified)
            return verified, message

        monkeypatch.setattr(wiring_provider_runtime, "create_translation_backend", fail_backend)
        controller = ApiKeyVerificationController(
            secret_key=route.secret_key,
            provider=route.provider,
            on_save=save,
            on_verify=verify,
        )
        controller.set_value_getter(lambda: NEW_CREDENTIAL)
        controller.notify_edit()
        controller.handle_blur(NEW_CREDENTIAL)
        await controller.run_pending()

        assert controller.status == "success"
        assert _verification(app, route).status == "verified"
        _assert_saved_credential(app, route.secret_key, NEW_CREDENTIAL)
        _assert_backend_identity(_runtime(app), None)
        transaction = app._provider.provider_application.results.current
        assert transaction.status == "settings_commit_success_runtime_degraded"
        assert transaction.diagnostics.diagnostic_id == diagnostic.diagnostic_id
        assert transaction.diagnostics.code == diagnostic.code
        assert requests
        assert all(
            fingerprint == _fingerprint(NEW_CREDENTIAL)
            for _host, _method, _path, fingerprint in requests
        )
        await _assert_disabled(app)


@pytest.mark.asyncio
async def test_failed_managed_to_byok_apply_clears_managed_usage_view(composed, monkeypatch):
    route = ROUTES[3]
    async with composed(route, initial=OLD_CREDENTIAL, connection="managed") as (
        app,
        _store,
        _requests,
        _path,
    ):
        presentation = _presentation(app)
        presentation.set_settings_managed_key_state(
            visible=True, remaining_percent=50, referral_id=None, pass_status=None
        )
        diagnostic = ErrorDiagnostics(
            component="provider.llm",
            operation="provider_construction",
            code="provider.service_unavailable",
            category="service_unavailable",
            visibility="basic",
            content_policy="metadata_only",
            status_code=503,
            retry_after_ms=None,
            fields={},
        )

        def fail_backend(**_kwargs):
            error = RuntimeError("controlled provider construction failure")
            error.failure_diagnostics = diagnostic
            raise error

        monkeypatch.setattr(wiring_provider_runtime, "create_translation_backend", fail_backend)
        result = await _submit(
            app,
            "settings.apply",
            {
                "changes": {
                    "translation.model": ROUTES[0].model,
                    "translation.connection": "official_byok",
                }
            },
        )

        assert result["status"] == "degraded"
        assert result["transaction"]["diagnostics"]["diagnostic_id"] == diagnostic.diagnostic_id
        assert app.compatibility_settings().intent.translation.connection == "official_byok"
        assert presentation.projections["managed_key"]["visible"] is False
        assert presentation.projections["managed_key"]["remaining_percent"] is None
        _assert_backend_identity(_runtime(app), None)
        await _assert_disabled(app)


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ("peer", "self", "gpu", "convergence"))
async def test_llm_and_stt_failure_preserves_both_causes_and_unforced_recovery(
    composed, monkeypatch, stage
):
    route = ROUTES[0]
    async with composed(route, initial=OLD_CREDENTIAL) as (app, _store, requests, _path):
        owner = app._provider.provider_application.runtime
        llm_diagnostic = ErrorDiagnostics(
            component="provider.llm",
            operation="provider_construction",
            code="provider.auth",
            category="auth",
            visibility="basic",
            content_policy="metadata_only",
            status_code=401,
            retry_after_ms=None,
            fields={},
        )
        stt_diagnostic = ErrorDiagnostics(
            component="provider.stt",
            operation="provider_application",
            code="stt.provider_not_ready",
            category="lifecycle",
            visibility="basic",
            content_policy="metadata_only",
            status_code=None,
            retry_after_ms=None,
            fields={},
        )

        def fail_backend(**_kwargs):
            error = RuntimeError("controlled provider construction failure")
            error.failure_diagnostics = llm_diagnostic
            raise RuntimeError("controlled provider construction wrapper") from error

        async def fail_stt(*_args):
            error = RuntimeError("controlled independent runtime failure")
            error.failure_diagnostics = stt_diagnostic
            raise error

        build_plan = type(owner).build_plan

        def combined_plan(self, *args, **kwargs):
            return replace(
                build_plan(self, *args, **kwargs),
                should_refresh_peer=stage == "peer",
                should_refresh_self_stt=stage == "self",
                coordinated_gpu_restart=stage == "gpu",
            )

        with monkeypatch.context() as failure:
            failure.setattr(type(owner), "build_plan", combined_plan)
            failure.setattr(wiring_provider_runtime, "create_translation_backend", fail_backend)
            if stage == "convergence":
                state = owner.state_provider
                failure.setattr(owner, "self_runtime_convergence", lambda _settings: False)
                failure.setattr(
                    owner,
                    "state_provider",
                    lambda settings: replace(
                        state(settings), self_failure_diagnostics=stt_diagnostic
                    ),
                )
            else:
                failure.setattr(
                    owner,
                    {"peer": "refresh_peer", "self": "refresh_self_stt", "gpu": "recover_gpu"}[
                        stage
                    ],
                    fail_stt,
                )
            result = await _submit(
                app, "secrets.set", {"name": route.secret_key, "value": NEW_CREDENTIAL}
            )

        assert result["status"] == "degraded"
        transaction = result["transaction"]
        assert transaction["diagnostics"]["diagnostic_id"] == llm_diagnostic.diagnostic_id
        assert transaction["diagnostics"]["code"] == llm_diagnostic.code
        secondary = transaction["secondary_diagnostics"]
        assert len(secondary) == 1
        assert secondary[0]["diagnostic_id"] == stt_diagnostic.diagnostic_id
        assert secondary[0]["code"] == stt_diagnostic.code
        assert secondary[0]["cause_id"] == llm_diagnostic.diagnostic_id
        _assert_saved_credential(app, route.secret_key, NEW_CREDENTIAL)
        _assert_backend_identity(_runtime(app), None)
        await _assert_disabled(app)

        assert await app.apply_providers(persist_settings=False) is True
        await _assert_live_request(app, route, NEW_CREDENTIAL, requests)
        assert _verification(app, route).status == "unknown"
        await _assert_disabled(app)


@pytest.mark.asyncio
@pytest.mark.parametrize("restoration_fails", (False, True))
async def test_committed_runtime_resync_retains_both_failures_and_restore_cause(
    composed, monkeypatch, restoration_fails
):
    route = ROUTES[0]
    async with composed(route, initial=OLD_CREDENTIAL) as (app, store, requests, _path):
        application = app._provider.provider_application
        owner = application.runtime
        current = app.compatibility_settings()
        store.set(route.secret_key, NEW_CREDENTIAL)
        save_diagnostic = ErrorDiagnostics(
            component="settings_repository",
            operation="save",
            code="settings_save_failed",
            category="transaction",
            visibility="basic",
            content_policy="metadata_only",
            status_code=None,
            retry_after_ms=None,
            fields={},
        )
        llm_diagnostic = replace(
            save_diagnostic,
            diagnostic_id=uuid4().hex,
            component="provider.llm",
            code="provider.auth",
            category="auth",
            status_code=401,
        )
        stt_diagnostic = replace(
            save_diagnostic,
            diagnostic_id=uuid4().hex,
            component="provider.stt",
            code="stt.provider_not_ready",
            category="lifecycle",
        )
        failure_result = TransactionResult(
            status="settings_commit_success_runtime_degraded",
            message=None,
            diagnostics=save_diagnostic,
        )

        def fail_backend(**_kwargs):
            error = RuntimeError("controlled provider construction failure")
            error.failure_diagnostics = llm_diagnostic
            raise RuntimeError("controlled provider construction wrapper") from error

        async def fail_stt():
            error = RuntimeError("controlled independent runtime failure")
            error.failure_diagnostics = stt_diagnostic
            raise error

        sync_memory = application.sync_memory
        sync_attempts = 0

        def restore(settings):
            nonlocal sync_attempts
            sync_attempts += 1
            if restoration_fails and sync_attempts == 2:
                raise RuntimeError("controlled memory restoration failure")
            sync_memory(settings)

        with monkeypatch.context() as failure:
            failure.setattr(wiring_provider_runtime, "create_translation_backend", fail_backend)
            failure.setattr(owner, "refresh_self_stt", fail_stt)
            failure.setattr(application, "sync_memory", restore)
            await application._resync_committed_provider_runtime(
                base_settings=current,
                committed_settings=current,
                plan=ProviderRuntimeApplyPlan(True, False, True),
                failure_result=failure_result,
            )

        result = application.results.current
        _assert_private(result)
        assert result.status == "settings_commit_success_runtime_degraded"
        assert result.diagnostics.diagnostic_id == save_diagnostic.diagnostic_id
        secondary = result.secondary_diagnostics
        assert len(secondary) == (3 if restoration_fails else 2)
        assert secondary[0].diagnostic_id == llm_diagnostic.diagnostic_id
        assert secondary[0].cause_id == save_diagnostic.diagnostic_id
        assert secondary[1].diagnostic_id == stt_diagnostic.diagnostic_id
        assert secondary[1].cause_id == llm_diagnostic.diagnostic_id
        if restoration_fails:
            assert secondary[2].code == "settings_memory_restore_failed"
            assert secondary[2].cause_id == llm_diagnostic.diagnostic_id
        _assert_saved_credential(app, route.secret_key, NEW_CREDENTIAL)
        _assert_backend_identity(_runtime(app), None)

        assert await app.apply_providers(persist_settings=False) is True
        await _assert_live_request(app, route, NEW_CREDENTIAL, requests)
        await _assert_disabled(app)
