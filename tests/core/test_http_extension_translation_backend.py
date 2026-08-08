from __future__ import annotations

import asyncio
import json
import logging
import ssl
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from uuid import uuid4

import httpx
import pytest

from puripuly_heart.core.storage.secrets import InMemorySecretStore
from puripuly_heart.core.translation_backend import TranslationBackendRequest
from puripuly_heart.core.translation_extensions import (
    HttpExtensionTranslationBackend,
    HttpExtensionTranslationError,
    TranslationExtensionConfigurationError,
    TranslationExtensionResponseError,
    parse_translation_extension,
)


@dataclass
class FakeResponse:
    status_code: int
    text: str


@dataclass
class FakeClient:
    response: FakeResponse | None = None
    failure: Exception | None = None
    calls: list[tuple[str, dict[str, object]]] = field(default_factory=list)
    closed: bool = False

    async def post(self, url: str, **kwargs: object) -> FakeResponse:
        self.calls.append((url, kwargs))
        if self.failure is not None:
            raise self.failure
        assert self.response is not None
        return self.response

    async def aclose(self) -> None:
        self.closed = True


def request() -> TranslationBackendRequest:
    return TranslationBackendRequest(
        utterance_id=uuid4(),
        text="Hello",
        system_prompt="unused",
        source_language="en",
        target_language="es",
    )


def extension(
    body_type: str = "json", body_value: object | None = None, response: object | None = None
):
    return parse_translation_extension(
        {
            "schema_version": 1,
            "id": "demo",
            "name": "Demo",
            "url": "https://example.test/translate",
            "headers": {"X-Request": "{{text}}"},
            "request": {
                "query": {"target": "{{target_language}}"},
                "body": {
                    "type": body_type,
                    **({"value": body_value} if body_type != "none" else {}),
                },
            },
            "response": response or {"type": "json", "pointer": "/translatedText"},
        }
    )


@pytest.mark.asyncio
async def test_backend_posts_json_and_closes_owned_client() -> None:
    client = FakeClient(response=FakeResponse(200, '{"translatedText":"Hola"}'))
    factory_calls: list[dict[str, object]] = []

    def factory(**kwargs: object) -> FakeClient:
        factory_calls.append(kwargs)
        return client

    backend = HttpExtensionTranslationBackend(
        extension(
            body_value={"q": "{{text}}"},
        ),
        InMemorySecretStore(),
        client_factory=factory,
    )

    result = await backend.translate(request())
    await backend.close()
    await backend.close()

    assert result.text == "Hola"
    assert client.calls[0][0] == "https://example.test/translate"
    assert client.calls[0][1]["json"] == {"q": "Hello"}
    assert client.calls[0][1]["params"] == {"target": "es"}
    assert client.calls[0][1]["headers"] == {"X-Request": "Hello"}
    assert factory_calls == [{"timeout": 10.0, "follow_redirects": False, "trust_env": False}]
    assert client.closed is True


@pytest.mark.asyncio
async def test_backend_posts_form_and_none_without_body() -> None:
    form_client = FakeClient(response=FakeResponse(200, "Hola"))
    form_backend = HttpExtensionTranslationBackend(
        extension(body_type="form", body_value={"q": "{{text}}"}, response={"type": "text"}),
        InMemorySecretStore(),
        client_factory=lambda **_: form_client,
    )
    await form_backend.translate(request())

    none_client = FakeClient(response=FakeResponse(200, "Hola"))
    none_backend = HttpExtensionTranslationBackend(
        extension(body_type="none", response={"type": "text"}),
        InMemorySecretStore(),
        client_factory=lambda **_: none_client,
    )
    await none_backend.translate(request())

    assert form_client.calls[0][1]["data"] == {"q": "Hello"}
    assert "json" not in form_client.calls[0][1]
    assert "data" not in none_client.calls[0][1]
    assert "json" not in none_client.calls[0][1]


@pytest.mark.asyncio
async def test_backend_applies_translation_concurrency_limit() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    active = 0
    maximum_active = 0

    class BlockingClient(FakeClient):
        async def post(self, url: str, **kwargs: object) -> FakeResponse:
            nonlocal active, maximum_active
            active += 1
            maximum_active = max(maximum_active, active)
            started.set()
            await release.wait()
            active -= 1
            return FakeResponse(200, "Hola")

    client = BlockingClient()
    backend = HttpExtensionTranslationBackend(
        extension(response={"type": "text"}),
        InMemorySecretStore(),
        concurrency_limit=1,
        client_factory=lambda **_: client,
    )

    first = asyncio.create_task(backend.translate(request()))
    await started.wait()
    second = asyncio.create_task(backend.translate(request()))
    await asyncio.sleep(0)

    assert maximum_active == 1
    assert active == 1
    release.set()
    await asyncio.gather(first, second)
    await backend.close()


@pytest.mark.asyncio
async def test_backend_does_not_start_queued_request_after_close() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    class BlockingClient(FakeClient):
        async def post(self, url: str, **kwargs: object) -> FakeResponse:
            nonlocal calls
            calls += 1
            started.set()
            await release.wait()
            return FakeResponse(200, "Hola")

    client = BlockingClient()
    backend = HttpExtensionTranslationBackend(
        extension(response={"type": "text"}),
        InMemorySecretStore(),
        concurrency_limit=1,
        client_factory=lambda **_: client,
    )

    first = asyncio.create_task(backend.translate(request()))
    await started.wait()
    second = asyncio.create_task(backend.translate(request()))
    await asyncio.sleep(0)
    await backend.close()
    release.set()

    await first
    with pytest.raises(TranslationExtensionConfigurationError, match="backend is closed"):
        await second
    assert calls == 1


@pytest.mark.asyncio
async def test_backend_propagates_request_cancellation() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingClient(FakeClient):
        async def post(self, url: str, **kwargs: object) -> FakeResponse:
            started.set()
            await release.wait()
            return FakeResponse(200, "Hola")

    client = BlockingClient()
    backend = HttpExtensionTranslationBackend(
        extension(response={"type": "text"}),
        InMemorySecretStore(),
        client_factory=lambda **_: client,
    )
    task = asyncio.create_task(backend.translate(request()))
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    await backend.close()
    assert client.closed is True


@pytest.mark.asyncio
async def test_backend_missing_secret_fails_before_client_creation() -> None:
    client_created = False

    def factory(**_: object) -> FakeClient:
        nonlocal client_created
        client_created = True
        return FakeClient(response=FakeResponse(200, "unused"))

    definition = parse_translation_extension(
        {
            "schema_version": 1,
            "id": "demo",
            "name": "Demo",
            "url": "https://example.test/translate",
            "request": {
                "body": {
                    "type": "json",
                    "value": {"key": "{{secret:api_key}}", "q": "{{text}}"},
                }
            },
            "response": {"type": "text"},
            "secrets": [{"id": "api_key", "label": "API Key"}],
        }
    )
    backend = HttpExtensionTranslationBackend(
        definition, InMemorySecretStore(), client_factory=factory
    )

    with pytest.raises(TranslationExtensionConfigurationError, match="missing required credential"):
        await backend.translate(request())

    assert client_created is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "category"),
    [
        (httpx.ReadTimeout("timeout"), "timeout"),
        (httpx.ConnectError("connect"), "connect error"),
        (httpx.NetworkError("network"), "transport error"),
    ],
)
async def test_backend_maps_transport_failures_without_payloads(
    failure: Exception,
    category: str,
) -> None:
    client = FakeClient(failure=failure)
    backend = HttpExtensionTranslationBackend(
        extension(body_value={"q": "{{text}}"}),
        InMemorySecretStore(),
        client_factory=lambda **_: client,
    )

    with pytest.raises(HttpExtensionTranslationError, match=category) as error:
        await backend.translate(request())

    assert "Hello" not in str(error.value)


@pytest.mark.asyncio
async def test_backend_maps_tls_connect_failures_without_chaining_private_errors() -> None:
    try:
        raise httpx.ConnectError("connect") from ssl.SSLError("TLS failure")
    except httpx.ConnectError as failure:
        client = FakeClient(failure=failure)

    backend = HttpExtensionTranslationBackend(
        extension(body_value={"q": "{{text}}"}),
        InMemorySecretStore(),
        client_factory=lambda **_: client,
    )

    with pytest.raises(HttpExtensionTranslationError, match="TLS error") as error:
        await backend.translate(request())

    assert error.value.__cause__ is None
    assert error.value.__context__ is None


@pytest.mark.asyncio
async def test_backend_rejects_non_success_and_bad_response_without_body_leak() -> None:
    for status in (400, 401, 403, 429, 500):
        status_client = FakeClient(response=FakeResponse(status, "secret response body"))
        status_backend = HttpExtensionTranslationBackend(
            extension(body_value={"q": "{{text}}"}),
            InMemorySecretStore(),
            client_factory=lambda **_: status_client,
        )

        with pytest.raises(HttpExtensionTranslationError, match=str(status)) as status_error:
            await status_backend.translate(request())
        assert "secret response body" not in str(status_error.value)
        assert len(status_client.calls) == 1
        await status_backend.close()

    response_client = FakeClient(response=FakeResponse(200, "not json"))
    response_backend = HttpExtensionTranslationBackend(
        extension(body_value={"q": "{{text}}"}),
        InMemorySecretStore(),
        client_factory=lambda **_: response_client,
    )
    with pytest.raises(TranslationExtensionResponseError, match="invalid response JSON"):
        await response_backend.translate(request())


@pytest.mark.asyncio
async def test_backend_uses_local_fake_http_server_without_public_network(
    caplog: pytest.LogCaptureFixture,
) -> None:
    received: list[tuple[str, dict[str, str], bytes]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers["Content-Length"] or "0")
            received.append((self.path, dict(self.headers), self.rfile.read(length)))
            payload = b'{"translatedText":"Hola local"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    backend = None
    try:
        definition = parse_translation_extension(
            {
                "schema_version": 1,
                "id": "demo",
                "name": "Demo",
                "url": f"http://127.0.0.1:{server.server_port}/translate",
                "headers": {"X-Source": "{{source_language}}"},
                "request": {
                    "query": {
                        "target": "{{target_language}}",
                        "source": "{{text}}",
                        "credential": "{{secret:api_key}}",
                    },
                    "body": {
                        "type": "json",
                        "value": {
                            "q": "{{text}}",
                            "api_key": "{{secret:api_key}}",
                        },
                    },
                },
                "response": {"type": "json", "pointer": "/translatedText"},
                "secrets": [{"id": "api_key", "label": "API Key"}],
            }
        )
        secrets = InMemorySecretStore()
        secrets.set("translation_extension.demo.api_key", "local-secret")
        backend = HttpExtensionTranslationBackend(definition, secrets)

        caplog.set_level(logging.DEBUG, logger="httpx")
        caplog.set_level(logging.DEBUG, logger="httpcore")
        caplog.clear()
        result = await backend.translate(request())

        assert result.text == "Hola local"
        assert len(received) == 1
        path, headers, body = received[0]
        assert path == "/translate?target=es&source=Hello&credential=local-secret"
        assert headers["X-Source"] == "en"
        assert json.loads(body) == {"q": "Hello", "api_key": "local-secret"}
        http_records = [record for record in caplog.records if record.name in {"httpx", "httpcore"}]
        assert all("Hello" not in record.getMessage() for record in http_records)
        assert all("local-secret" not in record.getMessage() for record in http_records)
    finally:
        if backend is not None:
            await backend.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.asyncio
async def test_backend_suppresses_descendant_http_failure_logs_without_private_values(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            return

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    backend = None
    source_value = "SOURCE-LEAK-ff19"
    secret_value = "LEAKME-SECRET-ff19\n"
    try:
        definition = parse_translation_extension(
            {
                "schema_version": 1,
                "id": "privacy",
                "name": "Privacy",
                "url": f"http://127.0.0.1:{server.server_port}/translate",
                "headers": {
                    "X-Source": "{{text}}",
                    "X-Secret": "{{secret:api_key}}",
                },
                "request": {
                    "query": {
                        "private_query": "{{text}}",
                        "query_secret": "{{secret:api_key}}",
                    },
                    "body": {
                        "type": "json",
                        "value": {
                            "private_body": "{{text}}",
                            "body_secret": "{{secret:api_key}}",
                        },
                    },
                },
                "response": {"type": "text"},
                "secrets": [{"id": "api_key", "label": "API Key"}],
            }
        )
        secrets = InMemorySecretStore()
        secrets.set("translation_extension.privacy.api_key", secret_value)
        backend = HttpExtensionTranslationBackend(definition, secrets)
        caplog.set_level(logging.DEBUG, logger="httpx")
        caplog.set_level(logging.DEBUG, logger="httpcore")
        caplog.clear()

        with pytest.raises(HttpExtensionTranslationError) as error:
            await backend.translate(
                TranslationBackendRequest(
                    utterance_id=uuid4(),
                    text=source_value,
                    system_prompt="",
                    source_language="en",
                    target_language="es",
                )
            )

        assert source_value not in str(error.value)
        assert "LEAKME-SECRET-ff19" not in str(error.value)
        assert error.value.__cause__ is None
        assert error.value.__context__ is None
        http_records = [
            record
            for record in caplog.records
            if record.name == "httpx" or record.name.startswith("httpcore.")
        ]
        assert http_records == []
    finally:
        if backend is not None:
            await backend.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
