from __future__ import annotations

import asyncio
import socket
import ssl
import threading
import time
from contextlib import contextmanager

import httpx
import pytest
from deepgram import DeepgramClient
from deepgram.environment import DeepgramClientEnvironment
from websockets.sync.server import serve

from puripuly_heart.core import network_clients
from puripuly_heart.core.error_messages import format_error_report_for_log, stt_failure_report
from puripuly_heart.core.stt.backend import STTSessionProjection
from puripuly_heart.providers.stt.deepgram import DeepgramRealtimeSTTBackend
from tests.core.test_external_network import certificates as certificates
from tests.core.test_external_network import (
    isolated_network_environment as isolated_network_environment,
)


@pytest.mark.asyncio
async def test_deepgram_backend_requires_api_key() -> None:
    backend = DeepgramRealtimeSTTBackend(
        api_key="",
        language="en",
        model="nova-3",
        sample_rate_hz=16000,
    )

    with pytest.raises(ValueError):
        await backend.open_session()


@pytest.mark.asyncio
async def test_deepgram_backend_requires_valid_sample_rate() -> None:
    backend = DeepgramRealtimeSTTBackend(
        api_key="k",
        language="en",
        model="nova-3",
        sample_rate_hz=44100,
    )

    with pytest.raises(ValueError):
        await backend.open_session()


@pytest.mark.asyncio
async def test_deepgram_backend_requires_positive_connect_timeout() -> None:
    backend = DeepgramRealtimeSTTBackend(
        api_key="k",
        language="en",
        model="nova-3",
        sample_rate_hz=16000,
        connect_timeout_s=0.0,
    )

    with pytest.raises(ValueError):
        await backend.open_session()


@pytest.mark.asyncio
async def test_deepgram_backend_verify_api_key_handles_empty_and_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = []
    def respond(request):
        seen.append(request)
        return httpx.Response(200)
    monkeypatch.setattr(
        network_clients, "external_client",
        lambda **kwargs: httpx.Client(transport=httpx.MockTransport(respond)),
    )

    assert await DeepgramRealtimeSTTBackend.verify_api_key("") is False
    assert await DeepgramRealtimeSTTBackend.verify_api_key("secret") is True
    assert len(seen) == 1
    assert seen[0].headers["Authorization"] == "Token secret"


@pytest.mark.asyncio
async def test_deepgram_backend_verify_api_key_raises_on_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        network_clients, "external_client",
        lambda **kwargs: httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(401))),
    )

    with pytest.raises(httpx.HTTPStatusError) as caught:
        await DeepgramRealtimeSTTBackend.verify_api_key("secret")
    assert caught.value.response.status_code == 401


def local_sdk_client(monkeypatch: pytest.MonkeyPatch, url: str) -> None:
    def client(**kwargs):
        return DeepgramClient(
            **kwargs,
            environment=DeepgramClientEnvironment(base=url, production=url, agent=url),
        )

    monkeypatch.setattr("deepgram.DeepgramClient", client)


@pytest.mark.asyncio
async def test_scoped_deepgram_real_tls_failure_reaches_safe_report_without_startup_timeout(
    certificates, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    root, _, contexts = certificates
    monkeypatch.setenv("SSL_CERT_FILE", str(root))
    reached_handler = threading.Event()

    def handler(socket):
        reached_handler.set()

    with serve(handler, "127.0.0.1", 0, ssl=contexts["valid"]) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            local_sdk_client(monkeypatch, f"wss://127.0.0.1:{server.socket.getsockname()[1]}")
            backend = DeepgramRealtimeSTTBackend(
                api_key="synthetic-private-key", language="en", connect_timeout_s=10,
            )
            started = time.monotonic()
            with pytest.raises(ssl.SSLCertVerificationError) as caught:
                await asyncio.wait_for(
                    backend.open_session(projection=STTSessionProjection("scoped", "tls-failure")),
                    timeout=3,
                )
            assert time.monotonic() - started < 3
            report = stt_failure_report(
                caught.value, provider="deepgram", operation="open_session", channel="self",
            )
            assert report.diagnostics.category == "network"
            assert report.diagnostics.fields["tls_verify_code"] == 64
            assert report.diagnostics.fields["tls_backend"] == "openssl"
            assert report.diagnostics.fields["tls_source"] == "explicit_file"
            assert report.diagnostics.fields["transport"] == "wss"
            assert not reached_handler.is_set()
            rendered = format_error_report_for_log(report, sink="persisted_logs")
            assert "tls_verify_code=64" in rendered
            assert "127.0.0.1" not in rendered + caplog.text
            assert "synthetic-private-key" not in rendered + caplog.text
            assert str(root) not in rendered + caplog.text
            assert all(record.exc_info is None for record in caplog.records)
            assert not any(
                item.name == "deepgram-sdk" and item.is_alive() for item in threading.enumerate()
            )
        finally:
            server.shutdown()
            thread.join(timeout=5)
            assert not thread.is_alive()


@pytest.mark.asyncio
async def test_scoped_deepgram_real_success_opens_and_closes_sdk_session(
    certificates, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _, contexts = certificates
    monkeypatch.setenv("SSL_CERT_FILE", str(root))
    received = []
    audio_received = threading.Event()

    def handler(socket):
        received.append(socket.recv())
        audio_received.set()
        socket.recv()

    with serve(handler, "127.0.0.1", 0, ssl=contexts["valid"]) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            local_sdk_client(monkeypatch, f"wss://localhost:{server.socket.getsockname()[1]}")
            backend = DeepgramRealtimeSTTBackend(api_key="synthetic-key", language="en")
            session = await asyncio.wait_for(
                backend.open_session(projection=STTSessionProjection("scoped", "success")),
                timeout=3,
            )
            try:
                await session.send_audio(b"\\0\\0")
                assert await asyncio.to_thread(audio_received.wait, 3)
                assert received == [b"\\0\\0"]
            finally:
                await session.close()
            assert not any(
                item.name == "deepgram-sdk" and item.is_alive() for item in threading.enumerate()
            )
        finally:
            server.shutdown()
            thread.join(timeout=5)
            assert not thread.is_alive()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_scoped_deepgram_missing_open_preserves_timeout_or_cancellation_and_closes(
    monkeypatch: pytest.MonkeyPatch, cancel: bool
) -> None:
    entered = threading.Event()
    closed = threading.Event()

    class Connection:
        def on(self, *_args):
            pass

        def start_listening(self):
            pass

    @contextmanager
    def connect(*_args, **_kwargs):
        entered.set()
        try:
            yield Connection()
        finally:
            closed.set()

    monkeypatch.setattr("puripuly_heart.providers.stt.sdk_network.deepgram_listen_connect", connect)
    monkeypatch.setattr(
        network_clients, "external_client", lambda **kwargs: httpx.Client(trust_env=False)
    )
    backend = DeepgramRealtimeSTTBackend(
        api_key="synthetic-key", language="en", connect_timeout_s=10 if cancel else 0.1,
    )
    task = asyncio.create_task(
        backend.open_session(projection=STTSessionProjection("scoped", "missing-open"))
    )
    assert await asyncio.to_thread(entered.wait, 3)
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        with pytest.raises(RuntimeError, match="connection timeout") as caught:
            await task
        report = stt_failure_report(
            caught.value, provider="deepgram", operation="open_session", channel="self",
        )
        assert report.diagnostics.category == "timeout"
        assert report.diagnostics.fields["tls_backend"] == "unknown"
    assert closed.is_set()
    assert not any(
        item.name == "deepgram-sdk" and item.is_alive() for item in threading.enumerate()
    )


@pytest.mark.asyncio
async def test_scoped_deepgram_pending_tls_cancellation_keeps_event_loop_responsive(
    certificates, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _, _ = certificates
    monkeypatch.setenv("SSL_CERT_FILE", str(root))
    handshake_received = threading.Event()
    release = threading.Event()
    server_closed = threading.Event()
    release_time = []
    server_errors = []

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.settimeout(3)

        def pending_handshake():
            try:
                connection, _ = listener.accept()
                with connection:
                    connection.settimeout(3)
                    if connection.recv(4096):
                        handshake_received.set()
                    release.wait(3)
            except BaseException as exc:
                server_errors.append(exc)
            finally:
                server_closed.set()

        thread = threading.Thread(target=pending_handshake)
        thread.start()
        timer = None
        heartbeat = None
        opening = None
        try:
            local_sdk_client(monkeypatch, f"wss://127.0.0.1:{listener.getsockname()[1]}")
            backend = DeepgramRealtimeSTTBackend(
                api_key="synthetic-key", language="en", connect_timeout_s=10,
            )
            opening = asyncio.create_task(
                backend.open_session(projection=STTSessionProjection("scoped", "pending-tls"))
            )
            assert await asyncio.to_thread(handshake_received.wait, 3)
            started = time.monotonic()

            def release_server():
                release_time.append(time.monotonic() - started)
                release.set()

            async def tick():
                await asyncio.sleep(0.05)
                return time.monotonic() - started, release.is_set(), opening.done()

            timer = threading.Timer(0.6, release_server)
            timer.start()
            heartbeat = asyncio.create_task(tick())
            opening.cancel()
            elapsed, released, cancellation_finished = await heartbeat
            assert elapsed < 0.4
            assert not released
            assert not cancellation_finished
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(opening, 3)
            assert release_time[0] >= 0.6
            assert server_closed.is_set()
            assert not server_errors
            assert not any(
                item.name == "deepgram-sdk" and item.is_alive() for item in threading.enumerate()
            )
        finally:
            release.set()
            if timer is not None:
                timer.cancel()
                await asyncio.to_thread(timer.join, 3)
            if heartbeat is not None and not heartbeat.done():
                heartbeat.cancel()
                await asyncio.gather(heartbeat, return_exceptions=True)
            if opening is not None and not opening.done():
                opening.cancel()
                await asyncio.gather(opening, return_exceptions=True)
            await asyncio.to_thread(thread.join, 3)
            assert not thread.is_alive()
