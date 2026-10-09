from __future__ import annotations

import asyncio
import json
import socket
import ssl
import threading
import time
from contextlib import asynccontextmanager
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from deepgram import AsyncDeepgramClient
from deepgram.environment import DeepgramClientEnvironment
from websockets.asyncio.server import serve
from websockets.exceptions import InvalidStatus
from websockets.frames import Frame, Opcode
from websockets.protocol import State
from websockets.server import ServerProtocol

from puripuly_heart.core import network_clients
from puripuly_heart.core.error_messages import format_error_report_for_log, stt_failure_report
from puripuly_heart.core.stt.backend import STTSessionProjection
from puripuly_heart.providers.stt.deepgram import DeepgramRealtimeSTTBackend
from tests.core.test_external_network import certificates as certificates
from tests.core.test_external_network import (
    isolated_network_environment as isolated_network_environment,
)
from tests.providers.test_deepgram_reuse import _next, _request, _seal
from tests.providers.test_protocol_a_scoped_sessions import _deepgram_result


@pytest.fixture
def created_http_clients(monkeypatch: pytest.MonkeyPatch):
    clients = []
    original = httpx.AsyncClient.__init__

    def record(client, *args, **kwargs):
        original(client, *args, **kwargs)
        clients.append(client)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", record)
    yield clients
    assert all(client.is_closed for client in clients)


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
        return AsyncDeepgramClient(
            **kwargs,
            environment=DeepgramClientEnvironment(base=url, production=url, agent=url),
        )

    monkeypatch.setattr("deepgram.AsyncDeepgramClient", client)


@pytest.mark.asyncio
async def test_scoped_deepgram_real_tls_failure_reaches_safe_report_without_startup_timeout(
    certificates, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    created_http_clients,
) -> None:
    root, _, contexts = certificates
    monkeypatch.setenv("SSL_CERT_FILE", str(root))
    reached_handler = asyncio.Event()

    async def handler(socket):
        reached_handler.set()

    async with serve(handler, "127.0.0.1", 0, ssl=contexts["valid"]) as server:
        local_sdk_client(monkeypatch, f"wss://127.0.0.1:{server.sockets[0].getsockname()[1]}")
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
    assert len(created_http_clients) == 1
    assert not any(task.get_name().startswith("deepgram") for task in asyncio.all_tasks())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "keyterms", "expected_keyterms"),
    [("nova-3", ["first", "second"], ["first", "second"]), ("nova-3", [], None),
     ("nova-2", ["unsupported"], None)],
)
async def test_scoped_deepgram_real_sdk_audio_control_result_reuse_and_closure(
    certificates, monkeypatch: pytest.MonkeyPatch, created_http_clients,
    model, keyterms, expected_keyterms,
) -> None:
    root, _, contexts = certificates
    monkeypatch.setenv("SSL_CERT_FILE", str(root))
    received = []
    requests = []
    keepalive = asyncio.Event()
    closed = asyncio.Event()

    async def handler(socket):
        requests.append((socket.request.path, socket.request.headers))
        finalizations = 0
        try:
            async for message in socket:
                if isinstance(message, bytes):
                    received.append(message)
                    continue
                control = json.loads(message)["type"]
                received.append(control)
                if control == "KeepAlive":
                    keepalive.set()
                elif control == "Finalize":
                    finalizations += 1
                    if finalizations < 3:
                        result = _deepgram_result(
                            "recognized" if finalizations == 1 else "", from_finalize=True
                        )
                        await socket.send(result.model_dump_json())
                elif control == "CloseStream":
                    await socket.close()
        finally:
            closed.set()

    async with serve(handler, "127.0.0.1", 0, ssl=contexts["valid"]) as server:
        local_sdk_client(monkeypatch, f"wss://localhost:{server.sockets[0].getsockname()[1]}")
        backend = DeepgramRealtimeSTTBackend(
            api_key="synthetic-key", language="en", model=model, keyterms=keyterms,
            drain_timeout_s=0.1,
        )
        session = await asyncio.wait_for(
            backend.open_session(projection=STTSessionProjection("scoped", "epoch")), timeout=3
        )
        try:
            if expected_keyterms:
                await asyncio.wait_for(keepalive.wait(), 6)
            for order in (1, 2):
                request = _request(order)
                await session.begin_turn(request)
                await session.send_turn_audio(
                    request.identity, bytes([order, 0]), payload_sequence=1,
                    source_ranges=(), context_only=False,
                )
                await _seal(session, request)
                if order == 1:
                    assert (await _next(session)).text == "recognized"
                terminal = await _next(session)
                assert terminal.outcome == ("final" if order == 1 else "empty")
                assert terminal.epoch_disposition == "reuse"
                assert terminal.identity == request.identity
            request = _request(3)
            await session.begin_turn(request)
            await _seal(session, request)
            terminal = await _next(session)
            assert terminal.epoch_disposition == "retire"
            await asyncio.wait_for(closed.wait(), 1)
        finally:
            await asyncio.wait_for(session.close(), 1)
        assert [item for item in received if item != "KeepAlive"] == [
            b"\x01\0", "Finalize", b"\x02\0", "Finalize", "Finalize", "CloseStream",
        ]
        assert len(requests) == 1
        path, headers = requests[0]
        assert parse_qs(urlsplit(path).query).get("keyterm") == expected_keyterms
        assert headers["Authorization"] == "Token synthetic-key"
        assert headers["x-deepgram-session-id"]
    assert len(created_http_clients) == 1
    assert not any(task.get_name().startswith("deepgram") for task in asyncio.all_tasks())


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [300, 301, 302, 303, 307, 308, 401])
async def test_deepgram_rejects_redirect_or_auth_before_second_contact_and_closes(
    certificates, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    created_http_clients, status,
) -> None:
    from deepgram.core.api_error import ApiError

    root, _, contexts = certificates
    monkeypatch.setenv("SSL_CERT_FILE", str(root))
    destination_contacts = []
    origins = []

    async def destination(reader, writer):
        destination_contacts.append(writer.get_extra_info("peername"))
        writer.close()
        await writer.wait_closed()

    async def handler(socket):
        pytest.fail("Rejected handshake must not enter the SDK stream")

    async with await asyncio.start_server(destination, "127.0.0.1", 0) as canary:
        destination_url = f"wss://127.0.0.1:{canary.sockets[0].getsockname()[1]}/private-path"

        def reject(connection, request):
            origins.append(connection)
            response = connection.respond(status, "rejected")
            response.headers["Location"] = destination_url
            return response

        async with serve(
            handler, "::1", 0, ssl=contexts["valid"], process_request=reject,
        ) as server:
            url = f"wss://localhost:{server.sockets[0].getsockname()[1]}"
            local_sdk_client(monkeypatch, url)
            backend = DeepgramRealtimeSTTBackend(api_key="private-api-key", language="en")
            with pytest.raises(ApiError) as caught:
                await asyncio.wait_for(backend.open_session(), 2)
            assert caught.value.status_code == status
            assert isinstance(caught.value.__cause__, InvalidStatus)
            assert caught.value.body == (
                "Websocket initialized with invalid credentials."
                if status == 401 else "Unexpected error when initializing websocket connection."
            )
            assert len(origins) == 1
            await asyncio.wait_for(origins[0].wait_closed(), 1)
            assert not destination_contacts
            report = stt_failure_report(
                caught.value, provider="deepgram", operation="open_session", channel="self",
            )
            assert report.diagnostics.fields["transport"] == "wss"
            rendered = format_error_report_for_log(report, sink="persisted_logs")
            for private in ("private-api-key", destination_url, url, str(root)):
                assert private not in rendered + caplog.text
            assert all(record.exc_info is None for record in caplog.records)
    assert len(created_http_clients) == 1
    assert not any(task.get_name().startswith("deepgram") for task in asyncio.all_tasks())


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_scoped_deepgram_missing_open_preserves_timeout_or_cancellation_and_closes(
    monkeypatch: pytest.MonkeyPatch, cancel: bool
) -> None:
    entered = asyncio.Event()
    closed = asyncio.Event()

    class Connection:
        def on(self, *_args):
            pass

        async def start_listening(self):
            await asyncio.Event().wait()

    @asynccontextmanager
    async def connect(*_args, **_kwargs):
        entered.set()
        try:
            yield Connection()
        finally:
            closed.set()

    monkeypatch.setattr("puripuly_heart.providers.stt.sdk_network.deepgram_listen_connect", connect)
    monkeypatch.setattr(
        network_clients, "external_async_client", lambda **kwargs: httpx.AsyncClient(trust_env=False)
    )
    backend = DeepgramRealtimeSTTBackend(
        api_key="synthetic-key", language="en", connect_timeout_s=10 if cancel else 0.1,
    )
    task = asyncio.create_task(
        backend.open_session(projection=STTSessionProjection("scoped", "missing-open"))
    )
    await asyncio.wait_for(entered.wait(), 3)
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
    assert not any(task.get_name().startswith("deepgram") for task in asyncio.all_tasks())


@pytest.mark.asyncio
async def test_scoped_deepgram_pending_tls_cancellation_keeps_event_loop_responsive(
    certificates, monkeypatch: pytest.MonkeyPatch, created_http_clients,
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
            assert cancellation_finished
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(opening, 3)
            assert not release_time
            assert not server_errors
            assert len(created_http_clients) == 1
            assert not any(task.get_name().startswith("deepgram") for task in asyncio.all_tasks())
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


@pytest.mark.asyncio
async def test_deepgram_real_tls_duplex_legacy_results_finalize_and_stop(
    certificates, monkeypatch: pytest.MonkeyPatch, created_http_clients,
):
    root, _, contexts = certificates
    monkeypatch.setenv("SSL_CERT_FILE", str(root))
    received = []
    closed = asyncio.Event()
    finalized = asyncio.Event()
    transcripts = []

    async def handler(socket):
        try:
            async for message in socket:
                if isinstance(message, bytes):
                    received.append(message)
                    result = _deepgram_result(str(message[0]))
                    await socket.send(result.model_dump_json())
                else:
                    control = json.loads(message)["type"]
                    received.append(control)
                    if control == "Finalize":
                        await socket.send(_deepgram_result("", from_finalize=True).model_dump_json())
        finally:
            closed.set()

    async with serve(handler, "::1", 0, ssl=contexts["valid"]) as server:
        local_sdk_client(monkeypatch, f"wss://localhost:{server.sockets[0].getsockname()[1]}")
        session = await asyncio.wait_for(
            DeepgramRealtimeSTTBackend(api_key="synthetic-key", language="en").open_session(), 2
        )

        async def collect():
            async for event in session.events():
                transcripts.append(event.text)
                if event.text == "":
                    finalized.set()

        collecting = asyncio.create_task(collect())
        try:
            for order in range(1, 33):
                await session.send_audio(bytes([order]) * 2048)
                await asyncio.sleep(0.001)
            await session.on_speech_end()
            await asyncio.wait_for(finalized.wait(), 2)
            await session.stop()
            await asyncio.wait_for(session.close(), 1)
            await asyncio.wait_for(collecting, 1)
            await asyncio.wait_for(closed.wait(), 1)
            assert transcripts == [str(order) for order in range(1, 33)] + [""]
            assert received == [bytes([order]) * 2048 for order in range(1, 33)] + ["Finalize"]
        finally:
            if not collecting.done():
                collecting.cancel()
                await asyncio.gather(collecting, return_exceptions=True)
            await session.close()
    assert len(created_http_clients) == 1
    assert not any(task.get_name().startswith("deepgram") for task in asyncio.all_tasks())


@asynccontextmanager
async def unanswered_deepgram_peer(context):
    close_received = asyncio.Event()
    disconnected = asyncio.Event()
    media = asyncio.Queue()
    tasks = set()
    writers = set()

    async def handle(reader, writer):
        writers.add(writer)
        protocol = ServerProtocol()
        try:
            protocol.receive_data(await reader.readuntil(b"\r\n\r\n"))
            request, = protocol.events_received()
            protocol.send_response(protocol.accept(request))
            for data in protocol.data_to_send():
                writer.write(data)
            await writer.drain()
            while True:
                try:
                    data = await reader.read(65536)
                except ConnectionResetError:
                    disconnected.set()
                    return
                if not data:
                    disconnected.set()
                    return
                protocol.receive_data(data)
                for event in protocol.events_received():
                    if isinstance(event, Frame):
                        if event.opcode == Opcode.CLOSE:
                            close_received.set()
                        elif event.opcode == Opcode.BINARY:
                            media.put_nowait(event.data)
                protocol.data_to_send()
        finally:
            writers.discard(writer)
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionResetError:
                pass

    def accept(reader, writer):
        task = asyncio.create_task(handle(reader, writer))
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    server = await asyncio.start_server(accept, "::1", 0, ssl=context)
    try:
        yield server.sockets[0].getsockname()[1], close_received, disconnected, media
    finally:
        server.close()
        await server.wait_closed()
        for writer in tuple(writers):
            writer.transport.abort()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("writer_blocked", [False, True], ids=["closing", "blocked-writer"])
@pytest.mark.parametrize("cancel", [False, True], ids=["deadline", "caller-cancel"])
async def test_deepgram_unanswered_close_terminates_real_transport_within_owner_budget(
    certificates, monkeypatch: pytest.MonkeyPatch, created_http_clients,
    writer_blocked, cancel,
):
    from deepgram.listen.v1.socket_client import AsyncV1SocketClient

    root, _, contexts = certificates
    monkeypatch.setenv("SSL_CERT_FILE", str(root))
    connectors = []
    original_connect = network_clients.external_websocket_connect

    def connect(*args, **kwargs):
        connector = original_connect(*args, **kwargs)
        connectors.append(connector)
        return connector

    monkeypatch.setattr(network_clients, "external_websocket_connect", connect)
    async with unanswered_deepgram_peer(contexts["valid"]) as (
        port, close_received, disconnected, media,
    ):
        local_sdk_client(monkeypatch, f"wss://localhost:{port}")
        session = await asyncio.wait_for(
            DeepgramRealtimeSTTBackend(api_key="synthetic-key", language="en").open_session(), 2
        )
        protocol = connectors[0].connection
        writes = []
        closing = None
        try:
            await session.send_audio(b"first")
            assert await asyncio.wait_for(media.get(), 1) == b"first"
            if writer_blocked:
                original_send = AsyncV1SocketClient.send_media
                write_started = asyncio.Event()
                release_write = asyncio.Event()

                async def send_media(client, message):
                    write_started.set()
                    await release_write.wait()
                    await original_send(client, message)

                monkeypatch.setattr(AsyncV1SocketClient, "send_media", send_media)
                writes.append(asyncio.create_task(session._write_payload(b"blocked")))
                await asyncio.wait_for(write_started.wait(), 1)
                writes.append(asyncio.create_task(session._write_payload(b"pending")))
                await asyncio.sleep(0)
            started = time.monotonic()
            closing = asyncio.create_task(session.close())
            if cancel:
                if not writer_blocked:
                    await asyncio.wait_for(close_received.wait(), 1)
                await asyncio.sleep(0.05)
                closing.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(closing, 1)
            else:
                await asyncio.wait_for(closing, 5.5)
            elapsed = time.monotonic() - started
            assert elapsed < (1.0 if cancel else 5.5)
            if not cancel and not writer_blocked:
                assert elapsed >= 4.5
                assert close_received.is_set()
            await asyncio.wait_for(disconnected.wait(), 1)
            assert protocol.state is State.CLOSED
            assert protocol.transport.is_closing()
            assert protocol.connection_lost_waiter.done()
            assert protocol.keepalive_task is not None
            assert protocol.keepalive_task.done()
            assert all(client.is_closed for client in created_http_clients)
            assert session._run_task is None
            assert all(
                task.done() for task in
                (session._send_task, session._recv_task, session._keepalive_task)
            )
            if writes:
                results = await asyncio.wait_for(
                    asyncio.gather(*writes, return_exceptions=True), 1
                )
                assert all(isinstance(result, RuntimeError) for result in results)
                assert media.empty()
            print(json.dumps({
                "mode": "blocked-writer" if writer_blocked else "closing",
                "cancelled": cancel, "elapsed_s": elapsed,
                "peer_disconnected": disconnected.is_set(),
                "protocol": protocol.state.name,
                "websocket_keepalive_done": protocol.keepalive_task.done(),
                "http_clients_closed": all(client.is_closed for client in created_http_clients),
            }))
        finally:
            if closing is not None and not closing.done():
                closing.cancel()
                await asyncio.gather(closing, return_exceptions=True)
            await session.close()
            if writes:
                await asyncio.gather(*writes, return_exceptions=True)
