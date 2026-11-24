from __future__ import annotations

import asyncio
import json
import logging
import socket
import sys
import traceback
from contextlib import aclosing, asynccontextmanager

import pytest

from puripuly_heart.app.adapters.agent_relay_transport import AgentRelayTransport
from puripuly_heart.app.ports.agent_chat import AgentTransportError
from puripuly_heart.app.services.agent_chat.owner import AgentChatOwner
from puripuly_heart.core import external_network
from puripuly_heart.core.diagnostics import DIAGNOSTIC_RECORD_ATTRIBUTE, serialize_diagnostics
from puripuly_heart.core.network_clients import first_party_user_agent
from tests.app.test_agent_chat import Records, Tokens, answer
from tests.app.test_agent_relay_recovery import token_owner, token_response
from tests.core.test_external_network import certificates as certificates
from tests.core.test_external_network import (
    isolated_network_environment as isolated_network_environment,
)
from tests.core.test_external_network import recording_proxy

PRIVATE = "PRIVATE-AGENT-NETWORK-PROBE"
TURN = dict(input=[], turn_id="a" * 32, effort="low", route="social", allow_tools=False)


def auth_at(origin: str):
    return token_owner(None, base_url=origin, allow_loopback_http=origin.startswith("http:"))


def relay_at(origin: str, auth):
    return AgentRelayTransport(
        origin + "/v1/respond", auth, allow_loopback_http=origin.startswith("http:")
    )


def sse(event: dict) -> bytes:
    return f"data: {json.dumps(event)}\n\n".encode()


def reply(writer, body: bytes, content_type: str, *, status=200, size=None, headers=""):
    writer.write(
        f"HTTP/1.1 {status} Test\r\nContent-Type: {content_type}\r\n"
        f"Content-Length: {len(body) if size is None else size}\r\n{headers}\r\n".encode()
        + body
    )


@asynccontextmanager
async def service(respond, *, tls=None):
    requests = []
    disconnected = []
    tasks = []

    async def handle(reader, writer):
        closed = asyncio.Event()
        disconnected.append(closed)
        try:
            while True:
                raw = await reader.readuntil(b"\r\n\r\n")
                request, *headers = raw.decode("ascii").strip().split("\r\n")
                method, path, _ = request.split(" ")
                assert method == "POST"
                headers = {name.lower(): value.lstrip() for name, value in (h.split(":", 1) for h in headers)}
                body = await reader.readexactly(int(headers["content-length"]))
                requests.append((path, headers, body))
                if respond(path, writer) is False:
                    return
                await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionError:
                pass
            finally:
                closed.set()

    def accept(reader, writer):
        tasks.append(asyncio.create_task(handle(reader, writer)))

    server = await asyncio.start_server(accept, "127.0.0.1", 0, ssl=tls)
    servers = [server]
    port = server.sockets[0].getsockname()[1]
    try:
        if tls is not None and socket.has_ipv6:
            servers.append(await asyncio.start_server(accept, "::1", port, ssl=tls))
        yield f"https://localhost:{port}" if tls is not None else f"http://127.0.0.1:{port}", requests, disconnected
    finally:
        for listener in servers:
            listener.close()
            listener.abort_clients()
        async with asyncio.timeout(3):
            await asyncio.gather(*tasks)
            for listener in servers:
                await listener.wait_closed()


@pytest.mark.parametrize("route", ["env", "system", "bypass", "direct"])
async def test_agent_clients_send_through_shared_tls_and_proxy_policy_and_close(
    certificates, monkeypatch, caplog, route
):
    root, _, contexts = certificates
    monkeypatch.setenv("SSL_CERT_FILE", str(root))

    def respond(path, writer):
        if path == "/v1/puri/token":
            reply(writer, token_response(PRIVATE).content, "application/json")
        elif path == "/v1/classify":
            reply(writer, b'{"answers":[]}', "application/json")
        else:
            assert path == "/v1/respond"
            reply(writer, sse({"type": "completed", "response": answer()}), "text/event-stream")

    async with service(respond, tls=contexts["valid"]) as (origin, requests, disconnected):
        async with recording_proxy() as (proxy, forwarded):
            match route:
                case "env":
                    monkeypatch.setenv("HTTPS_PROXY", proxy)
                case "system":
                    monkeypatch.setattr(external_network, "_windows_proxy_settings", lambda: ({"https": proxy}, ""))
                case "bypass":
                    monkeypatch.setenv("HTTPS_PROXY", proxy)
                    monkeypatch.setenv("NO_PROXY", "localhost")
            auth = auth_at(origin)
            relay = relay_at(origin, auth)
            caplog.set_level(logging.DEBUG)
            try:
                assert await auth.get_token() == PRIVATE
                assert await relay.classify(input=[], turn_id=TURN["turn_id"]) == {"answers": []}
                assert [event["type"] async for event in relay.respond(**TURN)] == ["completed"]
                assert [path for path, _, _ in requests] == ["/v1/puri/token", "/v1/classify", "/v1/respond"]
                assert all(headers["user-agent"] == first_party_user_agent() for _, headers, _ in requests)
                assert all(headers["authorization"] == f"Bearer {PRIVATE}" for _, headers, _ in requests[1:])
                proof = json.loads(requests[0][2])
                assert proof["signature_alg"] == "ed25519"
                assert len(disconnected) == 2 and not any(event.is_set() for event in disconnected)
                assert forwarded == (
                    [("CONNECT", origin.removeprefix("https://"))] * 2
                    if route in {"env", "system"}
                    else []
                )
            finally:
                await relay.close()
            async with asyncio.timeout(3):
                await asyncio.gather(*(event.wait() for event in disconnected))
    assert PRIVATE not in caplog.text and proof["signature"] not in caplog.text
    assert not any(record.name.startswith(("httpx", "httpcore")) for record in caplog.records)


@pytest.mark.parametrize("operation", ["broker_token", "relay_classify"])
async def test_agent_tls_rejects_untrusted_certificate_without_dispatch(certificates, operation):
    _, _, contexts = certificates

    def unexpected(path, writer):
        pytest.fail(f"A request reached the untrusted origin: {path}")

    async with service(unexpected, tls=contexts["valid"]) as (origin, requests, disconnected):
        auth = auth_at(origin)
        relay = relay_at(origin, Tokens())
        try:
            with pytest.raises(AgentTransportError, match="tls_failed") as caught:
                if operation == "broker_token":
                    await auth.get_token()
                else:
                    await relay.classify(input=[], turn_id=TURN["turn_id"])
            fields = caught.value.evidence.fields
            assert fields["agent_endpoint"] == operation
            assert fields["network_error_kind"] == "tls"
            assert fields["proxy_source"] == "direct"
            assert fields["stage"] == "request_send"
            assert (fields.get("tls_source", "unknown"), fields["tls_backend"]) == (
                ("windows", "native") if sys.platform == "win32" else ("unknown", "openssl")
            )
            assert caught.value.__cause__ is None and caught.value.__context__ is None
        finally:
            await relay.close()
            await auth.close()
        assert requests == disconnected == []


@pytest.mark.parametrize("operation", ["broker_token", "relay_classify", "relay_response"])
async def test_agent_response_read_failure_is_sanitized_and_never_replayed(operation):
    def respond(path, writer):
        content_type = "application/json"
        body = PRIVATE.encode()
        if operation == "relay_response":
            content_type = "text/event-stream"
            body = sse({"type": "text_delta", "text": PRIVATE})
        reply(writer, body, content_type, size=len(body) + 100)
        return False

    async with service(respond) as (origin, requests, disconnected):
        auth = auth_at(origin)
        relay = relay_at(origin, Tokens())
        try:
            with pytest.raises(AgentTransportError) as caught:
                if operation == "broker_token":
                    await auth.get_token()
                elif operation == "relay_classify":
                    await relay.classify(input=[], turn_id=TURN["turn_id"])
                else:
                    async with aclosing(relay.respond(**TURN)) as stream:
                        assert await anext(stream) == {"type": "text_delta", "text": PRIVATE}
                        await anext(stream)
            assert caught.value.code == ("stream_interrupted" if operation == "relay_response" else "relay_unavailable")
            assert caught.value.evidence.status_code == 200
            fields = caught.value.evidence.fields
            assert fields["stage"] == "response_body"
            assert fields["dispatch_certainty"] == "response_received"
            assert fields["network_error_kind"] == "network"
            assert fields["agent_endpoint"] == operation
            assert caught.value.__cause__ is None and caught.value.__context__ is None
            assert PRIVATE not in "".join(traceback.format_exception(caught.value))
            assert len(requests) == len(disconnected) == 1
        finally:
            await relay.close()
            await auth.close()


async def test_agent_closes_held_token_read_and_cancelled_stream_without_replay():
    sent_token_headers = asyncio.Event()
    paths = []

    def respond(path, writer):
        paths.append(path)
        if path == "/v1/puri/token":
            reply(writer, b"{", "application/json", size=100)
            sent_token_headers.set()
        else:
            body = sse({"type": "text_delta", "text": "Synthetic partial."})
            if paths.count("/v1/respond") == 1:
                reply(writer, body, "text/event-stream", size=len(body) + 100)
            else:
                reply(writer, sse({"type": "completed", "response": answer()}), "text/event-stream")

    async with service(respond) as (origin, _, disconnected):
        auth = auth_at(origin)
        relay = relay_at(origin, Tokens())
        pending = asyncio.create_task(auth.get_token())
        try:
            async with asyncio.timeout(5):
                await sent_token_headers.wait()
                await auth.close()
                with pytest.raises(asyncio.CancelledError):
                    await pending
                await disconnected[0].wait()
                async with aclosing(relay.respond(**TURN)) as stream:
                    assert (await anext(stream))["type"] == "text_delta"
                await disconnected[1].wait()
                assert [event["type"] async for event in relay.respond(**{**TURN, "turn_id": "b" * 32})] == ["completed"]
                assert paths == ["/v1/puri/token", "/v1/respond", "/v1/respond"]
        finally:
            await auth.close()
            await asyncio.gather(pending, return_exceptions=True)
            await relay.close()


async def test_real_proxy_denial_reaches_app_diagnostics_without_proxy_content(monkeypatch, caplog):
    headers = []
    denied = asyncio.Event()

    async def refuse(reader, writer):
        try:
            headers.append(await reader.readuntil(b"\r\n\r\n"))
            writer.write(f"HTTP/1.1 407 {PRIVATE}\r\nContent-Length: 0\r\n\r\n".encode())
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            denied.set()

    proxy = await asyncio.start_server(refuse, "127.0.0.1", 0)
    port = proxy.sockets[0].getsockname()[1]
    monkeypatch.setenv("HTTPS_PROXY", f"http://synthetic:{PRIVATE}@127.0.0.1:{port}")
    caplog.set_level(logging.DEBUG)
    owner = AgentChatOwner(Records(), relay_at("https://localhost", auth_at("https://localhost")), localize=str)
    try:
        await owner.start()
        async with asyncio.timeout(5):
            result = await owner.send("Synthetic app probe")
            await denied.wait()
        assert result.status == "failed" and result.text == "agent_chat.proxy_failed"
        assert len(headers) == 1 and b"Proxy-Authorization: Basic " in headers[0]
        evidence = serialize_diagnostics(result.diagnostics)
        assert evidence["category"] == "network"
        assert evidence["fields"]["network_error_kind"] == "proxy"
        assert evidence["fields"]["proxy_source"] == "env"
        assert evidence["fields"]["agent_endpoint"] == "broker_token"
        (record,) = [r for r in caplog.records if hasattr(r, DIAGNOSTIC_RECORD_ATTRIBUTE)]
        assert getattr(record, DIAGNOSTIC_RECORD_ATTRIBUTE) == (result.diagnostics, "error", "failed", None, None)
        assert PRIVATE not in caplog.text and PRIVATE not in json.dumps(evidence)
        assert not any(r.name.startswith(("httpx", "httpcore")) for r in caplog.records)
    finally:
        await owner.close()
        proxy.close()
        proxy.abort_clients()
        await proxy.wait_closed()


@pytest.mark.parametrize("endpoint", ["broker_token", "relay_response"])
@pytest.mark.parametrize(
    ("status", "media_type", "body", "headers", "shape", "suffix"),
    [
        (
            503,
            "application/json",
            b"",
            "CF-Mitigated: challenge\r\nCF-Ray: 0123456789abcdef-ICN\r\n",
            "edge_challenge",
            "edge_challenge",
        ),
        (
            401,
            "application/json",
            b"",
            "CF-Mitigated: challenge\r\nCF-Ray: 0123456789abcdef-ICN\r\n",
            "edge_challenge",
            "edge_challenge",
        ),
        (
            401,
            "text/plain",
            f"<!doctype html><html>{PRIVATE}</html>".encode(),
            f"CF-Ray: {PRIVATE}\r\n",
            "html",
            "unexpected_response",
        ),
        (
            403,
            "application/json",
            json.dumps({"error": {"code": "service_authentication_failed", "message": PRIVATE}}).encode(),
            "CF-Ray: 0123456789abcdef-ICN\r\n",
            "public_error_envelope",
            "unexpected_response",
        ),
        (307, "text/plain", b"", "Location: /unexpected-redirect\r\n", None, "unexpected_response"),
    ],
)
async def test_unknown_rejection_and_edge_evidence_reach_app_without_reset_or_replay(
    caplog, endpoint, status, media_type, body, headers, shape, suffix
):
    rejected = False
    target = "/v1/puri/token" if endpoint == "broker_token" else "/v1/respond"

    def respond(path, writer):
        nonlocal rejected
        if path == target and not rejected:
            rejected = True
            reply(
                writer, body, media_type, status=status, headers=headers,
                size=100 if shape == "edge_challenge" else None,
            )
        elif path == "/v1/puri/token":
            reply(writer, token_response(PRIVATE).content, "application/json")
        elif path == "/v1/classify":
            reply(writer, b'{"answers":[]}', "application/json")
        else:
            reply(writer, sse({"type": "completed", "response": answer()}), "text/event-stream")

    async with service(respond) as (origin, requests, disconnected):
        owner = AgentChatOwner(Records(), relay_at(origin, auth_at(origin)), localize=str)
        await owner.start()
        try:
            caplog.set_level(logging.DEBUG)
            async with asyncio.timeout(5):
                failed = await owner.send(f"Synthetic prompt containing {PRIVATE}")
                assert failed.status == "failed" and failed.text == f"agent_chat.{suffix}"
                observed = [path for path, _, _ in requests]
                assert observed == (
                    [target]
                    if endpoint == "broker_token"
                    else ["/v1/puri/token", "/v1/classify", target]
                )
                if shape == "edge_challenge":
                    await disconnected[-1].wait()
                assert (await owner.send("A new synthetic prompt")).status == "completed"
            evidence = serialize_diagnostics(failed.diagnostics)
            assert evidence["category"] == "invalid_response"
            assert evidence["status_code"] == status
            assert evidence["fields"].get("response_shape") == shape
            assert evidence["fields"]["agent_endpoint"] == endpoint
            assert evidence["fields"].get("cf_ray") == (
                "0123456789abcdef-ICN" if shape in {"edge_challenge", "public_error_envelope"} else None
            )
            assert [path for path, _, _ in requests[len(observed):]] == (
                ["/v1/puri/token", "/v1/classify", "/v1/respond"]
                if endpoint == "broker_token"
                else ["/v1/classify", "/v1/respond"]
            )
            assert PRIVATE not in caplog.text and PRIVATE not in json.dumps(evidence)
        finally:
            await owner.close()


async def test_classify_transport_failure_records_degraded_cause_and_preserves_fallback(caplog):
    def respond(path, writer):
        if path == "/v1/classify":
            reply(writer, f"<html>{PRIVATE}</html>".encode(), "text/html", status=429)
        else:
            assert path == "/v1/respond"
            reply(writer, sse({"type": "completed", "response": answer()}), "text/event-stream")

    async with service(respond) as (origin, requests, _):
        owner = AgentChatOwner(Records(), relay_at(origin, Tokens()), localize=str)
        await owner.start()
        try:
            assert (await owner.send("Synthetic fallback")).status == "completed"
            assert [path for path, _, _ in requests] == ["/v1/classify", "/v1/respond"]
            (record,) = [r for r in caplog.records if hasattr(r, DIAGNOSTIC_RECORD_ATTRIBUTE)]
            diagnostics, severity, disposition, _, exception = getattr(
                record, DIAGNOSTIC_RECORD_ATTRIBUTE
            )
            assert (severity, disposition, exception) == ("warning", "degraded", None)
            evidence = serialize_diagnostics(diagnostics)
            assert evidence["status_code"] == 429 and evidence["fields"]["response_shape"] == "html"
            assert PRIVATE not in caplog.text and PRIVATE not in json.dumps(evidence)
        finally:
            await owner.close()
