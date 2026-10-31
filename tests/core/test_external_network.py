from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import os
import socket
import ssl
import threading
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager
from datetime import datetime, timedelta, timezone
from functools import cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import httpx
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from websockets.asyncio.server import serve

from puripuly_heart.core import external_network, network_clients
from puripuly_heart.core.external_network import ProxyPolicy, select_tls
from puripuly_heart.core.network_requests import ExternalRequestsSession

TEST_AUTHORIZATION = "Bearer synthetic-network-test-key"


@pytest.fixture(autouse=True)
def isolated_network_environment(monkeypatch):
    for name in tuple(os.environ):
        if name.lower().endswith("_proxy") or name in (
            "SSL_CERT_FILE",
            "SSL_CERT_DIR",
            "REQUESTS_CA_BUNDLE",
            "CURL_CA_BUNDLE",
        ):
            monkeypatch.delenv(name)
    monkeypatch.setattr(external_network, "_windows_proxy_settings", lambda: ({}, ""))


@cache
def _certificate_material():
    now = datetime.now(timezone.utc)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "network test root")])
    root = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=2))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(True, False, False, False, False, True, True, False, False), critical=True
        )
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .sign(key, hashes.SHA256())
    )
    leaf_material = []
    for variant in ("valid", "expired"):
        leaf_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        expiry = now - timedelta(hours=1) if variant == "expired" else now + timedelta(days=1)
        leaf = (
            x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")]))
            .issuer_name(name)
            .public_key(leaf_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(days=1))
            .not_valid_after(expiry)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost")]), critical=False)
            .add_extension(
                x509.AuthorityKeyIdentifier.from_issuer_public_key(key.public_key()), critical=False
            )
            .sign(key, hashes.SHA256())
        )
        leaf_material.append(
            (
                variant,
                leaf.public_bytes(serialization.Encoding.PEM),
                leaf_key.private_bytes(
                    serialization.Encoding.PEM,
                    serialization.PrivateFormat.PKCS8,
                    serialization.NoEncryption(),
                ),
            )
        )
    canonical_name = name.public_bytes()[2:]
    ca_hash = int.from_bytes(hashlib.sha1(canonical_name).digest()[:4], "little")
    return root.public_bytes(serialization.Encoding.PEM), f"{ca_hash:08x}.0", tuple(leaf_material)


@pytest.fixture
def certificates(tmp_path):
    root_pem, ca_filename, leaf_material = _certificate_material()
    root_path = tmp_path / "root.pem"
    root_path.write_bytes(root_pem)
    contexts = {}
    for variant, cert_pem, key_pem in leaf_material:
        cert_path, key_path = tmp_path / f"{variant}.pem", tmp_path / f"{variant}.key"
        cert_path.write_bytes(cert_pem)
        key_path.write_bytes(key_pem)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert_path, key_path)
        contexts[variant] = context
    ca_dir = tmp_path / "ca"
    ca_dir.mkdir()
    (ca_dir / ca_filename).write_bytes(root_pem)
    return root_path, ca_dir, contexts


@contextmanager
def http_server(context=None, *, ipv6=None, requests=None, handler=None):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if requests is not None:
                requests.append(self.path)
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *args):
            pass

    handler = handler or Handler
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    servers = [server]
    if ipv6 is True or (ipv6 is None and socket.has_ipv6):

        class IPv6Server(ThreadingHTTPServer):
            address_family = socket.AF_INET6

        servers.append(IPv6Server(("::1", server.server_address[1]), handler))
    threads = []
    for current in servers:
        if context is not None:
            current.socket = context.wrap_socket(current.socket, server_side=True)
        thread = threading.Thread(
            target=current.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        thread.start()
        threads.append(thread)
    try:
        yield server.server_address[1]
    finally:
        for current in servers:
            current.shutdown()
            current.server_close()
        for thread in threads:
            thread.join()


@asynccontextmanager
async def loopback_websocket_server(handler, **options):
    async with AsyncExitStack() as stack:
        server = await stack.enter_async_context(serve(handler, "127.0.0.1", 0, **options))
        if socket.has_ipv6:
            port = server.sockets[0].getsockname()[1]
            await stack.enter_async_context(serve(handler, "::1", port, **options))
        yield server


@asynccontextmanager
async def recording_proxy(context=None, *, socks=False):
    requests = []
    tasks = set()

    async def pipe(reader, writer):
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()

    async def handle(reader, writer):
        remote = None
        forwarding = ()
        try:
            if socks:
                version, count = await reader.readexactly(2)
                assert version == 5
                assert 0 in await reader.readexactly(count)
                writer.write(b"\x05\x00")
                await writer.drain()
                version, command, _, kind = await reader.readexactly(4)
                assert (version, command) == (5, 1)
                if kind == 3:
                    length = (await reader.readexactly(1))[0]
                    host = (await reader.readexactly(length)).decode()
                else:
                    host = str(
                        ipaddress.ip_address(await reader.readexactly(4 if kind == 1 else 16))
                    )
                port = int.from_bytes(await reader.readexactly(2), "big")
                requests.append(("SOCKS5", f"{host}:{port}"))
                upstream, remote = await asyncio.open_connection(host, port)
                writer.write(b"\x05\x00\x00\x01\x7f\x00\x00\x01\x00\x00")
                await writer.drain()
            else:
                header = await reader.readuntil(b"\r\n\r\n")
                method, target, _ = header.split(b"\r\n", 1)[0].decode().split(" ", 2)
                requests.append((method, target))
                if method == "CONNECT":
                    host, port = target.rsplit(":", 1)
                    upstream, remote = await asyncio.open_connection(host, int(port))
                    writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                    await writer.drain()
                else:
                    parsed = urlsplit(target)
                    upstream, remote = await asyncio.open_connection(
                        parsed.hostname, parsed.port or 80
                    )
                    first_line, remaining = header.split(b"\r\n", 1)
                    path = parsed.path or "/"
                    if parsed.query:
                        path += "?" + parsed.query
                    remote.write(f"{method} {path} HTTP/1.1\r\n".encode() + remaining)
                    await remote.drain()
            forwarding = (
                asyncio.create_task(pipe(reader, remote)),
                asyncio.create_task(pipe(upstream, writer)),
            )
            await asyncio.wait(forwarding, return_when=asyncio.FIRST_COMPLETED)
        except ConnectionError, asyncio.IncompleteReadError:
            pass
        finally:
            for task in forwarding:
                task.cancel()
            await asyncio.gather(*forwarding, return_exceptions=True)
            streams = (writer,) if remote is None else (writer, remote)
            for stream in streams:
                stream.close()
            await asyncio.gather(
                *(stream.wait_closed() for stream in streams), return_exceptions=True
            )

    def accept(reader, writer):
        task = asyncio.create_task(handle(reader, writer))
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    server = await asyncio.start_server(accept, "127.0.0.1", 0, ssl=context)
    servers = [server]
    try:
        if socket.has_ipv6:
            servers.append(
                await asyncio.start_server(
                    accept, "::1", server.sockets[0].getsockname()[1], ssl=context
                )
            )
        scheme = "socks5" if socks else "https" if context is not None else "http"
        yield f"{scheme}://localhost:{server.sockets[0].getsockname()[1]}", requests
    finally:
        for listener in servers:
            listener.close()
            listener.abort_clients()
        owned = tuple(tasks)
        for task in owned:
            task.cancel()
        await asyncio.gather(*owned, return_exceptions=True)
        for listener in servers:
            await listener.wait_closed()


async def test_proxy_teardown_closes_live_tunnel_and_drains_owned_tasks():
    upstream_received = []
    upstream_eof = []
    upstream_closed = asyncio.Event()

    async def upstream(reader, writer):
        try:
            data = await reader.readline()
            upstream_received.append(data)
            writer.write(b"echo:" + data)
            await writer.drain()
            upstream_eof.append(await reader.read())
        finally:
            writer.close()
            await writer.wait_closed()
            upstream_closed.set()

    server = await asyncio.start_server(upstream, "127.0.0.1", 0)
    target = f"127.0.0.1:{server.sockets[0].getsockname()[1]}"
    before = asyncio.all_tasks()
    writer = None
    try:
        async with asyncio.timeout(3):
            async with recording_proxy() as (proxy, recorded):
                endpoint = urlsplit(proxy)
                reader, writer = await asyncio.open_connection(endpoint.hostname, endpoint.port)
                writer.write(f"CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n\r\n".encode())
                await writer.drain()
                assert await reader.readuntil(b"\r\n\r\n") == (
                    b"HTTP/1.1 200 Connection Established\r\n\r\n"
                )
                writer.write(b"controlled-tunnel-progress\n")
                await writer.drain()
                assert await reader.readline() == b"echo:controlled-tunnel-progress\n"
                assert upstream_received == [b"controlled-tunnel-progress\n"]
                assert recorded == [("CONNECT", target)]
                admitted = {
                    task
                    for task in asyncio.all_tasks() - before
                    if task.get_coro().__qualname__.startswith(f"{recording_proxy.__name__}.")
                }
                assert admitted
            assert await reader.read() == b""
            await upstream_closed.wait()
            assert upstream_eof == [b""]
            assert all(task.done() for task in admitted), [
                (task.get_coro().__qualname__, task.cancelling())
                for task in admitted
                if not task.done()
            ]
    finally:
        if writer is not None:
            writer.close()
            await writer.wait_closed()
        server.close()
        server.close_clients()
        remaining = asyncio.all_tasks() - before
        for task in remaining:
            task.cancel()
        await asyncio.gather(*remaining, return_exceptions=True)
        await server.wait_closed()


def test_proxy_selection_preserves_system_with_no_proxy_only():
    policy = ProxyPolicy(
        environment={"NO_PROXY": ".example.com"},
        system={"https": "http://proxy.invalid:8080"},
        system_bypass="<local>;*.intranet",
    )
    for scheme in ("https", "wss"):
        assert policy.route(f"{scheme}://outside.invalid").source == "system"
        assert policy.route(f"{scheme}://example.com").source == "bypass"
        assert policy.route(f"{scheme}://host.intranet").source == "bypass"
        assert policy.route(f"{scheme}://printer").source == "bypass"


def test_proxy_override_and_common_bypass():
    policy = ProxyPolicy(
        environment={
            "HTTPS_PROXY": "http://https.invalid",
            "ALL_PROXY": "socks5://all.invalid",
            "WSS_PROXY": "http://wss.invalid",
            "NO_PROXY": "localhost:444,.example.com",
        },
        system={},
    )
    assert policy.route("https://host.invalid").url == "http://https.invalid"
    assert policy.route("wss://host.invalid").url == "http://wss.invalid"
    for scheme in ("https", "wss"):
        assert policy.route(f"{scheme}://localhost:444").source == "bypass"
        assert policy.route(f"{scheme}://example.com").source == "bypass"
        assert policy.route(f"{scheme}://localhost:445").source == "env"
    assert (
        ProxyPolicy(environment={"ALL_PROXY": "socks5://all.invalid"}, system={})
        .route("wss://host.invalid")
        .url
        == "socks5://all.invalid"
    )


def test_proxy_policy_refreshes_environment_and_preserves_explicit_overrides(monkeypatch):
    system = {"https": "http://system.invalid:8080"}
    bypass = ""
    monkeypatch.setattr(external_network, "_windows_proxy_settings", lambda: (system, bypass))
    live = ProxyPolicy()
    snapshot = live.snapshot()
    fixed = ProxyPolicy(environment={}, system=system, system_bypass="")
    explicit_bypass = ProxyPolicy(environment={}, system_bypass="fixed.example")
    assert live.route("https://api.example").url == "http://system.invalid:8080"

    monkeypatch.setenv("HTTPS_PROXY", "env.invalid:9090")
    assert live.route("https://api.example").url == "http://env.invalid:9090"
    monkeypatch.setenv("NO_PROXY", "api.example")
    assert live.route("https://api.example").source == "bypass"
    monkeypatch.delenv("HTTPS_PROXY")
    monkeypatch.delenv("NO_PROXY")
    system["https"] = "http://new-system.invalid:8080"
    bypass = "api.example"
    assert live.route("https://api.example").source == "bypass"
    assert fixed.route("https://api.example").url == "http://system.invalid:8080"
    assert snapshot.route("https://api.example").url == "http://system.invalid:8080"
    assert explicit_bypass.route("https://api.example").url == "http://new-system.invalid:8080"
    assert explicit_bypass.route("https://fixed.example").source == "bypass"


@pytest.mark.parametrize("kind", ["async", "sync", "requests", "genai"])
@pytest.mark.parametrize("secure", [False, True])
async def test_existing_clients_follow_system_proxy_changes(
    certificates, monkeypatch, kind, secure
):
    root, _, contexts = certificates
    monkeypatch.setenv("SSL_CERT_FILE", str(root))
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(root))
    system = {}
    bypass = ""
    monkeypatch.setattr(external_network, "_windows_proxy_settings", lambda: (system, bypass))
    with http_server(contexts["valid"] if secure else None) as port:
        async with (
            recording_proxy() as (first, first_requests),
            recording_proxy() as (
                second,
                second_requests,
            ),
        ):
            system.update(http=first, https=first)
            sync_client = None
            if kind == "async":
                client = network_clients.external_async_client(timeout=2)
            elif kind == "sync":
                client = network_clients.external_client(timeout=2)
            elif kind == "requests":
                client = ExternalRequestsSession()
            else:
                options = network_clients.genai_http_options()
                client = options["httpx_async_client"]
                sync_client = options["httpx_client"]
            url = f"{'https' if secure else 'http'}://localhost:{port}/"

            async def request():
                if kind in ("async", "genai"):
                    response = await client.get(url, timeout=2)
                else:
                    response = await asyncio.to_thread(client.get, url, timeout=2)
                assert response.status_code == 200
                assert response.text == "ok"

            try:
                await request()
                assert len(first_requests) == 1
                system.update(http=second, https=second)
                await request()
                assert len(first_requests) == 1
                assert len(second_requests) == 1
                system.clear()
                await request()
                system.update(http=first, https=first)
                bypass = "localhost"
                await request()
                assert len(first_requests) == len(second_requests) == 1
                bypass = ""
                await request()
                assert len(first_requests) == 2
                assert len(second_requests) == 1
            finally:
                if kind in ("async", "genai"):
                    await client.aclose()
                else:
                    await asyncio.to_thread(client.close)
                if sync_client is not None:
                    await asyncio.to_thread(sync_client.close)


async def test_proxy_change_preserves_inflight_response(monkeypatch):
    finish = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            if self.path == "/slow":
                finish.wait(5)
            self.wfile.write(b"ok")

        def log_message(self, *args):
            pass

    system = {}
    monkeypatch.setattr(external_network, "_windows_proxy_settings", lambda: (system, ""))
    with http_server(handler=Handler) as port:
        async with (
            recording_proxy() as (first, first_requests),
            recording_proxy() as (
                second,
                second_requests,
            ),
        ):
            system["http"] = first
            async with network_clients.external_async_client(timeout=2) as client:
                try:
                    async with client.stream("GET", f"http://localhost:{port}/slow") as slow:
                        system["http"] = second
                        assert (await client.get(f"http://localhost:{port}/fast")).text == "ok"
                        finish.set()
                        assert await slow.aread() == b"ok"
                    assert len(first_requests) == len(second_requests) == 1
                finally:
                    finish.set()


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_client_close_releases_keepalive_connections_to_previous_proxies(
    monkeypatch, asynchronous
):
    received = []
    received_eof = {}
    eof = {path: asyncio.Event() for path in ("/first", "/second")}
    tasks = []

    async def handle(reader, writer):
        try:
            header = await reader.readuntil(b"\r\n\r\n")
            method, path, _ = header.split(b"\r\n", 1)[0].decode().split(" ", 2)
            received.append((method, path))
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: keep-alive\r\n\r\nok"
            )
            await writer.drain()
            received_eof[path] = await reader.read()
            eof[path].set()
        finally:
            writer.close()
            await writer.wait_closed()

    def accept(reader, writer):
        tasks.append(asyncio.create_task(handle(reader, writer)))

    server = await asyncio.start_server(accept, "127.0.0.1", 0)
    url = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
    try:
        async with (
            recording_proxy() as (first, first_requests),
            recording_proxy() as (second, second_requests),
        ):
            if asynchronous:
                client = network_clients.external_async_client(timeout=2)
            else:
                client = network_clients.external_client(timeout=2)

            def consume_response(target):
                with client.stream("GET", target) as response:
                    assert response.status_code == 200
                    assert response.read() == b"ok"

            try:
                for path, proxy in (("/first", first), ("/second", second)):
                    monkeypatch.setenv("HTTP_PROXY", proxy)
                    if asynchronous:
                        async with client.stream("GET", url + path) as response:
                            assert response.status_code == 200
                            assert await response.aread() == b"ok"
                    else:
                        await asyncio.to_thread(consume_response, url + path)
                assert received == [("GET", "/first"), ("GET", "/second")]
                assert first_requests == [("GET", url + "/first")]
                assert second_requests == [("GET", url + "/second")]
                assert not any(event.is_set() for event in eof.values())
            finally:
                if asynchronous:
                    await client.aclose()
                else:
                    await asyncio.to_thread(client.close)
            async with asyncio.timeout(3):
                await asyncio.gather(*(event.wait() for event in eof.values()))
            assert received_eof == {"/first": b"", "/second": b""}
    finally:
        server.close()
        server.close_clients()
        async with asyncio.timeout(3):
            await asyncio.gather(*tasks)
            await server.wait_closed()


@pytest.mark.parametrize("directory", [False, True])
async def test_explicit_ca_https_and_wss(certificates, monkeypatch, directory):
    root, ca_dir, contexts = certificates
    monkeypatch.setenv(
        "SSL_CERT_DIR" if directory else "SSL_CERT_FILE", str(ca_dir if directory else root)
    )
    received_authorization = []

    async def echo(ws):
        received_authorization.append(ws.request.headers.get_all("Authorization"))
        async for message in ws:
            await ws.send(message)

    with http_server(contexts["valid"]) as port:
        async with network_clients.external_async_client() as client:
            assert (await client.get(f"https://localhost:{port}")).text == "ok"
        assert await asyncio.to_thread(lambda: _sync_get(f"https://localhost:{port}")) == "ok"
    async with loopback_websocket_server(echo, ssl=contexts["valid"]) as server:
        async with network_clients.external_websocket_connect(
            f"wss://localhost:{server.sockets[0].getsockname()[1]}",
            additional_headers={"Authorization": TEST_AUTHORIZATION},
        ) as ws:
            await ws.send("verified")
            assert await ws.recv() == "verified"
    assert received_authorization == [[TEST_AUTHORIZATION]]


def _sync_get(url):
    with network_clients.external_client() as client:
        return client.get(url).text


@pytest.mark.parametrize("failure", ["untrusted", "expired", "wronghost"])
async def test_tls_failures_reject_and_annotate(certificates, monkeypatch, failure):
    root, _, contexts = certificates
    if failure != "untrusted":
        monkeypatch.setenv("SSL_CERT_FILE", str(root))
    host = "127.0.0.1" if failure == "wronghost" else "localhost"
    with http_server(contexts["expired" if failure == "expired" else "valid"]) as port:
        async with network_clients.external_async_client() as client:
            with pytest.raises(httpx.ConnectError) as caught:
                await client.get(f"https://{host}:{port}")
        assert caught.value.connection_diagnostics["transport"] == "https"
        assert caught.value.connection_diagnostics["proxy_source"] == "direct"
        assert set(caught.value.connection_diagnostics) == {
            "transport",
            "proxy_source",
            "tls_source",
            "tls_backend",
        }


@pytest.mark.parametrize(
    "name", ["SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"]
)
def test_invalid_explicit_ca_fails(tmp_path, name):
    with pytest.raises(OSError):
        select_tls(
            requests=name in ("REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"),
            environment={name: str(tmp_path / "missing")},
        )


@pytest.mark.parametrize("https_proxy", [False, True])
@pytest.mark.parametrize("proxy_variable", ["HTTPS_PROXY", "ALL_PROXY"])
async def test_https_wss_proxy_parity(certificates, monkeypatch, https_proxy, proxy_variable):
    root, _, contexts = certificates
    monkeypatch.setenv("SSL_CERT_FILE", str(root))
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(root))
    received_authorization = []

    async def echo(ws):
        received_authorization.append(ws.request.headers.get_all("Authorization"))
        async for message in ws:
            await ws.send(message)

    async with recording_proxy(contexts["valid"] if https_proxy else None) as (proxy, recorded):
        monkeypatch.setenv(proxy_variable, proxy)
        with http_server(contexts["valid"]) as port:
            async with network_clients.external_async_client() as client:
                assert (await client.get(f"https://localhost:{port}")).text == "ok"

            def request():
                with ExternalRequestsSession() as session:
                    return session.get(f"https://localhost:{port}").text

            assert await asyncio.to_thread(request) == "ok"
        async with loopback_websocket_server(echo, ssl=contexts["valid"]) as server:
            url = f"wss://localhost:{server.sockets[0].getsockname()[1]}"
            async with network_clients.external_websocket_connect(
                url, additional_headers={"Authorization": TEST_AUTHORIZATION}
            ) as ws:
                await ws.send("proxied")
                assert await ws.recv() == "proxied"
        assert received_authorization == [[TEST_AUTHORIZATION]]
        assert len(recorded) == 3
        assert all(method == "CONNECT" for method, _ in recorded)


async def test_no_proxy_bypasses_both_transports(certificates, monkeypatch):
    root, _, contexts = certificates
    monkeypatch.setenv("SSL_CERT_FILE", str(root))
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "localhost")
    received_authorization = []

    async def echo(ws):
        received_authorization.append(ws.request.headers.get_all("Authorization"))
        await ws.send("direct")

    with http_server(contexts["valid"]) as port:
        async with network_clients.external_async_client() as client:
            assert (await client.get(f"https://localhost:{port}")).status_code == 200
    async with loopback_websocket_server(echo, ssl=contexts["valid"]) as server:
        async with network_clients.external_websocket_connect(
            f"wss://localhost:{server.sockets[0].getsockname()[1]}",
            additional_headers={"Authorization": TEST_AUTHORIZATION},
        ) as ws:
            assert await ws.recv() == "direct"
    assert received_authorization == [[TEST_AUTHORIZATION]]


async def test_mock_transport_injection_does_not_resolve_invalid_environment(monkeypatch):
    monkeypatch.setenv("SSL_CERT_FILE", "missing-explicit-ca")
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text="injected"))
    async with network_clients.external_async_client(transport=transport) as client:
        assert (await client.get("https://example.invalid")).text == "injected"


async def test_socks5_http_and_async_websocket_routes(certificates, monkeypatch):
    root, _, contexts = certificates
    monkeypatch.setenv("SSL_CERT_FILE", str(root))
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(root))
    received_authorization = []

    async def echo(ws):
        received_authorization.append(ws.request.headers.get_all("Authorization"))
        async for message in ws:
            await ws.send(message)

    async with recording_proxy(socks=True) as (proxy, recorded):
        monkeypatch.setenv("ALL_PROXY", proxy)
        with http_server(contexts["valid"], ipv6=True) as port:
            url = f"https://localhost:{port}"
            async with network_clients.external_async_client() as client:
                assert (await client.get(url)).text == "ok"
            assert await asyncio.to_thread(_sync_get, url) == "ok"

            def request():
                with ExternalRequestsSession() as session:
                    return session.get(url).text

            assert await asyncio.to_thread(request) == "ok"
        async with loopback_websocket_server(echo, ssl=contexts["valid"]) as server:
            url = f"wss://localhost:{server.sockets[0].getsockname()[1]}"
            async with network_clients.external_websocket_connect(
                url, additional_headers={"Authorization": TEST_AUTHORIZATION}
            ) as ws:
                await ws.send("async")
                assert await ws.recv() == "async"
        assert received_authorization == [[TEST_AUTHORIZATION]]
        assert len(recorded) == 4
        assert all(method == "SOCKS5" for method, _ in recorded)


async def test_local_http_and_custom_realtime_remain_direct(monkeypatch):
    from puripuly_heart.core.stt.custom_connection import validate_custom_stt_connection
    from puripuly_heart.providers.stt.custom import CustomSTTBackend

    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "nonmatching.invalid")
    with http_server() as port:
        async with httpx.AsyncClient(trust_env=False) as client:
            assert (await client.get(f"http://127.0.0.1:{port}")).text == "ok"
    exchanges = []

    async def handler(ws):
        exchanges.append(json.loads(await ws.recv()))
        await ws.send(json.dumps({"type": "session.updated"}))
        await ws.wait_closed()

    async with serve(handler, "127.0.0.1", 0) as server:
        endpoint = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
        result = await validate_custom_stt_connection(
            mode="realtime", compatibility="openai_realtime", endpoint=endpoint, model="local"
        )
        assert result.status == "transcription_unverified"
        backend = CustomSTTBackend(
            mode="realtime", compatibility="openai_realtime", endpoint=endpoint, model="local"
        )
        session = await backend.open_session()
        await session.close()
    assert len(exchanges) == 2
    assert all(message["type"] == "session.update" for message in exchanges)


@pytest.mark.parametrize(
    "policy_case", ["direct_bypass_only", "explicit_ca", "proxy_bypass", "proxy_normalized"]
)
async def test_huggingface_child_selects_compatible_transport(
    certificates, monkeypatch, tmp_path, policy_case
):
    import sys

    from puripuly_heart.core.local_asr.local_stt_download_port import HuggingFaceDownloadRequest
    from puripuly_heart.core.local_asr.local_stt_huggingface_xet_adapter import (
        HuggingFaceXetDownloadAdapter,
    )

    root, _, _ = certificates
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    if policy_case == "explicit_ca":
        monkeypatch.setenv("SSL_CERT_FILE", str(root))
    elif policy_case == "direct_bypass_only":
        monkeypatch.setenv("NO_PROXY", "localhost")
    elif policy_case == "proxy_bypass":
        monkeypatch.setenv("ALL_PROXY", "http://proxy.invalid:8888")
        monkeypatch.setenv("NO_PROXY", ".example.com")
    else:
        monkeypatch.setenv("ALL_PROXY", "http://all.invalid:8888")
        monkeypatch.setenv("HTTPS_PROXY", "http://upper.invalid:8888")
        monkeypatch.setenv("https_proxy", "http://lower.invalid:8888")
    script = """
import json, os, pathlib, sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
target = pathlib.Path(payload['local_dir']) / payload['remote_path']
target.write_text(json.dumps({name: os.environ.get(name) for name in ('SSL_CERT_FILE', 'HF_HUB_DISABLE_XET', 'HTTP_PROXY', 'HTTPS_PROXY', 'NO_PROXY')}))
pathlib.Path(sys.argv[2]).write_text(json.dumps({'type': 'complete', 'path': str(target)}) + '\\n')
"""
    adapter = HuggingFaceXetDownloadAdapter(
        worker_command_factory=lambda request, request_path, event_path: [
            sys.executable,
            "-c",
            script,
            str(request_path),
            str(event_path),
        ]
    )
    path = await adapter.download(
        HuggingFaceDownloadRequest(
            repo_id="fixture/repo",
            revision="pinned",
            remote_path="result.json",
            local_dir=tmp_path / "child",
            expected_size_bytes=None,
        ),
        cancel_event=None,
        on_progress=None,
    )
    result = json.loads(path.read_text())
    assert result["HF_HUB_DISABLE_XET"] == (
        "1" if policy_case in ("explicit_ca", "proxy_bypass") else None
    )
    if policy_case == "explicit_ca":
        assert result["SSL_CERT_FILE"] == str(root)
    if policy_case == "proxy_normalized":
        assert result["HTTP_PROXY"] == "http://all.invalid:8888"
        assert result["HTTPS_PROXY"] == "http://lower.invalid:8888"
        assert result["NO_PROXY"] == ","


async def test_plain_http_uses_selected_https_proxy_trust(certificates, monkeypatch):
    root, _, contexts = certificates
    monkeypatch.setenv("SSL_CERT_FILE", str(root))
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(root))
    async with recording_proxy(contexts["valid"]) as (proxy, recorded):
        monkeypatch.setenv("HTTP_PROXY", proxy)
        with http_server() as port:
            url = f"http://localhost:{port}"
            async with network_clients.external_async_client() as client:
                assert (await client.get(url)).text == "ok"

            def request():
                with ExternalRequestsSession() as session:
                    return session.get(url).text

            assert await asyncio.to_thread(request) == "ok"
        assert len(recorded) == 2
        assert all(method == "GET" for method, _ in recorded)


async def test_https_proxy_wrong_hostname_is_rejected_without_direct_fallback(
    certificates, monkeypatch
):
    root, _, contexts = certificates
    monkeypatch.setenv("SSL_CERT_FILE", str(root))
    async with recording_proxy(contexts["valid"]) as (proxy, recorded):
        monkeypatch.setenv("HTTP_PROXY", proxy.replace("localhost", "127.0.0.1"))
        with http_server() as port:
            async with network_clients.external_async_client() as client:
                with pytest.raises(httpx.ConnectError) as caught:
                    await client.get(f"http://localhost:{port}")
        assert recorded == []
        assert caught.value.connection_diagnostics["tls_source"] == "explicit_file"
        assert caught.value.connection_diagnostics["proxy_source"] == "env"


@pytest.mark.parametrize(
    "proxy,expected",
    [
        ("socks4://private-user:private-password@127.0.0.1:9", "unsupported_proxy_configuration"),
        ("http://private-proxy:invalid-port", "invalid_proxy_configuration"),
        ("https://[private-malformed", "invalid_proxy_configuration"),
    ],
)
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_invalid_static_proxy_never_falls_back_and_correction_succeeds(
    monkeypatch, asynchronous, proxy, expected
):
    from puripuly_heart.core.external_network import NetworkConfigurationError
    from puripuly_heart.core.qq_auth_diagnostics import exception_fields, safe_fields

    recorded = []
    with http_server(requests=recorded) as port:
        url = f"http://127.0.0.1:{port}/corrected"
        monkeypatch.setattr(
            external_network,
            "_windows_proxy_settings",
            lambda: ({"http": proxy}, ""),
        )
        with pytest.raises(NetworkConfigurationError) as caught:
            if asynchronous:
                async with network_clients.external_async_client() as client:
                    await client.get(url)
            else:
                await asyncio.to_thread(_sync_get, url)
        fields = safe_fields(exception_fields(caught.value))
        assert fields["dispatch_certainty"] == "not_dispatched"
        assert fields["configuration_error"] == expected
        assert fields["proxy_source"] == "system"
        assert recorded == []
        assert "private-" not in repr(fields)
        monkeypatch.setattr(external_network, "_windows_proxy_settings", lambda: ({}, ""))
        if asynchronous:
            async with network_clients.external_async_client() as client:
                assert (await client.get(url)).text == "ok"
        else:
            assert await asyncio.to_thread(_sync_get, url) == "ok"
        assert recorded == ["/corrected"]


@pytest.mark.parametrize("secure", [False, True])
@pytest.mark.parametrize("body", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_header_and_body_stalls_keep_selected_route_and_tls(
    certificates, monkeypatch, secure, body, asynchronous
):
    from puripuly_heart.core.qq_auth_diagnostics import exception_fields, safe_fields

    root, _, contexts = certificates
    if secure:
        monkeypatch.setenv("SSL_CERT_FILE", str(root))
    received = []
    release = asyncio.Event()
    connections = set()

    async def respond(reader, writer):
        connections.add(asyncio.current_task())
        try:
            received.append(await reader.readuntil(b"\r\n\r\n"))
            if body:
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\nx")
                await writer.drain()
            await release.wait()
        finally:
            writer.close()
            await writer.wait_closed()
            connections.discard(asyncio.current_task())

    server = await asyncio.start_server(
        respond, "127.0.0.1", 0, ssl=contexts["valid"] if secure else None
    )
    servers = [server]
    try:
        if socket.has_ipv6:
            servers.append(
                await asyncio.start_server(
                    respond,
                    "::1",
                    server.sockets[0].getsockname()[1],
                    ssl=contexts["valid"] if secure else None,
                )
            )
        scheme = "https" if secure else "http"
        url = f"{scheme}://localhost:{server.sockets[0].getsockname()[1]}"
        timeout = httpx.Timeout(2.0, read=0.1)

        def synchronous_get():
            with network_clients.external_client(timeout=timeout) as client:
                return client.get(url)

        with pytest.raises(httpx.ReadTimeout) as caught:
            if asynchronous:
                async with network_clients.external_async_client(timeout=timeout) as client:
                    await client.get(url)
            else:
                await asyncio.to_thread(synchronous_get)
        assert len(received) == 1
        fields = safe_fields(exception_fields(caught.value))
        assert fields["network_error_kind"] == "timeout"
        assert fields["transport"] == scheme
        assert fields["proxy_source"] == "direct"
        assert fields["tls_source"] == ("explicit_file" if secure else "none")
        assert fields["tls_backend"] == ("openssl" if secure else "none")
        assert fields["stage"] == ("response_body" if body else "request_send")
        assert fields["dispatch_certainty"] == ("response_received" if body else "possible")
        if body:
            assert fields["http_status"] == 200
    finally:
        release.set()
        for current in servers:
            current.close()
        for current in servers:
            await current.wait_closed()
        await asyncio.gather(*connections)


@pytest.mark.parametrize("name", ["SSL_CERT_FILE", "SSL_CERT_DIR"])
def test_invalid_ca_is_safe_local_configuration_and_valid_ca_recovers(certificates, tmp_path, name):
    from puripuly_heart.core.external_network import NetworkConfigurationError
    from puripuly_heart.core.qq_auth_diagnostics import exception_fields, safe_fields

    with pytest.raises(NetworkConfigurationError) as caught:
        select_tls(environment={name: str(tmp_path / "private-ca-path")})
    fields = safe_fields(exception_fields(caught.value))
    assert fields["configuration_error"] == "invalid_tls_configuration"
    assert fields["dispatch_certainty"] == "not_dispatched"
    assert fields["stage"] == "client_configuration"
    assert fields["tls_source"] == (
        "explicit_file" if name == "SSL_CERT_FILE" else "explicit_directory"
    )
    assert "private-ca-path" not in repr(fields)
    root, ca_dir, _ = certificates
    selection = select_tls(environment={name: str(root if name == "SSL_CERT_FILE" else ca_dir)})
    assert selection.context.verify_mode == ssl.CERT_REQUIRED


async def test_http_proxy_denial_has_no_direct_fallback_and_corrected_route_works(monkeypatch):
    target_requests = []
    proxy_requests = []
    handlers = set()

    async def deny(reader, writer):
        handlers.add(asyncio.current_task())
        try:
            proxy_requests.append(await reader.readuntil(b"\r\n\r\n"))
            writer.write(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            handlers.discard(asyncio.current_task())

    proxy = await asyncio.start_server(deny, "127.0.0.1", 0)
    try:
        monkeypatch.setenv("HTTP_PROXY", f"http://127.0.0.1:{proxy.sockets[0].getsockname()[1]}")
        with http_server(requests=target_requests) as port:
            url = f"http://127.0.0.1:{port}/expected"
            async with network_clients.external_async_client() as client:
                response = await client.get(url)
            assert response.status_code == 403
            assert len(proxy_requests) == 1
            assert target_requests == []
            assert response.extensions["connection_diagnostics"]["proxy_source"] == "env"
            monkeypatch.delenv("HTTP_PROXY")
            async with network_clients.external_async_client() as client:
                assert (await client.get(url)).text == "ok"
            assert target_requests == ["/expected"]
    finally:
        proxy.close()
        await proxy.wait_closed()
        await asyncio.gather(*handlers)


@pytest.mark.parametrize("proxy_scheme", ["http", "https", "socks5"])
async def test_actual_broker_oauth_start_retains_proxy_tls_rate_limit_and_valid_retry(
    certificates, monkeypatch, proxy_scheme
):
    from puripuly_heart.core.openrouter.managed_openrouter_broker_client import (
        HttpManagedOpenRouterBrokerClient,
    )
    from puripuly_heart.core.openrouter.managed_openrouter_release import (
        ManagedOpenRouterReleaseError,
    )

    root, _, contexts = certificates
    monkeypatch.setenv("SSL_CERT_FILE", str(root))
    requests = []
    success = {
        "authorization_url": "https://discord.invalid/?state=synthetic-current",
        "redirect_uri": "http://127.0.0.1:62187/discord/callback",
        "oauth_session_expires_at": "2099-01-01T00:00:00Z",
        "issue_nonce": "synthetic-nonce",
        "fingerprint_salt": {"version": 1, "salt": "synthetic-salt"},
        "fingerprint_salt_version": 1,
    }

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(
                (self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            )
            body = (
                b"<html>synthetic-limit</html>"
                if len(requests) == 1
                else json.dumps(success).encode()
            )
            self.send_response(429 if len(requests) == 1 else 200)
            self.send_header("Retry-After", "7")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    with http_server(contexts["valid"], handler=Handler) as port:
        async with recording_proxy(
            contexts["valid"] if proxy_scheme == "https" else None,
            socks=proxy_scheme == "socks5",
        ) as (proxy, recorded):
            monkeypatch.setenv("HTTPS_PROXY", proxy)
            client = HttpManagedOpenRouterBrokerClient(f"https://localhost:{port}")
            kwargs = {
                "installation_id": "synthetic-installation",
                "device_public_key": "synthetic-device",
                "redirect_uri": success["redirect_uri"],
                "app_version": "synthetic-version",
            }
            try:
                with pytest.raises(ManagedOpenRouterReleaseError) as caught:
                    await client.start_discord_oauth(**kwargs)
                diagnostics = caught.value.failure_diagnostics
                assert diagnostics.status_code == 429
                assert diagnostics.retry_after_ms == 7000
                assert diagnostics.fields["proxy_source"] == "env"
                assert diagnostics.fields["tls_source"] == "explicit_file"
                assert diagnostics.fields["tls_backend"] == "openssl"
                assert diagnostics.fields["response_shape"] == "non_json"
                assert diagnostics.fields["dispatch_certainty"] == "response_received"
                result = await client.start_discord_oauth(**kwargs)
                assert result.authorization_url == success["authorization_url"]
                assert result.redirect_uri == kwargs["redirect_uri"]
                assert requests[0] == requests[1]
                assert len(recorded) == 2
                assert all(
                    method == ("SOCKS5" if proxy_scheme == "socks5" else "CONNECT")
                    for method, _ in recorded
                )
            finally:
                await client.close()
