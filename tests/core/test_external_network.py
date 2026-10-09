from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import os
import socket
import ssl
import threading
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timedelta, timezone
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


@pytest.fixture(autouse=True)
def isolated_network_environment(monkeypatch):
    for name in tuple(os.environ):
        if name.lower().endswith("_proxy") or name in (
            "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"
        ):
            monkeypatch.delenv(name)
    monkeypatch.setattr(external_network, "_windows_proxy_settings", lambda: ({}, ""))


@pytest.fixture
def certificates(tmp_path):
    now = datetime.now(timezone.utc)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "network test root")])
    root = (
        x509.CertificateBuilder().subject_name(name).issuer_name(name)
        .public_key(key.public_key()).serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1)).not_valid_after(now + timedelta(days=2))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(x509.KeyUsage(True, False, False, False, False, True, True, False, False), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .sign(key, hashes.SHA256())
    )
    root_path = tmp_path / "root.pem"
    root_path.write_bytes(root.public_bytes(serialization.Encoding.PEM))
    contexts = {}
    for variant in ("valid", "expired"):
        leaf_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        expiry = now - timedelta(hours=1) if variant == "expired" else now + timedelta(days=1)
        leaf = (
            x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")]))
            .issuer_name(name).public_key(leaf_key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(days=1)).not_valid_after(expiry)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost")]), critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(key.public_key()), critical=False)
            .sign(key, hashes.SHA256())
        )
        cert_path, key_path = tmp_path / f"{variant}.pem", tmp_path / f"{variant}.key"
        cert_path.write_bytes(leaf.public_bytes(serialization.Encoding.PEM))
        key_path.write_bytes(leaf_key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert_path, key_path)
        contexts[variant] = context
    canonical_name = name.public_bytes()[2:]
    ca_dir = tmp_path / "ca"
    ca_dir.mkdir()
    ca_hash = int.from_bytes(hashlib.sha1(canonical_name).digest()[:4], "little")
    (ca_dir / f"{ca_hash:08x}.0").write_bytes(root_path.read_bytes())
    return root_path, ca_dir, contexts


@contextmanager
def http_server(context=None, *, ipv6=False):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    servers = [server]
    if ipv6:
        class IPv6Server(ThreadingHTTPServer):
            address_family = socket.AF_INET6
        servers.append(IPv6Server(("::1", server.server_address[1]), Handler))
    threads = []
    for current in servers:
        if context is not None:
            current.socket = context.wrap_socket(current.socket, server_side=True)
        thread = threading.Thread(target=current.serve_forever, daemon=True)
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
async def recording_proxy(context=None, *, socks=False):
    requests = []
    tasks = set()

    async def pipe(reader, writer):
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()

    async def handle(reader, writer):
        remote = None
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
                    host = str(ipaddress.ip_address(await reader.readexactly(4 if kind == 1 else 16)))
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
                    upstream, remote = await asyncio.open_connection(parsed.hostname, parsed.port or 80)
                    first_line, remaining = header.split(b"\r\n", 1)
                    path = parsed.path or "/"
                    if parsed.query:
                        path += "?" + parsed.query
                    remote.write(f"{method} {path} HTTP/1.1\r\n".encode() + remaining)
                    await remote.drain()
            forward = asyncio.create_task(pipe(reader, remote))
            backward = asyncio.create_task(pipe(upstream, writer))
            done, pending = await asyncio.wait((forward, backward), return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            await asyncio.gather(*done, *pending, return_exceptions=True)
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()
            if remote is not None:
                remote.close()

    def accept(reader, writer):
        task = asyncio.create_task(handle(reader, writer))
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    server = await asyncio.start_server(accept, "127.0.0.1", 0, ssl=context)
    try:
        scheme = "socks5" if socks else "https" if context is not None else "http"
        yield f"{scheme}://localhost:{server.sockets[0].getsockname()[1]}", requests
    finally:
        server.close()
        await server.wait_closed()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


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
    policy = ProxyPolicy(environment={
        "HTTPS_PROXY": "http://https.invalid", "ALL_PROXY": "socks5://all.invalid",
        "WSS_PROXY": "http://wss.invalid", "NO_PROXY": "localhost:444,.example.com",
    }, system={})
    assert policy.route("https://host.invalid").url == "http://https.invalid"
    assert policy.route("wss://host.invalid").url == "http://wss.invalid"
    for scheme in ("https", "wss"):
        assert policy.route(f"{scheme}://localhost:444").source == "bypass"
        assert policy.route(f"{scheme}://example.com").source == "bypass"
        assert policy.route(f"{scheme}://localhost:445").source == "env"
    assert ProxyPolicy(environment={"ALL_PROXY": "socks5://all.invalid"}, system={}).route("wss://host.invalid").url == "socks5://all.invalid"


@pytest.mark.parametrize("directory", [False, True])
async def test_explicit_ca_https_and_wss(certificates, monkeypatch, directory):
    root, ca_dir, contexts = certificates
    monkeypatch.setenv("SSL_CERT_DIR" if directory else "SSL_CERT_FILE", str(ca_dir if directory else root))
    async def echo(ws):
        async for message in ws:
            await ws.send(message)
    with http_server(contexts["valid"]) as port:
        async with network_clients.external_async_client() as client:
            assert (await client.get(f"https://localhost:{port}")).text == "ok"
        assert await asyncio.to_thread(lambda: _sync_get(f"https://localhost:{port}")) == "ok"
    async with serve(echo, "127.0.0.1", 0, ssl=contexts["valid"]) as server:
        async with network_clients.external_websocket_connect(f"wss://localhost:{server.sockets[0].getsockname()[1]}") as ws:
            await ws.send("verified")
            assert await ws.recv() == "verified"


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
        assert set(caught.value.connection_diagnostics) == {"transport", "proxy_source", "tls_source", "tls_backend"}


@pytest.mark.parametrize("name", ["SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"])
def test_invalid_explicit_ca_fails(tmp_path, name):
    with pytest.raises(OSError):
        select_tls(requests=name in ("REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"), environment={name: str(tmp_path / "missing")})


@pytest.mark.parametrize("https_proxy", [False, True])
@pytest.mark.parametrize("proxy_variable", ["HTTPS_PROXY", "ALL_PROXY"])
async def test_https_wss_proxy_parity(certificates, monkeypatch, https_proxy, proxy_variable):
    root, _, contexts = certificates
    monkeypatch.setenv("SSL_CERT_FILE", str(root))
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(root))
    async def echo(ws):
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
        async with serve(echo, "127.0.0.1", 0, ssl=contexts["valid"]) as server:
            url = f"wss://localhost:{server.sockets[0].getsockname()[1]}"
            async with network_clients.external_websocket_connect(url) as ws:
                await ws.send("proxied")
                assert await ws.recv() == "proxied"
        assert len(recorded) == 3
        assert all(method == "CONNECT" for method, _ in recorded)


async def test_no_proxy_bypasses_both_transports(certificates, monkeypatch):
    root, _, contexts = certificates
    monkeypatch.setenv("SSL_CERT_FILE", str(root))
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "localhost")
    async def echo(ws):
        await ws.send("direct")
    with http_server(contexts["valid"]) as port:
        async with network_clients.external_async_client() as client:
            assert (await client.get(f"https://localhost:{port}")).status_code == 200
    async with serve(echo, "127.0.0.1", 0, ssl=contexts["valid"]) as server:
        async with network_clients.external_websocket_connect(f"wss://localhost:{server.sockets[0].getsockname()[1]}") as ws:
            assert await ws.recv() == "direct"


async def test_mock_transport_injection_does_not_resolve_invalid_environment(monkeypatch):
    monkeypatch.setenv("SSL_CERT_FILE", "missing-explicit-ca")
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text="injected"))
    async with network_clients.external_async_client(transport=transport) as client:
        assert (await client.get("https://example.invalid")).text == "injected"


async def test_socks5_http_and_async_websocket_routes(certificates, monkeypatch):
    root, _, contexts = certificates
    monkeypatch.setenv("SSL_CERT_FILE", str(root))
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(root))
    async def echo(ws):
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
        async with serve(echo, "127.0.0.1", 0, ssl=contexts["valid"]) as server:
            url = f"wss://localhost:{server.sockets[0].getsockname()[1]}"
            async with network_clients.external_websocket_connect(url) as ws:
                await ws.send("async")
                assert await ws.recv() == "async"
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


@pytest.mark.parametrize("policy_case", ["direct_bypass_only", "explicit_ca", "proxy_bypass", "proxy_normalized"])
async def test_huggingface_child_selects_compatible_transport(certificates, monkeypatch, tmp_path, policy_case):
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
target.write_text(json.dumps({name: os.environ.get(name) for name in ('SSL_CERT_FILE', 'HF_HUB_DISABLE_XET', 'HTTP_PROXY', 'HTTPS_PROXY')}))
pathlib.Path(sys.argv[2]).write_text(json.dumps({'type': 'complete', 'path': str(target)}) + '\\n')
"""
    adapter = HuggingFaceXetDownloadAdapter(worker_command_factory=lambda request, request_path, event_path: [
        sys.executable, "-c", script, str(request_path), str(event_path),
    ])
    path = await adapter.download(
        HuggingFaceDownloadRequest(
            repo_id="fixture/repo", revision="pinned", remote_path="result.json",
            local_dir=tmp_path / "child", expected_size_bytes=None,
        ), cancel_event=None, on_progress=None,
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


async def test_https_proxy_wrong_hostname_is_rejected_without_direct_fallback(certificates, monkeypatch):
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
