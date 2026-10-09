from __future__ import annotations

import asyncio
import hashlib
import json
import os
import socket
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
from websockets.asyncio.server import serve
from websockets.exceptions import SecurityError

from puripuly_heart.core import external_network, network_clients, updater
from puripuly_heart.core.external_network import ProxyPolicy
from puripuly_heart.core.local_asr.local_stt_download_port import (
    HuggingFaceDownloadRequest,
    LocalSTTDownloadPortError,
)
from puripuly_heart.core.local_asr.local_stt_huggingface_xet_adapter import (
    HuggingFaceXetDownloadAdapter,
)
from puripuly_heart.core.network_requests import ExternalRequestsSession
from tests.core.test_external_network import certificates as certificates
from tests.core.test_external_network import http_server, recording_proxy
from tests.core.test_external_network import (
    isolated_network_environment as isolated_network_environment,
)


@pytest.mark.parametrize("scheme,port", [("http", 80), ("https", 443), ("ws", 80), ("wss", 443)])
@pytest.mark.parametrize("host", ["api.example.com", "::1"])
def test_default_port_bypass_survives_url_canonicalization(scheme, port, host):
    authority = f"[{host}]" if ":" in host else host
    policy = ProxyPolicy(environment={"ALL_PROXY": "proxy.invalid:8080", "NO_PROXY": f"{host}:{port}"}, system={})
    urls = [f"{scheme}://{authority}", f"{scheme}://{authority}:{port}"]
    if scheme in ("http", "https"):
        urls.append(str(httpx.Request("GET", urls[-1]).url))
    assert all(policy.route(url).source == "bypass" for url in urls)
    assert policy.route(f"{scheme}://{authority}:{port + 1}").source == "env"
    host_policy = ProxyPolicy(environment={"ALL_PROXY": "proxy.invalid:8080", "NO_PROXY": host}, system={})
    assert host_policy.route(f"{scheme}://{authority}:{port + 1}").source == "bypass"


async def test_bare_proxy_address_relays_httpx_requests_and_websocket(certificates, monkeypatch):
    root, _, contexts = certificates
    monkeypatch.setenv("SSL_CERT_FILE", str(root))
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(root))
    async def echo(ws):
        await ws.send(await ws.recv())
    async with recording_proxy() as (proxy, recorded):
        monkeypatch.setenv("HTTPS_PROXY", proxy.removeprefix("http://"))
        with http_server(contexts["valid"]) as port:
            url = f"https://localhost:{port}"
            async with network_clients.external_async_client() as client:
                assert (await client.get(url)).text == "ok"
            def request():
                with ExternalRequestsSession() as client:
                    return client.get(url, timeout=5).text
            assert await asyncio.to_thread(request) == "ok"
        async with serve(echo, "127.0.0.1", 0, ssl=contexts["valid"]) as server:
            async with network_clients.external_websocket_connect(f"wss://localhost:{server.sockets[0].getsockname()[1]}") as ws:
                await ws.send("bare-proxy-relay")
                assert await ws.recv() == "bare-proxy-relay"
        assert len(recorded) == 3
        assert all(method == "CONNECT" for method, _ in recorded)


async def test_unsupported_explicit_proxy_scheme_is_not_rewritten(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "ftp://localhost:9")
    assert ProxyPolicy().route("https://example.com").url == "ftp://localhost:9"
    async with network_clients.external_async_client() as client:
        with pytest.raises(ValueError, match="Unknown scheme"):
            await client.get("https://example.com")


@pytest.mark.parametrize("transition", ["direct_to_tls", "tls_to_direct", "tls_to_http", "http_to_tls"])
async def test_websocket_redirect_reselects_snapshot_and_proxy_tls(certificates, monkeypatch, transition):
    root, _, contexts = certificates
    monkeypatch.setenv("SSL_CERT_FILE", str(root))
    received = []
    async def target(ws):
        received.append(ws.request.headers)
        await ws.send("redirect-verified")
    async with recording_proxy(contexts["valid"]) as (tls_proxy, tls_recorded), recording_proxy() as (http_proxy, http_recorded):
        async with serve(target, "127.0.0.1", 0, ssl=contexts["valid"]) as target_server:
            target_port = target_server.sockets[0].getsockname()[1]
            target_url = f"wss://localhost:{target_port}/destination"
            def redirect(connection, request):
                response = connection.respond(302, "redirect")
                response.headers["Location"] = target_url
                return response
            async def unexpected(ws):
                pytest.fail("Redirect source must not finish a websocket handshake")
            upgraded = transition in ("tls_to_http", "http_to_tls")
            async with serve(unexpected, "127.0.0.1", 0, ssl=None if upgraded else contexts["valid"], process_request=redirect) as source_server:
                source_port = source_server.sockets[0].getsockname()[1]
                monkeypatch.setenv("HTTPS_PROXY", tls_proxy)
                if transition == "direct_to_tls":
                    monkeypatch.setenv("NO_PROXY", f"localhost:{source_port}")
                elif transition == "tls_to_direct":
                    monkeypatch.setenv("NO_PROXY", f"localhost:{target_port}")
                else:
                    monkeypatch.setenv("WS_PROXY", tls_proxy if transition == "tls_to_http" else http_proxy)
                    monkeypatch.setenv("WSS_PROXY", http_proxy if transition == "tls_to_http" else tls_proxy)
                connector = network_clients.external_websocket_connect(
                    f"{'ws' if upgraded else 'wss'}://localhost:{source_port}/source",
                    additional_headers={"Authorization": "private-key", "Cookie": "private-cookie"},
                )
                monkeypatch.setenv("HTTPS_PROXY", "http://localhost:1")
                monkeypatch.setenv("WS_PROXY", "http://localhost:1")
                monkeypatch.setenv("WSS_PROXY", "http://localhost:1")
                monkeypatch.setenv("NO_PROXY", "*")
                async with connector as ws:
                    assert await ws.recv() == "redirect-verified"
        assert len(tls_recorded) == 1
        assert len(http_recorded) == (1 if upgraded else 0)
        assert "Authorization" not in received[0]
        assert "Cookie" not in received[0]


async def test_websocket_redirect_retains_tls_downgrade_rejection(certificates, monkeypatch):
    root, _, contexts = certificates
    monkeypatch.setenv("SSL_CERT_FILE", str(root))
    contacts = []
    async def accept(reader, writer):
        contacts.append(True)
        writer.close()
        await writer.wait_closed()
    destination = await asyncio.start_server(accept, "127.0.0.1", 0)
    try:
        def redirect(connection, request):
            response = connection.respond(302, "redirect")
            response.headers["Location"] = f"ws://localhost:{destination.sockets[0].getsockname()[1]}"
            return response
        async def unused(ws):
            pass
        async with serve(unused, "127.0.0.1", 0, ssl=contexts["valid"], process_request=redirect) as source:
            with pytest.raises(SecurityError, match="non-secure"):
                await network_clients.external_websocket_connect(f"wss://localhost:{source.sockets[0].getsockname()[1]}")
        assert contacts == []
    finally:
        destination.close()
        await destination.wait_closed()


async def test_updater_actual_connection_failure_is_graceful(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "*")
    with socket.socket() as blocked:
        blocked.bind(("127.0.0.1", 0))
        monkeypatch.setattr(updater, "GITHUB_API_URL", f"http://127.0.0.1:{blocked.getsockname()[1]}/releases/latest")
        assert await updater.check_for_update() is None


async def test_updater_actual_release_request_succeeds(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "*")
    async def respond(reader, writer):
        await reader.readuntil(b"\r\n\r\n")
        body = json.dumps({"tag_name": "v999.0.0", "body": "controlled release", "assets": []}).encode()
        writer.write(f"HTTP/1.1 200 OK\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + body)
        await writer.drain()
        writer.close()
        await writer.wait_closed()
    server = await asyncio.start_server(respond, "127.0.0.1", 0)
    try:
        monkeypatch.setattr(updater, "GITHUB_API_URL", f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/releases/latest")
        info = await updater.check_for_update()
        assert info is not None
        assert info.version == "999.0.0"
        assert info.release_notes == "controlled release"
    finally:
        server.close()
        await server.wait_closed()


@contextmanager
def huggingface_file_server(context=None):
    payload = b"controlled-model-file\0" * 2048
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self):
            requests.append(("HEAD", self.path, self.headers.get("X-Amzn-Trace-Id")))
            self.send_response(200)
            self.send_header("X-Repo-Commit", "a" * 40)
            self.send_header("ETag", '"' + hashlib.sha256(payload).hexdigest() + '"')
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
        def do_GET(self):
            requests.append(("GET", self.path, self.headers.get("X-Amzn-Trace-Id")))
            if self.path != "/file":
                self.send_response(302)
                self.send_header("Location", "/file")
                self.send_header("Content-Length", "13")
                self.end_headers()
                self.wfile.write(b"redirect-body")
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    if context is not None:
        server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"{'https' if context else 'http'}://localhost:{server.server_port}", payload, requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("policy_case", ["direct_fallback", "system_fallback", "registry_bypass", "mixed", "bare_env", "explicit_ca", "explicit_disable", "failed_proxy"])
async def test_actual_huggingface_worker_retains_policy_and_downloads_redirect_hash(certificates, monkeypatch, tmp_path, policy_case):
    root, _, contexts = certificates
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    async with recording_proxy() as (proxy, recorded):
        original_system = {}
        bypass = ""
        if policy_case in ("system_fallback", "registry_bypass", "mixed", "failed_proxy"):
            original_system = {"http": proxy, "https": proxy}
        if policy_case == "failed_proxy":
            original_system = {"http": "http://localhost:1", "https": "http://localhost:1"}
            bypass = "*.intranet"
        elif policy_case == "registry_bypass":
            bypass = "localhost"
        elif policy_case == "mixed":
            original_system = {"http": proxy}
        elif policy_case == "bare_env":
            monkeypatch.setenv("HTTP_PROXY", proxy.removeprefix("http://"))
            monkeypatch.setenv("HTTPS_PROXY", proxy.removeprefix("http://"))
        elif policy_case == "explicit_ca":
            monkeypatch.setenv("SSL_CERT_FILE", str(root))
        elif policy_case == "explicit_disable":
            monkeypatch.setenv("HF_HUB_DISABLE_XET", "true")
        monkeypatch.setattr(external_network, "_windows_proxy_settings", lambda: (original_system, bypass))
        with huggingface_file_server(contexts["valid"] if policy_case == "explicit_ca" else None) as (endpoint, payload, requests):
            script = '''
import json, os, pathlib, sys
from puripuly_heart.core import external_network
from puripuly_heart.core.local_asr import local_stt_huggingface_xet_adapter as adapter
import huggingface_hub
external_network._windows_proxy_settings = lambda: ({'http': 'http://localhost:1', 'https': 'http://localhost:1'} if sys.argv[5] == 'proxy' else {}, '')
original = huggingface_hub.hf_hub_download
request_path, event_path = map(pathlib.Path, sys.argv[1:3])
payload = json.loads(request_path.read_text())
assert not any(name in payload for name in ('PURIPULY_HEART_HF_PROXY_POLICY', 'HTTP_PROXY', 'SSL_CERT_FILE'))
def download(**kwargs):
    try:
        target = original(endpoint=sys.argv[3], force_download=True, **kwargs)
    except Exception as exc:
        if sys.argv[5] == 'direct':
            seen = set()
            pending = [exc]
            sources = []
            while pending:
                error = pending.pop()
                if id(error) in seen:
                    continue
                seen.add(id(error))
                sources.append(getattr(error, 'connection_diagnostics', {}).get('proxy_source'))
                pending.extend(cause for cause in (error.__cause__, error.__context__) if cause is not None)
            assert 'system' in sources
        raise
    if sys.argv[4] == 'fallback' and os.environ.get('HF_HUB_DISABLE_XET', '').upper() not in {'1', 'TRUE'}:
        pathlib.Path(target).unlink()
        raise RuntimeError('controlled native failure')
    return target
huggingface_hub.hf_hub_download = download
raise SystemExit(adapter.run_huggingface_xet_worker(request_path=request_path, event_path=event_path))
'''
            fallback = policy_case in ("direct_fallback", "system_fallback")
            adapter = HuggingFaceXetDownloadAdapter(worker_command_factory=lambda request, request_path, event_path: [
                sys.executable, "-c", script, str(request_path), str(event_path), endpoint,
                "fallback" if fallback else "success",
                "direct" if policy_case == "failed_proxy" else "proxy",
            ])
            request = HuggingFaceDownloadRequest(repo_id="fixture/repo", revision="pinned", remote_path="model.gguf", local_dir=tmp_path / "child", expected_size_bytes=len(payload))
            captured_environment = dict(os.environ)
            if policy_case == "failed_proxy":
                with pytest.raises(LocalSTTDownloadPortError):
                    await adapter.download(request, cancel_event=None, on_progress=None)
                assert requests == []
            else:
                result = await adapter.download(request, cancel_event=None, on_progress=None)
                assert hashlib.sha256(result.read_bytes()).digest() == hashlib.sha256(payload).digest()
                assert len([method for method, _, _ in requests if method == "HEAD"]) == (2 if fallback else 1)
                assert len([path for method, path, _ in requests if method == "GET" and path == "/file"]) == (2 if fallback else 1)
                assert all(trace_id for _, _, trace_id in requests)
            assert dict(os.environ) == captured_environment
        if policy_case in ("system_fallback", "mixed", "bare_env"):
            assert recorded
        else:
            assert recorded == []
