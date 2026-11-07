from __future__ import annotations

import asyncio
import hashlib
import http.server
import json
import os
import socketserver
import sys
import threading
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import psutil
import pytest

from puripuly_heart.core.local_asr.local_stt_assets import (
    LOCAL_QWEN_GPU_MODEL_ID,
    LocalSTTAssetFile,
    LocalSTTAssetManifest,
    LocalSTTAssetSource,
)
from puripuly_heart.core.local_asr.local_stt_download_port import (
    HuggingFaceDownloadProgress,
    HuggingFaceDownloadRequest,
    LocalSTTDownloadPortError,
)
from puripuly_heart.core.local_asr.local_stt_huggingface_xet_adapter import (
    HuggingFaceXetDownloadAdapter,
    run_huggingface_xet_worker,
)
from puripuly_heart.core.local_asr.local_stt_runtime_installer import (
    LocalSTTRuntimeInstallCancelled,
    LocalSTTRuntimeInstallError,
    RuntimeLocalSTTStatusUpdate,
    ensure_local_stt_installed,
)
from puripuly_heart.core.network_diagnostics import safe_transport_fields
from puripuly_heart.core.runtime import LocalSTTDownloadRuntime
from tests.core.test_external_network import (
    isolated_network_environment as isolated_network_environment,
)
from tests.core.test_external_network import (
    recording_proxy,
)


@pytest.mark.parametrize("native", [False, True], ids=["source", "native"])
def test_worker_command_keeps_request_paths_and_current_host_protocol(
    tmp_path, monkeypatch, native
):
    from puripuly_heart.core.local_asr.local_stt_huggingface_xet_adapter import (
        _default_worker_command,
    )
    from puripuly_heart.runtime_layout import current_runtime_layout

    root = tmp_path / "설치 with spaces"
    if native:
        monkeypatch.setenv("PURIPULY_HEART_NATIVE_RESOURCE_ROOT", str(root / "app"))
        monkeypatch.setenv("PURIPULY_HEART_NATIVE_HOST_EXECUTABLE", str(root / "PuriPulyHeart.exe"))
    else:
        monkeypatch.delenv("PURIPULY_HEART_NATIVE_RESOURCE_ROOT", raising=False)
    request_path = tmp_path / "요청 with spaces.json"
    event_path = tmp_path / "events with spaces.jsonl"
    request = HuggingFaceDownloadRequest(
        repo_id="fixture/repo",
        revision="pinned",
        remote_path="asset.bin",
        local_dir=tmp_path,
        expected_size_bytes=1,
    )
    command = list(_default_worker_command(request, request_path, event_path))
    prefix = [str(current_runtime_layout().host_executable)]
    prefix += ["--headless"] if native else ["-m", "puripuly_heart.main"]
    assert command == [
        *prefix,
        "hf-xet-download-worker",
        "--request-file",
        str(request_path),
        "--event-file",
        str(event_path),
    ]


class _QuietHttpHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        _ = format, args


@pytest.fixture()
def file_server(tmp_path: Path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    handler = partial(_QuietHttpHandler, directory=str(source_dir))
    server = socketserver.TCPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        yield source_dir, f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@contextmanager
def _xet_advertising_file_server(payload: bytes, *, fail_native: bool = False):
    requests: list[tuple[str, str]] = []
    started = threading.Event()
    release = threading.Event()
    chunks = [payload[offset : offset + 65536] for offset in range(0, len(payload), 65536)]
    wire = b"".join(
        b"\0" + len(chunk).to_bytes(3, "little") + b"\0" + len(chunk).to_bytes(3, "little") + chunk
        for chunk in chunks
    )
    xorb_hash = "c" * 64

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_HEAD(self):
            requests.append(("HEAD", self.path))
            self.send_response(200)
            self.send_header("X-Repo-Commit", "a" * 40)
            self.send_header("ETag", f'"{hashlib.sha256(payload).hexdigest()}"')
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("X-Xet-Hash", "b" * 64)
            self.send_header(
                "X-Xet-Refresh-Route",
                f"http://127.0.0.1:{self.server.server_address[1]}/xet-auth",
            )
            self.end_headers()

        def do_GET(self):
            requests.append(("GET", self.path))
            endpoint = f"http://127.0.0.1:{self.server.server_address[1]}"
            if self.path == "/xet-auth":
                self.send_response(200)
                self.send_header("X-Xet-Cas-Url", endpoint)
                self.send_header("X-Xet-Access-Token", "synthetic-cas-token")
                self.send_header("X-Xet-Token-Expiration", "4102444800")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if self.path.startswith("/v2/reconstructions/"):
                if fail_native:
                    self.send_error(400)
                    return
                requested_range = self.headers.get("Range", "bytes=0-")
                range_start = int(requested_range.partition("=")[2].partition("-")[0])
                if range_start >= len(payload):
                    self.send_error(416)
                    return
                body = json.dumps(
                    {
                        "offset_into_first_range": 0,
                        "terms": [
                            {
                                "hash": xorb_hash,
                                "unpacked_length": len(payload),
                                "range": {"start": 0, "end": len(chunks)},
                            }
                        ],
                        "xorbs": {
                            xorb_hash: [
                                {
                                    "url": f"{endpoint}/xorb",
                                    "ranges": [
                                        {
                                            "chunks": {"start": 0, "end": len(chunks)},
                                            "bytes": {"start": 0, "end": len(wire) - 1},
                                        }
                                    ],
                                }
                            ]
                        },
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            content = wire if self.path == "/xorb" else payload
            self.send_response(206 if self.path == "/xorb" else 200)
            self.send_header("Content-Length", str(len(content)))
            if self.path == "/xorb":
                self.send_header("Content-Range", f"bytes 0-{len(wire) - 1}/{len(wire)}")
            self.end_headers()
            started.set()
            try:
                self.wfile.write(content[:1024])
                release.wait(timeout=10)
                self.wfile.write(content[1024:])
            except BrokenPipeError, ConnectionResetError, ConnectionAbortedError:
                pass

        def do_POST(self):
            requests.append(("POST", self.path))
            self.send_error(400)

        def log_message(self, *_args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", requests, started, release
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
async def download_proxy(transport):
    async with recording_proxy(socks=transport == "native_socks") as selected:
        yield selected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("transport", "outcome"),
    [
        ("native", "complete"),
        ("native", "cancel"),
        ("native_proxy", "complete"),
        ("native_socks", "complete"),
        ("native", "corrupt"),
        ("disabled", "complete"),
        ("disabled", "cancel"),
        ("native_failure", "complete"),
        ("native_failure", "cancel"),
        ("unavailable", "complete"),
    ],
)
async def test_actual_worker_transfer_preserves_verified_atomic_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str, transport: str, download_proxy
) -> None:
    payload = b"controlled-http-model\0" * 65536
    manifest = _gpu_manifest(payload if outcome != "corrupt" else b"x" * len(payload))
    cancel_event = threading.Event()
    statuses: list[RuntimeLocalSTTStatusUpdate] = []
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "true" if transport == "disabled" else "0")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "isolated-hf-home"))
    monkeypatch.setenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    monkeypatch.setenv("NO_PROXY", "*")
    for name in ("SSL_CERT_FILE", "SSL_CERT_DIR", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    proxy, proxied = download_proxy
    if transport in ("native_proxy", "native_socks"):
        monkeypatch.delenv("NO_PROXY")
        monkeypatch.setenv("HTTP_PROXY", proxy)
        monkeypatch.setenv("HTTPS_PROXY", proxy)
    script = """
import pathlib, sys
from huggingface_hub.utils import _runtime
_runtime._package_versions["hf_xet"] = "N/A"
from puripuly_heart.core.local_asr.local_stt_huggingface_xet_adapter import run_huggingface_xet_worker
raise SystemExit(run_huggingface_xet_worker(
    request_path=pathlib.Path(sys.argv[1]), event_path=pathlib.Path(sys.argv[2])
))
"""
    with _xet_advertising_file_server(payload, fail_native=transport == "native_failure") as (
        endpoint,
        requests,
        started,
        release,
    ):
        monkeypatch.setenv("HF_ENDPOINT", endpoint)
        adapter = HuggingFaceXetDownloadAdapter(
            worker_command_factory=(
                (
                    lambda _request, request_path, event_path: [
                        sys.executable,
                        "-c",
                        script,
                        str(request_path),
                        str(event_path),
                    ]
                )
                if transport == "unavailable"
                else None
            )
        )
        task = asyncio.create_task(
            ensure_local_stt_installed(
                manifest=manifest,
                model_root=tmp_path,
                preferred_source="huggingface",
                cancel_event=cancel_event,
                on_status=statuses.append,
                huggingface_downloader=adapter,
            )
        )
        try:
            assert await asyncio.to_thread(started.wait, 10)
            assert not (tmp_path / manifest.install_dirname).exists()
            worker_pid = int(adapter.child_states[0].split("pid=", 1)[1].split(":", 1)[0])
            if outcome == "cancel":
                cancel_event.set()
                with pytest.raises(LocalSTTRuntimeInstallCancelled):
                    await asyncio.wait_for(task, timeout=10)
                assert not (tmp_path / manifest.install_dirname).exists()
            elif outcome == "corrupt":
                release.set()
                with pytest.raises(LocalSTTRuntimeInstallError, match="checksum mismatch"):
                    await asyncio.wait_for(task, timeout=15)
                assert not (tmp_path / manifest.install_dirname).exists()
            else:
                release.set()
                installed = await asyncio.wait_for(task, timeout=15)
                assert installed.selected_source == "huggingface"
                target = tmp_path / manifest.install_dirname / manifest.files[0].relative_path
                assert target.read_bytes() == payload
                assert any(item.status == "downloading" and item.percent > 0 for item in statuses)
            assert adapter.child_states == ()
            assert not psutil.pid_exists(worker_pid)
        finally:
            release.set()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    resolve_path = (
        "/handy-computer/Qwen3-ASR-1.7B-gguf/resolve/"
        "92282af1610a2db19d66f2bef1e260f5deca782d/remote/model-q6-k.gguf"
    )
    assert requests[0] == ("HEAD", resolve_path)
    assert not any(method == "POST" for method, _path in requests)
    if transport in ("native", "native_proxy", "native_socks"):
        assert ("GET", "/xet-auth") in requests
        assert ("GET", "/v2/reconstructions/" + "b" * 64) in requests
        assert ("GET", "/xorb") in requests
        assert ("GET", resolve_path) not in requests
    elif transport == "native_failure":
        assert ("GET", "/xet-auth") in requests
        assert ("GET", "/v2/reconstructions/" + "b" * 64) in requests
        assert ("GET", resolve_path) in requests
    else:
        assert requests == [("HEAD", resolve_path), ("GET", resolve_path)]
    assert not tuple(tmp_path.rglob(".hf-xet-*"))
    assert not tuple(tmp_path.rglob(".cache"))
    assert not tuple(tmp_path.glob("*.staging-*"))
    if transport == "native_proxy":
        assert ("GET", endpoint + "/xorb") in proxied
    elif transport == "native_socks":
        assert any(method == "SOCKS5" for method, _target in proxied)
    else:
        assert proxied == []


def _gpu_manifest(payload: bytes, *, modelscope_url: str = "") -> LocalSTTAssetManifest:
    sources = {
        "huggingface": LocalSTTAssetSource(
            name="huggingface",
            revision="92282af1610a2db19d66f2bef1e260f5deca782d",
            repo_id="handy-computer/Qwen3-ASR-1.7B-gguf",
            download_url_template="https://example.invalid/{path}",
        )
    }
    if modelscope_url:
        sources["modelscope"] = LocalSTTAssetSource(
            name="modelscope",
            revision="modelscope-revision",
            download_url_template=f"{modelscope_url}/{{path}}",
        )
    return LocalSTTAssetManifest(
        manifest_version=1,
        installed_manifest_version=1,
        model_id=LOCAL_QWEN_GPU_MODEL_ID,
        engine="transcribe.cpp-vulkan",
        upstream_repo="Qwen/Qwen3-ASR-1.7B@revision",
        install_dirname=LOCAL_QWEN_GPU_MODEL_ID,
        sources=sources,
        files=(
            LocalSTTAssetFile(
                relative_path="Qwen3-ASR-1.7B-Q6_K.gguf",
                sha256=hashlib.sha256(payload).hexdigest(),
                size_bytes=len(payload),
                source_path_overrides={"huggingface": "remote/model-q6-k.gguf"},
            ),
        ),
    )


class _FakeDownloader:
    def __init__(self, payload: bytes, *, fail: bool = False) -> None:
        self.payload = payload
        self.fail = fail
        self.requests: list[HuggingFaceDownloadRequest] = []

    async def download(self, request, *, cancel_event, on_progress):
        self.requests.append(request)
        if self.fail:
            raise LocalSTTDownloadPortError("fixture failure")
        target = request.local_dir / request.remote_path
        target.parent.mkdir(parents=True, exist_ok=True)
        (request.local_dir / ".cache" / "huggingface").mkdir(parents=True)
        for downloaded in (3, 2, len(self.payload)):
            if on_progress is not None:
                on_progress(
                    HuggingFaceDownloadProgress(
                        downloaded_bytes=downloaded,
                        total_bytes=len(self.payload),
                    )
                )
            await asyncio.sleep(0.06)
        target.write_bytes(self.payload)
        return target


@pytest.mark.asyncio
async def test_gpu_huggingface_route_uses_exact_structured_request_and_monotonic_status(
    tmp_path: Path,
) -> None:
    payload = b"gpu-model-payload"
    manifest = _gpu_manifest(payload)
    downloader = _FakeDownloader(payload)
    statuses: list[RuntimeLocalSTTStatusUpdate] = []

    installed = await ensure_local_stt_installed(
        manifest=manifest,
        model_root=tmp_path,
        preferred_source="huggingface",
        huggingface_downloader=downloader,
        on_status=statuses.append,
    )

    request = downloader.requests[0]
    assert request.repo_id == "handy-computer/Qwen3-ASR-1.7B-gguf"
    assert request.revision == "92282af1610a2db19d66f2bef1e260f5deca782d"
    assert request.remote_path == "remote/model-q6-k.gguf"
    assert request.local_dir.parent == tmp_path
    assert ".staging-" in request.local_dir.name
    assert request.expected_size_bytes == len(payload)
    assert installed.selected_source == "huggingface"
    percents = [item.percent for item in statuses if item.status == "downloading"]
    assert percents == sorted(percents)
    assert not (tmp_path / manifest.install_dirname / ".cache").exists()


@pytest.mark.asyncio
async def test_gpu_huggingface_checksum_failure_does_not_promote(tmp_path: Path) -> None:
    manifest = _gpu_manifest(b"expected")
    downloader = _FakeDownloader(b"corrupt")

    with pytest.raises(LocalSTTRuntimeInstallError, match="checksum mismatch"):
        await ensure_local_stt_installed(
            manifest=manifest,
            model_root=tmp_path,
            huggingface_downloader=downloader,
        )

    assert not (tmp_path / manifest.install_dirname).exists()


@pytest.mark.asyncio
async def test_gpu_huggingface_failure_preserves_http_source_fallback(
    tmp_path: Path,
    file_server,
) -> None:
    source_dir, base_url = file_server
    payload = b"fallback-model"
    (source_dir / "Qwen3-ASR-1.7B-Q6_K.gguf").write_bytes(payload)
    manifest = _gpu_manifest(payload, modelscope_url=base_url)
    downloader = _FakeDownloader(payload, fail=True)

    installed = await ensure_local_stt_installed(
        manifest=manifest,
        model_root=tmp_path,
        locale="en",
        huggingface_downloader=downloader,
    )

    assert installed.selected_source == "modelscope"


@pytest.mark.asyncio
async def test_gpu_huggingface_cancel_stops_helper_and_never_promotes(tmp_path: Path) -> None:
    pid_path = tmp_path / "worker.pid"
    script = (
        "import json,os,pathlib,sys,time;"
        "event=pathlib.Path(sys.argv[2]);"
        "cache=pathlib.Path(sys.argv[1]).parent/'.cache';"
        "cache.mkdir(parents=True);"
        "held=(cache/'held.lock').open('w');"
        "pathlib.Path(r'" + str(pid_path) + "').write_text(str(os.getpid()));"
        "event.write_text(json.dumps({'type':'progress','downloaded_bytes':1,'total_bytes':100})+'\\n');"
        "time.sleep(60)"
    )
    adapter = HuggingFaceXetDownloadAdapter(
        worker_command_factory=lambda _request, request_path, event_path: [
            sys.executable,
            "-c",
            script,
            str(request_path),
            str(event_path),
        ]
    )
    cancel_event = threading.Event()
    statuses: list[RuntimeLocalSTTStatusUpdate] = []
    manifest = _gpu_manifest(b"expected-model")
    task = asyncio.create_task(
        ensure_local_stt_installed(
            manifest=manifest,
            model_root=tmp_path,
            cancel_event=cancel_event,
            on_status=statuses.append,
            huggingface_downloader=adapter,
        )
    )
    while not pid_path.exists():
        await asyncio.sleep(0.01)
    worker_pid = int(pid_path.read_text())

    cancel_event.set()
    with pytest.raises(LocalSTTRuntimeInstallCancelled):
        await asyncio.wait_for(task, timeout=2)
    status_count = len(statuses)
    await asyncio.sleep(0.1)

    assert not psutil.pid_exists(worker_pid)
    assert len(statuses) == status_count
    assert not (tmp_path / manifest.install_dirname).exists()
    assert not tuple(tmp_path.rglob(".hf-xet-*"))


@pytest.mark.asyncio
async def test_runtime_close_leaves_no_huggingface_xet_helper_or_late_status(
    tmp_path: Path,
) -> None:
    pid_path = tmp_path / "close-worker.pid"
    script = (
        "import json,os,pathlib,sys,time;"
        "event=pathlib.Path(sys.argv[2]);"
        "pathlib.Path(r'" + str(pid_path) + "').write_text(str(os.getpid()));"
        "event.write_text(json.dumps({'type':'progress','downloaded_bytes':1,'total_bytes':100})+'\\n');"
        "time.sleep(60)"
    )
    adapter = HuggingFaceXetDownloadAdapter(
        worker_command_factory=lambda _request, request_path, event_path: [
            sys.executable,
            "-c",
            script,
            str(request_path),
            str(event_path),
        ]
    )
    manifest = _gpu_manifest(b"expected-model")
    runtime = LocalSTTDownloadRuntime(cancel_timeout_s=2)
    statuses: list[RuntimeLocalSTTStatusUpdate] = []

    async def run_download(cancel_event: threading.Event, _generation: int):
        return await ensure_local_stt_installed(
            manifest=manifest,
            model_root=tmp_path,
            cancel_event=cancel_event,
            on_status=statuses.append,
            huggingface_downloader=adapter,
        )

    task = runtime.start(origin="test", run_download=run_download)
    while not pid_path.exists():
        await asyncio.sleep(0.01)
    worker_pid = int(pid_path.read_text())
    assert len(adapter.child_states) == 1
    diagnostic_worker_pid = int(adapter.child_states[0].split("pid=", 1)[1].split(":", 1)[0])
    assert psutil.pid_exists(diagnostic_worker_pid)

    await runtime.close()
    status_count = len(statuses)
    await asyncio.sleep(0.1)

    assert task.done()
    assert not psutil.pid_exists(worker_pid)
    assert adapter.child_states == ()
    assert not psutil.pid_exists(diagnostic_worker_pid)
    assert len(statuses) == status_count
    assert not (tmp_path / manifest.install_dirname).exists()
    assert not tuple(tmp_path.rglob(".hf-xet-*"))


def test_worker_uses_token_false_local_dir_and_removes_hf_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_dir = tmp_path / "local"
    request_path = tmp_path / "request.json"
    event_path = tmp_path / "events.jsonl"
    calls: list[dict[str, object]] = []

    def fake_download(**kwargs):
        calls.append({**kwargs, "xet_cache": os.environ.get("HF_XET_CACHE")})
        target = Path(kwargs["local_dir"]) / str(kwargs["filename"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"fixture")
        (Path(kwargs["local_dir"]) / ".cache" / "huggingface").mkdir(parents=True)
        progress = kwargs["tqdm_class"](total=7, initial=0)
        progress.update(7)
        return str(target)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    request_path.write_text(
        json.dumps(
            {
                "repo_id": "fixture/repo",
                "revision": "pinned-revision",
                "remote_path": "folder/model.gguf",
                "local_dir": str(local_dir),
                "expected_size_bytes": 7,
            }
        ),
        encoding="utf-8",
    )
    event_path.touch()

    assert run_huggingface_xet_worker(request_path=request_path, event_path=event_path) == 0

    assert calls[0]["repo_id"] == "fixture/repo"
    assert calls[0]["revision"] == "pinned-revision"
    assert calls[0]["filename"] == "folder/model.gguf"
    assert calls[0]["token"] is False
    assert calls[0]["local_dir"] == local_dir.resolve()
    assert calls[0]["xet_cache"] == str(local_dir.resolve() / ".cache" / "xet")
    assert not (local_dir / ".cache").exists()
    messages = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()]
    assert messages[0]["type"] == "progress"
    assert messages[-1]["type"] == "complete"


def test_worker_failure_reports_bounded_error_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_dir = tmp_path / "local"
    request_path = tmp_path / "request.json"
    event_path = tmp_path / "events.jsonl"

    def fail_download(**_kwargs):
        raise OSError(28, "fixture disk full")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fail_download)
    request_path.write_text(
        json.dumps(
            {
                "repo_id": "fixture/repo",
                "revision": "pinned-revision",
                "remote_path": "folder/model.gguf",
                "local_dir": str(local_dir),
                "expected_size_bytes": 7,
            }
        ),
        encoding="utf-8",
    )
    event_path.touch()

    assert run_huggingface_xet_worker(request_path=request_path, event_path=event_path) == 1

    message = json.loads(event_path.read_text(encoding="utf-8").splitlines()[-1])
    assert message["type"] == "error"
    assert message["failure_code"] == "download_failed"
    assert message["error_type"] == "OSError"
    assert message["os_error_code"] == 28


@pytest.mark.asyncio
async def test_adapter_preserves_worker_failure_metadata(tmp_path: Path) -> None:
    script = (
        "import json,pathlib,sys;"
        "pathlib.Path(sys.argv[1]).write_text(json.dumps({"
        "'type':'error','failure_code':'download_failed','error_type':'RuntimeError',"
        "'message':'fixture failure','status_code':503})+'\\n');"
        "raise SystemExit(1)"
    )
    adapter = HuggingFaceXetDownloadAdapter(
        worker_command_factory=lambda _request, _request_path, event_path: [
            sys.executable,
            "-c",
            script,
            str(event_path),
        ]
    )
    request = HuggingFaceDownloadRequest(
        repo_id="fixture/repo",
        revision="pinned-revision",
        remote_path="model.gguf",
        local_dir=tmp_path / "local",
        expected_size_bytes=7,
    )

    with pytest.raises(LocalSTTDownloadPortError) as captured:
        await adapter.download(request, cancel_event=None, on_progress=None)

    assert captured.value.failure_code == "download_failed"
    assert captured.value.cause_type == "RuntimeError"
    assert captured.value.worker_exit_code == 1
    assert captured.value.status_code == 503


@pytest.mark.asyncio
@pytest.mark.parametrize("inherited_disable_xet", [None, "0", "true"])
async def test_adapter_respects_inherited_native_xet_choice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inherited_disable_xet: str | None,
) -> None:
    if inherited_disable_xet is None:
        monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    else:
        monkeypatch.setenv("HF_HUB_DISABLE_XET", inherited_disable_xet)
    script = """
import json, os, pathlib, sys
request = json.loads(pathlib.Path(sys.argv[1]).read_text())
target = pathlib.Path(request["local_dir"]) / request["remote_path"]
target.write_text(json.dumps(os.environ.get("HF_HUB_DISABLE_XET")))
pathlib.Path(sys.argv[2]).write_text(json.dumps({"type": "complete", "path": str(target)}) + "\\n")
"""
    adapter = HuggingFaceXetDownloadAdapter(
        worker_command_factory=lambda _request, request_path, event_path: [
            sys.executable,
            "-c",
            script,
            str(request_path),
            str(event_path),
        ]
    )
    request = HuggingFaceDownloadRequest(
        repo_id="fixture/repo",
        revision="pinned-revision",
        remote_path="model.gguf",
        local_dir=tmp_path / "local",
        expected_size_bytes=None,
    )

    downloaded = await adapter.download(request, cancel_event=None, on_progress=None)

    assert json.loads(downloaded.read_text()) == inherited_disable_xet
    assert os.environ.get("HF_HUB_DISABLE_XET") == inherited_disable_xet
    assert adapter.child_states == ()
    assert not tuple(request.local_dir.glob(".hf-xet-*"))


@pytest.mark.asyncio
async def test_invalid_ca_prevents_worker_launch(tmp_path, monkeypatch):
    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "missing.pem"))

    def command(*args):
        pytest.fail("Invalid explicit trust must fail before a worker is launched")

    adapter = HuggingFaceXetDownloadAdapter(worker_command_factory=command)
    request = HuggingFaceDownloadRequest(
        repo_id="fixture/repo",
        revision="pinned",
        remote_path="model",
        local_dir=tmp_path / "local",
        expected_size_bytes=7,
    )
    with pytest.raises(LocalSTTDownloadPortError) as caught:
        await adapter.download(request, cancel_event=None, on_progress=None)
    assert caught.value.failure_code == "network_configuration_failed"
    fields = safe_transport_fields(caught.value)
    assert fields["configuration_error"] == "invalid_tls_configuration"
    assert fields["dispatch_certainty"] == "not_dispatched"
    assert fields["stage"] == "client_configuration"
    assert fields["tls_source"] == "explicit_file"
