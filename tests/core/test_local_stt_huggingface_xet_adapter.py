from __future__ import annotations

import asyncio
import hashlib
import http.server
import json
import os
import socketserver
import sys
import threading
from functools import partial
from pathlib import Path

import psutil
import pytest
from puripuly_heart.core.local_stt_assets import (
    LOCAL_QWEN_GPU_MODEL_ID,
    LocalSTTAssetFile,
    LocalSTTAssetManifest,
    LocalSTTAssetSource,
)
from puripuly_heart.core.local_stt_download_port import (
    HuggingFaceDownloadProgress,
    HuggingFaceDownloadRequest,
    LocalSTTDownloadPortError,
)
from puripuly_heart.core.local_stt_huggingface_xet_adapter import (
    HuggingFaceXetDownloadAdapter,
    run_huggingface_xet_worker,
)
from puripuly_heart.core.local_stt_runtime_installer import (
    LocalSTTRuntimeInstallCancelled,
    LocalSTTRuntimeInstallError,
    RuntimeLocalSTTStatusUpdate,
    ensure_local_stt_installed,
)

from puripuly_heart.core.runtime import LocalSTTDownloadRuntime


class _QuietHttpHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        _ = format, args


@pytest.fixture()
def file_server(tmp_path: Path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    handler = partial(_QuietHttpHandler, directory=str(source_dir))
    server = socketserver.TCPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield source_dir, f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


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
        preferred_source="huggingface",
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

    await runtime.close()
    status_count = len(statuses)
    await asyncio.sleep(0.1)

    assert task.done()
    assert not psutil.pid_exists(worker_pid)
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
