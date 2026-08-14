from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import threading
import time
from pathlib import Path

import httpx
import psutil
from puripuly_heart.core.local_stt_download_port import (
    HuggingFaceDownloadProgress,
    HuggingFaceDownloadRequest,
    LocalSTTDownloadPortCancelled,
)
from puripuly_heart.core.local_stt_huggingface_xet_adapter import (
    HuggingFaceXetDownloadAdapter,
)

SMOKE_REPO_ID = "Qwen/Qwen3-Reranker-0.6B"
SMOKE_REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
SMOKE_REMOTE_PATH = "tokenizer.json"
SMOKE_SIZE_BYTES = 11422654
SMOKE_SHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
CANCEL_REPO_ID = "handy-computer/Qwen3-ASR-1.7B-gguf"
CANCEL_REVISION = "92282af1610a2db19d66f2bef1e260f5deca782d"
CANCEL_REMOTE_PATH = "Qwen3-ASR-1.7B-Q6_K.gguf"
CANCEL_SIZE_BYTES = 1692554208


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frozen_worker_command(executable: Path):
    def build(_request, request_path: Path, event_path: Path):
        return [
            str(executable),
            "hf-xet-download-worker",
            "--request-file",
            str(request_path),
            "--event-file",
            str(event_path),
        ]

    return build


def _assert_xet_metadata() -> str:
    url = f"https://huggingface.co/{SMOKE_REPO_ID}/resolve/" f"{SMOKE_REVISION}/{SMOKE_REMOTE_PATH}"
    response = httpx.head(url, follow_redirects=False, timeout=20)
    if response.status_code >= 400:
        response.raise_for_status()
    xet_hash = response.headers.get("x-xet-hash")
    if not xet_hash:
        raise RuntimeError("frozen smoke fixture is not Xet-backed")
    return xet_hash


async def _run(executable: Path, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter = HuggingFaceXetDownloadAdapter(
        worker_command_factory=_frozen_worker_command(executable)
    )
    progress: list[HuggingFaceDownloadProgress] = []
    smoke_dir = output_dir / "small-xet"
    started_at = time.monotonic()
    downloaded_path = await adapter.download(
        HuggingFaceDownloadRequest(
            repo_id=SMOKE_REPO_ID,
            revision=SMOKE_REVISION,
            remote_path=SMOKE_REMOTE_PATH,
            local_dir=smoke_dir,
            expected_size_bytes=SMOKE_SIZE_BYTES,
        ),
        cancel_event=None,
        on_progress=progress.append,
    )
    elapsed_seconds = time.monotonic() - started_at
    if downloaded_path.stat().st_size != SMOKE_SIZE_BYTES:
        raise RuntimeError("frozen Xet smoke size mismatch")
    if _sha256(downloaded_path) != SMOKE_SHA256:
        raise RuntimeError("frozen Xet smoke checksum mismatch")
    if (smoke_dir / ".cache").exists():
        raise RuntimeError("frozen Xet smoke retained Hugging Face staging metadata")

    http_dir = output_dir / "httpx"
    http_dir.mkdir(parents=True, exist_ok=True)
    http_path = http_dir / SMOKE_REMOTE_PATH
    http_url = (
        f"https://huggingface.co/{SMOKE_REPO_ID}/resolve/" f"{SMOKE_REVISION}/{SMOKE_REMOTE_PATH}"
    )
    http_started_at = time.monotonic()
    with httpx.Client(follow_redirects=True, timeout=30) as client:
        with client.stream("GET", http_url) as response:
            response.raise_for_status()
            with http_path.open("wb") as handle:
                for chunk in response.iter_bytes():
                    handle.write(chunk)
    http_elapsed_seconds = time.monotonic() - http_started_at
    if http_path.stat().st_size != SMOKE_SIZE_BYTES or _sha256(http_path) != SMOKE_SHA256:
        raise RuntimeError("httpx comparison download validation failed")

    baseline_children = {child.pid for child in psutil.Process().children(recursive=True)}
    cancel_event = threading.Event()
    cancel_progress: list[HuggingFaceDownloadProgress] = []
    cancel_dir = output_dir / "cancel-xet"
    cancel_task = asyncio.create_task(
        adapter.download(
            HuggingFaceDownloadRequest(
                repo_id=CANCEL_REPO_ID,
                revision=CANCEL_REVISION,
                remote_path=CANCEL_REMOTE_PATH,
                local_dir=cancel_dir,
                expected_size_bytes=CANCEL_SIZE_BYTES,
            ),
            cancel_event=cancel_event,
            on_progress=cancel_progress.append,
        )
    )
    helper_pids: set[int] = set()
    network_observed = False
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        children = psutil.Process().children(recursive=True)
        helper_pids.update(child.pid for child in children if child.pid not in baseline_children)
        for child in children:
            if child.pid in helper_pids:
                try:
                    network_observed = network_observed or any(
                        connection.status == psutil.CONN_ESTABLISHED
                        for connection in child.net_connections(kind="inet")
                    )
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    pass
        if cancel_progress or network_observed:
            break
        await asyncio.sleep(0.05)
    cancel_event.set()
    try:
        await asyncio.wait_for(cancel_task, timeout=10)
    except LocalSTTDownloadPortCancelled:
        pass
    else:
        raise RuntimeError("frozen Xet cancellation smoke did not cancel")
    await asyncio.sleep(0.2)
    orphan_pids = [pid for pid in helper_pids if psutil.pid_exists(pid)]
    if orphan_pids:
        raise RuntimeError(f"frozen Xet cancellation left helper processes: {orphan_pids}")

    return {
        "contract_version": 1,
        "packaged_executable": str(executable),
        "hf_xet_high_performance_inherited": os.environ.get("HF_XET_HIGH_PERFORMANCE"),
        "small_xet": {
            "repo_id": SMOKE_REPO_ID,
            "revision": SMOKE_REVISION,
            "remote_path": SMOKE_REMOTE_PATH,
            "size_bytes": SMOKE_SIZE_BYTES,
            "sha256": SMOKE_SHA256,
            "elapsed_seconds": elapsed_seconds,
            "throughput_mib_per_second": SMOKE_SIZE_BYTES / 1024 / 1024 / elapsed_seconds,
            "progress_updates": len(progress),
        },
        "httpx_comparison": {
            "order": "xet_then_httpx",
            "size_bytes": SMOKE_SIZE_BYTES,
            "elapsed_seconds": http_elapsed_seconds,
            "throughput_mib_per_second": (SMOKE_SIZE_BYTES / 1024 / 1024 / http_elapsed_seconds),
        },
        "cancellation": {
            "repo_id": CANCEL_REPO_ID,
            "helper_pids": sorted(helper_pids),
            "network_observed": network_observed,
            "progress_observed": bool(cancel_progress),
            "orphan_pids": orphan_pids,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    executable = args.executable.resolve()
    output_dir = args.output_dir.resolve()
    xet_hash = _assert_xet_metadata()
    report = asyncio.run(_run(executable, output_dir))
    report["xet_hash"] = xet_hash
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
