from __future__ import annotations

import re
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from puripuly_heart.core.local_asr.local_stt_download_port import LocalSTTDownloadPortError
from puripuly_heart.core.network_clients import external_client


def download_public_file(payload: dict, *, on_progress, client_factory=external_client) -> Path:
    endpoint = payload["endpoint"]
    if endpoint != "https://hf-mirror.com":
        raise ValueError("Unsupported mirror")
    url = f"{endpoint}/{payload['repo_id']}/resolve/{payload['revision']}/{payload['remote_path']}"
    target = Path(payload["local_dir"]) / payload["remote_path"]
    expected = int(payload["expected_size_bytes"])
    partial = target.with_suffix(target.suffix + ".partial")
    with client_factory(timeout=httpx.Timeout(30, connect=15), follow_redirects=False) as client:
        for attempt in range(3):
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > expected:
                partial.unlink()
                offset = 0
            if offset == expected:
                partial.replace(target)
                on_progress(
                    {"type": "progress", "downloaded_bytes": offset, "total_bytes": expected}
                )
                return target
            next_url = url
            try:
                for hop in range(6):
                    if (
                        urlsplit(next_url).scheme != "https"
                        or urlsplit(next_url).hostname != "hf-mirror.com"
                    ):
                        raise LocalSTTDownloadPortError(
                            "Mirror redirected overseas; change source or import locally",
                            failure_code="mirror_redirect_external",
                        )
                    headers = {"Range": f"bytes={offset}-", "Accept-Encoding": "identity"}
                    with client.stream("GET", next_url, headers=headers) as response:
                        if response.is_redirect:
                            next_url = str(response.next_request.url)
                            continue
                        response.raise_for_status()
                        if response.status_code == 206:
                            match = re.fullmatch(
                                r"bytes (\d+)-(\d+)/(\d+)",
                                response.headers.get("content-range", ""),
                            )
                            if not match or tuple(map(int, match.groups())) != (
                                offset,
                                expected - 1,
                                expected,
                            ):
                                raise LocalSTTDownloadPortError(
                                    "Invalid server Range response", failure_code="range_invalid"
                                )
                        elif response.status_code == 200:
                            offset = 0
                        else:
                            raise LocalSTTDownloadPortError(
                                "Invalid download response", failure_code="range_invalid"
                            )
                        if (
                            response.headers.get("content-length")
                            and int(response.headers["content-length"]) != expected - offset
                        ):
                            raise LocalSTTDownloadPortError(
                                "Unexpected file size", failure_code="file_size_mismatch"
                            )
                        last_update = 0.0
                        with partial.open("ab" if offset else "wb") as handle:
                            for chunk in response.iter_bytes(chunk_size=256 * 1024):
                                offset += len(chunk)
                                if offset > expected:
                                    raise LocalSTTDownloadPortError(
                                        "File exceeds pinned size",
                                        failure_code="file_size_mismatch",
                                    )
                                handle.write(chunk)
                                if time.monotonic() - last_update >= 0.25:
                                    on_progress(
                                        {
                                            "type": "progress",
                                            "downloaded_bytes": offset,
                                            "total_bytes": expected,
                                        }
                                    )
                                    last_update = time.monotonic()
                        if offset != expected:
                            raise httpx.ReadError("Incomplete public file")
                        partial.replace(target)
                        on_progress(
                            {
                                "type": "progress",
                                "downloaded_bytes": offset,
                                "total_bytes": expected,
                            }
                        )
                        return target
                raise LocalSTTDownloadPortError(
                    "Too many redirects", failure_code="mirror_unavailable"
                )
            except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as error:
                status = (
                    error.response.status_code if isinstance(error, httpx.HTTPStatusError) else None
                )
                retryable = status is None or status in {408, 429, 500, 502, 503, 504}
                if attempt == 2 or not retryable:
                    raise LocalSTTDownloadPortError(
                        "Public download failed; retry or change source",
                        failure_code=(
                            "download_timeout"
                            if isinstance(error, httpx.TimeoutException)
                            else "mirror_unavailable"
                        ),
                        status_code=status,
                    ) from error
                time.sleep(attempt + 1)
    raise LocalSTTDownloadPortError("Public download failed")
