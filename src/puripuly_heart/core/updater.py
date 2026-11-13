"""Auto-update checker for GitHub Releases.

Checks for new releases on GitHub and provides update information.
Network failures are handled gracefully without affecting the main application.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, replace
from typing import Any

import httpx

from puripuly_heart import GITHUB_REPO, __version__
from puripuly_heart.core import external_network, network_clients
from puripuly_heart.core.diagnostics import emit_failure, exception_diagnostics
from puripuly_heart.core.messages import ErrorDiagnostics
from puripuly_heart.core.owned_thread import run_owned_thread_call

logger = logging.getLogger(__name__)

GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
GITHUB_RELEASES_URL = f"https://github.com/{GITHUB_REPO}/releases/latest"


@dataclass(slots=True)
class UpdateInfo:
    """Information about an available update."""

    version: str
    download_url: str
    release_notes: str


class UpdateCheckError(RuntimeError):
    def __init__(self, diagnostics: ErrorDiagnostics) -> None:
        self.failure_diagnostics = diagnostics
        super().__init__("Update availability could not be checked")


def _parse_version(version_str: str) -> tuple[int, ...]:
    """Parse version string like '0.1.0' or 'v0.1.0' into tuple of ints."""
    clean = version_str.lstrip("v")
    parts = clean.split(".")
    result = []
    for part in parts:
        # Handle pre-release versions like '0.1.0-beta'
        num_part = part.split("-")[0]
        try:
            result.append(int(num_part))
        except ValueError:
            result.append(0)
    return tuple(result)


def _is_newer(remote: str, current: str) -> bool:
    """Check if remote version is newer than current."""
    return _parse_version(remote) > _parse_version(current)


async def check_for_update() -> UpdateInfo | None:
    """Check GitHub for a newer release.

    Returns UpdateInfo for a newer release and None after a successful no-update observation.
    """
    status_code: int | None = None
    try:
        tls, policy = await run_owned_thread_call(
            lambda: (external_network.select_tls(), external_network.ProxyPolicy())
        )
        async with network_clients.external_async_client(
            tls=tls, policy=policy, timeout=5.0
        ) as client:
            resp = await client.get(
                GITHUB_API_URL,
                headers={"Accept": "application/vnd.github.v3+json"},
                follow_redirects=True,
            )
            status_code = resp.status_code
            if resp.status_code != 200:
                raise ValueError("Update service returned an unsuccessful status")

            data: dict[str, Any] = resp.json()
            if not isinstance(data, dict) or not isinstance(data.get("tag_name"), str):
                raise ValueError("Update service returned an invalid release")
            latest_version = data["tag_name"].lstrip("v")
            if re.fullmatch(r"[0-9]+(?:\.[0-9]+)*(?:-[0-9A-Za-z.-]+)?", latest_version) is None:
                raise ValueError("Update service returned an invalid release")
            if not _is_newer(latest_version, __version__):
                return None

            # Find installer download URL
            download_url = GITHUB_RELEASES_URL
            for asset in data.get("assets", []):
                name = asset.get("name", "")
                if name.endswith(".exe") or name.endswith(".zip"):
                    download_url = asset.get("browser_download_url", download_url)
                    break

            return UpdateInfo(
                version=latest_version,
                download_url=download_url,
                release_notes=data.get("body", ""),
            )

    except Exception as exc:
        category = (
            "timeout"
            if isinstance(exc, httpx.TimeoutException)
            else "network" if status_code is None else "invalid_response"
        )
        diagnostics = replace(
            exception_diagnostics(
                exc,
                component="updater",
                operation="update_check",
                code="updater_check_failed",
                category=category,
            ),
            status_code=status_code,
        )
        emit_failure(logger, diagnostics, severity="warning", disposition="degraded")
        raise UpdateCheckError(diagnostics) from exc
