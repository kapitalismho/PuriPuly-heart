"""Auto-update checker for GitHub Releases.

Checks for new releases on GitHub and provides update information.
Network failures are handled gracefully without affecting the main application.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx

from ajebal_daera_translator import __version__, GITHUB_REPO

logger = logging.getLogger(__name__)

GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
GITHUB_RELEASES_URL = f"https://github.com/{GITHUB_REPO}/releases/latest"


@dataclass(slots=True)
class UpdateInfo:
    """Information about an available update."""
    version: str
    download_url: str
    release_notes: str


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
    
    Returns UpdateInfo if a new version is available, None otherwise.
    Network errors are silently ignored (returns None).
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                GITHUB_API_URL,
                headers={"Accept": "application/vnd.github.v3+json"},
                follow_redirects=True,
            )
            
            if resp.status_code != 200:
                logger.debug(f"GitHub API returned {resp.status_code}")
                return None
            
            data: dict[str, Any] = resp.json()
            latest_version = data.get("tag_name", "").lstrip("v")
            
            if not latest_version:
                return None
            
            if not _is_newer(latest_version, __version__):
                logger.debug(f"Current version {__version__} is up to date")
                return None
            
            # Find installer download URL
            download_url = GITHUB_RELEASES_URL
            for asset in data.get("assets", []):
                name = asset.get("name", "")
                if name.endswith(".exe") or name.endswith(".zip"):
                    download_url = asset.get("browser_download_url", download_url)
                    break
            
            logger.info(f"New version available: {latest_version}")
            return UpdateInfo(
                version=latest_version,
                download_url=download_url,
                release_notes=data.get("body", ""),
            )
            
    except httpx.TimeoutException:
        logger.debug("Update check timed out")
        return None
    except Exception as exc:
        logger.debug(f"Update check failed: {exc}")
        return None
