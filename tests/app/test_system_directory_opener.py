from __future__ import annotations

from pathlib import Path

import pytest

from puripuly_heart.app.adapters import system_directory_opener as opener_module
from puripuly_heart.app.adapters.system_directory_opener import SystemDirectoryOpener
from puripuly_heart.app.services.http_extension_registry import (
    HttpExtensionRegistryService,
)
from puripuly_heart.core.http_extensions import HttpExtensionRegistry


@pytest.mark.parametrize(
    ("platform", "executable"),
    [
        ("win32", "explorer"),
        ("darwin", "open"),
        ("linux", "xdg-open"),
    ],
)
def test_system_directory_opener_uses_platform_command(
    platform: str,
    executable: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = Path("extensions")
    calls: list[list[str]] = []
    monkeypatch.setattr(opener_module.subprocess, "Popen", lambda command: calls.append(command))

    SystemDirectoryOpener(platform).open(directory)

    assert calls == [[executable, str(directory)]]


def test_registry_service_creates_and_opens_its_resolved_directory(tmp_path: Path) -> None:
    directory = tmp_path / "extensions"
    calls: list[Path] = []
    service = HttpExtensionRegistryService(
        HttpExtensionRegistry(directory),
        type("DirectoryOpener", (), {"open": lambda _self, value: calls.append(value)})(),
    )

    service.open_directory()

    assert directory.is_dir()
    assert calls == [directory]
