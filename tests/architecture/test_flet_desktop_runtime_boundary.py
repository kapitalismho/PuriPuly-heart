from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "src" / "puripuly_heart"
ADAPTER_PATH = SOURCE_ROOT / "ui" / "flet_desktop_runtime.py"


def _imports_flet_desktop(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(
            alias.name == "flet_desktop" for alias in node.names
        ):
            return True
        if isinstance(node, ast.ImportFrom) and node.module == "flet_desktop":
            return True
    return False


def test_flet_desktop_dependency_is_confined_to_one_ui_runtime_adapter() -> None:
    importers = {
        path.relative_to(REPO_ROOT).as_posix()
        for path in SOURCE_ROOT.rglob("*.py")
        if _imports_flet_desktop(path)
    }

    assert importers == {"src/puripuly_heart/ui/flet_desktop_runtime.py"}


def test_flet_desktop_private_hook_inventory_is_exact_and_explicit() -> None:
    from puripuly_heart.ui.flet_desktop_runtime import REQUIRED_FLET_DESKTOP_HOOKS

    assert REQUIRED_FLET_DESKTOP_HOOKS == (
        "__locate_and_unpack_flet_view",
        "open_flet_view_async",
    )

    source = ADAPTER_PATH.read_text(encoding="utf-8")
    for hook in REQUIRED_FLET_DESKTOP_HOOKS:
        assert hook in source
    assert "patch_hidden_view_launcher" in source
    assert "open_hidden_view" in source


def test_the_private_hooks_the_adapter_depends_on_exist_in_the_installed_runtime() -> None:
    """The adapter reaches two `flet_desktop` internals. This fails at import time on a runtime that
    renames or removes them, instead of failing when the overlay is first revealed."""
    import flet_desktop

    assert callable(getattr(flet_desktop, "__locate_and_unpack_flet_view"))
    assert callable(flet_desktop.open_flet_view_async)


def test_the_launcher_patch_restores_the_original_hook() -> None:
    import flet_desktop

    from puripuly_heart.ui.flet_desktop_runtime import patch_hidden_view_launcher

    original = flet_desktop.open_flet_view_async
    with patch_hidden_view_launcher():
        assert flet_desktop.open_flet_view_async is not original
    assert flet_desktop.open_flet_view_async is original


def test_a_runtime_without_the_hooks_is_reported_by_name_instead_of_attribute_error() -> None:
    from puripuly_heart.ui.flet_desktop_runtime import (
        UnsupportedFletDesktopRuntimeError,
        require_flet_desktop_hooks,
    )

    class _RenamedRuntime:
        pass

    with pytest.raises(UnsupportedFletDesktopRuntimeError) as excinfo:
        require_flet_desktop_hooks(_RenamedRuntime())

    message = str(excinfo.value)
    assert "__locate_and_unpack_flet_view" in message
    assert "open_flet_view_async" in message
    assert "flet-desktop" in message


def test_a_runtime_missing_only_one_hook_names_only_that_hook() -> None:
    from puripuly_heart.ui.flet_desktop_runtime import (
        UnsupportedFletDesktopRuntimeError,
        require_flet_desktop_hooks,
    )

    class _PartialRuntime:
        @staticmethod
        def open_flet_view_async() -> None:
            return None

    with pytest.raises(UnsupportedFletDesktopRuntimeError) as excinfo:
        require_flet_desktop_hooks(_PartialRuntime())

    message = str(excinfo.value)
    assert "__locate_and_unpack_flet_view" in message
    assert "open_flet_view_async" not in message.split("does not expose", 1)[1].split(",", 1)[0]


def test_the_launcher_patch_refuses_an_unsupported_runtime_without_patching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import types

    from puripuly_heart.ui.flet_desktop_runtime import (
        UnsupportedFletDesktopRuntimeError,
        patch_hidden_view_launcher,
    )

    stub = types.ModuleType("flet_desktop")
    monkeypatch.setitem(sys.modules, "flet_desktop", stub)

    with pytest.raises(UnsupportedFletDesktopRuntimeError):
        with patch_hidden_view_launcher():
            pass

    assert not hasattr(stub, "open_flet_view_async")


async def test_opening_the_hidden_view_on_an_unsupported_runtime_fails_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import types

    from puripuly_heart.ui.flet_desktop_runtime import (
        UnsupportedFletDesktopRuntimeError,
        open_hidden_view,
    )

    stub = types.ModuleType("flet_desktop")
    monkeypatch.setitem(sys.modules, "flet_desktop", stub)

    with pytest.raises(UnsupportedFletDesktopRuntimeError):
        await open_hidden_view("http://127.0.0.1:0", None, True)
