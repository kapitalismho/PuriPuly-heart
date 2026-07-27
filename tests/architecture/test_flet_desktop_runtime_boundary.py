from __future__ import annotations

import ast
from pathlib import Path

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
    source = ADAPTER_PATH.read_text(encoding="utf-8")

    assert source.count('"__locate_and_unpack_flet_view"') == 1
    assert source.count("open_flet_view_async") == 3
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
