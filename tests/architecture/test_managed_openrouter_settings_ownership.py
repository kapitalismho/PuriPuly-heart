import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTROLLER_PATH = REPO_ROOT / "src" / "puripuly_heart" / "ui" / "controller.py"
APP_BRANCHES_PATH = REPO_ROOT / "tests" / "ui" / "test_app_branches.py"


def _controller_method_source(method_name: str) -> str:
    source = CONTROLLER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    controller = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "GuiController"
    )
    method = next(
        node
        for node in controller.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    return ast.get_source_segment(source, method)


def test_controller_byok_target_projection_is_only_config_projection_delegate() -> None:
    method = _controller_method_source("build_managed_openrouter_byok_target_settings")

    assert method.count("build_managed_openrouter_byok_target_settings(") == 2
    assert "self.settings" in method
    assert "deepcopy" not in method
    assert "profile_for_alias" not in method
    assert "OpenRouterSelectionAlias" not in method


def test_app_branches_do_not_import_or_rebind_controller_implementation() -> None:
    source = APP_BRANCHES_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert not any(
        isinstance(node, ast.ImportFrom) and node.module == "puripuly_heart.ui.controller"
        for node in ast.walk(tree)
    )
    assert "MethodType" not in source
    assert "GuiController" not in source
