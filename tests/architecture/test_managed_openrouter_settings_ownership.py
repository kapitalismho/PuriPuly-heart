import ast

from tests.helpers.paths import REPO_ROOT

UI_RUNTIME_PATH = REPO_ROOT / "src" / "puripuly_heart" / "app" / "adapters" / "ui_runtime.py"
APP_BRANCHES_PATH = REPO_ROOT / "tests" / "ui" / "test_app_branches.py"


def _adapter_method_source(method_name: str) -> str:
    source = UI_RUNTIME_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    adapter = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "UiProviderRuntimeAdapter"
    )
    method = next(
        node
        for node in adapter.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    return ast.get_source_segment(source, method)


def test_ui_byok_target_projection_is_only_config_projection_delegate() -> None:
    method = _adapter_method_source("build_managed_openrouter_byok_target_settings")

    assert method.count("build_managed_openrouter_byok_target_settings(") == 1
    assert "self.build_byok_target_settings(self.settings.current)" in method
    assert "deepcopy" not in method
    assert "profile_for_alias" not in method
    assert "OpenRouterSelectionAlias" not in method


def test_app_branches_do_not_import_or_rebind_controller_implementation() -> None:
    source = APP_BRANCHES_PATH.read_text(encoding="utf-8")

    assert "MethodType" not in source
