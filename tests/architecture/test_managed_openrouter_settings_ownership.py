from tests.helpers.ast_sources import method_source
from tests.helpers.paths import REPO_ROOT

UI_RUNTIME_PATH = REPO_ROOT / "src" / "puripuly_heart" / "app" / "adapters" / "ui_runtime.py"
APP_BRANCHES_PATH = REPO_ROOT / "tests" / "ui" / "test_app_branches.py"


def _adapter_method_source(method_name: str) -> str:
    return method_source(UI_RUNTIME_PATH, "UiProviderRuntimeAdapter", method_name)


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
