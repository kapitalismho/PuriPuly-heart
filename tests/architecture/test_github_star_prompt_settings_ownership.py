from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTROLLER_PATH = ROOT / "src" / "puripuly_heart" / "ui" / "controller.py"
OWNER_PATH = ROOT / "src" / "puripuly_heart" / "app" / "services" / "github_star_prompt_settings.py"


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
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == method_name
    )
    return ast.get_source_segment(source, method) or ""


def test_controller_github_prompt_methods_delegate_to_settings_owner_composition() -> None:
    source = CONTROLLER_PATH.read_text(encoding="utf-8")
    getter = _controller_method_source("_get_github_star_prompt_owner")

    assert "compose_github_star_prompt_owner(" in getter
    assert "build_ui_prompt_clipboard_state_settings_path_patch" not in source
    assert "_persist_order24_state_mutation" not in source
    assert "_github_star_prompt_translation_connection_for" not in source
    assert "_github_star_prompt_has_managed_connection" not in source
    assert "_github_star_prompt_has_user_owned_cloud_connection" not in source


def test_github_prompt_settings_owner_has_no_ui_or_controller_dependency() -> None:
    source = OWNER_PATH.read_text(encoding="utf-8")

    assert "puripuly_heart.ui" not in source
    assert "GuiController" not in source
    assert "SettingsOwner" in source
    assert "UiPromptClipboardStateSettingsMutation" in source
