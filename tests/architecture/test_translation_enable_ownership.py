from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module)
    return names


def test_translation_enable_owner_and_adapter_do_not_import_ui_or_controller() -> None:
    paths = (
        REPO_ROOT / "src" / "puripuly_heart" / "app" / "services" / "translation_enable.py",
        REPO_ROOT / "src" / "puripuly_heart" / "app" / "wiring_managed_auth_factory.py",
    )

    for path in paths:
        imports = _imports(path)
        assert not any(name.startswith("puripuly_heart.ui") for name in imports)
        assert "puripuly_heart.ui.controller" not in imports


def test_controller_no_longer_owns_translation_enable_state_or_algorithms() -> None:
    controller_path = REPO_ROOT / "src" / "puripuly_heart" / "ui" / "controller.py"
    tree = ast.parse(controller_path.read_text(encoding="utf-8"))
    methods = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    fields = {
        node.target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }

    assert {
        "_handle_managed_translation_enable",
        "_show_founder_letter_dialog",
        "_disable_translation_for_managed_exhaustion",
        "_should_route_managed_trans_to_founder_letter",
        "_record_translation_toggle_intent",
        "_translation_toggle_intent_matches",
        "_should_show_managed_auth_pending_before_prepare",
        "_managed_auth_claim_guard_for_settings",
        "_managed_china_auth_relevant_for_translation_enable",
        "_show_qq_managed_auth_dialog",
    }.isdisjoint(methods)
    assert {
        "_translation_toggle_intent_enabled",
        "_translation_toggle_generation",
    }.isdisjoint(fields)
