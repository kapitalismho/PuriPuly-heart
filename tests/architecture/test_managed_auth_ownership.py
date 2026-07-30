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


def test_managed_auth_owner_and_adapter_do_not_import_ui_or_controller() -> None:
    paths = (
        REPO_ROOT / "src" / "puripuly_heart" / "app" / "services" / "managed" / "managed_auth.py",
        REPO_ROOT / "src" / "puripuly_heart" / "app" / "wiring_managed_auth_factory.py",
        REPO_ROOT / "src" / "puripuly_heart" / "app" / "wiring_managed_account.py",
    )

    for path in paths:
        imports = _imports(path)
        assert not any(name.startswith("puripuly_heart.ui") for name in imports)


def test_controller_no_longer_owns_managed_auth_flow_state_or_algorithms() -> None:
    controller_path = (
        REPO_ROOT / "src" / "puripuly_heart" / "composition" / "application_runtime.py"
    )
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
        "_discord_auth_message_key",
        "_discord_release_service_supports_transaction_auth",
        "_start_discord_managed_auth_via_release_service",
        "_get_managed_auth_runtime_adapter",
        "_ensure_managed_auth_runtime",
        "_create_managed_openrouter_release_service",
        "_replace_managed_openrouter_release_service",
        "_managed_openrouter_release_settings",
        "_create_openrouter_pkce_client",
        "_on_discord_managed_auth_callback_received",
    }.isdisjoint(methods)
    assert {
        "_managed_trial_pending_auth",
        "_discord_managed_auth_in_progress",
        "_discord_managed_auth_callback_received_hook",
        "last_discord_managed_auth_referral_bonus_applied",
        "telemetry_client",
        "_managed_openrouter_release_service",
        "_managed_auth_runtime_adapter",
        "_managed_auth_owner",
        "_managed_translation_runtime_adapter",
        "_translation_enable_owner",
        "_managed_usage_owner",
        "_openrouter_pkce_flow_owner",
        "_openrouter_pkce_application_owner",
    }.isdisjoint(fields)
