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


def test_managed_usage_owner_is_independent_of_controller_ui_and_settings_shapes() -> None:
    owner_path = REPO_ROOT / "src" / "puripuly_heart" / "app" / "services" / "managed_usage.py"

    imports = _imports(owner_path)

    assert not any(name.startswith("puripuly_heart.ui") for name in imports)
    assert "puripuly_heart.config.settings" not in imports


def test_controller_no_longer_owns_managed_usage_state_or_refresh_algorithms() -> None:
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
        "_managed_identity_scope",
        "_current_owned_referral_id",
        "_talk_together_pass_cache_key",
        "_clear_talk_together_pass_status_cache",
        "_cached_talk_together_pass_status_for",
        "_managed_key_card_visible_from_settings",
        "_refresh_managed_status_best_effort",
        "_schedule_owned_referral_id_status_refresh",
        "_get_managed_status_refresh_owner",
        "_clear_managed_trial_usage_metadata_cache",
        "_sync_managed_trial_usage_metadata_scope",
        "_schedule_managed_trial_usage_refresh",
        "_refresh_managed_trial_usage_state",
        "_refresh_managed_trial_usage_state_impl",
        "_set_managed_usage_view_state",
        "_managed_usage_state",
        "_fetch_managed_usage_metadata",
        "_managed_usage_auto_show_founder_letter",
        "_managed_usage_warning_sink",
    }.isdisjoint(methods)
    assert {
        "_managed_status_refresh_owner",
        "_managed_trial_usage_metadata",
        "_managed_trial_usage_metadata_entitlement_ref",
        "_talk_together_pass_status",
        "_talk_together_pass_status_key",
    }.isdisjoint(fields)
