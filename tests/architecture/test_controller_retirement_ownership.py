from __future__ import annotations

import ast

import pytest

from tests.helpers.ast_sources import imported_modules
from tests.helpers.paths import SOURCE_ROOT

CONTROLLER_PATH = SOURCE_ROOT / "composition" / "application_runtime.py"

CASES = [
    {
        "id": "managed_auth",
        "owner_paths": (
            "app/services/managed/managed_auth.py",
            "app/wiring/wiring_managed_auth_factory.py",
            "app/wiring/wiring_managed_account.py",
        ),
        "forbidden_prefixes": ("puripuly_heart.ui",),
        "forbidden_exact": (),
        "retired_methods": frozenset(
            {
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
            }
        ),
        "retired_fields": frozenset(
            {
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
            }
        ),
    },
    {
        "id": "managed_usage",
        "owner_paths": ("app/services/managed/managed_usage.py",),
        "forbidden_prefixes": ("puripuly_heart.ui",),
        "forbidden_exact": ("puripuly_heart.config.settings",),
        "retired_methods": frozenset(
            {
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
            }
        ),
        "retired_fields": frozenset(
            {
                "_managed_status_refresh_owner",
                "_managed_trial_usage_metadata",
                "_managed_trial_usage_metadata_entitlement_ref",
                "_talk_together_pass_status",
                "_talk_together_pass_status_key",
            }
        ),
    },
    {
        "id": "translation_enable",
        "owner_paths": (
            "app/services/translation_enable.py",
            "app/wiring/wiring_managed_auth_factory.py",
        ),
        "forbidden_prefixes": ("puripuly_heart.ui",),
        "forbidden_exact": (),
        "retired_methods": frozenset(
            {
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
            }
        ),
        "retired_fields": frozenset(
            {
                "_translation_toggle_intent_enabled",
                "_translation_toggle_generation",
            }
        ),
    },
]


def _controller_methods_and_fields() -> tuple[set[str], set[str]]:
    tree = ast.parse(CONTROLLER_PATH.read_text(encoding="utf-8"))
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
    return methods, fields


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_owner_does_not_import_ui_or_controller(case) -> None:
    for relative_path in case["owner_paths"]:
        imports = imported_modules(SOURCE_ROOT / relative_path)
        assert not any(
            name.startswith(prefix) for name in imports for prefix in case["forbidden_prefixes"]
        )
        assert not set(case["forbidden_exact"]) & imports


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_controller_no_longer_owns_retired_state(case) -> None:
    methods, fields = _controller_methods_and_fields()
    assert case["retired_methods"].isdisjoint(methods)
    assert case["retired_fields"].isdisjoint(fields)
