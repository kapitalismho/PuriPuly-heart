from __future__ import annotations

from puripuly_heart.ui.components.debug_preview_panel import DebugPreviewPanel

CURATED_ACTION_KEYS: tuple[str, ...] = (
    "display_turn_cycle",
    "telemetry_consent",
    "peer_translation_eula",
    "discord_auth",
    "qq_auth",
    "qq_auth_recoverable_error",
    "qq_auth_translation_gated",
    "discord_callback_page",
    "founder_letter",
    "local_qwen_hallucination_modal",
    "github_star_snackbar",
    "talk_together_pass_invite_progress",
    "foundation_primitives",
)
OPTIONAL_ACTION_KEY = "http_extension_form"

RETIRED_ACTION_KEYS: frozenset[str] = frozenset(
    {
        "brake_notice",
        "revoked_notice",
        "pkce_failure",
        "capture_fault_cycle",
        "stt_fault_cycle",
        "audio_fault_clear",
        "gpu_state_cycle",
        "stt_loading_button_cycle",
        "display_color_scheme_cycle",
    }
)


def _panel(**overrides: object) -> DebugPreviewPanel:
    def noop() -> None:
        return None

    callbacks: dict[str, object] = {
        f"on_{key}": noop for key in CURATED_ACTION_KEYS + (OPTIONAL_ACTION_KEY,)
    }
    callbacks.update(overrides)
    return DebugPreviewPanel(**callbacks)  # type: ignore[arg-type]


def _keys(panel: DebugPreviewPanel) -> tuple[str, ...]:
    return tuple(action.key for action in panel._actions)


def test_action_order_follows_the_curated_importance_order() -> None:
    assert _keys(_panel()) == CURATED_ACTION_KEYS + (OPTIONAL_ACTION_KEY,)


def test_the_optional_http_extension_action_is_omitted_without_a_callback() -> None:
    assert _keys(_panel(on_http_extension_form=None)) == CURATED_ACTION_KEYS


def test_retired_actions_are_not_offered() -> None:
    assert RETIRED_ACTION_KEYS.isdisjoint(_keys(_panel()))


def test_the_display_preview_action_leads_the_popover() -> None:
    assert _keys(_panel())[0] == "display_turn_cycle"


def test_the_popover_is_bounded_and_scrollable() -> None:
    import flet as ft

    from puripuly_heart.ui.components import debug_preview_panel as panel_module

    panel = _panel()
    column = panel._popover.content

    assert column.scroll == ft.ScrollMode.AUTO
    assert column.height <= panel_module.DEBUG_PREVIEW_POPOVER_MAX_HEIGHT
