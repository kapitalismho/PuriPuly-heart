from __future__ import annotations

from puripuly_heart.ui.components.debug_preview_panel import DebugPreviewPanel

CURATED_ACTION_KEYS: tuple[str, ...] = (
    "display_turn_cycle",
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
OPTIONAL_HTTP_EXTENSION_ACTION = "http_extension_form"
EXTENDED_DIAGNOSTIC_ACTION_KEYS: tuple[str, ...] = (
    "brake_notice",
    "revoked_notice",
    "pkce_failure",
    "capture_fault_cycle",
    "stt_fault_cycle",
    "audio_fault_clear",
    "gpu_state_cycle",
    "stt_loading_button_cycle",
)


def _panel(**overrides: object) -> DebugPreviewPanel:
    def noop() -> None:
        return None

    callbacks: dict[str, object] = {
        f"on_{key}": noop
        for key in CURATED_ACTION_KEYS
        + (OPTIONAL_HTTP_EXTENSION_ACTION,)
        + EXTENDED_DIAGNOSTIC_ACTION_KEYS
    }
    callbacks.update(overrides)
    return DebugPreviewPanel(**callbacks)  # type: ignore[arg-type]


def _keys(panel: DebugPreviewPanel) -> tuple[str, ...]:
    return tuple(action.key for action in panel._actions)


def _extended_keys(panel: DebugPreviewPanel) -> tuple[str, ...]:
    return tuple(action.key for action in panel._extended_actions)


def test_curated_and_extended_action_orders_remain_distinct() -> None:
    panel = _panel()

    assert _keys(panel) == CURATED_ACTION_KEYS + (OPTIONAL_HTTP_EXTENSION_ACTION,)
    assert _extended_keys(panel) == EXTENDED_DIAGNOSTIC_ACTION_KEYS
    assert panel._extended_actions_container.visible is False


def test_optional_http_extension_action_is_omitted_without_a_callback() -> None:
    panel = _panel(on_http_extension_form=None)

    assert _keys(panel) == CURATED_ACTION_KEYS
    assert _extended_keys(panel) == EXTENDED_DIAGNOSTIC_ACTION_KEYS


def test_extended_diagnostics_are_omitted_when_no_extended_callbacks_exist() -> None:
    panel = _panel(**{f"on_{key}": None for key in EXTENDED_DIAGNOSTIC_ACTION_KEYS})

    assert _keys(panel) == CURATED_ACTION_KEYS + (OPTIONAL_HTTP_EXTENSION_ACTION,)
    assert _extended_keys(panel) == ()
    assert panel._extended_toggle_button is None
