from __future__ import annotations

from puripuly_heart.ui.components.debug_preview_panel import DebugPreviewPanel

# Action key order rendered by the accepted canonical baseline at dev@3fb5ce83, in popover row order.
BASELINE_ACTION_KEYS: tuple[str, ...] = (
    "brake_notice",
    "revoked_notice",
    "github_star_snackbar",
    "founder_letter",
    "pkce_failure",
    "discord_auth",
    "qq_auth",
    "qq_auth_recoverable_error",
    "qq_auth_translation_gated",
    "discord_callback_page",
    "peer_translation_eula",
    "local_qwen_hallucination_modal",
    "talk_together_pass_invite_progress",
    "capture_fault_cycle",
    "stt_fault_cycle",
    "audio_fault_clear",
    "gpu_state_cycle",
    "stt_loading_button_cycle",
)

APPROVED_EXTRA_ACTION = "foundation_primitives"
APPROVED_HTTP_EXTENSION_ACTION = "http_extension_form"


def _panel(**overrides: object) -> DebugPreviewPanel:
    def noop() -> None:
        return None

    callbacks: dict[str, object] = {
        f"on_{key}": noop
        for key in BASELINE_ACTION_KEYS + (APPROVED_EXTRA_ACTION, APPROVED_HTTP_EXTENSION_ACTION)
    }
    callbacks.update(overrides)
    return DebugPreviewPanel(**callbacks)  # type: ignore[arg-type]


def _keys(panel: DebugPreviewPanel) -> tuple[str, ...]:
    return tuple(action.key for action in panel._actions)


def test_action_order_is_the_baseline_order_plus_the_approved_extras() -> None:
    keys = _keys(_panel())
    assert keys == BASELINE_ACTION_KEYS + (
        APPROVED_EXTRA_ACTION,
        APPROVED_HTTP_EXTENSION_ACTION,
    )


def test_the_optional_stt_loading_action_is_omitted_without_a_callback() -> None:
    keys = _keys(_panel(on_stt_loading_button_cycle=None))
    assert keys == BASELINE_ACTION_KEYS[:-1] + (
        APPROVED_EXTRA_ACTION,
        APPROVED_HTTP_EXTENSION_ACTION,
    )
