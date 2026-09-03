from __future__ import annotations

# ruff: noqa: I001

import json
from pathlib import Path

import pytest

pytest.importorskip("flet")

from puripuly_heart.ui.components.debug_preview_panel import (  # noqa: E402
    DEBUG_PREVIEW_ACTION_ROW_HEIGHT,
    DEBUG_PREVIEW_PANEL_DATA_KEY,
    DEBUG_PREVIEW_POPOVER_MAX_HEIGHT,
    DebugPreviewPanel,
)
import puripuly_heart.ui.components.debug_preview_panel as panel_module  # noqa: E402

PRIMARY_ACTION_KEYS = [
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
    "http_extension_form",
]
EXTENDED_ACTION_KEYS = [
    "brake_notice",
    "revoked_notice",
    "pkce_failure",
    "pkce_button_cycle",
    "capture_fault_cycle",
    "stt_fault_cycle",
    "audio_fault_clear",
    "gpu_state_cycle",
    "stt_loading_button_cycle",
]
DEBUG_PREVIEW_I18N_KEYS = {
    "debug_preview.button",
    "debug_preview.tooltip",
    "debug_preview.more_diagnostics",
    *(f"debug_preview.{key}" for key in PRIMARY_ACTION_KEYS + EXTENDED_ACTION_KEYS),
    "foundation.preview.title",
    "foundation.preview.body",
    "foundation.preview.ready",
    "foundation.preview.action",
    "foundation.preview.unavailable",
    "peer_translation_eula.body",
    "peer_translation_eula.accept",
    "peer_translation_eula.cancel",
    "peer_translation.disclosure",
}


def _callbacks(seen: list[str]):
    return {
        f"on_{action_key}": lambda action_key=action_key: seen.append(action_key)
        for action_key in PRIMARY_ACTION_KEYS + EXTENDED_ACTION_KEYS
    }


def _button_label(button) -> str:
    return button.content


def test_debug_preview_panel_starts_with_curated_actions_and_collapsed_diagnostics() -> None:
    seen: list[str] = []

    panel = DebugPreviewPanel(**_callbacks(seen))

    assert panel.data == DEBUG_PREVIEW_PANEL_DATA_KEY
    assert _button_label(panel._toggle_button) == panel_module.t("debug_preview.button")
    assert panel._toggle_button.tooltip == panel_module.t("debug_preview.tooltip")
    assert panel._popover.visible is False
    assert list(panel._action_buttons) == PRIMARY_ACTION_KEYS
    assert list(panel._extended_action_buttons) == EXTENDED_ACTION_KEYS
    assert panel._extended_actions_container.visible is False
    assert panel._popover.content.scroll == panel_module.ft.ScrollMode.AUTO
    assert panel._popover.content.height == min(
        (len(PRIMARY_ACTION_KEYS) + 1) * DEBUG_PREVIEW_ACTION_ROW_HEIGHT,
        DEBUG_PREVIEW_POPOVER_MAX_HEIGHT,
    )
    assert seen == []


def test_debug_preview_panel_expands_diagnostics_with_bounded_scroll_height() -> None:
    panel = DebugPreviewPanel(**_callbacks([]))

    panel._toggle_extended_actions(None)

    assert panel._extended_actions_container.visible is True
    assert panel._popover.content.height == DEBUG_PREVIEW_POPOVER_MAX_HEIGHT
    assert panel._popover.content.scroll == panel_module.ft.ScrollMode.AUTO

    panel._toggle_extended_actions(None)
    assert panel._extended_actions_container.visible is False
    assert panel._popover.content.height < DEBUG_PREVIEW_POPOVER_MAX_HEIGHT


def test_debug_preview_panel_omits_optional_http_and_extended_actions() -> None:
    callbacks = _callbacks([])
    callbacks["on_http_extension_form"] = None
    for key in EXTENDED_ACTION_KEYS:
        callbacks[f"on_{key}"] = None

    panel = DebugPreviewPanel(**callbacks)

    assert list(panel._action_buttons) == PRIMARY_ACTION_KEYS[:-1]
    assert panel._extended_action_buttons == {}
    assert panel._extended_toggle_button is None


def test_debug_preview_panel_toggle_shows_and_hides_popover() -> None:
    panel = DebugPreviewPanel(**_callbacks([]))

    panel._toggle(None)
    assert panel._popover.visible is True

    panel._toggle(None)
    assert panel._popover.visible is False


def test_debug_preview_panel_skips_update_when_detached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = DebugPreviewPanel(**_callbacks([]))

    def fail_update(self) -> None:
        _ = self
        raise AssertionError("detached debug preview panel should not update")

    monkeypatch.setattr(DebugPreviewPanel, "update", fail_update)

    panel._toggle(None)
    panel._toggle_extended_actions(None)
    panel.apply_locale()

    assert panel._popover.visible is True
    assert panel._extended_actions_container.visible is True


def test_debug_preview_panel_invokes_primary_and_extended_callbacks() -> None:
    seen: list[str] = []
    panel = DebugPreviewPanel(**_callbacks(seen))
    panel._toggle(None)

    for action_key in PRIMARY_ACTION_KEYS:
        panel._action_buttons[action_key].on_click(None)
    panel._toggle_extended_actions(None)
    for action_key in EXTENDED_ACTION_KEYS:
        panel._extended_action_buttons[action_key].on_click(None)

    assert seen == PRIMARY_ACTION_KEYS + EXTENDED_ACTION_KEYS
    assert panel._popover.visible is True


def test_debug_preview_panel_apply_locale_refreshes_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = DebugPreviewPanel(**_callbacks([]))
    monkeypatch.setattr(panel_module, "t", lambda key: f"label:{key}")

    panel.apply_locale()

    assert _button_label(panel._toggle_button) == "label:debug_preview.button"
    assert panel._toggle_button.tooltip == "label:debug_preview.tooltip"
    assert _button_label(panel._extended_toggle_button) == ("label:debug_preview.more_diagnostics")
    for action_key in PRIMARY_ACTION_KEYS:
        assert _button_label(panel._action_buttons[action_key]) == (
            f"label:debug_preview.{action_key}"
        )
    for action_key in EXTENDED_ACTION_KEYS:
        assert _button_label(panel._extended_action_buttons[action_key]) == (
            f"label:debug_preview.{action_key}"
        )


def test_debug_preview_panel_uses_flet_086_text_button_content_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[object] = []

    class ContentOnlyButton:
        def __init__(self, *, content, tooltip=None, on_click=None, style=None) -> None:
            self.content = content
            self.tooltip = tooltip
            self.on_click = on_click
            self.style = style
            created.append(self)

    monkeypatch.setattr(panel_module.ft, "TextButton", ContentOnlyButton)
    DebugPreviewPanel(**_callbacks([]))

    assert [button.content for button in created] == [
        panel_module.t("debug_preview.button"),
        *(panel_module.t(f"debug_preview.{key}") for key in PRIMARY_ACTION_KEYS),
        *(panel_module.t(f"debug_preview.{key}") for key in EXTENDED_ACTION_KEYS),
        panel_module.t("debug_preview.more_diagnostics"),
    ]


def test_debug_preview_i18n_keys_exist_in_all_locale_bundles() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    i18n_dir = repo_root / "src" / "puripuly_heart" / "data" / "i18n"

    for locale in ("en.json", "ko.json", "zh-CN.json", "ja.json", "ru.json"):
        bundle = json.loads((i18n_dir / locale).read_text(encoding="utf-8"))
        missing = DEBUG_PREVIEW_I18N_KEYS - set(bundle)
        assert not missing, f"{locale} missing {sorted(missing)}"


def test_debug_preview_panel_has_no_external_state_imports() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    source = (
        repo_root / "src" / "puripuly_heart" / "ui" / "components" / "debug_preview_panel.py"
    ).read_text(encoding="utf-8")

    forbidden_fragments = [
        "puripuly_heart.config.settings",
        "puripuly_heart.core.openrouter",
        "puripuly_heart.core.managed_openrouter",
        "SecretStore",
        "secrets",
        "broker",
        "webbrowser",
    ]
    for fragment in forbidden_fragments:
        assert fragment not in source
