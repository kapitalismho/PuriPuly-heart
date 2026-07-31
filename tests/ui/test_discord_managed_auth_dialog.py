from __future__ import annotations

from collections.abc import Callable

import pytest

from tests.helpers.flet_page import DialogTrackingPage as DummyPage

pytest.importorskip("flet")

import flet as ft  # noqa: E402

import puripuly_heart.ui.components.discord_managed_auth_dialog as discord_module  # noqa: E402
from puripuly_heart.ui.components.discord_managed_auth_dialog import DiscordManagedAuthDialog
from puripuly_heart.ui.i18n import get_locale, set_locale, t
from puripuly_heart.ui.theme import (  # noqa: E402
    COLOR_NEUTRAL_DARK,
    COLOR_PRIMARY,
)


@pytest.fixture(autouse=True)
def restore_locale_after_test():
    previous_locale = get_locale()
    try:
        yield
    finally:
        set_locale(previous_locale)


class SnapshotOpenPage(DummyPage):
    def __init__(self) -> None:
        super().__init__()
        self.body_control_classes_at_open: list[str] = []

    def show_dialog(self, dialog) -> None:
        modal_content = dialog.content
        body_column = modal_content.content.controls[0]
        self.body_control_classes_at_open = [
            control.__class__.__name__ for control in body_column.controls
        ]
        super().show_dialog(dialog)


def _dialog(
    page: DummyPage,
    events: list[str] | None = None,
    *,
    on_reopen_browser: Callable[[], None] | None = None,
    on_cancel: Callable[[], None] | None = None,
) -> DiscordManagedAuthDialog:
    calls = events if events is not None else []
    return DiscordManagedAuthDialog(
        page,
        on_continue=lambda: calls.append("continue"),
        on_byok=lambda: calls.append("byok"),
        on_close=lambda: calls.append("close"),
        on_reopen_browser=on_reopen_browser or (lambda: calls.append("reopen")),
        on_cancel=on_cancel,
    )


def _dialog_without_reopen(
    page: DummyPage,
    events: list[str] | None = None,
) -> DiscordManagedAuthDialog:
    calls = events if events is not None else []
    return DiscordManagedAuthDialog(
        page,
        on_continue=lambda: calls.append("continue"),
        on_byok=lambda: calls.append("byok"),
        on_close=lambda: calls.append("close"),
    )


def _modal_content(page: DummyPage):
    return page.dialog.content


def _content_column(page: DummyPage):
    return _modal_content(page).content


def _body_column(page: DummyPage):
    for control in _content_column(page).controls:
        if control.__class__.__name__ == "Column":
            return control
    raise AssertionError("dialog content did not include a body column")


def _action_row(page: DummyPage):
    return _content_column(page).controls[-1]


def _button_text_size(button) -> int:
    return button.style.text_style.size


def test_discord_managed_auth_dialog_declares_initial_action_labels() -> None:
    page = DummyPage()

    dialog = _dialog(page)

    assert dialog.action_labels == [
        "discord_auth.close",
        "discord_auth.continue",
    ]


def test_discord_managed_auth_dialog_uses_warm_document_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_keys: list[str] = []

    def fake_t(key: str) -> str:
        requested_keys.append(key)
        return f"value:{key}"

    monkeypatch.setattr(discord_module, "t", fake_t)
    monkeypatch.setattr(discord_module, "create_glow_stack", lambda content: content)

    page = DummyPage()
    dialog = DiscordManagedAuthDialog(
        page,
        on_continue=lambda: None,
        on_byok=lambda: None,
        on_close=lambda: None,
        on_reopen_browser=lambda: None,
    )

    dialog.open()

    modal_content = _modal_content(page)
    assert "discord_auth.title" not in requested_keys
    assert "discord_auth.requirements" not in requested_keys
    assert "discord_auth.byok" not in requested_keys
    assert modal_content.width == 720
    assert modal_content.height is None

    body_column = _body_column(page)
    body_text = body_column.controls[0]
    action_row = _action_row(page)

    assert len(body_column.controls) == 2
    assert body_text.value == "value:discord_auth.body"
    assert body_text.size == 24
    assert body_text.selectable is True
    assert [button.__class__.__name__ for button in action_row.controls] == [
        "TextButton",
        "TextButton",
    ]
    assert [button.content for button in action_row.controls] == [
        "value:discord_auth.close",
        "value:discord_auth.continue",
    ]
    assert [_button_text_size(button) for button in action_row.controls] == [26, 26]
    assert [button.style.color[ft.ControlState.DEFAULT] for button in action_row.controls] == [
        COLOR_NEUTRAL_DARK,
        COLOR_NEUTRAL_DARK,
    ]
    assert [button.style.color[ft.ControlState.HOVERED] for button in action_row.controls] == [
        COLOR_PRIMARY,
        COLOR_PRIMARY,
    ]
    assert [button.style.animation_duration for button in action_row.controls] == [0, 0]


def test_discord_managed_auth_dialog_renders_optional_referral_id_field() -> None:
    set_locale("en")
    page = DummyPage()
    dialog = _dialog(page)

    dialog.open()

    field = dialog._referral_id_field
    toggle = dialog._referral_toggle
    label = dialog._referral_toggle_label
    assert field is not None
    assert toggle is not None
    assert label is not None
    assert field.label == t("discord_auth.referral_id.label")
    assert field.helper == t("discord_auth.referral_id.helper")
    assert getattr(field, "hint_text", None) in (None, "")
    assert field.value == ""
    assert field.visible is False
    assert label.value == t("discord_auth.referral_id.expand")
    assert label.size == 24
    assert dialog._continue_button is not None
    assert not getattr(dialog._continue_button, "disabled", False)


def test_discord_managed_auth_dialog_expands_referral_id_field_on_demand() -> None:
    set_locale("en")
    page = DummyPage()
    dialog = _dialog(page)
    dialog.open()

    assert dialog._referral_id_field is not None
    assert dialog._referral_toggle is not None
    assert dialog._referral_toggle_label is not None
    assert dialog._referral_id_field.visible is False

    dialog._referral_toggle.on_click(None)

    assert dialog._referral_id_field.visible is True
    assert dialog._referral_toggle_label.value == t("discord_auth.referral_id.collapse")
    dialog._referral_id_field.value = "7KQ9M2"
    assert dialog.referral_id == "7KQ9M2"

    dialog._referral_toggle.on_click(None)

    assert dialog._referral_id_field.visible is False
    assert dialog._referral_toggle_label.value == t("discord_auth.referral_id.expand")
    assert dialog.referral_id == "7KQ9M2"


def test_discord_managed_auth_dialog_scales_referral_id_field_content() -> None:
    page = DummyPage()
    dialog = _dialog(page)

    dialog.open()

    field = dialog._referral_id_field
    assert field is not None
    assert field.height is None
    assert field.dense is False
    assert field.text_size == 22
    assert field.content_padding is not None
    assert field.content_padding.left == 16
    assert field.content_padding.right == 16
    assert field.content_padding.top == 20
    assert field.content_padding.bottom == 20
    assert field.bgcolor is None
    assert field.border_radius == 14
    assert field.focused_border_color == COLOR_PRIMARY


def test_discord_managed_auth_dialog_groups_referral_field_with_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(discord_module, "create_glow_stack", lambda content: content)
    page = DummyPage()
    dialog = _dialog(page)

    dialog.open()

    assert _body_column(page).spacing == 44
    action_spacer = _content_column(page).controls[1]
    assert action_spacer.height == 24


def test_discord_managed_auth_dialog_referral_field_is_present_before_page_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(discord_module, "create_glow_stack", lambda content: content)
    page = SnapshotOpenPage()
    dialog = _dialog(page)

    dialog.open()

    assert page.body_control_classes_at_open == ["Text", "Column"]


def test_discord_managed_auth_dialog_invalid_looking_referral_id_does_not_block_continue() -> None:
    page = DummyPage()
    events: list[str] = []
    dialog = _dialog(page, events)
    dialog.open()

    assert dialog._referral_id_field is not None
    dialog._referral_id_field.value = "not a referral id"
    assert dialog.referral_id == "not a referral id"
    assert dialog._continue_button is not None
    dialog._continue_button.on_click(None)

    assert events == ["continue"]
    assert page.closed == []


def test_discord_managed_auth_dialog_waiting_state_uses_waiting_labels() -> None:
    set_locale("ko")
    page = DummyPage()
    dialog = _dialog(page)

    dialog.open()
    dialog.set_waiting()

    assert dialog.waiting_action_labels == [
        "discord_auth.cancel",
        "discord_auth.reopen_browser",
    ]
    assert dialog._body_text is not None
    assert dialog._body_text.value == t("discord_auth.waiting_body")
    assert dialog._reopen_browser_button is not None
    assert dialog._cancel_button is not None
    assert [control.content for control in dialog._actions.controls] == [
        t("discord_auth.cancel"),
        t("discord_auth.reopen_browser"),
    ]
    assert dialog._cancel_button is dialog._actions.controls[0]
    assert dialog._reopen_browser_button is dialog._actions.controls[1]


def test_discord_managed_auth_dialog_removes_referral_field_when_waiting() -> None:
    page = DummyPage()
    dialog = _dialog(page)

    dialog.open()
    assert dialog._dialog_result is not None
    body_column = dialog._dialog_result.body_column
    section = dialog._referral_section
    assert section is not None
    assert section in body_column.controls

    dialog.set_waiting()

    assert section not in body_column.controls
    assert dialog._referral_id_field is None
    assert dialog._referral_section is None
    assert body_column.controls == [dialog._body_text]


def test_discord_managed_auth_dialog_callback_received_expands_body() -> None:
    set_locale("ko")
    page = DummyPage()
    dialog = _dialog(page)

    dialog.open()
    dialog.set_waiting()
    dialog.set_callback_received()

    assert dialog._body_text is not None
    assert dialog._body_text.value == t("discord_auth.callback_received_body")
    assert [control.content for control in dialog._actions.controls] == [
        t("discord_auth.cancel"),
        t("discord_auth.reopen_browser"),
    ]


def test_discord_managed_auth_dialog_callback_received_requires_open_waiting_dialog() -> None:
    set_locale("ko")
    page = DummyPage()
    dialog = _dialog(page)

    dialog.open()
    assert dialog._body_text is not None
    initial_body = dialog._body_text.value

    dialog.set_callback_received()

    assert dialog._body_text.value == initial_body

    dialog.set_waiting()
    waiting_body = dialog._body_text.value
    dialog.close()
    dialog.set_callback_received()

    assert dialog._body_text.value == waiting_body


def test_discord_managed_auth_dialog_opens_one_modal_dialog_on_page() -> None:
    set_locale("en")
    page = DummyPage()
    dialog = _dialog(page)

    dialog.open()

    assert page.dialog is dialog._dialog
    assert len(page.opened) == 1
    assert dialog._dialog is not None
    assert dialog._dialog.modal is True
    assert dialog._continue_button is not None
    assert dialog._byok_button is None
    assert dialog._close_button is not None
    assert dialog._actions is not None
    assert dialog._close_button is dialog._actions.controls[0]
    assert dialog._continue_button is dialog._actions.controls[1]


def test_discord_managed_auth_dialog_open_is_idempotent() -> None:
    page = DummyPage()
    dialog = _dialog(page)

    dialog.open()
    first_dialog = dialog._dialog
    dialog.open()

    assert dialog._dialog is first_dialog
    assert page.opened == [first_dialog]
    assert page.closed == []
    assert page.dialog is first_dialog


def test_discord_managed_auth_dialog_continue_does_not_close_before_callback() -> None:
    page = DummyPage()
    events: list[str] = []
    dialog = _dialog(page, events)
    dialog.open()

    assert dialog._continue_button is not None
    dialog._continue_button.on_click(None)

    assert events == ["continue"]
    assert page.closed == []
    assert page.dialog is dialog._dialog


def test_discord_managed_auth_dialog_close_closes_then_invokes_callback() -> None:
    page = DummyPage()
    events: list[str] = []
    close_dialog = DiscordManagedAuthDialog(
        page,
        on_continue=lambda: events.append("continue"),
        on_byok=lambda: events.append(f"byok_closed={page.dialog is None}"),
        on_close=lambda: events.append(f"close_closed={page.dialog is None}"),
        on_reopen_browser=lambda: events.append("reopen"),
    )
    close_dialog.open()

    assert close_dialog._close_button is not None
    close_dialog._close_button.on_click(None)

    assert events == ["close_closed=True"]
    assert page.closed[-1] == close_dialog._dialog
    assert page.dialog is None


def test_discord_managed_auth_dialog_waiting_reopen_and_cancel_behavior() -> None:
    page = DummyPage()
    events: list[str] = []
    dialog = DiscordManagedAuthDialog(
        page,
        on_continue=lambda: events.append("continue"),
        on_byok=lambda: events.append("byok"),
        on_close=lambda: events.append(f"close_closed={page.dialog is None}"),
        on_reopen_browser=lambda: events.append(f"reopen_closed={page.dialog is None}"),
        on_cancel=lambda: events.append(f"cancel_closed={page.dialog is None}"),
    )
    dialog.open()
    dialog.set_waiting()

    assert dialog._reopen_browser_button is not None
    dialog._reopen_browser_button.on_click(None)

    assert events == ["reopen_closed=False"]
    assert page.closed == []
    assert page.dialog is dialog._dialog

    assert dialog._cancel_button is not None
    dialog._cancel_button.on_click(None)

    assert events == ["reopen_closed=False", "cancel_closed=True"]
    assert page.closed == [dialog._dialog]
    assert page.dialog is None


def test_discord_managed_auth_dialog_hides_reopen_when_callback_is_absent() -> None:
    page = DummyPage()
    events: list[str] = []
    dialog = _dialog_without_reopen(page, events)
    dialog.open()

    dialog.set_waiting()

    assert dialog._reopen_browser_button is None
    assert dialog._cancel_button is not None
    assert [control.content for control in dialog._actions.controls] == [t("discord_auth.cancel")]


def test_discord_managed_auth_dialog_waiting_cancel_falls_back_to_close_callback() -> None:
    page = DummyPage()
    events: list[str] = []
    dialog = DiscordManagedAuthDialog(
        page,
        on_continue=lambda: events.append("continue"),
        on_byok=lambda: events.append("byok"),
        on_close=lambda: events.append(f"close_closed={page.dialog is None}"),
        on_reopen_browser=lambda: events.append("reopen"),
    )
    dialog.open()
    dialog.set_waiting()

    assert dialog._cancel_button is not None
    dialog._cancel_button.on_click(None)

    assert events == ["close_closed=True"]
    assert page.closed == [dialog._dialog]
    assert page.dialog is None


def test_discord_managed_auth_dialog_close_is_idempotent() -> None:
    page = DummyPage()
    dialog = _dialog(page)
    dialog.open()

    dialog.close()
    dialog.close()

    assert page.closed == [dialog._dialog]
    assert page.dialog is None
