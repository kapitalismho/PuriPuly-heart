from __future__ import annotations

from collections.abc import Callable

import pytest

from tests.helpers.flet_page import DialogTrackingPage as DummyPage

pytest.importorskip("flet")

import flet as ft  # noqa: E402

import puripuly_heart.ui.components.discord_managed_auth_dialog as discord_module  # noqa: E402
from puripuly_heart.ui.components import warm_document_dialog  # noqa: E402
from puripuly_heart.ui.components.discord_managed_auth_dialog import DiscordManagedAuthDialog
from puripuly_heart.ui.i18n import set_locale, t
from puripuly_heart.ui.theme import (  # noqa: E402
    COLOR_NEUTRAL_DARK,
    COLOR_PRIMARY,
)


def _dialog(
    page: DummyPage,
    events: list[str] | None = None,
    *,
    on_cancel: Callable[[], None] | None = None,
) -> DiscordManagedAuthDialog:
    calls = events if events is not None else []
    return DiscordManagedAuthDialog(
        page,
        on_continue=lambda: calls.append("continue"),
        on_byok=lambda: calls.append("byok"),
        on_close=lambda: calls.append("close"),
        on_cancel=on_cancel,
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


def test_discord_managed_auth_dialog_uses_warm_document_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_keys: list[str] = []

    def fake_t(key: str) -> str:
        requested_keys.append(key)
        return f"value:{key}"

    monkeypatch.setattr(discord_module, "t", fake_t)

    page = DummyPage()
    dialog = DiscordManagedAuthDialog(
        page,
        on_continue=lambda: None,
        on_byok=lambda: None,
        on_close=lambda: None,
    )

    dialog.open()

    modal_content = _modal_content(page)
    assert "discord_auth.title" not in requested_keys
    assert "discord_auth.requirements" not in requested_keys
    assert "discord_auth.byok" not in requested_keys
    assert modal_content.width == warm_document_dialog.DIALOG_WIDTH
    assert modal_content.height is None

    body_column = _body_column(page)
    body_text = body_column.controls[0]
    action_row = _action_row(page)

    assert len(body_column.controls) == 1
    assert body_text.value == "value:discord_auth.body"
    assert body_text.size == warm_document_dialog.BODY_TEXT_SIZE
    assert body_text.selectable is True
    assert [button.__class__.__name__ for button in action_row.controls] == [
        "TextButton",
        "TextButton",
    ]
    assert [button.content for button in action_row.controls] == [
        "value:openrouter.handoff.chatgpt",
        "value:discord_auth.continue",
    ]
    assert [_button_text_size(button) for button in action_row.controls] == [
        warm_document_dialog.BUTTON_TEXT_SIZE
    ] * 2
    assert [button.style.color[ft.ControlState.DEFAULT] for button in action_row.controls] == [
        COLOR_NEUTRAL_DARK,
        COLOR_NEUTRAL_DARK,
    ]
    assert [button.style.color[ft.ControlState.HOVERED] for button in action_row.controls] == [
        COLOR_PRIMARY,
        COLOR_PRIMARY,
    ]


def test_discord_managed_auth_dialog_waiting_state_uses_waiting_labels() -> None:
    set_locale("ko")
    page = DummyPage()
    dialog = _dialog(page)

    dialog.open()
    dialog.set_waiting()

    assert dialog.waiting_action_labels == ["discord_auth.cancel"]
    assert dialog._body_text is not None
    assert dialog._body_text.value == t("discord_auth.waiting_body")
    assert dialog._cancel_button is not None
    assert [control.content for control in dialog._actions.controls] == [t("discord_auth.cancel")]
    assert dialog._cancel_button is dialog._actions.controls[0]


def test_discord_managed_auth_dialog_callback_received_expands_body() -> None:
    set_locale("ko")
    page = DummyPage()
    dialog = _dialog(page)

    dialog.open()
    dialog.set_waiting()
    dialog.set_callback_received()

    assert dialog._body_text is not None
    assert dialog._body_text.value == t("discord_auth.callback_received_body")
    assert [control.content for control in dialog._actions.controls] == [t("discord_auth.cancel")]


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


def test_discord_managed_auth_dialog_dismisses_outside_until_waiting() -> None:
    set_locale("en")
    page = DummyPage()
    dialog = _dialog(page)

    dialog.open()

    assert page.dialog is dialog._dialog
    assert len(page.opened) == 1
    assert dialog._dialog is not None
    assert dialog._dialog.modal is False
    assert dialog._continue_button is not None
    assert dialog._byok_button is None
    assert dialog._close_button is None
    assert dialog._actions is not None
    assert dialog._chatgpt_button is dialog._actions.controls[0]
    assert dialog._continue_button is dialog._actions.controls[1]

    dialog.set_waiting()

    assert dialog._dialog.modal is True


def test_discord_managed_auth_dialog_outside_dismiss_notifies_close_once() -> None:
    page = DummyPage()
    events: list[str] = []
    dialog = _dialog(page, events)
    dialog.open()

    dialog._dialog.on_dismiss(None)
    dialog._dialog.on_dismiss(None)

    assert events == ["close"]
    assert dialog.is_open is False
    assert page.closed == []


def test_discord_managed_auth_dialog_ignores_dismiss_while_waiting() -> None:
    page = DummyPage()
    events: list[str] = []
    dialog = _dialog(page, events)
    dialog.open()
    dialog.set_waiting()

    dialog._dialog.on_dismiss(None)

    assert events == []
    assert dialog.is_open is True


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


def test_discord_managed_auth_dialog_chatgpt_closes_then_invokes_callback() -> None:
    page = DummyPage()
    events: list[str] = []
    dialog = DiscordManagedAuthDialog(
        page,
        on_continue=lambda: events.append("continue"),
        on_byok=lambda: events.append("byok"),
        on_close=lambda: events.append("close"),
        on_chatgpt=lambda: events.append(f"chatgpt_closed={page.dialog is None}"),
    )
    dialog.open()

    assert dialog._chatgpt_button is not None
    dialog._chatgpt_button.on_click(None)

    assert events == ["chatgpt_closed=True"]
    assert page.closed[-1] == dialog._dialog
    assert page.dialog is None
    assert dialog.is_open is False


def test_discord_managed_auth_dialog_waiting_cancel_behavior() -> None:
    page = DummyPage()
    events: list[str] = []
    dialog = DiscordManagedAuthDialog(
        page,
        on_continue=lambda: events.append("continue"),
        on_byok=lambda: events.append("byok"),
        on_close=lambda: events.append(f"close_closed={page.dialog is None}"),
        on_cancel=lambda: events.append(f"cancel_closed={page.dialog is None}"),
    )
    dialog.open()
    dialog.set_waiting()

    assert dialog._cancel_button is not None
    dialog._cancel_button.on_click(None)

    assert events == ["cancel_closed=True"]
    assert page.closed == [dialog._dialog]
    assert page.dialog is None


def test_discord_managed_auth_dialog_waiting_cancel_falls_back_to_close_callback() -> None:
    page = DummyPage()
    events: list[str] = []
    dialog = DiscordManagedAuthDialog(
        page,
        on_continue=lambda: events.append("continue"),
        on_byok=lambda: events.append("byok"),
        on_close=lambda: events.append(f"close_closed={page.dialog is None}"),
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


def test_exhaustion_exposes_resume_and_ignores_stale_progress_callbacks() -> None:
    page = DummyPage()
    calls: list[str] = []
    dialog = _dialog(page, calls)
    dialog.open()
    dialog.set_waiting()
    dialog.set_recovering()
    dialog.set_error("discord_auth.error.recovery_pending", resume=True)

    assert dialog.is_open is True
    assert dialog.is_waiting is False
    assert dialog._dialog.modal is False
    assert [button.content for button in dialog._actions.controls] == [
        t("discord_auth.close"),
        t("discord_auth.resume"),
    ]
    stopped_body = dialog._body_text.value
    dialog.set_callback_received()
    dialog.set_recovering()
    assert dialog._body_text.value == stopped_body
    dialog._continue_button.on_click(None)
    assert calls == ["continue"]
    dialog.set_waiting()
    dialog._submit()
    assert calls == ["continue"]
