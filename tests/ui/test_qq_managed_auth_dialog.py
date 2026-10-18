from __future__ import annotations

import pytest

from tests.helpers.flet_page import DialogTrackingPage

pytest.importorskip("flet")

from puripuly_heart.ui.components.qq_managed_auth_dialog import QqManagedAuthDialog  # noqa: E402
from puripuly_heart.ui.i18n import set_locale, t  # noqa: E402


class DummyPage(DialogTrackingPage):
    def __init__(self) -> None:
        super().__init__()
        self.updated = 0

    def update(self) -> None:
        self.updated += 1


def _dialog(page: DummyPage, events: list[str] | None = None) -> QqManagedAuthDialog:
    calls = events if events is not None else []
    return QqManagedAuthDialog(
        page,
        on_continue=lambda: calls.append("continue"),
        on_close=lambda: calls.append("close"),
        on_cancel=lambda: calls.append("cancel"),
    )


def test_qq_managed_auth_dialog_renders_inputs_and_actions() -> None:
    set_locale("en")
    page = DummyPage()
    dialog = _dialog(page)

    dialog.open()

    assert dialog.action_labels == ["openrouter.handoff.chatgpt", "qq_auth.submit"]
    assert dialog._dialog.modal is False
    assert page.dialog is dialog._dialog
    assert dialog._qq_identity_field is not None
    assert dialog._credential_field is not None
    assert dialog._qq_identity_field.label == t("qq_auth.qq_identity.label")
    assert dialog._qq_identity_field.helper == t("qq_auth.qq_identity.helper")
    assert dialog._credential_field.label == t("qq_auth.credential.label")
    assert dialog._credential_field.helper == t("qq_auth.credential.helper")
    assert dialog._credential_field.password is True
    assert not hasattr(dialog, "referral_id")
    assert [
        control.__class__.__name__ for control in dialog._dialog_result.body_column.controls
    ].count("TextField") == 2
    assert [control.content for control in dialog._actions.controls] == [
        t("openrouter.handoff.chatgpt"),
        t("qq_auth.submit"),
    ]


def test_qq_managed_auth_dialog_validates_required_fields_without_closing() -> None:
    page = DummyPage()
    events: list[str] = []
    dialog = _dialog(page, events)
    dialog.open()

    dialog._continue_button.on_click(None)

    assert events == []
    assert page.closed == []
    assert dialog._error_text is not None
    assert dialog._error_text.visible is True
    assert dialog._error_text.value == t("qq_auth.error.invalid_input")


def test_qq_managed_auth_dialog_submit_waiting_error_and_cancel_states() -> None:
    page = DummyPage()
    events: list[str] = []
    dialog = _dialog(page, events)
    dialog.open()
    dialog._qq_identity_field.value = "qq-user"
    dialog._credential_field.value = "0123456789abcdef" * 4

    dialog._continue_button.on_click(None)

    assert events == ["continue"]
    assert page.closed == []
    dialog.set_waiting()
    assert dialog.is_waiting is True
    assert dialog._qq_identity_field.disabled is True
    assert dialog._credential_field.disabled is True
    assert [control.content for control in dialog._actions.controls] == [t("qq_auth.cancel")]

    dialog.set_error("qq_auth.error.credential_mismatch")
    assert dialog.is_waiting is False
    assert dialog._qq_identity_field.disabled is False
    assert dialog._credential_field.disabled is False
    assert dialog._error_text.value == t("qq_auth.error.credential_mismatch")

    dialog.set_waiting()
    dialog._cancel_button.on_click(None)
    assert events == ["continue", "cancel"]
    assert page.closed == [dialog._dialog]


def test_qq_managed_auth_dialog_outside_dismiss_closes_only_before_waiting() -> None:
    page = DummyPage()
    events: list[str] = []
    dialog = _dialog(page, events)
    dialog.open()

    dialog._dialog.on_dismiss(None)

    assert events == ["close"]
    assert dialog.is_open is False

    waiting_events: list[str] = []
    waiting = _dialog(page, waiting_events)
    waiting.open()
    waiting.set_waiting()

    assert waiting._dialog.modal is True
    waiting._dialog.on_dismiss(None)
    assert waiting_events == []
    assert waiting.is_open is True

    waiting.set_error("qq_auth.error.retry")

    assert waiting._dialog.modal is False


def test_qq_managed_auth_dialog_chatgpt_closes_then_invokes_callback() -> None:
    page = DummyPage()
    events: list[str] = []
    dialog = QqManagedAuthDialog(
        page,
        on_continue=lambda: events.append("continue"),
        on_close=lambda: events.append("close"),
        on_chatgpt=lambda: events.append(f"chatgpt_closed={page.dialog is None}"),
    )
    dialog.open()

    assert dialog._chatgpt_button is not None
    dialog._chatgpt_button.on_click(None)

    assert events == ["chatgpt_closed=True"]
    assert dialog.is_open is False


def test_qq_recovery_is_idle_and_resume_preserves_inputs_until_admission() -> None:
    page = DummyPage()
    calls: list[str] = []
    dialog = _dialog(page, calls)
    dialog.open()
    dialog._qq_identity_field.value = "qq-user"
    dialog._credential_field.value = "a" * 64
    dialog.set_waiting()
    dialog._submit()
    assert calls == []
    dialog.set_error("qq_auth.error.delivery_ack_pending")
    assert dialog.is_waiting is False
    assert dialog._continue_button.content == t("qq_auth.resume")
    assert dialog.qq_identity == "qq-user"
    assert dialog.credential == "a" * 64
    assert dialog._qq_identity_field.disabled is False
    assert dialog._credential_field.disabled is False
    dialog._continue_button.on_click(None)
    assert calls == ["continue"]
