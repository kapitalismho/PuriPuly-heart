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

    assert dialog.action_labels == ["qq_auth.close", "qq_auth.submit"]
    assert page.dialog is dialog._dialog
    assert dialog._qq_identity_field is not None
    assert dialog._credential_field is not None
    assert dialog._qq_identity_field.label == t("qq_auth.qq_identity.label")
    assert dialog._qq_identity_field.helper == t("qq_auth.qq_identity.helper")
    assert dialog._credential_field.label == t("qq_auth.credential.label")
    assert dialog._credential_field.helper == t("qq_auth.credential.helper")
    assert dialog._credential_field.password is True
    assert [control.content for control in dialog._actions.controls] == [
        t("qq_auth.close"),
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
