from __future__ import annotations

from collections.abc import Callable

import pytest

pytest.importorskip("flet")

import puripuly_heart.ui.components.qq_managed_auth_dialog as qq_module  # noqa: E402
from puripuly_heart.ui.components.qq_managed_auth_dialog import QqManagedAuthDialog
from puripuly_heart.ui.i18n import get_locale, set_locale, t

VALID_CREDENTIAL = "a" * 64


@pytest.fixture(autouse=True)
def restore_locale_after_test():
    previous_locale = get_locale()
    try:
        yield
    finally:
        set_locale(previous_locale)


class DummyPage:
    def __init__(self) -> None:
        self.dialog = None
        self.opened: list[object] = []
        self.closed: list[object] = []
        self.update_count = 0

    def open(self, dialog) -> None:
        self.dialog = dialog
        self.opened.append(dialog)

    def close(self, dialog) -> None:
        self.closed.append(dialog)
        if self.dialog is dialog:
            self.dialog = None

    def update(self) -> None:
        self.update_count += 1


def _dialog(
    page: DummyPage,
    events: list[str] | None = None,
    *,
    on_cancel: Callable[[], None] | None = None,
    on_completion_message: Callable[[str], None] | None = None,
) -> QqManagedAuthDialog:
    calls = events if events is not None else []
    return QqManagedAuthDialog(
        page,
        on_submit=lambda: calls.append("submit"),
        on_close=lambda: calls.append("close"),
        on_cancel=on_cancel,
        on_completion_message=on_completion_message,
    )


def _click(button) -> None:
    button.on_click(None)


def _body_column(dialog: QqManagedAuthDialog):
    assert dialog._dialog_result is not None
    return dialog._dialog_result.body_column


def test_qq_managed_auth_dialog_declares_initial_and_waiting_action_labels() -> None:
    page = DummyPage()
    dialog = _dialog(page)

    assert dialog.action_labels == [
        "qq_auth.close",
        "qq_auth.submit",
    ]
    assert dialog.waiting_action_labels == ["qq_auth.cancel"]


def test_qq_managed_auth_dialog_renders_initial_fields_and_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qq_module, "create_glow_stack", lambda content: content)
    set_locale("en")
    page = DummyPage()
    dialog = _dialog(page)

    dialog.open()

    assert dialog._qq_identity_field is not None
    assert dialog._credential_field is not None
    assert dialog._qq_identity_field.label == t("qq_auth.qq_identity.label")
    assert dialog._credential_field.label == t("qq_auth.credential.label")
    assert dialog._close_button is not None
    assert dialog._submit_button is not None
    assert dialog._actions is not None
    assert [button.text for button in dialog._actions.controls] == [
        t("qq_auth.close"),
        t("qq_auth.submit"),
    ]
    assert [control.__class__.__name__ for control in _body_column(dialog).controls] == [
        "Text",
        "TextField",
        "TextField",
    ]


def test_qq_managed_auth_dialog_credential_field_is_masked_single_line() -> None:
    page = DummyPage()
    dialog = _dialog(page)

    dialog.open()

    field = dialog._credential_field
    assert field is not None
    assert getattr(field, "password", None) is True
    assert getattr(field, "multiline", None) is False
    assert getattr(field, "max_lines", None) == 1


@pytest.mark.parametrize(
    ("qq_identity", "credential"),
    [
        ("", VALID_CREDENTIAL),
        ("a" * 129, VALID_CREDENTIAL),
        ("qq identity", VALID_CREDENTIAL),
        ("qq\tidentity", VALID_CREDENTIAL),
        ("qq\x00identity", VALID_CREDENTIAL),
        ("qq-id", "a" * 63),
        ("qq-id", "A" * 64),
        ("qq-id", "g" * 64),
    ],
)
def test_qq_managed_auth_dialog_rejects_invalid_inputs_without_submit_callback(
    qq_identity: str,
    credential: str,
) -> None:
    page = DummyPage()
    events: list[str] = []
    dialog = _dialog(page, events)
    dialog.open()
    assert dialog._qq_identity_field is not None
    assert dialog._credential_field is not None
    assert dialog._submit_button is not None

    dialog._qq_identity_field.value = qq_identity
    dialog._credential_field.value = credential
    _click(dialog._submit_button)

    assert events == []
    assert dialog.is_open is True
    assert dialog.is_waiting is False
    assert dialog._body_text is not None
    assert t("qq_auth.error.invalid_input") in dialog._body_text.value


def test_qq_managed_auth_dialog_rejects_whitespace_wrapped_credential_without_submit_callback() -> (
    None
):
    page = DummyPage()
    events: list[str] = []
    dialog = _dialog(page, events)
    dialog.open()
    assert dialog._qq_identity_field is not None
    assert dialog._credential_field is not None
    assert dialog._submit_button is not None

    dialog._qq_identity_field.value = "qq-user-123"
    dialog._credential_field.value = f" {VALID_CREDENTIAL} "
    _click(dialog._submit_button)

    assert events == []
    assert dialog.is_open is True
    assert dialog.is_waiting is False
    assert dialog._body_text is not None
    assert t("qq_auth.error.invalid_input") in dialog._body_text.value


def test_qq_managed_auth_dialog_trims_valid_identity_and_submits_exact_credential_once() -> None:
    page = DummyPage()
    events: list[str] = []
    dialog = _dialog(page, events)
    dialog.open()
    assert dialog._qq_identity_field is not None
    assert dialog._credential_field is not None
    assert dialog._submit_button is not None

    dialog._qq_identity_field.value = "  qq-user-123  "
    dialog._credential_field.value = VALID_CREDENTIAL
    _click(dialog._submit_button)

    assert events == ["submit"]
    assert dialog.qq_identity == "qq-user-123"
    assert dialog.credential == VALID_CREDENTIAL
    assert dialog.is_waiting is True


def test_qq_managed_auth_dialog_prevents_duplicate_submit_while_waiting() -> None:
    page = DummyPage()
    events: list[str] = []
    dialog = _dialog(page, events)
    dialog.open()
    assert dialog._qq_identity_field is not None
    assert dialog._credential_field is not None
    assert dialog._submit_button is not None

    dialog._qq_identity_field.value = "qq-user-123"
    dialog._credential_field.value = VALID_CREDENTIAL
    _click(dialog._submit_button)
    _click(dialog._submit_button)

    assert events == ["submit"]


def test_qq_managed_auth_dialog_waiting_state_uses_cancel_and_removes_sensitive_fields() -> None:
    set_locale("ko")
    page = DummyPage()
    events: list[str] = []
    dialog = _dialog(page, events, on_cancel=lambda: events.append("cancel"))
    dialog.open()
    assert dialog._qq_identity_field is not None
    assert dialog._credential_field is not None
    dialog._qq_identity_field.value = "qq-user-123"
    dialog._credential_field.value = VALID_CREDENTIAL
    opened_dialog = dialog._dialog

    dialog.set_waiting()

    assert dialog.is_waiting is True
    assert dialog._body_text is not None
    assert dialog._body_text.value == t("qq_auth.waiting_body")
    assert dialog._actions is not None
    assert [control.text for control in dialog._actions.controls] == [t("qq_auth.cancel")]
    assert dialog._cancel_button is dialog._actions.controls[0]
    assert dialog._qq_identity_field is None
    assert dialog._credential_field is None
    assert dialog.qq_identity == ""
    assert dialog.credential == ""
    assert _body_column(dialog).controls == [dialog._body_text]

    _click(dialog._cancel_button)

    assert events == ["cancel"]
    assert page.closed == [opened_dialog]
    assert page.dialog is None
    assert dialog.is_open is False
    assert dialog.is_waiting is False
    assert dialog._dialog is None
    assert dialog._dialog_result is None
    assert dialog.qq_identity == ""
    assert dialog.credential == ""


def test_qq_managed_auth_dialog_close_clears_sensitive_values_and_references() -> None:
    page = DummyPage()
    dialog = _dialog(page)
    dialog.open()
    assert dialog._qq_identity_field is not None
    assert dialog._credential_field is not None
    dialog._qq_identity_field.value = "qq-user-123"
    dialog._credential_field.value = VALID_CREDENTIAL
    opened_dialog = dialog._dialog

    dialog.close()

    assert page.closed == [opened_dialog]
    assert dialog._qq_identity_field is None
    assert dialog._credential_field is None
    assert dialog._dialog is None
    assert dialog._dialog_result is None
    assert dialog.qq_identity == ""
    assert dialog.credential == ""


def test_qq_managed_auth_dialog_recoverable_credential_failure_restores_identity_and_clears_credential() -> (
    None
):
    page = DummyPage()
    dialog = _dialog(page)
    dialog.open()
    assert dialog._qq_identity_field is not None
    assert dialog._credential_field is not None
    dialog._qq_identity_field.value = "qq-user-123"
    dialog._credential_field.value = VALID_CREDENTIAL
    dialog.set_waiting()

    dialog.set_recoverable_error(
        "qq_auth.error.credential_mismatch",
        clear_credential=True,
    )

    assert dialog.is_open is True
    assert dialog.is_waiting is False
    assert dialog._qq_identity_field is not None
    assert dialog._credential_field is not None
    assert dialog._qq_identity_field.value == "qq-user-123"
    assert dialog._credential_field.value == ""
    assert t("qq_auth.error.credential_mismatch") in dialog._body_text.value
    assert [button.text for button in dialog._actions.controls] == [
        t("qq_auth.close"),
        t("qq_auth.submit"),
    ]


def test_qq_managed_auth_dialog_recoverable_retry_does_not_display_raw_broker_details() -> None:
    page = DummyPage()
    dialog = _dialog(page)
    dialog.open()
    assert dialog._qq_identity_field is not None
    assert dialog._credential_field is not None
    dialog._qq_identity_field.value = "qq-user-123"
    dialog._credential_field.value = VALID_CREDENTIAL
    dialog.set_waiting()

    dialog.set_recoverable_error(
        "broker says wait 999999999ms for raw-secret-detail",
        message_kwargs={
            "retry_after_ms": 999999999,
            "details": "raw-secret-detail",
        },
    )

    assert dialog.is_open is True
    assert dialog.is_waiting is False
    assert dialog._body_text is not None
    assert t("qq_auth.error.retry") in dialog._body_text.value
    assert "broker says" not in dialog._body_text.value
    assert "999999999" not in dialog._body_text.value
    assert "raw-secret-detail" not in dialog._body_text.value
    assert dialog._qq_identity_field is not None
    assert dialog._credential_field is not None
    assert dialog._qq_identity_field.value == "qq-user-123"
    assert dialog._credential_field.value == VALID_CREDENTIAL


def test_qq_managed_auth_dialog_completion_messages_close_and_clear_sensitive_state() -> None:
    page = DummyPage()
    messages: list[str] = []
    dialog = _dialog(page, on_completion_message=messages.append)
    dialog.open()
    assert dialog._qq_identity_field is not None
    assert dialog._credential_field is not None
    dialog._qq_identity_field.value = "qq-user-123"
    dialog._credential_field.value = VALID_CREDENTIAL
    dialog.set_waiting()
    opened_dialog = dialog._dialog

    dialog.complete_success(generation=dialog.auth_generation)

    assert messages == ["qq_auth.success"]
    assert page.closed == [opened_dialog]
    assert dialog.is_open is False
    assert dialog.qq_identity == ""
    assert dialog.credential == ""

    dialog.open()
    assert dialog._qq_identity_field is not None
    assert dialog._credential_field is not None
    dialog._qq_identity_field.value = "qq-user-456"
    dialog._credential_field.value = "b" * 64
    dialog.set_waiting()
    second_dialog = dialog._dialog

    dialog.complete_key_unavailable(generation=dialog.auth_generation)

    assert messages == ["qq_auth.success", "qq_auth.error.key_unavailable"]
    assert page.closed == [opened_dialog, second_dialog]
    assert dialog.is_open is False
    assert dialog.qq_identity == ""
    assert dialog.credential == ""


def test_qq_managed_auth_dialog_ignores_stale_completion_after_cancel() -> None:
    page = DummyPage()
    messages: list[str] = []
    events: list[str] = []
    dialog = _dialog(
        page,
        events,
        on_cancel=lambda: events.append("cancel"),
        on_completion_message=messages.append,
    )
    dialog.open()
    assert dialog._qq_identity_field is not None
    assert dialog._credential_field is not None
    dialog._qq_identity_field.value = "qq-user-123"
    dialog._credential_field.value = VALID_CREDENTIAL
    dialog.set_waiting()
    canceled_generation = dialog.auth_generation
    assert dialog._cancel_button is not None

    _click(dialog._cancel_button)
    dialog.complete_success(generation=canceled_generation)
    dialog.set_recoverable_error("qq_auth.error.retry", generation=canceled_generation)

    assert events == ["cancel"]
    assert messages == []
    assert page.dialog is None
    assert dialog.is_open is False
    assert dialog._body_text is None
