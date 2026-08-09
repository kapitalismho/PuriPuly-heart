from __future__ import annotations

import inspect

import pytest

from tests.helpers.flet_page import DialogTrackingPage as DummyPage

pytest.importorskip("flet")

from puripuly_heart.ui.components.founder_letter_dialog import (
    FOUNDER_LETTER_PARAGRAPH_KEYS,
    FounderLetterDialog,
)
from puripuly_heart.ui.i18n import set_locale, t


def _body_text_value(page: DummyPage) -> str:
    def walk(control):
        yield control
        nested_controls = getattr(control, "controls", None)
        if nested_controls:
            for nested in nested_controls:
                yield from walk(nested)
        nested_content = getattr(control, "content", None)
        if nested_content is not None:
            yield from walk(nested_content)

    for control in walk(page.dialog.content):
        if control.__class__.__name__ == "Text" and getattr(control, "selectable", False):
            return control.value
    raise AssertionError("dialog content did not include selectable body text")


def _dialog_with_readme_action(page: DummyPage, action) -> FounderLetterDialog:
    signature = inspect.signature(FounderLetterDialog)
    if "on_readme" not in signature.parameters:
        pytest.fail("FounderLetterDialog must expose an explicit on_readme callback")
    return FounderLetterDialog(
        page,
        on_readme=action,
        on_connect=lambda: pytest.fail("legacy on_connect must not be reused"),
        on_contact=lambda: pytest.fail("Founder Letter no longer uses contact action"),
    )


def test_founder_letter_dialog_opens_with_two_actions() -> None:
    set_locale("ko")
    page = DummyPage()

    dialog = FounderLetterDialog(page)

    dialog.open()

    assert page.dialog is dialog._dialog
    assert dialog._acknowledge_button is not None
    assert dialog._cancel_button is not None
    assert len(page.opened) == 1
    assert dialog._cancel_button.content == t("openrouter.handoff.close")
    assert dialog._acknowledge_button.content == t("openrouter.handoff.readme")


def test_founder_letter_dialog_uses_requested_letter_copy() -> None:
    set_locale("ko")
    page = DummyPage()

    FounderLetterDialog(page).open()

    expected_body = "\n\n".join(t(key) for key in FOUNDER_LETTER_PARAGRAPH_KEYS)
    assert _body_text_value(page) == expected_body


def test_founder_letter_dialog_is_modal_to_prevent_outside_dismissal() -> None:
    set_locale("ko")
    page = DummyPage()

    dialog = FounderLetterDialog(page)

    dialog.open()

    assert dialog._dialog is not None
    assert dialog._dialog.modal is True


def test_founder_letter_dialog_close_and_readme_actions() -> None:
    set_locale("ko")
    page = DummyPage()
    readme_calls: list[bool] = []

    readme_dialog = _dialog_with_readme_action(
        page,
        lambda: readme_calls.append(page.dialog is None),
    )
    readme_dialog.open()

    assert readme_dialog._acknowledge_button is not None
    readme_dialog._acknowledge_button.on_click(None)

    assert readme_calls == [True]
    assert page.closed == [readme_dialog._dialog]
    assert page.dialog is None

    close_dialog = _dialog_with_readme_action(
        page,
        lambda: readme_calls.append(False),
    )
    close_dialog.open()

    assert close_dialog._cancel_button is not None
    close_dialog._cancel_button.on_click(None)

    assert readme_calls == [True]
    assert page.closed[-1] == close_dialog._dialog
    assert page.dialog is None


def test_founder_letter_dialog_ignores_legacy_callbacks() -> None:
    set_locale("ko")
    page = DummyPage()

    legacy_dialog = FounderLetterDialog(
        page,
        on_connect=lambda: pytest.fail("legacy connect callback must stay ignored"),
        on_contact=lambda: pytest.fail("legacy contact callback must stay ignored"),
    )
    legacy_dialog.open()

    assert legacy_dialog._acknowledge_button is not None
    legacy_dialog._acknowledge_button.on_click(None)

    assert page.closed == [legacy_dialog._dialog]
    assert page.dialog is None
