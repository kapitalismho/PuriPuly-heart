from __future__ import annotations

import inspect

import pytest

from tests.helpers.flet_page import DialogTrackingPage as DummyPage

pytest.importorskip("flet")

from puripuly_heart.ui.components.founder_letter_dialog import FounderLetterDialog
from puripuly_heart.ui.i18n import set_locale

REQUESTED_FOUNDER_LETTER_COPY = (
    "PuriPuly Heart를 사용해줘서 고마워요.\n"
    "제가 충전해놓은 크레딧은 여기서 끝이에요.\n\n"
    "어땠을까요?\n"
    "새로움이라든가 놀라움이라든가\n"
    "즐거운 순간이 당신과 함께했길 바라요.\n\n"
    "더 이용하고 싶으시면\n"
    "자신의 API 키를 발급해서 사용해주세요.\n\n"
    "방법은 README에 적어뒀어요.\n"
    "천천히 따라가면 어렵지 않을 거예요.\n\n"
    "어디선가 막히거나 건의하고 싶은 게 있다면\n"
    "무슨 일이든 편하게 저한테 연락해주세요.\n\n"
    "그럼 다시 만나길 바랄게요."
)


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
    assert dialog._cancel_button.content == "닫기"
    assert dialog._acknowledge_button.content == "README 열기"


def test_founder_letter_dialog_uses_requested_letter_copy() -> None:
    set_locale("ko")
    page = DummyPage()

    FounderLetterDialog(page).open()

    assert _body_text_value(page) == REQUESTED_FOUNDER_LETTER_COPY


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
