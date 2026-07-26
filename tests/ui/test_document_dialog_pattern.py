from __future__ import annotations

import pytest

pytest.importorskip("flet")

import flet as ft  # noqa: E402

import puripuly_heart.ui.components.founder_letter_dialog as founder_module  # noqa: E402
import puripuly_heart.ui.components.peer_translation_eula_dialog as eula_module  # noqa: E402
import puripuly_heart.ui.components.warm_document_dialog as warm_dialog_module  # noqa: E402
from puripuly_heart.ui.components.founder_letter_dialog import (  # noqa: E402
    FounderLetterDialog,
)
from puripuly_heart.ui.components.peer_translation_eula_dialog import (  # noqa: E402
    PeerTranslationEulaDialog,
)
from puripuly_heart.ui.components.warm_document_dialog import (  # noqa: E402
    WarmDocumentDialogAction,
    open_warm_document_dialog,
)
from puripuly_heart.ui.theme import (  # noqa: E402
    COLOR_BACKGROUND,
    COLOR_NEUTRAL_DARK,
    COLOR_PRIMARY,
)


class FakePage:
    def __init__(self) -> None:
        self.dialog = None
        self.closed: list[object] = []

    def show_dialog(self, dialog) -> None:
        self.dialog = dialog

    def pop_dialog(self):
        dialog = self.dialog
        if dialog is None:
            return None
        self.closed.append(dialog)
        self.dialog = None
        return dialog


def _open_founder_letter(monkeypatch: pytest.MonkeyPatch) -> tuple[FakePage, list[str]]:
    requested_keys: list[str] = []

    def fake_t(key: str) -> str:
        requested_keys.append(key)
        return f"value:{key}"

    monkeypatch.setattr(founder_module, "t", fake_t)
    monkeypatch.setattr(founder_module, "create_glow_stack", lambda content: content)

    page = FakePage()
    FounderLetterDialog(
        page,
        on_connect=lambda: None,
        on_contact=lambda: None,
    ).open()
    return page, requested_keys


def _open_eula(monkeypatch: pytest.MonkeyPatch) -> tuple[FakePage, list[str]]:
    requested_keys: list[str] = []

    def fake_t(key: str) -> str:
        requested_keys.append(key)
        return f"value:{key}"

    monkeypatch.setattr(eula_module, "t", fake_t)
    monkeypatch.setattr(eula_module, "create_glow_stack", lambda content: content)

    page = FakePage()
    PeerTranslationEulaDialog(page, on_accept=lambda: None).open()
    return page, requested_keys


def _modal_content(page: FakePage):
    return page.dialog.content


def _content_column(page: FakePage):
    return _modal_content(page).content


def _first_nested_column(page: FakePage):
    for control in _content_column(page).controls:
        if control.__class__.__name__ == "Column":
            return control
    raise AssertionError("dialog content did not include a body column")


def _button_vertical_padding(button) -> tuple[int, int]:
    padding = button.style.padding
    return padding.top, padding.bottom


def _button_text_size(button) -> int:
    return button.style.text_style.size


def _button_color(button, state: ft.ControlState) -> str:
    return button.style.color[state]


@pytest.mark.parametrize("dialog_opener", [_open_founder_letter, _open_eula])
def test_document_dialogs_use_large_vr_readable_body_type(
    monkeypatch: pytest.MonkeyPatch,
    dialog_opener,
) -> None:
    page, _requested_keys = dialog_opener(monkeypatch)

    modal_content = _modal_content(page)
    body_column = _first_nested_column(page)

    assert modal_content.width == 720
    assert modal_content.height is None
    assert body_column.spacing == 22
    assert [control.size for control in body_column.controls] == [24]


@pytest.mark.parametrize("dialog_opener", [_open_founder_letter, _open_eula])
def test_document_dialog_content_is_vertically_centered(
    monkeypatch: pytest.MonkeyPatch,
    dialog_opener,
) -> None:
    page, _requested_keys = dialog_opener(monkeypatch)

    content_column = _content_column(page)

    assert content_column.alignment.name == "CENTER"


@pytest.mark.parametrize("dialog_opener", [_open_founder_letter, _open_eula])
def test_document_dialog_body_text_is_selectable(
    monkeypatch: pytest.MonkeyPatch,
    dialog_opener,
) -> None:
    page, _requested_keys = dialog_opener(monkeypatch)

    body_column = _first_nested_column(page)

    assert len(body_column.controls) == 1
    assert body_column.controls[0].selectable is True


def test_founder_letter_body_text_is_one_selectable_text_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page, _requested_keys = _open_founder_letter(monkeypatch)

    body_column = _first_nested_column(page)

    assert len(body_column.controls) == 1
    assert body_column.controls[0].value == (
        "value:openrouter.handoff.letter.p1\n\n"
        "value:openrouter.handoff.letter.p2\n\n"
        "value:openrouter.handoff.letter.p3\n\n"
        "value:openrouter.handoff.letter.p4\n\n"
        "value:openrouter.handoff.letter.p5\n\n"
        "value:openrouter.handoff.letter.p6"
    )


def test_document_dialogs_do_not_render_title_or_status_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _founder_page, founder_keys = _open_founder_letter(monkeypatch)
    _eula_page, eula_keys = _open_eula(monkeypatch)

    assert "openrouter.handoff.status" not in founder_keys
    assert "peer_translation_eula.title" not in eula_keys


@pytest.mark.parametrize("dialog_opener", [_open_founder_letter, _open_eula])
def test_document_dialogs_use_large_standalone_action_buttons(
    monkeypatch: pytest.MonkeyPatch,
    dialog_opener,
) -> None:
    page, _requested_keys = dialog_opener(monkeypatch)

    action_row = _content_column(page).controls[-1]

    assert action_row.__class__.__name__ == "Row"
    assert [button.__class__.__name__ for button in action_row.controls] == [
        "TextButton",
        "TextButton",
    ]
    assert action_row.spacing == 14
    assert _button_vertical_padding(action_row.controls[0]) == (20, 20)
    assert _button_vertical_padding(action_row.controls[1]) == (20, 20)
    assert _button_text_size(action_row.controls[0]) == 26
    assert _button_text_size(action_row.controls[1]) == 26
    assert _button_color(action_row.controls[0], ft.ControlState.DEFAULT) == COLOR_NEUTRAL_DARK
    assert _button_color(action_row.controls[0], ft.ControlState.HOVERED) == COLOR_PRIMARY
    assert _button_color(action_row.controls[1], ft.ControlState.DEFAULT) == COLOR_NEUTRAL_DARK
    assert _button_color(action_row.controls[1], ft.ControlState.HOVERED) == COLOR_PRIMARY
    assert action_row.controls[0].style.bgcolor == ft.Colors.TRANSPARENT
    assert action_row.controls[1].style.bgcolor == ft.Colors.TRANSPARENT
    assert action_row.controls[0].style.animation_duration == 0
    assert action_row.controls[1].style.animation_duration == 0
    assert getattr(action_row, "bgcolor", None) != COLOR_BACKGROUND


def test_warm_document_dialog_supports_three_ordered_actions_with_close_policy() -> None:
    page = FakePage()
    events: list[str] = []

    result = open_warm_document_dialog(
        page,
        body_paragraphs=["Body copy"],
        actions=[
            WarmDocumentDialogAction(label="BYOK", on_select=lambda: events.append("byok")),
            WarmDocumentDialogAction(label="Close", on_select=lambda: events.append("close")),
            WarmDocumentDialogAction(
                label="Continue",
                on_select=lambda: events.append("continue"),
                close_before_action=False,
            ),
        ],
        glow_factory=lambda content: content,
    )

    action_row = _content_column(page).controls[-1]

    assert action_row is result.action_row
    assert result.initial_action_buttons == tuple(action_row.controls)
    assert [button.content for button in action_row.controls] == ["BYOK", "Close", "Continue"]
    assert [button.__class__.__name__ for button in action_row.controls] == [
        "TextButton",
        "TextButton",
        "TextButton",
    ]

    result.initial_action_buttons[2].on_click(None)

    assert events == ["continue"]
    assert page.closed == []
    assert page.dialog is result.dialog

    result.initial_action_buttons[0].on_click(None)

    assert events == ["continue", "byok"]
    assert page.closed == [result.dialog]
    assert page.dialog is None


def test_warm_document_dialog_supports_content_only_text_button_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ContentOnlyTextButton:
        def __init__(self, *, content, on_click=None, style=None) -> None:
            self.content = content
            self.on_click = on_click
            self.style = style

    monkeypatch.setattr(warm_dialog_module.ft, "TextButton", ContentOnlyTextButton)
    page = FakePage()

    result = open_warm_document_dialog(
        page,
        body_paragraphs=["Body copy"],
        actions=[WarmDocumentDialogAction(label="Close")],
        glow_factory=lambda content: content,
    )

    assert result.initial_action_buttons[0].content == "Close"


def test_warm_document_dialog_can_replace_actions_with_shared_style() -> None:
    page = FakePage()
    events: list[str] = []

    result = open_warm_document_dialog(
        page,
        body_paragraphs=["Body copy"],
        actions=[
            WarmDocumentDialogAction(label="Left", on_select=lambda: events.append("left")),
            WarmDocumentDialogAction(label="Right", on_select=lambda: events.append("right")),
        ],
        glow_factory=lambda content: content,
    )

    replacement_buttons = result.set_actions(
        [
            WarmDocumentDialogAction(
                label="Cancel",
                on_select=lambda: events.append("cancel"),
                close_before_action=False,
            )
        ]
    )

    assert replacement_buttons == tuple(result.action_row.controls)
    assert result.action_row.controls == list(replacement_buttons)
    assert [button.content for button in result.action_row.controls] == ["Cancel"]
    assert result.action_row.controls[0].style.color[ft.ControlState.DEFAULT] == COLOR_NEUTRAL_DARK
    assert result.action_row.controls[0].style.color[ft.ControlState.HOVERED] == COLOR_PRIMARY
    assert result.action_row.controls[0].style.animation_duration == 0

    replacement_buttons[0].on_click(None)

    assert events == ["cancel"]
    assert page.closed == []
    assert page.dialog is result.dialog
