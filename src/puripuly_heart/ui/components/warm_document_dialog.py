from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import flet as ft

from puripuly_heart.ui.components.glow import create_glow_stack
from puripuly_heart.ui.theme import (
    COLOR_DIVIDER,
    COLOR_NEUTRAL_DARK,
    COLOR_ON_BACKGROUND,
    COLOR_PRIMARY,
    COLOR_SURFACE,
    get_card_shadow,
)

DIALOG_WIDTH = 720
DIALOG_HORIZONTAL_PADDING = 36
DIALOG_VERTICAL_PADDING = 34
BODY_TEXT_SIZE = 24
PARAGRAPH_SPACING = 22
ACTION_TOP_MARGIN = 32
ACTION_SPACING = 14
BUTTON_HORIZONTAL_PADDING = 30
BUTTON_VERTICAL_PADDING = 20
BUTTON_RADIUS = 20
BUTTON_TEXT_SIZE = 26


def _action_button_style() -> ft.ButtonStyle:
    return ft.ButtonStyle(
        color={
            ft.ControlState.DEFAULT: COLOR_NEUTRAL_DARK,
            ft.ControlState.HOVERED: COLOR_PRIMARY,
        },
        bgcolor=ft.Colors.TRANSPARENT,
        padding=ft.Padding.symmetric(
            horizontal=BUTTON_HORIZONTAL_PADDING,
            vertical=BUTTON_VERTICAL_PADDING,
        ),
        shape=ft.RoundedRectangleBorder(radius=BUTTON_RADIUS),
        overlay_color=ft.Colors.TRANSPARENT,
        text_style=ft.TextStyle(size=BUTTON_TEXT_SIZE, weight=ft.FontWeight.BOLD),
        animation_duration=0,
    )


def _make_text_button(label: str, **kwargs) -> ft.TextButton:
    return ft.TextButton(content=label, **kwargs)


@dataclass(frozen=True)
class WarmDocumentDialogAction:
    label: str
    on_select: Callable[[], None] | None = None
    close_before_action: bool = True


@dataclass(frozen=True)
class WarmDocumentDialogResult:
    dialog: ft.AlertDialog
    primary_button: ft.TextButton
    secondary_button: ft.TextButton
    body_text: ft.Text
    body_column: ft.Column
    action_row: ft.Row
    initial_action_buttons: tuple[ft.TextButton, ...]
    set_actions: Callable[[Sequence[WarmDocumentDialogAction]], tuple[ft.TextButton, ...]]
    title_text: ft.Text | None = None


def split_body_paragraphs(body: str) -> list[str]:
    return [paragraph.strip() for paragraph in body.split("\n\n") if paragraph.strip()]


def join_body_paragraphs(body_paragraphs: Sequence[str]) -> str:
    return "\n\n".join(paragraph.strip() for paragraph in body_paragraphs if paragraph.strip())


def open_warm_document_dialog(
    page: ft.Page,
    *,
    body_paragraphs: Sequence[str],
    title: str | None = None,
    extra_body_controls: Sequence[ft.Control] | None = None,
    body_spacing: int = PARAGRAPH_SPACING,
    action_top_margin: int = ACTION_TOP_MARGIN,
    primary_label: str | None = None,
    primary_action: Callable[[], None] | None = None,
    secondary_label: str | None = None,
    secondary_action: Callable[[], None] | None = None,
    actions: Sequence[WarmDocumentDialogAction] | None = None,
    glow_factory: Callable[[ft.Control], ft.Control] = create_glow_stack,
) -> WarmDocumentDialogResult:
    dialog: ft.AlertDialog | None = None

    def select(action: WarmDocumentDialogAction) -> None:
        if action.close_before_action and dialog is not None:
            page.pop_dialog()
        if action.on_select is not None:
            action.on_select()

    def normalize_actions(
        action_specs: Sequence[WarmDocumentDialogAction],
    ) -> tuple[WarmDocumentDialogAction, ...]:
        normalized = tuple(action_specs)
        if len(normalized) not in {1, 2, 3}:
            raise ValueError("warm document dialogs support one to three actions")
        return normalized

    if actions is None:
        if primary_label is None or secondary_label is None:
            raise ValueError("primary_label and secondary_label are required")
        initial_actions = normalize_actions(
            (
                WarmDocumentDialogAction(
                    label=secondary_label,
                    on_select=secondary_action,
                ),
                WarmDocumentDialogAction(
                    label=primary_label,
                    on_select=primary_action,
                ),
            )
        )
    else:
        initial_actions = normalize_actions(actions)

    def make_action_buttons(
        action_specs: Sequence[WarmDocumentDialogAction],
    ) -> tuple[ft.TextButton, ...]:
        return tuple(
            _make_text_button(
                action.label,
                on_click=lambda _, selected_action=action: select(selected_action),
                style=_action_button_style(),
            )
            for action in normalize_actions(action_specs)
        )

    title_text: ft.Text | None = None
    if title is not None and title.strip():
        title_text = ft.Text(
            title.strip(),
            size=BODY_TEXT_SIZE,
            color=COLOR_ON_BACKGROUND,
            selectable=True,
            weight=ft.FontWeight.BOLD,
        )
    body_text = ft.Text(
        join_body_paragraphs(body_paragraphs),
        size=BODY_TEXT_SIZE,
        color=COLOR_ON_BACKGROUND,
        selectable=True,
    )
    body_controls: list[ft.Control] = []
    if title_text is not None:
        body_controls.append(title_text)
    body_controls.append(body_text)
    body_controls.extend(extra_body_controls or ())
    body = ft.Column(
        controls=body_controls,
        spacing=body_spacing,
        tight=True,
    )

    initial_action_buttons = make_action_buttons(initial_actions)

    action_row = ft.Row(
        controls=list(initial_action_buttons),
        spacing=ACTION_SPACING,
        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        wrap=True,
    )

    def set_actions(
        replacement_actions: Sequence[WarmDocumentDialogAction],
    ) -> tuple[ft.TextButton, ...]:
        replacement_buttons = make_action_buttons(replacement_actions)
        action_row.controls = list(replacement_buttons)
        return replacement_buttons

    secondary_button = initial_action_buttons[0]
    primary_button = initial_action_buttons[-1]

    modal_content = ft.Container(
        width=DIALOG_WIDTH,
        padding=ft.Padding.symmetric(
            horizontal=DIALOG_HORIZONTAL_PADDING,
            vertical=DIALOG_VERTICAL_PADDING,
        ),
        bgcolor=COLOR_SURFACE,
        border_radius=30,
        border=ft.Border.all(1, ft.Colors.with_opacity(0.35, COLOR_DIVIDER)),
        shadow=get_card_shadow(),
        content=ft.Column(
            controls=[
                body,
                ft.Container(height=action_top_margin),
                action_row,
            ],
            spacing=0,
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        ),
    )

    dialog = ft.AlertDialog(
        modal=True,
        content=glow_factory(modal_content),
        content_padding=0,
        bgcolor=ft.Colors.TRANSPARENT,
    )
    page.show_dialog(dialog)
    return WarmDocumentDialogResult(
        dialog=dialog,
        primary_button=primary_button,
        secondary_button=secondary_button,
        body_text=body_text,
        body_column=body,
        action_row=action_row,
        initial_action_buttons=initial_action_buttons,
        set_actions=set_actions,
        title_text=title_text,
    )
