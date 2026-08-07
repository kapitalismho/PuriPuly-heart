from __future__ import annotations

import contextlib
import inspect
from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import flet as ft

from puripuly_heart.ui.flet_runtime import (
    control_page,
    is_hover_active,
    update_control_if_mounted,
)
from puripuly_heart.ui.theme import (
    COLOR_BACKGROUND,
    COLOR_DIVIDER,
    COLOR_ON_BACKGROUND,
    COLOR_ON_PRIMARY_CONTAINER,
    COLOR_PRIMARY,
    COLOR_PRIMARY_CONTAINER,
    COLOR_SECONDARY,
    COLOR_SURFACE,
)


@dataclass(frozen=True)
class TextSubtab:
    key: str
    label: str
    controls: Sequence[ft.Control]


class _ScrollBody(ft.Column):
    def __init__(self, tab_key: str, controls: Sequence[ft.Control], *, on_scroll) -> None:
        super().__init__(
            controls=list(controls),
            expand=True,
            spacing=16,
            scroll=ft.ScrollMode.AUTO,
            on_scroll=on_scroll,
            scroll_interval=0,
            visible=False,
        )
        self.tab_key = tab_key

    def restore_scroll(self, offset: float) -> None:
        page = control_page(self)
        if page is None:
            return
        run_task = getattr(page, "run_task", None)
        if callable(run_task):
            run_task(self._restore_scroll, offset)
            return
        with contextlib.suppress(Exception):
            result = self.scroll_to(offset=offset, duration=0)
            if inspect.isawaitable(result):
                result.close()

    async def _restore_scroll(self, offset: float) -> None:
        result = self.scroll_to(offset=offset, duration=0)
        if inspect.isawaitable(result):
            await result


class _BottomDockedTextTab(ft.Container):
    def __init__(
        self,
        *,
        tab_key: str,
        label: str,
        font_family: str | None,
        active: bool,
        on_select: Callable[[str], None],
    ) -> None:
        self.tab_key = tab_key
        self._active = active
        self._on_select = on_select
        self.label = ft.Text(
            value=label,
            size=20,
            weight=ft.FontWeight.W_600,
            font_family=font_family,
            color=COLOR_PRIMARY if active else COLOR_SECONDARY,
            text_align=ft.TextAlign.CENTER,
            max_lines=1,
            overflow=ft.TextOverflow.ELLIPSIS,
        )
        super().__init__(
            content=self.label,
            expand=True,
            alignment=ft.Alignment.CENTER,
            on_click=lambda _e: self._on_select(self.tab_key),
            on_hover=self._handle_hover,
        )

    @property
    def text(self) -> str:
        return self.label.value

    def set_active(self, active: bool) -> None:
        self._active = active
        self.label.color = COLOR_PRIMARY if active else COLOR_SECONDARY
        self._refresh()

    def set_font_family(self, font_family: str | None) -> None:
        self.label.font_family = font_family
        self._refresh()

    def set_label(self, label: str) -> None:
        self.label.value = label
        self._refresh()

    def _handle_hover(self, e) -> None:
        if self._active:
            return
        self.label.color = COLOR_PRIMARY if is_hover_active(e) else COLOR_SECONDARY
        self._refresh()

    def _refresh(self) -> None:
        update_control_if_mounted(self)


class TextSubtabShell(ft.Column):
    def __init__(
        self,
        *,
        title: ft.Control | None = None,
        tabs: Sequence[TextSubtab],
        font_family: str | None = None,
        initial_key: str | None = None,
        subtab_bar_position: Literal["top", "bottom"] = "top",
        on_tab_change: Callable[[str], None] | None = None,
    ) -> None:
        if not tabs:
            raise ValueError("TextSubtabShell requires at least one tab")
        if subtab_bar_position not in {"top", "bottom"}:
            raise ValueError(f"Unknown subtab bar position: {subtab_bar_position}")

        keys = tuple(tab.key for tab in tabs)
        if len(set(keys)) != len(keys):
            raise ValueError("TextSubtabShell requires unique tab keys")

        self._font_family = font_family
        self._on_tab_change = on_tab_change
        self.subtab_bar_position = subtab_bar_position
        self.tab_order = keys
        if initial_key is not None and initial_key not in self.tab_order:
            raise ValueError(f"Unknown initial tab key: {initial_key}")
        self.active_key = initial_key or self.tab_order[0]
        self.scroll_offsets = {tab.key: 0.0 for tab in tabs}

        self.title_region = (
            ft.Container(content=title, padding=ft.Padding.only(top=4, bottom=4))
            if title is not None
            else None
        )
        self.button_by_key = {tab.key: self._build_button(tab.key, tab.label) for tab in tabs}
        self.subtab_row = ft.Row(
            controls=self._build_subtab_row_controls(),
            expand=self._is_bottom_docked,
            spacing=0 if self._is_bottom_docked else 8,
            wrap=False,
            scroll=None if self._is_bottom_docked else ft.ScrollMode.AUTO,
        )
        self.subtab_bar = ft.Container(
            content=self.subtab_row,
            bgcolor=COLOR_BACKGROUND if self._is_bottom_docked else COLOR_SURFACE,
            border=self._build_subtab_bar_border(),
            border_radius=None if self._is_bottom_docked else 24,
            height=64 if self._is_bottom_docked else None,
            padding=(
                None if self._is_bottom_docked else ft.Padding.symmetric(horizontal=8, vertical=8)
            ),
        )
        self.body_by_key = {
            tab.key: _ScrollBody(
                tab.key,
                tab.controls,
                on_scroll=lambda e, tab_key=tab.key: self.record_scroll(tab_key, e),
            )
            for tab in tabs
        }
        self.body_host = ft.Stack(controls=list(self.body_by_key.values()), expand=True)
        self.body_region = (
            ft.Container(
                content=self.body_host,
                expand=True,
                padding=ft.Padding.only(left=16, top=16, right=16),
            )
            if self._is_bottom_docked
            else self.body_host
        )

        shell_controls: list[ft.Control] = []
        if self.title_region is not None:
            shell_controls.append(self.title_region)
        if self.subtab_bar_position == "bottom":
            shell_controls.extend([self.body_region, self.subtab_bar])
        else:
            shell_controls.extend([self.subtab_bar, self.body_region])

        super().__init__(
            controls=shell_controls,
            expand=True,
            spacing=0 if self._is_bottom_docked else 16,
        )
        self._apply_button_states()
        self._set_visible_body(self.active_key)

    @property
    def _is_bottom_docked(self) -> bool:
        return self.subtab_bar_position == "bottom"

    def _build_subtab_bar_border(self) -> ft.Border | None:
        if self._is_bottom_docked:
            return ft.Border.only(top=ft.BorderSide(1, COLOR_DIVIDER))
        return None

    def _build_subtab_row_controls(self) -> list[ft.Control]:
        if not self._is_bottom_docked:
            return [self.button_by_key[key] for key in self.tab_order]

        controls: list[ft.Control] = []
        for index, key in enumerate(self.tab_order):
            controls.append(self.button_by_key[key])
            if index < len(self.tab_order) - 1:
                controls.append(ft.VerticalDivider(width=1, color=COLOR_DIVIDER, thickness=1))
        return controls

    def _button_style(self, *, active: bool) -> ft.ButtonStyle:
        if self._is_bottom_docked:
            default_text = COLOR_PRIMARY if active else COLOR_SECONDARY
            hovered_text = COLOR_PRIMARY
            background = ft.Colors.TRANSPARENT
            shape = ft.RoundedRectangleBorder(radius=0)
            padding = ft.Padding.symmetric(horizontal=12, vertical=16)
        else:
            default_text = COLOR_ON_PRIMARY_CONTAINER if active else COLOR_ON_BACKGROUND
            hovered_text = default_text
            background = COLOR_PRIMARY_CONTAINER if active else ft.Colors.TRANSPARENT
            shape = ft.RoundedRectangleBorder(radius=18)
            padding = ft.Padding.symmetric(horizontal=18, vertical=12)

        return ft.ButtonStyle(
            color={
                ft.ControlState.DEFAULT: default_text,
                ft.ControlState.HOVERED: hovered_text,
            },
            bgcolor={
                ft.ControlState.DEFAULT: background,
                ft.ControlState.HOVERED: background,
            },
            text_style=ft.TextStyle(
                size=16,
                weight=ft.FontWeight.W_600,
                font_family=self._font_family,
            ),
            overlay_color=ft.Colors.TRANSPARENT,
            padding=padding,
            shape=shape,
            animation_duration=0,
        )

    def _build_button(self, key: str, label: str) -> ft.Control:
        if self._is_bottom_docked:
            return _BottomDockedTextTab(
                tab_key=key,
                label=label,
                font_family=self._font_family,
                active=key == self.active_key,
                on_select=self.select_tab,
            )
        return ft.TextButton(
            content=label,
            on_click=lambda _e, tab_key=key: self.select_tab(tab_key),
            expand=self._is_bottom_docked,
            style=self._button_style(active=key == self.active_key),
        )

    def _apply_button_states(self) -> None:
        for key, button in self.button_by_key.items():
            if isinstance(button, _BottomDockedTextTab):
                button.set_active(key == self.active_key)
            else:
                button.style = self._button_style(active=key == self.active_key)

    def _set_visible_body(self, key: str) -> None:
        for tab_key, body in self.body_by_key.items():
            body.visible = tab_key == key

    def set_font_family(self, font_family: str | None) -> None:
        self._font_family = font_family
        for button in self.button_by_key.values():
            if isinstance(button, _BottomDockedTextTab):
                button.set_font_family(font_family)
        self._apply_button_states()

    def set_tab_label(self, key: str, label: str) -> None:
        button = self.button_by_key[key]
        if isinstance(button, _BottomDockedTextTab):
            button.set_label(label)
        else:
            button.content = label

    def select_tab(self, key: str) -> None:
        if key not in self.body_by_key or key == self.active_key:
            return
        self.active_key = key
        self._set_visible_body(key)
        self._apply_button_states()
        self.body_by_key[key].restore_scroll(self.scroll_offsets.get(key, 0.0))
        if self._on_tab_change is not None:
            self._on_tab_change(key)
        update_control_if_mounted(self)

    def record_scroll(self, key: str, e) -> None:
        if key not in self.scroll_offsets:
            return
        self.scroll_offsets[key] = float(getattr(e, "pixels", 0.0) or 0.0)
