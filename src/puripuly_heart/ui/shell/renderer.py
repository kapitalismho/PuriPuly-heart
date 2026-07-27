from __future__ import annotations

import flet as ft

from puripuly_heart.ui.shell.contract import AppShellRegions, AppShellSlots

APP_SHELL_LAYOUT_SPACING = 0
APP_SHELL_ROOT_PADDING = 0


def compose_app_shell(slots: AppShellSlots) -> AppShellRegions:
    content_area = ft.Container(
        expand=True,
        padding=slots.content_padding,
        content=slots.content,
    )
    layout = ft.Column(
        controls=[slots.title_bar, content_area, slots.bottom_nav],
        expand=True,
        spacing=APP_SHELL_LAYOUT_SPACING,
    )
    root_content = ft.Container(content=layout, expand=True, padding=APP_SHELL_ROOT_PADDING)

    if slots.debug_panel is None:
        return AppShellRegions(
            root=root_content,
            layout=layout,
            content_area=content_area,
            debug_stack=None,
        )

    debug_stack = ft.Stack(
        controls=[root_content, slots.debug_panel],
        fit=ft.StackFit.EXPAND,
        expand=True,
    )
    return AppShellRegions(
        root=ft.Container(
            content=debug_stack,
            expand=True,
            padding=APP_SHELL_ROOT_PADDING,
        ),
        layout=layout,
        content_area=content_area,
        debug_stack=debug_stack,
    )


__all__ = [
    "APP_SHELL_LAYOUT_SPACING",
    "APP_SHELL_ROOT_PADDING",
    "compose_app_shell",
]
