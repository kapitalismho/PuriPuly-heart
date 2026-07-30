from __future__ import annotations

from collections.abc import Sequence

import flet as ft

from puripuly_heart.ui.components.shared_card_wrapper import SharedCardWrapper


class SettingsUnitCard(SharedCardWrapper):
    """Settings-specific 1x1 card with a strong title/value rhythm."""

    DEFAULT_HEIGHT = 228

    def __init__(
        self,
        *,
        title: ft.Control,
        value: ft.Control,
        extra_controls: Sequence[ft.Control] = (),
        height: float | int | None = DEFAULT_HEIGHT,
    ) -> None:
        controls: list[ft.Control] = [
            title,
            ft.Container(content=value, expand=True, alignment=ft.Alignment.CENTER),
        ]
        if extra_controls:
            controls.append(ft.Container(height=12))
            controls.extend(extra_controls)

        super().__init__(
            ft.Column(
                controls=controls,
                spacing=0,
                expand=True,
            ),
            expand=True,
            height=height,
        )
