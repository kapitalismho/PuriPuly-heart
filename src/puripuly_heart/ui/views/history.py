import flet as ft
from flet import Colors as colors

from puripuly_heart.ui.components.bento_card import BentoCard
from puripuly_heart.ui.theme import COLOR_ON_BACKGROUND, COLOR_PRIMARY


class HistoryView(ft.Container):
    def __init__(self):
        super().__init__(expand=True)
        self.history_list = ft.ListView(
            expand=True,
            spacing=10,
            padding=10,
            auto_scroll=True,
        )

        self.content = BentoCard(
            content=ft.Column(
                [
                    ft.Text(
                        "CONVERSATION HISTORY",
                        size=14,
                        color=colors.GREY_500,
                        weight=ft.FontWeight.BOLD,
                    ),
                    self.history_list,
                ],
                expand=True,
            ),
            expand=True,
        )

    def add_message(self, source: str, text: str):
        self.history_list.controls.append(
            ft.Container(
                content=ft.Column(
                    [
                        ft.Text(source, size=10, color=COLOR_PRIMARY),
                        ft.Text(text, size=14, color=COLOR_ON_BACKGROUND),
                    ],
                    spacing=2,
                ),
                bgcolor=colors.with_opacity(0.05, colors.WHITE),
                padding=12,
                border_radius=8,
            )
        )
        if self.page is not None:
            self.update()
