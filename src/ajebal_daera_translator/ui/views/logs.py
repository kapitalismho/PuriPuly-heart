import flet as ft
from ajebal_daera_translator.ui.theme import COLOR_SURFACE

class LogsView(ft.Container):
    def __init__(self):
        super().__init__(
            bgcolor=COLOR_SURFACE,
            padding=10,
            border_radius=10,
            expand=True,
        )
        self.log_list = ft.ListView(expand=True, spacing=5, auto_scroll=True)
        self.content = self.log_list

    def append_log(self, record: str):
        self.log_list.controls.append(
            ft.Text(record, size=12, font_family="Consolas")
        )
        self.update()
