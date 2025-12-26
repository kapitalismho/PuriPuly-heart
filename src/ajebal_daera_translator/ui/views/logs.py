import logging

import flet as ft

from ajebal_daera_translator.ui.theme import COLOR_SURFACE


class FletLogHandler(logging.Handler):
    """Custom log handler that forwards logs to a LogsView."""

    def __init__(self, logs_view: "LogsView"):
        super().__init__()
        self.logs_view = logs_view
        self.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.logs_view.append_log(msg)
        except Exception:
            pass  # Ignore errors during logging


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
        self._handler: FletLogHandler | None = None

    def attach_log_handler(self) -> None:
        """Attach this view as a logging handler to capture app logs."""
        if self._handler is not None:
            return  # Already attached
        self._handler = FletLogHandler(self)
        logging.getLogger().addHandler(self._handler)

    def append_log(self, record: str):
        self.log_list.controls.append(
            ft.Text(record, size=12, font_family="Consolas", selectable=True)
        )
        if self.page:
            self.update()
