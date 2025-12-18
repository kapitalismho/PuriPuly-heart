import flet as ft
import logging

from ajebal_daera_translator.ui.theme import get_app_theme, COLOR_BACKGROUND, COLOR_DIVIDER
from ajebal_daera_translator.ui.components.sidebar import AppSidebar
from ajebal_daera_translator.ui.views.dashboard import DashboardView
from ajebal_daera_translator.ui.views.settings import SettingsView
from ajebal_daera_translator.ui.views.logs import LogsView
from ajebal_daera_translator.ui.views.history import HistoryView  # Import HistoryView

logger = logging.getLogger(__name__)

class TranslatorApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self._setup_page()
        self._build_layout()
        
        # Link Dashboard submit to App's history handler
        self.view_dashboard.on_send_message = self.add_history_entry

    def _setup_page(self):
        self.page.title = "A-Jebal-Daera Translator"
        self.page.theme_mode = ft.ThemeMode.DARK
        self.page.theme = get_app_theme()
        self.page.bgcolor = COLOR_BACKGROUND
        self.page.padding = 0
        self.page.window_min_width = 800
        self.page.window_min_height = 600

    def _build_layout(self):
        # Initialize Views
        self.view_dashboard = DashboardView()
        self.view_history = HistoryView() # Init History
        self.view_settings = SettingsView()
        self.view_logs = LogsView()

        self.sidebar = AppSidebar(on_change=self._on_nav_change)
        
        self.content_area = ft.Container(
            expand=True,
            padding=0,
            content=self.view_dashboard  # Default view
        )

        self.layout = ft.Row(
            controls=[
                self.sidebar,
                self.content_area,
            ],
            expand=True,
            spacing=20,
        )
        self.page.add(ft.Container(
            content=self.layout,
            expand=True,
            padding=20 
        ))

    def _on_nav_change(self, index: int):
        if index == 0:
            self.content_area.content = self.view_dashboard
        elif index == 1: # New History Tab
            self.content_area.content = self.view_history
        elif index == 2:
            self.content_area.content = self.view_settings
        elif index == 3:
            self.content_area.content = self.view_logs
        
        self.content_area.update()

    def add_history_entry(self, source: str, text: str):
        # Update History View
        self.view_history.add_message(source, text)
        
        # Also update Dashboard's hero text if needed (it does it locally, but good to know)


async def main_gui(page: ft.Page):
    app = TranslatorApp(page)
    # Ensure asyncio loop runs smoothly if needed
    # page.run_task(task) could be used here for event bridge later
