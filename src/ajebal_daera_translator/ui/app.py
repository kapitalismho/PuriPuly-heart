import flet as ft
import logging

from ajebal_daera_translator.ui.theme import get_app_theme, COLOR_BACKGROUND, COLOR_DIVIDER
from ajebal_daera_translator.ui.components.sidebar import AppSidebar
from ajebal_daera_translator.ui.views.dashboard import DashboardView
from ajebal_daera_translator.ui.views.settings import SettingsView
from ajebal_daera_translator.ui.views.logs import LogsView
from ajebal_daera_translator.ui.views.history import HistoryView  # Import HistoryView
from ajebal_daera_translator.ui.controller import GuiController
from ajebal_daera_translator.core.language import get_stt_compatibility_warning

logger = logging.getLogger(__name__)

class TranslatorApp:
    def __init__(self, page: ft.Page, *, config_path):
        self.page = page
        self.controller = GuiController(page=page, app=self, config_path=config_path)
        self._setup_page()
        self._build_layout()
        
        # Link Dashboard submit to App's history handler
        self.view_dashboard.on_send_message = self._on_manual_submit
        self.view_dashboard.on_toggle_translation = self._on_translation_toggle
        self.view_dashboard.on_toggle_stt = self._on_stt_toggle
        self.view_dashboard.on_language_change = self._on_language_change

        self.view_settings.on_settings_changed = self._on_settings_changed
        self.view_settings.on_providers_changed = self._on_providers_changed
        self.view_settings.on_verify_api_key = self._on_verify_api_key

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

    def _on_manual_submit(self, _source: str, text: str) -> None:
        # UI already wrote to hero/history; pipeline should run asynchronously.
        async def _task():
            await self.controller.submit_text(text)
        self.page.run_task(_task)

    def _on_translation_toggle(self, enabled: bool) -> None:
        async def _task():
            await self.controller.set_translation_enabled(enabled)
        self.page.run_task(_task)

    def _on_stt_toggle(self, enabled: bool) -> None:
        async def _task():
            await self.controller.set_stt_enabled(enabled)
        self.page.run_task(_task)

    def _on_language_change(self, source_code: str, target_code: str) -> None:
        if self.controller.settings is None:
            return
        settings = self.controller.settings
        settings.languages.source_language = source_code
        settings.languages.target_language = target_code

        # Check STT provider compatibility and show warning if needed
        stt_provider = settings.provider.stt.value  # "deepgram" or "qwen_asr"
        warning = get_stt_compatibility_warning(source_code, stt_provider)
        if warning:
            from flet.core.colors import Colors as colors
            self.page.open(ft.SnackBar(
                ft.Text(warning),
                bgcolor=colors.ORANGE_700,
                duration=4000,
            ))

        async def _task():
            await self.controller.apply_settings(settings)
        self.page.run_task(_task)

    def _on_settings_changed(self, settings) -> None:
        async def _task():
            await self.controller.apply_settings(settings)
        self.page.run_task(_task)

    def _on_providers_changed(self) -> None:
        async def _task():
            await self.controller.apply_providers()
        self.page.run_task(_task)

    async def _on_verify_api_key(self, provider: str, key: str) -> tuple[bool, str]:
        return await self.controller.verify_api_key(provider, key)

async def main_gui(page: ft.Page, *, config_path):
    app = TranslatorApp(page, config_path=config_path)
    await app.controller.start()
