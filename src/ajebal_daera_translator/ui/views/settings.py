import flet as ft
from flet.core.colors import Colors as colors
from flet.core.icons import Icons as icons
from ajebal_daera_translator.ui.theme import COLOR_SURFACE, COLOR_PRIMARY
from ajebal_daera_translator.ui.components.bento_card import BentoCard

class SettingsView(ft.ListView): # Scrollable settings
    def __init__(self):
        super().__init__(expand=True, spacing=15, padding=10)
        # Control References for Logic
        self.provider_dropdown = ft.Dropdown(
            label="Provider", 
            options=[ft.dropdown.Option("Google (Gemini)"), ft.dropdown.Option("Alibaba (Qwen)")], 
            value="Google (Gemini)", 
            text_size=14, 
            border_radius=8,
            on_change=self._on_provider_change
        )
        self.llm_key_field = ft.TextField(
            label="LLM API Key", 
            password=True, 
            can_reveal_password=True, 
            text_size=14, 
            border_radius=8,
            on_change=lambda e: self._on_key_change("llm")
        )
        self.stt_key_field = ft.TextField(
            label="STT API Key", 
            password=True, 
            can_reveal_password=True, 
            text_size=14, 
            border_radius=8,
            on_change=lambda e: self._on_key_change("stt")
        )

        self.controls = [
            ft.Text("Settings", size=20, weight=ft.FontWeight.BOLD, color=colors.WHITE),
            self._build_section("Audio Configuration", [
                ft.Dropdown(label="Audio Host API", options=[
                    ft.dropdown.Option("MME"), 
                    ft.dropdown.Option("DirectSound"),
                    ft.dropdown.Option("WASAPI"),
                    ft.dropdown.Option("ASIO")
                ], value="WASAPI", text_size=14, border_radius=8),
                
                ft.Dropdown(label="Microphone", options=[ft.dropdown.Option("Default Mic")], value="Default Mic", text_size=14, border_radius=8),
                
                ft.Text("VAD Sensitivity (Speech Detection)", size=12, color=colors.GREY_400),
                ft.Slider(
                    min=0.0, max=1.0, divisions=10, value=0.5, 
                    label="Sensitivity: {value}",
                    active_color=COLOR_PRIMARY
                )
            ]),
            self._build_section("AI Service Setup", [
                ft.Text("Language Model (LLM)", size=12, color=colors.GREY_400),
                self.provider_dropdown,
                self.llm_key_field,
                
                ft.Divider(height=5, color=colors.TRANSPARENT),
                
                ft.Text("Speech-to-Text (STT)", size=12, color=colors.GREY_400),
                self.stt_key_field,
                
                ft.Container(height=10),
                ft.ElevatedButton(
                    text="Verify & Authenticate Keys",
                    icon=icons.VERIFIED_USER_ROUNDED,
                    style=ft.ButtonStyle(
                        color=colors.WHITE,
                        bgcolor=COLOR_PRIMARY,
                        shape=ft.RoundedRectangleBorder(radius=8),
                    ),
                    on_click=self._on_validate_keys
                )
            ]),
            self._build_section("Persona", [
                ft.TextField(label="System Prompt", multiline=True, min_lines=3, value="You are a helpful translator.", text_size=14, border_radius=8),
            ]),
        ]

    def did_mount(self):
        # Initial Load
        self._load_keys_for_provider(self.provider_dropdown.value)

    def _on_provider_change(self, e):
        # Provider changed, load keys for the new provider
        # Note: Previous keys were already saved by _on_key_change
        new_provider = self.provider_dropdown.value
        self._load_keys_for_provider(new_provider)
        self.update()

    def _load_keys_for_provider(self, provider):
        prefix = "google" if "Google" in provider else "alibaba"
        
        # Load from storage (silent if missing)
        llm_key = self.page.client_storage.get(f"{prefix}_llm_key") or ""
        stt_key = self.page.client_storage.get(f"{prefix}_stt_key") or ""
        
        self.llm_key_field.value = llm_key
        self.stt_key_field.value = stt_key
        # Update references to current prefix could be useful but we calculate on save

    def _on_key_change(self, key_type):
        # Save immediately to current provider slot
        provider = self.provider_dropdown.value
        prefix = "google" if "Google" in provider else "alibaba"
        
        storage_key = f"{prefix}_{key_type}_key"
        value = self.llm_key_field.value if key_type == "llm" else self.stt_key_field.value
        
        if self.page:
            self.page.client_storage.set(storage_key, value)

    def _on_validate_keys(self, e):
        # Placeholder validation logic
        llm = self.llm_key_field.value
        stt = self.stt_key_field.value
        
        if not llm or not stt:
            if self.page:
                self.page.show_snack_bar(ft.SnackBar(content=ft.Text("Please enter both API keys.")))
            return
            
        # TODO: Actual validation call
        if self.page:
            self.page.show_snack_bar(ft.SnackBar(content=ft.Text("API Keys verified locally (Placeholder).")))

    def _build_section(self, title: str, controls: list[ft.Control]):
        return BentoCard(
            content=ft.Column([
                ft.Text(title, size=12, weight=ft.FontWeight.BOLD, color=colors.GREY_500),
                ft.Column(controls, spacing=15)
            ])
        )
