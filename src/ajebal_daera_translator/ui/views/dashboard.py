import flet as ft
from flet.core.icons import Icons as icons
from flet.core.colors import Colors as colors
from ajebal_daera_translator.ui.theme import COLOR_SURFACE, COLOR_ON_BACKGROUND, COLOR_PRIMARY, COLOR_SUCCESS, COLOR_ERROR
from ajebal_daera_translator.ui.components.bento_card import BentoCard

class DashboardView(ft.Column):
    LANG_LABEL_TO_CODE = {
        "Korean": "ko-KR",
        "English": "en-US",
        "Japanese": "ja-JP",
    }
    LANG_CODE_TO_LABEL = {v: k for k, v in LANG_LABEL_TO_CODE.items()}

    def __init__(self):
        super().__init__(expand=True, spacing=15) # increased spacing for grid gaps
        # State placeholders
        self.is_connected = False
        self.is_power_on = False
        self.last_sent_text = "Ready to translate..."
        self.history_items = []
        self.on_send_message = None # Callback function assigned by App
        self.on_toggle_translation = None
        self.on_toggle_stt = None
        self.on_language_change = None
        
        self._build_ui()

    def _build_ui(self):
        # 1. Status Card (Left Top)
        self.status_indicator = ft.Icon(
            name=icons.CIRCLE,
            size=12,
            color=COLOR_ERROR,  # Default disconnected
            tooltip="VRChat Status: Disconnected"
        )
        self.status_text = ft.Text("Disconnected", size=14, color=colors.GREY_400, weight=ft.FontWeight.W_500)
        
        status_content = ft.Row(
            controls=[
                self.status_indicator,
                ft.Column([
                    ft.Text("SYSTEM STATUS", size=10, color=colors.GREY_500, weight=ft.FontWeight.BOLD),
                    self.status_text
                ], spacing=2)
            ],
            alignment=ft.MainAxisAlignment.START,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
        # Status card
        status_card = BentoCard(status_content)

         # 2. Language Control Section (Left Bottom)
        self.source_lang = ft.Dropdown(
            label="Source", 
            options=[ft.dropdown.Option("Korean"), ft.dropdown.Option("English"), ft.dropdown.Option("Japanese")],
            value="Korean", 
            expand=True, 
            text_size=15, 
            border_radius=12,
            content_padding=12,
            filled=True,
            bgcolor=colors.GREY_800,
            color=colors.WHITE,
            border_color=colors.GREY_700,
            focused_bgcolor=colors.GREY_700,
            on_change=self._on_lang_change,
        )
        
        self.target_lang = ft.Dropdown(
            label="Target", 
            options=[ft.dropdown.Option("English"), ft.dropdown.Option("Japanese"), ft.dropdown.Option("Korean")],
            value="English", 
            expand=True, 
            text_size=15, 
            border_radius=12,
            content_padding=12,
            filled=True,
            bgcolor=colors.GREY_800,
            color=colors.WHITE,
            border_color=colors.GREY_700,
            focused_bgcolor=colors.GREY_700,
            on_change=self._on_lang_change,
        )
        
        # Segmented Control for Presets
        self.presets = [
            {"label": "KR → EN", "src": "Korean", "tgt": "English"},
            {"label": "EN → KR", "src": "English", "tgt": "Korean"},
            {"label": "KR → JP", "src": "Korean", "tgt": "Japanese"},
        ]
        self.preset_controls = [] # To hold the individual preset clickable containers
        
        # Build the segmented control container
        preset_segment = ft.Container(
            content=ft.Row(
                controls=self._build_preset_controls(),
                spacing=0, # Joined together
                alignment=ft.MainAxisAlignment.CENTER,
            ),
            bgcolor=colors.with_opacity(0.38, colors.BLACK), # Darker track
            border_radius=12,
            padding=4, # Inner padding for the "floating pill" look
        )

        lang_content = ft.Column(
            controls=[
                ft.Text("TRANSLATION PAIR", size=11, color=colors.GREY_500, weight=ft.FontWeight.BOLD),
                ft.Row([
                    self.source_lang,
                    ft.Icon(name=icons.ARROW_FORWARD_ROUNDED, color=colors.GREY_500),
                    self.target_lang,
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN, vertical_alignment=ft.CrossAxisAlignment.CENTER, spacing=10),
                preset_segment # Add segmented control
            ],
            spacing=15,
            expand=True, # Content expands
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH
        )
        
        # Language card
        lang_card = BentoCard(lang_content, expand=True)

        # 3. Power Card (Right Tall)
        # 3. Power Control Panel (Right Tall)
        
        # State Initialization (if not already done)
        if not hasattr(self, 'is_translation_on'): self.is_translation_on = False
        if not hasattr(self, 'is_stt_on'): self.is_stt_on = False

        # Tile 1: Translation
        self.tile_translation = self._build_power_tile(
            "TRANSLATION", 
            icons.TRANSLATE_ROUNDED, 
            self.is_translation_on, 
            self._toggle_translation
        )
        
        # Tile 2: STT
        self.tile_stt = self._build_power_tile(
            "VOICE (STT)", 
            icons.MIC_ROUNDED, 
            self.is_stt_on, 
            self._toggle_stt
        )

        power_content = ft.Column(
            controls=[
                ft.Row([
                    ft.Icon(name=icons.POWER_SETTINGS_NEW_ROUNDED, color=colors.GREY_500, size=16),
                    ft.Text("SYSTEM POWER", size=10, color=colors.GREY_500, weight=ft.FontWeight.BOLD),
                ], spacing=5, alignment=ft.MainAxisAlignment.CENTER),
                
                # Two side-by-side control blocks
                ft.Row([
                    self.tile_translation,
                    self.tile_stt,
                ], spacing=10, expand=True)

            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=15
        )
        # Power card fills vertical space
        power_card = BentoCard(power_content, expand=True)

        # --- GRID ASSEMBLY ---
        
        # Left Column: Status + Lang
        left_column = ft.Column(
            controls=[status_card, lang_card],
            spacing=15,
            expand=2 
        )

        # Top Grid
        top_grid = ft.Row(
            controls=[
                left_column, 
                ft.Container(content=power_card, expand=1) 
            ],
            spacing=15,
            vertical_alignment=ft.CrossAxisAlignment.STRETCH,
            expand=True 
        )
        
        # 4. Hero Section
        self.hero_text = ft.Text(
            self.last_sent_text,
            size=28,
            weight=ft.FontWeight.BOLD,
            color=colors.WHITE,
            text_align=ft.TextAlign.CENTER,
            selectable=True,
        )
        
        hero_content = ft.Column(
            controls=[
                ft.Text("LAST TRANSLATED", size=11, color=colors.GREY_500, weight=ft.FontWeight.BOLD),
                ft.Container(content=self.hero_text, alignment=ft.alignment.center, expand=True)
            ],
            expand=True
        )
        hero_card = BentoCard(hero_content, height=180)
 
        # 5. Input Area
        self.input_field = ft.TextField(
            hint_text="Type message to send...",
            border_radius=12,
            bgcolor=colors.with_opacity(0.05, colors.WHITE),
            border_color=colors.TRANSPARENT,
            expand=True,
            content_padding=15,
            text_size=14,
            on_submit=self._on_submit
        )

        send_button = ft.IconButton(
            icon=icons.SEND_ROUNDED,
            icon_color=COLOR_PRIMARY,
            on_click=self._on_submit,
            bgcolor=colors.with_opacity(0.1, COLOR_PRIMARY),
            icon_size=20,
        )

        input_row = ft.Row(
            controls=[self.input_field, send_button],
            spacing=10,
            vertical_alignment=ft.CrossAxisAlignment.CENTER
        )
        
        input_card = BentoCard(input_row, height=80)

        # Main Layout Assembly
        self.controls = [
            top_grid,
            hero_card, 
            input_card
        ]

    def _build_preset_controls(self):
        controls = []
        for i, preset in enumerate(self.presets):
            # Check if this preset is currently active
            is_active = (self.source_lang.value == preset['src'] and 
                         self.target_lang.value == preset['tgt'])
            
            btn = ft.Container(
                content=ft.Text(
                    preset['label'], 
                    size=12, 
                    weight=ft.FontWeight.BOLD if is_active else ft.FontWeight.NORMAL,
                    color=colors.WHITE if is_active else colors.GREY_500,
                    text_align=ft.TextAlign.CENTER
                ),
                data=i, # Store index
                expand=True, # Each takes equal width
                padding=ft.padding.symmetric(vertical=8),
                bgcolor=colors.with_opacity(0.2, COLOR_PRIMARY) if is_active else colors.TRANSPARENT,
                border_radius=8,
                on_click=self._on_preset_click,
                animate=ft.Animation(200, ft.AnimationCurve.EASE_OUT),
            )
            controls.append(btn)
        self.preset_controls = controls
        return controls

    def _on_preset_click(self, e):
        index = e.control.data
        preset = self.presets[index]
        
        # Apply Logic
        self.source_lang.value = preset['src']
        self.target_lang.value = preset['tgt']
        
        # Save
        self._on_lang_change(None)
        
        # Update Visuals
        self._update_preset_visuals()
        self.update()

    def _update_preset_visuals(self):
        for i, btn in enumerate(self.preset_controls):
            preset = self.presets[i]
            is_active = (self.source_lang.value == preset['src'] and 
                         self.target_lang.value == preset['tgt'])
            
            btn.bgcolor = colors.with_opacity(0.6, COLOR_PRIMARY) if is_active else colors.TRANSPARENT
            # Update Text Style (Need to access content)
            txt = btn.content
            txt.weight = ft.FontWeight.BOLD if is_active else ft.FontWeight.NORMAL
            txt.color = colors.WHITE if is_active else colors.GREY_500
            btn.update() # Update individual control for performance? Or parent. 
            # Flet often needs parent update or page update if deep property change.
            # safe to just update main view in event handler.

    def _on_lang_change(self, e):
        # Update visuals when manually changed
        if e is not None:
            self._update_preset_visuals()
            self.update()

        if self.on_language_change:
            source_code = self.LANG_LABEL_TO_CODE.get(self.source_lang.value, "ko-KR")
            target_code = self.LANG_LABEL_TO_CODE.get(self.target_lang.value, "en-US")
            self.on_language_change(source_code, target_code)
        
    def _build_power_tile(self, label, icon_name, is_active, on_click):
        icon_color = colors.WHITE if is_active else colors.GREY_500
        text_color = colors.WHITE if is_active else colors.GREY_500
        bg_color = COLOR_SUCCESS if is_active else colors.with_opacity(0.05, colors.WHITE)
        
        if label == "VOICE (STT)" and is_active:
             bg_color = COLOR_PRIMARY # Distinguish STT active color if desired, or keep uniform
        
        return ft.Container(
            content=ft.Column([
                ft.Icon(name=icon_name, size=28, color=icon_color),
                ft.Text(label, size=10, weight=ft.FontWeight.BOLD, color=text_color),
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=5, alignment=ft.MainAxisAlignment.CENTER),
            padding=10,
            bgcolor=bg_color,
            border_radius=12,
            expand=True,
            on_click=on_click,
            animate=ft.Animation(200, ft.AnimationCurve.EASE_OUT),
            data=label # Store label to ID if needed
        )

    def _toggle_translation(self, e):
        self.is_translation_on = not self.is_translation_on
        self._update_tile_visuals(self.tile_translation, self.is_translation_on, is_stt=False)
        self.is_power_on = self.is_translation_on # Sync main power flag relative to Translation
        self.update()
        if self.on_toggle_translation:
            self.on_toggle_translation(self.is_translation_on)

    def _toggle_stt(self, e):
        self.is_stt_on = not self.is_stt_on
        self._update_tile_visuals(self.tile_stt, self.is_stt_on, is_stt=True)
        self.update()
        if self.on_toggle_stt:
            self.on_toggle_stt(self.is_stt_on)

    def _update_tile_visuals(self, tile, is_active, is_stt=False):
        # Determine colors
        if is_active:
            bg_color = COLOR_PRIMARY if is_stt else COLOR_SUCCESS
            fg_color = colors.WHITE
        else:
            bg_color = colors.with_opacity(0.05, colors.WHITE)
            fg_color = colors.GREY_500
            
        tile.bgcolor = bg_color
        # Update content (Icon and Text)
        col = tile.content
        icon = col.controls[0]
        text = col.controls[1]
        
        icon.color = fg_color
        text.color = fg_color
        tile.update()

    def _on_submit(self, e):
        text = self.input_field.value
        if not text:
            return
        
        # Update UI locally
        self.hero_text.value = text
        self.input_field.value = ""
        self.input_field.focus()
        self.update()
        
        # Propagate to App logic (for History)
        if self.on_send_message:
            self.on_send_message("You", text)

    def set_status(self, connected: bool):
        self.is_connected = connected
        self.status_indicator.color = COLOR_SUCCESS if connected else COLOR_ERROR
        self.status_text.value = "Connected" if connected else "Disconnected"
        self.status_text.color = COLOR_SUCCESS if connected else colors.GREY_400
        self.update()

    def set_languages_from_codes(self, source_code: str, target_code: str) -> None:
        src_label = self.LANG_CODE_TO_LABEL.get(source_code, "Korean")
        tgt_label = self.LANG_CODE_TO_LABEL.get(target_code, "English")
        self.source_lang.value = src_label
        self.target_lang.value = tgt_label
        self._update_preset_visuals()
        self.update()

    def set_translation_enabled(self, enabled: bool) -> None:
        self.is_translation_on = bool(enabled)
        self._update_tile_visuals(self.tile_translation, self.is_translation_on, is_stt=False)
        self.update()

    def set_stt_enabled(self, enabled: bool) -> None:
        self.is_stt_on = bool(enabled)
        self._update_tile_visuals(self.tile_stt, self.is_stt_on, is_stt=True)
        self.update()
