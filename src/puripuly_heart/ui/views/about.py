"""About page view with version, credits, acknowledgments, and license info."""

import webbrowser
from importlib import resources

import flet as ft

from puripuly_heart import __version__
from puripuly_heart.ui.components.shared_card_wrapper import SharedCardWrapper
from puripuly_heart.ui.flet_runtime import is_hover_active
from puripuly_heart.ui.i18n import t
from puripuly_heart.ui.theme import (
    COLOR_DIVIDER,
    COLOR_NEUTRAL,
    COLOR_ON_BACKGROUND,
    COLOR_PRIMARY,
    COLOR_SURFACE,
)

_CENTER_ALIGNMENT = ft.Alignment(0, 0)


def _load_third_party_notices() -> str:
    """Load THIRD_PARTY_NOTICES.txt from package data."""
    try:
        return (
            resources.files("puripuly_heart.data")
            .joinpath("THIRD_PARTY_NOTICES.txt")
            .read_text(encoding="utf-8")
        )
    except Exception:
        return "Could not load license information."


def _get_profile_image_path() -> str:
    """Get the profile image path from package data."""
    try:
        return str(resources.files("puripuly_heart.data.pictures").joinpath("salee_pic.png"))
    except Exception:
        return ""


class AboutView(ft.Column):
    """About page with version, credits, inspired by, special thanks, and licenses."""

    def __init__(self):
        super().__init__(expand=True, scroll=ft.ScrollMode.AUTO, spacing=16)

        self._build_ui()

    def _build_ui(self):
        """Build the About page UI."""
        # First row: Credits + Inspired By (50/50 split, taller like 2 rows)
        top_row = ft.Container(
            content=ft.Row(
                controls=[
                    ft.Container(
                        content=self._build_credits_card(),
                        expand=True,
                    ),
                    ft.Container(
                        content=self._build_inspired_by_card(),
                        expand=True,
                    ),
                ],
                spacing=16,
                expand=True,
            ),
        )

        self.controls = [
            self._build_header(),
            top_row,
            self._build_special_thanks_card(),
            self._build_licenses_card(),
        ]

    def _build_header(self) -> ft.Control:
        """Build app name and version header as two separate cards."""
        # Left card: App name
        app_name_card = self._wrap_card(
            ft.Container(
                content=ft.Text(
                    t("app.title"),
                    size=48,
                    weight=ft.FontWeight.BOLD,
                    color=COLOR_PRIMARY,
                ),
                alignment=_CENTER_ALIGNMENT,
            )
        )

        # Right card: Version (clickable, opens git repo)
        # Same structure as settings view 1x1 boxes
        version_title = ft.Text(
            t("about.version"),
            size=24,
            weight=ft.FontWeight.BOLD,
            color=COLOR_NEUTRAL,
        )
        version_text = ft.Container(
            content=ft.Text(
                f"v{__version__}",
                size=28,
                color=COLOR_ON_BACKGROUND,
                text_align=ft.TextAlign.CENTER,
            ),
            alignment=_CENTER_ALIGNMENT,
            expand=True,
            on_click=lambda _: webbrowser.open("https://github.com/kapitalismho/PuriPuly-heart"),
            on_hover=self._on_version_hover,
        )
        version_card = self._wrap_card(
            ft.Column([version_title, version_text], spacing=0, expand=True)
        )

        return ft.Container(
            content=ft.Row(
                controls=[
                    ft.Container(content=app_name_card, expand=True),
                    ft.Container(content=version_card, expand=True),
                ],
                spacing=16,
                expand=True,
            ),
        )

    def _build_credits_card(self) -> ft.Control:
        """Build credits section with profile picture."""
        profile_path = _get_profile_image_path()

        profile_image = ft.Container(
            content=(
                ft.Image(
                    src=profile_path,
                    width=160,
                    height=160,
                    fit=ft.BoxFit.COVER,
                    border_radius=100,
                )
                if profile_path
                else ft.Icon(ft.Icons.PERSON, size=100, color=COLOR_ON_BACKGROUND)
            ),
            width=160,
            height=160,
            border_radius=100,
            bgcolor=COLOR_DIVIDER,
            clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
        )

        name_link = ft.Container(
            content=ft.Text(
                "salee",
                size=32,
                weight=ft.FontWeight.BOLD,
                color=COLOR_ON_BACKGROUND,
            ),
            on_click=lambda _: webbrowser.open("https://x.com/kapitalismho"),
            on_hover=self._on_name_hover,
        )

        card_content = ft.Column(
            controls=[
                ft.Text(
                    t("about.developed_by"),
                    size=24,
                    weight=ft.FontWeight.BOLD,
                    color=COLOR_NEUTRAL,
                ),
                ft.Container(height=16),
                ft.Row(
                    controls=[
                        profile_image,
                        ft.Container(width=24),
                        name_link,
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
            ],
        )

        return self._wrap_card(card_content)

    def _build_inspired_by_card(self) -> ft.Control:
        """Build Inspired By section with project links."""
        projects = [
            ("VRCT", "https://github.com/misyaguziya/VRCT"),
            ("mimiuchi", "https://github.com/naeruru/mimiuchi"),
            ("Yakutan", "https://github.com/febilly/Yakutan"),
        ]

        project_links = []
        for name, url in projects:
            link = ft.Container(
                content=ft.Text(
                    name,
                    size=28,
                    color=COLOR_ON_BACKGROUND,
                ),
                on_click=lambda _, u=url: webbrowser.open(u),
                on_hover=self._on_link_hover,
            )
            project_links.append(link)

        card_content = ft.Column(
            controls=[
                ft.Text(
                    t("about.inspired_by"),
                    size=24,
                    weight=ft.FontWeight.BOLD,
                    color=COLOR_NEUTRAL,
                ),
                ft.Container(height=16),
                ft.Container(
                    content=ft.Column(
                        controls=project_links,
                        spacing=12,
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    width=float("inf"),
                ),
            ],
        )

        return self._wrap_card(card_content)

    # Special thanks name keys - add new names here and update locale bundles
    _SPECIAL_THANKS_NAME_KEYS = [
        "about.special_thanks.name.sui_32c",
        "about.special_thanks.name.nagikokoro",
        "about.special_thanks.name.motoka96",
        "about.special_thanks.name.ykol",
        "about.special_thanks.name.kascr",
        "about.special_thanks.name.just_monika_v",
        "about.special_thanks.name.fluvia",
        "about.special_thanks.name.han_chole",
        "about.special_thanks.name.ea_pe",
        "about.special_thanks.name.ephedrine",
        "about.special_thanks.name.eri",
        "about.special_thanks.name.fzcfweasdferttgg_png",
        "about.special_thanks.name.welcius",
        "about.special_thanks.name.nunu299",
    ]

    def _build_special_thanks_card(self) -> ft.Control:
        """Build Special Thanks section."""
        thanks_items = []
        for name_key in self._SPECIAL_THANKS_NAME_KEYS:
            name = t(name_key)
            item = ft.Container(
                content=ft.Text(name, size=28, color=COLOR_ON_BACKGROUND),
                on_hover=self._on_thanks_hover,
            )
            thanks_items.append(item)

        # Add "and you!" at the end
        thanks_items.append(ft.Text("\nand you!", size=28, color=COLOR_ON_BACKGROUND, italic=True))

        card_content = ft.Column(
            controls=[
                ft.Text(
                    t("about.special_thanks"),
                    size=24,
                    weight=ft.FontWeight.BOLD,
                    color=COLOR_NEUTRAL,
                ),
                ft.Container(height=16),
                ft.Container(
                    content=ft.Column(
                        controls=thanks_items,
                        spacing=8,
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    width=float("inf"),
                ),
            ],
            spacing=8,
        )

        return self._wrap_card(card_content, height=None)

    def _build_licenses_card(self) -> ft.Control:
        """Build Open Source Licenses section."""
        licenses_text = _load_third_party_notices()

        card_content = ft.Column(
            controls=[
                ft.Text(
                    t("about.licenses"),
                    size=24,
                    weight=ft.FontWeight.BOLD,
                    color=COLOR_NEUTRAL,
                ),
                ft.Container(height=16),
                ft.Container(
                    content=ft.Text(
                        licenses_text,
                        size=16,
                        color=COLOR_ON_BACKGROUND,
                        selectable=True,
                    ),
                    width=float("inf"),
                    border=ft.Border.all(1, COLOR_DIVIDER),
                    border_radius=12,
                    padding=16,
                    bgcolor=COLOR_SURFACE,
                ),
            ],
        )

        return self._wrap_card(card_content, height=None)

    def _wrap_card(
        self,
        content: ft.Control,
        *,
        height: float | int | None = SharedCardWrapper.DEFAULT_HEIGHT,
    ) -> SharedCardWrapper:
        return SharedCardWrapper(
            content,
            height=height,
        )

    def _on_name_hover(self, e):
        """Handle hover on name link."""
        text = e.control.content
        text.color = COLOR_PRIMARY if is_hover_active(e) else COLOR_ON_BACKGROUND
        text.update()

    def _on_link_hover(self, e):
        """Handle hover on project links."""
        text = e.control.content
        text.color = COLOR_PRIMARY if is_hover_active(e) else COLOR_ON_BACKGROUND
        text.update()

    def _on_version_hover(self, e):
        """Handle hover on version link."""
        text = e.control.content
        text.color = COLOR_PRIMARY if is_hover_active(e) else COLOR_ON_BACKGROUND
        text.update()

    def _on_thanks_hover(self, e):
        """Handle hover on thanks text."""
        text = e.control.content
        text.color = COLOR_PRIMARY if is_hover_active(e) else COLOR_ON_BACKGROUND
        text.update()

    def apply_locale(self) -> None:
        """Refresh UI text when locale changes."""
        self._build_ui()
        self.update()
