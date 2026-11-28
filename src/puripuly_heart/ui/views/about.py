"""About page view with version, credits, acknowledgments, and license info."""

import webbrowser
from dataclasses import dataclass

import flet as ft

from puripuly_heart import __version__
from puripuly_heart.runtime_layout import current_runtime_layout
from puripuly_heart.ui.about.contract import AboutSurfaceSlots
from puripuly_heart.ui.about.renderer import compose_about_surface
from puripuly_heart.ui.components.shared_card_wrapper import SharedCardWrapper
from puripuly_heart.ui.flet_runtime import is_hover_active
from puripuly_heart.ui.i18n import t
from puripuly_heart.ui.theme import (
    COLOR_DIVIDER,
    COLOR_ON_BACKGROUND,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_SURFACE,
)

_CENTER_ALIGNMENT = ft.Alignment(0, 0)
_SPECIAL_THANKS_AVATAR_SIZE = 96


def _load_third_party_notices() -> str:
    """Load THIRD_PARTY_NOTICES.txt from package data."""
    try:
        return (
            current_runtime_layout()
            .package_resource("data", "THIRD_PARTY_NOTICES.txt")
            .read_text(encoding="utf-8")
        )
    except Exception:
        return "Could not load license information."


def _get_profile_image_path() -> str:
    """Get the profile image path from package data."""
    try:
        return str(current_runtime_layout().package_resource("data", "pictures", "salee_pic.png"))
    except Exception:
        return ""


def _get_special_thanks_image_path(image_file: str) -> str:
    """Get a Special Thanks avatar path from package data, or "" when it is not bundled."""
    try:
        path = current_runtime_layout().package_resource(
            "data", "pictures", "special_thanks", image_file
        )
    except Exception:
        return ""
    return str(path) if path.is_file() else ""


def _special_thanks_initial(display_name: str) -> str:
    """Return the first letter shown when a Special Thanks avatar is not bundled."""
    return next((char for char in display_name if char.isalnum()), "?").upper()


@dataclass(frozen=True, slots=True)
class SpecialThanksEntry:
    name_key: str
    image_file: str
    url: str | None = None


SPECIAL_THANKS_COLUMNS = 3

# Special thanks entries - add new people here, update locale bundles,
# and place the avatar in data/pictures/special_thanks/
SPECIAL_THANKS_ENTRIES: tuple[SpecialThanksEntry, ...] = (
    SpecialThanksEntry(
        "about.special_thanks.name.sui_32c",
        "sui_32c.png",
        "https://x.com/SUI32C",
    ),
    SpecialThanksEntry(
        "about.special_thanks.name.nagikokoro",
        "nagikokoro.png",
        "https://www.nagikokoro.best/",
    ),
    SpecialThanksEntry("about.special_thanks.name.motoka96", "motoka96.png"),
    SpecialThanksEntry("about.special_thanks.name.ykol", "ykol.png"),
    SpecialThanksEntry(
        "about.special_thanks.name.kascr",
        "kascr.png",
        "https://github.com/kascr",
    ),
    SpecialThanksEntry("about.special_thanks.name.just_monika_v", "just_monika_v.png"),
    SpecialThanksEntry(
        "about.special_thanks.name.fluvia",
        "fluvia.png",
        "https://vrchat.com/home/user/usr_250dc79b-edbe-4d1d-be71-c7f2cd207ea6",
    ),
    SpecialThanksEntry("about.special_thanks.name.han_chole", "han_chole.png"),
    SpecialThanksEntry(
        "about.special_thanks.name.ea_pe",
        "ea_pe.png",
        "https://discord.com/users/859837257917136918",
    ),
    SpecialThanksEntry(
        "about.special_thanks.name.ephedrine",
        "ephedrine.png",
        "https://x.com/NUN1EL",
    ),
    SpecialThanksEntry(
        "about.special_thanks.name.eri",
        "eri.png",
        "https://x.com/39eri_vr",
    ),
    SpecialThanksEntry(
        "about.special_thanks.name.fzcfweasdferttgg_png",
        "fzcfweasdferttgg_png.png",
        "https://github.com/fzcfweasdferttgg-png",
    ),
    SpecialThanksEntry(
        "about.special_thanks.name.welcius",
        "welcius.png",
        "https://vrchat.com/home/user/usr_7d77c22e-bc62-472f-acf3-33a3ab71ec57",
    ),
    SpecialThanksEntry(
        "about.special_thanks.name.nunu299",
        "nunu299.png",
        "https://x.com/nunu2991",
    ),
    SpecialThanksEntry(
        "about.special_thanks.name.tsuyu_shiro",
        "tsuyu_shiro.png",
        "https://www.instagram.com/moena1213?fqnt=MWNnMnJzemRsZ3o2MQ%3D%3D",
    ),
    SpecialThanksEntry(
        "about.special_thanks.name.xxxxxshiro",
        "xxxxxshiro.png",
        "https://vrchat.com/home/user/usr_39788233-af01-47a4-92da-c5ee39b20e06",
    ),
    SpecialThanksEntry(
        "about.special_thanks.name.koz39",
        "koz39.png",
        "https://github.com/KOZ39",
    ),
)


class AboutView(ft.Column):
    """About page with version, credits, inspired by, special thanks, and licenses."""

    def __init__(self):
        super().__init__(expand=True, scroll=ft.ScrollMode.AUTO, spacing=16)

        self._build_ui()

    def _build_ui(self):
        """Build the About page UI."""
        regions = compose_about_surface(
            AboutSurfaceSlots(
                app_name_card=self._build_app_name_card(),
                version_card=self._build_version_card(),
                credits_card=self._build_credits_card(),
                inspired_by_card=self._build_inspired_by_card(),
                special_thanks_card=self._build_special_thanks_card(),
                licenses_card=self._build_licenses_card(),
            )
        )
        self.controls = list(regions.controls)

    def _build_app_name_card(self) -> ft.Control:
        """Build the app name card."""
        return self._wrap_card(
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

    def _build_version_card(self) -> ft.Control:
        """Build the clickable version card that opens the git repository."""
        version_title = ft.Text(
            t("about.version"),
            size=24,
            weight=ft.FontWeight.BOLD,
            color=COLOR_SECONDARY,
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
        return self._wrap_card(ft.Column([version_title, version_text], spacing=0, expand=True))

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
                    color=COLOR_SECONDARY,
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
                    color=COLOR_SECONDARY,
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

    def _build_special_thanks_card(self) -> ft.Control:
        """Build Special Thanks section as a grid of people inside one card."""
        people = [self._build_special_thanks_person(entry) for entry in SPECIAL_THANKS_ENTRIES]

        grid_rows = []
        for start in range(0, len(people), SPECIAL_THANKS_COLUMNS):
            row_people = people[start : start + SPECIAL_THANKS_COLUMNS]
            row_people += [ft.Container()] * (SPECIAL_THANKS_COLUMNS - len(row_people))
            grid_rows.append(
                ft.Row(
                    controls=[ft.Container(content=person, expand=True) for person in row_people],
                    spacing=24,
                )
            )

        card_content = ft.Column(
            controls=[
                ft.Text(
                    t("about.special_thanks"),
                    size=24,
                    weight=ft.FontWeight.BOLD,
                    color=COLOR_SECONDARY,
                ),
                ft.Container(height=16),
                ft.Column(controls=grid_rows, spacing=28),
                ft.Container(
                    content=ft.Text("and you!", size=28, color=COLOR_ON_BACKGROUND, italic=True),
                    alignment=_CENTER_ALIGNMENT,
                    padding=ft.Padding.only(top=24),
                ),
            ],
        )

        return self._wrap_card(card_content, height=None)

    def _build_special_thanks_person(self, entry: SpecialThanksEntry) -> ft.Control:
        """Build one Special Thanks person with a round avatar and a name."""
        display_name = t(entry.name_key)
        image_path = _get_special_thanks_image_path(entry.image_file)
        avatar = ft.Container(
            content=(
                ft.Image(
                    src=image_path,
                    width=_SPECIAL_THANKS_AVATAR_SIZE,
                    height=_SPECIAL_THANKS_AVATAR_SIZE,
                    fit=ft.BoxFit.COVER,
                    border_radius=_SPECIAL_THANKS_AVATAR_SIZE,
                )
                if image_path
                else ft.Text(
                    _special_thanks_initial(display_name),
                    size=40,
                    weight=ft.FontWeight.BOLD,
                    color=COLOR_SECONDARY,
                )
            ),
            width=_SPECIAL_THANKS_AVATAR_SIZE,
            height=_SPECIAL_THANKS_AVATAR_SIZE,
            border_radius=_SPECIAL_THANKS_AVATAR_SIZE,
            bgcolor=ft.Colors.WHITE,
            alignment=_CENTER_ALIGNMENT,
            clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
        )
        name = ft.Text(
            display_name,
            size=28,
            weight=ft.FontWeight.BOLD,
            color=COLOR_ON_BACKGROUND,
            max_lines=2,
            overflow=ft.TextOverflow.ELLIPSIS,
        )
        return ft.Container(
            content=ft.Row(
                controls=[avatar, ft.Container(content=name, expand=True)],
                spacing=16,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            on_click=(lambda _, u=entry.url: webbrowser.open(u)) if entry.url else None,
            on_hover=self._on_thanks_hover,
            data=name,
        )

    def _build_licenses_card(self) -> ft.Control:
        """Build Open Source Licenses section."""
        licenses_text = _load_third_party_notices()

        card_content = ft.Column(
            controls=[
                ft.Text(
                    t("about.licenses"),
                    size=24,
                    weight=ft.FontWeight.BOLD,
                    color=COLOR_SECONDARY,
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
        """Handle hover on a Special Thanks person."""
        text = e.control.data
        text.color = COLOR_PRIMARY if is_hover_active(e) else COLOR_ON_BACKGROUND
        text.update()

    def apply_locale(self) -> None:
        """Refresh UI text when locale changes."""
        self._build_ui()
        self.update()
