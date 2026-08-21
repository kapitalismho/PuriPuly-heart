from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from puripuly_heart.core.language import get_language_info
from puripuly_heart.ui.foundation.resources import DEFAULT_FOUNDATION_RESOURCES
from puripuly_heart.ui.foundation.tokens import FOUNDATION_DESIGN_TOKENS

if TYPE_CHECKING:
    import flet as ft


FONT_FAMILY_NANUM = "NanumSquareRound"
FONT_FAMILY_MPLUS = "MPLUSRounded1c"
FONT_FAMILY_RESOURCE_HAN_CN = "ResourceHanRoundedCN"
FONT_FAMILY_NOTO_SANS = "Noto Sans"
FONT_FAMILY_NOTO_SANS_CJK_KR = "Noto Sans CJK KR"
FONT_FAMILY_NOTO_SANS_CJK_JP = "Noto Sans CJK JP"
FONT_FAMILY_NOTO_SANS_CJK_SC = "Noto Sans CJK SC"
FONT_FAMILY_NOTO_SANS_CJK_TC = "Noto Sans CJK TC"
_NOTO_CJK_FILENAME = "NotoSansCJK-Medium.ttc"

DEFAULT_FONT_FAMILY = FOUNDATION_DESIGN_TOKENS.default_font_family

_FONT_FILE_CANDIDATES: dict[str, tuple[str, ...]] = {
    FONT_FAMILY_NANUM: (
        "NanumSquareRoundEB.ttf",
        "NanumSquareRoundB.ttf",
        "NanumSquareRound-Bold.ttf",
    ),
    FONT_FAMILY_MPLUS: (
        "MPLUSRounded1c-Bold.ttf",
        "MPLUSRounded1c-Bold.otf",
    ),
    FONT_FAMILY_RESOURCE_HAN_CN: ("ResourceHanRoundedCN-Bold.ttf",),
    FONT_FAMILY_NOTO_SANS: (_NOTO_CJK_FILENAME,),
    FONT_FAMILY_NOTO_SANS_CJK_KR: (_NOTO_CJK_FILENAME,),
    FONT_FAMILY_NOTO_SANS_CJK_JP: (_NOTO_CJK_FILENAME,),
    FONT_FAMILY_NOTO_SANS_CJK_SC: (_NOTO_CJK_FILENAME,),
    FONT_FAMILY_NOTO_SANS_CJK_TC: (_NOTO_CJK_FILENAME,),
}


def assets_dir() -> Path:
    return DEFAULT_FOUNDATION_RESOURCES.assets_root


def fonts_dir() -> Path:
    return assets_dir() / "fonts"


def register_fonts(page: "ft.Page") -> None:
    fonts: dict[str, str] = {}
    for family in _FONT_FILE_CANDIDATES:
        asset_path = font_asset_path(family)
        if asset_path:
            fonts[family] = asset_path
    if fonts:
        page.fonts = fonts


def font_asset_path(font_family: str) -> str | None:
    filename = _resolve_font_file(font_family)
    if not filename:
        return None
    return f"/{DEFAULT_FOUNDATION_RESOURCES.asset_url(f'fonts/{filename}')}"


def default_font_family() -> str | None:
    if _font_available(DEFAULT_FONT_FAMILY):
        return DEFAULT_FONT_FAMILY
    return None


def noto_cjk_family_for_ui_locale(locale: str | None) -> str:
    if not locale:
        return FONT_FAMILY_NOTO_SANS_CJK_JP
    normalized = locale.strip().replace("_", "-")
    if not normalized:
        return FONT_FAMILY_NOTO_SANS_CJK_JP
    lowered = normalized.lower()
    if lowered in {"zh-cn", "zh-hans"} or lowered.startswith("zh-hans-"):
        return FONT_FAMILY_NOTO_SANS_CJK_SC
    if lowered in {"zh-tw", "zh-hk", "zh-hant", "zh-mo"} or lowered.startswith("zh-hant-"):
        return FONT_FAMILY_NOTO_SANS_CJK_TC
    base = lowered.split("-", 1)[0]
    if base == "ko":
        return FONT_FAMILY_NOTO_SANS_CJK_KR
    if base == "ja":
        return FONT_FAMILY_NOTO_SANS_CJK_JP
    if base == "zh":
        return FONT_FAMILY_NOTO_SANS_CJK_SC
    return FONT_FAMILY_NOTO_SANS_CJK_JP


def font_for_language(code: str | None) -> str | None:
    if not code:
        return default_font_family()

    info = get_language_info(code)
    lang_code = info.code if info else code

    if lang_code == "zh-CN":
        return _resolve_family(FONT_FAMILY_RESOURCE_HAN_CN)

    base_code = lang_code.split("-")[0].lower()
    if base_code == "ja":
        return _resolve_family(FONT_FAMILY_MPLUS)
    if base_code in ("ko", "en"):
        return _resolve_family(FONT_FAMILY_NANUM)

    return None


def _resolve_family(font_family: str) -> str | None:
    if _font_available(font_family):
        return font_family
    return default_font_family()


def _font_available(font_family: str) -> bool:
    return _resolve_font_file(font_family) is not None


def _resolve_font_file(font_family: str) -> str | None:
    fonts_root = fonts_dir()
    for filename in _FONT_FILE_CANDIDATES.get(font_family, ()):
        if (fonts_root / filename).is_file():
            return filename
    return None
