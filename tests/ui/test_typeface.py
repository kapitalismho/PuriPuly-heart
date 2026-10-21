from __future__ import annotations

import pytest

pytest.importorskip("flet")

from puripuly_heart.ui.fonts import (
    LOG_TYPEFACE,
    TypeFace,
    font_for_language,
    metrics_strut_typeface,
    typeface_for_language,
)
from puripuly_heart.ui.i18n import get_locale, set_locale
from puripuly_heart.ui.theme import get_app_theme, text_style


def test_typeface_pairs_each_locale_font_with_its_tracking() -> None:
    japanese = typeface_for_language("ja")
    chinese = typeface_for_language("zh-CN")
    korean = typeface_for_language("ko")
    english = typeface_for_language("en")
    russian = typeface_for_language("ru")

    assert japanese == TypeFace(font_for_language("ja"), 0.5)
    assert chinese == TypeFace(font_for_language("zh-CN"), 0.4)
    assert korean == TypeFace(font_for_language("ko"), None)
    assert english == TypeFace(font_for_language("en"), None)
    assert russian == TypeFace(None, None)
    assert metrics_strut_typeface() == TypeFace(font_for_language("ko"), 0)
    assert LOG_TYPEFACE == TypeFace("Consolas", 0)


def test_theme_replaces_body_medium_only_when_tracking_is_part_of_the_face() -> None:
    japanese = typeface_for_language("ja")
    theme = get_app_theme(face=japanese)
    assert theme.font_family == japanese.font_family
    assert theme.text_theme is not None
    body = theme.text_theme.body_medium
    assert body is not None
    assert body.font_family == japanese.font_family
    assert body.letter_spacing == 0.5

    korean = typeface_for_language("ko")
    theme = get_app_theme(face=korean)
    assert theme.font_family == korean.font_family
    assert theme.text_theme is None

    russian = typeface_for_language("ru")
    theme = get_app_theme(face=russian)
    assert theme.font_family is None
    assert theme.text_theme is None


def test_get_app_theme_keeps_the_legacy_font_arguments() -> None:
    theme = get_app_theme("Arial")
    assert theme.font_family == "Arial"
    assert theme.text_theme is None

    theme = get_app_theme(font_family="Pretendard", body_letter_spacing=0.4)
    assert theme.font_family == "Pretendard"
    assert theme.text_theme is not None
    body = theme.text_theme.body_medium
    assert body is not None
    assert body.font_family == "Pretendard"
    assert body.letter_spacing == 0.4


def test_text_style_uses_the_current_ui_typeface() -> None:
    previous = get_locale()
    try:
        set_locale("ja")
        style = text_style(size=18)
        assert style.font_family == typeface_for_language("ja").font_family
        assert style.letter_spacing == 0.5
        assert style.size == 18

        set_locale("ko")
        style = text_style(size=18)
        assert style.font_family == typeface_for_language("ko").font_family
        assert style.letter_spacing is None
    finally:
        set_locale(previous)

    style = text_style(LOG_TYPEFACE)
    assert style.font_family == "Consolas"
    assert style.letter_spacing == 0
