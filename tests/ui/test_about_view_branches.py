from __future__ import annotations

import types

import pytest

pytest.importorskip("flet")

from puripuly_heart.ui import i18n as i18n_module
from puripuly_heart.ui.views import about as about_module
from puripuly_heart.ui.views.about import AboutView


def _collect_click_handlers(control) -> list:
    handlers = []
    on_click = getattr(control, "on_click", None)
    if callable(on_click):
        handlers.append(on_click)
    content = getattr(control, "content", None)
    if content is not None:
        handlers.extend(_collect_click_handlers(content))
    controls = getattr(control, "controls", None)
    if controls:
        for child in controls:
            handlers.extend(_collect_click_handlers(child))
    return handlers


def _collect_text_values(control) -> list[str]:
    values = []
    value = getattr(control, "value", None)
    if isinstance(value, str):
        values.append(value)
    content = getattr(control, "content", None)
    if content is not None:
        values.extend(_collect_text_values(content))
    controls = getattr(control, "controls", None)
    if controls:
        for child in controls:
            values.extend(_collect_text_values(child))
    return values


def _row_cards(container) -> list:
    return [child.content for child in container.content.controls]


def test_about_view_uses_shared_card_wrapper_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    from puripuly_heart.ui.components.shared_card_wrapper import SharedCardWrapper

    monkeypatch.setattr(about_module, "_get_profile_image_path", lambda: "")
    monkeypatch.setattr(about_module, "_load_third_party_notices", lambda: "licenses")

    view = AboutView()

    default_cards = [*_row_cards(view.controls[0]), *_row_cards(view.controls[1])]
    full_width_cards = [view.controls[2], view.controls[3]]

    assert len(default_cards) == 4
    assert all(isinstance(card, SharedCardWrapper) for card in default_cards)
    assert {card.height for card in default_cards} == {300}
    assert all(card.expand is True for card in default_cards)
    assert all(isinstance(card, SharedCardWrapper) for card in full_width_cards)
    assert all(card.height is None for card in full_width_cards)
    assert all(card.expand is False for card in full_width_cards)
    assert all(card.content.expand is False for card in full_width_cards)


def test_about_view_link_actions_handle_missing_page_gracefully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[str] = []
    monkeypatch.setattr(about_module, "_get_profile_image_path", lambda: "")
    monkeypatch.setattr(about_module, "_load_third_party_notices", lambda: "licenses")
    monkeypatch.setattr(about_module.webbrowser, "open", lambda url: opened.append(url))

    view = AboutView()
    click_handlers = []
    for control in view.controls:
        click_handlers.extend(_collect_click_handlers(control))

    for handler in click_handlers:
        handler(None)

    assert "https://github.com/kapitalismho/PuriPuly-heart" in opened
    assert "https://x.com/kapitalismho" in opened
    assert "https://github.com/misyaguziya/VRCT" in opened
    assert "https://github.com/naeruru/mimiuchi" in opened
    assert "https://github.com/febilly/Yakutan" in opened


def test_about_view_special_thanks_names_render_in_requested_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(about_module, "_get_profile_image_path", lambda: "")
    monkeypatch.setattr(about_module, "_load_third_party_notices", lambda: "licenses")

    view = AboutView()

    expected_names = [
        "SUI_32C",
        "Nagikokoro",
        "motoka96",
        "_Ykol魚",
        "kascr_",
        "Just Monika V",
        "FLUVIA",
        "Han โชเล่ย์",
        "EA_PE",
        "Ephedrine",
        "~ eri ~",
        "梅雨Shiro",
    ]
    rendered_names = [
        value for value in _collect_text_values(view.controls[2]) if value in expected_names
    ]
    assert rendered_names == expected_names


def test_about_view_special_thanks_names_render_from_i18n(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(about_module, "_get_profile_image_path", lambda: "")
    monkeypatch.setattr(about_module, "_load_third_party_notices", lambda: "licenses")

    localized_names = {
        "about.special_thanks.name.sui_32c": "localized:SUI_32C",
        "about.special_thanks.name.nagikokoro": "localized:Nagikokoro",
        "about.special_thanks.name.motoka96": "localized:motoka96",
        "about.special_thanks.name.ykol": "localized:_Ykol魚",
        "about.special_thanks.name.kascr": "localized:kascr_",
        "about.special_thanks.name.just_monika_v": "localized:Just Monika V",
        "about.special_thanks.name.fluvia": "localized:FLUVIA",
        "about.special_thanks.name.han_chole": "localized:Han โชเล่ย์",
        "about.special_thanks.name.ea_pe": "localized:EA_PE",
        "about.special_thanks.name.ephedrine": "localized:Ephedrine",
        "about.special_thanks.name.eri": "localized:~ eri ~",
        "about.special_thanks.name.tsuyu_shiro": "localized:梅雨Shiro",
    }

    monkeypatch.setattr(
        about_module,
        "t",
        lambda key, **_params: localized_names.get(key, key),
    )

    view = AboutView()

    rendered_names = [
        value for value in _collect_text_values(view.controls[2]) if value.startswith("localized:")
    ]
    assert rendered_names == list(localized_names.values())


@pytest.mark.parametrize("locale", ["en", "ko", "zh-CN", "ja"])
def test_about_view_special_thanks_name_keys_exist_in_locale_bundles(locale: str) -> None:
    expected_names = {
        "about.special_thanks.name.sui_32c": "SUI_32C",
        "about.special_thanks.name.nagikokoro": "Nagikokoro",
        "about.special_thanks.name.motoka96": "motoka96",
        "about.special_thanks.name.ykol": "_Ykol魚",
        "about.special_thanks.name.kascr": "kascr_",
        "about.special_thanks.name.just_monika_v": "Just Monika V",
        "about.special_thanks.name.fluvia": "FLUVIA",
        "about.special_thanks.name.han_chole": "Han โชเล่ย์",
        "about.special_thanks.name.ea_pe": "EA_PE",
        "about.special_thanks.name.ephedrine": "Ephedrine",
        "about.special_thanks.name.eri": "~ eri ~",
        "about.special_thanks.name.fzcfweasdferttgg_png": "fzcfweasdferttgg-png",
        "about.special_thanks.name.tsuyu_shiro": "梅雨Shiro",
    }

    bundle = i18n_module._load_bundle(locale)

    for key, expected_name in expected_names.items():
        assert bundle.get(key) == expected_name


def test_about_view_hover_handlers_and_locale_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(about_module, "_get_profile_image_path", lambda: "")
    monkeypatch.setattr(about_module, "_load_third_party_notices", lambda: "licenses")
    monkeypatch.setattr(AboutView, "update", lambda self: setattr(self, "_updated", True))
    view = AboutView()

    text = types.SimpleNamespace(color=about_module.COLOR_ON_BACKGROUND, update=lambda: None)
    evt = types.SimpleNamespace(control=types.SimpleNamespace(content=text), data="true")
    view._on_name_hover(evt)
    assert text.color == about_module.COLOR_PRIMARY
    evt.data = "false"
    view._on_link_hover(evt)
    assert text.color == about_module.COLOR_ON_BACKGROUND
    evt.data = "true"
    view._on_version_hover(evt)
    assert text.color == about_module.COLOR_PRIMARY
    evt.data = "false"
    view._on_thanks_hover(evt)
    assert text.color == about_module.COLOR_ON_BACKGROUND

    view.apply_locale()
    assert getattr(view, "_updated", False) is True


def test_about_helper_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenFiles:
        def joinpath(self, _name):
            raise RuntimeError("missing")

    monkeypatch.setattr(about_module.resources, "files", lambda _name: BrokenFiles())

    assert about_module._load_third_party_notices() == "Could not load license information."
    assert about_module._get_profile_image_path() == ""
