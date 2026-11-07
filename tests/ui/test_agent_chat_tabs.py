from __future__ import annotations

from puripuly_heart.ui.app import TAB_KEYS, TranslatorApp
from puripuly_heart.ui.components.bottom_nav import BottomNavBar
from puripuly_heart.ui.i18n import get_locale, set_locale


def test_agent_chat_is_always_the_second_tab_and_settings_the_third():
    app = TranslatorApp.__new__(TranslatorApp)
    assert TAB_KEYS == ("home", "agent", "settings", "about")
    assert [app._tab_key(index) for index in range(4)] == list(TAB_KEYS)
    assert app._tab_index("settings") == 2
    assert app._content_padding_for_index(2) == 0
    assert app._content_padding_for_index(1) != 0
    assert len(BottomNavBar(on_change=lambda index: None)._tabs) == len(TAB_KEYS)


class Page:
    def __init__(self) -> None:
        self.dialogs = []
        self.popped = 0

    def show_dialog(self, dialog) -> None:
        self.dialogs.append(dialog)

    def pop_dialog(self) -> None:
        self.popped += 1


class AgentView:
    def __init__(self) -> None:
        self.asked = []

    def ask_about_error(self, text, diagnostic_id) -> None:
        self.asked.append((text, diagnostic_id))


def agent_app(view):
    app = TranslatorApp.__new__(TranslatorApp)
    app.page = Page()
    app.view_agent = view
    app.navigated = []
    app._on_nav_change = app.navigated.append
    app._set_bottom_nav_selected = lambda index: None
    return app


def test_error_snackbar_asks_puri_with_the_diagnostic_id_on_the_agent_tab():
    previous = get_locale()
    set_locale("ko")
    try:
        view = AgentView()
        app = agent_app(view)
        app.on_error_presented("Deepgram 인증에 실패했어요", "diag-1")
        app.on_error_presented("Deepgram 인증에 실패했어요", "diag-2")
        (snackbar,) = app.page.dialogs
        text, button = snackbar.content.controls
        assert text.value == "Deepgram 인증에 실패했어요"
        assert button.content == "Puri에게 물어보기"
        button.on_click(None)
        assert app.page.popped == 1
        assert app.navigated == [app._tab_index("agent")]
        assert view.asked == [("Deepgram 인증에 실패했어요", "diag-1")]
    finally:
        set_locale(previous)


def test_error_snackbar_has_no_puri_action_without_agent_chat():
    app = agent_app(None)
    app.on_error_presented("Deepgram 인증에 실패했어요", "diag-1")
    assert app.page.dialogs == []
