from __future__ import annotations

import asyncio
import logging

import pytest

ft = pytest.importorskip("flet")

from puripuly_heart.app.ports.ui_models import ManagedGemmaDashboardNotice
from puripuly_heart.ui.dashboard import capture as dashboard_capture_module
from puripuly_heart.ui.dashboard.contract import (
    DashboardCaptureIntents,
    DashboardTranslationIntents,
)
from puripuly_heart.ui.gpu_notice import GpuDashboardNotice
from puripuly_heart.ui.overlay_peer_contract import (
    OverlayPeerConsumerContract,
    OverlayPeerToggleContract,
)
from puripuly_heart.ui.views import dashboard as dashboard_module
from tests.helpers.flet_page import attach_dummy_page


class FakePowerButton:
    def __init__(self, label, icon, on_click, **kwargs):
        self.icon = icon
        self.kwargs = dict(kwargs)
        self.label = label
        self.on_click = on_click
        self.states: list[dict[str, object]] = []

    def set_state(
        self,
        is_on: bool,
        needs_key: bool = False,
        *,
        is_starting: bool = False,
        status_text: str | None = None,
        helper_text: str | None = None,
    ):
        state = {
            "is_on": is_on,
            "needs_key": needs_key,
            "status_text": status_text,
            "helper_text": helper_text,
        }
        if is_starting:
            state["is_starting"] = True
        self.states.append(state)

    def set_label(self, label: str) -> None:
        self.label = label


class FakeDisplayCard:
    def __init__(self, on_submit, on_input_focus_change=None, on_input_activity=None):
        self._on_submit = on_submit
        self._on_input_focus_change = on_input_focus_change
        self._on_input_activity = on_input_activity
        self.statuses: list[tuple[str, str | None]] = []
        self.display_calls: list[tuple[str, bool, str | None]] = []
        self.translation_calls: list[tuple[str | None, str | None]] = []
        self.translation_metadata_calls: list[dict[str, object]] = []
        self.notice_calls: list[tuple[str | None, str | None]] = []
        self.notice_actions: list[tuple[str | None, object | None]] = []
        self.notice_yields: list[bool] = []
        self.input_fonts: list[str | None] = []
        self.locale_calls: list[tuple[str | None, str | None]] = []
        self.input_is_focused = False
        self.focus_calls = 0

    def set_status(self, status: str, font_family: str | None = None) -> None:
        self.statuses.append((status, font_family))

    def set_display(
        self,
        text: str,
        *,
        is_error: bool = False,
        font_family: str | None = None,
        **_metadata,
    ) -> None:
        self.display_calls.append((text, is_error, font_family))

    def set_display_translation(
        self,
        text: str | None,
        font_family: str | None = None,
        **metadata,
    ) -> None:
        self.translation_calls.append((text, font_family))
        self.translation_metadata_calls.append(dict(metadata))

    def set_notice(
        self,
        text: str | None,
        tone: str | None = None,
        *,
        action_label: str | None = None,
        on_action=None,
        yields_to_content: bool = False,
    ) -> None:
        self.notice_calls.append((text, tone))
        self.notice_actions.append((action_label, on_action))
        self.notice_yields.append(bool(yields_to_content))

    def set_input_font(self, font_family: str | None) -> None:
        self.input_fonts.append(font_family)

    def apply_locale(self, display_font_family: str | None, input_font_family: str | None) -> None:
        self.locale_calls.append((display_font_family, input_font_family))

    def set_input_focus_for_test(self, focused: bool) -> None:
        self.input_is_focused = focused
        if self._on_input_focus_change is not None:
            self._on_input_focus_change(focused)

    def set_input_activity_for_test(self, has_text: bool) -> None:
        if self._on_input_activity is not None:
            self._on_input_activity(has_text)

    def focus_input(self) -> None:
        self.focus_calls += 1


class FakeLanguageCard:
    def __init__(
        self,
        on_self_source_click,
        on_self_target_click,
        on_self_swap_click,
        on_peer_source_click,
        on_peer_target_click,
        on_peer_swap_click,
    ):
        self.on_self_source_click = on_self_source_click
        self.on_self_target_click = on_self_target_click
        self.on_self_swap_click = on_self_swap_click
        self.on_peer_source_click = on_peer_source_click
        self.on_peer_target_click = on_peer_target_click
        self.on_peer_swap_click = on_peer_swap_click
        self.languages: list[tuple[str, str, str, str]] = []
        self.row_labels: list[tuple[str, str]] = []

    def set_languages(
        self,
        self_source: str,
        self_target: str,
        peer_source: str,
        peer_target: str,
    ) -> None:
        self.languages.append((self_source, self_target, peer_source, peer_target))

    def set_row_labels(self, self_label: str, peer_label: str) -> None:
        self.row_labels.append((self_label, peer_label))


class FakeLanguageModal:
    opened: list[tuple[str, list[str]]] = []
    languages: list[tuple[str, str]] = []

    def __init__(
        self,
        page,
        languages,
        on_select,
        label_for_code=None,
        description_for_code=None,
        disabled_codes=None,
    ):
        _ = (page, label_for_code, description_for_code, disabled_codes)
        self.__class__.languages = list(languages)
        self.on_select = on_select

    def open(self, *, current: str, recent: list[str]) -> None:
        self.__class__.opened.append((current, list(recent)))


def _make_dashboard(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(dashboard_module, "PowerButton", FakePowerButton)
    monkeypatch.setattr(dashboard_capture_module, "PowerButton", FakePowerButton)
    monkeypatch.setattr(dashboard_module, "DisplayCard", FakeDisplayCard)
    monkeypatch.setattr(dashboard_module, "LanguageCard", FakeLanguageCard)
    monkeypatch.setattr(dashboard_module, "LanguageModal", FakeLanguageModal)
    monkeypatch.setattr(dashboard_module, "font_for_language", lambda code: f"font-{code}")
    monkeypatch.setattr(dashboard_module, "language_name", lambda code: f"name-{code}")
    monkeypatch.setattr(dashboard_module, "get_locale", lambda: "en")
    view = dashboard_module.DashboardView()
    FakeLanguageModal.opened = []
    return view


def _button_labels(row) -> list[str]:
    return [slot.content.label for slot in row.controls]


def _make_overlay_peer_contract(
    *,
    overlay_intent_enabled: bool,
    overlay_state: str,
    overlay_status_text: str,
    overlay_helper_text: str = "",
    peer_intent_enabled: bool,
    peer_effective_enabled: bool,
    peer_status_text: str,
    peer_helper_text: str = "",
    peer_state: str | None = None,
) -> OverlayPeerConsumerContract:
    return OverlayPeerConsumerContract(
        overlay=OverlayPeerToggleContract(
            intent_enabled=overlay_intent_enabled,
            effective_enabled=overlay_state == "connected",
            action_enabled=True,
            state=(
                "on"
                if overlay_state == "connected"
                else ("off" if not overlay_intent_enabled else "warning")
            ),
            status_text=overlay_status_text,
            helper_text=overlay_helper_text,
        ),
        peer=OverlayPeerToggleContract(
            intent_enabled=peer_intent_enabled,
            effective_enabled=peer_effective_enabled,
            action_enabled=True,
            state=(
                peer_state
                or (
                    "on"
                    if peer_effective_enabled
                    else ("off" if not peer_intent_enabled else "warning")
                )
            ),
            status_text=peer_status_text,
            helper_text=peer_helper_text,
        ),
    )


def test_dashboard_initial_peer_language_defaults_to_english_to_korean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view = _make_dashboard(monkeypatch)

    assert view._source_lang_code == "ko"
    assert view._target_lang_code == "en"
    assert view._peer_source_lang_code == "en"
    assert view._peer_target_lang_code == "ko"
    assert view.language_card.languages[-1] == (
        "name-ko",
        "name-en",
        "name-en",
        "name-ko",
    )


def test_dashboard_stt_toggle_warning_and_enable_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    view = _make_dashboard(monkeypatch)
    seen: list[bool] = []
    view.on_toggle_stt = lambda enabled: seen.append(enabled)
    view.stt_needs_key = True
    warn_stt = dashboard_module.t("dashboard.warn_stt_key")

    view._toggle_stt()
    assert view._current_display_text == warn_stt
    assert view._stt_showing_warning is True

    view._toggle_stt()
    assert view._current_display_text is None
    assert view._stt_showing_warning is False
    assert view.display_card.statuses[-1][0] == "disconnected"

    # Blank leftover must still restore the idle status face.
    view.set_display_text("")
    view._stt_showing_warning = True
    view._toggle_stt()
    assert view._current_display_text is None
    assert view.display_card.statuses[-1][0] == "disconnected"

    view.stt_needs_key = False
    view._toggle_stt()
    assert view.stt_button.states[-1]["is_starting"] is True
    view._toggle_stt()

    assert seen == [False, False, False, True, False]
    assert view.is_stt_on is False
    assert view._stt_showing_warning is False
    assert any(call[0] == warn_stt for call in view.display_card.display_calls)


def test_dashboard_translation_toggle_controls_power_state(monkeypatch: pytest.MonkeyPatch) -> None:
    view = _make_dashboard(monkeypatch)
    seen: list[bool] = []
    view.on_toggle_translation = lambda enabled: seen.append(enabled)
    view.translation_needs_key = True
    warn_llm = dashboard_module.t("dashboard.warn_llm_key")

    view._toggle_translation()
    assert view._current_display_text == warn_llm
    assert view._translation_showing_warning is True

    view._toggle_translation()
    assert view._current_display_text is None
    assert view._translation_showing_warning is False
    assert view.display_card.statuses[-1][0] == "disconnected"

    view.set_display_text("")
    view._translation_showing_warning = True
    view._toggle_translation()
    assert view._current_display_text is None
    assert view.display_card.statuses[-1][0] == "disconnected"

    view.translation_needs_key = False
    view._toggle_translation()
    view._toggle_translation()

    assert seen == [False, False, False, True, False]
    assert view.is_power_on is False
    assert any(call[0] == warn_llm for call in view.display_card.display_calls)


def test_dashboard_translation_visual_commit_forwards_metadata_and_runtime_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view = _make_dashboard(monkeypatch)

    def fake_runtime_log_detailed(message: str, *, level: int = logging.INFO) -> bool:
        _ = (message, level)
        return True

    view.runtime_log_detailed = fake_runtime_log_detailed

    view.set_display_translation_text(
        "dst",
        language_code="en",
        update_id="upd-1",
        origin_wall_clock_ms=1712345678901,
        utterance_id="utt-1",
        channel="peer",
        session_scope="session-1",
        source_text_hash="src-hash-1",
        source_text_len=12,
        logical_turn_key="peer:utt-1",
    )

    assert view.display_card.translation_calls[-1] == ("dst", "font-en")
    assert view.display_card.translation_metadata_calls[-1] == {
        "runtime_log_detailed": fake_runtime_log_detailed,
        "update_id": "upd-1",
        "origin_wall_clock_ms": 1712345678901,
        "utterance_id": "utt-1",
        "channel": "peer",
        "session_scope": "session-1",
        "source_text_hash": "src-hash-1",
        "source_text_len": 12,
        "logical_turn_key": "peer:utt-1",
        "debug_prefix": None,
    }


def test_dashboard_submit_and_language_selection_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    view = _make_dashboard(monkeypatch)
    sends: list[tuple[str, str]] = []
    lang_changes: list[tuple[str, str, str, str]] = []
    view.on_send_message = lambda source, text: sends.append((source, text))
    view.on_language_change = lambda change: lang_changes.append(
        (change.source_code, change.target_code, change.peer_source_code, change.peer_target_code)
    )

    view._on_submit("hello")
    view._on_source_select("ja")
    view._on_target_select("fr")
    view._swap_languages()

    assert sends == [("You", "hello")]
    assert view._recent_source_langs == ["ja"]
    assert view._recent_target_langs == ["fr"]
    assert lang_changes[-1] == ("fr", "ja", "en", "ko")
    assert view.language_card.languages[-1] == ("name-fr", "name-ja", "name-en", "name-ko")


def test_dashboard_forwards_non_content_message_input_activity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view = _make_dashboard(monkeypatch)
    activity: list[bool] = []
    view.on_message_input_activity = activity.append

    view.display_card.set_input_activity_for_test(True)
    view.display_card.set_input_activity_for_test(False)

    assert activity == [True, False]


def test_dashboard_tab_in_focused_message_input_swaps_self_languages_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view = _make_dashboard(monkeypatch)
    changes: list[tuple[str, str, str, str]] = []
    view.on_language_change = lambda change: changes.append(
        (change.source_code, change.target_code, change.peer_source_code, change.peer_target_code)
    )
    view.set_languages_from_codes("ko", "en", "ja", "fr")
    view.display_card.set_input_focus_for_test(True)

    handled = view.handle_message_input_tab_key()

    assert handled is True
    assert view._source_lang_code == "en"
    assert view._target_lang_code == "ko"
    assert view._peer_source_lang_code == "ja"
    assert view._peer_target_lang_code == "fr"
    assert changes[-1] == ("en", "ko", "ja", "fr")
    assert view.language_card.languages[-1] == ("name-en", "name-ko", "name-ja", "name-fr")
    assert view.display_card.focus_calls == 1


def test_dashboard_tab_ignored_when_message_input_is_not_focused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view = _make_dashboard(monkeypatch)
    changes: list[tuple[str, str, str, str]] = []
    view.on_language_change = lambda change: changes.append(
        (change.source_code, change.target_code, change.peer_source_code, change.peer_target_code)
    )
    view.set_languages_from_codes("ko", "en", "ja", "fr")
    view.display_card.set_input_focus_for_test(False)

    handled = view.handle_message_input_tab_key()

    assert handled is False
    assert view._source_lang_code == "ko"
    assert view._target_lang_code == "en"
    assert changes == []
    assert view.display_card.focus_calls == 0


def test_dashboard_recent_languages_are_included_in_language_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view = _make_dashboard(monkeypatch)
    changes = []
    view.on_language_change = changes.append

    for idx in range(8):
        view._add_to_recent(f"s{idx}", is_source=True)
        view._add_to_recent(f"t{idx}", is_source=False)

    assert len(view._recent_source_langs) == 6
    assert len(view._recent_target_langs) == 6
    assert view._recent_source_langs[0] == "s7"
    assert view._recent_source_langs[-1] == "s2"
    view._notify_language_change()
    assert changes[-1].recent_source_codes == tuple(view._recent_source_langs)
    assert changes[-1].recent_target_codes == tuple(view._recent_target_langs)


def test_dashboard_public_setters_update_components(monkeypatch: pytest.MonkeyPatch) -> None:
    view = _make_dashboard(monkeypatch)
    view.set_status("connected")
    view.set_languages_from_codes("ko", "en")
    view.set_translation_enabled(False)
    view.set_stt_enabled(False)
    view.set_translation_needs_key(True, update_ui=True)
    view.set_stt_needs_key(True, update_ui=True)
    view.set_local_stt_notice("missing")
    view.set_managed_auth_pending(True)
    view.set_display_text("src", language_code="ko")
    view.set_display_translation_text("dst", language_code="en")
    view.set_recent_languages(["a", "b", "c", "d", "e", "f", "g"], ["x", "y", "z"])

    assert view.is_connected is True
    assert view.display_card.statuses[-1] == ("connected", "font-en")
    assert view.display_card.display_calls[-1] == ("src", False, "font-ko")
    assert view.display_card.translation_calls[-1] == ("dst", "font-en")
    assert view.display_card.notice_calls[-1] == (
        dashboard_module.t("dashboard.local_stt_notice_missing"),
        "warning",
    )
    assert view.language_card.languages[-1] == ("name-ko", "name-en", "name-ko", "name-en")
    assert view.trans_button.states[-1] == {
        "is_on": False,
        "needs_key": True,
        "status_text": None,
        "helper_text": None,
    }
    assert view.stt_button.states[-1] == {
        "is_on": False,
        "needs_key": True,
        "status_text": None,
        "helper_text": None,
    }
    assert view._recent_source_langs == ["a", "b", "c", "d", "e", "f"]


def test_dashboard_managed_auth_pending_restores_local_stt_notice_when_cleared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view = _make_dashboard(monkeypatch)

    view.set_local_stt_notice("missing")
    view.set_managed_auth_pending(True)
    view.set_managed_auth_pending(False)

    assert view.display_card.notice_calls == [
        (dashboard_module.t("dashboard.local_stt_notice_missing"), "warning"),
        (dashboard_module.t("dashboard.local_stt_notice_missing"), "warning"),
        (dashboard_module.t("dashboard.local_stt_notice_missing"), "warning"),
    ]


def test_dashboard_apply_locale_reapplies_managed_auth_pending_notice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view = _make_dashboard(monkeypatch)

    view.set_managed_auth_pending(True)
    view.apply_locale()

    assert view.display_card.notice_calls == [
        (dashboard_module.t("dashboard.managed_auth_pending"), "info"),
        (dashboard_module.t("dashboard.managed_auth_pending"), "info"),
    ]


def test_dashboard_builds_4x3_friendly_shell_without_managed_trial_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view = _make_dashboard(monkeypatch)

    assert view.controls[0] is view.shell_content
    assert view.shell_content.controls == [view.main_surface]
    assert view.shell_content.spacing == dashboard_module.DASHBOARD_LAYOUT_GAP
    assert view.main_surface.controls == [view.control_region, view.info_region]
    assert view.main_surface.spacing == dashboard_module.DASHBOARD_LAYOUT_GAP
    assert dashboard_module.DASHBOARD_CONTROL_REGION_EXPAND == 45
    assert dashboard_module.DASHBOARD_INFO_REGION_EXPAND == 55
    assert view.control_region.expand == 45
    assert view.info_region.expand == 55
    assert view.control_grid.spacing == dashboard_module.DASHBOARD_LAYOUT_GAP
    assert view.top_controls.spacing == dashboard_module.DASHBOARD_LAYOUT_GAP
    assert view.bottom_controls.spacing == dashboard_module.DASHBOARD_LAYOUT_GAP
    assert _button_labels(view.top_controls) == [
        dashboard_module.t("dashboard.stt_label"),
        dashboard_module.t("dashboard.peer_label"),
    ]
    assert _button_labels(view.bottom_controls) == [
        dashboard_module.t("dashboard.trans_label"),
        dashboard_module.t("dashboard.overlay_label"),
    ]
    assert view.stt_button.kwargs["icon_size"] == dashboard_module.DASHBOARD_POWER_BUTTON_ICON_SIZE
    assert view.peer_button.kwargs["icon_size"] == dashboard_module.DASHBOARD_POWER_BUTTON_ICON_SIZE
    assert (
        view.trans_button.kwargs["icon_size"] == dashboard_module.DASHBOARD_POWER_BUTTON_ICON_SIZE
    )
    assert (
        view.overlay_button.kwargs["icon_size"] == dashboard_module.DASHBOARD_POWER_BUTTON_ICON_SIZE
    )
    assert (
        view.stt_button.kwargs["label_size"] == dashboard_module.DASHBOARD_POWER_BUTTON_LABEL_SIZE
    )
    assert (
        view.peer_button.kwargs["label_size"] == dashboard_module.DASHBOARD_POWER_BUTTON_LABEL_SIZE
    )
    assert (
        view.trans_button.kwargs["label_size"] == dashboard_module.DASHBOARD_POWER_BUTTON_LABEL_SIZE
    )
    assert (
        view.overlay_button.kwargs["label_size"]
        == dashboard_module.DASHBOARD_POWER_BUTTON_LABEL_SIZE
    )
    assert view.overlay_button.icon == ft.Icons.SUBTITLES
    assert "color_on" not in view.peer_button.kwargs
    assert "color_on" not in view.trans_button.kwargs
    assert "color_on" not in view.overlay_button.kwargs
    assert view.info_stack.spacing == dashboard_module.DASHBOARD_LAYOUT_GAP
    assert view.info_stack.controls == [view.display_card_slot, view.language_card_slot]
    assert view.display_card_slot.content is view.display_card
    assert dashboard_module.DASHBOARD_DISPLAY_CARD_EXPAND == 1
    assert view.display_card_slot.expand == 1
    assert view.language_card_slot.content is view.language_card
    assert dashboard_module.DASHBOARD_LANGUAGE_CARD_EXPAND == 1
    assert view.language_card_slot.expand == 1
    assert not hasattr(view, "_managed_trial_card")


def test_dashboard_apply_locale_and_dialog_open_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    view = _make_dashboard(monkeypatch)
    attach_dummy_page(monkeypatch, view)
    view._stt_showing_warning = True
    view._open_source_dialog()
    view._open_target_dialog()
    view.apply_locale()
    view._translation_showing_warning = True
    view._stt_showing_warning = False
    view.apply_locale()

    assert FakeLanguageModal.opened[0][0] == "ko"
    assert FakeLanguageModal.opened[1][0] == "en"
    assert view.stt_button.label == dashboard_module.t("dashboard.stt_label")
    assert view.peer_button.label == dashboard_module.t("dashboard.peer_label")
    assert view.trans_button.label == dashboard_module.t("dashboard.trans_label")
    assert view.overlay_button.label == dashboard_module.t("dashboard.overlay_label")
    warning_texts = [text for text, _is_error, _font in view.display_card.display_calls]
    assert dashboard_module.t("dashboard.warn_stt_key") in warning_texts
    assert dashboard_module.t("dashboard.warn_llm_key") in warning_texts


def test_dashboard_peer_source_dialog_lists_automatic_detection_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view = _make_dashboard(monkeypatch)
    attach_dummy_page(monkeypatch, view)

    view._open_peer_source_dialog()

    assert FakeLanguageModal.languages[0][0] == dashboard_module.PEER_SOURCE_MODE_AUTO


def test_dashboard_automatic_peer_selection_preserves_manual_source_and_emits_auto_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view = _make_dashboard(monkeypatch)
    changes: list[tuple[str, str, str, str, str]] = []
    view.on_language_change = lambda change: changes.append(
        (
            change.source_code,
            change.target_code,
            change.peer_source_code,
            change.peer_target_code,
            change.peer_source_mode,
        )
    )
    view.set_languages_from_codes("ko", "en", "ja", "fr")

    view._on_peer_source_select(dashboard_module.PEER_SOURCE_MODE_AUTO)

    assert view._peer_source_lang_code == "ja"
    assert changes[-1] == ("ko", "en", "ja", "fr", "auto")


def test_dashboard_automatic_peer_callback_type_error_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view = _make_dashboard(monkeypatch)
    calls: list[tuple[str, ...]] = []

    def fail_after_recording(change) -> None:
        calls.append((change,))
        raise TypeError("callback failure")

    view.on_language_change = fail_after_recording

    with pytest.raises(TypeError, match="callback failure"):
        view._on_peer_source_select(dashboard_module.PEER_SOURCE_MODE_AUTO)

    assert len(calls) == 1
    assert calls[0][0].peer_source_mode == "auto"


def test_dashboard_automatic_peer_selection_emits_typed_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view = _make_dashboard(monkeypatch)
    calls = []
    view.on_language_change = calls.append

    view._on_peer_source_select(dashboard_module.PEER_SOURCE_MODE_AUTO)

    assert calls[-1].peer_source_mode == "auto"


def test_dashboard_peer_swap_keeps_automatic_mode_and_valid_languages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view = _make_dashboard(monkeypatch)
    changes: list[tuple[str, str, str, str, str]] = []
    view.on_language_change = lambda change: changes.append(
        (
            change.source_code,
            change.target_code,
            change.peer_source_code,
            change.peer_target_code,
            change.peer_source_mode,
        )
    )
    view.set_languages_from_codes("ko", "en", "ja", "fr", "auto")

    view._swap_peer_languages()

    assert view._peer_source_mode == "auto"
    assert view._peer_source_lang_code == "fr"
    assert view._peer_target_lang_code == "ja"
    assert changes[-1] == ("ko", "en", "fr", "ja", "auto")
    assert view.language_card.languages[-1] == (
        "name-ko",
        "name-en",
        dashboard_module.t("dashboard.peer_source.automatic"),
        "name-ja",
    )


def test_dashboard_overlay_peer_buttons_render_consumer_contract_state_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view = _make_dashboard(monkeypatch)
    contract = _make_overlay_peer_contract(
        overlay_intent_enabled=True,
        overlay_state="failed",
        overlay_status_text="Overlay failed",
        overlay_helper_text="Overlay helper copy",
        peer_intent_enabled=True,
        peer_effective_enabled=False,
        peer_status_text="Peer waiting",
        peer_helper_text="Overlay is starting",
    )

    view.set_overlay_peer_contract(contract)

    assert view.overlay_button.states[-1] == {
        "is_on": False,
        "needs_key": True,
        "status_text": None,
        "helper_text": None,
    }
    assert view.peer_button.states[-1] == {
        "is_on": False,
        "needs_key": True,
        "status_text": None,
        "helper_text": None,
    }


def test_dashboard_peer_button_renders_starting_contract_with_spinner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view = _make_dashboard(monkeypatch)
    contract = _make_overlay_peer_contract(
        overlay_intent_enabled=True,
        overlay_state="connected",
        overlay_status_text="Overlay ready",
        peer_intent_enabled=True,
        peer_effective_enabled=False,
        peer_status_text="Peer starting",
        peer_state="starting",
    )

    view.set_overlay_peer_contract(contract)

    assert view.peer_button.states[-1]["is_starting"] is True


def test_dashboard_overlay_failure_notice_is_lowest_priority_notice_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view = _make_dashboard(monkeypatch)
    overlay_failure_notice = dashboard_module.t(
        "settings.overlay.status.failed_with_reason",
        status=dashboard_module.t("settings.overlay.status.failed"),
        reason=dashboard_module.t(
            "settings.overlay.failure.runtime_unavailable",
            default="runtime_unavailable",
        ),
        default="Overlay failed: runtime_unavailable",
    )

    view.set_overlay_peer_contract(
        OverlayPeerConsumerContract(
            overlay=OverlayPeerToggleContract(
                intent_enabled=True,
                effective_enabled=False,
                action_enabled=True,
                state="warning",
                status_text="Failed: runtime unavailable",
                failure_reason="runtime_unavailable",
            ),
            peer=OverlayPeerToggleContract(
                intent_enabled=False,
                effective_enabled=False,
                action_enabled=True,
                state="off",
                status_text="Off",
            ),
        )
    )
    view.set_local_stt_notice("missing")
    view.set_managed_auth_pending(True)
    view.set_managed_auth_pending(False)
    view.set_local_stt_notice(None)

    assert view.display_card.notice_calls == [
        (overlay_failure_notice, "error"),
        (overlay_failure_notice, "error"),
        (overlay_failure_notice, "error"),
        (overlay_failure_notice, "error"),
        (overlay_failure_notice, "error"),
    ]


@pytest.mark.parametrize(
    ("status", "key", "action", "action_key", "tone"),
    (
        (
            "discovery_failed",
            "dashboard.gpu_notice.discovery_failed",
            "rediscover",
            "dashboard.gpu_action.rediscover",
            "error",
        ),
        ("unsupported", "dashboard.gpu_notice.unsupported", None, None, "warning"),
        (
            "unavailable_device",
            "dashboard.gpu_notice.unavailable_device",
            None,
            None,
            "warning",
        ),
        (
            "not_installed",
            "dashboard.gpu_notice.not_installed",
            None,
            None,
            "warning",
        ),
        (
            "invalid",
            "dashboard.gpu_notice.invalid",
            None,
            None,
            "warning",
        ),
        (
            "install_failed",
            "dashboard.gpu_notice.install_failed",
            None,
            None,
            "error",
        ),
        (
            "activation_failed",
            "dashboard.gpu_notice.activation_failed",
            "restart",
            "dashboard.gpu_action.restart",
            "error",
        ),
    ),
)
def test_dashboard_gpu_notice_text_tone_and_action(
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    key: str,
    action: str | None,
    action_key: str | None,
    tone: str,
) -> None:
    view = _make_dashboard(monkeypatch)

    view.set_gpu_notice(GpuDashboardNotice(status=status, action=action))

    assert view.display_card.notice_calls[-1] == (dashboard_module.t(key), tone)
    assert view.display_card.notice_actions[-1][0] == (
        dashboard_module.t(action_key) if action_key is not None else None
    )


def test_dashboard_gpu_download_preempts_and_restores_first_active_notice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view = _make_dashboard(monkeypatch)
    view.set_vrchat_osc_notice(True)
    view.set_managed_auth_pending(True)

    view.set_gpu_notice(GpuDashboardNotice(status="installing", progress_percent=37))
    assert view.display_card.notice_calls[-1] == (
        dashboard_module.t("dashboard.gpu_notice.installing", percent=37),
        "info",
    )

    view.set_gpu_notice(None)
    assert view.display_card.notice_calls[-1] == (
        dashboard_module.t("dashboard.vrchat_osc_disabled"),
        "warning",
    )


@pytest.mark.asyncio
async def test_dashboard_gpu_action_runs_as_page_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view = _make_dashboard(monkeypatch)
    tasks: list[asyncio.Task[None]] = []
    actions: list[str] = []

    class Page:
        def run_task(self, callback) -> None:
            tasks.append(asyncio.create_task(callback()))

    async def on_action(action: str) -> None:
        actions.append(action)

    attach_dummy_page(monkeypatch, view, Page())
    view.on_gpu_notice_action = on_action
    view.set_gpu_notice(GpuDashboardNotice(status="not_installed", action="install"))
    action_callback = view.display_card.notice_actions[-1][1]
    assert callable(action_callback)

    action_callback()
    await tasks[-1]

    assert actions == ["install"]


@pytest.mark.asyncio
async def test_managed_gemma_download_preempts_other_notices_and_can_be_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view = _make_dashboard(monkeypatch)
    tasks: list[asyncio.Task[None]] = []
    actions: list[str] = []

    class Page:
        def run_task(self, callback) -> None:
            tasks.append(asyncio.create_task(callback()))

    async def on_action(action: str) -> None:
        actions.append(action)

    attach_dummy_page(monkeypatch, view, Page())
    view.on_managed_gemma_notice_action = on_action
    view.set_vrchat_osc_notice(True)
    view.set_managed_gemma_notice(
        ManagedGemmaDashboardNotice(
            status="downloading",
            progress_percent=37,
            action="cancel",
        )
    )

    assert view.display_card.notice_calls[-1] == (
        dashboard_module.t("dashboard.managed_gemma_notice.downloading", percent=37),
        "info",
    )
    action_label, action_callback = view.display_card.notice_actions[-1]
    assert action_label == dashboard_module.t("dashboard.managed_gemma_action.cancel")
    assert callable(action_callback)

    action_callback()
    await tasks[-1]
    assert actions == ["cancel"]

    view.set_managed_gemma_notice(None)
    assert view.display_card.notice_calls[-1] == (
        dashboard_module.t("dashboard.vrchat_osc_disabled"),
        "warning",
    )


def test_dashboard_steamvr_overlay_failure_notice_uses_actionable_reason_without_status_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view = _make_dashboard(monkeypatch)
    expected_notice = dashboard_module.t(
        "settings.overlay.failure.steamvr_not_running",
        default="steamvr_not_running",
    )

    view.set_overlay_peer_contract(
        OverlayPeerConsumerContract(
            overlay=OverlayPeerToggleContract(
                intent_enabled=True,
                effective_enabled=False,
                action_enabled=True,
                state="warning",
                status_text="stale contract literal",
                failure_reason="steamvr_not_running",
            ),
            peer=OverlayPeerToggleContract(
                intent_enabled=False,
                effective_enabled=False,
                action_enabled=True,
                state="off",
                status_text="Off",
            ),
        )
    )

    assert view.display_card.notice_calls[-1] == (expected_notice, "error")
    assert view.display_card.notice_yields[-1] is True


def test_dashboard_overlay_notices_yield_to_content_while_others_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fallback_view = _make_dashboard(monkeypatch)
    fallback_view.set_overlay_session_fallback_notice(True)
    assert fallback_view.display_card.notice_yields[-1] is True

    failure_view = _make_dashboard(monkeypatch)
    failure_view.set_overlay_peer_contract(
        OverlayPeerConsumerContract(
            overlay=OverlayPeerToggleContract(
                intent_enabled=True,
                effective_enabled=False,
                action_enabled=True,
                state="warning",
                status_text="stale contract literal",
                failure_reason="steamvr_not_running",
            ),
            peer=OverlayPeerToggleContract(
                intent_enabled=False,
                effective_enabled=False,
                action_enabled=True,
                state="off",
                status_text="Off",
            ),
        )
    )
    assert failure_view.display_card.notice_yields[-1] is True

    gpu_view = _make_dashboard(monkeypatch)
    gpu_view.set_gpu_notice(GpuDashboardNotice(status="install_failed"))
    assert gpu_view.display_card.notice_yields[-1] is False

    osc_view = _make_dashboard(monkeypatch)
    osc_view.set_vrchat_osc_notice(True)
    assert osc_view.display_card.notice_yields[-1] is False


def test_dashboard_overlay_failure_notice_relocalizes_on_apply_locale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view = _make_dashboard(monkeypatch)
    view.set_overlay_peer_contract(
        OverlayPeerConsumerContract(
            overlay=OverlayPeerToggleContract(
                intent_enabled=True,
                effective_enabled=False,
                action_enabled=True,
                state="warning",
                status_text="stale contract literal",
                failure_reason="runtime_disconnected",
            ),
            peer=OverlayPeerToggleContract(
                intent_enabled=False,
                effective_enabled=False,
                action_enabled=True,
                state="off",
                status_text="Off",
            ),
        )
    )

    def localized_t(key: str, **kwargs: object) -> str:
        if key == "settings.overlay.status.failed":
            return "localized failed"
        if key == "settings.overlay.failure.runtime_disconnected":
            return "localized disconnect"
        if key == "settings.overlay.status.failed_with_reason":
            return f"{kwargs['status']} :: {kwargs['reason']}"
        return f"i18n:{key}"

    monkeypatch.setattr(dashboard_module, "t", localized_t)

    view.apply_locale()

    assert view.display_card.notice_calls[-1] == (
        "localized failed :: localized disconnect",
        "error",
    )
    assert view.display_card.notice_calls[-1][0] != "stale contract literal"


def test_dashboard_overlay_and_peer_buttons_toggle_live_from_contract_intent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view = _make_dashboard(monkeypatch)
    peer_toggles: list[bool] = []
    overlay_toggles: list[bool] = []
    view.on_toggle_peer_translation = lambda enabled: peer_toggles.append(enabled)
    view.on_toggle_overlay = lambda enabled: overlay_toggles.append(enabled)

    view.set_overlay_peer_contract(
        _make_overlay_peer_contract(
            overlay_intent_enabled=False,
            overlay_state="off",
            overlay_status_text="Overlay off",
            peer_intent_enabled=False,
            peer_effective_enabled=False,
            peer_status_text="Peer off",
        )
    )
    view.peer_button.on_click()
    view.overlay_button.on_click()

    view.set_overlay_peer_contract(
        _make_overlay_peer_contract(
            overlay_intent_enabled=True,
            overlay_state="connected",
            overlay_status_text="Overlay on",
            peer_intent_enabled=True,
            peer_effective_enabled=True,
            peer_status_text="Peer on",
        )
    )
    view.peer_button.on_click()
    view.overlay_button.on_click()

    assert peer_toggles == [True, False]
    assert overlay_toggles == [True, False]


def test_dashboard_peer_source_selection_restores_follow_self_blank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view = _make_dashboard(monkeypatch)
    changes: list[tuple[str, str, str, str]] = []
    view.on_language_change = lambda change: changes.append(
        (change.source_code, change.target_code, change.peer_source_code, change.peer_target_code)
    )
    view.set_languages_from_codes("ko", "en", "ja", "fr")

    view._on_peer_source_select("ko")

    assert view._peer_source_lang_code == ""
    assert view._peer_target_lang_code == "fr"
    assert view._recent_source_langs == ["ko"]
    assert changes[-1] == ("ko", "en", "", "fr")
    assert view.language_card.languages[-1] == ("name-ko", "name-en", "name-ko", "name-fr")


def test_dashboard_peer_target_selection_restores_follow_self_blank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view = _make_dashboard(monkeypatch)
    changes: list[tuple[str, str, str, str]] = []
    view.on_language_change = lambda change: changes.append(
        (change.source_code, change.target_code, change.peer_source_code, change.peer_target_code)
    )
    view.set_languages_from_codes("ko", "en", "ja", "fr")

    view._on_peer_target_select("en")

    assert view._peer_source_lang_code == "ja"
    assert view._peer_target_lang_code == ""
    assert view._recent_target_langs == ["en"]
    assert changes[-1] == ("ko", "en", "ja", "")
    assert view.language_card.languages[-1] == ("name-ko", "name-en", "name-ja", "name-en")


def test_dashboard_self_source_change_preserves_explicit_peer_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view = _make_dashboard(monkeypatch)
    changes: list[tuple[str, str, str, str]] = []
    view.on_language_change = lambda change: changes.append(
        (change.source_code, change.target_code, change.peer_source_code, change.peer_target_code)
    )
    view.set_languages_from_codes("ko", "en", "ja", "fr")

    view._on_source_select("ja")
    view._on_source_select("de")

    assert view._peer_source_lang_code == "ja"
    assert view._peer_target_lang_code == "fr"
    assert changes[-2] == ("ja", "en", "ja", "fr")
    assert changes[-1] == ("de", "en", "ja", "fr")
    assert view.language_card.languages[-1] == ("name-de", "name-en", "name-ja", "name-fr")


def test_dashboard_peer_language_edits_share_controller_update_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view = _make_dashboard(monkeypatch)
    changes: list[tuple[str, str, str, str]] = []
    view.on_language_change = lambda change: changes.append(
        (change.source_code, change.target_code, change.peer_source_code, change.peer_target_code)
    )

    view._on_peer_source_select("ja")
    view._on_peer_target_select("fr")

    assert changes == [("ko", "en", "ja", "ko"), ("ko", "en", "ja", "fr")]


def test_dashboard_peer_swap_exchanges_source_and_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view = _make_dashboard(monkeypatch)
    changes: list[tuple[str, str, str, str]] = []
    view.on_language_change = lambda change: changes.append(
        (change.source_code, change.target_code, change.peer_source_code, change.peer_target_code)
    )
    view.set_languages_from_codes("ko", "en", "ja", "fr")

    view._swap_peer_languages()

    assert view._peer_source_lang_code == "fr"
    assert view._peer_target_lang_code == "ja"
    assert changes[-1] == ("ko", "en", "fr", "ja")
    assert view.language_card.languages[-1] == ("name-ko", "name-en", "name-fr", "name-ja")


def test_dashboard_self_and_peer_language_row_labels_render_from_i18n(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dashboard_module, "t", lambda key, **_kwargs: f"i18n:{key}")
    view = _make_dashboard(monkeypatch)

    assert view.language_card.row_labels[0] == (
        "i18n:dashboard.language.self",
        "i18n:dashboard.language.peer",
    )

    view.apply_locale()

    assert view.language_card.row_labels[-1] == (
        "i18n:dashboard.language.self",
        "i18n:dashboard.language.peer",
    )


def test_dashboard_peer_and_overlay_button_labels_render_from_i18n(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dashboard_module, "t", lambda key, **_kwargs: f"i18n:{key}")
    monkeypatch.setattr(dashboard_capture_module, "t", lambda key, **_kwargs: f"i18n:{key}")
    view = _make_dashboard(monkeypatch)

    assert view.peer_button.label == "i18n:dashboard.peer_label"
    assert view.overlay_button.label == "i18n:dashboard.overlay_label"

    view.apply_locale()

    assert view.peer_button.label == "i18n:dashboard.peer_label"
    assert view.overlay_button.label == "i18n:dashboard.overlay_label"


def test_dashboard_local_stt_notice_can_change_and_clear_without_touching_display(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view = _make_dashboard(monkeypatch)

    view.set_local_stt_notice("missing")
    view.set_display_text("hello", language_code="ko")
    view.set_local_stt_notice("downloading", percent=63)
    view.set_local_stt_notice(None)

    assert view.display_card.display_calls == [("hello", False, "font-ko")]
    assert view.display_card.notice_calls == [
        (dashboard_module.t("dashboard.local_stt_notice_missing"), "warning"),
        (dashboard_module.t("dashboard.local_stt_notice_downloading_progress", percent=63), "info"),
        (None, None),
    ]


def test_dashboard_bound_intents_receive_every_owned_interaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view = _make_dashboard(monkeypatch)
    calls: dict[str, list] = {
        "submit": [],
        "translation": [],
        "language": [],
        "activity": [],
        "self_capture": [],
        "peer_capture": [],
        "overlay": [],
        "retry_peer": [],
        "gpu": [],
    }
    view.bind_dashboard_intents(
        translation=DashboardTranslationIntents(
            submit_message=lambda source, text: calls["submit"].append((source, text)),
            toggle_translation=lambda enabled: calls["translation"].append(enabled),
            change_language=lambda change: calls["language"].append(change),
            report_input_activity=lambda has_text: calls["activity"].append(has_text),
        ),
        capture=DashboardCaptureIntents(
            toggle_self_capture=lambda enabled: calls["self_capture"].append(enabled),
            toggle_peer_capture=lambda enabled: calls["peer_capture"].append(enabled),
            toggle_overlay=lambda enabled: calls["overlay"].append(enabled),
            retry_peer_process_capture=lambda: calls["retry_peer"].append(True),
            run_gpu_notice_action=lambda action: calls["gpu"].append(action),
        ),
    )

    view._on_submit("hello")
    view._on_message_input_activity(True)
    view._toggle_translation()
    view._toggle_stt()
    view._toggle_peer_translation()
    view._toggle_overlay()
    view._on_source_select("ja")

    assert calls["submit"] == [("You", "hello")]
    assert calls["activity"] == [True]
    assert calls["translation"] == [True]
    assert calls["self_capture"] == [True]
    assert calls["peer_capture"] == [True]
    assert calls["overlay"] == [True]
    assert [change.source_code for change in calls["language"]] == ["ja"]
    assert calls["retry_peer"] == []
    assert calls["gpu"] == []
