from __future__ import annotations

from types import SimpleNamespace

import pytest

from puripuly_heart.app.ports.ui_models import OverlayPeerPresentationState
from puripuly_heart.ui.event_bridge import UIEventBridge
from puripuly_heart.ui.presentation_adapter import FletUiPresentationAdapter


def test_presentation_adapter_exposes_only_named_destinations_and_events() -> None:
    events: list[tuple[object, ...]] = []
    dashboard = SimpleNamespace(
        is_translation_on=False,
        set_translation_enabled=lambda value: events.append(("translation-enabled", value)),
        set_stt_enabled=lambda value: events.append(("stt-enabled", value)),
        set_translation_needs_key=lambda value: events.append(("translation-key", value)),
        set_stt_needs_key=lambda value: events.append(("stt-key", value)),
        set_managed_auth_pending=lambda value: events.append(("managed-auth", value)),
        set_gpu_notice=lambda value: events.append(("gpu-notice", value)),
        set_managed_gemma_notice=lambda value: events.append(("gemma-notice", value)),
        set_stt_starting=lambda value: events.append(("stt-starting", value)),
        set_local_stt_notice_model=lambda value: events.append(("stt-model", value)),
        set_local_stt_notice=lambda value, **kwargs: events.append(("stt-notice", value, kwargs)),
        set_vrchat_osc_notice=lambda value: events.append(("osc-notice", value)),
        set_overlay_session_fallback_notice=lambda value: events.append(
            ("overlay-fallback", value)
        ),
        set_languages_from_codes=lambda *args: events.append(("languages", args)),
        set_recent_languages=lambda *args: events.append(("recent-languages", args)),
        set_peer_auto_detect_available=lambda value: events.append(("peer-auto", value)),
        set_overlay_peer_contract=lambda value: events.append(
            ("dashboard-overlay-contract", value)
        ),
    )
    settings = SimpleNamespace(
        set_gpu_devices=lambda **kwargs: events.append(("gpu-devices", kwargs)),
        load_from_settings=lambda *args, **kwargs: events.append(("load-settings", args, kwargs)),
        refresh_after_openrouter_pkce_success=lambda *args, **kwargs: events.append(
            ("pkce-refresh", args, kwargs)
        ),
        set_overlay_calibration=lambda value: events.append(("calibration", value)),
        refresh_loopback_capture_target=lambda value: events.append(("capture-target", value)),
        set_local_cpu_auto_available=lambda value: events.append(("cpu-auto", value)),
        set_managed_key_state=lambda **kwargs: events.append(("managed-key", kwargs)),
        set_overlay_peer_contract=lambda value: events.append(("settings-overlay-contract", value)),
    )
    logs = SimpleNamespace(append_conversation_record=lambda **kwargs: None)
    host = SimpleNamespace(
        view_dashboard=dashboard,
        view_settings=settings,
        view_logs=logs,
        debug_ui_preview=True,
        refresh_overlay_peer_contract=lambda: events.append(("refresh",)),
        apply_locale=lambda: events.append(("locale",)),
        add_history_entry=lambda *args, **kwargs: events.append(("history", args, kwargs)),
        get_event_language_codes=lambda: ("ko", "en"),
        is_event_translation_enabled=lambda: True,
        get_event_stt_state=lambda: "listening",
        clear_managed_auth_pending_state=lambda: events.append(("clear-auth",)),
        show_snackbar=lambda *args, **kwargs: events.append(("snackbar", args, kwargs)),
        on_github_star_translation_success=lambda: events.append(("star",)),
        on_telemetry_translation_success=lambda: events.append(("telemetry",)),
        on_overlay_state_changed=lambda **kwargs: events.append(("overlay", kwargs)),
        on_desktop_overlay_state_changed=lambda *args, **kwargs: events.append(
            ("desktop-overlay", args, kwargs)
        ),
        show_qq_managed_auth_dialog=lambda: events.append(("qq",)),
        show_founder_letter_dialog=lambda: events.append(("founder",)),
        show_local_qwen_hallucination_dialog=lambda: events.append(("qwen",)),
    )
    adapter = FletUiPresentationAdapter(host)

    assert not hasattr(adapter, "view_dashboard")
    assert not hasattr(adapter, "view_settings")
    assert not hasattr(adapter, "view_logs")
    assert adapter.debug_ui_preview is True
    assert adapter.get_event_language_codes() == ("ko", "en")
    assert adapter.is_event_translation_enabled() is True
    assert adapter.get_event_stt_state() == "listening"
    adapter.refresh_overlay_peer_contract(
        OverlayPeerPresentationState(
            overlay_intent_enabled=True,
            overlay_state="connected",
            overlay_failure_reason=None,
            peer_intent_enabled=True,
            peer_effective_enabled=True,
            peer_warning_reason=None,
            peer_activation_starting=False,
        )
    )
    adapter.apply_locale()
    adapter.add_history_entry("self", text="hello")
    adapter.clear_managed_auth_pending_state()
    adapter.show_snackbar("message", "orange")
    adapter.on_github_star_translation_success()
    adapter.on_telemetry_translation_success()
    adapter.on_overlay_state_changed(state="connected")
    adapter.on_desktop_overlay_state_changed("connected", interaction_mode="locked")
    adapter.show_qq_managed_auth_dialog()
    adapter.show_founder_letter_dialog()
    adapter.show_local_qwen_hallucination_dialog()

    assert events == [
        ("settings-overlay-contract", host.overlay_peer_contract),
        ("dashboard-overlay-contract", host.overlay_peer_contract),
        ("locale",),
        ("history", ("self",), {"text": "hello"}),
        ("clear-auth",),
        ("snackbar", ("message", "orange"), {}),
        ("star",),
        ("telemetry",),
        ("overlay", {"state": "connected"}),
        ("desktop-overlay", ("connected",), {"interaction_mode": "locked"}),
        ("qq",),
        ("founder",),
        ("qwen",),
    ]
    assert not hasattr(adapter, "controller")
    assert not hasattr(adapter, "hub")
    assert not hasattr(adapter, "settings")


def test_presentation_adapter_projects_semantic_dashboard_and_settings_outputs(
    tmp_path,
) -> None:
    events: list[tuple[object, ...]] = []
    dashboard = SimpleNamespace(
        is_translation_on=False,
        set_translation_enabled=lambda value: events.append(("translation-enabled", value)),
        set_stt_enabled=lambda value: events.append(("stt-enabled", value)),
        set_translation_needs_key=lambda value: events.append(("translation-key", value)),
        set_stt_needs_key=lambda value: events.append(("stt-key", value)),
        set_managed_auth_pending=lambda value: events.append(("managed-auth", value)),
        set_gpu_notice=lambda value: events.append(("gpu-notice", value)),
        set_managed_gemma_notice=lambda value: events.append(("gemma-notice", value)),
        set_stt_starting=lambda value: events.append(("stt-starting", value)),
        set_local_stt_notice_model=lambda value: events.append(("stt-model", value)),
        set_local_stt_notice=lambda value, **kwargs: events.append(("stt-notice", value, kwargs)),
        set_vrchat_osc_notice=lambda value: events.append(("osc-notice", value)),
        set_overlay_session_fallback_notice=lambda value: events.append(
            ("overlay-fallback", value)
        ),
        set_languages_from_codes=lambda *args: events.append(("languages", args)),
        set_recent_languages=lambda *args: events.append(("recent-languages", args)),
        set_peer_auto_detect_available=lambda value: events.append(("peer-auto", value)),
    )
    settings = SimpleNamespace(
        set_gpu_devices=lambda **kwargs: events.append(("gpu-devices", kwargs)),
        load_from_settings=lambda *args, **kwargs: events.append(("load-settings", args, kwargs)),
        refresh_after_openrouter_pkce_success=lambda *args, **kwargs: events.append(
            ("pkce-refresh", args, kwargs)
        ),
        set_overlay_calibration=lambda value: events.append(("calibration", value)),
        refresh_loopback_capture_target=lambda value: events.append(("capture-target", value)),
        set_local_cpu_auto_available=lambda value: events.append(("cpu-auto", value)),
        set_managed_key_state=lambda **kwargs: events.append(("managed-key", kwargs)),
    )
    logs = SimpleNamespace(append_conversation_record=lambda **kwargs: None)
    host = SimpleNamespace(
        view_dashboard=dashboard,
        view_settings=settings,
        view_logs=logs,
        get_event_language_codes=lambda: ("ko", "en"),
        is_event_translation_enabled=lambda: True,
        get_event_stt_state=lambda: "listening",
        clear_managed_auth_pending_state=lambda: None,
        show_snackbar=lambda *args, **kwargs: None,
        on_github_star_translation_success=lambda: None,
        on_telemetry_translation_success=lambda: None,
        on_overlay_state_changed=lambda **kwargs: None,
    )
    adapter = FletUiPresentationAdapter(host)
    runtime_logging = SimpleNamespace(
        attach_realtime_sink=lambda sink: events.append(("log-sink", sink))
    )
    settings_value = object()
    calibration = object()
    notice = object()
    gemma_notice = object()
    devices = (object(),)

    adapter.attach_runtime_log_sink(runtime_logging)
    adapter.set_dashboard_translation_enabled(True)
    adapter.set_dashboard_stt_enabled(True)
    adapter.set_dashboard_translation_needs_key(False)
    adapter.set_dashboard_stt_needs_key(False)
    adapter.set_dashboard_managed_auth_pending(True)
    adapter.set_dashboard_gpu_state(
        devices=devices,
        state="ready",
        progress_percent=None,
        notice=notice,
        publish_notice=True,
    )
    adapter.set_dashboard_local_stt_notice(
        status="downloading",
        model_id="model",
        percent=25,
        starting=True,
    )
    adapter.set_dashboard_managed_gemma_notice(gemma_notice)
    adapter.set_dashboard_vrchat_osc_notice(True)
    adapter.set_dashboard_overlay_session_fallback_notice(True)
    adapter.set_dashboard_languages(
        source_language="ko",
        target_language="en",
        peer_source_language="ja",
        peer_target_language="ko",
        peer_source_mode="auto",
        recent_source_languages=["ko"],
        recent_target_languages=["en"],
        peer_auto_detect_available=True,
    )
    assert adapter.render_settings(settings_value, config_path=tmp_path / "settings.json")
    assert adapter.refresh_settings_after_openrouter_pkce_success(
        settings_value,
        config_path=tmp_path / "settings.json",
    )
    adapter.set_settings_overlay_calibration(calibration)
    adapter.refresh_settings_loopback_capture_target(settings_value)
    adapter.set_settings_local_cpu_auto_available(True)
    adapter.set_settings_managed_key_state(
        visible=True,
        remaining_percent=50,
        referral_id="ABC123",
        pass_status=None,
    )

    assert adapter.dashboard_translation_enabled() is False
    assert ("log-sink", logs) in events
    assert ("translation-enabled", True) in events
    assert ("gpu-devices", {"devices": devices}) in events
    assert ("gpu-notice", notice) in events
    assert ("stt-notice", "downloading", {"percent": 25}) in events
    assert ("gemma-notice", gemma_notice) in events
    assert (
        "load-settings",
        (settings_value,),
        {"config_path": tmp_path / "settings.json", "preserve_custom_vocab_draft": False},
    ) in events
    assert (
        "pkce-refresh",
        (settings_value,),
        {"config_path": tmp_path / "settings.json"},
    ) in events
    assert (
        "managed-key",
        {"visible": True, "remaining_percent": 50, "referral_id": "ABC123", "pass_status": None},
    ) in events


def test_presentation_adapter_owns_ui_event_bridge_composition() -> None:
    host = SimpleNamespace(
        view_dashboard=SimpleNamespace(),
        view_logs=SimpleNamespace(),
        get_event_language_codes=lambda: ("ko", "en"),
        is_event_translation_enabled=lambda: True,
        get_event_stt_state=lambda: None,
        clear_managed_auth_pending_state=lambda: None,
        show_snackbar=lambda *args: None,
        on_github_star_translation_success=lambda: None,
        on_telemetry_translation_success=lambda: None,
        on_overlay_state_changed=lambda **kwargs: None,
    )
    adapter = FletUiPresentationAdapter(host)

    bridge = adapter.create_ui_event_bridge(
        event_queue=object(),
        runtime_logging=SimpleNamespace(),
    )

    assert isinstance(bridge, UIEventBridge)


def test_presentation_adapter_preserves_missing_optional_history_destination() -> None:
    adapter = FletUiPresentationAdapter(SimpleNamespace())

    adapter.add_history_entry("Mic", "hello", language_code="en")


@pytest.mark.asyncio
async def test_presentation_adapter_awaits_ui_owned_shutdown_hooks() -> None:
    events: list[str] = []

    async def record(name: str) -> None:
        events.append(name)

    host = SimpleNamespace(
        close_after_launch_tasks=lambda: record("after-launch"),
        close_github_star_prompt_runtime=lambda: record("star-runtime"),
        close_oauth_runtime=lambda: record("oauth-runtime"),
    )
    adapter = FletUiPresentationAdapter(host)

    await adapter.close_after_launch_tasks()
    await adapter.close_github_star_prompt_runtime()
    await adapter.close_oauth_runtime()

    assert events == ["after-launch", "star-runtime", "oauth-runtime"]
