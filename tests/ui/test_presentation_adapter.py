from __future__ import annotations

from types import SimpleNamespace

import pytest

from puripuly_heart.ui.presentation_adapter import FletUiPresentationAdapter


def test_presentation_adapter_exposes_only_named_destinations_and_events() -> None:
    events: list[tuple[object, ...]] = []
    dashboard = object()
    settings = object()
    logs = object()
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

    assert adapter.view_dashboard is dashboard
    assert adapter.view_settings is settings
    assert adapter.view_logs is logs
    assert adapter.debug_ui_preview is True
    assert adapter.get_event_language_codes() == ("ko", "en")
    assert adapter.is_event_translation_enabled() is True
    assert adapter.get_event_stt_state() == "listening"
    adapter.refresh_overlay_peer_contract()
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
        ("refresh",),
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
