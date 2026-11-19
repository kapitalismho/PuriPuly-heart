from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from puripuly_heart.app.services.settings.settings_application import (
    settings_view_surface_snapshots,
)
from puripuly_heart.composition.application_runtime import compose_application_runtime
from puripuly_heart.config.provider_values import STTProviderName
from puripuly_heart.config.settings_vnext.facade import load_vnext_settings, save_vnext_settings
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.ui.i18n import t
from puripuly_heart.ui.presentation_adapter import FletUiPresentationAdapter
from puripuly_heart.ui.views.settings import SettingsView


def _diarization_values(settings: AppSettingsVNext) -> dict[str, bool]:
    return {
        "soniox": settings.intent.stt.soniox.enable_speaker_diarization,
        "elevenlabs_scribe": settings.intent.stt.elevenlabs_scribe.enable_speaker_diarization,
    }


@pytest.mark.parametrize("already_selected", [False, True])
@pytest.mark.parametrize("peer_provider", ["soniox", "elevenlabs_scribe", "rolling_free"])
async def test_peer_provider_options_survive_composed_apply_and_reload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    already_selected: bool,
    peer_provider: str,
) -> None:
    monkeypatch.setattr(SettingsView, "_load_secrets", lambda self, *args: None)
    monkeypatch.setattr(SettingsView, "update", lambda self: None)
    view = SettingsView()
    presentation = FletUiPresentationAdapter(
        SimpleNamespace(debug_ui_preview=False, view_settings=view)
    )
    config_path = tmp_path / "settings.json"
    app = compose_application_runtime(presentation=presentation, config_path=config_path)
    base = AppSettingsVNext()
    initial = replace(
        base,
        intent=replace(
            base.intent,
            stt=replace(base.intent.stt, provider="soniox"),
            peer_stt=replace(
                base.intent.peer_stt,
                provider=peer_provider if already_selected else "qwen_audio",
            ),
        ),
    )
    assert save_vnext_settings(config_path, initial).ok
    app._settings.settings.canonical = initial
    provider, general, prompt, overlay = settings_view_surface_snapshots(initial)
    presentation.render_settings(
        provider=provider,
        general=general,
        prompt=prompt,
        overlay=overlay,
        config_path=config_path,
    )
    pending = []
    view.on_providers_changed = lambda: pending.append(view.build_provider_apply_settings())
    view._on_peer_stt_selected(peer_provider)
    selected_label = view._peer_stt_text.content.value

    diarization = peer_provider != "rolling_free"
    for step in range(3):
        selection = step % 2 == 0
        if diarization:
            view._on_speaker_diarization_click(None)
        else:
            pool = ("deepgram",) if selection else ("elevenlabs_scribe", "gemini_transcribe")
            view._on_cloud_free_tier_changed(pool)

        assert len(pending) == 1
        intent = pending.pop()
        assert intent is not None
        outcome = await app.apply_settings_change(intent)
        assert outcome.saved is True
        assert outcome.transaction is not None
        view.acknowledge_provider_apply_settings(intent)
        app.refresh_settings_projection()

        canonical = app._settings.settings.canonical
        loaded = load_vnext_settings(config_path)
        assert loaded.ok
        assert canonical is not None
        assert loaded.settings is not None
        for settings in (canonical, loaded.settings):
            assert settings.intent.peer_stt.provider == peer_provider
            assert settings.intent.stt.provider == "soniox"
            if diarization:
                assert _diarization_values(settings) == {
                    "soniox": selection,
                    "elevenlabs_scribe": selection,
                }
            else:
                assert tuple(settings.intent.stt.cloud_free_tier_providers) == pool
        assert view._peer_stt_text.content.value == selected_label
        assert not view.has_provider_changes
        if diarization:
            assert not view.speaker_diarization_control().ignore_interactions
            assert view._speaker_diarization_text.content.value == t(
                "settings.option.on" if selection else "settings.option.off"
            )
        else:
            assert not view.cloud_free_tier_control().ignore_interactions


async def test_shared_speaker_recognition_survives_overlapping_apply_and_provider_switch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(SettingsView, "_load_secrets", lambda self, *args: None)
    monkeypatch.setattr(SettingsView, "update", lambda self: None)
    view = SettingsView()
    presentation = FletUiPresentationAdapter(
        SimpleNamespace(debug_ui_preview=False, view_settings=view)
    )
    config_path = tmp_path / "settings.json"
    app = compose_application_runtime(presentation=presentation, config_path=config_path)
    base = AppSettingsVNext()
    initial = replace(
        base,
        intent=replace(
            base.intent,
            peer_stt=replace(base.intent.peer_stt, provider="elevenlabs_scribe"),
        ),
    )
    assert save_vnext_settings(config_path, initial).ok
    app._settings.settings.canonical = initial
    provider, general, prompt, overlay = settings_view_surface_snapshots(initial)
    presentation.render_settings(
        provider=provider,
        general=general,
        prompt=prompt,
        overlay=overlay,
        config_path=config_path,
    )
    pending = []
    view.on_providers_changed = lambda: pending.append(view.build_provider_apply_settings())

    def shown() -> str:
        return view._speaker_diarization_text.content.value

    async def apply(intent) -> None:
        outcome = await app.apply_settings_change(intent)
        assert outcome.saved is True
        view.acknowledge_provider_apply_settings(intent)
        app.refresh_settings_projection()

    view._on_speaker_diarization_click(None)
    view._on_speaker_diarization_click(None)
    view._on_speaker_diarization_click(None)
    first = pending[-1]
    view._on_peer_stt_selected("soniox")
    view._on_speaker_diarization_click(None)
    latest = pending[-1]

    await apply(first)
    committed = app._settings.settings.canonical
    assert committed is not None
    assert _diarization_values(committed) == {"soniox": True, "elevenlabs_scribe": True}
    assert view._peer_stt_text.content.value == view._stt_provider_display_label(
        STTProviderName.SONIOX
    )
    assert shown() == t("settings.option.off")
    assert view.has_provider_changes

    await apply(latest)
    assert not view.has_provider_changes
    assert shown() == t("settings.option.off")
    view._on_peer_stt_selected("elevenlabs_scribe")
    assert shown() == t("settings.option.off")

    loaded = load_vnext_settings(config_path)
    assert loaded.ok and loaded.settings is not None
    assert loaded.settings.intent.peer_stt.provider == "soniox"
    assert _diarization_values(loaded.settings) == {"soniox": False, "elevenlabs_scribe": False}
    reopened, *_ = settings_view_surface_snapshots(loaded.settings)
    assert reopened.scribe_speaker_diarization_enabled is False
    assert reopened.soniox_speaker_diarization_enabled is False
