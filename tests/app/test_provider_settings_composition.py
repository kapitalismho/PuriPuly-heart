from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from puripuly_heart.app.services.settings_application import settings_view_surface_snapshots

from puripuly_heart.composition.application_runtime import compose_application_runtime
from puripuly_heart.config.settings_vnext.facade import load_vnext_settings, save_vnext_settings
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.ui.presentation_adapter import FletUiPresentationAdapter
from puripuly_heart.ui.views.settings import SettingsView


@pytest.mark.parametrize("already_selected", [False, True])
async def test_cloud_free_tier_options_survive_composed_apply_and_reload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    already_selected: bool,
) -> None:
    peer_provider = "rolling_free"
    monkeypatch.setattr(SettingsView, "_populate_host_apis", lambda self: None)
    monkeypatch.setattr(SettingsView, "_refresh_microphones", lambda self: None)
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

    for selection in (False, True, False):
        pool = ("deepgram",) if selection else ("elevenlabs_scribe", "gemini_transcribe")
        view._on_cloud_free_tier_changed(pool)
        assert len(pending) == 1
        intent = pending.pop()
        assert intent is not None
        await app.apply_provider_intent(intent)
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
            assert tuple(settings.intent.stt.cloud_free_tier_providers) == pool
        assert view._peer_stt_text.content.value == selected_label
        assert not view.has_provider_changes
        assert not view.cloud_free_tier_control().ignore_interactions
