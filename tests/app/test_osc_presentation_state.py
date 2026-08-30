from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest
from puripuly_heart.app.services.settings_application import osc_control_presentation_state

from puripuly_heart.app.services.osc.state_publisher import state_from_settings
from puripuly_heart.config.provider_values import STTProviderName
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.config.translation_values import TranslationConnection, TranslationModel


def test_presentation_state_captures_the_post_apply_canonical_snapshot() -> None:
    settings = AppSettingsVNext()
    settings = replace(
        settings,
        intent=replace(
            settings.intent,
            languages=replace(settings.intent.languages, source_language="ja"),
            stt=replace(settings.intent.stt, provider=STTProviderName.SONIOX.value),
            translation=replace(
                settings.intent.translation,
                model=TranslationModel.GEMINI_37_FLASH.value,
                connection=TranslationConnection.OFFICIAL_BYOK.value,
            ),
        ),
    )
    canonical = state_from_settings(
        settings,
        self_capture=True,
        translation=True,
    )

    state = osc_control_presentation_state(
        settings,
        canonical_state=canonical,
        changed_control="PuriPuly_Translator",
        self_capture_effective=False,
    )

    assert state.changed_control == "PuriPuly_Translator"
    assert state.self_capture is False
    assert state.translation is True
    assert state.self_source_language == "ja"
    assert state.self_asr == "soniox"
    assert state.self_asr_setting == STTProviderName.SONIOX.value
    assert state.translation_model == TranslationModel.GEMINI_37_FLASH.value
    assert state.translation_connection == TranslationConnection.OFFICIAL_BYOK.value
    assert isinstance(state.translation_connection_history, tuple)
    with pytest.raises(FrozenInstanceError):
        setattr(state, "translation", False)
