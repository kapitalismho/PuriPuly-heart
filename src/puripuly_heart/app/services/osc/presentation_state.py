from __future__ import annotations

from typing import Any

from puripuly_heart.app.ports.ui_models import (
    OscControlPresentationName,
    OscControlPresentationState,
)
from puripuly_heart.app.services.osc.state_publisher import OscCanonicalState


def presentation_state_from_settings(
    settings: Any,
    *,
    canonical_state: OscCanonicalState,
    changed_control: OscControlPresentationName,
    self_capture_effective: bool | None = None,
) -> OscControlPresentationState:
    translation = settings.translation
    fallback = translation.fallback
    return OscControlPresentationState(
        changed_control=changed_control,
        self_capture=(
            canonical_state.self_capture
            if self_capture_effective is None
            else bool(self_capture_effective)
        ),
        peer_capture=canonical_state.peer_capture,
        translation=canonical_state.translation,
        captions=canonical_state.captions,
        peer_source_mode=settings.languages.peer_source_mode,
        mute_sync=canonical_state.mute_sync,
        chatbox_source=canonical_state.chatbox_source,
        self_source_language=canonical_state.self_source_language,
        self_target_language=canonical_state.self_target_language,
        peer_source_language=canonical_state.peer_source_language,
        peer_target_language=canonical_state.peer_target_language,
        self_asr=canonical_state.self_asr,
        peer_asr=canonical_state.peer_asr,
        self_asr_setting=str(getattr(settings.provider.stt, "value", settings.provider.stt)),
        peer_asr_setting=str(
            getattr(settings.provider.peer_stt, "value", settings.provider.peer_stt)
        ),
        custom_stt_mode=str(settings.custom_stt.mode),
        custom_stt_compatibility=str(settings.custom_stt.compatibility),
        llm_provider=str(getattr(settings.provider.llm, "value", settings.provider.llm)),
        translation_model=str(getattr(translation.model, "value", translation.model)),
        translation_connection=str(
            getattr(translation.connection, "value", translation.connection)
        ),
        translation_connection_history=tuple(
            sorted(
                (
                    str(getattr(model, "value", model)),
                    str(getattr(connection, "value", connection)),
                )
                for model, connection in translation.connection_history.items()
            )
        ),
        translation_http_extension_id=translation.http_extension_id,
        translation_previous_model=(
            None
            if translation.previous_llm_model is None
            else str(
                getattr(
                    translation.previous_llm_model,
                    "value",
                    translation.previous_llm_model,
                )
            )
        ),
        fallback=canonical_state.fallback,
        fallback_enabled=bool(fallback.enabled),
        fallback_model=str(getattr(fallback.model, "value", fallback.model)),
        fallback_connection=str(getattr(fallback.connection, "value", fallback.connection)),
    )


__all__ = ["presentation_state_from_settings"]
