from __future__ import annotations

import json
from dataclasses import replace

import pytest

from puripuly_heart.app.wiring.wiring_stt_factory import (
    create_stt_backend_from_resolved_config,
    resolve_peer_stt_runtime_config_from_vnext,
    resolve_self_stt_runtime_config_from_vnext,
)
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.core.stt.backend import STTBackendTranscriptEvent
from tests.app.test_wiring_providers import InMemorySecretStore
from tests.providers.test_soniox_backend import _make_session


@pytest.mark.parametrize("channel", ["self", "peer"])
def test_saved_native_soniox_ids_cannot_drive_product_splits_or_fallback(channel):
    base = AppSettingsVNext()
    settings = replace(
        base,
        intent=replace(
            base.intent,
            stt=replace(
                base.intent.stt,
                provider="soniox",
                soniox=replace(base.intent.stt.soniox, enable_speaker_diarization=True),
            ),
            peer_stt=replace(base.intent.peer_stt, provider="soniox"),
        ),
    )
    resolve = (
        resolve_peer_stt_runtime_config_from_vnext
        if channel == "peer"
        else resolve_self_stt_runtime_config_from_vnext
    )
    resolved = resolve(settings)
    assert resolved.provider_options["enable_speaker_diarization"] is False
    resolved = replace(
        resolved, provider_options={**resolved.provider_options, "enable_speaker_diarization": True}
    )
    secrets = InMemorySecretStore()
    secrets.set("soniox_api_key", "unused-local-test")
    backend = create_stt_backend_from_resolved_config(resolved, secrets=secrets)
    assert backend.enable_speaker_diarization is False
    session = _make_session()
    session.enable_speaker_diarization = backend.enable_speaker_diarization
    session._pending_finalize_requests = 1
    session._handle_message(
        json.dumps(
            {
                "tokens": [
                    {
                        "text": "one ",
                        "speaker": "native-A",
                        "is_final": True,
                        "start_ms": 0,
                        "end_ms": 100,
                    },
                    {
                        "text": "two",
                        "speaker": "native-B",
                        "is_final": True,
                        "start_ms": 100,
                        "end_ms": 200,
                    },
                    {"text": "<fin>", "is_final": True},
                ]
            }
        )
    )
    event = session._event_projection._legacy_events.get_nowait()
    assert isinstance(event, STTBackendTranscriptEvent)
    assert event.text == "one two"
    assert event.final_speaker_runs == ()
