from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from puripuly_heart.composition.application_runtime import (
    _require_self_capture_owner,
)


def test_reacquires_self_capture_with_direct_self_translation_owner() -> None:
    self_translation_channel = object()
    local_asr_runtime = object()
    audio_gate = object()
    created_owner = object()
    calls: list[tuple[object, object, object, object]] = []
    pipeline = SimpleNamespace(
        self_capture=None,
        self_translation_channel=self_translation_channel,
        local_asr_runtime=local_asr_runtime,
        vrc_mic_audio_gate=audio_gate,
    )

    class Factory:
        @staticmethod
        def compose_self(
            vad_runtime: object,
            provider_runtime: object,
            channel_reset: object,
            gate: object,
        ) -> object:
            calls.append((vad_runtime, provider_runtime, channel_reset, gate))
            return created_owner

    owner = _require_self_capture_owner(
        cast(Any, pipeline),
        cast(Any, Factory()),
    )
    reused = _require_self_capture_owner(
        cast(Any, pipeline),
        cast(Any, Factory()),
    )

    assert owner is created_owner
    assert reused is created_owner
    assert pipeline.self_capture is created_owner
    assert calls == [
        (
            self_translation_channel,
            local_asr_runtime,
            self_translation_channel,
            audio_gate,
        )
    ]
