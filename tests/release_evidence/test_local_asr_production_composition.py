from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from puripuly_heart.composition.local_asr_production_evidence import (
    compose_local_asr_production_evidence,
)
from puripuly_heart.core.stt.backend import (
    STTProviderTurnIdentity,
    STTProviderTurnRequest,
    STTProviderTurnTerminal,
)
from puripuly_heart.core.stt.scoped_engine import ScopedRecognitionEngine
from puripuly_heart.core.stt.scoped_event_buffer import STTProviderEventBuffer
from puripuly_heart.domain.events import STTFinalEvent
from puripuly_heart.domain.models import Transcript
from puripuly_heart.release_evidence import local_asr_production_composition as evidence


def _final_event(*, channel: str, text: str = "transcript") -> object:
    return SimpleNamespace(
        transcript=SimpleNamespace(
            text=text,
            is_final=True,
            channel=channel,
            final_language_runs=(),
        )
    )


def test_require_final_preserves_channel_evidence() -> None:
    fact = evidence._require_final(
        _final_event(channel="peer"),
        channel="peer",
        stage="peer inference",
    )

    assert fact["text"] == "transcript"
    assert fact["is_final"] is True
    assert fact["channel"] == "peer"


def test_require_final_rejects_cross_channel_result() -> None:
    with pytest.raises(RuntimeError, match="expected 'peer'"):
        evidence._require_final(
            _final_event(channel="self"),
            channel="peer",
            stage="peer inference",
        )


def test_runner_rejects_non_packaged_execution_with_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delattr(evidence.sys, "frozen", raising=False)
    report_path = tmp_path / "report.json"

    result = evidence.run_local_asr_production_composition(
        audio_path=tmp_path / "speech.wav",
        report_path=report_path,
        candidate="candidate-sha",
        expected_gpu_name="RX 7900 XTX",
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert result == 1
    assert report["status"] == "failed"
    assert report["candidate"] == "candidate-sha"
    assert report["failure_type"] == "RuntimeError"
    assert "packaged Windows app" in report["failure"]


def test_execute_defaults_to_the_package_evidence_composition_factory() -> None:
    parameter = inspect.signature(evidence._execute).parameters["composition_factory"]

    assert parameter.default is compose_local_asr_production_evidence


@pytest.mark.asyncio
async def test_peer_production_probe_uses_owned_scoped_recognition_path() -> None:
    events: list[object] = []

    class Session:
        def __init__(self) -> None:
            self.buffer = STTProviderEventBuffer()
            self.audio = bytearray()

        async def begin_turn(self, _request: STTProviderTurnRequest) -> None:
            return None

        async def send_turn_audio(
            self,
            _identity: STTProviderTurnIdentity,
            pcm16le: bytes,
            **_kwargs,
        ) -> None:
            self.audio.extend(pcm16le)

        async def seal_turn(
            self,
            identity: STTProviderTurnIdentity,
            **_kwargs,
        ) -> None:
            self.buffer.put(
                STTProviderTurnTerminal(
                    identity=identity,
                    outcome="final",
                    text="scoped transcript",
                    text_authority="authoritative",
                )
            )

        async def abort_turn(self, _identity: STTProviderTurnIdentity, **_kwargs) -> None:
            return None

        async def turn_events(self):
            async for event in self.buffer.events():
                yield event

        async def stop(self) -> None:
            return None

        async def close(self) -> None:
            self.buffer.close()

    request = SimpleNamespace(
        provider_id="fake-scoped",
        provider_signature=("fake-scoped",),
        runtime_signature=("fake-scoped", "runtime"),
        session_options=None,
        config=SimpleNamespace(
            source_language="en",
            source_mode="manual",
            sample_rate_hz=16000,
            vad_speech_threshold=0.6,
            vad_hangover_ms=900,
            vad_pre_roll_ms=500,
        ),
    )
    session = Session()
    engine = ScopedRecognitionEngine(
        session_factory=lambda _settings, _epoch: asyncio.sleep(0, result=session),
        event_sink=events.append,
        accepted_settings_scope=(
            request.provider_id,
            request.provider_signature,
            request.runtime_signature,
        ),
    )

    class PeerRuntime:
        def __init__(self) -> None:
            self.receipts = []

        async def handle_peer_owned_vad_event(self, owned: object) -> None:
            await engine.handle_owned_vad_event(owned)

        async def handle_provider_turn_terminal(self, receipt, terminal):
            self.receipts.append(receipt)
            transcript = Transcript(
                utterance_id=receipt.identity.segment_id,
                text=terminal.text,
                is_final=True,
                channel="peer",
                publication_generation=receipt.identity.activation_generation,
                source_order=receipt.identity.segment_order,
            )
            return STTFinalEvent(receipt.identity.segment_id, transcript)

    peer_runtime = PeerRuntime()
    samples = np.linspace(-0.5, 0.5, 160, dtype=np.float32)
    try:
        final = await evidence._send_utterance(
            application=SimpleNamespace(peer_vad=peer_runtime),
            channel="peer",
            samples=samples,
            events=events,
            peer_request=request,
            activation_generation=7,
        )
    finally:
        await engine.close()

    assert final.transcript.text == "scoped transcript"
    assert final.transcript.publication_generation == 7
    assert peer_runtime.receipts[0].segment.content_sample_count == samples.size
    assert len(session.audio) == samples.size * 2
