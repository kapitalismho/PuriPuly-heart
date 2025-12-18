from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from types import ModuleType

from ajebal_daera_translator.providers.stt.google_speech_v2 import GoogleSpeechV2Backend


def test_google_speech_v2_backend_builds_requests_and_maps_results(monkeypatch):
    # Stub google.cloud.speech_v2 so tests don't require external deps.
    google = ModuleType("google")
    cloud = ModuleType("google.cloud")
    speech_v2 = ModuleType("google.cloud.speech_v2")
    google.cloud = cloud  # type: ignore[attr-defined]
    cloud.speech_v2 = speech_v2  # type: ignore[attr-defined]

    class _AudioEncoding:
        LINEAR16 = "LINEAR16"

    @dataclass
    class ExplicitDecodingConfig:
        encoding: str
        sample_rate_hertz: int
        audio_channel_count: int

        AudioEncoding = _AudioEncoding

    @dataclass
    class RecognitionConfig:
        explicit_decoding_config: ExplicitDecodingConfig
        language_codes: list[str]

    @dataclass
    class StreamingRecognitionFeatures:
        interim_results: bool = False

    @dataclass
    class StreamingRecognitionConfig:
        config: RecognitionConfig
        streaming_features: StreamingRecognitionFeatures | None = None

    @dataclass
    class StreamingRecognizeRequest:
        recognizer: str = ""
        streaming_config: StreamingRecognitionConfig | None = None
        audio: bytes = b""

    @dataclass
    class _Alt:
        transcript: str

    @dataclass
    class _Result:
        is_final: bool
        alternatives: list[_Alt]

    @dataclass
    class _Response:
        results: list[_Result]

    @dataclass
    class FakeClient:
        seen: list[StreamingRecognizeRequest]

        def __init__(self) -> None:
            self.seen = []

        def streaming_recognize(self, *, requests):
            for req in requests:
                self.seen.append(req)
                if req.audio:
                    yield _Response(results=[_Result(is_final=False, alternatives=[_Alt("PARTIAL")])])
            yield _Response(results=[_Result(is_final=True, alternatives=[_Alt("FINAL")])])

    speech_v2.ExplicitDecodingConfig = ExplicitDecodingConfig  # type: ignore[attr-defined]
    speech_v2.RecognitionConfig = RecognitionConfig  # type: ignore[attr-defined]
    speech_v2.StreamingRecognitionFeatures = StreamingRecognitionFeatures  # type: ignore[attr-defined]
    speech_v2.StreamingRecognitionConfig = StreamingRecognitionConfig  # type: ignore[attr-defined]
    speech_v2.StreamingRecognizeRequest = StreamingRecognizeRequest  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.cloud", cloud)
    monkeypatch.setitem(sys.modules, "google.cloud.speech_v2", speech_v2)

    async def run():
        client = FakeClient()
        backend = GoogleSpeechV2Backend(
            recognizer="projects/p/locations/l/recognizers/r",
            endpoint="speech.googleapis.com",
            sample_rate_hz=16000,
            language_codes=("ko-KR",),
            client=client,
        )
        session = await backend.open_session()
        await session.send_audio(b"\x01\x02")
        await session.stop()

        texts = []
        async for ev in session.events():
            texts.append((ev.text, ev.is_final))
        await session.close()

        assert texts == [("PARTIAL", False), ("FINAL", True)]
        assert client.seen[0].recognizer == "projects/p/locations/l/recognizers/r"
        assert client.seen[0].streaming_config is not None
        assert client.seen[0].streaming_config.config.language_codes == ["ko-KR"]
        assert client.seen[0].streaming_config.config.explicit_decoding_config.sample_rate_hertz == 16000
        assert client.seen[1].audio == b"\x01\x02"

    asyncio.run(run())

