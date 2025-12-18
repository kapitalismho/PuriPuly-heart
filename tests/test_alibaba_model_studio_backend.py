from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from types import ModuleType

from ajebal_daera_translator.providers.stt.alibaba_model_studio import AlibabaModelStudioRealtimeSTTBackend


def test_alibaba_model_studio_backend_wraps_dashscope_recognition(monkeypatch):
    dashscope = ModuleType("dashscope")
    dashscope.api_key = None  # type: ignore[attr-defined]
    dashscope.base_websocket_api_url = None  # type: ignore[attr-defined]

    audio = ModuleType("dashscope.audio")
    asr = ModuleType("dashscope.audio.asr")
    audio.asr = asr  # type: ignore[attr-defined]
    dashscope.audio = audio  # type: ignore[attr-defined]

    @dataclass
    class RecognitionResult:
        sentence: dict

        def get_sentence(self):
            return self.sentence

        @staticmethod
        def is_sentence_end(sentence: dict) -> bool:
            return bool(sentence.get("sentence_end", False))

    class RecognitionCallback:
        def on_event(self, result):  # pragma: no cover
            raise NotImplementedError

        def on_error(self, message):  # pragma: no cover
            raise NotImplementedError

        def on_close(self) -> None:  # pragma: no cover
            return

        def on_complete(self) -> None:  # pragma: no cover
            return

    class _FailOnce:
        value = True

    class Recognition:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = dict(kwargs)
            self.callback = kwargs["callback"]
            self.started = False
            self.stopped = False
            self.sent = 0
            Recognition.instances.append(self)

        def start(self):
            self.started = True

        def send_audio_frame(self, data: bytes):
            if _FailOnce.value:
                _FailOnce.value = False
                raise RuntimeError("disconnect")
            self.sent += 1
            if self.sent == 1:
                self.callback.on_event(RecognitionResult({"text": "PARTIAL", "sentence_end": False}))
            else:
                self.callback.on_event(RecognitionResult({"text": "FINAL", "sentence_end": True}))

        def stop(self):
            self.stopped = True

    asr.Recognition = Recognition  # type: ignore[attr-defined]
    asr.RecognitionCallback = RecognitionCallback  # type: ignore[attr-defined]
    asr.RecognitionResult = RecognitionResult  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "dashscope", dashscope)
    monkeypatch.setitem(sys.modules, "dashscope.audio", audio)
    monkeypatch.setitem(sys.modules, "dashscope.audio.asr", asr)

    async def run():
        backend = AlibabaModelStudioRealtimeSTTBackend(
            api_key="k",
            model="paraformer-realtime-v2",
            endpoint="wss://example",
            sample_rate_hz=16000,
            max_retries=2,
            retry_backoff_s=0.0,
        )

        session = await backend.open_session()
        await session.send_audio(b"\x01\x02")
        await session.send_audio(b"\x03\x04")
        await session.stop()

        out = []
        async for ev in session.events():
            out.append((ev.text, ev.is_final))
        await session.close()

        assert out == [("PARTIAL", False), ("FINAL", True)]
        assert dashscope.api_key == "k"
        assert dashscope.base_websocket_api_url == "wss://example"
        assert len(Recognition.instances) >= 2  # reconnect happened

    asyncio.run(run())

