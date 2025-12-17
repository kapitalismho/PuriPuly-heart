from __future__ import annotations

from dataclasses import dataclass

from ajebal_daera_translator.core.stt.backend import STTBackend, STTBackendSession


@dataclass(slots=True)
class GoogleSpeechV2Backend(STTBackend):
    recognizer: str

    async def open_session(self) -> STTBackendSession:
        raise NotImplementedError(
            "GoogleSpeechV2Backend streaming is not implemented yet; "
            "wire this backend to google-cloud-speech v2 gRPC in a later phase."
        )

