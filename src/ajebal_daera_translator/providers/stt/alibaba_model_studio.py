from __future__ import annotations

from dataclasses import dataclass

from ajebal_daera_translator.core.stt.backend import STTBackend, STTBackendSession


@dataclass(slots=True)
class AlibabaModelStudioRealtimeSTTBackend(STTBackend):
    api_key: str
    model: str = "paraformer-realtime-v2"
    endpoint: str = "wss://dashscope-intl.aliyuncs.com/api-ws/v1/inference"

    async def open_session(self) -> STTBackendSession:
        raise NotImplementedError(
            "AlibabaModelStudioRealtimeSTTBackend streaming is not implemented yet; "
            "wire this backend to DashScope WebSocket real-time ASR in a later phase."
        )

