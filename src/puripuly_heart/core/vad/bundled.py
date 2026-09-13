from __future__ import annotations

from pathlib import Path

SILERO_VAD_VERSION = "6.2.1"
SILERO_VAD_RESOURCE_RELATIVE_PATH = "data/vad/silero_vad.onnx"
SILERO_VAD_RESOURCE_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"


def bundled_silero_vad_onnx_path() -> Path:
    model_path = Path(__file__).resolve().parents[2] / SILERO_VAD_RESOURCE_RELATIVE_PATH
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Bundled Silero VAD model missing: {SILERO_VAD_RESOURCE_RELATIVE_PATH}"
        )
    return model_path
