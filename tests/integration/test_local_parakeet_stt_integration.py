from __future__ import annotations

import importlib.util
import subprocess
import sys

import pytest

from tests.integration.helpers import (
    integration_mark,
    load_required_audio_wav,
    require_local_parakeet_model_assets,
    require_optional_module,
    skip_if_local_sherpa_runtime_unavailable,
)

pytestmark = integration_mark()

_SHERPA_UNAVAILABLE_REASON = "sherpa_onnx is unavailable for Japanese Parakeet STT integration"
_SHERPA_IMPORT_PROBE_TIMEOUT_S = 10


def _assert_non_empty_transcript_text(transcript: str) -> None:
    assert isinstance(transcript, str), "expected transcript text"
    assert transcript.strip(), "expected non-empty transcript text"


def _require_sherpa_onnx_present() -> None:
    if importlib.util.find_spec("sherpa_onnx") is not None:
        return
    require_optional_module("sherpa_onnx", reason=_SHERPA_UNAVAILABLE_REASON)


def _require_sherpa_onnx_importable() -> None:
    try:
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                "from puripuly_heart.core.local_asr.local_sherpa_runtime import "
                "ensure_local_sherpa_windows_runtime; "
                "ensure_local_sherpa_windows_runtime(); "
                "import sherpa_onnx; import sherpa_onnx.offline_recognizer",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=_SHERPA_IMPORT_PROBE_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        pytest.skip("Japanese Parakeet Sherpa runtime is unavailable")
    if probe.returncode != 0:
        pytest.skip("Japanese Parakeet Sherpa runtime is unavailable")


@pytest.mark.asyncio
async def test_local_parakeet_sherpa_decode_f32_smoke() -> None:
    _require_sherpa_onnx_present()
    model_dir = require_local_parakeet_model_assets()
    samples, sample_rate_hz = load_required_audio_wav()
    _require_sherpa_onnx_importable()

    from puripuly_heart.providers.stt.local_parakeet_sherpa import (
        LocalParakeetJapaneseSherpaSTTBackend,
    )

    backend = LocalParakeetJapaneseSherpaSTTBackend(
        model_dir=model_dir,
        sample_rate_hz=sample_rate_hz,
    )
    try:
        transcript = await backend.decode_f32(samples)
    except Exception as exc:
        skip_if_local_sherpa_runtime_unavailable(exc)
        raise
    finally:
        await backend.close()

    _assert_non_empty_transcript_text(transcript)
