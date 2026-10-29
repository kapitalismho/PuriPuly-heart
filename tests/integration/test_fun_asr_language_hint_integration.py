from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import resample_poly

from puripuly_heart.app.adapters.asr_worker_process import DefaultASRWorkerProcessFactory
from puripuly_heart.core.local_asr.local_stt_assets import FUN_ASR_MLT_NANO_MODEL_ID
from puripuly_heart.providers.stt.local_cpu import create_local_cpu_backend
from tests.integration.helpers import integration_mark, load_audio_wav, require_env

pytestmark = integration_mark()


@pytest.mark.asyncio
async def test_korean_hint_keeps_ambiguous_korean_speech_in_korean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = Path(require_env("ASR_WORKER_PATH"))
    model_root = Path(require_env("LOCAL_ASR_MODEL_ROOT"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "roaming"))
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "asr" / "korean-eung.wav"
    samples, rate = load_audio_wav(fixture)
    factor = math.gcd(rate, 16000)
    samples = resample_poly(samples, 16000 // factor, rate // factor).astype(np.float32)
    backend = create_local_cpu_backend(
        FUN_ASR_MLT_NANO_MODEL_ID,
        model_root=model_root,
        source_language="ko-KR",
        sample_rate_hz=16000,
        stream_label="self",
        process_factory=DefaultASRWorkerProcessFactory(executable_path=worker),
    )
    try:
        transcript = await backend.decode_f32(samples)
    finally:
        await backend.close()

    assert transcript.strip().rstrip(".!?。！？") == "응"
