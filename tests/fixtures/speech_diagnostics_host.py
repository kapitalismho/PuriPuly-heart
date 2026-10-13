from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from puripuly_heart.app.adapters.self_capture.self_capture_source import SelfCaptureSourceAdapter
from puripuly_heart.app.adapters.self_capture.self_capture_vad import SelfCaptureVadAdapter
from puripuly_heart.config.resolved import vad_exit_threshold
from puripuly_heart.core.audio.format import AudioCaptureSpan, AudioFrameF32
from puripuly_heart.core.vad.gating import VadGating
from puripuly_heart.main import main as application_main
from tests.helpers.vad import SequenceVadEngine


class GeneratedAudioSource:
    def __init__(self, sample_rate_hz: int) -> None:
        self.sample_rate_hz = sample_rate_hz
        self.closed = False

    async def frames(self):
        frame_size = self.sample_rate_hz * 32 // 1000
        start = asyncio.get_running_loop().time()
        index = 0
        while not self.closed:
            await asyncio.sleep(0.032)
            left, right = index * frame_size, (index + 1) * frame_size
            speech = index % 70 < 24
            span = AudioCaptureSpan(
                capture_epoch=1,
                callback_sequence=index,
                source_sample_rate_hz=self.sample_rate_hz,
                source_start_sample=left,
                source_end_sample=right,
                source_start_monotonic_s=start + left / self.sample_rate_hz,
                source_end_monotonic_s=start + right / self.sample_rate_hz,
            )
            yield AudioFrameF32(
                np.full(frame_size, 0.1 if speech else 0.0, dtype=np.float32),
                self.sample_rate_hz,
                capture=span,
            )
            index += 1

    async def close(self) -> None:
        self.closed = True


def generated_source(_adapter, config):
    return GeneratedAudioSource(config.target_sample_rate_hz)


def generated_vad(_adapter, config):
    return VadGating(
        engine=SequenceVadEngine(([0.99] * 24 + [0.0] * 46) * 100),
        sample_rate_hz=config.target_sample_rate_hz,
        speech_threshold=config.vad_speech_threshold,
        continuation_threshold=vad_exit_threshold(config.vad_speech_threshold),
        ring_buffer_ms=config.ring_buffer_ms,
        hangover_ms=config.vad_hangover_ms,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--failure-stage", choices=("provider_build", "provider_warmup"))
    parser.add_argument("--failure-gate", type=Path)
    args = parser.parse_args()
    SelfCaptureSourceAdapter.__call__ = generated_source
    SelfCaptureVadAdapter.__call__ = generated_vad
    if args.failure_stage is not None:
        from puripuly_heart.app.wiring.wiring_local_asr_provider_runtime import (
            SharedSTTProviderFactory,
        )
        from puripuly_heart.core.runtime.self_capture import SelfCaptureSessionOwner

        create_provider = SharedSTTProviderFactory.create
        create_owner = SelfCaptureSessionOwner.__init__

        def armed():
            return args.failure_gate is not None and args.failure_gate.exists()

        async def create(self, request, **kwargs):
            if armed() and args.failure_stage == "provider_build":
                raise ConnectionError("PRIVATE_ADMISSION_CREATION")
            provider = await create_provider(self, request, **kwargs)
            if armed():

                async def fail_warmup(**identity):
                    raise TimeoutError("PRIVATE_ADMISSION_WARMUP")

                provider.backend_prepare = fail_warmup
            return provider

        def initialize(self, *arguments, **kwargs):
            create_owner(self, *arguments, **kwargs)
            request_factory = self._provider_request_factory

            def request(config, warmup):
                native = request_factory(config, warmup)
                if armed() and args.failure_stage == "provider_warmup":
                    native = replace(native, warmup=True)
                return native

            self._provider_request_factory = request

        SharedSTTProviderFactory.create = create
        SelfCaptureSessionOwner.__init__ = initialize
    return application_main(
        [
            "run-headless",
            "--config",
            str(args.config),
            "--ready-file",
            str(args.ready_file),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
