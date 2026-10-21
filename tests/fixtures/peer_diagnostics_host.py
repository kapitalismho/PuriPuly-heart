from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from puripuly_heart.app.adapters.peer_capture.peer_capture_source import PeerCaptureSourceAdapter
from puripuly_heart.app.adapters.peer_capture.peer_capture_vad import PeerCaptureVadAdapter
from puripuly_heart.app.services.peer_application import PeerApplicationOwner
from puripuly_heart.app.wiring.wiring_local_asr_provider_runtime import SharedSTTProviderFactory
from puripuly_heart.config.resolved import vad_exit_threshold
from puripuly_heart.core.runtime.peer_channel import PeerCaptureSessionOwner
from puripuly_heart.core.vad.gating import VadGating
from puripuly_heart.main import main as application_main
from tests.fixtures.speech_diagnostics_host import GeneratedAudioSource
from tests.helpers.vad import SequenceVadEngine


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument(
        "--failure-stage", choices=("provider_build", "provider_warmup"), required=True
    )
    parser.add_argument("--failure-gate", type=Path, required=True)
    args = parser.parse_args()

    def source(_adapter, config, _target):
        return GeneratedAudioSource(config.target_sample_rate_hz)

    def vad(_adapter, config):
        return VadGating(
            engine=SequenceVadEngine(([0.99] * 24 + [0.0] * 46) * 100),
            sample_rate_hz=config.target_sample_rate_hz,
            speech_threshold=config.vad_speech_threshold,
            continuation_threshold=vad_exit_threshold(config.vad_speech_threshold),
            ring_buffer_ms=500,
            hangover_ms=config.vad_hangover_ms,
        )

    initialize_application = PeerApplicationOwner.__init__

    def initialize_peer_application(self, *arguments, **kwargs):
        initialize_application(self, *arguments, **kwargs)
        state_provider = self.state_provider
        self.state_provider = lambda: replace(state_provider(), overlay_state="connected")

    create_provider = SharedSTTProviderFactory.create

    async def create(self, request, **kwargs):
        armed = args.failure_gate.exists() and request.channel == "peer"
        if armed and args.failure_stage == "provider_build":
            raise ConnectionError("PRIVATE_PEER_ADMISSION_CREATION")
        provider = await create_provider(self, request, **kwargs)
        if armed:

            async def fail_warmup(**identity):
                raise TimeoutError("PRIVATE_PEER_ADMISSION_WARMUP")

            provider.backend_prepare = fail_warmup
        return provider

    initialize_capture = PeerCaptureSessionOwner.__init__

    def initialize_peer_capture(self, *arguments, **kwargs):
        initialize_capture(self, *arguments, **kwargs)
        request_factory = self._provider_request_factory

        def request(config, warmup):
            native = request_factory(config, warmup)
            if args.failure_gate.exists() and args.failure_stage == "provider_warmup":
                native = replace(native, warmup=True)
            return native

        self._provider_request_factory = request

    PeerCaptureSourceAdapter.__call__ = source
    PeerCaptureVadAdapter.__call__ = vad
    PeerApplicationOwner.__init__ = initialize_peer_application
    SharedSTTProviderFactory.create = create
    PeerCaptureSessionOwner.__init__ = initialize_peer_capture
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
