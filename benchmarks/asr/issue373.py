from __future__ import annotations

import argparse
import asyncio
import dataclasses
import enum
import hashlib
import importlib
import json
import os
import sys
import time
import wave
from pathlib import Path
from uuid import UUID

import numpy as np


def encode(value):
    if dataclasses.is_dataclass(value):
        return {
            "$type": f"{type(value).__module__}:{type(value).__qualname__}",
            "fields": {
                f.name: encode(getattr(value, f.name))
                for f in dataclasses.fields(value)
                if f.init and not callable(getattr(value, f.name))
            },
        }
    if isinstance(value, UUID):
        return {"$uuid": str(value)}
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, tuple):
        return {"$tuple": [encode(item) for item in value]}
    if isinstance(value, list):
        return [encode(item) for item in value]
    if isinstance(value, dict):
        return {str(key): encode(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return {"samples": int(value.size)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def decode(value):
    if isinstance(value, list):
        return [decode(item) for item in value]
    if not isinstance(value, dict):
        return value
    if "$uuid" in value:
        return UUID(value["$uuid"])
    if "$tuple" in value:
        return tuple(decode(item) for item in value["$tuple"])
    if "$type" in value:
        module, name = value["$type"].split(":")
        cls = getattr(importlib.import_module(module), name)
        fields = {f.name for f in dataclasses.fields(cls) if f.init}
        return cls(**{key: decode(item) for key, item in value["fields"].items() if key in fields})
    return {key: decode(item) for key, item in value.items()}


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def credentials(path):
    from puripuly_heart.app.wiring.wiring_secrets_factory import require_secret
    from puripuly_heart.core.storage.secrets import InMemorySecretStore

    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        key, sep, value = line.strip().partition("=")
        key = key.removeprefix("export ").strip()
        if sep and key in ("SONIOX_API_KEY", "OPENROUTER_API_KEY"):
            os.environ[key] = value.strip().strip("\"'")
    return lambda name: require_secret(InMemorySecretStore(), key=name.lower(), env_var=name)


async def capture(args):
    from puripuly_heart.core.audio.format import AudioCaptureSpan, AudioFrameF32
    from puripuly_heart.core.clock import SystemClock
    from puripuly_heart.core.peer_capture import PeerCaptureLanguageFacts
    from puripuly_heart.core.runtime.audio_vad_loop import run_audio_vad_loop
    from puripuly_heart.core.stt.backend import (
        STTProviderTurnTerminal,
        STTProviderTurnUpdate,
        STTSessionProjection,
    )
    from puripuly_heart.core.stt.scoped_engine import ScopedRecognitionEngine
    from puripuly_heart.core.vad.gating import create_peer_vad_gating
    from puripuly_heart.core.vad.silero import SileroVadOnnx
    from puripuly_heart.providers.stt.soniox import SonioxRealtimeSTTBackend, _SonioxSession
    from tests.core.runtime.test_peer_capture_session import FakeProvider, make_config, make_owner

    get_secret = credentials(args.env)
    start = time.monotonic()
    rows = []

    def record(kind, **values):
        rows.append(
            {
                "t": time.monotonic() - start,
                "kind": kind,
                **{key: encode(value) for key, value in values.items()},
            }
        )

    original = _SonioxSession._handle_message

    async def message(session, payload):
        record("native", session=session.speaker_session_scope, payload=json.loads(payload))
        return await original(session, payload)

    _SonioxSession._handle_message = message
    original_audio = _SonioxSession.send_turn_audio

    async def audio(session, identity, pcm16le, **kwargs):
        record("write", identity=identity, bytes=len(pcm16le), **kwargs)
        return await original_audio(session, identity, pcm16le, **kwargs)

    _SonioxSession.send_turn_audio = audio
    original_seal = _SonioxSession.seal_turn

    async def seal(session, identity, **kwargs):
        record("seal", identity=identity, **kwargs)
        return await original_seal(session, identity, **kwargs)

    _SonioxSession.seal_turn = seal
    backend = SonioxRealtimeSTTBackend(
        api_key=get_secret("SONIOX_API_KEY"), language_hints=[args.language]
    )

    async def factory(settings, epoch):
        session = await backend.open_session(projection=STTSessionProjection("scoped", epoch))
        record("session", scope=session.speaker_session_scope, epoch=epoch)
        return session

    engine = ScopedRecognitionEngine(session_factory=factory, channel="peer")
    done = asyncio.Event()

    class Source:
        terminal_reason = None

        async def frames(self):
            with wave.open(str(args.audio), "rb") as wav:
                assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
                origin = time.monotonic()
                offset = 0
                sequence = 0
                while payload := wav.readframes(512):
                    samples = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768
                    end = offset + len(samples)
                    await asyncio.sleep(max(0, origin + end / 16000 - time.monotonic()))
                    yield AudioFrameF32(
                        samples=samples,
                        sample_rate_hz=16000,
                        capture=AudioCaptureSpan(
                            capture_epoch=0,
                            callback_sequence=sequence,
                            source_sample_rate_hz=16000,
                            source_start_sample=offset,
                            source_end_sample=end,
                            source_start_monotonic_s=origin + offset / 16000,
                            source_end_monotonic_s=origin + end / 16000,
                        ),
                    )
                    offset = end
                    sequence += 1
                record("input_complete", frames=offset, duration_s=offset / 16000)

        async def close(self):
            done.set()

    class Provider(FakeProvider):
        async def release(self, **kwargs):
            record("release", **kwargs)
            await engine.close()
            await super().release(**kwargs)

    async def loop(**kwargs):
        await run_audio_vad_loop(**kwargs, channel_label="peer")

    owner, *_ = make_owner(
        provider=Provider(),
        source_factory=lambda *_: Source(),
        vad_factory=lambda cfg: create_peer_vad_gating(
            SileroVadOnnx(Path(args.root) / "src/puripuly_heart/data/vad/silero_vad.onnx"),
            sample_rate_hz=16000,
            ring_buffer_ms=cfg.vad_pre_roll_ms,
            hangover_ms=cfg.vad_hangover_ms,
        ),
        run_audio_loop=loop,
        clock=SystemClock(),
        sink=engine,
    )

    async def event_sink(event):
        if isinstance(event, STTProviderTurnTerminal):
            record("provider_terminal", event=event)
            for receipt, terminal in owner.admit_provider_terminal(event):
                record("final", receipt=receipt, event=terminal)
        elif isinstance(event, STTProviderTurnUpdate):
            record("update", event=event)
        else:
            record("event", event=event)

    engine.bind_event_sink(event_sink)
    cfg = dataclasses.replace(
        make_config(
            language=PeerCaptureLanguageFacts(source_mode="manual", source_language=args.language)
        ),
        vad_speech_threshold=0.4,
    )
    record(
        "config",
        model=backend.model,
        language_hints=list(backend.language_hints),
        diarization=backend.enable_speaker_diarization,
        language_identification=backend.enable_language_identification,
        strict=backend.language_hints_strict,
        endpoint_detection=False,
        context_terms=[],
        sample_rate=16000,
        channels=1,
        trailing_silence_ms=backend.trailing_silence_ms,
        config=cfg,
        audio_sha256=hashlib.sha256(Path(args.audio).read_bytes()).hexdigest(),
    )
    try:
        await owner.apply_intent(cfg, enabled=True)
        await asyncio.wait_for(done.wait(), timeout=160)
        await asyncio.sleep(0.2)
        record("segments", ledgers=[ledger.snapshots for ledger in owner.segment_ledgers])
        record("completed", state=owner.snapshot)
    finally:
        await owner.close()
        await engine.close()
        save(args.output, {"language": args.language, "rows": rows})
    print(
        json.dumps(
            {
                "language": args.language,
                "rows": len(rows),
                "finals": sum(row["kind"] == "final" for row in rows),
                "output": args.output,
            }
        )
    )


async def replay(args):
    from types import SimpleNamespace

    from puripuly_heart.config.prompts import get_default_prompt
    from puripuly_heart.core.clock import FakeClock
    from puripuly_heart.core.orchestrator.peer_caption import PeerCaptionOwner
    from puripuly_heart.core.translation_backend import TranslationBackend
    from puripuly_heart.domain.models import Translation
    from tests.helpers.fakes import RecordingOscQueue
    from tests.helpers.translation_owners import compose_translation_test_harness

    trace = json.loads(Path(args.trace).read_text(encoding="utf-8"))
    first_update = next(row for row in trace["rows"] if row["kind"] == "update")
    origin = decode(first_update["event"]).observed_at - first_update["t"]
    clock = FakeClock(origin)
    events, requests, captions, groups = [], [], [], []

    class Backend(TranslationBackend):
        async def translate(self, request):
            requests.append({"t": clock.now(), "request": encode(request)})
            return Translation(utterance_id=request.utterance_id, text="번역 " + request.text)

        async def close(self):
            pass

    async def emit(event):
        events.append({"t": clock.now(), "event": encode(event)})

    original_prepare = PeerCaptionOwner._prepare

    def prepare(owner, caption, source, kind, config):
        captions.append(
            {"t": clock.now(), "kind": kind, "source": source, "caption": encode(caption)}
        )
        return original_prepare(owner, caption, source, kind, config)

    PeerCaptionOwner._prepare = prepare
    original_fragment = PeerCaptionOwner._final_fragment

    async def fragment(owner, transcript, assignment, **kwargs):
        previous = owner._last.utterance_id if owner._last is not None else None
        await original_fragment(owner, transcript, assignment, **kwargs)
        groups.append(
            {
                "t": clock.now(),
                "transcript": encode(transcript),
                "assignment": encode(assignment),
                "previous": encode(previous),
                "accepted_caption": encode(owner._last),
                "kind": captions[-1]["kind"],
            }
        )

    PeerCaptionOwner._final_fragment = fragment
    harness = compose_translation_test_harness(
        stt=None,
        llm=Backend(),
        osc=RecordingOscQueue(),
        clock=clock,
        source_language=trace["language"],
        target_language="ko",
        system_prompt=get_default_prompt(),
        peer_translation_enabled=True,
        overlay_sink=SimpleNamespace(emit=emit, active_self_overlay_metadata=lambda: None),
    )

    async def idle():
        await harness.peer_owner.cloud_captions.wait_for_idle()
        await harness.output_runtime.wait_for_peer_output_idle()
        for _ in range(12):
            await asyncio.sleep(0)

    await harness.start()
    try:
        for row in trace["rows"]:
            if row["kind"] not in ("update", "final"):
                continue
            target = origin + row["t"]
            while True:
                due = [
                    segment.next_check_at
                    for segment in harness.peer_owner.cloud_captions._segments.values()
                    if not segment.closed and segment.next_check_at <= target
                ]
                if not due:
                    break
                clock.advance(max(0, min(due) - clock.now()))
                await harness.peer_owner.cloud_captions.check_previews()
                await idle()
            clock.advance(max(0, target - clock.now()))
            event = decode(row["event"])
            if row["kind"] == "update":
                await harness.peer_owner.handle_provider_update(event)
            else:
                await harness.peer_owner.handle_provider_turn_terminal(
                    decode(row["receipt"]), event
                )
            await idle()
        save(
            args.output,
            {
                "arm": args.arm,
                "root": str(Path(args.root).resolve()),
                "language": trace["language"],
                "requests": requests,
                "captions": captions,
                "groups": groups,
                "events": events,
                "counts": harness.peer_owner.cloud_captions.request_counts,
                "clock_origin_estimate": origin,
                "clock_origin_note": "First provider observed_at minus actual sink arrival offset; sub-millisecond sink delay is shared by both arms.",
                "history": encode(harness.peer_runtime.translation_history),
            },
        )
    finally:
        await harness.stop()
    print(
        json.dumps(
            {
                "arm": args.arm,
                "language": trace["language"],
                "requests": len(requests),
                "captions": len(captions),
                "events": len(events),
            }
        )
    )


async def present(args):
    from puripuly_heart.config.overlay_calibration import OverlayCalibration
    from puripuly_heart.core.clock import FakeClock
    from puripuly_heart.core.overlay.presenter import OverlayPresenter

    arm = json.loads(Path(args.trace).read_text(encoding="utf-8"))
    clock = FakeClock(arm["clock_origin_estimate"])
    presenter = OverlayPresenter(calibration=OverlayCalibration(), clock=clock)
    records = []
    try:
        for row in arm["events"]:
            clock.advance(max(0, row["t"] - clock.now()))
            event = decode(row["event"])
            receipt = await presenter.emit(event)
            entry = presenter._entries.get(("peer", event.utterance_id))
            records.append(
                {
                    "t": clock.now(),
                    "utterance_id": str(event.utterance_id),
                    "revision": event.peer_caption_revision,
                    "is_final": event.is_final,
                    "receipt": encode(receipt),
                    "entry": encode(entry),
                    "snapshot": encode(presenter.snapshot()),
                }
            )
        save(args.output, {"arm": arm["arm"], "language": arm["language"], "records": records})
    finally:
        await presenter.close()
    print(
        json.dumps(
            {"arm": arm["arm"], "language": arm["language"], "presentation_records": len(records)}
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("capture", "replay", "present"))
    parser.add_argument("--root", default=".")
    parser.add_argument("--env")
    parser.add_argument("--language")
    parser.add_argument("--audio")
    parser.add_argument("--trace")
    parser.add_argument("--arm")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    sys.path[:0] = [str(root / "src"), str(root)]
    asyncio.run({"capture": capture, "replay": replay, "present": present}[args.mode](args))


if __name__ == "__main__":
    main()
