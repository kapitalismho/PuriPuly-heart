from __future__ import annotations

import asyncio
import json
import os
import secrets
import sys
import time
from pathlib import Path
from uuid import uuid4

if __package__:
    from . import bench_ovr_hmd_measurement as control
else:
    import bench_ovr_hmd_measurement as control
from puripuly_heart.config.overlay_calibration import OverlayCalibration
from puripuly_heart.core.audio.ownership import AudioSegmentIdentity
from puripuly_heart.core.overlay.bridge import OverlayBridge
from puripuly_heart.core.overlay.diagnostics import OverlayDiagnosticsRecorder
from puripuly_heart.core.overlay.presenter import OverlayPresenter
from puripuly_heart.core.overlay.process import DefaultOverlayProcessRunner, OverlayProcessManager
from puripuly_heart.core.runtime.overlay import OverlayRuntimeHandle
from puripuly_heart.core.stt.backend import (
    STTProviderTurnIdentity,
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
    STTRecognitionUnit,
    STTTextContribution,
)
from puripuly_heart.core.vad.gating import SpeechChunk, SpeechEnd, SpeechStart
from puripuly_heart.domain.events import UIEvent, UIEventType
from puripuly_heart.domain.models import Translation
from puripuly_heart.domain.recognition import RecognitionStreamIdentity, RecognitionUnitIdentity
from tests.helpers.fakes import RecordingOscQueue
from tests.helpers.translation_owners import compose_translation_test_harness


def require(condition: bool, message: str) -> None:
    if not condition:
        raise control.MeasurementError(message)


async def until(predicate, timeout: float = 5.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


class GatedProvider:
    def __init__(self, records: list[dict]) -> None:
        self.calls = []
        self.records = records

    async def translate(self, *, utterance_id, text, **kwargs):
        release = asyncio.Event()
        call = {"text": text, "release": release, "utterance_id": utterance_id}
        self.calls.append(call)
        self.records.append(
            {
                "stage": "provider_call",
                "at": time.monotonic(),
                "index": len(self.calls),
                "source_chars": len(text),
            }
        )
        await release.wait()
        self.records.append(
            {"stage": "translation_ready", "at": time.monotonic(), "index": len(self.calls)}
        )
        return Translation(utterance_id=utterance_id, text="translated:" + text)

    def release_all(self) -> None:
        for call in self.calls:
            call["release"].set()

    async def close(self) -> None:
        self.release_all()


class RecordingSink:
    def __init__(self, presenter, records):
        self.presenter = presenter
        self.records = records

    def record(self, event, receipt):
        self.records.append(
            {
                "stage": "Presenter_application",
                "at": time.monotonic(),
                "event": event.type,
                "outcome": receipt.outcome,
                "scene_revision": receipt.scene_revision,
                "cause": receipt.cause,
                "snapshot": self.presenter.snapshot().to_dict(),
            }
        )
        return receipt

    async def emit(self, event):
        return self.record(event, await self.presenter.emit(event))

    async def emit_peer_when_admissible(self, event, **kwargs):
        return self.record(event, await self.presenter.emit_peer_when_admissible(event, **kwargs))

    def active_self_overlay_metadata(self):
        return self.presenter.active_self_overlay_metadata()

    def __getattr__(self, name):
        return getattr(self.presenter, name)


class ScenarioEngine:
    def __init__(self, presenter, args, records, restart):
        self.presenter = presenter
        self.args = args
        self.records = records
        self.restart = restart
        self.provider = GatedProvider(records)
        self.sink = RecordingSink(presenter, records)
        self.harness = compose_translation_test_harness(
            stt=None,
            llm=self.provider,
            osc=RecordingOscQueue(),
            overlay_sink=self.sink,
            ui_queue_maxsize=1,
            low_latency_mode=True,
            peer_translation_enabled=True,
            source_language="en",
            target_language="ko",
        )
        self.harness.output_runtime.activate_peer_generation(1)
        self.ui_release = asyncio.Event()
        self.ui_entered = asyncio.Event()
        self.consumer = None
        self.ingress = set()
        self.sequence = 0
        self.checks = {}
        self.assembled = {}

    async def start(self):
        await self.harness.start()

        async def consume():
            while True:
                await self.harness.ui_events.get()
                self.ui_entered.set()
                await self.ui_release.wait()

        self.consumer = asyncio.create_task(consume())
        self.harness.ui_events.put_nowait(UIEvent(UIEventType.SESSION_STATE_CHANGED))
        await self.ui_entered.wait()
        self.harness.ui_events.put_nowait(UIEvent(UIEventType.SESSION_STATE_CHANGED))

    async def close(self):
        self.provider.release_all()
        self.ui_release.set()
        for task in self.ingress:
            task.cancel()
        await asyncio.gather(*self.ingress, return_exceptions=True)
        try:
            await self.harness.stop()
        finally:
            if self.consumer:
                self.consumer.cancel()
                await asyncio.gather(self.consumer, return_exceptions=True)

    async def hold(self, seconds=1.0):
        await asyncio.sleep(seconds if self.args.live else min(seconds, 0.03))

    def ready(self, kind, text):
        self.records.append(
            {
                "stage": "source_ready",
                "kind": kind,
                "at": time.monotonic(),
                "source_chars": len(text),
            }
        )

    async def independent(self, text, channel="self"):
        self.sequence += 1
        stream = RecognitionStreamIdentity(
            channel, 1, 1 if channel == "self" else 2, "synthetic", ("gemini_transcribe",)
        )
        event = STTRecognitionUnit(RecognitionUnitIdentity(stream, uuid4(), self.sequence), text)
        self.ready("independent_" + channel, text)
        await (
            self.harness.self_owner if channel == "self" else self.harness.peer_owner
        ).handle_recognition_unit(event)

    def identity(self):
        self.sequence += 1
        return STTProviderTurnIdentity(
            AudioSegmentIdentity(1, 1, uuid4(), self.sequence), "synthetic", f"turn-{self.sequence}"
        )

    async def stable(self, identity, text, seq, start=0):
        self.ready("stable", text)
        prefix = self.assembled.get(identity, "")
        require(start == len(prefix), "stable contribution must follow assembled prefix")
        self.assembled[identity] = prefix + text
        await self.harness.self_owner.handle_stt_event(
            STTProviderTurnUpdate(
                identity,
                seq,
                "stable",
                "append",
                prefix + text,
                contribution=STTTextContribution(text, start, start + len(text)),
            )
        )

    async def terminal(self, identity, text, contributions=()):
        self.ready("terminal", text)
        event = STTProviderTurnTerminal(
            identity,
            "final",
            text=text,
            text_authority="authoritative",
            included_contributions=contributions,
        )
        await self.harness.self_owner.handle_stt_event(event)
        await self.harness.self_owner.handle_stt_event(event)
        await self.harness.self_owner.handle_vad_event(SpeechEnd(identity.segment.segment_id))

    async def complete(self):
        self.provider.release_all()
        await until(lambda: self.harness.self_owner.merge_buffer is None)
        await self.harness.translation_turns.wait_for_idle()
        await self.harness.output_runtime.wait_for_peer_output_idle()

    async def source_first(self, route):
        identity = None
        text = "SELF short source"
        self.ready(route, text)
        if route == "independent":
            operation = self.independent(text)
        else:
            identity = self.identity()
            require(
                not self.presenter.snapshot().blocks,
                "final-only must not fabricate preterminal source",
            )
            operation = (
                self.stable(identity, text, 1)
                if route == "stable"
                else self.harness.self_owner.handle_stt_event(
                    STTProviderTurnTerminal(
                        identity, "final", text=text, text_authority="authoritative"
                    )
                )
            )
        task = asyncio.create_task(operation)
        self.ingress.add(task)
        await asyncio.sleep(1.0 if self.args.live else 0.2)
        progressed = bool(self.provider.calls and self.presenter.snapshot().blocks)
        expected = self.args.arm != "baseline" or route != "independent"
        require(
            progressed == expected, "UI-pressure comparison differs from pinned expected behavior"
        )
        self.checks["before_ui_release"] = {
            "source_applied": bool(self.presenter.snapshot().blocks),
            "provider_calls": len(self.provider.calls),
            "expected_progress": expected,
            "ui_full": self.harness.ui_events.full(),
        }
        if not expected:
            self.ui_release.set()
        await asyncio.wait_for(task, 5)
        await until(lambda: len(self.provider.calls) == 1)
        source = self.presenter.snapshot().blocks[0]
        require(source.primary_text == text and not source.secondary_text, "source-first layout")
        await self.hold()
        if identity:
            await self.terminal(
                identity,
                text,
                (STTTextContribution(text, 0, len(text)),) if route == "stable" else (),
            )
            await asyncio.sleep(0.45)
            require(len(self.provider.calls) == 1, "grace must not duplicate speculative work")
        self.provider.release_all()
        await until(lambda: any(b.secondary_text for b in self.presenter.snapshot().blocks))
        if identity:
            await self.complete()
        translated = self.presenter.snapshot().blocks[0]
        require(
            (translated.id, translated.occupant_key, translated.appearance_seq)
            == (source.id, source.occupant_key, source.appearance_seq),
            "source/translation identity",
        )
        require(len(self.provider.calls) == 1, "one provider call expected")
        self.checks["identity_preserved"] = True
        await self.hold()

    async def burst(self, sustained=False):
        identity = self.identity()
        pieces = ["SELF", " stable", " source", " grows"]
        if sustained:
            count = min(120, int(self.args.duration / 0.5))
            pieces = ["SELF"] + [f" word{i}" for i in range(1, count)]
        text = ""
        original = None
        episode = None
        burst_started = time.monotonic()
        burst_deadline = burst_started + self.args.duration
        used_pieces = []
        for index, piece in enumerate(pieces, 1):
            if sustained and time.monotonic() >= burst_deadline:
                break
            used_pieces.append(piece)
            await self.stable(identity, piece, index, len(text))
            text += piece
            block = self.presenter.snapshot().blocks[0]
            if original is None:
                original = block
                episode = self.presenter.snapshot().to_dict().get("native_quiet_tail_episodes")
            require(
                block.id == original.id and block.block_variant == "active_self",
                "burst must retain active identity",
            )
            require(
                self.presenter.snapshot().to_dict().get("native_quiet_tail_episodes") == episode,
                "same stream episode",
            )
            await asyncio.sleep(
                max(0, min(0.5, burst_deadline - time.monotonic()))
                if sustained
                else (0.15 if self.args.live else 0.02)
            )
        await until(lambda: bool(self.provider.calls))
        self.checks["burst_elapsed_seconds"] = time.monotonic() - burst_started
        contributions = []
        offset = 0
        for piece in used_pieces:
            contributions.append(STTTextContribution(piece, offset, offset + len(piece)))
            offset += len(piece)
        await self.terminal(identity, text, tuple(contributions))
        await asyncio.sleep(0.45)
        before = len(self.provider.calls)
        self.provider.release_all()
        await self.complete()
        self.checks.update(
            {
                "stable_updates": len(used_pieces),
                "stream_episode": episode,
                "provider_calls_before_release": before,
                "provider_calls": len(self.provider.calls),
                "native_accounting": "diagnostics only; offline cannot assert native completed-count/deadline",
            }
        )
        await self.hold()

    async def mixed(self, active):
        self.ui_release.set()
        identity = self.identity() if active else None
        if active:
            await self.stable(identity, "SELF protected source", 1)
        else:
            await self.independent("SELF finalized source")
        await until(lambda: len(self.provider.calls) == 1)
        original = self.presenter.snapshot().blocks[0]
        original_slot = 0
        for number in (1, 2):
            before = len(self.provider.calls)
            await self.independent(f"Peer-{number} source", "peer")
            await until(lambda: len(self.provider.calls) > before)
            require(
                not any(
                    f"Peer-{number}" in b.primary_text for b in self.presenter.snapshot().blocks
                ),
                "PEER translated-first",
            )
            self.provider.calls[-1]["release"].set()
            await until(
                lambda: any(
                    f"Peer-{number}" in b.primary_text for b in self.presenter.snapshot().blocks
                )
            )
            await self.harness.output_runtime.wait_for_peer_output_idle()
            await self.hold()
        visible = self.presenter.snapshot().blocks
        require(len(visible) == 2, "default shared two-slot window")
        require(
            any(b.id == original.id for b in visible) == active, "active protection/final eviction"
        )
        self.provider.calls[0]["release"].set()
        await asyncio.sleep(0.15)
        if active:
            await until(
                lambda: any(
                    b.id == original.id and b.secondary_text
                    for b in self.presenter.snapshot().blocks
                )
            )
            block = self.presenter.snapshot().blocks[original_slot]
            require(
                (block.id, block.occupant_key, block.appearance_seq)
                == (original.id, original.occupant_key, original.appearance_seq),
                "mixed identity and slot",
            )
            await self.terminal(
                identity,
                "SELF protected source",
                (STTTextContribution("SELF protected source", 0, 21),),
            )
            await self.complete()
        else:
            await self.harness.translation_turns.wait_for_idle()
            require(
                not any(b.id == original.id for b in self.presenter.snapshot().blocks),
                "evicted SELF must not resurrect",
            )
            require(
                any(r.get("outcome") == "stale" for r in self.records),
                "evicted translation should be stale",
            )
        self.checks["active_protection" if active else "finalized_eviction"] = True
        await self.hold()

    async def sticky(self):
        self.ui_release.set()
        identity = self.identity()
        await self.stable(identity, "SELF source", 1)
        await until(lambda: len(self.provider.calls) == 1)
        self.provider.release_all()
        await until(lambda: bool(self.presenter.snapshot().blocks[0].secondary_text))
        first = self.presenter.snapshot().blocks[0]
        await self.stable(identity, " extension", 2, len("SELF source"))
        await until(lambda: len(self.provider.calls) == 2)
        await until(lambda: self.presenter.snapshot().blocks[0].primary_text != first.primary_text)
        next_block = self.presenter.snapshot().blocks[0]
        require(
            next_block.secondary_text == first.secondary_text
            and next_block.primary_text != first.primary_text,
            "sticky older-prefix secondary policy",
        )
        await self.hold()
        await self.terminal(
            identity,
            "SELF source extension",
            (STTTextContribution("SELF source", 0, 11), STTTextContribution(" extension", 11, 21)),
        )
        await asyncio.sleep(0.45)
        self.provider.release_all()
        await self.complete()
        require(len(self.provider.calls) == 2, "sticky speculation reuse")
        self.checks["sticky_prefix_preserved"] = True
        await self.hold()

    async def resume(self):
        import numpy as np

        self.ui_release.set()
        first, second = self.identity(), self.identity()

        def samples():
            return np.full(512, 0.5, dtype=np.float32)

        await self.harness.self_owner.handle_vad_event(
            SpeechStart(first.segment.segment_id, pre_roll=samples(), chunk=samples())
        )
        await self.stable(first, "SELF first phrase", 1)
        await until(lambda: len(self.provider.calls) == 1)
        await self.terminal(
            first, "SELF first phrase", (STTTextContribution("SELF first phrase", 0, 17),)
        )
        buffer = self.harness.self_owner.merge_buffer
        original = self.presenter.snapshot().blocks[0]
        await self.harness.self_owner.handle_vad_event(
            SpeechStart(second.segment.segment_id, pre_roll=samples(), chunk=samples())
        )
        require(buffer.resume_pending, "actual resume pending")
        for _ in range(3):
            await self.harness.self_owner.handle_vad_event(
                SpeechChunk(second.segment.segment_id, chunk=samples())
            )
        require(buffer.resume_confirmed, "actual resume confirmed")
        await self.stable(second, " resumed phrase", 1)
        await self.terminal(
            second, " resumed phrase", (STTTextContribution(" resumed phrase", 0, 15),)
        )
        require(self.presenter.snapshot().blocks[0].id == original.id, "resume same merge caption")
        await asyncio.sleep(0.45)
        self.provider.release_all()
        await self.complete()
        self.checks.update(
            {"actual_resume": True, "grace_ms": 400, "provider_calls": len(self.provider.calls)}
        )
        await self.hold()

    async def expiry_clear_restart(self, scenario):
        self.ui_release.set()
        await self.independent("SELF lifetime source")
        await until(lambda: len(self.provider.calls) == 1)
        original = self.presenter.snapshot().blocks[0]
        if scenario == "expiry":
            await until(lambda: not self.presenter.snapshot().blocks, 10)
            self.provider.release_all()
            await self.harness.translation_turns.wait_for_idle()
            require(not self.presenter.snapshot().blocks, "expired result must not resurrect")
            require(
                any(r.get("outcome") == "stale" for r in self.records), "expired late result stale"
            )
        elif scenario == "clear_off":
            await self.presenter.clear_for_runtime_detach()
            await self.harness.self_owner.reset_provider_channel("self")
            self.provider.release_all()
            await asyncio.sleep(0.1)
            require(not self.presenter.snapshot().blocks, "clear late callback must not resurrect")
            self.checks["off_scope"] = "owned runtime close, not application overlay set off"
        else:
            before = self.presenter._entry_expiration_deadline(
                next(iter(self.presenter._entries.values()))
            )
            await self.restart()
            block = self.presenter.snapshot().blocks[0]
            after = self.presenter._entry_expiration_deadline(
                next(iter(self.presenter._entries.values()))
            )
            require(
                before == after and block.id == original.id,
                "restart must preserve original identity/deadline",
            )
            self.checks.update(
                {
                    "original_deadline": before,
                    "replayed_deadline": after,
                    "recovery_scope": "controlled owned process/runtime restart and bridge reconnect; not application auto-recovery",
                }
            )
            await until(lambda: not self.presenter.snapshot().blocks, 10)
            await self.restart()
            require(not self.presenter.snapshot().blocks, "expired replay must stay empty")
            self.provider.release_all()
            await self.harness.translation_turns.wait_for_idle()
            require(
                not self.presenter.snapshot().blocks, "restart late translation must not resurrect"
            )
        await self.hold()
        self.checks[scenario] = True

    async def run(self):
        await self.start()
        name = self.args.scenario
        if name in ("independent", "stable", "final_only"):
            await self.source_first(name)
        elif name in ("stable_burst", "sustained"):
            self.ui_release.set()
            await self.burst(name == "sustained")
        elif name in ("mixed_active", "finalized_eviction"):
            await self.mixed(name == "mixed_active")
        elif name == "sticky":
            await self.sticky()
        elif name == "resume":
            await self.resume()
        else:
            await self.expiry_clear_restart(name)


def shutdown_ok(receipt):
    return (
        receipt.get("cleanup_succeeded") is True
        and receipt.get("exit_confirmed") is True
        and receipt.get("graceful_completed") is True
        and receipt.get("acknowledged") is True
        and receipt.get("forced") is False
        and receipt.get("exit_code") == 0
        and receipt.get("terminal_cause") is None
    )


def validate_arm_imports(source):
    import puripuly_heart

    require(
        Path(puripuly_heart.__file__).resolve().is_relative_to(source / "src"),
        "source arm import separation failed",
    )
    require(
        Path(sys.modules[compose_translation_test_harness.__module__].__file__)
        .resolve()
        .is_relative_to(source),
        "test composition import separation failed",
    )
    return str(Path(puripuly_heart.__file__).resolve())


def _prepare_source(args):
    if args.live:
        require(args.confirm_hmd_ready, "live requires --confirm-hmd-ready")
        control.validate_live_guard(control.inspect_process_names(), confirmed_hmd_ready=True)
        if args.scenario == "sustained":
            control.validate_sustained_guard(args.stage, args.arm, args.anchor)
        overrides = {key: value for key, value in os.environ.items() if key.startswith("PURIPULY_")}
        require(
            all(
                (key, value)
                in (
                    ("PURIPULY_OVERLAY_QUIET_TAIL_PROFILE", "p05"),
                    ("PURIPULY_OVERLAY_HANDOFF_EXPERIMENT", "off"),
                )
                for key, value in overrides.items()
            ),
            "inherited experiment/config overrides refused",
        )
    require(
        args.arm in control.ARMS
        and args.scenario in control.SCENARIOS
        and args.anchor in control.ANCHORS,
        "invalid scenario",
    )
    require(
        1 <= args.duration <= 60 and 1 <= args.timeout <= 120,
        "finite bounded duration/timeout required",
    )
    preparation = control.load_prepared_stage(args.stage)
    source = (args.stage / "source" / args.arm).resolve()
    source_import = validate_arm_imports(source)
    return preparation, source_import


async def run_measurement(args):
    if args.live:
        require(args.confirm_hmd_ready, "live requires --confirm-hmd-ready")
    run = args.stage / "runs" / args.run_id
    require((run / "owned.json").is_file(), "run must be controller-owned")
    lock = control.LIVE_LOCK if args.live else args.stage / "active.lock"
    require(
        lock.is_file()
        and json.loads(lock.read_text(encoding="utf-8")).get("run_id") == args.run_id,
        "run-correlated controller lock required",
    )
    records = []
    shutdowns = []
    started = time.monotonic()
    outcome, failure, cleanup = "failed", None, "not_started"
    runtime, manager, bridge, engine = None, None, None, None
    manager_start_attempted = False
    transitioning = True
    started_at = control._utc_now()
    preparation = {}
    source_import = "not_validated"
    environment = {"preparation": "not_completed"}
    diagnostics, presenter = None, None
    generation = 0
    owned_receipt_failure = None

    class OwnedProcessRunner(DefaultOverlayProcessRunner):
        async def spawn(self, executable_path, manifest_path):
            nonlocal owned_receipt_failure
            process = await super().spawn(executable_path, manifest_path)
            try:
                owned = json.loads((run / "owned.json").read_text(encoding="utf-8"))
                owned.update(
                    {
                        "worker_pid": os.getpid(),
                        "native_pid": getattr(process, "pid", "unknown"),
                        "spawn_observed_at": control._utc_now(),
                        "generation": generation,
                        "native_sha256": control.SOURCE_EXE_SHA256,
                        "ownership": "managed process handle; stop never targets arbitrary PID",
                    }
                )
                control._write_json(run / "owned.json", owned)
            except Exception as exc:
                owned_receipt_failure = str(exc)
            return process

    async def start_runtime():
        nonlocal runtime, manager, bridge, transitioning, generation, manager_start_attempted
        generation += 1
        transitioning = True
        require(not (run / "stop.request").exists(), "operator_stop")
        control.load_prepared_stage(args.stage)
        runtime = OverlayRuntimeHandle(
            overlay_instance_id="hmd-" + args.run_id, shutdown_grace_s=3.0
        )
        runtime.attach_diagnostics(diagnostics)
        runtime.adopt_presenter(presenter)
        await presenter.begin_native_retry_epoch(enabled=True)
        bridge = OverlayBridge(
            session_token=secrets.token_urlsafe(16),
            initial_snapshot=presenter.snapshot(),
            overlay_instance_id=runtime.overlay_instance_id,
            runtime_generation=generation,
            diagnostics=diagnostics,
            task_factory=runtime.create_child_task,
        )
        runtime.attach_bridge(bridge)
        await bridge.start()
        presenter.attach_bridge(bridge)
        await presenter.update_calibration(OverlayCalibration(anchor=args.anchor))
        await presenter.begin_native_retry_epoch(enabled=True)
        manager = None
        manager_start_attempted = False
        if args.live:
            require(not (run / "stop.request").exists(), "operator_stop")
            control.validate_live_guard(
                control.inspect_process_names(), confirmed_hmd_ready=args.confirm_hmd_ready
            )
            runner = OwnedProcessRunner(
                executable_path=args.stage / "runtime/run/PuriPulyHeartOverlay.exe",
                task_factory=runtime.create_child_task,
                quiet_tail_profile="p05",
                handoff_experiment="off",
            )
            manager = OverlayProcessManager(
                process_runner=runner,
                bridge_url=bridge.url,
                bridge_messages=bridge.messages,
                bridge_messages_authenticated=True,
                session_token=bridge.session_token,
                overlay_instance_id=runtime.overlay_instance_id,
                log_dir=str(run / "logs"),
                diagnostics_dir=run / "diagnostics",
                diagnostics=diagnostics,
                task_factory=runtime.create_child_task,
                quiet_tail_profile="p05",
                handoff_experiment="off",
                selected_target="steamvr",
                geometry_authority="native",
                graceful_shutdown_request=bridge.broadcast_shutdown,
            )
            runtime.attach_process_manager(manager)
            require(not (run / "stop.request").exists(), "operator_stop")
            manager_start_attempted = True
            await manager.start()
            require(
                manager.state == "connected", f"native startup failed: {manager.failure_reason}"
            )
            require(
                owned_receipt_failure is None,
                f"owned process receipt failed: {owned_receipt_failure}",
            )
        transitioning = False

    async def close_runtime(preserve=False):
        nonlocal runtime
        if runtime:
            await runtime.close(preserve_presenter_state=preserve)
            if manager and manager_start_attempted:
                receipt = manager.shutdown_receipt()
                shutdowns.append(receipt)
                require(shutdown_ok(receipt), "owned native shutdown failed")
            if preserve:
                runtime.detach_preserved_presenter()
            runtime = None
        elif presenter is not None:
            await presenter.close()

    async def restart():
        nonlocal transitioning
        transitioning = True
        records.append(
            {
                "stage": "owned_runtime_restart",
                "at": time.monotonic(),
                "scope": "not application auto-recovery",
            }
        )
        await close_runtime(True)
        await start_runtime()

    async def monitor(task):
        while not task.done():
            if (run / "stop.request").exists():
                raise control.MeasurementError("operator_stop")
            if manager and not transitioning and manager.state != "connected":
                raise control.MeasurementError(
                    f"native runtime unhealthy: {manager.failure_reason or manager.state}"
                )
            await asyncio.sleep(0.1)

    async def execute():
        nonlocal engine, preparation, source_import, environment, diagnostics, presenter
        require(not (run / "stop.request").exists(), "operator_stop")
        preparation, source_import = _prepare_source(args)
        environment = await asyncio.to_thread(control.environment_inventory)
        require(not (run / "stop.request").exists(), "operator_stop")
        diagnostics = OverlayDiagnosticsRecorder(
            overlay_instance_id="hmd-" + args.run_id,
            diagnostics_dir=run / "diagnostics",
            capture_measurements=True,
        )
        presenter = OverlayPresenter(
            calibration=OverlayCalibration(anchor=args.anchor), native_retry_enabled=True
        )
        await start_runtime()
        engine = ScenarioEngine(presenter, args, records, restart)
        await engine.run()

    task, watch = None, None
    try:
        async with asyncio.timeout(args.timeout):
            task = asyncio.create_task(execute())
            watch = asyncio.create_task(monitor(task))
            done, _ = await asyncio.wait((task, watch), return_when=asyncio.FIRST_COMPLETED)
            for completed in done:
                await completed
            await task
            outcome = "pass"
    except asyncio.CancelledError:
        failure = "cancelled"
        raise
    except TimeoutError:
        failure = "run_timeout"
    except Exception as exc:
        failure = str(exc)
    finally:
        for owned in (task, watch):
            if owned:
                owned.cancel()
        await asyncio.gather(*(t for t in (task, watch) if t), return_exceptions=True)
        cleanup_failures = []
        for callback in ((engine.close,) if engine else ()) + (close_runtime,):
            try:
                await callback()
            except Exception as exc:
                cleanup_failures.append(str(exc))
        cleanup = "failed" if cleanup_failures else "complete"
        if cleanup_failures:
            outcome = "failed"
            failure = f"{failure or ''}; cleanup: {'; '.join(cleanup_failures)}"
        diagnostic_receipt = {"outcome": "not_started"}
        if diagnostics is not None:
            try:
                diagnostic_receipt = await diagnostics.dump_evidence(
                    outcome="success" if outcome == "pass" else "failure", run_id=args.run_id
                )
            except Exception as exc:
                diagnostic_receipt = {"outcome": "failed", "reason": str(exc)}
                outcome = "failed"
                failure = f"{failure or ''}; diagnostics: {exc}"
        payload = {
            "schema": control.RUN_SCHEMA,
            "run_id": args.run_id,
            "arm": args.arm,
            "scenario": args.scenario,
            "anchor": args.anchor,
            "mode": "live" if args.live else "offline_dry_run",
            "source_revision": control.ARMS.get(args.arm, "not_validated"),
            "source_import": source_import,
            "harness": preparation.get("control", "not_validated"),
            "native": preparation.get("native", "not_validated"),
            "source_tree_sha256": preparation.get("sources", {})
            .get(args.arm, {})
            .get("tree", {})
            .get("sha256", "not_validated"),
            "started_at": started_at,
            "elapsed_seconds": time.monotonic() - started,
            "requested_sustained_seconds": args.duration if args.scenario == "sustained" else None,
            "environment": {
                **environment,
                "device": args.device,
                "firmware": args.firmware,
                "connection": args.connection,
                "clock_uncertainty": "cross-process/GPU/HMD unknown",
            },
            "software": {
                "outcome": outcome,
                "failure_reason": failure,
                "cleanup": cleanup,
                "checks": engine.checks if engine else {},
                "receipts": records,
                "provider_calls": len(engine.provider.calls) if engine else 0,
                "shutdowns": shutdowns,
                "diagnostics": diagnostic_receipt,
                "native_evidence": (
                    diagnostics.evidence_summary()
                    if args.live and diagnostics is not None
                    else "not_run"
                ),
                "native_startup": (
                    "attempted" if manager_start_attempted or shutdowns else "not_started"
                ),
                "scope": "real developer translation/Presenter/projection/bridge owners; native only in live",
            },
            "physical_hmd": {"result": "not_observed", "api_success_is_not_physical_pass": True},
        }
        control._write_json(run / "report.json", payload)
        (run / "report.sha256").write_text(control._sha256(run / "report.json"), encoding="ascii")
    if outcome != "pass":
        raise control.MeasurementError(failure or "run failed")
    return run / "report.json"
