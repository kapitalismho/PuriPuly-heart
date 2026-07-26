from __future__ import annotations

import asyncio
import contextlib
import json
import os
import secrets
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

import websockets

from puripuly_heart.core.desktop_overlay_repro_artifacts import (
    ReproArtifactError,
    validate_repro_run_records,
    write_repro_artifacts,
)
from puripuly_heart.core.diagnostic_validation import (
    DESKTOP_OVERLAY_REPRO_SCHEMA_VERSION,
    validate_desktop_overlay_repro_record,
    validate_desktop_overlay_repro_result,
)
from puripuly_heart.core.overlay.protocol import (
    OverlayPresentationBlock,
    OverlayPresentationSnapshot,
)

DEFAULT_CYCLES = 100
DEFAULT_DWELL_MS = 150
MIN_CYCLES = 1
MAX_CYCLES = 1000
MIN_DWELL_MS = 1
MAX_DWELL_MS = 10000
_RENDER_ACK_TIMEOUT_S = 5.0


class ReproValidationError(Exception):
    pass


class ReproRunError(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class ReproArguments:
    cycles: int = DEFAULT_CYCLES
    dwell_ms: int = DEFAULT_DWELL_MS
    output_dir: Path = Path()


@dataclass(frozen=True, slots=True)
class ScheduledRevision:
    revision: int
    expected_disposition: str
    snapshot: OverlayPresentationSnapshot


@dataclass(frozen=True, slots=True)
class ScheduledBatch:
    name: str
    revisions: tuple[ScheduledRevision, ...]


class ReproBackdrop(Protocol):
    async def start(self) -> None: ...
    async def close(self) -> None: ...


class _NeverExitParentMonitor:
    async def wait_for_parent_exit(self, stop_event: asyncio.Event) -> None:
        await stop_event.wait()


@dataclass(slots=True)
class StaticCheckerboardBackdrop:
    _page: Any | None = None
    _task: asyncio.Task[None] | None = None
    _ready: asyncio.Event = field(default_factory=asyncio.Event)
    _closed: bool = False

    async def start(self) -> None:
        import flet as ft

        async def target(page: Any) -> None:
            self._page = page
            page.window.frameless = True
            page.window.always_on_top = False
            page.window.width = 1344
            page.window.height = 336
            page.window.visible = True
            tiles = [
                ft.Container(
                    width=168,
                    height=84,
                    bgcolor="#135BFF" if (row + column) % 2 == 0 else "#E000A8",
                )
                for row in range(4)
                for column in range(8)
            ]
            page.add(ft.GridView(controls=tiles, runs_count=8, spacing=0, run_spacing=0))
            self._ready.set()

        self._task = asyncio.create_task(ft.run_async(main=target))
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=_RENDER_ACK_TIMEOUT_S)
        except TimeoutError as exc:
            raise ReproRunError("startup_failed") from exc

    async def close(self) -> None:
        self._closed = True
        if self._page is not None:
            with contextlib.suppress(Exception):
                self._page.window.close()
        await _finish_owned_task(self._task, cancel=True)


@dataclass(slots=True)
class LocalAuthenticatedRawIngress:
    token: str
    initial_snapshot: OverlayPresentationSnapshot
    _server: Any | None = None
    _socket: Any | None = None
    _connected: asyncio.Event = field(default_factory=asyncio.Event)
    _closed: bool = False

    async def start(self) -> str:
        self._server = await websockets.serve(self._handle, "127.0.0.1", 0, ping_interval=None)
        port = self._server.sockets[0].getsockname()[1]
        return f"ws://127.0.0.1:{port}"

    async def _handle(self, websocket: Any) -> None:
        try:
            raw = await asyncio.wait_for(websocket.recv(), timeout=_RENDER_ACK_TIMEOUT_S)
            message = json.loads(raw)
            if not isinstance(message, dict) or message != {
                "type": "auth",
                "session_token": self.token,
            }:
                await websocket.send(json.dumps({"type": "auth_error"}))
                return
            self._socket = websocket
            await websocket.send(_serialized_snapshot_envelope(self.initial_snapshot))
            self._connected.set()
            await websocket.wait_closed()
        finally:
            if self._socket is websocket:
                self._socket = None

    async def wait_connected(self) -> None:
        try:
            await asyncio.wait_for(self._connected.wait(), timeout=_RENDER_ACK_TIMEOUT_S)
        except TimeoutError as exc:
            raise ReproRunError("bridge_failed") from exc

    async def send_runtime_control(self, payload: Mapping[str, object]) -> None:
        await self.wait_connected()
        if self._socket is None:
            raise ReproRunError("bridge_failed")
        await self._socket.send(json.dumps({"type": "runtime_control", "payload": dict(payload)}))

    async def write_batch(self, snapshots: Sequence[OverlayPresentationSnapshot]) -> None:
        await self.wait_connected()
        if self._socket is None:
            raise ReproRunError("bridge_failed")
        for snapshot in snapshots:
            await self._socket.send(_serialized_snapshot_envelope(snapshot))

    async def close(self) -> None:
        self._closed = True
        if self._socket is not None:
            with contextlib.suppress(Exception):
                await self._socket.send(json.dumps({"type": "shutdown"}))
            with contextlib.suppress(Exception):
                await self._socket.close()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()


def _serialized_snapshot_envelope(snapshot: OverlayPresentationSnapshot) -> str:
    return json.dumps({"type": "snapshot", "payload": snapshot.to_dict()})


@dataclass(slots=True)
class DesktopOverlayReproOwner:
    arguments: ReproArguments
    backdrop_factory: Callable[[], ReproBackdrop] = StaticCheckerboardBackdrop
    renderer_factory: Callable[..., Any] | None = None
    ingress_factory: Callable[[str, OverlayPresentationSnapshot], LocalAuthenticatedRawIngress] = (
        lambda token, snapshot: LocalAuthenticatedRawIngress(token, snapshot)
    )
    _records: list[Mapping[str, object]] = field(default_factory=list)
    _expected: dict[int, tuple[int, str]] = field(default_factory=dict)
    _acknowledged: set[int] = field(default_factory=set)
    _completion: dict[int, asyncio.Future[None]] = field(default_factory=dict)
    _renderer_shutdown_completed: bool = False
    _bridge_shutdown_completed: bool = False
    _backdrop_shutdown_completed: bool = False
    _renderer_task: asyncio.Task[int] | None = None
    _consumer_task: asyncio.Task[None] | None = None
    _monotonic_origin: float = 0.0

    async def run(self) -> int:
        from puripuly_heart.core.overlay.manifest import (
            OVERLAY_CONTRACT_VERSION,
            OverlayLaunchManifest,
        )
        from puripuly_heart.ui.desktop_overlay import (
            DesktopOverlayRenderer,
            DiagnosticIngressGate,
            DiagnosticLocalRendererPort,
        )

        self._monotonic_origin = time.monotonic()
        token = secrets.token_urlsafe(32)
        initial = OverlayPresentationSnapshot(revision=0, blocks=[])
        ingress = self.ingress_factory(token, initial)
        backdrop = self.backdrop_factory()
        port = DiagnosticLocalRendererPort(acknowledgement_timeout_s=_RENDER_ACK_TIMEOUT_S)
        gate = DiagnosticIngressGate()
        renderer: Any | None = None
        reason: str | None = None
        cancelled = False
        try:
            try:
                bridge_url = await ingress.start()
            except Exception as exc:
                raise ReproRunError("bridge_failed") from exc
            try:
                await backdrop.start()
            except Exception as exc:
                raise ReproRunError("startup_failed") from exc
            manifest = OverlayLaunchManifest(
                contract_version=OVERLAY_CONTRACT_VERSION,
                app_version="desktop-overlay-repro",
                overlay_instance_id="desktop-overlay-repro",
                bridge_url=bridge_url,
                session_token=token,
                parent_pid=os.getpid(),
                startup_deadline_ms=5000,
                log_dir="diagnostic",
                log_level="WARNING",
                locale="en",
            )
            lifecycle_ready = asyncio.Event()

            async def lifecycle_sink(event: dict[str, object]) -> None:
                if event.get("type") == "overlay_ready":
                    lifecycle_ready.set()

            factory = self.renderer_factory or DesktopOverlayRenderer
            renderer = factory(
                manifest,
                lifecycle_sink=type("Lifecycle", (), {"emit": staticmethod(lifecycle_sink)})(),
                parent_monitor=_NeverExitParentMonitor(),
                diagnostic_port=port,
                diagnostic_ingress_gate=gate,
            )
            self._consumer_task = asyncio.create_task(self._consume_renderer_events(port))
            self._renderer_task = asyncio.create_task(renderer.run())
            await ingress.wait_connected()
            try:
                await asyncio.wait_for(lifecycle_ready.wait(), timeout=_RENDER_ACK_TIMEOUT_S)
            except TimeoutError as exc:
                raise ReproRunError("startup_failed") from exc
            await ingress.send_runtime_control(
                {"command": "set_interaction_mode", "mode": "pass_through"}
            )
            await _wait_for_locked_interaction_mode(renderer.window)
            await ingress.send_runtime_control(
                {
                    "command": "apply_window_bounds",
                    "x": 0,
                    "y": 0,
                    "width": 1344,
                    "height": 336,
                }
            )
            await asyncio.sleep(0)
            for cycle in range(1, self.arguments.cycles + 1):
                for batch in normative_repro_schedule(cycle):
                    _raise_owned_task_failure(self._consumer_task)
                    _raise_owned_task_failure(self._renderer_task)
                    await self._run_batch(ingress, gate, batch)
                    await asyncio.sleep(self.arguments.dwell_ms / 1000.0)
            try:
                await ingress.close()
                self._bridge_shutdown_completed = True
            except Exception as exc:
                raise ReproRunError("bridge_failed") from exc
            if self._renderer_task is None:
                raise ReproRunError("render_failed")
            exit_code = await asyncio.wait_for(self._renderer_task, timeout=_RENDER_ACK_TIMEOUT_S)
            self._renderer_shutdown_completed = bool(renderer.is_shutdown)
            if exit_code != 0:
                raise ReproRunError("render_failed")
            await _finish_owned_task(self._consumer_task, cancel=True)
            self._consumer_task = None
        except ReproRunError as exc:
            reason = exc.reason
        except TimeoutError:
            reason = "render_timeout"
        except asyncio.CancelledError:
            reason = "cleanup_failed"
            cancelled = True
        except Exception:
            reason = "render_failed"
        finally:
            cleanup_reason = await self._cleanup_resources(ingress, renderer, backdrop)
            reason = reason or cleanup_reason
        completed_cycles = self._completed_cycles()
        if reason is None:
            try:
                validate_repro_run_records(self._records, cycles=self.arguments.cycles)
            except ReproArtifactError:
                reason = "validation_failed"
        result = _result_record(
            outcome="completed" if reason is None else "failed",
            reason=reason,
            cycles_requested=self.arguments.cycles,
            cycles_completed=completed_cycles,
            records=self._records,
            renderer_shutdown_completed=self._renderer_shutdown_completed,
            bridge_shutdown_completed=self._bridge_shutdown_completed,
            backdrop_shutdown_completed=self._backdrop_shutdown_completed,
        )
        try:
            write_repro_artifacts(self.arguments.output_dir, self._records, result)
        except ReproArtifactError:
            return 1
        if cancelled:
            return 1
        return 0 if reason is None else 1

    async def _cleanup_resources(
        self,
        ingress: LocalAuthenticatedRawIngress,
        renderer: Any | None,
        backdrop: ReproBackdrop,
    ) -> str | None:
        reason: str | None = None
        try:
            await ingress.close()
            self._bridge_shutdown_completed = True
        except Exception:
            reason = "bridge_failed"
        if renderer is not None:
            try:
                await renderer.shutdown()
                self._renderer_shutdown_completed = bool(renderer.is_shutdown)
            except Exception:
                reason = reason or "cleanup_failed"
        for task in (self._renderer_task, self._consumer_task):
            try:
                await _finish_owned_task(task, cancel=True)
            except Exception:
                reason = reason or "cleanup_failed"
        try:
            await backdrop.close()
            self._backdrop_shutdown_completed = True
        except Exception:
            reason = reason or "cleanup_failed"
        return reason

    async def _run_batch(
        self,
        ingress: LocalAuthenticatedRawIngress,
        gate: Any,
        batch: ScheduledBatch,
    ) -> None:
        loop = asyncio.get_running_loop()
        for item in batch.revisions:
            if item.revision in self._expected:
                raise ReproRunError("validation_failed")
            self._expected[item.revision] = (
                self._cycle_for_revision(item.revision),
                item.expected_disposition,
            )
            self._completion[item.revision] = loop.create_future()
        revisions = tuple(item.revision for item in batch.revisions)
        await gate.hold(revisions)
        await ingress.write_batch([item.snapshot for item in batch.revisions])
        try:
            await gate.wait_until_queued(_RENDER_ACK_TIMEOUT_S)
        except Exception as exc:
            raise ReproRunError("bridge_failed") from exc
        gate.release()
        try:
            await self._wait_for_batch_completion(batch)
        except TimeoutError as exc:
            raise ReproRunError("render_timeout") from exc

    async def _wait_for_batch_completion(self, batch: ScheduledBatch) -> None:
        pending = {self._completion[item.revision] for item in batch.revisions}
        deadline = asyncio.get_running_loop().time() + _RENDER_ACK_TIMEOUT_S
        while pending:
            waiters: set[asyncio.Future[Any]] = set(pending)
            if self._consumer_task is not None:
                waiters.add(self._consumer_task)
            if self._renderer_task is not None:
                waiters.add(self._renderer_task)
            timeout_s = deadline - asyncio.get_running_loop().time()
            if timeout_s <= 0:
                raise TimeoutError
            done, _ = await asyncio.wait(
                waiters, timeout=timeout_s, return_when=asyncio.FIRST_COMPLETED
            )
            if not done:
                raise TimeoutError
            if self._consumer_task is not None and self._consumer_task in done:
                try:
                    self._consumer_task.result()
                except ReproRunError:
                    raise
                except Exception as exc:
                    raise ReproRunError("validation_failed") from exc
                raise ReproRunError("validation_failed")
            if self._renderer_task is not None and self._renderer_task in done:
                try:
                    exit_code = self._renderer_task.result()
                except Exception as exc:
                    raise ReproRunError("render_failed") from exc
                if exit_code != 0:
                    raise ReproRunError("render_failed")
                raise ReproRunError("validation_failed")
            for completion in pending.intersection(done):
                completion.result()
            pending.difference_update(done)

    def _cycle_for_revision(self, revision: int) -> int:
        return ((revision - 1) // 17) + 1

    async def _consume_renderer_events(self, port: Any) -> None:
        while True:
            try:
                envelope = await port.next_event()
            except Exception as exc:
                if type(exc).__name__ == "RendererDiagnosticPortClosed":
                    return
                raise
            record = envelope.record
            revision = record.get("renderer_revision")
            event_type = record.get("event_type")
            if not isinstance(revision, int) or revision not in self._expected:
                continue
            if event_type == "render_commit":
                if not port.acknowledge_render_commit(revision):
                    raise ReproRunError("validation_failed")
                continue
            if event_type == "receipt":
                disposition = record.get("actual_disposition")
                if disposition == "committed":
                    continue
                self._append_outcome(revision, record, acknowledged=False)
                continue
            if event_type == "render_commit_acknowledgement":
                self._append_outcome(revision, record, acknowledged=True)
                continue
            if event_type == "failed":
                self._append_outcome(revision, record, acknowledged=False, allow_failed=True)

    def _append_outcome(
        self,
        revision: int,
        renderer_record: Mapping[str, object],
        *,
        acknowledged: bool,
        allow_failed: bool = False,
    ) -> None:
        completion = self._completion[revision]
        if completion.done():
            raise ReproRunError("validation_failed")
        cycle, expected = self._expected[revision]
        actual = renderer_record.get("actual_disposition")
        if (actual != expected and not (allow_failed and actual == "failed")) or acknowledged != (
            actual == "committed"
        ):
            raise ReproRunError("validation_failed")
        outcome = validate_desktop_overlay_repro_record(
            {
                "schema_version": DESKTOP_OVERLAY_REPRO_SCHEMA_VERSION,
                "record_type": "revision_outcome",
                "cycle": cycle,
                "wall_clock_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "monotonic_ms": int((time.monotonic() - self._monotonic_origin) * 1000),
                "synthetic_revision": revision,
                "expected_disposition": expected,
                "actual_disposition": actual,
                "render_commit_acknowledged": acknowledged,
                "slot_count": renderer_record.get("slot_count"),
                "line_count": renderer_record.get("line_count"),
                "surface_visible": renderer_record.get("surface_visible"),
                "interaction_mode": renderer_record.get("interaction_mode"),
                "window_width": renderer_record.get("window_width"),
                "window_height": renderer_record.get("window_height"),
            }
        )
        if outcome is None:
            raise ReproRunError("validation_failed")
        self._records.append(outcome)
        if acknowledged:
            self._acknowledged.add(revision)
        if allow_failed:
            completion.set_exception(ReproRunError("render_failed"))
        else:
            completion.set_result(None)

    def _completed_cycles(self) -> int:
        complete = 0
        for cycle in range(1, self.arguments.cycles + 1):
            expected = normative_repro_schedule(cycle)
            revisions = [item.revision for batch in expected for item in batch.revisions]
            if all(
                revision in self._completion and self._completion[revision].done()
                for revision in revisions
            ):
                complete = cycle
            else:
                break
        return complete


async def _finish_owned_task(task: asyncio.Task[Any] | None, *, cancel: bool) -> None:
    if task is None:
        return
    if cancel and not task.done():
        task.cancel()
    results = await asyncio.gather(task, return_exceptions=True)
    result = results[0]
    if isinstance(result, asyncio.CancelledError):
        return
    if isinstance(result, BaseException):
        raise result


def _raise_owned_task_failure(task: asyncio.Task[Any] | None) -> None:
    if task is None or not task.done() or task.cancelled():
        return
    exception = task.exception()
    if exception is not None:
        raise exception


async def _wait_for_locked_interaction_mode(window: Any) -> None:
    deadline = asyncio.get_running_loop().time() + _RENDER_ACK_TIMEOUT_S
    while asyncio.get_running_loop().time() < deadline:
        state = window.renderer_visual_state()
        if state.get("interaction_mode") == "locked":
            return
        await asyncio.sleep(0.01)
    raise ReproRunError("render_timeout")


def normative_repro_schedule(cycle: int) -> tuple[ScheduledBatch, ...]:
    if cycle < 1:
        raise ValueError("cycle must be positive")
    base = 17 * (cycle - 1)
    fixtures = _cycle_fixtures(base)
    return (
        _batch("self_source", ((1, "committed", fixtures["self_source"]),), base),
        _batch("self_translation", ((2, "committed", fixtures["self_translation"]),), base),
        _batch("peer_source", ((3, "committed", fixtures["peer_source"]),), base),
        _batch("peer_translation", ((4, "committed", fixtures["peer_translation"]),), base),
        _batch(
            "fifo_stale",
            (
                (10, "superseded", fixtures["fifo_short"]),
                (9, "stale", fixtures["fifo_stale"]),
                (11, "committed", fixtures["fifo_final"]),
            ),
            base,
        ),
        _batch(
            "width_history",
            (
                (12, "superseded", fixtures["width_short"]),
                (13, "superseded", fixtures["width_long"]),
                (14, "committed", fixtures["width_short_final"]),
            ),
            base,
        ),
        _batch("two_slot", ((15, "committed", fixtures["two_slot"]),), base),
        _batch("empty", ((16, "committed", fixtures["empty"]),), base),
        _batch("return_caption", ((17, "committed", fixtures["return_caption"]),), base),
    )


def _batch(
    name: str, values: tuple[tuple[int, str, OverlayPresentationSnapshot], ...], base: int
) -> ScheduledBatch:
    return ScheduledBatch(
        name=name,
        revisions=tuple(
            ScheduledRevision(base + offset, disposition, _with_revision(snapshot, base + offset))
            for offset, disposition, snapshot in values
        ),
    )


def _with_revision(
    snapshot: OverlayPresentationSnapshot, revision: int
) -> OverlayPresentationSnapshot:
    return OverlayPresentationSnapshot(revision=revision, blocks=snapshot.blocks)


def _cycle_fixtures(base: int) -> dict[str, OverlayPresentationSnapshot]:
    def snapshot(*blocks: OverlayPresentationBlock) -> OverlayPresentationSnapshot:
        return OverlayPresentationSnapshot(revision=base, blocks=list(blocks))

    def block(
        identifier: str,
        *,
        channel: str = "self",
        variant: str = "finalized",
        primary: str = "synthetic caption",
        secondary: str = "",
        appearance: int = 1,
    ) -> OverlayPresentationBlock:
        return OverlayPresentationBlock(
            id=identifier,
            occupant_key=f"fixture:{identifier}",
            appearance_seq=appearance,
            channel=cast(Any, channel),
            block_variant=cast(Any, variant),
            primary_text=primary,
            secondary_text=secondary,
            secondary_enabled=bool(secondary),
        )

    width_identifier = "width-fixture"
    return {
        "self_source": snapshot(block("self", variant="active_self", primary="source")),
        "self_translation": snapshot(block("self", primary="source", secondary="translation")),
        "peer_source": snapshot(block("peer", channel="peer", primary="peer source")),
        "peer_translation": snapshot(
            block("peer", channel="peer", primary="translation", secondary="source")
        ),
        "fifo_short": snapshot(block("fifo", primary="first")),
        "fifo_stale": snapshot(block("fifo", primary="stale")),
        "fifo_final": snapshot(block("fifo", primary="final")),
        "width_short": snapshot(block(width_identifier, primary="short", appearance=70)),
        "width_long": snapshot(
            block(width_identifier, primary="long synthetic fixture width history", appearance=70)
        ),
        "width_short_final": snapshot(block(width_identifier, primary="short", appearance=70)),
        "two_slot": snapshot(
            block("one", primary="one"), block("two", channel="peer", primary="two", appearance=2)
        ),
        "empty": snapshot(),
        "return_caption": snapshot(block("return", primary="return")),
    }


def preflight_repro_arguments(arguments: ReproArguments) -> None:
    if (
        isinstance(arguments.cycles, bool)
        or not isinstance(arguments.cycles, int)
        or not MIN_CYCLES <= arguments.cycles <= MAX_CYCLES
    ):
        raise ReproValidationError("invalid_argument")
    if (
        isinstance(arguments.dwell_ms, bool)
        or not isinstance(arguments.dwell_ms, int)
        or not MIN_DWELL_MS <= arguments.dwell_ms <= MAX_DWELL_MS
    ):
        raise ReproValidationError("invalid_argument")
    output_dir = arguments.output_dir
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        if not output_dir.is_dir() or any(output_dir.iterdir()):
            raise ReproValidationError("invalid_argument")
        probe = output_dir / ".desktop-overlay-repro-write-probe"
        probe.write_bytes(b"")
        probe.unlink()
    except ReproValidationError:
        raise
    except (OSError, ValueError) as exc:
        raise ReproValidationError("invalid_argument") from exc


def run_desktop_overlay_repro(
    *,
    cycles: int = DEFAULT_CYCLES,
    dwell_ms: int = DEFAULT_DWELL_MS,
    output_dir: Path,
    owner_factory: Callable[[ReproArguments], DesktopOverlayReproOwner] = DesktopOverlayReproOwner,
) -> int:
    arguments = ReproArguments(cycles=cycles, dwell_ms=dwell_ms, output_dir=output_dir)
    try:
        preflight_repro_arguments(arguments)
    except ReproValidationError:
        print("invalid_argument")
        return 2
    return asyncio.run(owner_factory(arguments).run())


def _result_record(
    *,
    outcome: str,
    reason: str | None,
    cycles_requested: int,
    cycles_completed: int,
    records: Sequence[Mapping[str, object]],
    renderer_shutdown_completed: bool,
    bridge_shutdown_completed: bool,
    backdrop_shutdown_completed: bool,
) -> Mapping[str, object]:
    counts = {disposition: 0 for disposition in ("committed", "superseded", "stale", "failed")}
    for record in records:
        disposition = record.get("actual_disposition")
        if disposition in counts:
            counts[disposition] += 1
    candidate = {
        "schema_version": DESKTOP_OVERLAY_REPRO_SCHEMA_VERSION,
        "record_type": "run_result",
        "outcome": outcome,
        "reason": reason,
        "cycles_requested": cycles_requested,
        "cycles_completed": cycles_completed,
        "committed_count": counts["committed"],
        "superseded_count": counts["superseded"],
        "stale_count": counts["stale"],
        "failed_count": counts["failed"],
        "renderer_shutdown_completed": renderer_shutdown_completed,
        "bridge_shutdown_completed": bridge_shutdown_completed,
        "backdrop_shutdown_completed": backdrop_shutdown_completed,
    }
    validated = validate_desktop_overlay_repro_result(candidate)
    if validated is None:
        raise ReproValidationError("validation_failed")
    return validated
