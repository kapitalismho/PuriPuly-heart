from __future__ import annotations

import asyncio
import json
import logging
import socket
import time

import pytest
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosedError

from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.overlay.bridge import OverlayBridge
from puripuly_heart.core.overlay.diagnostics import OverlayDiagnosticsRecorder
from puripuly_heart.core.overlay.protocol import (
    NativeFreshRenderTargets,
    NativeQuietTailEpisode,
    NativeQuietTailEpisodes,
    OverlayPresentationBlock,
    OverlayPresentationCalibration,
    OverlayPresentationSnapshot,
)


class _AbruptAuthenticatedConnection:
    def __init__(self) -> None:
        self.sent_payloads: list[dict[str, object]] = []
        self.closed = False

    async def recv(self) -> str:
        return json.dumps({"type": "auth", "session_token": "expected-token"})

    async def send(self, payload: str) -> None:
        self.sent_payloads.append(json.loads(payload))

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        raise ConnectionClosedError(None, None)

    async def close(self) -> None:
        self.closed = True


class _BlockingInitialSnapshotConnection:
    def __init__(self) -> None:
        self.sent_payloads: list[dict[str, object]] = []
        self.closed = False
        self.initial_send_started = asyncio.Event()
        self.release_initial_send = asyncio.Event()
        self.allow_disconnect = asyncio.Event()

    async def recv(self) -> str:
        return json.dumps({"type": "auth", "session_token": "expected-token"})

    async def send(self, payload: str) -> None:
        message = json.loads(payload)
        if (
            message.get("type") == "snapshot"
            and message.get("payload", {}).get("revision") == 0
            and not self.initial_send_started.is_set()
        ):
            self.initial_send_started.set()
            await self.release_initial_send.wait()
        self.sent_payloads.append(message)

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        await self.allow_disconnect.wait()
        raise ConnectionClosedError(None, None)

    async def close(self) -> None:
        self.closed = True


class _FailingSendConnection:
    def __init__(self) -> None:
        self.close_calls = 0

    async def send(self, payload: str) -> None:
        _ = payload
        raise RuntimeError("boom")

    async def close(self) -> None:
        self.close_calls += 1


class _RecordingSendConnection:
    def __init__(self) -> None:
        self.sent_payloads: list[dict[str, object]] = []

    async def send(self, payload: str) -> None:
        self.sent_payloads.append(json.loads(payload))

    async def close(self) -> None:
        return None


class _BlockingSendConnection(_RecordingSendConnection):
    def __init__(self) -> None:
        super().__init__()
        self.send_started = asyncio.Event()
        self.release_send = asyncio.Event()

    async def send(self, payload: str) -> None:
        self.send_started.set()
        await self.release_send.wait()
        await super().send(payload)


async def _wait_until(condition) -> None:
    async def poll() -> None:
        while not condition():
            await asyncio.sleep(0)

    await asyncio.wait_for(poll(), timeout=0.5)


def _refresh_marker_snapshot(
    *,
    revision: int,
    peer_session_scope: str | None,
    self_session_scope: str | None,
) -> OverlayPresentationSnapshot:
    return OverlayPresentationSnapshot(
        revision=revision,
        calibration=OverlayPresentationCalibration(),
        blocks=[
            OverlayPresentationBlock(
                id="peer:refresh-turn",
                occupant_key="peer:refresh-turn",
                appearance_seq=1,
                channel="peer",
                block_variant="finalized",
                primary_text="peer translated text",
                secondary_text="peer source text",
                secondary_enabled=True,
                update_id="peer-refresh-update",
                session_scope=peer_session_scope,
            ),
            OverlayPresentationBlock(
                id="self:refresh-turn",
                occupant_key="self:refresh-turn",
                appearance_seq=2,
                channel="self",
                block_variant="finalized",
                primary_text="self source text",
                secondary_text="",
                secondary_enabled=True,
                session_scope=self_session_scope,
            ),
        ],
    )


@pytest.mark.asyncio
async def test_overlay_bridge_stop_bounds_and_tracks_failed_connection_close() -> None:
    class FailingOnceCloseConnection:
        def __init__(self) -> None:
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1
            if self.close_calls == 1:
                raise RuntimeError("connection close failed")

    bridge = OverlayBridge(session_token="expected-token")
    connection = FailingOnceCloseConnection()
    bridge._authenticated_connections.add(connection)

    with pytest.raises(ExceptionGroup, match="OverlayBridge stop failed"):
        await bridge.stop()

    assert connection not in bridge._authenticated_connections
    assert connection in bridge._unresolved_connections

    with pytest.raises(ExceptionGroup, match="OverlayBridge stop failed"):
        await bridge.stop()

    assert connection.close_calls == 1
    assert connection in bridge._unresolved_connections


@pytest.mark.asyncio
async def test_overlay_bridge_stop_clears_server_state_when_wait_closed_fails() -> None:
    class FailingServer:
        def __init__(self) -> None:
            self.close_calls = 0
            self.wait_closed_calls = 0

        def close(self) -> None:
            self.close_calls += 1

        async def wait_closed(self) -> None:
            self.wait_closed_calls += 1
            raise RuntimeError("server close failed")

    bridge = OverlayBridge(session_token="expected-token")
    server = FailingServer()
    bridge._server = server
    bridge.url = "ws://127.0.0.1:9999"
    bridge._token_consumed = True
    await bridge.messages.put({"type": "pending"})

    with pytest.raises(ExceptionGroup, match="OverlayBridge stop failed"):
        await bridge.stop()

    assert server.close_calls == 1
    assert server.wait_closed_calls == 1
    assert bridge._server is None
    assert bridge.url == ""
    assert bridge._token_consumed is False
    assert bridge.messages.empty()


@pytest.mark.asyncio
async def test_overlay_bridge_requires_session_token() -> None:
    bridge = OverlayBridge(session_token="expected-token")
    await bridge.start()

    try:
        async with connect(bridge.url) as ws:
            await ws.send(json.dumps({"type": "auth", "session_token": "wrong-token"}))
            message = json.loads(await asyncio.wait_for(ws.recv(), timeout=0.5))
    finally:
        await bridge.stop()

    assert message["type"] == "auth_error"


@pytest.mark.asyncio
async def test_overlay_bridge_sends_authenticated_initial_snapshot() -> None:
    bridge = OverlayBridge(
        session_token="expected-token",
        initial_snapshot=OverlayPresentationSnapshot(
            revision=0,
            calibration=OverlayPresentationCalibration(),
            blocks=[],
        ),
    )
    await bridge.start()

    try:
        async with connect(bridge.url) as ws:
            await ws.send(json.dumps({"type": "auth", "session_token": "expected-token"}))
            message = json.loads(await asyncio.wait_for(ws.recv(), timeout=0.5))
    finally:
        await bridge.stop()

    assert message["type"] == "snapshot"
    assert message["payload"]["revision"] == 0
    assert message["payload"]["blocks"] == []


@pytest.mark.asyncio
async def test_overlay_bridge_emits_heartbeat_after_authentication() -> None:
    bridge = OverlayBridge(
        session_token="expected-token",
        initial_snapshot=OverlayPresentationSnapshot(
            revision=0,
            calibration=OverlayPresentationCalibration(),
            blocks=[],
        ),
        heartbeat_interval_ms=50,
    )
    await bridge.start()

    try:
        async with connect(bridge.url) as ws:
            await ws.send(json.dumps({"type": "auth", "session_token": "expected-token"}))
            await asyncio.wait_for(ws.recv(), timeout=0.5)
            message = json.loads(await asyncio.wait_for(ws.recv(), timeout=0.5))
    finally:
        await bridge.stop()

    assert message["type"] == "heartbeat"


@pytest.mark.asyncio
async def test_overlay_bridge_resets_one_time_token_after_stop_and_restart() -> None:
    bridge = OverlayBridge(
        session_token="expected-token",
        initial_snapshot=OverlayPresentationSnapshot(
            revision=0,
            calibration=OverlayPresentationCalibration(),
            blocks=[],
        ),
    )

    await bridge.start()
    try:
        async with connect(bridge.url) as ws:
            await ws.send(json.dumps({"type": "auth", "session_token": "expected-token"}))
            first_message = json.loads(await asyncio.wait_for(ws.recv(), timeout=0.5))
            await ws.send(json.dumps({"type": "runtime_error", "failure_reason": "boom"}))
            queued = await asyncio.wait_for(bridge.messages.get(), timeout=0.5)
            assert queued["type"] == "runtime_error"
    finally:
        await bridge.stop()

    assert bridge.messages.empty()

    await bridge.start()
    try:
        async with connect(bridge.url) as ws:
            await ws.send(json.dumps({"type": "auth", "session_token": "expected-token"}))
            second_message = json.loads(await asyncio.wait_for(ws.recv(), timeout=0.5))
    finally:
        await bridge.stop()

    assert first_message["type"] == "snapshot"
    assert second_message["type"] == "snapshot"


@pytest.mark.asyncio
async def test_overlay_bridge_swallows_authenticated_disconnect_without_close_frame() -> None:
    bridge = OverlayBridge(
        session_token="expected-token",
        initial_snapshot=OverlayPresentationSnapshot(
            revision=0,
            calibration=OverlayPresentationCalibration(),
            blocks=[],
        ),
    )
    connection = _AbruptAuthenticatedConnection()

    await bridge._handle_connection(connection)

    assert connection.closed is True
    assert connection.sent_payloads == [
        {
            "type": "snapshot",
            "payload": {
                "revision": 0,
                "calibration": OverlayPresentationCalibration().to_dict(),
                "blocks": [],
            },
        }
    ]
    assert bridge._authenticated_connections == set()


@pytest.mark.asyncio
async def test_overlay_bridge_broadcasts_full_snapshot_replacements() -> None:
    bridge = OverlayBridge(
        session_token="expected-token",
        initial_snapshot=OverlayPresentationSnapshot(
            revision=0,
            calibration=OverlayPresentationCalibration(),
            blocks=[],
        ),
    )
    await bridge.start()

    try:
        async with connect(bridge.url) as ws:
            await ws.send(json.dumps({"type": "auth", "session_token": "expected-token"}))
            await asyncio.wait_for(ws.recv(), timeout=0.5)

            await bridge.replace_snapshot(
                OverlayPresentationSnapshot(
                    revision=1,
                    calibration=OverlayPresentationCalibration(distance=1.4),
                    blocks=[],
                )
            )

            message = json.loads(await asyncio.wait_for(ws.recv(), timeout=0.5))
    finally:
        await bridge.stop()

    assert message["type"] == "snapshot"
    assert message["payload"]["revision"] == 1
    assert message["payload"]["calibration"]["distance"] == 1.4


@pytest.mark.asyncio
async def test_overlay_bridge_coalesces_unsent_refreshes_to_latest_scene() -> None:
    bridge = OverlayBridge(
        session_token="expected-token",
        initial_snapshot=_refresh_marker_snapshot(
            revision=1,
            peer_session_scope="session:peer",
            self_session_scope=None,
        ),
    )
    connection = _RecordingSendConnection()
    bridge._authenticated_connections.add(connection)  # type: ignore[arg-type]

    await bridge.replace_snapshot(
        _refresh_marker_snapshot(
            revision=2,
            peer_session_scope="session:peer|peer_presentation_refresh=1",
            self_session_scope="self_presentation_refresh=1",
        )
    )
    await bridge.replace_snapshot(
        _refresh_marker_snapshot(
            revision=3,
            peer_session_scope="session:peer|peer_presentation_refresh=2",
            self_session_scope="self_presentation_refresh=2",
        )
    )
    await bridge.replace_snapshot(
        _refresh_marker_snapshot(
            revision=4,
            peer_session_scope="session:peer",
            self_session_scope=None,
        )
    )
    await _wait_until(lambda: len(connection.sent_payloads) == 1)

    snapshot_payloads = [
        payload["payload"]
        for payload in connection.sent_payloads
        if payload.get("type") == "snapshot"
    ]
    assert [payload["revision"] for payload in snapshot_payloads] == [4]
    assert [
        (
            payload["blocks"][0]["primary_text"],
            payload["blocks"][0]["secondary_text"],
            payload["blocks"][1]["primary_text"],
        )
        for payload in snapshot_payloads
    ] == [("peer translated text", "peer source text", "self source text")]
    assert [payload["blocks"][0].get("session_scope") for payload in snapshot_payloads] == [
        "session:peer"
    ]
    assert [payload["blocks"][1].get("session_scope") for payload in snapshot_payloads] == [None]


@pytest.mark.asyncio
async def test_overlay_bridge_does_not_send_stale_initial_snapshot_after_newer_live_snapshot() -> (
    None
):
    bridge = OverlayBridge(
        session_token="expected-token",
        initial_snapshot=OverlayPresentationSnapshot(
            revision=0,
            calibration=OverlayPresentationCalibration(),
            blocks=[],
        ),
    )
    connection = _BlockingInitialSnapshotConnection()

    handle_task = asyncio.create_task(bridge._handle_connection(connection))
    await asyncio.wait_for(connection.initial_send_started.wait(), timeout=0.5)

    replace_task = asyncio.create_task(
        bridge.replace_snapshot(
            OverlayPresentationSnapshot(
                revision=1,
                calibration=OverlayPresentationCalibration(distance=1.6),
                blocks=[],
            )
        )
    )
    await asyncio.sleep(0)
    connection.release_initial_send.set()
    await asyncio.wait_for(replace_task, timeout=0.5)
    await _wait_until(lambda: len(connection.sent_payloads) == 2)
    connection.allow_disconnect.set()
    await handle_task

    assert [payload["payload"]["revision"] for payload in connection.sent_payloads] == [0, 1]


@pytest.mark.asyncio
async def test_overlay_bridge_ignores_stale_snapshot_replacements() -> None:
    bridge = OverlayBridge(
        session_token="expected-token",
        initial_snapshot=OverlayPresentationSnapshot(
            revision=2,
            calibration=OverlayPresentationCalibration(distance=1.4),
            blocks=[],
        ),
    )

    await bridge.replace_snapshot(
        OverlayPresentationSnapshot(
            revision=1,
            calibration=OverlayPresentationCalibration(distance=1.8),
            blocks=[],
        )
    )

    assert bridge.snapshot().revision == 2
    assert bridge.snapshot().calibration.distance == 1.4


@pytest.mark.asyncio
async def test_overlay_bridge_revalidates_successor_age_after_stalled_send() -> None:
    clock = FakeClock(_now=10.0)
    bridge = OverlayBridge(session_token="expected-token", clock=clock)
    connection = _BlockingSendConnection()
    bridge._authenticated_connections.add(connection)  # type: ignore[arg-type]

    await bridge.replace_snapshot(
        OverlayPresentationSnapshot(
            revision=1,
            calibration=OverlayPresentationCalibration(),
            blocks=[
                OverlayPresentationBlock(
                    id="peer:first",
                    occupant_key="peer:first",
                    appearance_seq=1,
                    channel="peer",
                    block_variant="finalized",
                    primary_text="first",
                    secondary_text="",
                    secondary_enabled=False,
                )
            ],
        )
    )
    await connection.send_started.wait()
    await bridge.replace_snapshot(
        OverlayPresentationSnapshot(
            revision=2,
            calibration=OverlayPresentationCalibration(),
            blocks=[
                OverlayPresentationBlock(
                    id="peer:expired",
                    occupant_key="peer:expired",
                    appearance_seq=2,
                    channel="peer",
                    block_variant="finalized",
                    primary_text="expired",
                    secondary_text="",
                    secondary_enabled=False,
                )
            ],
            native_fresh_render_targets=NativeFreshRenderTargets(peer="peer:expired"),
            native_quiet_tail_episodes=NativeQuietTailEpisodes(
                peer=NativeQuietTailEpisode(phase="final", generation=1)
            ),
        ),
        block_expirations={"peer:expired": 11.0},
    )
    clock.advance(2.0)
    connection.release_send.set()
    await _wait_until(lambda: len(connection.sent_payloads) == 2)

    assert connection.sent_payloads[0]["payload"]["revision"] == 1
    assert connection.sent_payloads[1]["payload"] == {
        "revision": 2,
        "calibration": OverlayPresentationCalibration().to_dict(),
        "blocks": [],
    }
    await bridge.stop()


@pytest.mark.asyncio
async def test_overlay_bridge_cancellation_is_ambiguous_and_replays_latest_on_reconnect() -> None:
    bridge = OverlayBridge(session_token="expected-token")
    stalled = _BlockingSendConnection()
    bridge._authenticated_connections.add(stalled)  # type: ignore[arg-type]

    await bridge.replace_snapshot(
        OverlayPresentationSnapshot(
            revision=1,
            calibration=OverlayPresentationCalibration(distance=1.6),
            blocks=[],
        )
    )
    await stalled.send_started.wait()
    writer = bridge._writer_task
    assert writer is not None
    writer.cancel()
    await asyncio.gather(writer, return_exceptions=True)

    ambiguous = [
        receipt
        for receipt in bridge.delivery_receipts
        if receipt.outcome == "ambiguous" and receipt.scene_revision == 1
    ]
    assert len(ambiguous) == 1
    assert ambiguous[0].stage == "ambiguous_receipt"
    assert stalled not in bridge._authenticated_connections
    assert len(bridge._unresolved_transport_tasks) == 1

    stalled.release_send.set()
    await _wait_until(lambda: not bridge._unresolved_transport_tasks)
    reconnected = _AbruptAuthenticatedConnection()
    await bridge._handle_connection(reconnected)  # type: ignore[arg-type]

    assert [payload["payload"]["revision"] for payload in reconnected.sent_payloads] == [1]
    assert reconnected.sent_payloads[0]["payload"]["calibration"]["distance"] == 1.6
    await bridge.stop()


@pytest.mark.asyncio
async def test_overlay_bridge_replays_runtime_logging_mode_after_authentication() -> None:
    bridge = OverlayBridge(
        session_token="expected-token",
        runtime_logging_mode="detailed",
        initial_snapshot=OverlayPresentationSnapshot(
            revision=0,
            calibration=OverlayPresentationCalibration(),
            blocks=[],
        ),
    )
    await bridge.start()

    try:
        async with connect(bridge.url) as ws:
            await ws.send(json.dumps({"type": "auth", "session_token": "expected-token"}))
            runtime_control = json.loads(await asyncio.wait_for(ws.recv(), timeout=0.5))
            snapshot = json.loads(await asyncio.wait_for(ws.recv(), timeout=0.5))
    finally:
        await bridge.stop()
    assert runtime_control == {
        "type": "runtime_control",
        "payload": {"logging_mode": "detailed"},
    }
    assert snapshot["type"] == "snapshot"


@pytest.mark.asyncio
async def test_overlay_bridge_runtime_control_logging_wire_format_remains_exact() -> None:
    bridge = OverlayBridge(
        session_token="expected-token",
        initial_snapshot=OverlayPresentationSnapshot(
            revision=0,
            calibration=OverlayPresentationCalibration(),
            blocks=[],
        ),
    )
    connection = _RecordingSendConnection()
    bridge._authenticated_connections.add(connection)  # type: ignore[arg-type]

    await bridge.broadcast_runtime_control(logging_mode="detailed")
    await _wait_until(lambda: len(connection.sent_payloads) == 1)

    assert connection.sent_payloads == [
        {
            "type": "runtime_control",
            "payload": {"logging_mode": "detailed"},
        }
    ]


@pytest.mark.asyncio
async def test_overlay_bridge_desktop_runtime_control_broadcasts_payload_when_enabled() -> None:
    bridge = OverlayBridge(
        session_token="expected-token",
        desktop_runtime_controls_enabled=True,
        initial_snapshot=OverlayPresentationSnapshot(
            revision=0,
            calibration=OverlayPresentationCalibration(),
            blocks=[],
        ),
    )
    connection = _RecordingSendConnection()
    bridge._authenticated_connections.add(connection)  # type: ignore[arg-type]
    payload = {"command": "set_interaction_mode", "mode": "edit"}

    await bridge.broadcast_desktop_runtime_control(payload)
    await _wait_until(lambda: len(connection.sent_payloads) == 1)

    assert connection.sent_payloads == [
        {
            "type": "runtime_control",
            "payload": payload,
        }
    ]


@pytest.mark.asyncio
async def test_overlay_bridge_desktop_initial_control_replay_after_snapshot_and_logging() -> None:
    bridge = OverlayBridge(
        session_token="expected-token",
        desktop_runtime_controls_enabled=True,
        runtime_logging_mode="detailed",
        initial_snapshot=OverlayPresentationSnapshot(
            revision=0,
            calibration=OverlayPresentationCalibration(),
            blocks=[],
        ),
    )
    initial_controls = [
        {"command": "set_interaction_mode", "mode": "edit"},
        {
            "command": "apply_window_bounds",
            "x": 320,
            "y": 720,
            "width": 1280,
            "height": 330,
        },
    ]
    bridge.set_initial_desktop_runtime_controls(initial_controls)
    await bridge.start()

    try:
        async with connect(bridge.url) as ws:
            await ws.send(json.dumps({"type": "auth", "session_token": "expected-token"}))
            messages = [json.loads(await asyncio.wait_for(ws.recv(), timeout=0.5))]
    finally:
        await bridge.stop()

    assert messages == [
        {
            "type": "snapshot",
            "payload": {
                "revision": 0,
                "calibration": OverlayPresentationCalibration().to_dict(),
                "blocks": [],
            },
            "startup_runtime_controls": [
                {"logging_mode": "detailed"},
                initial_controls[0],
                initial_controls[1],
            ],
        },
    ]


@pytest.mark.asyncio
async def test_overlay_bridge_desktop_runtime_control_is_target_gated_from_steamvr_path() -> None:
    bridge = OverlayBridge(
        session_token="expected-token",
        initial_snapshot=OverlayPresentationSnapshot(
            revision=0,
            calibration=OverlayPresentationCalibration(),
            blocks=[],
        ),
    )
    connection = _RecordingSendConnection()
    bridge._authenticated_connections.add(connection)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="desktop runtime controls"):
        await bridge.broadcast_desktop_runtime_control(
            {"command": "set_interaction_mode", "mode": "edit"}
        )
    with pytest.raises(RuntimeError, match="desktop runtime controls"):
        bridge.set_initial_desktop_runtime_controls(
            [{"command": "set_interaction_mode", "mode": "edit"}]
        )

    assert connection.sent_payloads == []


@pytest.mark.asyncio
async def test_overlay_bridge_broadcasts_runtime_logging_mode_updates() -> None:
    bridge = OverlayBridge(
        session_token="expected-token",
        initial_snapshot=OverlayPresentationSnapshot(
            revision=0,
            calibration=OverlayPresentationCalibration(),
            blocks=[],
        ),
    )
    await bridge.start()

    try:
        async with connect(bridge.url) as ws:
            await ws.send(json.dumps({"type": "auth", "session_token": "expected-token"}))
            await asyncio.wait_for(ws.recv(), timeout=0.5)

            await bridge.broadcast_runtime_control(logging_mode="detailed")
            runtime_control = json.loads(await asyncio.wait_for(ws.recv(), timeout=0.5))
    finally:
        await bridge.stop()

    assert runtime_control == {
        "type": "runtime_control",
        "payload": {"logging_mode": "detailed"},
    }


@pytest.mark.asyncio
async def test_overlay_bridge_replace_snapshot_does_not_log_snapshot_updated(
    caplog: pytest.LogCaptureFixture,
) -> None:
    bridge = OverlayBridge(
        session_token="expected-token",
        overlay_instance_id="quiet-overlay",
        initial_snapshot=OverlayPresentationSnapshot(
            revision=0,
            calibration=OverlayPresentationCalibration(),
            blocks=[],
        ),
    )
    connection = _RecordingSendConnection()
    bridge._authenticated_connections.add(connection)  # type: ignore[arg-type]
    caplog.set_level(logging.INFO, logger="puripuly_heart.core.overlay.bridge")

    await bridge.replace_snapshot(
        OverlayPresentationSnapshot(
            revision=1,
            calibration=OverlayPresentationCalibration(distance=1.5),
            blocks=[],
        )
    )
    await _wait_until(lambda: len(connection.sent_payloads) == 1)

    assert connection.sent_payloads[-1]["type"] == "snapshot"
    assert not any(
        "[OverlayBridge] Snapshot updated" in record.getMessage() for record in caplog.records
    )


@pytest.mark.asyncio
async def test_overlay_bridge_records_disconnect_code_and_reason(
    tmp_path,
) -> None:
    diagnostics = OverlayDiagnosticsRecorder(
        overlay_instance_id="overlay-test",
        diagnostics_dir=tmp_path,
        logging_mode="detailed",
    )
    bridge = OverlayBridge(
        session_token="expected-token",
        overlay_instance_id="overlay-test",
        diagnostics=diagnostics,
        initial_snapshot=OverlayPresentationSnapshot(
            revision=0,
            calibration=OverlayPresentationCalibration(),
            blocks=[],
        ),
    )
    await bridge.start()

    async def _wait_until_connection_closed() -> None:
        while not any(event["event"] == "connection_closed" for event in diagnostics.bridge_events):
            await asyncio.sleep(0)

    try:
        async with connect(bridge.url) as ws:
            await ws.send(json.dumps({"type": "auth", "session_token": "expected-token"}))
            await asyncio.wait_for(ws.recv(), timeout=0.5)
            await ws.close(code=4001, reason="client_bye")
            await ws.wait_closed()
            await asyncio.wait_for(_wait_until_connection_closed(), timeout=2.0)
    finally:
        await bridge.stop()
    events = list(diagnostics.bridge_events)

    assert [event["event"] for event in events] == [
        "connection_authenticated",
        "send_start",
        "send_finish",
        "connection_closed",
        "connection_detached",
    ]
    closed = events[3]
    assert closed["code"] == 4001
    assert closed["reason"] == "client_bye"


@pytest.mark.asyncio
async def test_overlay_bridge_records_send_failures_and_prunes_stale_connections(
    tmp_path,
) -> None:
    diagnostics = OverlayDiagnosticsRecorder(
        overlay_instance_id="overlay-test",
        diagnostics_dir=tmp_path,
        logging_mode="detailed",
    )
    bridge = OverlayBridge(
        session_token="expected-token",
        overlay_instance_id="overlay-test",
        diagnostics=diagnostics,
        initial_snapshot=OverlayPresentationSnapshot(
            revision=0,
            calibration=OverlayPresentationCalibration(),
            blocks=[],
        ),
    )
    connection = _FailingSendConnection()
    bridge._authenticated_connections.add(connection)  # type: ignore[arg-type]

    await bridge.replace_snapshot(
        OverlayPresentationSnapshot(
            revision=1,
            calibration=OverlayPresentationCalibration(),
            blocks=[],
        )
    )
    await _wait_until(
        lambda: any(event["event"] == "send_finish" for event in diagnostics.bridge_events)
    )

    events = list(diagnostics.bridge_events)
    assert [event["event"] for event in events] == [
        "send_start",
        "send_failure",
        "connection_retired",
        "send_finish",
    ]
    assert events[1]["removed"] is True
    assert events[1]["exception_type"]
    assert events[3]["stale_connections"] == 1
    assert bridge._authenticated_connections == set()


@pytest.mark.asyncio
async def test_overlay_bridge_snapshot_broadcast_logs_only_in_detailed_mode(
    caplog: pytest.LogCaptureFixture,
) -> None:
    detailed_bridge = OverlayBridge(
        session_token="expected-token",
        overlay_instance_id="detailed-overlay",
        runtime_logging_mode="detailed",
        initial_snapshot=OverlayPresentationSnapshot(
            revision=0,
            calibration=OverlayPresentationCalibration(),
            blocks=[],
        ),
    )
    basic_bridge = OverlayBridge(
        session_token="expected-token",
        overlay_instance_id="basic-overlay",
        runtime_logging_mode="basic",
        initial_snapshot=OverlayPresentationSnapshot(
            revision=0,
            calibration=OverlayPresentationCalibration(),
            blocks=[],
        ),
    )
    detailed_connection = _RecordingSendConnection()
    basic_connection = _RecordingSendConnection()
    detailed_bridge._authenticated_connections.add(detailed_connection)  # type: ignore[arg-type]
    basic_bridge._authenticated_connections.add(basic_connection)  # type: ignore[arg-type]

    snapshot = OverlayPresentationSnapshot(
        revision=1,
        calibration=OverlayPresentationCalibration(distance=1.5),
        blocks=[
            OverlayPresentationBlock(
                id="peer:one",
                occupant_key="peer:one",
                appearance_seq=1,
                channel="peer",
                block_variant="finalized",
                primary_text="peer translation",
                secondary_text="peer original",
                secondary_enabled=True,
                update_id="bridge-upd-1",
            ),
            OverlayPresentationBlock(
                id="self:two",
                occupant_key="self:two",
                appearance_seq=2,
                channel="self",
                block_variant="finalized",
                primary_text="self original",
                secondary_text="self translation",
                secondary_enabled=True,
                update_id="bridge-upd-2",
            ),
        ],
    )

    caplog.set_level(logging.INFO, logger="puripuly_heart.core.overlay.bridge")

    await basic_bridge.replace_snapshot(snapshot)
    await detailed_bridge.replace_snapshot(snapshot)
    await _wait_until(
        lambda: any(
            "[OverlayBridge][Broadcast] stage=finish overlay_instance_id=detailed-overlay"
            in record.getMessage()
            for record in caplog.records
        )
    )

    broadcast_messages = [
        record.getMessage()
        for record in caplog.records
        if "[OverlayBridge][Broadcast]" in record.getMessage()
    ]

    assert not any("overlay_instance_id=basic-overlay" in message for message in broadcast_messages)
    assert any(
        "stage=start" in message
        and "overlay_instance_id=detailed-overlay" in message
        and "revision=1" in message
        and "block_update_ids=['bridge-upd-1', 'bridge-upd-2']" in message
        for message in broadcast_messages
    )
    assert any(
        "stage=finish" in message
        and "overlay_instance_id=detailed-overlay" in message
        and "revision=1" in message
        and "block_update_ids=['bridge-upd-1', 'bridge-upd-2']" in message
        and "elapsed_ms=" in message
        for message in broadcast_messages
    )


@pytest.mark.asyncio
async def test_overlay_bridge_records_snapshot_send_stages_without_payload_text() -> None:
    recorder = OverlayDiagnosticsRecorder(
        overlay_instance_id="bridge-stage-test",
        logging_mode="detailed",
    )
    bridge = OverlayBridge(session_token="expected-token", diagnostics=recorder)
    snapshot = OverlayPresentationSnapshot(
        revision=1,
        calibration=OverlayPresentationCalibration(),
        blocks=[],
    )

    await bridge.replace_snapshot(snapshot)

    unsent = list(recorder.bridge_events)
    assert [event["event"] for event in unsent] == ["snapshot_stored_unsent"]
    assert unsent[0]["revision"] == 1

    connection = _RecordingSendConnection()
    bridge._authenticated_connections.add(connection)  # type: ignore[arg-type]
    await bridge.replace_snapshot(
        OverlayPresentationSnapshot(
            revision=2,
            calibration=OverlayPresentationCalibration(),
            blocks=[],
        )
    )
    await _wait_until(
        lambda: any(event["event"] == "send_finish" for event in recorder.bridge_events)
    )

    events = list(recorder.bridge_events)
    assert [event["event"] for event in events[-2:]] == ["send_start", "send_finish"]
    assert events[-1]["revision"] == 2
    assert "elapsed_ms" in events[-1]
    dumped = json.dumps(events)
    assert "primary_text" not in dumped
    assert "secondary_text" not in dumped


@pytest.mark.asyncio
async def test_overlay_bridge_unresolved_transport_caps_one_epoch_and_rejects_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import puripuly_heart.core.overlay.bridge as bridge_module

    monkeypatch.setattr(bridge_module, "_SCENE_WRITE_TIMEOUT_SECONDS", 0.02)
    monkeypatch.setattr(bridge_module, "_CLOSE_TIMEOUT_SECONDS", 0.02)

    class IneffectiveTransport:
        def __init__(self) -> None:
            self.abort_calls = 0

        def abort(self) -> None:
            self.abort_calls += 1

    class CancellationResistantConnection:
        def __init__(self) -> None:
            self.transport = IneffectiveTransport()
            self.send_started = asyncio.Event()
            self.close_started = asyncio.Event()
            self.release = asyncio.Event()
            self.send_calls = 0
            self.send_completions = 0
            self.close_calls = 0

        async def send(self, _payload: str) -> None:
            self.send_calls += 1
            self.send_started.set()
            while not self.release.is_set():
                try:
                    await self.release.wait()
                except asyncio.CancelledError:
                    continue
            self.send_completions += 1

        async def close(self) -> None:
            self.close_calls += 1
            self.close_started.set()
            while not self.release.is_set():
                try:
                    await self.release.wait()
                except asyncio.CancelledError:
                    continue

    bridge = OverlayBridge(session_token="expected-token")
    original = CancellationResistantConnection()
    bridge._authenticated_connections.add(original)  # type: ignore[arg-type]
    await bridge.replace_snapshot(
        OverlayPresentationSnapshot(
            revision=1,
            calibration=OverlayPresentationCalibration(),
            blocks=[],
        )
    )
    await original.send_started.wait()
    await original.close_started.wait()
    await _wait_until(lambda: len(bridge._unresolved_transport_tasks) == 2)

    replacements = [CancellationResistantConnection() for _ in range(6)]
    for replacement in replacements:
        await bridge._handle_connection(replacement)  # type: ignore[arg-type]

    assert original.send_calls == 1
    assert len(bridge._unresolved_connections) == 1
    assert len(bridge._unresolved_transport_tasks) == 2
    assert sum(replacement.transport.abort_calls for replacement in replacements) == 6
    assert all(replacement.send_calls == 0 for replacement in replacements)
    assert all(replacement.close_calls == 0 for replacement in replacements)

    with pytest.raises(ExceptionGroup) as raised:
        await bridge.stop()
    assert len(raised.value.exceptions) == 2
    assert original.close_calls == 1

    original.release.set()
    await _wait_until(lambda: not bridge._unresolved_transport_tasks)
    assert original.send_completions == 1


@pytest.mark.asyncio
async def test_overlay_bridge_reserves_shutdown_and_coalesces_control_overflow_retirement() -> None:
    bridge = OverlayBridge(session_token="expected-token")
    for index in range(8):
        bridge._enqueue_control(
            f"control-{index}",
            {"type": f"control-{index}"},
        )
    await bridge.broadcast_shutdown()

    assert len(bridge._pending_controls) == 8
    assert "shutdown" in bridge._pending_controls

    connection = _BlockingSendConnection()
    bridge._authenticated_connections.add(connection)  # type: ignore[arg-type]
    bridge._ensure_writer()
    bridge._writer_wakeup.set()
    await connection.send_started.wait()

    bridge._enqueue_control("fill-after-shutdown", {"type": "fill-after-shutdown"})
    for index in range(200):
        with pytest.raises(RuntimeError, match="control capacity"):
            bridge._enqueue_control(
                f"overflow-{index}",
                {"type": f"overflow-{index}"},
            )
    retirement_tasks = [
        task
        for task in asyncio.all_tasks()
        if task.get_name() == "OverlayBridge:connection-retirement" and not task.done()
    ]
    assert len(retirement_tasks) == 1

    connection.release_send.set()
    await asyncio.gather(*retirement_tasks)
    assert connection.sent_payloads[0]["type"] == "shutdown"
    await bridge.stop()


@pytest.mark.asyncio
async def test_overlay_bridge_stop_closes_ingress_and_terminalizes_pending_scene() -> None:
    class ControlledServer:
        def __init__(self) -> None:
            self.release = asyncio.Event()

        def close(self) -> None:
            return None

        async def wait_closed(self) -> None:
            await self.release.wait()

    bridge = OverlayBridge(session_token="expected-token")
    server = ControlledServer()
    bridge._server = server  # type: ignore[assignment]
    admitted = await bridge.replace_snapshot(
        OverlayPresentationSnapshot(
            revision=1,
            calibration=OverlayPresentationCalibration(),
            blocks=[],
        )
    )
    stop_task = asyncio.create_task(bridge.stop())
    await asyncio.sleep(0)
    assert bridge._stopping

    stopping = await bridge.replace_snapshot(
        OverlayPresentationSnapshot(
            revision=2,
            calibration=OverlayPresentationCalibration(),
            blocks=[],
        )
    )
    with pytest.raises(RuntimeError, match="not accepting controls"):
        await bridge.broadcast_runtime_control(logging_mode="basic")
    server.release.set()
    await stop_task
    stopped = await bridge.replace_snapshot(
        OverlayPresentationSnapshot(
            revision=3,
            calibration=OverlayPresentationCalibration(),
            blocks=[],
        )
    )

    assert admitted.outcome == "admitted"
    assert stopping.outcome == "delivery_rejected"
    assert stopping.cause == "bridge_stopping"
    assert stopped.outcome == "delivery_rejected"
    assert stopped.cause == "bridge_stopped"
    assert any(
        receipt.scene_revision == 1
        and receipt.outcome == "delivery_rejected"
        and receipt.cause == "bridge_stopping"
        for receipt in bridge.delivery_receipts
    )


@pytest.mark.asyncio
async def test_overlay_bridge_real_socket_stopped_reader_stops_bounded_and_truthfully() -> None:
    bridge = OverlayBridge(session_token="real-smoke-token")
    await bridge.start()
    client = await connect(
        bridge.url,
        ping_interval=None,
        compression=None,
        max_size=2 * 1024 * 1024,
    )
    raw_socket = client.transport.get_extra_info("socket")
    raw_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    await client.send('{"type":"auth","session_token":"real-smoke-token"}')
    await client.recv()
    client.transport.pause_reading()
    payload = "x" * 900_000
    blocked_revision = None

    try:
        for revision in range(1, 65):
            await bridge.replace_snapshot(
                OverlayPresentationSnapshot(
                    revision=revision,
                    calibration=OverlayPresentationCalibration(),
                    blocks=[
                        OverlayPresentationBlock(
                            id="self:real-socket",
                            occupant_key="self:real-socket",
                            appearance_seq=1,
                            channel="self",
                            block_variant="finalized",
                            primary_text=payload,
                            secondary_text="",
                            secondary_enabled=False,
                            update_id=f"real-socket-{revision}",
                        )
                    ],
                )
            )
            await asyncio.sleep(0.02)
            if bridge._active_scene is not None:
                await asyncio.sleep(0.08)
                if bridge._active_scene is not None:
                    blocked_revision = bridge._active_scene.snapshot.revision
                    break
        assert blocked_revision is not None

        started = time.perf_counter()
        await bridge.stop()
        elapsed = time.perf_counter() - started

        assert elapsed < 3.0
        assert bridge._stopped
        assert bridge._authenticated_connections == set()
        assert [task for task in bridge._unresolved_transport_tasks if not task.done()] == []
    finally:
        client.transport.resume_reading()
        try:
            await asyncio.wait_for(client.close(), timeout=1.0)
        except Exception:
            client.transport.abort()
        if not bridge._stopped:
            await bridge.stop()
