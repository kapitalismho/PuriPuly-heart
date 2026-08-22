from __future__ import annotations

import asyncio
import json
import logging

import pytest
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosedError

from puripuly_heart.core.overlay.bridge import OverlayBridge
from puripuly_heart.core.overlay.diagnostics import OverlayDiagnosticsRecorder
from puripuly_heart.core.overlay.protocol import (
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
async def test_overlay_bridge_stop_preserves_authenticated_connection_when_close_fails() -> None:
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

    with pytest.raises(RuntimeError, match="connection close failed"):
        await bridge.stop()

    assert connection in bridge._authenticated_connections

    await bridge.stop()

    assert connection.close_calls == 2
    assert connection not in bridge._authenticated_connections


@pytest.mark.asyncio
async def test_overlay_bridge_stop_preserves_server_when_wait_closed_fails() -> None:
    class FailingOnceServer:
        def __init__(self) -> None:
            self.close_calls = 0
            self.wait_closed_calls = 0

        def close(self) -> None:
            self.close_calls += 1

        async def wait_closed(self) -> None:
            self.wait_closed_calls += 1
            if self.wait_closed_calls == 1:
                raise RuntimeError("server close failed")

    bridge = OverlayBridge(session_token="expected-token")
    server = FailingOnceServer()
    bridge._server = server
    bridge.url = "ws://127.0.0.1:9999"
    bridge._token_consumed = True
    await bridge.messages.put({"type": "pending"})

    with pytest.raises(RuntimeError, match="server close failed"):
        await bridge.stop()

    assert bridge._server is server
    assert bridge.url == "ws://127.0.0.1:9999"
    assert bridge._token_consumed is True
    assert bridge.messages.qsize() == 1

    await bridge.stop()

    assert server.close_calls == 2
    assert server.wait_closed_calls == 2
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
async def test_overlay_bridge_broadcasts_every_refresh_marker_revision_and_cleanup() -> None:
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

    snapshot_payloads = [
        payload["payload"]
        for payload in connection.sent_payloads
        if payload.get("type") == "snapshot"
    ]
    assert [payload["revision"] for payload in snapshot_payloads] == [2, 3, 4]
    assert [
        (
            payload["blocks"][0]["primary_text"],
            payload["blocks"][0]["secondary_text"],
            payload["blocks"][1]["primary_text"],
        )
        for payload in snapshot_payloads
    ] == [
        ("peer translated text", "peer source text", "self source text"),
        ("peer translated text", "peer source text", "self source text"),
        ("peer translated text", "peer source text", "self source text"),
    ]
    assert [payload["blocks"][0].get("session_scope") for payload in snapshot_payloads] == [
        "session:peer|peer_presentation_refresh=1",
        "session:peer|peer_presentation_refresh=2",
        "session:peer",
    ]
    assert [payload["blocks"][1].get("session_scope") for payload in snapshot_payloads] == [
        "self_presentation_refresh=1",
        "self_presentation_refresh=2",
        None,
    ]


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
            snapshot = json.loads(await asyncio.wait_for(ws.recv(), timeout=0.5))
            runtime_control = json.loads(await asyncio.wait_for(ws.recv(), timeout=0.5))
    finally:
        await bridge.stop()

    assert snapshot["type"] == "snapshot"
    assert runtime_control == {
        "type": "runtime_control",
        "payload": {"logging_mode": "detailed"},
    }


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

    try:
        async with connect(bridge.url) as ws:
            await ws.send(json.dumps({"type": "auth", "session_token": "expected-token"}))
            await asyncio.wait_for(ws.recv(), timeout=0.5)
            await ws.close(code=4001, reason="client_bye")
            await asyncio.sleep(0.05)
    finally:
        await bridge.stop()

    events = list(diagnostics.bridge_events)
    assert [event["event"] for event in events] == [
        "connection_authenticated",
        "connection_closed",
        "connection_detached",
    ]
    closed = events[1]
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

    events = list(diagnostics.bridge_events)
    assert [event["event"] for event in events] == [
        "send_start",
        "send_failure",
        "send_finish",
    ]
    assert events[1]["removed"] is True
    assert events[1]["exception_type"]
    assert events[2]["stale_connections"] == 1
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

    events = list(recorder.bridge_events)
    assert [event["event"] for event in events[-2:]] == ["send_start", "send_finish"]
    assert events[-1]["revision"] == 2
    assert "elapsed_ms" in events[-1]
    dumped = json.dumps(events)
    assert "primary_text" not in dumped
    assert "secondary_text" not in dumped
