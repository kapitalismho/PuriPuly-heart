from __future__ import annotations

import copy

import pytest

from tests.helpers.overlay_refresh_trace import build_overlay_refresh_trace_contract


def _without_anchor(snapshots: list[dict[str, object]]) -> list[dict[str, object]]:
    normalized = copy.deepcopy(snapshots)
    for snapshot in normalized:
        snapshot["calibration"]["anchor"] = "normalized"
        for block in snapshot["blocks"]:
            block.pop("update_id", None)
    return normalized


@pytest.mark.asyncio
async def test_production_presenter_refresh_trace_contract() -> None:
    contract = await build_overlay_refresh_trace_contract()
    traces = contract["traces"]

    for channel in ("peer", "self"):
        head = traces[f"{channel}_head_natural"]
        spatial = traces[f"{channel}_spatial_natural"]
        assert head["sleep_delays"] == [0.1] * 21
        assert spatial["sleep_delays"] == head["sleep_delays"]
        assert _without_anchor(spatial["snapshots"]) == _without_anchor(head["snapshots"])
        scopes = [snapshot["blocks"][-1].get("session_scope") for snapshot in spatial["snapshots"]]
        assert scopes[-22:] == [
            *(f"{channel}_presentation_refresh={nonce}" for nonce in range(1, 22)),
            None,
        ]

    lifecycle = traces["spatial_lifecycle"]
    assert [snapshot["calibration"]["anchor"] for snapshot in lifecycle["snapshots"]] == [
        "spatial_locked",
        "spatial_locked",
        "spatial_locked",
        "spatial_locked",
        "spatial_locked",
        "head_locked",
        "spatial_locked",
        "spatial_locked",
    ]
    assert lifecycle["snapshots"][5]["blocks"][-1]["session_scope"] == (
        "self_presentation_refresh=1"
    )
    assert lifecycle["snapshots"][6]["blocks"][-1]["session_scope"] == (
        "self_presentation_refresh=1"
    )
    assert lifecycle["snapshots"][-1]["blocks"][-1].get("session_scope") is None

    ownership = traces["spatial_ownership"]
    assert [
        snapshot.get("native_fresh_render_generations") for snapshot in ownership["snapshots"]
    ] == [None, None, None, None, {"peer": 1}, None, None, None]
    assert ownership["snapshots"][2]["blocks"][0]["session_scope"] == (
        "peer_presentation_refresh=1"
    )
    assert ownership["snapshots"][4]["blocks"][0].get("session_scope") is None
    assert ownership["snapshots"][6]["blocks"][0]["session_scope"] == (
        "peer_presentation_refresh=1"
    )
    assert ownership["snapshots"][-1]["blocks"][0].get("session_scope") is None
