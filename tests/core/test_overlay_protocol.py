from __future__ import annotations

from uuid import uuid4

import pytest

from puripuly_heart.core.overlay.protocol import (
    U64_MAX,
    NativeFreshRenderGenerations,
    NativeFreshRenderTargets,
    NativeQuietTailEpisode,
    NativeQuietTailEpisodes,
    OverlayPresentationBlock,
    OverlayPresentationCalibration,
    OverlayPresentationSnapshot,
)


def test_native_quiet_tail_episodes_are_optional_runtime_only_facts() -> None:
    legacy = OverlayPresentationSnapshot.from_dict({"revision": 1, "blocks": []})
    assert legacy.native_quiet_tail_episodes is None
    snapshot = OverlayPresentationSnapshot(
        native_quiet_tail_episodes=NativeQuietTailEpisodes(
            self=NativeQuietTailEpisode(phase="final", generation=4),
            peer=NativeQuietTailEpisode(phase="stream", generation=9),
        )
    )
    assert OverlayPresentationSnapshot.from_dict(snapshot.to_dict()) == snapshot
    with pytest.raises(ValueError):
        NativeQuietTailEpisode(phase="unknown", generation=1)


def test_native_fresh_render_generations_are_optional_and_round_trip_independently() -> None:
    legacy = OverlayPresentationSnapshot.from_dict({"revision": 1, "blocks": []})
    assert legacy.native_fresh_render_generations is None
    assert "native_fresh_render_generations" not in legacy.to_dict()

    snapshot = OverlayPresentationSnapshot(
        native_fresh_render_generations=NativeFreshRenderGenerations(self=3, peer=8)
    )
    assert OverlayPresentationSnapshot.from_dict(snapshot.to_dict()) == snapshot
    assert (
        "native_fresh_render_generations"
        not in OverlayPresentationSnapshot(
            native_fresh_render_generations=NativeFreshRenderGenerations()
        ).to_dict()
    )
    maximum = NativeFreshRenderGenerations(self=U64_MAX, peer=0)
    assert NativeFreshRenderGenerations.from_dict(maximum.to_dict()) == maximum


def test_native_fresh_render_targets_are_optional_and_backward_compatible() -> None:
    legacy = OverlayPresentationSnapshot.from_dict({"revision": 1, "blocks": []})
    assert legacy.native_fresh_render_targets is None
    snapshot = OverlayPresentationSnapshot(
        native_fresh_render_targets=NativeFreshRenderTargets(
            self="self:synthetic-a",
            peer="peer:synthetic-b",
        )
    )
    assert OverlayPresentationSnapshot.from_dict(snapshot.to_dict()) == snapshot
    with pytest.raises(ValueError):
        NativeFreshRenderTargets.from_dict({"self": " "})


@pytest.mark.parametrize(
    "value",
    [-1, U64_MAX + 1, True, 1.5, "1"],
)
def test_native_fresh_render_generations_reject_invalid_values(value: object) -> None:
    with pytest.raises(ValueError):
        OverlayPresentationSnapshot.from_dict({"native_fresh_render_generations": {"self": value}})


@pytest.mark.parametrize("value", [-1, U64_MAX + 1, True, 1.5, "1"])
def test_native_fresh_render_generations_reject_invalid_direct_values(value: object) -> None:
    with pytest.raises(ValueError):
        NativeFreshRenderGenerations(peer=value)  # type: ignore[arg-type]


from puripuly_heart.core.overlay.sink import (
    OverlayEventAdapter,
    PeerActiveUpdate,
    SelfActiveUpdate,
)
from puripuly_heart.domain.models import Translation


def test_overlay_presentation_snapshot_round_trips_blocks_and_calibration() -> None:
    snapshot = OverlayPresentationSnapshot(
        revision=7,
        calibration=OverlayPresentationCalibration(
            anchor="spatial_locked",
            offset_x=0.15,
            offset_y=-0.2,
            distance=1.1,
            text_scale=1.25,
            background_alpha=0.4,
        ),
        blocks=[
            OverlayPresentationBlock(
                id="self:1",
                occupant_key="self:1",
                appearance_seq=1,
                channel="self",
                block_variant="finalized",
                primary_text="hello",
                secondary_text="안녕",
                secondary_enabled=True,
            ),
            OverlayPresentationBlock(
                id="self:active",
                occupant_key="self:active",
                appearance_seq=2,
                channel="self",
                block_variant="active_self",
                primary_text="hola",
                secondary_text="",
                secondary_enabled=False,
            ),
        ],
    )

    restored = OverlayPresentationSnapshot.from_dict(snapshot.to_dict())

    assert restored.revision == 7
    assert restored.calibration.anchor == "spatial_locked"
    assert restored.calibration.distance == 1.1
    assert restored.blocks == [
        OverlayPresentationBlock(
            id="self:1",
            occupant_key="self:1",
            appearance_seq=1,
            channel="self",
            block_variant="finalized",
            primary_text="hello",
            secondary_text="안녕",
            secondary_enabled=True,
        ),
        OverlayPresentationBlock(
            id="self:active",
            occupant_key="self:active",
            appearance_seq=2,
            channel="self",
            block_variant="active_self",
            primary_text="hola",
            secondary_text="",
            secondary_enabled=False,
        ),
    ]


def test_overlay_presentation_block_round_trips_occupant_metadata() -> None:
    block = OverlayPresentationBlock(
        id="self:1234",
        occupant_key="self:1234",
        appearance_seq=7,
        channel="self",
        block_variant="finalized",
        primary_text="hello",
        secondary_text="안녕",
        secondary_enabled=True,
    )

    encoded = block.to_dict()

    assert encoded["occupant_key"] == "self:1234"
    assert encoded["appearance_seq"] == 7
    assert OverlayPresentationBlock.from_dict(encoded) == block


def test_overlay_presentation_block_round_trips_optional_content_languages() -> None:
    block = OverlayPresentationBlock(
        id="self:1234",
        occupant_key="self:1234",
        appearance_seq=7,
        channel="self",
        block_variant="finalized",
        primary_text="안녕하세요",
        secondary_text="hello",
        secondary_enabled=True,
        primary_language="ko",
        secondary_language="en",
    )

    encoded = block.to_dict()

    assert encoded["primary_language"] == "ko"
    assert encoded["secondary_language"] == "en"
    assert OverlayPresentationBlock.from_dict(encoded) == block


def test_overlay_presentation_block_defaults_missing_or_null_content_languages_to_none() -> None:
    restored = OverlayPresentationBlock.from_dict(
        {
            "id": "self:1",
            "occupant_key": "self:1",
            "appearance_seq": 1,
            "channel": "self",
            "block_variant": "finalized",
            "primary_text": "hello",
            "secondary_text": "",
            "secondary_enabled": False,
            "primary_language": None,
        }
    )

    assert restored.primary_language is None
    assert restored.secondary_language is None


def test_overlay_event_adapter_self_active_update_carries_occupant_key() -> None:
    adapter = OverlayEventAdapter()
    utterance_id = uuid4()

    event = adapter.self_active_update(
        text="hello live",
        utterance_id=utterance_id,
        occupant_key=f"self:{utterance_id}",
        created_at=11.0,
    )

    assert event.type == "self_active_update"
    assert event.utterance_id == utterance_id
    assert event.occupant_key == f"self:{utterance_id}"
    assert event.secondary_text == ""


def test_overlay_event_adapter_self_active_update_accepts_secondary_text() -> None:
    adapter = OverlayEventAdapter()
    utterance_id = uuid4()

    event = adapter.self_active_update(
        text="hello live",
        secondary_text="translated live",
        utterance_id=utterance_id,
        occupant_key=f"self:{utterance_id}",
        created_at=11.0,
    )

    assert event.type == "self_active_update"
    assert event.utterance_id == utterance_id
    assert event.occupant_key == f"self:{utterance_id}"
    assert event.secondary_text == "translated live"


def test_overlay_presentation_snapshot_round_trips_active_peer_block() -> None:
    block = OverlayPresentationBlock(
        id="peer:turn-1",
        occupant_key="peer:turn-1",
        appearance_seq=3,
        channel="peer",
        block_variant="active_peer",
        primary_text="",
        secondary_text="can you hear me",
        secondary_enabled=True,
    )
    snapshot = OverlayPresentationSnapshot(blocks=[block])

    restored = OverlayPresentationSnapshot.from_dict(snapshot.to_dict())

    assert restored.blocks == [block]


def test_overlay_presentation_block_rejects_self_active_peer_combination() -> None:
    with pytest.raises(ValueError, match="active_peer blocks require channel='peer'"):
        OverlayPresentationBlock.from_dict(
            {
                "id": "self:active-peer",
                "occupant_key": "self:active-peer",
                "appearance_seq": 1,
                "channel": "self",
                "block_variant": "active_peer",
                "primary_text": "hello",
                "secondary_text": "",
                "secondary_enabled": False,
            }
        )


def test_overlay_event_adapter_peer_active_update_carries_occupant_key() -> None:
    adapter = OverlayEventAdapter()
    utterance_id = uuid4()

    event = adapter.peer_active_update(
        text="peer live",
        utterance_id=utterance_id,
        occupant_key=f"peer:{utterance_id}",
        created_at=11.0,
    )

    assert event.type == "peer_active_update"
    assert event.channel == "peer"
    assert event.utterance_id == utterance_id
    assert event.occupant_key == f"peer:{utterance_id}"
    assert event.text == "peer live"


def test_peer_active_update_requires_peer_channel_and_utterance_id() -> None:
    utterance_id = uuid4()

    with pytest.raises(ValueError, match="PeerActiveUpdate requires channel='peer'"):
        PeerActiveUpdate(
            event_id="evt-1",
            seq=1,
            utterance_id=utterance_id,
            channel="self",
            created_at=10.0,
            text="peer live",
            occupant_key=f"peer:{utterance_id}",
        )

    with pytest.raises(ValueError, match="PeerActiveUpdate requires utterance_id"):
        PeerActiveUpdate(
            event_id="evt-2",
            seq=2,
            utterance_id=None,
            channel="peer",
            created_at=10.0,
            text="peer live",
            occupant_key=f"peer:{utterance_id}",
        )


def test_overlay_event_adapter_repeated_same_utterance_translation_updates_get_distinct_update_id() -> (
    None
):
    adapter = OverlayEventAdapter()
    utterance_id = uuid4()

    first = adapter.translation_final(
        utterance_id=utterance_id,
        channel="self",
        text="first",
        source_language="ko",
        target_language="en",
        applied_context_mode=None,
        created_at=11.0,
    )
    second = adapter.translation_final(
        utterance_id=utterance_id,
        channel="self",
        text="second",
        source_language="ko",
        target_language="en",
        applied_context_mode=None,
        created_at=12.0,
    )

    assert first.utterance_id == utterance_id
    assert second.utterance_id == utterance_id
    assert isinstance(first.update_id, str) and first.update_id
    assert isinstance(second.update_id, str) and second.update_id
    assert first.update_id != second.update_id


def test_overlay_presentation_block_round_trips_update_id_and_origin_wall_clock_ms() -> None:
    block = OverlayPresentationBlock(
        id="self:1234",
        occupant_key="self:1234",
        appearance_seq=7,
        channel="self",
        block_variant="finalized",
        primary_text="hello",
        secondary_text="안녕",
        secondary_enabled=True,
        update_id="upd-1234",
        origin_wall_clock_ms=1712345678901,
        session_scope="session:self",
        source_text_hash="abc123def456",
        source_text_len=5,
        logical_turn_key="self:1234",
    )

    encoded = block.to_dict()

    assert encoded["update_id"] == "upd-1234"
    assert encoded["origin_wall_clock_ms"] == 1712345678901
    assert encoded["session_scope"] == "session:self"
    assert encoded["source_text_hash"] == "abc123def456"
    assert encoded["source_text_len"] == 5
    assert encoded["logical_turn_key"] == "self:1234"
    assert OverlayPresentationBlock.from_dict(encoded) == block


def test_overlay_presentation_block_from_dict_defaults_missing_update_id_and_origin_wall_clock_ms_to_none() -> (
    None
):
    restored = OverlayPresentationBlock.from_dict(
        {
            "id": "self:1",
            "occupant_key": "self:1",
            "appearance_seq": 1,
            "channel": "self",
            "block_variant": "finalized",
            "primary_text": "hello",
            "secondary_text": "",
            "secondary_enabled": False,
        }
    )

    assert restored.update_id is None
    assert restored.origin_wall_clock_ms is None
    assert restored.session_scope is None
    assert restored.source_text_hash is None
    assert restored.source_text_len is None
    assert restored.logical_turn_key is None


def test_translation_text_alias_still_supports_legacy_constructor_shape_with_generated_update_id() -> (
    None
):
    translation = Translation(utterance_id=uuid4(), text="legacy")

    assert translation.text == "legacy"
    assert translation.translated_text == "legacy"
    assert isinstance(translation.update_id, str) and translation.update_id


def test_self_active_update_requires_utterance_id() -> None:
    utterance_id = uuid4()

    with pytest.raises(ValueError, match="SelfActiveUpdate requires utterance_id"):
        SelfActiveUpdate(
            event_id="evt-1",
            seq=1,
            utterance_id=None,
            channel="self",
            created_at=10.0,
            text="hello live",
            occupant_key=f"self:{utterance_id}",
        )


def test_overlay_presentation_snapshot_rejects_non_list_blocks() -> None:
    with pytest.raises(ValueError, match="blocks must be a list"):
        OverlayPresentationSnapshot.from_dict(
            {
                "revision": 1,
                "calibration": OverlayPresentationCalibration().to_dict(),
                "blocks": "not-a-list",
            }
        )


def test_overlay_presentation_snapshot_rejects_non_dict_block_items() -> None:
    with pytest.raises(ValueError, match="dict items"):
        OverlayPresentationSnapshot.from_dict(
            {
                "revision": 1,
                "calibration": OverlayPresentationCalibration().to_dict(),
                "blocks": ["not-a-dict"],
            }
        )


def test_overlay_presentation_snapshot_rejects_invalid_block_variant() -> None:
    with pytest.raises(ValueError, match="invalid overlay presentation block variant"):
        OverlayPresentationSnapshot.from_dict(
            {
                "revision": 1,
                "calibration": OverlayPresentationCalibration().to_dict(),
                "blocks": [
                    {
                        "id": "self:1",
                        "channel": "self",
                        "block_variant": "streaming",
                        "primary_text": "hello",
                        "secondary_text": "",
                        "secondary_enabled": True,
                    }
                ],
            }
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("id", 123, "id must be a string"),
        ("primary_text", 123, "primary_text must be a string"),
        ("secondary_text", 123, "secondary_text must be a string"),
        ("secondary_enabled", "true", "secondary_enabled must be a bool"),
    ],
)
def test_overlay_presentation_block_rejects_invalid_field_types(
    field: str,
    value: object,
    message: str,
) -> None:
    payload = {
        "id": "self:1",
        "occupant_key": "self:1",
        "appearance_seq": 1,
        "channel": "self",
        "block_variant": "finalized",
        "primary_text": "hello",
        "secondary_text": "",
        "secondary_enabled": True,
    }
    payload[field] = value

    with pytest.raises(ValueError, match=message):
        OverlayPresentationBlock.from_dict(payload)


def test_overlay_presentation_block_rejects_peer_active_self_combination() -> None:
    with pytest.raises(ValueError, match="active_self blocks require channel='self'"):
        OverlayPresentationBlock.from_dict(
            {
                "id": "peer:active",
                "channel": "peer",
                "block_variant": "active_self",
                "primary_text": "hello",
                "secondary_text": "",
                "secondary_enabled": False,
            }
        )


@pytest.mark.parametrize("payload", [None, [], "not-a-dict"])
def test_overlay_presentation_block_rejects_non_dict_payload(payload: object) -> None:
    with pytest.raises(ValueError, match="overlay presentation block must be an object"):
        OverlayPresentationBlock.from_dict(payload)  # type: ignore[arg-type]
