from __future__ import annotations

import json
from dataclasses import replace

import pytest

from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.overlay.bridge_mailbox import OverlayBridgeMailbox
from puripuly_heart.core.overlay.presenter_projection import NativeRetryIntentProjection
from puripuly_heart.core.overlay.protocol import (
    NATIVE_SELF_SCENE_TARGET,
    U64_MAX,
    NativeFreshRenderGenerations,
    NativeFreshRenderTargets,
    NativeQuietTailEpisode,
    NativeQuietTailEpisodes,
    OverlayPresentationBlock,
    OverlayPresentationSnapshot,
)


def block(identity: str, *, active: bool = False) -> OverlayPresentationBlock:
    channel = "peer" if identity.startswith("peer:") else "self"
    return OverlayPresentationBlock(
        id=identity,
        occupant_key=identity,
        appearance_seq=1,
        channel=channel,
        block_variant="active_self" if active else "finalized",
        primary_text="synthetic source",
        secondary_text="",
        secondary_enabled=True,
    )


def mailbox(clock: FakeClock) -> OverlayBridgeMailbox:
    return OverlayBridgeMailbox(
        initial_snapshot=OverlayPresentationSnapshot(),
        clock=clock,
        scene_byte_limit=65536,
        control_byte_limit=4096,
        control_slot_limit=8,
        delivery_receipt_limit=16,
    )


def snapshot(projection: NativeRetryIntentProjection, blocks: list, revision: int):
    return OverlayPresentationSnapshot(
        revision=revision,
        blocks=blocks,
        native_fresh_render_generations=projection.generations,
        native_fresh_render_targets=projection.targets,
        native_quiet_tail_episodes=projection.episodes,
    )


def test_self_milestones_rearm_both_identities_and_noops_keep_intent() -> None:
    projection = NativeRetryIntentProjection(enabled=True)
    first = block("self:A", active=True)
    assert projection.reconcile_self_scene([], [first])
    initial = snapshot(projection, [first], 1)
    assert initial.native_fresh_render_targets.self == NATIVE_SELF_SCENE_TARGET
    assert initial.native_quiet_tail_episodes.self == NativeQuietTailEpisode("stream", 1)
    metadata = replace(first, update_id="boundary-only", origin_wall_clock_ms=100)
    assert not projection.reconcile_self_scene([first], [metadata])
    assert snapshot(projection, [first], 1) == initial
    peer = block("peer:P")
    assert not projection.reconcile_self_scene([first], [peer, first])
    projection.prune([peer, first])
    assert projection.generations == initial.native_fresh_render_generations
    assert projection.episodes == initial.native_quiet_tail_episodes
    hidden = replace(first, secondary_enabled=False)
    assert not projection.reconcile_self_scene([first], [hidden])
    before_hidden = projection.episodes
    assert not projection.reconcile_self_scene(
        [hidden], [replace(hidden, secondary_text="not displayed", secondary_language="ko")]
    )
    assert projection.episodes == before_hidden
    grown = replace(hidden, primary_text="synthetic source extension")
    assert projection.reconcile_self_scene([hidden], [grown])
    assert projection.generations.self == 3
    assert projection.episodes.self == NativeQuietTailEpisode("stream", 3)
    equal_new_identity = replace(grown, id="self:B", occupant_key="self:B", appearance_seq=2)
    assert projection.reconcile_self_scene([grown], [equal_new_identity])
    assert projection.episodes.self.generation == 5
    styled = replace(equal_new_identity, block_variant="finalized")
    assert projection.reconcile_self_scene([equal_new_identity], [styled])
    assert projection.episodes.self == NativeQuietTailEpisode("final", 7)


def test_retained_patch_retirement_and_self_removal_cover_surviving_scene() -> None:
    projection = NativeRetryIntentProjection(enabled=True)
    a, b, peer = block("self:A"), block("self:B", active=True), block("peer:P")
    assert projection.reconcile_self_scene([], [a, b])
    patched = replace(a, secondary_text="synthetic translation")
    assert projection.reconcile_self_scene([a, b], [patched, b])
    assert projection.episodes.self.phase == "stream"
    assert projection.reconcile_self_scene([patched, b], [b])
    projection.prune([b])
    assert projection.targets.self == NATIVE_SELF_SCENE_TARGET
    assert projection.episodes.self.generation == 5
    assert projection.reconcile_self_scene([b, peer], [peer])
    projection.prune([peer])
    assert projection.targets.self == NATIVE_SELF_SCENE_TARGET
    assert projection.episodes.self == NativeQuietTailEpisode("final", 7)
    assert projection.generations.peer is None
    assert projection.targets.peer is None
    assert not projection.reconcile_self_scene([peer], [replace(peer, primary_text="PEER change")])
    projection.prune([])
    assert projection.targets.self is None
    assert projection.episodes.self is None


def test_mailbox_coalesces_latest_coverage_without_scene_backlog() -> None:
    clock = FakeClock(_now=10.0)
    outbox = mailbox(clock)
    projection = NativeRetryIntentProjection(enabled=True)
    a = block("self:A", active=True)
    previous = []
    for revision in range(1, 101):
        current = [replace(a, primary_text=f"synthetic revision {revision}")]
        assert projection.reconcile_self_scene(previous, current)
        outbox.admit_scene(snapshot(projection, current, revision), {a.id: 18.0})
        previous = current
    latest = outbox.take_scene_for_write()
    assert latest.snapshot.revision == 100
    assert latest.snapshot.native_quiet_tail_episodes.self.generation == 199
    assert outbox.pending_scene is None
    assert len(outbox.delivery_receipts) == 16
    assert outbox.retained_scene_bytes <= 65536
    assert all("synthetic revision" not in str(receipt) for receipt in outbox.delivery_receipts)


def test_send_time_self_expiry_rearms_survivor_once_and_next_milestone_is_distinct() -> None:
    clock = FakeClock(_now=10.0)
    outbox = mailbox(clock)
    projection = NativeRetryIntentProjection(enabled=True)
    a, b = block("self:A"), block("self:B", active=True)
    projection.reconcile_self_scene([], [a, b])
    envelope = outbox.make_scene(snapshot(projection, [a, b], 1), {a.id: 10.2, b.id: 18.0})
    clock.advance(0.8)
    sent = json.loads(outbox.revalidated_scene_message(envelope))["payload"]
    repeated = json.loads(outbox.revalidated_scene_message(envelope))["payload"]
    assert sent == repeated
    assert sent["native_fresh_render_generations"]["self"] == 2
    assert sent["native_quiet_tail_episodes"]["self"] == {"phase": "stream", "generation": 2}
    assert [row["id"] for row in sent["blocks"]] == [b.id]
    changed = replace(b, secondary_text="synthetic newer translation")
    projection.reconcile_self_scene([a, b], [changed])
    assert projection.generations.self == 3
    assert projection.episodes.self.generation == 3
    assert projection.generations.self != sent["native_fresh_render_generations"]["self"]
    clock.advance(10)
    empty = json.loads(outbox.revalidated_scene_message(envelope))["payload"]
    assert empty["blocks"] == []
    assert "native_quiet_tail_episodes" not in empty
    assert "native_fresh_render_targets" not in empty
    assert "native_fresh_render_generations" not in empty


def test_send_time_self_removal_preserves_peer_policy_and_wrap_namespaces() -> None:
    clock = FakeClock(_now=10.0)
    outbox = mailbox(clock)
    self_block, peer = block("self:A", active=True), block("peer:P")
    projection = NativeRetryIntentProjection(enabled=True)
    projection.generations = NativeFreshRenderGenerations(self=U64_MAX)
    projection.targets = NativeFreshRenderTargets(self=NATIVE_SELF_SCENE_TARGET)
    projection.episodes = NativeQuietTailEpisodes(self=NativeQuietTailEpisode("stream", U64_MAX))
    projection._episode_generations["self"] = U64_MAX
    envelope = outbox.make_scene(snapshot(projection, [self_block, peer], 1), {self_block.id: 10.1})
    clock.advance(1)
    sent = json.loads(outbox.revalidated_scene_message(envelope))["payload"]
    assert sent["blocks"] == [peer.to_dict()]
    assert sent["native_fresh_render_generations"] == {"self": 0}
    assert sent["native_quiet_tail_episodes"] == {"self": {"phase": "final", "generation": 0}}
    assert projection.reconcile_self_scene([self_block, peer], [peer])
    assert projection.generations.self == 1
    assert projection.episodes.self.generation == 1
    projection.begin(enabled=True, snapshot=OverlayPresentationSnapshot(blocks=[peer]))
    assert projection.generations.self is None
    assert projection.generations.peer == 1
    assert projection.targets.peer == peer.id
    assert projection.episodes.peer == NativeQuietTailEpisode("final", 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("peer_mode", ["secondary_only", "translated_primary"])
async def test_actual_presenter_and_mailbox_retained_patch_and_partial_removal(
    peer_mode: str,
) -> None:
    from tests.helpers.overlay_scene_protection import projected_scenes

    result = await projected_scenes(peer_mode=peer_mode)
    scenes = result["scenes"]
    a, b = scenes[3]["blocks"]
    assert a["secondary_text"] == "synthetic retained patch"
    assert b["primary_text"] == "synthetic B"
    assert result["coalesced"] == scenes[3]
    survivor = scenes[5]
    assert len(survivor["blocks"]) == 1
    assert survivor["blocks"][0]["id"] == b["id"]
    assert survivor["blocks"][0]["appearance_seq"] == b["appearance_seq"]
    assert survivor["native_fresh_render_targets"]["self"] == NATIVE_SELF_SCENE_TARGET
    mixed = scenes[6]
    assert len(mixed["blocks"]) == 2
    assert mixed["blocks"][1]["primary_text"] == (
        "" if peer_mode == "secondary_only" else "synthetic peer translation"
    )
    assert mixed["blocks"][1]["secondary_enabled"]
    assert mixed["blocks"][1]["secondary_text"] == "synthetic peer source"
    peer_only = scenes[7]
    assert peer_only["blocks"] == [mixed["blocks"][1]]
    assert peer_only["native_quiet_tail_episodes"]["self"]["phase"] == "final"
    if peer_mode == "secondary_only":
        assert "peer" not in peer_only["native_fresh_render_targets"]
        assert "peer" not in mixed["native_fresh_render_targets"]
        assert "peer" not in peer_only["native_quiet_tail_episodes"]
    else:
        assert (
            peer_only["native_fresh_render_targets"]["peer"]
            == mixed["native_fresh_render_targets"]["peer"]
        )
    pruned = result["pruned"]
    assert pruned == result["repeated_pruned"]
    assert pruned["blocks"] == peer_only["blocks"]
    assert pruned["native_fresh_render_generations"]["self"] % 2 == 0
    assert peer_only["native_fresh_render_generations"]["self"] % 2 == 1
    assert scenes[-1]["blocks"] == []
    assert "native_quiet_tail_episodes" not in scenes[-1]
    assert result["mailbox_pending_scenes"] == 1
    assert result["mailbox_receipts"] <= 16
    assert result["mailbox_retained_bytes"] <= 65536


def test_send_time_expiry_does_not_create_native_intent_for_desktop() -> None:
    clock = FakeClock(_now=10.0)
    outbox = mailbox(clock)
    a, peer = block("self:A"), block("peer:P")
    envelope = outbox.make_scene(
        OverlayPresentationSnapshot(revision=1, blocks=[a, peer]), {a.id: 10.1}
    )
    clock.advance(1)
    sent = json.loads(outbox.revalidated_scene_message(envelope))["payload"]
    assert sent["blocks"] == [peer.to_dict()]
    assert "native_fresh_render_generations" not in sent
    assert "native_fresh_render_targets" not in sent
    assert "native_quiet_tail_episodes" not in sent


def test_same_epoch_clear_preserves_trigger_high_water_but_fresh_epoch_resets() -> None:
    projection = NativeRetryIntentProjection(enabled=True)
    a, b = block("self:A", active=True), block("self:B", active=True)
    projection.reconcile_self_scene([], [a])
    first_trigger = projection.generations.self
    first_episode = projection.episodes.self.generation
    projection.clear_scene()
    assert projection.targets.self is None
    assert projection.episodes.self is None
    projection.reconcile_self_scene([], [b])
    assert projection.generations.self != first_trigger
    assert projection.episodes.self.generation != first_episode
    assert projection.generations.self == projection.episodes.self.generation == 3
    projection.begin(enabled=True, snapshot=OverlayPresentationSnapshot(blocks=[b]))
    assert projection.generations.self == 1
    assert projection.episodes.self == NativeQuietTailEpisode("stream", 1)


@pytest.mark.asyncio
async def test_presenter_clear_coalescing_and_fresh_epoch_seed_distinct_valid_scenes() -> None:
    from tests.helpers.overlay_scene_protection import projected_scenes

    result = await projected_scenes()
    prior = result["scenes"][7]
    after_clear = result["clear_rearm"]
    fresh_epoch = result["fresh_epoch"]
    assert result["scenes"][8]["blocks"] == []
    assert "native_fresh_render_targets" not in result["scenes"][8]
    assert "native_quiet_tail_episodes" not in result["scenes"][8]
    assert (
        after_clear["native_fresh_render_generations"]["self"]
        > prior["native_fresh_render_generations"]["self"]
    )
    assert after_clear["native_quiet_tail_episodes"]["self"]["phase"] == "stream"
    assert fresh_epoch["blocks"] == after_clear["blocks"]
    assert fresh_epoch["native_fresh_render_generations"]["self"] == 1
    assert fresh_epoch["native_quiet_tail_episodes"]["self"] == {
        "phase": "stream",
        "generation": 1,
    }
    assert result["empty_after_rearm"]["blocks"] == []


@pytest.mark.parametrize(
    ("secondary_enabled", "secondary_text", "drawable"),
    [(True, "source", True), (False, "source", False), (True, "", False)],
)
def test_scene_survivor_drawability_preserves_independent_peer_policy(
    secondary_enabled: bool, secondary_text: str, drawable: bool
) -> None:
    projection = NativeRetryIntentProjection(enabled=True)
    self_block = block("self:A", active=True)
    peer = replace(
        block("peer:P"),
        primary_text="",
        secondary_enabled=secondary_enabled,
        secondary_text=secondary_text,
    )
    projection.reconcile_self_scene([], [self_block, peer])
    assert projection.reconcile_self_scene([self_block, peer], [peer]) == drawable
    projection.prune([peer])
    assert (projection.targets.self == NATIVE_SELF_SCENE_TARGET) == drawable
    assert (projection.episodes.self is not None) == drawable
    assert projection.targets.peer is None
    assert projection.episodes.peer is None
    projection.begin(enabled=True, snapshot=OverlayPresentationSnapshot(blocks=[peer]))
    clock = FakeClock(_now=10.0)
    outbox = mailbox(clock)
    seeded = NativeRetryIntentProjection(enabled=True)
    seeded.reconcile_self_scene([], [self_block, peer])
    envelope = outbox.make_scene(snapshot(seeded, [self_block, peer], 1), {self_block.id: 10.1})
    clock.advance(1)
    sent = json.loads(outbox.revalidated_scene_message(envelope))["payload"]
    assert (
        sent.get("native_fresh_render_targets", {}).get("self") == NATIVE_SELF_SCENE_TARGET
    ) == drawable
    assert projection.targets.peer is None
    assert projection.episodes.peer is None
