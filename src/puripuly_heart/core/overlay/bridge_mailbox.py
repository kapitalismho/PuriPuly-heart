from __future__ import annotations

import json
from collections import OrderedDict, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from puripuly_heart.core.clock import Clock

from .protocol import NativeFreshRenderTargets, NativeQuietTailEpisodes, OverlayPresentationSnapshot


@dataclass(frozen=True, slots=True)
class SceneEnvelope:
    snapshot: OverlayPresentationSnapshot
    message: str
    block_expirations: Mapping[str, float | None]
    admitted_at: float


@dataclass(frozen=True, slots=True)
class ControlEnvelope:
    key: str
    message: str
    payload_type: str


@dataclass(frozen=True, slots=True)
class OverlayDeliveryReceipt:
    stage: str
    outcome: str
    scene_revision: int | None
    connection_epoch: int | None
    cause: str | None
    observed_at: float


@dataclass(frozen=True, slots=True)
class SceneAdmission:
    receipt: OverlayDeliveryReceipt
    stored_snapshot: OverlayPresentationSnapshot
    writer_required: bool


class OverlayBridgeMailbox:
    def __init__(
        self,
        *,
        initial_snapshot: OverlayPresentationSnapshot,
        clock: Clock,
        scene_byte_limit: int,
        control_byte_limit: int,
        control_slot_limit: int,
        delivery_receipt_limit: int,
    ) -> None:
        self._clock = clock
        self._scene_byte_limit = scene_byte_limit
        self._control_byte_limit = control_byte_limit
        self._control_slot_limit = control_slot_limit
        self.snapshot = initial_snapshot
        self.current_scene = self.make_scene(initial_snapshot, {})
        self.pending_scene: SceneEnvelope | None = None
        self.active_scene: SceneEnvelope | None = None
        self.pending_controls: OrderedDict[str, ControlEnvelope] = OrderedDict()
        self.replay_required = False
        self.initial_desktop_runtime_controls: list[dict[str, Any]] = []
        self.startup_barrier_epoch: int | None = None
        self.last_snapshot_revision = initial_snapshot.revision
        self.delivery_receipts: deque[OverlayDeliveryReceipt] = deque(maxlen=delivery_receipt_limit)

    @property
    def retained_scene_bytes(self) -> int:
        envelopes = {id(self.current_scene): self.current_scene}
        if self.active_scene is not None:
            envelopes[id(self.active_scene)] = self.active_scene
        if self.pending_scene is not None:
            envelopes[id(self.pending_scene)] = self.pending_scene
        return sum(len(envelope.message.encode("utf-8")) for envelope in envelopes.values())

    def make_scene(
        self,
        snapshot: OverlayPresentationSnapshot,
        block_expirations: Mapping[str, float | None],
        *,
        startup_runtime_controls: Iterable[Mapping[str, Any]] | None = None,
    ) -> SceneEnvelope:
        payload: dict[str, Any] = {
            "type": "snapshot",
            "payload": snapshot.to_dict(),
        }
        if startup_runtime_controls is not None:
            payload["startup_runtime_controls"] = [
                dict(control) for control in startup_runtime_controls
            ]
        message = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return SceneEnvelope(
            snapshot=snapshot,
            message=message,
            block_expirations=dict(block_expirations),
            admitted_at=self._clock.now(),
        )

    def rebuild_current(
        self,
        *,
        startup_runtime_controls: Iterable[Mapping[str, Any]] | None = None,
    ) -> None:
        self.current_scene = self.make_scene(
            self.current_scene.snapshot,
            self.current_scene.block_expirations,
            startup_runtime_controls=startup_runtime_controls,
        )

    def admit_scene(
        self,
        snapshot: OverlayPresentationSnapshot,
        block_expirations: Mapping[str, float | None],
        *,
        startup_runtime_controls: Iterable[Mapping[str, Any]] | None = None,
    ) -> SceneAdmission:
        if snapshot.revision <= self.last_snapshot_revision:
            return SceneAdmission(
                receipt=self.record_delivery(
                    outcome="superseded_product",
                    scene_revision=snapshot.revision,
                    cause="stale_revision",
                ),
                stored_snapshot=self.snapshot,
                writer_required=False,
            )
        self.last_snapshot_revision = snapshot.revision
        envelope = self.make_scene(
            snapshot,
            block_expirations,
            startup_runtime_controls=startup_runtime_controls,
        )
        if len(envelope.message.encode("utf-8")) > self._scene_byte_limit:
            safety_snapshot = replace(snapshot, blocks=[])
            envelope = self.make_scene(
                safety_snapshot,
                {},
                startup_runtime_controls=startup_runtime_controls,
            )
            self.snapshot = safety_snapshot
            self.current_scene = envelope
            self.pending_scene = envelope
            receipt = self.record_delivery(
                outcome="delivery_rejected",
                scene_revision=snapshot.revision,
                cause="scene_payload_exhausted",
            )
        else:
            self.snapshot = snapshot
            self.current_scene = envelope
            if self.pending_scene is not None:
                self.record_delivery(
                    outcome="superseded_product",
                    scene_revision=self.pending_scene.snapshot.revision,
                    cause="newer_scene",
                )
            self.pending_scene = envelope
            receipt = self.record_delivery(
                outcome="admitted",
                scene_revision=snapshot.revision,
                cause=None,
            )
        return SceneAdmission(
            receipt=receipt,
            stored_snapshot=self.snapshot,
            writer_required=True,
        )

    def enqueue_control(
        self,
        key: str,
        payload: Mapping[str, Any],
        *,
        terminal: bool = False,
    ) -> bool:
        message = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if len(message.encode("utf-8")) > self._control_byte_limit:
            raise ValueError("overlay control exceeds maximum size")
        if terminal and key not in self.pending_controls:
            while len(self.pending_controls) >= self._control_slot_limit:
                evicted_key = next(
                    (candidate for candidate in self.pending_controls if candidate != "shutdown"),
                    None,
                )
                if evicted_key is None:
                    break
                self.pending_controls.pop(evicted_key)
        nonterminal_limit = self._control_slot_limit - int("shutdown" in self.pending_controls)
        if (
            key not in self.pending_controls
            and not terminal
            and len(self.pending_controls) >= nonterminal_limit
        ):
            return False
        self.pending_controls.pop(key, None)
        self.pending_controls[key] = ControlEnvelope(
            key=key,
            message=message,
            payload_type=str(payload.get("type", key)),
        )
        return True

    def take_control(self, *, epoch: int) -> ControlEnvelope | None:
        shutdown = self.pending_controls.pop("shutdown", None)
        if shutdown is not None:
            return shutdown
        if self.startup_barrier_epoch == epoch:
            return None
        if not self.pending_controls:
            return None
        _, control = self.pending_controls.popitem(last=False)
        return control

    def abandon_startup_scene(self, *, cause: str) -> None:
        self.startup_barrier_epoch = None
        self.replay_required = False
        scene = self.pending_scene
        self.pending_scene = None
        if scene is not None:
            self.record_delivery(
                outcome="delivery_rejected",
                scene_revision=scene.snapshot.revision,
                cause=cause,
            )

    def take_scene_for_write(self) -> SceneEnvelope | None:
        if self.replay_required:
            self.replay_required = False
            self.pending_scene = None
            return self.current_scene
        scene = self.pending_scene
        self.pending_scene = None
        return scene

    def revalidated_scene_message(
        self,
        envelope: SceneEnvelope,
        *,
        startup_runtime_controls: Iterable[Mapping[str, Any]] | None = None,
    ) -> str:
        now = self._clock.now()
        valid_blocks = [
            block
            for block in envelope.snapshot.blocks
            if envelope.block_expirations.get(block.id) is None
            or now < envelope.block_expirations[block.id]
        ]
        if len(valid_blocks) == len(envelope.snapshot.blocks):
            return envelope.message
        visible = {block.id for block in valid_blocks}
        targets = envelope.snapshot.native_fresh_render_targets
        self_target = targets.self if targets is not None and targets.self in visible else None
        peer_target = targets.peer if targets is not None and targets.peer in visible else None
        episodes = envelope.snapshot.native_quiet_tail_episodes
        snapshot = replace(
            envelope.snapshot,
            blocks=valid_blocks,
            native_fresh_render_targets=NativeFreshRenderTargets(
                self=self_target,
                peer=peer_target,
            ),
            native_quiet_tail_episodes=NativeQuietTailEpisodes(
                self=(episodes.self if episodes is not None and self_target is not None else None),
                peer=(episodes.peer if episodes is not None and peer_target is not None else None),
            ),
        )
        return self.make_scene(
            snapshot,
            {},
            startup_runtime_controls=startup_runtime_controls,
        ).message

    def record_delivery(
        self,
        *,
        outcome: str,
        scene_revision: int | None,
        cause: str | None,
        connection_epoch: int | None = None,
    ) -> OverlayDeliveryReceipt:
        stages = {
            "written": "transport_written",
            "ambiguous": "ambiguous_receipt",
            "superseded_product": "superseded_product",
        }
        receipt = OverlayDeliveryReceipt(
            stage=stages.get(outcome, "delivery_admitted"),
            outcome=outcome,
            scene_revision=scene_revision,
            connection_epoch=connection_epoch,
            cause=cause,
            observed_at=self._clock.now(),
        )
        self.delivery_receipts.append(receipt)
        return receipt
