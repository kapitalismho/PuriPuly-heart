from __future__ import annotations

from dataclasses import dataclass, field

from .protocol import (
    NATIVE_SELF_SCENE_TARGET,
    U64_MAX,
    NativeFreshRenderGenerations,
    NativeFreshRenderTargets,
    NativeQuietTailEpisode,
    NativeQuietTailEpisodes,
    OverlayPresentationBlock,
    OverlayPresentationSnapshot,
)


@dataclass(slots=True)
class NativeRetryIntentProjection:
    enabled: bool = False
    generations: NativeFreshRenderGenerations = field(default_factory=NativeFreshRenderGenerations)
    targets: NativeFreshRenderTargets = field(default_factory=NativeFreshRenderTargets)
    episodes: NativeQuietTailEpisodes = field(default_factory=NativeQuietTailEpisodes)
    _episode_generations: dict[str, int] = field(default_factory=dict)

    def reset(self, *, enabled: bool = False) -> None:
        self.enabled = enabled
        self.generations = NativeFreshRenderGenerations()
        self.targets = NativeFreshRenderTargets()
        self.episodes = NativeQuietTailEpisodes()
        self._episode_generations.clear()

    def clear_scene(self) -> None:
        enabled = self.enabled
        generations = self.generations
        episode_generations = dict(self._episode_generations)
        self.reset(enabled=enabled)
        self.generations = generations
        self._episode_generations.update(episode_generations)

    def begin(self, *, enabled: bool, snapshot: OverlayPresentationSnapshot) -> None:
        self.reset(enabled=enabled)
        if not enabled:
            return
        generations: dict[str, int] = {}
        targets: dict[str, str] = {}
        episodes: dict[str, NativeQuietTailEpisode] = {}
        scene_phase = (
            "stream"
            if any(
                block.channel == "self"
                and block.block_variant == "active_self"
                and self._scene_drawable(block)
                for block in snapshot.blocks
            )
            else "final"
        )
        for block in snapshot.blocks:
            channel = block.channel
            if channel not in {"self", "peer"} or not (
                self._scene_drawable(block) if channel == "self" else block.primary_text.strip()
            ):
                continue
            generations[channel] = 1
            targets[channel] = NATIVE_SELF_SCENE_TARGET if channel == "self" else block.id
            episodes[channel] = NativeQuietTailEpisode(
                phase=(
                    scene_phase
                    if channel == "self"
                    else ("final" if block.block_variant == "finalized" else "stream")
                ),
                generation=1,
            )
        if (
            "self" not in targets
            and snapshot.native_fresh_render_targets is not None
            and snapshot.native_fresh_render_targets.self == NATIVE_SELF_SCENE_TARGET
            and snapshot.native_fresh_render_generations is not None
            and snapshot.native_fresh_render_generations.self is not None
            and snapshot.native_quiet_tail_episodes is not None
            and snapshot.native_quiet_tail_episodes.self is not None
            and any(self._scene_drawable(block) for block in snapshot.blocks)
        ):
            generations["self"] = 1
            targets["self"] = NATIVE_SELF_SCENE_TARGET
            episodes["self"] = NativeQuietTailEpisode(phase=scene_phase, generation=1)
        self.generations = NativeFreshRenderGenerations(
            self=generations.get("self"), peer=generations.get("peer")
        )
        self.targets = NativeFreshRenderTargets(self=targets.get("self"), peer=targets.get("peer"))
        self.episodes = NativeQuietTailEpisodes(
            self=episodes.get("self"), peer=episodes.get("peer")
        )
        self._episode_generations.update(generations)

    def reconcile_self_scene(
        self,
        previous_blocks: list[OverlayPresentationBlock],
        rendered_blocks: list[OverlayPresentationBlock],
    ) -> bool:
        if not self.enabled or self._self_signature(previous_blocks) == self._self_signature(
            rendered_blocks
        ):
            return False
        if not any(self._scene_drawable(block) for block in rendered_blocks):
            return False
        generation = self.next_self_generation(self.generations.self)
        episode_generation = self.next_self_generation(self._episode_generations.get("self"))
        self.generations = NativeFreshRenderGenerations(self=generation, peer=self.generations.peer)
        self.targets = NativeFreshRenderTargets(
            self=NATIVE_SELF_SCENE_TARGET, peer=self.targets.peer
        )
        self.episodes = NativeQuietTailEpisodes(
            self=NativeQuietTailEpisode(
                phase=self.self_scene_phase(rendered_blocks), generation=episode_generation
            ),
            peer=self.episodes.peer,
        )
        self._episode_generations["self"] = episode_generation
        return True

    @staticmethod
    def _scene_drawable(block: OverlayPresentationBlock) -> bool:
        return bool(
            block.primary_text.strip() or (block.secondary_enabled and block.secondary_text.strip())
        )

    @staticmethod
    def self_scene_phase(blocks: list[OverlayPresentationBlock]) -> str:
        return (
            "stream"
            if any(
                block.channel == "self"
                and block.block_variant == "active_self"
                and block.primary_text.strip()
                for block in blocks
            )
            else "final"
        )

    @staticmethod
    def _self_signature(blocks: list[OverlayPresentationBlock]) -> tuple[object, ...]:
        return tuple(
            (
                block.id,
                block.occupant_key,
                block.appearance_seq,
                block.block_variant,
                block.primary_text,
                block.secondary_text if block.secondary_enabled else "",
                block.primary_language,
                (
                    block.secondary_language
                    if block.secondary_enabled and block.secondary_text
                    else None
                ),
                block.speaker_style,
            )
            for block in blocks
            if block.channel == "self" and block.primary_text.strip()
        )

    def reconcile_peer_scene(
        self,
        previous_blocks: list[OverlayPresentationBlock],
        rendered_blocks: list[OverlayPresentationBlock],
        *,
        phases: dict[str, str],
    ) -> bool:
        if not self.enabled:
            return False
        previous = {
            block.id: self._peer_signature(block)
            for block in previous_blocks
            if block.channel == "peer" and block.primary_text.strip()
        }
        drawable = [
            block
            for block in rendered_blocks
            if block.channel == "peer" and block.primary_text.strip()
        ]
        changed = [
            block for block in drawable if previous.get(block.id) != self._peer_signature(block)
        ]
        if not changed and self.targets.peer not in {block.id for block in drawable}:
            changed = drawable
        if not changed:
            return False
        target = changed[-1]
        generation = self.next_generation(self._episode_generations.get("peer"))
        self.generations = NativeFreshRenderGenerations(
            self=self.generations.self, peer=self.next_generation(self.generations.peer)
        )
        self.targets = NativeFreshRenderTargets(self=self.targets.self, peer=target.id)
        self.episodes = NativeQuietTailEpisodes(
            self=self.episodes.self,
            peer=NativeQuietTailEpisode(
                phase=phases.get(target.id, "final"), generation=generation
            ),
        )
        self._episode_generations["peer"] = generation
        return True

    @staticmethod
    def _peer_signature(block: OverlayPresentationBlock) -> tuple[object, ...]:
        return (
            block.primary_text,
            block.secondary_text if block.secondary_enabled else "",
            block.primary_language,
            block.secondary_language if block.secondary_enabled else None,
            block.speaker_style,
        )

    def prune(self, rendered_blocks: list[OverlayPresentationBlock]) -> None:
        visible = {block.id for block in rendered_blocks if block.primary_text.strip()}
        self_target = (
            self.targets.self
            if (
                self.targets.self in visible
                or (
                    self.targets.self == NATIVE_SELF_SCENE_TARGET
                    and any(self._scene_drawable(block) for block in rendered_blocks)
                )
            )
            else None
        )
        peer_target = self.targets.peer if self.targets.peer in visible else None
        self.targets = NativeFreshRenderTargets(self=self_target, peer=peer_target)
        self.episodes = NativeQuietTailEpisodes(
            self=self.episodes.self if self_target is not None else None,
            peer=self.episodes.peer if peer_target is not None else None,
        )

    @staticmethod
    def next_generation(current: int | None) -> int:
        if current is None:
            return 1
        if current == U64_MAX:
            return 0
        return current + 1

    @staticmethod
    def next_self_generation(current: int | None) -> int:
        return 1 if current is None or current >= U64_MAX - 1 else current + 2
