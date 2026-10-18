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
from .sink import (
    OverlayEventUnion,
    PeerTranscriptFinal,
    TranslationFinal,
)


@dataclass(slots=True)
class NativeRetryIntentProjection:
    enabled: bool = False
    generations: NativeFreshRenderGenerations = field(default_factory=NativeFreshRenderGenerations)
    targets: NativeFreshRenderTargets = field(default_factory=NativeFreshRenderTargets)
    episodes: NativeQuietTailEpisodes = field(default_factory=NativeQuietTailEpisodes)
    _episode_targets: dict[str, str] = field(default_factory=dict)
    _episode_generations: dict[str, int] = field(default_factory=dict)

    def reset(self, *, enabled: bool = False) -> None:
        self.enabled = enabled
        self.generations = NativeFreshRenderGenerations()
        self.targets = NativeFreshRenderTargets()
        self.episodes = NativeQuietTailEpisodes()
        self._episode_targets.clear()
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
        for block in snapshot.blocks:
            channel = block.channel
            if channel not in {"self", "peer"} or not block.primary_text.strip():
                continue
            generations[channel] = 1
            targets[channel] = NATIVE_SELF_SCENE_TARGET if channel == "self" else block.id
            episodes[channel] = NativeQuietTailEpisode(
                phase=(
                    self.self_scene_phase(snapshot.blocks)
                    if channel == "self"
                    else ("final" if block.block_variant == "finalized" else "stream")
                ),
                generation=1,
            )
        self.generations = NativeFreshRenderGenerations(
            self=generations.get("self"), peer=generations.get("peer")
        )
        self.targets = NativeFreshRenderTargets(self=targets.get("self"), peer=targets.get("peer"))
        self.episodes = NativeQuietTailEpisodes(
            self=episodes.get("self"), peer=episodes.get("peer")
        )
        self._episode_targets.update(targets)
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
        if not any(block.primary_text.strip() for block in rendered_blocks):
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
        self._episode_targets["self"] = NATIVE_SELF_SCENE_TARGET
        self._episode_generations["self"] = episode_generation
        return True

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

    def advance(self, channel: str, event: OverlayEventUnion | None) -> None:
        if channel != "peer" or event is None or event.utterance_id is None:
            return
        target = f"peer:{event.utterance_id}"
        self.generations = NativeFreshRenderGenerations(
            self=self.generations.self, peer=self.next_generation(self.generations.peer)
        )
        self.targets = NativeFreshRenderTargets(self=self.targets.self, peer=target)
        phase = "final" if isinstance(event, (PeerTranscriptFinal, TranslationFinal)) else "stream"
        current = self.episodes.peer
        if current is None or current.phase != phase or self._episode_targets.get("peer") != target:
            generation = self.next_generation(self._episode_generations.get("peer"))
            current = NativeQuietTailEpisode(phase=phase, generation=generation)
            self._episode_generations["peer"] = generation
        self._episode_targets["peer"] = target
        self.episodes = NativeQuietTailEpisodes(self=self.episodes.self, peer=current)

    def prune(self, rendered_blocks: list[OverlayPresentationBlock]) -> None:
        visible = {block.id for block in rendered_blocks if block.primary_text.strip()}
        self_target = (
            self.targets.self
            if (
                self.targets.self in visible
                or (self.targets.self == NATIVE_SELF_SCENE_TARGET and visible)
            )
            else None
        )
        peer_target = self.targets.peer if self.targets.peer in visible else None
        if self_target is None:
            self._episode_targets.pop("self", None)
        if peer_target is None:
            self._episode_targets.pop("peer", None)
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
