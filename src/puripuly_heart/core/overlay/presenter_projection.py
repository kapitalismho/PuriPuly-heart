from __future__ import annotations

from dataclasses import dataclass, field

from .protocol import (
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
    SelfTranscriptFinal,
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
        episode_generations = dict(self._episode_generations)
        self.reset(enabled=enabled)
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
            targets[channel] = block.id
            episodes[channel] = NativeQuietTailEpisode(
                phase="final" if block.block_variant == "finalized" else "stream",
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

    def advance(self, channel: str, event: OverlayEventUnion | None) -> None:
        if event is None or event.utterance_id is None:
            return
        target = f"{channel}:{event.utterance_id}"
        generations = self.generations
        targets = self.targets
        next_generation = self.next_generation(
            generations.self if channel == "self" else generations.peer
        )
        if channel == "self":
            self.generations = NativeFreshRenderGenerations(
                self=next_generation, peer=generations.peer
            )
            self.targets = NativeFreshRenderTargets(self=target, peer=targets.peer)
        else:
            self.generations = NativeFreshRenderGenerations(
                self=generations.self, peer=next_generation
            )
            self.targets = NativeFreshRenderTargets(self=targets.self, peer=target)
        self._advance_episode(channel, event)

    def prune(self, rendered_blocks: list[OverlayPresentationBlock]) -> None:
        visible = {block.id for block in rendered_blocks if block.primary_text.strip()}
        self_target = self.targets.self if self.targets.self in visible else None
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

    def _advance_episode(self, channel: str, event: OverlayEventUnion) -> None:
        phase = (
            "final"
            if isinstance(event, (SelfTranscriptFinal, PeerTranscriptFinal, TranslationFinal))
            else "stream"
        )
        current = self.episodes.self if channel == "self" else self.episodes.peer
        target = f"{channel}:{event.utterance_id}"
        current_target = self._episode_targets.get(channel)
        replace_episode = current is None or current.phase != phase or current_target != target
        if channel == "self" and isinstance(event, TranslationFinal):
            replace_episode = True
        if replace_episode:
            generation = self.next_generation(self._episode_generations.get(channel))
            episode = NativeQuietTailEpisode(phase=phase, generation=generation)
            self._episode_generations[channel] = generation
        else:
            episode = current
        self._episode_targets[channel] = target
        if channel == "self":
            self.episodes = NativeQuietTailEpisodes(self=episode, peer=self.episodes.peer)
        else:
            self.episodes = NativeQuietTailEpisodes(self=self.episodes.self, peer=episode)

    @staticmethod
    def next_generation(current: int | None) -> int:
        if current is None:
            return 1
        if current == U64_MAX:
            return 0
        return current + 1
