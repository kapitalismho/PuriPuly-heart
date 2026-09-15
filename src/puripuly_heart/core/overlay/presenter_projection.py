from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable
from uuid import UUID

from .diagnostics import OverlayDiagnosticsRecorder
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
from .state import OverlayLogicalTurnEntry

RuntimeDiagnosticLogger = Callable[[str], bool]
EntryKey = tuple[str, UUID]


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


@dataclass(slots=True)
class PresenterDiagnosticProjection:
    diagnostics: OverlayDiagnosticsRecorder | None = None
    runtime_log_diagnostic: Callable[..., bool] | None = None
    _visible_window_signature: tuple[object, ...] | None = None

    def reset(self) -> None:
        self._visible_window_signature = None

    def configure(
        self,
        *,
        diagnostics: OverlayDiagnosticsRecorder | None,
        runtime_log_diagnostic: Callable[..., bool] | None,
    ) -> None:
        self.diagnostics = diagnostics
        self.runtime_log_diagnostic = runtime_log_diagnostic

    def emit_lazy(self, build_message: Callable[[], str], *, level: int = logging.INFO) -> bool:
        logger = self.runtime_log_diagnostic
        if logger is None:
            return False
        owner = getattr(logger, "__self__", None)
        try:
            if owner is not None:
                emit = getattr(owner, "emit_diagnostic_lazy", None)
                if callable(emit):
                    return emit(build_message, level=level)
                emit = getattr(owner, "log_diagnostic_lazy", None)
                if callable(emit):
                    return emit(build_message, level=level)
            return logger(build_message(), level=level)
        except Exception:
            return False

    def emit_turn_decision(
        self,
        decision: str,
        *,
        disposition: str | None,
        key: EntryKey | None,
        entry: OverlayLogicalTurnEntry | None,
        block: OverlayPresentationBlock | None,
        extras: dict[str, object] | None,
        publishable: bool | None,
    ) -> bool:
        def build() -> str:
            resolved_key = key or (
                (entry.channel, entry.utterance_id) if entry is not None else None
            )
            parts = [f"decision={decision}"]
            if disposition is not None:
                parts.append(f"disposition={disposition}")
            if resolved_key is not None:
                parts.append(f"entry={self.format_key(resolved_key)}")
            if entry is not None:
                parts.extend(
                    [
                        f"channel={entry.channel}",
                        f"publishable={publishable}",
                        f"ever_visible={entry.ever_visible}",
                        "ever_visible_with_translation="
                        f"{entry.translation_observed_visible_since is not None}",
                        f"retained_hidden={entry.retained_hidden}",
                    ]
                )
            if block is not None:
                parts.extend(
                    [
                        f"block_variant={block.block_variant}",
                        f"primary_len={len(block.primary_text)}",
                        f"secondary_len={len(block.secondary_text)}",
                    ]
                )
            if extras is not None:
                parts.extend(f"{name}={value}" for name, value in extras.items())
            return f"[OverlayPresenter][Decision] {' '.join(parts)}"

        return self.emit_lazy(build)

    def emit_pair_state(
        self,
        key: EntryKey,
        entry: OverlayLogicalTurnEntry,
        block: OverlayPresentationBlock,
        *,
        publish_kind: str,
        rendered_sources: tuple[str, str],
        rendered_pair_state: str,
        elapsed_ms: int | None,
    ) -> bool:
        def build() -> str:
            primary_source, secondary_source = rendered_sources
            parts = [
                "[OverlayPresenter][PairState]",
                f"entry={self.format_key(key)}",
                f"channel={entry.channel}",
                f"block_variant={block.block_variant}",
                f"publish_kind={publish_kind}",
                f"update_id={block.update_id}",
                f"origin_wall_clock_ms={block.origin_wall_clock_ms}",
                f"source_text_hash={block.source_text_hash}",
                f"source_text_len={block.source_text_len}",
                f"original_seq={entry.original_seq}",
                f"translation_seq={entry.translation_seq}",
                f"rendered_pair_state={rendered_pair_state}",
                f"rendered_primary_source={primary_source}",
                f"rendered_secondary_source={secondary_source}",
                f"appearance_seq={block.appearance_seq}",
                f"primary_len={len(block.primary_text)}",
                f"secondary_len={len(block.secondary_text) if block.secondary_enabled else 0}",
            ]
            if elapsed_ms is not None:
                parts.append(f"elapsed_ms={elapsed_ms}")
            return " ".join(parts)

        return self.emit_lazy(build)

    def record_visible_window(
        self,
        *,
        active_self_present: bool,
        finalized_limit: int,
        candidate_keys: list[EntryKey],
        selected_keys: list[EntryKey],
        protected_selected: list[EntryKey],
        retained_hidden: list[EntryKey],
    ) -> None:
        if self.diagnostics is None:
            return
        candidate = [self.format_key(key) for key in candidate_keys]
        selected = [self.format_key(key) for key in selected_keys]
        dropped = [label for label in candidate if label not in selected]
        protected = [self.format_key(key) for key in protected_selected]
        hidden = [self.format_key(key) for key in retained_hidden]
        signature = (
            active_self_present,
            finalized_limit,
            tuple(candidate),
            tuple(selected),
            tuple(dropped),
            tuple(protected),
            tuple(hidden),
        )
        if signature == self._visible_window_signature:
            return
        self._visible_window_signature = signature
        self.diagnostics.record_presenter(
            "visible_window",
            active_self_present=active_self_present,
            finalized_limit=finalized_limit,
            candidate_keys=candidate,
            selected_keys=selected,
            dropped_keys=dropped,
            protected_selected=protected,
            retained_hidden=hidden,
        )

    def record_snapshot(
        self,
        snapshot: OverlayPresentationSnapshot,
        *,
        blocks: list[dict[str, object]],
        bridge_attached: bool,
    ) -> None:
        self.emit_lazy(
            lambda: (
                "[OverlayPresenter] Snapshot publish: revision=%s block_count=%s "
                "bridge_attached=%s blocks=%s"
                % (snapshot.revision, len(snapshot.blocks), bridge_attached, blocks)
            )
        )
        if self.diagnostics is not None:
            self.diagnostics.record_presenter(
                "snapshot_publish",
                revision=snapshot.revision,
                block_count=len(snapshot.blocks),
                bridge_attached=bridge_attached,
                blocks=blocks,
            )

    def record_deadline(
        self,
        entry: OverlayLogicalTurnEntry,
        *,
        visible_deadline: float | None,
        translation_deadline: float | None,
        effective_deadline: float | None,
    ) -> None:
        if self.diagnostics is None:
            return
        self.diagnostics.record_presenter(
            "deadline_scheduled",
            entry_key=self.format_key((entry.channel, entry.utterance_id)),
            channel=entry.channel,
            visible_since=entry.visible_since,
            translation_visible_since=entry.translation_visible_since,
            closed_at=entry.closed_at,
            visible_deadline=visible_deadline,
            translation_deadline=translation_deadline,
            effective_deadline=effective_deadline,
        )

    def record_removal(
        self,
        *,
        reason: str,
        key: EntryKey,
        entry: OverlayLogicalTurnEntry,
        now: float,
        visible_deadline: float | None,
        translation_deadline: float | None,
        effective_deadline: float | None,
        extra_fields: dict[str, object],
    ) -> None:
        if self.diagnostics is None:
            return
        self.diagnostics.record_presenter_removal(
            reason=reason,
            entry_key=self.format_key(key),
            appearance_seq=entry.appearance_seq,
            channel=entry.channel,
            primary_len=len(entry.original_text.strip()),
            secondary_len=len(entry.translation_text.strip()),
            visible_since=entry.visible_since,
            translation_visible_since=entry.translation_visible_since,
            closed_at=entry.closed_at,
            now=now,
            visible_deadline=visible_deadline,
            translation_deadline=translation_deadline,
            effective_deadline=effective_deadline,
            **extra_fields,
        )

    @staticmethod
    def format_key(key: EntryKey) -> str:
        return f"{key[0]}:{key[1]}"
