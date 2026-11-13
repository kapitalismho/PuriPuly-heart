from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from uuid import UUID, uuid5

from puripuly_heart.core.overlay.peer_caption_layout import peer_translation_fragments
from puripuly_heart.core.overlay.protocol import (
    NativeFreshRenderGenerations,
    NativeFreshRenderTargets,
    NativeQuietTailEpisodes,
    OverlayPresentationBlock,
    OverlayPresentationCalibration,
    OverlayPresentationSnapshot,
    SemanticRetirementFrontier,
)
from puripuly_heart.core.overlay.sink import (
    PeerActiveUpdate,
    PeerTranscriptFinal,
    SelfActiveClear,
    SelfActiveUpdate,
    SelfTranscriptFinal,
    TranslationFinal,
    TranslationStreamUpdate,
    UtteranceClosed,
)
from puripuly_heart.domain.models import SpeakerAssignment

OverlayEntryKey = tuple[str, UUID]
NextAppearanceSeq = Callable[[], int]
OverlayTerminalUpdatePredicate = Callable[[str | None, UUID | None], bool]
OverlayTerminalUpdateReason = Callable[[str | None, UUID | None], str | None]

_RETIRED_PREVIEW_SELF_SEQ_LIMIT = 64
PEER_FRAGMENT_REGISTRATION_BYTES = 384
PEER_FRAGMENT_DEPARTURE_BYTES = 64


@dataclass(slots=True)
class OverlayLogicalTurnEntry:
    channel: str
    utterance_id: UUID
    first_input_seq: int | None = None
    live_text: str = ""
    live_secondary_text: str = ""
    live_primary_language: str | None = None
    live_secondary_language: str | None = None
    live_update_id: str | None = None
    live_origin_wall_clock_ms: int | None = None
    live_session_scope: str | None = None
    live_source_text_hash: str | None = None
    live_source_text_len: int | None = None
    live_logical_turn_key: str | None = None
    live_seq: int | None = None
    original_text: str = ""
    original_language: str | None = None
    original_seq: int | None = None
    translation_text: str = ""
    translation_language: str | None = None
    translation_update_id: str | None = None
    translation_origin_wall_clock_ms: int | None = None
    translation_session_scope: str | None = None
    translation_source_text_hash: str | None = None
    translation_source_text_len: int | None = None
    translation_logical_turn_key: str | None = None
    speaker_assignment: SpeakerAssignment | None = None
    speaker_style: str | None = None
    speaker_gray_reason: str | None = None
    speaker_handoff_scope: tuple[str, str] | None = None
    translation_seq: int | None = None
    occupant_key: str = ""
    appearance_seq: int | None = None
    publishable_seq: int | None = None
    ever_publishable: bool = False
    ever_visible: bool = False
    visible_since: float | None = None
    last_meaningful_visible_at: float | None = None
    translation_visible_since: float | None = None
    translation_observed_visible_since: float | None = None
    last_updated_seq: int = 0
    closed_seq: int | None = None
    closed_at: float | None = None
    retained_hidden: bool = False
    window_evicted_at: float | None = None
    expiration_revision: int = 0
    caption_revision: int = 0
    source_closed: bool = False
    caption_staged: bool = False
    peer_caption_revision: int = -1
    peer_caption_final: bool = False
    peer_source_only: bool = False
    peer_fragments: tuple[str, ...] = ()
    peer_partition_text: str | None = None
    peer_partition_scale: float = 1.0
    peer_publication_id: str | None = None
    peer_parent_id: UUID | None = None
    peer_fragment_index: int = 0
    peer_departed_indices: set[int] = field(default_factory=set)
    peer_pending: bool = False
    peer_departed: bool = False
    peer_registered_at: float | None = None
    peer_promoted_at: float | None = None
    peer_protected_until: float | None = None
    peer_publication_generation: int | None = None
    peer_is_current: Callable[[], bool] | None = None
    peer_partition_is_source: bool = False
    peer_source_index: int | None = None
    peer_translation_index: int | None = None
    peer_source_departed_indices: set[int] = field(default_factory=set)
    peer_translation_departed_indices: set[int] = field(default_factory=set)
    peer_dormant: bool = False
    peer_registration_count: int = 0
    peer_last_visible_at: float | None = None
    peer_cached_at: float | None = None

    @property
    def block_id(self) -> str:
        return f"{self.channel}:{self.utterance_id}"


@dataclass(slots=True)
class PeerFragmentRegistration:
    parent: OverlayLogicalTurnEntry
    utterance_id: UUID
    peer_fragment_index: int
    appearance_seq: int
    peer_registered_at: float
    channel: str = field(init=False, default="peer")
    peer_pending: bool = True
    peer_departed: bool = False
    ever_visible: bool = False
    visible_since: float | None = None
    last_meaningful_visible_at: float | None = None
    peer_promoted_at: float | None = None
    peer_protected_until: float | None = None
    expiration_revision: int = 0
    retained_hidden: bool = False
    window_evicted_at: float | None = None
    closed_at: float | None = None
    closed_seq: int | None = None
    caption_staged: bool = False
    peer_source_index: int | None = None
    peer_translation_index: int | None = None
    peer_dormant: bool = False

    @property
    def block_id(self) -> str:
        return f"peer:{self.utterance_id}"

    @property
    def occupant_key(self) -> str:
        return self.block_id

    @property
    def first_input_seq(self) -> int:
        return self.appearance_seq

    @property
    def publishable_seq(self) -> int:
        return self.appearance_seq

    @property
    def last_updated_seq(self) -> int:
        return self.parent.last_updated_seq

    @property
    def peer_parent_id(self) -> UUID:
        return self.parent.utterance_id

    @property
    def peer_caption_final(self) -> bool:
        return self.parent.peer_caption_final

    @property
    def peer_caption_revision(self) -> int:
        return self.parent.peer_caption_revision


@dataclass(frozen=True, slots=True)
class PeerFragmentBinding:
    index: int
    source_index: int | None
    translation_index: int | None
    dormant: bool


@dataclass(frozen=True, slots=True)
class PeerFragmentPlan:
    fragments: tuple[str, ...]
    text: str
    source: bool
    bindings: tuple[PeerFragmentBinding, ...]
    removed: frozenset[int]
    departed: frozenset[int]
    source_departed: frozenset[int]
    translation_departed: frozenset[int]

    @property
    def retained_bytes(self) -> int:
        return (
            sum(len(fragment.encode("utf-8")) for fragment in self.fragments)
            + len(self.bindings) * PEER_FRAGMENT_REGISTRATION_BYTES
            + (len(self.departed) + len(self.source_departed) + len(self.translation_departed))
            * PEER_FRAGMENT_DEPARTURE_BYTES
        )


@dataclass(frozen=True, slots=True)
class ActiveSelfOverlayMetadata:
    text: str
    secondary_text: str
    utterance_id: UUID
    occupant_key: str
    update_id: str | None
    origin_wall_clock_ms: int | None
    session_scope: str | None
    source_text_hash: str | None
    source_text_len: int | None
    logical_turn_key: str | None
    primary_language: str | None = None
    secondary_language: str | None = None


@dataclass(frozen=True, slots=True)
class OverlayEntryRemovalRecord:
    key: OverlayEntryKey
    entry: OverlayLogicalTurnEntry | PeerFragmentRegistration
    reason: str
    now: float | None
    tombstone_seq: int | None


@dataclass(frozen=True, slots=True)
class OverlayReductionResult:
    changed: bool

    def __bool__(self) -> bool:
        return self.changed


OverlayPresentationEntry = OverlayLogicalTurnEntry | PeerFragmentRegistration


@dataclass(frozen=True)
class OverlayVisibleBlockSelection:
    rendered_entries: list[tuple[OverlayEntryKey, OverlayPresentationBlock]]
    active_self_present: bool
    finalized_limit: int
    candidate_keys: list[OverlayEntryKey]
    selected_keys: list[OverlayEntryKey]
    protected_keys: list[OverlayEntryKey]


@dataclass
class OverlayPresentationState:
    """Pure presentation state for overlay snapshots.

    This object intentionally does not own async tasks, sleeps, timers,
    cancellation, bridge I/O, provider I/O, OpenVR I/O, native overlay I/O,
    or dashboard side effects.
    """

    entries: dict[OverlayEntryKey, OverlayLogicalTurnEntry] = field(default_factory=dict)
    peer_registrations: dict[OverlayEntryKey, PeerFragmentRegistration] = field(
        default_factory=dict
    )
    retired_preview_self_seqs: OrderedDict[OverlayEntryKey, int] = field(
        default_factory=OrderedDict
    )
    live_self_turn_key: OverlayEntryKey | None = None
    live_peer_turn_key: OverlayEntryKey | None = None
    peer_text_scale: float = 1.0
    _pending_removals: list[OverlayEntryRemovalRecord] = field(default_factory=list)
    _snapshot: OverlayPresentationSnapshot = field(default_factory=OverlayPresentationSnapshot)

    def snapshot(self) -> OverlayPresentationSnapshot:
        return self._snapshot

    def active_self_overlay_metadata(self) -> ActiveSelfOverlayMetadata | None:
        live_key = self.live_self_turn_key
        if live_key is None:
            return None
        entry = self.entries.get(live_key)
        if entry is None or not entry.live_text:
            return None
        return ActiveSelfOverlayMetadata(
            text=entry.live_text,
            secondary_text=entry.live_secondary_text,
            utterance_id=entry.utterance_id,
            occupant_key=entry.occupant_key,
            update_id=entry.live_update_id,
            origin_wall_clock_ms=entry.live_origin_wall_clock_ms,
            session_scope=entry.live_session_scope,
            source_text_hash=entry.live_source_text_hash,
            source_text_len=entry.live_source_text_len,
            logical_turn_key=entry.live_logical_turn_key,
            primary_language=entry.live_primary_language,
            secondary_language=entry.live_secondary_language,
        )

    def drain_pending_removals(self) -> list[OverlayEntryRemovalRecord]:
        removals = self._pending_removals
        self._pending_removals = []
        return removals

    def entry_for(
        self,
        channel: str | None,
        utterance_id: UUID | None,
    ) -> OverlayLogicalTurnEntry:
        key = self.entry_key(channel, utterance_id)
        entry = self.entries.get(key)
        if entry is None:
            entry = OverlayLogicalTurnEntry(channel=key[0], utterance_id=key[1])
            self.entries[key] = entry
        return entry

    def entry_key(self, channel: str | None, utterance_id: UUID | None) -> OverlayEntryKey:
        if channel not in ("self", "peer"):
            raise ValueError(f"invalid overlay channel: {channel!r}")
        if utterance_id is None:
            raise ValueError("overlay presenter requires utterance_id for finalized entries")
        return (channel, utterance_id)

    def live_turn_key_for_channel(self, channel: str) -> OverlayEntryKey | None:
        if channel == "self":
            return self.live_self_turn_key
        if channel == "peer":
            return self.live_peer_turn_key
        raise ValueError(f"invalid overlay channel: {channel!r}")

    def set_live_turn_key_for_channel(
        self,
        channel: str,
        key: OverlayEntryKey | None,
    ) -> None:
        if channel == "self":
            self.live_self_turn_key = key
            return
        if channel == "peer":
            self.live_peer_turn_key = key
            return
        raise ValueError(f"invalid overlay channel: {channel!r}")

    def live_entry_for_channel(
        self,
        channel: str,
    ) -> tuple[OverlayEntryKey, OverlayLogicalTurnEntry] | None:
        live_key = self.live_turn_key_for_channel(channel)
        if live_key is None:
            return None
        entry = self.entries.get(live_key)
        if entry is None:
            self.set_live_turn_key_for_channel(channel, None)
            return None
        return live_key, entry

    def display_entry(self, key: OverlayEntryKey) -> OverlayPresentationEntry | None:
        return self.entries.get(key) or self.peer_registrations.get(key)

    def logical_entry(self, key: OverlayEntryKey) -> OverlayLogicalTurnEntry | None:
        entry = self.display_entry(key)
        return entry.parent if isinstance(entry, PeerFragmentRegistration) else entry

    def display_entries(self) -> dict[OverlayEntryKey, OverlayPresentationEntry]:
        return self.entries | {
            key: entry
            for key, entry in self.peer_registrations.items()
            if not entry.peer_pending and not entry.peer_departed and not entry.peer_dormant
        }

    def peer_fragment_plan(
        self, event: object, *, translation_enabled: bool = True
    ) -> PeerFragmentPlan | None:
        if not isinstance(event, TranslationStreamUpdate) or event.channel != "peer":
            return None
        source_only = (
            isinstance(event, TranslationFinal)
            and event.peer_caption_revision > 0
            and not event.text.strip()
        )
        if not event.text.strip() and not source_only:
            return None
        entry = self.entries.get(("peer", event.utterance_id))
        source_text = (
            event.source_text
            if event.source_text.strip()
            else ("" if entry is None else entry.original_text or entry.live_text)
        )
        return self.peer_view_plan(
            entry,
            source_text=source_text,
            translation_text=event.text,
            source_view=not translation_enabled or source_only,
            revision=True,
        )

    def peer_view_plan(
        self,
        entry: OverlayLogicalTurnEntry | None,
        *,
        source_text: str,
        translation_text: str,
        source_view: bool,
        revision: bool,
    ) -> PeerFragmentPlan:
        source_view = source_view or not translation_text.strip()
        source_parts = peer_translation_fragments(source_text, text_scale=self.peer_text_scale)
        translation_parts = peer_translation_fragments(
            translation_text, text_scale=self.peer_text_scale
        )
        fragments = source_parts if source_view else translation_parts
        mode = 0 if source_view else 1
        counts = len(source_parts), len(translation_parts)
        records = (
            {}
            if entry is None
            else {
                candidate.peer_fragment_index: candidate
                for candidate in (entry, *self.peer_registrations.values())
                if (candidate is entry or candidate.peer_parent_id == entry.utterance_id)
                and not candidate.peer_departed
            }
        )
        mappings = {
            index: [candidate.peer_source_index, candidate.peer_translation_index]
            for index, candidate in records.items()
        }
        if entry is None:
            mappings[0] = [None, None]
        departed = set() if entry is None else set(entry.peer_departed_indices)
        mode_departed = [
            set() if entry is None else set(entry.peer_source_departed_indices),
            set() if entry is None else set(entry.peer_translation_departed_indices),
        ]
        old_mode = mode if entry is None else (0 if entry.peer_partition_is_source else 1)
        removed = set()
        if revision or old_mode == mode:
            for index, mapping in mappings.items():
                original = tuple(mapping)
                for view in (0, 1):
                    if mapping[view] is not None and mapping[view] >= counts[view]:
                        if index in records and records[index].peer_dormant:
                            mode_departed[view].add(mapping[view])
                        mapping[view] = None
                candidate = records.get(index)
                if (
                    candidate is not None
                    and not candidate.peer_dormant
                    and old_mode == mode
                    and original[mode] is not None
                    and original[mode] >= len(fragments)
                ):
                    removed.add(index)
        active = {
            mapping[mode]: index
            for index, mapping in mappings.items()
            if index not in removed
            and mapping[mode] is not None
            and mapping[mode] < len(fragments)
            and mapping[mode] not in mode_departed[mode]
        }
        visible_ids = {block.id for block in self._snapshot.blocks}
        first_pending = min(
            (
                index
                for index, candidate in records.items()
                if candidate.peer_pending and not candidate.peer_dormant
            ),
            key=lambda index: records[index].appearance_seq or 0,
            default=None,
        )
        has_visible = any(candidate.block_id in visible_ids for candidate in records.values())
        reusable = sorted(
            (
                index
                for index, candidate in records.items()
                if index not in removed
                and index not in active.values()
                and (
                    candidate.block_id in visible_ids
                    or (not has_visible and index == first_pending)
                )
            ),
            key=lambda index: records[index].appearance_seq or 0,
        )
        next_index = max((*mappings, *departed, -1)) + 1
        for position in range(len(fragments)):
            if position in active or position in mode_departed[mode]:
                continue
            if reusable:
                index = reusable.pop(0)
            elif (
                0 in mappings
                and 0 not in removed
                and mappings[0] == [None, None]
                and (entry is None or entry.peer_registered_at is None)
            ):
                index = 0
            else:
                index = next_index
                next_index += 1
                mappings[index] = [None, None]
            mappings[index][mode] = position
            active[position] = index
        active_indices = set(active.values())
        if revision:
            removed.update(
                index
                for index, candidate in records.items()
                if not candidate.peer_dormant and index not in active_indices
            )
        for index, mapping in mappings.items():
            if index not in active_indices and mapping == [None, None]:
                removed.add(index)
        for index in removed:
            departed.add(index)
            candidate = records.get(index)
            if candidate is not None:
                for view, position in enumerate(
                    (candidate.peer_source_index, candidate.peer_translation_index)
                ):
                    if position is not None:
                        mode_departed[view].add(position)
        return PeerFragmentPlan(
            fragments=fragments,
            text=source_text if source_view else translation_text,
            source=source_view,
            bindings=tuple(
                PeerFragmentBinding(index, mapping[0], mapping[1], index not in active_indices)
                for index, mapping in sorted(mappings.items())
                if index not in removed
            ),
            removed=frozenset(removed),
            departed=frozenset(departed),
            source_departed=frozenset(mode_departed[0]),
            translation_departed=frozenset(mode_departed[1]),
        )

    def remove_entry(
        self,
        key: OverlayEntryKey,
        *,
        reason: str,
        now: float | None = None,
        tombstone_seq: int | None = None,
        retire_parent: bool = True,
    ) -> OverlayEntryRemovalRecord | None:
        entry = self.display_entry(key)
        if entry is not None and entry.channel == "peer":
            parent = self.logical_entry(key)
            parent_id = parent.utterance_id
            parent.peer_departed_indices.add(entry.peer_fragment_index)
            if entry.peer_source_index is not None:
                parent.peer_source_departed_indices.add(entry.peer_source_index)
            if entry.peer_translation_index is not None:
                parent.peer_translation_departed_indices.add(entry.peer_translation_index)
            if not entry.peer_departed and entry.peer_registered_at is not None:
                parent.peer_registration_count -= 1
            if entry is parent and any(
                candidate.peer_parent_id == parent_id
                for candidate in self.peer_registrations.values()
            ):
                entry.peer_departed = True
                entry.peer_pending = False
                entry.peer_protected_until = None
                if retire_parent and not any(
                    child.parent is parent and not child.peer_dormant
                    for child in self.peer_registrations.values()
                ):
                    self.remove_peer_caption(
                        ("peer", parent_id),
                        reason=reason,
                        now=now,
                        seq=parent.last_updated_seq,
                    )
                return None
        if self.live_self_turn_key == key:
            self.live_self_turn_key = None
        if self.live_peer_turn_key == key:
            self.live_peer_turn_key = None
        entry = self.entries.pop(key, None) or self.peer_registrations.pop(key, None)
        if entry is None:
            return None
        record = OverlayEntryRemovalRecord(
            key=key,
            entry=entry,
            reason=reason,
            now=now,
            tombstone_seq=tombstone_seq,
        )
        self._pending_removals.append(record)
        if entry.peer_parent_id is not None:
            parent_key = ("peer", entry.peer_parent_id)
            parent = self.entries.get(parent_key)
            if (
                retire_parent
                and parent is not None
                and (parent.peer_departed or parent.peer_dormant)
                and not any(
                    candidate.parent is parent and not candidate.peer_dormant
                    for candidate in self.peer_registrations.values()
                )
            ):
                self.remove_peer_caption(
                    parent_key, reason=reason, now=now, seq=parent.last_updated_seq
                )
        return record

    def remove_peer_caption(
        self, key: OverlayEntryKey, *, reason: str, now: float | None, seq: int
    ) -> None:
        for child_key, child in tuple(self.peer_registrations.items()):
            if child.peer_parent_id == key[1]:
                self.remove_entry(
                    child_key, reason=reason, now=now, tombstone_seq=seq, retire_parent=False
                )
        self.remove_entry(key, reason=reason, now=now, tombstone_seq=seq, retire_parent=False)

    def reconcile_peer_fragments(
        self,
        entry: OverlayLogicalTurnEntry,
        *,
        now: float,
        next_appearance_seq: NextAppearanceSeq,
        fragments: PeerFragmentPlan | None = None,
        translation_enabled: bool = True,
    ) -> None:
        if ("peer", entry.utterance_id) not in self.entries:
            return
        plan = fragments or self.peer_view_plan(
            entry,
            source_text=entry.original_text or entry.live_text,
            translation_text=entry.translation_text,
            source_view=not translation_enabled or entry.peer_source_only,
            revision=False,
        )
        if entry.peer_cached_at is None:
            entry.peer_cached_at = now
        entry.peer_fragments = plan.fragments
        entry.peer_partition_text = plan.text
        entry.peer_partition_scale = self.peer_text_scale
        entry.peer_partition_is_source = plan.source
        for binding in plan.bindings:
            key = (
                "peer",
                (
                    entry.utterance_id
                    if binding.index == 0
                    else uuid5(entry.utterance_id, f"fragment:{binding.index}")
                ),
            )
            item = self.display_entry(key)
            if item is None:
                item = PeerFragmentRegistration(
                    parent=entry,
                    utterance_id=key[1],
                    peer_fragment_index=binding.index,
                    appearance_seq=next_appearance_seq(),
                    peer_registered_at=now,
                )
                self.peer_registrations[key] = item
                entry.peer_registration_count += 1
            elif binding.index == 0 and entry.peer_registered_at is None:
                visible = next(
                    (block for block in self._snapshot.blocks if block.id == entry.block_id), None
                )
                sequence = (
                    entry.appearance_seq
                    if visible is not None
                    and (visible.primary_text.strip() or visible.block_variant == "active_peer")
                    else next_appearance_seq()
                )
                entry.peer_pending = visible is None
                entry.peer_registered_at = now
                entry.appearance_seq = sequence
                entry.publishable_seq = sequence
                entry.occupant_key = entry.block_id
                entry.visible_since = None
                entry.last_meaningful_visible_at = None
                entry.closed_at = None
                entry.peer_registration_count += 1
            if item.peer_dormant and not binding.dormant:
                item.peer_pending = True
            item.peer_source_index = binding.source_index
            item.peer_translation_index = binding.translation_index
            item.peer_dormant = binding.dormant
        for index in plan.removed:
            key = (
                "peer",
                (
                    entry.utterance_id
                    if index == 0
                    else uuid5(entry.utterance_id, f"fragment:{index}")
                ),
            )
            if index == 0 and entry.peer_registered_at is None and not plan.bindings:
                entry.peer_dormant = True
                continue
            self.remove_entry(
                key, reason="revision_tail_removed", now=now, tombstone_seq=entry.last_updated_seq
            )
        entry.peer_departed_indices.update(plan.departed)
        entry.peer_source_departed_indices.update(plan.source_departed)
        entry.peer_translation_departed_indices.update(plan.translation_departed)
        entry.peer_registration_count = len(plan.bindings)

    def apply_self_caption_update(
        self,
        event: SelfActiveUpdate,
        *,
        now: float,
        show_translation: bool,
        show_peer_original: bool,
        translation_enabled: bool,
        next_appearance_seq: NextAppearanceSeq,
        visible_window_target_blocks: int,
        terminal_update_reason: OverlayTerminalUpdateReason,
    ) -> OverlayReductionResult:
        key = self.entry_key(event.channel, event.utterance_id)
        if terminal_update_reason(event.channel, event.utterance_id) is not None:
            return OverlayReductionResult(False)
        entry = self.entries.get(key)
        if event.retained_only and (entry is None or entry.retained_hidden):
            return OverlayReductionResult(False)
        if entry is not None and (
            event.caption_revision <= entry.caption_revision or event.seq < entry.last_updated_seq
        ):
            return OverlayReductionResult(False)
        if event.caption_cancelled:
            self.remove_entry(key, reason="caption_cancelled", now=now, tombstone_seq=event.seq)
            return OverlayReductionResult(entry is not None)
        if entry is None:
            entry = self.entry_for(event.channel, event.utterance_id)
        previous_translation = self._rendered_self_translation_text(entry)
        previous_payload = (
            entry.original_text,
            entry.translation_text,
            entry.original_language,
            entry.translation_language,
            entry.source_closed,
            self.live_self_turn_key == key,
            entry.caption_staged,
        )
        if event.secondary_text.strip():
            for old_key, old_entry in tuple(self.entries.items()):
                if (
                    old_key != key
                    and old_entry.caption_staged
                    and (old_entry.first_input_seq or 0) < (entry.first_input_seq or event.seq)
                ):
                    self.remove_entry(
                        old_key,
                        reason="evicted_by_newer_turn",
                        now=now,
                        tombstone_seq=old_entry.last_updated_seq,
                    )
        self._remember_entry_input_seq(entry, event_seq=event.seq)
        if not entry.occupant_key:
            entry.occupant_key = event.occupant_key
        if not entry.source_closed:
            entry.original_text = event.text
            entry.original_language = _line_language(event.source_language, event.text)
            entry.original_seq = event.seq
        entry.source_closed = bool(event.caption_source_closed)
        self._refresh_entry_visibility_and_expiration(
            key,
            entry,
            now=now,
            publishable_seq=event.seq,
            next_appearance_seq=next_appearance_seq,
        )
        entry.translation_text = event.secondary_text
        entry.translation_language = _line_language(event.target_language, event.secondary_text)
        entry.translation_update_id = event.update_id
        entry.translation_origin_wall_clock_ms = event.origin_wall_clock_ms
        entry.translation_session_scope = event.session_scope
        entry.translation_source_text_hash = event.source_text_hash
        entry.translation_source_text_len = event.source_text_len
        entry.translation_logical_turn_key = event.logical_turn_key
        entry.translation_seq = event.seq if event.secondary_text.strip() else None
        staged = not event.secondary_text.strip() and self._self_caption_would_displace_translation(
            key,
            entry,
            activate=not event.caption_complete,
            show_translation=show_translation,
            show_peer_original=show_peer_original,
            translation_enabled=translation_enabled,
            capacity=visible_window_target_blocks,
        )
        if entry.caption_staged and event.retained_only and not event.secondary_text.strip():
            staged = True
        entry.caption_staged = staged
        if not event.retained_only and not staged and self.live_self_turn_key != key:
            self._clear_live_self_pointer(
                reason="live_self_replaced", now=now, show_translation=show_translation
            )
        if not event.retained_only and not staged:
            self.live_self_turn_key = key
        if self.live_self_turn_key == key or staged:
            if event.caption_complete:
                self._clear_self_live_payload(entry)
                if self.live_self_turn_key == key:
                    self.live_self_turn_key = None
            else:
                entry.live_text = entry.original_text
                entry.live_primary_language = entry.original_language
                entry.live_secondary_text = entry.translation_text
                entry.live_secondary_language = entry.translation_language
                entry.live_update_id = event.update_id
                entry.live_origin_wall_clock_ms = event.origin_wall_clock_ms
                entry.live_session_scope = event.session_scope
                entry.live_source_text_hash = event.source_text_hash
                entry.live_source_text_len = event.source_text_len
                entry.live_logical_turn_key = event.logical_turn_key
                entry.live_seq = event.seq
        if event.caption_complete and entry.closed_at is None:
            entry.closed_at = now
            entry.closed_seq = event.seq
        entry.caption_revision = event.caption_revision
        entry.last_updated_seq = event.seq
        self._update_self_translation_visibility(
            entry,
            previous_rendered_text=previous_translation,
            next_rendered_text=self._rendered_self_translation_text(entry),
            now=now,
            show_translation=show_translation,
        )
        return OverlayReductionResult(
            previous_payload
            != (
                entry.original_text,
                entry.translation_text,
                entry.original_language,
                entry.translation_language,
                entry.source_closed,
                self.live_self_turn_key == key,
                entry.caption_staged,
            )
        )

    def promote_staged_self(
        self,
        *,
        now: float,
        show_translation: bool,
        show_peer_original: bool,
        translation_enabled: bool,
        capacity: int,
    ) -> None:
        staged = [(key, entry) for key, entry in self.entries.items() if entry.caption_staged]
        if not staged:
            return
        key, entry = max(staged, key=lambda item: item[1].first_input_seq or 0)
        if self._self_caption_would_displace_translation(
            key,
            entry,
            activate=entry.closed_seq is None,
            show_translation=show_translation,
            show_peer_original=show_peer_original,
            translation_enabled=translation_enabled,
            capacity=capacity,
        ):
            return
        for old_key, old_entry in staged:
            if old_key != key:
                self.remove_entry(
                    old_key,
                    reason="evicted_by_newer_turn",
                    now=now,
                    tombstone_seq=old_entry.last_updated_seq,
                )
        entry.caption_staged = False
        if entry.closed_seq is None:
            self._clear_live_self_pointer(
                reason="live_self_replaced", now=now, show_translation=show_translation
            )
            self.live_self_turn_key = key

    def _self_caption_would_displace_translation(
        self,
        key: OverlayEntryKey,
        entry: OverlayLogicalTurnEntry,
        *,
        activate: bool,
        show_translation: bool,
        show_peer_original: bool,
        translation_enabled: bool,
        capacity: int,
    ) -> bool:
        if not show_translation or any(
            block.id == entry.block_id for block in self._snapshot.blocks
        ):
            return False
        drawable_ids = {
            value.block_id
            for value in self.entries.values()
            if not value.caption_staged and not value.retained_hidden
        }
        translated_ids = {
            block.id
            for block in self._snapshot.blocks
            if block.channel == "self"
            and block.secondary_enabled
            and block.secondary_text.strip()
            and block.id in drawable_ids
        }
        if not translated_ids:
            return False
        candidate = replace(
            entry,
            caption_staged=False,
            live_text=entry.original_text if activate else "",
        )
        entries = {**self.entries, key: candidate}
        live_self_entry = (key, candidate) if activate else None
        live_peer_entry = (
            (self.live_peer_turn_key, entries[self.live_peer_turn_key])
            if self.live_peer_turn_key in entries
            else None
        )
        selection = self.visible_block_selection(
            entries=entries,
            live_self_entry=live_self_entry,
            live_peer_entry=live_peer_entry,
            visible_window_target_blocks=capacity,
            show_translation=show_translation,
            show_peer_original=show_peer_original,
            translation_enabled=translation_enabled,
            next_appearance_seq=None,
        )
        return not translated_ids.issubset(block.id for _, block in selection.rendered_entries)

    def apply_self_active_update(
        self,
        event: SelfActiveUpdate,
        *,
        now: float,
        show_translation: bool,
        terminal_update_reason: OverlayTerminalUpdateReason,
    ) -> OverlayReductionResult:

        key = self.entry_key(event.channel, event.utterance_id)
        retired_preview_seq = self.retired_preview_self_seqs.get(key)
        if retired_preview_seq is not None and event.seq <= retired_preview_seq:
            return OverlayReductionResult(False)
        active_entry = self._active_update_entry_or_none(
            channel=event.channel,
            utterance_id=event.utterance_id,
            event_seq=event.seq,
            now=now,
            show_translation=show_translation,
            terminal_update_reason=terminal_update_reason,
        )
        if active_entry is None:
            return OverlayReductionResult(False)
        key, entry = active_entry

        previous_rendered_translation_text = self._rendered_self_translation_text(entry)

        if self._active_update_matches_live_payload(
            channel=event.channel,
            key=key,
            entry=entry,
            text=event.text,
            secondary_text=event.secondary_text,
            occupant_key=event.occupant_key,
            update_id=event.update_id,
            origin_wall_clock_ms=event.origin_wall_clock_ms,
            session_scope=event.session_scope,
            source_text_hash=event.source_text_hash,
            source_text_len=event.source_text_len,
            logical_turn_key=event.logical_turn_key,
        ):
            next_primary_language = _line_language(event.source_language, event.text)
            next_secondary_language = _line_language(event.target_language, event.secondary_text)
            language_changed = (
                entry.live_primary_language != next_primary_language
                or entry.live_secondary_language != next_secondary_language
            )
            entry.live_primary_language = next_primary_language
            entry.live_secondary_language = next_secondary_language

            entry.last_updated_seq = event.seq
            return OverlayReductionResult(language_changed)

        self._remember_entry_input_seq(entry, event_seq=event.seq)
        if not entry.occupant_key:
            entry.occupant_key = event.occupant_key
        if entry.visible_since is None:
            entry.visible_since = now

        if retired_preview_seq is not None and event.seq > retired_preview_seq:
            self.retired_preview_self_seqs.pop(key, None)
        entry.live_text = event.text
        entry.live_primary_language = _line_language(event.source_language, event.text)
        entry.live_seq = event.seq
        entry.original_seq = event.seq
        entry.live_secondary_text = event.secondary_text
        if event.secondary_text.strip():
            entry.live_secondary_language = _line_language(
                event.target_language,
                event.secondary_text,
            )
            entry.live_update_id = event.update_id
            entry.live_origin_wall_clock_ms = event.origin_wall_clock_ms
            entry.live_session_scope = event.session_scope
            entry.live_source_text_hash = event.source_text_hash
            entry.live_source_text_len = event.source_text_len
            entry.live_logical_turn_key = event.logical_turn_key
            entry.translation_seq = event.seq
        else:
            entry.live_secondary_language = None
            entry.live_update_id = None
            entry.live_origin_wall_clock_ms = None
            entry.live_session_scope = None
            entry.live_source_text_hash = None
            entry.live_source_text_len = None
            entry.live_logical_turn_key = None
            if not entry.translation_text.strip():
                entry.translation_seq = None
        self._update_self_translation_visibility(
            entry,
            previous_rendered_text=previous_rendered_translation_text,
            next_rendered_text=self._rendered_self_translation_text(entry),
            now=now,
            show_translation=show_translation,
        )
        entry.last_updated_seq = event.seq
        self.set_live_turn_key_for_channel(event.channel, key)
        return OverlayReductionResult(True)

    def apply_self_active_clear(
        self,
        event: SelfActiveClear,
        *,
        now: float,
        show_translation: bool,
    ) -> OverlayReductionResult:

        live_self = self.live_entry_for_channel("self")
        if live_self is None:
            return OverlayReductionResult(False)
        key, entry = live_self
        if event.seq < entry.last_updated_seq:
            return OverlayReductionResult(False)
        if not entry.live_text:
            self.live_self_turn_key = None
            entry.last_updated_seq = event.seq

            return OverlayReductionResult(False)

        previous_rendered_translation_text = self._rendered_self_translation_text(entry)
        self._clear_self_live_payload(entry)
        self._update_self_translation_visibility(
            entry,
            previous_rendered_text=previous_rendered_translation_text,
            next_rendered_text=self._rendered_self_translation_text(entry),
            now=now,
            show_translation=show_translation,
        )
        entry.last_updated_seq = event.seq
        if self.live_self_turn_key == key:
            self.live_self_turn_key = None
        if self._should_retire_preview_only_self_entry(entry):
            self._retire_preview_only_self_entry(
                key,
                entry,
                reason="live_self_cleared",
                now=now,
            )
        return OverlayReductionResult(True)

    def apply_self_finalized_update(
        self,
        event: SelfTranscriptFinal,
        *,
        now: float,
        show_translation: bool,
        next_appearance_seq: NextAppearanceSeq,
        terminal_update_reason: OverlayTerminalUpdateReason,
    ) -> OverlayReductionResult:

        key = self.entry_key(event.channel, event.utterance_id)
        if terminal_update_reason(event.channel, event.utterance_id) is not None:
            return OverlayReductionResult(False)
        entry = self.entry_for(event.channel, event.utterance_id)
        if entry.retained_hidden:
            return OverlayReductionResult(False)
        if event.seq < entry.last_updated_seq:
            return OverlayReductionResult(False)
        event_source_language = _content_language_or_none(event.source_language)
        if (
            entry.original_text == event.text
            and entry.original_language == event_source_language
            and entry.last_updated_seq == event.seq
        ):
            return OverlayReductionResult(False)

        previous_rendered_translation_text = self._rendered_self_translation_text(entry)
        self._remember_entry_input_seq(entry, event_seq=event.seq)
        entry.original_text = event.text
        entry.original_language = event_source_language
        entry.original_seq = event.seq
        entry.last_updated_seq = event.seq
        if self.live_self_turn_key == key:
            promoted_secondary_text = entry.live_secondary_text.strip()
            if promoted_secondary_text:
                entry.translation_text = promoted_secondary_text
                entry.translation_language = entry.live_secondary_language
                entry.translation_update_id = entry.live_update_id
                entry.translation_origin_wall_clock_ms = entry.live_origin_wall_clock_ms
                entry.translation_session_scope = entry.live_session_scope
                entry.translation_source_text_hash = entry.live_source_text_hash
                entry.translation_source_text_len = entry.live_source_text_len
                entry.translation_logical_turn_key = entry.live_logical_turn_key
                if entry.live_seq is not None:
                    entry.translation_seq = entry.live_seq
                if show_translation:
                    if entry.translation_visible_since is None:
                        entry.translation_visible_since = now
                    if entry.translation_observed_visible_since is None:
                        entry.translation_observed_visible_since = now
            self._clear_self_live_payload(entry)
            self.live_self_turn_key = None
        self._update_self_translation_visibility(
            entry,
            previous_rendered_text=previous_rendered_translation_text,
            next_rendered_text=self._rendered_self_translation_text(entry),
            now=now,
            show_translation=show_translation,
        )
        self.retired_preview_self_seqs.pop(key, None)
        self._refresh_entry_visibility_and_expiration(
            key,
            entry,
            now=now,
            publishable_seq=event.seq,
            next_appearance_seq=next_appearance_seq,
        )
        return OverlayReductionResult(True)

    def apply_self_translation_update(
        self,
        event: TranslationStreamUpdate | TranslationFinal,
        *,
        now: float,
        show_translation: bool,
        next_appearance_seq: NextAppearanceSeq,
        terminal_update_reason: OverlayTerminalUpdateReason,
    ) -> OverlayReductionResult:

        key = self.entry_key(event.channel, event.utterance_id)
        if terminal_update_reason(event.channel, event.utterance_id) is not None:
            return OverlayReductionResult(False)
        entry = self.entry_for(event.channel, event.utterance_id)
        if event.seq < entry.last_updated_seq:
            return OverlayReductionResult(False)
        event_source_language = _content_language_or_none(event.source_language)
        event_target_language = _content_language_or_none(event.target_language)
        if (
            entry.translation_text == event.text
            and (event_source_language is None or entry.original_language == event_source_language)
            and entry.translation_language == event_target_language
            and entry.last_updated_seq == event.seq
        ):
            return OverlayReductionResult(False)
        if event_source_language is not None:
            entry.original_language = event_source_language
        previous_rendered_translation_text = self._rendered_self_translation_text(entry)
        self._remember_entry_input_seq(entry, event_seq=event.seq)
        if entry.retained_hidden and event.text.strip():
            entry.retained_hidden = False
            entry.window_evicted_at = None
        entry.translation_text = event.text
        if event.text.strip():
            entry.translation_language = event_target_language
            entry.translation_update_id = event.update_id
            entry.translation_origin_wall_clock_ms = event.origin_wall_clock_ms
            entry.translation_session_scope = event.session_scope
            entry.translation_source_text_hash = event.source_text_hash
            entry.translation_source_text_len = event.source_text_len
            entry.translation_logical_turn_key = event.logical_turn_key
            entry.translation_seq = event.seq
        else:
            entry.translation_language = None
            entry.translation_update_id = None
            entry.translation_origin_wall_clock_ms = None
            entry.translation_session_scope = None
            entry.translation_source_text_hash = None
            entry.translation_source_text_len = None
            entry.translation_logical_turn_key = None
            if not entry.live_secondary_text.strip():
                entry.translation_seq = None
        if not entry.retained_hidden:
            self._update_self_translation_visibility(
                entry,
                previous_rendered_text=previous_rendered_translation_text,
                next_rendered_text=self._rendered_self_translation_text(entry),
                now=now,
                show_translation=show_translation,
            )
        entry.last_updated_seq = event.seq
        self.retired_preview_self_seqs.pop(key, None)
        self._refresh_entry_visibility_and_expiration(
            key,
            entry,
            now=now,
            publishable_seq=event.seq,
            next_appearance_seq=next_appearance_seq,
        )
        return OverlayReductionResult(True)

    def apply_self_utterance_closed(
        self,
        event: UtteranceClosed,
        *,
        now: float,
        is_tombstoned: OverlayTerminalUpdatePredicate,
    ) -> OverlayReductionResult:

        key = self.entry_key(event.channel, event.utterance_id)
        if is_tombstoned(event.channel, event.utterance_id):
            return OverlayReductionResult(False)
        entry = self.entries.get(key)
        if entry is None:
            return OverlayReductionResult(False)
        if event.seq < entry.last_updated_seq:
            return OverlayReductionResult(False)
        if entry.closed_seq == event.seq:
            return OverlayReductionResult(False)
        entry.closed_seq = event.seq
        entry.closed_at = now
        entry.last_updated_seq = event.seq
        return OverlayReductionResult(True)

    # Product decision: peer overlay text is emitted with translation arrival. Peer source-only/active updates must not become visible normal-flow rows.
    def apply_peer_active_update(
        self,
        event: PeerActiveUpdate,
        *,
        now: float,
        show_peer_original: bool,
        next_appearance_seq: NextAppearanceSeq,
        terminal_update_reason: OverlayTerminalUpdateReason,
        translation_enabled: bool = True,
    ) -> OverlayReductionResult:
        """Apply reserved peer active fallback state without normalizing it as product flow."""

        active_entry = self._active_update_entry_or_none(
            channel=event.channel,
            utterance_id=event.utterance_id,
            event_seq=event.seq,
            now=now,
            show_translation=True,
            terminal_update_reason=terminal_update_reason,
        )
        if active_entry is None:
            return OverlayReductionResult(False)
        key, entry = active_entry
        if self._active_update_matches_live_payload(
            channel=event.channel,
            key=key,
            entry=entry,
            text=event.text,
            secondary_text="",
            occupant_key=event.occupant_key,
            update_id=event.update_id,
            origin_wall_clock_ms=event.origin_wall_clock_ms,
            session_scope=event.session_scope,
            source_text_hash=event.source_text_hash,
            source_text_len=event.source_text_len,
            logical_turn_key=event.logical_turn_key,
        ):
            next_original_language = _line_language(event.source_language, event.text)
            language_changed = entry.original_language != next_original_language
            entry.original_language = next_original_language

            entry.last_updated_seq = event.seq
            return OverlayReductionResult(language_changed)

        self._remember_entry_input_seq(entry, event_seq=event.seq)
        if not entry.occupant_key:
            entry.occupant_key = event.occupant_key
        entry.live_text = event.text
        entry.original_text = event.text
        entry.original_language = _line_language(event.source_language, event.text)
        entry.live_seq = event.seq
        entry.original_seq = event.seq
        entry.last_updated_seq = event.seq
        self._refresh_entry_visibility_and_expiration(
            key,
            entry,
            now=now,
            publishable_seq=event.seq,
            next_appearance_seq=next_appearance_seq,
            show_peer_original=show_peer_original,
            translation_enabled=translation_enabled,
        )
        self.set_live_turn_key_for_channel(event.channel, key)
        return OverlayReductionResult(True)

    def apply_peer_finalized_update(
        self,
        event: PeerTranscriptFinal,
        *,
        now: float,
        show_peer_original: bool,
        next_appearance_seq: NextAppearanceSeq,
        terminal_update_reason: OverlayTerminalUpdateReason,
        translation_enabled: bool = True,
    ) -> OverlayReductionResult:

        key = self.entry_key(event.channel, event.utterance_id)
        if terminal_update_reason(event.channel, event.utterance_id) is not None:
            return OverlayReductionResult(False)
        entry = self.entry_for(event.channel, event.utterance_id)
        if entry.retained_hidden:
            return OverlayReductionResult(False)
        if event.seq < entry.last_updated_seq:
            return OverlayReductionResult(False)
        event_source_language = _content_language_or_none(event.source_language)
        if (
            entry.original_text == event.text
            and entry.original_language == event_source_language
            and entry.last_updated_seq == event.seq
        ):
            return OverlayReductionResult(False)

        self._remember_entry_input_seq(entry, event_seq=event.seq)
        entry.original_text = event.text
        entry.original_language = event_source_language
        first_readable_original = bool(event.text.strip()) and (
            not translation_enabled or show_peer_original
        )
        if first_readable_original and entry.speaker_style is None:
            entry.speaker_assignment = event.speaker_assignment
        entry.original_seq = event.seq
        entry.live_text = ""
        entry.live_seq = None
        entry.last_updated_seq = event.seq
        if entry.peer_caption_revision <= 0 and entry.translation_text.strip():
            entry.peer_caption_final = True
        self._refresh_entry_visibility_and_expiration(
            key,
            entry,
            now=now,
            publishable_seq=event.seq,
            next_appearance_seq=next_appearance_seq,
            show_peer_original=show_peer_original,
            translation_enabled=translation_enabled,
        )
        return OverlayReductionResult(True)

    def apply_peer_translation_update(
        self,
        event: TranslationStreamUpdate | TranslationFinal,
        *,
        now: float,
        show_peer_original: bool,
        next_appearance_seq: NextAppearanceSeq,
        terminal_update_reason: OverlayTerminalUpdateReason,
        translation_enabled: bool = True,
        peer_fragments: PeerFragmentPlan | None = None,
    ) -> OverlayReductionResult:

        key = self.entry_key(event.channel, event.utterance_id)
        if terminal_update_reason(event.channel, event.utterance_id) is not None:
            return OverlayReductionResult(False)
        entry = self.entry_for(event.channel, event.utterance_id)
        if event.seq < entry.last_updated_seq:
            return OverlayReductionResult(False)
        event_source_language = _content_language_or_none(event.source_language)
        event_target_language = _content_language_or_none(event.target_language)
        if event.peer_caption_revision < entry.peer_caption_revision or (
            event.peer_caption_revision == entry.peer_caption_revision
            and event.peer_caption_revision > 0
        ):
            return OverlayReductionResult(False)
        if entry.retained_hidden or (entry.peer_caption_final and not event.is_final):
            return OverlayReductionResult(False)
        if (
            entry.translation_text == event.text
            and (event_source_language is None or entry.original_language == event_source_language)
            and entry.translation_language == event_target_language
            and entry.last_updated_seq == event.seq
        ):
            return OverlayReductionResult(False)
        source_only = (
            isinstance(event, TranslationFinal)
            and event.peer_caption_revision > 0
            and not event.text.strip()
        )
        if source_only and not event.source_text.strip():
            retired_keys = {("peer", unit_id) for unit_id in event.peer_superseded_unit_ids}
            retired_keys.add(key)
            for retired_key in retired_keys:
                self.remove_peer_caption(
                    retired_key, reason="peer_caption_retired", now=now, seq=event.seq
                )
            return OverlayReductionResult(True)

        native_caption = (
            event.peer_caption_revision > 0
            and event.speaker_assignment is not None
            and event.speaker_assignment.attribution.source == "soniox"
        )
        self._remember_entry_input_seq(entry, event_seq=event.seq)
        if event.source_text.strip():
            entry.original_text = event.source_text
            entry.original_language = event_source_language
            entry.original_seq = event.seq
        elif event_source_language is not None:
            entry.original_language = event_source_language
        if not entry.retained_hidden:
            entry.translation_visible_since = self._next_translation_visible_since(
                previous_text=entry.translation_text,
                next_text=event.text,
                previous_visible_since=entry.translation_visible_since,
                now=now,
            )
        entry.translation_text = event.text
        entry.peer_caption_revision = event.peer_caption_revision
        entry.peer_caption_final = isinstance(event, TranslationFinal)
        entry.peer_publication_id = event.event_id
        entry.peer_source_only = source_only
        if event.text.strip() or source_only:
            if native_caption:
                assignment = event.speaker_assignment if event.is_final else None
                previous = entry.speaker_assignment
                previous_style = (
                    (previous.attribution, previous.palette_index, previous.palette_overflow)
                    if previous is not None
                    else None
                )
                next_style = (
                    (assignment.attribution, assignment.palette_index, assignment.palette_overflow)
                    if assignment is not None
                    else None
                )
                if previous_style != next_style:
                    entry.speaker_style = None
                    entry.speaker_gray_reason = None
                entry.speaker_assignment = assignment
            elif event.peer_caption_revision > 0 or entry.speaker_style is None:
                entry.speaker_assignment = event.speaker_assignment
            entry.translation_language = None if source_only else event_target_language
            entry.translation_update_id = event.update_id
            entry.translation_origin_wall_clock_ms = event.origin_wall_clock_ms
            entry.translation_session_scope = event.session_scope
            entry.translation_source_text_hash = event.source_text_hash
            entry.translation_source_text_len = event.source_text_len
            entry.translation_logical_turn_key = event.logical_turn_key
            entry.live_text = ""
            entry.live_seq = None
            entry.live_secondary_text = ""
        else:
            entry.translation_language = None
            entry.translation_update_id = None
            entry.translation_origin_wall_clock_ms = None
            entry.translation_session_scope = None
            entry.translation_source_text_hash = None
            entry.translation_source_text_len = None
            entry.translation_logical_turn_key = None
            if entry.speaker_style is None:
                entry.speaker_assignment = None
            if not entry.live_secondary_text.strip():
                entry.translation_seq = None
        if event.text.strip() and entry.translation_observed_visible_since is None:
            entry.translation_observed_visible_since = now
        entry.last_updated_seq = event.seq
        entry.caption_staged = False
        if event.text.strip() or source_only:
            self.reconcile_peer_fragments(
                entry,
                now=now,
                next_appearance_seq=next_appearance_seq,
                fragments=peer_fragments,
                translation_enabled=translation_enabled,
            )
        else:
            self._refresh_entry_visibility_and_expiration(
                key,
                entry,
                now=now,
                publishable_seq=event.seq,
                next_appearance_seq=next_appearance_seq,
                show_peer_original=show_peer_original,
                translation_enabled=translation_enabled,
            )
        superseded = {("peer", unit_id) for unit_id in event.peer_superseded_unit_ids}
        if superseded and (event.text.strip() or source_only):
            for old_key in superseded - {key}:
                self.remove_peer_caption(
                    old_key, reason="absorbed_by_repair", now=now, seq=event.seq
                )
        return OverlayReductionResult(True)

    def apply_peer_utterance_closed(
        self,
        event: UtteranceClosed,
        *,
        now: float,
        is_tombstoned: OverlayTerminalUpdatePredicate,
    ) -> OverlayReductionResult:

        key = self.entry_key(event.channel, event.utterance_id)
        if is_tombstoned(event.channel, event.utterance_id):
            return OverlayReductionResult(False)
        entry = self.entries.get(key)
        if entry is None:
            return OverlayReductionResult(False)
        if event.seq < entry.last_updated_seq:
            return OverlayReductionResult(False)
        if entry.closed_seq == event.seq:
            return OverlayReductionResult(False)
        entry.closed_seq = event.seq
        entry.closed_at = now
        entry.last_updated_seq = event.seq
        return OverlayReductionResult(True)

    def expire_entries(
        self,
        *,
        now: float,
        show_translation: bool,
        late_arrival_window_seconds: float,
        visible_ttl_seconds: float,
        self_translation_min_visible_seconds: float,
    ) -> bool:
        expired_keys = [
            key
            for key, entry in (*self.entries.items(), *self.peer_registrations.items())
            if (
                deadline := self.entry_expiration_deadline(
                    entry,
                    show_translation=show_translation,
                    late_arrival_window_seconds=late_arrival_window_seconds,
                    visible_ttl_seconds=visible_ttl_seconds,
                    self_translation_min_visible_seconds=self_translation_min_visible_seconds,
                )
            )
            is not None
            and now >= deadline
        ]
        for key in expired_keys:
            entry = self.display_entry(key)
            if entry is None:
                continue
            self.remove_entry(
                key,
                reason="expired",
                now=now,
                tombstone_seq=entry.last_updated_seq if entry.closed_seq is None else None,
            )
        return bool(expired_keys)

    def generate_snapshot(
        self,
        *,
        revision: int,
        calibration: OverlayPresentationCalibration,
        rendered_entries: list[tuple[OverlayEntryKey, OverlayPresentationBlock]],
        native_fresh_render_generations: NativeFreshRenderGenerations | None = None,
        native_fresh_render_targets: NativeFreshRenderTargets | None = None,
        native_quiet_tail_episodes: NativeQuietTailEpisodes | None = None,
        entry_ordering: Mapping[OverlayEntryKey, tuple[str, int, int, int]] | None = None,
        semantic_retirement_frontiers: Mapping[tuple[str, int], tuple[int, int]] | None = None,
        speaker_divider: bool = False,
    ) -> OverlayPresentationSnapshot:
        ordering = entry_ordering or {}
        blocks = []
        for key, block in rendered_entries:
            block_ordering = ordering.get(key)
            entry = self.display_entry(key)
            if block_ordering is None and entry is not None and entry.peer_parent_id is not None:
                block_ordering = ordering.get(("peer", entry.peer_parent_id))
            if block_ordering is not None:
                scope, generation, order, index = block_ordering
                block = replace(
                    block,
                    publication_scope=scope,
                    publication_generation=generation,
                    publication_order=order,
                    publication_index=index,
                )
            blocks.append(block)
        snapshot = OverlayPresentationSnapshot(
            revision=revision,
            calibration=calibration,
            blocks=blocks,
            native_fresh_render_generations=native_fresh_render_generations,
            native_fresh_render_targets=native_fresh_render_targets,
            native_quiet_tail_episodes=native_quiet_tail_episodes,
            semantic_retirement_frontiers=[
                SemanticRetirementFrontier(
                    scope=scope, generation=generation, order=order, index=index
                )
                for (scope, generation), (order, index) in sorted(
                    (semantic_retirement_frontiers or {}).items()
                )
            ],
            speaker_divider=speaker_divider,
        )
        self._snapshot = snapshot
        return snapshot

    def visible_block_selection(
        self,
        *,
        entries: Mapping[OverlayEntryKey, OverlayPresentationEntry],
        live_self_entry: tuple[OverlayEntryKey, OverlayPresentationEntry] | None,
        live_peer_entry: tuple[OverlayEntryKey, OverlayPresentationEntry] | None,
        visible_window_target_blocks: int,
        show_translation: bool,
        show_peer_original: bool,
        next_appearance_seq: NextAppearanceSeq | None,
        translation_enabled: bool = True,
    ) -> OverlayVisibleBlockSelection:
        active_self_key = (
            live_self_entry[0]
            if live_self_entry is not None and live_self_entry[1].live_text
            else None
        )
        active_self_present = active_self_key is not None
        active_peer_key = (
            live_peer_entry[0]
            if live_peer_entry is not None
            and self._live_peer_entry_is_drawable(
                live_peer_entry[1],
                show_peer_original=show_peer_original,
                translation_enabled=translation_enabled,
            )
            else None
        )
        protected_keys = [key for key in (active_self_key, active_peer_key) if key is not None]
        protected_key_set = set(protected_keys)
        finalized_limit = max(
            visible_window_target_blocks - len(protected_keys),
            0,
        )
        visible_entry_keys, candidate_keys = self._logical_visible_entry_keys(
            entries=entries,
            finalized_limit=finalized_limit,
            excluded_keys=protected_key_set,
            show_peer_original=show_peer_original,
            translation_enabled=translation_enabled,
            next_appearance_seq=next_appearance_seq,
        )
        rendered_entries = [
            (key, block)
            for key in visible_entry_keys
            if (entry := entries.get(key)) is not None
            and (
                block := self._build_presentation_block(
                    entry,
                    show_translation=show_translation,
                    show_peer_original=show_peer_original,
                    translation_enabled=translation_enabled,
                )
            )
            is not None
        ]
        for protected_key in protected_keys:
            active_entry = entries.get(protected_key)
            if active_entry is None:
                continue
            block = self._build_presentation_block(
                active_entry,
                prefer_live_self=protected_key == active_self_key,
                show_translation=show_translation,
                show_peer_original=show_peer_original,
                translation_enabled=translation_enabled,
            )
            if block is None:
                continue
            rendered_entries.append((protected_key, block))
        rendered_entries.sort(key=lambda item: (item[1].appearance_seq, item[1].occupant_key))
        return OverlayVisibleBlockSelection(
            rendered_entries=rendered_entries,
            active_self_present=active_self_present,
            finalized_limit=finalized_limit,
            candidate_keys=candidate_keys,
            selected_keys=visible_entry_keys,
            protected_keys=protected_keys,
        )

    def entry_is_publishable(
        self,
        entry: OverlayPresentationEntry,
        *,
        show_peer_original: bool,
        translation_enabled: bool = True,
    ) -> bool:
        if isinstance(entry, PeerFragmentRegistration):
            entry = entry.parent
        if entry.channel == "peer":
            if not translation_enabled:
                return bool(entry.live_text.strip() or entry.original_text.strip())
            return bool(
                entry.translation_text.strip()
                or (entry.peer_source_only and entry.original_text.strip())
                or (show_peer_original and (entry.live_text.strip() or entry.original_text.strip()))
            )
        return bool(entry.original_text.strip())

    def entry_is_selectable(
        self,
        entry: OverlayPresentationEntry,
        *,
        show_peer_original: bool,
        translation_enabled: bool = True,
    ) -> bool:
        return (
            self.entry_is_publishable(
                entry,
                show_peer_original=show_peer_original,
                translation_enabled=translation_enabled,
            )
            and not entry.retained_hidden
            and not entry.caption_staged
            and not entry.peer_pending
            and not entry.peer_departed
            and not entry.peer_dormant
        )

    def rendered_block_signature(
        self,
        block: OverlayPresentationBlock,
    ) -> tuple[object, ...]:
        secondary_text = block.secondary_text if block.secondary_enabled else ""
        include_translation_metadata = block.channel == "peer" or bool(secondary_text)
        return (
            block.id,
            block.occupant_key,
            block.appearance_seq,
            block.channel,
            block.block_variant,
            block.primary_text,
            secondary_text,
            block.secondary_enabled,
            block.primary_language,
            block.secondary_language if block.secondary_enabled else None,
            block.update_id if include_translation_metadata else None,
            block.origin_wall_clock_ms if include_translation_metadata else None,
            block.session_scope if include_translation_metadata else None,
            block.source_text_hash if include_translation_metadata else None,
            block.source_text_len if include_translation_metadata else None,
            block.logical_turn_key if include_translation_metadata else None,
            block.speaker_style,
        )

    def rendered_blocks_signature(
        self,
        blocks: list[OverlayPresentationBlock],
    ) -> tuple[object, ...]:
        return tuple(self.rendered_block_signature(block) for block in blocks)

    def visible_block_content_signature(
        self,
        block: OverlayPresentationBlock,
    ) -> tuple[str, str, str, bool]:
        secondary_text = block.secondary_text if block.secondary_enabled else ""
        return (
            block.block_variant,
            block.primary_text,
            secondary_text,
            block.secondary_enabled,
        )

    def entry_expiration_deadline(
        self,
        entry: OverlayPresentationEntry,
        *,
        show_translation: bool,
        late_arrival_window_seconds: float,
        visible_ttl_seconds: float,
        self_translation_min_visible_seconds: float,
    ) -> float | None:
        return self.entry_expiration_components(
            entry,
            show_translation=show_translation,
            late_arrival_window_seconds=late_arrival_window_seconds,
            visible_ttl_seconds=visible_ttl_seconds,
            self_translation_min_visible_seconds=self_translation_min_visible_seconds,
        )[0]

    def entry_expiration_components(
        self,
        entry: OverlayPresentationEntry,
        *,
        show_translation: bool,
        late_arrival_window_seconds: float,
        visible_ttl_seconds: float,
        self_translation_min_visible_seconds: float,
    ) -> tuple[float | None, float | None, float | None]:
        if entry.channel == "peer" and entry.peer_dormant and not entry.peer_departed:
            parent = entry.parent if isinstance(entry, PeerFragmentRegistration) else entry
            anchor = entry.last_meaningful_visible_at
            if anchor is None:
                anchor = parent.peer_last_visible_at
            if anchor is not None:
                return anchor + visible_ttl_seconds, None, None
            anchor = (
                parent.closed_at
                if parent.closed_at is not None
                else (
                    parent.peer_registered_at
                    if parent.peer_registered_at is not None
                    else parent.peer_cached_at
                )
            )
            return (
                None if anchor is None else anchor + late_arrival_window_seconds,
                None,
                None,
            )
        if entry.channel == "peer" and entry.caption_staged and entry.closed_at is not None:
            return entry.closed_at + late_arrival_window_seconds, None, None
        if entry.channel == "peer" and (entry.peer_pending or entry.peer_departed):
            return None, None, None
        hidden_deadline = (
            entry.window_evicted_at + late_arrival_window_seconds
            if entry.retained_hidden and entry.window_evicted_at is not None
            else None
        )
        visible_anchor = entry.last_meaningful_visible_at
        if visible_anchor is None and entry.visible_since is not None:
            visible_anchor = entry.visible_since

        visible_deadline: float | None = None
        if visible_anchor is not None:
            visible_deadline = visible_anchor + visible_ttl_seconds
        elif entry.closed_at is not None:
            visible_deadline = entry.closed_at + late_arrival_window_seconds
        if entry.caption_staged and visible_deadline is not None:
            visible_deadline += late_arrival_window_seconds

        translation_deadline: float | None = None
        if (
            entry.channel == "self"
            and show_translation
            and entry.translation_visible_since is not None
        ):
            translation_deadline = (
                entry.translation_visible_since + self_translation_min_visible_seconds
            )
        effective_deadline = visible_deadline
        if translation_deadline is not None:
            if effective_deadline is None:
                effective_deadline = translation_deadline
            else:
                effective_deadline = max(effective_deadline, translation_deadline)
        if hidden_deadline is not None:
            if effective_deadline is None:
                effective_deadline = hidden_deadline
            else:
                effective_deadline = min(effective_deadline, hidden_deadline)
        return effective_deadline, visible_deadline, translation_deadline

    def _active_update_entry_or_none(
        self,
        *,
        channel: str,
        utterance_id: UUID | None,
        event_seq: int,
        now: float,
        show_translation: bool,
        terminal_update_reason: OverlayTerminalUpdateReason,
    ) -> tuple[OverlayEntryKey, OverlayLogicalTurnEntry] | None:
        key = self.entry_key(channel, utterance_id)
        if terminal_update_reason(channel, utterance_id) is not None:
            return None
        live_entry = self.live_entry_for_channel(channel)
        if live_entry is not None:
            live_key, current_live_entry = live_entry
            if live_key != key and event_seq < current_live_entry.last_updated_seq:
                return None

        entry = self.entry_for(channel, utterance_id)
        if event_seq < entry.last_updated_seq:
            return None
        if live_entry is not None and live_entry[0] != key:
            if channel == "self":
                self._clear_live_self_pointer(
                    reason="live_self_replaced",
                    now=now,
                    show_translation=show_translation,
                )
            else:
                self._clear_live_peer_pointer()
        return key, entry

    def _active_update_matches_live_payload(
        self,
        *,
        channel: str,
        key: OverlayEntryKey,
        entry: OverlayLogicalTurnEntry,
        text: str,
        secondary_text: str,
        occupant_key: str,
        update_id: str | None,
        origin_wall_clock_ms: int | None,
        session_scope: str | None,
        source_text_hash: str | None,
        source_text_len: int | None,
        logical_turn_key: str | None,
    ) -> bool:
        if self.live_turn_key_for_channel(channel) != key:
            return False
        if entry.live_text != text or entry.occupant_key != occupant_key:
            return False
        if channel == "peer":
            return True
        if channel != "self":
            raise ValueError(f"invalid overlay channel: {channel!r}")
        return (
            entry.live_secondary_text == secondary_text
            and entry.live_update_id == update_id
            and entry.live_origin_wall_clock_ms == origin_wall_clock_ms
            and entry.live_session_scope == session_scope
            and entry.live_source_text_hash == source_text_hash
            and entry.live_source_text_len == source_text_len
            and entry.live_logical_turn_key == logical_turn_key
        )

    def _clear_live_self_pointer(
        self,
        *,
        reason: str,
        now: float,
        show_translation: bool,
    ) -> None:
        live_self = self.live_entry_for_channel("self")
        if live_self is None:
            return
        key, entry = live_self
        previous_rendered_translation_text = self._rendered_self_translation_text(entry)
        self._clear_self_live_payload(entry)
        self.live_self_turn_key = None
        self._update_self_translation_visibility(
            entry,
            previous_rendered_text=previous_rendered_translation_text,
            next_rendered_text=self._rendered_self_translation_text(entry),
            now=now,
            show_translation=show_translation,
        )
        if self._should_retire_preview_only_self_entry(entry):
            self._retire_preview_only_self_entry(key, entry, reason=reason, now=now)

    def _clear_live_peer_pointer(self) -> None:
        live_peer = self.live_entry_for_channel("peer")
        if live_peer is None:
            return
        _, entry = live_peer
        entry.live_text = ""
        entry.live_secondary_text = ""
        entry.live_seq = None
        self.live_peer_turn_key = None

    def _clear_self_live_payload(self, entry: OverlayLogicalTurnEntry) -> None:
        entry.live_text = ""
        entry.live_secondary_text = ""
        entry.live_primary_language = None
        entry.live_secondary_language = None
        entry.live_update_id = None
        entry.live_origin_wall_clock_ms = None
        entry.live_session_scope = None
        entry.live_source_text_hash = None
        entry.live_source_text_len = None
        entry.live_logical_turn_key = None
        entry.live_seq = None

    def _rendered_self_translation_text(self, entry: OverlayLogicalTurnEntry) -> str:
        live_secondary_text = entry.live_secondary_text.strip()
        if live_secondary_text:
            return live_secondary_text
        return entry.translation_text.strip()

    def _update_self_translation_visibility(
        self,
        entry: OverlayLogicalTurnEntry,
        *,
        previous_rendered_text: str,
        next_rendered_text: str,
        now: float,
        show_translation: bool,
    ) -> None:
        if not show_translation:
            return
        entry.translation_visible_since = self._next_translation_visible_since(
            previous_text=previous_rendered_text,
            next_text=next_rendered_text,
            previous_visible_since=entry.translation_visible_since,
            now=now,
        )
        if next_rendered_text and entry.translation_observed_visible_since is None:
            entry.translation_observed_visible_since = now

    def _next_translation_visible_since(
        self,
        *,
        previous_text: str,
        next_text: str,
        previous_visible_since: float | None,
        now: float,
    ) -> float | None:
        next_clean = next_text.strip()
        if not next_clean:
            return None
        if previous_text.strip() != next_clean:
            return now
        return previous_visible_since

    def _remember_entry_input_seq(
        self,
        entry: OverlayLogicalTurnEntry,
        *,
        event_seq: int,
    ) -> None:
        if entry.first_input_seq is None:
            entry.first_input_seq = event_seq

    def _refresh_entry_visibility_and_expiration(
        self,
        key: OverlayEntryKey,
        entry: OverlayLogicalTurnEntry,
        *,
        now: float,
        publishable_seq: int | None,
        next_appearance_seq: NextAppearanceSeq,
        show_peer_original: bool = True,
        translation_enabled: bool = True,
    ) -> None:
        if self.entry_is_publishable(
            entry,
            show_peer_original=show_peer_original,
            translation_enabled=translation_enabled,
        ):
            self._ensure_entry_visibility_metadata(
                entry,
                occupant_key=self._finalized_occupant_key(entry.channel, entry.utterance_id),
                publishable_seq=publishable_seq,
                next_appearance_seq=next_appearance_seq,
            )
            entry.ever_publishable = True
            if entry.visible_since is None:
                entry.visible_since = now

    def _should_retire_preview_only_self_entry(
        self,
        entry: OverlayLogicalTurnEntry,
    ) -> bool:
        return (
            entry.channel == "self"
            and not entry.original_text.strip()
            and not entry.translation_text.strip()
        )

    def _retire_preview_only_self_entry(
        self,
        key: OverlayEntryKey,
        entry: OverlayLogicalTurnEntry,
        *,
        reason: str,
        now: float,
    ) -> None:
        self._remember_retired_preview_self_seq(key, entry.last_updated_seq)
        self.remove_entry(key, reason=reason, now=now)

    def _remember_retired_preview_self_seq(
        self,
        key: OverlayEntryKey,
        retired_seq: int,
    ) -> None:
        self.retired_preview_self_seqs.pop(key, None)
        self.retired_preview_self_seqs[key] = retired_seq
        while len(self.retired_preview_self_seqs) > _RETIRED_PREVIEW_SELF_SEQ_LIMIT:
            self.retired_preview_self_seqs.popitem(last=False)

    def _logical_visible_entry_keys(
        self,
        *,
        entries: Mapping[OverlayEntryKey, OverlayPresentationEntry],
        finalized_limit: int,
        excluded_keys: set[OverlayEntryKey],
        show_peer_original: bool,
        next_appearance_seq: NextAppearanceSeq | None,
        translation_enabled: bool = True,
    ) -> tuple[list[OverlayEntryKey], list[OverlayEntryKey]]:
        if finalized_limit == 0:
            return [], []

        publishable: list[tuple[int, int, str, str, OverlayEntryKey]] = []
        for key, entry in entries.items():
            if key in excluded_keys:
                continue
            if not self.entry_is_selectable(
                entry,
                show_peer_original=show_peer_original,
                translation_enabled=translation_enabled,
            ):
                continue
            if next_appearance_seq is not None and isinstance(entry, OverlayLogicalTurnEntry):
                self._ensure_entry_visibility_metadata(
                    entry,
                    occupant_key=self._finalized_occupant_key(entry.channel, entry.utterance_id),
                    next_appearance_seq=next_appearance_seq,
                )
            publishable_seq = (
                entry.publishable_seq
                if entry.publishable_seq is not None
                else entry.last_updated_seq if entry.last_updated_seq > 0 else None
            )
            appearance_seq = (
                entry.appearance_seq if entry.appearance_seq is not None else entry.first_input_seq
            )
            if publishable_seq is None or appearance_seq is None:
                continue
            publishable.append(
                (
                    publishable_seq,
                    appearance_seq,
                    entry.occupant_key
                    or self._finalized_occupant_key(entry.channel, entry.utterance_id),
                    self._format_entry_key(key),
                    key,
                )
            )

        display_order = sorted(publishable, key=lambda item: (item[1], item[2], item[3]))
        selected_candidates = sorted(
            publishable,
            key=lambda item: (item[0], item[1], item[2], item[3]),
        )[-finalized_limit:]
        selected_set = {key for *_, key in selected_candidates}
        selected = [key for *_, key in display_order if key in selected_set]
        return selected, [key for *_, key in display_order]

    def _build_presentation_block(
        self,
        entry: OverlayPresentationEntry,
        *,
        prefer_live_self: bool = False,
        show_translation: bool,
        show_peer_original: bool,
        translation_enabled: bool = True,
        peer_partition_index: int | None = None,
    ) -> OverlayPresentationBlock | None:
        if isinstance(entry, PeerFragmentRegistration):
            position = (
                entry.peer_source_index
                if entry.parent.peer_partition_is_source
                else entry.peer_translation_index
            )
            block = self._build_presentation_block(
                entry.parent,
                show_translation=show_translation,
                show_peer_original=show_peer_original,
                translation_enabled=translation_enabled,
                peer_partition_index=position,
            )
            if block is None:
                return None
            return replace(
                block,
                id=entry.block_id,
                occupant_key=entry.occupant_key,
                appearance_seq=entry.appearance_seq,
            )
        if prefer_live_self and entry.channel == "self":
            primary_text = entry.live_text.strip()
            if not primary_text:
                return None
            live_secondary_text = entry.live_secondary_text.strip()
            secondary_text = live_secondary_text or entry.translation_text.strip()
            secondary_language = (
                entry.live_secondary_language if live_secondary_text else entry.translation_language
            )
            if live_secondary_text:
                update_id = entry.live_update_id
                origin_wall_clock_ms = entry.live_origin_wall_clock_ms
                session_scope = entry.live_session_scope
                source_text_hash = entry.live_source_text_hash
                source_text_len = entry.live_source_text_len
                logical_turn_key = entry.live_logical_turn_key
            else:
                update_id = entry.translation_update_id
                origin_wall_clock_ms = entry.translation_origin_wall_clock_ms
                session_scope = entry.translation_session_scope
                source_text_hash = entry.translation_source_text_hash
                source_text_len = entry.translation_source_text_len
                logical_turn_key = entry.translation_logical_turn_key
            return OverlayPresentationBlock(
                id=entry.block_id,
                occupant_key=entry.occupant_key,
                appearance_seq=self._block_appearance_seq(entry),
                channel="self",
                block_variant="active_self",
                primary_text=primary_text,
                secondary_text=secondary_text,
                secondary_enabled=show_translation,
                primary_language=_line_language(entry.live_primary_language, primary_text),
                secondary_language=_line_language(
                    secondary_language,
                    secondary_text,
                    enabled=show_translation,
                ),
                update_id=update_id,
                origin_wall_clock_ms=origin_wall_clock_ms,
                session_scope=session_scope,
                source_text_hash=source_text_hash,
                source_text_len=source_text_len,
                logical_turn_key=logical_turn_key,
            )
        if entry.channel == "peer":
            position = peer_partition_index
            if position is None:
                position = (
                    entry.peer_source_index
                    if entry.peer_partition_is_source
                    else entry.peer_translation_index
                )
            fragment = (
                entry.peer_fragments[position]
                if position is not None and position < len(entry.peer_fragments)
                else None
            )
            if not translation_enabled:
                live_source_text = entry.live_text.strip()
                if live_source_text:
                    return OverlayPresentationBlock(
                        id=entry.block_id,
                        occupant_key=entry.occupant_key,
                        appearance_seq=self._block_appearance_seq(entry),
                        channel="peer",
                        block_variant="active_peer",
                        primary_text=live_source_text,
                        secondary_text="",
                        secondary_enabled=False,
                        primary_language=_line_language(entry.original_language, live_source_text),
                        secondary_language=None,
                        session_scope=None,
                    )
                finalized_source_text = entry.original_text.strip() or entry.live_text.strip()
                if finalized_source_text:
                    return OverlayPresentationBlock(
                        id=entry.block_id,
                        occupant_key=entry.occupant_key,
                        appearance_seq=(
                            entry.appearance_seq
                            if entry.appearance_seq is not None
                            else self._block_appearance_seq(entry)
                        ),
                        channel="peer",
                        block_variant="finalized",
                        primary_text=fragment if fragment is not None else finalized_source_text,
                        secondary_text="",
                        secondary_enabled=False,
                        primary_language=_line_language(
                            entry.original_language, finalized_source_text
                        ),
                        secondary_language=None,
                        session_scope=None,
                    )
                return None
            translated_text = entry.translation_text.strip()
            original_text = entry.original_text.strip() or entry.live_text.strip()
            if entry.peer_source_only and original_text:
                return OverlayPresentationBlock(
                    id=entry.block_id,
                    occupant_key=entry.occupant_key,
                    appearance_seq=self._block_appearance_seq(entry),
                    channel="peer",
                    block_variant="finalized",
                    primary_text=fragment if fragment is not None else original_text,
                    secondary_text="",
                    secondary_enabled=False,
                    primary_language=_line_language(entry.original_language, original_text),
                    secondary_language=None,
                    update_id=entry.translation_update_id,
                    origin_wall_clock_ms=entry.translation_origin_wall_clock_ms,
                    session_scope=entry.translation_session_scope,
                    source_text_hash=entry.translation_source_text_hash,
                    source_text_len=entry.translation_source_text_len,
                    logical_turn_key=entry.translation_logical_turn_key,
                )
            if translated_text:
                secondary_text = original_text if position in (None, 0) else ""
                return OverlayPresentationBlock(
                    id=entry.block_id,
                    occupant_key=entry.occupant_key,
                    appearance_seq=(
                        entry.appearance_seq
                        if entry.appearance_seq is not None
                        else self._block_appearance_seq(entry)
                    ),
                    channel="peer",
                    block_variant="finalized",
                    primary_text=fragment if fragment is not None else translated_text,
                    secondary_text=secondary_text,
                    secondary_enabled=show_peer_original and bool(secondary_text),
                    primary_language=_line_language(
                        entry.translation_language,
                        translated_text,
                    ),
                    secondary_language=_line_language(
                        entry.original_language,
                        secondary_text,
                        enabled=show_peer_original,
                    ),
                    update_id=entry.translation_update_id,
                    origin_wall_clock_ms=entry.translation_origin_wall_clock_ms,
                    session_scope=entry.translation_session_scope,
                    source_text_hash=entry.translation_source_text_hash,
                    source_text_len=entry.translation_source_text_len,
                    logical_turn_key=entry.translation_logical_turn_key,
                    speaker_style=None,
                )
            active_text = entry.live_text.strip()
            if active_text:
                if not show_peer_original:
                    return None
                return OverlayPresentationBlock(
                    id=entry.block_id,
                    occupant_key=entry.occupant_key,
                    appearance_seq=self._block_appearance_seq(entry),
                    channel="peer",
                    block_variant="active_peer",
                    primary_text="",
                    secondary_text=fragment if fragment is not None else active_text,
                    secondary_enabled=True,
                    primary_language=None,
                    secondary_language=_line_language(entry.original_language, active_text),
                    session_scope=None,
                )
            if original_text:
                if not show_peer_original:
                    return None
                return OverlayPresentationBlock(
                    id=entry.block_id,
                    occupant_key=entry.occupant_key,
                    appearance_seq=(
                        entry.appearance_seq
                        if entry.appearance_seq is not None
                        else self._block_appearance_seq(entry)
                    ),
                    channel="peer",
                    block_variant="finalized",
                    primary_text="",
                    secondary_text=fragment if fragment is not None else original_text,
                    secondary_enabled=True,
                    primary_language=None,
                    secondary_language=_line_language(entry.original_language, original_text),
                    session_scope=None,
                )
            return None

        primary_text = entry.original_text.strip()
        if not primary_text:
            return None
        secondary_text = entry.translation_text.strip()
        secondary_enabled = show_translation

        return OverlayPresentationBlock(
            id=entry.block_id,
            occupant_key=entry.occupant_key,
            appearance_seq=entry.appearance_seq,
            channel=entry.channel,  # type: ignore[arg-type]
            block_variant="finalized",
            primary_text=primary_text,
            secondary_text=secondary_text,
            secondary_enabled=secondary_enabled,
            primary_language=_line_language(entry.original_language, primary_text),
            secondary_language=_line_language(
                entry.translation_language,
                secondary_text,
                enabled=secondary_enabled,
            ),
            update_id=entry.translation_update_id,
            origin_wall_clock_ms=entry.translation_origin_wall_clock_ms,
            session_scope=entry.translation_session_scope,
            source_text_hash=entry.translation_source_text_hash,
            source_text_len=entry.translation_source_text_len,
            logical_turn_key=entry.translation_logical_turn_key,
        )

    def _live_peer_entry_is_drawable(
        self,
        entry: OverlayPresentationEntry,
        *,
        show_peer_original: bool,
        translation_enabled: bool = True,
    ) -> bool:
        if not translation_enabled:
            return bool(entry.live_text.strip() or entry.original_text.strip())
        if entry.translation_text.strip():
            return True
        if not show_peer_original:
            return False
        return bool(entry.live_text.strip() or entry.original_text.strip())

    def _ensure_entry_visibility_metadata(
        self,
        entry: OverlayLogicalTurnEntry,
        *,
        occupant_key: str,
        next_appearance_seq: NextAppearanceSeq,
        publishable_seq: int | None = None,
    ) -> None:
        if not entry.occupant_key:
            entry.occupant_key = occupant_key
        if entry.appearance_seq is None:
            if entry.first_input_seq is not None:
                entry.appearance_seq = entry.first_input_seq
            elif publishable_seq is not None:
                entry.appearance_seq = publishable_seq
            else:
                entry.appearance_seq = next_appearance_seq()
        if entry.publishable_seq is None:
            if publishable_seq is not None:
                entry.publishable_seq = publishable_seq
            elif entry.last_updated_seq > 0:
                entry.publishable_seq = entry.last_updated_seq

    def _block_appearance_seq(self, entry: OverlayPresentationEntry) -> int:
        if entry.appearance_seq is not None:
            return entry.appearance_seq
        if entry.first_input_seq is not None:
            return entry.first_input_seq
        if entry.last_updated_seq > 0:
            return entry.last_updated_seq
        return 0

    def _finalized_occupant_key(self, channel: str, utterance_id: UUID) -> str:
        return f"{channel}:{utterance_id}"

    def _format_entry_key(self, key: OverlayEntryKey) -> str:
        return f"{key[0]}:{key[1]}"


def _content_language_or_none(language: str | None) -> str | None:
    if language is None:
        return None
    normalized = language.strip()
    return normalized or None


def _line_language(
    language: str | None,
    text: str,
    *,
    enabled: bool = True,
) -> str | None:
    if not enabled or not text.strip():
        return None
    return _content_language_or_none(language)
