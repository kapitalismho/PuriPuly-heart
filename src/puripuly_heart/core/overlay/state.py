from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Protocol
from uuid import UUID

from puripuly_heart.core.overlay.protocol import (
    NativeFreshRenderGenerations,
    NativeFreshRenderTargets,
    NativeQuietTailEpisodes,
    OverlayPresentationBlock,
    OverlayPresentationCalibration,
    SemanticRetirementFrontier,
    OverlayPresentationSnapshot,
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

OverlayEntryKey = tuple[str, UUID]
NextAppearanceSeq = Callable[[], int]
OverlayTerminalUpdatePredicate = Callable[[str | None, UUID | None], bool]
OverlayTerminalUpdateReason = Callable[[str | None, UUID | None], str | None]

_RETIRED_PREVIEW_SELF_SEQ_LIMIT = 64


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

    @property
    def block_id(self) -> str:
        return f"{self.channel}:{self.utterance_id}"


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
    entry: OverlayLogicalTurnEntry
    reason: str
    now: float | None
    tombstone_seq: int | None


@dataclass(frozen=True, slots=True)
class OverlayTurnDecisionRecord:
    decision: str
    disposition: str | None = None
    key: OverlayEntryKey | None = None
    entry: OverlayLogicalTurnEntry | None = None
    block: OverlayPresentationBlock | None = None
    extras: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class OverlayReductionResult:
    changed: bool
    decisions: tuple[OverlayTurnDecisionRecord, ...] = ()

    def __bool__(self) -> bool:
        return self.changed


class OverlayPresentationEntry(Protocol):
    channel: str
    utterance_id: UUID
    first_input_seq: int | None
    live_text: str
    live_secondary_text: str
    live_primary_language: str | None
    live_secondary_language: str | None
    live_update_id: str | None
    live_origin_wall_clock_ms: int | None
    live_session_scope: str | None
    live_source_text_hash: str | None
    live_source_text_len: int | None
    live_logical_turn_key: str | None
    original_text: str
    original_language: str | None
    translation_text: str
    translation_language: str | None
    translation_update_id: str | None
    translation_origin_wall_clock_ms: int | None
    translation_session_scope: str | None
    translation_source_text_hash: str | None
    translation_source_text_len: int | None
    translation_logical_turn_key: str | None
    occupant_key: str
    appearance_seq: int | None
    publishable_seq: int | None
    retained_hidden: bool
    ever_visible: bool
    last_updated_seq: int

    @property
    def block_id(self) -> str: ...


@dataclass(frozen=True)
class OverlayVisibleBlockSelection:
    rendered_entries: list[tuple[OverlayEntryKey, OverlayPresentationBlock]]
    active_self_present: bool
    finalized_limit: int
    candidate_keys: list[OverlayEntryKey]
    selected_keys: list[OverlayEntryKey]
    protected_keys: list[OverlayEntryKey]
    retained_hidden: list[OverlayEntryKey] = field(default_factory=list)


@dataclass
class OverlayPresentationState:
    """Pure presentation state for overlay snapshots.

    This object intentionally does not own async tasks, sleeps, timers,
    cancellation, bridge I/O, provider I/O, OpenVR I/O, native overlay I/O,
    or dashboard side effects.
    """

    entries: dict[OverlayEntryKey, OverlayLogicalTurnEntry] = field(default_factory=dict)
    retired_preview_self_seqs: OrderedDict[OverlayEntryKey, int] = field(
        default_factory=OrderedDict
    )
    live_self_turn_key: OverlayEntryKey | None = None
    live_peer_turn_key: OverlayEntryKey | None = None
    peer_presentation_refresh_target_key: OverlayEntryKey | None = None
    peer_presentation_refresh_nonce: int = 0
    self_presentation_refresh_target_key: OverlayEntryKey | None = None
    self_presentation_refresh_nonce: int = 0
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

    def begin_peer_presentation_refresh(self, key: OverlayEntryKey) -> bool:
        """Select the peer refresh target and reset any previous refresh nonce."""
        had_visible_marker = self._snapshot_has_peer_presentation_refresh_marker()
        self.peer_presentation_refresh_target_key = key
        self.peer_presentation_refresh_nonce = 0
        return had_visible_marker

    def tick_peer_presentation_refresh(self, key: OverlayEntryKey) -> bool:
        """Advance the load-bearing peer refresh nonce for the active target."""
        if self.peer_presentation_refresh_target_key != key:
            return False
        # LOAD-BEARING: peer_presentation_refresh=<n> prevents revision/dedup
        # coalescing during the product-permanent burst. Do not normalize this
        # away unless Stage 2 HMD QA proves an alternative fresh-render path.
        self.peer_presentation_refresh_nonce += 1
        return True

    def end_peer_presentation_refresh(self, key: OverlayEntryKey) -> bool:
        """Clear the peer refresh nonce and request cleanup publish if needed."""
        if self.peer_presentation_refresh_target_key != key:
            return False
        had_refresh_metadata = self._snapshot_has_peer_presentation_refresh_marker()
        self.peer_presentation_refresh_target_key = None
        self.peer_presentation_refresh_nonce = 0
        return had_refresh_metadata

    def _snapshot_has_peer_presentation_refresh_marker(self) -> bool:
        for block in self._snapshot.blocks:
            session_scope = block.session_scope
            if session_scope is None:
                continue
            if any(
                part.startswith("peer_presentation_refresh=") for part in session_scope.split("|")
            ):
                return True
        return False

    def begin_self_presentation_refresh(self, key: OverlayEntryKey) -> bool:
        """Select the finalized self refresh target and reset any previous nonce."""
        had_visible_marker = self._snapshot_has_self_presentation_refresh_marker()
        self.self_presentation_refresh_target_key = key
        self.self_presentation_refresh_nonce = 0
        return had_visible_marker

    def tick_self_presentation_refresh(self, key: OverlayEntryKey) -> bool:
        """Advance the load-bearing self refresh nonce for the active target."""
        if self.self_presentation_refresh_target_key != key:
            return False
        # LOAD-BEARING: self_presentation_refresh=<n> must be revision-worthy
        # for finalized self rows, including source-only captions with no
        # secondary text, so the local overlay path receives fresh snapshots.
        self.self_presentation_refresh_nonce += 1
        return True

    def end_self_presentation_refresh(self, key: OverlayEntryKey) -> bool:
        """Clear the self refresh nonce and request cleanup publish if needed."""
        if self.self_presentation_refresh_target_key != key:
            return False
        had_refresh_metadata = self._snapshot_has_self_presentation_refresh_marker()
        self.self_presentation_refresh_target_key = None
        self.self_presentation_refresh_nonce = 0
        return had_refresh_metadata

    def _snapshot_has_self_presentation_refresh_marker(self) -> bool:
        for block in self._snapshot.blocks:
            if block.channel != "self":
                continue
            session_scope = block.session_scope
            if session_scope is None:
                continue
            if any(
                part.startswith("self_presentation_refresh=") for part in session_scope.split("|")
            ):
                return True
        return False

    def remove_entry(
        self,
        key: OverlayEntryKey,
        *,
        reason: str,
        now: float | None = None,
        tombstone_seq: int | None = None,
    ) -> OverlayEntryRemovalRecord | None:
        if self.live_self_turn_key == key:
            self.live_self_turn_key = None
        if self.live_peer_turn_key == key:
            self.live_peer_turn_key = None
        entry = self.entries.pop(key, None)
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
        return record

    def apply_self_active_update(
        self,
        event: SelfActiveUpdate,
        *,
        now: float,
        show_translation: bool,
        terminal_update_reason: OverlayTerminalUpdateReason,
    ) -> OverlayReductionResult:
        decisions: list[OverlayTurnDecisionRecord] = []
        key = self.entry_key(event.channel, event.utterance_id)
        retired_preview_seq = self.retired_preview_self_seqs.get(key)
        if retired_preview_seq is not None and event.seq <= retired_preview_seq:
            decisions.append(
                OverlayTurnDecisionRecord(
                    decision="overlay_turn_superseded",
                    disposition="superseded",
                    key=key,
                    extras={"event_seq": event.seq, "retired_preview_seq": retired_preview_seq},
                )
            )
            return OverlayReductionResult(False, tuple(decisions))
        active_entry = self._active_update_entry_or_none(
            channel=event.channel,
            utterance_id=event.utterance_id,
            event_seq=event.seq,
            now=now,
            show_translation=show_translation,
            terminal_update_reason=terminal_update_reason,
            decisions=decisions,
        )
        if active_entry is None:
            return OverlayReductionResult(False, tuple(decisions))
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
            decisions.append(
                OverlayTurnDecisionRecord(
                    decision="overlay_turn_coalesced",
                    disposition="coalesced",
                    key=key,
                    entry=entry,
                    extras={"event_seq": event.seq},
                )
            )
            entry.last_updated_seq = event.seq
            return OverlayReductionResult(language_changed, tuple(decisions))

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
        return OverlayReductionResult(True, tuple(decisions))

    def apply_self_active_clear(
        self,
        event: SelfActiveClear,
        *,
        now: float,
        show_translation: bool,
    ) -> OverlayReductionResult:
        decisions: list[OverlayTurnDecisionRecord] = []
        live_self = self.live_entry_for_channel("self")
        if live_self is None:
            return OverlayReductionResult(False)
        key, entry = live_self
        if event.seq < entry.last_updated_seq:
            decisions.append(
                OverlayTurnDecisionRecord(
                    decision="overlay_turn_superseded",
                    disposition="superseded",
                    key=key,
                    entry=entry,
                    extras={"event_seq": event.seq, "last_updated_seq": entry.last_updated_seq},
                )
            )
            return OverlayReductionResult(False, tuple(decisions))
        if not entry.live_text:
            self.live_self_turn_key = None
            entry.last_updated_seq = event.seq
            decisions.append(
                OverlayTurnDecisionRecord(
                    decision="overlay_turn_coalesced",
                    disposition="coalesced",
                    key=key,
                    entry=entry,
                    extras={"event_seq": event.seq},
                )
            )
            return OverlayReductionResult(False, tuple(decisions))

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
        return OverlayReductionResult(True, tuple(decisions))

    def apply_self_finalized_update(
        self,
        event: SelfTranscriptFinal,
        *,
        now: float,
        show_translation: bool,
        next_appearance_seq: NextAppearanceSeq,
        terminal_update_reason: OverlayTerminalUpdateReason,
    ) -> OverlayReductionResult:
        decisions: list[OverlayTurnDecisionRecord] = []
        key = self.entry_key(event.channel, event.utterance_id)
        if self._append_terminal_update_decision(
            terminal_update_reason(event.channel, event.utterance_id),
            key=key,
            decisions=decisions,
        ):
            return OverlayReductionResult(False, tuple(decisions))
        entry = self.entry_for(event.channel, event.utterance_id)
        if entry.retained_hidden:
            return OverlayReductionResult(False)
        if event.seq < entry.last_updated_seq:
            decisions.append(
                OverlayTurnDecisionRecord(
                    decision="overlay_turn_superseded",
                    disposition="superseded",
                    key=key,
                    entry=entry,
                    extras={"event_seq": event.seq, "last_updated_seq": entry.last_updated_seq},
                )
            )
            return OverlayReductionResult(False, tuple(decisions))
        event_source_language = _content_language_or_none(event.source_language)
        if (
            entry.original_text == event.text
            and entry.original_language == event_source_language
            and entry.last_updated_seq == event.seq
        ):
            decisions.append(
                OverlayTurnDecisionRecord(
                    decision="overlay_turn_coalesced",
                    disposition="coalesced",
                    key=key,
                    entry=entry,
                    extras={"event_seq": event.seq},
                )
            )
            return OverlayReductionResult(False, tuple(decisions))

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
            decisions=decisions,
        )
        return OverlayReductionResult(True, tuple(decisions))

    def apply_self_translation_update(
        self,
        event: TranslationStreamUpdate | TranslationFinal,
        *,
        now: float,
        show_translation: bool,
        next_appearance_seq: NextAppearanceSeq,
        terminal_update_reason: OverlayTerminalUpdateReason,
    ) -> OverlayReductionResult:
        decisions: list[OverlayTurnDecisionRecord] = []
        key = self.entry_key(event.channel, event.utterance_id)
        if self._append_terminal_update_decision(
            terminal_update_reason(event.channel, event.utterance_id),
            key=key,
            decisions=decisions,
        ):
            return OverlayReductionResult(False, tuple(decisions))
        entry = self.entry_for(event.channel, event.utterance_id)
        if event.seq < entry.last_updated_seq:
            decisions.append(
                OverlayTurnDecisionRecord(
                    decision="overlay_turn_superseded",
                    disposition="superseded",
                    key=key,
                    entry=entry,
                    extras={"event_seq": event.seq, "last_updated_seq": entry.last_updated_seq},
                )
            )
            return OverlayReductionResult(False, tuple(decisions))
        event_source_language = _content_language_or_none(event.source_language)
        event_target_language = _content_language_or_none(event.target_language)
        if (
            entry.translation_text == event.text
            and (event_source_language is None or entry.original_language == event_source_language)
            and entry.translation_language == event_target_language
            and entry.last_updated_seq == event.seq
        ):
            decisions.append(
                OverlayTurnDecisionRecord(
                    decision="overlay_turn_coalesced",
                    disposition="coalesced",
                    key=key,
                    entry=entry,
                    extras={"event_seq": event.seq},
                )
            )
            return OverlayReductionResult(False, tuple(decisions))
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
            decisions=decisions,
        )
        return OverlayReductionResult(True, tuple(decisions))

    def apply_self_utterance_closed(
        self,
        event: UtteranceClosed,
        *,
        now: float,
        is_tombstoned: OverlayTerminalUpdatePredicate,
    ) -> OverlayReductionResult:
        decisions: list[OverlayTurnDecisionRecord] = []
        key = self.entry_key(event.channel, event.utterance_id)
        if is_tombstoned(event.channel, event.utterance_id):
            return OverlayReductionResult(False)
        entry = self.entries.get(key)
        if entry is None:
            return OverlayReductionResult(False)
        if event.seq < entry.last_updated_seq:
            decisions.append(
                OverlayTurnDecisionRecord(
                    decision="overlay_turn_superseded",
                    disposition="superseded",
                    key=key,
                    entry=entry,
                    extras={"event_seq": event.seq, "last_updated_seq": entry.last_updated_seq},
                )
            )
            return OverlayReductionResult(False, tuple(decisions))
        if entry.closed_seq == event.seq:
            decisions.append(
                OverlayTurnDecisionRecord(
                    decision="overlay_turn_coalesced",
                    disposition="coalesced",
                    key=key,
                    entry=entry,
                    extras={"event_seq": event.seq},
                )
            )
            return OverlayReductionResult(False, tuple(decisions))
        entry.closed_seq = event.seq
        entry.closed_at = now
        entry.last_updated_seq = event.seq
        return OverlayReductionResult(True, tuple(decisions))

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

        decisions: list[OverlayTurnDecisionRecord] = []
        active_entry = self._active_update_entry_or_none(
            channel=event.channel,
            utterance_id=event.utterance_id,
            event_seq=event.seq,
            now=now,
            show_translation=True,
            terminal_update_reason=terminal_update_reason,
            decisions=decisions,
        )
        if active_entry is None:
            return OverlayReductionResult(False, tuple(decisions))
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
            decisions.append(
                OverlayTurnDecisionRecord(
                    decision="overlay_turn_coalesced",
                    disposition="coalesced",
                    key=key,
                    entry=entry,
                    extras={"event_seq": event.seq},
                )
            )
            entry.last_updated_seq = event.seq
            return OverlayReductionResult(language_changed, tuple(decisions))

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
            decisions=decisions,
        )
        self.set_live_turn_key_for_channel(event.channel, key)
        return OverlayReductionResult(True, tuple(decisions))

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
        decisions: list[OverlayTurnDecisionRecord] = []
        key = self.entry_key(event.channel, event.utterance_id)
        if self._append_terminal_update_decision(
            terminal_update_reason(event.channel, event.utterance_id),
            key=key,
            decisions=decisions,
        ):
            return OverlayReductionResult(False, tuple(decisions))
        entry = self.entry_for(event.channel, event.utterance_id)
        if entry.retained_hidden:
            return OverlayReductionResult(False)
        if event.seq < entry.last_updated_seq:
            decisions.append(
                OverlayTurnDecisionRecord(
                    decision="overlay_turn_superseded",
                    disposition="superseded",
                    key=key,
                    entry=entry,
                    extras={"event_seq": event.seq, "last_updated_seq": entry.last_updated_seq},
                )
            )
            return OverlayReductionResult(False, tuple(decisions))
        event_source_language = _content_language_or_none(event.source_language)
        if (
            entry.original_text == event.text
            and entry.original_language == event_source_language
            and entry.last_updated_seq == event.seq
        ):
            decisions.append(
                OverlayTurnDecisionRecord(
                    decision="overlay_turn_coalesced",
                    disposition="coalesced",
                    key=key,
                    entry=entry,
                    extras={"event_seq": event.seq},
                )
            )
            return OverlayReductionResult(False, tuple(decisions))

        self._remember_entry_input_seq(entry, event_seq=event.seq)
        entry.original_text = event.text
        entry.original_language = event_source_language
        entry.original_seq = event.seq
        entry.live_text = ""
        entry.live_seq = None
        entry.last_updated_seq = event.seq
        self._refresh_entry_visibility_and_expiration(
            key,
            entry,
            now=now,
            publishable_seq=event.seq,
            next_appearance_seq=next_appearance_seq,
            show_peer_original=show_peer_original,
            translation_enabled=translation_enabled,
            decisions=decisions,
        )
        return OverlayReductionResult(True, tuple(decisions))

    def apply_peer_translation_update(
        self,
        event: TranslationStreamUpdate | TranslationFinal,
        *,
        now: float,
        show_peer_original: bool,
        next_appearance_seq: NextAppearanceSeq,
        terminal_update_reason: OverlayTerminalUpdateReason,
        translation_enabled: bool = True,
    ) -> OverlayReductionResult:
        decisions: list[OverlayTurnDecisionRecord] = []
        key = self.entry_key(event.channel, event.utterance_id)
        if self._append_terminal_update_decision(
            terminal_update_reason(event.channel, event.utterance_id),
            key=key,
            decisions=decisions,
        ):
            return OverlayReductionResult(False, tuple(decisions))
        entry = self.entry_for(event.channel, event.utterance_id)
        if event.seq < entry.last_updated_seq:
            decisions.append(
                OverlayTurnDecisionRecord(
                    decision="overlay_turn_superseded",
                    disposition="superseded",
                    key=key,
                    entry=entry,
                    extras={"event_seq": event.seq, "last_updated_seq": entry.last_updated_seq},
                )
            )
            return OverlayReductionResult(False, tuple(decisions))
        event_source_language = _content_language_or_none(event.source_language)
        event_target_language = _content_language_or_none(event.target_language)
        if (
            entry.translation_text == event.text
            and (event_source_language is None or entry.original_language == event_source_language)
            and entry.translation_language == event_target_language
            and entry.last_updated_seq == event.seq
        ):
            decisions.append(
                OverlayTurnDecisionRecord(
                    decision="overlay_turn_coalesced",
                    disposition="coalesced",
                    key=key,
                    entry=entry,
                    extras={"event_seq": event.seq},
                )
            )
            return OverlayReductionResult(False, tuple(decisions))

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
        if event.text.strip():
            entry.translation_language = event_target_language
            entry.translation_update_id = event.update_id
            entry.translation_origin_wall_clock_ms = event.origin_wall_clock_ms
            entry.translation_session_scope = event.session_scope
            entry.translation_source_text_hash = event.source_text_hash
            entry.translation_source_text_len = event.source_text_len
            entry.translation_logical_turn_key = event.logical_turn_key
            entry.translation_seq = event.seq
            entry.live_text = ""
            entry.live_seq = None
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
        if event.text.strip() and entry.translation_observed_visible_since is None:
            entry.translation_observed_visible_since = now
        entry.last_updated_seq = event.seq
        self._refresh_entry_visibility_and_expiration(
            key,
            entry,
            now=now,
            publishable_seq=event.seq,
            next_appearance_seq=next_appearance_seq,
            show_peer_original=show_peer_original,
            translation_enabled=translation_enabled,
            decisions=decisions,
        )
        return OverlayReductionResult(True, tuple(decisions))

    def apply_peer_utterance_closed(
        self,
        event: UtteranceClosed,
        *,
        now: float,
        is_tombstoned: OverlayTerminalUpdatePredicate,
    ) -> OverlayReductionResult:
        decisions: list[OverlayTurnDecisionRecord] = []
        key = self.entry_key(event.channel, event.utterance_id)
        if is_tombstoned(event.channel, event.utterance_id):
            return OverlayReductionResult(False)
        entry = self.entries.get(key)
        if entry is None:
            return OverlayReductionResult(False)
        if event.seq < entry.last_updated_seq:
            decisions.append(
                OverlayTurnDecisionRecord(
                    decision="overlay_turn_superseded",
                    disposition="superseded",
                    key=key,
                    entry=entry,
                    extras={"event_seq": event.seq, "last_updated_seq": entry.last_updated_seq},
                )
            )
            return OverlayReductionResult(False, tuple(decisions))
        if entry.closed_seq == event.seq:
            decisions.append(
                OverlayTurnDecisionRecord(
                    decision="overlay_turn_coalesced",
                    disposition="coalesced",
                    key=key,
                    entry=entry,
                    extras={"event_seq": event.seq},
                )
            )
            return OverlayReductionResult(False, tuple(decisions))
        entry.closed_seq = event.seq
        entry.closed_at = now
        entry.last_updated_seq = event.seq
        return OverlayReductionResult(True, tuple(decisions))

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
            for key, entry in self.entries.items()
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
            entry = self.entries.get(key)
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
        entry_ordering: Mapping[OverlayEntryKey, tuple[str, int, int]] | None = None,
        semantic_retirement_frontiers: Mapping[tuple[str, int], int] | None = None,
    ) -> OverlayPresentationSnapshot:
        ordering = entry_ordering or {}
        blocks = []
        for key, block in rendered_entries:
            block_ordering = ordering.get(key)
            if block_ordering is not None:
                scope, generation, order = block_ordering
                block = replace(
                    block,
                    publication_scope=scope,
                    publication_generation=generation,
                    publication_order=order,
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
                SemanticRetirementFrontier(scope=scope, generation=generation, order=order)
                for (scope, generation), order in sorted(
                    (semantic_retirement_frontiers or {}).items()
                )
            ],
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
        peer_presentation_refresh_burst: bool,
        next_appearance_seq: NextAppearanceSeq,
        self_presentation_refresh_burst: bool = True,
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
                    peer_presentation_refresh_burst=peer_presentation_refresh_burst,
                    self_presentation_refresh_burst=self_presentation_refresh_burst,
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
                peer_presentation_refresh_burst=peer_presentation_refresh_burst,
                self_presentation_refresh_burst=self_presentation_refresh_burst,
                translation_enabled=translation_enabled,
            )
            if block is None:
                continue
            rendered_entries.append((protected_key, block))
        rendered_entries.sort(key=lambda item: (item[1].appearance_seq, item[1].occupant_key))
        retained_hidden = [key for key, entry in entries.items() if entry.retained_hidden]
        return OverlayVisibleBlockSelection(
            rendered_entries=rendered_entries,
            active_self_present=active_self_present,
            finalized_limit=finalized_limit,
            candidate_keys=candidate_keys,
            selected_keys=visible_entry_keys,
            protected_keys=protected_keys,
            retained_hidden=retained_hidden,
        )

    def entry_is_publishable(
        self,
        entry: OverlayPresentationEntry,
        *,
        show_peer_original: bool,
        translation_enabled: bool = True,
    ) -> bool:
        if entry.channel == "peer":
            if not translation_enabled:
                return bool(entry.live_text.strip() or entry.original_text.strip())
            return bool(
                entry.translation_text.strip()
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
        )

    def rendered_block_signature(
        self,
        block: OverlayPresentationBlock,
    ) -> tuple[object, ...]:
        secondary_text = block.secondary_text if block.secondary_enabled else ""
        include_translation_metadata = block.channel == "peer" or bool(secondary_text)
        include_self_refresh_metadata = (
            block.channel == "self"
            and block.block_variant == "finalized"
            and _session_scope_has_presentation_refresh_marker(
                block.session_scope,
                marker_prefix="self_presentation_refresh=",
            )
        )
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
            (
                block.session_scope
                if include_translation_metadata or include_self_refresh_metadata
                else None
            ),
            block.source_text_hash if include_translation_metadata else None,
            block.source_text_len if include_translation_metadata else None,
            block.logical_turn_key if include_translation_metadata else None,
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

    def _append_terminal_update_decision(
        self,
        terminal_reason: str | None,
        *,
        key: OverlayEntryKey,
        decisions: list[OverlayTurnDecisionRecord],
    ) -> bool:
        if terminal_reason is None:
            return False
        if terminal_reason == "evicted_by_newer_turn":
            decisions.append(
                OverlayTurnDecisionRecord(
                    decision="overlay_turn_late_update_ignored_after_eviction",
                    disposition="evicted",
                    key=key,
                    extras={"terminal_reason": terminal_reason},
                )
            )
        elif terminal_reason == "expired":
            decisions.append(
                OverlayTurnDecisionRecord(
                    decision="overlay_turn_late_update_ignored_after_idle_hide",
                    disposition="hidden_idle_ttl",
                    key=key,
                    extras={"terminal_reason": terminal_reason},
                )
            )
        return True

    def _active_update_entry_or_none(
        self,
        *,
        channel: str,
        utterance_id: UUID | None,
        event_seq: int,
        now: float,
        show_translation: bool,
        terminal_update_reason: OverlayTerminalUpdateReason,
        decisions: list[OverlayTurnDecisionRecord],
    ) -> tuple[OverlayEntryKey, OverlayLogicalTurnEntry] | None:
        key = self.entry_key(channel, utterance_id)
        if self._append_terminal_update_decision(
            terminal_update_reason(channel, utterance_id),
            key=key,
            decisions=decisions,
        ):
            return None
        live_entry = self.live_entry_for_channel(channel)
        if live_entry is not None:
            live_key, current_live_entry = live_entry
            if live_key != key and event_seq < current_live_entry.last_updated_seq:
                decisions.append(
                    OverlayTurnDecisionRecord(
                        decision="overlay_turn_superseded",
                        disposition="superseded",
                        key=key,
                        entry=current_live_entry,
                        extras={
                            "event_seq": event_seq,
                            "superseded_by_entry": self._format_entry_key(live_key),
                            "superseded_by_seq": current_live_entry.last_updated_seq,
                        },
                    )
                )
                return None

        entry = self.entry_for(channel, utterance_id)
        if event_seq < entry.last_updated_seq:
            decisions.append(
                OverlayTurnDecisionRecord(
                    decision="overlay_turn_superseded",
                    disposition="superseded",
                    key=key,
                    entry=entry,
                    extras={"event_seq": event_seq, "last_updated_seq": entry.last_updated_seq},
                )
            )
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
        decisions: list[OverlayTurnDecisionRecord],
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
        else:
            decisions.append(
                OverlayTurnDecisionRecord(
                    decision="overlay_turn_not_yet_publishable",
                    key=key,
                    entry=entry,
                )
            )

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
        next_appearance_seq: NextAppearanceSeq,
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
            self._ensure_entry_visibility_metadata(
                entry,
                occupant_key=self._finalized_occupant_key(entry.channel, entry.utterance_id),
                next_appearance_seq=next_appearance_seq,
            )
            if entry.publishable_seq is None or entry.appearance_seq is None:
                continue
            publishable.append(
                (
                    entry.publishable_seq,
                    entry.appearance_seq,
                    entry.occupant_key,
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
        peer_presentation_refresh_burst: bool,
        self_presentation_refresh_burst: bool = True,
        translation_enabled: bool = True,
    ) -> OverlayPresentationBlock | None:
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
                        session_scope=self._peer_session_scope_with_presentation_refresh(
                            entry,
                            None,
                            peer_presentation_refresh_burst=peer_presentation_refresh_burst,
                        ),
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
                        primary_text=finalized_source_text,
                        secondary_text="",
                        secondary_enabled=False,
                        primary_language=_line_language(
                            entry.original_language, finalized_source_text
                        ),
                        secondary_language=None,
                        session_scope=self._peer_session_scope_with_presentation_refresh(
                            entry,
                            None,
                            peer_presentation_refresh_burst=peer_presentation_refresh_burst,
                        ),
                    )
                return None
            translated_text = entry.translation_text.strip()
            original_text = entry.original_text.strip() or entry.live_text.strip()
            if translated_text:
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
                    primary_text=translated_text,
                    secondary_text=original_text,
                    secondary_enabled=show_peer_original and bool(original_text),
                    primary_language=_line_language(
                        entry.translation_language,
                        translated_text,
                    ),
                    secondary_language=_line_language(
                        entry.original_language,
                        original_text,
                        enabled=show_peer_original,
                    ),
                    update_id=entry.translation_update_id,
                    origin_wall_clock_ms=entry.translation_origin_wall_clock_ms,
                    session_scope=self._peer_session_scope_with_presentation_refresh(
                        entry,
                        entry.translation_session_scope,
                        peer_presentation_refresh_burst=peer_presentation_refresh_burst,
                    ),
                    source_text_hash=entry.translation_source_text_hash,
                    source_text_len=entry.translation_source_text_len,
                    logical_turn_key=entry.translation_logical_turn_key,
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
                    secondary_text=active_text,
                    secondary_enabled=True,
                    primary_language=None,
                    secondary_language=_line_language(entry.original_language, active_text),
                    session_scope=self._peer_session_scope_with_presentation_refresh(
                        entry,
                        None,
                        peer_presentation_refresh_burst=peer_presentation_refresh_burst,
                    ),
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
                    secondary_text=original_text,
                    secondary_enabled=True,
                    primary_language=None,
                    secondary_language=_line_language(entry.original_language, original_text),
                    session_scope=self._peer_session_scope_with_presentation_refresh(
                        entry,
                        None,
                        peer_presentation_refresh_burst=peer_presentation_refresh_burst,
                    ),
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
            session_scope=self._self_session_scope_with_presentation_refresh(
                entry,
                entry.translation_session_scope,
                primary_text=primary_text,
                block_variant="finalized",
                self_presentation_refresh_burst=self_presentation_refresh_burst,
            ),
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
        entry: OverlayPresentationEntry,
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

    def _peer_session_scope_with_presentation_refresh(
        self,
        entry: OverlayPresentationEntry,
        session_scope: str | None,
        *,
        peer_presentation_refresh_burst: bool,
    ) -> str | None:
        if (
            not peer_presentation_refresh_burst
            or self.peer_presentation_refresh_nonce <= 0
            or self.peer_presentation_refresh_target_key != (entry.channel, entry.utterance_id)
        ):
            return session_scope
        # LOAD-BEARING: this marker is not cosmetic metadata. The 2026-04-28
        # submit-only resubmit regression showed stored-frame resubmits are not
        # equivalent to fresh snapshot/render/GPU work, so each nonce value must
        # produce revision-worthy session_scope metadata for native to render.
        marker = f"peer_presentation_refresh={self.peer_presentation_refresh_nonce}"
        if session_scope:
            return f"{session_scope}|{marker}"
        return marker

    def _self_session_scope_with_presentation_refresh(
        self,
        entry: OverlayPresentationEntry,
        session_scope: str | None,
        *,
        primary_text: str,
        block_variant: str,
        self_presentation_refresh_burst: bool,
    ) -> str | None:
        if (
            not self_presentation_refresh_burst
            or self.self_presentation_refresh_nonce <= 0
            or self.self_presentation_refresh_target_key != (entry.channel, entry.utterance_id)
            or entry.channel != "self"
            or block_variant != "finalized"
            or not primary_text.strip()
        ):
            return session_scope
        marker = f"self_presentation_refresh={self.self_presentation_refresh_nonce}"
        if session_scope:
            return f"{session_scope}|{marker}"
        return marker

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


def _session_scope_has_presentation_refresh_marker(
    session_scope: str | None,
    *,
    marker_prefix: str,
) -> bool:
    if session_scope is None:
        return False
    return any(part.startswith(marker_prefix) for part in session_scope.split("|"))
