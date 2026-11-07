from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from uuid import UUID, uuid4

import pytest

from puripuly_heart.config.overlay_calibration import OverlayCalibration
from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.overlay.peer_caption_layout import (
    estimate_peer_translation_lines,
    peer_translation_pages,
)
from puripuly_heart.core.overlay.presenter import OverlayPresenter
from puripuly_heart.core.overlay.protocol import OverlayPresentationSnapshot
from puripuly_heart.core.overlay.sink import (
    PeerCaptionAdmissionPort,
    PeerCaptionLayoutPort,
    SelfActiveUpdate,
    TranslationFinal,
    TranslationStreamUpdate,
)
from puripuly_heart.domain.models import SpeakerAssignment, SpeakerAttribution, SpeakerKey


@dataclass
class Timeline:
    clock: FakeClock = field(default_factory=FakeClock)
    sleepers: list[tuple[float, asyncio.Future[None]]] = field(default_factory=list)

    async def sleep(self, seconds: float) -> None:
        future = asyncio.get_running_loop().create_future()
        self.sleepers.append((self.clock.now() + seconds, future))
        await future

    async def advance(self, seconds: float) -> None:
        self.clock.advance(seconds)
        for _ in range(12):
            for deadline, future in self.sleepers:
                if deadline <= self.clock.now() and not future.done():
                    future.set_result(None)
            await asyncio.sleep(0)


@dataclass
class Surface:
    snapshots: list[OverlayPresentationSnapshot] = field(default_factory=list)
    expirations: dict[str, float | None] = field(default_factory=dict)

    async def replace_snapshot(self, snapshot, *, block_expirations=None) -> None:
        self.snapshots.append(snapshot)
        self.expirations = dict(block_expirations or {})

    async def broadcast_shutdown(self) -> None:
        return None


def caption(
    seq: int,
    unit: UUID,
    text: str,
    *,
    revision: int = 1,
    final: bool = True,
    superseded: tuple[UUID, ...] = (),
    source: str = "source",
    assignment: SpeakerAssignment | None = None,
) -> TranslationStreamUpdate:
    kind = TranslationFinal if final else TranslationStreamUpdate
    return kind(
        event_id=str(uuid4()),
        seq=seq,
        utterance_id=unit,
        channel="peer",
        created_at=0.0,
        text=text,
        source_text=source,
        source_language="en",
        target_language="ko",
        peer_caption_revision=revision,
        peer_superseded_unit_ids=superseded,
        speaker_assignment=assignment,
    )


def presenter_for(timeline: Timeline, surface: Surface | None = None) -> OverlayPresenter:
    return OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=timeline.clock,
        sleep=timeline.sleep,
        bridge=surface,
    )


@pytest.mark.asyncio
async def test_pages_turn_on_timer_and_upper_protection_follows_actual_page_change() -> None:
    timeline, surface = Timeline(), Surface()
    presenter = presenter_for(timeline, surface)
    first, second = uuid4(), uuid4()
    text = "가" * 56 + "나" * 56 + "다" * 20
    pages = peer_translation_pages(text)
    try:
        await presenter.emit(caption(1, first, text))
        await timeline.advance(1.5)
        await presenter.emit(caption(2, second, "lower"))
        assert [block.id for block in presenter.snapshot().blocks] == [
            f"peer:{first}",
            f"peer:{second}",
        ]
        assert surface.expirations[f"peer:{first}"] == pytest.approx(3.5)
        await timeline.advance(1.5)
        assert presenter.snapshot().blocks[0].primary_text == pages[1]
        assert surface.expirations[f"peer:{first}"] == pytest.approx(6.0)
        await timeline.advance(0.5)
        assert presenter.snapshot().blocks[0].id == f"peer:{first}"
        await timeline.advance(2.5)
        assert presenter.snapshot().blocks[0].primary_text == pages[2]
        assert surface.expirations[f"peer:{first}"] > 6.0
    finally:
        await presenter.close()


@pytest.mark.asyncio
async def test_correction_extends_expiry_and_obsolete_timer_cannot_remove_current_version() -> None:
    timeline, surface = Timeline(), Surface()
    presenter = presenter_for(timeline, surface)
    upper, lower = uuid4(), uuid4()
    try:
        await presenter.emit(caption(1, upper, "short"))
        await presenter.emit(caption(2, lower, "lower"))
        old_revision = presenter._entries[("peer", upper)].expiration_revision
        assert surface.expirations[f"peer:{upper}"] == 2.0
        await timeline.advance(1.7)
        receipt = await presenter.emit_peer_when_admissible(
            caption(3, upper, "정" * 40, revision=2)
        )
        assert receipt.outcome == "applied"
        assert timeline.clock.now() == 1.7
        assert surface.expirations[f"peer:{upper}"] == pytest.approx(4.7)
        await presenter._expire_entry_after_ttl(("peer", upper), old_revision)
        await timeline.advance(0.4)
        assert presenter.snapshot().blocks[0].primary_text == "정" * 28 + "\n" + "정" * 12
        await timeline.advance(2.6)
        assert all(block.id != f"peer:{upper}" for block in presenter.snapshot().blocks)
    finally:
        await presenter.close()


@pytest.mark.asyncio
async def test_new_preview_waits_at_most_one_second_then_evicts_protected_upper() -> None:
    timeline = Timeline()
    presenter = presenter_for(timeline)
    first, second, third = uuid4(), uuid4(), uuid4()
    pending = None
    try:
        await presenter.emit(caption(1, first, "가" * 40))
        await presenter.emit(caption(2, second, "lower"))
        pending = asyncio.create_task(
            presenter.emit_peer_when_admissible(caption(3, third, "new preview", final=False))
        )
        await timeline.advance(0)
        assert not pending.done()
        await timeline.advance(0.9)
        assert not pending.done()
        await timeline.advance(0.1)
        assert pending.done()
        assert (await pending).outcome == "applied"
        assert [block.id for block in presenter.snapshot().blocks] == [
            f"peer:{second}",
            f"peer:{third}",
        ]
        stale = await presenter.emit(caption(4, first, "late repair", revision=2))
        assert stale.outcome == "stale"
    finally:
        if pending is not None and not pending.done():
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        await presenter.close()


@pytest.mark.asyncio
async def test_new_version_restarts_pages_and_settlement_is_immediate() -> None:
    timeline = Timeline()
    presenter = presenter_for(timeline)
    unit = uuid4()
    text = "가" * 56 + "나" * 56
    try:
        await presenter.emit_peer_when_admissible(caption(1, unit, text, final=False))
        await timeline.advance(3.0)
        assert presenter.snapshot().blocks[0].primary_text == peer_translation_pages(text)[1]
        await presenter.emit_peer_when_admissible(
            caption(2, unit, text + "다", revision=2, final=False)
        )
        assert presenter.snapshot().blocks[0].primary_text == peer_translation_pages(text)[0]
        await timeline.advance(0.2)
        await presenter.emit_peer_when_admissible(caption(3, unit, "settled", revision=3))
        assert timeline.clock.now() == 3.2
        assert presenter.snapshot().blocks[0].primary_text == "settled"
        await timeline.advance(3.0)
        assert presenter.snapshot().blocks[0].primary_text == "settled"
        stale = await presenter.emit(caption(4, unit, "obsolete", revision=2))
        preview_after_final = await presenter.emit(
            caption(5, unit, "late preview", revision=4, final=False)
        )
        assert stale.outcome == preview_after_final.outcome == "stale"
    finally:
        await presenter.close()


@pytest.mark.asyncio
async def test_repair_absorbs_preview_atomically_and_stale_chained_repair_cannot_resurrect_it() -> (
    None
):
    timeline, surface = Timeline(), Surface()
    presenter = presenter_for(timeline, surface)
    unit, preview, next_preview = uuid4(), uuid4(), uuid4()
    try:
        await presenter.emit(caption(1, unit, "first"))
        await presenter.emit(caption(2, preview, "preview", final=False))
        await presenter.emit_peer_when_admissible(
            caption(3, unit, "first repaired", revision=2, superseded=(preview,))
        )
        assert len(surface.snapshots[-1].blocks) == 1
        assert surface.snapshots[-1].blocks[0].primary_text == "first repaired"
        await presenter.emit(caption(4, next_preview, "next preview", final=False))
        await presenter.emit_peer_when_admissible(
            caption(5, unit, "chained repair", revision=4, superseded=(preview, next_preview))
        )
        old = await presenter.emit(
            caption(6, unit, "old chained repair", revision=3, superseded=(preview,))
        )
        retired = [
            await presenter.emit(caption(7 + index, target, "late", revision=10))
            for index, target in enumerate((preview, next_preview))
        ]
        assert old.outcome == "stale"
        assert all(receipt.outcome == "stale" for receipt in retired)
        assert [(block.id, block.primary_text) for block in presenter.snapshot().blocks] == [
            (f"peer:{unit}", "chained repair")
        ]
    finally:
        await presenter.close()


@pytest.mark.asyncio
async def test_self_preempts_peer_protection_without_waiting_and_keeps_its_translation() -> None:
    timeline = Timeline()
    presenter = presenter_for(timeline)
    first, second, own = uuid4(), uuid4(), uuid4()
    try:
        await presenter.emit(caption(1, first, "가" * 40))
        await presenter.emit(caption(2, second, "나" * 40))
        await presenter.emit(
            SelfActiveUpdate(
                event_id=str(uuid4()),
                seq=3,
                utterance_id=own,
                channel="self",
                created_at=0,
                text="Self source",
                secondary_text="Self translation",
                occupant_key=f"self:{own}",
                source_language="en",
                target_language="ko",
                caption_revision=1,
                caption_source_closed=False,
            )
        )
        assert timeline.clock.now() == 0
        assert [block.id for block in presenter.snapshot().blocks] == [
            f"peer:{second}",
            f"self:{own}",
        ]
        assert presenter.snapshot().blocks[-1].secondary_text == "Self translation"
        await timeline.advance(0.2)
        await presenter.emit(caption(4, uuid4(), "other peer"))
        assert any(block.id == f"self:{own}" for block in presenter.snapshot().blocks)
        assert (
            next(
                block for block in presenter.snapshot().blocks if block.channel == "self"
            ).secondary_text
            == "Self translation"
        )
    finally:
        await presenter.close()


@pytest.mark.asyncio
async def test_genuine_repeated_units_remain_distinct_and_new_visible_occurrence_resets_budget() -> (
    None
):
    timeline = Timeline()
    presenter = presenter_for(timeline)
    first, second = uuid4(), uuid4()
    try:
        await presenter.emit(caption(1, first, "yes", source=""))
        await presenter.emit(caption(2, second, "yes", source=""))
        assert len(presenter.snapshot().blocks) == 2
        await timeline.advance(0.5)
        await presenter.update_translation_enabled(False)
        assert presenter.snapshot().blocks == []
        await timeline.advance(0.4)
        await presenter.update_translation_enabled(True)
        entry = presenter._entries[("peer", first)]
        assert entry.visible_since == pytest.approx(0.9)
        assert entry.peer_protected_until == pytest.approx(2.9)
        await timeline.advance(1.2)
        assert len(presenter.snapshot().blocks) == 2
    finally:
        await presenter.close()


@pytest.mark.asyncio
async def test_metadata_only_version_does_not_extend_actual_upper_reading_clock() -> None:
    timeline = Timeline()
    presenter = presenter_for(timeline)
    upper, lower = uuid4(), uuid4()
    try:
        await presenter.emit(caption(1, upper, "same page"))
        await presenter.emit(caption(2, lower, "lower"))
        await timeline.advance(0.8)
        await presenter.emit(caption(3, upper, "same page", revision=2))
        entry = presenter._entries[("peer", upper)]
        assert entry.last_meaningful_visible_at == 0.0
        assert entry.peer_protected_until == 2.0
        await timeline.advance(1.2)
        assert all(block.id != f"peer:{upper}" for block in presenter.snapshot().blocks)
    finally:
        await presenter.close()


@pytest.mark.asyncio
async def test_calibration_changes_reflow_peer_pages_on_three_second_timer() -> None:
    timeline = Timeline()
    presenter = presenter_for(timeline)
    unit = uuid4()
    text = "가" * 100
    try:
        await presenter.emit(caption(1, unit, text))
        await presenter.update_calibration(OverlayCalibration(text_scale=2.0))
        assert (
            presenter.snapshot().blocks[0].primary_text
            == peer_translation_pages(text, text_scale=2.0)[0]
        )
        await timeline.advance(3.0)
        assert (
            presenter.snapshot().blocks[0].primary_text
            == peer_translation_pages(text, text_scale=2.0)[1]
        )
    finally:
        await presenter.close()


@pytest.mark.asyncio
async def test_effective_layout_port_tracks_supported_large_calibration_in_join_estimate() -> None:
    timeline = Timeline()
    presenter = presenter_for(timeline)
    source = "가" * 50
    try:
        assert isinstance(presenter, PeerCaptionLayoutPort)
        initial_lines = estimate_peer_translation_lines(
            source, "ko", "ko", text_scale=presenter.peer_caption_text_scale
        )
        await presenter.update_calibration(OverlayCalibration(text_scale=1.2))
        effective_lines = estimate_peer_translation_lines(
            source, "ko", "ko", text_scale=presenter.peer_caption_text_scale
        )
        assert presenter.peer_caption_text_scale == 1.2
        assert initial_lines <= 2 < effective_lines
        assert effective_lines == pytest.approx(initial_lines * 1.2)
        assert len(peer_translation_pages(source)) == 1
        assert (
            len(peer_translation_pages(source, text_scale=presenter.peer_caption_text_scale)) == 2
        )
    finally:
        await presenter.close()


def overflow_assignment(speaker: str, *, revision: str = "first-final") -> SpeakerAssignment:
    return SpeakerAssignment(
        attribution=SpeakerAttribution(
            state="identified",
            key=SpeakerKey("soniox", "same-scope", speaker),
            source="soniox",
            speaker_scope_id="same-scope",
        ),
        scope_order=(1, 1, 1),
        palette_overflow=True,
        source_text_range=(0, 6),
        source_text_revision=revision,
    )


def native_assignment(speaker: str | None, *, scope: str = "same-scope", order: int = 1):
    return SpeakerAssignment(
        SpeakerAttribution(
            "identified" if speaker is not None else "missing",
            SpeakerKey("soniox", scope, speaker) if speaker is not None else None,
            "soniox",
            scope,
        ),
        scope_order=(1, order, 0),
        palette_index=1 if speaker is not None else None,
    )


@pytest.mark.asyncio
async def test_native_preview_final_and_source_only_repair_settle_matching_presentation() -> None:
    timeline, surface = Timeline(), Surface()
    presenter = presenter_for(timeline, surface)
    unit = uuid4()
    known = native_assignment("B")
    try:
        await presenter.emit(
            caption(1, unit, "preview", final=False, source="provisional", assignment=known)
        )
        assert presenter.snapshot().blocks[0].speaker_style == "gray"
        await presenter.emit(
            caption(2, unit, "확정 번역", revision=2, source="final source", assignment=known)
        )
        settled = surface.snapshots[-1]
        assert settled.blocks[0].primary_text == "확정 번역"
        assert settled.blocks[0].secondary_text == "final source"
        assert settled.blocks[0].speaker_style == "cyan"
        assert presenter._entries[("peer", unit)].speaker_assignment == known
        assert presenter._entries[("peer", unit)].speaker_gray_reason is None
        await presenter.emit(
            caption(3, unit, "", revision=3, source="source-only repair", assignment=known)
        )
        repaired = presenter.snapshot()
        assert repaired.blocks[0].primary_text == "source-only repair"
        assert repaired.blocks[0].speaker_style == "cyan"
        assert presenter._entries[("peer", unit)].peer_source_only is True
        stale = await presenter.emit(
            caption(4, unit, "obsolete", revision=2, assignment=native_assignment(None))
        )
        assert stale.outcome == "stale"
        assert presenter.snapshot() == repaired
        payload = repaired.to_dict()
        assert payload["blocks"][0]["speaker_style"] == "cyan"
        assert "speaker_scope_id" not in str(payload)
        assert "local_speaker_id" not in str(payload)
    finally:
        await presenter.close()


@pytest.mark.asyncio
async def test_native_attribution_repair_and_duplicate_preserve_current_page_and_lifetime() -> None:
    timeline, surface = Timeline(), Surface()
    presenter = presenter_for(timeline, surface)
    unit = uuid4()
    text = "가" * 56 + "나" * 56 + "다" * 20
    try:
        await presenter.emit(caption(1, unit, text, assignment=native_assignment(None)))
        await timeline.advance(3.0)
        entry = presenter._entries[("peer", unit)]
        assert entry.peer_page_index == 1
        before = (
            entry.peer_page_due_at,
            entry.last_meaningful_visible_at,
            entry.visible_since,
            entry.appearance_seq,
            surface.expirations[f"peer:{unit}"],
        )
        await timeline.advance(0.5)
        known = native_assignment("B")
        accepted = caption(2, unit, text, revision=2, assignment=known)
        assert (await presenter.emit(accepted)).outcome == "applied"
        assert entry.speaker_style == "cyan"
        assert entry.speaker_gray_reason is None
        assert entry.peer_page_index == 1
        assert presenter.snapshot().blocks[0].primary_text == peer_translation_pages(text)[1]
        assert before == (
            entry.peer_page_due_at,
            entry.last_meaningful_visible_at,
            entry.visible_since,
            entry.appearance_seq,
            surface.expirations[f"peer:{unit}"],
        )
        snapshot = presenter.snapshot()
        duplicate = replace(
            accepted, event_id=str(uuid4()), seq=3, speaker_assignment=native_assignment(None)
        )
        assert (await presenter.emit(duplicate)).outcome == "stale"
        assert presenter.snapshot() == snapshot
        assert entry.speaker_assignment == known
        await timeline.advance(2.5)
        assert entry.peer_page_index == 2
        await timeline.advance(30.0)
        assert not presenter.snapshot().blocks
        late = await presenter.emit(caption(4, unit, text, revision=3, assignment=known))
        assert late.outcome == "stale"
        assert not presenter.snapshot().blocks
    finally:
        await presenter.close()


@pytest.mark.asyncio
async def test_native_gray_repairs_gain_overflow_divider_without_extending_reading() -> None:
    timeline, surface = Timeline(), Surface()
    presenter = presenter_for(timeline, surface)
    first, second = uuid4(), uuid4()
    try:
        await presenter.emit(caption(1, first, "first", assignment=native_assignment(None)))
        await presenter.emit(caption(2, second, "second", assignment=native_assignment(None)))
        deadlines = dict(surface.expirations)
        await timeline.advance(0.5)
        await presenter.emit(
            caption(3, first, "first", revision=2, assignment=overflow_assignment("E"))
        )
        assert presenter.snapshot().speaker_divider is False
        await presenter.emit(
            caption(4, second, "second", revision=2, assignment=overflow_assignment("F"))
        )
        snapshot = presenter.snapshot()
        assert [block.speaker_style for block in snapshot.blocks] == ["gray", "gray"]
        assert snapshot.speaker_divider is True
        assert surface.snapshots[-1].to_dict()["speaker_divider"] is True
        assert surface.expirations == deadlines
        assert all(
            entry.speaker_gray_reason == "palette_overflow" for entry in presenter._entries.values()
        )
    finally:
        await presenter.close()


@pytest.mark.asyncio
async def test_native_revisions_preserve_handoff_gray_after_old_color_leaves() -> None:
    timeline = Timeline()
    presenter = presenter_for(timeline)
    old, handoff, later = uuid4(), uuid4(), uuid4()
    old_assignment = native_assignment("A", scope="old", order=1)
    new_assignment = native_assignment("A", scope="new", order=2)
    try:
        await presenter.emit(caption(1, old, "old", assignment=old_assignment))
        await presenter.emit(caption(2, handoff, "new", assignment=new_assignment))
        assert presenter._entries[("peer", handoff)].speaker_gray_reason == "scope_handoff"
        await presenter.emit(
            caption(3, old, "old corrected", revision=2, assignment=old_assignment)
        )
        assert presenter._entries[("peer", old)].speaker_style == "cyan"
        await timeline.advance(1)
        await presenter.emit(
            caption(4, later, "later", assignment=replace(new_assignment, scope_order=(1, 3, 0)))
        )
        assert ("peer", old) not in presenter._entries
        await presenter.emit(
            caption(
                5,
                handoff,
                "new corrected",
                revision=2,
                assignment=replace(new_assignment, source_text_revision="corrected"),
            )
        )
        assert presenter._entries[("peer", handoff)].speaker_gray_reason == "scope_handoff"
        assert presenter._entries[("peer", handoff)].speaker_style == "gray"
        assert presenter._entries[("peer", later)].speaker_style == "cyan"
        await presenter.emit(
            caption(
                6,
                handoff,
                "now unknown",
                revision=3,
                assignment=native_assignment(None, scope="new", order=2),
            )
        )
        assert presenter._entries[("peer", handoff)].speaker_gray_reason == "missing"
        await presenter.emit(
            caption(7, handoff, "known again", revision=4, assignment=new_assignment)
        )
        assert presenter._entries[("peer", handoff)].speaker_gray_reason == "scope_handoff"
        assert presenter._entries[("peer", handoff)].speaker_style == "gray"
    finally:
        await presenter.close()


@pytest.mark.asyncio
async def test_other_provider_caption_revisions_keep_first_readable_style() -> None:
    timeline = Timeline()
    presenter = presenter_for(timeline)
    unit = uuid4()
    missing = SpeakerAssignment(SpeakerAttribution("missing", source="other"))
    known = replace(native_assignment("B"), attribution=SpeakerAttribution("non_diarized"))
    try:
        await presenter.emit(caption(1, unit, "first", assignment=missing))
        await presenter.emit(caption(2, unit, "repaired", revision=2, assignment=known))
        assert presenter._entries[("peer", unit)].speaker_style == "gray"
        assert presenter.snapshot().blocks[0].primary_text == "repaired"
    finally:
        await presenter.close()


@pytest.mark.asyncio
async def test_cross_final_mixed_repair_clears_semantic_ranges_and_overflow_divider() -> None:
    timeline = Timeline()
    presenter = presenter_for(timeline)
    repaired, other = uuid4(), uuid4()
    initial_assignment = overflow_assignment("speaker-5")
    mixed = SpeakerAssignment(
        attribution=SpeakerAttribution("mixed", source="soniox", speaker_scope_id="same-scope"),
        scope_order=(1, 1, 1),
    )
    try:
        await presenter.emit(
            caption(1, repaired, "first", source="alpha,", assignment=initial_assignment)
        )
        await presenter.emit(
            caption(2, other, "second", assignment=overflow_assignment("speaker-6"))
        )
        first = presenter._entries[("peer", repaired)]
        initial_style = first.speaker_style
        assert initial_style == "gray"
        assert presenter.snapshot().speaker_divider is True
        receipt = await presenter.emit_peer_when_admissible(
            caption(3, repaired, "combined", revision=2, source="alpha, beta", assignment=mixed)
        )
        assert receipt.outcome == "applied"
        assert first.original_text == "alpha, beta"
        assert first.speaker_assignment is mixed
        assert first.speaker_assignment.attribution.state == "mixed"
        assert first.speaker_assignment.source_text_range is None
        assert first.speaker_assignment.source_text_revision is None
        assert first.speaker_gray_reason == "mixed"
        assert presenter.snapshot().speaker_divider is False
        stale = await presenter.emit(
            caption(4, repaired, "obsolete", revision=1, assignment=initial_assignment)
        )
        assert stale.outcome == "stale"
        assert first.speaker_assignment is mixed
        assert presenter.snapshot().speaker_divider is False
    finally:
        await presenter.close()


@pytest.mark.asyncio
async def test_authoritative_identity_corrections_refresh_style_and_divider_together() -> None:
    timeline = Timeline()
    presenter = presenter_for(timeline)
    corrected, other = uuid4(), uuid4()
    try:
        await presenter.emit(caption(1, corrected, "first", assignment=overflow_assignment("five")))
        await presenter.emit(caption(2, other, "second", assignment=overflow_assignment("six")))
        assert presenter.snapshot().speaker_divider is True
        current = overflow_assignment("seven", revision="corrected-final")
        current = replace(current, source_text_range=(2, 12))
        await presenter.emit(caption(3, corrected, "corrected", revision=2, assignment=current))
        entry = presenter._entries[("peer", corrected)]
        assert entry.speaker_assignment is current
        assert entry.speaker_assignment.source_text_range == (2, 12)
        assert entry.speaker_assignment.source_text_revision == "corrected-final"
        assert entry.speaker_style == "gray"
        assert presenter.snapshot().speaker_divider is True
        same_speaker = overflow_assignment("six", revision="same-speaker-correction")
        await presenter.emit(
            caption(4, corrected, "same speaker", revision=3, assignment=same_speaker)
        )
        assert presenter.snapshot().speaker_divider is False
        allocated = replace(overflow_assignment("two"), palette_index=1, palette_overflow=False)
        await presenter.emit(
            caption(5, corrected, "allocated speaker", revision=4, assignment=allocated)
        )
        assert entry.speaker_assignment is allocated
        assert entry.speaker_style == "cyan"
        assert entry.speaker_gray_reason is None
        assert presenter.snapshot().speaker_divider is False
        missing = replace(
            allocated,
            attribution=SpeakerAttribution(
                "missing", source="soniox", speaker_scope_id="same-scope"
            ),
            palette_index=None,
        )
        await presenter.emit(
            caption(6, corrected, "unknown speaker", revision=5, assignment=missing)
        )
        assert entry.speaker_assignment is missing
        assert entry.speaker_style == "gray"
        assert presenter.snapshot().speaker_divider is False
    finally:
        await presenter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("show_original", [False, True])
@pytest.mark.parametrize("with_preview", [False, True])
async def test_source_only_settlement_replaces_preview_with_authoritative_paged_source(
    show_original: bool,
    with_preview: bool,
) -> None:
    timeline = Timeline()
    presenter = presenter_for(timeline)
    presenter.show_peer_original = show_original
    unit = uuid4()
    source = "Authoritative source with all final words. " * 6
    try:
        if with_preview:
            await presenter.emit(caption(1, unit, "obsolete preview", final=False))
        settled = replace(
            caption(2, unit, "", revision=2, source=source),
            update_id="source-only-settlement",
        )
        receipt = await presenter.emit_peer_when_admissible(settled)
        assert receipt.outcome == "applied"
        assert timeline.clock.now() == 0
        entry = presenter._entries[("peer", unit)]
        assert entry.translation_text == ""
        assert entry.peer_source_only is True
        assert entry.peer_caption_final is True
        assert entry.peer_caption_revision == 2
        first = presenter.snapshot().blocks[0]
        assert first.primary_text == peer_translation_pages(source)[0]
        assert first.primary_language == "en"
        assert first.secondary_text == ""
        assert first.secondary_enabled is False
        assert first.update_id == "source-only-settlement"
        await timeline.advance(3.0)
        assert presenter.snapshot().blocks[0].primary_text == peer_translation_pages(source)[1]
        await presenter.update_calibration(OverlayCalibration(text_scale=1.2))
        assert (
            presenter.snapshot().blocks[0].primary_text
            == peer_translation_pages(source, text_scale=1.2)[0]
        )
        late_preview = await presenter.emit(
            caption(3, unit, "late preview", revision=3, final=False)
        )
        obsolete = await presenter.emit(caption(4, unit, "obsolete final", revision=1))
        assert late_preview.outcome == obsolete.outcome == "stale"
        await presenter.emit(caption(5, unit, "successful later repair", revision=4, source=source))
        assert entry.peer_source_only is False
        assert presenter.snapshot().blocks[0].primary_text == "successful later repair"
    finally:
        await presenter.close()


@pytest.mark.asyncio
async def test_source_only_repair_absorbs_all_previews_and_rejects_obsolete_retirement() -> None:
    timeline, surface = Timeline(), Surface()
    presenter = presenter_for(timeline, surface)
    presenter.show_peer_original = False
    unit, preview, absent_preview, unrelated = uuid4(), uuid4(), uuid4(), uuid4()
    try:
        await presenter.emit(caption(1, unit, "first final"))
        await presenter.emit(caption(2, preview, "preview to absorb", final=False))
        await presenter.emit_peer_when_admissible(
            caption(
                3,
                unit,
                "",
                revision=3,
                source="first source and corrected next source",
                superseded=(preview, absent_preview),
            )
        )
        assert len(surface.snapshots[-1].blocks) == 1
        assert surface.snapshots[-1].blocks[0].id == f"peer:{unit}"
        assert "corrected next source" in surface.snapshots[-1].blocks[0].primary_text
        assert ("peer", preview) not in presenter._entries
        for index, absorbed in enumerate((preview, absent_preview), start=4):
            late = await presenter.emit(
                caption(index, absorbed, "late absorbed final", revision=10)
            )
            assert late.outcome == "stale"
        await presenter.emit(caption(6, unrelated, "unrelated"))
        stale = await presenter.emit(
            caption(7, unit, "", revision=2, source="", superseded=(unrelated,))
        )
        assert stale.outcome == "stale"
        assert [block.id for block in presenter.snapshot().blocks] == [
            f"peer:{unit}",
            f"peer:{unrelated}",
        ]
    finally:
        await presenter.close()


@pytest.mark.asyncio
async def test_empty_authoritative_retirement_cancels_pages_without_waiting_or_resurrection() -> (
    None
):
    timeline = Timeline()
    presenter = presenter_for(timeline)
    preview, unrelated, absent = uuid4(), uuid4(), uuid4()
    try:
        await presenter.emit(caption(1, preview, "가" * 120, final=False))
        await presenter.emit(caption(2, unrelated, "unrelated final"))
        retired_absent = await presenter.emit_peer_when_admissible(
            caption(3, absent, "", revision=2, source="")
        )
        assert retired_absent.outcome == "applied"
        assert timeline.clock.now() == 0
        assert len(presenter.snapshot().blocks) == 2
        retired = await presenter.emit_peer_when_admissible(
            caption(4, preview, "", revision=2, source="", superseded=(absent,))
        )
        assert retired.outcome == "applied"
        assert ("peer", preview) not in presenter._entries
        assert ("peer", preview) not in presenter._expiration_tasks
        assert [block.id for block in presenter.snapshot().blocks] == [f"peer:{unrelated}"]
        await timeline.advance(3.1)
        for index, unit in enumerate((preview, absent), start=5):
            late = await presenter.emit(caption(index, unit, "must not return", revision=5))
            assert late.outcome == "stale"
        assert [block.id for block in presenter.snapshot().blocks] == [f"peer:{unrelated}"]
    finally:
        await presenter.close()


@pytest.mark.asyncio
async def test_legacy_local_empty_translation_keeps_existing_source_presentation() -> None:
    timeline = Timeline()
    presenter = presenter_for(timeline)
    unit = uuid4()
    try:
        await presenter.emit(caption(1, unit, "", revision=0, source="legacy local source"))
        entry = presenter._entries[("peer", unit)]
        assert entry.peer_source_only is False
        block = presenter.snapshot().blocks[0]
        assert block.primary_text == ""
        assert block.secondary_text == "legacy local source"
        await presenter.update_display_preferences(show_translation=True, show_peer_original=False)
        assert presenter.snapshot().blocks == []
        await presenter.emit(caption(2, unit, "", revision=0, source=""))
        assert ("peer", unit) in presenter._entries
    finally:
        await presenter.close()


@pytest.mark.asyncio
async def test_superseding_final_uses_remaining_original_caption_admission_budget() -> None:
    timeline = Timeline()
    presenter = presenter_for(timeline)
    upper, lower, unit = uuid4(), uuid4(), uuid4()
    preview_task = final_task = None
    try:
        assert isinstance(presenter, PeerCaptionAdmissionPort)
        await presenter.emit(caption(1, upper, "가" * 40))
        await presenter.emit(caption(2, lower, "lower"))
        admission_started_at = timeline.clock.now()
        preview_task = asyncio.create_task(
            presenter.emit_peer_when_admissible(caption(3, unit, "waiting preview", final=False))
        )
        await timeline.advance(0)
        assert not preview_task.done()
        await timeline.advance(0.6)
        preview_task.cancel()
        await asyncio.gather(preview_task, return_exceptions=True)
        final_task = asyncio.create_task(
            presenter.emit_peer_when_admissible(
                caption(4, unit, "authoritative final", revision=2),
                admission_started_at=admission_started_at,
            )
        )
        await timeline.advance(0)
        assert not final_task.done()
        await timeline.advance(0.39)
        assert not final_task.done()
        await timeline.advance(0.01)
        assert final_task.done()
        assert (await final_task).outcome == "applied"
        assert timeline.clock.now() == pytest.approx(1.0)
        assert [block.id for block in presenter.snapshot().blocks] == [
            f"peer:{lower}",
            f"peer:{unit}",
        ]
        assert presenter.snapshot().blocks[-1].primary_text == "authoritative final"
        stale = await presenter.emit(caption(5, unit, "waiting preview", final=False))
        assert stale.outcome == "stale"
    finally:
        for task in (preview_task, final_task):
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        await presenter.close()
