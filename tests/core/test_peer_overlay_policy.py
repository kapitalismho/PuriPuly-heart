from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from uuid import UUID, uuid4

import pytest

from puripuly_heart.config.overlay_calibration import OverlayCalibration
from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.overlay.peer_caption_layout import peer_translation_pages
from puripuly_heart.core.overlay.presenter import OverlayPresenter
from puripuly_heart.core.overlay.protocol import OverlayPresentationSnapshot
from puripuly_heart.core.overlay.sink import (
    SelfActiveUpdate,
    TranslationFinal,
    TranslationStreamUpdate,
)


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
