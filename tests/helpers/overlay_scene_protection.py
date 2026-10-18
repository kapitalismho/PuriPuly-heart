from __future__ import annotations

import asyncio
import json
from uuid import UUID

from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.overlay.bridge_mailbox import OverlayBridgeMailbox
from puripuly_heart.core.overlay.presenter import OverlayPresenter
from puripuly_heart.core.overlay.protocol import OverlayPresentationSnapshot
from puripuly_heart.core.overlay.sink import OverlayEventAdapter
from puripuly_heart.domain.models import Transcript
from puripuly_heart.ui.overlay_calibration import OverlayCalibration


async def projected_scenes(*, peer_mode: str = "secondary_only") -> dict[str, object]:
    clock = FakeClock(_now=10.0)
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(anchor="spatial_locked"),
        clock=clock,
        native_retry_enabled=True,
    )
    adapter = OverlayEventAdapter(clock=clock)
    a, b, peer = [UUID(int=value) for value in (2651, 2652, 2653)]
    scenes = []
    outbox = OverlayBridgeMailbox(
        initial_snapshot=OverlayPresentationSnapshot(),
        clock=clock,
        scene_byte_limit=65536,
        control_byte_limit=4096,
        control_slot_limit=8,
        delivery_receipt_limit=16,
    )

    def capture() -> None:
        current = presenter.snapshot()
        scenes.append(current.to_dict())
        outbox.admit_scene(current, {block.id: None for block in current.blocks})

    async def caption(turn: UUID, text: str, translation: str, revision: int, **fields) -> None:
        await presenter.emit(
            adapter.self_active_update(
                text=text,
                secondary_text=translation,
                utterance_id=turn,
                occupant_key=f"self:{turn}",
                caption_revision=revision,
                target_language="ko",
                **fields,
            )
        )
        capture()

    try:
        await caption(a, "synthetic A", "", 1, caption_source_closed=True)
        await caption(
            a,
            "synthetic A",
            "synthetic translation A",
            2,
            caption_source_closed=True,
            caption_complete=True,
        )
        clock.advance(1)
        await caption(b, "synthetic B", "", 1, caption_source_closed=False)
        await caption(
            a,
            "synthetic A",
            "synthetic retained patch",
            3,
            caption_source_closed=True,
            caption_complete=True,
            retained_only=True,
        )
        coalesced = outbox.take_scene_for_write().snapshot.to_dict()
        clock.advance(6)
        await caption(b, "synthetic B extension", "", 2, caption_source_closed=False)
        clock.advance(3)
        await presenter._publish_if_changed()
        capture()
        await presenter.emit(
            adapter.transcript_final(
                Transcript(
                    utterance_id=peer,
                    channel="peer",
                    text="synthetic peer source",
                    is_final=True,
                    created_at=clock.now(),
                ),
                source_language="en",
                target_language="ko",
            )
        )
        if peer_mode == "translated_primary":
            await presenter.emit(
                adapter.translation_final(
                    utterance_id=peer,
                    channel="peer",
                    text="synthetic peer translation",
                    source_language="en",
                    target_language="ko",
                    applied_context_mode=None,
                )
            )
        capture()
        mixed = presenter.snapshot()
        held = outbox.make_scene(mixed, {f"self:{b}": clock.now() + 0.1})
        clock.advance(0.8)
        pruned = json.loads(outbox.revalidated_scene_message(held))["payload"]
        repeated_pruned = json.loads(outbox.revalidated_scene_message(held))["payload"]
        await caption(
            b,
            "synthetic B extension",
            "",
            3,
            caption_source_closed=True,
            caption_cancelled=True,
            caption_complete=True,
        )
        await presenter.clear_for_runtime_detach()
        capture()
        after_clear_turn = UUID(int=2654)
        await presenter.emit(
            adapter.self_active_update(
                text="synthetic after clear",
                utterance_id=after_clear_turn,
                occupant_key=f"self:{after_clear_turn}",
                caption_source_closed=False,
                caption_revision=1,
            )
        )
        clear_rearm = presenter.snapshot().to_dict()
        await presenter.begin_native_retry_epoch(enabled=True)
        fresh_epoch = presenter.snapshot().to_dict()
        await presenter.clear_for_runtime_detach()
        empty_after_rearm = presenter.snapshot().to_dict()
        return {
            "scenes": scenes,
            "coalesced": coalesced,
            "pruned": pruned,
            "repeated_pruned": repeated_pruned,
            "clear_rearm": clear_rearm,
            "fresh_epoch": fresh_epoch,
            "empty_after_rearm": empty_after_rearm,
            "mailbox_retained_bytes": outbox.retained_scene_bytes,
            "mailbox_pending_scenes": int(outbox.pending_scene is not None),
            "mailbox_receipts": len(outbox.delivery_receipts),
        }
    finally:
        await presenter.close()


async def recovery_scenes(*, peer_mode: str = "secondary_only") -> dict[str, object]:
    cases = {}
    for case in ("covered", "ordinary", "hidden", "empty", "expired", "disabled", "cleared"):
        clock = FakeClock(_now=10.0)
        presenter = OverlayPresenter(
            calibration=OverlayCalibration(anchor="spatial_locked"),
            clock=clock,
            native_retry_enabled=True,
            show_peer_original=case != "hidden",
        )
        adapter = OverlayEventAdapter(clock=clock)
        self_turn, peer = UUID(int=2661), UUID(int=2662)
        try:
            if case != "ordinary":
                await presenter.emit(
                    adapter.self_active_update(
                        text="synthetic recovery self",
                        utterance_id=self_turn,
                        occupant_key=f"self:{self_turn}",
                        caption_revision=1,
                    )
                )
            await presenter.emit(
                adapter.transcript_final(
                    Transcript(
                        utterance_id=peer,
                        channel="peer",
                        text="" if case == "empty" else "synthetic recovery peer",
                        is_final=True,
                        created_at=clock.now(),
                    ),
                    source_language="en",
                    target_language="ko",
                )
            )
            if peer_mode == "translated_primary" and case not in {"hidden", "empty"}:
                await presenter.emit(
                    adapter.translation_final(
                        utterance_id=peer,
                        channel="peer",
                        text="synthetic recovery translation",
                        source_language="en",
                        target_language="ko",
                        applied_context_mode=None,
                    )
                )
            if case != "ordinary":
                await presenter.emit(
                    adapter.self_active_update(
                        text="synthetic recovery self",
                        utterance_id=self_turn,
                        occupant_key=f"self:{self_turn}",
                        caption_revision=2,
                        caption_source_closed=True,
                        caption_cancelled=True,
                        caption_complete=True,
                    )
                )
            before = presenter.snapshot().to_dict()
            entry = presenter._entries.get(("peer", peer))
            deadline_before = (
                presenter._entry_expiration_deadline(entry) if entry is not None else None
            )
            clock.advance(1)
            if case == "expired" and deadline_before is not None:
                clock.advance(deadline_before - clock.now() + 0.01)
            if case == "cleared":
                await presenter.clear_for_runtime_detach()
            await presenter.begin_native_retry_epoch(enabled=case != "disabled")
            recovered = presenter.snapshot().to_dict()
            entry = presenter._entries.get(("peer", peer))
            deadline_after = (
                presenter._entry_expiration_deadline(entry) if entry is not None else None
            )
            if deadline_before is not None and clock.now() < deadline_before:
                clock.advance(deadline_before - clock.now() + 0.01)
            await presenter._publish_if_changed()
            expired = presenter.snapshot().to_dict()
            await presenter.clear_for_runtime_detach()
            await presenter.begin_native_retry_epoch(enabled=True)
            cases[case] = {
                "before": before,
                "recovered": recovered,
                "deadline_before": deadline_before,
                "deadline_after": deadline_after,
                "at_original_expiry": expired,
                "clear_fresh": presenter.snapshot().to_dict(),
            }
        finally:
            await presenter.close()
    return cases


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--peer-mode", choices=("secondary_only", "translated_primary"), default="secondary_only"
    )
    parser.add_argument("--recovery", action="store_true")
    args = parser.parse_args()
    project = recovery_scenes if args.recovery else projected_scenes
    print(json.dumps(asyncio.run(project(peer_mode=args.peer_mode))))
