from __future__ import annotations

import asyncio
import json
from uuid import UUID

from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.overlay.presenter import OverlayPresenter
from puripuly_heart.core.overlay.sink import OverlayEventAdapter
from puripuly_heart.domain.models import Transcript
from puripuly_heart.ui.overlay_calibration import OverlayCalibration


async def projected_self_lifecycle() -> list[dict[str, object]]:
    clock = FakeClock(_now=10.0)
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(anchor="spatial_locked"),
        clock=clock,
        native_retry_enabled=True,
    )
    adapter = OverlayEventAdapter(clock=clock)
    turn = UUID("00000000-0000-0000-0000-000000000206")
    snapshots = []
    try:
        for text in ("source", "source extension", "source extension complete", "source churn"):
            await presenter.emit(
                adapter.self_active_update(
                    text=text, utterance_id=turn, occupant_key=f"self:{turn}"
                )
            )
            snapshots.append(presenter.snapshot().to_dict())
        await presenter.emit(
            adapter.transcript_final(
                Transcript(
                    utterance_id=turn,
                    channel="self",
                    text="source churn",
                    is_final=True,
                    created_at=clock.now(),
                ),
                source_language="en",
                target_language="ko",
            )
        )
        snapshots.append(presenter.snapshot().to_dict())
        await presenter.emit(
            adapter.translation_final(
                utterance_id=turn,
                channel="self",
                text="synthetic translation",
                source_language="en",
                target_language="ko",
                applied_context_mode=None,
            )
        )
        snapshots.append(presenter.snapshot().to_dict())
        await presenter.clear_for_runtime_detach()
        snapshots.append(presenter.snapshot().to_dict())
        return snapshots
    finally:
        await presenter.close()


if __name__ == "__main__":
    print(json.dumps(asyncio.run(projected_self_lifecycle())))
