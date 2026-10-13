from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

from puripuly_heart.core.overlay.protocol import (
    OverlayPresentationBlock,
    OverlayPresentationSnapshot,
)
from puripuly_heart.ui.desktop_overlay import FletDesktopRendererWindow
from puripuly_heart.ui.desktop_overlay_repro import StaticCheckerboardBackdrop

SYNTHETIC_WINDOW_TITLE = "PuriPuly #217 Synthetic Caption Cards"


@dataclass(frozen=True, slots=True)
class CaptionCardReproStage:
    name: str
    snapshot: OverlayPresentationSnapshot
    source_units: tuple[str, ...] = ()


def caption_card_repro_stages() -> tuple[CaptionCardReproStage, ...]:
    a = OverlayPresentationBlock(
        id="card:a",
        occupant_key="card:a",
        appearance_seq=1,
        card_revision=1,
        channel="peer",
        block_variant="finalized",
        primary_text="Again.",
        secondary_text="Encore.",
        secondary_enabled=True,
        speaker_style="gold",
    )
    b = replace(
        a,
        id="card:b",
        occupant_key="card:b",
        appearance_seq=2,
        primary_text="B stays unchanged.",
        secondary_text="B reste inchangé.",
        speaker_style="cyan",
    )
    accumulated = replace(
        a,
        appearance_seq=3,
        card_revision=2,
        primary_text="Again. Again.",
        secondary_text="Encore. Encore.",
    )
    new_epoch = replace(
        b,
        id="card:new-epoch",
        occupant_key="card:new-epoch",
        appearance_seq=4,
        primary_text="New epoch remains gray.",
        secondary_text="La nouvelle époque reste grise.",
        speaker_style="gray",
    )
    overflow_a = replace(accumulated, speaker_style="gray", card_revision=3)
    overflow_b = replace(b, speaker_style="gray")
    self_card = replace(
        a,
        id="card:self",
        occupant_key="card:self",
        channel="self",
        primary_text="SELF stays white.",
        secondary_text="SELF reste blanc.",
    )
    values = (
        ("first_a", [a], False, ("source:a1",)),
        ("a_b", [a, b], False, ("source:a1", "source:b1")),
        ("a_b_a_accumulation", [accumulated, b], False, ("source:a1", "source:b1", "source:a2")),
        ("replacement_replay", [b, accumulated], False, ("source:a1", "source:b1", "source:a2")),
        (
            "same_text_style_redraw",
            [b, replace(accumulated, speaker_style="coral", card_revision=3)],
            False,
            (),
        ),
        ("epoch_handoff_gray", [accumulated, new_epoch], False, ()),
        ("established_overflow_divider", [overflow_a, overflow_b], True, ()),
        ("unknown_gray_no_divider", [overflow_a, overflow_b], False, ()),
        ("self_white", [self_card], False, ()),
        ("python_owned_removal", [], False, ()),
    )
    return tuple(
        CaptionCardReproStage(
            name,
            OverlayPresentationSnapshot(
                revision=revision,
                blocks=blocks,
                speaker_divider=divider,
            ),
            units,
        )
        for revision, (name, blocks, divider, units) in enumerate(values, start=1)
    )


class _SyntheticCaptionWindow(FletDesktopRendererWindow):
    def _window_title(self) -> str:
        return SYNTHETIC_WINDOW_TITLE


async def run_caption_card_repro(output_dir: Path, dwell_ms: int) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise ValueError("output directory must be empty")
    stages = caption_card_repro_stages()
    (output_dir / "snapshots.json").write_text(
        json.dumps(
            [
                {
                    "name": stage.name,
                    "source_units": stage.source_units,
                    "snapshot": stage.snapshot.to_dict(),
                }
                for stage in stages
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    backdrop = StaticCheckerboardBackdrop()
    window = _SyntheticCaptionWindow(locale="en", startup_timeout_s=10.0)
    window.prime_startup_runtime_controls(
        (
            {
                "command": "apply_window_bounds",
                "x": 0,
                "y": 0,
                "width": 1344,
                "height": 336,
            },
        )
    )
    records: list[dict[str, object]] = []
    origin = time.monotonic()
    status = "failed"
    try:
        await backdrop.start()
        await window.start(OverlayPresentationSnapshot(revision=0))
        await window.dispatch_runtime_control(
            {
                "command": "set_interaction_mode",
                "mode": "pass_through",
            }
        )
        for stage in stages:
            await window.dispatch_snapshot(stage.snapshot)
            record = {
                "name": stage.name,
                "revision": stage.snapshot.revision,
                "monotonic_ms": round((time.monotonic() - origin) * 1000),
                "visual_state": window.renderer_visual_state(),
            }
            records.append(record)
            print(json.dumps(record), flush=True)
            await asyncio.sleep(dwell_ms / 1000)
        status = "dispatched"
        return 0
    finally:
        try:
            await window.close()
        finally:
            await backdrop.close()
            (output_dir / "run_report.json").write_text(
                json.dumps(
                    {
                        "status": status,
                        "window_title": SYNTHETIC_WINDOW_TITLE,
                        "synthetic_only": True,
                        "dwell_ms": dwell_ms,
                        "records": records,
                        "pixel_inspection": "not performed by this executable",
                        "repetition": "two distinct synthetic source contributions retained; no acoustic claim",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dwell-ms", type=int, default=2000)
    arguments = parser.parse_args()
    if not 100 <= arguments.dwell_ms <= 10000:
        parser.error("dwell-ms must be between 100 and 10000")
    return asyncio.run(run_caption_card_repro(arguments.output_dir, arguments.dwell_ms))


if __name__ == "__main__":
    raise SystemExit(main())
