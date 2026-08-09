from __future__ import annotations

from dataclasses import asdict, dataclass, replace

OVERLAY_CALIBRATION_ANCHOR_HEAD_LOCKED = "head_locked"
OVERLAY_CALIBRATION_ANCHOR_SPATIAL_LOCKED = "spatial_locked"
OVERLAY_CALIBRATION_ANCHORS = (
    OVERLAY_CALIBRATION_ANCHOR_HEAD_LOCKED,
    OVERLAY_CALIBRATION_ANCHOR_SPATIAL_LOCKED,
)


@dataclass(slots=True)
class OverlayCalibration:
    anchor: str = OVERLAY_CALIBRATION_ANCHOR_HEAD_LOCKED
    offset_x: float = 0.0
    offset_y: float = -0.45
    distance: float = 1.1
    text_scale: float = 1.0
    background_alpha: float = 0.24

    def validate(self) -> None:
        if self.anchor not in OVERLAY_CALIBRATION_ANCHORS:
            raise ValueError(f"unsupported overlay calibration anchor: {self.anchor!r}")
        if self.distance <= 0.0:
            raise ValueError("overlay calibration distance must be > 0")
        if self.text_scale <= 0.0:
            raise ValueError("overlay calibration text_scale must be > 0")
        if not (0.0 <= self.background_alpha <= 1.0):
            raise ValueError("overlay calibration background_alpha must be in 0.0..1.0")

    def copy(self) -> "OverlayCalibration":
        return replace(self)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
