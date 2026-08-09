from __future__ import annotations

import importlib

import pytest


def _canonical_overlay_calibration_module():
    return importlib.import_module("puripuly_heart.config.overlay_calibration")


def _ui_overlay_calibration_facade_module():
    return importlib.import_module("puripuly_heart.ui.overlay_calibration")


def test_config_overlay_calibration_exports_canonical_constants_and_defaults() -> None:
    module = _canonical_overlay_calibration_module()

    assert module.OVERLAY_CALIBRATION_ANCHOR_HEAD_LOCKED == "head_locked"
    assert module.OVERLAY_CALIBRATION_ANCHOR_SPATIAL_LOCKED == "spatial_locked"
    assert module.OVERLAY_CALIBRATION_ANCHORS == ("head_locked", "spatial_locked")

    calibration = module.OverlayCalibration()

    assert calibration.anchor == module.OVERLAY_CALIBRATION_ANCHOR_HEAD_LOCKED
    assert calibration.offset_x == pytest.approx(0.0)
    assert calibration.offset_y == pytest.approx(-0.45)
    assert calibration.distance == pytest.approx(1.1)
    assert calibration.text_scale == pytest.approx(1.0)
    assert calibration.background_alpha == pytest.approx(0.24)


def test_config_overlay_calibration_preserves_serialization_copy_and_validation() -> None:
    module = _canonical_overlay_calibration_module()
    calibration = module.OverlayCalibration(
        offset_x=0.25,
        offset_y=-0.2,
        distance=1.4,
        text_scale=1.3,
        background_alpha=0.5,
    )

    assert calibration.to_dict() == {
        "anchor": "head_locked",
        "offset_x": 0.25,
        "offset_y": -0.2,
        "distance": 1.4,
        "text_scale": 1.3,
        "background_alpha": 0.5,
    }
    clone = calibration.copy()
    assert clone == calibration
    assert clone is not calibration
    calibration.validate()
    module.OverlayCalibration(anchor="spatial_locked").validate()

    invalid_cases = (
        ({"anchor": "unsupported"}, "unsupported overlay calibration anchor"),
        ({"distance": 0.0}, "distance must be > 0"),
        ({"text_scale": 0.0}, "text_scale must be > 0"),
        ({"background_alpha": -0.01}, "background_alpha"),
        ({"background_alpha": 1.01}, "background_alpha"),
    )
    for kwargs, match in invalid_cases:
        with pytest.raises(ValueError, match=match):
            module.OverlayCalibration(**kwargs).validate()


def test_ui_overlay_calibration_facade_reexports_canonical_objects() -> None:
    canonical = _canonical_overlay_calibration_module()
    facade = _ui_overlay_calibration_facade_module()

    assert facade.OverlayCalibration is canonical.OverlayCalibration
    assert (
        facade.OVERLAY_CALIBRATION_ANCHOR_HEAD_LOCKED
        == canonical.OVERLAY_CALIBRATION_ANCHOR_HEAD_LOCKED
    )
    assert (
        facade.OVERLAY_CALIBRATION_ANCHOR_SPATIAL_LOCKED
        == canonical.OVERLAY_CALIBRATION_ANCHOR_SPATIAL_LOCKED
    )
    assert facade.OVERLAY_CALIBRATION_ANCHORS is canonical.OVERLAY_CALIBRATION_ANCHORS
