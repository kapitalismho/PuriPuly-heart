from __future__ import annotations

from unittest.mock import AsyncMock

from puripuly_heart.app.services import photo_application
from puripuly_heart.app.services.photo_application import PhotoApplicationOwner
from puripuly_heart.core.photo.capture import PhotoCaptureWatcher


async def test_native_prepare_timings_keep_identity_and_ignore_invalid_values(
    tmp_path, monkeypatch
):
    async def no_route():
        return None

    observations = []
    owner = PhotoApplicationOwner(
        resolve_route=no_route,
        target_language=lambda: "ko",
        diagnostic=lambda stage, fields: observations.append((stage, fields)),
        watcher=PhotoCaptureWatcher(tmp_path / "photos", tmp_path / "logs"),
        temporary_root=tmp_path,
    )
    prepared = object()
    reader = AsyncMock(return_value=prepared)
    monkeypatch.setattr(photo_application, "load_prepared_photo", reader)
    owner.bridge.prepare = AsyncMock(
        return_value={
            "decode_ms": 3.25,
            "resize_ms": 7.5,
            "jpeg_ms": 1.125,
            "source_path": "must-not-enter-diagnostics",
            "source": "private text",
        }
    )
    photo_id = "fce34b2a-a0b3-48aa-952e-de4b05002061"
    try:
        assert (
            await owner._prepare(
                photo_id, 1, tmp_path / "source.png", tmp_path, expected_file_id=(1, 2)
            )
            is prepared
        )
        assert observations == [
            ("decode", {"photo_id": photo_id, "attempt_id": 1, "duration_ms": 3.25}),
            ("resize", {"photo_id": photo_id, "attempt_id": 1, "duration_ms": 7.5}),
            ("jpeg", {"photo_id": photo_id, "attempt_id": 1, "duration_ms": 1.125}),
        ]
        observations.clear()
        owner.bridge.prepare.return_value = {
            "decode_ms": float("nan"),
            "resize_ms": -1.0,
            "jpeg_ms": True,
        }
        await owner._prepare(
            photo_id, 1, tmp_path / "source.png", tmp_path, expected_file_id=(1, 2)
        )
        assert observations == []
        assert reader.await_count == 2
    finally:
        await owner.close()
