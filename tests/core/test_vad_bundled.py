from __future__ import annotations

import hashlib

import pytest

from puripuly_heart.core.vad import bundled as bundled_vad
from puripuly_heart.core.vad.bundled import (
    SILERO_VAD_RESOURCE_SHA256,
    bundled_silero_vad_onnx_path,
)


def test_bundled_silero_vad_sha256_matches_constant():
    bundled = bundled_silero_vad_onnx_path()

    with bundled.open("rb") as fh:
        digest = hashlib.file_digest(fh, "sha256").hexdigest()

    assert digest == SILERO_VAD_RESOURCE_SHA256


def test_missing_silero_bundle_fails_without_user_cache_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    cache = tmp_path / "puripuly-heart" / "silero_vad.onnx"
    cache.parent.mkdir()
    cache.write_bytes(b"obsolete-cache")
    monkeypatch.setattr(
        bundled_vad, "SILERO_VAD_RESOURCE_RELATIVE_PATH", str(tmp_path / "absent-bundle.onnx")
    )
    with pytest.raises(FileNotFoundError):
        bundled_silero_vad_onnx_path()
    assert cache.read_bytes() == b"obsolete-cache"
