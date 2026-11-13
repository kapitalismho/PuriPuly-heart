from __future__ import annotations

import asyncio

import pytest

from puripuly_heart.core import feature_diagnostics


def test_feature_recording_is_optional_but_validation_and_cancellation_are_not(monkeypatch):
    calls = []

    def unavailable(*args, **kwargs):
        calls.append((args, kwargs))
        raise RuntimeError("synthetic sink unavailable")

    monkeypatch.setattr(feature_diagnostics, "emit_basic_log", unavailable)
    feature_diagnostics.log_feature("ModelDownload", "download", "download_failed")
    assert len(calls) == 1
    with pytest.raises(ValueError):
        feature_diagnostics.log_feature("unexpected", "download", "download_failed")
    with pytest.raises(ValueError):
        feature_diagnostics.log_feature("ModelDownload", "private/path", "download_failed")
    assert len(calls) == 1

    def cancelled(*args, **kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(feature_diagnostics, "emit_basic_log", cancelled)
    with pytest.raises(asyncio.CancelledError):
        feature_diagnostics.log_feature("ModelDownload", "download", "download_failed")
