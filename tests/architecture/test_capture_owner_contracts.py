from __future__ import annotations

from puripuly_heart.core.peer_capture import PeerCaptureSessionConfig, PeerCaptureSessionSnapshot
from puripuly_heart.core.self_capture import SelfCaptureSessionConfig, SelfCaptureSessionSnapshot


def test_capture_session_configs_and_snapshots_are_immutable() -> None:
    for value_type in (
        SelfCaptureSessionConfig,
        SelfCaptureSessionSnapshot,
        PeerCaptureSessionConfig,
        PeerCaptureSessionSnapshot,
    ):
        assert value_type.__dataclass_params__.frozen, value_type.__name__
