from __future__ import annotations

import pytest

from puripuly_heart.core.vrchat_scene import VrchatSceneSnapshot


def test_default_snapshot_is_unavailable_without_count() -> None:
    snapshot = VrchatSceneSnapshot()

    assert snapshot.status == "unavailable"
    assert snapshot.participant_count is None


def test_ready_snapshot_requires_positive_integer_count() -> None:
    assert VrchatSceneSnapshot(status="ready", participant_count=1).participant_count == 1
    assert VrchatSceneSnapshot(status="ready", participant_count=15).participant_count == 15


@pytest.mark.parametrize("count", [0, -1, -40])
def test_ready_snapshot_rejects_non_positive_count(count: int) -> None:
    with pytest.raises(ValueError):
        VrchatSceneSnapshot(status="ready", participant_count=count)


@pytest.mark.parametrize("count", [True, False, "3", 3.0, (3,)])
def test_ready_snapshot_rejects_non_integer_count(count: object) -> None:
    with pytest.raises(TypeError):
        VrchatSceneSnapshot(status="ready", participant_count=count)  # type: ignore[arg-type]


def test_ready_snapshot_rejects_missing_count() -> None:
    with pytest.raises(TypeError):
        VrchatSceneSnapshot(status="ready")  # type: ignore[call-arg]


@pytest.mark.parametrize("status", ["unavailable", "syncing", "degraded"])
def test_non_ready_snapshot_rejects_any_count(status: object) -> None:
    with pytest.raises(ValueError):
        VrchatSceneSnapshot(status=status, participant_count=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("status", ["unavailable", "syncing", "degraded"])
def test_non_ready_snapshot_defaults_to_no_count(status: object) -> None:
    snapshot = VrchatSceneSnapshot(status=status)  # type: ignore[arg-type]

    assert snapshot.status == status
    assert snapshot.participant_count is None


def test_snapshot_is_immutable() -> None:
    snapshot = VrchatSceneSnapshot(status="ready", participant_count=2)

    with pytest.raises(Exception):
        snapshot.participant_count = 5  # type: ignore[misc]
