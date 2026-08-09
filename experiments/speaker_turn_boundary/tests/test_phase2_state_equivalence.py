from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from uuid import uuid4

import numpy as np

from experiments.speaker_turn_boundary.turn_episode import state_equivalence
from experiments.speaker_turn_boundary.turn_episode.build_episodes import (
    CHUNK_SAMPLES,
    WindowBounds,
)


class FakeRing:
    def __init__(self, value: float) -> None:
        self._buffer = np.full(CHUNK_SAMPLES, value, dtype=np.float32)
        self._filled = True
        self._write_pos = 0


class FakeReplay:
    def __init__(self) -> None:
        pending_id = uuid4()
        self._gating = SimpleNamespace(
            _ring=FakeRing(1.0),
            _pending_start_id=pending_id,
            _pending_payload={"phase": "captured"},
        )
        self.boundaries = [
            {
                "boundary_source_sample": 0,
                "emitted_monotonic_ns": id(self),
                "utterance_id": str(pending_id),
            }
        ]
        self.progress = [{"observed_source_sample": CHUNK_SAMPLES}]

    def start_epoch(self, audio_epoch: int) -> None:
        self.audio_epoch = audio_epoch

    def process_chunk(self, chunk: np.ndarray) -> None:
        self._gating._ring = FakeRing(2.0)
        self._gating._ring._write_pos = int(chunk.size)
        self._gating._pending_start_id = uuid4()
        self._gating._pending_payload = {"phase": "resumed"}
        self.boundaries.append({"boundary_source_sample": CHUNK_SAMPLES})
        self.progress.append({"observed_source_sample": CHUNK_SAMPLES * 2})


def test_snapshot_round_trip_evidence_is_repeatable_and_time_consistent(monkeypatch) -> None:
    def capture_state(replay: FakeReplay) -> dict[str, object]:
        return {
            "gating": {
                "_ring": FakeRing(float(replay._gating._ring._buffer[0])),
                "_pending_start_id": replay._gating._pending_start_id,
                "_pending_payload": deepcopy(replay._gating._pending_payload),
            },
            "engine_state": {},
            "replay": {
                "_boundaries": deepcopy(replay.boundaries),
                "_progress": deepcopy(replay.progress),
                "_identity": replay._gating._pending_start_id,
            },
        }

    def restore_state(replay: FakeReplay, capture: dict[str, object]) -> None:
        gating = capture["gating"]
        replay_state = capture["replay"]
        replay._gating._ring = FakeRing(float(gating["_ring"]._buffer[0]))
        replay._gating._pending_start_id = gating["_pending_start_id"]
        replay._gating._pending_payload = deepcopy(gating["_pending_payload"])
        replay.boundaries = deepcopy(replay_state["_boundaries"])
        replay.progress = deepcopy(replay_state["_progress"])

    monkeypatch.setattr(
        state_equivalence,
        "_load_wav",
        lambda path: np.zeros(CHUNK_SAMPLES * 2, dtype=np.float32),
    )
    monkeypatch.setattr(
        state_equivalence,
        "_replay_region",
        lambda samples, start_sample, end_sample: FakeReplay(),
    )
    monkeypatch.setattr(
        state_equivalence,
        "_make_replay",
        lambda engine_factory: FakeReplay(),
    )
    monkeypatch.setattr(state_equivalence, "capture_state", capture_state)
    monkeypatch.setattr(state_equivalence, "restore_state", restore_state)
    monkeypatch.setattr(
        state_equivalence,
        "_pending_start_id",
        lambda replay: str(replay._gating._pending_start_id),
    )
    monkeypatch.setattr(
        state_equivalence,
        "_pending_content",
        lambda replay: deepcopy(replay._gating._pending_payload),
    )
    monkeypatch.setattr(state_equivalence, "_boundary_rows", lambda replay: replay.boundaries)
    monkeypatch.setattr(state_equivalence, "_progress_rows", lambda replay: replay.progress)

    bounds = WindowBounds(
        warm_start=0,
        scored_start=CHUNK_SAMPLES,
        scored_end=CHUNK_SAMPLES * 2,
        tail_end=CHUNK_SAMPLES * 2,
        unaligned_source_end=False,
    )
    first = state_equivalence.snapshot_round_trip("unused.wav", bounds, CHUNK_SAMPLES * 2)
    second = state_equivalence.snapshot_round_trip("unused.wav", bounds, CHUNK_SAMPLES * 2)

    assert first["capture_sha256"] == second["capture_sha256"]
    assert first["capture_hash_contract"] == "runtime_identity_normalized_v1"
    assert first["ring_payload_before_resume"] == first["ring_payload_captured"]
    assert first["pending_start_before_resume"] == first["pending_start_captured"]
    assert first["pending_content_before_resume"] == first["pending_content_captured"]
    assert first["ring_payload_restored"] != first["ring_payload_before_resume"]
    assert first["pending_start_restored"] != first["pending_start_before_resume"]
    assert first["ring_fidelity"]
    assert first["pending_fidelity"]
    assert first["ring_parity"]
    assert first["pending_parity"]
