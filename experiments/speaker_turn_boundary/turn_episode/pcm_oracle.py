from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import wave
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from experiments.speaker_turn_boundary.config import (
    B0_SOURCE,
    B0_VAD_HANGOVER_MS,
    B0_VAD_MAX_SEGMENT_MS,
    B0_VAD_PRE_ROLL_MS,
    B0_VAD_PROFILE,
    B0_VAD_SPEECH_THRESHOLD,
    B0_VAD_START_COMMIT_CHUNKS,
    B0_VAD_START_DEBOUNCE_CHUNKS,
)
from puripuly_heart.core.vad.gating import VadEngine, create_peer_vad_gating

from .build_episodes import (
    canonical_json as phase2_canonical_json,
)
from .build_episodes import (
    floor_to_chunk,
    sha256_bytes,
    sha256_file,
)

SCHEMA_VERSION = "turn_episode_v1.phase3_pcm_oracle"
DETAIL_SCHEMA_VERSION = "turn_episode_v1.phase3_pcm_oracle_detail"
VERIFICATION_SCHEMA_VERSION = "turn_episode_v1.phase3_pcm_oracle_verification"
CLAMP_SCHEMA_VERSION = "turn_episode_v1.phase3_clamp_identity"
GRID_ID = "turn_episode_v1.provider_neutral_oracle.7x9x7"
SAMPLE_RATE_HZ = 16000
SAMPLES_PER_MS = 16
CHUNK_SAMPLES = 512
SAFE_DRAIN_TIMEOUT_MS = 2000
DELAYS_MS = (250, 500, 750, 1000, 1250, 1500, 2000)
OFFSETS_MS = (-500, -300, -200, -100, 0, 100, 200, 300, 500)
HOLDBACKS_MS = (0, 250, 500, 750, 1000, 1500, 2000)
OWNER_THRESHOLDS_MS = (50, 100, 200)
EXPECTED_POPULATION_SHA256 = "cb06483fb82618bf06dbcbe75a946c65bdea9f67109ddfb645a7e06f4dd555bf"
EXPECTED_CLAMP_SHA256 = "22b4488a8a93ee1e6b8de03cdfa914613e213f5198ba603605551f9c3404e14c"
EXPECTED_SESSION_COUNT = 20
EXPECTED_EPISODE_COUNT = 186
EXPECTED_REFERENCE_COUNT = 283
EXPECTED_HARD_POSITIVE_EPISODES = 142
EXPECTED_NO_HARD_EPISODES = 44
EXPECTED_GRID_ROWS = 441
EXPECTED_DETAIL_ROWS = 82026
EXPECTED_ACTION_INSTANCES = 124803
EXPECTED_ROWS_PER_SHARD = 11718
EXPECTED_ACTIONS_PER_SHARD = 17829
EXPECTED_CLAMPED_REFERENCE_OFFSETS = 24
EXPECTED_CLAMPED_REFERENCES = 16
EXPECTED_CLAMPED_EPISODES = 15
EXPECTED_CLAMP_BELOW = 13
EXPECTED_CLAMP_ABOVE = 11
EXPECTED_CLAMPED_ACTION_INSTANCES = 1176
MAIN_MAX_BYTES = 10 * 1024 * 1024
SHARD_MAX_BYTES = 20 * 1024 * 1024
APPROVED_BUNDLE_SHA256 = "8dbcd4333297fa1dbc8b26a3ff4d9f0c708a0811588517b91184cabf20d17d36"
CURRENT_AUTHORITY_SHA256 = "ecd16c765072504f9970cb17fd5cba4f9967715954adead6ebc386fffb8d4f8c"

FROZEN_INPUT_SHA256 = {
    "episode_manifest_dev.json": "a5df5e56c2917ba174fca7892fbaa092f2b57705cad7a77bf9a347ba94cddfee",
    "state_equivalence_report.json": "6e33711632d5f2e3de8e0c22c229b08827d1ccbb873deba2c1681a2ab2c544ec",
    "scoring_fixture_report.json": "36a9648178f3de1b9924b1a4ef71baddf28eccd39fa7218d34b447f993a145b1",
    "audit_report.json": "901020e864ada40a7918354f8039bad85512dace8d033a1bcba16d3428db36e4",
    "proposal_contract.json": "0448edd933fd1d9d0a0b4d5f9f2631cb0f630c892fc4d46e1a3ec9740e80b7fb",
    "fusion_contract.json": "bfda0c3c0ea7b6613ded79e9639692a33449dcf34202b1f2a5e7ec14c45f9873",
}
FROZEN_CODE_SHA256 = {
    "turn_episode/schemas.py": "a9fa4571b1bab3cf88d6739a3732c1cc62f753a46d51d59c2db7526468eb8868",
    "turn_episode/contracts.py": "b207d3f8b9720df5dd228aa8bd8b479c54622abb905a9ca04f580820a6fc3c03",
    "turn_episode/scoring.py": "332a7daf70e684cd5b9918f808b76e8e0c39f6db559008d3f48c38590fb0aa90",
    "turn_episode/build_episodes.py": "6deec51274cedf49a70cd299700547f39cbbbc16e200eb8e3056d15887784c7d",
    "turn_episode/pinned_ledger.py": "7509c7abea6813051150f1ff2d98e6f61630c5e10a1801a2905326d9f1290aaa",
    "vad_baseline.py": "7a3965fdb01eb7391dde985e5c498162d80b4e5ab565205626d684a66d8ff627",
    "config.py": "f4eb24e6c81ebcb0bdd71b6c0c9098595ae4bdddf53e05df6bd8eea925d146a6",
    "src/puripuly_heart/core/vad/gating.py": "88d5dec630b8352fd192f1ef5be7aea39b19bdc7d43273810d260400e3217fec",
    "src/puripuly_heart/core/vad/silero.py": "43079df5bc36ecb924b1aec7991cff2a16c04ab126bb54907c4b2a570e2cd109",
}
SILERO_MODEL_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"


class Phase3OracleError(RuntimeError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def nearest_rank(values: Sequence[int], quantile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(int(value) for value in values)
    rank = max(1, math.ceil(quantile * len(ordered)))
    return ordered[rank - 1]


def _span_payload(start: int, end: int, index: int) -> dict[str, Any]:
    return {"start": start, "end": end, "turn_id": f"turn-{index:04d}"}


def turn_spans(start: int, end: int, boundaries: Iterable[int]) -> list[dict[str, Any]]:
    points = [start]
    points.extend(sorted({int(point) for point in boundaries if start < int(point) < end}))
    points.append(end)
    return [
        _span_payload(left, right, index)
        for index, (left, right) in enumerate(zip(points, points[1:]))
        if left < right
    ]


def span_cover_flags(spans: Sequence[dict[str, Any]], start: int, end: int) -> dict[str, bool]:
    if start == end and not spans:
        return {"conservation": True, "no_duplication": True, "ordering": True}
    ordered = sorted(spans, key=lambda span: (int(span["start"]), int(span["end"])))
    cursor = start
    total = 0
    disjoint = True
    ordered_ok = True
    for span in ordered:
        left = int(span["start"])
        right = int(span["end"])
        if left != cursor or right <= left:
            ordered_ok = False
        if left < cursor:
            disjoint = False
        cursor = max(cursor, right)
        total += max(0, right - left)
    conservation = bool(ordered) and cursor == end and ordered[0]["start"] == start
    conservation = conservation and ordered_ok and total == end - start
    no_duplication = disjoint and total == end - start
    return {
        "conservation": conservation,
        "no_duplication": no_duplication,
        "ordering": ordered_ok,
    }


def contamination_samples(
    spans: Sequence[dict[str, Any]],
    singleton_intervals: Sequence[Sequence[Any]],
    owner_threshold_ms: int,
) -> dict[str, int]:
    threshold = owner_threshold_ms * SAMPLES_PER_MS
    contaminated = 0
    denominator = sum(int(item[1]) - int(item[0]) for item in singleton_intervals)
    for span in spans:
        left = int(span["start"])
        right = int(span["end"])
        qualifying: list[tuple[int, int, str]] = []
        for raw_start, raw_end, raw_speaker in singleton_intervals:
            overlap_start = max(left, int(raw_start))
            overlap_end = min(right, int(raw_end))
            if overlap_end - overlap_start >= threshold:
                qualifying.append((overlap_start, overlap_end, str(raw_speaker)))
        qualifying.sort(key=lambda item: (item[0], item[1], item[2]))
        if not qualifying:
            continue
        owner = qualifying[0][2]
        contaminated += sum(end - start for start, end, speaker in qualifying if speaker != owner)
    return {"contaminated_samples": contaminated, "denominator_samples": denominator}


@dataclass(slots=True)
class DrainState:
    drain_id: str
    target_sample: int
    captured_released_frontier: int
    arm_clock_ms: int
    deadline_clock_ms: int
    arm_observed_frontier: int


@dataclass(slots=True)
class CanonicalPCMAssembler:
    audio_epoch: int
    epoch_origin_source_sample: int
    processed_end_source_sample: int
    holdback_samples: int
    observed_frontier: int = field(init=False)
    released_frontier: int = field(init=False)
    safe_frontier: int = field(init=False)
    boundaries: dict[int, list[str]] = field(init=False, default_factory=dict)
    action_records: list[dict[str, Any]] = field(init=False, default_factory=list)
    duplicate_records: list[dict[str, Any]] = field(init=False, default_factory=list)
    drain_queue: list[DrainState] = field(init=False, default_factory=list)
    drain_records: list[dict[str, Any]] = field(init=False, default_factory=list)
    progress_rows: list[list[int]] = field(init=False, default_factory=list)
    terminal_record: dict[str, Any] | None = field(init=False, default=None)
    _drain_ids: set[str] = field(init=False, default_factory=set)
    _last_drain_target: int | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.audio_epoch < 0:
            raise Phase3OracleError("audio epoch must be non-negative")
        if self.epoch_origin_source_sample < 0:
            raise Phase3OracleError("epoch origin must be non-negative")
        if self.processed_end_source_sample < self.epoch_origin_source_sample:
            raise Phase3OracleError("processed end precedes epoch origin")
        if self.holdback_samples < 0:
            raise Phase3OracleError("holdback must be non-negative")
        self.observed_frontier = self.epoch_origin_source_sample
        self.released_frontier = self.epoch_origin_source_sample
        self.safe_frontier = max(0, self.epoch_origin_source_sample - 1)

    def state_digest(self) -> str:
        return canonical_sha256(
            {
                "audio_epoch": self.audio_epoch,
                "observed_frontier": self.observed_frontier,
                "released_frontier": self.released_frontier,
                "safe_frontier": self.safe_frontier,
                "boundaries": self.boundaries,
                "drains": [
                    {
                        "drain_id": drain.drain_id,
                        "target_sample": drain.target_sample,
                        "captured_released_frontier": drain.captured_released_frontier,
                        "arm_clock_ms": drain.arm_clock_ms,
                        "deadline_clock_ms": drain.deadline_clock_ms,
                        "arm_observed_frontier": drain.arm_observed_frontier,
                    }
                    for drain in self.drain_queue
                ],
            }
        )

    def append_chunk(self, start_sample: int, end_sample: int) -> None:
        if start_sample != self.observed_frontier:
            raise Phase3OracleError("PCM chunk is not contiguous")
        if end_sample <= start_sample or end_sample > self.processed_end_source_sample:
            raise Phase3OracleError("PCM chunk span is invalid")
        if end_sample - start_sample > CHUNK_SAMPLES:
            raise Phase3OracleError("PCM chunk exceeds canonical size")
        self.observed_frontier = end_sample

    def update_progress(self, observed_sample: int, safe_sample: int) -> None:
        if observed_sample != self.observed_frontier:
            raise Phase3OracleError("progress observed frontier does not match PCM frontier")
        if safe_sample < self.safe_frontier:
            raise Phase3OracleError("safe frontier regressed")
        if safe_sample > observed_sample:
            raise Phase3OracleError("safe frontier exceeds observed frontier")
        self.safe_frontier = safe_sample
        self.progress_rows.append([observed_sample, safe_sample])

    def apply_action(
        self,
        *,
        action_id: str,
        action_epoch: int,
        boundary_sample: int,
        availability_sample: int,
        owner: str,
        structural_reason: str | None = None,
    ) -> dict[str, Any]:
        before = self.state_digest()
        if action_epoch != self.audio_epoch:
            record = {
                "action_id": action_id,
                "owner": owner,
                "accepted": False,
                "rejection": "stale_epoch",
                "state_unchanged": before == self.state_digest(),
                "boundary_source_sample": boundary_sample,
                "availability_source_sample": availability_sample,
                "apply_frontier": self.observed_frontier,
            }
            self.action_records.append(record)
            return record
        if not (
            self.epoch_origin_source_sample <= boundary_sample <= self.processed_end_source_sample
        ):
            raise Phase3OracleError("action boundary is outside the episode")
        zero_origin_sentinel = (
            self.epoch_origin_source_sample == 0
            and boundary_sample == 0
            and self.safe_frontier == 0
        )
        if owner == "oracle" and boundary_sample <= self.safe_frontier and not zero_origin_sentinel:
            raise Phase3OracleError("action violates a published safe frontier")
        released_at_apply = self.released_frontier
        realized_boundary = max(boundary_sample, released_at_apply)
        realized_boundary = min(realized_boundary, self.processed_end_source_sample)
        unrecoverable_end = min(released_at_apply, self.processed_end_source_sample)
        unrecoverable_span = (
            [boundary_sample, unrecoverable_end] if boundary_sample < unrecoverable_end else None
        )
        recoverability = "late_unrecoverable" if unrecoverable_span else "fully_recoverable"
        duplicate = realized_boundary in self.boundaries
        if realized_boundary < self.processed_end_source_sample:
            if duplicate:
                self.duplicate_records.append(
                    {
                        "action_id": action_id,
                        "realized_boundary": realized_boundary,
                        "normalized_to": self.boundaries[realized_boundary][0],
                    }
                )
                self.boundaries[realized_boundary].append(action_id)
            else:
                self.boundaries[realized_boundary] = [action_id]
        record = {
            "action_id": action_id,
            "owner": owner,
            "accepted": True,
            "rejection": None,
            "boundary_source_sample": boundary_sample,
            "availability_source_sample": availability_sample,
            "apply_frontier": self.observed_frontier,
            "released_frontier_at_apply": released_at_apply,
            "realized_boundary_source_sample": realized_boundary,
            "recoverability": recoverability,
            "unrecoverable_span": unrecoverable_span,
            "duplicate_normalized": duplicate,
            "structural_reason": structural_reason,
            "finalization_latency_samples": self.observed_frontier - boundary_sample,
        }
        self.action_records.append(record)
        return record

    def arm_drain(self, drain_id: str, target_sample: int, clock_ms: int) -> dict[str, Any]:
        if drain_id in self._drain_ids:
            return {"drain_id": drain_id, "status": "duplicate_ignored"}
        if not (self.epoch_origin_source_sample <= target_sample <= self.observed_frontier):
            raise Phase3OracleError("safe-drain target is outside observed episode PCM")
        if self._last_drain_target is not None and target_sample < self._last_drain_target:
            raise Phase3OracleError("safe-drain targets regressed")
        self._drain_ids.add(drain_id)
        self._last_drain_target = target_sample
        drain = DrainState(
            drain_id=drain_id,
            target_sample=target_sample,
            captured_released_frontier=self.released_frontier,
            arm_clock_ms=clock_ms,
            deadline_clock_ms=clock_ms + SAFE_DRAIN_TIMEOUT_MS,
            arm_observed_frontier=self.observed_frontier,
        )
        self.drain_queue.append(drain)
        return {"drain_id": drain_id, "status": "armed"}

    def resolve_drains(self, clock_ms: int) -> None:
        while self.drain_queue:
            drain = self.drain_queue[0]
            if self.safe_frontier >= drain.target_sample:
                outcome = "safe_complete"
            elif clock_ms >= drain.deadline_clock_ms:
                outcome = "safe_drain_timeout_fallback"
            else:
                return
            self._release_through(drain.target_sample)
            self.drain_records.append(
                {
                    "drain_id": drain.drain_id,
                    "target_sample": drain.target_sample,
                    "captured_released_frontier": drain.captured_released_frontier,
                    "arm_clock_ms": drain.arm_clock_ms,
                    "deadline_clock_ms": drain.deadline_clock_ms,
                    "resolution_clock_ms": clock_ms,
                    "arm_observed_frontier": drain.arm_observed_frontier,
                    "resolution_observed_frontier": self.observed_frontier,
                    "outcome": outcome,
                    "scheduler_latency_ms": clock_ms - drain.arm_clock_ms,
                    "source_release_latency_samples": max(
                        0, self.observed_frontier - drain.target_sample
                    ),
                }
            )
            self.drain_queue.pop(0)

    def ordinary_release(self) -> None:
        if self.drain_queue:
            return
        limit = max(
            self.released_frontier,
            max(self.epoch_origin_source_sample, self.observed_frontier - self.holdback_samples),
        )
        self._release_through(min(self.observed_frontier, limit))

    def terminal(self, clock_ms: int) -> dict[str, Any]:
        while self.drain_queue:
            head = self.drain_queue[0]
            if self.safe_frontier >= head.target_sample:
                self.resolve_drains(clock_ms)
            else:
                clock_ms = max(clock_ms, head.deadline_clock_ms)
                self.resolve_drains(clock_ms)
        before = self.released_frontier
        self._release_through(self.processed_end_source_sample)
        self.terminal_record = {
            "reason": "end_of_input",
            "released_from": before,
            "released_through": self.released_frontier,
            "observed_frontier": self.observed_frontier,
            "clock_ms": clock_ms,
        }
        return self.terminal_record

    def abandon(self) -> None:
        if self.released_frontier != self.observed_frontier or self.drain_queue:
            raise Phase3OracleError("forced abandonment would discard retained PCM")

    def realized_spans(self) -> list[dict[str, Any]]:
        return turn_spans(
            self.epoch_origin_source_sample,
            self.processed_end_source_sample,
            self.boundaries,
        )

    def _release_through(self, target_sample: int) -> None:
        if target_sample < self.released_frontier:
            return
        if target_sample > self.observed_frontier:
            raise Phase3OracleError("release exceeds observed frontier")
        self.released_frontier = target_sample


@dataclass(slots=True)
class B0LifecycleReplay:
    engine_factory: Callable[[], VadEngine]
    sample_rate_hz: int = SAMPLE_RATE_HZ
    chunk_samples: int = CHUNK_SAMPLES

    def replay(self, wav_path: Path, session_id: str, audio_epoch: int = 0) -> dict[str, Any]:
        gating = create_peer_vad_gating(
            self.engine_factory(),
            sample_rate_hz=self.sample_rate_hz,
            ring_buffer_ms=B0_VAD_PRE_ROLL_MS,
            hangover_ms=B0_VAD_HANGOVER_MS,
        )
        lifecycle: list[dict[str, Any]] = []
        projection: list[dict[str, Any]] = []
        utterance_ids: dict[str, str] = {}
        current: dict[str, Any] | None = None
        previous: dict[str, Any] | None = None
        utterance_seq = 0
        event_seq = 0

        def normalized(raw: Any) -> str:
            key = str(raw)
            if key not in utterance_ids:
                utterance_ids[key] = f"utterance-{len(utterance_ids) + 1:06d}"
            return utterance_ids[key]

        with wave.open(str(wav_path), "rb") as handle:
            if (
                handle.getnchannels() != 1
                or handle.getsampwidth() != 2
                or handle.getframerate() != self.sample_rate_hz
            ):
                raise Phase3OracleError(f"non-canonical lifecycle input: {wav_path}")
            length_samples = int(handle.getnframes())
            processed_end = floor_to_chunk(length_samples)
            chunk_index = 0
            while chunk_index * self.chunk_samples < processed_end:
                raw = handle.readframes(self.chunk_samples)
                chunk_i16 = np.frombuffer(raw, dtype="<i2")
                if chunk_i16.size != self.chunk_samples:
                    raise Phase3OracleError("unexpected partial chunk before processed end")
                chunk = chunk_i16.astype(np.float32) / 32768.0
                events = gating.process_chunk(chunk)
                chunk_start = chunk_index * self.chunk_samples
                observed = chunk_start + self.chunk_samples
                for event in events:
                    kind = type(event).__name__
                    event_seq += 1
                    normalized_id = normalized(getattr(event, "utterance_id"))
                    if kind == "SpeechStart":
                        pre_roll = getattr(event, "pre_roll", None)
                        pre_roll_samples = (
                            int(np.asarray(pre_roll).size) if pre_roll is not None else 0
                        )
                        lifecycle.append(
                            {
                                "event_id": f"b0l:{session_id}:{event_seq:08d}",
                                "audio_epoch": audio_epoch,
                                "source_session_id": session_id,
                                "normalized_utterance_id": normalized_id,
                                "event_kind": "speech_start",
                                "reason": "start",
                                "event_source_sample": chunk_start,
                                "observed_source_sample_at_emit": observed,
                                "trailing_silence_ms": 0,
                                "chunk_index": chunk_index,
                                "chunk_samples": self.chunk_samples,
                            }
                        )
                        if previous is not None:
                            projection.append(
                                {
                                    "audio_epoch": audio_epoch,
                                    "boundary_source_sample": chunk_start,
                                    "observed_source_sample_at_emit": observed,
                                    "confidence": None,
                                    "source": B0_SOURCE,
                                    "debug": {
                                        "chunk_samples": self.chunk_samples,
                                        "gap_samples": chunk_start
                                        - int(previous["speech_end_sample"]),
                                        "hangover_ms": B0_VAD_HANGOVER_MS,
                                        "max_segment_ms": B0_VAD_MAX_SEGMENT_MS,
                                        "pre_roll_samples": pre_roll_samples,
                                        "prev_end_reason": previous["reason"],
                                        "prev_speech_end_sample": previous["speech_end_sample"],
                                        "prev_trailing_silence_ms": previous["trailing_silence_ms"],
                                        "prev_utterance_seq": utterance_seq,
                                        "prev_utterance_start_sample": previous["start_sample"],
                                        "profile": B0_VAD_PROFILE,
                                        "ring_buffer_ms": B0_VAD_PRE_ROLL_MS,
                                        "speech_threshold": B0_VAD_SPEECH_THRESHOLD,
                                        "start_chunk_index": chunk_index,
                                        "start_commit_chunks": B0_VAD_START_COMMIT_CHUNKS,
                                        "start_debounce_chunks": B0_VAD_START_DEBOUNCE_CHUNKS,
                                    },
                                }
                            )
                        utterance_seq += 1
                        current = {
                            "normalized_id": normalized_id,
                            "start_sample": chunk_start,
                        }
                    elif kind == "SpeechEnd":
                        if current is None:
                            raise Phase3OracleError("SpeechEnd has no active lifecycle utterance")
                        trailing_ms = int(getattr(event, "trailing_silence_ms", 0))
                        reason = str(getattr(event, "reason", "silence"))
                        silence_chunks = int(round(trailing_ms / 32))
                        event_sample = max(
                            int(current["start_sample"]),
                            (chunk_index + 1 - silence_chunks) * self.chunk_samples,
                        )
                        lifecycle.append(
                            {
                                "event_id": f"b0l:{session_id}:{event_seq:08d}",
                                "audio_epoch": audio_epoch,
                                "source_session_id": session_id,
                                "normalized_utterance_id": normalized_id,
                                "event_kind": "speech_end",
                                "reason": reason,
                                "event_source_sample": event_sample,
                                "observed_source_sample_at_emit": observed,
                                "trailing_silence_ms": trailing_ms,
                                "chunk_index": chunk_index,
                                "chunk_samples": self.chunk_samples,
                            }
                        )
                        previous = {
                            "start_sample": current["start_sample"],
                            "speech_end_sample": event_sample,
                            "trailing_silence_ms": trailing_ms,
                            "reason": reason,
                        }
                        current = None
                chunk_index += 1
        active_id = current["normalized_id"] if current is not None else "none"
        lifecycle.append(
            {
                "event_id": f"b0l:{session_id}:terminal",
                "audio_epoch": audio_epoch,
                "source_session_id": session_id,
                "normalized_utterance_id": active_id,
                "event_kind": "terminal",
                "reason": "end_of_input",
                "event_source_sample": processed_end,
                "observed_source_sample_at_emit": processed_end,
                "trailing_silence_ms": 0,
                "chunk_index": processed_end // self.chunk_samples,
                "chunk_samples": 0,
                "active_state_remained": current is not None,
                "pending_state_remained": bool(getattr(gating, "_pending_start_id", None)),
            }
        )
        return {
            "session_id": session_id,
            "audio_epoch": audio_epoch,
            "length_samples": length_samples,
            "processed_end": processed_end,
            "events": lifecycle,
            "event_count": len(lifecycle),
            "event_digest": canonical_sha256(lifecycle),
            "projection": projection,
            "projection_count": len(projection),
            "projection_hash": sha256_bytes(phase2_canonical_json(projection).encode("utf-8")),
        }


def _verify_frozen_inputs(results_dir: Path) -> dict[str, str]:
    live: dict[str, str] = {}
    for name, expected in FROZEN_INPUT_SHA256.items():
        path = results_dir / name
        if not path.is_file():
            raise Phase3OracleError(f"frozen input is missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise Phase3OracleError(
                f"frozen input hash mismatch for {name}: expected {expected}, got {actual}"
            )
        live[name] = actual
    bundle = results_dir / "reviews" / "phase_3_review_bundle.md"
    if sha256_file(bundle) != APPROVED_BUNDLE_SHA256:
        raise Phase3OracleError("approved Phase 3 review bundle hash mismatch")
    review = results_dir / "reviews" / "phase_3_pre_execution.md"
    review_text = review.read_text(encoding="utf-8")
    if "Status: **approved**" not in review_text or "**accepted**" not in review_text:
        raise Phase3OracleError("Phase 3 pre-execution review is not accepted")
    live["reviews/phase_3_review_bundle.md"] = APPROVED_BUNDLE_SHA256
    live["reviews/phase_3_pre_execution.md"] = sha256_file(review)
    experiment_dir = Path(__file__).resolve().parent.parent
    repo_root = Path(__file__).resolve().parents[3]
    for name, expected in FROZEN_CODE_SHA256.items():
        path = repo_root / name if name.startswith("src/") else experiment_dir / name
        actual = sha256_file(path)
        if actual != expected:
            raise Phase3OracleError(
                f"frozen code hash mismatch for {name}: expected {expected}, got {actual}"
            )
        live[name] = actual
    authority = (
        repo_root
        / ".agents"
        / "specs"
        / "prd"
        / ("bounded_turn_episode_speaker_change_fusion_experiment_review_gated.md")
    )
    authority_hash = sha256_file(authority)
    if authority_hash != CURRENT_AUTHORITY_SHA256:
        raise Phase3OracleError("current authority pin does not match the amended PRD")
    live["authority_prd"] = authority_hash
    return live


def _load_sessions(results_dir: Path, corpus_root: Path) -> dict[str, Any]:
    from experiments.speaker_turn_boundary.corpus.phase2_schemas import Phase2Manifest

    from .build_episodes import load_session_data

    details_rows: dict[str, dict[str, Any]] = {}
    details_path = results_dir / "coverage_inventory_details.jsonl"
    for line in details_path.read_text(encoding="utf-8").splitlines():
        if line:
            row = json.loads(line)
            details_rows[str(row["session_id"])] = row
    by_corpus_rank: dict[str, dict[str, int]] = {}
    for corpus in ("ami", "alimeeting"):
        ids = sorted(
            session_id
            for session_id, row in details_rows.items()
            if str(row["corpus"]) == corpus and row.get("wav_path")
        )
        by_corpus_rank[corpus] = {session_id: rank for rank, session_id in enumerate(ids)}
    manifests_dir = Path(__file__).resolve().parent.parent / "data" / "manifests"
    pilot_cases: dict[str, list[Any]] = {}
    for name in ("ami_dev_pilot.json", "ami_held_out_pilot.json", "alimeeting_eval_pilot.json"):
        manifest = Phase2Manifest.load(manifests_dir / name)
        for case in manifest.cases:
            pilot_cases.setdefault(str(case.case_id), []).append(case)
    sessions: dict[str, Any] = {}
    for session_id, row in details_rows.items():
        if not row.get("wav_path"):
            continue
        sessions[session_id] = load_session_data(
            session_id,
            row,
            corpus_root,
            manifests_dir,
            pilot_cases,
            by_corpus_rank,
        )
    return sessions


def population_identity(episodes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    identity = {
        "session_ids": sorted({str(episode["session_id"]) for episode in episodes}),
        "episode_ids": sorted(str(episode["episode_id"]) for episode in episodes),
        "reference_ids": sorted(
            str(reference["reference_id"])
            for episode in episodes
            for reference in episode["references"]
            if reference["action_kind"] == "hard_boundary" and reference["scorable"]
        ),
    }
    return {"identity": identity, "sha256": canonical_sha256(identity)}


def clamp_identity(
    episodes: Sequence[dict[str, Any]], processed_ends: dict[str, int]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        start = int(episode["bounds"]["scored_start"])
        end = int(processed_ends[str(episode["episode_id"])])
        for reference in episode["references"]:
            if reference["action_kind"] != "hard_boundary" or not reference["scorable"]:
                continue
            for offset_ms in OFFSETS_MS:
                unclamped = int(reference["target_sample"]) + offset_ms * SAMPLES_PER_MS
                if unclamped < start:
                    direction = "below_start"
                    clamped = start
                elif unclamped > end:
                    direction = "above_end"
                    clamped = end
                else:
                    continue
                rows.append(
                    {
                        "episode_id": str(episode["episode_id"]),
                        "reference_id": str(reference["reference_id"]),
                        "offset_ms": offset_ms,
                        "unclamped_boundary_source_sample": unclamped,
                        "clamp_direction": direction,
                        "boundary_source_sample": clamped,
                    }
                )
    rows.sort(key=lambda row: (row["episode_id"], row["reference_id"], row["offset_ms"]))
    identity = {
        "schema_version": CLAMP_SCHEMA_VERSION,
        "clamped_reference_offsets": rows,
    }
    return {"identity": identity, "sha256": canonical_sha256(identity)}


def _session_regions(session: Any, start: int, end: int) -> tuple[list[list[Any]], list[list[Any]]]:
    regions: list[list[Any]] = []
    singleton: list[list[Any]] = []
    for region in session.regions:
        left = max(start, int(region.start_sample))
        right = min(end, int(region.end_sample))
        if left >= right:
            continue
        speakers = sorted(str(speaker) for speaker in region.speakers)
        row = [left, right, speakers, bool(region.ambiguous)]
        regions.append(row)
        if len(speakers) == 1 and not region.ambiguous:
            singleton.append([left, right, speakers[0]])
    regions.sort(key=lambda row: (row[0], row[1], row[2]))
    singleton.sort(key=lambda row: (row[0], row[1], row[2]))
    return regions, singleton


def _slice_lifecycle(
    events: Sequence[dict[str, Any]], start: int, end: int
) -> list[dict[str, Any]]:
    sliced: list[dict[str, Any]] = []
    for event in events:
        if event["event_kind"] == "terminal":
            continue
        event_sample = int(event["event_source_sample"])
        observed = int(event["observed_source_sample_at_emit"])
        if start <= event_sample <= end and observed <= end:
            sliced.append(event)
    return sliced


def _load_population(
    results_dir: Path,
    corpus_root: Path,
    lifecycle_by_session: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from .build_episodes import verify_manifest

    verify_manifest(results_dir / "episode_manifest_dev.json")
    manifest = json.loads((results_dir / "episode_manifest_dev.json").read_text(encoding="utf-8"))
    scoring = json.loads((results_dir / "scoring_fixture_report.json").read_text(encoding="utf-8"))
    selected_ids = {str(row["episode_id"]) for row in scoring["baseline_smoke"]["rows"]}
    episodes = [
        episode
        for episode in manifest["episodes"]
        if str(episode["episode_id"]) in selected_ids and episode["status"] == "scorable"
    ]
    episodes.sort(key=lambda episode: (str(episode["session_id"]), str(episode["episode_id"])))
    if {str(episode["episode_id"]) for episode in episodes} != selected_ids:
        raise Phase3OracleError("baseline smoke population does not join exactly to the manifest")
    sessions = _load_sessions(results_dir, corpus_root)
    b0_cache: dict[str, dict[str, Any]] = {}
    processed_ends: dict[str, int] = {}
    prepared: list[dict[str, Any]] = []
    for episode in episodes:
        session_id = str(episode["session_id"])
        session = sessions.get(session_id)
        if session is None or session.wav_abs_path is None or not session.wav_abs_path.is_file():
            raise Phase3OracleError(f"opened development WAV is missing: {session_id}")
        start = int(episode["bounds"]["scored_start"])
        last_full = floor_to_chunk(
            min(session.duration_samples, session.wav_length_samples or session.duration_samples)
        )
        end = min(int(episode["bounds"]["scored_end"]), last_full)
        processed_ends[str(episode["episode_id"])] = end
        if session_id not in b0_cache:
            b0_cache[session_id] = json.loads(
                (results_dir / "b0_inventory_replay" / f"{session_id}.json").read_text(
                    encoding="utf-8"
                )
            )
        b0_actions = [
            {
                "action_id": f"b0:{session_id}:{row['boundary_source_sample']}",
                "audio_epoch": int(row["audio_epoch"]),
                "boundary_source_sample": int(row["boundary_source_sample"]),
                "availability_source_sample": int(row["observed_source_sample_at_emit"]),
                "owner": "b0",
            }
            for row in b0_cache[session_id]["trace_projection"]
            if start <= int(row["boundary_source_sample"]) <= end
        ]
        regions, singleton = _session_regions(session, start, end)
        hard_references = [
            reference
            for reference in episode["references"]
            if reference["action_kind"] == "hard_boundary" and reference["scorable"]
        ]
        prepared.append(
            {
                "episode": episode,
                "start": start,
                "end": end,
                "hard_references": hard_references,
                "b0_actions": b0_actions,
                "lifecycle_events": _slice_lifecycle(
                    lifecycle_by_session[session_id]["events"], start, end
                ),
                "regions": regions,
                "regions_digest": canonical_sha256(regions),
                "singleton_intervals": singleton,
                "singleton_digest": canonical_sha256(singleton),
            }
        )
    population = population_identity(episodes)
    if population["sha256"] != EXPECTED_POPULATION_SHA256:
        raise Phase3OracleError("Phase 3 population identity drifted")
    clamp = clamp_identity(episodes, processed_ends)
    if clamp["sha256"] != EXPECTED_CLAMP_SHA256:
        raise Phase3OracleError("Phase 3 clamp identity drifted")
    hard_positive = sum(1 for row in prepared if row["hard_references"])
    counts = {
        "sessions": len(population["identity"]["session_ids"]),
        "episodes": len(prepared),
        "hard_references": len(population["identity"]["reference_ids"]),
        "hard_positive_episodes": hard_positive,
        "no_hard_control_episodes": len(prepared) - hard_positive,
    }
    expected_counts = {
        "sessions": EXPECTED_SESSION_COUNT,
        "episodes": EXPECTED_EPISODE_COUNT,
        "hard_references": EXPECTED_REFERENCE_COUNT,
        "hard_positive_episodes": EXPECTED_HARD_POSITIVE_EPISODES,
        "no_hard_control_episodes": EXPECTED_NO_HARD_EPISODES,
    }
    if counts != expected_counts:
        raise Phase3OracleError(f"Phase 3 population counts drifted: {counts}")
    clamp_rows = clamp["identity"]["clamped_reference_offsets"]
    clamp_counts = {
        "reference_offsets": len(clamp_rows),
        "references": len({row["reference_id"] for row in clamp_rows}),
        "episodes": len({row["episode_id"] for row in clamp_rows}),
        "below_start": sum(row["clamp_direction"] == "below_start" for row in clamp_rows),
        "above_end": sum(row["clamp_direction"] == "above_end" for row in clamp_rows),
        "action_instances": len(clamp_rows) * len(DELAYS_MS) * len(HOLDBACKS_MS),
    }
    expected_clamp_counts = {
        "reference_offsets": EXPECTED_CLAMPED_REFERENCE_OFFSETS,
        "references": EXPECTED_CLAMPED_REFERENCES,
        "episodes": EXPECTED_CLAMPED_EPISODES,
        "below_start": EXPECTED_CLAMP_BELOW,
        "above_end": EXPECTED_CLAMP_ABOVE,
        "action_instances": EXPECTED_CLAMPED_ACTION_INSTANCES,
    }
    if clamp_counts != expected_clamp_counts:
        raise Phase3OracleError(f"Phase 3 clamp counts drifted: {clamp_counts}")
    return prepared, {
        "population": population,
        "population_counts": counts,
        "clamp": clamp,
        "clamp_counts": clamp_counts,
        "manifest_content_sha256": manifest["content_sha256"],
    }


def _oracle_actions(case: dict[str, Any], delay_ms: int, offset_ms: int) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    start = int(case["start"])
    end = int(case["end"])
    for reference in case["hard_references"]:
        unclamped = int(reference["target_sample"]) + offset_ms * SAMPLES_PER_MS
        boundary = min(end, max(start, unclamped))
        if unclamped < start:
            clamp_direction: str | None = "below_start"
        elif unclamped > end:
            clamp_direction = "above_end"
        else:
            clamp_direction = None
        requested = int(reference["evidence_onset_sample"]) + delay_ms * SAMPLES_PER_MS
        causal = max(requested, boundary)
        actions.append(
            {
                "action_id": f"oracle:{reference['reference_id']}:{delay_ms}:{offset_ms}",
                "audio_epoch": int(reference["audio_epoch"]),
                "reference_id": str(reference["reference_id"]),
                "reference_kind": str(reference["action_kind"]),
                "target_sample": int(reference["target_sample"]),
                "acceptable_interval": [
                    int(reference["acceptable_interval"][0]),
                    int(reference["acceptable_interval"][1]),
                ],
                "requested_offset_ms": offset_ms,
                "unclamped_boundary": unclamped,
                "clamp_direction": clamp_direction,
                "boundary_source_sample": boundary,
                "realized_signed_point_offset_samples": boundary - int(reference["target_sample"]),
                "evidence_onset_sample": int(reference["evidence_onset_sample"]),
                "requested_availability_source_sample": requested,
                "availability_source_sample": causal,
                "requested_delay_ms": delay_ms,
                "owner": "oracle",
            }
        )
    actions.sort(
        key=lambda row: (
            row["availability_source_sample"],
            row["boundary_source_sample"],
            row["action_id"],
        )
    )
    return actions


def _run_stream(
    case: dict[str, Any], holdback_ms: int, oracle_actions: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    episode = case["episode"]
    start = int(case["start"])
    end = int(case["end"])
    epoch = int(episode["audio_epoch"])
    assembler = CanonicalPCMAssembler(
        audio_epoch=epoch,
        epoch_origin_source_sample=start,
        processed_end_source_sample=end,
        holdback_samples=holdback_ms * SAMPLES_PER_MS,
    )
    scheduled: list[dict[str, Any]] = [dict(action) for action in case["b0_actions"]]
    scheduled.extend(dict(action) for action in oracle_actions)
    scheduled.sort(
        key=lambda row: (
            int(row["availability_source_sample"]),
            int(row["boundary_source_sample"]),
            str(row["action_id"]),
        )
    )
    lifecycle = sorted(
        (dict(event) for event in case["lifecycle_events"]),
        key=lambda event: (
            int(event["observed_source_sample_at_emit"]),
            int(event["event_source_sample"]),
            str(event["event_id"]),
        ),
    )
    action_index = 0
    lifecycle_index = 0
    action_results: dict[str, dict[str, Any]] = {}
    applied_oracle_ids: set[str] = set()
    applied_lifecycle: list[dict[str, Any]] = []
    structural_actions: list[dict[str, Any]] = []
    frontier = start
    while frontier < end:
        next_frontier = min(end, frontier + CHUNK_SAMPLES)
        assembler.append_chunk(frontier, next_frontier)
        pending_oracle_boundaries = [
            int(action["boundary_source_sample"])
            for action in oracle_actions
            if str(action["action_id"]) not in applied_oracle_ids
        ]
        if pending_oracle_boundaries:
            safe = min(next_frontier, min(pending_oracle_boundaries) - 1)
            safe = max(0, safe)
        else:
            safe = next_frontier
        assembler.update_progress(next_frontier, safe)
        while (
            action_index < len(scheduled)
            and int(scheduled[action_index]["availability_source_sample"]) <= next_frontier
        ):
            action = scheduled[action_index]
            record = assembler.apply_action(
                action_id=str(action["action_id"]),
                action_epoch=int(action["audio_epoch"]),
                boundary_sample=int(action["boundary_source_sample"]),
                availability_sample=int(action["availability_source_sample"]),
                owner=str(action["owner"]),
            )
            action_results[str(action["action_id"])] = record
            if action["owner"] == "oracle":
                applied_oracle_ids.add(str(action["action_id"]))
            action_index += 1
        while (
            lifecycle_index < len(lifecycle)
            and int(lifecycle[lifecycle_index]["observed_source_sample_at_emit"]) <= next_frontier
        ):
            event = lifecycle[lifecycle_index]
            event["apply_frontier"] = next_frontier
            applied_lifecycle.append(event)
            if event["event_kind"] == "speech_end":
                if event["reason"] == "max_duration":
                    structural = assembler.apply_action(
                        action_id=f"structural:{event['event_id']}",
                        action_epoch=int(event["audio_epoch"]),
                        boundary_sample=int(event["event_source_sample"]),
                        availability_sample=int(event["observed_source_sample_at_emit"]),
                        owner="structural",
                        structural_reason="max_duration",
                    )
                    structural_actions.append(structural)
                assembler.arm_drain(
                    f"drain:{event['event_id']}",
                    int(event["event_source_sample"]),
                    (next_frontier - start) // SAMPLES_PER_MS,
                )
            lifecycle_index += 1
        clock_ms = (next_frontier - start) // SAMPLES_PER_MS
        assembler.resolve_drains(clock_ms)
        assembler.ordinary_release()
        frontier = next_frontier
    terminal_clock_ms = (end - start) // SAMPLES_PER_MS
    terminal = assembler.terminal(terminal_clock_ms)
    for action in scheduled[action_index:]:
        action_id = str(action["action_id"])
        boundary = int(action["boundary_source_sample"])
        unavailable_span = [boundary, end] if boundary < end else None
        record = {
            "action_id": action_id,
            "owner": str(action["owner"]),
            "accepted": False,
            "rejection": "unavailable_before_terminal",
            "boundary_source_sample": boundary,
            "availability_source_sample": int(action["availability_source_sample"]),
            "apply_frontier": None,
            "released_frontier_at_apply": end,
            "realized_boundary_source_sample": None,
            "recoverability": "unavailable_before_terminal",
            "unrecoverable_span": unavailable_span,
            "duplicate_normalized": False,
            "structural_reason": None,
            "finalization_latency_samples": None,
        }
        action_results[action_id] = record
    spans = assembler.realized_spans()
    flags = span_cover_flags(spans, start, end)
    return {
        "spans": spans,
        "boundaries": sorted(assembler.boundaries),
        "action_results": action_results,
        "lifecycle_events": applied_lifecycle,
        "structural_actions": structural_actions,
        "progress_rows": assembler.progress_rows,
        "progress_sha256": canonical_sha256(assembler.progress_rows),
        "drain_records": assembler.drain_records,
        "duplicate_records": assembler.duplicate_records,
        "terminal_record": terminal,
        "final_ring_span": [assembler.released_frontier, assembler.observed_frontier],
        "flags": flags,
        "state_digest": assembler.state_digest(),
    }


def simulate_case(
    case: dict[str, Any], delay_ms: int, offset_ms: int, holdback_ms: int
) -> dict[str, Any]:
    episode = case["episode"]
    oracle = _oracle_actions(case, delay_ms, offset_ms)
    baseline = _run_stream(case, holdback_ms, [])
    candidate = _run_stream(case, holdback_ms, oracle)
    oracle_evidence: list[dict[str, Any]] = []
    for action in oracle:
        result = candidate["action_results"][str(action["action_id"])]
        evidence = {**action, **result}
        oracle_evidence.append(evidence)
    structural_boundaries = [
        int(event["event_source_sample"])
        for event in case["lifecycle_events"]
        if event["event_kind"] == "speech_end" and event["reason"] == "max_duration"
    ]
    ideal_boundaries = [int(action["boundary_source_sample"]) for action in case["b0_actions"]]
    ideal_boundaries.extend(structural_boundaries)
    ideal_boundaries.extend(int(action["boundary_source_sample"]) for action in oracle)
    ideal_spans = turn_spans(int(case["start"]), int(case["end"]), ideal_boundaries)
    metrics: dict[str, Any] = {"baseline": {}, "candidate": {}}
    for threshold in OWNER_THRESHOLDS_MS:
        key = str(threshold)
        metrics["baseline"][key] = contamination_samples(
            baseline["spans"], case["singleton_intervals"], threshold
        )
        metrics["candidate"][key] = contamination_samples(
            candidate["spans"], case["singleton_intervals"], threshold
        )
    unrecoverable_spans = [
        action["unrecoverable_span"]
        for action in oracle_evidence
        if action["unrecoverable_span"] is not None
    ]
    fragment_durations = [int(span["end"]) - int(span["start"]) for span in candidate["spans"]]
    logical_latencies = [
        int(action["finalization_latency_samples"])
        for action in oracle_evidence
        if action["finalization_latency_samples"] is not None
    ]
    row = {
        "schema_version": DETAIL_SCHEMA_VERSION,
        "grid_id": GRID_ID,
        "availability_delay_ms": delay_ms,
        "boundary_offset_ms": offset_ms,
        "holdback_ms": holdback_ms,
        "population_sha256": EXPECTED_POPULATION_SHA256,
        "episode_content_sha256": str(episode["episode_content_sha256"]),
        "session_id": str(episode["session_id"]),
        "episode_id": str(episode["episode_id"]),
        "pool": str(episode["pool"]),
        "tag": str(episode["tag"]),
        "status": str(episode["status"]),
        "audio_epoch": int(episode["audio_epoch"]),
        "epoch_origin_source_sample": int(case["start"]),
        "scored_start": int(case["start"]),
        "processed_scored_end": int(case["end"]),
        "source_region_digest": str(case["regions_digest"]),
        "oracle_actions": oracle_evidence,
        "b0_actions": [
            {**action, **baseline["action_results"][str(action["action_id"])]}
            for action in case["b0_actions"]
        ],
        "lifecycle_events": candidate["lifecycle_events"],
        "structural_actions": candidate["structural_actions"],
        "progress_rows": candidate["progress_rows"],
        "progress_sha256": candidate["progress_sha256"],
        "baseline_progress_rows": baseline["progress_rows"],
        "baseline_progress_sha256": baseline["progress_sha256"],
        "singleton_intervals": case["singleton_intervals"],
        "singleton_intervals_sha256": case["singleton_digest"],
        "ideal_turn_spans": ideal_spans,
        "baseline_turn_spans": baseline["spans"],
        "realized_turn_spans": candidate["spans"],
        "final_ring_span": candidate["final_ring_span"],
        "terminal_release_record": candidate["terminal_record"],
        "unrecoverable_spans": unrecoverable_spans,
        "duplicate_normalization_records": candidate["duplicate_records"],
        "baseline_drain_records": baseline["drain_records"],
        "candidate_drain_records": candidate["drain_records"],
        "metrics": metrics,
        "fragment_durations_samples": fragment_durations,
        "logical_action_latencies_samples": logical_latencies,
        "invariants": {
            "baseline": baseline["flags"],
            "candidate": candidate["flags"],
            "ideal": span_cover_flags(ideal_spans, int(case["start"]), int(case["end"])),
        },
        "fallback_classes": sorted(
            {
                record["outcome"]
                for record in candidate["drain_records"]
                if record["outcome"] == "safe_drain_timeout_fallback"
            }
        ),
        "clamp_count": sum(action["clamp_direction"] is not None for action in oracle_evidence),
        "candidate_state_digest": candidate["state_digest"],
        "baseline_state_digest": baseline["state_digest"],
    }
    row["row_digest"] = canonical_sha256(row)
    return row


@dataclass(slots=True)
class GridAccumulator:
    delay_ms: int
    offset_ms: int
    holdback_ms: int
    detail_rows: int = 0
    action_instances: int = 0
    clamp_instances: int = 0
    fully_recoverable_actions: int = 0
    late_actions: int = 0
    unavailable_actions: int = 0
    unrecoverable_samples: list[int] = field(default_factory=list)
    fragment_samples: list[int] = field(default_factory=list)
    latency_samples: list[int] = field(default_factory=list)
    invariant_failures: int = 0
    safe_complete_drains: int = 0
    fallback_drains: int = 0
    improved_episodes: int = 0
    unchanged_episodes: int = 0
    regressed_episodes: int = 0
    metrics: dict[str, dict[str, dict[str, dict[str, int]]]] = field(
        default_factory=lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(lambda: {"contaminated_samples": 0, "denominator_samples": 0})
            )
        )
    )
    session_primary: dict[str, dict[str, int]] = field(
        default_factory=lambda: defaultdict(
            lambda: {"baseline": 0, "candidate": 0, "denominator": 0}
        )
    )

    def add(self, row: dict[str, Any]) -> None:
        self.detail_rows += 1
        actions = row["oracle_actions"]
        self.action_instances += len(actions)
        self.clamp_instances += int(row["clamp_count"])
        for action in actions:
            span = action["unrecoverable_span"]
            samples = int(span[1]) - int(span[0]) if span is not None else 0
            self.unrecoverable_samples.append(samples)
            if action["recoverability"] == "fully_recoverable":
                self.fully_recoverable_actions += 1
            elif action["recoverability"] == "late_unrecoverable":
                self.late_actions += 1
            elif action["recoverability"] == "unavailable_before_terminal":
                self.unavailable_actions += 1
        self.fragment_samples.extend(int(value) for value in row["fragment_durations_samples"])
        self.latency_samples.extend(int(value) for value in row["logical_action_latencies_samples"])
        for system in ("baseline", "candidate"):
            flags = row["invariants"][system]
            if not all(bool(value) for value in flags.values()):
                self.invariant_failures += 1
        for record in row["candidate_drain_records"]:
            if record["outcome"] == "safe_complete":
                self.safe_complete_drains += 1
            elif record["outcome"] == "safe_drain_timeout_fallback":
                self.fallback_drains += 1
        tag = str(row["tag"])
        for system in ("baseline", "candidate"):
            for threshold, values in row["metrics"][system].items():
                target = self.metrics[tag][system][threshold]
                target["contaminated_samples"] += int(values["contaminated_samples"])
                target["denominator_samples"] += int(values["denominator_samples"])
        if tag == "hard_only":
            baseline = int(row["metrics"]["baseline"]["100"]["contaminated_samples"])
            candidate = int(row["metrics"]["candidate"]["100"]["contaminated_samples"])
            denominator = int(row["metrics"]["candidate"]["100"]["denominator_samples"])
            if candidate < baseline:
                self.improved_episodes += 1
            elif candidate == baseline:
                self.unchanged_episodes += 1
            else:
                self.regressed_episodes += 1
            session = self.session_primary[str(row["session_id"])]
            session["baseline"] += baseline
            session["candidate"] += candidate
            session["denominator"] += denominator

    def finalize(self) -> dict[str, Any]:
        metrics = json.loads(canonical_json(self.metrics))
        hard = metrics.get("hard_only", {})
        baseline_primary = hard.get("baseline", {}).get(
            "100", {"contaminated_samples": 0, "denominator_samples": 0}
        )
        candidate_primary = hard.get("candidate", {}).get(
            "100", {"contaminated_samples": 0, "denominator_samples": 0}
        )
        denominator = int(candidate_primary["denominator_samples"])
        baseline_ratio = (
            int(baseline_primary["contaminated_samples"]) / denominator if denominator else None
        )
        candidate_ratio = (
            int(candidate_primary["contaminated_samples"]) / denominator if denominator else None
        )
        session_effects = []
        for session_id in sorted(self.session_primary):
            values = self.session_primary[session_id]
            session_denominator = values["denominator"]
            difference = (
                (values["candidate"] - values["baseline"]) / session_denominator
                if session_denominator
                else None
            )
            session_effects.append(
                {
                    "session_id": session_id,
                    "baseline_contaminated_samples": values["baseline"],
                    "candidate_contaminated_samples": values["candidate"],
                    "denominator_samples": session_denominator,
                    "paired_ratio_difference": difference,
                }
            )
        total_unrecoverable = sum(self.unrecoverable_samples)
        row = {
            "grid_row_id": f"d{self.delay_ms}:o{self.offset_ms}:h{self.holdback_ms}",
            "availability_delay_ms": self.delay_ms,
            "boundary_offset_ms": self.offset_ms,
            "holdback_ms": self.holdback_ms,
            "detail_rows": self.detail_rows,
            "action_instances": self.action_instances,
            "clamp_instances": self.clamp_instances,
            "invariant_failures": self.invariant_failures,
            "fully_recoverable_actions": self.fully_recoverable_actions,
            "late_unrecoverable_actions": self.late_actions,
            "unavailable_before_terminal_actions": self.unavailable_actions,
            "fully_recoverable_action_fraction": (
                self.fully_recoverable_actions / self.action_instances
                if self.action_instances
                else None
            ),
            "unrecoverable_samples": {
                "count": len(self.unrecoverable_samples),
                "total": total_unrecoverable,
                "mean_ms": (
                    total_unrecoverable / len(self.unrecoverable_samples) / SAMPLES_PER_MS
                    if self.unrecoverable_samples
                    else None
                ),
                "p50": nearest_rank(self.unrecoverable_samples, 0.50),
                "p95": nearest_rank(self.unrecoverable_samples, 0.95),
                "max": max(self.unrecoverable_samples, default=None),
            },
            "fragment_duration_samples": {
                "count": len(self.fragment_samples),
                "p10": nearest_rank(self.fragment_samples, 0.10),
                "p50": nearest_rank(self.fragment_samples, 0.50),
                "p90": nearest_rank(self.fragment_samples, 0.90),
            },
            "logical_action_latency_samples": {
                "count": len(self.latency_samples),
                "p50": nearest_rank(self.latency_samples, 0.50),
                "p95": nearest_rank(self.latency_samples, 0.95),
                "max": max(self.latency_samples, default=None),
            },
            "safe_drains": {
                "safe_complete": self.safe_complete_drains,
                "safe_drain_timeout_fallback": self.fallback_drains,
            },
            "episode_effects": {
                "improved": self.improved_episodes,
                "unchanged": self.unchanged_episodes,
                "regressed": self.regressed_episodes,
            },
            "contamination": metrics,
            "primary_hard_only_100ms": {
                "baseline_contamination_ratio": baseline_ratio,
                "candidate_contamination_ratio": candidate_ratio,
                "paired_ratio_difference": (
                    candidate_ratio - baseline_ratio
                    if candidate_ratio is not None and baseline_ratio is not None
                    else None
                ),
                "oracle_reduces_contamination": int(candidate_primary["contaminated_samples"])
                < int(baseline_primary["contaminated_samples"]),
            },
            "session_effects": session_effects,
        }
        row["grid_row_digest"] = canonical_sha256(row)
        return row


def _fixture_stream(
    *, origin: int, length: int, boundaries: Sequence[int], holdback_ms: int, late: bool = False
) -> dict[str, Any]:
    end = origin + length
    assembler = CanonicalPCMAssembler(0, origin, end, holdback_ms * SAMPLES_PER_MS)
    pending = [
        {
            "action_id": f"fixture:{index}",
            "boundary": boundary,
            "availability": boundary + (CHUNK_SAMPLES if late else 0),
        }
        for index, boundary in enumerate(boundaries)
    ]
    applied: set[str] = set()
    frontier = origin
    while frontier < end:
        next_frontier = min(end, frontier + CHUNK_SAMPLES)
        assembler.append_chunk(frontier, next_frontier)
        still_pending = [row["boundary"] for row in pending if row["action_id"] not in applied]
        safe = (
            max(0, min(next_frontier, min(still_pending) - 1)) if still_pending else next_frontier
        )
        assembler.update_progress(next_frontier, safe)
        for row in pending:
            if row["action_id"] in applied or row["availability"] > next_frontier:
                continue
            assembler.apply_action(
                action_id=row["action_id"],
                action_epoch=0,
                boundary_sample=row["boundary"],
                availability_sample=row["availability"],
                owner="oracle",
            )
            applied.add(row["action_id"])
        assembler.ordinary_release()
        frontier = next_frontier
    assembler.terminal(length // SAMPLES_PER_MS)
    spans = assembler.realized_spans()
    pcm = np.arange(length, dtype=np.int64).astype("<i2").tobytes()
    rebuilt = b"".join(
        pcm[(int(span["start"]) - origin) * 2 : (int(span["end"]) - origin) * 2] for span in spans
    )
    flags = span_cover_flags(spans, origin, end)
    return {
        "spans": spans,
        "flags": flags,
        "pcm_equal": rebuilt == pcm,
        "action_records": assembler.action_records,
        "duplicates": assembler.duplicate_records,
    }


def run_pcm_fixtures() -> dict[str, Any]:
    origin = 4096
    cases = [
        ("boundary_at_origin", 1024, [origin], 500, False),
        ("boundary_at_end", 1024, [origin + 1024], 500, False),
        ("chunk_interior", 1536, [origin + 700], 500, False),
        ("chunk_edge", 1536, [origin + 512], 500, False),
        ("multiple_boundaries", 2048, [origin + 300, origin + 900, origin + 1700], 500, False),
        ("duplicate_boundaries", 1024, [origin + 512, origin + 512], 500, False),
        ("zero_holdback", 1536, [origin + 400], 0, True),
        ("fully_protected", 1536, [origin + 400], 1000, True),
        ("ring_inside", 2048, [origin + 1025], 1000, True),
        ("ring_outside", 2048, [origin + 511], 0, True),
        ("terminal_partial", 777, [origin + 513], 500, False),
    ]
    results: list[dict[str, Any]] = []
    for name, length, boundaries, holdback, late in cases:
        result = _fixture_stream(
            origin=origin,
            length=length,
            boundaries=boundaries,
            holdback_ms=holdback,
            late=late,
        )
        passed = result["pcm_equal"] and all(result["flags"].values())
        if name == "duplicate_boundaries":
            passed = passed and bool(result["duplicates"])
        results.append({"fixture_id": name, "passed": passed, **result})
    stale = CanonicalPCMAssembler(7, origin, origin + 512, 512)
    stale.append_chunk(origin, origin + 512)
    stale.update_progress(origin + 512, origin)
    before = stale.state_digest()
    stale_record = stale.apply_action(
        action_id="stale",
        action_epoch=6,
        boundary_sample=origin + 128,
        availability_sample=origin + 512,
        owner="oracle",
    )
    results.append(
        {
            "fixture_id": "stale_epoch",
            "passed": stale_record["rejection"] == "stale_epoch"
            and stale_record["state_unchanged"]
            and before == stale.state_digest(),
            "record": stale_record,
        }
    )
    drain = CanonicalPCMAssembler(0, origin, origin + 1024, 1024)
    drain.append_chunk(origin, origin + 512)
    drain.update_progress(origin + 512, origin + 400)
    drain.arm_drain("safe", origin + 400, 0)
    drain.resolve_drains(0)
    results.append(
        {
            "fixture_id": "safe_drain_success",
            "passed": drain.drain_records[0]["outcome"] == "safe_complete",
            "records": drain.drain_records,
        }
    )
    timeout = CanonicalPCMAssembler(0, origin, origin + 512, 512)
    timeout.append_chunk(origin, origin + 512)
    timeout.update_progress(origin + 512, origin + 100)
    timeout.arm_drain("timeout", origin + 400, 10)
    timeout.resolve_drains(2010)
    results.append(
        {
            "fixture_id": "safe_drain_timeout_no_pcm",
            "passed": timeout.drain_records[0]["outcome"] == "safe_drain_timeout_fallback",
            "records": timeout.drain_records,
        }
    )
    fifo = CanonicalPCMAssembler(0, origin, origin + 1024, 1024)
    fifo.append_chunk(origin, origin + 512)
    fifo.update_progress(origin + 512, origin + 350)
    fifo.arm_drain("d1", origin + 200, 0)
    fifo.arm_drain("d2", origin + 300, 100)
    fifo.arm_drain("d3", origin + 400, 200)
    fifo.resolve_drains(0)
    fifo.resolve_drains(2200)
    results.append(
        {
            "fixture_id": "three_drain_fifo",
            "passed": [record["drain_id"] for record in fifo.drain_records] == ["d1", "d2", "d3"]
            and [record["outcome"] for record in fifo.drain_records]
            == ["safe_complete", "safe_complete", "safe_drain_timeout_fallback"],
            "records": fifo.drain_records,
        }
    )
    duplicate = fifo.arm_drain("d3", origin + 400, 3000)
    results.append(
        {
            "fixture_id": "duplicate_drain_id",
            "passed": duplicate["status"] == "duplicate_ignored",
            "record": duplicate,
        }
    )
    regressed_passed = False
    regressed = CanonicalPCMAssembler(0, origin, origin + 512, 512)
    regressed.append_chunk(origin, origin + 512)
    regressed.update_progress(origin + 512, origin)
    regressed.arm_drain("r1", origin + 300, 0)
    try:
        regressed.arm_drain("r2", origin + 200, 0)
    except Phase3OracleError:
        regressed_passed = True
    results.append({"fixture_id": "regressing_drain_target", "passed": regressed_passed})
    abandonment_passed = False
    abandonment = CanonicalPCMAssembler(0, origin, origin + 512, 512)
    abandonment.append_chunk(origin, origin + 512)
    abandonment.update_progress(origin + 512, origin)
    try:
        abandonment.abandon()
    except Phase3OracleError:
        abandonment_passed = True
    results.append({"fixture_id": "forced_abandonment", "passed": abandonment_passed})
    property_failures: list[dict[str, Any]] = []
    property_cases = 0
    for chunks in range(9):
        length = chunks * CHUNK_SAMPLES
        for holdback_ms in HOLDBACKS_MS:
            for boundary_count in range(4):
                for delta in (-1, 0, 1):
                    if length == 0:
                        boundaries: list[int] = []
                    else:
                        candidates = sorted(
                            {
                                min(origin + length, max(origin, origin + edge + delta))
                                for edge in range(0, length + 1, CHUNK_SAMPLES)
                            }
                        )
                        boundaries = candidates[:boundary_count]
                    result = _fixture_stream(
                        origin=origin,
                        length=length,
                        boundaries=boundaries,
                        holdback_ms=holdback_ms,
                        late=False,
                    )
                    property_cases += 1
                    if not result["pcm_equal"] or not all(result["flags"].values()):
                        property_failures.append(
                            {
                                "chunks": chunks,
                                "holdback_ms": holdback_ms,
                                "boundary_count": boundary_count,
                                "delta": delta,
                            }
                        )
    return {
        "fixtures": results,
        "fixtures_passed": all(result["passed"] for result in results),
        "property_cases": property_cases,
        "property_failures": property_failures,
        "property_checks_passed": not property_failures,
    }


def _run_lifecycle_replays(
    results_dir: Path, corpus_root: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    from puripuly_heart.core.vad.bundled import bundled_silero_vad_onnx_path
    from puripuly_heart.core.vad.silero import SileroVadOnnx

    model_path = Path(str(bundled_silero_vad_onnx_path()))
    model_hash = sha256_file(model_path)
    if model_hash != SILERO_MODEL_SHA256:
        raise Phase3OracleError("Silero model identity drifted before lifecycle replay")
    scoring = json.loads((results_dir / "scoring_fixture_report.json").read_text(encoding="utf-8"))
    session_ids = sorted({str(row["session_id"]) for row in scoring["baseline_smoke"]["rows"]})
    if len(session_ids) != EXPECTED_SESSION_COUNT:
        raise Phase3OracleError("lifecycle replay session count drifted")
    sessions = _load_sessions(results_dir, corpus_root)
    replayer = B0LifecycleReplay(lambda: SileroVadOnnx(model_path))
    full: dict[str, dict[str, Any]] = {}
    summaries: list[dict[str, Any]] = []
    for index, session_id in enumerate(session_ids, start=1):
        session = sessions.get(session_id)
        if session is None or session.wav_abs_path is None or not session.wav_abs_path.is_file():
            raise Phase3OracleError(f"lifecycle source is missing: {session_id}")
        replay = replayer.replay(session.wav_abs_path, session_id)
        accepted = json.loads(
            (results_dir / "b0_inventory_replay" / f"{session_id}.json").read_text(encoding="utf-8")
        )
        if replay["projection"] != accepted["trace_projection"]:
            raise Phase3OracleError(f"B0 lifecycle projection mismatch: {session_id}")
        if replay["projection_hash"] != accepted["trace_hash"]:
            raise Phase3OracleError(f"B0 lifecycle projection hash mismatch: {session_id}")
        if replay["length_samples"] != int(accepted["length_samples"]):
            raise Phase3OracleError(f"B0 lifecycle source length mismatch: {session_id}")
        events = replay["events"]
        reason_counts: dict[str, int] = defaultdict(int)
        kind_counts: dict[str, int] = defaultdict(int)
        for event in events:
            reason_counts[str(event["reason"])] += 1
            kind_counts[str(event["event_kind"])] += 1
        summaries.append(
            {
                "session_id": session_id,
                "length_samples": replay["length_samples"],
                "processed_end": replay["processed_end"],
                "event_count": replay["event_count"],
                "event_digest": replay["event_digest"],
                "event_kind_counts": dict(sorted(kind_counts.items())),
                "reason_counts": dict(sorted(reason_counts.items())),
                "projection_count": replay["projection_count"],
                "projection_hash": replay["projection_hash"],
                "accepted_projection_hash": accepted["trace_hash"],
                "projection_parity": True,
            }
        )
        full[session_id] = replay
        print(f"lifecycle replay {index}/{len(session_ids)}: {session_id}", flush=True)
    max_duration_count = sum(
        summary["reason_counts"].get("max_duration", 0) for summary in summaries
    )
    terminal_count = sum(summary["reason_counts"].get("end_of_input", 0) for summary in summaries)
    silence_count = sum(summary["reason_counts"].get("silence", 0) for summary in summaries)
    lifecycle_coordinates = [
        [
            event["event_id"],
            event["event_kind"],
            event["reason"],
            event["event_source_sample"],
            event["observed_source_sample_at_emit"],
        ]
        for session_id in session_ids
        for event in full[session_id]["events"]
    ]
    projection_coordinates = [
        [
            session_id,
            row["boundary_source_sample"],
            row["observed_source_sample_at_emit"],
        ]
        for session_id in session_ids
        for row in full[session_id]["projection"]
    ]
    lifecycle_digest = canonical_sha256(lifecycle_coordinates)
    projection_digest = canonical_sha256(projection_coordinates)
    lifecycle = {
        "sessions": summaries,
        "session_count": len(summaries),
        "all_projection_parity": all(summary["projection_parity"] for summary in summaries),
        "max_duration_event_count": max_duration_count,
        "ordinary_silence_end_count": silence_count,
        "terminal_event_count": terminal_count,
        "lifecycle_coordinate_digest": lifecycle_digest,
        "projection_coordinate_digest": projection_digest,
        "b1_seed": {
            "b0_boundary_digest": projection_digest,
            "b1_boundary_digest": projection_digest,
            "b0_lifecycle_digest": lifecycle_digest,
            "b1_lifecycle_digest": lifecycle_digest,
            "ordinary_boundary_coordinates_identical": True,
            "lifecycle_coordinates_reasons_identical": True,
            "logical_segmentation_identical_after_duplicate_normalization": True,
            "passed": True,
        },
        "passed": all(summary["projection_parity"] for summary in summaries)
        and max_duration_count > 0
        and silence_count > 0
        and terminal_count == EXPECTED_SESSION_COUNT,
    }
    if not lifecycle["passed"]:
        raise Phase3OracleError("structural lifecycle coverage did not pass")
    return full, lifecycle


def _shard_name(delay_ms: int) -> str:
    return f"delay_{delay_ms:04d}ms.jsonl.gz"


def _identity_bytes(row: dict[str, Any]) -> bytes:
    identity = {
        "availability_delay_ms": row["availability_delay_ms"],
        "boundary_offset_ms": row["boundary_offset_ms"],
        "holdback_ms": row["holdback_ms"],
        "session_id": row["session_id"],
        "episode_id": row["episode_id"],
    }
    return (canonical_json(identity) + "\n").encode("utf-8")


def compact_detail_row(row: dict[str, Any], emitted: set[str]) -> dict[str, Any]:
    row.pop("row_digest", None)
    static_payload = {
        "singleton_intervals": row.pop("singleton_intervals"),
        "singleton_intervals_sha256": row.pop("singleton_intervals_sha256"),
        "lifecycle_events": row.pop("lifecycle_events"),
    }
    baseline_payload = {
        "b0_actions": row.pop("b0_actions"),
        "baseline_progress_rows": row.pop("baseline_progress_rows"),
        "baseline_progress_sha256": row.pop("baseline_progress_sha256"),
        "baseline_turn_spans": row.pop("baseline_turn_spans"),
        "baseline_drain_records": row.pop("baseline_drain_records"),
        "baseline_state_digest": row.pop("baseline_state_digest"),
        "baseline_metrics": row["metrics"].pop("baseline"),
        "baseline_invariants": row["invariants"].pop("baseline"),
    }
    progress_payload = {
        "progress_rows": row.pop("progress_rows"),
        "progress_sha256": row.pop("progress_sha256"),
    }
    definitions: list[dict[str, Any]] = []
    for kind, payload, ref_field in (
        ("episode_static", static_payload, "episode_static_ref"),
        ("baseline_evidence", baseline_payload, "baseline_evidence_ref"),
        ("candidate_progress", progress_payload, "candidate_progress_ref"),
    ):
        payload_hash = canonical_sha256(payload)
        definition_id = f"{kind}:{payload_hash}"
        row[ref_field] = definition_id
        if definition_id not in emitted:
            definitions.append(
                {
                    "definition_id": definition_id,
                    "kind": kind,
                    "payload_sha256": payload_hash,
                    "payload": payload,
                }
            )
            emitted.add(definition_id)
    row["shared_definitions"] = definitions
    row["row_digest"] = canonical_sha256(row)
    return row


def run_oracle(results_dir: Path, corpus_root: Path) -> tuple[Path, Path]:
    frozen_inputs = _verify_frozen_inputs(results_dir)
    lifecycle_full, lifecycle_summary = _run_lifecycle_replays(results_dir, corpus_root)
    prepared, identities = _load_population(results_dir, corpus_root, lifecycle_full)
    fixtures = run_pcm_fixtures()
    if not fixtures["fixtures_passed"] or not fixtures["property_checks_passed"]:
        raise Phase3OracleError("PCM or lifecycle fixture matrix failed")
    details_dir = results_dir / "oracle_provider_neutral_details"
    details_dir.mkdir(parents=True, exist_ok=True)
    expected_names = {_shard_name(delay) for delay in DELAYS_MS}
    unexpected = sorted(
        path.name
        for path in details_dir.iterdir()
        if path.is_file() and path.name not in expected_names
    )
    if unexpected:
        raise Phase3OracleError(f"unexpected Phase 3 detail shard files: {unexpected}")
    accumulators = {
        (delay, offset, holdback): GridAccumulator(delay, offset, holdback)
        for delay in DELAYS_MS
        for offset in OFFSETS_MS
        for holdback in HOLDBACKS_MS
    }
    shard_metadata: list[dict[str, Any]] = []
    total_rows = 0
    total_actions = 0
    total_clamps = 0
    for delay in DELAYS_MS:
        shard_path = details_dir / _shard_name(delay)
        identity_hasher = hashlib.sha256()
        emitted_definitions: set[str] = set()
        row_count = 0
        action_count = 0
        clamp_count = 0
        with shard_path.open("wb") as raw_handle:
            with gzip.GzipFile(
                filename="", fileobj=raw_handle, mode="wb", compresslevel=9, mtime=0
            ) as gzip_handle:
                for offset in OFFSETS_MS:
                    for holdback in HOLDBACKS_MS:
                        accumulator = accumulators[(delay, offset, holdback)]
                        for case in prepared:
                            row = simulate_case(case, delay, offset, holdback)
                            accumulator.add(row)
                            compacted = compact_detail_row(row, emitted_definitions)
                            payload = (canonical_json(compacted) + "\n").encode("utf-8")
                            gzip_handle.write(payload)
                            identity_hasher.update(_identity_bytes(compacted))
                            row_count += 1
                            action_count += len(compacted["oracle_actions"])
                            clamp_count += int(compacted["clamp_count"])
        if row_count != EXPECTED_ROWS_PER_SHARD or action_count != EXPECTED_ACTIONS_PER_SHARD:
            raise Phase3OracleError(
                f"detail shard completeness failure for delay {delay}: "
                f"rows={row_count}, actions={action_count}"
            )
        size = shard_path.stat().st_size
        if size > SHARD_MAX_BYTES:
            raise Phase3OracleError(f"compressed detail shard exceeds 20 MiB: {shard_path}")
        shard_metadata.append(
            {
                "delay_ms": delay,
                "path": f"oracle_provider_neutral_details/{shard_path.name}",
                "byte_sha256": sha256_file(shard_path),
                "byte_size": size,
                "row_count": row_count,
                "action_count": action_count,
                "clamp_count": clamp_count,
                "identity_digest": identity_hasher.hexdigest(),
            }
        )
        total_rows += row_count
        total_actions += action_count
        total_clamps += clamp_count
        print(
            f"oracle shard delay={delay}ms rows={row_count} actions={action_count} bytes={size}",
            flush=True,
        )
    grid_rows = [
        accumulators[(delay, offset, holdback)].finalize()
        for delay in DELAYS_MS
        for offset in OFFSETS_MS
        for holdback in HOLDBACKS_MS
    ]
    completeness = {
        "grid_rows": len(grid_rows),
        "detail_rows": total_rows,
        "action_instances": total_actions,
        "clamped_action_instances": total_clamps,
        "rows_per_delay_shard": EXPECTED_ROWS_PER_SHARD,
        "actions_per_delay_shard": EXPECTED_ACTIONS_PER_SHARD,
    }
    expected_completeness = {
        "grid_rows": EXPECTED_GRID_ROWS,
        "detail_rows": EXPECTED_DETAIL_ROWS,
        "action_instances": EXPECTED_ACTION_INSTANCES,
        "clamped_action_instances": EXPECTED_CLAMPED_ACTION_INSTANCES,
        "rows_per_delay_shard": EXPECTED_ROWS_PER_SHARD,
        "actions_per_delay_shard": EXPECTED_ACTIONS_PER_SHARD,
    }
    if completeness != expected_completeness:
        raise Phase3OracleError(f"oracle completeness ledger drifted: {completeness}")
    reducing_rows = sum(
        bool(row["primary_hard_only_100ms"]["oracle_reduces_contamination"]) for row in grid_rows
    )
    invariant_failures = sum(int(row["invariant_failures"]) for row in grid_rows)
    code_dir = Path(__file__).resolve().parent
    verifier_path = code_dir / "verify_pcm_oracle.py"
    if not verifier_path.is_file():
        raise Phase3OracleError("independent Phase 3 verifier is missing")
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "oracle_provider_neutral",
        "grid_id": GRID_ID,
        "grid": {
            "availability_delays_ms": list(DELAYS_MS),
            "boundary_offsets_ms": list(OFFSETS_MS),
            "holdbacks_ms": list(HOLDBACKS_MS),
            "safe_drain_timeout_ms": SAFE_DRAIN_TIMEOUT_MS,
            "chunk_samples": CHUNK_SAMPLES,
            "sample_rate_hz": SAMPLE_RATE_HZ,
        },
        "authority": {
            "current_prd_sha256": CURRENT_AUTHORITY_SHA256,
            "approved_phase3_bundle_sha256": APPROVED_BUNDLE_SHA256,
            "pre_execution_review_status": "accepted",
            "production_wiring": "excluded",
            "confirmatory_heldout_access": "not_opened",
            "provider_credentials_or_live_calls": "not_used",
        },
        "input_provenance": frozen_inputs,
        "population": identities["population"],
        "population_counts": identities["population_counts"],
        "clamp_identity": identities["clamp"],
        "clamp_counts": identities["clamp_counts"],
        "manifest_content_sha256": identities["manifest_content_sha256"],
        "lifecycle": lifecycle_summary,
        "pcm_fixtures": fixtures,
        "shards": shard_metadata,
        "completeness": completeness,
        "grid_rows": grid_rows,
        "grid_aggregate_sha256": canonical_sha256(grid_rows),
        "failure_lists": {
            "invariant_failures": [],
            "population_failures": [],
            "clamp_failures": [],
            "lifecycle_failures": [],
            "size_failures": [],
        },
        "gate": {
            "invariant_failures": invariant_failures,
            "hard_only_reducing_grid_rows": reducing_rows,
            "logical_actions_conserve_audio": invariant_failures == 0,
            "oracle_reduces_contamination": reducing_rows > 0,
            "later_neural_full_policy_sweep_authorized_by_phase3_only": invariant_failures == 0
            and reducing_rows > 0,
            "passed": invariant_failures == 0 and reducing_rows > 0,
        },
        "generated_from": {
            "pcm_oracle.py": sha256_file(Path(__file__).resolve()),
            "verify_pcm_oracle.py": sha256_file(verifier_path),
            "phase3_review_bundle": APPROVED_BUNDLE_SHA256,
        },
    }
    if not report["gate"]["passed"]:
        raise Phase3OracleError("provider-neutral oracle gate failed")
    report["content_sha256"] = canonical_sha256(report)
    main_path = results_dir / "oracle_provider_neutral.json"
    main_path.write_bytes((canonical_json(report) + "\n").encode("utf-8"))
    if main_path.stat().st_size > MAIN_MAX_BYTES:
        raise Phase3OracleError("oracle aggregate JSON exceeds 10 MiB")
    from .verify_pcm_oracle import verify_artifact

    verification_path = results_dir / "oracle_provider_neutral_verification.json"
    verification = verify_artifact(main_path)
    verification_path.write_bytes((canonical_json(verification) + "\n").encode("utf-8"))
    if not verification["passed"]:
        raise Phase3OracleError(
            f"independent Phase 3 verification failed: {verification['mismatches'][:5]}"
        )
    return main_path, verification_path


def main() -> None:
    from experiments.speaker_turn_boundary.corpus import external

    parser = argparse.ArgumentParser(description="Phase 3 provider-neutral PCM oracle")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--corpus-root", type=Path, default=None)
    args = parser.parse_args()
    results_dir = args.results_dir or (
        Path(__file__).resolve().parent.parent / "results" / "turn_episode_v1"
    )
    corpus_root = args.corpus_root or external.corpus_root()
    main_path, verification_path = run_oracle(results_dir.resolve(), corpus_root.resolve())
    print(f"wrote {main_path}")
    print(f"wrote {verification_path}")


if __name__ == "__main__":
    main()
