from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from experiments.speaker_representation_scd.provenance import (
    canonical_json_bytes,
    sha256_bytes,
)

SAMPLE_RATE_HZ = 16000
EXPECTED_FIXTURE_SHA256 = {
    "silence": "ea0787f65f73b0013d03b359490e3125211b28ad5c1502ffb1544c0ded4192f5",
    "one_speaker": "03b6b0a0ef06456ad1dfbbb0327806dab9a5275d8020a1e47c87c4bd67791c2c",
    "clean_a_to_b": "f5477f1f1a376c9acb8baf68c5a83f602761ef71c144afb0affa733bdb54e417",
    "gap_a_to_b": "9d8a8a6dde37cdb4a5d77c279edcce0ff51e5639e761eaf3735264c73e45c6f6",
    "overlap_a_to_b": "5bc04b984b581ab7318cc8eacd8e8c7c33fc8954cd97cc51457070de31fd9b9c",
    "backchannel_b_to_a": "3abc6cf37534f2d5b4366e9f89041450cf497f27026f514d85672d1264f27228",
    "gain_step_same_speaker": "d3020d09f3b334af20cce7737cd6e78e09968048c234befb4375278d54bb4998",
    "noise_step_same_speaker": "9853c6ecdc81a892467c41fc5ccfa8aa0b22847d770758d8814fc6487a6a9c02",
    "impulse_coordinates": "5ef70298483210f57beeef78c9decfd4eaf29975bb8409b28919379c35011306",
    "channel_chirp_same_speaker": "2d5e98197ce69b9d0fdbf46f833ce636f4a6c8576fe5f30fd08046bb5a149b2e",
}
EXPECTED_FIXTURE_MANIFEST_SHA256 = (
    "5a82813fd5f1b8b40ff1f7ccc4e16fd7cdd05afee6f7df07a6e165e63dedee53"
)


@dataclass(frozen=True, slots=True)
class D0Fixture:
    fixture_id: str
    scenario_kind: str
    waveform: np.ndarray
    frontier_sample: int
    window_samples: int
    waveform_sha256: str
    speaker_segments: tuple[tuple[str, int, int], ...]
    event_samples: tuple[int, ...]
    nuisance_reference: np.ndarray | None


def _tone(samples: int, frequency_hz: float, amplitude: float) -> np.ndarray:
    return np.fromiter(
        (
            amplitude * math.sin(2 * math.pi * frequency_hz * index / SAMPLE_RATE_HZ)
            for index in range(samples)
        ),
        dtype=np.float32,
        count=samples,
    )


def _chirp(samples: int, start_hz: float, end_hz: float, amplitude: float) -> np.ndarray:
    duration = samples / SAMPLE_RATE_HZ
    slope = (end_hz - start_hz) / duration
    return np.fromiter(
        (
            amplitude
            * math.sin(
                2
                * math.pi
                * (start_hz * index / SAMPLE_RATE_HZ + 0.5 * slope * (index / SAMPLE_RATE_HZ) ** 2)
            )
            for index in range(samples)
        ),
        dtype=np.float32,
        count=samples,
    )


def _noise(samples: int, seed: int, amplitude: float) -> np.ndarray:
    state = seed & 0xFFFFFFFF

    def values():
        nonlocal state
        for _ in range(samples):
            state = (1664525 * state + 1013904223) & 0xFFFFFFFF
            yield amplitude * ((state / 4294967296.0) * 2 - 1)

    return np.fromiter(values(), dtype=np.float32, count=samples)


def _channel_transform(waveform: np.ndarray) -> np.ndarray:
    transformed = waveform * np.float32(0.72)
    transformed[1:] += waveform[:-1] * np.float32(0.21)
    transformed[2:] -= waveform[:-2] * np.float32(0.08)
    return np.ascontiguousarray(transformed, dtype=np.float32)


def _fixture_sources() -> list[dict[str, object]]:
    samples = 3 * SAMPLE_RATE_HZ
    boundary = 24000
    a = _tone(samples, 180, 0.08)
    b = _tone(samples, 440, 0.08)
    clean = a.copy()
    clean[boundary:] = b[boundary:]
    gap = a.copy()
    gap[22400:25600] = 0
    gap[25600:] = b[25600:]
    overlap = a.copy()
    overlap[22400:27200] += b[22400:27200]
    overlap[27200:] = b[27200:]
    backchannel = a.copy()
    backchannel[22400:25600] += b[22400:25600]
    gain = _tone(samples, 190, 0.03)
    gain[boundary:] *= 4
    noise_step = a.copy()
    noise_step[boundary:] += _noise(samples - boundary, 20260810, 0.04)
    channel_reference = _chirp(samples, 100, 900, 0.1)
    channel_filtered = _channel_transform(channel_reference)
    channel_step = channel_reference.copy()
    channel_step[boundary:] = channel_filtered[boundary:]
    impulse = np.zeros(samples, dtype=np.float32)
    for coordinate, amplitude in ((16000, 0.2), (23999, -0.35), (31999, 0.5)):
        impulse[coordinate] = amplitude
    return [
        {
            "fixture_id": "silence",
            "scenario_kind": "silence",
            "waveform": np.zeros(samples, dtype=np.float32),
            "window_ms": 100,
            "frontier_sample": 32000,
            "segments": (),
            "events": (),
            "nuisance_reference": None,
        },
        {
            "fixture_id": "one_speaker",
            "scenario_kind": "one_speaker",
            "waveform": a,
            "window_ms": 200,
            "frontier_sample": 32000,
            "segments": (("A", 0, samples),),
            "events": (),
            "nuisance_reference": None,
        },
        {
            "fixture_id": "clean_a_to_b",
            "scenario_kind": "clean_a_to_b",
            "waveform": clean,
            "window_ms": 300,
            "frontier_sample": 26400,
            "segments": (("A", 0, boundary), ("B", boundary, samples)),
            "events": (boundary,),
            "nuisance_reference": None,
        },
        {
            "fixture_id": "gap_a_to_b",
            "scenario_kind": "gap_a_to_b",
            "waveform": gap,
            "window_ms": 500,
            "frontier_sample": 27200,
            "segments": (("A", 0, 22400), ("B", 25600, samples)),
            "events": (25600,),
            "nuisance_reference": None,
        },
        {
            "fixture_id": "overlap_a_to_b",
            "scenario_kind": "overlap_a_to_b",
            "waveform": overlap,
            "window_ms": 750,
            "frontier_sample": 32000,
            "segments": (("A", 0, 27200), ("B", 22400, samples)),
            "events": (22400, 27200),
            "nuisance_reference": None,
        },
        {
            "fixture_id": "backchannel_b_to_a",
            "scenario_kind": "backchannel_b_to_a",
            "waveform": backchannel,
            "window_ms": 1000,
            "frontier_sample": 32000,
            "segments": (("A", 0, samples), ("B", 22400, 25600)),
            "events": (22400, 25600),
            "nuisance_reference": None,
        },
        {
            "fixture_id": "gain_step_same_speaker",
            "scenario_kind": "gain_step_same_speaker",
            "waveform": gain,
            "window_ms": 500,
            "frontier_sample": 28000,
            "segments": (("A", 0, samples),),
            "events": (boundary,),
            "nuisance_reference": _tone(samples, 190, 0.03),
        },
        {
            "fixture_id": "noise_step_same_speaker",
            "scenario_kind": "noise_step_same_speaker",
            "waveform": noise_step,
            "window_ms": 750,
            "frontier_sample": 32000,
            "segments": (("A", 0, samples),),
            "events": (boundary,),
            "nuisance_reference": a,
        },
        {
            "fixture_id": "impulse_coordinates",
            "scenario_kind": "timestamp_impulses",
            "waveform": impulse,
            "window_ms": 1000,
            "frontier_sample": 32000,
            "segments": (),
            "events": (16000, 23999, 31999),
            "nuisance_reference": None,
        },
        {
            "fixture_id": "channel_chirp_same_speaker",
            "scenario_kind": "channel_chirp_same_speaker",
            "waveform": channel_step,
            "window_ms": 300,
            "frontier_sample": 26400,
            "segments": (("A", 0, samples),),
            "events": (boundary,),
            "nuisance_reference": channel_reference,
        },
    ]


def _active_speakers(fixture: D0Fixture, sample: int) -> set[str]:
    return {speaker for speaker, start, end in fixture.speaker_segments if start <= sample < end}


def fixture_window_contract(fixture: D0Fixture) -> dict[str, object]:
    start = fixture.frontier_sample - fixture.window_samples
    end = fixture.frontier_sample
    errors: list[str] = []
    inside = tuple(event for event in fixture.event_samples if start < event < end)
    scenario = fixture.scenario_kind
    if start < 0 or end > fixture.waveform.shape[0]:
        errors.append("trailing window is outside the waveform")
    if scenario == "silence" and _active_speakers(fixture, start):
        errors.append("silence window has an active speaker")
    if scenario == "one_speaker" and _active_speakers(fixture, start) != {"A"}:
        errors.append("one-speaker window does not contain A")
    if scenario == "clean_a_to_b":
        event = fixture.event_samples[0]
        if not start < event < end:
            errors.append("clean change is not strictly inside the window")
        if _active_speakers(fixture, event - 1) != {"A"}:
            errors.append("clean change lacks A immediately before onset")
        if _active_speakers(fixture, event) != {"B"}:
            errors.append("clean change lacks B at onset")
    if scenario == "gap_a_to_b":
        event = fixture.event_samples[0]
        a_end = max(
            segment_end for speaker, _, segment_end in fixture.speaker_segments if speaker == "A"
        )
        if not start < a_end < event < end:
            errors.append("gap window does not contain A, silence, and B")
        if _active_speakers(fixture, a_end - 1) != {"A"}:
            errors.append("gap lacks A before silence")
        if _active_speakers(fixture, a_end):
            errors.append("gap interval is not silent")
        if _active_speakers(fixture, event) != {"B"}:
            errors.append("gap lacks B at onset")
    if scenario == "overlap_a_to_b":
        onset, exclusive = fixture.event_samples
        if not start < onset < exclusive < end:
            errors.append("overlap window lacks all three regions")
        if _active_speakers(fixture, onset - 1) != {"A"}:
            errors.append("overlap lacks A-only lead-in")
        if _active_speakers(fixture, onset) != {"A", "B"}:
            errors.append("overlap onset lacks A+B")
        if _active_speakers(fixture, exclusive) != {"B"}:
            errors.append("overlap lacks B-only tail")
    if scenario == "backchannel_b_to_a":
        onset, offset = fixture.event_samples
        if not start < onset < offset < end:
            errors.append("backchannel window lacks lead-in, B, and return")
        if _active_speakers(fixture, onset - 1) != {"A"}:
            errors.append("backchannel lacks A-only lead-in")
        if _active_speakers(fixture, onset) != {"A", "B"}:
            errors.append("backchannel lacks B activity")
        if _active_speakers(fixture, offset) != {"A"}:
            errors.append("backchannel lacks A-only return")
    if scenario in {
        "gain_step_same_speaker",
        "noise_step_same_speaker",
        "channel_chirp_same_speaker",
    }:
        event = fixture.event_samples[0]
        reference = fixture.nuisance_reference
        if not start < event < end:
            errors.append("nuisance event is not strictly inside the window")
        if _active_speakers(fixture, event - 1) != {"A"} or _active_speakers(fixture, event) != {
            "A"
        }:
            errors.append("nuisance transition changes speaker identity")
        if reference is None:
            errors.append("nuisance fixture lacks a counterfactual reference")
        elif not np.array_equal(fixture.waveform[start:event], reference[start:event]):
            errors.append("nuisance transform is present before its event")
        elif np.array_equal(fixture.waveform[event:end], reference[event:end]):
            errors.append("nuisance transform is absent after its event")
    return {
        "window_start_sample": start,
        "window_end_sample": end,
        "strictly_inside_event_samples": list(inside),
        "nuisance_reference_sha256": (
            sha256_bytes(np.ascontiguousarray(fixture.nuisance_reference).tobytes())
            if fixture.nuisance_reference is not None
            else None
        ),
        "errors": errors,
        "passed": not errors,
    }


def fixture_manifest(fixtures: tuple[D0Fixture, ...]) -> list[dict[str, object]]:
    return [
        {
            "fixture_id": fixture.fixture_id,
            "scenario_kind": fixture.scenario_kind,
            "waveform_sha256": fixture.waveform_sha256,
            "waveform_samples": int(fixture.waveform.shape[0]),
            "frontier_sample": fixture.frontier_sample,
            "window_samples": fixture.window_samples,
            "speaker_segments": [list(row) for row in fixture.speaker_segments],
            "event_samples": list(fixture.event_samples),
            "scenario_window_contract": fixture_window_contract(fixture),
        }
        for fixture in fixtures
    ]


def d0_fixtures() -> tuple[D0Fixture, ...]:
    fixtures: list[D0Fixture] = []
    for source in _fixture_sources():
        fixture_id = str(source["fixture_id"])
        waveform = np.ascontiguousarray(source["waveform"], dtype=np.float32)
        waveform_sha256 = sha256_bytes(waveform.tobytes())
        if waveform_sha256 != EXPECTED_FIXTURE_SHA256[fixture_id]:
            raise RuntimeError(f"D0 fixture identity changed: {fixture_id}")
        fixtures.append(
            D0Fixture(
                fixture_id=fixture_id,
                scenario_kind=str(source["scenario_kind"]),
                waveform=waveform,
                frontier_sample=int(source["frontier_sample"]),
                window_samples=int(source["window_ms"]) * 16,
                waveform_sha256=waveform_sha256,
                speaker_segments=tuple(source["segments"]),
                event_samples=tuple(source["events"]),
                nuisance_reference=(
                    np.ascontiguousarray(source["nuisance_reference"], dtype=np.float32)
                    if source["nuisance_reference"] is not None
                    else None
                ),
            )
        )
        contract = fixture_window_contract(fixtures[-1])
        if not contract["passed"]:
            raise RuntimeError(f"D0 fixture scenario contract failed: {fixture_id}")
    result = tuple(fixtures)
    manifest_sha256 = sha256_bytes(canonical_json_bytes(fixture_manifest(result)))
    if manifest_sha256 != EXPECTED_FIXTURE_MANIFEST_SHA256:
        raise RuntimeError("D0 fixture manifest identity changed")
    return result


def mutate_future(fixture: D0Fixture) -> np.ndarray:
    changed = fixture.waveform.copy()
    changed[fixture.frontier_sample :] = _noise(
        changed.shape[0] - fixture.frontier_sample,
        1_000_000 + fixture.window_samples,
        0.2,
    )
    return changed
