from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import wave
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent
DEFAULT_PROFILE = ROOT / "profile.json"
DEFAULT_MANIFEST = ROOT / "manifest.json"
LIVE_AUTHORIZATION = "ISSUE-157"
REQUIRED_COVERAGE = {
    "continuation_over_6s",
    "aba",
    "abc",
    "brief_response",
    "laughter",
    "silence",
    "pause_boundary",
    "hard_boundary",
    "overlap",
    "interruption",
    "same_speaker_continuation",
    "voice_chat_codec_noise",
    "similar_voices",
}


class PreparationBlocked(RuntimeError):
    pass


@dataclass(frozen=True)
class Inputs:
    profile: dict[str, Any]
    manifest: dict[str, Any]
    manifest_path: Path


@dataclass(frozen=True)
class Recording:
    raw: dict[str, Any]
    path: Path
    reference_path: Path
    frame_count: int


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top level must be an object")
    return value


def write_text_with_parents(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

def execution_identity(profile_path: Path, manifest_path: Path) -> dict[str, str]:
    repository = ROOT.parents[1]
    completed = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    revision = completed.stdout.strip()
    if len(revision) != 40:
        raise RuntimeError("git rev-parse HEAD did not return a full revision")
    return {
        "git_revision": revision,
        "replay_sha256": sha256_file(Path(__file__).resolve()),
        "profile_sha256": sha256_file(profile_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path.resolve()),
    }


def resolve_local_path(manifest_path: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty local path")
    path = Path(value)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def load_inputs(profile_path: Path, manifest_path: Path) -> Inputs:
    profile = read_json(profile_path)
    manifest = read_json(manifest_path)
    if profile.get("schema_version") != 1 or manifest.get("schema_version") != 1:
        raise ValueError("profile and manifest schema_version must both be 1")
    return Inputs(profile=profile, manifest=manifest, manifest_path=manifest_path.resolve())


def validate_profile(profile: dict[str, Any]) -> None:
    if profile.get("baseline_revision") != "9b95293b2041760401d82d80da4daf07c30f6fb9":
        raise ValueError("profile baseline_revision is not the authorized baseline")
    if profile.get("sample_rate_hz") != 16000:
        raise ValueError("the canonical source rate must be 16000 Hz")
    if profile.get("sample_width_bytes") != 2 or profile.get("channels") != 1:
        raise ValueError("the replay format must be mono PCM16LE")
    if profile.get("session_lifetime_s") != 300:
        raise ValueError("primary comparison sessions must be exactly five minutes")
    if profile.get("source_frame_samples") != 512:
        raise ValueError("the current 16 kHz VAD source frame must remain 512 samples")
    policy = profile.get("source_boundary_policy")
    if policy != {"step_age_ms": 4000, "pause_seal_ms": 224, "hard_seal_ms": 6000}:
        raise ValueError("source boundary policy differs from the frozen 4s/224ms/6s policy")
    if profile.get("boundary_schedule_scope") != "issue_157_fixed_4s_224ms_6s":
        raise ValueError("boundary schedule must remain scoped to the issue-authorized fixed schedule")
    excluded = profile.get("excluded_production_boundary_paths")
    if excluded != [
        "delivery_profile_off_vad_hangover",
        "smart_turn_pre_4s_completion_or_fallback",
    ]:
        raise ValueError("excluded early production boundary paths must remain explicit")
    if profile.get("production_scoped_engine_healthy_reset_age_s") != 180:
        raise ValueError("the disclosed production healthy session reset age must remain 180 seconds")
    if (
        profile.get("session_strategy")
        != "direct_experiment_websocket_300s_without_production_healthy_rotation"
    ):
        raise ValueError("the direct five-minute experiment session strategy must remain explicit")
    soniox = profile.get("soniox", {})
    if soniox.get("model") != "stt-rt-v5":
        raise ValueError("Soniox model must remain frozen at stt-rt-v5")
    if soniox.get("enable_endpoint_detection") is not False:
        raise ValueError("Soniox endpoint detection must remain disabled")
    if soniox.get("enable_speaker_diarization") is not True:
        raise ValueError("speaker diarization must be enabled")
    if soniox.get("final_receipt_timeout_s") != 20:
        raise ValueError("the scoped final receipt timeout must remain 20 seconds")
    arm_ids = [arm.get("id") for arm in profile.get("initial_arms", [])]
    expected = ["B0", "S200", "T200", "W200", "S200-paced", "T200-paced", "C"]
    if arm_ids != expected:
        raise ValueError(f"initial arms must be ordered exactly as {expected}")


def validate_recordings(inputs: Inputs, *, require_any: bool) -> tuple[list[Recording], set[str]]:
    profile = inputs.profile
    recordings_raw = inputs.manifest.get("recordings")
    if not isinstance(recordings_raw, list):
        raise ValueError("manifest recordings must be an array")
    if require_any and not recordings_raw:
        raise PreparationBlocked("no selected recording is present")
    sample_rate = int(profile["sample_rate_hz"])
    session_samples = sample_rate * int(profile["session_lifetime_s"])
    seen_ids: set[str] = set()
    seen_capture_epochs: set[int] = set()
    coverage: set[str] = set()
    recordings: list[Recording] = []
    for raw in recordings_raw:
        if not isinstance(raw, dict):
            raise ValueError("each recording must be an object")
        recording_id = raw.get("id")
        if not isinstance(recording_id, str) or not recording_id or recording_id in seen_ids:
            raise ValueError("recording ids must be non-empty and unique")
        seen_ids.add(recording_id)
        capture_epoch = require_int(raw, "capture_epoch", minimum=0)
        if raw.get("boundary_schedule_scope") != profile["boundary_schedule_scope"]:
            raise ValueError(
                f"{recording_id}: boundary_schedule_scope must identify the fixed experiment schedule"
            )
        if capture_epoch in seen_capture_epochs:
            raise ValueError("capture_epoch must be unique within the manifest")
        seen_capture_epochs.add(capture_epoch)
        path = resolve_local_path(inputs.manifest_path, raw.get("path"), f"{recording_id}.path")
        if not path.is_file():
            raise PreparationBlocked(f"{recording_id}: audio file is unavailable: {path}")
        expected_hash = raw.get("sha256")
        if not isinstance(expected_hash, str) or sha256_file(path) != expected_hash.lower():
            raise PreparationBlocked(f"{recording_id}: WAV sha256 is missing or does not match")
        with wave.open(str(path), "rb") as wav:
            if (
                wav.getframerate() != sample_rate
                or wav.getnchannels() != 1
                or wav.getsampwidth() != 2
                or wav.getcomptype() != "NONE"
            ):
                raise ValueError(f"{recording_id}: WAV must be uncompressed mono 16 kHz PCM16")
            frame_count = wav.getnframes()
        session_start = require_int(raw, "session_source_start_sample", minimum=0)
        session_end = require_int(raw, "session_source_end_sample", minimum=1)
        if session_end - session_start != session_samples:
            raise ValueError(f"{recording_id}: session source span must be exactly five minutes")
        if session_end > frame_count:
            raise ValueError(f"{recording_id}: session source span exceeds the WAV")
        reference = raw.get("human_reference")
        if not isinstance(reference, dict):
            raise PreparationBlocked(f"{recording_id}: human-checked reference metadata is absent")
        reference_path = resolve_local_path(
            inputs.manifest_path, reference.get("path"), f"{recording_id}.human_reference.path"
        )
        if not reference_path.is_file():
            raise PreparationBlocked(f"{recording_id}: human reference file is unavailable")
        reference_hash = reference.get("sha256")
        if not isinstance(reference_hash, str) or sha256_file(reference_path) != reference_hash.lower():
            raise PreparationBlocked(f"{recording_id}: human reference sha256 is missing or mismatched")
        if not reference.get("checked_by") or not reference.get("checked_utc_date"):
            raise PreparationBlocked(f"{recording_id}: human checker evidence is incomplete")
        if not raw.get("license_or_consent_basis"):
            raise PreparationBlocked(f"{recording_id}: license/consent basis is absent")
        tags = raw.get("coverage_tags")
        if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
            raise ValueError(f"{recording_id}: coverage_tags must be an array of strings")
        coverage.update(tags)
        validate_segments(
            raw,
            session_start,
            session_end,
            sample_rate,
            int(profile["source_frame_samples"]),
        )
        recordings.append(Recording(raw, path, reference_path, frame_count))
    return recordings, REQUIRED_COVERAGE - coverage


def require_int(value: dict[str, Any], key: str, *, minimum: int) -> int:
    result = value.get(key)
    if type(result) is not int or result < minimum:
        raise ValueError(f"{key} must be an integer >= {minimum}")
    return result


def validate_segments(
    raw: dict[str, Any],
    session_start: int,
    session_end: int,
    rate: int,
    frame_samples: int,
) -> None:
    segments = raw.get("segments")
    if not isinstance(segments, list) or not segments:
        raise PreparationBlocked(f"{raw.get('id')}: frozen source segments are absent")
    previous_end = session_start
    ids: set[str] = set()
    for segment in segments:
        if not isinstance(segment, dict):
            raise ValueError("segments must contain objects")
        segment_id = segment.get("id")
        if not isinstance(segment_id, str) or not segment_id or segment_id in ids:
            raise ValueError("segment ids must be non-empty and unique per recording")
        ids.add(segment_id)
        start = require_int(segment, "source_start_sample", minimum=session_start)
        end = require_int(segment, "source_end_sample", minimum=start + 1)
        if start < previous_end or end > session_end:
            raise ValueError(f"{segment_id}: content spans must be source ordered and non-overlapping")
        if end - start > 6 * rate:
            raise ValueError(f"{segment_id}: content exceeds the frozen six-second hard boundary")
        boundary_type = segment.get("boundary_type")
        if boundary_type not in {"pause_224ms", "hard_6s"}:
            raise ValueError(f"{segment_id}: boundary_type must be pause_224ms or hard_6s")
        if boundary_type == "pause_224ms" and end - start < 4 * rate:
            raise ValueError(f"{segment_id}: a pause seal cannot precede the four-second step age")
        if boundary_type == "hard_6s" and end - start != 6 * rate:
            raise ValueError(f"{segment_id}: a hard segment must contain exactly six seconds")
        trailing = require_int(segment, "transmitted_trailing_silence_samples", minimum=0)
        if frame_samples != 512:
            raise ValueError(f"{segment_id}: source frame must be the current 512 samples")
        if trailing % frame_samples:
            raise ValueError(f"{segment_id}: trailing silence must align to 512-sample VAD frames")
        pause_samples = 224 * rate // 1000
        if boundary_type == "pause_224ms" and trailing < pause_samples:
            raise ValueError(
                f"{segment_id}: pause_224ms requires at least {pause_samples} transmitted "
                "VAD-classified trailing-silence samples"
            )
        if boundary_type == "hard_6s" and trailing > pause_samples:
            raise ValueError(
                f"{segment_id}: a hard cut cannot retain more than {pause_samples} contiguous "
                "VAD-classified trailing-silence samples"
            )
        if trailing > end - start:
            raise ValueError(f"{segment_id}: trailing silence exceeds segment content")
        speech_end = segment.get("speech_end_source_sample")
        if speech_end is not None and (type(speech_end) is not int or not start <= speech_end <= end):
            raise ValueError(f"{segment_id}: invalid speech_end_source_sample")
        if not segment.get("vad_classification_note"):
            raise ValueError(f"{segment_id}: VAD classification limitations must be recorded")
        if "prefix_spans" not in segment:
            raise ValueError(f"{segment_id}: prefix_spans is required (use [] when empty)")
        prefixes = segment["prefix_spans"]
        if not isinstance(prefixes, list):
            raise ValueError(f"{segment_id}: prefix_spans must be an array")
        for prefix in prefixes:
            if (
                not isinstance(prefix, list)
                or len(prefix) != 2
                or any(type(point) is not int for point in prefix)
                or not session_start <= prefix[0] < prefix[1] <= start
            ):
                raise ValueError(f"{segment_id}: prefix spans must be real source spans ending by content start")
        previous_end = end


def approval_blockers(inputs: Inputs, recordings: list[Recording], estimated_usd: float) -> list[str]:
    approval = inputs.manifest.get("approval")
    blockers: list[str] = []
    if not isinstance(approval, dict):
        return ["manifest approval object is absent"]
    if approval.get("external_processing_approved") is not True:
        blockers.append("selected recordings are not approved for external processing")
    if not approval.get("approved_by") or not approval.get("approved_utc_date"):
        blockers.append("approval authority/date is absent")
    if approval.get("observer_stream_included") is not True:
        blockers.append("approval does not include the continuous observer stream")
    budget = approval.get("paid_budget_usd")
    if not isinstance(budget, (int, float)) or isinstance(budget, bool) or budget <= 0:
        blockers.append("explicit paid API budget is absent")
    elif float(budget) + 1e-12 < estimated_usd:
        blockers.append(f"paid budget ${budget:.4f} is below estimated realtime usage ${estimated_usd:.4f}")
    if not recordings:
        blockers.append("no selected recordings")
    return blockers


def arm_by_id(profile: dict[str, Any], arm_id: str) -> dict[str, Any]:
    for arm in profile["initial_arms"]:
        if arm["id"] == arm_id:
            return arm
    for arm in profile["conditional_arms"]:
        if arm["id"] == arm_id:
            if arm.get("opened") is not True:
                raise ValueError(f"conditional arm {arm_id} has not been opened by a useful 200 ms result")
            return arm
    raise ValueError(f"unknown arm: {arm_id}")




def estimated_realtime_cost(
    profile: dict[str, Any], recordings: list[Recording], arms: Iterable[str]
) -> float:
    rate = int(profile["sample_rate_hz"])
    billed_seconds = 0.0
    for recording in recordings:
        for arm in arms:
            primary = plan_primary(recording, profile, arm)
            finalize_count = sum(
                event["event"] == "finalize" for event in primary["events"]
            )
            billed_seconds += float(profile["session_lifetime_s"])
            billed_seconds += primary["accounting"]["synthetic_silence_samples"] / rate
            billed_seconds += float(profile["soniox"]["final_receipt_timeout_s"]) * finalize_count
            billed_seconds += sum(
                event.get("wait_before_finalize_ms", 0) / 1000.0
                for event in primary["events"]
                if event["event"] == "finalize"
            )
            if arm == "C":
                billed_seconds += float(profile["session_lifetime_s"])
                billed_seconds += float(profile["soniox"]["final_receipt_timeout_s"])
    return (
        billed_seconds
        / 3600.0
        * float(profile["pricing_snapshot"]["realtime_usd_per_hour_equivalent"])
    )


def append_audio_event(
    events: list[dict[str, Any]],
    *,
    kind: str,
    source_start: int | None,
    source_end: int | None,
    samples: int,
    available_s: float,
    send_clock_s: float,
    provider_cursor: int,
) -> tuple[float, int]:
    send_s = max(available_s, send_clock_s)
    events.append(
        {
            "event": "audio",
            "kind": kind,
            "source_start_sample": source_start,
            "source_end_sample": source_end,
            "provider_start_sample": provider_cursor,
            "provider_end_sample": provider_cursor + samples,
            "source_available_offset_s": available_s,
            "planned_send_offset_s": send_s,
            "planned_backlog_ms": max(0.0, (send_s - available_s) * 1000.0),
        }
    )
    return send_s, provider_cursor + samples


def plan_primary(recording: Recording, profile: dict[str, Any], arm_id: str) -> dict[str, Any]:
    rate = int(profile["sample_rate_hz"])
    chunk_samples = rate * int(profile["chunk_ms"]) // 1000
    arm = arm_by_id(profile, "B0" if arm_id == "C" else arm_id)
    session_start = int(recording.raw["session_source_start_sample"])
    events: list[dict[str, Any]] = []
    provider_cursor = 0
    send_clock_s = 0.0
    duplicated_source_samples = 0
    synthetic_samples = 0
    for segment in recording.raw["segments"]:
        segment_start = int(segment["source_start_sample"])
        segment_end = int(segment["source_end_sample"])
        segment_event_start = len(events)
        for prefix_start, prefix_end in segment["prefix_spans"]:
            count = prefix_end - prefix_start
            duplicated_source_samples += count
            send_clock_s, provider_cursor = append_audio_event(
                events,
                kind="prefix_context",
                source_start=prefix_start,
                source_end=prefix_end,
                samples=count,
                available_s=(segment_start - session_start) / rate,
                send_clock_s=send_clock_s,
                provider_cursor=provider_cursor,
            )
        cursor = segment_start
        while cursor < segment_end:
            end = min(cursor + chunk_samples, segment_end)
            send_clock_s, provider_cursor = append_audio_event(
                events,
                kind="real_content",
                source_start=cursor,
                source_end=end,
                samples=end - cursor,
                available_s=(end - session_start) / rate,
                send_clock_s=send_clock_s,
                provider_cursor=provider_cursor,
            )
            cursor = end
        boundary_available_s = (segment_end - session_start) / rate
        send_clock_s = max(send_clock_s, boundary_available_s)
        wait_ms = int(arm.get("wait_before_finalize_ms", 0))
        send_clock_s += wait_ms / 1000.0
        padding_ms = int(arm.get("padding_ms", 0))
        if "top_up_trailing_silence_ms" in arm:
            target = rate * int(arm["top_up_trailing_silence_ms"]) // 1000
            padding_samples = max(0, target - int(segment["transmitted_trailing_silence_samples"]))
        else:
            padding_samples = rate * padding_ms // 1000
        if padding_samples:
            synthetic_samples += padding_samples
            pacing = arm.get("padding_pacing")
            remaining = padding_samples
            while remaining:
                count = remaining if pacing == "immediate" else min(chunk_samples, remaining)
                send_clock_s, provider_cursor = append_audio_event(
                    events,
                    kind="synthetic_silence",
                    source_start=None,
                    source_end=None,
                    samples=count,
                    available_s=boundary_available_s,
                    send_clock_s=send_clock_s,
                    provider_cursor=provider_cursor,
                )
                if pacing == "realtime":
                    send_clock_s += count / rate
                remaining -= count
        events.append(
            {
                "event": "finalize",
                "segment_id": segment["id"],
                "boundary_type": segment["boundary_type"],
                "source_seal_sample": segment_end,
                "speech_end_source_sample": segment.get("speech_end_source_sample"),
                "planned_send_offset_s": send_clock_s,
                "transmitted_trailing_silence_samples": segment[
                    "transmitted_trailing_silence_samples"
                ],
                "synthetic_padding_samples": padding_samples,
                "wait_before_finalize_ms": wait_ms,
            }
        )
        for event in events[segment_event_start:]:
            event["segment_id"] = segment["id"]
    real_content = sum(
        event["provider_end_sample"] - event["provider_start_sample"]
        for event in events
        if event.get("kind") == "real_content"
    )
    expected_content = sum(
        segment["source_end_sample"] - segment["source_start_sample"]
        for segment in recording.raw["segments"]
    )
    if real_content != expected_content:
        raise AssertionError("source content conservation failed")
    provider_epoch_id = f"{recording.raw['id']}::{arm_id}::primary"
    for event in events:
        if event["event"] == "audio":
            event["capture_epoch"] = recording.raw["capture_epoch"]
            event["provider_epoch_id"] = provider_epoch_id
    session_end_offset_s = float(profile["session_lifetime_s"])
    events.append(
        {
            "event": "session_lifetime_reached",
            "planned_send_offset_s": max(session_end_offset_s, send_clock_s),
            "configured_session_lifetime_s": session_end_offset_s,
        }
    )
    return {
        "stream_role": "primary",
        "arm": arm_id,
        "recording_id": recording.raw["id"],
        "capture_epoch": recording.raw["capture_epoch"],
        "provider_epoch_id": provider_epoch_id,
        "events": events,
        "accounting": {
            "real_content_samples": real_content,
            "prefix_context_samples": duplicated_source_samples,
            "synthetic_silence_samples": synthetic_samples,
            "provider_input_samples": provider_cursor,
            "max_planned_backlog_without_terminal_wait_ms": max(
                (event.get("planned_backlog_ms", 0.0) for event in events), default=0.0
            ),
        },
        "terminal_receipt_wait": "unmeasured lower bound; live sender gates the next segment",
    }


def plan_observer(recording: Recording, profile: dict[str, Any]) -> dict[str, Any]:
    rate = int(profile["sample_rate_hz"])
    chunk_samples = rate * int(profile["chunk_ms"]) // 1000
    start = int(recording.raw["session_source_start_sample"])
    end = int(recording.raw["session_source_end_sample"])
    events: list[dict[str, Any]] = []
    provider_cursor = 0
    cursor = start
    while cursor < end:
        right = min(cursor + chunk_samples, end)
        _, provider_cursor = append_audio_event(
            events,
            kind="continuous_observation",
            source_start=cursor,
            source_end=right,
            samples=right - cursor,
            available_s=(right - start) / rate,
            send_clock_s=0.0,
            provider_cursor=provider_cursor,
        )
        cursor = right
    events.append(
        {
            "event": "finalize",
            "segment_id": "recording-end",
            "boundary_type": "recording_end",
            "source_seal_sample": end,
            "speech_end_source_sample": None,
            "planned_send_offset_s": (end - start) / rate,
            "transmitted_trailing_silence_samples": 0,
            "synthetic_padding_samples": 0,
            "wait_before_finalize_ms": 0,
        }
    )
    for event in events:
        event["segment_id"] = "recording-end"
    provider_epoch_id = f"{recording.raw['id']}::C::observer"
    for event in events:
        if event["event"] == "audio":
            event["capture_epoch"] = recording.raw["capture_epoch"]
            event["provider_epoch_id"] = provider_epoch_id
    return {
        "stream_role": "observer",
        "arm": "C",
        "recording_id": recording.raw["id"],
        "capture_epoch": recording.raw["capture_epoch"],
        "provider_epoch_id": provider_epoch_id,
        "events": events,
        "accounting": {
            "real_content_samples": end - start,
            "prefix_context_samples": 0,
            "synthetic_silence_samples": 0,
            "provider_input_samples": provider_cursor,
            "max_planned_backlog_without_terminal_wait_ms": 0.0,
        },
        "terminal_receipt_wait": "recording-end only",
    }


def plans_for(recordings: list[Recording], profile: dict[str, Any], arms: list[str]) -> list[dict[str, Any]]:
    plans: list[dict[str, Any]] = []
    for recording in recordings:
        for arm in arms:
            plans.append(plan_primary(recording, profile, arm))
            if arm == "C":
                plans.append(plan_observer(recording, profile))
    return plans


def disposition(inputs: Inputs, recordings: list[Recording], missing_coverage: set[str]) -> dict[str, Any]:
    arms = [arm["id"] for arm in inputs.profile["initial_arms"]]
    estimate = estimated_realtime_cost(inputs.profile, recordings, arms)
    blockers = approval_blockers(inputs, recordings, estimate)
    state = "ready" if not blockers and not missing_coverage else "blocked"
    per_arm = {
        arm: {
            "disposition": "blocked" if state == "blocked" else "ready_not_executed",
            "reason": blockers + ([f"missing coverage: {sorted(missing_coverage)}"] if missing_coverage else []),
        }

        for arm in arms
    }
    for arm in inputs.profile["conditional_arms"]:
        opened = arm.get("opened") is True
        per_arm[arm["id"]] = {
            "disposition": (
                ("blocked" if state == "blocked" else "ready_not_executed")
                if opened
                else "conditional_not_opened"
            ),
            "reason": (
                blockers
                + ([f"missing coverage: {sorted(missing_coverage)}"] if missing_coverage else [])
                if opened
                else arm["open_only_if"]
            ),
        }
    return {
        "status": state,
        "manifest_id": inputs.manifest.get("manifest_id"),
        "recording_count": len(recordings),
        "missing_coverage": sorted(missing_coverage),
        "estimated_initial_realtime_cost_usd": round(estimate, 6),
        "estimate_limit": "Not a maximum: Soniox bills tokens; output/context/retries are unknown.",
        "arms": per_arm,
        "blockers": blockers,
    }


class TraceWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle = path.open("x", encoding="utf-8")

    def write(self, value: dict[str, Any]) -> None:
        self._handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()
class FinalReceiptGate:
    def __init__(self) -> None:
        self._pending: deque[tuple[str, asyncio.Future[int]]] = deque()
        self._closed_error: BaseException | None = None

    @property
    def pending_segment_id(self) -> str | None:
        return self._pending[0][0] if self._pending else None

    def open(self, segment_id: str) -> asyncio.Future[int]:
        if self._closed_error is not None:
            raise RuntimeError("final receipt stream is unavailable") from self._closed_error
        if self._pending:
            raise RuntimeError("a finalize receipt is already pending")
        future: asyncio.Future[int] = asyncio.get_running_loop().create_future()
        self._pending.append((segment_id, future))
        return future

    def resolve_fin(self, receipt_ns: int) -> str | None:
        if not self._pending:
            return None
        segment_id, future = self._pending.popleft()
        if not future.done():
            future.set_result(receipt_ns)
        return segment_id

    def fail_all(self, error: BaseException) -> None:
        if self._closed_error is None:
            self._closed_error = error
        while self._pending:
            _segment_id, future = self._pending.popleft()
            if not future.done():
                future.set_exception(error)


async def exercise_final_receipt_gate() -> None:
    gate = FinalReceiptGate()
    capture_progressed = False
    next_segment_sent = False
    receipt = gate.open("first")

    async def progress_capture() -> None:
        nonlocal capture_progressed
        await asyncio.sleep(0)
        capture_progressed = True

    async def send_next_segment() -> None:
        nonlocal next_segment_sent
        await receipt
        next_segment_sent = True

    capture_task = asyncio.create_task(progress_capture())
    sender_task = asyncio.create_task(send_next_segment())
    await asyncio.sleep(0)
    await capture_task
    if not capture_progressed or next_segment_sent or gate.pending_segment_id != "first":
        raise AssertionError("source must progress while next-segment transmission remains gated")
    if gate.resolve_fin(123) != "first":
        raise AssertionError("final receipt must resolve the scoped segment")
    await sender_task
    if not next_segment_sent or gate.pending_segment_id is not None:
        raise AssertionError("next segment must release only after the scoped final receipt")

    failed_gate = FinalReceiptGate()
    failed_receipt = failed_gate.open("failed")
    failed_gate.fail_all(RuntimeError("offline provider failure"))
    try:
        await failed_receipt
    except RuntimeError:
        pass
    else:
        raise AssertionError("provider failure must fail the scoped gate")

    timed_gate = FinalReceiptGate()
    timed_receipt = timed_gate.open("timeout")
    try:
        await asyncio.wait_for(asyncio.shield(timed_receipt), timeout=0.001)
    except TimeoutError:
        timed_receipt.cancel()
        timed_gate.fail_all(RuntimeError("offline timeout"))
    else:
        raise AssertionError("missing final receipt must reach the bounded timeout path")


async def run_live_stream(
    recording: Recording,
    profile: dict[str, Any],
    plan: dict[str, Any],
    api_key: str,
    output_dir: Path,
) -> dict[str, Any]:
    import websockets

    role = plan["stream_role"]
    arm = plan["arm"]
    trace = TraceWriter(output_dir / f"{recording.raw['id']}--{arm}--{role}.jsonl")
    soniox = profile["soniox"]
    config: dict[str, Any] = {
        "api_key": api_key,
        "model": soniox["model"],
        "audio_format": soniox["audio_format"],
        "sample_rate": profile["sample_rate_hz"],
        "num_channels": profile["channels"],
        "enable_endpoint_detection": False,
        "enable_speaker_diarization": True,
        "enable_language_identification": soniox["enable_language_identification"],
    }
    if soniox["language_hints"]:
        config["language_hints"] = soniox["language_hints"]
    if soniox["context_terms"]:
        config["context"] = {"terms": soniox["context_terms"]}
    started_ns = time.monotonic_ns()
    connection_open_ns = started_ns
    last_send_ns = started_ns
    send_lock = asyncio.Lock()
    final_receipts = 0
    max_actual_backlog_ms = 0.0
    request_ids: set[str] = set()
    final_gate = FinalReceiptGate()
    active_segment_id: str | None = None
    recv_task: asyncio.Task[None] | None = None
    keepalive_task: asyncio.Task[None] | None = None
    trace.write(
        {
            "event": "session_start",
            "monotonic_ns": started_ns,
            "utc": utc_now(),
            "config": {key: value for key, value in config.items() if key != "api_key"},
            "profile_id": profile["profile_id"],
            "baseline_revision": profile["baseline_revision"],
            "execution_identity": plan["execution_identity"],
            "capture_epoch": plan["capture_epoch"],
            "provider_epoch_id": plan["provider_epoch_id"],
            "audio_sha256": recording.raw["sha256"],
            "reference_sha256": recording.raw["human_reference"]["sha256"],
        }
    )
    try:
        async with websockets.connect(soniox["endpoint"], ping_interval=None, open_timeout=10) as ws:
            connection_open_ns = time.monotonic_ns()
            await ws.send(json.dumps(config))
            last_send_ns = time.monotonic_ns()
            trace.write({"event": "config_sent", "monotonic_ns": last_send_ns})

            async def receive() -> None:
                nonlocal active_segment_id, final_receipts
                try:
                    async for message in ws:
                        receipt_ns = time.monotonic_ns()
                        if isinstance(message, bytes):
                            message = message.decode("utf-8", errors="replace")
                        try:
                            payload = json.loads(message)
                        except json.JSONDecodeError:
                            payload = {"unparsed_text": message}
                        attributed_segment_id = (
                            final_gate.pending_segment_id or active_segment_id
                        )
                        provider_error = False
                        if isinstance(payload, dict):
                            request_id = payload.get("request_id")
                            if request_id is not None:
                                request_ids.add(str(request_id))
                            provider_error = "error" in payload or "error_code" in payload
                            tokens = payload.get("tokens")
                            if isinstance(tokens, list):
                                for token in tokens:
                                    if (
                                        isinstance(token, dict)
                                        and token.get("text") == "<fin>"
                                        and token.get("is_final") is True
                                    ):
                                        resolved_segment = final_gate.resolve_fin(receipt_ns)
                                        if resolved_segment is not None:
                                            final_receipts += 1
                                            if active_segment_id == resolved_segment:
                                                active_segment_id = None
                        trace.write(
                            {
                                "event": "provider_receipt",
                                "monotonic_ns": receipt_ns,
                                "attributed_segment_id": attributed_segment_id,
                                "payload": payload,
                            }
                        )
                        if provider_error:
                            error = RuntimeError("Soniox returned an error response")
                            final_gate.fail_all(error)
                            raise error
                finally:
                    final_gate.fail_all(
                        RuntimeError("Soniox connection ended before scoped finalize receipt")
                    )

            async def keepalive() -> None:
                nonlocal last_send_ns
                interval = float(soniox["keepalive_interval_s"])
                while True:
                    await asyncio.sleep(interval / 2)
                    now_ns = time.monotonic_ns()
                    if (now_ns - last_send_ns) / 1e9 < interval:
                        continue
                    async with send_lock:
                        await ws.send(json.dumps({"type": "keepalive"}))
                        last_send_ns = time.monotonic_ns()
                        trace.write({"event": "keepalive_sent", "monotonic_ns": last_send_ns})

            recv_task = asyncio.create_task(receive())
            keepalive_task = asyncio.create_task(keepalive())
            with wave.open(str(recording.path), "rb") as wav:
                for event in plan["events"]:
                    target_ns = connection_open_ns + round(event["planned_send_offset_s"] * 1e9)
                    delay_s = (target_ns - time.monotonic_ns()) / 1e9
                    if delay_s > 0:
                        await asyncio.sleep(delay_s)
                    if event["event"] == "audio":
                        active_segment_id = event["segment_id"]
                        sample_count = event["provider_end_sample"] - event["provider_start_sample"]
                        if event["source_start_sample"] is None:
                            payload = bytes(sample_count * 2)
                        else:
                            wav.setpos(event["source_start_sample"])
                            payload = wav.readframes(sample_count)
                            if len(payload) != sample_count * 2:
                                raise RuntimeError("short WAV read during replay")
                        async with send_lock:
                            send_started_ns = time.monotonic_ns()
                            await ws.send(payload)
                            send_finished_ns = time.monotonic_ns()
                            last_send_ns = send_finished_ns
                        available_ns = connection_open_ns + round(
                            event["source_available_offset_s"] * 1e9
                        )
                        backlog_ms = max(0.0, (send_started_ns - available_ns) / 1e6)
                        max_actual_backlog_ms = max(max_actual_backlog_ms, backlog_ms)
                        trace.write(
                            event
                            | {
                                "event": "audio_sent",
                                "send_started_monotonic_ns": send_started_ns,
                                "send_finished_monotonic_ns": send_finished_ns,
                                "actual_backlog_ms": backlog_ms,
                            }
                        )
                    elif event["event"] == "finalize":
                        receipt = final_gate.open(event["segment_id"])
                        async with send_lock:
                            send_started_ns = time.monotonic_ns()
                            await ws.send(json.dumps({"type": "finalize"}))
                            send_finished_ns = time.monotonic_ns()
                            last_send_ns = send_finished_ns
                        trace.write(
                            event
                            | {
                                "event": "finalize_sent",
                                "send_started_monotonic_ns": send_started_ns,
                                "send_finished_monotonic_ns": send_finished_ns,
                            }
                        )
                        try:
                            receipt_ns = await asyncio.wait_for(
                                asyncio.shield(receipt),
                                timeout=float(soniox["final_receipt_timeout_s"]),
                            )
                        except BaseException as exc:
                            receipt.cancel()
                            trace.write(
                                {
                                    "event": "finalize_gate_failed",
                                    "segment_id": event["segment_id"],
                                    "monotonic_ns": time.monotonic_ns(),
                                    "error_type": type(exc).__name__,
                                }
                            )
                            raise
                        trace.write(
                            {
                                "event": "finalize_gate_released",
                                "segment_id": event["segment_id"],
                                "receipt_monotonic_ns": receipt_ns,
                                "wait_ms": (receipt_ns - send_finished_ns) / 1e6,
                            }
                        )
                    else:
                        trace.write(
                            event
                            | {
                                "event": "session_lifetime_reached",
                                "monotonic_ns": time.monotonic_ns(),
                            }
                        )
            async with send_lock:
                await ws.send("")
                last_send_ns = time.monotonic_ns()
                trace.write({"event": "stream_end_sent", "monotonic_ns": last_send_ns})
            try:
                await asyncio.wait_for(recv_task, timeout=30)
            finally:
                keepalive_task.cancel()
                await asyncio.gather(keepalive_task, return_exceptions=True)
    except BaseException as exc:
        trace.write(
            {
                "event": "connection_error",
                "monotonic_ns": time.monotonic_ns(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        raise
    finally:
        tasks = tuple(
            task for task in (recv_task, keepalive_task) if task is not None
        )
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        trace.close()
    return {
        "recording_id": recording.raw["id"],
        "arm": arm,
        "stream_role": role,
        "provider_request_ids": sorted(request_ids),
        "finalize_receipts": final_receipts,
        "max_actual_backlog_ms": max_actual_backlog_ms,
        "connection_lifetime_s": (time.monotonic_ns() - connection_open_ns) / 1e9,
        "execution_identity": plan["execution_identity"],
        "disposition": "executed",
    }


async def execute_live(
    recordings: list[Recording],
    profile: dict[str, Any],
    arms: list[str],
    output_dir: Path,
) -> list[dict[str, Any]]:
    api_key = os.environ["SONIOX_API_KEY"]
    all_plans = plans_for(recordings, profile, arms)
    output: list[dict[str, Any]] = []
    for recording in recordings:
        for arm in arms:
            selected = [
                plan
                for plan in all_plans
                if plan["recording_id"] == recording.raw["id"] and plan["arm"] == arm
            ]
            results = await asyncio.gather(
                *(run_live_stream(recording, profile, plan, api_key, output_dir) for plan in selected)
            )
            output.extend(results)
    return output


def environment_record() -> dict[str, Any]:
    try:
        import importlib.metadata

        websockets_version = importlib.metadata.version("websockets")
    except Exception:
        websockets_version = "unavailable"
    return {
        "recorded_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "implementation": platform.python_implementation(),
        "websockets": websockets_version,
    }


def select_arms(profile: dict[str, Any], value: str) -> list[str]:
    initial = [arm["id"] for arm in profile["initial_arms"]]
    conditional = {arm["id"]: arm for arm in profile["conditional_arms"]}
    requested = (
        initial if value == "all" else [item.strip() for item in value.split(",") if item.strip()]
    )
    if len(requested) != len(set(requested)):
        raise ValueError("duplicate arm ids are not allowed")
    unknown = set(requested) - set(initial) - set(conditional)
    if unknown:
        raise ValueError(f"unknown arms: {sorted(unknown)}")
    closed = [arm for arm in requested if arm in conditional and conditional[arm].get("opened") is not True]
    if closed:
        raise ValueError(f"conditional arms are not opened: {closed}")
    if not requested:
        raise ValueError("at least one arm must be selected")
    return requested


def self_check(profile: dict[str, Any]) -> dict[str, Any]:
    rate = int(profile["sample_rate_hz"])
    with tempfile.TemporaryDirectory(prefix="issue157-") as temp_dir_text:
        temp_dir = Path(temp_dir_text)
        wav_path = temp_dir / "fixture.wav"
        reference_path = temp_dir / "reference.json"
        with wave.open(str(wav_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            block = bytes(rate * 2)
            for _ in range(300):
                wav.writeframesraw(block)
        reference_path.write_text('{"fixture":"not accuracy evidence"}\n', encoding="utf-8")
        fixture_manifest = {
            "schema_version": 1,
            "manifest_id": "offline-accounting-fixture",
            "approval": {
                "external_processing_approved": False,
                "approved_by": None,
                "approved_utc_date": None,
                "paid_budget_usd": None,
                "observer_stream_included": False,
            },
            "recordings": [
                {
                    "id": "offline",
                    "path": str(wav_path),
                    "sha256": sha256_file(wav_path),
                    "license_or_consent_basis": "generated zero PCM; offline accounting only",
                    "capture_epoch": 0,
                    "boundary_schedule_scope": profile["boundary_schedule_scope"],
                    "human_reference": {
                        "path": str(reference_path),
                        "sha256": sha256_file(reference_path),
                        "checked_by": "self-check fixture",
                        "checked_utc_date": "2026-09-12",
                    },
                    "session_source_start_sample": 0,
                    "session_source_end_sample": 300 * rate,
                    "coverage_tags": sorted(REQUIRED_COVERAGE),
                    "segments": [
                        {
                            "id": "hard",
                            "source_start_sample": 0,
                            "source_end_sample": 6 * rate,
                            "prefix_spans": [],
                            "boundary_type": "hard_6s",
                            "transmitted_trailing_silence_samples": 0,
                            "speech_end_source_sample": None,
                            "vad_classification_note": "generated fixture; not acoustic evidence",
                        },
                        {
                            "id": "pause",
                            "source_start_sample": 6 * rate,
                            "source_end_sample": 10 * rate,
                            "prefix_spans": [[int(5.5 * rate), 6 * rate]],
                            "boundary_type": "pause_224ms",
                            "transmitted_trailing_silence_samples": 224 * rate // 1000,
                            "speech_end_source_sample": int(9.776 * rate),
                            "vad_classification_note": "generated fixture; not acoustic evidence",
                        },
                    ],
                }
            ],
        }
        manifest_path = temp_dir / "manifest.json"
        manifest_path.write_text(json.dumps(fixture_manifest), encoding="utf-8")
        inputs = Inputs(profile, fixture_manifest, manifest_path)
        recordings, missing = validate_recordings(inputs, require_any=True)
        if missing:
            raise AssertionError("self-check coverage setup failed")
        fixture_recording = fixture_manifest["recordings"][0]

        def require_valid_segment_annotation(mutator: Any) -> None:
            valid = json.loads(json.dumps(fixture_recording))
            mutator(valid["segments"])
            validate_segments(
                valid,
                0,
                300 * rate,
                rate,
                int(profile["source_frame_samples"]),
            )

        def require_invalid_segment_annotation(mutator: Any) -> None:
            invalid = json.loads(json.dumps(fixture_recording))
            mutator(invalid["segments"])
            try:
                validate_segments(
                    invalid,
                    0,
                    300 * rate,
                    rate,
                    int(profile["source_frame_samples"]),
                )
            except ValueError:
                return
            raise AssertionError("impossible boundary annotation was accepted")

        require_invalid_segment_annotation(
            lambda segments: segments[1].__setitem__("transmitted_trailing_silence_samples", 0)
        )
        require_invalid_segment_annotation(
            lambda segments: segments[1].__setitem__("transmitted_trailing_silence_samples", 1600)
        )
        require_invalid_segment_annotation(
            lambda segments: segments[0].__setitem__("transmitted_trailing_silence_samples", 8000)
        )
        require_valid_segment_annotation(
            lambda segments: segments[1].__setitem__(
                "transmitted_trailing_silence_samples", 12800
            )
        )
        require_valid_segment_annotation(
            lambda segments: segments[0].__setitem__(
                "transmitted_trailing_silence_samples", 3584
            )
        )
        require_invalid_segment_annotation(lambda segments: segments[0].pop("prefix_spans"))
        try:
            select_arms(profile, "B0,B0")
        except ValueError:
            pass
        else:
            raise AssertionError("duplicate arm ids were accepted")
        nested_plan = temp_dir / "new" / "nested" / "plan.json"
        write_text_with_parents(nested_plan, "{}\n")
        if nested_plan.read_text(encoding="utf-8") != "{}\n":
            raise AssertionError("nested plan output was not created")
        identity = execution_identity(DEFAULT_PROFILE, manifest_path)
        if any(len(value) != 40 and key == "git_revision" for key, value in identity.items()):
            raise AssertionError("execution git revision was not recorded")
        if any(len(value) != 64 for key, value in identity.items() if key != "git_revision"):
            raise AssertionError("execution artifact hashes were not recorded")
        asyncio.run(exercise_final_receipt_gate())
        plans = plans_for(
            recordings,
            profile,
            ["B0", "S200", "T200", "W200", "S200-paced", "T200-paced", "C"],
        )
        by_key = {(plan["arm"], plan["stream_role"]): plan for plan in plans}
        if by_key[("S200", "primary")]["accounting"]["synthetic_silence_samples"] != 6400:
            raise AssertionError("S200 must add 200 ms at both fixture boundaries")
        if by_key[("T200", "primary")]["accounting"]["synthetic_silence_samples"] != 3200:
            raise AssertionError("T200 must top up the hard cut and add zero at the 224 ms pause")
        if by_key[("W200", "primary")]["accounting"]["synthetic_silence_samples"] != 0:
            raise AssertionError("W200 must not synthesize audio")
        if by_key[("C", "observer")]["accounting"]["provider_input_samples"] != 300 * rate:
            raise AssertionError("C observer must preserve the continuous five-minute source")
        if by_key[("W200", "primary")]["accounting"][
            "max_planned_backlog_without_terminal_wait_ms"
        ] < 199:
            raise AssertionError("W200 must expose next-turn head-of-line backlog")
        if by_key[("B0", "primary")]["events"][-1] != {
            "event": "session_lifetime_reached",
            "planned_send_offset_s": 300.0,
            "configured_session_lifetime_s": 300.0,
        }:
            raise AssertionError("primary stream must remain open for the five-minute profile")
        if not approval_blockers(inputs, recordings, 0.01):
            raise AssertionError("live guard must reject the unapproved fixture")
        return {
            "status": "passed",
            "checks": [
                "five-minute normalized WAV and primary-session lifetime validation",
                "fixed hard/pause source spans, boundary schedule scope, and epochs",
                "pause tail 12800 and hard tie 3584 accepted; impossible annotations rejected",
                "B0/S200/T200/W200/immediate-vs-paced/C accounting",
                "prefix duplication remains source-mapped",
                "source progresses while scoped <fin> gates next-segment transmission",
                "terminal failure and bounded timeout stop the scoped gate",
                "W200 lower-bound backlog remains visible before terminal wait",
                "continuous observer source conservation",
                "duplicate arms and missing plan-output parents guarded",
                "git revision and replay/profile/manifest hashes recorded without credentials",
                "live approval and paid-budget guard",
            ],
            "accuracy_claim": "none; generated zero PCM is accounting-only",
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #157 isolated Soniox replay preparation")
    parser.add_argument("command", choices=("check", "plan", "self-check", "live"))
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--arms", default="all", help="comma-separated initial arm ids or all")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--authorize-paid-run", default="")
    args = parser.parse_args()
    try:
        inputs = load_inputs(args.profile, args.manifest)
        validate_profile(inputs.profile)
        arms = select_arms(inputs.profile, args.arms)
        if args.command == "self-check":
            print(json.dumps(self_check(inputs.profile), indent=2))
            return 0
        recordings, missing_coverage = validate_recordings(
            inputs, require_any=args.command in {"plan", "live"}
        )
        if args.command == "check":
            print(json.dumps(disposition(inputs, recordings, missing_coverage), indent=2))
            return 0
        plans = plans_for(recordings, inputs.profile, arms)
        identity = execution_identity(args.profile, args.manifest)
        for plan in plans:
            plan["execution_identity"] = identity
        if args.command == "plan":
            result = {
                "status": "planned_unmeasured",
                "environment": environment_record(),
                "profile_id": inputs.profile["profile_id"],
                "execution_identity": identity,
                "manifest_id": inputs.manifest["manifest_id"],
                "estimated_realtime_cost_usd": estimated_realtime_cost(
                    inputs.profile, recordings, arms
                ),
                "plans": plans,
            }
            text = json.dumps(result, indent=2)
            if args.output:
                write_text_with_parents(args.output, text + "\n")
            else:
                print(text)
            return 0
        estimate = estimated_realtime_cost(inputs.profile, recordings, arms)
        blockers = approval_blockers(inputs, recordings, estimate)
        if missing_coverage:
            blockers.append(f"required input coverage is missing: {sorted(missing_coverage)}")
        if args.authorize_paid_run != LIVE_AUTHORIZATION:
            blockers.append(f"--authorize-paid-run must equal {LIVE_AUTHORIZATION}")
        if not os.getenv("SONIOX_API_KEY"):
            blockers.append("SONIOX_API_KEY is absent")
        if blockers:
            raise PreparationBlocked("; ".join(blockers))
        if args.output is None:
            raise ValueError("live requires --output pointing to a new local run directory")
        output_dir = args.output.resolve()
        output_dir.mkdir(parents=True, exist_ok=False)
        (output_dir / "plan.json").write_text(
            json.dumps(
                {
                    "environment": environment_record(),
                    "execution_identity": identity,
                    "profile": inputs.profile,
                    "manifest_identity": {
                        "manifest_id": inputs.manifest["manifest_id"],
                        "sha256": sha256_file(inputs.manifest_path),
                    },
                    "estimated_realtime_cost_usd": estimate,
                    "plans": plans,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        result = asyncio.run(execute_live(recordings, inputs.profile, arms, output_dir))
        (output_dir / "run_summary.json").write_text(
            json.dumps({"status": "executed", "streams": result}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"status": "executed", "output": str(output_dir), "streams": result}, indent=2))
        return 0
    except (ValueError, PreparationBlocked) as exc:
        print(json.dumps({"status": "blocked" if isinstance(exc, PreparationBlocked) else "invalid", "reason": str(exc)}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
