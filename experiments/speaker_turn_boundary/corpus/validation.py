from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.speaker_turn_boundary.corpus.phase2_schemas import (
    Phase2Manifest,
    expected_change_kind,
    validate_phase2_manifest,
)
from experiments.speaker_turn_boundary.ground_truth import (
    classify_active_speaker_transitions,
    rebase_regions_to_epoch,
)
from experiments.speaker_turn_boundary.vad_baseline import load_canonical_wav

ZERO_GAP_MIN_JUNCTION_RMS = 1e-3
ZERO_GAP_JUNCTION_WINDOW_SAMPLES = 640


class CorpusValidationError(ValueError):
    pass


def resolve_case_wav(
    manifest: Phase2Manifest,
    case: Any,
    wav_roots: list[Path],
) -> Path | None:
    if not case.wav_relative_path:
        return None
    for root in wav_roots:
        candidate = (root / case.wav_relative_path).resolve()
        if candidate.is_file():
            return candidate
    return None


def validate_phase2_wavs(manifest: Phase2Manifest, wav_roots: list[Path]) -> list[str]:
    return validate_phase2_manifest(manifest, wav_roots)


def check_disjoint_speakers(manifest_a: Phase2Manifest, manifest_b: Phase2Manifest) -> list[str]:
    speakers_a = {source.speaker for case in manifest_a.cases for source in case.sources}
    speakers_b = {source.speaker for case in manifest_b.cases for source in case.sources}
    overlap = sorted(speakers_a & speakers_b)
    if overlap:
        return [
            f"speaker overlap between {manifest_a.manifest_id} and {manifest_b.manifest_id}: {overlap}"
        ]
    return []


def check_disjoint_sessions(manifest_a: Phase2Manifest, manifest_b: Phase2Manifest) -> list[str]:
    sessions_a = {source.session for case in manifest_a.cases for source in case.sources}
    sessions_b = {source.session for case in manifest_b.cases for source in case.sources}
    overlap = sorted(sessions_a & sessions_b)
    if overlap:
        return [
            f"session overlap between {manifest_a.manifest_id} and {manifest_b.manifest_id}: {overlap}"
        ]
    return []


def global_actor_ids(manifest: Phase2Manifest) -> set[str]:
    actors: set[str] = set()
    for case in manifest.cases:
        if case.condition.get("corpus") != "ami":
            continue
        agents = (case.condition.get("partition_meta") or {}).get("agents") or {}
        for global_name in agents.values():
            name = str(global_name).strip()
            if name:
                actors.add(name)
    return actors


def check_disjoint_global_actors(
    manifest_a: Phase2Manifest, manifest_b: Phase2Manifest
) -> list[str]:
    overlap = sorted(global_actor_ids(manifest_a) & global_actor_ids(manifest_b))
    if overlap:
        return [
            f"AMI global actor overlap between {manifest_a.manifest_id} and "
            f"{manifest_b.manifest_id}: {overlap}"
        ]
    return []


def check_case_gt_transitions(manifest: Phase2Manifest) -> list[str]:
    problems: list[str] = []
    for case in manifest.cases:
        if not case.regions:
            continue
        changes, transitions = classify_active_speaker_transitions(
            rebase_regions_to_epoch(list(case.regions), 0)
        )
        expected = expected_change_kind(case.condition, case.kind)
        if expected is None:
            if case.kind in {
                "different_speaker_gap",
                "different_speaker_overlap",
                "same_speaker",
                "gain_variation",
            }:
                if changes:
                    problems.append(f"case {case.case_id}: expected no GT change, got {changes}")
            continue
        if len(changes) != 1:
            problems.append(
                f"case {case.case_id}: expected exactly 1 GT change ({expected}), got {len(changes)}"
            )
            continue
        change = changes[0]
        if change.kind != expected:
            problems.append(
                f"case {case.case_id}: expected change kind {expected}, got {change.kind}"
            )
        splice = case.splice
        if splice is not None and change.change_sample != splice.b_onset_sample:
            problems.append(
                f"case {case.case_id}: GT change sample {change.change_sample} != splice b_onset {splice.b_onset_sample}"
            )
    return problems


def check_splice_equations(manifest: Phase2Manifest) -> list[str]:
    problems: list[str] = []
    for case in manifest.cases:
        splice = case.splice
        if splice is None:
            continue
        if splice.gap_samples is not None:
            expected_onset = splice.a_end_sample + splice.gap_samples
            if splice.b_onset_sample != expected_onset:
                problems.append(
                    f"case {case.case_id}: b_onset equation violated for gap "
                    f"({splice.b_onset_sample} != {splice.a_end_sample} + {splice.gap_samples})"
                )
        if splice.overlap_samples is not None:
            expected_onset = splice.a_end_sample - splice.overlap_samples
            if splice.b_onset_sample != expected_onset:
                problems.append(
                    f"case {case.case_id}: b_onset equation violated for overlap "
                    f"({splice.b_onset_sample} != {splice.a_end_sample} - {splice.overlap_samples})"
                )
    return problems


def check_zero_gap_acoustic(manifest: Phase2Manifest, wav_roots: list[Path]) -> list[str]:
    problems: list[str] = []
    for case in manifest.cases:
        if case.condition.get("gap_ms") != 0 or case.splice is None:
            continue
        wav_path = resolve_case_wav(manifest, case, wav_roots)
        if wav_path is None:
            problems.append(f"case {case.case_id}: wav missing")
            continue
        samples = load_canonical_wav(wav_path)
        junction = case.splice.b_onset_sample
        if junction != case.splice.a_end_sample:
            problems.append(f"case {case.case_id}: junction != a_end")
            continue
        pre = samples[max(0, junction - ZERO_GAP_JUNCTION_WINDOW_SAMPLES) : junction]
        post = samples[junction : junction + ZERO_GAP_JUNCTION_WINDOW_SAMPLES]
        pre_rms = float(np.sqrt(np.mean(np.square(pre.astype(np.float64))))) if pre.size else 0.0
        post_rms = float(np.sqrt(np.mean(np.square(post.astype(np.float64))))) if post.size else 0.0
        if pre_rms < ZERO_GAP_MIN_JUNCTION_RMS or post_rms < ZERO_GAP_MIN_JUNCTION_RMS:
            problems.append(
                f"case {case.case_id}: junction silence detected (pre_rms={pre_rms:.6f} post_rms={post_rms:.6f})"
            )
        evidence = case.zero_gap_evidence
        if evidence is not None:
            if not evidence.b_onset_is_a_end:
                problems.append(
                    f"case {case.case_id}: stored zero-gap evidence claims onset != a_end"
                )
            if abs(evidence.pre_junction_rms - pre_rms) > 1e-6:
                problems.append(
                    f"case {case.case_id}: stored pre-junction rms mismatch "
                    f"({evidence.pre_junction_rms} != {pre_rms})"
                )
            if abs(evidence.post_junction_rms - post_rms) > 1e-6:
                problems.append(
                    f"case {case.case_id}: stored post-junction rms mismatch "
                    f"({evidence.post_junction_rms} != {post_rms})"
                )
    return problems


def check_case_duration_bounds(manifest: Phase2Manifest, wav_roots: list[Path]) -> list[str]:
    problems: list[str] = []
    for case in manifest.cases:
        for source in case.sources:
            if not (0 <= source.cut_start_sample <= source.cut_end_sample):
                problems.append(f"case {case.case_id}: source {source.role} cut bounds invalid")
            if (
                source.trimmed_start_sample < 0
                or source.trimmed_end_sample > source.original_end_sample
            ):
                problems.append(
                    f"case {case.case_id}: source {source.role} trim out of original bounds"
                )
            if (
                source.cut_start_sample < source.trimmed_start_sample
                or source.cut_end_sample > source.trimmed_end_sample
            ):
                problems.append(
                    f"case {case.case_id}: source {source.role} cut outside trimmed region"
                )
        if case.regions:
            total = sum(r.end_sample - r.start_sample for r in case.regions)
            if total != case.duration_samples:
                problems.append(f"case {case.case_id}: regions do not tile duration")
    return problems


def manifest_identity_evidence(path: Path) -> dict[str, Any]:
    from experiments.speaker_turn_boundary.schemas import canonical_json

    raw = path.read_bytes()
    manifest = Phase2Manifest.from_dict(json.loads(raw.decode("utf-8")))
    canonical_bytes = canonical_json(manifest.to_dict()).encode("utf-8")
    normalized = raw.replace(b"\r\n", b"\n")
    return {
        "manifest_semantic_hash": manifest.hash,
        "manifest_canonical_file_sha256": hashlib.sha256(normalized).hexdigest(),
        "manifest_canonical_bytes_ok": normalized == canonical_bytes,
    }


def collect_validation_report(
    manifest: Phase2Manifest,
    wav_roots: Path | list[Path],
    *,
    counterpart_manifests: list[Phase2Manifest] | None = None,
) -> dict[str, Any]:
    roots = [wav_roots] if isinstance(wav_roots, Path) else list(wav_roots)
    problems = validate_phase2_wavs(manifest, roots)
    problems.extend(check_case_gt_transitions(manifest))
    problems.extend(check_splice_equations(manifest))
    problems.extend(check_zero_gap_acoustic(manifest, roots))
    problems.extend(check_case_duration_bounds(manifest, roots))
    disjointness: list[str] = []
    actor_evidence: list[dict[str, Any]] = []
    for counterpart in counterpart_manifests or []:
        disjointness.extend(check_disjoint_speakers(manifest, counterpart))
        disjointness.extend(check_disjoint_sessions(manifest, counterpart))
        disjointness.extend(check_disjoint_global_actors(manifest, counterpart))
        actors_a = sorted(global_actor_ids(manifest))
        actors_b = sorted(global_actor_ids(counterpart))
        if actors_a or actors_b:
            overlap = sorted(set(actors_a) & set(actors_b))
            actor_evidence.append(
                {
                    "manifest_a": manifest.manifest_id,
                    "manifest_b": counterpart.manifest_id,
                    "global_actors_a": actors_a,
                    "global_actors_b": actors_b,
                    "overlap": overlap,
                    "disjoint": not overlap,
                }
            )
    return {
        "manifest_id": manifest.manifest_id,
        "problems": problems,
        "valid": not problems and not disjointness,
        "disjointness_problems": disjointness,
        "global_actor_disjointness": actor_evidence,
    }
