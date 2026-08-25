from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable, Sequence

from experiments.psem_relative_occupancy_gate.contracts import (
    ActivityInterval,
    AnchorLifecycle,
    RelativeState,
    relative_state,
)
from experiments.psem_relative_occupancy_gate.decoder import simulate_gt_session
from experiments.psem_relative_occupancy_gate.evaluate import (
    aggregate_gate0_metrics,
    audit_gt_session,
    gate0_session_metrics,
    monotonic_boundary_matches,
)
from experiments.psem_relative_occupancy_gate.io_utils import (
    CONFIG_PATH,
    PACKAGE_ROOT,
    canonical_sha256,
    config,
    load_json,
    load_jsonl,
    percentile,
    safe_child,
    safe_output_path,
    sha256_file,
    write_json,
    write_jsonl,
    write_text,
)
from experiments.psem_relative_occupancy_gate.preflight import (
    validate_preflight_receipt,
)
from experiments.psem_relative_occupancy_gate.provenance import load_frozen_dataset

PRIMARY_TOPOLOGIES = (
    "clean_direct_different_speaker_handoff",
    "silence_gap_different_speaker_handoff",
    "same_speaker_silence_gap_resume",
    "overlap_return",
    "overlap_takeover",
    "short_backchannel_return",
)

CONTRACT_ARTIFACTS = (
    "BASELINE.md",
    "ONTOLOGY.md",
    "EVALUATOR.md",
    "config.json",
    "contracts.py",
    "decoder.py",
    "derive_relative_occupancy.py",
    "evaluate.py",
    "io_utils.py",
    "preflight.py",
    "provenance.py",
    "run_gate0.py",
    "verify_gate0.py",
)


class Gate0Error(RuntimeError):
    pass


def _validate_manifest_bindings(
    manifest: Sequence[dict[str, Any]], preflight: dict[str, Any]
) -> dict[str, Any]:
    cfg = config()
    dataset = load_frozen_dataset()
    expected_ids = set(dataset.source_ids("PSEM-STRATEGY-DEV"))
    observed_ids = [str(row.get("source_id")) for row in manifest]
    if len(observed_ids) != len(set(observed_ids)) or set(observed_ids) != expected_ids:
        raise Gate0Error("Gate 0 manifest is not the exact frozen V2 DEV source set")
    corpus = Path(str(preflight["paths"]["corpus_root"])).resolve()
    artifact_hashes = dataset.freeze["artifact_sha256"]
    expected_common = {
        "schema_version": "psem.relative_occupancy.manifest_row.v1",
        "ontology": cfg["experiment_id"],
        "role": "PSEM-STRATEGY-DEV",
        "sample_rate_hz": 16000,
        "dataset_freeze_file_sha256": cfg["dataset"]["freeze_file_sha256"],
        "dataset_freeze_payload_sha256": cfg["dataset"]["freeze_payload_sha256"],
        "source_manifest_sha256": artifact_hashes["source_manifest.jsonl"],
        "normalization_manifest_sha256": artifact_hashes["normalization_manifest.jsonl"],
        "split_manifest_sha256": artifact_hashes["split_manifest.json"],
        "reference_checkout_sha256": canonical_sha256(preflight["reference_receipt"]),
        "config_sha256": sha256_file(CONFIG_PATH),
        "eval_selection_sha256": None,
        "eval_status": "sealed",
    }
    for row in manifest:
        source_id = str(row["source_id"])
        source = dataset.sources[source_id]
        normalization = dataset.normalizations[source_id]
        assignment = dataset.assignments[source_id]
        payload = dict(row)
        stored_row_sha = payload.pop("row_sha256", None)
        if stored_row_sha != canonical_sha256(payload):
            raise Gate0Error(f"derived manifest row hash mismatch: {source_id}")
        if any(row.get(field) != expected for field, expected in expected_common.items()):
            raise Gate0Error(f"derived manifest common binding mismatch: {source_id}")
        expected_source = {
            "corpus": source["corpus"],
            "session_id": source["session_id"],
            "component_id": assignment["component_id"],
            "scored_start_sample": normalization["scored_start_sample"],
            "scored_end_sample": normalization["scored_end_sample"],
            "audio_ref": source["audio_ref"],
            "waveform_sha256": source["waveform_sha256"],
            "waveform_size_bytes": source["waveform_size_bytes"],
            "source_duration_samples": source["duration_samples"],
            "source_speaker_ids": source["speaker_ids"],
            "source_annotation_ref": source["annotation_ref"],
            "source_annotation_sha256": source["annotation_sha256"],
            "source_manifest_row_sha256": canonical_sha256(source),
            "normalization_manifest_row_sha256": canonical_sha256(normalization),
            "reference_ref": normalization["reference_ref"],
            "reference_sha256": normalization["reference_sha256"],
            "reference_repository": normalization["reference_repository"],
            "reference_commit": normalization["reference_commit"],
            "reference_git_tree": normalization["reference_git_tree"],
            "reference_metadata_files": normalization["reference_metadata_files"],
            "reference_metadata_sha256": normalization["reference_metadata_sha256"],
            "speaker_mapping_sha256": normalization["speaker_mapping_sha256"],
            "v2_exposure": normalization["exposure"],
            "v2_canonical_intervals_sha256": normalization["canonical_intervals_sha256"],
            "v2_label_result_sha256": normalization["label_result_sha256"],
            "v2_nonlexical_mask_sha256": normalization["nonlexical_mask_sha256"],
            "v2_source_record_sha256": normalization["source_record_sha256"],
        }
        if any(row.get(field) != expected for field, expected in expected_source.items()):
            raise Gate0Error(f"derived manifest source binding mismatch: {source_id}")
        waveform = safe_child(corpus, str(row["audio_ref"]), f"waveform {source_id}")
        if (
            str(waveform) != row.get("audio_path")
            or waveform.is_symlink()
            or not waveform.is_file()
            or waveform.stat().st_size != int(row["waveform_size_bytes"])
            or sha256_file(waveform) != row["waveform_sha256"]
        ):
            raise Gate0Error(f"derived manifest waveform mismatch: {source_id}")
        intervals = _intervals(row)
        if (
            canonical_sha256([interval.to_dict() for interval in intervals])
            != row.get("intervals_sha256")
            or not intervals
            or intervals[0].start_sample != int(row["scored_start_sample"])
            or intervals[-1].end_sample != int(row["scored_end_sample"])
            or any(
                left.end_sample != right.start_sample
                for left, right in zip(intervals, intervals[1:], strict=False)
            )
        ):
            raise Gate0Error(f"derived interval timeline mismatch: {source_id}")
        if not isinstance(row.get("transitions"), list) or not isinstance(
            row.get("topology_episodes"), list
        ):
            raise Gate0Error(f"derived topology records are missing: {source_id}")
    return dataset.summary()


def _intervals(row: dict[str, Any]) -> tuple[ActivityInterval, ...]:
    return tuple(ActivityInterval.from_dict(value) for value in row["intervals"])


def _clip_intervals(
    intervals: Sequence[ActivityInterval], start: int, end: int
) -> tuple[ActivityInterval, ...]:
    result = []
    for interval in intervals:
        left = max(start, interval.start_sample)
        right = min(end, interval.end_sample)
        if right > left:
            result.append(
                ActivityInterval(
                    start_sample=left,
                    end_sample=right,
                    active_speakers=interval.active_speakers,
                    masked=interval.masked,
                )
            )
    if not result or result[0].start_sample != start or result[-1].end_sample != end:
        raise Gate0Error("topology window is outside the canonical interval timeline")
    return tuple(result)


def _qualifying_other_only_runs(
    intervals: Sequence[ActivityInterval], anchor: str
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    boundary: int | None = None
    evidence = 0
    wall_end: int | None = None
    evidence_spans: list[list[int]] = []
    for interval in intervals:
        if interval.masked:
            if boundary is not None:
                wall_end = interval.end_sample
            continue
        state = relative_state(anchor, interval.active_speakers)
        if state is RelativeState.OTHER_ONLY:
            if boundary is None:
                boundary = interval.start_sample
            evidence += interval.end_sample - interval.start_sample
            wall_end = interval.end_sample
            evidence_spans.append([interval.start_sample, interval.end_sample])
            continue
        if boundary is not None and wall_end is not None:
            runs.append(
                {
                    "boundary_source_sample": boundary,
                    "evidence_samples": evidence,
                    "wall_end_sample": wall_end,
                    "evidence_spans": evidence_spans,
                }
            )
        boundary = None
        evidence = 0
        wall_end = None
        evidence_spans = []
    if boundary is not None and wall_end is not None:
        runs.append(
            {
                "boundary_source_sample": boundary,
                "evidence_samples": evidence,
                "wall_end_sample": wall_end,
                "evidence_spans": evidence_spans,
            }
        )
    return runs


def _first_run_expectation(
    runs: Sequence[dict[str, Any]], confirmation_samples: int
) -> dict[str, int] | None:
    for run in runs:
        if int(run["evidence_samples"]) < confirmation_samples:
            continue
        accumulated = 0
        for start, end in run["evidence_spans"]:
            duration = int(end) - int(start)
            needed = confirmation_samples - accumulated
            if duration >= needed:
                return {
                    "boundary_source_sample": int(run["boundary_source_sample"]),
                    "qualification_sample": int(start) + needed,
                }
            accumulated += duration
    return None


def _fixture_intervals(
    name: str,
) -> tuple[tuple[ActivityInterval, ...], AnchorLifecycle, str | None]:
    def span(
        start: int, end: int, speakers: Iterable[str], masked: bool = False
    ) -> ActivityInterval:
        return ActivityInterval(start, end, tuple(sorted(speakers)), masked)

    fixtures = {
        "direct_replacement": (
            (span(0, 1600, ["A"]), span(1600, 12800, ["B"])),
            AnchorLifecycle.ANCHORED,
            "A",
        ),
        "silence_gap_replacement": (
            (
                span(0, 1600, ["A"]),
                span(1600, 4800, []),
                span(4800, 16000, ["B"]),
            ),
            AnchorLifecycle.ANCHORED,
            "A",
        ),
        "same_anchor_pause_resume": (
            (
                span(0, 1600, ["A"]),
                span(1600, 4800, []),
                span(4800, 12800, ["A"]),
            ),
            AnchorLifecycle.ANCHORED,
            "A",
        ),
        "overlap_return": (
            (
                span(0, 1600, ["A"]),
                span(1600, 6400, ["A", "B"]),
                span(6400, 12800, ["A"]),
            ),
            AnchorLifecycle.ANCHORED,
            "A",
        ),
        "overlap_takeover": (
            (
                span(0, 1600, ["A"]),
                span(1600, 6400, ["A", "B"]),
                span(6400, 17600, ["B"]),
            ),
            AnchorLifecycle.ANCHORED,
            "A",
        ),
        "short_other_return": (
            (
                span(0, 1600, ["A"]),
                span(1600, 4800, ["B"]),
                span(4800, 12800, ["A"]),
            ),
            AnchorLifecycle.ANCHORED,
            "A",
        ),
        "initial_enrollment": (
            (span(0, 12800, ["A"]),),
            AnchorLifecycle.UNANCHORED,
            None,
        ),
        "masked_other": (
            (
                span(0, 1600, ["A"]),
                span(1600, 12800, ["B"], True),
                span(12800, 16000, ["A"]),
            ),
            AnchorLifecycle.ANCHORED,
            "A",
        ),
        "ongoing_singleton": (
            (span(0, 12800, ["A"]),),
            AnchorLifecycle.ANCHORED,
            "A",
        ),
        "ongoing_overlap": (
            (span(0, 12800, ["A", "B"]),),
            AnchorLifecycle.ANCHORED,
            "A",
        ),
    }
    return fixtures[name]


def _synthetic_examples(
    confirmation_values: Sequence[int], enrollment_samples: int, silence_reset_samples: int
) -> list[dict[str, Any]]:
    rows = []
    for name in (
        "direct_replacement",
        "silence_gap_replacement",
        "same_anchor_pause_resume",
        "overlap_return",
        "overlap_takeover",
        "short_other_return",
        "initial_enrollment",
        "masked_other",
        "ongoing_singleton",
        "ongoing_overlap",
    ):
        intervals, lifecycle, anchor = _fixture_intervals(name)
        for confirmation in confirmation_values:
            outcome = simulate_gt_session(
                source_id=f"fixture:{name}",
                intervals=intervals,
                confirmation_samples=confirmation,
                enrollment_samples=enrollment_samples,
                silence_reset_samples=silence_reset_samples,
                initial_lifecycle=lifecycle,
                initial_anchor=anchor,
            )
            rows.append(
                {
                    "schema_version": "psem.relative_occupancy.gate0_example.v1",
                    "kind": "synthetic_fixture",
                    "name": name,
                    "confirmation_samples": confirmation,
                    "intervals": [value.to_dict() for value in intervals],
                    "events": [value.to_dict() for value in outcome.events],
                    "enrollments": [value.to_dict() for value in outcome.enrollments],
                    "timeline": [value.to_dict() for value in outcome.timeline],
                    "boundary_audit": audit_gt_session(outcome),
                }
            )
    return rows


def _natural_examples(
    manifest: Sequence[dict[str, Any]],
    confirmation_values: Sequence[int],
    enrollment_samples: int,
    silence_reset_samples: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    selected: dict[str, tuple[dict[str, Any], dict[str, Any], str]] = {}
    for source in manifest:
        transitions = {value["transition_id"]: value for value in source["transitions"]}
        for episode in source["topology_episodes"]:
            topology = str(episode["primary_topology"])
            if topology not in PRIMARY_TOPOLOGIES or topology in selected:
                continue
            if not episode.get("coverage_gate_eligible", False):
                continue
            episode_transitions = [
                transitions[value] for value in episode["transition_ids"] if value in transitions
            ]
            anchor = next(
                (
                    str(value["from_speaker"])
                    for value in episode_transitions
                    if value.get("from_speaker")
                ),
                None,
            )
            if anchor is not None:
                selected[topology] = (source, episode, anchor)
    missing = sorted(set(PRIMARY_TOPOLOGIES) - set(selected))
    rows = []
    violations = []
    for topology in PRIMARY_TOPOLOGIES:
        if topology not in selected:
            continue
        source, episode, anchor = selected[topology]
        source_intervals = _intervals(source)
        transitions = {value["transition_id"]: value for value in source["transitions"]}
        episode_transitions = [transitions[value] for value in episode["transition_ids"]]
        interval_indices = [
            int(index)
            for transition in episode_transitions
            for index in (
                transition.get("from_interval_index"),
                transition.get("to_interval_index"),
            )
            if index is not None
        ]
        if not interval_indices:
            violations.append(f"{source['source_id']}:{episode['episode_id']}:no_intervals")
            continue
        window_start = source_intervals[min(interval_indices)].start_sample
        window_end = source_intervals[max(interval_indices)].end_sample
        intervals = _clip_intervals(source_intervals, window_start, window_end)
        runs = _qualifying_other_only_runs(intervals, anchor)
        for confirmation in confirmation_values:
            outcome = simulate_gt_session(
                source_id=str(source["source_id"]),
                intervals=intervals,
                confirmation_samples=confirmation,
                enrollment_samples=enrollment_samples,
                silence_reset_samples=silence_reset_samples,
                initial_lifecycle=AnchorLifecycle.ANCHORED,
                initial_anchor=anchor,
            )
            expectation = _first_run_expectation(runs, confirmation)
            expected_cut = expectation is not None
            initial_episode_id = outcome.enrollments[0].anchor_episode_id
            initial_events = [
                event for event in outcome.events if event.anchor_episode_id == initial_episode_id
            ]
            observed_cut = len(initial_events) == 1
            exact_timing = not expected_cut and not initial_events
            if expectation is not None and len(initial_events) == 1:
                event = initial_events[0]
                exact_timing = (
                    event.boundary_source_sample == expectation["boundary_source_sample"]
                    and event.model_evidence_frontier_sample == expectation["qualification_sample"]
                    and event.decoder_emit_sample == expectation["qualification_sample"]
                )
            audit = audit_gt_session(outcome)
            if expected_cut != observed_cut or not exact_timing or not audit["passed"]:
                violations.append(f"{source['source_id']}:{episode['episode_id']}:{confirmation}")
            rows.append(
                {
                    "schema_version": "psem.relative_occupancy.gate0_example.v1",
                    "kind": "natural_v2_dev",
                    "source_id": source["source_id"],
                    "episode_id": episode["episode_id"],
                    "primary_topology": topology,
                    "anchor_speaker": anchor,
                    "confirmation_samples": confirmation,
                    "expected_cut_from_other_only_policy": expected_cut,
                    "expected_replacement": expectation,
                    "exact_replacement_timing": exact_timing,
                    "other_only_runs": runs,
                    "events": [value.to_dict() for value in outcome.events],
                    "additional_replacement_count": len(outcome.events) - len(initial_events),
                    "timeline": [value.to_dict() for value in outcome.timeline],
                    "boundary_audit": audit,
                }
            )
    return rows, missing + violations


def _diagnostic_handoff_alignment(
    manifest: Sequence[dict[str, Any]],
    outcomes: Sequence[Any],
    tolerance_samples: int,
    sample_rate_hz: int,
) -> dict[str, Any]:
    source_rows = {str(row["source_id"]): row for row in manifest}
    match_count = 0
    predicted_count = 0
    reference_count = 0
    absolute_errors_ms: list[float] = []
    for outcome in outcomes:
        source = source_rows[outcome.source_id]
        predicted = sorted(event.boundary_source_sample for event in outcome.events)
        reference = sorted(
            int(value["handoff_source_sample"])
            for value in source["transitions"]
            if value.get("handoff_confirmed") == 1
            and value.get("handoff_source_sample") is not None
        )
        matches = monotonic_boundary_matches(predicted, reference, tolerance_samples)
        predicted_count += len(predicted)
        reference_count += len(reference)
        match_count += len(matches)
        absolute_errors_ms.extend(
            abs(predicted[predicted_index] - reference[reference_index]) * 1000.0 / sample_rate_hz
            for predicted_index, reference_index in matches
        )
    return {
        "role": "historical_psem_handoff_v1_diagnostic_only",
        "tolerance_ms": tolerance_samples * 1000.0 / sample_rate_hz,
        "predicted_cut_count": predicted_count,
        "derived_handoff_count": reference_count,
        "matched_count": match_count,
        "unmatched_predicted_count": predicted_count - match_count,
        "unmatched_reference_count": reference_count - match_count,
        "precision": match_count / predicted_count if predicted_count else 0.0,
        "recall": match_count / reference_count if reference_count else 0.0,
        "absolute_boundary_displacement_ms": {
            "p50": percentile(absolute_errors_ms, 50),
            "p90": percentile(absolute_errors_ms, 90),
        },
    }


def run_gate0(manifest_path: Path, preflight_path: Path, output_dir: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    preflight_path = preflight_path.resolve()
    output_dir = safe_output_path(output_dir)
    cfg = config()
    preflight = load_json(preflight_path)
    validate_preflight_receipt(preflight)
    manifest = load_jsonl(manifest_path)
    dataset_binding = _validate_manifest_bindings(manifest, preflight)
    sample_rate = int(cfg["dataset"]["sample_rate_hz"])
    enrollment_samples = int(cfg["gate0_enrollment_confirm_ms"]) * sample_rate // 1000
    silence_reset_samples = int(cfg["lifecycle_proxy_silence_reset_ms"]) * sample_rate // 1000
    confirmation_values = [
        int(value) * sample_rate // 1000 for value in cfg["replacement_confirm_ms"]
    ]
    alignment_tolerance_samples = (
        int(cfg["derived_handoff_alignment_tolerance_ms"]) * sample_rate // 1000
    )
    active_speech_samples = sum(
        interval.end_sample - interval.start_sample
        for row in manifest
        for interval in _intervals(row)
        if interval.active_speakers
    )
    settings = []
    all_results: dict[int, list[Any]] = {}
    for confirmation in confirmation_values:
        outcomes = [
            simulate_gt_session(
                source_id=str(row["source_id"]),
                intervals=_intervals(row),
                confirmation_samples=confirmation,
                enrollment_samples=enrollment_samples,
                silence_reset_samples=silence_reset_samples,
            )
            for row in manifest
        ]
        all_results[confirmation] = outcomes
        source_metrics = [gate0_session_metrics(value, sample_rate) for value in outcomes]
        settings.append(
            {
                "confirmation_ms": confirmation * 1000 // sample_rate,
                "aggregate": aggregate_gate0_metrics(
                    source_metrics, active_speech_samples / sample_rate
                ),
                "derived_handoff_alignment": _diagnostic_handoff_alignment(
                    manifest, outcomes, alignment_tolerance_samples, sample_rate
                ),
                "sources": source_metrics,
            }
        )
    no_cut_confirmation = (
        max(int(row["scored_end_sample"]) - int(row["scored_start_sample"]) for row in manifest) + 1
    )
    no_cut_metrics = [
        gate0_session_metrics(
            simulate_gt_session(
                source_id=str(row["source_id"]),
                intervals=_intervals(row),
                confirmation_samples=no_cut_confirmation,
                enrollment_samples=enrollment_samples,
                silence_reset_samples=silence_reset_samples,
            ),
            sample_rate,
        )
        for row in manifest
    ]
    synthetic = _synthetic_examples(confirmation_values, enrollment_samples, silence_reset_samples)
    natural, natural_failures = _natural_examples(
        manifest, confirmation_values, enrollment_samples, silence_reset_samples
    )
    event_boundaries = {
        "direct_replacement": 1600,
        "silence_gap_replacement": 4800,
        "overlap_takeover": 6400,
    }
    synthetic_failures = []
    for row in synthetic:
        name = str(row["name"])
        confirmation = int(row["confirmation_samples"])
        boundary = event_boundaries.get(name)
        if name == "short_other_return" and confirmation <= 3200:
            boundary = 1600
        expected_event_count = 1 if boundary is not None else 0
        events = row["events"]
        exact_event = len(events) == expected_event_count
        if boundary is not None and len(events) == 1:
            expected_emit = boundary + confirmation
            exact_event = (
                int(events[0]["boundary_source_sample"]) == boundary
                and int(events[0]["model_evidence_frontier_sample"]) == expected_emit
                and int(events[0]["decoder_emit_sample"]) == expected_emit
            )
        exact_enrollment = True
        if name == "initial_enrollment":
            exact_enrollment = len(row["enrollments"]) == 1 and row["enrollments"][0] == {
                "source_id": "fixture:initial_enrollment",
                "anchor_episode_id": "fixture:initial_enrollment:A00001",
                "anchor_id": "A",
                "opportunity_start_sample": 0,
                "anchor_emit_sample": enrollment_samples,
            }
        if not exact_event or not exact_enrollment or row["boundary_audit"]["passed"] is not True:
            synthetic_failures.append(f"{name}:{confirmation}")
    passed = not synthetic_failures and not natural_failures
    output_dir.mkdir(parents=True, exist_ok=True)
    event_rows = [
        {
            "schema_version": "psem.relative_occupancy.gate0_event.v1",
            "confirmation_ms": confirmation * 1000 // sample_rate,
            **event.to_dict(),
        }
        for confirmation, outcomes in sorted(all_results.items())
        for outcome in outcomes
        for event in outcome.events
    ]
    events_path = output_dir / "gate0_oracle_events.jsonl"
    write_jsonl(events_path, event_rows)
    examples_path = output_dir / "gate0_topology_examples.jsonl"
    write_jsonl(examples_path, [*synthetic, *natural])
    metrics = {
        "schema_version": "psem.relative_occupancy.gate0_metrics.v1",
        "ontology": cfg["experiment_id"],
        "role": "PSEM-STRATEGY-DEV",
        "eval_status": "sealed",
        "config_sha256": sha256_file(CONFIG_PATH),
        "frozen_dataset": dataset_binding,
        "contract_artifacts": {
            name: sha256_file(PACKAGE_ROOT / name) for name in CONTRACT_ARTIFACTS
        },
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "preflight_path": str(preflight_path.resolve()),
        "preflight_sha256": sha256_file(preflight_path),
        "topology_examples_path": str(examples_path.resolve()),
        "topology_examples_sha256": sha256_file(examples_path),
        "oracle_events_path": str(events_path.resolve()),
        "oracle_events_sha256": sha256_file(events_path),
        "lifecycle_proxy": {
            "kind": "gt_speech_non_speech",
            "enrollment_confirm_ms": cfg["gate0_enrollment_confirm_ms"],
            "silence_reset_ms": cfg["lifecycle_proxy_silence_reset_ms"],
        },
        "no_speaker_cut_baseline": aggregate_gate0_metrics(
            no_cut_metrics, active_speech_samples / sample_rate
        ),
        "settings": settings,
        "synthetic_fixture_failures": synthetic_failures,
        "natural_topology_failures": natural_failures,
        "passed": passed,
    }
    metrics["content_sha256"] = canonical_sha256(metrics)
    metrics_path = output_dir / "gate0_oracle_metrics.json"
    write_json(metrics_path, metrics)
    result = "PASS" if passed else "FAIL"
    statement = (
        "The two relative-occupancy variables plus explicit anchor lifecycle and short causal "
        "history are information-sufficient for the mandatory behavior."
        if passed
        else "The mandatory relative-occupancy behavior was not established."
    )
    markdown = (
        "# Gate 0 ontology result\n\n"
        f"Result: **{result}**\n\n"
        f"{statement}\n\n"
        "V2 EVAL remained sealed. This result uses perfect GT activity, the frozen V2 masks, "
        "a 200 ms reliable-singleton enrollment rule, and the explicit 1200 ms GT lifecycle "
        "proxy. It does not select one replacement duration as product-optimal.\n\n"
        f"Synthetic fixture failures: `{synthetic_failures}`\n\n"
        f"Natural DEV topology failures: `{natural_failures}`\n"
    )
    write_text(output_dir / "GATE0_ONTOLOGY_RESULT.md", markdown)
    if not passed:
        raise Gate0Error("Gate 0 mandatory invariants failed")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    metrics = run_gate0(
        args.manifest.resolve(), args.preflight.resolve(), args.output_dir.resolve()
    )
    print(f"Gate 0 passed with {len(metrics['settings'])} fixed decoder settings and EVAL sealed")


if __name__ == "__main__":
    main()
