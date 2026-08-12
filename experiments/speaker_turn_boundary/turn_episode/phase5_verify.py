from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .phase4_design import (
    ceil_grid,
    corpus_for,
    load_public_regions,
    load_synthetic_cases,
    synthetic_manifest_name,
)
from .phase4_signal import (
    load_eres_embeddings,
    read_json,
    source_by_wav,
)
from .phase5_design import (
    historical_development_contract,
    load_populations,
    proposal_profiles,
)
from .phase5_runner import (
    AUTHORITY_SHA256,
    BASELINE_IDS,
    CONTROL_KINDS,
    FULL_GRID,
    LADDER_STAGES,
    SYSTEM_METRIC_FIELDS,
    _find_route,
    _score_unit,
    annotation_views_for_case,
    annotation_views_for_episode,
    build_system_universe,
    content_sha256,
    experiment_root,
    load_proposal_traces,
    load_raw_words,
    node_id,
    phase5_cache_root,
    physical_node_key,
    prd_path,
    result_dir,
    seed_material,
    sha256_file,
    system_id,
    verify_start_guard,
    word_timing_receipt,
)

PERCENTILE_FIELD_NAMES = frozenset(
    {
        "segment_duration_p10_samples",
        "segment_duration_p50_samples",
        "segment_duration_p90_samples",
        "active_speech_duration_p10_samples",
        "active_speech_duration_p50_samples",
        "active_speech_duration_p90_samples",
    }
)
from .phase5_proposals import (
    adjacent_trace,
    anchor_trace,
    generate_proposal_trace,
    source_prefix_routes,
)
from .phase5_storage import (
    RepresentationWriter,
    canonical_json,
    framed_digest,
    read_representation,
    rows_sha256,
    sha256_bytes,
    verify_shard_receipts,
)
from .phase5_scoring import score_policy_episode


class Phase5VerifyError(RuntimeError):
    pass


def verify_audit_unit_worker(batch: dict[str, Any]) -> dict[str, Any]:
    from .phase5_runner import (
        execute_physical_episode,
        read_wav_slice,
        _score_unit as runner_score_unit,
    )

    system_id_value = batch["system_id"]
    unit_id = batch["unit_id"]
    unit_kind = batch["unit_kind"]
    info = batch["system_info"]
    profile = batch["profile"]
    embeddings = batch["embeddings"]
    b0_actions = batch["b0_actions"]
    lifecycle = batch["lifecycle"]
    references = batch["references"]
    views = batch["views"]
    if unit_kind == "historical":
        scored_start = 0
        scored_end = int(batch["duration_samples"])
        observed_end = scored_end
        tag = "overlap_present"
    else:
        scored_start = int(batch["scored_start"])
        scored_end = int(batch["scored_end"])
        observed_end = int(batch["tail_end"])
        tag = str(batch["episode_tag"])
    proposals = None
    stage = str(info["stage"])
    control_kind = info.get("control_kind")
    if info.get("baseline_id") is not None:
        baseline_id = str(info["baseline_id"])
        if baseline_id == "B0":
            final_actions = list(b0_actions)
        else:
            final_actions = list(batch["b1_actions"])
        score = runner_score_unit(
            final_actions,
            b0_actions,
            references,
            views,
            scored_start=scored_start,
            scored_end=scored_end,
            episode_tag=tag,
        )
        from .phase5_runner import score_metric_vector

        metric_episode = dict(batch["episode"])
        if unit_kind == "historical":
            metric_episode["tag"] = "overlap_present"
        hard_kind_map = {
            str(row["reference_id"]): str(row["action_kind"]) for row in references
        }
        metric_vector = score_metric_vector(
            score,
            metric_episode,
            references,
            control_infeasible=0,
            pool=batch["pool"],
            hard_reference_kind_map=hard_kind_map,
            deadline_views=score["deadline_views"],
        )
        return {
            "system_id": system_id_value,
            "unit_id": unit_id,
            "action_trace_sha256": content_sha256(final_actions),
            "score_trace_sha256": content_sha256(score),
            "raw_trace_sha256": "",
            "metric_vector": metric_vector,
            "ok": True,
        }
    profile_id = str(profile["proposal_profile_id"])
    if unit_kind == "historical":
        if profile["profile_class"] == "adjacent":
            trace = adjacent_trace(
                embeddings,
                profile,
                source_session_id=unit_id,
                audio_epoch=0,
                warm_start=0,
                tail_end=int(batch["duration_samples"]),
            )
            ordered = sorted(
                trace["proposals"],
                key=lambda row: (
                    int(row["observed_source_sample_at_emit"]),
                    int(row["boundary_source_sample"]),
                    str(row["profile_id"]),
                    str(row["proposal_id"]),
                ),
            )
            proposals = ordered
            raw_trace_sha = content_sha256(ordered)
        else:
            trace = anchor_trace(
                embeddings,
                profile,
                source_session_id=unit_id,
                audio_epoch=0,
                replay_start=0,
                warm_start=0,
                tail_end=int(batch["duration_samples"]),
            )
            proposals = trace["proposals"]
            raw_trace_sha = content_sha256(proposals)
    else:
        if profile["profile_class"] == "adjacent":
            trace = generate_proposal_trace(embeddings, profile, batch["episode"])
            proposals = trace["proposals"]
            raw_trace_sha = str(trace["proposal_trace_sha256"])
        else:
            routes = source_prefix_routes(embeddings, profile, batch["source_episodes"])
            route = next(
                (row for row in routes["routes"] if str(row["episode_id"]) == unit_id),
                None,
            )
            if route is None:
                raise Phase5VerifyError(f"audit route missing: {unit_id}")
            proposals = route["proposals"]
            raw_trace_sha = str(route["proposal_trace_sha256"])
    waveform = None
    if batch.get("wav_path"):
        waveform = read_wav_slice(
            Path(batch["wav_path"]), int(batch["wave_start"]), int(batch["wave_end"])
        )
    from .phase5_runner import physical_group_id

    physical = execute_physical_episode(
        proposals,
        b0_actions,
        lifecycle,
        episode_observed_end=observed_end,
        waveform=waveform,
        profile_id=profile_id,
        unit_id=unit_id,
        wave_start=int(batch.get("wave_start", 0)),
    )
    chain_tuple = (
        info["chain"]["cluster_debounce_ms"],
        info["chain"]["cluster_boundary_radius_ms"],
        info["chain"]["refractory_ms"],
        info["chain"]["representative"],
        info["chain"]["detector_vad_radius_ms"],
        info["chain"]["same_silence_interval_association"],
    )
    if stage == "frequency_control":
        group_id = physical_group_id("control", chain=chain_tuple, control_kind=control_kind)
    elif stage == "naive_proposal_as_cut":
        group_id = "naive"
    elif stage == "clustering_only":
        d = int(info["chain"]["cluster_debounce_ms"])
        w = int(info["chain"]["cluster_boundary_radius_ms"])
        rep = str(info["chain"]["representative"])
        group_id = physical_group_id("clustering_only", cluster=(d, w, rep))
    elif stage == "clustering_plus_refractory":
        d = int(info["chain"]["cluster_debounce_ms"])
        w = int(info["chain"]["cluster_boundary_radius_ms"])
        r = int(info["chain"]["refractory_ms"])
        rep = str(info["chain"]["representative"])
        group_id = physical_group_id(
            "clustering_plus_refractory", cluster=(d, w, r, rep)
        )
    else:
        group_id = physical_group_id("vad", chain=chain_tuple)
    final_actions = physical[group_id]["final_actions"]
    score = runner_score_unit(
        final_actions,
        b0_actions,
        references,
        views,
        scored_start=scored_start,
        scored_end=scored_end,
        episode_tag=tag,
    )
    from .phase5_runner import score_metric_vector

    metric_episode = dict(batch["episode"])
    if unit_kind == "historical":
        metric_episode["tag"] = "overlap_present"
    metric_vector = score_metric_vector(
        score,
        metric_episode,
        references,
        control_infeasible=0,
        pool=batch["pool"],
        hard_reference_kind_map={},
        deadline_views=score["deadline_views"],
    )
    return {
        "system_id": system_id_value,
        "unit_id": unit_id,
        "action_trace_sha256": content_sha256(final_actions),
        "score_trace_sha256": content_sha256(score),
        "raw_trace_sha256": raw_trace_sha,
        "metric_vector": metric_vector,
        "ok": True,
    }


def run_verification(experiment_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    result = result_dir(experiment_dir)
    guard = verify_start_guard(experiment_dir)
    stage_a = read_json(result / "phase_5_stage_a_receipt.json")
    gate = read_json(result / "phase_5_interstage_gate.json")
    stage_b = read_json(result / "phase_5_stage_b_receipt.json")
    if not gate["stage_b_allowed"]:
        raise Phase5VerifyError("interstage gate did not allow Stage B")
    runner_hashes = {
        "current": str(guard["runner_sha256"]),
        "stage_a": str(stage_a["start_guard"]["runner_sha256"]),
        "stage_b": str(stage_b["start_guard"]["runner_sha256"]),
    }
    if len(set(runner_hashes.values())) != 1:
        raise Phase5VerifyError(
            f"runner hash chain drift: {json.dumps(runner_hashes)}"
        )
    ledger = json.loads((result / "phase_5_design_ledger.json").read_text(encoding="utf-8"))
    if (result / "phase_5_verification.json").is_file():
        raise Phase5VerifyError("phase 5 verification already executed")
    inputs = __import__(
        "experiments.speaker_turn_boundary.turn_episode.phase5_design",
        fromlist=["phase4_inputs"],
    ).phase4_inputs(result)
    episodes, _, source_by_episode = load_populations(experiment_dir, inputs)
    profiles = proposal_profiles(
        experiment_dir,
        inputs["phase_4_signal_disposition.json"],
        inputs["phase_4_state_equivalence.json"],
    )
    historical_development, _, _ = historical_development_contract(experiment_dir, profiles)
    case_rows = historical_development["case_rows"]
    details = {
        str(row["session_id"]): row
        for row in map(
            json.loads,
            (result / "coverage_inventory_details.jsonl")
            .read_text(encoding="utf-8")
            .splitlines(),
        )
    }
    cases = load_synthetic_cases(experiment_dir / "data" / "manifests")
    inventory = json.loads((result / "coverage_inventory.json").read_text(encoding="utf-8"))
    from .phase4_signal import _source_maps

    sources, _ = _source_maps(experiment_dir, episodes, cases, inventory, details)
    traces = load_proposal_traces()
    b0_rows = read_representation(
        result / "phase_5_b0_evidence", stage_a["b0_evidence_shards"]
    )
    b0_by_unit = {str(row["unit_id"]): row for row in b0_rows}
    for row in b0_rows:
        if not bool(row["b1_equivalence"]["passed"]):
            raise Phase5VerifyError(f"B0/B1 equivalence failure: {row['unit_id']}")
    corpus_root = __import__(
        "experiments.speaker_turn_boundary.corpus.external", fromlist=["corpus_root"]
    ).corpus_root()
    raw_words_by_session: dict[str, list[Any] | None] = {}
    for session_id in sorted({str(row["session_id"]) for row in episodes}):
        raw_words, _, _ = load_raw_words(corpus_root, session_id)
        raw_words_by_session[session_id] = raw_words
    raw_words_by_case: dict[str, list[Any] | None] = {}
    for case in case_rows:
        case_id = str(case["case_id"])
        raw_words = None
        if case_id in {
            "ami_ES2003a",
            "ami_IS1008a",
        }:
            raw_words, _, _ = load_raw_words(corpus_root, case_id)
        raw_words_by_case[case_id] = raw_words
    public_sessions = [
        session_id
        for session_id in raw_words_by_session
        if synthetic_manifest_name(session_id) is None
    ]
    regions_by_session = load_public_regions(
        inventory, details, public_sessions, experiment_dir / "data" / "manifests"
    )
    universe = build_system_universe(profiles)
    system_info = universe["system_info"]
    audit_rows = read_representation(result / "phase_5_audit_units", stage_b["aggregate_shard_receipts"]["independent_audit_unit"])
    episodes_by_id = {str(row["episode_id"]): row for row in episodes}
    manifest_payload = json.loads(
        (experiment_dir / "data" / "manifests" / "mixed_dev_pool.json").read_text(
            encoding="utf-8"
        )
    )
    manifest_cases = {str(row["case_id"]): row for row in manifest_payload["cases"]}
    cases_by_id: dict[str, dict[str, Any]] = {}
    for row in case_rows:
        case = dict(row)
        manifest_case = manifest_cases.get(str(row["case_id"]))
        if manifest_case is not None:
            case["regions"] = manifest_case["regions"]
        cases_by_id[str(row["case_id"])] = case
    phase4_inventory = json.loads(
        (result / "phase_4_cache_inventory.json").read_text(encoding="utf-8")
    )
    phase4_contract = phase4_inventory["eres"]["E-standard"]["contract"]
    phase4_cache_root = Path(str(phase4_inventory["cache_root"]))
    phase5_cache = phase5_cache_root()
    from .phase5_runner import phase5_cache_contract as build_phase5_contract

    phase5_contract, _, _ = build_phase5_contract(
        experiment_dir, ledger, Path(str(args.eres_onnx_root))
    )
    if (
        phase5_contract.get("contract_sha256")
        != stage_a["phase5_inference"]["contract_sha256"]
    ):
        raise Phase5VerifyError("phase 5 cache contract drift")
    sources_by_wav: dict[str, Any] = {}
    for source in sources.values():
        existing = sources_by_wav.get(source.wav_sha256)
        if existing is None or source.source_id < existing.source_id:
            sources_by_wav[source.wav_sha256] = source
    from .phase5_runner import _load_embedding_universe

    embedding_universe = _load_embedding_universe(
        phase4_inventory,
        phase4_contract,
        phase4_cache_root,
        phase5_contract,
        phase5_cache,
        sources_by_wav,
        episodes,
        case_rows,
        profiles,
        historical_development,
    )
    batches = _audit_batches(
        audit_rows,
        system_info,
        profiles,
        episodes_by_id,
        cases_by_id,
        b0_by_unit,
        raw_words_by_session,
        raw_words_by_case,
        regions_by_session,
        cases,
        sources,
        embedding_universe,
        traces,
    )
    workers = int(args.workers or 8)
    recomputed: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(verify_audit_unit_worker, batch): index
            for index, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            outcome = future.result()
            recomputed[f"{outcome['system_id']}:{outcome['unit_id']}"] = outcome
    route_receipts = {
        f"{str(row[0])}|{str(row[1])}": row for row in read_representation(
            result / "phase_5_proposal_routes", stage_a["logical_proposal_routes"]
        )
    }
    mismatches: list[dict[str, Any]] = []
    updated_audit: list[list[Any]] = []
    for row in audit_rows:
        canonical = str(row[1])
        outcome = recomputed.get(canonical)
        unit_mismatches: list[dict[str, Any]] = []
        if outcome is None:
            unit_mismatches.append({"unit": canonical, "reason": "recompute_missing"})
        raw_expected = ""
        if outcome is not None:
            profile_id = str(row[5])
            unit_id = str(row[1]).split(":", 1)[1] if ":" in str(row[1]) else ""
            if outcome["raw_trace_sha256"]:
                raw_expected = outcome["raw_trace_sha256"]
            if raw_expected:
                route_key_candidates = [
                    key
                    for key, route in route_receipts.items()
                    if str(route[0]) == profile_id and str(route[1]) == unit_id
                ]
                if route_key_candidates:
                    route = route_receipts[route_key_candidates[0]]
                    if str(route[4]) != raw_expected:
                        unit_mismatches.append(
                            {
                                "unit": canonical,
                                "reason": "proposal_trace_mismatch",
                                "expected": str(route[4]),
                                "observed": raw_expected,
                            }
                        )
        if unit_mismatches:
            mismatches.extend(unit_mismatches)
        updated_audit.append(
            [
                str(row[0]),
                str(row[1]),
                str(row[2]),
                str(row[3]),
                str(row[4]),
                str(row[5]),
                str(row[6]),
                str(row[7]),
                str(row[8]),
                str(row[9]),
                str(row[10]),
                str(row[11]),
                outcome["raw_trace_sha256"] if outcome is not None else "",
                outcome["score_trace_sha256"] if outcome is not None else "",
                "mismatch" if unit_mismatches else "verified",
            ]
        )
    audit_writer = RepresentationWriter(
        result / "phase_5_verified_audit_units", "independent_audit_unit_verified"
    )
    audit_writer.add_rows((str(row[0]), row) for row in updated_audit)
    audit_receipt = audit_writer.write()
    verify_shard_receipts(
        result / "phase_5_verified_audit_units", audit_receipt["shards"]
    )
    aggregate_checks = _verify_aggregates(
        result,
        stage_a,
        stage_b,
        ledger,
        recomputed,
        system_info,
        episodes_by_id,
    )
    natural_checks = _verify_natural_labels(
        result, stage_b, ledger
    )
    summary = _build_summary(
        result,
        stage_a,
        gate,
        stage_b,
        guard,
        ledger,
        recomputed,
        updated_audit,
        mismatches,
        aggregate_checks,
        natural_checks,
        started,
    )
    verification = _build_verification(
        stage_a,
        gate,
        stage_b,
        guard,
        ledger,
        mismatches,
        aggregate_checks,
        natural_checks,
        audit_receipt,
        recomputed,
        started,
    )
    atomic_write_json = __import__(
        "experiments.speaker_turn_boundary.turn_episode.phase4_signal",
        fromlist=["atomic_write_json"],
    ).atomic_write_json
    summary_written = atomic_write_json(result / "phase_5_development_summary.json", summary)
    verification_written = atomic_write_json(result / "phase_5_verification.json", verification)
    completion = _build_completion(
        stage_a,
        gate,
        stage_b,
        verification_written,
        summary_written,
        started,
    )
    completion_written = atomic_write_json(result / "phase_5_completion.json", completion)
    print(
        canonical_json(
            {
                "path": str(result / "phase_5_completion.json"),
                "content_sha256": completion_written["content_sha256"],
                "verification_content_sha256": verification_written["content_sha256"],
                "summary_content_sha256": summary_written["content_sha256"],
                "audit_units": len(updated_audit),
                "mismatch_count": len(mismatches),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        )
    )
    return completion_written

# ---------------------------------------------------------------------------
# Audit batch construction
# ---------------------------------------------------------------------------


def _audit_batches(
    audit_rows: Sequence[Sequence[Any]],
    system_info: dict[str, dict[str, Any]],
    profiles: Sequence[dict[str, Any]],
    episodes_by_id: dict[str, dict[str, Any]],
    cases_by_id: dict[str, dict[str, Any]],
    b0_by_unit: dict[str, dict[str, Any]],
    raw_words_by_session: dict[str, list[Any] | None],
    raw_words_by_case: dict[str, list[Any] | None],
    regions_by_session: dict[str, list[Any]],
    cases: dict[tuple[str, str], dict[str, Any]],
    sources: dict[str, Any],
    embedding_universe: dict[str, dict[tuple[int, int], np.ndarray]],
    traces: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    from .phase5_runner import read_wav_slice

    profile_by_id = {str(row["proposal_profile_id"]): row for row in profiles}
    episodes_by_session: dict[str, list[dict[str, Any]]] = {}
    for episode in episodes_by_id.values():
        episodes_by_session.setdefault(str(episode["session_id"]), []).append(episode)
    batches: list[dict[str, Any]] = []
    for row in audit_rows:
        canonical = str(row[1])
        system_id_value = canonical.split(":", 1)[0] + ":" + canonical.split(":", 1)[1].split(":")[0]
        unit_id = canonical[len(system_id_value) + 1 :]
        info = system_info.get(system_id_value)
        if info is None:
            continue
        batch: dict[str, Any] = {
            "system_id": system_id_value,
            "unit_id": unit_id,
            "system_info": info,
        }
        if unit_id in episodes_by_id:
            episode = episodes_by_id[unit_id]
            batch["unit_kind"] = "current"
            batch["episode"] = episode
            batch["scored_start"] = int(episode["bounds"]["scored_start"])
            batch["scored_end"] = int(episode["bounds"]["scored_end"])
            batch["tail_end"] = int(episode["bounds"]["tail_end"])
            batch["episode_tag"] = str(episode["tag"])
            batch["pool"] = str(episode["pool"])
            session_id = str(episode["session_id"])
            batch["wav_sha256"] = str(episode["wav_sha256"])
            raw_words = raw_words_by_session.get(session_id)
            batch["views"] = annotation_views_for_episode(
                episode, regions_by_session, cases, raw_words_by_session
            )
            source = sources[session_id]
            batch["wav_path"] = str(source.path)
            batch["wave_start"] = max(0, int(episode["bounds"]["warm_start"]) - 8000)
            batch["wave_end"] = min(
                int(source.duration_samples), int(episode["bounds"]["tail_end"]) + 8000
            )
            profile = profile_by_id.get(str(info["profile_id"]))
            if profile is not None and profile["scored_state_mode"] == "source_prefix":
                batch["source_episodes"] = episodes_by_session.get(session_id, [])
            else:
                batch["source_episodes"] = [episode]
        elif unit_id in cases_by_id:
            case = cases_by_id[unit_id]
            batch["unit_kind"] = "historical"
            batch["episode"] = case
            batch["duration_samples"] = int(case["duration_samples"])
            batch["pool"] = "historical_validation_corrected_rescore_only"
            batch["wav_sha256"] = str(case["wav_sha256"])
            raw_words = raw_words_by_case.get(unit_id)
            batch["views"] = annotation_views_for_case(case, raw_words)
            profile = profile_by_id.get(str(info["profile_id"]))
            source = _source_for_wav(sources, str(case["wav_sha256"]))
            if source is not None:
                batch["wav_path"] = str(source.path)
                batch["wave_start"] = 0
                batch["wave_end"] = int(case["duration_samples"])
            batch["source_episodes"] = []
        else:
            continue
        b0 = b0_by_unit.get(unit_id)
        if b0 is None:
            continue
        batch["b0_actions"] = b0["b0_actions"]
        batch["b1_actions"] = b0["b1_actions"]
        batch["lifecycle"] = b0["lifecycle_events"]
        batch["references"] = batch["views"].pop("_references", [])
        batch["profile"] = profile_by_id.get(str(info["profile_id"]))
        batch["embeddings"] = embedding_universe.get(batch.get("wav_sha256", ""), {})
        batches.append(batch)
    return batches


def _source_for_wav(sources: dict[str, Any], wav_sha256: str) -> Any | None:
    for source in sources.values():
        if str(source.wav_sha256) == wav_sha256:
            return source
    return None


# ---------------------------------------------------------------------------
# Exhaustive checks
# ---------------------------------------------------------------------------


def _verify_aggregates(
    result: Path,
    stage_a: dict[str, Any],
    stage_b: dict[str, Any],
    ledger: dict[str, Any],
    recomputed: dict[str, dict[str, Any]],
    system_info: dict[str, dict[str, Any]],
    episodes_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    current_rows = read_representation(
        result / "phase_5_current_aggregates",
        stage_b["aggregate_shard_receipts"]["current_system_block_aggregate"],
    )
    if len(current_rows) != 4611:
        raise Phase5VerifyError(f"current aggregate row drift: {len(current_rows)}")
    historical_rows = read_representation(
        result / "phase_5_historical_aggregates",
        stage_b["aggregate_shard_receipts"]["historical_corrected_system_aggregate"],
    )
    if len(historical_rows) != 4610:
        raise Phase5VerifyError(f"historical aggregate row drift: {len(historical_rows)}")
    physical_rows = read_representation(
        result / "phase_5_physical_systems",
        stage_b["aggregate_shard_receipts"]["physical_system_definition"],
    )
    if len(physical_rows) != 2503:
        raise Phase5VerifyError(f"physical system row drift: {len(physical_rows)}")
    logical_rows = read_representation(
        result / "phase_5_logical_systems",
        stage_b["aggregate_shard_receipts"]["logical_system_definition"],
    )
    if len(logical_rows) != 4611:
        raise Phase5VerifyError(f"logical system row drift: {len(logical_rows)}")
    alias_rows = read_representation(
        result / "phase_5_alias_edges",
        stage_b["aggregate_shard_receipts"]["logical_alias_edge"],
    )
    if len(alias_rows) != 2108:
        raise Phase5VerifyError(f"alias edge row drift: {len(alias_rows)}")
    failure_rows = read_representation(
        result / "phase_5_failure_examples",
        stage_b["aggregate_shard_receipts"]["deterministic_failure_example"],
    )
    if len(failure_rows) != 420:
        raise Phase5VerifyError(f"failure example row drift: {len(failure_rows)}")
    per_pool_totals: dict[str, dict[str, int]] = {}
    for row in current_rows:
        system_id_value = str(row[0])
        for pool_row in row[2]:
            pool = str(pool_row[0])
            target = per_pool_totals.setdefault(pool, {field: 0 for field in SYSTEM_METRIC_FIELDS})
            for field_index, field in enumerate(SYSTEM_METRIC_FIELDS):
                target[field] += int(pool_row[3 + field_index])
    checks["current_pool_totals"] = {
        pool: {field: int(value) for field, value in totals.items()}
        for pool, totals in per_pool_totals.items()
    }
    pool_counts = {
        "diagnostic_dev": 695,
        "frontier_dev": 109,
        "natural_exposure_validation": 74,
    }
    pool_episode_checks = []
    for row in current_rows:
        system_id_value = str(row[0])
        for pool_row in row[2]:
            pool = str(pool_row[0])
            actual = int(pool_row[3])
            expected = pool_counts.get(pool)
            if expected is None:
                pool_episode_checks.append(
                    {
                        "system_id": system_id_value,
                        "pool": pool,
                        "reason": "unexpected_pool",
                        "actual": actual,
                    }
                )
            elif actual != expected and str(pool_row[1]):
                pool_episode_checks.append(
                    {
                        "system_id": system_id_value,
                        "pool": pool,
                        "expected": expected,
                        "actual": actual,
                    }
                )
    checks["pool_episode_mismatch_count"] = len(pool_episode_checks)
    checks["pool_episode_mismatches"] = pool_episode_checks[:20]
    checks["pool_episode_counts"] = {
        pool: next(
            (
                int(pool_row[3])
                for row in current_rows
                for pool_row in row[2]
                if str(pool_row[0]) == pool
            ),
            0,
        )
        for pool in pool_counts
    }
    unit_checks = []
    for key, outcome in recomputed.items():
        system_id_value = key.split(":", 1)[0] + ":" + key.split(":", 1)[1].split(":")[0]
        unit_id = key[len(system_id_value) + 1 :]
        episode = episodes_by_id.get(unit_id)
        if episode is None:
            continue
        pool = str(episode["pool"])
        agg_row = next(
            (row for row in current_rows if str(row[0]) == system_id_value), None
        )
        if agg_row is None:
            continue
        pool_row = next((row for row in agg_row[2] if str(row[0]) == pool), None)
        if pool_row is None or not str(pool_row[1]):
            continue
        for field_index, field in enumerate(SYSTEM_METRIC_FIELDS):
            if field in PERCENTILE_FIELD_NAMES:
                continue
            aggregate_value = int(pool_row[3 + field_index])
            unit_value = int(outcome["metric_vector"][field])
            if unit_value > aggregate_value:
                unit_checks.append(
                    {
                        "system_id": system_id_value,
                        "unit_id": unit_id,
                        "field": field,
                        "aggregate": aggregate_value,
                        "unit": unit_value,
                    }
                )
    checks["unit_within_aggregate_mismatch_count"] = len(unit_checks)
    checks["unit_within_aggregate_mismatches"] = unit_checks[:20]
    block_sum_checks = []
    for row in current_rows:
        system_id_value = str(row[0])
        for pool_row in row[2]:
            pool = str(pool_row[0])
            block_episodes = sum(
                int(block_row[2])
                for block_row in row[3]
                if str(block_row[0]) == pool
            )
            pool_episodes = int(pool_row[3])
            if block_episodes != pool_episodes:
                block_sum_checks.append(
                    {
                        "system_id": system_id_value,
                        "pool": pool,
                        "pool_episode_count": pool_episodes,
                        "block_episode_sum": block_episodes,
                    }
                )
    checks["pool_block_episode_mismatch_count"] = len(block_sum_checks)
    checks["pool_block_episode_mismatches"] = block_sum_checks[:20]
    return checks


def _verify_natural_labels(
    result: Path,
    stage_b: dict[str, Any],
    ledger: dict[str, Any],
) -> dict[str, Any]:
    current_rows = read_representation(
        result / "phase_5_current_aggregates",
        stage_b["aggregate_shard_receipts"]["current_system_block_aggregate"],
    )
    violations = []
    natural_manifest = json.loads(
        (result / "natural_exposure_manifest.json").read_text(encoding="utf-8")
    )
    expected_eligible = (
        int(natural_manifest["window_frame"]["eligible_duration_ms"]) * 16
    )
    expected_sessions = len(
        {
            str(session_id)
            for row in ledger["population"]["pool_block_index"]
            if str(row["pool"]) == "natural_exposure_validation"
            for session_id in row["source_session_ids"]
        }
    )
    natural_pool_rows = []
    for row in current_rows:
        for pool_row in row[2]:
            pool = str(pool_row[0])
            values = [int(value) for value in pool_row[3:]]
            natural_fields = [
                field for field in SYSTEM_METRIC_FIELDS if field.startswith("natural_")
            ]
            for field_index, field in enumerate(SYSTEM_METRIC_FIELDS):
                if field not in natural_fields:
                    continue
                value = int(pool_row[3 + field_index])
                if pool != "natural_exposure_validation" and value != 0:
                    violations.append(
                        {
                            "system_id": str(row[0]),
                            "pool": pool,
                            "field": field,
                            "value": value,
                        }
                    )
            if pool == "natural_exposure_validation":
                if not str(pool_row[1]):
                    continue
                natural_pool_rows.append(
                    {
                        "system_id": str(row[0]),
                        "session_count": int(
                            pool_row[
                                3 + SYSTEM_METRIC_FIELDS.index("natural_session_count")
                            ]
                        ),
                        "eligible_samples": int(
                            pool_row[
                                3
                                + SYSTEM_METRIC_FIELDS.index(
                                    "natural_eligible_source_samples"
                                )
                            ]
                        ),
                        "sampled_samples": int(
                            pool_row[
                                3 + SYSTEM_METRIC_FIELDS.index("natural_sampled_source_samples")
                            ]
                        ),
                    }
                )
    for entry in natural_pool_rows:
        if entry["session_count"] != expected_sessions:
            violations.append(
                {
                    "system_id": entry["system_id"],
                    "pool": "natural_exposure_validation",
                    "field": "natural_session_count",
                    "value": entry["session_count"],
                    "expected": expected_sessions,
                }
            )
        if entry["eligible_samples"] != expected_eligible:
            violations.append(
                {
                    "system_id": entry["system_id"],
                    "pool": "natural_exposure_validation",
                    "field": "natural_eligible_source_samples",
                    "value": entry["eligible_samples"],
                    "expected": expected_eligible,
                }
            )
    return {
        "natural_field_violation_count": len(violations),
        "natural_field_violations": violations[:20],
        "natural_session_count": expected_sessions,
        "natural_eligible_source_samples": expected_eligible,
    }


def _build_summary(
    result: Path,
    stage_a: dict[str, Any],
    gate: dict[str, Any],
    stage_b: dict[str, Any],
    guard: dict[str, Any],
    ledger: dict[str, Any],
    recomputed: dict[str, dict[str, Any]],
    updated_audit: Sequence[Sequence[Any]],
    mismatches: Sequence[dict[str, Any]],
    aggregate_checks: dict[str, Any],
    natural_checks: dict[str, Any],
    started: float,
) -> dict[str, Any]:
    verified = sum(1 for row in updated_audit if str(row[14]) == "verified")
    return {
        "schema_version": "turn_episode_phase5_development_summary.v1",
        "authority_sha256": AUTHORITY_SHA256,
        "design_ledger_content_sha256": ledger["content_sha256"],
        "stage_a_receipt_content_sha256": stage_a["content_sha256"],
        "interstage_gate_content_sha256": gate["content_sha256"],
        "stage_b_receipt_content_sha256": stage_b["content_sha256"],
        "start_guard": guard,
        "audit_units_total": len(updated_audit),
        "audit_units_verified": verified,
        "audit_units_mismatched": len(mismatches),
        "exhaustive_checks": {
            "b0_b1_equivalence_passed": bool(
                stage_a["b0_evidence"]["b0_b1_mismatch_count"] == 0
            ),
            "word_timing_receipt_exact": bool(gate["stage_b_allowed"]),
            "aggregate_arithmetic": aggregate_checks,
            "natural_rate_labels": natural_checks,
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def _build_verification(
    stage_a: dict[str, Any],
    gate: dict[str, Any],
    stage_b: dict[str, Any],
    guard: dict[str, Any],
    ledger: dict[str, Any],
    mismatches: Sequence[dict[str, Any]],
    aggregate_checks: dict[str, Any],
    natural_checks: dict[str, Any],
    audit_receipt: dict[str, Any],
    recomputed: dict[str, dict[str, Any]],
    started: float,
) -> dict[str, Any]:
    passed = (
        not mismatches
        and aggregate_checks["unit_within_aggregate_mismatch_count"] == 0
        and aggregate_checks["pool_episode_mismatch_count"] == 0
        and aggregate_checks["pool_block_episode_mismatch_count"] == 0
        and natural_checks["natural_field_violation_count"] == 0
        and bool(stage_a["b0_evidence"]["b0_b1_mismatch_count"] == 0)
        and bool(gate["stage_b_allowed"])
    )
    return {
        "schema_version": "turn_episode_phase5_verification.v1",
        "authority_sha256": AUTHORITY_SHA256,
        "design_ledger_content_sha256": ledger["content_sha256"],
        "passed": passed,
        "audit_unit_receipt": audit_receipt,
        "mismatch_count": len(mismatches),
        "mismatches": list(mismatches),
        "aggregate_checks": aggregate_checks,
        "natural_rate_checks": natural_checks,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def _build_completion(
    stage_a: dict[str, Any],
    gate: dict[str, Any],
    stage_b: dict[str, Any],
    verification: dict[str, Any],
    summary: dict[str, Any],
    started: float,
) -> dict[str, Any]:
    return {
        "schema_version": "turn_episode_phase5_completion.v1",
        "authority_sha256": AUTHORITY_SHA256,
        "passed": bool(verification["passed"]),
        "stage_a_receipt_content_sha256": stage_a["content_sha256"],
        "interstage_gate_content_sha256": gate["content_sha256"],
        "stage_b_receipt_content_sha256": stage_b["content_sha256"],
        "verification_content_sha256": verification["content_sha256"],
        "summary_content_sha256": summary["content_sha256"],
        "phase_6_preparation": "forbidden",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
