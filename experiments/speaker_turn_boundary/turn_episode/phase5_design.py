from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .phase4_design import ceil_grid, load_synthetic_cases
from .phase4_signal import _source_maps, atomic_write_json, load_inputs

AUTHORITY_SHA256 = "e3efdd9410a84bd343da5ba41d634ceec2d54626e1b512f41e410c0668329e36"
PHASE4_ACCEPTED_CANDIDATE = "5edfa67f7bb73c352b15459fdde018b196b5b5ac"
PHASE4_BUNDLE_SHA256 = "a6afa3dc946815c162ee18d09b1c7ad3ad08e252f7286c110b37a685fe2b1759"
PHASE4_COMPLETION_CONTENT_SHA256 = (
    "db75772938fc4a59f21784e9fbc279ad3003bffc72b32594d7844fec8a28f14c"
)
PHASE4_VERIFICATION_CONTENT_SHA256 = (
    "f8ba0e6498d2bc6d87854b6bdaefb5f7f15a7263ea9f98c399cd8b56d8bab51c"
)
PHASE4_DISPOSITION_CONTENT_SHA256 = (
    "669f6d4200832816b7beee03161cac2b97ec2594af00bb78df878766081bc5bf"
)
PHASE4_STATE_CONTENT_SHA256 = "23813a61e3ed11495f155f59114c6351166e2a2f256c34e25902fcd30e0e022e"
INPUT_IDENTITIES = {
    "episode_manifest_dev.json": {
        "byte_sha256": "a5df5e56c2917ba174fca7892fbaa092f2b57705cad7a77bf9a347ba94cddfee",
        "content_sha256": "deb1713cd581c93cc13c47103643041c7551993985379869cfa0ca9a407dff68",
    },
    "natural_exposure_manifest.json": {
        "byte_sha256": "42b21562222f19fc880b93a40a3a999b122ec4afd00a3c7deeab46b9bc482e1c",
        "content_sha256": "e7c8562602685925e4ccb1964801d384c555813d3f565d5a67c7750770b088f3",
    },
    "proposal_contract.json": {
        "byte_sha256": "982c54a4164335e9be5e1823deb3f8b51915cf90f07845beb0e11f8abd22fd4b",
        "content_sha256": "33d7133a49343a6d5377c9f52bf8b3c90ea843551546248185d98a66ade25b72",
    },
    "fusion_contract.json": {
        "byte_sha256": "bfda0c3c0ea7b6613ded79e9639692a33449dcf34202b1f2a5e7ec14c45f9873",
        "content_sha256": "3c20e381c62026dc36038416b857e9f94e56abddcf0156bf07f77a64a4a020a9",
    },
    "phase_4_completion.json": {
        "byte_sha256": "368a5c23a30e10f1884fd3797166b23ee93df0a1d0f84fc7006010b17fdec565",
        "content_sha256": PHASE4_COMPLETION_CONTENT_SHA256,
    },
    "phase_4_eres_signal_report.json": {
        "byte_sha256": "473654f63cede6b33078725fb744fc1468f3b5354a6d84856fd802ee477ad135",
        "content_sha256": "f81e7c44232a0d5e176beb88ca071710bf761254c54da1d4ce7ebff8bf43a727",
    },
    "phase_4_signal_disposition.json": {
        "byte_sha256": "f9e799bd1f78aa45b25cb913928584ba0d63d07cfbf6978495d387da2ff5a9aa",
        "content_sha256": PHASE4_DISPOSITION_CONTENT_SHA256,
    },
    "phase_4_state_equivalence.json": {
        "byte_sha256": "033bfa6605408dc32838e0b3339a5fc36dd6769053176e4b1fc1d97a805f4878",
        "content_sha256": PHASE4_STATE_CONTENT_SHA256,
    },
    "phase_4_verification.json": {
        "byte_sha256": "dda1f1c1d9f51e9eec919e31f31635f07d22e0e84369cf87d6a755186ab12740",
        "content_sha256": PHASE4_VERIFICATION_CONTENT_SHA256,
    },
}
HISTORICAL_ROWS_BYTE_SHA256 = "6fc01ce8f679aad4e4d9c6d5c45a1d0552f0ba030e46306a7a308466155c8f19"
HISTORICAL_LEDGER_BYTE_SHA256 = "8bc7e6dccff46e350bc72b4483bb381b2eefa796bc47cb446dfc7b76fb5c4afa"
HISTORICAL_MANIFEST_BYTE_SHA256 = "1221176c92f50a2b096e4cd64d5da0168527918e3fba539273c614eabf07a398"
HISTORICAL_MANIFEST_CONTENT_SHA256 = (
    "8369c51a82777a01814a35bac34301252842492fb12583b5f20733d75d92ca8e"
)
HISTORICAL_WORD_FILE_SHA256 = {
    "ami_ES2003a": {
        "ES2003a.A.words.xml": "8ee1e2ba5ab4421e16ebf42d0e221247a83e174fe34aa06e3aadd7a08540e470",
        "ES2003a.B.words.xml": "d9af4424014f9e2ae55e1a33fc7fb95a2352903453e3afb51118845cea4c304d",
        "ES2003a.C.words.xml": "9393f7af2a0cd95da35d306e4861e2513544c54059b11515ff311cd9b051cff4",
        "ES2003a.D.words.xml": "bdcf987020238618bed2941bc906d31aa61aec6732e3104d16f11b7f1a104384",
    },
    "ami_IS1008a": {
        "IS1008a.A.words.xml": "db744e8ca1f45f794e288c5521ab5dfbc2b2f8e5e9f814dd5300053cdab09a2d",
        "IS1008a.B.words.xml": "4a92c5b967c6a32fdfc4a288b7a61064d7259710b6afb32c27e49e4bd146f322",
        "IS1008a.C.words.xml": "d61e501c37877f731058889d799e7ff5a3d19494b8a22f12c7f0f50a87c62758",
        "IS1008a.D.words.xml": "e7f475b606a950de682d275b912821f3d3d612136ce62bd41bc685146767f40b",
    },
}
HISTORICAL_RUN_CONTRACT_BYTE_SHA256 = (
    "099c7a5dc6ce38916b5937b29527397534549ac28030e73043574ff89c44145f"
)
HISTORICAL_PREFLIGHT_BYTE_SHA256 = (
    "87d684a1bf48b35afc796d390ad33ae814645d05c1f5a80195799e6b707042b4"
)
HISTORICAL_B0_BYTE_SHA256 = "769a987b0b0f05cc1b0b32cf61a20e64472823ad4c91e852589240825600d3c7"
CLUSTER_DEBOUNCE_MS = (0, 100, 250)
CLUSTER_RADIUS_MS = (250, 500)
REFRACTORY_MS = (0, 250, 500)
REPRESENTATIVES = ("first", "max_confidence")
VAD_RADIUS_MS = (250, 500)
SILENCE_ASSOCIATION = (False, True)
CONTROL_KINDS = (
    "uniform_vad_active",
    "causal_energy_change_peak",
    "within_vad_active_position_shuffle",
)
LADDER_STAGES = (
    "naive_proposal_as_cut",
    "clustering_only",
    "clustering_plus_refractory",
    "plus_vad_association",
    "full_hard_soft_fusion",
)
POLICY_BENCHMARK_FILENAME = "phase_5_policy_benchmark.json"
STORAGE_BENCHMARK_FILENAME = "phase_5_storage_benchmark.json"
WORD_TIMING_RECEIPT_FIELDS = (
    "unit_id",
    "annotation_source_identity",
    "word_record_sha256",
    "raw_word_record_count",
    "word_interval_count",
    "word_timing_observable",
    "lexical_scoring_disposition",
)
COMPACT_SIGNAL_EXTRACTOR_IDS = (
    "eres_adjacent_change.v1:E-standard:W8000:H500",
    "eres_prototype_change.v1:E-standard:prototype_memory_4:W8000:S1600:H500",
    "eres_prototype_change.v1:E-standard:prototype_memory_4:W8000:S4000:H500",
)
COMPACT_PROFILE_IDS = (
    "phase4_native:adjacent_direct:E-standard:W8000:S1600:T500",
    "phase4_native:adjacent_direct:E-standard:W8000:S4000:T500",
    "phase4_native:prototype_memory_4:E-standard:W8000:S1600:T500",
    "phase4_native:prototype_memory_4:E-standard:W8000:S4000:T500",
)
INDEPENDENT_AUDIT_SAMPLE_SIZE = 2048
EXECUTION_PLANNING_CEILING_HOURS = 3.0


class Phase5DesignError(RuntimeError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows_sha256(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(canonical_json(row).encode("utf-8") + b"\n")
    return digest.hexdigest()


def verify_self_hash(payload: dict[str, Any]) -> None:
    expected = str(payload.get("content_sha256") or "")
    body = dict(payload)
    body.pop("content_sha256", None)
    actual = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
    if actual != expected:
        raise Phase5DesignError("self hash mismatch")


def read_pinned_json(path: Path, identity: dict[str, str]) -> dict[str, Any]:
    if sha256_file(path) != identity["byte_sha256"]:
        raise Phase5DesignError(f"byte hash drift: {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("content_sha256") != identity["content_sha256"]:
        raise Phase5DesignError(f"content hash drift: {path.name}")
    return payload


def phase4_inputs(result_dir: Path) -> dict[str, dict[str, Any]]:
    return {
        name: read_pinned_json(result_dir / name, identity)
        for name, identity in INPUT_IDENTITIES.items()
    }


def state_dispositions(state: dict[str, Any]) -> dict[tuple[str, str], str]:
    return {
        (str(row["checkpoint"]), str(row["profile_class"])): str(row["disposition"])
        for row in state["eres_profile_classes"]
    }


def native_profiles(state: dict[str, Any]) -> list[dict[str, Any]]:
    dispositions = state_dispositions(state)
    profiles: list[dict[str, Any]] = []
    for step in (1600, 4000):
        profiles.append(
            {
                "proposal_profile_id": f"phase4_native:adjacent_direct:E-standard:W8000:S{step}:T500",
                "origin": "accepted_phase4_native_profile",
                "legacy_profile_id": None,
                "family": "eres2netv2",
                "checkpoint": "E-standard",
                "profile_class": "adjacent",
                "window_samples": 8000,
                "step_samples": step,
                "proposal_kind": "speaker_change_unknown",
                "confidence_semantics_id": "eres_adjacent_change.v1:one_minus_cosine",
                "proposal_threshold": {"field": "change_score", "operator": ">", "value": 0.5},
                "confirmation": "direct_each_qualifying_probe",
                "state_disposition": dispositions[("E-standard", "adjacent")],
                "scored_state_mode": "episode_reset",
                "semantic_alias": None,
            }
        )
        profiles.append(
            {
                "proposal_profile_id": f"phase4_native:prototype_memory_4:E-standard:W8000:S{step}:T500",
                "origin": "accepted_phase4_native_profile",
                "legacy_profile_id": None,
                "family": "eres2netv2",
                "checkpoint": "E-standard",
                "profile_class": "prototype_memory_4",
                "window_samples": 8000,
                "step_samples": step,
                "proposal_kind": "speaker_change_unknown",
                "confidence_semantics_id": "eres_prototype_memory_4.v1:one_minus_cosine",
                "proposal_threshold": {"field": "change_score", "operator": ">", "value": 0.5},
                "confirmation": "two_probe_mutual_cosine_gte_0.50",
                "state_disposition": dispositions[("E-standard", "prototype_memory_4")],
                "scored_state_mode": "source_prefix",
                "semantic_alias": None,
            }
        )
    profiles.sort(key=lambda row: str(row["proposal_profile_id"]))
    if len(profiles) != 4:
        raise Phase5DesignError("compact Phase 4 native profile universe drift")
    return profiles


def proposal_profiles(
    experiment_dir: Path,
    disposition: dict[str, Any],
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    family = disposition["families"]["eres2netv2"]
    if family["disposition"] != "signal_go":
        raise Phase5DesignError("ERes Phase 5 envelope is not signal_go")
    eligible = set(family["eligible_go_ids"])
    if not set(COMPACT_SIGNAL_EXTRACTOR_IDS).issubset(eligible):
        raise Phase5DesignError("compact Phase 4 signal-go identity drift")
    native_by_id = {str(row["proposal_profile_id"]): row for row in native_profiles(state)}
    if not set(COMPACT_PROFILE_IDS).issubset(native_by_id):
        raise Phase5DesignError("compact proposal-profile identity drift")
    profiles = [native_by_id[profile_id] for profile_id in COMPACT_PROFILE_IDS]
    if any(
        row["checkpoint"] != "E-standard"
        or row["window_samples"] != 8000
        or row["step_samples"] not in (1600, 4000)
        or row["profile_class"] not in ("adjacent", "prototype_memory_4")
        for row in profiles
    ):
        raise Phase5DesignError("compact proposal-profile scope drift")
    if len({row["proposal_profile_id"] for row in profiles}) != 4:
        raise Phase5DesignError("compact proposal-profile identity collision")
    return profiles


def load_populations(
    experiment_dir: Path,
    inputs: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, str]]:
    result_dir = experiment_dir / "results" / "turn_episode_v1"
    dev = inputs["episode_manifest_dev.json"]["episodes"]
    natural = inputs["natural_exposure_manifest.json"]["episodes"]
    episodes = list(dev) + list(natural)
    counts = Counter(str(row["pool"]) for row in episodes)
    expected = {
        "diagnostic_dev": 695,
        "frontier_dev": 109,
        "natural_exposure_validation": 74,
    }
    if dict(counts) != expected or len({row["episode_id"] for row in episodes}) != 878:
        raise Phase5DesignError("Phase 5 population drift")
    inventory = json.loads((result_dir / "coverage_inventory.json").read_text(encoding="utf-8"))
    details = {
        str(row["session_id"]): row
        for row in map(
            json.loads,
            (result_dir / "coverage_inventory_details.jsonl")
            .read_text(encoding="utf-8")
            .splitlines(),
        )
    }
    cases = load_synthetic_cases(experiment_dir / "data" / "manifests")
    sources, source_by_episode = _source_maps(
        experiment_dir,
        episodes,
        cases,
        inventory,
        details,
    )
    source_summary = {
        "source_count": len(sources),
        "public_source_count": sum(source.public for source in sources.values()),
        "synthetic_source_count": sum(not source.public for source in sources.values()),
    }
    if source_summary != {
        "source_count": 626,
        "public_source_count": 20,
        "synthetic_source_count": 606,
    }:
        raise Phase5DesignError("Phase 5 source universe drift")
    return episodes, source_summary, source_by_episode


def pool_block_index(
    episodes: list[dict[str, Any]], coverage: dict[str, Any]
) -> list[dict[str, Any]]:
    session_to_block: dict[str, str] = {}
    for block_id, sessions in coverage["group_graph"]["component_sessions"].items():
        for session_id in sessions:
            session_to_block[str(session_id)] = str(block_id)
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for episode in episodes:
        pool = str(episode["pool"])
        session_id = str(episode["session_id"])
        block_id = session_to_block.get(session_id)
        if block_id is None:
            synthetic_family = session_id.split(":", 1)[0]
            if synthetic_family not in (
                "ls_dev",
                "ls_held_out_clean",
                "ls_held_out_other",
                "mixed_dev_pool",
            ):
                raise Phase5DesignError(f"unmapped statistical block: {session_id}")
            block_id = f"synthetic:{synthetic_family}"
        key = (pool, block_id)
        row = grouped.setdefault(
            key,
            {
                "pool": pool,
                "block_id": block_id,
                "episode_count": 0,
                "source_session_ids": set(),
            },
        )
        row["episode_count"] += 1
        row["source_session_ids"].add(session_id)
    pool_order = {
        "diagnostic_dev": 0,
        "frontier_dev": 1,
        "natural_exposure_validation": 2,
    }
    rows = []
    for (pool, block_id), row in sorted(
        grouped.items(), key=lambda item: (pool_order[item[0][0]], item[0][1])
    ):
        rows.append(
            {
                "pool": pool,
                "pool_block_id": f"{pool}:{hashlib.sha256(block_id.encode('utf-8')).hexdigest()[:24]}",
                "statistical_block_id": block_id,
                "episode_count": int(row["episode_count"]),
                "source_session_ids": sorted(row["source_session_ids"]),
            }
        )
    counts = Counter(row["pool"] for row in rows)
    if dict(counts) != {
        "diagnostic_dev": 13,
        "frontier_dev": 10,
        "natural_exposure_validation": 20,
    }:
        raise Phase5DesignError("pool-block universe drift")
    return rows


def required_windows(
    episodes: list[dict[str, Any]],
    source_by_episode: dict[str, str],
    sources: dict[str, Any],
    profiles: list[dict[str, Any]],
) -> dict[str, set[tuple[int, int]]]:
    by_wav: dict[str, set[tuple[int, int]]] = defaultdict(set)
    source_lookup = {source.source_id: source for source in sources.values()}
    stateful_grid = sorted(
        {
            (int(row["window_samples"]), int(row["step_samples"]))
            for row in profiles
            if row["scored_state_mode"] == "source_prefix"
        }
    )
    adjacent_grid = sorted(
        {
            (int(row["window_samples"]), int(row["step_samples"]))
            for row in profiles
            if row["profile_class"] == "adjacent"
        }
    )
    source_maximum: dict[str, int] = {}
    for episode in episodes:
        source = source_lookup[source_by_episode[str(episode["episode_id"])]]
        bounds = episode["bounds"]
        source_maximum[source.wav_sha256] = max(
            source_maximum.get(source.wav_sha256, 0),
            int(bounds["tail_end"]),
        )
        for window, step in adjacent_grid:
            low = int(bounds["warm_start"]) + window
            high = int(bounds["tail_end"]) - window
            for boundary in range(ceil_grid(low, step), high + 1, step):
                by_wav[source.wav_sha256].add((boundary - window, boundary))
                by_wav[source.wav_sha256].add((boundary, boundary + window))
    for wav_sha256, maximum_tail in source_maximum.items():
        for window, step in stateful_grid:
            first = ceil_grid(window, step)
            for end in range(first, maximum_tail + 1, step):
                by_wav[wav_sha256].add((end - window, end))
    return dict(by_wav)


def inclusive_grid_count(first: int, last: int, step: int) -> int:
    if first > last:
        return 0
    return (last - first) // step + 1


def proposal_workload(
    episodes: list[dict[str, Any]],
    source_by_episode: dict[str, str],
    sources: dict[str, Any],
    profiles: list[dict[str, Any]],
) -> dict[str, Any]:
    source_lookup = {source.source_id: source for source in sources.values()}
    source_maximum: dict[str, int] = {}
    for episode in episodes:
        source_id = source_by_episode[str(episode["episode_id"])]
        source_maximum[source_id] = max(
            source_maximum.get(source_id, 0), int(episode["bounds"]["tail_end"])
        )
    source_prefix_steps = 0
    source_prefix_logical_positions = 0
    episode_reset_steps = 0
    source_prefix_passes = 0
    episode_reset_traces = 0
    maximum_source_prefix: dict[str, Any] | None = None
    maximum_episode: dict[str, Any] | None = None
    maximum_emittable_positions = 0
    for profile in profiles:
        window = int(profile["window_samples"])
        step = int(profile["step_samples"])
        if profile["scored_state_mode"] == "source_prefix":
            for source_id, tail_end in source_maximum.items():
                first = ceil_grid(window, step)
                count = inclusive_grid_count(first, tail_end, step)
                source_prefix_steps += count
                source_prefix_passes += 1
                candidate = {
                    "proposal_profile_id": str(profile["proposal_profile_id"]),
                    "source_id": source_id,
                    "wav_sha256": str(source_lookup[source_id].wav_sha256),
                    "tail_end_sample": tail_end,
                    "probe_step_count": count,
                }
                if maximum_source_prefix is None or count > int(
                    maximum_source_prefix["probe_step_count"]
                ):
                    maximum_source_prefix = candidate
            for episode in episodes:
                bounds = episode["bounds"]
                first = ceil_grid(window, step)
                eligible_first = ceil_grid(max(first, int(bounds["warm_start"])), step)
                eligible_count = inclusive_grid_count(eligible_first, int(bounds["tail_end"]), step)
                if eligible_first == first and eligible_count:
                    eligible_count -= 1
                maximum_emittable_positions = max(maximum_emittable_positions, eligible_count)
                source_prefix_logical_positions += eligible_count
        else:
            for episode in episodes:
                bounds = episode["bounds"]
                first = ceil_grid(int(bounds["warm_start"]) + window, step)
                last = int(bounds["tail_end"]) - window
                count = inclusive_grid_count(first, last, step)
                episode_reset_steps += count
                episode_reset_traces += 1
                candidate = {
                    "proposal_profile_id": str(profile["proposal_profile_id"]),
                    "episode_id": str(episode["episode_id"]),
                    "probe_step_count": count,
                }
                if maximum_episode is None or count > int(maximum_episode["probe_step_count"]):
                    maximum_episode = candidate
                maximum_emittable_positions = max(maximum_emittable_positions, count)
    if maximum_source_prefix is None or maximum_episode is None:
        raise Phase5DesignError("proposal workload class coverage drift")
    if maximum_emittable_positions > 320:
        raise Phase5DesignError("proposal count sentinel does not cover the current population")
    return {
        "logical_profile_episode_trace_count": len(profiles) * len(episodes),
        "source_prefix_profile_count": sum(
            row["scored_state_mode"] == "source_prefix" for row in profiles
        ),
        "episode_reset_profile_count": sum(
            row["scored_state_mode"] != "source_prefix" for row in profiles
        ),
        "source_prefix_physical_pass_count": source_prefix_passes,
        "episode_reset_physical_trace_count": episode_reset_traces,
        "source_prefix_probe_step_count": source_prefix_steps,
        "source_prefix_logical_emittable_position_count": source_prefix_logical_positions,
        "episode_reset_probe_step_count": episode_reset_steps,
        "episode_reset_logical_emittable_position_count": episode_reset_steps,
        "total_logical_emittable_position_count": source_prefix_logical_positions
        + episode_reset_steps,
        "total_physical_probe_step_count": source_prefix_steps + episode_reset_steps,
        "maximum_source_prefix_trace": maximum_source_prefix,
        "maximum_episode_reset_trace": maximum_episode,
        "maximum_emittable_proposal_position_count": maximum_emittable_positions,
        "proposal_count_sentinel_ceiling": 320,
        "source_prefix_execution_key_fields": [
            "proposal_profile_id",
            "source_id",
            "wav_sha256",
            "maximum_tail_end_sample",
            "embedding_window_universe_sha256",
            "phase5_proposals.py_sha256",
        ],
        "episode_route_key_fields": [
            "source_prefix_execution_id",
            "episode_id",
            "audio_epoch",
            "warm_start",
            "tail_end",
        ],
        "routing_rule": "execute each source-prefix profile/source state path once through the maximum required tail, retain content-addressed state/proposal snapshots at episode coordinates, then re-identify routed proposals in the episode epoch without replaying the prefix",
    }


def window_ledger(
    required: dict[str, set[tuple[int, int]]],
    phase4_windows: dict[str, set[tuple[int, int]]],
) -> dict[str, Any]:
    rows = [
        {"wav_sha256": wav, "start": start, "end": end}
        for wav, windows in required.items()
        for start, end in windows
    ]
    rows.sort(key=lambda row: (row["wav_sha256"], row["start"], row["end"]))
    hit_count = sum(
        len(windows & phase4_windows.get(wav, set())) for wav, windows in required.items()
    )
    total = len(rows)
    return {
        "unique_window_count": total,
        "window_rows_sha256": rows_sha256(rows),
        "reusable_window_count": hit_count,
        "new_inference_window_count": total - hit_count,
        "checkpoint_count": 1,
        "checkpoints": ["E-standard"],
        "total_checkpoint_window_jobs": total,
        "reusable_checkpoint_window_jobs": hit_count,
        "new_inference_checkpoint_window_jobs": total - hit_count,
    }


def merge_window_sets(
    *collections: dict[str, set[tuple[int, int]]],
) -> dict[str, set[tuple[int, int]]]:
    merged: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for collection in collections:
        for wav_sha256, windows in collection.items():
            merged[wav_sha256].update(windows)
    return dict(merged)


def historical_development_contract(
    experiment_dir: Path,
    compact_profiles: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, set[tuple[int, int]]], dict[str, set[tuple[int, int]]]]:
    manifest_path = experiment_dir / "data" / "manifests" / "mixed_dev_pool.json"
    run_contract_path = experiment_dir / "results" / "phase3" / "dev_run_contract_v2.json"
    preflight_path = experiment_dir / "results" / "phase3" / "preflight_v2.json"
    b0_path = experiment_dir / "results" / "phase3" / "dev_evidence" / "b0_vad_only.json"
    pinned = {
        manifest_path: HISTORICAL_MANIFEST_BYTE_SHA256,
        run_contract_path: HISTORICAL_RUN_CONTRACT_BYTE_SHA256,
        preflight_path: HISTORICAL_PREFLIGHT_BYTE_SHA256,
        b0_path: HISTORICAL_B0_BYTE_SHA256,
    }
    for path, expected in pinned.items():
        if sha256_file(path) != expected:
            raise Phase5DesignError(f"historical development input drift: {path.name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_contract = json.loads(run_contract_path.read_text(encoding="utf-8"))
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    b0 = json.loads(b0_path.read_text(encoding="utf-8"))
    if (
        run_contract["manifest_sha256"] != HISTORICAL_MANIFEST_CONTENT_SHA256
        or preflight["manifests"]["mixed_dev_pool"]["manifest_sha256"]
        != HISTORICAL_MANIFEST_CONTENT_SHA256
    ):
        raise Phase5DesignError("historical manifest content identity drift")
    manifest_cases = {str(row["case_id"]): row for row in manifest["cases"]}
    b0_cases = {str(row["case_id"]): row for row in b0["cases"]}
    if len(manifest_cases) != 204 or set(manifest_cases) != set(b0_cases):
        raise Phase5DesignError("historical 204-case population drift")
    case_rows: list[dict[str, Any]] = []
    required: dict[str, set[tuple[int, int]]] = defaultdict(set)
    adjacent_grid = sorted(
        {
            (int(row["window_samples"]), int(row["step_samples"]))
            for row in compact_profiles
            if row["profile_class"] == "adjacent"
        }
    )
    anchor_grid = sorted(
        {
            (int(row["window_samples"]), int(row["step_samples"]))
            for row in compact_profiles
            if row["scored_state_mode"] == "source_prefix"
        }
    )
    for case_id in sorted(manifest_cases):
        case = manifest_cases[case_id]
        baseline = b0_cases[case_id]
        duration = int(case["duration_samples"])
        if int(baseline["length_samples"]) != duration:
            raise Phase5DesignError("historical B0 duration drift")
        wav_sha256 = str(case["wav_sha256"])
        for window, step in adjacent_grid:
            for boundary in range(ceil_grid(window, step), duration - window + 1, step):
                required[wav_sha256].add((boundary - window, boundary))
                required[wav_sha256].add((boundary, boundary + window))
        for window, step in anchor_grid:
            for end in range(ceil_grid(window, step), duration + 1, step):
                required[wav_sha256].add((end - window, end))
        b0_actions = [
            {
                "boundary_source_sample": int(row["boundary_source_sample"]),
                "observed_source_sample_at_emit": int(row["observed_source_sample_at_emit"]),
                "audio_epoch": int(row["audio_epoch"]),
                "prev_speech_end_sample": int(row["debug"]["prev_speech_end_sample"]),
                "prev_end_reason": str(row["debug"]["prev_end_reason"]),
            }
            for row in baseline["vad_boundaries"]
        ]
        case_rows.append(
            {
                "case_id": case_id,
                "audio_epoch": int(baseline["audio_epoch"]),
                "wav_sha256": wav_sha256,
                "duration_samples": duration,
                "kind": str(case["kind"]),
                "regions_sha256": hashlib.sha256(
                    canonical_json(case["regions"]).encode("utf-8")
                ).hexdigest(),
                "b0_actions_sha256": hashlib.sha256(
                    canonical_json(b0_actions).encode("utf-8")
                ).hexdigest(),
            }
        )
    cache_root = (
        Path(tempfile.gettempdir()) / "opencode" / "stb_phase3_v2" / "cache" / "eres_embedding_v2"
    )
    cache_receipts: list[dict[str, Any]] = []
    cache_case_mapping: list[dict[str, Any]] = []
    windows_by_checkpoint: dict[str, dict[str, set[tuple[int, int]]]] = {}
    cached_case_ids_by_checkpoint: dict[str, set[str]] = {}
    for checkpoint in ("E-standard",):
        directory = cache_root / checkpoint / HISTORICAL_MANIFEST_CONTENT_SHA256[:16]
        metadata_by_case: dict[str, tuple[Path, dict[str, Any]]] = {}
        for path in sorted(directory.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            case_id = str(payload["case_id"])
            if case_id in metadata_by_case:
                raise Phase5DesignError("duplicate historical cache receipt")
            metadata_by_case[case_id] = (path, payload)
        if not set(metadata_by_case).issubset(manifest_cases):
            raise Phase5DesignError(f"historical cache has an unknown case: {checkpoint}")
        checkpoint_windows: dict[str, set[tuple[int, int]]] = defaultdict(set)
        for case_id in sorted(manifest_cases):
            if case_id not in metadata_by_case:
                cache_case_mapping.append(
                    {
                        "checkpoint": checkpoint,
                        "case_id": case_id,
                        "status": "missing_requires_phase5_inference",
                        "metadata_relative_path": None,
                        "npz_relative_path": None,
                    }
                )
                continue
            metadata_path, metadata = metadata_by_case[case_id]
            case = manifest_cases[case_id]
            npz_path = metadata_path.with_suffix(".npz")
            if (
                metadata.get("checkpoint") != checkpoint
                or metadata.get("manifest_sha256") != HISTORICAL_MANIFEST_CONTENT_SHA256
                or metadata.get("wav_sha256") != case["wav_sha256"]
                or not npz_path.is_file()
                or sha256_file(npz_path) != metadata.get("npz_sha256")
            ):
                raise Phase5DesignError("historical cache receipt mismatch")
            windows = {
                tuple(int(value) for value in key.split("-", maxsplit=1))
                for key in metadata["windows"]
            }
            checkpoint_windows[str(case["wav_sha256"])].update(windows)
            cache_receipts.append(
                {
                    "checkpoint": checkpoint,
                    "case_id": case_id,
                    "metadata_relative_path": metadata_path.relative_to(cache_root).as_posix(),
                    "metadata_byte_sha256": sha256_file(metadata_path),
                    "metadata_byte_count": metadata_path.stat().st_size,
                    "npz_relative_path": npz_path.relative_to(cache_root).as_posix(),
                    "npz_byte_sha256": sha256_file(npz_path),
                    "npz_byte_count": npz_path.stat().st_size,
                    "window_count": len(windows),
                    "windows_sha256": rows_sha256(
                        [{"start": start, "end": end} for start, end in sorted(windows)]
                    ),
                }
            )
            cache_case_mapping.append(
                {
                    "checkpoint": checkpoint,
                    "case_id": case_id,
                    "status": "verified_reusable_receipt",
                    "metadata_relative_path": metadata_path.relative_to(cache_root).as_posix(),
                    "npz_relative_path": npz_path.relative_to(cache_root).as_posix(),
                }
            )
        windows_by_checkpoint[checkpoint] = dict(checkpoint_windows)
        cached_case_ids_by_checkpoint[checkpoint] = set(metadata_by_case)
    historical_probe_steps = 0
    historical_emittable_positions = 0
    maximum_historical_trace: dict[str, Any] | None = None
    maximum_historical_emittable_positions = 0
    for profile in compact_profiles:
        window = int(profile["window_samples"])
        step = int(profile["step_samples"])
        for case_id in sorted(manifest_cases):
            duration = int(manifest_cases[case_id]["duration_samples"])
            first = ceil_grid(window, step)
            last = duration - window if profile["profile_class"] == "adjacent" else duration
            count = inclusive_grid_count(first, last, step)
            historical_probe_steps += count
            emittable = count if profile["profile_class"] == "adjacent" else max(0, count - 1)
            historical_emittable_positions += emittable
            maximum_historical_emittable_positions = max(
                maximum_historical_emittable_positions, emittable
            )
            candidate = {
                "proposal_profile_id": str(profile["proposal_profile_id"]),
                "case_id": case_id,
                "probe_step_count": count,
            }
            if maximum_historical_trace is None or count > int(
                maximum_historical_trace["probe_step_count"]
            ):
                maximum_historical_trace = candidate
    if maximum_historical_trace is None:
        raise Phase5DesignError("historical proposal workload is empty")
    historical_profile_case_count = len(compact_profiles) * len(case_rows)
    cluster_grid_count = (
        len(CLUSTER_DEBOUNCE_MS)
        * len(CLUSTER_RADIUS_MS)
        * len(REFRACTORY_MS)
        * len(REPRESENTATIVES)
    )
    full_grid_count = cluster_grid_count * len(VAD_RADIUS_MS) * len(SILENCE_ASSOCIATION)
    logical_slots_per_profile = full_grid_count * (len(LADDER_STAGES) + len(CONTROL_KINDS))
    physical_nodes_per_profile = 193 + full_grid_count * len(CONTROL_KINDS)
    historical_neural_system_count = len(compact_profiles) * logical_slots_per_profile
    historical_neural_case_identity_count = (
        historical_profile_case_count * logical_slots_per_profile
    )
    historical_baseline_system_count = 2
    historical_baseline_case_identity_count = historical_baseline_system_count * len(case_rows)
    compact_profile_ids = [str(row["proposal_profile_id"]) for row in compact_profiles]
    return (
        {
            "manifest_path": "data/manifests/mixed_dev_pool.json",
            "manifest_byte_sha256": HISTORICAL_MANIFEST_BYTE_SHA256,
            "manifest_content_sha256": HISTORICAL_MANIFEST_CONTENT_SHA256,
            "dev_run_contract_byte_sha256": HISTORICAL_RUN_CONTRACT_BYTE_SHA256,
            "dev_run_contract_content_sha256": str(run_contract["contract_sha256"]),
            "preflight_byte_sha256": HISTORICAL_PREFLIGHT_BYTE_SHA256,
            "b0_evidence_byte_sha256": HISTORICAL_B0_BYTE_SHA256,
            "case_count": len(case_rows),
            "case_coordinate_system": "case-local 16 kHz half-open samples with accepted audio_epoch; no current-pool episode ID is reused",
            "b0_projection_mapping": "case_id becomes source_session_id; vad_boundaries are ordered and receive action_id historical-b0:<case_id>:<index>; debug.prev_speech_end_sample plus debug.prev_end_reason=silence is the accepted causal silence projection; boundary and observed frontiers remain unchanged",
            "case_rows": case_rows,
            "case_rows_sha256": rows_sha256(case_rows),
            "cache_root": str(cache_root),
            "cache_receipt_count": len(cache_receipts),
            "cache_receipt_rows": cache_receipts,
            "cache_receipt_rows_sha256": rows_sha256(cache_receipts),
            "cache_case_mapping_count": len(cache_case_mapping),
            "cache_case_mapping_rows": cache_case_mapping,
            "cache_case_mapping_rows_sha256": rows_sha256(cache_case_mapping),
            "cached_case_count_per_checkpoint": len(cached_case_ids_by_checkpoint["E-standard"]),
            "missing_case_count_per_checkpoint": len(case_rows)
            - len(cached_case_ids_by_checkpoint["E-standard"]),
            "cache_byte_count": sum(
                int(row["metadata_byte_count"]) + int(row["npz_byte_count"])
                for row in cache_receipts
            ),
            "compact_profile_ids": compact_profile_ids,
            "compact_profile_ids_sha256": rows_sha256(
                [{"proposal_profile_id": profile_id} for profile_id in compact_profile_ids]
            ),
            "legacy_profile_count_preserved_not_replayed": 936,
            "legacy_profile_replay_count": 0,
            "proposal_profile_case_count": historical_profile_case_count,
            "proposal_probe_step_count": historical_probe_steps,
            "logical_emittable_position_count": historical_emittable_positions,
            "maximum_proposal_trace": maximum_historical_trace,
            "maximum_emittable_proposal_position_count": maximum_historical_emittable_positions,
            "cluster_execution_count": historical_profile_case_count * cluster_grid_count,
            "fusion_execution_count": historical_profile_case_count * full_grid_count,
            "frequency_control_execution_count": historical_profile_case_count
            * full_grid_count
            * len(CONTROL_KINDS),
            "logical_policy_system_count": historical_neural_system_count,
            "logical_policy_case_identity_count": historical_neural_case_identity_count,
            "baseline_system_count": historical_baseline_system_count,
            "baseline_case_identity_count": historical_baseline_case_identity_count,
            "logical_system_count_including_baselines": historical_neural_system_count
            + historical_baseline_system_count,
            "logical_case_identity_count_including_baselines": historical_neural_case_identity_count
            + historical_baseline_case_identity_count,
            "b0_b1_equivalence_case_count": len(case_rows),
            "physical_policy_case_node_count": historical_profile_case_count
            * physical_nodes_per_profile,
            "raw_cache_rule": "validate pinned E-standard Phase 3 dev receipts; import exact matching W8000 windows only; infer missing compact corrected full-case windows into the distinct Phase 5 cache",
            "heldout_artifacts_accessed": False,
        },
        dict(required),
        windows_by_checkpoint["E-standard"],
    )


def cache_presence(cache_inventory: dict[str, Any]) -> dict[str, Any]:
    files: list[Path] = []
    source_receipts = 0
    for checkpoint in ("E-standard",):
        for source in cache_inventory["eres"][checkpoint]["sources"]:
            source_receipts += 1
            files.append(Path(str(source["metadata_path"])))
            files.extend(Path(str(path)) for path in source["paths"])
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise Phase5DesignError(f"Phase 4 cache file unavailable: {missing[0]}")
    return {
        "cache_root": str(cache_inventory["cache_root"]),
        "source_receipt_count": source_receipts,
        "checkpoints": ["E-standard"],
        "file_count": len(files),
        "all_files_present": True,
        "actual_bytes": sum(path.stat().st_size for path in files),
        "execution_rule": "validate every E-standard Phase 4 source receipt before row import; infer only exact missing W8000 windows into a distinct Phase 5 cache contract; W24 receipts remain historical and are not imported",
    }


def policy_space(profile_count: int, pool_counts: dict[str, int]) -> dict[str, Any]:
    cluster_without_refractory = (
        len(CLUSTER_DEBOUNCE_MS) * len(CLUSTER_RADIUS_MS) * len(REPRESENTATIVES)
    )
    cluster_with_refractory = cluster_without_refractory * len(REFRACTORY_MS)
    vad_grid = len(VAD_RADIUS_MS) * len(SILENCE_ASSOCIATION)
    full_grid = cluster_with_refractory * vad_grid
    physical_ladder_nodes_per_profile = (
        1 + cluster_without_refractory + cluster_with_refractory + full_grid
    )
    logical_ladder_slots_per_profile = full_grid * len(LADDER_STAGES)
    controls_per_profile = full_grid * len(CONTROL_KINDS)
    physical_system_count = 3 + profile_count * (
        physical_ladder_nodes_per_profile + controls_per_profile
    )
    logical_system_count = 3 + profile_count * (
        logical_ladder_slots_per_profile + controls_per_profile
    )
    episode_rows = {pool: count * logical_system_count for pool, count in pool_counts.items()}
    proposal_episode_batches = profile_count * sum(pool_counts.values())
    cluster_executions = proposal_episode_batches * cluster_with_refractory
    fusion_executions = proposal_episode_batches * full_grid
    alias_system_count = profile_count * (
        logical_ladder_slots_per_profile - physical_ladder_nodes_per_profile
    )
    episode_count = sum(pool_counts.values())
    return {
        "cluster_grid": {
            "cluster_debounce_ms": list(CLUSTER_DEBOUNCE_MS),
            "cluster_boundary_radius_ms": list(CLUSTER_RADIUS_MS),
            "refractory_ms": list(REFRACTORY_MS),
            "representative": list(REPRESENTATIVES),
            "unique_cluster_without_refractory_count": cluster_without_refractory,
            "unique_cluster_with_refractory_count": cluster_with_refractory,
        },
        "vad_grid": {
            "detector_vad_radius_ms": list(VAD_RADIUS_MS),
            "same_silence_interval_association": list(SILENCE_ASSOCIATION),
            "count": vad_grid,
        },
        "full_fusion_grid_count_per_proposal_profile": full_grid,
        "same_proposal_ladder": {
            "stages": list(LADDER_STAGES),
            "paired_chain_count_per_proposal_profile": full_grid,
            "logical_stage_slot_count_per_proposal_profile": logical_ladder_slots_per_profile,
            "physical_execution_node_count_per_proposal_profile": physical_ladder_nodes_per_profile,
            "alias_edge_count_per_proposal_profile": (
                logical_ladder_slots_per_profile - physical_ladder_nodes_per_profile
            ),
            "inactive_later_parameters_are_execution_aliases": True,
            "eres_full_hard_soft_equals_vad_association": True,
            "cluster_r0_stage_aliases_full_cluster_r0_nodes": True,
        },
        "frequency_controls": {
            "kinds": list(CONTROL_KINDS),
            "applied_to": "every full_hard_soft_fusion policy",
            "count_per_proposal_profile": controls_per_profile,
            "same_episode_hard_action_count_required": True,
            "generation_inputs": "causally observed VAD-active regions, causal audio energy, tested action availability, deterministic seed; no GT labels",
        },
        "baseline_systems": ["B0", "B1", "no_neural_policy_control"],
        "proposal_profile_count": profile_count,
        "full_fusion_system_count": profile_count * full_grid,
        "control_system_count": profile_count * controls_per_profile,
        "physical_neural_ladder_node_count": profile_count * physical_ladder_nodes_per_profile,
        "logical_neural_ladder_system_count": profile_count * logical_ladder_slots_per_profile,
        "physical_execution_system_count": physical_system_count,
        "logical_system_count": logical_system_count,
        "logical_ladder_alias_edge_count": alias_system_count,
        "logical_episode_identities_by_pool": episode_rows,
        "logical_episode_identity_count": sum(episode_rows.values()),
        "content_addressed_execution_dag": {
            "proposal_profile_episode_batch_count": proposal_episode_batches,
            "cluster_execution_count": cluster_executions,
            "fusion_execution_count": fusion_executions,
            "policy_node_count": cluster_executions + fusion_executions,
            "maximum_frequency_control_episode_count": fusion_executions * len(CONTROL_KINDS),
            "physical_execution_episode_node_count": physical_system_count * episode_count,
            "logical_ladder_alias_episode_edge_count": alias_system_count * episode_count,
            "logical_episode_slots_expand_deterministically_from_alias_edges": True,
        },
        "logical_identity_contract": logical_identity_contract(),
    }


def logical_identity_contract() -> dict[str, Any]:
    chain_fields = [
        "cluster_debounce_ms",
        "cluster_boundary_radius_ms",
        "refractory_ms",
        "representative",
        "detector_vad_radius_ms",
        "same_silence_interval_association",
    ]
    return {
        "hash_function": "sha256",
        "canonical_json": "UTF-8 RFC8259 object, ensure_ascii=false, sorted keys, separators comma/colon",
        "logical_system_id": "system:" + "sha256(canonical_json(logical_system_key))",
        "physical_node_id": "node:" + "sha256(canonical_json(physical_node_key))",
        "baseline_key_fields": ["kind=baseline", "baseline_id"],
        "ladder_logical_key_fields": [
            "kind=ladder",
            "proposal_profile_id",
            *chain_fields,
            "stage",
        ],
        "control_logical_key_fields": [
            "kind=frequency_control",
            "proposal_profile_id",
            *chain_fields,
            "control_kind",
        ],
        "physical_node_dependency_fields": {
            "naive_proposal_as_cut": ["proposal_profile_id"],
            "clustering_only": [
                "proposal_profile_id",
                "cluster_debounce_ms",
                "cluster_boundary_radius_ms",
                "representative",
            ],
            "clustering_plus_refractory": [
                "proposal_profile_id",
                "cluster_debounce_ms",
                "cluster_boundary_radius_ms",
                "refractory_ms",
                "representative",
            ],
            "plus_vad_association": ["proposal_profile_id", *chain_fields],
            "full_hard_soft_fusion": ["proposal_profile_id", *chain_fields],
            "frequency_control": [
                "proposal_profile_id",
                *chain_fields,
                "control_kind",
            ],
        },
        "full_hard_soft_alias": "for ERes, full_hard_soft_fusion maps to the same physical fusion node as plus_vad_association while retaining a distinct logical system ID",
        "episode_expansion_record_field_order": [
            "pool",
            "logical_system_id",
            "episode_id",
            "physical_node_id",
            "proposal_trace_sha256",
            "action_trace_sha256",
            "score_trace_sha256",
        ],
        "episode_expansion_sort_order": [
            "pool_order=diagnostic_dev,frontier_dev,natural_exposure_validation",
            "logical_system_id",
            "episode_id",
        ],
        "rolling_digest_frame": "uint64 big-endian byte length followed by UTF-8 canonical_json(record), with no delimiter",
        "shard_receipt_fields": [
            "first_expansion_key",
            "last_expansion_key",
            "row_count",
            "rolling_content_sha256",
            "compressed_byte_sha256",
            "compressed_byte_count",
        ],
    }


def policy_benchmark(result_dir: Path) -> dict[str, Any]:
    path = result_dir / POLICY_BENCHMARK_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    verify_self_hash(payload)
    if payload.get("authority_sha256") != AUTHORITY_SHA256:
        raise Phase5DesignError("policy benchmark authority drift")
    if payload.get("schema_version") != "turn_episode_phase5_policy_benchmark.v7":
        raise Phase5DesignError("policy benchmark schema drift")
    if payload.get("selected_policy_workers") != 8:
        raise Phase5DesignError("policy benchmark worker count drift")
    sentinel = payload["sentinel_contract"]
    if sentinel.get("cluster_grid_count") != 36:
        raise Phase5DesignError("policy benchmark cluster grid drift")
    if sentinel.get("vad_grid_count") != 4:
        raise Phase5DesignError("policy benchmark VAD grid drift")
    if (
        sentinel.get("current_maximum_emittable_position_count") != 301
        or sentinel.get("historical_maximum_emittable_position_count") != 11392
        or sentinel.get("maximum_source_prefix_probe_step_count") != 32696
        or sentinel.get("total_logical_emittable_position_count") != 259876
        or sentinel.get("independent_audit_sample_size") != INDEPENDENT_AUDIT_SAMPLE_SIZE
        or sentinel.get("historical_word_annotation_file_sha256") != HISTORICAL_WORD_FILE_SHA256
    ):
        raise Phase5DesignError("policy benchmark trace envelope drift")
    if any(
        row.get("status") != "complete" or row.get("placed_action_count") != 11392
        for row in payload["frequency_controls"]["worst_shape_rows"]
    ):
        raise Phase5DesignError("policy benchmark control envelope failed")
    scoring = payload.get("scoring", {})
    exact_shapes = scoring.get("historical_exact_shapes", [])
    exact_by_case = {row.get("case_id"): row for row in exact_shapes}
    envelope = scoring.get("joint_forecast_envelope", {})
    expected_word_shapes = {
        "ami_ES2003a": (11392, 2038, 2387),
        "ami_IS1008a": (9433, 2504, 2890),
    }
    exact_word_shapes_valid = set(exact_by_case) == set(expected_word_shapes) and all(
        row.get("action_count") == expected[0]
        and row.get("word_interval_count") == expected[1]
        and row.get("raw_word_record_count") == expected[2]
        and row.get("word_timing_observable") is True
        and {
            receipt.get("filename"): receipt.get("byte_sha256")
            for receipt in row.get("word_annotation_files", [])
        }
        == HISTORICAL_WORD_FILE_SHA256[case_id]
        and bool(row.get("word_record_sha256"))
        for case_id, expected in expected_word_shapes.items()
        for row in (exact_by_case[case_id],)
    )
    parallel_scoring = scoring.get("parallel", {})
    verifier = payload.get("independent_verifier_recompute", {})
    verifier_scoring_shape = verifier.get("worst_shapes", {}).get("scoring_joint_envelope", {})
    if (
        scoring.get("source_manifest_byte_sha256") != HISTORICAL_MANIFEST_BYTE_SHA256
        or [(row.get("case_id"), row.get("action_count")) for row in exact_shapes]
        != [("ami_ES2003a", 11392), ("ami_IS1008a", 9433)]
        or not exact_word_shapes_valid
        or envelope.get("case_id") != "ami_IS1008a"
        or envelope.get("action_count") != 11392
        or envelope.get("reference_count") != 132
        or envelope.get("region_count") != 338
        or envelope.get("singleton_interval_count") != 169
        or envelope.get("overlap_interval_count") != 64
        or envelope.get("word_interval_count") != 2504
        or envelope.get("raw_word_record_count") != 2890
        or envelope.get("word_timing_observable") is not True
        or parallel_scoring.get("word_interval_count") != 2504
        or parallel_scoring.get("word_timing_observable") is not True
        or verifier_scoring_shape.get("word_intervals") != 2504
        or verifier_scoring_shape.get("word_timing_observable") is not True
        or verifier.get("serial", {}).get("scoring_word_interval_count") != 2504
        or verifier.get("serial", {}).get("scoring_word_timing_observable") is not True
        or verifier.get("parallel", {}).get("scoring_word_interval_count") != 2504
        or verifier.get("parallel", {}).get("scoring_word_timing_observable") is not True
        or float(parallel_scoring.get("conservative_actions_per_second", 0.0)) <= 0.0
    ):
        raise Phase5DesignError("policy benchmark scoring envelope drift")
    if len(verifier.get("algorithm", [])) != 4 or not verifier.get("parallel", {}).get(
        "trace_sha256s"
    ):
        raise Phase5DesignError("independent verifier benchmark incomplete")
    identity = payload.get("logical_identity_digest", {})
    if (
        int(identity.get("total_logical_identity_rows", 0)) != 4988898
        or float(identity.get("parallel", {}).get("conservative_rows_per_second", 0.0)) <= 0.0
    ):
        raise Phase5DesignError("logical identity digest benchmark incomplete")
    live_policy = sha256_file(Path(__file__).with_name("phase5_policy.py"))
    if payload["generated_from"].get("phase5_policy.py") != live_policy:
        raise Phase5DesignError("policy benchmark code drift")
    live_benchmark = sha256_file(Path(__file__).with_name("phase5_policy_benchmark.py"))
    if payload["generated_from"].get("phase5_policy_benchmark.py") != live_benchmark:
        raise Phase5DesignError("policy benchmark harness drift")
    for name in (
        "build_episodes.py",
        "pcm_oracle.py",
        "phase5_inputs.py",
        "phase5_proposals.py",
        "phase5_scoring.py",
        "scoring.py",
    ):
        if payload["generated_from"].get(name) != sha256_file(Path(__file__).with_name(name)):
            raise Phase5DesignError(f"policy benchmark code drift: {name}")
    return payload


def storage_benchmark(result_dir: Path) -> dict[str, Any]:
    path = result_dir / STORAGE_BENCHMARK_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    verify_self_hash(payload)
    if payload.get("authority_sha256") != AUTHORITY_SHA256:
        raise Phase5DesignError("storage benchmark authority drift")
    if payload.get("schema_version") != "turn_episode_phase5_storage_benchmark.v5":
        raise Phase5DesignError("storage benchmark schema drift")
    if not payload.get("within_result_ceiling"):
        raise Phase5DesignError("storage benchmark exceeds result ceiling")
    expected_rows = {
        "physical_proposal_execution_receipt": 3824,
        "logical_proposal_route_index": 4328,
        "physical_system_definition": 2503,
        "logical_system_definition": 4611,
        "logical_alias_edge": 2108,
        "current_system_block_aggregate": 4611,
        "historical_corrected_system_aggregate": 4610,
        "deterministic_failure_example": 420,
        "independent_audit_unit": INDEPENDENT_AUDIT_SAMPLE_SIZE,
    }
    actual_rows = {
        name: int(row["expected_row_count"]) for name, row in payload["representation"].items()
    }
    if actual_rows != expected_rows or payload.get("untyped_fixed_reserve_bytes") != 0:
        raise Phase5DesignError("storage benchmark schema/count drift")
    shared = payload.get("shared_schema", {})
    if shared.get("pool_order") != [
        "diagnostic_dev",
        "frontier_dev",
        "natural_exposure_validation",
    ] or shared.get("pool_block_counts") != {
        "diagnostic_dev": 13,
        "frontier_dev": 10,
        "natural_exposure_validation": 20,
    }:
        raise Phase5DesignError("storage pool dimension drift")
    if shared.get("independent_audit_sample_size") != INDEPENDENT_AUDIT_SAMPLE_SIZE:
        raise Phase5DesignError("storage audit sample drift")
    historical_baselines = shared.get("historical_baseline_contract", {})
    if (
        historical_baselines.get("baseline_system_ids") != ["B0", "B1"]
        or historical_baselines.get("baseline_system_count") != 2
        or historical_baselines.get("case_count_per_baseline") != 204
        or historical_baselines.get("baseline_case_identity_count") != 408
        or historical_baselines.get("neural_system_count") != 4608
        or historical_baselines.get("neural_case_identity_count") != 940032
        or historical_baselines.get("total_system_count") != 4610
        or historical_baselines.get("total_case_identity_count") != 940440
        or historical_baselines.get("b0_b1_equivalence_receipt_required") is not True
    ):
        raise Phase5DesignError("historical baseline storage contract drift")
    reconstruction = shared.get("reconstruction_authority", {})
    if not all(
        reconstruction.get(field)
        for field in (
            "persisted_evidence",
            "accepted_inputs",
            "proposal_and_progress_rule",
            "derived_trace_rule",
            "acceptance_rule",
        )
    ):
        raise Phase5DesignError("storage reconstruction authority incomplete")
    if not all(
        row["serialization_benchmark"].get("verification_passed")
        for row in payload["representation"].values()
    ):
        raise Phase5DesignError("storage serialization verification failed")
    live = sha256_file(Path(__file__).with_name("phase5_storage_benchmark.py"))
    if payload["generated_from"].get("phase5_storage_benchmark.py") != live:
        raise Phase5DesignError("storage benchmark harness drift")
    return payload


def historical_word_timing_expectation(
    row: dict[str, Any], manifest_byte_sha256: str
) -> dict[str, Any]:
    return {
        "unit_id": str(row["case_id"]),
        "annotation_source_identity": {
            "manifest_byte_sha256": manifest_byte_sha256,
            "word_annotation_files": row["word_annotation_files"],
        },
        "word_record_sha256": str(row["word_record_sha256"]),
        "raw_word_record_count": int(row["raw_word_record_count"]),
        "word_interval_count": int(row["word_interval_count"]),
        "word_timing_observable": bool(row["word_timing_observable"]),
        "lexical_scoring_disposition": "scored_with_trusted_word_timing",
    }


def validate_interstage_word_timing_receipts(
    gate: dict[str, Any], receipt: dict[str, Any]
) -> dict[str, Any]:
    contract = gate.get("word_timing_receipt_contract", {})
    fields = list(contract.get("field_order", []))
    if fields != list(WORD_TIMING_RECEIPT_FIELDS):
        raise Phase5DesignError("interstage word receipt schema drift")
    expected = receipt.get("expected_word_timing_receipts")
    observed = receipt.get("observed_word_timing_receipts")
    if not isinstance(expected, list) or not isinstance(observed, list):
        raise Phase5DesignError("interstage word receipt missing")
    if any(set(row) != set(fields) for row in expected + observed):
        raise Phase5DesignError("interstage word receipt field drift")
    expected_ids = [str(row["unit_id"]) for row in expected]
    observed_ids = [str(row["unit_id"]) for row in observed]
    if expected_ids != sorted(set(expected_ids)) or observed_ids != expected_ids:
        raise Phase5DesignError("interstage word receipt unit drift")
    if canonical_json(expected) != canonical_json(observed):
        raise Phase5DesignError("interstage word receipt exact-value drift")
    expected_by_id = {str(row["unit_id"]): row for row in expected}
    for sentinel in contract.get("historical_sentinels", []):
        if expected_by_id.get(str(sentinel["unit_id"])) != sentinel:
            raise Phase5DesignError("interstage historical word sentinel drift")
    for row in observed:
        observable = bool(row["word_timing_observable"])
        disposition = str(row["lexical_scoring_disposition"])
        if observable and disposition != "scored_with_trusted_word_timing":
            raise Phase5DesignError("observable word timing disposition drift")
        if not observable and disposition != "unscored_missing_word_timing":
            raise Phase5DesignError("missing word timing was treated as lexical absence")
        if int(row["word_interval_count"]) < 0 or int(row["raw_word_record_count"]) < 0:
            raise Phase5DesignError("negative word timing cardinality")
        if not row["annotation_source_identity"]:
            raise Phase5DesignError("word timing source identity missing")
        if observable and not row["word_record_sha256"]:
            raise Phase5DesignError("word timing record hash missing for observable timing")
    return {
        "stage_b_allowed": True,
        "word_timing_receipt_count": len(observed),
        "word_timing_receipts_sha256": hashlib.sha256(
            canonical_json(observed).encode("utf-8")
        ).hexdigest(),
    }


def runtime_forecast(
    window_info: dict[str, Any],
    policy_info: dict[str, Any],
    benchmark: dict[str, Any],
    storage: dict[str, Any],
    current_workload: dict[str, Any],
    historical_workload: dict[str, Any],
) -> dict[str, Any]:
    benchmark_rates = {"E-standard": 84.38591067602052}
    missing_windows = int(window_info["new_inference_window_count"])
    inference_seconds = missing_windows / benchmark_rates["E-standard"]
    current_policy_batches = int(
        policy_info["content_addressed_execution_dag"]["proposal_profile_episode_batch_count"]
    )
    historical_policy_batches = int(historical_workload["proposal_profile_case_count"])
    policy_batches = current_policy_batches + historical_policy_batches
    policy_batches_per_second_floor = float(benchmark["conservative_parallel_batches_per_second"])
    policy_batch_overhead_seconds = policy_batches / policy_batches_per_second_floor
    logical_emittable_positions = int(
        current_workload["total_logical_emittable_position_count"]
    ) + int(historical_workload["logical_emittable_position_count"])
    policy_positions_per_second_floor = float(
        benchmark["historical_worst_policy_grid_parallel"][
            "conservative_proposal_positions_per_second"
        ]
    )
    policy_position_seconds = logical_emittable_positions / policy_positions_per_second_floor
    policy_seconds = policy_batch_overhead_seconds + policy_position_seconds
    import_rows_per_second_floor = 250.0
    import_seconds = (
        int(window_info["reusable_checkpoint_window_jobs"]) / import_rows_per_second_floor
    )
    proposal_probe_steps = int(current_workload["total_physical_probe_step_count"]) + int(
        historical_workload["proposal_probe_step_count"]
    )
    proposal_probe_steps_per_second_floor = float(
        benchmark["source_prefix_state"]["conservative_probe_steps_per_second_floor"]
    )
    proposal_state_seconds = proposal_probe_steps / proposal_probe_steps_per_second_floor
    maximum_control_action_placements = logical_emittable_positions * 144 * len(CONTROL_KINDS)
    control_placements_per_second_floor = float(
        benchmark["frequency_controls"]["parallel"]["conservative_action_placements_per_second"]
    )
    controls_seconds = maximum_control_action_placements / control_placements_per_second_floor
    maximum_proposal_derived_score_actions = logical_emittable_positions * 625
    scoring_envelope = benchmark["scoring"]["joint_forecast_envelope"]
    scoring_actions_per_second_floor = float(
        benchmark["scoring"]["parallel"]["conservative_actions_per_second"]
    )
    scoring_seconds = maximum_proposal_derived_score_actions / scoring_actions_per_second_floor
    serialization_seconds = sum(
        float(row["projected_plain_bytes"])
        / float(row["serialization_benchmark"]["plain_bytes_per_second"])
        for row in storage["representation"].values()
    )
    storage_verification_seconds = sum(
        float(row["expected_row_count"])
        / float(row["sample_row_count"])
        * float(row["serialization_benchmark"]["decompress_and_hash_verify_seconds"])
        for row in storage["representation"].values()
    )
    logical_identity_count = int(policy_info["logical_episode_identity_count"]) + int(
        historical_workload["logical_case_identity_count_including_baselines"]
    )
    logical_identity_rows_per_second_floor = float(
        benchmark["logical_identity_digest"]["parallel"]["conservative_rows_per_second"]
    )
    logical_identity_digest_seconds = (
        logical_identity_count / logical_identity_rows_per_second_floor
    )
    verifier_recompute = benchmark["independent_verifier_recompute"]
    audit_workers = int(verifier_recompute["parallel"]["workers"])
    if audit_workers != int(benchmark["selected_policy_workers"]):
        raise Phase5DesignError("verifier worker contract drift")
    audit_parallel_batches = math.ceil(INDEPENDENT_AUDIT_SAMPLE_SIZE / audit_workers)
    sampled_trace_audit_seconds = audit_parallel_batches * float(
        verifier_recompute["parallel"]["wall_seconds"]
    )
    verification_seconds = (
        import_seconds
        + logical_identity_digest_seconds
        + storage_verification_seconds
        + sampled_trace_audit_seconds
    )
    stage_a_seconds = inference_seconds + import_seconds + proposal_state_seconds
    total = (
        stage_a_seconds
        + policy_seconds
        + controls_seconds
        + scoring_seconds
        + logical_identity_digest_seconds
        + serialization_seconds
        + verification_seconds
    )
    execution_ceiling_hours = EXECUTION_PLANNING_CEILING_HOURS
    historical_word_sentinels = sorted(
        (
            historical_word_timing_expectation(
                row, benchmark["scoring"]["source_manifest_byte_sha256"]
            )
            for row in benchmark["scoring"]["historical_exact_shapes"]
        ),
        key=lambda row: row["unit_id"],
    )
    return {
        "hardware_identity": "same declared Phase 4 CPUExecutionProvider host",
        "new_inference_workers": 10,
        "policy_replay_workers": int(benchmark["selected_policy_workers"]),
        "onnxruntime": "1.28.0",
        "new_inference_seconds": inference_seconds,
        "phase_4_cache_validation_and_import_seconds": import_seconds,
        "proposal_state_probe_step_count": proposal_probe_steps,
        "proposal_state_probe_steps_per_second_floor": proposal_probe_steps_per_second_floor,
        "proposal_state_seconds": proposal_state_seconds,
        "stage_a_inference_cache_and_proposal_seconds": stage_a_seconds,
        "stage_a_inference_cache_and_proposal_hours": stage_a_seconds / 3600.0,
        "policy_replay_benchmark_content_sha256": benchmark["content_sha256"],
        "policy_replay_batches_per_second_floor": policy_batches_per_second_floor,
        "current_policy_replay_batch_count": current_policy_batches,
        "historical_policy_replay_batch_count": historical_policy_batches,
        "policy_replay_batch_count": policy_batches,
        "policy_batch_overhead_seconds": policy_batch_overhead_seconds,
        "logical_emittable_position_hard_upper_count": logical_emittable_positions,
        "policy_positions_per_second_floor": policy_positions_per_second_floor,
        "policy_position_seconds": policy_position_seconds,
        "policy_replay_seconds": policy_seconds,
        "maximum_control_action_placement_count": maximum_control_action_placements,
        "control_action_placements_per_second_floor": control_placements_per_second_floor,
        "controls_seconds_hard_upper": controls_seconds,
        "maximum_proposal_derived_score_action_count": maximum_proposal_derived_score_actions,
        "scoring_joint_forecast_envelope": {
            "shape_id": scoring_envelope["shape_id"],
            "source_manifest_byte_sha256": benchmark["scoring"]["source_manifest_byte_sha256"],
            "action_count": int(scoring_envelope["action_count"]),
            "reference_count": int(scoring_envelope["reference_count"]),
            "region_count": int(scoring_envelope["region_count"]),
            "singleton_interval_count": int(scoring_envelope["singleton_interval_count"]),
            "overlap_interval_count": int(scoring_envelope["overlap_interval_count"]),
            "pause_interval_count": int(scoring_envelope["pause_interval_count"]),
            "unscored_interval_count": int(scoring_envelope["unscored_interval_count"]),
            "word_interval_count": int(scoring_envelope["word_interval_count"]),
            "word_timing_observable": bool(scoring_envelope["word_timing_observable"]),
            "raw_word_record_count": int(scoring_envelope["raw_word_record_count"]),
            "word_annotation_files": scoring_envelope["word_annotation_files"],
            "word_record_sha256": scoring_envelope["word_record_sha256"],
        },
        "scoring_actions_per_second_floor": scoring_actions_per_second_floor,
        "scoring_seconds_hard_upper": scoring_seconds,
        "logical_identity_count": logical_identity_count,
        "logical_identity_rows_per_second_floor": logical_identity_rows_per_second_floor,
        "logical_identity_digest_seconds": logical_identity_digest_seconds,
        "serialization_hash_compression_verification_seconds": serialization_seconds,
        "independent_verifier_benchmark": {
            "algorithm": verifier_recompute["algorithm"],
            "representative_shapes": verifier_recompute["representative_shapes"],
            "worst_shapes": verifier_recompute["worst_shapes"],
            "serial_elapsed_seconds": verifier_recompute["serial"]["elapsed_seconds"],
            "parallel_workers": verifier_recompute["parallel"]["workers"],
            "parallel_wall_seconds": verifier_recompute["parallel"]["wall_seconds"],
            "trace_sha256s": verifier_recompute["parallel"]["trace_sha256s"],
        },
        "independent_audit_contract": {
            "sample_size": INDEPENDENT_AUDIT_SAMPLE_SIZE,
            "selection_seed": "turn-episode-v1-phase5-audit-v1",
            "include_mandatory_sentinels": True,
            "include_deterministic_failure_examples": True,
            "required_strata": [
                "checkpoint",
                "proposal_policy_class",
                "pool",
                "corpus",
                "ladder_stage",
                "fusion_mode",
                "control_kind",
            ],
            "hash_fill_order": "ascending sha256(selection_seed || canonical_unit_id)",
            "recompute_from": "accepted cache/audio/annotation inputs, never trusted derived outputs",
            "mismatch_tolerance": 0,
        },
        "independent_verification_cache_validation_seconds": import_seconds,
        "independent_verification_logical_identity_digest_seconds": logical_identity_digest_seconds,
        "independent_verification_storage_seconds": storage_verification_seconds,
        "independent_verification_audit_parallel_batch_count": audit_parallel_batches,
        "independent_verification_sampled_trace_seconds": sampled_trace_audit_seconds,
        "independent_verification_seconds": verification_seconds,
        "total_forecast_seconds": total,
        "total_forecast_hours": total / 3600.0,
        "execution_ceiling_hours": execution_ceiling_hours,
        "within_execution_ceiling": total <= execution_ceiling_hours * 3600,
        "interstage_exact_cardinality_gate": {
            "timing": "after complete proposal/B0 trace generation and before the compact policy/control/scoring sweep",
            "required_counts": [
                "proposal_count_per_profile_episode_or_historical_case",
                "post_fusion_detector_hard_action_count_per_physical_node",
                "B0_action_count_per_episode_or_case",
                "scorable_reference_count_per_episode_or_historical_case",
                "region_singleton_overlap_pause_unscored_and_word_interval_counts_per_episode_or_historical_case",
                "word_timing_observable_and_annotation_source_identity_per_episode_or_historical_case",
                "joint_scoring_shape_count_by_action_reference_and_timeline_cardinalities",
                "unique_action_trace_count",
                "logical_identity_count_by_current_and_historical_pool",
                "pool_metric_row_count_and_pool_block_metric_row_count",
                "projected_output_bytes_by_typed_representation",
            ],
            "word_timing_receipt_contract": {
                "validator": "phase5_design.validate_interstage_word_timing_receipts",
                "scope": "every current episode and corrected historical case",
                "field_order": list(WORD_TIMING_RECEIPT_FIELDS),
                "expected_value_authority": "recompute from accepted manifests and pinned raw annotation bytes independently of the scoring input; the independent verifier repeats that recomputation",
                "comparison": "exact canonical equality of expected and observed values, including observability, clipped interval count, raw record count, record hash, and annotation source identity",
                "missing_timing_rule": "word_timing_observable=false requires lexical_scoring_disposition=unscored_missing_word_timing and never represents zero lexical harm",
                "historical_sentinels": historical_word_sentinels,
            },
            "formula": "recompute compact execution, exhaustive cache/identity/completeness/aggregate checks, the frozen 2048-unit stratified raw/derived audit, pool-separated typed serialization, decompression/hash verification, and exact word-timing receipts from accepted benchmark floors",
            "stop_condition": "do not start Stage B if any word-timing observability, source identity, record hash, raw/clipped count, or lexical disposition differs from the independently recomputed expected receipt; also stop and report before execution if the exact compact forecast materially exceeds 3 hours, peak RSS exceeds 16 GiB, or typed output exceeds 8 GiB",
            "scientific_aggregate_before_gate": False,
        },
        "peak_rss_limit_bytes": 16 * 1024**3,
        "cache_limit_bytes": 8 * 1024**3,
        "result_limit_bytes": 8 * 1024**3,
        "aggregate_json_limit_bytes": 10 * 1024**2,
        "detail_shard_limit_bytes": 20 * 1024**2,
        "network": "forbidden",
        "credentials": "forbidden",
        "provider_cost_usd": 0,
    }


def historical_input(experiment_dir: Path, result_dir: Path) -> dict[str, Any]:
    rows_path = experiment_dir / "results" / "phase3" / "dev_rows_v2.jsonl"
    ledger_path = result_dir / "reviews" / "historical_artifact_ledger.json"
    if sha256_file(rows_path) != HISTORICAL_ROWS_BYTE_SHA256:
        raise Phase5DesignError("historical development rows drift")
    if sha256_file(ledger_path) != HISTORICAL_LEDGER_BYTE_SHA256:
        raise Phase5DesignError("historical artifact ledger drift")
    count = sum(1 for line in rows_path.read_text(encoding="utf-8").splitlines() if line)
    if count != 1369:
        raise Phase5DesignError("historical row count drift")
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line]
    family_counts = Counter(str(row["family"]) for row in rows)
    if dict(family_counts) != {"eres2netv2": 936, "ls_eend": 432, "b0": 1}:
        raise Phase5DesignError("historical family counts drift")
    return {
        "path": "../phase3/dev_rows_v2.jsonl",
        "byte_sha256": HISTORICAL_ROWS_BYTE_SHA256,
        "row_count": count,
        "family_row_counts": dict(family_counts),
        "role": "historical_validation_corrected_rescore_only",
        "selection_eligible": False,
        "overwrite_forbidden": True,
        "eres_contract": "preserve all 936 legacy ERes rows as immutable historical evidence without replay; corrected historical-case rescoring runs only B0/B1 and the four compact E-standard profiles",
        "legacy_eres_profile_replay_count": 0,
        "compact_eres_profile_rescore_count": 4,
        "ls_contract": "preserve all 432 rows and emit not_replayed_signal_stop; no LS neural policy replay",
        "b0_contract": "recompute through the accepted B0/B1 engine and require exact action identity",
    }


def build_payload(experiment_dir: Path) -> dict[str, Any]:
    result_dir = experiment_dir / "results" / "turn_episode_v1"
    inputs = phase4_inputs(result_dir)
    completion = inputs["phase_4_completion.json"]
    verification = inputs["phase_4_verification.json"]
    disposition = inputs["phase_4_signal_disposition.json"]
    state = inputs["phase_4_state_equivalence.json"]
    if completion["phase_4_bundle_sha256"] != PHASE4_BUNDLE_SHA256:
        raise Phase5DesignError("Phase 4 bundle drift")
    if not verification["passed"] or verification["mismatches"]:
        raise Phase5DesignError("Phase 4 verification is not accepted input")
    profiles = proposal_profiles(experiment_dir, disposition, state)
    historical_development, historical_required, historical_available = (
        historical_development_contract(experiment_dir, profiles)
    )
    episodes, source_summary, source_by_episode = load_populations(experiment_dir, inputs)
    phase4 = load_inputs(experiment_dir)
    inventory = json.loads(
        (result_dir / "phase_4_cache_inventory.json").read_text(encoding="utf-8")
    )
    cases = load_synthetic_cases(experiment_dir / "data" / "manifests")
    coverage = json.loads((result_dir / "coverage_inventory.json").read_text(encoding="utf-8"))
    details = {
        str(row["session_id"]): row
        for row in map(
            json.loads,
            (result_dir / "coverage_inventory_details.jsonl")
            .read_text(encoding="utf-8")
            .splitlines(),
        )
    }
    sources, rebuilt_source_by_episode = _source_maps(
        experiment_dir,
        episodes,
        cases,
        coverage,
        details,
    )
    if source_by_episode != rebuilt_source_by_episode:
        raise Phase5DesignError("source map is not reproducible")
    current_required = required_windows(episodes, source_by_episode, sources, profiles)
    window_info = window_ledger(
        merge_window_sets(current_required, historical_required),
        merge_window_sets(phase4.embedding_windows, historical_available),
    )
    current_proposal_workload = proposal_workload(episodes, source_by_episode, sources, profiles)
    pool_counts = dict(Counter(str(row["pool"]) for row in episodes))
    pool_blocks = pool_block_index(episodes, coverage)
    policy_info = policy_space(len(profiles), pool_counts)
    benchmark = policy_benchmark(result_dir)
    storage = storage_benchmark(result_dir)
    forecast = runtime_forecast(
        window_info,
        policy_info,
        benchmark,
        storage,
        current_proposal_workload,
        historical_development,
    )
    script_path = Path(__file__).resolve()
    return {
        "schema_version": "turn_episode_phase5_design.v7",
        "authority_sha256": AUTHORITY_SHA256,
        "accepted_phase_4_candidate": PHASE4_ACCEPTED_CANDIDATE,
        "accepted_phase_4_bundle_sha256": PHASE4_BUNDLE_SHA256,
        "accepted_phase_4_completion_content_sha256": PHASE4_COMPLETION_CONTENT_SHA256,
        "accepted_phase_4_verification_content_sha256": PHASE4_VERIFICATION_CONTENT_SHA256,
        "input_identities": INPUT_IDENTITIES,
        "population": {
            "pool_counts": pool_counts,
            "episode_count": len(episodes),
            **source_summary,
            "pool_block_count": len(pool_blocks),
            "unique_statistical_block_count": len(
                {row["statistical_block_id"] for row in pool_blocks}
            ),
            "pool_block_index_sha256": rows_sha256(pool_blocks),
            "pool_block_index": pool_blocks,
            "confirmatory_heldout_episode_count": 0,
            "heldout_paths_resolved": False,
        },
        "family_compute_envelopes": {
            "eres2netv2": {
                "phase_4_disposition": "signal_go",
                "selection_rule": "owner-directed compact allowlist: exactly four E-standard profiles derived from the three accepted eligible_go extractor IDs",
                "compact_profile_count": 4,
                "historical_profile_count_preserved_not_replayed": 936,
                "w24_phase4_results_preserved_historical_only": True,
                "w24_phase5_inference_or_replay": False,
                "proposal_profiles": profiles,
            },
            "ls_eend": {
                "phase_4_disposition": "signal_stop",
                "systems": ["B0", "B1", "no_neural_policy_control"],
                "raw_diagnostic_artifact": "phase_4_ls_signal_report.json",
                "new_neural_inference": False,
                "neural_policy_sweep": False,
            },
        },
        "proposal_contract": {
            "compact_profile_ids": list(COMPACT_PROFILE_IDS),
            "compact_signal_extractor_ids": list(COMPACT_SIGNAL_EXTRACTOR_IDS),
            "threshold_rule": "strict change_score > 0.50",
            "adaptive_thresholds": False,
            "new_step_threshold_cross_product": False,
            "proposal_order": [
                "observed_source_sample_at_emit",
                "boundary_source_sample",
                "profile_id",
                "proposal_id",
            ],
            "source_prefix_required_classes": [
                row["proposal_profile_id"]
                for row in profiles
                if row["scored_state_mode"] == "source_prefix"
            ],
            "native_adjacent_semantics": "every strict change_score >0.50 qualifying probe emits directly",
            "native_state_semantics": {
                "prototype_memory_4": "source-prefix bounded four-prototype memory; deterministic oldest eviction; strict >0.50 two-probe mutual-cosine >=0.50 creation",
            },
            "source_prefix_execution": "phase5_proposals.source_prefix_routes executes each profile/source once to the maximum tail and routes content-identical episode traces; scientific fixtures require exact equality with independent per-episode source-prefix replay",
            "confidence": "mean one-minus-cosine change strength over the one or two confirmation probes; higher means more change; never compare across confidence_semantics_id",
            "tail": "unconfirmed proposal state is suppressed and recorded at episode end; no future or artificial audio is added",
            "progress": "proposal progress is monotonic and retains the earliest pending retrospective boundary until confirmation, suppression, or tail closure",
        },
        "cluster_and_fusion_contract": {
            "clustering_algorithm": "PRD Section 9.2 exact causal order, boundary-radius membership, compatible-kind representative selection, strict refractory arrival cutoff, and tail closure",
            "max_confidence_tie_order": [
                "higher_confidence",
                "earlier_observation",
                "smaller_distance_to_compatible_subset_boundary_median",
                "earlier_boundary",
                "proposal_id",
            ],
            "queued_r0_rule": "an out-of-radius proposal already waiting at cluster availability opens at that availability when R=0 and is not spuriously suppressed",
            "fusion_order": [
                "observed_source_sample_at_emit",
                "boundary_source_sample",
                "vad_before_detector",
                "event_id",
            ],
            "association": "prior VAD suppresses a detector duplicate; earlier detector replaces/accelerates at most one later associated VAD; forbidden new-turn and structural associations remain separate",
            "safe_frontier": "minimum of proposal safe frontier and every known open-cluster boundary minus one; exact episode-end closure required",
        },
        "frequency_control_contract": {
            "scope": "all full-fusion policies, with exact per-episode detector-created hard-action count or explicit infeasibility",
            "detector_created_count": "add_hard_boundary plus detector-owned accelerate_or_replace_vad only; retained B0 VAD actions are excluded",
            "uniform": "deterministic quantile positions over the VAD-active source points causally known at each matched neural availability",
            "energy": "largest unused causal 250 ms left/right log-RMS change candidate inside VAD-active exposure available by the matched neural availability",
            "shuffle": "project every neural boundary to the nearest causally known VAD-active 512-sample point at its own availability; activate candidates only when their source availability is reached; choose the smallest sha256(seed, projected boundary, source availability, source action ID) unused rank while preserving each target action availability exactly",
            "placement_quantum_samples": 512,
            "ground_truth_generation_input": False,
            "infeasible": "visible per-action reason; never silently lower the required count",
        },
        "baseline_contract": {
            "b0": "accepted peer Silero lifecycle replay projected into each bounded episode",
            "b1": "the exact B0 action stream routed through Phase 5 schemas with no neural proposals",
            "identity_gate": "action kind, boundary source sample, observation frontier, lifecycle owner, and final segmentation must be exactly equal for every episode",
            "ls_eend": "432 historical rows remain visible as not_replayed_signal_stop; no LS neural sweep or new LS inference",
        },
        "policy_space": policy_info,
        "execution_workload": {
            "current_turn_episode_population": current_proposal_workload,
            "historical_204_case_population": {
                key: historical_development[key]
                for key in (
                    "proposal_profile_case_count",
                    "proposal_probe_step_count",
                    "logical_emittable_position_count",
                    "maximum_proposal_trace",
                    "maximum_emittable_proposal_position_count",
                    "cluster_execution_count",
                    "fusion_execution_count",
                    "frequency_control_execution_count",
                    "logical_policy_system_count",
                    "logical_policy_case_identity_count",
                    "baseline_system_count",
                    "baseline_case_identity_count",
                    "logical_system_count_including_baselines",
                    "logical_case_identity_count_including_baselines",
                    "b0_b1_equivalence_case_count",
                    "physical_policy_case_node_count",
                )
            },
            "total_proposal_probe_step_count": current_proposal_workload[
                "total_physical_probe_step_count"
            ]
            + historical_development["proposal_probe_step_count"],
            "current_320_sentinel_bounds_emitted_episode_positions_only_and_historical_11392_shape_is_measured_separately": True,
        },
        "policy_benchmark": {
            "path": POLICY_BENCHMARK_FILENAME,
            "content_sha256": benchmark["content_sha256"],
            "workers": benchmark["selected_policy_workers"],
            "physical_core_count": benchmark["hardware"]["physical_core_count"],
            "parallel_batches_per_second": benchmark["parallel"]["batches_per_second"],
            "conservative_batches_per_second": benchmark[
                "conservative_parallel_batches_per_second"
            ],
        },
        "storage_benchmark": {
            "path": STORAGE_BENCHMARK_FILENAME,
            "content_sha256": storage["content_sha256"],
            "sample_rows_per_representation": storage["representation"][
                "logical_proposal_route_index"
            ]["sample_row_count"],
            "projected_result_bytes": storage["projected_result_bytes"],
            "result_ceiling_bytes": storage["result_ceiling_bytes"],
            "result_headroom_bytes": storage["result_ceiling_bytes"]
            - storage["projected_result_bytes"],
            "within_result_ceiling": storage["within_result_ceiling"],
            "pool_order": storage["shared_schema"]["pool_order"],
            "pool_block_counts": storage["shared_schema"]["pool_block_counts"],
            "reconstruction_authority": storage["shared_schema"]["reconstruction_authority"],
        },
        "cache_reuse": {**window_info, **cache_presence(inventory)},
        "historical_correction": {
            **historical_input(experiment_dir, result_dir),
            "development_population": historical_development,
        },
        "scoring_and_statistics": {
            "baselines": ["B0", "B1"],
            "b0_b1_action_identity_required": True,
            "benefit_and_harm_axes_orthogonal": True,
            "structural_action_rule": "structural_max_duration remains in final segmentation and structural counts but is excluded from B0/candidate matching, benefit, duplicate, lexical, overlap, harmful, and same-speaker-extra-turn attribution",
            "unscored_interval_rule": "subtract explicit unscored spans from every exposure timeline; actions inside are excluded from matching, segmentation, contamination, harm, and fragmentation and contribute only unscored_action",
            "clean_gap_headline_excludes_overlap": True,
            "owner_threshold_ms": 100,
            "owner_sensitivity_ms": [50, 200],
            "harm_guard_ms": 200,
            "harm_guard_sensitivity_ms": [100, 300],
            "localization_tolerance_ms": [250, 500],
            "availability_deadlines_ms": [250, 500, 1000, 1500, 2000],
            "bootstrap_replicates": 10000,
            "bootstrap_unit": "source_connected_block",
            "bootstrap_seed": "sha256('turn_episode_v1|phase5|block_bootstrap|' + system_id) first 64 bits",
            "frontier_or_panel_construction": False,
            "natural_rates_only_from": "natural_exposure_validation",
            "historical_rows_selection_eligible": False,
        },
        "outputs": {
            "physical_proposal_execution_receipts": "phase_5_proposal_executions/<shard>.jsonl.gz",
            "logical_proposal_routes": "phase_5_proposal_routes/<shard>.jsonl.gz",
            "physical_system_definitions": "phase_5_physical_systems/<shard>.jsonl.gz",
            "logical_system_definitions": "phase_5_logical_systems/<shard>.jsonl.gz",
            "logical_alias_edges": "phase_5_alias_edges/<shard>.jsonl.gz",
            "current_system_block_aggregates": "phase_5_current_aggregates/<shard>.jsonl.gz",
            "historical_corrected_system_aggregates": "phase_5_historical_aggregates/<shard>.jsonl.gz",
            "deterministic_failure_examples": "phase_5_failure_examples/<shard>.jsonl.gz",
            "independent_audit_units": "phase_5_audit_units/<shard>.jsonl.gz",
            "phase_5_summary": "phase_5_development_summary.json",
            "phase_5_verification": "phase_5_verification.json",
            "phase_5_completion": "phase_5_completion.json",
            "atomic_partial_rule": "per-shard temporary write, count/hash/size validation, then atomic replace",
            "aggregate_json_limit_bytes": 10 * 1024**2,
            "detail_shard_limit_bytes": 20 * 1024**2,
            "single_large_json_forbidden": True,
            "content_addressing": "typed proposal reconstruction receipts bind every accepted embedding/cache/lifecycle/profile/code input, proposal and progress count/digest, state snapshot/final state, and tail evidence; derived cluster, fusion, control, match, harm, contamination, and timing traces are sampled by the frozen audit contract rather than persisted individually; every logical identity still contributes to an exhaustive ordered digest and pool-separated aggregate",
            "empty_trace_rule": "materialize one canonical empty trace definition and bind every empty logical identity through the policy index digest",
            "logical_row_completeness": "per-pool counts plus ordered rolling SHA-256 over every logical policy-episode identity and referenced node hash",
            "dense_metric_rule": "block and system aggregates use deterministic gzip JSONL shards; no aggregate JSON may exceed 10 MiB",
            "reconstruction_rule": "the accepted embedding/cache receipts, episode/historical manifests, typed B0/B1 lifecycle evidence and exact equivalence receipt, typed proposal reconstruction receipts, exact profile/system definitions, and reviewed code hashes are the reconstruction authority; the independent verifier exhaustively checks identities/completeness/aggregates and recomputes the frozen 2,048-unit raw/derived audit plus mandatory sentinels and failure examples",
            "storage_projection": "4,096-row deterministic high-entropy pilots cover nine typed persisted representations: complete proposal reconstruction receipts, logical proposal routes, physical/logical system definitions, alias edges, explicit three-pool/43-pool-block current aggregates, 204-case historical corrected neural plus B0/B1 aggregates and identity digests, deterministic failure examples, and the 2,048-unit audit index; there is no untyped reserve; repeat the guard at the interstage gate and stop before Stage B if the 8 GiB result ceiling or 20 MiB shard ceiling would fail",
        },
        "completeness": {
            "expected_logical_episode_identity_count": policy_info[
                "logical_episode_identity_count"
            ],
            "expected_historical_input_row_count": 1369,
            "expected_historical_logical_policy_case_identity_count": historical_development[
                "logical_policy_case_identity_count"
            ],
            "expected_historical_baseline_case_identity_count": historical_development[
                "baseline_case_identity_count"
            ],
            "expected_historical_logical_case_identity_count_including_baselines": historical_development[
                "logical_case_identity_count_including_baselines"
            ],
            "expected_historical_physical_policy_case_node_count": historical_development[
                "physical_policy_case_node_count"
            ],
            "every_system_pool_episode_key_unique": True,
            "every_current_aggregate_has_all_three_pool_rows": True,
            "every_current_pool_block_row_uses_frozen_43_row_index": True,
            "natural_rate_numerator_and_denominator_exist_only_in_natural_pool_row": True,
            "every_action_references_one_cluster_or_baseline_lifecycle_action": True,
            "every_control_matches_tested_hard_action_count_or_records_infeasible": True,
            "incomplete_shards_cannot_enter_aggregate": True,
            "every_logical_action_trace_reconstructible_from_accepted_inputs": True,
            "ordered_action_digest_recomputed_independently": True,
            "independent_raw_derived_audit_sample_size": INDEPENDENT_AUDIT_SAMPLE_SIZE,
        },
        "exit_gate": {
            "proposal_schema_and_causal_timing": "zero mismatches",
            "b0_b1_action_identity": "every episode exact",
            "logical_policy_episode_count": policy_info["logical_episode_identity_count"],
            "historical_logical_policy_case_count": historical_development[
                "logical_policy_case_identity_count"
            ],
            "historical_baseline_case_count": historical_development[
                "baseline_case_identity_count"
            ],
            "historical_logical_case_count_including_baselines": historical_development[
                "logical_case_identity_count_including_baselines"
            ],
            "controls": "every required count matched or explicit infeasible row",
            "exhaustive_recomputation": [
                "file and self hashes",
                "expected identities and split completeness",
                "B0/B1 equivalence and generation-time causal/schema guards",
                "per-session, per-pool, and per-block aggregates",
                "summary arithmetic and ordered identity digests",
            ],
            "sampled_raw_derived_recomputation": [
                "cluster membership and refractory ownership",
                "fusion actions and duplicate suppression",
                "ordered reference matching",
                "clean/gap contamination samples and denominator",
                "orthogonal harm and structure flags",
                "timing and safe-frontier progression",
                "natural-exposure-only rate labels",
                "historical correction disposition",
            ],
            "sampled_raw_derived_unit_count": INDEPENDENT_AUDIT_SAMPLE_SIZE,
            "phase_6_preparation": "forbidden after acceptance until explicit owner resume",
        },
        "runtime_forecast": forecast,
        "execution_boundaries": {
            "production_wiring": "forbidden",
            "public_entrypoints": "forbidden",
            "confirmatory_heldout": "forbidden",
            "provider_calls": "forbidden",
            "network": "forbidden",
            "credentials": "forbidden",
            "phase_6_frontier_or_freeze": "forbidden until owner resumes after verified Phase 5 report",
        },
        "generated_from": {
            name: sha256_file(script_path.with_name(name))
            for name in (
                "phase5_controls.py",
                "phase5_design.py",
                "phase5_inputs.py",
                "phase5_policy.py",
                "phase5_policy_benchmark.py",
                "phase5_proposals.py",
                "phase5_scoring.py",
                "phase5_storage_benchmark.py",
            )
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    experiment_dir = args.experiment_dir.resolve()
    output = args.output or (
        experiment_dir / "results" / "turn_episode_v1" / "phase_5_design_ledger.json"
    )
    payload = build_payload(experiment_dir)
    atomic_write_json(output, payload)
    written = json.loads(output.read_text(encoding="utf-8"))
    print(
        canonical_json(
            {
                "path": str(output),
                "content_sha256": written["content_sha256"],
                "proposal_profile_count": len(
                    written["family_compute_envelopes"]["eres2netv2"]["proposal_profiles"]
                ),
                "policy_episode_rows": written["completeness"][
                    "expected_logical_episode_identity_count"
                ],
                "forecast_hours": written["runtime_forecast"]["total_forecast_hours"],
            }
        )
    )


if __name__ == "__main__":
    main()
