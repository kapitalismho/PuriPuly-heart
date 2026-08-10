from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from experiments.speaker_representation_scd.provenance import self_sha256_valid

EXPERIMENT_ID = "speaker_representation_scd_v1"
PROTOCOL_VERSION = "r0-1"
AUTHORITY_PATH = "experiments/speaker_representation_scd/EXPERIMENT_PLAN.en.md"
AUTHORITY_SHA256 = "ca46bce33b90c89597b5c9f2092b952a3f76d638c9c5524d4ca7ba23800e9191"
SCHEMA_VERSION = 1
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
LICENSE_STATUSES = frozenset(
    {"research_allowed", "product_allowed", "restricted", "unknown"}
)
SPLITS = frozenset({"development_known", "confirmatory_test", "future_train"})
ROLES = frozenset(
    {
        "r0_protocol",
        "analysis_contract",
        "compute_ceiling",
        "license_disposition",
        "source_ledger",
        "split_contract",
        "confirmatory_access_policy",
        "model_registry",
    }
)
EXPECTED_MODEL_IDENTITIES = {
    "mhubert-147": (
        "7ad3fc0bc5106c58c9c13526abccad527150d135",
        "2359b3e9dc6869cb0855119a2866f056aeb400e46252da9cbcc8e9b7aee50c8b",
        377510584,
    ),
    "wavlm-base-plus": (
        "4c66d4806a428f2e922ccfa1a962776e232d487b",
        "3bb273a6ace99408b50cfc81afdbb7ef2de02da2eab0234e18db608ce692fe51",
        377617425,
    ),
    "unispeech-sat-base-plus": (
        "74f559583458188867750f1b8cb6710b11f5be41",
        "0ebc4dd3edc1e10e21a4d16791ad65b9217033d9205317e999a973304b27eda4",
        382236294,
    ),
    "eres2netv2-standard-prepool": (
        "1cf80d41fb3435bd3d8df185b5c423333b2db42a",
        "0eb4057106b2573dd7b132cf0c36273ab29afd192c1610f80baa9c556dbb963c",
        71768231,
    ),
}
EXPECTED_SOURCE_IDS = frozenset(
    {
        "legacy-common-gt-v1",
        "zeroth-korean-development",
        "jvs-development",
        "voxconverse-v03-confirmatory-natural",
        "aishell4-confirmatory-natural-zh",
        "zeroth-korean-confirmatory",
        "jvs-confirmatory",
    }
)
EXPECTED_LEGACY_BYTE_SHA256 = "a5df5e56c2917ba174fca7892fbaa092f2b57705cad7a77bf9a347ba94cddfee"
EXPECTED_LEGACY_CONTENT_SHA256 = "deb1713cd581c93cc13c47103643041c7551993985379869cfa0ca9a407dff68"
EXPECTED_SELF_SHA256 = {
    "r0_protocol": "b104bee3f76356916cac48250eca6d45fe7dce695f43c8f768aec74d911c3e35",
    "analysis_contract": "eb070c15d930d00b3b5f791ba482b34fb1210d504e8b0513f0b8d55f097051e1",
    "compute_ceiling": "29b0225a39f38d9a03584286f932be2722f276be95cfa922bda305155ef312c6",
    "license_disposition": "a9c027a006a9fb1aa5207e25fd37a688135f35885be30f415b5deafe0d4e855d",
    "source_ledger": "26a29b8bc8ab7ff7a4d52459d04e2bb112cd60044001ebe74797b27520ad97fc",
    "split_contract": "df90e37bca150675bbb4780363fb41a6dd57f64fc505910a35b3cae5b1bf97e8",
    "confirmatory_access_policy": "dafae80be50a36f78f116ccf80964e712df5a5869893eba0a0c538e1e38efa8b",
    "model_registry": "54d0538fa74130dcfafc2b272f3399de9ca1269f15c35e678432884b26e789a5",
}


class ContractError(ValueError):
    pass


def _error(errors: list[str], path: str, message: str) -> None:
    errors.append(f"{path}: {message}")


def _exact_keys(value: Any, expected: Iterable[str], path: str, errors: list[str]) -> bool:
    if not isinstance(value, dict):
        _error(errors, path, "must be an object")
        return False
    expected_set = set(expected)
    actual = set(value)
    missing = sorted(expected_set - actual)
    unknown = sorted(actual - expected_set)
    if missing:
        _error(errors, path, f"missing keys {missing}")
    if unknown:
        _error(errors, path, f"unknown keys {unknown}")
    return not missing and not unknown


def _nonempty_string(value: Any, path: str, errors: list[str]) -> None:
    if not isinstance(value, str) or not value:
        _error(errors, path, "must be a non-empty string")


def _hash(value: Any, path: str, errors: list[str], nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if not isinstance(value, str) or HASH_PATTERN.fullmatch(value) is None:
        _error(errors, path, "must be a lowercase SHA-256")


def _revision(value: Any, path: str, errors: list[str], nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if not isinstance(value, str) or REVISION_PATTERN.fullmatch(value) is None:
        _error(errors, path, "must be an immutable 40-character Git revision")


def _positive_int(value: Any, path: str, errors: list[str]) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        _error(errors, path, "must be a positive integer")


def _positive_number(value: Any, path: str, errors: list[str]) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        _error(errors, path, "must be a positive number")


def _unique_strings(value: Any, path: str, errors: list[str], allow_empty: bool = False) -> None:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        _error(errors, path, "must be a list of non-empty strings")
        return
    if not allow_empty and not value:
        _error(errors, path, "must not be empty")
    if len(value) != len(set(value)):
        _error(errors, path, "must not contain duplicates")


def _validate_common(document: dict[str, Any], expected_role: str, errors: list[str]) -> None:
    common = {
        "schema_version",
        "artifact_role",
        "experiment_id",
        "protocol_version",
        "authority",
        "self_sha256",
    }
    if not common <= set(document):
        _error(errors, expected_role, f"missing common keys {sorted(common - set(document))}")
        return
    if document["schema_version"] != SCHEMA_VERSION:
        _error(errors, f"{expected_role}.schema_version", f"must equal {SCHEMA_VERSION}")
    if document["artifact_role"] != expected_role:
        _error(errors, f"{expected_role}.artifact_role", f"must equal {expected_role}")
    if document["experiment_id"] != EXPERIMENT_ID:
        _error(errors, f"{expected_role}.experiment_id", f"must equal {EXPERIMENT_ID}")
    if document["protocol_version"] != PROTOCOL_VERSION:
        _error(errors, f"{expected_role}.protocol_version", f"must equal {PROTOCOL_VERSION}")
    authority = document["authority"]
    if _exact_keys(authority, {"path", "sha256"}, f"{expected_role}.authority", errors):
        if authority["path"] != AUTHORITY_PATH:
            _error(errors, f"{expected_role}.authority.path", f"must equal {AUTHORITY_PATH}")
        if authority["sha256"] != AUTHORITY_SHA256:
            _error(
                errors,
                f"{expected_role}.authority.sha256",
                f"must equal {AUTHORITY_SHA256}",
            )
    _hash(document["self_sha256"], f"{expected_role}.self_sha256", errors)
    if not self_sha256_valid(document):
        _error(errors, f"{expected_role}.self_sha256", "does not match canonical content")
    if document["self_sha256"] != EXPECTED_SELF_SHA256[expected_role]:
        _error(errors, f"{expected_role}.self_sha256", "does not match the frozen R0 contract")


def _validate_protocol(document: dict[str, Any], errors: list[str]) -> None:
    keys = {
        "schema_version",
        "artifact_role",
        "experiment_id",
        "protocol_version",
        "authority",
        "self_sha256",
        "scope",
        "timeline",
        "context_modes",
        "representation_grid",
        "event_taxonomy",
        "speech_activity_conditions",
        "comparison_panels",
        "contract_paths",
        "amendment_policy",
    }
    _exact_keys(document, keys, "r0_protocol", errors)
    scope = document.get("scope", {})
    if _exact_keys(
        scope,
        {"authorized_phases", "deferred_phases", "training_authorized", "public_data_only"},
        "r0_protocol.scope",
        errors,
    ):
        if scope["authorized_phases"] != ["R0", "R1", "R2", "R3", "R4", "R6-Z"]:
            _error(errors, "r0_protocol.scope.authorized_phases", "unexpected phase sequence")
        if scope["training_authorized"] is not False:
            _error(errors, "r0_protocol.scope.training_authorized", "must be false")
        if scope["public_data_only"] is not True:
            _error(errors, "r0_protocol.scope.public_data_only", "must be true")
        if not {"R5", "R6-T", "R7", "R8", "R9"} <= set(scope["deferred_phases"]):
            _error(errors, "r0_protocol.scope.deferred_phases", "learned phases must be deferred")
    timeline = document.get("timeline", {})
    if _exact_keys(
        timeline,
        {"sample_rate_hz", "coordinate_unit", "interval_convention", "event_times"},
        "r0_protocol.timeline",
        errors,
    ):
        if timeline["sample_rate_hz"] != 16000:
            _error(errors, "r0_protocol.timeline.sample_rate_hz", "must equal 16000")
        if timeline["coordinate_unit"] != "source_sample":
            _error(errors, "r0_protocol.timeline.coordinate_unit", "must be source_sample")
        if timeline["interval_convention"] != "half_open":
            _error(errors, "r0_protocol.timeline.interval_convention", "must be half_open")
        if timeline["event_times"] != [
            "boundary_source_sample",
            "observed_source_sample_at_emit",
            "compute_completed_monotonic_ns",
        ]:
            _error(errors, "r0_protocol.timeline.event_times", "must freeze the three-time contract")
    modes = document.get("context_modes")
    if not isinstance(modes, list) or {item.get("id") for item in modes if isinstance(item, dict)} != {
        "local_trailing_window",
        "left_context_tail_pool",
        "offline_full_context",
    }:
        _error(errors, "r0_protocol.context_modes", "must define all three context modes")
    grid = document.get("representation_grid", {})
    if _exact_keys(
        grid,
        {"pooling_ms", "continuous_hop_ms", "ssl_layers", "pooling", "normalization"},
        "r0_protocol.representation_grid",
        errors,
    ):
        if grid["pooling_ms"] != [100, 200, 300, 500, 750, 1000]:
            _error(errors, "r0_protocol.representation_grid.pooling_ms", "unexpected pooling grid")
        if grid["continuous_hop_ms"] != {"primary": 50, "sensitivity": [20, 100]}:
            _error(errors, "r0_protocol.representation_grid.continuous_hop_ms", "unexpected hops")
        if grid["ssl_layers"] != ["L1", "L3", "L6", "L9", "L12"]:
            _error(errors, "r0_protocol.representation_grid.ssl_layers", "unexpected layers")
        if grid["pooling"] != "valid_frame_mean" or grid["normalization"] != "l2_after_pooling":
            _error(errors, "r0_protocol.representation_grid", "primary reducer is not frozen")
    taxonomy = document.get("event_taxonomy", {})
    if taxonomy.get("primary") != [
        "new_speaker_onset_clean",
        "new_speaker_onset_gap",
        "overlap_start_new_speaker",
    ]:
        _error(errors, "r0_protocol.event_taxonomy.primary", "unexpected primary targets")
    if document.get("speech_activity_conditions") != [
        "common_causal_vad",
        "ungated_full_stream",
        "oracle_activity",
    ]:
        _error(errors, "r0_protocol.speech_activity_conditions", "unexpected activity panels")
    paths = document.get("contract_paths")
    if not isinstance(paths, dict) or set(paths) != ROLES - {"r0_protocol"}:
        _error(errors, "r0_protocol.contract_paths", "must name every linked R0 contract")


def _validate_analysis(document: dict[str, Any], errors: list[str]) -> None:
    keys = {
        "schema_version",
        "artifact_role",
        "experiment_id",
        "protocol_version",
        "authority",
        "self_sha256",
        "primary_endpoints",
        "metrics",
        "matching",
        "uncertainty",
        "promotion",
        "missing_reasons",
        "reporting",
    }
    _exact_keys(document, keys, "analysis_contract", errors)
    endpoints = document.get("primary_endpoints", {})
    if endpoints.get("representation") != {
        "metric": "prototype_distance_roc_auc_and_eer",
        "key_contexts_ms": [300, 500],
    }:
        _error(errors, "analysis_contract.primary_endpoints.representation", "unexpected endpoint")
    continuous = endpoints.get("continuous_zero_shot", {})
    if continuous.get("selection") != "integer_false_event_pareto_frontier":
        _error(errors, "analysis_contract.primary_endpoints.continuous_zero_shot.selection", "must use integer frontier")
    if continuous.get("reference_false_events_per_hour") != 1:
        _error(errors, "analysis_contract.primary_endpoints.continuous_zero_shot.reference_false_events_per_hour", "must equal 1")
    matching = document.get("matching", {})
    if matching.get("cardinality") != "one_to_one" or matching.get("optimization") != [
        "maximize_matches",
        "minimize_availability_latency",
        "minimize_absolute_localization_error",
    ]:
        _error(errors, "analysis_contract.matching", "deterministic matching is not frozen")
    uncertainty = document.get("uncertainty", {})
    if uncertainty.get("bootstrap_replicates") != 10000 or uncertainty.get("unit") != "whole_connected_block":
        _error(errors, "analysis_contract.uncertainty", "whole-block bootstrap is not frozen")
    promotion = document.get("promotion", {})
    if promotion.get("test_rows_per_encoder") != 1 or promotion.get("sentinel_per_encoder") != 1:
        _error(errors, "analysis_contract.promotion", "one locked row and sentinel are required")
    _unique_strings(document.get("missing_reasons"), "analysis_contract.missing_reasons", errors)


def _validate_compute(document: dict[str, Any], errors: list[str]) -> None:
    keys = {
        "schema_version",
        "artifact_role",
        "experiment_id",
        "protocol_version",
        "authority",
        "self_sha256",
        "hardware_identity",
        "ceilings",
        "smoke_gate",
        "legacy_contention_guard",
        "execution_state",
    }
    _exact_keys(document, keys, "compute_ceiling", errors)
    hardware = document.get("hardware_identity", {})
    if hardware.get("backend") != "cpu" or hardware.get("gpu") != "none_visible":
        _error(errors, "compute_ceiling.hardware_identity", "must identify the accepted local CPU path")
    ceilings = document.get("ceilings", {})
    expected = {
        "max_parallel_models",
        "max_worker_processes",
        "max_cpu_threads",
        "max_resident_ram_gib",
        "max_source_download_gib",
        "max_derived_cache_gib",
        "max_external_storage_gib",
        "min_free_disk_gib_before_download",
        "max_total_wall_hours",
        "max_per_model_wall_hours",
    }
    if _exact_keys(ceilings, expected, "compute_ceiling.ceilings", errors):
        for name, value in ceilings.items():
            _positive_number(value, f"compute_ceiling.ceilings.{name}", errors)
        if ceilings["max_parallel_models"] != 1 or ceilings["max_worker_processes"] != 1:
            _error(errors, "compute_ceiling.ceilings", "execution must remain sequential")
        if ceilings["max_source_download_gib"] > ceilings["max_external_storage_gib"]:
            _error(errors, "compute_ceiling.ceilings", "download ceiling exceeds external storage")
        if ceilings["max_derived_cache_gib"] > ceilings["max_external_storage_gib"]:
            _error(errors, "compute_ceiling.ceilings", "cache ceiling exceeds external storage")
    smoke = document.get("smoke_gate", {})
    if smoke.get("fixtures_per_model") != 10 or smoke.get("benchmark_windows_per_model") != 100:
        _error(errors, "compute_ceiling.smoke_gate", "10-fixture/100-window smoke is required")
    if smoke.get("forecast_required") is not True or smoke.get("fail_closed") is not True:
        _error(errors, "compute_ceiling.smoke_gate", "forecast gate must fail closed")
    guard = document.get("legacy_contention_guard", {})
    if guard.get("required_state") != "inactive" or guard.get("on_conflict") != "abort":
        _error(errors, "compute_ceiling.legacy_contention_guard", "legacy contention must abort")
    state = document.get("execution_state", {})
    if state.get("full_extraction_enabled") is not False:
        _error(errors, "compute_ceiling.execution_state.full_extraction_enabled", "must remain false at R0")


def _validate_license(document: dict[str, Any], errors: list[str]) -> None:
    keys = {
        "schema_version",
        "artifact_role",
        "experiment_id",
        "protocol_version",
        "authority",
        "self_sha256",
        "allowed_statuses",
        "policy",
        "models",
    }
    _exact_keys(document, keys, "license_disposition", errors)
    if set(document.get("allowed_statuses", [])) != LICENSE_STATUSES:
        _error(errors, "license_disposition.allowed_statuses", "unexpected status vocabulary")
    policy = document.get("policy", {})
    if policy.get("restricted_models_research_only") is not True:
        _error(errors, "license_disposition.policy", "restricted models must be research-only")
    models = document.get("models")
    if not isinstance(models, list) or len(models) != 4:
        _error(errors, "license_disposition.models", "must contain four model dispositions")
        return
    ids: list[str] = []
    for index, model in enumerate(models):
        path = f"license_disposition.models[{index}]"
        if not _exact_keys(
            model,
            {"model_id", "license_id", "license_sha256", "research_status", "product_status", "product_claim_allowed", "notes"},
            path,
            errors,
        ):
            continue
        ids.append(model["model_id"])
        if model["research_status"] not in LICENSE_STATUSES or model["product_status"] not in LICENSE_STATUSES:
            _error(errors, path, "contains an invalid license status")
        _hash(model["license_sha256"], f"{path}.license_sha256", errors, nullable=True)
        if model["product_status"] != "product_allowed" and model["product_claim_allowed"] is not False:
            _error(errors, f"{path}.product_claim_allowed", "must be false without product_allowed status")
    if len(ids) != len(set(ids)):
        _error(errors, "license_disposition.models", "model IDs must be unique")


def _validate_sources(document: dict[str, Any], errors: list[str]) -> None:
    keys = {
        "schema_version",
        "artifact_role",
        "experiment_id",
        "protocol_version",
        "authority",
        "self_sha256",
        "sources",
        "forbidden_inheritance",
    }
    _exact_keys(document, keys, "source_ledger", errors)
    sources = document.get("sources")
    if not isinstance(sources, list) or not sources:
        _error(errors, "source_ledger.sources", "must be a non-empty list")
        return
    ids: list[str] = []
    for index, source in enumerate(sources):
        path = f"source_ledger.sources[{index}]"
        expected = {
            "source_id",
            "corpus",
            "tier",
            "language",
            "claim_scope",
            "release",
            "source_url",
            "license_id",
            "usage_status",
            "acquisition_status",
            "artifact_sha256",
            "artifact_size_bytes",
            "selection",
        }
        if not _exact_keys(source, expected, path, errors):
            continue
        ids.append(source["source_id"])
        if source["tier"] not in SPLITS - {"future_train"}:
            _error(errors, f"{path}.tier", "invalid evidence tier")
        if source["usage_status"] not in LICENSE_STATUSES:
            _error(errors, f"{path}.usage_status", "invalid usage status")
        if source["acquisition_status"] == "existing_verified":
            _hash(source["artifact_sha256"], f"{path}.artifact_sha256", errors)
            _positive_int(source["artifact_size_bytes"], f"{path}.artifact_size_bytes", errors)
        elif source["acquisition_status"] in {"unacquired", "metadata_only"}:
            if source["artifact_sha256"] is not None or source["artifact_size_bytes"] is not None:
                _error(errors, path, "unacquired sources cannot claim binary identity")
        else:
            _error(errors, f"{path}.acquisition_status", "invalid acquisition status")
        if source["tier"] == "confirmatory_test" and source["selection"].get("frozen") is not True:
            _error(errors, f"{path}.selection", "confirmatory selection must be frozen")
    if len(ids) != len(set(ids)):
        _error(errors, "source_ledger.sources", "source IDs must be unique")
    if set(ids) != EXPECTED_SOURCE_IDS:
        _error(errors, "source_ledger.sources", "source identity set differs from the frozen ledger")
    by_id = {source.get("source_id"): source for source in sources if isinstance(source, dict)}
    legacy = by_id.get("legacy-common-gt-v1", {})
    if legacy.get("artifact_sha256") != EXPECTED_LEGACY_BYTE_SHA256:
        _error(errors, "source_ledger.sources.legacy-common-gt-v1", "legacy byte identity changed")
    if legacy.get("selection", {}).get("canonical_content_sha256") != EXPECTED_LEGACY_CONTENT_SHA256:
        _error(errors, "source_ledger.sources.legacy-common-gt-v1", "legacy content identity changed")
    forbidden = set(document.get("forbidden_inheritance", []))
    required = {"legacy_thresholds", "legacy_shortlists", "legacy_go_stop", "legacy_feature_caches", "legacy_reducer_state"}
    if not required <= forbidden:
        _error(errors, "source_ledger.forbidden_inheritance", "legacy inheritance guard is incomplete")


def component_leakage(assignments: list[dict[str, Any]]) -> list[str]:
    owners: dict[tuple[str, str], set[str]] = {}
    for assignment in assignments:
        key = (str(assignment.get("namespace")), str(assignment.get("component_id")))
        owners.setdefault(key, set()).add(str(assignment.get("tier")))
    return [f"{namespace}:{component}" for (namespace, component), tiers in owners.items() if len(tiers) > 1]


def _validate_split(document: dict[str, Any], errors: list[str]) -> None:
    keys = {
        "schema_version",
        "artifact_role",
        "experiment_id",
        "protocol_version",
        "authority",
        "self_sha256",
        "tiers",
        "legacy_manifest",
        "component_assignments",
        "selection_rules",
        "leakage_policy",
    }
    _exact_keys(document, keys, "split_contract", errors)
    if set(document.get("tiers", [])) != SPLITS:
        _error(errors, "split_contract.tiers", "unexpected split vocabulary")
    legacy = document.get("legacy_manifest", {})
    if legacy.get("tier") != "development_known":
        _error(errors, "split_contract.legacy_manifest.tier", "legacy data must be development-known")
    _hash(legacy.get("byte_sha256"), "split_contract.legacy_manifest.byte_sha256", errors)
    _hash(legacy.get("content_sha256"), "split_contract.legacy_manifest.content_sha256", errors)
    if legacy.get("byte_sha256") != EXPECTED_LEGACY_BYTE_SHA256:
        _error(errors, "split_contract.legacy_manifest.byte_sha256", "legacy byte identity changed")
    if legacy.get("content_sha256") != EXPECTED_LEGACY_CONTENT_SHA256:
        _error(errors, "split_contract.legacy_manifest.content_sha256", "legacy content identity changed")
    assignments = document.get("component_assignments")
    if not isinstance(assignments, list) or not assignments:
        _error(errors, "split_contract.component_assignments", "must be a non-empty list")
        return
    for index, assignment in enumerate(assignments):
        path = f"split_contract.component_assignments[{index}]"
        if not _exact_keys(assignment, {"namespace", "component_id", "tier", "source_id"}, path, errors):
            continue
        if assignment["tier"] not in SPLITS:
            _error(errors, f"{path}.tier", "invalid tier")
    leaked = component_leakage(assignments)
    if leaked:
        _error(errors, "split_contract.component_assignments", f"cross-tier leakage {sorted(leaked)}")
    policy = document.get("leakage_policy", {})
    if policy.get("connected_component_blocking") is not True or policy.get("on_overlap") != "reject":
        _error(errors, "split_contract.leakage_policy", "must reject connected-component overlap")


def _lock_complete(lock: dict[str, Any]) -> bool:
    required_hashes = [
        "run_contract_sha256",
        "code_sha256",
        "model_registry_sha256",
        "split_contract_sha256",
        "analysis_contract_sha256",
        "evaluation_environment_sha256",
        "development_promotion_ledger_sha256",
        "verifier_sha256",
    ]
    configurations = lock.get("locked_encoder_configuration_ids")
    return (
        all(
            isinstance(lock.get(key), str) and HASH_PATTERN.fullmatch(lock[key])
            for key in required_hashes
        )
        and isinstance(configurations, list)
        and len(configurations) == 4
        and len(set(configurations)) == 4
        and all(isinstance(item, str) and item for item in configurations)
        and isinstance(lock.get("locked_at_utc"), str)
        and bool(lock["locked_at_utc"])
    )


def _validate_access(document: dict[str, Any], errors: list[str]) -> None:
    keys = {
        "schema_version",
        "artifact_role",
        "experiment_id",
        "protocol_version",
        "authority",
        "self_sha256",
        "state",
        "allowed_prelock_operations",
        "forbidden_prelock_operations",
        "lock_requirements",
        "lock",
        "access_ledger",
    }
    _exact_keys(document, keys, "confirmatory_access_policy", errors)
    state = document.get("state")
    if state != "sealed":
        _error(errors, "confirmatory_access_policy.state", "r0-1 is seal-only")
    forbidden = set(document.get("forbidden_prelock_operations", []))
    if not {"audio_content", "annotation_content", "derived_gt", "features", "scores", "aggregates"} <= forbidden:
        _error(errors, "confirmatory_access_policy.forbidden_prelock_operations", "forbidden operations are incomplete")
    allowed = set(document.get("allowed_prelock_operations", []))
    if allowed & forbidden:
        _error(errors, "confirmatory_access_policy.allowed_prelock_operations", "overlaps forbidden operations")
    lock = document.get("lock", {})
    _exact_keys(
        lock,
        {
            "run_contract_sha256",
            "code_sha256",
            "model_registry_sha256",
            "split_contract_sha256",
            "analysis_contract_sha256",
            "evaluation_environment_sha256",
            "development_promotion_ledger_sha256",
            "verifier_sha256",
            "locked_encoder_configuration_ids",
            "locked_at_utc",
        },
        "confirmatory_access_policy.lock",
        errors,
    )
    ledger = document.get("access_ledger")
    if not isinstance(ledger, list):
        _error(errors, "confirmatory_access_policy.access_ledger", "must be a list")
        ledger = []
    for key, value in lock.items():
        expected = [] if key == "locked_encoder_configuration_ids" else None
        if value != expected:
            _error(errors, f"confirmatory_access_policy.lock.{key}", "must remain empty in seal-only r0-1")
    if ledger:
        _error(errors, "confirmatory_access_policy.access_ledger", "must be empty while sealed")


def _validate_registry(document: dict[str, Any], errors: list[str]) -> None:
    keys = {
        "schema_version",
        "artifact_role",
        "experiment_id",
        "protocol_version",
        "authority",
        "self_sha256",
        "models",
    }
    _exact_keys(document, keys, "model_registry", errors)
    models = document.get("models")
    if not isinstance(models, list) or len(models) != 4:
        _error(errors, "model_registry.models", "must contain four primary encoders")
        return
    ids: list[str] = []
    for index, model in enumerate(models):
        path = f"model_registry.models[{index}]"
        expected = {
            "model_id",
            "family",
            "repository",
            "revision",
            "artifact",
            "metadata",
            "sample_rate_hz",
            "layer_or_tap_candidates",
            "license_id",
            "acquisition_status",
            "extraction_status",
            "unresolved",
        }
        if not _exact_keys(model, expected, path, errors):
            continue
        ids.append(model["model_id"])
        _revision(model["revision"], f"{path}.revision", errors)
        artifact = model["artifact"]
        if _exact_keys(artifact, {"file_name", "sha256", "size_bytes", "format"}, f"{path}.artifact", errors):
            _hash(artifact["sha256"], f"{path}.artifact.sha256", errors)
            _positive_int(artifact["size_bytes"], f"{path}.artifact.size_bytes", errors)
        metadata = model["metadata"]
        if _exact_keys(metadata, {"config_sha256", "processor_sha256", "parameter_count", "parameter_count_status"}, f"{path}.metadata", errors):
            _hash(metadata["config_sha256"], f"{path}.metadata.config_sha256", errors, nullable=True)
            _hash(metadata["processor_sha256"], f"{path}.metadata.processor_sha256", errors, nullable=True)
            if metadata["parameter_count"] is not None:
                _positive_int(metadata["parameter_count"], f"{path}.metadata.parameter_count", errors)
        if model["sample_rate_hz"] != 16000:
            _error(errors, f"{path}.sample_rate_hz", "must equal 16000")
        _unique_strings(model["layer_or_tap_candidates"], f"{path}.layer_or_tap_candidates", errors)
        if model["acquisition_status"] != "metadata_pinned_unacquired":
            _error(errors, f"{path}.acquisition_status", "R0 must not claim acquired weights")
    if len(ids) != len(set(ids)):
        _error(errors, "model_registry.models", "model IDs must be unique")
    if set(ids) != set(EXPECTED_MODEL_IDENTITIES):
        _error(errors, "model_registry.models", "model identity set differs from the frozen registry")
    by_id = {model.get("model_id"): model for model in models if isinstance(model, dict)}
    for model_id, (revision, artifact_hash, artifact_size) in EXPECTED_MODEL_IDENTITIES.items():
        model = by_id.get(model_id, {})
        if model.get("revision") != revision:
            _error(errors, f"model_registry.models.{model_id}.revision", "frozen revision changed")
        artifact = model.get("artifact", {})
        if artifact.get("sha256") != artifact_hash or artifact.get("size_bytes") != artifact_size:
            _error(errors, f"model_registry.models.{model_id}.artifact", "frozen artifact identity changed")


VALIDATORS = {
    "r0_protocol": _validate_protocol,
    "analysis_contract": _validate_analysis,
    "compute_ceiling": _validate_compute,
    "license_disposition": _validate_license,
    "source_ledger": _validate_sources,
    "split_contract": _validate_split,
    "confirmatory_access_policy": _validate_access,
    "model_registry": _validate_registry,
}


def validate_document(document: dict[str, Any], expected_role: str) -> list[str]:
    errors: list[str] = []
    if expected_role not in VALIDATORS:
        raise ContractError(f"unknown artifact role {expected_role!r}")
    _validate_common(document, expected_role, errors)
    VALIDATORS[expected_role](document, errors)
    return errors
