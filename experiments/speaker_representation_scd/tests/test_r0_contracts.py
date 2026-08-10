from __future__ import annotations

import copy
from pathlib import Path

from experiments.speaker_representation_scd.provenance import load_json, with_self_sha256
from experiments.speaker_representation_scd.schemas import validate_document
from experiments.speaker_representation_scd.validate_r0 import validate_bundle

ROOT = Path(__file__).resolve().parents[1]


def _document(relative: str) -> dict:
    return load_json(ROOT / relative)


def _validated_mutation(document: dict, mutation) -> list[str]:
    changed = copy.deepcopy(document)
    mutation(changed)
    changed = with_self_sha256(changed)
    return validate_document(changed, document["artifact_role"])


def test_r0_bundle_is_valid_but_execution_remains_fail_closed() -> None:
    result = validate_bundle(ROOT)
    assert result.valid
    assert not result.neural_execution_allowed
    assert "legacy experiment release is not evidenced" in result.neural_execution_blockers
    assert "smoke forecast is not approved" in result.neural_execution_blockers
    assert not result.confirmatory_access_allowed
    assert "r0-1 policy is seal-only" in result.confirmatory_access_blockers


def test_protocol_freezes_zero_shot_public_only_scope() -> None:
    protocol = _document("configs/protocol/r0_protocol.json")
    assert not validate_document(protocol, "r0_protocol")
    assert protocol["scope"]["training_authorized"] is False
    assert protocol["scope"]["public_data_only"] is True
    assert protocol["representation_grid"]["pooling_ms"] == [100, 200, 300, 500, 750, 1000]
    assert protocol["representation_grid"]["continuous_hop_ms"] == {
        "primary": 50,
        "sensitivity": [20, 100],
    }
    assert protocol["timeline"]["event_times"] == [
        "boundary_source_sample",
        "observed_source_sample_at_emit",
        "compute_completed_monotonic_ns",
    ]
    assert protocol["event_taxonomy"]["primary"] == [
        "new_speaker_onset_clean",
        "new_speaker_onset_gap",
        "overlap_start_new_speaker",
    ]
    assert protocol["speech_activity_conditions"][0] == "common_causal_vad"


def test_unknown_protocol_field_is_rejected_even_when_rehashed() -> None:
    protocol = _document("configs/protocol/r0_protocol.json")
    errors = _validated_mutation(protocol, lambda value: value.__setitem__("extra", True))
    assert any("unknown keys ['extra']" in error for error in errors)


def test_training_authorization_mutation_is_rejected() -> None:
    protocol = _document("configs/protocol/r0_protocol.json")
    errors = _validated_mutation(
        protocol,
        lambda value: value["scope"].__setitem__("training_authorized", True),
    )
    assert any("training_authorized" in error for error in errors)


def test_integer_false_event_frontier_and_reference_are_frozen() -> None:
    analysis = _document("configs/protocol/analysis_contract.json")
    continuous = analysis["primary_endpoints"]["continuous_zero_shot"]
    assert continuous["selection"] == "integer_false_event_pareto_frontier"
    assert continuous["reference_false_events_per_hour"] == 1
    errors = _validated_mutation(
        analysis,
        lambda value: value["primary_endpoints"]["continuous_zero_shot"].__setitem__(
            "selection", "best_test_f1"
        ),
    )
    assert any("integer frontier" in error for error in errors)


def test_compute_parallelism_and_smoke_gate_mutations_are_rejected() -> None:
    compute = _document("configs/protocol/compute_ceiling.json")
    parallel_errors = _validated_mutation(
        compute,
        lambda value: value["ceilings"].__setitem__("max_parallel_models", 2),
    )
    smoke_errors = _validated_mutation(
        compute,
        lambda value: value["smoke_gate"].__setitem__("forecast_required", False),
    )
    assert any("sequential" in error for error in parallel_errors)
    assert any("fail closed" in error for error in smoke_errors)


def test_rehashed_future_audio_and_extreme_ceiling_mutations_are_rejected() -> None:
    protocol = _document("configs/protocol/r0_protocol.json")
    compute = _document("configs/protocol/compute_ceiling.json")
    future_errors = _validated_mutation(
        protocol,
        lambda value: value["context_modes"][0].__setitem__("future_audio_allowed", True),
    )
    ceiling_errors = _validated_mutation(
        compute,
        lambda value: value["ceilings"].__setitem__("max_total_wall_hours", 1000000),
    )
    assert any("frozen R0 contract" in error for error in future_errors)
    assert any("frozen R0 contract" in error for error in ceiling_errors)


def test_restricted_or_unknown_license_cannot_enable_product_claim() -> None:
    licenses = _document("configs/protocol/license_disposition.json")
    errors = _validated_mutation(
        licenses,
        lambda value: value["models"][0].__setitem__("product_claim_allowed", True),
    )
    assert any("product_claim_allowed" in error for error in errors)


def test_rehashed_product_allowed_and_registry_license_mutations_are_rejected() -> None:
    licenses = _document("configs/protocol/license_disposition.json")
    registry = _document("models/registry.json")

    def grant_product(value: dict) -> None:
        value["models"][0]["product_status"] = "product_allowed"
        value["models"][0]["product_claim_allowed"] = True

    product_errors = _validated_mutation(licenses, grant_product)
    registry_errors = _validated_mutation(
        registry,
        lambda value: value["models"][0].__setitem__("license_id", "Apache-2.0"),
    )
    assert any("frozen R0 contract" in error for error in product_errors)
    assert any("frozen R0 contract" in error for error in registry_errors)


def test_model_revision_and_artifact_identity_are_exact() -> None:
    registry = _document("models/registry.json")
    revision_errors = _validated_mutation(
        registry,
        lambda value: value["models"][0].__setitem__("revision", "1" * 40),
    )
    artifact_errors = _validated_mutation(
        registry,
        lambda value: value["models"][1]["artifact"].__setitem__("sha256", "2" * 64),
    )
    assert any("frozen revision changed" in error for error in revision_errors)
    assert any("frozen artifact identity changed" in error for error in artifact_errors)
