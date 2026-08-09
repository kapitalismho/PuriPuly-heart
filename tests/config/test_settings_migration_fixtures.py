from __future__ import annotations

import copy
import json
from dataclasses import fields, is_dataclass
from pathlib import Path

import pytest

from puripuly_heart.config.settings import (
    SETTINGS_SCHEMA_VERSION,
    AppSettings,
    QwenRegion,
    _migrate_settings_dict,
    from_dict,
    to_dict,
)
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.ui.overlay_calibration import OVERLAY_CALIBRATION_ANCHORS, OverlayCalibration
from tests.config import settings_migration_fixtures as fixtures
from tests.config.settings_migration_fixtures import (
    ADR_RESOLVED_CURRENT_DESTINATIONS,
    EXPLICIT_MISSING_FIELD_DEFAULT_EXPECTATIONS,
    LEGACY_MIGRATION_CLASSIFICATION,
    V24_MIGRATION_CLASSIFICATION,
    legacy_compatibility_settings_fixture,
    maximal_v24_settings_fixture,
    migrated_serialization,
    missing_classification_paths,
    missing_field_default_expectations,
    path_get,
    path_remove,
    serialized_field_paths,
)

SCHEMA_OR_SINGLETON_SAME_AS_DEFAULT_PATHS = frozenset(
    {
        "cerebras.llm_model",
        "openrouter.routing_mode",
        "overlay.calibration.anchor",
        "settings_version",
    }
)


def _dataclass_leaf_paths(value: object, prefix: str = "") -> set[str]:
    if not is_dataclass(value) or isinstance(value, type):
        return {prefix} if prefix else set()

    paths: set[str] = set()
    for field in fields(value):
        child = getattr(value, field.name)
        child_path = f"{prefix}.{field.name}" if prefix else field.name
        if is_dataclass(child) and not isinstance(child, type):
            paths.update(_dataclass_leaf_paths(child, child_path))
        else:
            paths.add(child_path)
    return paths


def _claims_vnext_schema_destination(destination: str, status: str) -> bool:
    if "no_vnext_write_projection" in status:
        return False
    return destination == "settings_version" or destination.startswith(("intent.", "state."))


def test_v24_migration_classification_covers_current_serialized_settings_paths() -> None:
    current_paths = set(serialized_field_paths(to_dict(AppSettings())))

    missing_paths = missing_classification_paths(current_paths, V24_MIGRATION_CLASSIFICATION)
    extra_paths = sorted(set(V24_MIGRATION_CLASSIFICATION).difference(current_paths))

    assert missing_paths == []
    assert extra_paths == []


def test_final_dev_v30_fixture_has_an_explicit_destination_for_every_persisted_path() -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "final_dev_v30_settings.json"
    raw = json.loads(fixture_path.read_text(encoding="utf-8"))
    final_dev_paths = set(serialized_field_paths(raw))
    v30_destinations = {
        "managed_identity.pending_delivery_ack_id": (
            "state.managed_connection.pending_delivery_ack_delivery_id"
        ),
        "telemetry.identifier": "state.telemetry.anonymous_id",
        "telemetry.sent_utc_dates": "state.telemetry.sent_translation_success_dates_utc",
        "translation.fallback_selection_alias": "intent.translation.fallback.selection_alias",
    }
    classified_paths = set(V24_MIGRATION_CLASSIFICATION) | set(v30_destinations)

    assert sorted(final_dev_paths - classified_paths) == []
    assert raw["settings_version"] == 30


def test_migration_classification_guard_reports_removed_entries() -> None:
    current_paths = set(serialized_field_paths(to_dict(AppSettings())))
    incomplete = dict(V24_MIGRATION_CLASSIFICATION)
    incomplete.pop("provider.stt")

    missing_paths = missing_classification_paths(current_paths, incomplete)

    assert missing_paths == ["provider.stt"]


def test_current_classification_does_not_keep_dead_path_set_constants() -> None:
    assert not {
        "CURRENT_COMPATIBILITY_INPUT_PATHS",
        "PERSISTED_OPERATIONAL_STATE_CURRENT_PATHS",
        "REPAIR_TO_CANONICAL_DEFAULT_CURRENT_PATHS",
        "SCHEMA_METADATA_CURRENT_PATHS",
        "SINGLETON_SUPPORTED_VALUE_CURRENT_PATHS",
        "USER_INTENT_CURRENT_PATHS",
    }.intersection(vars(fixtures))


def test_current_classification_vnext_destinations_resolve_to_schema_leaves() -> None:
    vnext_leaf_paths = _dataclass_leaf_paths(AppSettingsVNext())

    unresolved = {
        path: classification.destination
        for path, classification in sorted(V24_MIGRATION_CLASSIFICATION.items())
        if _claims_vnext_schema_destination(classification.destination, classification.status)
        and classification.destination not in vnext_leaf_paths
    }

    assert unresolved == {}


def test_vnext_schema_persisted_leaves_are_classified_from_current_settings() -> None:
    vnext_leaf_paths = {
        path
        for path in _dataclass_leaf_paths(AppSettingsVNext())
        if path == "settings_version" or path.startswith(("intent.", "state."))
    }
    classified_vnext_destinations = {
        classification.destination
        for classification in V24_MIGRATION_CLASSIFICATION.values()
        if _claims_vnext_schema_destination(classification.destination, classification.status)
    }

    assert (
        sorted(
            vnext_leaf_paths
            - classified_vnext_destinations
            - fixtures.VNEXT_NATIVE_PERSISTED_LEAF_PATHS
        )
        == []
    )


def test_vnext_native_provider_verification_evidence_leaves_are_explicitly_excluded() -> None:
    providers = {
        "alibaba_beijing",
        "alibaba_singapore",
        "cerebras",
        "deepgram",
        "deepseek",
        "google",
        "openrouter",
        "soniox",
    }
    fields = {
        "provider",
        "secret_key",
        "secret_revision",
        "secret_fingerprint",
        "verifier_context",
        "verifier_evidence",
    }
    expected = frozenset(
        f"state.provider_verification.{provider}.{field}"
        for provider in providers
        for field in fields
    ) | frozenset(
        {
            "intent.desktop_audio.capture_target.kind",
            "intent.desktop_audio.capture_target.device_name",
            "intent.desktop_audio.capture_target.process",
            "intent.translation.http_extension_id",
            "intent.translation.previous_llm_model",
        }
    )
    classified_vnext_destinations = {
        classification.destination
        for classification in V24_MIGRATION_CLASSIFICATION.values()
        if _claims_vnext_schema_destination(classification.destination, classification.status)
    }

    assert fixtures.VNEXT_NATIVE_PERSISTED_LEAF_PATHS == expected
    assert fixtures.VNEXT_NATIVE_PERSISTED_LEAF_PATHS.isdisjoint(classified_vnext_destinations)


def test_serialized_field_paths_include_empty_non_dynamic_dicts() -> None:
    data = {
        "present": {},
        "nested": {"empty": {}},
        "stt": {"custom_terms": {}},
    }

    assert set(serialized_field_paths(data)) == {
        "present",
        "nested.empty",
        "stt.custom_terms",
    }


def test_maximal_v24_fixture_covers_current_paths_with_safe_non_default_values() -> None:
    default_data = to_dict(AppSettings())
    maximal_data = maximal_v24_settings_fixture()
    current_paths = set(serialized_field_paths(default_data))
    maximal_paths = set(serialized_field_paths(maximal_data))

    assert maximal_data["settings_version"] == SETTINGS_SCHEMA_VERSION
    assert maximal_paths == current_paths

    default_matches = sorted(
        path
        for path in current_paths
        if type(path_get(maximal_data, path)) is type(path_get(default_data, path))
        and path_get(maximal_data, path) == path_get(default_data, path)
    )
    assert default_matches == sorted(SCHEMA_OR_SINGLETON_SAME_AS_DEFAULT_PATHS)
    assert V24_MIGRATION_CLASSIFICATION["settings_version"].category == "schema_metadata"
    assert (
        V24_MIGRATION_CLASSIFICATION["overlay.calibration.anchor"].status
        == "default_supported_value"
    )


def test_repairable_non_default_maximal_v24_values_normalize_to_canonical_values() -> None:
    maximal_data = maximal_v24_settings_fixture()

    migrated, _changed = _migrate_settings_dict(copy.deepcopy(maximal_data))
    loaded_data = to_dict(from_dict(migrated))

    assert maximal_data["audio"]["internal_sample_rate_hz"] == 8000
    assert loaded_data["audio"]["internal_sample_rate_hz"] == 16000
    assert maximal_data["audio"]["internal_channels"] == "1"
    assert loaded_data["audio"]["internal_channels"] == 1
    assert maximal_data["local_llm"]["backend"] == "fixture_backend"
    assert loaded_data["local_llm"]["backend"] == "ollama"


def test_overlay_anchor_classification_matches_current_validation() -> None:
    assert OVERLAY_CALIBRATION_ANCHORS == ("head_locked", "spatial_locked")
    assert path_get(maximal_v24_settings_fixture(), "overlay.calibration.anchor") == "head_locked"
    assert (
        V24_MIGRATION_CLASSIFICATION["overlay.calibration.anchor"].notes
        == "Current overlay calibration supports 'head_locked' and 'spatial_locked'; other values fail validation."
    )

    with pytest.raises(ValueError, match="unsupported overlay calibration anchor"):
        OverlayCalibration(anchor="fixture_anchor").validate()


def test_missing_fields_load_and_serialize_for_every_current_path() -> None:
    current_paths = set(serialized_field_paths(to_dict(AppSettings())))
    expectations = missing_field_default_expectations()

    missing_paths = missing_classification_paths(current_paths, expectations)
    extra_paths = sorted(set(expectations).difference(current_paths))

    assert missing_paths == []
    assert extra_paths == []
    for path, expected in sorted(expectations.items()):
        raw = to_dict(AppSettings())
        path_remove(raw, path)

        loaded_data = to_dict(from_dict(raw))

        assert V24_MIGRATION_CLASSIFICATION[path].fixture == "maximal_v24_settings"
        assert (
            V24_MIGRATION_CLASSIFICATION[path].missing_default_fixture == "missing_field_defaults"
        )
        loaded_value = path_get(loaded_data, path)
        if path == "telemetry_state.anonymous_id":
            assert isinstance(loaded_value, str)
            assert len(loaded_value) == 32
        else:
            assert loaded_value == expected


def test_non_obvious_missing_fields_load_and_serialize_documented_defaults() -> None:
    for path, expected in sorted(EXPLICIT_MISSING_FIELD_DEFAULT_EXPECTATIONS.items()):
        raw = to_dict(AppSettings())
        path_remove(raw, path)

        loaded_data = to_dict(from_dict(raw))

        assert path_get(loaded_data, path) == expected


def test_maximal_v24_fixture_round_trip_retains_explicit_stable_fields() -> None:
    maximal_data = maximal_v24_settings_fixture()
    round_tripped_data = migrated_serialization(maximal_data)

    expected_retained_paths = {
        "api_key_verified.deepgram",
        "audio.input_device",
        "desktop_audio.output_device",
        "local_llm.base_url",
        "managed_identity.installation_id",
        "languages.peer_expected_languages",
        "languages.peer_source_mode",
        "openrouter.broker_base_url",
        "provider.peer_stt",
        "provider.stt",
        "qwen.region",
        "secrets.backend",
        "soniox_stt.endpoint",
        "stt.drain_timeout_s",
        "system_prompt",
        "translation.connection",
        "translation.model",
        "ui.clipboard_auto_translate_enabled",
        "ui.github_star_prompt_clicked",
        "ui.integrated_context_enabled",
        "ui.locale",
    }

    assert {path: path_get(round_tripped_data, path) for path in expected_retained_paths} == {
        path: path_get(maximal_data, path) for path in expected_retained_paths
    }


def test_adr_resolved_current_paths_have_exact_vnext_destinations() -> None:
    assert set(ADR_RESOLVED_CURRENT_DESTINATIONS) == {
        "ui.integrated_context_bootstrapped",
        "ui.integrated_context_enabled",
        "ui.peer_translation_eula_accepted",
    }
    assert {
        path: V24_MIGRATION_CLASSIFICATION[path].destination
        for path in ADR_RESOLVED_CURRENT_DESTINATIONS
    } == ADR_RESOLVED_CURRENT_DESTINATIONS
    assert {
        path: V24_MIGRATION_CLASSIFICATION[path].category
        for path in ADR_RESOLVED_CURRENT_DESTINATIONS
    } == {
        "ui.peer_translation_eula_accepted": "persisted_operational_state",
        "ui.integrated_context_enabled": "persisted_user_intent",
        "ui.integrated_context_bootstrapped": "persisted_operational_state",
    }
    assert {
        V24_MIGRATION_CLASSIFICATION[path].status for path in ADR_RESOLVED_CURRENT_DESTINATIONS
    } == {"retained_adr_resolved"}
    assert "decision_pending" not in {
        classification.category for classification in V24_MIGRATION_CLASSIFICATION.values()
    }


def test_legacy_migration_classification_covers_all_legacy_fixture_extra_paths() -> None:
    current_paths = set(serialized_field_paths(to_dict(AppSettings())))
    legacy_paths = set(serialized_field_paths(legacy_compatibility_settings_fixture()))
    introduced_paths = legacy_paths.difference(current_paths)

    missing_paths = missing_classification_paths(introduced_paths, LEGACY_MIGRATION_CLASSIFICATION)
    extra_paths = sorted(set(LEGACY_MIGRATION_CLASSIFICATION).difference(introduced_paths))

    assert missing_paths == []
    assert extra_paths == []


def test_legacy_migration_classification_guard_reports_removed_entries() -> None:
    current_paths = set(serialized_field_paths(to_dict(AppSettings())))
    legacy_paths = set(serialized_field_paths(legacy_compatibility_settings_fixture()))
    introduced_paths = legacy_paths.difference(current_paths)
    incomplete = dict(LEGACY_MIGRATION_CLASSIFICATION)
    incomplete.pop("osc.cooldown_s")

    missing_paths = missing_classification_paths(introduced_paths, incomplete)

    assert missing_paths == ["osc.cooldown_s"]


@pytest.mark.parametrize(
    ("legacy_key", "legacy_value", "expected_selected_source"),
    [
        ("credential_source", "byok", "byok"),
        ("selected_credential_source", "managed", "managed"),
    ],
)
def test_legacy_openrouter_source_keys_drive_migration_when_current_source_is_absent(
    legacy_key: str,
    legacy_value: str,
    expected_selected_source: str,
) -> None:
    raw = to_dict(AppSettings())
    raw["settings_version"] = 9
    raw["provider"]["llm"] = "openrouter"
    raw.pop("translation", None)
    raw["openrouter"].pop("selected_source", None)
    raw["openrouter"].pop("selection_alias", None)
    raw["openrouter"][legacy_key] = legacy_value

    migrated, changed = _migrate_settings_dict(raw)

    assert changed is True
    assert migrated["openrouter"]["selected_source"] == expected_selected_source
    assert legacy_key not in migrated["openrouter"]
    assert (
        LEGACY_MIGRATION_CLASSIFICATION[f"openrouter.{legacy_key}"].status
        == "accepted_read_drives_migration_when_current_source_absent"
    )


def test_legacy_fixture_covers_compatibility_runtime_only_and_retired_inputs() -> None:
    raw = legacy_compatibility_settings_fixture()
    migrated, changed = _migrate_settings_dict(copy.deepcopy(raw))
    settings = from_dict(migrated)
    persisted = to_dict(settings)

    assert changed is True
    assert migrated["settings_version"] == SETTINGS_SCHEMA_VERSION
    assert migrated["overlay"]["calibration"]["offset_x"] == 0.42
    assert migrated["overlay"]["show_translation"] is False
    assert migrated["overlay"]["show_peer_original"] is False
    assert settings.ui.overlay_enabled is False
    assert settings.ui.peer_translation_enabled is False
    assert settings.peer_qwen_asr_stt.model == "legacy-peer-qwen-model"
    assert settings.peer_qwen_asr_stt.region == QwenRegion.SINGAPORE
    assert settings.peer_soniox_stt.endpoint == "wss://legacy-soniox.fixture.test/transcribe"

    assert "credential_source" not in migrated["openrouter"]
    assert "selected_credential_source" not in migrated["openrouter"]
    assert "cooldown_s" not in migrated["osc"]
    assert "ttl_s" not in migrated["osc"]
    assert "locked" not in migrated["overlay"]["desktop_flet"]
    assert "overlay_calibration" not in migrated
    assert "peer_deepgram_stt" not in migrated
    assert "system_prompts" not in migrated
    assert "overlay_enabled" not in migrated["ui"]
    assert "peer_translation_enabled" not in migrated["ui"]
    assert "show_overlay_peer_original" not in migrated["ui"]
    assert "show_overlay_translation" not in migrated["ui"]
    assert "peer_qwen_asr_stt" not in persisted
    assert "peer_soniox_stt" not in persisted

    assert {
        LEGACY_MIGRATION_CLASSIFICATION[path].fixture
        for path in (
            "openrouter.credential_source",
            "osc.cooldown_s",
            "overlay.desktop_flet.locked",
            "peer_qwen_asr_stt.model",
            "system_prompts.legacy",
            "ui.overlay_enabled",
        )
    } == {"legacy_compatibility_settings"}
