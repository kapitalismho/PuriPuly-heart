from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from puripuly_heart.app.services.canonical_settings_persistence import compose_settings_owner
from puripuly_heart.config.settings import (
    AppSettings,
    SecretsBackend,
    STTProviderName,
)
from puripuly_heart.config.settings import (
    to_dict as legacy_to_dict,
)
from puripuly_heart.config.settings_vnext import serialization
from puripuly_heart.config.settings_vnext.facade import save_vnext_settings
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext


def _write_json(path: Path, value: dict[str, object]) -> bytes:
    content = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")
    path.write_bytes(content)
    return content


def test_owner_normalizes_explicit_false_legacy_policies_after_exact_backup(
    tmp_path: Path,
) -> None:
    path = tmp_path / "settings.json"
    raw = legacy_to_dict(AppSettings())
    raw["stt"]["low_latency_mode"] = False
    raw["ui"]["integrated_context_enabled"] = False
    raw["ui"]["integrated_context_bootstrapped"] = True
    original = _write_json(path, raw)

    result = compose_settings_owner(path).start()

    assert result.migrated is True
    assert result.backup_path is not None
    assert result.backup_path.read_bytes() == original
    assert result.settings.stt.low_latency_mode is True
    assert result.settings.ui.integrated_context_enabled is True
    assert result.settings.ui.integrated_context_bootstrapped is True
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["intent"]["stt"]["low_latency_mode"] is True
    assert persisted["intent"]["integrated_context"]["enabled"] is True
    assert persisted["state"]["integrated_context"]["bootstrapped"] is True


def test_owner_normalizes_missing_legacy_policies_to_fixed_defaults(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    raw = legacy_to_dict(AppSettings())
    raw["stt"].pop("low_latency_mode")
    raw["ui"].pop("integrated_context_enabled")
    raw["ui"]["integrated_context_bootstrapped"] = False
    _write_json(path, raw)

    result = compose_settings_owner(path).start()

    assert result.settings.stt.low_latency_mode is True
    assert result.settings.ui.integrated_context_enabled is True
    assert result.settings.ui.integrated_context_bootstrapped is False


def test_owner_backs_up_canonical_false_policies_before_normalizing(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    raw = serialization.to_dict(AppSettingsVNext())
    raw["intent"]["stt"]["low_latency_mode"] = False
    raw["intent"]["integrated_context"]["enabled"] = False
    original = _write_json(path, raw)

    result = compose_settings_owner(path).start()

    assert result.migrated is True
    assert result.backup_path is not None
    assert result.backup_path.read_bytes() == original
    assert result.settings.stt.low_latency_mode is True
    assert result.settings.ui.integrated_context_enabled is True


def test_owner_does_not_rewrite_already_canonical_fixed_policies(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    assert save_vnext_settings(path, AppSettingsVNext()).ok
    original = path.read_bytes()

    result = compose_settings_owner(path).start()

    assert result.migrated is False
    assert result.backup_path is None
    assert path.read_bytes() == original
    assert list(tmp_path.glob("*.bak")) == []


def test_owner_roundtrip_preserves_established_choices_and_safe_extensions(
    tmp_path: Path,
) -> None:
    path = tmp_path / "settings.json"
    settings = AppSettings()
    settings.provider.stt = STTProviderName.LOCAL_PARAKEET_V3
    settings.provider.peer_stt = STTProviderName.LOCAL_CPU_AUTO
    settings.languages.peer_source_mode = "auto"
    settings.languages.peer_expected_languages = ["ja", "zh-TW"]
    settings.stt.low_latency_vad_hangover_ms = 777
    settings.stt.low_latency_merge_gap_ms = 888
    settings.stt.low_latency_spec_retry_max = 9
    settings.managed_identity.installation_id = "installation-owner-roundtrip"
    settings.secrets.backend = SecretsBackend.ENCRYPTED_FILE
    settings.secrets.encrypted_file_path = "owner-secrets.json"
    raw = legacy_to_dict(settings)
    raw["legacy_extension"] = {"theme_hint": "violet", "version": 2}
    raw["discarded_api_key"] = "must-not-persist"
    _write_json(path, raw)

    first_owner = compose_settings_owner(path)
    first = first_owner.start()
    changed = copy.deepcopy(first.settings)
    changed.ui.locale = "ja"
    changed.stt.low_latency_mode = False
    changed.ui.integrated_context_enabled = False
    first_owner.apply_legacy_delta(first.settings, changed)
    first_owner.persist()
    second = compose_settings_owner(path).start()

    assert second.settings.provider.stt == STTProviderName.LOCAL_PARAKEET_V3
    assert second.settings.provider.peer_stt == STTProviderName.LOCAL_CPU_AUTO
    assert second.settings.languages.peer_source_mode == "auto"
    assert second.settings.languages.peer_expected_languages == ["ja", "zh-TW"]
    assert second.settings.stt.low_latency_vad_hangover_ms == 777
    assert second.settings.stt.low_latency_merge_gap_ms == 888
    assert second.settings.stt.low_latency_spec_retry_max == 9
    assert second.settings.managed_identity.installation_id == "installation-owner-roundtrip"
    assert second.settings.secrets.backend == SecretsBackend.ENCRYPTED_FILE
    assert second.settings.secrets.encrypted_file_path == "owner-secrets.json"
    assert second.settings.stt.low_latency_mode is True
    assert second.settings.ui.integrated_context_enabled is True
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["legacy_compatibility"] == {
        "legacy_extension": {"theme_hint": "violet", "version": 2}
    }
    assert "discarded_api_key" not in persisted


def test_owner_failed_save_keeps_last_persisted_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "settings.json"
    _write_json(path, serialization.to_dict(AppSettingsVNext()))
    owner = compose_settings_owner(path)
    loaded = owner.start()
    original = path.read_bytes()
    changed = copy.deepcopy(loaded.settings)
    changed.ui.locale = "ko"
    owner.begin()
    owner.apply_legacy_delta(loaded.settings, changed)

    def fail_persist(_path: Path, _settings: AppSettingsVNext) -> None:
        raise OSError("injected owner save failure")

    monkeypatch.setattr(owner.persistence, "persist", fail_persist)

    with pytest.raises(OSError, match="injected owner save failure"):
        owner.persist()
    owner.rollback()

    assert path.read_bytes() == original
    assert owner.canonical is not None
    assert owner.canonical.intent.ui.locale == "en"


def test_owner_nested_completion_keeps_outer_rollback_snapshot(
    tmp_path: Path,
) -> None:
    path = tmp_path / "settings.json"
    _write_json(path, serialization.to_dict(AppSettingsVNext()))
    owner = compose_settings_owner(path)
    loaded = owner.start()
    legacy_before = copy.deepcopy(loaded.settings)
    canonical_before = copy.deepcopy(owner.canonical)
    owner.authoritative = True

    owner.begin(legacy_snapshot=legacy_before)
    assert owner.current is not None
    owner.current.ui.locale = "ja"
    owner.apply_legacy_delta(legacy_before, owner.current)
    owner.begin()
    owner.complete()

    assert owner.rollback_pending is True
    assert owner.mutation_depth == 1
    assert owner.current.ui.locale == "ja"
    assert owner.canonical != canonical_before

    owner.rollback()

    assert owner.current == legacy_before
    assert owner.canonical == canonical_before
    assert owner.authoritative is True
    assert owner.rollback_pending is False
    assert owner.mutation_depth == 0


def test_owner_legacy_delta_projection_is_side_effect_free(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    _write_json(path, serialization.to_dict(AppSettingsVNext()))
    owner = compose_settings_owner(path)
    loaded = owner.start()
    original_canonical = copy.deepcopy(owner.canonical)
    candidate = copy.deepcopy(loaded.settings)
    candidate.llm.concurrency_limit = 3
    candidate.stt.low_latency_mode = False

    projected = owner.project_legacy_delta(loaded.settings, candidate)

    assert owner.canonical == original_canonical
    assert candidate.stt.low_latency_mode is False
    assert projected.intent.translation.concurrency_limit == 3
    assert projected.intent.stt.low_latency_mode is True
