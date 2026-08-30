from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from puripuly_heart.app.services.canonical_settings_persistence import compose_settings_owner
from puripuly_heart.config.settings_vnext import serialization
from puripuly_heart.config.settings_vnext.facade import save_vnext_settings
from puripuly_heart.config.settings_vnext.schema import (
    AppSettingsVNext,
    TranslationFallbackIntent,
)


def _write_json(path: Path, value: dict[str, object]) -> bytes:
    content = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")
    path.write_bytes(content)
    return content


def _with_locale(settings: AppSettingsVNext, locale: str) -> AppSettingsVNext:
    return replace(
        settings,
        intent=replace(settings.intent, ui=replace(settings.intent.ui, locale=locale)),
    )


def test_owner_persists_default_gemma_fallback_selected_from_disabled_state(
    tmp_path: Path,
) -> None:
    path = tmp_path / "settings.json"
    canonical = AppSettingsVNext()
    canonical = replace(
        canonical,
        intent=replace(
            canonical.intent,
            translation=replace(
                canonical.intent.translation,
                fallback=TranslationFallbackIntent(selection_alias="none"),
            ),
        ),
    )
    _write_json(path, serialization.to_dict(canonical))
    owner = compose_settings_owner(path)
    loaded = owner.start()
    changed = replace(
        loaded.settings,
        intent=replace(
            loaded.settings.intent,
            translation=replace(
                loaded.settings.intent.translation,
                fallback=TranslationFallbackIntent(
                    selection_alias="openrouter_gemma4_26b_31b",
                ),
            ),
        ),
    )

    owner.apply_canonical_delta(loaded.settings, changed)
    owner.persist()

    reloaded = compose_settings_owner(path).start().settings
    assert reloaded.intent.translation.fallback == changed.intent.translation.fallback


def test_owner_unrelated_delta_preserves_disabled_fallback(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    raw = serialization.to_dict(AppSettingsVNext())
    raw["intent"]["translation"]["fallback"] = {
        "enabled": False,
        "model": "deepseek_v4_flash",
        "connection": "official_byok",
        "selection_alias": "none",
    }
    _write_json(path, raw)
    owner = compose_settings_owner(path)
    loaded = owner.start()
    changed = _with_locale(loaded.settings, "ja")

    owner.apply_canonical_delta(loaded.settings, changed)
    owner.persist()

    reloaded = compose_settings_owner(path).start().settings
    assert reloaded.intent.translation.fallback.enabled is False


def test_owner_normalizes_missing_legacy_policies_to_fixed_defaults(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    raw = serialization.to_dict(AppSettingsVNext())
    raw["intent"]["stt"].pop("low_latency_mode")
    raw["intent"]["integrated_context"].pop("enabled")
    raw["state"]["integrated_context"]["bootstrapped"] = False
    _write_json(path, raw)

    result = compose_settings_owner(path).start()

    assert result.settings.intent.stt.low_latency_mode is True
    assert result.settings.intent.integrated_context.enabled is True
    assert result.settings.state.integrated_context.bootstrapped is False


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
    assert result.settings.intent.stt.low_latency_mode is True
    assert result.settings.intent.integrated_context.enabled is True


def test_owner_does_not_rewrite_already_canonical_fixed_policies(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    assert save_vnext_settings(path, AppSettingsVNext()).ok
    original = path.read_bytes()

    result = compose_settings_owner(path).start()

    assert result.migrated is False
    assert result.backup_path is None
    assert path.read_bytes() == original
    assert list(tmp_path.glob("*.bak")) == []


def test_owner_failed_save_keeps_last_persisted_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "settings.json"
    _write_json(path, serialization.to_dict(AppSettingsVNext()))
    owner = compose_settings_owner(path)
    loaded = owner.start()
    original = path.read_bytes()
    changed = _with_locale(loaded.settings, "ko")
    owner.begin()
    owner.apply_canonical_delta(loaded.settings, changed)

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
    before = copy.deepcopy(loaded.settings)
    canonical_before = copy.deepcopy(owner.canonical)
    owner.authoritative = True

    owner.begin()
    changed = _with_locale(before, "ja")
    owner.apply_canonical_delta(before, changed)
    owner.begin()
    owner.complete()

    assert owner.rollback_pending is True
    assert owner.mutation_depth == 1
    assert owner.canonical is not None
    assert owner.canonical.intent.ui.locale == "ja"
    assert owner.canonical != canonical_before

    owner.rollback()

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
    candidate = replace(
        loaded.settings,
        intent=replace(
            loaded.settings.intent,
            translation=replace(loaded.settings.intent.translation, concurrency_limit=3),
            stt=replace(loaded.settings.intent.stt, low_latency_mode=False),
        ),
    )

    projected = owner.project_canonical_delta(loaded.settings, candidate)

    assert owner.canonical == original_canonical
    assert candidate.intent.stt.low_latency_mode is False
    assert projected.intent.translation.concurrency_limit == 3
    assert projected.intent.stt.low_latency_mode is True


def test_owner_persist_round_trips_unknown_compatibility_extensions(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    raw = serialization.to_dict(AppSettingsVNext())
    raw["future_product_flag"] = True
    raw["intent"]["ui"]["future_toggle"] = "keep-me"
    _write_json(path, raw)
    owner = compose_settings_owner(path)
    loaded = owner.start()
    changed = _with_locale(loaded.settings, "ja")

    owner.begin()
    owner.apply_canonical_delta(loaded.settings, changed)
    owner.persist()
    owner.complete()

    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded["future_product_flag"] is True
    assert reloaded["intent"]["ui"]["future_toggle"] == "keep-me"
    assert reloaded["intent"]["ui"]["locale"] == "ja"
    restarted = compose_settings_owner(path).start()
    persisted = serialization.to_dict(restarted.settings)
    assert persisted["future_product_flag"] is True
    assert persisted["intent"]["ui"]["future_toggle"] == "keep-me"
    assert persisted["intent"]["ui"]["locale"] == "ja"
