from __future__ import annotations

from puripuly_heart.config.settings import AppSettings, from_dict, to_dict
from puripuly_heart.config.settings_vnext import serialization
from puripuly_heart.config.settings_vnext.migration import from_dict as from_vnext_dict
from puripuly_heart.config.settings_vnext.migration import (
    from_legacy_app_settings,
    to_legacy_dict,
)


def test_legacy_osc_settings_round_trip_mode_and_ports() -> None:
    settings = AppSettings()
    settings.osc.connection_mode = "manual"
    settings.osc.send_port = 9100
    settings.osc.receive_port = 9101

    persisted = to_dict(settings)
    loaded = from_dict(persisted)

    assert persisted["osc"]["connection_mode"] == "manual"
    assert persisted["osc"]["send_port"] == 9100
    assert persisted["osc"]["receive_port"] == 9101
    assert loaded.osc.connection_mode == "manual"
    assert loaded.osc.port == 9100
    assert loaded.osc.send_port == 9100
    assert loaded.osc.receive_port == 9101


def test_automatic_osc_settings_round_trip_keeps_mode_and_ports() -> None:
    settings = AppSettings()
    settings.osc.connection_mode = "automatic"
    settings.osc.send_port = 9120
    settings.osc.receive_port = 9121

    persisted = to_dict(settings)
    loaded = from_dict(persisted)

    assert persisted["osc"]["connection_mode"] == "automatic"
    assert persisted["osc"]["send_port"] == 9120
    assert persisted["osc"]["receive_port"] == 9121
    assert loaded.osc.connection_mode == "automatic"
    assert (loaded.osc.send_port, loaded.osc.receive_port) == (9120, 9121)


def test_vnext_migration_preserves_osc_mode_and_ports() -> None:
    settings = AppSettings()
    settings.osc.connection_mode = "off"
    settings.osc.send_port = 9200
    settings.osc.receive_port = 9201

    canonical = from_legacy_app_settings(settings)
    legacy = to_legacy_dict(canonical)

    assert canonical.intent.osc.connection_mode == "off"
    assert canonical.intent.osc.send_port == 9200
    assert canonical.intent.osc.receive_port == 9201
    assert legacy["osc"]["connection_mode"] == "off"
    assert legacy["osc"]["send_port"] == 9200
    assert legacy["osc"]["receive_port"] == 9201


def test_pre_feature_vnext_osc_records_become_automatic_and_round_trip_idempotently() -> None:
    settings = AppSettings()
    settings.osc.connection_mode = "manual"
    settings.osc.port = 9137
    settings.osc.receive_port = 9001
    canonical = from_legacy_app_settings(settings)
    persisted = serialization.to_dict(canonical)
    persisted["intent"]["osc"].pop("connection_mode", None)
    persisted["intent"]["osc"].pop("send_port", None)
    persisted["intent"]["osc"].pop("receive_port", None)
    persisted["intent"]["osc"]["port"] = 9137

    loaded = from_vnext_dict(persisted)
    round_tripped = serialization.to_dict(loaded)

    assert loaded.intent.osc.connection_mode == "automatic"
    assert loaded.intent.osc.send_port == 9137
    assert loaded.intent.osc.receive_port == 9001
    assert round_tripped["intent"]["osc"]["connection_mode"] == "automatic"
    assert round_tripped["intent"]["osc"]["send_port"] == 9137
    assert round_tripped["intent"]["osc"]["receive_port"] == 9001


def test_explicit_vnext_osc_mode_is_preserved_on_reload() -> None:
    settings = AppSettings()
    settings.osc.connection_mode = "automatic"
    settings.osc.send_port = 9138
    settings.osc.receive_port = 9139
    persisted = serialization.to_dict(from_legacy_app_settings(settings))

    loaded = from_vnext_dict(persisted)

    assert loaded.intent.osc.connection_mode == "automatic"
    assert loaded.intent.osc.send_port == 9138
    assert loaded.intent.osc.receive_port == 9139


def test_legacy_port_only_osc_settings_become_automatic_and_preserve_ports() -> None:
    persisted = to_dict(AppSettings())
    persisted["osc"].pop("connection_mode", None)
    persisted["osc"].pop("send_port", None)
    persisted["osc"]["port"] = 9130

    loaded = from_dict(persisted)

    assert loaded.osc.connection_mode == "automatic"
    assert loaded.osc.port == 9130
    assert loaded.osc.send_port == 9130
    assert loaded.osc.receive_port == 9001


def test_osc_mode_switch_preserves_manual_ports_when_controls_are_off() -> None:
    settings = AppSettings()
    settings.osc.connection_mode = "manual"
    settings.osc.send_port = 9140
    settings.osc.receive_port = 9141

    settings.osc.connection_mode = "off"
    assert (settings.osc.send_port, settings.osc.receive_port) == (9140, 9141)
    settings.osc.connection_mode = "automatic"
    assert (settings.osc.send_port, settings.osc.receive_port) == (9140, 9141)
