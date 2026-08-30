from __future__ import annotations

from dataclasses import replace

from puripuly_heart.config.settings_vnext import serialization
from puripuly_heart.config.settings_vnext.migration import from_dict as from_vnext_dict
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext


def _with_osc(settings: AppSettingsVNext, **changes: object) -> AppSettingsVNext:
    return replace(
        settings,
        intent=replace(
            settings.intent,
            osc=replace(settings.intent.osc, **changes),
        ),
    )


def test_osc_settings_round_trip_mode_and_ports() -> None:
    settings = _with_osc(
        AppSettingsVNext(),
        connection_mode="manual",
        send_port=9100,
        receive_port=9101,
    )

    persisted = serialization.to_dict(settings)
    loaded = serialization.from_dict(persisted)

    assert persisted["intent"]["osc"]["connection_mode"] == "manual"
    assert persisted["intent"]["osc"]["send_port"] == 9100
    assert persisted["intent"]["osc"]["receive_port"] == 9101
    assert loaded.intent.osc.connection_mode == "manual"
    assert loaded.intent.osc.send_port == 9100
    assert loaded.intent.osc.receive_port == 9101


def test_automatic_osc_settings_round_trip_keeps_mode_and_ports() -> None:
    settings = _with_osc(
        AppSettingsVNext(),
        connection_mode="automatic",
        send_port=9120,
        receive_port=9121,
    )

    persisted = serialization.to_dict(settings)
    loaded = serialization.from_dict(persisted)

    assert persisted["intent"]["osc"]["connection_mode"] == "automatic"
    assert persisted["intent"]["osc"]["send_port"] == 9120
    assert persisted["intent"]["osc"]["receive_port"] == 9121
    assert loaded.intent.osc.connection_mode == "automatic"
    assert (loaded.intent.osc.send_port, loaded.intent.osc.receive_port) == (9120, 9121)


def test_vnext_osc_mode_and_ports_round_trip() -> None:
    settings = _with_osc(
        AppSettingsVNext(),
        connection_mode="off",
        send_port=9200,
        receive_port=9201,
    )
    persisted = serialization.to_dict(settings)
    loaded = serialization.from_dict(persisted)

    assert loaded.intent.osc.connection_mode == "off"
    assert loaded.intent.osc.send_port == 9200
    assert loaded.intent.osc.receive_port == 9201
    assert persisted["intent"]["osc"]["connection_mode"] == "off"
    assert persisted["intent"]["osc"]["send_port"] == 9200
    assert persisted["intent"]["osc"]["receive_port"] == 9201


def test_pre_feature_vnext_osc_records_become_automatic_and_round_trip_idempotently() -> None:
    settings = _with_osc(
        AppSettingsVNext(),
        connection_mode="manual",
        send_port=9137,
        receive_port=9001,
    )
    persisted = serialization.to_dict(settings)
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
    persisted = serialization.to_dict(
        _with_osc(
            AppSettingsVNext(),
            connection_mode="automatic",
            send_port=9138,
            receive_port=9139,
        )
    )

    loaded = from_vnext_dict(persisted)

    assert loaded.intent.osc.connection_mode == "automatic"
    assert loaded.intent.osc.send_port == 9138
    assert loaded.intent.osc.receive_port == 9139


def test_older_vnext_port_only_osc_settings_become_automatic_and_preserve_ports() -> None:
    persisted = serialization.to_dict(AppSettingsVNext())
    persisted["intent"]["osc"].pop("connection_mode", None)
    persisted["intent"]["osc"].pop("send_port", None)
    persisted["intent"]["osc"]["port"] = 9130

    loaded = from_vnext_dict(persisted)

    assert loaded.intent.osc.connection_mode == "automatic"
    assert loaded.intent.osc.send_port == 9130
    assert loaded.intent.osc.receive_port == 9001


def test_osc_mode_replace_preserves_manual_ports() -> None:
    settings = _with_osc(
        AppSettingsVNext(),
        connection_mode="manual",
        send_port=9140,
        receive_port=9141,
    )

    off = _with_osc(settings, connection_mode="off")
    assert (off.intent.osc.send_port, off.intent.osc.receive_port) == (9140, 9141)
    automatic = _with_osc(off, connection_mode="automatic")
    assert (automatic.intent.osc.send_port, automatic.intent.osc.receive_port) == (9140, 9141)
