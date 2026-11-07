from __future__ import annotations

from dataclasses import replace

import pytest

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


@pytest.mark.parametrize("mode", ["manual", "automatic", "off"])
def test_osc_settings_round_trip_mode_and_ports(mode: str) -> None:
    settings = _with_osc(
        AppSettingsVNext(),
        connection_mode=mode,
        send_port=9100,
        receive_port=9101,
    )

    persisted = serialization.to_dict(settings)
    loaded = from_vnext_dict(persisted)

    assert persisted["intent"]["osc"]["connection_mode"] == mode
    assert persisted["intent"]["osc"]["send_port"] == 9100
    assert persisted["intent"]["osc"]["receive_port"] == 9101
    assert loaded.intent.osc.connection_mode == mode
    assert loaded.intent.osc.send_port == 9100
    assert loaded.intent.osc.receive_port == 9101


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


def test_older_vnext_port_only_osc_settings_become_automatic_and_preserve_ports() -> None:
    persisted = serialization.to_dict(AppSettingsVNext())
    persisted["intent"]["osc"].pop("connection_mode", None)
    persisted["intent"]["osc"].pop("send_port", None)
    persisted["intent"]["osc"]["port"] = 9130

    loaded = from_vnext_dict(persisted)

    assert loaded.intent.osc.connection_mode == "automatic"
    assert loaded.intent.osc.send_port == 9130
    assert loaded.intent.osc.receive_port == 9001


@pytest.mark.parametrize("settings_version", [1, 49, 50])
def test_old_osc_settings_default_activation_notice_on(settings_version: int) -> None:
    persisted = serialization.to_dict(AppSettingsVNext())
    persisted["settings_version"] = settings_version
    del persisted["intent"]["osc"]["activation_notice_enabled"]

    loaded = from_vnext_dict(persisted)

    assert loaded.intent.osc.activation_notice_enabled is True
    assert serialization.to_dict(loaded)["intent"]["osc"]["activation_notice_enabled"] is True


def test_old_settings_without_osc_block_default_activation_notice_on() -> None:
    loaded = from_vnext_dict({"settings_version": 50, "intent": {}, "state": {}})

    assert loaded.intent.osc.activation_notice_enabled is True


def test_activation_notice_false_round_trips_without_changing_osc_settings() -> None:
    settings = _with_osc(
        AppSettingsVNext(),
        activation_notice_enabled=False,
        connection_mode="manual",
        send_port=9123,
        receive_port=9124,
        chatbox_include_source=True,
        vrc_mic_intercept=True,
    )

    persisted = serialization.to_dict(settings)
    loaded = from_vnext_dict(persisted)

    assert persisted["intent"]["osc"]["activation_notice_enabled"] is False
    assert loaded.intent.osc == settings.intent.osc
    assert serialization.to_dict(loaded) == persisted


@pytest.mark.parametrize("invalid", [None, 0, 1, "false", "true", [], {}])
def test_activation_notice_requires_boolean(invalid: object) -> None:
    persisted = serialization.to_dict(AppSettingsVNext())
    persisted["intent"]["osc"]["activation_notice_enabled"] = invalid

    with pytest.raises(ValueError, match="activation_notice_enabled"):
        from_vnext_dict(persisted)
    with pytest.raises(ValueError, match="activation_notice_enabled"):
        _with_osc(AppSettingsVNext(), activation_notice_enabled=invalid)
