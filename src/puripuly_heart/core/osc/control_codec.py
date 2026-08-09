from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from puripuly_heart.core.osc.control_schema import (
    OSC_PARAMETER_DEFINITIONS,
    OscParameterDefinition,
    parameter_definition_for_address,
    registry_for_parameter,
)


class OscControlCodecError(ValueError):
    pass


class UnknownOscControlValueError(OscControlCodecError):
    pass


@dataclass(frozen=True, slots=True)
class OscControlMessage:
    parameter: OscParameterDefinition
    value: bool | int

    @property
    def name(self) -> str:
        return self.parameter.name

    @property
    def address(self) -> str:
        return self.parameter.address


def validate_control_value(parameter: str | OscParameterDefinition, value: Any) -> bool | int:
    definition = (
        parameter
        if isinstance(parameter, OscParameterDefinition)
        else OSC_PARAMETER_DEFINITIONS.get(parameter)
    )
    if definition is None:
        raise OscControlCodecError(f"unknown PuriPuly OSC parameter: {parameter!r}")
    if definition.value_type == "bool":
        if not isinstance(value, bool):
            raise OscControlCodecError(f"{definition.name} requires an OSC boolean")
        return value
    if isinstance(value, bool) or not isinstance(value, int):
        raise OscControlCodecError(f"{definition.name} requires an OSC integer")
    registry = registry_for_parameter(definition.name)
    if registry is None or value not in registry:
        raise UnknownOscControlValueError(f"unknown {definition.name} OSC value: {value!r}")
    return value


def decode_control_message(address: str, *args: Any) -> OscControlMessage:
    try:
        definition = parameter_definition_for_address(address)
    except ValueError as exc:
        raise OscControlCodecError(str(exc)) from exc
    if not args:
        raise OscControlCodecError(f"{definition.name} requires one OSC value")
    if len(args) != 1:
        raise OscControlCodecError(f"{definition.name} requires exactly one OSC value")
    return OscControlMessage(definition, validate_control_value(definition, args[0]))


def encode_control_message(parameter: str, value: bool | int) -> tuple[str, bool | int]:
    definition = OSC_PARAMETER_DEFINITIONS.get(parameter)
    if definition is None:
        raise OscControlCodecError(f"unknown PuriPuly OSC parameter: {parameter!r}")
    return definition.address, validate_control_value(definition, value)


def decode_control_address(address: str, *args: Any) -> OscControlMessage:
    return decode_control_message(address, *args)


__all__ = [
    "OscControlCodecError",
    "OscControlMessage",
    "UnknownOscControlValueError",
    "decode_control_address",
    "decode_control_message",
    "encode_control_message",
    "validate_control_value",
]
