from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

ARMS = ("F0-FROZEN-FLOAT", "H-HEAD", "T2-TOP", "TA-ALL-TEMPORAL")
TEMPORAL_LAYER_COUNT = 18
ACTIVITY_HEAD_PREFIXES = (
    "sortformer_modules.first_hidden_to_hidden.",
    "sortformer_modules.single_hidden_to_spks.",
)


class ParameterPolicyError(RuntimeError):
    pass


def _sortformer_name(name: str) -> str:
    return name.removeprefix("sortformer.")


def temporal_layer_index(name: str) -> int | None:
    match = re.match(r"^transformer_encoder\.layers\.(\d+)\.", _sortformer_name(name))
    return int(match.group(1)) if match else None


def is_activity_head(name: str) -> bool:
    value = _sortformer_name(name)
    return any(value.startswith(prefix) for prefix in ACTIVITY_HEAD_PREFIXES)


def should_train(name: str, arm: str) -> bool:
    if arm not in ARMS:
        raise ParameterPolicyError(f"unknown arm: {arm}")
    if name.startswith("psem_head."):
        return arm != "F0-FROZEN-FLOAT"
    layer = temporal_layer_index(name)
    if arm == "T2-TOP" and layer in {16, 17}:
        return True
    if arm == "TA-ALL-TEMPORAL" and layer is not None:
        return True
    if arm in {"T2-TOP", "TA-ALL-TEMPORAL"} and is_activity_head(name):
        return True
    return False


def audit_parameter_graph(names: Iterable[str]) -> dict[str, Any]:
    values = tuple(names)
    if len(values) != len(set(values)):
        raise ParameterPolicyError("parameter names must be unique")
    layers = {index for name in values if (index := temporal_layer_index(name)) is not None}
    if layers != set(range(TEMPORAL_LAYER_COUNT)):
        raise ParameterPolicyError(f"expected temporal layers 0..17, observed {sorted(layers)}")
    activity_groups = {
        prefix: tuple(name for name in values if _sortformer_name(name).startswith(prefix))
        for prefix in ACTIVITY_HEAD_PREFIXES
    }
    if any(not group for group in activity_groups.values()):
        raise ParameterPolicyError("the exact Sortformer activity-head groups are required")
    psem = tuple(name for name in values if name.startswith("psem_head."))
    if not psem:
        raise ParameterPolicyError("the PSEM head parameter group is required")
    return {
        "parameter_count": len(values),
        "temporal_layers": sorted(layers),
        "activity_head": {key: list(group) for key, group in activity_groups.items()},
        "psem_head": list(psem),
        "trainable_by_arm": {
            arm: [name for name in values if should_train(name, arm)] for arm in ARMS
        },
    }


def apply_parameter_policy(model: Any, arm: str) -> dict[str, Any]:
    parameters = tuple(model.named_parameters())
    inventory = audit_parameter_graph(name for name, _ in parameters)
    for name, parameter in parameters:
        parameter.requires_grad_(should_train(name, arm))
    actual = [name for name, parameter in parameters if parameter.requires_grad]
    expected = inventory["trainable_by_arm"][arm]
    if actual != expected:
        raise ParameterPolicyError("applied trainability differs from the audited whitelist")
    return {**inventory, "arm": arm, "trainable": actual}
