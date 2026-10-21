from __future__ import annotations

MANAGED_OPERATION_ID_PREFIX = "ph-mop-v1_"


def is_valid_operation_id(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith(MANAGED_OPERATION_ID_PREFIX):
        return False
    suffix = value[len(MANAGED_OPERATION_ID_PREFIX) :]
    if len(suffix) != 32:
        return False
    return all(character.isalnum() or character in "-_" for character in suffix)
