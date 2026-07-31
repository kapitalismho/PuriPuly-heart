from __future__ import annotations


def assert_lifecycle_structure(snapshot: dict[str, object]) -> None:
    assert isinstance(snapshot["owner"], str) and snapshot["owner"]
    assert isinstance(snapshot["resource_fields"], tuple)
    assert all(isinstance(field, str) and field for field in snapshot["resource_fields"])
    assert len(snapshot["resource_fields"]) > 0
    assert isinstance(snapshot["stop_ingress"], str) and snapshot["stop_ingress"]
    assert isinstance(snapshot["shutdown_policy"], str) and snapshot["shutdown_policy"]
    assert isinstance(snapshot["late_callback_rule"], str) and snapshot["late_callback_rule"]
