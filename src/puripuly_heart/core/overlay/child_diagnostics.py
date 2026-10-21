from __future__ import annotations

import builtins
import math
from collections.abc import Mapping

NATIVE_FAILURE_REASONS = frozenset(
    {
        "manifest_invalid",
        "contract_mismatch",
        "bridge_auth_failed",
        "steamvr_not_installed",
        "steamvr_not_running",
        "hmd_not_found",
        "openvr_init_failed",
        "renderer_init_failed",
        "gpu_readiness_late",
        "gpu_readiness_cancelled",
        "gpu_query_failed",
        "gpu_stalled",
        "bridge_failed",
        "render_failed",
        "openvr_failed",
        "runtime_disconnected",
        "stopped",
        "unknown",
        "startup_timeout",
        "spawn_failed",
        "unsupported_binary",
        "shutdown_cleanup_failed",
        "termination_unconfirmed",
        "reverse_control_capacity",
        "native_owner_unresponsive",
        "native_acceptance_timeout",
        "runtime_replaced",
        "window_reveal_lost",
        "window_visibility_unstable",
        "window_observation_failed",
        "window_bounds_failed",
        "window_native_ready_failed",
        "window_identity_failed",
        "window_configuration_failed",
        "missing_executable",
        "invalid_manifest",
        "runtime_control_invalid",
        "runtime_crashed",
        "startup_cancelled",
        "shutdown_terminal_unavailable",
        "shutdown_forced",
        "shutdown_not_acknowledged",
        "runtime_exit_nonzero",
    }
)
_CLASSIFICATIONS = frozenset(
    {
        "terminal_failed",
        "pose_unavailable",
        "recovering",
        "due",
        "intentional_hidden",
        "no_drawable_content",
        "healthy_idle",
    }
)
_STAGES = frozenset(
    {
        "logical_revision_accepted",
        "render_returned",
        "readiness_observed",
        "submission_attempted",
        "submission_returned",
        "visibility_requested",
        "compositor_observed",
        "visibility_observed",
    }
)
_COUNTERS = ("diagnostic_dropped", "reliable_delivery_failures", "rejected_records")
CHILD_PHASES = frozenset(
    {
        "launched",
        "page_configured",
        "native_ready",
        "bounds_confirmed",
        "visible_confirmed",
        "ready",
        "startup",
        "connected",
        "shutdown",
        "failed",
        "off",
    }
)
_PORT_REASONS = frozenset(
    {
        "binding_changed",
        "window_changed",
        "window_not_found",
        "ambiguous_window",
        "enum_windows_failed",
        "visible_bounds_not_retained",
        "bounds_not_retained",
        "native_ready_timeout",
        "pid_file_mismatch",
        "port_error",
        "closed",
        "process_unbound",
        "canonical_bounds_missing",
    }
)
_CLEANUP_STAGES = frozenset(
    {
        "window_close",
        "diagnostic_close",
        "parent_monitor_close",
        "writer_close",
        "process_terminate",
        "process_wait",
        "server_close",
        "task_close",
        "temp_cleanup",
        "reader_cleanup",
    }
)
_EXCEPTION_CLASSES = frozenset(
    name
    for name in dir(builtins)
    if isinstance(getattr(builtins, name), type)
    and issubclass(getattr(builtins, name), BaseException)
)


def safe_failure_reason(value: object) -> str:
    return value if isinstance(value, str) and value in NATIVE_FAILURE_REASONS else "unknown"


def project_child_owner_status(message: Mapping[str, object]) -> dict[str, object]:
    """Ingress allowlist: never forward arbitrary child status/detail to diagnostics."""
    projected: dict[str, object] = {"type": "owner_status"}
    rejected = len(
        set(message)
        - {
            "type",
            "overlay_instance_id",
            "runtime_generation",
            "health_challenge_id",
            "health_challenge_validated",
            "health_challenge_issued_at",
            "classification",
            "latest_applied_revision",
            "latest_handoff_revision",
            "current_covered_handoff",
            "confirmed_hide",
            "desired_visible",
            "observed_runtime_visible",
            "due_elapsed_ms",
            "in_flight_stage",
            "primary_failure_reason",
            "cleanup_failure_reason",
            "diagnostic_dropped",
            "reliable_delivery_failures",
            "rejected_records",
            "terminal_status_delivery",
        }
    )
    for key in ("overlay_instance_id", "runtime_generation", "health_challenge_id"):
        projected[key] = message.get(key)
    projected["health_challenge_validated"] = message.get("health_challenge_validated") is True
    classification = message.get("classification")
    projected["classification"] = (
        classification
        if isinstance(classification, str) and classification in _CLASSIFICATIONS
        else "unknown"
    )
    rejected += int(not isinstance(classification, str) or classification not in _CLASSIFICATIONS)
    for key in ("primary_failure_reason", "cleanup_failure_reason"):
        value = message.get(key)
        projected[key] = None if value is None else safe_failure_reason(value)
        rejected += int(value is not None and projected[key] != value)
    for key in ("latest_applied_revision", "latest_handoff_revision", "due_elapsed_ms", *_COUNTERS):
        value = message.get(key)
        if value is None:
            projected[key] = None
        elif type(value) is int and 0 <= value <= (1 << 63) - 1:
            projected[key] = value
        else:
            projected[key] = None
            rejected += 1
    for key in (
        "current_covered_handoff",
        "confirmed_hide",
        "desired_visible",
        "observed_runtime_visible",
    ):
        value = message.get(key)
        projected[key] = value if value is None or type(value) is bool else None
        rejected += int(value is not None and type(value) is not bool)
    stage = message.get("in_flight_stage")
    projected["in_flight_stage"] = stage if isinstance(stage, str) and stage in _STAGES else None
    rejected += int(stage is not None and (not isinstance(stage, str) or stage not in _STAGES))
    projected["rejected_records"] = (projected.get("rejected_records") or 0) + rejected
    return projected


def project_child_failure(message: Mapping[str, object]) -> dict[str, object]:
    projected = {
        "type": message.get("type"),
        "failure_reason": safe_failure_reason(message.get("failure_reason")),
    }
    for key in ("overlay_instance_id", "runtime_generation"):
        if key in message:
            projected[key] = message[key]
    for key in ("primary_failure_reason", "cleanup_failure_reason"):
        if key in message:
            value = message[key]
            projected[key] = None if value is None else safe_failure_reason(value)
    rejected = 0
    for key in ("exit_code", "errno", "winerror", *_COUNTERS):
        value = message.get(key)
        if type(value) is int and -(1 << 31) <= value < (1 << 63):
            projected[key] = value
        elif value is not None:
            rejected += 1
    for key in (
        "exception_class",
        "cause_class_1",
        "cause_class_2",
        "cause_class_3",
        "cause_class_4",
    ):
        value = message.get(key)
        if isinstance(value, str) and (value == "unclassified" or value in _EXCEPTION_CLASSES):
            projected[key] = value
        elif value is not None:
            rejected += 1
    for key, allowed in (
        ("startup_phase", CHILD_PHASES),
        ("stage", _STAGES | CHILD_PHASES),
        ("cleanup_stage", _CLEANUP_STAGES),
        ("evidence_state", frozenset({"retained", "incomplete", "unavailable", "unknown"})),
    ):
        value = message.get(key)
        if isinstance(value, str) and value in allowed:
            projected[key] = value
        elif value is not None:
            rejected += 1
    evidence = message.get("evidence")
    if isinstance(evidence, Mapping):
        safe_evidence: dict[str, object] = {}
        for key, value in evidence.items():
            if key == "overlay_instance_id":
                safe_evidence[key] = value
            elif key == "failure_reason":
                safe_evidence[key] = safe_failure_reason(value)
            elif key in {"startup_phase", "port_reason", "target"}:
                allowed = (
                    CHILD_PHASES
                    if key == "startup_phase"
                    else _PORT_REASONS if key == "port_reason" else {"desktop"}
                )
                if isinstance(value, str) and value in allowed:
                    safe_evidence[key] = value
                elif value is not None:
                    rejected += 1
            elif key in {
                "desktop_target",
                "title_confirmed",
                "visible_confirmed",
                "bounds_confirmed",
                "bounds_drift",
                "endpoint_matches",
            }:
                if value is None or type(value) is bool:
                    safe_evidence[key] = value
                else:
                    rejected += 1
            elif key in {
                "generation",
                "win32_error",
                "hwnd",
                "hwnd_owner_pid",
                "owner_pid",
                "pid_file_pid",
            }:
                if value is None or type(value) is int and 0 <= value < (1 << 63):
                    safe_evidence[key] = value
                else:
                    rejected += 1
            elif key in {"canonical_bounds", "observed_bounds"}:
                if value is None or (
                    isinstance(value, (list, tuple))
                    and len(value) == 4
                    and all(type(item) in (int, float) and math.isfinite(item) for item in value)
                ):
                    safe_evidence[key] = value
                else:
                    rejected += 1
            else:
                rejected += 1
        projected["evidence"] = safe_evidence
    elif evidence is not None:
        rejected += 1
    known = {
        "type",
        "failure_reason",
        "primary_failure_reason",
        "cleanup_failure_reason",
        "overlay_instance_id",
        "runtime_generation",
        "exit_code",
        "errno",
        "winerror",
        *_COUNTERS,
        "exception_class",
        "cause_class_1",
        "cause_class_2",
        "cause_class_3",
        "cause_class_4",
        "startup_phase",
        "stage",
        "cleanup_stage",
        "evidence_state",
        "evidence",
    }
    rejected += len(set(message) - known)
    if projected["failure_reason"] != message.get("failure_reason"):
        rejected += 1
    if rejected:
        projected["rejected_records"] = (projected.get("rejected_records") or 0) + rejected
    return projected
