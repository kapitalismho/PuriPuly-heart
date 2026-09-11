from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import io
import json
import os
import platform
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from puripuly_heart.config.overlay_calibration import OverlayCalibration
from puripuly_heart.core.overlay.bridge import OverlayBridge
from puripuly_heart.core.overlay.diagnostics import OverlayDiagnosticsRecorder
from puripuly_heart.core.overlay.openvr_vendor import validate_vendored_openvr_bundle
from puripuly_heart.core.overlay.presenter import OverlayPresenter
from puripuly_heart.core.overlay.process import (
    HANDOFF_EXPERIMENT_CACHED_FRAME_REHANDOFF,
    HANDOFF_EXPERIMENT_OFF,
    DefaultOverlayProcessRunner,
    OverlayProcessManager,
    normalize_handoff_experiment,
)
from puripuly_heart.core.overlay.sink import OverlayApplicationReceipt, OverlayEventAdapter
from puripuly_heart.core.runtime.overlay import OverlayRuntimeHandle
from puripuly_heart.domain.models import Transcript

SCHEMA = "ovr-hmd-measurement-preparation-v2"
RUN_SCHEMA = "ovr-hmd-measurement-run-v2"
OBSERVATION_SCHEMA = "ovr-hmd-measurement-observation-v1"
SOURCE_EXE_SHA256 = "4a2cb8c815f900a0cc67347fa406d4e6cef825c6e593d2682bd3bc1f687ba737"
VENDORED_DLL_SHA256 = "bab8ac6ef64e68a9ca53315b0014d131088584b2efdfa6db511d67ec03cfcb4a"
ACCEPTED_SOURCE = "9867b819afb2d26d3e8cfbc09f4de83f815f8fde"
NATIVE_SOURCE = "65104b780a49186f4287901a31147cdd4a075595"
PYTHON_SOURCE = "65104b780a49186f4287901a31147cdd4a075595"
EXPERIMENT_ARMS = (
    HANDOFF_EXPERIMENT_OFF,
    HANDOFF_EXPERIMENT_CACHED_FRAME_REHANDOFF,
)
EXPECTED_STARTUP_CONTRACT = {
    "app_version": "2.6.1",
    "contract_version": 8,
    "execution_contract": {"revision": "r2", "version": 1},
    "native_presentation_retry": {"ownership": "exclusive", "version": 1},
}
SEQUENCE_REVISION = "ov01-short-r2"
SEQUENCE_ORDER = (
    "initial_clear",
    "self_m1_provisional",
    "self_m1_final",
    "self_m1_translation",
    "self_plus_peer",
    "clear_self",
    "close_peer",
    "natural_ttl_hide_wait",
    "true_input_idle",
    "self_m2_redisplay",
    "final_clear",
)
PHYSICAL_RESULTS = ("no_issue", "stale", "missing", "wrong_text", "placement")


class MeasurementError(RuntimeError):
    pass


class MeasurementProcessRunner(DefaultOverlayProcessRunner):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.last_process: Any | None = None

    async def spawn(self, executable_path: Path, manifest_path: Path) -> Any:
        process = await super().spawn(executable_path, manifest_path)
        self.last_process = process
        return process


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _check_startup_contract(executable: Path) -> dict[str, object]:
    try:
        completed = subprocess.run(
            [str(executable), "--check-startup-contract"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise MeasurementError(f"startup contract inspection failed: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no output"
        raise MeasurementError(
            f"startup contract inspection exited {completed.returncode}: {detail}"
        )
    try:
        contract = json.loads(completed.stdout.strip())
    except json.JSONDecodeError as exc:
        raise MeasurementError("startup contract inspection returned invalid JSON") from exc
    if contract != EXPECTED_STARTUP_CONTRACT:
        raise MeasurementError(
            "startup contract mismatch; refusing to substitute a different native binary"
        )
    return contract


def _steamvr_inventory() -> dict[str, object]:
    roots: list[Path] = []
    for variable in ("ProgramFiles(x86)", "ProgramFiles"):
        value = os.environ.get(variable)
        if value:
            roots.append(Path(value) / "Steam" / "steamapps")
    manifest = next(
        (
            root / "appmanifest_250820.acf"
            for root in roots
            if (root / "appmanifest_250820.acf").is_file()
        ),
        None,
    )
    install = next(
        (root / "common" / "SteamVR" for root in roots if (root / "common" / "SteamVR").is_dir()),
        None,
    )
    build_id: str | None = None
    channel: str | None = None
    if manifest is not None:
        text = manifest.read_text(encoding="utf-8", errors="replace")
        build_match = re.search(r'"buildid"\s+"([^"]+)"', text)
        channel_match = re.search(r'"BetaKey"\s+"([^"]+)"', text, re.IGNORECASE)
        build_id = build_match.group(1) if build_match else None
        channel = channel_match.group(1) if channel_match else None
    return {
        "installed": bool(install),
        "install_path": str(install) if install is not None else "unknown",
        "build_id": build_id or "unknown",
        "channel": channel or "unknown",
    }


def _gpu_inventory() -> list[dict[str, str]] | str:
    if os.name != "nt":
        return "unknown"
    command = (
        "Get-CimInstance Win32_VideoController | "
        "Select-Object Name,DriverVersion | ConvertTo-Json -Compress"
    )
    try:
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if completed.returncode != 0 or not completed.stdout.strip():
            return "unknown"
        parsed = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return "unknown"
    rows = parsed if isinstance(parsed, list) else [parsed]
    result: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        result.append(
            {
                "name": str(row.get("Name") or "unknown"),
                "driver_version": str(row.get("DriverVersion") or "unknown"),
            }
        )
    return result or "unknown"


def _environment_inventory() -> dict[str, object]:
    return {
        "os": {
            "system": platform.system() or "unknown",
            "release": platform.release() or "unknown",
            "version": platform.version() or "unknown",
            "machine": platform.machine() or "unknown",
        },
        "gpu": _gpu_inventory(),
        "steamvr": _steamvr_inventory(),
        "hmd": "not_observable",
        "connection_runtime_path": "unknown",
    }


def _stage_root(session: str) -> Path:
    return Path(tempfile.gettempdir()) / "puripuly-heart" / "ovr-measurement" / session


def prepare_session(executable: Path) -> Path:
    executable = executable.expanduser().resolve()
    if not executable.is_file():
        raise MeasurementError(f"native executable not found: {executable}")
    actual_exe_hash = _sha256(executable)
    if actual_exe_hash != SOURCE_EXE_SHA256:
        raise MeasurementError(
            f"native executable hash mismatch: expected {SOURCE_EXE_SHA256}, got {actual_exe_hash}"
        )
    contract = _check_startup_contract(executable)
    try:
        bundle = validate_vendored_openvr_bundle(ROOT / "third_party" / "openvr")
    except (FileNotFoundError, ValueError) as exc:
        raise MeasurementError(f"vendored OpenVR bundle validation failed: {exc}") from exc
    if bundle.dll_sha256 != VENDORED_DLL_SHA256:
        raise MeasurementError("vendored OpenVR DLL receipt mismatch")
    session = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + secrets.token_hex(4)
    stage = _stage_root(session)
    run_dir = stage / "run"
    run_dir.mkdir(parents=True, exist_ok=False)
    staged_exe = run_dir / "PuriPulyHeartOverlay.exe"
    staged_dll = run_dir / "openvr_api.dll"
    shutil.copy2(executable, staged_exe)
    shutil.copy2(bundle.dll_path, staged_dll)
    if _sha256(staged_exe) != SOURCE_EXE_SHA256 or _sha256(staged_dll) != VENDORED_DLL_SHA256:
        raise MeasurementError("isolated pair verification failed after copy")
    if _check_startup_contract(staged_exe) != contract:
        raise MeasurementError("isolated executable contract changed after copy")
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "prepared_at": _utc_now(),
        "session": session,
        "provenance": {
            "accepted_source": ACCEPTED_SOURCE,
            "native_source": NATIVE_SOURCE,
            "python_source": PYTHON_SOURCE,
            "measurement_script_sha256": _sha256(Path(__file__).resolve()),
            "relationship": "historical_acceptance_native_build_and_python_source_recorded_separately",
            "build_provenance": "receipt_verified_release_binary_from_native_source",
        },
        "pair": {
            "executable": {
                "name": staged_exe.name,
                "sha256": SOURCE_EXE_SHA256,
                "app_version": "2.6.1",
            },
            "openvr_runtime": {
                "name": staged_dll.name,
                "sha256": VENDORED_DLL_SHA256,
                "vendor_ref": "ValveSoftware/openvr@v2.15.6",
            },
            "startup_contract": contract,
            "protocol": 8,
            "execution_contract": {"version": 1, "revision": "r2"},
            "native_presentation_retry": {"version": 1, "ownership": "exclusive"},
        },
        "effective_behavior": {
            "backend": "D3D11_DirectX_OpenVR_API_only",
            "quiet_tail_profile": "p05",
            "post_submit_flush": "absent",
            "target": "steamvr",
            "calibration": {
                "anchor": "head_locked",
                "offset_x": 0.0,
                "offset_y": -0.45,
                "distance": 1.1,
                "text_scale": 1.0,
                "background_alpha": 0.24,
            },
            "logging_mode": "detailed",
            "logging_comparison": "both_arms_use_detailed_instead_of_earlier_basic_measurement",
            "handoff_experiment_default": HANDOFF_EXPERIMENT_OFF,
        },
        "preregistration": {
            "sequence_revision": SEQUENCE_REVISION,
            "order": list(SEQUENCE_ORDER),
            "operation_count": len(SEQUENCE_ORDER),
            "injected_event_count": 9,
            "default_readable_hold_seconds": 3.0,
            "minimum_true_input_idle_seconds": 30.0,
            "marker_policy": "changes_only_on_logical_caption_revision",
            "physical_default": "not_observable",
            "failure_rule": "any guard, startup, runtime, receipt, hide, timeout, or cleanup failure fails the software run",
            "stop_rule": "a failed software run cannot be upgraded by manual observation",
            "experiment": {
                "arms": list(EXPERIMENT_ARMS),
                "only_controlled_difference": "PURIPULY_OVERLAY_HANDOFF_EXPERIMENT",
                "same_staged_binary_required": True,
                "automatic_live_launch": False,
                "cached_frame_rehandoff": "experiment_only_not_r2_conformance",
                "actual_reuse_required_for_discrimination": True,
            },
            "excluded": [
                "spatial_fault_injection",
                "sleep_wake_automation",
                "crash_injection",
                "long_session_acceptance",
                "latency_claims",
            ],
        },
        "environment": _environment_inventory(),
        "privacy": {
            "session_local_alias_only": True,
            "contains_session_token": False,
            "contains_hmd_serial": False,
            "contains_private_user_path": False,
            "contains_raw_chat": False,
        },
    }
    report = stage / "preparation.json"
    _write_json(report, payload)
    (stage / "preparation.sha256").write_text(
        _sha256(report) + "  preparation.json\n", encoding="ascii"
    )
    return stage


def load_prepared_stage(stage: Path) -> tuple[dict[str, object], Path, Path]:
    stage = stage.expanduser().resolve()
    report = stage / "preparation.json"
    if not report.is_file():
        raise MeasurementError(f"preparation report not found: {report}")
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MeasurementError(f"invalid preparation report: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise MeasurementError("unsupported preparation report schema")
    pair = payload.get("pair")
    if not isinstance(pair, dict) or pair.get("startup_contract") != EXPECTED_STARTUP_CONTRACT:
        raise MeasurementError("prepared startup contract metadata mismatch")
    executable = stage / "run" / "PuriPulyHeartOverlay.exe"
    dll = stage / "run" / "openvr_api.dll"
    if not executable.is_file() or _sha256(executable) != SOURCE_EXE_SHA256:
        raise MeasurementError("prepared executable is missing or mismatched")
    if not dll.is_file() or _sha256(dll) != VENDORED_DLL_SHA256:
        raise MeasurementError("prepared OpenVR DLL is missing or mismatched")
    _check_startup_contract(executable)
    return payload, executable, dll


def inspect_process_names() -> set[str]:
    if os.name != "nt":
        raise MeasurementError("live process inspection is unavailable on this operating system")
    try:
        completed = subprocess.run(
            ["tasklist", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise MeasurementError(f"process inspection failed: {exc}") from exc
    if completed.returncode != 0:
        raise MeasurementError("process inspection failed; live launch is blocked")
    try:
        rows = list(csv.reader(io.StringIO(completed.stdout)))
    except csv.Error as exc:
        raise MeasurementError(f"process inspection returned invalid data: {exc}") from exc
    names = {row[0].strip().lower() for row in rows if row and row[0].strip()}
    if not names:
        raise MeasurementError("process inspection returned no processes; live launch is blocked")
    return names


def validate_live_guard(process_names: set[str], *, confirmed_hmd_ready: bool) -> None:
    normalized = {name.lower() for name in process_names}
    if not confirmed_hmd_ready:
        raise MeasurementError("live mode requires --confirm-hmd-ready")
    if "puripulyheartoverlay.exe" in normalized:
        raise MeasurementError(
            "PuriPulyHeartOverlay.exe is already running; it was not stopped or replaced"
        )
    required = {"vrserver.exe", "vrcompositor.exe"}
    missing = sorted(required - normalized)
    if missing:
        raise MeasurementError(
            "SteamVR is not ready; start it yourself before live mode (missing "
            + ", ".join(missing)
            + ")"
        )


def _receipt_row(step: str, receipt: OverlayApplicationReceipt) -> dict[str, object]:
    return {
        "step": step,
        "stage": receipt.stage,
        "outcome": receipt.outcome,
        "scene_revision": receipt.scene_revision,
        "cause": receipt.cause,
    }


async def _wait_while_healthy(manager: OverlayProcessManager | None, seconds: float) -> None:
    deadline = asyncio.get_running_loop().time() + seconds
    while True:
        if manager is not None and manager.state != "connected":
            raise MeasurementError(
                f"native supervisor left connected state: {manager.failure_reason or manager.state}"
            )
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            return
        await asyncio.sleep(min(0.2, remaining))


async def _wait_for_natural_hide(
    presenter: OverlayPresenter,
    manager: OverlayProcessManager | None,
    timeout_seconds: float,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while presenter.snapshot().blocks:
        if asyncio.get_running_loop().time() >= deadline:
            raise MeasurementError("caption did not clear through the existing TTL policy")
        await _wait_while_healthy(manager, 0.1)


async def _run_fixed_sequence(
    presenter: OverlayPresenter,
    manager: OverlayProcessManager | None,
    *,
    hold_seconds: float,
    idle_seconds: float,
    offline: bool,
    diagnostics: OverlayDiagnosticsRecorder,
) -> tuple[list[dict[str, object]], float]:
    adapter = OverlayEventAdapter()
    self_turn = uuid4()
    peer_turn = uuid4()
    second_self_turn = uuid4()
    receipts: list[dict[str, object]] = []

    async def emit(step: str, event: Any) -> OverlayApplicationReceipt:
        receipt = await presenter.emit(event)
        receipts.append(_receipt_row(step, receipt))
        if receipt.outcome != "applied":
            raise MeasurementError(f"{step} was not applied: {receipt.outcome} {receipt.cause}")
        return receipt

    async def hold_and_checkpoint(
        step: str, receipt: OverlayApplicationReceipt, seconds: float
    ) -> None:
        await _wait_while_healthy(manager, seconds)
        diagnostics.capture_measurement_phase(step, scene_revision=receipt.scene_revision)

    receipt = await emit("initial_clear", adapter.self_active_clear())
    await hold_and_checkpoint("initial_clear", receipt, hold_seconds)
    receipt = await emit(
        "self_m1_provisional",
        adapter.self_active_update(
            text="[OV01-M1-P] synthetic provisional",
            utterance_id=self_turn,
            occupant_key="ov01-self-m1",
            source_language="en",
            target_language="ko",
        ),
    )
    await hold_and_checkpoint("self_m1_provisional", receipt, hold_seconds)
    receipt = await emit(
        "self_m1_final",
        adapter.transcript_final(
            Transcript(
                utterance_id=self_turn,
                text="[OV01-M1-F] synthetic final",
                is_final=True,
                channel="self",
            ),
            source_language="en",
            target_language="ko",
        ),
    )
    await hold_and_checkpoint("self_m1_final", receipt, hold_seconds)
    receipt = await emit(
        "self_m1_translation",
        adapter.translation_final(
            utterance_id=self_turn,
            channel="self",
            text="[OV01-M1-T] controlled translation",
            source_text="[OV01-M1-F] synthetic final",
            source_language="en",
            target_language="ko",
            applied_context_mode=None,
        ),
    )
    await hold_and_checkpoint("self_m1_translation", receipt, hold_seconds)
    receipt = await emit(
        "self_plus_peer",
        adapter.transcript_final(
            Transcript(
                utterance_id=peer_turn,
                text="[OV01-P1] synthetic peer row",
                is_final=True,
                channel="peer",
            ),
            source_language="en",
            target_language="ko",
        ),
    )
    await hold_and_checkpoint("self_plus_peer", receipt, hold_seconds)
    await emit("clear_self", adapter.self_active_clear())
    await emit(
        "close_peer",
        adapter.utterance_closed(utterance_id=peer_turn, channel="peer", is_final=True),
    )
    await _wait_for_natural_hide(presenter, manager, 12.0)
    idle_started = time.monotonic()
    await _wait_while_healthy(manager, idle_seconds)
    actual_idle = time.monotonic() - idle_started
    diagnostics.capture_measurement_phase("true_input_idle", scene_revision=None)
    if not offline and actual_idle < 30.0:
        raise MeasurementError("live input-idle interval was shorter than 30 seconds")
    receipt = await emit(
        "self_m2_redisplay",
        adapter.self_active_update(
            text="[OV01-M2] synthetic redisplay after idle",
            utterance_id=second_self_turn,
            occupant_key="ov01-self-m2",
            source_language="en",
            target_language="ko",
        ),
    )
    await hold_and_checkpoint("self_m2_redisplay", receipt, hold_seconds)
    receipt = await emit("final_clear", adapter.self_active_clear())
    await hold_and_checkpoint("final_clear", receipt, hold_seconds)
    if presenter.snapshot().blocks:
        raise MeasurementError("final clear left drawable caption blocks")
    return receipts, actual_idle


async def run_measurement(
    stage: Path,
    *,
    live: bool,
    hold_seconds: float,
    idle_seconds: float,
    run_timeout_seconds: float,
    arm: str = HANDOFF_EXPERIMENT_OFF,
) -> Path:
    if live and hold_seconds != 3.0:
        raise MeasurementError(
            "live readable hold must be exactly 3.0 seconds for preregistered comparison"
        )
    arm = normalize_handoff_experiment(arm)
    preparation, executable, _dll = load_prepared_stage(stage)
    session = str(preparation["session"])
    mode = "live" if live else "offline_dry_run"
    run_id = f"{mode}-{arm}-{secrets.token_hex(4)}"
    report_path = stage / f"run-{run_id}.json"
    print(f"Run report: {report_path}", flush=True)
    overlay_instance_id = f"overlay-{uuid4().hex}"
    runtime = OverlayRuntimeHandle(
        overlay_instance_id=overlay_instance_id,
        shutdown_grace_s=3.0,
    )
    diagnostics_dir = stage / "diagnostics" / run_id
    diagnostics = OverlayDiagnosticsRecorder(
        overlay_instance_id=overlay_instance_id,
        diagnostics_dir=diagnostics_dir,
        logging_mode="detailed",
    )
    runtime.attach_diagnostics(diagnostics)
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        diagnostics=diagnostics,
        runtime_log_detailed=None,
        show_translation=True,
        show_peer_original=True,
        translation_enabled=True,
        peer_presentation_refresh_burst=True,
        self_presentation_refresh_burst=True,
        native_retry_trigger_emission=False,
        task_factory=runtime.create_child_task,
    )
    runtime.adopt_presenter(presenter)
    bridge = OverlayBridge(
        session_token=secrets.token_urlsafe(16),
        initial_snapshot=presenter.snapshot(),
        overlay_instance_id=overlay_instance_id,
        runtime_generation=1,
        diagnostics=diagnostics,
        runtime_logging_mode="detailed",
        desktop_runtime_controls_enabled=False,
        task_factory=runtime.create_child_task,
    )
    runtime.attach_bridge(bridge)
    manager: OverlayProcessManager | None = None
    receipts: list[dict[str, object]] = []
    actual_idle: float | None = None
    software_outcome = "failed"
    failure_reason: str | None = None
    manager_state_before_teardown: str | None = None
    cleanup_outcome = "not_started"
    shutdown_receipt: dict[str, object] | str = "not_applicable"
    diagnostics_receipt: dict[str, object] = {"outcome": "not_started"}
    started_at = _utc_now()
    started_monotonic = time.monotonic()
    try:
        await bridge.start()
        presenter.attach_bridge(bridge)
        if bridge.snapshot() != presenter.snapshot():
            await bridge.replace_snapshot(presenter.snapshot())
        if live:
            runner = MeasurementProcessRunner(
                executable_path=executable,
                task_factory=runtime.create_child_task,
                quiet_tail_profile="p05",
                handoff_experiment=arm,
            )
            manager = OverlayProcessManager(
                process_runner=runner,
                bridge_url=bridge.url,
                bridge_messages=bridge.messages,
                session_token=bridge.session_token,
                locale="en",
                log_dir=str(diagnostics_dir),
                startup_timeout_ms=15000,
                overlay_instance_id=overlay_instance_id,
                logging_mode="detailed",
                quiet_tail_profile="p05",
                handoff_experiment=arm,
                diagnostics_dir=diagnostics_dir,
                diagnostics=diagnostics,
                task_factory=runtime.create_child_task,
                selected_target="steamvr",
                geometry_authority="native",
                graceful_shutdown_request=bridge.broadcast_shutdown,
                retry_ownership_changed=presenter.update_native_retry_ownership,
            )
            runtime.attach_process_manager(manager)
            await manager.start()
            if manager.state != "connected":
                raise MeasurementError(
                    f"native startup failed: {manager.failure_reason or manager.state}"
                )
        effective_hold = hold_seconds if live else min(hold_seconds, 0.05)
        effective_idle = idle_seconds if live else min(idle_seconds, 0.1)
        async with asyncio.timeout(run_timeout_seconds):
            receipts, actual_idle = await _run_fixed_sequence(
                presenter,
                manager,
                hold_seconds=effective_hold,
                idle_seconds=effective_idle,
                offline=not live,
                diagnostics=diagnostics,
            )
        manager_state_before_teardown = manager.state if manager is not None else "not_applicable"
        software_outcome = "pass"
    except TimeoutError:
        failure_reason = "run_timeout"
    except asyncio.CancelledError:
        failure_reason = "cancelled"
        raise
    except Exception as exc:
        failure_reason = str(exc)
    finally:
        try:
            await runtime.close(preserve_presenter_state=False)
            cleanup_outcome = "complete"
        except Exception as exc:
            cleanup_outcome = "failed"
            software_outcome = "failed"
            cleanup_failure = f"cleanup failed: {exc}"
            failure_reason = (
                f"{failure_reason}; {cleanup_failure}" if failure_reason else cleanup_failure
            )
        if manager is not None:
            shutdown_receipt = manager.shutdown_receipt()
            if shutdown_receipt.get("cleanup_succeeded") is not True:
                cleanup_outcome = "failed"
        child_exit: int | str
        if not live:
            child_exit = "not_applicable"
        elif manager is None:
            child_exit = "not_started"
        else:
            receipt_exit = (
                shutdown_receipt.get("exit_code") if isinstance(shutdown_receipt, dict) else None
            )
            child_exit = receipt_exit if isinstance(receipt_exit, int) else "unconfirmed"
            receipt_passed = (
                isinstance(shutdown_receipt, dict)
                and shutdown_receipt.get("exit_confirmed") is True
                and shutdown_receipt.get("graceful_completed") is True
                and shutdown_receipt.get("exit_code") == 0
                and shutdown_receipt.get("acknowledged") is True
                and shutdown_receipt.get("forced") is False
                and shutdown_receipt.get("cleanup_succeeded") is True
                and shutdown_receipt.get("terminal_cause") is None
                and manager.state == "off"
            )
            if not receipt_passed:
                software_outcome = "failed"
                terminal_cause = (
                    shutdown_receipt.get("terminal_cause")
                    if isinstance(shutdown_receipt, dict)
                    else None
                )
                failure_reason = failure_reason or (
                    terminal_cause
                    if isinstance(terminal_cause, str) and terminal_cause
                    else "owned child shutdown did not complete normally"
                )
        if manager is not None and manager.failure_reason and software_outcome == "pass":
            software_outcome = "failed"
            failure_reason = manager.failure_reason
        evidence = diagnostics.evidence_summary()
        actual_reuse = evidence["cached_frame_rehandoff_observed"] is True
        if live and arm == HANDOFF_EXPERIMENT_CACHED_FRAME_REHANDOFF and not actual_reuse:
            software_outcome = "failed"
            failure_reason = failure_reason or "cached-frame arm produced no actual reuse evidence"
        owned_failure_receipt = (
            diagnostics.last_dump_receipt
            if software_outcome == "failed"
            and manager is not None
            and manager.failure_reason is not None
            else None
        )
        diagnostics_receipt = owned_failure_receipt or await diagnostics.dump_evidence(
            outcome="success" if software_outcome == "pass" else "failure",
            run_id=run_id,
            experiment_arm=arm,
            experiment_only=arm != HANDOFF_EXPERIMENT_OFF,
            manager_state=manager.state if manager is not None else "not_applicable",
        )
        run_payload: dict[str, object] = {
            "schema": RUN_SCHEMA,
            "run_id": run_id,
            "session": session,
            "mode": mode,
            "started_at": started_at,
            "finished_at": _utc_now(),
            "elapsed_seconds": time.monotonic() - started_monotonic,
            "sequence_revision": SEQUENCE_REVISION,
            "preregistered_order": list(SEQUENCE_ORDER),
            "experiment": {
                "arm": arm,
                "environment_override": {
                    "name": "PURIPULY_OVERLAY_HANDOFF_EXPERIMENT",
                    "effective_value": arm,
                },
                "experiment_only": arm != HANDOFF_EXPERIMENT_OFF,
                "qualifies_for_r2_conformance": arm == HANDOFF_EXPERIMENT_OFF,
                "qualifies_for_supervisor_refill": arm == HANDOFF_EXPERIMENT_OFF,
                "same_binary_pair_id": session,
                "actual_evidence": evidence,
                "discrimination": (
                    "actual_cached_frame_rehandoff_observed" if actual_reuse else "not_observed"
                ),
            },
            "timing": {
                "requested_hold_seconds": hold_seconds,
                "requested_true_idle_seconds": idle_seconds,
                "effective_hold_seconds": hold_seconds if live else min(hold_seconds, 0.05),
                "effective_input_idle_seconds": idle_seconds if live else min(idle_seconds, 0.1),
                "actual_input_idle_seconds": actual_idle if actual_idle is not None else "unknown",
                "offline_waits_shortened": not live,
                "run_timeout_seconds": run_timeout_seconds,
                "physical_clock_uncertainty": "unknown",
            },
            "software": {
                "outcome": software_outcome,
                "failure_reason": failure_reason,
                "acceptance_scope": (
                    "presenter_bridge_native_process_seam"
                    if live
                    else "presenter_bridge_local_acceptance_only"
                ),
                "manager_state_before_teardown": manager_state_before_teardown or "unknown",
                "manager_state_after_teardown": (
                    manager.state if manager is not None else "not_applicable"
                ),
                "cleanup": cleanup_outcome,
                "owned_child_exit": child_exit,
                "shutdown": shutdown_receipt,
                "receipts": receipts,
                "diagnostics": diagnostics_receipt,
            },
            "physical_hmd": {
                "result": "not_observable",
                "manual_observation_recorded": False,
                "api_success_is_not_physical_pass": True,
            },
            "pair": preparation["pair"],
            "provenance": preparation["provenance"],
            "privacy": {
                "contains_session_token": False,
                "contains_hmd_serial": False,
                "contains_private_user_path": False,
                "contains_raw_chat": False,
            },
        }
        _write_json(report_path, run_payload)
        (report_path.with_suffix(report_path.suffix + ".sha256")).write_text(
            _sha256(report_path) + f"  {report_path.name}\n", encoding="ascii"
        )
    if software_outcome != "pass":
        raise MeasurementError(failure_reason or "measurement run failed")
    return report_path


def record_observation(
    run_report: Path,
    *,
    result: str,
    note: str,
    uncertainty: str,
) -> Path:
    if result not in PHYSICAL_RESULTS:
        raise MeasurementError(f"unsupported physical observation result: {result}")
    if not note.strip() or not uncertainty.strip():
        raise MeasurementError("manual observation requires non-empty note and uncertainty")
    run_report = run_report.expanduser().resolve()
    try:
        run = json.loads(run_report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MeasurementError(f"invalid run report: {exc}") from exc
    if not isinstance(run, dict) or run.get("schema") != RUN_SCHEMA:
        raise MeasurementError("unsupported run report schema")
    software = run.get("software")
    if not isinstance(software, dict):
        raise MeasurementError("run report has no software outcome")
    software_outcome = software.get("outcome")
    observation = {
        "schema": OBSERVATION_SCHEMA,
        "recorded_at": _utc_now(),
        "run_id": run.get("run_id"),
        "session": run.get("session"),
        "new_run_performed": True,
        "result": result,
        "qualitative_note": note,
        "clock_uncertainty": uncertainty,
        "latency_claim": "not_measured",
        "software_run_outcome": software_outcome,
        "overall_outcome": "failed" if software_outcome != "pass" else "observation_recorded",
        "cannot_upgrade_failed_software_run": True,
    }
    path = run_report.with_name(f"observation-{run.get('run_id', 'unknown')}.json")
    _write_json(path, observation)
    (path.with_suffix(path.suffix + ".sha256")).write_text(
        _sha256(path) + f"  {path.name}\n", encoding="ascii"
    )
    return path


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _live_hold(value: str) -> float:
    parsed = float(value)
    if parsed != 3.0:
        raise argparse.ArgumentTypeError(
            "live readable hold must be exactly 3.0 seconds for preregistered comparison"
        )
    return parsed


def _live_idle(value: str) -> float:
    parsed = float(value)
    if parsed < 30.0:
        raise argparse.ArgumentTypeError("live idle must be at least 30 seconds")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and run paired bounded OV01 residual-flicker experiment arms.",
        epilog=(
            "Operator flow:\n"
            "  1. python scripts/bench_ovr_hmd_measurement.py prepare\n"
            "  2. python scripts/bench_ovr_hmd_measurement.py dry-run --stage <printed-stage> --arm off\n"
            "  3. python scripts/bench_ovr_hmd_measurement.py dry-run --stage <printed-stage> --arm cached_frame_rehandoff\n"
            "  4. Start SteamVR yourself, wear the HMD, and confirm no other PuriPuly overlay is running.\n"
            "  5. Run live --arm off and live --arm cached_frame_rehandoff separately against the same stage, "
            "each with --confirm-hmd-ready.\n"
            "  6. Record each observation against its printed report; preparation never claims a pair ran.\n\n"
            "Preparation and dry-run never launch SteamVR or the native overlay. Live mode never launches "
            "SteamVR, VRChat, or the installed app, never kills a preexisting process, and aborts if process "
            "inspection fails. Reports keep physical HMD visibility not_observable until an operator records "
            "a qualitative observation; API success is never a physical pass. The live readable hold is fixed "
            "at exactly 3.0 seconds and the live input-idle interval cannot be set below 30 seconds."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser(
        "prepare",
        help="validate the pinned pair and copy it to an isolated OS-temporary stage without launching it",
    )
    prepare.add_argument(
        "--executable",
        type=Path,
        default=Path("C:/ovr-target/release/PuriPulyHeartOverlay.exe"),
        help="receipt-pinned native executable (default: C:/ovr-target/release/PuriPulyHeartOverlay.exe)",
    )
    dry = subparsers.add_parser(
        "dry-run",
        help="validate the prepared pair and exercise actual presenter-to-bridge local acceptance only",
    )
    dry.add_argument("--stage", type=Path, required=True, help="stage printed by prepare")
    dry.add_argument("--arm", choices=EXPERIMENT_ARMS, required=True)
    dry.add_argument("--hold-seconds", type=_positive_float, default=3.0)
    dry.add_argument("--idle-seconds", type=_positive_float, default=30.0)
    dry.add_argument("--run-timeout-seconds", type=_positive_float, default=30.0)
    live = subparsers.add_parser(
        "live",
        help="run the staged native process through current presenter/bridge/process owners",
    )
    live.add_argument("--stage", type=Path, required=True, help="stage printed by prepare")
    live.add_argument("--arm", choices=EXPERIMENT_ARMS, required=True)
    live.add_argument(
        "--confirm-hmd-ready",
        action="store_true",
        help="required acknowledgment that SteamVR was operator-started, the HMD is worn, and no other overlay is expected",
    )
    live.add_argument(
        "--hold-seconds",
        type=_live_hold,
        default=3.0,
        help="preregistered readable hold after each caption revision (fixed: 3.0 seconds)",
    )
    live.add_argument("--idle-seconds", type=_live_idle, default=30.0)
    live.add_argument("--run-timeout-seconds", type=_positive_float, default=120.0)
    observe = subparsers.add_parser(
        "observe",
        help="write a run-correlated qualitative manual HMD observation without inventing latency",
    )
    observe.add_argument("--run-report", type=Path, required=True)
    observe.add_argument("--result", choices=PHYSICAL_RESULTS, required=True)
    observe.add_argument(
        "--note", required=True, help="brief qualitative description of what was seen"
    )
    observe.add_argument(
        "--uncertainty",
        required=True,
        help="manual observation/clock uncertainty in the operator's own terms",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            stage = prepare_session(args.executable)
            print(f"Prepared stage: {stage}")
            print(f"Preparation report: {stage / 'preparation.json'}")
            return 0
        if args.command == "dry-run":
            report = asyncio.run(
                run_measurement(
                    args.stage,
                    live=False,
                    hold_seconds=args.hold_seconds,
                    idle_seconds=args.idle_seconds,
                    run_timeout_seconds=args.run_timeout_seconds,
                    arm=args.arm,
                )
            )
            print(f"Offline dry-run passed: {report}")
            return 0
        if args.command == "live":
            names = inspect_process_names()
            validate_live_guard(names, confirmed_hmd_ready=args.confirm_hmd_ready)
            report = asyncio.run(
                run_measurement(
                    args.stage,
                    live=True,
                    hold_seconds=args.hold_seconds,
                    idle_seconds=args.idle_seconds,
                    run_timeout_seconds=args.run_timeout_seconds,
                    arm=args.arm,
                )
            )
            print(
                "Software run passed; experiment evidence is recorded and physical HMD remains "
                f"not_observable: {report}"
            )
            return 0
        if args.command == "observe":
            path = record_observation(
                args.run_report,
                result=args.result,
                note=args.note,
                uncertainty=args.uncertainty,
            )
            print(f"Observation recorded: {path}")
            return 0
    except KeyboardInterrupt:
        print(
            "Interrupted; owned runtime cleanup was requested and no physical pass is claimed.",
            file=sys.stderr,
        )
        return 130
    except MeasurementError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
