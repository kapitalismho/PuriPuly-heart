from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

HANDS = ("left", "right")
BUTTONS = ("grip", "trigger")
APP_KEY = "puripuly.probe.x1"
BINDING_HELP = (
    "SteamVR Input needs a binding for the actual controllers. While this probe runs, "
    "open SteamVR > Devices > Controller Input Binding, select 'PuriPuly X1 controller probe', "
    "create/activate a binding for /actions/x1: map each hand's physical grip to Grip, "
    "trigger click to Trigger, and haptic output to Haptic. Do not replace VRChat's bindings "
    "or enable experimental input overrides. No controller-specific defaults are shipped. "
    "Enable binding UI debugging options if needed to export/Replace Default Binding beside "
    "the printed private action manifest; reuse it with --action-manifest. "
    "Manifest loading alone does not establish binding activation or grip availability."
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def describe_error(exc):
    code = getattr(exc, "error_value", None)
    return f"{type(exc).__name__}{f' (code={code})' if code is not None else ''}: {exc}"


def pose_data(pose):
    valid = bool(pose.bDeviceIsConnected and pose.bPoseIsValid)
    matrix = None
    if valid:
        matrix = [[float(pose.mDeviceToAbsoluteTracking[r][c]) for c in range(4)] for r in range(3)]
        if not all(math.isfinite(value) for row in matrix for value in row):
            valid = False
            matrix = None
    return {
        "connected": bool(pose.bDeviceIsConnected),
        "pose_valid": valid,
        "tracking_result": int(pose.eTrackingResult),
        "position_m": [row[3] for row in matrix] if matrix else None,
        "device_to_standing": matrix,
    }


def hands_distance(hands):
    left, right = (hands[hand]["position_m"] for hand in HANDS)
    if left is None or right is None or hands["left"]["index"] == hands["right"]["index"]:
        return None
    return math.dist(left, right)


class Evidence:
    def __init__(self, root, input_mode, haptics):
        name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:10]
        self.path = root.expanduser().resolve() / name
        self.path.mkdir(parents=True, mode=0o700)
        self.events = (self.path / "events.jsonl").open("x", encoding="utf-8")
        self.started = time.perf_counter()
        self.previous = {}
        self.summary = {
            "status": "starting",
            "input_mode": input_mode,
            "haptics_opt_in": haptics,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "script": str(Path(__file__).resolve()),
            "python_executable": sys.executable,
            "samples": 0,
            "paired_pose_samples": 0,
            "hands": {
                hand: {
                    "connected_samples": 0,
                    "valid_pose_samples": 0,
                    "grip_available_samples": 0,
                    "trigger_available_samples": 0,
                    "grip_presses": 0,
                    "grip_releases": 0,
                    "trigger_presses": 0,
                    "trigger_releases": 0,
                }
                for hand in HANDS
            },
            "haptic_calls": [],
            "physical_observations": {
                key: "unperformed: human observation required"
                for key in (
                    "vrchat_running_in_vr",
                    "physical_hand_role_mapping",
                    "left_grip_feasibility",
                    "camera_or_object_grab_interference",
                    "avatar_gesture_interference",
                    "left_haptic_sensation",
                    "right_haptic_sensation",
                )
            },
            "distance_reference": "controller origins in one fresh standing-space pose snapshot; not a calibrated wrist anchor",
            "binding_help": BINDING_HELP,
        }

    def event(self, kind, **fields):
        record = {"event": kind, "elapsed_s": time.perf_counter() - self.started, **fields}
        self.events.write(json.dumps(record, allow_nan=False) + "\n")
        self.events.flush()
        return record

    def sample(self, hands):
        sequence = self.summary["samples"]
        distance = hands_distance(hands)
        self.summary["samples"] += 1
        self.summary["paired_pose_samples"] += distance is not None
        self.event("sample", sequence=sequence, hands=hands, hands_distance_m=distance)
        for hand, current in hands.items():
            counts = self.summary["hands"][hand]
            counts["connected_samples"] += current["connected"]
            counts["valid_pose_samples"] += current["pose_valid"]
            for button in BUTTONS:
                state = current["inputs"][button]
                counts[f"{button}_available_samples"] += state["available"]
                key = (hand, button)
                value = (current["index"], state["pressed"]) if state["available"] else None
                previous = self.previous.get(key)
                if (
                    value is not None
                    and previous is not None
                    and value[0] == previous[0]
                    and value[1] != previous[1]
                ):
                    counts[f"{button}_{'presses' if value[1] else 'releases'}"] += 1
                    self.event(
                        "button",
                        sequence=sequence,
                        hand=hand,
                        button=button,
                        pressed=value[1],
                        hands_distance_m=distance,
                    )
                    print(f"{hand} {button}={value[1]} distance_m={distance}", flush=True)
                self.previous[key] = value
        return distance

    def close(self):
        self.summary["elapsed_s"] = time.perf_counter() - self.started
        self.summary["ended_utc"] = datetime.now(timezone.utc).isoformat()
        self.events.close()
        write_json(self.path / "summary.json", self.summary)


class SteamVRInput:
    def __init__(self, vr, manifest):
        self.vr = vr
        self.api = vr.VRInput()
        self.api.setActionManifestPath(str(manifest))
        self.actions = {
            name: self.api.getActionHandle(
                f"/actions/x1/{'out' if name == 'haptic' else 'in'}/{name}"
            )
            for name in (*BUTTONS, "haptic")
        }
        self.sources = {hand: self.api.getInputSourceHandle(f"/user/hand/{hand}") for hand in HANDS}
        self.sets = (vr.VRActiveActionSet_t * 1)()
        self.sets[0].ulActionSet = self.api.getActionSetHandle("/actions/x1")

    def update(self):
        self.api.updateActionState(self.sets)

    def read(self, hand):
        result = {}
        for button in BUTTONS:
            try:
                data = self.api.getDigitalActionData(self.actions[button], self.sources[hand])
                result[button] = {
                    "available": bool(data.bActive),
                    "pressed": bool(data.bState) if data.bActive else None,
                    "active_origin": int(data.activeOrigin),
                }
            except self.vr.error_code.InputError_NoData as exc:
                result[button] = {"available": False, "pressed": None, "error": describe_error(exc)}
        return result

    def bindings(self):
        result = {}
        for name, action in self.actions.items():
            buffer = (self.vr.InputBindingInfo_t * self.vr.k_unMaxActionOriginCount)()
            try:
                info, count = self.api.getActionBindingInfo(action, buffer)
            except self.vr.error_code.InputError_NoData:
                result[name] = []
                continue
            result[name] = [
                {
                    "device_path": item.rchDevicePathName.decode("utf-8", errors="replace"),
                    "input_path": item.rchInputPathName.decode("utf-8", errors="replace"),
                    "mode": item.rchModeName.decode("utf-8", errors="replace"),
                    "slot": item.rchSlotName.decode("utf-8", errors="replace"),
                }
                for item in info[:count]
            ]
        return result

    def haptic_bound(self, hand):
        return any(
            item["device_path"] == f"/user/hand/{hand}" for item in self.bindings()["haptic"]
        )

    def pulse(self, hand):
        self.api.triggerHapticVibrationAction(
            self.actions["haptic"], 0, 0.04, 80, 0.25, self.sources[hand]
        )


def read_hands(vr, system, poses, action_input=None):
    hands = {}
    roles = (vr.TrackedControllerRole_LeftHand, vr.TrackedControllerRole_RightHand)
    masks = {"grip": 1 << vr.k_EButton_Grip, "trigger": 1 << vr.k_EButton_SteamVR_Trigger}
    for hand, role in zip(HANDS, roles):
        index = system.getTrackedDeviceIndexForControllerRole(role)
        current = {
            "index": None,
            "connected": False,
            "pose_valid": False,
            "tracking_result": None,
            "position_m": None,
            "device_to_standing": None,
            "inputs": {button: {"available": False, "pressed": None} for button in BUTTONS},
        }
        if index != vr.k_unTrackedDeviceIndexInvalid and 0 <= index < len(poses):
            current.update(pose_data(poses[index]), index=int(index))
            if current["connected"]:
                if action_input is not None:
                    current["inputs"] = action_input.read(hand)
                else:
                    ok, state = system.getControllerState(index)
                    current["legacy_state_ok"] = bool(ok)
                    current["inputs"] = {
                        button: {
                            "available": bool(ok),
                            "pressed": bool(state.ulButtonPressed & mask) if ok else None,
                        }
                        for button, mask in masks.items()
                    }
        hands[hand] = current
    return hands


def controller_metadata(vr, system, index):
    result = {"index": index}
    for name, prop in (
        ("controller_type", vr.Prop_ControllerType_String),
        ("model", vr.Prop_ModelNumber_String),
        ("tracking_system", vr.Prop_TrackingSystemName_String),
    ):
        try:
            result[name] = system.getStringTrackedDeviceProperty(index, prop)
        except Exception as exc:
            result[name] = {"unavailable": describe_error(exc)}
    try:
        supported = system.getUint64TrackedDeviceProperty(index, vr.Prop_SupportedButtons_Uint64)
        result["legacy_supported_buttons"] = supported
        result["legacy_grip_supported"] = bool(supported & (1 << vr.k_EButton_Grip))
    except Exception as exc:
        result["legacy_supported_buttons"] = {"unavailable": describe_error(exc)}
    return result


def pulse_if_due(evidence, mode, elapsed, hands, attempted, system, action_input):
    selected = HANDS if mode == "both" else (mode,) if mode in HANDS else ()
    for hand in selected:
        due = 5.0 + (2.0 if hand == "right" and mode == "both" else 0.0)
        if hand in attempted or elapsed < due:
            continue
        attempted.add(hand)
        result = {
            "hand": hand,
            "physical_sensation": "unobserved",
            "index": hands[hand]["index"],
            "api_call_made": False,
        }
        if not hands[hand]["connected"]:
            result["api_outcome"] = "skipped_disconnected"
        else:
            try:
                if action_input is None:
                    result["api"] = "IVRSystem.triggerHapticPulse"
                    result["parameters"] = {"axis": 0, "duration_microseconds": 1000}
                    result["api_call_made"] = True
                    system.triggerHapticPulse(hands[hand]["index"], 0, 1000)
                    result["api_outcome"] = "void_call_returned_no_delivery_acknowledgement"
                else:
                    result["api"] = "IVRInput.triggerHapticVibrationAction"
                    result["parameters"] = {
                        "duration_s": 0.04,
                        "frequency_hz": 80,
                        "amplitude": 0.25,
                    }
                    if action_input.haptic_bound(hand):
                        result["api_call_made"] = True
                        action_input.pulse(hand)
                        result["api_outcome"] = "returned_no_error_not_physical_proof"
                    else:
                        result["api_outcome"] = "skipped_unbound"
            except Exception as exc:
                result["api_outcome"] = "error"
                result["error"] = describe_error(exc)
        evidence.summary["haptic_calls"].append(evidence.event("haptic", **result))
        print(json.dumps(result), flush=True)


def run(args):
    evidence = Evidence(args.output_root, args.input, args.haptics)
    print(f"Private X1 run: {evidence.path}", flush=True)
    initialized = False
    applications = None
    registered = False
    exit_code = 0
    try:
        import openvr as vr

        evidence.summary["openvr_package_version"] = vr.__version__
        evidence.summary["runtime_installed"] = bool(vr.isRuntimeInstalled())
        try:
            system = vr.init(vr.VRApplication_Background)
            initialized = True
        except Exception as exc:
            error = describe_error(exc)
            evidence.summary.update(status="openvr_unavailable", error=error)
            evidence.event("openvr_unavailable", error=error)
            print(f"OpenVR unavailable (background mode never starts SteamVR): {error}", flush=True)
            return 2
        evidence.event("openvr_initialized", application_type="Background")
        action_input = None
        if args.input == "steamvr":
            manifest = (
                args.action_manifest.expanduser().resolve()
                if args.action_manifest
                else evidence.path / "actions.json"
            )
            if not args.action_manifest:
                shutil.copyfile(Path(__file__).with_name("x1_actions.json"), manifest)
            app_manifest = evidence.path / "application.vrmanifest"
            write_json(
                app_manifest,
                {
                    "source": "builtin",
                    "applications": [
                        {
                            "app_key": APP_KEY,
                            "launch_type": "binary",
                            "binary_path_windows": sys.executable,
                            "binary_path_linux": sys.executable,
                            "action_manifest_path": str(manifest),
                            "strings": {"en_us": {"name": "PuriPuly X1 controller probe"}},
                        }
                    ],
                },
            )
            applications = vr.VRApplications()
            applications.addApplicationManifest(str(app_manifest), True)
            registered = True
            applications.identifyApplication(os.getpid(), APP_KEY)
            action_input = SteamVRInput(vr, manifest)
            evidence.summary["action_manifest"] = str(manifest)
            evidence.event(
                "action_manifest_loaded",
                path=str(manifest),
                binding_activation="not_yet_established",
            )
            print(f"Action manifest: {manifest}\n{BINDING_HELP}", flush=True)
        print("No photo, camera control, network or UI. Haptics: " + args.haptics, flush=True)
        if args.haptics != "none":
            print(
                "One bounded haptic attempt per selected hand at +5s (right +7s when both). Ctrl+C stops.",
                flush=True,
            )
        poses = (vr.TrackedDevicePose_t * vr.k_unMaxTrackedDeviceCount)()
        previous_devices = {}
        previous_bindings = None
        next_binding_check = 0.0
        attempted = set()
        started = time.perf_counter()
        while (elapsed := time.perf_counter() - started) < args.seconds:
            if action_input is not None:
                action_input.update()
                if elapsed >= next_binding_check:
                    bindings = action_input.bindings()
                    if bindings != previous_bindings:
                        evidence.event("bindings", actions=bindings)
                        previous_bindings = bindings
                    next_binding_check = elapsed + 1.0
            system.getDeviceToAbsoluteTrackingPose(vr.TrackingUniverseStanding, 0, poses)
            hands = read_hands(vr, system, poses, action_input)
            for hand, current in hands.items():
                identity = (current["index"], current["connected"])
                if previous_devices.get(hand) != identity:
                    metadata = (
                        controller_metadata(vr, system, current["index"])
                        if current["connected"]
                        else None
                    )
                    evidence.event("controller", hand=hand, identity=identity, metadata=metadata)
                    previous_devices[hand] = identity
            evidence.sample(hands)
            pulse_if_due(evidence, args.haptics, elapsed, hands, attempted, system, action_input)
            time.sleep(min(1 / 90, max(0, args.seconds - (time.perf_counter() - started))))
        evidence.summary["status"] = "capture_completed_not_physical_pass"
    except KeyboardInterrupt:
        evidence.summary["status"] = "interrupted"
        exit_code = 130
    except Exception as exc:
        evidence.summary.update(status="error", error=describe_error(exc))
        evidence.event("error", error=evidence.summary["error"])
        print(evidence.summary["error"], file=sys.stderr, flush=True)
        exit_code = 1
    finally:
        cleanup = []
        if registered:
            try:
                applications.removeApplicationManifest(str(app_manifest))
            except Exception as exc:
                cleanup.append(f"remove temporary application manifest: {describe_error(exc)}")
        if initialized:
            try:
                vr.shutdown()
            except Exception as exc:
                cleanup.append(f"OpenVR shutdown: {describe_error(exc)}")
        evidence.summary["cleanup_errors"] = cleanup
        if cleanup:
            evidence.summary["status"] = "cleanup_error"
            exit_code = 1
        evidence.close()
        print(f"Summary: {evidence.path / 'summary.json'}", flush=True)
    return exit_code


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="X1 only: observe controller grip/trigger and fresh paired poses alongside human-operated VRChat. Never starts SteamVR."
    )
    parser.add_argument(
        "--input",
        choices=("legacy", "steamvr"),
        default="legacy",
        help="Try legacy first; explicit SteamVR Input fallback needs active per-hand bindings.",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=30,
        help="Capture duration, >0 to 600 seconds (default: 30).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(tempfile.gettempdir()) / "puripuly-x1",
        help="Private local output root; every run gets a unique child directory outside the repo by default.",
    )
    parser.add_argument(
        "--action-manifest",
        type=Path,
        help="SteamVR Input manifest to reuse after exporting a hardware-specific binding; default copies the unbound X1 manifest into this private run.",
    )
    parser.add_argument(
        "--haptics",
        choices=("none", "left", "right", "both"),
        default="none",
        help="Explicit opt-in: one bounded attempt per selected controller, after 5s; both staggers right to 7s. API return is not sensation proof.",
    )
    args = parser.parse_args(argv)
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 600:
        parser.error("--seconds must be finite, greater than zero and at most 600")
    if args.action_manifest and args.input != "steamvr":
        parser.error("--action-manifest requires --input steamvr")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
