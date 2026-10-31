from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

try:
    import openvr
except ModuleNotFoundError as exc:
    if exc.name != "openvr":
        raise
    raise unittest.SkipTest(
        "X1 uses optional standalone openvr==2.12.1401; run with uv run --no-project --with "
        "openvr==2.12.1401 as documented in scripts/probes/README.md"
    ) from exc

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/probes/x1_controller_probe.py"
spec = importlib.util.spec_from_file_location("x1_controller_probe", SCRIPT)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def make_pose(x, *, valid=True, connected=True):
    pose = openvr.TrackedDevicePose_t()
    pose.bDeviceIsConnected = connected
    pose.bPoseIsValid = valid
    pose.eTrackingResult = openvr.TrackingResult_Running_OK
    for r in range(3):
        pose.mDeviceToAbsoluteTracking[r][r] = 1
    pose.mDeviceToAbsoluteTracking[0][3] = x
    return pose


class SyntheticSystem:
    def __init__(self):
        self.roles = {
            openvr.TrackedControllerRole_LeftHand: 1,
            openvr.TrackedControllerRole_RightHand: 2,
        }
        self.states = {1: (True, 0), 2: (True, 0)}
        self.poses = (openvr.TrackedDevicePose_t * openvr.k_unMaxTrackedDeviceCount)()
        self.poses[1] = make_pose(0)
        self.poses[2] = make_pose(0.1)
        self.triggerHapticPulse = Mock()

    def getTrackedDeviceIndexForControllerRole(self, role):
        return self.roles[role]

    def getControllerState(self, index):
        ok, buttons = self.states[index]
        return ok, SimpleNamespace(ulButtonPressed=buttons)

    def getDeviceToAbsoluteTrackingPose(self, origin, prediction, target):
        for index, pose in enumerate(self.poses):
            target[index] = pose

    def getStringTrackedDeviceProperty(self, index, prop):
        return "synthetic-controller"

    def getUint64TrackedDeviceProperty(self, index, prop):
        return 1 << openvr.k_EButton_Grip


class SyntheticInput:
    def __init__(self):
        self.manifests = []
        self.handles = {"grip": 10, "trigger": 11, "haptic": 12}
        self.sources = {"/user/hand/left": 20, "/user/hand/right": 21}
        self.data = {}
        self.reads = []
        self.bound = []
        self.updated = False
        self.triggerHapticVibrationAction = Mock()
        self.no_data = False

    def setActionManifestPath(self, path):
        self.manifests.append(path)

    def getActionHandle(self, path):
        return self.handles[path.rsplit("/", 1)[1]]

    def getInputSourceHandle(self, path):
        return self.sources[path]

    def getActionSetHandle(self, path):
        return 30

    def updateActionState(self, sets):
        self.updated = True
        self.active_set = sets[0]

    def getDigitalActionData(self, action, source):
        self.reads.append((action, source))
        if self.no_data:
            raise openvr.error_code.InputError_NoData(openvr.VRInputError_NoData)
        active, state = self.data.get((action, source), (False, False))
        return SimpleNamespace(bActive=active, bState=state, activeOrigin=source)

    def getActionBindingInfo(self, action, buffer):
        if self.no_data:
            raise openvr.error_code.InputError_NoData(openvr.VRInputError_NoData)
        for item, path in zip(buffer, self.bound):
            item.rchDevicePathName = path.encode()
            item.rchInputPathName = b"/output/haptic" if action == 12 else b"/input/grip"
            item.rchModeName = b"button"
            item.rchSlotName = b"click"
        return buffer, len(self.bound)


class SyntheticRuntime:
    def __init__(self):
        self.system = SyntheticSystem()
        self.input = SyntheticInput()
        self.applications = SimpleNamespace(
            addApplicationManifest=Mock(),
            identifyApplication=Mock(),
            removeApplicationManifest=Mock(),
        )
        self.init = Mock(return_value=self.system)
        self.shutdown = Mock()

    def __getattr__(self, name):
        return getattr(openvr, name)

    def isRuntimeInstalled(self):
        return True

    def VRApplications(self):
        return self.applications

    def VRInput(self):
        return self.input


class X1BehaviorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.system = SyntheticSystem()

    def evidence(self, mode="legacy", haptics="none"):
        evidence = probe.Evidence(self.root, mode, haptics)
        self.addCleanup(evidence.close)
        return evidence

    def hands(self):
        return probe.read_hands(openvr, self.system, self.system.poses)

    def events(self, evidence, kind):
        return [
            record
            for line in (evidence.path / "events.jsonl").read_text().splitlines()
            if (record := json.loads(line))["event"] == kind
        ]

    def test_button_event_pairs_both_current_poses_not_previous_right_pose(self):
        evidence = self.evidence()
        evidence.sample(self.hands())
        self.system.poses[1] = make_pose(0.2)
        self.system.poses[2] = make_pose(0.5)
        self.system.states[1] = (True, 1 << openvr.k_EButton_Grip)
        evidence.sample(self.hands())
        event = self.events(evidence, "button")[0]
        self.assertEqual((event["hand"], event["button"], event["pressed"]), ("left", "grip", True))
        self.assertAlmostEqual(event["hands_distance_m"], 0.3)
        self.assertEqual(event["sequence"], 1)

    def test_tracking_loss_clears_pose_distance_and_recovers_fresh_pair(self):
        evidence = self.evidence()
        evidence.sample(self.hands())
        self.system.poses[2] = make_pose(9, valid=False)
        self.system.states[1] = (True, 1 << openvr.k_EButton_Grip)
        hands = self.hands()
        self.assertIsNone(hands["right"]["position_m"])
        self.assertIsNone(hands["right"]["device_to_standing"])
        self.assertIsNone(evidence.sample(hands))
        self.assertIsNone(self.events(evidence, "button")[0]["hands_distance_m"])
        self.system.poses[2] = make_pose(0.7)
        self.assertAlmostEqual(evidence.sample(self.hands()), 0.7)
        self.assertEqual(evidence.summary["paired_pose_samples"], 2)

    def test_disconnected_or_missing_role_cannot_contribute_a_stale_pose(self):
        self.system.poses[2] = make_pose(9, connected=False)
        self.assertIsNone(probe.hands_distance(self.hands()))
        self.system.roles[openvr.TrackedControllerRole_RightHand] = (
            openvr.k_unTrackedDeviceIndexInvalid
        )
        hands = self.hands()
        self.assertIsNone(hands["right"]["index"])
        self.assertFalse(hands["right"]["inputs"]["grip"]["available"])
        self.assertIsNone(probe.hands_distance(hands))

    def test_nonfinite_pose_and_duplicate_role_pair_are_not_distance_evidence(self):
        self.system.poses[2] = make_pose(float("nan"))
        self.assertIsNone(probe.hands_distance(self.hands()))
        self.system.roles[openvr.TrackedControllerRole_RightHand] = 1
        self.assertIsNone(probe.hands_distance(self.hands()))

    def test_input_loss_recovery_does_not_invent_edges_and_hands_count_separately(self):
        evidence = self.evidence()
        evidence.sample(self.hands())
        self.system.states[1] = (False, 0)
        self.system.states[2] = (True, 1 << openvr.k_EButton_Grip)
        evidence.sample(self.hands())
        self.system.states[1] = (True, 1 << openvr.k_EButton_Grip)
        evidence.sample(self.hands())
        self.system.states[1] = (True, 0)
        evidence.sample(self.hands())
        left, right = (evidence.summary["hands"][hand] for hand in probe.HANDS)
        self.assertEqual((left["grip_presses"], left["grip_releases"]), (0, 1))
        self.assertEqual(right["grip_presses"], 1)
        self.assertEqual((left["grip_available_samples"], right["grip_available_samples"]), (3, 4))

    def test_controller_role_reassignment_does_not_invent_button_edge(self):
        evidence = self.evidence()
        evidence.sample(self.hands())
        self.system.roles[openvr.TrackedControllerRole_LeftHand] = 3
        self.system.poses[3] = make_pose(0.4)
        self.system.states[3] = (True, 1 << openvr.k_EButton_Grip)
        evidence.sample(self.hands())
        self.assertEqual(self.events(evidence, "button"), [])

    def test_action_input_is_restricted_per_hand_and_inactive_is_not_false_release(self):
        runtime = SyntheticRuntime()
        actions = probe.SteamVRInput(runtime, self.root / "actions.json")
        runtime.input.data[(10, 20)] = (True, True)
        runtime.input.data[(10, 21)] = (False, True)
        actions.update()
        hands = probe.read_hands(runtime, self.system, self.system.poses, actions)
        self.assertTrue(hands["left"]["inputs"]["grip"]["pressed"])
        self.assertIsNone(hands["right"]["inputs"]["grip"]["pressed"])
        self.assertEqual(runtime.input.reads, [(10, 20), (11, 20), (10, 21), (11, 21)])
        self.assertTrue(runtime.input.updated)
        self.assertEqual(runtime.input.active_set.nPriority, 0)

    def test_no_data_stays_unavailable_for_manual_binding_then_recovers(self):
        runtime = SyntheticRuntime()
        actions = probe.SteamVRInput(runtime, self.root / "actions.json")
        runtime.input.no_data = True
        self.assertFalse(actions.read("left")["grip"]["available"])
        self.assertFalse(actions.haptic_bound("left"))
        runtime.input.no_data = False
        runtime.input.data[(10, 20)] = (True, True)
        runtime.input.bound = ["/user/hand/left"]
        self.assertTrue(actions.read("left")["grip"]["pressed"])
        self.assertTrue(actions.haptic_bound("left"))
        self.assertFalse(actions.haptic_bound("right"))

    def test_haptics_default_off_delay_once_per_hand_and_no_sensation_claim(self):
        evidence = self.evidence(haptics="both")
        attempted = set()
        hands = self.hands()
        for mode, elapsed in (
            ("none", 100),
            ("both", 4.9),
            ("both", 5),
            ("both", 6),
            ("both", 7),
            ("both", 50),
        ):
            probe.pulse_if_due(evidence, mode, elapsed, hands, attempted, self.system, None)
        self.assertEqual(
            self.system.triggerHapticPulse.call_args_list,
            [unittest.mock.call(1, 0, 1000), unittest.mock.call(2, 0, 1000)],
        )
        self.assertEqual(len(evidence.summary["haptic_calls"]), 2)
        self.assertTrue(
            all(
                item["physical_sensation"] == "unobserved"
                for item in evidence.summary["haptic_calls"]
            )
        )
        self.assertTrue(all(item["api_call_made"] for item in evidence.summary["haptic_calls"]))

    def test_action_haptic_requires_matching_binding_and_is_bounded(self):
        evidence = self.evidence("steamvr", "both")
        runtime = SyntheticRuntime()
        actions = probe.SteamVRInput(runtime, self.root / "actions.json")
        runtime.input.bound = ["/user/hand/left"]
        probe.pulse_if_due(evidence, "both", 10, self.hands(), set(), self.system, actions)
        runtime.input.triggerHapticVibrationAction.assert_called_once_with(
            12, 0, 0.04, 80, 0.25, 20
        )
        self.assertEqual(evidence.summary["haptic_calls"][1]["api_outcome"], "skipped_unbound")
        self.assertFalse(evidence.summary["haptic_calls"][1]["api_call_made"])
        self.system.triggerHapticPulse.assert_not_called()

    def test_disconnected_haptic_and_api_error_do_not_claim_delivery_or_retry(self):
        evidence = self.evidence(haptics="both")
        self.system.poses[2] = make_pose(9, connected=False)
        self.system.triggerHapticPulse.side_effect = RuntimeError("synthetic failure")
        attempted = set()
        for _ in range(2):
            probe.pulse_if_due(evidence, "both", 10, self.hands(), attempted, self.system, None)
        self.assertEqual(self.system.triggerHapticPulse.call_count, 1)
        self.assertEqual(
            [item["api_outcome"] for item in evidence.summary["haptic_calls"]],
            ["error", "skipped_disconnected"],
        )

    def test_unique_runs_remain_outside_checkout_and_do_not_mark_physical_pass(self):
        first, second = self.evidence(), self.evidence()
        self.assertNotEqual(first.path, second.path)
        self.assertEqual(first.path.parent, self.root)
        self.assertTrue(
            all(
                value.startswith("unperformed")
                for value in first.summary["physical_observations"].values()
            )
        )

    def run_synthetic(self, runtime, mode):
        clock = iter(index / 1000 for index in range(1000))
        with (
            patch.dict("sys.modules", {"openvr": runtime}),
            patch.object(probe.time, "perf_counter", side_effect=lambda: next(clock)),
            patch.object(probe.time, "sleep"),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            result = probe.main(
                ["--input", mode, "--seconds", "0.02", "--output-root", str(self.root)]
            )
        run = next(self.root.iterdir())
        return result, run, json.loads((run / "summary.json").read_text())

    def test_capture_lifecycle_collects_real_struct_samples_and_cleans_temporary_registration(self):
        runtime = SyntheticRuntime()
        result, run, summary = self.run_synthetic(runtime, "steamvr")
        self.assertEqual(result, 0)
        self.assertGreater(summary["samples"], 0)
        self.assertEqual(summary["samples"], summary["paired_pose_samples"])
        self.assertEqual(summary["status"], "capture_completed_not_physical_pass")
        runtime.init.assert_called_once_with(openvr.VRApplication_Background)
        runtime.shutdown.assert_called_once()
        runtime.applications.addApplicationManifest.assert_called_once_with(
            str(run / "application.vrmanifest"), True
        )
        runtime.applications.removeApplicationManifest.assert_called_once_with(
            str(run / "application.vrmanifest")
        )
        runtime.applications.identifyApplication.assert_called_once()
        runtime.input.triggerHapticVibrationAction.assert_not_called()
        manifest = json.loads((run / "actions.json").read_text())
        self.assertEqual(manifest["default_bindings"], [])

    def test_setup_error_after_registration_cleans_session_and_retains_error(self):
        runtime = SyntheticRuntime()
        runtime.input.setActionManifestPath = Mock(
            side_effect=RuntimeError("synthetic setup failure")
        )
        result, run, summary = self.run_synthetic(runtime, "steamvr")
        self.assertEqual(result, 1)
        self.assertIn("synthetic setup failure", summary["error"])
        runtime.shutdown.assert_called_once()
        runtime.applications.removeApplicationManifest.assert_called_once()

    def test_poll_error_cleans_session_and_preserves_failure(self):
        runtime = SyntheticRuntime()
        runtime.system.getDeviceToAbsoluteTrackingPose = Mock(
            side_effect=RuntimeError("synthetic tracking failure")
        )
        result, run, summary = self.run_synthetic(runtime, "legacy")
        self.assertEqual(result, 1)
        self.assertIn("synthetic tracking failure", summary["error"])
        runtime.shutdown.assert_called_once()
        runtime.system.triggerHapticPulse.assert_not_called()

    def test_interrupt_cleans_session_and_reports_interruption(self):
        runtime = SyntheticRuntime()
        runtime.system.getDeviceToAbsoluteTrackingPose = Mock(side_effect=KeyboardInterrupt())
        result, run, summary = self.run_synthetic(runtime, "legacy")
        self.assertEqual(result, 130)
        self.assertEqual(summary["status"], "interrupted")
        runtime.shutdown.assert_called_once()

    def test_invalid_duration_and_manifest_mode_are_rejected_before_runtime(self):
        for arguments in (
            ("--seconds", "nan"),
            ("--seconds", "601"),
            ("--seconds", "0"),
            ("--action-manifest", "x.json"),
        ):
            with (
                self.subTest(arguments=arguments),
                contextlib.redirect_stderr(io.StringIO()),
                patch.object(probe, "run") as run,
            ):
                with self.assertRaises(SystemExit) as error:
                    probe.main(arguments)
                self.assertEqual(error.exception.code, 2)
                run.assert_not_called()

    def test_no_runtime_records_typed_error_without_binding_or_haptic_side_effects(self):
        runtime = SyntheticRuntime()
        runtime.init.side_effect = openvr.error_code.InitError_Init_NoServerForBackgroundApp(121)
        result, run, summary = self.run_synthetic(runtime, "steamvr")
        self.assertEqual(result, 2)
        self.assertEqual(summary["status"], "openvr_unavailable")
        self.assertIn("InitError_Init_NoServerForBackgroundApp (code=121)", summary["error"])
        self.assertEqual(summary["samples"], 0)
        runtime.applications.addApplicationManifest.assert_not_called()
        runtime.system.triggerHapticPulse.assert_not_called()
        runtime.input.triggerHapticVibrationAction.assert_not_called()

    def test_cleanup_error_is_preserved_and_shutdown_still_occurs(self):
        runtime = SyntheticRuntime()
        runtime.applications.removeApplicationManifest.side_effect = RuntimeError(
            "synthetic unregister failure"
        )
        result, run, summary = self.run_synthetic(runtime, "steamvr")
        self.assertEqual(result, 1)
        self.assertEqual(summary["status"], "cleanup_error")
        self.assertIn("synthetic unregister failure", summary["cleanup_errors"][0])
        runtime.shutdown.assert_called_once()


if __name__ == "__main__":
    unittest.main()
