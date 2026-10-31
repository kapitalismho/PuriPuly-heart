# X1: controller input alongside VRChat

Preparation for [#341](https://github.com/kapitalismho/PuriPuly-heart/issues/341), under [#338](https://github.com/kapitalismho/PuriPuly-heart/issues/338). This standalone probe reuses the original `controller-watch` approach from the recovered capture probe (`probe.py`, `cmd_controller_watch`; `HANDOFF.md`, section 10 step 4). It does not import that probe's camera controls or revive its excluded shutter/performance experiments.

X1 asks whether left grip and both hand poses are readable while VRChat runs, whether the interaction also grabs the camera/objects or changes avatar gestures, and whether each controller can produce a felt haptic. Budget about ten minutes of physical interaction. No photo capture, photo transmission, model calls, OSC commands, product UI, or VR overlays are added. The probe does not start SteamVR or VRChat. Haptics are off unless explicitly requested.

## First physical run: legacy API

Run commands from the repository root in PowerShell. `uv` supplies the pinned standalone OpenVR package without changing the application's dependencies.

1. Start SteamVR manually, connect the headset and both controllers, and enter VRChat in VR in a private instance. Confirm which physical controller is assigned left/right. Leave the existing VRChat input bindings unchanged.
2. Tell the operator when ready, then start:

   ```powershell
   uv run --no-project --with openvr==2.12.1401 python scripts/probes/x1_controller_probe.py --seconds 30
   ```

3. Start with grip released. Bring the left hand near the right wrist, grip, move the left hand roughly 30 cm away, then release. Repeat five times. Also move both arms together without changing their relative distance. Exercise right grip and both triggers briefly to identify the separate inputs; triggers are diagnostic controls, not a new translation gesture.
4. Watch whether gripping changes the avatar hand gesture or picks up a nearby object. Repeat with the VRChat camera manually open/nearby to check accidental camera grabbing. Run another 30-second capture if needed; keep the observations tied to each run.
5. When ready to feel the pulses, run:

   ```powershell
   uv run --no-project --with openvr==2.12.1401 python scripts/probes/x1_controller_probe.py --seconds 12 --haptics both
   ```

   One left attempt occurs about five seconds after polling starts; one right attempt at seven seconds. Legacy uses a 1000-microsecond pulse. Record each hand's sensation independently. An API return is not proof that a pulse was felt. `Ctrl+C` stops capture. A missing/disconnected controller produces a skipped attempt, not an automatic delayed pulse; rerun deliberately when ready.

No wrist icon is expected. Distances are between controller origins from the same current pose snapshot, **not a calibrated wrist anchor or an implemented 12 cm pull trigger**. Invalid/disconnected tracking removes the distance rather than reusing stale coordinates.

## SteamVR Input fallback: only if needed

No grip transitions after deliberate trials, or unavailable legacy state, is not proof that grip is impossible. Check the explicit action path next. The controller type is discovered at runtime; no guessed hardware-specific default binding is shipped.

```powershell
uv run --no-project --with openvr==2.12.1401 python scripts/probes/x1_controller_probe.py --input steamvr --seconds 180
```

While the process runs:

1. Open **SteamVR > Devices > Controller Input Binding** and select **PuriPuly X1 controller probe**. Create and activate a binding for the actual controllers and the `X1 observation only` action set.
2. Bind each hand's physical grip to **Grip**, trigger click to optional **Trigger**, and its vibration output to **Haptic**. If a force/analog input must be converted to a boolean, record the actual mapping and threshold. Substituting another physical control is not grip success.
3. Do not edit VRChat's binding or enable experimental input overrides. Enable binding-editor debugging options if needed to export/Replace Default Binding beside the printed private `actions.json`. Ensure that manifest references the exported binding. Keep both files for reuse.
4. Repeat the physical trials. Manifest loading is not binding activation: inspect the `bindings` records, per-hand availability, and actual transitions. If the editor cannot expose/activate the binding, retain the error and report that prerequisite instead of declaring the input unavailable.

Reuse the absolute private manifest path in subsequent captures:

```powershell
$manifest = 'C:\absolute\path\from\the\run\actions.json'
uv run --no-project --with openvr==2.12.1401 python scripts/probes/x1_controller_probe.py --input steamvr --action-manifest $manifest --seconds 30
uv run --no-project --with openvr==2.12.1401 python scripts/probes/x1_controller_probe.py --input steamvr --action-manifest $manifest --seconds 12 --haptics both
```

Action haptics request 40 ms at 80 Hz and amplitude 0.25, once per selected hand. Unbound outputs are skipped. This is a feasibility check, not selection of the final product pulse. The temporary application registration is removed on normal/error/interrupt exit; exported private binding files remain for the next run. Run only one X1 probe at a time.

References: [SteamVR Input](https://github.com/ValveSoftware/openvr/wiki/SteamVR-Input), [action manifests](https://github.com/ValveSoftware/openvr/wiki/Action-manifest), [OpenVR application types](https://github.com/ValveSoftware/openvr/wiki/API-Documentation).

## Evidence and next decision

Each invocation prints a unique local directory under `%TEMP%\puripuly-x1`. It contains `events.jsonl` and `summary.json`; action mode also creates its private manifests. These files contain local paths, controller metadata and pose traces. Do not commit or publish raw output.

Read per-hand `grip_available_samples`, `grip_presses`/`grip_releases`, `valid_pose_samples`, and `paired_pose_samples` alongside the deliberate trials. Input/role loss resets edge baselines: reconnection is not invented as a button press. The first sampled held state is a baseline, not a press transition. Haptic records distinguish API calls, returns, errors and skipped attempts.

Keep a separate human observation with:

- Run directory, headset/controller model, connection method, and confirmation that VRChat remained in VR.
- API path and, for action mode, the active binding and any force threshold.
- Left/right physical role agreement and deliberate grip trial count versus recorded transitions.
- Camera/object grabbing and avatar gesture changes, with camera-nearby versus camera-away context.
- Left and right haptics: felt, not felt, or not attempted.
- Tracking loss, unavailable bindings, interruptions or other missing observations.

Machine `physical_observations` remain explicitly unperformed; accompany them with the human note rather than interpreting software success as physical proof. Exit 0 means capture completed, 2 means OpenVR initialization unavailable, 130 means interrupted, and 1 means another setup/poll/cleanup error. Read `cleanup_errors` before another action-mode run.

Choose the input path only after reviewing this evidence together. If neither API establishes usable grip after correct binding and deliberate trials, record the original pose-only pull fallback as the next decision path; this probe does not implement or qualify it. X3 follows X1 and will consume #340's existing X11 1080p/1440p renders. X3, X8, assembled-flow validation and final UI are not completed by this preparation.

## Local preparation verification

The standalone help and both actual no-runtime entry paths were exercised. With SteamVR stopped, background initialization returned code 121 (`InitError_Init_NoServerForBackgroundApp`), with no controller samples or haptic calls. This establishes safe unavailable-runtime behavior, not live input or binding activation.

Focused synthetic behavior/lifecycle checks use the pinned package's ctypes structures with replaced OpenVR boundaries:

```powershell
uv run --no-project --with openvr==2.12.1401 python -m unittest discover -s tests/scripts -p test_x1_controller_probe.py -v
```

All 19 tests executed and passed during preparation. They cover fresh pose pairing, invalid tracking and recovery, separate-hand input, binding availability, bounded opt-in haptics, errors and cleanup. Without the optional package, the module explicitly skips; that skip is not behavior evidence. No physical grip, binding-editor operation, game interference, HMD presentation or haptic sensation has yet been verified.
