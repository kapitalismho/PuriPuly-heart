# Temporary VR photo experiments

Temporary native-interaction and standalone operator tools for [#341](https://github.com/kapitalismho/PuriPuly-heart/issues/341), under [#338](https://github.com/kapitalismho/PuriPuly-heart/issues/338). The optional X1 diagnostic reuses the recovered `controller-watch` approach (`probe.py`, `cmd_controller_watch`; `HANDOFF.md`, section 10 step 4), without importing its camera controls or reviving excluded experiments.

## Current owner decisions

The owner approved shared input with VRChat and proceeding directly to temporary interaction implementation. Input capture/suppression and a no-interference gate are not required. The standalone X1 probe below is an optional input-mapping diagnostic, not a prerequisite for implementing or trying the temporary VR flow. Physical input availability and felt haptics still need observation; they are not established by code inspection.

Feedback routing is settled: only the valid 12 cm pull trigger gives a short pulse on the left controller. The first drawn translated block gives one pulse on the right controller per request attempt, using the existing application-authorized feedback grant. The owner explicitly selected first-block display, not whole-photo completion. Renderer recovery and reopening do not replay this notification. There is no grab-start, shutter, release, or additional whole-photo-completion haptic. The owner's latest decision removes the previously approved grab-start pulse; grab highlighting and interaction remain unchanged.

The source precedents use shared legacy input, not exclusive game-input ownership: [VRCT](https://github.com/misyaguziya/VRCT/blob/79419a1f2ce78d38993b0f33e273155667eb0b6a/src-python/models/overlay/overlay.py#L53-L55) explicitly notes simultaneous VRChat delivery; [VRCLT](https://github.com/VRCLT/VRCLT/blob/775fab0d460feb3dd034c5902523cdf1fc510f13/vrclt/vr/wrist_ui.py#L1-L11) keeps game input active. Their ray-targeted drag implementations do not establish physical success of the near-wrist pull on the user's hardware.

These decisions replace the earlier requirement to finish a separate X1 interference experiment before implementation. Physical X3 legibility/interaction observations, X8 language qualification, and assembled-flow evidence remain pending. Final UI remains #342; no Python product OCR UI is authorized.

### Approved photo readability changes

After a local comparison of the original 23 photos and saved translations, the owner selected both the bounded layout improvement and erase-mask expansion. These decisions supersede only the minimum font size, horizontal fallback selection and erase dilation recorded in [#340](https://github.com/kapitalismho/PuriPuly-heart/issues/340); the model, prompt, source cap, source geometry, other image rules and deferred distribution obligations remain unchanged.

- Minimum font-em size is `max(8*s, 0.015*H)`, where `H` is the actual working-photo content height and `s = H/1440`. This replaces 1.1% with 1.5%: 16.2 px at 1080 high and 21.6 px at 1440 high. Font-em size is not glyph ink height; the existing 0.75-size vertical two-character forms remain unchanged.
- Normal strict fitting is unchanged. Only after no collision-free strict fit, for an unrotated horizontal source whose existing minimum-size fallback candidates all overflow, add a horizontal candidate in the union of measured ink and model box with half the minimum font size of margin per side, clamped to the photo. Preserve existing candidates and earlier placements. Ranking still prioritizes prior overlap, overflow and direction-adjusted font size; only this fallback path breaks ties in favor of horizontal wrapping, then shorter vertical ink span. It can still overflow or overlap; it is not unrestricted region growth.
- The ellipse erase-mask radius is now `max(1, round_ties_even(0.05*g)) + max(1, ceil(s))` working pixels in both preparation and erase refresh. This adds one pixel at the tested 719/1068/1080 heights and at 1440; a 2160-high working image adds two. Model-core inference, model box, ROI, color sampling and the plate decision formula are unchanged. A larger erase mask can reduce the leftover heuristic but cannot improve sparse-core coverage by itself.

Verification used 23 photos × four experimental conditions at the default 1920×1080 bounds, followed by all 23 jobs with the ordinary production executable. All 92 ordinary original/erase/composite/original-view frames matched the selected combined experiment pixel-for-pixel; common stable metadata matched for all 229 blocks. There were no new provider requests or input overwrites. In that cohort, layout-region overflows fell from 10 to 4 and image-boundary ink crossings from 1 to 0; significant prior-text overlaps remained 9 and plates remained 5. The erase footprint increased about 10%, not a measured 10% increase in background damage. The owner accepted possible wider Telea smears and the remaining overlap/overflow limits. These offline observations do not establish headset readability. The executable does not contain experiment flags or diagnostic-only mask/layout exports.

## X3: no-provider native photo preview

`x3_photo_preview.py` launches the actual native photo role and uses its authenticated loopback bridge. It offers an explicitly selected existing PNG, without provider calls, credentials, OCR or pixel processing in Python. Native decodes and draws the image. A recorded translated composite is shown as a **base-image preview with zero translation blocks**; it cannot produce a genuine first-translated-block notification. Test right-hand notification through the actual block path, not by pretending this prerecorded image is a newly translated block.

Native's optional manifest `interaction` object exposes `grab_m`, `icon_width_m`, `photo_width_m`, `facing_show_deg`, `forward_show_deg`, `facing_hide_deg`, `forward_hide_deg`, `dwell_ms`, `wrist_transform` and `photo_transform`. The transforms are rigid 3×4 controller-relative matrices in metres. The owner selected a default held-photo width of **37.5 cm**, matching the serialized VRChat Print prefab's width only. Preserve the source aspect ratio without adding card borders or letterboxing: a 16:9 photo is 37.5 × 21.09375 cm. An explicit `photo_width_m` still overrides the default for temporary tuning. Grab range remains 7 cm and icon width remains 3 cm. Held-photo appearance/release remains linear over 150 ms; icon return remains 225 ms ease-out.

The owner selected VRCT's wrist visibility and input timing policy, based on its pinned [constants](https://github.com/misyaguziya/VRCT/blob/61e153aca43c011d65afb05357c9ad4b9120915f/src-python/models/overlay/overlay.py#L77-L89) and [visibility logic](https://github.com/misyaguziya/VRCT/blob/61e153aca43c011d65afb05357c9ad4b9120915f/src-python/models/overlay/overlay.py#L2232-L2286):

| Gate or transition | Default |
| --- | --- |
| Head-forward angle toward wrist | Show within 40°; retain within 45° |
| Icon face-normal angle toward head | Show within 45°; retain within 55° |
| Show dwell (`dwell_ms`) | 50 ms; the existing temporary override remains supported |
| Hide dwell | 300 ms |
| Icon fade | 100 ms linear; per-frame delta capped at 100 ms |
| New grab | Fully opaque and at least 200 ms since the shown transition, with a fresh grip press |

Both angle conditions must hold; this uses HMD orientation, not eye tracking. Interrupted show dwell resets. Reentering both retain cones cancels pending hide. An active grab or icon return keeps the icon visible despite looking away. Tracking/input loss, controller-role changes and photo invalidation still cancel immediately: VRCT's stale-pose retention, ray hover, intro and always-visible mode are not imported. A grip pressed before input readiness does not activate later while held; release and press again. See the [interaction owner](../../native/overlay/src/photo/interaction.rs). Physical wrist/hand alignment, visibility and comfort remain to be observed rather than inferred from software checks.

Size provenance: read-only inspection of VRChat `2026.4.1p1-1937-299983b13b-Release` (Steam build `26309255`) found `Print Prefab` in `VRChat_Data/resources.assets` (GameObject 26792, root Transform 49493). Its root scale is 0.75; child `Mesh` Transform 47927 has X/Z scale 0.05/0.03515625 and references the built-in `Plane` mesh 10209, whose X/Z bounds are 10/10 units. The serialized outer card is therefore nominally 0.375 × 0.263671875 metres. `resources.assets` SHA-256: `a7db7942724288a4afb09f9c16777cdf43fff56537af06a5d304d40b773b6c1b`. The owner selected width-only matching, not this outer card's aspect ratio. Runtime spawn/avatar scaling and native portrait behavior were not observed; this is an asset-derived target, not proof of an in-headset size match. The [official Print documentation](https://wiki.vrchat.com/wiki/Prints) supplies pixel margins, not physical dimensions.

Compared with the former 50 cm candidate, the card's geometry is 25% smaller in each physical dimension. At the same viewing distance, unchanged text sizes subtend smaller angles despite higher pixels per metre. The separately approved 1.5% minimum font size approximately compensates for this shrinkage for text at the former 1.1% floor; it does not enlarge all text. Moving the card to 75% of the former eye-to-card distance restores its angular size. The 1080p default, source aspect ratio and hand-relative placement are unchanged; headset legibility remains an X3 observation.

Build and assemble the executable using the [native instructions](../../native/overlay/README.md#build-and-verification). Start SteamVR and VRChat manually before a physical run. From the repository root:

```powershell
$overlay = Join-Path $env:TEMP 'puripuly-overlay-tests/release/PuriPulyHeartOverlay.exe'
$renderRoot = 'C:\path\to\retained\native_photo340_20261031'
$image = Join-Path $renderRoot 'native_outline_1080/u01_photo/composite.png'
python scripts/probes/x3_photo_preview.py --executable $overlay --image $image --offer-delay 5 --seconds 180
```

Use a Python environment with the project's existing dependencies. No new project dependency is required. The launcher prints the exact executable identity and a private evidence directory. It refuses OpenVR launch when SteamVR is not already running.

After native readiness, this command creates one synthetic offer after five seconds. Look at the right wrist, wait for the icon's input-ready highlight with the left hand nearby, then start a fresh left grip inside the grab range. Pull the left hand 12 cm farther from the wrist anchor:

- Grab-start is silent. The valid pull requests one left-hand pulse. The icon returns to the wrist; the first pull authorizes the preview but does not leave the photo held.
- Release, then grab and pull again to open the retained preview in the left hand. It appears over 150 ms; releasing reverses alpha and width over 150 ms, then hides it. Reopening does not resend or regenerate anything.
- Releasing before the pull threshold returns the icon without opening/authorizing the photo. Moving both hands together without increasing their separation must not trigger.
- The initial opportunity lasts ten seconds; only the latest authorized preview remains available for ten minutes after the offer. Start another invocation for another synthetic opportunity, or use actual photo signals below.

For the scoped real-shot observations, append `--watch-dir '<explicit VRChat photo directory>'`. The existing watcher ignores files present at startup. Every new PNG signal offers **the same selected preview image**, not a translation of the newly captured photo. No camera control or transmission is performed. This mode lets the operator record reaction, false triggers and missed windows using the existing rendered material; it is not the assembled provider flow.

For the 1440p reference, select the corresponding `native_outline_1440/.../composite.png` and add `--reference-1440`; otherwise the native default deliberately fits within 1920×1080. Compare at the same held geometry. For original comparison, run separately with that case's `original.png` or `original_view.png`. Changing the asset creates a separate preview photo; this is not the product's original-layer toggle test. The real compositor retains its original/erase/text layer control.

Physical observations should cover wrist visibility, grip recognition, cancellation/relative pull, held-photo readability at the candidate geometry, release/reopen and felt left feedback. Shared game input is accepted. Record what was actually seen/felt; software composition or submission does not establish HMD consumption. The planned X3 sample remains about 20 real shots, not a new image-quality campaign or a new decision on the selected 1080p default.

### Quiet GPU handoff

The photo compositor flushes its existing D3D11 immediate context after each successful OpenVR texture submission, for both the wrist icon and held photo. Producer readiness before submission remains required; it does not dispatch copy work queued by `SetOverlayTexture` afterward. The post-submit flush dispatches work without a completion wait, extra redraw, or resubmission. It does not acknowledge physical headset presentation.

The missing-icon investigation isolated this boundary with a hidden native overlay. The producer icon matched its source before and after the empty-offer reset, but SteamVR image readback stayed zero for more than 1.3 seconds after submission. One subsequent producer-context flush, without resubmission, delivered the expected icon pixels. This was not a photo-detection, provider, or source-decoding prerequisite. Gaze, grab, timeout, haptic, and size policies are unchanged.

With SteamVR already running, the opt-in regression exercises initial content, replacement content, a second producer device, and restoration without further rendering after each handoff. It compares complete destination pixels, keeps its overlay hidden, and destroys its owned surface:

```powershell
cargo test --locked --manifest-path native/overlay/Cargo.toml --lib photo::runtime::tests::windows_graphics_photo341_quiet_texture_handoff_reaches_steamvr -- --ignored --exact
```

A separate synthetic one-frame probe reproduced the same missing post-submit dispatch in the current caption renderer. This supports the previously unresolved hypothesis in [#147](https://github.com/kapitalismho/PuriPuly-heart/issues/147), not a conclusion that every historical caption delay had this cause. This photo repair does not modify the caption renderer or its P05 fresh-render protection. Physical icon visibility and caption latency remain separate headset observations.

### Software-only preview smoke

```powershell
python scripts/probes/x3_photo_preview.py --executable $overlay --image $image --backend offscreen --offer-delay 0.1 --software-pull-after 0.2 --seconds 15
```

This explicit offscreen control performs a software pull, exports `preview.png`, hides, reopens, exports `reopened.png`, and shuts down the child. Add `--reference-1440` with the reference asset to preserve its resolution. Software injection is rejected in OpenVR mode. `bridge_control_mode=probe` permits the existing debug commands, while `requested_native_backend=offscreen` selects real GPU rendering; the receipt-only probe backend is not claimed to draw.

The evidence directory contains `events.jsonl`, `summary.json` and the requested exports. The token-bearing launch manifest is removed during owned cleanup. Keep outputs private when the selected asset contains personal photos. Exit success and `shutdown_acknowledged`/`native_exit_code` establish different facts from physical display.

## Optional X1 input diagnostic

X1 records left grip, both hand poses and controller haptic API calls while the user operates VRChat. The diagnostic itself adds no photo capture, photo transmission, model calls, OSC commands, product UI, or VR overlays. It does not start SteamVR or VRChat. Haptics are off unless explicitly requested.

### Legacy API capture

Run commands from the repository root in PowerShell. `uv` supplies the pinned standalone OpenVR package without changing the application's dependencies.

1. Start SteamVR manually, connect the headset and both controllers, and enter VRChat in VR in a private instance. Confirm which physical controller is assigned left/right. Leave the existing VRChat input bindings unchanged.
2. Tell the operator when ready, then start:

   ```powershell
   uv run --no-project --with openvr==2.12.1401 python scripts/probes/x1_controller_probe.py --seconds 30
   ```

3. Start with grip released. Bring the left hand near the right wrist, grip, move the left hand roughly 30 cm away, then release. Repeat five times. Also move both arms together without changing their relative distance. Exercise right grip and both triggers briefly to identify the separate inputs; triggers are diagnostic controls, not a new translation gesture.
4. Camera/object grabbing and avatar gesture changes may be noted, but shared input is accepted and their absence is not a pass condition. No separate interference trial is required.
5. When ready to feel the pulses, run:

   ```powershell
   uv run --no-project --with openvr==2.12.1401 python scripts/probes/x1_controller_probe.py --seconds 12 --haptics both
   ```

   One left attempt occurs about five seconds after polling starts; one right attempt at seven seconds. Legacy uses a 1000-microsecond pulse. Record each hand's sensation independently. An API return is not proof that a pulse was felt. `Ctrl+C` stops capture. A missing/disconnected controller produces a skipped attempt, not an automatic delayed pulse; rerun deliberately when ready.

No wrist icon is expected. Distances are between controller origins from the same current pose snapshot, **not a calibrated wrist anchor or an implemented 12 cm pull trigger**. Invalid/disconnected tracking removes the distance rather than reusing stale coordinates.

### SteamVR Input fallback: only if needed

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

### Evidence and next decision

Each invocation prints a unique local directory under `%TEMP%\puripuly-x1`. It contains `events.jsonl` and `summary.json`; action mode also creates its private manifests. These files contain local paths, controller metadata and pose traces. Do not commit or publish raw output.

Read per-hand `grip_available_samples`, `grip_presses`/`grip_releases`, `valid_pose_samples`, and `paired_pose_samples` alongside the deliberate trials. Input/role loss resets edge baselines: reconnection is not invented as a button press. The first sampled held state is a baseline, not a press transition. Haptic records distinguish API calls, returns, errors and skipped attempts.

Keep a separate human observation with:

- Run directory, headset/controller model, connection method, and confirmation that VRChat remained in VR.
- API path and, for action mode, the active binding and any force threshold.
- Left/right physical role agreement and deliberate grip trial count versus recorded transitions.
- Any useful camera/object or gesture observations; absence of shared-input effects is not required.
- Left and right haptics: felt, not felt, or not attempted.
- Tracking loss, unavailable bindings, interruptions or other missing observations.

Machine `physical_observations` remain explicitly unperformed; accompany them with the human note rather than interpreting software success as physical proof. Exit 0 means capture completed, 2 means OpenVR initialization unavailable, 130 means interrupted, and 1 means another setup/poll/cleanup error. Read `cleanup_errors` before another action-mode run.

Use the legacy input path for the temporary implementation. If the actual hardware does not expose the expected physical grip, use this diagnostic to establish the mapping and evaluate the SteamVR Input path. If neither API establishes usable grip after correct binding and deliberate trials, the original pose-only pull remains a decision path, not an implemented or qualified fallback. X3 consumes #340's existing X11 1080p/1440p renders without reopening the selected 1080p default.

### Local diagnostic verification

The standalone help and both actual no-runtime entry paths were exercised. With SteamVR stopped, background initialization returned code 121 (`InitError_Init_NoServerForBackgroundApp`), with no controller samples or haptic calls. This establishes safe unavailable-runtime behavior, not live input or binding activation.

Focused synthetic behavior/lifecycle checks use the pinned package's ctypes structures with replaced OpenVR boundaries:

```powershell
uv run --no-project --with openvr==2.12.1401 python -m unittest discover -s tests/scripts -p test_x1_controller_probe.py -v
```

All 19 tests executed and passed during preparation. They cover fresh pose pairing, invalid tracking and recovery, separate-hand input, binding availability, bounded opt-in haptics, errors and cleanup. Without the optional package, the module explicitly skips; that skip is not behavior evidence. No physical grip, binding-editor operation, game interference, HMD presentation or haptic sensation has yet been verified.
