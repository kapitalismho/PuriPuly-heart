# Temporary photo interaction: wristwatch and held photo

This records the bounded temporary interaction work for [#341](https://github.com/kapitalismho/PuriPuly-heart/issues/341), under [#338](https://github.com/kapitalismho/PuriPuly-heart/issues/338). It is not final UI, complete physical X3 acceptance or issue closure. On November 10, after the wristwatch/edge-held candidate, the owner reported that 10 cm admission was sufficient and holding the photo was quite good. The owner requested direct wrist placement, slight card thickness, full-size emergence instead of whole-image scaling, and a release that freezes in space and quickly fades. The owner explicitly requested both opacity and clipped-reveal candidates for comparison. On November 12, after a manual wrist setup and sequential A/B run, the owner confirmed the saved wrist position, selected B (clipped reveal), and rejected the 1 mm thickness and edge planes. The final backface presentation remains open; showing the same image on both sides is one provisional candidate only.

## Selected reference and provenance

Use OpenVR's literal `handgrip` component, resolved from each actual controller's render-model identity. Do not use an OpenXR-first fallback or infer the component transform from headset branding.

The Director's `controller-anchor-decision.json` research record preserves these inspected sources:

| Controller family | Inspected source | `handgrip` | `openxr_grip` |
| --- | --- | --- | --- |
| Quest 3 / Touch Plus | Installed SteamVR build `25998080`, explicit `oculus_quest_plus_controller_left/right` model JSON | Both sides | Both sides, with a different origin and rotation |
| PICO 4 | Official PICO Connect `10.6.6` Phoenix models; installed SteamVR VRLink PICO models independently agree | Ordinary, joystick and trackpad variants | Ordinary models; absent from inspected joystick/trackpad variants |
| PICO 4 Ultra | Official PICO Connect `10.6.6` `pico_4s` models; installed explicit `pico_4_ultra` models independently agree | Both sides | Absent from the inspected native and installed Ultra definitions |

The [official PICO distribution page](https://www.picoxr.com/global/software/pico-link) supplied the Connect installer. Its SHA-256 was `4be12464d73f48fad0b7cfbed6b9400710f18ad85b50271bb3e8e58c8ffb6fba`; its driver marker was commit `8c81412321e8340e6f990677b57fbe8162c4cbde`, build `86341169`. The installer was inspected without executing it or modifying the installed driver.

These are authored-resource findings, not physical controller qualification. OpenXR `/input/grip/pose` support does not guarantee an OpenVR component literally named `openxr_grip`. The [OpenVR component contract](https://github.com/ValveSoftware/openvr/blob/0924064316de3effbcd1acf1e309182a2deb1c05/headers/openvr.h) distinguishes the components; the [OpenXR grip semantics](https://github.com/KhronosGroup/OpenXR-Docs/blob/18ebc0374da23c020ef4c949f9b144ffae6160ef/specification/sources/chapters/semantic_paths.adoc) are not a universal axis definition for `handgrip`.

The common positive `handgrip` evidence, including the inspected Ultra models without `openxr_grip`, supports this selection. It supersedes the earlier OpenXR-first recommendation, not the underlying recorded Quest Pro component matrices.

## Watch placement and explicit overrides

The owner-selected grab radius is **0.10 m**. Before manual adjustment, the initial watch candidate uses this transform relative to the actual right `handgrip`, in metres:

```text
[ 0,            0,           1, 0.035      ]
[-0.268297781, -0.963336027,  0, 0.10153344 ]
[ 0.963336027, -0.268297781,  0, 0.03473191 ]
```

Compose the **whole matrix**, including rotation, with the runtime component transform. Watch outward normal is handgrip `+X`; its top points distally. The previous candidate only transformed position and retained raw-device `+Y` as its normal. All A/B/C points had handgrip `x=0`, so their ±2 cm sweep never tested the new dorsal plane.

The recorded Quest Pro and inspected Touch Plus component frames share the OpenXR grip X axis. The sourced right-hand grip definition supports `+X` toward the dorsal side. The new point adds 3.5 cm dorsally and 4 cm along the neutral proximal direction to former B. These distances remain **heuristics**, not a tracked wrist joint, measured skin clearance or universal forearm axis. Runtime queries `handgrip` alone; no model table or OpenXR fallback is introduced.

The manifest supports mutually exclusive placement sources:

- With no calibrated or raw matrix, compose actual right handgrip with the watch rotation and `interaction.wrist_handgrip_point_m`, or the default point above.
- Explicit rigid 3×4 `interaction.wrist_transform` is the complete raw-right-device-relative matrix.
- `interaction.wrist_calibration` contains a rigid handgrip-relative `transform` and a `reference` with the actual right controller's `model`, `serial` and `handgrip_transform`. Native rejects a changed saved reference instead of silently applying another device's calibration.

Point components must be finite. Invalid matrices and conflicting placement sources are rejected. These are temporary native controls, not product settings UI.

### Manual wrist setup

`interaction.debug_calibrate_wrist` requires OpenVR and `image.debug_controls`. Before any photo offer, show the wrist icon with `WRIST SETUP`. In this phase only, valid tracking suffices for visibility; normal gaze/readiness gates do not prevent adjustment. A fresh left side-grip within 10 cm picks up the icon without changing its offset. Move and rotate the left hand, then release to lock the whole pose relative to the right handgrip. Tracking/reference loss cancels an unfinished drag and requires a new release/press.

Calibration cannot load a photo, authorize a request or emit a haptic. Native applies the result immediately and emits the authenticated `photo_debug_wrist_calibrated` receipt. The standalone probe atomically saves the calibration before starting photo offers or the candidate timer. A save failure aborts. Later candidates and runs load the same file; a changed controller/transport reference requires explicit recalibration. These local reference identifiers are not published.

## Shared geometry, attachment and loss handling

The native photo process resolves each actual role/device's render-model and serial identity. Its photo-input path lazily obtains the OpenVR render-model API and queries `handgrip` through `GetComponentState` with current legacy controller state. This preserves shared VRChat input without a new action manifest or input migration.

Near admission, the initial pull distance and the current pull distance use the world-left-handgrip point and the **same effective right icon center** used for gaze and display. A request requires a **12 cm increase** in that separation. Common rigid movement of both hands does not count. The selected 10 cm radius admits a fresh grab; it is not a different pull threshold.

Keep raw left/right device poses for OpenVR tracked-device-relative attachment. With raw device transforms `D_L`, `D_R` and the world icon transform `W`:

- Grab: `relative = inverse(D_L) * W`.
- Follow: `followed_W = D_L * relative`.
- Return start: `from = inverse(D_R) * followed_W`.
- Return endpoint: the current effective raw-right-relative wrist transform.

Missing or invalid component data makes photo input unavailable. Do not substitute identity, raw-origin or OpenXR references, or reuse another device's cached transform. Device, serial, model and component-transform changes advance a persistent reference epoch. The interaction reset key includes that epoch, so a replacement cancels and rearms even when indices are unchanged or another consumer, such as haptic lookup, observes the change or loss first. Recovery requires the existing visibility/readiness timing and a fresh grip after release.

Resolution is photo-input-only. Caption startup and caption process lifetime do not depend on handgrip availability.

## Pulling and holding the photo

An accepted left grip captures a photo orientation facing the HMD, using projected HMD up. It then locks that orientation relative to the left controller; it does not keep aiming at the head or use a ray. While held, the photo's **left-edge midpoint** stays at the actual left handgrip.

`progress = clamp((current separation - grab separation) / 0.12, 0, 1)`. The photo uses the owner-selected clipped reveal (B): the leftmost `progress` fraction of the image is shown with unchanged physical texel/text scale and full height. Width and UV extent change together; the whole image is not squeezed into a smaller card. The former full-size opacity candidate (A) and its `interaction.photo_emergence` selector are removed.

Alpha eases in over the first 3 cm of pull, `1 - (1 - min(pull / 0.03, 1))^2`, then stays one. Moving back below 3 cm fades it again; zero pull shows nothing. This replaces the earlier instant full-opacity appearance and is a temporary curve, not a physically observed result.

At 12 cm, the full photo latches until release, even if the hands move closer. The first continuous grip stays held across request authorization and the attempt-0-to-1 snapshot. Reopening retained/in-progress state makes no new request.

On release, capture the current world position, rotation, revealed dimensions and alpha. Detach from the controller, keep the photo fixed in space for **2.5 s**, then fade only alpha to zero over 150 ms. Moving the hand no longer moves the released photo; nothing shrinks. Early release authorizes nothing; release after authorization does not cancel the request. Photo invalidation and tracking/reference loss retain their immediate cancellation behavior.

Until the fade completes, a fresh side-grip press of **either hand** whose actual `handgrip` point lies within 5 cm of the photo's visible rectangle and 8 cm of its plane picks the photo up without moving it, restoring its release alpha. It then follows that controller with the preserved offset. Releasing places it again and restarts the 2.5 s linger. A photo regrab takes precedence over a wrist-icon grab on the same left press. Quiescing blocks regrabs. These distances are temporary values (2026-11-13 owner request), not observed ergonomics.

Native reports `photo_hidden` when the photo actually disappears after the linger and fade, not when the hand releases it.

While the authorized request is still preparing or requesting, a released photo stays in place; its 2.5 s linger starts only after the request settles (2026-11-13 owner request).

The application passes `interaction.wrist_calibration` from `photo-wrist-calibration.json` beside its settings file when that file holds a valid calibration; otherwise the default placement applies. The probe's `--calibrate-wrist --wrist-calibration <that path>` writes the same format. Photos are translated into the Self source language (`intent.languages.source_language`). Without a source-only override, the photo route is the Relay whenever the installation token owner exists; `PURIPULY_PHOTO_BENCHMARK_ROUTE=personal` restores the ChatGPT-then-Gemini route.

Native interaction notes also travel over the photo bridge as validated `photo_interaction` messages and appear in the application log as `[Photo] interaction …` records.

Original-photo viewing is removed (2026-11-13 owner decision): the `photo_debug_original` control, its compositor state and the offline `original_view.png` output no longer exist.

Portrait sources (height greater than width) stay upright throughout. Display content fits within 1080×1920 instead of 1920×1080, the send JPEG uses a 720×1280 canvas instead of 1280×720, and the 1440 reference scale uses the short side. The card keeps its long side at the selected length, so a 9:16 portrait card is 21.09375 × 37.5 cm, revealed from the held left edge like a landscape card. Python derives the same send canvas from the source dimensions; the Relay request carries it as `image_size` under photo protocol 2. Landscape files whose content was shot with a tilted camera are not detected or rotated.

The card has **no thickness**: the photo front and one flat neutral back occupy the same plane in the same OpenVR session, with no edge planes. The back mirrors the front's revealed width, full height, placement and alpha; only the head-facing side is shown, so the hidden side cannot bleed through the fade. The front keeps its original aspect and no border is added. The neutral back is an interim presentation, not a backface decision; same-image front/back remains a provisional candidate. This is an overlay card, not game-world collision, scene-depth occlusion or a per-eye mesh renderer.

`photo_transform` defaults to omitted/null for this edge-pinned placement. An explicit rigid matrix preserves the complete raw-left-relative **center** override; it is not reinterpreted as an edge anchor.

## Local pixels versus request authorization

Accepted grab starts identity-checked source loading on the owned native image worker. The original can therefore appear during the first pre-threshold pull. No pixels are read in Python, no model initializes for this original-only preview, and no send JPEG or provider photo payload is produced before valid-pull authorization.

The first stable read retains immutable encoded bytes and shared decoded/display pixels for the current photo. Authorized preparation reuses them, revalidating captured identity and the stable length/modified signature before and after private send preparation. A replaced path or changed same-file contents fail explicitly rather than silently sending a different image from the one held. Incomplete-source waiting remains bounded. A failed local read emits a safe diagnostic without authorizing a request; another explicit grab can recover.

Real pixels may finish after a fast pull. Presentation then uses the current progress/held pose, not a placeholder or second gesture. A late completion cannot recreate a released, expired, replaced or reference-lost placement. Current-photo source data may be reused after an early release; retention does not imply visibility.

Photo protocol **3** requires `expected_file_id` as two decimal strings in ordinary snapshots, immutable with path and shutter time, and carries `excluded_blocks` for bounded block recovery. Offers remain attempt 0 with `image: null`. Native and Python must be updated together; older versions are rejected without a compatibility shim, so an IPC-2 executable cannot run with current Python. Caption protocol and provider authority are unchanged.

For isolated offscreen verification only, `photo_debug_preview_source {photo_id}` requires `image.debug_controls` and the offscreen backend. It invokes the same source enqueue path as a real grab without injecting grip, hold, authorization or haptics. Existing exports observe actual pre-pull GPU pixels; this is not physical interaction evidence.

## Preserved behavior and evidence limits

The correction leaves these contracts unchanged:

- VRCT forward angles 40°/45°, face angles 45°/55°, 50 ms default show dwell, 300 ms hide dwell, 100 ms icon fade and 200 ms new-grab readiness, with a fresh grip.
- 37.5 cm long side and source aspect; 150 ms release fade after the 2.5 s linger and 225 ms ease-out icon return. Clipped reveal and world-fixed alpha-only release supersede whole-image scaling and shrinking/controller-following release. The icon return now interpolates rotation (quaternion slerp) with the same ease as position instead of snapping to the wrist orientation.
- Silent grab start, left-controller valid-pull pulse and the application-authorized right-controller first-actually-drawn-block feedback once per request attempt.
- The 10-second offer, 600-second latest-photo retention, no pre-pull photo transmission, retained-result reopening without another request and separate photo/caption process ownership.
- No Python product OCR UI, input suppression, ray fallback, provider-policy change or new retry/telemetry framework.

An authorized retained photo can display translated content during a partial pull, before another 12 cm crossing. Its first translated draw uses the existing native presentation report, not Python's `visible` intent, to request the right-hand feedback grant. A back-facing photo or translated runs outside the revealed horizontal area cannot qualify. Current photo/attempt/block/renderer identity, draw eligibility and once-per-attempt/recovery fences still apply. This neither authorizes another provider request nor changes visibility intent; original-only preview cannot produce translated-block feedback.

Prior live component evidence used Quest Pro controllers, not physical Quest 3, PICO 4 or Ultra tests. The owner rejected A/B/C anatomical placement and the former remotely bound photo behavior, then accepted the range and reported improved holding on the November 10 candidate. That observation does not qualify the new manual setup, thickness, A/B emergence or world-fixed release. The November 12 run confirmed only the saved wrist position and the B preference; the release was not observed, and the owner asked not to repeat it. The flat card, 3 cm pull fade-in and rotated icon return have not been seen in a headset. Headset appearance, comfort, legibility and felt haptics remain separate observations.

PICO Connect, Steam Link, Virtual Desktop and ALVR may expose different or emulated model identities. Inspecting one transport's assets is not a guarantee for every transport/version. Legacy Streaming Assistant `9.5.6.1` was not extractable with the available archive tools; current Connect findings are not relabeled as historical support.

## Calibrated repeat preview

The former six-case position/radius comparison and the November 12 A/B comparison are complete and retired. A standalone `--repeat-offers` run reuses the saved wrist calibration and the visible 10 cm admission range with one native child. Repeated offers use the same recorded image with fresh photo IDs; no capture or provider request is needed. `--calibrate-wrist` remains available for explicit recalibration.

`interaction.debug_case_label` identifies the candidate. It accepts at most 128 printable ASCII characters in one or two nonempty lines. `interaction.debug_grab_volume` enables three orthogonal, thin, transparent great-circle planes. Both diagnostic fields require `image.debug_controls`; they are absent/disabled by default.

The rings share the exact right-wrist center and `grab_m` used by admission, even while the icon follows the left hand or returns. Their centerlines mark the spherical boundary; they are not a filled sphere or a planar grab gate. A 256-pixel texture has a 112-pixel centerline radius, so the quad width is `2 * grab_m * 256 / (2 * 112)`, not simply `2 * grab_m`. Each plane can flip toward the head without changing its center or plane. Premultiplied transparent textures use the existing photo compositor and post-submission flush.

Outside the explicit setup phase, labels and rings follow the existing valid-input, active-photo and wrist-icon visibility policy. They do not assert that a new grab is admissible during a hold or return. They use additional handles in the same photo OpenVR session, not another input owner or caption process.

The debug-only `photo_debug_quiesce`/`photo_debug_quiescent` handshake stops new grabs while an existing interaction finishes. It never forces a hold to end. The acknowledgement remains latched until an accepted new photo ID resumes admission, or the child shuts down; same-photo/stale snapshots cannot rearm it.

See the [probe instructions](../scripts/probes/README.md#one-launch-watch-and-held-photo-experiment) for repeated ten-second opportunities, minimum spacing and release-to-continue behavior. Actual offscreen execution, texture readback and production-geometry checks support software preparation only.

Protected-document drift: `docs/architecture.md`'s isolated-photo section and `native/overlay/README.md`'s photo runtime boundary still describe source decoding only after authorized pull. Local display decoding now starts at accepted grab; send-JPEG preparation remains authorized-pull-only. Those protected files are intentionally unchanged.
