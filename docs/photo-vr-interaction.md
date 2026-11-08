# Temporary photo interaction: hand reference and wrist position

This records the bounded native correction for [#341](https://github.com/kapitalismho/PuriPuly-heart/issues/341), under [#338](https://github.com/kapitalismho/PuriPuly-heart/issues/338). It is temporary VR interaction, not final UI, physical X3 acceptance or issue closure. The owner requested cross-controller reference research and an icon position toward the wrist when the fist is closed, rather than immediately above the thumb.

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

## Default placement and explicit overrides

The initial grab-radius candidate is **0.08 m**. The default icon center is the actual right `handgrip` transform applied to the handgrip-local point **`(0, 0.063, 0.024)` m**. For a component transform with rotation `R_H` and translation `t_H`, the effective raw-right-device-relative center is:

`center = t_H + R_H * (0, 0.063, 0.024)`

This rounded wristward heuristic derives from the recorded Quest Pro frame relationship. It is not anatomical wrist tracking, a universal forearm axis or final ergonomic tuning. Runtime code queries the actual component; it does not hardcode Quest Pro's derived raw-device offset or maintain a headset-model offset table.

Preserve the existing raw-right-device-relative icon rotation exactly:

```text
[ 1,  0,  0 ]
[ 0,  0,  1 ]
[ 0, -1,  0 ]
```

The normal remains raw `+Y`. Apply the component rotation to the candidate position only, not to this face rotation. There is no new dorsal-facing rotation or 90-degree plane change.

The manifest offers two mutually exclusive placement overrides:

- Omitted or `null` `interaction.wrist_transform`: apply the actual right `handgrip` to `interaction.wrist_handgrip_point_m`, or the default point above when that field is omitted or `null`. Keep the unchanged raw face rotation.
- Explicit rigid 3×4 `interaction.wrist_transform`: use the complete matrix relative to the raw right tracked device, preserving the existing override contract.

Provided handgrip-local point components must be finite. Invalid matrices and simultaneous explicit matrix/point overrides are rejected rather than silently ignoring one.

`photo_transform` remains fully raw-left-device-relative and unchanged. There is no additional settings UI or configuration framework.

## Shared geometry, attachment and loss handling

The native photo process resolves each actual role/device's render-model and serial identity. Its photo-input path lazily obtains the OpenVR render-model API and queries `handgrip` through `GetComponentState` with current legacy controller state. This preserves shared VRChat input without a new action manifest or input migration.

Near admission, the initial pull distance and the current pull distance use the world-left-handgrip point and the **same effective right icon center** used for gaze and display. A request requires a **12 cm increase** in that separation. Common rigid movement of both hands does not count. The 8 cm candidate radius admits a fresh grab; it is not a different pull threshold.

Keep raw left/right device poses for OpenVR tracked-device-relative attachment. With raw device transforms `D_L`, `D_R` and the world icon transform `W`:

- Grab: `relative = inverse(D_L) * W`.
- Follow: `followed_W = D_L * relative`.
- Return start: `from = inverse(D_R) * followed_W`.
- Return endpoint: the current effective raw-right-relative wrist transform.

Missing or invalid component data makes photo input unavailable. Do not substitute identity, raw-origin or OpenXR references, or reuse another device's cached transform. Device, serial, model and component-transform changes advance a persistent reference epoch. The interaction reset key includes that epoch, so a replacement cancels and rearms even when indices are unchanged or another consumer, such as haptic lookup, observes the change or loss first. Recovery requires the existing visibility/readiness timing and a fresh grip after release.

Resolution is photo-input-only. Caption startup and caption process lifetime do not depend on handgrip availability.

## Preserved behavior and evidence limits

The correction leaves these contracts unchanged:

- VRCT forward angles 40°/45°, face angles 45°/55°, 50 ms default show dwell, 300 ms hide dwell, 100 ms icon fade and 200 ms new-grab readiness, with a fresh grip.
- Raw-left-relative held-photo placement and 37.5 cm width; 150 ms appearance/release and 225 ms icon return.
- Silent grab start, left-controller valid-pull pulse and the application-authorized right-controller first-actually-drawn-block feedback once per request attempt.
- The 10-second offer, 600-second latest-photo retention, no pre-pull photo transmission, retained-result reopening without another request and separate photo/caption process ownership.
- No Python product OCR UI, input suppression, ray fallback, provider-policy change or new retry/telemetry framework.

Prior live component evidence used Quest Pro controllers, not physical Quest 3, PICO 4 or Ultra tests. The corrected candidate's production-math smoke used nonidentity synthetic state because SteamVR was unavailable; no live candidate input/component sampling, physical grip, felt haptic or HMD visibility result is claimed. The existing X3 physical alignment, closed-fist placement and comfort observations remain pending. Historical distance-only counts at the old center do not establish admission or successful grabbing at the relocated center.

PICO Connect, Steam Link, Virtual Desktop and ALVR may expose different or emulated model identities. Inspecting one transport's assets is not a guarantee for every transport/version. Legacy Streaming Assistant `9.5.6.1` was not extractable with the available archive tools; current Connect findings are not relabeled as historical support.

## Separate position and radius experiment

The standalone probe compares two factors separately, not a Cartesian matrix. Its position phase keeps the radius at 8 cm and tries these right-handgrip-local points:

| Position | Point in metres | Relative to B |
| --- | --- | --- |
| A | `(0, 0.04373328, 0.01863404)` | 2 cm toward the hand, heuristically |
| B | `(0, 0.063, 0.024)` | Current candidate |
| C | `(0, 0.08226672, 0.02936596)` | 2 cm toward the forearm, heuristically |

A/C use B ± `0.02 * (0, 0.963336, 0.268298)`. This direction derives from the earlier controller-frame relationship; it does not locate an anatomical wrist joint. The radius phase then keeps B fixed and tries 8, 12 and 15 cm. These options do not change the normal default or the separate 12 cm pull-distance increase.

`interaction.debug_case_label` identifies the current phase, value and fixed factor. It accepts at most 128 printable ASCII characters in one or two nonempty lines. `interaction.debug_grab_volume` enables three orthogonal, thin, transparent great-circle planes. Both diagnostic fields require `image.debug_controls`; they are absent/disabled by default.

The rings share the exact right-wrist center and `grab_m` used by admission, even while the icon follows the left hand or returns. Their centerlines mark the spherical boundary; they are not a filled sphere or a planar grab gate. A 256-pixel texture has a 112-pixel centerline radius, so the quad width is `2 * grab_m * 256 / (2 * 112)`, not simply `2 * grab_m`. Each plane can flip toward the head without changing its center or plane. Premultiplied transparent textures use the existing photo compositor and post-submission flush.

Labels and rings follow the existing valid-input, active-photo and wrist-icon visibility policy. They do not assert that a new grab is admissible during a hold or return. They use additional handles in the same photo OpenVR session, not another input owner or caption process.

Each case starts a fresh native child with immutable configuration. The debug-only `photo_debug_quiesce`/`photo_debug_quiescent` handshake stops new grabs while an existing interaction finishes; it does not force release or replace the case mid-hold. The acknowledgement remains latched until an accepted new photo ID resumes the same case, or the child shuts down. Same-photo/stale snapshots cannot rearm it. The next child starts only after acknowledged, successful cleanup.

See the [probe instructions](../scripts/probes/README.md#one-launch-wrist-position-and-grab-radius-experiment) for the six-case order, repeated ten-second opportunities, target timing and release-to-continue behavior. Actual offscreen child execution, texture readback and production-geometry checks support software preparation only. Sphere depth appearance, anatomical alignment, label legibility and felt feedback still require the owner's headset observations.
