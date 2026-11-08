# Temporary photo interaction: wristwatch and held photo

This records the bounded native correction for [#341](https://github.com/kapitalismho/PuriPuly-heart/issues/341), under [#338](https://github.com/kapitalismho/PuriPuly-heart/issues/338). It is temporary VR interaction, not final UI, physical X3 acceptance or issue closure. After the separate position/radius experiment, the owner selected 10 cm admission, reported every A/B/C position below the thumb rather than at the wrist, and requested a wristwatch placement and a photo that unfolds from the gripping left hand.

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

The owner-selected grab radius is **0.10 m**. The new watch candidate uses this transform relative to the actual right `handgrip`, in metres:

```text
[ 0,            0,           1, 0.035      ]
[-0.268297781, -0.963336027,  0, 0.10153344 ]
[ 0.963336027, -0.268297781,  0, 0.03473191 ]
```

Compose the **whole matrix**, including rotation, with the runtime component transform. Watch outward normal is handgrip `+X`; its top points distally. The previous candidate only transformed position and retained raw-device `+Y` as its normal. All A/B/C points had handgrip `x=0`, so their ±2 cm sweep never tested the new dorsal plane.

The recorded Quest Pro and inspected Touch Plus component frames share the OpenXR grip X axis. The sourced right-hand grip definition supports `+X` toward the dorsal side. The new point adds 3.5 cm dorsally and 4 cm along the neutral proximal direction to former B. These distances remain **heuristics**, not a tracked wrist joint, measured skin clearance or universal forearm axis. Runtime queries `handgrip` alone; no model table or OpenXR fallback is introduced.

The manifest retains two mutually exclusive placement overrides:

- Omitted/null `interaction.wrist_transform`: compose actual right handgrip with the watch rotation and `interaction.wrist_handgrip_point_m`, or the default point above when omitted/null.
- Explicit rigid 3×4 `interaction.wrist_transform`: use the complete raw-right-device-relative matrix.

Point components must be finite. Invalid matrices and simultaneous explicit matrix/point overrides are rejected. These are temporary native controls, not product settings UI.

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

An accepted left grip captures a photo orientation facing the HMD, using projected HMD up. It then locks that orientation relative to the left controller; it does not keep aiming at the head or use a ray. The photo's **left-edge midpoint** is pinned to the actual left handgrip. Its center moves half the current width along photo X, so growth, release and controller rotation preserve the held edge.

While pulling, `progress = clamp((current separation - grab separation) / 0.12, 0, 1)`. Alpha equals progress; width equals `photo_width_m * progress`. At 12 cm, full progress latches until release, even if the hands move closer again. Full width remains 37.5 cm with source aspect and no outer margins.

The first continuous grip now remains held across request authorization and the attempt-0-to-1 snapshot. It no longer hides the photo and requires another pull. Release reverses alpha/width from their current values over 150 ms; an early release sends no authorization. A post-trigger release hides presentation without cancelling the authorized request. Reopening retained/in-progress state does not send another request.

`photo_transform` defaults to omitted/null for this edge-pinned placement. An explicit rigid matrix preserves the complete raw-left-relative **center** override; it is not reinterpreted as an edge anchor.

## Local pixels versus request authorization

Accepted grab starts identity-checked source loading on the owned native image worker. The original can therefore appear during the first pre-threshold pull. No pixels are read in Python, no model initializes for this original-only preview, and no send JPEG or provider photo payload is produced before valid-pull authorization.

The first stable read retains immutable encoded bytes and shared decoded/display pixels for the current photo. Authorized preparation reuses them, revalidating captured identity and the stable length/modified signature before and after private send preparation. A replaced path or changed same-file contents fail explicitly rather than silently sending a different image from the one held. Incomplete-source waiting remains bounded. A failed local read emits a safe diagnostic without authorizing a request; another explicit grab can recover.

Real pixels may finish after a fast pull. Presentation then uses the current progress/held pose, not a placeholder or second gesture. A late completion cannot recreate a released, expired, replaced or reference-lost placement. Current-photo source data may be reused after an early release; retention does not imply visibility.

Photo protocol **2** requires `expected_file_id` as two decimal strings in ordinary snapshots, immutable with path and shutter time. Offers remain attempt 0 with `image: null`. Native and Python must be updated together; version 1 is rejected without a compatibility shim. Caption protocol and provider authority are unchanged.

For isolated offscreen verification only, `photo_debug_preview_source {photo_id}` requires `image.debug_controls` and the offscreen backend. It invokes the same source enqueue path as a real grab without injecting grip, hold, authorization or haptics. Existing exports observe actual pre-pull GPU pixels; this is not physical interaction evidence.

## Preserved behavior and evidence limits

The correction leaves these contracts unchanged:

- VRCT forward angles 40°/45°, face angles 45°/55°, 50 ms default show dwell, 300 ms hide dwell, 100 ms icon fade and 200 ms new-grab readiness, with a fresh grip.
- 37.5 cm full width and source aspect; 150 ms release and 225 ms icon return. Distance-driven opening supersedes the former time-only 150 ms appearance.
- Silent grab start, left-controller valid-pull pulse and the application-authorized right-controller first-actually-drawn-block feedback once per request attempt.
- The 10-second offer, 600-second latest-photo retention, no pre-pull photo transmission, retained-result reopening without another request and separate photo/caption process ownership.
- No Python product OCR UI, input suppression, ray fallback, provider-policy change or new retry/telemetry framework.

Prior live component evidence used Quest Pro controllers, not physical Quest 3, PICO 4 or Ultra tests. The owner rejected A/B/C anatomical placement and the former remotely bound photo behavior. The new candidate has synthetic nonidentity-pose checks and actual offscreen source/texture/preparation evidence, not a physical pass. Wrist alignment, edge grip, rotation comfort, HMD visibility/readability and felt haptics require a new owner observation.

PICO Connect, Steam Link, Virtual Desktop and ALVR may expose different or emulated model identities. Inspecting one transport's assets is not a guarantee for every transport/version. Legacy Streaming Assistant `9.5.6.1` was not extractable with the available archive tools; current Connect findings are not relabeled as historical support.

## Single-candidate repeat experiment

The former six-case position/radius comparison is retired. The next standalone run uses one native process, the new watch/held-photo defaults and a visible 10 cm admission range. It automatically offers the selected recorded image with fresh photo IDs; no new capture or provider request is necessary.

`interaction.debug_case_label` identifies the candidate. It accepts at most 128 printable ASCII characters in one or two nonempty lines. `interaction.debug_grab_volume` enables three orthogonal, thin, transparent great-circle planes. Both diagnostic fields require `image.debug_controls`; they are absent/disabled by default.

The rings share the exact right-wrist center and `grab_m` used by admission, even while the icon follows the left hand or returns. Their centerlines mark the spherical boundary; they are not a filled sphere or a planar grab gate. A 256-pixel texture has a 112-pixel centerline radius, so the quad width is `2 * grab_m * 256 / (2 * 112)`, not simply `2 * grab_m`. Each plane can flip toward the head without changing its center or plane. Premultiplied transparent textures use the existing photo compositor and post-submission flush.

Labels and rings follow the existing valid-input, active-photo and wrist-icon visibility policy. They do not assert that a new grab is admissible during a hold or return. They use additional handles in the same photo OpenVR session, not another input owner or caption process.

The debug-only `photo_debug_quiesce`/`photo_debug_quiescent` handshake stops new grabs while an existing interaction finishes. It never forces a hold to end. The acknowledgement remains latched until an accepted new photo ID resumes admission, or the child shuts down; same-photo/stale snapshots cannot rearm it.

See the [probe instructions](../scripts/probes/README.md#one-launch-watch-and-held-photo-experiment) for repeated ten-second opportunities, minimum spacing and release-to-continue behavior. Actual offscreen execution, texture readback and production-geometry checks support software preparation only.

Protected-document drift: `docs/architecture.md`'s isolated-photo section and `native/overlay/README.md`'s photo runtime boundary still describe source decoding only after authorized pull. Local display decoding now starts at accepted grab; send-JPEG preparation remains authorized-pull-only. Those protected files are intentionally unchanged.
