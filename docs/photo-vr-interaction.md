# Temporary photo interaction: floating wrist icon and held photo

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

## Wrist anchor and floating icon

The owner-selected grab radius is **0.10 m**. Before manual adjustment, the wrist base uses this transform relative to the actual right `handgrip`, in metres:

```text
[ 0,            0,           1, 0.035      ]
[-0.268297781, -0.963336027,  0, 0.10153344 ]
[ 0.963336027, -0.268297781,  0, 0.03473191 ]
```

Compose the **whole matrix**, including rotation, with the runtime component transform. Watch outward normal is handgrip `+X`; its top points distally. The previous candidate only transformed position and retained raw-device `+Y` as its normal. All A/B/C points had handgrip `x=0`, so their ±2 cm sweep never tested the new dorsal plane.

The recorded Quest Pro and inspected Touch Plus component frames share the OpenXR grip X axis. The sourced right-hand grip definition supports `+X` toward the dorsal side. The new point adds 3.5 cm dorsally and 4 cm along the neutral proximal direction to former B. These distances remain **heuristics**, not a tracked wrist joint, measured skin clearance or universal forearm axis. Runtime queries `handgrip` alone; no model table or OpenXR fallback is introduced.

The manifest supports mutually exclusive placement sources:

- With no calibrated or raw matrix, compose actual right handgrip with the watch rotation and `interaction.wrist_handgrip_point_m`, or the default point above.
- Explicit rigid 3×4 `interaction.wrist_transform` is the complete raw-right-device-relative base matrix, before the shared floating displacement.
- `interaction.wrist_calibration` contains a rigid handgrip-relative base `transform` and a `reference` with the actual right controller's `model`, `serial` and `handgrip_transform`. Native rejects a changed saved reference instead of silently applying another device's calibration.

Point components must be finite. Invalid matrices and conflicting placement sources are rejected. These are temporary native controls, not product settings UI.

The November 13 owner decision selects a **3 cm floating offset** as a temporary single candidate, with physical VR comparison deferred to a later combined session. Resolve the selected base pose first, then add `0.03 * base.local_Z` to its translation without changing its rotation. Local `+Z` is the icon's outward normal, not world up; for the default base it is handgrip `+X`, making the effective default handgrip-local X position 0.065 m. This applies to saved calibration, explicit raw-device base matrices and default/overridden handgrip points alike. It is additional clearance from the existing base, not a measured distance from skin or avatar clothing.

The saved calibration and manifest base transforms are not rewritten when loaded. The same effective floating center drives icon presentation, gaze, admission, relative pull, debug rings and the return endpoint. Icon width, rotation, 10 cm grab radius, 12 cm pull threshold and visibility timing are unchanged. There is no bobbing animation, A/B preview or new setting.

### Head-facing icon and retained-photo approach gate

The November 14 owner decision prioritizes an effortless pull in the 10 seconds after a capture without leaving the icon in view while a photo is retained. Two photos in the owner's first run missed the icon at face angles of 63–64° while the head-forward angle was within 40°, and one shown icon hid at an 80° face angle while the head-forward angle was 26°.

- The icon stays at the effective floating wrist center, but it always faces the HMD and uses projected HMD up, like the held photo. The calibrated or default wrist rotation now sets only the floating direction; it no longer orients the icon or gates its visibility. Grab, return and the debug label use this head-facing pose.
- During an unpulled offer, the icon shows when the head-forward angle is within 40° and hides beyond 45°, with the existing dwell and fade.
- After the offer has been pulled, and for retained attempts, the icon also requires the actual left handgrip within `grab_m + 0.10 m` of the icon center to show and within `grab_m + 0.15 m` to stay (20/25 cm at the default 10 cm radius). Glancing at the right hand alone no longer shows a retained photo's icon.
- Near-miss notes use the head-forward margin alone and, for retained photos, are recorded only while the left handgrip is within the stay distance. The `facing_deg` note field still reports the wrist base normal's angle to the head, for diagnosis only.

The owner will judge this by direct use; no separate measurement was run.

### Manual wrist setup

`interaction.debug_calibrate_wrist` requires OpenVR and `image.debug_controls`. Before any photo offer, show the floating wrist icon with `WRIST SETUP`. In this phase only, valid tracking suffices for visibility; normal gaze/readiness gates do not prevent adjustment. A fresh left side-grip within 10 cm picks up the icon without changing its offset. Move and rotate the left hand, then release to lock the whole pose relative to the right handgrip. Before saving, remove the 3 cm displacement along the adjusted pose's own outward normal so the file continues to hold the base pose. Reapplying the float reproduces the released position without a jump or accumulated height on later loads or adjustments. Tracking/reference loss cancels an unfinished drag and requires a new release/press.

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

The application passes `interaction.wrist_calibration` from `photo-wrist-calibration.json` beside its settings file when that file holds a valid calibration; otherwise the default placement applies. The probe's `--calibrate-wrist --wrist-calibration <that path>` writes the same format. Photos are translated into the Self source language (`intent.languages.source_language`). Without a source-only override, the photo route is the user's `intent.photo.route` selection (`managed` Relay by default, `chatgpt` or `gemini`); a selected route without its credential makes no offer. `intent.photo.enabled=false` suppresses offers, and `intent.photo.delete_translated=true` moves a translated source PNG to the Recycle Bin when its retention ends. `PURIPULY_PHOTO_BENCHMARK_ROUTE=personal` restores the ChatGPT-then-Gemini route.

Native interaction notes also travel over the photo bridge as validated `photo_interaction` messages and appear in the application log as `[Photo] interaction …` records.

Original-photo viewing is removed (2026-11-13 owner decision): the `photo_debug_original` control, its compositor state and the offline `original_view.png` output no longer exist.

Portrait sources (height greater than width) stay upright throughout. Display content fits within 1080×1920 instead of 1920×1080, the send JPEG uses a 720×1280 canvas instead of 1280×720, and the 1440 reference scale uses the short side. The card keeps its long side at the selected length, so a 9:16 portrait card is 21.09375 × 37.5 cm, revealed from the held left edge like a landscape card. Python derives the same send canvas from the source dimensions; the Relay request carries it as `image_size` under photo protocol 2. Landscape files whose content was shot with a tilted camera are not detected or rotated.

The card has **no thickness**: the photo front and one flat neutral back occupy the same plane in the same OpenVR session, with no edge planes. The back mirrors the front's revealed width, full height, placement and alpha; only the head-facing side is shown, so the hidden side cannot bleed through the fade. The front keeps its original aspect and no border is added. The neutral back is an interim presentation, not a backface decision; same-image front/back remains a provisional candidate. This is an overlay card, not game-world collision, scene-depth occlusion or a per-eye mesh renderer.

`photo_transform` defaults to omitted/null for this edge-pinned placement. An explicit rigid matrix preserves the complete raw-left-relative **center** override; it is not reinterpreted as an edge anchor.

## Local pixels versus request authorization

An eligible offer reads only identity-checked PNG/JPEG header dimensions on the native image worker, without decoding pixels, resizing, populating the source cache or preparing output files. Format detection does not depend on the filename extension. Metadata work has one replaceable low-priority slot and yields to accepted grab/preparation work; incomplete-header waiting is bounded. Accepted grab still starts actual source loading. No pixels are read in Python, no model initializes for this original-only preview, and no send JPEG or provider photo payload is produced before valid-pull authorization.

The first stable read retains immutable encoded bytes and shared decoded/display pixels for the current photo. Authorized preparation reuses them, revalidating captured identity and the stable length/modified signature before and after private send preparation. A replaced path or changed same-file contents fail explicitly rather than silently sending a different image from the one held. Incomplete-source waiting remains bounded. A failed local read emits a safe diagnostic without authorizing a request; another explicit grab can recover.

While pixels are pending, the current pull/held placement uses the existing neutral gray card material on the front as well as the back. The header supplies the correct landscape/portrait ratio before display; if the header itself is incomplete, card display waits for dimensions rather than showing a wrong-ratio placeholder. Real pixels replace only the front texture at the current progress/held pose, without a second gesture or animation restart. The gray front cannot claim translated first-draw or haptic authority, and source failure removes pending-card eligibility. A late completion cannot recreate a released, expired, replaced or reference-lost placement. Current-photo source data may be reused after an early release; retention does not imply visibility.

Photo protocol **3** requires `expected_file_id` as two decimal strings in ordinary snapshots, immutable with path and shutter time, and carries `excluded_blocks` for bounded block recovery. Offers remain attempt 0 with `image: null`. Native and Python must be updated together; older versions are rejected without a compatibility shim, so an IPC-2 executable cannot run with current Python. Caption protocol and provider authority are unchanged.

For isolated offscreen verification only, `photo_debug_preview_source {photo_id}` requires `image.debug_controls` and the offscreen backend. It invokes the same source enqueue path as a real grab without injecting grip, hold, authorization or haptics. Existing exports observe actual pre-pull GPU pixels; this is not physical interaction evidence.

### Native processing resources

Masking preserves the selected model, bilinear tile interpolation, maximum-overlap threshold semantics and fractional bbox clipping, but accumulates the final byte mask directly instead of retaining a full ROI float probability map. Quantized Lab channels use bytes; color histograms preserve channel medians, including the average of the two middle values for even sample counts. Telea reuses distance coefficients without changing its visit order, radius or output rules. ONNX Runtime remains sequential with intra-op 2/inter-op 1; intra-op worker spinning is disabled.

Authorized preparation releases the encoded/high-resolution source cache after private-file creation and final identity/signature validation. The image boundary transfers either decoded source pixels or the already-resized working image once, rather than retaining both. The compositor keeps dimensions/reference scale after upload, not a CPU image reference. After draining all completed blocks in an authoritative terminal snapshot, the image worker releases working pixels, the model session, accumulated erase background, unused preparation regions, Telea workspace and photo-specific layout caches. Completed-block identity, GPU layers and published layouts survive, so hide/reopen neither repeats inference nor moves accepted placements. Hidden requesting photos retain their computation state. A later attempt can decode the retained private source and lazily reload the model; it does not reopen the camera file or resend a provider request. The font engine remains reusable.

Telea consumes the owned ROI crop in place. State, distance, dilation, band and heap storage are reused within an active attempt; alpha still compares against immutable global erase-source coordinates. The renderer converts the owned RGBA patch to premultiplied BGRA in place before upload. Neither change alters mask, erase, sampling or layout rules.

GPU submission and completion are separate runtime events. Pending work is polled at 2 ms intervals with the existing 50 ms failure bound while bridge and input events remain serviceable. The last completed photo remains visible; a second render target is allocated lazily when a presented texture must not be overwritten. This adds one BGRA target (about 7.91 MiB at 1920×1080) on that graphics path, not an equivalent claim about process Working Set. Retry/replacement fences discard stale completion, and first-draw authority uses the completed frame's timestamp and block cutoff. Offline/debug readback explicitly waits for completion without busy-yield polling.

These choices exchange retained computation memory and busy waiting for private-source reload on a later completed-attempt retry and, when needed, a second GPU target. Allocator/graphics caches need not return memory immediately to the OS. Report Working Set, private commit, process CPU and wall-clock timings separately. Isolated native offscreen comparisons exercise the real model, Telea, DirectWrite and D3D paths, but do not exercise the OpenVR-only second-target allocation or qualify headset appearance, input latency, GPU contention or felt haptics.

#### Buffered JPEG output

Send-JPEG encoding uses a buffered file writer, with an explicit fallible flush and file close before preparation succeeds. Encoding and final-flush failures retain `jpeg_failed` and the existing owned-file cleanup. `jpeg_ms` includes the flush and close. JPEG quality 85, dimensions, letterboxing, source validation and pull authorization are unchanged; this change does not alter timer resolution or model lifetime.

November 14 verification passed all nine native preparation tests. One isolated offscreen child completed three synthetic 3840×2160 photo requests and acknowledged shutdown. All three send JPEGs and retained source files matched the previously recorded unbuffered outputs byte-for-byte. JPEG-stage medians were 105.9 ms in the earlier run and 10.9 ms in the candidate run; these are separate runs of one synthetic fixture, not a provider/HMD latency guarantee or proof that the reported first-use stutter is fixed. The verified executable SHA-256 is `333963bb42daaeff6ab01b7fbe556fd37425a2daeb167416498459e3195d6b61`, staged at `build/overlay/PuriPulyHeartOverlay.exe` without restarting the running app. Local source pin, test output and native receipts are under `C:/tmp/puri_jpeg_buffer_20261114`.

#### First-draw warmup and completed-text reuse

Before `photo_ready`, each native photo process exercises its real compositor with a small synthetic image and outlined text: original drawing, active text fade, completed-text accumulation and cache reuse. It waits for GPU completion, then clears synthetic photo resources, counters and draw authority. This moves first-use rendering work into startup; it does not preload a real photo, the segmentation model or a provider request. Caption rendering, process priority and timer policy are unchanged.

Fully faded text accumulates incrementally into one transparent photo-sized BGRA bitmap, in the existing block order. Newly fading blocks remain live. Per-block opacity grouping and outlined glyph commands are preserved; every erase patch is drawn before the cached and live text, so later erasing cannot cover earlier translations. Erased-only diagnostics exclude text. Retry/replacement clears the text cache; retained hide/reopen keeps accepted layouts and does not repeat inference. Completed-frame fences still determine first-draw authority.

The cache adds one BGRA bitmap when completed text first needs it: 8,294,400 bytes (7.91 MiB) at 1920×1080, not one full-photo bitmap per block. It does not add continuous idle repainting. Startup work and this retained GPU resource are explicit tradeoffs; releasing a bitmap does not guarantee an immediate equivalent Working Set reduction.

Verification passed 156 distinct focused tests: 117 photo-library cases, 17 preparation cases, 20 default native-runtime cases, the selected-model native pipeline, and one controlled OpenVR function-table session using real D3D textures. Coverage includes header-only PNG/JPEG, fast-grab priority, portrait placement before/after pixels, gray-to-image texture submission without translated feedback, incomplete-source recovery, stale completions, overlapping erase/text fades, cache rollover, retry and warmup-state isolation. The packaged native executable also completed three synthetic 3840×2160 photo operations, acknowledged shutdown and exited 0 without provider requests. Initial verification found a Windows binding conversion compile error, missing fonts in the isolated fixture layout, and a landscape-only assertion after the grab fixture became portrait; these were corrected and the failed checks passed.

Two fresh processes per variant each rendered three sequential synthetic photos, in baseline/candidate/candidate/baseline order. The dense case uses 16 generated outlined-text regions without erase patches; a separate selected-model block exercises real erasing and half/full text fade. The following are median render-call wall times, not input latency or headset frame times. First-photo rows have two samples per variant; other rows have six.

| Render call, milliseconds | Before | After |
| --- | ---: | ---: |
| First original | 3.01 | 0.47 |
| First translated fade | 7.75 | 1.69 |
| Completed-text append, 1 block | 1.24 | 1.34 |
| Completed-text append, 4 blocks | 4.12 | 1.18 |
| Completed-text append, 8 blocks | 7.64 | 1.21 |
| Completed-text append, 16 blocks | 15.40 | 1.17 |
| Reuse of 16 completed blocks | 15.48 | 0.13 |

The synthetic startup warmup took 71.0–86.7 ms in these two candidate processes. Dense 16-block append GPU intervals had medians of 14.34 ms before and 0.18 ms after; these overlap the render-call measurements and must not be added to them. The single-block case does not establish a speedup. OS/driver caches were not reset, and active fades still draw their unfinished text.

All three non-timing layout/geometry metadata comparisons matched. Original and real-block half-fade PNGs matched byte-for-byte. Completed cached text is **not pixel-identical**: the real-block full image differs at 29 pixels by one channel step; dense final images differ at about 25,585 of 2,073,600 pixels, overwhelmingly by one step. Some dense comparisons have 1–3 pixels with larger differences, up to 15/255. The baseline itself varies at three greater-than-one-step pixels between its sequential photos, but that does not explain every cross-variant difference. Flattening through BGRA8 adds rounding; no bit-exact fidelity claim is made. Dense before/after, model-generated output, held translated output and portrait native readbacks were visually inspected without an obvious new layout/clipping defect. Controlled presentation and readbacks are not physical HMD qualification or reproduction of the reported intermittent stutter.

Evidence, full failure/success logs, source pins, image comparisons and probe archives are under `C:/tmp/puri_photo_render_fix`. The verified executable SHA-256 is `5244f641a7e2b89f4d4eb8b929bd3ce0cec597face6f9b8c66e572244105b0c9`, staged at `build/overlay/PuriPulyHeartOverlay.exe`; the prior staged executable is preserved in the evidence directory. The running app was not restarted. Unselected external fixtures and actual SteamVR/HMD tests were not run.

#### Initial optimization evidence (before the structural changes)

Verification on 2026-11-13 passed 40 focused native tests, including the controlled Telea reference; four other external-fixture cases in the selected suites were not run. A fresh-process before/after comparison used only the synthetic 2560×1440 six-block photo, the selected model/runtime and actual 1920×1080 offscreen rendering. All compared original, mask, erase and composite pixels and non-timing metadata matched. Retained hide/reopen and attempt-two reprocessing preserved the result, and both children shut down normally. A separate same-runtime spinning comparison preserved model output bits. This single pair reduced retained-state memory and repeated-processing CPU; it did not demonstrate lower processing peaks or post-clear memory. It is not a statistical benchmark, private-photo quality campaign or physical HMD observation.

The source-frozen baseline executable SHA-256 was `553e2ba218d7b438f313204e324dc96dd372142d3c0c2baa5e3384621e894c78`; the verified candidate staged under `build/overlay` was `fcb2dfbe2995217d9c7761d098bfb397b0c1ceb3201a9a7171458a4757ba68cd`. The staged photo startup contract reported application `2.10.3`, protocol `3`. Candidate compilation used an isolated target after a shared-target Cargo cache reused the baseline executable; that rejected duplicate was not used for the comparison.

#### Structural optimization evidence

The structural pass compared the immediately preceding build with the repaired final candidate, not the original pre-optimization implementation. Three alternating fresh-process pairs used the synthetic 2560×1440 six-block photo at 1920×1080, with identical model/runtime/source bytes. The following values are medians across the three runs. Memory peaks are sampled process peaks; retained/cleared values are medians within each observation interval.

| Metric | Before | After |
| --- | ---: | ---: |
| First-processing peak Working Set, MiB | 180.6 | 172.4 |
| First-processing peak private commit, MiB | 207.1 | 198.0 |
| Prepared/waiting Working Set, MiB | 68.5 | 58.0 |
| Prepared/waiting private commit, MiB | 87.9 | 77.4 |
| Retained-result Working Set, MiB | 97.0 | 80.8 |
| Retained-result private commit, MiB | 118.6 | 102.8 |
| Retry peak Working Set, MiB | 193.6 | 190.7 |
| Retry peak private commit, MiB | 217.3 | 212.3 |
| Working Set three seconds after clear, MiB | 89.5 | 89.5 |
| Private commit three seconds after clear, MiB | 113.3 | 111.3 |
| First-processing process CPU, seconds | 0.969 | 0.906 |
| Retry process CPU, seconds | 0.828 | 0.766 |
| Retry source-recovery elapsed time, milliseconds | 0.74 | 52.08 |

CPU sums user and kernel time across all process threads, from preview start through terminal settling for the first pass and from source recovery through terminal settling for retry; PNG exports are excluded. CPU ranges overlap, so three runs do not establish a reliable speedup. Sampling requested 10 ms and observed approximately 15.5 ms median intervals; shorter memory peaks can be missed. Retained results were observed for 1.5 seconds after 250 ms settling; clear measurements followed three seconds of settling. The scripted gaps are not end-user latency measurements. Post-clear Working Set did not improve, and this is not a long-duration leak test.

All 68 compared PNG pairs matched both decoded pixels and encoded bytes. The corpus covers landscape, portrait, fractional downscaling, overlapping blocks, plate fallback and KEEP, plus cold/hide-reopen/retry exports in all three process pairs. Geometry, placement, font and decision metadata matched after excluding timing fields; every private source and authorized JPEG also matched byte-for-byte. Native landscape and portrait composites were opened for visual inspection.

The targeted verification passed 137 distinct tests: 105 default photo-library tests, nine preparation integration tests, 20 native-runtime integration tests including the selected-model pipeline, two controlled Telea references and one controlled OpenVR function-table session. The external photo-processing caller also compiled. Private-case/model/resampling/angle fixtures and real SteamVR/HMD cases were not run. Fixture corrections preserved assertions: controlled sequential erasing uses explicit reference units; the VR fixture follows the independently changed floating wrist center; the render drain waits for both pending and dirty work; preview/recovery dimensions agree in the private-handoff case.

Independent memory/ROI review found no material defect. GPU review found three defects in the intermediate candidate, all repaired and independently reverified: same-photo attempt transitions now retain the completed visible frame without retaining old draw/haptic authority; original reuse distinguishes sources and permits only verified camera-to-private handoff; unavailable debug export leaves presentation untouched. Regression checks cover both 0→1 and 1→2 transitions, same-size source replacement, late completion, timeout, bridge servicing and once-per-attempt feedback.

One earlier baseline probe lost its connection during retry and remained incomplete. Its cause is unresolved because the original collector did not persist stderr/exit status on failure. That run is preserved separately and excluded from the summary; after correcting diagnostic collection, all six final measured children exited normally with empty stderr. This does not establish that the earlier baseline failure was fixed.

The source-frozen baseline executable SHA-256 is `8fcdafc0c8cfcf3ec07655feba86d846a208098ab4c6cf1508edcf39eee035bd`; the final candidate is `16df95fe6db9ae0fa9a90c2b88f3067e919fc3f98870ffc6e26bf30518e89f40`. The measured candidate was staged at `build/overlay/PuriPulyHeartOverlay.exe`, whose startup contract reported application `2.10.3`, protocol `3`. Local raw samples, synthetic inputs, output comparisons, source pins, failed-run evidence and final test logs are under `C:/tmp/puri_photo_structure_20261113`; `final-comparison.json` records the complete metrics, provenance, review outcomes and limitations. Temporary measurement scripts were removed.

## Preserved behavior and evidence limits

The correction leaves these contracts unchanged:

- VRCT forward angles 40°/45°, 50 ms default show dwell, 300 ms hide dwell, 100 ms icon fade and 200 ms new-grab readiness, with a fresh grip.
- 37.5 cm long side and source aspect; 150 ms release fade after the 2.5 s linger and 225 ms ease-out icon return. Clipped reveal and world-fixed alpha-only release supersede whole-image scaling and shrinking/controller-following release. The icon return now interpolates rotation (quaternion slerp) with the same ease as position instead of snapping to the resting orientation, which now faces the head.
- Silent grab start, left-controller valid-pull pulse and the application-authorized right-controller first-actually-drawn-block feedback once per request attempt.
- The 10-second offer, 600-second latest-photo retention, no pre-pull photo transmission, retained-result reopening without another request and separate photo/caption process ownership.
- No Python product OCR UI, input suppression, ray fallback, provider-policy change or new retry/telemetry framework.

An authorized retained photo can display translated content during a partial pull, before another 12 cm crossing. Its first translated draw uses the existing native presentation report, not Python's `visible` intent, to request the right-hand feedback grant. A back-facing photo or translated runs outside the revealed horizontal area cannot qualify. Current photo/attempt/block/renderer identity, draw eligibility and once-per-attempt/recovery fences still apply. This neither authorizes another provider request nor changes visibility intent; original-only preview cannot produce translated-block feedback.

Prior live component evidence used Quest Pro controllers, not physical Quest 3, PICO 4 or Ultra tests. The owner rejected A/B/C anatomical placement and the former remotely bound photo behavior, then accepted the range and reported improved holding on the November 10 candidate. That observation does not qualify the new manual setup, thickness, A/B emergence or world-fixed release. The November 12 run confirmed only the saved wrist position and the B preference; the release was not observed, and the owner asked not to repeat it. The flat card, 3 cm pull fade-in and rotated icon return have not been seen in a headset. Headset appearance, comfort, legibility and felt haptics remain separate observations.

The floating candidate has not been observed in a headset. Per the owner's decision, apparent clearance, grasp comfort, controller interference and avatar/sleeve overlap remain for the later combined VR observation; software geometry and calibration round trips do not qualify those outcomes.

Software verification for the floating candidate passed 44 focused native tests covering interaction geometry/timing, calibrated save/reload/re-adjustment without drift, controller-reference loss and the quiescent bridge. The bridge fixture was aligned with the floating center after its initial pull assertion failed. A release build was staged at `build/overlay/PuriPulyHeartOverlay.exe`; its photo startup contract reported application `2.10.3`, protocol `3`. No VR session was launched.

PICO Connect, Steam Link, Virtual Desktop and ALVR may expose different or emulated model identities. Inspecting one transport's assets is not a guarantee for every transport/version. Legacy Streaming Assistant `9.5.6.1` was not extractable with the available archive tools; current Connect findings are not relabeled as historical support.

## Calibrated repeat preview

The former six-case position/radius comparison and the November 12 A/B comparison are complete and retired. A standalone `--repeat-offers` run reuses the saved wrist calibration and the visible 10 cm admission range with one native child. Repeated offers use the same recorded image with fresh photo IDs; no capture or provider request is needed. `--calibrate-wrist` remains available for explicit recalibration.

`interaction.debug_case_label` identifies the candidate. It accepts at most 128 printable ASCII characters in one or two nonempty lines. `interaction.debug_grab_volume` enables three orthogonal, thin, transparent great-circle planes. Both diagnostic fields require `image.debug_controls`; they are absent/disabled by default.

The rings share the exact right-wrist center and `grab_m` used by admission, even while the icon follows the left hand or returns. Their centerlines mark the spherical boundary; they are not a filled sphere or a planar grab gate. A 256-pixel texture has a 112-pixel centerline radius, so the quad width is `2 * grab_m * 256 / (2 * 112)`, not simply `2 * grab_m`. Each plane can flip toward the head without changing its center or plane. Premultiplied transparent textures use the existing photo compositor and post-submission flush.

Outside the explicit setup phase, labels and rings follow the existing valid-input, active-photo and wrist-icon visibility policy. They do not assert that a new grab is admissible during a hold or return. They use additional handles in the same photo OpenVR session, not another input owner or caption process.

The debug-only `photo_debug_quiesce`/`photo_debug_quiescent` handshake stops new grabs while an existing interaction finishes. It never forces a hold to end. The acknowledgement remains latched until an accepted new photo ID resumes admission, or the child shuts down; same-photo/stale snapshots cannot rearm it.

See the [probe instructions](../scripts/probes/README.md#one-launch-watch-and-held-photo-experiment) for repeated ten-second opportunities, minimum spacing and release-to-continue behavior. Actual offscreen execution, texture readback and production-geometry checks support software preparation only.

Protected-document drift: `docs/architecture.md`'s isolated-photo section and `native/overlay/README.md`'s photo runtime boundary still describe source decoding only after authorized pull. Local display decoding now starts at accepted grab, offer-time work reads only headers, and the photo compositor has a synthetic startup warmup; send-JPEG preparation remains authorized-pull-only. Those protected files are intentionally unchanged.
