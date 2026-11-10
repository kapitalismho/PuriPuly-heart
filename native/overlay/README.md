# Native Overlay

Windows Rust runtime for SteamVR captions and a separate photo process. Python owns application policy and child-process lifecycle; native code owns OpenVR presentation and photo pixel work. The Desktop caption overlay remains a separate renderer using the shared Python caption boundary.

This guide covers ownership, cross-language contracts, and build/verification entry points. Protocol versions, timing constants, layout values, cache mechanics, and regression details belong in the [owning implementation](#source-map). See the [architecture](../../docs/architecture.md) for the wider output path and the [CLI guide](../../docs/cli.md) for application controls.

## Ownership

| Owner | Responsibility |
| --- | --- |
| Python presenter | Caption identity, publication order, speaker style, visible content, expiry, and replacement snapshots |
| Python runtime and bridge | Target selection, process supervision, authenticated sessions, snapshot delivery, readiness observation, and bounded recovery |
| Native caption runtime | Accepting ordered caption snapshots, rendering, GPU readiness, OpenVR submission, spatial placement, and presentation retries |
| Python photo owner | Passive capture, route eligibility, pull authorization, provider requests, retention, and independent photo supervision |
| Native photo runtime | Photo-session/surface ownership; off-loop source decoding, send-JPEG, masking, erase and text layout; layered GPU composition |

Python and native must agree on the manifest, protocol, and advertised execution capabilities. Treat changes as a shared-boundary cutover, not independent renderer upgrades. Caption contracts live in the Python [protocol](../../src/puripuly_heart/core/overlay/protocol.py) and [manifest](../../src/puripuly_heart/core/overlay/manifest.py), with native validation in [bridge](src/bridge.rs) and [manifest](src/manifest.rs). Photo uses a separate [manifest](src/photo/manifest.rs) and [authenticated bridge](../../src/puripuly_heart/core/photo/bridge.py).

## Caption and Presentation Contracts

- Snapshots replace the scene; equal or older revisions are rejected. Python prunes expired content before sending. Native does not independently expire captions or use semantic retirement metadata to remove blocks.
- Python rejects stale publications. Native receives presentation styles and overflow-divider state, not speaker identities; source and translation share the selected style. Self and non-diarized Peer captions keep their distinct styles without consuming diarized speaker slots.
- Speaker style normally freezes at first readable publication. Native Soniox Peer previews instead stay gray; accepted authoritative final/repair revisions update matching source, translation, attribution, style and divider together. Style-only delivery or duplicate events do not independently restart pages/lifetime, reorder captions or revive expired content.
- Qwen Audio Peer captions carry explicit non-diarized attribution from their first preview, preserving the same gold style through preview refreshes and final settlement.
- Native exclusively owns bounded presentation retries. Eligible scene changes protect the latest accepted scene rather than replaying intermediate snapshots. Fresh rendering and GPU readiness precede submission; a cached texture or scheduled retry is not evidence of a completed presentation opportunity.
- Retry metadata cannot extend caption lifetime, revive removed content, or reset stalled-work responsibility. Runtime replacement seeds only currently valid content. Desktop presentation does not schedule native retries.
- Spatial history preserves placement for drawable identities and reclaims only eligible absent entries. It does not own caption lifetime and is not an unbounded record of every displayed caption.
- Removal still requires successful delivery and rendering. If either fails, old content can remain visible. Health and diagnostics report acceptance/progress, not caption freshness or physical headset visibility; diagnostics exclude caption text.

## Startup and Shutdown

- Native completes SteamVR preflight and OpenVR, renderer, and font initialization before bridge authentication. Initialization failures retain their startup cause rather than waiting for a connection.
- The Python bridge bounds both payload bytes and fragment assembly before authentication completes. Excessive fragmentation closes the offending connection without consuming the session token or preventing a fresh valid connection.
- Python owns one bounded startup budget, including preparation, spawn, initialization, and readiness. Delayed authentication receives the latest pruned scene, not a snapshot retained from before initialization.
- Scene acceptance requires a valid health-challenge response covering the current scene. A live process or successful resource creation is not sufficient; presentation progress and physical visibility remain separate observations.
- OFF, failure recovery, and shutdown retain process ownership through bounded cleanup. Timeout does not prove cleanup is complete. Empty output follows the successful-empty-frame/hide contract rather than reviving a prior scene.
- Explicit OpenVR runtime or driver shutdown requests stop CAPTIONS normally only after a matching native completion, successful child exit, and resource cleanup. Python clears the ephemeral enabled intent without changing the saved VR target, calibration, or desktop bounds. It does not restart the overlay or switch to Desktop; restarting SteamVR requires enabling CAPTIONS again. Capture and translation remain independent.
- Own-process quit faults, connection loss, rendering failures, and unexpected exits remain failures; quit events for another process are ignored. Already-polled faults and cleanup failures remain visible even alongside an explicit runtime shutdown request. Once that request is admitted, failure recovery cannot restart into the quitting runtime.
- Initial enable while SteamVR is absent retains the existing Desktop fallback. This is distinct from an already-running VR overlay receiving an explicit shutdown request.

### Desktop Startup and Interaction

- The Desktop renderer uses a separately owned Flet view process. Flet is the sole ordinary-startup geometry and visibility writer; Win32 binds the owned process and observes bounds/visibility while providing native-only topmost and click-through behavior.
- Startup is generation-scoped: page configuration, native readiness, canonical bounds, visible-state confirmation, then READY. An early first-visible observation is distinct from full readiness. Native window-state confirmation is not proof of rendered pixels or freedom from occlusion.
- The actual owned process is binding authority. PID-file and endpoint observations validate identity; they must not silently redirect binding to a different view. Retired generations cannot acknowledge readiness or persist startup bounds as user movement.
- Failed, cancelled and replaced views retain an explicit cleanup owner through confirmed process exit and PID-file cleanup. A replacement is a different generation, not retroactive success of the failed attempt; unresolved cleanup is not success.
- While locked, the Windows adapter owns an event-driven relative-Z-order watcher. It may restore an overlapping topmost peer's ordering without activation or geometry changes; unlock, rebind and close retire the watcher. Window visibility/topmost flags alone do not establish caption pixels.

## Rendering Boundaries

- Python owns Peer pagination, page timing, version resets and reading protection. Desktop and VR render the current page and retain occupant identity across promotion.
- Desktop and VR fit each text region within its reserved line budget. A continuing caption may shrink but does not regrow on updates; a new occupant resets fitting history. Self overflow and renderer-level residual overflow retain clipping/ellipsis behavior; native code does not independently paginate or advance Python's page.
- Fitting does not alter full caption strings, persisted settings, or Python-owned lifetime. VR surface growth accommodates calibration while stable slot geometry keeps long captions from moving neighboring captions.
- Layout/composition caches reuse compatible visual work, not publication authority. Cache hits still draw the current frame, remove stale damage, and follow producer-readiness and retry requirements.
- Placement reuse applies only after complete success on the current OpenVR handle. Changes, recreation, or failed application require placement to be applied again; spatial reanchoring remains independent.

## Photo Runtime Boundary

- The same executable runs the photo role in a separate OS process using `--role photo --config <photo-manifest>`. Role dispatch precedes caption resource initialization. Caption OFF, photo failures, and photo animation do not share caption process lifetime or presentation-retry progress.
- Each photo process has an independent authenticated loopback bridge, instance identity and generation. Its OpenVR session owns distinct icon/photo handles; releasing one surface does not shut down the other.
- Python opens an image-only provider connection at the first photo-file signal. No image leaves the PC before a valid pull. Luna uses the existing Sign in with ChatGPT session; Google uses the existing AI Studio credential owner. Neither transport borrows the speech connection pool.
- A valid pull authorizes native source decoding, Lanczos fit/letterbox and JPEG encoding on the owned image worker. The worker returns completed private source/JPEG paths, original dimensions and the precise send transform. Python reads/base64-serializes the JPEG. A brief incomplete-file wait ends in explicit failure; process loss during preparation does not create a new job-resumption system.
- A prepared JPEG survives the permitted provider retry independently of renderer recovery. Reopening retained state does not send another request. Retry clears failed-attempt blocks; the request budget, expiry and terminal-error policy belong to the [Python lifecycle](../../src/puripuly_heart/core/photo/lifecycle.py).
- Full snapshots retain all current-attempt blocks. Geometry is floating-point original-photo `x0,y0,x1,y1`, not Google normalized coordinates or display-composite pixels. Bbox-ready identifies a candidate region; source-ready supplies the complete source and physical line count for final mask/erase geometry; block-complete applies the keep rule and typesets the complete translation. Early work stays hidden. The amount of erase work overlapped with streaming is measured, not assumed.
- The image worker resamples the retained decoded original, runs the selected CPU ONNX model and sequential Rust Telea, then measures and fits text using DirectWrite and the existing bundled Noto CJK/fallback chain. ONNX Runtime uses intra-op 2/inter-op 1. The render loop consumes prepared layouts instead of performing inference or fitting.
- Photo outline width denotes the exterior ink radius. The photo caller doubles that value for Direct2D's centered geometric stroke before drawing the fill. The shared caption renderer retains its existing stroke-width contract.
- Mask inference, fill sampling and source measurement read the immutable original. Telea reads the current attempt's committed erase background in block-arrival order. A hidden prepared erase is refreshed without repeating inference only when a prior completed patch changes its input region. Kept blocks and plates do not alter that background; retry resets it. Reported erase cost includes any refresh rather than claiming all preparation was hidden.
- Native image timings use the authenticated photo bridge and the existing metadata-only `[Photo] image_*` diagnostics. Mask/erase preparation is measured even if the completed block is later kept. Additional erase refresh records are additive work segments, not repeated cumulative totals. Typesetting, CPU ROI upload and valid GPU composition duration remain separate; none is physical headset latency or Python product log-view content.
- The original image is uploaded once for its display lifetime. Completed blocks add ROI erase patches and text without moving previous placements. Erase and complete text fade in together over 100 ms. Original viewing disables both upper layers; retry clears them while retaining the original image.
- Display content fits within 1920×1080 by default. The fixed spatial constants retain their 1440p reference units and scale by the actual resized photo-content height divided by 1440, excluding padding. Bboxes and derived glyph/font sizes are already in working coordinates and are not scaled again. Reference/custom bounds do not create a Python resolution control.
- The explicit `probe` backend remains receipt-only and does not initialize image inference or graphics. `offscreen` executes the real native image and D3D11/Direct2D path without OpenVR; `openvr` uses the photo process's own session and matched display adapter. Offscreen pixels and successful submissions are not proof of physical HMD visibility.
- Native preparation stages, connection time, actual request send, first text, first completed block and stream completion are separate measurements. None is a headset-display timestamp. First-drawn-block feedback requires a real drawing report and is not replayed during renderer recovery.

### Temporary image controls and offline verification

The optional native manifest `image` object accepts `model_path`, `runtime_path`, `max_size`, `export_dir` and `debug_controls`. `max_size` defaults to `[1920,1080]`; debug controls default off. Without an explicit model path, native looks beside the executable under `assets/photo/mbnv4_text_seg.onnx`. A missing model is an explicit image-processing failure, not an alternate model or plate-only fallback. Caption startup remains independent.

For a temporary authenticated bridge probe, enable `image.debug_controls`. `photo_debug_original` accepts the current `photo_id` and `enabled` to toggle the original layer. `photo_debug_export` accepts the current `photo_id` and a safe `label`, and writes a readback under the configured `export_dir`. The existing probe pull/hide messages are also enabled on this temporary surface. These are native verification controls, not product Python OCR controls or final wrist/peek gestures.

The standalone offline path reuses the decoder, image pipeline, typesetter and GPU compositor:

```powershell
PuriPulyHeartOverlay.exe --role photo --offline <job.json>
```

The job contains `source_path`, `output_dir`, `model_path`, optional `runtime_path`, optional `max_size`, and `blocks`. Each block has `block_id`, original-photo floating-point `bbox`, `source` and complete `translation`. It writes original, erase-only, composite and original-view PNGs, per-block mask/erase artifacts and `metadata.json`. CPU mask/erase/typeset durations, GPU timestamp durations and total artifact wall time are separate observations; readback and file output are not the ROI-upload estimate. Invalid GPU timing is unavailable, not zero.

Keep offline outputs private when inputs contain personal photos. Use the recovered original 23 cases for [#340's parity and X11 comparison](https://github.com/kapitalismho/PuriPuly-heart/issues/340); do not send them to a provider again. Every earlier display/control remains temporary. This port does not establish physical headset legibility, and no prior X3 observation was recovered. Physical interaction belongs to [#341](https://github.com/kapitalismho/PuriPuly-heart/issues/341); final UI remains last.

### Selected model distribution

The selected export is `mbnv4_text_seg.onnx`, 48,103,781 bytes (45.88 MiB), SHA-256 `bfe95d72db1eea72101074293c8d27b77eb87663fa48a23e8258e59249510bcf`, derived from upstream `mbnet_manga_text_seg.pth`, SHA-256 `371ad2c07453e6f52da6c042fb2bcdc6eeb9520ca386f4616b6526ae8483e140`. This export accepts RGB uint8 NCHW and already contains ImageNet normalization and sigmoid; applying those again changes inference.

The [upstream README](https://github.com/daominhwysi/manga-translator/tree/3fd406347926b3edfc90e837fe127df6a69decbe) asserts MIT, but the inspected release has no clear weight-specific redistribution/conversion grant or applicable copyright/third-party notice inventory. **M4 distribution permission remains unresolved.** No weights are bundled or automatically downloaded. Obtain an authoritative permission basis covering the identified checkpoint and converted ONNX before shipping either. The external local artifacts used for parity do not establish distribution rights. Rust Telea/resize attribution is retained in the [third-party notices](../../src/puripuly_heart/data/THIRD_PARTY_NOTICES.txt); the product has no OpenCV dependency.

The model-distribution decision, owner responsibilities and installer resumption criteria are tracked in [#340](https://github.com/kapitalismho/PuriPuly-heart/issues/340). Installer integration remains deferred; the current product code and download behavior are unchanged.


### Issue #340 verification snapshot — 2026-10-31

Implementation: `5e16714d`; exterior-outline repair: `5597bfa0`. The final optimized executable SHA-256 is `8d4626e8722e13afa163c00d859cea15eef6ff307f12da974852ccccd8f8bb5c`. These are observations on the recovered sample, not new quality or latency acceptance thresholds.

All 23 originals (10 user PNGs and 13 hard JPEGs), 229 fixed provider blocks and selected expected outputs were recovered. Exact historical Python reproduction had zero action, core-mask, clean-pixel or final-layout-metadata differences. The export script was unavailable; the actual selected ONNX graph and checkpoint identity were inspected. Original photos, fixed inputs, expected artifacts and prototype/model files remained unchanged; no provider request or photo publication was needed.

The final native executable completed all 69 runs: the same 23 cases at original working dimensions, 2560×1440 fit and 1920×1080 fit. All reported `Noto Sans CJK KR` / `BundledNotoCjkMedium`, measured Hangul ink ratio `0.916003942489624`, and one original upload. Actual content height controls scaling: the two non-full-height 1080p cases were 1068 and 719 pixels high, not 1080.

| Pipeline / bounds | Inpaint | Plate | Collisions | Below floor | Inpaint blocks reaching exact source cap | Residual OCR characters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Python / original dynamic | 204 | 6 | 9 | 1 | 71 / 204 | 158 |
| Native / original dynamic | 202 | 8 | 9 | 0 | 60 / 202 | 158 |
| Python / 2560×1440 | 202 | 8 | 9 | 1 | 69 / 202 | 160 |
| Native / 2560×1440 | 201 | 9 | 9 | 0 | 58 / 201 | 158 |
| Python / 1920×1080 | 201 | 9 | 9 | 1 | 73 / 201 | 128 |
| Native / 1920×1080 | 205 | 5 | 9 | 0 | 73 / 205 | 115 |

Every row also has 19 kept blocks. Residual OCR uses the existing confidence ≥0.6 rule, excludes whitespace, and covers that row's inpaint cohort using actual working bboxes and the erase background before plates. On the 201 common native inpaint blocks, 1080p/reference residual counts were 114/158, but source OCR also read fewer characters at 1080p (5634/5811). Lower residual counts do not establish better readability. The selected 1080p default is unchanged; X3 physical legibility remains unobserved.

Meaningful differences and repaired regressions:

- All 630 non-kept observations matched independent operative action/core-leftover and full-foreground angle/direction oracles; all 57 keep observations matched normalization. All 25 native/Python action differences were attributed to the specified original-core leftover rule versus the historical eroded-core/count-gated implementation.
- All 15 angle differences were traced to the historical plate measurement omission, random 4000-point sampling or small working-pixel/float-ROI differences. The large 45° differences were checked against full foreground on both working-pixel/ROI variants. Four selected crop-boundary outliers reproduced the native core exactly when given the same native crop.
- DirectWrite metrics/rasterization, fractional geometry and actual physical source-line counts differ from Pillow/integer historical behavior. The issue-required at-floor fallback replaces the prototype's below-floor output; collision counts are not a zero-collision gate.
- Fixed port regressions included JPEG decoding, cumulative erase context, stride-sampling angle aliasing, double plate alpha and half-width exterior outlines. Missing executable-relative Noto assembly invalidated an early comparison and was corrected before the final runs.
- The final outline refresh preserved 1514 source/core/erase stage files and stable layout/font/dimension metadata across all 69 runs, with zero equivalence failures. Only then were 608 unchanged OCR block observations reused; final composites and visual sheets were regenerated and inspected.

Release-stage timings below are milliseconds, shown as p50 / p95 / maximum. Each stage uses its measured block cohort; final-composite GPU timing is per photo. These fresh-process-per-photo native observations are not the prototype's fresh-process-per-stage timing/memory records.

| Stage | 1920×1080 default | 2560×1440 reference |
| --- | ---: | ---: |
| Mask | 49.69 / 136.58 / 185.67 | 53.03 / 146.48 / 223.50 |
| Erase | 5.07 / 40.40 / 106.70 | 13.70 / 108.18 / 283.18 |
| Typeset | 4.27 / 32.05 / 59.59 | 6.71 / 52.32 / 103.63 |
| CPU ROI upload | 0.215 / 1.100 / 1.811 | 0.409 / 1.292 / 2.533 |
| Final-composite GPU | 2.85 / 9.90 / 11.34 | 4.53 / 13.75 / 18.09 |

Mask/erase timings include hidden preparation for subsequently kept blocks. All 69 final-frame GPU queries were valid. ROI timestamp markers can defer first-use transfer into composition cost, so they are not isolated upload-bandwidth measurements. The sub-1 ms ROI estimate held at the median, not universally. Offline wall time includes initialization, model loading, readback and PNG export; it does not establish the plan's first-visible-result estimate or physical headset latency.

The existing Monitor path observed the final 69-run batch on RX 7900 XTX with SteamVR/VRChat active: 5 s baseline, 91.92 s native, 5 s after. Owned photo PIDs had 3D utilization p95 0.134% / maximum 0.75%, and whole-process VRAM median 37.15 MiB / maximum 73.6 MiB. Native-window compositor observations included zero dropped and four mispresented frames; reprojection was present throughout baseline, native and after windows. Two transient PDH invalid-data samples were retained. Uncontrolled game load and 250 ms sampling can hide short bursts: this is relevant contention evidence, not a causal “no contention” result, a texture-only memory measurement or HMD visibility proof. No existing PresentMon session was interrupted.

Integration verification:

- Final `cargo check --locked` passed; the non-`windows_graphics_` suite passed 271 cases and the graphics selection passed 30. Seven default-suite external-fixture/device cases were ignored there and separately exercised: byte-resize, Telea, overlapping-context and original-angle oracles, selected-model inference, and actual offscreen/OpenVR pipelines.
- Actual release offscreen/OpenVR checks passed both cases with selected Noto/model/runtime. They exercised retained decode, hidden preparation, cumulative blocks, joint fade, original view, retry reuse, first-draw/reopen fencing and independent OpenVR session survival. The new exterior-outline GPU regression failed before the repair and passed after it without weakening the existing layer/plate assertions.
- Affected Python checks passed 237 cases against the final executable, including real photo-process recovery and composed caption-OFF startup. Architecture contracts passed 352 cases. Both native role startup-contract commands succeeded.
- An isolated actual composed-application probe exercised current authenticated CLI operations and real native offscreen rendering. Native image timings crossed the real bridge, owner and existing persisted diagnostic sink; hidden KEEP work remained invisible, a later translated block uploaded an actual ROI, and no photo text/path or live-log delivery escaped. Owned native/application processes acknowledged shutdown and exited cleanly. This probe preceded only the photo-outline scalar repair; its unchanged bridge/lifecycle/privacy coverage remains applicable.
- Ruff, Black and workflow Python test-selection checks passed. Full Rust formatting inspection still reports unrelated baseline formatting differences; only task-owned formatting was corrected. The existing `logging.rs` atomic deprecation warning was not changed.

M6 retains measured quality limitations: overlapping tiny/title blocks remain; no free-space region growth was added; Telea lobes/smears remain in large holes; and whole-logo damage/coverage remains. The owner accepted the current outcome for #340; the authoritative disposition is recorded in that issue. Further overlap splitting, region growth, large-hole treatment or semantic logo preservation would require separately scoped decisions, not additional approval for this port. No replacement inpainter, new quality threshold, X2/X4/X6/X10 campaign, Python OCR UI or final VR UI was introduced. M4 remains the distribution blocker above. No physical held-photo legibility observation was available.

The private focus sheets below illustrate the accepted quality limitations:

- `overlap_new_books.jpg` and `overlap_title.jpg`: later blocks are fitted around existing placements, but shrinking stops at the readability floor. Overlapping source regions can therefore leave tiny text over a large title. Already placed blocks are not rearranged; changing region ownership/splitting would require a separately scoped policy.
- `small_region_no_growth.jpg`: fitting uses the recovered ink region and permitted model-box candidates, not an automatically expanded neighboring free-space region. Longer translations can become small even when nearby space appears available. Growth needs rules that avoid covering illustrations and other text.
- `large_title_smear.jpg`: Telea propagates local boundary colors into erased regions; it cannot reconstruct the illustration concealed by a large title. The erase-only column exposes dark lobes that the translated title only partly covers. Choosing KEEP or a plate for such regions trades background restoration for preservation or coverage.
- `logo.jpg`: text-shaped parts of a decorative logo are treated as removable text. Translation can erase or cover anniversary digits and ornamentation. The current normalized source-equals-translation KEEP rule is not a semantic logo-preservation rule; semantic logo preservation would require a separately scoped policy.

Focus-sheet columns are original, erase/background result and composite. Rows are Python 1440, final native 1440, Python 1080 and final native 1080; the bottom-right panel is the final native 1080 composite. These are offline quality observations, not physical VR legibility evidence.

Private reproducibility records remain under the local `native_photo340_20261031` run directory: `verification_record.json`, `native_timing_summary.json`, `original_integrity.json`, `native_outline_execution.json`, `outline_stage_equivalence.json`, `final_dispositions_summary.json`, `angle_dispositions_refined.json`, `native_crop_boundary_oracle.json`, final comparison sheets and `contention/20261031T003605Z_native_batch_offline/`. Images and provider text are not committed; the native offline job format above is the reproduction entry point.

## Build and Verification

Run from the repository root on Windows with the Windows build tools and the Python/uv versions pinned in the [PR workflow](../../.github/workflows/pr-ci.yml). Use ordinary Rustup setup for the compiler selected by [rust-toolchain.toml](../../rust-toolchain.toml). Install the locked Python development environment: cross-language Rust tests invoke the real Python presenter. Use a short target path to avoid MSBuild tracking-file path limits.

```powershell
uv sync --locked --extra dev --python 3.14.10
$overlayTarget = Join-Path $env:TEMP 'puripuly-overlay-tests'
cargo test --locked --manifest-path native/overlay/Cargo.toml --target-dir $overlayTarget -- --skip windows_graphics_

$fontDir = Join-Path $overlayTarget 'debug/deps/puripuly_heart/data/fonts'
New-Item -ItemType Directory -Force -Path $fontDir | Out-Null
Copy-Item -LiteralPath 'src/puripuly_heart/data/fonts/NotoSansCJK-Medium.ttc' -Destination $fontDir -Force
cargo test --locked --manifest-path native/overlay/Cargo.toml --target-dir $overlayTarget windows_graphics_ -- --test-threads=1

cargo build --locked --manifest-path native/overlay/Cargo.toml --release --bin PuriPulyHeartOverlay --target-dir $overlayTarget
New-Item -ItemType Directory -Force -Path build/overlay | Out-Null
Copy-Item (Join-Path $overlayTarget 'release/PuriPulyHeartOverlay.exe') build/overlay/PuriPulyHeartOverlay.exe -Force
Copy-Item third_party/openvr/win64/openvr_api.dll build/overlay/openvr_api.dll -Force
Copy-Item third_party/openvr/win64/openvr_api.dll.sha256 build/overlay/openvr_api.dll.sha256 -Force
.\build\overlay\PuriPulyHeartOverlay.exe --check-startup-contract
```

Graphics tests use the production executable-relative bundled-font path; release-mode tests need the font under `release/deps` instead of `debug/deps`.
For a release executable outside the checkout's staged `build/overlay` directory, also place the trusted OpenVR DLL and its `.sha256` sidecar beside that executable before using the Python process supervisor. Direct offline execution does not exercise the supervisor's packaged-runtime integrity check.

The Windows-only [Overlay CI job](../../.github/workflows/pr-ci.yml) defines the deterministic, graphics, and real-process Python integration tiers. Integration runs with `INTEGRATION=1` and rejects skipped cases; use the workflow for the current selection rather than maintaining another test list here. Shared protocol/startup changes also require the Python boundary checks:

```powershell
uv run --locked pytest tests/core/test_overlay_protocol.py tests/core/test_overlay_manifest.py tests/app/test_desktop_overlay_runner.py
```

For behavior, protocol, or startup changes, complete Rust checks, a Windows release build, runtime assembly, and startup-contract verification, plus affected Python integration checks. Verify presentation and interaction separately in the intended application instance; passing startup or graphics checks alone does not establish physical headset display.

## Source Map

| Question | Read |
| --- | --- |
| Logical captions and visual projection | [Presenter](../../src/puripuly_heart/core/overlay/presenter.py), [projection](../../src/puripuly_heart/core/overlay/presenter_projection.py) |
| Process/session ownership and readiness | [Python runtime](../../src/puripuly_heart/core/runtime/overlay.py), [process supervisor](../../src/puripuly_heart/core/overlay/process.py), [bridge](../../src/puripuly_heart/core/overlay/bridge.py) |
| Native lifecycle and accepted state | [Runtime](src/runtime.rs), [state](src/state.rs) |
| Photo capture, requests and child supervision | [Application owner](../../src/puripuly_heart/app/services/photo_application.py), [lifecycle](../../src/puripuly_heart/core/photo/lifecycle.py), [process supervisor](../../src/puripuly_heart/core/photo/process.py) |
| Photo native state, preparation and image handoff | [Photo runtime](src/photo/runtime.rs), [state](src/photo/state.rs), [preparation](src/photo/prepare.rs), [image boundary](src/photo/image.rs) |
| Photo mask, erase and source geometry | [CPU processing](src/photo/processing.rs), [Rust Telea](src/photo/telea.rs) |
| Photo text fitting and layered GPU rendering | [Typesetting](src/photo/typeset.rs), [image worker](src/photo/worker.rs), [compositor](src/photo/compositor.rs), [offline runner](src/photo/offline.rs) |
| Retry ownership and execution evidence | [Retry episodes](src/retry_episode.rs), [frame cycle](src/frame_cycle.rs), [diagnostics](src/runtime_diagnostics.rs) |
| Rendering, fitting, and spatial placement | [Renderer](src/renderer/), [spatial policy](src/spatial_policy.rs), [OpenVR](src/openvr.rs) |
| Desktop renderer | [Desktop overlay](../../src/puripuly_heart/ui/desktop_overlay.py) |
| Windows executable version resources | [Shared metadata generator](../windows_version.rs), [overlay build](build.rs), [ASR worker build](../asr_worker/build.rs) |
| Native behavior and integration selection | [Rust tests](tests/), [CI workflow](../../.github/workflows/pr-ci.yml) |
