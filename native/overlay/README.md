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
- Python rejects stale publications and fixes speaker style at first readable publication. Native receives presentation styles, not speaker identities; source and translation share that style. Self and non-diarized Peer captions keep their distinct styles without consuming diarized speaker slots.
- Native exclusively owns bounded presentation retries. Eligible scene changes protect the latest accepted scene rather than replaying intermediate snapshots. Fresh rendering and GPU readiness precede submission; a cached texture or scheduled retry is not evidence of a completed presentation opportunity.
- Retry metadata cannot extend caption lifetime, revive removed content, or reset stalled-work responsibility. Runtime replacement seeds only currently valid content. Desktop presentation does not schedule native retries.
- Spatial history preserves placement for drawable identities and reclaims only eligible absent entries. It does not own caption lifetime and is not an unbounded record of every displayed caption.
- Removal still requires successful delivery and rendering. If either fails, old content can remain visible. Health and diagnostics report acceptance/progress, not caption freshness or physical headset visibility; diagnostics exclude caption text.

## Startup and Shutdown

- Native completes SteamVR preflight and OpenVR, renderer, and font initialization before bridge authentication. Initialization failures retain their startup cause rather than waiting for a connection.
- Python owns one bounded startup budget, including preparation, spawn, initialization, and readiness. Delayed authentication receives the latest pruned scene, not a snapshot retained from before initialization.
- Scene acceptance requires a valid health-challenge response covering the current scene. A live process or successful resource creation is not sufficient; presentation progress and physical visibility remain separate observations.
- OFF, failure recovery, and shutdown retain process ownership through bounded cleanup. Timeout does not prove cleanup is complete. Empty output follows the successful-empty-frame/hide contract rather than reviving a prior scene.
- OpenVR quit events for another process cannot terminate this overlay. Owned-process and global runtime quit signals remain fatal.

## Rendering Boundaries

- Desktop and VR fit each text region within its reserved line budget. A continuing caption may shrink but does not regrow on updates; a new occupant resets fitting history. Overflow keeps clipping/ellipsis behavior, not pagination.
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

Keep offline outputs private when inputs contain personal photos. Use the recovered original 23 cases for [#340's parity and X11 comparison](https://github.com/kapitalismho/PuriPuly-heart/issues/340); do not send them to a provider again. Every earlier display/control remains temporary. Physical interaction and headset legibility remain [#341](https://github.com/kapitalismho/PuriPuly-heart/issues/341); final UI remains last.

### Selected model distribution

The selected export is `mbnv4_text_seg.onnx`, SHA-256 `bfe95d72db1eea72101074293c8d27b77eb87663fa48a23e8258e59249510bcf`, derived from upstream `mbnet_manga_text_seg.pth`, SHA-256 `371ad2c07453e6f52da6c042fb2bcdc6eeb9520ca386f4616b6526ae8483e140`. This export accepts RGB uint8 NCHW and already contains ImageNet normalization and sigmoid; applying those again changes inference.

The [upstream README](https://github.com/daominhwysi/manga-translator/tree/3fd406347926b3edfc90e837fe127df6a69decbe) asserts MIT, but the inspected release has no clear weight-specific redistribution/conversion grant or applicable copyright/third-party notice inventory. **M4 distribution permission remains unresolved.** No weights are bundled or automatically downloaded. Obtain an authoritative permission basis covering the identified checkpoint and converted ONNX before shipping either. The external local artifacts used for parity do not establish distribution rights. Rust Telea/resize attribution is retained in the [third-party notices](../../src/puripuly_heart/data/THIRD_PARTY_NOTICES.txt); the product has no OpenCV dependency.


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
