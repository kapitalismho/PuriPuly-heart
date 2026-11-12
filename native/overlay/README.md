# Native Overlay

Windows Rust runtime for SteamVR subtitles. Python owns logical captions and child-process lifecycle; native code owns OpenVR presentation. The Desktop overlay is a separate renderer using the shared Python caption boundary.

This guide covers ownership, cross-language contracts, and build/verification entry points. Protocol versions, timing constants, layout values, cache mechanics, and regression details belong in the [owning implementation](#source-map). See the [architecture](../../docs/architecture.md) for the wider output path and the [CLI guide](../../docs/cli.md) for application controls.

## Ownership

| Owner | Responsibility |
| --- | --- |
| Python presenter | Caption identity, publication order, speaker style, visible content, expiry, and replacement snapshots |
| Python runtime and bridge | Target selection, process supervision, authenticated sessions, snapshot delivery, readiness observation, and bounded recovery |
| Native runtime | Accepting ordered snapshots, rendering, GPU readiness, OpenVR submission, spatial placement, and presentation retries |

Python and native must agree on the manifest, protocol, and advertised execution capabilities. Treat changes as a shared-boundary cutover, not independent renderer upgrades. Current contracts live in the Python [protocol](../../src/puripuly_heart/core/overlay/protocol.py) and [manifest](../../src/puripuly_heart/core/overlay/manifest.py), with native validation in [bridge](src/bridge.rs) and [manifest](src/manifest.rs).

## Caption and Presentation Contracts

- Snapshots replace the scene; equal or older revisions are rejected. Python prunes expired content before sending. Native does not independently expire captions or use semantic retirement metadata to remove blocks.
- Python rejects stale publications. Native receives presentation styles and overflow-divider state, not speaker identities; source and translation share the selected style. Self and non-diarized Peer captions keep their distinct styles without consuming diarized speaker slots.
- Speaker style normally freezes at first readable publication. Native Soniox Peer previews with committed speaker diarization enabled instead stay gray; accepted authoritative final/repair revisions update matching source, translation, attribution, style and divider together. Style-only delivery or duplicate events do not independently renew lifetime, reorder captions or revive expired content.
- Non-diarized cloud Peer captions, including Qwen Audio and Soniox with speaker diarization disabled, carry explicit non-diarized attribution from their first preview and retain gold through preview refreshes and final settlement.
- Native exclusively owns bounded presentation retries. Meaningful drawable Peer revisions receive fresh bounded episodes targeting a currently drawable fragment; pending fragments do not consume retry opportunities before becoming visible. Eligible scene changes protect the whole latest accepted scene rather than replaying intermediate snapshots. Fresh rendering and GPU readiness precede submission; a cached texture or scheduled retry is not evidence of a completed presentation opportunity.
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

- Python partitions full Peer revisions into ordered display fragments instead of advancing pages on a timer. Cloud and local captions share this path; fragments retain one logical caption and do not create extra translation requests or history entries. Desktop and VR render the selected fragments and retain each drawable identity across promotion.
- Visible corrections bypass unrelated new-caption waits. Pending corrections retain their registration order and wait origin; added tails register after existing waiters, obsolete tails are removed, and departed fragments cannot be revived. Ready overlay publication is independent of source-ordered context and history publication.
- Python owns reading protection and visibility-anchored idle expiry. Protection governs replacement, not deletion; upper and lower Peer fragments share the ordinary idle lifetime. Self priority and the existing maximum new-item wait can still cause earlier replacement.
- Preference and calibration reflows preflight complete fragment plans against the existing aggregate storage limit. An unretainable logical caption is retired rather than truncated or allowed to block Self; changing preferences again cannot revive it.
- Desktop and VR fit each text region within its reserved line budget. A continuing caption may shrink but does not regrow on updates; a new occupant resets fitting history. Self overflow and renderer-level residual overflow retain clipping/ellipsis behavior; native code does not independently partition captions or admit fragments.
- Fitting does not alter full caption strings, persisted settings, or Python-owned lifetime. VR surface growth accommodates calibration while stable slot geometry keeps long captions from moving neighboring captions.
- Layout/composition caches reuse compatible visual work, not publication authority. Cache hits still draw the current frame, remove stale damage, and follow producer-readiness and retry requirements.
- Placement reuse applies only after complete success on the current OpenVR handle. Changes, recreation, or failed application require placement to be applied again; spatial reanchoring remains independent.

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
.\build\overlay\PuriPulyHeartOverlay.exe --check-startup-contract
```

Graphics tests use the production executable-relative bundled-font path; release-mode tests need the font under `release/deps` instead of `debug/deps`.

The Windows-only [Overlay CI job](../../.github/workflows/pr-ci.yml) defines the deterministic, graphics, and real-process Python integration tiers. Integration runs with `INTEGRATION=1` and rejects skipped cases; use the workflow for the current selection rather than maintaining another test list here. Shared protocol/startup changes also require the Python boundary checks:

```powershell
uv run --locked pytest tests/core/test_overlay_protocol.py tests/core/test_overlay_manifest.py tests/app/test_desktop_overlay_runner.py
```

For behavior, protocol, or startup changes, complete Rust checks, a Windows release build, runtime assembly, and startup-contract verification, plus affected Python integration checks. Verify presentation and interaction separately in the intended application instance; passing startup or graphics checks alone does not establish physical headset display.

## Source Map

| Question | Read |
| --- | --- |
| Logical captions and visual projection | [Presenter](../../src/puripuly_heart/core/overlay/presenter.py), [state](../../src/puripuly_heart/core/overlay/state.py), [fragment layout](../../src/puripuly_heart/core/overlay/peer_caption_layout.py), [projection](../../src/puripuly_heart/core/overlay/presenter_projection.py) |
| Process/session ownership and readiness | [Python runtime](../../src/puripuly_heart/core/runtime/overlay.py), [process supervisor](../../src/puripuly_heart/core/overlay/process.py), [bridge](../../src/puripuly_heart/core/overlay/bridge.py) |
| Native lifecycle and accepted state | [Runtime](src/runtime.rs), [state](src/state.rs) |
| Retry ownership and execution evidence | [Retry episodes](src/retry_episode.rs), [frame cycle](src/frame_cycle.rs), [diagnostics](src/runtime_diagnostics.rs) |
| Rendering, fitting, and spatial placement | [Renderer](src/renderer/), [spatial policy](src/spatial_policy.rs), [OpenVR](src/openvr.rs) |
| Desktop renderer | [Desktop overlay](../../src/puripuly_heart/ui/desktop_overlay.py) |
| Windows executable version resources | [Shared metadata generator](../windows_version.rs), [overlay build](build.rs), [ASR worker build](../asr_worker/build.rs) |
| Native behavior and integration selection | [Rust tests](tests/), [CI workflow](../../.github/workflows/pr-ci.yml) |
