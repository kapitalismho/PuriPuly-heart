# Native Overlay

Windows Rust runtime for SteamVR captions and a separate photo process. Python owns application policy and child-process lifecycle. Native code owns OpenVR presentation and photo pixel work. Desktop captions use a separate Python renderer.

See the [architecture](../../docs/architecture.md) for ownership across the application and the [CLI guide](../../docs/cli.md) for controls.

## Ownership

| Owner | Responsibility |
| --- | --- |
| Python presenter | Caption content, identity, ordering, styles, fragments and expiry |
| Python runtime and bridge | Target selection, authenticated sessions, child supervision and recovery |
| Native caption runtime | Snapshot rendering, GPU readiness, OpenVR placement and presentation retries |
| Python photo owner | Capture, pull authorization, provider requests, retention and child supervision |
| Native photo runtime | Independent OpenVR session, image preparation and layered composition |

## Compatibility Boundary

Python and native must agree on manifests, protocols and advertised capabilities. Change both sides together. Caption contracts live in the Python [protocol](../../src/puripuly_heart/core/overlay/protocol.py) and [manifest](../../src/puripuly_heart/core/overlay/manifest.py), with native validation in [bridge](src/bridge.rs) and [manifest](src/manifest.rs).

Snapshots replace the scene. Native rejects stale revisions and renders Python's selected content. Presentation retries cannot renew caption lifetime or revive removed content. Python owns startup, recovery and cleanup.

SteamVR captions Flush the producer context once after successful texture submission, before visibility handling; pre-submit readiness is unchanged. The default avoids periodic redraws/resubmissions. Settings → Overlay → SteamVR Caption Compatibility Mode (`overlay.vr_compatibility_mode`, default `false`) retains Flush and enables bounded P05 redraws/resubmissions, with cached rehandoff off. This is manual selection, not automatic fallback or a pre-Flush rollback; photo and desktop overlays are unaffected.

The default manifest omits the option. Older native binaries reject explicit compatibility opt-in rather than silently ignoring it. Cached rehandoff remains a separate diagnostic.

## Photo Runtime Boundary

The executable runs the photo role in a separate process with `--role photo --config <photo-manifest>`. Caption and photo lifecycles are independent. Photo uses its own [manifest](src/photo/manifest.rs) and [authenticated bridge](../../src/puripuly_heart/core/photo/bridge.py).

Python authorizes provider requests; no image leaves the PC before a valid pull. Native prepares images and renders complete translation blocks. Block geometry uses floating-point coordinates in the original photo.

[Windows Setup](../../installer/photo-model.iss) provisions the required model. Python supplies model and ONNX Runtime paths to the child. Model provenance and licenses are in the [third-party notices](../../src/puripuly_heart/data/THIRD_PARTY_NOTICES.txt).

### Offline Rendering

Run the native image pipeline without a provider or OpenVR:

```powershell
PuriPulyHeartOverlay.exe --role photo --offline <job.json>
```

Define jobs with the [offline runner's schema](src/photo/offline.rs). Jobs contain image/model paths and translation blocks in original-photo coordinates.

The runner writes image artifacts and `metadata.json`. Keep outputs private for personal photos. Offline pixels and timing are not physical headset evidence.

## Build and Verification

Run from the repository root on Windows. Use the compiler in [rust-toolchain.toml](../../rust-toolchain.toml) and Windows build tools. Cross-language tests need the locked Python development environment. A short target path avoids MSBuild path limits.

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

Graphics tests use the executable-relative bundled-font path shown above; release-mode tests need `release/deps` instead of `debug/deps`. When staging elsewhere, keep the trusted OpenVR DLL and `.sha256` sidecar beside the executable.

The [Overlay CI job](../../.github/workflows/pr-ci.yml) owns the Rust and Python integration check selection.

## Source Map

| Question | Read |
| --- | --- |
| Caption policy | [Presenter](../../src/puripuly_heart/core/overlay/presenter.py) |
| Caption process and session ownership | [Python runtime](../../src/puripuly_heart/core/runtime/overlay.py), [bridge](../../src/puripuly_heart/core/overlay/bridge.py) |
| Native caption rendering | [Runtime](src/runtime.rs), [renderer](src/renderer/) |
| Photo requests and supervision | [Application owner](../../src/puripuly_heart/app/services/photo_application.py), [lifecycle](../../src/puripuly_heart/core/photo/lifecycle.py) |
| Native photo processing | [Runtime](src/photo/runtime.rs), [image pipeline](src/photo/worker.rs) |
