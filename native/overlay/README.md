# Native Overlay

Windows Rust runtime for the VR subtitle overlay.

## Ownership

- Rust implementation: `native/overlay/src`
- Rust tests: `native/overlay/tests`
- Python protocol and process integration:
  - `src/puripuly_heart/core/overlay`
  - `src/puripuly_heart/core/runtime/overlay.py`
  - `src/puripuly_heart/ui/desktop_overlay.py`

Run commands from the repository root.

## Verification

```powershell
cargo test --manifest-path native/overlay/Cargo.toml -q
cargo build --manifest-path native/overlay/Cargo.toml --locked --release --bin PuriPulyHeartOverlay --target-dir target

New-Item -ItemType Directory -Force -Path build/overlay | Out-Null
Copy-Item target/release/PuriPulyHeartOverlay.exe build/overlay/PuriPulyHeartOverlay.exe -Force
Copy-Item third_party/openvr/win64/openvr_api.dll build/overlay/openvr_api.dll -Force

.\build\overlay\PuriPulyHeartOverlay.exe --check-startup-contract

```

Shared protocol or startup changes also run:

```powershell
python -m pytest tests/core/test_overlay_protocol.py tests/core/test_overlay_manifest.py tests/app/test_desktop_overlay_runner.py

```

## Completion

Overlay behavior, protocol, or startup changes complete with:

1. Rust tests
2. Windows release build
3. Runtime assembly
4. Startup-contract verification
5. Python integration tests when the shared boundary changes

