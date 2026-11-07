# Vendored OpenVR runtime DLL

This directory vendors the Windows x64 OpenVR client runtime DLL bundled with PuriPuly Heart.

- Upstream pin: `ValveSoftware/openvr@v2.15.6`
- DLL source: `https://raw.githubusercontent.com/ValveSoftware/openvr/v2.15.6/bin/win64/openvr_api.dll`
- LICENSE source: `https://raw.githubusercontent.com/ValveSoftware/openvr/v2.15.6/LICENSE`
- Pinned SHA256: `bab8ac6ef64e68a9ca53315b0014d131088584b2efdfa6db511d67ec03cfcb4a`

`scripts/ci/build-native-experimental.ps1` stages this vendored bundle in the native Windows payload.
`native_distribution` validates its pinned identity and compliance provenance.
