# Contributing to PuriPuly Heart

Start with [Development](../README.md#development) for setup and run commands. Use the [architecture](../docs/architecture.md) for system boundaries and [issues](https://github.com/kapitalismho/PuriPuly-heart/issues) for questions and proposals.

## Making a Change

1. Discuss large or cross-component changes in an issue first.
2. Create a descriptive branch and keep the change focused.
3. Update affected tests and documentation; run the relevant checks.
4. Open a pull request against `dev` with the scope, verification evidence, and known limits.

## PR verification

The [PR workflow](workflows/pr-ci.yml) owns check commands, tool versions, and test selections. New tests must belong to a declared selection. Use `--locked` for verification; do not regenerate `uv.lock` during checks.

Test observable behavior with isolated state. External integrations require their own setup. Synthetic checks do not establish real device, model, or installed-product behavior; report what actually ran and any verification gaps.

## Native ASR worker builds

Use CMake 3.19 or newer, the Windows Vulkan SDK, and the supported build entry point:

```powershell
$env:VULKAN_SDK = "C:\VulkanSDK\<installed-version>"
pwsh -File scripts/ci/build-asr-worker-release.ps1 -TargetDir build/asr-worker-cargo -OutputDir build/asr_worker
```

This entry point applies the required upstream patches. Review the [Fun-ASR language overlay](../native/asr_worker/cmake/funasr-language.cmake) when changing the transcribe.cpp pin; do not bypass it with a plain Cargo build.

## Releases

Pushing a `vMAJOR.MINOR.PATCH` tag runs [Native Release](workflows/release-native.yml), producing a Windows installer and a draft release containing that installer.

Before tagging, synchronize versions in `pyproject.toml`, `src/puripuly_heart/__init__.py`, `installer/application-identity.iss`, `native/overlay/Cargo.toml`, and `native/asr_worker/Cargo.toml`.

Manual reruns use the selected tag's commit, not newer branch changes. Never move a published release tag; ship fixes under a new version.
