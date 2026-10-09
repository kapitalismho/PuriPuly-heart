# Contributing to PuriPuly Heart

Thanks for taking an interest in this project.

## How to Get Started

1. Follow the Development section in the [README](../README.md).
2. Find something to work on, e.g., issues labeled `good first issue`.
3. See [Making a Change](#making-a-change) below.

## Where to Get Help

- Post in the issues
- DM me on Twitter

## Making a Change

1. Create a branch with a descriptive name.
2. Do the work you'd like to do. If the change is too big, please discuss it with me in an issue first.
3. Make sure the CI tests pass.
4. Open a pull request against `dev`.

## Releases

Pushing a `vMAJOR.MINOR.PATCH` tag, such as `v2.8.2`, runs the [native release workflow](workflows/release-native.yml). It builds and verifies the native Windows installer, uploads only `PuriPulyHeart-Setup-MAJOR.MINOR.PATCH.exe` as the Actions artifact, and creates a GitHub draft release containing that installer alone. The legacy release workflow has been removed; `native-v*` tags no longer trigger a release.

The native Windows shell disables optional CMake JNI discovery before configuring Flutter plugins. Android-only transitive dependencies must not introduce a Java runtime dependency based on the build machine's installed JDK. Native DLL dependency closure validation remains enforced.

The draft-release job installs the pinned `PYTHON_VERSION` before running release helpers. These helpers use the project's Python syntax and must not run with the Ubuntu image's default interpreter.

SHA-256 and provenance verification remain inside the pipeline; provenance passes between jobs through job outputs rather than downloadable JSON files. Required third-party licenses and corresponding-source bundles remain embedded in the installer. Successful reruns remove non-installer attachments from the selected release.

Before creating a tag, commit matching versions in:

- `pyproject.toml`
- `src/puripuly_heart/__init__.py`
- `installer.iss`
- `native/overlay/Cargo.toml`
- `native/gpu_worker/Cargo.toml`

Tag the commit containing the workflow changes and version updates, then push that tag:

```sh
git tag v2.8.2
git push origin v2.8.2
```

To rerun an existing release tag manually, use **Native Release → Run workflow** and enter the same `vMAJOR.MINOR.PATCH` tag. The workflow checks out that tag and rejects mismatched versions or source commits. Rerunning does not pick up newer branch commits: commit source fixes and update an unpublished tag before retrying, or create a new version tag. Do not move a tag whose release has already been published.

