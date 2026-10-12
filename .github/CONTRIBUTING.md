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

Pushing a synchronized `vMAJOR.MINOR.PATCH` tag runs the [GitHub-hosted native release workflow](workflows/release-native.yml). Production execution is fail-closed until its explicit cutover prerequisites are configured. The workflow prepares four explicit component targets, submits the components and rebuilt installer through two separately manually approved SignPath v3 artifact-ID requests, validates the outer installer and installed targets, and stages only the validated production-signed installer in a draft. Publication remains a separate maintainer action. Test-signing dispatches create clearly named Actions candidates/evidence and never a draft. The legacy release workflow has been removed; `native-v*` tags no longer trigger a release.

The native Windows shell disables optional CMake JNI discovery before configuring Flutter plugins. Android-only transitive dependencies must not introduce a Java runtime dependency based on the build machine's installed JDK. Native DLL dependency closure validation remains enforced.

The draft-release job installs the pinned `PYTHON_VERSION` before running release helpers. These helpers use the project's Python syntax and must not run with the Ubuntu image's default interpreter.

SHA-256, Authenticode chain/publisher/timestamp, metadata, signed-byte handoff and provenance validation remain inside the pipeline. Durable Actions evidence links the tag, source SHA, workflow run and attempt, both uploaded artifact IDs and signing requests, validated outputs, and production draft asset ID/URL/digest. Candidate and evidence transfer between jobs uses returned artifact IDs. Required third-party licenses and corresponding-source bundles remain embedded in the installer.

Before creating a tag, commit matching versions in:

- `pyproject.toml`
- `src/puripuly_heart/__init__.py`
- `installer.iss`
- `native/overlay/Cargo.toml`
- `native/gpu_worker/Cargo.toml`

Keep the root package versions in `uv.lock`, `native/overlay/Cargo.lock`, and `native/gpu_worker/Cargo.lock` synchronized too.

Tag the commit containing the workflow changes and version updates. For the initial authorized test probe:

```sh
git tag v2.8.3
git push origin v2.8.3
gh workflow run release-native.yml --ref v2.8.3 -f tag=v2.8.3 -f test_signing=true
```

The tag push takes the production path and remains blocked until production cutover prerequisites are satisfied; the explicit dispatch above selects the test-only path. Prepare the independently approved test certificate variables and verify the SignPath artifact configurations first. Both test requests still need manual approval. See [Code signing policy](../CODE_SIGNING.md) for exact settings and the remaining production requirements.

To rerun an existing tag, select that exact tag as the **Native Release → Run workflow** ref and enter the same tag input. Branch-ref dispatches and mismatched workflow/source revisions are rejected. Select `test_signing=true` for a test-only candidate. Reruns do not pick up newer branch commits: use a new synchronized immutable version tag for source fixes, and never move a published tag. Historical unsigned releases must not be represented as signed by this workflow.

