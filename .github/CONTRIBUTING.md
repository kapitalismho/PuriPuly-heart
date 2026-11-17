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

Maintained repository tools are tracked under [`scripts/ci/`](../scripts/ci/) and [`scripts/release/`](../scripts/release/); reusable ASR benchmarks live under [`benchmarks/asr/`](../benchmarks/asr/README.md). These tool directories do not need ignore-rule allowlists. Put new disposable or personal scripts in ignored `scripts/local/` or `.tmp/`, never credentials in maintained tools. Existing root-level personal scripts remain individually ignored until their ownership is resolved. See the [workspace cleanup record](../docs/workspace-cleanup.md) for preservation boundaries and retired experiment commands.

Cloud services are sibling projects at [`services/broker/`](../services/broker/README.md) and [`services/agent-relay/`](../services/agent-relay/README.md), outside the Python package. The development-only [`benchmarks/chatbot/`](../benchmarks/chatbot/README.md) owns chatbot measurements. Follow the service guides when preserving ignored Wrangler state and regenerating workspace dependency links after a checkout layout change.

Application-owned translation templates, target-language rules, and language-pair examples live in [`src/puripuly_heart/data/prompts/`](../src/puripuly_heart/data/prompts/). Keep prompt contents and provider/language selection intact during resource-only changes. [`#70`](https://github.com/kapitalismho/PuriPuly-heart/issues/70) and [`#71`](https://github.com/kapitalismho/PuriPuly-heart/issues/71) remain open for rule and example contributions. `PURIPULY_HEART_PROMPTS_DIR` still selects an external prompt directory; saved user overrides remain user data, not bundled assets.

Native upgrade cleanup removes old `app/prompts/` defaults only when their size and hash match the packaged resources. Modified or unrecognized old files and root user `prompts/` are preserved; retained files are not implicit runtime fallbacks.

The [HTTP extension guide](../docs/http-extensions/index.md), [schema](../docs/http-extensions/http-extension.schema.json), and [example](../docs/http-extensions/examples/mymemory.json) are grouped under `docs/http-extensions/`. The native build copies that example to the unchanged installed `examples/http_extensions/` location. User extension discovery stays in the user's `http_extensions` directory.

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
