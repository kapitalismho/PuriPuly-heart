# Code signing policy

The SignPath Foundation project was accepted. Hosted TEST-only signing passed at signing-branch revision `4358383ea319d26807f7dd435d66cefbd27769f7` ([#225 evidence](https://github.com/kapitalismho/PuriPuly-heart/issues/225#issuecomment-6157644990)); this does not qualify the current integrated source or production trust.
The integrated Native Release workflow does not yet perform SignPath signing. Production signing remains unready, and test-signed candidates are not production releases.

Free code signing provided by SignPath.io, certificate by SignPath Foundation.

## Official source and downloads

- Source: [kapitalismho/PuriPuly-heart](https://github.com/kapitalismho/PuriPuly-heart)
- Downloads: [GitHub Releases](https://github.com/kapitalismho/PuriPuly-heart/releases)
- Product name: `PuriPuly <3`

Signing verifies artifact identity and build origin. It does not guarantee bug-free software or prevent Windows or antivirus warnings.

## Responsibilities

Approved responsibilities:

| Role | Member | Responsibility |
| --- | --- | --- |
| Author | [@kapitalismho](https://github.com/kapitalismho) | Maintain source, dependencies, and build scripts. |
| Reviewer | [@kapitalismho](https://github.com/kapitalismho) | Review contributions from non-committers, including changes to build and signing rules. |
| Approver | [@kapitalismho](https://github.com/kapitalismho) | Review the build and manually approve every signing request. |

One person holds all three roles; no second approver is required. Signing waits if the Approver is unavailable. Repository review rules still apply.

GitHub and SignPath access must use MFA. The accepted test setup does not establish production account/MFA readiness. Actual production certificate, valid policy, narrowly approved origin/build restrictions and public signer/root evidence remain prerequisites; the last published portal snapshot is historical, not a fresh current-state check.

## Signing scope

- `PuriPulyHeart.exe`
- `PuriPulyHeartOverlay.exe`
- `PuriPulyHeartGpuWorker.exe`
- `PuriPulyHeart-Setup-<version>.exe`
- The generated Inno Setup uninstaller, installed as `unins*.exe`

Sign only these explicit targets, never all packaged executables by wildcard. Additional targets require ownership, license, provenance, and metadata review.

Do not re-sign untouched third-party binaries. Preserve their signatures. Packaging them in a signed installer does not individually sign them.

All targets must use `PuriPuly <3` and the same release product version. Certificate publisher identity is separate.

## Release process

Retain GitHub-hosted, tag-triggered builds and manually published draft releases.

1. Build project binaries and the uninstaller from the intended revision. Finish byte-changing packaging.
2. Upload those exact files as a workflow artifact and request signing with verified build provenance.
3. Obtain manual approval, then retrieve and validate the signed files.
4. Build the installer from those signed bytes. Submit it as a workflow artifact for a second manual signing approval.
5. Validate the installer and installed signing targets. Attach only the validated signed installer to a draft release.
6. Publish only by explicit maintainer action.

Changing this two-request sequence requires maintainer approval. The historical signing-branch TEST run completed both manually approved requests, exact signed-byte rebuild, actual installation, installed-target validation and uninstall. It used an approved test certificate, not production trust, and does not qualify a changed integrated candidate. See [the TEST receipt and production handoff](https://github.com/kapitalismho/PuriPuly-heart/issues/225#issuecomment-6159370038).

Missing artifacts, signing rejection or failure, or invalid signatures, publisher, metadata, or provenance must block the signed release. Never substitute unsigned files. Verify the certificate chain, applicable timestamp, expected publisher, product metadata, and installed inner artifacts.

Link the tag, source SHA, workflow run, submitted artifacts, signing requests, validated outputs, and release asset names. Signing approval is not publication approval.

## Privacy

The prepared installer displays the full [Privacy Policy](PRIVACY.md) offline in English, Korean, Japanese, Simplified Chinese, and Traditional Chinese.

Usage statistics default to ON, with agreement preselected. Declining turns statistics OFF before first launch without blocking installation. Upgrades retain the choice unless changed; Settings also allows changes. This is opt-out, not affirmative opt-in.

Statistics OFF does not disable update checks, model downloads, authentication, or selected external services. See the Privacy Policy for these flows.

Preparation-era published installers were unsigned. Foundation project acceptance and the historical TEST result do not establish production readiness or authorize signing, deployment or publication.

## Foundation attribution

After Foundation acceptance, include:

> Free code signing provided by [SignPath.io](https://about.signpath.io), certificate by [SignPath Foundation](https://signpath.org)

## Unsigned artifacts

Existing releases may be unsigned. Local builds and CI artifacts are not validated signed releases; this policy introduces no official nightly distribution. After cutover, signing failure must block the official installer release.
