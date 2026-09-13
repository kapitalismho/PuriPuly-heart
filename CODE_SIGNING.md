# Code signing policy

**Draft for maintainer review.** PuriPuly has not applied to or been accepted by SignPath Foundation. The current official installer is unsigned. The signing requirements below describe the proposed signed-release process, not a claim about existing releases.

## Official source and downloads

- Source: [kapitalismho/PuriPuly-heart](https://github.com/kapitalismho/PuriPuly-heart)
- Downloads: [GitHub Releases](https://github.com/kapitalismho/PuriPuly-heart/releases)
- Product name: `PuriPuly <3`

Code signing will authenticate release artifacts and their build origin. It does not guarantee that software is bug-free or that Windows or antivirus products will show no warnings.

## Responsibilities

The maintainer approved these responsibilities:

| Role | Member | Responsibility |
| --- | --- | --- |
| Author | [@kapitalismho](https://github.com/kapitalismho) | Maintain source, dependencies, and build scripts. |
| Reviewer | [@kapitalismho](https://github.com/kapitalismho) | Review contributions from non-committers, including changes to build and signing rules. |
| Approver | [@kapitalismho](https://github.com/kapitalismho) | Review the originating build and personally approve every signing request. |

The same person may author, review, and approve a release. There is no second-person approval requirement. Signing is blocked while the sole Approver is unavailable. Existing repository review rules still apply.

GitHub and SignPath access must use MFA. SignPath account configuration and MFA verification are pending; these role descriptions do not establish either.

## Signing scope

The proposed signing scope is:

- `PuriPulyHeart.exe`
- `PuriPulyHeartOverlay.exe`
- `PuriPulyHeartGpuWorker.exe`
- `PuriPulyHeart-Setup-<version>.exe`
- The generated Inno Setup uninstaller, installed as `unins*.exe`

An explicit target list will select these artifacts. Wildcard signing of every executable in the package is not permitted. New signing targets require ownership, license, provenance, and metadata review before inclusion.

Untouched upstream EXEs, DLLs, PYDs, and other runtime binaries are outside the PuriPuly signing identity. Their existing signatures must be preserved. Inclusion inside a signed installer does not mean those components were individually signed by PuriPuly.

All targets must use the product name `PuriPuly <3` and the same release product version. Certificate publisher identity is separate from product name.

## Signed-release requirements

The existing GitHub-hosted, tag-triggered build and human-published draft-release process will be retained.

1. Build the project binaries and generated uninstaller from the intended source revision. Finish all byte-changing packaging steps.
2. Upload those exact candidates as a GitHub workflow artifact and request signing with verified build provenance.
3. Obtain the Approver's manual approval. Retrieve and validate the signed inner artifacts.
4. Build the installer from those exact signed bytes, submit its workflow artifact for signing, and obtain a second manual approval.
5. Validate the installer and installed signing targets, then attach only the validated signed installer to a draft release.
6. Publish only by explicit maintainer action.

Two signing requests are the planned sequence. The generated-uninstaller mechanism has been tested locally, but Foundation integration and a real signed candidate remain unverified. Any required change to this sequence needs maintainer approval.

Missing artifacts, rejected or failed signing, invalid signatures, wrong publisher identity, inconsistent metadata, or broken provenance must fail the signed-release path. An unsigned artifact must never be substituted as a successful signed release. Verification must cover the certificate chain, applicable timestamp, expected publisher, product metadata, and installed inner artifacts—not just signature presence on the outer installer.

Evidence must connect the release tag, source SHA, workflow run, submitted artifacts, signing requests, validated outputs, and release asset names. A successful signing request does not authorize publication.

## Privacy

Anonymous app-usage telemetry defaults to enabled. The prepared installer displays the complete Privacy Policy offline. On a new installation, agreement is selected by default; choosing not to agree disables telemetry before first launch without preventing installation. Upgrades preserve the existing preference unless the user changes it, and the preference remains available in Settings. Default-selected agreement is an opt-out preference, not affirmative opt-in.

See the [Privacy Policy](PRIVACY.md), also included in the installer in English, Korean, Japanese, Simplified Chinese, and Traditional Chinese. Disabling telemetry does not disable all network features: automatic update checks, model downloads, authentication, and selected external services are described separately in that policy. This describes the prepared implementation, not the existing published installer. Foundation assessment of these flows and acceptance of the project remain pending.

## Foundation attribution

Foundation acceptance and signing service provision are pending. After acceptance, the approved public policy will include the required attribution:

> Free code signing provided by [SignPath.io](https://about.signpath.io), certificate by [SignPath Foundation](https://signpath.org)

This draft does not claim that the service is already provided to PuriPuly.

## Unsigned artifacts

Existing releases predate this proposed signing process and may be unsigned. Local development builds and CI artifacts are not validated signed releases. No new official nightly distribution is introduced by this policy. After the signed-release cutover, failed signing must not cause an unsigned installer to be presented as the official release download.
