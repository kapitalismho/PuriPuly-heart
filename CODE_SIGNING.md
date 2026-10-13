# Code signing policy

SignPath has accepted the project. The native release workflow implements two manually approved signing requests, but production signing is not active: the release certificate remains `CSR PENDING` and `release-signing` is invalid.
Historical published releases remain unsigned. Repository implementation and local validation do not establish production acceptance.

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

GitHub and SignPath access involved in production signing must use MFA. Production cutover requires maintainer verification; the workflow does not infer or certify account MFA state.

## Signing scope

- `PuriPulyHeart.exe`
- `PuriPulyHeartOverlay.exe`
- `PuriPulyHeartGpuWorker.exe`
- `PuriPulyHeart-Setup-<version>.exe`
- The generated Inno Setup uninstaller, installed as `unins*.exe`

Sign only these explicit targets, never all packaged executables by wildcard. Additional targets require ownership, license, provenance, and metadata review.

Do not re-sign untouched third-party binaries. Preserve their signatures. Packaging them in a signed installer does not individually sign them.

All targets must use the same logical product identity, `PuriPuly <3`, and release product version. CI compares product names and canonical textual versions after removing trailing ASCII spaces, and checks the numeric product version. Padding width is not a CI acceptance condition.

Validate every version-resource language and string table, and preserve their raw values in signing/provenance evidence. A later table must not conceal conflicting product or version metadata. Ignoring trailing ASCII spaces does not permit a different product or version.

The stock Inno compiler emits padded metadata. Compiler binaries, templates and their original verification remain unchanged. This does not change application branding or installer UI.

## Release process

Retain GitHub-hosted, tag-triggered builds and manually published draft releases.

1. Build project binaries and the uninstaller from the intended revision. Finish byte-changing packaging.
2. Upload those exact files as a workflow artifact and request signing with verified build provenance.
3. Obtain manual approval, then retrieve and validate the signed files.
4. Build the installer from those signed bytes. Submit it as a workflow artifact for a second manual signing approval.
5. Validate the installer and installed signing targets. Attach only the validated signed installer to a draft release.
6. Publish only by explicit maintainer action.

The repository stages the three explicit project executables and generated uninstaller as `uninstaller.exe`, validates the signed results, rebuilds from those exact bytes, and requests signing of the installer. Local real-PE/Inno probes verified signing-boundary handoff and installed uninstaller reuse with an independently pinned local probe authority; these probes do not establish real SignPath or production acceptance.

Missing artifacts, signing rejection or failure, or invalid signatures, publisher, metadata, or provenance must block the signed release. Never substitute unsigned files. Verify the certificate chain, applicable timestamp, expected publisher, product metadata, and installed inner artifacts.

Link the tag, source SHA, workflow run, submitted artifacts, signing requests, validated outputs, and release asset names. Signing approval is not publication approval.

## Test probe and production cutover

Use organization `41a8602a-e07d-4d35-876f-8876d5acb31d`, project `PuriPuly-heart`, and the existing `SIGNPATH_API_TOKEN` repository secret. Never copy its value into logs, evidence, issues, or release text.

The `components` artifact configuration must select only the three named project executables and `uninstaller.exe` at the root of the uploaded artifact ZIP. The `installer` configuration must select only `PuriPulyHeart-Setup-${version}.exe` at that ZIP root. A nested ZIP or wildcard executable selection is not the approved contract.

Both configurations require only canonical `version`. Keep service-side product-name and product-version restrictions, as described in the [Foundation's published conditions](https://signpath.org/terms). The three project executables use `product-name="PuriPuly &lt;3"` and `product-version="${version}"`. Installer filenames also use canonical `${version}`.

For the generated installer and uninstaller, place the stock Inno padding directly in the XML attributes, not in a request parameter. Their product name is `PuriPuly <3` followed by 49 ASCII spaces. Their product version is `${version}` followed by enough literal ASCII spaces to make the expanded value 50 characters; XML `&#x20;` references express those spaces. A five-character version such as `2.8.3` requires 45 spaces. If the canonical version's string length changes, update both configurations before the next signing run. This is portal configuration maintenance, not a target-specific CI metadata profile.

Both requests send and validate canonical `version`; there is no separate padded-version parameter. Save and freshly read back the XML before testing a configuration change. Local schema validation does not prove that live service substitution and matching accept the resulting attributes.

Dispatch **Native Release** from the selected immutable `vMAJOR.MINOR.PATCH` tag, enter that same `tag`, and set `test_signing=true` for the first end-to-end probe. Both requests use `test-signing`; both require manual approval and all artifact, signature, metadata, provenance and installed-file checks. Successful test runs produce clearly named `test-signed-candidate-<run>-<attempt>` and `test-signing-evidence-<run>-<attempt>` Actions artifacts, never a GitHub draft release.

Configure these repository variables from independently approved public certificate material:

| Variable | Required value |
| --- | --- |
| `SIGNPATH_TEST_PUBLISHER` | Exact approved test signer certificate Subject. |
| `SIGNPATH_TEST_ROOT_SHA256` | SHA-256 of the approved test authority DER certificate. |
| `SIGNPATH_TEST_ROOT_CERTIFICATE_BASE64` | Approved test authority DER encoded as base64. |
| `SIGNPATH_RELEASE_PUBLISHER` | Exact approved SignPath Foundation signer certificate Subject. |
| `SIGNPATH_RELEASE_ROOT_SHA256` | SHA-256 of its independently verified Windows trust-root DER certificate. |
| `SIGNPATH_RELEASE_ORIGIN` | Exact observed SignPath `branchName`, optionally containing literal `{tag}`; no wildcards. |
| `SIGNPATH_PRODUCTION_READY` | Exact `true`, set only after certificate, policy, origin restriction and MFA review. |

Test certificates are not publicly trusted production certificates. The approved test authority is installed only in the isolated test runner; an independently approved self-signed test leaf may be its own authority. Never trust a certificate merely because it came back with a signed artifact, and never use local probe certificates as the project's approved authority.

Test trust initialization requires a disposable GitHub-hosted Windows runner with administrator privileges. It imports the independently pinned certificate into `LocalMachine\Root` to avoid interactive per-user root confirmation, then verifies the exact stored certificate bytes. The preflight has a five-minute timeout and reports import/readback phases; it must complete before any build or signing request. Local workstation imports are rejected, and production mode never adds a trust anchor.

Derive values from independently exported public certificates with PowerShell:

```powershell
$signer = [System.Security.Cryptography.X509Certificates.X509Certificate2]::new((Resolve-Path './approved-signer.cer').Path)
$authority = [System.Security.Cryptography.X509Certificates.X509Certificate2]::new((Resolve-Path './approved-authority.cer').Path)
$signer.Subject
[Convert]::ToHexString([System.Security.Cryptography.SHA256]::HashData($authority.RawData)).ToLowerInvariant()
[Convert]::ToBase64String($authority.RawData)
```

For an approved self-signed test leaf, both paths may refer to the same certificate. Do not export or upload a private key.

Both requests use `signpath/github-action-submit-signing-request@v3` with the exact `artifact-id` returned by their preceding `actions/upload-artifact` step. Durable Actions evidence records GitHub tag/ref, source SHA, workflow run and attempt, artifact IDs, actual SignPath request IDs and API-reported origin, hashes and validation results. Production evidence additionally links the delivered draft asset ID, URL and digest. GitHub-side tag/ref data must not be presented as SignPath-reported origin.

After the first successful test run, review the literal `origin.repositoryData.branchName` and build URL returned for both requests. Set `SIGNPATH_RELEASE_ORIGIN` to that exact tag representation (for example, `{tag}` or `refs/tags/{tag}` only if observed), and narrow the production policy's allowed-origin rule accordingly. The existing `**` rule is not the intended production state.

Before setting `SIGNPATH_PRODUCTION_READY=true`, the maintainer must verify a valid production certificate and `release-signing` policy, the narrow origin restriction, approved Foundation publisher/root values, and MFA for the involved accounts. This explicit gate does not replace signature or origin validation. A complete production-signed candidate and clean Windows install/uninstall verification remain required before production cutover can be called complete.

Missing configuration, rejected or withheld approval, timeout, malformed response or validation failure blocks the release. Only validated production-signed installers may enter a draft; publication remains a separate explicit maintainer action.

## Privacy

The prepared installer displays the full [Privacy Policy](PRIVACY.md) offline in English, Korean, Japanese, Simplified Chinese, and Traditional Chinese.

Usage statistics default to ON, with agreement preselected. Declining turns statistics OFF before first launch without blocking installation. Upgrades retain the choice unless changed; Settings also allows changes. This is opt-out, not affirmative opt-in.

Statistics OFF does not disable update checks, model downloads, authentication, or selected external services. See the Privacy Policy for these flows.

Existing published installers predate this implementation. Foundation acceptance does not establish production signing readiness.

## Foundation attribution

Foundation acceptance attribution:

> Free code signing provided by [SignPath.io](https://about.signpath.io), certificate by [SignPath Foundation](https://signpath.org)

## Unsigned artifacts

Historical releases are unsigned. Local unsigned builds and test-signed Actions candidates are not production-signed releases; this policy introduces no official nightly distribution. The official draft-release path rejects unsigned and test-signed candidates and never substitutes an unsigned installer after signing failure.
