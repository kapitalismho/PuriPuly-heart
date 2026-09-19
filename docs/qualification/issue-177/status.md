# Issue 177 migration status

Authority: [issue #177](https://github.com/kapitalismho/PuriPuly-heart/issues/177). Implementation was requested on 2026-09-19. No merge, push, publication, deployment, release, paid-call, credential-access, or native-adoption approval is implied.

## Candidate chain

| Candidate | Source identity | Meaning |
| --- | --- | --- |
| A | `be59afc041122eaa0a1da41d807de454140023c6` | Executed Python 3.12.10 / Flet 0.86.1 reference, not a fully qualified release. |
| B | `ba44ae1201935fc4d66cbba05589b5fea440310d` | Python 3.12.10 / Flet 1.0.0 checkpoint; independently reviewed, evidence repairs verified, accepted for Flet compatibility. |
| C software candidate | `107ed1820005872561c98809c0d4734fe8374cfd` (product source `651f0c3fecc1451f5511c0f3b098a117d2fada9c`) | Ordinary-GIL Python 3.14.7 / Flet 1.0.0 / PyInstaller implementation and repaired Windows qualification. Independently reviewed software; complete release-capable C acceptance remains blocked. |
| D preparatory probe | Pinned upstream tuple in `native-bootstrap-probe.md` | Isolated real native GUI and standalone headless API experiments, not the PuriPuly native product package. |

The original issue planning baseline was `9da5badeffde193bb26445bc18633284ab22a56e`. Existing changes through `be59afc` were preserved rather than reverting to the older behavior.

## Acceptance ledger

| Criterion | Current disposition | Evidence or remaining requirement |
| --- | --- | --- |
| A — Reference | Recorded; incomplete | `windows-baseline-be59afc.md` and `native-workload-reference.md`. Source/package execution, isolated install lifecycle, public CPU models, downloads and process capture were exercised. Historical packaged evidence-command failures and a viewer surviving forced host loss remain recorded, not rewritten as successes. Physical/VR and published-upgrade evidence are absent. |
| B — Flet checkpoint | Accepted at the reviewed B commit | `flet-checkpoint.md`. Full review findings on installed capture attribution, process accounting and dependency inventory were repaired and independently verified. Product semantics, private hooks, actual GUI/overlay and isolated installation were checked. This does not satisfy later C–F obligations. |
| C — Python checkpoint | Implemented and exercised; not release-qualified | `python-checkpoint.md`, `native-evidence-repair.md`, `shutdown-qualification.md`. The final full suite passed 6,046 tests with 37 skips. Actual packaged CPU/GPU/Xet, worker recovery/release, main/overlay GUI, custom soxr, native identities, provenance and isolated installation were exercised. Source adversarial shutdown and final packaged repeated-close/host-loss checks passed. Physical and published-upgrade qualification remain absent. |
| Boundary contract | Source/PyInstaller implemented; native product unproven | `RuntimeLayout` consolidates execution/resource paths and preserves writable roots. Source and PyInstaller Unicode/unrelated-CWD checks were run. The native probe is not proof of product GUI/headless/overlay integration. |
| D — Native proof | Implementation and independent probes resumed; acceptance blocked | `native-bootstrap-probe.md`. Standalone headless Unicode/exit 0/23 and a minimal real native GUI were exercised, not the product. Stock GUI startup fails with poisoned Python environment variables. The native overlay's GUI command no longer includes `--headless`; real product-host execution remains unproved. |
| E — Decision | Blocked; no adoption/rejection decision | There is no matched qualified C-versus-product-D comparison. Missing qualification is not evidence that native packaging failed. Native adoption and material tradeoffs remain the maintainer's decision. |
| F — Conditional cutover | Not authorized; pending E, not marked complete or inapplicable | PyInstaller and required viewer glue remain. Inno product identity is unchanged. No dual production packaging path was introduced and no release was performed. |
| Integrated acceptance | Blocked on remaining Goal gates | Real assembled C software/native-owner checks and independent review were performed. They do not substitute for missing physical/upgrade evidence, product-native qualification, or the maintainer's E decision. |
| Completion record | Current execution recorded; final disposition blocked | This ledger and linked readable evidence retain exact candidates/artifacts, test dispositions, failures, limitations, review outcomes and separate release status. The required final maintainer disposition cannot be recorded before E. |

## Remaining execution/access boundaries

- Published-release upgrade and rollback need a disposable Windows installation with the original AppId. The current machine's production AppId is occupied. Windows Sandbox, VirtualBox, VMware execution tools, `Get-VM`, and the Hyper-V management service were not found in the inspected installed locations/commands. An alternate-AppId same-build reinstall is not a published-version upgrade.
- Physical microphone/device-loopback and sustained VRChat/SteamVR/HMD behavior require a controlled lawful session. Public recorded speech, generated process-isolated tones and model inference provide narrower evidence only.
- Live cloud qualification requires existing authorized credentials and any necessary paid-call approval. None was accessed or solicited during this work.
- R5 local checks now include actual composed initialization/repeated close, controlled download-child/hang scenarios, real GPU inference in flight during close, and live production owner/generation/native/download-child diagnostics. Final rebuilt-C GUI repeated close and actual host-loss containment passed with PID/image/ancestry accounting including cached Flet viewers; graceful close and forced-host containment are recorded separately in `shutdown-qualification.md`.
- Native product GUI/headless shared-distribution integration, overlay, shutdown, provenance, installer lifecycle and matched resource/performance comparison remain unqualified. The small experiments add no new production service or bridge fork.
- Lifecycle finding F1 concerned `DesktopFletOverlayRunner.build_command()`: a native desktop renderer was incorrectly sent through `--headless`. The bounded repair removes that flag for GUI overlay dispatch while preserving source/PyInstaller commands and the true Xet headless path. Its native-command regression failed before repair and passed afterward; related runner/layout/compatibility checks reported 9 passed and 2 integration skips. This is command-contract evidence, not native renderer execution or D qualification.

## Evidence and test disposition

- CPU evidence no longer parses removed human log wording; the obsolete log-format fixture was deleted. The real backend final, identities, timing and errors are the evidence contract.
- GPU evidence opens actual lazy sessions, uses generation-scoped requests after recovery, and tests worker sharing/retention/release against observable production state. Stale eager-start assumptions were removed rather than changing provider policy.
- Obsolete Flet test doubles were migrated to the supported API. Intentionally sync/async application callbacks remain supported.
- The obsolete ProcTap source-text collection assertion was removed; runtime payload and native-process capture results are retained.
- The UI boundary's manually pinned method inventory and `__wrapped__` assertion were removed. Its replacement checks observable behavior: diagnostics remain callable during frozen shutdown while mutating intents are rejected.
- `release.yml` now defines packaged headless/process-capture gates and binds the shipped overlay. Actual local implementations of the license, compliance and release-identity gates passed against the final payload. No GitHub Actions workflow ran; separate installer/native/manual checks are not represented as hosted CI evidence.
- Shutdown timeout diagnostics now traverse the production UI/application boundary and existing runtime logger; the report is bounded, excludes arbitrary owner data, and records coordinator state/terminal/failure count plus `native_stack_available=false`.
- Scratch probes, large model/audio fixtures, build outputs and isolated environments are not production scaffolding. They remain outside tracked product code for reproducibility; no worktree/environment cleanup was authorized.

## Review disposition

The first C review found a committed-license newline mismatch, missing final-package soxr compliance staging, absent C-specific download evidence, a stale source snapshot, and an overlay provenance path gap. These were repaired and re-exercised from committed source. Lifecycle review also found unwired stall diagnostics, an async callback test double that failed in a detached task, and missing CPU failure reports; the repaired production integration and executable evidence are recorded in the linked qualification files.

The real GPU handoff observed the current sink; retired-sink behavior is covered by a behavioral test, not claimed as a real native observation. No new requirement to force that branch was inferred. B's identical static screenshots were not treated as proof of fabrication; the actual attribution gap was closed by a fresh PID/path-bound installed capture.

Distribution repair verification at `887b6cb1` closed its five findings. Fresh lifecycle review of the same software found no C behavior failure. The retained-probe evidence gap was repaired with independently assessable reports, raw outputs, identities and hashes in `shutdown-qualification.md`; targeted verification at `03d2c7af` closed F2. F1, previously deferred to D, is now under active repair and native-boundary verification. Unchanged historical software/package evidence remains applicable to its recorded candidates, not automatically to later builds. These software reviews do not fulfill the external qualification or maintainer-decision gates.

An execution-isolation mistake is documented in `python-checkpoint.md`: one telemetry command omitted `--config` and persisted the real user's preference as disabled. The user was informed and explicitly requested enablement; the Director applied it with the exact configuration path and the command's persist/reload verification returned 0. Previous values were not recorded, so no claim of restoring the previous anonymous identity or byte-identical configuration is made. Some no-config diagnostic commands may also have appended the production log. These execution incidents are not described as isolated runs.

The user's subsequent request resumed the repair workflow. Re-reading R1 corrected an overly broad scheduling restriction: D acceptance depends on the qualified comparator, but independent repairs and isolated host experiments need not wait for unrelated external checks. No acceptance criterion was waived. The user separately authorized a local merge of the latest `dev` changes after preserving the bounded repair; that authorization does not include push, publication or release.

## Architecture and release disposition

The application/core/provider owners, native workers, protocols and Inno authority remain in place. The intended architectural change is a small immutable execution/layout boundary, plus on-demand shutdown diagnostics on the existing shutdown owner. No backend service, global process supervisor, settings-schema migration, user-data relocation or native packaging cutover is authorized by this record. Independent review must assess implementation against that boundary.

No issue closure, Project completion, push, PR publication, merge or release has occurred. The issue remains in progress. A release rollback is unproven; the recorded isolated reinstall repairs only known application-owned payload under a distinct test identity.
