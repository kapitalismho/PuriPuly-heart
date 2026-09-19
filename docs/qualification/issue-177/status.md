# Issue 177 migration status

Authority: [issue #177](https://github.com/kapitalismho/PuriPuly-heart/issues/177). Implementation was requested on 2026-09-19. No merge, push, publication, deployment, release, paid-call, credential-access, or native-adoption approval is implied.

## Candidate chain

| Candidate | Source identity | Meaning |
| --- | --- | --- |
| A | `be59afc041122eaa0a1da41d807de454140023c6` | Executed Python 3.12.10 / Flet 0.86.1 reference, not a fully qualified release. |
| B | `ba44ae1201935fc4d66cbba05589b5fea440310d` | Python 3.12.10 / Flet 1.0.0 checkpoint; independently reviewed, evidence repairs verified, accepted for Flet compatibility. |
| C software candidate | `6e5aa754ca022ca171ae6ceb3a37e0b78d13f3cb` | Ordinary-GIL Python 3.14.7 / Flet 1.0.0 / PyInstaller implementation and executed Windows evidence; independent review and complete release-capable acceptance remain separate gates. |
| D preparatory probe | Pinned upstream tuple in `native-bootstrap-probe.md` | Isolated real native GUI and standalone headless API experiments, not the PuriPuly native product package. |

The original issue planning baseline was `9da5badeffde193bb26445bc18633284ab22a56e`. Existing changes through `be59afc` were preserved rather than reverting to the older behavior.

## Acceptance ledger

| Criterion | Current disposition | Evidence or remaining requirement |
| --- | --- | --- |
| A — Reference | Recorded; incomplete | `windows-baseline-be59afc.md` and `native-workload-reference.md`. Source/package execution, isolated install lifecycle, public CPU models, downloads and process capture were exercised. Historical packaged evidence-command failures and a viewer surviving forced host loss remain recorded, not rewritten as successes. Physical/VR and published-upgrade evidence are absent. |
| B — Flet checkpoint | Accepted at the reviewed B commit | `flet-checkpoint.md`. Full review findings on installed capture attribution, process accounting and dependency inventory were repaired and independently verified. Product semantics, private hooks, actual GUI/overlay and isolated installation were checked. This does not satisfy later C–F obligations. |
| C — Python checkpoint | Implemented and exercised; not release-qualified | `python-checkpoint.md`, `native-evidence-repair.md`. The definitive full suite passed 6,042 tests with 37 skips. Actual packaged CPU/GPU owners, worker recovery and release, GUI/overlay, custom soxr, DLLs and isolated installation were exercised. Full issue acceptance still requires all applicable R5–R8 evidence, including the missing physical/upgrade qualification. |
| Boundary contract | Source/PyInstaller implemented; native product unproven | `RuntimeLayout` consolidates execution/resource paths and preserves writable roots. Source and PyInstaller Unicode/unrelated-CWD checks were run. The native probe is not proof of product GUI/headless/overlay integration. |
| D — Native proof | Preparatory proof only; blocked on qualified C and product integration | `native-bootstrap-probe.md`. The standalone API preserves Unicode arguments and exit 0/23 without Flutter. The real minimal native GUI works under a clean environment but stock startup fails with poisoned `PYTHONHOME`/`PYTHONPATH`. No integrated product-native artifact is qualified. |
| E — Decision | Blocked; no adoption/rejection decision | There is no matched qualified C-versus-product-D comparison. Missing qualification is not evidence that native packaging failed. Native adoption and material tradeoffs remain the maintainer's decision. |
| F — Conditional cutover | Not authorized; pending E, not marked complete or inapplicable | PyInstaller and required viewer glue remain. Inno product identity is unchanged. No dual production packaging path was introduced and no release was performed. |
| Integrated acceptance | Open | Passing unit tests, imports, isolated host probes or individual native checks do not stand in for the complete assembled-runtime acceptance matrix. |
| Completion record | In progress | This ledger and the linked readable evidence retain exact candidate/artifact identities, failures, limitations, and separate release status. A final maintainer disposition cannot be recorded before E. |

## Remaining execution/access boundaries

- Published-release upgrade and rollback need a disposable Windows installation with the original AppId. The current machine's production AppId is occupied. Windows Sandbox, VirtualBox, VMware execution tools, `Get-VM`, and the Hyper-V management service were not found in the inspected installed locations/commands. An alternate-AppId same-build reinstall is not a published-version upgrade.
- Physical microphone/device-loopback and sustained VRChat/SteamVR/HMD behavior require a controlled lawful session. Public recorded speech, generated process-isolated tones and model inference provide narrower evidence only.
- Live cloud qualification requires existing authorized credentials and any necessary paid-call approval. None was accessed or solicited during this work.
- R5 source shutdown repair and locally executable adversarial evidence are recorded in `shutdown-qualification.md`: live composed owner/generation/native/download-child diagnostics, repeated real-application close, active initialization/download close, child hang, and PID/image/ancestry-bound host-loss containment passed. The pre-repair package does not satisfy these checks; final rebuilt-C packaged normal/repeated close and host-loss accounting remain pending.
- Native product GUI/headless shared-distribution integration, overlay, shutdown, provenance, installer lifecycle and matched resource/performance comparison remain unqualified. The small experiments add no new production service or bridge fork.

## Evidence and test disposition

- CPU evidence no longer parses removed human log wording; the obsolete log-format fixture was deleted. The real backend final, identities, timing and errors are the evidence contract.
- GPU evidence opens actual lazy sessions, uses generation-scoped requests after recovery, and tests worker sharing/retention/release against observable production state. Stale eager-start assumptions were removed rather than changing provider policy.
- Obsolete Flet test doubles were migrated to the supported API. Intentionally sync/async application callbacks remain supported.
- The obsolete ProcTap source-text collection assertion was removed; runtime payload and native-process capture results are retained.
- Workflow-level packaged headless/process-capture gates now run in `release.yml`; separate installer/native/manual evidence is not credited to that workflow merely because a script exists.
- Shutdown timeout diagnostics now traverse the production UI/application boundary and existing runtime logger; the report is bounded, excludes arbitrary owner data, and records coordinator state/terminal/failure count plus `native_stack_available=false`.
- Scratch probes, large model/audio fixtures, build outputs and isolated environments are not production scaffolding. They remain outside tracked product code for reproducibility; no worktree/environment cleanup was authorized.

## Architecture and release disposition

The application/core/provider owners, native workers, protocols and Inno authority remain in place. The intended architectural change is a small immutable execution/layout boundary, plus on-demand shutdown diagnostics on the existing shutdown owner. No backend service, global process supervisor, settings-schema migration, user-data relocation or native packaging cutover is authorized by this record. Independent review must assess implementation against that boundary.

No issue closure, Project completion, push, PR publication, merge or release has occurred. The issue remains in progress. A release rollback is unproven; the recorded isolated reinstall repairs only known application-owned payload under a distinct test identity.
