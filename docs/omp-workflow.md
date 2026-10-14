# Local OMP verification workflow

This guide records the local instruction adaptation for [#239](https://github.com/kapitalismho/PuriPuly-heart/issues/239). It is not an OMP installer, a new orchestration layer, or a product behavior change. The owner still starts an ordinary Orca `/vibe <issue URL>` task. Director/worker/reviewer ownership, independent review, internal result contracts, and explicit publication/merge/release authority remain intact.

## Effective sources and responsibilities

The inspected installation reported `omp/18.10.4`: the `~/.bun/bin/omp.exe` launcher resolves through `omp.bunx` to `@oh-my-pi/pi-coding-agent/dist/cli.js` in Bun's global installation. The active user-managed agent directory is `~/.omp/agent`. Instruction-related environment/launch inspection found no alternate agent/profile or custom-system override. Relevant native configuration retains leaf depth 1; no relevant skill/extension suppression was found. These observations describe this installation, not universal defaults.

| Source | Responsibility and loading |
| --- | --- |
| `~/.omp/agent/vibe/director-policy.md` | Current scope/decisions, WHAT before HOW, proportionate evidence, instruction activation, and authorized GitHub knowledge. `extensions/vibe-director-tools.ts` reads it when its extension factory is instantiated and replaces the default Vibe policy context. |
| `~/.agents/skills/vibe-worker/SKILL.md` | Delegated outcome/invariant reasoning, executed tests, actual-app boundaries, safe isolation, and useful results/limits. `extensions/vibe-worker-contract.ts` directly reads it for each fresh child extension factory, appends the fixed role contract, and saves a version-2 session snapshot. |
| `~/.agents/skills/implementation-critic/SKILL.md` | Checkpoint review examines actual changes and original evidence, weakened/omitted checks, and reachable failures. Repair verification closes affected failures and legitimate progress without automatically repeating unrelated checks. Read through `skill://implementation-critic`. |
| `~/.agents/skills/implementation-reviewer/SKILL.md` | Completion audit distinguishes implementation, instruction loading, pilot execution, missing outcomes, and shared-result publication. Read through `skill://implementation-reviewer`. |
| `~/.omp/agent/prompts/vibe-{implementation-workstream,independent-review,repair-and-rereview}.md` | Existing handoff inputs now connect outcome/invariants, affected verification, observation boundaries, and explicit project guidance. The Director reads these files for assignments; slash-template contents are separately cached at startup. |
| Worktree `AGENTS.md` | PuriPuly-specific controls, isolation, observation boundaries, and shared GitHub context, alongside [architecture](architecture.md) and [CLI contracts](cli.md). |

Only these seven local text sources and project guidance were adapted. Extensions, `vibe/roles.ts`, tool restrictions, model/provider/security configuration, and distributed framework files were not changed. Existing aligned instructions and internal reviewer result formats were preserved. The fixed Project-state wording was replaced with existing-convention/permission-aware guidance, not another prescribed board state or reporting form.

### Inheritance and overrides that matter

- Fresh Vibe children receive newly rebound extension factories. Although the extension module is already imported, its factory reads the common worker file again. Existing workers instead retain the saved contract and role template, including on persistent revival; sending another message does not activate new common instructions.
- Native Vibe spawn filters inherited files named `AGENTS.md` and supplies an explicit remaining context array, bypassing normal rediscovery. Assignments therefore name the applicable project guidance and require a direct read. Parent access is not evidence of child loading.
- The primary checkout's ancestor `AGENTS.md` is also a discovery candidate for this nested worktree. Its inspected content matched the pre-change worktree guidance. The additions preserve its general constraints and clarify CLI versus GUI evidence; no conflicting ancestor rule needed editing. No further applicable global/system override or nested agent Markdown was found in the bounded inspection.
- The local Director injection already replaces distributed default Vibe instructions that would otherwise forbid direct Director work. The distributed common default forbids delegated verification unless explicitly assigned; pilot and implementation handoffs explicitly grant focused verification. No framework patch is needed.
- The named critic/reviewer skills are discoverable and `skill://` reads their current file contents. The disabled bundled `code-reviewer` agent is not the explicitly assigned Vibe reviewer role. A different hidden skill is not evidence that these review skills are disabled.
- Researchers/reviewers remain read-only, including indirect execution paths. The local extension also blocks their write-based peer/report transports; findings must use the existing return protocol, not a shell bypass. The Director can relay a tool issue when needed.

Decisive installed-source references (relative to the package root): `src/vibe/runtime.ts:1270-1335` (fresh child inputs and AGENTS filtering), `src/task/executor.ts:4388-4486` (child construction), `src/sdk.ts:2650-2686` and `src/extensibility/extensions/loader.ts:397-518` (factory rebinding), `src/internal-urls/skill-protocol.ts:127-174` (current-file skill reads), and `src/config/prompt-templates.ts:165-201` (cached template expansion). The local worker extension's lines 30-38 and 59-74 distinguish fresh source loading from saved contracts. These source reads establish the mechanism, not execution of the pilot.

## Activation and reversal

The non-secret [instruction delta](omp-local-guidance.patch) records the seven actual home-relative text changes for review and reversal; it does not copy personal configuration, credentials, or sessions into this repository. The effective maintained sources remain the installed user-managed files. Re-resolve another installation before using the patch; different source versions require integration, not forced application.

For this implementation, the seven original files were copied to `~/.omp/agent/vibe/issue-239-before/`, retaining their home-relative paths. Forward patch applicability against that baseline and reverse applicability against the edited home were checked with `git apply --check`.

To activate safely:

1. Save edits in the effective sources. Distinguish that from loading.
2. In an already running Director, explicitly read/adopt the revised policy as the issue-authorized task-local supplement and read current handoff files. This is not an automatic refresh of its cached injected policy or slash templates.
3. Start fresh workers/reviewers for assignments that need the new common contract. Have the pilot worker identify relevant inherited guidance before reading the common source or receiving quotations; then observe its ordinary work. Give explicit project-guidance paths. Reviewers read the current assigned skill before inspection.
4. For automatic Director-policy injection and refreshed startup slash templates, use a new owner-started OMP application instance in the intended checkout/profile. `/new`, `/fresh`, and `/reload-plugins` in the old instance are not evidence of re-instantiating this cached policy. Do not restart unrelated work or reinterpret existing workers as freshly loaded.

To reverse only this adaptation, first check the inverse against current files. From PowerShell, with `$patch` set to this checkout's absolute `docs/omp-local-guidance.patch` path:

```powershell
git -C $HOME apply --check --reverse $patch
git -C $HOME apply --reverse $patch
```

Stop on a failed check rather than overwriting later edits. The retained seven-file backup is an alternative for deliberate, individually reconciled restoration, not permission to restore the entire agent directory. Restore repository guidance through the normal reviewed Git workflow. Loading rules apply in reverse too: use fresh workers for restored common text, withdraw/replace task-local supplements, and use a fresh owner-started instance for automatic Director/template reversal. Do not delete or rewrite persistent worker history.

## Selecting ordinary checks

Use the issue and current decisions to identify the outcome, preserved behavior, and relevant failure/recovery case before choosing commands. Follow affected owners/consumers, not just touched filenames. Ask whether all proposed checks could pass while the claim is wrong; close the concrete gap rather than creating a universal matrix or separate approval ceremony.

Use focused deterministic checks with actual collection/results and a small real-app action/readback where the claim requires one. PuriPuly's CLI reaches shared owners, but JSON success alone is not the oracle: follow terminal operations and owner-backed state, and use a new host for restart-persistence. Actual GUI claims require actual GUI input and visible response. Record replaced dependencies and the boundary observed; headless readiness does not establish VR readability or destination delivery. Test failures, skipped/uncollected checks, interrupted runs, and unavailable prerequisites remain visible.

A separate settings file does not isolate secrets, provider environment fallbacks, extension discovery, model caches, scene logs/process discovery, telemetry, or output sockets. Inspect the relevant startup path and coordinate only owned resources. Do not use real credentials, private audio, paid providers, downloads, or an active VRChat session to manufacture completion. Proportionate checks do not waive explicit issue outcomes.

## Shared GitHub knowledge

Read current issue scope, applicable decisions and related work before meaningful updates. A newer bot summary cannot restore an excluded feature. Use existing readable Project conventions only with actual permission; unavailable board access leaves membership/status unknown, not all implementation blocked.

The authorized publishing role explains the useful result, changed diagnosis, exact blocker, or next decision in ordinary prose. Link an adequate PR explanation instead of copying it. No fixed fields, comment cadence, proof card, approval loop, or private-session export is required. Local success and authorized publication are separate facts. On a lost response, read the target before retrying; on a denial, do not bypass it or replay completed implementation. Required internal role outputs remain unchanged.

## Executed bounded pilot

The ordinary fresh `worker` assignment used the adapted guidance on the unchanged application at `52a1b0ba992c70d8248b29224cce178a1f18ca62`. Its pre-read record identified the newly injected outcome/invariant, actual-test-execution, runtime-boundary, fresh-process, isolation, and reporting directions before opening local guidance, the issue, or research notes. It then explicitly read worktree `AGENTS.md`. This establishes common-contract loading and use in that worker, not automatic refresh of the current Director or every persistent role.

The Director read and used the revised policy as a task-local supplement and read current handoff files. Reviewer activation/use and the independent checkpoint/terminal judgments are recorded in the issue's review/result update; this execution record does not itself grant review approval.

### Deterministic pillar

Three existing selectors executed once under CPython 3.14.7, with actual collection and JUnit inspection: **11 passed, 0 failed/errors/skipped**.

```text
tests/app/test_settings_view_boundary.py::test_activation_notice_intent_replays_only_its_canonical_preference
tests/app/test_settings_view_boundary.py::test_activation_notice_intent_rejects_non_boolean_values
tests/app/test_settings_application_owner.py::test_activation_notice_edit_commits_or_restores_without_generic_runtime_apply
```

The command used the existing absolute virtual-environment interpreter from the intended worktree, `PYTHONPATH=<worktree>/src`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, and `-m pytest -vv -rA -o addopts= -p no:cacheprovider -p pytest_asyncio.plugin`, with owned `--basetemp` and `--junitxml` paths. Later disposable-aid repairs reused the same unchanged test evidence; they were not additional passes. Transaction tests replace runtime effects/presentation, so they do not establish real output delivery.

### Actual application pillar

A disposable driver launched two sequential real headless hosts and separate authenticated module-CLI clients. The host bootstrap invoked the existing `run_headless`; settings, persistence, output, transport, and control owners remained unchanged. Both actual hosts asserted that relevant imports came from this worktree, avoiding the existing virtual environment's adjacent editable installation. Actual host PID, Windows virtual-environment launcher lineage, process birth identity, and distinct instance UUIDs were correlated with `app.status`.

After live `capabilities` and a post-startup baseline, the pilot used documented `settings apply`, `settings current`, `app/osc/overlay status`, and `app stop`:

| Action | Observed result |
| --- | --- |
| Save notice preference `false` | Committed value `false`; revision 1 → 2; runtime-applied settings transaction; all unrelated committed/file fields unchanged. |
| Submit `true` using stale revision 1 | Terminal `rejected`, exit 4, exact `revision_conflict` at revision 2; complete committed result and saved bytes unchanged. |
| Submit valid `true`, then valid `false` | Genuine later commits at revisions 3 and 4, preserving all unrelated fields and inactive runtime boundaries. |
| Stop first host; start a fresh host with the same saved profile | Before further mutations, owner readback and independent saved-file observation matched the entire last committed snapshot, including `false`. Fresh-instance revision was 1, not incorrectly required to persist as 4. |
| Stop second host | Both final hosts stopped normally; receipts retained `cleanup_outcome: unobserved`, not full resource-release/flush proof. |

The three valid saves returned terminal **`degraded`, exit 4**, not `applied`/exit 0. Empty credentials left the selected self Soniox provider unavailable before any mutation. `application_control.py:1557-1571` adds overall provider-convergence status to otherwise completed settings transactions. Each save therefore required `settings_commit_success_runtime_applied`, exact agreement with the preexisting provider mismatch, advanced revision, full owner/file comparisons, and unchanged channel state. Arbitrary degradation was not accepted. The Director corrected the preliminary exit-0 oracle to the actual preference contract without changing product code, introducing credentials, or claiming whole-runtime health.

Synthetic state included a real empty encrypted-file secret store, allowlisted child environments, owned AppData/UserProfile/discovery/extension/model/log/cache roots, disabled telemetry/clipboard, OSC off, and a private loopback UDP sink instead of VRChat's destination. Only external VRChat process snapshots were replaced with an empty adapter; the actual scene service used empty owned logs. Capture desired/effective state, translation, microphone testing, OSC activity, and overlay activity remained inactive. Offline cache flags were defense in depth, not proof of universal network isolation.

Four unsuccessful disposable-aid attempts were retained for review: an incorrect Windows launcher/host PID assumption, an incorrect comparison against CLI-redacted secrets, an invented OSC observation key, and the incorrect exit-0 expectation. The aid was corrected against actual contracts, retaining semantic assertions. The final complete run exercised 31 separate CLI clients; these are not additional tests. No permanent tests were weakened or product repairs made.

All owned runtime/profile/secrets/cache/discovery trees were removed and sockets closed. The first attempt needed explicitly recorded forced launcher cleanup; its recorded host was subsequently observed absent. Final hosts used normal CLI stop. Disposable aids and original non-secret execution evidence were retained for independent inspection, with aid removal after review.

Limits: no actual GUI/audio/HMD interaction, visible notice suppression, live output-owner flag readback, provider-health claim, VRChat destination acknowledgment, physical disk flush, or full teardown claim. Installed `sherpa-onnx/core` was 1.13.8 versus the core 1.13.4 pin; no dependency mutation or ASR decoding was performed. There is no claimed clean lock-matched environment. No architecture ownership or user-visible product behavior changed.

