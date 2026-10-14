## General

- Code comments are added only when explicitly requested.
- Merge, push, publish, deployment, release, and worktree cleanup proceed only with explicit approval.
- Use `docs/architecture.md` as the system map when a task requires reasoning about how the system fits together.
- Report suspected architecture drift introduced by your changes to the user.
- Report any user-visible changes after completing the work.

## Test

- Prefer contract and behavior tests over implementation-detail tests.
- Test observable outcomes and invariants so behavior-preserving refactors normally do not require test changes.
- Derive the promised outcome, preserved invariants, and relevant failure/recovery behavior before choosing checks. Follow affected owners and consumers using `docs/architecture.md`; ask whether all selected checks could pass while the outcome is wrong.
- Use the current `capabilities` and documented controls in `docs/cli.md` for actual-app checks. Inspect terminal operations and owner-backed effective state, not just CLI exit codes, admission receipts, saved intent, or logs. A persistence-across-restart claim needs a fresh host using the saved isolated profile.
- Confirm the intended checkout/import or executable and instance are running. Check presentation and input separately in the actual GUI when the claim reaches them; headless readiness does not prove visible behavior.
- Use isolated synthetic state for probes. Inspect startup secrets, discovery, provider, telemetry, and output behavior: a separate `--config` alone is not isolation. Coordinate owned processes and ports; do not use private audio, live credentials, paid providers, downloads, or an active VRChat session merely to finish verification.
- Confirm the selected tests actually execute; retain failures, skips, missing dependencies, and unperformed checks. Do not weaken meaningful assertions to obtain green. Expand verification only for a concrete gap or invalidated evidence.

## Shared work context

- Read the current issue and applicable decisions before work or a meaningful GitHub update; preserve exclusions even when a newer summary conflicts. Use readable Projects only within existing conventions and granted permissions.
- An authorized publishing role should leave the scoped result, useful evidence and limits, and the next decision understandable without local-session access. Link an adequate PR explanation instead of duplicating it; never publish personal paths, credentials, or private transcripts. Missing publication permission does not invalidate local execution.
- Local OMP loading, activation/reversal, and the bounded verification pilot are documented in `docs/omp-workflow.md`. General role policy remains in the effective user-managed OMP sources, not in product code.

## Experimentation

- Treat experiments as engineering probes, using the minimum validation needed to support the immediate engineering decision.
- Expand an experiment when a plausible result could change the engineering decision, stating the remaining uncertainty and the decision it could affect.
