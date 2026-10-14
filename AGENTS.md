## General

- Code comments are added only when explicitly requested.
- Merge, push, publish, deployment, release, and worktree cleanup proceed only with explicit approval.
- Write all content published to GitHub in English.
- Use `docs/architecture.md` as the system map when a task requires reasoning about how the system fits together.
- Report suspected architecture drift introduced by your changes to the user.
- Report any user-visible changes after completing the work.

## Test

- Prefer contract and behavior tests over implementation-detail tests.
- Test observable outcomes and invariants, keeping tests stable across behavior-preserving refactors.
- Derive the promised outcome, preserved invariants, and relevant failure/recovery behavior before choosing checks. Follow affected owners and consumers using `docs/architecture.md`; ask whether all selected checks could pass while the outcome is wrong.
- Use the current `capabilities` and documented controls in `docs/cli.md` for actual-app checks. Follow operations to terminal results and inspect owner-backed effective state, interpreting CLI exit codes, admission receipts, saved intent, and logs at their respective boundaries. Verify persistence across restart with a fresh host using the saved isolated profile.
- Confirm the intended checkout/import or executable and instance are running. Check presentation and input separately in the actual GUI when the claim reaches them, and describe headless readiness within its observed boundary.
- Use isolated synthetic state for probes. Inspect startup secrets, discovery, provider, telemetry, and output behavior alongside the selected `--config`. Coordinate owned processes and ports. Choose synthetic inputs, empty secrets, local fixtures, and owned output destinations; keep private audio, live credentials, paid providers, new model downloads, and active VRChat sessions outside routine verification.
- Confirm the selected tests actually execute; retain failures, skips, missing dependencies, and unperformed checks. Preserve meaningful assertions and use failure evidence to guide implementation or verification corrections. Expand verification for a concrete gap or invalidated evidence.

## Shared work context

- Read the current issue and applicable decisions before work or a meaningful GitHub update; preserve exclusions even when a newer summary conflicts. Use readable Projects only within existing conventions and granted permissions.
- An authorized publishing role should leave the scoped result, useful evidence and limits, and the next decision understandable independently of local-session access. Link adequate existing explanations and publish non-sensitive facts while retaining personal paths, credentials, and private transcripts locally. Track local execution and publication authority separately.
- Maintain general role policy in the effective user-managed OMP sources and product-specific guidance in this project.

## Experimentation

- Treat experiments as engineering probes, using the minimum validation needed to support the immediate engineering decision.
- Expand an experiment when a plausible result could change the engineering decision, stating the remaining uncertainty and the decision it could affect.
