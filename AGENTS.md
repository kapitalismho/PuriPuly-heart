## General

- Code comments are added only when explicitly requested.
- Use `docs/architecture.md` as the system map when a task requires reasoning about how the system fits together.
- Report suspected architecture drift introduced by your changes to the user.
- Report any user-visible changes after completing the work.

## Test

- Prefer contract and behavior tests over implementation-detail tests.
- Test observable outcomes and invariants, keeping tests stable across behavior-preserving refactors.
- Choose checks that demonstrate promised outcomes, invariants, and failure/recovery behavior across affected components.
- Use current `capabilities` and `docs/cli.md` for actual-app checks; verify completed operations and effective state, including restart persistence when relevant.
- Verify the intended build and instance; check presentation and interaction separately in the actual GUI.
- Use isolated synthetic state and controlled local resources for probes; keep private data and live services outside routine verification.
- Confirm checks execute, report failures and verification gaps, and use the evidence to guide corrections or further checks.

## Shared work context

- Read the current issue and applicable decisions before work or GitHub updates, preserving agreed scope and exclusions.
- Make results, evidence, limits, and next steps understandable without local-session access.
- Link existing explanations and share only non-sensitive information.
- Use clear, direct sentences and structured formats such as bullet points for shared work records and GitHub content.
- Write all content published to GitHub in English.

## Experimentation

- Treat experiments as engineering probes, using the minimum validation needed to support the immediate engineering decision.
- Expand an experiment when a plausible result could change the engineering decision, stating the remaining uncertainty and the decision it could affect.
