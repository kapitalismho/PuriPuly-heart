## General

- Code comments are added only when explicitly requested.
- Merge, push, publish, deployment, release, and worktree cleanup proceed only with explicit approval.
- Use `ARCHITECTURE.md` as the system map when a task requires reasoning about how the system fits together.
- Report suspected architecture drift introduced by your changes to the user.

## Test

- Prefer contract and behavior tests over implementation-detail tests.
- Test observable outcomes and invariants so behavior-preserving refactors normally do not require test changes.
## Experimentation

- Treat experiments as engineering probes, using the minimum validation needed to support the immediate engineering decision.
- Match each claim to the available evidence and state key uncertainties and limitations.
- Record the setup, relevant code and data versions, and observations to enable later research-grade validation.
- Conclude once the evidence supports the stated decision, and present further validation as optional follow-up.
- Expand an experiment only when a plausible result could change the engineering decision, stating the remaining uncertainty and the decision it could affect.

