- Code comments are added only when explicitly requested.
- Merge, push, publish, deployment, release, and worktree cleanup proceed only with explicit approval.
- Use `ARCHITECTURE.md` as the system map when a task requires reasoning about how the system fits together.
- Report suspected architecture drift introduced by your changes to the user.
- Prefer contract and behavior tests over implementation-detail tests.
- Test observable outcomes and invariants so behavior-preserving refactors normally do not require test changes.

