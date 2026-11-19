## General

- Code comments are added only when explicitly requested.
- Do not modify README files, architecture.md, PRIVACY.md, CONTRIBUTING.md, `docs/cli.md`, `docs/vrchat-osc.md`, or `.github/release-template.md` unless explicitly requested. Report resulting drift by file and mismatch.
- Use `docs/architecture.md` as the system map.
- When reporting to the user, explain the key points directly in language that is easy to understand. Include user-visible changes.

## Verification

- Product behavior: prioritize observed results in the running app. Compose the smallest relevant path from `docs/verification.md`; follow its setup and evidence boundaries.
- Code tests: reuse or add checks that cheaply and repeatedly catch meaningful failures or protect required invariants. Test observable contracts, not implementation details.
- This priority is not an execution order. Use proportionate checks; narrower evidence can establish narrow internal claims. Confirm execution, address failures, and briefly report results and meaningful gaps.
- Update `docs/verification.md` when reusable paths, entrypoints, inputs, prerequisites, expected observations, or evidence boundaries change. Restoring documented behavior alone does not require a new example.

## Shared work context

- Read the current issue and decisions before work or GitHub updates; preserve scope and exclusions.
- Keep shared records clear, structured, and understandable without local-session access: results, evidence, limits, and next steps. Link existing explanations; share only non-sensitive information.
- Publish GitHub content in English.

## Experimentation

- Use the smallest probe that informs the engineering decision. Expand only when a plausible result could change it; name the uncertainty and affected decision.
