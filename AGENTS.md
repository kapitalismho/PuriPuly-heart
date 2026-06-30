## Purpose & Authority

- If implementation facts and docs disagree, treat code as the source of truth and then align docs.
- Keep this file concise and operational. Record only rules an agent must remember before touching the codebase.
- Do not add comments to code unless explicitly requested.
- If a compacted or resumed context references an active `/implement-work` run, reload the `implement-work` skill, read its ledger and selected source input, validate source identity/hash, and continue only from the recorded next action. Do not continue from compressed chat memory alone; stop for context or decision if the ledger or source is ambiguous.

## Architecture Work Model

- Before changing behavior, name the boundary being changed: settings/persistence, runtime resolution, transaction service, lifecycle owner, output routing, message/diagnostics, adapter wiring, or UI rendering.
- Keep persisted user intent, persisted operational state, resolved runtime config, and runtime-only state separate.
- Prefer explicit contracts over convenient imports. A new dependency should be identifiable as a schema value, resolved DTO, service port, adapter, lifecycle owner, or renderer.
- Split files by owner and reason-to-change after the boundary exists; avoid cosmetic or mechanical file splits.

## Compatibility & Persistence

- Settings load compatibility is mandatory:
  - Keep `to_dict` and `from_dict` synchronized.
  - New settings must have defaults so existing `settings.json` continues loading.
  - If a setting key is renamed, accept the old key in `from_dict` for backward compatibility.
- Persisted schema changes require safe forward migration and a pre-migration backup.
- Preserve compatibility surfaces unless explicitly approved: SecretStore keys, Broker `/v1`, overlay protocol and startup contract, prompt fallback, provider aliases, i18n key parity, installer identity, and Rust overlay startup behavior.

## UI, Preview & User Messages

- All new user-facing UI text must go through i18n keys, and all locale bundles must be updated.
- Debug UI preview mode may exist for hard-to-reproduce UI states.
  - Verify the exact CLI flag and preview actions in code before use.
  - Preview actions must not persist settings, mutate secrets, or call external providers/brokers.
  - Use preview mode for manual QA of hidden UI states instead of forcing real broker/OpenRouter states.
  - Debug preview controls must remain hidden unless the explicit debug flag is enabled.

## Security, Async & Lifecycle

- Keep provider and I/O calls async; avoid blocking the event loop.
- Long-running work needs an explicit lifecycle owner, cancellation path, and shutdown behavior.
- Avoid unmanaged `asyncio.create_task`; owner-scoped tasks must be cancelled on shutdown.
- Always `await` provider `close()` in teardown paths.
- In Flet UI callbacks, use `page.run_task` for async work.
- Secrets are loaded through `SecretStore` (keyring/encrypted file/env fallback).
- When `secrets.backend` is `encrypted_file`, require `PURIPULY_HEART_SECRETS_PASSPHRASE`.
- Never commit real credentials, API keys, or secret material.

## Output, Diagnostics & Logs

- Treat self, peer, and system outputs as separate product channels.
- Peer utterances must not route to the VRChat chatbox.
- Diagnostics and logs must be safe to display or persist: no raw secrets, credentials, or unredacted provider payloads.

## Environment & Verification

- Prefer a project virtual environment for tests, verification, and development commands whenever one exists.
- If `.venv` exists, Windows shells should use `.venv`.
- If `.venv-wsl` exists, Linux / WSL shells should use `.venv-wsl`.
- In WSL shells, use `direnv exec <repo> ...` or explicit `UV_PROJECT_ENVIRONMENT=.venv-wsl`; do not rely on `bash -i -c`.
- Broker Node verification (`pnpm`, `vitest`, `wrangler`) must run from a Linux / WSL workspace only; do not run it from Windows shells.
- In WSL, install broker Node dependencies inside the Linux workspace; do not reuse Windows-installed `node_modules` from `/mnt/c/...`.
- If a task modifies Rust code, the final step of the overall task must recompile the Rust overlay for Windows.
- Local installer smoke tests must use an alternate `AppId` and an isolated install directory; never reuse the production `AppId` for test installs.
- Do not claim completion without verification evidence from the relevant commands or checks.

## Freshness Checks

- Do not hardcode volatile defaults or file format assumptions in this file.
- Prompt file naming/extensions and fallback order must be verified in `src/puripuly_heart/config/prompts.py`.
- Orchestrator default parameters (including context memory values) must be verified in `src/puripuly_heart/core/orchestrator/hub.py`.
