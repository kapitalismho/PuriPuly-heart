# Broker Service

Managed identity, entitlement, and credential release for PuriPuly Heart. The Broker is a separate TypeScript/Hono service on Cloudflare Workers with D1 and Worker secrets. The application calls OpenRouter directly; the Broker is not a translation proxy or part of the normal utterance path.

This guide covers service boundaries, recovery invariants, and development/deployment entry points. Exact request schemas, policy values, storage layouts, and regression evidence belong in the [owning implementation](#source-map), not this guide. See the [architecture](../docs/architecture.md) for application ownership and the [CLI guide](../docs/cli.md#authentication-and-recovery) for client-side recovery.

## Ownership and Scope

- The Broker authenticates installation/device authority, Discord OAuth, and QQ assertions, then applies managed entitlement and issuance policy.
- D1 owns durable operations, provider-key attempts, deliveries, entitlements, abuse controls, and cleanup responsibility. Worker secrets hold service credentials and identity-derivation keys.
- OpenRouter owns child keys, guardrail enforcement, and live usage/budget metadata. Broker-local revocation alone cannot stop an already issued key's direct use; operators must also disable or delete it upstream.
- The application owns local credential persistence, delivery acknowledgment, and provider activation. These are separate from successful authorization or issuance.
- The deployment assumes a single-region rollout. Translation proxying, multi-region infrastructure, and an admin dashboard are outside this service's scope. Referrals, Talk Together Pass, and reward settlement are no longer supported; existing credentials and allowances are preserved.

## Issuance, Delivery, and Recovery

- Device challenge verification binds release authority to the registered installation key and verified fingerprint. Discord and QQ onboarding use their own identity boundaries; QQ does not synthesize device installations.
- Managed issuance is one durable logical operation that may span multiple provider-key attempts. The client retains its operation identity and protected resume authority before issuance. Raw OpenRouter keys exist only transiently in Broker memory and cannot be recovered from Broker storage.
- Key creation, response delivery, client persistence, delivery ACK, and activation are distinct transitions. ACK finalizes delivery without increasing the provider allowance. A lost response is not permission to create another key.
- Use the managed-operation `status` and `resume` routes to reconcile the same operation. Recovery authority is bounded and identity-bound; its expiry does not end the Broker's responsibility for unresolved resources.
- Unknown creates remain owned until authoritative settlement or confirmed cleanup of the exact resource. Timeouts, incomplete enumeration, or an empty inventory after uncertain dispatch do not prove that replacement issuance is safe.
- Cleanup claims are attempt-scoped and exclude acknowledged resources. Older attempts may retain cleanup debt while a newer credential remains active; reconciliation must preserve that credential, entitlement, and allowance.
- Scheduled cleanup uses durable retry eligibility, fair rotation, and shared work limits rather than rewriting every unresolved row on every cron tick. Authenticated recovery can bypass the unattended cooldown, not cleanup ownership or resource protection. Policy and cost rationale are recorded in [issue #286](https://github.com/kapitalismho/PuriPuly-heart/issues/286).

Public error envelopes distinguish retryable, terminal, and security failures from entitlement lifecycle data. Read the returned state and retry guidance rather than treating every non-active state as a new-issuance opportunity. Route and payload contracts live in [routing](src/app.ts), [errors](src/broker-error.ts), and the owning handlers.

## Privacy and Operations

- Persist derived identity references, not raw QQ identities, assertion credentials, API keys, or recovery/ACK tokens. Logs and alerts must exclude secrets, raw request payloads, conversation content, and audio.
- Network abuse accounting uses versioned, scope-separated HMAC identities. Rotation keeps a previous/current key overlap for active accounting windows; operator IP hooks use keyed identities rather than raw addresses. Key/version configuration and staged conversion must remain consistent.
- App activity accepts only an anonymous identifier and UTC active date, stores only the derived subject/date pair, and deduplicates repeated reports. It is separate from retired translation telemetry and carries no conversation, provider, device, or network metadata.
- Scheduled work owns retention, unresolved-resource cleanup, and daily reporting. Reports aggregate delivered-key counts and app activity; immediate incidents describe issuance pressure or cleanup failures without exposing personal data.
- Editable abuse policy and mutable enforcement state remain separate. Notification failure must not replace the primary issuance or cleanup outcome.

## Deployment and Schema Changes

The manual [direct-deploy workflow](../.github/workflows/deploy-broker-direct.yml) is the production entry point and authority for required environment secrets/variables. It renders deployment configuration, exports D1 before migrations, reconciles the OpenRouter guardrail, deploys, checks health, and runs a synthetic issuance/routing smoke. The checked-in [Wrangler config](wrangler.jsonc) contains a placeholder D1 identifier and is not production-ready by itself.

- Keep credentials in the production environment, never source or logs. [Config rendering](scripts/render-production-wrangler-config.mjs) supplies deployment-specific bindings and network-identity key versions; [fingerprint bootstrap](scripts/render-fingerprint-bootstrap-sql.mjs) is guarded against replacing initialized state.
- Existing databases must complete the historical `0021`/`0024` privacy transition with the prior supported Worker before installing the referral-free runtime. Empty databases complete those migrations before their first deployment. Do not rewrite historical migrations, force migration state, or bypass request/hook conversion gates.
- Health success is not issuance or model-routing success. Account-level OpenRouter settings can narrow effective routing; review the production smoke before connecting app/public traffic. A failed post-deploy check does not imply rollback.
- Ordinary deployment excludes both `0025` telemetry deletion and `0027` referral-schema retirement. Migration files and local checks are not evidence that production data was removed.

### Separately Authorized Retirement

- **Legacy translation telemetry (`0025`):** inspect production aggregate counts/date ranges and external or operator-created dependencies, then obtain explicit deletion approval. Use the workflow's dedicated deletion controls; dispatch confirmation does not prove that review or approval occurred. Deletion uses the run's backup and applies only the held migration after health and production smoke checks.
- **Referral storage (`0027`):** complete the privacy prerequisites, back up D1, deploy and verify the referral-free Worker, and establish that prior-version requests and scheduled invocations have drained. Health alone does not prove this. Apply only the pending referral-removal migration through a dedicated staged directory, without including `0025`. Read back the migration journal/schema and verify surviving operation, attempt, delivery, cleanup, and entitlement relationships. Retirement does not change provider budgets.

The [migration directory](migrations/) defines the forward schema history. Production deletion and deployment require their own authorization; development verification does not authorize either.

## Development and Verification

Run Broker tooling from a Linux-native checkout, including WSL-internal storage, not Windows or shared `/mnt/c/...` dependencies. From the repository root:

```sh
pnpm install --frozen-lockfile
pnpm typecheck
pnpm --filter @puripuly-heart/broker run verify:config
pnpm exec vitest run broker/tests --exclude "broker/tests/deploy-smoke/**"
```

`verify:config` checks the pinned Wrangler binding contract without cloud credentials. Use isolated synthetic state and controlled provider responses for local behavior checks. The [Broker CI job](../.github/workflows/pr-ci.yml) defines automated checks; production smoke is a separate, explicitly authorized live operation.

## Source Map

| Question | Read |
| --- | --- |
| Routes, environment bindings, public errors | [Routing](src/app.ts), [contract](src/contract.ts), [errors](src/broker-error.ts) |
| Device, Discord, and QQ authentication/issuance | [Trial handshake](src/trial-handshake.ts), [Discord issuance](src/discord-managed-issue.ts), [QQ assertions](src/qq-auth.ts), [QQ issuance](src/qq-managed-issue.ts) |
| Durable issuance, recovery, delivery ACK | [Issuance](src/managed-issuance.ts), [operation owner](src/managed-operation.ts), [resume](src/managed-operation-resume.ts), [delivery](src/managed-key-delivery.ts) |
| Provider resources and cleanup | [OpenRouter management](src/openrouter-management.ts), [cleanup](src/managed-cleanup.ts) |
| Trial policy and persistence | [Policy](src/trial-policy.ts), [persistence](src/persistence.ts), [migrations](migrations/) |
| Abuse controls, identity rotation, retention | [Controls](src/abuse-controls.ts), [network identity](src/network-identity.ts), [fingerprint salt](src/fingerprint-salt.ts), [scheduler](src/scheduled.ts) |
| Activity, reports, and alerts | [Telemetry](src/telemetry.ts), [monitoring](src/abuse-monitoring.ts), [Discord delivery](src/discord-alerts.ts) |
| Behavior and deployment evidence | [Broker tests](tests/), [production workflow](../.github/workflows/deploy-broker-direct.yml) |
