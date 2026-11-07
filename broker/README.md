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
- QQ subject references are derived using `QQ_AUTH_HMAC_PSK`. Replacing that production secret requires a versioned subject-reference rotation plan with dual lookup/backfill semantics; simply replacing the secret is not supported.
- Network abuse accounting uses versioned, scope-separated HMAC identities. Rotation keeps a previous/current key overlap for active accounting windows; operator IP hooks use keyed identities rather than raw addresses. Key/version configuration and staged conversion must remain consistent.
- App activity accepts only an anonymous identifier and UTC active date, stores only the derived subject/date pair, and deduplicates repeated reports. It is separate from retired translation telemetry and carries no conversation, provider, device, or network metadata.
- Scheduled work owns retention, unresolved-resource cleanup, and daily reporting. Reports aggregate delivered-key counts and app activity; immediate incidents describe issuance pressure or cleanup failures without exposing personal data.
- Editable abuse policy and mutable enforcement state remain separate. Notification failure must not replace the primary issuance or cleanup outcome.

## Deployment and Schema Changes

The manual [direct-deploy workflow](../.github/workflows/deploy-broker-direct.yml) deploys code to an **already initialized, supported Broker database**. Leave `apply_migrations` off for ordinary code updates. The workflow checks the canonical `dev` target and database readiness, deploys once, and verifies `/healthz` and `/v1/foundation`. These public checks do not prove authentication, key issuance, or upstream model routing.

The GitHub `production` environment needs:

- Secrets: `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID`, `BROKER_D1_DATABASE_ID_PRODUCTION`.
- Variables: `BROKER_CANONICAL_WORKERS_DEV_URL`, `NETWORK_IDENTITY_HMAC_KEY_VERSION_PRODUCTION`; optionally `NETWORK_IDENTITY_HMAC_KEY_VERSION_PREVIOUS_PRODUCTION`.
- Dispatch confirmation: `deploy puripuly-heart-broker from dev`.

Existing Worker secrets are preserved, not uploaded again. Deployment uses `--keep-vars` to preserve non-owned runtime variables. Omitting the optional previous version does not remove a deployed previous-version variable or secret; this workflow is not a key-rotation/removal mechanism. Keep configured key versions aligned with the deployed secrets.

### Ordinary Migration Maintenance

Select `apply_migrations` only when ordinary schema changes are intended. One staged migration set excludes `0025` telemetry deletion and `0027` referral retirement. The workflow requires completed historical privacy maintenance and initialized fingerprint state before applying anything, then checks the actual ordinary migration journal and required cleanup schema before deployment. A code-only run with pending ordinary migrations stops rather than deploying against an outdated schema.

This workflow does not install an empty database, bootstrap salts, wait for historical backfills, finalize legacy configuration, change OpenRouter guardrails, issue provider keys, or delete retired data. Existing databases requiring the old `0021`/`0024` transition must use the prior supported Worker and historical SQL gates. Do not rewrite migrations or force their journal state.

**No workflow backup or recovery checkpoint is created or required.** There is no SQL export, encryption key, backup artifact, Time Travel bookmark gate, or automatic restore. The owner chose this simpler maintenance contract. Any available Cloudflare-native history is an operator recovery option, not a verified restore guarantee or an independent recovery copy. A failed migration/deployment can require manual repair; successfully applied migrations are not automatically rolled back if deployment or a later check fails.

### Operator-Only Maintenance

Provision or change secrets through the canonical Worker's Cloudflare settings or an explicitly authorized Wrangler secret operation, not ordinary code deployment. Preserve unrelated bindings and runtime settings. Replacing `QQ_AUTH_HMAC_PSK` requires the versioned subject-reference migration described above, not a simple secret replacement. Change the existing OpenRouter guardrail deliberately when model/provider policy changes; retain its enforcement for issued keys.

For authorized one-time SQL maintenance, render the production config from `broker/`:

```sh
node scripts/render-production-wrangler-config.mjs \
  --database-id "$BROKER_D1_DATABASE_ID_PRODUCTION" \
  --network-identity-hmac-key-version "$NETWORK_IDENTITY_HMAC_KEY_VERSION_PRODUCTION" \
  --out wrangler.production.jsonc
```

Include `--network-identity-hmac-key-version-previous` with its configured value when applicable. The [fingerprint renderer](scripts/render-fingerprint-bootstrap-sql.mjs) accepts `--salt` and `--out` for a private SQL file and changes only an uninitialized placeholder; never use it to rotate initialized state. The [daily-summary](deploy/finalize-daily-summary-v2.sql) and [active-day](deploy/finalize-app-active-day.sql) finalizers remain standalone upgrade tools, not deployment steps. For example, after specifically approving active-day configuration cleanup, run `pnpm exec wrangler d1 execute puripuly-heart-broker --remote --config wrangler.production.jsonc --file deploy/finalize-app-active-day.sql --yes` from `broker/`.

The [daily-auth-cap maintenance workflow](../.github/workflows/maintenance-broker-abuse-controls.yml) remains separately confirmed and verifies its update by readback. Caps must be canonical positive decimals from 1 through 100000. It uses the same explicit database and network-key-version configuration; invalid configuration stops before D1 access.

Full live issuance/routing acceptance remains available from the repository root when explicitly authorized:

```sh
BROKER_DEPLOY_SMOKE_RUN=true \
BROKER_DEPLOY_SMOKE_BASE_URL="$BROKER_CANONICAL_WORKERS_DEV_URL" \
BROKER_DEPLOY_SMOKE_QQ_AUTH_HMAC_PSK="$QQ_AUTH_HMAC_PSK" \
BROKER_DEPLOY_SMOKE_DISALLOWED_MODEL="$DISALLOWED_MODEL" \
pnpm exec vitest run broker/tests/deploy-smoke/canonical-production.spec.ts
```

Supply operator-held values without publishing them; the disallowed model must be outside the current managed allowlist. This check creates real provider keys and makes model requests. It is not part of normal deployment, and public health success is not a substitute for it after relevant authentication, issuance, or policy changes.

### Separately Authorized Retirement

- **Legacy translation telemetry (`0025`):** inspect aggregate counts/date ranges and external or operator-created consumers, obtain explicit deletion approval, and apply only that migration through a dedicated staged directory. There is no deployment deletion checkbox or backup prerequisite.
- **Referral storage (`0027`):** complete privacy prerequisites, verify the referral-free Worker, and establish that prior-version requests and scheduled invocations have drained. Apply only the pending referral-removal migration through a dedicated staged directory without including `0025`. Read back journal/schema and surviving operation, attempt, delivery, cleanup, and entitlement relationships. Retirement does not change provider budgets.

These operations are destructive and have no workflow-managed recovery copy. Health alone does not authorize them. The [migration directory](migrations/) preserves forward schema history; local verification does not authorize production changes.

## Development and Verification

Run Broker tooling from a Linux-native checkout, including WSL-internal storage, not Windows or shared `/mnt/c/...` dependencies. From the repository root:

```sh
pnpm install --frozen-lockfile
pnpm typecheck
pnpm --filter @puripuly-heart/broker run verify:config
pnpm exec vitest run broker/tests --exclude "broker/tests/deploy-smoke/**"
```

`verify:config` checks the pinned Wrangler binding contract without cloud credentials. Use isolated synthetic state and controlled provider responses for local behavior checks. The [Broker CI job](../.github/workflows/pr-ci.yml) defines automated checks; production smoke is a separate, explicitly authorized live operation.

For focused offline deployment/maintenance coverage, run `pnpm exec vitest run broker/tests/deploy-automation.spec.ts broker/tests/minimal-deploy.spec.ts`. It exercises the actual workflow shell, renderer, read-only default, explicit ordinary migrations, rejection/failure ordering, public HTTP checks, fingerprint helper, and cap update/readback with isolated synthetic SQLite and controlled local ports. It does not exercise live Cloudflare deployment, provider issuance, or recovery.

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
