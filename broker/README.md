# Broker Service

Managed identity, entitlement, and credential release for PuriPuly Heart. Broker runs on Cloudflare Workers with D1 and Worker secrets. The app calls OpenRouter directly; Broker is not a translation proxy.

See [application ownership](../docs/architecture.md) and [client authentication and recovery](../docs/cli.md#authentication-and-recovery).

## Service Boundaries

- Broker authenticates installation, Discord, and QQ authority and manages credential issuance. The app owns local persistence and provider activation.
- Recovery reconciles the same durable operation. A lost response is not permission to issue another key.
- OpenRouter enforces issued keys. Broker-local revocation cannot stop direct use; disable or delete the key upstream.
- Broker receives no chat, app context, tool results, or support bundles. Those belong to [Agent Relay](../agent-relay/README.md).
- Keep raw identities, provider keys, recovery tokens, and request bodies out of durable storage and logs.

## Puri Installation Tokens

`POST /v1/puri/token` verifies installation-key proof and issues a Relay bearer token. The app keeps it in memory. Managed entitlement is not required. Relay receives a derived subject, not the installation ID.

`PURI_TOKEN_SECRET` gives Broker and Relay signing authority; coordinate its rotation. `PURI_IDENTITY_HMAC_SECRET` stays Broker-only; changing it changes accounting subjects. See the [token handler](src/puri-token.ts) and [secret provisioning tool](scripts/manage-puri-secrets.mjs).

## Deployment and Maintenance

Run the [direct-deploy workflow](../.github/workflows/deploy-broker-direct.yml) from `dev` with confirmation `deploy puripuly-heart-broker from dev`. Use an initialized, supported database. Configuration lives in the workflow and [renderer](scripts/render-production-wrangler-config.mjs).

Leave `apply_migrations` off for code-only updates. Pending ordinary migrations block deployment; never force journal entries. Ordinary migrations exclude destructive `0025` and `0027`.

Deployment preserves secrets. Coordinate rotation with [network key versions](src/network-identity.ts); changing `QQ_AUTH_HMAC_PSK` requires subject-reference migration. The [fingerprint helper](scripts/render-fingerprint-bootstrap-sql.mjs) initializes state; do not use it for rotation.

Committed migrations have no automatic rollback. Authorize database recovery separately.

Deployment checks public health and foundation routes. The [live issuance/routing smoke](tests/deploy-smoke/canonical-production.spec.ts) creates real provider keys and model requests; it requires separate authorization.

### Separately Authorized Maintenance

- Run `0025` (telemetry deletion) only after data/consumer review and deletion approval.
- Run `0027` (referral deletion) only after privacy prerequisites and old-request drainage. Stage it separately from `0025` and verify surviving credential relationships.
- These [migrations](migrations/) have no workflow-managed recovery copy. Use the [maintenance workflow](../.github/workflows/maintenance-broker-abuse-controls.yml) for authorization-cap changes.

## Local Development

Use a Linux-native checkout and dependencies, including WSL-internal storage. Use isolated local D1 and credentials, synthetic identities, and controlled provider responses. From the repository root:

```sh
pnpm install --frozen-lockfile
pnpm typecheck
pnpm --filter @puripuly-heart/broker run verify:config
pnpm exec vitest run broker/tests --exclude "broker/tests/deploy-smoke/**"
pnpm --filter @puripuly-heart/broker dev --local --ip 127.0.0.1
```

Initialize local state with [migrations](migrations/) and the fingerprint helper. Keep production credentials out of local bindings. [PR CI](../.github/workflows/pr-ci.yml) defines automated checks.

## Source Map

| Owner | Source |
| --- | --- |
| HTTP and authentication | [Routes](src/app.ts), [installation handshake](src/trial-handshake.ts), [QQ assertions](src/qq-auth.ts) |
| Credential lifecycle | [Issuance](src/managed-issuance.ts), [operation owner](src/managed-operation.ts), [delivery](src/managed-key-delivery.ts) |
| Identity and retention | [Network identity](src/network-identity.ts), [scheduler](src/scheduled.ts) |
| Behavior checks | [Tests](tests/) |
