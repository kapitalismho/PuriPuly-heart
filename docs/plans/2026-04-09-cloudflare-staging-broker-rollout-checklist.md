# Cloudflare Direct Broker Rollout Checklist

Date: 2026-04-09
Last Updated: 2026-04-10
Status: Working Notes

## Why this document exists

This document records the current rollout context plus the next-step checklist for taking the broker from terminal verification into a real Cloudflare deployment on the canonical production worker.

The goal is to let us work through the rollout step-by-step without losing the decisions we already made in chat.

## Current verified state

- The current session reached a **review-ready task `13` verification result in dedicated worktrees**.
- In that session, the assembled rollout passed fresh broker, Python/UI/runtime, installer-script, and Windows L3 build/smoke reruns.
- The current session also produced a candidate fix for the missing local STT installer script and used it to unblock installer smoke verification.
- Because that verification evidence currently lives in the active session worktrees rather than in a committed repo log in this checkout, treat this section as **session context for the next rollout step**, not as the canonical long-term audit record.

## Important boundary

Task `13` passing does **not** mean the broker is already deployed or production-ready by default.

What it does mean:

- the assembled rollout code and packaging path passed local / CI-like verification
- the broker contract, client wiring, dashboard rendering, and STT runtime-readiness path are in a good state to deploy

What it does **not** mean:

- Cloudflare production configuration is already set
- Cloudflare production deployment has already been validated live
- D1 / Worker secret / deploy workflow automation is already configured with real production secrets and ready to run

## Repo facts relevant to deployment

- `broker/wrangler.jsonc` still contains a placeholder D1 identifier:
  - `database_id: "REQUIRED_AT_DEPLOY_TIME"`
  - it also hardcodes `name: "puripuly-heart-broker"`, so direct deploy work targets the canonical worker name unless a deploy-time config overrides it
- The broker requires these runtime bindings:
  - D1 binding: `BROKER_DB`
  - Worker secrets: `OPENROUTER_MANAGED_API_KEY` (transitional compatibility only), `OPENROUTER_MANAGEMENT_API_KEY`, `OPENROUTER_MANAGED_GUARDRAIL_ID`, `OPENROUTER_MANAGED_USER_HMAC_SECRET`, `QQ_AUTH_HMAC_PSK`, `DISCORD_CLIENT_ID`, `DISCORD_CLIENT_SECRET`, `DISCORD_REDIRECT_URI_ALLOWLIST`, `DISCORD_USER_REF_SECRET`, `DISCORD_IMMEDIATE_ALERT_WEBHOOK_URL`, `DISCORD_DAILY_REPORT_WEBHOOK_URL`
- Current broker deploy command:
  - `pnpm --filter @puripuly-heart/broker run deploy`
- Broker `pnpm` / `vitest` / `wrangler` verification should run from a Linux-native workspace.
- The repo now includes a dedicated manual direct-production broker deployment workflow at `.github/workflows/deploy-broker-direct.yml`.
- The initial broker rollout assumes a single-region D1 deployment with `location_hint` set to `apac`.
- The initial D1 migration seeds `fingerprint_salt` with a bootstrap placeholder and explicitly requires deployment bootstrap to replace it before `challenge` / `verify` traffic is enabled.
- The Worker uses a minute-resolution cron trigger, but the daily Discord heartbeat is gated by runtime-configured `dailyReport` timing plus persisted `abuse_runtime_state`, so only one heartbeat should emit per UTC day after the configured due time.
- Manual revocation is a two-step operator action in the current architecture: revoke the broker-side entitlement **and** disable or delete the upstream OpenRouter child key, because broker-only revocation does not stop already-issued direct OpenRouter traffic.

## Decisions already made

- The previous staging-first plan is superseded for the current rollout.
- Do **not** create a separate staging worker for this rollout.
- The first live validation will happen on the canonical worker: `puripuly-heart-broker`.
- Use the canonical `workers.dev` URL for the first live smoke test; do not add a custom domain yet.
- Use a deploy-time generated Wrangler config so the production D1 `database_id` stays out of the checked-in `broker/wrangler.jsonc`.
- Use Wrangler D1 remote commands for migration and bootstrap:
  - `wrangler d1 migrations apply <db> --remote --config <generated-config>`
  - `wrangler d1 execute <db> --remote --config <generated-config> --file <sql> --yes`
- Put the broker smoke runner under `broker/tests/deploy-smoke/`.
- The deployment order remains:
  1. migrations
  2. fingerprint salt bootstrap
  3. guardrail reconcile
  4. secrets
  5. deploy
  6. automated smoke test
  7. human review before app / public traffic is pointed at the broker
- Secrets should live in GitHub Environments / CI secret storage, not in the repo.
- Because the smoke test runs against the canonical worker, the app must remain disconnected from the broker until the smoke path passes.

## Required production inputs

- [ ] Cloudflare account / token with Worker + D1 + secret-management permissions
- [ ] canonical Worker name confirmed as `puripuly-heart-broker`
- [ ] production Worker URL convention confirmed
- [ ] production D1 database created with the intended rollout location settings (`apac` for the current single-region assumption)
- [ ] production D1 `database_id` captured
- [ ] production bootstrap plan defined for replacing `fingerprint_salt` placeholder before challenge / verify traffic
- [ ] production OpenRouter managed key prepared (`OPENROUTER_MANAGED_API_KEY_PRODUCTION`, transitional compatibility only)
- [ ] production OpenRouter management key prepared (`OPENROUTER_MANAGEMENT_API_KEY_PRODUCTION`)
- [ ] production OpenRouter managed guardrail id prepared (`OPENROUTER_MANAGED_GUARDRAIL_ID_PRODUCTION`)
- [ ] production OpenRouter managed user-id HMAC secret prepared (`OPENROUTER_MANAGED_USER_HMAC_SECRET_PRODUCTION`)
- [ ] production QQ assertion HMAC PSK prepared (`QQ_AUTH_HMAC_PSK_PRODUCTION`, copied to Worker secret `QQ_AUTH_HMAC_PSK`; required with `OPENROUTER_MANAGEMENT_API_KEY` and `OPENROUTER_MANAGED_GUARDRAIL_ID` for QQ production issuance; never store the value in the repo)
- [ ] production Discord OAuth client id prepared (`DISCORD_CLIENT_ID_PRODUCTION`, copied to Worker secret `DISCORD_CLIENT_ID`)
- [ ] production Discord OAuth client secret prepared (`DISCORD_CLIENT_SECRET_PRODUCTION`, copied to Worker secret `DISCORD_CLIENT_SECRET`; never store the value in the repo)
- [ ] production Discord redirect allowlist prepared (`DISCORD_REDIRECT_URI_ALLOWLIST_PRODUCTION`, copied to Worker secret `DISCORD_REDIRECT_URI_ALLOWLIST`)
- [ ] production Discord user-ref HMAC secret prepared (`DISCORD_USER_REF_SECRET_PRODUCTION`, copied to Worker secret `DISCORD_USER_REF_SECRET`; never store the value in the repo)
- [ ] production Discord operations webhook prepared (`DISCORD_OPERATIONS_WEBHOOK_URL_PRODUCTION`)
- [ ] GitHub Environment `production` created if CI automation is used
- [ ] GitHub secret `CLOUDFLARE_API_TOKEN` registered
- [ ] GitHub secret `CLOUDFLARE_ACCOUNT_ID` registered if the workflow needs it
- [ ] GitHub secret `BROKER_D1_DATABASE_ID_PRODUCTION` registered
- [ ] GitHub secret `OPENROUTER_MANAGED_API_KEY_PRODUCTION` registered (transitional compatibility only)
- [ ] GitHub secret `OPENROUTER_MANAGEMENT_API_KEY_PRODUCTION` registered
- [ ] GitHub secret `OPENROUTER_MANAGED_GUARDRAIL_ID_PRODUCTION` registered
- [ ] GitHub secret `OPENROUTER_MANAGED_USER_HMAC_SECRET_PRODUCTION` registered
- [ ] GitHub secret `QQ_AUTH_HMAC_PSK_PRODUCTION` registered
- [ ] GitHub secret `DISCORD_CLIENT_ID_PRODUCTION` registered
- [ ] GitHub secret `DISCORD_CLIENT_SECRET_PRODUCTION` registered
- [ ] GitHub secret `DISCORD_REDIRECT_URI_ALLOWLIST_PRODUCTION` registered
- [ ] GitHub secret `DISCORD_USER_REF_SECRET_PRODUCTION` registered
- [ ] GitHub secret `DISCORD_OPERATIONS_WEBHOOK_URL_PRODUCTION` registered
- [ ] GitHub Environment variable `BROKER_CANONICAL_WORKERS_DEV_URL` registered
- [ ] GitHub Environment variable `BROKER_DEPLOY_SMOKE_DISALLOWED_MODEL_PRODUCTION` registered

## Operational safety reminders

- [ ] Manual revocation playbook is documented for operators: broker-side revocation alone is insufficient; the matching upstream OpenRouter child key must also be disabled or deleted.

## Automation scope we intend to build

- [x] Generate a deploy-time Wrangler config or equivalent environment-specific config with the production D1 `database_id`
- [x] Apply remote D1 migrations in CI
- [x] Replace the D1 `fingerprint_salt` bootstrap placeholder in CI before smoke traffic
- [x] Reconcile the production OpenRouter guardrail in CI before deploy / smoke
- [x] Push production Worker secret(s) in CI
- [x] Deploy the canonical Worker in CI
- [x] Run automated smoke tests after deploy
- [x] Fail the workflow if smoke tests fail
- [ ] Keep app / public traffic disconnected from the broker until smoke passes

## Settled execution choices

- [x] production Worker URL convention
  - use the canonical `workers.dev` URL from `puripuly-heart-broker`
  - defer custom domain / route cutover until after smoke passes
- [x] config strategy
  - generate a deploy-time Wrangler config with the production `database_id`
- [x] D1 migration / bootstrap command shape
  - `wrangler d1 migrations apply <db> --remote --config <generated-config>`
  - guarded bootstrap via `wrangler d1 execute <db> --remote --config <generated-config> --file <sql> --yes`
- [x] smoke-test script location
  - `broker/tests/deploy-smoke/`

## Direct production smoke test checklist

Minimum automated smoke coverage should include the broker’s real HTTP contract:

- [x] `GET /healthz`
- [x] `GET /v1/foundation`
- [x] `POST /v1/trial/challenge`
- [x] `POST /v1/trial/challenge/verify`
- [x] `GET /v1/trial/status`
- [x] `POST /v1/providers/openrouter/issue`
- [x] `POST /v1/auth/qq/assert` with a synthetic non-PII `deploy-smoke-qq-<uuid>` identity and a credential computed from `BROKER_DEPLOY_SMOKE_QQ_AUTH_HMAC_PSK`; when production issuance configuration is present, this returns `issued` with a one-time `openrouter_api_key`

Implementation notes for the minimum smoke path:

- [x] create a fresh installation ID and Ed25519 device keypair per run
- [x] use canonical signing for `verify`, `status`, and `issue`
- [x] use a managed issue payload drawn from the curated allowlist:
  - `reason = llm_start`
  - `budget_usd = 0.07`
  - `model = google/gemma-4-26b-a4b-it`
- [x] after issue, probe positive routing for:
  - `qwen/qwen3.5-flash-02-23`
  - `deepseek/deepseek-v4-flash`
  - `google/gemini-2.5-flash-lite`
- [x] prefer existing broker test helpers where possible:
  - `broker/tests/test-support/ed25519.ts`
  - `broker/tests/test-support/trial-api.ts`
- [x] redact or avoid logging full `issue` and QQ assertion responses because successful responses contain `openrouter_api_key`

Recommended failure-path smoke coverage:

- [ ] invalid signature rejection
- [ ] expired / invalid challenge rejection
- [ ] status request with unknown installation rejection
- [ ] issue request without a valid release token rejection

Canonical production smoke now also verifies:

- [x] synthetic non-PII QQ Managed issuance returns `issued`, includes a one-time `openrouter_api_key`, and redacts the key, QQ identity, credential, and derived subject material from success/failure output
- [x] issued QQ child-key metadata reflects the managed limit / expiry contract
- [x] repeat QQ assertion for the same synthetic subject proves one-time key-bearing behavior by returning lifetime-used behavior without key recovery
- [x] the deploy path performs guardrail reconcile before deploy / smoke
- [x] positive routing for `qwen/qwen3.5-flash-02-23`, `deepseek/deepseek-v4-flash`, and `google/gemini-2.5-flash-lite`
- [x] a known disallowed model is rejected after guardrail assignment
- [x] operational docs distinguish issuance-disabled verification-only behavior from enabled production issuance failures; enabled failures must use bounded retryable/internal envelopes rather than falling back to `verified`

Recommended later expansion:

- [ ] app-to-broker smoke test for one real managed onboarding path

## Open questions still to settle

- [x] first canonical deployment trigger
  - use manual workflow dispatch from `dev` for the initial canonical production deploy
  - keep protected-branch / tag-trigger expansion as a later option, not the current path

## Immediate next step

Repo-side direct-production automation now exists:

1. `broker/scripts/render-production-wrangler-config.mjs`
2. `broker/scripts/render-fingerprint-bootstrap-sql.mjs`
3. `broker/tests/deploy-smoke/canonical-production.spec.ts`
4. `.github/workflows/deploy-broker-direct.yml`

The remaining rollout work is operational, not repo-side automation:

1. register the production Environment secrets / variable
    - `OPENROUTER_MANAGED_API_KEY_PRODUCTION` remains transitional compatibility only
    - `OPENROUTER_MANAGEMENT_API_KEY_PRODUCTION` and `OPENROUTER_MANAGED_GUARDRAIL_ID_PRODUCTION` are required for child-key issuance
    - `OPENROUTER_MANAGED_USER_HMAC_SECRET_PRODUCTION` must be registered so deploy sync can populate runtime `OPENROUTER_MANAGED_USER_HMAC_SECRET` for deterministic managed OpenRouter user ids
    - `QQ_AUTH_HMAC_PSK_PRODUCTION` must be registered so deploy sync can populate runtime `QQ_AUTH_HMAC_PSK` and the smoke test can exercise QQ production issuance through `POST /v1/auth/qq/assert`
    - `DISCORD_CLIENT_ID_PRODUCTION`, `DISCORD_CLIENT_SECRET_PRODUCTION`, `DISCORD_REDIRECT_URI_ALLOWLIST_PRODUCTION`, and `DISCORD_USER_REF_SECRET_PRODUCTION` must be registered so deploy sync can populate runtime `DISCORD_CLIENT_ID`, `DISCORD_CLIENT_SECRET`, `DISCORD_REDIRECT_URI_ALLOWLIST`, and `DISCORD_USER_REF_SECRET` for Discord OAuth onboarding
    - `BROKER_DEPLOY_SMOKE_DISALLOWED_MODEL_PRODUCTION` must be set to a model blocked by the configured guardrail
2. create or confirm the production D1 database and capture its `database_id`
3. review the manual workflow inputs / guards
4. run the first canonical deployment and smoke
5. keep app / public traffic disconnected until that smoke passes and is reviewed

If that live smoke still cannot route Qwen or Gemini after the repo-controlled guardrail reconcile, inspect account-level OpenRouter privacy / provider settings outside repo control.

## Related references

- `.opencode/docs/plans/13-verify-openrouter-gemma4-trial-broker-rollout.yaml`
- `.opencode/docs/openrouter-gemma4-trial-broker-rollout-execution-order.md`
- `broker/wrangler.jsonc`
- `broker/README.md`
- `broker/migrations/0000_define_broker_persistent_state.sql`
- `broker/tests/test-support/ed25519.ts`
- `broker/tests/test-support/trial-api.ts`
