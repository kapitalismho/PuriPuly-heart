# Broker service foundation

This directory establishes the managed-trial broker as a separate deployable service in the monorepo.

## Explicit rollout boundary

- Runtime stack: TypeScript + Hono on Cloudflare Workers with native D1 and Worker secrets.
- Hosting scope: single-region rollout assumption for the initial Worker deployment, with D1 `location_hint` set to `apac`.
- Managed free-trial path: `OpenRouter` + curated allowlist `google/gemma-4-26b-a4b-it`, `google/gemma-4-31b-it`, `deepseek/deepseek-v4.1-flash`, `deepseek/deepseek-v4-flash-0731`, and `deepseek/deepseek-v4-flash`.
- Inference boundary: the app talks to OpenRouter directly; the broker remains a trial and credential broker.
- Out of scope in this foundation: translation proxying, multi-region deployment, KV, R2, and admin dashboard work.

## Deploy note

`broker/wrangler.jsonc` intentionally uses a non-secret placeholder `database_id`. A real Cloudflare D1 identifier must be supplied in deployment-specific configuration before the service is deployed.

Use `pnpm --filter @puripuly-heart/broker run verify:config` to exercise the pinned Wrangler CLI against `broker/wrangler.jsonc` without requiring cloud credentials.

## Direct deploy automation

- `broker/scripts/render-production-wrangler-config.mjs` renders a temporary deploy-time Wrangler config from `broker/wrangler.jsonc`, injects the production D1 `database_id`, and fails if the checked-in worker name stops being the canonical `puripuly-heart-broker`.
- `broker/deploy/fingerprint-bootstrap.template.sql` plus `broker/scripts/render-fingerprint-bootstrap-sql.mjs` render guarded bootstrap SQL for `wrangler d1 execute --file ... --yes`. The rendered SQL only replaces the migration placeholder and fails before mutating `broker_config` if the placeholder is already gone.
- `.github/workflows/deploy-broker-direct.yml` is the manual `workflow_dispatch` deployment path. It exports production D1 to a restricted seven-day artifact before migrations, bootstraps the fingerprint salt, reconciles the managed guardrail, syncs worker secrets, deploys the Worker, verifies health, finalizes retired runtime configuration and runs the production smoke. Existing databases must have completed `0021` and `0024` with the prior supported Worker before deploying the referral-free runtime. Empty databases complete those privacy migrations before their first deployment.
- Ordinary broker deployments exclude `0025_drop_legacy_translation_telemetry.sql` from both the pre-backfill and post-backfill migration sets. Migration code is ready, but that does not mean the live production schema or rows have been deleted. Before an authorized deletion, the operator must inspect production aggregate row counts and date ranges and confirm that no external consumer or operator-created dependency still uses the retired tables, then obtain separate explicit approval for the production deletion. The workflow's deletion controls record dispatch intent only and do not prove that the external review or approval occurred. To use the authorized path, set `delete_legacy_translation_telemetry` to `true` and type `delete retired translation telemetry from production D1` into `confirm_legacy_translation_telemetry_deletion`; the generic deploy confirmation cannot trigger deletion. The workflow then stages only `0025` and applies it through Wrangler after the deployed Worker passes health and production smoke checks, using the backup exported and uploaded earlier in the same run.
- `OPENROUTER_MANAGED_API_KEY_PRODUCTION` remains transitional runtime compatibility only; `OPENROUTER_MANAGEMENT_API_KEY_PRODUCTION` drives managed child-key creation / cleanup, `OPENROUTER_MANAGED_GUARDRAIL_ID_PRODUCTION` assigns the production guardrail to each issued key, and `OPENROUTER_MANAGED_USER_HMAC_SECRET_PRODUCTION` is copied into the runtime secret `OPENROUTER_MANAGED_USER_HMAC_SECRET` so the worker can derive a deterministic versioned managed OpenRouter user id per installation or QQ subject.
- `QQ_AUTH_HMAC_PSK_PRODUCTION` is copied into the runtime secret `QQ_AUTH_HMAC_PSK` for `POST /v1/auth/qq/assert`. The endpoint is production issuance-capable when runtime issuance configuration is present (`QQ_AUTH_HMAC_PSK`, `OPENROUTER_MANAGEMENT_API_KEY`, and `OPENROUTER_MANAGED_GUARDRAIL_ID` are all non-blank). The issuance-disabled verification-only behavior preserves `verified` / `already_verified` compatibility without touching `qq_managed_entitlements`; when issuance is enabled, OpenRouter, guardrail, cleanup, or D1 failures return a bounded retryable/internal error envelope instead of falling back to verification-only success. The PSK value, raw QQ identity, raw credential, and raw key-bearing payloads must stay out of source, docs, logs, and test output.
- `TELEMETRY_SUBJECT_HMAC_SECRET_PRODUCTION` is copied into the runtime secret `TELEMETRY_SUBJECT_HMAC_SECRET` for `POST /v1/telemetry/app-active-day`. Production migration rollout must take a D1 backup before applying `0015_add_app_active_days.sql`; that forward-only migration creates the isolated minimal app active-day table while preserving the previous Worker's required abuse-control shape and the previous translation telemetry tables and rows. After the new Worker passes health verification, `broker/deploy/finalize-app-active-day.sql` removes the obsolete telemetry endpoint rate-limit setting.
- `DISCORD_CLIENT_ID_PRODUCTION`, `DISCORD_CLIENT_SECRET_PRODUCTION`, `DISCORD_REDIRECT_URI_ALLOWLIST_PRODUCTION`, and `DISCORD_USER_REF_SECRET_PRODUCTION` are copied into the runtime secrets `DISCORD_CLIENT_ID`, `DISCORD_CLIENT_SECRET`, `DISCORD_REDIRECT_URI_ALLOWLIST`, and `DISCORD_USER_REF_SECRET` for Discord OAuth onboarding.
- `DISCORD_OPERATIONS_WEBHOOK_URL_PRODUCTION` is copied into the runtime secrets `DISCORD_IMMEDIATE_ALERT_WEBHOOK_URL` and `DISCORD_DAILY_REPORT_WEBHOOK_URL` so the broker can send real-time alerts and the `puripuly_daily_summary.v2` report. The minute-resolution cron consults `abuse_controls.dailyReport` and the v2 delivery ledger, then sends at 00:05 UTC for the last completed UTC date.
- The deploy reconcile step sets `allowed_models` to `google/gemma-4-26b-a4b-it`, `google/gemma-4-31b-it`, `deepseek/deepseek-v4.1-flash`, `deepseek/deepseek-v4-flash-0731`, and `deepseek/deepseek-v4-flash`, clears provider restrictions inside the guardrail (`allowed_providers` / `ignored_providers`), and sets `enforce_zdr = false` before smoke.
- The deploy smoke verifies a synthetic non-PII QQ Managed assertion through `POST /v1/auth/qq/assert`, expects `status: "issued"` with a one-time `openrouter_api_key`, verifies duplicate/lifetime guardrail behavior without key recovery, verifies issued child-key metadata through `https://openrouter.ai/api/v1/key`, proves positive routing through `google/gemma-4-31b-it`, `deepseek/deepseek-v4.1-flash`, `deepseek/deepseek-v4-flash-0731`, and `deepseek/deepseek-v4-flash`, and still probes `https://openrouter.ai/api/v1/chat/completions` with `BROKER_DEPLOY_SMOKE_DISALLOWED_MODEL_PRODUCTION` to confirm guardrail enforcement.
- Config verification is split by surface: `pnpm --filter @puripuly-heart/broker run verify:config` checks the checked-in Worker binding contract, while the direct-deploy guard step fails before migrations if the production secrets `QQ_AUTH_HMAC_PSK_PRODUCTION`, `TELEMETRY_SUBJECT_HMAC_SECRET_PRODUCTION`, `OPENROUTER_MANAGEMENT_API_KEY_PRODUCTION`, or `OPENROUTER_MANAGED_GUARDRAIL_ID_PRODUCTION` are missing or blank. Neither path prints secret values.
- Account-level OpenRouter privacy / provider settings remain outside repo control and may still narrow effective routing even after the guardrail reconcile; the production smoke is the proof point for the resulting path.
- The workflow expects CI-managed secrets / vars in the `production` GitHub Environment: `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID`, `BROKER_D1_DATABASE_ID_PRODUCTION`, `OPENROUTER_MANAGED_API_KEY_PRODUCTION`, `OPENROUTER_MANAGEMENT_API_KEY_PRODUCTION`, `OPENROUTER_MANAGED_GUARDRAIL_ID_PRODUCTION`, `OPENROUTER_MANAGED_USER_HMAC_SECRET_PRODUCTION`, `QQ_AUTH_HMAC_PSK_PRODUCTION`, `TELEMETRY_SUBJECT_HMAC_SECRET_PRODUCTION`, `DISCORD_CLIENT_ID_PRODUCTION`, `DISCORD_CLIENT_SECRET_PRODUCTION`, `DISCORD_REDIRECT_URI_ALLOWLIST_PRODUCTION`, `DISCORD_USER_REF_SECRET_PRODUCTION`, `DISCORD_OPERATIONS_WEBHOOK_URL_PRODUCTION`, `BROKER_CANONICAL_WORKERS_DEV_URL`, and `BROKER_DEPLOY_SMOKE_DISALLOWED_MODEL_PRODUCTION`.
- App / public traffic must stay disconnected from the broker until the direct deploy smoke run passes and is explicitly reviewed.

## Referral removal and schema retirement

The application and Broker no longer accept referral codes, expose Talk Together Pass, reserve rewards or run reward settlement. Delivery ACK activates the credential without scheduling a bonus. Existing credentials, entitlement identity, expiry and already-applied provider allowances are preserved; removal does not lower limits or revoke keys. Pending rewards are not processed.

`0027_remove_managed_referrals.sql` removes obsolete referral tables, configuration and columns while preserving operations, attempts, deliveries, cleanup ownership and entitlements. Historical migrations remain immutable.

Ordinary deployment deliberately excludes `0027` from both migration stages. Applying it before the previous Worker has stopped can break its in-flight ACK or scheduled work. Schema retirement requires a separately authorized maintenance action:

1. Complete the historical `0021`/`0024` privacy migration and verify its gates. Do not force `keyed_only` or skip unconverted request/hook data.
2. Back up D1, deploy and verify the referral-free Worker, then establish that prior-version requests and scheduled invocations have drained. A successful health request alone is not this evidence.
3. Apply only the pending referral-removal migration through a dedicated staged Wrangler migration directory. Do not accidentally include separately held `0025` telemetry deletion.
4. Read back the migration journal and schema. Verify surviving operation/attempt/delivery relationships, cleanup ownership and existing entitlement budgets. No provider budget PATCH is part of retirement.

Repository migration availability and local migration checks are not evidence that production data was removed. An already-dispatched old provider request cannot be recalled by deleting its local job; operational drain evidence is required. The separate telemetry-deletion approval remains unchanged.

## Verification environment

Broker verification is Linux-only. Run `pnpm install`, Vitest, and Wrangler from a Linux-native workspace (for example, a WSL-internal path or a regular Linux checkout), not from Windows or shared `/mnt/c/...` `node_modules`.

## Trial challenge + verify handshake

- `POST /v1/trial/challenge`
  - request: `installation_id`, base64url `device_public_key`, `app_version`
  - public input bounds: `installation_id` `1-128` chars, `app_version` `1-64` chars
  - `installation_id` and `app_version` must not be blank or whitespace-only, and must not contain embedded control characters or newline separators
  - rejects client-supplied `hardware_hash`, `signed_at`, and `signature`
  - response: `challenge`, `challenge_expires_at`, `fingerprint_salt`, normalized `managed_state`, and `current_entitlement`
  - challenge TTL: `5` minutes
  - never returns `release_token`, release-session state, or raw managed credentials
- `POST /v1/trial/challenge/verify`
  - request: `installation_id`, base64url `device_public_key`, `challenge`, `challenge_expires_at`, `hardware_hash`, `app_version`, `signed_at`, base64url `signature`
  - public input bounds: `installation_id` `1-128` chars, `app_version` `1-64` chars, `hardware_hash` `1-128` chars
  - `installation_id`, `app_version`, and `hardware_hash` must not be blank or whitespace-only, and must not contain embedded control characters or newline separators
  - supported timestamp subset for `challenge_expires_at` and `signed_at`: `YYYY-MM-DDTHH:MM:SS(.mmm)?(Z|±HH:MM)` with a real calendar date/time
  - Ed25519 signature payload is canonical UTF-8 text joined by newlines in this order:
    1. `installation_id`
    2. `device_public_key`
    3. `challenge`
    4. `challenge_expires_at`
    5. `hardware_hash`
    6. `app_version`
    7. `signed_at`
  - enforces signed clock skew within `±60` seconds
  - uses the already registered `device_public_key`; verify does not rebind installation identity
  - successful verify consumes the active challenge, persists `hardware_hash` with the issued challenge salt version, and returns `release_token`, `release_token_expires_at`, normalized `managed_state`, and `current_entitlement`
  - release token TTL: `15` minutes
- `GET /v1/trial/status`
  - query: `installation_id`
  - headers: `X-Puripuly-Timestamp`, `X-Puripuly-Signature`
  - `installation_id` keeps the same public bound: `1-128` chars
  - `installation_id` must not be blank or whitespace-only, and must not contain embedded control characters or newline separators
  - `X-Puripuly-Timestamp` must be a valid ISO-8601 timestamp in the same strict subset used by verify
  - `X-Puripuly-Signature` must transport a base64url Ed25519 signature
  - canonical status-signing payload is UTF-8 text joined by newlines in this order:
    1. `installation_id`
    2. `timestamp`
  - enforces signed clock skew within `±60` seconds
  - status requests are verified against the already registered `device_public_key` for the installation; unknown `installation_id` values return `installation_not_found`
  - response: normalized `managed_state`, `current_entitlement`, and `onboarding_eligibility`
  - onboarding eligibility is broker-side metadata only: no entitlement returns `{ eligible: true, reason: "discord_required", requires_discord_oauth: true }` so the app can show the Discord dialog without a silent browser launch or `authorization_url`
  - current entitlements are ineligible for new Discord onboarding and return `{ eligible: false, reason: <stored entitlement status>, requires_discord_oauth: false }`; `pending_release`, `active`, `expired`, and `revoked` reasons come from the stored entitlement status rather than lifecycle derivation
  - `expired` and `revoked` are returned as `200` lifecycle data, not public error codes
  - live remaining budget stays upstream in OpenRouter metadata instead of being mirrored into broker status
- `POST /v1/providers/openrouter/issue`
  - request: `installation_id`, base64url `device_public_key`, base64url `release_token`, `hardware_hash`, `reason`, `budget_usd`, `model`, `signed_at`, base64url `signature`
  - `installation_id` and `hardware_hash` keep the same public bound: `1-128` chars and must not be blank or whitespace-only, and must not contain embedded control characters or newline separators
  - activation reason is fixed to `llm_start`
  - `budget_usd` must match the managed-trial hard limit and `model` must be one of the curated managed OpenRouter models
  - supported timestamp subset for `signed_at`: `YYYY-MM-DDTHH:MM:SS(.mmm)?(Z|±HH:MM)` with a real calendar date/time
  - Ed25519 signature payload is canonical UTF-8 text joined by newlines in this order:
    1. `installation_id`
    2. `device_public_key`
    3. `release_token`
    4. `hardware_hash`
    5. `reason`
    6. `budget_usd`
    7. `model`
    8. `signed_at`
  - enforces signed clock skew within `±60` seconds
  - consumes the `pending_release` token, upgrades the entitlement to `active`, and returns terminal `managed_key_unrecoverable` for same-session retries after activation because the issued child key cannot be recovered
  - success response returns `openrouter_api_key`, distinct `managed_credential_ref`, optional `openrouter_user_id`, normalized `managed_state`, `expires_at`, `budget_usd`, and `model`
  - `openrouter_api_key` is a newly created per-installation OpenRouter child key, not the shared worker secret
  - when `OPENROUTER_MANAGED_USER_HMAC_SECRET` is configured, `openrouter_user_id` carries the deterministic versioned managed OpenRouter user id for that installation; otherwise the field is omitted
  - the child key is created with the managed-trial limit (`0.07` USD), a three-month expiry anchored to `issued_at`, and the configured managed guardrail before the broker returns it
  - live remaining budget and usage stay upstream in OpenRouter metadata and are not mirrored into the issue response
  - manual broker revocation is only a broker-local stop for future onboarding; because the app calls OpenRouter directly after issue succeeds, operators must also disable or delete the upstream OpenRouter child key when they need a revocation to stop existing direct use
- `POST /v1/auth/qq/assert`
  - required request fields are `qq_identity`, `credential`, and `asserted_at`; delivery support and bounded installation/operation authority accompany managed issuance; no device key or hardware hash is used
  - `credential` remains `HMAC-SHA256-HEX(QQ_AUTH_HMAC_PSK, qq_identity)`; installation metadata and the bounded `asserted_at` timestamp are not part of that payload
  - `QQ_AUTH_HMAC_PSK` is mandatory for all QQ assertion handling; `OPENROUTER_MANAGEMENT_API_KEY` and `OPENROUTER_MANAGED_GUARDRAIL_ID` are the runtime gate for production issuance
  - with the OpenRouter issuance gate disabled, valid assertions preserve verification-only compatibility and return `verified` / `already_verified` without creating or mutating `qq_managed_entitlements`
  - with the gate enabled, an eligible first assertion returns `issued`, `qq_subject_ref`, one-time top-level `openrouter_api_key`, `managed_credential_ref`, `expires_at`, optional `openrouter_user_id`, and delivery-ACK metadata when the client advertises ACK support; the Broker never stores the raw key
  - duplicate active, cleanup-required, or revoked QQ entitlements return `qq_lifetime_used`; concurrent current issuance returns `qq_already_issuing`; invalid credentials return `qq_credential_invalid`
  - new keys use the base `0.07` USD budget; ACK does not change their allowance
  - installation binding is independent of retired incentive configuration; ambiguous resources must be cleaned before another issuance attempt
  - QQ lifetime, monitoring and cleanup use derived `qq_subject_ref` / `issue_ref` metadata and must not create fake installation rows or persist/log raw QQ identity or credential
  - the former `/v1/auth/qq/status` Pass endpoint is removed; managed-operation status/resume remains the recovery interface
- `POST /v1/telemetry/app-active-day`
  - request: `anonymous_id` and `active_date_utc` as `YYYY-MM-DD`
  - accepts only those two fields and only the current or immediately previous UTC date; malformed JSON, invalid values, stale dates, and additional metadata return the existing public `invalid_request` envelope
  - derives `subject_ref = ph-app-subject-v1_ + HMAC-SHA256-HEX(TELEMETRY_SUBJECT_HMAC_SECRET, anonymous_id)` and persists only that derived reference and UTC active date
  - duplicate same-subject same-date payloads are no-ops and cannot inflate active-user counts
  - has no endpoint rate limit, does not read or persist IP addresses, and does not write request events
  - app-version, operating-system, installation, Discord, QQ, hardware, provider, model, language, output-route, translation-content, receipt-timestamp, and other metadata are neither accepted nor stored
  - active-day rows older than the rolling 35-day retention window are deleted by the scheduled job

## Daily summary v2

- `puripuly_daily_summary.v2` is generated at 00:05 UTC for the last completed UTC date.
- `window_start <= observed_at < window_end` is used for both Discord and QQ delivered-key rows, so UTC midnight belongs to exactly one report.
- DAU covers `report_date_utc`; WAU and MAU cover the seven and thirty completed UTC dates ending on that date.
- The report contains only delivered-key total/Discord/QQ counts and app-launch DAU/WAU/MAU calculated from `app_active_days`.
- Healthy-state security fields, legacy challenge/verify funnel metrics, ASN analysis, stickiness, and D1/D7/D30 cohort rows are excluded.
- A per-report-date D1 lease prevents overlapping cron invocations from posting the same report twice. Failed dates remain pending across UTC midnight, retries keep their original fixed window, and later completed dates catch up in order without allowing retention to delete unreported issue events.
- Delivery ACK finalization atomically commits the source owner, one idempotent issue-success event, and the acknowledgement ledger before evaluating immediate incidents. Stale reconciliation promotes already-finalized pending rows to acknowledged; otherwise it acquires a durable cleanup claim, recovers abandoned claims only after the scheduled invocation limit, and atomically terminalizes the owner and delivery ledger.

## Immediate abuse incidents

- Source-aware successful-delivery events feed a rolling 60-minute issuance count with one `warning` threshold and one automatic `brake` threshold. Healthy observations do not call the immediate-alert webhook, and a transition that crosses both thresholds emits only the brake incident.
- A warning is emitted once per above-threshold interval and rearms only after the count drops back to or below its threshold. A brake incident is emitted only for the successful persisted transition into the automatic brake state.
- Discord and QQ managed child-key cleanup failures, including stale-delivery reconciliation failures, persist `cleanup_required` where ownership exists and emit one immediate cleanup incident. An indeterminate provider create result also preserves lifetime-blocking remediation state and alerts instead of permitting another key. Notification failures are audited without replacing the original issuance or cleanup result.
- Immediate incident payloads contain only operational counts, thresholds, source, cleanup phase/state, and nullable derived credential references. They must not contain raw anonymous identifiers, managed identities, translation content, audio, or API keys.

## Persistence model

`broker/src/persistence.ts` and `broker/migrations/*.sql` define the D1-backed state contract and its upgrade path.

- `0001_harden_installation_public_inputs.sql` rebuilds `installations` (and the dependent `openrouter_entitlements` table) under deferred foreign-key checks so already-initialized clean schemas pick up the hardened public-input constraints.
- `0002_add_entitlement_verified_hardware_snapshot.sql` adds `verified_hardware_hash` and `verified_hardware_hash_salt_version` to `openrouter_entitlements` for the verified release-session hardware snapshot consumed by `/v1/providers/openrouter/issue`.
- `0003_add_abuse_runtime_state_and_issue_success_events.sql` adds the persisted abuse runtime-state row plus append-only issue-success and runtime-audit tables used by alerting, brake state, daily summary delivery, and retention.
- `0004_add_discord_oauth_managed_issue.sql` adds Discord OAuth session and identity storage plus Discord-managed issue columns on `openrouter_entitlements`.
- `0005`–`0007`, `0016`, `0017` and `0019` remain historical referral migrations. Their feature storage is retired by `0027`; they do not describe a current referral API.
- `0008_add_qq_auth_assertions.sql` adds the `qq_auth_assertions` evidence table and inserts the `qqAuthAssertIp` abuse-control default without replacing operator-tuned `abuse_controls` values.
- `0009_add_qq_managed_entitlements.sql` adds the `qq_managed_entitlements` lifecycle table for QQ production issuance without rewriting existing assertion evidence.
- `0010_source_aware_issue_success_events.sql` rebuilds `broker_issue_success_events` so successful issue monitoring is source-aware: Discord rows keep installation identity, while QQ rows use `issue_source = 'qq'`, nullable `installation_id`, and `subject_ref = qq_subject_ref` instead of fake installation rows.
- `0011_add_telemetry_active_days.sql` creates the isolated `telemetry_active_days` table and additively inserts the telemetry endpoint IP rate-limit default into `abuse_controls`; production rollout requires a pre-migration D1 backup/export before this forward migration is applied.
- `0012_add_managed_key_delivery_ack.sql` adds the shared Discord/QQ delivery acknowledgement ledger and delivery-pending lifecycle states.
- `0013_add_telemetry_subjects_and_daily_summary_v2.sql` creates and backfills `telemetry_subjects`, keeps it synchronized for the previous Worker during rollout, creates the v2 delivery ledger, preserves existing active-day rows, sets the daily report schedule to 00:05 UTC, and raises issue-event retention to the report-safe two-day minimum without replacing unrelated operator-tuned controls. It intentionally retains `includeZeroActivity` while the previous Worker may still run; the deploy workflow removes that dead field only after the new Worker passes its health check.
- `0014_simplify_abuse_incidents.sql` additively derives the `warning` and `brake` thresholds, the ordered warning observation state, and the request-event safety margin from existing persisted controls. It also adds a QQ child-key-creation-start marker so ambiguous post-provider failures cannot be stale-reclaimed into a second key. Legacy alert/ASN JSON fields remain during the migration-before-deploy compatibility window; unused physical columns and indexes require a separate forward migration after stabilization.
- `0015_add_app_active_days.sql` creates `app_active_days` with only HMAC-derived subject and UTC-date columns while retaining the previous Worker's required abuse-control shape and preserving the legacy translation telemetry tables and rows without mixing them into app usage metrics. The deploy workflow removes `telemetryTranslationSuccessDayIp` with `broker/deploy/finalize-app-active-day.sql` only after the new Worker passes its health check.
- `0025_drop_legacy_translation_telemetry.sql` is the code-ready, idempotent forward removal of the retired translation-success trigger, indexes, tables, and rows. It does not alter `app_active_days` or current usage aggregation. The direct-deploy workflow holds it out of ordinary migration application and requires the separately authorized deletion path described above.
- `0026_managed_attempt_cleanup_ownership.sql` adds attempt-scoped cleanup ownership and delivery-ACK exclusion.
- `0027_remove_managed_referrals.sql` retires referral storage without changing existing entitlement budgets or credential references; its separate rollout prerequisites are described above.

- `broker_config`
  - columns: `key`, `value`, `updated_at`
  - current bootstrap rows include `fingerprint_salt`, `abuse_controls`, `abuse_runtime_state` and `network_identity_migration`
  - runtime-tunable non-secret operational controls live in `abuse_controls` so operators do not need code changes for threshold updates
  - persisted mutable runtime state lives separately in `abuse_runtime_state` so brake status, alert latches, and legacy v1 daily-heartbeat delivery metadata do not get mixed into the editable threshold policy blob
  - malformed `abuse_controls` payloads fall back to the built-in default layout/thresholds instead of disabling enforcement or surfacing 500s
  - constraints: keys are limited to the supported config rows for this rollout and `value` must be valid JSON
  - `abuse_controls` fixes the settled endpoint/dimension layout:
    - `POST /v1/trial/challenge`: per IP, `10` requests / `15` minutes
    - `POST /v1/trial/challenge/verify`: per `installation_id`, `5` requests / `15` minutes
    - `POST /v1/providers/openrouter/issue`: per `installation_id`, `3` requests / `15` minutes
    - `GET /v1/trial/status`: per `installation_id`, `30` requests / `15` minutes
    - `POST /v1/auth/qq/assert`: per IP via `qqAuthAssertIp`, `20` requests / `15` minutes
    - global UTC-day cap on new active entitlements, counted by `issued_at` semantics even if an entitlement is later revoked, stored as a runtime-configurable broker value
- `broker_issue_success_events`
  - append-only successful issue observations recorded only after child-key creation and entitlement persistence both succeed
  - feeds immediate-alert evaluation, source-aware completed-day delivery totals, and retention cleanup
  - columns include `issue_source`, nullable `installation_id`, `subject_ref`, `managed_credential_ref`, safe network metadata, and `observed_at`
  - Discord rows use `issue_source = 'discord'`, retain `installation_id`, and set `subject_ref` to that same installation identity; QQ rows use `issue_source = 'qq'`, leave `installation_id` `NULL`, and set `subject_ref` to `qq_subject_ref`
  - QQ monitoring/reporting must not synthesize installation rows, and no raw QQ identity, raw credential, raw OpenRouter key, raw Broker payload, or raw OpenRouter payload belongs in issue-success events
- `broker_abuse_runtime_audit`
  - append-only audit trail for brake transitions and other persisted abuse-runtime actions
- `broker_request_events`
  - append-only request observations used for per-endpoint rate limiting and cross-endpoint velocity hooks
  - columns: `id`, `endpoint`, `ip`, `installation_id`, `observed_at`
  - indexes cover endpoint-scoped and subject-scoped sliding-window lookups
  - retention is calculated at cleanup time from the longest configured endpoint rate-limit window and longest active, unexpired velocity-hook window, plus the explicit `requestEventSafetyMarginDays` margin; the default margin is one day
- `telemetry_subjects` and `telemetry_active_days`
  - retired translation-success storage removed by the code-ready forward migration `0025_drop_legacy_translation_telemetry.sql`
  - ordinary broker deployments hold `0025` out of every automatically applied migration set, so repository state must not be interpreted as proof that the live production tables or rows are gone
  - no app active-day writes or app DAU/WAU/MAU reads use these tables; authorized production removal still requires the prerequisite production aggregate and external-consumer inspection described above
- `app_active_days`
  - app-launch active-day rows keyed by `(subject_ref, active_date_utc)`
  - the only columns are HMAC-derived `subject_ref` and UTC active date; there are no IP, receipt timestamp, metadata, or raw anonymous-ID columns
  - scheduled retention keeps only the 35-day window needed for rolling thirty-day MAU reporting
- `broker_daily_summary_deliveries`
  - one row per `report_date_utc` coordinates the v2 send with a bounded lease and records the delivered outcome
  - columns: `report_date_utc`, `status`, `lease_token`, `lease_expires_at`, `attempted_at`, and `delivered_at`
  - a failed webhook expires but preserves its pending claim so the same fixed completed-day window survives midnight; a delivered row permanently suppresses duplicate sends for that report date

- `broker_velocity_cap_hooks`
  - explicit operator-controlled cross-endpoint velocity hooks with observable public outcomes
  - columns: `id`, `subject_type`, `subject_value`, `max_requests`, `window_minutes`, `outcome_code`, `outcome_class`, `outcome_subcode`, `reason`, `active`, `created_at`, `expires_at`
  - supported subjects: `ip`, `installation_id`
- `broker_abuse_subject_hooks`
  - explicit denylist, reputation, and fast-revocation hooks with observable outcomes
  - columns: `id`, `hook_kind`, `subject_type`, `subject_value`, `outcome_code`, `outcome_class`, `outcome_subcode`, `reason`, `active`, `created_at`, `expires_at`
  - supported hook kinds: `denylist`, `reputation`, `revocation`
  - supported subjects: `ip`, `installation_id`, `hardware_hash`
- `installations`
  - columns: `installation_id`, `device_public_key`, `hardware_hash`, `hardware_hash_salt_version`, `app_version`, `challenge`, `challenge_expires_at`, `challenge_salt_version`, `created_at`, `last_seen_at`
  - constraints: `installation_id` primary key, `device_public_key` unique, `hardware_hash` indexed, bounded persisted public text (`installation_id <= 128`, `app_version <= 64`, `hardware_hash <= 128` when present), no blank/whitespace-only persisted public values, and rejected embedded control/newline characters for those persisted public fields
  - update rules: each challenge overwrites `challenge`, `challenge_expires_at`, `challenge_salt_version`, and `app_version`; it clears stored `hardware_hash` / `hardware_hash_salt_version` only when lifecycle is `none` or `pending_release`, and preserves fingerprint state for `active`, `expired`, and `revoked`; verify clears the challenge fields; `hardware_hash` stays `NULL` until verify succeeds
- `openrouter_entitlements`
  - zero or one row per installation, keyed by `installation_id` when present
  - columns: `installation_id`, `status`, `budget_usd`, `managed_credential_ref`, `issued_at`, `expires_at`, minimal release-session columns `release_session_ref`, `release_token_hash`, `release_token_expires_at`, `verified_hardware_hash`, `verified_hardware_hash_salt_version`, `discord_user_ref`, `discord_issue_status`, `discord_issue_reserved_at`, `discord_issue_delivered_at`
  - constraints: `managed_credential_ref` unique, `status` indexed, `expires_at` indexed
  - `release_token_hash` is protected by a partial unique index when non-`NULL`
  - stored `status` values are `pending_release`, `active`, `expired`, and `revoked`; `none` is represented by the absence of a row
  - update rules: entitlement status, release-session fields, verified hardware snapshot, and credential metadata are updated in place; append-only entitlement history is intentionally out of scope for the initial rollout
  - remaining live budget stays upstream in OpenRouter metadata instead of being mirrored into broker storage; the release token remains installation-bound, one-time, and `15` minutes TTL
- `discord_oauth_sessions`
  - bounded Discord OAuth PKCE/session rows keyed by `state_hash`
  - columns include session/device/PKCE fields, Discord eligibility fields and lifecycle timestamps
  - indexed by installation/status/creation time and expiry
- `discord_identities`
  - durable HMAC Discord user reference uniqueness for Discord-managed issuance
  - columns: `discord_user_ref`, `entitlement_installation_id`, `status`, `ref_secret_version`, `created_at`, `updated_at`
  - `entitlement_installation_id` uses `ON DELETE SET NULL` so identity evidence is not cascade-deleted with aged installation rows
- `qq_auth_assertions`
  - QQ Bot HMAC assertion evidence keyed by derived `qq_subject_ref`; it supports verification-only compatibility and production issuance eligibility, but does not own entitlement lifecycle
  - columns: `qq_subject_ref`, `credential_hash`, `asserted_at`, `received_at`, `status`
  - stores only derived subject references and credential digests; raw QQ identities and raw credentials do not belong in D1, logs, docs, or checked-in tests
- `qq_managed_entitlements`
  - QQ Managed production issuance lifecycle keyed by derived `qq_subject_ref`; absence means the subject has not reserved or used production issuance
  - columns: `qq_subject_ref`, `status`, `issue_ref`, nullable `managed_credential_ref`, `budget_usd`, `reserved_at`, `issued_at`, `expires_at`, `delivered_at`, `created_at`, `updated_at`, and nullable `child_key_creation_started_at`
  - stored statuses are `issuing`, `delivery_pending`, `active`, `cleanup_required`, and `revoked`; `delivery_pending`, `active`, `cleanup_required`, and `revoked` block automatic reissue
  - `active` requires `managed_credential_ref`, `issued_at`, `expires_at`, and `delivered_at`; `cleanup_required` requires `managed_credential_ref`; stale `issuing` rows can be reclaimed only when neither a child-key hash nor a child-key-creation-start marker was recorded
  - new delivery remains at the base `0.07` budget; ACK changes delivery/entitlement state, not the provider limit
  - existing `qq_auth_assertions` rows without a `qq_managed_entitlements` row remain eligible for their first production issuance
  - stores derived and operational metadata only; raw QQ identities, raw credentials, and raw OpenRouter API keys do not belong in this table

## Managed operations and durable issuance

- Managed key issuance is one durable logical operation (`managed_operations`) that may span several provider-key attempts (`managed_operation_attempts`); raw OpenRouter API keys stay in transient Broker memory only and are never persisted in any form.
- Clients generate `operation_id` (`ph-mop-v1_` plus base64url of 24 random bytes) and `resume_token` (base64url of 32 random bytes) before issuance; issue requests carry both. Only `ph-mop-resume-v1_` plus the SHA-256 token hash is persisted, compared in constant time, and bound to source subject, installation, and device identity.
- Operation states are `AUTHENTICATED`, `ISSUE_READY`, `CREATING`, `CREATE_UNKNOWN`, `RECONCILING`, `CLEANUP_REQUIRED`, `CLEAN`, `RETRY_READY`, `DELIVERY_PENDING`, `ACTIVE`, and `FAILED`; recovery follows unknown, reconcile, cleanup-required, clean, retry-ready ordering, and fresh issuance is blocked until cleanup is verified against the provider by deterministic non-secret attempt key names.
- Public recovery routes are `POST /v1/providers/openrouter/managed-operation/status` and `POST /v1/providers/openrouter/managed-operation/resume` with body `{operation_id, resume_token, installation_id}`; known nonterminal re-POST is idempotent and cannot create a provider key unless the operation is retry-ready.
- Recovery authorization lasts exactly 60 minutes from operation creation, is installation/device bound, and is non-renewable. Expiry ends client issuance authority, not the Broker's responsibility to clean unresolved resources.
- Deliveries link to their operation and attempt; delivery ACK TTL is exactly 15 minutes (valid at the exact expiry instant, expired one millisecond later); stale delivery attempts are reconciled and cleaned server-side even if the client never returns.
- Destructive cleanup is bound to its claimed attempt. A stale resume or scheduled invocation cannot select a newer acknowledged key; incomplete or malformed provider enumeration is unknown, not proof that reissuance is safe.
- Provider enumeration explicitly includes disabled and expired keys, which the provider list otherwise excludes by default. Authenticated recovery scans at most 20 pages of 100 keys; scheduled cleanup shares a smaller invocation-wide page budget described below. Both persist continuation offsets to locate later resources. A matching resource can be cleaned at any offset. Reaching the end through pagination is still unknown and restarts coverage from zero: concurrent deletions can move a key behind an offset, even within one sweep. Only a complete, short first page can establish absence through listing. Large inventories without a matching resource therefore remain unresolved rather than authorizing unsafe reissuance; the provider supplies no snapshot guarantee. A known credential hash can instead be checked directly. The page bound is not a cleanup retry limit.
- A durable attempt marker retains responsibility for a dispatched create until an authoritative result settles it or its exact resource is located and confirmed cleaned. A network failure, ambiguous server response or elapsed timeout is not that proof. If the invocation is permanently lost and no resource can be located, the attempt remains unknown rather than authorizing a replacement. This outstanding-dispatch limit is separate from incomplete pagination, and cleanup discovery continues after client authorization expires.
- Upgrade also retains cleanup debt for older attempts whose historical state could have hidden a late-created resource, including null-reference attempts previously marked cleaned. Classification remains compatible with rows inserted by an older Worker during rollout. This attempt-only cleanup runs even when the parent operation has a newer ACTIVE credential; it does not rewrite that parent, entitlement or identity, and it excludes acknowledged resources. Bounded scans rotate through unresolved debt. An empty listing does not discharge uncertain legacy dispatch; confirmed cleanup of the exact resource does.
- Operational states correlate operation, attempt, delivery and credential-reference identifiers without logging raw keys, ACK or resume tokens, OAuth secrets, raw IPs or keyed digests; `managed_credential_ref` is nonrecoverable operational metadata and may appear in diagnostics.
- Routine all-zero managed-operation sweeps emit no console summary; nonzero sweeps retain their counters. `managed_operation_expired` describes changed database rows, not repeated no-op requests. Expiration reports the resulting `FAILED` state.

### Background cleanup policy and cost

- Operation-linked deliveries and current/historical attempts have one background destructive owner: the attempt sweep. The earlier delivery phase handles unlinked historical deliveries and finalized ACK repair, not a second attempt cleanup pass.
- Settled unresolved attempts wait at least six hours between unattended claims. The existing `legacy_cleanup_checked_at` field records the shared eligibility timestamp atomically with each claim, including authenticated cleanup. A released, not-yet-due attempt is not claimed, updated or enumerated. Authenticated recovery bypasses only this cooldown, not ownership or resource protection.
- A held cleanup token retains the existing sixteen-minute abandoned-claim recovery rule. Unresolved cleanup has no retry cap and does not expire with client authorization. A fixed interval avoids another retry counter, scheduling table or migration; schemas through `0026` and after the separately controlled `0027` remain supported.
- Each scheduled invocation shares at most six cleanup/repair candidates and six provider-list pages. At most two candidates go to the delivery phase; the remaining capacity serves one oldest-checked attempt queue, with creation time and stable IDs breaking ties. Historical classification separately handles at most six rows per pass. These are per-invocation caps, not a global lock; durable attempt claims fence competing owners.
- Authorization expiry and stale-creation maintenance retain their existing bounded minute-resolution passes. The 60-minute authorization and 15-minute ACK windows are unchanged, including ACK validity at the exact expiry instant. Exact-resource disable/delete/verification requests are additional to list pages; six candidates allow up to eighteen such provider requests.

The isolated [#286](https://github.com/kapitalismho/PuriPuly-heart/issues/286) probe used 42 uncertain historical attempts plus four current attempts, controlled provider responses, and the real scheduled entry point. Each supported schema was exercised over 25 timestamps spanning 370 simulated minutes. These are local SQLite observations, **not measured D1 billed writes**:

| Observation per schema | Result |
| --- | ---: |
| Backlog attempts given a first turn by minute 7 | 46 |
| SQL statements / table-row changes over sampled ticks | 1,021 / 215 |
| Provider-list pages / total scheduled provider requests | 90 / 107 |
| Table changes / provider requests at idle minutes 8, 15 and 359, each | 0 / 0 |

At the retry boundary, later-discovered historical/current orphans and previously failed linked/unlinked resources were cleaned. Remaining uncertain debt stayed unresolved; newer acknowledged credentials and their allowances survived. A fresh delivery accepted ACK at exactly fifteen minutes; authorization expiry and retention also progressed.

For a conservative recurring no-progress estimate, allow three table-row changes per historical attempt and six per current attempt, including a known-resource pin and linked-delivery bookkeeping. Charge each changed row as one table write plus removal/insertion for every index, even unchanged indexes: using at most six indexes gives a deliberately conservative factor of thirteen. Four six-hour rounds give `(42 × 3 + 4 × 6) × 4 × 13 = 7,800` projected written rows/day, leaving **92,200 of the 100,000-row allowance** for ordinary traffic, existing maintenance and one-time cleanup progress/classification. Pure empty-inventory retries cost less. [D1 index-write accounting](https://developers.cloudflare.com/d1/platform/pricing/) is distinct from SQLite table changes; this estimate is not a billing measurement.

The projection assumes settled retries, not repeated abandoned invocations. Sixteen-minute reclaim can admit work more frequently; crashes after partial work, concurrent invocations, larger backlogs and ordinary traffic can cost more. The policy does not guarantee account-wide quota safety. No production data, live provider or deployment was used for verification; production improvement requires a separately authorized rollout.

## Network identity HMAC and staged migration

- Persisted IP-derived identifiers use HMAC-SHA-256 with explicit key version and bounded UTC-day epoch, domain-separated per scope. Worker secrets required: `NETWORK_IDENTITY_HMAC_SECRET` (current) and optional `NETWORK_IDENTITY_HMAC_SECRET_PREVIOUS` (rotation overlap). Non-secret version vars required: `NETWORK_IDENTITY_HMAC_KEY_VERSION` (current, explicit positive integer, no silent default) and optional `NETWORK_IDENTITY_HMAC_KEY_VERSION_PREVIOUS`, which must be set together with the previous secret, be a positive integer, and differ from the current version. The render script takes `--network-identity-hmac-key-version` (required) and `--network-identity-hmac-key-version-previous` (optional) and injects these vars into the production wrangler config. The deploy workflow must copy `NETWORK_IDENTITY_HMAC_SECRET_PRODUCTION` (and, during rotation only, `NETWORK_IDENTITY_HMAC_SECRET_PREVIOUS_PRODUCTION`) into those runtime secrets and must never print their values; the guard step must fail before migrations when the current secret is missing or blank or the version pair is unpaired, malformed, or equal.
- Generate the current secret from at least 32 random bytes (base64url or hex) and store it only in the production GitHub Environment alongside the existing broker secrets. Rotation: install the previous/current secret and version pair, deploy, wait out the longest applicable request/hook accounting window, then retire the previous pair. Recovery/status bearers stay time-bounded even for an ACTIVE operation; server cleanup remains independent. Persisted digests carry their explicit key version so overlap and removal remain auditable without silently undercounting.
- Reads dual-compare current plus previous digests only inside still-active policy windows, enumerating every UTC-day epoch the window intersects; rotation never silently weakens rate limits, and cross-period correlation is bounded to the overlapping epoch.
- IP spellings are canonicalized before hashing (IPv4 leading zeros stripped, IPv6 lowercased with leading zeros stripped, `::ffff:a.b.c.d` mapped to the IPv4), so equivalent textual forms share one digest; undecodable values yield no digest and never match. Operator `ip` hooks (`broker_velocity_cap_hooks`, `broker_abuse_subject_hooks`) must be registered with stable keyed digests, not raw IPs. The scheduled worker converts parseable pre-cutover raw-IP hook values to stable digests and refuses to finalize the migration while any convertible ones remain; unparseable values never match, keep the `0021` gate aborting until removed manually, and must be deleted by the operator.
- Historical schema transition is staged: `0020_network_identity_hmac.sql` records `dual_write` when legacy data needs conversion, and the prior supported Worker completes keyed backfill before `0021_network_identity_purge.sql` and `0024_allow_unattributed_request_events.sql`. Those migrations retain their original privacy gates, including retiring referral data. Existing deployments must finish that transition before the referral-free runtime is installed; do not rewrite historical migrations or bypass surviving request/hook privacy checks.

## Retention and salt rotation

- Inactive `pending_release` installations may be deleted after `30` days from `last_seen_at`.
- Preflight-only `none` rows created by challenge issuance but never verified may be deleted after `1` day from `max(last_seen_at, challenge_expires_at)`, so cleanup does not invalidate an in-flight challenge before its TTL boundary.
- Broker request handling opportunistically applies that preflight cleanup when the installation identity is touched again, so stale unauthenticated rows can age out without broadening retention into a separate store redesign.
- Terminal `expired` or `revoked` installations may be deleted after `90` days from `max(last_seen_at, expires_at)`.
- Retention cleanup deletes from `installations`; the entitlement row is removed by `ON DELETE CASCADE`.
- Because `hardware_hash` remains `NULL` until verify succeeds, preflight-row cleanup does not discard duplicate-detection fingerprint state.
- `fingerprint_salt` remains one server-managed global salt shared across clients for duplicate detection.
- Rotation keeps one current salt and one previous salt version. Duplicate matching only uses `hardware_hash` values tagged with the current version. In-flight challenges may complete on the previous version until their existing `challenge_expires_at`, after which stale hashes are refreshed in place on successful verify or cleared when the broker reissues a challenge for `none` / `pending_release` state.

## Public error normalization and abuse outcomes

- Public error `code` values are bounded to: `invalid_request`, `rate_limited`, `challenge_expired`, `challenge_invalid`, `issuance_suspended`, `trial_unavailable`, `trial_not_eligible`, `internal_error`.
- Public error `class` values are bounded to: `retryable`, `terminal`, `security_fail`.
- Current subcodes include endpoint rate-limit dimensions (`ip_rate_limited`, `installation_rate_limited`), challenge/release validation details (`release_token_expired`, `signature_mismatch`, `timestamp_skew`), duplicate suppression (`hardware_duplicate`), and issuance suspension (`global_cap_reached`).
- Abuse-hook rows may store operator metadata, but hook-specific labels do not expand the public subcode vocabulary; public hook responses normalize to bounded existing subcodes or `null`.
- Error envelopes also carry `retry_after_ms` plus companion `managed_state` / `current_entitlement` fields so clients can distinguish retryable suspension from lifecycle-managed states such as `expired` and `revoked`.
