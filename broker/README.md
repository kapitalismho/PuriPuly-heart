# Broker service foundation

This directory establishes the managed-trial broker as a separate deployable service in the monorepo.

## Explicit rollout boundary

- Runtime stack: TypeScript + Hono on Cloudflare Workers with native D1 and Worker secrets.
- Hosting scope: single-region rollout assumption for the initial Worker deployment, with D1 `location_hint` set to `apac`.
- Managed free-trial path: `OpenRouter` + curated allowlist `google/gemma-4-26b-a4b-it`, `google/gemma-4-31b-it`, `deepseek/deepseek-v4-flash-0731`, and `deepseek/deepseek-v4-flash`.
- Inference boundary: the app talks to OpenRouter directly; the broker remains a trial and credential broker.
- Out of scope in this foundation: translation proxying, multi-region deployment, KV, R2, and admin dashboard work.

## Deploy note

`broker/wrangler.jsonc` intentionally uses a non-secret placeholder `database_id`. A real Cloudflare D1 identifier must be supplied in deployment-specific configuration before the service is deployed.

Use `pnpm --filter @puripuly-heart/broker run verify:config` to exercise the pinned Wrangler CLI against `broker/wrangler.jsonc` without requiring cloud credentials.

## Direct deploy automation

- `broker/scripts/render-production-wrangler-config.mjs` renders a temporary deploy-time Wrangler config from `broker/wrangler.jsonc`, injects the production D1 `database_id`, and fails if the checked-in worker name stops being the canonical `puripuly-heart-broker`.
- `broker/deploy/fingerprint-bootstrap.template.sql` plus `broker/scripts/render-fingerprint-bootstrap-sql.mjs` render guarded bootstrap SQL for `wrangler d1 execute --file ... --yes`. The rendered SQL only replaces the migration placeholder and fails before mutating `broker_config` if the placeholder is already gone.
- `.github/workflows/deploy-broker-direct.yml` is the manual `workflow_dispatch` path for the first canonical deploy. It exports the remote production D1 database to a restricted seven-day workflow artifact before applying migrations, bootstraps the fingerprint salt, reconciles the production OpenRouter guardrail through `PATCH /api/v1/guardrails/{id}`, syncs the OpenRouter, Discord, and QQ worker secrets needed for managed child-key issuance and QQ production issuance, deploys the canonical worker, verifies health, removes transitional runtime fields with `broker/deploy/finalize-daily-summary-v2.sql` and `broker/deploy/finalize-app-active-day.sql`, and runs `broker/tests/deploy-smoke/canonical-production.spec.ts` against the canonical `workers.dev` URL. Because migrations run before Worker deployment, `0016_make_referrals_source_aware.sql` retains the previous Worker's Discord referral columns and synchronizes Discord inserts into both legacy and source-aware identities; QQ rows keep those compatibility columns `NULL`.
- `OPENROUTER_MANAGED_API_KEY_PRODUCTION` remains transitional runtime compatibility only; `OPENROUTER_MANAGEMENT_API_KEY_PRODUCTION` drives managed child-key creation / cleanup, `OPENROUTER_MANAGED_GUARDRAIL_ID_PRODUCTION` assigns the production guardrail to each issued key, and `OPENROUTER_MANAGED_USER_HMAC_SECRET_PRODUCTION` is copied into the runtime secret `OPENROUTER_MANAGED_USER_HMAC_SECRET` so the worker can derive a deterministic versioned managed OpenRouter user id per installation or QQ subject.
- `QQ_AUTH_HMAC_PSK_PRODUCTION` is copied into the runtime secret `QQ_AUTH_HMAC_PSK` for `POST /v1/auth/qq/assert`. The endpoint is production issuance-capable when runtime issuance configuration is present (`QQ_AUTH_HMAC_PSK`, `OPENROUTER_MANAGEMENT_API_KEY`, and `OPENROUTER_MANAGED_GUARDRAIL_ID` are all non-blank). The issuance-disabled verification-only behavior preserves `verified` / `already_verified` compatibility without touching `qq_managed_entitlements`; when issuance is enabled, OpenRouter, guardrail, cleanup, or D1 failures return a bounded retryable/internal error envelope instead of falling back to verification-only success. The PSK value, raw QQ identity, raw credential, and raw key-bearing payloads must stay out of source, docs, logs, and test output.
- `TELEMETRY_SUBJECT_HMAC_SECRET_PRODUCTION` is copied into the runtime secret `TELEMETRY_SUBJECT_HMAC_SECRET` for `POST /v1/telemetry/app-active-day`. Production migration rollout must take a D1 backup before applying `0015_add_app_active_days.sql`; that forward-only migration creates the isolated minimal app active-day table while preserving the previous Worker's required abuse-control shape and the previous translation telemetry tables and rows. After the new Worker passes health verification, `broker/deploy/finalize-app-active-day.sql` removes the obsolete telemetry endpoint rate-limit setting.
- `DISCORD_CLIENT_ID_PRODUCTION`, `DISCORD_CLIENT_SECRET_PRODUCTION`, `DISCORD_REDIRECT_URI_ALLOWLIST_PRODUCTION`, and `DISCORD_USER_REF_SECRET_PRODUCTION` are copied into the runtime secrets `DISCORD_CLIENT_ID`, `DISCORD_CLIENT_SECRET`, `DISCORD_REDIRECT_URI_ALLOWLIST`, and `DISCORD_USER_REF_SECRET` for Discord OAuth onboarding.
- `DISCORD_OPERATIONS_WEBHOOK_URL_PRODUCTION` is copied into the runtime secrets `DISCORD_IMMEDIATE_ALERT_WEBHOOK_URL` and `DISCORD_DAILY_REPORT_WEBHOOK_URL` so the broker can send real-time alerts and the `puripuly_daily_summary.v2` report. The minute-resolution cron consults `abuse_controls.dailyReport` and the v2 delivery ledger, then sends at 00:05 UTC for the last completed UTC date.
- The deploy reconcile step sets `allowed_models` to `google/gemma-4-26b-a4b-it`, `google/gemma-4-31b-it`, `deepseek/deepseek-v4-flash-0731`, and `deepseek/deepseek-v4-flash`, clears provider restrictions inside the guardrail (`allowed_providers` / `ignored_providers`), and sets `enforce_zdr = false` before smoke.
- The deploy smoke verifies a synthetic non-PII QQ Managed assertion through `POST /v1/auth/qq/assert`, expects `status: "issued"` with a one-time `openrouter_api_key`, verifies duplicate/lifetime guardrail behavior without key recovery, verifies issued child-key metadata through `https://openrouter.ai/api/v1/key`, proves positive routing through `google/gemma-4-31b-it`, `deepseek/deepseek-v4-flash-0731`, and `deepseek/deepseek-v4-flash`, and still probes `https://openrouter.ai/api/v1/chat/completions` with `BROKER_DEPLOY_SMOKE_DISALLOWED_MODEL_PRODUCTION` to confirm guardrail enforcement.
- Config verification is split by surface: `pnpm --filter @puripuly-heart/broker run verify:config` checks the checked-in Worker binding contract, while the direct-deploy guard step fails before migrations if the production secrets `QQ_AUTH_HMAC_PSK_PRODUCTION`, `TELEMETRY_SUBJECT_HMAC_SECRET_PRODUCTION`, `OPENROUTER_MANAGEMENT_API_KEY_PRODUCTION`, or `OPENROUTER_MANAGED_GUARDRAIL_ID_PRODUCTION` are missing or blank. Neither path prints secret values.
- Account-level OpenRouter privacy / provider settings remain outside repo control and may still narrow effective routing even after the guardrail reconcile; the production smoke is the proof point for the resulting path.
- The workflow expects CI-managed secrets / vars in the `production` GitHub Environment: `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID`, `BROKER_D1_DATABASE_ID_PRODUCTION`, `OPENROUTER_MANAGED_API_KEY_PRODUCTION`, `OPENROUTER_MANAGEMENT_API_KEY_PRODUCTION`, `OPENROUTER_MANAGED_GUARDRAIL_ID_PRODUCTION`, `OPENROUTER_MANAGED_USER_HMAC_SECRET_PRODUCTION`, `QQ_AUTH_HMAC_PSK_PRODUCTION`, `TELEMETRY_SUBJECT_HMAC_SECRET_PRODUCTION`, `DISCORD_CLIENT_ID_PRODUCTION`, `DISCORD_CLIENT_SECRET_PRODUCTION`, `DISCORD_REDIRECT_URI_ALLOWLIST_PRODUCTION`, `DISCORD_USER_REF_SECRET_PRODUCTION`, `DISCORD_OPERATIONS_WEBHOOK_URL_PRODUCTION`, `BROKER_CANONICAL_WORKERS_DEV_URL`, and `BROKER_DEPLOY_SMOKE_DISALLOWED_MODEL_PRODUCTION`.
- App / public traffic must stay disconnected from the broker until the direct deploy smoke run passes and is explicitly reviewed.

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
  - required request fields are `qq_identity`, `credential`, and `asserted_at`; optional fields are `delivery_ack_supported`, normalized `referral_id`, and bounded `installation_id`; no device key or hardware hash is used
  - `credential` remains `HMAC-SHA256-HEX(QQ_AUTH_HMAC_PSK, qq_identity)`; referral and installation metadata are not added to the HMAC payload, and `asserted_at` is validated as bounded timestamp text but is not part of that payload
  - `QQ_AUTH_HMAC_PSK` is mandatory for all QQ assertion handling; `OPENROUTER_MANAGEMENT_API_KEY` and `OPENROUTER_MANAGED_GUARDRAIL_ID` are the runtime gate for production issuance
  - with the OpenRouter issuance gate disabled, valid assertions preserve verification-only compatibility and return `verified` / `already_verified` without creating or mutating `qq_managed_entitlements`
  - with the gate enabled, an eligible first assertion returns `issued`, `qq_subject_ref`, one-time top-level `openrouter_api_key`, `managed_credential_ref`, `expires_at`, optional `openrouter_user_id`, and delivery-ACK metadata when the client advertises ACK support; the Broker never stores the raw key
  - duplicate active, cleanup-required, or revoked QQ entitlements return `qq_lifetime_used`; concurrent current issuance returns `qq_already_issuing`; invalid credentials return `qq_credential_invalid`
  - `qq_talk_together_pass.enabled` controls QQ owned Pass/status creation and referral input; `rewards_enabled` independently controls reward reservation while leaving base QQ issuance available; both default to `false`
  - a QQ referral reservation is best-effort and cannot block the base `0.07` USD key; a counted invitee remains `reserved` until durable delivery ACK, then the broker raises the invitee key to at least `0.09` USD, persists the verified limit, credits the ledger, and applies the referrer reward
  - counted QQ rewards enforce the configured UTC-day warning/default cap of `30`/`50`; reserved and credited rows both count toward global and per-Pass limits, and one source-aware Pass rewards at most three invitees across Discord and QQ
  - QQ lifetime, monitoring, cleanup, Pass ownership, and reward accounting use derived `qq_subject_ref` / `issue_ref` metadata and must not create fake installation rows or persist/log raw QQ identity or credential
- `POST /v1/auth/qq/status`
  - request: `qq_identity`, `credential`, and optional bounded `installation_id`; the credential uses the unchanged QQ HMAC contract
  - requires an active delivered, unexpired QQ entitlement and returns `status: "active"`; inactive lifecycle returns `qq_entitlement_inactive`
  - when `qq_talk_together_pass.enabled` is true, status lazily creates or resolves the QQ subject's owned global Referral ID and returns `referral_id` plus `talk_together_pass`; when disabled, it does not create an ID
  - rate limited per IP by `qqAuthStatusIp` at `30` requests / `15` minutes
  - the desktop stores status authentication only in the local secret store and binds it to the active managed credential reference; raw QQ identity and credential do not enter D1 or logs
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
- `0005_add_referral_persistence_foundation.sql` adds nullable OAuth session `referral_id` storage plus the referral code and referral reward ledger foundation.
- `0008_add_qq_auth_assertions.sql` adds the `qq_auth_assertions` evidence table and inserts the `qqAuthAssertIp` abuse-control default without replacing operator-tuned `abuse_controls` values.
- `0009_add_qq_managed_entitlements.sql` adds the `qq_managed_entitlements` lifecycle table for QQ production issuance without rewriting existing assertion evidence.
- `0010_source_aware_issue_success_events.sql` rebuilds `broker_issue_success_events` so successful issue monitoring is source-aware: Discord rows keep installation identity, while QQ rows use `issue_source = 'qq'`, nullable `installation_id`, and `subject_ref = qq_subject_ref` instead of fake installation rows.
- `0011_add_telemetry_active_days.sql` creates the isolated `telemetry_active_days` table and additively inserts the telemetry endpoint IP rate-limit default into `abuse_controls`; production rollout requires a pre-migration D1 backup/export before this forward migration is applied.
- `0012_add_managed_key_delivery_ack.sql` adds the shared Discord/QQ delivery acknowledgement ledger and delivery-pending lifecycle states.
- `0013_add_telemetry_subjects_and_daily_summary_v2.sql` creates and backfills `telemetry_subjects`, keeps it synchronized for the previous Worker during rollout, creates the v2 delivery ledger, preserves existing active-day rows, sets the daily report schedule to 00:05 UTC, and raises issue-event retention to the report-safe two-day minimum without replacing unrelated operator-tuned controls. It intentionally retains `includeZeroActivity` while the previous Worker may still run; the deploy workflow removes that dead field only after the new Worker passes its health check.
- `0014_simplify_abuse_incidents.sql` additively derives the `warning` and `brake` thresholds, the ordered warning observation state, and the request-event safety margin from existing persisted controls. It also adds a QQ child-key-creation-start marker so ambiguous post-provider failures cannot be stale-reclaimed into a second key. Legacy alert/ASN JSON fields remain during the migration-before-deploy compatibility window; unused physical columns and indexes require a separate forward migration after stabilization.
- `0015_add_app_active_days.sql` creates `app_active_days` with only HMAC-derived subject and UTC-date columns while retaining the previous Worker's required abuse-control shape and preserving the legacy translation telemetry tables and rows without mixing them into app usage metrics. The deploy workflow removes `telemetryTranslationSuccessDayIp` with `broker/deploy/finalize-app-active-day.sql` only after the new Worker passes its health check.
- `0016_make_referrals_source_aware.sql` migrates Referral IDs and reward rows to the shared `discord`/`qq` subject namespace, adds the disabled-by-default `qq_talk_together_pass` config and QQ status rate limit, and preserves migration-before-deploy compatibility. Existing and previous-Worker Discord writes retain `owner_discord_user_ref`, `referrer_discord_user_ref`, and `referred_discord_user_ref`; insert triggers synchronize those values with the source-aware columns. New source-aware Discord writes synchronize back to the compatibility columns, while QQ rows leave them `NULL` so they cannot be mistaken for Discord identities. Removing these compatibility columns requires a separate post-stabilization migration.

- `broker_config`
  - columns: `key`, `value`, `updated_at`
  - bootstrap rows: `fingerprint_salt`, `abuse_controls`, `abuse_runtime_state`, `qq_talk_together_pass`
  - `qq_talk_together_pass` defaults to `{ enabled: false, rewards_enabled: false, daily_warning_count: 30, daily_max_count: 50 }`; malformed values fall back to those disabled defaults
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
    - `POST /v1/auth/qq/status`: per IP via `qqAuthStatusIp`, `30` requests / `15` minutes
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
  - legacy translation-success telemetry tables retained unchanged for forward-migration safety
  - no new app active-day writes or app DAU/WAU/MAU reads use these tables, so old-version translation activity cannot enter the new app usage metrics
  - destructive deletion of the legacy rows remains a separate operator decision
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
  - columns include session/device/PKCE fields, Discord eligibility fields, lifecycle timestamps, and nullable normalized `referral_id`
  - `referral_id` accepts only six uppercase approved-alphabet characters (`0`, `O`, `1`, `I`, and `L` excluded) or `NULL`
  - indexed by installation/status/creation time, expiry, and non-`NULL` `referral_id`
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
  - a referred delivery remains at base budget `0.07` through `delivery_pending`; successful ACK settlement raises and verifies at least `0.09` before the entitlement budget and reward ledger are credited
  - existing `qq_auth_assertions` rows without a `qq_managed_entitlements` row remain eligible for their first production issuance
  - stores derived and operational metadata only; raw QQ identities, raw credentials, and raw OpenRouter API keys do not belong in this table
- `referral_codes`
  - stable global Referral ID rows keyed by `referral_id`, owned by `(owner_source, owner_subject_ref)` for `discord` or `qq`
  - source-aware columns are `referral_id`, `owner_source`, `owner_subject_ref`, nullable `owner_installation_id`, status/audit timestamps, and disable metadata
  - `owner_discord_user_ref` is a migration-before-deploy compatibility column for Discord rows; insert synchronization keeps it aligned with `owner_subject_ref`, and it stays `NULL` for QQ rows
  - Referral IDs are exactly six characters from the approved uppercase alphabet excluding `0`, `O`, `1`, `I`, and `L`; statuses are `active` or `disabled`
  - `(owner_source, owner_subject_ref)` is unique across each source and the `referral_id` primary key provides the shared cross-source namespace; raw Discord/QQ identities do not belong in this table
  - no `ON DELETE CASCADE` dependency on `installations`, preserving code history when installation rows age out
- `referral_rewards`
  - append-only source-aware referral attempt/reward ledger rows keyed by `id`
  - source-aware identity is stored as referrer/referred source and derived subject references, with nullable installation evidence, Discord-only referred hardware evidence, bonus statuses, bounded reason codes, managed credential refs, and lifecycle timestamps
  - `referrer_discord_user_ref` and `referred_discord_user_ref` remain synchronized compatibility columns for Discord rows during rollout and stay `NULL` for QQ identities
  - `attempt_ip_digest`, `attempt_ip_key_version`, and `attempt_ip_epoch` carry the server-secret keyed network identity digest; no raw IP or unkeyed IP hash is persisted
  - `operation_id` is unique when bound: one referral reservation per managed operation, reused across issuance retries without consuming additional velocity or caps
  - Referral IDs use the same approved six-character constraint; referred-side statuses are `reserved`, `credited`, `skipped`, and `failed`; referrer-side statuses are `pending`, `applying`, `credited`, `skipped`, and `failed`
  - reserved and credited rows both count toward the three-invitee Pass cap, one lifetime reward per source-aware referred subject, installation duplicate prevention, and the QQ UTC-day cap
  - source-aware and compatibility indexes support both Workers during migration-before-deploy; ledger rows do not cascade-delete with `installations`, preserving cap/accounting history when installation rows age out

## Managed operations and durable issuance

- Managed key issuance is one durable logical operation (`managed_operations`) that may span several provider-key attempts (`managed_operation_attempts`); raw OpenRouter API keys stay in transient Broker memory only and are never persisted in any form.
- Clients generate `operation_id` (`ph-mop-v1_` plus base64url of 24 random bytes) and `resume_token` (base64url of 32 random bytes) before issuance; issue requests carry both. Only `ph-mop-resume-v1_` plus the SHA-256 token hash is persisted, compared in constant time, and bound to source subject, installation, and device identity.
- Operation states are `AUTHENTICATED`, `ISSUE_READY`, `CREATING`, `CREATE_UNKNOWN`, `RECONCILING`, `CLEANUP_REQUIRED`, `CLEAN`, `RETRY_READY`, `DELIVERY_PENDING`, `ACTIVE`, and `FAILED`; recovery follows unknown, reconcile, cleanup-required, clean, retry-ready ordering, and fresh issuance is blocked until cleanup is verified against the provider by deterministic non-secret attempt key names.
- Public recovery routes are `POST /v1/providers/openrouter/managed-operation/status` and `POST /v1/providers/openrouter/managed-operation/resume` with body `{operation_id, resume_token, installation_id}`; known nonterminal re-POST is idempotent and cannot create a provider key unless the operation is retry-ready.
- Recovery authorization lasts exactly 60 minutes from operation creation, is installation/device bound, and is non-renewable; expiry fails the operation with `authorization_expired`, and only then may an unresolved referral reservation terminally fail.
- Deliveries link to their operation and attempt; delivery ACK TTL is exactly 15 minutes (valid at the exact expiry instant, expired one millisecond later); stale delivery attempts are reconciled and cleaned server-side even if the client never returns.
- Discord referred keys are created at the base `0.07` budget; after confirmed delivery ACK, durable source-agnostic settlement (`managed_referral_settlement_jobs`) raises and provider-verifies at least `0.09` for the invitee before local credit and settles the referrer `0.02` reward with the same leases, fencing, one-minute to one-hour retry, read-back verification, convergence, and repair semantics for Discord and QQ.
- Operational states correlate across operation, attempt, delivery, credential reference, referral, and settlement identifiers without logging raw keys, ACK or resume tokens, OAuth secrets, raw IPs, or keyed digests; `managed_credential_ref` is nonrecoverable operational metadata and may appear in diagnostics.

## Network identity HMAC and staged migration

- Persisted IP-derived identifiers use HMAC-SHA-256 with explicit key version and bounded UTC-day epoch, domain-separated per scope. Worker secrets required: `NETWORK_IDENTITY_HMAC_SECRET` (current) and optional `NETWORK_IDENTITY_HMAC_SECRET_PREVIOUS` (rotation overlap). Non-secret version vars required: `NETWORK_IDENTITY_HMAC_KEY_VERSION` (current, explicit positive integer, no silent default) and optional `NETWORK_IDENTITY_HMAC_KEY_VERSION_PREVIOUS`, which must be set together with the previous secret, be a positive integer, and differ from the current version. The render script takes `--network-identity-hmac-key-version` (required) and `--network-identity-hmac-key-version-previous` (optional) and injects these vars into the production wrangler config. The deploy workflow must copy `NETWORK_IDENTITY_HMAC_SECRET_PRODUCTION` (and, during rotation only, `NETWORK_IDENTITY_HMAC_SECRET_PREVIOUS_PRODUCTION`) into those runtime secrets and must never print their values; the guard step must fail before migrations when the current secret is missing or blank or the version pair is unpaired, malformed, or equal.
- Generate the current secret from at least 32 random bytes (base64url or hex) and store it only in the production GitHub Environment alongside the existing broker secrets. Rotation: set the previous secret to the outgoing value, set the previous version var to the outgoing version, set the current version var to the new version, deploy, wait out the longest abuse-policy window (referrer velocity default 1440 minutes plus any operator velocity-hook windows), then replace current and clear previous. Recovery/status bearers stay time-bounded past activation: an expired token cannot query even an ACTIVE operation, and settlement jobs continue without bearer authorization. Current-secret digests carry the current version and previous-secret digests carry the explicit previous version, so rotation overlap and previous removal stay auditable from persisted rows and never silently undercount.
- Reads dual-compare current plus previous digests only inside still-active policy windows, enumerating every UTC-day epoch the window intersects; rotation never silently weakens rate limits, and cross-period correlation is bounded to the overlapping epoch.
- IP spellings are canonicalized before hashing (IPv4 leading zeros stripped, IPv6 lowercased with leading zeros stripped, `::ffff:a.b.c.d` mapped to the IPv4), so equivalent textual forms share one digest; undecodable values yield no digest and never match. Operator `ip` hooks (`broker_velocity_cap_hooks`, `broker_abuse_subject_hooks`) must be registered with stable keyed digests, not raw IPs. The scheduled worker converts parseable pre-cutover raw-IP hook values to stable digests and refuses to finalize the migration while any convertible ones remain; unparseable values never match, keep the `0021` gate aborting until removed manually, and must be deleted by the operator.
- Schema transition is explicitly staged: `0020_network_identity_hmac.sql` adds keyed columns and records `dual_write` in `broker_config.network_identity_migration` whenever decision-relevant legacy rows exist; the scheduled worker backfills raw request-event rows with the Worker secret while dual-reading both representations so no active window is lost; once no un-backfilled in-window rows remain it purges legacy values and flips to `keyed_only`; `0021_network_identity_purge.sql` aborts loudly unless that phase is reached with no legacy rows inside the actual maximum policy window (endpoint configs, referral controls, and active velocity hooks) and no unconverted raw-IP hooks remaining, and then drops the legacy columns.

## Retention and salt rotation

- Inactive `pending_release` installations may be deleted after `30` days from `last_seen_at`.
- Preflight-only `none` rows created by challenge issuance but never verified may be deleted after `1` day from `max(last_seen_at, challenge_expires_at)`, so cleanup does not invalidate an in-flight challenge before its TTL boundary.
- Broker request handling opportunistically applies that preflight cleanup when the installation identity is touched again, so stale unauthenticated rows can age out without broadening retention into a separate store redesign.
- Terminal `expired` or `revoked` installations may be deleted after `90` days from `max(last_seen_at, expires_at)`.
- Retention cleanup deletes from `installations`; the entitlement row is removed by `ON DELETE CASCADE`.
- Referral code and reward ledger rows are intentionally not cascade-deleted by installation retention cleanup, so cap/accounting history remains stable.
- Because `hardware_hash` remains `NULL` until verify succeeds, preflight-row cleanup does not discard duplicate-detection fingerprint state.
- `fingerprint_salt` remains one server-managed global salt shared across clients for duplicate detection.
- Rotation keeps one current salt and one previous salt version. Duplicate matching only uses `hardware_hash` values tagged with the current version. In-flight challenges may complete on the previous version until their existing `challenge_expires_at`, after which stale hashes are refreshed in place on successful verify or cleared when the broker reissues a challenge for `none` / `pending_release` state.

## Public error normalization and abuse outcomes

- Public error `code` values are bounded to: `invalid_request`, `rate_limited`, `challenge_expired`, `challenge_invalid`, `issuance_suspended`, `trial_unavailable`, `trial_not_eligible`, `internal_error`.
- Public error `class` values are bounded to: `retryable`, `terminal`, `security_fail`.
- Current subcodes include endpoint rate-limit dimensions (`ip_rate_limited`, `installation_rate_limited`), challenge/release validation details (`release_token_expired`, `signature_mismatch`, `timestamp_skew`), duplicate suppression (`hardware_duplicate`), and issuance suspension (`global_cap_reached`).
- Abuse-hook rows may store operator metadata, but hook-specific labels do not expand the public subcode vocabulary; public hook responses normalize to bounded existing subcodes or `null`.
- Error envelopes also carry `retry_after_ms` plus companion `managed_state` / `current_entitlement` fields so clients can distinguish retryable suspension from lifecycle-managed states such as `expired` and `revoked`.
