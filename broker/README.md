# Broker service foundation

This directory establishes the managed-trial broker as a separate deployable service in the monorepo.

## Explicit rollout boundary

- Runtime stack: TypeScript + Hono on Cloudflare Workers with native D1 and Worker secrets.
- Hosting scope: single-region rollout assumption for the initial Worker deployment, with D1 `location_hint` set to `apac`.
- Managed free-trial path: `OpenRouter` + curated allowlist `google/gemma-4-26b-a4b-it`, `qwen/qwen3.5-flash-02-23`, `deepseek/deepseek-v4-flash`, and `google/gemini-2.5-flash-lite`.
- Inference boundary: the app talks to OpenRouter directly; the broker remains a trial and credential broker.
- Out of scope in this foundation: translation proxying, multi-region deployment, KV, R2, and admin dashboard work.

## Deploy note

`broker/wrangler.jsonc` intentionally uses a non-secret placeholder `database_id`. A real Cloudflare D1 identifier must be supplied in deployment-specific configuration before the service is deployed.

Use `pnpm --filter @puripuly-heart/broker run verify:config` to exercise the pinned Wrangler CLI against `broker/wrangler.jsonc` without requiring cloud credentials.

## Direct deploy automation

- `broker/scripts/render-production-wrangler-config.mjs` renders a temporary deploy-time Wrangler config from `broker/wrangler.jsonc`, injects the production D1 `database_id`, and fails if the checked-in worker name stops being the canonical `puripuly-heart-broker`.
- `broker/deploy/fingerprint-bootstrap.template.sql` plus `broker/scripts/render-fingerprint-bootstrap-sql.mjs` render guarded bootstrap SQL for `wrangler d1 execute --file ... --yes`. The rendered SQL only replaces the migration placeholder and fails before mutating `broker_config` if the placeholder is already gone.
- `.github/workflows/deploy-broker-direct.yml` is the manual `workflow_dispatch` path for the first canonical deploy. It applies remote D1 migrations, bootstraps the fingerprint salt, reconciles the production OpenRouter guardrail through `PATCH /api/v1/guardrails/{id}`, syncs the OpenRouter, Discord, and QQ worker secrets needed for managed child-key issuance and QQ production issuance, deploys the canonical worker, and runs `broker/tests/deploy-smoke/canonical-production.spec.ts` against the canonical `workers.dev` URL.
- `OPENROUTER_MANAGED_API_KEY_PRODUCTION` remains transitional runtime compatibility only; `OPENROUTER_MANAGEMENT_API_KEY_PRODUCTION` drives managed child-key creation / cleanup, `OPENROUTER_MANAGED_GUARDRAIL_ID_PRODUCTION` assigns the production guardrail to each issued key, and `OPENROUTER_MANAGED_USER_HMAC_SECRET_PRODUCTION` is copied into the runtime secret `OPENROUTER_MANAGED_USER_HMAC_SECRET` so the worker can derive a deterministic versioned managed OpenRouter user id per installation or QQ subject.
- `QQ_AUTH_HMAC_PSK_PRODUCTION` is copied into the runtime secret `QQ_AUTH_HMAC_PSK` for `POST /v1/auth/qq/assert`. The endpoint is production issuance-capable when runtime issuance configuration is present (`QQ_AUTH_HMAC_PSK`, `OPENROUTER_MANAGEMENT_API_KEY`, and `OPENROUTER_MANAGED_GUARDRAIL_ID` are all non-blank). The issuance-disabled verification-only behavior preserves `verified` / `already_verified` compatibility without touching `qq_managed_entitlements`; when issuance is enabled, OpenRouter, guardrail, cleanup, or D1 failures return a bounded retryable/internal error envelope instead of falling back to verification-only success. The PSK value, raw QQ identity, raw credential, and raw key-bearing payloads must stay out of source, docs, logs, and test output.
- `TELEMETRY_SUBJECT_HMAC_SECRET_PRODUCTION` is copied into the runtime secret `TELEMETRY_SUBJECT_HMAC_SECRET` for `POST /v1/telemetry/translation-success-day`. Production migration rollout must take a D1 backup before applying `0011_add_telemetry_active_days.sql` (for example, Cloudflare D1 export/backup of the target database), then apply the forward-only migration that creates the isolated telemetry table and additively patches the telemetry IP rate-limit default without replacing operator-tuned abuse controls.
- Broker responses expose the AGPL-3.0-or-later source offer through `Link: <https://github.com/kapitalismho/PuriPuly-heart>; rel="source"`, and `GET /source` returns the corresponding source URL for network users.
- `DISCORD_CLIENT_ID_PRODUCTION`, `DISCORD_CLIENT_SECRET_PRODUCTION`, `DISCORD_REDIRECT_URI_ALLOWLIST_PRODUCTION`, and `DISCORD_USER_REF_SECRET_PRODUCTION` are copied into the runtime secrets `DISCORD_CLIENT_ID`, `DISCORD_CLIENT_SECRET`, `DISCORD_REDIRECT_URI_ALLOWLIST`, and `DISCORD_USER_REF_SECRET` for Discord OAuth onboarding.
- `DISCORD_OPERATIONS_WEBHOOK_URL_PRODUCTION` is copied into the runtime secrets `DISCORD_IMMEDIATE_ALERT_WEBHOOK_URL` and `DISCORD_DAILY_REPORT_WEBHOOK_URL` so the broker can send real-time alerts, while the minute-resolution cron trigger consults `abuse_controls.dailyReport` plus persisted `abuse_runtime_state` to emit the daily Discord heartbeat only once per UTC day.
- The deploy reconcile step sets `allowed_models` to `google/gemma-4-26b-a4b-it`, `qwen/qwen3.5-flash-02-23`, `deepseek/deepseek-v4-flash`, and `google/gemini-2.5-flash-lite`, clears provider restrictions inside the guardrail (`allowed_providers` / `ignored_providers`), and sets `enforce_zdr = false` before smoke.
- The deploy smoke verifies a synthetic non-PII QQ Managed assertion through `POST /v1/auth/qq/assert`, expects `status: "issued"` with a one-time `openrouter_api_key`, verifies duplicate/lifetime guardrail behavior without key recovery, verifies issued child-key metadata through `https://openrouter.ai/api/v1/key`, proves positive routing through `qwen/qwen3.5-flash-02-23`, `deepseek/deepseek-v4-flash`, and `google/gemini-2.5-flash-lite`, and still probes `https://openrouter.ai/api/v1/chat/completions` with `BROKER_DEPLOY_SMOKE_DISALLOWED_MODEL_PRODUCTION` to confirm guardrail enforcement.
- Config verification is split by surface: `pnpm --filter @puripuly-heart/broker run verify:config` checks the checked-in Worker binding contract, while the direct-deploy guard step fails before migrations if the production secrets `QQ_AUTH_HMAC_PSK_PRODUCTION`, `OPENROUTER_MANAGEMENT_API_KEY_PRODUCTION`, or `OPENROUTER_MANAGED_GUARDRAIL_ID_PRODUCTION` are missing or blank. Neither path prints secret values.
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
  - request body remains `qq_identity`, `credential`, and `asserted_at`; no installation ID, device key, hardware hash, or app-side endpoint sequence is required for QQ production issuance
  - `credential` remains `HMAC-SHA256-HEX(QQ_AUTH_HMAC_PSK, qq_identity)`; `asserted_at` is validated as bounded timestamp text but is not part of the HMAC payload
  - `QQ_AUTH_HMAC_PSK` is mandatory for all QQ assertion handling; `OPENROUTER_MANAGEMENT_API_KEY` and `OPENROUTER_MANAGED_GUARDRAIL_ID` are the runtime gate for production issuance
  - with the OpenRouter issuance gate disabled, valid assertions preserve verification-only compatibility and return `verified` / `already_verified` without creating or mutating `qq_managed_entitlements`
  - with the gate enabled, an eligible first assertion returns `issued`, `qq_subject_ref`, one-time top-level `openrouter_api_key`, `managed_credential_ref`, `expires_at`, and optional `openrouter_user_id`; the Broker never stores the raw key
  - duplicate active, cleanup-required, or revoked QQ entitlements return `qq_lifetime_used`; concurrent current issuance returns `qq_already_issuing`; invalid credentials return `qq_credential_invalid`
  - QQ uses the base Managed trial budget, expiry, allowed-model policy, OpenRouter child-key creation, and configured guardrail; it does not run Discord referral reservation/rewards, owned Referral ID, Talk Together Pass, or referral bonus budget paths
  - QQ lifetime, monitoring, and cleanup use `qq_subject_ref` / `issue_ref` metadata and must not create fake installation rows or pass raw QQ identity to OpenRouter child-key names
- `POST /v1/telemetry/translation-success-day`
  - request: `signal: "translation_success_day"`, `telemetry_identifier`, `active_date_utc` as `YYYY-MM-DD`
  - accepts only the single active-day telemetry shape; malformed JSON, invalid identifiers/dates, unsupported signals, and additional telemetry fields return the existing public `invalid_request` envelope
  - derives `subject_ref = ph-telemetry-subject-v1_ + HMAC-SHA256-HEX(TELEMETRY_SUBJECT_HMAC_SECRET, telemetry_identifier)` and persists only `subject_ref`, UTC active date, and receipt timestamps
  - duplicate same-subject same-date payloads update `last_received_at` on the same row and cannot inflate active-day counts
  - per-IP rate limiting uses `abuse_controls.telemetryTranslationSuccessDayIp`; the endpoint does not collect installation, Discord, QQ, hardware, provider, model, language, output-route, or content identity

## Persistence model

`broker/src/persistence.ts` and `broker/migrations/*.sql` define the D1-backed state contract and its upgrade path.

- `0001_harden_installation_public_inputs.sql` rebuilds `installations` (and the dependent `openrouter_entitlements` table) under deferred foreign-key checks so already-initialized clean schemas pick up the hardened public-input constraints.
- `0002_add_entitlement_verified_hardware_snapshot.sql` adds `verified_hardware_hash` and `verified_hardware_hash_salt_version` to `openrouter_entitlements` for the verified release-session hardware snapshot consumed by `/v1/providers/openrouter/issue`.
- `0003_add_abuse_runtime_state_and_issue_success_events.sql` adds the persisted abuse runtime-state row plus append-only issue-success and runtime-audit tables used by alerting, brake state, daily heartbeat delivery, and retention.
- `0004_add_discord_oauth_managed_issue.sql` adds Discord OAuth session and identity storage plus Discord-managed issue columns on `openrouter_entitlements`.
- `0005_add_referral_persistence_foundation.sql` adds nullable OAuth session `referral_id` storage plus the referral code and referral reward ledger foundation.
- `0008_add_qq_auth_assertions.sql` adds the `qq_auth_assertions` evidence table and inserts the `qqAuthAssertIp` abuse-control default without replacing operator-tuned `abuse_controls` values.
- `0009_add_qq_managed_entitlements.sql` adds the `qq_managed_entitlements` lifecycle table for QQ production issuance without rewriting existing assertion evidence.
- `0010_source_aware_issue_success_events.sql` rebuilds `broker_issue_success_events` so successful issue monitoring is source-aware: Discord rows keep installation identity, while QQ rows use `issue_source = 'qq'`, nullable `installation_id`, and `subject_ref = qq_subject_ref` instead of fake installation rows.
- `0011_add_telemetry_active_days.sql` creates the isolated `telemetry_active_days` table and additively inserts the telemetry endpoint IP rate-limit default into `abuse_controls`; production rollout requires a pre-migration D1 backup/export before this forward migration is applied.

- `broker_config`
  - columns: `key`, `value`, `updated_at`
  - bootstrap rows: `fingerprint_salt`, `abuse_controls`, `abuse_runtime_state`
  - runtime-tunable non-secret operational controls live in `abuse_controls` so operators do not need code changes for threshold updates
  - persisted mutable runtime state lives separately in `abuse_runtime_state` so brake status, alert latches, and last daily-heartbeat delivery metadata do not get mixed into the editable threshold policy blob
  - malformed `abuse_controls` payloads fall back to the built-in default layout/thresholds instead of disabling enforcement or surfacing 500s
  - constraints: keys are limited to the supported config rows for this rollout and `value` must be valid JSON
  - `abuse_controls` fixes the settled endpoint/dimension layout:
    - `POST /v1/trial/challenge`: per IP, `10` requests / `15` minutes
    - `POST /v1/trial/challenge/verify`: per `installation_id`, `5` requests / `15` minutes
    - `POST /v1/providers/openrouter/issue`: per `installation_id`, `3` requests / `15` minutes
    - `GET /v1/trial/status`: per `installation_id`, `30` requests / `15` minutes
    - `POST /v1/auth/qq/assert`: per IP via `qqAuthAssertIp`, `20` requests / `15` minutes
    - `POST /v1/telemetry/translation-success-day`: per IP via `telemetryTranslationSuccessDayIp`, `60` requests / `15` minutes
    - global UTC-day cap on new active entitlements, counted by `issued_at` semantics even if an entitlement is later revoked, stored as a runtime-configurable broker value
- `broker_issue_success_events`
  - append-only successful issue observations recorded only after child-key creation and entitlement persistence both succeed
  - feeds immediate-alert evaluation, daily heartbeat rollups, and retention cleanup
  - columns include `issue_source`, nullable `installation_id`, `subject_ref`, `managed_credential_ref`, safe network metadata, and `observed_at`
  - Discord rows use `issue_source = 'discord'`, retain `installation_id`, and set `subject_ref` to that same installation identity; QQ rows use `issue_source = 'qq'`, leave `installation_id` `NULL`, and set `subject_ref` to `qq_subject_ref`
  - QQ monitoring/reporting must not synthesize installation rows, and no raw QQ identity, raw credential, raw OpenRouter key, raw Broker payload, or raw OpenRouter payload belongs in issue-success events
- `broker_abuse_runtime_audit`
  - append-only audit trail for brake transitions and other persisted abuse-runtime actions
- `broker_request_events`
  - append-only request observations used for per-endpoint rate limiting and cross-endpoint velocity hooks
  - columns: `id`, `endpoint`, `ip`, `installation_id`, `observed_at`
  - indexes cover endpoint-scoped and subject-scoped sliding-window lookups
- `telemetry_active_days`
  - active-day telemetry rows keyed by `(subject_ref, active_date_utc)`
  - columns: HMAC-derived `subject_ref`, UTC active date, `first_received_at`, and `last_received_at`
  - raw telemetry identifiers, account identities, Discord/QQ/hardware identities, provider payloads, model/language/output-route values, and Translation content do not belong in this table
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
  - columns: `qq_subject_ref`, `status`, `issue_ref`, nullable `managed_credential_ref`, `budget_usd`, `reserved_at`, `issued_at`, `expires_at`, `delivered_at`, `created_at`, and `updated_at`
  - stored statuses are `issuing`, `active`, `cleanup_required`, and `revoked`; `active`, `cleanup_required`, and `revoked` block automatic reissue
  - `active` requires `managed_credential_ref`, `issued_at`, `expires_at`, and `delivered_at`; `cleanup_required` requires `managed_credential_ref`; stale `issuing` rows can be reclaimed only when no child-key hash was recorded
  - existing `qq_auth_assertions` rows without a `qq_managed_entitlements` row remain eligible for their first production issuance
  - stores derived and operational metadata only; raw QQ identities, raw credentials, and raw OpenRouter API keys do not belong in this table
- `referral_codes`
  - stable owned Referral ID rows keyed by `referral_id`
  - columns: `referral_id`, `owner_discord_user_ref`, `owner_installation_id`, `status`, `created_at`, `updated_at`
  - Referral IDs are exactly six characters from the approved uppercase alphabet excluding `0`, `O`, `1`, `I`, and `L`; statuses are `active` or `disabled`
  - `owner_discord_user_ref` is unique and raw Discord IDs or email addresses do not belong in this table
  - no `ON DELETE CASCADE` dependency on `installations`, preserving code history when installation rows age out
- `referral_rewards`
  - append-only referral attempt/reward ledger rows keyed by `id`
  - columns: `id`, `referral_id`, referrer/referred identity and installation references, referred hardware hash/salt version, referred/referrer bonus statuses, bounded reason codes, managed credential refs, and lifecycle timestamps
  - Referral IDs use the same approved six-character constraint; referred-side statuses are `reserved`, `credited`, `skipped`, and `failed`; referrer-side statuses are `pending`, `applying`, `credited`, `skipped`, and `failed`
  - cap queries are indexed by `referrer_discord_user_ref` plus referred-side status, and referral lookup is indexed by `referral_id`
  - partial unique indexes enforce one counted (`reserved`/`credited`) reward per referred Discord identity and per referred installation
  - ledger rows do not cascade-delete with `installations`, preserving cap/accounting history when installation rows age out

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
