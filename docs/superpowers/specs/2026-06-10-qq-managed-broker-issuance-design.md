# QQ Managed Broker Issuance Design

## Status

Draft for user review. The selected direction is approved in conversation:

- Keep the app-facing QQ API contract compatible with the current desktop app.
- Use existing Discord Managed broker behavior as the model wherever it fits.
- Reuse existing broker defense, OpenRouter child-key, guardrail, cleanup, cap, and monitoring infrastructure as much as possible.

## Context

The current Broker endpoint `POST /v1/auth/qq/assert` is a narrow QQ Bot HMAC assertion endpoint. It verifies a QQ assertion, stores anonymized assertion evidence in `qq_auth_assertions`, and returns `verified` or `already_verified` with a derived `qq_subject_ref`. It does not issue OpenRouter child keys.

The desktop app has been aligned for the future key-bearing QQ response shape. It treats `verified` / `already_verified` as verification-only and not ready for translation, and it stores a QQ Managed key only when the Broker returns a valid top-level `openrouter_api_key` in an issued response.

This design turns the Broker QQ path into production Managed China issuance while preserving the app-facing endpoint and request contract.

## Goals

1. Make valid QQ assertions issue a Managed OpenRouter child key through the existing `POST /v1/auth/qq/assert` endpoint.
2. Preserve the current app API contract: no new required request fields and no new app-side endpoint sequence.
3. Model QQ issuance after Discord Managed issuance internally: reservation, child-key creation, guardrail assignment, activation, cleanup, and cleanup-required handling.
4. Keep QQ identity handling privacy-preserving: no raw QQ identity, raw credential, raw Broker payload, or raw OpenRouter key in D1 logs, diagnostics, docs, or tests except the one-time success response that must deliver the key to the app.
5. Reuse existing Managed trial policy, OpenRouter management helpers, guardrail assignment, issue caps, monitoring, error envelopes, and abuse-control infrastructure where technically compatible.

## Non-Goals

- Add a new app-facing endpoint such as `POST /v1/providers/openrouter/qq/issue`.
- Require app installation ID, device public key, hardware hash, or app signature for the first QQ production issuance path.
- Change the QQ Bot credential formula in this work.
- Store raw QQ identity, raw credential, or raw OpenRouter API keys.
- Rework Discord Managed behavior or referral behavior.
- Build an admin dashboard or manual remediation UI.

## Compatibility Contract

### Runtime Gate

QQ issuance is enabled only when all issuance-critical runtime configuration is present and non-blank:

- `QQ_AUTH_HMAC_PSK`
- `OPENROUTER_MANAGEMENT_API_KEY`
- `OPENROUTER_MANAGED_GUARDRAIL_ID`

When issuance is disabled because OpenRouter issuance configuration is absent, valid assertions preserve the existing verification-only behavior: they may return `verified` or `already_verified` and must not touch `qq_managed_entitlements`. When issuance is enabled but an OpenRouter call, guardrail assignment, cleanup, or D1 operation fails, the Broker returns a bounded retryable/internal error envelope rather than falling back to `verified`.

### Request

The endpoint remains:

```http
POST /v1/auth/qq/assert
Content-Type: application/json
```

Request body remains:

```json
{
  "qq_identity": "stable-bot-observed-qq-identifier",
  "credential": "64char_lowercase_hex_hmac_sha256",
  "asserted_at": "2026-06-05T12:03:00Z"
}
```

The credential contract stays as implemented today:

```text
credential = HMAC-SHA256-HEX(QQ_AUTH_HMAC_PSK, qq_identity)
```

`asserted_at` remains a required audit/debug input. It is not part of the HMAC payload in this design. Broker trust decisions use server-side receive time plus the derived `qq_subject_ref` and entitlement lifecycle.

Request validation must remain stricter than the current test-only endpoint before production issuance is enabled:

- `qq_identity` must be a bounded non-empty string accepted by the current app contract; raw value is never persisted.
- `credential` must be exactly 64 lowercase hexadecimal characters before HMAC comparison.
- `asserted_at` must be a bounded ISO-8601 timestamp in the same strict subset used by existing Broker signed timestamps, and the Broker stores only normalized timestamp text. Arbitrary request text must not be persisted through `asserted_at`.

### Success Response

When production issuance is available and a valid QQ assertion is eligible for first issuance, the endpoint returns the top-level key-bearing shape the app already supports:

```json
{
  "ok": true,
  "status": "issued",
  "qq_subject_ref": "ph-qq-subject-v1_...",
  "openrouter_api_key": "sk-or-v1-...",
  "managed_credential_ref": "openrouter-child-key-hash",
  "expires_at": "2026-09-10T00:00:00.000Z",
  "openrouter_user_id": "managed-user-v1_..."
}
```

`openrouter_user_id` remains optional, following the existing Managed OpenRouter user-id derivation behavior. `openrouter_api_key` is returned only once as the immediate issue response and is never stored by the Broker.

### Legacy Verification Responses

The existing `verified` and `already_verified` success shapes remain recognizable for backward compatibility and for non-issuing environments. In the production issuance path, a valid eligible assertion must produce `issued`; inability to issue must return a bounded error envelope rather than pretending translation is ready.

## Subject and Identity Model

The Broker continues to derive the production v1 subject as:

```text
qq_subject_ref = "ph-qq-subject-v1_" + base64url(
  HMAC-SHA256(QQ_AUTH_HMAC_PSK, "puripuly-heart:qq-subject:v1\n" + qq_identity)
)
```

Because `qq_subject_ref` is the lifetime-enforcement key, `QQ_AUTH_HMAC_PSK` rotation must not silently change the subject for an already-eligible QQ identity. If the credential PSK must rotate, the Broker must use a versioned subject-ref rotation plan before changing production behavior: continue recognizing existing v1 refs for lifetime checks, or introduce a new subject-ref HMAC secret and prefix in a separate migration with dual lookup/backfill semantics. A simple secret replacement that makes old QQ identities derive unrelated subject refs is not allowed.

All QQ production issuance state is keyed by `qq_subject_ref`. Raw `qq_identity` is used only for immediate HMAC verification and subject derivation during the request. It must not be persisted or logged.

OpenRouter child-key naming, issue-session references, monitoring subjects, and managed OpenRouter user-id derivation must use `qq_subject_ref` or a derived non-sensitive shortened reference, never raw QQ identity.

## Persistence Model

### Existing `qq_auth_assertions`

`qq_auth_assertions` remains assertion evidence only:

- `qq_subject_ref`
- `credential_hash`
- `asserted_at`
- `received_at`
- `status = 'verified'`

Production issuance must not mutate this table into the entitlement lifecycle source of truth. Existing verified-only rows remain valid assertion evidence.

If a valid request arrives for a `qq_subject_ref` that already exists in `qq_auth_assertions` but has no QQ entitlement, the Broker must treat it as the subject's first production issuance attempt and proceed with issuance.

### New `qq_managed_entitlements`

Add a QQ-specific entitlement table keyed by `qq_subject_ref`. It must store only derived and operational metadata:

- `qq_subject_ref TEXT PRIMARY KEY`
- `status TEXT NOT NULL CHECK(status IN ('issuing', 'active', 'cleanup_required', 'revoked'))` with values:
  - `issuing`
  - `active`
  - `cleanup_required`
  - `revoked`
- `issue_ref TEXT NOT NULL UNIQUE`, a non-sensitive issue attempt reference derived from `qq_subject_ref`, issue source, and server-side issue time or randomness
- `managed_credential_ref TEXT`, unique when present through a partial unique index
- `budget_usd REAL NOT NULL CHECK(budget_usd >= 0)`
- `reserved_at TEXT NOT NULL`
- `issued_at TEXT`
- `expires_at TEXT`
- `delivered_at TEXT`
- `created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP`
- `updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP`

The table must not contain raw QQ identity, raw credential, or raw OpenRouter API key.

The D1 table must be `STRICT`. Required indexes:

- partial unique index on `managed_credential_ref` where it is not `NULL`;
- index on `status, updated_at` for cleanup/remediation scans;
- index on `expires_at` for future retention/lifecycle work;
- index on `issue_ref` if the unique constraint is not already represented by an index.

State invariants:

- `active` requires `managed_credential_ref`, `issued_at`, `expires_at`, and `delivered_at`.
- `cleanup_required` requires `managed_credential_ref`.
- `issuing` requires `managed_credential_ref IS NULL` unless the implementation records the child-key hash before guardrail/activation for crash recovery; if it does record the hash, stale issuing rows with a hash must be treated as remediation candidates rather than silently overwritten.
- `revoked` blocks automatic reissue whether or not historical key metadata is present.

`revoked` is included for Discord-like operational control. A revoked QQ entitlement blocks future automatic issuance unless a later explicit admin remediation design says otherwise.

## Issuance Lifecycle

The external endpoint is one-step, but the internal lifecycle follows Discord Managed patterns.

1. Record the request event and apply the existing QQ assert IP rate limit.
2. Validate JSON and required fields.
3. Fail closed if `QQ_AUTH_HMAC_PSK` is missing or blank.
4. Verify `credential = HMAC_SHA256_HEX(QQ_AUTH_HMAC_PSK, qq_identity)`.
5. Derive `qq_subject_ref` and `credential_hash`.
6. Insert assertion evidence into `qq_auth_assertions` with `ON CONFLICT DO NOTHING` so legacy evidence remains stable.
7. Apply the active issuance brake before external OpenRouter side effects. QQ must use the existing brake semantics through a source-aware adapter; it must not fake an `openrouter_entitlements` row.
8. Reserve a QQ entitlement:
   - no row: insert `issuing`
   - `issuing`: return `qq_already_issuing`
   - `active`, `cleanup_required`, or `revoked`: return `qq_lifetime_used`
9. Apply global daily issuance cap and compatible abuse hooks before creating an OpenRouter child key. The cap check and reservation must be concurrency-safe: either the cap is part of the reservation transaction/conditional write, or a cap rejection releases the matching `issuing` reservation.
10. Create an OpenRouter Managed child key with the existing Managed trial policy.
11. Assign the configured Managed guardrail to the child key.
12. Activate the QQ entitlement as `active` with `managed_credential_ref`, `issued_at`, `expires_at`, `delivered_at`, and budget metadata.
13. Record issue-success monitoring using redacted/safe metadata.
14. Return the one-time issued response with `openrouter_api_key` and metadata.

Activation and cleanup must be attempt-scoped. Updates and deletes that change entitlement lifecycle must match `qq_subject_ref`, `issue_ref`, expected `status`, and, once a child key exists, the expected `managed_credential_ref`. A stale or late cleanup must not touch a later attempt or an unrelated active entitlement.

Failure handling follows the Discord cleanup pattern with separate phases:

- Before a child key exists: release only the matching `issuing` reservation for the current `issue_ref`.
- After child-key creation but before activation: attempt to disable/delete the child key. If cleanup succeeds, delete only the matching `issuing` reservation for the current `issue_ref`; if cleanup fails, mark that row `cleanup_required` with the child-key hash.
- After activation but before the response is safely deliverable: either treat downstream monitoring as non-critical, or, if a critical failure requires cleanup, cleanup may rollback/delete the same-attempt `active` QQ row only when `issue_ref` and `managed_credential_ref` both match the cleaned child key. Never delete unrelated `active`, `cleanup_required`, or `revoked` rows.

Stale `issuing` reservations must not block users forever. The implementation must define a QQ issuing reservation TTL. A stale `issuing` row with no `managed_credential_ref` may be released/reclaimed by a later valid request for the same `qq_subject_ref`. A stale `issuing` row that has a `managed_credential_ref` is a cleanup/remediation candidate and must not be silently overwritten.

## Error Mapping

All errors use the existing public error-envelope style.

QQ-specific subcodes:

- `qq_credential_invalid`: credential does not match the QQ HMAC contract.
- `qq_lifetime_used`: `qq_subject_ref` already has `active`, `cleanup_required`, or `revoked` entitlement state.
- `qq_already_issuing`: another request already reserved issuance for the same `qq_subject_ref`. This must be returned with a retryable public envelope so existing app builds can fall back to generic QQ retry copy without a subcode-specific UI update.

Existing/shared subcodes and classes remain in use where applicable:

- `ip_rate_limited` for the existing QQ assert IP rate limit.
- `global_cap_reached` or the existing global issuance cap subcode if the shared cap is reached. Current app builds are not required to show QQ-specific daily-cap copy; a retryable/suspended public envelope is acceptable.
- existing retryable/internal error envelopes for OpenRouter or cleanup failures.

Raw Broker payloads, raw OpenRouter payloads, raw credentials, raw QQ identities, raw exception text, and raw OpenRouter keys must not be reflected in public errors or logs.

## Policy Reuse

QQ issuance uses the same Managed trial policy as Discord Managed issuance:

- same budget hard limit;
- same expiry duration;
- same allowed model policy and guardrail;
- same OpenRouter child-key creation and guardrail assignment helpers;
- same raw-key non-persistence rule.

The key name or release-session reference supplied to OpenRouter must distinguish QQ issuance from Discord while avoiding raw identity. Use the source tag `qq` plus `issue_ref`; do not place the raw QQ identity or credential in the child-key name.

Refactor Managed child-key helpers so they accept source-aware naming inputs such as `issue_source`, `subject_ref`, and `issue_ref`. Discord continues to pass its installation identity; QQ passes `qq_subject_ref`. Do not create synthetic installation rows or pass fake installation values to satisfy existing helper names.

QQ does not run Discord referral reservation, referral reward, owned Referral ID, Talk Together Pass, or referral bonus budget logic. QQ uses the base Managed trial budget.

## Abuse, Monitoring, and Operations

The QQ path must reuse compatible broker defenses:

- existing `qqAuthAssertIp` endpoint rate limit;
- active issuance brake before OpenRouter side effects;
- global daily issuance cap, counted with QQ `issuing`, `active`, and `cleanup_required` managed entitlement rows alongside Discord/other Managed issuance;
- issue-success monitoring and daily reporting using safe QQ metadata;
- OpenRouter child-key cleanup and cleanup-required alerting pattern.

Where existing helpers assume `installation_id`, the implementation must add the smallest source-aware extension rather than forcing fake installation values. `qq_subject_ref` is the QQ subject for lifetime, concurrency, monitoring, and future subject hooks.

The existing `broker_issue_success_events` table currently requires `installation_id`. QQ issuance must not create synthetic installation rows just to satisfy that shape. Instead, migrate issue-success events to a source-aware model in the same spirit as the existing monitoring path:

- add `issue_source` with values `discord` and `qq`;
- make `installation_id` nullable so QQ rows do not require a fake installation;
- add `subject_ref`, populated with `installation_id` for Discord rows and `qq_subject_ref` for QQ rows;
- existing network metadata, credential-ref, and observed-at columns.

Daily reporting and immediate alert logic must continue to work for Discord and include QQ issue successes without exposing raw QQ identity. Discord rows must remain queryable by `installation_id`; QQ rows must be queryable by `subject_ref = qq_subject_ref`.

The issue-success migration must be a safe table rebuild if D1/SQLite cannot alter the existing `installation_id NOT NULL` foreign-key shape in place. It must preserve existing `id` and `observed_at` values where possible, copy current rows as `issue_source = 'discord'` and `subject_ref = installation_id`, recreate compatible indexes, and add an index on `(issue_source, subject_ref, observed_at)`. Tests must prove existing Discord monitoring/reporting still works after the migration and QQ rows can be inserted with `installation_id = NULL`.

Subject and velocity hooks may need a schema/type extension before they can target `qq_subject_ref`. That extension is not required for the first production issuance if it would broaden the implementation. Initial QQ lifetime/concurrency protection is provided by `qq_managed_entitlements`, the global issuance cap, and endpoint IP rate limiting.

## Security and Privacy

- Never store raw QQ identity.
- Never store raw credential.
- Never store raw OpenRouter API keys.
- Never log raw Broker request/response payloads.
- Never log raw OpenRouter payloads or exception text that may contain key material.
- `qq_subject_ref` is allowed as internal operational metadata, but must not be shown in user-facing UI copy.
- Tests and docs must use synthetic placeholders only.
- If a cleanup-required path logs diagnostics, redact all known sensitive values first.

## App Behavior Alignment

The desktop app already supports the issued response shape. Expected app behavior after this Broker change:

- `status: "issued"` with a valid `openrouter_api_key`: app stores the key under the QQ Managed secret and enables Managed China translation.
- `verified` / `already_verified`: app treats as key unavailable and does not enable translation.
- `qq_credential_invalid`: app keeps the QQ dialog recoverable and clears only the credential.
- `qq_lifetime_used`: app shows the existing QQ lifetime-used copy.
- `qq_already_issuing` or retryable failures: app shows retry/recoverable copy without falling back to Discord.

No app request-shape change is required by this design.

## Migration and Backward Compatibility

The migration adds `qq_managed_entitlements` without rewriting existing `qq_auth_assertions` rows.

Existing subjects that have only assertion evidence remain eligible for their first production issuance. Existing `verified` / `already_verified` client parsing remains valid. Broker deployment must continue to require `QQ_AUTH_HMAC_PSK`, `OPENROUTER_MANAGEMENT_API_KEY`, and `OPENROUTER_MANAGED_GUARDRAIL_ID` before production issuance can succeed.

The Broker README, persistence contract, direct deploy workflow, and deploy smoke tests must be updated to describe that `/v1/auth/qq/assert` is no longer test-only when issuance is enabled. The direct deploy smoke must exercise QQ production issuance with synthetic non-PII values: valid QQ assertion returns `issued`, includes a one-time `openrouter_api_key`, verifies child-key metadata/guardrail behavior, and redacts the key, QQ identity, and credential from failure output.

## Testing Requirements

Broker tests must cover at minimum:

- valid first QQ assertion returns `issued` and top-level OpenRouter issue fields;
- raw QQ identity and raw credential are not persisted;
- raw OpenRouter key is returned once but not persisted;
- existing assertion row without entitlement can still issue;
- active entitlement returns `qq_lifetime_used` and does not create another child key;
- cleanup-required and revoked entitlement states return `qq_lifetime_used`;
- concurrent issuing returns `qq_already_issuing`;
- invalid credential returns `qq_credential_invalid` and does not reserve or issue;
- strict request validation rejects malformed `credential`, over-broad `qq_identity`, and `asserted_at` smuggling attempts without persisting raw request text;
- IP rate limiting still counts malformed and invalid attempts;
- active issuance brake and global issuance cap apply to QQ issuance;
- stale `issuing` reservation TTL behavior is covered for no-key and key-hash cases;
- OpenRouter child key creation, guardrail assignment, activation, cleanup success, and cleanup-required paths follow Discord-like behavior;
- source-aware issue-success migration preserves existing Discord rows, allows QQ rows without fake installations, and keeps daily reporting/immediate alert queries working;
- missing/blank `OPENROUTER_MANAGEMENT_API_KEY`, `OPENROUTER_MANAGED_GUARDRAIL_ID`, and `QQ_AUTH_HMAC_PSK` return bounded redacted behavior according to the runtime gate;
- public errors and captured logs exclude raw QQ identity, credential, OpenRouter key, raw Broker payload, and raw OpenRouter payload;
- `verified` / `already_verified` legacy behavior remains covered only where explicitly expected for non-issuing compatibility.

Because Broker Node verification is Linux-only in this repository, implementation verification must run from a Linux/WSL workspace, not from Windows shells or Windows-installed `node_modules`.

Expected verification commands from a Linux-native or WSL workspace include:

```bash
pnpm install --frozen-lockfile
pnpm exec vitest run broker/tests
pnpm --filter @puripuly-heart/broker run verify:config
```

## Readiness Criteria

The Broker QQ production issuance work is ready when:

1. `POST /v1/auth/qq/assert` preserves its request contract and returns `issued` for eligible valid first issuance.
2. QQ issuance uses a Discord-like reservation lifecycle with `issuing`, `active`, `cleanup_required`, and `revoked` states.
3. Existing Managed trial policy, child-key creation, guardrail assignment, cleanup, cap, and monitoring behavior are reused or minimally source-extended.
4. Duplicate/lifetime/concurrency cases are bounded by `qq_lifetime_used` and `qq_already_issuing` without raw key recovery or reissue.
5. No raw QQ identity, raw credential, raw OpenRouter key, or raw provider payload is persisted or logged.
6. Existing app parsing and Managed China QQ route behavior continue to work without a request-shape change.
7. Broker tests and Linux/WSL verification pass.
