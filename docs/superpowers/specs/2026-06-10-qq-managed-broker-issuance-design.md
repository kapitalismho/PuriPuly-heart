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

The existing `verified` and `already_verified` success shapes remain recognizable for backward compatibility and for non-issuing environments. In the production issuance path, a valid eligible assertion should produce `issued`; inability to issue should return a bounded error envelope rather than pretending translation is ready.

## Subject and Identity Model

The Broker continues to derive:

```text
qq_subject_ref = "ph-qq-subject-v1_" + base64url(
  HMAC-SHA256(QQ_AUTH_HMAC_PSK, "puripuly-heart:qq-subject:v1\n" + qq_identity)
)
```

All QQ production issuance state is keyed by `qq_subject_ref`. Raw `qq_identity` is used only for immediate HMAC verification and subject derivation during the request. It must not be persisted or logged.

OpenRouter child-key naming, issue-session references, monitoring subjects, and managed OpenRouter user-id derivation should use `qq_subject_ref` or a derived non-sensitive shortened reference, never raw QQ identity.

## Persistence Model

### Existing `qq_auth_assertions`

`qq_auth_assertions` remains assertion evidence only:

- `qq_subject_ref`
- `credential_hash`
- `asserted_at`
- `received_at`
- `status = 'verified'`

Production issuance must not mutate this table into the entitlement lifecycle source of truth. Existing verified-only rows remain valid assertion evidence.

If a valid request arrives for a `qq_subject_ref` that already exists in `qq_auth_assertions` but has no QQ entitlement, the Broker should treat it as the subject's first production issuance attempt and proceed with issuance.

### New `qq_managed_entitlements`

Add a QQ-specific entitlement table keyed by `qq_subject_ref`. It should store only derived and operational metadata:

- `qq_subject_ref` primary key
- `status` with values:
  - `issuing`
  - `active`
  - `cleanup_required`
  - `revoked`
- `managed_credential_ref` nullable, unique when present
- `budget_usd`
- `issued_at`
- `expires_at`
- `issue_ref`, a non-sensitive issue attempt reference derived from `qq_subject_ref`, issue source, and server-side issue time or randomness
- `created_at`
- `updated_at`

The table must not contain raw QQ identity, raw credential, or raw OpenRouter API key.

`revoked` is included for Discord-like operational control. A revoked QQ entitlement blocks future automatic issuance unless a later explicit admin remediation design says otherwise.

## Issuance Lifecycle

The external endpoint is one-step, but the internal lifecycle follows Discord Managed patterns.

1. Record the request event and apply the existing QQ assert IP rate limit.
2. Validate JSON and required fields.
3. Fail closed if `QQ_AUTH_HMAC_PSK` is missing or blank.
4. Verify `credential = HMAC_SHA256_HEX(QQ_AUTH_HMAC_PSK, qq_identity)`.
5. Derive `qq_subject_ref` and `credential_hash`.
6. Insert assertion evidence into `qq_auth_assertions` with `ON CONFLICT DO NOTHING` so legacy evidence remains stable.
7. Reserve a QQ entitlement:
   - no row: insert `issuing`
   - `issuing`: return `qq_already_issuing`
   - `active`, `cleanup_required`, or `revoked`: return `qq_lifetime_used`
8. Apply global daily issuance cap and compatible abuse hooks before creating an OpenRouter child key.
9. Create an OpenRouter Managed child key with the existing Managed trial policy.
10. Assign the configured Managed guardrail to the child key.
11. Activate the QQ entitlement as `active` with `managed_credential_ref`, `issued_at`, `expires_at`, and budget metadata.
12. Record issue-success monitoring using redacted/safe metadata.
13. Return the one-time issued response with `openrouter_api_key` and metadata.

If child-key creation succeeds but guardrail assignment, activation, or monitoring-critical state fails, the Broker should follow the Discord cleanup pattern:

- attempt to disable/delete the child key through existing cleanup helpers;
- if cleanup succeeds, delete the QQ `issuing` reservation only when the row still belongs to the same `qq_subject_ref` and either has no `managed_credential_ref` or has the cleaned child-key hash; do not delete `active`, `cleanup_required`, or `revoked` rows;
- if cleanup fails, mark the QQ entitlement `cleanup_required`, log only redacted operational metadata, and return a bounded error.

## Error Mapping

All errors use the existing public error-envelope style.

QQ-specific subcodes:

- `qq_credential_invalid`: credential does not match the QQ HMAC contract.
- `qq_lifetime_used`: `qq_subject_ref` already has `active`, `cleanup_required`, or `revoked` entitlement state.
- `qq_already_issuing`: another request already reserved issuance for the same `qq_subject_ref`.

Existing/shared subcodes and classes remain in use where applicable:

- `ip_rate_limited` for the existing QQ assert IP rate limit.
- `global_cap_reached` or the existing global issuance cap subcode if the shared cap is reached.
- existing retryable/internal error envelopes for OpenRouter or cleanup failures.

Raw Broker payloads, raw OpenRouter payloads, raw credentials, raw QQ identities, raw exception text, and raw OpenRouter keys must not be reflected in public errors or logs.

## Policy Reuse

QQ issuance uses the same Managed trial policy as Discord Managed issuance:

- same budget hard limit;
- same expiry duration;
- same allowed model policy and guardrail;
- same OpenRouter child-key creation and guardrail assignment helpers;
- same raw-key non-persistence rule.

The key name or release-session reference supplied to OpenRouter should distinguish QQ issuance from Discord while avoiding raw identity. Use the source tag `qq` plus `issue_ref`; do not place the raw QQ identity or credential in the child-key name.

## Abuse, Monitoring, and Operations

The QQ path should reuse compatible broker defenses:

- existing `qqAuthAssertIp` endpoint rate limit;
- global daily issuance cap, counted with QQ active issuance alongside Discord/other Managed issuance;
- issue-success monitoring and daily reporting using safe QQ metadata;
- OpenRouter child-key cleanup and cleanup-required alerting pattern.

Where existing helpers assume `installation_id`, the implementation must add the smallest source-aware extension rather than forcing fake installation values. `qq_subject_ref` is the QQ subject for lifetime, concurrency, monitoring, and future subject hooks.

The existing `broker_issue_success_events` table currently requires `installation_id`. QQ issuance must not create synthetic installation rows just to satisfy that shape. Instead, migrate issue-success events to a source-aware model in the same spirit as the existing monitoring path:

- add `issue_source` with values `discord` and `qq`;
- make `installation_id` nullable so QQ rows do not require a fake installation;
- add `subject_ref`, populated with `installation_id` for Discord rows and `qq_subject_ref` for QQ rows;
- existing network metadata, credential-ref, and observed-at columns.

Daily reporting and immediate alert logic should continue to work for Discord and include QQ issue successes without exposing raw QQ identity. Discord rows must remain queryable by `installation_id`; QQ rows must be queryable by `subject_ref = qq_subject_ref`.

Subject and velocity hooks may need a schema/type extension before they can target `qq_subject_ref`. That extension is not required for the first production issuance if it would broaden the implementation. Initial QQ lifetime/concurrency protection is provided by `qq_managed_entitlements`, the global issuance cap, and endpoint IP rate limiting.

## Security and Privacy

- Never store raw QQ identity.
- Never store raw credential.
- Never store raw OpenRouter API keys.
- Never log raw Broker request/response payloads.
- Never log raw OpenRouter payloads or exception text that may contain key material.
- `qq_subject_ref` is allowed as internal operational metadata, but should not be shown in user-facing UI copy.
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

The Broker README, persistence contract, and tests should be updated to describe that `/v1/auth/qq/assert` is no longer test-only when issuance is enabled.

## Testing Requirements

Broker tests should cover at minimum:

- valid first QQ assertion returns `issued` and top-level OpenRouter issue fields;
- raw QQ identity and raw credential are not persisted;
- raw OpenRouter key is returned once but not persisted;
- existing assertion row without entitlement can still issue;
- active entitlement returns `qq_lifetime_used` and does not create another child key;
- cleanup-required and revoked entitlement states return `qq_lifetime_used`;
- concurrent issuing returns `qq_already_issuing`;
- invalid credential returns `qq_credential_invalid` and does not reserve or issue;
- IP rate limiting still counts malformed and invalid attempts;
- global issuance cap applies to QQ issuance;
- OpenRouter child key creation, guardrail assignment, activation, cleanup success, and cleanup-required paths follow Discord-like behavior;
- public errors and captured logs exclude raw QQ identity, credential, OpenRouter key, raw Broker payload, and raw OpenRouter payload;
- `verified` / `already_verified` legacy behavior remains covered only where explicitly expected for non-issuing compatibility.

Because Broker Node verification is Linux-only in this repository, implementation verification must run from a Linux/WSL workspace, not from Windows shells or Windows-installed `node_modules`.

## Readiness Criteria

The Broker QQ production issuance work is ready when:

1. `POST /v1/auth/qq/assert` preserves its request contract and returns `issued` for eligible valid first issuance.
2. QQ issuance uses a Discord-like reservation lifecycle with `issuing`, `active`, `cleanup_required`, and `revoked` states.
3. Existing Managed trial policy, child-key creation, guardrail assignment, cleanup, cap, and monitoring behavior are reused or minimally source-extended.
4. Duplicate/lifetime/concurrency cases are bounded by `qq_lifetime_used` and `qq_already_issuing` without raw key recovery or reissue.
5. No raw QQ identity, raw credential, raw OpenRouter key, or raw provider payload is persisted or logged.
6. Existing app parsing and Managed China QQ route behavior continue to work without a request-shape change.
7. Broker tests and Linux/WSL verification pass.
