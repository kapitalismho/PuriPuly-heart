import { describe, expect, it } from 'vitest';

import type { ManagedCleanupOutcomeForAudit } from '../src/managed-issuance';
import {
  buildManagedCleanupRequiredAuditPayload,
  getManagedIssuanceSourcePolicy,
} from '../src/managed-issuance';

type AssertFalse<T extends false> = T;
type HasSuccessfulCleanupOutcome = Extract<
  ManagedCleanupOutcomeForAudit,
  { ok: true }
> extends never
  ? false
  : true;
type RejectSuccessfulCleanupOutcome = AssertFalse<HasSuccessfulCleanupOutcome>;

describe('Managed issuance source policy', () => {
  it('keeps QQ on the base Managed trial policy and excludes Discord referral surfaces', () => {
    expect(getManagedIssuanceSourcePolicy('qq')).toEqual({
      issue_source: 'qq',
      budget_usd: 0.07,
      uses_managed_trial_guardrail: true,
      discord_referral_reservation: false,
      owned_referral_id: false,
      talk_together_pass: false,
      referral_bonus_budget: false,
    });
  });

  it('keeps Discord marked as the only source that can use referral and Talk Together surfaces', () => {
    expect(getManagedIssuanceSourcePolicy('discord')).toEqual({
      issue_source: 'discord',
      budget_usd: 0.07,
      uses_managed_trial_guardrail: true,
      discord_referral_reservation: true,
      owned_referral_id: true,
      talk_together_pass: true,
      referral_bonus_budget: true,
    });
  });
});

describe('Managed cleanup-required audit payloads', () => {
  it('uses source-aware QQ metadata and redacts sensitive cleanup diagnostics', () => {
    const rawCredential = 'abcdef'.repeat(10) + 'abcd';
    const rawChildKey = 'sk-or-v1-sensitive-child-key-value';
    const payload = buildManagedCleanupRequiredAuditPayload({
      issueSource: 'qq',
      subjectRef: 'ph-qq-subject-v1_synthetic-cleanup-subject',
      issueRef: 'qq-issue-cleanup-required',
      managedCredentialRef: 'hash_qq_managed_cleanup_required',
      failure: new Error(`guardrail failure ${rawCredential} ${rawChildKey}`),
      cleanupOutcome: {
        disable: {
          ok: false,
          error: {
            operation: 'disable_key',
            code: 'upstream_http_error',
            status: 500,
            upstreamCode: 500,
            message: `cleanup disable failed ${rawCredential}`,
          },
        },
        delete: { ok: true },
      },
      sensitiveValues: [rawCredential, rawChildKey],
      brokerTimestamp: '2026-06-10T06:00:00.000Z',
    });

    expect(payload).toEqual(
      expect.objectContaining({
        event: 'managed_child_key_cleanup_required',
        issue_source: 'qq',
        subject_ref: 'ph-qq-subject-v1_synthetic-cleanup-subject',
        issue_ref: 'qq-issue-cleanup-required',
        managed_credential_ref: 'hash_qq_managed_cleanup_required',
        broker_timestamp: '2026-06-10T06:00:00.000Z',
      }),
    );
    expect(payload).not.toHaveProperty('installation_id');
    expect(JSON.stringify(payload)).not.toContain(rawCredential);
    expect(JSON.stringify(payload)).not.toContain(rawChildKey);
    expect(payload.failure).toEqual({ name: 'Error' });
    expect(payload.cleanup_outcome).toEqual({
      disable: {
        ok: false,
        error: {
          operation: 'disable_key',
          code: 'upstream_http_error',
          status: 500,
          upstreamCode: 500,
        },
      },
      delete: { ok: true },
    });
  });

  it('omits unlisted raw exception and upstream cleanup messages from QQ cleanup diagnostics', () => {
    const rawExceptionMessage = 'synthetic-unlisted-openrouter-exception-text';
    const rawCleanupMessage = 'synthetic-unlisted-cleanup-upstream-message';

    const payload = buildManagedCleanupRequiredAuditPayload({
      issueSource: 'qq',
      subjectRef: 'ph-qq-subject-v1_synthetic-bounded-cleanup-subject',
      issueRef: 'qq-issue-bounded-cleanup-required',
      managedCredentialRef: 'hash_qq_managed_bounded_cleanup_required',
      failure: new Error(rawExceptionMessage),
      cleanupOutcome: {
        disable: {
          ok: false,
          error: {
            operation: 'disable_key',
            code: 'upstream_http_error',
            status: 502,
            upstreamCode: 500,
            message: rawCleanupMessage,
          },
        },
        delete: { ok: true },
      },
      sensitiveValues: [],
      brokerTimestamp: '2026-06-10T06:00:00.000Z',
    });

    const serializedPayload = JSON.stringify(payload);
    expect(serializedPayload).not.toContain(rawExceptionMessage);
    expect(serializedPayload).not.toContain(rawCleanupMessage);
    expect(payload.failure).toEqual({ name: 'Error' });
    expect(payload.cleanup_outcome).toEqual({
      disable: {
        ok: false,
        error: {
          operation: 'disable_key',
          code: 'upstream_http_error',
          status: 502,
          upstreamCode: 500,
        },
      },
      delete: { ok: true },
    });
  });

  it('unwraps the full cleanup failure result before auditing per-step diagnostics', () => {
    const rawDisableMessage = 'synthetic-disable-step-raw-openrouter-message';
    const rawDeleteMessage = 'synthetic-delete-step-raw-openrouter-message';

    const payload = buildManagedCleanupRequiredAuditPayload({
      issueSource: 'qq',
      subjectRef: 'ph-qq-subject-v1_synthetic-full-cleanup-result-subject',
      issueRef: 'qq-issue-full-cleanup-result',
      managedCredentialRef: 'hash_qq_managed_full_cleanup_result',
      failure: new Error('synthetic-child-key-post-create-failure'),
      cleanupOutcome: {
        ok: false,
        reason: {
          disable: {
            ok: false,
            error: {
              operation: 'disable_key',
              code: 'network_error',
              status: null,
              upstreamCode: null,
              message: rawDisableMessage,
            },
          },
          delete: {
            ok: false,
            error: {
              operation: 'delete_key',
              code: 'upstream_http_error',
              status: 502,
              upstreamCode: 500,
              message: rawDeleteMessage,
            },
          },
        },
      },
      brokerTimestamp: '2026-06-10T06:00:00.000Z',
    });

    expect(payload.cleanup_outcome).toEqual({
      disable: {
        ok: false,
        error: {
          operation: 'disable_key',
          code: 'network_error',
          status: null,
          upstreamCode: null,
        },
      },
      delete: {
        ok: false,
        error: {
          operation: 'delete_key',
          code: 'upstream_http_error',
          status: 502,
          upstreamCode: 500,
        },
      },
    });
    const serializedPayload = JSON.stringify(payload);
    expect(serializedPayload).not.toContain(rawDisableMessage);
    expect(serializedPayload).not.toContain(rawDeleteMessage);
    expect(serializedPayload).not.toContain('"error":null');
  });
});
