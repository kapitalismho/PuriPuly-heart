import { afterEach, describe, expect, it, vi } from 'vitest';

import app from '../src/index';
import {
  createDeviceKeyPair,
  signCanonicalStatusRequest,
} from './test-support/ed25519';
import { insertSubjectHook, stableIpHookValue } from './test-support/abuse-controls';
import { activatePendingReleaseSession } from './test-support/openrouter-issue';
import { createTestBrokerEnv } from './test-support/sqlite-d1';
import { getTrialStatus } from './test-support/trial-api';

describe('broker denylist, reputation, and revocation hooks', () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it.each(['denylist', 'reputation'] as const)(
    'rejects challenge traffic when a matching %s hook is active',
    async (hookKind: 'denylist' | 'reputation') => {
      vi.useFakeTimers();
      vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

      const env = createTestBrokerEnv();
      const keyPair = await createDeviceKeyPair();

      insertSubjectHook(env, {
        hook_kind: hookKind,
        subject_type: 'ip',
        subject_value: await stableIpHookValue(env, '203.0.113.51'),
        outcome_code: 'trial_unavailable',
        outcome_class: 'security_fail',
        outcome_subcode: `${hookKind}_blocked`,
        reason: `${hookKind} abuse-control hook blocked this request`,
      });

      const response = await app.request(
        'http://broker.test/v1/trial/challenge',
        {
          method: 'POST',
          headers: {
            'content-type': 'application/json',
            'cf-connecting-ip': '203.0.113.51',
          },
          body: JSON.stringify({
            installation_id: `install-${hookKind}-blocked`,
            device_public_key: keyPair.devicePublicKey,
            app_version: '1.2.3',
          }),
        },
        env,
      );

      expect(response.status).toBe(503);
      await expect(response.json()).resolves.toEqual({
        error: {
          code: 'trial_unavailable',
          class: 'security_fail',
          subcode: null,
          retry_after_ms: null,
          message: `${hookKind} abuse-control hook blocked this request`,
        },
        managed_state: {
          lifecycle: 'none',
          managed_availability: true,
        },
        current_entitlement: null,
      });

      const installationCount = env.__db
        .prepare('SELECT COUNT(*) AS count FROM installations')
        .get() as { count: number };
      expect(installationCount.count).toBe(0);
    },
  );

  it('applies revocation hooks immediately and returns lifecycle data instead of a revoked public error code on status', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    const active = await activatePendingReleaseSession({
      env,
      installationId: 'install-revocation-hook',
      appVersion: '1.2.3',
      hardwareHash: 'hardware-hash-revocation-hook',
    });
    expect(active.response.status).toBe(200);

    insertSubjectHook(env, {
      hook_kind: 'revocation',
      subject_type: 'installation_id',
      subject_value: 'install-revocation-hook',
      outcome_code: 'trial_not_eligible',
      outcome_class: 'security_fail',
      outcome_subcode: 'revoked_by_hook',
      reason: 'manual revocation hook marked this installation as revoked',
    });

    const signedStatus = await signCanonicalStatusRequest(active.keyPair.privateKey, {
      installation_id: 'install-revocation-hook',
      timestamp: '2026-04-08T06:00:30.000Z',
    });
    const statusResponse = await getTrialStatus({
      env,
      installationId: 'install-revocation-hook',
      headers: {
        'X-Puripuly-Timestamp': signedStatus.timestamp,
        'X-Puripuly-Signature': signedStatus.signature,
      },
    });

    expect(statusResponse.status).toBe(200);
    await expect(statusResponse.json()).resolves.toEqual({
      managed_state: {
        lifecycle: 'revoked',
        managed_availability: false,
      },
      current_entitlement: {
        provider: 'OpenRouter',
        budget_usd: 0.07,
        issued_at: '2026-04-08T06:00:00.000Z',
        expires_at: '2026-07-08T06:00:00.000Z',
      },
      onboarding_eligibility: {
        eligible: false,
        reason: 'revoked',
        requires_discord_oauth: false,
      },
    });

    const entitlement = env.__db
      .prepare(
        `SELECT status, release_session_ref, release_token_hash, release_token_expires_at
           FROM openrouter_entitlements
          WHERE installation_id = ?`,
      )
      .get('install-revocation-hook') as Record<string, unknown>;
    expect(entitlement).toEqual({
      status: 'revoked',
      release_session_ref: null,
      release_token_hash: null,
      release_token_expires_at: null,
    });
  });

  it('returns fresh revoked lifecycle state on challenge responses after a revocation hook mutates the entitlement', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    const active = await activatePendingReleaseSession({
      env,
      installationId: 'install-revocation-fresh-state',
      appVersion: '1.2.3',
      hardwareHash: 'hardware-hash-revocation-fresh-state',
    });
    expect(active.response.status).toBe(200);

    insertSubjectHook(env, {
      hook_kind: 'revocation',
      subject_type: 'installation_id',
      subject_value: 'install-revocation-fresh-state',
      outcome_code: 'trial_unavailable',
      outcome_class: 'security_fail',
      outcome_subcode: 'revoked_by_hook',
      reason: 'manual revocation hook blocked challenge issuance',
    });

    const challengeResponse = await app.request(
      'http://broker.test/v1/trial/challenge',
      {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          'cf-connecting-ip': '203.0.113.61',
        },
        body: JSON.stringify({
          installation_id: 'install-revocation-fresh-state',
          device_public_key: active.keyPair.devicePublicKey,
          app_version: '1.2.4',
        }),
      },
      env,
    );

    expect(challengeResponse.status).toBe(503);
    await expect(challengeResponse.json()).resolves.toEqual({
      error: {
        code: 'trial_unavailable',
        class: 'security_fail',
        subcode: null,
        retry_after_ms: null,
        message: 'manual revocation hook blocked challenge issuance',
      },
      managed_state: {
        lifecycle: 'revoked',
        managed_availability: false,
      },
      current_entitlement: {
        provider: 'OpenRouter',
        budget_usd: 0.07,
        issued_at: '2026-04-08T06:00:00.000Z',
        expires_at: '2026-07-08T06:00:00.000Z',
      },
    });
  });
});
