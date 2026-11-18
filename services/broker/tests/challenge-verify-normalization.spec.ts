import { afterEach, describe, expect, it, vi } from 'vitest';

import app from '../src/index';
import { createDeviceKeyPair, signCanonicalVerifyRequest } from './test-support/ed25519';
import { createTestBrokerEnv, insertEntitlement } from './test-support/sqlite-d1';
import { issueChallenge, postVerify } from './test-support/trial-api';

interface NonVerifiableLifecycleCase {
  lifecycle: 'active' | 'expired' | 'revoked';
  budgetUsd: number;
  expiresAt: string;
}

describe('challenge and verify managed-state normalization', () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it('normalizes active entitlement metadata on challenge responses without leaking internal credential refs', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    const keyPair = await createDeviceKeyPair();
    await issueChallenge({
      env,
      installationId: 'install-active-state',
      devicePublicKey: keyPair.devicePublicKey,
      appVersion: '1.2.3',
    });
    insertEntitlement(env, {
      installation_id: 'install-active-state',
      status: 'active',
      budget_usd: 0.04,
      managed_credential_ref: 'internal-ref-should-not-leak',
      issued_at: '2026-04-01T00:00:00Z',
      expires_at: '2026-10-01T00:00:00Z',
    });

    const response = await app.request(
      'http://broker.test/v1/trial/challenge',
      {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          installation_id: 'install-active-state',
          device_public_key: keyPair.devicePublicKey,
          app_version: '1.2.4',
        }),
      },
      env,
    );

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({
      challenge: expect.any(String),
      challenge_expires_at: '2026-04-08T06:05:00.000Z',
      fingerprint_salt: {
        version: 7,
        salt: 'shared-server-fingerprint-salt',
      },
      managed_state: {
        lifecycle: 'active',
        managed_availability: true,
      },
      current_entitlement: {
        provider: 'OpenRouter',
        budget_usd: 0.04,
        issued_at: '2026-04-01T00:00:00Z',
        expires_at: '2026-10-01T00:00:00Z',
      },
    });
  });

  it('normalizes revoked entitlement state as unavailable on challenge responses', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    const keyPair = await createDeviceKeyPair();
    await issueChallenge({
      env,
      installationId: 'install-revoked-state',
      devicePublicKey: keyPair.devicePublicKey,
      appVersion: '1.2.3',
    });
    insertEntitlement(env, {
      installation_id: 'install-revoked-state',
      status: 'revoked',
      budget_usd: 0.07,
      managed_credential_ref: 'internal-ref-revoked',
      issued_at: '2026-04-01T00:00:00Z',
      expires_at: '2026-04-05T00:00:00Z',
    });

    const response = await app.request(
      'http://broker.test/v1/trial/challenge',
      {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          installation_id: 'install-revoked-state',
          device_public_key: keyPair.devicePublicKey,
          app_version: '1.2.4',
        }),
      },
      env,
    );

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({
      challenge: expect.any(String),
      challenge_expires_at: '2026-04-08T06:05:00.000Z',
      fingerprint_salt: {
        version: 7,
        salt: 'shared-server-fingerprint-salt',
      },
      managed_state: {
        lifecycle: 'revoked',
        managed_availability: false,
      },
      current_entitlement: {
        provider: 'OpenRouter',
        budget_usd: 0.07,
        issued_at: '2026-04-01T00:00:00Z',
        expires_at: '2026-04-05T00:00:00Z',
      },
    });
  });

  it('reuses pending_release metadata on verify responses while keeping internal storage fields hidden', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    const keyPair = await createDeviceKeyPair();
    const challenge = await issueChallenge({
      env,
      installationId: 'install-normalized-pending',
      devicePublicKey: keyPair.devicePublicKey,
      appVersion: '1.2.3',
    });
    insertEntitlement(env, {
      installation_id: 'install-normalized-pending',
      status: 'pending_release',
      budget_usd: 0.05,
      release_session_ref: 'existing-session',
      release_token_hash: 'existing-token-hash',
      release_token_expires_at: '2026-04-08T06:10:00Z',
    });

    const requestBody = await signCanonicalVerifyRequest(keyPair.privateKey, {
      installation_id: 'install-normalized-pending',
      device_public_key: keyPair.devicePublicKey,
      challenge: challenge.challenge,
      challenge_expires_at: challenge.challenge_expires_at,
      hardware_hash: 'hardware-hash-normalized',
      app_version: '1.2.4',
      signed_at: '2026-04-08T06:00:30.000Z',
    });

    const response = await postVerify(env, requestBody);
    expect(response.status).toBe(200);

    const payload = (await response.json()) as Record<string, unknown>;
    expect(payload).toMatchObject({
      managed_state: {
        lifecycle: 'pending_release',
        managed_availability: true,
      },
      current_entitlement: {
        provider: 'OpenRouter',
        budget_usd: 0.05,
        issued_at: null,
        expires_at: null,
      },
    });
    expect(payload).not.toHaveProperty('release_session_ref');
    expect(payload).not.toHaveProperty('release_token_hash');
    expect(payload).not.toHaveProperty('managed_credential_ref');
  });

  it.each([
    {
      lifecycle: 'active',
      budgetUsd: 0.04,
      expiresAt: '2026-10-01T00:00:00Z',
    },
    {
      lifecycle: 'expired',
      budgetUsd: 0.03,
      expiresAt: '2026-04-02T00:00:00Z',
    },
    {
      lifecycle: 'revoked',
      budgetUsd: 0.02,
      expiresAt: '2026-04-03T00:00:00Z',
    },
  ] satisfies NonVerifiableLifecycleCase[])(
    'preserves fingerprint state on challenge reissue for non-verifiable lifecycle $lifecycle',
    async (testCase: NonVerifiableLifecycleCase) => {
      const { lifecycle, budgetUsd, expiresAt } = testCase;
      vi.useFakeTimers();
      vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

      const env = createTestBrokerEnv();
      const keyPair = await createDeviceKeyPair();
      const initialChallenge = await issueChallenge({
        env,
        installationId: `install-preserve-${lifecycle}`,
        devicePublicKey: keyPair.devicePublicKey,
        appVersion: '1.2.3',
      });
      const verifyRequest = await signCanonicalVerifyRequest(keyPair.privateKey, {
        installation_id: `install-preserve-${lifecycle}`,
        device_public_key: keyPair.devicePublicKey,
        challenge: initialChallenge.challenge,
        challenge_expires_at: initialChallenge.challenge_expires_at,
        hardware_hash: `hardware-hash-${lifecycle}`,
        app_version: '1.2.3',
        signed_at: '2026-04-08T06:00:30.000Z',
      });

      const verifyResponse = await postVerify(env, verifyRequest);
      expect(verifyResponse.status).toBe(200);

      env.__db
        .prepare(
          `UPDATE openrouter_entitlements
              SET status = ?,
                  budget_usd = ?,
                  managed_credential_ref = ?,
                  issued_at = ?,
                  expires_at = ?,
                  release_session_ref = NULL,
                  release_token_hash = NULL,
                  release_token_expires_at = NULL
            WHERE installation_id = ?`,
        )
        .run(
          lifecycle,
          budgetUsd,
          `credential-${lifecycle}`,
          '2026-04-01T00:00:00Z',
          expiresAt,
          `install-preserve-${lifecycle}`,
        );

      vi.setSystemTime(new Date('2026-04-08T06:01:00Z'));

      const response = await app.request(
        'http://broker.test/v1/trial/challenge',
        {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({
            installation_id: `install-preserve-${lifecycle}`,
            device_public_key: keyPair.devicePublicKey,
            app_version: '1.2.4',
          }),
        },
        env,
      );

      expect(response.status).toBe(200);

      const installation = env.__db
        .prepare(
          `SELECT hardware_hash, hardware_hash_salt_version, challenge, challenge_expires_at,
                  challenge_salt_version, app_version
             FROM installations
            WHERE installation_id = ?`,
        )
        .get(`install-preserve-${lifecycle}`) as Record<string, unknown>;

      expect(installation).toEqual({
        hardware_hash: `hardware-hash-${lifecycle}`,
        hardware_hash_salt_version: 7,
        challenge: expect.any(String),
        challenge_expires_at: '2026-04-08T06:06:00.000Z',
        challenge_salt_version: 7,
        app_version: '1.2.4',
      });
    },
  );
});
