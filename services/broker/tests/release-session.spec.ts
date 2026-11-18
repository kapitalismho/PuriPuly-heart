import { afterEach, describe, expect, it, vi } from 'vitest';

import { createDeviceKeyPair, signCanonicalVerifyRequest } from './test-support/ed25519';
import { normalizedErrorEnvelope } from './test-support/errors';
import { sha256Base64Url } from './test-support/hash';
import { createTestBrokerEnv, insertEntitlement } from './test-support/sqlite-d1';
import { issueChallenge, postVerify } from './test-support/trial-api';

interface NonPendingLifecycleCase {
  lifecycle: 'active' | 'expired' | 'revoked';
  managedAvailability: boolean;
  issuedAt: string;
  expiresAt: string;
}

describe('release-session handshake state', () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it('rotates token and release session state for an existing pending_release row without creating extra rows', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    const keyPair = await createDeviceKeyPair();
    const firstChallenge = await issueChallenge({
      env,
      installationId: 'install-rotate',
      devicePublicKey: keyPair.devicePublicKey,
      appVersion: '1.2.3',
    });
    const firstVerify = await signCanonicalVerifyRequest(keyPair.privateKey, {
      installation_id: 'install-rotate',
      device_public_key: keyPair.devicePublicKey,
      challenge: firstChallenge.challenge,
      challenge_expires_at: firstChallenge.challenge_expires_at,
      hardware_hash: 'hardware-hash-rotate-001',
      app_version: '1.2.3',
      signed_at: '2026-04-08T06:00:30.000Z',
    });

    const firstResponse = await postVerify(env, firstVerify);
    const firstPayload = (await firstResponse.json()) as {
      release_token: string;
    };

    const firstRow = env.__db
      .prepare(
        `SELECT status, budget_usd, release_session_ref, release_token_hash,
                release_token_expires_at
           FROM openrouter_entitlements
          WHERE installation_id = ?`,
      )
      .get('install-rotate') as Record<string, unknown>;

    vi.setSystemTime(new Date('2026-04-08T06:01:00Z'));

    const secondChallenge = await issueChallenge({
      env,
      installationId: 'install-rotate',
      devicePublicKey: keyPair.devicePublicKey,
      appVersion: '1.2.4',
    });
    const secondVerify = await signCanonicalVerifyRequest(keyPair.privateKey, {
      installation_id: 'install-rotate',
      device_public_key: keyPair.devicePublicKey,
      challenge: secondChallenge.challenge,
      challenge_expires_at: secondChallenge.challenge_expires_at,
      hardware_hash: 'hardware-hash-rotate-002',
      app_version: '1.2.4',
      signed_at: '2026-04-08T06:01:30.000Z',
    });

    const secondResponse = await postVerify(env, secondVerify);
    expect(secondResponse.status).toBe(200);

    const secondPayload = (await secondResponse.json()) as {
      release_token: string;
      release_token_expires_at: string;
    };

    const secondRow = env.__db
      .prepare(
        `SELECT status, budget_usd, release_session_ref, release_token_hash,
                release_token_expires_at
           FROM openrouter_entitlements
          WHERE installation_id = ?`,
      )
      .get('install-rotate') as Record<string, unknown>;
    const count = env.__db
      .prepare('SELECT COUNT(*) AS count FROM openrouter_entitlements WHERE installation_id = ?')
      .get('install-rotate') as { count: number };

    expect(count.count).toBe(1);
    expect(firstRow.status).toBe('pending_release');
    expect(secondRow.status).toBe('pending_release');
    expect(firstRow.budget_usd).toBe(0.07);
    expect(secondRow.budget_usd).toBe(0.07);
    expect(secondRow.release_session_ref).not.toBe(firstRow.release_session_ref);
    expect(secondRow.release_token_hash).not.toBe(firstRow.release_token_hash);
    expect(secondRow.release_token_expires_at).toBe('2026-04-08T06:16:00.000Z');
    await expect(sha256Base64Url(firstPayload.release_token)).resolves.toBe(
      firstRow.release_token_hash,
    );
    await expect(sha256Base64Url(secondPayload.release_token)).resolves.toBe(
      secondRow.release_token_hash,
    );
    expect(secondPayload.release_token).not.toBe(firstPayload.release_token);
    expect(secondPayload.release_token_expires_at).toBe(
      '2026-04-08T06:16:00.000Z',
    );
  });

  it('stores only hashed release tokens and never raw managed credentials in release-session state', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    const keyPair = await createDeviceKeyPair();
    const challenge = await issueChallenge({
      env,
      installationId: 'install-hashed-token',
      devicePublicKey: keyPair.devicePublicKey,
      appVersion: '1.2.3',
    });
    const requestBody = await signCanonicalVerifyRequest(keyPair.privateKey, {
      installation_id: 'install-hashed-token',
      device_public_key: keyPair.devicePublicKey,
      challenge: challenge.challenge,
      challenge_expires_at: challenge.challenge_expires_at,
      hardware_hash: 'hardware-hash-release-003',
      app_version: '1.2.3',
      signed_at: '2026-04-08T06:00:20.000Z',
    });

    const response = await postVerify(env, requestBody);
    const payload = (await response.json()) as {
      release_token: string;
    };

    const row = env.__db
      .prepare(
        `SELECT managed_credential_ref, release_token_hash
           FROM openrouter_entitlements
          WHERE installation_id = ?`,
      )
      .get('install-hashed-token') as Record<string, unknown>;

    expect(row.managed_credential_ref).toBeNull();
    expect(row.release_token_hash).not.toBe(payload.release_token);
    await expect(sha256Base64Url(payload.release_token)).resolves.toBe(
      row.release_token_hash,
    );
  });

  it.each([
    {
      lifecycle: 'active',
      managedAvailability: true,
      issuedAt: '2026-04-01T00:00:00Z',
      expiresAt: '2026-10-01T00:00:00Z',
    },
    {
      lifecycle: 'expired',
      managedAvailability: false,
      issuedAt: '2026-04-01T00:00:00Z',
      expiresAt: '2026-04-05T00:00:00Z',
    },
    {
      lifecycle: 'revoked',
      managedAvailability: false,
      issuedAt: '2026-04-01T00:00:00Z',
      expiresAt: '2026-04-06T00:00:00Z',
    },
  ] satisfies NonPendingLifecycleCase[])(
    'does not mint a fresh release token when lifecycle is $lifecycle',
    async (testCase: NonPendingLifecycleCase) => {
      const { lifecycle, managedAvailability, issuedAt, expiresAt } = testCase;
      vi.useFakeTimers();
      vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

      const env = createTestBrokerEnv();
      const keyPair = await createDeviceKeyPair();
      const challenge = await issueChallenge({
        env,
        installationId: `install-${lifecycle}`,
        devicePublicKey: keyPair.devicePublicKey,
        appVersion: '1.2.3',
      });

      insertEntitlement(env, {
        installation_id: `install-${lifecycle}`,
        status: lifecycle,
        budget_usd: 0.05,
        managed_credential_ref: `internal-${lifecycle}-credential`,
        issued_at: issuedAt,
        expires_at: expiresAt,
      });

      const requestBody = await signCanonicalVerifyRequest(keyPair.privateKey, {
        installation_id: `install-${lifecycle}`,
        device_public_key: keyPair.devicePublicKey,
        challenge: challenge.challenge,
        challenge_expires_at: challenge.challenge_expires_at,
        hardware_hash: `hardware-hash-${lifecycle}`,
        app_version: '1.2.4',
        signed_at: '2026-04-08T06:00:30.000Z',
      });

      const response = await postVerify(env, requestBody);

      expect(response.status).toBe(409);
      await expect(response.json()).resolves.toEqual(
        normalizedErrorEnvelope({
          code: 'trial_not_eligible',
          class: 'terminal',
          subcode: 'lifecycle_not_eligible',
          message:
            'verify may only mint release_token for lifecycle none or pending_release',
          managedState: {
            lifecycle,
            managed_availability: managedAvailability,
          },
          currentEntitlement: {
            provider: 'OpenRouter',
            budget_usd: 0.05,
            issued_at: issuedAt,
            expires_at: expiresAt,
          },
        }),
      );

      const installation = env.__db
        .prepare(
          `SELECT challenge, challenge_expires_at, hardware_hash
             FROM installations
            WHERE installation_id = ?`,
        )
        .get(`install-${lifecycle}`) as Record<string, unknown>;

      expect(installation).toEqual({
        challenge: challenge.challenge,
        challenge_expires_at: challenge.challenge_expires_at,
        hardware_hash: null,
      });

      const entitlement = env.__db
        .prepare(
          `SELECT status, managed_credential_ref, release_session_ref, release_token_hash,
                  release_token_expires_at
             FROM openrouter_entitlements
            WHERE installation_id = ?`,
        )
        .get(`install-${lifecycle}`) as Record<string, unknown>;

      expect(entitlement).toEqual({
        status: lifecycle,
        managed_credential_ref: `internal-${lifecycle}-credential`,
        release_session_ref: null,
        release_token_hash: null,
        release_token_expires_at: null,
      });
    },
  );
});
