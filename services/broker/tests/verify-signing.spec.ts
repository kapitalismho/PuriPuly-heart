import { afterEach, describe, expect, it, vi } from 'vitest';

import { createDeviceKeyPair, signCanonicalVerifyRequest, signNonCanonicalVerifyRequest } from './test-support/ed25519';
import { normalizedErrorEnvelope } from './test-support/errors';
import { createTestBrokerEnv } from './test-support/sqlite-d1';
import { issueChallenge, postVerify } from './test-support/trial-api';

describe('POST /v1/trial/challenge/verify signing contract', () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it('accepts a canonical Ed25519-signed request and returns a release token', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    const keyPair = await createDeviceKeyPair();
    const challenge = await issueChallenge({
      env,
      installationId: 'install-signed',
      devicePublicKey: keyPair.devicePublicKey,
      appVersion: '1.2.3',
    });

    const requestBody = await signCanonicalVerifyRequest(keyPair.privateKey, {
      installation_id: 'install-signed',
      device_public_key: keyPair.devicePublicKey,
      challenge: challenge.challenge,
      challenge_expires_at: challenge.challenge_expires_at,
      hardware_hash: 'hardware-hash-001',
      app_version: '1.2.4',
      signed_at: '2026-04-08T06:00:30.000Z',
    });

    const response = await postVerify(env, requestBody);

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({
      release_token: expect.stringMatching(/^[A-Za-z0-9_-]{43}$/),
      release_token_expires_at: '2026-04-08T06:15:00.000Z',
      managed_state: {
        lifecycle: 'pending_release',
        managed_availability: true,
      },
      current_entitlement: {
        provider: 'OpenRouter',
        budget_usd: 0.07,
        issued_at: null,
        expires_at: null,
      },
    });
  });

  it('rejects signatures that do not use the canonical newline-delimited payload order', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    const keyPair = await createDeviceKeyPair();
    const challenge = await issueChallenge({
      env,
      installationId: 'install-wrong-order',
      devicePublicKey: keyPair.devicePublicKey,
      appVersion: '1.2.3',
    });

    const requestBody = await signNonCanonicalVerifyRequest(keyPair.privateKey, {
      installation_id: 'install-wrong-order',
      device_public_key: keyPair.devicePublicKey,
      challenge: challenge.challenge,
      challenge_expires_at: challenge.challenge_expires_at,
      hardware_hash: 'hardware-hash-002',
      app_version: '1.2.3',
      signed_at: '2026-04-08T06:00:30.000Z',
    });

    const response = await postVerify(env, requestBody);

    expect(response.status).toBe(401);
    await expect(response.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'challenge_invalid',
        class: 'security_fail',
        subcode: 'signature_mismatch',
        message: 'signature verification failed for the registered device_public_key',
      }),
    );
  });

  it('rejects signatures outside the ±60 second skew window', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    const keyPair = await createDeviceKeyPair();
    const challenge = await issueChallenge({
      env,
      installationId: 'install-skew',
      devicePublicKey: keyPair.devicePublicKey,
      appVersion: '1.2.3',
    });

    const requestBody = await signCanonicalVerifyRequest(keyPair.privateKey, {
      installation_id: 'install-skew',
      device_public_key: keyPair.devicePublicKey,
      challenge: challenge.challenge,
      challenge_expires_at: challenge.challenge_expires_at,
      hardware_hash: 'hardware-hash-003',
      app_version: '1.2.3',
      signed_at: '2026-04-08T06:01:01.000Z',
    });

    const response = await postVerify(env, requestBody);

    expect(response.status).toBe(401);
    await expect(response.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'challenge_invalid',
        class: 'security_fail',
        subcode: 'timestamp_skew',
        message: 'signed_at must be within ±60 seconds of broker time',
      }),
    );
  });
});
