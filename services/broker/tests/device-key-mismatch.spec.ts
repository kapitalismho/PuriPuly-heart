import { afterEach, describe, expect, it, vi } from 'vitest';

import app from '../src/index';
import { createDeviceKeyPair, signCanonicalVerifyRequest } from './test-support/ed25519';
import { normalizedErrorEnvelope } from './test-support/errors';
import { createTestBrokerEnv } from './test-support/sqlite-d1';
import { issueChallenge, postVerify } from './test-support/trial-api';

describe('device_public_key binding protection', () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it('rejects challenge requests that reuse an existing device_public_key under a different installation_id', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    const keyPair = await createDeviceKeyPair();
    await issueChallenge({
      env,
      installationId: 'install-a',
      devicePublicKey: keyPair.devicePublicKey,
      appVersion: '1.2.3',
    });

    const response = await app.request(
      'http://broker.test/v1/trial/challenge',
      {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          installation_id: 'install-b',
          device_public_key: keyPair.devicePublicKey,
          app_version: '1.2.3',
        }),
      },
      env,
    );

    expect(response.status).toBe(409);
    await expect(response.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'trial_not_eligible',
        class: 'security_fail',
        subcode: 'device_public_key_registered',
        message:
          'device_public_key is already registered to a different installation_id',
      }),
    );

    const installationCount = env.__db
      .prepare('SELECT COUNT(*) AS count FROM installations')
      .get() as { count: number };
    expect(installationCount.count).toBe(1);
  });

  it('rejects verify requests that try to rebind installation identity to a different device_public_key', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    const registeredKeyPair = await createDeviceKeyPair();
    const mismatchedKeyPair = await createDeviceKeyPair();
    const challenge = await issueChallenge({
      env,
      installationId: 'install-registered',
      devicePublicKey: registeredKeyPair.devicePublicKey,
      appVersion: '1.2.3',
    });

    const requestBody = await signCanonicalVerifyRequest(
      mismatchedKeyPair.privateKey,
      {
        installation_id: 'install-registered',
        device_public_key: mismatchedKeyPair.devicePublicKey,
        challenge: challenge.challenge,
        challenge_expires_at: challenge.challenge_expires_at,
        hardware_hash: 'hardware-hash-mismatch',
        app_version: '1.2.3',
        signed_at: '2026-04-08T06:00:30.000Z',
      },
    );

    const response = await postVerify(env, requestBody);
    expect(response.status).toBe(409);
    await expect(response.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'trial_not_eligible',
        class: 'security_fail',
        subcode: 'installation_binding_mismatch',
        message: 'verify must use the registered device_public_key for installation_id',
      }),
    );

    const installation = env.__db
      .prepare(
        'SELECT device_public_key, challenge, challenge_expires_at FROM installations WHERE installation_id = ?',
      )
      .get('install-registered') as Record<string, unknown>;

    expect(installation).toEqual({
      device_public_key: registeredKeyPair.devicePublicKey,
      challenge: challenge.challenge,
      challenge_expires_at: challenge.challenge_expires_at,
    });
  });
});
