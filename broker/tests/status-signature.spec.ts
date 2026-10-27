import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  createDeviceKeyPair,
  signCanonicalStatusRequest,
  signNonCanonicalStatusRequest,
} from './test-support/ed25519';
import { normalizedErrorEnvelope } from './test-support/errors';
import { createTestBrokerEnv } from './test-support/sqlite-d1';
import { getTrialStatus, issueChallenge } from './test-support/trial-api';

describe('GET /v1/trial/status signing contract', () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it('rejects signatures that do not use the canonical installation_id-then-timestamp payload order', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    const keyPair = await createDeviceKeyPair();
    await issueChallenge({
      env,
      installationId: 'install-status-wrong-order',
      devicePublicKey: keyPair.devicePublicKey,
      appVersion: '1.2.3',
    });
    const signedRequest = await signNonCanonicalStatusRequest(keyPair.privateKey, {
      installation_id: 'install-status-wrong-order',
      timestamp: '2026-04-08T06:00:30.000Z',
    });

    const response = await getTrialStatus({
      env,
      installationId: 'install-status-wrong-order',
      headers: {
        'X-Puripuly-Timestamp': signedRequest.timestamp,
        'X-Puripuly-Signature': signedRequest.signature,
      },
    });

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

  it('rejects status headers outside the ±60 second skew window', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    const keyPair = await createDeviceKeyPair();
    await issueChallenge({
      env,
      installationId: 'install-status-skew',
      devicePublicKey: keyPair.devicePublicKey,
      appVersion: '1.2.3',
    });
    const signedRequest = await signCanonicalStatusRequest(keyPair.privateKey, {
      installation_id: 'install-status-skew',
      timestamp: '2026-04-08T06:01:01.000Z',
    });

    const response = await getTrialStatus({
      env,
      installationId: 'install-status-skew',
      headers: {
        'X-Puripuly-Timestamp': signedRequest.timestamp,
        'X-Puripuly-Signature': signedRequest.signature,
      },
    });

    expect(response.status).toBe(401);
    await expect(response.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'challenge_invalid',
        class: 'security_fail',
        subcode: 'timestamp_skew',
        message: 'X-Puripuly-Timestamp must be within ±60 seconds of broker time',
      }),
    );
  });

  it('requires X-Puripuly-Signature to transport a base64url Ed25519 signature', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    const keyPair = await createDeviceKeyPair();
    await issueChallenge({
      env,
      installationId: 'install-status-bad-signature',
      devicePublicKey: keyPair.devicePublicKey,
      appVersion: '1.2.3',
    });

    const response = await getTrialStatus({
      env,
      installationId: 'install-status-bad-signature',
      headers: {
        'X-Puripuly-Timestamp': '2026-04-08T06:00:30.000Z',
        'X-Puripuly-Signature': 'not-base64url!!',
      },
    });

    expect(response.status).toBe(400);
    await expect(response.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'invalid_request',
        class: 'terminal',
        message: 'X-Puripuly-Signature must be a base64url-encoded Ed25519 signature',
      }),
    );
  });
});
