import { afterEach, describe, expect, it, vi } from 'vitest';

import app from '../src/index';
import {
  createDeviceKeyPair,
  signCanonicalStatusRequest,
} from './test-support/ed25519';
import { insertVelocityCapHook } from './test-support/abuse-controls';
import {
  createTestBrokerEnv,
  seedRequestEvent,
} from './test-support/sqlite-d1';
import { getTrialStatus, issueChallenge } from './test-support/trial-api';

describe('broker cross-endpoint velocity-cap hooks', () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it('blocks later requests across endpoints when a configured installation velocity cap is exceeded', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    const keyPair = await createDeviceKeyPair();
    await issueChallenge({
      env,
      installationId: 'install-velocity-cap',
      devicePublicKey: keyPair.devicePublicKey,
      appVersion: '1.2.3',
    });

    insertVelocityCapHook(env, {
      subject_type: 'installation_id',
      subject_value: 'install-velocity-cap',
      max_requests: 2,
      window_minutes: 15,
      outcome_code: 'trial_unavailable',
      outcome_class: 'retryable',
      outcome_subcode: 'velocity_capped',
      reason: 'manual abuse-defense velocity hook for review',
    });

    const signedStatus = await signCanonicalStatusRequest(keyPair.privateKey, {
      installation_id: 'install-velocity-cap',
      timestamp: '2026-04-08T06:00:30.000Z',
    });
    const firstResponse = await getTrialStatus({
      env,
      installationId: 'install-velocity-cap',
      headers: {
        'X-Puripuly-Timestamp': signedStatus.timestamp,
        'X-Puripuly-Signature': signedStatus.signature,
      },
    });
    expect(firstResponse.status).toBe(200);

    const blockedResponse = await app.request(
      'http://broker.test/v1/trial/challenge',
      {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          'cf-connecting-ip': '203.0.113.41',
        },
        body: JSON.stringify({
          installation_id: 'install-velocity-cap',
          device_public_key: keyPair.devicePublicKey,
          app_version: '1.2.4',
        }),
      },
      env,
    );

    expect(blockedResponse.status).toBe(503);
    await expect(blockedResponse.json()).resolves.toEqual({
      error: {
        code: 'trial_unavailable',
        class: 'retryable',
        subcode: null,
        retry_after_ms: 900000,
        message: 'manual abuse-defense velocity hook for review',
      },
      managed_state: {
        lifecycle: 'none',
        managed_availability: true,
      },
      current_entitlement: null,
    });
  });

  it('does not let a mismatched challenge request poison installation-scoped velocity state for an existing installation', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    const legitimateKeyPair = await createDeviceKeyPair();
    const spoofedKeyPair = await createDeviceKeyPair();

    env.__db
      .prepare(
        `INSERT INTO installations (
            installation_id,
            device_public_key,
            hardware_hash,
            hardware_hash_salt_version,
            app_version,
            challenge,
            challenge_expires_at,
            challenge_salt_version,
            created_at,
            last_seen_at
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(
        'install-challenge-velocity-target',
        legitimateKeyPair.devicePublicKey,
        null,
        null,
        '1.2.3',
        null,
        null,
        null,
        '2026-04-08T06:00:00.000Z',
        '2026-04-08T06:00:00.000Z',
      );

    insertVelocityCapHook(env, {
      subject_type: 'installation_id',
      subject_value: 'install-challenge-velocity-target',
      max_requests: 1,
      window_minutes: 15,
      outcome_code: 'trial_unavailable',
      outcome_class: 'retryable',
      outcome_subcode: 'velocity_capped',
      reason: 'manual abuse-defense velocity hook for review',
    });

    const spoofedResponse = await app.request(
      'http://broker.test/v1/trial/challenge',
      {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          'cf-connecting-ip': '203.0.113.42',
        },
        body: JSON.stringify({
          installation_id: 'install-challenge-velocity-target',
          device_public_key: spoofedKeyPair.devicePublicKey,
          app_version: '1.2.4',
        }),
      },
      env,
    );

    expect(spoofedResponse.status).toBe(409);

    const legitimateResponse = await app.request(
      'http://broker.test/v1/trial/challenge',
      {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          'cf-connecting-ip': '203.0.113.42',
        },
        body: JSON.stringify({
          installation_id: 'install-challenge-velocity-target',
          device_public_key: legitimateKeyPair.devicePublicKey,
          app_version: '1.2.4',
        }),
      },
      env,
    );

    expect(legitimateResponse.status).toBe(200);

    const requestEventCount = env.__db
      .prepare(
        `SELECT COUNT(*) AS count
           FROM broker_request_events
          WHERE endpoint = ?
            AND installation_id = ?`,
      )
      .get('POST /v1/trial/challenge', 'install-challenge-velocity-target') as {
      count: number;
    };
    expect(requestEventCount.count).toBe(1);
  });

  it('ignores synthetic verify outcome telemetry when evaluating installation-scoped velocity hooks', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    const keyPair = await createDeviceKeyPair();
    await issueChallenge({
      env,
      installationId: 'install-velocity-synthetic-outcomes',
      devicePublicKey: keyPair.devicePublicKey,
      appVersion: '1.2.3',
    });

    await seedRequestEvent(env, {
      endpoint: 'POST /v1/trial/challenge/verify/success',
      ip: '203.0.113.43',
      installationId: 'install-velocity-synthetic-outcomes',
      observedAt: '2026-04-08T06:00:10.000Z',
    });
    await seedRequestEvent(env, {
      endpoint: 'POST /v1/trial/challenge/verify/fail',
      ip: '203.0.113.43',
      installationId: 'install-velocity-synthetic-outcomes',
      observedAt: '2026-04-08T06:00:20.000Z',
    });

    insertVelocityCapHook(env, {
      subject_type: 'installation_id',
      subject_value: 'install-velocity-synthetic-outcomes',
      max_requests: 2,
      window_minutes: 15,
      outcome_code: 'trial_unavailable',
      outcome_class: 'retryable',
      outcome_subcode: 'velocity_capped',
      reason: 'manual abuse-defense velocity hook for review',
    });

    const signedStatus = await signCanonicalStatusRequest(keyPair.privateKey, {
      installation_id: 'install-velocity-synthetic-outcomes',
      timestamp: '2026-04-08T06:00:30.000Z',
    });
    const response = await getTrialStatus({
      env,
      installationId: 'install-velocity-synthetic-outcomes',
      headers: {
        'X-Puripuly-Timestamp': signedStatus.timestamp,
        'X-Puripuly-Signature': signedStatus.signature,
      },
    });

    expect(response.status).toBe(200);
  });
});
