import { afterEach, describe, expect, it, vi } from 'vitest';

import app from '../src/index';
import {
  createDeviceKeyPair,
  signCanonicalIssueRequest,
  signCanonicalStatusRequest,
  signCanonicalVerifyRequest,
} from './test-support/ed25519';
import {
  createPendingReleaseSession,
  mockOpenRouterManagementApi,
} from './test-support/openrouter-issue';
import { createTestBrokerEnv } from './test-support/sqlite-d1';
import { issueChallenge, getTrialStatus, postIssue, postVerify } from './test-support/trial-api';
import {
  insertVelocityCapHook,
  updateAbuseControls,
} from './test-support/abuse-controls';
import { normalizedErrorEnvelope } from './test-support/errors';

describe('broker abuse-control rate limiting', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it('rate limits trial challenge by client IP using the runtime-configured threshold', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    updateAbuseControls(env, (controls) => {
      controls.trialChallenge.maxRequests = 2;
    });

    for (const suffix of ['one', 'two']) {
      const keyPair = await createDeviceKeyPair();
      const response = await app.request(
        'http://broker.test/v1/trial/challenge',
        {
          method: 'POST',
          headers: {
            'content-type': 'application/json',
            'cf-connecting-ip': '203.0.113.10',
          },
          body: JSON.stringify({
            installation_id: `install-rate-limit-${suffix}`,
            device_public_key: keyPair.devicePublicKey,
            app_version: '1.2.3',
          }),
        },
        env,
      );

      expect(response.status).toBe(200);
    }

    const blockedKeyPair = await createDeviceKeyPair();
    const blockedResponse = await app.request(
      'http://broker.test/v1/trial/challenge',
      {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          'cf-connecting-ip': '203.0.113.10',
        },
        body: JSON.stringify({
          installation_id: 'install-rate-limit-three',
          device_public_key: blockedKeyPair.devicePublicKey,
          app_version: '1.2.3',
        }),
      },
      env,
    );

    expect(blockedResponse.status).toBe(429);
    await expect(blockedResponse.json()).resolves.toEqual({
      error: {
        code: 'rate_limited',
        class: 'retryable',
        subcode: 'ip_rate_limited',
        retry_after_ms: 900000,
        message: 'request rate limit exceeded for POST /v1/trial/challenge',
      },
      managed_state: {
        lifecycle: 'none',
        managed_availability: true,
      },
      current_entitlement: null,
    });
  });

  it('rate limits verify by installation_id using the runtime-configured threshold', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    updateAbuseControls(env, (controls) => {
      controls.trialChallengeVerify.maxRequests = 1;
    });

    const keyPair = await createDeviceKeyPair();
    const firstChallenge = await issueChallenge({
      env,
      installationId: 'install-rate-limit-verify',
      devicePublicKey: keyPair.devicePublicKey,
      appVersion: '1.2.3',
    });
    const firstVerify = await signCanonicalVerifyRequest(keyPair.privateKey, {
      installation_id: 'install-rate-limit-verify',
      device_public_key: keyPair.devicePublicKey,
      challenge: firstChallenge.challenge,
      challenge_expires_at: firstChallenge.challenge_expires_at,
      hardware_hash: 'hardware-hash-rate-limit-verify',
      app_version: '1.2.3',
      signed_at: '2026-04-08T06:00:30.000Z',
    });

    expect((await postVerify(env, firstVerify)).status).toBe(200);

    const secondChallenge = await issueChallenge({
      env,
      installationId: 'install-rate-limit-verify',
      devicePublicKey: keyPair.devicePublicKey,
      appVersion: '1.2.4',
    });
    const secondVerify = await signCanonicalVerifyRequest(keyPair.privateKey, {
      installation_id: 'install-rate-limit-verify',
      device_public_key: keyPair.devicePublicKey,
      challenge: secondChallenge.challenge,
      challenge_expires_at: secondChallenge.challenge_expires_at,
      hardware_hash: 'hardware-hash-rate-limit-verify',
      app_version: '1.2.4',
      signed_at: '2026-04-08T06:00:45.000Z',
    });
    const blockedResponse = await postVerify(env, secondVerify);

    expect(blockedResponse.status).toBe(429);
    await expect(blockedResponse.json()).resolves.toEqual({
      error: {
        code: 'rate_limited',
        class: 'retryable',
        subcode: 'installation_rate_limited',
        retry_after_ms: 900000,
        message: 'request rate limit exceeded for POST /v1/trial/challenge/verify',
      },
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

  it('prefers active-reissue rejection over issue endpoint rate limits', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    mockOpenRouterManagementApi();
    const env = createTestBrokerEnv();
    updateAbuseControls(env, (controls) => {
      controls.openrouterIssue.maxRequests = 1;
    });

    const release = await createPendingReleaseSession({
      env,
      installationId: 'install-rate-limit-issue',
      appVersion: '1.2.3',
      hardwareHash: 'hardware-hash-rate-limit-issue',
    });
    const requestBody = await signCanonicalIssueRequest(release.keyPair.privateKey, {
      installation_id: 'install-rate-limit-issue',
      device_public_key: release.keyPair.devicePublicKey,
      release_token: release.releaseToken,
      hardware_hash: release.hardwareHash,
      reason: 'llm_start',
      budget_usd: 0.07,
      model: 'google/gemma-4-26b-a4b-it',
      signed_at: '2026-04-08T06:00:45.000Z',
    });

    expect((await postIssue(env, requestBody)).status).toBe(200);

    const blockedResponse = await postIssue(env, requestBody);

    expect(blockedResponse.status).toBe(409);
    await expect(blockedResponse.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'trial_not_eligible',
        class: 'terminal',
        subcode: 'managed_key_unrecoverable',
        message: 'managed key was already issued and cannot be recovered',
        managedState: {
          lifecycle: 'active',
          managed_availability: true,
        },
        currentEntitlement: {
          provider: 'OpenRouter',
          budget_usd: 0.07,
          issued_at: '2026-04-08T06:00:00.000Z',
          expires_at: '2026-07-08T06:00:00.000Z',
        },
      }),
    );
  });

  it('prefers hardware snapshot mismatch rejection over velocity-cap exits', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    mockOpenRouterManagementApi();
    const env = createTestBrokerEnv();
    const release = await createPendingReleaseSession({
      env,
      installationId: 'install-issue-velocity-hardware-mismatch',
      appVersion: '1.2.3',
      hardwareHash: 'hardware-hash-verified-velocity',
    });
    insertVelocityCapHook(env, {
      subject_type: 'installation_id',
      subject_value: 'install-issue-velocity-hardware-mismatch',
      max_requests: 1,
      window_minutes: 15,
      outcome_code: 'trial_unavailable',
      outcome_class: 'retryable',
      outcome_subcode: 'velocity_capped',
      reason: 'manual issue velocity hook',
    });

    const response = await postIssue(
      env,
      await signCanonicalIssueRequest(release.keyPair.privateKey, {
        installation_id: 'install-issue-velocity-hardware-mismatch',
        device_public_key: release.keyPair.devicePublicKey,
        release_token: release.releaseToken,
        hardware_hash: 'hardware-hash-different-velocity',
        reason: 'llm_start',
        budget_usd: 0.07,
        model: 'google/gemma-4-26b-a4b-it',
        signed_at: '2026-04-08T06:00:45.000Z',
      }),
    );

    expect(response.status).toBe(409);
    await expect(response.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'trial_not_eligible',
        class: 'terminal',
        subcode: 'hardware_duplicate',
        message: 'hardware_hash no longer matches the verified release session',
        managedState: {
          lifecycle: 'pending_release',
          managed_availability: true,
        },
        currentEntitlement: {
          provider: 'OpenRouter',
          budget_usd: 0.07,
          issued_at: null,
          expires_at: null,
        },
      }),
    );
  });

  it('rate limits status by installation_id using the runtime-configured threshold', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    updateAbuseControls(env, (controls) => {
      controls.trialStatus.maxRequests = 1;
    });

    const keyPair = await createDeviceKeyPair();
    await issueChallenge({
      env,
      installationId: 'install-rate-limit-status',
      devicePublicKey: keyPair.devicePublicKey,
      appVersion: '1.2.3',
    });
    const signedRequest = await signCanonicalStatusRequest(keyPair.privateKey, {
      installation_id: 'install-rate-limit-status',
      timestamp: '2026-04-08T06:00:30.000Z',
    });

    expect(
      (
        await getTrialStatus({
          env,
          installationId: 'install-rate-limit-status',
          headers: {
            'X-Puripuly-Timestamp': signedRequest.timestamp,
            'X-Puripuly-Signature': signedRequest.signature,
          },
        })
      ).status,
    ).toBe(200);

    const blockedResponse = await getTrialStatus({
      env,
      installationId: 'install-rate-limit-status',
      headers: {
        'X-Puripuly-Timestamp': signedRequest.timestamp,
        'X-Puripuly-Signature': signedRequest.signature,
      },
    });

    expect(blockedResponse.status).toBe(429);
    await expect(blockedResponse.json()).resolves.toEqual({
      error: {
        code: 'rate_limited',
        class: 'retryable',
        subcode: 'installation_rate_limited',
        retry_after_ms: 900000,
        message: 'request rate limit exceeded for GET /v1/trial/status',
      },
      managed_state: {
        lifecycle: 'none',
        managed_availability: true,
      },
      current_entitlement: null,
    });
  });

  it('does not let unauthenticated verify requests burn another installation_id quota', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    updateAbuseControls(env, (controls) => {
      controls.trialChallengeVerify.maxRequests = 1;
    });

    const legitimateKeyPair = await createDeviceKeyPair();
    const spoofedKeyPair = await createDeviceKeyPair();
    const challenge = await issueChallenge({
      env,
      installationId: 'install-verify-poisoning-target',
      devicePublicKey: legitimateKeyPair.devicePublicKey,
      appVersion: '1.2.3',
    });

    const spoofedVerify = await signCanonicalVerifyRequest(spoofedKeyPair.privateKey, {
      installation_id: 'install-verify-poisoning-target',
      device_public_key: spoofedKeyPair.devicePublicKey,
      challenge: challenge.challenge,
      challenge_expires_at: challenge.challenge_expires_at,
      hardware_hash: 'hardware-hash-verify-poisoning-target',
      app_version: '1.2.3',
      signed_at: '2026-04-08T06:00:15.000Z',
    });
    const spoofedResponse = await postVerify(env, spoofedVerify);
    expect(spoofedResponse.status).toBe(409);

    const legitimateVerify = await signCanonicalVerifyRequest(
      legitimateKeyPair.privateKey,
      {
        installation_id: 'install-verify-poisoning-target',
        device_public_key: legitimateKeyPair.devicePublicKey,
        challenge: challenge.challenge,
        challenge_expires_at: challenge.challenge_expires_at,
        hardware_hash: 'hardware-hash-verify-poisoning-target',
        app_version: '1.2.3',
        signed_at: '2026-04-08T06:00:30.000Z',
      },
    );
    const legitimateResponse = await postVerify(env, legitimateVerify);

    expect(legitimateResponse.status).toBe(200);

    const requestEventCount = env.__db
      .prepare(
        `SELECT COUNT(*) AS count
           FROM broker_request_events
          WHERE endpoint = ?
            AND installation_id = ?`,
      )
      .get(
        'POST /v1/trial/challenge/verify',
        'install-verify-poisoning-target',
      ) as { count: number };
    expect(requestEventCount.count).toBe(1);
  });

  it('does not let unauthenticated issue requests burn another installation_id quota', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    mockOpenRouterManagementApi();
    const env = createTestBrokerEnv();
    updateAbuseControls(env, (controls) => {
      controls.openrouterIssue.maxRequests = 1;
    });

    const release = await createPendingReleaseSession({
      env,
      installationId: 'install-issue-poisoning-target',
      appVersion: '1.2.3',
      hardwareHash: 'hardware-hash-issue-poisoning-target',
    });
    const spoofedKeyPair = await createDeviceKeyPair();
    const spoofedRequest = await signCanonicalIssueRequest(spoofedKeyPair.privateKey, {
      installation_id: 'install-issue-poisoning-target',
      device_public_key: spoofedKeyPair.devicePublicKey,
      release_token: release.releaseToken,
      hardware_hash: release.hardwareHash,
      reason: 'llm_start',
      budget_usd: 0.07,
      model: 'google/gemma-4-26b-a4b-it',
      signed_at: '2026-04-08T06:00:30.000Z',
    });
    const spoofedResponse = await postIssue(env, spoofedRequest);
    expect(spoofedResponse.status).toBe(409);

    const legitimateRequest = await signCanonicalIssueRequest(release.keyPair.privateKey, {
      installation_id: 'install-issue-poisoning-target',
      device_public_key: release.keyPair.devicePublicKey,
      release_token: release.releaseToken,
      hardware_hash: release.hardwareHash,
      reason: 'llm_start',
      budget_usd: 0.07,
      model: 'google/gemma-4-26b-a4b-it',
      signed_at: '2026-04-08T06:00:45.000Z',
    });
    const legitimateResponse = await postIssue(env, legitimateRequest);

    expect(legitimateResponse.status).toBe(200);

    const requestEventCount = env.__db
      .prepare(
        `SELECT COUNT(*) AS count
           FROM broker_request_events
          WHERE endpoint = ?
            AND installation_id = ?`,
      )
      .get(
        'POST /v1/providers/openrouter/issue',
        'install-issue-poisoning-target',
      ) as { count: number };
    expect(requestEventCount.count).toBe(1);
  });

  it('does not let unauthenticated status requests burn another installation_id quota', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    updateAbuseControls(env, (controls) => {
      controls.trialStatus.maxRequests = 1;
    });

    const legitimateKeyPair = await createDeviceKeyPair();
    const spoofedKeyPair = await createDeviceKeyPair();
    await issueChallenge({
      env,
      installationId: 'install-status-poisoning-target',
      devicePublicKey: legitimateKeyPair.devicePublicKey,
      appVersion: '1.2.3',
    });

    const spoofedRequest = await signCanonicalStatusRequest(spoofedKeyPair.privateKey, {
      installation_id: 'install-status-poisoning-target',
      timestamp: '2026-04-08T06:00:15.000Z',
    });
    const spoofedResponse = await getTrialStatus({
      env,
      installationId: 'install-status-poisoning-target',
      headers: {
        'X-Puripuly-Timestamp': spoofedRequest.timestamp,
        'X-Puripuly-Signature': spoofedRequest.signature,
      },
    });
    expect(spoofedResponse.status).toBe(401);

    const legitimateRequest = await signCanonicalStatusRequest(
      legitimateKeyPair.privateKey,
      {
        installation_id: 'install-status-poisoning-target',
        timestamp: '2026-04-08T06:00:30.000Z',
      },
    );
    const legitimateResponse = await getTrialStatus({
      env,
      installationId: 'install-status-poisoning-target',
      headers: {
        'X-Puripuly-Timestamp': legitimateRequest.timestamp,
        'X-Puripuly-Signature': legitimateRequest.signature,
      },
    });

    expect(legitimateResponse.status).toBe(200);

    const requestEventCount = env.__db
      .prepare(
        `SELECT COUNT(*) AS count
           FROM broker_request_events
          WHERE endpoint = ?
            AND installation_id = ?`,
      )
      .get('GET /v1/trial/status', 'install-status-poisoning-target') as {
      count: number;
    };
    expect(requestEventCount.count).toBe(1);
  });
});
