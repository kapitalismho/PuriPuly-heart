import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  signCanonicalIssueRequest,
  signNonCanonicalIssueRequest,
} from './test-support/ed25519';
import { normalizedErrorEnvelope } from './test-support/errors';
import {
  createPendingReleaseSession,
} from './test-support/openrouter-issue';
import { createTestBrokerEnv } from './test-support/sqlite-d1';
import { postIssue } from './test-support/trial-api';

describe('POST /v1/providers/openrouter/issue signing contract', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it('requires hardware_hash in the issue request body', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    const release = await createPendingReleaseSession({
      env,
      installationId: 'install-issue-missing-hardware-hash',
      appVersion: '1.2.3',
      hardwareHash: 'hardware-hash-issue-missing-hardware-hash',
    });
    const signedRequest = await signCanonicalIssueRequest(release.keyPair.privateKey, {
      installation_id: 'install-issue-missing-hardware-hash',
      device_public_key: release.keyPair.devicePublicKey,
      release_token: release.releaseToken,
      hardware_hash: release.hardwareHash,
      reason: 'llm_start',
      budget_usd: 0.07,
      model: 'google/gemma-4-26b-a4b-it',
      signed_at: '2026-04-08T06:00:45.000Z',
    });
    const { hardware_hash: _hardwareHash, ...requestBody } = signedRequest;

    const response = await postIssue(env, requestBody);

    expect(response.status).toBe(400);
    await expect(response.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'invalid_request',
        class: 'terminal',
        message:
          'installation_id, device_public_key, release_token, hardware_hash, reason, budget_usd, model, signed_at, and signature are required',
      }),
    );
  });

  it('binds hardware_hash into the canonical issue signature payload', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    const release = await createPendingReleaseSession({
      env,
      installationId: 'install-issue-hardware-signed',
      appVersion: '1.2.3',
      hardwareHash: 'hardware-hash-issue-hardware-signed',
    });
    const requestBody = await signCanonicalIssueRequest(release.keyPair.privateKey, {
      installation_id: 'install-issue-hardware-signed',
      device_public_key: release.keyPair.devicePublicKey,
      release_token: release.releaseToken,
      hardware_hash: release.hardwareHash,
      reason: 'llm_start',
      budget_usd: 0.07,
      model: 'google/gemma-4-26b-a4b-it',
      signed_at: '2026-04-08T06:00:45.000Z',
    });

    const response = await postIssue(env, {
      ...requestBody,
      hardware_hash: 'hardware-hash-issue-hardware-signed-tampered',
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

  it('rejects signatures that do not use the canonical newline-delimited payload order', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    const release = await createPendingReleaseSession({
      env,
      installationId: 'install-issue-wrong-order',
      appVersion: '1.2.3',
      hardwareHash: 'hardware-hash-issue-wrong-order',
    });
    const requestBody = await signNonCanonicalIssueRequest(release.keyPair.privateKey, {
      installation_id: 'install-issue-wrong-order',
      device_public_key: release.keyPair.devicePublicKey,
      release_token: release.releaseToken,
      hardware_hash: release.hardwareHash,
      reason: 'llm_start',
      budget_usd: 0.07,
      model: 'google/gemma-4-26b-a4b-it',
      signed_at: '2026-04-08T06:00:45.000Z',
    });

    const response = await postIssue(env, requestBody);

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
    const release = await createPendingReleaseSession({
      env,
      installationId: 'install-issue-skew',
      appVersion: '1.2.3',
      hardwareHash: 'hardware-hash-issue-skew',
    });
    const requestBody = await signCanonicalIssueRequest(release.keyPair.privateKey, {
      installation_id: 'install-issue-skew',
      device_public_key: release.keyPair.devicePublicKey,
      release_token: release.releaseToken,
      hardware_hash: release.hardwareHash,
      reason: 'llm_start',
      budget_usd: 0.07,
      model: 'google/gemma-4-26b-a4b-it',
      signed_at: '2026-04-08T06:01:01.000Z',
    });

    const response = await postIssue(env, requestBody);

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
