import { afterEach, describe, expect, it, vi } from 'vitest';

import { signCanonicalIssueRequest } from './test-support/ed25519';
import { normalizedErrorEnvelope } from './test-support/errors';
import {
  createPendingReleaseSession,
  mockOpenRouterManagementApi,
} from './test-support/openrouter-issue';
import { createTestBrokerEnv } from './test-support/sqlite-d1';
import { postIssue } from './test-support/trial-api';

function createDeferred(): {
  promise: Promise<void>;
  resolve: () => void;
} {
  let resolve!: () => void;
  return {
    promise: new Promise<void>((resolvePromise) => {
      resolve = resolvePromise;
    }),
    resolve,
  };
}

describe('POST /v1/providers/openrouter/issue idempotency', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it('rejects reissue for an active entitlement', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const managementApi = mockOpenRouterManagementApi();
    const env = createTestBrokerEnv();
    const release = await createPendingReleaseSession({
      env,
      installationId: 'install-issue-idempotent',
      appVersion: '1.2.3',
      hardwareHash: 'hardware-hash-issue-idempotent',
    });
    const requestBody = await signCanonicalIssueRequest(release.keyPair.privateKey, {
      installation_id: 'install-issue-idempotent',
      device_public_key: release.keyPair.devicePublicKey,
      release_token: release.releaseToken,
      hardware_hash: release.hardwareHash,
      reason: 'llm_start',
      budget_usd: 0.07,
      model: 'google/gemma-4-26b-a4b-it',
      signed_at: '2026-04-08T06:00:45.000Z',
    });

    const firstResponse = await postIssue(env, requestBody);
    const secondResponse = await postIssue(env, requestBody);

    expect(firstResponse.status).toBe(200);
    expect(secondResponse.status).toBe(409);
    await expect(secondResponse.json()).resolves.toEqual(
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

    const entitlement = env.__db
      .prepare(
        `SELECT status, managed_credential_ref, issued_at, expires_at,
                release_session_ref, release_token_hash, release_token_expires_at
           FROM openrouter_entitlements
          WHERE installation_id = ?`,
      )
      .get('install-issue-idempotent') as Record<string, unknown>;

    expect(entitlement).toEqual({
      status: 'active',
      managed_credential_ref: managementApi.childKey.hash,
      issued_at: '2026-04-08T06:00:00.000Z',
      expires_at: '2026-07-08T06:00:00.000Z',
      release_session_ref: expect.any(String),
      release_token_hash: expect.any(String),
      release_token_expires_at: release.releaseTokenExpiresAt,
    });
    expect(managementApi.fetchMock).toHaveBeenCalledTimes(2);
  });

  it('creates at most one child key for concurrent issue attempts on one release session', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const managementApi = mockOpenRouterManagementApi();
    const env = createTestBrokerEnv();
    const release = await createPendingReleaseSession({
      env,
      installationId: 'install-issue-single-flight',
      appVersion: '1.2.3',
      hardwareHash: 'hardware-hash-issue-single-flight',
    });

    const requestBody = await signCanonicalIssueRequest(release.keyPair.privateKey, {
      installation_id: 'install-issue-single-flight',
      device_public_key: release.keyPair.devicePublicKey,
      release_token: release.releaseToken,
      hardware_hash: release.hardwareHash,
      reason: 'llm_start',
      budget_usd: 0.07,
      model: 'google/gemma-4-26b-a4b-it',
      signed_at: '2026-04-08T06:00:45.000Z',
    });

    const [firstResponse, secondResponse] = await Promise.all([
      postIssue(env, requestBody),
      postIssue(env, requestBody),
    ]);

    expect([firstResponse.status, secondResponse.status].sort()).toEqual([200, 409]);
    expect(
      managementApi.fetchMock.mock.calls.filter(
        ([url, init]) =>
          String(url) === 'https://openrouter.ai/api/v1/keys' &&
          (init?.method ?? 'GET') === 'POST',
      ),
    ).toHaveLength(1);
    expect(
      managementApi.fetchMock.mock.calls.filter(
        ([url]) => String(url).includes('/guardrails/test-managed-guardrail-id/assignments/keys'),
      ),
    ).toHaveLength(1);
    await expect(firstResponse.text()).resolves.not.toHaveLength(0);
    await expect(secondResponse.text()).resolves.not.toHaveLength(0);
  });

  it('returns managed_key_unrecoverable when lock acquisition loses to an already-activated request', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    let pausePendingRequest = false;
    let pendingRequestPausedOnce = false;
    const pendingRequestPaused = createDeferred();
    const releasePendingRequest = createDeferred();
    const env = createTestBrokerEnv({
      beforeRun: async ({ sql, params }) => {
        if (
          pausePendingRequest &&
          !pendingRequestPausedOnce &&
          sql.includes('INSERT INTO broker_request_events') &&
          params[0] === 'POST /v1/providers/openrouter/issue'
        ) {
          pendingRequestPausedOnce = true;
          pendingRequestPaused.resolve();
          await releasePendingRequest.promise;
        }
      },
    });
    const managementApi = mockOpenRouterManagementApi();
    const release = await createPendingReleaseSession({
      env,
      installationId: 'install-issue-lock-loss-active',
      appVersion: '1.2.3',
      hardwareHash: 'hardware-hash-issue-lock-loss-active',
    });
    const requestBody = await signCanonicalIssueRequest(release.keyPair.privateKey, {
      installation_id: 'install-issue-lock-loss-active',
      device_public_key: release.keyPair.devicePublicKey,
      release_token: release.releaseToken,
      hardware_hash: release.hardwareHash,
      reason: 'llm_start',
      budget_usd: 0.07,
      model: 'google/gemma-4-26b-a4b-it',
      signed_at: '2026-04-08T06:00:45.000Z',
    });

    pausePendingRequest = true;
    const pendingResponsePromise = postIssue(env, requestBody);
    await pendingRequestPaused.promise;

    const activatingResponse = await postIssue(env, requestBody);
    releasePendingRequest.resolve();
    const pendingResponse = await pendingResponsePromise;

    expect(activatingResponse.status).toBe(200);
    expect(pendingResponse.status).toBe(409);
    await expect(pendingResponse.json()).resolves.toEqual(
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
    expect(
      managementApi.fetchMock.mock.calls.filter(
        ([url, init]) =>
          String(url) === 'https://openrouter.ai/api/v1/keys' &&
          (init?.method ?? 'GET') === 'POST',
      ),
    ).toHaveLength(1);
  });

  it('invalidates the release session after post-mint failure so the same token cannot mint again', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const managementApi = mockOpenRouterManagementApi();
    const env = createTestBrokerEnv({
      beforeRun({ sql }) {
        if (sql.includes('SET status = ?')) {
          throw new Error('simulated activation failure');
        }
      },
    });
    const release = await createPendingReleaseSession({
      env,
      installationId: 'install-issue-post-mint-failure',
      appVersion: '1.2.3',
      hardwareHash: 'hardware-hash-issue-post-mint-failure',
    });

    const requestBody = await signCanonicalIssueRequest(release.keyPair.privateKey, {
      installation_id: 'install-issue-post-mint-failure',
      device_public_key: release.keyPair.devicePublicKey,
      release_token: release.releaseToken,
      hardware_hash: release.hardwareHash,
      reason: 'llm_start',
      budget_usd: 0.07,
      model: 'google/gemma-4-26b-a4b-it',
      signed_at: '2026-04-08T06:00:45.000Z',
    });

    const firstResponse = await postIssue(env, requestBody);
    const secondResponse = await postIssue(env, requestBody);

    expect(firstResponse.status).toBe(500);
    expect(secondResponse.status).toBe(401);
    await expect(secondResponse.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'challenge_invalid',
        class: 'security_fail',
        subcode: 'release_token_invalid',
        message: 'release_token does not match the active release session for installation_id',
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

    expect(
      managementApi.fetchMock.mock.calls.filter(
        ([url, init]) =>
          String(url) === 'https://openrouter.ai/api/v1/keys' &&
          (init?.method ?? 'GET') === 'POST',
      ),
    ).toHaveLength(1);

    const entitlement = env.__db
      .prepare(
        `SELECT status, managed_credential_ref, release_session_ref, release_token_hash, release_token_expires_at
           FROM openrouter_entitlements
          WHERE installation_id = ?`,
      )
      .get('install-issue-post-mint-failure') as Record<string, unknown>;

    expect(entitlement).toEqual({
      status: 'pending_release',
      managed_credential_ref: null,
      release_session_ref: null,
      release_token_hash: null,
      release_token_expires_at: null,
    });
  });
});
