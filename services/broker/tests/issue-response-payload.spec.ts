import { afterEach, describe, expect, it, vi } from 'vitest';

import * as openRouterUserId from '../src/openrouter-user-id';
import { signCanonicalIssueRequest } from './test-support/ed25519';
import {
  createPendingReleaseSession,
  mockOpenRouterManagementApi,
} from './test-support/openrouter-issue';
import { createTestBrokerEnv } from './test-support/sqlite-d1';
import { postIssue } from './test-support/trial-api';

describe('POST /v1/providers/openrouter/issue response payload', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it('returns the provider-created child key once without leaking release-session fields or management credentials', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const managementApi = mockOpenRouterManagementApi();
    const env = createTestBrokerEnv();
    const release = await createPendingReleaseSession({
      env,
      installationId: 'install-issue-response',
      appVersion: '1.2.3',
      hardwareHash: 'hardware-hash-issue-response',
    });
    const requestBody = await signCanonicalIssueRequest(release.keyPair.privateKey, {
      installation_id: 'install-issue-response',
      device_public_key: release.keyPair.devicePublicKey,
      release_token: release.releaseToken,
      hardware_hash: release.hardwareHash,
      reason: 'llm_start',
      budget_usd: 0.07,
      model: 'google/gemma-4-26b-a4b-it',
      signed_at: '2026-04-08T06:00:45.000Z',
    });
    const expectedOpenRouterUserId = await openRouterUserId.deriveManagedOpenRouterUserId({
      installationId: 'install-issue-response',
      secret: env.OPENROUTER_MANAGED_USER_HMAC_SECRET,
    });
    if (!expectedOpenRouterUserId) {
      throw new Error('expected managed OpenRouter user ID for test setup');
    }

    const response = await postIssue(env, requestBody);
    expect(response.status).toBe(200);

    const payload = (await response.json()) as Record<string, unknown>;
    expect(payload).toEqual({
      openrouter_api_key: managementApi.childKey.rawKey,
      openrouter_user_id: expectedOpenRouterUserId,
      managed_credential_ref: managementApi.childKey.hash,
      managed_state: {
        lifecycle: 'active',
        managed_availability: true,
      },
      expires_at: '2026-07-08T06:00:00.000Z',
      budget_usd: 0.07,
      model: 'google/gemma-4-26b-a4b-it',
    });
    expect(payload.openrouter_api_key).not.toBe(payload.managed_credential_ref);
    expect(payload.openrouter_api_key).not.toBe(env.OPENROUTER_MANAGEMENT_API_KEY);
    expect(payload).not.toHaveProperty('release_token');
    expect(payload).not.toHaveProperty('release_session_ref');
    expect(payload).not.toHaveProperty('release_token_hash');
    expect(payload).not.toHaveProperty('release_token_expires_at');
  });

  it('omits openrouter_user_id when managed user ID derivation cannot produce a value', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:10:00Z'));

    const managementApi = mockOpenRouterManagementApi();
    const env = createTestBrokerEnv();
    const release = await createPendingReleaseSession({
      env,
      installationId: 'install-issue-response-no-user-id',
      appVersion: '1.2.3',
      hardwareHash: 'hardware-hash-issue-response-no-user-id',
      verifySignedAt: '2026-04-08T06:10:30.000Z',
    });
    delete (env as Record<string, unknown>).OPENROUTER_MANAGED_USER_HMAC_SECRET;
    expect(env).not.toHaveProperty('OPENROUTER_MANAGED_USER_HMAC_SECRET');
    const deriveManagedOpenRouterUserIdSpy = vi.spyOn(
      openRouterUserId,
      'deriveManagedOpenRouterUserId',
    );
    const requestBody = await signCanonicalIssueRequest(release.keyPair.privateKey, {
      installation_id: 'install-issue-response-no-user-id',
      device_public_key: release.keyPair.devicePublicKey,
      release_token: release.releaseToken,
      hardware_hash: release.hardwareHash,
      reason: 'llm_start',
      budget_usd: 0.07,
      model: 'google/gemma-4-26b-a4b-it',
      signed_at: '2026-04-08T06:10:45.000Z',
    });

    const response = await postIssue(env, requestBody);
    expect(response.status).toBe(200);

    const payload = (await response.json()) as Record<string, unknown>;
    expect(payload).toEqual({
      openrouter_api_key: managementApi.childKey.rawKey,
      managed_credential_ref: managementApi.childKey.hash,
      managed_state: {
        lifecycle: 'active',
        managed_availability: true,
      },
      expires_at: '2026-07-08T06:10:00.000Z',
      budget_usd: 0.07,
      model: 'google/gemma-4-26b-a4b-it',
    });
    expect(deriveManagedOpenRouterUserIdSpy).not.toHaveBeenCalled();
    expect(payload).not.toHaveProperty('openrouter_user_id');
  });
});
