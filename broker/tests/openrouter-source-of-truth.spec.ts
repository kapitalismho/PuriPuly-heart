import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  signCanonicalIssueRequest,
  signCanonicalStatusRequest,
} from './test-support/ed25519';
import {
  createPendingReleaseSession,
  mockOpenRouterManagementApi,
} from './test-support/openrouter-issue';
import { createTestBrokerEnv } from './test-support/sqlite-d1';
import { getTrialStatus, postIssue } from './test-support/trial-api';

describe('managed OpenRouter live usage source of truth', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it('delegates the non-resetting budget to OpenRouter while exposing lifecycle rather than live usage', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const management = mockOpenRouterManagementApi();
    const env = createTestBrokerEnv();
    const release = await createPendingReleaseSession({
      env,
      installationId: 'install-issue-source-of-truth',
      appVersion: '1.2.3',
      hardwareHash: 'hardware-hash-issue-source-of-truth',
    });
    const requestBody = await signCanonicalIssueRequest(release.keyPair.privateKey, {
      installation_id: 'install-issue-source-of-truth',
      device_public_key: release.keyPair.devicePublicKey,
      release_token: release.releaseToken,
      hardware_hash: release.hardwareHash,
      reason: 'llm_start',
      budget_usd: 0.07,
      model: 'google/gemma-4-26b-a4b-it',
      signed_at: '2026-04-08T06:00:45.000Z',
    });

    const response = await postIssue(env, requestBody);
    expect(response.status).toBe(200);

    expect(management.fetchMock).toHaveBeenCalledTimes(2);
    const [createUrl, createInit] = management.fetchMock.mock.calls[0]!;
    expect(createUrl).toBe('https://openrouter.ai/api/v1/keys');
    expect(createInit?.method).toBe('POST');
    expect(JSON.parse(String(createInit?.body))).toEqual({
      name: expect.stringContaining('puripuly-heart:install-issue-source-of-truth:'),
      limit: 0.07,
      limit_reset: null,
      include_byok_in_limit: false,
      expires_at: '2026-07-08T06:00:00.000Z',
    });
    const [guardrailUrl, guardrailInit] = management.fetchMock.mock.calls[1]!;
    expect(guardrailUrl).toBe(
      'https://openrouter.ai/api/v1/guardrails/test-managed-guardrail-id/assignments/keys',
    );
    expect(guardrailInit?.method).toBe('POST');
    expect(JSON.parse(String(guardrailInit?.body))).toEqual({
      key_hashes: [management.childKey.hash],
    });
    const payload = (await response.json()) as Record<string, unknown>;
    expect(payload).not.toHaveProperty('remaining_budget_usd');
    expect(payload).not.toHaveProperty('remaining_budget');
    expect(payload).not.toHaveProperty('usage_usd');
    expect(payload).not.toHaveProperty('usage');
    expect(payload).not.toHaveProperty('key_metadata');

    const signedStatus = await signCanonicalStatusRequest(release.keyPair.privateKey, {
      installation_id: 'install-issue-source-of-truth',
      timestamp: '2026-04-08T06:00:45.000Z',
    });
    const statusResponse = await getTrialStatus({
      env,
      installationId: 'install-issue-source-of-truth',
      headers: {
        'X-Puripuly-Timestamp': signedStatus.timestamp,
        'X-Puripuly-Signature': signedStatus.signature,
      },
    });
    expect(statusResponse.status).toBe(200);
    await expect(statusResponse.json()).resolves.toEqual({
      managed_state: { lifecycle: 'active', managed_availability: true },
      current_entitlement: {
        provider: 'OpenRouter',
        budget_usd: 0.07,
        issued_at: '2026-04-08T06:00:00.000Z',
        expires_at: '2026-07-08T06:00:00.000Z',
      },
      onboarding_eligibility: {
        eligible: false,
        reason: 'active',
        requires_discord_oauth: false,
      },
    });
    expect(management.fetchMock).toHaveBeenCalledTimes(2);
  });
});
