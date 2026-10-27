import { afterEach, describe, expect, it, vi } from 'vitest';

import { createDeviceKeyPair, signCanonicalStatusRequest } from './test-support/ed25519';
import { createTestBrokerEnv, insertEntitlement } from './test-support/sqlite-d1';
import { getTrialStatus, issueChallenge } from './test-support/trial-api';

describe('GET /v1/trial/status source of truth boundary', () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it('keeps live remaining budget out of broker status responses', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    const keyPair = await createDeviceKeyPair();
    await issueChallenge({
      env,
      installationId: 'install-status-source-of-truth',
      devicePublicKey: keyPair.devicePublicKey,
      appVersion: '1.2.3',
    });
    insertEntitlement(env, {
      installation_id: 'install-status-source-of-truth',
      status: 'active',
      budget_usd: 0.07,
      issued_at: '2026-04-01T00:00:00Z',
      expires_at: '2026-10-01T00:00:00Z',
    });
    const signedRequest = await signCanonicalStatusRequest(keyPair.privateKey, {
      installation_id: 'install-status-source-of-truth',
      timestamp: '2026-04-08T06:00:30.000Z',
    });

    const response = await getTrialStatus({
      env,
      installationId: 'install-status-source-of-truth',
      headers: {
        'X-Puripuly-Timestamp': signedRequest.timestamp,
        'X-Puripuly-Signature': signedRequest.signature,
      },
    });
    expect(response.status).toBe(200);

    const payload = (await response.json()) as Record<string, unknown>;
    expect(payload).toMatchObject({
      current_entitlement: {
        provider: 'OpenRouter',
        budget_usd: 0.07,
        expires_at: '2026-10-01T00:00:00Z',
      },
    });
    expect(payload).not.toHaveProperty('remaining_budget_usd');
    expect(payload).not.toHaveProperty('live_remaining_budget_usd');
    expect(payload).not.toHaveProperty('consumed_budget_usd');
  });
});
