import { describe, expect, it } from 'vitest';

import { normalizedErrorEnvelope } from './test-support/errors';
import { createTestBrokerEnv } from './test-support/sqlite-d1';
import { getTrialStatus } from './test-support/trial-api';

describe('GET /v1/trial/status header names', () => {
  it('requires the X-Puripuly-Timestamp header name', async () => {
    const env = createTestBrokerEnv();

    const response = await getTrialStatus({
      env,
      installationId: 'install-status-headers',
      headers: {
        'X-Puripuly-Signature': 'placeholder',
      },
    });

    expect(response.status).toBe(400);
    await expect(response.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'invalid_request',
        class: 'terminal',
        message: 'X-Puripuly-Timestamp header is required',
      }),
    );
  });

  it('requires the X-Puripuly-Signature header name', async () => {
    const env = createTestBrokerEnv();

    const response = await getTrialStatus({
      env,
      installationId: 'install-status-headers',
      headers: {
        'X-Puripuly-Timestamp': '2026-04-08T06:00:00.000Z',
      },
    });

    expect(response.status).toBe(400);
    await expect(response.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'invalid_request',
        class: 'terminal',
        message: 'X-Puripuly-Signature header is required',
      }),
    );
  });
});
