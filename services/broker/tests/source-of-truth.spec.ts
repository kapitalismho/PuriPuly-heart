import { describe, expect, it } from 'vitest';

import app from '../src/index';

describe('managed trial live usage source of truth', () => {
  it('publishes the live-usage ownership contract independently of entitlement lifecycle', async () => {
    const response = await app.request('http://broker.test/v1/foundation');
    expect(response.status).toBe(200);

    const payload = (await response.json()) as {
      managedTrialPolicy: {
        liveUsage: unknown;
      };
    };

    expect(payload.managedTrialPolicy.liveUsage).toEqual({
      managedAvailability: {
        field: 'managed_availability',
        reportedSeparatelyFromLifecycle: true,
      },
      sourceOfTruthAfterRelease: {
        provider: 'OpenRouter',
        signals: ['key-metadata', 'provider-failures'],
      },
      brokerTracksRemainingBudget: false,
    });
  });
});
