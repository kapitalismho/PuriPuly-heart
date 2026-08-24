import { describe, expect, it } from 'vitest';

import { MANAGED_TRIAL_POLICY, TRIAL_PROVIDER_POLICY } from '../src/contract';

describe('managed trial policy', () => {
  it('keeps the managed path pinned to OpenRouter and reuses the curated model-pool contract', () => {
    expect(MANAGED_TRIAL_POLICY.managedPath).toEqual({
      provider: 'OpenRouter',
      models: [
        'google/gemma-4-26b-a4b-it',
        'google/gemma-4-31b-it',
        'deepseek/deepseek-v4-flash-0731',
        'deepseek/deepseek-v4-flash',
      ],
    });
    expect(MANAGED_TRIAL_POLICY.managedPath).toBe(
      TRIAL_PROVIDER_POLICY.managedFreeTrial,
    );
  });

  it('limits issuance to one user-specific managed key per eligible installation with three-month expiry', () => {
    expect(MANAGED_TRIAL_POLICY.entitlement.issuance).toEqual({
      keyScope: 'user-specific',
      maxManagedKeysPerEligibleInstallation: 1,
      expiry: {
        durationMonths: 3,
        anchor: 'issued_at',
      },
    });
  });
});
