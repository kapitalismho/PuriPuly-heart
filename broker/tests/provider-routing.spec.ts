import { describe, expect, it } from 'vitest';

import { TRIAL_PROVIDER_POLICY } from '../src/contract';

describe('broker provider routing', () => {
  it('fixes managed free-trial onboarding to OpenRouter plus the curated managed model pool', () => {
    expect(TRIAL_PROVIDER_POLICY.managedFreeTrial).toEqual({
      provider: 'OpenRouter',
      models: [
        'google/gemma-4-26b-a4b-it',
        'google/gemma-4-31b-it',
        'deepseek/deepseek-v4-flash-0731',
        'deepseek/deepseek-v4-flash',
      ],
    });
  });

  it('keeps upstream provider routing unpinned and excludes Alibaba from the broker surface', () => {
    expect(TRIAL_PROVIDER_POLICY.upstreamProviderRouting).toBe(
      'unpinned-by-broker',
    );
    expect(TRIAL_PROVIDER_POLICY.excludedProviders).toEqual(['Alibaba']);
  });
});
