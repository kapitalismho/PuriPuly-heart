import { describe, expect, it } from 'vitest';

import { MANAGED_TRIAL_BUDGET_POLICY } from '../src/contract';

describe('managed OpenRouter budget policy', () => {
  it('freezes the hard managed budget at 7 cents with no reset window', () => {
    expect(MANAGED_TRIAL_BUDGET_POLICY).toEqual({
      currency: 'USD',
      hardLimit: 0.07,
      limitReset: null,
    });
  });
});
