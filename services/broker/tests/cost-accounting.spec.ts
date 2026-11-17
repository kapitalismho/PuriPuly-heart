import { describe, expect, it } from 'vitest';

import { MANAGED_TRIAL_COST_ACCOUNTING_POLICY } from '../src/contract';

describe('managed trial onboarding cost accounting', () => {
  it('uses the frozen llm-only estimation basis for onboarding cost calculations', () => {
    expect(MANAGED_TRIAL_COST_ACCOUNTING_POLICY).toEqual({
      scope: 'llm-only',
      estimationBasis: {
        inputTokens: 1000,
        outputTokens: 15,
        llmCallsPerUtterance: 1.3,
      },
      operationalBufferPercent: {
        min: 5,
        max: 10,
      },
    });
  });
});
