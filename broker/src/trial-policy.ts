export const MANAGED_TRIAL_ALLOWED_MODELS = [
  'google/gemma-4-26b-a4b-it',
  'qwen/qwen3.5-flash-02-23',
  'deepseek/deepseek-v4-flash-0731',
  'deepseek/deepseek-v4-flash-0423',
  'deepseek/deepseek-v4-flash',
  'google/gemini-2.5-flash-lite',
] as const;

const MANAGED_FREE_TRIAL_POLICY = {
  provider: 'OpenRouter',
  models: MANAGED_TRIAL_ALLOWED_MODELS,
} as const;

export const TRIAL_PROVIDER_POLICY = {
  managedFreeTrial: MANAGED_FREE_TRIAL_POLICY,
  upstreamProviderRouting: 'unpinned-by-broker',
  excludedProviders: ['Alibaba'],
} as const;

export const MANAGED_TRIAL_BUDGET_POLICY = {
  currency: 'USD',
  hardLimit: 0.07,
  limitReset: null,
} as const;

export const MANAGED_TRIAL_COST_ACCOUNTING_POLICY = {
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
} as const;

export const MANAGED_TRIAL_LIFECYCLE_VALUES = [
  'none',
  'pending_release',
  'active',
  'expired',
  'revoked',
] as const;

export type ManagedTrialLifecycle =
  (typeof MANAGED_TRIAL_LIFECYCLE_VALUES)[number];

export const MANAGED_TRIAL_ENTITLEMENT_POLICY = {
  lifecycle: MANAGED_TRIAL_LIFECYCLE_VALUES,
  managedAvailability: {
    field: 'managed_availability',
    reportedSeparatelyFromLifecycle: true,
  },
  issuance: {
    keyScope: 'user-specific',
    maxManagedKeysPerEligibleInstallation: 1,
    expiry: {
      durationMonths: 3,
      anchor: 'issued_at',
    },
  },
} as const;

export const MANAGED_TRIAL_LIVE_USAGE_POLICY = {
  managedAvailability: MANAGED_TRIAL_ENTITLEMENT_POLICY.managedAvailability,
  sourceOfTruthAfterRelease: {
    provider: 'OpenRouter',
    signals: ['key-metadata', 'provider-failures'],
  },
  brokerTracksRemainingBudget: false,
} as const;

export const MANAGED_TRIAL_POLICY = {
  managedPath: MANAGED_FREE_TRIAL_POLICY,
  budget: MANAGED_TRIAL_BUDGET_POLICY,
  onboardingCostAccounting: MANAGED_TRIAL_COST_ACCOUNTING_POLICY,
  entitlement: MANAGED_TRIAL_ENTITLEMENT_POLICY,
  liveUsage: MANAGED_TRIAL_LIVE_USAGE_POLICY,
} as const;
