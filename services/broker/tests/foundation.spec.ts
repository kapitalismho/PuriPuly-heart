import { describe, expect, it } from 'vitest';

import app from '../src/index';
import { BROKER_SERVICE_NAME } from '../src/contract';

describe('broker foundation', () => {
  it('publishes the Broker boundary and managed trial policy', async () => {
    const managedPath = {
      provider: 'OpenRouter',
      models: [
        'google/gemma-4-26b-a4b-it',
        'google/gemma-4-31b-it',
        'deepseek/deepseek-v4.1-flash-1017',
        'deepseek/deepseek-v4-flash-0731',
        'deepseek/deepseek-v4-flash',
      ],
    };

    const response = await app.request('http://broker.test/v1/foundation');
    expect(response.status).toBe(200);

    await expect(response.json()).resolves.toEqual({
      service: BROKER_SERVICE_NAME,
      runtime: {
        language: 'TypeScript',
        framework: 'Hono',
        runtime: 'Cloudflare Workers',
        database: 'Cloudflare D1',
        secretStorage: 'Worker secrets',
      },
      bindings: {
        d1: 'BROKER_DB',
        secrets: [
          'OPENROUTER_MANAGEMENT_API_KEY',
          'OPENROUTER_MANAGED_GUARDRAIL_ID',
          'OPENROUTER_MANAGED_USER_HMAC_SECRET',
          'DISCORD_CLIENT_ID',
          'DISCORD_CLIENT_SECRET',
          'DISCORD_REDIRECT_URI_ALLOWLIST',
          'DISCORD_USER_REF_SECRET',
          'DISCORD_IMMEDIATE_ALERT_WEBHOOK_URL',
          'DISCORD_DAILY_REPORT_WEBHOOK_URL',
          'QQ_AUTH_HMAC_PSK',
          'TELEMETRY_SUBJECT_HMAC_SECRET',
          'NETWORK_IDENTITY_HMAC_SECRET',
          'NETWORK_IDENTITY_HMAC_SECRET_PREVIOUS',
        ],
      },
      hosting: {
        regionMode: 'single-region-rollout-assumption',
        d1LocationHint: 'apac',
        infrastructure: ['worker-service', 'd1-database', 'worker-secrets'],
        exclusions: [
          'translation-proxying',
          'multi-region-deployment',
          'kv',
          'r2',
          'admin-dashboard',
        ],
      },
      serviceBoundary: {
        role: 'trial-credential-broker',
        proxiesTranslationText: false,
        inferencePath: 'app-direct-to-openrouter',
      },
      trialProviderPolicy: {
        managedFreeTrial: managedPath,
        upstreamProviderRouting: 'unpinned-by-broker',
        excludedProviders: ['Alibaba'],
      },
      managedTrialPolicy: {
        managedPath,
        budget: { currency: 'USD', hardLimit: 0.07, limitReset: null },
        onboardingCostAccounting: {
          scope: 'llm-only',
          estimationBasis: { inputTokens: 1000, outputTokens: 15, llmCallsPerUtterance: 1.3 },
          operationalBufferPercent: { min: 5, max: 10 },
        },
        entitlement: {
          lifecycle: ['none', 'pending_release', 'active', 'expired', 'revoked'],
          managedAvailability: { field: 'managed_availability', reportedSeparatelyFromLifecycle: true },
          issuance: {
            keyScope: 'user-specific',
            maxManagedKeysPerEligibleInstallation: 1,
            expiry: { durationMonths: 3, anchor: 'issued_at' },
          },
        },
        liveUsage: {
          managedAvailability: { field: 'managed_availability', reportedSeparatelyFromLifecycle: true },
          sourceOfTruthAfterRelease: { provider: 'OpenRouter', signals: ['key-metadata', 'provider-failures'] },
          brokerTracksRemainingBudget: false,
        },
      },
    });
  });

  it('reports broker liveness without implying proxy behavior', async () => {
    const response = await app.request('http://broker.test/healthz');

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({
      ok: true,
      service: BROKER_SERVICE_NAME,
    });
  });
});
