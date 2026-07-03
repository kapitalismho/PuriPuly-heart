import { describe, expect, it } from 'vitest';

import app from '../src/index';
import {
  BROKER_SERVICE_NAME,
  BROKER_SOURCE_OFFER,
  BROKER_SOURCE_OFFER_LINK_HEADER,
  MANAGED_TRIAL_BUDGET_POLICY,
  MANAGED_TRIAL_COST_ACCOUNTING_POLICY,
  MANAGED_TRIAL_ENTITLEMENT_POLICY,
  MANAGED_TRIAL_LIVE_USAGE_POLICY,
  TRIAL_PROVIDER_POLICY,
} from '../src/contract';

describe('broker foundation', () => {
  it('exposes a fetch-compatible Hono worker app with explicit bindings', async () => {
    expect(app.fetch).toBeTypeOf('function');

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
          'OPENROUTER_MANAGED_API_KEY',
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
        managedFreeTrial: {
          provider: 'OpenRouter',
          models: [
            'google/gemma-4-26b-a4b-it',
            'qwen/qwen3.5-flash-02-23',
            'deepseek/deepseek-v4-flash',
            'google/gemini-2.5-flash-lite',
          ],
        },
        upstreamProviderRouting: 'unpinned-by-broker',
        excludedProviders: ['Alibaba'],
      },
      managedTrialPolicy: {
        managedPath: TRIAL_PROVIDER_POLICY.managedFreeTrial,
        budget: MANAGED_TRIAL_BUDGET_POLICY,
        onboardingCostAccounting: MANAGED_TRIAL_COST_ACCOUNTING_POLICY,
        entitlement: MANAGED_TRIAL_ENTITLEMENT_POLICY,
        liveUsage: MANAGED_TRIAL_LIVE_USAGE_POLICY,
      },
    });
  });

  it('keeps the public foundation contract at the boundary level only', async () => {
    const response = await app.request('http://broker.test/v1/foundation');
    expect(response.status).toBe(200);

    const payload = (await response.json()) as {
      serviceBoundary: Record<string, unknown>;
    };

    expect(payload.serviceBoundary).not.toHaveProperty('responsibilities');
  });

  it('reports broker liveness without implying proxy behavior', async () => {
    const response = await app.request('http://broker.test/healthz');

    expect(response.status).toBe(200);
    expect(response.headers.get('Link')).toBe(BROKER_SOURCE_OFFER_LINK_HEADER);
    await expect(response.json()).resolves.toEqual({
      ok: true,
      service: BROKER_SERVICE_NAME,
    });
  });

  it('offers corresponding source for network users', async () => {
    const response = await app.request('http://broker.test/source');

    expect(response.status).toBe(200);
    expect(response.headers.get('Link')).toBe(BROKER_SOURCE_OFFER_LINK_HEADER);
    await expect(response.json()).resolves.toEqual(BROKER_SOURCE_OFFER);
  });

  it('keeps the source offer visible on broker error responses', async () => {
    const response = await app.request('http://broker.test/v1/trial/status');

    expect(response.status).toBe(400);
    expect(response.headers.get('Link')).toBe(BROKER_SOURCE_OFFER_LINK_HEADER);
  });
});
