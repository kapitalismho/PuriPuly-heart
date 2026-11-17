import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  createDeviceKeyPair,
  signCanonicalDiscordIssueRequest,
  type DeviceKeyPair,
  type SignedDiscordIssueRequestInput,
} from './test-support/ed25519';
import { normalizedErrorEnvelope } from './test-support/errors';
import { createTestBrokerEnv, type TestBrokerEnv } from './test-support/sqlite-d1';
import { postDiscordIssue, postDiscordStart } from './test-support/trial-api';
import { updateAbuseControls } from './test-support/abuse-controls';

const REGISTERED_REDIRECT_URI = 'http://127.0.0.1:62187/discord/callback';
const APP_VERSION = '1.2.3';
const MODEL = 'google/gemma-4-26b-a4b-it';
const NOW_ISO = '2026-04-30T06:00:00.000Z';
const SIGNED_AT_ISO = '2026-04-30T06:00:30.000Z';
const DISCORD_TOKEN_URL = 'https://discord.com/api/oauth2/token';
const DISCORD_USER_URL = 'https://discord.com/api/users/@me';
const OPENROUTER_KEYS_URL = 'https://openrouter.ai/api/v1/keys';
const OPENROUTER_GUARDRAIL_URL =
  'https://openrouter.ai/api/v1/guardrails/test-managed-guardrail-id/assignments/keys';
const DISCORD_EPOCH_MS = 1420070400000n;

interface StartedDiscordSession {
  env: TestBrokerEnv;
  keyPair: DeviceKeyPair;
  installationId: string;
  state: string;
  issueNonce: string;
  redirectUri: string;
  appVersion: string;
  fingerprintSaltVersion: number;
}

describe('Discord issue concurrency', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it('cap concurrent reservations allow exactly one active delivery and reject the other at maxCount=1', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const env = createTestBrokerEnv();
    updateAbuseControls(env, (controls) => {
      controls.newActiveEntitlementsPerDay.maxCount = 1;
    });

    const first = await startDiscordSession('install-discord-cap-concurrent-a', env);
    const second = await startDiscordSession('install-discord-cap-concurrent-b', env);
    const api = mockConcurrentDiscordAndOpenRouterApi({
      'discord-oauth-code-cap-concurrent-a': {
        id: discordSnowflakeForAgeDays(31),
        verified: true,
      },
      'discord-oauth-code-cap-concurrent-b': {
        id: discordSnowflakeForAgeDays(32),
        verified: true,
      },
    });

    const firstBody = await signedIssueRequest(first, {
      code: 'discord-oauth-code-cap-concurrent-a',
      hardware_hash: 'hardware-hash-cap-concurrent-a',
    });
    const secondBody = await signedIssueRequest(second, {
      code: 'discord-oauth-code-cap-concurrent-b',
      hardware_hash: 'hardware-hash-cap-concurrent-b',
    });

    const responses = await Promise.all([
      postDiscordIssue(env, firstBody),
      postDiscordIssue(env, secondBody),
    ]);

    const successResponses = responses.filter((response) => response.status === 200);
    const capResponses = responses.filter((response) => response.status === 503);
    expect(successResponses).toHaveLength(1);
    expect(capResponses).toHaveLength(1);
    await expect(capResponses[0]!.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'issuance_suspended',
        class: 'retryable',
        subcode: 'global_cap_reached',
        retryAfterMs: 64_800_000,
        message: 'Daily managed issuance cap reached',
      }),
    );

    expect(api.openRouterCreateCalls).toHaveLength(1);
    expect(api.openRouterGuardrailCalls).toHaveLength(1);
    await expect(successResponses[0]!.json()).resolves.toEqual(
      expect.objectContaining({
        openrouter_api_key: expect.stringMatching(
          /^or-discord-managed-child-key-concurrent-/u,
        ),
      }),
    );
    expect(countRows(env, "openrouter_entitlements WHERE status = 'active'")).toBe(1);
    expect(
      countRows(env, "openrouter_entitlements WHERE discord_issue_status = 'issuing'"),
    ).toBe(0);
    expect(countRows(env, 'discord_identities')).toBe(1);
  });

  it('same hardware concurrent reservations allow exactly one key creation and reject the duplicate reservation', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const env = createReservationBarrierEnv();
    updateAbuseControls(env, (controls) => {
      controls.newActiveEntitlementsPerDay.maxCount = 5;
    });

    const first = await startDiscordSession('install-discord-same-hardware-concurrent-a', env);
    const second = await startDiscordSession('install-discord-same-hardware-concurrent-b', env);
    const api = mockConcurrentDiscordAndOpenRouterApi({
      'discord-oauth-code-same-hardware-concurrent-a': {
        id: discordSnowflakeForAgeDays(31),
        verified: true,
      },
      'discord-oauth-code-same-hardware-concurrent-b': {
        id: discordSnowflakeForAgeDays(32),
        verified: true,
      },
    });

    const firstBody = await signedIssueRequest(first, {
      code: 'discord-oauth-code-same-hardware-concurrent-a',
      hardware_hash: 'hardware-hash-same-hardware-concurrent',
    });
    const secondBody = await signedIssueRequest(second, {
      code: 'discord-oauth-code-same-hardware-concurrent-b',
      hardware_hash: 'hardware-hash-same-hardware-concurrent',
    });

    const responses = await Promise.all([
      postDiscordIssue(env, firstBody),
      postDiscordIssue(env, secondBody),
    ]);

    const successResponses = responses.filter((response) => response.status === 200);
    const duplicateResponses = responses.filter((response) => response.status === 409);
    expect(successResponses).toHaveLength(1);
    expect(duplicateResponses).toHaveLength(1);
    await expect(duplicateResponses[0]!.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'trial_not_eligible',
        class: 'terminal',
        subcode: 'hardware_duplicate',
        message: 'This device has already used a managed trial',
      }),
    );

    expect(api.openRouterCreateCalls).toHaveLength(1);
    expect(api.openRouterGuardrailCalls).toHaveLength(1);
    expect(countRows(env, "openrouter_entitlements WHERE status = 'active'")).toBe(1);
    expect(
      countRows(env, "openrouter_entitlements WHERE discord_issue_status = 'issuing'"),
    ).toBe(0);
    expect(countRows(env, 'discord_identities')).toBe(1);
  });

  it('same final issue replayed concurrently creates and returns exactly one managed child key', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const env = createTestBrokerEnv();
    const started = await startDiscordSession('install-discord-same-state-concurrent', env);
    const api = mockConcurrentDiscordAndOpenRouterApi({
      'discord-oauth-code-same-state-concurrent': {
        id: discordSnowflakeForAgeDays(31),
        verified: true,
      },
    });
    const body = await signedIssueRequest(started, {
      code: 'discord-oauth-code-same-state-concurrent',
      hardware_hash: 'hardware-hash-same-state-concurrent',
    });

    const responses = await Promise.all([
      postDiscordIssue(env, body),
      postDiscordIssue(env, body),
    ]);

    const successResponses = responses.filter((response) => response.status === 200);
    const rejectedResponses = responses.filter((response) => response.status === 409);
    expect(successResponses).toHaveLength(1);
    expect(rejectedResponses).toHaveLength(1);
    await expect(successResponses[0]!.json()).resolves.toEqual(
      expect.objectContaining({
        openrouter_api_key: 'or-discord-managed-child-key-concurrent-1',
        managed_credential_ref: 'hash_discord_managed_child_concurrent_1',
      }),
    );
    expect(api.openRouterCreateCalls).toHaveLength(1);
    expect(api.openRouterGuardrailCalls).toHaveLength(1);
    expect(countRows(env, "openrouter_entitlements WHERE status = 'active'")).toBe(1);
    expect(
      countRows(env, "openrouter_entitlements WHERE discord_issue_status = 'issuing'"),
    ).toBe(0);
  });
});

function createReservationBarrierEnv(): TestBrokerEnv {
  let reservationAttempts = 0;
  let releaseReservations: (() => void) | null = null;
  const bothReservationsReached = new Promise<void>((resolve) => {
    releaseReservations = resolve;
  });

  return createTestBrokerEnv({
    beforeRun: async ({ sql }) => {
      if (
        !sql.includes('INSERT INTO openrouter_entitlements') ||
        !sql.includes('discord_issue_status')
      ) {
        return;
      }

      reservationAttempts += 1;
      if (reservationAttempts >= 2) {
        releaseReservations?.();
        return;
      }

      await bothReservationsReached;
    },
  });
}

async function startDiscordSession(
  installationId: string,
  env: TestBrokerEnv,
): Promise<StartedDiscordSession> {
  const keyPair = await createDeviceKeyPair();
  const response = await postDiscordStart(env, {
    installation_id: installationId,
    device_public_key: keyPair.devicePublicKey,
    redirect_uri: REGISTERED_REDIRECT_URI,
    app_version: APP_VERSION,
  });

  if (response.status !== 200) {
    throw new Error(`Discord start failed with status ${response.status}`);
  }

  const payload = (await response.json()) as {
    authorization_url: string;
    issue_nonce: string;
    redirect_uri: string;
    fingerprint_salt_version: number;
  };
  const state = new URL(payload.authorization_url).searchParams.get('state');
  if (!state) {
    throw new Error('Discord authorization URL did not include state');
  }

  return {
    env,
    keyPair,
    installationId,
    state,
    issueNonce: payload.issue_nonce,
    redirectUri: payload.redirect_uri,
    appVersion: APP_VERSION,
    fingerprintSaltVersion: payload.fingerprint_salt_version,
  };
}

async function signedIssueRequest(
  started: StartedDiscordSession,
  overrides: Partial<SignedDiscordIssueRequestInput> = {},
): Promise<
  SignedDiscordIssueRequestInput & {
    signature_alg: 'ed25519';
    signature: string;
  }
> {
  return signCanonicalDiscordIssueRequest(started.keyPair.privateKey, {
    installation_id: started.installationId,
    device_public_key: started.keyPair.devicePublicKey,
    state: started.state,
    code: 'discord-oauth-code',
    redirect_uri: started.redirectUri,
    hardware_hash: 'hardware-hash-discord-issue',
    hardware_hash_salt_version: started.fingerprintSaltVersion,
    app_version: started.appVersion,
    reason: 'llm_start',
    budget_usd: 0.07,
    model: MODEL,
    issue_nonce: started.issueNonce,
    signed_at: SIGNED_AT_ISO,
    ...overrides,
  });
}

function mockConcurrentDiscordAndOpenRouterApi(
  usersByCode: Record<string, Record<string, unknown>>,
): {
  fetchMock: ReturnType<typeof vi.fn>;
  openRouterCreateCalls: Array<{ input: string | URL; init?: RequestInit }>;
  openRouterGuardrailCalls: Array<{ input: string | URL; init?: RequestInit }>;
} {
  const openRouterCreateCalls: Array<{ input: string | URL; init?: RequestInit }> = [];
  const openRouterGuardrailCalls: Array<{ input: string | URL; init?: RequestInit }> = [];
  const tokenToCode = new Map<string, string>();

  const fetchMock = vi.fn(async (input: string | URL, init?: RequestInit) => {
    const url = String(input);
    const method = init?.method ?? 'GET';

    if (url === DISCORD_TOKEN_URL && method === 'POST') {
      const params = new URLSearchParams(String(init?.body ?? ''));
      const code = params.get('code') ?? '';
      const accessToken = `discord-access-token-for-${code}`;
      tokenToCode.set(accessToken, code);
      return jsonResponse({
        access_token: accessToken,
        token_type: 'Bearer',
      });
    }

    if (url === DISCORD_USER_URL && method === 'GET') {
      const accessToken = String((init?.headers as Record<string, string>)?.authorization ?? '')
        .replace(/^Bearer /u, '');
      const code = tokenToCode.get(accessToken);
      if (!code || !usersByCode[code]) {
        throw new Error(`missing Discord user for token ${accessToken}`);
      }
      return jsonResponse(usersByCode[code]);
    }

    if (url === OPENROUTER_KEYS_URL && method === 'POST') {
      openRouterCreateCalls.push({ input, init });
      const sequence = openRouterCreateCalls.length;
      return jsonResponse(
        {
          key: `or-discord-managed-child-key-concurrent-${sequence}`,
          data: {
            hash: `hash_discord_managed_child_concurrent_${sequence}`,
          },
        },
        201,
      );
    }

    if (url === OPENROUTER_GUARDRAIL_URL && method === 'POST') {
      openRouterGuardrailCalls.push({ input, init });
      return jsonResponse({ assigned_count: 1 });
    }

    throw new Error(`unexpected API request: ${method} ${url}`);
  });

  vi.stubGlobal('fetch', fetchMock as typeof fetch);
  return { fetchMock, openRouterCreateCalls, openRouterGuardrailCalls };
}

function discordSnowflakeForAgeDays(days: number): string {
  return discordSnowflakeForDate(new Date(Date.now() - days * 24 * 60 * 60 * 1000));
}

function discordSnowflakeForDate(createdAt: Date): string {
  const timestamp = BigInt(createdAt.getTime()) - DISCORD_EPOCH_MS;
  return (timestamp << 22n).toString();
}

function countRows(env: TestBrokerEnv, tableExpression: string): number {
  const row = env.__db
    .prepare(`SELECT COUNT(*) AS count FROM ${tableExpression}`)
    .get() as { count: number };
  return Number(row.count);
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      'content-type': 'application/json',
    },
  });
}
