import { afterEach, describe, expect, it, vi } from 'vitest';

import app from '../src/index';
import {
  attachReferralToOperation,
  buildManagedOperationId,
  buildManagedOperationResumeToken,
  createManagedOperation,
  hashManagedOperationResumeToken,
} from '../src/managed-operation';
import {
  createDeviceKeyPair,
  signCanonicalDiscordIssueRequest,
  type DeviceKeyPair,
} from './test-support/ed25519';
import { postDiscordIssue, postDiscordStart } from './test-support/trial-api';
import {
  createTestBrokerEnv,
  insertEntitlement,
  type TestBrokerEnv,
} from './test-support/sqlite-d1';

const NOW_ISO = '2026-04-30T06:00:00.000Z';
const SIGNED_AT_ISO = '2026-04-30T06:00:30.000Z';
const REGISTERED_REDIRECT_URI = 'http://127.0.0.1:62187/discord/callback';
const APP_VERSION = '1.2.3';
const MODEL = 'google/gemma-4-26b-a4b-it';
const DISCORD_TOKEN_URL = 'https://discord.com/api/oauth2/token';
const DISCORD_USER_URL = 'https://discord.com/api/users/@me';
const OPENROUTER_KEYS_URL = 'https://openrouter.ai/api/v1/keys';
const OPENROUTER_GUARDRAIL_URL =
  'https://openrouter.ai/api/v1/guardrails/test-managed-guardrail-id/assignments/keys';
const IMMEDIATE_ALERT_URL = 'https://discord.test/immediate-alert';
const QQ_AUTH_ASSERT_URL = 'http://broker.test/v1/auth/qq/assert';
const RESUME_URL = 'http://broker.test/v1/providers/openrouter/managed-operation/resume';
const STATUS_URL = 'http://broker.test/v1/providers/openrouter/managed-operation/status';
const ACK_URL = 'http://broker.test/v1/providers/openrouter/managed-key-delivery/ack';

interface ProviderKeyRecord {
  hash: string;
  limit: number;
}

function jsonResponse(body: unknown, status = 200): Response {
  return Response.json(body, { status });
}

function mockProviderPlane(options: {
  createMode?: 'success' | 'throw-once' | 'fail-400-once';
} = {}) {
  const keys = new Map<string, ProviderKeyRecord>();
  const calls: string[] = [];
  let createCalls = 0;
  let threwOnce = false;
  const fetchMock = vi.fn(async (input: string | URL, init?: RequestInit) => {
    const url = String(input);
    const method = init?.method ?? 'GET';
    calls.push(`${method} ${url}`);
    if (url === OPENROUTER_KEYS_URL && method === 'POST') {
      createCalls += 1;
      if (options.createMode === 'throw-once' && !threwOnce) {
        threwOnce = true;
        throw new TypeError('OpenRouter create response was interrupted');
      }
      if (options.createMode === 'fail-400-once' && createCalls === 1) {
        return jsonResponse({ error: { message: 'bad request' } }, 400);
      }
      const body = JSON.parse(String(init?.body ?? '{}')) as {
        name?: unknown;
        limit?: unknown;
      };
      const name = typeof body.name === 'string' ? body.name : `unnamed-${createCalls}`;
      const limit = typeof body.limit === 'number' ? body.limit : 0.07;
      const hash = `hash_provider_key_${createCalls}`;
      keys.set(name, { hash, limit });
      return jsonResponse({ key: `sk-or-resume-test-${createCalls}`, data: { hash, limit } }, 201);
    }
    if (url.startsWith(`${OPENROUTER_KEYS_URL}?limit=`) && method === 'GET') {
      const entries = [...keys.entries()].map(([name, record]) => ({
        name,
        hash: record.hash,
        limit: record.limit,
      }));
      return jsonResponse({ data: entries });
    }
    const keyMatch = url.match(/\/keys\/([^/?]+)$/u);
    if (keyMatch && method === 'PATCH') {
      const hash = keyMatch[1] ?? '';
      const found = [...keys.entries()].find(([, record]) => record.hash === hash);
      if (!found) {
        return jsonResponse({ error: { message: 'not found' } }, 404);
      }
      return jsonResponse({ data: { hash, disabled: true } });
    }
    if (keyMatch && method === 'DELETE') {
      const hash = keyMatch[1] ?? '';
      const found = [...keys.entries()].find(([, record]) => record.hash === hash);
      if (!found) {
        return jsonResponse({ error: { message: 'not found' } }, 404);
      }
      keys.delete(found[0]);
      return new Response(null, { status: 204 });
    }
    if (url === OPENROUTER_GUARDRAIL_URL && method === 'POST') {
      return jsonResponse({ assigned_count: 1 });
    }
    if (url === IMMEDIATE_ALERT_URL && method === 'POST') {
      return new Response(null, { status: 204 });
    }
    throw new Error(`unexpected provider request: ${method} ${url}`);
  });
  return { keys, calls, fetchMock, get createCalls() { return createCalls; } };
}

function mockDiscordPlane(user: Record<string, unknown>) {
  const calls: string[] = [];
  const fetchMock = vi.fn(async (input: string | URL, init?: RequestInit) => {
    const url = String(input);
    const method = init?.method ?? 'GET';
    calls.push(`${method} ${url}`);
    if (url === DISCORD_TOKEN_URL && method === 'POST') {
      return jsonResponse({ access_token: 'discord-access-token', token_type: 'Bearer' });
    }
    if (url === DISCORD_USER_URL && method === 'GET') {
      return jsonResponse(user);
    }
    throw new Error(`unexpected discord request: ${method} ${url}`);
  });
  return { calls, fetchMock };
}

function stubFetch(handlers: Array<(input: string | URL, init?: RequestInit) => Promise<Response> | Response | null>) {
  const fetchMock = vi.fn(async (input: string | URL, init?: RequestInit) => {
    for (const handler of handlers) {
      const result = await handler(input, init);
      if (result) {
        return result;
      }
    }
    throw new Error(`unexpected request: ${String(init?.method ?? 'GET')} ${String(input)}`);
  });
  vi.stubGlobal('fetch', fetchMock as typeof fetch);
  return fetchMock;
}

function discordSnowflakeForAgeDays(days: number): string {
  const createdAtMs = Date.parse(NOW_ISO) - days * 24 * 60 * 60 * 1000;
  const discordEpochMs = 1420070400000;
  const snowflake = (BigInt(createdAtMs - discordEpochMs) << 22n).toString();
  return snowflake;
}

interface StartedSession {
  env: TestBrokerEnv;
  keyPair: DeviceKeyPair;
  installationId: string;
  state: string;
  issueNonce: string;
  redirectUri: string;
  appVersion: string;
  fingerprintSaltVersion: number;
}

async function startDiscordSession(installationId: string, env: TestBrokerEnv): Promise<StartedSession> {
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

async function postResume(
  env: TestBrokerEnv,
  body: Record<string, unknown>,
): Promise<Response> {
  return app.request(
    RESUME_URL,
    {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    },
    env,
  );
}

async function postStatus(
  env: TestBrokerEnv,
  body: Record<string, unknown>,
): Promise<Response> {
  return app.request(
    STATUS_URL,
    {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    },
    env,
  );
}

function readOperation(env: TestBrokerEnv, operationId: string): Record<string, unknown> {
  const row = env.__db
    .prepare('SELECT * FROM managed_operations WHERE operation_id = ?')
    .get(operationId) as Record<string, unknown>;
  if (!row) {
    throw new Error('operation row missing');
  }
  return row;
}

function readAttempts(env: TestBrokerEnv, operationId: string): Array<Record<string, unknown>> {
  return env.__db
    .prepare('SELECT * FROM managed_operation_attempts WHERE operation_id = ? ORDER BY attempt_index ASC')
    .all(operationId) as Array<Record<string, unknown>>;
}

function readDeliveries(env: TestBrokerEnv, operationId: string): Array<Record<string, unknown>> {
  return env.__db
    .prepare('SELECT * FROM managed_key_deliveries WHERE operation_id = ? ORDER BY delivery_id ASC')
    .all(operationId) as Array<Record<string, unknown>>;
}

describe('managed operation resume issuance', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it('restarts a timed-out Discord issue via resume without repeating OAuth', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const env = createTestBrokerEnv();
    const started = await startDiscordSession('install-discord-resume-restart', env);
    const discordUserId = discordSnowflakeForAgeDays(31);
    const provider = mockProviderPlane({ createMode: 'throw-once' });
    const discord = mockDiscordPlane({ id: discordUserId, verified: true });
    stubFetch([
      (input, init) => {
        const url = String(input);
        return url.startsWith('https://discord.com/') ? discord.fetchMock(input, init) : null;
      },
      (input, init) => provider.fetchMock(input, init),
    ]);

    const operationId = buildManagedOperationId();
    const resumeToken = buildManagedOperationResumeToken();
    const signed = await signCanonicalDiscordIssueRequest(started.keyPair.privateKey, {
      installation_id: started.installationId,
      device_public_key: started.keyPair.devicePublicKey,
      state: started.state,
      code: 'discord-oauth-code-resume-restart',
      redirect_uri: started.redirectUri,
      hardware_hash: 'hardware-hash-resume-restart',
      hardware_hash_salt_version: started.fingerprintSaltVersion,
      app_version: started.appVersion,
      reason: 'llm_start',
      budget_usd: 0.07,
      model: MODEL,
      issue_nonce: started.issueNonce,
      signed_at: SIGNED_AT_ISO,
    });
    const issueResponse = await postDiscordIssue(env, {
      ...signed,
      delivery_ack_supported: true,
      operation_id: operationId,
      resume_token: resumeToken,
    });
    expect(issueResponse.status).toBe(500);

    const attemptsAfterFailure = readAttempts(env, operationId);
    expect(attemptsAfterFailure).toHaveLength(1);
    expect(attemptsAfterFailure[0]).toMatchObject({ attempt_index: 1, outcome: 'cleaned' });

    const statusResponse = await postStatus(env, {
      operation_id: operationId,
      resume_token: resumeToken,
      installation_id: started.installationId,
    });
    expect(statusResponse.status).toBe(200);
    await expect(statusResponse.json()).resolves.toMatchObject({
      ok: true,
      state: 'RETRY_READY',
      client_action: 'retry_authorized',
    });

    const discordCallsBeforeResume = discord.calls.length;
    const resumeResponse = await postResume(env, {
      operation_id: operationId,
      resume_token: resumeToken,
      installation_id: started.installationId,
    });
    expect(resumeResponse.status).toBe(200);
    const resumePayload = (await resumeResponse.json()) as Record<string, unknown>;
    expect(resumePayload).toMatchObject({
      delivery_ack_required: true,
      delivery_id: expect.any(String),
      delivery_ack_token: expect.any(String),
      managed_credential_ref: expect.any(String),
    });
    expect(typeof resumePayload.openrouter_api_key).toBe('string');
    expect(discord.calls.length).toBe(discordCallsBeforeResume);

    const attempts = readAttempts(env, operationId);
    expect(attempts).toHaveLength(2);
    expect(attempts[1]).toMatchObject({ attempt_index: 2, outcome: 'created' });
    expect(readOperation(env, operationId)).toMatchObject({ state: 'DELIVERY_PENDING' });
    const creates = provider.calls.filter((call) => call === `POST ${OPENROUTER_KEYS_URL}`);
    expect(creates).toHaveLength(2);

    const ackResponse = await app.request(
      ACK_URL,
      {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          delivery_id: resumePayload.delivery_id,
          managed_credential_ref: resumePayload.managed_credential_ref,
          delivery_ack_token: resumePayload.delivery_ack_token,
        }),
      },
      env,
    );
    expect(ackResponse.status).toBe(200);
    expect(readOperation(env, operationId)).toMatchObject({ state: 'ACTIVE' });
  });

  it('lets exactly one concurrent resume create the fresh key', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const env = createTestBrokerEnv();
    const installationId = 'install-discord-resume-race';
    const devicePublicKey = 'device-public-key-resume-race';
    env.__db
      .prepare(
        `INSERT INTO installations (
          installation_id, device_public_key, hardware_hash, hardware_hash_salt_version,
          app_version, created_at, last_seen_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(installationId, devicePublicKey, 'hardware-hash-resume-race', 1, APP_VERSION, NOW_ISO, NOW_ISO);
    const subjectRef = 'ph-discord-user-v1_resume_race_subject';
    env.__db
      .prepare(
        `INSERT INTO discord_identities (discord_user_ref, entitlement_installation_id, status, created_at, updated_at)
         VALUES (?, ?, 'issuing', ?, ?)`,
      )
      .run(subjectRef, installationId, NOW_ISO, NOW_ISO);
    insertEntitlement(env, {
      installation_id: installationId,
      status: 'pending_release',
      budget_usd: 0.07,
      discord_user_ref: subjectRef,
      discord_issue_status: 'issuing',
      discord_issue_reserved_at: NOW_ISO,
    });
    const operationId = buildManagedOperationId();
    const resumeToken = buildManagedOperationResumeToken();
    env.__db
      .prepare(
        `INSERT INTO managed_operations (
          operation_id, issue_source, subject_ref, installation_id, device_public_key,
          state, attempt_count, current_attempt_index, resume_token_hash, auth_expires_at,
          failure_reason, client_action, referral_reward_id, referral_status, settlement_status,
          hardware_hash, hardware_hash_salt_version, app_version,
          created_at, updated_at, last_reconciled_at, cleanup_attempts
        ) VALUES (?, 'discord', ?, ?, ?, 'RETRY_READY', 1, 1, ?, ?, NULL, 'retry_authorized',
          NULL, 'none', 'none', ?, 1, ?, ?, ?, NULL, 0)`,
      )
      .run(
        operationId,
        subjectRef,
        installationId,
        devicePublicKey,
        await hashManagedOperationResumeToken(resumeToken),
        new Date(Date.parse(NOW_ISO) + 60 * 60_000).toISOString(),
        'hardware-hash-resume-race',
        APP_VERSION,
        NOW_ISO,
        NOW_ISO,
      );
    env.__db
      .prepare(
        `INSERT INTO managed_operation_attempts (
          operation_id, attempt_index, provider_key_name, managed_credential_ref, outcome, created_at, updated_at
        ) VALUES (?, 1, ?, NULL, 'cleaned', ?, ?)`,
      )
      .run(operationId, `puripuly-heart:mop:discord:resume_race:a1`, NOW_ISO, NOW_ISO);

    const provider = mockProviderPlane();
    stubFetch([(input, init) => provider.fetchMock(input, init)]);
    const body = {
      operation_id: operationId,
      resume_token: resumeToken,
      installation_id: installationId,
    };
    const [first, second] = await Promise.all([postResume(env, body), postResume(env, body)]);
    expect(first.status).toBe(200);
    expect(second.status).toBe(200);
    const firstPayload = (await first.json()) as Record<string, unknown>;
    const secondPayload = (await second.json()) as Record<string, unknown>;
    const withKey = [firstPayload, secondPayload].filter((payload) => typeof payload.openrouter_api_key === 'string');
    const withoutKey = [firstPayload, secondPayload].filter((payload) => typeof payload.openrouter_api_key !== 'string');
    expect(withKey).toHaveLength(1);
    expect(withoutKey).toHaveLength(1);
    expect(provider.createCalls).toBe(1);
    expect(readDeliveries(env, operationId)).toHaveLength(1);
    expect(readOperation(env, operationId)).toMatchObject({ state: 'DELIVERY_PENDING', attempt_count: 2 });
  });

  it('reconciles an ambiguous orphan before creating the fresh key', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const env = createTestBrokerEnv();
    const installationId = 'install-discord-resume-orphan';
    const devicePublicKey = 'device-public-key-resume-orphan';
    env.__db
      .prepare(
        `INSERT INTO installations (
          installation_id, device_public_key, hardware_hash, hardware_hash_salt_version,
          app_version, created_at, last_seen_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(installationId, devicePublicKey, 'hardware-hash-resume-orphan', 1, APP_VERSION, NOW_ISO, NOW_ISO);
    const subjectRef = 'ph-discord-user-v1_resume_orphan_subject';
    env.__db
      .prepare(
        `INSERT INTO discord_identities (discord_user_ref, entitlement_installation_id, status, created_at, updated_at)
         VALUES (?, ?, 'issuing', ?, ?)`,
      )
      .run(subjectRef, installationId, NOW_ISO, NOW_ISO);
    insertEntitlement(env, {
      installation_id: installationId,
      status: 'pending_release',
      budget_usd: 0.07,
      discord_user_ref: subjectRef,
      discord_issue_status: 'issuing',
      discord_issue_reserved_at: NOW_ISO,
    });
    const operationId = buildManagedOperationId();
    const resumeToken = buildManagedOperationResumeToken();
    env.__db
      .prepare(
        `INSERT INTO managed_operations (
          operation_id, issue_source, subject_ref, installation_id, device_public_key,
          state, attempt_count, current_attempt_index, resume_token_hash, auth_expires_at,
          failure_reason, client_action, referral_reward_id, referral_status, settlement_status,
          hardware_hash, hardware_hash_salt_version, app_version,
          created_at, updated_at, last_reconciled_at, cleanup_attempts
        ) VALUES (?, 'discord', ?, ?, ?, 'CREATE_UNKNOWN', 1, 1, ?, ?, NULL, 'wait',
          NULL, 'none', 'none', ?, 1, ?, ?, ?, NULL, 0)`,
      )
      .run(
        operationId,
        subjectRef,
        installationId,
        devicePublicKey,
        await hashManagedOperationResumeToken(resumeToken),
        new Date(Date.parse(NOW_ISO) + 60 * 60_000).toISOString(),
        'hardware-hash-resume-orphan',
        APP_VERSION,
        NOW_ISO,
        NOW_ISO,
      );
    const orphanName = `puripuly-heart:mop:discord:${operationId.slice('ph-mop-v1_'.length).slice(0, 16)}:a1`;
    env.__db
      .prepare(
        `INSERT INTO managed_operation_attempts (
          operation_id, attempt_index, provider_key_name, managed_credential_ref, outcome, created_at, updated_at
        ) VALUES (?, 1, ?, NULL, 'unknown', ?, ?)`,
      )
      .run(operationId, orphanName, NOW_ISO, NOW_ISO);

    const provider = mockProviderPlane();
    provider.keys.set(orphanName, { hash: 'hash_orphan_attempt_1', limit: 0.07 });
    stubFetch([(input, init) => provider.fetchMock(input, init)]);

    const resumeResponse = await postResume(env, {
      operation_id: operationId,
      resume_token: resumeToken,
      installation_id: installationId,
    });
    expect(resumeResponse.status).toBe(200);
    const payload = (await resumeResponse.json()) as Record<string, unknown>;
    expect(typeof payload.openrouter_api_key).toBe('string');

    const deleteIndex = provider.calls.findIndex((call) => call.startsWith('DELETE '));
    const secondCreateIndex = provider.calls.reduce((found, call, index) => {
      if (call === `POST ${OPENROUTER_KEYS_URL}` && found === -1 && index !== 0) {
        return index;
      }
      return found;
    }, -1);
    expect(provider.calls.filter((call) => call === `POST ${OPENROUTER_KEYS_URL}`)).toHaveLength(1);
    expect(deleteIndex).toBeGreaterThanOrEqual(0);
    expect(deleteIndex).toBeLessThan(secondCreateIndex === -1 ? provider.calls.length : secondCreateIndex);
    expect(provider.keys.has(orphanName)).toBe(false);

    const attempts = readAttempts(env, operationId);
    expect(attempts).toHaveLength(2);
    expect(attempts[0]).toMatchObject({ attempt_index: 1, outcome: 'cleaned' });
    expect(attempts[1]).toMatchObject({ attempt_index: 2, outcome: 'created' });
  });

  it('reuses the existing operation referral instead of reserving again', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const env = createTestBrokerEnv();
    const installationId = 'install-discord-resume-referral';
    const devicePublicKey = 'device-public-key-resume-referral';
    env.__db
      .prepare(
        `INSERT INTO installations (
          installation_id, device_public_key, hardware_hash, hardware_hash_salt_version,
          app_version, created_at, last_seen_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(installationId, devicePublicKey, 'hardware-hash-resume-referral', 1, APP_VERSION, NOW_ISO, NOW_ISO);
    const subjectRef = 'ph-discord-user-v1_resume_referral_subject';
    env.__db
      .prepare(
        `INSERT INTO discord_identities (discord_user_ref, entitlement_installation_id, status, created_at, updated_at)
         VALUES (?, ?, 'issuing', ?, ?)`,
      )
      .run(subjectRef, installationId, NOW_ISO, NOW_ISO);
    insertEntitlement(env, {
      installation_id: installationId,
      status: 'pending_release',
      budget_usd: 0.07,
      discord_user_ref: subjectRef,
      discord_issue_status: 'issuing',
      discord_issue_reserved_at: NOW_ISO,
    });
    const operationId = buildManagedOperationId();
    const resumeToken = buildManagedOperationResumeToken();
    env.__db
      .prepare(
        `INSERT INTO managed_operations (
          operation_id, issue_source, subject_ref, installation_id, device_public_key,
          state, attempt_count, current_attempt_index, resume_token_hash, auth_expires_at,
          failure_reason, client_action, referral_reward_id, referral_status, settlement_status,
          hardware_hash, hardware_hash_salt_version, app_version,
          created_at, updated_at, last_reconciled_at, cleanup_attempts
        ) VALUES (?, 'discord', ?, ?, ?, 'RETRY_READY', 1, 1, ?, ?, NULL, 'retry_authorized',
          NULL, 'none', 'none', ?, 1, ?, ?, ?, NULL, 0)`,
      )
      .run(
        operationId,
        subjectRef,
        installationId,
        devicePublicKey,
        await hashManagedOperationResumeToken(resumeToken),
        new Date(Date.parse(NOW_ISO) + 60 * 60_000).toISOString(),
        'hardware-hash-resume-referral',
        APP_VERSION,
        NOW_ISO,
        NOW_ISO,
      );
    env.__db
      .prepare(
        `INSERT INTO managed_operation_attempts (
          operation_id, attempt_index, provider_key_name, managed_credential_ref, outcome, created_at, updated_at
        ) VALUES (?, 1, ?, NULL, 'cleaned', ?, ?)`,
      )
      .run(operationId, `puripuly-heart:mop:discord:resume_referral:a1`, NOW_ISO, NOW_ISO);
    env.__db
      .prepare(
        `INSERT INTO referral_rewards (
          referral_id, referrer_source, referrer_subject_ref, referrer_installation_id,
          referred_source, referred_subject_ref, referred_installation_id,
          referred_hardware_hash, referred_hardware_hash_salt_version,
          referred_bonus_status, referrer_bonus_status, operation_id, created_at, updated_at
        ) VALUES (?, 'discord', ?, ?, 'discord', ?, ?, ?, 1, 'reserved', 'pending', ?, ?, ?)`,
      )
      .run(
        'ABCDEF',
        'ph-discord-user-v1_resume_referrer',
        'install-resume-referrer',
        subjectRef,
        installationId,
        'hardware-hash-resume-referral',
        operationId,
        NOW_ISO,
        NOW_ISO,
      );
    const rewardId = Number(
      (env.__db.prepare('SELECT last_insert_rowid() AS id').get() as { id: number }).id,
    );
    await attachReferralToOperation(env.BROKER_DB, operationId, rewardId, 'reserved', 'none', new Date(NOW_ISO));

    const provider = mockProviderPlane();
    stubFetch([(input, init) => provider.fetchMock(input, init)]);
    const resumeResponse = await postResume(env, {
      operation_id: operationId,
      resume_token: resumeToken,
      installation_id: installationId,
    });
    expect(resumeResponse.status).toBe(200);
    const payload = (await resumeResponse.json()) as Record<string, unknown>;
    expect(typeof payload.openrouter_api_key).toBe('string');
    const resumeCreateBody = JSON.parse(
      String(provider.fetchMock.mock.calls.find(([input]) => String(input) === OPENROUTER_KEYS_URL)?.[1]?.body ?? '{}'),
    ) as Record<string, unknown>;
    expect(resumeCreateBody.limit).toBe(0.07);

    const rewards = env.__db
      .prepare('SELECT * FROM referral_rewards WHERE operation_id = ?')
      .all(operationId) as Array<Record<string, unknown>>;
    expect(rewards).toHaveLength(1);
    expect(rewards[0]).toMatchObject({ id: rewardId, referred_bonus_status: 'reserved' });
    expect(readOperation(env, operationId)).toMatchObject({
      referral_reward_id: rewardId,
      referral_status: 'reserved',
    });
  });

  it('resumes a timed-out QQ issue without the raw identity or credential', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const env = createTestBrokerEnv();
    const provider = mockProviderPlane({ createMode: 'throw-once' });
    stubFetch([(input, init) => provider.fetchMock(input, init)]);

    const operationId = buildManagedOperationId();
    const resumeToken = buildManagedOperationResumeToken();
    const qqIdentity = 'qq-openid-resume-restart';
    const key = await crypto.subtle.importKey(
      'raw',
      new TextEncoder().encode(env.QQ_AUTH_HMAC_PSK),
      { name: 'HMAC', hash: 'SHA-256' },
      false,
      ['sign'],
    );
    const signature = await crypto.subtle.sign('HMAC', key, new TextEncoder().encode(qqIdentity));
    const credential = Array.from(new Uint8Array(signature), (value) =>
      value.toString(16).padStart(2, '0'),
    ).join('');
    const assertResponse = await app.request(
      QQ_AUTH_ASSERT_URL,
      {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          qq_identity: qqIdentity,
          credential,
          asserted_at: NOW_ISO,
          delivery_ack_supported: true,
          installation_id: 'install-qq-resume-restart',
          operation_id: operationId,
          resume_token: resumeToken,
        }),
      },
      env,
    );
    expect(assertResponse.status).toBe(500);

    const statusResponse = await postStatus(env, {
      operation_id: operationId,
      resume_token: resumeToken,
      installation_id: 'install-qq-resume-restart',
    });
    await expect(statusResponse.json()).resolves.toMatchObject({
      ok: true,
      state: 'RETRY_READY',
      issue_source: 'qq',
    });

    const resumeResponse = await postResume(env, {
      operation_id: operationId,
      resume_token: resumeToken,
      installation_id: 'install-qq-resume-restart',
    });
    expect(resumeResponse.status).toBe(200);
    const payload = (await resumeResponse.json()) as Record<string, unknown>;
    expect(payload).toMatchObject({
      ok: true,
      status: 'delivery_pending',
      delivery_ack_required: true,
      delivery_id: expect.any(String),
      delivery_ack_token: expect.any(String),
      managed_credential_ref: expect.any(String),
    });
    expect(typeof payload.openrouter_api_key).toBe('string');
    expect(JSON.stringify(payload)).not.toContain(qqIdentity);
    expect(JSON.stringify(payload)).not.toContain(credential);

    const creates = provider.calls.filter((call) => call === `POST ${OPENROUTER_KEYS_URL}`);
    expect(creates).toHaveLength(2);
    const secondCreateBody = JSON.parse(
      String(
        (await provider.fetchMock.mock.calls.find(
          ([input, init]) =>
            String(input) === OPENROUTER_KEYS_URL &&
            (init?.method ?? 'GET') === 'POST' &&
            String(init?.body ?? '').includes(':a2'),
        )?.[1]?.body ?? '{}'),
      ),
    ) as Record<string, unknown>;
    expect(secondCreateBody.name).toContain(':a2');
    expect(readOperation(env, operationId)).toMatchObject({ state: 'DELIVERY_PENDING', attempt_count: 2 });
  });

  it('treats a repeated resume after delivery as a wait without a new key', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const env = createTestBrokerEnv();
    const installationId = 'install-discord-resume-repeat';
    const devicePublicKey = 'device-public-key-resume-repeat';
    env.__db
      .prepare(
        `INSERT INTO installations (
          installation_id, device_public_key, hardware_hash, hardware_hash_salt_version,
          app_version, created_at, last_seen_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(installationId, devicePublicKey, 'hardware-hash-resume-repeat', 1, APP_VERSION, NOW_ISO, NOW_ISO);
    const subjectRef = 'ph-discord-user-v1_resume_repeat_subject';
    env.__db
      .prepare(
        `INSERT INTO discord_identities (discord_user_ref, entitlement_installation_id, status, created_at, updated_at)
         VALUES (?, ?, 'issuing', ?, ?)`,
      )
      .run(subjectRef, installationId, NOW_ISO, NOW_ISO);
    insertEntitlement(env, {
      installation_id: installationId,
      status: 'pending_release',
      budget_usd: 0.07,
      discord_user_ref: subjectRef,
      discord_issue_status: 'issuing',
      discord_issue_reserved_at: NOW_ISO,
    });
    const operationId = buildManagedOperationId();
    const resumeToken = buildManagedOperationResumeToken();
    env.__db
      .prepare(
        `INSERT INTO managed_operations (
          operation_id, issue_source, subject_ref, installation_id, device_public_key,
          state, attempt_count, current_attempt_index, resume_token_hash, auth_expires_at,
          failure_reason, client_action, referral_reward_id, referral_status, settlement_status,
          hardware_hash, hardware_hash_salt_version, app_version,
          created_at, updated_at, last_reconciled_at, cleanup_attempts
        ) VALUES (?, 'discord', ?, ?, ?, 'RETRY_READY', 1, 1, ?, ?, NULL, 'retry_authorized',
          NULL, 'none', 'none', ?, 1, ?, ?, ?, NULL, 0)`,
      )
      .run(
        operationId,
        subjectRef,
        installationId,
        devicePublicKey,
        await hashManagedOperationResumeToken(resumeToken),
        new Date(Date.parse(NOW_ISO) + 60 * 60_000).toISOString(),
        'hardware-hash-resume-repeat',
        APP_VERSION,
        NOW_ISO,
        NOW_ISO,
      );
    env.__db
      .prepare(
        `INSERT INTO managed_operation_attempts (
          operation_id, attempt_index, provider_key_name, managed_credential_ref, outcome, created_at, updated_at
        ) VALUES (?, 1, ?, NULL, 'cleaned', ?, ?)`,
      )
      .run(operationId, `puripuly-heart:mop:discord:resume_repeat:a1`, NOW_ISO, NOW_ISO);

    const provider = mockProviderPlane();
    stubFetch([(input, init) => provider.fetchMock(input, init)]);
    const body = {
      operation_id: operationId,
      resume_token: resumeToken,
      installation_id: installationId,
    };
    const first = await postResume(env, body);
    expect(first.status).toBe(200);
    const firstPayload = (await first.json()) as Record<string, unknown>;
    expect(typeof firstPayload.openrouter_api_key).toBe('string');

    const second = await postResume(env, body);
    expect(second.status).toBe(200);
    const secondPayload = (await second.json()) as Record<string, unknown>;
    expect(secondPayload).not.toHaveProperty('openrouter_api_key');
    expect(secondPayload).toMatchObject({ state: 'DELIVERY_PENDING', client_action: 'acknowledge_delivery' });
    expect(provider.createCalls).toBe(1);
    expect(readAttempts(env, operationId)).toHaveLength(2);
  });

  it('marks a resume-time ambiguous create unknown instead of blind-retrying', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const env = createTestBrokerEnv();
    const installationId = 'install-discord-resume-ambiguous';
    const devicePublicKey = 'device-public-key-resume-ambiguous';
    env.__db
      .prepare(
        `INSERT INTO installations (
          installation_id, device_public_key, hardware_hash, hardware_hash_salt_version,
          app_version, created_at, last_seen_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(installationId, devicePublicKey, 'hardware-hash-resume-ambiguous', 1, APP_VERSION, NOW_ISO, NOW_ISO);
    const subjectRef = 'ph-discord-user-v1_resume_ambiguous_subject';
    env.__db
      .prepare(
        `INSERT INTO discord_identities (discord_user_ref, entitlement_installation_id, status, created_at, updated_at)
         VALUES (?, ?, 'issuing', ?, ?)`,
      )
      .run(subjectRef, installationId, NOW_ISO, NOW_ISO);
    insertEntitlement(env, {
      installation_id: installationId,
      status: 'pending_release',
      budget_usd: 0.07,
      discord_user_ref: subjectRef,
      discord_issue_status: 'issuing',
      discord_issue_reserved_at: NOW_ISO,
    });
    const operationId = buildManagedOperationId();
    const resumeToken = buildManagedOperationResumeToken();
    env.__db
      .prepare(
        `INSERT INTO managed_operations (
          operation_id, issue_source, subject_ref, installation_id, device_public_key,
          state, attempt_count, current_attempt_index, resume_token_hash, auth_expires_at,
          failure_reason, client_action, referral_reward_id, referral_status, settlement_status,
          hardware_hash, hardware_hash_salt_version, app_version,
          created_at, updated_at, last_reconciled_at, cleanup_attempts
        ) VALUES (?, 'discord', ?, ?, ?, 'RETRY_READY', 1, 1, ?, ?, NULL, 'retry_authorized',
          NULL, 'none', 'none', ?, 1, ?, ?, ?, NULL, 0)`,
      )
      .run(
        operationId,
        subjectRef,
        installationId,
        devicePublicKey,
        await hashManagedOperationResumeToken(resumeToken),
        new Date(Date.parse(NOW_ISO) + 60 * 60_000).toISOString(),
        'hardware-hash-resume-ambiguous',
        APP_VERSION,
        NOW_ISO,
        NOW_ISO,
      );
    env.__db
      .prepare(
        `INSERT INTO managed_operation_attempts (
          operation_id, attempt_index, provider_key_name, managed_credential_ref, outcome, created_at, updated_at
        ) VALUES (?, 1, ?, NULL, 'cleaned', ?, ?)`,
      )
      .run(operationId, `puripuly-heart:mop:discord:resume_ambiguous:a1`, NOW_ISO, NOW_ISO);

    const provider = mockProviderPlane({ createMode: 'throw-once' });
    stubFetch([(input, init) => provider.fetchMock(input, init)]);
    const body = {
      operation_id: operationId,
      resume_token: resumeToken,
      installation_id: installationId,
    };
    const ambiguous = await postResume(env, body);
    expect(ambiguous.status).toBe(200);
    const ambiguousPayload = (await ambiguous.json()) as Record<string, unknown>;
    expect(ambiguousPayload).not.toHaveProperty('openrouter_api_key');
    expect(ambiguousPayload).toMatchObject({ state: 'CREATE_UNKNOWN', client_action: 'wait' });
    expect(readAttempts(env, operationId)).toHaveLength(2);
    expect(readDeliveries(env, operationId)).toHaveLength(0);

    const retry = await postResume(env, body);
    expect(retry.status).toBe(200);
    const retryPayload = (await retry.json()) as Record<string, unknown>;
    expect(typeof retryPayload.openrouter_api_key).toBe('string');
    expect(provider.createCalls).toBe(2);
    expect(readAttempts(env, operationId)).toHaveLength(3);
  });

  it('keeps status read-only while resume progresses recovery', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const env = createTestBrokerEnv();
    const operationId = buildManagedOperationId();
    const resumeToken = buildManagedOperationResumeToken();
    await createManagedOperation(env.BROKER_DB, {
      operationId,
      resumeTokenHash: await hashManagedOperationResumeToken(resumeToken),
      issueSource: 'discord',
      subjectRef: 'ph-discord-user-v1_status_readonly',
      installationId: 'install-status-readonly',
      devicePublicKey: 'device-status-readonly',
      now: new Date(NOW_ISO),
    });
    const before = readOperation(env, operationId);
    const first = await postStatus(env, {
      operation_id: operationId,
      resume_token: resumeToken,
      installation_id: 'install-status-readonly',
    });
    const second = await postStatus(env, {
      operation_id: operationId,
      resume_token: resumeToken,
      installation_id: 'install-status-readonly',
    });
    expect(first.status).toBe(200);
    expect(second.status).toBe(200);
    await expect(first.json()).resolves.toEqual(await second.json());
    expect(readOperation(env, operationId)).toEqual(before);
    expect(readAttempts(env, operationId)).toHaveLength(0);
  });
});
