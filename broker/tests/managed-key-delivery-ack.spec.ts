import { afterEach, describe, expect, it, vi } from 'vitest';

import app from '../src/index';
import { sha256Base64Url } from './test-support/hash';
import { mockOpenRouterManagementApi } from './test-support/openrouter-issue';
import { createTestBrokerEnv, type TestBrokerEnv } from './test-support/sqlite-d1';

const ACK_URL = 'http://broker.test/v1/providers/openrouter/managed-key-delivery/ack';
const NOW_ISO = '2026-07-04T12:00:00.000Z';
const EXPIRES_ISO = '2099-07-04T12:15:00.000Z';

describe('managed key delivery ACK route', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it('activates a Discord delivery only after a valid ACK and makes duplicate ACK idempotent', async () => {
    const env = createTestBrokerEnv();
    const token = 'ack-token-discord-test';
    insertDiscordDeliveryPending(env, {
      deliveryId: 'mkd_v1_discord_test',
      ackTokenHash: await ackTokenHash(token),
    });

    const response = await postAck(env, {
      delivery_id: 'mkd_v1_discord_test',
      managed_credential_ref: 'hash_discord_delivery_ack_test',
      delivery_ack_token: token,
    });

    expect(response.status).toBe(200);
    const payload = (await response.json()) as Record<string, unknown>;
    expect(payload).toEqual(
      expect.objectContaining({
        ok: true,
        status: 'acknowledged',
      }),
    );
    expect(readDiscordEntitlement(env)).toEqual(
      expect.objectContaining({
        status: 'active',
        discord_issue_status: 'active',
        discord_issue_delivered_at: expect.any(String),
      }),
    );
    expect(readDelivery(env, 'mkd_v1_discord_test')).toEqual(
      expect.objectContaining({
        status: 'acknowledged',
        acknowledged_at: expect.any(String),
      }),
    );
    expect(countIssueSuccessEvents(env)).toBe(1);

    const duplicateResponse = await postAck(env, {
      delivery_id: 'mkd_v1_discord_test',
      managed_credential_ref: 'hash_discord_delivery_ack_test',
      delivery_ack_token: token,
    });

    expect(duplicateResponse.status).toBe(200);
    await expect(duplicateResponse.json()).resolves.toEqual({
      ok: true,
      status: 'already_acknowledged',
    });
    expect(countIssueSuccessEvents(env)).toBe(1);
  });

  it('activates a QQ delivery only after a valid ACK', async () => {
    const env = createTestBrokerEnv();
    const token = 'ack-token-qq-test';
    insertQqDeliveryPending(env, {
      deliveryId: 'mkd_v1_qq_test',
      ackTokenHash: await ackTokenHash(token),
    });

    const response = await postAck(env, {
      delivery_id: 'mkd_v1_qq_test',
      managed_credential_ref: 'hash_qq_delivery_ack_test',
      delivery_ack_token: token,
    });

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({
      ok: true,
      status: 'acknowledged',
    });
    expect(readQqEntitlement(env)).toEqual(
      expect.objectContaining({
        status: 'active',
        delivered_at: expect.any(String),
      }),
    );
    expect(readDelivery(env, 'mkd_v1_qq_test')).toEqual(
      expect.objectContaining({
        status: 'acknowledged',
        acknowledged_at: expect.any(String),
      }),
    );
    expect(countIssueSuccessEvents(env)).toBe(1);
  });

  it('rejects an invalid ACK token without activating delivery', async () => {
    const env = createTestBrokerEnv();
    insertDiscordDeliveryPending(env, {
      deliveryId: 'mkd_v1_invalid_token_test',
      ackTokenHash: await ackTokenHash('correct-token'),
    });

    const response = await postAck(env, {
      delivery_id: 'mkd_v1_invalid_token_test',
      managed_credential_ref: 'hash_discord_delivery_ack_test',
      delivery_ack_token: 'wrong-token',
    });

    expect(response.status).toBe(401);
    expect(readDiscordEntitlement(env)).toEqual(
      expect.objectContaining({
        status: 'pending_release',
        discord_issue_status: 'delivery_pending',
        discord_issue_delivered_at: null,
      }),
    );
    expect(readDelivery(env, 'mkd_v1_invalid_token_test')).toEqual(
      expect.objectContaining({ status: 'pending' }),
    );
  });

  it('cleans up and releases a Discord delivery when a valid ACK arrives after expiry', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-07-04T12:20:00.000Z'));
    const env = createTestBrokerEnv();
    const token = 'ack-token-expired-discord-test';
    const managementApi = mockOpenRouterManagementApi({
      keyHash: 'hash_discord_delivery_ack_test',
    });
    insertDiscordDeliveryPending(env, {
      deliveryId: 'mkd_v1_expired_discord_test',
      ackTokenHash: await ackTokenHash(token),
      expiresAt: '2026-07-04T12:15:00.000Z',
    });

    const response = await postAck(env, {
      delivery_id: 'mkd_v1_expired_discord_test',
      managed_credential_ref: 'hash_discord_delivery_ack_test',
      delivery_ack_token: token,
    });

    expect(response.status).toBe(409);
    expect(readDiscordEntitlement(env)).toBeUndefined();
    expect(readDelivery(env, 'mkd_v1_expired_discord_test')).toEqual(
      expect.objectContaining({
        status: 'expired',
        failed_at: '2026-07-04T12:20:00.000Z',
        failure_reason: 'delivery_ack_expired',
      }),
    );
    expect(countDiscordIdentities(env)).toBe(0);
    expect(managementApi.fetchMock).toHaveBeenCalledTimes(2);
  });

  it('applies referred bonus budget only when a reserved Discord referral delivery is ACKed', async () => {
    const env = createTestBrokerEnv();
    const token = 'ack-token-discord-referral-test';
    const fetchMock = vi.fn(async (input: string | URL, init?: RequestInit) => {
      const url = String(input);
      const method = init?.method ?? 'GET';
      if (
        url === 'https://openrouter.ai/api/v1/keys/hash_discord_delivery_ack_test' &&
        method === 'PATCH'
      ) {
        return jsonResponse({ data: { hash: 'hash_discord_delivery_ack_test', limit: 0.09 } });
      }
      throw new Error(`unexpected OpenRouter request: ${method} ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);
    insertDiscordDeliveryPending(env, {
      deliveryId: 'mkd_v1_discord_referral_test',
      ackTokenHash: await ackTokenHash(token),
    });
    insertReservedReferralReward(env);

    expect(readDiscordEntitlement(env)).toEqual(
      expect.objectContaining({ budget_usd: 0.07, status: 'pending_release' }),
    );
    expect(readReferralReward(env)).toEqual(
      expect.objectContaining({ referred_bonus_status: 'reserved' }),
    );

    const response = await postAck(env, {
      delivery_id: 'mkd_v1_discord_referral_test',
      managed_credential_ref: 'hash_discord_delivery_ack_test',
      delivery_ack_token: token,
    });

    expect(response.status).toBe(200);
    expect((await response.json()) as Record<string, unknown>).toEqual(
      expect.objectContaining({ referral_bonus_applied: true }),
    );
    expect(readDiscordEntitlement(env)).toEqual(
      expect.objectContaining({ budget_usd: 0.09, status: 'active' }),
    );
    expect(readReferralReward(env)).toEqual(
      expect.objectContaining({
        referred_bonus_status: 'credited',
        referred_managed_credential_ref: 'hash_discord_delivery_ack_test',
      }),
    );
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(JSON.parse(String(fetchMock.mock.calls[0]?.[1]?.body))).toEqual({ limit: 0.09 });
  });

  it('marks an expired delivery cleanup-required when state release rows are missing', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-07-04T12:20:00.000Z'));
    const env = createTestBrokerEnv();
    const token = 'ack-token-expired-release-failure-test';
    const managementApi = mockOpenRouterManagementApi({
      keyHash: 'hash_discord_delivery_ack_test',
    });
    insertDiscordDeliveryPending(env, {
      deliveryId: 'mkd_v1_expired_release_failure_test',
      ackTokenHash: await ackTokenHash(token),
      expiresAt: '2026-07-04T12:15:00.000Z',
    });
    env.__db
      .prepare('DELETE FROM openrouter_entitlements WHERE installation_id = ?')
      .run('install-discord-delivery-ack');

    const response = await postAck(env, {
      delivery_id: 'mkd_v1_expired_release_failure_test',
      managed_credential_ref: 'hash_discord_delivery_ack_test',
      delivery_ack_token: token,
    });

    expect(response.status).toBe(409);
    expect(readDelivery(env, 'mkd_v1_expired_release_failure_test')).toEqual(
      expect.objectContaining({
        status: 'cleanup_required',
        failure_reason: 'state_release_failed',
      }),
    );
    expect(managementApi.fetchMock).toHaveBeenCalledTimes(2);
  });
});

async function postAck(
  env: TestBrokerEnv,
  body: Record<string, unknown>,
): Promise<Response> {
  return app.request(
    ACK_URL,
    {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    },
    env,
  );
}

async function ackTokenHash(token: string): Promise<string> {
  return `sha256-base64url-v1_${await sha256Base64Url(token)}`;
}

function insertDiscordDeliveryPending(
  env: TestBrokerEnv,
  input: { deliveryId: string; ackTokenHash: string; expiresAt?: string },
): void {
  env.__db
    .prepare(
      `INSERT INTO installations (
        installation_id,
        device_public_key,
        app_version
      ) VALUES (?, ?, ?)`,
    )
    .run('install-discord-delivery-ack', 'device-public-key-discord-delivery-ack', '1.0.0');
  env.__db
    .prepare(
      `INSERT INTO discord_identities (
        discord_user_ref,
        entitlement_installation_id,
        status,
        created_at,
        updated_at
      ) VALUES (?, ?, 'issuing', ?, ?)`,
    )
    .run(
      'ph-discord-user-v1_delivery_ack',
      'install-discord-delivery-ack',
      NOW_ISO,
      NOW_ISO,
    );
  env.__db
    .prepare(
      `INSERT INTO openrouter_entitlements (
        installation_id,
        status,
        budget_usd,
        managed_credential_ref,
        issued_at,
        expires_at,
        discord_user_ref,
        discord_issue_status,
        discord_issue_reserved_at,
        discord_issue_delivered_at
      ) VALUES (?, 'pending_release', ?, ?, ?, ?, ?, 'delivery_pending', ?, NULL)`,
    )
    .run(
      'install-discord-delivery-ack',
      0.07,
      'hash_discord_delivery_ack_test',
      NOW_ISO,
      '2026-10-04T12:00:00.000Z',
      'ph-discord-user-v1_delivery_ack',
      NOW_ISO,
    );
  env.__db
    .prepare(
      `INSERT INTO managed_key_deliveries (
        delivery_id,
        issue_source,
        subject_ref,
        installation_id,
        managed_credential_ref,
        ack_token_hash,
        status,
        created_at,
        expires_at
      ) VALUES (?, 'discord', ?, ?, ?, ?, 'pending', ?, ?)`,
    )
    .run(
      input.deliveryId,
      'ph-discord-user-v1_delivery_ack',
      'install-discord-delivery-ack',
      'hash_discord_delivery_ack_test',
      input.ackTokenHash,
      NOW_ISO,
      input.expiresAt ?? EXPIRES_ISO,
    );
}

function insertQqDeliveryPending(
  env: TestBrokerEnv,
  input: { deliveryId: string; ackTokenHash: string },
): void {
  env.__db
    .prepare(
      `INSERT INTO qq_managed_entitlements (
        qq_subject_ref,
        status,
        issue_ref,
        managed_credential_ref,
        budget_usd,
        reserved_at,
        issued_at,
        expires_at,
        delivered_at
      ) VALUES (?, 'delivery_pending', ?, ?, ?, ?, ?, ?, NULL)`,
    )
    .run(
      'ph-qq-subject-v1_delivery_ack',
      'qq-issue-v1_delivery_ack',
      'hash_qq_delivery_ack_test',
      0.07,
      NOW_ISO,
      NOW_ISO,
      '2026-10-04T12:00:00.000Z',
    );
  env.__db
    .prepare(
      `INSERT INTO managed_key_deliveries (
        delivery_id,
        issue_source,
        subject_ref,
        installation_id,
        managed_credential_ref,
        ack_token_hash,
        status,
        created_at,
        expires_at
      ) VALUES (?, 'qq', ?, NULL, ?, ?, 'pending', ?, ?)`,
    )
    .run(
      input.deliveryId,
      'ph-qq-subject-v1_delivery_ack',
      'hash_qq_delivery_ack_test',
      input.ackTokenHash,
      NOW_ISO,
      EXPIRES_ISO,
    );
}

function readDiscordEntitlement(env: TestBrokerEnv): Record<string, unknown> {
  return env.__db
    .prepare(
      `SELECT status, budget_usd, discord_issue_status, discord_issue_delivered_at
         FROM openrouter_entitlements
        WHERE installation_id = ?`,
    )
    .get('install-discord-delivery-ack') as Record<string, unknown>;
}

function readQqEntitlement(env: TestBrokerEnv): Record<string, unknown> {
  return env.__db
    .prepare(
      `SELECT status, delivered_at
         FROM qq_managed_entitlements
        WHERE qq_subject_ref = ?`,
    )
    .get('ph-qq-subject-v1_delivery_ack') as Record<string, unknown>;
}

function readDelivery(env: TestBrokerEnv, deliveryId: string): Record<string, unknown> {
  return env.__db
    .prepare(
      `SELECT status, acknowledged_at, failed_at, failure_reason
         FROM managed_key_deliveries
        WHERE delivery_id = ?`,
    )
    .get(deliveryId) as Record<string, unknown>;
}

function countIssueSuccessEvents(env: TestBrokerEnv): number {
  const row = env.__db
    .prepare('SELECT COUNT(*) AS count FROM broker_issue_success_events')
    .get() as { count: number };
  return Number(row.count);
}

function countDiscordIdentities(env: TestBrokerEnv): number {
  const row = env.__db
    .prepare('SELECT COUNT(*) AS count FROM discord_identities')
    .get() as { count: number };
  return Number(row.count);
}

function insertReservedReferralReward(env: TestBrokerEnv): void {
  env.__db
    .prepare(
      `INSERT INTO referral_rewards (
        referral_id,
        referrer_discord_user_ref,
        referrer_installation_id,
        referred_discord_user_ref,
        referred_installation_id,
        referred_hardware_hash,
        referred_hardware_hash_salt_version,
        referred_bonus_status,
        referrer_bonus_status,
        created_at,
        updated_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, 'reserved', 'pending', ?, ?)`,
    )
    .run(
      '7KQ9M2',
      'ph-discord-user-v1_referrer_delivery_ack',
      'install-referrer-delivery-ack',
      'ph-discord-user-v1_delivery_ack',
      'install-discord-delivery-ack',
      'hardware-hash-delivery-ack',
      7,
      NOW_ISO,
      NOW_ISO,
    );
}

function readReferralReward(env: TestBrokerEnv): Record<string, unknown> {
  return env.__db
    .prepare(
      `SELECT referred_bonus_status, referrer_bonus_status, referred_managed_credential_ref
         FROM referral_rewards
        WHERE referred_installation_id = ?`,
    )
    .get('install-discord-delivery-ack') as Record<string, unknown>;
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}
