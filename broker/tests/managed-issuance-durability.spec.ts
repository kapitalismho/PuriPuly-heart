import { DatabaseSync } from 'node:sqlite';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { checkEndpointRateLimit } from '../src/abuse-controls';
import {
  acknowledgeManagedKeyDelivery,
  createManagedKeyDelivery,
} from '../src/managed-key-delivery';
import { processManagedReferralSettlementJobs } from '../src/managed-referral-settlement';
import { expireManagedOperation } from '../src/managed-operation';
import {
  deriveStableNetworkIdentityDigest,
  normalizeNetworkIdentityIp,
  resolveNetworkIdentitySecrets,
  resolveRequestNetworkIdentity,
} from '../src/network-identity';
import { runNetworkIdentityBackfill } from '../src/network-identity-migration';
import { reserveIssueReferralReward } from '../src/referral';
import {
  applyBrokerMigrations,
  readBrokerMigrationSql,
} from './test-support/migrations';
import {
  createTestBrokerEnv,
  testNetworkIdentitySecrets,
  type TestBrokerEnv,
} from './test-support/sqlite-d1';

const NOW_ISO = '2026-09-01T10:00:00.000Z';
const NOW = new Date(NOW_ISO);

function mockProviderKeyApi(limits: Map<string, number>) {
  const calls: Array<{ url: string; method: string; body: string }> = [];
  const fetchMock = vi.fn(async (input: string | URL, init?: RequestInit) => {
    const url = String(input);
    const method = init?.method ?? 'GET';
    const body = String(init?.body ?? '');
    calls.push({ url, method, body });
    const match = url.match(/\/keys\/([^/?]+)$/u);
    if (match && method === 'GET') {
      const hash = match[1] ?? '';
      return Response.json({ data: { hash, limit: limits.get(hash) ?? 0.07 } });
    }
    if (match && method === 'PATCH') {
      const hash = match[1] ?? '';
      const parsed = JSON.parse(body || '{}') as { limit?: unknown };
      if (typeof parsed.limit === 'number') {
        limits.set(hash, parsed.limit);
      }
      return Response.json({ data: { hash, limit: limits.get(hash) ?? 0.07, disabled: true } });
    }
    if (match && method === 'DELETE') {
      const hash = match[1] ?? '';
      limits.delete(hash);
      return new Response(null, { status: 204 });
    }
    throw new Error(`unexpected provider request: ${method} ${url}`);
  });
  vi.stubGlobal('fetch', fetchMock as typeof fetch);
  return { calls };
}

describe('managed issuance durability', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it('treats the exact ACK expiry instant as valid and one millisecond later as expired', async () => {
    const env = createTestBrokerEnv();
    const createdAt = new Date('2026-09-01T10:00:00.000Z');
    const expiresAt = new Date(createdAt.getTime() + 15 * 60_000);
    const delivery = await createManagedKeyDelivery(env.BROKER_DB, {
      issueSource: 'discord',
      subjectRef: 'ph-discord-user-v1_ack_boundary',
      installationId: 'install-ack-boundary',
      managedCredentialRef: 'hash_ack_boundary',
      createdAt,
      expiresAt,
    });

    await expect(
      acknowledgeManagedKeyDelivery(env.BROKER_DB, {
        deliveryId: delivery.deliveryId,
        managedCredentialRef: 'hash_ack_boundary',
        deliveryAckToken: delivery.deliveryAckToken,
        now: expiresAt,
      }),
    ).resolves.toEqual({ ok: true, status: 'acknowledged' });

    const second = await createManagedKeyDelivery(env.BROKER_DB, {
      issueSource: 'discord',
      subjectRef: 'ph-discord-user-v1_ack_boundary_late',
      installationId: 'install-ack-boundary-late',
      managedCredentialRef: 'hash_ack_boundary_late',
      createdAt,
      expiresAt,
    });
    await expect(
      acknowledgeManagedKeyDelivery(env.BROKER_DB, {
        deliveryId: second.deliveryId,
        managedCredentialRef: 'hash_ack_boundary_late',
        deliveryAckToken: second.deliveryAckToken,
        now: new Date(expiresAt.getTime() + 1),
      }),
    ).resolves.toEqual({ ok: false, reason: 'expired' });
  });

  it('dual-compares rotated secrets without weakening per-IP limits', async () => {
    const env = createTestBrokerEnv();
    env.NETWORK_IDENTITY_HMAC_SECRET = 'new-secret';
    env.NETWORK_IDENTITY_HMAC_SECRET_PREVIOUS = 'old-secret';
    (env as unknown as Record<string, unknown>).NETWORK_IDENTITY_HMAC_KEY_VERSION = '2';
    (env as unknown as Record<string, unknown>).NETWORK_IDENTITY_HMAC_KEY_VERSION_PREVIOUS = '1';

    const secrets = resolveNetworkIdentitySecrets(env);
    expect(secrets).toMatchObject({
      current: 'new-secret',
      previous: 'old-secret',
      previousVersion: 1,
      currentVersion: 2,
    });

    const oldIdentity = await resolveRequestNetworkIdentity(
      '203.0.113.90',
      { current: 'old-secret', previous: null, previousVersion: null, currentVersion: 1 },
      NOW,
    );
    expect(oldIdentity?.keyVersion).toBe(1);
    env.__db
      .prepare(
        `INSERT INTO broker_request_events (
          endpoint, ip_digest, ip_key_version, ip_epoch, installation_id, observed_at
        ) VALUES (?, ?, ?, ?, ?, ?)`,
      )
      .run(
        'POST /v1/auth/qq/assert',
        oldIdentity?.digest ?? '',
        1,
        '2026-09-01',
        null,
        '2026-09-01T09:59:00.000Z',
      );
    env.__db
      .prepare(
        `INSERT INTO broker_request_events (
          endpoint, ip_digest, ip_key_version, ip_epoch, installation_id, observed_at
        ) VALUES (?, ?, ?, ?, ?, ?)`,
      )
      .run(
        'POST /v1/auth/qq/assert',
        oldIdentity?.digest ?? '',
        1,
        '2026-09-01',
        null,
        '2026-09-01T09:59:30.000Z',
      );

    const { updateAbuseControls } = await import('./test-support/abuse-controls');
    updateAbuseControls(env, (controls) => {
      controls.qqAuthAssertIp.maxRequests = 1;
    });
    await expect(
      checkEndpointRateLimit(env.BROKER_DB, {
        endpoint: 'POST /v1/auth/qq/assert',
        now: NOW,
        ip: '203.0.113.90',
        installationId: null,
        hardwareHash: null,
        networkIdentitySecrets: testNetworkIdentitySecrets(env),
      }),
    ).resolves.toMatchObject({ status: 429, subcode: 'ip_rate_limited' });

    await expect(
      checkEndpointRateLimit(env.BROKER_DB, {
        endpoint: 'POST /v1/auth/qq/assert',
        now: NOW,
        ip: '203.0.113.91',
        installationId: null,
        hardwareHash: null,
        networkIdentitySecrets: testNetworkIdentitySecrets(env),
      }),
    ).resolves.toBeNull();
  });

  it('correlates digests across the epoch boundary only inside the active window', async () => {
    const env = createTestBrokerEnv();
    const secrets = resolveNetworkIdentitySecrets(env)!;
    const previousDay = new Date('2026-08-31T23:59:00.000Z');
    const previousIdentity = await resolveRequestNetworkIdentity('203.0.113.92', secrets, previousDay);
    expect(previousIdentity?.epoch).toBe('2026-08-31');
    env.__db
      .prepare(
        `INSERT INTO broker_request_events (
          endpoint, ip_digest, ip_key_version, ip_epoch, installation_id, observed_at
        ) VALUES (?, ?, ?, ?, ?, ?)`,
      )
      .run(
        'POST /v1/auth/qq/assert',
        previousIdentity?.digest ?? '',
        1,
        previousIdentity?.epoch ?? '',
        null,
        '2026-08-31T23:59:00.000Z',
      );
    env.__db
      .prepare(
        `INSERT INTO broker_request_events (
          endpoint, ip_digest, ip_key_version, ip_epoch, installation_id, observed_at
        ) VALUES (?, ?, ?, ?, ?, ?)`,
      )
      .run(
        'POST /v1/auth/qq/assert',
        previousIdentity?.digest ?? '',
        1,
        previousIdentity?.epoch ?? '',
        null,
        '2026-08-31T23:59:30.000Z',
      );

    const { updateAbuseControls } = await import('./test-support/abuse-controls');
    updateAbuseControls(env, (controls) => {
      controls.qqAuthAssertIp.maxRequests = 1;
    });
    await expect(
      checkEndpointRateLimit(env.BROKER_DB, {
        endpoint: 'POST /v1/auth/qq/assert',
        now: new Date('2026-09-01T00:10:00.000Z'),
        ip: '203.0.113.92',
        installationId: null,
        hardwareHash: null,
        networkIdentitySecrets: testNetworkIdentitySecrets(env),
      }),
    ).resolves.toMatchObject({ status: 429 });

    await expect(
      checkEndpointRateLimit(env.BROKER_DB, {
        endpoint: 'POST /v1/auth/qq/assert',
        now: new Date('2026-09-01T00:16:00.000Z'),
        ip: '203.0.113.92',
        installationId: null,
        hardwareHash: null,
        networkIdentitySecrets: testNetworkIdentitySecrets(env),
      }),
    ).resolves.toBeNull();
  });

  it('keeps operator IP hooks keyed and stable across epochs', async () => {
    const env = createTestBrokerEnv();
    const secrets = resolveNetworkIdentitySecrets(env)!;
    const normalized = normalizeNetworkIdentityIp('203.0.113.93')!;
    const digests = await deriveStableNetworkIdentityDigest(secrets, normalized, 'ip');
    expect(digests).toHaveLength(1);
    expect(digests[0]?.digest).toMatch(/^[a-f0-9]{64}$/u);

    const rawSha = await crypto.subtle.digest(
      'SHA-256',
      new TextEncoder().encode('203.0.113.93'),
    );
    const rawHex = Array.from(new Uint8Array(rawSha), (byte) =>
      byte.toString(16).padStart(2, '0'),
    ).join('');
    expect(digests[0]?.digest).not.toBe(rawHex);
    expect(digests[0]?.digest).not.toBe('203.0.113.93');
  });

  it('reserves a referral once per operation and reuses it across retries', async () => {
    const env = createTestBrokerEnv();
    const ownerRef = `ph-discord-user-v1_${'R'.repeat(43)}`;
    const referredRef = `ph-discord-user-v1_${'S'.repeat(43)}`;
    for (const installationId of ['install-referral-owner', 'install-referred-once']) {
      env.__db
        .prepare(
          `INSERT INTO installations (installation_id, device_public_key, app_version, created_at, last_seen_at)
           VALUES (?, ?, ?, ?, ?)`,
        )
        .run(installationId, `device-key-${installationId}`, '1.2.3', NOW_ISO, NOW_ISO);
    }
    env.__db
      .prepare(
        `INSERT INTO discord_identities (discord_user_ref, entitlement_installation_id, status, created_at, updated_at)
         VALUES (?, ?, 'active', ?, ?)`,
      )
      .run(ownerRef, 'install-referral-owner', NOW_ISO, NOW_ISO);
    env.__db
      .prepare(
        `INSERT INTO referral_codes (referral_id, owner_source, owner_subject_ref, owner_installation_id, status, created_at, updated_at)
         VALUES (?, 'discord', ?, ?, 'active', ?, ?)`,
      )
      .run('9ABCDX', ownerRef, 'install-referral-owner', NOW_ISO, NOW_ISO);
    env.__db
      .prepare(
        `INSERT INTO managed_operations (
          operation_id, issue_source, subject_ref, installation_id, device_public_key,
          state, attempt_count, current_attempt_index, resume_token_hash, auth_expires_at,
          failure_reason, client_action, referral_reward_id, referral_status, settlement_status,
          created_at, updated_at, last_reconciled_at, cleanup_attempts
        ) VALUES (?, 'discord', ?, ?, ?, 'ISSUE_READY', 0, 0, ?, ?, NULL, 'wait', NULL, 'none', 'none', ?, ?, NULL, 0)`,
      )
      .run(
        'ph-mop-v1_reserve_once_test_operation_1',
        referredRef,
        'install-referred-once',
        'device-once',
        'ph-mop-resume-v1_' + 'a'.repeat(64),
        new Date(NOW.getTime() + 60 * 60_000).toISOString(),
        NOW_ISO,
        NOW_ISO,
      );

    const identity = await resolveRequestNetworkIdentity('203.0.113.94', resolveNetworkIdentitySecrets(env), NOW);
    const first = await reserveIssueReferralReward(env.BROKER_DB, {
      referralId: '9ABCDX',
      referredSource: 'discord',
      referredSubjectRef: referredRef,
      referredInstallationId: 'install-referred-once',
      referredHardwareHash: 'hardware-reserve-once',
      referredHardwareHashSaltVersion: 7,
      attemptIpDigest: identity ? { digest: identity.digest, keyVersion: identity.keyVersion, epoch: identity.epoch } : null,
      operationId: 'ph-mop-v1_reserve_once_test_operation_1',
      nowIso: NOW_ISO,
    });
    expect(first).toMatchObject({ outcome: 'reserved' });

    const second = await reserveIssueReferralReward(env.BROKER_DB, {
      referralId: '9ABCDX',
      referredSource: 'discord',
      referredSubjectRef: referredRef,
      referredInstallationId: 'install-referred-once',
      referredHardwareHash: 'hardware-reserve-once',
      referredHardwareHashSaltVersion: 7,
      attemptIpDigest: identity ? { digest: identity.digest, keyVersion: identity.keyVersion, epoch: identity.epoch } : null,
      operationId: 'ph-mop-v1_reserve_once_test_operation_1',
      nowIso: NOW_ISO,
    });
    expect(second).toMatchObject({ outcome: 'reserved', referralId: '9ABCDX' });

    const rows = env.__db
      .prepare(`SELECT COUNT(*) AS count FROM referral_rewards WHERE operation_id = ?`)
      .get('ph-mop-v1_reserve_once_test_operation_1') as { count: number };
    expect(rows.count).toBe(1);
  });

  it('fails an unresolved referral only when the operation terminally expires', async () => {
    const env = createTestBrokerEnv();
    const { createManagedOperation, getManagedOperation, hashManagedOperationResumeToken } =
      await import('../src/managed-operation');
    const operationId = 'ph-mop-v1_referral_terminal_expiry_01';
    await createManagedOperation(env.BROKER_DB, {
      operationId,
      resumeTokenHash: await hashManagedOperationResumeToken('resume-terminal-test'),
      issueSource: 'discord',
      subjectRef: 'ph-discord-user-v1_terminal_expiry',
      installationId: 'install-terminal-expiry',
      devicePublicKey: 'device-terminal-expiry',
      now: NOW,
    });
    env.__db
      .prepare(
        `INSERT INTO referral_rewards (
          referral_id, referrer_source, referrer_subject_ref, referred_source, referred_subject_ref,
          referred_installation_id, referred_hardware_hash, referred_hardware_hash_salt_version,
          referred_bonus_status, referrer_bonus_status, operation_id, created_at, updated_at
        ) VALUES (?, 'discord', ?, 'discord', ?, ?, ?, 7, 'reserved', 'pending', ?, ?, ?)`,
      )
      .run(
        '9ABCDX',
        'ph-discord-user-v1_owner_terminal',
        'ph-discord-user-v1_terminal_expiry',
        'install-terminal-expiry',
        'hardware-terminal-expiry',
        operationId,
        NOW_ISO,
        NOW_ISO,
      );
    env.__db
      .prepare(`UPDATE managed_operations SET referral_reward_id = ?, referral_status = 'reserved', updated_at = ? WHERE operation_id = ?`)
      .run(1, NOW_ISO, operationId);
    const operation = (await getManagedOperation(env.BROKER_DB, operationId))!;
    await expireManagedOperation(env.BROKER_DB, operation, new Date(NOW.getTime() + 61 * 60_000));

    const reward = env.__db
      .prepare(`SELECT referred_bonus_status, referrer_bonus_status, failure_reason FROM referral_rewards WHERE operation_id = ?`)
      .get(operationId) as Record<string, string>;
    expect(reward).toMatchObject({
      referred_bonus_status: 'failed',
      referrer_bonus_status: 'failed',
      failure_reason: 'authorization_expired',
    });
  });

  it('leaves a reserved referral intact when the delivery was acknowledged', async () => {
    for (const terminal of ['expire', 'fail'] as const) {
      const env = createTestBrokerEnv();
      const { createManagedOperation, getManagedOperation, hashManagedOperationResumeToken } =
        await import('../src/managed-operation');
      const { expireManagedOperation, failManagedOperationTerminal } = await import(
        '../src/managed-operation'
      );
      const { createManagedKeyDelivery, markManagedKeyDeliveryAcknowledged } = await import(
        '../src/managed-key-delivery'
      );
      const operationId = `ph-mop-v1_referral_acked_${terminal}_01`;
      await createManagedOperation(env.BROKER_DB, {
        operationId,
        resumeTokenHash: await hashManagedOperationResumeToken('resume-acked-test'),
        issueSource: 'discord',
        subjectRef: 'ph-discord-user-v1_acked_expiry',
        installationId: 'install-acked-expiry',
        devicePublicKey: 'device-acked-expiry',
        now: NOW,
      });
      env.__db
        .prepare(
          `INSERT INTO referral_rewards (
            referral_id, referrer_source, referrer_subject_ref, referred_source, referred_subject_ref,
            referred_installation_id, referred_hardware_hash, referred_hardware_hash_salt_version,
            referred_bonus_status, referrer_bonus_status, operation_id, created_at, updated_at
          ) VALUES (?, 'discord', ?, 'discord', ?, ?, ?, 7, 'reserved', 'pending', ?, ?, ?)`,
        )
        .run(
          '9ABCDX',
          'ph-discord-user-v1_owner_acked',
          'ph-discord-user-v1_acked_expiry',
          'install-acked-expiry',
          'hardware-acked-expiry',
          operationId,
          NOW_ISO,
          NOW_ISO,
        );
      env.__db
        .prepare(`UPDATE managed_operations SET referral_reward_id = ?, referral_status = 'reserved', updated_at = ? WHERE operation_id = ?`)
        .run(1, NOW_ISO, operationId);
      const delivery = await createManagedKeyDelivery(env.BROKER_DB, {
        issueSource: 'discord',
        subjectRef: 'ph-discord-user-v1_acked_expiry',
        installationId: 'install-acked-expiry',
        managedCredentialRef: 'hash_acked_expiry_1',
        createdAt: NOW,
        expiresAt: new Date(NOW.getTime() + 15 * 60_000),
        operationId,
        attemptIndex: 1,
      });
      await markManagedKeyDeliveryAcknowledged(env.BROKER_DB, {
        deliveryId: delivery.deliveryId,
        acknowledgedAt: NOW,
      });
      const operation = (await getManagedOperation(env.BROKER_DB, operationId))!;
      if (terminal === 'expire') {
        await expireManagedOperation(env.BROKER_DB, operation, new Date(NOW.getTime() + 61 * 60_000));
      } else {
        await failManagedOperationTerminal(env.BROKER_DB, operation, NOW, 'terminal_provider_failure');
      }
      const reward = env.__db
        .prepare(`SELECT referred_bonus_status, referrer_bonus_status FROM referral_rewards WHERE operation_id = ?`)
        .get(operationId) as Record<string, string>;
      expect(reward).toMatchObject({
        referred_bonus_status: 'reserved',
        referrer_bonus_status: 'pending',
      });
      expect((await getManagedOperation(env.BROKER_DB, operationId))?.referral_status).toBe('reserved');
    }
  });

  it('reuses one skipped reservation per operation without new velocity rows', async () => {
    const env = createTestBrokerEnv();
    const { createManagedOperation, hashManagedOperationResumeToken } = await import(
      '../src/managed-operation'
    );
    const operationId = 'ph-mop-v1_skip_reuse_test_operation_1';
    await createManagedOperation(env.BROKER_DB, {
      operationId,
      resumeTokenHash: await hashManagedOperationResumeToken('resume-skip-reuse'),
      issueSource: 'qq',
      subjectRef: 'ph-qq-subject-v1_skip_reuse',
      installationId: 'install-skip-reuse',
      devicePublicKey: null,
      now: NOW,
    });
    const identity = await resolveRequestNetworkIdentity('203.0.113.95', resolveNetworkIdentitySecrets(env), NOW);
    const input = {
      referralId: '9ZZZZZ',
      referredSource: 'qq' as const,
      referredSubjectRef: 'ph-qq-subject-v1_skip_reuse',
      referredInstallationId: 'install-skip-reuse',
      referredHardwareHash: null,
      referredHardwareHashSaltVersion: null,
      attemptIpDigest: identity ? { digest: identity.digest, keyVersion: identity.keyVersion, epoch: identity.epoch } : null,
      operationId,
      nowIso: NOW_ISO,
    };
    const first = await reserveIssueReferralReward(env.BROKER_DB, input);
    expect(first.outcome).toBe('skipped');
    const second = await reserveIssueReferralReward(env.BROKER_DB, input);
    expect(second).toEqual(first);
    const rows = env.__db
      .prepare(`SELECT COUNT(*) AS count FROM referral_rewards WHERE operation_id = ?`)
      .get(operationId) as { count: number };
    expect(rows.count).toBe(1);
  });

  it('links the first QQ reservation to the bound operation for settlement reuse', async () => {
    const env = createTestBrokerEnv();
    const ownerRef = `ph-discord-user-v1_${'Q'.repeat(43)}`;
    for (const installationId of ['install-qq-op-link-owner', 'install-qq-op-link']) {
      env.__db
        .prepare(
          `INSERT INTO installations (installation_id, device_public_key, app_version, created_at, last_seen_at)
           VALUES (?, ?, ?, ?, ?)`,
        )
        .run(installationId, `device-key-${installationId}`, '1.2.3', NOW_ISO, NOW_ISO);
    }
    env.__db
      .prepare(
        `INSERT INTO discord_identities (discord_user_ref, entitlement_installation_id, status, created_at, updated_at)
         VALUES (?, ?, 'active', ?, ?)`,
      )
      .run(ownerRef, 'install-qq-op-link-owner', NOW_ISO, NOW_ISO);
    env.__db
      .prepare(
        `INSERT INTO referral_codes (referral_id, owner_source, owner_subject_ref, owner_installation_id, status, created_at, updated_at)
         VALUES (?, 'discord', ?, ?, 'active', ?, ?)`,
      )
      .run('9ABCDX', ownerRef, 'install-qq-op-link-owner', NOW_ISO, NOW_ISO);
    const { createManagedOperation, hashManagedOperationResumeToken } = await import(
      '../src/managed-operation'
    );
    const { getOperationReferralReward } = await import('../src/referral');
    const operationId = 'ph-mop-v1_qq_op_link_test_01';
    await createManagedOperation(env.BROKER_DB, {
      operationId,
      resumeTokenHash: await hashManagedOperationResumeToken('resume-qq-op-link'),
      issueSource: 'qq',
      subjectRef: 'ph-qq-subject-v1_op_link',
      installationId: 'install-qq-op-link',
      devicePublicKey: null,
      now: NOW,
    });
    const identity = await resolveRequestNetworkIdentity('203.0.113.96', resolveNetworkIdentitySecrets(env), NOW);
    const reserved = await reserveIssueReferralReward(env.BROKER_DB, {
      referralId: '9ABCDX',
      referredSource: 'qq',
      referredSubjectRef: 'ph-qq-subject-v1_op_link',
      referredInstallationId: 'install-qq-op-link',
      referredHardwareHash: null,
      referredHardwareHashSaltVersion: null,
      attemptIpDigest: identity ? { digest: identity.digest, keyVersion: identity.keyVersion, epoch: identity.epoch } : null,
      operationId,
      nowIso: NOW_ISO,
    });
    expect(reserved).toMatchObject({ outcome: 'reserved' });
    const row = env.__db
      .prepare(`SELECT operation_id, referred_bonus_status FROM referral_rewards WHERE operation_id = ?`)
      .get(operationId) as Record<string, string>;
    expect(row).toMatchObject({ operation_id: operationId, referred_bonus_status: 'reserved' });
    await expect(getOperationReferralReward(env.BROKER_DB, operationId)).resolves.toMatchObject({
      outcome: 'reserved',
      referralId: '9ABCDX',
    });
  });

  it('keeps operation-bound referrals settleable through stale delivery cleanup', async () => {
    const env = createTestBrokerEnv();
    const { createManagedOperation, getManagedOperation, hashManagedOperationResumeToken } =
      await import('../src/managed-operation');
    const { createManagedKeyDelivery } = await import('../src/managed-key-delivery');
    const { reconcileStaleManagedKeyDeliveries } = await import('../src/scheduled');
    const subjectRef = 'ph-discord-user-v1_stale_survive';
    const installationId = 'install-stale-survive';
    env.__db
      .prepare(
        `INSERT INTO installations (installation_id, device_public_key, app_version, created_at, last_seen_at)
         VALUES (?, ?, ?, ?, ?)`,
      )
      .run(installationId, 'device-stale-survive', '1.2.3', NOW_ISO, NOW_ISO);
    env.__db
      .prepare(
        `INSERT INTO discord_identities (discord_user_ref, entitlement_installation_id, status, created_at, updated_at)
         VALUES (?, ?, 'issuing', ?, ?)`,
      )
      .run(subjectRef, installationId, NOW_ISO, NOW_ISO);
    const operationId = 'ph-mop-v1_stale_survive_operation_1';
    await createManagedOperation(env.BROKER_DB, {
      operationId,
      resumeTokenHash: await hashManagedOperationResumeToken('resume-stale-survive'),
      issueSource: 'discord',
      subjectRef,
      installationId,
      devicePublicKey: 'device-stale-survive',
      now: NOW,
    });
    const { startManagedOperationAttempt, recordAttemptCredential, transitionManagedOperation } =
      await import('../src/managed-operation');
    const operation = (await getManagedOperation(env.BROKER_DB, operationId))!;
    const started = await startManagedOperationAttempt(env.BROKER_DB, operation, NOW);
    expect(started.ok).toBe(true);
    if (!started.ok) {
      return;
    }
    await recordAttemptCredential(env.BROKER_DB, operationId, 1, 'hash_stale_survive_1', NOW);
    await transitionManagedOperation(env.BROKER_DB, operationId, 'DELIVERY_PENDING', NOW);
    const delivery = await createManagedKeyDelivery(env.BROKER_DB, {
      issueSource: 'discord',
      subjectRef,
      installationId,
      managedCredentialRef: 'hash_stale_survive_1',
      createdAt: new Date(NOW.getTime() - 30 * 60_000),
      expiresAt: new Date(NOW.getTime() - 15 * 60_000),
      operationId,
      attemptIndex: 1,
    });
    env.__db
      .prepare(
        `INSERT INTO referral_rewards (
          referral_id, referrer_source, referrer_subject_ref, referred_source, referred_subject_ref,
          referred_installation_id, referred_hardware_hash, referred_hardware_hash_salt_version,
          referred_bonus_status, referrer_bonus_status, operation_id, created_at, updated_at
        ) VALUES (?, 'discord', ?, 'discord', ?, ?, ?, 7, 'reserved', 'pending', ?, ?, ?)`,
      )
      .run(
        '9ABCDX',
        'ph-discord-user-v1_owner_stale',
        subjectRef,
        installationId,
        'hardware-stale-survive',
        operationId,
        NOW_ISO,
        NOW_ISO,
      );
    const legacySubjectRef = 'ph-discord-user-v1_stale_legacy';
    const legacyInstallationId = 'install-stale-legacy';
    env.__db
      .prepare(
        `INSERT INTO installations (installation_id, device_public_key, app_version, created_at, last_seen_at)
         VALUES (?, ?, ?, ?, ?)`,
      )
      .run(legacyInstallationId, 'device-stale-legacy', '1.2.3', NOW_ISO, NOW_ISO);
    env.__db
      .prepare(
        `INSERT INTO referral_rewards (
          referral_id, referrer_source, referrer_subject_ref, referred_source, referred_subject_ref,
          referred_installation_id, referred_hardware_hash, referred_hardware_hash_salt_version,
          referred_bonus_status, referrer_bonus_status, operation_id, created_at, updated_at
        ) VALUES (?, 'discord', ?, 'discord', ?, ?, ?, 7, 'reserved', 'pending', NULL, ?, ?)`,
      )
      .run(
        '9ABCDX',
        'ph-discord-user-v1_owner_stale',
        legacySubjectRef,
        legacyInstallationId,
        'hardware-stale-legacy',
        NOW_ISO,
        NOW_ISO,
      );
    const legacyDelivery = await createManagedKeyDelivery(env.BROKER_DB, {
      issueSource: 'discord',
      subjectRef: legacySubjectRef,
      installationId: legacyInstallationId,
      managedCredentialRef: 'hash_stale_legacy_1',
      createdAt: new Date(NOW.getTime() - 30 * 60_000),
      expiresAt: new Date(NOW.getTime() - 15 * 60_000),
      operationId: null,
      attemptIndex: null,
    });
    const liveOperationId = 'ph-mop-v1_liveB_survive_operation_2';
    await createManagedOperation(env.BROKER_DB, {
      operationId: liveOperationId,
      resumeTokenHash: await hashManagedOperationResumeToken('resume-stale-survive-2'),
      issueSource: 'discord',
      subjectRef,
      installationId,
      devicePublicKey: 'device-stale-survive',
      now: NOW,
    });
    const liveOperation = (await getManagedOperation(env.BROKER_DB, liveOperationId))!;
    const liveStarted = await startManagedOperationAttempt(env.BROKER_DB, liveOperation, NOW);
    expect(liveStarted.ok).toBe(true);
    if (!liveStarted.ok) {
      return;
    }
    await recordAttemptCredential(env.BROKER_DB, liveOperationId, 1, 'hash_stale_live_1', NOW);
    await transitionManagedOperation(env.BROKER_DB, liveOperationId, 'DELIVERY_PENDING', NOW);
    await createManagedKeyDelivery(env.BROKER_DB, {
      issueSource: 'discord',
      subjectRef,
      installationId,
      managedCredentialRef: 'hash_stale_live_1',
      createdAt: NOW,
      expiresAt: new Date(NOW.getTime() + 15 * 60_000),
      operationId: liveOperationId,
      attemptIndex: 1,
    });
    env.__db
      .prepare(`UPDATE managed_operations SET referral_reward_id = ?, referral_status = 'reserved', updated_at = ? WHERE operation_id = ?`)
      .run(1, NOW_ISO, operationId);
    mockProviderKeyApi(new Map([['hash_stale_survive_1', 0.07], ['hash_stale_legacy_1', 0.07]]));

    const result = await reconcileStaleManagedKeyDeliveries(env, NOW);
    expect(result).toMatchObject({ expired: 2 });
    const survivor = env.__db
      .prepare(`SELECT referred_bonus_status, referrer_bonus_status FROM referral_rewards WHERE operation_id = ?`)
      .get(operationId) as Record<string, string>;
    expect(survivor).toMatchObject({ referred_bonus_status: 'reserved', referrer_bonus_status: 'pending' });
    const legacy = env.__db
      .prepare(`SELECT referred_bonus_status FROM referral_rewards WHERE operation_id IS NULL`)
      .get() as Record<string, string>;
    expect(legacy).toMatchObject({ referred_bonus_status: 'failed' });
    expect((await getManagedOperation(env.BROKER_DB, operationId))?.state).toBe('DELIVERY_PENDING');
    expect((await getManagedOperation(env.BROKER_DB, liveOperationId))?.state).toBe('DELIVERY_PENDING');
    expect(
      env.__db
        .prepare(`SELECT status FROM discord_identities WHERE discord_user_ref = ?`)
        .get(subjectRef) as Record<string, string>,
    ).toMatchObject({ status: 'issuing' });
    const cleaned = env.__db
      .prepare(`SELECT status, failure_reason FROM managed_key_deliveries WHERE delivery_id = ?`)
      .get(delivery.deliveryId) as Record<string, string>;
    expect(cleaned).toMatchObject({ status: 'expired', failure_reason: 'ack_expired_child_key_cleaned' });
    const legacyCleaned = env.__db
      .prepare(`SELECT status, failure_reason FROM managed_key_deliveries WHERE delivery_id = ?`)
      .get(legacyDelivery.deliveryId) as Record<string, string>;
    expect(legacyCleaned).toMatchObject({ status: 'expired', failure_reason: 'ack_expired_child_key_cleaned' });
  });
  it('never credits referrer settlement when the owner entitlement died mid-batch', async () => {
    const armed = { expired: false };
    let env: ReturnType<typeof createTestBrokerEnv>;
    env = createTestBrokerEnv({
      beforeRun: ({ sql }) => {
        if (
          !armed.expired &&
          sql.includes('UPDATE openrouter_entitlements') &&
          sql.includes('SET budget_usd')
        ) {
          armed.expired = true;
          env.__db
            .prepare(
              `UPDATE openrouter_entitlements SET expires_at = ? WHERE managed_credential_ref = ?`,
            )
            .run('2026-01-01T00:00:00.000Z', 'hash_referrer_race_owner');
        }
      },
    });
    env.__db
      .prepare(
        `INSERT INTO installations (installation_id, device_public_key, app_version, created_at, last_seen_at)
         VALUES (?, ?, ?, ?, ?)`,
      )
      .run('install-referrer-race', 'device-referrer-race', '1.2.3', NOW_ISO, NOW_ISO);
    env.__db
      .prepare(
        `INSERT INTO discord_identities (discord_user_ref, entitlement_installation_id, status, created_at, updated_at)
         VALUES (?, ?, 'active', ?, ?)`,
      )
      .run('ph-discord-user-v1_referrer_race', 'install-referrer-race', NOW_ISO, NOW_ISO);
    env.__db
      .prepare(
        `INSERT INTO openrouter_entitlements (
          installation_id, status, budget_usd, managed_credential_ref, issued_at, expires_at,
          discord_user_ref, discord_issue_status, discord_issue_delivered_at
        ) VALUES (?, 'active', 10, ?, ?, ?, ?, 'active', ?)`,
      )
      .run(
        'install-referrer-race',
        'hash_referrer_race_owner',
        NOW_ISO,
        '2026-12-01T00:00:00.000Z',
        'ph-discord-user-v1_referrer_race',
        NOW_ISO,
      );
    env.__db
      .prepare(
        `INSERT INTO installations (installation_id, device_public_key, app_version, created_at, last_seen_at)
         VALUES (?, ?, ?, ?, ?)`,
      )
      .run('install-referred-race', 'device-referred-race', '1.2.3', NOW_ISO, NOW_ISO);
    env.__db
      .prepare(
        `INSERT INTO discord_identities (discord_user_ref, entitlement_installation_id, status, created_at, updated_at)
         VALUES (?, ?, 'active', ?, ?)`,
      )
      .run('ph-discord-user-v1_referred_race', 'install-referred-race', NOW_ISO, NOW_ISO);
    env.__db
      .prepare(
        `INSERT INTO openrouter_entitlements (
          installation_id, status, budget_usd, managed_credential_ref, issued_at, expires_at,
          discord_user_ref, discord_issue_status, discord_issue_delivered_at
        ) VALUES (?, 'active', 0.09, ?, ?, ?, ?, 'active', ?)`,
      )
      .run(
        'install-referred-race',
        'hash_referred_race_1',
        NOW_ISO,
        '2026-12-01T00:00:00.000Z',
        'ph-discord-user-v1_referred_race',
        NOW_ISO,
      );
    env.__db
      .prepare(
        `INSERT INTO referral_rewards (
          referral_id, referrer_source, referrer_subject_ref, referred_source, referred_subject_ref,
          referred_installation_id, referred_hardware_hash, referred_hardware_hash_salt_version,
          referred_bonus_status, referrer_bonus_status, referred_managed_credential_ref, created_at, updated_at
        ) VALUES (?, 'discord', ?, 'discord', ?, ?, ?, 7, 'credited', 'pending', ?, ?, ?)`,
      )
      .run(
        '9ABCDX',
        'ph-discord-user-v1_referrer_race',
        'ph-discord-user-v1_referred_race',
        'install-referred-race',
        'hardware-referrer-race',
        'hash_referred_race_1',
        NOW_ISO,
        NOW_ISO,
      );
    const rewardId = Number((env.__db.prepare('SELECT last_insert_rowid() AS id').get() as { id: number }).id);
    env.__db
      .prepare(
        `INSERT INTO managed_referral_settlement_jobs (
          source, referral_reward_id, delivery_id, operation_id, phase,
          attempt_count, last_attempt_at, next_attempt_at,
          fencing_token, lease_expires_at, last_error_code, created_at, updated_at, completed_at
        ) VALUES ('discord', ?, ?, NULL, 'referrer_pending', 1, ?, ?, NULL, NULL, NULL, ?, ?, NULL)`,
      )
      .run(rewardId, 'ph-delivery-v1_referrer_race', NOW_ISO, NOW_ISO, NOW_ISO, NOW_ISO);

    const limits = new Map([
      ['hash_referrer_race_owner', 10],
      ['hash_referred_race_1', 0.09],
    ]);
    const { calls } = mockProviderKeyApi(limits);
    await expect(
      processManagedReferralSettlementJobs(env, { now: NOW }),
    ).resolves.toMatchObject({ completed: 0 });
    expect(armed.expired).toBe(true);
    expect(
      env.__db
        .prepare(`SELECT referrer_bonus_status FROM referral_rewards WHERE id = ?`)
        .get(rewardId) as Record<string, string>,
    ).toMatchObject({ referrer_bonus_status: 'pending' });
    expect(
      env.__db.prepare(`SELECT phase FROM managed_referral_settlement_jobs WHERE id = ?`).get(1) as Record<string, string>,
    ).toMatchObject({ phase: 'referrer_pending' });
    expect(calls.filter((call) => call.method === 'PATCH')).toHaveLength(0);
  });

  it('converges a crash-after-mutation settlement without duplicating provider effects', async () => {
    const env = createTestBrokerEnv();
    env.__db
      .prepare(
        `INSERT INTO installations (installation_id, device_public_key, app_version, created_at, last_seen_at)
         VALUES (?, ?, ?, ?, ?)`,
      )
      .run('install-crash-converge', 'device-crash-converge', '1.2.3', NOW_ISO, NOW_ISO);
    env.__db
      .prepare(
        `INSERT INTO discord_identities (discord_user_ref, entitlement_installation_id, status, created_at, updated_at)
         VALUES (?, ?, 'active', ?, ?)`,
      )
      .run('ph-discord-user-v1_crash_converge', 'install-crash-converge', NOW_ISO, NOW_ISO);
    env.__db
      .prepare(
        `INSERT INTO openrouter_entitlements (
          installation_id, status, budget_usd, managed_credential_ref, issued_at, expires_at,
          discord_user_ref, discord_issue_status, discord_issue_delivered_at
        ) VALUES (?, 'active', 0.09, ?, ?, ?, ?, 'active', ?)`,
      )
      .run('install-crash-converge', 'hash_crash_converge', NOW_ISO, '2026-12-01T00:00:00.000Z', 'ph-discord-user-v1_crash_converge', NOW_ISO);
    env.__db
      .prepare(
        `INSERT INTO referral_rewards (
          referral_id, referrer_source, referrer_subject_ref, referred_source, referred_subject_ref,
          referred_installation_id, referred_hardware_hash, referred_hardware_hash_salt_version,
          referred_bonus_status, referrer_bonus_status,
          referred_managed_credential_ref, created_at, updated_at, credited_at
        ) VALUES (?, 'discord', ?, 'discord', ?, ?, ?, 7, 'credited', 'pending', ?, ?, ?, ?)`,
      )
      .run(
        '9ABCDX',
        'ph-discord-user-v1_crash_owner',
        'ph-discord-user-v1_crash_converge',
        'install-crash-converge',
        'hardware-crash-converge',
        'hash_crash_converge',
        NOW_ISO,
        NOW_ISO,
        NOW_ISO,
      );
    const rewardId = Number((env.__db.prepare('SELECT last_insert_rowid() AS id').get() as { id: number }).id);
    env.__db
      .prepare(
        `INSERT INTO managed_referral_settlement_jobs (
          source, referral_reward_id, delivery_id, operation_id, phase,
          attempt_count, last_attempt_at, next_attempt_at,
          fencing_token, lease_expires_at, last_error_code, created_at, updated_at, completed_at
        ) VALUES ('discord', ?, ?, NULL, 'invitee_pending', 1, ?, ?, NULL, NULL, 'crash_before_commit', ?, ?, NULL)`,
      )
      .run(rewardId, 'ph-delivery-v1_crash', NOW_ISO, NOW_ISO, NOW_ISO, NOW_ISO);

    const limits = new Map([['hash_crash_converge', 0.09]]);
    const { calls } = mockProviderKeyApi(limits);
    await expect(
      processManagedReferralSettlementJobs(env, { now: NOW }),
    ).resolves.toMatchObject({ advanced: 1, completed: 1, retried: 0 });
    expect(calls.filter((call) => call.method === 'PATCH')).toHaveLength(0);
    expect(
      env.__db
        .prepare(`SELECT referrer_bonus_status, skip_reason FROM referral_rewards WHERE id = ?`)
        .get(rewardId) as Record<string, string>,
    ).toMatchObject({ referrer_bonus_status: 'skipped', skip_reason: 'referrer_managed_key_missing' });
  });

  it('never logs raw keys, tokens, digests detail, or raw IPs during failure handling', async () => {
    const env = createTestBrokerEnv();
    const seen: string[] = [];
    const serialize = (args: unknown[]) =>
      args.map((arg) => (typeof arg === 'string' ? arg : JSON.stringify(arg))).join(' ');
    const infoSpy = vi.spyOn(console, 'info').mockImplementation((...args: unknown[]) => {
      seen.push(serialize(args));
    });
    const errorSpy = vi.spyOn(console, 'error').mockImplementation((...args: unknown[]) => {
      seen.push(serialize(args));
    });
    try {
      const { reserveIssueReferralReward } = await import('../src/referral');
      await reserveIssueReferralReward(env.BROKER_DB, {
        referralId: 'ZZZZZZ',
        referredSource: 'discord',
        referredSubjectRef: 'ph-discord-user-v1_log_probe',
        referredInstallationId: 'install-log-probe',
        referredHardwareHash: 'hardware-log-probe',
        referredHardwareHashSaltVersion: 7,
        attemptIpDigest: null,
        nowIso: NOW_ISO,
      });
      const { runNetworkIdentityBackfill } = await import('../src/network-identity-migration');
      await runNetworkIdentityBackfill(env.BROKER_DB, resolveNetworkIdentitySecrets(env), NOW);

      const { createManagedOperation, expireManagedOperation, getManagedOperation, hashManagedOperationResumeToken } =
        await import('../src/managed-operation');
      const operationId = 'ph-mop-v1_log_probe_operation_01';
      const resumeToken = 'log-probe-resume-token-value-which-is-secret';
      await createManagedOperation(env.BROKER_DB, {
        operationId,
        resumeTokenHash: await hashManagedOperationResumeToken(resumeToken),
        issueSource: 'discord',
        subjectRef: 'ph-discord-user-v1_log_probe',
        installationId: 'install-log-probe-op',
        devicePublicKey: 'device-log-probe-op',
        now: new Date('2026-09-01T09:00:00.000Z'),
      });
      const created = (await getManagedOperation(env.BROKER_DB, operationId))!;
      await expireManagedOperation(env.BROKER_DB, created, NOW);

      const { sweepStaleManagedOperations } = await import('../src/managed-operation');
      await sweepStaleManagedOperations(env, NOW);

      const rawIp = '203.0.113.200';
      const rawKey = 'or-raw-key-must-never-log';
      const rawToken = 'raw-ack-token-must-never-log';
      const corpus = seen.join('\n');
      expect(corpus).not.toContain(rawIp);
      expect(corpus).not.toContain(rawKey);
      expect(corpus).not.toContain(rawToken);
      expect(corpus).not.toContain(resumeToken);
      expect(corpus).not.toContain(env.NETWORK_IDENTITY_HMAC_SECRET);
      expect(corpus).toContain(operationId);
    } finally {
      infoSpy.mockRestore();
      errorSpy.mockRestore();
    }
  });
});

describe('network identity windows and hooks', () => {
  it('counts requests across every UTC-day epoch inside a multi-day window', async () => {
    const env = createTestBrokerEnv();
    const secrets = resolveNetworkIdentitySecrets(env)!;
    for (const iso of [
      '2026-08-30T12:00:00.000Z',
      '2026-08-31T12:00:00.000Z',
      '2026-09-01T11:50:00.000Z',
    ]) {
      const identity = await resolveRequestNetworkIdentity('203.0.113.92', secrets, new Date(iso));
      env.__db
        .prepare(
          `INSERT INTO broker_request_events (
            endpoint, ip_digest, ip_key_version, ip_epoch, installation_id, observed_at
          ) VALUES (?, ?, ?, ?, ?, ?)`,
        )
        .run(
          'POST /v1/auth/qq/assert',
          identity?.digest ?? '',
          identity?.keyVersion ?? 1,
          identity?.epoch ?? '',
          null,
          iso,
        );
    }

    const { updateAbuseControls } = await import('./test-support/abuse-controls');
    updateAbuseControls(env, (controls) => {
      controls.qqAuthAssertIp.maxRequests = 2;
      controls.qqAuthAssertIp.windowMinutes = 2880;
    });
    await expect(
      checkEndpointRateLimit(env.BROKER_DB, {
        endpoint: 'POST /v1/auth/qq/assert',
        now: new Date('2026-09-01T12:00:00.000Z'),
        ip: '203.0.113.92',
        installationId: null,
        hardwareHash: null,
        networkIdentitySecrets: testNetworkIdentitySecrets(env),
      }),
    ).resolves.toMatchObject({ status: 429, subcode: 'ip_rate_limited' });

    await expect(
      checkEndpointRateLimit(env.BROKER_DB, {
        endpoint: 'POST /v1/auth/qq/assert',
        now: new Date('2026-09-01T12:00:00.000Z'),
        ip: '203.0.113.93',
        installationId: null,
        hardwareHash: null,
        networkIdentitySecrets: testNetworkIdentitySecrets(env),
      }),
    ).resolves.toBeNull();
  });

  it('converts raw-IP operator hooks to stable digests without losing enforcement', async () => {
    const provision = createTestBrokerEnv();
    const secrets = resolveNetworkIdentitySecrets(provision)!;
    const db = new DatabaseSync(':memory:');
    try {
      const { BROKER_MIGRATION_FILENAMES, readBrokerMigrationSql } = await import(
        './test-support/migrations'
      );
      for (const file of BROKER_MIGRATION_FILENAMES) {
        if (file <= '0019_managed_referral_settlement.sql') {
          db.exec(readBrokerMigrationSql(file));
        }
      }
      db.prepare(
        `INSERT INTO broker_abuse_subject_hooks (
          hook_kind, subject_type, subject_value, outcome_code, outcome_class
        ) VALUES ('denylist', 'ip', ?, 'trial_unavailable', 'terminal')`,
      ).run('203.0.113.99');
      for (const file of BROKER_MIGRATION_FILENAMES) {
        if (file === '0020_network_identity_hmac.sql') {
          db.exec(readBrokerMigrationSql(file));
        }
      }
      const phase = db
        .prepare(`SELECT value FROM broker_config WHERE key = 'network_identity_migration'`)
        .get() as { value: string };
      expect(JSON.parse(phase.value)).toMatchObject({ phase: 'dual_write' });

      const { matchSubjectHook } = await import('../src/abuse-controls');
      const wrapped = wrapDatabaseSync(db);
      const context = {
        endpoint: 'POST /v1/trial/challenge',
        now: NOW,
        ip: '203.0.113.99',
        installationId: null,
        hardwareHash: null,
        networkIdentitySecrets: secrets,
      };
      await expect(matchSubjectHook(wrapped, context)).resolves.toMatchObject({
        hookKind: 'denylist',
      });

      const converted = await runNetworkIdentityBackfill(wrapped, secrets, NOW);
      expect(converted.hooksConverted).toBe(1);
      expect(converted.pendingHooks).toBe(0);
      const stored = db
        .prepare(`SELECT subject_value FROM broker_abuse_subject_hooks WHERE subject_type = 'ip'`)
        .get() as { subject_value: string };
      expect(stored.subject_value).toMatch(/^[a-f0-9]{64}$/u);
      expect(stored.subject_value).not.toBe('203.0.113.99');
      await expect(matchSubjectHook(wrapped, context)).resolves.toMatchObject({
        hookKind: 'denylist',
      });

      const later = new Date(NOW.getTime() + 25 * 60 * 60_000);
      const converged = await runNetworkIdentityBackfill(wrapped, secrets, later);
      expect(converged.finalized).toBe(true);
      db.exec(readBrokerMigrationSql('0021_network_identity_purge.sql'));
      await expect(matchSubjectHook(wrapped, context)).resolves.toMatchObject({
        hookKind: 'denylist',
      });
    } finally {
      db.close();
    }
  });

  it('marks unparseable request rows with a sentinel so migration still converges', async () => {
    const provision = createTestBrokerEnv();
    const secrets = resolveNetworkIdentitySecrets(provision)!;
    const db = new DatabaseSync(':memory:');
    try {
      const { BROKER_MIGRATION_FILENAMES, readBrokerMigrationSql } = await import(
        './test-support/migrations'
      );
      for (const file of BROKER_MIGRATION_FILENAMES) {
        if (file <= '0019_managed_referral_settlement.sql') {
          db.exec(readBrokerMigrationSql(file));
        }
      }
      db.prepare(
        `INSERT INTO broker_request_events (endpoint, ip, installation_id, observed_at)
         VALUES (?, ?, ?, ?)`,
      ).run('POST /v1/auth/qq/assert', 'not-an-ip', null, NOW_ISO);
      for (const file of BROKER_MIGRATION_FILENAMES) {
        if (file === '0020_network_identity_hmac.sql') {
          db.exec(readBrokerMigrationSql(file));
        }
      }
      const wrapped = wrapDatabaseSync(db);
      const result = await runNetworkIdentityBackfill(wrapped, secrets, NOW);
      expect(result.requestEventsBackfilled).toBe(0);
      expect(result.pendingRequestEvents).toBe(0);
      expect(result.finalized).toBe(true);
      const wedge = db
        .prepare(`SELECT COUNT(*) AS count FROM broker_request_events WHERE ip IS NOT NULL`)
        .get() as { count: number };
      expect(wedge.count).toBe(0);
      const phase = db
        .prepare(`SELECT value FROM broker_config WHERE key = 'network_identity_migration'`)
        .get() as { value: string };
      expect(JSON.parse(phase.value)).toMatchObject({ phase: 'keyed_only' });
      db.exec(readBrokerMigrationSql('0021_network_identity_purge.sql'));
    } finally {
      db.close();
    }
  });
});

describe('network identity window inventory and fail-closed finalize', () => {
  it('keeps the purge gate and the backfill horizon on one window inventory', async () => {
    const { NETWORK_IDENTITY_WINDOW_CONFIG_PATHS, resolveNetworkIdentityMaxWindowMinutes } =
      await import('../src/network-identity-migration');
    const { readBrokerMigrationSql } = await import('./test-support/migrations');
    const sql = readBrokerMigrationSql('0021_network_identity_purge.sql');
    const gatePaths = new Set(
      [...sql.matchAll(/\$\.([\w.]+)\.windowMinutes/gu)].map((match) => match[1]),
    );
    expect(gatePaths).toEqual(new Set(NETWORK_IDENTITY_WINDOW_CONFIG_PATHS));
    expect(NETWORK_IDENTITY_WINDOW_CONFIG_PATHS).toContain('qqAuthAssertIp');
    expect(NETWORK_IDENTITY_WINDOW_CONFIG_PATHS).toContain('trialStatus');

    const env = createTestBrokerEnv();
    const { updateAbuseControls } = await import('./test-support/abuse-controls');
    updateAbuseControls(env, (controls) => {
      controls.trialStatus.windowMinutes = 10080;
    });
    await expect(
      resolveNetworkIdentityMaxWindowMinutes(env.BROKER_DB),
    ).resolves.toBe(10080);
  });

  it('fails closed instead of flipping phase when migration state cannot persist', async () => {
    const db = new DatabaseSync(':memory:');
    try {
      const { BROKER_MIGRATION_FILENAMES, readBrokerMigrationSql } = await import(
        './test-support/migrations'
      );
      for (const file of BROKER_MIGRATION_FILENAMES) {
        if (file <= '0019_managed_referral_settlement.sql') {
          db.exec(readBrokerMigrationSql(file));
        }
      }
      db.prepare(
        `INSERT INTO broker_request_events (endpoint, ip, installation_id, observed_at)
         VALUES (?, ?, ?, ?)`,
      ).run('POST /v1/auth/qq/assert', '203.0.113.103', null, NOW_ISO);
      for (const file of BROKER_MIGRATION_FILENAMES) {
        if (file === '0020_network_identity_hmac.sql') {
          db.exec(readBrokerMigrationSql(file));
        }
      }
      const { finalizeNetworkIdentityMigration } = await import(
        '../src/network-identity-migration'
      );
      const wrapped = wrapDatabaseSync(db);
      expect(await finalizeNetworkIdentityMigration(wrapped, NOW)).toBe(true);

      db.prepare(`UPDATE broker_config SET value = ? WHERE key = 'network_identity_migration'`).run(
        JSON.stringify({ phase: 'dual_write', purge_after: null }),
      );
      db.prepare(
        `INSERT INTO broker_request_events (endpoint, ip, installation_id, observed_at)
         VALUES (?, ?, ?, ?)`,
      ).run('POST /v1/auth/qq/assert', '203.0.113.104', null, NOW_ISO);
      db.prepare(`DELETE FROM broker_config WHERE key = 'network_identity_migration'`).run();
      expect(await finalizeNetworkIdentityMigration(wrapped, NOW)).toBe(false);
      const phase = db
        .prepare(`SELECT value FROM broker_config WHERE key = 'network_identity_migration'`)
        .get() as { value: string } | undefined;
      expect(phase).toBeUndefined();
    } finally {
      db.close();
    }
  });

  it('blocks finalization on unparseable hooks and surfaces non-secret diagnostics', async () => {
    const provision = createTestBrokerEnv();
    const secrets = resolveNetworkIdentitySecrets(provision)!;
    const db = new DatabaseSync(':memory:');
    try {
      const { BROKER_MIGRATION_FILENAMES, readBrokerMigrationSql } = await import(
        './test-support/migrations'
      );
      for (const file of BROKER_MIGRATION_FILENAMES) {
        if (file <= '0019_managed_referral_settlement.sql') {
          db.exec(readBrokerMigrationSql(file));
        }
      }
      db.prepare(
        `INSERT INTO broker_velocity_cap_hooks (
          subject_type, subject_value, max_requests, window_minutes,
          outcome_code, outcome_class, active
        ) VALUES ('ip', ?, 1, 60, 'rate_limited', 'retryable', 1)`,
      ).run('not-an-ip-hook');
      for (const file of BROKER_MIGRATION_FILENAMES) {
        if (file === '0020_network_identity_hmac.sql') {
          db.exec(readBrokerMigrationSql(file));
        }
      }
      const wrapped = wrapDatabaseSync(db);
      const blocked = await runNetworkIdentityBackfill(wrapped, secrets, NOW);
      expect(blocked.finalized).toBe(false);
      expect(blocked.rawHooks).toBe(1);
      expect(blocked.unparseableHooks).toBe(1);
      expect(blocked.rawHookSampleIds).toEqual([
        expect.objectContaining({ table: 'broker_velocity_cap_hooks', id: expect.any(Number) }),
      ]);
      const phase = db
        .prepare(`SELECT value FROM broker_config WHERE key = 'network_identity_migration'`)
        .get() as { value: string };
      expect(JSON.parse(phase.value)).toMatchObject({ phase: 'dual_write' });

      db.prepare(`UPDATE broker_velocity_cap_hooks SET active = 0 WHERE subject_value = ?`).run(
        'not-an-ip-hook',
      );
      const converged = await runNetworkIdentityBackfill(wrapped, secrets, NOW);
      expect(converged.rawHooks).toBe(0);
      expect(converged.finalized).toBe(true);
    } finally {
      db.close();
    }
  });
});

describe('network identity purge gate', () => {
  it('holds the purge gate open for the longest operator hook window until backfill converges', async () => {
    const provision = createTestBrokerEnv();
    const secrets = resolveNetworkIdentitySecrets(provision)!;
    const db = new DatabaseSync(':memory:');
    try {
      const { BROKER_MIGRATION_FILENAMES, readBrokerMigrationSql } = await import(
        './test-support/migrations'
      );
      for (const file of BROKER_MIGRATION_FILENAMES) {
        if (file <= '0019_managed_referral_settlement.sql') {
          db.exec(readBrokerMigrationSql(file));
        }
      }
      db.prepare(
        `INSERT INTO broker_velocity_cap_hooks (
          subject_type, subject_value, max_requests, window_minutes,
          outcome_code, outcome_class, active
        ) VALUES ('ip', ?, 1, 10080, 'rate_limited', 'retryable', 1)`,
      ).run('203.0.113.102');
      const threeDaysAgo = new Date(NOW.getTime() - 3 * 24 * 60 * 60_000).toISOString();
      db.prepare(
        `INSERT INTO broker_request_events (endpoint, ip, installation_id, observed_at)
         VALUES (?, ?, ?, ?)`,
      ).run('POST /v1/auth/qq/assert', '203.0.113.102', null, threeDaysAgo);
      for (const file of BROKER_MIGRATION_FILENAMES) {
        if (file === '0020_network_identity_hmac.sql') {
          db.exec(readBrokerMigrationSql(file));
        }
      }
      const phase = db
        .prepare(`SELECT value FROM broker_config WHERE key = 'network_identity_migration'`)
        .get() as { value: string };
      expect(JSON.parse(phase.value)).toMatchObject({ phase: 'dual_write' });

      expect(() => db.exec(readBrokerMigrationSql('0021_network_identity_purge.sql'))).toThrow(
        /constraint/i,
      );

      const wrapped = wrapDatabaseSync(db);
      const backfilled = await runNetworkIdentityBackfill(wrapped, secrets, NOW);
      expect(backfilled.requestEventsBackfilled).toBe(1);
      expect(backfilled.hooksConverted).toBe(1);
      expect(backfilled.finalized).toBe(true);

      db.exec(readBrokerMigrationSql('0021_network_identity_purge.sql'));
      const columns = db.prepare(`PRAGMA table_info(broker_request_events)`).all() as Array<{
        name: string;
      }>;
      expect(columns.map((column) => column.name)).not.toContain('ip');
    } finally {
      db.close();
    }
  });
});

describe('network identity staged migration', () => {
  it('preserves active windows through dual-write, backfill, and gated purge', async () => {
    const db = new DatabaseSync(':memory:');
    try {
      const files = (await import('./test-support/migrations')).BROKER_MIGRATION_FILENAMES;
      const { readBrokerMigrationSql } = await import('./test-support/migrations');
      for (const file of files) {
        if (file <= '0019_managed_referral_settlement.sql') {
          db.exec(readBrokerMigrationSql(file));
        }
      }
      db.prepare(
        `INSERT INTO broker_request_events (endpoint, ip, installation_id, observed_at)
         VALUES (?, ?, ?, ?)`,
      ).run('POST /v1/auth/qq/assert', '203.0.113.100', null, NOW_ISO);

      for (const file of files) {
        if (file === '0020_network_identity_hmac.sql') {
          db.exec(readBrokerMigrationSql(file));
        }
      }
      const phase = db
        .prepare(`SELECT value FROM broker_config WHERE key = 'network_identity_migration'`)
        .get() as { value: string };
      expect(JSON.parse(phase.value)).toMatchObject({ phase: 'dual_write' });

      for (const file of files) {
        if (file === '0021_network_identity_purge.sql') {
          expect(() => db.exec(readBrokerMigrationSql(file))).toThrow(/constraint/i);
        }
      }
    } finally {
      db.close();
    }
  });

  it('backfills raw request rows with the worker secret and finalizes after windows expire', async () => {
    const env = createTestBrokerEnv();
    const db = new DatabaseSync(':memory:');
    try {
      const { BROKER_MIGRATION_FILENAMES, readBrokerMigrationSql } = await import(
        './test-support/migrations'
      );
      for (const file of BROKER_MIGRATION_FILENAMES) {
        if (file <= '0019_managed_referral_settlement.sql') {
          db.exec(readBrokerMigrationSql(file));
        }
      }
      db.prepare(
        `INSERT INTO broker_request_events (endpoint, ip, installation_id, observed_at)
         VALUES (?, ?, ?, ?)`,
      ).run('POST /v1/auth/qq/assert', '203.0.113.101', null, NOW_ISO);
      db.prepare(
        `INSERT INTO referral_rewards (
          referral_id, referred_source, referred_subject_ref, referred_installation_id,
          referred_hardware_hash, referred_hardware_hash_salt_version,
          referred_bonus_status, referrer_bonus_status, skip_reason, attempt_ip_hash, created_at, updated_at
        ) VALUES (?, 'discord', ?, ?, ?, 7, 'skipped', 'skipped', 'unknown_referral_id', ?, ?, ?)`,
      ).run(
        '9ABCDX',
        'ph-discord-user-v1_staged_legacy',
        'install-staged-legacy',
        'hardware-staged-legacy',
        'b'.repeat(64),
        NOW_ISO,
        NOW_ISO,
      );
      for (const file of BROKER_MIGRATION_FILENAMES) {
        if (file === '0020_network_identity_hmac.sql') {
          db.exec(readBrokerMigrationSql(file));
        }
      }
      const stagedPhase = db
        .prepare(`SELECT value FROM broker_config WHERE key = 'network_identity_migration'`)
        .get() as { value: string };
      expect(JSON.parse(stagedPhase.value)).toMatchObject({ phase: 'dual_write' });

      const secrets = resolveNetworkIdentitySecrets(env);
      const wrapped = wrapDatabaseSync(db);
      const first = await runNetworkIdentityBackfill(wrapped, secrets, NOW);
      expect(first.requestEventsBackfilled).toBe(1);
      expect(first.finalized).toBe(false);

      const later = new Date(NOW.getTime() + 25 * 60 * 60_000);
      const second = await runNetworkIdentityBackfill(wrapped, secrets, later);
      expect(second.finalized).toBe(true);

      const legacy = db
        .prepare(`SELECT COUNT(*) AS count FROM broker_request_events WHERE ip IS NOT NULL`)
        .get() as { count: number };
      expect(legacy.count).toBe(0);

      db.exec(readBrokerMigrationSql('0021_network_identity_purge.sql'));
      const columns = db
        .prepare("SELECT name FROM pragma_table_info('broker_request_events') ORDER BY cid")
        .all() as Array<{ name: string }>;
      expect(columns.map((column) => column.name)).not.toContain('ip');
    } finally {
      db.close();
    }
  });
});

function wrapDatabaseSync(db: DatabaseSync): D1Database {
  const bound = (sql: string, params: unknown[]) => ({
    first: async () => db.prepare(sql).get(...(params as [])) ?? null,
    run: async () => {
      const result = db.prepare(sql).run(...(params as [])) as unknown as {
        changes: number | bigint;
      };
      return { meta: { changes: Number(result.changes ?? 0) } };
    },
    all: async <T,>() => ({ results: (db.prepare(sql).all(...(params as [])) as T[]) ?? [] }),
  });
  return {
    prepare: (sql: string) => ({
      bind: (...params: unknown[]) => bound(sql, params),
      first: async () => db.prepare(sql).get() ?? null,
      run: async () => {
        const result = db.prepare(sql).run() as unknown as { changes: number | bigint };
        return { meta: { changes: Number(result.changes ?? 0) } };
      },
      all: async <T,>() => ({ results: (db.prepare(sql).all() as T[]) ?? [] }),
    }),
  } as unknown as D1Database;
}
