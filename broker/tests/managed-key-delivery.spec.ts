import { describe, expect, it, vi, afterEach } from 'vitest';

vi.mock('../src/openrouter-management', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../src/openrouter-management')>()),
  cleanupManagedChildKey: vi.fn(),
}));

import app from '../src/index';
import {
  acknowledgeManagedKeyDelivery,
  claimStaleManagedKeyDeliveryCleanup,
  createManagedKeyDelivery,
  hashDeliveryAckToken,
  listStalePendingManagedKeyDeliveries,
} from '../src/managed-key-delivery';
import { cleanupManagedChildKey } from '../src/openrouter-management';
import { handleScheduled, reconcileStaleManagedKeyDeliveries } from '../src/scheduled';
import { normalizedErrorEnvelope } from './test-support/errors';
import { BROKER_MIGRATION_FILENAMES } from './test-support/migrations';
import { createTestBrokerEnv } from './test-support/sqlite-d1';

describe('managed key delivery ACK foundation', () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.clearAllMocks();
    vi.unstubAllGlobals();
  });

  it('orders migration 0012 after telemetry 0011 and creates delivery ACK schema', () => {
    const env = createTestBrokerEnv();

    expect(BROKER_MIGRATION_FILENAMES.at(-4)).toBe('0011_add_telemetry_active_days.sql');
    expect(BROKER_MIGRATION_FILENAMES.at(-3)).toBe('0012_add_managed_key_delivery_ack.sql');
    expect(BROKER_MIGRATION_FILENAMES.at(-2)).toBe(
      '0013_add_telemetry_subjects_and_daily_summary_v2.sql',
    );
    expect(BROKER_MIGRATION_FILENAMES.at(-1)).toBe(
      '0014_simplify_abuse_incidents.sql',
    );

    env.__db
      .prepare(
        `INSERT INTO installations (installation_id, device_public_key, app_version)
         VALUES (?, ?, ?)`,
      )
      .run('discord-installation', 'device-public-key', '1.0.0');

    const discordDeliveryPending = env.__db
      .prepare(
        `INSERT INTO openrouter_entitlements (
            installation_id,
            status,
            budget_usd,
            discord_issue_status
          ) VALUES (?, 'pending_release', 0, 'delivery_pending')`,
      )
      .run('discord-installation');
    expect(Number(discordDeliveryPending.changes)).toBe(1);

    const qqDeliveryPending = env.__db
      .prepare(
        `INSERT INTO qq_managed_entitlements (
            qq_subject_ref,
            status,
            issue_ref,
            managed_credential_ref,
            budget_usd,
            reserved_at,
            issued_at,
            expires_at
          ) VALUES (?, 'delivery_pending', ?, ?, 0, ?, ?, ?)`,
      )
      .run(
        'ph-qq-subject-v1_schema',
        'qq-issue-schema',
        'managed-credential-schema',
        '2026-07-05T00:00:00.000Z',
        '2026-07-05T00:00:00.000Z',
        '2026-08-05T00:00:00.000Z',
      );
    expect(Number(qqDeliveryPending.changes)).toBe(1);

    const deliveryColumns = env.__db
      .prepare("SELECT name FROM pragma_table_info('managed_key_deliveries') ORDER BY cid")
      .all() as Array<{ name: string }>;
    expect(deliveryColumns.map(({ name }) => name)).toEqual([
      'delivery_id',
      'issue_source',
      'subject_ref',
      'installation_id',
      'managed_credential_ref',
      'ack_token_hash',
      'status',
      'created_at',
      'expires_at',
      'acknowledged_at',
      'failed_at',
      'failure_reason',
    ]);

    const indexes = env.__db
      .prepare("SELECT name FROM pragma_index_list('managed_key_deliveries') ORDER BY name")
      .all() as Array<{ name: string }>;
    expect(indexes.map(({ name }) => name)).toEqual(
      expect.arrayContaining([
        'idx_managed_key_deliveries_issue_source_created_at',
        'idx_managed_key_deliveries_managed_credential_ref',
        'idx_managed_key_deliveries_status_expires_at',
      ]),
    );
  });

  it('creates pending delivery rows with hashed ACK tokens only', async () => {
    const env = createTestBrokerEnv();
    const createdAt = new Date('2026-07-05T00:00:00.000Z');
    const expiresAt = new Date('2026-07-05T00:10:00.000Z');

    const delivery = await createManagedKeyDelivery(env.BROKER_DB, {
      issueSource: 'discord',
      subjectRef: 'ph-discord-user-v1_subject',
      installationId: 'installation-1',
      managedCredentialRef: 'managed-credential-1',
      createdAt,
      expiresAt,
    });

    const row = env.__db
      .prepare('SELECT * FROM managed_key_deliveries WHERE delivery_id = ?')
      .get(delivery.deliveryId) as { ack_token_hash: string; status: string; created_at: string; expires_at: string };

    expect(row.status).toBe('pending');
    expect(row.created_at).toBe(createdAt.toISOString());
    expect(row.expires_at).toBe(expiresAt.toISOString());
    expect(row.ack_token_hash).toBe(await hashDeliveryAckToken(delivery.deliveryAckToken));
    expect(row.ack_token_hash).not.toBe(delivery.deliveryAckToken);
    expect(JSON.stringify(row)).not.toContain(delivery.deliveryAckToken);
  });

  it('acknowledges once and treats duplicate valid ACK as idempotent', async () => {
    const env = createTestBrokerEnv();
    const delivery = await createManagedKeyDelivery(env.BROKER_DB, {
      issueSource: 'qq',
      subjectRef: 'ph-qq-subject-v1_subject',
      managedCredentialRef: 'managed-credential-2',
      createdAt: new Date('2026-07-05T00:00:00.000Z'),
      expiresAt: new Date('2026-07-05T00:10:00.000Z'),
    });

    const first = await acknowledgeManagedKeyDelivery(env.BROKER_DB, {
      deliveryId: delivery.deliveryId,
      managedCredentialRef: 'managed-credential-2',
      deliveryAckToken: delivery.deliveryAckToken,
      now: new Date('2026-07-05T00:01:00.000Z'),
    });
    const second = await acknowledgeManagedKeyDelivery(env.BROKER_DB, {
      deliveryId: delivery.deliveryId,
      managedCredentialRef: 'managed-credential-2',
      deliveryAckToken: delivery.deliveryAckToken,
      now: new Date('2026-07-05T00:02:00.000Z'),
    });

    expect(first).toEqual({ ok: true, status: 'acknowledged' });
    expect(second).toEqual({ ok: true, status: 'already_acknowledged' });
    expect(
      env.__db
        .prepare("SELECT COUNT(*) AS count FROM managed_key_deliveries WHERE status = 'acknowledged'")
        .get(),
    ).toEqual({ count: 1 });
  });

  it('lists stale pending deliveries for cleanup helpers', async () => {
    const env = createTestBrokerEnv();
    const stale = await createManagedKeyDelivery(env.BROKER_DB, {
      issueSource: 'discord',
      managedCredentialRef: 'managed-credential-stale',
      createdAt: new Date('2026-07-05T00:00:00.000Z'),
      expiresAt: new Date('2026-07-05T00:05:00.000Z'),
    });
    await createManagedKeyDelivery(env.BROKER_DB, {
      issueSource: 'discord',
      managedCredentialRef: 'managed-credential-fresh',
      createdAt: new Date('2026-07-05T00:00:00.000Z'),
      expiresAt: new Date('2026-07-05T00:30:00.000Z'),
    });

    const staleRows = await listStalePendingManagedKeyDeliveries(env.BROKER_DB, {
      now: new Date('2026-07-05T00:10:00.000Z'),
      limit: 10,
    });

    expect(staleRows.map((row) => row.delivery_id)).toEqual([stale.deliveryId]);
  });

  it('serves ACK route safe public errors and keeps missing-owner ACK pending', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-07-05T00:01:00.000Z'));
    const env = createTestBrokerEnv();
    const delivery = await createManagedKeyDelivery(env.BROKER_DB, {
      issueSource: 'discord',
      managedCredentialRef: 'managed-credential-route',
      createdAt: new Date('2026-07-05T00:00:00.000Z'),
      expiresAt: new Date('2026-07-05T00:10:00.000Z'),
    });

    const payload = {
      delivery_id: delivery.deliveryId,
      managed_credential_ref: 'managed-credential-route',
      delivery_ack_token: delivery.deliveryAckToken,
    };
    const missingOwner = await postAck(env, payload);
    const invalid = await postAck(env, { ...payload, delivery_ack_token: 'wrong-token' });
    const mismatched = await postAck(env, { ...payload, managed_credential_ref: 'other-credential' });
    const malformed = await postAck(env, { ...payload, delivery_ack_token: '' });

    expect(missingOwner.status).toBe(409);
    await expect(missingOwner.json()).resolves.toMatchObject({
      error: { subcode: 'delivery_ack_failed' },
    });
    expect(
      env.__db
        .prepare('SELECT status, acknowledged_at FROM managed_key_deliveries WHERE delivery_id = ?')
        .get(delivery.deliveryId),
    ).toEqual({ status: 'pending', acknowledged_at: null });
    expect(invalid.status).toBe(404);
    const invalidBody = await invalid.json();
    expect(invalidBody).toEqual(
      normalizedErrorEnvelope({
        code: 'invalid_request',
        class: 'terminal',
        subcode: 'delivery_ack_invalid',
        message: 'delivery acknowledgement is invalid',
      }),
    );
    expect(mismatched.status).toBe(409);
    await expect(mismatched.json()).resolves.toMatchObject({
      error: { subcode: 'delivery_ack_mismatched' },
    });
    expect(malformed.status).toBe(400);
    await expect(malformed.json()).resolves.toMatchObject({
      error: { subcode: 'delivery_ack_malformed' },
    });
    expect(JSON.stringify(invalidBody)).not.toContain('wrong-token');
  });

  it('rejects expired ACK route attempts while leaving delivery pending for cleanup', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-07-05T00:11:00.000Z'));
    const env = createTestBrokerEnv();
    const delivery = await createManagedKeyDelivery(env.BROKER_DB, {
      issueSource: 'qq',
      managedCredentialRef: 'managed-credential-expired',
      createdAt: new Date('2026-07-05T00:00:00.000Z'),
      expiresAt: new Date('2026-07-05T00:10:00.000Z'),
    });

    const response = await postAck(env, {
      delivery_id: delivery.deliveryId,
      managed_credential_ref: 'managed-credential-expired',
      delivery_ack_token: delivery.deliveryAckToken,
    });

    expect(response.status).toBe(410);
    await expect(response.json()).resolves.toMatchObject({
      error: { subcode: 'delivery_ack_expired' },
    });
    expect(
      env.__db
        .prepare('SELECT status, failure_reason FROM managed_key_deliveries WHERE delivery_id = ?')
        .get(delivery.deliveryId),
    ).toEqual({ status: 'pending', failure_reason: null });

    const staleRows = await listStalePendingManagedKeyDeliveries(env.BROKER_DB, {
      now: new Date('2026-07-05T00:11:00.000Z'),
      limit: 10,
    });
    expect(staleRows.map((row) => row.delivery_id)).toEqual([delivery.deliveryId]);
  });

  it('finalizes Discord delivery only after a valid ACK and keeps duplicate ACK idempotent', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-07-05T00:01:00.000Z'));
    const env = createTestBrokerEnv();
    insertDiscordDeliveryPendingOwner(env, {
      installationId: 'discord-install-ack',
      discordUserRef: 'ph-discord-user-v1_ack',
      managedCredentialRef: 'hash_discord_ack',
    });
    const delivery = await createManagedKeyDelivery(env.BROKER_DB, {
      issueSource: 'discord',
      subjectRef: 'ph-discord-user-v1_ack',
      installationId: 'discord-install-ack',
      managedCredentialRef: 'hash_discord_ack',
      createdAt: new Date('2026-07-05T00:00:00.000Z'),
      expiresAt: new Date('2026-07-05T00:15:00.000Z'),
    });

    const invalid = await postAck(env, {
      delivery_id: delivery.deliveryId,
      managed_credential_ref: 'hash_discord_ack',
      delivery_ack_token: 'invalid-token',
    });
    expect(invalid.status).toBe(404);
    expect(selectScalar(env, "SELECT COUNT(*) FROM broker_issue_success_events WHERE managed_credential_ref = 'hash_discord_ack'")).toBe(0);
    expect(selectDiscordEntitlement(env, 'hash_discord_ack')).toMatchObject({
      status: 'pending_release',
      discord_issue_status: 'delivery_pending',
      discord_issue_delivered_at: null,
    });

    const valid = await postAck(env, {
      delivery_id: delivery.deliveryId,
      managed_credential_ref: 'hash_discord_ack',
      delivery_ack_token: delivery.deliveryAckToken,
    });
    expect(valid.status).toBe(200);
    await expect(valid.json()).resolves.toEqual({ ok: true, status: 'acknowledged' });
    expect(selectDiscordEntitlement(env, 'hash_discord_ack')).toMatchObject({
      status: 'active',
      discord_issue_status: 'active',
      discord_issue_delivered_at: '2026-07-05T00:01:00.000Z',
    });
    expect(selectDiscordIdentityStatus(env, 'ph-discord-user-v1_ack', 'discord-install-ack')).toBe('active');
    expect(selectScalar(env, "SELECT COUNT(*) FROM broker_issue_success_events WHERE managed_credential_ref = 'hash_discord_ack'")).toBe(1);

    const duplicate = await postAck(env, {
      delivery_id: delivery.deliveryId,
      managed_credential_ref: 'hash_discord_ack',
      delivery_ack_token: delivery.deliveryAckToken,
    });
    expect(duplicate.status).toBe(200);
    await expect(duplicate.json()).resolves.toEqual({ ok: true, status: 'already_acknowledged' });
    expect(selectScalar(env, "SELECT COUNT(*) FROM broker_issue_success_events WHERE managed_credential_ref = 'hash_discord_ack'")).toBe(1);
  });

  it('serializes concurrent Discord ACK finalization into one delivery event', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-07-05T00:01:00.000Z'));
    const env = createTestBrokerEnv();
    insertDiscordDeliveryPendingOwner(env, {
      installationId: 'discord-install-concurrent-ack',
      discordUserRef: 'ph-discord-user-v1_concurrent_ack',
      managedCredentialRef: 'hash_discord_concurrent_ack',
    });
    const delivery = await createManagedKeyDelivery(env.BROKER_DB, {
      issueSource: 'discord',
      subjectRef: 'ph-discord-user-v1_concurrent_ack',
      installationId: 'discord-install-concurrent-ack',
      managedCredentialRef: 'hash_discord_concurrent_ack',
      createdAt: new Date('2026-07-05T00:00:00.000Z'),
      expiresAt: new Date('2026-07-05T00:15:00.000Z'),
    });
    const payload = {
      delivery_id: delivery.deliveryId,
      managed_credential_ref: 'hash_discord_concurrent_ack',
      delivery_ack_token: delivery.deliveryAckToken,
    };

    const responses = await Promise.all([postAck(env, payload), postAck(env, payload)]);
    expect(responses.map(({ status }) => status)).toEqual([200, 200]);
    const bodies = await Promise.all(responses.map((response) => response.json()));
    expect(bodies.map((body) => (body as { status: string }).status).sort()).toEqual([
      'acknowledged',
      'already_acknowledged',
    ]);
    expect(selectScalar(env, "SELECT COUNT(*) FROM broker_issue_success_events WHERE managed_credential_ref = 'hash_discord_concurrent_ack'")).toBe(1);
    expect(selectDiscordEntitlement(env, 'hash_discord_concurrent_ack')).toMatchObject({
      status: 'active',
      discord_issue_status: 'active',
    });
  });

  it('rolls Discord ownership back when issue-success persistence fails', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-07-05T00:01:00.000Z'));
    let rejectIssueSuccess = true;
    const env = createTestBrokerEnv({
      beforeRun: ({ sql }) => {
        if (
          rejectIssueSuccess &&
          sql.includes('INSERT INTO broker_issue_success_events')
        ) {
          throw new Error('issue-success persistence unavailable');
        }
      },
    });
    insertDiscordDeliveryPendingOwner(env, {
      installationId: 'discord-install-retry-ack',
      discordUserRef: 'ph-discord-user-v1_retry_ack',
      managedCredentialRef: 'hash_discord_retry_ack',
    });
    const delivery = await createManagedKeyDelivery(env.BROKER_DB, {
      issueSource: 'discord',
      subjectRef: 'ph-discord-user-v1_retry_ack',
      installationId: 'discord-install-retry-ack',
      managedCredentialRef: 'hash_discord_retry_ack',
      createdAt: new Date('2026-07-05T00:00:00.000Z'),
      expiresAt: new Date('2026-07-05T00:15:00.000Z'),
    });

    const failed = await postAck(env, {
      delivery_id: delivery.deliveryId,
      managed_credential_ref: 'hash_discord_retry_ack',
      delivery_ack_token: delivery.deliveryAckToken,
    });

    expect(failed.status).toBe(409);
    expect(selectDiscordEntitlement(env, 'hash_discord_retry_ack')).toMatchObject({
      status: 'pending_release',
      discord_issue_status: 'delivery_pending',
      discord_issue_delivered_at: null,
    });
    expect(
      selectDiscordIdentityStatus(
        env,
        'ph-discord-user-v1_retry_ack',
        'discord-install-retry-ack',
      ),
    ).toBe('issuing');
    expect(selectScalar(env, "SELECT COUNT(*) FROM broker_issue_success_events WHERE managed_credential_ref = 'hash_discord_retry_ack'")).toBe(0);

    rejectIssueSuccess = false;
    const retried = await postAck(env, {
      delivery_id: delivery.deliveryId,
      managed_credential_ref: 'hash_discord_retry_ack',
      delivery_ack_token: delivery.deliveryAckToken,
    });

    expect(retried.status).toBe(200);
    await expect(retried.json()).resolves.toEqual({ ok: true, status: 'acknowledged' });
    expect(selectScalar(env, "SELECT COUNT(*) FROM broker_issue_success_events WHERE managed_credential_ref = 'hash_discord_retry_ack'")).toBe(1);
  });

  it('keeps Discord owner, event, and ledger finalization atomic when ledger persistence fails', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-07-05T00:01:00.000Z'));
    let rejectAckLedgerUpdate = true;
    const env = createTestBrokerEnv({
      beforeRun: ({ sql }) => {
        if (
          rejectAckLedgerUpdate &&
          sql.includes('UPDATE managed_key_deliveries') &&
          sql.includes("SET status = 'acknowledged'")
        ) {
          throw new Error('ACK ledger persistence unavailable');
        }
      },
    });
    insertDiscordDeliveryPendingOwner(env, {
      installationId: 'discord-install-ledger-repair',
      discordUserRef: 'ph-discord-user-v1_ledger_repair',
      managedCredentialRef: 'hash_discord_ledger_repair',
    });
    const delivery = await createManagedKeyDelivery(env.BROKER_DB, {
      issueSource: 'discord',
      subjectRef: 'ph-discord-user-v1_ledger_repair',
      installationId: 'discord-install-ledger-repair',
      managedCredentialRef: 'hash_discord_ledger_repair',
      createdAt: new Date('2026-07-05T00:00:00.000Z'),
      expiresAt: new Date('2026-07-05T00:15:00.000Z'),
    });

    const failed = await postAck(env, {
      delivery_id: delivery.deliveryId,
      managed_credential_ref: 'hash_discord_ledger_repair',
      delivery_ack_token: delivery.deliveryAckToken,
    });

    expect(failed.status).toBe(409);
    expect(selectDiscordEntitlement(env, 'hash_discord_ledger_repair')).toMatchObject({
      status: 'pending_release',
      discord_issue_status: 'delivery_pending',
      discord_issue_delivered_at: null,
    });
    expect(
      env.__db
        .prepare('SELECT status FROM managed_key_deliveries WHERE delivery_id = ?')
        .get(delivery.deliveryId),
    ).toEqual({ status: 'pending' });
    expect(selectScalar(env, "SELECT COUNT(*) FROM broker_issue_success_events WHERE managed_credential_ref = 'hash_discord_ledger_repair'")).toBe(0);

    rejectAckLedgerUpdate = false;
    const retried = await postAck(env, {
      delivery_id: delivery.deliveryId,
      managed_credential_ref: 'hash_discord_ledger_repair',
      delivery_ack_token: delivery.deliveryAckToken,
    });

    expect(retried.status).toBe(200);
    await expect(retried.json()).resolves.toEqual({
      ok: true,
      status: 'acknowledged',
    });
    expect(cleanupManagedChildKey).not.toHaveBeenCalled();
    expect(
      env.__db
        .prepare('SELECT status FROM managed_key_deliveries WHERE delivery_id = ?')
        .get(delivery.deliveryId),
    ).toEqual({ status: 'acknowledged' });
    expect(selectDiscordEntitlement(env, 'hash_discord_ledger_repair')).toMatchObject({
      status: 'active',
      discord_issue_status: 'active',
    });
  });

  it('finalizes QQ delivery only after a valid ACK and keeps duplicate ACK idempotent', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-07-05T00:01:00.000Z'));
    const env = createTestBrokerEnv();
    insertQqDeliveryPendingOwner(env, {
      qqSubjectRef: 'ph-qq-subject-v1_ack',
      issueRef: 'qq-issue-ack',
      managedCredentialRef: 'hash_qq_ack',
    });
    const delivery = await createManagedKeyDelivery(env.BROKER_DB, {
      issueSource: 'qq',
      subjectRef: 'ph-qq-subject-v1_ack',
      managedCredentialRef: 'hash_qq_ack',
      createdAt: new Date('2026-07-05T00:00:00.000Z'),
      expiresAt: new Date('2026-07-05T00:15:00.000Z'),
    });

    const invalid = await postAck(env, {
      delivery_id: delivery.deliveryId,
      managed_credential_ref: 'hash_qq_ack',
      delivery_ack_token: 'invalid-token',
    });
    expect(invalid.status).toBe(404);
    expect(selectQqEntitlement(env, 'hash_qq_ack')).toMatchObject({
      status: 'delivery_pending',
      delivered_at: null,
    });

    const valid = await postAck(env, {
      delivery_id: delivery.deliveryId,
      managed_credential_ref: 'hash_qq_ack',
      delivery_ack_token: delivery.deliveryAckToken,
    });
    expect(valid.status).toBe(200);
    expect(selectQqEntitlement(env, 'hash_qq_ack')).toMatchObject({
      status: 'active',
      delivered_at: '2026-07-05T00:01:00.000Z',
    });
    expect(selectScalar(env, "SELECT COUNT(*) FROM broker_issue_success_events WHERE managed_credential_ref = 'hash_qq_ack'")).toBe(1);

    const duplicate = await postAck(env, {
      delivery_id: delivery.deliveryId,
      managed_credential_ref: 'hash_qq_ack',
      delivery_ack_token: delivery.deliveryAckToken,
    });
    expect(duplicate.status).toBe(200);
    await expect(duplicate.json()).resolves.toEqual({ ok: true, status: 'already_acknowledged' });
    expect(selectScalar(env, "SELECT COUNT(*) FROM broker_issue_success_events WHERE managed_credential_ref = 'hash_qq_ack'")).toBe(1);
  });

  it('serializes concurrent QQ ACK finalization into one delivery event', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-07-05T00:01:00.000Z'));
    const env = createTestBrokerEnv();
    insertQqDeliveryPendingOwner(env, {
      qqSubjectRef: 'ph-qq-subject-v1_concurrent_ack',
      issueRef: 'qq-issue-concurrent-ack',
      managedCredentialRef: 'hash_qq_concurrent_ack',
    });
    const delivery = await createManagedKeyDelivery(env.BROKER_DB, {
      issueSource: 'qq',
      subjectRef: 'ph-qq-subject-v1_concurrent_ack',
      managedCredentialRef: 'hash_qq_concurrent_ack',
      createdAt: new Date('2026-07-05T00:00:00.000Z'),
      expiresAt: new Date('2026-07-05T00:15:00.000Z'),
    });
    const payload = {
      delivery_id: delivery.deliveryId,
      managed_credential_ref: 'hash_qq_concurrent_ack',
      delivery_ack_token: delivery.deliveryAckToken,
    };

    const responses = await Promise.all([postAck(env, payload), postAck(env, payload)]);
    expect(responses.map(({ status }) => status)).toEqual([200, 200]);
    const bodies = await Promise.all(responses.map((response) => response.json()));
    expect(bodies.map((body) => (body as { status: string }).status).sort()).toEqual([
      'acknowledged',
      'already_acknowledged',
    ]);
    expect(selectScalar(env, "SELECT COUNT(*) FROM broker_issue_success_events WHERE managed_credential_ref = 'hash_qq_concurrent_ack'")).toBe(1);
    expect(selectQqEntitlement(env, 'hash_qq_concurrent_ack')).toMatchObject({
      status: 'active',
    });
  });

  it('keeps QQ acknowledgement pending until issue-success recording is durable', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-07-05T00:01:00.000Z'));
    let rejectIssueSuccess = true;
    const env = createTestBrokerEnv({
      beforeRun: ({ sql }) => {
        if (
          rejectIssueSuccess &&
          sql.includes('INSERT INTO broker_issue_success_events')
        ) {
          throw new Error('issue-success persistence unavailable');
        }
      },
    });
    insertQqDeliveryPendingOwner(env, {
      qqSubjectRef: 'ph-qq-subject-v1_retry_ack',
      issueRef: 'qq-issue-retry-ack',
      managedCredentialRef: 'hash_qq_retry_ack',
    });
    const delivery = await createManagedKeyDelivery(env.BROKER_DB, {
      issueSource: 'qq',
      subjectRef: 'ph-qq-subject-v1_retry_ack',
      managedCredentialRef: 'hash_qq_retry_ack',
      createdAt: new Date('2026-07-05T00:00:00.000Z'),
      expiresAt: new Date('2026-07-05T00:15:00.000Z'),
    });

    const failed = await postAck(env, {
      delivery_id: delivery.deliveryId,
      managed_credential_ref: 'hash_qq_retry_ack',
      delivery_ack_token: delivery.deliveryAckToken,
    });

    expect(failed.status).toBe(409);
    expect(
      env.__db
        .prepare('SELECT status FROM managed_key_deliveries WHERE delivery_id = ?')
        .get(delivery.deliveryId),
    ).toEqual({ status: 'pending' });
    expect(selectQqEntitlement(env, 'hash_qq_retry_ack')).toMatchObject({
      status: 'delivery_pending',
      delivered_at: null,
    });
    expect(selectScalar(env, "SELECT COUNT(*) FROM broker_issue_success_events WHERE managed_credential_ref = 'hash_qq_retry_ack'")).toBe(0);

    rejectIssueSuccess = false;
    const retried = await postAck(env, {
      delivery_id: delivery.deliveryId,
      managed_credential_ref: 'hash_qq_retry_ack',
      delivery_ack_token: delivery.deliveryAckToken,
    });

    expect(retried.status).toBe(200);
    await expect(retried.json()).resolves.toEqual({ ok: true, status: 'acknowledged' });
    expect(selectScalar(env, "SELECT COUNT(*) FROM broker_issue_success_events WHERE managed_credential_ref = 'hash_qq_retry_ack'")).toBe(1);
  });

  it('keeps QQ owner, event, and ledger finalization atomic when ledger persistence fails', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-07-05T00:01:00.000Z'));
    let rejectAckLedgerUpdate = true;
    const env = createTestBrokerEnv({
      beforeRun: ({ sql }) => {
        if (
          rejectAckLedgerUpdate &&
          sql.includes('UPDATE managed_key_deliveries') &&
          sql.includes("SET status = 'acknowledged'")
        ) {
          throw new Error('ACK ledger persistence unavailable');
        }
      },
    });
    insertQqDeliveryPendingOwner(env, {
      qqSubjectRef: 'ph-qq-subject-v1_ledger_repair',
      issueRef: 'qq-issue-ledger-repair',
      managedCredentialRef: 'hash_qq_ledger_repair',
    });
    const delivery = await createManagedKeyDelivery(env.BROKER_DB, {
      issueSource: 'qq',
      subjectRef: 'ph-qq-subject-v1_ledger_repair',
      managedCredentialRef: 'hash_qq_ledger_repair',
      createdAt: new Date('2026-07-05T00:00:00.000Z'),
      expiresAt: new Date('2026-07-05T00:15:00.000Z'),
    });

    const failed = await postAck(env, {
      delivery_id: delivery.deliveryId,
      managed_credential_ref: 'hash_qq_ledger_repair',
      delivery_ack_token: delivery.deliveryAckToken,
    });

    expect(failed.status).toBe(409);
    expect(selectQqEntitlement(env, 'hash_qq_ledger_repair')).toMatchObject({
      status: 'delivery_pending',
      delivered_at: null,
    });
    expect(
      env.__db
        .prepare('SELECT status FROM managed_key_deliveries WHERE delivery_id = ?')
        .get(delivery.deliveryId),
    ).toEqual({ status: 'pending' });
    expect(selectScalar(env, "SELECT COUNT(*) FROM broker_issue_success_events WHERE managed_credential_ref = 'hash_qq_ledger_repair'")).toBe(0);

    rejectAckLedgerUpdate = false;
    const retried = await postAck(env, {
      delivery_id: delivery.deliveryId,
      managed_credential_ref: 'hash_qq_ledger_repair',
      delivery_ack_token: delivery.deliveryAckToken,
    });

    expect(retried.status).toBe(200);
    await expect(retried.json()).resolves.toEqual({
      ok: true,
      status: 'acknowledged',
    });
    expect(cleanupManagedChildKey).not.toHaveBeenCalled();
    expect(
      env.__db
        .prepare('SELECT status FROM managed_key_deliveries WHERE delivery_id = ?')
        .get(delivery.deliveryId),
    ).toEqual({ status: 'acknowledged' });
    expect(selectQqEntitlement(env, 'hash_qq_ledger_repair')).toMatchObject({
      status: 'active',
    });
  });

  it('stale cleanup releases Discord reservation identity on success and marks it cleanup_required on failure', async () => {
    const env = createTestBrokerEnv();
    insertDiscordDeliveryPendingOwner(env, {
      installationId: 'discord-install-stale-ok',
      discordUserRef: 'ph-discord-user-v1_stale_ok',
      managedCredentialRef: 'hash_discord_stale_ok',
    });
    insertDiscordDeliveryPendingOwner(env, {
      installationId: 'discord-install-stale-fail',
      discordUserRef: 'ph-discord-user-v1_stale_fail',
      managedCredentialRef: 'hash_discord_stale_fail',
    });
    await createManagedKeyDelivery(env.BROKER_DB, {
      issueSource: 'discord',
      subjectRef: 'ph-discord-user-v1_stale_ok',
      installationId: 'discord-install-stale-ok',
      managedCredentialRef: 'hash_discord_stale_ok',
      createdAt: new Date('2026-07-05T00:00:00.000Z'),
      expiresAt: new Date('2026-07-05T00:15:00.000Z'),
    });
    await createManagedKeyDelivery(env.BROKER_DB, {
      issueSource: 'discord',
      subjectRef: 'ph-discord-user-v1_stale_fail',
      installationId: 'discord-install-stale-fail',
      managedCredentialRef: 'hash_discord_stale_fail',
      createdAt: new Date('2026-07-05T00:00:00.000Z'),
      expiresAt: new Date('2026-07-05T00:15:00.000Z'),
    });
    vi.mocked(cleanupManagedChildKey)
      .mockResolvedValueOnce({ ok: true })
      .mockResolvedValueOnce({ ok: false, reason: cleanupFailureReason() });

    await reconcileStaleManagedKeyDeliveries(env, new Date('2026-07-05T00:16:00.000Z'));

    expect(selectDiscordEntitlement(env, 'hash_discord_stale_ok')).toBeNull();
    expect(selectDiscordIdentityStatus(env, 'ph-discord-user-v1_stale_ok', 'discord-install-stale-ok')).toBeNull();
    expect(selectDiscordEntitlement(env, 'hash_discord_stale_fail')).toMatchObject({
      discord_issue_status: 'cleanup_required',
    });
    expect(selectDiscordIdentityStatus(env, 'ph-discord-user-v1_stale_fail', 'discord-install-stale-fail')).toBe('cleanup_required');
  });

  it('stale cleanup marks QQ cleanup_required on cleanup failure to block no-overwrite reissue', async () => {
    const env = createTestBrokerEnv();
    const fetchMock = vi.fn(
      async (_input: RequestInfo | URL, _init?: RequestInit) =>
        new Response(null, { status: 204 }),
    );
    vi.stubGlobal('fetch', fetchMock);
    insertQqDeliveryPendingOwner(env, {
      qqSubjectRef: 'ph-qq-subject-v1_stale_fail',
      issueRef: 'qq-issue-stale-fail',
      managedCredentialRef: 'hash_qq_stale_fail',
    });
    await createManagedKeyDelivery(env.BROKER_DB, {
      issueSource: 'qq',
      subjectRef: 'ph-qq-subject-v1_stale_fail',
      managedCredentialRef: 'hash_qq_stale_fail',
      createdAt: new Date('2026-07-05T00:00:00.000Z'),
      expiresAt: new Date('2026-07-05T00:15:00.000Z'),
    });
    vi.mocked(cleanupManagedChildKey).mockResolvedValueOnce({ ok: false, reason: cleanupFailureReason() });

    await reconcileStaleManagedKeyDeliveries(env, new Date('2026-07-05T00:16:00.000Z'));

    expect(selectQqEntitlement(env, 'hash_qq_stale_fail')).toMatchObject({
      status: 'cleanup_required',
      delivered_at: null,
    });
    expect(selectScalar(env, "SELECT COUNT(*) FROM managed_key_deliveries WHERE managed_credential_ref = 'hash_qq_stale_fail' AND status = 'cleanup_required'")).toBe(1);
    expect(fetchMock).toHaveBeenCalledOnce();
    expect(String(fetchMock.mock.calls[0]?.[0])).toBe(
      env.DISCORD_IMMEDIATE_ALERT_WEBHOOK_URL,
    );
    const cleanupIncidentBody = String(fetchMock.mock.calls[0]?.[1]?.body);
    expect(cleanupIncidentBody).toContain('Broker managed-key cleanup incident');
    expect(cleanupIncidentBody).toContain('stale_delivery');
  });

  it('notifies a stale cleanup state failure while scheduled reporting and retention still progress', async () => {
    const env = createTestBrokerEnv({
      beforeRun: ({ sql }) => {
        if (
          sql.includes('UPDATE qq_managed_entitlements') &&
          sql.includes("SET status = 'cleanup_required'") &&
          sql.includes('managed_key_deliveries')
        ) {
          throw new Error('synthetic stale cleanup state failure');
        }
      },
    });
    const fetchMock = vi.fn(
      async (_input: RequestInfo | URL, _init?: RequestInit) =>
        new Response(null, { status: 204 }),
    );
    vi.stubGlobal('fetch', fetchMock);
    insertQqDeliveryPendingOwner(env, {
      qqSubjectRef: 'ph-qq-subject-v1_stale_state_failure',
      issueRef: 'qq-issue-stale-state-failure',
      managedCredentialRef: 'hash_qq_stale_state_failure',
    });
    await createManagedKeyDelivery(env.BROKER_DB, {
      issueSource: 'qq',
      subjectRef: 'ph-qq-subject-v1_stale_state_failure',
      managedCredentialRef: 'hash_qq_stale_state_failure',
      createdAt: new Date('2026-07-05T00:00:00.000Z'),
      expiresAt: new Date('2026-07-05T00:15:00.000Z'),
    });
    env.__db
      .prepare(
        `INSERT INTO broker_request_events (
            endpoint, ip, installation_id, observed_at
          ) VALUES (?, ?, ?, ?)`,
      )
      .run(
        'POST /v1/auth/qq/assert',
        '203.0.113.210',
        null,
        '2025-01-01T00:00:00.000Z',
      );
    vi.mocked(cleanupManagedChildKey).mockResolvedValueOnce({
      ok: false,
      reason: cleanupFailureReason(),
    });

    await expect(
      handleScheduled(
        { scheduledTime: Date.parse('2026-07-06T00:05:00.000Z') },
        env,
        {},
      ),
    ).rejects.toThrow('synthetic stale cleanup state failure');

    expect(
      fetchMock.mock.calls.map(([request]) => String(request)),
    ).toEqual([
      env.DISCORD_IMMEDIATE_ALERT_WEBHOOK_URL,
      env.DISCORD_DAILY_REPORT_WEBHOOK_URL,
    ]);
    expect(String(fetchMock.mock.calls[0]?.[1]?.body)).toContain(
      'cleanup_required state could not be confirmed',
    );
    expect(
      env.__db
        .prepare(
          `SELECT status
             FROM broker_daily_summary_deliveries
            WHERE report_date_utc = '2026-07-05'`,
        )
        .get(),
    ).toEqual({ status: 'delivered' });
    expect(selectScalar(env, 'SELECT COUNT(*) FROM broker_request_events')).toBe(0);
    expect(selectQqEntitlement(env, 'hash_qq_stale_state_failure')).toMatchObject({
      status: 'delivery_pending',
    });
  });

  it('claims a stale delivery once across concurrent reconcilers', async () => {
    const env = createTestBrokerEnv();
    const fetchMock = vi.fn(
      async (_input: RequestInfo | URL, _init?: RequestInit) =>
        new Response(null, { status: 204 }),
    );
    vi.stubGlobal('fetch', fetchMock);
    insertQqDeliveryPendingOwner(env, {
      qqSubjectRef: 'ph-qq-subject-v1_concurrent_cleanup',
      issueRef: 'qq-issue-concurrent-cleanup',
      managedCredentialRef: 'hash_qq_concurrent_cleanup',
    });
    await createManagedKeyDelivery(env.BROKER_DB, {
      issueSource: 'qq',
      subjectRef: 'ph-qq-subject-v1_concurrent_cleanup',
      managedCredentialRef: 'hash_qq_concurrent_cleanup',
      createdAt: new Date('2026-07-05T00:00:00.000Z'),
      expiresAt: new Date('2026-07-05T00:15:00.000Z'),
    });
    vi.mocked(cleanupManagedChildKey).mockResolvedValue({
      ok: false,
      reason: cleanupFailureReason(),
    });

    const results = await Promise.all([
      reconcileStaleManagedKeyDeliveries(
        env,
        new Date('2026-07-05T00:16:00.000Z'),
      ),
      reconcileStaleManagedKeyDeliveries(
        env,
        new Date('2026-07-05T00:16:00.000Z'),
      ),
    ]);

    expect(cleanupManagedChildKey).toHaveBeenCalledOnce();
    expect(fetchMock).toHaveBeenCalledOnce();
    expect(results.reduce((sum, result) => sum + result.cleanupRequired, 0)).toBe(1);
  });

  it('fences a stale claimant after an abandoned cleanup claim is recovered', async () => {
    const env = createTestBrokerEnv();
    const fetchMock = vi.fn(
      async (_input: RequestInfo | URL, _init?: RequestInit) =>
        new Response(null, { status: 204 }),
    );
    vi.stubGlobal('fetch', fetchMock);
    insertQqDeliveryPendingOwner(env, {
      qqSubjectRef: 'ph-qq-subject-v1_recovered_claim',
      issueRef: 'qq-issue-recovered-claim',
      managedCredentialRef: 'hash_qq_recovered_claim',
    });
    await createManagedKeyDelivery(env.BROKER_DB, {
      issueSource: 'qq',
      subjectRef: 'ph-qq-subject-v1_recovered_claim',
      managedCredentialRef: 'hash_qq_recovered_claim',
      createdAt: new Date('2026-07-05T00:00:00.000Z'),
      expiresAt: new Date('2026-07-05T00:15:00.000Z'),
    });
    type CleanupResult = Awaited<ReturnType<typeof cleanupManagedChildKey>>;
    let resolveFirstCleanup!: (result: CleanupResult) => void;
    let resolveSecondCleanup!: (result: CleanupResult) => void;
    const firstCleanup = new Promise<CleanupResult>((resolve) => {
      resolveFirstCleanup = resolve;
    });
    const secondCleanup = new Promise<CleanupResult>((resolve) => {
      resolveSecondCleanup = resolve;
    });
    vi.mocked(cleanupManagedChildKey)
      .mockImplementationOnce(() => firstCleanup)
      .mockImplementationOnce(() => secondCleanup);

    const firstRun = reconcileStaleManagedKeyDeliveries(
      env,
      new Date('2026-07-05T00:16:00.000Z'),
    );
    await vi.waitFor(() => expect(cleanupManagedChildKey).toHaveBeenCalledTimes(1));
    const secondRun = reconcileStaleManagedKeyDeliveries(
      env,
      new Date('2026-07-05T00:33:00.000Z'),
    );
    await vi.waitFor(() => expect(cleanupManagedChildKey).toHaveBeenCalledTimes(2));

    resolveFirstCleanup({ ok: false, reason: cleanupFailureReason() });
    await expect(firstRun).resolves.toEqual({ expired: 0, cleanupRequired: 0 });
    expect(selectQqEntitlement(env, 'hash_qq_recovered_claim')).toMatchObject({
      status: 'delivery_pending',
    });
    expect(
      env.__db
        .prepare("SELECT status, failed_at FROM managed_key_deliveries WHERE managed_credential_ref = 'hash_qq_recovered_claim'")
        .get(),
    ).toEqual({
      status: 'expired',
      failed_at: '2026-07-05T00:33:00.000Z',
    });
    expect(fetchMock).not.toHaveBeenCalled();

    resolveSecondCleanup({ ok: false, reason: cleanupFailureReason() });
    await expect(secondRun).resolves.toEqual({ expired: 0, cleanupRequired: 1 });
    expect(selectQqEntitlement(env, 'hash_qq_recovered_claim')).toMatchObject({
      status: 'cleanup_required',
    });
    expect(fetchMock).toHaveBeenCalledOnce();
  });

  it('keeps an ACK activation in progress out of stale cleanup before its event is recorded', async () => {
    const env = createTestBrokerEnv();
    insertQqDeliveryPendingOwner(env, {
      qqSubjectRef: 'ph-qq-subject-v1_ack_in_progress',
      issueRef: 'qq-issue-ack-in-progress',
      managedCredentialRef: 'hash_qq_ack_in_progress',
    });
    await createManagedKeyDelivery(env.BROKER_DB, {
      issueSource: 'qq',
      subjectRef: 'ph-qq-subject-v1_ack_in_progress',
      managedCredentialRef: 'hash_qq_ack_in_progress',
      createdAt: new Date('2026-07-05T00:00:00.000Z'),
      expiresAt: new Date('2026-07-05T00:15:00.000Z'),
    });
    env.__db
      .prepare(
        `UPDATE qq_managed_entitlements
            SET status = 'active', delivered_at = '2026-07-05T00:15:59.000Z'
          WHERE managed_credential_ref = 'hash_qq_ack_in_progress'`,
      )
      .run();

    await expect(
      reconcileStaleManagedKeyDeliveries(
        env,
        new Date('2026-07-05T00:16:00.000Z'),
      ),
    ).resolves.toEqual({ expired: 0, cleanupRequired: 0 });
    expect(cleanupManagedChildKey).not.toHaveBeenCalled();
    expect(
      env.__db
        .prepare("SELECT status FROM managed_key_deliveries WHERE managed_credential_ref = 'hash_qq_ack_in_progress'")
        .get(),
    ).toEqual({ status: 'pending' });
  });

  it('prevents ACK owner activation after stale cleanup has claimed the ledger', async () => {
    const env = createTestBrokerEnv();
    insertQqDeliveryPendingOwner(env, {
      qqSubjectRef: 'ph-qq-subject-v1_cleanup_claim_first',
      issueRef: 'qq-issue-cleanup-claim-first',
      managedCredentialRef: 'hash_qq_cleanup_claim_first',
    });
    const delivery = await createManagedKeyDelivery(env.BROKER_DB, {
      issueSource: 'qq',
      subjectRef: 'ph-qq-subject-v1_cleanup_claim_first',
      managedCredentialRef: 'hash_qq_cleanup_claim_first',
      createdAt: new Date('2026-07-05T00:00:00.000Z'),
      expiresAt: new Date('2026-07-05T00:15:00.000Z'),
    });
    const [stale] = await listStalePendingManagedKeyDeliveries(env.BROKER_DB, {
      now: new Date('2026-07-05T00:16:00.000Z'),
      limit: 1,
    });
    expect(stale).toBeDefined();
    await expect(
      claimStaleManagedKeyDeliveryCleanup(env.BROKER_DB, {
        delivery: stale!,
        claimedAt: '2026-07-05T00:16:00.000Z',
      }),
    ).resolves.toBe(true);

    const response = await postAck(env, {
      delivery_id: delivery.deliveryId,
      delivery_ack_token: delivery.deliveryAckToken,
      managed_credential_ref: 'hash_qq_cleanup_claim_first',
    });

    expect(response.status).toBe(410);
    expect(selectQqEntitlement(env, 'hash_qq_cleanup_claim_first')).toMatchObject({
      status: 'delivery_pending',
      delivered_at: null,
    });
  });

  it('keeps a failed cleanup claim retryable until owner and ledger terminalize atomically', async () => {
    const env = createTestBrokerEnv();
    const fetchMock = vi.fn(
      async (_input: RequestInfo | URL, _init?: RequestInit) =>
        new Response(null, { status: 204 }),
    );
    vi.stubGlobal('fetch', fetchMock);
    insertQqDeliveryPendingOwner(env, {
      qqSubjectRef: 'ph-qq-subject-v1_retry_cleanup',
      issueRef: 'qq-issue-retry-cleanup',
      managedCredentialRef: 'hash_qq_retry_cleanup',
    });
    await createManagedKeyDelivery(env.BROKER_DB, {
      issueSource: 'qq',
      subjectRef: 'ph-qq-subject-v1_retry_cleanup',
      managedCredentialRef: 'hash_qq_retry_cleanup',
      createdAt: new Date('2026-07-05T00:00:00.000Z'),
      expiresAt: new Date('2026-07-05T00:15:00.000Z'),
    });
    vi.mocked(cleanupManagedChildKey).mockResolvedValue({
      ok: false,
      reason: cleanupFailureReason(),
    });
    const originalBatch = env.BROKER_DB.batch.bind(env.BROKER_DB);
    let rejectFirstBatch = true;
    env.BROKER_DB.batch = async (statements) => {
      if (rejectFirstBatch) {
        rejectFirstBatch = false;
        throw new Error('synthetic cleanup transition failure');
      }
      return originalBatch(statements);
    };

    await expect(
      reconcileStaleManagedKeyDeliveries(
        env,
        new Date('2026-07-05T00:16:00.000Z'),
      ),
    ).rejects.toThrow('synthetic cleanup transition failure');
    expect(selectQqEntitlement(env, 'hash_qq_retry_cleanup')).toMatchObject({
      status: 'delivery_pending',
    });
    expect(
      env.__db
        .prepare("SELECT status, failure_reason FROM managed_key_deliveries WHERE managed_credential_ref = 'hash_qq_retry_cleanup'")
        .get(),
    ).toEqual({
      status: 'expired',
      failure_reason: 'stale_delivery_cleanup_claimed',
    });
    expect(fetchMock).toHaveBeenCalledOnce();
    expect(String(fetchMock.mock.calls[0]?.[1]?.body)).toContain(
      'cleanup_required state could not be confirmed',
    );

    await expect(
      reconcileStaleManagedKeyDeliveries(
        env,
        new Date('2026-07-05T00:31:00.000Z'),
      ),
    ).resolves.toEqual({ expired: 0, cleanupRequired: 0 });
    expect(cleanupManagedChildKey).toHaveBeenCalledTimes(1);
    expect(fetchMock).toHaveBeenCalledOnce();

    await expect(
      reconcileStaleManagedKeyDeliveries(
        env,
        new Date('2026-07-05T00:33:00.000Z'),
      ),
    ).resolves.toEqual({ expired: 0, cleanupRequired: 1 });
    expect(cleanupManagedChildKey).toHaveBeenCalledTimes(2);
    expect(selectQqEntitlement(env, 'hash_qq_retry_cleanup')).toMatchObject({
      status: 'cleanup_required',
    });
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});

async function postAck(env: ReturnType<typeof createTestBrokerEnv>, payload: unknown): Promise<Response> {
  return app.request(
    'http://broker.test/v1/providers/openrouter/managed-key-delivery/ack',
    {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(payload),
    },
    env,
  );
}

function insertDiscordDeliveryPendingOwner(
  env: ReturnType<typeof createTestBrokerEnv>,
  input: { installationId: string; discordUserRef: string; managedCredentialRef: string },
): void {
  env.__db
    .prepare('INSERT INTO installations (installation_id, device_public_key, app_version) VALUES (?, ?, ?)')
    .run(input.installationId, `${input.installationId}-device-key`, '1.0.0');
  env.__db
    .prepare(
      `INSERT INTO discord_identities (
          discord_user_ref, entitlement_installation_id, status, ref_secret_version, created_at, updated_at
        ) VALUES (?, ?, 'issuing', 1, ?, ?)`,
    )
    .run(
      input.discordUserRef,
      input.installationId,
      '2026-07-05T00:00:00.000Z',
      '2026-07-05T00:00:00.000Z',
    );
  env.__db
    .prepare(
      `INSERT INTO openrouter_entitlements (
          installation_id, status, budget_usd, managed_credential_ref, issued_at, expires_at,
          discord_user_ref, discord_issue_status, discord_issue_reserved_at, discord_issue_delivered_at
        ) VALUES (?, 'pending_release', 0.5, ?, ?, ?, ?, 'delivery_pending', ?, NULL)`,
    )
    .run(
      input.installationId,
      input.managedCredentialRef,
      '2026-07-05T00:00:00.000Z',
      '2026-08-05T00:00:00.000Z',
      input.discordUserRef,
      '2026-07-05T00:00:00.000Z',
    );
}

function insertQqDeliveryPendingOwner(
  env: ReturnType<typeof createTestBrokerEnv>,
  input: { qqSubjectRef: string; issueRef: string; managedCredentialRef: string },
): void {
  env.__db
    .prepare(
      `INSERT INTO qq_managed_entitlements (
          qq_subject_ref, status, issue_ref, managed_credential_ref, budget_usd,
          reserved_at, issued_at, expires_at, delivered_at, created_at, updated_at
        ) VALUES (?, 'delivery_pending', ?, ?, 0.5, ?, ?, ?, NULL, ?, ?)`,
    )
    .run(
      input.qqSubjectRef,
      input.issueRef,
      input.managedCredentialRef,
      '2026-07-05T00:00:00.000Z',
      '2026-07-05T00:00:00.000Z',
      '2026-08-05T00:00:00.000Z',
      '2026-07-05T00:00:00.000Z',
      '2026-07-05T00:00:00.000Z',
    );
}

function selectDiscordEntitlement(
  env: ReturnType<typeof createTestBrokerEnv>,
  managedCredentialRef: string,
): Record<string, unknown> | null {
  return (env.__db
    .prepare(
      `SELECT status, discord_issue_status, discord_issue_delivered_at
         FROM openrouter_entitlements
        WHERE managed_credential_ref = ?`,
    )
    .get(managedCredentialRef) as Record<string, unknown> | undefined) ?? null;
}

function selectQqEntitlement(
  env: ReturnType<typeof createTestBrokerEnv>,
  managedCredentialRef: string,
): Record<string, unknown> | null {
  return (env.__db
    .prepare(
      `SELECT status, delivered_at
         FROM qq_managed_entitlements
        WHERE managed_credential_ref = ?`,
    )
    .get(managedCredentialRef) as Record<string, unknown> | undefined) ?? null;
}

function selectDiscordIdentityStatus(
  env: ReturnType<typeof createTestBrokerEnv>,
  discordUserRef: string,
  installationId: string,
): string | null {
  const row = env.__db
    .prepare(
      `SELECT status
         FROM discord_identities
        WHERE discord_user_ref = ?
          AND entitlement_installation_id = ?`,
    )
    .get(discordUserRef, installationId) as { status: string } | undefined;
  return row?.status ?? null;
}

function selectScalar(env: ReturnType<typeof createTestBrokerEnv>, sql: string): number {
  const row = env.__db.prepare(sql).get() as Record<string, number>;
  return Number(Object.values(row)[0] ?? 0);
}

type CleanupFailureReason = Extract<
  Awaited<ReturnType<typeof cleanupManagedChildKey>>,
  { ok: false }
>['reason'];

function cleanupFailureReason(): CleanupFailureReason {
  const failure = {
    operation: 'delete_key' as const,
    code: 'upstream_http_error' as const,
    status: 500,
    upstreamCode: null,
    message: 'cleanup failed',
  };
  return {
    disable: { ok: true },
    delete: { ok: false, error: failure },
  };
}
