import { describe, expect, it, vi, afterEach } from 'vitest';

vi.mock('../src/openrouter-management', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../src/openrouter-management')>()),
  cleanupManagedChildKey: vi.fn(),
}));

import app from '../src/index';
import {
  acknowledgeManagedKeyDelivery,
  createManagedKeyDelivery,
  hashDeliveryAckToken,
  listStalePendingManagedKeyDeliveries,
} from '../src/managed-key-delivery';
import { cleanupManagedChildKey } from '../src/openrouter-management';
import { reconcileStaleManagedKeyDeliveries } from '../src/scheduled';
import { normalizedErrorEnvelope } from './test-support/errors';
import { BROKER_MIGRATION_FILENAMES } from './test-support/migrations';
import { createTestBrokerEnv } from './test-support/sqlite-d1';

describe('managed key delivery ACK foundation', () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.clearAllMocks();
  });

  it('orders migration 0012 after telemetry 0011 and creates delivery ACK schema', () => {
    const env = createTestBrokerEnv();

    expect(BROKER_MIGRATION_FILENAMES.at(-3)).toBe('0011_add_telemetry_active_days.sql');
    expect(BROKER_MIGRATION_FILENAMES.at(-2)).toBe('0012_add_managed_key_delivery_ack.sql');
    expect(BROKER_MIGRATION_FILENAMES.at(-1)).toBe(
      '0013_add_telemetry_subjects_and_daily_summary_v2.sql',
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

  it('reconciles a finalized Discord owner before stale ACK cleanup', async () => {
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

    expect(failed.status).toBe(500);
    expect(selectDiscordEntitlement(env, 'hash_discord_ledger_repair')).toMatchObject({
      status: 'active',
      discord_issue_status: 'active',
      discord_issue_delivered_at: '2026-07-05T00:01:00.000Z',
    });
    expect(
      env.__db
        .prepare('SELECT status FROM managed_key_deliveries WHERE delivery_id = ?')
        .get(delivery.deliveryId),
    ).toEqual({ status: 'pending' });
    expect(selectScalar(env, "SELECT COUNT(*) FROM broker_issue_success_events WHERE managed_credential_ref = 'hash_discord_ledger_repair'")).toBe(1);

    rejectAckLedgerUpdate = false;
    const result = await reconcileStaleManagedKeyDeliveries(
      env,
      new Date('2026-07-05T00:16:00.000Z'),
    );

    expect(result).toEqual({ expired: 0, cleanupRequired: 0 });
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

  it('reconciles a finalized QQ owner before stale ACK cleanup', async () => {
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

    expect(failed.status).toBe(500);
    expect(selectQqEntitlement(env, 'hash_qq_ledger_repair')).toMatchObject({
      status: 'active',
      delivered_at: '2026-07-05T00:01:00.000Z',
    });
    expect(
      env.__db
        .prepare('SELECT status FROM managed_key_deliveries WHERE delivery_id = ?')
        .get(delivery.deliveryId),
    ).toEqual({ status: 'pending' });
    expect(selectScalar(env, "SELECT COUNT(*) FROM broker_issue_success_events WHERE managed_credential_ref = 'hash_qq_ledger_repair'")).toBe(1);

    rejectAckLedgerUpdate = false;
    const result = await reconcileStaleManagedKeyDeliveries(
      env,
      new Date('2026-07-05T00:16:00.000Z'),
    );

    expect(result).toEqual({ expired: 0, cleanupRequired: 0 });
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
