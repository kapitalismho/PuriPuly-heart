import { DatabaseSync } from 'node:sqlite';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { applyBrokerMigrations, readBrokerMigrationSql } from './test-support/migrations';
import { createTestBrokerEnv } from './test-support/sqlite-d1';
import { ACK_PATH, RESUME_PATH, controlledProvider, seedRecoverableOperation, workerPost } from './test-support/managed-recovery';
import worker from '../src/index';
import { getManagedOperation, sweepStaleManagedOperations } from '../src/managed-operation';

const removal = '0027_remove_managed_referrals.sql';
const now = new Date('2026-10-17T10:00:00.000Z');

afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals(); });

describe('referral retirement without allowance or credential mutation', () => {
  it('preserves earned budgets, identities, exact operations, attempts, deliveries, cleanup ownership and tuned policy', () => {
    const db = new DatabaseSync(':memory:');
    db.exec('PRAGMA foreign_keys = ON');
    applyBrokerMigrations(db, { through: '0026_managed_attempt_cleanup_ownership.sql' });
    db.prepare(`INSERT INTO installations (installation_id, device_public_key, app_version, created_at, last_seen_at) VALUES ('earned-install', 'earned-device', '1.0.0', ?, ?)`).run(now.toISOString(), now.toISOString());
    db.prepare(`INSERT INTO discord_identities (discord_user_ref, entitlement_installation_id, status, created_at, updated_at) VALUES ('ph-discord-user-v1_earned', 'earned-install', 'active', ?, ?)`).run(now.toISOString(), now.toISOString());
    db.prepare(`INSERT INTO openrouter_entitlements (installation_id, status, budget_usd, managed_credential_ref, issued_at, expires_at, discord_user_ref, discord_issue_status, discord_issue_delivered_at) VALUES ('earned-install', 'active', 0.11, 'earned-discord-key', ?, '2027-01-17T10:00:00.000Z', 'ph-discord-user-v1_earned', 'active', ?)`).run(now.toISOString(), now.toISOString());
    db.prepare(`INSERT INTO qq_managed_entitlements (qq_subject_ref, status, issue_ref, managed_credential_ref, budget_usd, reserved_at, issued_at, expires_at, delivered_at, created_at, updated_at) VALUES ('ph-qq-subject-v1_earned', 'active', 'earned-qq-issue', 'earned-qq-key', 0.09, ?, ?, '2027-01-17T10:00:00.000Z', ?, ?, ?)`).run(...Array(5).fill(now.toISOString()));
    for (const [index, state, status] of [[1, 'ACTIVE', 'acknowledged'], [2, 'DELIVERY_PENDING', 'pending'], [3, 'FAILED', 'cleanup_required']] as const) {
      const operationId = `ph-mop-v1_preserved_operation_${index}`;
      db.prepare(`INSERT INTO managed_operations (operation_id, issue_source, subject_ref, installation_id, device_public_key, state, attempt_count, current_attempt_index, resume_token_hash, auth_expires_at, failure_reason, client_action, referral_reward_id, referral_status, settlement_status, hardware_hash, hardware_hash_salt_version, app_version, created_at, updated_at, cleanup_attempts) VALUES (?, 'discord', 'ph-discord-user-v1_earned', 'earned-install', 'earned-device', ?, 1, 1, ?, '2026-10-17T11:00:00.000Z', ?, ?, ?, 'reserved', 'invitee_pending', 'earned-hardware', 7, '1.0.0', ?, ?, 8)`).run(operationId, state, `ph-mop-resume-v1_${String(index).repeat(64)}`, state === 'FAILED' ? 'cleanup_failed_terminal' : null, state === 'FAILED' ? 'action_required' : 'wait', index, now.toISOString(), now.toISOString());
      db.prepare(`INSERT INTO managed_operation_attempts (id, operation_id, attempt_index, provider_key_name, managed_credential_ref, outcome, created_at, updated_at, cleanup_token, cleanup_started_at, scan_offset, create_in_flight, legacy_cleanup_state, legacy_cleanup_checked_at) VALUES (?, ?, 1, ?, ?, 'created', ?, ?, ?, ?, ?, ?, ?, ?)`).run(index, operationId, `preserved-provider-${index}`, `preserved-key-${index}`, now.toISOString(), now.toISOString(), index === 3 ? 'owned-cleanup-token' : null, index === 3 ? now.toISOString() : null, index === 3 ? 2000 : 0, index === 3 ? 1 : 0, index === 3 ? 'pending' : 'resolved', index === 3 ? now.toISOString() : null);
      db.prepare(`INSERT INTO managed_key_deliveries (delivery_id, issue_source, subject_ref, installation_id, managed_credential_ref, ack_token_hash, status, created_at, expires_at, acknowledged_at, failed_at, failure_reason, operation_id, attempt_index) VALUES (?, 'discord', 'ph-discord-user-v1_earned', 'earned-install', ?, ?, ?, ?, '2026-10-17T10:15:00.000Z', ?, ?, ?, ?, 1)`).run(`preserved-delivery-${index}`, `preserved-key-${index}`, `ph-delivery-ack-token-v1_${String(index).repeat(64)}`, status, now.toISOString(), status === 'acknowledged' ? now.toISOString() : null, status === 'cleanup_required' ? now.toISOString() : null, status === 'cleanup_required' ? 'managed_child_key_cleanup_failed' : null, operationId);
    }
    db.prepare(`UPDATE sqlite_sequence SET seq = 500 WHERE name = 'managed_operation_attempts'`).run();
    db.prepare(`INSERT INTO referral_codes (referral_id, owner_source, owner_subject_ref, owner_installation_id, status, created_at, updated_at) VALUES ('ABCDEF', 'discord', 'ph-discord-user-v1_earned', 'earned-install', 'active', ?, ?)`).run(now.toISOString(), now.toISOString());
    db.prepare(`INSERT INTO referral_rewards (id, referral_id, referrer_source, referrer_subject_ref, referred_source, referred_subject_ref, referred_bonus_status, referrer_bonus_status, operation_id, created_at, updated_at) VALUES (1, 'ABCDEF', 'discord', 'ph-discord-user-v1_earned', 'qq', 'ph-qq-subject-v1_pending', 'reserved', 'pending', 'ph-mop-v1_preserved_operation_2', ?, ?)`).run(now.toISOString(), now.toISOString());
    db.prepare(`INSERT INTO managed_referral_settlement_jobs (source, referral_reward_id, operation_id, phase, attempt_count, next_attempt_at, created_at, updated_at) VALUES ('qq', 1, 'ph-mop-v1_preserved_operation_2', 'invitee_pending', 8, ?, ?, ?)`).run(...Array(3).fill(now.toISOString()));
    db.exec(`UPDATE broker_config SET value = json_set(value, '$.newActiveEntitlementsPerDay.maxCount', 37, '$.trialChallenge.maxRequests', 7) WHERE key = 'abuse_controls'`);
    const preservedTables = ['installations', 'discord_identities', 'openrouter_entitlements', 'qq_managed_entitlements', 'managed_operation_attempts', 'managed_key_deliveries'];
    const before = preservedTables.map((table) => db.prepare(`SELECT * FROM ${table}`).all());
    const operations = db.prepare('SELECT * FROM managed_operations ORDER BY operation_id').all().map((row) => {
      const { referral_reward_id, referral_status, settlement_status, ...retained } = row;
      return retained;
    });
    const triggers = db.prepare("SELECT name, sql FROM sqlite_master WHERE type = 'trigger' AND name LIKE 'managed_%' ORDER BY name").all();
    db.exec('BEGIN');
    try { db.exec(readBrokerMigrationSql(removal)); db.exec('COMMIT'); } catch (error) { db.exec('ROLLBACK'); throw error; }
    expect(preservedTables.map((table) => db.prepare(`SELECT * FROM ${table}`).all())).toEqual(before);
    expect(db.prepare('SELECT * FROM managed_operations ORDER BY operation_id').all()).toEqual(operations);
    expect(db.prepare("SELECT name, sql FROM sqlite_master WHERE type = 'trigger' AND name LIKE 'managed_%' ORDER BY name").all()).toEqual(triggers);
    expect(db.prepare("SELECT seq FROM sqlite_sequence WHERE name = 'managed_operation_attempts'").get()).toEqual({ seq: 500 });
    expect(db.prepare('PRAGMA foreign_key_check').all()).toEqual([]);
    expect(db.prepare("SELECT name FROM sqlite_master WHERE lower(name) LIKE '%referral%' OR lower(name) LIKE '%qq_pass%'").all()).toEqual([]);
    expect(db.prepare("SELECT name FROM pragma_table_info('managed_operations') WHERE name IN ('referral_reward_id', 'referral_status', 'settlement_status')").all()).toEqual([]);
    expect(db.prepare("SELECT name FROM pragma_table_info('discord_oauth_sessions') WHERE name = 'referral_id'").all()).toEqual([]);
    const controls = JSON.parse(String(db.prepare("SELECT value FROM broker_config WHERE key = 'abuse_controls'").get()?.value));
    expect(controls.newActiveEntitlementsPerDay.maxCount).toBe(37); expect(controls.trialChallenge.maxRequests).toBe(7);
    expect(controls).not.toHaveProperty('referralAttempts'); expect(controls).not.toHaveProperty('qqAuthStatusIp');
    expect(controls.retention).not.toHaveProperty('referralSkippedDays'); expect(controls.retention).not.toHaveProperty('referralFailedDays');
    expect(() => db.exec(`INSERT INTO broker_config (key, value) VALUES ('qq_talk_together_pass', '{}')`)).toThrow();
    expect(() => db.exec(`UPDATE managed_key_deliveries SET status = 'acknowledged', acknowledged_at = '${now.toISOString()}' WHERE delivery_id = 'preserved-delivery-3'`)).toThrow('managed delivery cleanup owns attempt');
    db.close();
  });

  it('fails before retirement if the historical privacy schema has not completed', () => {
    const db = new DatabaseSync(':memory:'); applyBrokerMigrations(db, { through: '0020_network_identity_hmac.sql' });
    const phaseBefore = db.prepare("SELECT value FROM broker_config WHERE key = 'network_identity_migration'").get();
    db.exec('BEGIN');
    expect(() => db.exec(readBrokerMigrationSql(removal))).toThrow(); db.exec('ROLLBACK');
    expect(db.prepare("SELECT value FROM broker_config WHERE key = 'network_identity_migration'").get()).toEqual(phaseBefore);
    expect(db.prepare("SELECT name FROM sqlite_master WHERE name = 'referral_rewards'").get()).toEqual({ name: 'referral_rewards' });
    db.close();
  });

  it.each([
    ['discord', '0026_managed_attempt_cleanup_ownership.sql'],
    ['qq', '0026_managed_attempt_cleanup_ownership.sql'],
    ['discord', undefined],
    ['qq', undefined],
  ] as const)('delivers and acknowledges %s on schema %s without feature work or allowance PATCH', async (source, throughMigration) => {
    vi.useFakeTimers(); vi.setSystemTime(now);
    const env = createTestBrokerEnv({ throughMigration }); const provider = controlledProvider(); vi.stubGlobal('fetch', provider.fetchImpl);
    if (throughMigration) env.__db.prepare(`INSERT INTO managed_referral_settlement_jobs (source, referral_reward_id, phase, attempt_count, next_attempt_at, created_at, updated_at) VALUES ('qq', 999, 'invitee_pending', 0, ?, ?, ?)`).run(...Array(3).fill(now.toISOString()));
    const queuedBefore = throughMigration ? env.__db.prepare('SELECT * FROM managed_referral_settlement_jobs').all() : null;
    const seeded = await seedRecoverableOperation(env, source, now);
    const issued = await workerPost(env, RESUME_PATH, seeded.authority); const body = await issued.json() as Record<string, string>;
    expect(issued.status).toBe(200); expect(body.openrouter_api_key).toBe('sk-controlled-1');
    expect(body).not.toHaveProperty('referral_id'); expect(body).not.toHaveProperty('talk_together_pass'); expect(body).not.toHaveProperty('referral');
    const callsAfterIssuance = provider.calls.filter((call) => new URL(call.url).hostname === 'openrouter.ai').length;
    const ackBody = { delivery_id: body.delivery_id, managed_credential_ref: body.managed_credential_ref, delivery_ack_token: body.delivery_ack_token };
    const ack = await workerPost(env, ACK_PATH, ackBody); expect(ack.status).toBe(200); expect(await ack.json()).toEqual({ ok: true, status: 'acknowledged' });
    expect((await workerPost(env, ACK_PATH, ackBody)).status).toBe(200);
    await sweepStaleManagedOperations(env, now, provider.fetchImpl);
    await worker.scheduled({ scheduledTime: now.getTime() }, env, {});
    expect(await getManagedOperation(env.BROKER_DB, seeded.operationId)).toMatchObject({ state: 'ACTIVE' });
    expect(provider.keys.get(body.managed_credential_ref!)).toMatchObject({ limit: 0.07, disabled: false });
    expect(provider.calls.filter((call) => new URL(call.url).hostname === 'openrouter.ai')).toHaveLength(callsAfterIssuance);
    expect(provider.calls.filter((call) => call.method === 'PATCH' && call.body && Object.hasOwn(call.body, 'limit'))).toEqual([]);
    expect((await workerPost(env, '/v1/auth/qq/status', {})).status).toBe(404);
    if (throughMigration) expect(env.__db.prepare('SELECT * FROM managed_referral_settlement_jobs').all()).toEqual(queuedBefore);
  });
});
