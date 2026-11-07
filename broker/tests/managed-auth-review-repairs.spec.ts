import { afterEach, describe, expect, it, vi } from 'vitest';
import { buildManagedOperationId, buildManagedOperationResumeToken } from './test-support/managed-operation';

import {
  createManagedOperation,
  getManagedOperation, hashManagedOperationResumeToken, listManagedOperationAttempts,
  reconcileUnknownAttempt, sweepStaleManagedOperations,
} from '../src/managed-operation';
import { readBrokerMigrationSql } from './test-support/migrations';
import { ACK_PATH, RESUME_PATH, controlledProvider, seedRecoverableOperation, workerPost } from './test-support/managed-recovery';
import { createTestBrokerEnv, type TestBrokerEnv } from './test-support/sqlite-d1';

const now = new Date('2026-10-17T10:00:00.000Z');
const sixMinutesLater = new Date(now.getTime() + 6 * 60_000);

afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals(); vi.restoreAllMocks(); });

async function acknowledge(env: TestBrokerEnv, delivered: Record<string, string>) {
  const response = await workerPost(env, ACK_PATH, {
    delivery_id: delivered.delivery_id,
    managed_credential_ref: delivered.managed_credential_ref,
    delivery_ack_token: delivered.delivery_ack_token,
  });
  expect(response.status).toBe(200);
}

function seedDiscordReservation(
  env: TestBrokerEnv,
  seeded: Awaited<ReturnType<typeof seedRecoverableOperation>>,
  input: { status: 'pending_release' | 'revoked'; issueStatus: string; credentialRef: string | null },
) {
  env.__db.prepare(`INSERT INTO installations (installation_id, device_public_key, app_version, created_at, last_seen_at) VALUES (?, ?, '1.0.0', ?, ?)`)
    .run(seeded.installationId, seeded.operation.device_public_key, now.toISOString(), now.toISOString());
  env.__db.prepare(`INSERT INTO discord_identities (discord_user_ref, entitlement_installation_id, status, created_at, updated_at) VALUES (?, ?, 'issuing', ?, ?)`)
    .run(seeded.subjectRef, seeded.installationId, now.toISOString(), now.toISOString());
  env.__db.prepare(`INSERT INTO openrouter_entitlements (installation_id, status, budget_usd, managed_credential_ref, discord_user_ref, discord_issue_status, discord_issue_reserved_at) VALUES (?, ?, 0.07, ?, ?, ?, ?)`)
    .run(seeded.installationId, input.status, input.credentialRef, seeded.subjectRef, input.issueStatus, now.toISOString());
}

describe.each(['discord', 'qq'] as const)('%s outstanding create responsibility', (source) => {
  it.each([false, true])('retains a held real create before a later ACK, provider visibility=%s', async (visibleBeforeResponse) => {
    vi.useFakeTimers(); vi.setSystemTime(now);
    const env = createTestBrokerEnv(); const provider = controlledProvider(); vi.stubGlobal('fetch', provider.fetchImpl);
    const seeded = await seedRecoverableOperation(env, source, now);
    const gate = provider.gateCreate(visibleBeforeResponse);
    const delayed = workerPost(env, RESUME_PATH, seeded.authority);
    await gate.entered;
    expect((await listManagedOperationAttempts(env.BROKER_DB, seeded.operationId))[1]).toMatchObject({ create_in_flight: 1, outcome: 'unknown', managed_credential_ref: null });
    vi.setSystemTime(sixMinutesLater);
    await sweepStaleManagedOperations(env, sixMinutesLater);
    if (visibleBeforeResponse) {
      expect(provider.keys.has('controlled-key-1')).toBe(false);
      const issued = await workerPost(env, RESUME_PATH, seeded.authority);
      const delivered = await issued.json() as Record<string, string>;
      expect(delivered.openrouter_api_key).toBe('sk-controlled-2');
      await acknowledge(env, delivered);
      gate.release();
      expect(await (await delayed).json()).not.toHaveProperty('openrouter_api_key');
      await sweepStaleManagedOperations(env, sixMinutesLater);
      expect(provider.keys.size).toBe(1);
      expect(provider.keys.has(delivered.managed_credential_ref!)).toBe(true);
      expect(await getManagedOperation(env.BROKER_DB, seeded.operationId)).toMatchObject({ state: 'ACTIVE', current_attempt_index: 3 });
    } else {
      expect(provider.creates).toBe(0);
      expect((await listManagedOperationAttempts(env.BROKER_DB, seeded.operationId))[1]).toMatchObject({ create_in_flight: 1, outcome: 'unknown' });
      expect(await (await workerPost(env, RESUME_PATH, seeded.authority)).json()).not.toHaveProperty('openrouter_api_key');
      expect((await getManagedOperation(env.BROKER_DB, seeded.operationId))?.current_attempt_index).toBe(2);
      gate.release();
      expect(await (await delayed).json()).not.toHaveProperty('openrouter_api_key');
      expect(provider.keys.has('controlled-key-1')).toBe(true);
      expect((await listManagedOperationAttempts(env.BROKER_DB, seeded.operationId))[1]).toMatchObject({ create_in_flight: 0, managed_credential_ref: 'controlled-key-1' });
      await reconcileUnknownAttempt(env.BROKER_DB, env.OPENROUTER_MANAGEMENT_API_KEY, (await getManagedOperation(env.BROKER_DB, seeded.operationId))!, sixMinutesLater);
      expect(provider.keys.has('controlled-key-1')).toBe(false);
      const delivered = await (await workerPost(env, RESUME_PATH, seeded.authority)).json() as Record<string, string>;
      expect(delivered.openrouter_api_key).toBe('sk-controlled-2');
      await acknowledge(env, delivered);
      expect(provider.keys.size).toBe(1);
      expect(provider.keys.has(delivered.managed_credential_ref!)).toBe(true);
      expect(await getManagedOperation(env.BROKER_DB, seeded.operationId)).toMatchObject({ state: 'ACTIVE', current_attempt_index: 3 });
    }
    expect((await listManagedOperationAttempts(env.BROKER_DB, seeded.operationId))[1]).toMatchObject({ outcome: 'cleaned', create_in_flight: 0 });
    expect(provider.creates).toBe(2);
  });

  it.each(['network', 'http503'] as const)('does not equate local %s settlement with remote creation settlement', async (failure) => {
    vi.useFakeTimers(); vi.setSystemTime(now);
    const env = createTestBrokerEnv(); const provider = controlledProvider();
    let lateCreate: (() => Promise<Response>) | null = null;
    const fetchImpl: typeof fetch = async (input, init) => {
      if (!lateCreate && init?.method === 'POST' && new URL(String(input)).pathname === '/api/v1/keys') {
        lateCreate = () => provider.fetchImpl(input, init);
        if (failure === 'network') throw new Error('controlled transport ended before remote completion');
        return Response.json({ error: { message: 'controlled ambiguous upstream failure' } }, { status: 503 });
      }
      return provider.fetchImpl(input, init);
    };
    vi.stubGlobal('fetch', fetchImpl);
    const seeded = await seedRecoverableOperation(env, source, now);
    expect(await (await workerPost(env, RESUME_PATH, seeded.authority)).json()).not.toHaveProperty('openrouter_api_key');
    vi.setSystemTime(sixMinutesLater);
    await sweepStaleManagedOperations(env, sixMinutesLater);
    expect(await (await workerPost(env, RESUME_PATH, seeded.authority)).json()).not.toHaveProperty('openrouter_api_key');
    expect(provider.creates).toBe(0);
    expect((await listManagedOperationAttempts(env.BROKER_DB, seeded.operationId))[1]).toMatchObject({ create_in_flight: 1, outcome: 'unknown' });
    await (lateCreate as unknown as () => Promise<Response>)();
    expect(provider.keys.has('controlled-key-1')).toBe(true);
    await reconcileUnknownAttempt(env.BROKER_DB, env.OPENROUTER_MANAGEMENT_API_KEY, (await getManagedOperation(env.BROKER_DB, seeded.operationId))!, sixMinutesLater);
    expect(provider.keys.has('controlled-key-1')).toBe(false);
    const delivered = await (await workerPost(env, RESUME_PATH, seeded.authority)).json() as Record<string, string>;
    expect(delivered.openrouter_api_key).toBe('sk-controlled-2');
    await acknowledge(env, delivered);
    expect(provider.keys.size).toBe(1);
    expect(provider.keys.has(delivered.managed_credential_ref!)).toBe(true);
  });

  it('keeps lost dispatch responsibility UNKNOWN after authorization expiry without a locatable resource', async () => {
    vi.useFakeTimers(); vi.setSystemTime(now);
    const env = createTestBrokerEnv(); const provider = controlledProvider();
    vi.stubGlobal('fetch', (async (input, init) => {
      if (init?.method === 'POST' && new URL(String(input)).pathname === '/api/v1/keys') throw new Error('controlled lost response');
      return provider.fetchImpl(input, init);
    }) satisfies typeof fetch);
    const seeded = await seedRecoverableOperation(env, source, now);
    await workerPost(env, RESUME_PATH, seeded.authority);
    for (const minutes of [6, 62, 90]) {
      const later = new Date(now.getTime() + minutes * 60_000); vi.setSystemTime(later);
      await sweepStaleManagedOperations(env, later);
    }
    expect((await listManagedOperationAttempts(env.BROKER_DB, seeded.operationId))[1]).toMatchObject({ outcome: 'unknown', create_in_flight: 1 });
    expect(await getManagedOperation(env.BROKER_DB, seeded.operationId)).toMatchObject({ state: 'FAILED', failure_reason: 'authorization_expired' });
    expect(provider.creates).toBe(0);
  });
});

describe('Discord durable policy through reconciliation', () => {
  it.each(['issuing', 'delivery_pending', 'cleanup_required'])('retains pre-ACK revocation and identity while cleaning %s resources', async (issueStatus) => {
    vi.useFakeTimers(); vi.setSystemTime(now);
    const env = createTestBrokerEnv(); const provider = controlledProvider(); vi.stubGlobal('fetch', provider.fetchImpl);
    const seeded = await seedRecoverableOperation(env, 'discord', now);
    seedDiscordReservation(env, seeded, { status: 'revoked', issueStatus, credentialRef: seeded.hash });
    provider.keys.set(seeded.hash, { hash: seeded.hash, name: seeded.keyName, limit: 0.07, disabled: false });
    const before = env.__db.prepare('SELECT * FROM openrouter_entitlements WHERE installation_id = ?').get(seeded.installationId);
    const identity = env.__db.prepare('SELECT * FROM discord_identities WHERE discord_user_ref = ?').get(seeded.subjectRef);
    const response = await workerPost(env, RESUME_PATH, seeded.authority);
    const body = await response.json();
    expect(body).not.toHaveProperty('openrouter_api_key');
    expect(body).toMatchObject({ state: 'FAILED', client_action: 'action_required', failure_reason: 'terminal_provider_failure' });
    expect(provider.keys.has(seeded.hash)).toBe(false); expect(provider.creates).toBe(0);
    expect(env.__db.prepare('SELECT * FROM openrouter_entitlements WHERE installation_id = ?').get(seeded.installationId)).toEqual(before);
    expect(env.__db.prepare('SELECT * FROM discord_identities WHERE discord_user_ref = ?').get(seeded.subjectRef)).toEqual(identity);
    await workerPost(env, RESUME_PATH, seeded.authority);
    expect(provider.creates).toBe(0);
  });

  it.each(['none', 'discord', 'qq'] as const)('excludes only its own retained null-hash allocation from cap1 with other=%s', async (otherSource) => {
    vi.useFakeTimers(); vi.setSystemTime(now);
    const env = createTestBrokerEnv(); const provider = controlledProvider(); vi.stubGlobal('fetch', provider.fetchImpl);
    const seeded = await seedRecoverableOperation(env, 'discord', now);
    env.__db.prepare('UPDATE managed_operation_attempts SET managed_credential_ref = NULL WHERE operation_id = ?').run(seeded.operationId);
    seedDiscordReservation(env, seeded, { status: 'pending_release', issueStatus: 'cleanup_required', credentialRef: null });
    env.__db.prepare(`UPDATE broker_config SET value = json_set(value, '$.newActiveEntitlementsPerDay.maxCount', 1) WHERE key = 'abuse_controls'`).run();
    if (otherSource === 'discord') {
      env.__db.prepare(`INSERT INTO installations (installation_id, device_public_key, app_version, created_at, last_seen_at) VALUES ('other-cap-slot', 'other-cap-device', '1.0.0', ?, ?)`).run(now.toISOString(), now.toISOString());
      env.__db.prepare(`INSERT INTO openrouter_entitlements (installation_id, status, budget_usd, issued_at) VALUES ('other-cap-slot', 'active', 0.07, ?)`).run(now.toISOString());
    } else if (otherSource === 'qq') {
      env.__db.prepare(`INSERT INTO qq_managed_entitlements (qq_subject_ref, status, issue_ref, budget_usd, reserved_at, created_at, updated_at) VALUES ('ph-qq-subject-v1_other_cap', 'issuing', 'other-cap-issue', 0.07, ?, ?, ?)`).run(...Array(3).fill(now.toISOString()));
    }
    let response = await workerPost(env, RESUME_PATH, seeded.authority);
    if (otherSource !== 'none') {
      expect(response.status).toBe(503); expect(await response.json()).toMatchObject({ error: { subcode: 'global_cap_reached' } });
      expect(provider.creates).toBe(0);
      if (otherSource === 'discord') env.__db.prepare(`DELETE FROM openrouter_entitlements WHERE installation_id = 'other-cap-slot'`).run();
      else env.__db.prepare(`DELETE FROM qq_managed_entitlements WHERE qq_subject_ref = 'ph-qq-subject-v1_other_cap'`).run();
      response = await workerPost(env, RESUME_PATH, seeded.authority);
    }
    expect(response.status).toBe(200);
    const delivered = await response.json() as Record<string, string>;
    expect(delivered.openrouter_api_key).toBe('sk-controlled-1');
    await acknowledge(env, delivered);
    expect(await getManagedOperation(env.BROKER_DB, seeded.operationId)).toMatchObject({ state: 'ACTIVE' });
    expect(provider.creates).toBe(1); expect(provider.keys.size).toBe(1);
  });
});

it('carries historical ambiguous dispatch responsibility through0026/0027 without blocking untouched authorization', async () => {
  vi.useFakeTimers(); vi.setSystemTime(now);
  const env = createTestBrokerEnv({ throughMigration: '0025_drop_legacy_translation_telemetry.sql' });
  const provider = controlledProvider(); vi.stubGlobal('fetch', provider.fetchImpl);
  const authorities = [];
  for (const [index, state, hash] of [[1, 'CREATE_UNKNOWN', null], [2, 'CREATING', 'historical-known-key'], [3, 'AUTHENTICATED', null]] as const) {
    const operationId = buildManagedOperationId(); const resumeToken = buildManagedOperationResumeToken();
    await createManagedOperation(env.BROKER_DB, { operationId, resumeTokenHash: await hashManagedOperationResumeToken(resumeToken), issueSource: 'qq', subjectRef: `ph-qq-subject-v1_historical_${index}`, installationId: `historical-install-${index}`, devicePublicKey: null, now });
    if (index !== 3) {
      env.__db.prepare(`UPDATE managed_operations SET state = ?, attempt_count = 1, current_attempt_index = 1 WHERE operation_id = ?`).run(state, operationId);
      env.__db.prepare(`INSERT INTO managed_operation_attempts (operation_id, attempt_index, provider_key_name, managed_credential_ref, outcome, created_at, updated_at) VALUES (?, 1, ?, ?, ?, ?, ?)`)
        .run(operationId, `historical-key-${index}`, hash, hash ? 'created' : 'unknown', now.toISOString(), now.toISOString());
    }
    authorities.push({ operation_id: operationId, resume_token: resumeToken, installation_id: `historical-install-${index}` });
  }
  env.__db.exec(readBrokerMigrationSql('0026_managed_attempt_cleanup_ownership.sql'));
  expect(env.__db.prepare('SELECT create_in_flight FROM managed_operation_attempts ORDER BY id').all()).toEqual([{ create_in_flight: 1 }, { create_in_flight: 0 }]);
  env.__db.exec('BEGIN'); env.__db.exec(readBrokerMigrationSql('0027_remove_managed_referrals.sql')); env.__db.exec('COMMIT');
  expect(env.__db.prepare('SELECT create_in_flight FROM managed_operation_attempts ORDER BY id').all()).toEqual([{ create_in_flight: 1 }, { create_in_flight: 0 }]);
  expect(await (await workerPost(env, RESUME_PATH, authorities[0]!)).json()).not.toHaveProperty('openrouter_api_key');
  expect((await getManagedOperation(env.BROKER_DB, authorities[0]!.operation_id))?.current_attempt_index).toBe(1);
  const delivered = await (await workerPost(env, RESUME_PATH, authorities[2]!)).json() as Record<string, string>;
  expect(delivered.openrouter_api_key).toBe('sk-controlled-1');
  await acknowledge(env, delivered);
  expect(provider.creates).toBe(1);
});
