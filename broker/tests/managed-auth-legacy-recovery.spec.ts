import { afterEach, describe, expect, it, vi } from 'vitest';

import worker from '../src/index';
import {
  buildManagedOperationId, buildManagedOperationResumeToken, createManagedOperation,
  hashManagedOperationResumeToken, providerKeyNameForOperationAttempt, sweepStaleManagedOperations,
} from '../src/managed-operation';
import { readBrokerMigrationSql } from './test-support/migrations';
import { STATUS_PATH, controlledProvider, workerPost } from './test-support/managed-recovery';
import { createTestBrokerEnv, type TestBrokerEnv } from './test-support/sqlite-d1';

const now = new Date('2026-10-17T10:00:00.000Z');
afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals(); vi.restoreAllMocks(); });

async function seedOriginalFailure(env: TestBrokerEnv, provider: ReturnType<typeof controlledProvider>, source: 'discord' | 'qq') {
  const operationId = buildManagedOperationId(); const resumeToken = buildManagedOperationResumeToken();
  const installationId = `legacy-${operationId}`.toLowerCase(); const subject = `ph-${source === 'qq' ? 'qq-subject' : 'discord-user'}-v1_${operationId.slice(-24)}`;
  await createManagedOperation(env.BROKER_DB, { operationId, resumeTokenHash: await hashManagedOperationResumeToken(resumeToken), issueSource: source, subjectRef: subject, installationId, devicePublicKey: source === 'discord' ? `device-${operationId}` : null, now });
  env.__db.prepare(`UPDATE managed_operations SET state='ACTIVE', attempt_count=3, current_attempt_index=3 WHERE operation_id=?`).run(operationId);
  const currentName = providerKeyNameForOperationAttempt(operationId, source, 3);
  const oldName = providerKeyNameForOperationAttempt(operationId, source, 2);
  const limit = source === 'discord' ? 0.11 : 0.09;
  const currentResponse = await provider.fetchImpl('https://openrouter.ai/api/v1/keys', { method: 'POST', body: JSON.stringify({ name: currentName, limit }) });
  const current = await currentResponse.json() as { key: string; data: { hash: string } };
  const currentHash = current.data.hash;
  for (const [index, hash, outcome] of [[1, `genuinely-clean-${operationId}`, 'cleaned'], [2, null, 'cleaned'], [3, currentHash, 'created']] as const) {
    env.__db.prepare(`INSERT INTO managed_operation_attempts (operation_id,attempt_index,provider_key_name,managed_credential_ref,outcome,created_at,updated_at) VALUES (?,?,?,?,?,?,?)`)
      .run(operationId, index, providerKeyNameForOperationAttempt(operationId, source, index), hash, outcome, now.toISOString(), now.toISOString());
  }
  const deliveryId = `legacy-ack-${operationId}`;
  const ackToken = (await hashManagedOperationResumeToken(operationId)).replace('ph-mop-resume-v1_', 'ph-delivery-ack-token-v1_');
  env.__db.prepare(`INSERT INTO managed_key_deliveries (delivery_id,issue_source,subject_ref,installation_id,managed_credential_ref,ack_token_hash,status,created_at,expires_at,acknowledged_at,operation_id,attempt_index) VALUES (?,?,?,?,?,?,'acknowledged',?,?,?,?,3)`)
    .run(deliveryId, source, subject, installationId, currentHash, ackToken, now.toISOString(), '2026-10-17T10:15:00.000Z', now.toISOString(), operationId);
  if (source === 'discord') {
    env.__db.prepare(`INSERT INTO installations (installation_id,device_public_key,app_version,created_at,last_seen_at) VALUES (?,?,'1.0.0',?,?)`).run(installationId, `device-${operationId}`, now.toISOString(), now.toISOString());
    env.__db.prepare(`INSERT INTO discord_identities (discord_user_ref,entitlement_installation_id,status,created_at,updated_at) VALUES (?,?,'active',?,?)`).run(subject, installationId, now.toISOString(), now.toISOString());
    env.__db.prepare(`INSERT INTO openrouter_entitlements (installation_id,status,budget_usd,managed_credential_ref,issued_at,expires_at,discord_user_ref,discord_issue_status,discord_issue_delivered_at) VALUES (?,'active',?,?,?,'2027-01-17T10:00:00.000Z',?,'active',?)`)
      .run(installationId, limit, currentHash, now.toISOString(), subject, now.toISOString());
  } else {
    env.__db.prepare(`INSERT INTO qq_managed_entitlements (qq_subject_ref,status,issue_ref,managed_credential_ref,budget_usd,reserved_at,issued_at,expires_at,delivered_at,created_at,updated_at) VALUES (?,'active',?,?,?, ?,?,'2027-01-17T10:00:00.000Z',?,?,?)`)
      .run(subject, `issue-${operationId}`, currentHash, limit, ...Array(5).fill(now.toISOString()));
  }
  return { operationId, currentName, oldName, currentHash, limit, subject, installationId, authority: { operation_id: operationId, resume_token: resumeToken, installation_id: installationId } };
}

function migrate(env: TestBrokerEnv) {
  env.__db.exec('BEGIN'); env.__db.exec(readBrokerMigrationSql('0026_managed_attempt_cleanup_ownership.sql')); env.__db.exec('COMMIT');
  env.__db.exec('BEGIN'); env.__db.exec(readBrokerMigrationSql('0027_remove_managed_referrals.sql')); env.__db.exec('COMMIT');
}

function retained(env: TestBrokerEnv, seeded: Awaited<ReturnType<typeof seedOriginalFailure>>) {
  return [
    env.__db.prepare('SELECT * FROM managed_operations WHERE operation_id=?').get(seeded.operationId),
    env.__db.prepare('SELECT * FROM managed_operation_attempts WHERE operation_id=? AND attempt_index=3').get(seeded.operationId),
    env.__db.prepare('SELECT * FROM managed_key_deliveries WHERE operation_id=? AND attempt_index=3').get(seeded.operationId),
    env.__db.prepare('SELECT * FROM openrouter_entitlements ORDER BY installation_id').all(),
    env.__db.prepare('SELECT * FROM qq_managed_entitlements ORDER BY qq_subject_ref').all(),
    env.__db.prepare('SELECT * FROM discord_identities ORDER BY discord_user_ref').all(),
  ];
}

async function createOld(provider: ReturnType<typeof controlledProvider>, seeded: Awaited<ReturnType<typeof seedOriginalFailure>>) {
  const response = await provider.fetchImpl('https://openrouter.ai/api/v1/keys', { method: 'POST', body: JSON.stringify({ name: seeded.oldName, limit: 0.07 }) });
  return (await response.json() as { data: { hash: string } }).data.hash;
}

async function sweepAt(env: TestBrokerEnv, minutes: number) {
  const time = new Date(now.getTime() + minutes * 60_000); vi.setSystemTime(time);
  await sweepStaleManagedOperations(env, time);
}

describe.each(['discord', 'qq'] as const)('%s original persisted late-create failure', (source) => {
  it('migrates cleaned/null old debt and cleans its real resource without changing newer ACK/lifetime/allowance', async () => {
    vi.useFakeTimers(); vi.setSystemTime(now);
    const env = createTestBrokerEnv({ throughMigration: '0025_drop_legacy_translation_telemetry.sql' });
    const provider = controlledProvider(); vi.stubGlobal('fetch', provider.fetchImpl);
    const seeded = await seedOriginalFailure(env, provider, source); const oldHash = await createOld(provider, seeded);
    migrate(env);
    expect(env.__db.prepare('SELECT legacy_cleanup_state FROM managed_operation_attempts WHERE operation_id=? ORDER BY attempt_index').all(seeded.operationId))
      .toEqual([{ legacy_cleanup_state: 'resolved' }, { legacy_cleanup_state: 'pending' }, { legacy_cleanup_state: 'resolved' }]);
    const before = retained(env, seeded); const calls = provider.calls.length;
    vi.setSystemTime(new Date(now.getTime() + 6 * 60_000));
    await worker.scheduled({ scheduledTime: Date.now() }, env, {});
    expect(provider.keys.has(oldHash)).toBe(false); expect(provider.keys.get(seeded.currentHash)).toMatchObject({ limit: seeded.limit, disabled: false });
    expect(retained(env, seeded)).toEqual(before);
    expect(provider.calls.slice(calls).filter(call => ['PATCH', 'DELETE'].includes(call.method)).every(call => call.url.endsWith(`/${oldHash}`))).toBe(true);
    expect(env.__db.prepare('SELECT managed_credential_ref,legacy_cleanup_state FROM managed_operation_attempts WHERE operation_id=? AND attempt_index=2').get(seeded.operationId))
      .toEqual({ managed_credential_ref: oldHash, legacy_cleanup_state: 'resolved' });
    const status = await workerPost(env, STATUS_PATH, seeded.authority); expect(await status.json()).toMatchObject({ state: 'ACTIVE', current_attempt_index: 3 });
    const callsAfter = provider.calls.filter(call => new URL(call.url).hostname === 'openrouter.ai').length;
    await sweepAt(env, 62);
    expect(provider.calls.filter(call => new URL(call.url).hostname === 'openrouter.ai')).toHaveLength(callsAfter);
    expect((await provider.fetchImpl(`https://openrouter.ai/api/v1/keys/${seeded.currentHash}`)).status).toBe(200);
    expect(retained(env, seeded)).toEqual(before);
  });

  it('keeps complete empty inventory UNKNOWN until a late old resource appears after auth expiry', async () => {
    vi.useFakeTimers(); vi.setSystemTime(now);
    const env = createTestBrokerEnv({ throughMigration: '0025_drop_legacy_translation_telemetry.sql' });
    const provider = controlledProvider(); vi.stubGlobal('fetch', provider.fetchImpl);
    const seeded = await seedOriginalFailure(env, provider, source); migrate(env); const before = retained(env, seeded);
    for (const minutes of [6, 62]) {
      await sweepAt(env, minutes);
      expect(env.__db.prepare('SELECT legacy_cleanup_state,managed_credential_ref FROM managed_operation_attempts WHERE operation_id=? AND attempt_index=2').get(seeded.operationId))
        .toEqual({ legacy_cleanup_state: 'pending', managed_credential_ref: null });
      expect(retained(env, seeded)).toEqual(before);
    }
    const oldHash = await createOld(provider, seeded);
    await sweepAt(env, 63);
    expect(provider.keys.has(oldHash)).toBe(false); expect(provider.keys.has(seeded.currentHash)).toBe(true);
    expect(retained(env, seeded)).toEqual(before);
  });

  it('classifies old-Worker inserts made after migration rather than default-certifying them', async () => {
    vi.useFakeTimers(); vi.setSystemTime(now);
    const env = createTestBrokerEnv({ throughMigration: '0025_drop_legacy_translation_telemetry.sql' });
    const provider = controlledProvider(); vi.stubGlobal('fetch', provider.fetchImpl);
    const seeded = await seedOriginalFailure(env, provider, source);
    env.__db.prepare('DELETE FROM managed_operation_attempts WHERE operation_id=? AND attempt_index=2').run(seeded.operationId);
    migrate(env); const before = retained(env, seeded);
    env.__db.prepare(`INSERT INTO managed_operation_attempts (operation_id,attempt_index,provider_key_name,managed_credential_ref,outcome,created_at,updated_at) VALUES (?,2,?,NULL,'cleaned',?,?)`)
      .run(seeded.operationId, seeded.oldName, now.toISOString(), now.toISOString());
    expect(env.__db.prepare('SELECT legacy_cleanup_state FROM managed_operation_attempts WHERE operation_id=? AND attempt_index=2').get(seeded.operationId)).toEqual({ legacy_cleanup_state: 'unclassified' });
    const oldHash = await createOld(provider, seeded); await sweepAt(env, 6);
    expect(provider.keys.has(oldHash)).toBe(false); expect(provider.keys.has(seeded.currentHash)).toBe(true);
    expect(retained(env, seeded)).toEqual(before);
  });

  it('retries failed or malformed legacy cleanup while another claim cannot delete the current resource', async () => {
    vi.useFakeTimers(); vi.setSystemTime(now);
    const env = createTestBrokerEnv({ throughMigration: '0025_drop_legacy_translation_telemetry.sql' });
    const provider = controlledProvider(); vi.stubGlobal('fetch', provider.fetchImpl);
    const seeded = await seedOriginalFailure(env, provider, source); const oldHash = await createOld(provider, seeded); migrate(env); const before = retained(env, seeded);
    provider.malformedList = true; await sweepAt(env, 6);
    expect(provider.keys.has(oldHash)).toBe(true);
    provider.malformedList = false; provider.failCleanup = true; await sweepAt(env, 62);
    expect(provider.keys.has(oldHash)).toBe(true);
    provider.failCleanup = false; const gate = provider.gateCleanup();
    const cleaning = sweepAt(env, 63); await gate.entered;
    const callsBefore = provider.calls.length;
    await sweepStaleManagedOperations(env, new Date(now.getTime() + 63 * 60_000));
    expect(provider.calls).toHaveLength(callsBefore);
    gate.release(); await cleaning;
    expect(provider.keys.has(oldHash)).toBe(false); expect(provider.keys.get(seeded.currentHash)).toMatchObject({ disabled: false, limit: seeded.limit });
    expect(retained(env, seeded)).toEqual(before);
  });

  it('refuses a matching name that resolves to the newer ACKed hash, then cleans a separately owned orphan', async () => {
    vi.useFakeTimers(); vi.setSystemTime(now);
    const env = createTestBrokerEnv({ throughMigration: '0025_drop_legacy_translation_telemetry.sql' });
    const provider = controlledProvider(); vi.stubGlobal('fetch', provider.fetchImpl);
    const seeded = await seedOriginalFailure(env, provider, source); migrate(env); const before = retained(env, seeded);
    provider.keys.get(seeded.currentHash)!.name = seeded.oldName;
    const calls = provider.calls.length; await sweepAt(env, 6);
    expect(provider.calls.slice(calls).filter(call => ['PATCH', 'DELETE'].includes(call.method))).toEqual([]);
    expect(provider.keys.get(seeded.currentHash)).toMatchObject({ disabled: false, limit: seeded.limit });
    expect(env.__db.prepare('SELECT legacy_cleanup_state FROM managed_operation_attempts WHERE operation_id=? AND attempt_index=2').get(seeded.operationId)).toEqual({ legacy_cleanup_state: 'pending' });
    provider.keys.get(seeded.currentHash)!.name = seeded.currentName;
    const oldHash = await createOld(provider, seeded); await sweepAt(env, 62);
    expect(provider.keys.has(oldHash)).toBe(false); expect(provider.keys.has(seeded.currentHash)).toBe(true); expect(retained(env, seeded)).toEqual(before);
  });

  it('does not invent debt for known-hash genuinely cleaned old attempts or change ACTIVE authority', async () => {
    vi.useFakeTimers(); vi.setSystemTime(now);
    const env = createTestBrokerEnv({ throughMigration: '0025_drop_legacy_translation_telemetry.sql' });
    const provider = controlledProvider(); vi.stubGlobal('fetch', provider.fetchImpl);
    const seeded = await seedOriginalFailure(env, provider, source);
    env.__db.prepare('DELETE FROM managed_operation_attempts WHERE operation_id=? AND attempt_index=2').run(seeded.operationId);
    migrate(env); const before = retained(env, seeded); const calls = provider.calls.length;
    await sweepAt(env, 62);
    expect(provider.calls).toHaveLength(calls); expect(retained(env, seeded)).toEqual(before);
  });

  it('retains debt when an old Worker cleans a classified dispatch and ACKs a later attempt during rollout', async () => {
    vi.useFakeTimers(); vi.setSystemTime(now);
    const env = createTestBrokerEnv({ throughMigration: '0025_drop_legacy_translation_telemetry.sql' });
    const provider = controlledProvider(); vi.stubGlobal('fetch', provider.fetchImpl);
    const seeded = await seedOriginalFailure(env, provider, source);
    const attempt = env.__db.prepare('SELECT * FROM managed_operation_attempts WHERE operation_id=? AND attempt_index=3').get(seeded.operationId)!;
    const delivery = env.__db.prepare('SELECT * FROM managed_key_deliveries WHERE operation_id=?').get(seeded.operationId)!;
    const table = source === 'discord' ? 'openrouter_entitlements' : 'qq_managed_entitlements';
    const entitlement = env.__db.prepare(`SELECT * FROM ${table}`).get()!;
    const identity = source === 'discord' ? env.__db.prepare('SELECT * FROM discord_identities').get()! : null;
    env.__db.prepare('DELETE FROM managed_key_deliveries WHERE operation_id=?').run(seeded.operationId);
    env.__db.prepare('DELETE FROM managed_operation_attempts WHERE operation_id=? AND attempt_index=3').run(seeded.operationId);
    env.__db.exec(`DELETE FROM ${table}`);
    if (identity) env.__db.exec('DELETE FROM discord_identities');
    env.__db.prepare(`UPDATE managed_operations SET state='CREATING',attempt_count=2,current_attempt_index=2 WHERE operation_id=?`).run(seeded.operationId);
    env.__db.prepare(`UPDATE managed_operation_attempts SET outcome='unknown' WHERE operation_id=? AND attempt_index=2`).run(seeded.operationId);
    migrate(env); await sweepAt(env, 6);
    expect(env.__db.prepare('SELECT create_in_flight,legacy_cleanup_state FROM managed_operation_attempts WHERE operation_id=? AND attempt_index=2').get(seeded.operationId))
      .toEqual({ create_in_flight: 1, legacy_cleanup_state: 'resolved' });
    env.__db.prepare(`UPDATE managed_operation_attempts SET outcome='cleaned' WHERE operation_id=? AND attempt_index=2`).run(seeded.operationId);
    env.__db.prepare(`UPDATE managed_operations SET state='ACTIVE',attempt_count=3,current_attempt_index=3 WHERE operation_id=?`).run(seeded.operationId);
    const rows: Array<[string, Record<string, unknown>]> = [['managed_operation_attempts', attempt], ['managed_key_deliveries', delivery]];
    if (identity) rows.push(['discord_identities', identity]);
    rows.push([table, entitlement]);
    for (const [name, row] of rows) env.__db.prepare(`INSERT INTO ${name} (${Object.keys(row).join(',')}) VALUES (${Object.keys(row).map(() => '?').join(',')})`).run(...Object.values(row) as Array<string | number | null>);
    const before = retained(env, seeded); const oldHash = await createOld(provider, seeded);
    (before[1] as Record<string, unknown>).legacy_cleanup_state = 'resolved';
    await sweepAt(env, 7);
    expect(provider.keys.has(oldHash)).toBe(false); expect(provider.keys.has(seeded.currentHash)).toBe(true);
    expect(retained(env, seeded)).toEqual(before);
  });
});

it('rotates a bounded legacy-debt cohort so absent old resources cannot starve a later live orphan', async () => {
  vi.useFakeTimers(); vi.setSystemTime(now);
  const env = createTestBrokerEnv({ throughMigration: '0025_drop_legacy_translation_telemetry.sql' });
  const provider = controlledProvider(); vi.stubGlobal('fetch', provider.fetchImpl);
  const cohort = [];
  for (let index = 0; index < 26; index++) cohort.push(await seedOriginalFailure(env, provider, 'qq'));
  const last = cohort[25]!; const oldHash = await createOld(provider, last); migrate(env);
  const before = cohort.map(item => retained(env, item)); const calls = provider.calls.length;
  await sweepAt(env, 6);
  expect(provider.calls.slice(calls).filter(call => call.method === 'GET' && new URL(call.url).pathname === '/api/v1/keys')).toHaveLength(25);
  expect(provider.keys.has(oldHash)).toBe(true);
  await sweepAt(env, 7);
  expect(provider.keys.has(oldHash)).toBe(false);
  expect(cohort.map(item => retained(env, item))).toEqual(before);
  expect(cohort.every(item => provider.keys.has(item.currentHash))).toBe(true);
});
