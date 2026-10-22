import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  expireManagedOperation, failManagedOperationTerminal, getManagedOperation, listManagedOperationAttempts,
  markAttemptUnknown, reconcileUnknownAttempt, recordAttemptCredential, releaseUnstartedOperationAttempt,
  startManagedOperationAttempt, sweepStaleManagedOperations, transitionOperationToPostCreateState,
} from '../src/managed-operation';
import { createManagedKeyDelivery } from '../src/managed-key-delivery';
import { reconcileStaleManagedKeyDeliveries } from '../src/scheduled';
import { createTestBrokerEnv, seedRequestEvent } from './test-support/sqlite-d1';
import { ACK_PATH, RESUME_PATH, STATUS_PATH, controlledProvider, deferred, seedRecoverableOperation, workerPost } from './test-support/managed-recovery';
import { MANAGED_CLEANUP_RETRY_MS } from '../src/managed-cleanup';

const now = new Date('2026-10-17T10:00:00.000Z');

afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals(); vi.restoreAllMocks(); });

describe.each(['discord', 'qq'] as const)('%s managed recovery safety', (source) => {
  it.each(['resume', 'cron'] as const)('does not let a delayed %s snapshot delete a later acknowledged key', async (consumer) => {
    vi.useFakeTimers(); vi.setSystemTime(now);
    const gate = deferred(); const entered = deferred();
    let armed = false; let reads = 0;
    const env = createTestBrokerEnv({
      afterFirst: async ({ sql }) => {
        if (armed && consumer === 'resume' && sql === 'SELECT * FROM managed_operations WHERE operation_id = ?' && ++reads === 3) {
          armed = false; entered.resolve(); await gate.promise;
        }
      },
      afterAll: async ({ sql }) => {
        if (armed && consumer === 'cron' && sql.includes('FROM managed_operation_attempts attempt')) {
          armed = false; entered.resolve(); await gate.promise;
        }
      },
    });
    const provider = controlledProvider(); vi.stubGlobal('fetch', provider.fetchImpl);
    const seeded = await seedRecoverableOperation(env, source, now);
    provider.keys.set(seeded.hash, { hash: seeded.hash, name: seeded.keyName, limit: 0.07, disabled: false });
    armed = true;
    const delayed = consumer === 'resume' ? workerPost(env, RESUME_PATH, seeded.authority) : sweepStaleManagedOperations(env, now);
    await entered.promise;
    const fresh = await workerPost(env, RESUME_PATH, seeded.authority);
    expect(fresh.status).toBe(200);
    const delivered = await fresh.json() as Record<string, string>;
    expect(delivered.openrouter_api_key).toBe('sk-controlled-1');
    const ack = await workerPost(env, ACK_PATH, { delivery_id: delivered.delivery_id, managed_credential_ref: delivered.managed_credential_ref, delivery_ack_token: delivered.delivery_ack_token });
    expect(ack.status).toBe(200);
    const callsBefore = provider.calls.length;
    gate.resolve(); await delayed;
    expect(provider.calls).toHaveLength(callsBefore);
    expect(provider.keys.has(seeded.hash)).toBe(false);
    expect(provider.keys.has(delivered.managed_credential_ref!)).toBe(true);
    expect(provider.creates).toBe(1);
    expect(await getManagedOperation(env.BROKER_DB, seeded.operationId)).toMatchObject({ state: 'ACTIVE', current_attempt_index: 2 });
    expect(await listManagedOperationAttempts(env.BROKER_DB, seeded.operationId)).toEqual([
      expect.objectContaining({ attempt_index: 1, outcome: 'cleaned' }),
      expect.objectContaining({ attempt_index: 2, outcome: 'created', managed_credential_ref: delivered.managed_credential_ref }),
    ]);
    expect(env.__db.prepare('SELECT status FROM managed_key_deliveries WHERE delivery_id = ?').get(delivered.delivery_id)).toEqual({ status: 'acknowledged' });
  });

  it('excludes ACK and fresh issuance while destructive cleanup owns the exact attempt', async () => {
    vi.useFakeTimers(); vi.setSystemTime(now);
    const env = createTestBrokerEnv(); const provider = controlledProvider(); vi.stubGlobal('fetch', provider.fetchImpl);
    const seeded = await seedRecoverableOperation(env, source, now);
    env.__db.prepare("UPDATE managed_operations SET state = 'CREATING' WHERE operation_id = ?").run(seeded.operationId);
    await recordAttemptCredential(env.BROKER_DB, seeded.operationId, 1, seeded.hash, now);
    const delivery = await createManagedKeyDelivery(env.BROKER_DB, { issueSource: source, subjectRef: seeded.subjectRef, installationId: seeded.installationId, managedCredentialRef: seeded.hash, createdAt: now, expiresAt: new Date(now.getTime() + 1), operationId: seeded.operationId, attemptIndex: 1 });
    env.__db.prepare("UPDATE managed_operations SET state = 'DELIVERY_PENDING' WHERE operation_id = ?").run(seeded.operationId);
    provider.keys.set(seeded.hash, { hash: seeded.hash, name: seeded.keyName, limit: 0.07, disabled: false });
    const cleanupGate = provider.gateCleanup();
    const cleaningAt = new Date(now.getTime() + 2); vi.setSystemTime(cleaningAt);
    const cleaning = reconcileUnknownAttempt(env.BROKER_DB, env.OPENROUTER_MANAGEMENT_API_KEY, { ...seeded.operation, state: 'DELIVERY_PENDING' }, cleaningAt);
    await cleanupGate.entered;
    expect(env.__db.prepare('SELECT cleanup_token FROM managed_operation_attempts').get()).toMatchObject({ cleanup_token: expect.any(String) });
    const ack = await workerPost(env, ACK_PATH, { delivery_id: delivery.deliveryId, managed_credential_ref: seeded.hash, delivery_ack_token: delivery.deliveryAckToken });
    expect(ack.status).not.toBe(200);
    const current = (await getManagedOperation(env.BROKER_DB, seeded.operationId))!;
    expect(await startManagedOperationAttempt(env.BROKER_DB, current, cleaningAt)).toEqual({ ok: false, reason: 'not_retry_ready' });
    cleanupGate.release(); await cleaning;
    expect(provider.keys.has(seeded.hash)).toBe(false);
    expect(await getManagedOperation(env.BROKER_DB, seeded.operationId)).toMatchObject({ state: 'RETRY_READY' });
    const resumed = await workerPost(env, RESUME_PATH, seeded.authority);
    expect(resumed.status).toBe(200);
    expect(await resumed.json()).toHaveProperty('openrouter_api_key', 'sk-controlled-1');
  });

  it.each(['authorization_expired', 'cleanup_failed_terminal'] as const)('cleans legacy FAILED resources independently of %s and repeated provider failures', async (reason) => {
    const env = createTestBrokerEnv(); const provider = controlledProvider();
    const seeded = await seedRecoverableOperation(env, source, now);
    provider.keys.set(seeded.hash, { hash: seeded.hash, name: seeded.keyName, limit: 0.07, disabled: false });
    env.__db.prepare("UPDATE managed_operations SET state = 'FAILED', failure_reason = ?, cleanup_attempts = 8, client_action = 'action_required' WHERE operation_id = ?").run(reason, seeded.operationId);
    provider.failCleanup = true;
    await sweepStaleManagedOperations(env, new Date(now.getTime() + 61 * 60_000), provider.fetchImpl);
    expect(provider.keys.has(seeded.hash)).toBe(true);
    expect((await listManagedOperationAttempts(env.BROKER_DB, seeded.operationId))[0]?.outcome).not.toBe('cleaned');
    provider.failCleanup = false;
    await sweepStaleManagedOperations(env, new Date(now.getTime() + 61 * 60_000 + MANAGED_CLEANUP_RETRY_MS), provider.fetchImpl);
    expect(provider.keys.has(seeded.hash)).toBe(false);
    expect((await listManagedOperationAttempts(env.BROKER_DB, seeded.operationId))[0]?.outcome).toBe('cleaned');
    expect(await getManagedOperation(env.BROKER_DB, seeded.operationId)).toMatchObject({ state: 'FAILED', failure_reason: reason, client_action: 'action_required' });
    expect(provider.creates).toBe(0);
  });

  it('durably defers unattended retries while authenticated same-operation recovery stays usable', async () => {
    vi.useFakeTimers(); vi.setSystemTime(now);
    const env = createTestBrokerEnv(); const provider = controlledProvider(); vi.stubGlobal('fetch', provider.fetchImpl);
    const seeded = await seedRecoverableOperation(env, source, now);
    env.__db.prepare(`UPDATE managed_operation_attempts SET managed_credential_ref = NULL, create_in_flight = 1 WHERE operation_id = ?`).run(seeded.operationId);
    await sweepStaleManagedOperations(env, now, provider.fetchImpl);
    const attempts = await listManagedOperationAttempts(env.BROKER_DB, seeded.operationId);
    const operation = await getManagedOperation(env.BROKER_DB, seeded.operationId);
    const changes = env.__db.prepare('SELECT total_changes() AS count').get()!.count;
    const calls = provider.calls.length;
    for (const minute of [1, 2, 3]) await sweepStaleManagedOperations(env, new Date(now.getTime() + minute * 60_000), provider.fetchImpl);
    expect(await listManagedOperationAttempts(env.BROKER_DB, seeded.operationId)).toEqual(attempts);
    expect(await getManagedOperation(env.BROKER_DB, seeded.operationId)).toEqual(operation);
    expect(env.__db.prepare('SELECT total_changes() AS count').get()!.count).toBe(changes);
    expect(provider.calls).toHaveLength(calls);
    expect(operation).toMatchObject({ state: 'CLEANUP_REQUIRED' });
    provider.keys.set(seeded.hash, { hash: seeded.hash, name: seeded.keyName, limit: 0.07, disabled: false });
    const delivered = await (await workerPost(env, RESUME_PATH, seeded.authority)).json() as Record<string, string>;
    expect(delivered.openrouter_api_key).toBe('sk-controlled-1');
    expect(provider.keys.has(seeded.hash)).toBe(false);
    expect((await workerPost(env, ACK_PATH, { delivery_id: delivered.delivery_id, managed_credential_ref: delivered.managed_credential_ref, delivery_ack_token: delivered.delivery_ack_token })).status).toBe(200);
    expect(await getManagedOperation(env.BROKER_DB, seeded.operationId)).toMatchObject({ state: 'ACTIVE', current_attempt_index: 2 });
    expect(provider.keys.has(delivered.managed_credential_ref!)).toBe(true);
  });

  it('reclaims an abandoned exact-attempt cleanup at sixteen minutes rather than waiting six hours', async () => {
    const env = createTestBrokerEnv(); const provider = controlledProvider();
    const seeded = await seedRecoverableOperation(env, source, now);
    provider.keys.set(seeded.hash, { hash: seeded.hash, name: seeded.keyName, limit: 0.07, disabled: false });
    env.__db.prepare(`UPDATE managed_operation_attempts SET cleanup_token = 'abandoned', cleanup_started_at = ?, legacy_cleanup_checked_at = ? WHERE operation_id = ?`).run(now.toISOString(), now.toISOString(), seeded.operationId);
    await sweepStaleManagedOperations(env, new Date(now.getTime() + 16 * 60_000 - 1), provider.fetchImpl);
    expect(provider.calls).toHaveLength(0);
    await sweepStaleManagedOperations(env, new Date(now.getTime() + 16 * 60_000), provider.fetchImpl);
    expect(provider.keys.has(seeded.hash)).toBe(false);
    expect((await listManagedOperationAttempts(env.BROKER_DB, seeded.operationId))[0]).toMatchObject({ outcome: 'cleaned', cleanup_token: null });
  });

  it('keeps cleanup eligible through repeated failures, auth expiry and persisted cleanup-required delivery', async () => {
    vi.useFakeTimers(); vi.setSystemTime(now);
    const env = createTestBrokerEnv(); const provider = controlledProvider(); vi.stubGlobal('fetch', provider.fetchImpl);
    const seeded = await seedRecoverableOperation(env, source, now);
    const issued = await workerPost(env, RESUME_PATH, seeded.authority);
    const delivered = await issued.json() as Record<string, string>;
    expect(issued.status).toBe(200);
    provider.failCleanup = true;
    for (let index = 0; index < 6; index++) {
      const later = new Date(now.getTime() + 16 * 60_000 + index * MANAGED_CLEANUP_RETRY_MS);
      vi.setSystemTime(later);
      await sweepStaleManagedOperations(env, later, provider.fetchImpl);
    }
    expect(provider.keys.has(delivered.managed_credential_ref!)).toBe(true);
    expect(await getManagedOperation(env.BROKER_DB, seeded.operationId)).toMatchObject({ state: 'FAILED', cleanup_attempts: 6 });
    await expireManagedOperation(env.BROKER_DB, (await getManagedOperation(env.BROKER_DB, seeded.operationId))!, new Date(now.getTime() + 61 * 60_000));
    env.__db.prepare("UPDATE managed_key_deliveries SET status = 'cleanup_required', failure_reason = 'managed_child_key_cleanup_failed' WHERE delivery_id = ?").run(delivered.delivery_id);
    provider.failCleanup = false;
    await sweepStaleManagedOperations(env, new Date(now.getTime() + 16 * 60_000 + 6 * MANAGED_CLEANUP_RETRY_MS), provider.fetchImpl);
    expect(provider.keys.has(delivered.managed_credential_ref!)).toBe(false);
    expect(await getManagedOperation(env.BROKER_DB, seeded.operationId)).toMatchObject({ state: 'FAILED', failure_reason: 'authorization_expired' });
    expect(env.__db.prepare('SELECT status, failure_reason FROM managed_key_deliveries WHERE delivery_id = ?').get(delivered.delivery_id)).toMatchObject({ status: 'expired', failure_reason: 'ack_expired_child_key_cleaned' });
    expect(provider.creates).toBe(1);
  });

  it.each(['brake', 'cap'] as const)('respects the current %s and makes legitimate same-operation progress when it is lifted', async (control) => {
    vi.useFakeTimers(); vi.setSystemTime(now);
    const env = createTestBrokerEnv(); const provider = controlledProvider(); vi.stubGlobal('fetch', provider.fetchImpl);
    const seeded = await seedRecoverableOperation(env, source, now);
    if (control === 'brake') env.__db.exec("UPDATE broker_config SET value = json_set(value, '$.brake.active', json('true'), '$.brake.reason', 'manual') WHERE key = 'abuse_runtime_state'");
    else {
      env.__db.exec("UPDATE broker_config SET value = json_set(value, '$.newActiveEntitlementsPerDay.maxCount', 1) WHERE key = 'abuse_controls'");
      env.__db.prepare(`INSERT INTO qq_managed_entitlements (qq_subject_ref, status, issue_ref, managed_credential_ref, budget_usd, reserved_at, issued_at, expires_at, delivered_at, created_at, updated_at) VALUES ('ph-qq-subject-v1_cap_other', 'active', 'cap-other-issue', 'cap-other-key', 0.07, ?, ?, '2027-01-17T10:00:00.000Z', ?, ?, ?)`).run(...Array(5).fill(now.toISOString()));
    }
    const blocked = await workerPost(env, RESUME_PATH, seeded.authority);
    expect(blocked.status).toBe(503); expect(provider.creates).toBe(0);
    expect(await getManagedOperation(env.BROKER_DB, seeded.operationId)).toMatchObject({ state: 'RETRY_READY' });
    env.__db.exec("UPDATE broker_config SET value = json_set(value, '$.brake.active', json('false')) WHERE key = 'abuse_runtime_state'");
    env.__db.exec("UPDATE broker_config SET value = json_set(value, '$.newActiveEntitlementsPerDay.maxCount', 500) WHERE key = 'abuse_controls'");
    const resumed = await workerPost(env, RESUME_PATH, seeded.authority);
    expect(resumed.status).toBe(200); expect(await resumed.json()).toHaveProperty('openrouter_api_key'); expect(provider.creates).toBe(1);
  });

  it('continues malformed and capped unknown enumeration without treating it as absence', async () => {
    const env = createTestBrokerEnv(); const provider = controlledProvider();
    const seeded = await seedRecoverableOperation(env, source, now);
    env.__db.prepare('UPDATE managed_operation_attempts SET managed_credential_ref = NULL WHERE operation_id = ?').run(seeded.operationId);
    provider.malformedList = true;
    await reconcileUnknownAttempt(env.BROKER_DB, env.OPENROUTER_MANAGEMENT_API_KEY, seeded.operation, now, provider.fetchImpl);
    expect(await getManagedOperation(env.BROKER_DB, seeded.operationId)).toMatchObject({ state: 'CLEANUP_REQUIRED' });
    expect(await startManagedOperationAttempt(env.BROKER_DB, (await getManagedOperation(env.BROKER_DB, seeded.operationId))!, now)).toEqual({ ok: false, reason: 'not_retry_ready' });
    provider.malformedList = false;
    for (let index = 0; index < 2000; index++) provider.keys.set(`other-${index}`, { hash: `other-${index}`, name: `other-${index}`, limit: 0.07, disabled: false });
    provider.keys.set(seeded.hash, { hash: seeded.hash, name: seeded.keyName, limit: 0.07, disabled: true, expiresAt: '2020-01-01T00:00:00.000Z' });
    await reconcileUnknownAttempt(env.BROKER_DB, env.OPENROUTER_MANAGEMENT_API_KEY, seeded.operation, now, provider.fetchImpl);
    expect(provider.keys.has(seeded.hash)).toBe(true);
    expect((await listManagedOperationAttempts(env.BROKER_DB, seeded.operationId))[0]).toMatchObject({ scan_offset: 2000, outcome: 'unknown' });
    await reconcileUnknownAttempt(env.BROKER_DB, env.OPENROUTER_MANAGEMENT_API_KEY, seeded.operation, now, provider.fetchImpl);
    expect(provider.keys.has(seeded.hash)).toBe(false);
    expect(provider.keys.size).toBe(2000);
    expect(await getManagedOperation(env.BROKER_DB, seeded.operationId)).toMatchObject({ state: 'RETRY_READY' });
    expect(provider.creates).toBe(0);
  });

  it('keeps multi-page absence unknown until a complete single-response inventory permits retry', async () => {
    const env = createTestBrokerEnv(); const provider = controlledProvider();
    const seeded = await seedRecoverableOperation(env, source, now);
    env.__db.prepare('UPDATE managed_operation_attempts SET managed_credential_ref = NULL WHERE operation_id = ?').run(seeded.operationId);
    for (let index = 0; index < 2001; index++) provider.keys.set(`other-${index}`, { hash: `other-${index}`, name: `other-${index}`, limit: 0.07, disabled: false });
    await reconcileUnknownAttempt(env.BROKER_DB, env.OPENROUTER_MANAGEMENT_API_KEY, seeded.operation, now, provider.fetchImpl);
    expect(await getManagedOperation(env.BROKER_DB, seeded.operationId)).toMatchObject({ state: 'CLEANUP_REQUIRED' });
    expect(provider.calls).toHaveLength(20);
    expect((await listManagedOperationAttempts(env.BROKER_DB, seeded.operationId))[0]).toMatchObject({ scan_offset: 2000, outcome: 'unknown' });
    await reconcileUnknownAttempt(env.BROKER_DB, env.OPENROUTER_MANAGEMENT_API_KEY, seeded.operation, now, provider.fetchImpl);
    expect(await getManagedOperation(env.BROKER_DB, seeded.operationId)).toMatchObject({ state: 'CLEANUP_REQUIRED' });
    expect(provider.keys.size).toBe(2001);
    expect(provider.calls).toHaveLength(21);
    expect(provider.calls.every((call) => call.method === 'GET')).toBe(true);
    expect((await startManagedOperationAttempt(env.BROKER_DB, (await getManagedOperation(env.BROKER_DB, seeded.operationId))!, now)).ok).toBe(false);
    for (let index = 0; index < 1902; index++) provider.keys.delete(`other-${index}`);
    await reconcileUnknownAttempt(env.BROKER_DB, env.OPENROUTER_MANAGEMENT_API_KEY, seeded.operation, now, provider.fetchImpl);
    expect(provider.keys.size).toBe(99);
    expect(await getManagedOperation(env.BROKER_DB, seeded.operationId)).toMatchObject({ state: 'RETRY_READY' });
    expect((await startManagedOperationAttempt(env.BROKER_DB, (await getManagedOperation(env.BROKER_DB, seeded.operationId))!, now)).ok).toBe(true);
  });

  it.each(['within-sweep', 'between-sweeps'] as const)('does not infer absence when provider deletions move a live target %s', async (movement) => {
    const env = createTestBrokerEnv(); const provider = controlledProvider();
    const seeded = await seedRecoverableOperation(env, source, now);
    env.__db.prepare('UPDATE managed_operation_attempts SET managed_credential_ref = NULL WHERE operation_id = ?').run(seeded.operationId);
    const beforeTarget = movement === 'within-sweep' ? 150 : 2200;
    const deleted = movement === 'within-sweep' ? 100 : 300;
    for (let index = 0; index < beforeTarget; index++) provider.keys.set(`other-${index}`, { hash: `other-${index}`, name: `other-${index}`, limit: 0.07, disabled: false });
    provider.keys.set(seeded.hash, { hash: seeded.hash, name: seeded.keyName, limit: 0.07, disabled: false });
    let shifted = false;
    const movingFetch: typeof fetch = async (input, init) => {
      const response = await provider.fetchImpl(input, init);
      const url = new URL(String(input));
      if (movement === 'within-sweep' && !shifted && url.pathname === '/api/v1/keys' && url.searchParams.get('offset') === '0') {
        shifted = true;
        for (let index = 0; index < deleted; index++) provider.keys.delete(`other-${index}`);
      }
      return response;
    };
    await reconcileUnknownAttempt(env.BROKER_DB, env.OPENROUTER_MANAGEMENT_API_KEY, seeded.operation, now, movingFetch);
    if (movement === 'between-sweeps') {
      for (let index = 0; index < deleted; index++) provider.keys.delete(`other-${index}`);
      await reconcileUnknownAttempt(env.BROKER_DB, env.OPENROUTER_MANAGEMENT_API_KEY, seeded.operation, now, movingFetch);
    }
    expect(provider.keys.has(seeded.hash)).toBe(true);
    expect(provider.creates).toBe(0);
    expect(await getManagedOperation(env.BROKER_DB, seeded.operationId)).toMatchObject({ state: 'CLEANUP_REQUIRED' });
    expect((await listManagedOperationAttempts(env.BROKER_DB, seeded.operationId))[0]).toMatchObject({ scan_offset: 0, outcome: 'unknown' });
    expect((await startManagedOperationAttempt(env.BROKER_DB, (await getManagedOperation(env.BROKER_DB, seeded.operationId))!, now)).ok).toBe(false);
    await reconcileUnknownAttempt(env.BROKER_DB, env.OPENROUTER_MANAGEMENT_API_KEY, seeded.operation, now, movingFetch);
    expect(provider.keys.has(seeded.hash)).toBe(false);
    expect(await getManagedOperation(env.BROKER_DB, seeded.operationId)).toMatchObject({ state: 'RETRY_READY' });
    expect((await startManagedOperationAttempt(env.BROKER_DB, (await getManagedOperation(env.BROKER_DB, seeded.operationId))!, now)).ok).toBe(true);
  });

  it('keeps delayed record, release, terminal and post-create work fenced after later delivery', async () => {
    vi.useFakeTimers(); vi.setSystemTime(now);
    const env = createTestBrokerEnv(); const provider = controlledProvider(); vi.stubGlobal('fetch', provider.fetchImpl);
    const seeded = await seedRecoverableOperation(env, source, now);
    const fresh = await workerPost(env, RESUME_PATH, seeded.authority); const delivered = await fresh.json() as Record<string, string>;
    expect(fresh.status).toBe(200);
    expect((await workerPost(env, ACK_PATH, { delivery_id: delivered.delivery_id, managed_credential_ref: delivered.managed_credential_ref, delivery_ack_token: delivered.delivery_ack_token })).status).toBe(200);
    const before = provider.calls.length;
    expect(await recordAttemptCredential(env.BROKER_DB, seeded.operationId, 1, 'late-old-key', now)).toBe(false);
    await markAttemptUnknown(env.BROKER_DB, seeded.operationId, 1, now);
    expect(await releaseUnstartedOperationAttempt(env.BROKER_DB, seeded.operationId, 1, now)).toBe(false);
    await failManagedOperationTerminal(env.BROKER_DB, seeded.operation, now, 'terminal_provider_failure');
    expect(await transitionOperationToPostCreateState(env.BROKER_DB, env.OPENROUTER_MANAGEMENT_API_KEY, seeded.operationId, 1, 'DELIVERY_PENDING', now)).toBeNull();
    expect(provider.calls).toHaveLength(before);
    expect(provider.keys.has(delivered.managed_credential_ref!)).toBe(true);
    expect(await getManagedOperation(env.BROKER_DB, seeded.operationId)).toMatchObject({ state: 'ACTIVE', current_attempt_index: 2 });
  });
});

describe('source-specific resume policy', () => {
  it.each(['denylist', 'reputation', 'revocation'] as const)('checks current Discord %s hooks but excludes internal recovery from user velocity accounting', async (hook) => {
    vi.useFakeTimers(); vi.setSystemTime(now);
    const env = createTestBrokerEnv(); const provider = controlledProvider(); vi.stubGlobal('fetch', provider.fetchImpl);
    const seeded = await seedRecoverableOperation(env, 'discord', now);
    env.__db.prepare(`INSERT INTO broker_abuse_subject_hooks (hook_kind, subject_type, subject_value, outcome_code, outcome_class, active, reason) VALUES (?, 'installation_id', ?, 'trial_not_eligible', 'terminal', 1, 'controlled policy')`).run(hook, seeded.installationId);
    const blocked = await workerPost(env, RESUME_PATH, seeded.authority);
    expect(blocked.status).toBe(409); expect(provider.creates).toBe(0);
    env.__db.prepare('DELETE FROM broker_abuse_subject_hooks').run();
    env.__db.prepare(`INSERT INTO broker_velocity_cap_hooks (subject_type, subject_value, max_requests, window_minutes, outcome_code, outcome_class, active) VALUES ('installation_id', ?, 1, 60, 'rate_limited', 'retryable', 1)`).run(seeded.installationId);
    for (const endpoint of [STATUS_PATH, RESUME_PATH, ACK_PATH]) for (let index = 0; index < 4; index++) await seedRequestEvent(env, { endpoint: `POST ${endpoint}`, installationId: seeded.installationId, observedAt: now.toISOString() });
    for (let index = 0; index < 2; index++) await seedRequestEvent(env, { endpoint: 'POST /v1/auth/discord/start', installationId: seeded.installationId, observedAt: now.toISOString() });
    expect((await workerPost(env, RESUME_PATH, seeded.authority)).status).toBe(429);
    expect(provider.creates).toBe(0);
    env.__db.prepare("DELETE FROM broker_request_events WHERE endpoint = 'POST /v1/auth/discord/start'").run();
    const recovered = await workerPost(env, RESUME_PATH, seeded.authority);
    expect(recovered.status).toBe(200); expect(await recovered.json()).toHaveProperty('openrouter_api_key'); expect(provider.creates).toBe(1);
  });

  it('rejects legacy unbound QQ authority without rebinding and still cleans its old resource', async () => {
    vi.useFakeTimers(); vi.setSystemTime(now);
    const env = createTestBrokerEnv(); const provider = controlledProvider(); vi.stubGlobal('fetch', provider.fetchImpl);
    const seeded = await seedRecoverableOperation(env, 'qq', now, null);
    provider.keys.set(seeded.hash, { hash: seeded.hash, name: seeded.keyName, limit: 0.07, disabled: false });
    const result = await workerPost(env, RESUME_PATH, { ...seeded.authority, installation_id: 'unverified-resumer' });
    expect(result.status).toBe(200); expect(await result.json()).toMatchObject({ state: 'FAILED', client_action: 'action_required' });
    expect(provider.creates).toBe(0);
    expect(await getManagedOperation(env.BROKER_DB, seeded.operationId)).toMatchObject({ installation_id: null });
    await sweepStaleManagedOperations(env, now, provider.fetchImpl);
    expect(provider.keys.has(seeded.hash)).toBe(false);
  });

  it('returns terminal revoked QQ guidance without creating repeated ambiguous attempts', async () => {
    vi.useFakeTimers(); vi.setSystemTime(now);
    const env = createTestBrokerEnv(); const provider = controlledProvider(); vi.stubGlobal('fetch', provider.fetchImpl);
    const seeded = await seedRecoverableOperation(env, 'qq', now);
    env.__db.prepare(`INSERT INTO qq_managed_entitlements (qq_subject_ref, status, issue_ref, budget_usd, reserved_at, created_at, updated_at) VALUES (?, 'revoked', 'controlled-revoked', 0.11, ?, ?, ?)`).run(seeded.subjectRef, now.toISOString(), now.toISOString(), now.toISOString());
    for (let index = 0; index < 2; index++) {
      const response = await workerPost(env, RESUME_PATH, seeded.authority);
      expect(response.status).toBe(200); expect(await response.json()).toMatchObject({ state: 'FAILED', client_action: 'action_required', failure_reason: 'terminal_provider_failure' });
    }
    expect(provider.creates).toBe(0);
    expect(await getManagedOperation(env.BROKER_DB, seeded.operationId)).toMatchObject({ attempt_count: 2 });
  });
});

it('shares scheduled enumeration pages and resumes the oldest due attempt without abandoning the next', async () => {
  const env = createTestBrokerEnv(); const provider = controlledProvider();
  const first = await seedRecoverableOperation(env, 'qq', now, 'page-first');
  const second = await seedRecoverableOperation(env, 'qq', now, 'page-second');
  env.__db.exec(`UPDATE managed_operation_attempts SET managed_credential_ref = NULL, create_in_flight = 1`);
  env.__db.exec(`UPDATE managed_operations SET state = 'FAILED', failure_reason = 'authorization_expired', client_action = 'action_required'`);
  for (let index = 0; index < 700; index++) provider.keys.set(`other-${index}`, { hash: `other-${index}`, name: `other-${index}`, limit: 0.07, disabled: false });
  await sweepStaleManagedOperations(env, now, provider.fetchImpl);
  expect(provider.calls).toHaveLength(6);
  expect((await listManagedOperationAttempts(env.BROKER_DB, first.operationId))[0]).toMatchObject({ scan_offset: 600, outcome: 'unknown' });
  await sweepStaleManagedOperations(env, new Date(now.getTime() + 60_000), provider.fetchImpl);
  expect(provider.calls).toHaveLength(12);
  expect((await listManagedOperationAttempts(env.BROKER_DB, second.operationId))[0]).toMatchObject({ scan_offset: 600, outcome: 'unknown' });
  provider.keys.set(first.hash, { hash: first.hash, name: first.keyName, limit: 0.07, disabled: false });
  await sweepStaleManagedOperations(env, new Date(now.getTime() + MANAGED_CLEANUP_RETRY_MS), provider.fetchImpl);
  expect(provider.calls.slice(12).filter(call => new URL(call.url).pathname === '/api/v1/keys')).toHaveLength(2);
  expect(provider.keys.has(first.hash)).toBe(false);
  expect((await listManagedOperationAttempts(env.BROKER_DB, first.operationId))[0]).toMatchObject({ outcome: 'cleaned', scan_offset: 0 });
  expect((await listManagedOperationAttempts(env.BROKER_DB, second.operationId))[0]).toMatchObject({ outcome: 'unknown', create_in_flight: 1, scan_offset: 600 });
});
