import { afterEach, describe, expect, it, vi } from 'vitest';

import { app } from '../src/app';
import {
  authorizeManagedOperationRequest,
  bindOperationForIssue,
  buildManagedOperationId,
  buildManagedOperationResumeToken,
  createManagedOperation,
  getManagedOperation,
  hashManagedOperationResumeToken,
  listManagedOperationAttempts,
  markAttemptUnknown,
  providerKeyNameForOperationAttempt,
  reconcileUnknownAttempt,
  recordAttemptCredential,
  startManagedOperationAttempt,
  sweepStaleManagedOperations,
  transitionManagedOperation,
  transitionOperationToPostCreateState,
  failManagedOperationTerminal,
  expireManagedOperation,
  findConflictingOperationDelivery,
  markOperationActiveOnAck,
} from '../src/managed-operation';
import {
  createTestBrokerEnv,
  type TestBrokerEnv,
} from './test-support/sqlite-d1';

const NOW = new Date('2026-09-01T10:00:00.000Z');
const SUBJECT = 'ph-discord-user-v1_operation_subject_test';
const INSTALLATION = 'install-managed-operation-test';
const DEVICE_KEY = 'device-public-key-operation-test';

async function createBoundOperation(env: TestBrokerEnv, now: Date = NOW) {
  const operationId = buildManagedOperationId();
  const resumeToken = buildManagedOperationResumeToken();
  const resumeTokenHash = await hashManagedOperationResumeToken(resumeToken);
  const created = await createManagedOperation(env.BROKER_DB, {
    operationId,
    resumeTokenHash,
    issueSource: 'discord',
    subjectRef: SUBJECT,
    installationId: INSTALLATION,
    devicePublicKey: DEVICE_KEY,
    now,
  });
  return { operationId, resumeToken, operation: created.operation };
}

function mockProviderList(initialKeysByName: Record<string, string>) {
  const calls: Array<{ url: string; method: string }> = [];
  const liveKeys = new Map(Object.entries(initialKeysByName));
  const fetchMock = vi.fn(async (input: string | URL, init?: RequestInit) => {
    const url = String(input);
    const method = init?.method ?? 'GET';
    calls.push({ url, method });
    if (url.includes('/keys?limit=') && method === 'GET') {
      return Response.json({
        data: [...liveKeys.entries()].map(([name, hash]) => ({
          name,
          hash,
          limit: 0.07,
        })),
      });
    }
    if (method === 'PATCH') {
      return Response.json({ data: { disabled: true } });
    }
    if (method === 'DELETE') {
      const hash = url.split('/').pop() ?? '';
      for (const [name, keyHash] of liveKeys) {
        if (keyHash === hash) {
          liveKeys.delete(name);
        }
      }
      return new Response(null, { status: 204 });
    }
    throw new Error(`unexpected provider request: ${method} ${url}`);
  });
  vi.stubGlobal('fetch', fetchMock as typeof fetch);
  return { calls, fetchMock };
}

describe('managed operation lifecycle', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it('creates an operation identity with hashed resume token and deterministic attempt names', async () => {
    const env = createTestBrokerEnv();
    const { operationId, operation } = await createBoundOperation(env);

    expect(operationId.startsWith('ph-mop-v1_')).toBe(true);
    expect(operation.resume_token_hash.startsWith('ph-mop-resume-v1_')).toBe(true);
    expect(operation.state).toBe('AUTHENTICATED');
    expect(operation.client_action).toBe('wait');
    expect(operation.auth_expires_at).toBe(
      new Date(NOW.getTime() + 60 * 60_000).toISOString(),
    );

    const started = await startManagedOperationAttempt(env.BROKER_DB, operation, NOW);
    expect(started.ok).toBe(true);
    if (!started.ok) {
      return;
    }
    expect(started.attempt.attempt_index).toBe(1);
    expect(started.attempt.provider_key_name).toBe(
      providerKeyNameForOperationAttempt(operationId, 'discord', 1),
    );
    expect(started.attempt.outcome).toBe('unknown');

    const refreshed = await getManagedOperation(env.BROKER_DB, operationId);
    expect(refreshed?.state).toBe('CREATING');
    expect(refreshed?.attempt_count).toBe(1);
  });

  it('recovers an unknown attempt with no provider resource to retry-ready without creating a key', async () => {
    const env = createTestBrokerEnv();
    const { operationId, operation } = await createBoundOperation(env);
    const started = await startManagedOperationAttempt(env.BROKER_DB, operation, NOW);
    expect(started.ok).toBe(true);
    if (!started.ok) {
      return;
    }
    await markAttemptUnknown(env.BROKER_DB, operationId, 1, NOW);

    const { calls } = mockProviderList({});
    const { reconcileUnknownAttempt } = await import('../src/managed-operation');
    const reconciled = await reconcileUnknownAttempt(
      env.BROKER_DB,
      env.OPENROUTER_MANAGEMENT_API_KEY,
      (await getManagedOperation(env.BROKER_DB, operationId))!,
      NOW,
    );
    expect(reconciled?.state).toBe('RETRY_READY');
    expect(calls.some((call) => call.method === 'POST')).toBe(false);

    const attempts = await listManagedOperationAttempts(env.BROKER_DB, operationId);
    expect(attempts).toHaveLength(1);
    expect(attempts[0]?.outcome).toBe('cleaned');
  });

  it('reconciles an orphan provider key by name, verifies cleanup, then allows attempt two', async () => {
    const env = createTestBrokerEnv();
    const { operationId, operation } = await createBoundOperation(env);
    const started = await startManagedOperationAttempt(env.BROKER_DB, operation, NOW);
    expect(started.ok).toBe(true);
    if (!started.ok) {
      return;
    }
    const keyName = providerKeyNameForOperationAttempt(operationId, 'discord', 1);
    await recordAttemptCredential(env.BROKER_DB, operationId, 1, 'hash_orphan_attempt_1', NOW);
    await markAttemptUnknown(env.BROKER_DB, operationId, 1, NOW);

    const listed = { [keyName]: 'hash_orphan_attempt_1' };
    const { calls } = mockProviderList(listed);
    const module = await import('../src/managed-operation');
    const reconciled = await module.reconcileUnknownAttempt(
      env.BROKER_DB,
      env.OPENROUTER_MANAGEMENT_API_KEY,
      (await getManagedOperation(env.BROKER_DB, operationId))!,
      NOW,
    );
    expect(calls.some((call) => call.method === 'POST')).toBe(false);
    expect(calls.some((call) => call.method === 'PATCH')).toBe(true);
    expect(calls.some((call) => call.method === 'DELETE')).toBe(true);
    expect(reconciled?.state).toBe('RETRY_READY');

    const second = await startManagedOperationAttempt(
      env.BROKER_DB,
      (await getManagedOperation(env.BROKER_DB, operationId))!,
      NOW,
    );
    expect(second.ok).toBe(true);
    if (second.ok) {
      expect(second.attempt.attempt_index).toBe(2);
      expect(second.attempt.provider_key_name).not.toBe(keyName);
    }
  });

  it('blocks fresh issuance until cleanup is verified', async () => {
    const env = createTestBrokerEnv();
    const { operation } = await createBoundOperation(env);
    const started = await startManagedOperationAttempt(env.BROKER_DB, operation, NOW);
    expect(started.ok).toBe(true);

    const again = await startManagedOperationAttempt(
      env.BROKER_DB,
      (await getManagedOperation(env.BROKER_DB, operation.operation_id))!,
      NOW,
    );
    expect(again).toEqual({ ok: false, reason: 'not_retry_ready' });

    await transitionManagedOperation(env.BROKER_DB, operation.operation_id, 'CREATE_UNKNOWN', NOW, {
      from: ['CREATING'],
    });
    const blocked = await startManagedOperationAttempt(
      env.BROKER_DB,
      (await getManagedOperation(env.BROKER_DB, operation.operation_id))!,
      NOW,
    );
    expect(blocked).toEqual({ ok: false, reason: 'not_retry_ready' });
  });

  it('returns wait for nonterminal re-POST without side effects', async () => {
    const env = createTestBrokerEnv();
    const { operationId, resumeToken, operation } = await createBoundOperation(env);
    await startManagedOperationAttempt(env.BROKER_DB, operation, NOW);

    const binding = await bindOperationForIssue(env.BROKER_DB, {
      operationId,
      resumeToken,
      issueSource: 'discord',
      subjectRef: SUBJECT,
      installationId: INSTALLATION,
      devicePublicKey: DEVICE_KEY,
      now: NOW,
    });
    expect(binding.status).toBe('wait');
    if (binding.status === 'wait') {
      expect(binding.operation.client_action).toBe('wait');
    }
    const attempts = await listManagedOperationAttempts(env.BROKER_DB, operationId);
    expect(attempts).toHaveLength(1);
  });

  it('rejects binding mismatches and unknown tokens', async () => {
    const env = createTestBrokerEnv();
    const { operationId } = await createBoundOperation(env);

    await expect(
      bindOperationForIssue(env.BROKER_DB, {
        operationId,
        resumeToken: 'wrong-token',
        issueSource: 'discord',
        subjectRef: SUBJECT,
        installationId: INSTALLATION,
        devicePublicKey: DEVICE_KEY,
        now: NOW,
      }),
    ).resolves.toMatchObject({ status: 'invalid' });

    await expect(
      bindOperationForIssue(env.BROKER_DB, {
        operationId,
        resumeToken: buildManagedOperationResumeToken(),
        issueSource: 'qq',
        subjectRef: SUBJECT,
        installationId: INSTALLATION,
        devicePublicKey: DEVICE_KEY,
        now: NOW,
      }),
    ).resolves.toMatchObject({ status: 'invalid', reason: 'binding_mismatch' });

    await expect(
      bindOperationForIssue(env.BROKER_DB, {
        operationId: buildManagedOperationId(),
        resumeToken: buildManagedOperationResumeToken(),
        issueSource: 'discord',
        subjectRef: SUBJECT,
        installationId: INSTALLATION,
        devicePublicKey: DEVICE_KEY,
        now: NOW,
      }),
    ).resolves.toMatchObject({ status: 'proceed', created: true });
  });

  it('expires recovery authorization after 60 minutes and reports action required', async () => {
    const env = createTestBrokerEnv();
    const { operationId, resumeToken } = await createBoundOperation(env);

    const late = new Date(NOW.getTime() + 61 * 60_000);
    const auth = await authorizeManagedOperationRequest(env.BROKER_DB, {
      operationId,
      resumeToken,
      installationId: INSTALLATION,
      now: late,
    });
    expect(auth).toEqual({ ok: false, reason: 'expired' });

    const terminal = await getManagedOperation(env.BROKER_DB, operationId);
    expect(terminal?.state).toBe('FAILED');
    expect(terminal?.failure_reason).toBe('authorization_expired');
    expect(terminal?.client_action).toBe('action_required');
  });

  it('fails terminally on definitive provider rejection with terminal referral failure', async () => {
    const env = createTestBrokerEnv();
    const { operationId, operation } = await createBoundOperation(env);
    const { failManagedOperationTerminal } = await import('../src/managed-operation');
    const terminal = await failManagedOperationTerminal(env.BROKER_DB, operation, NOW, 'terminal_provider_failure');
    expect(terminal?.state).toBe('FAILED');
    expect(terminal?.failure_reason).toBe('terminal_provider_failure');
    expect(terminal?.client_action).toBe('action_required');
    expect((await getManagedOperation(env.BROKER_DB, operationId))?.state).toBe('FAILED');
  });

  it('serves status and resume routes with the frozen request contract', async () => {
    const env = createTestBrokerEnv();
    const { operationId, resumeToken } = await createBoundOperation(env, new Date());

    const statusResponse = await app.request(
      'http://broker.test/v1/providers/openrouter/managed-operation/status',
      {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          operation_id: operationId,
          resume_token: resumeToken,
          installation_id: INSTALLATION,
        }),
      },
      env,
    );
    expect(statusResponse.status).toBe(200);
    const statusBody = (await statusResponse.json()) as Record<string, unknown>;
    expect(statusBody).toMatchObject({
      ok: true,
      operation_id: operationId,
      issue_source: 'discord',
      state: 'AUTHENTICATED',
      client_action: 'wait',
      failure_reason: null,
      attempt_count: 0,
      referral: { status: 'none', settlement: 'none' },
      attempts: [],
      delivery: null,
    });

    const badResponse = await app.request(
      'http://broker.test/v1/providers/openrouter/managed-operation/resume',
      {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          operation_id: operationId,
          resume_token: 'wrong',
          installation_id: INSTALLATION,
        }),
      },
      env,
    );
    expect(badResponse.status).toBe(404);

    const resumeResponse = await app.request(
      'http://broker.test/v1/providers/openrouter/managed-operation/resume',
      {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          operation_id: operationId,
          resume_token: resumeToken,
          installation_id: INSTALLATION,
        }),
      },
      env,
    );
    expect(resumeResponse.status).toBe(409);
    await expect(resumeResponse.json()).resolves.toMatchObject({
      ok: false,
      code: 'resume_issuance_unavailable',
      state: 'AUTHENTICATED',
    });
  });

  it('keeps terminal failures visible instead of reporting unknown operation', async () => {
    const env = createTestBrokerEnv();
    const { operationId, resumeToken, operation } = await createBoundOperation(env, new Date());
    const { failManagedOperationTerminal } = await import('../src/managed-operation');
    await failManagedOperationTerminal(env.BROKER_DB, operation, new Date(), 'terminal_provider_failure');

    const statusResponse = await app.request(
      'http://broker.test/v1/providers/openrouter/managed-operation/status',
      {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          operation_id: operationId,
          resume_token: resumeToken,
          installation_id: INSTALLATION,
        }),
      },
      env,
    );
    expect(statusResponse.status).toBe(200);
    await expect(statusResponse.json()).resolves.toMatchObject({
      ok: true,
      state: 'FAILED',
      failure_reason: 'terminal_provider_failure',
      client_action: 'action_required',
    });
  });

  it('reports expired authorization with the terminal body', async () => {
    const env = createTestBrokerEnv();
    const { operationId, resumeToken } = await createBoundOperation(env, new Date(Date.now() - 61 * 60_000));

    const statusResponse = await app.request(
      'http://broker.test/v1/providers/openrouter/managed-operation/status',
      {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          operation_id: operationId,
          resume_token: resumeToken,
          installation_id: INSTALLATION,
        }),
      },
      env,
    );
    expect(statusResponse.status).toBe(410);
    await expect(statusResponse.json()).resolves.toMatchObject({
      ok: true,
      state: 'FAILED',
      failure_reason: 'authorization_expired',
    });
  });

  it('mirrors ACK activation and settlement progress onto the operation', async () => {
    const env = createTestBrokerEnv();
    const { operationId } = await createBoundOperation(env);
    const { transitionManagedOperation, markOperationActiveOnAck, markOperationSettlementStatus } =
      await import('../src/managed-operation');
    await transitionManagedOperation(env.BROKER_DB, operationId, 'DELIVERY_PENDING', NOW);
    env.__db
      .prepare(
        `INSERT INTO managed_key_deliveries (
          delivery_id, issue_source, subject_ref, installation_id, managed_credential_ref,
          ack_token_hash, status, created_at, expires_at, acknowledged_at, operation_id, attempt_index
        ) VALUES (?, 'discord', ?, ?, ?, ?, 'acknowledged', ?, ?, ?, ?, 1)`,
      )
      .run(
        'ph-delivery-v1_mirror_test',
        SUBJECT,
        INSTALLATION,
        'hash_mirror_test',
        'ph-delivery-ack-token-v1_' + 'b'.repeat(64),
        NOW.toISOString(),
        new Date(NOW.getTime() + 15 * 60_000).toISOString(),
        NOW.toISOString(),
        operationId,
      );
    await markOperationActiveOnAck(env.BROKER_DB, 'ph-delivery-v1_mirror_test', NOW);
    expect((await getManagedOperation(env.BROKER_DB, operationId))?.state).toBe('ACTIVE');

    await markOperationSettlementStatus(env.BROKER_DB, operationId, { settlement: 'invitee_pending' }, NOW);
    expect((await getManagedOperation(env.BROKER_DB, operationId))?.settlement_status).toBe(
      'invitee_pending',
    );
    await markOperationSettlementStatus(
      env.BROKER_DB,
      operationId,
      { referral: 'credited', settlement: 'referrer_pending' },
      NOW,
    );
    const mirrored = await getManagedOperation(env.BROKER_DB, operationId);
    expect(mirrored?.referral_status).toBe('credited');
    expect(mirrored?.settlement_status).toBe('referrer_pending');
  });

  it('reconciles with a stale caller snapshot by re-reading the current attempt', async () => {
    const env = createTestBrokerEnv();
    const { operationId, operation } = await createBoundOperation(env);
    const started = await startManagedOperationAttempt(env.BROKER_DB, operation, NOW);
    expect(started.ok).toBe(true);
    if (!started.ok) {
      return;
    }
    const keyName = providerKeyNameForOperationAttempt(operationId, 'discord', 1);
    await recordAttemptCredential(env.BROKER_DB, operationId, 1, 'hash_stale_snapshot_1', NOW);
    await markAttemptUnknown(env.BROKER_DB, operationId, 1, NOW);

    mockProviderList({ [keyName]: 'hash_stale_snapshot_1' });
    const reconciled = await reconcileUnknownAttempt(
      env.BROKER_DB,
      env.OPENROUTER_MANAGEMENT_API_KEY,
      operation,
      NOW,
    );
    expect(reconciled?.state).toBe('RETRY_READY');
    const attempts = await listManagedOperationAttempts(env.BROKER_DB, operationId);
    expect(attempts).toHaveLength(1);
    expect(attempts[0]).toEqual(
      expect.objectContaining({ attempt_index: 1, outcome: 'cleaned' }),
    );
  });

  it('fences concurrent attempt claims so only one attempt row exists', async () => {
    const env = createTestBrokerEnv();
    const { operationId, operation } = await createBoundOperation(env);
    const [first, second] = await Promise.all([
      startManagedOperationAttempt(env.BROKER_DB, operation, NOW),
      startManagedOperationAttempt(env.BROKER_DB, operation, NOW),
    ]);
    const winners = [first, second].filter((result) => result.ok);
    expect(winners).toHaveLength(1);
    expect([first, second]).toContainEqual({ ok: false, reason: 'not_retry_ready' });
    expect(await listManagedOperationAttempts(env.BROKER_DB, operationId)).toHaveLength(1);
  });

  it('activates a delivery-pending operation from an acknowledged delivery and stays converged on retry', async () => {
    const env = createTestBrokerEnv();
    const { operationId, operation } = await createBoundOperation(env);
    const started = await startManagedOperationAttempt(env.BROKER_DB, operation, NOW);
    expect(started.ok).toBe(true);
    if (!started.ok) {
      return;
    }
    await recordAttemptCredential(env.BROKER_DB, operationId, 1, 'hash_ack_converge_1', NOW);
    const { createManagedKeyDelivery, markManagedKeyDeliveryAcknowledged } = await import(
      '../src/managed-key-delivery'
    );
    const delivery = await createManagedKeyDelivery(env.BROKER_DB, {
      issueSource: 'discord',
      subjectRef: SUBJECT,
      installationId: INSTALLATION,
      managedCredentialRef: 'hash_ack_converge_1',
      createdAt: NOW,
      expiresAt: new Date(NOW.getTime() + 15 * 60_000),
      operationId,
      attemptIndex: 1,
    });
    await transitionManagedOperation(env.BROKER_DB, operationId, 'DELIVERY_PENDING', NOW);
    await markManagedKeyDeliveryAcknowledged(env.BROKER_DB, {
      deliveryId: delivery.deliveryId,
      acknowledgedAt: NOW,
    });

    const { markOperationActiveOnAck, buildManagedOperationStatusBodyWithDelivery } = await import(
      '../src/managed-operation'
    );
    expect(await markOperationActiveOnAck(env.BROKER_DB, delivery.deliveryId, NOW)).toBe(true);
    expect(await markOperationActiveOnAck(env.BROKER_DB, delivery.deliveryId, NOW)).toBe(true);
    const active = (await getManagedOperation(env.BROKER_DB, operationId))!;
    expect(active.state).toBe('ACTIVE');
    const body = await buildManagedOperationStatusBodyWithDelivery(
      env.BROKER_DB,
      active,
      await listManagedOperationAttempts(env.BROKER_DB, operationId),
    );
    expect(body).toEqual(expect.objectContaining({ state: 'ACTIVE', client_action: 'wait' }));
  });

  it('sweeps a stale delivery-pending operation to active when its delivery was acknowledged', async () => {
    const env = createTestBrokerEnv();
    const { operationId, operation } = await createBoundOperation(env);
    const started = await startManagedOperationAttempt(env.BROKER_DB, operation, NOW);
    expect(started.ok).toBe(true);
    if (!started.ok) {
      return;
    }
    const { markManagedKeyDeliveryAcknowledged } = await import('../src/managed-key-delivery');
    const { createManagedKeyDelivery } = await import('../src/managed-key-delivery');
    const delivery = await createManagedKeyDelivery(env.BROKER_DB, {
      issueSource: 'discord',
      subjectRef: SUBJECT,
      installationId: INSTALLATION,
      managedCredentialRef: 'hash_sweep_ack_1',
      createdAt: NOW,
      expiresAt: new Date(NOW.getTime() + 15 * 60_000),
      operationId,
      attemptIndex: 1,
    });
    await transitionManagedOperation(env.BROKER_DB, operationId, 'DELIVERY_PENDING', NOW);
    await markManagedKeyDeliveryAcknowledged(env.BROKER_DB, {
      deliveryId: delivery.deliveryId,
      acknowledgedAt: NOW,
    });

    mockProviderList({});
    const result = await sweepStaleManagedOperations(env, new Date(NOW.getTime() + 16 * 60_000));
    expect(result.reconciled).toBe(1);
    const swept = (await getManagedOperation(env.BROKER_DB, operationId))!;
    expect(swept.state).toBe('ACTIVE');
    expect(swept.client_action).toBe('wait');
  });

  it('sweeps a stale delivery-pending operation with no acknowledgement back to retry-ready', async () => {
    const env = createTestBrokerEnv();
    const { operationId, operation } = await createBoundOperation(env);
    const started = await startManagedOperationAttempt(env.BROKER_DB, operation, NOW);
    expect(started.ok).toBe(true);
    if (!started.ok) {
      return;
    }
    await recordAttemptCredential(env.BROKER_DB, operationId, 1, 'hash_sweep_retry_1', NOW);
    const { createManagedKeyDelivery } = await import('../src/managed-key-delivery');
    await createManagedKeyDelivery(env.BROKER_DB, {
      issueSource: 'discord',
      subjectRef: SUBJECT,
      installationId: INSTALLATION,
      managedCredentialRef: 'hash_sweep_retry_1',
      createdAt: NOW,
      expiresAt: new Date(NOW.getTime() + 15 * 60_000),
      operationId,
      attemptIndex: 1,
    });
    await transitionManagedOperation(env.BROKER_DB, operationId, 'DELIVERY_PENDING', NOW);

    const { calls } = mockProviderList({});
    const result = await sweepStaleManagedOperations(env, new Date(NOW.getTime() + 16 * 60_000));
    expect(result.retryReady).toBe(1);
    expect(calls.some((call) => call.method === 'POST')).toBe(false);
    expect((await getManagedOperation(env.BROKER_DB, operationId))?.state).toBe('RETRY_READY');
    const second = await startManagedOperationAttempt(
      env.BROKER_DB,
      (await getManagedOperation(env.BROKER_DB, operationId))!,
      new Date(NOW.getTime() + 16 * 60_000),
    );
    expect(second.ok).toBe(true);
  });
  it('holds a stale delivery-pending operation in cleanup-required when the key survives cleanup', async () => {
    const env = createTestBrokerEnv();
    const { operationId, operation } = await createBoundOperation(env);
    const started = await startManagedOperationAttempt(env.BROKER_DB, operation, NOW);
    expect(started.ok).toBe(true);
    if (!started.ok) {
      return;
    }
    await recordAttemptCredential(env.BROKER_DB, operationId, 1, 'hash_sweep_sticky_1', NOW);
    const { createManagedKeyDelivery } = await import('../src/managed-key-delivery');
    await createManagedKeyDelivery(env.BROKER_DB, {
      issueSource: 'discord',
      subjectRef: SUBJECT,
      installationId: INSTALLATION,
      managedCredentialRef: 'hash_sweep_sticky_1',
      createdAt: NOW,
      expiresAt: new Date(NOW.getTime() + 15 * 60_000),
      operationId,
      attemptIndex: 1,
    });
    await transitionManagedOperation(env.BROKER_DB, operationId, 'DELIVERY_PENDING', NOW);

    const keyName = providerKeyNameForOperationAttempt(operationId, 'discord', 1);
    const stickyKeys = new Map([[keyName, 'hash_sweep_sticky_1']]);
    const stickyFetch = vi.fn(async (input: string | URL, init?: RequestInit) => {
      const url = String(input);
      const method = init?.method ?? 'GET';
      if (url.includes('/keys?limit=') && method === 'GET') {
        return Response.json({
          data: [...stickyKeys.entries()].map(([name, hash]) => ({ name, hash, limit: 0.07 })),
        });
      }
      if (method === 'PATCH') {
        return Response.json({ data: { disabled: true } });
      }
      if (method === 'DELETE') {
        return new Response(null, { status: 204 });
      }
      throw new Error(`unexpected provider request: ${method} ${url}`);
    });
    vi.stubGlobal('fetch', stickyFetch as typeof fetch);
    const result = await sweepStaleManagedOperations(env, new Date(NOW.getTime() + 16 * 60_000));
    expect(result.retryReady).toBe(0);
    expect((await getManagedOperation(env.BROKER_DB, operationId))?.state).toBe('CLEANUP_REQUIRED');
    const attempts = await listManagedOperationAttempts(env.BROKER_DB, operationId);
    expect(attempts).toHaveLength(1);
    expect(attempts[0]?.outcome).not.toBe('cleaned');
    const retry = await startManagedOperationAttempt(
      env.BROKER_DB,
      (await getManagedOperation(env.BROKER_DB, operationId))!,
      new Date(NOW.getTime() + 16 * 60_000),
    );
    expect(retry).toEqual({ ok: false, reason: 'not_retry_ready' });
  });

  it('recovers a lost post-create transition with verified provider cleanup and no false success', async () => {
    const env = createTestBrokerEnv();
    const { operationId, operation } = await createBoundOperation(env);
    const started = await startManagedOperationAttempt(env.BROKER_DB, operation, NOW);
    expect(started.ok).toBe(true);
    if (!started.ok) {
      return;
    }
    const keyName = providerKeyNameForOperationAttempt(operationId, 'discord', 1);
    await recordAttemptCredential(env.BROKER_DB, operationId, 1, 'hash_lost_transition_1', NOW);
    await markAttemptUnknown(env.BROKER_DB, operationId, 1, NOW);
    await transitionManagedOperation(env.BROKER_DB, operationId, 'FAILED', NOW, {
      failureReason: 'terminal_provider_failure',
    });

    const { calls } = mockProviderList({ [keyName]: 'hash_lost_transition_1' });
    const settled = await transitionOperationToPostCreateState(
      env.BROKER_DB,
      env.OPENROUTER_MANAGEMENT_API_KEY,
      operationId,
      'DELIVERY_PENDING',
      NOW,
    );
    expect(settled?.state).not.toBe('DELIVERY_PENDING');
    expect(calls.some((call) => call.method === 'POST')).toBe(false);
    expect(calls.some((call) => call.method === 'DELETE')).toBe(true);
    const attempts = await listManagedOperationAttempts(env.BROKER_DB, operationId);
    expect(attempts).toHaveLength(1);
    expect(attempts[0]).toEqual(
      expect.objectContaining({ attempt_index: 1, outcome: 'cleaned' }),
    );
  });

  it('repairs a missing attempt row during reconciliation without gaps', async () => {
    const env = createTestBrokerEnv();
    const { operationId, operation } = await createBoundOperation(env);
    const started = await startManagedOperationAttempt(env.BROKER_DB, operation, NOW);
    expect(started.ok).toBe(true);
    if (!started.ok) {
      return;
    }
    env.__db
      .prepare(`DELETE FROM managed_operation_attempts WHERE operation_id = ?`)
      .run(operationId);
    expect(await listManagedOperationAttempts(env.BROKER_DB, operationId)).toHaveLength(0);
    await markAttemptUnknown(env.BROKER_DB, operationId, 1, NOW);

    mockProviderList({});
    const reconciled = await reconcileUnknownAttempt(
      env.BROKER_DB,
      env.OPENROUTER_MANAGEMENT_API_KEY,
      (await getManagedOperation(env.BROKER_DB, operationId))!,
      NOW,
    );
    expect(reconciled?.state).toBe('RETRY_READY');
    const attempts = await listManagedOperationAttempts(env.BROKER_DB, operationId);
    expect(attempts).toHaveLength(1);
    expect(attempts[0]).toEqual(
      expect.objectContaining({
        attempt_index: 1,
        provider_key_name: providerKeyNameForOperationAttempt(operationId, 'discord', 1),
        outcome: 'cleaned',
      }),
    );
  });

  it('activates from FAILED when the delivery was acknowledged', async () => {
    const env = createTestBrokerEnv();
    const { operationId, operation } = await createBoundOperation(env);
    const started = await startManagedOperationAttempt(env.BROKER_DB, operation, NOW);
    expect(started.ok).toBe(true);
    if (!started.ok) {
      return;
    }
    await recordAttemptCredential(env.BROKER_DB, operationId, 1, 'hash_failed_ack_1', NOW);
    const { createManagedKeyDelivery, markManagedKeyDeliveryAcknowledged } = await import(
      '../src/managed-key-delivery'
    );
    const delivery = await createManagedKeyDelivery(env.BROKER_DB, {
      issueSource: 'discord',
      subjectRef: SUBJECT,
      installationId: INSTALLATION,
      managedCredentialRef: 'hash_failed_ack_1',
      createdAt: NOW,
      expiresAt: new Date(NOW.getTime() + 15 * 60_000),
      operationId,
      attemptIndex: 1,
    });
    await transitionManagedOperation(env.BROKER_DB, operationId, 'DELIVERY_PENDING', NOW);
    await markManagedKeyDeliveryAcknowledged(env.BROKER_DB, {
      deliveryId: delivery.deliveryId,
      acknowledgedAt: NOW,
    });
    await transitionManagedOperation(env.BROKER_DB, operationId, 'FAILED', NOW, {
      failureReason: 'terminal_provider_failure',
    });

    expect(await markOperationActiveOnAck(env.BROKER_DB, delivery.deliveryId, NOW)).toBe(true);
    const active = (await getManagedOperation(env.BROKER_DB, operationId))!;
    expect(active.state).toBe('ACTIVE');
    expect(active.client_action).toBe('wait');
  });

  it('finds another operation owning a live delivery for the same subject and installation', async () => {
    const env = createTestBrokerEnv();
    const first = await createBoundOperation(env);
    const started = await startManagedOperationAttempt(env.BROKER_DB, first.operation, NOW);
    expect(started.ok).toBe(true);
    if (!started.ok) {
      return;
    }
    const { createManagedKeyDelivery } = await import('../src/managed-key-delivery');
    await createManagedKeyDelivery(env.BROKER_DB, {
      issueSource: 'discord',
      subjectRef: SUBJECT,
      installationId: INSTALLATION,
      managedCredentialRef: 'hash_conflict_live_1',
      createdAt: NOW,
      expiresAt: new Date(NOW.getTime() + 15 * 60_000),
      operationId: first.operationId,
      attemptIndex: 1,
    });
    await transitionManagedOperation(env.BROKER_DB, first.operationId, 'DELIVERY_PENDING', NOW);

    const second = await createBoundOperation(env);
    const conflict = await findConflictingOperationDelivery(env.BROKER_DB, {
      issueSource: 'discord',
      subjectRef: SUBJECT,
      installationId: INSTALLATION,
      excludeOperationId: second.operationId,
    });
    expect(conflict).toMatchObject({
      operationId: first.operationId,
      deliveryStatus: 'pending',
    });
    const self = await findConflictingOperationDelivery(env.BROKER_DB, {
      issueSource: 'discord',
      subjectRef: SUBJECT,
      installationId: INSTALLATION,
      excludeOperationId: first.operationId,
    });
    expect(self).toBeNull();
  });

  it('sweeps expired operations without touching active ones', async () => {
    const env = createTestBrokerEnv();
    const { operationId } = await createBoundOperation(env);
    await transitionManagedOperation(env.BROKER_DB, operationId, 'ACTIVE', NOW);

    const other = await createBoundOperation(env);
    mockProviderList({});
    const result = await sweepStaleManagedOperations(env, new Date(NOW.getTime() + 61 * 60_000));
    expect(result.expired).toBe(1);
    expect((await getManagedOperation(env.BROKER_DB, operationId))?.state).toBe('ACTIVE');
    expect((await getManagedOperation(env.BROKER_DB, other.operationId))?.state).toBe('FAILED');
  });
});
