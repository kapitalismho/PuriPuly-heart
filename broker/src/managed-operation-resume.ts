import type { Context } from 'hono';

import type { BrokerEnv } from './contract';
import { executeDiscordResumeIssuance } from './discord-managed-issue';
import {
  authenticateManagedOperationRequest,
  authorizeManagedOperationRequest,
  buildManagedOperationStatusBodyWithDelivery,
  getManagedOperation,
  isManagedOperationId,
  listManagedOperationAttempts,
  markOperationActiveOnAck,
  findConflictingOperationDelivery,
  reconcileUnknownAttempt,
  startManagedOperationAttempt,
  transitionManagedOperation,
  type ManagedOperationAttemptRecord,
  type ManagedOperationRecord,
} from './managed-operation';
import { instrumentPublicPostRoute } from './abuse-controls';
import type { ManagedKeyDeliveryRecord } from './persistence';
import { executeQqResumeIssuance } from './qq-managed-issue';

async function readJsonBody(c: Context<BrokerEnv>): Promise<{ ok: true; value: Record<string, unknown> } | { ok: false }> {
  let parsed: unknown;
  try {
    parsed = await c.req.json();
  } catch {
    return { ok: false };
  }
  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
    return { ok: false };
  }
  return { ok: true, value: parsed as Record<string, unknown> };
}

async function getLatestOperationDelivery(
  db: D1Database,
  operationId: string,
): Promise<ManagedKeyDeliveryRecord | null> {
  return db
    .prepare(
      `SELECT delivery_id, issue_source, subject_ref, installation_id, managed_credential_ref,
              ack_token_hash, status, created_at, expires_at, acknowledged_at, failed_at, failure_reason,
              operation_id, attempt_index
         FROM managed_key_deliveries
        WHERE operation_id = ?
        ORDER BY created_at DESC, delivery_id DESC
        LIMIT 1`,
    )
    .bind(operationId)
    .first<ManagedKeyDeliveryRecord>()
    .catch(() => null);
}

async function resumeStateBody(
  db: D1Database,
  operation: ManagedOperationRecord,
): Promise<Record<string, unknown>> {
  const attempts = await listManagedOperationAttempts(db, operation.operation_id);
  return buildManagedOperationStatusBodyWithDelivery(db, operation, attempts);
}

async function expireOperationDelivery(
  db: D1Database,
  delivery: ManagedKeyDeliveryRecord,
  now: Date,
): Promise<boolean> {
  const result = await db
    .prepare(
      `UPDATE managed_key_deliveries
          SET status = 'expired', failed_at = ?, failure_reason = 'resume_stale_delivery_expired'
        WHERE delivery_id = ?
          AND status = 'pending'`,
    )
    .bind(now.toISOString(), delivery.delivery_id)
    .run();
  return Number(result.meta?.changes ?? 0) === 1;
}

export async function handleManagedOperationResume(c: Context<BrokerEnv>): Promise<Response> {
  const body = await readJsonBody(c);
  if (!body.ok) {
    return c.json({ ok: false, code: 'invalid_request', message: 'request body must be a JSON object' }, 400);
  }
  const operationId = typeof body.value.operation_id === 'string' ? body.value.operation_id : null;
  const resumeToken = typeof body.value.resume_token === 'string' ? body.value.resume_token : null;
  const installationId = typeof body.value.installation_id === 'string' ? body.value.installation_id : null;
  if (!operationId || !isManagedOperationId(operationId) || !resumeToken || !installationId) {
    return c.json({ ok: false, code: 'invalid_request', message: 'operation_id, resume_token, and installation_id are required' }, 400);
  }
  const resumeRateLimit = await instrumentPublicPostRoute(c.env.BROKER_DB, c, {
    endpoint: 'POST /v1/providers/openrouter/managed-operation/resume',
    installationId,
  });
  if (resumeRateLimit) {
    return c.json(
      {
        ok: false,
        code: 'rate_limited',
        message: resumeRateLimit.message,
        retry_after_ms: resumeRateLimit.retryAfterMs,
      },
      429,
    );
  }
  const db = c.env.BROKER_DB;
  const now = new Date();
  const authenticated = await authenticateManagedOperationRequest(db, { operationId, resumeToken, installationId });
  if (!authenticated.ok) {
    return c.json({ ok: false, code: 'invalid_request', message: 'unknown operation' }, 404);
  }
  if (authenticated.operation.state === 'FAILED') {
    return c.json(await resumeStateBody(db, authenticated.operation));
  }
  const auth = await authorizeManagedOperationRequest(db, { operationId, resumeToken, installationId, now });
  if (!auth.ok) {
    if (auth.reason === 'expired') {
      const terminal = await getManagedOperation(db, operationId);
      if (terminal) {
        return c.json(await resumeStateBody(db, terminal), 410);
      }
      return c.json({ ok: false, code: 'authorization_expired', message: 'operation authorization expired' }, 410);
    }
    return c.json({ ok: false, code: 'invalid_request', message: 'unknown operation' }, 404);
  }
  let operation = auth.operation;
  const deliveryConflict = await findConflictingOperationDelivery(db, {
    issueSource: operation.issue_source,
    subjectRef: operation.subject_ref,
    installationId: operation.installation_id,
    excludeOperationId: operation.operation_id,
  });
  if (deliveryConflict) {
    return c.json(await resumeStateBody(db, operation));
  }
  if (
    operation.state === 'CREATE_UNKNOWN' ||
    operation.state === 'RECONCILING' ||
    operation.state === 'CLEANUP_REQUIRED' ||
    operation.state === 'CLEAN'
  ) {
    const reconciled = await reconcileUnknownAttempt(db, c.env.OPENROUTER_MANAGEMENT_API_KEY, operation, now);
    if (reconciled) {
      operation = reconciled;
    }
    if (operation.state !== 'RETRY_READY') {
      return c.json(await resumeStateBody(db, operation));
    }
  }
  let hasLiveDelivery = false;
  if (operation.state === 'DELIVERY_PENDING') {
    const delivery = await getLatestOperationDelivery(db, operation.operation_id);
    if (delivery && delivery.status === 'acknowledged') {
      await markOperationActiveOnAck(db, delivery.delivery_id, now);
      const live = (await getManagedOperation(db, operation.operation_id)) ?? operation;
      return c.json(await resumeStateBody(db, live));
    }
    if (delivery && delivery.status === 'pending' && delivery.expires_at > now.toISOString()) {
      hasLiveDelivery = true;
      const live = (await getManagedOperation(db, operation.operation_id)) ?? operation;
      return c.json(await resumeStateBody(db, live));
    }
    if (delivery && delivery.status === 'pending') {
      await expireOperationDelivery(db, delivery, now);
    }
    await transitionManagedOperation(db, operation.operation_id, 'CREATE_UNKNOWN', now, {
      from: ['DELIVERY_PENDING'],
    });
    const afterStale = await getManagedOperation(db, operation.operation_id);
    if (afterStale) {
      operation = afterStale;
    }
    if (operation.state === 'CREATE_UNKNOWN') {
      const reconciled = await reconcileUnknownAttempt(db, c.env.OPENROUTER_MANAGEMENT_API_KEY, operation, now);
      if (reconciled) {
        operation = reconciled;
      }
    }
    if (operation.state !== 'RETRY_READY') {
      return c.json(await resumeStateBody(db, operation));
    }
  }
  if (operation.state !== 'AUTHENTICATED' && operation.state !== 'ISSUE_READY' && operation.state !== 'RETRY_READY') {
    return c.json(await resumeStateBody(db, operation));
  }
  if (
    operation.issue_source === 'discord' &&
    (!operation.installation_id ||
      !operation.device_public_key ||
      !operation.hardware_hash ||
      operation.hardware_hash_salt_version === null ||
      !operation.app_version)
  ) {
    return c.json(
      {
        ...(await resumeStateBody(db, operation)),
        ok: false,
        code: 'resume_issuance_unavailable',
        message: 'operation predates durable issuance context; start a fresh managed issue',
      },
      409,
    );
  }
  const started = await startManagedOperationAttempt(db, operation, now);
  if (!started.ok) {
    const current = (await getManagedOperation(db, operation.operation_id)) ?? operation;
    return c.json(await resumeStateBody(db, current));
  }
  const attempt: ManagedOperationAttemptRecord = started.attempt;
  const nowIso = now.toISOString();
  if (!hasLiveDelivery) {
    const latest = await getLatestOperationDelivery(db, operation.operation_id);
    hasLiveDelivery = !!latest && latest.status === 'pending' && latest.expires_at > now.toISOString();
  }
  if (operation.issue_source === 'discord') {
    return executeDiscordResumeIssuance(c, {
      operation,
      attemptIndex: attempt.attempt_index,
      hasLiveDelivery,
      now,
      nowIso,
    });
  }
  return executeQqResumeIssuance(c, {
    operation,
    attemptIndex: attempt.attempt_index,
    hasLiveDelivery,
    now,
    nowIso,
  });
}
