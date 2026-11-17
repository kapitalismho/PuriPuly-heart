import type { Context } from 'hono';
import { authenticate, hasTokenSecret } from '../../agent-relay/src/auth';
import type { BrokerEnv } from './contract';
import { readJsonBody } from './public-input';
import {
  createManagedOperation, createManagedOperationChildKey, getManagedOperation,
  hashManagedOperationResumeToken, listManagedOperationAttempts, markOperationActiveOnAck,
  reconcileUnknownAttempt, startManagedOperationAttempt, transitionManagedOperation,
  transitionOperationToPostCreateState, type ManagedOperationRecord,
} from './managed-operation';
import {
  createManagedKeyDelivery, markManagedKeyDeliveryAcknowledged, validateManagedKeyDeliveryAck,
} from './managed-key-delivery';
import {
  assignManagedGuardrail, setManagedChildKeyDisabled, verifyManagedPhotoKey, verifyManagedPhotoPolicy,
} from './openrouter-management';

const periodMs = 86_400_000;

function status(operation: ManagedOperationRecord) {
  return {
    operation_id: operation.operation_id, state: operation.state,
    expires_at: new Date(Number(operation.photo_period_start) + periodMs).toISOString(),
    limit_usd: operation.photo_limit_usd, model: 'openai/gpt-6-luna',
    activated: operation.photo_activated_at !== null, revoked: operation.photo_revoked_at !== null,
  };
}

async function revoke(db: D1Database, managementApiKey: string, operation: ManagedOperationRecord, now: Date) {
  const at = now.toISOString();
  await db.batch([
    db.prepare(`UPDATE managed_operations SET photo_revoked_at = COALESCE(photo_revoked_at, ?),
      state = 'CLEANUP_REQUIRED', updated_at = ?, client_action = 'wait'
      WHERE operation_id = ?`).bind(at, at, operation.operation_id),
    db.prepare(`UPDATE managed_key_deliveries SET status = 'expired', acknowledged_at = NULL,
      failed_at = ?, failure_reason = 'photo_access_revoked' WHERE operation_id = ?`).bind(at, operation.operation_id),
  ]);
  const current = await getManagedOperation(db, operation.operation_id);
  if (!current) throw new Error('Photo operation disappeared');
  await reconcileUnknownAttempt(db, managementApiKey, current, now);
  const attempts = await listManagedOperationAttempts(db, operation.operation_id);
  return attempts.every((attempt) => attempt.outcome === 'cleaned' && attempt.create_in_flight === 0);
}

export async function handlePhotoAccess(c: Context<BrokerEnv>): Promise<Response> {
  c.header('Cache-Control', 'no-store');
  const env = c.env;
  if (env.PHOTO_OPENROUTER_ENABLED !== 'true' || !hasTokenSecret(env.PURI_TOKEN_SECRET)
    || !env.OPENROUTER_MANAGEMENT_API_KEY || !env.OPENROUTER_PHOTO_GUARDRAIL_ID) {
    return c.json({ error: { code: 'photo_access_unavailable' } }, 503);
  }
  const subject = await authenticate(c.req.raw, env.PURI_TOKEN_SECRET);
  if (!subject) return c.json({ error: { code: 'invalid_token' } }, 401);
  const body = await readJsonBody(c);
  if (!body.ok || !['get', 'acknowledge', 'revoke'].includes(String(body.value.action))) {
    return c.json({ error: { code: 'invalid_request' } }, 400);
  }
  const fields = body.value.action === 'acknowledge'
    ? ['action', 'operation_id', 'delivery_id', 'managed_credential_ref', 'delivery_ack_token']
    : body.value.action === 'revoke' ? ['action', 'operation_id'] : ['action'];
  if (Object.keys(body.value).length !== fields.length || fields.some((name) => typeof body.value[name] !== 'string')) {
    return c.json({ error: { code: 'invalid_request' } }, 400);
  }
  const db = env.BROKER_DB;
  const now = new Date();
  const period = Math.floor(now.getTime() / periodMs) * periodMs;
  const expiresAt = new Date(period + periodMs).toISOString();
  const policy = { managementApiKey: env.OPENROUTER_MANAGEMENT_API_KEY, guardrailId: env.OPENROUTER_PHOTO_GUARDRAIL_ID };
  if (body.value.action !== 'get') {
    const operation = await getManagedOperation(db, String(body.value.operation_id));
    if (!operation || operation.issue_source !== 'photo' || operation.subject_ref !== subject) {
      return c.json({ error: { code: 'unknown_photo_access' } }, 404);
    }
    if (body.value.action === 'revoke') {
      const revoked = await revoke(db, policy.managementApiKey, operation, now);
      return c.json({ revoked, operation_id: operation.operation_id }, revoked ? 200 : 202);
    }
    if (operation.photo_revoked_at || operation.photo_period_start !== period) {
      return c.json({ error: { code: 'photo_access_expired' } }, 409);
    }
    const validation = await validateManagedKeyDeliveryAck(db, {
      deliveryId: String(body.value.delivery_id), managedCredentialRef: String(body.value.managed_credential_ref),
      deliveryAckToken: String(body.value.delivery_ack_token), now,
    });
    if (!validation.ok || validation.delivery.operation_id !== operation.operation_id
      || validation.delivery.issue_source !== 'photo' || validation.delivery.attempt_index !== operation.current_attempt_index) {
      return c.json({ error: { code: 'invalid_photo_delivery' } }, 409);
    }
    const acknowledged = await markManagedKeyDeliveryAcknowledged(db, {
      deliveryId: validation.delivery.delivery_id, acknowledgedAt: now,
    });
    if (!acknowledged.ok || !await markOperationActiveOnAck(db, validation.delivery.delivery_id, now)) {
      return c.json({ error: { code: 'photo_access_recovery_required' } }, 409);
    }
    const held = await db.prepare(`UPDATE managed_operations SET photo_activated_at = COALESCE(photo_activated_at, ?)
      WHERE operation_id = ? AND state = 'ACTIVE' AND photo_revoked_at IS NULL RETURNING operation_id`)
      .bind(now.toISOString(), operation.operation_id).first();
    if (!held) return c.json({ error: { code: 'photo_access_recovery_required' } }, 409);
    const keyPolicy = { ...policy, keyHash: validation.delivery.managed_credential_ref,
      limitUsd: Number(operation.photo_limit_usd), expiresAt };
    await verifyManagedPhotoKey(keyPolicy);
    await setManagedChildKeyDisabled({ ...keyPolicy, disabled: false });
    await verifyManagedPhotoKey({ ...keyPolicy, disabled: false });
    const live = await getManagedOperation(db, operation.operation_id);
    if (!live || live.photo_revoked_at || live.state !== 'ACTIVE') {
      return c.json({ error: { code: 'photo_access_recovery_required' } }, 409);
    }
    return c.json({ ...status(live), acknowledged: true });
  }
  const previous = await db.prepare(`SELECT * FROM managed_operations WHERE issue_source = 'photo'
    AND subject_ref = ? AND photo_period_start < ? AND EXISTS (
      SELECT 1 FROM managed_operation_attempts attempt WHERE attempt.operation_id = managed_operations.operation_id
      AND (attempt.outcome <> 'cleaned' OR attempt.create_in_flight = 1)) ORDER BY photo_period_start LIMIT 2`)
    .bind(subject, period).all<ManagedOperationRecord>();
  for (const old of previous.results ?? []) {
    if (!await revoke(db, policy.managementApiKey, old, now)) {
      return c.json({ error: { code: 'photo_access_recovery_required' } }, 409);
    }
  }
  const limitUsd = Number(env.PHOTO_OPENROUTER_LIMIT_USD ?? '0.25');
  if (!Number.isFinite(limitUsd) || limitUsd <= 0 || limitUsd > 10) {
    return c.json({ error: { code: 'photo_access_unavailable' } }, 503);
  }
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(`photo:${subject}:${period}`));
  const operationId = `ph-mop-v1_${Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('')}`;
  let operation = (await createManagedOperation(db, {
    operationId, resumeTokenHash: await hashManagedOperationResumeToken(crypto.randomUUID()),
    issueSource: 'photo', subjectRef: subject, installationId: subject, devicePublicKey: null, now,
    photoPeriodStart: period, photoLimitUsd: limitUsd,
  })).operation;
  if (operation.photo_revoked_at || operation.photo_activated_at) {
    return c.json({ ...status(operation), error: { code: 'photo_credential_already_delivered' } }, 409);
  }
  if (['CREATE_UNKNOWN', 'RECONCILING', 'CLEANUP_REQUIRED', 'DELIVERY_PENDING', 'FAILED'].includes(operation.state)) {
    operation = await reconcileUnknownAttempt(db, policy.managementApiKey, operation, now) ?? operation;
  } else if (operation.state === 'CREATING' && now.getTime() - Date.parse(operation.updated_at) > 120_000) {
    operation = await transitionManagedOperation(db, operationId, 'CREATE_UNKNOWN', now, { from: ['CREATING'] }) ?? operation;
    operation = await reconcileUnknownAttempt(db, policy.managementApiKey, operation, now) ?? operation;
  }
  if (!['AUTHENTICATED', 'ISSUE_READY', 'RETRY_READY'].includes(operation.state)) {
    return c.json({ ...status(operation), error: { code: 'photo_access_recovery_required' } }, 409);
  }
  await verifyManagedPhotoPolicy(policy);
  await db.prepare(`UPDATE managed_operations SET auth_expires_at = ? WHERE operation_id = ? AND photo_revoked_at IS NULL`)
    .bind(expiresAt, operationId).run();
  operation = await getManagedOperation(db, operationId) ?? operation;
  const started = await startManagedOperationAttempt(db, operation, now);
  if (!started.ok) return c.json({ error: { code: 'photo_access_busy' } }, 409);
  const attempt = started.attempt;
  try {
    const key = await createManagedOperationChildKey(db, operationId, attempt.attempt_index, {
      managementApiKey: policy.managementApiKey, issueSource: 'photo', subjectRef: subject,
      issueRef: operationId, keyName: attempt.provider_key_name, expiresAt,
      limitUsd: Number(operation.photo_limit_usd), includeByokInLimit: true,
    });
    await setManagedChildKeyDisabled({ ...policy, keyHash: key.hash, disabled: true });
    await assignManagedGuardrail({ ...policy, keyHash: key.hash });
    await verifyManagedPhotoKey({ ...policy, keyHash: key.hash, disabled: true,
      limitUsd: Number(operation.photo_limit_usd), expiresAt });
    const delivery = await createManagedKeyDelivery(db, {
      issueSource: 'photo', subjectRef: subject, installationId: subject, managedCredentialRef: key.hash,
      createdAt: new Date(), expiresAt: new Date(Math.min(Date.now() + 120_000, period + periodMs)),
      operationId, attemptIndex: attempt.attempt_index,
    });
    const pending = await transitionOperationToPostCreateState(db, policy.managementApiKey, operationId,
      attempt.attempt_index, 'DELIVERY_PENDING', new Date());
    if (!pending || pending.state !== 'DELIVERY_PENDING' || pending.photo_revoked_at || Date.now() >= period + periodMs) {
      throw new Error('Photo delivery no longer owns attempt');
    }
    return c.json({ ...status(pending), key: key.rawKey, managed_credential_ref: key.hash,
      delivery_id: delivery.deliveryId, delivery_ack_token: delivery.deliveryAckToken, acknowledged: false });
  } catch {
    await transitionManagedOperation(db, operationId, 'CREATE_UNKNOWN', new Date(), {
      from: ['CREATING'], attemptIndex: attempt.attempt_index,
    });
    return c.json({ error: { code: 'photo_access_recovery_required' }, operation_id: operationId }, 503);
  }
}
