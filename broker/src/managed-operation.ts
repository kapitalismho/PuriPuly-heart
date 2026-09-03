import type { Context } from 'hono';

import type { BrokerBindings, BrokerEnv } from './contract';
import {
  cleanupManagedChildKey,
  findManagedChildKeyByName,
} from './openrouter-management';

export const MANAGED_OPERATION_ID_PREFIX = 'ph-mop-v1_';
export const MANAGED_OPERATION_RESUME_HASH_PREFIX = 'ph-mop-resume-v1_';
export const MANAGED_OPERATION_AUTH_TTL_MS = 60 * 60_000;
export const MANAGED_OPERATION_DELIVERY_TTL_MS = 15 * 60_000;
export const STALE_CREATING_THRESHOLD_MS = 5 * 60_000;

export const MANAGED_OPERATION_STATES = [
  'AUTHENTICATED',
  'ISSUE_READY',
  'CREATING',
  'CREATE_UNKNOWN',
  'RECONCILING',
  'CLEANUP_REQUIRED',
  'CLEAN',
  'RETRY_READY',
  'DELIVERY_PENDING',
  'ACTIVE',
  'FAILED',
] as const;

export type ManagedOperationState = (typeof MANAGED_OPERATION_STATES)[number];

export const MANAGED_OPERATION_FAILURE_REASONS = [
  'authorization_expired',
  'terminal_provider_failure',
  'cleanup_failed_terminal',
] as const;

export type ManagedOperationFailureReason = (typeof MANAGED_OPERATION_FAILURE_REASONS)[number];

export type ManagedOperationClientAction =
  | 'wait'
  | 'retry_authorized'
  | 'acknowledge_delivery'
  | 'action_required';

export interface ManagedOperationRecord {
  operation_id: string;
  issue_source: 'discord' | 'qq';
  subject_ref: string;
  installation_id: string | null;
  device_public_key: string | null;
  state: ManagedOperationState;
  attempt_count: number;
  current_attempt_index: number;
  resume_token_hash: string;
  auth_expires_at: string;
  failure_reason: string | null;
  client_action: ManagedOperationClientAction;
  referral_reward_id: number | null;
  referral_status: 'none' | 'reserved' | 'credited' | 'skipped' | 'failed';
  settlement_status: 'none' | 'invitee_pending' | 'referrer_pending' | 'completed';
  hardware_hash: string | null;
  hardware_hash_salt_version: number | null;
  app_version: string | null;
  created_at: string;
  updated_at: string;
  last_reconciled_at: string | null;
  cleanup_attempts: number;
}

export interface ManagedOperationIssuanceContext {
  hardwareHash: string;
  hardwareHashSaltVersion: number;
  appVersion: string;
}

export async function saveOperationIssuanceContext(
  db: D1Database,
  operationId: string,
  context: ManagedOperationIssuanceContext,
  now: Date,
): Promise<void> {
  await db
    .prepare(
      `UPDATE managed_operations
          SET hardware_hash = ?,
              hardware_hash_salt_version = ?,
              app_version = ?,
              updated_at = ?
        WHERE operation_id = ?`,
    )
    .bind(
      context.hardwareHash,
      context.hardwareHashSaltVersion,
      context.appVersion,
      now.toISOString(),
      operationId,
    )
    .run();
}

export interface ManagedOperationAttemptRecord {
  id: number;
  operation_id: string;
  attempt_index: number;
  provider_key_name: string;
  managed_credential_ref: string | null;
  outcome: 'created' | 'unknown' | 'cleaned';
  created_at: string;
  updated_at: string;
}

export function buildManagedOperationId(): string {
  return `${MANAGED_OPERATION_ID_PREFIX}${randomBase64Url(24)}`;
}

export function buildManagedOperationResumeToken(): string {
  return randomBase64Url(32);
}

export async function hashManagedOperationResumeToken(resumeToken: string): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(`puripuly-heart:managed-operation-resume:v1\n${resumeToken}`));
  return `${MANAGED_OPERATION_RESUME_HASH_PREFIX}${Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('')}`;
}

export function isManagedOperationId(value: unknown): value is string {
  if (typeof value !== 'string' || !value.startsWith(MANAGED_OPERATION_ID_PREFIX)) {
    return false;
  }
  return isBase64Url(value.slice(MANAGED_OPERATION_ID_PREFIX.length), 24);
}

export function providerKeyNameForOperationAttempt(operationId: string, issueSource: 'discord' | 'qq', attemptIndex: number): string {
  const short = operationId.slice(MANAGED_OPERATION_ID_PREFIX.length).replace(/[^A-Za-z0-9_-]/g, '').slice(0, 16);
  return `puripuly-heart:mop:${issueSource}:${short}:a${attemptIndex}`;
}

export function clientActionForState(state: ManagedOperationState): ManagedOperationClientAction {
  switch (state) {
    case 'RETRY_READY':
      return 'retry_authorized';
    case 'DELIVERY_PENDING':
      return 'acknowledge_delivery';
    case 'FAILED':
      return 'action_required';
    case 'ACTIVE':
    default:
      return 'wait';
  }
}

export async function createManagedOperation(
  db: D1Database,
  input: {
    operationId: string;
    resumeTokenHash: string;
    issueSource: 'discord' | 'qq';
    subjectRef: string;
    installationId: string | null;
    devicePublicKey: string | null;
    now: Date;
  },
): Promise<{ created: boolean; operation: ManagedOperationRecord }> {
  const nowIso = input.now.toISOString();
  const authExpiresAt = new Date(input.now.getTime() + MANAGED_OPERATION_AUTH_TTL_MS).toISOString();
  const existing = await getManagedOperation(db, input.operationId);
  if (existing) {
    return { created: false, operation: existing };
  }
  await db
    .prepare(
      `INSERT OR IGNORE INTO managed_operations (
          operation_id, issue_source, subject_ref, installation_id, device_public_key,
          state, attempt_count, current_attempt_index, resume_token_hash, auth_expires_at,
          failure_reason, client_action, referral_reward_id, referral_status, settlement_status,
          created_at, updated_at, last_reconciled_at, cleanup_attempts
        ) VALUES (?, ?, ?, ?, ?, 'AUTHENTICATED', 0, 0, ?, ?, NULL, 'wait', NULL, 'none', 'none', ?, ?, NULL, 0)`,
    )
    .bind(
      input.operationId, input.issueSource, input.subjectRef, input.installationId, input.devicePublicKey,
      input.resumeTokenHash, authExpiresAt, nowIso, nowIso,
    )
    .run();
  const operation = await getManagedOperation(db, input.operationId);
  if (!operation) {
    throw new Error('managed operation insert failed');
  }
  return { created: true, operation };
}

export async function getManagedOperation(db: D1Database, operationId: string): Promise<ManagedOperationRecord | null> {
  return db
    .prepare(`SELECT * FROM managed_operations WHERE operation_id = ?`)
    .bind(operationId)
    .first<ManagedOperationRecord>();
}

export async function authenticateManagedOperationRequest(
  db: D1Database,
  input: { operationId: string; resumeToken: string; installationId: string | null },
): Promise<{ ok: true; operation: ManagedOperationRecord } | { ok: false; reason: 'invalid' }> {
  const operation = await getManagedOperation(db, input.operationId);
  if (!operation) {
    return { ok: false, reason: 'invalid' };
  }
  const candidateHash = await hashManagedOperationResumeToken(input.resumeToken);
  if (!(await timingSafeEqual(operation.resume_token_hash, candidateHash))) {
    return { ok: false, reason: 'invalid' };
  }
  if (operation.installation_id !== null && input.installationId !== operation.installation_id) {
    return { ok: false, reason: 'invalid' };
  }
  return { ok: true, operation };
}

export async function authorizeManagedOperationRequest(
  db: D1Database,
  input: { operationId: string; resumeToken: string; installationId: string | null; now: Date },
): Promise<{ ok: true; operation: ManagedOperationRecord } | { ok: false; reason: 'invalid' | 'expired' }> {
  const authenticated = await authenticateManagedOperationRequest(db, input);
  if (!authenticated.ok) {
    return authenticated;
  }
  const operation = authenticated.operation;
  if (operation.state === 'FAILED') {
    return operation.failure_reason === 'authorization_expired' ? { ok: false, reason: 'expired' } : { ok: false, reason: 'invalid' };
  }
  if (input.now.toISOString() >= operation.auth_expires_at) {
    if (operation.state !== 'ACTIVE') {
      await expireManagedOperation(db, operation, input.now);
    }
    return { ok: false, reason: 'expired' };
  }
  return { ok: true, operation: (await getManagedOperation(db, input.operationId)) ?? operation };
}

async function hasAcknowledgedOperationDelivery(db: D1Database, operationId: string): Promise<boolean> {
  const row = await db
    .prepare(`SELECT 1 AS found FROM managed_key_deliveries WHERE operation_id = ? AND status = 'acknowledged' LIMIT 1`)
    .bind(operationId)
    .first<{ found: number }>()
    .catch(() => null);
  return Number(row?.found ?? 0) === 1;
}

export async function failManagedOperationTerminal(
  db: D1Database,
  operation: ManagedOperationRecord,
  now: Date,
  reason: Extract<ManagedOperationFailureReason, 'terminal_provider_failure' | 'cleanup_failed_terminal'>,
): Promise<ManagedOperationRecord | null> {
  const nowIso = now.toISOString();
  await transitionManagedOperation(db, operation.operation_id, 'FAILED', now, { failureReason: reason });
  const current = await getManagedOperation(db, operation.operation_id);
  const onboardingComplete = await hasAcknowledgedOperationDelivery(db, operation.operation_id);
  if (current && current.referral_status === 'reserved' && !onboardingComplete) {
    await db
      .prepare(
        `UPDATE referral_rewards
            SET referred_bonus_status = 'failed', referrer_bonus_status = 'failed', failure_reason = ?, updated_at = ?
          WHERE referred_bonus_status = 'reserved'
            AND (id = ? OR operation_id = ?)`,
      )
      .bind(reason, nowIso, current.referral_reward_id, operation.operation_id)
      .run();
    await db
      .prepare(`UPDATE managed_operations SET referral_status = 'failed', updated_at = ? WHERE operation_id = ?`)
      .bind(nowIso, operation.operation_id)
      .run();
  }
  logManagedOperationEvent('managed_operation_failed', {
    operation_id: operation.operation_id,
    issue_source: operation.issue_source,
    failure_reason: reason,
  });
  return getManagedOperation(db, operation.operation_id);
}

export async function expireManagedOperation(db: D1Database, operation: ManagedOperationRecord, now: Date): Promise<void> {
  const nowIso = now.toISOString();
  logManagedOperationEvent('managed_operation_expired', {
    operation_id: operation.operation_id,
    issue_source: operation.issue_source,
    state: operation.state,
    failure_reason: 'authorization_expired',
  });
  await db
    .prepare(
      `UPDATE managed_operations SET state = 'FAILED', failure_reason = 'authorization_expired', client_action = 'action_required', updated_at = ? WHERE operation_id = ? AND state <> 'ACTIVE' AND state <> 'FAILED'`,
    )
    .bind(nowIso, operation.operation_id)
    .run();
  const current = await getManagedOperation(db, operation.operation_id);
  const acknowledged = await hasAcknowledgedOperationDelivery(db, operation.operation_id);
  if (current && current.referral_status === 'reserved' && !acknowledged) {
    await db
      .prepare(
        `UPDATE referral_rewards
            SET referred_bonus_status = 'failed', referrer_bonus_status = 'failed', failure_reason = 'authorization_expired', updated_at = ?
          WHERE referred_bonus_status = 'reserved'
            AND (id = ? OR operation_id = ?)`,
      )
      .bind(nowIso, current.referral_reward_id, operation.operation_id)
      .run();
    await db
      .prepare(`UPDATE managed_operations SET referral_status = 'failed', updated_at = ? WHERE operation_id = ?`)
      .bind(nowIso, operation.operation_id)
      .run();
  }
}

export async function transitionManagedOperation(
  db: D1Database,
  operationId: string,
  to: ManagedOperationState,
  now: Date,
  options: { failureReason?: ManagedOperationFailureReason | null; from?: ManagedOperationState[] } = {},
): Promise<ManagedOperationRecord | null> {
  const nowIso = now.toISOString();
  const clientAction = clientActionForState(to);
  let sql = `UPDATE managed_operations SET state = ?, client_action = ?, updated_at = ?`;
  const binds: unknown[] = [to, clientAction, nowIso];
  if (options.failureReason !== undefined) {
    sql += `, failure_reason = ?`;
    binds.push(options.failureReason);
  }
  sql += ` WHERE operation_id = ?`;
  binds.push(operationId);
  if (options.from && options.from.length > 0) {
    sql += ` AND state IN (${options.from.map(() => '?').join(', ')})`;
    binds.push(...options.from);
  }
  await db.prepare(sql).bind(...(binds as string[])).run();
  return getManagedOperation(db, operationId);
}
export async function transitionOperationToPostCreateState(
  db: D1Database,
  managementApiKey: string,
  operationId: string,
  to: 'DELIVERY_PENDING' | 'ACTIVE',
  now: Date,
  fetchImpl?: typeof fetch,
): Promise<ManagedOperationRecord | null> {
  await transitionManagedOperation(db, operationId, to, now, { from: ['CREATING'] });
  const settled = await getManagedOperation(db, operationId);
  if (settled && settled.state === to) {
    return settled;
  }
  const anchor = (await getManagedOperation(db, operationId)) ?? settled;
  if (anchor) {
    await reconcileUnknownAttempt(db, managementApiKey, anchor, now, fetchImpl);
  }
  return getManagedOperation(db, operationId);
}
export interface ConflictingOperationDelivery {
  operationId: string;
  deliveryId: string;
  deliveryStatus: string;
  operationState: ManagedOperationState;
}

export async function findConflictingOperationDelivery(
  db: D1Database,
  input: {
    issueSource: 'discord' | 'qq';
    subjectRef: string;
    installationId: string | null;
    excludeOperationId: string | null;
  },
): Promise<ConflictingOperationDelivery | null> {
  const row = await db
    .prepare(
      `SELECT delivery.operation_id AS operationId, delivery.delivery_id AS deliveryId,
              delivery.status AS deliveryStatus, operation.state AS operationState
         FROM managed_key_deliveries AS delivery
         JOIN managed_operations AS operation ON operation.operation_id = delivery.operation_id
        WHERE operation.issue_source = ?
          AND operation.subject_ref = ?
          AND (? IS NULL OR operation.installation_id IS NULL OR operation.installation_id = ?)
          AND delivery.status IN ('pending', 'acknowledged')
          AND operation.state <> 'FAILED'
          AND (? IS NULL OR delivery.operation_id <> ?)
        ORDER BY delivery.created_at DESC, delivery.delivery_id DESC
        LIMIT 1`,
    )
    .bind(
      input.issueSource,
      input.subjectRef,
      input.installationId,
      input.installationId,
      input.excludeOperationId,
      input.excludeOperationId,
    )
    .first<ConflictingOperationDelivery>()
    .catch(() => null);
  return row;
}

export async function hasOtherLiveOperation(
  db: D1Database,
  input: {
    issueSource: 'discord' | 'qq';
    subjectRef: string;
    installationId: string | null;
    excludeOperationId: string | null;
  },
): Promise<boolean> {
  return (await findConflictingOperationDelivery(db, input)) !== null;
}
export async function startManagedOperationAttempt(
  db: D1Database,
  operation: ManagedOperationRecord,
  now: Date,
): Promise<{ ok: true; attempt: ManagedOperationAttemptRecord } | { ok: false; reason: 'not_retry_ready' }> {
  if (
    operation.state !== 'AUTHENTICATED' &&
    operation.state !== 'ISSUE_READY' &&
    operation.state !== 'RETRY_READY'
  ) {
    return { ok: false, reason: 'not_retry_ready' };
  }
  const nextIndex = operation.attempt_count + 1;
  const providerKeyName = providerKeyNameForOperationAttempt(operation.operation_id, operation.issue_source, nextIndex);
  const nowIso = now.toISOString();
  const nextState: ManagedOperationState = 'CREATING';
  let claimed = false;
  try {
    const batchResult = await db.batch([
      db
        .prepare(`UPDATE managed_operations SET state = ?, client_action = ?, attempt_count = ?, current_attempt_index = ?, updated_at = ? WHERE operation_id = ? AND state IN ('AUTHENTICATED', 'ISSUE_READY', 'RETRY_READY') AND attempt_count = ?`)
        .bind(nextState, clientActionForState(nextState), nextIndex, nextIndex, nowIso, operation.operation_id, operation.attempt_count),
      db
        .prepare(
          `INSERT INTO managed_operation_attempts (operation_id, attempt_index, provider_key_name, managed_credential_ref, outcome, created_at, updated_at)
           SELECT ?, ?, ?, NULL, 'unknown', ?, ?
            WHERE EXISTS (
              SELECT 1 FROM managed_operations
               WHERE operation_id = ?
                 AND state = 'CREATING'
                 AND attempt_count = ?
                 AND current_attempt_index = ?
            )
            AND NOT EXISTS (
              SELECT 1 FROM managed_operation_attempts WHERE operation_id = ? AND attempt_index = ?
            )`,
        )
        .bind(
          operation.operation_id, nextIndex, providerKeyName, nowIso, nowIso,
          operation.operation_id, nextIndex, nextIndex,
          operation.operation_id, nextIndex,
        ),
    ]);
    const claimedChanges = Number(batchResult[0]?.meta?.changes ?? 0);
    claimed = claimedChanges === 1;
  } catch {
    claimed = false;
  }
  if (!claimed) {
    return { ok: false, reason: 'not_retry_ready' };
  }
  const attempt = await db
    .prepare(`SELECT * FROM managed_operation_attempts WHERE operation_id = ? AND attempt_index = ?`)
    .bind(operation.operation_id, nextIndex)
    .first<ManagedOperationAttemptRecord>();
  if (!attempt) {
    return { ok: false, reason: 'not_retry_ready' };
  }
  return { ok: true, attempt };
}

export async function recordAttemptCredential(
  db: D1Database,
  operationId: string,
  attemptIndex: number,
  managedCredentialRef: string,
  now: Date,
): Promise<void> {
  await db
    .prepare(`UPDATE managed_operation_attempts SET managed_credential_ref = ?, outcome = 'created', updated_at = ? WHERE operation_id = ? AND attempt_index = ?`)
    .bind(managedCredentialRef, now.toISOString(), operationId, attemptIndex)
    .run();
}

export async function markAttemptUnknown(db: D1Database, operationId: string, attemptIndex: number, now: Date): Promise<void> {
  const nowIso = now.toISOString();
  await db
    .prepare(`UPDATE managed_operation_attempts SET outcome = 'unknown', updated_at = ? WHERE operation_id = ? AND attempt_index = ?`)
    .bind(nowIso, operationId, attemptIndex)
    .run();
  await transitionManagedOperation(db, operationId, 'CREATE_UNKNOWN', now, { from: ['CREATING'] });
}

export async function markAttemptCleaned(db: D1Database, operationId: string, attemptIndex: number, now: Date): Promise<void> {
  await db
    .prepare(`UPDATE managed_operation_attempts SET outcome = 'cleaned', updated_at = ? WHERE operation_id = ? AND attempt_index = ?`)
    .bind(now.toISOString(), operationId, attemptIndex)
    .run();
}

export async function listManagedOperationAttempts(db: D1Database, operationId: string): Promise<ManagedOperationAttemptRecord[]> {
  const result = await db
    .prepare(`SELECT * FROM managed_operation_attempts WHERE operation_id = ? ORDER BY attempt_index ASC`)
    .bind(operationId)
    .all<ManagedOperationAttemptRecord>();
  return result.results ?? [];
}

export async function attachReferralToOperation(
  db: D1Database,
  operationId: string,
  referralRewardId: number | null,
  status: ManagedOperationRecord['referral_status'],
  settlement: ManagedOperationRecord['settlement_status'],
  now: Date,
): Promise<void> {
  await db
    .prepare(`UPDATE managed_operations SET referral_reward_id = COALESCE(referral_reward_id, ?), referral_status = ?, settlement_status = ?, updated_at = ? WHERE operation_id = ?`)
    .bind(referralRewardId, status, settlement, now.toISOString(), operationId)
    .run();
}

export async function reconcileUnknownAttempt(
  db: D1Database,
  managementApiKey: string,
  operation: ManagedOperationRecord,
  now: Date,
  fetchImpl?: typeof fetch,
): Promise<ManagedOperationRecord | null> {
  const current = (await getManagedOperation(db, operation.operation_id)) ?? operation;
  const attempts = await listManagedOperationAttempts(db, operation.operation_id);
  const storedTarget = attempts.find((entry) => entry.attempt_index === current.current_attempt_index) ?? null;
  if (!storedTarget && attempts.length === 0 && current.current_attempt_index === 0) {
    return getManagedOperation(db, operation.operation_id);
  }
  let target = storedTarget ?? {
    attempt_index: current.current_attempt_index,
    provider_key_name: providerKeyNameForOperationAttempt(
      operation.operation_id,
      current.issue_source,
      current.current_attempt_index,
    ),
    managed_credential_ref: null as string | null,
  };
  if (!storedTarget) {
    const repaired = await db
      .prepare(
        `INSERT OR IGNORE INTO managed_operation_attempts (operation_id, attempt_index, provider_key_name, managed_credential_ref, outcome, created_at, updated_at) VALUES (?, ?, ?, NULL, 'unknown', ?, ?)`,
      )
      .bind(operation.operation_id, target.attempt_index, target.provider_key_name, now.toISOString(), now.toISOString())
      .run()
      .catch(() => null);
    if (repaired && Number(repaired.meta?.changes ?? 0) === 1) {
      const reread = await db
        .prepare(`SELECT * FROM managed_operation_attempts WHERE operation_id = ? AND attempt_index = ?`)
        .bind(operation.operation_id, target.attempt_index)
        .first<ManagedOperationAttemptRecord>()
        .catch(() => null);
      if (reread) {
        target = reread;
      }
    }
  }
  return reconcileAttemptTarget(db, managementApiKey, current, target, now, fetchImpl);
}

async function reconcileAttemptTarget(
  db: D1Database,
  managementApiKey: string,
  operation: ManagedOperationRecord,
  target: Pick<ManagedOperationAttemptRecord, 'attempt_index' | 'provider_key_name' | 'managed_credential_ref'>,
  now: Date,
  fetchImpl?: typeof fetch,
): Promise<ManagedOperationRecord | null> {
  await transitionManagedOperation(db, operation.operation_id, 'RECONCILING', now, { from: ['CREATE_UNKNOWN', 'RECONCILING', 'CLEANUP_REQUIRED', 'CLEAN'] });
  const found = await findManagedChildKeyByName({ managementApiKey, keyName: target.provider_key_name, fetchImpl });
  if (!found.found) {
    const credentialRef = target.managed_credential_ref ?? null;
    if (credentialRef) {
      await transitionManagedOperation(db, operation.operation_id, 'CLEANUP_REQUIRED', now, { from: ['RECONCILING', 'CREATE_UNKNOWN', 'CLEANUP_REQUIRED'] });
      const cleanup = await cleanupManagedChildKey({ managementApiKey, keyHash: credentialRef, fetchImpl });
      if (!cleanup.ok) {
        return getManagedOperation(db, operation.operation_id);
      }
      const verify = await findManagedChildKeyByName({ managementApiKey, keyName: target.provider_key_name, fetchImpl });
      if (verify.found) {
        return getManagedOperation(db, operation.operation_id);
      }
    }
    await markAttemptCleaned(db, operation.operation_id, target.attempt_index, now);
    await transitionManagedOperation(db, operation.operation_id, 'CLEAN', now, { from: ['RECONCILING', 'CREATE_UNKNOWN', 'CLEANUP_REQUIRED', 'CLEAN'] });
    await transitionManagedOperation(db, operation.operation_id, 'RETRY_READY', now, { from: ['CLEAN'] });
    return getManagedOperation(db, operation.operation_id);
  }
  await transitionManagedOperation(db, operation.operation_id, 'CLEANUP_REQUIRED', now, { from: ['RECONCILING', 'CREATE_UNKNOWN', 'CLEANUP_REQUIRED'] });
  const credentialRef = target.managed_credential_ref ?? found.keyHash ?? null;
  const cleanupKeyHash = credentialRef ?? found.keyHash;
  if (!cleanupKeyHash) {
    await transitionManagedOperation(db, operation.operation_id, 'CLEANUP_REQUIRED', now);
    return getManagedOperation(db, operation.operation_id);
  }
  const cleanup = await cleanupManagedChildKey({ managementApiKey, keyHash: cleanupKeyHash, fetchImpl });
  if (!cleanup.ok) {
    const next = (operation.cleanup_attempts ?? 0) + 1;
    await db.prepare(`UPDATE managed_operations SET cleanup_attempts = ?, last_reconciled_at = ?, updated_at = ? WHERE operation_id = ?`).bind(next, now.toISOString(), now.toISOString(), operation.operation_id).run();
    if (next >= 5) {
      await transitionManagedOperation(db, operation.operation_id, 'FAILED', now, { failureReason: 'cleanup_failed_terminal' });
    }
    return getManagedOperation(db, operation.operation_id);
  }
  const verify = await findManagedChildKeyByName({ managementApiKey, keyName: target.provider_key_name, fetchImpl });
  if (verify.found) {
    return getManagedOperation(db, operation.operation_id);
  }
  await markAttemptCleaned(db, operation.operation_id, target.attempt_index, now);
  await transitionManagedOperation(db, operation.operation_id, 'CLEAN', now, { from: ['CLEANUP_REQUIRED', 'RECONCILING'] });
  await transitionManagedOperation(db, operation.operation_id, 'RETRY_READY', now, { from: ['CLEAN'] });
  return getManagedOperation(db, operation.operation_id);
}

function logManagedOperationEvent(
  event: string,
  fields: Record<string, string | number | null>,
): void {
  console.info(event, { ...fields, broker_timestamp: new Date().toISOString() });
}

export function describeManagedOperationForLog(operation: ManagedOperationRecord): Record<string, string | number | null> {
  return {
    operation_id: operation.operation_id,
    issue_source: operation.issue_source,
    state: operation.state,
    attempt_count: operation.attempt_count,
    current_attempt_index: operation.current_attempt_index,
    failure_reason: operation.failure_reason,
    referral_status: operation.referral_status,
    settlement_status: operation.settlement_status,
  };
}

export async function sweepStaleManagedOperations(
  env: Pick<BrokerBindings, 'BROKER_DB' | 'OPENROUTER_MANAGEMENT_API_KEY'>,
  now: Date,
  fetchImpl?: typeof fetch,
): Promise<{ expired: number; reconciled: number; retryReady: number }> {
  const db = env.BROKER_DB;
  let expired = 0;
  let reconciled = 0;
  let retryReady = 0;
  const stale = await db
    .prepare(
      `SELECT * FROM managed_operations WHERE state NOT IN ('ACTIVE', 'FAILED') AND auth_expires_at <= ? LIMIT 50`,
    )
    .bind(now.toISOString())
    .all<ManagedOperationRecord>();
  for (const operation of stale.results ?? []) {
    await expireManagedOperation(db, operation, now);
    expired += 1;
  }
  const creatingStale = await db
    .prepare(`SELECT * FROM managed_operations WHERE state = 'CREATING' AND updated_at <= ? LIMIT 25`)
    .bind(new Date(now.getTime() - STALE_CREATING_THRESHOLD_MS).toISOString())
    .all<ManagedOperationRecord>();
  for (const operation of creatingStale.results ?? []) {
    await transitionManagedOperation(db, operation.operation_id, 'CREATE_UNKNOWN', now, { from: ['CREATING'] });
    reconciled += 1;
  }
  const unknown = await db
    .prepare(`SELECT * FROM managed_operations WHERE state IN ('CREATE_UNKNOWN', 'RECONCILING', 'CLEANUP_REQUIRED', 'CLEAN') LIMIT 25`)
    .all<ManagedOperationRecord>();
  for (const operation of unknown.results ?? []) {
    const before = operation.state;
    const after = await reconcileUnknownAttempt(db, env.OPENROUTER_MANAGEMENT_API_KEY, operation, now, fetchImpl);
    if (after && after.state !== before) {
      reconciled += 1;
    }
    if (after && after.state === 'RETRY_READY') {
      retryReady += 1;
    }
  }
  const deliveryStale = await db
    .prepare(`SELECT * FROM managed_operations WHERE state = 'DELIVERY_PENDING' AND updated_at <= ? LIMIT 25`)
    .bind(new Date(now.getTime() - MANAGED_OPERATION_DELIVERY_TTL_MS).toISOString())
    .all<ManagedOperationRecord>();
  for (const operation of deliveryStale.results ?? []) {
    const acknowledged = await db
      .prepare(
        `SELECT delivery_id FROM managed_key_deliveries
          WHERE operation_id = ? AND status = 'acknowledged'
          ORDER BY created_at DESC, delivery_id DESC
          LIMIT 1`,
      )
      .bind(operation.operation_id)
      .first<{ delivery_id: string }>()
      .catch(() => null);
    if (acknowledged) {
      await transitionManagedOperation(db, operation.operation_id, 'ACTIVE', now, { from: ['DELIVERY_PENDING'] });
      reconciled += 1;
      continue;
    }
    const attempts = await listManagedOperationAttempts(db, operation.operation_id);
    const current = attempts.find((entry) => entry.attempt_index === operation.current_attempt_index) ?? null;
    if (!current || !current.managed_credential_ref) {
      await transitionManagedOperation(db, operation.operation_id, 'CREATE_UNKNOWN', now, { from: ['DELIVERY_PENDING'] });
      reconciled += 1;
      continue;
    }
    const cleanup = await cleanupManagedChildKey({ managementApiKey: env.OPENROUTER_MANAGEMENT_API_KEY, keyHash: current.managed_credential_ref, fetchImpl });
    if (cleanup.ok) {
      await markAttemptCleaned(db, operation.operation_id, current.attempt_index, now);
      await transitionManagedOperation(db, operation.operation_id, 'CLEAN', now, { from: ['DELIVERY_PENDING'] });
      await transitionManagedOperation(db, operation.operation_id, 'RETRY_READY', now, { from: ['CLEAN'] });
      retryReady += 1;
    } else {
      await transitionManagedOperation(db, operation.operation_id, 'CLEANUP_REQUIRED', now, { from: ['DELIVERY_PENDING'] });
    }
  }
  logManagedOperationEvent('managed_operation_sweep_completed', {
    expired,
    reconciled,
    retry_ready: retryReady,
  });
  return { expired, reconciled, retryReady };
}

export async function buildManagedOperationStatusBodyWithDelivery(
  db: D1Database,
  operation: ManagedOperationRecord,
  attempts: ManagedOperationAttemptRecord[],
): Promise<Record<string, unknown>> {
  const delivery = await db
    .prepare(
      `SELECT delivery_id, status, expires_at, acknowledged_at
         FROM managed_key_deliveries
        WHERE operation_id = ?
        ORDER BY created_at DESC, delivery_id DESC
        LIMIT 1`,
    )
    .bind(operation.operation_id)
    .first<{ delivery_id: string; status: string; expires_at: string; acknowledged_at: string | null }>()
    .catch(() => null);
  return {
    ...buildManagedOperationStatusBody(operation, attempts),
    delivery: delivery
      ? {
          delivery_id: delivery.delivery_id,
          status: delivery.status,
          expires_at: delivery.expires_at,
        }
      : null,
  };
}

export function buildManagedOperationStatusBody(operation: ManagedOperationRecord, attempts: ManagedOperationAttemptRecord[]): Record<string, unknown> {
  return {
    ok: true,
    operation_id: operation.operation_id,
    issue_source: operation.issue_source,
    state: operation.state,
    client_action: operation.client_action,
    failure_reason: operation.failure_reason,
    attempt_count: operation.attempt_count,
    current_attempt_index: operation.current_attempt_index,
    auth_expires_at: operation.auth_expires_at,
    referral: { status: operation.referral_status, settlement: operation.settlement_status },
    attempts: attempts.map((attempt) => ({
      attempt_index: attempt.attempt_index,
      provider_key_name: attempt.provider_key_name,
      managed_credential_ref: attempt.managed_credential_ref,
      outcome: attempt.outcome,
    })),
  };
}

async function timingSafeEqual(left: string, right: string): Promise<boolean> {
  if (left.length !== right.length) {
    return false;
  }
  let diff = 0;
  for (let index = 0; index < left.length; index += 1) {
    diff |= left.charCodeAt(index) ^ right.charCodeAt(index);
  }
  return diff === 0;
}

function randomBase64Url(byteLength: number): string {
  const bytes = crypto.getRandomValues(new Uint8Array(byteLength));
  return encodeBase64Url(bytes);
}

function encodeBase64Url(bytes: Uint8Array): string {
  let binary = '';
  for (const byte of bytes) {
    binary += String.fromCharCode(byte);
  }
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/g, '');
}

function isBase64Url(value: string, byteLength?: number): boolean {
  if (!/^[A-Za-z0-9_-]*$/u.test(value)) {
    return false;
  }
  if (byteLength !== undefined) {
    const expected = Math.ceil((byteLength * 8) / 6);
    if (value.length !== expected && value.length !== expected - 1) {
      return false;
    }
  }
  return value.length > 0;
}

export async function handleManagedOperationStatus(c: Context<BrokerEnv>): Promise<Response> {
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
  const authenticated = await authenticateManagedOperationRequest(c.env.BROKER_DB, { operationId, resumeToken, installationId });
  if (!authenticated.ok) {
    return c.json({ ok: false, code: 'invalid_request', message: 'unknown operation' }, 404);
  }
  if (authenticated.operation.state === 'FAILED') {
    const attempts = await listManagedOperationAttempts(c.env.BROKER_DB, operationId);
    return c.json(await buildManagedOperationStatusBodyWithDelivery(c.env.BROKER_DB, authenticated.operation, attempts));
  }
  const auth = await authorizeManagedOperationRequest(c.env.BROKER_DB, { operationId, resumeToken, installationId, now: new Date() });
  if (!auth.ok) {
    if (auth.reason === 'expired') {
      const terminal = await getManagedOperation(c.env.BROKER_DB, operationId);
      if (terminal) {
        const attempts = await listManagedOperationAttempts(c.env.BROKER_DB, operationId);
        return c.json(await buildManagedOperationStatusBodyWithDelivery(c.env.BROKER_DB, terminal, attempts), 410);
      }
      return c.json({ ok: false, code: 'authorization_expired', message: 'operation authorization expired' }, 410);
    }
    return c.json({ ok: false, code: 'invalid_request', message: 'unknown operation' }, 404);
  }
  const attempts = await listManagedOperationAttempts(c.env.BROKER_DB, operationId);
  return c.json(await buildManagedOperationStatusBodyWithDelivery(c.env.BROKER_DB, auth.operation, attempts));
}


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

export type OperationIssueBinding =
  | { status: 'proceed'; operation: ManagedOperationRecord; created: boolean }
  | { status: 'wait' | 'done' | 'failed'; operation: ManagedOperationRecord }
  | { status: 'invalid'; reason: 'malformed' | 'unknown' | 'binding_mismatch' | 'expired' };

export async function bindOperationForIssue(
  db: D1Database,
  input: {
    operationId: string | null;
    resumeToken: string | null;
    issueSource: 'discord' | 'qq';
    subjectRef: string;
    installationId: string | null;
    devicePublicKey: string | null;
    now: Date;
  },
): Promise<OperationIssueBinding> {
  if (!input.operationId && !input.resumeToken) {
    return { status: 'invalid', reason: 'malformed' };
  }
  if (!input.operationId || !isManagedOperationId(input.operationId) || !input.resumeToken) {
    return { status: 'invalid', reason: 'malformed' };
  }
  const existing = await getManagedOperation(db, input.operationId);
  if (!existing) {
    const resumeTokenHash = await hashManagedOperationResumeToken(input.resumeToken);
    const created = await createManagedOperation(db, {
      operationId: input.operationId,
      resumeTokenHash,
      issueSource: input.issueSource,
      subjectRef: input.subjectRef,
      installationId: input.installationId,
      devicePublicKey: input.devicePublicKey,
      now: input.now,
    });
    return { status: 'proceed', operation: created.operation, created: true };
  }
  if (
    existing.issue_source !== input.issueSource ||
    existing.subject_ref !== input.subjectRef ||
    (existing.installation_id !== null && input.installationId !== existing.installation_id) ||
    (existing.device_public_key !== null && input.devicePublicKey !== existing.device_public_key)
  ) {
    return { status: 'invalid', reason: 'binding_mismatch' };
  }
  const auth = await authorizeManagedOperationRequest(db, {
    operationId: input.operationId,
    resumeToken: input.resumeToken,
    installationId: input.installationId,
    now: input.now,
  });
  if (!auth.ok) {
    return { status: 'invalid', reason: auth.reason === 'expired' ? 'expired' : 'unknown' };
  }
  const operation = auth.operation;
  if (operation.state === 'FAILED') {
    return { status: 'failed', operation };
  }
  if (operation.state === 'ACTIVE') {
    return { status: 'done', operation };
  }
  if (operation.state === 'AUTHENTICATED') {
    await transitionManagedOperation(db, operation.operation_id, 'ISSUE_READY', input.now, {
      from: ['AUTHENTICATED'],
    });
    const ready = (await getManagedOperation(db, operation.operation_id)) ?? operation;
    return { status: 'proceed', operation: ready, created: false };
  }
  if (operation.state === 'ISSUE_READY' || operation.state === 'RETRY_READY') {
    return { status: 'proceed', operation, created: false };
  }
  return { status: 'wait', operation };
}

export function operationBindingResponseBody(operation: ManagedOperationRecord, attempts: ManagedOperationAttemptRecord[]): Record<string, unknown> {
  return buildManagedOperationStatusBody(operation, attempts);
}

export async function getManagedOperationStatusSnapshot(
  db: D1Database,
  operationId: string,
): Promise<ManagedOperationRecord | null> {
  return getManagedOperation(db, operationId);
}

export async function markOperationActiveOnAck(
  db: D1Database,
  deliveryId: string,
  now: Date,
): Promise<boolean> {
  const nowIso = now.toISOString();
  const linked = await db
    .prepare(`SELECT operation_id FROM managed_key_deliveries WHERE delivery_id = ?`)
    .bind(deliveryId)
    .first<{ operation_id: string | null }>()
    .catch(() => null);
  if (!linked || !linked.operation_id) {
    return true;
  }
  await db
    .prepare(
      `UPDATE managed_operations
          SET state = 'ACTIVE', client_action = 'wait', failure_reason = NULL, updated_at = ?
        WHERE operation_id = (SELECT operation_id FROM managed_key_deliveries WHERE delivery_id = ? AND status = 'acknowledged')
          AND operation_id IS NOT NULL
          AND state IN ('DELIVERY_PENDING', 'FAILED')`,
    )
    .bind(nowIso, deliveryId)
    .run();
  await db
    .prepare(
      `UPDATE managed_operations
          SET settlement_status = 'invitee_pending', updated_at = ?
        WHERE operation_id = (SELECT operation_id FROM managed_key_deliveries WHERE delivery_id = ? AND status = 'acknowledged')
          AND operation_id IS NOT NULL
          AND EXISTS (
            SELECT 1 FROM managed_referral_settlement_jobs job
            JOIN managed_key_deliveries delivery ON delivery.delivery_id = job.delivery_id
           WHERE delivery.delivery_id = ?
          )`,
    )
    .bind(nowIso, deliveryId, deliveryId)
    .run();
  const operation = await db
    .prepare(
      `SELECT state FROM managed_operations
        WHERE operation_id = (SELECT operation_id FROM managed_key_deliveries WHERE delivery_id = ?)`,
    )
    .bind(deliveryId)
    .first<{ state: string }>()
    .catch(() => null);
  return operation?.state === 'ACTIVE';
}

export async function markOperationSettlementStatus(
  db: D1Database,
  operationId: string | null,
  input: { referral?: 'credited'; settlement: 'invitee_pending' | 'referrer_pending' | 'completed' },
  now: Date,
): Promise<void> {
  if (!operationId) {
    return;
  }
  const nowIso = now.toISOString();
  if (input.referral === 'credited') {
    await db
      .prepare(
        `UPDATE managed_operations
            SET referral_status = 'credited', settlement_status = ?, updated_at = ?
          WHERE operation_id = ? AND state <> 'FAILED'`,
      )
      .bind(input.settlement, nowIso, operationId)
      .run()
      .catch(() => null);
    return;
  }
  await db
    .prepare(
      `UPDATE managed_operations
          SET settlement_status = ?, updated_at = ?
        WHERE operation_id = ? AND state <> 'FAILED'`,
    )
    .bind(input.settlement, nowIso, operationId)
    .run()
    .catch(() => null);
}
