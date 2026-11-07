import type { Context } from 'hono';

import { instrumentPublicPostRoute } from './abuse-controls';
import type { BrokerBindings, BrokerEnv } from './contract';
import { readJsonBody } from './public-input';
import {
  createManagedCleanupBudget,
  MANAGED_CLEANUP_CLAIM_TTL_MS,
  MANAGED_CLEANUP_RETRY_MS,
  SCHEDULED_MANAGED_CLEANUP_LIMIT,
  type ManagedCleanupBudget,
} from './managed-cleanup';
import {
  cleanupManagedChildKey,
  createManagedChildKey,
  OpenRouterManagementError,
  type CreateManagedChildKeyInput,
  type ManagedChildKeyMaterial,
  findManagedChildKeyByName,
  isDefinitiveManagedChildKeyCreateRejection,
  verifyManagedChildKeyAbsent,
} from './openrouter-management';

export const MANAGED_OPERATION_ID_PREFIX = 'ph-mop-v1_';
export const MANAGED_OPERATION_RESUME_HASH_PREFIX = 'ph-mop-resume-v1_';
export const MANAGED_OPERATION_AUTH_TTL_MS = 60 * 60_000;

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
  cleanup_token: string | null;
  cleanup_started_at: string | null;
  scan_offset: number;
  create_in_flight: number;
  legacy_cleanup_state: 'unclassified' | 'pending' | 'resolved';
  legacy_cleanup_checked_at: string | null;
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
          failure_reason, client_action,
          created_at, updated_at, last_reconciled_at, cleanup_attempts
        ) VALUES (?, ?, ?, ?, ?, 'AUTHENTICATED', 0, 0, ?, ?, NULL, 'wait', ?, ?, NULL, 0)`,
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

export async function rejectUnboundQqOperation(
  db: D1Database,
  operation: ManagedOperationRecord,
  now: Date,
): Promise<ManagedOperationRecord> {
  if (operation.issue_source === 'qq' && operation.installation_id === null && operation.state !== 'ACTIVE') {
    return (await failManagedOperationTerminal(db, operation, now, 'terminal_provider_failure')) ?? operation;
  }
  return operation;
}


export async function failManagedOperationTerminal(
  db: D1Database,
  operation: ManagedOperationRecord,
  now: Date,
  reason: Extract<ManagedOperationFailureReason, 'terminal_provider_failure' | 'cleanup_failed_terminal'>,
): Promise<ManagedOperationRecord | null> {
  const result = await db.prepare(
    `UPDATE managed_operations SET state = 'FAILED', client_action = 'action_required', failure_reason = ?, updated_at = ?
      WHERE operation_id = ? AND current_attempt_index = ? AND state NOT IN ('ACTIVE', 'FAILED')
       AND NOT EXISTS (SELECT 1 FROM managed_operation_attempts WHERE operation_id = managed_operations.operation_id AND attempt_index = ? AND cleanup_token IS NOT NULL)`,
  ).bind(reason, now.toISOString(), operation.operation_id, operation.current_attempt_index, operation.current_attempt_index).run();
  if (Number(result.meta.changes ?? 0) === 1) {
    logManagedOperationEvent('managed_operation_failed', {
      operation_id: operation.operation_id,
      issue_source: operation.issue_source,
      failure_reason: reason,
    });
  }
  return getManagedOperation(db, operation.operation_id);
}

export async function expireManagedOperation(db: D1Database, operation: ManagedOperationRecord, now: Date): Promise<void> {
  const nowIso = now.toISOString();
  const result = await db
    .prepare(
      `UPDATE managed_operations SET state = 'FAILED', failure_reason = 'authorization_expired', client_action = 'action_required', updated_at = ? WHERE operation_id = ? AND state <> 'ACTIVE' AND state <> 'FAILED'`,
    )
    .bind(nowIso, operation.operation_id)
    .run();
  if (Number(result.meta.changes ?? 0) > 0) {
    logManagedOperationEvent('managed_operation_expired', {
      operation_id: operation.operation_id,
      issue_source: operation.issue_source,
      state: 'FAILED',
      failure_reason: 'authorization_expired',
    });
  }
}

export async function transitionManagedOperation(
  db: D1Database,
  operationId: string,
  to: ManagedOperationState,
  now: Date,
  options: { failureReason?: ManagedOperationFailureReason | null; from?: ManagedOperationState[]; attemptIndex?: number } = {},
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
  if (options.attemptIndex !== undefined) {
    sql += ` AND current_attempt_index = ? AND NOT EXISTS (SELECT 1 FROM managed_operation_attempts WHERE operation_id = managed_operations.operation_id AND attempt_index = ? AND cleanup_token IS NOT NULL)`;
    binds.push(options.attemptIndex, options.attemptIndex);
  }
  await db.prepare(sql).bind(...(binds as string[])).run();
  return getManagedOperation(db, operationId);
}
export async function transitionOperationToPostCreateState(
  db: D1Database,
  managementApiKey: string,
  operationId: string,
  attemptIndex: number,
  to: 'DELIVERY_PENDING' | 'ACTIVE',
  now: Date,
  fetchImpl?: typeof fetch,
): Promise<ManagedOperationRecord | null> {
  await transitionManagedOperation(db, operationId, to, now, { from: ['CREATING'], attemptIndex });
  const settled = await getManagedOperation(db, operationId);
  if (settled && settled.current_attempt_index === attemptIndex && settled.state === to) {
    return settled;
  }
  const anchor = (await getManagedOperation(db, operationId)) ?? settled;
  if (anchor) {
    await reconcileUnknownAttempt(db, managementApiKey, { ...anchor, current_attempt_index: attemptIndex }, now, fetchImpl);
  }
  const current = await getManagedOperation(db, operationId);
  return current?.current_attempt_index === attemptIndex ? current : null;
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
  await classifyLegacyManagedAttempts(db);
  const nextIndex = operation.attempt_count + 1;
  const providerKeyName = providerKeyNameForOperationAttempt(operation.operation_id, operation.issue_source, nextIndex);
  const nowIso = now.toISOString();
  const nextState: ManagedOperationState = 'CREATING';
  let claimed = false;
  try {
    const batchResult = await db.batch([
      db
        .prepare(`UPDATE managed_operations SET state = ?, client_action = ?, attempt_count = ?, current_attempt_index = ?, updated_at = ? WHERE operation_id = ? AND state IN ('AUTHENTICATED', 'ISSUE_READY', 'RETRY_READY') AND attempt_count = ? AND auth_expires_at > ? AND NOT EXISTS (SELECT 1 FROM managed_operation_attempts WHERE operation_id = managed_operations.operation_id AND (outcome <> 'cleaned' OR cleanup_token IS NOT NULL OR create_in_flight = 1 OR legacy_cleanup_state <> 'resolved')) AND NOT EXISTS (SELECT 1 FROM managed_key_deliveries WHERE operation_id = managed_operations.operation_id AND status IN ('pending', 'acknowledged', 'cleanup_required')) AND NOT EXISTS (SELECT 1 FROM managed_operations other JOIN managed_operation_attempts attempt ON attempt.operation_id = other.operation_id WHERE other.operation_id <> managed_operations.operation_id AND other.issue_source = managed_operations.issue_source AND other.subject_ref = managed_operations.subject_ref AND (attempt.outcome <> 'cleaned' OR attempt.cleanup_token IS NOT NULL OR attempt.create_in_flight = 1 OR attempt.legacy_cleanup_state <> 'resolved'))`)
        .bind(nextState, clientActionForState(nextState), nextIndex, nextIndex, nowIso, operation.operation_id, operation.attempt_count, nowIso),
      db
        .prepare(
          `INSERT INTO managed_operation_attempts (operation_id, attempt_index, provider_key_name, managed_credential_ref, outcome, created_at, updated_at, legacy_cleanup_state)
           SELECT ?, ?, ?, NULL, 'unknown', ?, ?, 'resolved'
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

export async function createManagedOperationChildKey(
  db: D1Database,
  operationId: string | null,
  attemptIndex: number | null,
  input: CreateManagedChildKeyInput,
): Promise<ManagedChildKeyMaterial> {
  if (operationId === null || attemptIndex === null) return createManagedChildKey(input);
  const claimed = await db.prepare(
    `UPDATE managed_operation_attempts SET create_in_flight = 1, updated_at = ?
      WHERE operation_id = ? AND attempt_index = ? AND create_in_flight = 0
        AND outcome = 'unknown' AND managed_credential_ref IS NULL AND cleanup_token IS NULL
        AND EXISTS (SELECT 1 FROM managed_operations
          WHERE operation_id = managed_operation_attempts.operation_id
            AND current_attempt_index = managed_operation_attempts.attempt_index AND state = 'CREATING')`,
  ).bind(new Date().toISOString(), operationId, attemptIndex).run();
  if (Number(claimed.meta.changes ?? 0) !== 1) throw new Error('managed create attempt is no longer owned');
  let childKey: ManagedChildKeyMaterial | null = null;
  let definitivelyRejected = false;
  try {
    childKey = await createManagedChildKey(input);
    return childKey;
  } catch (error) {
    if (error instanceof OpenRouterManagementError) childKey = error.createdChildKey;
    definitivelyRejected = isDefinitiveManagedChildKeyCreateRejection(error);
    throw error;
  } finally {
    if (childKey || definitivelyRejected) {
      await db.prepare(
        `UPDATE managed_operation_attempts
            SET create_in_flight = 0, managed_credential_ref = COALESCE(?, managed_credential_ref),
                outcome = CASE WHEN ? IS NULL THEN outcome ELSE 'created' END, updated_at = ?
          WHERE operation_id = ? AND attempt_index = ? AND create_in_flight = 1`,
      ).bind(childKey?.hash ?? null, childKey?.hash ?? null, new Date().toISOString(), operationId, attemptIndex).run();
    }
  }
}

export async function recordAttemptCredential(
  db: D1Database,
  operationId: string,
  attemptIndex: number,
  managedCredentialRef: string,
  now: Date,
): Promise<boolean> {
  const result = await db
    .prepare(`UPDATE managed_operation_attempts SET managed_credential_ref = ?, outcome = 'created', updated_at = ? WHERE operation_id = ? AND attempt_index = ? AND outcome <> 'cleaned' AND cleanup_token IS NULL AND (managed_credential_ref IS NULL OR managed_credential_ref = ?) AND EXISTS (SELECT 1 FROM managed_operations WHERE operation_id = managed_operation_attempts.operation_id AND current_attempt_index = managed_operation_attempts.attempt_index AND state IN ('CREATING', 'CREATE_UNKNOWN', 'CLEANUP_REQUIRED', 'FAILED'))`)
    .bind(managedCredentialRef, now.toISOString(), operationId, attemptIndex, managedCredentialRef)
    .run();
  return Number(result.meta.changes ?? 0) === 1;
}

export async function markAttemptUnknown(db: D1Database, operationId: string, attemptIndex: number, now: Date): Promise<void> {
  const nowIso = now.toISOString();
  await db
    .prepare(`UPDATE managed_operation_attempts SET outcome = 'unknown', updated_at = ? WHERE operation_id = ? AND attempt_index = ? AND outcome <> 'cleaned' AND cleanup_token IS NULL AND EXISTS (SELECT 1 FROM managed_operations WHERE operation_id = managed_operation_attempts.operation_id AND current_attempt_index = managed_operation_attempts.attempt_index AND state IN ('CREATING', 'CREATE_UNKNOWN', 'CLEANUP_REQUIRED', 'FAILED'))`)
    .bind(nowIso, operationId, attemptIndex)
    .run();
  await transitionManagedOperation(db, operationId, 'CREATE_UNKNOWN', now, { from: ['CREATING'], attemptIndex });
}

export async function releaseUnstartedOperationAttempt(
  db: D1Database,
  operationId: string,
  attemptIndex: number,
  now: Date,
): Promise<boolean> {
  const cleaned = await db
    .prepare(
      `UPDATE managed_operation_attempts
          SET outcome = 'cleaned', legacy_cleanup_state = 'resolved', updated_at = ?
        WHERE operation_id = ?
          AND attempt_index = ?
          AND outcome = 'unknown'
          AND managed_credential_ref IS NULL AND cleanup_token IS NULL AND create_in_flight = 0
          AND EXISTS (SELECT 1 FROM managed_operations WHERE operation_id = managed_operation_attempts.operation_id AND current_attempt_index = managed_operation_attempts.attempt_index AND state IN ('CREATING', 'CREATE_UNKNOWN'))
          AND NOT EXISTS (SELECT 1 FROM managed_key_deliveries WHERE operation_id = ? AND attempt_index = ?)`,
    )
    .bind(now.toISOString(), operationId, attemptIndex, operationId, attemptIndex)
    .run()
    .catch(() => null);
  if (Number(cleaned?.meta.changes ?? 0) !== 1) {
    return false;
  }
  const released = await transitionManagedOperation(db, operationId, 'RETRY_READY', now, {
    from: ['CREATING', 'CREATE_UNKNOWN'],
    attemptIndex,
  });
  return released?.state === 'RETRY_READY';
}

export async function listManagedOperationAttempts(db: D1Database, operationId: string): Promise<ManagedOperationAttemptRecord[]> {
  const result = await db
    .prepare(`SELECT * FROM managed_operation_attempts WHERE operation_id = ? ORDER BY attempt_index ASC`)
    .bind(operationId)
    .all<ManagedOperationAttemptRecord>();
  return result.results ?? [];
}


export async function reconcileUnknownAttempt(
  db: D1Database,
  managementApiKey: string,
  operation: ManagedOperationRecord,
  now: Date,
  fetchImpl?: typeof fetch,
  budget?: ManagedCleanupBudget,
): Promise<ManagedOperationRecord | null> {
  const operationId = operation.operation_id;
  const attemptIndex = operation.current_attempt_index;
  if (attemptIndex === 0) {
    return getManagedOperation(db, operationId);
  }
  const nowIso = now.toISOString();
  const token = crypto.randomUUID();
  const abandonedBefore = new Date(now.getTime() - MANAGED_CLEANUP_CLAIM_TTL_MS).toISOString();
  const retryBefore = new Date(now.getTime() - MANAGED_CLEANUP_RETRY_MS).toISOString();
  const providerName = providerKeyNameForOperationAttempt(operationId, operation.issue_source, attemptIndex);
  await db.prepare(
    `INSERT OR IGNORE INTO managed_operation_attempts
       (operation_id, attempt_index, provider_key_name, managed_credential_ref, outcome, created_at, updated_at, create_in_flight, legacy_cleanup_state)
     SELECT operation_id, current_attempt_index, ?, NULL, 'unknown', ?, ?, 1, 'resolved'
       FROM managed_operations WHERE operation_id = ? AND current_attempt_index = ?
        AND state IN ('CREATE_UNKNOWN', 'RECONCILING', 'CLEANUP_REQUIRED', 'CLEAN', 'DELIVERY_PENDING', 'FAILED')`,
  ).bind(providerName, nowIso, nowIso, operationId, attemptIndex).run();
  const claimed = await db.batch([
    db.prepare(
      `UPDATE managed_operation_attempts SET cleanup_token = ?, cleanup_started_at = ?, legacy_cleanup_checked_at = ?, updated_at = ?
        WHERE operation_id = ? AND attempt_index = ? AND outcome <> 'cleaned'
          AND ((cleanup_token IS NOT NULL AND cleanup_started_at <= ?)
            OR (cleanup_token IS NULL AND (? = 0 OR legacy_cleanup_checked_at IS NULL OR legacy_cleanup_checked_at <= ?)))
          AND EXISTS (
            SELECT 1 FROM managed_operations operation
             WHERE operation.operation_id = managed_operation_attempts.operation_id
               AND operation.current_attempt_index = managed_operation_attempts.attempt_index
               AND operation.state IN ('CREATE_UNKNOWN', 'RECONCILING', 'CLEANUP_REQUIRED', 'CLEAN', 'DELIVERY_PENDING', 'FAILED')
          )
          AND NOT EXISTS (
            SELECT 1 FROM managed_key_deliveries delivery
             WHERE delivery.operation_id = managed_operation_attempts.operation_id
               AND delivery.attempt_index = managed_operation_attempts.attempt_index
               AND (delivery.status = 'acknowledged' OR (delivery.status = 'pending' AND delivery.expires_at >= ?))
          )
          AND NOT EXISTS (
            SELECT 1 FROM openrouter_entitlements entitlement
             WHERE entitlement.managed_credential_ref = managed_operation_attempts.managed_credential_ref
               AND entitlement.discord_issue_delivered_at IS NOT NULL AND entitlement.discord_issue_status = 'active'
          )
          AND NOT EXISTS (
            SELECT 1 FROM qq_managed_entitlements entitlement
             WHERE entitlement.managed_credential_ref = managed_operation_attempts.managed_credential_ref
               AND entitlement.status = 'active' AND entitlement.delivered_at IS NOT NULL
          )`,
    ).bind(token, nowIso, nowIso, nowIso, operationId, attemptIndex, abandonedBefore, budget ? 1 : 0, retryBefore, nowIso),
    db.prepare(
      `UPDATE managed_key_deliveries SET status = 'expired', failed_at = ?, failure_reason = 'operation_cleanup_claimed'
        WHERE operation_id = ? AND attempt_index = ? AND status IN ('pending', 'cleanup_required', 'expired')
          AND EXISTS (SELECT 1 FROM managed_operation_attempts
            WHERE operation_id = ? AND attempt_index = ? AND cleanup_token = ?)`,
    ).bind(nowIso, operationId, attemptIndex, operationId, attemptIndex, token),
    db.prepare(
      `UPDATE managed_operations SET state = CASE WHEN state = 'FAILED' THEN state ELSE 'RECONCILING' END,
          client_action = CASE WHEN state = 'FAILED' THEN 'action_required' ELSE 'wait' END,
          last_reconciled_at = ?, updated_at = ?
        WHERE operation_id = ? AND current_attempt_index = ?
          AND EXISTS (SELECT 1 FROM managed_operation_attempts
            WHERE operation_id = ? AND attempt_index = ? AND cleanup_token = ?)`,
    ).bind(nowIso, nowIso, operationId, attemptIndex, operationId, attemptIndex, token),
  ]);
  if (Number(claimed[0]?.meta.changes ?? 0) !== 1) {
    return getManagedOperation(db, operationId);
  }
  const target = await db.prepare(
    `SELECT * FROM managed_operation_attempts WHERE operation_id = ? AND attempt_index = ? AND cleanup_token = ?`,
  ).bind(operationId, attemptIndex, token).first<ManagedOperationAttemptRecord>();
  if (!target) {
    return getManagedOperation(db, operationId);
  }
  let clean = false;
  let credentialRef = target.managed_credential_ref;
  let nextOffset = target.scan_offset;
  try {
    if (!credentialRef) {
      const found = await findManagedChildKeyByName({
        managementApiKey, keyName: target.provider_key_name, offset: target.scan_offset, pageBudget: budget, fetchImpl,
      });
      if (found.found) {
        credentialRef = found.keyHash;
      } else {
        clean = found.complete && target.create_in_flight === 0;
        nextOffset = found.complete ? 0 : found.nextOffset;
      }
    }
    if (credentialRef) {
      const pinned = await db.prepare(
        `UPDATE managed_operation_attempts SET managed_credential_ref = ?, create_in_flight = 0
          WHERE operation_id = ? AND attempt_index = ? AND cleanup_token = ?`,
      ).bind(credentialRef, operationId, attemptIndex, token).run();
      if (Number(pinned.meta.changes ?? 0) !== 1) {
        return getManagedOperation(db, operationId);
      }
      const cleanup = await cleanupManagedChildKey({ managementApiKey, keyHash: credentialRef, fetchImpl });
      clean = cleanup.ok && await verifyManagedChildKeyAbsent({ managementApiKey, keyHash: credentialRef, fetchImpl });
    }
    if (clean) {
      await db.batch([
        db.prepare(
          `DELETE FROM openrouter_entitlements WHERE managed_credential_ref = ?
            AND status = 'pending_release'
            AND discord_issue_status IN ('issuing', 'delivery_pending', 'cleanup_required')
            AND discord_issue_delivered_at IS NULL
            AND EXISTS (SELECT 1 FROM managed_operation_attempts WHERE operation_id = ? AND attempt_index = ? AND cleanup_token = ?)`,
        ).bind(credentialRef, operationId, attemptIndex, token),
        db.prepare(
          `DELETE FROM discord_identities WHERE discord_user_ref = ? AND status IN ('issuing', 'cleanup_required')
            AND NOT EXISTS (SELECT 1 FROM openrouter_entitlements WHERE discord_user_ref = ?)
            AND EXISTS (SELECT 1 FROM managed_operation_attempts WHERE operation_id = ? AND attempt_index = ? AND cleanup_token = ?)`,
        ).bind(operation.subject_ref, operation.subject_ref, operationId, attemptIndex, token),
        db.prepare(
          `DELETE FROM qq_managed_entitlements WHERE qq_subject_ref = ?
            AND (managed_credential_ref = ? OR (managed_credential_ref IS NULL AND status IN ('issuing', 'cleanup_required')))
            AND status IN ('issuing', 'delivery_pending', 'cleanup_required') AND delivered_at IS NULL
            AND EXISTS (SELECT 1 FROM managed_operation_attempts WHERE operation_id = ? AND attempt_index = ? AND cleanup_token = ?)`,
        ).bind(operation.subject_ref, credentialRef, operationId, attemptIndex, token),
        db.prepare(
          `UPDATE managed_key_deliveries SET status = 'expired', failed_at = ?, failure_reason = 'ack_expired_child_key_cleaned'
            WHERE operation_id = ? AND attempt_index = ? AND status <> 'acknowledged'
            AND EXISTS (SELECT 1 FROM managed_operation_attempts WHERE operation_id = ? AND attempt_index = ? AND cleanup_token = ?)`,
        ).bind(nowIso, operationId, attemptIndex, operationId, attemptIndex, token),
        db.prepare(
          `UPDATE managed_operations SET state = 'RETRY_READY', client_action = 'retry_authorized', updated_at = ?
            WHERE operation_id = ? AND current_attempt_index = ? AND state = 'RECONCILING' AND auth_expires_at > ?
            AND EXISTS (SELECT 1 FROM managed_operation_attempts WHERE operation_id = ? AND attempt_index = ? AND cleanup_token = ?)`,
        ).bind(nowIso, operationId, attemptIndex, nowIso, operationId, attemptIndex, token),
        db.prepare(
          `UPDATE managed_operation_attempts SET outcome = 'cleaned', legacy_cleanup_state = 'resolved', cleanup_token = NULL, cleanup_started_at = NULL, scan_offset = 0, updated_at = ?
            WHERE operation_id = ? AND attempt_index = ? AND cleanup_token = ?`,
        ).bind(nowIso, operationId, attemptIndex, token),
      ]);
    }
  } finally {
    if (!clean) {
      await db.batch([
        db.prepare(
          `UPDATE managed_operations SET cleanup_attempts = cleanup_attempts + 1, last_reconciled_at = ?,
              state = CASE WHEN state = 'FAILED' THEN state ELSE 'CLEANUP_REQUIRED' END, updated_at = ?
            WHERE operation_id = ? AND current_attempt_index = ?
              AND EXISTS (SELECT 1 FROM managed_operation_attempts WHERE operation_id = ? AND attempt_index = ? AND cleanup_token = ?)`,
        ).bind(nowIso, nowIso, operationId, attemptIndex, operationId, attemptIndex, token),
        db.prepare(
          `UPDATE managed_operation_attempts SET cleanup_token = NULL, cleanup_started_at = NULL, scan_offset = ?, updated_at = ?
            WHERE operation_id = ? AND attempt_index = ? AND cleanup_token = ?`,
        ).bind(nextOffset, nowIso, operationId, attemptIndex, token),
      ]);
    }
  }
  return getManagedOperation(db, operationId);
}

function logManagedOperationEvent(
  event: string,
  fields: Record<string, string | number | null>,
): void {
  console.info(event, { ...fields, broker_timestamp: new Date().toISOString() });
}

async function classifyLegacyManagedAttempts(db: D1Database): Promise<void> {
  await db.prepare(
    `UPDATE managed_operation_attempts
        SET create_in_flight = CASE WHEN outcome = 'unknown' AND managed_credential_ref IS NULL THEN 1 ELSE create_in_flight END,
            legacy_cleanup_state = CASE
              WHEN (outcome = 'cleaned' AND (managed_credential_ref IS NULL OR create_in_flight = 1))
                OR (outcome <> 'cleaned' AND EXISTS (
                  SELECT 1 FROM managed_operations operation
                  WHERE operation.operation_id = managed_operation_attempts.operation_id
                    AND operation.current_attempt_index > managed_operation_attempts.attempt_index
                )) THEN 'pending'
              ELSE 'resolved' END
      WHERE id IN (
        SELECT id FROM managed_operation_attempts
        WHERE legacy_cleanup_state = 'unclassified'
          OR (legacy_cleanup_state = 'resolved' AND create_in_flight = 1
            AND (outcome = 'cleaned' OR EXISTS (
              SELECT 1 FROM managed_operations operation
              WHERE operation.operation_id = managed_operation_attempts.operation_id
                AND operation.current_attempt_index > managed_operation_attempts.attempt_index
            )))
        ORDER BY id LIMIT ?
      )`,
  ).bind(SCHEDULED_MANAGED_CLEANUP_LIMIT).run();
}

async function cleanLegacyManagedAttempt(
  env: Pick<BrokerBindings, 'BROKER_DB' | 'OPENROUTER_MANAGEMENT_API_KEY'>,
  now: Date,
  target: ManagedOperationAttemptRecord & { issue_source: 'discord' | 'qq' },
  budget: ManagedCleanupBudget,
  fetchImpl?: typeof fetch,
): Promise<void> {
  const db = env.BROKER_DB;
  const nowIso = now.toISOString();
  const abandonedBefore = new Date(now.getTime() - MANAGED_CLEANUP_CLAIM_TTL_MS).toISOString();
  const retryBefore = new Date(now.getTime() - MANAGED_CLEANUP_RETRY_MS).toISOString();
  const token = crypto.randomUUID();
  const claimed = await db.prepare(
    `UPDATE managed_operation_attempts SET cleanup_token = ?, cleanup_started_at = ?, legacy_cleanup_checked_at = ?
      WHERE id = ? AND legacy_cleanup_state = 'pending'
        AND ((cleanup_token IS NOT NULL AND cleanup_started_at <= ?)
          OR (cleanup_token IS NULL AND (legacy_cleanup_checked_at IS NULL OR legacy_cleanup_checked_at <= ?)))
        AND EXISTS (SELECT 1 FROM managed_operations operation
          WHERE operation.operation_id = managed_operation_attempts.operation_id
            AND operation.current_attempt_index >= managed_operation_attempts.attempt_index)
        AND NOT EXISTS (SELECT 1 FROM managed_key_deliveries delivery
          WHERE delivery.operation_id = managed_operation_attempts.operation_id
            AND delivery.attempt_index = managed_operation_attempts.attempt_index
            AND (delivery.status = 'acknowledged' OR (delivery.status = 'pending' AND delivery.expires_at >= ?)))`,
  ).bind(token, nowIso, nowIso, target.id, abandonedBefore, retryBefore, nowIso).run();
  if (Number(claimed.meta.changes ?? 0) !== 1) return;
  let nextOffset = target.scan_offset;
  try {
    if (target.provider_key_name !== providerKeyNameForOperationAttempt(target.operation_id, target.issue_source, target.attempt_index)) return;
    let hash = target.managed_credential_ref;
    if (!hash) {
      const found = await findManagedChildKeyByName({
        managementApiKey: env.OPENROUTER_MANAGEMENT_API_KEY,
        keyName: target.provider_key_name, offset: target.scan_offset, pageBudget: budget, fetchImpl,
      });
      if (!found.found) {
        nextOffset = found.complete ? 0 : found.nextOffset;
        return;
      }
      hash = found.keyHash;
    }
    const pinned = await db.prepare(
      `UPDATE managed_operation_attempts SET managed_credential_ref = ?, create_in_flight = 0
        WHERE id = ? AND cleanup_token = ? AND legacy_cleanup_state = 'pending'
          AND NOT EXISTS (SELECT 1 FROM managed_key_deliveries delivery
            WHERE delivery.managed_credential_ref = ?
              AND (delivery.status = 'acknowledged' OR (delivery.status = 'pending' AND delivery.expires_at >= ?)))
          AND NOT EXISTS (SELECT 1 FROM openrouter_entitlements entitlement
            WHERE entitlement.managed_credential_ref = ? AND (entitlement.status = 'active' OR entitlement.discord_issue_delivered_at IS NOT NULL))
          AND NOT EXISTS (SELECT 1 FROM qq_managed_entitlements entitlement
            WHERE entitlement.managed_credential_ref = ? AND (entitlement.status = 'active' OR entitlement.delivered_at IS NOT NULL))
          AND NOT EXISTS (
            SELECT 1 FROM managed_operation_attempts other
            JOIN managed_operations operation ON operation.operation_id = other.operation_id
            WHERE other.managed_credential_ref = ? AND operation.current_attempt_index = other.attempt_index
              AND other.id <> managed_operation_attempts.id
          )`,
    ).bind(hash, target.id, token, hash, nowIso, hash, hash, hash).run();
    if (Number(pinned.meta.changes ?? 0) !== 1) return;
    const cleanup = await cleanupManagedChildKey({ managementApiKey: env.OPENROUTER_MANAGEMENT_API_KEY, keyHash: hash, fetchImpl });
    if (!cleanup.ok || !(await verifyManagedChildKeyAbsent({ managementApiKey: env.OPENROUTER_MANAGEMENT_API_KEY, keyHash: hash, fetchImpl }))) return;
    await db.batch([
      db.prepare(
        `UPDATE managed_key_deliveries SET status = 'expired', failed_at = ?, failure_reason = 'legacy_attempt_child_key_cleaned'
          WHERE operation_id = ? AND attempt_index = ? AND status <> 'acknowledged'
            AND EXISTS (SELECT 1 FROM managed_operation_attempts WHERE id = ? AND cleanup_token = ?)`,
      ).bind(nowIso, target.operation_id, target.attempt_index, target.id, token),
      db.prepare(
        `UPDATE managed_operation_attempts
            SET outcome = 'cleaned', create_in_flight = 0, legacy_cleanup_state = 'resolved',
                cleanup_token = NULL, cleanup_started_at = NULL, scan_offset = 0, updated_at = ?
          WHERE id = ? AND cleanup_token = ?`,
      ).bind(nowIso, target.id, token),
    ]);
  } finally {
    await db.prepare(
      `UPDATE managed_operation_attempts SET cleanup_token = NULL, cleanup_started_at = NULL, scan_offset = ?
        WHERE id = ? AND cleanup_token = ?`,
    ).bind(nextOffset, target.id, token).run();
  }
}

export async function sweepStaleManagedOperations(
  env: Pick<BrokerBindings, 'BROKER_DB' | 'OPENROUTER_MANAGEMENT_API_KEY'>,
  now: Date,
  fetchImpl?: typeof fetch,
  budget: ManagedCleanupBudget = createManagedCleanupBudget(),
): Promise<{ expired: number; reconciled: number; retryReady: number }> {
  const db = env.BROKER_DB;
  await classifyLegacyManagedAttempts(db);
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
    await transitionManagedOperation(db, operation.operation_id, 'CREATE_UNKNOWN', now, { from: ['CREATING'], attemptIndex: operation.current_attempt_index });
    reconciled += 1;
  }
  const due = await db.prepare(
    `SELECT operation.*, attempt.id AS cleanup_attempt_id, attempt.legacy_cleanup_state AS cleanup_state,
            COALESCE(attempt.legacy_cleanup_checked_at, attempt.created_at) AS cleanup_checked_at
       FROM managed_operation_attempts attempt
       JOIN managed_operations operation ON operation.operation_id = attempt.operation_id
      WHERE (
        (attempt.legacy_cleanup_state = 'pending' AND operation.current_attempt_index >= attempt.attempt_index)
        OR (attempt.legacy_cleanup_state = 'resolved' AND attempt.attempt_index = operation.current_attempt_index
          AND attempt.outcome <> 'cleaned'
          AND operation.state IN ('CREATE_UNKNOWN', 'RECONCILING', 'CLEANUP_REQUIRED', 'CLEAN', 'FAILED', 'DELIVERY_PENDING'))
      )
        AND ((attempt.cleanup_token IS NOT NULL AND attempt.cleanup_started_at <= ?)
          OR (attempt.cleanup_token IS NULL AND (attempt.legacy_cleanup_checked_at IS NULL OR attempt.legacy_cleanup_checked_at <= ?)))
        AND NOT EXISTS (
          SELECT 1 FROM managed_key_deliveries delivery
          WHERE delivery.operation_id = attempt.operation_id AND delivery.attempt_index = attempt.attempt_index
            AND (delivery.status = 'acknowledged' OR (delivery.status = 'pending' AND delivery.expires_at >= ?))
        )
      UNION ALL
     SELECT operation.*, NULL AS cleanup_attempt_id, NULL AS cleanup_state,
            COALESCE(operation.last_reconciled_at, operation.created_at) AS cleanup_checked_at
       FROM managed_operations operation
      WHERE operation.current_attempt_index > 0
        AND operation.state IN ('CREATE_UNKNOWN', 'RECONCILING', 'CLEANUP_REQUIRED', 'CLEAN', 'FAILED', 'DELIVERY_PENDING')
        AND NOT EXISTS (
          SELECT 1 FROM managed_operation_attempts attempt WHERE attempt.operation_id = operation.operation_id
            AND attempt.attempt_index = operation.current_attempt_index
        )
      ORDER BY cleanup_checked_at, cleanup_attempt_id, operation_id LIMIT ?`,
  ).bind(
    new Date(now.getTime() - MANAGED_CLEANUP_CLAIM_TTL_MS).toISOString(),
    new Date(now.getTime() - MANAGED_CLEANUP_RETRY_MS).toISOString(),
    now.toISOString(), budget.remainingCandidates,
  ).all<ManagedOperationRecord & { cleanup_attempt_id: number | null; cleanup_state: string | null }>();
  for (const operation of due.results ?? []) {
    if (budget.remainingCandidates === 0 || budget.remainingProviderPages === 0) break;
    budget.remainingCandidates -= 1;
    if (operation.cleanup_state === 'pending') {
      const attempt = await db.prepare(`SELECT * FROM managed_operation_attempts WHERE id = ?`)
        .bind(operation.cleanup_attempt_id).first<ManagedOperationAttemptRecord>();
      if (attempt) await cleanLegacyManagedAttempt(env, now, { ...attempt, issue_source: operation.issue_source }, budget, fetchImpl);
      continue;
    }
    const after = await reconcileUnknownAttempt(db, env.OPENROUTER_MANAGEMENT_API_KEY, operation, now, fetchImpl, budget);
    if (after && after.state !== operation.state) reconciled += 1;
    if (after?.state === 'RETRY_READY') retryReady += 1;
  }
  if (expired || reconciled || retryReady) {
    logManagedOperationEvent('managed_operation_sweep_completed', {
      expired,
      reconciled,
      retry_ready: retryReady,
    });
  }
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
  const statusRateLimit = await instrumentPublicPostRoute(c.env.BROKER_DB, c, {
    endpoint: 'POST /v1/providers/openrouter/managed-operation/status',
    installationId,
  });
  if (statusRateLimit) {
    return c.json(
      {
        ok: false,
        code: 'rate_limited',
        message: statusRateLimit.message,
        retry_after_ms: statusRateLimit.retryAfterMs,
      },
      429,
    );
  }
  const authenticated = await authenticateManagedOperationRequest(c.env.BROKER_DB, { operationId, resumeToken, installationId });
  if (!authenticated.ok) {
    return c.json({ ok: false, code: 'invalid_request', message: 'unknown operation' }, 404);
  }
  authenticated.operation = await rejectUnboundQqOperation(c.env.BROKER_DB, authenticated.operation, new Date());
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
  if (existing.issue_source === 'qq' && existing.installation_id === null && existing.state !== 'ACTIVE') {
    const authenticated = await authenticateManagedOperationRequest(db, {
      operationId: input.operationId, resumeToken: input.resumeToken, installationId: input.installationId,
    });
    if (!authenticated.ok) return { status: 'invalid', reason: 'unknown' };
    return { status: 'failed', operation: await rejectUnboundQqOperation(db, authenticated.operation, input.now) };
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
          AND state IN ('DELIVERY_PENDING', 'FAILED')
          AND current_attempt_index = (SELECT attempt_index FROM managed_key_deliveries WHERE delivery_id = ?)
          AND NOT EXISTS (SELECT 1 FROM managed_operation_attempts WHERE operation_id = managed_operations.operation_id AND attempt_index = current_attempt_index AND cleanup_token IS NOT NULL)`,
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

