import type { Context } from 'hono';

import {
  checkActiveIssuanceBrake,
  extractRequestNetworkMetadata,
  getManagedDailyIssuanceCapState,
  type AbuseDecision,
} from './abuse-controls';
import {
  deliverImmediateMonitoringSideEffects,
  evaluateImmediateAbuseState,
  recordIssueSuccess,
} from './abuse-monitoring';
import {
  errorResponse as publicErrorResponse,
  internalErrorResponse,
} from './broker-error';
import type { BrokerEnv } from './contract';
import {
  buildManagedCleanupRequiredAuditPayload,
  getManagedIssuanceSourcePolicy,
} from './managed-issuance';
import { createManagedKeyDelivery } from './managed-key-delivery';
import {
  assignManagedGuardrail,
  cleanupManagedChildKey,
  createManagedChildKey,
  OpenRouterManagementError,
  type ManagedChildKeyCleanupResult,
} from './openrouter-management';
import { deriveManagedOpenRouterUserId } from './openrouter-user-id';
import {
  QQ_MANAGED_ENTITLEMENT_STALE_ISSUING_POLICY,
  type QqManagedEntitlementRecord,
} from './persistence';
import { nonEmptyString } from './public-input';
import { MANAGED_TRIAL_POLICY } from './trial-policy';

const QQ_ISSUE_REF_PREFIX = 'qq-issue-v1_';
const ISSUE_SOURCE = 'qq' as const;
const MANAGED_KEY_DELIVERY_ACK_TTL_MS = 15 * 60_000;

interface QqManagedIssueInput {
  qqSubjectRef: string;
  now: Date;
  deliveryAckSupported?: boolean;
}

type QqReservationResult =
  | { ok: true; issueRef: string }
  | { ok: false; reason: 'lifetime_used' }
  | { ok: false; reason: 'already_issuing'; retryAfterMs: number | null }
  | { ok: false; reason: 'reservation_failed' };

export async function issueQqManagedEntitlement(
  c: Context<BrokerEnv>,
  input: QqManagedIssueInput,
): Promise<Response> {
  const nowIso = input.now.toISOString();
  const sourcePolicy = getManagedIssuanceSourcePolicy(ISSUE_SOURCE);
  const issueMetadata = {
    issueSource: ISSUE_SOURCE,
    subjectRef: input.qqSubjectRef,
    issueRef: createIssueRef(),
  };
  let reservationCreated = false;
  let childKey: { rawKey: string; hash: string } | null = null;
  let childKeyAttached = false;

  try {
    const currentEntitlement = await getQqManagedEntitlement(
      c.env.BROKER_DB,
      input.qqSubjectRef,
    );
    if (isLifetimeBlockingQqEntitlement(currentEntitlement)) {
      return qqLifetimeUsedResponse(c);
    }

    const brakeDecision = await checkActiveIssuanceBrake(
      c.env.BROKER_DB,
      currentEntitlement,
    );
    if (brakeDecision) {
      return abuseDecisionResponse(c, brakeDecision);
    }

    const reservation = await reserveQqManagedEntitlement(c.env.BROKER_DB, {
      qqSubjectRef: input.qqSubjectRef,
      issueRef: issueMetadata.issueRef,
      budgetUsd: sourcePolicy.budget_usd,
      now: input.now,
      nowIso,
    });
    if (!reservation.ok) {
      return qqReservationErrorResponse(c, reservation);
    }
    reservationCreated = true;

    const cap = await getManagedDailyIssuanceCapState(c.env.BROKER_DB, input.now, {
      excludeCurrent: issueMetadata,
    });
    if (cap.reached) {
      await releaseQqReservationBeforeChildKey(c.env.BROKER_DB, {
        qqSubjectRef: input.qqSubjectRef,
        issueRef: issueMetadata.issueRef,
      });
      reservationCreated = false;
      return publicErrorResponse(c, 503, {
        code: 'issuance_suspended',
        class: 'retryable',
        subcode: 'global_cap_reached',
        retryAfterMs: cap.retryAfterMs,
        message: 'Daily managed issuance cap reached',
        entitlement: null,
      });
    }

    const issuedAt = nowIso;
    const deliveredAt = nowIso;
    const expiresAt = addMonthsUtc(
      input.now,
      MANAGED_TRIAL_POLICY.entitlement.issuance.expiry.durationMonths,
    ).toISOString();

    childKey = await createManagedChildKey({
      managementApiKey: c.env.OPENROUTER_MANAGEMENT_API_KEY,
      issueSource: ISSUE_SOURCE,
      subjectRef: input.qqSubjectRef,
      issueRef: issueMetadata.issueRef,
      expiresAt,
      limitUsd: sourcePolicy.budget_usd,
    });

    const attached = await attachManagedCredentialToQqReservation(c.env.BROKER_DB, {
      qqSubjectRef: input.qqSubjectRef,
      issueRef: issueMetadata.issueRef,
      managedCredentialRef: childKey.hash,
      nowIso,
    });
    if (!attached) {
      throw new Error('QQ managed child key reservation attachment failed');
    }
    childKeyAttached = true;

    await assignManagedGuardrail({
      managementApiKey: c.env.OPENROUTER_MANAGEMENT_API_KEY,
      guardrailId: c.env.OPENROUTER_MANAGED_GUARDRAIL_ID,
      keyHash: childKey.hash,
    });

    if (input.deliveryAckSupported === true) {
      const pending = await markQqReservationDeliveryPending(c.env.BROKER_DB, {
        qqSubjectRef: input.qqSubjectRef,
        issueRef: issueMetadata.issueRef,
        managedCredentialRef: childKey.hash,
        budgetUsd: sourcePolicy.budget_usd,
        issuedAt,
        expiresAt,
        nowIso,
      });
      if (!pending) {
        throw new Error('QQ managed entitlement delivery-pending transition failed');
      }
      const deliveryAckExpiresAt = new Date(
        input.now.getTime() + MANAGED_KEY_DELIVERY_ACK_TTL_MS,
      );
      const delivery = await createManagedKeyDelivery(c.env.BROKER_DB, {
        issueSource: ISSUE_SOURCE,
        subjectRef: input.qqSubjectRef,
        managedCredentialRef: childKey.hash,
        createdAt: input.now,
        expiresAt: deliveryAckExpiresAt,
      });
      const openRouterUserId = await deriveOptionalOpenRouterUserId({
        subjectRef: input.qqSubjectRef,
        secret: c.env.OPENROUTER_MANAGED_USER_HMAC_SECRET,
      });
      return c.json({
        ok: true,
        status: 'delivery_pending',
        qq_subject_ref: input.qqSubjectRef,
        openrouter_api_key: childKey.rawKey,
        managed_credential_ref: childKey.hash,
        expires_at: expiresAt,
        delivery_ack_required: true,
        delivery_id: delivery.deliveryId,
        delivery_ack_token: delivery.deliveryAckToken,
        delivery_ack_expires_at: deliveryAckExpiresAt.toISOString(),
        ...(openRouterUserId ? { openrouter_user_id: openRouterUserId } : {}),
      });
    }

    const activated = await activateQqReservation(c.env.BROKER_DB, {
      qqSubjectRef: input.qqSubjectRef,
      issueRef: issueMetadata.issueRef,
      managedCredentialRef: childKey.hash,
      budgetUsd: sourcePolicy.budget_usd,
      issuedAt,
      expiresAt,
      deliveredAt,
    });
    if (!activated) {
      throw new Error('QQ managed entitlement activation failed');
    }

    await runQqIssueSuccessMonitoring(c, {
      qqSubjectRef: input.qqSubjectRef,
      managedCredentialRef: childKey.hash,
      observedAt: issuedAt,
      now: input.now,
    });

    const openRouterUserId = await deriveOptionalOpenRouterUserId({
      subjectRef: input.qqSubjectRef,
      secret: c.env.OPENROUTER_MANAGED_USER_HMAC_SECRET,
    });

    return c.json({
      ok: true,
      status: 'issued',
      qq_subject_ref: input.qqSubjectRef,
      openrouter_api_key: childKey.rawKey,
      managed_credential_ref: childKey.hash,
      expires_at: expiresAt,
      ...(openRouterUserId ? { openrouter_user_id: openRouterUserId } : {}),
    });
  } catch (error) {
    if (
      !childKey &&
      error instanceof OpenRouterManagementError &&
      error.createdChildKey
    ) {
      childKey = error.createdChildKey;
    }

    if (reservationCreated) {
      if (!childKey) {
        await bestEffortReleaseQqReservationBeforeChildKey(c.env.BROKER_DB, {
          qqSubjectRef: input.qqSubjectRef,
          issueRef: issueMetadata.issueRef,
        });
      } else {
        await handleQqManagedChildKeyFailure(c, {
          qqSubjectRef: input.qqSubjectRef,
          issueRef: issueMetadata.issueRef,
          childKey,
          childKeyAttached,
          nowIso,
          error,
        });
      }
    }

    return internalErrorResponse(c);
  }
}

async function reserveQqManagedEntitlement(
  db: D1Database,
  input: {
    qqSubjectRef: string;
    issueRef: string;
    budgetUsd: number;
    now: Date;
    nowIso: string;
  },
): Promise<QqReservationResult> {
  const insertResult = await db
    .prepare(
      `INSERT INTO qq_managed_entitlements (
          qq_subject_ref,
          status,
          issue_ref,
          managed_credential_ref,
          budget_usd,
          reserved_at,
          issued_at,
          expires_at,
          delivered_at,
          created_at,
          updated_at
        ) VALUES (?, 'issuing', ?, NULL, ?, ?, NULL, NULL, NULL, ?, ?)
        ON CONFLICT(qq_subject_ref) DO NOTHING`,
    )
    .bind(
      input.qqSubjectRef,
      input.issueRef,
      input.budgetUsd,
      input.nowIso,
      input.nowIso,
      input.nowIso,
    )
    .run();
  if (Number(insertResult.meta.changes ?? 0) === 1) {
    return { ok: true, issueRef: input.issueRef };
  }

  const current = await getQqManagedEntitlement(db, input.qqSubjectRef);
  if (!current) {
    return { ok: false, reason: 'reservation_failed' };
  }
  if (isLifetimeBlockingQqEntitlement(current)) {
    return { ok: false, reason: 'lifetime_used' };
  }
  if (current.status !== 'issuing') {
    return { ok: false, reason: 'reservation_failed' };
  }

  if (!isStaleQqIssuingReservation(current, input.now)) {
    return {
      ok: false,
      reason: 'already_issuing',
      retryAfterMs: retryAfterStaleIssuingTtl(current, input.now),
    };
  }

  if (current.managed_credential_ref) {
    await markStaleQqIssuingCleanupRequired(db, {
      qqSubjectRef: input.qqSubjectRef,
      issueRef: current.issue_ref,
      managedCredentialRef: current.managed_credential_ref,
      nowIso: input.nowIso,
    });
    return { ok: false, reason: 'lifetime_used' };
  }

  const reclaimResult = await db
    .prepare(
      `UPDATE qq_managed_entitlements
          SET issue_ref = ?,
              budget_usd = ?,
              reserved_at = ?,
              issued_at = NULL,
              expires_at = NULL,
              delivered_at = NULL,
              updated_at = ?
        WHERE qq_subject_ref = ?
          AND issue_ref = ?
          AND status = 'issuing'
          AND managed_credential_ref IS NULL`,
    )
    .bind(
      input.issueRef,
      input.budgetUsd,
      input.nowIso,
      input.nowIso,
      input.qqSubjectRef,
      current.issue_ref,
    )
    .run();

  return Number(reclaimResult.meta.changes ?? 0) === 1
    ? { ok: true, issueRef: input.issueRef }
    : { ok: false, reason: 'reservation_failed' };
}

async function getQqManagedEntitlement(
  db: D1Database,
  qqSubjectRef: string,
): Promise<QqManagedEntitlementRecord | null> {
  return db
    .prepare(
      `SELECT qq_subject_ref, status, issue_ref, managed_credential_ref,
              budget_usd, reserved_at, issued_at, expires_at, delivered_at,
              created_at, updated_at
         FROM qq_managed_entitlements
        WHERE qq_subject_ref = ?`,
    )
    .bind(qqSubjectRef)
    .first<QqManagedEntitlementRecord>();
}

function isLifetimeBlockingQqEntitlement(
  entitlement: QqManagedEntitlementRecord | null,
): boolean {
  return (
    entitlement?.status === 'active' ||
    entitlement?.status === 'delivery_pending' ||
    entitlement?.status === 'cleanup_required' ||
    entitlement?.status === 'revoked'
  );
}

function isStaleQqIssuingReservation(
  entitlement: QqManagedEntitlementRecord,
  now: Date,
): boolean {
  const reservedAtMs = Date.parse(entitlement.reserved_at);
  if (!Number.isFinite(reservedAtMs)) {
    return true;
  }

  return now.getTime() - reservedAtMs >= staleIssuingTtlMs();
}

function retryAfterStaleIssuingTtl(
  entitlement: QqManagedEntitlementRecord,
  now: Date,
): number | null {
  const reservedAtMs = Date.parse(entitlement.reserved_at);
  if (!Number.isFinite(reservedAtMs)) {
    return null;
  }

  return Math.max(reservedAtMs + staleIssuingTtlMs() - now.getTime(), 0);
}

function staleIssuingTtlMs(): number {
  return QQ_MANAGED_ENTITLEMENT_STALE_ISSUING_POLICY.ttlMinutes * 60_000;
}

async function attachManagedCredentialToQqReservation(
  db: D1Database,
  input: {
    qqSubjectRef: string;
    issueRef: string;
    managedCredentialRef: string;
    nowIso: string;
  },
): Promise<boolean> {
  const result = await db
    .prepare(
      `UPDATE qq_managed_entitlements
          SET managed_credential_ref = ?,
              updated_at = ?
        WHERE qq_subject_ref = ?
          AND issue_ref = ?
          AND status = 'issuing'
          AND managed_credential_ref IS NULL`,
    )
    .bind(
      input.managedCredentialRef,
      input.nowIso,
      input.qqSubjectRef,
      input.issueRef,
    )
    .run();

  return Number(result.meta.changes ?? 0) === 1;
}

async function activateQqReservation(
  db: D1Database,
  input: {
    qqSubjectRef: string;
    issueRef: string;
    managedCredentialRef: string;
    budgetUsd: number;
    issuedAt: string;
    expiresAt: string;
    deliveredAt: string;
  },
): Promise<boolean> {
  const result = await db
    .prepare(
      `UPDATE qq_managed_entitlements
          SET status = 'active',
              managed_credential_ref = ?,
              budget_usd = ?,
              issued_at = ?,
              expires_at = ?,
              delivered_at = ?,
              updated_at = ?
        WHERE qq_subject_ref = ?
          AND issue_ref = ?
          AND status IN ('issuing', 'delivery_pending')
          AND managed_credential_ref = ?`,
    )
    .bind(
      input.managedCredentialRef,
      input.budgetUsd,
      input.issuedAt,
      input.expiresAt,
      input.deliveredAt,
      input.deliveredAt,
      input.qqSubjectRef,
      input.issueRef,
      input.managedCredentialRef,
    )
    .run();

  return Number(result.meta.changes ?? 0) === 1;
}

async function markQqReservationDeliveryPending(
  db: D1Database,
  input: {
    qqSubjectRef: string;
    issueRef: string;
    managedCredentialRef: string;
    budgetUsd: number;
    issuedAt: string;
    expiresAt: string;
    nowIso: string;
  },
): Promise<boolean> {
  const result = await db
    .prepare(
      `UPDATE qq_managed_entitlements
          SET status = 'delivery_pending',
              managed_credential_ref = ?,
              budget_usd = ?,
              issued_at = ?,
              expires_at = ?,
              delivered_at = NULL,
              updated_at = ?
        WHERE qq_subject_ref = ?
          AND issue_ref = ?
          AND status = 'issuing'
          AND managed_credential_ref = ?`,
    )
    .bind(
      input.managedCredentialRef,
      input.budgetUsd,
      input.issuedAt,
      input.expiresAt,
      input.nowIso,
      input.qqSubjectRef,
      input.issueRef,
      input.managedCredentialRef,
    )
    .run();

  return Number(result.meta.changes ?? 0) === 1;
}

export async function finalizeQqManagedKeyDeliveryAck(
  c: Context<BrokerEnv>,
  input: { managedCredentialRef: string; acknowledgedAt: Date },
): Promise<void> {
  const entitlement = await c.env.BROKER_DB.prepare(
    `SELECT qq_subject_ref, status, issue_ref, managed_credential_ref,
            budget_usd, reserved_at, issued_at, expires_at, delivered_at,
            created_at, updated_at
       FROM qq_managed_entitlements
      WHERE managed_credential_ref = ?`,
  )
    .bind(input.managedCredentialRef)
    .first<QqManagedEntitlementRecord>();
  if (!entitlement) {
    throw new Error('QQ delivery ACK target is missing');
  }
  const alreadyActive = entitlement.status === 'active';
  if (!alreadyActive && entitlement.status !== 'delivery_pending') {
    throw new Error('QQ delivery ACK target is not pending');
  }
  const deliveredAt = entitlement.delivered_at ?? input.acknowledgedAt.toISOString();
  if (!alreadyActive) {
    const activated = await activateQqReservation(c.env.BROKER_DB, {
      qqSubjectRef: entitlement.qq_subject_ref,
      issueRef: entitlement.issue_ref,
      managedCredentialRef: input.managedCredentialRef,
      budgetUsd: entitlement.budget_usd,
      issuedAt: entitlement.issued_at ?? deliveredAt,
      expiresAt: entitlement.expires_at!,
      deliveredAt,
    });
    if (!activated) {
      throw new Error('QQ delivery ACK activation failed');
    }
  }
  if (!(await hasQqIssueSuccessRecord(c.env.BROKER_DB, input.managedCredentialRef))) {
    try {
      await runQqIssueSuccessMonitoring(c, {
        qqSubjectRef: entitlement.qq_subject_ref,
        managedCredentialRef: input.managedCredentialRef,
        observedAt: deliveredAt,
        now: input.acknowledgedAt,
      });
    } catch (error) {
      const reverted = await c.env.BROKER_DB.prepare(
        `UPDATE qq_managed_entitlements
            SET status = 'delivery_pending', delivered_at = NULL, updated_at = ?
          WHERE managed_credential_ref = ?
            AND status = 'active'`,
      )
        .bind(input.acknowledgedAt.toISOString(), input.managedCredentialRef)
        .run();
      if (Number(reverted.meta.changes ?? 0) !== 1) {
        throw new Error('QQ delivery ACK monitoring rollback failed');
      }
      throw error;
    }
  }
}

async function hasQqIssueSuccessRecord(
  db: D1Database,
  managedCredentialRef: string,
): Promise<boolean> {
  const row = await db
    .prepare(
      `SELECT 1 AS found
         FROM broker_issue_success_events
        WHERE managed_credential_ref = ?
        LIMIT 1`,
    )
    .bind(managedCredentialRef)
    .first<{ found: number }>();
  return Number(row?.found ?? 0) === 1;
}

async function releaseQqReservationBeforeChildKey(
  db: D1Database,
  input: {
    qqSubjectRef: string;
    issueRef: string;
  },
): Promise<void> {
  await db
    .prepare(
      `DELETE FROM qq_managed_entitlements
        WHERE qq_subject_ref = ?
          AND issue_ref = ?
          AND status = 'issuing'
          AND managed_credential_ref IS NULL`,
    )
    .bind(input.qqSubjectRef, input.issueRef)
    .run();
}

async function bestEffortReleaseQqReservationBeforeChildKey(
  db: D1Database,
  input: {
    qqSubjectRef: string;
    issueRef: string;
  },
): Promise<void> {
  try {
    await releaseQqReservationBeforeChildKey(db, input);
  } catch {
    // Keep public failure bounded; a later request sees the still-issuing row.
  }
}

async function releaseQqReservationAfterManagedCleanup(
  db: D1Database,
  input: {
    qqSubjectRef: string;
    issueRef: string;
    managedCredentialRef: string;
  },
): Promise<void> {
  await db
    .prepare(
      `DELETE FROM qq_managed_entitlements
        WHERE qq_subject_ref = ?
          AND issue_ref = ?
          AND status IN ('issuing', 'delivery_pending', 'active')
          AND managed_credential_ref = ?`,
    )
    .bind(input.qqSubjectRef, input.issueRef, input.managedCredentialRef)
    .run();
}

async function markQqCleanupRequired(
  db: D1Database,
  input: {
    qqSubjectRef: string;
    issueRef: string;
    managedCredentialRef: string;
    nowIso: string;
  },
): Promise<void> {
  await db
    .prepare(
      `UPDATE qq_managed_entitlements
          SET status = 'cleanup_required',
              managed_credential_ref = ?,
              issued_at = NULL,
              expires_at = NULL,
              delivered_at = NULL,
              updated_at = ?
        WHERE qq_subject_ref = ?
          AND issue_ref = ?
          AND status IN ('issuing', 'delivery_pending', 'active')
          AND managed_credential_ref = ?`,
    )
    .bind(
      input.managedCredentialRef,
      input.nowIso,
      input.qqSubjectRef,
      input.issueRef,
      input.managedCredentialRef,
    )
    .run();
}

async function markUnattachedQqCleanupRequired(
  db: D1Database,
  input: {
    qqSubjectRef: string;
    issueRef: string;
    managedCredentialRef: string;
    nowIso: string;
  },
): Promise<void> {
  await db
    .prepare(
      `UPDATE qq_managed_entitlements
          SET status = 'cleanup_required',
              managed_credential_ref = ?,
              issued_at = NULL,
              expires_at = NULL,
              delivered_at = NULL,
              updated_at = ?
        WHERE qq_subject_ref = ?
          AND issue_ref = ?
          AND status = 'issuing'
          AND managed_credential_ref IS NULL`,
    )
    .bind(
      input.managedCredentialRef,
      input.nowIso,
      input.qqSubjectRef,
      input.issueRef,
    )
    .run();
}

async function markStaleQqIssuingCleanupRequired(
  db: D1Database,
  input: {
    qqSubjectRef: string;
    issueRef: string;
    managedCredentialRef: string;
    nowIso: string;
  },
): Promise<void> {
  await db
    .prepare(
      `UPDATE qq_managed_entitlements
          SET status = 'cleanup_required',
              updated_at = ?
        WHERE qq_subject_ref = ?
          AND issue_ref = ?
          AND status = 'issuing'
          AND managed_credential_ref = ?`,
    )
    .bind(
      input.nowIso,
      input.qqSubjectRef,
      input.issueRef,
      input.managedCredentialRef,
    )
    .run();
}

async function handleQqManagedChildKeyFailure(
  c: Context<BrokerEnv>,
  input: {
    qqSubjectRef: string;
    issueRef: string;
    childKey: { rawKey: string; hash: string };
    childKeyAttached: boolean;
    nowIso: string;
    error: unknown;
  },
): Promise<void> {
  const cleanup = await cleanupManagedChildKey({
    managementApiKey: c.env.OPENROUTER_MANAGEMENT_API_KEY,
    keyHash: input.childKey.hash,
  });

  if (cleanup.ok) {
    try {
      if (input.childKeyAttached) {
        await releaseQqReservationAfterManagedCleanup(c.env.BROKER_DB, {
          qqSubjectRef: input.qqSubjectRef,
          issueRef: input.issueRef,
          managedCredentialRef: input.childKey.hash,
        });
      } else {
        await releaseQqReservationBeforeChildKey(c.env.BROKER_DB, {
          qqSubjectRef: input.qqSubjectRef,
          issueRef: input.issueRef,
        });
      }
    } catch (error) {
      logQqCleanupReleaseFailure({
        qqSubjectRef: input.qqSubjectRef,
        issueRef: input.issueRef,
        managedCredentialRef: input.childKey.hash,
        childKeyAttached: input.childKeyAttached,
        error,
        nowIso: input.nowIso,
      });
    }
    return;
  }

  logQqCleanupRequired({
    qqSubjectRef: input.qqSubjectRef,
    issueRef: input.issueRef,
    managedCredentialRef: input.childKey.hash,
    error: input.error,
    cleanup,
    nowIso: input.nowIso,
  });
  try {
    if (input.childKeyAttached) {
      await markQqCleanupRequired(c.env.BROKER_DB, {
        qqSubjectRef: input.qqSubjectRef,
        issueRef: input.issueRef,
        managedCredentialRef: input.childKey.hash,
        nowIso: input.nowIso,
      });
    } else {
      await markUnattachedQqCleanupRequired(c.env.BROKER_DB, {
        qqSubjectRef: input.qqSubjectRef,
        issueRef: input.issueRef,
        managedCredentialRef: input.childKey.hash,
        nowIso: input.nowIso,
      });
    }
  } catch (markError) {
    logQqCleanupStateUpdateFailure({
      qqSubjectRef: input.qqSubjectRef,
      issueRef: input.issueRef,
      managedCredentialRef: input.childKey.hash,
      childKeyAttached: input.childKeyAttached,
      error: markError,
      nowIso: input.nowIso,
    });
  }
}

function logQqCleanupRequired(input: {
  qqSubjectRef: string;
  issueRef: string;
  managedCredentialRef: string;
  error: unknown;
  cleanup: Extract<ManagedChildKeyCleanupResult, { ok: false }>;
  nowIso: string;
}): void {
  console.error(
    'qq_managed_child_key_cleanup_required',
    buildManagedCleanupRequiredAuditPayload({
      issueSource: ISSUE_SOURCE,
      subjectRef: input.qqSubjectRef,
      issueRef: input.issueRef,
      managedCredentialRef: input.managedCredentialRef,
      failure: input.error,
      cleanupOutcome: cleanupOutcomeReason(input.cleanup),
      brokerTimestamp: input.nowIso,
    }),
  );
}

function cleanupOutcomeReason(
  cleanup: Extract<ManagedChildKeyCleanupResult, { ok: false }>,
): Extract<ManagedChildKeyCleanupResult, { ok: false }>['reason'] {
  return cleanup.reason;
}

function logQqCleanupStateUpdateFailure(input: {
  qqSubjectRef: string;
  issueRef: string;
  managedCredentialRef: string;
  childKeyAttached: boolean;
  error: unknown;
  nowIso: string;
}): void {
  console.error('qq_managed_child_key_cleanup_state_update_failed', {
    issue_source: ISSUE_SOURCE,
    subject_ref: input.qqSubjectRef,
    issue_ref: input.issueRef,
    managed_credential_ref: input.managedCredentialRef,
    child_key_attached: input.childKeyAttached,
    error_name: safeErrorName(input.error),
    broker_timestamp: input.nowIso,
  });
}

function logQqCleanupReleaseFailure(input: {
  qqSubjectRef: string;
  issueRef: string;
  managedCredentialRef: string;
  childKeyAttached: boolean;
  error: unknown;
  nowIso: string;
}): void {
  console.error('qq_managed_child_key_cleanup_release_failed', {
    issue_source: ISSUE_SOURCE,
    subject_ref: input.qqSubjectRef,
    issue_ref: input.issueRef,
    managed_credential_ref: input.managedCredentialRef,
    child_key_attached: input.childKeyAttached,
    error_name: safeErrorName(input.error),
    broker_timestamp: input.nowIso,
  });
}

async function runQqIssueSuccessMonitoring(
  c: Context<BrokerEnv>,
  input: {
    qqSubjectRef: string;
    managedCredentialRef: string;
    observedAt: string;
    now: Date;
  },
): Promise<void> {
  try {
    const network = await extractRequestNetworkMetadata(c, c.env.BROKER_DB);
    await recordIssueSuccess(c.env.BROKER_DB, {
      issueSource: ISSUE_SOURCE,
      subjectRef: input.qqSubjectRef,
      managedCredentialRef: input.managedCredentialRef,
      observedAt: input.observedAt,
      network,
    });
  } catch (error) {
    logQqIssueSuccessMonitoringFailure(input, 'record', error);
    throw error;
  }

  try {
    const monitoringResult = await evaluateImmediateAbuseState(
      c.env.BROKER_DB,
      input.now,
    );
    await deliverImmediateMonitoringSideEffects(c.env, monitoringResult);
  } catch (error) {
    logQqIssueSuccessMonitoringFailure(input, 'evaluate_or_deliver', error);
  }
}

function logQqIssueSuccessMonitoringFailure(
  input: {
    qqSubjectRef: string;
    managedCredentialRef: string;
  },
  stage: 'record' | 'evaluate_or_deliver',
  error: unknown,
): void {
  console.error('qq_issue_success_monitoring_failed', {
    issue_source: ISSUE_SOURCE,
    subject_ref: input.qqSubjectRef,
    managed_credential_ref: input.managedCredentialRef,
    stage,
    error_name: safeErrorName(error),
    broker_timestamp: new Date().toISOString(),
  });
}

async function deriveOptionalOpenRouterUserId(input: {
  subjectRef: string;
  secret: unknown;
}): Promise<string | null> {
  const secret = nonEmptyString(input.secret);
  if (!secret) {
    return null;
  }

  try {
    return await deriveManagedOpenRouterUserId({
      issueSource: ISSUE_SOURCE,
      subjectRef: input.subjectRef,
      secret,
    });
  } catch {
    return null;
  }
}

function abuseDecisionResponse(c: Context<BrokerEnv>, decision: AbuseDecision): Response {
  return publicErrorResponse(c, decision.status, {
    code: decision.code,
    class: decision.class,
    subcode: decision.subcode,
    retryAfterMs: decision.retryAfterMs,
    message: decision.message,
    entitlement: null,
  });
}

function qqReservationErrorResponse(
  c: Context<BrokerEnv>,
  reservation: Exclude<QqReservationResult, { ok: true }>,
): Response {
  switch (reservation.reason) {
    case 'lifetime_used':
      return qqLifetimeUsedResponse(c);
    case 'already_issuing':
      return publicErrorResponse(c, 409, {
        code: 'trial_not_eligible',
        class: 'retryable',
        subcode: 'qq_already_issuing',
        retryAfterMs: reservation.retryAfterMs,
        message: 'QQ managed issuance is already in progress',
        entitlement: null,
      });
    case 'reservation_failed':
      return publicErrorResponse(c, 500, {
        code: 'internal_error',
        class: 'retryable',
        subcode: 'entitlement_reservation_failed',
        message: 'Managed entitlement reservation failed',
        entitlement: null,
      });
  }
}

function qqLifetimeUsedResponse(c: Context<BrokerEnv>): Response {
  return publicErrorResponse(c, 409, {
    code: 'trial_not_eligible',
    class: 'terminal',
    subcode: 'qq_lifetime_used',
    message: 'QQ subject has already used a managed trial',
    entitlement: null,
  });
}

function addMonthsUtc(value: Date, months: number): Date {
  const next = new Date(value.getTime());
  next.setUTCMonth(next.getUTCMonth() + months);
  return next;
}

function createIssueRef(): string {
  return `${QQ_ISSUE_REF_PREFIX}${crypto.randomUUID()}`;
}

function safeErrorName(error: unknown): string {
  if (!(error instanceof Error)) {
    return 'UnknownFailure';
  }

  return ['Error', 'TypeError', 'OpenRouterManagementError'].includes(error.name)
    ? error.name
    : 'Error';
}
