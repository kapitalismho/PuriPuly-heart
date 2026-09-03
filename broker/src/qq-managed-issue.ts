import type { Context } from 'hono';

import {
  checkActiveIssuanceBrake,
  extractRequestNetworkMetadata,
  getManagedDailyIssuanceCapState,
  resolveRequestNetworkIdentitySecrets,
  type AbuseDecision,
} from './abuse-controls';
import {
  attachReferralToOperation,
  bindOperationForIssue,
  buildManagedOperationStatusBodyWithDelivery,
  failManagedOperationTerminal,
  markOperationActiveOnAck,
  type ManagedOperationRecord as StrictManagedOperationRecord,
  getManagedOperationStatusSnapshot,
  listManagedOperationAttempts,
  markAttemptUnknown,
  operationBindingResponseBody,
  reconcileUnknownAttempt,
  providerKeyNameForOperationAttempt,
  recordAttemptCredential,
  startManagedOperationAttempt,
  transitionManagedOperation,
} from './managed-operation';
import {
  deliverManagedCleanupIncident,
  deliverImmediateMonitoringSideEffects,
  evaluateImmediateAbuseState,
  prepareIssueSuccessInsert,
  recordIssueSuccess,
} from './abuse-monitoring';
import {
  errorResponse as publicErrorResponse,
  internalErrorResponse,
} from './broker-error';
import type { BrokerBindings, BrokerEnv } from './contract';
import {
  buildManagedCleanupRequiredAuditPayload,
  getManagedIssuanceSourcePolicy,
} from './managed-issuance';
import type { TalkTogetherPassStatusResponse } from './managed-state';
import { createManagedKeyDelivery } from './managed-key-delivery';
import {
  assignManagedGuardrail,
  cleanupManagedChildKey,
  createManagedChildKey,
  isDefinitiveManagedChildKeyCreateRejection,
  OpenRouterManagementError,
  type ManagedChildKeyCleanupResult,
} from './openrouter-management';
import { deriveManagedOpenRouterUserId } from './openrouter-user-id';
import {
  QQ_MANAGED_ENTITLEMENT_STALE_ISSUING_POLICY,
  type BrokerQqTalkTogetherPassConfigValue,
  type QqManagedEntitlementRecord,
} from './persistence';
import {
  countCountedQqReferralRewardsSince,
  ensureOwnedReferralIdForActiveQqManagedUser,
  getOperationReferralReward,
  getReservedReferralRewardIdForOperation,
  markReservedIssueReferralFailed,
  recordSkippedIssueReferralReward,
  reserveIssueReferralReward,
  resolveOwnedReferralStatusForManagedSubject,
  resolveTalkTogetherPassStatusForOwnedReferralCode,
  type IssueReferralReservationResult,
  type ReferralAttemptIpDigest,
} from './referral';
import { resolveNetworkIdentityWriteMode } from './network-identity';
import {
  getQqTalkTogetherPassConfig,
  qqReferralUtcDayStartIso,
} from './qq-talk-together-pass';
import { nonEmptyString } from './public-input';
import { MANAGED_TRIAL_POLICY } from './trial-policy';

const QQ_ISSUE_REF_PREFIX = 'qq-issue-v1_';
const ISSUE_SOURCE = 'qq' as const;
const MANAGED_KEY_DELIVERY_ACK_TTL_MS = 15 * 60_000;

interface QqManagedIssueInput {
  qqSubjectRef: string;
  now: Date;
  deliveryAckSupported?: boolean;
  referralId: string | null;
  referredInstallationId: string | null;
  attemptIpDigest: ReferralAttemptIpDigest | null;
  attemptIpLegacyHash: string | null;
  operationId: string | null;
  resumeToken: string | null;
  passConfig: BrokerQqTalkTogetherPassConfigValue;
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
  let childKeyCreationStarted = false;
  let referralReservation: IssueReferralReservationResult | null = null;
  const issueBudgetUsd = sourcePolicy.budget_usd;
  let boundOperation: StrictManagedOperationRecord | null = null;
  let boundAttemptIndex: number | null = null;

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

    const operationBinding = input.operationId
      ? await bindOperationForIssue(c.env.BROKER_DB, {
          operationId: input.operationId,
          resumeToken: input.resumeToken,
          issueSource: ISSUE_SOURCE,
          subjectRef: input.qqSubjectRef,
          installationId: input.referredInstallationId,
          devicePublicKey: null,
          now: input.now,
        })
      : null;
    if (operationBinding && operationBinding.status !== 'proceed') {
      if (operationBinding.status === 'invalid') {
        return publicErrorResponse(c, 400, {
          code: 'invalid_request',
          class: 'terminal',
          subcode: `operation_${operationBinding.reason}`,
          retryAfterMs: null,
          message: `managed operation binding ${operationBinding.reason}`,
          entitlement: null,
        });
      }
      const attempts = await listManagedOperationAttempts(c.env.BROKER_DB, operationBinding.operation.operation_id);
      return c.json(operationBindingResponseBody(operationBinding.operation, attempts));
    }
    boundOperation = operationBinding && operationBinding.status === 'proceed' ? operationBinding.operation : null;
    if (boundOperation) {
      const started = await startManagedOperationAttempt(c.env.BROKER_DB, boundOperation, input.now);
      if (!started.ok) {
        const attempts = await listManagedOperationAttempts(c.env.BROKER_DB, boundOperation.operation_id);
        const current = await getManagedOperationStatusSnapshot(c.env.BROKER_DB, boundOperation.operation_id);
        return c.json(operationBindingResponseBody(current ?? boundOperation, attempts));
      }
      boundAttemptIndex = started.attempt.attempt_index;
    }
    const reservation = await reserveQqManagedEntitlement(c.env, {
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

    if (input.referralId && input.deliveryAckSupported === true) {
      referralReservation = await bestEffortReserveQqIssueReferralReward(
        c.env.BROKER_DB,
        {
          referralId: input.referralId,
          qqSubjectRef: input.qqSubjectRef,
          referredInstallationId: input.referredInstallationId,
          attemptIpDigest: input.attemptIpDigest ?? null,
          passConfig: input.passConfig,
          now: input.now,
          nowIso,
        },
      );
      if (referralReservation?.outcome === 'reserved') {
        await bestEffortWarnQqReferralDailyThreshold(c.env.BROKER_DB, {
          passConfig: input.passConfig,
          now: input.now,
        });
      }
    }

    if (boundOperation) {
      const rewardId = await getReservedReferralRewardIdForOperation(c.env.BROKER_DB, boundOperation.operation_id);
      await attachReferralToOperation(
        c.env.BROKER_DB,
        boundOperation.operation_id,
        rewardId,
        referralReservation?.outcome === 'reserved' ? 'reserved' : referralReservation?.outcome === 'skipped' ? 'skipped' : 'none',
        'none',
        input.now,
      );
    }
    const issuedAt = nowIso;
    const deliveredAt = nowIso;
    const expiresAt = addMonthsUtc(
      input.now,
      MANAGED_TRIAL_POLICY.entitlement.issuance.expiry.durationMonths,
    ).toISOString();

    childKeyCreationStarted = await markQqChildKeyCreationStarted(
      c.env.BROKER_DB,
      {
        qqSubjectRef: input.qqSubjectRef,
        issueRef: issueMetadata.issueRef,
        budgetUsd: issueBudgetUsd,
        nowIso,
      },
    );
    if (!childKeyCreationStarted) {
      throw new Error('QQ managed child key creation start persistence failed');
    }

    childKey = await createManagedChildKey({
      managementApiKey: c.env.OPENROUTER_MANAGEMENT_API_KEY,
      issueSource: ISSUE_SOURCE,
      subjectRef: input.qqSubjectRef,
      issueRef: issueMetadata.issueRef,
      expiresAt,
      limitUsd: issueBudgetUsd,
      ...(boundOperation && boundAttemptIndex !== null
        ? {
            keyName: providerKeyNameForOperationAttempt(
              boundOperation.operation_id,
              'qq',
              boundAttemptIndex,
            ),
          }
        : {}),
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
    if (boundOperation && boundAttemptIndex !== null) {
      await recordAttemptCredential(c.env.BROKER_DB, boundOperation.operation_id, boundAttemptIndex, childKey.hash, input.now);
    }

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
        budgetUsd: issueBudgetUsd,
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
        installationId: input.referredInstallationId,
        managedCredentialRef: childKey.hash,
        createdAt: input.now,
        expiresAt: deliveryAckExpiresAt,
        operationId: boundOperation?.operation_id ?? null,
        attemptIndex: boundAttemptIndex,
      });
      if (boundOperation) {
        await transitionManagedOperation(c.env.BROKER_DB, boundOperation.operation_id, 'DELIVERY_PENDING', input.now);
      }
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
      budgetUsd: issueBudgetUsd,
      issuedAt,
      expiresAt,
      deliveredAt,
    });
    if (!activated) {
      throw new Error('QQ managed entitlement activation failed');
    }
    if (boundOperation) {
      await transitionManagedOperation(c.env.BROKER_DB, boundOperation.operation_id, 'ACTIVE', input.now);
    }

    await runQqIssueSuccessMonitoring(c, {
      qqSubjectRef: input.qqSubjectRef,
      managedCredentialRef: childKey.hash,
      observedAt: issuedAt,
      now: input.now,
    });

    const ownedStatus = await bestEffortResolveQqOwnedReferralStatus(
      c.env.BROKER_DB,
      {
        qqSubjectRef: input.qqSubjectRef,
        ownerInstallationId: input.referredInstallationId,
        passEnabled: input.passConfig.enabled,
        nowIso,
      },
    );
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
      ...(ownedStatus
        ? {
            referral_id: ownedStatus.referralCode.referral_id,
            talk_together_pass: ownedStatus.talkTogetherPass,
          }
        : {}),
    });
  } catch (error) {
    if (
      !childKey &&
      error instanceof OpenRouterManagementError &&
      error.createdChildKey
    ) {
      childKey = error.createdChildKey;
    }

    if (boundOperation && boundAttemptIndex !== null) {
      if (childKey) {
        await recordAttemptCredential(c.env.BROKER_DB, boundOperation.operation_id, boundAttemptIndex, childKey.hash, input.now);
      }
      await markAttemptUnknown(c.env.BROKER_DB, boundOperation.operation_id, boundAttemptIndex, input.now);
    }
    let boundProviderCleanupVerified = false;
    if (boundOperation && isDefinitiveManagedChildKeyCreateRejection(error)) {
      await failManagedOperationTerminal(c.env.BROKER_DB, boundOperation, input.now, 'terminal_provider_failure');
    } else if (boundOperation && boundAttemptIndex !== null) {
      const reconciled = await reconcileUnknownAttempt(c.env.BROKER_DB, c.env.OPENROUTER_MANAGEMENT_API_KEY, boundOperation, input.now);
      boundProviderCleanupVerified = reconciled?.state === 'RETRY_READY' || reconciled?.state === 'CLEAN';
    }
    if (!boundOperation) {
      await bestEffortMarkQqIssueReferralFailed(c.env.BROKER_DB, {
        referralReservation,
        qqSubjectRef: input.qqSubjectRef,
        referredInstallationId: input.referredInstallationId,
        nowIso,
      });
    }

    if (reservationCreated) {
      if (!childKey) {
        const definitiveCreateRejection =
          isDefinitiveManagedChildKeyCreateRejection(error);
        if (
          !childKeyCreationStarted ||
          definitiveCreateRejection ||
          boundOperation !== null
        ) {
          const release = childKeyCreationStarted
            ? releaseStartedQqReservationWithoutChildKey
            : bestEffortReleaseQqReservationBeforeChildKey;
          await release(c.env.BROKER_DB, {
            qqSubjectRef: input.qqSubjectRef,
            issueRef: issueMetadata.issueRef,
          });
        } else {
          await deliverManagedCleanupIncident(c.env, {
            issueSource: ISSUE_SOURCE,
            managedCredentialRef: null,
            phase: 'managed_issue',
            cleanupRequiredRecorded: false,
            occurredAt: nowIso,
          });
        }
      } else {
        await handleQqManagedChildKeyFailure(c, {
          qqSubjectRef: input.qqSubjectRef,
          issueRef: issueMetadata.issueRef,
          childKey,
          childKeyAttached,
          nowIso,
          error,
          providerCleanupHandled: boundProviderCleanupVerified,
        });
      }
    }

    return internalErrorResponse(c);
  }
}

export async function executeQqResumeIssuance(
  c: Context<BrokerEnv>,
  input: {
    operation: StrictManagedOperationRecord;
    attemptIndex: number;
    hasLiveDelivery: boolean;
    now: Date;
    nowIso: string;
  },
): Promise<Response> {
  const db = c.env.BROKER_DB;
  const operation = input.operation;
  const qqSubjectRef = operation.subject_ref;
  const referredInstallationId = operation.installation_id;
  const passConfig = await getQqTalkTogetherPassConfig(db);
  const sourcePolicy = getManagedIssuanceSourcePolicy(ISSUE_SOURCE);
  const issueBudgetUsd = sourcePolicy.budget_usd;
  const issueRef = createIssueRef();
  let reservationCreated = false;
  let childKey: { rawKey: string; hash: string } | null = null;
  let childKeyAttached = false;
  let childKeyCreationStarted = false;
  const staleEntitlement = await getQqManagedEntitlement(db, qqSubjectRef);
  if (
    staleEntitlement?.status === 'delivery_pending' &&
    staleEntitlement.managed_credential_ref &&
    !input.hasLiveDelivery
  ) {
    await recordAttemptCredential(
      db,
      operation.operation_id,
      input.attemptIndex,
      staleEntitlement.managed_credential_ref,
      input.now,
    );
    const staleCleanup = await cleanupManagedChildKey({
      managementApiKey: c.env.OPENROUTER_MANAGEMENT_API_KEY,
      keyHash: staleEntitlement.managed_credential_ref,
    });
    if (!staleCleanup.ok) {
      await markAttemptUnknown(db, operation.operation_id, input.attemptIndex, input.now);
      const attempts = await listManagedOperationAttempts(db, operation.operation_id);
      const current = await getManagedOperationStatusSnapshot(db, operation.operation_id);
      return c.json(await buildManagedOperationStatusBodyWithDelivery(db, current ?? operation, attempts));
    }
    const reset = await resetStaleQqReservationForRetry(db, {
      qqSubjectRef,
      managedCredentialRef: staleEntitlement.managed_credential_ref,
      nowIso: input.nowIso,
    });
    if (!reset) {
      await markAttemptUnknown(db, operation.operation_id, input.attemptIndex, input.now);
      const attempts = await listManagedOperationAttempts(db, operation.operation_id);
      const current = await getManagedOperationStatusSnapshot(db, operation.operation_id);
      return c.json(await buildManagedOperationStatusBodyWithDelivery(db, current ?? operation, attempts));
    }
  }
  try {
    const currentEntitlement = await getQqManagedEntitlement(db, qqSubjectRef);
    if (currentEntitlement && isLifetimeBlockingQqEntitlement(currentEntitlement)) {
      await markAttemptUnknown(db, operation.operation_id, input.attemptIndex, input.now);
      const attempts = await listManagedOperationAttempts(db, operation.operation_id);
      const current = await getManagedOperationStatusSnapshot(db, operation.operation_id);
      return c.json(await buildManagedOperationStatusBodyWithDelivery(db, current ?? operation, attempts));
    }
    const brakeDecision = await checkActiveIssuanceBrake(db, currentEntitlement);
    if (brakeDecision) {
      await markAttemptUnknown(db, operation.operation_id, input.attemptIndex, input.now);
      return abuseDecisionResponse(c, brakeDecision);
    }
    const reservation = await reserveQqManagedEntitlement(c.env, {
      qqSubjectRef,
      issueRef,
      budgetUsd: sourcePolicy.budget_usd,
      now: input.now,
      nowIso: input.nowIso,
    });
    if (!reservation.ok) {
      await markAttemptUnknown(db, operation.operation_id, input.attemptIndex, input.now);
      if (reservation.reason === 'lifetime_used') {
        await failManagedOperationTerminal(db, operation, input.now, 'terminal_provider_failure');
        const attempts = await listManagedOperationAttempts(db, operation.operation_id);
        const terminal = await getManagedOperationStatusSnapshot(db, operation.operation_id);
        return c.json(await buildManagedOperationStatusBodyWithDelivery(db, terminal ?? operation, attempts));
      }
      if (reservation.reason === 'already_issuing') {
        return qqReservationErrorResponse(c, reservation);
      }
      const attempts = await listManagedOperationAttempts(db, operation.operation_id);
      const current = await getManagedOperationStatusSnapshot(db, operation.operation_id);
      return c.json(await buildManagedOperationStatusBodyWithDelivery(db, current ?? operation, attempts));
    }
    reservationCreated = true;
    const cap = await getManagedDailyIssuanceCapState(db, input.now, {
      excludeCurrent: { issueSource: ISSUE_SOURCE, subjectRef: qqSubjectRef, issueRef },
    });
    if (cap.reached) {
      await releaseQqReservationBeforeChildKey(db, { qqSubjectRef, issueRef });
      reservationCreated = false;
      await markAttemptUnknown(db, operation.operation_id, input.attemptIndex, input.now);
      return publicErrorResponse(c, 503, {
        code: 'issuance_suspended',
        class: 'retryable',
        subcode: 'global_cap_reached',
        retryAfterMs: cap.retryAfterMs,
        message: 'Daily managed issuance cap reached',
        entitlement: null,
      });
    }
    const referralReservation = await getOperationReferralReward(db, operation.operation_id);
    if (operation.referral_status === 'reserved') {
      const rewardId = await getReservedReferralRewardIdForOperation(db, operation.operation_id);
      await attachReferralToOperation(db, operation.operation_id, rewardId, 'reserved', 'none', input.now);
    }
    const issuedAt = input.nowIso;
    const expiresAt = addMonthsUtc(
      input.now,
      MANAGED_TRIAL_POLICY.entitlement.issuance.expiry.durationMonths,
    ).toISOString();
    childKeyCreationStarted = await markQqChildKeyCreationStarted(db, {
      qqSubjectRef,
      issueRef,
      budgetUsd: issueBudgetUsd,
      nowIso: input.nowIso,
    });
    if (!childKeyCreationStarted) {
      throw new Error('QQ managed child key creation start persistence failed');
    }
    childKey = await createManagedChildKey({
      managementApiKey: c.env.OPENROUTER_MANAGEMENT_API_KEY,
      issueSource: ISSUE_SOURCE,
      subjectRef: qqSubjectRef,
      issueRef,
      expiresAt,
      limitUsd: issueBudgetUsd,
      keyName: providerKeyNameForOperationAttempt(operation.operation_id, 'qq', input.attemptIndex),
    });
    const attached = await attachManagedCredentialToQqReservation(db, {
      qqSubjectRef,
      issueRef,
      managedCredentialRef: childKey.hash,
      nowIso: input.nowIso,
    });
    if (!attached) {
      throw new Error('QQ managed child key reservation attachment failed');
    }
    childKeyAttached = true;
    await recordAttemptCredential(db, operation.operation_id, input.attemptIndex, childKey.hash, input.now);
    await assignManagedGuardrail({
      managementApiKey: c.env.OPENROUTER_MANAGEMENT_API_KEY,
      guardrailId: c.env.OPENROUTER_MANAGED_GUARDRAIL_ID,
      keyHash: childKey.hash,
    });
    const pending = await markQqReservationDeliveryPending(db, {
      qqSubjectRef,
      issueRef,
      managedCredentialRef: childKey.hash,
      budgetUsd: issueBudgetUsd,
      issuedAt,
      expiresAt,
      nowIso: input.nowIso,
    });
    if (!pending) {
      throw new Error('QQ managed entitlement delivery-pending transition failed');
    }
    const deliveryAckExpiresAt = new Date(input.now.getTime() + MANAGED_KEY_DELIVERY_ACK_TTL_MS);
    const delivery = await createManagedKeyDelivery(db, {
      issueSource: ISSUE_SOURCE,
      subjectRef: qqSubjectRef,
      installationId: referredInstallationId,
      managedCredentialRef: childKey.hash,
      createdAt: input.now,
      expiresAt: deliveryAckExpiresAt,
      operationId: operation.operation_id,
      attemptIndex: input.attemptIndex,
    });
    await transitionManagedOperation(db, operation.operation_id, 'DELIVERY_PENDING', input.now, {
      from: ['CREATING'],
    });
    const settled = await getManagedOperationStatusSnapshot(db, operation.operation_id);
    if (!settled || settled.state !== 'DELIVERY_PENDING') {
      await cleanupManagedChildKey({
        managementApiKey: c.env.OPENROUTER_MANAGEMENT_API_KEY,
        keyHash: childKey.hash,
      }).catch(() => null);
      const attempts = await listManagedOperationAttempts(db, operation.operation_id);
      const current = await getManagedOperationStatusSnapshot(db, operation.operation_id);
      return c.json(await buildManagedOperationStatusBodyWithDelivery(db, current ?? operation, attempts));
    }
    const openRouterUserId = await deriveOptionalOpenRouterUserId({
      subjectRef: qqSubjectRef,
      secret: c.env.OPENROUTER_MANAGED_USER_HMAC_SECRET,
    });
    return c.json({
      ok: true,
      status: 'delivery_pending',
      qq_subject_ref: qqSubjectRef,
      openrouter_api_key: childKey.rawKey,
      managed_credential_ref: childKey.hash,
      expires_at: expiresAt,
      delivery_ack_required: true,
      delivery_id: delivery.deliveryId,
      delivery_ack_token: delivery.deliveryAckToken,
      delivery_ack_expires_at: deliveryAckExpiresAt.toISOString(),
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
    if (childKey) {
      await recordAttemptCredential(db, operation.operation_id, input.attemptIndex, childKey.hash, input.now);
    }
    await markAttemptUnknown(db, operation.operation_id, input.attemptIndex, input.now);
    if (isDefinitiveManagedChildKeyCreateRejection(error)) {
      await failManagedOperationTerminal(db, operation, input.now, 'terminal_provider_failure');
      const attempts = await listManagedOperationAttempts(db, operation.operation_id);
      const terminal = await getManagedOperationStatusSnapshot(db, operation.operation_id);
      return c.json(await buildManagedOperationStatusBodyWithDelivery(db, terminal ?? operation, attempts));
    }
    if (reservationCreated) {
      if (!childKey) {
        const definitiveCreateRejection = isDefinitiveManagedChildKeyCreateRejection(error);
        if (!childKeyCreationStarted || definitiveCreateRejection) {
          const release = childKeyCreationStarted
            ? releaseStartedQqReservationWithoutChildKey
            : bestEffortReleaseQqReservationBeforeChildKey;
          await release(db, { qqSubjectRef, issueRef });
        }
      } else {
        await handleQqManagedChildKeyFailure(c, {
          qqSubjectRef,
          issueRef,
          childKey,
          childKeyAttached,
          nowIso: input.nowIso,
          error,
        });
      }
    }
    const attempts = await listManagedOperationAttempts(db, operation.operation_id);
    const current = await getManagedOperationStatusSnapshot(db, operation.operation_id);
    return c.json(await buildManagedOperationStatusBodyWithDelivery(db, current ?? operation, attempts));
  }
}

async function resetStaleQqReservationForRetry(
  db: D1Database,
  input: { qqSubjectRef: string; managedCredentialRef: string; nowIso: string },
): Promise<boolean> {
  const result = await db
    .prepare(
      `UPDATE qq_managed_entitlements
          SET status = 'issuing',
              managed_credential_ref = NULL,
              issued_at = NULL,
              expires_at = NULL,
              delivered_at = NULL,
              child_key_creation_started_at = NULL,
              updated_at = ?
        WHERE qq_subject_ref = ?
          AND status = 'delivery_pending'
          AND managed_credential_ref = ?`,
    )
    .bind(input.nowIso, input.qqSubjectRef, input.managedCredentialRef)
    .run();
  return Number(result.meta.changes ?? 0) === 1;
}

async function bestEffortReserveQqIssueReferralReward(
  db: D1Database,
  input: {
    referralId: string;
    qqSubjectRef: string;
    referredInstallationId: string | null;
    attemptIpDigest: ReferralAttemptIpDigest | null;
    attemptIpLegacyHash?: string | null;
    operationId?: string | null;
    passConfig: BrokerQqTalkTogetherPassConfigValue;
    now: Date;
    nowIso: string;
  },
): Promise<IssueReferralReservationResult | null> {
  try {
    if (!input.passConfig.enabled) {
      return null;
    }
    if (!input.passConfig.rewards_enabled) {
      return await recordSkippedIssueReferralReward(db, {
        referralId: input.referralId,
        referredSource: 'qq',
        referredSubjectRef: input.qqSubjectRef,
        referredInstallationId: input.referredInstallationId,
        referredHardwareHash: null,
        referredHardwareHashSaltVersion: null,
        skipReason: 'rewards_disabled',
        attemptIpDigest: input.attemptIpDigest ?? null,
        attemptIpLegacyHash: input.attemptIpLegacyHash ?? null,
        nowIso: input.nowIso,
      });
    }
    if (!input.referredInstallationId) {
      return await recordSkippedIssueReferralReward(db, {
        referralId: input.referralId,
        referredSource: 'qq',
        referredSubjectRef: input.qqSubjectRef,
        referredInstallationId: null,
        referredHardwareHash: null,
        referredHardwareHashSaltVersion: null,
        skipReason: 'invalid_installation',
        attemptIpDigest: input.attemptIpDigest ?? null,
        attemptIpLegacyHash: input.attemptIpLegacyHash ?? null,
        nowIso: input.nowIso,
      });
    }
    return await reserveIssueReferralReward(db, {
      referralId: input.referralId,
      referredSource: 'qq',
      referredSubjectRef: input.qqSubjectRef,
      referredInstallationId: input.referredInstallationId,
      referredHardwareHash: null,
      referredHardwareHashSaltVersion: null,
      attemptIpDigest: input.attemptIpDigest ?? null,
      attemptIpLegacyHash: input.attemptIpLegacyHash ?? null,
      operationId: input.operationId ?? null,
      globalCountLimit: input.passConfig.daily_max_count,
      globalCountWindowStartIso: qqReferralUtcDayStartIso(input.now),
      nowIso: input.nowIso,
    });
  } catch {
    return null;
  }
}

async function bestEffortWarnQqReferralDailyThreshold(
  db: D1Database,
  input: { passConfig: BrokerQqTalkTogetherPassConfigValue; now: Date },
): Promise<void> {
  try {
    const count = await countCountedQqReferralRewardsSince(
      db,
      qqReferralUtcDayStartIso(input.now),
    );
    if (count >= input.passConfig.daily_warning_count) {
      console.warn('qq_referral_daily_warning_threshold_reached', {
        counted_rewards: count,
        warning_threshold: input.passConfig.daily_warning_count,
        daily_max_count: input.passConfig.daily_max_count,
        broker_timestamp: input.now.toISOString(),
      });
    }
  } catch {
    return;
  }
}

async function bestEffortMarkQqIssueReferralFailed(
  db: D1Database,
  input: {
    referralReservation: IssueReferralReservationResult | null;
    qqSubjectRef: string;
    referredInstallationId: string | null;
    nowIso: string;
  },
): Promise<void> {
  if (input.referralReservation?.outcome !== 'reserved') {
    return;
  }
  try {
    await markReservedIssueReferralFailed(db, {
      referralId: input.referralReservation.referralId,
      referredSource: 'qq',
      referredSubjectRef: input.qqSubjectRef,
      referredInstallationId: input.referredInstallationId,
      failureReason: 'issue_delivery_failed',
      nowIso: input.nowIso,
    });
  } catch {
    return;
  }
}

async function bestEffortResolveQqOwnedReferralStatus(
  db: D1Database,
  input: {
    qqSubjectRef: string;
    ownerInstallationId: string | null;
    passEnabled: boolean;
    nowIso: string;
  },
) {
  try {
    let status = await resolveOwnedReferralStatusForManagedSubject(db, {
      source: 'qq',
      subjectRef: input.qqSubjectRef,
    });
    if (!status && input.passEnabled) {
      const ensured = await ensureOwnedReferralIdForActiveQqManagedUser(db, {
        qqSubjectRef: input.qqSubjectRef,
        ownerInstallationId: input.ownerInstallationId,
        nowIso: input.nowIso,
      });
      if (ensured.ok) {
        status = {
          referralCode: ensured.referralCode,
          talkTogetherPass: await resolveTalkTogetherPassStatusForOwnedReferralCode(
            db,
            ensured.referralCode,
          ),
        };
      }
    }
    return status;
  } catch {
    return null;
  }
}

async function reserveQqManagedEntitlement(
  env: Pick<
    BrokerBindings,
    'BROKER_DB' | 'DISCORD_IMMEDIATE_ALERT_WEBHOOK_URL'
  >,
  input: {
    qqSubjectRef: string;
    issueRef: string;
    budgetUsd: number;
    now: Date;
    nowIso: string;
  },
): Promise<QqReservationResult> {
  const db = env.BROKER_DB;
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
    let cleanupRequiredRecorded = false;
    let stateUpdateFailed = false;
    try {
      cleanupRequiredRecorded = await markStaleQqIssuingCleanupRequired(db, {
        qqSubjectRef: input.qqSubjectRef,
        issueRef: current.issue_ref,
        managedCredentialRef: current.managed_credential_ref,
        nowIso: input.nowIso,
      });
    } catch {
      stateUpdateFailed = true;
    }
    if (cleanupRequiredRecorded || stateUpdateFailed) {
      await deliverManagedCleanupIncident(env, {
        issueSource: 'qq',
        managedCredentialRef: current.managed_credential_ref,
        phase: 'stale_reservation',
        cleanupRequiredRecorded,
        occurredAt: input.nowIso,
      });
    }
    return { ok: false, reason: 'lifetime_used' };
  }

  if (current.child_key_creation_started_at) {
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
              child_key_creation_started_at = NULL,
              updated_at = ?
        WHERE qq_subject_ref = ?
          AND issue_ref = ?
          AND status = 'issuing'
          AND managed_credential_ref IS NULL
          AND child_key_creation_started_at IS NULL`,
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
              child_key_creation_started_at, created_at, updated_at
         FROM qq_managed_entitlements
        WHERE qq_subject_ref = ?`,
    )
    .bind(qqSubjectRef)
    .first<QqManagedEntitlementRecord>();
}

async function markQqChildKeyCreationStarted(
  db: D1Database,
  input: {
    qqSubjectRef: string;
    issueRef: string;
    budgetUsd: number;
    nowIso: string;
  },
): Promise<boolean> {
  const result = await db
    .prepare(
      `UPDATE qq_managed_entitlements
          SET child_key_creation_started_at = ?, budget_usd = ?, updated_at = ?
        WHERE qq_subject_ref = ?
          AND issue_ref = ?
          AND status = 'issuing'
          AND managed_credential_ref IS NULL
          AND child_key_creation_started_at IS NULL`,
    )
    .bind(
      input.nowIso,
      input.budgetUsd,
      input.nowIso,
      input.qqSubjectRef,
      input.issueRef,
    )
    .run();
  return Number(result.meta.changes ?? 0) === 1;
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
          AND managed_credential_ref IS NULL
          AND child_key_creation_started_at IS NOT NULL`,
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
  input: {
    deliveryId: string;
    managedCredentialRef: string;
    acknowledgedAt: Date;
  },
): Promise<{
  acknowledgementStatus: 'acknowledged' | 'already_acknowledged';
  referralId?: string;
  talkTogetherPass?: TalkTogetherPassStatusResponse;
}> {
  const entitlement = await c.env.BROKER_DB.prepare(
    `SELECT qq_subject_ref, status, issue_ref, managed_credential_ref,
            budget_usd, reserved_at, issued_at, expires_at, delivered_at,
            child_key_creation_started_at, created_at, updated_at
       FROM qq_managed_entitlements
      WHERE managed_credential_ref = ?`,
  )
    .bind(input.managedCredentialRef)
    .first<QqManagedEntitlementRecord>();
  if (!entitlement) {
    throw new Error('QQ delivery ACK target is missing');
  }
  const delivery = await c.env.BROKER_DB.prepare(
    `SELECT installation_id
       FROM managed_key_deliveries
      WHERE delivery_id = ?
        AND issue_source = 'qq'
        AND managed_credential_ref = ?`,
  )
    .bind(input.deliveryId, input.managedCredentialRef)
    .first<{ installation_id: string | null }>();
  if (!delivery) {
    throw new Error('QQ delivery ACK metadata is missing');
  }
  const alreadyActive = entitlement.status === 'active';
  if (!alreadyActive && entitlement.status !== 'delivery_pending') {
    throw new Error('QQ delivery ACK target is not pending');
  }
  if (!entitlement.issued_at || !entitlement.expires_at) {
    throw new Error('QQ delivery ACK target is missing entitlement metadata');
  }

  const acknowledgedAtIso = input.acknowledgedAt.toISOString();
  const deliveredAt = entitlement.delivered_at ?? acknowledgedAtIso;
  const network = await extractRequestNetworkMetadata(c, { secrets: resolveRequestNetworkIdentitySecrets(c), now: new Date() });
  const netMode = await resolveNetworkIdentityWriteMode(c.env.BROKER_DB);
  const results = await c.env.BROKER_DB.batch([
    c.env.BROKER_DB.prepare(
      `UPDATE qq_managed_entitlements
          SET status = 'active',
              delivered_at = ?,
              updated_at = ?
        WHERE qq_subject_ref = ?
          AND issue_ref = ?
          AND status = 'delivery_pending'
          AND managed_credential_ref = ?
          AND EXISTS (
            SELECT 1 FROM managed_key_deliveries
             WHERE delivery_id = ?
               AND managed_credential_ref = ?
               AND status = 'pending'
          )`,
    ).bind(
      deliveredAt,
      deliveredAt,
      entitlement.qq_subject_ref,
      entitlement.issue_ref,
      input.managedCredentialRef,
      input.deliveryId,
      input.managedCredentialRef,
    ),
    prepareIssueSuccessInsert(c.env.BROKER_DB, {
      issueSource: ISSUE_SOURCE,
      subjectRef: entitlement.qq_subject_ref,
      managedCredentialRef: input.managedCredentialRef,
      observedAt: deliveredAt,
      network,
      deliveryId: input.deliveryId,
    }, netMode),
    c.env.BROKER_DB.prepare(
      `UPDATE managed_key_deliveries
          SET status = 'acknowledged', acknowledged_at = ?
        WHERE delivery_id = ?
          AND managed_credential_ref = ?
          AND status = 'pending'
          AND EXISTS (
            SELECT 1 FROM qq_managed_entitlements
             WHERE qq_subject_ref = ?
               AND issue_ref = ?
               AND managed_credential_ref = ?
               AND status = 'active'
               AND delivered_at IS NOT NULL
          )
          AND EXISTS (
            SELECT 1 FROM broker_issue_success_events
             WHERE issue_source = 'qq'
               AND managed_credential_ref = ?
          )`,
    ).bind(
      acknowledgedAtIso,
      input.deliveryId,
      input.managedCredentialRef,
      entitlement.qq_subject_ref,
      entitlement.issue_ref,
      input.managedCredentialRef,
      input.managedCredentialRef,
    ),
    c.env.BROKER_DB.prepare(
      `INSERT INTO managed_referral_settlement_jobs (
          source,
          referral_reward_id,
          delivery_id,
          operation_id,
          phase,
          attempt_count,
          last_attempt_at,
          next_attempt_at,
          fencing_token,
          lease_expires_at,
          last_error_code,
          created_at,
          updated_at,
          completed_at
        )
        SELECT 'qq',
               reward.id,
               delivery.delivery_id,
               reward.operation_id,
               'invitee_pending',
               0,
               NULL,
               ?,
               NULL,
               NULL,
               NULL,
               ?,
               ?,
               NULL
          FROM referral_rewards reward
          JOIN managed_key_deliveries delivery
            ON delivery.delivery_id = ?
           AND delivery.issue_source = 'qq'
           AND delivery.subject_ref = reward.referred_subject_ref
           AND delivery.installation_id IS reward.referred_installation_id
           AND delivery.managed_credential_ref = ?
           AND delivery.status = 'acknowledged'
         WHERE reward.id = (
           SELECT candidate.id
             FROM referral_rewards candidate
            WHERE candidate.referred_source = 'qq'
              AND candidate.referred_subject_ref = ?
              AND candidate.referred_installation_id IS ?
              AND candidate.referred_bonus_status = 'reserved'
            ORDER BY candidate.created_at DESC, candidate.id DESC
            LIMIT 1
         )
        ON CONFLICT(referral_reward_id) DO NOTHING`,
    ).bind(
      acknowledgedAtIso,
      acknowledgedAtIso,
      acknowledgedAtIso,
      input.deliveryId,
      input.managedCredentialRef,
      entitlement.qq_subject_ref,
      delivery.installation_id,
    ),
  ]);
  const finalized = await c.env.BROKER_DB.prepare(
    `SELECT 1 AS finalized
       FROM managed_key_deliveries AS delivery
       JOIN qq_managed_entitlements AS entitlement
         ON entitlement.managed_credential_ref = delivery.managed_credential_ref
      WHERE delivery.delivery_id = ?
        AND delivery.status = 'acknowledged'
        AND entitlement.status = 'active'
        AND entitlement.delivered_at IS NOT NULL
        AND EXISTS (
          SELECT 1 FROM broker_issue_success_events
           WHERE issue_source = 'qq'
             AND managed_credential_ref = delivery.managed_credential_ref
        )
        AND NOT EXISTS (
          SELECT 1
            FROM referral_rewards reward
           WHERE reward.referred_source = 'qq'
             AND reward.referred_subject_ref = entitlement.qq_subject_ref
             AND reward.referred_installation_id IS delivery.installation_id
             AND reward.referred_bonus_status = 'reserved'
             AND NOT EXISTS (
               SELECT 1
                 FROM managed_referral_settlement_jobs job
                WHERE job.referral_reward_id = reward.id
                  AND job.delivery_id = delivery.delivery_id
             )
        )`,
  )
    .bind(input.deliveryId)
    .first<{ finalized: number }>();
  if (Number(finalized?.finalized ?? 0) !== 1) {
    throw new Error('QQ delivery ACK finalization failed');
  }
  const operationActivated = await markOperationActiveOnAck(c.env.BROKER_DB, input.deliveryId, input.acknowledgedAt);
  if (!operationActivated) {
    throw new Error('Managed operation ACK activation failed');
  }

  try {
    await runQqIssueSuccessMonitoring(c, {
      qqSubjectRef: entitlement.qq_subject_ref,
      managedCredentialRef: input.managedCredentialRef,
      observedAt: deliveredAt,
      now: input.acknowledgedAt,
    });
  } catch {
  }

  return buildQqDeliveryAckResult(c, {
    acknowledgementStatus:
      Number(results[2]?.meta.changes ?? 0) === 1
        ? 'acknowledged'
        : 'already_acknowledged',
    qqSubjectRef: entitlement.qq_subject_ref,
    ownerInstallationId: delivery.installation_id,
    nowIso: deliveredAt,
  });
}

async function buildQqDeliveryAckResult(
  c: Context<BrokerEnv>,
  input: {
    acknowledgementStatus: 'acknowledged' | 'already_acknowledged';
    qqSubjectRef: string;
    ownerInstallationId: string | null;
    nowIso: string;
  },
): Promise<{
  acknowledgementStatus: 'acknowledged' | 'already_acknowledged';
  referralId?: string;
  talkTogetherPass?: TalkTogetherPassStatusResponse;
}> {
  try {
    const passConfig = await getQqTalkTogetherPassConfig(c.env.BROKER_DB);
    const ownedStatus = await bestEffortResolveQqOwnedReferralStatus(
      c.env.BROKER_DB,
      {
        qqSubjectRef: input.qqSubjectRef,
        ownerInstallationId: input.ownerInstallationId,
        passEnabled: passConfig.enabled,
        nowIso: input.nowIso,
      },
    );
    return {
      acknowledgementStatus: input.acknowledgementStatus,
      ...(ownedStatus
        ? {
            referralId: ownedStatus.referralCode.referral_id,
            talkTogetherPass: ownedStatus.talkTogetherPass,
          }
        : {}),
    };
  } catch {
    return { acknowledgementStatus: input.acknowledgementStatus };
  }
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
          AND managed_credential_ref IS NULL
          AND child_key_creation_started_at IS NULL`,
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

async function releaseStartedQqReservationWithoutChildKey(
  db: D1Database,
  input: { qqSubjectRef: string; issueRef: string },
): Promise<void> {
  await db
    .prepare(
      `DELETE FROM qq_managed_entitlements
        WHERE qq_subject_ref = ?
          AND issue_ref = ?
          AND status = 'issuing'
          AND managed_credential_ref IS NULL
          AND child_key_creation_started_at IS NOT NULL`,
    )
    .bind(input.qqSubjectRef, input.issueRef)
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
): Promise<boolean> {
  const result = await db
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
  return Number(result.meta.changes ?? 0) === 1;
}

async function markUnattachedQqCleanupRequired(
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
          SET status = 'cleanup_required',
              managed_credential_ref = ?,
              issued_at = NULL,
              expires_at = NULL,
              delivered_at = NULL,
              updated_at = ?
        WHERE qq_subject_ref = ?
          AND issue_ref = ?
          AND status = 'issuing'
          AND managed_credential_ref IS NULL
          AND child_key_creation_started_at IS NOT NULL`,
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

async function markStaleQqIssuingCleanupRequired(
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
  return Number(result.meta.changes ?? 0) === 1;
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
    providerCleanupHandled?: boolean;
  },
): Promise<void> {
  const cleanup = input.providerCleanupHandled
    ? { ok: true as const }
    : await cleanupManagedChildKey({
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
        await releaseStartedQqReservationWithoutChildKey(c.env.BROKER_DB, {
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
      await deliverManagedCleanupIncident(c.env, {
        issueSource: 'qq',
        managedCredentialRef: input.childKey.hash,
        phase: 'managed_issue',
        cleanupRequiredRecorded: false,
        occurredAt: input.nowIso,
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
  let cleanupRequiredRecorded = false;
  let cleanupStateUpdateFailed = false;
  try {
    if (input.childKeyAttached) {
      cleanupRequiredRecorded = await markQqCleanupRequired(c.env.BROKER_DB, {
        qqSubjectRef: input.qqSubjectRef,
        issueRef: input.issueRef,
        managedCredentialRef: input.childKey.hash,
        nowIso: input.nowIso,
      });
    } else {
      cleanupRequiredRecorded = await markUnattachedQqCleanupRequired(c.env.BROKER_DB, {
        qqSubjectRef: input.qqSubjectRef,
        issueRef: input.issueRef,
        managedCredentialRef: input.childKey.hash,
        nowIso: input.nowIso,
      });
    }
  } catch (markError) {
    cleanupStateUpdateFailed = true;
    logQqCleanupStateUpdateFailure({
      qqSubjectRef: input.qqSubjectRef,
      issueRef: input.issueRef,
      managedCredentialRef: input.childKey.hash,
      childKeyAttached: input.childKeyAttached,
      error: markError,
      nowIso: input.nowIso,
    });
  }
  if (cleanupRequiredRecorded || cleanupStateUpdateFailed) {
    await deliverManagedCleanupIncident(c.env, {
      issueSource: 'qq',
      managedCredentialRef: input.childKey.hash,
      phase: 'managed_issue',
      cleanupRequiredRecorded,
      occurredAt: input.nowIso,
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
    const network = await extractRequestNetworkMetadata(c, { secrets: resolveRequestNetworkIdentitySecrets(c), now: new Date() });
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
