import { getBrokerAbuseControlsConfig } from './abuse-controls';
import { resolveNetworkIdentityWriteMode } from './network-identity';
import type { TalkTogetherPassStatusResponse } from './managed-state';
import { resolveEffectiveEntitlementLifecycle } from './managed-state';
import {
  readManagedChildKeyEffectiveLimit,
  updateManagedChildKeyLimit,
} from './openrouter-management';
import type {
  BrokerAbuseControlsConfigValue,
  OpenRouterEntitlementRecord,
  ReferralCodeRecord,
  ReferralRewardRecord,
  ReferralSource,
} from './persistence';
import { MANAGED_TRIAL_BUDGET_POLICY } from './trial-policy';

export const REFERRAL_ID_LENGTH = 6;
export const REFERRAL_ID_ALPHABET = '23456789ABCDEFGHJKMNPQRSTUVWXYZ';
export const TALK_TOGETHER_PASS_INVITE_LIMIT = 3;
export const TALK_TOGETHER_PASS_BONUS_TRANSLATIONS_PER_FRIEND = 200;

const REFERRAL_ID_PATTERN = new RegExp(
  `^[${REFERRAL_ID_ALPHABET}]{${REFERRAL_ID_LENGTH}}$`,
  'u',
);
const REFERRAL_RANDOM_REJECTION_THRESHOLD =
  Math.floor(256 / REFERRAL_ID_ALPHABET.length) * REFERRAL_ID_ALPHABET.length;
const REFERRAL_ID_MAX_RANDOM_DRAWS = 64;
const OWNED_DISCORD_USER_REF_PATTERN = /^ph-discord-user-v\d+_[A-Za-z0-9_-]{32,128}$/u;
const OWNED_QQ_SUBJECT_REF_PATTERN = /^ph-qq-subject-v1_[A-Za-z0-9_-]{32,128}$/u;
const DEFAULT_REFERRAL_ID_COLLISION_ATTEMPTS = 12;
const USD_CENTS = 100;
const REFERRER_REFERRAL_REWARD_CENTS = 2;
const REFERRER_APPLYING_LEASE_MS = 5 * 60 * 1000;
const REFERRER_REWARD_DRAIN_ATTEMPTS = 6;
const DEFAULT_STALE_RESERVED_RECONCILE_MS = 15 * 60 * 1000;
const DEFAULT_STALE_APPLYING_RECONCILE_MS = REFERRER_APPLYING_LEASE_MS;
const REFERRAL_REWARD_LOG_EVENT = 'referral_reward_outcome';

const ISSUE_REFERRAL_SKIP_REASONS = [
  'unknown_referral_id',
  'disabled_referral_id',
  'self_referral',
  'self_or_cross_source_installation',
  'duplicate_hardware',
  'invalid_installation',
  'rewards_disabled',
  'global_reward_cap_reached',
  'referred_already_rewarded',
  'referrer_cap_reached',
  'referrer_not_eligible',
  'referred_not_first_successful',
  'pre_existing_managed_user',
  'reservation_conflict',
  'referral_attempt_rate_limited',
  'unknown_referral_id_rate_limited',
  'referral_velocity_limited',
  'referrer_velocity_limited',
] as const;

const ISSUE_REFERRAL_FAILURE_REASONS = [
  'issue_delivery_failed',
  'referrer_patch_failed',
  'stale_reserved_reconciled',
] as const;

const REFERRAL_DISABLE_REASONS = [
  'abuse',
  'compromised',
  'operator_request',
  'policy_violation',
] as const;
const REFERRAL_DISABLE_ACTOR_PATTERN = /^[A-Za-z0-9._:-]{1,64}$/u;

export type ReferralIdRandomBytes = (byteLength: number) => Uint8Array;
export type ReferralIdGenerator = () => string;

export type ReferrerRewardLimitUpdateResult =
  | {
      outcome: 'not_applicable';
      reason: 'no_referrer_reward_rows' | 'no_pending_referrer_rewards';
    }
  | {
      outcome: 'applying';
      reason: 'active_lease';
    }
  | {
      outcome: 'skipped';
      reason: 'referrer_managed_key_missing';
      skippedRows: number;
    }
  | {
      outcome: 'failed';
      reason: 'referrer_patch_failed';
      failedRows: number;
    }
  | {
      outcome: 'credited';
      creditedRows: number;
      targetLimitUsd: number;
    };

export type IssueReferralSkipReason =
  | 'unknown_referral_id'
  | 'disabled_referral_id'
  | 'self_referral'
  | 'self_or_cross_source_installation'
  | 'duplicate_hardware'
  | 'invalid_installation'
  | 'rewards_disabled'
  | 'global_reward_cap_reached'
  | 'referred_already_rewarded'
  | 'referrer_cap_reached'
  | 'referrer_not_eligible'
  | 'referred_not_first_successful'
  | 'pre_existing_managed_user'
  | 'reservation_conflict'
  | 'referral_attempt_rate_limited'
  | 'unknown_referral_id_rate_limited'
  | 'referral_velocity_limited'
  | 'referrer_velocity_limited';

export type IssueReferralFailureReason =
  | 'issue_delivery_failed'
  | 'referrer_patch_failed'
  | 'stale_reserved_reconciled';

export type ReferralDisableReason = (typeof REFERRAL_DISABLE_REASONS)[number];

export type DisableReferralIdResult =
  | { ok: true; status: 'disabled' | 'already_disabled' }
  | {
      ok: false;
      reason:
        | 'invalid_referral_id'
        | 'invalid_disable_reason'
        | 'invalid_disabled_by'
        | 'not_found';
    };

export interface ReferralRewardRetentionResult {
  skippedDeleted: number;
  failedDeleted: number;
}

export interface StaleReferralRewardReconciliationResult {
  staleReservedCredited: number;
  staleReservedFailed: number;
  staleApplyingRequeued: number;
}

export type IssueReferralReservationResult =
  | {
      outcome: 'not_applicable';
      reason: 'no_referral_input' | 'malformed_referral_input';
    }
  | {
      outcome: 'reserved';
      referralId: string;
    }
  | {
      outcome: 'skipped';
      reason: IssueReferralSkipReason;
    };

export type OwnedReferralIdEnsureFailureReason =
  | 'not_eligible'
  | 'unsafe_subject_ref'
  | 'disabled'
  | 'collision_exhausted';

export type OwnedReferralIdEnsureResult =
  | {
      ok: true;
      referralCode: ReferralCodeRecord;
      created: boolean;
    }
  | {
      ok: false;
      reason: OwnedReferralIdEnsureFailureReason;
    };

export interface ReferralSubject {
  source: ReferralSource;
  subjectRef: string;
  installationId: string | null;
}

export interface ReferralRewardSubject extends ReferralSubject {
  hardwareHash: string | null;
  hardwareHashSaltVersion: number | null;
}

interface ActiveManagedReferralOwner extends ReferralSubject {
  entitlementRef: string;
  managedCredentialRef: string;
  budgetUsd: number;
  expiresAt: string;
}

export function normalizeReferralId(value: unknown): string | null {
  if (typeof value !== 'string') {
    return null;
  }

  const normalized = value.trim().toUpperCase();
  if (!normalized || !REFERRAL_ID_PATTERN.test(normalized)) {
    return null;
  }

  return normalized;
}

export async function getOperationReferralReward(
  db: D1Database,
  operationId: string,
): Promise<IssueReferralReservationResult | null> {
  const row = await db
    .prepare(
      `SELECT referral_id, referred_bonus_status, referrer_bonus_status, skip_reason
         FROM referral_rewards
        WHERE operation_id = ?
        ORDER BY id DESC
        LIMIT 1`,
    )
    .bind(operationId)
    .first<{
      referral_id: string;
      referred_bonus_status: string;
      referrer_bonus_status: string;
      skip_reason: string | null;
    }>()
    .catch(() => null);
  if (!row) {
    return null;
  }
  if (row.referred_bonus_status === 'reserved' || row.referred_bonus_status === 'credited') {
    return { outcome: 'reserved', referralId: row.referral_id };
  }
  if (row.skip_reason && (ISSUE_REFERRAL_SKIP_REASONS as readonly string[]).includes(row.skip_reason)) {
    return { outcome: 'skipped', reason: row.skip_reason as IssueReferralSkipReason };
  }
  return { outcome: 'skipped', reason: 'reservation_conflict' };
}

export async function reserveIssueReferralReward(
  db: D1Database,
  input: {
    referralId: string | null;
    referredSource: ReferralSource;
    referredSubjectRef: string;
    referredInstallationId: string | null;
    referredHardwareHash: string | null;
    referredHardwareHashSaltVersion: number | null;
    attemptIpDigest?: ReferralAttemptIpDigest | null;
    attemptIpLegacyHash?: string | null;
    operationId?: string | null;
    globalCountLimit?: number | null;
    globalCountWindowStartIso?: string | null;
    nowIso: string;
  },
): Promise<IssueReferralReservationResult> {
  if (input.referralId === null || input.referralId.trim().length === 0) {
    return { outcome: 'not_applicable', reason: 'no_referral_input' };
  }

  const referralId = normalizeReferralId(input.referralId);
  if (!referralId) {
    return { outcome: 'not_applicable', reason: 'malformed_referral_input' };
  }

  if (input.operationId) {
    const reused = await getOperationReferralReward(db, input.operationId);
    if (reused) {
      return reused;
    }
  }

  const controls = await getBrokerAbuseControlsConfig(db);
  const attemptIpDigest = (input.attemptIpDigest ?? null);
  const attemptIpLegacyHash = (input.attemptIpLegacyHash ?? null);
  const existingCode = await getReferralCodeByReferralId(db, referralId);

  if (
    await isValidShapedReferralAttemptRateLimited(db, {
      referredSource: input.referredSource,
      referredSubjectRef: input.referredSubjectRef,
      referredInstallationId: input.referredInstallationId,
      attemptIpDigest,
      excludeOperationId: input.operationId ?? null,
      nowIso: input.nowIso,
      controls,
    })
  ) {
    const referrerFields = existingCode
      ? referralRewardReferrerFields(existingCode)
      : { referrerSource: null, referrerSubjectRef: null, referrerInstallationId: null };
    await insertSkippedIssueReferralReward(db, {
      referralId,
      ...referrerFields,
      referredSource: input.referredSource,
      referredSubjectRef: input.referredSubjectRef,
      referredInstallationId: input.referredInstallationId,
      referredHardwareHash: input.referredHardwareHash,
      referredHardwareHashSaltVersion: input.referredHardwareHashSaltVersion,
      skipReason: 'referral_attempt_rate_limited',
      attemptIpDigest,
      operationId: input.operationId ?? null,
      nowIso: input.nowIso,
    });
    return { outcome: 'skipped', reason: 'referral_attempt_rate_limited' };
  }

  if (!existingCode) {
    const reason: IssueReferralSkipReason = (await isUnknownReferralAttemptRateLimited(
      db,
      {
        referredSource: input.referredSource,
        referredSubjectRef: input.referredSubjectRef,
        referredInstallationId: input.referredInstallationId,
        attemptIpDigest,
        excludeOperationId: input.operationId ?? null,
        nowIso: input.nowIso,
        controls,
      },
    ))
      ? 'unknown_referral_id_rate_limited'
      : 'unknown_referral_id';
    await insertSkippedIssueReferralReward(db, {
      referralId,
      referrerSource: null,
      referrerSubjectRef: null,
      referrerInstallationId: null,
      referredSource: input.referredSource,
      referredSubjectRef: input.referredSubjectRef,
      referredInstallationId: input.referredInstallationId,
      referredHardwareHash: input.referredHardwareHash,
      referredHardwareHashSaltVersion: input.referredHardwareHashSaltVersion,
      skipReason: reason,
      attemptIpDigest,
      operationId: input.operationId ?? null,
      nowIso: input.nowIso,
    });
    return { outcome: 'skipped', reason };
  }

  if (existingCode.status !== 'active') {
    const referrerFields = referralRewardReferrerFields(existingCode);
    await insertSkippedIssueReferralReward(db, {
      referralId,
      ...referrerFields,
      referredSource: input.referredSource,
      referredSubjectRef: input.referredSubjectRef,
      referredInstallationId: input.referredInstallationId,
      referredHardwareHash: input.referredHardwareHash,
      referredHardwareHashSaltVersion: input.referredHardwareHashSaltVersion,
      skipReason: 'disabled_referral_id',
      attemptIpDigest,
      operationId: input.operationId ?? null,
      nowIso: input.nowIso,
    });
    return { outcome: 'skipped', reason: 'disabled_referral_id' };
  }

  if (
    await isReferralIdVelocityLimited(db, {
      referralId,
      nowIso: input.nowIso,
      controls,
      excludeOperationId: input.operationId ?? null,
    })
  ) {
    const referrerFields = referralRewardReferrerFields(existingCode);
    await insertSkippedIssueReferralReward(db, {
      referralId,
      ...referrerFields,
      referredSource: input.referredSource,
      referredSubjectRef: input.referredSubjectRef,
      referredInstallationId: input.referredInstallationId,
      referredHardwareHash: input.referredHardwareHash,
      referredHardwareHashSaltVersion: input.referredHardwareHashSaltVersion,
      skipReason: 'referral_velocity_limited',
      attemptIpDigest,
      operationId: input.operationId ?? null,
      nowIso: input.nowIso,
    });
    return { outcome: 'skipped', reason: 'referral_velocity_limited' };
  }

  const reserved = await insertReservedIssueReferralReward(db, {
    referralId,
    referredSource: input.referredSource,
    referredSubjectRef: input.referredSubjectRef,
    referredInstallationId: input.referredInstallationId,
    referredHardwareHash: input.referredHardwareHash,
    referredHardwareHashSaltVersion: input.referredHardwareHashSaltVersion,
    attemptIpDigest,
    attemptIpLegacyHash,
    operationId: input.operationId ?? null,
    controls,
    globalCountLimit: input.globalCountLimit ?? null,
    globalCountWindowStartIso: input.globalCountWindowStartIso ?? null,
    nowIso: input.nowIso,
  });
  if (reserved) {
    logReferralRewardOutcome({
      outcome: 'reserved',
      referralId,
      referredInstallationId: input.referredInstallationId,
      referrerSource: existingCode.owner_source,
      referrerSubjectRef: existingCode.owner_subject_ref,
    });
    return { outcome: 'reserved', referralId };
  }

  const skip = await resolveIssueReferralSkip(db, {
    referralId,
    referredSource: input.referredSource,
    referredSubjectRef: input.referredSubjectRef,
    referredInstallationId: input.referredInstallationId,
    referredHardwareHash: input.referredHardwareHash,
    referredHardwareHashSaltVersion: input.referredHardwareHashSaltVersion,
    controls,
    globalCountLimit: input.globalCountLimit ?? null,
    globalCountWindowStartIso: input.globalCountWindowStartIso ?? null,
    excludeOperationId: input.operationId ?? null,
    nowIso: input.nowIso,
  });
  await insertSkippedIssueReferralReward(db, {
    referralId,
    referrerSource: skip.referrerSource,
    referrerSubjectRef: skip.referrerSubjectRef,
    referrerInstallationId: skip.referrerInstallationId,
    referredSource: input.referredSource,
    referredSubjectRef: input.referredSubjectRef,
    referredInstallationId: input.referredInstallationId,
    referredHardwareHash: input.referredHardwareHash,
    referredHardwareHashSaltVersion: input.referredHardwareHashSaltVersion,
    skipReason: skip.reason,
    attemptIpDigest,
    operationId: input.operationId ?? null,
    nowIso: input.nowIso,
  });

  return { outcome: 'skipped', reason: skip.reason };
}

export async function markReservedIssueReferralFailed(
  db: D1Database,
  input: {
    referralId: string;
    referredSource: ReferralSource;
    referredSubjectRef: string;
    referredInstallationId: string | null;
    failureReason: 'issue_delivery_failed';
    nowIso: string;
  },
): Promise<void> {
  assertIssueReferralFailureReason(input.failureReason);
  await db
    .prepare(
      `UPDATE referral_rewards
          SET referred_bonus_status = 'failed',
              referrer_bonus_status = 'failed',
              failure_reason = ?,
              updated_at = ?
        WHERE referral_id = ?
          AND referred_source = ?
          AND referred_subject_ref = ?
          AND referred_installation_id IS ?
          AND referred_bonus_status = 'reserved'`,
    )
    .bind(
      input.failureReason,
      input.nowIso,
      input.referralId,
      input.referredSource,
      input.referredSubjectRef,
      input.referredInstallationId,
    )
    .run();
  logReferralRewardOutcome({
    outcome: 'failed',
    referralId: input.referralId,
    referredInstallationId: input.referredInstallationId,
    reason: input.failureReason,
  });
}

export async function markReservedIssueReferralCredited(
  db: D1Database,
  input: {
    referralId: string;
    referredSource: ReferralSource;
    referredSubjectRef: string;
    referredInstallationId: string | null;
    referredManagedCredentialRef: string;
    nowIso: string;
  },
): Promise<boolean> {
  const result = await db
    .prepare(
      `UPDATE referral_rewards
          SET referred_bonus_status = 'credited',
              referred_managed_credential_ref = ?,
              failure_reason = NULL,
              updated_at = ?,
              credited_at = ?
        WHERE referral_id = ?
          AND referred_source = ?
          AND referred_subject_ref = ?
          AND referred_installation_id IS ?
          AND referred_bonus_status = 'reserved'`,
    )
    .bind(
      input.referredManagedCredentialRef,
      input.nowIso,
      input.nowIso,
      input.referralId,
      input.referredSource,
      input.referredSubjectRef,
      input.referredInstallationId,
    )
    .run();

  const credited = Number(result.meta.changes ?? 0) === 1;
  if (credited) {
    logReferralRewardOutcome({
      outcome: 'credited',
      referralId: input.referralId,
      referredInstallationId: input.referredInstallationId,
    });
  }
  return credited;
}

export async function applyCreditedIssueReferrerRewardLimitUpdate(
  db: D1Database,
  input: {
    referralId: string;
    referredSource: ReferralSource;
    referredSubjectRef: string;
    referredInstallationId: string | null;
    managementApiKey: string;
    nowIso: string;
    fetchImpl?: typeof fetch;
  },
): Promise<ReferrerRewardLimitUpdateResult> {
  const referrer = await getCreditedIssueReferralReferrer(db, {
    referralId: input.referralId,
    referredSource: input.referredSource,
    referredSubjectRef: input.referredSubjectRef,
    referredInstallationId: input.referredInstallationId,
  });
  if (!referrer) {
    return { outcome: 'not_applicable', reason: 'no_referrer_reward_rows' };
  }

  return applyReferrerRewardLimitUpdates(db, {
    referrerSource: referrer.source,
    referrerSubjectRef: referrer.subjectRef,
    managementApiKey: input.managementApiKey,
    nowIso: input.nowIso,
    fetchImpl: input.fetchImpl,
  });
}

export async function applyReferrerRewardLimitUpdates(
  db: D1Database,
  input: {
    referrerSource: ReferralSource;
    referrerSubjectRef: string;
    managementApiKey: string;
    nowIso: string;
    fetchImpl?: typeof fetch;
  },
): Promise<ReferrerRewardLimitUpdateResult> {
  let lastCreditedResult: Extract<
    ReferrerRewardLimitUpdateResult,
    { outcome: 'credited' }
  > | null = null;

  for (let attempt = 0; attempt < REFERRER_REWARD_DRAIN_ATTEMPTS; attempt += 1) {
    const result = await applyReferrerRewardLimitUpdateAttempt(db, input);
    if (result.outcome !== 'credited') {
      return result;
    }

    lastCreditedResult = result;
    if (
      !(await hasPendingReferrerRewardRows(db, {
        referrerSource: input.referrerSource,
        referrerSubjectRef: input.referrerSubjectRef,
      }))
    ) {
      return result;
    }
  }

  return (
    lastCreditedResult ?? {
      outcome: 'not_applicable',
      reason: 'no_pending_referrer_rewards',
    }
  );
}

async function applyReferrerRewardLimitUpdateAttempt(
  db: D1Database,
  input: {
    referrerSource: ReferralSource;
    referrerSubjectRef: string;
    managementApiKey: string;
    nowIso: string;
    fetchImpl?: typeof fetch;
  },
): Promise<ReferrerRewardLimitUpdateResult> {
  const now = new Date(input.nowIso);
  if (Number.isNaN(now.getTime())) {
    throw new Error('nowIso must be a valid ISO timestamp');
  }

  const activeEntitlement = await getActiveReferrerRewardEntitlement(db, {
    referrerSource: input.referrerSource,
    referrerSubjectRef: input.referrerSubjectRef,
    nowIso: input.nowIso,
  });
  if (!activeEntitlement) {
    const skippedRows = await markReferrerRewardRowsSkipped(db, {
      referrerSource: input.referrerSource,
      referrerSubjectRef: input.referrerSubjectRef,
      nowIso: input.nowIso,
      skipReason: 'referrer_managed_key_missing',
    });
    if (skippedRows === 0) {
      return { outcome: 'not_applicable', reason: 'no_pending_referrer_rewards' };
    }

    return {
      outcome: 'skipped',
      reason: 'referrer_managed_key_missing',
      skippedRows,
    };
  }

  const managedCredentialRef = activeEntitlement.managedCredentialRef;
  const leaseCutoffIso = new Date(
    now.getTime() - REFERRER_APPLYING_LEASE_MS,
  ).toISOString();
  const claimedRows = await claimReferrerRewardApplicationLease(db, {
    referrerSource: input.referrerSource,
    referrerSubjectRef: input.referrerSubjectRef,
    managedCredentialRef,
    nowIso: input.nowIso,
    leaseCutoffIso,
  });
  if (claimedRows === 0) {
    if (
      await hasActiveReferrerRewardApplicationLease(db, {
        referrerSource: input.referrerSource,
        referrerSubjectRef: input.referrerSubjectRef,
        managedCredentialRef,
        leaseCutoffIso,
      })
    ) {
      logReferralRewardOutcome({
        outcome: 'applying',
        referrerSource: input.referrerSource,
        referrerSubjectRef: input.referrerSubjectRef,
        referrerManagedCredentialRef: managedCredentialRef,
        reason: 'active_lease',
      });
      return { outcome: 'applying', reason: 'active_lease' };
    }

    return { outcome: 'not_applicable', reason: 'no_pending_referrer_rewards' };
  }

  try {
    const reflectedRewardCount = await countReferrerRewardsForTargetLimit(db, {
      referrerSource: input.referrerSource,
      referrerSubjectRef: input.referrerSubjectRef,
      managedCredentialRef,
    });
    const ledgerTargetLimitUsd = referrerRewardTargetLimitUsd(reflectedRewardCount);
    const providerLimitUsd = await readManagedChildKeyEffectiveLimit({
      managementApiKey: input.managementApiKey,
      keyHash: managedCredentialRef,
      fetchImpl: input.fetchImpl,
    });
    const targetLimitUsd = maxUsd(
      ledgerTargetLimitUsd,
      providerLimitUsd,
      activeEntitlement.budgetUsd,
    );
    let verifiedLimitUsd = providerLimitUsd;

    if (currencyCents(providerLimitUsd) < currencyCents(targetLimitUsd)) {
      verifiedLimitUsd = await updateManagedChildKeyLimit({
        managementApiKey: input.managementApiKey,
        keyHash: managedCredentialRef,
        limitUsd: targetLimitUsd,
        fetchImpl: input.fetchImpl,
      });
    }

    const consistentLimitUsd = maxUsd(targetLimitUsd, verifiedLimitUsd);
    const budgetUpdated = await updateReferrerEntitlementBudget(db, {
      owner: activeEntitlement,
      managedCredentialRef,
      budgetUsd: consistentLimitUsd,
    });
    if (!budgetUpdated) {
      throw new Error('referrer entitlement budget update failed');
    }

    const creditedRows = await markReferrerRewardRowsCredited(db, {
      referrerSource: input.referrerSource,
      referrerSubjectRef: input.referrerSubjectRef,
      managedCredentialRef,
      nowIso: input.nowIso,
    });
    if (creditedRows > 0) {
      logReferralRewardOutcome({
        outcome: 'credited',
        referrerSource: input.referrerSource,
        referrerSubjectRef: input.referrerSubjectRef,
        referrerManagedCredentialRef: managedCredentialRef,
        affectedRows: creditedRows,
      });
    }
    return {
      outcome: 'credited',
      creditedRows,
      targetLimitUsd: consistentLimitUsd,
    };
  } catch {
    const failedRows = await markReferrerRewardRowsFailed(db, {
      referrerSource: input.referrerSource,
      referrerSubjectRef: input.referrerSubjectRef,
      managedCredentialRef,
      nowIso: input.nowIso,
      failureReason: 'referrer_patch_failed',
    });
    return {
      outcome: 'failed',
      reason: 'referrer_patch_failed',
      failedRows,
    };
  }
}

export async function recordSkippedIssueReferralReward(
  db: D1Database,
  input: {
    referralId: string | null;
    referredSource: ReferralSource;
    referredSubjectRef: string;
    referredInstallationId: string | null;
    referredHardwareHash: string | null;
    referredHardwareHashSaltVersion: number | null;
    skipReason: IssueReferralSkipReason;
    attemptIpDigest?: ReferralAttemptIpDigest | null;
    attemptIpLegacyHash?: string | null;
    operationId?: string | null;
    nowIso: string;
  },
): Promise<IssueReferralReservationResult> {
  if (input.operationId) {
    const reused = await getOperationReferralReward(db, input.operationId);
    if (reused) {
      return reused;
    }
  }
  if (input.referralId === null || input.referralId.trim().length === 0) {
    return { outcome: 'not_applicable', reason: 'no_referral_input' };
  }

  const referralId = normalizeReferralId(input.referralId);
  if (!referralId) {
    return { outcome: 'not_applicable', reason: 'malformed_referral_input' };
  }

  const skip = await resolveForcedIssueReferralSkip(db, {
    referralId,
    fallbackReason: input.skipReason,
  });
  const attemptIpDigest = (input.attemptIpDigest ?? null);
  const attemptIpLegacyHash = (input.attemptIpLegacyHash ?? null);
  await insertSkippedIssueReferralReward(db, {
    referralId,
    referrerSource: skip.referrerSource,
    referrerSubjectRef: skip.referrerSubjectRef,
    referrerInstallationId: skip.referrerInstallationId,
    referredSource: input.referredSource,
    referredSubjectRef: input.referredSubjectRef,
    referredInstallationId: input.referredInstallationId,
    referredHardwareHash: input.referredHardwareHash,
    referredHardwareHashSaltVersion: input.referredHardwareHashSaltVersion,
    skipReason: skip.reason,
    attemptIpDigest,
    operationId: input.operationId ?? null,
    nowIso: input.nowIso,
  });

  return { outcome: 'skipped', reason: skip.reason };
}

export async function reconcileStaleReferralRewards(
  db: D1Database,
  input: {
    nowIso: string;
    staleReservedAfterMinutes?: number;
    staleApplyingAfterMinutes?: number;
  },
): Promise<StaleReferralRewardReconciliationResult> {
  const now = new Date(input.nowIso);
  if (Number.isNaN(now.getTime())) {
    throw new Error('nowIso must be a valid ISO timestamp');
  }

  const reservedCutoffIso = new Date(
    now.getTime() -
      (input.staleReservedAfterMinutes === undefined
        ? DEFAULT_STALE_RESERVED_RECONCILE_MS
        : input.staleReservedAfterMinutes * 60_000),
  ).toISOString();
  const applyingCutoffIso = new Date(
    now.getTime() -
      (input.staleApplyingAfterMinutes === undefined
        ? DEFAULT_STALE_APPLYING_RECONCILE_MS
        : input.staleApplyingAfterMinutes * 60_000),
  ).toISOString();
  let staleReservedCredited = 0;
  let staleReservedFailed = 0;
  const staleReservedRows = await listStaleReservedReferralRewards(
    db,
    reservedCutoffIso,
  );
  for (const row of staleReservedRows) {
    const deliveredEntitlement = await getDeliveredReferredEntitlement(db, row);
    if (deliveredEntitlement?.managed_credential_ref) {
      const credited = await reconcileStaleReservedReferralToCredited(db, {
        rewardId: row.id,
        referralId: row.referral_id,
        referredInstallationId: row.referred_installation_id,
        referrerSource: row.referrer_source,
        referrerSubjectRef: row.referrer_subject_ref,
        managedCredentialRef: deliveredEntitlement.managed_credential_ref,
        expectedUpdatedAt: row.updated_at,
        expectedFailureReason: row.failure_reason,
        nowIso: input.nowIso,
      });
      staleReservedCredited += credited;
      continue;
    }

    const failed = await reconcileStaleReservedReferralToFailed(db, {
      rewardId: row.id,
      referralId: row.referral_id,
      referredInstallationId: row.referred_installation_id,
      referrerSource: row.referrer_source,
      referrerSubjectRef: row.referrer_subject_ref,
      expectedUpdatedAt: row.updated_at,
      expectedFailureReason: row.failure_reason,
      nowIso: input.nowIso,
    });
    staleReservedFailed += failed;
  }

  const staleApplyingRequeued = await requeueStaleApplyingReferralRewards(db, {
    cutoffIso: applyingCutoffIso,
    nowIso: input.nowIso,
  });

  return {
    staleReservedCredited,
    staleReservedFailed,
    staleApplyingRequeued,
  };
}

export async function applyReferralRewardRetention(
  db: D1Database,
  now: Date,
): Promise<ReferralRewardRetentionResult> {
  if (Number.isNaN(now.getTime())) {
    throw new Error('now must be a valid Date');
  }

  const controls = await getBrokerAbuseControlsConfig(db);
  const skippedDeleted = await deleteTerminalReferralRewardsOlderThan(db, {
    referredBonusStatus: 'skipped',
    cutoffIso: new Date(
      now.getTime() - controls.retention.referralSkippedDays * 24 * 60 * 60_000,
    ).toISOString(),
  });
  const failedDeleted = await deleteTerminalReferralRewardsOlderThan(db, {
    referredBonusStatus: 'failed',
    cutoffIso: new Date(
      now.getTime() - controls.retention.referralFailedDays * 24 * 60 * 60_000,
    ).toISOString(),
  });

  return {
    skippedDeleted,
    failedDeleted,
  };
}

export async function disableReferralId(
  db: D1Database,
  input: {
    referralId: string;
    reason: unknown;
    disabledBy: unknown;
    nowIso: string;
  },
): Promise<DisableReferralIdResult> {
  const referralId = normalizeReferralId(input.referralId);
  if (!referralId) {
    return { ok: false, reason: 'invalid_referral_id' };
  }

  const disableReason = normalizeReferralDisableReason(input.reason);
  if (!disableReason) {
    return { ok: false, reason: 'invalid_disable_reason' };
  }

  const disabledBy = normalizeReferralDisableActor(input.disabledBy);
  if (!disabledBy) {
    return { ok: false, reason: 'invalid_disabled_by' };
  }

  const existing = await getReferralCodeByReferralId(db, referralId);
  if (!existing) {
    return { ok: false, reason: 'not_found' };
  }

  if (existing.status === 'disabled') {
    return { ok: true, status: 'already_disabled' };
  }

  await db
    .prepare(
      `UPDATE referral_codes
          SET status = 'disabled',
              disabled_reason = ?,
              disabled_by = ?,
              disabled_at = ?,
              updated_at = ?
        WHERE referral_id = ?
          AND status = 'active'`,
    )
    .bind(disableReason, disabledBy, input.nowIso, input.nowIso, referralId)
    .run();
  await appendReferralRuntimeAudit(db, {
    eventKind: 'referral_id_disabled',
    reason: disableReason,
    payload: {
      referral_id: referralId,
      disabled_by: disabledBy,
      previous_status: existing.status,
    },
    createdAt: input.nowIso,
  });
  logReferralRewardOutcome({
    outcome: 'disabled',
    referralId,
    reason: disableReason,
  });

  return { ok: true, status: 'disabled' };
}

export function generateReferralId(
  randomBytes: ReferralIdRandomBytes = cryptoReferralRandomBytes,
): string {
  let referralId = '';
  let drawCount = 0;

  while (referralId.length < REFERRAL_ID_LENGTH) {
    drawCount += 1;
    if (drawCount > REFERRAL_ID_MAX_RANDOM_DRAWS) {
      throw new Error('unable to generate Referral ID from random source');
    }

    const bytes = randomBytes(REFERRAL_ID_LENGTH - referralId.length);
    if (bytes.length === 0) {
      throw new Error('Referral ID random source returned no bytes');
    }

    for (const byte of bytes) {
      if (byte >= REFERRAL_RANDOM_REJECTION_THRESHOLD) {
        continue;
      }

      referralId += REFERRAL_ID_ALPHABET[byte % REFERRAL_ID_ALPHABET.length];
      if (referralId.length === REFERRAL_ID_LENGTH) {
        break;
      }
    }
  }

  return referralId;
}

async function isValidShapedReferralAttemptRateLimited(
  db: D1Database,
  input: {
    referredSource: ReferralSource;
    referredSubjectRef: string;
    referredInstallationId: string | null;
    attemptIpDigest: ReferralAttemptIpDigest | null;
    attemptIpLegacyHash?: string | null;
    excludeOperationId?: string | null;
    nowIso: string;
    controls: BrokerAbuseControlsConfigValue;
  },
): Promise<boolean> {
  const config = input.controls.referralAttempts.validShaped;
  const windowStart = windowStartIso(input.nowIso, config.windowMinutes);
  const installationCount = await countReferralAttemptsByInstallation(db, {
    referredSource: input.referredSource,
    referredSubjectRef: input.referredSubjectRef,
    referredInstallationId: input.referredInstallationId,
    windowStartIso: windowStart,
    excludeOperationId: input.excludeOperationId ?? null,
  });
  if (installationCount >= config.maxPerInstallation) {
    return true;
  }

  if (!input.attemptIpDigest) {
    return false;
  }

  const ipCount = await countReferralAttemptsByIpDigest(db, {
    attemptIpDigest: input.attemptIpDigest?.digest ?? null,
    attemptIpLegacyHash: input.attemptIpLegacyHash,
    windowStartIso: windowStart,
    excludeOperationId: input.excludeOperationId ?? null,
  });
  return ipCount >= config.maxPerIp;
}

async function isUnknownReferralAttemptRateLimited(
  db: D1Database,
  input: {
    referredSource: ReferralSource;
    referredSubjectRef: string;
    referredInstallationId: string | null;
    attemptIpDigest: ReferralAttemptIpDigest | null;
    attemptIpLegacyHash?: string | null;
    excludeOperationId?: string | null;
    nowIso: string;
    controls: BrokerAbuseControlsConfigValue;
  },
): Promise<boolean> {
  const config = input.controls.referralAttempts.unknown;
  const windowStart = windowStartIso(input.nowIso, config.windowMinutes);
  const installationCount = await countUnknownReferralAttemptsByInstallation(db, {
    referredSource: input.referredSource,
    referredSubjectRef: input.referredSubjectRef,
    referredInstallationId: input.referredInstallationId,
    windowStartIso: windowStart,
    excludeOperationId: input.excludeOperationId ?? null,
  });
  if (installationCount >= config.maxPerInstallation) {
    return true;
  }

  if (!input.attemptIpDigest) {
    return false;
  }

  const ipCount = await countUnknownReferralAttemptsByIpDigest(db, {
    attemptIpDigest: input.attemptIpDigest?.digest ?? null,
    attemptIpLegacyHash: input.attemptIpLegacyHash,
    windowStartIso: windowStart,
    excludeOperationId: input.excludeOperationId ?? null,
  });
  return ipCount >= config.maxPerIp;
}

async function isReferralIdVelocityLimited(
  db: D1Database,
  input: {
    referralId: string;
    nowIso: string;
    controls: BrokerAbuseControlsConfigValue;
    excludeOperationId?: string | null;
  },
): Promise<boolean> {
  const config = input.controls.referralAttempts.perReferralIdVelocity;
  const count = await countReferralAttemptsForReferralId(db, {
    referralId: input.referralId,
    windowStartIso: windowStartIso(input.nowIso, config.windowMinutes),
    excludeOperationId: input.excludeOperationId ?? null,
  });
  return count >= config.maxAttempts;
}

async function isReferrerRewardVelocityLimited(
  db: D1Database,
  input: {
    referrerSource: ReferralSource;
    referrerSubjectRef: string;
    nowIso: string;
    controls: BrokerAbuseControlsConfigValue;
    excludeOperationId?: string | null;
  },
): Promise<boolean> {
  const config = input.controls.referralAttempts.perReferrerRewardVelocity;
  const count = await countRecentCountedRewardsForReferrer(db, {
    referrerSource: input.referrerSource,
    referrerSubjectRef: input.referrerSubjectRef,
    windowStartIso: windowStartIso(input.nowIso, config.windowMinutes),
    excludeOperationId: input.excludeOperationId ?? null,
  });
  return count >= config.maxRewards;
}

async function countReferralAttemptsByInstallation(
  db: D1Database,
  input: {
    referredSource: ReferralSource;
    referredSubjectRef: string;
    referredInstallationId: string | null;
    windowStartIso: string;
    excludeOperationId?: string | null;
  },
): Promise<number> {
  const row = await db
    .prepare(
      `SELECT COUNT(*) AS count
         FROM referral_rewards
        WHERE created_at >= ?
          AND (
            (referred_source = ? AND referred_subject_ref = ?)
            OR (? IS NOT NULL AND referred_installation_id = ?)
          )
          AND (operation_id IS NULL OR ? IS NULL OR operation_id <> ?)`,
    )
    .bind(
      input.windowStartIso,
      input.referredSource,
      input.referredSubjectRef,
      input.referredInstallationId,
      input.referredInstallationId,
      input.excludeOperationId ?? null,
      input.excludeOperationId ?? null,
    )
    .first<{ count: number }>();
  return Number(row?.count ?? 0);
}

async function countReferralAttemptsByIpDigest(
  db: D1Database,
  input: { attemptIpDigest: string | null; attemptIpLegacyHash?: string | null; windowStartIso: string; excludeOperationId?: string | null },
): Promise<number> {
  const mode = await resolveNetworkIdentityWriteMode(db);
  const clauses: string[] = [];
  const binds: string[] = [];
  if (mode !== 'legacy' && input.attemptIpDigest) {
    clauses.push('attempt_ip_digest = ?');
    binds.push(input.attemptIpDigest);
  }
  if (mode !== 'keyed' && input.attemptIpLegacyHash) {
    clauses.push('attempt_ip_hash = ?');
    binds.push(input.attemptIpLegacyHash);
  }
  if (clauses.length === 0) {
    return 0;
  }
  const row = await db
    .prepare(
      `SELECT COUNT(*) AS count
         FROM referral_rewards
        WHERE (${clauses.join(' OR ')})
          AND created_at >= ?
          AND (operation_id IS NULL OR ? IS NULL OR operation_id <> ?)`,
    )
    .bind(...binds, input.windowStartIso, input.excludeOperationId ?? null, input.excludeOperationId ?? null)
    .first<{ count: number }>();
  return Number(row?.count ?? 0);
}

async function countUnknownReferralAttemptsByInstallation(
  db: D1Database,
  input: {
    referredSource: ReferralSource;
    referredSubjectRef: string;
    referredInstallationId: string | null;
    windowStartIso: string;
    excludeOperationId?: string | null;
  },
): Promise<number> {
  const row = await db
    .prepare(
      `SELECT COUNT(*) AS count
         FROM referral_rewards
        WHERE skip_reason IN ('unknown_referral_id', 'unknown_referral_id_rate_limited')
          AND created_at >= ?
          AND (
            (referred_source = ? AND referred_subject_ref = ?)
            OR (? IS NOT NULL AND referred_installation_id = ?)
          )
          AND (operation_id IS NULL OR ? IS NULL OR operation_id <> ?)`,
    )
    .bind(
      input.windowStartIso,
      input.referredSource,
      input.referredSubjectRef,
      input.referredInstallationId,
      input.referredInstallationId,
      input.excludeOperationId ?? null,
      input.excludeOperationId ?? null,
    )
    .first<{ count: number }>();
  return Number(row?.count ?? 0);
}

async function countUnknownReferralAttemptsByIpDigest(
  db: D1Database,
  input: { attemptIpDigest: string | null; attemptIpLegacyHash?: string | null; windowStartIso: string; excludeOperationId?: string | null },
): Promise<number> {
  const mode = await resolveNetworkIdentityWriteMode(db);
  const clauses: string[] = [];
  const binds: string[] = [];
  if (mode !== 'legacy' && input.attemptIpDigest) {
    clauses.push('attempt_ip_digest = ?');
    binds.push(input.attemptIpDigest);
  }
  if (mode !== 'keyed' && input.attemptIpLegacyHash) {
    clauses.push('attempt_ip_hash = ?');
    binds.push(input.attemptIpLegacyHash);
  }
  if (clauses.length === 0) {
    return 0;
  }
  const row = await db
    .prepare(
      `SELECT COUNT(*) AS count
         FROM referral_rewards
        WHERE (${clauses.join(' OR ')})
          AND skip_reason IN ('unknown_referral_id', 'unknown_referral_id_rate_limited')
          AND created_at >= ?
          AND (operation_id IS NULL OR ? IS NULL OR operation_id <> ?)`,
    )
    .bind(...binds, input.windowStartIso, input.excludeOperationId ?? null, input.excludeOperationId ?? null)
    .first<{ count: number }>();
  return Number(row?.count ?? 0);
}

async function countReferralAttemptsForReferralId(
  db: D1Database,
  input: { referralId: string; windowStartIso: string; excludeOperationId?: string | null },
): Promise<number> {
  const row = await db
    .prepare(
      `SELECT COUNT(*) AS count
         FROM referral_rewards
        WHERE referral_id = ?
          AND created_at >= ?
          AND (operation_id IS NULL OR ? IS NULL OR operation_id <> ?)`,
    )
    .bind(input.referralId, input.windowStartIso, input.excludeOperationId ?? null, input.excludeOperationId ?? null)
    .first<{ count: number }>();
  return Number(row?.count ?? 0);
}

async function countRecentCountedRewardsForReferrer(
  db: D1Database,
  input: {
    referrerSource: ReferralSource;
    referrerSubjectRef: string;
    windowStartIso: string;
    excludeOperationId?: string | null;
  },
): Promise<number> {
  const row = await db
    .prepare(
      `SELECT COUNT(*) AS count
         FROM referral_rewards
        WHERE referrer_source = ?
          AND referrer_subject_ref = ?
          AND referred_bonus_status IN ('reserved', 'credited')
          AND created_at >= ?
          AND (operation_id IS NULL OR ? IS NULL OR operation_id <> ?)`,
    )
    .bind(input.referrerSource, input.referrerSubjectRef, input.windowStartIso, input.excludeOperationId ?? null, input.excludeOperationId ?? null)
    .first<{ count: number }>();
  return Number(row?.count ?? 0);
}

function windowStartIso(nowIso: string, windowMinutes: number): string {
  const now = new Date(nowIso);
  if (Number.isNaN(now.getTime())) {
    throw new Error('nowIso must be a valid ISO timestamp');
  }

  return new Date(now.getTime() - windowMinutes * 60_000).toISOString();
}

export interface ReferralAttemptIpDigest {
  digest: string;
  keyVersion: number;
  epoch: string;
}

function referralAttemptIdentityColumnSet(
  mode: 'dual' | 'keyed',
): { columns: string; placeholders: string } {
  if (mode === 'dual') {
    return {
      columns: 'attempt_ip_hash,\n          attempt_ip_digest,\n          attempt_ip_key_version,\n          attempt_ip_epoch',
      placeholders: '?,\n               ?,\n               ?,\n               ?'
    };
  }
  return {
    columns: 'attempt_ip_digest,\n          attempt_ip_key_version,\n          attempt_ip_epoch',
    placeholders: '?,\n               ?,\n               ?'
  };
}

function referralAttemptIdentityBinds(
  mode: 'dual' | 'keyed',
  digest: ReferralAttemptIpDigest | null | undefined,
  legacyHash: string | null | undefined,
): Array<string | number | null> {
  if (mode === 'dual') {
    return [legacyHash ?? null, digest?.digest ?? null, digest?.keyVersion ?? null, digest?.epoch ?? null];
  }
  return [digest?.digest ?? null, digest?.keyVersion ?? null, digest?.epoch ?? null];
}

async function insertReservedIssueReferralReward(
  db: D1Database,
  input: {
    referralId: string;
    referredSource: ReferralSource;
    referredSubjectRef: string;
    referredInstallationId: string | null;
    referredHardwareHash: string | null;
    referredHardwareHashSaltVersion: number | null;
    attemptIpDigest: ReferralAttemptIpDigest | null;
    attemptIpLegacyHash?: string | null;
    operationId?: string | null;
    controls: BrokerAbuseControlsConfigValue;
    globalCountLimit: number | null;
    globalCountWindowStartIso: string | null;
    nowIso: string;
  },
): Promise<boolean> {
  const referralVelocityWindowStartIso = windowStartIso(
    input.nowIso,
    input.controls.referralAttempts.perReferralIdVelocity.windowMinutes,
  );
  const referrerVelocityWindowStartIso = windowStartIso(
    input.nowIso,
    input.controls.referralAttempts.perReferrerRewardVelocity.windowMinutes,
  );
  const identityMode = (await resolveNetworkIdentityWriteMode(db)) === 'keyed' ? 'keyed' : 'dual';
  const identityColumns = referralAttemptIdentityColumnSet(identityMode);
  const result = await db
    .prepare(
      `INSERT OR IGNORE INTO referral_rewards (
          referral_id,
          referrer_source,
          referrer_subject_ref,
          referrer_installation_id,
          referred_source,
          referred_subject_ref,
          referred_installation_id,
          referred_hardware_hash,
          referred_hardware_hash_salt_version,
          referred_bonus_status,
          referrer_bonus_status,
          skip_reason,
          failure_reason,
          referred_managed_credential_ref,
          referrer_managed_credential_ref,
          ${identityColumns.columns},
          operation_id,
          created_at,
          updated_at
        )
        SELECT code.referral_id,
               code.owner_source,
               code.owner_subject_ref,
               code.owner_installation_id,
               ?,
               ?,
               ?,
               ?,
               ?,
               'reserved',
               'pending',
               NULL,
               NULL,
               NULL,
               NULL,
               ${identityColumns.placeholders},
               ?,
               ?,
               ?
          FROM referral_codes code
         WHERE code.referral_id = ?
           AND code.status = 'active'
           AND (
             (
               code.owner_source = 'discord'
               AND code.owner_installation_id IS NOT NULL
               AND EXISTS (
                 SELECT 1
                   FROM discord_identities identity
                  WHERE identity.discord_user_ref = code.owner_subject_ref
                    AND identity.entitlement_installation_id = code.owner_installation_id
                    AND identity.status = 'active'
               )
             )
             OR (
               code.owner_source = 'qq'
               AND EXISTS (
                 SELECT 1
                   FROM qq_managed_entitlements entitlement
                  WHERE entitlement.qq_subject_ref = code.owner_subject_ref
                    AND entitlement.status = 'active'
                    AND entitlement.managed_credential_ref IS NOT NULL
                    AND length(trim(entitlement.managed_credential_ref)) > 0
                    AND entitlement.delivered_at IS NOT NULL
                    AND entitlement.expires_at IS NOT NULL
                    AND datetime(entitlement.expires_at) >= datetime(?)
               )
             )
           )
           AND NOT (
             code.owner_source = ?
             AND code.owner_subject_ref = ?
           )
           AND NOT (
             ? IS NOT NULL
             AND code.owner_installation_id IS NOT NULL
             AND code.owner_installation_id = ?
           )
           AND (
             code.owner_source <> 'discord'
             OR ? <> 'discord'
             OR NOT EXISTS (
               SELECT 1
                 FROM openrouter_entitlements referrer_entitlement
                WHERE referrer_entitlement.discord_user_ref = code.owner_subject_ref
                  AND referrer_entitlement.status = 'active'
                  AND referrer_entitlement.discord_issue_status = 'active'
                  AND referrer_entitlement.verified_hardware_hash = ?
                  AND referrer_entitlement.verified_hardware_hash_salt_version = ?
             )
           )
           AND (
             SELECT COUNT(*)
               FROM referral_rewards counted
              WHERE counted.referrer_source = code.owner_source
                AND counted.referrer_subject_ref = code.owner_subject_ref
                AND counted.referred_bonus_status IN ('reserved', 'credited')
           ) < ?
           AND NOT EXISTS (
             SELECT 1
               FROM referral_rewards counted_referred
              WHERE counted_referred.referred_bonus_status IN ('reserved', 'credited')
                AND (
                  (
                    counted_referred.referred_source = ?
                    AND counted_referred.referred_subject_ref = ?
                  )
                  OR (
                    ? IS NOT NULL
                    AND counted_referred.referred_installation_id = ?
                  )
                )
           )
           AND (
             ? IS NULL
             OR ? IS NULL
             OR ? <> 'qq'
             OR (
               SELECT COUNT(*)
                 FROM referral_rewards qq_daily
                WHERE qq_daily.referred_source = 'qq'
                  AND qq_daily.referred_bonus_status IN ('reserved', 'credited')
                  AND qq_daily.created_at >= ?
             ) < ?
           )
           AND (
             SELECT COUNT(*)
               FROM referral_rewards referral_velocity
              WHERE referral_velocity.referral_id = code.referral_id
                AND referral_velocity.created_at >= ?
           ) < ?
           AND (
             SELECT COUNT(*)
               FROM referral_rewards referrer_velocity
              WHERE referrer_velocity.referrer_source = code.owner_source
                AND referrer_velocity.referrer_subject_ref = code.owner_subject_ref
                AND referrer_velocity.referred_bonus_status IN ('reserved', 'credited')
                AND referrer_velocity.created_at >= ?
           ) < ?`,
    )
    .bind(
      input.referredSource,
      input.referredSubjectRef,
      input.referredInstallationId,
      input.referredHardwareHash,
      input.referredHardwareHashSaltVersion,
      ...referralAttemptIdentityBinds(identityMode, input.attemptIpDigest, input.attemptIpLegacyHash),
      input.operationId ?? null,
      input.nowIso,
      input.nowIso,
      input.referralId,
      input.nowIso,
      input.referredSource,
      input.referredSubjectRef,
      input.referredInstallationId,
      input.referredInstallationId,
      input.referredSource,
      input.referredHardwareHash,
      input.referredHardwareHashSaltVersion,
      TALK_TOGETHER_PASS_INVITE_LIMIT,
      input.referredSource,
      input.referredSubjectRef,
      input.referredInstallationId,
      input.referredInstallationId,
      input.globalCountLimit,
      input.globalCountWindowStartIso,
      input.referredSource,
      input.globalCountWindowStartIso,
      input.globalCountLimit,
      referralVelocityWindowStartIso,
      input.controls.referralAttempts.perReferralIdVelocity.maxAttempts,
      referrerVelocityWindowStartIso,
      input.controls.referralAttempts.perReferrerRewardVelocity.maxRewards,
    )
    .run();

  return Number(result.meta.changes ?? 0) === 1;
}

async function resolveIssueReferralSkip(
  db: D1Database,
  input: {
    referralId: string;
    referredSource: ReferralSource;
    referredSubjectRef: string;
    referredInstallationId: string | null;
    referredHardwareHash: string | null;
    referredHardwareHashSaltVersion: number | null;
    controls: BrokerAbuseControlsConfigValue;
    globalCountLimit: number | null;
    globalCountWindowStartIso: string | null;
    excludeOperationId?: string | null;
    nowIso: string;
  },
): Promise<{
  reason: IssueReferralSkipReason;
  referrerSource: ReferralSource | null;
  referrerSubjectRef: string | null;
  referrerInstallationId: string | null;
}> {
  const code = await getReferralCodeByReferralId(db, input.referralId);
  if (!code) {
    return {
      reason: 'unknown_referral_id',
      referrerSource: null,
      referrerSubjectRef: null,
      referrerInstallationId: null,
    };
  }

  const referrerFields = referralRewardReferrerFields(code);
  if (code.status !== 'active') {
    return { reason: 'disabled_referral_id', ...referrerFields };
  }

  if (
    code.owner_source === input.referredSource &&
    code.owner_subject_ref === input.referredSubjectRef
  ) {
    return { reason: 'self_referral', ...referrerFields };
  }

  if (
    input.referredInstallationId !== null &&
    code.owner_installation_id === input.referredInstallationId
  ) {
    return { reason: 'self_or_cross_source_installation', ...referrerFields };
  }

  if (
    !(await hasActiveReferralOwnerIdentity(db, {
      referrerSource: code.owner_source,
      referrerSubjectRef: code.owner_subject_ref,
      referrerInstallationId: code.owner_installation_id,
      nowIso: input.nowIso,
    }))
  ) {
    return { reason: 'referrer_not_eligible', ...referrerFields };
  }

  if (
    code.owner_source === 'discord' &&
    input.referredSource === 'discord' &&
    (await hasIssueReferralDuplicateHardware(db, input, code.owner_subject_ref))
  ) {
    return { reason: 'duplicate_hardware', ...referrerFields };
  }

  if (await hasCountedIssueReferralForReferred(db, input)) {
    return { reason: 'referred_already_rewarded', ...referrerFields };
  }

  if (
    await hasReachedIssueReferralCap(db, code.owner_source, code.owner_subject_ref)
  ) {
    return { reason: 'referrer_cap_reached', ...referrerFields };
  }

  if (
    input.referredSource === 'qq' &&
    input.globalCountLimit !== null &&
    input.globalCountWindowStartIso !== null &&
    (await countCountedQqReferralRewardsSince(
      db,
      input.globalCountWindowStartIso,
    )) >= input.globalCountLimit
  ) {
    return { reason: 'global_reward_cap_reached', ...referrerFields };
  }

  if (
    await isReferralIdVelocityLimited(db, {
      referralId: input.referralId,
      nowIso: input.nowIso,
      controls: input.controls,
      excludeOperationId: input.excludeOperationId ?? null,
    })
  ) {
    return { reason: 'referral_velocity_limited', ...referrerFields };
  }

  if (
    await isReferrerRewardVelocityLimited(db, {
      referrerSource: code.owner_source,
      referrerSubjectRef: code.owner_subject_ref,
      nowIso: input.nowIso,
      controls: input.controls,
      excludeOperationId: input.excludeOperationId ?? null,
    })
  ) {
    return { reason: 'referrer_velocity_limited', ...referrerFields };
  }

  return { reason: 'reservation_conflict', ...referrerFields };
}

async function resolveForcedIssueReferralSkip(
  db: D1Database,
  input: {
    referralId: string;
    fallbackReason: IssueReferralSkipReason;
  },
): Promise<{
  reason: IssueReferralSkipReason;
  referrerSource: ReferralSource | null;
  referrerSubjectRef: string | null;
  referrerInstallationId: string | null;
}> {
  const code = await getReferralCodeByReferralId(db, input.referralId);
  if (!code) {
    return {
      reason: 'unknown_referral_id',
      referrerSource: null,
      referrerSubjectRef: null,
      referrerInstallationId: null,
    };
  }

  const referrerFields = referralRewardReferrerFields(code);
  if (code.status !== 'active') {
    return { reason: 'disabled_referral_id', ...referrerFields };
  }

  return { reason: input.fallbackReason, ...referrerFields };
}

function referralRewardReferrerFields(code: ReferralCodeRecord): {
  referrerSource: ReferralSource;
  referrerSubjectRef: string;
  referrerInstallationId: string | null;
} {
  return {
    referrerSource: code.owner_source,
    referrerSubjectRef: code.owner_subject_ref,
    referrerInstallationId: code.owner_installation_id,
  };
}

async function getReferralCodeByReferralId(
  db: D1Database,
  referralId: string,
): Promise<ReferralCodeRecord | null> {
  return db
    .prepare(
      `SELECT referral_id,
              owner_source,
              owner_subject_ref,
              owner_installation_id,
              status,
              created_at,
              updated_at
         FROM referral_codes
        WHERE referral_id = ?`,
    )
    .bind(referralId)
    .first<ReferralCodeRecord>();
}

async function hasIssueReferralDuplicateHardware(
  db: D1Database,
  input: {
    referredHardwareHash: string | null;
    referredHardwareHashSaltVersion: number | null;
  },
  referrerSubjectRef: string,
): Promise<boolean> {
  const row = await db
    .prepare(
      `SELECT EXISTS(
          SELECT 1
            FROM openrouter_entitlements referrer_entitlement
           WHERE referrer_entitlement.discord_user_ref = ?
             AND referrer_entitlement.status = 'active'
             AND referrer_entitlement.discord_issue_status = 'active'
             AND referrer_entitlement.verified_hardware_hash = ?
             AND referrer_entitlement.verified_hardware_hash_salt_version = ?
        ) AS duplicate_found`,
    )
    .bind(
      referrerSubjectRef,
      input.referredHardwareHash,
      input.referredHardwareHashSaltVersion,
    )
    .first<{ duplicate_found: number }>();

  return Number(row?.duplicate_found ?? 0) === 1;
}

async function hasActiveReferralOwnerIdentity(
  db: D1Database,
  input: {
    referrerSource: ReferralSource;
    referrerSubjectRef: string;
    referrerInstallationId: string | null;
    nowIso: string;
  },
): Promise<boolean> {
  if (input.referrerSource === 'discord') {
    if (!input.referrerInstallationId) {
      return false;
    }
    const row = await db
      .prepare(
        `SELECT EXISTS(
            SELECT 1
              FROM discord_identities identity
             WHERE identity.discord_user_ref = ?
               AND identity.entitlement_installation_id = ?
               AND identity.status = 'active'
          ) AS active_found`,
      )
      .bind(input.referrerSubjectRef, input.referrerInstallationId)
      .first<{ active_found: number }>();
    return Number(row?.active_found ?? 0) === 1;
  }

  const row = await db
    .prepare(
      `SELECT EXISTS(
          SELECT 1
            FROM qq_managed_entitlements entitlement
           WHERE entitlement.qq_subject_ref = ?
             AND entitlement.status = 'active'
             AND entitlement.managed_credential_ref IS NOT NULL
             AND length(trim(entitlement.managed_credential_ref)) > 0
             AND entitlement.delivered_at IS NOT NULL
             AND entitlement.expires_at IS NOT NULL
             AND datetime(entitlement.expires_at) >= datetime(?)
        ) AS active_found`,
    )
    .bind(input.referrerSubjectRef, input.nowIso)
    .first<{ active_found: number }>();
  return Number(row?.active_found ?? 0) === 1;
}

async function hasCountedIssueReferralForReferred(
  db: D1Database,
  input: {
    referredSource: ReferralSource;
    referredSubjectRef: string;
    referredInstallationId: string | null;
  },
): Promise<boolean> {
  const row = await db
    .prepare(
      `SELECT EXISTS(
          SELECT 1
            FROM referral_rewards counted
           WHERE counted.referred_bonus_status IN ('reserved', 'credited')
             AND (
               (
                 counted.referred_source = ?
                 AND counted.referred_subject_ref = ?
               )
               OR (
                 ? IS NOT NULL
                 AND counted.referred_installation_id = ?
               )
             )
        ) AS counted_found`,
    )
    .bind(
      input.referredSource,
      input.referredSubjectRef,
      input.referredInstallationId,
      input.referredInstallationId,
    )
    .first<{ counted_found: number }>();

  return Number(row?.counted_found ?? 0) === 1;
}

export async function resolveTalkTogetherPassStatusForOwnedReferralCode(
  db: D1Database,
  referralCode: Pick<
    ReferralCodeRecord,
    'referral_id' | 'owner_source' | 'owner_subject_ref'
  >,
): Promise<TalkTogetherPassStatusResponse> {
  const inviteCount = await countCountedIssueReferralRewardsForReferrer(
    db,
    referralCode.owner_source,
    referralCode.owner_subject_ref,
  );
  return {
    pass_id: referralCode.referral_id,
    invite_count: Math.min(inviteCount, TALK_TOGETHER_PASS_INVITE_LIMIT),
    invite_limit: TALK_TOGETHER_PASS_INVITE_LIMIT,
    bonus_translations_per_friend: TALK_TOGETHER_PASS_BONUS_TRANSLATIONS_PER_FRIEND,
  };
}

async function countCountedIssueReferralRewardsForReferrer(
  db: D1Database,
  referrerSource: ReferralSource,
  referrerSubjectRef: string,
): Promise<number> {
  const row = await db
    .prepare(
      `SELECT COUNT(*) AS count
         FROM referral_rewards counted
        WHERE counted.referrer_source = ?
          AND counted.referrer_subject_ref = ?
          AND counted.referred_bonus_status IN ('reserved', 'credited')`,
    )
    .bind(referrerSource, referrerSubjectRef)
    .first<{ count: number }>();

  return Math.max(0, Number(row?.count ?? 0));
}

export async function countCountedQqReferralRewardsSince(
  db: D1Database,
  windowStartIso: string,
): Promise<number> {
  const row = await db
    .prepare(
      `SELECT COUNT(*) AS count
         FROM referral_rewards
        WHERE referred_source = 'qq'
          AND referred_bonus_status IN ('reserved', 'credited')
          AND created_at >= ?`,
    )
    .bind(windowStartIso)
    .first<{ count: number }>();
  return Math.max(0, Number(row?.count ?? 0));
}

async function hasReachedIssueReferralCap(
  db: D1Database,
  referrerSource: ReferralSource,
  referrerSubjectRef: string,
): Promise<boolean> {
  return (
    (await countCountedIssueReferralRewardsForReferrer(
      db,
      referrerSource,
      referrerSubjectRef,
    )) >= TALK_TOGETHER_PASS_INVITE_LIMIT
  );
}

async function insertSkippedIssueReferralReward(
  db: D1Database,
  input: {
    referralId: string;
    referrerSource: ReferralSource | null;
    referrerSubjectRef: string | null;
    referrerInstallationId: string | null;
    referredSource: ReferralSource;
    referredSubjectRef: string;
    referredInstallationId: string | null;
    referredHardwareHash: string | null;
    referredHardwareHashSaltVersion: number | null;
    skipReason: IssueReferralSkipReason;
    attemptIpDigest?: ReferralAttemptIpDigest | null;
    attemptIpLegacyHash?: string | null;
    operationId?: string | null;
    nowIso: string;
  },
): Promise<void> {
  assertIssueReferralSkipReason(input.skipReason);
  const skippedIdentityMode = (await resolveNetworkIdentityWriteMode(db)) === 'keyed' ? 'keyed' : 'dual';
  const skippedIdentityColumns = referralAttemptIdentityColumnSet(skippedIdentityMode);
  const skippedIdentityPlaceholders = referralAttemptIdentityBinds(skippedIdentityMode, input.attemptIpDigest, input.attemptIpLegacyHash).map(() => '?').join(', ');
  await db
    .prepare(
      `INSERT INTO referral_rewards (
          referral_id,
          referrer_source,
          referrer_subject_ref,
          referrer_installation_id,
          referred_source,
          referred_subject_ref,
          referred_installation_id,
          referred_hardware_hash,
          referred_hardware_hash_salt_version,
          referred_bonus_status,
          referrer_bonus_status,
          skip_reason,
          failure_reason,
          referred_managed_credential_ref,
          referrer_managed_credential_ref,
          ${skippedIdentityColumns.columns},
          operation_id,
          created_at,
          updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'skipped', 'skipped', ?, NULL, NULL, NULL, ${skippedIdentityPlaceholders}, ?, ?, ?)`,
    )
    .bind(
      input.referralId,
      input.referrerSource,
      input.referrerSubjectRef,
      input.referrerInstallationId,
      input.referredSource,
      input.referredSubjectRef,
      input.referredInstallationId,
      input.referredHardwareHash,
      input.referredHardwareHashSaltVersion,
      input.skipReason,
      ...referralAttemptIdentityBinds(skippedIdentityMode, input.attemptIpDigest, input.attemptIpLegacyHash),
      input.operationId ?? null,
      input.nowIso,
      input.nowIso,
    )
    .run();
  logReferralRewardOutcome({
    outcome: 'skipped',
    referralId: input.referralId,
    referredInstallationId: input.referredInstallationId,
    referrerSource: input.referrerSource,
    referrerSubjectRef: input.referrerSubjectRef,
    reason: input.skipReason,
  });
}

async function getCreditedIssueReferralReferrer(
  db: D1Database,
  input: {
    referralId: string;
    referredSource: ReferralSource;
    referredSubjectRef: string;
    referredInstallationId: string | null;
  },
): Promise<ReferralSubject | null> {
  const row = await db
    .prepare(
      `SELECT referrer_source,
              referrer_subject_ref,
              referrer_installation_id
         FROM referral_rewards
        WHERE referral_id = ?
          AND referred_source = ?
          AND referred_subject_ref = ?
          AND referred_installation_id IS ?
          AND referred_bonus_status = 'credited'
          AND referrer_bonus_status IN ('pending', 'applying', 'credited')`,
    )
    .bind(
      input.referralId,
      input.referredSource,
      input.referredSubjectRef,
      input.referredInstallationId,
    )
    .first<{
      referrer_source: ReferralSource | null;
      referrer_subject_ref: string | null;
      referrer_installation_id: string | null;
    }>();

  if (!row?.referrer_source || !row.referrer_subject_ref) {
    return null;
  }
  return {
    source: row.referrer_source,
    subjectRef: row.referrer_subject_ref,
    installationId: row.referrer_installation_id,
  };
}

async function getActiveReferrerRewardEntitlement(
  db: D1Database,
  input: {
    referrerSource: ReferralSource;
    referrerSubjectRef: string;
    nowIso: string;
  },
): Promise<ActiveManagedReferralOwner | null> {
  const now = new Date(input.nowIso);
  if (Number.isNaN(now.getTime())) {
    return null;
  }

  if (input.referrerSource === 'discord') {
    const row = await db
      .prepare(
        `SELECT installation_id,
                status,
                budget_usd,
                managed_credential_ref,
                issued_at,
                expires_at,
                release_session_ref,
                release_token_hash,
                release_token_expires_at,
                verified_hardware_hash,
                verified_hardware_hash_salt_version,
                discord_user_ref,
                discord_issue_status,
                discord_issue_reserved_at,
                discord_issue_delivered_at
           FROM openrouter_entitlements
          WHERE discord_user_ref = ?
            AND status = 'active'
            AND discord_issue_status = 'active'
            AND managed_credential_ref IS NOT NULL
            AND length(trim(managed_credential_ref)) > 0`,
      )
      .bind(input.referrerSubjectRef)
      .first<OpenRouterEntitlementRecord>();

    if (
      !row?.managed_credential_ref ||
      !row.expires_at ||
      !row.discord_user_ref ||
      resolveEffectiveEntitlementLifecycle(row, now) !== 'active'
    ) {
      return null;
    }
    return {
      source: 'discord',
      subjectRef: row.discord_user_ref,
      installationId: row.installation_id,
      entitlementRef: row.installation_id,
      managedCredentialRef: row.managed_credential_ref,
      budgetUsd: row.budget_usd,
      expiresAt: row.expires_at,
    };
  }

  const row = await db
    .prepare(
      `SELECT qq_subject_ref,
              issue_ref,
              managed_credential_ref,
              budget_usd,
              expires_at
         FROM qq_managed_entitlements
        WHERE qq_subject_ref = ?
          AND status = 'active'
          AND managed_credential_ref IS NOT NULL
          AND length(trim(managed_credential_ref)) > 0
          AND delivered_at IS NOT NULL
          AND expires_at IS NOT NULL`,
    )
    .bind(input.referrerSubjectRef)
    .first<{
      qq_subject_ref: string;
      issue_ref: string;
      managed_credential_ref: string;
      budget_usd: number;
      expires_at: string;
    }>();

  if (!row || new Date(row.expires_at).getTime() < now.getTime()) {
    return null;
  }
  return {
    source: 'qq',
    subjectRef: row.qq_subject_ref,
    installationId: null,
    entitlementRef: row.issue_ref,
    managedCredentialRef: row.managed_credential_ref,
    budgetUsd: row.budget_usd,
    expiresAt: row.expires_at,
  };
}

async function claimReferrerRewardApplicationLease(
  db: D1Database,
  input: {
    referrerSource: ReferralSource;
    referrerSubjectRef: string;
    managedCredentialRef: string;
    nowIso: string;
    leaseCutoffIso: string;
  },
): Promise<number> {
  const result = await db
    .prepare(
      `UPDATE referral_rewards
          SET referrer_bonus_status = 'applying',
              referrer_managed_credential_ref = ?,
              failure_reason = NULL,
              updated_at = ?
        WHERE referrer_source = ?
          AND referrer_subject_ref = ?
          AND referred_bonus_status = 'credited'
          AND (
            referrer_bonus_status = 'pending'
            OR (
              referrer_bonus_status = 'applying'
              AND updated_at < ?
            )
          )
          AND (
            referrer_managed_credential_ref IS NULL
            OR referrer_managed_credential_ref = ?
          )
          AND NOT EXISTS (
            SELECT 1
              FROM referral_rewards active_lease
             WHERE active_lease.referrer_source = ?
               AND active_lease.referrer_subject_ref = ?
               AND active_lease.referred_bonus_status = 'credited'
               AND active_lease.referrer_bonus_status = 'applying'
               AND active_lease.updated_at >= ?
               AND (
                 active_lease.referrer_managed_credential_ref IS NULL
                 OR active_lease.referrer_managed_credential_ref = ?
               )
          )`,
    )
    .bind(
      input.managedCredentialRef,
      input.nowIso,
      input.referrerSource,
      input.referrerSubjectRef,
      input.leaseCutoffIso,
      input.managedCredentialRef,
      input.referrerSource,
      input.referrerSubjectRef,
      input.leaseCutoffIso,
      input.managedCredentialRef,
    )
    .run();

  return Number(result.meta.changes ?? 0);
}

async function hasActiveReferrerRewardApplicationLease(
  db: D1Database,
  input: {
    referrerSource: ReferralSource;
    referrerSubjectRef: string;
    managedCredentialRef: string;
    leaseCutoffIso: string;
  },
): Promise<boolean> {
  const row = await db
    .prepare(
      `SELECT EXISTS(
          SELECT 1
            FROM referral_rewards active_lease
           WHERE active_lease.referrer_source = ?
             AND active_lease.referrer_subject_ref = ?
             AND active_lease.referred_bonus_status = 'credited'
             AND active_lease.referrer_bonus_status = 'applying'
             AND active_lease.updated_at >= ?
             AND (
               active_lease.referrer_managed_credential_ref IS NULL
               OR active_lease.referrer_managed_credential_ref = ?
             )
        ) AS active_found`,
    )
    .bind(
      input.referrerSource,
      input.referrerSubjectRef,
      input.leaseCutoffIso,
      input.managedCredentialRef,
    )
    .first<{ active_found: number }>();

  return Number(row?.active_found ?? 0) === 1;
}

async function hasPendingReferrerRewardRows(
  db: D1Database,
  input: {
    referrerSource: ReferralSource;
    referrerSubjectRef: string;
  },
): Promise<boolean> {
  const row = await db
    .prepare(
      `SELECT EXISTS(
          SELECT 1
            FROM referral_rewards pending_reward
           WHERE pending_reward.referrer_source = ?
             AND pending_reward.referrer_subject_ref = ?
             AND pending_reward.referred_bonus_status = 'credited'
             AND pending_reward.referrer_bonus_status = 'pending'
        ) AS pending_found`,
    )
    .bind(input.referrerSource, input.referrerSubjectRef)
    .first<{ pending_found: number }>();

  return Number(row?.pending_found ?? 0) === 1;
}

async function countReferrerRewardsForTargetLimit(
  db: D1Database,
  input: {
    referrerSource: ReferralSource;
    referrerSubjectRef: string;
    managedCredentialRef: string;
  },
): Promise<number> {
  const row = await db
    .prepare(
      `SELECT COUNT(*) AS count
         FROM referral_rewards
        WHERE referrer_source = ?
          AND referrer_subject_ref = ?
          AND referred_bonus_status = 'credited'
          AND referrer_bonus_status IN ('pending', 'applying', 'credited')
          AND (
            referrer_managed_credential_ref IS NULL
            OR referrer_managed_credential_ref = ?
          )`,
    )
    .bind(input.referrerSource, input.referrerSubjectRef, input.managedCredentialRef)
    .first<{ count: number }>();

  return Number(row?.count ?? 0);
}

async function updateReferrerEntitlementBudget(
  db: D1Database,
  input: {
    owner: ActiveManagedReferralOwner;
    managedCredentialRef: string;
    budgetUsd: number;
  },
): Promise<boolean> {
  if (input.owner.source === 'discord') {
    if (!input.owner.installationId) {
      return false;
    }
    const result = await db
      .prepare(
        `UPDATE openrouter_entitlements
            SET budget_usd = ?
          WHERE installation_id = ?
            AND discord_user_ref = ?
            AND status = 'active'
            AND discord_issue_status = 'active'
            AND managed_credential_ref = ?`,
      )
      .bind(
        input.budgetUsd,
        input.owner.installationId,
        input.owner.subjectRef,
        input.managedCredentialRef,
      )
      .run();
    return Number(result.meta.changes ?? 0) === 1;
  }

  const result = await db
    .prepare(
      `UPDATE qq_managed_entitlements
          SET budget_usd = ?
        WHERE qq_subject_ref = ?
          AND issue_ref = ?
          AND status = 'active'
          AND managed_credential_ref = ?`,
    )
    .bind(
      input.budgetUsd,
      input.owner.subjectRef,
      input.owner.entitlementRef,
      input.managedCredentialRef,
    )
    .run();
  return Number(result.meta.changes ?? 0) === 1;
}

async function markReferrerRewardRowsCredited(
  db: D1Database,
  input: {
    referrerSource: ReferralSource;
    referrerSubjectRef: string;
    managedCredentialRef: string;
    nowIso: string;
  },
): Promise<number> {
  const result = await db
    .prepare(
      `UPDATE referral_rewards
          SET referrer_bonus_status = 'credited',
              referrer_managed_credential_ref = ?,
              failure_reason = NULL,
              updated_at = ?
        WHERE referrer_source = ?
          AND referrer_subject_ref = ?
          AND referred_bonus_status = 'credited'
          AND referrer_bonus_status = 'applying'
          AND (
            referrer_managed_credential_ref IS NULL
            OR referrer_managed_credential_ref = ?
          )`,
    )
    .bind(
      input.managedCredentialRef,
      input.nowIso,
      input.referrerSource,
      input.referrerSubjectRef,
      input.managedCredentialRef,
    )
    .run();

  return Number(result.meta.changes ?? 0);
}

async function markReferrerRewardRowsFailed(
  db: D1Database,
  input: {
    referrerSource: ReferralSource;
    referrerSubjectRef: string;
    managedCredentialRef: string;
    nowIso: string;
    failureReason: 'referrer_patch_failed';
  },
): Promise<number> {
  assertIssueReferralFailureReason(input.failureReason);
  const result = await db
    .prepare(
      `UPDATE referral_rewards
          SET referrer_bonus_status = 'failed',
              referrer_managed_credential_ref = ?,
              failure_reason = ?,
              updated_at = ?
        WHERE referrer_source = ?
          AND referrer_subject_ref = ?
          AND referred_bonus_status = 'credited'
          AND referrer_bonus_status = 'applying'
          AND (
            referrer_managed_credential_ref IS NULL
            OR referrer_managed_credential_ref = ?
          )`,
    )
    .bind(
      input.managedCredentialRef,
      input.failureReason,
      input.nowIso,
      input.referrerSource,
      input.referrerSubjectRef,
      input.managedCredentialRef,
    )
    .run();

  const failedRows = Number(result.meta.changes ?? 0);
  if (failedRows > 0) {
    logReferralRewardOutcome({
      outcome: 'failed',
      referrerSource: input.referrerSource,
      referrerSubjectRef: input.referrerSubjectRef,
      referrerManagedCredentialRef: input.managedCredentialRef,
      reason: input.failureReason,
      affectedRows: failedRows,
    });
  }
  return failedRows;
}

async function markReferrerRewardRowsSkipped(
  db: D1Database,
  input: {
    referrerSource: ReferralSource;
    referrerSubjectRef: string;
    nowIso: string;
    skipReason: 'referrer_managed_key_missing';
  },
): Promise<number> {
  const result = await db
    .prepare(
      `UPDATE referral_rewards
          SET referrer_bonus_status = 'skipped',
              skip_reason = ?,
              updated_at = ?
        WHERE referrer_source = ?
          AND referrer_subject_ref = ?
          AND referred_bonus_status = 'credited'
          AND referrer_bonus_status IN ('pending', 'applying')`,
    )
    .bind(input.skipReason, input.nowIso, input.referrerSource, input.referrerSubjectRef)
    .run();

  const skippedRows = Number(result.meta.changes ?? 0);
  if (skippedRows > 0) {
    logReferralRewardOutcome({
      outcome: 'skipped',
      referrerSource: input.referrerSource,
      referrerSubjectRef: input.referrerSubjectRef,
      reason: input.skipReason,
      affectedRows: skippedRows,
    });
  }
  return skippedRows;
}

async function listStaleReservedReferralRewards(
  db: D1Database,
  cutoffIso: string,
): Promise<ReferralRewardRecord[]> {
  const result = await db
    .prepare(
      `SELECT id,
              referral_id,
              referrer_source,
              referrer_subject_ref,
              referrer_installation_id,
              referred_source,
              referred_subject_ref,
              referred_installation_id,
              referred_hardware_hash,
              referred_hardware_hash_salt_version,
              referred_bonus_status,
              referrer_bonus_status,
              skip_reason,
              failure_reason,
              referred_managed_credential_ref,
              referrer_managed_credential_ref,
              attempt_ip_digest,
          attempt_ip_key_version,
          attempt_ip_epoch,
              created_at,
              updated_at,
              credited_at
         FROM referral_rewards
        WHERE referred_bonus_status = 'reserved'
          AND referred_source <> 'qq'
          AND updated_at < ?
        ORDER BY id ASC`,
    )
    .bind(cutoffIso)
    .all<ReferralRewardRecord>();
  return result.results;
}

async function getDeliveredReferredEntitlement(
  db: D1Database,
  reward: ReferralRewardRecord,
): Promise<{ managed_credential_ref: string } | null> {
  const row =
    reward.referred_source === 'discord'
      ? await db
          .prepare(
            `SELECT managed_credential_ref
               FROM openrouter_entitlements
              WHERE installation_id = ?
                AND discord_user_ref = ?
                AND status = 'active'
                AND discord_issue_status = 'active'
                AND managed_credential_ref IS NOT NULL
                AND length(trim(managed_credential_ref)) > 0`,
          )
          .bind(reward.referred_installation_id, reward.referred_subject_ref)
          .first<{ managed_credential_ref: string }>()
      : await db
          .prepare(
            `SELECT managed_credential_ref
               FROM qq_managed_entitlements
              WHERE qq_subject_ref = ?
                AND status = 'active'
                AND delivered_at IS NOT NULL
                AND managed_credential_ref IS NOT NULL
                AND length(trim(managed_credential_ref)) > 0
                AND budget_usd >= ?`,
          )
          .bind(reward.referred_subject_ref, referrerRewardTargetLimitUsd(1))
          .first<{ managed_credential_ref: string }>();

  if (
    !row ||
    (reward.referred_managed_credential_ref &&
      reward.referred_managed_credential_ref !== row.managed_credential_ref)
  ) {
    return null;
  }
  return row;
}

async function reconcileStaleReservedReferralToCredited(
  db: D1Database,
  input: {
    rewardId: number;
    referralId: string;
    referredInstallationId: string | null;
    referrerSource: ReferralSource | null;
    referrerSubjectRef: string | null;
    managedCredentialRef: string;
    expectedUpdatedAt: string;
    expectedFailureReason: string | null;
    nowIso: string;
  },
): Promise<number> {
  const result = await db
    .prepare(
      `UPDATE referral_rewards
          SET referred_bonus_status = 'credited',
              referred_managed_credential_ref = ?,
              failure_reason = NULL,
              updated_at = ?,
              credited_at = COALESCE(credited_at, ?)
        WHERE id = ?
          AND referred_bonus_status = 'reserved'
          AND updated_at = ?
          AND failure_reason IS ?`,
    )
    .bind(
      input.managedCredentialRef,
      input.nowIso,
      input.nowIso,
      input.rewardId,
      input.expectedUpdatedAt,
      input.expectedFailureReason,
    )
    .run();
  const changed = Number(result.meta.changes ?? 0);
  if (changed > 0) {
    logReferralRewardOutcome({
      outcome: 'credited',
      referralId: input.referralId,
      referredInstallationId: input.referredInstallationId,
      referrerSource: input.referrerSource,
      referrerSubjectRef: input.referrerSubjectRef,
      reason: 'stale_reserved_reconciled',
    });
  }
  return changed;
}

async function reconcileStaleReservedReferralToFailed(
  db: D1Database,
  input: {
    rewardId: number;
    referralId: string;
    referredInstallationId: string | null;
    referrerSource: ReferralSource | null;
    referrerSubjectRef: string | null;
    expectedUpdatedAt: string;
    expectedFailureReason: string | null;
    nowIso: string;
  },
): Promise<number> {
  const failureReason = 'stale_reserved_reconciled';
  assertIssueReferralFailureReason(failureReason);
  const result = await db
    .prepare(
      `UPDATE referral_rewards
          SET referred_bonus_status = 'failed',
              referrer_bonus_status = 'failed',
              failure_reason = ?,
              updated_at = ?
        WHERE id = ?
          AND referred_bonus_status = 'reserved'
          AND updated_at = ?
          AND failure_reason IS ?`,
    )
    .bind(
      failureReason,
      input.nowIso,
      input.rewardId,
      input.expectedUpdatedAt,
      input.expectedFailureReason,
    )
    .run();
  const changed = Number(result.meta.changes ?? 0);
  if (changed > 0) {
    logReferralRewardOutcome({
      outcome: 'failed',
      referralId: input.referralId,
      referredInstallationId: input.referredInstallationId,
      referrerSource: input.referrerSource,
      referrerSubjectRef: input.referrerSubjectRef,
      reason: failureReason,
    });
  }
  return changed;
}

async function requeueStaleApplyingReferralRewards(
  db: D1Database,
  input: { cutoffIso: string; nowIso: string },
): Promise<number> {
  const result = await db
    .prepare(
      `UPDATE referral_rewards
          SET referrer_bonus_status = 'pending',
              referrer_managed_credential_ref = NULL,
              failure_reason = NULL,
              updated_at = ?
        WHERE referred_bonus_status = 'credited'
          AND referred_source <> 'qq'
          AND referrer_bonus_status = 'applying'
          AND updated_at < ?`,
    )
    .bind(input.nowIso, input.cutoffIso)
    .run();
  const changed = Number(result.meta.changes ?? 0);
  if (changed > 0) {
    logReferralRewardOutcome({
      outcome: 'pending',
      reason: 'stale_applying_requeued',
      affectedRows: changed,
    });
  }
  return changed;
}

async function deleteTerminalReferralRewardsOlderThan(
  db: D1Database,
  input: {
    referredBonusStatus: 'skipped' | 'failed';
    cutoffIso: string;
  },
): Promise<number> {
  const result = await db
    .prepare(
      `DELETE FROM referral_rewards
        WHERE referred_bonus_status = ?
          AND updated_at < ?`,
    )
    .bind(input.referredBonusStatus, input.cutoffIso)
    .run();
  return Number(result.meta.changes ?? 0);
}

function normalizeReferralDisableReason(value: unknown): ReferralDisableReason | null {
  if (typeof value !== 'string') {
    return null;
  }

  const normalized = value.trim();
  return (REFERRAL_DISABLE_REASONS as readonly string[]).includes(normalized)
    ? (normalized as ReferralDisableReason)
    : null;
}

function normalizeReferralDisableActor(value: unknown): string | null {
  if (typeof value !== 'string') {
    return null;
  }

  const normalized = value.trim();
  return REFERRAL_DISABLE_ACTOR_PATTERN.test(normalized) ? normalized : null;
}

async function appendReferralRuntimeAudit(
  db: D1Database,
  input: {
    eventKind: 'referral_id_disabled';
    reason: ReferralDisableReason;
    payload: Record<string, unknown>;
    createdAt: string;
  },
): Promise<void> {
  await db
    .prepare(
      `INSERT INTO broker_abuse_runtime_audit (
          event_kind,
          reason,
          payload_json,
          created_at
        ) VALUES (?, ?, ?, ?)`,
    )
    .bind(
      input.eventKind,
      input.reason,
      JSON.stringify(input.payload),
      input.createdAt,
    )
    .run();
}

function referrerRewardTargetLimitUsd(reflectedRewardCount: number): number {
  return usdFromCents(
    currencyCents(MANAGED_TRIAL_BUDGET_POLICY.hardLimit) +
      reflectedRewardCount * REFERRER_REFERRAL_REWARD_CENTS,
  );
}

function maxUsd(...values: number[]): number {
  return usdFromCents(Math.max(...values.map(currencyCents)));
}

function currencyCents(value: number): number {
  if (!Number.isFinite(value) || value < 0) {
    throw new Error('managed budget must be a finite non-negative USD value');
  }

  return Math.round(value * USD_CENTS);
}

function usdFromCents(cents: number): number {
  return Number((cents / USD_CENTS).toFixed(2));
}

function assertIssueReferralSkipReason(reason: IssueReferralSkipReason): void {
  if (!(ISSUE_REFERRAL_SKIP_REASONS as readonly string[]).includes(reason)) {
    throw new Error('unbounded issue referral skip reason');
  }
}

function assertIssueReferralFailureReason(reason: IssueReferralFailureReason): void {
  if (!(ISSUE_REFERRAL_FAILURE_REASONS as readonly string[]).includes(reason)) {
    throw new Error('unbounded issue referral failure reason');
  }
}

function logReferralRewardOutcome(input: {
  outcome:
    | 'reserved'
    | 'skipped'
    | 'failed'
    | 'applying'
    | 'pending'
    | 'credited'
    | 'disabled';
  referralId?: string | null;
  referredInstallationId?: string | null;
  referrerSource?: ReferralSource | null;
  referrerSubjectRef?: string | null;
  referrerManagedCredentialRef?: string | null;
  reason?: string | null;
  affectedRows?: number;
}): void {
  const payload: Record<string, string | number | null> = {
    outcome: input.outcome,
    broker_timestamp: new Date().toISOString(),
  };

  if (input.referralId) {
    payload.referral_id = input.referralId;
  }
  if (input.referredInstallationId) {
    payload.referred_installation_id = input.referredInstallationId;
  }
  if (input.referrerSource) {
    payload.referrer_source = input.referrerSource;
  }
  if (input.referrerSubjectRef) {
    payload.referrer_subject_ref = input.referrerSubjectRef;
  }
  if (input.referrerManagedCredentialRef) {
    payload.referrer_managed_credential_ref = input.referrerManagedCredentialRef;
  }
  if (input.reason) {
    payload.reason = boundLogReason(input.reason);
  }
  if (input.affectedRows !== undefined) {
    payload.affected_rows = input.affectedRows;
  }

  console.info(REFERRAL_REWARD_LOG_EVENT, payload);
}

function boundLogReason(reason: string): string {
  const normalized = reason.trim();
  if (/^[a-z0-9_:-]{1,64}$/u.test(normalized)) {
    return normalized;
  }

  return 'unclassified';
}

export type ActiveManagedReferralOwnerLookup =
  | { source: 'discord'; installationId: string }
  | { source: 'qq'; subjectRef: string };

export async function ensureOwnedReferralIdForActiveDiscordManagedUser(
  db: D1Database,
  input: {
    installationId: string;
    nowIso: string;
    generateReferralId?: ReferralIdGenerator;
    maxCollisionAttempts?: number;
  },
): Promise<OwnedReferralIdEnsureResult> {
  return ensureOwnedReferralIdForActiveManagedSubject(db, {
    owner: { source: 'discord', installationId: input.installationId },
    nowIso: input.nowIso,
    generateReferralId: input.generateReferralId,
    maxCollisionAttempts: input.maxCollisionAttempts,
  });
}

export async function ensureOwnedReferralIdForActiveQqManagedUser(
  db: D1Database,
  input: {
    qqSubjectRef: string;
    ownerInstallationId?: string | null;
    nowIso: string;
    generateReferralId?: ReferralIdGenerator;
    maxCollisionAttempts?: number;
  },
): Promise<OwnedReferralIdEnsureResult> {
  return ensureOwnedReferralIdForActiveManagedSubject(db, {
    owner: { source: 'qq', subjectRef: input.qqSubjectRef },
    ownerInstallationId: input.ownerInstallationId ?? null,
    nowIso: input.nowIso,
    generateReferralId: input.generateReferralId,
    maxCollisionAttempts: input.maxCollisionAttempts,
  });
}

export async function resolveOwnedReferralStatusForManagedSubject(
  db: D1Database,
  input: { source: ReferralSource; subjectRef: string },
): Promise<{
  referralCode: ReferralCodeRecord;
  talkTogetherPass: TalkTogetherPassStatusResponse;
} | null> {
  const referralCode = await getActiveReferralCodeForOwner(
    db,
    input.source,
    input.subjectRef,
  );
  if (!referralCode) {
    return null;
  }
  return {
    referralCode,
    talkTogetherPass: await resolveTalkTogetherPassStatusForOwnedReferralCode(
      db,
      referralCode,
    ),
  };
}

export async function ensureOwnedReferralIdForActiveManagedSubject(
  db: D1Database,
  input: {
    owner: ActiveManagedReferralOwnerLookup;
    ownerInstallationId?: string | null;
    nowIso: string;
    generateReferralId?: ReferralIdGenerator;
    maxCollisionAttempts?: number;
  },
): Promise<OwnedReferralIdEnsureResult> {
  const activeOwner = await getActiveManagedReferralOwner(db, input.owner, input.nowIso);
  if (!activeOwner) {
    return { ok: false, reason: 'not_eligible' };
  }
  const owner: ActiveManagedReferralOwner = {
    ...activeOwner,
    installationId:
      input.owner.source === 'qq'
        ? (input.ownerInstallationId ?? activeOwner.installationId)
        : activeOwner.installationId,
  };

  const subjectRef = owner.subjectRef.trim();
  if (!isPersistableOwnedSubjectRef(owner.source, subjectRef)) {
    return { ok: false, reason: 'unsafe_subject_ref' };
  }

  const existing = await getReferralCodeForOwner(db, owner.source, subjectRef);
  if (existing) {
    if (existing.status === 'disabled') {
      return { ok: false, reason: 'disabled' };
    }

    const refreshed = await refreshActiveReferralCodeOwnerInstallation(db, {
      referralId: existing.referral_id,
      source: owner.source,
      subjectRef,
      installationId: owner.installationId,
      nowIso: input.nowIso,
    });
    if (!refreshed) {
      const latest = await getReferralCodeForOwner(db, owner.source, subjectRef);
      if (latest?.status === 'disabled') {
        return { ok: false, reason: 'disabled' };
      }
      return { ok: false, reason: 'not_eligible' };
    }

    return { ok: true, referralCode: refreshed, created: false };
  }

  const createReferralId = input.generateReferralId ?? generateReferralId;
  const maxCollisionAttempts =
    input.maxCollisionAttempts ?? DEFAULT_REFERRAL_ID_COLLISION_ATTEMPTS;

  for (let attempt = 0; attempt < maxCollisionAttempts; attempt += 1) {
    const referralId = normalizeReferralId(createReferralId());
    if (!referralId) {
      throw new Error('generated Referral ID did not match the approved format');
    }

    const inserted = await insertActiveOwnedReferralCode(db, {
      referralId,
      source: owner.source,
      subjectRef,
      installationId: owner.installationId,
      nowIso: input.nowIso,
    });
    if (inserted) {
      const created = await getActiveReferralCodeForOwner(
        db,
        owner.source,
        subjectRef,
      );
      if (!created) {
        const latest = await getReferralCodeForOwner(db, owner.source, subjectRef);
        if (latest?.status === 'disabled') {
          return { ok: false, reason: 'disabled' };
        }
        throw new Error('created Referral ID could not be read back as active');
      }
      return { ok: true, referralCode: created, created: true };
    }

    const concurrentlyCreated = await getReferralCodeForOwner(
      db,
      owner.source,
      subjectRef,
    );
    if (concurrentlyCreated) {
      if (concurrentlyCreated.status === 'disabled') {
        return { ok: false, reason: 'disabled' };
      }
      return { ok: true, referralCode: concurrentlyCreated, created: false };
    }
  }

  return { ok: false, reason: 'collision_exhausted' };
}

function cryptoReferralRandomBytes(byteLength: number): Uint8Array {
  const bytes = new Uint8Array(byteLength);
  crypto.getRandomValues(bytes);
  return bytes;
}

async function getActiveManagedReferralOwner(
  db: D1Database,
  lookup: ActiveManagedReferralOwnerLookup,
  nowIso: string,
): Promise<ActiveManagedReferralOwner | null> {
  return lookup.source === 'discord'
    ? getActiveDiscordManagedReferralOwner(db, nowIso, lookup.installationId)
    : getActiveQqManagedReferralOwner(db, nowIso, lookup.subjectRef);
}

async function getActiveDiscordManagedReferralOwner(
  db: D1Database,
  nowIso: string,
  installationId: string,
): Promise<ActiveManagedReferralOwner | null> {
  const row = await db
    .prepare(
      `SELECT entitlement.installation_id,
              entitlement.status,
              entitlement.budget_usd,
              entitlement.managed_credential_ref,
              entitlement.issued_at,
              entitlement.expires_at,
              entitlement.release_session_ref,
              entitlement.release_token_hash,
              entitlement.release_token_expires_at,
              entitlement.verified_hardware_hash,
              entitlement.verified_hardware_hash_salt_version,
              entitlement.discord_user_ref,
              entitlement.discord_issue_status,
              entitlement.discord_issue_reserved_at,
              entitlement.discord_issue_delivered_at
         FROM openrouter_entitlements entitlement
         JOIN discord_identities identity
           ON identity.discord_user_ref = entitlement.discord_user_ref
        WHERE entitlement.installation_id = ?
          AND entitlement.status = 'active'
          AND entitlement.discord_user_ref IS NOT NULL
          AND length(trim(entitlement.discord_user_ref)) > 0
          AND entitlement.managed_credential_ref IS NOT NULL
          AND length(trim(entitlement.managed_credential_ref)) > 0
          AND entitlement.expires_at IS NOT NULL
          AND length(trim(entitlement.expires_at)) > 0
          AND entitlement.discord_issue_status = 'active'
          AND entitlement.discord_issue_delivered_at IS NOT NULL
          AND length(trim(entitlement.discord_issue_delivered_at)) > 0
          AND identity.status = 'active'
          AND identity.entitlement_installation_id = entitlement.installation_id`,
    )
    .bind(installationId)
    .first<OpenRouterEntitlementRecord>();

  const now = new Date(nowIso);
  if (
    !row?.discord_user_ref ||
    !row.managed_credential_ref ||
    !row.expires_at ||
    Number.isNaN(now.getTime()) ||
    resolveEffectiveEntitlementLifecycle(row, now) !== 'active'
  ) {
    return null;
  }

  return {
    source: 'discord',
    subjectRef: row.discord_user_ref,
    installationId: row.installation_id,
    entitlementRef: row.installation_id,
    managedCredentialRef: row.managed_credential_ref,
    budgetUsd: row.budget_usd,
    expiresAt: row.expires_at,
  };
}

async function getActiveQqManagedReferralOwner(
  db: D1Database,
  nowIso: string,
  subjectRef: string,
): Promise<ActiveManagedReferralOwner | null> {
  const row = await db
    .prepare(
      `SELECT qq_subject_ref,
              issue_ref,
              managed_credential_ref,
              budget_usd,
              expires_at
         FROM qq_managed_entitlements
        WHERE qq_subject_ref = ?
          AND status = 'active'
          AND managed_credential_ref IS NOT NULL
          AND length(trim(managed_credential_ref)) > 0
          AND delivered_at IS NOT NULL
          AND expires_at IS NOT NULL`,
    )
    .bind(subjectRef)
    .first<{
      qq_subject_ref: string;
      issue_ref: string;
      managed_credential_ref: string;
      budget_usd: number;
      expires_at: string;
    }>();
  const now = new Date(nowIso);
  if (
    !row ||
    Number.isNaN(now.getTime()) ||
    new Date(row.expires_at).getTime() < now.getTime()
  ) {
    return null;
  }

  return {
    source: 'qq',
    subjectRef: row.qq_subject_ref,
    installationId: null,
    entitlementRef: row.issue_ref,
    managedCredentialRef: row.managed_credential_ref,
    budgetUsd: row.budget_usd,
    expiresAt: row.expires_at,
  };
}

function isPersistableOwnedSubjectRef(
  source: ReferralSource,
  value: string,
): boolean {
  return source === 'discord'
    ? OWNED_DISCORD_USER_REF_PATTERN.test(value)
    : OWNED_QQ_SUBJECT_REF_PATTERN.test(value);
}

async function getReferralCodeForOwner(
  db: D1Database,
  source: ReferralSource,
  subjectRef: string,
): Promise<ReferralCodeRecord | null> {
  return db
    .prepare(
      `SELECT referral_id,
              owner_source,
              owner_subject_ref,
              owner_installation_id,
              status,
              created_at,
              updated_at
         FROM referral_codes
        WHERE owner_source = ?
          AND owner_subject_ref = ?`,
    )
    .bind(source, subjectRef)
    .first<ReferralCodeRecord>();
}

async function getActiveReferralCodeForOwner(
  db: D1Database,
  source: ReferralSource,
  subjectRef: string,
): Promise<ReferralCodeRecord | null> {
  return db
    .prepare(
      `SELECT referral_id,
              owner_source,
              owner_subject_ref,
              owner_installation_id,
              status,
              created_at,
              updated_at
         FROM referral_codes
        WHERE owner_source = ?
          AND owner_subject_ref = ?
          AND status = 'active'`,
    )
    .bind(source, subjectRef)
    .first<ReferralCodeRecord>();
}

async function refreshActiveReferralCodeOwnerInstallation(
  db: D1Database,
  input: {
    referralId: string;
    source: ReferralSource;
    subjectRef: string;
    installationId: string | null;
    nowIso: string;
  },
): Promise<ReferralCodeRecord | null> {
  await db
    .prepare(
      `UPDATE referral_codes
          SET owner_installation_id = ?,
              updated_at = ?
        WHERE referral_id = ?
          AND owner_source = ?
          AND owner_subject_ref = ?
          AND status = 'active'
          AND owner_installation_id IS NOT ?`,
    )
    .bind(
      input.installationId,
      input.nowIso,
      input.referralId,
      input.source,
      input.subjectRef,
      input.installationId,
    )
    .run();

  return getActiveReferralCodeForOwner(db, input.source, input.subjectRef);
}

async function insertActiveOwnedReferralCode(
  db: D1Database,
  input: {
    referralId: string;
    source: ReferralSource;
    subjectRef: string;
    installationId: string | null;
    nowIso: string;
  },
): Promise<boolean> {
  const result = await db
    .prepare(
      `INSERT OR IGNORE INTO referral_codes (
          referral_id,
          owner_source,
          owner_subject_ref,
          owner_installation_id,
          status,
          created_at,
          updated_at
        ) VALUES (?, ?, ?, ?, 'active', ?, ?)`,
    )
    .bind(
      input.referralId,
      input.source,
      input.subjectRef,
      input.installationId,
      input.nowIso,
      input.nowIso,
    )
    .run();

  return Number(result.meta.changes ?? 0) === 1;
}

export async function getReservedReferralRewardIdForOperation(
  db: D1Database,
  operationId: string,
): Promise<number | null> {
  const row = await db
    .prepare(`SELECT id FROM referral_rewards WHERE operation_id = ? ORDER BY id DESC LIMIT 1`)
    .bind(operationId)
    .first<{ id: number }>()
    .catch(() => null);
  return row ? Number(row.id) : null;
}
