import type { BrokerBindings } from './contract';
import { markOperationSettlementStatus } from './managed-operation';
import {
  readManagedChildKeyEffectiveLimit,
  updateManagedChildKeyLimit,
} from './openrouter-management';
import type {
  ManagedReferralSettlementJobRecord,
  ReferralReferrerBonusStatus,
  ReferralReferredBonusStatus,
  ReferralSource,
} from './persistence';
import { MANAGED_TRIAL_BUDGET_POLICY } from './trial-policy';

const REFERRER_REWARD_USD = 0.02;
const MIN_INVITEE_LIMIT_USD = 0.09;
const SETTLEMENT_LEASE_MS = 5 * 60_000;
const MAX_SETTLEMENT_JOBS_PER_RUN = 25;
const INITIAL_RETRY_DELAY_MS = 60_000;
const MAX_RETRY_DELAY_MS = 60 * 60_000;

type SettlementEnv = Pick<BrokerBindings, 'BROKER_DB' | 'OPENROUTER_MANAGEMENT_API_KEY'>;

type ClaimedJob = Omit<ManagedReferralSettlementJobRecord, 'fencing_token' | 'lease_expires_at'> & {
  fencing_token: string;
  lease_expires_at: string;
};

interface InviteeContext {
  source: ReferralSource;
  referredBonusStatus: ReferralReferredBonusStatus;
  referredManagedCredentialRef: string | null;
  referredSubjectRef: string;
  entitlementRef: string;
  entitlementManagedCredentialRef: string;
  entitlementBudgetUsd: number;
}

interface ReferrerRewardContext {
  referrerSource: ReferralSource;
  referrerSubjectRef: string;
  referrerInstallationId: string | null;
  referredBonusStatus: ReferralReferredBonusStatus;
  referrerBonusStatus: ReferralReferrerBonusStatus;
  referrerManagedCredentialRef: string | null;
}

interface ActiveReferrerOwner {
  source: ReferralSource;
  subjectRef: string;
  installationId: string | null;
  entitlementRef: string;
  managedCredentialRef: string;
  budgetUsd: number;
}

export interface ManagedReferralSettlementRunResult {
  repaired: number;
  claimed: number;
  advanced: number;
  completed: number;
  retried: number;
}

export async function processManagedReferralSettlementJobs(
  env: SettlementEnv,
  input: { now?: Date; limit?: number; fetchImpl?: typeof fetch } = {},
): Promise<ManagedReferralSettlementRunResult> {
  const now = input.now ?? new Date();
  if (Number.isNaN(now.getTime())) {
    throw new Error('now must be a valid Date');
  }
  const limit = Math.min(Math.max(Math.trunc(input.limit ?? MAX_SETTLEMENT_JOBS_PER_RUN), 0), MAX_SETTLEMENT_JOBS_PER_RUN);
  const repaired = await repairManagedReferralSettlementJobs(env.BROKER_DB, now.toISOString());
  let claimed = 0;
  let advanced = 0;
  let completed = 0;
  let retried = 0;
  for (let index = 0; index < limit; index += 1) {
    const job = await claimNextManagedReferralSettlementJob(env.BROKER_DB, now);
    if (!job) {
      break;
    }
    claimed += 1;
    try {
      const outcome =
        job.phase === 'invitee_pending'
          ? await processInviteePhase(env, job, now, input.fetchImpl)
          : await processReferrerPhase(env, job, now, input.fetchImpl);
      if (outcome === 'advanced') {
        advanced += 1;
      } else if (outcome === 'completed') {
        completed += 1;
      } else {
        retried += 1;
      }
    } catch (error) {
      if (await hasConvergedAfterFailure(env.BROKER_DB, job)) {
        if (job.phase === 'invitee_pending') {
          advanced += 1;
        } else {
          completed += 1;
        }
        continue;
      }
      await releaseManagedReferralSettlementJob(env.BROKER_DB, job, now, boundedErrorCode(error));
      retried += 1;
    }
  }
  return { repaired, claimed, advanced, completed, retried };
}

export async function ensureReferralSettlementJobsForDelivery(
  db: D1Database,
  input: { source: ReferralSource; deliveryId: string; now: Date },
): Promise<number> {
  const nowIso = input.now.toISOString();
  const result = await db
    .prepare(
      `INSERT INTO managed_referral_settlement_jobs (
          source, referral_reward_id, delivery_id, operation_id, phase,
          attempt_count, last_attempt_at, next_attempt_at,
          fencing_token, lease_expires_at, last_error_code,
          created_at, updated_at, completed_at
        )
        SELECT ?, reward.id, delivery.delivery_id, reward.operation_id,
               'invitee_pending', 0, NULL, ?, NULL, NULL, NULL, ?, ?, NULL
          FROM referral_rewards reward
          JOIN managed_key_deliveries delivery
            ON delivery.delivery_id = ?
           AND delivery.issue_source = ?
           AND delivery.subject_ref = reward.referred_subject_ref
           AND delivery.installation_id IS reward.referred_installation_id
           AND delivery.status = 'acknowledged'
         WHERE reward.referred_source = ?
           AND reward.referred_bonus_status = 'reserved'
           AND NOT EXISTS (
             SELECT 1 FROM managed_referral_settlement_jobs job
              WHERE job.referral_reward_id = reward.id
                AND job.delivery_id = delivery.delivery_id
           )
        ON CONFLICT(referral_reward_id) DO NOTHING`,
    )
    .bind(
      input.source,
      nowIso,
      nowIso,
      nowIso,
      input.deliveryId,
      input.source,
      input.source,
    )
    .run()
    .catch(() => null);
  return Number(result?.meta?.changes ?? 0);
}

export async function hasUnsettledReservedRewardWithoutJob(
  db: D1Database,
  input: { source: ReferralSource; subjectRef: string; installationId: string | null; deliveryId: string },
): Promise<boolean> {
  const row = await db
    .prepare(
      `SELECT COUNT(*) AS count
         FROM referral_rewards reward
        WHERE reward.referred_source = ?
          AND reward.referred_subject_ref = ?
          AND reward.referred_installation_id IS ?
          AND reward.referred_bonus_status = 'reserved'
          AND NOT EXISTS (
            SELECT 1 FROM managed_referral_settlement_jobs job
             WHERE job.referral_reward_id = reward.id
               AND job.delivery_id = ?
          )
          AND NOT EXISTS (
            SELECT 1 FROM managed_referral_settlement_jobs job
             WHERE job.referral_reward_id = reward.id
          )`,
    )
    .bind(input.source, input.subjectRef, input.installationId, input.deliveryId)
    .first<{ count: number }>()
    .catch(() => null);
  return Number(row?.count ?? 0) > 0;
}

export async function scheduleManagedReferralSettlement(
  db: D1Database,
  input: { source: ReferralSource; referralRewardId: number; deliveryId: string | null; operationId: string | null; now: Date },
): Promise<void> {
  const nowIso = input.now.toISOString();
  await db
    .prepare(
      `INSERT OR IGNORE INTO managed_referral_settlement_jobs (
          source, referral_reward_id, delivery_id, operation_id, phase,
          attempt_count, last_attempt_at, next_attempt_at,
          fencing_token, lease_expires_at, last_error_code,
          created_at, updated_at, completed_at
        ) VALUES (?, ?, ?, ?, 'invitee_pending', 0, NULL, ?, NULL, NULL, NULL, ?, ?, NULL)`,
    )
    .bind(input.source, input.referralRewardId, input.deliveryId, input.operationId, nowIso, nowIso, nowIso)
    .run();
  await markOperationSettlementStatus(db, input.operationId, { settlement: 'invitee_pending' }, input.now);
}

async function repairManagedReferralSettlementJobs(db: D1Database, nowIso: string): Promise<number> {
  const discord = await db
    .prepare(
      `INSERT OR IGNORE INTO managed_referral_settlement_jobs (
          source, referral_reward_id, delivery_id, operation_id, phase,
          attempt_count, last_attempt_at, next_attempt_at,
          fencing_token, lease_expires_at, last_error_code,
          created_at, updated_at, completed_at
        )
        SELECT 'discord', reward.id, delivery.delivery_id, reward.operation_id,
               CASE WHEN reward.referred_bonus_status = 'credited' THEN 'referrer_pending' ELSE 'invitee_pending' END,
               0, NULL, ?, NULL, NULL, NULL, ?, ?, NULL
          FROM referral_rewards reward
          JOIN openrouter_entitlements entitlement
            ON entitlement.discord_user_ref = reward.referred_subject_ref
           AND entitlement.status = 'active'
           AND entitlement.discord_issue_status = 'active'
           AND entitlement.managed_credential_ref IS NOT NULL
           AND length(trim(entitlement.managed_credential_ref)) > 0
           AND entitlement.discord_issue_delivered_at IS NOT NULL
          JOIN managed_key_deliveries delivery
            ON delivery.delivery_id = (
                 SELECT matching_delivery.delivery_id
                   FROM managed_key_deliveries matching_delivery
                  WHERE matching_delivery.issue_source = 'discord'
                    AND matching_delivery.subject_ref = reward.referred_subject_ref
                    AND matching_delivery.installation_id IS reward.referred_installation_id
                    AND matching_delivery.managed_credential_ref = entitlement.managed_credential_ref
                    AND matching_delivery.status = 'acknowledged'
                  ORDER BY matching_delivery.acknowledged_at DESC, matching_delivery.created_at DESC, matching_delivery.delivery_id DESC
                  LIMIT 1
               )
         WHERE reward.referred_source = 'discord'
           AND reward.referrer_source IS NOT NULL
           AND reward.referrer_subject_ref IS NOT NULL
           AND (reward.referred_bonus_status = 'reserved' OR (reward.referred_bonus_status = 'credited' AND reward.referrer_bonus_status IN ('pending', 'applying', 'failed')))`,
    )
    .bind(nowIso, nowIso, nowIso)
    .run();
  const qq = await db
    .prepare(
      `INSERT OR IGNORE INTO managed_referral_settlement_jobs (
          source, referral_reward_id, delivery_id, operation_id, phase,
          attempt_count, last_attempt_at, next_attempt_at,
          fencing_token, lease_expires_at, last_error_code,
          created_at, updated_at, completed_at
        )
        SELECT 'qq', reward.id, delivery.delivery_id, reward.operation_id,
               CASE WHEN reward.referred_bonus_status = 'credited' THEN 'referrer_pending' ELSE 'invitee_pending' END,
               0, NULL, ?, NULL, NULL, NULL, ?, ?, NULL
          FROM referral_rewards reward
          JOIN qq_managed_entitlements entitlement
            ON entitlement.qq_subject_ref = reward.referred_subject_ref
           AND entitlement.status = 'active'
           AND entitlement.managed_credential_ref IS NOT NULL
           AND length(trim(entitlement.managed_credential_ref)) > 0
           AND entitlement.delivered_at IS NOT NULL
          JOIN managed_key_deliveries delivery
            ON delivery.delivery_id = (
                 SELECT matching_delivery.delivery_id
                   FROM managed_key_deliveries matching_delivery
                  WHERE matching_delivery.issue_source = 'qq'
                    AND matching_delivery.subject_ref = reward.referred_subject_ref
                    AND matching_delivery.installation_id IS reward.referred_installation_id
                    AND matching_delivery.managed_credential_ref = entitlement.managed_credential_ref
                    AND matching_delivery.status = 'acknowledged'
                  ORDER BY matching_delivery.acknowledged_at DESC, matching_delivery.created_at DESC, matching_delivery.delivery_id DESC
                  LIMIT 1
               )
         WHERE reward.referred_source = 'qq'
           AND reward.referrer_source IS NOT NULL
           AND reward.referrer_subject_ref IS NOT NULL
           AND (reward.referred_bonus_status = 'reserved' OR (reward.referred_bonus_status = 'credited' AND reward.referrer_bonus_status IN ('pending', 'applying', 'failed')))`,
    )
    .bind(nowIso, nowIso, nowIso)
    .run();
  return Number(discord.meta.changes ?? 0) + Number(qq.meta.changes ?? 0);
}

async function claimNextManagedReferralSettlementJob(db: D1Database, now: Date): Promise<ClaimedJob | null> {
  const nowIso = now.toISOString();
  const fencingToken = `ph-settle-fence-v1_${randomBase64Url(24)}`;
  const leaseExpiresAt = new Date(now.getTime() + SETTLEMENT_LEASE_MS).toISOString();
  const candidate = await db
    .prepare(
      `SELECT id, source, referral_reward_id, delivery_id, operation_id, phase, attempt_count,
              last_attempt_at, next_attempt_at, fencing_token, lease_expires_at, last_error_code,
              created_at, updated_at, completed_at
         FROM managed_referral_settlement_jobs
        WHERE phase IN ('invitee_pending', 'referrer_pending')
          AND next_attempt_at <= ?
          AND (lease_expires_at IS NULL OR lease_expires_at <= ?)
        ORDER BY next_attempt_at ASC, id ASC
        LIMIT 1`,
    )
    .bind(nowIso, nowIso)
    .first<ManagedReferralSettlementJobRecord>();
  if (!candidate) {
    return null;
  }
  const claimed = await db
    .prepare(
      `UPDATE managed_referral_settlement_jobs
          SET fencing_token = ?, lease_expires_at = ?, attempt_count = attempt_count + 1, last_attempt_at = ?, updated_at = ?
        WHERE id = ?
          AND phase IN ('invitee_pending', 'referrer_pending')
          AND (fencing_token IS NULL OR lease_expires_at <= ?)`,
    )
    .bind(fencingToken, leaseExpiresAt, nowIso, nowIso, candidate.id, nowIso)
    .run();
  if (Number(claimed.meta.changes ?? 0) !== 1) {
    return null;
  }
  const row = await db
    .prepare(`SELECT * FROM managed_referral_settlement_jobs WHERE id = ? AND fencing_token = ?`)
    .bind(candidate.id, fencingToken)
    .first<ManagedReferralSettlementJobRecord>();
  if (!row || !row.fencing_token || !row.lease_expires_at) {
    return null;
  }
  return row as ClaimedJob;
}

async function processInviteePhase(env: SettlementEnv, job: ClaimedJob, now: Date, fetchImpl?: typeof fetch): Promise<'advanced' | 'retry'> {
  const context = await readInviteeContext(env.BROKER_DB, job);
  if (!context) {
    return retryClaim(env.BROKER_DB, job, now, 'invitee_context_unavailable');
  }
  if (context.referredBonusStatus === 'credited' && context.referredManagedCredentialRef === context.entitlementManagedCredentialRef) {
    return advanceToReferrer(env.BROKER_DB, job, now);
  }
  if (context.referredBonusStatus !== 'reserved') {
    return retryClaim(env.BROKER_DB, job, now, 'invitee_reward_state_ambiguous');
  }
  const providerLimitUsd = await readManagedChildKeyEffectiveLimit({
    managementApiKey: env.OPENROUTER_MANAGEMENT_API_KEY,
    keyHash: context.entitlementManagedCredentialRef,
    fetchImpl,
  });
  const targetLimitUsd = maxUsd(MIN_INVITEE_LIMIT_USD, context.entitlementBudgetUsd, providerLimitUsd);
  if (currencyCents(providerLimitUsd) < currencyCents(targetLimitUsd)) {
    if (!(await stillOwnsSettlementClaim(env.BROKER_DB, job))) {
      return 'retry';
    }
    await updateManagedChildKeyLimit({
      managementApiKey: env.OPENROUTER_MANAGEMENT_API_KEY,
      keyHash: context.entitlementManagedCredentialRef,
      limitUsd: targetLimitUsd,
      fetchImpl,
    });
  }
  const verifiedLimitUsd = await readManagedChildKeyEffectiveLimit({
    managementApiKey: env.OPENROUTER_MANAGEMENT_API_KEY,
    keyHash: context.entitlementManagedCredentialRef,
    fetchImpl,
  });
  if (currencyCents(verifiedLimitUsd) < currencyCents(targetLimitUsd)) {
    return retryClaim(env.BROKER_DB, job, now, 'invitee_limit_not_verified');
  }
  const persistedBudgetUsd = maxUsd(targetLimitUsd, verifiedLimitUsd);
  const nowIso = now.toISOString();
  const entitlementUpdate =
    context.source === 'discord'
      ? env.BROKER_DB.prepare(
          `UPDATE openrouter_entitlements
              SET budget_usd = CASE WHEN budget_usd < ? THEN ? ELSE budget_usd END
            WHERE discord_user_ref = ?
              AND status = 'active'
              AND discord_issue_status = 'active'
              AND managed_credential_ref = ?
              AND discord_issue_delivered_at IS NOT NULL
              AND EXISTS (SELECT 1 FROM managed_referral_settlement_jobs job WHERE job.id = ? AND job.fencing_token = ? AND job.phase = 'invitee_pending')`,
        ).bind(persistedBudgetUsd, persistedBudgetUsd, context.referredSubjectRef, context.entitlementManagedCredentialRef, job.id, job.fencing_token)
      : env.BROKER_DB.prepare(
          `UPDATE qq_managed_entitlements
              SET budget_usd = CASE WHEN budget_usd < ? THEN ? ELSE budget_usd END,
                  updated_at = ?
            WHERE qq_subject_ref = ?
              AND issue_ref = ?
              AND status = 'active'
              AND managed_credential_ref = ?
              AND delivered_at IS NOT NULL
              AND EXISTS (SELECT 1 FROM managed_referral_settlement_jobs job WHERE job.id = ? AND job.fencing_token = ? AND job.phase = 'invitee_pending')`,
        ).bind(persistedBudgetUsd, persistedBudgetUsd, nowIso, context.referredSubjectRef, context.entitlementRef, context.entitlementManagedCredentialRef, job.id, job.fencing_token);
  const results = await env.BROKER_DB.batch([
    entitlementUpdate,
    env.BROKER_DB.prepare(
      `UPDATE referral_rewards
          SET referred_bonus_status = 'credited',
              referred_managed_credential_ref = ?,
              failure_reason = NULL,
              updated_at = ?,
              credited_at = COALESCE(credited_at, ?)
        WHERE id = ?
          AND referred_subject_ref = ?
          AND (referred_bonus_status = 'reserved' OR (referred_bonus_status = 'credited' AND referred_managed_credential_ref = ?))
          AND EXISTS (SELECT 1 FROM managed_referral_settlement_jobs job WHERE job.id = ? AND job.fencing_token = ? AND job.phase = 'invitee_pending')
          AND (
            EXISTS (
              SELECT 1 FROM openrouter_entitlements entitlement
               WHERE entitlement.discord_user_ref = ?
                 AND entitlement.status = 'active'
                 AND entitlement.discord_issue_status = 'active'
                 AND entitlement.managed_credential_ref = ?
                 AND entitlement.discord_issue_delivered_at IS NOT NULL
            )
            OR EXISTS (
              SELECT 1 FROM qq_managed_entitlements entitlement
               WHERE entitlement.qq_subject_ref = ?
                 AND entitlement.status = 'active'
                 AND entitlement.managed_credential_ref = ?
                 AND entitlement.delivered_at IS NOT NULL
            )
          )`,
    ).bind(context.entitlementManagedCredentialRef, nowIso, nowIso, job.referral_reward_id, context.referredSubjectRef, context.entitlementManagedCredentialRef, job.id, job.fencing_token, context.referredSubjectRef, context.entitlementManagedCredentialRef, context.referredSubjectRef, context.entitlementManagedCredentialRef),
    env.BROKER_DB.prepare(
      `UPDATE managed_referral_settlement_jobs
          SET phase = 'referrer_pending', next_attempt_at = ?, fencing_token = NULL, lease_expires_at = NULL, last_error_code = NULL, updated_at = ?
        WHERE id = ? AND fencing_token = ? AND phase = 'invitee_pending'
          AND EXISTS (SELECT 1 FROM referral_rewards reward WHERE reward.id = managed_referral_settlement_jobs.referral_reward_id AND reward.referred_bonus_status = 'credited' AND reward.referred_managed_credential_ref = ?)
          AND (
            EXISTS (
              SELECT 1 FROM openrouter_entitlements entitlement
               WHERE entitlement.discord_user_ref = ?
                 AND entitlement.status = 'active'
                 AND entitlement.discord_issue_status = 'active'
                 AND entitlement.managed_credential_ref = ?
                 AND entitlement.discord_issue_delivered_at IS NOT NULL
            )
            OR EXISTS (
              SELECT 1 FROM qq_managed_entitlements entitlement
               WHERE entitlement.qq_subject_ref = ?
                 AND entitlement.status = 'active'
                 AND entitlement.managed_credential_ref = ?
                 AND entitlement.delivered_at IS NOT NULL
            )
          )`,
    ).bind(nowIso, nowIso, job.id, job.fencing_token, context.entitlementManagedCredentialRef, context.referredSubjectRef, context.entitlementManagedCredentialRef, context.referredSubjectRef, context.entitlementManagedCredentialRef),
  ]);
  const ownerPersisted = Number(results[0]?.meta.changes ?? 0) === 1;
  if (Number(results[2]?.meta.changes ?? 0) === 1 && ownerPersisted) {
    logSettlementEvent('managed_referral_settlement_invitee_settled', job, { to_phase: 'referrer_pending' });
    await markOperationSettlementStatus(env.BROKER_DB, job.operation_id, { referral: 'credited', settlement: 'referrer_pending' }, now);
    return 'advanced';
  }
  if (ownerPersisted) {
    return retryClaim(env.BROKER_DB, job, now, 'invitee_commit_incomplete');
  }
  if (await hasInviteeConverged(env.BROKER_DB, job, persistedBudgetUsd, context.entitlementManagedCredentialRef)) {
    return 'advanced';
  }
  return retryClaim(env.BROKER_DB, job, now, 'invitee_commit_incomplete');
}

async function processReferrerPhase(env: SettlementEnv, job: ClaimedJob, now: Date, fetchImpl?: typeof fetch): Promise<'completed' | 'retry'> {
  const reward = await readReferrerRewardContext(env.BROKER_DB, job);
  if (!reward) {
    return retryClaim(env.BROKER_DB, job, now, 'referrer_context_unavailable');
  }
  if (reward.referrerBonusStatus === 'credited') {
    return completeSettledJob(env.BROKER_DB, job, now, 'credited');
  }
  if (reward.referrerBonusStatus === 'skipped') {
    return completeSettledJob(env.BROKER_DB, job, now, 'skipped');
  }
  if (reward.referredBonusStatus !== 'credited') {
    return retryClaim(env.BROKER_DB, job, now, 'referrer_invitee_not_credited');
  }
  const owner = await getActiveReferrerOwner(env.BROKER_DB, reward, now.toISOString());
  if (!owner) {
    return completeMissingReferrer(env.BROKER_DB, job, now);
  }
  const reflectedRewardCount = await countReferrerRewardsForTarget(env.BROKER_DB, job.referral_reward_id, owner);
  const ledgerTargetLimitUsd = Number((MANAGED_TRIAL_BUDGET_POLICY.hardLimit + reflectedRewardCount * REFERRER_REWARD_USD).toFixed(2));
  const providerLimitUsd = await readManagedChildKeyEffectiveLimit({
    managementApiKey: env.OPENROUTER_MANAGEMENT_API_KEY,
    keyHash: owner.managedCredentialRef,
    fetchImpl,
  });
  const targetLimitUsd = maxUsd(ledgerTargetLimitUsd, owner.budgetUsd, providerLimitUsd);
  if (currencyCents(providerLimitUsd) < currencyCents(targetLimitUsd)) {
    if (!(await stillOwnsSettlementClaim(env.BROKER_DB, job))) {
      return 'retry';
    }
    await updateManagedChildKeyLimit({
      managementApiKey: env.OPENROUTER_MANAGEMENT_API_KEY,
      keyHash: owner.managedCredentialRef,
      limitUsd: targetLimitUsd,
      fetchImpl,
    });
  }
  const verifiedLimitUsd = await readManagedChildKeyEffectiveLimit({
    managementApiKey: env.OPENROUTER_MANAGEMENT_API_KEY,
    keyHash: owner.managedCredentialRef,
    fetchImpl,
  });
  if (currencyCents(verifiedLimitUsd) < currencyCents(targetLimitUsd)) {
    return retryClaim(env.BROKER_DB, job, now, 'referrer_limit_not_verified');
  }
  const persistedBudgetUsd = maxUsd(targetLimitUsd, verifiedLimitUsd);
  const nowIso = now.toISOString();
  const ownerUpdate =
    owner.source === 'discord'
      ? env.BROKER_DB.prepare(
          `UPDATE openrouter_entitlements
              SET budget_usd = CASE WHEN budget_usd < ? THEN ? ELSE budget_usd END
            WHERE installation_id = ?
              AND discord_user_ref = ?
              AND status = 'active'
              AND discord_issue_status = 'active'
              AND managed_credential_ref = ?
              AND expires_at IS NOT NULL
              AND datetime(expires_at) >= datetime(?)
              AND EXISTS (SELECT 1 FROM managed_referral_settlement_jobs job WHERE job.id = ? AND job.fencing_token = ? AND job.phase = 'referrer_pending')`,
        ).bind(persistedBudgetUsd, persistedBudgetUsd, owner.installationId, owner.subjectRef, owner.managedCredentialRef, nowIso, job.id, job.fencing_token)
      : env.BROKER_DB.prepare(
          `UPDATE qq_managed_entitlements
              SET budget_usd = CASE WHEN budget_usd < ? THEN ? ELSE budget_usd END,
                  updated_at = ?
            WHERE qq_subject_ref = ?
              AND issue_ref = ?
              AND status = 'active'
              AND managed_credential_ref = ?
              AND delivered_at IS NOT NULL
              AND expires_at IS NOT NULL
              AND datetime(expires_at) >= datetime(?)
              AND EXISTS (SELECT 1 FROM managed_referral_settlement_jobs job WHERE job.id = ? AND job.fencing_token = ? AND job.phase = 'referrer_pending')`,
        ).bind(persistedBudgetUsd, persistedBudgetUsd, nowIso, owner.subjectRef, owner.entitlementRef, owner.managedCredentialRef, nowIso, job.id, job.fencing_token);
  const results = await env.BROKER_DB.batch([
    ownerUpdate,
    env.BROKER_DB.prepare(
      `UPDATE referral_rewards
          SET referrer_bonus_status = 'credited',
              referrer_managed_credential_ref = ?,
              skip_reason = NULL,
              failure_reason = NULL,
              updated_at = ?
        WHERE id = ?
          AND referred_bonus_status = 'credited'
          AND referrer_source = ?
          AND referrer_subject_ref = ?
          AND (referrer_bonus_status IN ('pending', 'applying', 'failed') OR (referrer_bonus_status = 'credited' AND referrer_managed_credential_ref = ?))
          AND EXISTS (SELECT 1 FROM managed_referral_settlement_jobs job WHERE job.id = ? AND job.fencing_token = ? AND job.phase = 'referrer_pending')
          AND (
            EXISTS (
              SELECT 1 FROM openrouter_entitlements entitlement
               WHERE entitlement.installation_id = ?
                 AND entitlement.discord_user_ref = ?
                 AND entitlement.status = 'active'
                 AND entitlement.discord_issue_status = 'active'
                 AND entitlement.managed_credential_ref = ?
                 AND entitlement.expires_at IS NOT NULL
                 AND datetime(entitlement.expires_at) >= datetime(?)
            )
            OR EXISTS (
              SELECT 1 FROM qq_managed_entitlements entitlement
               WHERE entitlement.qq_subject_ref = ?
                 AND entitlement.issue_ref = ?
                 AND entitlement.status = 'active'
                 AND entitlement.managed_credential_ref = ?
                 AND entitlement.delivered_at IS NOT NULL
                 AND entitlement.expires_at IS NOT NULL
                 AND datetime(entitlement.expires_at) >= datetime(?)
            )
          )`,
    ).bind(owner.managedCredentialRef, nowIso, job.referral_reward_id, owner.source, owner.subjectRef, owner.managedCredentialRef, job.id, job.fencing_token, owner.installationId, owner.subjectRef, owner.managedCredentialRef, nowIso, owner.subjectRef, owner.entitlementRef, owner.managedCredentialRef, nowIso),
    env.BROKER_DB.prepare(
      `UPDATE managed_referral_settlement_jobs
          SET phase = 'completed', next_attempt_at = ?, fencing_token = NULL, lease_expires_at = NULL, last_error_code = NULL, updated_at = ?, completed_at = ?
        WHERE id = ? AND fencing_token = ? AND phase = 'referrer_pending'
          AND EXISTS (SELECT 1 FROM referral_rewards reward WHERE reward.id = managed_referral_settlement_jobs.referral_reward_id AND reward.referrer_bonus_status = 'credited' AND reward.referrer_managed_credential_ref = ?)
          AND (
            EXISTS (
              SELECT 1 FROM openrouter_entitlements entitlement
               WHERE entitlement.installation_id = ?
                 AND entitlement.discord_user_ref = ?
                 AND entitlement.status = 'active'
                 AND entitlement.discord_issue_status = 'active'
                 AND entitlement.managed_credential_ref = ?
                 AND entitlement.expires_at IS NOT NULL
                 AND datetime(entitlement.expires_at) >= datetime(?)
            )
            OR EXISTS (
              SELECT 1 FROM qq_managed_entitlements entitlement
               WHERE entitlement.qq_subject_ref = ?
                 AND entitlement.issue_ref = ?
                 AND entitlement.status = 'active'
                 AND entitlement.managed_credential_ref = ?
                 AND entitlement.delivered_at IS NOT NULL
                 AND entitlement.expires_at IS NOT NULL
                 AND datetime(entitlement.expires_at) >= datetime(?)
            )
          )`,
    ).bind(nowIso, nowIso, nowIso, job.id, job.fencing_token, owner.managedCredentialRef, owner.installationId, owner.subjectRef, owner.managedCredentialRef, nowIso, owner.subjectRef, owner.entitlementRef, owner.managedCredentialRef, nowIso),
  ]);
  const ownerPersisted = Number(results[0]?.meta.changes ?? 0) === 1;
  if (Number(results[2]?.meta.changes ?? 0) === 1 && ownerPersisted) {
    logSettlementEvent('managed_referral_settlement_completed', job);
    await markOperationSettlementStatus(env.BROKER_DB, job.operation_id, { settlement: 'completed' }, now);
    return 'completed';
  }
  if (ownerPersisted) {
    return retryClaim(env.BROKER_DB, job, now, 'referrer_commit_incomplete');
  }
  if (await hasReferrerConverged(env.BROKER_DB, job, owner.managedCredentialRef, persistedBudgetUsd, owner)) {
    return 'completed';
  }
  return retryClaim(env.BROKER_DB, job, now, 'referrer_commit_incomplete');
}

async function readInviteeContext(db: D1Database, job: ClaimedJob): Promise<InviteeContext | null> {
  const reward = await db
    .prepare(`SELECT referred_source, referred_subject_ref, referred_bonus_status, referred_managed_credential_ref FROM referral_rewards WHERE id = ?`)
    .bind(job.referral_reward_id)
    .first<{ referred_source: string; referred_subject_ref: string; referred_bonus_status: ReferralReferredBonusStatus; referred_managed_credential_ref: string | null }>();
  if (!reward) {
    return null;
  }
  if (reward.referred_source === 'discord') {
    const row = await db
      .prepare(
        `SELECT entitlement.managed_credential_ref, entitlement.budget_usd, entitlement.installation_id AS issue_ref
           FROM openrouter_entitlements entitlement
          WHERE entitlement.discord_user_ref = ?
            AND entitlement.status = 'active'
            AND entitlement.discord_issue_status = 'active'
            AND entitlement.managed_credential_ref IS NOT NULL
            AND entitlement.discord_issue_delivered_at IS NOT NULL`,
      )
      .bind(reward.referred_subject_ref)
      .first<{ managed_credential_ref: string; budget_usd: number; issue_ref: string }>();
    if (!row) {
      return null;
    }
    return {
      source: 'discord',
      referredBonusStatus: reward.referred_bonus_status,
      referredManagedCredentialRef: reward.referred_managed_credential_ref,
      referredSubjectRef: reward.referred_subject_ref,
      entitlementRef: row.issue_ref,
      entitlementManagedCredentialRef: row.managed_credential_ref,
      entitlementBudgetUsd: row.budget_usd,
    };
  }
  const row = await db
    .prepare(
      `SELECT entitlement.issue_ref, entitlement.managed_credential_ref, entitlement.budget_usd
         FROM qq_managed_entitlements entitlement
         JOIN referral_rewards reward ON reward.id = ?
        WHERE entitlement.qq_subject_ref = reward.referred_subject_ref
          AND entitlement.status = 'active'
          AND entitlement.managed_credential_ref IS NOT NULL
          AND entitlement.delivered_at IS NOT NULL
          AND EXISTS (SELECT 1 FROM managed_referral_settlement_jobs job WHERE job.id = ? AND job.fencing_token = ?)`,
    )
    .bind(job.referral_reward_id, job.id, job.fencing_token)
    .first<{ issue_ref: string; managed_credential_ref: string; budget_usd: number }>();
  if (!row) {
    return null;
  }
  return {
    source: 'qq',
    referredBonusStatus: reward.referred_bonus_status,
    referredManagedCredentialRef: reward.referred_managed_credential_ref,
    referredSubjectRef: reward.referred_subject_ref,
    entitlementRef: row.issue_ref,
    entitlementManagedCredentialRef: row.managed_credential_ref,
    entitlementBudgetUsd: row.budget_usd,
  };
}

async function readReferrerRewardContext(db: D1Database, job: ClaimedJob): Promise<ReferrerRewardContext | null> {
  const row = await db
    .prepare(
      `SELECT reward.referrer_source, reward.referrer_subject_ref, reward.referrer_installation_id,
              reward.referred_bonus_status, reward.referrer_bonus_status, reward.referrer_managed_credential_ref
         FROM referral_rewards reward
        WHERE reward.id = ?
          AND EXISTS (SELECT 1 FROM managed_referral_settlement_jobs job WHERE job.id = ? AND job.fencing_token = ?)`,
    )
    .bind(job.referral_reward_id, job.id, job.fencing_token)
    .first<{
      referrer_source: ReferralSource;
      referrer_subject_ref: string;
      referrer_installation_id: string | null;
      referred_bonus_status: ReferralReferredBonusStatus;
      referrer_bonus_status: ReferralReferrerBonusStatus;
      referrer_managed_credential_ref: string | null;
    }>();
  if (!row || !row.referrer_source || !row.referrer_subject_ref) {
    return null;
  }
  return {
    referrerSource: row.referrer_source,
    referrerSubjectRef: row.referrer_subject_ref,
    referrerInstallationId: row.referrer_installation_id,
    referredBonusStatus: row.referred_bonus_status,
    referrerBonusStatus: row.referrer_bonus_status,
    referrerManagedCredentialRef: row.referrer_managed_credential_ref,
  };
}

async function getActiveReferrerOwner(db: D1Database, reward: ReferrerRewardContext, nowIso: string): Promise<ActiveReferrerOwner | null> {
  if (reward.referrerSource === 'discord') {
    const row = await db
      .prepare(
        `SELECT installation_id, discord_user_ref, managed_credential_ref, budget_usd, expires_at
           FROM openrouter_entitlements
          WHERE discord_user_ref = ?
            AND status = 'active'
            AND discord_issue_status = 'active'
            AND managed_credential_ref IS NOT NULL
            AND discord_issue_delivered_at IS NOT NULL
            AND expires_at IS NOT NULL`,
      )
      .bind(reward.referrerSubjectRef)
      .first<{ installation_id: string; discord_user_ref: string; managed_credential_ref: string; budget_usd: number; expires_at: string }>();
    if (!row || new Date(row.expires_at).getTime() < new Date(nowIso).getTime()) {
      return null;
    }
    return { source: 'discord', subjectRef: row.discord_user_ref, installationId: row.installation_id, entitlementRef: row.installation_id, managedCredentialRef: row.managed_credential_ref, budgetUsd: row.budget_usd };
  }
  const row = await db
    .prepare(
      `SELECT qq_subject_ref, issue_ref, managed_credential_ref, budget_usd, expires_at
         FROM qq_managed_entitlements
        WHERE qq_subject_ref = ?
          AND status = 'active'
          AND managed_credential_ref IS NOT NULL
          AND delivered_at IS NOT NULL
          AND expires_at IS NOT NULL`,
    )
    .bind(reward.referrerSubjectRef)
    .first<{ qq_subject_ref: string; issue_ref: string; managed_credential_ref: string; budget_usd: number; expires_at: string }>();
  if (!row || new Date(row.expires_at).getTime() < new Date(nowIso).getTime()) {
    return null;
  }
  return { source: 'qq', subjectRef: row.qq_subject_ref, installationId: null, entitlementRef: row.issue_ref, managedCredentialRef: row.managed_credential_ref, budgetUsd: row.budget_usd };
}

async function countReferrerRewardsForTarget(db: D1Database, currentRewardId: number, owner: ActiveReferrerOwner): Promise<number> {
  const row = await db
    .prepare(
      `SELECT COUNT(*) AS count FROM referral_rewards
        WHERE id <> ?
          AND referrer_source = ?
          AND referrer_subject_ref = ?
          AND referred_bonus_status = 'credited'
          AND referrer_bonus_status IN ('pending', 'applying', 'credited', 'failed')`,
    )
    .bind(currentRewardId, owner.source, owner.subjectRef)
    .first<{ count: number }>();
  return Number(row?.count ?? 0) + 1;
}

function logSettlementEvent(
  event: string,
  job: ClaimedJob,
  extra: Record<string, string | number | null> = {},
): void {
  (event === 'managed_referral_settlement_released' ? console.warn : console.info)(event, {
    settlement_job_id: job.id,
    source: job.source,
    referral_reward_id: job.referral_reward_id,
    delivery_id: job.delivery_id,
    operation_id: job.operation_id,
    phase: job.phase,
    attempt_count: job.attempt_count,
    ...extra,
    broker_timestamp: new Date().toISOString(),
  });
}

async function advanceToReferrer(db: D1Database, job: ClaimedJob, now: Date): Promise<'advanced' | 'retry'> {
  const nowIso = now.toISOString();
  const result = await db
    .prepare(
      `UPDATE managed_referral_settlement_jobs
          SET phase = 'referrer_pending', next_attempt_at = ?, fencing_token = NULL, lease_expires_at = NULL, last_error_code = NULL, updated_at = ?
        WHERE id = ? AND fencing_token = ? AND phase = 'invitee_pending'`,
    )
    .bind(nowIso, nowIso, job.id, job.fencing_token)
    .run();
  return Number(result.meta.changes ?? 0) === 1 ? 'advanced' : 'retry';
}

async function completeMissingReferrer(db: D1Database, job: ClaimedJob, now: Date): Promise<'completed' | 'retry'> {
  const nowIso = now.toISOString();
  const reward = await db
    .prepare(`SELECT referrer_source, referrer_subject_ref FROM referral_rewards WHERE id = ?`)
    .bind(job.referral_reward_id)
    .first<{ referrer_source: string | null; referrer_subject_ref: string | null }>();
  const skipReason = (await hasReferrerEntitlementWithoutCredential(db, reward)) ? 'referrer_managed_key_missing' : 'referrer_not_eligible';
  const result = await db
    .prepare(
      `UPDATE referral_rewards
          SET referrer_bonus_status = 'skipped', skip_reason = COALESCE(skip_reason, ?), updated_at = ?
        WHERE id = ? AND referred_bonus_status = 'credited' AND referrer_bonus_status IN ('pending', 'applying', 'failed')`,
    )
    .bind(skipReason, nowIso, job.referral_reward_id)
    .run();
  if (Number(result.meta.changes ?? 0) === 0) {
    const current = await db.prepare(`SELECT referrer_bonus_status FROM referral_rewards WHERE id = ?`).bind(job.referral_reward_id).first<{ referrer_bonus_status: string }>();
    if (current?.referrer_bonus_status !== 'skipped' && current?.referrer_bonus_status !== 'credited') {
      return retryClaim(db, job, now, 'referrer_skip_incomplete');
    }
  }
  return completeSettledJob(db, job, now, 'skipped');
}

async function hasReferrerEntitlementWithoutCredential(
  db: D1Database,
  reward: { referrer_source: string | null; referrer_subject_ref: string | null } | null,
): Promise<boolean> {
  if (!reward?.referrer_source || !reward.referrer_subject_ref) {
    return false;
  }
  if (reward.referrer_source === 'discord') {
    const row = await db
      .prepare(`SELECT managed_credential_ref FROM openrouter_entitlements WHERE discord_user_ref = ? AND managed_credential_ref IS NOT NULL AND length(trim(managed_credential_ref)) > 0 LIMIT 1`)
      .bind(reward.referrer_subject_ref)
      .first<{ managed_credential_ref: string }>();
    return row === null;
  }
  const row = await db
    .prepare(`SELECT managed_credential_ref FROM qq_managed_entitlements WHERE qq_subject_ref = ? AND managed_credential_ref IS NOT NULL AND length(trim(managed_credential_ref)) > 0 LIMIT 1`)
    .bind(reward.referrer_subject_ref)
    .first<{ managed_credential_ref: string }>();
  return row === null;
}

async function completeSettledJob(db: D1Database, job: ClaimedJob, now: Date, status: 'credited' | 'skipped'): Promise<'completed' | 'retry'> {
  const nowIso = now.toISOString();
  const result = await db
    .prepare(
      `UPDATE managed_referral_settlement_jobs
          SET phase = 'completed', next_attempt_at = ?, fencing_token = NULL, lease_expires_at = NULL, last_error_code = NULL, updated_at = ?, completed_at = ?
        WHERE id = ? AND fencing_token = ?`,
    )
    .bind(nowIso, nowIso, nowIso, job.id, job.fencing_token)
    .run();
  void status;
  return Number(result.meta.changes ?? 0) === 1 ? 'completed' : 'retry';
}

async function stillOwnsSettlementClaim(db: D1Database, job: ClaimedJob): Promise<boolean> {
  const row = await db.prepare(`SELECT id FROM managed_referral_settlement_jobs WHERE id = ? AND fencing_token = ?`).bind(job.id, job.fencing_token).first<{ id: number }>();
  return row !== null;
}

async function retryClaim(db: D1Database, job: ClaimedJob, now: Date, errorCode: string): Promise<'retry'> {
  await releaseManagedReferralSettlementJob(db, job, now, errorCode);
  return 'retry';
}

async function releaseManagedReferralSettlementJob(db: D1Database, job: ClaimedJob, now: Date, errorCode: string): Promise<void> {
  logSettlementEvent('managed_referral_settlement_released', job, { error_code: errorCode.slice(0, 64) });
  const delayMs = Math.min(MAX_RETRY_DELAY_MS, INITIAL_RETRY_DELAY_MS * 2 ** Math.min(job.attempt_count, 6));
  const nextAttemptAt = new Date(now.getTime() + delayMs).toISOString();
  await db
    .prepare(
      `UPDATE managed_referral_settlement_jobs
          SET fencing_token = NULL, lease_expires_at = NULL, next_attempt_at = ?, last_error_code = ?, updated_at = ?
        WHERE id = ? AND fencing_token = ?`,
    )
    .bind(nextAttemptAt, errorCode.slice(0, 64), now.toISOString(), job.id, job.fencing_token)
    .run();
}

async function hasConvergedAfterFailure(db: D1Database, job: ClaimedJob): Promise<boolean> {
  const reward = await db.prepare(`SELECT referred_bonus_status, referred_managed_credential_ref, referrer_bonus_status, referrer_managed_credential_ref FROM referral_rewards WHERE id = ?`).bind(job.referral_reward_id).first<{
    referred_bonus_status: string; referred_managed_credential_ref: string | null; referrer_bonus_status: string; referrer_managed_credential_ref: string | null;
  }>();
  if (!reward) {
    return false;
  }
  if (job.phase === 'invitee_pending') {
    if (reward.referred_bonus_status !== 'credited' || !reward.referred_managed_credential_ref) {
      return false;
    }
    return hasInviteeConverged(db, job, 0, reward.referred_managed_credential_ref);
  }
  if (reward.referrer_bonus_status !== 'credited' || !reward.referrer_managed_credential_ref) {
    return false;
  }
  const owner = await db.prepare(`SELECT 1 AS ok`).bind().first<{ ok: number }>();
  void owner;
  return true;
}

async function hasInviteeConverged(db: D1Database, job: ClaimedJob, budgetUsd: number, managedCredentialRef: string): Promise<boolean> {
  const reward = await db.prepare(`SELECT referred_bonus_status, referred_managed_credential_ref FROM referral_rewards WHERE id = ?`).bind(job.referral_reward_id).first<{
    referred_bonus_status: string; referred_managed_credential_ref: string | null;
  }>();
  if (reward?.referred_bonus_status !== 'credited' || reward.referred_managed_credential_ref !== managedCredentialRef) {
    return false;
  }
  if (budgetUsd > 0) {
    const discord = await db.prepare(`SELECT budget_usd FROM openrouter_entitlements WHERE managed_credential_ref = ?`).bind(managedCredentialRef).first<{ budget_usd: number }>();
    const qq = await db.prepare(`SELECT budget_usd FROM qq_managed_entitlements WHERE managed_credential_ref = ?`).bind(managedCredentialRef).first<{ budget_usd: number }>();
    const local = discord?.budget_usd ?? qq?.budget_usd ?? null;
    if (local !== null && currencyCents(local) < currencyCents(budgetUsd)) {
      return false;
    }
  }
  const settled = await db.prepare(`SELECT phase FROM managed_referral_settlement_jobs WHERE id = ?`).bind(job.id).first<{ phase: string }>();
  if (settled && (settled.phase === 'referrer_pending' || settled.phase === 'completed')) {
    return true;
  }
  await db.prepare(`UPDATE managed_referral_settlement_jobs SET phase = 'referrer_pending', fencing_token = NULL, lease_expires_at = NULL, last_error_code = NULL, updated_at = ? WHERE id = ?`).bind(new Date().toISOString(), job.id).run();
  return true;
}

async function hasReferrerConverged(db: D1Database, job: ClaimedJob, managedCredentialRef: string, budgetUsd: number, owner: ActiveReferrerOwner): Promise<boolean> {
  const reward = await db.prepare(`SELECT referrer_bonus_status, referrer_managed_credential_ref FROM referral_rewards WHERE id = ?`).bind(job.referral_reward_id).first<{
    referrer_bonus_status: string; referrer_managed_credential_ref: string | null;
  }>();
  if (reward?.referrer_bonus_status !== 'credited' || reward.referrer_managed_credential_ref !== managedCredentialRef) {
    return false;
  }
  const liveOwnerBudget =
    owner.source === 'discord'
      ? (
          await db
            .prepare(
              `SELECT budget_usd FROM openrouter_entitlements
                WHERE installation_id = ?
                  AND discord_user_ref = ?
                  AND status = 'active'
                  AND discord_issue_status = 'active'
                  AND managed_credential_ref = ?
                  AND expires_at IS NOT NULL
                  AND datetime(expires_at) >= datetime('now')`,
            )
            .bind(owner.installationId, owner.subjectRef, owner.managedCredentialRef)
            .first<{ budget_usd: number }>()
            .catch(() => null)
        )?.budget_usd ?? null
      : (
          await db
            .prepare(
              `SELECT budget_usd FROM qq_managed_entitlements
                WHERE qq_subject_ref = ?
                  AND issue_ref = ?
                  AND status = 'active'
                  AND managed_credential_ref = ?
                  AND delivered_at IS NOT NULL
                  AND expires_at IS NOT NULL
                  AND datetime(expires_at) >= datetime('now')`,
            )
            .bind(owner.subjectRef, owner.entitlementRef, owner.managedCredentialRef)
            .first<{ budget_usd: number }>()
            .catch(() => null)
        )?.budget_usd ?? null;
  if (liveOwnerBudget === null || currencyCents(liveOwnerBudget) < currencyCents(budgetUsd)) {
    return false;
  }
  const settled = await db.prepare(`SELECT phase FROM managed_referral_settlement_jobs WHERE id = ?`).bind(job.id).first<{ phase: string }>();
  if (settled?.phase === 'completed') {
    return true;
  }
  await db.prepare(`UPDATE managed_referral_settlement_jobs SET phase = 'completed', fencing_token = NULL, lease_expires_at = NULL, last_error_code = NULL, updated_at = ?, completed_at = ? WHERE id = ?`).bind(new Date().toISOString(), new Date().toISOString(), job.id).run();
  return true;
}

function maxUsd(...values: number[]): number {
  return Number(Math.max(...values).toFixed(2));
}

function currencyCents(value: number): number {
  return Math.floor(value * 100 + 1e-9);
}

function boundedErrorCode(error: unknown): string {
  if (error instanceof Error && error.name) {
    return error.name.slice(0, 64);
  }
  return 'settlement_error';
}

function randomBase64Url(byteLength: number): string {
  const bytes = crypto.getRandomValues(new Uint8Array(byteLength));
  let binary = '';
  for (const byte of bytes) {
    binary += String.fromCharCode(byte);
  }
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/g, '');
}
