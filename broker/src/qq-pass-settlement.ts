import type { BrokerBindings } from './contract';
import {
  OpenRouterManagementError,
  readManagedChildKeyEffectiveLimit,
  updateManagedChildKeyLimit,
} from './openrouter-management';
import type {
  QqPassSettlementJobRecord,
  ReferralReferrerBonusStatus,
  ReferralReferredBonusStatus,
  ReferralSource,
} from './persistence';
import { MANAGED_TRIAL_BUDGET_POLICY } from './trial-policy';

const QQ_REWARD_USD = 0.02;
const MIN_INVITEE_LIMIT_USD = 0.09;
const SETTLEMENT_LEASE_MS = 5 * 60_000;
const MAX_SETTLEMENT_JOBS_PER_RUN = 25;
const INITIAL_RETRY_DELAY_MS = 60_000;
const MAX_RETRY_DELAY_MS = 60 * 60_000;

type SettlementEnv = Pick<
  BrokerBindings,
  'BROKER_DB' | 'OPENROUTER_MANAGEMENT_API_KEY'
>;

type ClaimedJob = Omit<
  QqPassSettlementJobRecord,
  'fencing_token' | 'lease_expires_at'
> & {
  fencing_token: string;
  lease_expires_at: string;
};

interface InviteeContext {
  referredBonusStatus: ReferralReferredBonusStatus;
  referredManagedCredentialRef: string | null;
  referredSubjectRef: string;
  entitlementIssueRef: string;
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

export interface QqPassSettlementRunResult {
  repaired: number;
  claimed: number;
  advanced: number;
  completed: number;
  retried: number;
}

export async function processQqPassSettlementJobs(
  env: SettlementEnv,
  input: {
    now?: Date;
    limit?: number;
    fetchImpl?: typeof fetch;
  } = {},
): Promise<QqPassSettlementRunResult> {
  const now = input.now ?? new Date();
  if (Number.isNaN(now.getTime())) {
    throw new Error('now must be a valid Date');
  }
  const limit = Math.min(
    Math.max(Math.trunc(input.limit ?? MAX_SETTLEMENT_JOBS_PER_RUN), 0),
    MAX_SETTLEMENT_JOBS_PER_RUN,
  );
  const repaired = await repairQqPassSettlementJobs(
    env.BROKER_DB,
    now.toISOString(),
  );
  let claimed = 0;
  let advanced = 0;
  let completed = 0;
  let retried = 0;

  for (let index = 0; index < limit; index += 1) {
    const job = await claimNextQqPassSettlementJob(env.BROKER_DB, now);
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
      await releaseQqPassSettlementJob(
        env.BROKER_DB,
        job,
        now,
        boundedErrorCode(error),
      );
      retried += 1;
    }
  }

  return { repaired, claimed, advanced, completed, retried };
}

async function repairQqPassSettlementJobs(
  db: D1Database,
  nowIso: string,
): Promise<number> {
  const result = await db
    .prepare(
      `INSERT OR IGNORE INTO qq_pass_settlement_jobs (
          referral_reward_id,
          delivery_id,
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
        SELECT reward.id,
               delivery.delivery_id,
               CASE
                 WHEN reward.referred_bonus_status = 'credited'
                   THEN 'referrer_pending'
                 ELSE 'invitee_pending'
               END,
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
                  ORDER BY matching_delivery.acknowledged_at DESC,
                           matching_delivery.created_at DESC,
                           matching_delivery.delivery_id DESC
                  LIMIT 1
               )
         WHERE reward.referred_source = 'qq'
           AND reward.referrer_source IS NOT NULL
           AND reward.referrer_subject_ref IS NOT NULL
           AND (
             reward.referred_bonus_status = 'reserved'
             OR (
               reward.referred_bonus_status = 'credited'
               AND reward.referrer_bonus_status IN ('pending', 'applying', 'failed')
             )
           )`,
    )
    .bind(nowIso, nowIso, nowIso)
    .run();
  return Number(result.meta.changes ?? 0);
}

async function claimNextQqPassSettlementJob(
  db: D1Database,
  now: Date,
): Promise<ClaimedJob | null> {
  const nowIso = now.toISOString();
  const fencingToken = crypto.randomUUID();
  const leaseExpiresAt = new Date(
    now.getTime() + SETTLEMENT_LEASE_MS,
  ).toISOString();
  return db
    .prepare(
      `UPDATE qq_pass_settlement_jobs
          SET fencing_token = ?,
              lease_expires_at = ?,
              last_attempt_at = ?,
              attempt_count = attempt_count + 1,
              updated_at = ?
        WHERE id = (
          SELECT candidate.id
            FROM qq_pass_settlement_jobs candidate
           WHERE candidate.phase IN ('invitee_pending', 'referrer_pending')
             AND candidate.next_attempt_at <= ?
             AND (
               candidate.lease_expires_at IS NULL
               OR candidate.lease_expires_at <= ?
             )
           ORDER BY candidate.next_attempt_at ASC, candidate.id ASC
           LIMIT 1
        )
        RETURNING id,
                  referral_reward_id,
                  delivery_id,
                  phase,
                  attempt_count,
                  last_attempt_at,
                  next_attempt_at,
                  fencing_token,
                  lease_expires_at,
                  last_error_code,
                  created_at,
                  updated_at,
                  completed_at`,
    )
    .bind(
      fencingToken,
      leaseExpiresAt,
      nowIso,
      nowIso,
      nowIso,
      nowIso,
    )
    .first<ClaimedJob>();
}

async function processInviteePhase(
  env: SettlementEnv,
  job: ClaimedJob,
  now: Date,
  fetchImpl?: typeof fetch,
): Promise<'advanced' | 'retry'> {
  const context = await readInviteeContext(env.BROKER_DB, job);
  if (!context) {
    return retryClaim(
      env.BROKER_DB,
      job,
      now,
      'qq_invitee_context_unavailable',
    );
  }
  if (
    context.referredBonusStatus === 'credited' &&
    context.referredManagedCredentialRef ===
      context.entitlementManagedCredentialRef
  ) {
    return advanceToReferrer(env.BROKER_DB, job, now);
  }
  if (context.referredBonusStatus !== 'reserved') {
    return retryClaim(
      env.BROKER_DB,
      job,
      now,
      'qq_invitee_reward_state_ambiguous',
    );
  }

  const providerLimitUsd = await readManagedChildKeyEffectiveLimit({
    managementApiKey: env.OPENROUTER_MANAGEMENT_API_KEY,
    keyHash: context.entitlementManagedCredentialRef,
    fetchImpl,
  });
  const targetLimitUsd = maxUsd(
    MIN_INVITEE_LIMIT_USD,
    context.entitlementBudgetUsd,
    providerLimitUsd,
  );
  if (currencyCents(providerLimitUsd) < currencyCents(targetLimitUsd)) {
    if (!(await stillOwnsQqPassSettlementClaim(env.BROKER_DB, job))) {
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
    return retryClaim(
      env.BROKER_DB,
      job,
      now,
      'qq_invitee_limit_not_verified',
    );
  }

  const persistedBudgetUsd = maxUsd(targetLimitUsd, verifiedLimitUsd);
  const nowIso = now.toISOString();
  const results = await env.BROKER_DB.batch([
    env.BROKER_DB.prepare(
      `UPDATE qq_managed_entitlements
          SET budget_usd = CASE
                WHEN budget_usd < ? THEN ?
                ELSE budget_usd
              END,
              updated_at = ?
        WHERE qq_subject_ref = ?
          AND issue_ref = ?
          AND status = 'active'
          AND managed_credential_ref = ?
          AND delivered_at IS NOT NULL
          AND EXISTS (
            SELECT 1
              FROM qq_pass_settlement_jobs job
             WHERE job.id = ?
               AND job.fencing_token = ?
               AND job.phase = 'invitee_pending'
          )`,
    ).bind(
      persistedBudgetUsd,
      persistedBudgetUsd,
      nowIso,
      context.referredSubjectRef,
      context.entitlementIssueRef,
      context.entitlementManagedCredentialRef,
      job.id,
      job.fencing_token,
    ),
    env.BROKER_DB.prepare(
      `UPDATE referral_rewards
          SET referred_bonus_status = 'credited',
              referred_managed_credential_ref = ?,
              failure_reason = NULL,
              updated_at = ?,
              credited_at = COALESCE(credited_at, ?)
        WHERE id = ?
          AND referred_source = 'qq'
          AND referred_subject_ref = ?
          AND (
            referred_bonus_status = 'reserved'
            OR (
              referred_bonus_status = 'credited'
              AND referred_managed_credential_ref = ?
            )
          )
          AND EXISTS (
            SELECT 1
              FROM qq_pass_settlement_jobs job
             WHERE job.id = ?
               AND job.fencing_token = ?
               AND job.phase = 'invitee_pending'
          )`,
    ).bind(
      context.entitlementManagedCredentialRef,
      nowIso,
      nowIso,
      job.referral_reward_id,
      context.referredSubjectRef,
      context.entitlementManagedCredentialRef,
      job.id,
      job.fencing_token,
    ),
    env.BROKER_DB.prepare(
      `UPDATE qq_pass_settlement_jobs
          SET phase = 'referrer_pending',
              next_attempt_at = ?,
              fencing_token = NULL,
              lease_expires_at = NULL,
              last_error_code = NULL,
              updated_at = ?
        WHERE id = ?
          AND fencing_token = ?
          AND phase = 'invitee_pending'
          AND EXISTS (
            SELECT 1
              FROM referral_rewards reward
             WHERE reward.id = qq_pass_settlement_jobs.referral_reward_id
               AND reward.referred_bonus_status = 'credited'
               AND reward.referred_managed_credential_ref = ?
          )
          AND EXISTS (
            SELECT 1
              FROM qq_managed_entitlements entitlement
             WHERE entitlement.qq_subject_ref = ?
               AND entitlement.managed_credential_ref = ?
               AND entitlement.status = 'active'
               AND entitlement.delivered_at IS NOT NULL
               AND entitlement.budget_usd >= ?
          )`,
    ).bind(
      nowIso,
      nowIso,
      job.id,
      job.fencing_token,
      context.entitlementManagedCredentialRef,
      context.referredSubjectRef,
      context.entitlementManagedCredentialRef,
      persistedBudgetUsd,
    ),
  ]);
  if (Number(results[2]?.meta.changes ?? 0) === 1) {
    return 'advanced';
  }
  if (
    await hasInviteeConverged(
      env.BROKER_DB,
      job,
      persistedBudgetUsd,
      context.entitlementManagedCredentialRef,
    )
  ) {
    return 'advanced';
  }
  return retryClaim(env.BROKER_DB, job, now, 'qq_invitee_commit_incomplete');
}

async function processReferrerPhase(
  env: SettlementEnv,
  job: ClaimedJob,
  now: Date,
  fetchImpl?: typeof fetch,
): Promise<'completed' | 'retry'> {
  const reward = await readReferrerRewardContext(env.BROKER_DB, job);
  if (!reward) {
    return retryClaim(
      env.BROKER_DB,
      job,
      now,
      'qq_referrer_context_unavailable',
    );
  }
  if (reward.referrerBonusStatus === 'credited') {
    return completeSettledJob(env.BROKER_DB, job, now, 'credited');
  }
  if (reward.referrerBonusStatus === 'skipped') {
    return completeSettledJob(env.BROKER_DB, job, now, 'skipped');
  }
  if (reward.referredBonusStatus !== 'credited') {
    return retryClaim(
      env.BROKER_DB,
      job,
      now,
      'qq_referrer_invitee_not_credited',
    );
  }

  const owner = await getActiveReferrerOwner(
    env.BROKER_DB,
    reward,
    now.toISOString(),
  );
  if (!owner) {
    return completeMissingReferrer(env.BROKER_DB, job, now);
  }

  const reflectedRewardCount = await countReferrerRewardsForTarget(
    env.BROKER_DB,
    job.referral_reward_id,
    owner,
  );
  const ledgerTargetLimitUsd = Number(
    (
      MANAGED_TRIAL_BUDGET_POLICY.hardLimit +
      reflectedRewardCount * QQ_REWARD_USD
    ).toFixed(2),
  );
  const providerLimitUsd = await readManagedChildKeyEffectiveLimit({
    managementApiKey: env.OPENROUTER_MANAGEMENT_API_KEY,
    keyHash: owner.managedCredentialRef,
    fetchImpl,
  });
  const targetLimitUsd = maxUsd(
    ledgerTargetLimitUsd,
    owner.budgetUsd,
    providerLimitUsd,
  );
  if (currencyCents(providerLimitUsd) < currencyCents(targetLimitUsd)) {
    if (!(await stillOwnsQqPassSettlementClaim(env.BROKER_DB, job))) {
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
    return retryClaim(
      env.BROKER_DB,
      job,
      now,
      'qq_referrer_limit_not_verified',
    );
  }

  const persistedBudgetUsd = maxUsd(targetLimitUsd, verifiedLimitUsd);
  const nowIso = now.toISOString();
  const ownerUpdate =
    owner.source === 'discord'
      ? env.BROKER_DB.prepare(
          `UPDATE openrouter_entitlements
              SET budget_usd = CASE
                    WHEN budget_usd < ? THEN ?
                    ELSE budget_usd
                  END
            WHERE installation_id = ?
              AND discord_user_ref = ?
              AND status = 'active'
              AND discord_issue_status = 'active'
              AND managed_credential_ref = ?
              AND expires_at IS NOT NULL
              AND datetime(expires_at) >= datetime(?)
              AND EXISTS (
                SELECT 1
                  FROM qq_pass_settlement_jobs job
                 WHERE job.id = ?
                   AND job.fencing_token = ?
                   AND job.phase = 'referrer_pending'
              )`,
        ).bind(
          persistedBudgetUsd,
          persistedBudgetUsd,
          owner.installationId,
          owner.subjectRef,
          owner.managedCredentialRef,
          nowIso,
          job.id,
          job.fencing_token,
        )
      : env.BROKER_DB.prepare(
          `UPDATE qq_managed_entitlements
              SET budget_usd = CASE
                    WHEN budget_usd < ? THEN ?
                    ELSE budget_usd
                  END,
                  updated_at = ?
            WHERE qq_subject_ref = ?
              AND issue_ref = ?
              AND status = 'active'
              AND managed_credential_ref = ?
              AND delivered_at IS NOT NULL
              AND expires_at IS NOT NULL
              AND datetime(expires_at) >= datetime(?)
              AND EXISTS (
                SELECT 1
                  FROM qq_pass_settlement_jobs job
                 WHERE job.id = ?
                   AND job.fencing_token = ?
                   AND job.phase = 'referrer_pending'
              )`,
        ).bind(
          persistedBudgetUsd,
          persistedBudgetUsd,
          nowIso,
          owner.subjectRef,
          owner.entitlementRef,
          owner.managedCredentialRef,
          nowIso,
          job.id,
          job.fencing_token,
        );
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
          AND (
            referrer_bonus_status IN ('pending', 'applying', 'failed')
            OR (
              referrer_bonus_status = 'credited'
              AND referrer_managed_credential_ref = ?
            )
          )
          AND EXISTS (
            SELECT 1
              FROM qq_pass_settlement_jobs job
             WHERE job.id = ?
               AND job.fencing_token = ?
               AND job.phase = 'referrer_pending'
          )`,
    ).bind(
      owner.managedCredentialRef,
      nowIso,
      job.referral_reward_id,
      owner.source,
      owner.subjectRef,
      owner.managedCredentialRef,
      job.id,
      job.fencing_token,
    ),
    env.BROKER_DB.prepare(
      `UPDATE qq_pass_settlement_jobs
          SET phase = 'completed',
              next_attempt_at = ?,
              fencing_token = NULL,
              lease_expires_at = NULL,
              last_error_code = NULL,
              updated_at = ?,
              completed_at = ?
        WHERE id = ?
          AND fencing_token = ?
          AND phase = 'referrer_pending'
          AND EXISTS (
            SELECT 1
              FROM referral_rewards reward
             WHERE reward.id = qq_pass_settlement_jobs.referral_reward_id
               AND reward.referrer_bonus_status = 'credited'
               AND reward.referrer_managed_credential_ref = ?
          )
          AND EXISTS (
            SELECT 1
              FROM ${owner.source === 'discord' ? 'openrouter_entitlements' : 'qq_managed_entitlements'} entitlement
             WHERE ${owner.source === 'discord' ? 'entitlement.installation_id' : 'entitlement.qq_subject_ref'} = ?
               AND entitlement.managed_credential_ref = ?
               AND entitlement.budget_usd >= ?
          )`,
    ).bind(
      nowIso,
      nowIso,
      nowIso,
      job.id,
      job.fencing_token,
      owner.managedCredentialRef,
      owner.source === 'discord' ? owner.installationId : owner.subjectRef,
      owner.managedCredentialRef,
      persistedBudgetUsd,
    ),
  ]);
  if (Number(results[2]?.meta.changes ?? 0) === 1) {
    return 'completed';
  }
  if (
    await hasReferrerConverged(
      env.BROKER_DB,
      job,
      owner.managedCredentialRef,
      persistedBudgetUsd,
      owner,
    )
  ) {
    return 'completed';
  }
  return retryClaim(env.BROKER_DB, job, now, 'qq_referrer_commit_incomplete');
}

async function readInviteeContext(
  db: D1Database,
  job: ClaimedJob,
): Promise<InviteeContext | null> {
  const row = await db
    .prepare(
      `SELECT reward.referred_bonus_status,
              reward.referred_managed_credential_ref,
              reward.referred_subject_ref,
              entitlement.issue_ref,
              entitlement.managed_credential_ref,
              entitlement.budget_usd
         FROM qq_pass_settlement_jobs job
         JOIN referral_rewards reward
           ON reward.id = job.referral_reward_id
         JOIN managed_key_deliveries delivery
           ON delivery.delivery_id = job.delivery_id
          AND delivery.issue_source = 'qq'
          AND delivery.subject_ref = reward.referred_subject_ref
          AND delivery.installation_id IS reward.referred_installation_id
          AND delivery.status = 'acknowledged'
         JOIN qq_managed_entitlements entitlement
           ON entitlement.qq_subject_ref = reward.referred_subject_ref
          AND entitlement.managed_credential_ref = delivery.managed_credential_ref
          AND entitlement.status = 'active'
          AND entitlement.delivered_at IS NOT NULL
        WHERE job.id = ?
          AND job.fencing_token = ?
          AND job.phase = 'invitee_pending'
          AND reward.referred_source = 'qq'`,
    )
    .bind(job.id, job.fencing_token)
    .first<{
      referred_bonus_status: ReferralReferredBonusStatus;
      referred_managed_credential_ref: string | null;
      referred_subject_ref: string;
      issue_ref: string;
      managed_credential_ref: string;
      budget_usd: number;
    }>();
  if (!row) {
    return null;
  }
  return {
    referredBonusStatus: row.referred_bonus_status,
    referredManagedCredentialRef: row.referred_managed_credential_ref,
    referredSubjectRef: row.referred_subject_ref,
    entitlementIssueRef: row.issue_ref,
    entitlementManagedCredentialRef: row.managed_credential_ref,
    entitlementBudgetUsd: Number(row.budget_usd),
  };
}

async function readReferrerRewardContext(
  db: D1Database,
  job: ClaimedJob,
): Promise<ReferrerRewardContext | null> {
  const row = await db
    .prepare(
      `SELECT reward.referrer_source,
              reward.referrer_subject_ref,
              reward.referrer_installation_id,
              reward.referred_bonus_status,
              reward.referrer_bonus_status,
              reward.referrer_managed_credential_ref
         FROM qq_pass_settlement_jobs job
         JOIN referral_rewards reward
           ON reward.id = job.referral_reward_id
        WHERE job.id = ?
          AND job.fencing_token = ?
          AND job.phase = 'referrer_pending'
          AND reward.referred_source = 'qq'
          AND reward.referrer_source IS NOT NULL
          AND reward.referrer_subject_ref IS NOT NULL`,
    )
    .bind(job.id, job.fencing_token)
    .first<{
      referrer_source: ReferralSource;
      referrer_subject_ref: string;
      referrer_installation_id: string | null;
      referred_bonus_status: ReferralReferredBonusStatus;
      referrer_bonus_status: ReferralReferrerBonusStatus;
      referrer_managed_credential_ref: string | null;
    }>();
  if (!row) {
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

async function getActiveReferrerOwner(
  db: D1Database,
  reward: ReferrerRewardContext,
  nowIso: string,
): Promise<ActiveReferrerOwner | null> {
  if (reward.referrerSource === 'discord') {
    const row = await db
      .prepare(
        `SELECT installation_id,
                discord_user_ref,
                managed_credential_ref,
                budget_usd
           FROM openrouter_entitlements
          WHERE installation_id IS ?
            AND discord_user_ref = ?
            AND status = 'active'
            AND discord_issue_status = 'active'
            AND managed_credential_ref IS NOT NULL
            AND length(trim(managed_credential_ref)) > 0
            AND expires_at IS NOT NULL
            AND datetime(expires_at) >= datetime(?)`,
      )
      .bind(
        reward.referrerInstallationId,
        reward.referrerSubjectRef,
        nowIso,
      )
      .first<{
        installation_id: string;
        discord_user_ref: string;
        managed_credential_ref: string;
        budget_usd: number;
      }>();
    return row
      ? {
          source: 'discord',
          subjectRef: row.discord_user_ref,
          installationId: row.installation_id,
          entitlementRef: row.installation_id,
          managedCredentialRef: row.managed_credential_ref,
          budgetUsd: Number(row.budget_usd),
        }
      : null;
  }

  const row = await db
    .prepare(
      `SELECT qq_subject_ref,
              issue_ref,
              managed_credential_ref,
              budget_usd
         FROM qq_managed_entitlements
        WHERE qq_subject_ref = ?
          AND status = 'active'
          AND managed_credential_ref IS NOT NULL
          AND length(trim(managed_credential_ref)) > 0
          AND delivered_at IS NOT NULL
          AND expires_at IS NOT NULL
          AND datetime(expires_at) >= datetime(?)`,
    )
    .bind(reward.referrerSubjectRef, nowIso)
    .first<{
      qq_subject_ref: string;
      issue_ref: string;
      managed_credential_ref: string;
      budget_usd: number;
    }>();
  return row
    ? {
        source: 'qq',
        subjectRef: row.qq_subject_ref,
        installationId: null,
        entitlementRef: row.issue_ref,
        managedCredentialRef: row.managed_credential_ref,
        budgetUsd: Number(row.budget_usd),
      }
    : null;
}

async function countReferrerRewardsForTarget(
  db: D1Database,
  currentRewardId: number,
  owner: ActiveReferrerOwner,
): Promise<number> {
  const row = await db
    .prepare(
      `SELECT COUNT(*) AS count
         FROM referral_rewards
        WHERE referrer_source = ?
          AND referrer_subject_ref = ?
          AND referred_bonus_status = 'credited'
          AND (
            referrer_bonus_status IN ('pending', 'applying', 'credited')
            OR id = ?
          )
          AND (
            referrer_managed_credential_ref IS NULL
            OR referrer_managed_credential_ref = ?
          )`,
    )
    .bind(
      owner.source,
      owner.subjectRef,
      currentRewardId,
      owner.managedCredentialRef,
    )
    .first<{ count: number }>();
  return Number(row?.count ?? 0);
}

async function advanceToReferrer(
  db: D1Database,
  job: ClaimedJob,
  now: Date,
): Promise<'advanced' | 'retry'> {
  const nowIso = now.toISOString();
  const result = await db
    .prepare(
      `UPDATE qq_pass_settlement_jobs
          SET phase = 'referrer_pending',
              next_attempt_at = ?,
              fencing_token = NULL,
              lease_expires_at = NULL,
              last_error_code = NULL,
              updated_at = ?
        WHERE id = ?
          AND fencing_token = ?
          AND phase = 'invitee_pending'
          AND EXISTS (
            SELECT 1
              FROM referral_rewards reward
             WHERE reward.id = qq_pass_settlement_jobs.referral_reward_id
               AND reward.referred_bonus_status = 'credited'
          )`,
    )
    .bind(nowIso, nowIso, job.id, job.fencing_token)
    .run();
  return Number(result.meta.changes ?? 0) === 1 ? 'advanced' : 'retry';
}

async function completeMissingReferrer(
  db: D1Database,
  job: ClaimedJob,
  now: Date,
): Promise<'completed' | 'retry'> {
  const nowIso = now.toISOString();
  const results = await db.batch([
    db.prepare(
      `UPDATE referral_rewards
          SET referrer_bonus_status = 'skipped',
              skip_reason = 'referrer_managed_key_missing',
              failure_reason = NULL,
              updated_at = ?
        WHERE id = ?
          AND referred_bonus_status = 'credited'
          AND referrer_bonus_status IN ('pending', 'applying', 'failed')
          AND EXISTS (
            SELECT 1
              FROM qq_pass_settlement_jobs job
             WHERE job.id = ?
               AND job.fencing_token = ?
               AND job.phase = 'referrer_pending'
          )`,
    ).bind(
      nowIso,
      job.referral_reward_id,
      job.id,
      job.fencing_token,
    ),
    db.prepare(
      `UPDATE qq_pass_settlement_jobs
          SET phase = 'completed',
              next_attempt_at = ?,
              fencing_token = NULL,
              lease_expires_at = NULL,
              last_error_code = NULL,
              updated_at = ?,
              completed_at = ?
        WHERE id = ?
          AND fencing_token = ?
          AND phase = 'referrer_pending'
          AND EXISTS (
            SELECT 1
              FROM referral_rewards reward
             WHERE reward.id = qq_pass_settlement_jobs.referral_reward_id
               AND reward.referrer_bonus_status = 'skipped'
          )`,
    ).bind(nowIso, nowIso, nowIso, job.id, job.fencing_token),
  ]);
  return Number(results[1]?.meta.changes ?? 0) === 1 ? 'completed' : 'retry';
}

async function completeSettledJob(
  db: D1Database,
  job: ClaimedJob,
  now: Date,
  status: 'credited' | 'skipped',
): Promise<'completed' | 'retry'> {
  const nowIso = now.toISOString();
  const result = await db
    .prepare(
      `UPDATE qq_pass_settlement_jobs
          SET phase = 'completed',
              next_attempt_at = ?,
              fencing_token = NULL,
              lease_expires_at = NULL,
              last_error_code = NULL,
              updated_at = ?,
              completed_at = ?
        WHERE id = ?
          AND fencing_token = ?
          AND phase = 'referrer_pending'
          AND EXISTS (
            SELECT 1
              FROM referral_rewards reward
             WHERE reward.id = qq_pass_settlement_jobs.referral_reward_id
               AND reward.referrer_bonus_status = ?
          )`,
    )
    .bind(
      nowIso,
      nowIso,
      nowIso,
      job.id,
      job.fencing_token,
      status,
    )
    .run();
  return Number(result.meta.changes ?? 0) === 1 ? 'completed' : 'retry';
}

async function stillOwnsQqPassSettlementClaim(
  db: D1Database,
  job: ClaimedJob,
): Promise<boolean> {
  const row = await db
    .prepare(
      `SELECT 1 AS owned
         FROM qq_pass_settlement_jobs
        WHERE id = ?
          AND fencing_token = ?
          AND phase = ?`,
    )
    .bind(job.id, job.fencing_token, job.phase)
    .first<{ owned: number }>();
  return Number(row?.owned ?? 0) === 1;
}

async function retryClaim(
  db: D1Database,
  job: ClaimedJob,
  now: Date,
  errorCode: string,
): Promise<'retry'> {
  await releaseQqPassSettlementJob(db, job, now, errorCode);
  return 'retry';
}

async function releaseQqPassSettlementJob(
  db: D1Database,
  job: ClaimedJob,
  now: Date,
  errorCode: string,
): Promise<void> {
  const exponent = Math.min(Math.max(job.attempt_count - 1, 0), 6);
  const delayMs = Math.min(
    INITIAL_RETRY_DELAY_MS * 2 ** exponent,
    MAX_RETRY_DELAY_MS,
  );
  await db
    .prepare(
      `UPDATE qq_pass_settlement_jobs
          SET next_attempt_at = ?,
              fencing_token = NULL,
              lease_expires_at = NULL,
              last_error_code = ?,
              updated_at = ?
        WHERE id = ?
          AND fencing_token = ?
          AND phase IN ('invitee_pending', 'referrer_pending')`,
    )
    .bind(
      new Date(now.getTime() + delayMs).toISOString(),
      errorCode,
      now.toISOString(),
      job.id,
      job.fencing_token,
    )
    .run();
}

async function hasConvergedAfterFailure(
  db: D1Database,
  job: ClaimedJob,
): Promise<boolean> {
  if (job.phase === 'invitee_pending') {
    const row = await db
      .prepare(
        `SELECT 1 AS converged
           FROM qq_pass_settlement_jobs job
           JOIN referral_rewards reward
             ON reward.id = job.referral_reward_id
          WHERE job.id = ?
            AND job.phase IN ('referrer_pending', 'completed')
            AND reward.referred_bonus_status = 'credited'`,
      )
      .bind(job.id)
      .first<{ converged: number }>();
    return Number(row?.converged ?? 0) === 1;
  }
  const row = await db
    .prepare(
      `SELECT 1 AS converged
         FROM qq_pass_settlement_jobs job
         JOIN referral_rewards reward
           ON reward.id = job.referral_reward_id
        WHERE job.id = ?
          AND job.phase = 'completed'
          AND reward.referrer_bonus_status IN ('credited', 'skipped')`,
    )
    .bind(job.id)
    .first<{ converged: number }>();
  return Number(row?.converged ?? 0) === 1;
}

async function hasInviteeConverged(
  db: D1Database,
  job: ClaimedJob,
  budgetUsd: number,
  managedCredentialRef: string,
): Promise<boolean> {
  const row = await db
    .prepare(
      `SELECT 1 AS converged
         FROM qq_pass_settlement_jobs job
         JOIN referral_rewards reward
           ON reward.id = job.referral_reward_id
         JOIN qq_managed_entitlements entitlement
           ON entitlement.qq_subject_ref = reward.referred_subject_ref
        WHERE job.id = ?
          AND job.phase IN ('referrer_pending', 'completed')
          AND reward.referred_bonus_status = 'credited'
          AND reward.referred_managed_credential_ref = ?
          AND entitlement.status = 'active'
          AND entitlement.managed_credential_ref = ?
          AND entitlement.budget_usd >= ?
          AND entitlement.delivered_at IS NOT NULL`,
    )
    .bind(job.id, managedCredentialRef, managedCredentialRef, budgetUsd)
    .first<{ converged: number }>();
  return Number(row?.converged ?? 0) === 1;
}

async function hasReferrerConverged(
  db: D1Database,
  job: ClaimedJob,
  managedCredentialRef: string,
  budgetUsd: number,
  owner: ActiveReferrerOwner,
): Promise<boolean> {
  const entitlementTable =
    owner.source === 'discord'
      ? 'openrouter_entitlements'
      : 'qq_managed_entitlements';
  const ownerColumn =
    owner.source === 'discord' ? 'installation_id' : 'qq_subject_ref';
  const ownerValue =
    owner.source === 'discord' ? owner.installationId : owner.subjectRef;
  const row = await db
    .prepare(
      `SELECT 1 AS converged
         FROM qq_pass_settlement_jobs job
         JOIN referral_rewards reward
           ON reward.id = job.referral_reward_id
         JOIN ${entitlementTable} entitlement
           ON entitlement.${ownerColumn} = ?
        WHERE job.id = ?
          AND job.phase = 'completed'
          AND reward.referrer_bonus_status = 'credited'
          AND reward.referrer_managed_credential_ref = ?
          AND entitlement.managed_credential_ref = ?
          AND entitlement.budget_usd >= ?`,
    )
    .bind(
      ownerValue,
      job.id,
      managedCredentialRef,
      managedCredentialRef,
      budgetUsd,
    )
    .first<{ converged: number }>();
  return Number(row?.converged ?? 0) === 1;
}

function maxUsd(...values: number[]): number {
  const finiteValues = values.filter(
    (value) => Number.isFinite(value) && value >= 0,
  );
  if (finiteValues.length !== values.length) {
    throw new Error('managed budget must be a finite non-negative USD value');
  }
  return Number(Math.max(...finiteValues).toFixed(2));
}

function currencyCents(value: number): number {
  if (!Number.isFinite(value) || value < 0) {
    throw new Error('managed budget must be a finite non-negative USD value');
  }
  return Math.round(value * 100);
}

function boundedErrorCode(error: unknown): string {
  if (error instanceof OpenRouterManagementError) {
    const status = error.status === null ? 'none' : String(error.status);
    return `provider_${error.code}_${status}`.slice(0, 64);
  }
  return 'transient_settlement_error';
}
