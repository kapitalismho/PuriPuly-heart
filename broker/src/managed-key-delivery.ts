import type { Context } from 'hono';

import {
  extractRequestNetworkMetadata,
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
import type { BrokerBindings, BrokerEnv } from './contract';
import {
  buildManagedCleanupRequiredAuditPayload,
  type ManagedIssueSource,
} from './managed-issuance';
import {
  cleanupManagedChildKey,
  updateManagedChildKeyLimit,
  type ManagedChildKeyCleanupResult,
} from './openrouter-management';
import type {
  ManagedKeyDeliveryRecord,
  OpenRouterEntitlementRecord,
  QqManagedEntitlementRecord,
  ReferralRewardRecord,
} from './persistence';
import { nonEmptyString } from './public-input';
import {
  applyCreditedIssueReferrerRewardLimitUpdate,
  ensureOwnedReferralIdForActiveDiscordManagedUser,
  markReservedIssueReferralCredited,
  markReservedIssueReferralFailed,
  resolveTalkTogetherPassStatusForOwnedReferralCode,
} from './referral';
import { MANAGED_TRIAL_BUDGET_POLICY } from './trial-policy';

export const MANAGED_KEY_DELIVERY_ACK_TTL_MS = 15 * 60_000;

const DELIVERY_ACK_TOKEN_HASH_PREFIX = 'sha256-base64url-v1_';
const DELIVERY_ACK_TOKEN_PREFIX = 'mkd_ack_v1_';
const USD_CENTS = 100;
const REFERRED_REFERRAL_REWARD_CENTS = 2;
const textEncoder = new TextEncoder();

interface ManagedKeyDeliveryAckRequestBody {
  delivery_id?: unknown;
  managed_credential_ref?: unknown;
  delivery_ack_token?: unknown;
}

export interface PendingManagedKeyDelivery {
  delivery_id: string;
  delivery_ack_token: string;
  delivery_ack_expires_at: string;
}

interface DeliveryTalkTogetherPassStatus {
  pass_id: string;
  invite_count: number;
  invite_limit: number;
  bonus_translations_per_friend: number;
}

interface DiscordDeliveryAckFinalizationResult {
  referral_bonus_applied: boolean;
  referral_id: string | null;
  talk_together_pass: DeliveryTalkTogetherPassStatus | null;
}

export interface ManagedKeyDeliveryReconciliationResult {
  scanned: number;
  expired: number;
  cleanupRequired: number;
}

export async function createPendingManagedKeyDelivery(
  db: D1Database,
  input: {
    issueSource: ManagedIssueSource;
    subjectRef: string | null;
    installationId: string | null;
    managedCredentialRef: string;
    now: Date;
    ttlMs?: number;
  },
): Promise<PendingManagedKeyDelivery> {
  const deliveryId = `mkd_v1_${crypto.randomUUID()}`;
  const ackToken = generateDeliveryAckToken();
  const expiresAt = new Date(
    input.now.getTime() + (input.ttlMs ?? MANAGED_KEY_DELIVERY_ACK_TTL_MS),
  ).toISOString();
  await db
    .prepare(
      `INSERT INTO managed_key_deliveries (
          delivery_id,
          issue_source,
          subject_ref,
          installation_id,
          managed_credential_ref,
          ack_token_hash,
          status,
          created_at,
          expires_at,
          acknowledged_at,
          failed_at,
          failure_reason
        ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, NULL, NULL, NULL)`,
    )
    .bind(
      deliveryId,
      input.issueSource,
      input.subjectRef,
      input.installationId,
      input.managedCredentialRef,
      await hashDeliveryAckToken(ackToken),
      input.now.toISOString(),
      expiresAt,
    )
    .run();

  return {
    delivery_id: deliveryId,
    delivery_ack_token: ackToken,
    delivery_ack_expires_at: expiresAt,
  };
}

export async function handleManagedKeyDeliveryAck(
  c: Context<BrokerEnv>,
): Promise<Response> {
  const body = await readJsonBody<ManagedKeyDeliveryAckRequestBody>(c);
  if (!body.ok) {
    return invalidRequestBodyResponse(c, body.reason);
  }

  const deliveryId = nonEmptyString(body.value.delivery_id);
  const managedCredentialRef = nonEmptyString(body.value.managed_credential_ref);
  const ackToken = nonEmptyString(body.value.delivery_ack_token);
  if (!deliveryId || !managedCredentialRef || !ackToken) {
    return invalidRequestResponse(
      c,
      'delivery_id, managed_credential_ref, and delivery_ack_token are required',
    );
  }

  const delivery = await getManagedKeyDelivery(c.env.BROKER_DB, deliveryId);
  if (!delivery) {
    return invalidAckTokenResponse(c);
  }
  const tokenHash = await hashDeliveryAckToken(ackToken);
  const tokenValid = constantTimeEqual(delivery.ack_token_hash, tokenHash);
  if (!tokenValid || delivery.managed_credential_ref !== managedCredentialRef) {
    return invalidAckTokenResponse(c);
  }

  if (delivery.status === 'acknowledged') {
    return c.json({ ok: true, status: 'already_acknowledged' });
  }
  if (delivery.status !== 'pending') {
    return deliveryNoLongerPendingResponse(c, delivery.status);
  }

  const now = new Date();
  const nowIso = now.toISOString();
  if (Date.parse(delivery.expires_at) <= now.getTime()) {
    await expireManagedKeyDeliveryAfterCleanup(c.env, { delivery, nowIso });
    return deliveryExpiredResponse(c);
  }

  try {
    if (delivery.issue_source === 'discord') {
      const result = await finalizeDiscordManagedKeyDeliveryAck(c, {
        delivery,
        now,
        nowIso,
      });
      await markDeliveryAcknowledged(c.env.BROKER_DB, {
        deliveryId: delivery.delivery_id,
        managedCredentialRef: delivery.managed_credential_ref,
        nowIso,
      });
      return c.json({
        ok: true,
        status: 'acknowledged',
        ...(result.referral_bonus_applied
          ? { referral_bonus_applied: true }
          : {}),
        ...(result.referral_id ? { referral_id: result.referral_id } : {}),
        ...(result.talk_together_pass
          ? { talk_together_pass: result.talk_together_pass }
          : {}),
      });
    }

    await finalizeQqManagedKeyDeliveryAck(c, {
      delivery,
      now,
      nowIso,
    });
    await markDeliveryAcknowledged(c.env.BROKER_DB, {
      deliveryId: delivery.delivery_id,
      managedCredentialRef: delivery.managed_credential_ref,
      nowIso,
    });
    return c.json({ ok: true, status: 'acknowledged' });
  } catch (error) {
    console.error('managed_key_delivery_ack_failed', {
      delivery_id: delivery.delivery_id,
      issue_source: delivery.issue_source,
      managed_credential_ref: delivery.managed_credential_ref,
      error_name: safeErrorName(error),
      broker_timestamp: nowIso,
    });
    return internalErrorResponse(c);
  }
}

export async function reconcileStaleManagedKeyDeliveries(
  env: Pick<BrokerBindings, 'BROKER_DB' | 'OPENROUTER_MANAGEMENT_API_KEY'>,
  input: { now: Date },
): Promise<ManagedKeyDeliveryReconciliationResult> {
  const nowIso = input.now.toISOString();
  const rows = await listExpiredPendingManagedKeyDeliveries(env.BROKER_DB, nowIso);
  let expired = 0;
  let cleanupRequired = 0;

  for (const delivery of rows) {
    const outcome = await expireManagedKeyDeliveryAfterCleanup(env, {
      delivery,
      nowIso,
    });
    if (outcome === 'expired') {
      expired += 1;
    } else {
      cleanupRequired += 1;
    }
  }

  return {
    scanned: rows.length,
    expired,
    cleanupRequired,
  };
}

async function expireManagedKeyDeliveryAfterCleanup(
  env: Pick<BrokerBindings, 'BROKER_DB' | 'OPENROUTER_MANAGEMENT_API_KEY'>,
  input: { delivery: ManagedKeyDeliveryRecord; nowIso: string },
): Promise<'expired' | 'cleanup_required'> {
  const cleanup = await cleanupManagedChildKey({
    managementApiKey: env.OPENROUTER_MANAGEMENT_API_KEY,
    keyHash: input.delivery.managed_credential_ref,
  });

  if (cleanup.ok) {
    const released = await releaseExpiredDeliveryAfterCleanup(env.BROKER_DB, {
      delivery: input.delivery,
      nowIso: input.nowIso,
    });
    if (released) {
      await markDeliveryExpired(env.BROKER_DB, {
        deliveryId: input.delivery.delivery_id,
        managedCredentialRef: input.delivery.managed_credential_ref,
        nowIso: input.nowIso,
      });
      return 'expired';
    }
  }

  await markExpiredDeliveryCleanupRequired(env.BROKER_DB, {
    delivery: input.delivery,
    nowIso: input.nowIso,
    failureReason: cleanup.ok ? 'state_release_failed' : 'child_key_cleanup_failed',
  });
  logDeliveryCleanupRequired({
    delivery: input.delivery,
    cleanup,
    nowIso: input.nowIso,
  });
  return 'cleanup_required';
}

async function finalizeDiscordManagedKeyDeliveryAck(
  c: Context<BrokerEnv>,
  input: {
    delivery: ManagedKeyDeliveryRecord;
    now: Date;
    nowIso: string;
  },
): Promise<DiscordDeliveryAckFinalizationResult> {
  if (!input.delivery.installation_id || !input.delivery.subject_ref) {
    throw new Error('Discord delivery row is missing subject metadata');
  }

  const entitlement = await getEntitlementByManagedCredentialRef(
    c.env.BROKER_DB,
    input.delivery.managed_credential_ref,
  );
  if (!entitlement || entitlement.discord_user_ref !== input.delivery.subject_ref) {
    throw new Error('Discord delivery entitlement is missing');
  }

  const referralBudget = await resolveDiscordDeliveryReferralBudget(c.env.BROKER_DB, {
    delivery: input.delivery,
    managementApiKey: c.env.OPENROUTER_MANAGEMENT_API_KEY,
  });
  const finalBudgetUsd = referralBudget?.budgetUsd ?? entitlement.budget_usd;

  if (entitlement.discord_issue_status !== 'active') {
    const activated = await activateDiscordDelivery(c.env.BROKER_DB, {
      installationId: input.delivery.installation_id,
      discordUserRef: input.delivery.subject_ref,
      managedCredentialRef: input.delivery.managed_credential_ref,
      budgetUsd: finalBudgetUsd,
      deliveredAt: input.nowIso,
    });
    if (!activated) {
      throw new Error('Discord delivery activation failed');
    }
  } else {
    if (referralBudget) {
      await updateActiveDiscordDeliveryBudget(c.env.BROKER_DB, {
        installationId: input.delivery.installation_id,
        discordUserRef: input.delivery.subject_ref,
        managedCredentialRef: input.delivery.managed_credential_ref,
        budgetUsd: referralBudget.budgetUsd,
      });
    }
    await markDiscordIdentityActive(c.env.BROKER_DB, {
      installationId: input.delivery.installation_id,
      discordUserRef: input.delivery.subject_ref,
      deliveredAt: input.nowIso,
    });
  }

  await bestEffortRecordDeliveryIssueSuccess(c, {
    issueSource: 'discord',
    installationId: input.delivery.installation_id,
    subjectRef: input.delivery.installation_id,
    managedCredentialRef: input.delivery.managed_credential_ref,
    observedAt: input.nowIso,
    now: input.now,
  });

  const referralBonusApplied = await creditReservedReferralRewardForDelivery(
    c.env.BROKER_DB,
    {
      referredDiscordUserRef: input.delivery.subject_ref,
      referredInstallationId: input.delivery.installation_id,
      referredManagedCredentialRef: input.delivery.managed_credential_ref,
      managementApiKey: c.env.OPENROUTER_MANAGEMENT_API_KEY,
      nowIso: input.nowIso,
    },
  );
  const ownedReferral = await bestEffortResolveOwnedReferralForAck(c.env.BROKER_DB, {
    installationId: input.delivery.installation_id,
    nowIso: input.nowIso,
  });

  return {
    referral_bonus_applied: referralBonusApplied,
    referral_id: ownedReferral?.referral_id ?? null,
    talk_together_pass: ownedReferral?.talk_together_pass ?? null,
  };
}

async function finalizeQqManagedKeyDeliveryAck(
  c: Context<BrokerEnv>,
  input: {
    delivery: ManagedKeyDeliveryRecord;
    now: Date;
    nowIso: string;
  },
): Promise<void> {
  if (!input.delivery.subject_ref) {
    throw new Error('QQ delivery row is missing subject metadata');
  }

  const entitlement = await getQqEntitlementByManagedCredentialRef(
    c.env.BROKER_DB,
    input.delivery.managed_credential_ref,
  );
  if (!entitlement || entitlement.qq_subject_ref !== input.delivery.subject_ref) {
    throw new Error('QQ delivery entitlement is missing');
  }

  if (entitlement.status !== 'active') {
    const activated = await activateQqDelivery(c.env.BROKER_DB, {
      qqSubjectRef: input.delivery.subject_ref,
      managedCredentialRef: input.delivery.managed_credential_ref,
      deliveredAt: input.nowIso,
    });
    if (!activated) {
      throw new Error('QQ delivery activation failed');
    }
  }

  await bestEffortRecordDeliveryIssueSuccess(c, {
    issueSource: 'qq',
    installationId: null,
    subjectRef: input.delivery.subject_ref,
    managedCredentialRef: input.delivery.managed_credential_ref,
    observedAt: input.nowIso,
    now: input.now,
  });
}

async function bestEffortRecordDeliveryIssueSuccess(
  c: Context<BrokerEnv>,
  input: {
    issueSource: ManagedIssueSource;
    installationId: string | null;
    subjectRef: string;
    managedCredentialRef: string;
    observedAt: string;
    now: Date;
  },
): Promise<void> {
  try {
    const network = await extractRequestNetworkMetadata(c, c.env.BROKER_DB);
    if (input.issueSource === 'qq') {
      await recordIssueSuccess(c.env.BROKER_DB, {
        issueSource: 'qq',
        subjectRef: input.subjectRef,
        managedCredentialRef: input.managedCredentialRef,
        observedAt: input.observedAt,
        network,
      });
    } else {
      await recordIssueSuccess(c.env.BROKER_DB, {
        issueSource: 'discord',
        installationId: input.installationId!,
        subjectRef: input.subjectRef,
        managedCredentialRef: input.managedCredentialRef,
        observedAt: input.observedAt,
        network,
      });
    }
    const monitoringResult = await evaluateImmediateAbuseState(c.env.BROKER_DB, input.now);
    await deliverImmediateMonitoringSideEffects(c.env, monitoringResult);
  } catch (error) {
    console.error('managed_key_delivery_issue_success_monitoring_failed', {
      issue_source: input.issueSource,
      subject_ref: input.subjectRef,
      managed_credential_ref: input.managedCredentialRef,
      error_name: safeErrorName(error),
      broker_timestamp: new Date().toISOString(),
    });
  }
}

async function creditReservedReferralRewardForDelivery(
  db: D1Database,
  input: {
    referredDiscordUserRef: string;
    referredInstallationId: string;
    referredManagedCredentialRef: string;
    managementApiKey: string;
    nowIso: string;
  },
): Promise<boolean> {
  const reward = await findReservedReferralReward(db, {
    referredDiscordUserRef: input.referredDiscordUserRef,
    referredInstallationId: input.referredInstallationId,
  });
  if (!reward) {
    return false;
  }

  const credited = await markReservedIssueReferralCredited(db, {
    referralId: reward.referral_id,
    referredDiscordUserRef: input.referredDiscordUserRef,
    referredInstallationId: input.referredInstallationId,
    referredManagedCredentialRef: input.referredManagedCredentialRef,
    nowIso: input.nowIso,
  });
  if (!credited) {
    return false;
  }

  try {
    await applyCreditedIssueReferrerRewardLimitUpdate(db, {
      referralId: reward.referral_id,
      referredDiscordUserRef: input.referredDiscordUserRef,
      referredInstallationId: input.referredInstallationId,
      managementApiKey: input.managementApiKey,
      nowIso: input.nowIso,
    });
  } catch {
  }

  return true;
}

async function bestEffortResolveOwnedReferralForAck(
  db: D1Database,
  input: { installationId: string; nowIso: string },
): Promise<{
  referral_id: string;
  talk_together_pass: DeliveryTalkTogetherPassStatus | null;
} | null> {
  try {
    const result = await ensureOwnedReferralIdForActiveDiscordManagedUser(db, input);
    if (!result.ok) {
      return null;
    }
    try {
      return {
        referral_id: result.referralCode.referral_id,
        talk_together_pass: await resolveTalkTogetherPassStatusForOwnedReferralCode(
          db,
          result.referralCode,
        ),
      };
    } catch {
      return {
        referral_id: result.referralCode.referral_id,
        talk_together_pass: null,
      };
    }
  } catch {
    return null;
  }
}

async function releaseExpiredDeliveryAfterCleanup(
  db: D1Database,
  input: { delivery: ManagedKeyDeliveryRecord; nowIso: string },
): Promise<boolean> {
  if (input.delivery.issue_source === 'discord') {
    if (!input.delivery.subject_ref || !input.delivery.installation_id) {
      return false;
    }
    await failReservedReferralForDelivery(db, {
      referredDiscordUserRef: input.delivery.subject_ref,
      referredInstallationId: input.delivery.installation_id,
      nowIso: input.nowIso,
    });
    const entitlementResult = await db
      .prepare(
        `DELETE FROM openrouter_entitlements
          WHERE installation_id = ?
            AND discord_user_ref = ?
            AND managed_credential_ref = ?
            AND status = 'pending_release'
            AND discord_issue_status = 'delivery_pending'`,
      )
      .bind(
        input.delivery.installation_id,
        input.delivery.subject_ref,
        input.delivery.managed_credential_ref,
      )
      .run();
    const identityResult = await db
      .prepare(
        `DELETE FROM discord_identities
          WHERE discord_user_ref = ?
            AND entitlement_installation_id = ?
            AND status = 'issuing'`,
      )
      .bind(input.delivery.subject_ref, input.delivery.installation_id)
      .run();
    return (
      Number(entitlementResult.meta.changes ?? 0) === 1 &&
      Number(identityResult.meta.changes ?? 0) === 1
    );
  }

  if (!input.delivery.subject_ref) {
    return false;
  }
  const result = await db
    .prepare(
      `DELETE FROM qq_managed_entitlements
        WHERE qq_subject_ref = ?
          AND managed_credential_ref = ?
          AND status = 'delivery_pending'`,
    )
    .bind(input.delivery.subject_ref, input.delivery.managed_credential_ref)
    .run();
  return Number(result.meta.changes ?? 0) === 1;
}

async function resolveDiscordDeliveryReferralBudget(
  db: D1Database,
  input: { delivery: ManagedKeyDeliveryRecord; managementApiKey: string },
): Promise<{ budgetUsd: number } | null> {
  if (!input.delivery.subject_ref || !input.delivery.installation_id) {
    return null;
  }

  const reward = await findReservedReferralReward(db, {
    referredDiscordUserRef: input.delivery.subject_ref,
    referredInstallationId: input.delivery.installation_id,
  });
  if (!reward) {
    return null;
  }

  const budgetUsd = referredReferralBudgetUsd();
  await updateManagedChildKeyLimit({
    managementApiKey: input.managementApiKey,
    keyHash: input.delivery.managed_credential_ref,
    limitUsd: budgetUsd,
  });
  return { budgetUsd };
}

function referredReferralBudgetUsd(): number {
  return usdFromCents(
    centsFromUsd(MANAGED_TRIAL_BUDGET_POLICY.hardLimit) +
      REFERRED_REFERRAL_REWARD_CENTS,
  );
}

function centsFromUsd(value: number): number {
  if (!Number.isFinite(value) || value < 0) {
    throw new Error('managed budget must be a finite non-negative USD value');
  }
  return Math.round(value * USD_CENTS);
}

function usdFromCents(cents: number): number {
  return Number((cents / USD_CENTS).toFixed(2));
}

async function markExpiredDeliveryCleanupRequired(
  db: D1Database,
  input: {
    delivery: ManagedKeyDeliveryRecord;
    nowIso: string;
    failureReason: string;
  },
): Promise<void> {
  await db
    .prepare(
      `UPDATE managed_key_deliveries
          SET status = 'cleanup_required',
              failed_at = ?,
              failure_reason = ?
        WHERE delivery_id = ?
          AND status = 'pending'`,
    )
    .bind(input.nowIso, input.failureReason, input.delivery.delivery_id)
    .run();

  if (input.delivery.issue_source === 'discord') {
    if (!input.delivery.subject_ref || !input.delivery.installation_id) {
      return;
    }
    await failReservedReferralForDelivery(db, {
      referredDiscordUserRef: input.delivery.subject_ref,
      referredInstallationId: input.delivery.installation_id,
      nowIso: input.nowIso,
    });
    await db
      .prepare(
        `UPDATE openrouter_entitlements
            SET discord_issue_status = 'cleanup_required',
                discord_issue_delivered_at = NULL
          WHERE installation_id = ?
            AND discord_user_ref = ?
            AND managed_credential_ref = ?
            AND status = 'pending_release'
            AND discord_issue_status = 'delivery_pending'`,
      )
      .bind(
        input.delivery.installation_id,
        input.delivery.subject_ref,
        input.delivery.managed_credential_ref,
      )
      .run();
    await db
      .prepare(
        `UPDATE discord_identities
            SET status = 'cleanup_required',
                updated_at = ?
          WHERE discord_user_ref = ?
            AND entitlement_installation_id = ?
            AND status = 'issuing'`,
      )
      .bind(input.nowIso, input.delivery.subject_ref, input.delivery.installation_id)
      .run();
    return;
  }

  if (!input.delivery.subject_ref) {
    return;
  }
  await db
    .prepare(
      `UPDATE qq_managed_entitlements
          SET status = 'cleanup_required',
              updated_at = ?
        WHERE qq_subject_ref = ?
          AND managed_credential_ref = ?
          AND status = 'delivery_pending'`,
    )
    .bind(input.nowIso, input.delivery.subject_ref, input.delivery.managed_credential_ref)
    .run();
}

async function failReservedReferralForDelivery(
  db: D1Database,
  input: {
    referredDiscordUserRef: string;
    referredInstallationId: string;
    nowIso: string;
  },
): Promise<void> {
  const reward = await findReservedReferralReward(db, input);
  if (!reward) {
    return;
  }
  await markReservedIssueReferralFailed(db, {
    referralId: reward.referral_id,
    referredDiscordUserRef: input.referredDiscordUserRef,
    referredInstallationId: input.referredInstallationId,
    failureReason: 'issue_delivery_failed',
    nowIso: input.nowIso,
  });
}

async function findReservedReferralReward(
  db: D1Database,
  input: {
    referredDiscordUserRef: string;
    referredInstallationId: string;
  },
): Promise<ReferralRewardRecord | null> {
  return db
    .prepare(
      `SELECT id, referral_id, referrer_discord_user_ref, referrer_installation_id,
              referred_discord_user_ref, referred_installation_id,
              referred_hardware_hash, referred_hardware_hash_salt_version,
              referred_bonus_status, referrer_bonus_status, skip_reason,
              failure_reason, referred_managed_credential_ref,
              referrer_managed_credential_ref, attempt_ip_hash, created_at,
              updated_at, credited_at
         FROM referral_rewards
        WHERE referred_discord_user_ref = ?
          AND referred_installation_id = ?
          AND referred_bonus_status = 'reserved'
        ORDER BY id ASC
        LIMIT 1`,
    )
    .bind(input.referredDiscordUserRef, input.referredInstallationId)
    .first<ReferralRewardRecord>();
}

async function getManagedKeyDelivery(
  db: D1Database,
  deliveryId: string,
): Promise<ManagedKeyDeliveryRecord | null> {
  return db
    .prepare(
      `SELECT delivery_id, issue_source, subject_ref, installation_id,
              managed_credential_ref, ack_token_hash, status, created_at,
              expires_at, acknowledged_at, failed_at, failure_reason
         FROM managed_key_deliveries
        WHERE delivery_id = ?`,
    )
    .bind(deliveryId)
    .first<ManagedKeyDeliveryRecord>();
}

async function listExpiredPendingManagedKeyDeliveries(
  db: D1Database,
  nowIso: string,
): Promise<ManagedKeyDeliveryRecord[]> {
  const result = await db
    .prepare(
      `SELECT delivery_id, issue_source, subject_ref, installation_id,
              managed_credential_ref, ack_token_hash, status, created_at,
              expires_at, acknowledged_at, failed_at, failure_reason
         FROM managed_key_deliveries
        WHERE status = 'pending'
          AND expires_at <= ?
        ORDER BY expires_at ASC, created_at ASC`,
    )
    .bind(nowIso)
    .all<ManagedKeyDeliveryRecord>();
  return result.results;
}

async function markDeliveryAcknowledged(
  db: D1Database,
  input: { deliveryId: string; managedCredentialRef: string; nowIso: string },
): Promise<void> {
  const result = await db
    .prepare(
      `UPDATE managed_key_deliveries
          SET status = 'acknowledged',
              acknowledged_at = ?
        WHERE delivery_id = ?
          AND managed_credential_ref = ?
          AND status = 'pending'`,
    )
    .bind(input.nowIso, input.deliveryId, input.managedCredentialRef)
    .run();
  if (Number(result.meta.changes ?? 0) !== 1) {
    throw new Error('managed key delivery acknowledgement transition failed');
  }
}

async function markDeliveryExpired(
  db: D1Database,
  input: { deliveryId: string; managedCredentialRef: string; nowIso: string },
): Promise<void> {
  await db
    .prepare(
      `UPDATE managed_key_deliveries
          SET status = 'expired',
              failed_at = ?,
              failure_reason = 'delivery_ack_expired'
        WHERE delivery_id = ?
          AND managed_credential_ref = ?
          AND status = 'pending'`,
    )
    .bind(input.nowIso, input.deliveryId, input.managedCredentialRef)
    .run();
}

async function activateDiscordDelivery(
  db: D1Database,
  input: {
    installationId: string;
    discordUserRef: string;
    managedCredentialRef: string;
    budgetUsd: number;
    deliveredAt: string;
  },
): Promise<boolean> {
  const entitlementResult = await db
    .prepare(
      `UPDATE openrouter_entitlements
          SET status = 'active',
              budget_usd = ?,
              discord_issue_status = 'active',
              discord_issue_delivered_at = ?
        WHERE installation_id = ?
          AND discord_user_ref = ?
          AND managed_credential_ref = ?
          AND status = 'pending_release'
          AND discord_issue_status = 'delivery_pending'`,
    )
    .bind(
      input.budgetUsd,
      input.deliveredAt,
      input.installationId,
      input.discordUserRef,
      input.managedCredentialRef,
    )
    .run();
  if (Number(entitlementResult.meta.changes ?? 0) !== 1) {
    return false;
  }

  const identityResult = await db
    .prepare(
      `UPDATE discord_identities
          SET status = 'active',
              updated_at = ?
        WHERE discord_user_ref = ?
          AND entitlement_installation_id = ?
          AND status IN ('issuing', 'active')`,
    )
    .bind(input.deliveredAt, input.discordUserRef, input.installationId)
    .run();
  return Number(identityResult.meta.changes ?? 0) === 1;
}

async function updateActiveDiscordDeliveryBudget(
  db: D1Database,
  input: {
    installationId: string;
    discordUserRef: string;
    managedCredentialRef: string;
    budgetUsd: number;
  },
): Promise<void> {
  const result = await db
    .prepare(
      `UPDATE openrouter_entitlements
          SET budget_usd = ?
        WHERE installation_id = ?
          AND discord_user_ref = ?
          AND managed_credential_ref = ?
          AND status = 'active'
          AND discord_issue_status = 'active'`,
    )
    .bind(
      input.budgetUsd,
      input.installationId,
      input.discordUserRef,
      input.managedCredentialRef,
    )
    .run();
  if (Number(result.meta.changes ?? 0) !== 1) {
    throw new Error('Discord delivery referral budget update failed');
  }
}

async function markDiscordIdentityActive(
  db: D1Database,
  input: { installationId: string; discordUserRef: string; deliveredAt: string },
): Promise<void> {
  await db
    .prepare(
      `UPDATE discord_identities
          SET status = 'active',
              updated_at = ?
        WHERE discord_user_ref = ?
          AND entitlement_installation_id = ?
          AND status IN ('issuing', 'active')`,
    )
    .bind(input.deliveredAt, input.discordUserRef, input.installationId)
    .run();
}

async function activateQqDelivery(
  db: D1Database,
  input: { qqSubjectRef: string; managedCredentialRef: string; deliveredAt: string },
): Promise<boolean> {
  const result = await db
    .prepare(
      `UPDATE qq_managed_entitlements
          SET status = 'active',
              delivered_at = ?,
              updated_at = ?
        WHERE qq_subject_ref = ?
          AND managed_credential_ref = ?
          AND status = 'delivery_pending'`,
    )
    .bind(
      input.deliveredAt,
      input.deliveredAt,
      input.qqSubjectRef,
      input.managedCredentialRef,
    )
    .run();
  return Number(result.meta.changes ?? 0) === 1;
}

async function getEntitlementByManagedCredentialRef(
  db: D1Database,
  managedCredentialRef: string,
): Promise<OpenRouterEntitlementRecord | null> {
  return db
    .prepare(
      `SELECT installation_id, status, budget_usd, managed_credential_ref, issued_at,
              expires_at, release_session_ref, release_token_hash, release_token_expires_at,
              verified_hardware_hash, verified_hardware_hash_salt_version,
              discord_user_ref, discord_issue_status, discord_issue_reserved_at,
              discord_issue_delivered_at
         FROM openrouter_entitlements
        WHERE managed_credential_ref = ?`,
    )
    .bind(managedCredentialRef)
    .first<OpenRouterEntitlementRecord>();
}

async function getQqEntitlementByManagedCredentialRef(
  db: D1Database,
  managedCredentialRef: string,
): Promise<QqManagedEntitlementRecord | null> {
  return db
    .prepare(
      `SELECT qq_subject_ref, status, issue_ref, managed_credential_ref,
              budget_usd, reserved_at, issued_at, expires_at, delivered_at,
              created_at, updated_at
         FROM qq_managed_entitlements
        WHERE managed_credential_ref = ?`,
    )
    .bind(managedCredentialRef)
    .first<QqManagedEntitlementRecord>();
}

function logDeliveryCleanupRequired(input: {
  delivery: ManagedKeyDeliveryRecord;
  cleanup: ManagedChildKeyCleanupResult;
  nowIso: string;
}): void {
  if (input.cleanup.ok) {
    console.error('managed_key_delivery_cleanup_required', {
      event: 'managed_key_delivery_cleanup_required',
      delivery_id: input.delivery.delivery_id,
      issue_source: input.delivery.issue_source,
      subject_ref: input.delivery.subject_ref,
      managed_credential_ref: input.delivery.managed_credential_ref,
      cleanup_outcome: { ok: true },
      broker_timestamp: input.nowIso,
    });
    return;
  }

  console.error(
    'managed_key_delivery_cleanup_required',
    buildManagedCleanupRequiredAuditPayload({
      issueSource: input.delivery.issue_source,
      subjectRef: input.delivery.subject_ref ?? 'unknown',
      issueRef: input.delivery.delivery_id,
      managedCredentialRef: input.delivery.managed_credential_ref,
      failure: { name: 'DeliveryAckExpired' },
      cleanupOutcome: input.cleanup.reason,
      brokerTimestamp: input.nowIso,
    }),
  );
}

function generateDeliveryAckToken(): string {
  return `${DELIVERY_ACK_TOKEN_PREFIX}${crypto.randomUUID()}_${crypto.randomUUID()}`;
}

async function hashDeliveryAckToken(token: string): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', textEncoder.encode(token));
  return `${DELIVERY_ACK_TOKEN_HASH_PREFIX}${base64UrlEncode(new Uint8Array(digest))}`;
}

function base64UrlEncode(bytes: Uint8Array): string {
  let binary = '';
  for (const byte of bytes) {
    binary += String.fromCharCode(byte);
  }
  return btoa(binary).replaceAll('+', '-').replaceAll('/', '_').replaceAll('=', '');
}

function constantTimeEqual(left: string, right: string): boolean {
  const leftBytes = textEncoder.encode(left);
  const rightBytes = textEncoder.encode(right);
  if (leftBytes.length !== rightBytes.length) {
    return false;
  }
  let diff = 0;
  for (let index = 0; index < leftBytes.length; index += 1) {
    diff |= leftBytes[index]! ^ rightBytes[index]!;
  }
  return diff === 0;
}

async function readJsonBody<T>(
  c: Context<BrokerEnv>,
): Promise<
  | { ok: true; value: T }
  | { ok: false; reason: 'invalid_json' | 'not_object' }
> {
  try {
    const value = await c.req.json();
    if (typeof value !== 'object' || value === null || Array.isArray(value)) {
      return { ok: false, reason: 'not_object' };
    }

    return { ok: true, value: value as T };
  } catch {
    return { ok: false, reason: 'invalid_json' };
  }
}

function invalidRequestBodyResponse(
  c: Context<BrokerEnv>,
  reason: 'invalid_json' | 'not_object',
): Response {
  return invalidRequestResponse(
    c,
    reason === 'invalid_json'
      ? 'request body must be valid JSON'
      : 'request body must be a JSON object',
  );
}

function invalidRequestResponse(c: Context<BrokerEnv>, message: string): Response {
  return publicErrorResponse(c, 400, {
    code: 'invalid_request',
    class: 'terminal',
    message,
    entitlement: null,
  });
}

function invalidAckTokenResponse(c: Context<BrokerEnv>): Response {
  return publicErrorResponse(c, 401, {
    code: 'invalid_request',
    class: 'security_fail',
    subcode: 'managed_key_delivery_ack_invalid',
    message: 'Managed key delivery acknowledgement is invalid',
    entitlement: null,
  });
}

function deliveryExpiredResponse(c: Context<BrokerEnv>): Response {
  return publicErrorResponse(c, 409, {
    code: 'trial_not_eligible',
    class: 'terminal',
    subcode: 'managed_key_delivery_ack_expired',
    message: 'Managed key delivery acknowledgement has expired',
    entitlement: null,
  });
}

function deliveryNoLongerPendingResponse(
  c: Context<BrokerEnv>,
  status: ManagedKeyDeliveryRecord['status'],
): Response {
  return publicErrorResponse(c, 409, {
    code: 'trial_not_eligible',
    class: 'terminal',
    subcode: `managed_key_delivery_${status}`,
    message: 'Managed key delivery is no longer pending',
    entitlement: null,
  });
}

function safeErrorName(error: unknown): string {
  if (!(error instanceof Error)) {
    return 'UnknownFailure';
  }
  return ['Error', 'TypeError', 'OpenRouterManagementError'].includes(error.name)
    ? error.name
    : 'Error';
}
