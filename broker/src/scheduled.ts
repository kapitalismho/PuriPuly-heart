import { getBrokerAbuseControlsConfig } from './abuse-controls';
import {
  applyAbuseMonitoringRetention,
  deliverManagedCleanupIncident,
} from './abuse-monitoring';
import type { BrokerBindings } from './contract';
import { sendDailyReport, type DailyReportPayload } from './discord-alerts';
import {
  acknowledgeManagedKeyDeliveryCleanupClaim,
  claimStaleManagedKeyDeliveryCleanup,
  listStalePendingManagedKeyDeliveries,
  markManagedKeyDeliveryAcknowledged,
  STALE_DELIVERY_CLEANUP_CLAIM_REASON,
} from './managed-key-delivery';
import { cleanupManagedChildKey } from './openrouter-management';
import type {
  BrokerAbuseControlsConfigValue,
  ManagedKeyDeliveryRecord,
} from './persistence';
import {
  applyReferralRewardRetention,
  reconcileStaleReferralRewards,
} from './referral';
import {
  applyAppActiveDayRetention,
  getAppUsageDailyMetrics,
} from './telemetry';

const DAILY_REPORT_LEASE_MS = 15 * 60_000;
const ONE_DAY_MS = 24 * 60 * 60_000;

interface SourceCountRow {
  issue_source: 'discord' | 'qq';
  count: number;
}

interface ScheduledControllerLike {
  cron?: string;
  scheduledTime: number;
}

interface ExecutionContextLike {
  waitUntil?(promise: Promise<unknown>): void;
  passThroughOnException?(): void;
}

export interface DailyReportWindow {
  reportDateUtc: string;
  windowStart: string;
  windowEnd: string;
}

interface DailyReportDeliveryClaim {
  reportDateUtc: string;
  leaseToken: string;
}

interface DailyReportDeliveryRow {
  report_date_utc: string;
  status: 'pending' | 'delivered';
}

export async function handleScheduled(
  controller: ScheduledControllerLike,
  env: Pick<
    BrokerBindings,
    | 'BROKER_DB'
    | 'DISCORD_DAILY_REPORT_WEBHOOK_URL'
    | 'DISCORD_IMMEDIATE_ALERT_WEBHOOK_URL'
    | 'OPENROUTER_MANAGEMENT_API_KEY'
  >,
  _ctx: ExecutionContextLike,
): Promise<void> {
  const now = new Date(controller.scheduledTime);
  const failures: unknown[] = [];

  await runScheduledPhase(failures, () =>
    reconcileStaleManagedKeyDeliveries(env, now),
  );
  await runScheduledPhase(failures, () =>
    reconcileStaleReferralRewards(env.BROKER_DB, { nowIso: now.toISOString() }),
  );
  await runScheduledPhase(failures, () =>
    applyReferralRewardRetention(env.BROKER_DB, now),
  );

  let controls: BrokerAbuseControlsConfigValue | null = null;
  try {
    controls = await getBrokerAbuseControlsConfig(env.BROKER_DB);
  } catch (error) {
    failures.push(error);
  }

  if (controls && isDailyReportScheduleDue(controls.dailyReport, now)) {
    await runScheduledPhase(failures, () => runDailyReport(env, now));
  }

  await runScheduledPhase(failures, () => applyScheduledRetention(env.BROKER_DB, now));
  throwFirstScheduledFailure(failures);
}

async function applyScheduledRetention(db: D1Database, now: Date): Promise<void> {
  const failures: unknown[] = [];
  let preserveIssueSuccessFrom: string | null = null;
  await runScheduledPhase(failures, async () => {
    preserveIssueSuccessFrom = await resolveIssueSuccessPreservationStart(db, now);
  });
  const preservationStart = preserveIssueSuccessFrom;
  if (preservationStart !== null) {
    await runScheduledPhase(failures, () =>
      applyAbuseMonitoringRetention(db, now, {
        preserveIssueSuccessFrom: preservationStart,
      }),
    );
  }
  await runScheduledPhase(failures, () =>
    applyAppActiveDayRetention(db, now),
  );
  throwFirstScheduledFailure(failures);
}

async function runScheduledPhase(
  failures: unknown[],
  phase: () => Promise<unknown>,
): Promise<void> {
  try {
    await phase();
  } catch (error) {
    failures.push(error);
  }
}

function throwFirstScheduledFailure(failures: unknown[]): void {
  if (failures.length > 0) {
    throw failures[0];
  }
}

export async function reconcileStaleManagedKeyDeliveries(
  env: Pick<
    BrokerBindings,
    | 'BROKER_DB'
    | 'OPENROUTER_MANAGEMENT_API_KEY'
    | 'DISCORD_IMMEDIATE_ALERT_WEBHOOK_URL'
  >,
  now: Date,
): Promise<{ expired: number; cleanupRequired: number }> {
  let expired = 0;
  let cleanupRequired = 0;
  const staleDeliveries = await listStalePendingManagedKeyDeliveries(env.BROKER_DB, {
    now,
    limit: 50,
  });
  const nowIso = now.toISOString();
  for (const delivery of staleDeliveries) {
    if (await isManagedKeyDeliveryFinalized(env.BROKER_DB, delivery)) {
      const repaired =
        delivery.status === 'pending'
          ? (
              await markManagedKeyDeliveryAcknowledged(env.BROKER_DB, {
                deliveryId: delivery.delivery_id,
                acknowledgedAt: now,
              })
            ).ok
          : await acknowledgeManagedKeyDeliveryCleanupClaim(env.BROKER_DB, {
              deliveryId: delivery.delivery_id,
              acknowledgedAt: nowIso,
              expectedClaimedAt: delivery.failed_at,
            });
      if (!repaired) {
        throw new Error('failed to reconcile finalized managed key delivery');
      }
      continue;
    }
    const claimed = await claimStaleManagedKeyDeliveryCleanup(env.BROKER_DB, {
      delivery,
      claimedAt: nowIso,
    });
    if (!claimed) {
      continue;
    }
    if (await isManagedKeyDeliveryFinalized(env.BROKER_DB, delivery)) {
      const repaired = await acknowledgeManagedKeyDeliveryCleanupClaim(
        env.BROKER_DB,
        {
          deliveryId: delivery.delivery_id,
          acknowledgedAt: nowIso,
          expectedClaimedAt: nowIso,
        },
      );
      if (!repaired) {
        throw new Error('failed to reconcile claimed finalized managed key delivery');
      }
      continue;
    }
    const cleanup = await cleanupManagedChildKey({
      managementApiKey: env.OPENROUTER_MANAGEMENT_API_KEY,
      keyHash: delivery.managed_credential_ref,
    });
    if (!cleanup.ok) {
      let transition: Awaited<ReturnType<typeof markDeliveryCleanupRequired>>;
      try {
        transition = await markDeliveryCleanupRequired(
          env.BROKER_DB,
          delivery,
          nowIso,
        );
      } catch (error) {
        await deliverManagedCleanupIncident(env, {
          issueSource: delivery.issue_source,
          managedCredentialRef: delivery.managed_credential_ref,
          phase: 'stale_delivery',
          cleanupRequiredRecorded: false,
          occurredAt: nowIso,
        });
        throw error;
      }
      if (transition.incidentRecorded) {
        await deliverManagedCleanupIncident(env, {
          issueSource: delivery.issue_source,
          managedCredentialRef: delivery.managed_credential_ref,
          phase: 'stale_delivery',
          cleanupRequiredRecorded: transition.ownerRecorded,
          occurredAt: nowIso,
        });
      }
      cleanupRequired += transition.incidentRecorded ? 1 : 0;
      continue;
    }
    const completed = await completeDeliveryCleanup(
      env.BROKER_DB,
      delivery,
      nowIso,
    );
    expired += completed ? 1 : 0;
  }
  return { expired, cleanupRequired };
}

async function completeDeliveryCleanup(
  db: D1Database,
  delivery: ManagedKeyDeliveryRecord,
  nowIso: string,
): Promise<boolean> {
  if (delivery.issue_source === 'discord') {
    const results = await db.batch([
      db.prepare(
        `DELETE FROM openrouter_entitlements
          WHERE managed_credential_ref = ?
            AND status IN ('pending_release', 'active')
            AND discord_issue_status IN ('delivery_pending', 'active')
            AND EXISTS (
              SELECT 1 FROM managed_key_deliveries
               WHERE delivery_id = ?
                 AND status = 'expired'
                 AND failure_reason = ?
                 AND failed_at = ?
            )`,
      ).bind(
        delivery.managed_credential_ref,
        delivery.delivery_id,
        STALE_DELIVERY_CLEANUP_CLAIM_REASON,
        nowIso,
      ),
      db.prepare(
        `DELETE FROM discord_identities
          WHERE discord_user_ref = ?
            AND entitlement_installation_id = ?
            AND status IN ('issuing', 'active')
            AND EXISTS (
              SELECT 1 FROM managed_key_deliveries
               WHERE delivery_id = ?
                 AND status = 'expired'
                 AND failure_reason = ?
                 AND failed_at = ?
            )`,
      ).bind(
        delivery.subject_ref ?? '',
        delivery.installation_id ?? '',
        delivery.delivery_id,
        STALE_DELIVERY_CLEANUP_CLAIM_REASON,
        nowIso,
      ),
      db.prepare(
        `UPDATE referral_rewards
            SET referred_bonus_status = 'failed', referrer_bonus_status = 'failed', failure_reason = 'issue_delivery_failed', updated_at = ?
          WHERE referred_managed_credential_ref IS NULL
            AND referred_bonus_status = 'reserved'
            AND referred_installation_id = ?
            AND EXISTS (
              SELECT 1 FROM managed_key_deliveries
               WHERE delivery_id = ?
                 AND status = 'expired'
                 AND failure_reason = ?
                 AND failed_at = ?
            )`,
      ).bind(
        nowIso,
        delivery.installation_id ?? '',
        delivery.delivery_id,
        STALE_DELIVERY_CLEANUP_CLAIM_REASON,
        nowIso,
      ),
      db.prepare(
        `UPDATE managed_key_deliveries
            SET failed_at = ?, failure_reason = 'ack_expired_child_key_cleaned'
          WHERE delivery_id = ?
            AND status = 'expired'
            AND failure_reason = ?
            AND failed_at = ?`,
      ).bind(
        nowIso,
        delivery.delivery_id,
        STALE_DELIVERY_CLEANUP_CLAIM_REASON,
        nowIso,
      ),
    ]);
    return Number(results.at(-1)?.meta.changes ?? 0) === 1;
  }
  if (delivery.issue_source === 'qq') {
    const results = await db.batch([
      db.prepare(
        `DELETE FROM qq_managed_entitlements
          WHERE managed_credential_ref = ?
            AND status IN ('delivery_pending', 'active')
            AND EXISTS (
              SELECT 1 FROM managed_key_deliveries
               WHERE delivery_id = ?
                 AND status = 'expired'
                 AND failure_reason = ?
                 AND failed_at = ?
            )`,
      ).bind(
        delivery.managed_credential_ref,
        delivery.delivery_id,
        STALE_DELIVERY_CLEANUP_CLAIM_REASON,
        nowIso,
      ),
      db.prepare(
        `UPDATE managed_key_deliveries
            SET failed_at = ?, failure_reason = 'ack_expired_child_key_cleaned'
          WHERE delivery_id = ?
            AND status = 'expired'
            AND failure_reason = ?
            AND failed_at = ?`,
      ).bind(
        nowIso,
        delivery.delivery_id,
        STALE_DELIVERY_CLEANUP_CLAIM_REASON,
        nowIso,
      ),
    ]);
    return Number(results.at(-1)?.meta.changes ?? 0) === 1;
  }
  return false;
}

async function markDeliveryCleanupRequired(
  db: D1Database,
  delivery: ManagedKeyDeliveryRecord,
  nowIso: string,
): Promise<{ incidentRecorded: boolean; ownerRecorded: boolean }> {
  if (delivery.issue_source === 'discord') {
    const results = await db.batch([
      db.prepare(
        `UPDATE openrouter_entitlements
            SET status = 'pending_release',
                discord_issue_status = 'cleanup_required',
                discord_issue_delivered_at = NULL
          WHERE managed_credential_ref = ?
            AND status IN ('pending_release', 'active')
            AND discord_issue_status IN ('delivery_pending', 'active')
            AND EXISTS (
              SELECT 1 FROM managed_key_deliveries
               WHERE delivery_id = ?
                 AND status = 'expired'
                 AND failure_reason = ?
                 AND failed_at = ?
            )`,
      ).bind(
        delivery.managed_credential_ref,
        delivery.delivery_id,
        STALE_DELIVERY_CLEANUP_CLAIM_REASON,
        nowIso,
      ),
      db.prepare(
        `UPDATE discord_identities
            SET status = 'cleanup_required', updated_at = ?
          WHERE discord_user_ref = ?
            AND entitlement_installation_id = ?
            AND status IN ('issuing', 'active')
            AND EXISTS (
              SELECT 1 FROM managed_key_deliveries
               WHERE delivery_id = ?
                 AND status = 'expired'
                 AND failure_reason = ?
                 AND failed_at = ?
            )`,
      ).bind(
        nowIso,
        delivery.subject_ref ?? '',
        delivery.installation_id ?? '',
        delivery.delivery_id,
        STALE_DELIVERY_CLEANUP_CLAIM_REASON,
        nowIso,
      ),
      db.prepare(
        `UPDATE managed_key_deliveries
            SET status = 'cleanup_required',
                failed_at = ?,
                failure_reason = 'managed_child_key_cleanup_failed'
          WHERE delivery_id = ?
            AND status = 'expired'
            AND failure_reason = ?
            AND failed_at = ?`,
      ).bind(
        nowIso,
        delivery.delivery_id,
        STALE_DELIVERY_CLEANUP_CLAIM_REASON,
        nowIso,
      ),
    ]);
    return {
      incidentRecorded: Number(results[2]?.meta.changes ?? 0) === 1,
      ownerRecorded:
        Number(results[0]?.meta.changes ?? 0) === 1 &&
        Number(results[1]?.meta.changes ?? 0) === 1,
    };
  }
  if (delivery.issue_source === 'qq') {
    const results = await db.batch([
      db.prepare(
        `UPDATE qq_managed_entitlements
            SET status = 'cleanup_required', delivered_at = NULL, updated_at = ?
          WHERE managed_credential_ref = ?
            AND status IN ('delivery_pending', 'active')
            AND EXISTS (
              SELECT 1 FROM managed_key_deliveries
               WHERE delivery_id = ?
                 AND status = 'expired'
                 AND failure_reason = ?
                 AND failed_at = ?
            )`,
      ).bind(
        nowIso,
        delivery.managed_credential_ref,
        delivery.delivery_id,
        STALE_DELIVERY_CLEANUP_CLAIM_REASON,
        nowIso,
      ),
      db.prepare(
        `UPDATE managed_key_deliveries
            SET status = 'cleanup_required',
                failed_at = ?,
                failure_reason = 'managed_child_key_cleanup_failed'
          WHERE delivery_id = ?
            AND status = 'expired'
            AND failure_reason = ?
            AND failed_at = ?`,
      ).bind(
        nowIso,
        delivery.delivery_id,
        STALE_DELIVERY_CLEANUP_CLAIM_REASON,
        nowIso,
      ),
    ]);
    return {
      incidentRecorded: Number(results[1]?.meta.changes ?? 0) === 1,
      ownerRecorded: Number(results[0]?.meta.changes ?? 0) === 1,
    };
  }
  return { incidentRecorded: false, ownerRecorded: false };
}

async function isManagedKeyDeliveryFinalized(
  db: D1Database,
  delivery: ManagedKeyDeliveryRecord,
): Promise<boolean> {
  if (delivery.issue_source === 'discord') {
    const row = await db
      .prepare(
        `SELECT 1 AS finalized
           FROM openrouter_entitlements AS entitlement
           JOIN discord_identities AS identity
             ON identity.entitlement_installation_id = entitlement.installation_id
            AND identity.discord_user_ref = entitlement.discord_user_ref
          WHERE entitlement.managed_credential_ref = ?
            AND entitlement.status = 'active'
            AND entitlement.discord_issue_status = 'active'
            AND entitlement.discord_issue_delivered_at IS NOT NULL
            AND identity.status = 'active'
            AND EXISTS (
              SELECT 1
                FROM broker_issue_success_events AS event
               WHERE event.issue_source = 'discord'
                 AND event.managed_credential_ref = entitlement.managed_credential_ref
            )
          LIMIT 1`,
      )
      .bind(delivery.managed_credential_ref)
      .first<{ finalized: number }>();
    return Number(row?.finalized ?? 0) === 1;
  }

  const row = await db
    .prepare(
      `SELECT 1 AS finalized
         FROM qq_managed_entitlements AS entitlement
        WHERE entitlement.managed_credential_ref = ?
          AND entitlement.status = 'active'
          AND entitlement.delivered_at IS NOT NULL
          AND EXISTS (
            SELECT 1
              FROM broker_issue_success_events AS event
             WHERE event.issue_source = 'qq'
               AND event.managed_credential_ref = entitlement.managed_credential_ref
          )
        LIMIT 1`,
    )
    .bind(delivery.managed_credential_ref)
    .first<{ finalized: number }>();
  return Number(row?.finalized ?? 0) === 1;
}

export function resolveDailyReportWindow(now: Date): DailyReportWindow {
  const windowEndDate = startOfUtcDate(now);
  const windowStartDate = new Date(windowEndDate.getTime() - ONE_DAY_MS);
  return resolveDailyReportWindowForDate(windowStartDate.toISOString().slice(0, 10));
}

export function resolveDailyReportWindowForDate(
  reportDateUtc: string,
): DailyReportWindow {
  const windowStartDate = new Date(`${reportDateUtc}T00:00:00.000Z`);
  if (
    !/^\d{4}-\d{2}-\d{2}$/.test(reportDateUtc) ||
    Number.isNaN(windowStartDate.getTime()) ||
    windowStartDate.toISOString().slice(0, 10) !== reportDateUtc
  ) {
    throw new Error('daily report date must be a valid UTC calendar date');
  }
  const windowEndDate = new Date(windowStartDate.getTime() + ONE_DAY_MS);
  return {
    reportDateUtc,
    windowStart: windowStartDate.toISOString(),
    windowEnd: windowEndDate.toISOString(),
  };
}

function isDailyReportScheduleDue(
  config: BrokerAbuseControlsConfigValue['dailyReport'],
  now: Date,
): boolean {
  if (!config.enabled) {
    return false;
  }

  const dueMinutesUtc = config.hourUtc * 60 + config.minuteUtc;
  const nowMinutesUtc = now.getUTCHours() * 60 + now.getUTCMinutes();

  if (nowMinutesUtc < dueMinutesUtc) {
    return false;
  }

  return true;
}

export async function runDailyReport(
  env: Pick<BrokerBindings, 'BROKER_DB' | 'DISCORD_DAILY_REPORT_WEBHOOK_URL'>,
  now: Date,
): Promise<{ ok: true; payload: DailyReportPayload; sent: boolean }> {
  const reportDateUtc = await resolveNextDailyReportDate(env.BROKER_DB, now);
  const payload = await buildDailySummaryPacketForDate(
    env.BROKER_DB,
    reportDateUtc,
  );
  const claim = await claimDailyReportDelivery(
    env.BROKER_DB,
    payload.report_date_utc,
    now,
  );
  if (!claim) {
    return { ok: true, payload, sent: false };
  }

  try {
    await sendDailyReport(env.DISCORD_DAILY_REPORT_WEBHOOK_URL, payload);
  } catch (error) {
    await releaseDailyReportDelivery(env.BROKER_DB, claim, now);
    throw error;
  }
  await completeDailyReportDelivery(env.BROKER_DB, claim, now);

  return { ok: true, payload, sent: true };
}

export async function buildDailySummaryPacket(
  db: BrokerBindings['BROKER_DB'],
  now: Date,
): Promise<DailyReportPayload> {
  const window = resolveDailyReportWindow(now);
  return buildDailySummaryPacketForWindow(db, window);
}

async function buildDailySummaryPacketForDate(
  db: BrokerBindings['BROKER_DB'],
  reportDateUtc: string,
): Promise<DailyReportPayload> {
  return buildDailySummaryPacketForWindow(
    db,
    resolveDailyReportWindowForDate(reportDateUtc),
  );
}

async function buildDailySummaryPacketForWindow(
  db: BrokerBindings['BROKER_DB'],
  window: DailyReportWindow,
): Promise<DailyReportPayload> {
  const [sourceCountsResult, appUsage] = await Promise.all([
    db
      .prepare(
        `SELECT issue_source,
                COUNT(DISTINCT CASE
                  WHEN managed_credential_ref IS NULL THEN 'legacy-event:' || id
                  ELSE 'managed-credential:' || managed_credential_ref
                END) AS count
           FROM broker_issue_success_events
          WHERE julianday(observed_at) >= julianday(?)
            AND julianday(observed_at) < julianday(?)
          GROUP BY issue_source`,
      )
      .bind(window.windowStart, window.windowEnd)
      .all<SourceCountRow>(),
    getAppUsageDailyMetrics(db, window.reportDateUtc),
  ]);
  const sourceCounts = Object.fromEntries(
    sourceCountsResult.results.map((row) => [row.issue_source, Number(row.count)]),
  ) as Partial<Record<SourceCountRow['issue_source'], number>>;
  const keysDeliveredDiscord = sourceCounts.discord ?? 0;
  const keysDeliveredQq = sourceCounts.qq ?? 0;

  return {
    schema_version: 'puripuly_daily_summary.v2',
    report_date_utc: window.reportDateUtc,
    window_start: window.windowStart,
    window_end: window.windowEnd,
    summary: {
      keys_delivered_total: keysDeliveredDiscord + keysDeliveredQq,
      keys_delivered_discord: keysDeliveredDiscord,
      keys_delivered_qq: keysDeliveredQq,
      ...appUsage,
    },
  };
}

async function resolveNextDailyReportDate(
  db: BrokerBindings['BROKER_DB'],
  now: Date,
): Promise<string> {
  const latestCompletedDateUtc = resolveDailyReportWindow(now).reportDateUtc;
  const rows = await db
    .prepare(
      `SELECT report_date_utc, status
         FROM broker_daily_summary_deliveries
        WHERE report_date_utc <= ?
        ORDER BY report_date_utc`,
    )
    .bind(latestCompletedDateUtc)
    .all<DailyReportDeliveryRow>();
  const pendingDateUtc = rows.results.find(
    (row) => row.status === 'pending',
  )?.report_date_utc;
  const lastDeliveredDateUtc = rows.results
    .filter((row) => row.status === 'delivered')
    .at(-1)?.report_date_utc;
  const nextAfterDeliveredUtc = lastDeliveredDateUtc
    ? addUtcDays(lastDeliveredDateUtc, 1)
    : null;
  const candidates = [pendingDateUtc, nextAfterDeliveredUtc]
    .filter((value): value is string => Boolean(value))
    .filter((value) => value <= latestCompletedDateUtc)
    .sort();
  return candidates[0] ?? latestCompletedDateUtc;
}

async function resolveIssueSuccessPreservationStart(
  db: BrokerBindings['BROKER_DB'],
  now: Date,
): Promise<string> {
  const reportDateUtc = await resolveNextDailyReportDate(db, now);
  return resolveDailyReportWindowForDate(reportDateUtc).windowStart;
}

async function claimDailyReportDelivery(
  db: BrokerBindings['BROKER_DB'],
  reportDateUtc: string,
  now: Date,
): Promise<DailyReportDeliveryClaim | null> {
  const leaseToken = crypto.randomUUID();
  const attemptedAt = now.toISOString();
  const leaseExpiresAt = new Date(
    now.getTime() + DAILY_REPORT_LEASE_MS,
  ).toISOString();
  const result = await db
    .prepare(
      `INSERT INTO broker_daily_summary_deliveries (
          report_date_utc,
          status,
          lease_token,
          lease_expires_at,
          attempted_at,
          delivered_at
        ) VALUES (?, 'pending', ?, ?, ?, NULL)
        ON CONFLICT(report_date_utc) DO UPDATE SET
          status = 'pending',
          lease_token = excluded.lease_token,
          lease_expires_at = excluded.lease_expires_at,
          attempted_at = excluded.attempted_at,
          delivered_at = NULL
        WHERE broker_daily_summary_deliveries.status = 'pending'
          AND broker_daily_summary_deliveries.lease_expires_at <= excluded.attempted_at`,
    )
    .bind(reportDateUtc, leaseToken, leaseExpiresAt, attemptedAt)
    .run();

  if (Number(result.meta?.changes ?? 0) !== 1) {
    return null;
  }

  return { reportDateUtc, leaseToken };
}

async function releaseDailyReportDelivery(
  db: BrokerBindings['BROKER_DB'],
  claim: DailyReportDeliveryClaim,
  now: Date,
): Promise<void> {
  await db
    .prepare(
      `UPDATE broker_daily_summary_deliveries
          SET lease_expires_at = ?
        WHERE report_date_utc = ?
          AND status = 'pending'
          AND lease_token = ?`,
    )
    .bind(now.toISOString(), claim.reportDateUtc, claim.leaseToken)
    .run();
}

async function completeDailyReportDelivery(
  db: BrokerBindings['BROKER_DB'],
  claim: DailyReportDeliveryClaim,
  now: Date,
): Promise<void> {
  const result = await db
    .prepare(
      `UPDATE broker_daily_summary_deliveries
          SET status = 'delivered', delivered_at = ?
        WHERE report_date_utc = ?
          AND status = 'pending'
          AND lease_token = ?`,
    )
    .bind(now.toISOString(), claim.reportDateUtc, claim.leaseToken)
    .run();

  if (Number(result.meta?.changes ?? 0) !== 1) {
    throw new Error('failed to persist daily summary delivery outcome');
  }
}

function startOfUtcDate(date: Date): Date {
  return new Date(
    Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate()),
  );
}

function addUtcDays(dateUtc: string, days: number): string {
  const date = new Date(`${dateUtc}T00:00:00.000Z`);
  date.setUTCDate(date.getUTCDate() + days);
  return date.toISOString().slice(0, 10);
}
