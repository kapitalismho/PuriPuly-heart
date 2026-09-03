import { getBrokerAbuseControlsConfig } from './abuse-controls';
import {
  resolveNetworkIdentityWriteMode,
  resolveRequestNetworkIdentity,
  type NetworkIdentitySecrets,
  type NetworkIdentityWriteMode,
} from './network-identity';

export async function resolveNetworkIdentityMaxWindowMinutes(
  db: D1Database,
): Promise<number> {
  const controls = await getBrokerAbuseControlsConfig(db);
  let maxMinutes = 1440;
  const endpointWindows = [
    controls.trialChallenge.windowMinutes,
    controls.trialChallengeVerify.windowMinutes,
    controls.openrouterIssue.windowMinutes,
    controls.trialStatus.windowMinutes,
    controls.qqAuthStatusIp.windowMinutes,
    controls.pendingDiscordOAuthSessions.windowMinutes,
    controls.referralAttempts.validShaped.windowMinutes,
    controls.referralAttempts.unknown.windowMinutes,
    controls.referralAttempts.perReferralIdVelocity.windowMinutes,
    controls.referralAttempts.perReferrerRewardVelocity.windowMinutes,
  ];
  for (const windowMinutes of endpointWindows) {
    if (Number.isFinite(windowMinutes) && windowMinutes > maxMinutes) {
      maxMinutes = windowMinutes;
    }
  }
  const hooks = await db
    .prepare(`SELECT MAX(window_minutes) AS maxWindow FROM broker_velocity_cap_hooks WHERE active = 1`)
    .first<{ maxWindow: number | null }>()
    .catch(() => null);
  if (hooks && Number.isFinite(Number(hooks.maxWindow)) && Number(hooks.maxWindow) > maxMinutes) {
    maxMinutes = Number(hooks.maxWindow);
  }
  return Math.max(1, Math.ceil(maxMinutes));
}

const BACKFILL_BATCH_LIMIT = 200;

export interface NetworkIdentityBackfillResult {
  mode: NetworkIdentityWriteMode;
  requestEventsBackfilled: number;
  pendingRequestEvents: number;
  pendingReferralEvents: number;
  finalized: boolean;
}

export async function runNetworkIdentityBackfill(
  db: D1Database,
  secrets: NetworkIdentitySecrets | null,
  now: Date,
): Promise<NetworkIdentityBackfillResult> {
  const mode = await resolveNetworkIdentityWriteMode(db);
  if (mode !== 'dual' || !secrets) {
    return { mode, requestEventsBackfilled: 0, pendingRequestEvents: 0, pendingReferralEvents: 0, finalized: mode === 'keyed' };
  }
  const maxWindowMinutes = await resolveNetworkIdentityMaxWindowMinutes(db);
  const windowStartIso = new Date(now.getTime() - maxWindowMinutes * 60_000).toISOString();
  let backfilled = 0;
  const candidates = await db
    .prepare(
      `SELECT id, ip, observed_at FROM broker_request_events
        WHERE ip IS NOT NULL AND ip_digest IS NULL AND observed_at >= ?
        ORDER BY observed_at ASC LIMIT ?`,
    )
    .bind(windowStartIso, BACKFILL_BATCH_LIMIT)
    .all<{ id: number; ip: string; observed_at: string }>();
  for (const candidate of candidates.results ?? []) {
    const observedAt = new Date(candidate.observed_at);
    const identity = await resolveRequestNetworkIdentity(candidate.ip, secrets, Number.isNaN(observedAt.getTime()) ? now : observedAt);
    if (!identity) {
      continue;
    }
    const updated = await db
      .prepare(
        `UPDATE broker_request_events
            SET ip_digest = ?, ip_key_version = ?, ip_epoch = ?
          WHERE id = ? AND ip_digest IS NULL`,
      )
      .bind(identity.digest, identity.keyVersion, identity.epoch, candidate.id)
      .run();
    backfilled += Number(updated.meta.changes ?? 0);
  }
  const pendingRequestEvents = await countPendingLegacyRequestEvents(db, windowStartIso);
  const pendingReferralEvents = await countPendingLegacyReferralEvents(db, windowStartIso);
  let finalized = false;
  if (pendingRequestEvents === 0 && pendingReferralEvents === 0) {
    finalized = await finalizeNetworkIdentityMigration(db, now);
  }
  return { mode, requestEventsBackfilled: backfilled, pendingRequestEvents, pendingReferralEvents, finalized };
}

async function countPendingLegacyRequestEvents(db: D1Database, windowStartIso: string): Promise<number> {
  const row = await db
    .prepare(
      `SELECT COUNT(*) AS count FROM broker_request_events
        WHERE ip IS NOT NULL AND ip_digest IS NULL AND observed_at >= ?`,
    )
    .bind(windowStartIso)
    .first<{ count: number }>()
    .catch(() => ({ count: 0 }));
  return Number(row?.count ?? 0);
}

async function countPendingLegacyReferralEvents(db: D1Database, windowStartIso: string): Promise<number> {
  const row = await db
    .prepare(
      `SELECT COUNT(*) AS count FROM referral_rewards
        WHERE attempt_ip_hash IS NOT NULL AND attempt_ip_digest IS NULL AND created_at >= ?`,
    )
    .bind(windowStartIso)
    .first<{ count: number }>()
    .catch(() => ({ count: 0 }));
  return Number(row?.count ?? 0);
}

async function finalizeNetworkIdentityMigration(db: D1Database, now: Date): Promise<boolean> {
  const nowIso = now.toISOString();
  await db
    .prepare(`UPDATE broker_request_events SET ip = NULL WHERE ip IS NOT NULL`)
    .run()
    .catch(() => null);
  await db
    .prepare(`UPDATE broker_issue_success_events SET ip_hash = NULL, ip_prefix_hash = NULL WHERE ip_hash IS NOT NULL OR ip_prefix_hash IS NOT NULL`)
    .run()
    .catch(() => null);
  await db
    .prepare(`UPDATE referral_rewards SET attempt_ip_hash = NULL WHERE attempt_ip_hash IS NOT NULL`)
    .run()
    .catch(() => null);
  const result = await db
    .prepare(`UPDATE broker_config SET value = ?, updated_at = ? WHERE key = 'network_identity_migration'`)
    .bind(JSON.stringify({ phase: 'keyed_only', purge_after: nowIso }), nowIso)
    .run()
    .catch(() => null);
  return Number(result?.meta.changes ?? 0) === 1;
}

