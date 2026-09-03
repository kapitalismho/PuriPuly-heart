import { getBrokerAbuseControlsConfig } from './abuse-controls';
import {
  deriveStableNetworkIdentityDigest,
  normalizeNetworkIdentityIp,
  resolveNetworkIdentityWriteMode,
  resolveRequestNetworkIdentity,
  type NetworkIdentitySecrets,
  type NetworkIdentityWriteMode,
} from './network-identity';

export const NETWORK_IDENTITY_WINDOW_CONFIG_PATHS = [
  'trialChallenge',
  'trialChallengeVerify',
  'openrouterIssue',
  'trialStatus',
  'qqAuthAssertIp',
  'qqAuthStatusIp',
  'pendingDiscordOAuthSessions',
  'referralAttempts.validShaped',
  'referralAttempts.unknown',
  'referralAttempts.perReferralIdVelocity',
  'referralAttempts.perReferrerRewardVelocity',
  'managedOperationStatusIp',
  'managedOperationStatusInstallation',
  'managedOperationResumeIp',
  'managedOperationResumeInstallation',
  'managedKeyDeliveryAckIp',
] as const;

function readWindowMinutesAtPath(controls: unknown, path: string): number | null {
  let current: unknown = controls;
  for (const segment of path.split('.')) {
    if (typeof current !== 'object' || current === null || !(segment in current)) {
      return null;
    }
    current = (current as Record<string, unknown>)[segment];
  }
  const windowMinutes = (current as { windowMinutes?: unknown } | null)?.windowMinutes;
  return typeof windowMinutes === 'number' ? windowMinutes : null;
}

export async function resolveNetworkIdentityMaxWindowMinutes(
  db: D1Database,
): Promise<number> {
  const controls = await getBrokerAbuseControlsConfig(db);
  let maxMinutes = 1440;
  const endpointWindows = NETWORK_IDENTITY_WINDOW_CONFIG_PATHS.map((path) => readWindowMinutesAtPath(controls, path));
  for (const windowMinutes of endpointWindows) {
    if (typeof windowMinutes === 'number' && Number.isFinite(windowMinutes) && windowMinutes > maxMinutes) {
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

export const UNPARSEABLE_IP_EPOCH_SENTINEL = '0000-00-00';

export interface NetworkIdentityHookBackfillResult {
  converted: number;
  pending: number;
  unparseable: number;
}

export interface NetworkIdentityBackfillResult {
  mode: NetworkIdentityWriteMode;
  requestEventsBackfilled: number;
  pendingRequestEvents: number;
  pendingReferralEvents: number;
  hooksConverted: number;
  pendingHooks: number;
  unparseableHooks: number;
  rawHooks: number;
  rawHookSampleIds: Array<{ table: string; id: number }>;
  finalized: boolean;
}

const RAW_HOOK_SAMPLE_LIMIT = 25;

async function countActiveRawHooks(
  db: D1Database,
): Promise<{ count: number; sampleIds: Array<{ table: string; id: number }> }> {
  let count = 0;
  const sampleIds: Array<{ table: string; id: number }> = [];
  for (const table of ['broker_velocity_cap_hooks', 'broker_abuse_subject_hooks'] as const) {
    const total = await db
      .prepare(
        `SELECT COUNT(*) AS count FROM ${table}
          WHERE subject_type = 'ip' AND active = 1
          AND (length(subject_value) != 64 OR subject_value GLOB '*[^0-9a-f]*')`,
      )
      .first<{ count: number }>()
      .catch(() => ({ count: 0 }));
    count += Number(total?.count ?? 0);
    if (sampleIds.length < RAW_HOOK_SAMPLE_LIMIT) {
      const sample = await db
        .prepare(
          `SELECT id FROM ${table}
            WHERE subject_type = 'ip' AND active = 1
            AND (length(subject_value) != 64 OR subject_value GLOB '*[^0-9a-f]*')
            ORDER BY id ASC LIMIT ?`,
        )
        .bind(RAW_HOOK_SAMPLE_LIMIT - sampleIds.length)
        .all<{ id: number }>()
        .catch(() => ({ results: [] as Array<{ id: number }> }));
      for (const row of sample.results ?? []) {
        sampleIds.push({ table, id: row.id });
      }
    }
  }
  return { count, sampleIds };
}

export async function runNetworkIdentityBackfill(
  db: D1Database,
  secrets: NetworkIdentitySecrets | null,
  now: Date,
): Promise<NetworkIdentityBackfillResult> {
  const hooks = secrets
    ? await runNetworkIdentityHookBackfill(db, secrets)
    : { converted: 0, pending: 0, unparseable: 0, rawHooks: 0, rawHookSampleIds: [] as Array<{ table: string; id: number }> };
  const rawHooks = await countActiveRawHooks(db);
  const mode = await resolveNetworkIdentityWriteMode(db);
  if (mode !== 'dual' || !secrets) {
    return { mode, requestEventsBackfilled: 0, pendingRequestEvents: 0, pendingReferralEvents: 0, hooksConverted: hooks.converted, pendingHooks: hooks.pending, unparseableHooks: hooks.unparseable, rawHooks: rawHooks.count, rawHookSampleIds: rawHooks.sampleIds, finalized: mode === 'keyed' && rawHooks.count === 0 };
  }
  const maxWindowMinutes = await resolveNetworkIdentityMaxWindowMinutes(db);
  const windowStartIso = new Date(now.getTime() - maxWindowMinutes * 60_000).toISOString();
  let backfilled = 0;
  const candidates = await db
    .prepare(
      `SELECT id, ip, installation_id, observed_at FROM broker_request_events
        WHERE ip IS NOT NULL AND ip_digest IS NULL AND observed_at >= ?
        ORDER BY observed_at ASC LIMIT ?`,
    )
    .bind(windowStartIso, BACKFILL_BATCH_LIMIT)
    .all<{ id: number; ip: string; installation_id: string | null; observed_at: string }>();
  for (const candidate of candidates.results ?? []) {
    const observedAt = new Date(candidate.observed_at);
    const identity = await resolveRequestNetworkIdentity(candidate.ip, secrets, Number.isNaN(observedAt.getTime()) ? now : observedAt);
    if (!identity) {
      if (!candidate.installation_id) {
        await db
          .prepare(
            `UPDATE broker_request_events
                SET ip_epoch = ?
              WHERE id = ? AND ip_digest IS NULL`,
          )
          .bind(UNPARSEABLE_IP_EPOCH_SENTINEL, candidate.id)
          .run();
      }
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
  if (pendingRequestEvents === 0 && pendingReferralEvents === 0 && hooks.pending === 0 && rawHooks.count === 0) {
    finalized = await finalizeNetworkIdentityMigration(db, now);
  }
  return { mode, requestEventsBackfilled: backfilled, pendingRequestEvents, pendingReferralEvents, hooksConverted: hooks.converted, pendingHooks: hooks.pending, unparseableHooks: hooks.unparseable, rawHooks: rawHooks.count, rawHookSampleIds: rawHooks.sampleIds, finalized };
}

export async function runNetworkIdentityHookBackfill(
  db: D1Database,
  secrets: NetworkIdentitySecrets,
): Promise<NetworkIdentityHookBackfillResult> {
  let converted = 0;
  let pending = 0;
  let unparseable = 0;
  for (const table of ['broker_velocity_cap_hooks', 'broker_abuse_subject_hooks'] as const) {
    const rows = await db
      .prepare(
        `SELECT id, subject_value FROM ${table}
          WHERE subject_type = 'ip' AND active = 1
          AND (length(subject_value) != 64 OR subject_value GLOB '*[^0-9a-f]*')
          ORDER BY id ASC LIMIT ?`,
      )
      .bind(BACKFILL_BATCH_LIMIT)
      .all<{ id: number; subject_value: string }>()
      .catch(() => ({ results: [] as Array<{ id: number; subject_value: string }> }));
    for (const row of rows.results ?? []) {
      const normalized = normalizeNetworkIdentityIp(row.subject_value);
      if (!normalized) {
        unparseable += 1;
        continue;
      }
      const digests = await deriveStableNetworkIdentityDigest(secrets, normalized, 'ip');
      const digest = digests[0]?.digest;
      if (!digest) {
        unparseable += 1;
        continue;
      }
      const updated = await db
        .prepare(`UPDATE ${table} SET subject_value = ? WHERE id = ? AND subject_value = ?`)
        .bind(digest, row.id, row.subject_value)
        .run()
        .catch(() => null);
      if (Number(updated?.meta?.changes ?? 0) === 1) {
        converted += 1;
      }
    }
  }
  const remaining = await db
    .prepare(
      `SELECT subject_value FROM broker_velocity_cap_hooks
          WHERE subject_type = 'ip' AND active = 1
          AND (length(subject_value) != 64 OR subject_value GLOB '*[^0-9a-f]*')
        UNION ALL
        SELECT subject_value FROM broker_abuse_subject_hooks
          WHERE subject_type = 'ip' AND active = 1
          AND (length(subject_value) != 64 OR subject_value GLOB '*[^0-9a-f]*')
        LIMIT ?`,
    )
    .bind(BACKFILL_BATCH_LIMIT)
    .all<{ subject_value: string }>()
    .catch(() => ({ results: [] as Array<{ subject_value: string }> }));
  for (const row of remaining.results ?? []) {
    if (normalizeNetworkIdentityIp(row.subject_value)) {
      pending += 1;
    }
  }
  return { converted, pending, unparseable };
}

async function countPendingLegacyRequestEvents(db: D1Database, windowStartIso: string): Promise<number> {
  const row = await db
    .prepare(
      `SELECT COUNT(*) AS count FROM broker_request_events
        WHERE ip IS NOT NULL AND ip_digest IS NULL
          AND (ip_epoch IS NULL OR ip_epoch != ?)
          AND observed_at >= ?`,
    )
    .bind(UNPARSEABLE_IP_EPOCH_SENTINEL, windowStartIso)
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
async function countForbiddenLegacyValues(db: D1Database): Promise<number> {
  const queries = [
    `SELECT COUNT(*) AS count FROM broker_request_events WHERE ip IS NOT NULL`,
    `SELECT COUNT(*) AS count FROM broker_request_events WHERE ip_digest IS NULL AND ip_epoch = '${UNPARSEABLE_IP_EPOCH_SENTINEL}'`,
    `SELECT COUNT(*) AS count FROM broker_issue_success_events WHERE ip_hash IS NOT NULL OR ip_prefix_hash IS NOT NULL`,
    `SELECT COUNT(*) AS count FROM referral_rewards WHERE attempt_ip_hash IS NOT NULL`,
  ];
  let total = 0;
  for (const sql of queries) {
    const row = await db
      .prepare(sql)
      .first<{ count: number }>()
      .catch(() => null);
    if (!row) {
      return Number.POSITIVE_INFINITY;
    }
    total += Number(row.count ?? 0);
  }
  const rawHooks = await countActiveRawHooks(db);
  return total + rawHooks.count;
}

export async function finalizeNetworkIdentityMigration(db: D1Database, now: Date): Promise<boolean> {
  const nowIso = now.toISOString();
  try {
    await db
      .prepare(
        `DELETE FROM broker_request_events
          WHERE ip_digest IS NULL AND installation_id IS NULL AND ip IS NOT NULL`,
      )
      .run();
    await db
      .prepare(`UPDATE broker_request_events SET ip = NULL WHERE ip IS NOT NULL`)
      .run();
    await db
      .prepare(
        `UPDATE broker_request_events SET ip_epoch = NULL
          WHERE ip_digest IS NULL AND ip_epoch = ?`,
      )
      .bind(UNPARSEABLE_IP_EPOCH_SENTINEL)
      .run();
    await db
      .prepare(`UPDATE broker_issue_success_events SET ip_hash = NULL, ip_prefix_hash = NULL WHERE ip_hash IS NOT NULL OR ip_prefix_hash IS NOT NULL`)
      .run();
    await db
      .prepare(`UPDATE referral_rewards SET attempt_ip_hash = NULL WHERE attempt_ip_hash IS NOT NULL`)
      .run();
  } catch {
    return false;
  }
  if ((await countForbiddenLegacyValues(db)) !== 0) {
    return false;
  }
  const result = await db
    .prepare(`UPDATE broker_config SET value = ?, updated_at = ? WHERE key = 'network_identity_migration'`)
    .bind(JSON.stringify({ phase: 'keyed_only', purge_after: nowIso }), nowIso)
    .run()
    .catch(() => null);
  return Number(result?.meta.changes ?? 0) === 1;
}

