import {
  classifyAsn,
  getBrokerAbuseControlsConfig,
  getBrokerAbuseRuntimeState,
  persistBrokerAbuseRuntimeState,
} from './abuse-controls';
import {
  calculateSharePercentage,
  applyAbuseMonitoringRetention,
  resolveBrakeReason,
  resolveCurrentAlertLevel,
} from './abuse-monitoring';
import type { BrokerBindings } from './contract';
import {
  sendDailyReport,
  type DailyReportPayload,
} from './discord-alerts';
import type {
  BrokerAbuseControlsConfigValue,
  BrokerAbuseRuntimeStateValue,
} from './persistence';
import {
  applyReferralRewardRetention,
  reconcileStaleReferralRewards,
} from './referral';
import {
  applyTelemetryActiveDayRetention,
  getTelemetryUsageDailyMetrics,
} from './telemetry';

type AlertLevel = 'warn1' | 'warn2' | 'warn3' | 'critical';

const DAILY_HEARTBEAT_SCHEMA_VERSION = 'broker_daily_heartbeat.v1';
const DAILY_REPORT_WINDOW_MS = 24 * 60 * 60_000;
const ROLLING_ALERT_WINDOW_MS = 60 * 60_000;
const DAILY_REPORT_PERSIST_MAX_ATTEMPTS = 3;

interface CountRow {
  count: number;
}

interface EndpointCountRow {
  endpoint: string;
  count: number;
}

interface AsnCountRow {
  asn: number | null;
  count: number;
}

interface AuditRow {
  event_kind: string;
}

interface IssueSuccessSeverityRow {
  asn: number | null;
  observed_at: string;
}

interface ScheduledControllerLike {
  cron?: string;
  scheduledTime: number;
}

interface ExecutionContextLike {
  waitUntil?(promise: Promise<unknown>): void;
  passThroughOnException?(): void;
}

export async function handleScheduled(
  controller: ScheduledControllerLike,
  env: Pick<BrokerBindings, 'BROKER_DB' | 'DISCORD_DAILY_REPORT_WEBHOOK_URL'>,
  _ctx: ExecutionContextLike,
): Promise<void> {
  const now = new Date(controller.scheduledTime);

  await applyAbuseMonitoringRetention(env.BROKER_DB, now);
  await reconcileStaleReferralRewards(env.BROKER_DB, { nowIso: now.toISOString() });
  await applyReferralRewardRetention(env.BROKER_DB, now);

  const controls = await getBrokerAbuseControlsConfig(env.BROKER_DB);
  const runtimeState = await getBrokerAbuseRuntimeState(env.BROKER_DB);

  if (!shouldSendDailyReport(runtimeState, controls.dailyReport, now)) {
    await applyTelemetryActiveDayRetention(env.BROKER_DB, now);
    return;
  }

  await runDailyReport(env, now, controls);
  await applyTelemetryActiveDayRetention(env.BROKER_DB, now);
}

export function shouldSendDailyReport(
  runtimeState: BrokerAbuseRuntimeStateValue,
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

  return runtimeState.dailyReport.lastDeliveredDateUtc !== now.toISOString().slice(0, 10);
}

export async function runDailyReport(
  env: Pick<BrokerBindings, 'BROKER_DB' | 'DISCORD_DAILY_REPORT_WEBHOOK_URL'>,
  now: Date,
  controls?: BrokerAbuseControlsConfigValue,
): Promise<{
  ok: true;
  payload: DailyReportPayload;
}> {
  const effectiveControls =
    controls ?? (await getBrokerAbuseControlsConfig(env.BROKER_DB));
  const payload = await buildDailyHeartbeatPacket(
    env.BROKER_DB,
    now,
    effectiveControls,
  );

  await sendDailyReport(env.DISCORD_DAILY_REPORT_WEBHOOK_URL, payload);
  await markDailyReportDelivered(env.BROKER_DB, now);

  return {
    ok: true,
    payload,
  };
}

export async function buildDailyHeartbeatPacket(
  db: BrokerBindings['BROKER_DB'],
  now: Date,
  controls?: BrokerAbuseControlsConfigValue,
): Promise<DailyReportPayload> {
  const effectiveControls = controls ?? (await getBrokerAbuseControlsConfig(db));
  const nowIso = now.toISOString();
  const windowStart24h = new Date(now.getTime() - DAILY_REPORT_WINDOW_MS).toISOString();

  const [
    requestCountsResult,
    issueCountRow,
    asnCountResult,
    issueSeverityResult,
    auditResult,
    manualRevocationCountRow,
    translationUsage,
  ] =
    await Promise.all([
      db
        .prepare(
          `SELECT endpoint, COUNT(*) AS count
             FROM broker_request_events
            WHERE observed_at >= ?
              AND observed_at <= ?
            GROUP BY endpoint`,
        )
        .bind(windowStart24h, nowIso)
        .all<EndpointCountRow>(),
      db
        .prepare(
          `SELECT COUNT(*) AS count
             FROM broker_issue_success_events
            WHERE observed_at >= ?
              AND observed_at <= ?`,
        )
        .bind(windowStart24h, nowIso)
        .first<CountRow>(),
      db
        .prepare(
          `SELECT asn, COUNT(*) AS count
             FROM broker_issue_success_events
            WHERE observed_at >= ?
              AND observed_at <= ?
              AND asn IS NOT NULL
            GROUP BY asn
            ORDER BY count DESC, asn ASC`,
        )
        .bind(windowStart24h, nowIso)
        .all<AsnCountRow>(),
      db
        .prepare(
          `SELECT asn, observed_at
             FROM broker_issue_success_events
            WHERE observed_at >= ?
              AND observed_at <= ?
            ORDER BY observed_at ASC`,
        )
        .bind(
          new Date(now.getTime() - DAILY_REPORT_WINDOW_MS - ROLLING_ALERT_WINDOW_MS).toISOString(),
          nowIso,
        )
        .all<IssueSuccessSeverityRow>(),
      db
        .prepare(
          `SELECT event_kind
             FROM broker_abuse_runtime_audit
            WHERE created_at >= ?
              AND created_at <= ?
              AND event_kind IN ('immediate_alert_levels_emitted', 'brake_transition')`,
        )
        .bind(windowStart24h, nowIso)
        .all<AuditRow>(),
      db
        .prepare(
          `SELECT COUNT(*) AS count
             FROM broker_abuse_subject_hooks
            WHERE hook_kind = 'revocation'
              AND created_at >= ?
              AND created_at <= ?`,
        )
        .bind(windowStart24h, nowIso)
        .first<CountRow>(),
      getTelemetryUsageDailyMetrics(db, now),
    ]);

  const issueSuccess24h = Number(issueCountRow?.count ?? 0);
  const requestCounts = Object.fromEntries(
    requestCountsResult.results.map((row: EndpointCountRow) => [
      row.endpoint,
      Number(row.count),
    ]),
  ) as Record<string, number>;
  const allAsnCounts = asnCountResult.results.map((row: AsnCountRow) => {
    const count = Number(row.count);
    const classification = effectiveControls.asnClassifications.find(
      (entry) => entry.asn === row.asn,
    );

    return {
      asn: Number(row.asn),
      count,
      share: issueSuccess24h === 0 ? 0 : Math.round((count / issueSuccess24h) * 100),
      kind: classifyAsn(row.asn, effectiveControls),
      display_name: classification?.displayName ?? null,
    };
  });
  const topAsns: DailyReportPayload['summary']['top_asns'] = allAsnCounts.slice(0, 5);
  const cloudAsnIssueCount = allAsnCounts
    .filter(
      (entry: DailyReportPayload['summary']['top_asns'][number]) =>
        entry.kind === 'cloud_or_vps',
    )
    .reduce(
      (sum: number, entry: DailyReportPayload['summary']['top_asns'][number]) =>
        sum + entry.count,
      0,
    );
  return {
    schema_version: DAILY_HEARTBEAT_SCHEMA_VERSION,
    generated_at: nowIso,
    window_start_24h: windowStart24h,
    window_end_24h: nowIso,
    summary: {
      challenge_24h: requestCounts['POST /v1/trial/challenge'] ?? 0,
      verify_24h: requestCounts['POST /v1/trial/challenge/verify'] ?? 0,
      issue_success_24h: issueSuccess24h,
      highest_alert_level_24h: resolveHighestAlertLevelFromIssueEvents({
        rows: issueSeverityResult.results,
        now,
        controls: effectiveControls,
      }),
      brake_triggered_24h: auditResult.results.some(
        (row: AuditRow) => row.event_kind === 'brake_transition',
      ),
      top_asns: topAsns,
      cloud_asn_share_24h:
        issueSuccess24h === 0 ? 0 : Math.round((cloudAsnIssueCount / issueSuccess24h) * 100),
      manual_revocations_24h: Number(manualRevocationCountRow?.count ?? 0),
      translation_usage: translationUsage,
    },
  };
}

export async function markDailyReportDelivered(
  db: BrokerBindings['BROKER_DB'],
  now: Date,
): Promise<void> {
  const deliveredAt = now.toISOString();
  const deliveredDateUtc = deliveredAt.slice(0, 10);
  let runtimeState = await getBrokerAbuseRuntimeState(db);

  for (let attempt = 0; attempt < DAILY_REPORT_PERSIST_MAX_ATTEMPTS; attempt += 1) {
    if (
      runtimeState.dailyReport.lastDeliveredAt === deliveredAt &&
      runtimeState.dailyReport.lastDeliveredDateUtc === deliveredDateUtc
    ) {
      return;
    }

    const nextRuntimeState = structuredClone(runtimeState);
    nextRuntimeState.dailyReport = {
      lastDeliveredAt: deliveredAt,
      lastDeliveredDateUtc: deliveredDateUtc,
    };

    if (await persistBrokerAbuseRuntimeState(db, runtimeState, nextRuntimeState)) {
      return;
    }

    runtimeState = await getBrokerAbuseRuntimeState(db);
  }

  if (
    runtimeState.dailyReport.lastDeliveredAt === deliveredAt &&
    runtimeState.dailyReport.lastDeliveredDateUtc === deliveredDateUtc
  ) {
    return;
  }

  throw new Error(
    'failed to persist daily report delivery stamp after runtime-state write conflict',
  );
}

function resolveHighestAlertLevelFromIssueEvents(input: {
  rows: IssueSuccessSeverityRow[];
  now: Date;
  controls: BrokerAbuseControlsConfigValue;
}): AlertLevel | null {
  const windowStartMs = input.now.getTime() - DAILY_REPORT_WINDOW_MS;
  const windowEndMs = input.now.getTime();
  const priority: Record<AlertLevel, number> = {
    warn1: 1,
    warn2: 2,
    warn3: 3,
    critical: 4,
  };

  let highest: AlertLevel | null = null;
  const windowRows = input.rows
    .map((row) => ({
      ...row,
      observedAtMs: new Date(row.observed_at).getTime(),
    }))
    .sort((left, right) => left.observedAtMs - right.observedAtMs);
  const evaluationTimesMs = new Set<number>([windowStartMs]);

  for (const row of windowRows) {
    if (row.observedAtMs >= windowStartMs && row.observedAtMs <= windowEndMs) {
      evaluationTimesMs.add(row.observedAtMs);
    }

    const windowExitMs = row.observedAtMs + ROLLING_ALERT_WINDOW_MS + 1;
    if (windowExitMs >= windowStartMs && windowExitMs <= windowEndMs) {
      evaluationTimesMs.add(windowExitMs);
    }
  }

  const sortedEvaluationTimesMs = [...evaluationTimesMs].sort((left, right) => left - right);
  const asnCounts = new Map<number, number>();
  let enterIndex = 0;
  let exitIndex = 0;
  let issueSuccess1h = 0;

  for (const evaluationTimeMs of sortedEvaluationTimesMs) {
    while (
      enterIndex < windowRows.length &&
      windowRows[enterIndex]!.observedAtMs <= evaluationTimeMs
    ) {
      issueSuccess1h += 1;
      incrementAsnCount(asnCounts, windowRows[enterIndex]!.asn);
      enterIndex += 1;
    }

    const rollingWindowStartMs = evaluationTimeMs - ROLLING_ALERT_WINDOW_MS;
    while (
      exitIndex < enterIndex &&
      windowRows[exitIndex]!.observedAtMs < rollingWindowStartMs
    ) {
      issueSuccess1h -= 1;
      decrementAsnCount(asnCounts, windowRows[exitIndex]!.asn);
      exitIndex += 1;
    }

    const baseAlertLevel = resolveCurrentAlertLevel(
      issueSuccess1h,
      input.controls.immediateAlerts,
    );
    const topAsn = resolveTopAsn(asnCounts);
    const topAsnSharePct = calculateSharePercentage(
      topAsn?.count ?? 0,
      issueSuccess1h,
    );
    const effectiveAlertLevel =
      resolveBrakeReason({
        issueSuccess1h,
        topAsnSharePct,
        topAsnKind: classifyAsn(topAsn?.asn ?? null, input.controls),
        controls: input.controls,
      }) !== null
        ? 'critical'
        : baseAlertLevel;

    if (
      effectiveAlertLevel !== null &&
      (highest === null || priority[effectiveAlertLevel] > priority[highest])
    ) {
      highest = effectiveAlertLevel;
    }
  }

  return highest;
}

function incrementAsnCount(counts: Map<number, number>, asn: number | null): void {
  if (asn === null) {
    return;
  }

  counts.set(asn, (counts.get(asn) ?? 0) + 1);
}

function decrementAsnCount(counts: Map<number, number>, asn: number | null): void {
  if (asn === null) {
    return;
  }

  const nextCount = (counts.get(asn) ?? 0) - 1;
  if (nextCount <= 0) {
    counts.delete(asn);
    return;
  }

  counts.set(asn, nextCount);
}

function resolveTopAsn(
  counts: Map<number, number>,
): { asn: number; count: number } | null {
  let topAsn: { asn: number; count: number } | null = null;

  for (const [asn, count] of counts.entries()) {
    if (
      topAsn === null ||
      count > topAsn.count ||
      (count === topAsn.count && asn < topAsn.asn)
    ) {
      topAsn = { asn, count };
    }
  }

  return topAsn;
}
