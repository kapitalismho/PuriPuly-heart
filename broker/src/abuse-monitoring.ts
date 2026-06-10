import {
  classifyAsn,
  getBrokerAbuseControlsConfig,
  getBrokerAbuseRuntimeState,
  persistBrokerAbuseRuntimeState,
  type BrokerAsnKind,
  type RequestNetworkMetadata,
} from './abuse-controls';
import type { BrokerBindings } from './contract';
import type { BrokerIssueSuccessSource } from './contract';
import { sendDiscordEmbed } from './discord-alerts';

type AlertLevel = 'warn1' | 'warn2' | 'warn3' | 'critical';
type BrakeReason = 'global_threshold' | 'asn_fast_path';

const INTERPRETATION_PACKET_SCHEMA_VERSION =
  'broker_abuse_interpretation_packet.v1';
const FIVE_MINUTES_MS = 5 * 60_000;
const FIFTEEN_MINUTES_MS = 15 * 60_000;
const SIXTY_MINUTES_MS = 60 * 60_000;
const SEVEN_DAYS_MS = 7 * 24 * 60 * 60_000;
const TIMELINE_BUCKET_COUNT = 12;
const RUNTIME_STATE_PERSIST_MAX_ATTEMPTS = 3;

interface IssueSuccessWindowRow {
  issue_source: BrokerIssueSuccessSource;
  installation_id: string | null;
  subject_ref: string;
  ip_hash: string | null;
  ip_prefix_hash: string | null;
  asn: number | null;
  country: string | null;
  http_protocol: string | null;
  tls_version: string | null;
  tls_cipher: string | null;
  risk_label: 'low' | 'medium' | 'high' | null;
  observed_at: string;
}

interface RequestEventCountRow {
  endpoint: string;
  count: number;
}

interface HistoricalHourlyCountRow {
  hour_bucket: string;
  count: number;
}

export interface InterpretationPacket {
  schema_version: string;
  alert_id: string;
  generated_at: string;
  window_start_60m: string;
  window_end_60m: string;
  trigger_context: {
    alert_level: AlertLevel | null;
    trigger_reason: BrakeReason | 'threshold_crossed' | 'state_observation';
    triggered_at: string;
    brake_state: boolean;
    brake_reason: 'global_threshold' | 'asn_fast_path' | 'manual' | null;
  };
  rolling_issue_counts: {
    issue_success: {
      last_5m: number;
      last_15m: number;
      last_60m: number;
      timeline_5m_buckets: Array<{
        window_start: string;
        window_end: string;
        count: number;
      }>;
    };
  };
  funnel_60m: {
    challenge_60m: number;
    verify_success_60m: number;
    verify_fail_60m: number;
    issue_success_60m: number;
    challenge_to_verify_rate: number | null;
    verify_to_issue_rate: number | null;
  };
  asn_context: {
    top_asns: Array<{
      asn: number;
      count: number;
      share: number;
      kind: BrokerAsnKind;
      display_name: string | null;
    }>;
    cloud_asn_share_60m: number;
  };
  spread_metrics: {
    unique_ip_hashes_60m: number;
    unique_ip_prefixes_60m: number;
    unique_installations_60m: number;
    top_ip_prefix_share: number;
    issues_per_installation_avg: number;
  };
  protocol_risk_signals: {
    http_protocol_mix: Record<string, number>;
    tls_version_mix: Record<string, number>;
    suspicious_proto_combo_count: number;
    risk_label_mix: Record<string, number>;
  };
  baseline_comparison: {
    hourly_issue_median_7d: number;
    hourly_issue_p95_7d: number;
    current_vs_median: number;
    current_vs_p95: number;
  };
  derived_flags: {
    cloud_asn_concentration: boolean;
    sudden_issue_spike: boolean;
    high_verify_to_issue_rate: boolean;
    browser_like_signal_weak: boolean;
  };
}

export interface ImmediateAbuseEvaluationResult {
  alertsToEmit: AlertLevel[];
  brakeTransition: null | { active: true; reason: BrakeReason };
  packet: InterpretationPacket;
}

interface CommonIssueSuccessInput {
  managedCredentialRef: string;
  observedAt: string;
  network: RequestNetworkMetadata;
}

export type RecordIssueSuccessInput = CommonIssueSuccessInput &
  (
    | {
        issueSource?: 'discord';
        installationId: string;
        subjectRef?: string;
      }
    | {
        issueSource: 'qq';
        installationId?: null;
        subjectRef: string;
      }
  );

export async function recordIssueSuccess(
  db: D1Database,
  input: RecordIssueSuccessInput,
): Promise<void> {
  const issueSubject = normalizeIssueSuccessSubject(input);
  await db
    .prepare(
      `INSERT INTO broker_issue_success_events (
          issue_source,
          installation_id,
          subject_ref,
          managed_credential_ref,
          ip_hash,
          ip_prefix_hash,
          asn,
          country,
          http_protocol,
          tls_version,
          tls_cipher,
          risk_label,
          observed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    )
    .bind(
      issueSubject.issueSource,
      issueSubject.installationId,
      issueSubject.subjectRef,
      input.managedCredentialRef,
      input.network.ipHash,
      input.network.ipPrefixHash,
      input.network.asn,
      input.network.country,
      input.network.httpProtocol,
      input.network.tlsVersion,
      input.network.tlsCipher,
      input.network.riskLabel,
      input.observedAt,
    )
    .run();
}

function normalizeIssueSuccessSubject(input: RecordIssueSuccessInput): {
  issueSource: BrokerIssueSuccessSource;
  installationId: string | null;
  subjectRef: string;
} {
  if (input.issueSource === 'qq') {
    const subjectRef = input.subjectRef;
    if (!subjectRef.trim()) {
      throw new Error('QQ issue-success subject_ref is required');
    }
    return {
      issueSource: 'qq',
      installationId: null,
      subjectRef,
    };
  }

  const installationId = input.installationId;
  const subjectRef = input.subjectRef ?? input.installationId;
  if (!installationId.trim() || !subjectRef.trim()) {
    throw new Error('Discord issue-success installation_id and subject_ref are required');
  }
  return {
    issueSource: 'discord',
    installationId,
    subjectRef,
  };
}

export async function evaluateImmediateAbuseState(
  db: D1Database,
  now: Date,
): Promise<ImmediateAbuseEvaluationResult> {
  const nowIso = now.toISOString();
  const windowStart60m = new Date(now.getTime() - SIXTY_MINUTES_MS).toISOString();
  const controls = await getBrokerAbuseControlsConfig(db);

  const [issueSuccessRows, requestEventRows, historicalHourlyRows] = await Promise.all([
    db
      .prepare(
        `SELECT issue_source, installation_id, subject_ref, ip_hash, ip_prefix_hash,
                asn, country, http_protocol, tls_version, tls_cipher, risk_label, observed_at
           FROM broker_issue_success_events
          WHERE observed_at >= ?
            AND observed_at <= ?
          ORDER BY observed_at ASC`,
      )
      .bind(windowStart60m, nowIso)
      .all<IssueSuccessWindowRow>(),
    db
      .prepare(
        `SELECT endpoint, COUNT(*) AS count
           FROM broker_request_events
          WHERE observed_at >= ?
            AND observed_at <= ?
          GROUP BY endpoint`,
      )
      .bind(windowStart60m, nowIso)
      .all<RequestEventCountRow>(),
    db
      .prepare(
        `SELECT substr(observed_at, 1, 13) AS hour_bucket, COUNT(*) AS count
           FROM broker_issue_success_events
          WHERE observed_at >= ?
            AND observed_at < ?
          GROUP BY substr(observed_at, 1, 13)
          ORDER BY hour_bucket ASC`,
      )
      .bind(new Date(now.getTime() - SEVEN_DAYS_MS).toISOString(), windowStart60m)
      .all<HistoricalHourlyCountRow>(),
  ]);

  const issueSuccessEvents = issueSuccessRows.results;
  const requestEventCounts = toCountMap(requestEventRows.results);
  const issueSuccess1h = issueSuccessEvents.length;
  const currentAlertLevel = resolveCurrentAlertLevel(
    issueSuccess1h,
    controls.immediateAlerts,
  );
  const topAsns = summarizeTopAsns(issueSuccessEvents, controls, issueSuccess1h);
  const topAsn = topAsns[0] ?? null;
  const topAsnSharePct = calculateSharePercentage(topAsn?.count ?? 0, issueSuccess1h);
  const topAsnKind = topAsn?.kind ?? 'other';

  const brakeReason = resolveBrakeReason({
    issueSuccess1h,
    topAsnSharePct,
    topAsnKind,
    controls,
  });
  const effectiveAlertLevel = resolveEffectiveAlertLevel(
    currentAlertLevel,
    brakeReason,
  );
  const runtimeUpdate = await applyImmediateRuntimeStateChanges({
    db,
    nowIso,
    issueSuccess1h,
    controls,
    brakeReason,
  });

  const packet = buildInterpretationPacket({
    now,
    nowIso,
    windowStart60m,
    currentAlertLevel: effectiveAlertLevel,
    alertsToEmit: runtimeUpdate.alertsToEmit,
    brakeTransition: runtimeUpdate.brakeTransition,
    controls,
    runtimeState: runtimeUpdate.runtimeState,
    issueSuccessEvents,
    requestEventCounts,
    historicalHourlyCounts: historicalHourlyRows.results.map(
      ({ count }: HistoricalHourlyCountRow) => Number(count),
    ),
    topAsns,
  });

  if (runtimeUpdate.alertsToEmit.length > 0) {
    await appendRuntimeAudit(db, {
      eventKind: 'immediate_alert_levels_emitted',
      reason: null,
      payload: {
        alerts_to_emit: runtimeUpdate.alertsToEmit,
        issue_success_1h: issueSuccess1h,
        alert_level: effectiveAlertLevel,
      },
      createdAt: nowIso,
    });
  }

  if (runtimeUpdate.brakeTransition) {
    await appendRuntimeAudit(db, {
      eventKind: 'brake_transition',
      reason: runtimeUpdate.brakeTransition.reason,
      payload: {
        brake: runtimeUpdate.runtimeState.brake,
        issue_success_1h: issueSuccess1h,
        top_asn: topAsn?.asn ?? null,
        top_asn_share_pct: topAsnSharePct,
      },
      createdAt: nowIso,
    });
  }

  return {
    alertsToEmit: runtimeUpdate.alertsToEmit,
    brakeTransition: runtimeUpdate.brakeTransition,
    packet,
  };
}

export async function deliverImmediateMonitoringSideEffects(
  env: Pick<
    BrokerBindings,
    'BROKER_DB' | 'DISCORD_IMMEDIATE_ALERT_WEBHOOK_URL'
  >,
  monitoringResult: ImmediateAbuseEvaluationResult,
): Promise<void> {
  if (
    monitoringResult.alertsToEmit.length === 0 &&
    monitoringResult.brakeTransition === null
  ) {
    return;
  }

  const topAsn = monitoringResult.packet.asn_context.top_asns[0] ?? null;
  try {
    await sendDiscordEmbed(env.DISCORD_IMMEDIATE_ALERT_WEBHOOK_URL, {
      title: 'Broker immediate abuse alert',
      color: monitoringResult.brakeTransition ? 0xed4245 : 0xfee75c,
      description: 'Immediate broker abuse threshold crossing.',
      jsonCodeBlock: {
        attachmentFilename: 'broker-immediate-abuse-alert.json',
        payload: monitoringResult.packet,
      },
      fields: [
        {
          name: 'Alert levels',
          value:
            monitoringResult.alertsToEmit.length === 0
              ? 'none'
              : monitoringResult.alertsToEmit.join(', '),
          inline: true,
        },
        {
          name: 'Brake',
          value: [
            `active=${monitoringResult.packet.trigger_context.brake_state}`,
            `reason=${monitoringResult.packet.trigger_context.brake_reason ?? 'none'}`,
          ].join('\n'),
          inline: true,
        },
        {
          name: 'Issue success 60m',
          value: String(
            monitoringResult.packet.rolling_issue_counts.issue_success.last_60m,
          ),
          inline: true,
        },
        {
          name: 'Top ASN',
          value: topAsn
            ? `AS${topAsn.asn}${topAsn.display_name ? ` (${topAsn.display_name})` : ''}: ${topAsn.share}% (${topAsn.kind})`
            : 'none observed',
        },
        {
          name: 'Cloud/VPS concentration',
          value: [
            `cloud_asn_share_60m=${monitoringResult.packet.asn_context.cloud_asn_share_60m}%`,
            topAsn
              ? `top_asn_kind=${topAsn.kind}`
              : 'top_asn_kind=none',
          ].join('\n'),
        },
      ],
    });
  } catch (error) {
    await appendRuntimeAudit(env.BROKER_DB, {
      eventKind: 'immediate_monitoring_side_effects_failed',
      reason:
        monitoringResult.brakeTransition?.reason ??
        monitoringResult.packet.trigger_context.trigger_reason,
      payload: {
        alerts_to_emit: monitoringResult.alertsToEmit,
        brake_transition: monitoringResult.brakeTransition,
        error_message: error instanceof Error ? error.message : String(error),
      },
      createdAt: monitoringResult.packet.generated_at,
    });
    return;
  }

  await appendRuntimeAudit(env.BROKER_DB, {
    eventKind: 'immediate_monitoring_side_effects_enqueued',
    reason:
      monitoringResult.brakeTransition?.reason ??
      monitoringResult.packet.trigger_context.trigger_reason,
    payload: {
      alerts_to_emit: monitoringResult.alertsToEmit,
      brake_transition: monitoringResult.brakeTransition,
      packet: monitoringResult.packet,
    },
    createdAt: monitoringResult.packet.generated_at,
  });
}

export async function applyAbuseMonitoringRetention(
  db: D1Database,
  now: Date,
): Promise<{
  requestEventsDeleted: number;
  issueSuccessDeleted: number;
  runtimeAuditDeleted: number;
}> {
  const controls = await getBrokerAbuseControlsConfig(db);

  const requestEventsDeleted = await deleteRowsOlderThan({
    db,
    table: 'broker_request_events',
    column: 'observed_at',
    cutoffIso: new Date(
      now.getTime() - controls.retention.requestEventsDays * 24 * 60 * 60_000,
    ).toISOString(),
  });
  const issueSuccessDeleted = await deleteRowsOlderThan({
    db,
    table: 'broker_issue_success_events',
    column: 'observed_at',
    cutoffIso: resolveIssueSuccessRetentionCutoff(now, controls.retention.issueSuccessDays),
  });
  const runtimeAuditDeleted = await deleteRowsOlderThan({
    db,
    table: 'broker_abuse_runtime_audit',
    column: 'created_at',
    cutoffIso: new Date(
      now.getTime() - controls.retention.runtimeAuditDays * 24 * 60 * 60_000,
    ).toISOString(),
  });

  return {
    requestEventsDeleted,
    issueSuccessDeleted,
    runtimeAuditDeleted,
  };
}

function resolveIssueSuccessRetentionCutoff(
  now: Date,
  issueSuccessDays: number,
): string {
  return new Date(now.getTime() - issueSuccessDays * 24 * 60 * 60_000).toISOString();
}

function buildInterpretationPacket(input: {
  now: Date;
  nowIso: string;
  windowStart60m: string;
  currentAlertLevel: AlertLevel | null;
  alertsToEmit: AlertLevel[];
  brakeTransition: null | { active: true; reason: BrakeReason };
  controls: Awaited<ReturnType<typeof getBrokerAbuseControlsConfig>>;
  runtimeState: Awaited<ReturnType<typeof getBrokerAbuseRuntimeState>>;
  issueSuccessEvents: IssueSuccessWindowRow[];
  requestEventCounts: Record<string, number>;
  historicalHourlyCounts: number[];
  topAsns: InterpretationPacket['asn_context']['top_asns'];
}): InterpretationPacket {
  const last5m = countEventsWithin(
    input.issueSuccessEvents,
    input.now.getTime() - FIVE_MINUTES_MS,
  );
  const last15m = countEventsWithin(
    input.issueSuccessEvents,
    input.now.getTime() - FIFTEEN_MINUTES_MS,
  );
  const issueSuccess1h = input.issueSuccessEvents.length;
  const challenge60m = input.requestEventCounts['POST /v1/trial/challenge'] ?? 0;
  const verifySuccess60m =
    input.requestEventCounts['POST /v1/trial/challenge/verify/success'] ?? 0;
  const verifyFail60m =
    input.requestEventCounts['POST /v1/trial/challenge/verify/fail'] ?? 0;
  const challengeToVerifyRate = safeRatio(verifySuccess60m, challenge60m);
  const verifyToIssueRate = safeRatio(issueSuccess1h, verifySuccess60m);
  const cloudAsnShare60m = calculateCloudAsnShare(
    input.issueSuccessEvents,
    input.controls,
  );
  const topAsnShare60m = calculateSharePercentage(
    input.topAsns[0]?.count ?? 0,
    issueSuccess1h,
  );
  const topIpPrefixShare = calculateTopIpPrefixShare(input.issueSuccessEvents);
  const historicalMedian = calculateMedian(input.historicalHourlyCounts);
  const historicalP95 = calculateNearestRankPercentile(
    input.historicalHourlyCounts,
    95,
  );
  const httpProtocolMix = countStringValues(
    input.issueSuccessEvents.map(({ http_protocol }) => http_protocol),
  );
  const tlsVersionMix = countStringValues(
    input.issueSuccessEvents.map(({ tls_version }) => tls_version),
  );
  const riskLabelMix = countStringValues(
    input.issueSuccessEvents.map(({ risk_label }) => risk_label),
  );
  const sourceQualifiedIssueSubjects = input.issueSuccessEvents.map(
    toSourceQualifiedIssueSubjectKey,
  );
  const suspiciousProtoComboCount = input.issueSuccessEvents.filter(
    ({ http_protocol, tls_version }) =>
      http_protocol === 'HTTP/1.1' &&
      tls_version !== null &&
      /^TLSv1(?:\.0|\.1|\.2)?$/u.test(tls_version),
  ).length;

  return {
    schema_version: INTERPRETATION_PACKET_SCHEMA_VERSION,
    alert_id: buildAlertId(input.nowIso, input.currentAlertLevel),
    generated_at: input.nowIso,
    window_start_60m: input.windowStart60m,
    window_end_60m: input.nowIso,
    trigger_context: {
      alert_level: input.currentAlertLevel,
      trigger_reason:
        input.brakeTransition?.reason ??
        (input.alertsToEmit.length > 0 ? 'threshold_crossed' : 'state_observation'),
      triggered_at: input.nowIso,
      brake_state: input.runtimeState.brake.active,
      brake_reason: input.runtimeState.brake.reason,
    },
    rolling_issue_counts: {
      issue_success: {
        last_5m: last5m,
        last_15m: last15m,
        last_60m: issueSuccess1h,
        timeline_5m_buckets: buildTimelineBuckets(
          input.issueSuccessEvents,
          new Date(input.windowStart60m),
          input.now,
        ),
      },
    },
    funnel_60m: {
      challenge_60m: challenge60m,
      verify_success_60m: verifySuccess60m,
      verify_fail_60m: verifyFail60m,
      issue_success_60m: issueSuccess1h,
      challenge_to_verify_rate: challengeToVerifyRate,
      verify_to_issue_rate: verifyToIssueRate,
    },
    asn_context: {
      top_asns: input.topAsns,
      cloud_asn_share_60m: cloudAsnShare60m,
    },
    spread_metrics: {
      unique_ip_hashes_60m: countUnique(
        input.issueSuccessEvents.map(({ ip_hash }) => ip_hash),
      ),
      unique_ip_prefixes_60m: countUnique(
        input.issueSuccessEvents.map(({ ip_prefix_hash }) => ip_prefix_hash),
      ),
      unique_installations_60m: countUnique(sourceQualifiedIssueSubjects),
      top_ip_prefix_share: topIpPrefixShare,
      issues_per_installation_avg: safeRatio(
        issueSuccess1h,
        countUnique(sourceQualifiedIssueSubjects),
      ) ?? 0,
    },
    protocol_risk_signals: {
      http_protocol_mix: httpProtocolMix,
      tls_version_mix: tlsVersionMix,
      suspicious_proto_combo_count: suspiciousProtoComboCount,
      risk_label_mix: riskLabelMix,
    },
    baseline_comparison: {
      hourly_issue_median_7d: historicalMedian,
      hourly_issue_p95_7d: historicalP95,
      current_vs_median: ratioOrAbsolute(issueSuccess1h, historicalMedian),
      current_vs_p95: ratioOrAbsolute(issueSuccess1h, historicalP95),
    },
    derived_flags: {
      cloud_asn_concentration:
        (input.topAsns[0]?.kind ?? 'other') === 'cloud_or_vps' &&
        topAsnShare60m >= input.controls.asnFastPath.minTopAsnSharePct,
      sudden_issue_spike: input.currentAlertLevel !== null,
      high_verify_to_issue_rate: (verifyToIssueRate ?? 0) >= 0.9,
      browser_like_signal_weak: suspiciousProtoComboCount > 0,
    },
  };
}

function toSourceQualifiedIssueSubjectKey(row: IssueSuccessWindowRow): string {
  if (row.issue_source === 'discord') {
    return `discord:${row.installation_id ?? row.subject_ref}`;
  }

  return `qq:${row.subject_ref}`;
}

function updateAlertLatches(
  state: Awaited<ReturnType<typeof getBrokerAbuseRuntimeState>>,
  issueSuccess1h: number,
  thresholds: Awaited<ReturnType<typeof getBrokerAbuseControlsConfig>>['immediateAlerts'],
  forceCritical: boolean,
): AlertLevel[] {
  const alertsToEmit: AlertLevel[] = [];

  for (const level of ['warn1', 'warn2', 'warn3', 'critical'] as const) {
    const threshold = thresholds[level];
    const shouldLatch = issueSuccess1h > threshold || (forceCritical && level === 'critical');
    if (shouldLatch && !state.alertLatches[level]) {
      alertsToEmit.push(level);
      state.alertLatches[level] = true;
      continue;
    }

    if (!shouldLatch && state.alertLatches[level]) {
      state.alertLatches[level] = false;
    }
  }

  return alertsToEmit;
}

export function resolveBrakeReason(input: {
  issueSuccess1h: number;
  topAsnSharePct: number;
  topAsnKind: BrokerAsnKind;
  controls: Awaited<ReturnType<typeof getBrokerAbuseControlsConfig>>;
}): BrakeReason | null {
  if (input.issueSuccess1h > input.controls.immediateAlerts.critical) {
    return 'global_threshold';
  }

  if (
    input.controls.asnFastPath.enabled &&
    input.issueSuccess1h > input.controls.asnFastPath.minIssueSuccess1h &&
    input.topAsnSharePct >= input.controls.asnFastPath.minTopAsnSharePct &&
    input.topAsnKind === 'cloud_or_vps'
  ) {
    return 'asn_fast_path';
  }

  return null;
}

function resolveEffectiveAlertLevel(
  currentAlertLevel: AlertLevel | null,
  brakeReason: BrakeReason | null,
): AlertLevel | null {
  if (brakeReason === 'asn_fast_path') {
    return 'critical';
  }

  return currentAlertLevel;
}

function maybeApplyBrakeTransition(
  state: Awaited<ReturnType<typeof getBrokerAbuseRuntimeState>>,
  nowIso: string,
  reason: BrakeReason | null,
): null | { active: true; reason: BrakeReason } {
  if (state.brake.active || reason === null) {
    return null;
  }

  state.brake.active = true;
  state.brake.reason = reason;
  state.brake.changedAt = nowIso;
  state.brake.changedBy = 'system';

  return {
    active: true,
    reason,
  };
}

async function applyImmediateRuntimeStateChanges(input: {
  db: D1Database;
  nowIso: string;
  issueSuccess1h: number;
  controls: Awaited<ReturnType<typeof getBrokerAbuseControlsConfig>>;
  brakeReason: BrakeReason | null;
}): Promise<{
  runtimeState: Awaited<ReturnType<typeof getBrokerAbuseRuntimeState>>;
  alertsToEmit: AlertLevel[];
  brakeTransition: null | { active: true; reason: BrakeReason };
}> {
  let runtimeState = await getBrokerAbuseRuntimeState(input.db);

  for (
    let attempt = 0;
    attempt < RUNTIME_STATE_PERSIST_MAX_ATTEMPTS;
    attempt += 1
  ) {
    const nextRuntimeState = structuredClone(runtimeState);
    const alertsToEmit = updateAlertLatches(
      nextRuntimeState,
      input.issueSuccess1h,
      input.controls.immediateAlerts,
      input.brakeReason === 'asn_fast_path',
    );
    const brakeTransition = maybeApplyBrakeTransition(
      nextRuntimeState,
      input.nowIso,
      input.brakeReason,
    );

    if (!hasRuntimeStateChanged(runtimeState, nextRuntimeState)) {
      return {
        runtimeState: nextRuntimeState,
        alertsToEmit,
        brakeTransition,
      };
    }

    if (
      await persistBrokerAbuseRuntimeState(
        input.db,
        runtimeState,
        nextRuntimeState,
      )
    ) {
      return {
        runtimeState: nextRuntimeState,
        alertsToEmit,
        brakeTransition,
      };
    }

    runtimeState = await getBrokerAbuseRuntimeState(input.db);
  }

  return {
    runtimeState,
    alertsToEmit: [],
    brakeTransition: null,
  };
}

function hasRuntimeStateChanged(
  before: Awaited<ReturnType<typeof getBrokerAbuseRuntimeState>>,
  after: Awaited<ReturnType<typeof getBrokerAbuseRuntimeState>>,
): boolean {
  return JSON.stringify(before) !== JSON.stringify(after);
}

async function appendRuntimeAudit(
  db: D1Database,
  input: {
    eventKind: string;
    reason: string | null;
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

async function deleteRowsOlderThan(input: {
  db: D1Database;
  table:
    | 'broker_request_events'
    | 'broker_issue_success_events'
    | 'broker_abuse_runtime_audit';
  column: 'observed_at' | 'created_at';
  cutoffIso: string;
}): Promise<number> {
  const result = await input.db
    .prepare(`DELETE FROM ${input.table} WHERE ${input.column} < ?`)
    .bind(input.cutoffIso)
    .run();

  return Number(result.meta.changes ?? 0);
}

function toCountMap(rows: RequestEventCountRow[]): Record<string, number> {
  return Object.fromEntries(
    rows.map(({ endpoint, count }) => [endpoint, Number(count)]),
  );
}

function summarizeTopAsns(
  rows: IssueSuccessWindowRow[],
  controls: Awaited<ReturnType<typeof getBrokerAbuseControlsConfig>>,
  totalCount: number,
): InterpretationPacket['asn_context']['top_asns'] {
  const counts = new Map<number, number>();
  for (const row of rows) {
    if (row.asn === null) {
      continue;
    }
    counts.set(row.asn, (counts.get(row.asn) ?? 0) + 1);
  }

  return [...counts.entries()]
    .sort(([leftAsn, leftCount], [rightAsn, rightCount]) => {
      if (rightCount !== leftCount) {
        return rightCount - leftCount;
      }
      return leftAsn - rightAsn;
    })
    .slice(0, 5)
    .map(([asn, count]) => {
      const classification = controls.asnClassifications.find((entry) => entry.asn === asn);
      return {
        asn,
        count,
        share: totalCount === 0 ? 0 : Math.round((count / totalCount) * 100),
        kind: classifyAsn(asn, controls),
        display_name: classification?.displayName ?? null,
      };
    });
}

function calculateCloudAsnShare(
  rows: IssueSuccessWindowRow[],
  controls: Awaited<ReturnType<typeof getBrokerAbuseControlsConfig>>,
): number {
  if (rows.length === 0) {
    return 0;
  }

  const cloudCount = rows.filter(
    ({ asn }) => classifyAsn(asn, controls) === 'cloud_or_vps',
  ).length;

  return Math.round((cloudCount / rows.length) * 100);
}

function calculateTopIpPrefixShare(rows: IssueSuccessWindowRow[]): number {
  if (rows.length === 0) {
    return 0;
  }

  const prefixCounts = new Map<string, number>();
  for (const { ip_prefix_hash } of rows) {
    if (!ip_prefix_hash) {
      continue;
    }
    prefixCounts.set(ip_prefix_hash, (prefixCounts.get(ip_prefix_hash) ?? 0) + 1);
  }

  const topPrefixCount = Math.max(0, ...prefixCounts.values());
  return Math.round((topPrefixCount / rows.length) * 100);
}

function countEventsWithin(
  rows: IssueSuccessWindowRow[],
  startTimestampMs: number,
): number {
  return rows.filter(
    ({ observed_at }) => new Date(observed_at).getTime() >= startTimestampMs,
  ).length;
}

function buildTimelineBuckets(
  rows: IssueSuccessWindowRow[],
  windowStart: Date,
  windowEnd: Date,
): InterpretationPacket['rolling_issue_counts']['issue_success']['timeline_5m_buckets'] {
  const buckets: InterpretationPacket['rolling_issue_counts']['issue_success']['timeline_5m_buckets'] = [];
  const finalBucketIndex = TIMELINE_BUCKET_COUNT - 1;

  for (let index = 0; index < TIMELINE_BUCKET_COUNT; index += 1) {
    const bucketStart = new Date(windowStart.getTime() + index * FIVE_MINUTES_MS);
    const bucketEnd =
      index === finalBucketIndex
        ? windowEnd
        : new Date(bucketStart.getTime() + FIVE_MINUTES_MS);
    const count = rows.filter(({ observed_at }) => {
      const observedAt = new Date(observed_at).getTime();
      return (
        observedAt >= bucketStart.getTime() &&
        (index === finalBucketIndex
          ? observedAt <= bucketEnd.getTime()
          : observedAt < bucketEnd.getTime())
      );
    }).length;

    buckets.push({
      window_start: bucketStart.toISOString(),
      window_end: bucketEnd.toISOString(),
      count,
    });
  }

  return buckets;
}

function countUnique(values: Array<string | null>): number {
  return new Set(values.filter((value): value is string => value !== null)).size;
}

function countStringValues(values: Array<string | null>): Record<string, number> {
  const counts = new Map<string, number>();

  for (const value of values) {
    if (!value) {
      continue;
    }
    counts.set(value, (counts.get(value) ?? 0) + 1);
  }

  return Object.fromEntries(
    [...counts.entries()].sort(([left], [right]) => left.localeCompare(right)),
  );
}

function calculateMedian(values: number[]): number {
  if (values.length === 0) {
    return 0;
  }

  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);

  if (sorted.length % 2 === 0) {
    return roundToTwo((sorted[middle - 1] + sorted[middle]) / 2);
  }

  return sorted[middle] ?? 0;
}

function calculateNearestRankPercentile(values: number[], percentile: number): number {
  if (values.length === 0) {
    return 0;
  }

  const sorted = [...values].sort((left, right) => left - right);
  const rank = Math.max(1, Math.ceil((percentile / 100) * sorted.length));
  return sorted[Math.min(rank - 1, sorted.length - 1)] ?? 0;
}

function ratioOrAbsolute(current: number, baseline: number): number {
  if (baseline <= 0) {
    return current;
  }

  return roundToTwo(current / baseline);
}

function safeRatio(numerator: number, denominator: number): number | null {
  if (denominator <= 0) {
    return null;
  }

  return roundToTwo(numerator / denominator);
}

function roundToTwo(value: number): number {
  return Math.round(value * 100) / 100;
}

export function resolveCurrentAlertLevel(
  issueSuccess1h: number,
  thresholds: Awaited<ReturnType<typeof getBrokerAbuseControlsConfig>>['immediateAlerts'],
): AlertLevel | null {
  if (issueSuccess1h > thresholds.critical) {
    return 'critical';
  }

  if (issueSuccess1h > thresholds.warn3) {
    return 'warn3';
  }

  if (issueSuccess1h > thresholds.warn2) {
    return 'warn2';
  }

  if (issueSuccess1h > thresholds.warn1) {
    return 'warn1';
  }

  return null;
}

export function calculateSharePercentage(count: number, totalCount: number): number {
  if (totalCount === 0) {
    return 0;
  }

  return (count / totalCount) * 100;
}

function buildAlertId(nowIso: string, alertLevel: AlertLevel | null): string {
  const timestamp = nowIso.replace(/[:.]/gu, '-');
  return `abuse-${alertLevel ?? 'state'}-${timestamp}-${crypto.randomUUID()}`;
}
