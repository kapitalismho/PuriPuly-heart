import {
  getBrokerAbuseControlsConfig,
  getBrokerAbuseRuntimeState,
  persistBrokerAbuseRuntimeState,
  type RequestNetworkMetadata,
} from './abuse-controls';
import type { BrokerBindings, BrokerIssueSuccessSource } from './contract';
import { sendDiscordEmbed } from './discord-alerts';

type IssuanceIncidentKind = 'issuance_spike_warning' | 'automatic_issuance_brake';
type BrakeReason = 'global_threshold';
type CleanupIncidentPhase = 'managed_issue' | 'stale_delivery' | 'stale_reservation';

const ISSUANCE_INCIDENT_SCHEMA_VERSION = 'puripuly_issuance_incident.v2';
const CLEANUP_INCIDENT_SCHEMA_VERSION = 'puripuly_cleanup_incident.v1';
const SIXTY_MINUTES_MS = 60 * 60_000;
const ONE_DAY_MS = 24 * 60 * 60_000;
const RUNTIME_STATE_PERSIST_MAX_ATTEMPTS = 3;

export interface IssuanceIncidentPacket {
  schema_version: typeof ISSUANCE_INCIDENT_SCHEMA_VERSION;
  incident_id: string;
  incident_kind: IssuanceIncidentKind | null;
  observed_at: string;
  window_start: string;
  window_end: string;
  issue_success_60m: number;
  warning_threshold: number;
  brake_threshold: number;
  brake_active: boolean;
  brake_reason: 'global_threshold' | 'asn_fast_path' | 'manual' | null;
}

export interface ImmediateAbuseEvaluationResult {
  warningTransition: boolean;
  brakeTransition: null | { active: true; reason: BrakeReason };
  packet: IssuanceIncidentPacket;
}

interface CommonIssueSuccessInput {
  managedCredentialRef: string;
  observedAt: string;
  network: RequestNetworkMetadata;
  deliveryId?: string;
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
  await prepareIssueSuccessInsert(db, input).run();
}

export function prepareIssueSuccessInsert(
  db: D1Database,
  input: RecordIssueSuccessInput,
): D1PreparedStatement {
  const issueSubject = normalizeIssueSuccessSubject(input);
  const deliveryGuard = input.deliveryId
    ? `AND EXISTS (
         SELECT 1 FROM managed_key_deliveries
          WHERE delivery_id = ?
            AND managed_credential_ref = ?
            AND status IN ('pending', 'acknowledged')
       )`
    : '';
  return db
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
        )
        SELECT ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
         WHERE NOT EXISTS (
           SELECT 1 FROM broker_issue_success_events
            WHERE managed_credential_ref = ?
         )
         ${deliveryGuard}`,
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
      input.managedCredentialRef,
      ...(input.deliveryId
        ? [input.deliveryId, input.managedCredentialRef]
        : []),
    );
}

function normalizeIssueSuccessSubject(input: RecordIssueSuccessInput): {
  issueSource: BrokerIssueSuccessSource;
  installationId: string | null;
  subjectRef: string;
} {
  if (input.issueSource === 'qq') {
    if (!input.subjectRef.trim()) {
      throw new Error('QQ issue-success subject_ref is required');
    }
    return {
      issueSource: 'qq',
      installationId: null,
      subjectRef: input.subjectRef,
    };
  }

  const subjectRef = input.subjectRef ?? input.installationId;
  if (!input.installationId.trim() || !subjectRef.trim()) {
    throw new Error('Discord issue-success installation_id and subject_ref are required');
  }
  return {
    issueSource: 'discord',
    installationId: input.installationId,
    subjectRef,
  };
}

export async function evaluateImmediateAbuseState(
  db: D1Database,
  now: Date,
): Promise<ImmediateAbuseEvaluationResult> {
  const nowIso = now.toISOString();
  const windowStart = new Date(now.getTime() - SIXTY_MINUTES_MS).toISOString();
  const controls = await getBrokerAbuseControlsConfig(db);
  const row = await db
    .prepare(
      `SELECT COUNT(*) AS count
         FROM broker_issue_success_events
        WHERE julianday(observed_at) >= julianday(?)
          AND julianday(observed_at) <= julianday(?)`,
    )
    .bind(windowStart, nowIso)
    .first<{ count: number }>();
  const issueSuccess60m = Number(row?.count ?? 0);
  const runtimeUpdate = await applyImmediateRuntimeStateChanges({
    db,
    nowIso,
    issueSuccess60m,
    warningThreshold: controls.immediateAlerts.warning,
    brakeThreshold: controls.immediateAlerts.brake,
  });
  const incidentKind = runtimeUpdate.brakeTransition
    ? 'automatic_issuance_brake'
    : runtimeUpdate.warningTransition
      ? 'issuance_spike_warning'
      : null;
  const packet: IssuanceIncidentPacket = {
    schema_version: ISSUANCE_INCIDENT_SCHEMA_VERSION,
    incident_id: buildIncidentId(nowIso, incidentKind),
    incident_kind: incidentKind,
    observed_at: nowIso,
    window_start: windowStart,
    window_end: nowIso,
    issue_success_60m: issueSuccess60m,
    warning_threshold: controls.immediateAlerts.warning,
    brake_threshold: controls.immediateAlerts.brake,
    brake_active: runtimeUpdate.runtimeState.brake.active,
    brake_reason: runtimeUpdate.runtimeState.brake.reason,
  };

  if (incidentKind) {
    await appendRuntimeAudit(db, {
      eventKind: incidentKind,
      reason: runtimeUpdate.brakeTransition?.reason ?? 'threshold_crossed',
      payload: packet,
      createdAt: nowIso,
    });
  }

  return {
    warningTransition: runtimeUpdate.warningTransition,
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
  const incidentKind = monitoringResult.packet.incident_kind;
  if (!incidentKind) {
    return;
  }

  try {
    await sendDiscordEmbed(env.DISCORD_IMMEDIATE_ALERT_WEBHOOK_URL, {
      title:
        incidentKind === 'automatic_issuance_brake'
          ? 'Broker automatic issuance brake'
          : 'Broker issuance-spike warning',
      color: incidentKind === 'automatic_issuance_brake' ? 0xed4245 : 0xfee75c,
      description:
        incidentKind === 'automatic_issuance_brake'
          ? 'Managed-key issuance was automatically suspended after the brake threshold was crossed.'
          : 'Managed-key deliveries crossed the one-hour warning threshold.',
      jsonCodeBlock: {
        attachmentFilename: 'broker-issuance-incident.json',
        payload: monitoringResult.packet,
      },
      fields: [
        {
          name: 'Issue success 60m',
          value: String(monitoringResult.packet.issue_success_60m),
          inline: true,
        },
        {
          name: 'Threshold',
          value: String(
            incidentKind === 'automatic_issuance_brake'
              ? monitoringResult.packet.brake_threshold
              : monitoringResult.packet.warning_threshold,
          ),
          inline: true,
        },
        {
          name: 'Brake active',
          value: String(monitoringResult.packet.brake_active),
          inline: true,
        },
      ],
    });
  } catch (error) {
    await appendRuntimeAuditBestEffort(env.BROKER_DB, {
      eventKind: 'immediate_monitoring_side_effects_failed',
      reason: incidentKind,
      payload: {
        incident_id: monitoringResult.packet.incident_id,
        incident_kind: incidentKind,
        error_name: safeErrorName(error),
      },
      createdAt: monitoringResult.packet.observed_at,
    });
    return;
  }

  await appendRuntimeAuditBestEffort(env.BROKER_DB, {
    eventKind: 'immediate_monitoring_side_effects_delivered',
    reason: incidentKind,
    payload: {
      incident_id: monitoringResult.packet.incident_id,
      incident_kind: incidentKind,
    },
    createdAt: monitoringResult.packet.observed_at,
  });
}

export async function deliverManagedCleanupIncident(
  env: Pick<
    BrokerBindings,
    'BROKER_DB' | 'DISCORD_IMMEDIATE_ALERT_WEBHOOK_URL'
  >,
  input: {
    issueSource: BrokerIssueSuccessSource;
    managedCredentialRef: string | null;
    phase: CleanupIncidentPhase;
    cleanupRequiredRecorded: boolean;
    occurredAt: string;
  },
): Promise<void> {
  const payload = {
    schema_version: CLEANUP_INCIDENT_SCHEMA_VERSION,
    incident_id: buildIncidentId(input.occurredAt, 'managed_child_key_cleanup_failure'),
    incident_kind: 'managed_child_key_cleanup_failure',
    issue_source: input.issueSource,
    managed_credential_ref: input.managedCredentialRef,
    phase: input.phase,
    cleanup_required_recorded: input.cleanupRequiredRecorded,
    occurred_at: input.occurredAt,
  };

  try {
    await sendDiscordEmbed(env.DISCORD_IMMEDIATE_ALERT_WEBHOOK_URL, {
      title: 'Broker managed-key cleanup incident',
      color: 0xed4245,
      description: input.cleanupRequiredRecorded
        ? 'Managed child-key cleanup failed and the owner was marked cleanup_required.'
        : 'Managed child-key cleanup failed and cleanup_required state could not be confirmed.',
      jsonCodeBlock: {
        attachmentFilename: 'broker-cleanup-incident.json',
        payload,
      },
      fields: [
        {
          name: 'Source',
          value: input.issueSource,
          inline: true,
        },
        {
          name: 'Phase',
          value: input.phase,
          inline: true,
        },
        {
          name: 'cleanup_required',
          value: String(input.cleanupRequiredRecorded),
          inline: true,
        },
      ],
    });
  } catch (error) {
    await appendRuntimeAuditBestEffort(env.BROKER_DB, {
      eventKind: 'cleanup_incident_notification_failed',
      reason: input.phase,
      payload: {
        incident_id: payload.incident_id,
        issue_source: input.issueSource,
        managed_credential_ref: input.managedCredentialRef,
        cleanup_required_recorded: input.cleanupRequiredRecorded,
        error_name: safeErrorName(error),
      },
      createdAt: input.occurredAt,
    });
    return;
  }

  await appendRuntimeAuditBestEffort(env.BROKER_DB, {
    eventKind: 'cleanup_incident_notification_delivered',
    reason: input.phase,
    payload,
    createdAt: input.occurredAt,
  });
}

export async function applyAbuseMonitoringRetention(
  db: D1Database,
  now: Date,
  options: { preserveIssueSuccessFrom?: string } = {},
): Promise<{
  requestEventsDeleted: number;
  issueSuccessDeleted: number;
  runtimeAuditDeleted: number;
}> {
  const controls = await getBrokerAbuseControlsConfig(db);
  const longestEnforcementWindowMinutes = await resolveLongestRequestEventWindowMinutes(
    db,
    now,
    controls,
  );
  const requestEventRetentionMs =
    controls.retention.requestEventSafetyMarginDays * ONE_DAY_MS +
    longestEnforcementWindowMinutes * 60_000;
  const requestEventsDeleted = await deleteRowsOlderThan({
    db,
    table: 'broker_request_events',
    column: 'observed_at',
    cutoffIso: new Date(now.getTime() - requestEventRetentionMs).toISOString(),
  });
  const configuredIssueSuccessCutoff = new Date(
    now.getTime() - controls.retention.issueSuccessDays * ONE_DAY_MS,
  ).toISOString();
  const issueSuccessDeleted = await deleteIssueSuccessRowsOlderThan(
    db,
    options.preserveIssueSuccessFrom &&
      options.preserveIssueSuccessFrom < configuredIssueSuccessCutoff
      ? options.preserveIssueSuccessFrom
      : configuredIssueSuccessCutoff,
  );
  const runtimeAuditDeleted = await deleteRowsOlderThan({
    db,
    table: 'broker_abuse_runtime_audit',
    column: 'created_at',
    cutoffIso: new Date(
      now.getTime() - controls.retention.runtimeAuditDays * ONE_DAY_MS,
    ).toISOString(),
  });

  return {
    requestEventsDeleted,
    issueSuccessDeleted,
    runtimeAuditDeleted,
  };
}

async function applyImmediateRuntimeStateChanges(input: {
  db: D1Database;
  nowIso: string;
  issueSuccess60m: number;
  warningThreshold: number;
  brakeThreshold: number;
}): Promise<{
  runtimeState: Awaited<ReturnType<typeof getBrokerAbuseRuntimeState>>;
  warningTransition: boolean;
  brakeTransition: null | { active: true; reason: BrakeReason };
}> {
  for (let attempt = 0; attempt < RUNTIME_STATE_PERSIST_MAX_ATTEMPTS; attempt += 1) {
    const before = await getBrokerAbuseRuntimeState(input.db);
    const after = structuredClone(before);
    const warningActive = input.issueSuccess60m > input.warningThreshold;
    const priorWarningObservation = before.alertLatches.warningObservedAt;
    const warningSampleIsStale =
      priorWarningObservation !== null &&
      (priorWarningObservation > input.nowIso ||
        (priorWarningObservation === input.nowIso &&
          before.alertLatches.warning &&
          !warningActive));
    const warningTransition =
      !warningSampleIsStale && warningActive && !before.alertLatches.warning;
    if (!warningSampleIsStale) {
      after.alertLatches.warning = warningActive;
      after.alertLatches.warningObservedAt = input.nowIso;
    }
    let brakeTransition: null | { active: true; reason: BrakeReason } = null;

    if (input.issueSuccess60m > input.brakeThreshold && !before.brake.active) {
      after.brake.active = true;
      after.brake.reason = 'global_threshold';
      after.brake.changedAt = input.nowIso;
      after.brake.changedBy = 'system';
      brakeTransition = { active: true, reason: 'global_threshold' };
    }

    if (await persistBrokerAbuseRuntimeState(input.db, before, after)) {
      return {
        runtimeState: after,
        warningTransition,
        brakeTransition,
      };
    }
  }

  return {
    runtimeState: await getBrokerAbuseRuntimeState(input.db),
    warningTransition: false,
    brakeTransition: null,
  };
}

async function resolveLongestRequestEventWindowMinutes(
  db: D1Database,
  now: Date,
  controls: Awaited<ReturnType<typeof getBrokerAbuseControlsConfig>>,
): Promise<number> {
  const configuredWindows = [
    controls.trialChallenge.windowMinutes,
    controls.trialChallengeVerify.windowMinutes,
    controls.openrouterIssue.windowMinutes,
    controls.trialStatus.windowMinutes,
    controls.discordAuthStartIp.windowMinutes,
    controls.discordAuthStartInstallation.windowMinutes,
    controls.discordOpenrouterIssueIp.windowMinutes,
    controls.discordOpenrouterIssueInstallation.windowMinutes,
    controls.qqAuthAssertIp.windowMinutes,
    controls.qqAuthStatusIp.windowMinutes,
    controls.pendingDiscordOAuthSessions.windowMinutes,
  ];
  const hookRow = await db
    .prepare(
      `SELECT MAX(window_minutes) AS max_window_minutes
         FROM broker_velocity_cap_hooks
        WHERE active = 1
          AND (expires_at IS NULL OR julianday(expires_at) > julianday(?))`,
    )
    .bind(now.toISOString())
    .first<{ max_window_minutes: number | null }>();

  return Math.max(...configuredWindows, Number(hookRow?.max_window_minutes ?? 0));
}

async function deleteRowsOlderThan(input: {
  db: D1Database;
  table: 'broker_request_events' | 'broker_abuse_runtime_audit';
  column: 'observed_at' | 'created_at';
  cutoffIso: string;
}): Promise<number> {
  const result = await input.db
    .prepare(
      `DELETE FROM ${input.table}
        WHERE julianday(${input.column}) < julianday(?)`,
    )
    .bind(input.cutoffIso)
    .run();

  return Number(result.meta.changes ?? 0);
}

async function deleteIssueSuccessRowsOlderThan(
  db: D1Database,
  cutoffIso: string,
): Promise<number> {
  const result = await db
    .prepare(
      `DELETE FROM broker_issue_success_events
        WHERE julianday(observed_at) < julianday(?)`,
    )
    .bind(cutoffIso)
    .run();
  return Number(result.meta.changes ?? 0);
}

async function appendRuntimeAudit(
  db: D1Database,
  input: {
    eventKind: string;
    reason: string | null;
    payload: unknown;
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

async function appendRuntimeAuditBestEffort(
  db: D1Database,
  input: Parameters<typeof appendRuntimeAudit>[1],
): Promise<void> {
  try {
    await appendRuntimeAudit(db, input);
  } catch {
    return;
  }
}

function buildIncidentId(nowIso: string, kind: string | null): string {
  return `incident-${kind ?? 'observation'}-${Date.parse(nowIso)}-${crypto.randomUUID()}`;
}

function safeErrorName(error: unknown): string {
  return error instanceof Error && error.name ? error.name : 'Error';
}
