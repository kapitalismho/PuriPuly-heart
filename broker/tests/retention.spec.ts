import { existsSync, readFileSync } from 'node:fs';

import { describe, expect, it } from 'vitest';

import { applyAbuseMonitoringRetention } from '../src/abuse-monitoring';
import { applyTelemetryActiveDayRetention } from '../src/telemetry';
import { updateAbuseControls } from './test-support/abuse-controls';
import { createTestBrokerEnv } from './test-support/sqlite-d1';

const FIRST_MIGRATION = new URL(
  '../migrations/0000_define_broker_persistent_state.sql',
  import.meta.url,
);

describe('broker persistence retention model', () => {
  it('retains preflight-only none rows long enough to preserve in-flight challenges before cleanup', async () => {
    const contract = await import('../src/contract');

    expect(contract).toHaveProperty('BROKER_RETENTION_POLICY', {
      challengePreflight: {
        statuses: ['none'],
        entitlementRow: 'absent',
        challengeState: 'present',
        inactiveDays: 1,
        reference: 'max(installations.last_seen_at, installations.challenge_expires_at)',
        deleteFrom: 'installations',
        cascadesTo: [],
      },
      pendingRelease: {
        statuses: ['pending_release'],
        inactiveDays: 30,
        reference: 'installations.last_seen_at',
        deleteFrom: 'installations',
        cascadesTo: ['openrouter_entitlements'],
      },
      terminal: {
        statuses: ['expired', 'revoked'],
        inactiveDays: 90,
        reference: 'max(installations.last_seen_at, openrouter_entitlements.expires_at)',
        deleteFrom: 'installations',
        cascadesTo: ['openrouter_entitlements'],
      },
    });
  });

  it('keeps entitlement state as one in-place row per installation instead of append-only history', async () => {
    const contract = await import('../src/contract');

    expect(contract).toHaveProperty(
      'BROKER_PERSISTENCE_MODEL.tables.openrouterEntitlements.updateStrategy',
      'in-place',
    );
    expect(contract).toHaveProperty(
      'BROKER_PERSISTENCE_MODEL.tables.openrouterEntitlements.rowCardinality',
      'zero-or-one-row-per-installation',
    );
    expect(contract).toHaveProperty(
      'BROKER_PERSISTENCE_MODEL.tables.openrouterEntitlements.liveRemainingBudgetSource',
      'OpenRouter metadata',
    );
  });

  it('uses cascading delete from installations so retention cleanup removes entitlement rows too', () => {
    expect(existsSync(FIRST_MIGRATION)).toBe(true);
    if (!existsSync(FIRST_MIGRATION)) {
      return;
    }

    const migration = readFileSync(FIRST_MIGRATION, 'utf8');

    expect(migration).toContain(
      'installation_id TEXT PRIMARY KEY REFERENCES installations(installation_id) ON DELETE CASCADE',
    );
  });

  it('deletes expired request events, issue-success events, and runtime-audit rows using the configured retention windows', async () => {
    const env = createTestBrokerEnv();
    updateAbuseControls(env, (controls) => {
      controls.retention.requestEventsDays = 7;
      controls.retention.issueSuccessDays = 3;
      controls.retention.runtimeAuditDays = 10;
    });

    const insertInstallation = env.__db.prepare(
      `INSERT INTO installations (
          installation_id,
          device_public_key,
          hardware_hash,
          hardware_hash_salt_version,
          app_version,
          challenge,
          challenge_expires_at,
          challenge_salt_version,
          created_at,
          last_seen_at
        ) VALUES (?, ?, NULL, NULL, ?, NULL, NULL, NULL, ?, ?)`,
    );
    insertInstallation.run(
      'install-retention-old',
      'device-public-key-retention-old',
      '1.2.3',
      '2026-04-01T00:00:00.000Z',
      '2026-04-01T00:00:00.000Z',
    );
    insertInstallation.run(
      'install-retention-new',
      'device-public-key-retention-new',
      '1.2.3',
      '2026-04-17T00:00:00.000Z',
      '2026-04-17T00:00:00.000Z',
    );

    const insertRequestEvent = env.__db.prepare(
      `INSERT INTO broker_request_events (endpoint, ip, installation_id, observed_at)
        VALUES (?, ?, ?, ?)`,
    );
    insertRequestEvent.run(
      'POST /v1/trial/challenge',
      '203.0.113.1',
      'install-retention-old',
      '2026-04-01 00:00:00',
    );
    insertRequestEvent.run(
      'POST /v1/trial/challenge',
      '203.0.113.2',
      'install-retention-new',
      '2026-04-17 00:00:00',
    );

    const insertIssueSuccess = env.__db.prepare(
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
        ) VALUES ('discord', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    );
    insertIssueSuccess.run(
      'install-retention-old',
      'install-retention-old',
      'managed-retention-old',
      'ip-old',
      'prefix-old',
      64512,
      'US',
      'HTTP/2',
      'TLSv1.3',
      'TLS_AES_128_GCM_SHA256',
      'low',
      '2026-03-31 00:00:00',
    );
    insertIssueSuccess.run(
      'install-retention-new',
      'install-retention-new',
      'managed-retention-new',
      'ip-new',
      'prefix-new',
      64513,
      'US',
      'HTTP/2',
      'TLSv1.3',
      'TLS_AES_128_GCM_SHA256',
      'low',
      '2026-04-17 00:00:00',
    );

    const insertRuntimeAudit = env.__db.prepare(
      `INSERT INTO broker_abuse_runtime_audit (event_kind, reason, payload_json, created_at)
        VALUES (?, ?, ?, ?)`,
    );
    insertRuntimeAudit.run(
      'brake_transition',
      'global_threshold',
      '{"old":true}',
      '2026-04-01 00:00:00',
    );
    insertRuntimeAudit.run(
      'brake_transition',
      'global_threshold',
      '{"new":true}',
      '2026-04-17 00:00:00',
    );

    const result = await applyAbuseMonitoringRetention(
      env.BROKER_DB,
      new Date('2026-04-18T00:00:00.000Z'),
    );

    expect(result).toEqual({
      requestEventsDeleted: 1,
      issueSuccessDeleted: 1,
      runtimeAuditDeleted: 1,
    });

    expect(
      env.__db.prepare('SELECT COUNT(*) AS count FROM broker_request_events').get() as {
        count: number;
      },
    ).toEqual({ count: 1 });
    expect(
      env.__db
        .prepare('SELECT COUNT(*) AS count FROM broker_issue_success_events')
        .get() as { count: number },
    ).toEqual({ count: 1 });
    expect(
      env.__db
        .prepare('SELECT COUNT(*) AS count FROM broker_abuse_runtime_audit')
        .get() as { count: number },
    ).toEqual({ count: 1 });
  });

  it('deletes telemetry active-day rows older than 400 days and keeps rows at the cutoff or newer', async () => {
    const env = createTestBrokerEnv();

    insertTelemetryActiveDay(env, 'subject-old', '2025-03-13');
    insertTelemetryActiveDay(env, 'subject-cutoff', '2025-03-14');
    insertTelemetryActiveDay(env, 'subject-new', '2026-04-18');

    const result = await applyTelemetryActiveDayRetention(
      env.BROKER_DB,
      new Date('2026-04-18T22:00:00.000Z'),
    );

    expect(result).toEqual({
      deleted: 1,
      cutoffDateUtc: '2025-03-14',
    });
    expect(readTelemetryActiveDays(env)).toEqual([
      { subject_ref: validTelemetrySubjectRef('subject-cutoff'), active_date_utc: '2025-03-14' },
      { subject_ref: validTelemetrySubjectRef('subject-new'), active_date_utc: '2026-04-18' },
    ]);
    expect(JSON.stringify(result)).not.toContain('subject-old');
  });
});

function insertTelemetryActiveDay(
  env: ReturnType<typeof createTestBrokerEnv>,
  subjectRef: string,
  activeDateUtc: string,
): void {
  env.__db
    .prepare(
      `INSERT INTO telemetry_active_days (
          subject_ref,
          active_date_utc,
          first_received_at,
          last_received_at
        ) VALUES (?, ?, ?, ?)`,
    )
    .run(
      validTelemetrySubjectRef(subjectRef),
      activeDateUtc,
      `${activeDateUtc}T00:00:00.000Z`,
      `${activeDateUtc}T00:00:00.000Z`,
    );
}

function readTelemetryActiveDays(env: ReturnType<typeof createTestBrokerEnv>): Array<{
  subject_ref: string;
  active_date_utc: string;
}> {
  return env.__db
    .prepare(
      `SELECT subject_ref, active_date_utc
         FROM telemetry_active_days
        ORDER BY active_date_utc ASC, subject_ref ASC`,
    )
    .all() as Array<{ subject_ref: string; active_date_utc: string }>;
}

function validTelemetrySubjectRef(label: string): string {
  let hash = 0;
  for (const character of label) {
    hash = (hash * 31 + character.charCodeAt(0)) >>> 0;
  }

  return `ph-telemetry-subject-v1_${hash.toString(16).padStart(64, '0')}`;
}
