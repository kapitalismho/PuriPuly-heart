import { readFileSync } from 'node:fs';
import { DatabaseSync } from 'node:sqlite';

import { describe, expect, it } from 'vitest';

import { validatePublicInput } from '../src/public-input';
import {
  TEST_DEFAULT_ABUSE_CONTROLS,
  TEST_DEFAULT_ABUSE_RUNTIME_STATE,
} from './test-support/abuse-controls';
import {
  applyBrokerMigrations,
  readBrokerMigrationSql,
} from './test-support/migrations';

const FIRST_MIGRATION = new URL(
  '../migrations/0000_define_broker_persistent_state.sql',
  import.meta.url,
);
const SECOND_MIGRATION = new URL(
  '../migrations/0001_harden_installation_public_inputs.sql',
  import.meta.url,
);
const FOURTH_MIGRATION = new URL(
  '../migrations/0003_add_abuse_runtime_state_and_issue_success_events.sql',
  import.meta.url,
);
const DAILY_SUMMARY_V2_FINALIZER = new URL(
  '../deploy/finalize-daily-summary-v2.sql',
  import.meta.url,
);

const FIRST_MIGRATION_SQL = readFileSync(FIRST_MIGRATION, 'utf8');

function withMigratedDatabase(run: (db: DatabaseSync) => void): void {
  const db = new DatabaseSync(':memory:');
  try {
    applyBrokerMigrations(db);
    run(db);
  } finally {
    db.close();
  }
}

function withLegacyDatabase(run: (db: DatabaseSync) => void): void {
  const db = new DatabaseSync(':memory:');
  try {
    db.exec(FIRST_MIGRATION_SQL);
    run(db);
  } finally {
    db.close();
  }
}

describe('broker migration behavior', () => {
  it('seeds the expected broker_config rows and enforces supported keys with valid JSON', () => {
    withMigratedDatabase((db) => {
      const rows = db
        .prepare('SELECT key, value FROM broker_config ORDER BY key')
        .all() as Array<{ key: string; value: string }>;

      expect(rows.map(({ key }) => key)).toEqual([
        'abuse_controls',
        'abuse_runtime_state',
        'fingerprint_salt',
        'network_identity_migration',
        'qq_talk_together_pass',
      ]);
      expect(rows.map(({ value }) => JSON.parse(value))).toEqual([
        {
          ...TEST_DEFAULT_ABUSE_CONTROLS,
          telemetryTranslationSuccessDayIp: {
            endpoint: 'POST /v1/telemetry/translation-success-day',
            scope: 'ip',
            maxRequests: 60,
            windowMinutes: 15,
          },
          qqAuthStatusIp: {
            endpoint: 'POST /v1/auth/qq/status',
            scope: 'ip',
            maxRequests: 30,
            windowMinutes: 15,
          },
          immediateAlerts: {
            warn1: 10,
            warn2: 25,
            warn3: 50,
            critical: 70,
            ...TEST_DEFAULT_ABUSE_CONTROLS.immediateAlerts,
          },
          asnFastPath: {
            enabled: true,
            minIssueSuccess1h: 20,
            minTopAsnSharePct: 70,
          },
          asnClassifications: [],
          retention: {
            requestEventsDays: 30,
            ...TEST_DEFAULT_ABUSE_CONTROLS.retention,
          },
          dailyReport: {
            ...TEST_DEFAULT_ABUSE_CONTROLS.dailyReport,
            includeZeroActivity: false,
          },
        },
        {
          ...TEST_DEFAULT_ABUSE_RUNTIME_STATE,
          alertLatches: {
            warn1: false,
            warn2: false,
            warn3: false,
            critical: false,
            ...TEST_DEFAULT_ABUSE_RUNTIME_STATE.alertLatches,
          },
        },
        {
          current: {
            version: 1,
            salt: '__BOOTSTRAP_REQUIRED__',
          },
          previous: null,
          rotated_at: null,
        },
        {
          phase: 'keyed_only',
          purged_at: 'finalized',
        },
        {
          enabled: false,
          rewards_enabled: false,
          daily_warning_count: 30,
          daily_max_count: 50,
        },
      ]);

      const insertConfig = db.prepare(
        'INSERT INTO broker_config (key, value) VALUES (?, ?)',
      );
      const updateConfig = db.prepare(
        'UPDATE broker_config SET value = ? WHERE key = ?',
      );

      expect(() => insertConfig.run('unsupported_key', '{}')).toThrow(/constraint/i);
      expect(() => updateConfig.run('not-json', 'abuse_controls')).toThrow(
        /constraint|json/i,
      );
      expect(() => updateConfig.run('not-json', 'abuse_runtime_state')).toThrow(
        /constraint|json/i,
      );

      const tableNames = db
        .prepare("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")
        .all() as Array<{ name: string }>;

      expect(tableNames.map(({ name }) => name)).toContain('broker_issue_success_events');
      expect(tableNames.map(({ name }) => name)).toContain('broker_abuse_runtime_audit');

      const issueEventColumns = db
        .prepare("SELECT name FROM pragma_table_info('broker_issue_success_events') ORDER BY cid")
        .all() as Array<{ name: string }>;
      expect(issueEventColumns.map(({ name }) => name)).toEqual([
        'id',
        'issue_source',
        'installation_id',
        'subject_ref',
        'managed_credential_ref',
        'ip_digest',
        'ip_prefix_digest',
        'ip_key_version',
        'ip_epoch',
        'asn',
        'country',
        'http_protocol',
        'tls_version',
        'tls_cipher',
        'risk_label',
        'observed_at',
      ]);

      const runtimeAuditColumns = db
        .prepare("SELECT name FROM pragma_table_info('broker_abuse_runtime_audit') ORDER BY cid")
        .all() as Array<{ name: string }>;
      expect(runtimeAuditColumns.map(({ name }) => name)).toEqual([
        'id',
        'event_kind',
        'reason',
        'payload_json',
        'created_at',
      ]);
    });
  });

  it('enforces paired NULL rules on installation challenge and hardware hash fields', () => {
    withMigratedDatabase((db) => {
      const insertInstallation = db.prepare(
        `INSERT INTO installations (
          installation_id,
          device_public_key,
          hardware_hash,
          hardware_hash_salt_version,
          app_version,
          challenge,
          challenge_expires_at,
          challenge_salt_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
      );

      expect(() =>
        insertInstallation.run(
          'install-challenge-invalid',
          'device-public-key-challenge-invalid',
          null,
          null,
          '1.0.0',
          'challenge-token',
          null,
          null,
        ),
      ).toThrow(/constraint/i);

      expect(() =>
        insertInstallation.run(
          'install-hash-invalid',
          'device-public-key-hash-invalid',
          'hardware-hash',
          null,
          '1.0.0',
          null,
          null,
          null,
        ),
      ).toThrow(/constraint/i);

      expect(() =>
        insertInstallation.run(
          'install-valid',
          'device-public-key-valid',
          null,
          null,
          '1.0.0',
          'challenge-token',
          '2026-04-08T06:00:00Z',
          2,
        ),
      ).not.toThrow();
    });
  });

  it('enforces length bounds on persisted public installation inputs', () => {
    withMigratedDatabase((db) => {
      const insertInstallation = db.prepare(
        `INSERT INTO installations (
          installation_id,
          device_public_key,
          hardware_hash,
          hardware_hash_salt_version,
          app_version,
          challenge,
          challenge_expires_at,
          challenge_salt_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
      );

      expect(() =>
        insertInstallation.run(
          'i'.repeat(129),
          'device-public-key-too-long-installation-id',
          null,
          null,
          '1.0.0',
          'challenge-token',
          '2026-04-08T06:00:00Z',
          2,
        ),
      ).toThrow(/constraint/i);

      expect(() =>
        insertInstallation.run(
          'install-too-long-app-version',
          'device-public-key-too-long-app-version',
          null,
          null,
          'v'.repeat(65),
          'challenge-token',
          '2026-04-08T06:00:00Z',
          2,
        ),
      ).toThrow(/constraint/i);

      expect(() =>
        insertInstallation.run(
          'install-too-long-hardware-hash',
          'device-public-key-too-long-hardware-hash',
          'h'.repeat(129),
          2,
          '1.0.0',
          'challenge-token',
          '2026-04-08T06:00:00Z',
          2,
        ),
      ).toThrow(/constraint/i);

      expect(() =>
        insertInstallation.run(
          'i'.repeat(128),
          'device-public-key-at-limit',
          'h'.repeat(128),
          2,
          'v'.repeat(64),
          'challenge-token',
          '2026-04-08T06:00:00Z',
          2,
        ),
      ).not.toThrow();
    });
  });

  it('rejects persisted installation text that contains control characters or newlines', () => {
    withMigratedDatabase((db) => {
      const insertInstallation = db.prepare(
        `INSERT INTO installations (
          installation_id,
          device_public_key,
          hardware_hash,
          hardware_hash_salt_version,
          app_version,
          challenge,
          challenge_expires_at,
          challenge_salt_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
      );

      expect(() =>
        insertInstallation.run(
          'install-newline\nvalue',
          'device-public-key-newline-installation-id',
          null,
          null,
          '1.0.0',
          'challenge-token',
          '2026-04-08T06:00:00Z',
          2,
        ),
      ).toThrow(/constraint/i);

      expect(() =>
        insertInstallation.run(
          'install-control-app-version',
          'device-public-key-control-app-version',
          null,
          null,
          '1.0.0\tbeta',
          'challenge-token',
          '2026-04-08T06:00:00Z',
          2,
        ),
      ).toThrow(/constraint/i);

      expect(() =>
        insertInstallation.run(
          'install-control-hardware-hash',
          'device-public-key-control-hardware-hash',
          'hardware-hash\rvalue',
          2,
          '1.0.0',
          'challenge-token',
          '2026-04-08T06:00:00Z',
          2,
        ),
      ).toThrow(/constraint/i);
    });
  });

  it('rejects persisted installation text that is blank or whitespace-only', () => {
    withMigratedDatabase((db) => {
      const insertInstallation = db.prepare(
        `INSERT INTO installations (
          installation_id,
          device_public_key,
          hardware_hash,
          hardware_hash_salt_version,
          app_version,
          challenge,
          challenge_expires_at,
          challenge_salt_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
      );

      expect(() =>
        insertInstallation.run(
          '   ',
          'device-public-key-whitespace-installation-id',
          null,
          null,
          '1.0.0',
          'challenge-token',
          '2026-04-08T06:00:00Z',
          2,
        ),
      ).toThrow(/constraint/i);

      expect(() =>
        insertInstallation.run(
          'install-whitespace-app-version',
          'device-public-key-whitespace-app-version',
          null,
          null,
          ' \u00A0 ',
          'challenge-token',
          '2026-04-08T06:00:00Z',
          2,
        ),
      ).toThrow(/constraint/i);

      expect(() =>
        insertInstallation.run(
          'install-whitespace-hardware-hash',
          'device-public-key-whitespace-hardware-hash',
          '   ',
          2,
          '1.0.0',
          'challenge-token',
          '2026-04-08T06:00:00Z',
          2,
        ),
      ).toThrow(/constraint/i);
    });
  });

  it('ships a follow-up migration that upgrades already-initialized installations to hardened text constraints', () => {
    expect(() => readFileSync(SECOND_MIGRATION, 'utf8')).not.toThrow();
  });

  it('ships a follow-up migration that adds abuse runtime-state storage and issue-success event tables', () => {
    expect(() => readFileSync(FOURTH_MIGRATION, 'utf8')).not.toThrow();
  });

  it('upgrades clean legacy installations and entitlements in place while hardening future writes', () => {
    withLegacyDatabase((db) => {
      db.prepare(
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
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      ).run(
        'legacy-installation',
        'legacy-device-public-key',
        null,
        null,
        '1.0.0',
        'legacy-challenge',
        '2026-04-08T06:05:00.000Z',
        7,
        '2026-04-08T06:00:00.000Z',
        '2026-04-08T06:00:00.000Z',
      );
      db.prepare(
        `INSERT INTO openrouter_entitlements (
          installation_id,
          status,
          budget_usd,
          managed_credential_ref,
          issued_at,
          expires_at,
          release_session_ref,
          release_token_hash,
          release_token_expires_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      ).run(
        'legacy-installation',
        'active',
        0.07,
        'managed-credential-legacy-installation',
        '2026-04-08T06:00:00.000Z',
        '2026-10-08T06:00:00.000Z',
        null,
        null,
        null,
      );

      db.exec(readBrokerMigrationSql('0001_harden_installation_public_inputs.sql'));

      const upgradedRow = db
        .prepare(
          `SELECT installation_id, device_public_key, app_version, challenge,
                  challenge_expires_at, challenge_salt_version
             FROM installations
            WHERE installation_id = ?`,
        )
        .get('legacy-installation') as Record<string, unknown>;

      expect(upgradedRow).toEqual({
        installation_id: 'legacy-installation',
        device_public_key: 'legacy-device-public-key',
        app_version: '1.0.0',
        challenge: 'legacy-challenge',
        challenge_expires_at: '2026-04-08T06:05:00.000Z',
        challenge_salt_version: 7,
      });

      const upgradedEntitlement = db
        .prepare(
          `SELECT installation_id, status, managed_credential_ref, issued_at, expires_at
             FROM openrouter_entitlements
            WHERE installation_id = ?`,
        )
        .get('legacy-installation') as Record<string, unknown>;

      expect(upgradedEntitlement).toEqual({
        installation_id: 'legacy-installation',
        status: 'active',
        managed_credential_ref: 'managed-credential-legacy-installation',
        issued_at: '2026-04-08T06:00:00.000Z',
        expires_at: '2026-10-08T06:00:00.000Z',
      });

      expect(() =>
        db.prepare(
          `INSERT INTO installations (
            installation_id,
            device_public_key,
            hardware_hash,
            hardware_hash_salt_version,
            app_version,
            challenge,
            challenge_expires_at,
            challenge_salt_version
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
        ).run(
          'legacy-installation\nnewline',
          'legacy-device-public-key-newline',
          null,
          null,
          '1.0.0',
          'legacy-challenge',
          '2026-04-08T06:05:00.000Z',
          7,
        ),
      ).toThrow(/constraint/i);
    });
  });

  it('adds verified entitlement snapshot columns through a forward migration for already-migrated databases', () => {
    const db = new DatabaseSync(':memory:');

    try {
      applyBrokerMigrations(db, {
        through: '0001_harden_installation_public_inputs.sql',
      });

      db.prepare(
        `INSERT INTO installations (
          installation_id,
          device_public_key,
          app_version
        ) VALUES (?, ?, ?)`,
      ).run('already-migrated-installation', 'already-migrated-device-key', '1.0.0');
      db.prepare(
        `INSERT INTO openrouter_entitlements (
          installation_id,
          status,
          budget_usd,
          managed_credential_ref,
          issued_at,
          expires_at,
          release_session_ref,
          release_token_hash,
          release_token_expires_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      ).run(
        'already-migrated-installation',
        'active',
        0.07,
        'managed-credential-already-migrated',
        '2026-04-08T06:00:00.000Z',
        '2026-10-08T06:00:00.000Z',
        null,
        null,
        null,
      );

      applyBrokerMigrations(db, {
        after: '0001_harden_installation_public_inputs.sql',
      });

      const columns = db
        .prepare("SELECT name FROM pragma_table_info('openrouter_entitlements') ORDER BY cid")
        .all() as Array<{ name: string }>;

      expect(columns.map((column) => column.name)).toContain('verified_hardware_hash');
      expect(columns.map((column) => column.name)).toContain(
        'verified_hardware_hash_salt_version',
      );

      const upgradedRow = db
        .prepare(
          `SELECT installation_id, status, managed_credential_ref,
                  verified_hardware_hash, verified_hardware_hash_salt_version
             FROM openrouter_entitlements
            WHERE installation_id = ?`,
        )
        .get('already-migrated-installation') as Record<string, unknown>;

      expect(upgradedRow).toEqual({
        installation_id: 'already-migrated-installation',
        status: 'active',
        managed_credential_ref: 'managed-credential-already-migrated',
        verified_hardware_hash: null,
        verified_hardware_hash_salt_version: null,
      });
    } finally {
      db.close();
    }
  });

  it('adds the abuse runtime-state row and issue-success storage through a forward migration for already-migrated databases', () => {
    const db = new DatabaseSync(':memory:');

    try {
      applyBrokerMigrations(db, {
        through: '0002_add_entitlement_verified_hardware_snapshot.sql',
      });

      db.prepare(
        'INSERT INTO installations (installation_id, device_public_key, app_version) VALUES (?, ?, ?)',
      ).run('runtime-state-installation', 'runtime-state-device-key', '1.0.0');

      applyBrokerMigrations(db, {
        after: '0002_add_entitlement_verified_hardware_snapshot.sql',
        through: '0003_add_abuse_runtime_state_and_issue_success_events.sql',
      });

      const configKeys = db
        .prepare('SELECT key FROM broker_config ORDER BY key')
        .all() as Array<{ key: string }>;
      expect(configKeys.map(({ key }) => key)).toEqual([
        'abuse_controls',
        'abuse_runtime_state',
        'fingerprint_salt',
      ]);

      const runtimeStateRow = db
        .prepare('SELECT value FROM broker_config WHERE key = ?')
        .get('abuse_runtime_state') as { value: string };
      expect(JSON.parse(runtimeStateRow.value)).toEqual({
        ...TEST_DEFAULT_ABUSE_RUNTIME_STATE,
        alertLatches: {
          warn1: false,
          warn2: false,
          warn3: false,
          critical: false,
        },
      });

      const issueEventColumns = db
        .prepare("SELECT name FROM pragma_table_info('broker_issue_success_events') ORDER BY cid")
        .all() as Array<{ name: string }>;
      expect(issueEventColumns.map(({ name }) => name)).toEqual([
        'id',
        'installation_id',
        'managed_credential_ref',
        'ip_hash',
        'ip_prefix_hash',
        'asn',
        'country',
        'http_protocol',
        'tls_version',
        'tls_cipher',
        'risk_label',
        'observed_at',
      ]);

      const runtimeAuditColumns = db
        .prepare("SELECT name FROM pragma_table_info('broker_abuse_runtime_audit') ORDER BY cid")
        .all() as Array<{ name: string }>;
      expect(runtimeAuditColumns.map(({ name }) => name)).toEqual([
        'id',
        'event_kind',
        'reason',
        'payload_json',
        'created_at',
      ]);
    } finally {
      db.close();
    }
  });

  it('preserves previously tuned abuse controls when 0003 adds the new policy fields', () => {
    const db = new DatabaseSync(':memory:');

    try {
      applyBrokerMigrations(db, {
        through: '0002_add_entitlement_verified_hardware_snapshot.sql',
      });

      const tunedLegacyControls = {
        trialChallenge: {
          endpoint: 'POST /v1/trial/challenge',
          scope: 'ip',
          maxRequests: 14,
          windowMinutes: 9,
        },
        trialChallengeVerify: {
          endpoint: 'POST /v1/trial/challenge/verify',
          scope: 'installation_id',
          maxRequests: 7,
          windowMinutes: 11,
        },
        openrouterIssue: {
          endpoint: 'POST /v1/providers/openrouter/issue',
          scope: 'installation_id',
          maxRequests: 4,
          windowMinutes: 21,
        },
        trialStatus: {
          endpoint: 'GET /v1/trial/status',
          scope: 'installation_id',
          maxRequests: 45,
          windowMinutes: 17,
        },
        newActiveEntitlementsPerDay: {
          endpoint: 'POST /v1/providers/openrouter/issue',
          scope: 'global',
          maxCount: 123,
          windowDays: 3,
        },
      };

      db.prepare(
        'UPDATE broker_config SET value = ?, updated_at = ? WHERE key = ?',
      ).run(
        JSON.stringify(tunedLegacyControls),
        '2026-04-08T06:00:00.000Z',
        'abuse_controls',
      );

      applyBrokerMigrations(db, {
        after: '0002_add_entitlement_verified_hardware_snapshot.sql',
        through: '0003_add_abuse_runtime_state_and_issue_success_events.sql',
      });

      const migratedRow = db
        .prepare('SELECT value FROM broker_config WHERE key = ?')
        .get('abuse_controls') as { value: string };
      expect(JSON.parse(migratedRow.value)).toMatchObject({
        immediateAlerts: {
          warn1: 10,
          warn2: 25,
          warn3: 50,
          critical: 70,
        },
        asnFastPath: {
          enabled: true,
          minIssueSuccess1h: 20,
          minTopAsnSharePct: 70,
        },
        asnClassifications: [],
        dailyReport: {
          enabled: true,
          hourUtc: 13,
          minuteUtc: 0,
          includeZeroActivity: false,
        },
        retention: {
          requestEventsDays: 30,
          issueSuccessDays: 30,
          runtimeAuditDays: 90,
        },
        ...tunedLegacyControls,
      });
    } finally {
      db.close();
    }
  });

  it('adds telemetry active-day storage and preserves tuned abuse controls in a forward migration', () => {
    const db = new DatabaseSync(':memory:');

    try {
      applyBrokerMigrations(db, {
        through: '0010_source_aware_issue_success_events.sql',
      });

      const rowBefore = db
        .prepare('SELECT value FROM broker_config WHERE key = ?')
        .get('abuse_controls') as { value: string };
      const tunedControls = JSON.parse(rowBefore.value) as Record<string, any>;
      tunedControls.trialChallenge.maxRequests = 14;
      tunedControls.qqAuthAssertIp.maxRequests = 9;
      delete tunedControls.telemetryTranslationSuccessDayIp;

      db.prepare('UPDATE broker_config SET value = ?, updated_at = ? WHERE key = ?').run(
        JSON.stringify(tunedControls),
        '2026-07-02T00:00:00.000Z',
        'abuse_controls',
      );

      applyBrokerMigrations(db, {
        after: '0010_source_aware_issue_success_events.sql',
        through: '0011_add_telemetry_active_days.sql',
      });

      const table = db
        .prepare("SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?")
        .get('telemetry_active_days') as { name: string };
      expect(table.name).toBe('telemetry_active_days');

      const migratedRow = db
        .prepare('SELECT value FROM broker_config WHERE key = ?')
        .get('abuse_controls') as { value: string };
      expect(JSON.parse(migratedRow.value)).toEqual({
        ...tunedControls,
        telemetryTranslationSuccessDayIp: {
          endpoint: 'POST /v1/telemetry/translation-success-day',
          scope: 'ip',
          maxRequests: 60,
          windowMinutes: 15,
        },
      });
    } finally {
      db.close();
    }
  });

  it('backfills durable telemetry subjects and normalizes the v2 daily schedule', () => {
    const db = new DatabaseSync(':memory:');

    try {
      applyBrokerMigrations(db, {
        through: '0012_add_managed_key_delivery_ack.sql',
      });
      const firstSubject = `ph-telemetry-subject-v1_${'a'.repeat(64)}`;
      const secondSubject = `ph-telemetry-subject-v1_${'b'.repeat(64)}`;
      const insertActiveDay = db.prepare(
        `INSERT INTO telemetry_active_days (
            subject_ref,
            active_date_utc,
            first_received_at,
            last_received_at
          ) VALUES (?, ?, ?, ?)`,
      );
      insertActiveDay.run(
        firstSubject,
        '2026-04-01',
        '2026-04-01T00:00:00.000Z',
        '2026-04-01T00:00:00.000Z',
      );
      insertActiveDay.run(
        firstSubject,
        '2026-04-19',
        '2026-04-19T00:00:00.000Z',
        '2026-04-19T00:00:00.000Z',
      );
      insertActiveDay.run(
        secondSubject,
        '2026-04-10',
        '2026-04-10T00:00:00.000Z',
        '2026-04-10T00:00:00.000Z',
      );
      const rowBefore = db
        .prepare('SELECT value FROM broker_config WHERE key = ?')
        .get('abuse_controls') as { value: string };
      const tunedControls = JSON.parse(rowBefore.value) as Record<string, any>;
      tunedControls.trialChallenge.maxRequests = 17;
      tunedControls.retention.issueSuccessDays = 1;
      tunedControls.dailyReport = {
        enabled: true,
        hourUtc: 9,
        minuteUtc: 30,
        includeZeroActivity: true,
      };
      db.prepare('UPDATE broker_config SET value = ?, updated_at = ? WHERE key = ?').run(
        JSON.stringify(tunedControls),
        '2026-04-20T00:00:00.000Z',
        'abuse_controls',
      );

      applyBrokerMigrations(db, {
        after: '0012_add_managed_key_delivery_ack.sql',
        through: '0013_add_telemetry_subjects_and_daily_summary_v2.sql',
      });

      expect(
        db
          .prepare(
            `SELECT subject_ref, first_active_date_utc, last_active_date_utc
               FROM telemetry_subjects
              ORDER BY subject_ref`,
          )
          .all(),
      ).toEqual([
        {
          subject_ref: firstSubject,
          first_active_date_utc: '2026-04-01',
          last_active_date_utc: '2026-04-19',
        },
        {
          subject_ref: secondSubject,
          first_active_date_utc: '2026-04-10',
          last_active_date_utc: '2026-04-10',
        },
      ]);
      expect(
        db.prepare('SELECT COUNT(*) AS count FROM telemetry_active_days').get(),
      ).toEqual({ count: 3 });
      const postMigrationSubject = `ph-telemetry-subject-v1_${'c'.repeat(64)}`;
      insertActiveDay.run(
        firstSubject,
        '2026-03-31',
        '2026-03-31T00:00:00.000Z',
        '2026-03-31T00:00:00.000Z',
      );
      insertActiveDay.run(
        postMigrationSubject,
        '2026-04-20',
        '2026-04-20T00:00:00.000Z',
        '2026-04-20T00:00:00.000Z',
      );
      expect(
        db
          .prepare(
            `SELECT subject_ref, first_active_date_utc, last_active_date_utc
               FROM telemetry_subjects
              WHERE subject_ref IN (?, ?)
              ORDER BY subject_ref`,
          )
          .all(firstSubject, postMigrationSubject),
      ).toEqual([
        {
          subject_ref: firstSubject,
          first_active_date_utc: '2026-03-31',
          last_active_date_utc: '2026-04-19',
        },
        {
          subject_ref: postMigrationSubject,
          first_active_date_utc: '2026-04-20',
          last_active_date_utc: '2026-04-20',
        },
      ]);
      expect(
        db.prepare('SELECT COUNT(*) AS count FROM broker_daily_summary_deliveries').get(),
      ).toEqual({ count: 0 });
      expect(() =>
        db
          .prepare(
            `INSERT INTO broker_daily_summary_deliveries (
                report_date_utc,
                status,
                lease_token,
                lease_expires_at,
                attempted_at,
                delivered_at
              ) VALUES (?, 'pending', ?, ?, ?, NULL)`,
          )
          .run(
            '2026-04-19',
            'not-a-valid-lease-token',
            '2026-04-20T00:15:00.000Z',
            '2026-04-20T00:00:00.000Z',
          ),
      ).toThrow();
      expect(() =>
        db
          .prepare(
            `INSERT INTO broker_daily_summary_deliveries (
                report_date_utc,
                status,
                lease_token,
                lease_expires_at,
                attempted_at,
                delivered_at
              ) VALUES (?, 'pending', ?, ?, ?, NULL)`,
          )
          .run(
            '2026-04-19',
            '00000000-0000-0000-0000-000000000000',
            '2026-04-20T00:15:00.001Z',
            '2026-04-20T00:00:00.000Z',
          ),
      ).toThrow();
      const migratedRow = db
        .prepare('SELECT value FROM broker_config WHERE key = ?')
        .get('abuse_controls') as { value: string };
      const migratedControls = JSON.parse(migratedRow.value) as Record<string, any>;
      expect(migratedControls.trialChallenge.maxRequests).toBe(17);
      expect(migratedControls.retention.issueSuccessDays).toBe(2);
      expect(migratedControls.dailyReport).toEqual({
        enabled: true,
        hourUtc: 0,
        minuteUtc: 5,
        includeZeroActivity: true,
      });
      db.exec(readFileSync(DAILY_SUMMARY_V2_FINALIZER, 'utf8'));
      const finalizedRow = db
        .prepare('SELECT value FROM broker_config WHERE key = ?')
        .get('abuse_controls') as { value: string };
      expect(JSON.parse(finalizedRow.value).dailyReport).toEqual({
        enabled: true,
        hourUtc: 0,
        minuteUtc: 5,
      });
    } finally {
      db.close();
    }
  });

  it('preserves a tuned daily issuance cap when 0004 adds Discord OAuth controls', () => {
    const db = new DatabaseSync(':memory:');

    try {
      applyBrokerMigrations(db, {
        through: '0003_add_abuse_runtime_state_and_issue_success_events.sql',
      });

      const rowBefore = db
        .prepare('SELECT value FROM broker_config WHERE key = ?')
        .get('abuse_controls') as { value: string };
      const tunedControls = JSON.parse(rowBefore.value) as typeof TEST_DEFAULT_ABUSE_CONTROLS;
      tunedControls.newActiveEntitlementsPerDay.maxCount = 123;
      tunedControls.newActiveEntitlementsPerDay.windowDays = 3;

      db.prepare(
        'UPDATE broker_config SET value = ?, updated_at = ? WHERE key = ?',
      ).run(
        JSON.stringify(tunedControls),
        '2026-04-30T06:00:00.000Z',
        'abuse_controls',
      );

      applyBrokerMigrations(db, {
        after: '0003_add_abuse_runtime_state_and_issue_success_events.sql',
        through: '0004_add_discord_oauth_managed_issue.sql',
      });

      const migratedRow = db
        .prepare('SELECT value FROM broker_config WHERE key = ?')
        .get('abuse_controls') as { value: string };
      const migratedControls = JSON.parse(migratedRow.value) as typeof TEST_DEFAULT_ABUSE_CONTROLS;

      expect(migratedControls.newActiveEntitlementsPerDay).toEqual({
        endpoint: 'POST /v1/providers/openrouter/discord/issue',
        scope: 'global',
        maxCount: 123,
        windowDays: 3,
      });
      expect(migratedControls.pendingDiscordOAuthSessions).toEqual({
        maxPerInstallation: 2,
        maxPerIp: 20,
        windowMinutes: 15,
      });
    } finally {
      db.close();
    }
  });

  it('fails hardening migration when existing installations already contain newly invalid public input', () => {
    withLegacyDatabase((db) => {
      db.prepare(
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
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      ).run(
        'legacy-installation\nid',
        'legacy-device-public-key-invalid-installation',
        '   ',
        7,
        ' \u00A0 ',
        'legacy-challenge',
        '2026-04-08T06:05:00.000Z',
        7,
        '2026-04-08T06:00:00.000Z',
        '2026-04-08T06:00:00.000Z',
      );

      expect(() =>
        db.exec(readBrokerMigrationSql('0001_harden_installation_public_inputs.sql')),
      ).toThrow(/constraint/i);
    });
  });

  it('keeps runtime and migrated SQL validation aligned for representative public input samples', () => {
    const cases = [
      {
        field: 'installation_id' as const,
        value: 'install-aligned-ok',
        accepted: true,
      },
      {
        field: 'installation_id' as const,
        value: 'install-aligned\nnewline',
        accepted: false,
      },
      {
        field: 'installation_id' as const,
        value: `install-aligned${String.fromCharCode(0x80)}c1`,
        accepted: false,
      },
      {
        field: 'installation_id' as const,
        value: ' \u00A0 ',
        accepted: false,
      },
      {
        field: 'installation_id' as const,
        value: '😀'.repeat(128),
        accepted: true,
      },
      {
        field: 'installation_id' as const,
        value: '😀'.repeat(129),
        accepted: false,
      },
      {
        field: 'app_version' as const,
        value: '2.0.0',
        accepted: true,
      },
      {
        field: 'app_version' as const,
        value: '2.0.0\tbeta',
        accepted: false,
      },
      {
        field: 'app_version' as const,
        value: '   ',
        accepted: false,
      },
      {
        field: 'hardware_hash' as const,
        value: 'hardware-hash-aligned',
        accepted: true,
      },
      {
        field: 'hardware_hash' as const,
        value: `hardware${String.fromCharCode(0x2028)}hash`,
        accepted: false,
      },
      {
        field: 'hardware_hash' as const,
        value: ' \u00A0 ',
        accepted: false,
      },
    ];

    for (const testCase of cases) {
      withMigratedDatabase((db) => {
        const runtimeAccepted = validatePublicInput(testCase.field, testCase.value) === null;
        const insertInstallation = db.prepare(
          `INSERT INTO installations (
            installation_id,
            device_public_key,
            hardware_hash,
            hardware_hash_salt_version,
            app_version,
            challenge,
            challenge_expires_at,
            challenge_salt_version
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
        );

        let sqlAccepted = true;
        try {
          insertInstallation.run(
            testCase.field === 'installation_id' ? testCase.value : 'install-aligned',
            `device-${testCase.field}`,
            testCase.field === 'hardware_hash' ? testCase.value : null,
            testCase.field === 'hardware_hash' ? 7 : null,
            testCase.field === 'app_version' ? testCase.value : '1.0.0',
            'aligned-challenge',
            '2026-04-08T06:05:00.000Z',
            7,
          );
        } catch {
          sqlAccepted = false;
        }

        expect(runtimeAccepted).toBe(testCase.accepted);
        expect(sqlAccepted).toBe(testCase.accepted);
      });
    }
  });

  it('enforces release-session all-or-none fields, unique release token hashes, and cascades entitlement deletion', () => {
    withMigratedDatabase((db) => {
      db.prepare(
        'INSERT INTO installations (installation_id, device_public_key, app_version) VALUES (?, ?, ?)',
      ).run('install-a', 'device-public-key-a', '1.0.0');
      db.prepare(
        'INSERT INTO installations (installation_id, device_public_key, app_version) VALUES (?, ?, ?)',
      ).run('install-b', 'device-public-key-b', '1.0.0');

      const insertEntitlement = db.prepare(
        `INSERT INTO openrouter_entitlements (
          installation_id,
          status,
          budget_usd,
          managed_credential_ref,
          issued_at,
          expires_at,
          release_session_ref,
          release_token_hash,
          release_token_expires_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      );

      expect(() =>
        insertEntitlement.run(
          'install-a',
          'pending_release',
          0.07,
          null,
          null,
          null,
          'release-session-a',
          null,
          null,
        ),
      ).toThrow(/constraint/i);

      insertEntitlement.run(
        'install-a',
        'pending_release',
        0.07,
        null,
        null,
        null,
        'release-session-a',
        'token-hash-1',
        '2026-04-08T06:15:00Z',
      );

      expect(() =>
        insertEntitlement.run(
          'install-b',
          'pending_release',
          0.07,
          null,
          null,
          null,
          'release-session-b',
          'token-hash-1',
          '2026-04-08T06:15:00Z',
        ),
      ).toThrow(/unique|constraint/i);

      db.prepare('DELETE FROM installations WHERE installation_id = ?').run(
        'install-a',
      );

      const entitlementCount = db
        .prepare('SELECT COUNT(*) AS count FROM openrouter_entitlements')
        .get() as { count: number };

      expect(entitlementCount.count).toBe(0);
    });
  });

  it('adds QQ managed entitlements without rewriting existing QQ assertion evidence', () => {
    const db = new DatabaseSync(':memory:');

    try {
      applyBrokerMigrations(db, { through: '0008_add_qq_auth_assertions.sql' });
      db.prepare(
        `INSERT INTO qq_auth_assertions (
          qq_subject_ref,
          credential_hash,
          asserted_at,
          received_at,
          status
        ) VALUES (?, ?, ?, ?, 'verified')`,
      ).run(
        'ph-qq-subject-v1_existing-assertion-only',
        'sha256-base64url-v1_existing-credential-hash',
        '2026-06-05T12:03:00.000Z',
        '2026-06-05T12:04:00.000Z',
      );

      applyBrokerMigrations(db, { after: '0008_add_qq_auth_assertions.sql' });

      const assertionRow = db
        .prepare('SELECT * FROM qq_auth_assertions WHERE qq_subject_ref = ?')
        .get('ph-qq-subject-v1_existing-assertion-only') as Record<string, unknown>;
      expect(assertionRow).toEqual({
        qq_subject_ref: 'ph-qq-subject-v1_existing-assertion-only',
        credential_hash: 'sha256-base64url-v1_existing-credential-hash',
        asserted_at: '2026-06-05T12:03:00.000Z',
        received_at: '2026-06-05T12:04:00.000Z',
        status: 'verified',
      });

      const entitlementCount = db
        .prepare('SELECT COUNT(*) AS count FROM qq_managed_entitlements')
        .get() as { count: number };
      expect(entitlementCount.count).toBe(0);

      expect(() =>
        db.prepare(
          `INSERT INTO qq_managed_entitlements (
            qq_subject_ref,
            status,
            issue_ref,
            budget_usd,
            reserved_at
          ) VALUES (?, 'issuing', ?, ?, ?)`,
        ).run(
          'ph-qq-subject-v1_existing-assertion-only',
          'qq-issue-first-production-attempt',
          0.07,
          '2026-06-10T10:00:00.000Z',
        ),
      ).not.toThrow();
    } finally {
      db.close();
    }
  });

  it('migrates issue-success events to source-aware subjects while preserving Discord evidence and allowing QQ rows', () => {
    const db = new DatabaseSync(':memory:');

    try {
      applyBrokerMigrations(db, { through: '0009_add_qq_managed_entitlements.sql' });
      db.prepare(
        `INSERT INTO installations (
          installation_id,
          device_public_key,
          app_version,
          created_at,
          last_seen_at
        ) VALUES (?, ?, ?, ?, ?)`,
      ).run(
        'install-source-aware-discord-existing',
        'device-key-source-aware-discord-existing',
        '1.2.3',
        '2026-06-10T09:59:00.000Z',
        '2026-06-10T09:59:00.000Z',
      );
      db.prepare(
        `INSERT INTO broker_issue_success_events (
          id,
          installation_id,
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
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      ).run(
        42,
        'install-source-aware-discord-existing',
        'managed-source-aware-discord-existing',
        'ip-source-aware-discord-existing',
        'prefix-source-aware-discord-existing',
        64512,
        'US',
        'HTTP/2',
        'TLSv1.3',
        'TLS_AES_128_GCM_SHA256',
        'low',
        '2026-06-10T10:00:00.000Z',
      );

      applyBrokerMigrations(db, { after: '0009_add_qq_managed_entitlements.sql' });

      const migratedDiscordRow = db
        .prepare(
          `SELECT id,
                  issue_source,
                  installation_id,
                  subject_ref,
                  managed_credential_ref,
                  observed_at
             FROM broker_issue_success_events
            WHERE installation_id = ?`,
        )
        .get('install-source-aware-discord-existing') as Record<string, unknown>;
      expect(migratedDiscordRow).toEqual({
        id: 42,
        issue_source: 'discord',
        installation_id: 'install-source-aware-discord-existing',
        subject_ref: 'install-source-aware-discord-existing',
        managed_credential_ref: 'managed-source-aware-discord-existing',
        observed_at: '2026-06-10T10:00:00.000Z',
      });

      expect(() =>
        db.prepare(
          `INSERT INTO broker_issue_success_events (
            issue_source,
            installation_id,
            subject_ref,
            managed_credential_ref,
            ip_digest,
            ip_prefix_digest,
            ip_key_version,
            ip_epoch,
            asn,
            country,
            http_protocol,
            tls_version,
            tls_cipher,
            risk_label,
            observed_at
          ) VALUES ('qq', NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        ).run(
          'ph-qq-subject-v1_source-aware-monitoring-subject',
          'managed-source-aware-qq-existing',
          'a'.repeat(64),
          'b'.repeat(64),
          1,
          '2026-06-10',
          64513,
          'CN',
          'HTTP/2',
          'TLSv1.3',
          'TLS_AES_128_GCM_SHA256',
          'low',
          '2026-06-10T10:05:00.000Z',
        ),
      ).not.toThrow();

      const qqRow = db
        .prepare(
          `SELECT issue_source,
                  installation_id,
                  subject_ref,
                  managed_credential_ref,
                  observed_at
             FROM broker_issue_success_events
            WHERE issue_source = 'qq'
              AND subject_ref = ?
              AND observed_at >= ?`,
        )
        .get(
          'ph-qq-subject-v1_source-aware-monitoring-subject',
          '2026-06-10T10:00:00.000Z',
        ) as Record<string, unknown>;
      expect(qqRow).toEqual({
        issue_source: 'qq',
        installation_id: null,
        subject_ref: 'ph-qq-subject-v1_source-aware-monitoring-subject',
        managed_credential_ref: 'managed-source-aware-qq-existing',
        observed_at: '2026-06-10T10:05:00.000Z',
      });

      const sourceSubjectIndexColumns = db
        .prepare("SELECT name FROM pragma_index_info('idx_broker_issue_success_events_source_subject_time') ORDER BY seqno")
        .all() as Array<{ name: string }>;
      expect(sourceSubjectIndexColumns.map(({ name }) => name)).toEqual([
        'issue_source',
        'subject_ref',
        'observed_at',
      ]);

      expect(() =>
        db.prepare(
          `INSERT INTO broker_issue_success_events (
            issue_source,
            installation_id,
            subject_ref,
            managed_credential_ref,
            observed_at
          ) VALUES ('qq', NULL, 'ph-qq-subject-v1_', ?, ?)`,
        ).run(
          'managed-source-aware-qq-prefix-only-rejected',
          '2026-06-10T10:06:00.000Z',
        ),
      ).toThrow(/constraint/i);
    } finally {
      db.close();
    }
  });

  it('preserves issue-success AUTOINCREMENT sequence when deleted high-id rows existed before source-aware migration', () => {
    const db = new DatabaseSync(':memory:');

    try {
      applyBrokerMigrations(db, { through: '0009_add_qq_managed_entitlements.sql' });
      db.prepare(
        `INSERT INTO installations (
          installation_id,
          device_public_key,
          app_version,
          created_at,
          last_seen_at
        ) VALUES (?, ?, ?, ?, ?)`,
      ).run(
        'install-source-aware-sequence-existing',
        'device-key-source-aware-sequence-existing',
        '1.2.3',
        '2026-06-10T09:59:00.000Z',
        '2026-06-10T09:59:00.000Z',
      );
      const insertLegacyIssueSuccess = db.prepare(
        `INSERT INTO broker_issue_success_events (
          id,
          installation_id,
          managed_credential_ref,
          observed_at
        ) VALUES (?, ?, ?, ?)`,
      );
      insertLegacyIssueSuccess.run(
        42,
        'install-source-aware-sequence-existing',
        'managed-source-aware-sequence-existing-low',
        '2026-06-10T10:00:00.000Z',
      );
      insertLegacyIssueSuccess.run(
        100,
        'install-source-aware-sequence-existing',
        'managed-source-aware-sequence-existing-deleted-high',
        '2026-06-10T10:01:00.000Z',
      );
      db.prepare(
        'DELETE FROM broker_issue_success_events WHERE id = ?',
      ).run(100);

      applyBrokerMigrations(db, { after: '0009_add_qq_managed_entitlements.sql' });

      db.prepare(
        `INSERT INTO broker_issue_success_events (
          issue_source,
          installation_id,
          subject_ref,
          managed_credential_ref,
          observed_at
        ) VALUES ('discord', ?, ?, ?, ?)`,
      ).run(
        'install-source-aware-sequence-existing',
        'install-source-aware-sequence-existing',
        'managed-source-aware-sequence-new-after-migration',
        '2026-06-10T10:02:00.000Z',
      );

      const insertedRow = db
        .prepare(
          `SELECT id
             FROM broker_issue_success_events
            WHERE managed_credential_ref = ?`,
        )
        .get('managed-source-aware-sequence-new-after-migration') as { id: number };
      const sequenceRow = db
        .prepare("SELECT seq FROM sqlite_sequence WHERE name = 'broker_issue_success_events'")
        .get() as { seq: number };

      expect(insertedRow.id).toBe(101);
      expect(sequenceRow.seq).toBe(101);
    } finally {
      db.close();
    }
  });

  it('adds warning, brake, and enforcement-retention controls without breaking the previous Worker shape', () => {
    const db = new DatabaseSync(':memory:');

    try {
      applyBrokerMigrations(db, {
        through: '0013_add_telemetry_subjects_and_daily_summary_v2.sql',
      });
      const controlsRow = db
        .prepare("SELECT value FROM broker_config WHERE key = 'abuse_controls'")
        .get() as { value: string };
      const controls = JSON.parse(controlsRow.value) as Record<string, any>;
      controls.immediateAlerts.warn1 = 12;
      controls.immediateAlerts.critical = 80;
      db.prepare("UPDATE broker_config SET value = ? WHERE key = 'abuse_controls'").run(
        JSON.stringify(controls),
      );
      const runtimeRow = db
        .prepare("SELECT value FROM broker_config WHERE key = 'abuse_runtime_state'")
        .get() as { value: string };
      const runtime = JSON.parse(runtimeRow.value) as Record<string, any>;
      runtime.alertLatches.warn2 = true;
      db.prepare(
        "UPDATE broker_config SET value = ? WHERE key = 'abuse_runtime_state'",
      ).run(JSON.stringify(runtime));

      applyBrokerMigrations(db, {
        after: '0013_add_telemetry_subjects_and_daily_summary_v2.sql',
        through: '0014_simplify_abuse_incidents.sql',
      });

      const migratedControls = JSON.parse(
        (db
          .prepare("SELECT value FROM broker_config WHERE key = 'abuse_controls'")
          .get() as { value: string }).value,
      ) as Record<string, any>;
      const migratedRuntime = JSON.parse(
        (db
          .prepare("SELECT value FROM broker_config WHERE key = 'abuse_runtime_state'")
          .get() as { value: string }).value,
      ) as Record<string, any>;
      expect(migratedControls.immediateAlerts).toMatchObject({
        warning: 12,
        brake: 80,
        warn1: 12,
        critical: 80,
      });
      expect(migratedControls.retention.requestEventSafetyMarginDays).toBe(1);
      expect(migratedRuntime.alertLatches).toMatchObject({
        warning: true,
        warningObservedAt: null,
        warn2: true,
      });
      expect(
        db
          .prepare("SELECT COUNT(*) AS count FROM pragma_table_info('qq_managed_entitlements') WHERE name = 'child_key_creation_started_at'")
          .get(),
      ).toEqual({ count: 1 });
    } finally {
      db.close();
    }
  });

  it('adds minimal app active-day storage and defers legacy rate-limit removal until post-deploy finalization', () => {
    const db = new DatabaseSync(':memory:');

    try {
      applyBrokerMigrations(db, {
        through: '0014_simplify_abuse_incidents.sql',
      });
      const legacySubject = `ph-telemetry-subject-v1_${'d'.repeat(64)}`;
      db.prepare(
        `INSERT INTO telemetry_active_days (
          subject_ref,
          active_date_utc,
          first_received_at,
          last_received_at
        ) VALUES (?, ?, ?, ?)`,
      ).run(
        legacySubject,
        '2026-08-27',
        '2026-08-27T01:00:00.000Z',
        '2026-08-27T01:00:00.000Z',
      );
      const controlsRow = db
        .prepare("SELECT value FROM broker_config WHERE key = 'abuse_controls'")
        .get() as { value: string };
      const controls = JSON.parse(controlsRow.value) as Record<string, any>;
      expect(controls.telemetryTranslationSuccessDayIp).toBeDefined();
      controls.trialChallenge.maxRequests = 19;
      db.prepare("UPDATE broker_config SET value = ? WHERE key = 'abuse_controls'").run(
        JSON.stringify(controls),
      );

      applyBrokerMigrations(db, {
        after: '0014_simplify_abuse_incidents.sql',
        through: '0015_add_app_active_days.sql',
      });

      expect(
        db
          .prepare("SELECT name FROM pragma_table_info('app_active_days') ORDER BY cid")
          .all(),
      ).toEqual([{ name: 'subject_ref' }, { name: 'active_date_utc' }]);
      const migratedControls = JSON.parse(
        (db
          .prepare("SELECT value FROM broker_config WHERE key = 'abuse_controls'")
          .get() as { value: string }).value,
      ) as Record<string, any>;
      expect(migratedControls.telemetryTranslationSuccessDayIp).toBeDefined();
      expect(migratedControls.trialChallenge.maxRequests).toBe(19);
      db.exec(
        readFileSync(
          new URL('../deploy/finalize-app-active-day.sql', import.meta.url),
          'utf8',
        ),
      );
      const finalizedControls = JSON.parse(
        (db
          .prepare("SELECT value FROM broker_config WHERE key = 'abuse_controls'")
          .get() as { value: string }).value,
      ) as Record<string, any>;
      expect(finalizedControls.telemetryTranslationSuccessDayIp).toBeUndefined();
      expect(finalizedControls.trialChallenge.maxRequests).toBe(19);
      expect(
        db.prepare('SELECT COUNT(*) AS count FROM app_active_days').get(),
      ).toEqual({ count: 0 });
      expect(
        db.prepare('SELECT COUNT(*) AS count FROM telemetry_active_days').get(),
      ).toEqual({ count: 1 });
      expect(
        db.prepare('SELECT COUNT(*) AS count FROM telemetry_subjects').get(),
      ).toEqual({ count: 1 });
    } finally {
      db.close();
    }
  });
});
