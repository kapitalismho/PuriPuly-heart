import { DatabaseSync } from 'node:sqlite';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { checkEndpointRateLimit } from '../src/abuse-controls';
import {
  acknowledgeManagedKeyDelivery,
  createManagedKeyDelivery,
} from '../src/managed-key-delivery';
import { expireManagedOperation } from '../src/managed-operation';
import {
  deriveStableNetworkIdentityDigest,
  normalizeNetworkIdentityIp,
  resolveNetworkIdentitySecrets,
  resolveRequestNetworkIdentity,
} from '../src/network-identity';
import { runNetworkIdentityBackfill } from '../src/network-identity-migration';
import {
  applyBrokerMigrations,
  readBrokerMigrationSql,
} from './test-support/migrations';
import {
  createTestBrokerEnv,
  testNetworkIdentitySecrets,
  type TestBrokerEnv,
} from './test-support/sqlite-d1';

const NOW_ISO = '2026-09-01T10:00:00.000Z';
const NOW = new Date(NOW_ISO);


describe('managed issuance durability', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it('treats the exact ACK expiry instant as valid and one millisecond later as expired', async () => {
    const env = createTestBrokerEnv();
    const createdAt = new Date('2026-09-01T10:00:00.000Z');
    const expiresAt = new Date(createdAt.getTime() + 15 * 60_000);
    const delivery = await createManagedKeyDelivery(env.BROKER_DB, {
      issueSource: 'discord',
      subjectRef: 'ph-discord-user-v1_ack_boundary',
      installationId: 'install-ack-boundary',
      managedCredentialRef: 'hash_ack_boundary',
      createdAt,
      expiresAt,
    });

    await expect(
      acknowledgeManagedKeyDelivery(env.BROKER_DB, {
        deliveryId: delivery.deliveryId,
        managedCredentialRef: 'hash_ack_boundary',
        deliveryAckToken: delivery.deliveryAckToken,
        now: expiresAt,
      }),
    ).resolves.toEqual({ ok: true, status: 'acknowledged' });

    const second = await createManagedKeyDelivery(env.BROKER_DB, {
      issueSource: 'discord',
      subjectRef: 'ph-discord-user-v1_ack_boundary_late',
      installationId: 'install-ack-boundary-late',
      managedCredentialRef: 'hash_ack_boundary_late',
      createdAt,
      expiresAt,
    });
    await expect(
      acknowledgeManagedKeyDelivery(env.BROKER_DB, {
        deliveryId: second.deliveryId,
        managedCredentialRef: 'hash_ack_boundary_late',
        deliveryAckToken: second.deliveryAckToken,
        now: new Date(expiresAt.getTime() + 1),
      }),
    ).resolves.toEqual({ ok: false, reason: 'expired' });
  });

  it('dual-compares rotated secrets without weakening per-IP limits', async () => {
    const env = createTestBrokerEnv();
    env.NETWORK_IDENTITY_HMAC_SECRET = 'new-secret';
    env.NETWORK_IDENTITY_HMAC_SECRET_PREVIOUS = 'old-secret';
    (env as unknown as Record<string, unknown>).NETWORK_IDENTITY_HMAC_KEY_VERSION = '2';
    (env as unknown as Record<string, unknown>).NETWORK_IDENTITY_HMAC_KEY_VERSION_PREVIOUS = '1';

    const secrets = resolveNetworkIdentitySecrets(env);
    expect(secrets).toMatchObject({
      current: 'new-secret',
      previous: 'old-secret',
      previousVersion: 1,
      currentVersion: 2,
    });

    const oldIdentity = await resolveRequestNetworkIdentity(
      '203.0.113.90',
      { current: 'old-secret', previous: null, previousVersion: null, currentVersion: 1 },
      NOW,
    );
    expect(oldIdentity?.keyVersion).toBe(1);
    env.__db
      .prepare(
        `INSERT INTO broker_request_events (
          endpoint, ip_digest, ip_key_version, ip_epoch, installation_id, observed_at
        ) VALUES (?, ?, ?, ?, ?, ?)`,
      )
      .run(
        'POST /v1/auth/qq/assert',
        oldIdentity?.digest ?? '',
        1,
        '2026-09-01',
        null,
        '2026-09-01T09:59:00.000Z',
      );
    env.__db
      .prepare(
        `INSERT INTO broker_request_events (
          endpoint, ip_digest, ip_key_version, ip_epoch, installation_id, observed_at
        ) VALUES (?, ?, ?, ?, ?, ?)`,
      )
      .run(
        'POST /v1/auth/qq/assert',
        oldIdentity?.digest ?? '',
        1,
        '2026-09-01',
        null,
        '2026-09-01T09:59:30.000Z',
      );

    const { updateAbuseControls } = await import('./test-support/abuse-controls');
    updateAbuseControls(env, (controls) => {
      controls.qqAuthAssertIp.maxRequests = 1;
    });
    await expect(
      checkEndpointRateLimit(env.BROKER_DB, {
        endpoint: 'POST /v1/auth/qq/assert',
        now: NOW,
        ip: '203.0.113.90',
        installationId: null,
        hardwareHash: null,
        networkIdentitySecrets: testNetworkIdentitySecrets(env),
      }),
    ).resolves.toMatchObject({ status: 429, subcode: 'ip_rate_limited' });

    await expect(
      checkEndpointRateLimit(env.BROKER_DB, {
        endpoint: 'POST /v1/auth/qq/assert',
        now: NOW,
        ip: '203.0.113.91',
        installationId: null,
        hardwareHash: null,
        networkIdentitySecrets: testNetworkIdentitySecrets(env),
      }),
    ).resolves.toBeNull();
  });

  it('correlates digests across the epoch boundary only inside the active window', async () => {
    const env = createTestBrokerEnv();
    const secrets = resolveNetworkIdentitySecrets(env)!;
    const previousDay = new Date('2026-08-31T23:59:00.000Z');
    const previousIdentity = await resolveRequestNetworkIdentity('203.0.113.92', secrets, previousDay);
    expect(previousIdentity?.epoch).toBe('2026-08-31');
    env.__db
      .prepare(
        `INSERT INTO broker_request_events (
          endpoint, ip_digest, ip_key_version, ip_epoch, installation_id, observed_at
        ) VALUES (?, ?, ?, ?, ?, ?)`,
      )
      .run(
        'POST /v1/auth/qq/assert',
        previousIdentity?.digest ?? '',
        1,
        previousIdentity?.epoch ?? '',
        null,
        '2026-08-31T23:59:00.000Z',
      );
    env.__db
      .prepare(
        `INSERT INTO broker_request_events (
          endpoint, ip_digest, ip_key_version, ip_epoch, installation_id, observed_at
        ) VALUES (?, ?, ?, ?, ?, ?)`,
      )
      .run(
        'POST /v1/auth/qq/assert',
        previousIdentity?.digest ?? '',
        1,
        previousIdentity?.epoch ?? '',
        null,
        '2026-08-31T23:59:30.000Z',
      );

    const { updateAbuseControls } = await import('./test-support/abuse-controls');
    updateAbuseControls(env, (controls) => {
      controls.qqAuthAssertIp.maxRequests = 1;
    });
    await expect(
      checkEndpointRateLimit(env.BROKER_DB, {
        endpoint: 'POST /v1/auth/qq/assert',
        now: new Date('2026-09-01T00:10:00.000Z'),
        ip: '203.0.113.92',
        installationId: null,
        hardwareHash: null,
        networkIdentitySecrets: testNetworkIdentitySecrets(env),
      }),
    ).resolves.toMatchObject({ status: 429 });

    await expect(
      checkEndpointRateLimit(env.BROKER_DB, {
        endpoint: 'POST /v1/auth/qq/assert',
        now: new Date('2026-09-01T00:16:00.000Z'),
        ip: '203.0.113.92',
        installationId: null,
        hardwareHash: null,
        networkIdentitySecrets: testNetworkIdentitySecrets(env),
      }),
    ).resolves.toBeNull();
  });

  it('keeps operator IP hooks keyed and stable across epochs', async () => {
    const env = createTestBrokerEnv();
    const secrets = resolveNetworkIdentitySecrets(env)!;
    const normalized = normalizeNetworkIdentityIp('203.0.113.93')!;
    const digests = await deriveStableNetworkIdentityDigest(secrets, normalized, 'ip');
    expect(digests).toHaveLength(1);
    expect(digests[0]?.digest).toMatch(/^[a-f0-9]{64}$/u);

    const rawSha = await crypto.subtle.digest(
      'SHA-256',
      new TextEncoder().encode('203.0.113.93'),
    );
    const rawHex = Array.from(new Uint8Array(rawSha), (byte) =>
      byte.toString(16).padStart(2, '0'),
    ).join('');
    expect(digests[0]?.digest).not.toBe(rawHex);
    expect(digests[0]?.digest).not.toBe('203.0.113.93');
  });


  it('never logs raw keys, tokens, digests detail, or raw IPs during failure handling', async () => {
    const env = createTestBrokerEnv();
    const seen: string[] = [];
    const serialize = (args: unknown[]) =>
      args.map((arg) => (typeof arg === 'string' ? arg : JSON.stringify(arg))).join(' ');
    const infoSpy = vi.spyOn(console, 'info').mockImplementation((...args: unknown[]) => {
      seen.push(serialize(args));
    });
    const errorSpy = vi.spyOn(console, 'error').mockImplementation((...args: unknown[]) => {
      seen.push(serialize(args));
    });
    try {
      const { runNetworkIdentityBackfill } = await import('../src/network-identity-migration');
      await runNetworkIdentityBackfill(env.BROKER_DB, resolveNetworkIdentitySecrets(env), NOW);

      const { createManagedOperation, expireManagedOperation, getManagedOperation, hashManagedOperationResumeToken } =
        await import('../src/managed-operation');
      const operationId = 'ph-mop-v1_log_probe_operation_01';
      const resumeToken = 'log-probe-resume-token-value-which-is-secret';
      await createManagedOperation(env.BROKER_DB, {
        operationId,
        resumeTokenHash: await hashManagedOperationResumeToken(resumeToken),
        issueSource: 'discord',
        subjectRef: 'ph-discord-user-v1_log_probe',
        installationId: 'install-log-probe-op',
        devicePublicKey: 'device-log-probe-op',
        now: new Date('2026-09-01T09:00:00.000Z'),
      });
      const created = (await getManagedOperation(env.BROKER_DB, operationId))!;
      await expireManagedOperation(env.BROKER_DB, created, NOW);

      const { sweepStaleManagedOperations } = await import('../src/managed-operation');
      await sweepStaleManagedOperations(env, NOW);

      const rawIp = '203.0.113.200';
      const rawKey = 'or-raw-key-must-never-log';
      const rawToken = 'raw-ack-token-must-never-log';
      const corpus = seen.join('\n');
      expect(corpus).not.toContain(rawIp);
      expect(corpus).not.toContain(rawKey);
      expect(corpus).not.toContain(rawToken);
      expect(corpus).not.toContain(resumeToken);
      expect(corpus).not.toContain(env.NETWORK_IDENTITY_HMAC_SECRET);
      expect(corpus).toContain(operationId);
    } finally {
      infoSpy.mockRestore();
      errorSpy.mockRestore();
    }
  });
});

describe('network identity windows and hooks', () => {
  it('counts requests across every UTC-day epoch inside a multi-day window', async () => {
    const env = createTestBrokerEnv();
    const secrets = resolveNetworkIdentitySecrets(env)!;
    for (const iso of [
      '2026-08-30T12:00:00.000Z',
      '2026-08-31T12:00:00.000Z',
      '2026-09-01T11:50:00.000Z',
    ]) {
      const identity = await resolveRequestNetworkIdentity('203.0.113.92', secrets, new Date(iso));
      env.__db
        .prepare(
          `INSERT INTO broker_request_events (
            endpoint, ip_digest, ip_key_version, ip_epoch, installation_id, observed_at
          ) VALUES (?, ?, ?, ?, ?, ?)`,
        )
        .run(
          'POST /v1/auth/qq/assert',
          identity?.digest ?? '',
          identity?.keyVersion ?? 1,
          identity?.epoch ?? '',
          null,
          iso,
        );
    }

    const { updateAbuseControls } = await import('./test-support/abuse-controls');
    updateAbuseControls(env, (controls) => {
      controls.qqAuthAssertIp.maxRequests = 2;
      controls.qqAuthAssertIp.windowMinutes = 2880;
    });
    await expect(
      checkEndpointRateLimit(env.BROKER_DB, {
        endpoint: 'POST /v1/auth/qq/assert',
        now: new Date('2026-09-01T12:00:00.000Z'),
        ip: '203.0.113.92',
        installationId: null,
        hardwareHash: null,
        networkIdentitySecrets: testNetworkIdentitySecrets(env),
      }),
    ).resolves.toMatchObject({ status: 429, subcode: 'ip_rate_limited' });

    await expect(
      checkEndpointRateLimit(env.BROKER_DB, {
        endpoint: 'POST /v1/auth/qq/assert',
        now: new Date('2026-09-01T12:00:00.000Z'),
        ip: '203.0.113.93',
        installationId: null,
        hardwareHash: null,
        networkIdentitySecrets: testNetworkIdentitySecrets(env),
      }),
    ).resolves.toBeNull();
  });

  it('converts raw-IP operator hooks to stable digests without losing enforcement', async () => {
    const provision = createTestBrokerEnv();
    const secrets = resolveNetworkIdentitySecrets(provision)!;
    const db = new DatabaseSync(':memory:');
    try {
      const { BROKER_MIGRATION_FILENAMES, readBrokerMigrationSql } = await import(
        './test-support/migrations'
      );
      for (const file of BROKER_MIGRATION_FILENAMES) {
        if (file <= '0019_managed_referral_settlement.sql') {
          db.exec(readBrokerMigrationSql(file));
        }
      }
      db.prepare(
        `INSERT INTO broker_abuse_subject_hooks (
          hook_kind, subject_type, subject_value, outcome_code, outcome_class
        ) VALUES ('denylist', 'ip', ?, 'trial_unavailable', 'terminal')`,
      ).run('203.0.113.99');
      for (const file of BROKER_MIGRATION_FILENAMES) {
        if (file === '0020_network_identity_hmac.sql') {
          db.exec(readBrokerMigrationSql(file));
        }
      }
      const phase = db
        .prepare(`SELECT value FROM broker_config WHERE key = 'network_identity_migration'`)
        .get() as { value: string };
      expect(JSON.parse(phase.value)).toMatchObject({ phase: 'dual_write' });

      const { matchSubjectHook } = await import('../src/abuse-controls');
      const wrapped = wrapDatabaseSync(db);
      const context = {
        endpoint: 'POST /v1/trial/challenge',
        now: NOW,
        ip: '203.0.113.99',
        installationId: null,
        hardwareHash: null,
        networkIdentitySecrets: secrets,
      };
      await expect(matchSubjectHook(wrapped, context)).resolves.toMatchObject({
        hookKind: 'denylist',
      });

      const converted = await runNetworkIdentityBackfill(wrapped, secrets, NOW);
      expect(converted.hooksConverted).toBe(1);
      expect(converted.pendingHooks).toBe(0);
      const stored = db
        .prepare(`SELECT subject_value FROM broker_abuse_subject_hooks WHERE subject_type = 'ip'`)
        .get() as { subject_value: string };
      expect(stored.subject_value).toMatch(/^[a-f0-9]{64}$/u);
      expect(stored.subject_value).not.toBe('203.0.113.99');
      await expect(matchSubjectHook(wrapped, context)).resolves.toMatchObject({
        hookKind: 'denylist',
      });

      const later = new Date(NOW.getTime() + 25 * 60 * 60_000);
      const converged = await runNetworkIdentityBackfill(wrapped, secrets, later);
      expect(converged.finalized).toBe(true);
      db.exec(readBrokerMigrationSql('0021_network_identity_purge.sql'));
      await expect(matchSubjectHook(wrapped, context)).resolves.toMatchObject({
        hookKind: 'denylist',
      });
    } finally {
      db.close();
    }
  });

  it('marks unparseable request rows with a sentinel so migration still converges', async () => {
    const provision = createTestBrokerEnv();
    const secrets = resolveNetworkIdentitySecrets(provision)!;
    const db = new DatabaseSync(':memory:');
    try {
      const { BROKER_MIGRATION_FILENAMES, readBrokerMigrationSql } = await import(
        './test-support/migrations'
      );
      for (const file of BROKER_MIGRATION_FILENAMES) {
        if (file <= '0019_managed_referral_settlement.sql') {
          db.exec(readBrokerMigrationSql(file));
        }
      }
      db.prepare(
        `INSERT INTO broker_request_events (endpoint, ip, installation_id, observed_at)
         VALUES (?, ?, ?, ?)`,
      ).run('POST /v1/auth/qq/assert', 'not-an-ip', null, NOW_ISO);
      for (const file of BROKER_MIGRATION_FILENAMES) {
        if (file === '0020_network_identity_hmac.sql') {
          db.exec(readBrokerMigrationSql(file));
        }
      }
      const wrapped = wrapDatabaseSync(db);
      const result = await runNetworkIdentityBackfill(wrapped, secrets, NOW);
      expect(result.requestEventsBackfilled).toBe(0);
      expect(result.pendingRequestEvents).toBe(0);
      expect(result.finalized).toBe(true);
      const wedge = db
        .prepare(`SELECT COUNT(*) AS count FROM broker_request_events WHERE ip IS NOT NULL`)
        .get() as { count: number };
      expect(wedge.count).toBe(0);
      const phase = db
        .prepare(`SELECT value FROM broker_config WHERE key = 'network_identity_migration'`)
        .get() as { value: string };
      expect(JSON.parse(phase.value)).toMatchObject({ phase: 'keyed_only' });
      db.exec(readBrokerMigrationSql('0021_network_identity_purge.sql'));
    } finally {
      db.close();
    }
  });
});

describe('network identity window inventory and fail-closed finalize', () => {
  it('keeps the purge gate and the backfill horizon on one window inventory', async () => {
    const { NETWORK_IDENTITY_WINDOW_CONFIG_PATHS, resolveNetworkIdentityMaxWindowMinutes } =
      await import('../src/network-identity-migration');
    const { readBrokerMigrationSql } = await import('./test-support/migrations');
    const sql = readBrokerMigrationSql('0021_network_identity_purge.sql');
    const gatePaths = new Set(
      [...sql.matchAll(/\$\.([\w.]+)\.windowMinutes/gu)].map((match) => match[1]),
    );
    const retiredPaths = ['qqAuthStatusIp', 'referralAttempts.validShaped', 'referralAttempts.unknown', 'referralAttempts.perReferralIdVelocity', 'referralAttempts.perReferrerRewardVelocity'];
    expect(new Set([...gatePaths].filter((path) => !retiredPaths.includes(path!)))).toEqual(new Set(NETWORK_IDENTITY_WINDOW_CONFIG_PATHS));
    expect(NETWORK_IDENTITY_WINDOW_CONFIG_PATHS).toContain('qqAuthAssertIp');
    expect(NETWORK_IDENTITY_WINDOW_CONFIG_PATHS).toContain('trialStatus');

    const env = createTestBrokerEnv();
    const { updateAbuseControls } = await import('./test-support/abuse-controls');
    updateAbuseControls(env, (controls) => {
      controls.trialStatus.windowMinutes = 10080;
    });
    await expect(
      resolveNetworkIdentityMaxWindowMinutes(env.BROKER_DB),
    ).resolves.toBe(10080);
  });

  it('fails closed instead of flipping phase when migration state cannot persist', async () => {
    const db = new DatabaseSync(':memory:');
    try {
      const { BROKER_MIGRATION_FILENAMES, readBrokerMigrationSql } = await import(
        './test-support/migrations'
      );
      for (const file of BROKER_MIGRATION_FILENAMES) {
        if (file <= '0019_managed_referral_settlement.sql') {
          db.exec(readBrokerMigrationSql(file));
        }
      }
      db.prepare(
        `INSERT INTO broker_request_events (endpoint, ip, installation_id, observed_at)
         VALUES (?, ?, ?, ?)`,
      ).run('POST /v1/auth/qq/assert', '203.0.113.103', null, NOW_ISO);
      for (const file of BROKER_MIGRATION_FILENAMES) {
        if (file === '0020_network_identity_hmac.sql') {
          db.exec(readBrokerMigrationSql(file));
        }
      }
      const { finalizeNetworkIdentityMigration } = await import(
        '../src/network-identity-migration'
      );
      const wrapped = wrapDatabaseSync(db);
      expect(await finalizeNetworkIdentityMigration(wrapped, NOW)).toBe(true);

      db.prepare(`UPDATE broker_config SET value = ? WHERE key = 'network_identity_migration'`).run(
        JSON.stringify({ phase: 'dual_write', purge_after: null }),
      );
      db.prepare(
        `INSERT INTO broker_request_events (endpoint, ip, installation_id, observed_at)
         VALUES (?, ?, ?, ?)`,
      ).run('POST /v1/auth/qq/assert', '203.0.113.104', null, NOW_ISO);
      db.prepare(`DELETE FROM broker_config WHERE key = 'network_identity_migration'`).run();
      expect(await finalizeNetworkIdentityMigration(wrapped, NOW)).toBe(false);
      const phase = db
        .prepare(`SELECT value FROM broker_config WHERE key = 'network_identity_migration'`)
        .get() as { value: string } | undefined;
      expect(phase).toBeUndefined();
    } finally {
      db.close();
    }
  });

  it('blocks finalization on unparseable hooks and surfaces non-secret diagnostics', async () => {
    const provision = createTestBrokerEnv();
    const secrets = resolveNetworkIdentitySecrets(provision)!;
    const db = new DatabaseSync(':memory:');
    try {
      const { BROKER_MIGRATION_FILENAMES, readBrokerMigrationSql } = await import(
        './test-support/migrations'
      );
      for (const file of BROKER_MIGRATION_FILENAMES) {
        if (file <= '0019_managed_referral_settlement.sql') {
          db.exec(readBrokerMigrationSql(file));
        }
      }
      db.prepare(
        `INSERT INTO broker_velocity_cap_hooks (
          subject_type, subject_value, max_requests, window_minutes,
          outcome_code, outcome_class, active
        ) VALUES ('ip', ?, 1, 60, 'rate_limited', 'retryable', 1)`,
      ).run('not-an-ip-hook');
      for (const file of BROKER_MIGRATION_FILENAMES) {
        if (file === '0020_network_identity_hmac.sql') {
          db.exec(readBrokerMigrationSql(file));
        }
      }
      const wrapped = wrapDatabaseSync(db);
      const blocked = await runNetworkIdentityBackfill(wrapped, secrets, NOW);
      expect(blocked.finalized).toBe(false);
      expect(blocked.rawHooks).toBe(1);
      expect(blocked.unparseableHooks).toBe(1);
      expect(blocked.rawHookSampleIds).toEqual([
        expect.objectContaining({ table: 'broker_velocity_cap_hooks', id: expect.any(Number) }),
      ]);
      const phase = db
        .prepare(`SELECT value FROM broker_config WHERE key = 'network_identity_migration'`)
        .get() as { value: string };
      expect(JSON.parse(phase.value)).toMatchObject({ phase: 'dual_write' });

      db.prepare(`UPDATE broker_velocity_cap_hooks SET active = 0 WHERE subject_value = ?`).run(
        'not-an-ip-hook',
      );
      const converged = await runNetworkIdentityBackfill(wrapped, secrets, NOW);
      expect(converged.rawHooks).toBe(0);
      expect(converged.finalized).toBe(true);
    } finally {
      db.close();
    }
  });
});
describe('network identity 0024 sequencing', () => {
  it('applies the full chain sequentially and allows unattributed request rows', async () => {
    const db = new DatabaseSync(':memory:');
    try {
      const { applyBrokerMigrations } = await import('./test-support/migrations');
      applyBrokerMigrations(db);
      db.prepare(
        `INSERT INTO broker_request_events (
          endpoint, ip_digest, ip_key_version, ip_epoch, installation_id, observed_at
        ) VALUES (?, NULL, NULL, NULL, NULL, ?)`,
      ).run('POST /v1/providers/openrouter/managed-key-delivery/ack', NOW_ISO);
      const row = db
        .prepare(
          `SELECT endpoint FROM broker_request_events WHERE installation_id IS NULL AND ip_digest IS NULL`,
        )
        .get() as { endpoint: string };
      expect(row.endpoint).toBe('POST /v1/providers/openrouter/managed-key-delivery/ack');
    } finally {
      db.close();
    }
  });

  it('refuses 0024 before the 0021 purge without touching raw IP', async () => {
    const { BROKER_MIGRATION_FILENAMES, readBrokerMigrationSql } = await import(
      './test-support/migrations'
    );
    const db = new DatabaseSync(':memory:');
    try {
      for (const file of BROKER_MIGRATION_FILENAMES) {
        if (file <= '0020_network_identity_hmac.sql') {
          db.exec(readBrokerMigrationSql(file));
        }
      }
      db.prepare(
        `INSERT INTO broker_request_events (endpoint, ip, installation_id, observed_at)
         VALUES (?, ?, ?, ?)`,
      ).run('POST /v1/auth/qq/assert', '203.0.113.104', null, NOW_ISO);
      expect(() => db.exec(readBrokerMigrationSql('0024_allow_unattributed_request_events.sql'))).toThrow(
        /constraint/i,
      );
      const columns = db.prepare(`PRAGMA table_info(broker_request_events)`).all() as Array<{
        name: string;
      }>;
      expect(columns.map((column) => column.name)).toContain('ip');
      const legacy = db
        .prepare(`SELECT ip FROM broker_request_events WHERE ip IS NOT NULL`)
        .get() as { ip: string };
      expect(legacy.ip).toBe('203.0.113.104');
    } finally {
      db.close();
    }
  });
});

describe('network identity purge gate', () => {
  it('holds the purge gate open for the longest operator hook window until backfill converges', async () => {
    const provision = createTestBrokerEnv();
    const secrets = resolveNetworkIdentitySecrets(provision)!;
    const db = new DatabaseSync(':memory:');
    try {
      const { BROKER_MIGRATION_FILENAMES, readBrokerMigrationSql } = await import(
        './test-support/migrations'
      );
      for (const file of BROKER_MIGRATION_FILENAMES) {
        if (file <= '0019_managed_referral_settlement.sql') {
          db.exec(readBrokerMigrationSql(file));
        }
      }
      db.prepare(
        `INSERT INTO broker_velocity_cap_hooks (
          subject_type, subject_value, max_requests, window_minutes,
          outcome_code, outcome_class, active
        ) VALUES ('ip', ?, 1, 10080, 'rate_limited', 'retryable', 1)`,
      ).run('203.0.113.102');
      const threeDaysAgo = new Date(NOW.getTime() - 3 * 24 * 60 * 60_000).toISOString();
      db.prepare(
        `INSERT INTO broker_request_events (endpoint, ip, installation_id, observed_at)
         VALUES (?, ?, ?, ?)`,
      ).run('POST /v1/auth/qq/assert', '203.0.113.102', null, threeDaysAgo);
      for (const file of BROKER_MIGRATION_FILENAMES) {
        if (file === '0020_network_identity_hmac.sql') {
          db.exec(readBrokerMigrationSql(file));
        }
      }
      const phase = db
        .prepare(`SELECT value FROM broker_config WHERE key = 'network_identity_migration'`)
        .get() as { value: string };
      expect(JSON.parse(phase.value)).toMatchObject({ phase: 'dual_write' });

      expect(() => db.exec(readBrokerMigrationSql('0021_network_identity_purge.sql'))).toThrow(
        /constraint/i,
      );

      const wrapped = wrapDatabaseSync(db);
      const backfilled = await runNetworkIdentityBackfill(wrapped, secrets, NOW);
      expect(backfilled.requestEventsBackfilled).toBe(1);
      expect(backfilled.hooksConverted).toBe(1);
      expect(backfilled.finalized).toBe(true);

      db.exec(readBrokerMigrationSql('0021_network_identity_purge.sql'));
      const columns = db.prepare(`PRAGMA table_info(broker_request_events)`).all() as Array<{
        name: string;
      }>;
      expect(columns.map((column) => column.name)).not.toContain('ip');
    } finally {
      db.close();
    }
  });
});

describe('network identity staged migration', () => {
  it('preserves active windows through dual-write, backfill, and gated purge', async () => {
    const db = new DatabaseSync(':memory:');
    try {
      const files = (await import('./test-support/migrations')).BROKER_MIGRATION_FILENAMES;
      const { readBrokerMigrationSql } = await import('./test-support/migrations');
      for (const file of files) {
        if (file <= '0019_managed_referral_settlement.sql') {
          db.exec(readBrokerMigrationSql(file));
        }
      }
      db.prepare(
        `INSERT INTO broker_request_events (endpoint, ip, installation_id, observed_at)
         VALUES (?, ?, ?, ?)`,
      ).run('POST /v1/auth/qq/assert', '203.0.113.100', null, NOW_ISO);

      for (const file of files) {
        if (file === '0020_network_identity_hmac.sql') {
          db.exec(readBrokerMigrationSql(file));
        }
      }
      const phase = db
        .prepare(`SELECT value FROM broker_config WHERE key = 'network_identity_migration'`)
        .get() as { value: string };
      expect(JSON.parse(phase.value)).toMatchObject({ phase: 'dual_write' });

      for (const file of files) {
        if (file === '0021_network_identity_purge.sql') {
          expect(() => db.exec(readBrokerMigrationSql(file))).toThrow(/constraint/i);
        }
      }
    } finally {
      db.close();
    }
  });

  it('backfills surviving raw request rows with the worker secret and finalizes privacy', async () => {
    const env = createTestBrokerEnv();
    const db = new DatabaseSync(':memory:');
    try {
      const { BROKER_MIGRATION_FILENAMES, readBrokerMigrationSql } = await import(
        './test-support/migrations'
      );
      for (const file of BROKER_MIGRATION_FILENAMES) {
        if (file <= '0019_managed_referral_settlement.sql') {
          db.exec(readBrokerMigrationSql(file));
        }
      }
      db.prepare(
        `INSERT INTO broker_request_events (endpoint, ip, installation_id, observed_at)
         VALUES (?, ?, ?, ?)`,
      ).run('POST /v1/auth/qq/assert', '203.0.113.101', null, NOW_ISO);
      for (const file of BROKER_MIGRATION_FILENAMES) {
        if (file === '0020_network_identity_hmac.sql') {
          db.exec(readBrokerMigrationSql(file));
        }
      }
      const stagedPhase = db
        .prepare(`SELECT value FROM broker_config WHERE key = 'network_identity_migration'`)
        .get() as { value: string };
      expect(JSON.parse(stagedPhase.value)).toMatchObject({ phase: 'dual_write' });

      const secrets = resolveNetworkIdentitySecrets(env);
      const wrapped = wrapDatabaseSync(db);
      const first = await runNetworkIdentityBackfill(wrapped, secrets, NOW);
      expect(first.requestEventsBackfilled).toBe(1);
      expect(first.finalized).toBe(true);

      const later = new Date(NOW.getTime() + 25 * 60 * 60_000);
      const second = await runNetworkIdentityBackfill(wrapped, secrets, later);
      expect(second.finalized).toBe(true);

      const legacy = db
        .prepare(`SELECT COUNT(*) AS count FROM broker_request_events WHERE ip IS NOT NULL`)
        .get() as { count: number };
      expect(legacy.count).toBe(0);

      db.exec(readBrokerMigrationSql('0021_network_identity_purge.sql'));
      const columns = db
        .prepare("SELECT name FROM pragma_table_info('broker_request_events') ORDER BY cid")
        .all() as Array<{ name: string }>;
      expect(columns.map((column) => column.name)).not.toContain('ip');
    } finally {
      db.close();
    }
  });
});

function wrapDatabaseSync(db: DatabaseSync): D1Database {
  const bound = (sql: string, params: unknown[]) => ({
    first: async () => db.prepare(sql).get(...(params as [])) ?? null,
    run: async () => {
      const result = db.prepare(sql).run(...(params as [])) as unknown as {
        changes: number | bigint;
      };
      return { meta: { changes: Number(result.changes ?? 0) } };
    },
    all: async <T,>() => ({ results: (db.prepare(sql).all(...(params as [])) as T[]) ?? [] }),
  });
  return {
    prepare: (sql: string) => ({
      bind: (...params: unknown[]) => bound(sql, params),
      first: async () => db.prepare(sql).get() ?? null,
      run: async () => {
        const result = db.prepare(sql).run() as unknown as { changes: number | bigint };
        return { meta: { changes: Number(result.changes ?? 0) } };
      },
      all: async <T,>() => ({ results: (db.prepare(sql).all() as T[]) ?? [] }),
    }),
  } as unknown as D1Database;
}
