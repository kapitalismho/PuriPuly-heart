import { DatabaseSync } from 'node:sqlite';

import { describe, expect, it } from 'vitest';

import { applyBrokerMigrations, readBrokerMigrationSql } from './test-support/migrations';

const VALID_REFERRAL_ID = '7KQ9M2';
const SECOND_VALID_REFERRAL_ID = 'ABCDEF';

function withMigratedDatabase(run: (db: DatabaseSync) => void): void {
  const db = new DatabaseSync(':memory:');
  try {
    applyBrokerMigrations(db);
    run(db);
  } finally {
    db.close();
  }
}

function createOldReferralCheckSchemas(db: DatabaseSync): void {
  const oldCheck =
    "length(referral_id) = 6 AND referral_id GLOB '[23456789ABCDEFGHJKMNPQRSTUVWXYZ][23456789ABCDEFGHJKMNPQRSTUVWXYZ][23456789ABCDEFGHJKMNPQRSTUVWXYZ][23456789ABCDEFGHJKMNPQRSTUVWXYZ][23456789ABCDEFGHJKMNPQRSTUVWXYZ][23456789ABCDEFGHJKMNPQRSTUVWXYZ]'";

  db.exec(`
    CREATE TABLE discord_oauth_sessions (
      state_hash TEXT PRIMARY KEY,
      installation_id TEXT NOT NULL,
      device_public_key TEXT NOT NULL,
      redirect_uri TEXT NOT NULL,
      pkce_code_verifier TEXT,
      issue_nonce_hash TEXT NOT NULL,
      fingerprint_salt_version INTEGER NOT NULL,
      discord_user_ref TEXT,
      discord_email_verified INTEGER CHECK (discord_email_verified IS NULL OR discord_email_verified IN (0, 1)),
      discord_account_created_at TEXT,
      eligibility_checked_at TEXT,
      status TEXT NOT NULL CHECK(status IN ('pending', 'processing', 'consumed', 'canceled', 'failed', 'expired')),
      created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
      expires_at TEXT NOT NULL,
      processing_started_at TEXT,
      consumed_at TEXT,
      referral_id TEXT CHECK (referral_id IS NULL OR (${oldCheck})),
      CHECK (length(installation_id) BETWEEN 1 AND 128),
      CHECK (length(device_public_key) > 0),
      CHECK (length(redirect_uri) > 0)
    ) STRICT;
    CREATE INDEX idx_discord_oauth_sessions_installation_status
      ON discord_oauth_sessions(installation_id, status, created_at);
    CREATE INDEX idx_discord_oauth_sessions_expires_at
      ON discord_oauth_sessions(expires_at);
    CREATE INDEX idx_discord_oauth_sessions_referral_id
      ON discord_oauth_sessions(referral_id)
      WHERE referral_id IS NOT NULL;

    CREATE TABLE referral_codes (
      referral_id TEXT PRIMARY KEY CHECK (${oldCheck}),
      owner_discord_user_ref TEXT NOT NULL CHECK (length(owner_discord_user_ref) > 0),
      owner_installation_id TEXT,
      status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'disabled')),
      created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
      updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
      disabled_reason TEXT CHECK (disabled_reason IS NULL OR length(disabled_reason) BETWEEN 1 AND 64),
      disabled_by TEXT CHECK (disabled_by IS NULL OR length(disabled_by) BETWEEN 1 AND 64),
      disabled_at TEXT
    ) STRICT;
    CREATE UNIQUE INDEX idx_referral_codes_owner_discord_user_ref
      ON referral_codes(owner_discord_user_ref);
    CREATE INDEX idx_referral_codes_owner_installation_id
      ON referral_codes(owner_installation_id)
      WHERE owner_installation_id IS NOT NULL;
    CREATE INDEX idx_referral_codes_status
      ON referral_codes(status, referral_id);

    CREATE TABLE referral_rewards (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      referral_id TEXT NOT NULL CHECK (${oldCheck}),
      referrer_discord_user_ref TEXT CHECK (referrer_discord_user_ref IS NULL OR length(referrer_discord_user_ref) > 0),
      referrer_installation_id TEXT CHECK (referrer_installation_id IS NULL OR length(referrer_installation_id) > 0),
      referred_discord_user_ref TEXT NOT NULL CHECK (length(referred_discord_user_ref) > 0),
      referred_installation_id TEXT NOT NULL CHECK (length(referred_installation_id) > 0),
      referred_hardware_hash TEXT NOT NULL CHECK (length(referred_hardware_hash) BETWEEN 1 AND 128),
      referred_hardware_hash_salt_version INTEGER NOT NULL CHECK (referred_hardware_hash_salt_version > 0),
      referred_bonus_status TEXT NOT NULL CHECK (referred_bonus_status IN ('reserved', 'credited', 'skipped', 'failed')),
      referrer_bonus_status TEXT NOT NULL CHECK (referrer_bonus_status IN ('pending', 'applying', 'credited', 'skipped', 'failed')),
      skip_reason TEXT CHECK (skip_reason IS NULL OR length(skip_reason) BETWEEN 1 AND 64),
      failure_reason TEXT CHECK (failure_reason IS NULL OR length(failure_reason) BETWEEN 1 AND 64),
      referred_managed_credential_ref TEXT,
      referrer_managed_credential_ref TEXT,
      created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
      updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
      credited_at TEXT,
      attempt_ip_hash TEXT CHECK (attempt_ip_hash IS NULL OR length(attempt_ip_hash) = 64),
      CHECK ((referrer_discord_user_ref IS NULL AND referrer_installation_id IS NULL) OR (referrer_discord_user_ref IS NOT NULL AND referrer_installation_id IS NOT NULL)),
      CHECK (referrer_discord_user_ref IS NOT NULL OR (referred_bonus_status = 'skipped' AND referrer_bonus_status = 'skipped' AND skip_reason IS NOT NULL))
    ) STRICT;
    CREATE INDEX idx_referral_rewards_referral_id ON referral_rewards(referral_id);
    CREATE INDEX idx_referral_rewards_referrer_cap
      ON referral_rewards(referrer_discord_user_ref, referred_bonus_status)
      WHERE referrer_discord_user_ref IS NOT NULL AND referred_bonus_status IN ('reserved', 'credited');
    CREATE UNIQUE INDEX idx_referral_rewards_counted_referred_discord_user
      ON referral_rewards(referred_discord_user_ref)
      WHERE referred_bonus_status IN ('reserved', 'credited');
    CREATE UNIQUE INDEX idx_referral_rewards_counted_referred_installation
      ON referral_rewards(referred_installation_id)
      WHERE referred_bonus_status IN ('reserved', 'credited');
    CREATE INDEX idx_referral_rewards_attempt_installation_time
      ON referral_rewards(referred_installation_id, created_at);
    CREATE INDEX idx_referral_rewards_attempt_ip_hash_time
      ON referral_rewards(attempt_ip_hash, created_at)
      WHERE attempt_ip_hash IS NOT NULL;
    CREATE INDEX idx_referral_rewards_referral_velocity
      ON referral_rewards(referral_id, created_at);
    CREATE INDEX idx_referral_rewards_referrer_velocity
      ON referral_rewards(referrer_discord_user_ref, created_at)
      WHERE referrer_discord_user_ref IS NOT NULL;
  `);
}

describe('broker referral persistence foundation', () => {
  it('migrates strict referral tables, nullable OAuth session referral input, and referral lookup indexes', () => {
    withMigratedDatabase((db) => {
      const tableStrictness = db
        .prepare(
          `SELECT name, strict
             FROM pragma_table_list
            WHERE name IN ('referral_codes', 'referral_rewards')
            ORDER BY name`,
        )
        .all() as Array<{ name: string; strict: number }>;
      expect(tableStrictness).toEqual([
        { name: 'referral_codes', strict: 1 },
        { name: 'referral_rewards', strict: 1 },
      ]);

      const referralCodeColumns = columnNames(db, 'referral_codes');
      expect(referralCodeColumns).toEqual([
        'referral_id',
        'owner_source',
        'owner_subject_ref',
        'owner_discord_user_ref',
        'owner_installation_id',
        'status',
        'created_at',
        'updated_at',
        'disabled_reason',
        'disabled_by',
        'disabled_at',
      ]);

      const referralRewardColumns = columnNames(db, 'referral_rewards');
      expect(referralRewardColumns).toEqual([
        'id',
        'referral_id',
        'referrer_source',
        'referrer_subject_ref',
        'referrer_discord_user_ref',
        'referrer_installation_id',
        'referred_source',
        'referred_subject_ref',
        'referred_discord_user_ref',
        'referred_installation_id',
        'referred_hardware_hash',
        'referred_hardware_hash_salt_version',
        'referred_bonus_status',
        'referrer_bonus_status',
        'skip_reason',
        'failure_reason',
        'referred_managed_credential_ref',
        'referrer_managed_credential_ref',
        'created_at',
        'updated_at',
        'credited_at',
        'attempt_ip_digest',
        'attempt_ip_key_version',
        'attempt_ip_epoch',
        'operation_id',
      ]);

      expect(columnNames(db, 'discord_oauth_sessions')).toContain('referral_id');

      const indexes = db
        .prepare(
          `SELECT name, sql
             FROM sqlite_schema
            WHERE type = 'index'
              AND tbl_name IN ('referral_codes', 'referral_rewards', 'discord_oauth_sessions')
            ORDER BY name`,
        )
        .all() as Array<{ name: string; sql: string | null }>;

      expect(indexes.map((index) => index.name)).toEqual(
        expect.arrayContaining([
          'idx_referral_codes_owner_subject',
          'idx_referral_codes_owner_discord_user_ref',
          'idx_referral_codes_owner_installation_id',
          'idx_referral_codes_status',
          'idx_referral_rewards_referral_id_created_at',
          'idx_referral_rewards_referrer_subject_status',
          'idx_referral_rewards_counted_referred_subject',
          'idx_referral_rewards_counted_referred_installation',
          'idx_referral_rewards_referred_subject_created_at',
          'idx_referral_rewards_referred_installation_created_at',
          'idx_referral_rewards_attempt_digest_created_at',
          'idx_referral_rewards_referrer_subject_created_at',
          'idx_discord_oauth_sessions_referral_id',
        ]),
      );
      expect(indexes.map((index) => index.name)).not.toContain(
        'idx_referral_rewards_attempt_ip_hash_time',
      );
      expect(indexSql(indexes, 'idx_referral_rewards_counted_referred_subject').replace(/\s+/gu, ' ')).toContain(
        "referred_bonus_status IN ('reserved', 'credited')",
      );
      expect(indexSql(indexes, 'idx_referral_rewards_counted_referred_installation').replace(/\s+/gu, ' ')).toContain(
        "referred_bonus_status IN ('reserved', 'credited')",
      );
    });
  });

  it('keeps previous and source-aware Worker referral SQL compatible across migration-before-deploy', () => {
    const db = new DatabaseSync(':memory:');
    try {
      applyBrokerMigrations(db, {
        through: '0015_add_app_active_days.sql',
      });
      db.prepare(
        `INSERT INTO referral_codes (
          referral_id, owner_discord_user_ref, owner_installation_id, status
        ) VALUES (?, ?, ?, 'active')`,
      ).run(
        VALID_REFERRAL_ID,
        'ph-discord-user-v1_legacy-before-migration',
        'install-legacy-before-migration',
      );
      db.prepare(
        `INSERT INTO referral_rewards (
          referral_id, referrer_discord_user_ref, referrer_installation_id,
          referred_discord_user_ref, referred_installation_id, referred_hardware_hash,
          referred_hardware_hash_salt_version, referred_bonus_status,
          referrer_bonus_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'reserved', 'pending')`,
      ).run(
        VALID_REFERRAL_ID,
        'ph-discord-user-v1_legacy-before-migration',
        'install-legacy-before-migration',
        'ph-discord-user-v1_referred-before-migration',
        'install-referred-before-migration',
        'hardware-before-migration',
        7,
      );

      applyBrokerMigrations(db, {
        after: '0015_add_app_active_days.sql',
        through: '0016_make_referrals_source_aware.sql',
      });

      expect(
        db
          .prepare(
            `SELECT owner_source, owner_subject_ref, owner_discord_user_ref
               FROM referral_codes
              WHERE referral_id = ?`,
          )
          .get(VALID_REFERRAL_ID),
      ).toEqual({
        owner_source: 'discord',
        owner_subject_ref: 'ph-discord-user-v1_legacy-before-migration',
        owner_discord_user_ref: 'ph-discord-user-v1_legacy-before-migration',
      });
      expect(
        db
          .prepare(
            `SELECT referrer_source, referrer_subject_ref, referrer_discord_user_ref,
                    referred_source, referred_subject_ref, referred_discord_user_ref
               FROM referral_rewards
              WHERE referral_id = ?`,
          )
          .get(VALID_REFERRAL_ID),
      ).toEqual({
        referrer_source: 'discord',
        referrer_subject_ref: 'ph-discord-user-v1_legacy-before-migration',
        referrer_discord_user_ref: 'ph-discord-user-v1_legacy-before-migration',
        referred_source: 'discord',
        referred_subject_ref: 'ph-discord-user-v1_referred-before-migration',
        referred_discord_user_ref: 'ph-discord-user-v1_referred-before-migration',
      });

      db.prepare(
        `INSERT INTO referral_codes (
          referral_id, owner_discord_user_ref, owner_installation_id, status
        ) VALUES (?, ?, ?, 'active')`,
      ).run(
        SECOND_VALID_REFERRAL_ID,
        'ph-discord-user-v1_legacy-during-rollout',
        'install-legacy-during-rollout',
      );
      db.prepare(
        `INSERT INTO referral_rewards (
          referral_id, referrer_discord_user_ref, referrer_installation_id,
          referred_discord_user_ref, referred_installation_id, referred_hardware_hash,
          referred_hardware_hash_salt_version, referred_bonus_status,
          referrer_bonus_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'reserved', 'pending')`,
      ).run(
        SECOND_VALID_REFERRAL_ID,
        'ph-discord-user-v1_legacy-during-rollout',
        'install-legacy-during-rollout',
        'ph-discord-user-v1_referred-during-rollout',
        'install-referred-during-rollout',
        'hardware-during-rollout',
        7,
      );

      expect(
        db
          .prepare(
            `SELECT owner_source, owner_subject_ref
               FROM referral_codes
              WHERE owner_discord_user_ref = ?`,
          )
          .get('ph-discord-user-v1_legacy-during-rollout'),
      ).toEqual({
        owner_source: 'discord',
        owner_subject_ref: 'ph-discord-user-v1_legacy-during-rollout',
      });
      expect(
        db
          .prepare(
            `SELECT referrer_source, referrer_subject_ref,
                    referred_source, referred_subject_ref
               FROM referral_rewards
              WHERE referred_discord_user_ref = ?`,
          )
          .get('ph-discord-user-v1_referred-during-rollout'),
      ).toEqual({
        referrer_source: 'discord',
        referrer_subject_ref: 'ph-discord-user-v1_legacy-during-rollout',
        referred_source: 'discord',
        referred_subject_ref: 'ph-discord-user-v1_referred-during-rollout',
      });

      db.prepare(
        `UPDATE referral_codes
            SET owner_installation_id = ?
          WHERE owner_discord_user_ref = ?`,
      ).run(
        'install-legacy-during-rollout-refreshed',
        'ph-discord-user-v1_legacy-during-rollout',
      );
      db.prepare(
        `UPDATE referral_rewards
            SET referred_bonus_status = 'credited', updated_at = ?
          WHERE referred_discord_user_ref = ?`,
      ).run(
        '2026-08-30T00:00:00.000Z',
        'ph-discord-user-v1_referred-during-rollout',
      );
      expect(
        db
          .prepare(
            `SELECT owner_installation_id
               FROM referral_codes
              WHERE owner_source = 'discord' AND owner_subject_ref = ?`,
          )
          .get('ph-discord-user-v1_legacy-during-rollout'),
      ).toEqual({ owner_installation_id: 'install-legacy-during-rollout-refreshed' });
      expect(
        db
          .prepare(
            `SELECT referred_bonus_status
               FROM referral_rewards
              WHERE referred_source = 'discord' AND referred_subject_ref = ?`,
          )
          .get('ph-discord-user-v1_referred-during-rollout'),
      ).toEqual({ referred_bonus_status: 'credited' });

      db.prepare(
        `INSERT INTO referral_codes (
          referral_id, owner_source, owner_subject_ref, owner_installation_id, status
        ) VALUES (?, 'discord', ?, ?, 'active')`,
      ).run(
        'BCDEFG',
        'ph-discord-user-v1_source-aware-discord',
        'install-source-aware-discord',
      );
      expect(
        db
          .prepare(
            `SELECT owner_discord_user_ref
               FROM referral_codes
              WHERE owner_source = 'discord' AND owner_subject_ref = ?`,
          )
          .get('ph-discord-user-v1_source-aware-discord'),
      ).toEqual({ owner_discord_user_ref: 'ph-discord-user-v1_source-aware-discord' });

      db.prepare(
        `INSERT INTO referral_codes (
          referral_id, owner_source, owner_subject_ref, owner_installation_id, status
        ) VALUES (?, 'qq', ?, ?, 'active')`,
      ).run('CDEFGH', 'ph-qq-subject-v1_source-aware-qq', 'install-source-aware-qq');
      db.prepare(
        `INSERT INTO referral_rewards (
          referral_id, referrer_source, referrer_subject_ref,
          referrer_installation_id, referred_source, referred_subject_ref,
          referred_installation_id, referred_hardware_hash,
          referred_hardware_hash_salt_version, referred_bonus_status,
          referrer_bonus_status
        ) VALUES (?, 'discord', ?, ?, 'qq', ?, ?, NULL, NULL, 'reserved', 'pending')`,
      ).run(
        'BCDEFG',
        'ph-discord-user-v1_source-aware-discord',
        'install-source-aware-discord',
        'ph-qq-subject-v1_source-aware-referred',
        'install-source-aware-referred',
      );
      expect(
        db
          .prepare(
            `SELECT owner_discord_user_ref
               FROM referral_codes
              WHERE owner_source = 'qq' AND owner_subject_ref = ?`,
          )
          .get('ph-qq-subject-v1_source-aware-qq'),
      ).toEqual({ owner_discord_user_ref: null });
      expect(
        db
          .prepare(
            `SELECT referrer_discord_user_ref, referred_discord_user_ref
               FROM referral_rewards
              WHERE referred_source = 'qq' AND referred_subject_ref = ?`,
          )
          .get('ph-qq-subject-v1_source-aware-referred'),
      ).toEqual({
        referrer_discord_user_ref: 'ph-discord-user-v1_source-aware-discord',
        referred_discord_user_ref: null,
      });
    } finally {
      db.close();
    }
  });

  it('uses D1-safe Referral ID checks in all referral-bearing tables', () => {
    withMigratedDatabase((db) => {
      const schemas = db
        .prepare(
          `SELECT name, sql
             FROM sqlite_schema
            WHERE type = 'table'
              AND name IN ('discord_oauth_sessions', 'referral_codes', 'referral_rewards')
            ORDER BY name`,
        )
        .all() as Array<{ name: string; sql: string }>;

      expect(schemas.map((row) => row.name)).toEqual([
        'discord_oauth_sessions',
        'referral_codes',
        'referral_rewards',
      ]);
      for (const schema of schemas) {
        expect(schema.sql).toContain('length(referral_id) = 6');
        expect(schema.sql).toContain("NOT GLOB '*[^23456789ABCDEFGHJKMNPQRSTUVWXYZ]*'");
        expect(schema.sql).not.toContain(
          "GLOB '[23456789ABCDEFGHJKMNPQRSTUVWXYZ][23456789ABCDEFGHJKMNPQRSTUVWXYZ]",
        );
      }
    });
  });

  it('repairs already-applied old Referral ID CHECK constraints while preserving rows and indexes', () => {
    const db = new DatabaseSync(':memory:');
    try {
      createOldReferralCheckSchemas(db);
      const expectedSession = {
        state_hash: 'state-old-check',
        installation_id: 'install-old-check',
        device_public_key: 'device-old-check',
        redirect_uri: 'http://127.0.0.1:62187/discord/callback',
        pkce_code_verifier: 'pkce-old-check',
        issue_nonce_hash: 'issue-nonce-old-check',
        fingerprint_salt_version: 3,
        discord_user_ref: 'ph-discord-user-v1_session-old-check',
        discord_email_verified: 1,
        discord_account_created_at: '2024-02-01T01:02:03.000Z',
        eligibility_checked_at: '2026-05-20T00:01:00.000Z',
        status: 'consumed',
        created_at: '2026-05-20T00:00:00.000Z',
        expires_at: '2026-05-20T00:05:00.000Z',
        processing_started_at: '2026-05-20T00:02:00.000Z',
        consumed_at: '2026-05-20T00:03:00.000Z',
        referral_id: VALID_REFERRAL_ID,
      };
      db.prepare(
        `INSERT INTO discord_oauth_sessions (
          state_hash, installation_id, device_public_key, redirect_uri, pkce_code_verifier,
          issue_nonce_hash, fingerprint_salt_version, discord_user_ref, discord_email_verified,
          discord_account_created_at, eligibility_checked_at, status, created_at, expires_at,
          processing_started_at, consumed_at, referral_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      ).run(
        expectedSession.state_hash,
        expectedSession.installation_id,
        expectedSession.device_public_key,
        expectedSession.redirect_uri,
        expectedSession.pkce_code_verifier,
        expectedSession.issue_nonce_hash,
        expectedSession.fingerprint_salt_version,
        expectedSession.discord_user_ref,
        expectedSession.discord_email_verified,
        expectedSession.discord_account_created_at,
        expectedSession.eligibility_checked_at,
        expectedSession.status,
        expectedSession.created_at,
        expectedSession.expires_at,
        expectedSession.processing_started_at,
        expectedSession.consumed_at,
        expectedSession.referral_id,
      );

      const expectedReferralCode = {
        referral_id: VALID_REFERRAL_ID,
        owner_discord_user_ref: 'ph-discord-user-v1_owner-old-check',
        owner_installation_id: 'install-old-check',
        status: 'disabled',
        created_at: '2026-05-20T00:10:00.000Z',
        updated_at: '2026-05-20T00:11:00.000Z',
        disabled_reason: 'owner_request',
        disabled_by: 'support-console',
        disabled_at: '2026-05-20T00:12:00.000Z',
      };
      db.prepare(
        `INSERT INTO referral_codes (
          referral_id, owner_discord_user_ref, owner_installation_id, status, created_at,
          updated_at, disabled_reason, disabled_by, disabled_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      ).run(
        expectedReferralCode.referral_id,
        expectedReferralCode.owner_discord_user_ref,
        expectedReferralCode.owner_installation_id,
        expectedReferralCode.status,
        expectedReferralCode.created_at,
        expectedReferralCode.updated_at,
        expectedReferralCode.disabled_reason,
        expectedReferralCode.disabled_by,
        expectedReferralCode.disabled_at,
      );

      const expectedReferralReward = {
        id: 42,
        referral_id: VALID_REFERRAL_ID,
        referrer_discord_user_ref: 'ph-discord-user-v1_referrer-old-check',
        referrer_installation_id: 'install-referrer-old-check',
        referred_discord_user_ref: 'ph-discord-user-v1_referred-old-check',
        referred_installation_id: 'install-referred-old-check',
        referred_hardware_hash: 'hardware-old-check',
        referred_hardware_hash_salt_version: 5,
        referred_bonus_status: 'failed',
        referrer_bonus_status: 'failed',
        skip_reason: 'duplicate_hardware',
        failure_reason: 'broker_timeout',
        referred_managed_credential_ref: 'managed-credential-referred-old-check',
        referrer_managed_credential_ref: 'managed-credential-referrer-old-check',
        created_at: '2026-05-20T00:20:00.000Z',
        updated_at: '2026-05-20T00:21:00.000Z',
        credited_at: '2026-05-20T00:22:00.000Z',
        attempt_ip_hash: 'a'.repeat(64),
      };
      const expectedReferralRewardsSequence = 100;
      db.prepare(
        `INSERT INTO referral_rewards (
          id, referral_id, referrer_discord_user_ref, referrer_installation_id,
          referred_discord_user_ref, referred_installation_id, referred_hardware_hash,
          referred_hardware_hash_salt_version, referred_bonus_status, referrer_bonus_status,
          skip_reason, failure_reason, referred_managed_credential_ref,
          referrer_managed_credential_ref, created_at, updated_at, credited_at, attempt_ip_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      ).run(
        expectedReferralReward.id,
        expectedReferralReward.referral_id,
        expectedReferralReward.referrer_discord_user_ref,
        expectedReferralReward.referrer_installation_id,
        expectedReferralReward.referred_discord_user_ref,
        expectedReferralReward.referred_installation_id,
        expectedReferralReward.referred_hardware_hash,
        expectedReferralReward.referred_hardware_hash_salt_version,
        expectedReferralReward.referred_bonus_status,
        expectedReferralReward.referrer_bonus_status,
        expectedReferralReward.skip_reason,
        expectedReferralReward.failure_reason,
        expectedReferralReward.referred_managed_credential_ref,
        expectedReferralReward.referrer_managed_credential_ref,
        expectedReferralReward.created_at,
        expectedReferralReward.updated_at,
        expectedReferralReward.credited_at,
        expectedReferralReward.attempt_ip_hash,
      );
      db.prepare(`UPDATE sqlite_sequence SET seq = ? WHERE name = 'referral_rewards'`).run(
        expectedReferralRewardsSequence,
      );
      expect(sqliteSequence(db, 'referral_rewards')).toBe(expectedReferralRewardsSequence);

      db.exec(readBrokerMigrationSql('0007_simplify_referral_id_checks.sql'));

      const schemas = db
        .prepare(
          `SELECT name, sql FROM sqlite_schema
            WHERE type = 'table'
              AND name IN ('discord_oauth_sessions', 'referral_codes', 'referral_rewards')
            ORDER BY name`,
        )
        .all() as Array<{ name: string; sql: string }>;
      for (const schema of schemas) {
        expect(schema.sql).toContain('length(referral_id) = 6');
        expect(schema.sql).toContain("NOT GLOB '*[^23456789ABCDEFGHJKMNPQRSTUVWXYZ]*'");
        expect(schema.sql).not.toContain(
          "GLOB '[23456789ABCDEFGHJKMNPQRSTUVWXYZ][23456789ABCDEFGHJKMNPQRSTUVWXYZ]",
        );
      }
      expect(countRows(db, 'discord_oauth_sessions')).toBe(1);
      expect(countRows(db, 'referral_codes')).toBe(1);
      expect(countRows(db, 'referral_rewards')).toBe(1);
      expect(
        db
          .prepare(
            `SELECT state_hash, installation_id, device_public_key, redirect_uri,
                    pkce_code_verifier, issue_nonce_hash, fingerprint_salt_version,
                    discord_user_ref, discord_email_verified, discord_account_created_at,
                    eligibility_checked_at, status, created_at, expires_at,
                    processing_started_at, consumed_at, referral_id
               FROM discord_oauth_sessions
              WHERE state_hash = ?`,
          )
          .get(expectedSession.state_hash),
      ).toEqual(expectedSession);
      expect(
        db
          .prepare(
            `SELECT referral_id, owner_discord_user_ref, owner_installation_id, status,
                    created_at, updated_at, disabled_reason, disabled_by, disabled_at
               FROM referral_codes
              WHERE referral_id = ?`,
          )
          .get(expectedReferralCode.referral_id),
      ).toEqual(expectedReferralCode);
      expect(
        db
          .prepare(
            `SELECT id, referral_id, referrer_discord_user_ref, referrer_installation_id,
                    referred_discord_user_ref, referred_installation_id, referred_hardware_hash,
                    referred_hardware_hash_salt_version, referred_bonus_status,
                    referrer_bonus_status, skip_reason, failure_reason,
                    referred_managed_credential_ref, referrer_managed_credential_ref,
                    created_at, updated_at, credited_at, attempt_ip_hash
               FROM referral_rewards
              WHERE id = ?`,
          )
          .get(expectedReferralReward.id),
      ).toEqual(expectedReferralReward);
      expect(sqliteSequence(db, 'referral_rewards')).toBe(expectedReferralRewardsSequence);
      db.prepare(
        `INSERT INTO referral_rewards (
          referral_id, referrer_discord_user_ref, referrer_installation_id,
          referred_discord_user_ref, referred_installation_id, referred_hardware_hash,
          referred_hardware_hash_salt_version, referred_bonus_status, referrer_bonus_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'failed', 'failed')`,
      ).run(
        VALID_REFERRAL_ID,
        'ph-discord-user-v1_referrer-after-repair',
        'install-referrer-after-repair',
        'ph-discord-user-v1_referred-after-repair',
        'install-referred-after-repair',
        'hardware-after-repair',
        5,
      );
      const nextRewardId = Number(
        (db.prepare('SELECT last_insert_rowid() AS id').get() as { id: number | bigint }).id,
      );
      expect(nextRewardId).toBe(expectedReferralRewardsSequence + 1);
      expect(indexNames(db, 'referral_rewards')).toEqual(
        expect.arrayContaining([
          'idx_referral_rewards_referral_id',
          'idx_referral_rewards_referrer_cap',
          'idx_referral_rewards_counted_referred_discord_user',
          'idx_referral_rewards_counted_referred_installation',
          'idx_referral_rewards_attempt_installation_time',
          'idx_referral_rewards_attempt_ip_hash_time',
          'idx_referral_rewards_referral_velocity',
          'idx_referral_rewards_referrer_velocity',
        ]),
      );
    } finally {
      db.close();
    }
  });

  it('enforces Referral ID shape, owned-code status values, and nullable OAuth session referral input', () => {
    withMigratedDatabase((db) => {
      const insertReferralCode = db.prepare(
        `INSERT INTO referral_codes (
          referral_id,
          owner_source,
          owner_subject_ref,
          owner_installation_id,
          status
        ) VALUES (?, 'discord', ?, ?, ?)`,
      );

      expect(() =>
        insertReferralCode.run(
          VALID_REFERRAL_ID,
          'ph-discord-user-v1_owner-valid',
          'owner-installation-valid',
          'active',
        ),
      ).not.toThrow();
      expect(() =>
        insertReferralCode.run(
          '7KO9M2',
          'ph-discord-user-v1_owner-confusing-o',
          null,
          'active',
        ),
      ).toThrow(/constraint/i);
      expect(() =>
        insertReferralCode.run(
          '7KQ9M',
          'ph-discord-user-v1_owner-short',
          null,
          'active',
        ),
      ).toThrow(/constraint/i);
      expect(() =>
        insertReferralCode.run(
          '7kq9m2',
          'ph-discord-user-v1_owner-lowercase',
          null,
          'active',
        ),
      ).toThrow(/constraint/i);
      expect(() =>
        insertReferralCode.run(
          SECOND_VALID_REFERRAL_ID,
          'ph-discord-user-v1_owner-invalid-status',
          null,
          'archived',
        ),
      ).toThrow(/constraint/i);

      const insertSession = db.prepare(
        `INSERT INTO discord_oauth_sessions (
          state_hash,
          installation_id,
          device_public_key,
          redirect_uri,
          pkce_code_verifier,
          issue_nonce_hash,
          fingerprint_salt_version,
          referral_id,
          status,
          created_at,
          expires_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)`,
      );

      expect(() =>
        insertSession.run(
          'state-hash-valid-referral',
          'install-session-valid-referral',
          'device-public-key-valid-referral',
          'http://127.0.0.1:62187/discord/callback',
          'pkce-code-verifier-valid-referral',
          'issue-nonce-valid-referral',
          7,
          VALID_REFERRAL_ID,
          '2026-05-14T06:00:00.000Z',
          '2026-05-14T06:05:00.000Z',
        ),
      ).not.toThrow();
      expect(() =>
        insertSession.run(
          'state-hash-null-referral',
          'install-session-null-referral',
          'device-public-key-null-referral',
          'http://127.0.0.1:62187/discord/callback',
          'pkce-code-verifier-null-referral',
          'issue-nonce-null-referral',
          7,
          null,
          '2026-05-14T06:00:00.000Z',
          '2026-05-14T06:05:00.000Z',
        ),
      ).not.toThrow();
      expect(() =>
        insertSession.run(
          'state-hash-invalid-referral',
          'install-session-invalid-referral',
          'device-public-key-invalid-referral',
          'http://127.0.0.1:62187/discord/callback',
          'pkce-code-verifier-invalid-referral',
          'issue-nonce-invalid-referral',
          7,
          '7KO9M2',
          '2026-05-14T06:00:00.000Z',
          '2026-05-14T06:05:00.000Z',
        ),
      ).toThrow(/constraint/i);
    });
  });

  it('enforces reward status/reason bounds and partial unique counted-referral constraints', () => {
    withMigratedDatabase((db) => {
      const insertReward = db.prepare(
        `INSERT INTO referral_rewards (
          referral_id,
          referrer_source,
          referrer_subject_ref,
          referrer_installation_id,
          referred_source,
          referred_subject_ref,
          referred_installation_id,
          referred_hardware_hash,
          referred_hardware_hash_salt_version,
          referred_bonus_status,
          referrer_bonus_status,
          skip_reason,
          failure_reason,
          referred_managed_credential_ref,
          referrer_managed_credential_ref
        ) VALUES (?1, CASE WHEN ?2 IS NULL THEN NULL ELSE 'discord' END, ?2, ?3, 'discord', ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)`,
      );

      expect(() =>
        insertReward.run(
          VALID_REFERRAL_ID,
          'ph-discord-user-v1_referrer-a',
          'install-referrer-a',
          'ph-discord-user-v1_referred-a',
          'install-referred-a',
          'hardware-hash-a',
          7,
          'reserved',
          'pending',
          null,
          null,
          'managed-credential-referred-a',
          'managed-credential-referrer-a',
        ),
      ).not.toThrow();

      expect(() =>
        insertReward.run(
          '7KO9M2',
          'ph-discord-user-v1_referrer-invalid-id',
          'install-referrer-invalid-id',
          'ph-discord-user-v1_referred-invalid-id',
          'install-referred-invalid-id',
          'hardware-hash-invalid-id',
          7,
          'skipped',
          'skipped',
          'unknown_referral_id',
          null,
          null,
          null,
        ),
      ).toThrow(/constraint/i);
      expect(() =>
        insertReward.run(
          VALID_REFERRAL_ID,
          'ph-discord-user-v1_referrer-invalid-status',
          'install-referrer-invalid-status',
          'ph-discord-user-v1_referred-invalid-status',
          'install-referred-invalid-status',
          'hardware-hash-invalid-status',
          7,
          'queued',
          'pending',
          null,
          null,
          null,
          null,
        ),
      ).toThrow(/constraint/i);
      expect(() =>
        insertReward.run(
          VALID_REFERRAL_ID,
          'ph-discord-user-v1_referrer-long-reason',
          'install-referrer-long-reason',
          'ph-discord-user-v1_referred-long-reason',
          'install-referred-long-reason',
          'hardware-hash-long-reason',
          7,
          'skipped',
          'skipped',
          'x'.repeat(65),
          null,
          null,
          null,
        ),
      ).toThrow(/constraint/i);

      expect(() =>
        insertReward.run(
          VALID_REFERRAL_ID,
          'ph-discord-user-v1_referrer-duplicate-discord',
          'install-referrer-duplicate-discord',
          'ph-discord-user-v1_referred-a',
          'install-referred-b',
          'hardware-hash-b',
          7,
          'credited',
          'credited',
          null,
          null,
          'managed-credential-referred-b',
          'managed-credential-referrer-duplicate-discord',
        ),
      ).toThrow(/unique|constraint/i);
      expect(() =>
        insertReward.run(
          VALID_REFERRAL_ID,
          'ph-discord-user-v1_referrer-duplicate-installation',
          'install-referrer-duplicate-installation',
          'ph-discord-user-v1_referred-b',
          'install-referred-a',
          'hardware-hash-c',
          7,
          'reserved',
          'pending',
          null,
          null,
          'managed-credential-referred-c',
          'managed-credential-referrer-duplicate-installation',
        ),
      ).toThrow(/unique|constraint/i);

      expect(() =>
        insertReward.run(
          VALID_REFERRAL_ID,
          null,
          null,
          'ph-discord-user-v1_referred-a',
          'install-referred-a',
          'hardware-hash-skipped',
          7,
          'skipped',
          'skipped',
          'unknown_referral_id',
          null,
          null,
          null,
        ),
      ).not.toThrow();
    });
  });

  it('does not cascade-delete referral code or reward ledger history when installations age out', () => {
    withMigratedDatabase((db) => {
      db.prepare(
        'INSERT INTO installations (installation_id, device_public_key, app_version) VALUES (?, ?, ?)',
      ).run('install-owner-aging-out', 'device-owner-aging-out', '1.0.0');

      db.prepare(
        `INSERT INTO referral_codes (
          referral_id,
          owner_source,
          owner_subject_ref,
          owner_installation_id,
          status
        ) VALUES (?, 'discord', ?, ?, 'active')`,
      ).run(VALID_REFERRAL_ID, 'ph-discord-user-v1_owner-aging-out', 'install-owner-aging-out');

      db.prepare(
        `INSERT INTO referral_rewards (
          referral_id,
          referrer_source,
          referrer_subject_ref,
          referrer_installation_id,
          referred_source,
          referred_subject_ref,
          referred_installation_id,
          referred_hardware_hash,
          referred_hardware_hash_salt_version,
          referred_bonus_status,
          referrer_bonus_status
        ) VALUES (?, 'discord', ?, ?, 'discord', ?, ?, ?, ?, 'credited', 'credited')`,
      ).run(
        VALID_REFERRAL_ID,
        'ph-discord-user-v1_owner-aging-out',
        'install-owner-aging-out',
        'ph-discord-user-v1_referred-aging-out',
        'install-referred-aging-out',
        'hardware-hash-aging-out',
        7,
      );

      const cascadingForeignKeys = db
        .prepare(
          `SELECT table_name, on_delete
             FROM (
               SELECT 'referral_codes' AS table_name, upper(on_delete) AS on_delete
                 FROM pragma_foreign_key_list('referral_codes')
               UNION ALL
               SELECT 'referral_rewards' AS table_name, upper(on_delete) AS on_delete
                 FROM pragma_foreign_key_list('referral_rewards')
             )
            WHERE on_delete = 'CASCADE'`,
        )
        .all();
      expect(cascadingForeignKeys).toEqual([]);

      db.prepare('DELETE FROM installations WHERE installation_id = ?').run(
        'install-owner-aging-out',
      );

      expect(countRows(db, 'referral_codes')).toBe(1);
      expect(countRows(db, 'referral_rewards')).toBe(1);
    });
  });
});

function columnNames(db: DatabaseSync, tableName: string): string[] {
  return (db
    .prepare(`SELECT name FROM pragma_table_info('${tableName}') ORDER BY cid`)
    .all() as Array<{ name: string }>).map((column) => column.name);
}

function indexSql(
  indexes: Array<{ name: string; sql: string | null }>,
  indexName: string,
): string {
  const sql = indexes.find((index) => index.name === indexName)?.sql;
  if (!sql) {
    throw new Error(`missing index SQL for ${indexName}`);
  }
  return sql;
}

function countRows(db: DatabaseSync, tableName: string): number {
  const row = db
    .prepare(`SELECT COUNT(*) AS count FROM ${tableName}`)
    .get() as { count: number };
  return Number(row.count);
}

function indexNames(db: DatabaseSync, tableName: string): string[] {
  return (db
    .prepare(`SELECT name FROM sqlite_schema WHERE type = 'index' AND tbl_name = ? ORDER BY name`)
    .all(tableName) as Array<{ name: string }>).map((row) => row.name);
}

function sqliteSequence(db: DatabaseSync, tableName: string): number | null {
  const row = db
    .prepare(`SELECT seq FROM sqlite_sequence WHERE name = ?`)
    .get(tableName) as { seq: number | bigint } | undefined;
  return row ? Number(row.seq) : null;
}
