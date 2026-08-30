import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  applyReferralRewardRetention,
  disableReferralId,
  reconcileStaleReferralRewards,
  reserveIssueReferralReward as reserveSourceAwareIssueReferralReward,
} from '../src/referral';
import { updateAbuseControls } from './test-support/abuse-controls';
import {
  createTestBrokerEnv,
  insertEntitlement,
  type TestBrokerEnv,
} from './test-support/sqlite-d1';

const NOW_ISO = '2026-04-30T06:00:00.000Z';
const EXPIRES_AT_ISO = '2026-07-30T06:00:00.000Z';
const OLD_ISO = '2026-04-30T05:45:00.000Z';
const VERY_OLD_ISO = '2026-03-01T00:00:00.000Z';
const REFERRAL_ID = '7KQ9M2';
const UNKNOWN_REFERRAL_ID = 'ABCDEF';
const SECOND_UNKNOWN_REFERRAL_ID = 'BCDEFG';
const REFERRER_DISCORD_REF = `ph-discord-user-v1_${'A'.repeat(43)}`;
const REFERRER_INSTALLATION_ID = 'install-referral-ops-referrer';
const REFERRER_MANAGED_CREDENTIAL_REF = 'managed-referral-ops-referrer';
const REFERRED_DISCORD_REF = `ph-discord-user-v1_${'B'.repeat(43)}`;
const REFERRED_INSTALLATION_ID = 'install-referral-ops-referred';
const REFERRED_HARDWARE_HASH = 'raw-hardware-hash-never-in-referral-logs';

let infoSpy: ReturnType<typeof vi.spyOn>;

function reserveIssueReferralReward(
  db: D1Database,
  input: Omit<
    Parameters<typeof reserveSourceAwareIssueReferralReward>[1],
    'referredSource'
  >,
) {
  return reserveSourceAwareIssueReferralReward(db, {
    ...input,
    referredSource: 'discord',
  });
}

describe('referral reward operational hardening', () => {
  beforeEach(() => {
    infoSpy = vi.spyOn(console, 'info').mockImplementation(() => undefined);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('logs bounded redacted referral outcomes without raw hardware, IPs, or provider bodies', async () => {
    const env = createTestBrokerEnv();

    const result = await reserveIssueReferralReward(env.BROKER_DB, {
      referralId: UNKNOWN_REFERRAL_ID,
      referredSubjectRef: REFERRED_DISCORD_REF,
      referredInstallationId: REFERRED_INSTALLATION_ID,
      referredHardwareHash: REFERRED_HARDWARE_HASH,
      referredHardwareHashSaltVersion: 7,
      clientIp: '203.0.113.88',
      nowIso: NOW_ISO,
    });

    expect(result).toEqual({ outcome: 'skipped', reason: 'unknown_referral_id' });
    expect(infoSpy).toHaveBeenCalledWith(
      'referral_reward_outcome',
      expect.objectContaining({
        outcome: 'skipped',
        reason: 'unknown_referral_id',
        referral_id: UNKNOWN_REFERRAL_ID,
        referred_installation_id: REFERRED_INSTALLATION_ID,
      }),
    );
    const serializedLogs = JSON.stringify(infoSpy.mock.calls);
    expect(serializedLogs).not.toContain(REFERRED_HARDWARE_HASH);
    expect(serializedLogs).not.toContain('203.0.113.88');
    expect(serializedLogs).not.toContain('raw OpenRouter provider error');
  });

  it('reconciles stale reserved rows from delivered entitlement state and fails undelivered reservations', async () => {
    const env = createTestBrokerEnv();
    insertActiveReferrer(env);
    insertInstallation(env, {
      installationId: 'install-stale-reserved-delivered',
      devicePublicKey: 'device-stale-reserved-delivered',
      hardwareHash: 'hardware-stale-reserved-delivered',
    });
    insertDiscordIdentity(env, {
      discordUserRef: `ph-discord-user-v1_${'C'.repeat(43)}`,
      installationId: 'install-stale-reserved-delivered',
      status: 'active',
    });
    insertEntitlement(env, {
      installation_id: 'install-stale-reserved-delivered',
      status: 'active',
      budget_usd: 0.09,
      managed_credential_ref: 'managed-stale-reserved-delivered',
      issued_at: NOW_ISO,
      expires_at: EXPIRES_AT_ISO,
      verified_hardware_hash: 'hardware-stale-reserved-delivered',
      verified_hardware_hash_salt_version: 7,
      discord_user_ref: `ph-discord-user-v1_${'C'.repeat(43)}`,
      discord_issue_status: 'active',
      discord_issue_reserved_at: OLD_ISO,
      discord_issue_delivered_at: NOW_ISO,
    });
    insertReferralReward(env, {
      referredSubjectRef: `ph-discord-user-v1_${'C'.repeat(43)}`,
      referredInstallationId: 'install-stale-reserved-delivered',
      referredHardwareHash: 'hardware-stale-reserved-delivered',
      referredBonusStatus: 'reserved',
      referrerBonusStatus: 'pending',
      updatedAt: OLD_ISO,
    });
    insertReferralReward(env, {
      referredSubjectRef: `ph-discord-user-v1_${'D'.repeat(43)}`,
      referredInstallationId: 'install-stale-reserved-undelivered',
      referredHardwareHash: 'hardware-stale-reserved-undelivered',
      referredBonusStatus: 'reserved',
      referrerBonusStatus: 'pending',
      updatedAt: OLD_ISO,
    });
    env.__db
      .prepare(
        `INSERT INTO qq_managed_entitlements (
            qq_subject_ref, status, issue_ref, managed_credential_ref, budget_usd,
            reserved_at, issued_at, expires_at, delivered_at, created_at, updated_at
          ) VALUES (?, 'active', ?, ?, 0.07, ?, ?, ?, ?, ?, ?)`,
      )
      .run(
        `ph-qq-subject-v1_${'Q'.repeat(43)}`,
        'qq-issue-stale-settlement-lease',
        'managed-stale-settlement-lease',
        OLD_ISO,
        OLD_ISO,
        EXPIRES_AT_ISO,
        OLD_ISO,
        OLD_ISO,
        OLD_ISO,
      );
    insertReferralReward(env, {
      referredSource: 'qq',
      referredSubjectRef: `ph-qq-subject-v1_${'Q'.repeat(43)}`,
      referredInstallationId: 'install-stale-settlement-lease',
      referredHardwareHash: null,
      referredBonusStatus: 'reserved',
      referrerBonusStatus: 'pending',
      updatedAt: OLD_ISO,
    });

    await expect(
      reconcileStaleReferralRewards(env.BROKER_DB, {
        nowIso: NOW_ISO,
        staleReservedAfterMinutes: 5,
      }),
    ).resolves.toEqual({
      staleReservedCredited: 1,
      staleReservedFailed: 1,
      staleApplyingRequeued: 0,
    });

    expect(readReferralRewards(env)).toEqual([
      expect.objectContaining({
        referred_installation_id: 'install-stale-reserved-delivered',
        referred_bonus_status: 'credited',
        referrer_bonus_status: 'pending',
        referred_managed_credential_ref: 'managed-stale-reserved-delivered',
        credited_at: NOW_ISO,
        failure_reason: null,
      }),
      expect.objectContaining({
        referred_installation_id: 'install-stale-reserved-undelivered',
        referred_bonus_status: 'failed',
        referrer_bonus_status: 'failed',
        failure_reason: 'stale_reserved_reconciled',
        credited_at: null,
      }),
      expect.objectContaining({
        referred_installation_id: 'install-stale-settlement-lease',
        referred_bonus_status: 'reserved',
        referrer_bonus_status: 'pending',
        failure_reason: null,
        referred_managed_credential_ref: null,
        credited_at: null,
      }),
    ]);
    expect(countCountedRewards(env, REFERRER_DISCORD_REF)).toBe(2);
  });

  it('requeues stale applying referrer rows without claiming provider credit', async () => {
    const env = createTestBrokerEnv();
    insertActiveReferrer(env);
    insertReferralReward(env, {
      referredSubjectRef: `ph-discord-user-v1_${'E'.repeat(43)}`,
      referredInstallationId: 'install-stale-applying-requeue',
      referredHardwareHash: 'hardware-stale-applying-requeue',
      referredBonusStatus: 'credited',
      referrerBonusStatus: 'applying',
      referredManagedCredentialRef: 'managed-stale-applying-referred',
      referrerManagedCredentialRef: REFERRER_MANAGED_CREDENTIAL_REF,
      updatedAt: OLD_ISO,
    });

    await expect(
      reconcileStaleReferralRewards(env.BROKER_DB, {
        nowIso: NOW_ISO,
        staleApplyingAfterMinutes: 5,
      }),
    ).resolves.toEqual({
      staleReservedCredited: 0,
      staleReservedFailed: 0,
      staleApplyingRequeued: 1,
    });

    expect(readReferralRewards(env)).toEqual([
      expect.objectContaining({
        referrer_bonus_status: 'pending',
        referrer_managed_credential_ref: null,
        failure_reason: null,
        credited_at: null,
      }),
    ]);
  });

  it('retains counted reward rows while pruning old skipped and failed referral attempts', async () => {
    const env = createTestBrokerEnv();
    updateAbuseControls(env, (controls) => {
      controls.retention.referralSkippedDays = 7;
      controls.retention.referralFailedDays = 30;
    });
    insertActiveReferrer(env);
    insertReferralReward(env, {
      referredSubjectRef: `ph-discord-user-v1_${'F'.repeat(43)}`,
      referredInstallationId: 'install-retention-skipped',
      referredHardwareHash: 'hardware-retention-skipped',
      referredBonusStatus: 'skipped',
      referrerBonusStatus: 'skipped',
      skipReason: 'unknown_referral_id',
      updatedAt: VERY_OLD_ISO,
    });
    insertReferralReward(env, {
      referredSubjectRef: `ph-discord-user-v1_${'G'.repeat(43)}`,
      referredInstallationId: 'install-retention-failed',
      referredHardwareHash: 'hardware-retention-failed',
      referredBonusStatus: 'failed',
      referrerBonusStatus: 'failed',
      failureReason: 'issue_delivery_failed',
      updatedAt: VERY_OLD_ISO,
    });
    insertReferralReward(env, {
      referredSubjectRef: `ph-discord-user-v1_${'H'.repeat(43)}`,
      referredInstallationId: 'install-retention-reserved',
      referredHardwareHash: 'hardware-retention-reserved',
      referredBonusStatus: 'reserved',
      referrerBonusStatus: 'pending',
      updatedAt: VERY_OLD_ISO,
    });
    insertReferralReward(env, {
      referredSubjectRef: `ph-discord-user-v1_${'J'.repeat(43)}`,
      referredInstallationId: 'install-retention-credited',
      referredHardwareHash: 'hardware-retention-credited',
      referredBonusStatus: 'credited',
      referrerBonusStatus: 'credited',
      referredManagedCredentialRef: 'managed-retention-credited',
      referrerManagedCredentialRef: REFERRER_MANAGED_CREDENTIAL_REF,
      updatedAt: VERY_OLD_ISO,
    });

    await expect(
      applyReferralRewardRetention(env.BROKER_DB, new Date(NOW_ISO)),
    ).resolves.toEqual({ skippedDeleted: 1, failedDeleted: 1 });

    expect(readReferralRewards(env).map((row) => row.referred_installation_id)).toEqual([
      'install-retention-reserved',
      'install-retention-credited',
    ]);
    expect(countCountedRewards(env, REFERRER_DISCORD_REF)).toBe(2);
  });

  it('soft-throttles valid-shaped and repeated unknown Referral ID attempts without throwing', async () => {
    const env = createTestBrokerEnv();
    updateAbuseControls(env, (controls) => {
      controls.referralAttempts.validShaped.maxPerInstallation = 1;
      controls.referralAttempts.validShaped.maxPerIp = 50;
      controls.referralAttempts.unknown.maxPerInstallation = 50;
      controls.referralAttempts.unknown.maxPerIp = 50;
    });

    await expect(
      reserveIssueReferralReward(env.BROKER_DB, {
        referralId: UNKNOWN_REFERRAL_ID,
        referredSubjectRef: REFERRED_DISCORD_REF,
        referredInstallationId: REFERRED_INSTALLATION_ID,
        referredHardwareHash: REFERRED_HARDWARE_HASH,
        referredHardwareHashSaltVersion: 7,
        clientIp: '203.0.113.90',
        nowIso: NOW_ISO,
      }),
    ).resolves.toEqual({ outcome: 'skipped', reason: 'unknown_referral_id' });
    await expect(
      reserveIssueReferralReward(env.BROKER_DB, {
        referralId: SECOND_UNKNOWN_REFERRAL_ID,
        referredSubjectRef: `ph-discord-user-v1_${'K'.repeat(43)}`,
        referredInstallationId: REFERRED_INSTALLATION_ID,
        referredHardwareHash: 'hardware-second-valid-shaped-rate-limit',
        referredHardwareHashSaltVersion: 7,
        clientIp: '203.0.113.90',
        nowIso: NOW_ISO,
      }),
    ).resolves.toEqual({ outcome: 'skipped', reason: 'referral_attempt_rate_limited' });

    updateAbuseControls(env, (controls) => {
      controls.referralAttempts.validShaped.maxPerInstallation = 50;
      controls.referralAttempts.unknown.maxPerInstallation = 1;
    });
    await expect(
      reserveIssueReferralReward(env.BROKER_DB, {
        referralId: SECOND_UNKNOWN_REFERRAL_ID,
        referredSubjectRef: `ph-discord-user-v1_${'L'.repeat(43)}`,
        referredInstallationId: REFERRED_INSTALLATION_ID,
        referredHardwareHash: 'hardware-repeated-unknown-throttle',
        referredHardwareHashSaltVersion: 7,
        clientIp: '203.0.113.90',
        nowIso: NOW_ISO,
      }),
    ).resolves.toEqual({
      outcome: 'skipped',
      reason: 'unknown_referral_id_rate_limited',
    });
  });

  it('tracks per-referral and per-referrer reward velocity before reserving counted rows', async () => {
    const env = createTestBrokerEnv();
    insertActiveReferrer(env);
    updateAbuseControls(env, (controls) => {
      controls.referralAttempts.perReferralIdVelocity.maxAttempts = 1;
      controls.referralAttempts.perReferrerRewardVelocity.maxRewards = 50;
    });
    insertReferralReward(env, {
      referredSubjectRef: `ph-discord-user-v1_${'M'.repeat(43)}`,
      referredInstallationId: 'install-referral-velocity-seed',
      referredHardwareHash: 'hardware-referral-velocity-seed',
      referredBonusStatus: 'skipped',
      referrerBonusStatus: 'skipped',
      skipReason: 'unknown_referral_id',
      updatedAt: NOW_ISO,
    });

    await expect(
      reserveIssueReferralReward(env.BROKER_DB, {
        referralId: REFERRAL_ID,
        referredSubjectRef: REFERRED_DISCORD_REF,
        referredInstallationId: REFERRED_INSTALLATION_ID,
        referredHardwareHash: REFERRED_HARDWARE_HASH,
        referredHardwareHashSaltVersion: 7,
        nowIso: NOW_ISO,
      }),
    ).resolves.toEqual({ outcome: 'skipped', reason: 'referral_velocity_limited' });
    expect(countCountedRewards(env, REFERRER_DISCORD_REF)).toBe(0);

    const referrerLimitedEnv = createTestBrokerEnv();
    insertActiveReferrer(referrerLimitedEnv);
    updateAbuseControls(referrerLimitedEnv, (controls) => {
      controls.referralAttempts.perReferralIdVelocity.maxAttempts = 50;
      controls.referralAttempts.perReferrerRewardVelocity.maxRewards = 1;
    });
    insertReferralReward(referrerLimitedEnv, {
      referredSubjectRef: `ph-discord-user-v1_${'N'.repeat(43)}`,
      referredInstallationId: 'install-referrer-velocity-seed',
      referredHardwareHash: 'hardware-referrer-velocity-seed',
      referredBonusStatus: 'credited',
      referrerBonusStatus: 'credited',
      referredManagedCredentialRef: 'managed-referrer-velocity-seed',
      referrerManagedCredentialRef: REFERRER_MANAGED_CREDENTIAL_REF,
      updatedAt: NOW_ISO,
    });

    await expect(
      reserveIssueReferralReward(referrerLimitedEnv.BROKER_DB, {
        referralId: REFERRAL_ID,
        referredSubjectRef: REFERRED_DISCORD_REF,
        referredInstallationId: REFERRED_INSTALLATION_ID,
        referredHardwareHash: REFERRED_HARDWARE_HASH,
        referredHardwareHashSaltVersion: 7,
        nowIso: NOW_ISO,
      }),
    ).resolves.toEqual({ outcome: 'skipped', reason: 'referrer_velocity_limited' });
    expect(countCountedRewards(referrerLimitedEnv, REFERRER_DISCORD_REF)).toBe(1);
  });

  it('stores bounded operator disable audit metadata and prevents new rewards from disabled IDs', async () => {
    const env = createTestBrokerEnv();
    insertActiveReferrer(env);

    await expect(
      disableReferralId(env.BROKER_DB, {
        referralId: REFERRAL_ID,
        reason: 'abuse',
        disabledBy: 'operator',
        nowIso: NOW_ISO,
      }),
    ).resolves.toEqual({ ok: true, status: 'disabled' });
    await expect(
      disableReferralId(env.BROKER_DB, {
        referralId: REFERRAL_ID,
        reason: 'raw\noperator note' as never,
        disabledBy: 'operator',
        nowIso: NOW_ISO,
      }),
    ).resolves.toEqual({ ok: false, reason: 'invalid_disable_reason' });

    expect(readReferralCode(env, REFERRAL_ID)).toEqual(
      expect.objectContaining({
        status: 'disabled',
        disabled_reason: 'abuse',
        disabled_by: 'operator',
        disabled_at: NOW_ISO,
      }),
    );
    expect(readRuntimeAudit(env)).toEqual([
      expect.objectContaining({
        event_kind: 'referral_id_disabled',
        reason: 'abuse',
        payload: expect.objectContaining({
          referral_id: REFERRAL_ID,
          disabled_by: 'operator',
        }),
      }),
    ]);
    await expect(
      reserveIssueReferralReward(env.BROKER_DB, {
        referralId: REFERRAL_ID,
        referredSubjectRef: REFERRED_DISCORD_REF,
        referredInstallationId: REFERRED_INSTALLATION_ID,
        referredHardwareHash: REFERRED_HARDWARE_HASH,
        referredHardwareHashSaltVersion: 7,
        nowIso: NOW_ISO,
      }),
    ).resolves.toEqual({ outcome: 'skipped', reason: 'disabled_referral_id' });
  });
});

function insertActiveReferrer(env: TestBrokerEnv): void {
  insertInstallation(env, {
    installationId: REFERRER_INSTALLATION_ID,
    devicePublicKey: 'device-referral-ops-referrer',
    hardwareHash: 'hardware-referral-ops-referrer',
  });
  insertDiscordIdentity(env, {
    discordUserRef: REFERRER_DISCORD_REF,
    installationId: REFERRER_INSTALLATION_ID,
    status: 'active',
  });
  insertEntitlement(env, {
    installation_id: REFERRER_INSTALLATION_ID,
    status: 'active',
    budget_usd: 0.07,
    managed_credential_ref: REFERRER_MANAGED_CREDENTIAL_REF,
    issued_at: NOW_ISO,
    expires_at: EXPIRES_AT_ISO,
    verified_hardware_hash: 'hardware-referral-ops-referrer',
    verified_hardware_hash_salt_version: 7,
    discord_user_ref: REFERRER_DISCORD_REF,
    discord_issue_status: 'active',
    discord_issue_reserved_at: NOW_ISO,
    discord_issue_delivered_at: NOW_ISO,
  });
  insertReferralCode(env, {
    referralId: REFERRAL_ID,
    ownerSubjectRef: REFERRER_DISCORD_REF,
    ownerInstallationId: REFERRER_INSTALLATION_ID,
  });
}

function insertInstallation(
  env: TestBrokerEnv,
  input: {
    installationId: string;
    devicePublicKey: string;
    hardwareHash: string | null;
    hardwareHashSaltVersion?: number | null;
  },
): void {
  env.__db
    .prepare(
      `INSERT INTO installations (
          installation_id,
          device_public_key,
          hardware_hash,
          hardware_hash_salt_version,
          app_version,
          created_at,
          last_seen_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)`,
    )
    .run(
      input.installationId,
      input.devicePublicKey,
      input.hardwareHash,
      input.hardwareHashSaltVersion ?? 7,
      '1.2.3',
      NOW_ISO,
      NOW_ISO,
    );
}

function insertDiscordIdentity(
  env: TestBrokerEnv,
  input: {
    discordUserRef: string;
    installationId: string;
    status: 'issuing' | 'active' | 'failed' | 'cleanup_required';
  },
): void {
  env.__db
    .prepare(
      `INSERT INTO discord_identities (
          discord_user_ref,
          entitlement_installation_id,
          status,
          ref_secret_version,
          created_at,
          updated_at
        ) VALUES (?, ?, ?, 1, ?, ?)`,
    )
    .run(input.discordUserRef, input.installationId, input.status, NOW_ISO, NOW_ISO);
}

function insertReferralCode(
  env: TestBrokerEnv,
  input: {
    referralId: string;
    ownerSubjectRef: string;
    ownerInstallationId: string | null;
    status?: 'active' | 'disabled';
  },
): void {
  env.__db
    .prepare(
      `INSERT INTO referral_codes (
          referral_id,
          owner_source,
          owner_subject_ref,
          owner_installation_id,
          status,
          created_at,
          updated_at
        ) VALUES (?, 'discord', ?, ?, ?, ?, ?)`,
    )
    .run(
      input.referralId,
      input.ownerSubjectRef,
      input.ownerInstallationId,
      input.status ?? 'active',
      NOW_ISO,
      NOW_ISO,
    );
}

function insertReferralReward(
  env: TestBrokerEnv,
  input: {
    referredSource?: 'discord' | 'qq';
    referredSubjectRef: string;
    referredInstallationId: string;
    referredHardwareHash: string | null;
    referredBonusStatus: 'reserved' | 'credited' | 'skipped' | 'failed';
    referrerBonusStatus: 'pending' | 'applying' | 'credited' | 'skipped' | 'failed';
    skipReason?: string | null;
    failureReason?: string | null;
    referredManagedCredentialRef?: string | null;
    referrerManagedCredentialRef?: string | null;
    updatedAt?: string;
  },
): void {
  const createdAt = input.updatedAt ?? NOW_ISO;
  env.__db
    .prepare(
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
          referrer_managed_credential_ref,
          created_at,
          updated_at
        ) VALUES (?, 'discord', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    )
    .run(
      REFERRAL_ID,
      REFERRER_DISCORD_REF,
      REFERRER_INSTALLATION_ID,
      input.referredSource ?? 'discord',
      input.referredSubjectRef,
      input.referredInstallationId,
      input.referredHardwareHash,
      input.referredSource === 'qq' ? null : 7,
      input.referredBonusStatus,
      input.referrerBonusStatus,
      input.skipReason ?? null,
      input.failureReason ?? null,
      input.referredManagedCredentialRef ?? null,
      input.referrerManagedCredentialRef ?? null,
      createdAt,
      input.updatedAt ?? NOW_ISO,
    );
}

function readReferralRewards(env: TestBrokerEnv): Array<{
  referred_installation_id: string;
  referred_bonus_status: string;
  referrer_bonus_status: string;
  failure_reason: string | null;
  referred_managed_credential_ref: string | null;
  referrer_managed_credential_ref: string | null;
  credited_at: string | null;
}> {
  return env.__db
    .prepare(
      `SELECT referred_installation_id,
              referred_bonus_status,
              referrer_bonus_status,
              failure_reason,
              referred_managed_credential_ref,
              referrer_managed_credential_ref,
              credited_at
         FROM referral_rewards
        ORDER BY id ASC`,
    )
    .all() as Array<{
    referred_installation_id: string;
    referred_bonus_status: string;
    referrer_bonus_status: string;
    failure_reason: string | null;
    referred_managed_credential_ref: string | null;
    referrer_managed_credential_ref: string | null;
    credited_at: string | null;
  }>;
}

function readReferralCode(env: TestBrokerEnv, referralId: string): Record<string, unknown> | null {
  return (
    (env.__db
      .prepare(
        `SELECT referral_id,
                status,
                disabled_reason,
                disabled_by,
                disabled_at
           FROM referral_codes
          WHERE referral_id = ?`,
      )
      .get(referralId) as Record<string, unknown> | undefined) ?? null
  );
}

function readRuntimeAudit(env: TestBrokerEnv): Array<{
  event_kind: string;
  reason: string | null;
  payload: Record<string, unknown>;
}> {
  return (
    env.__db
      .prepare(
        `SELECT event_kind, reason, payload_json
           FROM broker_abuse_runtime_audit
          ORDER BY id ASC`,
      )
      .all() as Array<{ event_kind: string; reason: string | null; payload_json: string }>
  ).map((row) => ({
    event_kind: row.event_kind,
    reason: row.reason,
    payload: JSON.parse(row.payload_json) as Record<string, unknown>,
  }));
}

function countCountedRewards(env: TestBrokerEnv, referrerSubjectRef: string): number {
  const row = env.__db
    .prepare(
      `SELECT COUNT(*) AS count
         FROM referral_rewards
        WHERE referrer_subject_ref = ?
          AND referred_bonus_status IN ('reserved', 'credited')`,
    )
    .get(referrerSubjectRef) as { count: number };
  return Number(row.count);
}
