import { describe, expect, it } from 'vitest';

import {
  ensureOwnedReferralIdForActiveDiscordManagedUser,
  generateReferralId,
  normalizeReferralId,
  REFERRAL_ID_ALPHABET,
} from '../src/referral';
import {
  createTestBrokerEnv,
  insertEntitlement,
  type TestBrokerEnv,
} from './test-support/sqlite-d1';

const NOW_ISO = '2026-05-14T06:00:00.000Z';
const VALID_REFERRAL_ID = '7KQ9M2';
const SECOND_VALID_REFERRAL_ID = 'ABCDEF';
const SAFE_DISCORD_USER_REF = `ph-discord-user-v1_${'A'.repeat(43)}`;
const SECOND_SAFE_DISCORD_USER_REF = `ph-discord-user-v1_${'B'.repeat(43)}`;

describe('Referral ID lifecycle primitives', () => {
  it('normalizes valid Referral IDs case-insensitively and rejects invalid inputs', () => {
    expect(normalizeReferralId(' 7kq9m2 ')).toBe(VALID_REFERRAL_ID);
    expect(normalizeReferralId('abcdef')).toBe(SECOND_VALID_REFERRAL_ID);

    for (const value of [
      undefined,
      null,
      123456,
      {},
      '',
      '   ',
      '7KQ9M',
      '7KQ9M22',
      '7KQ-M2',
      '7KO9M2',
      '7KQ9I2',
      '7KQ9L2',
      '7KQ902',
      '7KQ912',
    ]) {
      expect(normalizeReferralId(value)).toBeNull();
    }
  });

  it('generates six approved-alphabet characters while skipping biased random bytes', () => {
    const randomBytes = bytesFrom([248, 249, 250, 0, 1, 2, 3, 4, 5]);

    const referralId = generateReferralId(randomBytes);

    expect(referralId).toBe('234567');
    expect(referralId).toHaveLength(6);
    expect([...referralId].every((character) => REFERRAL_ID_ALPHABET.includes(character))).toBe(
      true,
    );
  });

  it('retries ID collisions and creates only one active code per Discord user reference', async () => {
    const env = createTestBrokerEnv();
    insertInstallation(env, 'install-owned-collision');
    insertInstallation(env, 'install-owned-collision-existing');
    insertActiveDiscordManagedEntitlement(env, {
      installationId: 'install-owned-collision',
      discordUserRef: SAFE_DISCORD_USER_REF,
      managedCredentialRef: 'managed-owned-collision',
    });
    insertReferralCode(env, {
      referralId: VALID_REFERRAL_ID,
      discordUserRef: SECOND_SAFE_DISCORD_USER_REF,
      installationId: 'install-owned-collision-existing',
      status: 'active',
    });
    const generatedIds = generatorFrom([
      VALID_REFERRAL_ID,
      SECOND_VALID_REFERRAL_ID,
      '234567',
    ]);

    const firstResult = await ensureOwnedReferralIdForActiveDiscordManagedUser(
      env.BROKER_DB,
      {
        installationId: 'install-owned-collision',
        nowIso: NOW_ISO,
        generateReferralId: generatedIds.next,
      },
    );
    const secondResult = await ensureOwnedReferralIdForActiveDiscordManagedUser(
      env.BROKER_DB,
      {
        installationId: 'install-owned-collision',
        nowIso: NOW_ISO,
        generateReferralId: generatedIds.next,
      },
    );

    expect(firstResult).toMatchObject({
      ok: true,
      created: true,
      referralCode: {
        referral_id: SECOND_VALID_REFERRAL_ID,
        owner_discord_user_ref: SAFE_DISCORD_USER_REF,
        owner_installation_id: 'install-owned-collision',
        status: 'active',
      },
    });
    expect(secondResult).toMatchObject({
      ok: true,
      created: false,
      referralCode: {
        referral_id: SECOND_VALID_REFERRAL_ID,
        owner_discord_user_ref: SAFE_DISCORD_USER_REF,
        status: 'active',
      },
    });
    expect(generatedIds.callCount()).toBe(2);
    expect(countReferralCodes(env, SAFE_DISCORD_USER_REF)).toBe(1);
    expect(countActiveReferralCodes(env, SAFE_DISCORD_USER_REF)).toBe(1);
  });

  it('lazily creates an owned Referral ID only for active Discord-managed users', async () => {
    const env = createTestBrokerEnv();
    insertInstallation(env, 'install-owned-active');
    insertInstallation(env, 'install-owned-pending');
    insertInstallation(env, 'install-owned-no-discord');
    insertActiveDiscordManagedEntitlement(env, {
      installationId: 'install-owned-active',
      discordUserRef: SAFE_DISCORD_USER_REF,
      managedCredentialRef: 'managed-owned-active',
    });
    insertDiscordIdentity(env, {
      discordUserRef: SECOND_SAFE_DISCORD_USER_REF,
      installationId: 'install-owned-pending',
      status: 'issuing',
    });
    insertEntitlement(env, {
      installation_id: 'install-owned-pending',
      status: 'pending_release',
      budget_usd: 0.07,
      discord_user_ref: SECOND_SAFE_DISCORD_USER_REF,
      discord_issue_status: 'issuing',
    });
    insertEntitlement(env, {
      installation_id: 'install-owned-no-discord',
      status: 'active',
      budget_usd: 0.07,
      managed_credential_ref: 'managed-owned-no-discord',
      issued_at: NOW_ISO,
      expires_at: '2026-08-14T06:00:00.000Z',
      discord_user_ref: null,
      discord_issue_status: null,
    });

    await expect(
      ensureOwnedReferralIdForActiveDiscordManagedUser(env.BROKER_DB, {
        installationId: 'install-owned-active',
        nowIso: NOW_ISO,
        generateReferralId: () => VALID_REFERRAL_ID,
      }),
    ).resolves.toMatchObject({
      ok: true,
      created: true,
      referralCode: {
        referral_id: VALID_REFERRAL_ID,
        owner_discord_user_ref: SAFE_DISCORD_USER_REF,
        owner_installation_id: 'install-owned-active',
        status: 'active',
      },
    });
    await expect(
      ensureOwnedReferralIdForActiveDiscordManagedUser(env.BROKER_DB, {
        installationId: 'install-owned-pending',
        nowIso: NOW_ISO,
        generateReferralId: () => SECOND_VALID_REFERRAL_ID,
      }),
    ).resolves.toEqual({ ok: false, reason: 'not_eligible' });
    await expect(
      ensureOwnedReferralIdForActiveDiscordManagedUser(env.BROKER_DB, {
        installationId: 'install-owned-no-discord',
        nowIso: NOW_ISO,
        generateReferralId: () => SECOND_VALID_REFERRAL_ID,
      }),
    ).resolves.toEqual({ ok: false, reason: 'not_eligible' });
    expect(countAllReferralCodes(env)).toBe(1);
  });

  it('requires a delivered active Discord-managed entitlement and matching active identity', async () => {
    const env = createTestBrokerEnv();
    const ineligibleCases = [
      {
        installationId: 'install-owned-expired-entitlement',
        discordUserRef: `ph-discord-user-v1_${'C'.repeat(43)}`,
        entitlement: {
          managedCredentialRef: 'managed-owned-expired-entitlement',
          expiresAt: '2026-05-14T05:59:59.000Z',
          discordIssueStatus: 'active' as const,
          deliveredAt: NOW_ISO,
        },
        identity: { status: 'active' as const, installationId: 'install-owned-expired-entitlement' },
      },
      {
        installationId: 'install-owned-missing-credential',
        discordUserRef: `ph-discord-user-v1_${'D'.repeat(43)}`,
        entitlement: {
          managedCredentialRef: null,
          expiresAt: '2026-08-14T06:00:00.000Z',
          discordIssueStatus: 'active' as const,
          deliveredAt: NOW_ISO,
        },
        identity: { status: 'active' as const, installationId: 'install-owned-missing-credential' },
      },
      {
        installationId: 'install-owned-missing-expiry',
        discordUserRef: `ph-discord-user-v1_${'J'.repeat(43)}`,
        entitlement: {
          managedCredentialRef: 'managed-owned-missing-expiry',
          expiresAt: null,
          discordIssueStatus: 'active' as const,
          deliveredAt: NOW_ISO,
        },
        identity: { status: 'active' as const, installationId: 'install-owned-missing-expiry' },
      },
      {
        installationId: 'install-owned-inactive-issue',
        discordUserRef: `ph-discord-user-v1_${'E'.repeat(43)}`,
        entitlement: {
          managedCredentialRef: 'managed-owned-inactive-issue',
          expiresAt: '2026-08-14T06:00:00.000Z',
          discordIssueStatus: 'failed' as const,
          deliveredAt: NOW_ISO,
        },
        identity: { status: 'active' as const, installationId: 'install-owned-inactive-issue' },
      },
      {
        installationId: 'install-owned-missing-delivery',
        discordUserRef: `ph-discord-user-v1_${'F'.repeat(43)}`,
        entitlement: {
          managedCredentialRef: 'managed-owned-missing-delivery',
          expiresAt: '2026-08-14T06:00:00.000Z',
          discordIssueStatus: 'active' as const,
          deliveredAt: null,
        },
        identity: { status: 'active' as const, installationId: 'install-owned-missing-delivery' },
      },
      {
        installationId: 'install-owned-inactive-identity',
        discordUserRef: `ph-discord-user-v1_${'G'.repeat(43)}`,
        entitlement: {
          managedCredentialRef: 'managed-owned-inactive-identity',
          expiresAt: '2026-08-14T06:00:00.000Z',
          discordIssueStatus: 'active' as const,
          deliveredAt: NOW_ISO,
        },
        identity: { status: 'failed' as const, installationId: 'install-owned-inactive-identity' },
      },
      {
        installationId: 'install-owned-mismatched-identity',
        discordUserRef: `ph-discord-user-v1_${'H'.repeat(43)}`,
        entitlement: {
          managedCredentialRef: 'managed-owned-mismatched-identity',
          expiresAt: '2026-08-14T06:00:00.000Z',
          discordIssueStatus: 'active' as const,
          deliveredAt: NOW_ISO,
        },
        identity: {
          status: 'active' as const,
          installationId: 'install-owned-mismatched-identity-other',
        },
      },
    ];
    let generatorCalls = 0;

    for (const testCase of ineligibleCases) {
      insertInstallation(env, testCase.installationId);
      if (testCase.identity.installationId !== testCase.installationId) {
        insertInstallation(env, testCase.identity.installationId);
      }
      insertDiscordIdentity(env, {
        discordUserRef: testCase.discordUserRef,
        installationId: testCase.identity.installationId,
        status: testCase.identity.status,
      });
      insertEntitlement(env, {
        installation_id: testCase.installationId,
        status: 'active',
        budget_usd: 0.07,
        managed_credential_ref: testCase.entitlement.managedCredentialRef,
        issued_at: NOW_ISO,
        expires_at: testCase.entitlement.expiresAt,
        discord_user_ref: testCase.discordUserRef,
        discord_issue_status: testCase.entitlement.discordIssueStatus,
        discord_issue_delivered_at: testCase.entitlement.deliveredAt,
      });

      await expect(
        ensureOwnedReferralIdForActiveDiscordManagedUser(env.BROKER_DB, {
          installationId: testCase.installationId,
          nowIso: NOW_ISO,
          generateReferralId: () => {
            generatorCalls += 1;
            return VALID_REFERRAL_ID;
          },
        }),
      ).resolves.toEqual({ ok: false, reason: 'not_eligible' });
    }

    expect(generatorCalls).toBe(0);
    expect(countAllReferralCodes(env)).toBe(0);
  });

  it('does not reactivate disabled owned codes or persist unsafe raw Discord identity values', async () => {
    const env = createTestBrokerEnv();
    insertInstallation(env, 'install-owned-disabled');
    insertInstallation(env, 'install-owned-unsafe-ref');
    insertActiveDiscordManagedEntitlement(env, {
      installationId: 'install-owned-disabled',
      discordUserRef: SAFE_DISCORD_USER_REF,
      managedCredentialRef: 'managed-owned-disabled',
    });
    insertActiveDiscordManagedEntitlement(env, {
      installationId: 'install-owned-unsafe-ref',
      discordUserRef: 'raw-discord-user-123456789012345678:user@example.test',
      managedCredentialRef: 'managed-owned-unsafe-ref',
    });
    insertReferralCode(env, {
      referralId: VALID_REFERRAL_ID,
      discordUserRef: SAFE_DISCORD_USER_REF,
      installationId: 'install-owned-disabled',
      status: 'disabled',
    });

    await expect(
      ensureOwnedReferralIdForActiveDiscordManagedUser(env.BROKER_DB, {
        installationId: 'install-owned-disabled',
        nowIso: NOW_ISO,
        generateReferralId: () => SECOND_VALID_REFERRAL_ID,
      }),
    ).resolves.toEqual({ ok: false, reason: 'disabled' });
    await expect(
      ensureOwnedReferralIdForActiveDiscordManagedUser(env.BROKER_DB, {
        installationId: 'install-owned-unsafe-ref',
        nowIso: NOW_ISO,
        generateReferralId: () => SECOND_VALID_REFERRAL_ID,
      }),
    ).resolves.toEqual({ ok: false, reason: 'unsafe_discord_user_ref' });

    expect(countActiveReferralCodes(env, SAFE_DISCORD_USER_REF)).toBe(0);
    expect(countAllReferralCodes(env)).toBe(1);
    expect(JSON.stringify(readReferralCodes(env))).not.toContain('user@example.test');
    expect(JSON.stringify(readReferralCodes(env))).not.toContain('123456789012345678');
  });

  it('does not return ok when an active owned code is disabled during owner refresh', async () => {
    let env: TestBrokerEnv;
    let disabledDuringRefresh = false;
    env = createTestBrokerEnv({
      beforeRun: ({ sql }) => {
        if (disabledDuringRefresh || !sql.includes('UPDATE referral_codes')) {
          return;
        }

        disabledDuringRefresh = true;
        env.__db
          .prepare(
            `UPDATE referral_codes
                SET status = 'disabled',
                    updated_at = ?
              WHERE owner_discord_user_ref = ?`,
          )
          .run(NOW_ISO, SAFE_DISCORD_USER_REF);
      },
    });
    insertInstallation(env, 'install-owned-disabled-race');
    insertActiveDiscordManagedEntitlement(env, {
      installationId: 'install-owned-disabled-race',
      discordUserRef: SAFE_DISCORD_USER_REF,
      managedCredentialRef: 'managed-owned-disabled-race',
    });
    insertReferralCode(env, {
      referralId: VALID_REFERRAL_ID,
      discordUserRef: SAFE_DISCORD_USER_REF,
      installationId: 'install-owned-disabled-race-old-installation',
      status: 'active',
    });

    await expect(
      ensureOwnedReferralIdForActiveDiscordManagedUser(env.BROKER_DB, {
        installationId: 'install-owned-disabled-race',
        nowIso: NOW_ISO,
        generateReferralId: () => SECOND_VALID_REFERRAL_ID,
      }),
    ).resolves.toEqual({ ok: false, reason: 'disabled' });

    expect(disabledDuringRefresh).toBe(true);
    expect(countActiveReferralCodes(env, SAFE_DISCORD_USER_REF)).toBe(0);
  });

  it('reports collision exhaustion without creating an owner code after bounded retries', async () => {
    const env = createTestBrokerEnv();
    insertInstallation(env, 'install-owned-collision-exhausted');
    insertInstallation(env, 'install-owned-collision-exhausted-existing');
    insertActiveDiscordManagedEntitlement(env, {
      installationId: 'install-owned-collision-exhausted',
      discordUserRef: SAFE_DISCORD_USER_REF,
      managedCredentialRef: 'managed-owned-collision-exhausted',
    });
    insertReferralCode(env, {
      referralId: VALID_REFERRAL_ID,
      discordUserRef: SECOND_SAFE_DISCORD_USER_REF,
      installationId: 'install-owned-collision-exhausted-existing',
      status: 'active',
    });
    const generatedIds = generatorFrom([VALID_REFERRAL_ID, VALID_REFERRAL_ID]);

    await expect(
      ensureOwnedReferralIdForActiveDiscordManagedUser(env.BROKER_DB, {
        installationId: 'install-owned-collision-exhausted',
        nowIso: NOW_ISO,
        generateReferralId: generatedIds.next,
        maxCollisionAttempts: 2,
      }),
    ).resolves.toEqual({ ok: false, reason: 'collision_exhausted' });

    expect(generatedIds.callCount()).toBe(2);
    expect(countReferralCodes(env, SAFE_DISCORD_USER_REF)).toBe(0);
    expect(countAllReferralCodes(env)).toBe(1);
  });
});

function bytesFrom(values: number[]): (byteLength: number) => Uint8Array {
  let offset = 0;
  return (byteLength: number) => {
    const chunk = values.slice(offset, offset + byteLength);
    offset += byteLength;
    return Uint8Array.from(chunk);
  };
}

function generatorFrom(values: string[]): {
  next: () => string;
  callCount: () => number;
} {
  let offset = 0;
  return {
    next: () => {
      const value = values[offset];
      offset += 1;
      if (!value) {
        throw new Error('test Referral ID generator exhausted');
      }
      return value;
    },
    callCount: () => offset,
  };
}

function insertInstallation(env: TestBrokerEnv, installationId: string): void {
  env.__db
    .prepare(
      `INSERT INTO installations (
          installation_id,
          device_public_key,
          app_version,
          created_at,
          last_seen_at
        ) VALUES (?, ?, ?, ?, ?)`,
    )
    .run(
      installationId,
      `device-public-key-${installationId}`,
      '1.2.3',
      NOW_ISO,
      NOW_ISO,
    );
}

function insertActiveDiscordManagedEntitlement(
  env: TestBrokerEnv,
  input: {
    installationId: string;
    discordUserRef: string;
    managedCredentialRef: string;
  },
): void {
  insertDiscordIdentity(env, {
    discordUserRef: input.discordUserRef,
    installationId: input.installationId,
    status: 'active',
  });
  insertEntitlement(env, {
    installation_id: input.installationId,
    status: 'active',
    budget_usd: 0.07,
    managed_credential_ref: input.managedCredentialRef,
    issued_at: NOW_ISO,
    expires_at: '2026-08-14T06:00:00.000Z',
    discord_user_ref: input.discordUserRef,
    discord_issue_status: 'active',
    discord_issue_delivered_at: NOW_ISO,
  });
}

function insertDiscordIdentity(
  env: TestBrokerEnv,
  input: {
    discordUserRef: string;
    installationId: string;
    status: 'issuing' | 'delivery_pending' | 'active' | 'failed' | 'cleanup_required';
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
        ) VALUES (?, ?, ?, ?, ?, ?)`,
    )
    .run(
      input.discordUserRef,
      input.installationId,
      input.status,
      1,
      NOW_ISO,
      NOW_ISO,
    );
}

function insertReferralCode(
  env: TestBrokerEnv,
  input: {
    referralId: string;
    discordUserRef: string;
    installationId: string;
    status: 'active' | 'disabled';
  },
): void {
  env.__db
    .prepare(
      `INSERT INTO referral_codes (
          referral_id,
          owner_discord_user_ref,
          owner_installation_id,
          status,
          created_at,
          updated_at
        ) VALUES (?, ?, ?, ?, ?, ?)`,
    )
    .run(
      input.referralId,
      input.discordUserRef,
      input.installationId,
      input.status,
      NOW_ISO,
      NOW_ISO,
    );
}

function countReferralCodes(env: TestBrokerEnv, discordUserRef: string): number {
  const row = env.__db
    .prepare(
      `SELECT COUNT(*) AS count
         FROM referral_codes
        WHERE owner_discord_user_ref = ?`,
    )
    .get(discordUserRef) as { count: number };
  return Number(row.count);
}

function countActiveReferralCodes(env: TestBrokerEnv, discordUserRef: string): number {
  const row = env.__db
    .prepare(
      `SELECT COUNT(*) AS count
         FROM referral_codes
        WHERE owner_discord_user_ref = ?
          AND status = 'active'`,
    )
    .get(discordUserRef) as { count: number };
  return Number(row.count);
}

function countAllReferralCodes(env: TestBrokerEnv): number {
  const row = env.__db
    .prepare('SELECT COUNT(*) AS count FROM referral_codes')
    .get() as { count: number };
  return Number(row.count);
}

function readReferralCodes(env: TestBrokerEnv): Array<Record<string, unknown>> {
  return env.__db.prepare('SELECT * FROM referral_codes').all() as Array<
    Record<string, unknown>
  >;
}
