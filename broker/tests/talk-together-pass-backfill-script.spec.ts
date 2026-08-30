import { describe, expect, it } from 'vitest';

import {
  allocateReferralId,
  buildBackfillSql,
  extractD1Rows,
  syntheticDiscordUserRef,
  TALK_TOGETHER_PASS_BACKFILL_QUERIES,
  validateGeneratedSql,
} from '../scripts/prepare-talk-together-pass-backfill.mjs';

describe('Talk Together Pass backfill SQL generator', () => {
  it('generates real-ref referral code inserts with runtime eligibility predicates', () => {
    const { sql, summary } = buildBackfillSql({
      nowIso: '2026-05-20T00:00:00.000Z',
      existingReferralIds: [],
      realRefRows: [
        {
          installation_id: 'install-real-a',
          discord_user_ref: 'ph-discord-user-v1_real-a',
        },
      ],
      legacyRows: [],
      maxLegacy: 20,
      randomBytes: () => Uint8Array.from([0, 1, 2, 3, 4, 5]),
    });

    expect(summary).toMatchObject({ realRefRows: 1, legacyRows: 0 });
    expect(sql).toContain('INSERT INTO referral_codes');
    expect(sql).toContain('SELECT');
    expect(sql).toContain("identity.status = 'active'");
    expect(sql).toContain("identity.entitlement_installation_id = e.installation_id");
    expect(sql).toContain("e.status = 'active'");
    expect(sql).toContain('e.managed_credential_ref IS NOT NULL');
    expect(sql).toContain("e.discord_issue_status = 'active'");
    expect(sql).toContain("datetime(e.expires_at) >= datetime('now')");
    expect(sql).not.toContain("e.expires_at >= datetime('now')");
    expect(sql).toContain('e.discord_issue_delivered_at IS NOT NULL');
    expect(sql).toContain('NOT EXISTS');
    expect(sql).toContain("existing.owner_source = 'discord'");
    expect(sql).toContain('existing.owner_subject_ref = e.discord_user_ref');
    expect(sql).toContain("existing.referral_id = '234567'");
    expect(sql).toContain('install-real-a');
    expect(sql).toContain('ph-discord-user-v1_real-a');
    expect(sql).not.toContain('referral_rewards');
    expect(sql).not.toContain('discord_oauth_sessions');
    expect(sql).not.toMatch(/\bBEGIN\b|\bCOMMIT\b|\bROLLBACK\b|\bSAVEPOINT\b|\bRELEASE\b/iu);
    validateGeneratedSql(sql);
  });

  it('generates synthetic legacy identity, entitlement update, and referral code insert in order', () => {
    const { sql, summary } = buildBackfillSql({
      nowIso: '2026-05-20T00:00:00.000Z',
      existingReferralIds: [],
      realRefRows: [],
      legacyRows: [
        {
          installation_id: 'install-legacy-a',
          issued_at: '2026-05-01T00:00:00.000Z',
          expires_at: '2026-08-01T00:00:00.000Z',
          verified_hardware_hash: 'hardware-hash-legacy-a',
          verified_hardware_hash_salt_version: 1,
        },
      ],
      maxLegacy: 20,
      randomBytes: () => Uint8Array.from([6, 7, 8, 9, 10, 11]),
    });

    expect(summary).toMatchObject({ realRefRows: 0, legacyRows: 1 });
    const identityIndex = sql.indexOf('INSERT INTO discord_identities');
    const entitlementIndex = sql.indexOf('UPDATE openrouter_entitlements');
    const codeIndex = sql.indexOf('INSERT INTO referral_codes');
    expect(identityIndex).toBeGreaterThanOrEqual(0);
    expect(entitlementIndex).toBeGreaterThan(identityIndex);
    expect(codeIndex).toBeGreaterThan(entitlementIndex);
    expect(sql).toContain('ph-discord-user-v1_legacy_');
    expect(sql).not.toContain('ph-discord-user-v1_legacy_install-legacy-a');
    expect(sql).toContain('install-legacy-a');
    expect(sql).toContain("discord_issue_status = 'active'");
    expect(sql).toContain('discord_issue_delivered_at = COALESCE(discord_issue_delivered_at');
    expect(sql).toContain('e.discord_user_ref =');
    expect(sql).toContain("identity.status = 'active'");
    expect(sql).toContain("identity.entitlement_installation_id = e.installation_id");
    expect(sql).toContain('existing_active_identity.entitlement_installation_id = e.installation_id');
    expect(sql).toContain('existing_active_identity.discord_user_ref IS NULL');
    expect(sql).toContain('existing_active_identity.discord_user_ref <>');
    expect(sql).toContain("datetime(e.expires_at) >= datetime('now')");
    expect(sql).toContain("datetime(expires_at) >= datetime('now')");
    expect(sql).not.toContain("e.expires_at >= datetime('now')");
    expect(sql).not.toContain("expires_at >= datetime('now')");
    expect(sql.match(/existing_referral_id\.referral_id = '89ABCD'/gu) ?? []).toHaveLength(2);
    validateGeneratedSql(sql);
  });

  it('rejects extra openrouter entitlement updates even when another synthetic block is guarded', () => {
    const { sql: guardedSql } = buildBackfillSql({
      nowIso: '2026-05-20T00:00:00.000Z',
      existingReferralIds: [],
      realRefRows: [],
      legacyRows: [
        {
          installation_id: 'install-legacy-a',
          issued_at: '2026-05-01T00:00:00.000Z',
          expires_at: '2026-08-01T00:00:00.000Z',
          verified_hardware_hash: 'hardware-hash-legacy-a',
          verified_hardware_hash_salt_version: 1,
        },
      ],
      maxLegacy: 20,
      randomBytes: () => Uint8Array.from([6, 7, 8, 9, 10, 11]),
    });

    const unsafeSql = `${guardedSql}
UPDATE openrouter_entitlements
   SET discord_user_ref = 'x'
 WHERE installation_id = 'y';`;

    expect(() => validateGeneratedSql(unsafeSql)).toThrow(/guardrail|synthetic/i);
  });

  it('rejects malformed synthetic identity inserts independently', () => {
    const { sql: guardedSql } = buildBackfillSql({
      nowIso: '2026-05-20T00:00:00.000Z',
      existingReferralIds: [],
      realRefRows: [],
      legacyRows: [
        {
          installation_id: 'install-legacy-a',
          issued_at: '2026-05-01T00:00:00.000Z',
          expires_at: '2026-08-01T00:00:00.000Z',
          verified_hardware_hash: 'hardware-hash-legacy-a',
          verified_hardware_hash_salt_version: 1,
        },
      ],
      maxLegacy: 20,
      randomBytes: () => Uint8Array.from([6, 7, 8, 9, 10, 11]),
    });

    const unsafeSql = `${guardedSql}
INSERT INTO discord_identities (
  discord_user_ref, entitlement_installation_id, status, ref_secret_version, created_at, updated_at
)
SELECT 'ph-discord-user-v1_legacy_bad', 'install-legacy-b', 'active', 1, '2026-05-20T00:00:00.000Z', '2026-05-20T00:00:00.000Z';`;

    expect(() => validateGeneratedSql(unsafeSql)).toThrow(/guardrail|synthetic/i);
  });

  it('rejects synthetic statements with tautological OR guardrail bypasses', () => {
    const { sql: guardedSql } = buildBackfillSql({
      nowIso: '2026-05-20T00:00:00.000Z',
      existingReferralIds: [],
      realRefRows: [],
      legacyRows: [
        {
          installation_id: 'install-legacy-a',
          issued_at: '2026-05-01T00:00:00.000Z',
          expires_at: '2026-08-01T00:00:00.000Z',
          verified_hardware_hash: 'hardware-hash-legacy-a',
          verified_hardware_hash_salt_version: 1,
        },
      ],
      maxLegacy: 20,
      randomBytes: () => Uint8Array.from([6, 7, 8, 9, 10, 11]),
    });

    const identityBypassSql = guardedSql.replace(
      '   AND e.managed_credential_ref IS NOT NULL',
      '   OR TRUE\n   AND e.managed_credential_ref IS NOT NULL',
    );
    const updateBypassSql = guardedSql.replace(
      '   AND managed_credential_ref IS NOT NULL',
      '   OR 1=1\n   AND managed_credential_ref IS NOT NULL',
    );

    expect(() => validateGeneratedSql(identityBypassSql)).toThrow(/guardrail|unsafe/i);
    expect(() => validateGeneratedSql(updateBypassSql)).toThrow(/guardrail|unsafe/i);
  });

  it('emits a D1-safe no-op without transaction control when there are no candidates', () => {
    const { sql, summary } = buildBackfillSql({
      nowIso: '2026-05-20T00:00:00.000Z',
      existingReferralIds: [],
      realRefRows: [],
      legacyRows: [],
      maxLegacy: 20,
      randomBytes: () => Uint8Array.from([0, 1, 2, 3, 4, 5]),
    });

    expect(summary).toMatchObject({ realRefRows: 0, legacyRows: 0 });
    expect(sql.trim()).toBe("SELECT 'talk_together_pass_backfill_noop' AS status;");
    expect(sql).not.toMatch(/\bBEGIN\b|\bCOMMIT\b|\bROLLBACK\b|\bSAVEPOINT\b|\bRELEASE\b/iu);
    validateGeneratedSql(sql);
  });

  it('rejects malformed maxLegacy string values', () => {
    for (const maxLegacy of ['20abc', '1.5']) {
      expect(() =>
        buildBackfillSql({
          nowIso: '2026-05-20T00:00:00.000Z',
          existingReferralIds: [],
          realRefRows: [],
          legacyRows: [],
          maxLegacy,
          randomBytes: () => Uint8Array.from([0, 1, 2, 3, 4, 5]),
        }),
      ).toThrow(/max-legacy/i);
    }
  });

  it('rejects SQL statements outside the backfill whitelist', () => {
    for (const unexpectedSql of [
      "DELETE FROM referral_codes WHERE referral_id = '234567';",
      "UPDATE referral_codes SET status = 'active' WHERE referral_id = '234567';",
      "UPDATE discord_identities SET status = 'active' WHERE discord_user_ref = 'x';",
      'CREATE TABLE talk_together_pass_backfill_tmp (id TEXT);',
      'SELECT 1;',
    ]) {
      expect(() => validateGeneratedSql(unexpectedSql)).toThrow(/unexpected|whitelist/i);
    }
  });

  it('rejects transaction-control statements', () => {
    for (const statement of [
      'BEGIN TRANSACTION;',
      'COMMIT;',
      'ROLLBACK;',
      'END TRANSACTION;',
      'END;',
      'SAVEPOINT backfill;',
      'RELEASE backfill;',
    ]) {
      expect(() => validateGeneratedSql(statement)).toThrow(/transaction-control/i);
    }
  });

  it('rejects generated SQL that mutates referral_rewards or discord_oauth_sessions', () => {
    expect(() =>
      validateGeneratedSql("UPDATE referral_rewards SET failure_reason = 'x';"),
    ).toThrow(/forbidden/i);
    expect(() =>
      validateGeneratedSql("UPDATE OR IGNORE \"referral_rewards\" SET failure_reason = 'x';"),
    ).toThrow(/forbidden/i);
    expect(() =>
      validateGeneratedSql("UPDATE main.`discord_oauth_sessions` SET referral_id = '7KQ9M2';"),
    ).toThrow(/forbidden/i);
    expect(() =>
      validateGeneratedSql("UPDATE OR REPLACE [discord_oauth_sessions] SET referral_id = '7KQ9M2';"),
    ).toThrow(/forbidden/i);
    expect(() =>
      validateGeneratedSql("INSERT OR IGNORE INTO referral_rewards (referral_id) VALUES ('234567');"),
    ).toThrow(/forbidden/i);
    expect(() =>
      validateGeneratedSql("INSERT/**/INTO referral_rewards (referral_id) VALUES ('234567');"),
    ).toThrow(/forbidden/i);
    expect(() =>
      validateGeneratedSql("UPDATE/**/referral_rewards SET failure_reason = 'x';"),
    ).toThrow(/forbidden/i);
    expect(() =>
      validateGeneratedSql("SELECT '--'; UPDATE referral_rewards SET failure_reason = 'x';"),
    ).toThrow(/forbidden/i);
    expect(() =>
      validateGeneratedSql("INSERT INTO `referral_rewards` (referral_id) VALUES ('234567');"),
    ).toThrow(/forbidden/i);
    expect(() =>
      validateGeneratedSql("INSERT INTO main.\"referral_rewards\" (referral_id) VALUES ('234567');"),
    ).toThrow(/forbidden/i);
    expect(() =>
      validateGeneratedSql(
        "INSERT INTO discord_oauth_sessions (state_hash) VALUES ('x');",
      ),
    ).toThrow(/forbidden/i);
    expect(() =>
      validateGeneratedSql("REPLACE INTO discord_oauth_sessions (state_hash) VALUES ('x');"),
    ).toThrow(/forbidden/i);
    expect(() =>
      validateGeneratedSql("REPLACE INTO main.\"discord_oauth_sessions\" (state_hash) VALUES ('x');"),
    ).toThrow(/forbidden/i);
    expect(() =>
      validateGeneratedSql("UPDATE discord_oauth_sessions SET referral_id = '7KQ9M2';"),
    ).toThrow(/forbidden/i);
  });

  it('rejects generated SQL that omits required runtime guardrails', () => {
    expect(() =>
      validateGeneratedSql(
        "INSERT INTO referral_codes (referral_id) SELECT '234567' WHERE NOT EXISTS (SELECT 1);",
      ),
    ).toThrow(/guardrail/i);
  });

  it('does not count guardrail text inside string literals as real predicates', () => {
    const unsafeSql = `INSERT INTO referral_codes (
  referral_id, owner_discord_user_ref, owner_installation_id, status, created_at, updated_at
)
SELECT '234567', e.discord_user_ref, e.installation_id, 'active', 'e.discord_issue_delivered_at IS NOT NULL', '2026-05-20T00:00:00.000Z'
  FROM openrouter_entitlements e
  JOIN discord_identities identity
    ON identity.discord_user_ref = e.discord_user_ref
   AND identity.status = 'active'
   AND identity.entitlement_installation_id = e.installation_id
 WHERE e.installation_id = 'install-real-a'
   AND e.discord_user_ref = 'ph-discord-user-v1_real-a'
   AND e.status = 'active'
   AND e.managed_credential_ref IS NOT NULL
   AND length(trim(e.managed_credential_ref)) > 0
   AND e.discord_user_ref IS NOT NULL
   AND length(trim(e.discord_user_ref)) > 0
   AND e.expires_at IS NOT NULL
   AND length(trim(e.expires_at)) > 0
   AND datetime(e.expires_at) >= datetime('now')
   AND e.discord_issue_status = 'active'
   AND length(trim(e.discord_issue_delivered_at)) > 0
   AND NOT EXISTS (
     SELECT 1 FROM referral_codes existing
      WHERE existing.owner_discord_user_ref = e.discord_user_ref
         OR existing.referral_id = '234567'
   );`;

    expect(() => validateGeneratedSql(unsafeSql)).toThrow(/guardrail/i);
  });

  it('rejects referral code inserts that use VALUES', () => {
    expect(() =>
      validateGeneratedSql(
        "INSERT INTO referral_codes (referral_id, owner_discord_user_ref) VALUES ('234567', 'ph-discord-user-v1_owner');",
      ),
    ).toThrow(/SELECT/i);
    expect(() =>
      validateGeneratedSql(
        "INSERT OR IGNORE INTO referral_codes (referral_id, owner_discord_user_ref) VALUES ('234567', 'ph-discord-user-v1_owner');",
      ),
    ).toThrow(/SELECT/i);
    expect(() =>
      validateGeneratedSql(
        "INSERT/**/INTO referral_codes (referral_id, owner_discord_user_ref) VALUES ('234567', 'ph-discord-user-v1_owner');",
      ),
    ).toThrow(/SELECT/i);
    expect(() =>
      validateGeneratedSql(
        "INSERT INTO \"referral_codes\" (referral_id, owner_discord_user_ref) VALUES ('234567', 'ph-discord-user-v1_owner');",
      ),
    ).toThrow(/SELECT/i);
    expect(() =>
      validateGeneratedSql(
        "INSERT INTO main.referral_codes (referral_id, owner_discord_user_ref) VALUES ('234567', 'ph-discord-user-v1_owner');",
      ),
    ).toThrow(/SELECT/i);
    expect(() =>
      validateGeneratedSql(
        "INSERT INTO main.\"referral_codes\" (referral_id, owner_discord_user_ref) VALUES ('234567', 'ph-discord-user-v1_owner');",
      ),
    ).toThrow(/SELECT/i);
  });

  it('rejects referral code insert statements with tautological OR guardrail bypasses', () => {
    const { sql } = buildBackfillSql({
      nowIso: '2026-05-20T00:00:00.000Z',
      existingReferralIds: [],
      realRefRows: [
        {
          installation_id: 'install-real-a',
          discord_user_ref: 'ph-discord-user-v1_real-a',
        },
      ],
      legacyRows: [],
      maxLegacy: 20,
      randomBytes: () => Uint8Array.from([0, 1, 2, 3, 4, 5]),
    });

    for (const bypass of [
      'OR 1=1',
      'OR (1=1)',
      'OR(1=1)',
      'OR TRUE',
      'OR (TRUE)',
      'OR 2=2',
      'OR 1',
    ]) {
      const bypassSql = sql.replace(
        "   AND e.discord_issue_status = 'active'",
        `   ${bypass}\n   AND e.discord_issue_status = 'active'`,
      );

      expect(() => validateGeneratedSql(bypassSql)).toThrow(/guardrail|unsafe/i);
    }
  });

  it('ignores forbidden DML and tautology text inside escaped SQL string values', () => {
    const { sql } = buildBackfillSql({
      nowIso: '2026-05-20T00:00:00.000Z',
      existingReferralIds: [],
      realRefRows: [
        {
          installation_id: "install literal OR TRUE and UPDATE referral_rewards SET x = 'y'",
          discord_user_ref: "ph-discord-user-v1_value-OR-TRUE-UPDATE-referral_rewards",
        },
      ],
      legacyRows: [],
      maxLegacy: 20,
      randomBytes: () => Uint8Array.from([0, 1, 2, 3, 4, 5]),
    });

    expect(sql).toContain('OR TRUE');
    expect(sql).toContain('UPDATE referral_rewards');
    validateGeneratedSql(sql);
  });

  it('does not treat real-ref string values containing the synthetic prefix as synthetic DML', () => {
    expect(() =>
      buildBackfillSql({
        nowIso: '2026-05-20T00:00:00.000Z',
        existingReferralIds: [],
        realRefRows: [
          {
            installation_id: 'install-ph-discord-user-v1_legacy_marker',
            discord_user_ref: 'ph-discord-user-v1_real-a',
          },
        ],
        legacyRows: [],
        maxLegacy: 20,
        randomBytes: () => Uint8Array.from([0, 1, 2, 3, 4, 5]),
      }),
    ).not.toThrow();
  });

  it('validates each referral code insert independently', () => {
    const { sql: guardedSql } = buildBackfillSql({
      nowIso: '2026-05-20T00:00:00.000Z',
      existingReferralIds: [],
      realRefRows: [
        {
          installation_id: 'install-real-a',
          discord_user_ref: 'ph-discord-user-v1_real-a',
        },
      ],
      legacyRows: [],
      maxLegacy: 20,
      randomBytes: () => Uint8Array.from([0, 1, 2, 3, 4, 5]),
    });

    const unguardedSql = `${guardedSql}\nINSERT INTO referral_codes (
  referral_id, owner_discord_user_ref, owner_installation_id, status, created_at, updated_at
)
SELECT '89ABCD', e.discord_user_ref, e.installation_id, 'active', '2026-05-20T00:00:00.000Z', '2026-05-20T00:00:00.000Z'
  FROM openrouter_entitlements e
 WHERE e.installation_id = 'install-real-b';`;

    expect(() => validateGeneratedSql(unguardedSql)).toThrow(/guardrail/i);
  });

  it('escapes SQL string literals', () => {
    const { sql } = buildBackfillSql({
      nowIso: '2026-05-20T00:00:00.000Z',
      existingReferralIds: [],
      realRefRows: [
        {
          installation_id: "install-real-'quote",
          discord_user_ref: "ph-discord-user-v1_real-'quote",
        },
      ],
      legacyRows: [],
      maxLegacy: 20,
      randomBytes: () => Uint8Array.from([12, 13, 14, 15, 16, 17]),
    });

    expect(sql).toContain("install-real-''quote");
    expect(sql).toContain("ph-discord-user-v1_real-''quote");
    validateGeneratedSql(sql);
  });

  it('allocates collision-free allowed Pass IDs', () => {
    const existing = new Set(['234567']);
    let callCount = 0;
    const referralId = allocateReferralId(existing, () => {
      callCount += 1;
      return callCount === 1
        ? Uint8Array.from([0, 1, 2, 3, 4, 5])
        : Uint8Array.from([6, 7, 8, 9, 10, 11]);
    });

    expect(referralId).toMatch(/^[23456789ABCDEFGHJKMNPQRSTUVWXYZ]{6}$/u);
    expect(referralId).not.toBe('234567');
    expect(existing.has(referralId)).toBe(true);
  });

  it('builds stable synthetic refs without exposing raw installation data', () => {
    const first = syntheticDiscordUserRef('install-legacy-a');
    const second = syntheticDiscordUserRef('install-legacy-a');
    const other = syntheticDiscordUserRef('install-legacy-b');

    expect(first).toBe(second);
    expect(first).not.toBe(other);
    expect(first).toMatch(/^ph-discord-user-v1_legacy_[A-Za-z0-9_-]+$/u);
    expect(first).not.toContain('install-legacy-a');
  });

  it('aborts when legacy candidates exceed the configured maximum', () => {
    expect(() =>
      buildBackfillSql({
        nowIso: '2026-05-20T00:00:00.000Z',
        existingReferralIds: [],
        realRefRows: [],
        legacyRows: [
          {
            installation_id: 'install-legacy-a',
            issued_at: '2026-05-01T00:00:00.000Z',
            expires_at: '2026-08-01T00:00:00.000Z',
            verified_hardware_hash: 'hardware-hash-legacy-a',
            verified_hardware_hash_salt_version: 1,
          },
        ],
        maxLegacy: 0,
        randomBytes: () => Uint8Array.from([0, 1, 2, 3, 4, 5]),
      }),
    ).toThrow(/max-legacy/i);
  });

  it('uses synthetic candidate query guardrails for existing active installation identities', () => {
    expect(TALK_TOGETHER_PASS_BACKFILL_QUERIES.syntheticLegacyCandidates).toContain(
      'existing_active_identity.entitlement_installation_id = e.installation_id',
    );
    expect(TALK_TOGETHER_PASS_BACKFILL_QUERIES.syntheticLegacyCandidates).toContain(
      "existing_active_identity.status = 'active'",
    );
    expect(TALK_TOGETHER_PASS_BACKFILL_QUERIES.syntheticLegacyCandidates).toContain(
      'existing_active_identity.discord_user_ref IS NULL',
    );
    expect(TALK_TOGETHER_PASS_BACKFILL_QUERIES.realRefCandidates).toContain(
      "datetime(e.expires_at) >= datetime('now')",
    );
    expect(TALK_TOGETHER_PASS_BACKFILL_QUERIES.syntheticLegacyCandidates).toContain(
      "datetime(e.expires_at) >= datetime('now')",
    );
    expect(TALK_TOGETHER_PASS_BACKFILL_QUERIES.realRefCandidates).not.toContain(
      "e.expires_at >= datetime('now')",
    );
    expect(TALK_TOGETHER_PASS_BACKFILL_QUERIES.syntheticLegacyCandidates).not.toContain(
      "e.expires_at >= datetime('now')",
    );
  });

  it('does not echo invalid production referral ids in errors', () => {
    expect(() =>
      buildBackfillSql({
        nowIso: '2026-05-20T00:00:00.000Z',
        existingReferralIds: ['SECRET_BAD_REFERRAL_ID'],
        realRefRows: [],
        legacyRows: [],
        maxLegacy: 20,
        randomBytes: () => Uint8Array.from([0, 1, 2, 3, 4, 5]),
      }),
    ).toThrow(/existing referral_id is not a valid Pass ID/i);
    expect(() =>
      buildBackfillSql({
        nowIso: '2026-05-20T00:00:00.000Z',
        existingReferralIds: ['SECRET_BAD_REFERRAL_ID'],
        realRefRows: [],
        legacyRows: [],
        maxLegacy: 20,
        randomBytes: () => Uint8Array.from([0, 1, 2, 3, 4, 5]),
      }),
    ).not.toThrow(/SECRET_BAD_REFERRAL_ID/);
  });

  it('extracts Wrangler D1 rows and sanitizes failure wrappers', () => {
    expect(
      extractD1Rows([
        {
          success: true,
          results: [{ referral_id: '234567' }],
        },
      ]),
    ).toEqual([{ referral_id: '234567' }]);

    expect(() =>
      extractD1Rows([
        {
          success: false,
          error: 'production row secret should not be echoed',
        },
      ]),
    ).toThrow(/wrangler D1 query failed/i);
    expect(() =>
      extractD1Rows([
        {
          success: false,
          error: 'production row secret should not be echoed',
        },
      ]),
    ).not.toThrow(/production row secret/i);
  });
});
