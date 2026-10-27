import { DatabaseSync } from 'node:sqlite';

import { describe, expect, it } from 'vitest';

import {
  BROKER_MIGRATION_FILENAMES,
  applyBrokerMigrations,
} from './test-support/migrations';
import { createTestBrokerEnv } from './test-support/sqlite-d1';

describe('broker persistent state model', () => {
  it('preserves the migration history and creates the lookup and uniqueness indexes used by broker decisions', () => {
    expect(BROKER_MIGRATION_FILENAMES).toEqual([
      '0000_define_broker_persistent_state.sql',
      '0001_add_abuse_hook_state.sql',
      '0001_harden_installation_public_inputs.sql',
      '0002_add_entitlement_verified_hardware_snapshot.sql',
      '0003_add_abuse_runtime_state_and_issue_success_events.sql',
      '0004_add_discord_oauth_managed_issue.sql',
      '0005_add_referral_persistence_foundation.sql',
      '0006_harden_referral_reward_operations.sql',
      '0007_simplify_referral_id_checks.sql',
      '0008_add_qq_auth_assertions.sql',
      '0009_add_qq_managed_entitlements.sql',
      '0010_source_aware_issue_success_events.sql',
      '0011_add_telemetry_active_days.sql',
      '0012_add_managed_key_delivery_ack.sql',
      '0013_add_telemetry_subjects_and_daily_summary_v2.sql',
      '0014_simplify_abuse_incidents.sql',
      '0015_add_app_active_days.sql',
      '0016_make_referrals_source_aware.sql',
      '0017_add_qq_pass_settlement_jobs.sql',
      '0018_managed_operations.sql',
      '0019_managed_referral_settlement.sql',
      '0020_network_identity_hmac.sql',
      '0021_network_identity_purge.sql',
      '0022_managed_operation_issuance_context.sql',
      '0023_backfill_operation_route_rate_limits.sql',
      '0024_allow_unattributed_request_events.sql',
      '0025_drop_legacy_translation_telemetry.sql',
      '0026_managed_attempt_cleanup_ownership.sql',
      '0027_remove_managed_referrals.sql',
    ]);
    const env = createTestBrokerEnv();
    const requiredIndexes: Record<string, Array<{ columns: string[]; unique: number; partial: number }>> = {
      installations: [
        { columns: ['device_public_key'], unique: 1, partial: 0 },
        { columns: ['hardware_hash'], unique: 0, partial: 0 },
        { columns: ['hardware_hash_salt_version'], unique: 0, partial: 0 },
        { columns: ['challenge_expires_at'], unique: 0, partial: 0 },
        { columns: ['last_seen_at'], unique: 0, partial: 0 },
      ],
      openrouter_entitlements: [
        { columns: ['managed_credential_ref'], unique: 1, partial: 0 },
        { columns: ['release_token_hash'], unique: 1, partial: 1 },
        { columns: ['discord_user_ref'], unique: 1, partial: 1 },
        { columns: ['status'], unique: 0, partial: 0 },
        { columns: ['expires_at'], unique: 0, partial: 0 },
      ],
      qq_managed_entitlements: [
        { columns: ['issue_ref'], unique: 1, partial: 0 },
        { columns: ['managed_credential_ref'], unique: 1, partial: 1 },
        { columns: ['status', 'updated_at'], unique: 0, partial: 0 },
        { columns: ['expires_at'], unique: 0, partial: 0 },
      ],
      broker_request_events: [
        { columns: ['endpoint', 'ip_digest', 'observed_at'], unique: 0, partial: 0 },
        { columns: ['endpoint', 'installation_id', 'observed_at'], unique: 0, partial: 0 },
        { columns: ['ip_digest', 'observed_at'], unique: 0, partial: 1 },
        { columns: ['installation_id', 'observed_at'], unique: 0, partial: 0 },
      ],
      managed_key_deliveries: [
        { columns: ['ack_token_hash'], unique: 1, partial: 0 },
        { columns: ['status', 'expires_at'], unique: 0, partial: 0 },
        { columns: ['managed_credential_ref'], unique: 0, partial: 0 },
        { columns: ['issue_source', 'created_at'], unique: 0, partial: 0 },
      ],
    };
    for (const [table, expected] of Object.entries(requiredIndexes)) {
      const indexes = env.__db.prepare('SELECT name, \"unique\", partial FROM pragma_index_list(?)').all(table) as Array<{
        name: string; unique: number; partial: number;
      }>;
      const actual = indexes.map((index) => ({
        columns: (env.__db.prepare('SELECT name FROM pragma_index_info(?) ORDER BY seqno').all(index.name) as Array<{ name: string }>).map(({ name }) => name),
        unique: index.unique,
        partial: index.partial,
      }));
      for (const index of expected) {
        expect(actual, `${table}: index on (${index.columns.join(', ')}) unique=${index.unique} partial=${index.partial}`).toContainEqual(index);
      }
    }
  });

  it('inserts the QQ auth assertion abuse-control default without replacing tuned JSON', () => {
    const db = new DatabaseSync(':memory:');
    try {
      applyBrokerMigrations(db, { through: '0007_simplify_referral_id_checks.sql' });

      const beforeRow = db
        .prepare('SELECT value FROM broker_config WHERE key = ?')
        .get('abuse_controls') as { value: string };
      const before = JSON.parse(beforeRow.value) as {
        trialChallenge: { maxRequests: number };
        discordAuthStartIp: { maxRequests: number };
      } & Record<string, unknown>;
      before.trialChallenge.maxRequests = 7;
      before.discordAuthStartIp.maxRequests = 17;
      db.prepare('UPDATE broker_config SET value = ? WHERE key = ?').run(
        JSON.stringify(before),
        'abuse_controls',
      );

      applyBrokerMigrations(db, { after: '0007_simplify_referral_id_checks.sql' });

      const afterRow = db
        .prepare('SELECT value FROM broker_config WHERE key = ?')
        .get('abuse_controls') as { value: string };
      const after = JSON.parse(afterRow.value) as {
        trialChallenge: { maxRequests: number };
        discordAuthStartIp: { maxRequests: number };
        qqAuthAssertIp?: unknown;
      };

      expect(after.trialChallenge.maxRequests).toBe(7);
      expect(after.discordAuthStartIp.maxRequests).toBe(17);
      expect(after.qqAuthAssertIp).toEqual({
        endpoint: 'POST /v1/auth/qq/assert',
        scope: 'ip',
        maxRequests: 20,
        windowMinutes: 15,
      });
    } finally {
      db.close();
    }
  });
});
