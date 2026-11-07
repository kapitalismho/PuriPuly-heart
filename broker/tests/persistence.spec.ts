import { DatabaseSync } from 'node:sqlite';

import { describe, expect, it } from 'vitest';

import { applyBrokerMigrations } from './test-support/migrations';

describe('broker persistent state model', () => {
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
