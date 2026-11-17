import { describe, expect, it } from 'vitest';

import { accountingDatabase } from '../../agent-relay/tests/fixtures';
import { getRelayDailyUsage, type RelayDailyUsage } from '../src/relay-usage';

const reportDate = '2026-11-13';
const start = Date.UTC(2026, 10, 13);
const end = start + 86_400_000;
const emptyUsage: RelayDailyUsage = {
  chatbot: { classification_calls: 0, response_calls: 0, missing_usage_calls: 0, cost_usd: 0 },
  photo: { requests: 0, unsettled_requests: 0, missing_usage_requests: 0, estimated_date_requests: 0, cost_usd: 0 },
  total_cost_usd: 0,
};
const aggregateRow = {
  classification_calls: 0, response_calls: 0, missing_usage_calls: 0, chatbot_cost_usd: 0,
  requests: 0, unsettled_requests: 0, missing_usage_requests: 0, estimated_date_requests: 0, photo_cost_usd: 0,
};

function resultDatabase(result: unknown): D1Database {
  return { prepare: () => ({ bind: () => ({ all: async () => result }) }) } as unknown as D1Database;
}

describe('Relay daily usage aggregation', () => {
  it('sums exact chatbot dates across installations and separates settled, missing and uncertain photo usage', async () => {
    const { db, sqlite } = accountingDatabase();
    try {
      const chatbot = sqlite.prepare(`
        INSERT INTO usage_daily (installation_hash, utc_date, decisions_calls, responses_calls, missing_usage_count, cost_usd)
        VALUES (?, ?, ?, ?, ?, ?)
      `);
      chatbot.run('installation-a', reportDate, 2, 4, 1, 1.25);
      chatbot.run('installation-b', reportDate, 3, 5, 2, 0.5);
      for (const date of ['2026-11-12', '2026-11-14', '2026-11-13T00:00:00Z']) {
        chatbot.run('installation-a', date, 99, 99, 99, 99);
      }
      const photo = sqlite.prepare(`
        INSERT INTO photo_admissions
          (installation_hash, request_id, period_start, requested_at, owner, expires_at, settled, missing_usage, cost_usd)
        VALUES (?, ?, ?, ?, 'owner', ?, ?, ?, ?)
      `);
      photo.run('installation-a', 'measured', start - 18 * 3_600_000, start + 1, end, 1, 0, 0.25);
      photo.run('installation-b', 'missing', start - 18 * 3_600_000, start + 12 * 3_600_000, end, 1, 7, 0.5);
      photo.run('installation-b', 'unsettled', start, end - 1, end, 0, 1, 99);
      photo.run('installation-a', 'legacy-settled', start + 1, null, end, 1, 0, 0.125);
      photo.run('installation-b', 'legacy-unsettled', end - 1, null, end, 0, 7, 88);

      expect(await getRelayDailyUsage(db, reportDate)).toEqual({
        chatbot: { classification_calls: 5, response_calls: 9, missing_usage_calls: 3, cost_usd: 1.75 },
        photo: { requests: 5, unsettled_requests: 2, missing_usage_requests: 1, estimated_date_requests: 2, cost_usd: 0.875 },
        total_cost_usd: 2.625,
      });
    } finally {
      sqlite.close();
    }
  });

  it('uses half-open UTC milliseconds and only falls back to the period for historical null dates', async () => {
    const { db, sqlite } = accountingDatabase();
    try {
      const insert = sqlite.prepare(`
        INSERT INTO photo_admissions
          (installation_hash, request_id, period_start, requested_at, owner, expires_at, settled, missing_usage, cost_usd)
        VALUES ('installation', ?, ?, ?, 'owner', ?, 1, 0, 1)
      `);
      for (const [index, timestamp] of [start - 1, start, end - 1, end].entries()) {
        insert.run(`exact-${index}`, start - 18 * 3_600_000, timestamp, end);
        insert.run(`legacy-${index}`, timestamp, null, end);
      }
      expect(await getRelayDailyUsage(db, reportDate)).toEqual({
        ...emptyUsage,
        photo: { requests: 4, unsettled_requests: 0, missing_usage_requests: 0, estimated_date_requests: 2, cost_usd: 4 },
        total_cost_usd: 4,
      });
      const previous = await getRelayDailyUsage(db, '2026-11-12');
      expect(previous.photo).toEqual({
        requests: 2, unsettled_requests: 0, missing_usage_requests: 0, estimated_date_requests: 1, cost_usd: 2,
      });
      const following = await getRelayDailyUsage(db, '2026-11-14');
      expect(following.photo).toEqual({
        requests: 2, unsettled_requests: 0, missing_usage_requests: 0, estimated_date_requests: 1, cost_usd: 2,
      });
    } finally {
      sqlite.close();
    }
  });

  it('keeps a custom allowance period separate from the actual request calendar date', async () => {
    const { db, sqlite } = accountingDatabase();
    try {
      const periodMs = 30 * 3_600_000;
      const period = Math.floor((end - 1) / periodMs) * periodMs;
      sqlite.prepare(`
        INSERT INTO photo_admissions
          (installation_hash, request_id, period_start, requested_at, owner, expires_at, settled, missing_usage, cost_usd)
        VALUES ('installation', 'before-midnight', ?, ?, 'owner', ?, 1, 0, 0.25),
          ('installation', 'after-midnight', ?, ?, 'owner', ?, 1, 0, 0.5)
      `).run(period, end - 1, end + 300_000, period, end + 1, end + 300_000);
      expect(Math.floor((end + 1) / periodMs) * periodMs).toBe(period);
      const before = await getRelayDailyUsage(db, reportDate);
      const after = await getRelayDailyUsage(db, '2026-11-14');
      expect(before.photo).toEqual({
        requests: 1, unsettled_requests: 0, missing_usage_requests: 0, estimated_date_requests: 0, cost_usd: 0.25,
      });
      expect(after.photo).toEqual({
        requests: 1, unsettled_requests: 0, missing_usage_requests: 0, estimated_date_requests: 0, cost_usd: 0.5,
      });
    } finally {
      sqlite.close();
    }
  });

  it('returns real zeros for an empty day without returning installation or request data', async () => {
    const { db, sqlite } = accountingDatabase();
    try {
      expect(await getRelayDailyUsage(db, reportDate)).toEqual(emptyUsage);
      sqlite.prepare(`
        INSERT INTO usage_daily (installation_hash, utc_date, decisions_calls, cost_usd)
        VALUES ('installation', '2026-11-12', 1, 1)
      `).run();
      sqlite.prepare(`
        INSERT INTO photo_admissions
          (installation_hash, request_id, period_start, requested_at, owner, expires_at, settled, cost_usd)
        VALUES ('installation', 'other-day', ?, ?, 'owner', ?, 1, 1)
      `).run(start - 1, start - 1, start);
      expect(await getRelayDailyUsage(db, reportDate)).toEqual(emptyUsage);
    } finally {
      sqlite.close();
    }
  });

  it.each(['usage_daily', 'photo_admissions'])('fails rather than reporting zeros when SQL cannot read %s', async (table) => {
    const { db, sqlite } = accountingDatabase();
    try {
      sqlite.exec(`DROP TABLE ${table}`);
      await expect(getRelayDailyUsage(db, reportDate)).rejects.toThrow();
    } finally {
      sqlite.close();
    }
  });

  it.each([
    undefined,
    null,
    { success: false, results: [aggregateRow] },
    { success: true },
    { success: true, results: [] },
    { success: true, results: [aggregateRow, aggregateRow] },
    { success: true, results: [null] },
    { success: true, results: [[]] },
    { success: true, results: [{}] },
    { success: true, results: {} },
  ])('rejects unsuccessful or missing aggregate rows: %j', async (result) => {
    await expect(getRelayDailyUsage(resultDatabase(result), reportDate)).rejects.toThrow('relay_usage_unavailable');
  });

  it.each<[string, unknown]>([
    ['classification_calls', -1],
    ['response_calls', -1],
    ['missing_usage_calls', -1],
    ['requests', -1],
    ['unsettled_requests', -1],
    ['missing_usage_requests', -1],
    ['estimated_date_requests', -1],
    ['classification_calls', NaN],
    ['response_calls', Infinity],
    ['missing_usage_calls', 1.5],
    ['requests', Number.MAX_SAFE_INTEGER + 1],
    ['unsettled_requests', null],
    ['missing_usage_requests', '1'],
    ['estimated_date_requests', undefined],
    ['chatbot_cost_usd', -1],
    ['chatbot_cost_usd', NaN],
    ['chatbot_cost_usd', '1'],
    ['photo_cost_usd', -1],
    ['photo_cost_usd', Infinity],
    ['photo_cost_usd', null],
  ])('rejects invalid aggregate %s=%s', async (key, value) => {
    const db = resultDatabase({ success: true, results: [{ ...aggregateRow, [key]: value }] });
    await expect(getRelayDailyUsage(db, reportDate)).rejects.toThrow('relay_usage_unavailable');
  });

  it('rejects a nonfinite combined cost', async () => {
    const db = resultDatabase({ success: true, results: [{
      ...aggregateRow, chatbot_cost_usd: Number.MAX_VALUE, photo_cost_usd: Number.MAX_VALUE,
    }] });
    await expect(getRelayDailyUsage(db, reportDate)).rejects.toThrow('relay_usage_unavailable');
  });

  it.each(['2026-02-30', '2026-11-13T00:00:00Z', '2026-1-3', 'invalid'])('rejects a non-calendar report date: %s', async (date) => {
    await expect(getRelayDailyUsage(resultDatabase(undefined), date)).rejects.toThrow('invalid_report_date_utc');
  });
});
