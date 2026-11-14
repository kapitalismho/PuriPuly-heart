import { readFileSync } from 'node:fs';
import { afterEach, describe, expect, it, vi } from 'vitest';

import relay from '../src/index';
import { checkAllowance, checkPhotoAllowance, photoPolicy, terraDailyTokens, turnLifetimeMs, usageRecorder } from '../src/usage';
import { accountingDatabase, installationHash, relayEvents, relayMigrations, responseEvents, signToken, testEnv, turnId } from './fixtures';

const start = new Date('2026-11-11T23:59:50Z');
const otherInstallation = 'b'.repeat(64);
const usage = { input_tokens: 10, input_tokens_details: { cached_tokens: 4, cache_write_tokens: 3 },
  output_tokens: 2, output_tokens_details: { reasoning_tokens: 1 } };
const completed = { object: 'response', status: 'completed', output: [], usage };

function request(turn_id: string, installation = installationHash) {
  return new Request('https://relay.example/v1/respond', {
    method: 'POST', headers: { authorization: `Bearer ${signToken({ sub: installation })}` },
    body: JSON.stringify({ agent_protocol: 6, turn_id, route: 'work', effort: 'high', allow_tools: true,
      input: [{ type: 'configuration_update', reasoning: { effort: 'high' } }, { role: 'user', content: 'Synthetic app help.' }] }),
  });
}

async function admit(db: D1Database, installation: string, turn: string, classification = false, now = start) {
  const result = await checkAllowance({ AGENT_USAGE_DB: db }, installation, turn, classification, now);
  if (!('accounting' in result)) throw new Error(`Admission failed: ${result.error}`);
  return result.accounting;
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.useRealTimers();
});

describe('Terra daily response preference', () => {
  it('switches new turns at the shared threshold while keeping continuations on their admitted model', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(start);
    const upstream = vi.spyOn(globalThis, 'fetch').mockImplementation(async (_url, init) =>
      responseEvents({ ...completed, model: JSON.parse(init?.body as string).model }));
    const { sqlite, db } = accountingDatabase();
    const env = { ...testEnv(), AGENT_USAGE_DB: db };
    try {
      sqlite.prepare('INSERT INTO terra_daily VALUES (?, ?)').run('2026-11-11', terraDailyTokens - 12);
      const models = [];
      for (const [turn, installation] of [[turnId, installationHash], ['c'.repeat(32), installationHash],
        [turnId, installationHash], [turnId, otherInstallation]]) {
        const response = await relay.fetch(request(turn, installation), env);
        expect(response.status).toBe(200);
        const events = await relayEvents(response);
        models.push((events[0].response as Record<string, unknown>).model);
      }
      expect(models).toEqual(['gpt-5.6-terra', 'gpt-6-luna', 'gpt-5.6-terra', 'gpt-6-luna']);
      expect(upstream).toHaveBeenCalledTimes(4);
      expect(sqlite.prepare('SELECT tokens FROM terra_daily').get()).toEqual({ tokens: terraDailyTokens + 12 });
      expect(sqlite.prepare('SELECT model_admissions, response_model FROM prompt_turns WHERE installation_hash = ? AND turn_id = ?')
        .get(installationHash, turnId)).toEqual({ model_admissions: 2, response_model: 'gpt-5.6-terra' });
    } finally { sqlite.close(); }
  });

  it('atomically adds concurrent observed usage once without counting D, photos or unknown usage', async () => {
    const { sqlite, db } = accountingDatabase();
    try {
      const classifier = await admit(db, installationHash, turnId, true);
      await usageRecorder(classifier, true)({ usage: { input_tokens: terraDailyTokens, output_tokens: 0 } });
      expect(sqlite.prepare('SELECT response_model FROM prompt_turns').get()).toEqual({ response_model: null });
      const photo = await checkPhotoAllowance({ AGENT_USAGE_DB: db }, installationHash, 'synthetic-photo', photoPolicy(), start.getTime());
      if (!('settle' in photo)) throw new Error(photo.error);
      await photo.settle({ usage: { input_tokens: terraDailyTokens, output_tokens: 0 } });
      expect(sqlite.prepare('SELECT * FROM terra_daily').all()).toEqual([]);
      const requests = await Promise.all([admit(db, installationHash, turnId), admit(db, otherInstallation, turnId)]);
      expect(requests.map((value) => value.model)).toEqual(['gpt-5.6-terra', 'gpt-5.6-terra']);
      const recorders = requests.map((accounting) => usageRecorder(accounting, false));
      await Promise.all(recorders.flatMap((settle) => [settle(completed), settle(completed)]));
      expect(sqlite.prepare('SELECT tokens FROM terra_daily').get()).toEqual({ tokens: 24 });
      const unknown = await admit(db, installationHash, turnId);
      await usageRecorder(unknown, false)(undefined);
      expect(sqlite.prepare('SELECT tokens FROM terra_daily').get()).toEqual({ tokens: 24 });
      expect(sqlite.prepare('SELECT responses_calls, missing_usage_count FROM usage_daily WHERE installation_hash = ?')
        .get(installationHash)).toEqual({ responses_calls: 2, missing_usage_count: 1 });
      expect(sqlite.prepare('SELECT * FROM request_slots').all()).toEqual([]);
    } finally { sqlite.close(); }
  });

  it('starts a fresh UTC counter without repinning either model or renewing old turn budgets', async () => {
    const { sqlite, db } = accountingDatabase();
    try {
      const terra = await admit(db, installationHash, turnId);
      sqlite.prepare('INSERT INTO terra_daily VALUES (?, ?)').run('2026-11-11', terraDailyTokens);
      const luna = await admit(db, otherInstallation, turnId);
      const tomorrow = new Date('2026-11-12T00:00:01Z');
      await usageRecorder(terra, false)(completed);
      await usageRecorder(luna, false)(completed);
      const continuedTerra = await admit(db, installationHash, turnId, false, tomorrow);
      const continuedLuna = await admit(db, otherInstallation, turnId, false, tomorrow);
      expect([continuedTerra.model, continuedLuna.model]).toEqual(['gpt-5.6-terra', 'gpt-6-luna']);
      await usageRecorder(continuedTerra, false)(completed);
      await usageRecorder(continuedLuna, false)(completed);
      const fresh = await admit(db, installationHash, 'c'.repeat(32), false, tomorrow);
      expect(fresh.model).toBe('gpt-5.6-terra');
      expect(sqlite.prepare('SELECT utc_date, tokens FROM terra_daily ORDER BY utc_date').all())
        .toEqual([{ utc_date: '2026-11-11', tokens: terraDailyTokens + 12 }, { utc_date: '2026-11-12', tokens: 12 }]);
      expect(sqlite.prepare('SELECT utc_date, expires_at, model_admissions FROM prompt_turns WHERE installation_hash = ? AND turn_id = ?')
        .get(installationHash, turnId)).toEqual({ utc_date: '2026-11-11', expires_at: start.getTime() + turnLifetimeMs, model_admissions: 2 });
      await usageRecorder(fresh, false)(completed);
    } finally { sqlite.close(); }
  });

  it('preserves pre-migration response turns on Luna while allowing classifier-only turns to select Terra', async () => {
    const { sqlite, db } = accountingDatabase(relayMigrations.slice(0, -1));
    try {
      const seed = sqlite.prepare(`INSERT INTO prompt_turns
        (installation_hash, turn_id, utc_date, expires_at, model_admissions, classifier_admitted) VALUES (?, ?, ?, ?, ?, 1)`);
      seed.run(installationHash, turnId, '2026-11-11', start.getTime() + turnLifetimeMs, 2);
      seed.run(otherInstallation, turnId, '2026-11-11', start.getTime() + turnLifetimeMs, 1);
      sqlite.exec(readFileSync(new URL('../migrations/0007_terra_routing.sql', import.meta.url), 'utf8'));
      const existing = await admit(db, installationHash, turnId);
      const firstResponse = await admit(db, otherInstallation, turnId);
      expect([existing.model, firstResponse.model]).toEqual(['gpt-6-luna', 'gpt-5.6-terra']);
      expect(sqlite.prepare('SELECT model_admissions FROM prompt_turns WHERE installation_hash = ?')
        .get(installationHash)).toEqual({ model_admissions: 3 });
      await usageRecorder(existing, false)(completed);
      await usageRecorder(firstResponse, false)(completed);
      expect(sqlite.prepare('SELECT tokens FROM terra_daily').get()).toEqual({ tokens: 12 });
    } finally { sqlite.close(); }
  });

  it('prices Terra separately from Luna overrides and counts reported 12-hour writes without doubling tokens', async () => {
    const { sqlite, db } = accountingDatabase();
    const rates = { as_of: '2026-11-11', input: 1, cache_read: 2, cache_write: 3, responses_output: 4, decisions_output: 5 };
    const env = { AGENT_USAGE_DB: db, AGENT_USAGE_RATES: JSON.stringify(rates) };
    const result = { usage: { input_tokens: 100, input_tokens_details: { cached_tokens: 20, cache_write_tokens: 10,
      cache_write_tokens_details: { ttl_12h_tokens: 4, ttl_30m_tokens: 6 } }, output_tokens: 20,
      output_tokens_details: { reasoning_tokens: 15 } } };
    try {
      const terra = await checkAllowance(env, installationHash, turnId, false, start);
      if (!('accounting' in terra)) throw new Error(terra.error);
      await usageRecorder(terra.accounting, false)(result);
      expect(sqlite.prepare('SELECT cost_usd FROM usage_daily WHERE installation_hash = ?').get(installationHash)?.cost_usd)
        .toBeCloseTo(0.000419, 12);
      expect(sqlite.prepare('SELECT tokens FROM terra_daily').get()).toEqual({ tokens: 120 });
      sqlite.prepare('UPDATE terra_daily SET tokens = ?').run(terraDailyTokens);
      const luna = await checkAllowance(env, otherInstallation, turnId, false, start);
      if (!('accounting' in luna)) throw new Error(luna.error);
      await usageRecorder(luna.accounting, false)(result);
      expect(sqlite.prepare('SELECT cost_usd FROM usage_daily WHERE installation_hash = ?').get(otherInstallation)?.cost_usd)
        .toBeCloseTo(0.00022, 12);
      expect(sqlite.prepare('SELECT tokens FROM terra_daily').get()).toEqual({ tokens: terraDailyTokens });
    } finally { sqlite.close(); }
  });
});
