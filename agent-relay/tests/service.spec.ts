import { DatabaseSync } from 'node:sqlite';
import { afterEach, describe, expect, it, vi } from 'vitest';

import relay from '../src/index';
import { accountingDatabase, authHeaders, installationHash, responseEvents, signToken, testEnv, tokenSecret, turnId, usageDatabase } from './fixtures';

const classify = { agent_protocol: 6, turn_id: turnId, input: [{ role: 'user', content: 'Synthetic request' }] };
const respond = { agent_protocol: 6, turn_id: turnId, input: [
  { type: 'configuration_update', reasoning: { effort: 'high' } }, ...classify.input,
], route: 'work', effort: 'high', allow_tools: true };

function request(path = '/v1/respond', body: unknown = respond, authorization = authHeaders().authorization): Request {
  return new Request(`https://relay.example${path}`, { method: 'POST', headers: { authorization }, body: JSON.stringify(body) });
}

async function settled(request: Request, env: Parameters<typeof relay.fetch>[1]): Promise<Response> {
  const response = await relay.fetch(request, env);
  await response.clone().arrayBuffer();
  return response;
}

function result() {
  return { object: 'response', status: 'completed', output: [], answers: [], usage: { input_tokens: 100,
    input_tokens_details: { cached_tokens: 20, cache_write_tokens: 10 }, output_tokens: 20,
    output_tokens_details: { reasoning_tokens: 15 }, total_tokens: 120 } };
}

function seedTurns(sqlite: DatabaseSync, installation: string, day: string, count: number): void {
  const seed = sqlite.prepare('INSERT INTO prompt_turns (installation_hash, turn_id, utc_date) VALUES (?, ?, ?)');
  const prefix = day.replace(/-/g, '');
  for (let index = 0; index < count; index++) {
    seed.run(installation, `${prefix}${index.toString(16).padStart(24, '0')}`, day);
  }
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.useRealTimers();
});

describe('bounded relay service', () => {
  it('requires a signed short-lived broker identity on every content endpoint', async () => {
    const upstream = vi.spyOn(globalThis, 'fetch');
    const env = testEnv();
    const now = Math.floor(Date.now() / 1000);
    const invalid = ['', 'Bearer raw-installation-id', `Bearer ${signToken({}, { alg: 'none', typ: 'JWT' })}`,
      `Bearer ${signToken({}, { alg: 'HS256', typ: 'JWT', kid: 'untrusted' })}`,
      `Bearer ${signToken({}, { alg: 'HS256', typ: 'not-JWT' })}`,
      `Bearer ${signToken({}, undefined, 'different-synthetic-secret-at-least32bytes')}`,
      ...[{ iss: 'other' }, { aud: ['puripuly-agent-relay'] }, { sub: 'raw-installation-id' }, { sub: 'A'.repeat(64) },
        { iat: now + 1 }, { exp: now }, { exp: now + 3601 }, { iat: now + 0.5 }, { exp: String(now + 3600) }]
        .map((claims) => `Bearer ${signToken(claims)}`)];
    for (const [path, body] of [['/v1/respond', respond], ['/v1/classify', classify],
      ['/v1/support', { private: 'must not be read' }]] as const) {
      for (const authorization of invalid) {
        const response = await relay.fetch(request(path, body, authorization), env);
        expect(response.status).toBe(401);
        expect(await response.json()).toMatchObject({ error: { code: 'service_authentication_failed' } });
      }
    }
    for (const secret of [undefined, '', 'short']) {
      expect((await relay.fetch(request(), { ...env, PURI_TOKEN_SECRET: secret })).status).toBe(503);
    }
    expect(upstream).not.toHaveBeenCalled();
  });

  it('verifies each token and its expiry across verification-key reuse and secret changes', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-11-01T00:00:00Z'));
    const alternate = 'synthetic-rotated-relay-token-secret-32-bytes';
    const originalToken = signToken();
    const alternateToken = signToken({}, undefined, alternate);
    const upstream = vi.spyOn(globalThis, 'fetch').mockImplementation(async () => Response.json(result()));
    for (const [secret, token, status] of [
      [tokenSecret, originalToken, 200],
      [alternate, originalToken, 401],
      [alternate, alternateToken, 200],
      [tokenSecret, alternateToken, 401],
      [tokenSecret, originalToken, 200],
    ] as const) {
      const response = await relay.fetch(request('/v1/classify', classify, `Bearer ${token}`), { ...testEnv(), PURI_TOKEN_SECRET: secret });
      expect(response.status).toBe(status);
      if (status === 401) {
        expect(await response.json()).toMatchObject({ error: { code: 'service_authentication_failed' } });
      }
    }
    vi.advanceTimersByTime(3599000);
    expect((await relay.fetch(request('/v1/classify', classify, `Bearer ${originalToken}`), testEnv())).status).toBe(200);
    vi.advanceTimersByTime(1000);
    const expired = await relay.fetch(request('/v1/classify', classify, `Bearer ${originalToken}`), testEnv());
    expect(expired.status).toBe(401);
    expect(await expired.json()).toMatchObject({ error: { code: 'service_authentication_failed' } });
    expect(upstream).toHaveBeenCalledTimes(4);
  });

  it('recovers after a verification-key preparation failure without retaining a failed key', async () => {
    const secret = 'synthetic-transient-import-failure-secret-32-bytes';
    const token = signToken({}, undefined, secret);
    const upstream = vi.spyOn(globalThis, 'fetch').mockImplementation(async () => Response.json(result()));
    const env = { ...testEnv(), PURI_TOKEN_SECRET: secret };
    vi.spyOn(crypto.subtle, 'importKey').mockRejectedValueOnce(new Error('Synthetic key preparation failure'));
    const failed = await relay.fetch(request('/v1/classify', classify, `Bearer ${token}`), env);
    expect(failed.status).toBe(401);
    expect(await failed.json()).toMatchObject({ error: { code: 'service_authentication_failed' } });
    expect((await relay.fetch(request('/v1/classify', classify, `Bearer ${token}`), env)).status).toBe(200);
    expect(upstream).toHaveBeenCalledTimes(1);
  });

  it('keeps support disabled without reading or claiming receipt until configured', async () => {
    const upstream = vi.spyOn(globalThis, 'fetch');
    const upload = request('/v1/support', { private: 'synthetic report' });
    const read = vi.spyOn(upload, 'json');
    const response = await relay.fetch(upload, { PURI_TOKEN_SECRET: tokenSecret });
    expect(response.status).toBe(503);
    expect(await response.json()).toEqual({ error: { code: 'support_unavailable',
      message: 'Support reports are unavailable. No report was accepted or stored.' } });
    expect(upload.bodyUsed).toBe(false);
    expect(read).not.toHaveBeenCalled();
    expect(upstream).not.toHaveBeenCalled();
  });

  it('accumulates D and M with mutually exclusive cache categories and output charged once', async () => {
    const upstream = vi.spyOn(globalThis, 'fetch').mockImplementation(async (url) =>
      String(url).endsWith('/decisions') ? Response.json(result()) : responseEvents(result()));
    const database = usageDatabase();
    const env = { ...testEnv(), AGENT_USAGE_DB: database.db };
    expect((await relay.fetch(request('/v1/classify', classify), env)).status).toBe(200);
    expect((await settled(request(), env)).status).toBe(200);
    expect(database.admission).toHaveBeenCalledTimes(2);
    expect(database.writes).toHaveLength(2);
    const [d, m] = database.writes;
    expect(d.slice(0, 10)).toEqual([installationHash, new Date().toISOString().slice(0, 10), 70, 20, 10, 20, 0, 1, 0, 0]);
    expect(m.slice(0, 10)).toEqual([installationHash, new Date().toISOString().slice(0, 10), 70, 20, 10, 0, 20, 0, 1, 0]);
    expect(d[10]).toBeCloseTo(0.00000845, 12);
    expect(m[10]).toBeCloseTo(0.000409, 12);
    expect(upstream).toHaveBeenCalledTimes(2);
  });

  it('refuses both D and M before upstream when no new prompt turn can be admitted', async () => {
    const upstream = vi.spyOn(globalThis, 'fetch');
    const database = usageDatabase();
    const env = { ...testEnv(), AGENT_USAGE_DB: database.db };
    database.admission.mockResolvedValue(null);
    for (const [path, body] of [['/v1/classify', classify], ['/v1/respond', respond]] as const) {
      const response = await relay.fetch(request(path, body), env);
      expect(response.status).toBe(429);
      expect(await response.json()).toMatchObject({ error: { code: 'service_allowance_exhausted' } });
    }
    expect(database.writes).toEqual([]);
    expect(upstream).not.toHaveBeenCalled();
  });

  it.each([['/v1/classify', classify], ['/v1/respond', respond]] as const)(
    'requires a valid prompt turn ID through %s without allowing legacy clients to bypass admission', async (path, body) => {
      const upstream = vi.spyOn(globalThis, 'fetch');
      const database = usageDatabase();
      const env = { ...testEnv(), AGENT_USAGE_DB: database.db };
      const { turn_id, ...legacy } = body;
      const missing = await relay.fetch(request(path, legacy), env);
      expect(missing.status).toBe(400);
      expect(await missing.json()).toMatchObject({ error: { code: 'unsupported_agent_protocol' } });
      for (const invalid of [null, '', turn_id.slice(1), `${turn_id}0`, turn_id.toUpperCase(), 'g'.repeat(32), 32, [turn_id]]) {
        const response = await relay.fetch(request(path, { ...body, turn_id: invalid }), env);
        expect(response.status).toBe(400);
        expect(await response.json()).toMatchObject({ error: { code: 'invalid_request' } });
      }
      expect(database.admission).not.toHaveBeenCalled();
      expect(database.writes).toEqual([]);
      expect(upstream).not.toHaveBeenCalled();
    });

  it.each([['/v1/classify', classify], ['/v1/respond', respond]] as const)(
    'rejects the retired chat protocol through %s before accounting or model access', async (path, body) => {
      const upstream = vi.spyOn(globalThis, 'fetch');
      const database = usageDatabase();
      const response = await relay.fetch(request(path, { ...body, agent_protocol: 3 }),
        { ...testEnv(), AGENT_USAGE_DB: database.db });
      expect(response.status).toBe(400);
      expect(await response.json()).toMatchObject({ error: { code: 'unsupported_agent_protocol' } });
      expect(database.admission).not.toHaveBeenCalled();
      expect(database.writes).toEqual([]);
      expect(upstream).not.toHaveBeenCalled();
    });

  it.each([['/v1/classify', classify], ['/v1/respond', respond]] as const)(
    'admits the 30th prompt through %s independently of dollars and keeps D, M, tools and notices on that turn', async (path, body) => {
      vi.useFakeTimers();
      vi.setSystemTime(new Date('2026-10-30T12:00:00Z'));
      const upstream = vi.spyOn(globalThis, 'fetch').mockImplementation(async (url) =>
        String(url).endsWith('/decisions') ? Response.json(result()) : responseEvents(result()));
      const { sqlite, db } = accountingDatabase();
      const env = { ...testEnv(), AGENT_USAGE_DB: db };
      const count = sqlite.prepare('SELECT COUNT(*) AS total FROM prompt_turns WHERE installation_hash = ? AND utc_date = ?');
      const storedUsage = sqlite.prepare('SELECT * FROM usage_daily WHERE installation_hash = ? AND utc_date = ?');
      try {
        seedTurns(sqlite, installationHash, '2026-10-30', 29);
        seedTurns(sqlite, 'b'.repeat(64), '2026-10-30', 30);
        seedTurns(sqlite, installationHash, '2026-10-29', 30);
        sqlite.prepare('INSERT INTO usage_daily (installation_hash, utc_date, decisions_calls, responses_calls, cost_usd) VALUES (?, ?, 100, 100, 2)')
          .run(installationHash, '2026-10-30');
        expect((await settled(request(path, body), env)).status).toBe(200);
        const continuationPath = path === '/v1/classify' ? '/v1/respond' : '/v1/classify';
        const continuationBody = path === '/v1/classify' ? respond : classify;
        expect((await settled(request(continuationPath, continuationBody), env)).status).toBe(200);
        expect(count.get(installationHash, '2026-10-30')).toEqual({ total: 30 });
        const row = storedUsage.get(installationHash, '2026-10-30');
        expect(row).toMatchObject({ decisions_calls: 101, responses_calls: 101, uncached_input_tokens: 140,
          cache_read_tokens: 40, cache_write_tokens: 20, decisions_output_tokens: 20, responses_output_tokens: 20,
          missing_usage_count: 0 });
        expect(row?.cost_usd).toBeCloseTo(2.00041745, 12);
        expect(sqlite.prepare('SELECT agent_protocol, requests FROM protocol_daily WHERE installation_hash = ? AND utc_date = ?')
          .all(installationHash, '2026-10-30')).toEqual([{ agent_protocol: 6, requests: 2 }]);
        sqlite.prepare('UPDATE usage_daily SET cost_usd = 0.5 WHERE installation_hash = ? AND utc_date = ?')
          .run(installationHash, '2026-10-30');
        const exhausted = storedUsage.get(installationHash, '2026-10-30');
        for (const [nextPath, nextBody] of [['/v1/classify', classify], ['/v1/respond', respond]] as const) {
          const response = await relay.fetch(request(nextPath, { ...nextBody, turn_id: 'f'.repeat(32) }), env);
          expect(response.status).toBe(429);
          expect(await response.json()).toMatchObject({ error: { code: 'service_allowance_exhausted' } });
        }
        expect(storedUsage.get(installationHash, '2026-10-30')).toEqual(exhausted);
        expect(upstream).toHaveBeenCalledTimes(2);
        const duplicateClassifier = await relay.fetch(request('/v1/classify', classify), env);
        expect(duplicateClassifier.status).toBe(429);
        expect(await duplicateClassifier.json()).toMatchObject({ error: { code: 'service_turn_exhausted' } });
        const input = [...respond.input,
          { type: 'function_call', call_id: 'tool1', name: 'read_app', arguments: '{"topic":"overview"}' },
          { type: 'function_call_output', call_id: 'tool1', output: '{}' },
          { role: 'user', content: '[app notice] Synthetic state change' }];
        expect((await settled(request('/v1/respond', { ...respond, input }), env)).status).toBe(200);
        expect(count.get(installationHash, '2026-10-30')).toEqual({ total: 30 });
        expect(storedUsage.get(installationHash, '2026-10-30')).toMatchObject({ decisions_calls: 101, responses_calls: 102 });
        expect((await settled(request('/v1/respond', respond, `Bearer ${signToken({ sub: 'c'.repeat(64) })}`), env)).status).toBe(200);
        expect(count.get('c'.repeat(64), '2026-10-30')).toEqual({ total: 1 });
        expect(upstream).toHaveBeenCalledTimes(4);
      } finally {
        sqlite.close();
      }
    });

  it('resets new prompt admission at UTC midnight without charging an old turn again and records each call on its current day', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-10-30T23:59:59Z'));
    const upstream = vi.spyOn(globalThis, 'fetch').mockImplementation(async (url) =>
      String(url).endsWith('/decisions') ? Response.json(result()) : responseEvents(result()));
    const { sqlite, db } = accountingDatabase();
    const env = { ...testEnv(), AGENT_USAGE_DB: db };
    const count = sqlite.prepare('SELECT COUNT(*) AS total FROM prompt_turns WHERE installation_hash = ? AND utc_date = ?');
    try {
      seedTurns(sqlite, installationHash, '2026-10-30', 29);
      expect((await settled(request('/v1/classify', classify), env)).status).toBe(200);
      expect(count.get(installationHash, '2026-10-30')).toEqual({ total: 30 });
      vi.setSystemTime(new Date('2026-10-31T00:00:00Z'));
      expect((await settled(request(), env)).status).toBe(200);
      expect(count.get(installationHash, '2026-10-31')).toEqual({ total: 0 });
      expect(sqlite.prepare('SELECT utc_date FROM prompt_turns WHERE installation_hash = ? AND turn_id = ?')
        .get(installationHash, turnId)).toEqual({ utc_date: '2026-10-30' });
      for (const [path, body] of [['/v1/classify', classify], ['/v1/respond', respond]] as const) {
        expect((await settled(request(path, { ...body, turn_id: 'f'.repeat(32) }), env)).status).toBe(200);
      }
      expect(count.get(installationHash, '2026-10-31')).toEqual({ total: 1 });
      const rows = sqlite.prepare('SELECT * FROM usage_daily WHERE installation_hash = ? ORDER BY utc_date').all(installationHash);
      expect(rows[0]).toMatchObject({ utc_date: '2026-10-30', decisions_calls: 1, responses_calls: 0,
        uncached_input_tokens: 70, cache_read_tokens: 20, cache_write_tokens: 10, decisions_output_tokens: 20,
        responses_output_tokens: 0, missing_usage_count: 0 });
      expect(rows[0].cost_usd).toBeCloseTo(0.00000845, 12);
      expect(rows[1]).toMatchObject({ utc_date: '2026-10-31', decisions_calls: 1, responses_calls: 2,
        uncached_input_tokens: 210, cache_read_tokens: 60, cache_write_tokens: 30, decisions_output_tokens: 20,
        responses_output_tokens: 40, missing_usage_count: 0 });
      expect(rows[1].cost_usd).toBeCloseTo(0.00082645, 12);
      expect(upstream).toHaveBeenCalledTimes(4);
    } finally {
      sqlite.close();
    }
  });

  it.each([
    { name: 'duplicate', turns: [turnId, turnId, turnId] },
    { name: 'different', turns: [turnId, 'e'.repeat(32), 'f'.repeat(32)] },
    { name: 'mixed duplicate and different', turns: [turnId, turnId, 'f'.repeat(32)] },
  ])('atomically admits concurrent $name turn IDs at the 29-to-30 boundary', async ({ turns }) => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-10-30T12:00:00Z'));
    const upstream = vi.spyOn(globalThis, 'fetch').mockImplementation(async (url) =>
      String(url).endsWith('/decisions') ? Response.json(result()) : responseEvents(result()));
    const { sqlite, db } = accountingDatabase();
    const env = { ...testEnv(), AGENT_USAGE_DB: db };
    try {
      seedTurns(sqlite, installationHash, '2026-10-30', 29);
      const responses = await Promise.all(turns.map((turn_id, index) =>
        settled(request(index % 2 === 0 ? '/v1/classify' : '/v1/respond',
          { ...(index % 2 === 0 ? classify : respond), turn_id }), env)));
      const admitted = sqlite.prepare('SELECT turn_id FROM prompt_turns WHERE installation_hash = ? AND turn_id IN (?, ?, ?)')
        .all(installationHash, ...turns);
      expect(admitted).toHaveLength(1);
      for (let index = 0; index < turns.length; index++) {
        if (turns[index] !== admitted[0].turn_id) {
          expect(responses[index].status).toBe(429);
          expect(await responses[index].json()).toMatchObject({ error: { code: 'service_allowance_exhausted' } });
        } else {
          expect([200, 429]).toContain(responses[index].status);
          if (responses[index].status === 429) {
            expect((await responses[index].json() as { error: { code: string } }).error.code)
              .toMatch(/^service_(busy|turn_exhausted)$/);
          }
        }
      }
      expect(sqlite.prepare('SELECT COUNT(*) AS total FROM prompt_turns WHERE installation_hash = ? AND utc_date = ?')
        .get(installationHash, '2026-10-30')).toEqual({ total: 30 });
      const completed = responses.filter((response) => response.status === 200).length;
      expect(upstream).toHaveBeenCalledTimes(completed);
      expect(sqlite.prepare('SELECT decisions_calls + responses_calls AS calls FROM usage_daily WHERE installation_hash = ? AND utc_date = ?')
        .get(installationHash, '2026-10-30')).toEqual({ calls: completed });
    } finally {
      sqlite.close();
    }
  });

  it.each([['/v1/classify', classify], ['/v1/respond', respond]] as const)(
    'keeps a failed %s and missing or invalid usage continuations on one admitted prompt without retaining raw data', async (path, body) => {
      vi.useFakeTimers();
      vi.setSystemTime(new Date('2026-10-30T12:00:00Z'));
      const upstream = vi.spyOn(globalThis, 'fetch').mockImplementation(async () =>
        responseEvents({ object: 'response', status: 'completed', output: [] }));
      const { sqlite, db } = accountingDatabase();
      const env = { ...testEnv(), AGENT_USAGE_DB: db };
      try {
        seedTurns(sqlite, installationHash, '2026-10-30', 29);
        upstream.mockRejectedValueOnce(new Error('private synthetic content'));
        expect((await relay.fetch(request(path, body), env)).status).toBe(502);
        expect((await settled(request(), env)).status).toBe(200);
        upstream.mockResolvedValueOnce(responseEvents({ ...result(), usage: { ...result().usage, input_tokens: 1 } }));
        expect((await settled(request(), env)).status).toBe(200);
        const rows = sqlite.prepare('SELECT * FROM usage_daily').all();
        expect(rows).toHaveLength(1);
        expect(rows[0]).toMatchObject({ uncached_input_tokens: 0, cache_read_tokens: 0, cache_write_tokens: 0,
          decisions_output_tokens: 0, responses_output_tokens: 0, decisions_calls: path === '/v1/classify' ? 1 : 0,
          responses_calls: path === '/v1/classify' ? 2 : 3, missing_usage_count: 3, cost_usd: 0 });
        const turns = sqlite.prepare('SELECT * FROM prompt_turns WHERE installation_hash = ?').all(installationHash);
        expect(turns).toHaveLength(30);
        expect(turns.filter((turn) => turn.turn_id === turnId)).toHaveLength(1);
        expect(JSON.stringify({ rows, turns })).not.toContain('private');
        expect((await relay.fetch(request('/v1/respond', { ...respond, turn_id: 'f'.repeat(32) }), env)).status).toBe(429);
        expect(upstream).toHaveBeenCalledTimes(3);
      } finally {
        sqlite.close();
      }
    });

  it('fails closed before calls and reports post-call accounting uncertainty with no raw logs', async () => {
    const upstream = vi.spyOn(globalThis, 'fetch').mockImplementation(async () => responseEvents(result()));
    const log = vi.spyOn(console, 'log');
    const warn = vi.spyOn(console, 'warn');
    const error = vi.spyOn(console, 'error');
    const env = testEnv();
    expect((await relay.fetch(request(), { ...env, AGENT_USAGE_DB: undefined })).status).toBe(503);
    const database = usageDatabase();
    database.admission.mockRejectedValueOnce(new Error('private database details'));
    const configured = { ...env, AGENT_USAGE_DB: database.db };
    const before = await relay.fetch(request(), configured);
    expect(before.status).toBe(503);
    expect(await before.text()).toContain('No model request was started');
    expect(upstream).not.toHaveBeenCalled();
    for (const admitted of [undefined, false, 0, '0', {}, { turn_id: null }, { turn_id: 'f'.repeat(32) }]) {
      database.admission.mockResolvedValueOnce(admitted);
      expect((await relay.fetch(request(), configured)).status).toBe(503);
    }
    expect(upstream).not.toHaveBeenCalled();
    database.run.mockRejectedValueOnce(new Error('private database details'));
    const after = await settled(request(), configured);
    expect(after.status).toBe(200);
    expect(await after.text()).toContain('service_shutdown');
    expect(upstream).toHaveBeenCalledTimes(1);
    expect(log).not.toHaveBeenCalled();
    expect(warn).not.toHaveBeenCalled();
    expect(error).not.toHaveBeenCalled();
  });

  it('accepts explicit dated Luna rates and rejects unusable configuration before upstream', async () => {
    const upstream = vi.spyOn(globalThis, 'fetch').mockImplementation(async () => responseEvents(result()));
    const database = usageDatabase();
    database.admission.mockImplementation(async (turn_id: string) =>
      ({ turn_id, model_admissions: 1, expires_at: Date.now() + 86_400_000, response_model: 'gpt-6-luna' }));
    const env = { ...testEnv(), AGENT_USAGE_DB: database.db };
    const rates = { as_of: '2026-10-30', input: 1, cache_read: 2, cache_write: 3, responses_output: 4, decisions_output: 5 };
    expect((await settled(request(), { ...env, AGENT_USAGE_RATES: JSON.stringify(rates) })).status).toBe(200);
    expect(database.writes[0][10]).toBeCloseTo(0.00022, 12);
    for (const invalid of ['{', JSON.stringify({ ...rates, input: -1 }), JSON.stringify({ ...rates, cache_read: null })]) {
      expect((await relay.fetch(request(), { ...env, AGENT_USAGE_RATES: invalid })).status).toBe(503);
    }
    expect(upstream).toHaveBeenCalledTimes(1);
    expect(database.admission).toHaveBeenCalledTimes(1);
  });
});
