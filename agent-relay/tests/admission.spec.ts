import { readFileSync } from 'node:fs';
import { afterEach, describe, expect, it, vi } from 'vitest';

import relay from '../src/index';
import { requestDeadlineMs, requestLeaseMs } from '../src/lifetime';
import { maxClassifyBytes, maxRespondBytes } from '../src/requests';
import { checkAllowance, recordUsage, turnLifetimeMs } from '../src/usage';
import { accountingDatabase, authHeaders, installationHash, relayEvents, relayMigrations, responseEvents, testEnv, turnId, usageDatabase } from './fixtures';

const classify = { agent_protocol: 6, turn_id: turnId, input: [{ role: 'user', content: 'Synthetic request' }] };
const respond = { agent_protocol: 6, turn_id: turnId, input: [
  { type: 'configuration_update', reasoning: { effort: 'high' } }, ...classify.input,
], route: 'work', effort: 'high', allow_tools: true };
const returned = { object: 'response', status: 'completed', output: [], answers: [], usage: { input_tokens: 10, output_tokens: 2 } };
const start = new Date('2026-11-01T12:00:00Z');

function request(classification = false, body: unknown = classification ? classify : respond): Request {
  return new Request(`https://relay.example/v1/${classification ? 'classify' : 'respond'}`, {
    method: 'POST', headers: authHeaders(), body: JSON.stringify(body),
  });
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.useRealTimers();
});

describe('durable model request admission', () => {
  it('shares all 15 admissions across D, M, tools and notices and makes the last response tool-free', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(start);
    const upstream = vi.spyOn(globalThis, 'fetch').mockImplementation(async (url) =>
      String(url).endsWith('/decisions') ? Response.json(returned) : responseEvents(returned));
    const { sqlite, db } = accountingDatabase();
    const env = { ...testEnv(), AGENT_USAGE_DB: db };
    try {
      expect((await relay.fetch(request(true), env)).status).toBe(200);
      for (let admission = 2; admission <= 15; admission++) {
        const response = await relay.fetch(request(false, { ...respond, input: [...respond.input,
          { type: 'function_call', call_id: `call-${admission}`, name: 'read_app', arguments: '{"topic":"overview"}' },
          { type: 'function_call_output', call_id: `call-${admission}`, output: '{}' },
          { role: 'user', content: '[app notice] Synthetic state change' },
        ] }), env);
        expect(await relayEvents(response)).toEqual([{ type: 'completed', response: returned }]);
        expect(JSON.parse(upstream.mock.calls[admission - 1][1]?.body as string).tool_choice)
          .toBe(admission === 15 ? 'none' : 'auto');
      }
      for (const classification of [false, true]) {
        const rejected = await relay.fetch(request(classification), env);
        expect(rejected.status).toBe(429);
        expect(await rejected.json()).toMatchObject({ error: { code: 'service_turn_exhausted' } });
      }
      expect(upstream).toHaveBeenCalledTimes(15);
      expect(sqlite.prepare('SELECT model_admissions, classifier_admitted, expires_at FROM prompt_turns').get())
        .toEqual({ model_admissions: 15, classifier_admitted: 1, expires_at: start.getTime() + turnLifetimeMs });
      expect(sqlite.prepare('SELECT decisions_calls, responses_calls FROM usage_daily').get())
        .toEqual({ decisions_calls: 1, responses_calls: 14 });
      expect(sqlite.prepare('SELECT * FROM request_slots').all()).toEqual([]);
    } finally {
      sqlite.close();
    }
  });

  it('atomically admits only one classifier even when the same turn races', async () => {
    const { sqlite, db } = accountingDatabase();
    try {
      const attempts = await Promise.all(Array.from({ length: 4 }, () =>
        checkAllowance({ AGENT_USAGE_DB: db }, installationHash, turnId, true, start)));
      expect(attempts.filter((result) => 'accounting' in result)).toHaveLength(1);
      expect(attempts.filter((result) => 'error' in result))
        .toEqual(Array(3).fill({ error: 'service_turn_exhausted' }));
      expect(sqlite.prepare('SELECT model_admissions, classifier_admitted FROM prompt_turns').get())
        .toEqual({ model_admissions: 1, classifier_admitted: 1 });
      expect(sqlite.prepare('SELECT COUNT(*) AS count FROM request_slots').get()).toEqual({ count: 1 });
    } finally {
      sqlite.close();
    }
  });

  it('does not overshoot the 14-to-15 boundary under mixed classifier and response races', async () => {
    const { sqlite, db } = accountingDatabase();
    try {
      sqlite.prepare(`INSERT INTO prompt_turns
        (installation_hash, turn_id, utc_date, expires_at, model_admissions, classifier_admitted)
        VALUES (?, ?, ?, ?, 14, 0)`).run(installationHash, turnId, '2026-11-01', start.getTime() + turnLifetimeMs);
      const attempts = await Promise.all([true, false, false].map((classification) =>
        checkAllowance({ AGENT_USAGE_DB: db }, installationHash, turnId, classification, start)));
      expect(attempts.filter((result) => 'accounting' in result)).toHaveLength(1);
      expect(attempts.filter((result) => 'error' in result))
        .toEqual(Array(2).fill({ error: 'service_turn_exhausted' }));
      expect(sqlite.prepare('SELECT model_admissions FROM prompt_turns').get()).toEqual({ model_admissions: 15 });
      expect(sqlite.prepare('SELECT COUNT(*) AS count FROM request_slots').get()).toEqual({ count: 1 });
    } finally {
      sqlite.close();
    }
  });

  it('fixes lifetime at first admission, never recreates expired IDs and keeps UTC daily counting independent', async () => {
    const { sqlite, db } = accountingDatabase();
    const env = { AGENT_USAGE_DB: db };
    try {
      const first = await checkAllowance(env, installationHash, turnId, true, start);
      if (!('accounting' in first)) throw new Error('first admission failed');
      await recordUsage(first.accounting, returned, true);
      const lastMoment = new Date(start.getTime() + turnLifetimeMs - 1);
      const continuation = await checkAllowance(env, installationHash, turnId, false, lastMoment);
      if (!('accounting' in continuation)) throw new Error('continuation admission failed');
      expect(continuation.accounting.deadline).toBe(start.getTime() + turnLifetimeMs);
      await recordUsage(continuation.accounting, returned, false);
      for (const elapsed of [turnLifetimeMs, turnLifetimeMs * 2]) {
        for (const classification of [true, false]) {
          expect(await checkAllowance(env, installationHash, turnId, classification, new Date(start.getTime() + elapsed)))
            .toEqual({ error: 'service_turn_exhausted' });
        }
      }
      const nextDay = new Date(start.getTime() + turnLifetimeMs);
      const newTurn = await checkAllowance(env, installationHash, 'f'.repeat(32), false, nextDay);
      expect(newTurn).toHaveProperty('accounting');
      expect(await checkAllowance(env, 'b'.repeat(64), turnId, false, nextDay)).toHaveProperty('accounting');
      expect(sqlite.prepare('SELECT utc_date, model_admissions, expires_at FROM prompt_turns WHERE installation_hash = ? AND turn_id = ?')
        .get(installationHash, turnId)).toEqual({ utc_date: '2026-11-01', model_admissions: 2,
        expires_at: start.getTime() + turnLifetimeMs });
      expect(sqlite.prepare('SELECT utc_date, COUNT(*) AS count FROM prompt_turns WHERE installation_hash = ? GROUP BY utc_date ORDER BY utc_date')
        .all(installationHash)).toEqual([{ utc_date: '2026-11-01', count: 1 }, { utc_date: '2026-11-02', count: 1 }]);
    } finally {
      sqlite.close();
    }
  });

  it('caps installation concurrency at two and response concurrency at one without consuming denied admissions', async () => {
    const { sqlite, db } = accountingDatabase();
    const env = { AGENT_USAGE_DB: db };
    try {
      const response = await checkAllowance(env, installationHash, turnId, false, start);
      const classifier = await checkAllowance(env, installationHash, turnId, true, start);
      if (!('accounting' in response) || !('accounting' in classifier)) throw new Error('initial admissions failed');
      expect(await checkAllowance(env, installationHash, 'e'.repeat(32), false, start)).toEqual({ error: 'service_busy' });
      expect(await checkAllowance(env, installationHash, turnId, false, start)).toEqual({ error: 'service_busy' });
      expect(await checkAllowance(env, 'b'.repeat(64), 'e'.repeat(32), false, start)).toHaveProperty('accounting');
      await recordUsage(classifier.accounting, returned, true);
      expect(await checkAllowance(env, installationHash, turnId, false, start)).toEqual({ error: 'service_busy' });
      const otherTurn = await checkAllowance(env, installationHash, 'e'.repeat(32), false, start);
      expect(otherTurn).toHaveProperty('accounting');
      expect(sqlite.prepare('SELECT model_admissions FROM prompt_turns WHERE installation_hash = ? AND turn_id = ?')
        .get(installationHash, turnId)).toEqual({ model_admissions: 2 });
      await recordUsage(response.accounting, returned, false);
      expect(await checkAllowance(env, installationHash, turnId, false, start)).toHaveProperty('accounting');
      expect(sqlite.prepare('SELECT COUNT(*) AS count FROM request_slots WHERE installation_hash = ?').get(installationHash))
        .toEqual({ count: 2 });
    } finally {
      sqlite.close();
    }
  });

  it('atomically prevents a third simultaneous installation admission across distinct turns', async () => {
    const { sqlite, db } = accountingDatabase();
    try {
      const attempts = await Promise.all([turnId, 'e'.repeat(32), 'f'.repeat(32)].map((id) =>
        checkAllowance({ AGENT_USAGE_DB: db }, installationHash, id, false, start)));
      expect(attempts.filter((result) => 'accounting' in result)).toHaveLength(2);
      expect(attempts.filter((result) => 'error' in result)).toEqual([{ error: 'service_busy' }]);
      expect(sqlite.prepare('SELECT COUNT(*) AS count FROM prompt_turns').get()).toEqual({ count: 2 });
      expect(sqlite.prepare('SELECT COUNT(*) AS count FROM request_slots').get()).toEqual({ count: 2 });
    } finally {
      sqlite.close();
    }
  });

  it('recovers an uncertain lease by expiry while stale settlement cannot release its replacement', async () => {
    const { sqlite, db } = accountingDatabase();
    const env = { AGENT_USAGE_DB: db };
    try {
      const original = await checkAllowance(env, installationHash, turnId, false, start);
      const replacement = await checkAllowance(env, installationHash, turnId, false, new Date(start.getTime() + requestLeaseMs));
      if (!('accounting' in original) || !('accounting' in replacement)) throw new Error('lease admission failed');
      expect(original.accounting.owner).not.toBe(replacement.accounting.owner);
      await recordUsage(original.accounting, undefined, false);
      expect(sqlite.prepare('SELECT owner FROM request_slots').get()).toEqual({ owner: replacement.accounting.owner });
      await recordUsage(replacement.accounting, returned, false);
      expect(sqlite.prepare('SELECT * FROM request_slots').all()).toEqual([]);
      expect(sqlite.prepare('SELECT responses_calls, missing_usage_count FROM usage_daily').get())
        .toEqual({ responses_calls: 2, missing_usage_count: 1 });
    } finally {
      sqlite.close();
    }
  });

  it('replaces fixed slots rather than accumulating expired request IDs', async () => {
    const { sqlite, db } = accountingDatabase();
    try {
      for (let generation = 0; generation < 12; generation++) {
        expect(await checkAllowance({ AGENT_USAGE_DB: db }, installationHash, turnId, false,
          new Date(start.getTime() + generation * requestLeaseMs))).toHaveProperty('accounting');
      }
      expect(sqlite.prepare('SELECT COUNT(*) AS count FROM request_slots').get()).toEqual({ count: 1 });
      expect(sqlite.prepare('SELECT model_admissions FROM prompt_turns').get()).toEqual({ model_admissions: 12 });
    } finally {
      sqlite.close();
    }
  });

  it('migrates pre-existing prompt turns conservatively without resetting their IDs or daily allowance', async () => {
    const { sqlite, db } = accountingDatabase(relayMigrations.slice(0, 3));
    try {
      const seed = sqlite.prepare('INSERT INTO prompt_turns (installation_hash, turn_id, utc_date) VALUES (?, ?, ?)');
      for (let index = 0; index < 29; index++) seed.run(installationHash, index.toString(16).padStart(32, '0'), '2026-11-01');
      for (const migration of relayMigrations.slice(3)) {
        sqlite.exec(readFileSync(new URL(`../migrations/${migration}`, import.meta.url), 'utf8'));
      }
      const legacyId = '0'.repeat(32);
      for (const classification of [false, true]) {
        expect(await checkAllowance({ AGENT_USAGE_DB: db }, installationHash, legacyId, classification, start))
          .toEqual({ error: 'service_turn_exhausted' });
      }
      expect(await checkAllowance({ AGENT_USAGE_DB: db }, installationHash, turnId, false, start)).toHaveProperty('accounting');
      expect(await checkAllowance({ AGENT_USAGE_DB: db }, installationHash, 'f'.repeat(32), false, start))
        .toEqual({ error: 'service_allowance_exhausted' });
      expect(sqlite.prepare('SELECT utc_date, model_admissions, classifier_admitted, expires_at FROM prompt_turns WHERE turn_id = ?')
        .get(legacyId)).toEqual({ utc_date: '2026-11-01', model_admissions: 15, classifier_admitted: 1, expires_at: 0 });
      expect(sqlite.prepare('SELECT COUNT(*) AS count FROM prompt_turns').get()).toEqual({ count: 30 });
    } finally {
      sqlite.close();
    }
  });
});

describe('actual JSON request byte boundaries', () => {
  it.each([[true, maxClassifyBytes], [false, maxRespondBytes]] as const)(
    'bounds classify=%s before parsing or admission, including fragmented UTF-8 and dishonest lengths', async (classification, maximum) => {
      const upstream = vi.spyOn(globalThis, 'fetch').mockImplementation(async () =>
        classification ? Response.json(returned) : responseEvents(returned));
      const database = usageDatabase();
      const env = { ...testEnv(), AGENT_USAGE_DB: database.db };
      const base = classification ? { ...classify, input: [{ role: 'user', content: '가' }] }
        : { ...respond, input: [respond.input[0], { role: 'user', content: '가' }] };
      const overhead = new TextEncoder().encode(JSON.stringify(base)).byteLength;
      const content = `가${'x'.repeat(maximum - overhead)}`;
      const payload = classification ? { ...base, input: [{ role: 'user', content }] }
        : { ...base, input: [respond.input[0], { role: 'user', content }] };
      const bytes = new TextEncoder().encode(JSON.stringify(payload));
      expect(bytes.byteLength).toBe(maximum);
      const url = `https://relay.example/v1/${classification ? 'classify' : 'respond'}`;
      const inclusive = new ReadableStream<Uint8Array>({ start(controller) {
        const split = bytes.indexOf(0xea) + 1;
        controller.enqueue(bytes.subarray(0, split));
        for (let offset = split; offset < bytes.length; offset += 997) controller.enqueue(bytes.subarray(offset, offset + 997));
        controller.close();
      } });
      const accepted = await relay.fetch(new Request(url, { method: 'POST', headers: authHeaders(), body: inclusive,
        duplex: 'half' } as RequestInit), env);
      expect(accepted.status).toBe(200);
      await accepted.arrayBuffer();
      for (const declared of [undefined, '1']) {
        const cancel = vi.fn();
        const oversized = new ReadableStream<Uint8Array>({ start(controller) {
          controller.enqueue(bytes);
          controller.enqueue(new Uint8Array([32]));
        }, cancel });
        const response = await relay.fetch(new Request(url, { method: 'POST', headers: {
          ...authHeaders(), ...(declared === undefined ? {} : { 'content-length': declared }),
        }, body: oversized, duplex: 'half' } as RequestInit), env);
        expect(response.status).toBe(413);
        expect(await response.json()).toMatchObject({ error: { code: 'request_too_large' } });
        expect(cancel).toHaveBeenCalledTimes(1);
      }
      const early = new Request(url, { method: 'POST', headers: { ...authHeaders(), 'content-length': String(maximum + 1) }, body: '{' });
      expect((await relay.fetch(early, env)).status).toBe(413);
      expect(early.bodyUsed).toBe(false);
      expect(database.admission).toHaveBeenCalledTimes(1);
      expect(upstream).toHaveBeenCalledTimes(1);
    });
});

describe('absolute provider lifetime', () => {
  it('stops an intentionally live response before lease expiry and releases its durable slot', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(start);
    let controller!: ReadableStreamDefaultController<Uint8Array>;
    const cancel = vi.fn();
    const body = new ReadableStream<Uint8Array>({ start(value) { controller = value; }, cancel });
    const upstream = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(body, { headers: { 'content-type': 'text/event-stream' } }));
    const { sqlite, db } = accountingDatabase();
    const env = { ...testEnv(), AGENT_USAGE_DB: db };
    try {
      const response = await relay.fetch(request(), env);
      controller.enqueue(new TextEncoder().encode('data: {"type":"response.output_text.delta","delta":"Provisional"}\n\n'));
      const events = relayEvents(response);
      const signal = upstream.mock.calls[0][1]?.signal as AbortSignal;
      const busy = await relay.fetch(request(), env);
      expect(busy.status).toBe(429);
      expect(await busy.json()).toMatchObject({ error: { code: 'service_busy', retry_after_ms: 1000 } });
      expect(busy.headers.get('retry-after')).toBe('1');
      await vi.advanceTimersByTimeAsync(requestDeadlineMs - 1);
      expect(signal.aborted).toBe(false);
      expect(sqlite.prepare('SELECT COUNT(*) AS count FROM request_slots').get()).toEqual({ count: 1 });
      await vi.advanceTimersByTimeAsync(1);
      expect(await events).toEqual([{ type: 'text_delta', text: 'Provisional' }, { type: 'error', error: { code: 'upstream_unavailable' } }]);
      expect(signal.aborted).toBe(true);
      expect(cancel).toHaveBeenCalledTimes(1);
      expect(Date.now()).toBeLessThan(start.getTime() + requestLeaseMs);
      expect(sqlite.prepare('SELECT * FROM request_slots').all()).toEqual([]);
      expect(sqlite.prepare('SELECT responses_calls, missing_usage_count FROM usage_daily').get())
        .toEqual({ responses_calls: 1, missing_usage_count: 1 });
      upstream.mockImplementation(async () => responseEvents(returned));
      expect(await relayEvents(await relay.fetch(request(), env))).toEqual([{ type: 'completed', response: returned }]);
    } finally {
      sqlite.close();
    }
  });

  it('cancels a classifier JSON reader at the same fixed deadline and settles once', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(start);
    const cancel = vi.fn();
    const body = new ReadableStream<Uint8Array>({ start(controller) {
      controller.enqueue(new TextEncoder().encode('{"answers":'));
    }, cancel });
    const reading = Promise.withResolvers<void>();
    const upstream = vi.spyOn(globalThis, 'fetch').mockImplementation(async () => {
      reading.resolve();
      return new Response(body, { headers: { 'content-type': 'application/json' } });
    });
    const { sqlite, db } = accountingDatabase();
    try {
      const pending = relay.fetch(request(true), { ...testEnv(), AGENT_USAGE_DB: db });
      await reading.promise;
      await vi.advanceTimersByTimeAsync(requestDeadlineMs);
      const response = await pending;
      expect(response.status).toBe(502);
      expect(cancel).toHaveBeenCalledTimes(1);
      expect((upstream.mock.calls[0][1]?.signal as AbortSignal).aborted).toBe(true);
      expect(sqlite.prepare('SELECT * FROM request_slots').all()).toEqual([]);
      expect(sqlite.prepare('SELECT decisions_calls, missing_usage_count FROM usage_daily').get())
        .toEqual({ decisions_calls: 1, missing_usage_count: 1 });
    } finally {
      sqlite.close();
    }
  });

  it('aborts a provider that never returns headers without holding the slot until lease expiry', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(start);
    const dispatched = Promise.withResolvers<void>();
    const finish = Promise.withResolvers<Response>();
    const upstream = vi.spyOn(globalThis, 'fetch').mockImplementation(() => {
      dispatched.resolve();
      return finish.promise;
    });
    const { sqlite, db } = accountingDatabase();
    try {
      const pending = relay.fetch(request(true), { ...testEnv(), AGENT_USAGE_DB: db });
      await dispatched.promise;
      await vi.advanceTimersByTimeAsync(requestDeadlineMs);
      expect((await pending).status).toBe(502);
      expect((upstream.mock.calls[0][1]?.signal as AbortSignal).aborted).toBe(true);
      expect(sqlite.prepare('SELECT * FROM request_slots').all()).toEqual([]);
      expect(sqlite.prepare('SELECT decisions_calls, missing_usage_count FROM usage_daily').get())
        .toEqual({ decisions_calls: 1, missing_usage_count: 1 });
      finish.resolve(Response.json(returned));
      await finish.promise;
    } finally {
      sqlite.close();
    }
  });
});
