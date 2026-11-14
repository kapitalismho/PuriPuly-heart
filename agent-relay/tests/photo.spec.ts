import { afterEach, describe, expect, it, vi } from 'vitest';
import relay from '../src/index';
import { checkPhotoAllowance, photoPolicy } from '../src/usage';
import { accountingDatabase, installationHash, relayEvents, responseEvents, signToken, tokenSecret } from './fixtures';

const body = {
  photo_protocol: 1, request_id: '1'.repeat(32), image_base64: '/9j/AAAA', target_language: 'ko',
};
const result = { object: 'response', status: 'completed', model: 'gpt-6-luna', output: [],
  usage: { input_tokens: 1502, input_tokens_details: { cached_tokens: 0, cache_write_tokens: 1499 }, output_tokens: 2556 } };
function request(value: unknown = body, claims = {}) {
  return new Request('https://relay.test/v1/photo', {
    method: 'POST', headers: { authorization: `Bearer ${signToken(claims)}` }, body: JSON.stringify(value),
  });
}
function environment(db: D1Database) {
  return { AGENT_USAGE_DB: db, PURI_TOKEN_SECRET: tokenSecret, OPENAI_API_KEY: 'server-only-key', PHOTO_SERVICE_ENABLED: 'true' };
}
afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe('photo-only Relay admission', () => {
  it('streams useful output with server-selected Luna and durable photo-only settlement', async () => {
    const { db, sqlite } = accountingDatabase();
    const upstream = vi.fn().mockResolvedValue(responseEvents(result, [
      { type: 'response.output_text.delta', delta: '{"blocks":[]}' },
    ]));
    vi.stubGlobal('fetch', upstream);
    const response = await relay.fetch(request(), environment(db));
    expect(response.status).toBe(200);
    const events = await relayEvents(response);
    expect(events.map((event) => event.type)).toEqual(['text_delta', 'completed']);
    expect(upstream).toHaveBeenCalledTimes(1);
    const [endpoint, init] = upstream.mock.calls[0];
    expect(endpoint).toBe('https://api.openai.com/v1/responses');
    const sent = JSON.parse(init.body);
    expect(sent.model).toBe('gpt-6-luna');
    expect(sent.safety_identifier).toBe(installationHash);
    expect(sent.tools).toBeUndefined();
    expect(sent.input).toHaveLength(1);
    expect(sent.text.format.strict).toBe(true);
    const row = sqlite.prepare('SELECT * FROM photo_admissions').get()!;
    expect(row.settled).toBe(1);
    expect(row.output_tokens).toBe(2556);
    expect(row.missing_usage).toBe(0);
    expect(row.cost_usd).toBeCloseTo(0.001465675, 10);
    expect(sqlite.prepare('SELECT COUNT(*) AS n FROM prompt_turns').get()!.n).toBe(0);
    expect(sqlite.prepare('SELECT COUNT(*) AS n FROM usage_daily').get()!.n).toBe(0);
    sqlite.close();
  });

  it('states the client image size to the model and accepts only the two upright send canvases', async () => {
    const { db, sqlite } = accountingDatabase();
    const upstream = vi.fn().mockImplementation(async () => responseEvents(result, [
      { type: 'response.output_text.delta', delta: '{"blocks":[]}' },
    ]));
    vi.stubGlobal('fetch', upstream);
    const sized = (size: unknown, id: string) => ({ ...body, photo_protocol: 2, request_id: id.repeat(32), image_size: size });
    for (const [size, id, text] of [[[720, 1280], '2', '720x1280'], [[1280, 720], '3', '1280x720']] as const) {
      const response = await relay.fetch(request(sized(size, id)), environment(db));
      expect(response.status).toBe(200);
      await relayEvents(response);
      const sent = JSON.parse(upstream.mock.calls.at(-1)![1].body);
      expect(sent.input[0].content[1].text).toBe(`Target language: Korean.
The image is ${text} pixels.`);
    }
    const legacy = await relay.fetch(request(), environment(db));
    expect(legacy.status).toBe(200);
    await relayEvents(legacy);
    expect(JSON.parse(upstream.mock.calls.at(-1)![1].body).input[0].content[1].text).toContain('1280x720');
    const calls = upstream.mock.calls.length;
    for (const invalid of [sized([1920, 1080], '4'), sized('720x1280', '5'), { ...body, image_size: [720, 1280] }, sized(undefined, '6')]) {
      expect((await relay.fetch(request(invalid), environment(db))).status).toBe(400);
    }
    expect(upstream.mock.calls.length).toBe(calls);
    sqlite.close();
  });

  it('rejects model, endpoint, nested config and invalid tokens before admission or upstream work', async () => {
    const { db, sqlite } = accountingDatabase();
    const upstream = vi.fn(); vi.stubGlobal('fetch', upstream);
    for (const extra of [{ model: 'another' }, { endpoint: 'https://attacker.test' }, { config: { model: 'another' } }, { fallback: true }]) {
      expect((await relay.fetch(request({ ...body, ...extra }), environment(db))).status).toBe(400);
    }
    expect((await relay.fetch(request(body, { exp: 1 }), environment(db))).status).toBe(401);
    expect(upstream).not.toHaveBeenCalled();
    expect(sqlite.prepare('SELECT COUNT(*) AS n FROM photo_admissions').get()!.n).toBe(0);
    sqlite.close();
  });

  it('admits 25 photos per installation UTC day and resets only at the next midnight', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(Date.UTC(2026, 10, 13, 23, 59));
    const { db, sqlite } = accountingDatabase();
    const env = environment(db);
    const upstream = vi.fn().mockImplementation(() => Promise.resolve(responseEvents(result)));
    vi.stubGlobal('fetch', upstream);
    try {
      for (let index = 1; index <= 25; index++) {
        const response = await relay.fetch(request({ ...body, request_id: index.toString(16).padStart(32, '0') }), env);
        expect(response.status).toBe(200);
        expect((await relayEvents(response)).map((event) => event.type)).toEqual(['completed']);
      }
      expect(upstream).toHaveBeenCalledTimes(25);
      vi.setSystemTime(Date.UTC(2026, 10, 13, 23, 59, 59, 999));
      const nextPhoto = { ...body, request_id: (26).toString(16).padStart(32, '0') };
      const denied = await relay.fetch(request(nextPhoto), env);
      expect(denied.status).toBe(429);
      expect(await denied.json()).toEqual({ error: { code: 'service_allowance_exhausted' } });
      expect(denied.headers.get('retry-after')).toBe('1');
      expect(upstream).toHaveBeenCalledTimes(25);
      vi.setSystemTime(Date.UTC(2026, 10, 14));
      const renewed = await relay.fetch(request(nextPhoto), env);
      expect(renewed.status).toBe(200);
      expect((await relayEvents(renewed)).map((event) => event.type)).toEqual(['completed']);
      expect((await relay.fetch(request({ ...body, request_id: '1'.padStart(32, '0') }), env)).status).toBe(409);
      expect(upstream).toHaveBeenCalledTimes(26);
      expect(sqlite.prepare('SELECT COUNT(*) AS n FROM photo_admissions WHERE settled = 1').get()!.n).toBe(26);
      expect(sqlite.prepare('SELECT COUNT(*) AS n FROM prompt_turns').get()!.n).toBe(0);
      expect(sqlite.prepare('SELECT COUNT(*) AS n FROM usage_daily').get()!.n).toBe(0);
    } finally {
      sqlite.close();
    }
  });

  it('reports the remaining daily photo allowance without admitting a photo', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(Date.UTC(2026, 10, 13, 23, 0));
    const { db, sqlite } = accountingDatabase();
    const env = environment(db);
    vi.stubGlobal('fetch', vi.fn().mockImplementation(() => Promise.resolve(responseEvents(result))));
    const allowance = (claims = {}) => relay.fetch(new Request('https://relay.test/v1/photo/allowance', {
      method: 'POST', headers: { authorization: `Bearer ${signToken(claims)}` }, body: '{}',
    }), env);
    try {
      const fresh = await allowance();
      expect(fresh.status).toBe(200);
      expect(await fresh.json()).toEqual({ allowance: 25, remaining: 25, reset_after_s: 3600 });
      for (const id of ['a', 'b']) {
        await relayEvents(await relay.fetch(request({ ...body, request_id: id.repeat(32) }), env));
      }
      expect(await (await allowance()).json()).toEqual({ allowance: 25, remaining: 23, reset_after_s: 3600 });
      expect(sqlite.prepare('SELECT COUNT(*) AS n FROM photo_admissions').get()!.n).toBe(2);
      vi.setSystemTime(Date.UTC(2026, 10, 14));
      expect(await (await allowance()).json()).toEqual({ allowance: 25, remaining: 25, reset_after_s: 86400 });
      expect((await relay.fetch(new Request('https://relay.test/v1/photo/allowance', { method: 'POST', body: '{}' }), env)).status).toBe(401);
      expect((await relay.fetch(new Request('https://relay.test/v1/photo/allowance', {
        method: 'POST', headers: { authorization: `Bearer ${signToken()}` }, body: '{}',
      }), { ...env, PHOTO_SERVICE_ENABLED: 'false' })).status).toBe(503);
    } finally {
      sqlite.close();
    }
  });

  it('shares allowance across tokens and sessions, rejects duplicate generation and isolates installations', async () => {
    const { db, sqlite } = accountingDatabase();
    const env = { ...environment(db), PHOTO_POLICY: JSON.stringify({ allowance: 1 }) };
    const upstream = vi.fn().mockImplementation(() => Promise.resolve(responseEvents(result)));
    vi.stubGlobal('fetch', upstream);
    await relayEvents(await relay.fetch(request(), env));
    expect((await relay.fetch(request(), env)).status).toBe(409);
    const now = Math.floor(Date.now() / 1000);
    const denial = await relay.fetch(request({ ...body, request_id: '2'.repeat(32) }, { iat: now - 1, exp: now + 3599 }), env);
    expect(await denial.json()).toMatchObject({ error: { code: 'service_allowance_exhausted' } });
    expect(denial.status).toBe(429);
    await relayEvents(await relay.fetch(request(body, { sub: 'a'.repeat(64) }), env));
    expect(upstream).toHaveBeenCalledTimes(2);
    sqlite.close();
  });

  it('atomically enforces photo concurrency and retains uncertain admissions after lease expiry', async () => {
    const { db, sqlite } = accountingDatabase();
    const policy = photoPolicy(JSON.stringify({ allowance: 2 }));
    const now = Date.UTC(2026, 10, 10, 12);
    const env = environment(db);
    const [a, b] = await Promise.all([
      checkPhotoAllowance(env, installationHash, 'a', policy, now),
      checkPhotoAllowance(env, installationHash, 'b', policy, now),
    ]);
    expect('error' in a).toBe(false);
    expect(b).toEqual({ error: 'service_busy' });
    const later = now + policy.deadlineMs + 60_001;
    expect(await checkPhotoAllowance(env, installationHash, 'a', policy, later)).toEqual({ error: 'photo_request_already_admitted' });
    const recovered = await checkPhotoAllowance(env, installationHash, 'b', policy, later);
    expect('error' in recovered).toBe(false);
    if ('error' in recovered) throw new Error('expected admission');
    await recovered.settle(undefined);
    await recovered.settle(result);
    expect(sqlite.prepare('SELECT missing_usage FROM photo_admissions WHERE request_id = ?').get('b')!.missing_usage).toBe(1);
    expect(await checkPhotoAllowance(env, installationHash, 'c', policy, later)).toEqual({ error: 'service_allowance_exhausted', retryAfter: 42900 });
    expect(sqlite.prepare('SELECT COUNT(*) AS n FROM request_slots').get()!.n).toBe(0);
    sqlite.close();
  });

  it('does not publish completion if durable settlement fails', async () => {
    const { db, sqlite } = accountingDatabase();
    const prepare = db.prepare.bind(db);
    vi.spyOn(db, 'prepare').mockImplementation((sql) => {
      if (sql.includes('UPDATE photo_admissions SET settled')) throw new Error('synthetic settlement outage');
      return prepare(sql);
    });
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(responseEvents(result)));
    const events = await relayEvents(await relay.fetch(request(), environment(db)));
    expect(events.map((event) => event.type)).toEqual(['error']);
    expect(sqlite.prepare('SELECT settled FROM photo_admissions').get()!.settled).toBe(0);
    sqlite.close();
  });

  it('allows measured image-sized bodies independently of chatbot limits and rejects oversized input before spending', async () => {
    const { db, sqlite } = accountingDatabase();
    vi.stubGlobal('fetch', vi.fn().mockImplementation(() => Promise.resolve(responseEvents(result))));
    const image = '/9j/' + 'A'.repeat(1100 * 1024);
    const accepted = await relay.fetch(request({ ...body, image_base64: image }), environment(db));
    expect(accepted.status).toBe(200);
    await relayEvents(accepted);
    const denied = await relay.fetch(request({ ...body, request_id: '3'.repeat(32), image_base64: '/9j/' + 'A'.repeat(4 * 1024 * 1024) }), environment(db));
    expect(denied.status).toBe(413);
    expect(sqlite.prepare('SELECT COUNT(*) AS n FROM photo_admissions').get()!.n).toBe(1);
    sqlite.close();
  });
  it('cancels paid work, settles missing usage and releases only the concurrency lease', async () => {
    const { db, sqlite } = accountingDatabase();
    let cancelled = false;
    let signal: AbortSignal | undefined;
    const upstream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode('data: {"type":"response.output_text.delta","delta":"{"}\n\n'));
      },
      cancel() { cancelled = true; },
    });
    vi.stubGlobal('fetch', vi.fn(async (_url, init) => {
      signal = init.signal;
      return new Response(upstream, { headers: { 'content-type': 'text/event-stream' } });
    }));
    const response = await relay.fetch(request(), environment(db));
    const reader = response.body!.getReader();
    await reader.read();
    await reader.cancel();
    expect(cancelled).toBe(true);
    expect(signal?.aborted).toBe(true);
    expect(sqlite.prepare('SELECT settled, missing_usage FROM photo_admissions').get()).toMatchObject({
      settled: 1, missing_usage: 1,
    });
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(responseEvents(result)));
    const next = await relay.fetch(request({ ...body, request_id: '2'.repeat(32) }), environment(db));
    expect((await relayEvents(next)).at(-1)?.type).toBe('completed');
    expect(sqlite.prepare('SELECT COUNT(*) AS n FROM photo_admissions').get()!.n).toBe(2);
    sqlite.close();
  });

  it('rejects redirected provider responses and retains their admitted allowance', async () => {
    const { db, sqlite } = accountingDatabase();
    const upstream = vi.fn().mockResolvedValue(new Response(null, {
      status: 307, headers: { location: 'https://untrusted.test/collect' },
    }));
    vi.stubGlobal('fetch', upstream);
    const response = await relay.fetch(request(), environment(db));
    expect(response.status).toBe(502);
    expect(await response.json()).toEqual({ error: { code: 'photo_provider_unavailable' } });
    expect(upstream).toHaveBeenCalledTimes(1);
    const row = sqlite.prepare('SELECT settled, missing_usage FROM photo_admissions').get()!;
    expect(row).toEqual({ settled: 1, missing_usage: 1 });
    expect((await relay.fetch(request(), environment(db))).status).toBe(409);
    expect(upstream).toHaveBeenCalledTimes(1);
    sqlite.close();
  });
});
