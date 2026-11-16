import { afterEach, describe, expect, it, vi } from 'vitest';
import legacyContract from '../src/photo-contract.json';
import quadContract from '../src/photo-quad-contract.json';
import relay from '../src/index';
import { checkPhotoAllowance, photoAllowanceUsed, photoPolicy } from '../src/usage';
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
  return { AGENT_USAGE_DB: db, PURI_TOKEN_SECRET: tokenSecret, PHOTO_OPENAI_API_KEY: 'photo-only-key', PHOTO_SERVICE_ENABLED: 'true' };
}
function message(id = 'answer', phase: string | null = 'final_answer', text = '') {
  return { id, type: 'message', role: 'assistant', phase, content: [{ type: 'output_text', text }] };
}
function item(value = message(), index = 0, done = false) {
  return { type: done ? 'response.output_item.done' : 'response.output_item.added', output_index: index, item: value };
}
function delta(text: string, id = 'answer', index = 0) {
  return { type: 'response.output_text.delta', item_id: id, output_index: index, content_index: 0, delta: text };
}

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe('photo-only Relay admission', () => {
  it.each([undefined, 'chat-only-key'])('streams and settles photos with a separate key when the chatbot key is %s', async (chatKey) => {
    const { db, sqlite } = accountingDatabase();
    const upstream = vi.fn().mockResolvedValue(responseEvents(result, [
      { type: 'response.output_text.delta', delta: '{"blocks":[]}' },
    ]));
    vi.stubGlobal('fetch', upstream);
    const response = await relay.fetch(request(), { ...environment(db), OPENAI_API_KEY: chatKey });
    expect(response.status).toBe(200);
    const events = await relayEvents(response);
    expect(events.map((event) => event.type)).toEqual(['text_delta', 'completed']);
    expect(upstream).toHaveBeenCalledTimes(1);
    const [endpoint, init] = upstream.mock.calls[0];
    expect(endpoint).toBe('https://api.openai.com/v1/responses');
    expect(new Headers(init.headers).get('authorization')).toBe('Bearer photo-only-key');
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

  it.each([undefined, ''])('rejects photo requests and allowance checks without a photo key (%s), even with a chatbot key', async (photoKey) => {
    const { db, sqlite } = accountingDatabase();
    const upstream = vi.fn();
    vi.stubGlobal('fetch', upstream);
    const env = { ...environment(db), PHOTO_OPENAI_API_KEY: photoKey, OPENAI_API_KEY: 'chat-only-key' };
    try {
      for (const incoming of [request(), new Request('https://relay.test/v1/photo/allowance', {
        method: 'POST', headers: { authorization: `Bearer ${signToken()}` }, body: '{}',
      })]) {
        const response = await relay.fetch(incoming, env);
        expect(response.status).toBe(503);
        expect(await response.json()).toEqual({ error: { code: 'service_shutdown' } });
      }
      expect(upstream).not.toHaveBeenCalled();
      expect(sqlite.prepare('SELECT COUNT(*) AS n FROM photo_admissions').get()!.n).toBe(0);
    } finally {
      sqlite.close();
    }
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
  it.each([1, 2, 3])('preserves protocol %s geometry, JPEG upload and one settled request', async (version) => {
    const { db, sqlite } = accountingDatabase();
    const geometry = version === 3
      ? { quad: [[20.5, 10.25], [110.5, 30.25], [90.5, 100.25], [0.5, 80.25]] }
      : { bbox: [0, 0, 100, 100] };
    const block = { ...geometry, source: 'one\nunknown ?', translation: '하나\n?' };
    const text = JSON.stringify({ blocks: [block] });
    const upstream = vi.fn().mockResolvedValue(responseEvents(result, [
      { type: 'response.output_text.delta', delta: text },
    ]));
    vi.stubGlobal('fetch', upstream);
    try {
      const value = { ...body, photo_protocol: version, ...(version === 1 ? {} : { image_size: [720, 1280] }) };
      const response = await relay.fetch(request(value), environment(db));
      expect(response.status).toBe(200);
      const events = await relayEvents(response);
      expect(events.map((event) => event.type)).toEqual(['text_delta', 'completed']);
      expect(JSON.parse(events[0].text as string).blocks).toEqual([block]);
      expect(upstream).toHaveBeenCalledTimes(1);
      const sent = JSON.parse(upstream.mock.calls[0][1].body);
      const contract = version === 3 ? quadContract : legacyContract;
      expect(sent.text.format.schema).toEqual(contract.schema);
      expect(sent.text.format.schema.properties.blocks.items.required).toEqual([
        version === 3 ? 'quad' : 'bbox', 'source', 'translation',
      ]);
      expect(sent.text.format.schema.properties.blocks.items.properties).toEqual(version === 3 ? {
        quad: { type: 'array', items: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 }, minItems: 4, maxItems: 4 },
        source: { type: 'string' }, translation: { type: 'string' },
      } : {
        bbox: { type: 'array', items: { type: 'integer' }, minItems: 4, maxItems: 4 },
        source: { type: 'string' }, translation: { type: 'string' },
      });
      expect(sent.instructions).toBe(contract.instructions.replaceAll('${targetName}', 'Korean'));
      expect(sent.input[0].content[0]).toEqual({ type: 'input_image', image_url: `data:image/jpeg;base64,${body.image_base64}` });
      expect(sent.input[0].content[1].text).toContain(version === 1 ? '1280x720' : '720x1280');
      expect(sqlite.prepare('SELECT settled, output_tokens FROM photo_admissions').get()).toMatchObject({ settled: 1, output_tokens: 2556 });
    } finally {
      sqlite.close();
    }
  });

  it('rejects invalid v3 canvases, retired versions and non-JPEG input before admission', async () => {
    const { db, sqlite } = accountingDatabase();
    const upstream = vi.fn();
    vi.stubGlobal('fetch', upstream);
    try {
      const current = { ...body, photo_protocol: 3, image_size: [1280, 720] };
      for (const invalid of [
        { ...current, image_size: [1920, 1080] }, { ...current, image_size: undefined },
        { ...current, photo_protocol: true }, { ...current, photo_protocol: 4 },
        { ...current, image_base64: 'iVBORw0KGgo=' }, { ...current, image_base64: '/9j/AAA!' },
      ]) {
        expect((await relay.fetch(request(invalid), environment(db))).status).toBe(400);
      }
      expect(upstream).not.toHaveBeenCalled();
      expect(sqlite.prepare('SELECT COUNT(*) AS n FROM photo_admissions').get()!.n).toBe(0);
    } finally {
      sqlite.close();
    }
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

  it('preserves admission milliseconds across midnight settlement without resetting a custom allowance period', async () => {
    vi.useFakeTimers();
    const admittedAt = Date.UTC(2026, 10, 13, 23, 59, 59, 500);
    const settledAt = Date.UTC(2026, 10, 14, 0, 0, 0, 500);
    vi.setSystemTime(admittedAt);
    const { db, sqlite } = accountingDatabase();
    const policy = photoPolicy(JSON.stringify({ periodMs: 30 * 3_600_000, allowance: 1 }));
    const period = Math.floor(admittedAt / policy.periodMs) * policy.periodMs;
    const env = { ...environment(db), PHOTO_POLICY: JSON.stringify(policy) };
    const upstream = vi.fn(async () => {
      expect(sqlite.prepare('SELECT requested_at FROM photo_admissions').get()!.requested_at).toBe(admittedAt);
      vi.setSystemTime(settledAt);
      return responseEvents(result);
    });
    vi.stubGlobal('fetch', upstream);
    try {
      const response = await relay.fetch(request(), env);
      expect(response.status).toBe(200);
      expect((await relayEvents(response)).map((event) => event.type)).toEqual(['completed']);
      const row = sqlite.prepare('SELECT period_start, requested_at, settled, cost_usd FROM photo_admissions').get()!;
      expect(row).toMatchObject({ period_start: period, requested_at: admittedAt, settled: 1 });
      expect(row.cost_usd).toBeCloseTo(0.001465675, 10);
      expect(Math.floor(settledAt / policy.periodMs) * policy.periodMs).toBe(period);
      const resetAfter = Math.ceil((period + policy.periodMs - settledAt) / 1000);
      expect(await photoAllowanceUsed(env, installationHash, policy)).toEqual({ used: 1, resetAfter });
      expect(await checkPhotoAllowance(env, installationHash, 'next-request', policy)).toEqual({
        error: 'service_allowance_exhausted', retryAfter: resetAfter,
      });
      expect(upstream).toHaveBeenCalledTimes(1);
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
        controller.enqueue(new TextEncoder().encode(
          [item(), delta('{')].map((event) => `data: ${JSON.stringify(event)}\n\n`).join(''),
        ));
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

describe('photo structured-answer message selection', () => {
  it.each([
    [1, '{"blocks": invalid'], [1, '{"blocks":[]}'],
    [2, '{"blocks": invalid'], [2, '{"blocks":[]}'],
    [3, '{"blocks": invalid'], [3, '{"blocks":[]}'],
  ] as const)('streams only final blocks before terminal completion for protocol %s (%s)', async (version, commentary) => {
    const { db, sqlite } = accountingDatabase();
    let controller: ReadableStreamDefaultController<Uint8Array>;
    const upstream = new ReadableStream<Uint8Array>({ start(value) { controller = value; } });
    const emit = (event: unknown) => controller.enqueue(new TextEncoder().encode(`data: ${JSON.stringify(event)}\n\n`));
    const fetch = vi.fn().mockResolvedValue(new Response(upstream, { headers: { 'content-type': 'text/event-stream' } }));
    vi.stubGlobal('fetch', fetch);
    const block = {
      ...(version === 3 ? { quad: [[20.5, 10.25], [110.5, 30.25], [90.5, 100.25], [0.5, 80.25]] } : { bbox: [0, 0, 100, 100] }),
      source: 'FINAL\n?', translation: '최종\n?',
    };
    const first = '{"blocks":[' + JSON.stringify(block);
    const comment = message('comment', 'commentary', commentary);
    const final = message('answer', 'final_answer', first + ']}');
    const response = await relay.fetch(request({
      ...body, photo_protocol: version, ...(version === 1 ? {} : { image_size: [1280, 720] }),
    }), environment(db));
    const reader = response.body!.getReader();
    const read = async () => JSON.parse(new TextDecoder().decode((await reader.read()).value).slice(6));
    try {
      for (const event of [item(comment), delta(commentary, 'comment'), item(comment, 0, true), item(final, 1), delta(first, 'answer', 1)]) emit(event);
      expect(await read()).toEqual({ type: 'text_delta', text: first });
      expect(sqlite.prepare('SELECT settled FROM photo_admissions').get()!.settled).toBe(0);
      expect(fetch).toHaveBeenCalledTimes(1);
      emit(delta(']}', 'answer', 1));
      emit(item(final, 1, true));
      emit({ type: 'response.completed', response: { ...result, output: [comment, final] } });
      expect(await read()).toEqual({ type: 'text_delta', text: ']}' });
      expect((await read()).type).toBe('completed');
      expect((await reader.read()).done).toBe(true);
      expect(sqlite.prepare('SELECT settled, output_tokens FROM photo_admissions').get()).toMatchObject({ settled: 1, output_tokens: 2556 });
      expect(fetch).toHaveBeenCalledTimes(1);
    } finally {
      await reader.cancel();
      sqlite.close();
    }
  });

  it.each([null, 'final_answer'])('accepts a single initially unphased answer finalized as %s', async (terminalPhase) => {
    const { db, sqlite } = accountingDatabase();
    const text = '{"blocks":[]}';
    const unphased = message('answer', null, text);
    const fetch = vi.fn().mockResolvedValue(responseEvents({ ...result, output: [{ ...unphased, phase: terminalPhase }] }, [
      item(unphased), delta(text),
    ]));
    vi.stubGlobal('fetch', fetch);
    try {
      const events = await relayEvents(await relay.fetch(request({ ...body, photo_protocol: 3, image_size: [1280, 720] }), environment(db)));
      expect(events.map((event) => event.type)).toEqual(['text_delta', 'completed']);
      expect(events[0].text).toBe(text);
      expect(fetch).toHaveBeenCalledTimes(1);
      expect(sqlite.prepare('SELECT settled FROM photo_admissions').get()!.settled).toBe(1);
    } finally {
      sqlite.close();
    }
  });

  const comment = message('comment', 'commentary', '{"blocks":[]}');
  const unphased = message('answer', null, '{"blocks":[]}');
  const other = message('other', null, '{"blocks":[]}');
  it.each([
    { name: 'commentary-only', events: [item(comment), delta('{"blocks":[]}', 'comment')], output: [comment] },
    { name: 'two unphased answers', events: [item(unphased), delta('{"blocks":[]}'), item(other, 1), delta('{"blocks":[]}', 'other', 1)], output: [unphased, other] },
    { name: 'wrong item identity', events: [item(comment), delta('{"blocks":[]}', 'wrong')], output: [] },
    { name: 'missing item identity', events: [item(comment), { type: 'response.output_text.delta', delta: '{"blocks":[]}' }], output: [] },
    { name: 'two explicit final answers', events: [item(), item(message('other'), 1)], output: [] },
    { name: 'delta before item metadata', events: [delta('{"blocks":[]}'), item()], output: [] },
    { name: 'wrong content index', events: [item(unphased), { ...delta('{"blocks":[]}'), content_index: 1 }], output: [] },
    { name: 'changed phase', events: [item(comment), item({ ...comment, phase: 'final_answer' }, 0, true)], output: [] },
    { name: 'anonymous ambiguous output', events: [{ type: 'response.output_text.delta', delta: '{"blocks":[]}{"blocks":[]}' }], output: [comment, message()] },
    { name: 'commentary plus unphased message', events: [item(comment), item(unphased, 1), delta('{"blocks":[]}', 'answer', 1)], output: [comment, unphased] },
  ])('rejects $name without publishing structured text or replaying', async ({ events, output }) => {
    const { db, sqlite } = accountingDatabase();
    const fetch = vi.fn().mockResolvedValue(responseEvents({ ...result, output }, events));
    vi.stubGlobal('fetch', fetch);
    try {
      expect(await relayEvents(await relay.fetch(request({ ...body, photo_protocol: 3, image_size: [1280, 720] }), environment(db)))).toEqual([
        { type: 'error', error: { code: 'upstream_invalid_response' } },
      ]);
      expect(fetch).toHaveBeenCalledTimes(1);
      expect(sqlite.prepare('SELECT COUNT(*) AS n FROM photo_admissions WHERE settled = 1').get()!.n).toBe(1);
    } finally {
      sqlite.close();
    }
  });
});
