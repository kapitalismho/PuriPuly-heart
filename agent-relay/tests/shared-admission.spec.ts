import { afterEach, describe, expect, it, vi } from 'vitest';

import { currentProtocol, supportedProtocols } from '../src/contracts';
import relay from '../src/index';
import { accountingDatabase, installationHash, relayEvents, responseEvents, signToken, testEnv, turnId, usageDatabase } from './fixtures';

const usage = { input_tokens: 10, output_tokens: 2 };
const completed = { object: 'response', status: 'completed', output: [], usage };
const user = { role: 'user', content: 'Synthetic request' };
const highEffort = { type: 'configuration_update', reasoning: { effort: 'high' } };
const otherInstallation = 'c'.repeat(64);

function classifyBody(protocol: number, turn_id = turnId) {
  return { agent_protocol: protocol, turn_id, input: [user] };
}

function respondBody(protocol: number, turn_id = turnId, route = 'work') {
  return route === 'work' || route === 'unclear'
    ? { agent_protocol: protocol, turn_id, input: [highEffort, user], route, effort: 'high', allow_tools: true }
    : { agent_protocol: protocol, turn_id, input: [user], route, effort: 'low', allow_tools: false };
}

function post(path: string, body: unknown, token = signToken()): Request {
  return new Request(`https://relay.example${path}`, {
    method: 'POST', headers: { authorization: `Bearer ${token}`, 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
}

async function settled(request: Request, env: Parameters<typeof relay.fetch>[1]): Promise<Response> {
  const response = await relay.fetch(request, env);
  await response.clone().arrayBuffer();
  return response;
}

function mockLuna() {
  return vi.spyOn(globalThis, 'fetch').mockImplementation(async (url) =>
    String(url).endsWith('/decisions') ? Response.json({ answers: [], usage }) : responseEvents(completed));
}

function sentModels(upstream: ReturnType<typeof mockLuna>) {
  return upstream.mock.calls.map(([url, init]) => [String(url), JSON.parse(init?.body as string).model]);
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.useRealTimers();
});

describe('shared installation admission', () => {
  it.each(supportedProtocols)('keeps protocol %i classification on Luna and selects Terra for response routes', async (protocol) => {
    const upstream = mockLuna();
    const env = testEnv();
    expect((await relay.fetch(post('/v1/classify', classifyBody(protocol)), env)).status).toBe(200);
    for (const route of ['work', 'unclear', 'social', 'out_of_scope']) {
      expect((await settled(post('/v1/respond', respondBody(protocol, turnId, route)), env)).status).toBe(200);
    }
    expect(sentModels(upstream)).toEqual([
      ['https://api.openai.com/v1/decisions', 'gpt-6-luna'],
      ...Array(4).fill(['https://api.openai.com/v1/responses', 'gpt-5.6-terra']),
    ]);
  });

  it.each(supportedProtocols)('rejects protocol %i model, endpoint and retired-protocol overrides before admission or upstream', async (protocol) => {
    const upstream = mockLuna();
    const database = usageDatabase();
    const env = { ...testEnv(), AGENT_USAGE_DB: database.db };
    const attempts = [
      ['/v1/classify', { ...classifyBody(protocol), agent_protocol: 5 }],
      ['/v1/respond', { ...respondBody(protocol), agent_protocol: 5 }],
      ['/v1/classify', { ...classifyBody(protocol), model: 'gpt-6' }],
      ['/v1/classify', { ...classifyBody(protocol), endpoint: 'https://example.invalid/v1/decisions' }],
      ['/v1/classify', { ...classifyBody(protocol), input: [{ ...user, model: 'gpt-6' }] }],
      ['/v1/respond', { ...respondBody(protocol), model: 'gpt-6' }],
      ['/v1/respond', { ...respondBody(protocol, turnId, 'social'), model: 'gpt-6' }],
      ['/v1/respond', { ...respondBody(protocol), input: [{ ...highEffort, model: 'gpt-6' }, user] }],
      ['/v1/respond', { ...respondBody(protocol), input: [{ ...highEffort, reasoning: { effort: 'high', model: 'gpt-6' } }, user] }],
      ['/v1/respond', { ...respondBody(protocol), input: [highEffort, { ...user, model: 'gpt-6' }] }],
    ] as const;
    for (const [path, body] of attempts) {
      const response = await relay.fetch(post(path, body), env);
      expect(response.status, JSON.stringify(body)).toBe(400);
    }
    expect(database.admission).not.toHaveBeenCalled();
    expect(upstream).not.toHaveBeenCalled();
  });

  it.each(supportedProtocols)('reports protocol %i capability failures without retrying an alternate model', async (protocol) => {
    const upstream = vi.spyOn(globalThis, 'fetch').mockImplementation(async () =>
      Response.json({ error: { code: 'model_not_found' } }, { status: 404 }));
    const env = testEnv();
    for (const [path, body] of [
      ['/v1/classify', classifyBody(protocol)],
      ['/v1/respond', respondBody(protocol)],
    ] as const) {
      const response = await relay.fetch(post(path, body), env);
      expect(response.status).toBe(502);
      expect(await response.json()).toMatchObject({ error: { code: 'upstream_capability_unavailable' } });
    }
    expect(sentModels(upstream as ReturnType<typeof mockLuna>)).toEqual([
      ['https://api.openai.com/v1/decisions', 'gpt-6-luna'],
      ['https://api.openai.com/v1/responses', 'gpt-5.6-terra'],
    ]);
  });

  it('keeps daily and turn allowances on the verified subject across routes and renewed tokens', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-11-01T12:00:00Z'));
    const upstream = mockLuna();
    const { sqlite, db } = accountingDatabase();
    const env = { ...testEnv(), AGENT_USAGE_DB: db };
    const seed = sqlite.prepare('INSERT INTO prompt_turns (installation_hash, turn_id, utc_date) VALUES (?, ?, ?)');
    const turns = sqlite.prepare('SELECT COUNT(*) AS count FROM prompt_turns WHERE installation_hash = ? AND utc_date = ?');
    const protocol = currentProtocol;
    try {
      for (let index = 0; index < 28; index++) seed.run(installationHash, index.toString(16).padStart(32, '0'), '2026-11-01');
      expect((await relay.fetch(post('/v1/classify', classifyBody(protocol)), env)).status).toBe(200);
      vi.setSystemTime(new Date('2026-11-01T12:30:00Z'));
      const renewed = signToken();
      expect((await settled(post('/v1/respond', respondBody(protocol), renewed), env)).status).toBe(200);
      expect(turns.get(installationHash, '2026-11-01')).toEqual({ count: 29 });
      expect((await settled(post('/v1/respond', respondBody(protocol, 'e'.repeat(32), 'social'), renewed), env)).status).toBe(200);
      for (const [path, body] of [
        ['/v1/classify', classifyBody(protocol, 'f'.repeat(32))],
        ['/v1/respond', respondBody(protocol, 'f'.repeat(32))],
        ['/v1/respond', respondBody(protocol, 'f'.repeat(32), 'out_of_scope')],
      ] as const) {
        const response = await relay.fetch(post(path, body, renewed), env);
        expect(response.status).toBe(429);
        expect(await response.json()).toMatchObject({ error: { code: 'service_allowance_exhausted' } });
      }
      const routes = ['work', 'unclear', 'social'];
      for (let admission = 3; admission <= 15; admission++) {
        const route = routes[admission % routes.length];
        expect((await settled(post('/v1/respond', respondBody(protocol, turnId, route), renewed), env)).status).toBe(200);
      }
      for (const [path, body] of [
        ['/v1/respond', respondBody(protocol, turnId, 'out_of_scope')], ['/v1/classify', classifyBody(protocol)],
      ] as const) {
        const response = await relay.fetch(post(path, body, signToken()), env);
        expect(response.status).toBe(429);
        expect(await response.json()).toMatchObject({ error: { code: 'service_turn_exhausted' } });
      }
      expect((await settled(post('/v1/respond', respondBody(protocol, 'f'.repeat(32)),
        signToken({ sub: otherInstallation })), env)).status).toBe(200);
      expect(turns.get(installationHash, '2026-11-01')).toEqual({ count: 30 });
      expect(sqlite.prepare('SELECT model_admissions, classifier_admitted FROM prompt_turns WHERE installation_hash = ? AND turn_id = ?')
        .get(installationHash, turnId)).toEqual({ model_admissions: 15, classifier_admitted: 1 });
      expect(sqlite.prepare('SELECT agent_protocol, requests FROM protocol_daily WHERE installation_hash = ?')
        .all(installationHash)).toEqual([{ agent_protocol: protocol, requests: 16 }]);
      expect(upstream).toHaveBeenCalledTimes(17);
      expect(new Set(sentModels(upstream).map(([, model]) => model))).toEqual(new Set(['gpt-6-luna', 'gpt-5.6-terra']));
    } finally {
      sqlite.close();
    }
  });

  it('caps concurrent model requests per installation across routes and renewed tokens without blocking another installation', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-11-01T12:00:00Z'));
    const streams: ReadableStreamDefaultController<Uint8Array>[] = [];
    const upstream = vi.spyOn(globalThis, 'fetch').mockImplementation(async (url) => String(url).endsWith('/decisions')
      ? Response.json({ answers: [], usage })
      : new Response(new ReadableStream<Uint8Array>({ start(controller) { streams.push(controller); } }),
        { headers: { 'content-type': 'text/event-stream' } }));
    const { sqlite, db } = accountingDatabase();
    const env = { ...testEnv(), AGENT_USAGE_DB: db };
    const protocol = currentProtocol;
    try {
      const held = [
        await relay.fetch(post('/v1/respond', respondBody(protocol, 'a'.repeat(32))), env),
        await relay.fetch(post('/v1/respond', respondBody(protocol, 'b'.repeat(32), 'social')), env),
      ];
      vi.setSystemTime(new Date('2026-11-01T12:01:00Z'));
      const renewed = signToken();
      for (const [path, body] of [
        ['/v1/classify', classifyBody(protocol, turnId)], ['/v1/respond', respondBody(protocol, turnId, 'unclear')],
      ] as const) {
        const busy = await relay.fetch(post(path, body, renewed), env);
        expect(busy.status).toBe(429);
        expect(await busy.json()).toMatchObject({ error: { code: 'service_busy', retry_after_ms: 1000 } });
      }
      expect((await relay.fetch(post('/v1/classify', classifyBody(protocol), signToken({ sub: otherInstallation })), env)).status)
        .toBe(200);
      const encoder = new TextEncoder();
      for (const stream of streams) {
        stream.enqueue(encoder.encode(`data: ${JSON.stringify({ type: 'response.completed', response: completed })}\n\n`));
        stream.close();
      }
      for (const response of held) expect(await relayEvents(response)).toEqual([{ type: 'completed', response: completed }]);
      expect((await relay.fetch(post('/v1/classify', classifyBody(protocol), renewed), env)).status).toBe(200);
      expect(sqlite.prepare('SELECT model_admissions FROM prompt_turns WHERE installation_hash = ? AND turn_id = ?')
        .get(installationHash, turnId)).toEqual({ model_admissions: 1 });
      expect(upstream).toHaveBeenCalledTimes(4);
    } finally {
      sqlite.close();
    }
  });
});
