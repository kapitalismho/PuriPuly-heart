import { afterEach, describe, expect, it, vi } from 'vitest';

import relay from '../src/index';
import { authHeaders, relayEvents, responseEvents, testEnv, turnId, usageDatabase } from './fixtures';

function request(signal?: AbortSignal): Request {
  return new Request('https://relay.example/v1/respond', {
    method: 'POST', headers: authHeaders(), signal,
    body: JSON.stringify({ agent_protocol: 5, turn_id: turnId, input: [
      { type: 'configuration_update', reasoning: { effort: 'high' } }, { role: 'user', content: 'Synthetic request' },
    ], effort: 'high', route: 'work', allow_tools: true }),
  });
}

function terminal() {
  return { object: 'response', status: 'completed', output: [
    { type: 'reasoning', encrypted_content: 'opaque-final-only', summary: [] },
    { type: 'message', role: 'assistant', content: [{ type: 'output_text', text: 'Ready. Final fragment' }] },
  ], usage: { input_tokens: 100, output_tokens: 20 } };
}

function held() {
  let controller: ReadableStreamDefaultController<Uint8Array>;
  const cancel = vi.fn();
  const body = new ReadableStream<Uint8Array>({ start(value) { controller = value; }, cancel });
  return { response: new Response(body, { headers: { 'content-type': 'text/event-stream' } }), cancel,
    emit(event: Record<string, unknown>) { controller.enqueue(new TextEncoder().encode(`data: ${JSON.stringify(event)}\n\n`)); },
    close() { controller.close(); },
  };
}

afterEach(() => vi.restoreAllMocks());

describe('upstream response streaming', () => {
  it('forwards visible text while upstream stays open, hiding reasoning and tool argument deltas', async () => {
    const upstream = held();
    const fetch = vi.spyOn(globalThis, 'fetch').mockResolvedValue(upstream.response);
    const database = usageDatabase();
    const response = await relay.fetch(request(), { ...testEnv(), AGENT_USAGE_DB: database.db });
    const reader = response.body!.getReader();
    upstream.emit({ type: 'response.reasoning_text.delta', delta: 'never-visible' });
    upstream.emit({ type: 'response.function_call_arguments.delta', delta: '{"unsafe":true}' });
    upstream.emit({ type: 'response.output_text.delta', delta: 'Ready. ' });
    const first = await reader.read();
    expect(new TextDecoder().decode(first.value)).toBe('data: {"type":"text_delta","text":"Ready. "}\n\n');
    expect(database.writes).toHaveLength(0);
    expect(upstream.cancel).not.toHaveBeenCalled();
    const result = terminal();
    upstream.emit({ type: 'response.completed', response: result });
    const completed = await reader.read();
    expect(JSON.parse(new TextDecoder().decode(completed.value).slice(6))).toEqual({ type: 'completed', response: result });
    expect((await reader.read()).done).toBe(true);
    expect(database.writes).toHaveLength(1);
    expect(database.releases).toHaveLength(1);
    expect(database.writes[0].slice(2, 10)).toEqual([100, 0, 0, 0, 20, 0, 1, 0]);
    expect(upstream.cancel).toHaveBeenCalledTimes(1);
    expect((fetch.mock.calls[0][1]?.signal as AbortSignal).aborted).toBe(true);
  });

  it.each(['client', 'request'])('cancels %s upstream and durably schedules exactly one missing-usage record', async (source) => {
    const upstream = held();
    const fetch = vi.spyOn(globalThis, 'fetch').mockResolvedValue(upstream.response);
    const database = usageDatabase();
    const durable: Promise<unknown>[] = [];
    const abort = new AbortController();
    const response = await relay.fetch(request(abort.signal), { ...testEnv(), AGENT_USAGE_DB: database.db }, {
      waitUntil(promise) { durable.push(promise); },
    });
    const reader = response.body!.getReader();
    upstream.emit({ type: 'response.output_text.delta', delta: 'Ready.' });
    await reader.read();
    if (source === 'client') await reader.cancel();
    else abort.abort();
    await Promise.all(durable);
    expect(upstream.cancel).toHaveBeenCalledTimes(1);
    expect((fetch.mock.calls[0][1]?.signal as AbortSignal).aborted).toBe(true);
    expect(database.writes).toHaveLength(1);
    expect(database.releases).toHaveLength(1);
    expect(database.writes[0].slice(2)).toEqual([0, 0, 0, 0, 0, 0, 1, 1, 0]);
    expect(JSON.stringify(database.writes)).not.toContain('Ready');
    expect(durable.length).toBeGreaterThanOrEqual(2);
  });

  it('uses safe stream errors for interrupted or failed upstream without executing partial output', async () => {
    const upstream = held();
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(upstream.response);
    const database = usageDatabase();
    const response = await relay.fetch(request(), { ...testEnv(), AGENT_USAGE_DB: database.db });
    upstream.emit({ type: 'response.output_text.delta', delta: 'Provisional.' });
    upstream.emit({ type: 'error', message: 'PRIVATE provider body' });
    expect(await relayEvents(response)).toEqual([
      { type: 'text_delta', text: 'Provisional.' },
      { type: 'error', error: { code: 'upstream_unavailable' } },
    ]);
    expect(database.writes).toHaveLength(1);
    expect(database.releases).toHaveLength(1);
    expect(database.writes[0][9]).toBe(1);
  });

  it('fails terminal delivery explicitly on accounting failure without retrying or duplicating usage', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(async () => responseEvents(terminal()));
    const database = usageDatabase();
    database.run.mockRejectedValue(new Error('PRIVATE database details'));
    const response = await relay.fetch(request(), { ...testEnv(), AGENT_USAGE_DB: database.db });
    expect(await relayEvents(response)).toEqual([{ type: 'error', error: { code: 'service_shutdown' } }]);
    expect(database.writes).toHaveLength(1);
    expect(database.writes[0].slice(2, 10)).toEqual([100, 0, 0, 0, 20, 0, 1, 0]);
  });

  it('preserves split UTF-8, CRLF and multiline data through the completed response', async () => {
    const delta = '준비🌟';
    const result = { ...terminal(), output: [
      { type: 'reasoning', encrypted_content: 'opaque-final-only', summary: [] },
      { type: 'message', role: 'assistant', content: [{ type: 'output_text', text: delta }] },
    ] };
    const text = ': keepalive\r\nevent: response.output_text.delta\r\n'
      + [{ type: 'response.output_text.delta', delta }, { type: 'response.completed', response: result }]
        .map((event) => JSON.stringify(event, null, 2).split('\n').map((line) => `data: ${line}`).join('\r\n') + '\r\n\r\n')
        .join('');
    const bytes = new TextEncoder().encode(text);
    const body = new ReadableStream<Uint8Array>({ start(controller) {
      for (let offset = 0; offset < bytes.length; offset++) controller.enqueue(bytes.subarray(offset, offset + 1));
      controller.close();
    } });
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(body, { headers: { 'content-type': 'text/event-stream' } }));
    const database = usageDatabase();
    expect(await relayEvents(await relay.fetch(request(), { ...testEnv(), AGENT_USAGE_DB: database.db }))).toEqual([
      { type: 'text_delta', text: delta },
      { type: 'completed', response: result },
    ]);
    expect(database.writes).toHaveLength(1);
    expect(database.writes[0].slice(2, 10)).toEqual([100, 0, 0, 0, 20, 0, 1, 0]);
  });

  it('rejects an unfinished event at EOF without publishing a completed response', async () => {
    const text = 'data: {"type":"response.output_text.delta","delta":"Provisional."}\n\n'
      + 'data: {"type":"response.completed"}\n';
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(text, { headers: { 'content-type': 'text/event-stream' } }));
    const database = usageDatabase();
    expect(await relayEvents(await relay.fetch(request(), { ...testEnv(), AGENT_USAGE_DB: database.db }))).toEqual([
      { type: 'text_delta', text: 'Provisional.' },
      { type: 'error', error: { code: 'upstream_invalid_response' } },
    ]);
    expect(database.writes).toHaveLength(1);
    expect(database.writes[0][9]).toBe(1);
  });
});
