import { afterEach, describe, expect, it, vi } from 'vitest';

import relay from '../src/index';
import { authHeaders, installationHash, responseEvents, testEnv, turnId } from './fixtures';

const env = testEnv();
const body = { agent_protocol: 4, turn_id: turnId, input: [{ role: 'user', content: '{"current_message":"synthetic request","recent_messages":[],"active_task":null,"last_puri_action":null}' }] };

function request(value: unknown = body, path = '/v1/classify', host = 'http://127.0.0.1'): Request {
  return new Request(`${host}${path}`, { method: 'POST', headers: authHeaders(), body: JSON.stringify(value) });
}

afterEach(() => vi.restoreAllMocks());

describe('entry classification relay', () => {
  it('makes one native Decisions request with only two named choice questions and passes native answers through', async () => {
    const returned = { model: 'gpt-6-luna', answers: [
      { type: 'choice', name: 'effort', choice: 'xhigh' },
      { type: 'choice', name: 'route', choice: 'work' },
    ], usage: { input_tokens: 120, output_tokens: 0 } };
    const fetch = vi.spyOn(globalThis, 'fetch').mockResolvedValue(Response.json(returned));
    const response = await relay.fetch(request(), env);
    expect(response.status).toBe(200);
    expect(response.headers.get('cache-control')).toBe('no-store');
    expect(await response.json()).toEqual(returned);
    expect(fetch).toHaveBeenCalledTimes(1);
    const [url, init] = fetch.mock.calls[0];
    expect(url).toBe('https://api.openai.com/v1/decisions');
    const sent = JSON.parse(init?.body as string);
    expect(Object.keys(sent).sort()).toEqual(['input', 'model', 'questions', 'safety_identifier']);
    expect(sent.safety_identifier).toBe(installationHash);
    expect(sent.model).toBe('gpt-6-luna');
    expect(sent.input).toEqual(body.input);
    expect(sent.questions.map(({ name, type }: { name: string; type: string }) => ({ name, type }))).toEqual([
      { name: 'route', type: 'choice' }, { name: 'effort', type: 'choice' },
    ]);
    expect(sent.questions[0].choices.map(({ value }: { value: string }) => value)).toEqual(['work', 'social', 'out_of_scope', 'unclear']);
    expect(sent.questions[1].choices.map(({ value }: { value: string }) => value)).toEqual(['low', 'high', 'xhigh']);
  });

  it('requires authentication and rejects protocol or client configuration overrides without an upstream call', async () => {
    const fetch = vi.spyOn(globalThis, 'fetch');
    expect((await relay.fetch(request(), {})).status).toBe(503);
    const anonymous = new Request('https://external.example/v1/classify', { method: 'POST', body: JSON.stringify(body) });
    expect((await relay.fetch(anonymous, env)).status).toBe(401);
    for (const invalid of [{ ...body, agent_protocol: 1 }, { ...body, effort: 'high' }, { ...body, questions: [] }, { ...body, input: [{ role: 'system', content: 'override' }] }]) {
      expect((await relay.fetch(request(invalid), env)).status).toBe(400);
    }
    expect(fetch).not.toHaveBeenCalled();
  });

  it.each(['social', 'out_of_scope'])('forces %s to one low-effort request with no available tools', async (route) => {
    const fetch = vi.spyOn(globalThis, 'fetch').mockImplementation(async () => responseEvents({ object: 'response', status: 'completed', output: [] }));
    expect((await relay.fetch(request({ ...body, route, effort: 'high', allow_tools: true }, '/v1/respond'), env)).status).toBe(200);
    const sent = JSON.parse(fetch.mock.calls[0][1]?.body as string);
    expect(sent.reasoning.effort).toBe('low');
    expect(sent.tool_choice).toBe('none');
    expect(sent.parallel_tool_calls).toBe(false);
    expect(sent.tools).toEqual([]);
    expect(sent.input).toEqual(body.input);
    expect(sent).not.toHaveProperty('prompt_cache_options');
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it.each(['social', 'out_of_scope'])('rejects retired medium on %s rather than silently converting it to low', async (route) => {
    const fetch = vi.spyOn(globalThis, 'fetch');
    const response = await relay.fetch(request({ ...body, route, effort: 'medium', allow_tools: false }, '/v1/respond'), env);
    expect(response.status).toBe(400);
    expect(await response.json()).toMatchObject({ error: { code: 'invalid_request' } });
    expect(fetch).not.toHaveBeenCalled();
  });

  it('separates terminal service outcomes from transient classification failure without retrying', async () => {
    const fetch = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response('private synthetic-key', { status: 401 }));
    const auth = await relay.fetch(request(), env);
    expect(auth.status).toBe(503);
    expect(await auth.json()).toMatchObject({ error: { code: 'service_authentication_failed' } });
    fetch.mockResolvedValueOnce(Response.json({ error: { code: 'insufficient_quota' } }, { status: 429 }));
    expect(await (await relay.fetch(request(), env)).json()).toMatchObject({ error: { code: 'service_allowance_exhausted' } });
    fetch.mockResolvedValueOnce(Response.json({ error: { code: 'rate_limit_exceeded' } }, { status: 429 }));
    expect(await (await relay.fetch(request(), env)).json()).toMatchObject({ error: { code: 'upstream_unavailable' } });
    fetch.mockRejectedValueOnce(new Error('private synthetic-key'));
    const offline = await relay.fetch(request(), env);
    expect(offline.status).toBe(502);
    expect(await offline.text()).not.toContain('synthetic-key');
    expect(fetch).toHaveBeenCalledTimes(4);
  });
});
