import { afterEach, describe, expect, it, vi } from 'vitest';

import relay from '../src/index';
import { authHeaders, installationHash, responseEvents, testEnv, turnId } from './fixtures';

const env = testEnv();
const body = { agent_protocol: 6, turn_id: turnId, input: [{ role: 'user', content: '{"current_message":"synthetic request","recent_messages":[],"active_task":null,"previous_modules":[],"last_puri_action":null}' }] };

function request(value: unknown = body, path = '/v1/classify', host = 'http://127.0.0.1'): Request {
  return new Request(`${host}${path}`, { method: 'POST', headers: authHeaders(), body: JSON.stringify(value) });
}

afterEach(() => vi.restoreAllMocks());

describe('entry classification relay', () => {
  it('makes one native Decisions request with server-owned route, effort and context questions and passes native answers through', async () => {
    const returned = { model: 'gpt-6-luna', answers: [
      { type: 'choice', name: 'effort', choice: 'medium' },
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
    expect(Object.keys(sent).sort()).toEqual(['input', 'model', 'prompt_cache_options', 'questions', 'safety_identifier']);
    expect(sent.safety_identifier).toBe(installationHash);
    expect(sent.model).toBe('gpt-6-luna');
    expect(sent.prompt_cache_options).toEqual({ mode: 'explicit', ttl: '30m' });
    expect(sent.input).toHaveLength(2);
    expect(sent.input.slice(1)).toEqual(body.input);
    const [policy] = sent.input;
    expect(policy.role).toBe('developer');
    expect(policy.content).toHaveLength(1);
    expect(policy.content[0]).toMatchObject({ type: 'input_text', prompt_cache_breakpoint: { mode: 'explicit' } });
    for (const heading of ['## Input', '## Route', '## Effort', '## Context modules', '## Task relation']) {
      expect(policy.content[0].text).toContain(heading);
    }
    for (const module of ['audio', 'recognition', 'translation', 'accounts', 'languages', 'output', 'general']) {
      expect(policy.content[0].text).toContain(`- ${module}: `);
    }
    expect(policy.content[0].text).not.toContain('{{');
    expect(sent.questions.map(({ name, type }: { name: string; type: string }) => ({ name, type }))).toEqual([
      { name: 'route', type: 'choice' }, { name: 'effort', type: 'choice' },
      { name: 'context_audio', type: 'choice' }, { name: 'context_recognition', type: 'choice' },
      { name: 'context_translation', type: 'choice' }, { name: 'context_accounts', type: 'choice' },
      { name: 'context_languages', type: 'choice' }, { name: 'context_output', type: 'choice' },
      { name: 'context_general', type: 'choice' }, { name: 'context_relation', type: 'choice' },
    ]);
    expect(sent.questions[0].choices.map(({ value }: { value: string }) => value)).toEqual(['work', 'social', 'out_of_scope', 'unclear']);
    expect(sent.questions[1].choices.map(({ value }: { value: string }) => value)).toEqual(['low', 'medium', 'high']);
    for (const question of sent.questions.slice(2, -1)) {
      expect(question.choices.map(({ value }: { value: string }) => value)).toEqual(['include', 'omit', 'unclear']);
    }
    expect(sent.questions.at(-1).choices.map(({ value }: { value: string }) => value)).toEqual(['new', 'continue', 'unclear']);
  });

  it('keeps every fixed classifier rule in one identical cacheable prefix rather than in uncached questions', async () => {
    const fetch = vi.spyOn(globalThis, 'fetch').mockImplementation(async () => Response.json({ answers: [], usage: { input_tokens: 1, output_tokens: 0 } }));
    const other = { ...body, input: [{ role: 'user', content: '{"current_message":"다른 합성 요청","recent_messages":[{"role":"assistant","content":"synthetic"}],"active_task":"synthetic task","previous_modules":["core","audio"],"last_puri_action":null}' }] };
    expect((await relay.fetch(request(), env)).status).toBe(200);
    expect((await relay.fetch(request(other), env)).status).toBe(200);
    const [first, second] = fetch.mock.calls.map(([, init]) => JSON.parse(init?.body as string));
    expect(second.input[0]).toEqual(first.input[0]);
    expect(second.questions).toEqual(first.questions);
    expect(second.input.slice(1)).toEqual(other.input);
    const policy = first.input[0].content[0].text as string;
    for (const question of first.questions) {
      const section = /by the (.+) section\.$/.exec(question.instructions)?.[1];
      expect(policy).toContain(`\n## ${section}\n`);
      for (const choice of question.choices) {
        expect(Object.keys(choice)).toEqual(['value']);
        expect(policy).toContain(`\n- ${choice.value}: `);
      }
    }
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

  it.each(['social', 'out_of_scope'])('rejects retired xhigh on %s rather than silently converting it to low', async (route) => {
    const fetch = vi.spyOn(globalThis, 'fetch');
    const response = await relay.fetch(request({ ...body, route, effort: 'xhigh', allow_tools: false }, '/v1/respond'), env);
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
