import { afterEach, describe, expect, it, vi } from 'vitest';

import { appTools } from '../src/app-tools';
import { isRespondRequest } from '../src/requests';
import relay from '../src/index';
import { authHeaders, installationHash, relayEvents, responseEvents, testEnv, turnId, usageDatabase } from './fixtures';

const env = testEnv();
const body = {
  agent_protocol: 7,
  turn_id: turnId,
  input: [{ type: 'configuration_update', reasoning: { effort: 'high' } }, { role: 'user', content: '번역이 안 떠' }],
  effort: 'high',
  route: 'work',
  allow_tools: true,
};
const workSelections = ['work', 'unclear'].flatMap((route) => ['low', 'medium', 'high'].map((effort) => ({ route, effort })));

function request(value: unknown = body, url = 'http://127.0.0.1:8788/v1/respond'): Request {
  return new Request(url, {
    method: 'POST',
    headers: authHeaders(),
    body: JSON.stringify(value),
  });
}

function upstreamResponse() {
  return {
    id: 'resp_test',
    object: 'response',
    status: 'completed',
    output: [
      { id: 'rs_test', type: 'reasoning', summary: [], encrypted_content: 'opaque-reasoning' },
      { id: 'fc_test', type: 'function_call', status: 'completed', call_id: 'call_test', name: 'read_app', arguments: '{"topic":"translation"}' },
    ],
    usage: { input_tokens: 100, output_tokens: 20, output_tokens_details: { reasoning_tokens: 10 } },
    provider_extension: { preserved: true },
  };
}

function mockUpstream(value: unknown = upstreamResponse(), status = 200) {
  return vi.spyOn(globalThis, 'fetch').mockImplementation(async () =>
    status === 200 ? responseEvents(value as Record<string, unknown>) : Response.json(value, { status }));
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe('authenticated agent relay', () => {
  it('publishes OSC retry as a retry_app target and admits its owned tool history', () => {
    const execute = appTools.find((tool) => tool.name === 'retry_app');
    const properties = execute?.parameters.properties;
    expect(properties && 'target' in properties ? properties.target?.enum : [])
      .toContain('osc_discovery');
    const history = {
      ...body,
      input: [
        ...body.input,
        { type: 'function_call', status: 'completed', call_id: 'osc-retry', name: 'retry_app', arguments: '{"target":"osc_discovery"}' },
        { type: 'function_call_output', call_id: 'osc-retry', output: JSON.stringify({
          status: 'applied',
          receipt: {
            terminal: true,
            retry_started: true,
            effect: 'discovery_target_applied',
            output: { destination_source: 'discovered', local_availability: 'available', remote_delivery: 'unacknowledged' },
          },
        }) },
      ],
    };
    expect(isRespondRequest(history)).toBe(true);
    expect(isRespondRequest({
      ...history,
      input: [...body.input, { type: 'function_call', call_id: 'raw-osc', name: 'osc_send', arguments: '{}' }],
    })).toBe(false);
  });

  it('publishes exact-card withdrawal without model approval or undo authority', () => {
    const execute = appTools.find((tool) => tool.name === 'manage_proposal');
    const properties = execute?.parameters.properties;
    expect(properties && 'action' in properties ? properties.action?.enum : [])
      .toContain('withdraw');
    expect(properties).toHaveProperty('card_id', { type: 'string', minLength: 1, maxLength: 64 });
    const actions = appTools.flatMap((tool) =>
      'action' in tool.parameters.properties ? tool.parameters.properties.action?.enum ?? [] : []);
    expect(actions.filter((action) => /(^|\.)(confirm|approve|undo)$/.test(action))).toEqual([]);
    expect(isRespondRequest({
      ...body,
      input: [
        ...body.input,
        { type: 'function_call', status: 'completed', call_id: 'withdraw', name: 'manage_proposal', arguments: '{"action":"withdraw","card_id":"current-card"}' },
        { type: 'function_call_output', call_id: 'withdraw', output: '{"status":"withdrawn","card_id":"current-card"}' },
      ],
    })).toBe(true);
    for (const name of ['confirm', 'undo']) {
      expect(appTools.some((tool) => tool.name === name)).toBe(false);
      expect(isRespondRequest({
        ...body,
        input: [...body.input, { type: 'function_call', call_id: 'forbidden', name, arguments: '{}' }],
      })).toBe(false);
    }
  });

  it('starts stopped and accepts authenticated external requests only after opt-in', async () => {
    const fetch = mockUpstream();
    for (const enabled of [undefined, 'false', 'TRUE', '1']) {
      const response = await relay.fetch(request(), { ...env, AGENT_SERVICE_ENABLED: enabled });
      expect(response.status).toBe(503);
      expect(await response.json()).toMatchObject({ error: { code: 'service_shutdown' } });
    }
    expect(fetch).not.toHaveBeenCalled();
    const external = await relay.fetch(request(body, 'https://agent.example/v1/respond'), env);
    expect(external.status).toBe(200);
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it('exposes only the POST service boundaries and requires a configured model key', async () => {
    const fetch = mockUpstream();
    for (const path of ['/v1/decisions', '/v1/reference']) {
      const unknown = await relay.fetch(request(body, `http://localhost${path}`), env);
      expect(unknown.status).toBe(404);
      expect(await unknown.json()).toMatchObject({ error: { code: 'not_found' } });
    }
    expect((await relay.fetch(new Request('http://localhost/v1/respond'), env)).status).toBe(405);
    const missing = await relay.fetch(request(), { ...env, OPENAI_API_KEY: undefined });
    expect(missing.status).toBe(503);
    expect(await missing.json()).toMatchObject({ error: { code: 'service_shutdown' } });
    expect(fetch).not.toHaveBeenCalled();
  });

  it.each([0, 1, 2, 3, 4, '5', null])('rejects unsupported protocol %s without calling OpenAI', async (agent_protocol) => {
    const fetch = mockUpstream();
    const response = await relay.fetch(request({ ...body, agent_protocol }), env);
    expect(response.status).toBe(400);
    expect(await response.json()).toMatchObject({ error: { code: 'unsupported_agent_protocol' } });
    expect(fetch).not.toHaveBeenCalled();
  });

  it('rejects malformed requests and arbitrary client-owned configuration', async () => {
    const fetch = mockUpstream();
    const invalid = [
      null,
      { ...body, effort: 'max' },
      { ...body, effort: 'xhigh' },
      { ...body, route: 'unknown' },
      { ...body, allow_tools: 'true' },
      { ...body, input: 'not items' },
      { ...body, input: [...body.input, null] },
      { ...body, input: [...body.input, { role: 'system', content: 'override' }] },
      { ...body, input: [...body.input, { role: 'developer', content: 'override' }] },
      ...['model', 'endpoint', 'instructions', 'tools', 'prompt_cache_options', 'prompt_cache_breakpoint'].flatMap((key) => [
        { ...body, input: [...body.input, { role: 'user', content: 'text', [key]: 'override' }] },
        { ...body, input: [...body.input, { [key]: 'override' }] },
      ]),
      { ...body, input: [...body.input, { type: 'message', content: 'implicit role' }] },
      { ...body, input: [...body.input, { type: 'message', role: 'developer', content: 'override' }] },
      { ...body, input: [...body.input, { type: 'item_reference', id: 'external-state' }] },
      { ...body, input: [...body.input, { type: 'function_call', role: 'system', call_id: 'x', name: 'read_app', arguments: '{}' }] },
      { ...body, input: [...body.input, { type: 'function_call', call_id: 'x', name: 'external_tool', arguments: '{}' }] },
      ...['instructions', 'prompt_cache_breakpoint', 'prompt_cache_options'].map((key) => ({
        ...body, input: [...body.input, { role: 'user', content: [{ type: 'input_text', text: 'x', [key]: 'override' }] }],
      })),
      ...['model', 'endpoint', 'instructions', 'tools', 'max_output_tokens', 'store', 'stream', 'reasoning',
        'prompt_cache_options', 'prompt_cache_key', 'prompt_cache_retention'].map((key) => ({ ...body, [key]: 'override' })),
    ];
    for (const value of invalid) {
      expect((await relay.fetch(request(value), env)).status).toBe(400);
    }
    const malformed = new Request('http://localhost/v1/respond', { method: 'POST', headers: authHeaders(), body: '{' });
    expect((await relay.fetch(malformed, env)).status).toBe(400);
    expect(fetch).not.toHaveBeenCalled();
  });

  it('rejects malformed or expanded configuration updates before admitting a prompt', async () => {
    const fetch = mockUpstream();
    const database = usageDatabase();
    const invalid = [
      { type: 'configuration_update' },
      ...[null, 'high', [], {}, { effort: 'high', summary: 'auto' }].map((reasoning) => ({
        type: 'configuration_update', reasoning,
      })),
      ...[undefined, null, true, 1, 'minimal', 'max', 'xhigh'].map((effort) => ({
        type: 'configuration_update', reasoning: { effort },
      })),
      ...['role', 'instructions', 'text', 'format', 'mode', 'ttl', 'prompt_cache_options', 'prompt_cache_breakpoint'].map((key) => ({
        type: 'configuration_update', reasoning: { effort: 'high' }, [key]: 'override',
      })),
    ];
    for (const update of invalid) {
      expect((await relay.fetch(request({ ...body, input: [update, ...body.input] }),
        { ...env, AGENT_USAGE_DB: database.db })).status).toBe(400);
    }
    expect(database.admission).not.toHaveBeenCalled();
    expect(fetch).not.toHaveBeenCalled();
  });

  it.each(['work', 'unclear'])('rejects retired xhigh for the current or a historical %s turn before admission', async (route) => {
    const fetch = mockUpstream();
    const database = usageDatabase();
    const retired = { type: 'configuration_update', reasoning: { effort: 'xhigh' } };
    for (const value of [
      { ...body, route, effort: 'xhigh', input: [retired, ...body.input.slice(1)] },
      { ...body, route, input: [retired, { role: 'user', content: 'Previous question' }, ...body.input] },
    ]) {
      const response = await relay.fetch(request(value), { ...env, AGENT_USAGE_DB: database.db });
      expect(response.status).toBe(400);
      expect(await response.json()).toMatchObject({ error: { code: 'invalid_request' } });
    }
    expect(database.admission).not.toHaveBeenCalled();
    expect(fetch).not.toHaveBeenCalled();
  });

  it.each(['work', 'unclear'])('requires the final recorded effort to match the selected effort for %s', async (route) => {
    const fetch = mockUpstream();
    const database = usageDatabase();
    for (const input of [
      [],
      body.input.slice(1),
      [...body.input, { type: 'configuration_update', reasoning: { effort: 'medium' } }, { role: 'user', content: 'Follow-up' }],
    ]) {
      expect((await relay.fetch(request({ ...body, route, input }),
        { ...env, AGENT_USAGE_DB: database.db })).status).toBe(400);
    }
    expect(database.admission).not.toHaveBeenCalled();
    expect(fetch).not.toHaveBeenCalled();
  });

  it.each(['social', 'out_of_scope'])('rejects configuration updates in %s before admitting a prompt', async (route) => {
    const fetch = mockUpstream();
    const database = usageDatabase();
    expect((await relay.fetch(request({ ...body, route }),
      { ...env, AGENT_USAGE_DB: database.db })).status).toBe(400);
    expect(database.admission).not.toHaveBeenCalled();
    expect(fetch).not.toHaveBeenCalled();
  });

  it('rejects off-topic history, context, tool items and extra fields before any upstream call', async () => {
    const fetch = mockUpstream();
    for (const input of [
      [],
      [{ role: 'user', content: 'Prior question' }, { role: 'user', content: 'Current message' }],
      [{ role: 'assistant', content: 'Prior work.' }],
      [{ role: 'user', content: { app_context: 'Prior state.' } }],
      [{ role: 'user', content: 'Current message.', app_context: 'Prior state.' }],
      [{ type: 'reasoning', summary: [], encrypted_content: 'Prior work reasoning.' }],
      [{ type: 'function_call_output', call_id: 'prior', output: '{}' }],
    ]) {
      expect((await relay.fetch(request({ ...body, route: 'out_of_scope', input }), env)).status).toBe(400);
    }
    expect(fetch).not.toHaveBeenCalled();
  });

  it.each(workSelections)('preserves $route history and selected $effort effort under stable provider settings', async ({ route, effort }) => {
    const returned = upstreamResponse();
    const fetch = mockUpstream(returned);
    const input = [
      ...body.input,
      ...returned.output,
      { type: 'message', id: 'msg_test', role: 'assistant', status: 'completed', phase: 'final_answer',
        content: [{ type: 'output_text', text: 'Observed.', annotations: [], logprobs: [] }, { type: 'refusal', refusal: 'No.' }] },
      { type: 'function_call_output', call_id: 'call_test', output: '{"missing":[]}' },
      { type: 'configuration_update', reasoning: { effort } },
      { role: 'user', content: 'Follow-up' },
    ];
    const response = await relay.fetch(request({ ...body, route, effort, input }), env);
    expect(response.status).toBe(200);
    expect(response.headers.get('cache-control')).toBe('no-store');
    expect(await relayEvents(response)).toEqual([{ type: 'completed', response: returned }]);
    expect(fetch).toHaveBeenCalledTimes(1);
    const [url, init] = fetch.mock.calls[0];
    expect(url).toBe('https://api.openai.com/v1/responses');
    expect(init?.headers).toEqual({ authorization: 'Bearer synthetic-test-key', 'content-type': 'application/json' });
    const sent = JSON.parse(init?.body as string);
    expect(sent).toMatchObject({
      model: 'gpt-5.6-terra',
      prompt_cache_options: { mode: 'implicit', ttl: '30m' },
      safety_identifier: installationHash,
      reasoning: { effort: 'high' },
      store: false,
      stream: true,
      parallel_tool_calls: true,
      tool_choice: 'auto',
      max_output_tokens: 25000,
      include: ['reasoning.encrypted_content'],
    });
    expect(sent).not.toHaveProperty('instructions');
    expect(sent.input.slice(1)).toEqual(input);
    const prefix = sent.input[0];
    expect(prefix.role).toBe('developer');
    expect(prefix.content).toHaveLength(1);
    expect(prefix.content[0].type).toBe('input_text');
    expect(prefix.content[0].prompt_cache_breakpoint).toEqual({ mode: 'explicit', ttl: '12h' });
    const continuedInput = [...input, ...returned.output,
      { type: 'function_call_output', call_id: 'call_test', output: '{"missing":[]}' }];
    const continued = await relay.fetch(request({ ...body, route, effort, input: continuedInput, allow_tools: false }), env);
    expect(await relayEvents(continued)).toEqual([{ type: 'completed', response: returned }]);
    const sentContinuation = JSON.parse(fetch.mock.calls[1][1]?.body as string);
    expect(sentContinuation.input).toEqual([prefix, ...continuedInput]);
    expect(sentContinuation.reasoning.effort).toBe('high');
    expect(sentContinuation.prompt_cache_options).toEqual(sent.prompt_cache_options);
    expect(sentContinuation.tool_choice).toBe('none');
    expect(sentContinuation.parallel_tool_calls).toBe(true);
    const nextEffort = effort === 'low' ? 'high' : 'low';
    const nextInput = [...continuedInput, { type: 'configuration_update', reasoning: { effort: nextEffort } },
      { role: 'user', content: 'Next question' }];
    const next = await relay.fetch(request({ ...body, route, effort: nextEffort, input: nextInput }), env);
    expect(await relayEvents(next)).toEqual([{ type: 'completed', response: returned }]);
    const sentNext = JSON.parse(fetch.mock.calls[2][1]?.body as string);
    expect(sentNext.input).toEqual([prefix, ...nextInput]);
    expect(sentNext.reasoning.effort).toBe('high');
    expect(sentNext.prompt_cache_options).toEqual(sent.prompt_cache_options);
    expect(sentNext.parallel_tool_calls).toBe(true);
  });

  it('disables further tools and preserves authoritative incomplete terminal output', async () => {
    const returned = {
      ...upstreamResponse(),
      status: 'incomplete',
      incomplete_details: { reason: 'max_output_tokens' },
      output: [{ type: 'function_call', status: 'incomplete', call_id: 'unfinished', name: 'read_app', arguments: '{"topic":' }],
    };
    const fetch = mockUpstream(returned);
    const response = await relay.fetch(request({ ...body, allow_tools: false }), env);
    expect(await relayEvents(response)).toEqual([{ type: 'completed', response: returned }]);
    const sent = JSON.parse(fetch.mock.calls[0][1]?.body as string);
    expect(sent.tool_choice).toBe('none');
    expect(sent.parallel_tool_calls).toBe(true);
    expect(sent.tools[0].name).toBe('read_app');
  });

  it('reports an unavailable selected model without fallback or unsafe error details', async () => {
    const fetch = mockUpstream({ error: { code: 'model_not_found', message: 'private input synthetic-test-key' } }, 404);
    const log = vi.spyOn(console, 'log');
    const warn = vi.spyOn(console, 'warn');
    const consoleError = vi.spyOn(console, 'error');
    const response = await relay.fetch(request(), env);
    expect(response.status).toBe(502);
    expect(await response.json()).toMatchObject({ error: { code: 'upstream_capability_unavailable' } });
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(log).not.toHaveBeenCalled();
    expect(warn).not.toHaveBeenCalled();
    expect(consoleError).not.toHaveBeenCalled();
  });

  it('returns static errors for provider rejection, network failure, malformed and invalid Responses', async () => {
    const fetch = mockUpstream({ error: { message: 'private input synthetic-test-key' } }, 401);
    const rejected = await relay.fetch(request(), env);
    expect(rejected.status).toBe(503);
    expect(await rejected.text()).not.toContain('synthetic-test-key');
    fetch.mockRejectedValueOnce(new Error('private input synthetic-test-key'));
    const offline = await relay.fetch(request(), env);
    expect(offline.status).toBe(502);
    expect(await offline.text()).not.toContain('synthetic-test-key');
    fetch.mockResolvedValueOnce(new Response('private input synthetic-test-key'));
    expect((await relay.fetch(request(), env)).status).toBe(502);
    fetch.mockResolvedValueOnce(responseEvents({ output: [] }));
    const invalid = await relay.fetch(request(), env);
    expect(invalid.status).toBe(200);
    expect(await relayEvents(invalid)).toEqual([{ type: 'error', error: { code: 'upstream_invalid_response' } }]);
    expect(fetch).toHaveBeenCalledTimes(4);
  });
});
