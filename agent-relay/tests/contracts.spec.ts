import { readdirSync, readFileSync } from 'node:fs';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { contracts, currentProtocol, supportedProtocols } from '../src/contracts';
import relay from '../src/index';
import { authHeaders, responseEvents, testEnv } from './fixtures';

type AppRequest = { path: string; body: { agent_protocol: number; route?: string; input: Record<string, unknown>[] } };

const fixtures = new URL('./app-requests/', import.meta.url);
const recorded = new Map(readdirSync(fixtures).map((name) => {
  const protocol = /^protocol-(\d+)\.json$/.exec(name)?.[1];
  if (!protocol) throw new Error(`Unexpected app request fixture ${name}`);
  return [Number(protocol), JSON.parse(readFileSync(new URL(name, fixtures), 'utf8')) as AppRequest[]];
}));
const prompts = new URL('../prompts/agent/', import.meta.url);
const sharedPrompts = ['system.md', 'guide.md', 'scope.md', 'references/keys-and-costs.md', 'references/benchmark.md']
  .map((name) => readFileSync(new URL(name, prompts), 'utf8')).join('\n');

afterEach(() => vi.restoreAllMocks());

describe('app protocol contracts', () => {
  it('keeps a recorded app request fixture for exactly the supported protocols, including the current one', () => {
    expect(supportedProtocols).toContain(currentProtocol);
    expect([...recorded.keys()].sort((a, b) => a - b)).toEqual(supportedProtocols);
  });

  it.each(supportedProtocols)('serves recorded protocol %i app requests with that protocol contract', async (protocol) => {
    const contract = contracts.get(protocol)!;
    const names = contract.tools.map((tool) => tool.name);
    const fetch = vi.spyOn(globalThis, 'fetch').mockImplementation(async (url) => String(url).endsWith('/decisions')
      ? Response.json({ answers: [], usage: { input_tokens: 1, output_tokens: 0 } })
      : responseEvents({ object: 'response', status: 'completed', output: [] }));
    const env = testEnv();
    for (const { path, body } of recorded.get(protocol)!) {
      const response = await relay.fetch(new Request(`http://127.0.0.1${path}`, {
        method: 'POST', headers: authHeaders(), body: JSON.stringify(body),
      }), env);
      expect(response.status, `${path} ${body.route ?? ''}`).toBe(200);
      const sent = JSON.parse(fetch.mock.calls.at(-1)![1]?.body as string);
      expect(sent.input.slice(sent.input.length - body.input.length)).toEqual(body.input);
      if (path === '/v1/classify') {
        expect(sent.questions.map(({ name }: { name: string }) => name))
          .toEqual(expect.arrayContaining(contract.contextModules.map(({ name }) => `context_${name}`)));
      } else if (body.route === 'work' || body.route === 'unclear') {
        expect(sent.tools.map(({ name }: { name: string }) => name)).toEqual(names);
        expect(sent.input[0].content[0].text).toContain(contract.appContext.replaceAll('\r\n', '\n').trim());
      }
      for (const item of body.input) {
        if (item.type === 'function_call') expect(names).toContain(item.name);
      }
    }
    expect(fetch).toHaveBeenCalledTimes(recorded.get(protocol)!.length);
  });

  it('keeps shared prompts limited to tools every supported protocol provides', () => {
    const all = new Set([...contracts.values()].flatMap(({ tools }) => tools.map(({ name }) => name)));
    for (const name of all) {
      if (!new RegExp(`\\b${name}\\b`).test(sharedPrompts)) continue;
      for (const [protocol, { tools }] of contracts) {
        expect(tools.map((tool) => tool.name), `${name} in shared prompts, protocol ${protocol}`).toContain(name);
      }
    }
  });
});
