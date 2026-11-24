import { readdirSync, readFileSync } from 'node:fs';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { contracts, currentProtocol, supportedProtocols } from '../src/contracts';
import relay from '../src/index';
import { workInstructions } from '../src/instructions';
import { photoPolicy } from '../src/usage';
import { authHeaders, installationHash, responseEvents, testEnv } from './fixtures';

type AppRequest = { path: string; body: {
  agent_protocol: number;
  route?: string;
  allow_tools?: boolean;
  omitted_references?: string[];
  input: Record<string, unknown>[];
} };

const fixtures = new URL('./app-requests/', import.meta.url);
const recorded = new Map(readdirSync(fixtures).map((name) => {
  const protocol = /^protocol-(\d+)\.json$/.exec(name)?.[1];
  if (!protocol) throw new Error(`Unexpected app request fixture ${name}`);
  return [Number(protocol), JSON.parse(readFileSync(new URL(name, fixtures), 'utf8')) as AppRequest[]];
}));
const prompts = new URL('../prompts/agent/', import.meta.url);
const referenceFiles = {
  costs: 'references/keys-and-costs.md',
  benchmarks: 'references/benchmark.md',
  setup: 'references/provider-setup.md',
  photo: 'references/vr-photo.md',
};
const readPrompt = (file: string) => readFileSync(new URL(file, prompts), 'utf8').replaceAll('\r\n', '\n').trim();
const sharedPrompts = ['system.md', 'guide.md', 'scope.md', 'scope-references.md', ...Object.values(referenceFiles)]
  .map(readPrompt).join('\n');
const expectedReferences = (allowance: number, omitted: string[] = []) => Object.entries(referenceFiles)
  .filter(([name]) => !omitted.includes(name))
  .map(([, file]) => readPrompt(file).replace('{{photo_daily_allowance}}', String(allowance)));
const referenceMessage = (text: string) => ({ role: 'developer', content: [{ type: 'input_text', text }] });

afterEach(() => vi.restoreAllMocks());

describe('app protocol contracts', () => {
  it('keeps a recorded app request fixture for exactly the supported protocols, including the current one', () => {
    expect(currentProtocol).toBe(8);
    expect(supportedProtocols).toEqual([6, 7, 8]);
    expect([...recorded.keys()].sort((a, b) => a - b)).toEqual(supportedProtocols);
  });

  it('retains protocol six meaning and shares the complete protocol seven tool and app contract with eight', () => {
    const previous = contracts.get(6)!;
    const current = contracts.get(7)!;
    const selective = contracts.get(8)!;
    expect(previous.tools.map(({ name }) => name)).toEqual(['read_app', 'execute_app', 'propose_app']);
    expect(current.tools.map(({ name }) => name)).toEqual([
      'read_app', 'set_app', 'probe_app', 'retry_app', 'open_app', 'manage_proposal',
    ]);
    expect(current.tools.every((tool) => tool.strict === false)).toBe(true);
    expect(selective).toEqual({ ...current, selectReferences: true });
    const previousInstructions = workInstructions(previous, 23);
    const currentInstructions = workInstructions(current, 23);
    expect(previousInstructions).toContain(previous.executionBoundary);
    expect(previousInstructions).toContain('Explicit execute_app actions ui.accounts and ui.provider_setup');
    expect(previousInstructions).not.toContain('normal tools available on your next response');
    expect(currentInstructions).toContain(current.executionBoundary);
    expect(currentInstructions).toContain('Explicit open_app views accounts and provider_setup');
    expect(currentInstructions).toContain('independent reads still run and every call gets a result');
    expect(currentInstructions).not.toContain('{{execution_boundary}}');
    expect(currentInstructions).not.toContain('execute_app');
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
      const wireBody = fetch.mock.calls.at(-1)![1]?.body as string;
      const sent = JSON.parse(wireBody);
      expect(sent).not.toHaveProperty('omitted_references');
      if (path === '/v1/classify') {
        expect(sent.input.slice(1)).toEqual(body.input);
        expect(sent.questions.map(({ name }: { name: string }) => name))
          .toEqual(expect.arrayContaining(contract.contextModules.map(({ name }) => `context_${name}`)));
      } else if (body.route === 'work' || body.route === 'unclear') {
        const allowance = photoPolicy().allowance;
        const prefix = workInstructions(contract, allowance);
        const references = expectedReferences(allowance, body.omitted_references);
        const suffix = protocol === 8 && references.length ? [referenceMessage(references.join('\n\n'))] : [];
        if (protocol !== 8) expect(prefix.endsWith(`\n\n${references.join('\n\n')}`)).toBe(true);
        expect(wireBody).toBe(JSON.stringify({
          model: 'gpt-5.6-terra',
          safety_identifier: installationHash,
          tools: contract.tools,
          reasoning: { effort: 'high' },
          prompt_cache_options: { mode: 'implicit', ttl: '30m' },
          input: [{
            role: 'developer',
            content: [{
              type: 'input_text',
              text: prefix,
              prompt_cache_breakpoint: { mode: 'explicit', ttl: '12h' },
            }],
          }, ...body.input, ...suffix],
          store: false,
          stream: true,
          parallel_tool_calls: true,
          tool_choice: body.allow_tools ? 'auto' : 'none',
          max_output_tokens: 25000,
          include: ['reasoning.encrypted_content'],
        }));
      } else {
        expect(sent.input).toEqual(body.input);
      }
      for (const item of body.input) {
        if (item.type === 'function_call') expect(names).toContain(item.name);
      }
    }
    expect(fetch).toHaveBeenCalledTimes(recorded.get(protocol)!.length);
  });

  it.each(['work', 'unclear'])('appends only the selected %s references after client input without changing prefix bytes', async (route) => {
    const fetch = vi.spyOn(globalThis, 'fetch').mockImplementation(async () =>
      responseEvents({ object: 'response', status: 'completed', output: [] }));
    const work = recorded.get(8)!.find(({ body }) => body.route === 'work')!;
    const legacyReferences = `\n\n${expectedReferences(7).join('\n\n')}`;
    const prefixText = workInstructions(contracts.get(7)!, 7).slice(0, -legacyReferences.length);
    const prefix = { role: 'developer', content: [{
      type: 'input_text', text: prefixText, prompt_cache_breakpoint: { mode: 'explicit', ttl: '12h' },
    }] };
    const cases = [
      { allowance: 7, omitted_references: [] },
      { allowance: 9, omitted_references: ['photo', 'benchmarks', 'costs', 'setup'] },
      { allowance: 9, omitted_references: ['setup', 'costs'] },
    ];
    const wirePrefixes = [];
    for (const [index, { allowance, omitted_references }] of cases.entries()) {
      const input = [...work.body.input, { role: 'user', content: `Synthetic request ${index}` }];
      const response = await relay.fetch(new Request(`http://127.0.0.1${work.path}`, {
        method: 'POST', headers: authHeaders(), body: JSON.stringify({ ...work.body, route, input, omitted_references }),
      }), { ...testEnv(), PHOTO_POLICY: JSON.stringify({ allowance }) });
      expect(response.status).toBe(200);
      await response.text();
      const wire = fetch.mock.calls.at(-1)![1]?.body as string;
      const separator = wire.indexOf(',{"type":"configuration_update"');
      expect(separator).toBeGreaterThan(0);
      wirePrefixes.push(wire.slice(0, separator));
      const references = expectedReferences(allowance, omitted_references);
      expect(JSON.parse(wire).input).toEqual([prefix, ...input,
        ...(references.length ? [referenceMessage(references.join('\n\n'))] : [])]);
    }
    expect(new Set(wirePrefixes).size).toBe(1);
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
