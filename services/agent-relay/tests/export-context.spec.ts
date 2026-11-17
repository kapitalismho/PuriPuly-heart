import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';

import { exportRelayContext, extractContext } from '../operations/export-context.mjs';

const input = [{ type: 'configuration_update', reasoning: { effort: 'high' } },
  { role: 'user', content: 'Synthetic current app request' }];
const prefix = { role: 'developer', content: [{ type: 'input_text', text: 'Exact emitted trusted guide.',
  prompt_cache_breakpoint: { mode: 'explicit' } }] };

describe('offline emitted relay context export', () => {
  it('extracts the actual trusted prefix and static fields without identity or client input', () => {
    const options = { model: 'gpt-6-luna', tools: [{ type: 'function', name: 'synthetic_tool' }],
      reasoning: { effort: 'high' }, prompt_cache_options: { mode: 'implicit', ttl: '30m' },
      store: false, stream: true, parallel_tool_calls: true, max_output_tokens: 25000,
      include: ['reasoning.encrypted_content'] };
    expect(extractContext({ ...options, safety_identifier: 'synthetic-installation', tool_choice: 'none',
      input: [prefix, ...input] }, input)).toEqual({ input_prefix: [prefix], ...options });
  });

  it('retains server-owned classification questions and social instructions exactly', () => {
    const question = { type: 'choice', name: 'route', instructions: 'Exact emitted route instructions.',
      choices: [{ value: 'work', description: 'Synthetic work.' }] };
    expect(extractContext({ model: 'gpt-6-luna', input, questions: [question] }, input)).toEqual({
      input_prefix: [], model: 'gpt-6-luna', questions: [question],
    });
    expect(extractContext({ model: 'gpt-6-luna', input, instructions: 'Exact emitted social instructions.',
      tools: [], reasoning: { effort: 'low' } }, input)).toEqual({ input_prefix: [], model: 'gpt-6-luna',
      instructions: 'Exact emitted social instructions.', tools: [], reasoning: { effort: 'low' } });
  });

  it('never treats sampled allow_tools-dependent choice as static context', () => {
    const body = { model: 'gpt-6-luna', input: [prefix, ...input], tools: [{ type: 'function', name: 'synthetic_tool' }] };
    expect(extractContext({ ...body, tool_choice: 'auto' }, input))
      .toEqual(extractContext({ ...body, tool_choice: 'none' }, input));
  });

  it('rejects changed client input and unknown static fields rather than guessing', () => {
    expect(() => extractContext({ model: 'gpt-6-luna', input: [prefix] }, input)).toThrow('changed client input');
    expect(() => extractContext({ model: 'gpt-6-luna', input: [...input].reverse() }, input)).toThrow('changed client input');
    expect(() => extractContext({ model: 'gpt-6-luna', input, unknown_option: true }, input)).toThrow('Unrecognized outbound static field');
  });

  it('rejects a wrong emitted build before starting the controlled runtime', async () => {
    const root = await mkdtemp(join(tmpdir(), 'puri-context-build-'));
    try {
      const emitted = join(root, 'index.js');
      await writeFile(emitted, 'export default {};\n');
      await expect(exportRelayContext({ emitted, relayBuild: '0'.repeat(64), compatibilityDate: '2026-04-08' }))
        .rejects.toThrow('does not match SHA256');
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it('requires the measured runtime compatibility settings before reading any source', async () => {
    await expect(exportRelayContext({ emitted: 'not-a-bundle', compatibilityDate: '' }))
      .rejects.toThrow('compatibility date and flags explicitly');
  });
});
