import { randomUUID } from 'node:crypto';
import { readFile, rm } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { describe, expect, it, vi } from 'vitest';

import { configureSupportStorage } from '../operations/support-storage.mjs';
const desired = { id: 'support-report-retention-30d', enabled: true, conditions: { prefix: 'support/' },
  deleteObjectsTransition: { condition: { type: 'Age', maxAge: 2592000 } } };
const unrelated = { id: 'multipart-cleanup', enabled: true, conditions: { prefix: '' },
  abortMultipartUploadsTransition: { condition: { type: 'Age', maxAge: 86400 } } };

function api(rules: unknown[] = [desired, unrelated], managed = false, domains: unknown[] = []) {
  let current = rules;
  const fetcher = vi.fn(async (url: string, options: RequestInit) => {
    let result: unknown;
    if (url.endsWith('/domains/managed')) result = { enabled: managed };
    else if (url.endsWith('/domains/custom')) result = { domains };
    else if (options.method === 'PUT') {
      current = JSON.parse(options.body as string).rules;
      result = {};
    } else result = { rules: current };
    return Response.json({ success: true, result });
  });
  return { fetcher, options: { mode: 'verify', bucket: 'private-support', account: 'a'.repeat(32),
    token: 'synthetic-token', fetcher } };
}

describe('support storage operational readiness', () => {
  it('reads back exact expiration and both public-access boundaries without mutating verification', async () => {
    const { options, fetcher } = api();
    expect(await configureSupportStorage(options)).toEqual({ bucket: 'private-support', private: true,
      retention_days: 30, prefix: 'support/', enabled_config_created: false });
    expect(fetcher.mock.calls.map(([, value]) => value.method)).toEqual(Array(6).fill('GET'));
  });

  it('configures exact retention while preserving unrelated multipart rules', async () => {
    const { options, fetcher } = api([unrelated]);
    await configureSupportStorage({ ...options, mode: 'configure' });
    const puts = fetcher.mock.calls.filter(([, value]) => value.method === 'PUT');
    expect(puts).toHaveLength(1);
    expect(JSON.parse(puts[0][1].body as string)).toEqual({ rules: [unrelated, desired] });
  });

  it('refuses public buckets, missing retention and overlapping expiration', async () => {
    for (const options of [api(undefined, true).options, api(undefined, false, [{ enabled: false }]).options,
      api([unrelated]).options, api([desired, { ...desired, id: 'early', conditions: { prefix: '' } }]).options]) {
      await expect(configureSupportStorage(options)).rejects.toThrow();
      expect(options.fetcher.mock.calls.some(([, value]) => value.method === 'PUT')).toBe(false);
    }
  });

  it('writes an explicitly enabled deployment config only after successful remote readback', async () => {
    const destination = new URL(`../.support-enabled-test-${randomUUID()}.json`, import.meta.url);
    try {
      const { options } = api();
      const result = await configureSupportStorage({ ...options, enabledConfig: fileURLToPath(destination) });
      expect(result.enabled_config_created).toBe(true);
      const config = JSON.parse(await readFile(destination, 'utf8'));
      expect(config.vars.SUPPORT_REPORTS_ENABLED).toBe('true');
      expect(config.account_id).toBe('a'.repeat(32));
      expect(config.r2_buckets).toEqual([{ binding: 'SUPPORT_REPORTS', bucket_name: 'private-support' }]);
      const defaults = JSON.parse(await readFile(new URL('../wrangler.jsonc', import.meta.url), 'utf8'));
      expect(defaults.vars.SUPPORT_REPORTS_ENABLED).toBe('false');
    } finally {
      await rm(destination, { force: true });
    }
  });

  it('refuses unsuccessful API readback and leaves reception configuration untouched', async () => {
    const { options } = api();
    options.fetcher.mockResolvedValueOnce(Response.json({ success: false, result: {} }, { status: 403 }));
    await expect(configureSupportStorage(options)).rejects.toThrow('Cloudflare storage operation failed (403)');
  });
});
