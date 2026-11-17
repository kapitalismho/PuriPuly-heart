import { createServer } from 'node:http';
import type { AddressInfo } from 'node:net';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { smokeDeployment } from '../operations/smoke-deployment.mjs';
import relay from '../src/index';
import { tokenSecret } from './fixtures';

const networkFetch = globalThis.fetch;
afterEach(() => vi.restoreAllMocks());

async function loopback<T>(handler: (request: Request) => Promise<Response>, run: (origin: string) => Promise<T>) {
  const server = createServer(async (incoming, outgoing) => {
    try {
      const chunks: Buffer[] = [];
      for await (const chunk of incoming) chunks.push(Buffer.from(chunk));
      const headers = new Headers();
      for (const [name, value] of Object.entries(incoming.headers)) {
        if (typeof value === 'string') headers.set(name, value);
      }
      const request = new Request(`http://127.0.0.1${incoming.url}`, { method: incoming.method, headers,
        ...(['GET', 'HEAD'].includes(incoming.method ?? '') ? {} : { body: Buffer.concat(chunks).toString('utf8') }) });
      const response = await handler(request);
      outgoing.writeHead(response.status, Object.fromEntries(response.headers));
      outgoing.end(Buffer.from(await response.arrayBuffer()));
    } catch {
      outgoing.writeHead(500);
      outgoing.end('controlled server failure');
    }
  });
  const ready = Promise.withResolvers<void>();
  server.once('error', ready.reject);
  server.listen(0, '127.0.0.1', ready.resolve);
  await ready.promise;
  try { return await run(`http://127.0.0.1:${(server.address() as AddressInfo).port}`); }
  finally {
    const closed = Promise.withResolvers<void>();
    server.close((error) => error ? closed.reject(error) : closed.resolve());
    await closed.promise;
  }
}

describe('deployment rejection smoke against the real relay handler', () => {
  it.each([[true, true], [true, false], [false, true], [false, false]])(
    'observes authentication, method and protocol rejection with models=%s reports=%s and zero model, D1 or R2 operations',
    async (modelEnabled, supportEnabled) => {
      const upstream = vi.spyOn(globalThis, 'fetch').mockRejectedValue(new Error('External networking is forbidden.'));
      const database = vi.fn(() => { throw new Error('Smoke must not access D1.'); });
      const bucket = vi.fn(() => { throw new Error('Smoke must not access R2.'); });
      const env = { PURI_TOKEN_SECRET: tokenSecret, AGENT_SERVICE_ENABLED: String(modelEnabled),
        SUPPORT_REPORTS_ENABLED: String(supportEnabled), OPENAI_API_KEY: 'synthetic-provider-key-no-access',
        AGENT_USAGE_DB: { prepare: database, batch: database } as unknown as D1Database,
        SUPPORT_REPORTS: { head: bucket, put: bucket } as unknown as R2Bucket };
      await loopback((request) => relay.fetch(request, env), async (origin) => {
        const result = await smokeDeployment({ origin, secret: tokenSecret, modelEnabled, supportEnabled,
          allowLoopback: true, fetcher: networkFetch });
        expect(result).toEqual({ status: 'rejection_smoke_passed', probes: 13, model_requests: 0, support_uploads: 0 });
      });
      expect(upstream).not.toHaveBeenCalled();
      expect(database).not.toHaveBeenCalled();
      expect(bucket).not.toHaveBeenCalled();
    });

  it('detects a mismatched shared token secret rather than treating anonymous rejection as healthy', async () => {
    const upstream = vi.spyOn(globalThis, 'fetch').mockRejectedValue(new Error('External networking is forbidden.'));
    await loopback((request) => relay.fetch(request, { PURI_TOKEN_SECRET: tokenSecret, AGENT_SERVICE_ENABLED: 'true' }), async (origin) => {
      await expect(smokeDeployment({ origin, secret: 'different-synthetic-secret-at-least-32-bytes',
        modelEnabled: true, supportEnabled: false, allowLoopback: true, fetcher: networkFetch })).rejects.toThrow('rejection mismatch');
    });
    expect(upstream).not.toHaveBeenCalled();
  });

  it('detects an unexpected service flag without sending a model-valid body', async () => {
    const upstream = vi.spyOn(globalThis, 'fetch').mockRejectedValue(new Error('External networking is forbidden.'));
    await loopback((request) => relay.fetch(request, { PURI_TOKEN_SECRET: tokenSecret, AGENT_SERVICE_ENABLED: 'false' }), async (origin) => {
      await expect(smokeDeployment({ origin, secret: tokenSecret, modelEnabled: true, supportEnabled: false,
        allowLoopback: true, fetcher: networkFetch })).rejects.toThrow('expected 400 unsupported_agent_protocol');
    });
    expect(upstream).not.toHaveBeenCalled();
  });

  it('does not follow redirects, accept oversized/non-JSON errors, cacheable errors or leaked response text', async () => {
    for (const response of [new Response(null, { status: 302, headers: { location: 'https://must-not-contact.invalid' } }),
      new Response('sensitive-synthetic-response'), new Response('x'.repeat(4097)),
      Response.json({ error: { code: 'not_found' } }, { status: 404 })]) {
      await loopback(async () => response, async (origin) => {
        const result = smokeDeployment({ origin, secret: tokenSecret, modelEnabled: false, supportEnabled: false,
          allowLoopback: true, fetcher: networkFetch });
        await expect(result).rejects.toThrow(/^Relay smoke (request failed|rejection mismatch)/);
        await expect(result).rejects.not.toThrow('sensitive-synthetic-response');
      });
    }
  });

  it('rejects arbitrary targets and invalid secrets before networking', async () => {
    const fetcher = vi.fn<typeof fetch>();
    for (const change of [{ origin: 'https://not-the-relay.invalid' }, { origin: 'http://127.0.0.1:1234' },
      { secret: '' }, { secret: 'short' }]) {
      await expect(smokeDeployment({ origin: 'https://puripuly-heart-agent-relay.synthetic.workers.dev', secret: tokenSecret,
        modelEnabled: false, supportEnabled: false, fetcher, ...change })).rejects.toThrow();
    }
    expect(fetcher).not.toHaveBeenCalled();
  });
});
