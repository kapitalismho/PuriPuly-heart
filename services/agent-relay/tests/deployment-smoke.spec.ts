import { createServer } from 'node:http';
import type { AddressInfo } from 'node:net';
import { setTimeout as delay } from 'node:timers/promises';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { smokeDeployment } from '../operations/smoke-deployment.mjs';
import relay from '../src/index';
import { tokenSecret } from './fixtures';

vi.mock('node:timers/promises', () => ({ setTimeout: vi.fn(async () => {}) }));

const customOrigin = 'https://puripuly-relay.salee.sarl';
const readinessPath = '/deployment-smoke-not-an-endpoint';
const stoppedSmoke = { secret: tokenSecret, modelEnabled: false, supportEnabled: false };
const networkFetch = globalThis.fetch;
afterEach(() => {
  vi.mocked(delay).mockClear();
  vi.restoreAllMocks();
});

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

  it('does not accept oversized/non-JSON errors or cacheable errors or leak response text', async () => {
    for (const response of [new Response('sensitive-synthetic-response'), new Response('x'.repeat(4097)),
      Response.json({ error: { code: 'not_found' } }, { status: 404 })]) {
      await loopback(async () => response, async (origin) => {
        const result = smokeDeployment({ origin, secret: tokenSecret, modelEnabled: false, supportEnabled: false,
          allowLoopback: true, fetcher: networkFetch });
        await expect(result).rejects.toThrow(/^Relay smoke (request failed|rejection mismatch)/);
        await expect(result).rejects.not.toThrow('sensitive-synthetic-response');
      });
    }
  });

  it('does not send an authenticated rejection probe or payload to a reachable redirect destination', async () => {
    const env = { PURI_TOKEN_SECRET: tokenSecret, AGENT_SERVICE_ENABLED: 'true' };
    const destination = vi.fn((request: Request) => relay.fetch(request, env));
    const options = { ...stoppedSmoke, modelEnabled: true, allowLoopback: true, fetcher: networkFetch };
    await loopback(destination, async (secondOrigin) => {
      expect(await smokeDeployment({ origin: secondOrigin, ...options })).toMatchObject({ probes: 13 });
      destination.mockClear();
      const redirect = vi.fn(() => new Response(null, { status: 307, headers: { location: `${secondOrigin}/v1/respond` } }));
      await loopback((request) => {
        const authorization = request.headers.get('authorization');
        if (new URL(request.url).pathname === '/v1/respond' && authorization && authorization !== 'Bearer invalid') {
          return Promise.resolve(redirect());
        }
        return relay.fetch(request, env);
      }, async (origin) => {
        await expect(smokeDeployment({ origin, ...options })).rejects.toThrow('rejected redirect for /v1/respond (HTTP 307)');
      });
      expect(redirect).toHaveBeenCalledTimes(1);
      expect(destination).not.toHaveBeenCalled();
    });
  });

  it('waits through transient transport and gateway faults, then runs the full Relay contract once', async () => {
    const fetcher = vi.fn<typeof fetch>(async (input, init) =>
      relay.fetch(new Request(input, init), { PURI_TOKEN_SECRET: tokenSecret }));
    fetcher.mockRejectedValueOnce(new TypeError('synthetic TLS failure', { cause: { code: 'ERR_SSL_TLSV1_ALERT_INTERNAL_ERROR' } }));
    fetcher.mockResolvedValueOnce(new Response('synthetic provisioning HTML', { status: 525 }));
    await expect(smokeDeployment({ origin: customOrigin, ...stoppedSmoke, fetcher })).resolves.toMatchObject({ probes: 13 });
    expect(fetcher).toHaveBeenCalledTimes(15);
    expect(fetcher.mock.calls.slice(0, 3).map(([url, init]) => [url, init?.method, init?.headers])).toEqual(
      Array(3).fill([`${customOrigin}${readinessPath}`, 'GET', undefined]),
    );
    expect(delay).toHaveBeenCalledTimes(2);
  });

  it('bounds a persistent not-ready response and never progresses to authenticated requests', async () => {
    const fetcher = vi.fn<typeof fetch>(async () => new Response('synthetic provisioning HTML', { status: 503 }));
    await expect(smokeDeployment({ origin: customOrigin, ...stoppedSmoke, fetcher })).rejects
      .toThrow('readiness did not settle for /deployment-smoke-not-an-endpoint after 12 attempts (HTTP 503)');
    expect(fetcher).toHaveBeenCalledTimes(12);
    expect(delay).toHaveBeenCalledTimes(11);
    expect(fetcher.mock.calls.every(([url, init]) => url === `${customOrigin}${readinessPath}` && init?.method === 'GET' && !init.headers)).toBe(true);
  });

  it.each([
    ['redirect', () => new Response(null, { status: 302, headers: { location: 'https://ignored.invalid' } }), /rejected redirect/],
    ['contract mismatch', () => Response.json({ error: { code: 'other' } }, { status: 404 }), /rejection mismatch/],
    ['edge rejection', () => new Response('synthetic HTML', { status: 403 }), /request failed/],
    ['certificate validation failure', () => { throw new TypeError('synthetic invalid TLS', { cause: { code: 'ERR_TLS_CERT_ALTNAME_INVALID' } }); }, /request failed/],
  ] as const)('does not retry %s on the custom readiness URL', async (_label, response, failure) => {
    const fetcher = vi.fn<typeof fetch>(async () => response());
    await expect(smokeDeployment({ origin: customOrigin, ...stoppedSmoke, fetcher })).rejects.toThrow(failure);
    expect(fetcher).toHaveBeenCalledTimes(1);
    expect(delay).not.toHaveBeenCalled();
  });

  it('rejects arbitrary targets and invalid secrets before networking', async () => {
    const fetcher = vi.fn<typeof fetch>();
    for (const change of [{ origin: 'https://not-the-relay.invalid' }, { origin: 'http://127.0.0.1:1234' },
      { secret: '' }, { secret: 'short' }]) {
      await expect(smokeDeployment({ origin: 'https://puripuly-relay.salee.sarl', secret: tokenSecret,
        modelEnabled: false, supportEnabled: false, fetcher, ...change })).rejects.toThrow();
    }
    expect(fetcher).not.toHaveBeenCalled();
  });
});
