import type { Context } from 'hono';
import { describe, expect, it } from 'vitest';

import { extractRequestNetworkMetadata, resolveRequestNetworkIdentitySecrets } from '../src/abuse-controls';
import { resolveNetworkIdentitySecrets } from '../src/network-identity';
import type { BrokerEnv } from '../src/contract';
import { createTestBrokerEnv, type TestBrokerEnv } from './test-support/sqlite-d1';

function createContextWithRequest(request: Request, env: TestBrokerEnv): Context<BrokerEnv> {
  return {
    req: {
      raw: request,
      header: (name: string) => request.headers.get(name) ?? undefined,
    },
    env: env as unknown as Context<BrokerEnv>['env'],
  } as unknown as Context<BrokerEnv>;
}

function requestWithCloudflareMetadata(cf: Record<string, unknown>): Request {
  const request = new Request('https://broker.test/v1/providers/openrouter/issue', {
    headers: {
      'cf-connecting-ip': '203.0.113.42',
    },
  }) as Request & { cf?: Record<string, unknown> };
  request.cf = cf;

  return request;
}

describe('request network metadata extraction', () => {
  it('ignores spoofed ASN headers when Cloudflare request metadata is absent', async () => {
    const env = createTestBrokerEnv();
    const request = new Request('https://broker.test/v1/providers/openrouter/issue', {
      headers: {
        'cf-connecting-ip': '203.0.113.42',
        'x-test-cf-asn': '24940',
        'cf-asn': '15169',
      },
    });

    const metadata = await extractRequestNetworkMetadata(createContextWithRequest(request, env), {
      secrets: resolveNetworkIdentitySecrets(env),
      now: new Date('2026-04-08T06:00:00.000Z'),
    });

    expect(metadata.asn).toBeNull();
  });

  it('uses ASN from Cloudflare request metadata supplied by the test harness', async () => {
    const env = createTestBrokerEnv();
    const request = requestWithCloudflareMetadata({ asn: 24940 });

    const metadata = await extractRequestNetworkMetadata(createContextWithRequest(request, env), {
      secrets: resolveNetworkIdentitySecrets(env),
      now: new Date('2026-04-08T06:00:00.000Z'),
    });

    expect(metadata.asn).toBe(24940);
  });

  it('does not derive a risk label from HTTP or TLS metadata', async () => {
    const env = createTestBrokerEnv();
    const request = requestWithCloudflareMetadata({
      httpProtocol: 'HTTP/1.1',
      tlsVersion: 'TLSv1.0',
      tlsCipher: 'legacy-cipher',
    });

    const metadata = await extractRequestNetworkMetadata(createContextWithRequest(request, env), {
      secrets: resolveNetworkIdentitySecrets(env),
      now: new Date('2026-04-08T06:00:00.000Z'),
    });

    expect(metadata.riskLabel).toBeNull();
  });

  it('derives versioned keyed digests that differ from unkeyed hashes', async () => {
    const env = createTestBrokerEnv();
    const request = requestWithCloudflareMetadata({ asn: 24940 });

    const first = await extractRequestNetworkMetadata(createContextWithRequest(request, env), {
      secrets: resolveNetworkIdentitySecrets(env),
      now: new Date('2026-04-08T06:00:00.000Z'),
    });
    const second = await extractRequestNetworkMetadata(createContextWithRequest(request, env), {
      secrets: resolveNetworkIdentitySecrets(env),
      now: new Date('2026-04-08T18:00:00.000Z'),
    });

    expect(first.ipDigest).toMatch(/^[a-f0-9]{64}$/u);
    expect(first.ipPrefixDigest).toMatch(/^[a-f0-9]{64}$/u);
    expect(first.ipKeyVersion).toBe(1);
    expect(first.ipEpoch).toBe('2026-04-08');
    expect(second.ipDigest).toBe(first.ipDigest);
    const unkeyed = await crypto.subtle.digest(
      'SHA-256',
      new TextEncoder().encode('203.0.113.42'),
    );
    const unkeyedHex = Array.from(new Uint8Array(unkeyed), (byte) =>
      byte.toString(16).padStart(2, '0'),
    ).join('');
    expect(first.ipDigest).not.toBe(unkeyedHex);
    expect(first.legacyIp).toBeNull();
  });

  it('omits digests when the worker secret is unavailable', async () => {
    const env = createTestBrokerEnv();
    const request = requestWithCloudflareMetadata({ asn: 24940 });

    const metadata = await extractRequestNetworkMetadata(createContextWithRequest(request, env), {
      secrets: null,
      now: new Date('2026-04-08T06:00:00.000Z'),
    });

    expect(metadata.ipDigest).toBeNull();
    expect(metadata.asn).toBe(24940);
  });
});
