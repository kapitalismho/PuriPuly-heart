import type { Context } from 'hono';
import { describe, expect, it } from 'vitest';

import { extractRequestNetworkMetadata, resolveRequestNetworkIdentitySecrets } from '../src/abuse-controls';
import {
  deriveStableNetworkIdentityDigest,
  normalizeNetworkIdentityIp,
  resolveNetworkIdentitySecrets,
  resolveRequestNetworkIdentity,
} from '../src/network-identity';
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

  it('canonicalizes IPv4 and IPv6 spellings to one normalized form', () => {
    expect(normalizeNetworkIdentityIp('203.0.113.42')).toBe('203.0.113.42');
    expect(normalizeNetworkIdentityIp('203.000.113.042')).toBe('203.0.113.42');
    expect(normalizeNetworkIdentityIp('2001:DB8::1')).toBe('2001:db8:0:0:0:0:0:1');
    expect(normalizeNetworkIdentityIp('2001:0db8:0000:0000:0000:0000:0000:0001')).toBe(
      '2001:db8:0:0:0:0:0:1',
    );
    expect(normalizeNetworkIdentityIp('::ffff:203.0.113.42')).toBe('203.0.113.42');
    expect(normalizeNetworkIdentityIp(' ::FFFF:203.0.113.42 ')).toBe('203.0.113.42');
    expect(normalizeNetworkIdentityIp('not-an-ip')).toBeNull();
    expect(normalizeNetworkIdentityIp('999.0.113.42')).toBeNull();
    expect(normalizeNetworkIdentityIp('2001:db8:::1')).toBeNull();
  });

  it('derives identical digests for equivalent IP spellings', async () => {
    const env = createTestBrokerEnv();
    const secrets = resolveNetworkIdentitySecrets(env)!;
    const now = new Date('2026-04-08T06:00:00.000Z');
    const lower = await resolveRequestNetworkIdentity('2001:db8::1', secrets, now);
    const upper = await resolveRequestNetworkIdentity('2001:DB8::1', secrets, now);
    expect(upper?.digest).toBe(lower?.digest);
    const padded = await resolveRequestNetworkIdentity('203.000.113.042', secrets, now);
    const plain = await resolveRequestNetworkIdentity('203.0.113.42', secrets, now);
    expect(padded?.digest).toBe(plain?.digest);
    const mapped = await resolveRequestNetworkIdentity('::ffff:203.0.113.42', secrets, now);
    expect(mapped?.digest).toBe(plain?.digest);
  });

  it('stamps previous-secret digests with the previous version and drops them after removal', async () => {
    const env = createTestBrokerEnv();
    env.NETWORK_IDENTITY_HMAC_SECRET = 'new-secret';
    env.NETWORK_IDENTITY_HMAC_SECRET_PREVIOUS = 'old-secret';
    (env as unknown as Record<string, unknown>).NETWORK_IDENTITY_HMAC_KEY_VERSION = '2';
    const secrets = resolveNetworkIdentitySecrets(env)!;
    expect(secrets).toMatchObject({ currentVersion: 2 });

    const digests = await deriveStableNetworkIdentityDigest(secrets, '203.0.113.42', 'ip');
    expect(digests).toEqual([
      expect.objectContaining({ keyVersion: 2 }),
      expect.objectContaining({ keyVersion: 1 }),
    ]);
    expect(digests[0]?.digest).toMatch(/^[a-f0-9]{64}$/u);
    expect(digests[1]?.digest).toMatch(/^[a-f0-9]{64}$/u);
    expect(digests[0]?.digest).not.toBe(digests[1]?.digest);

    const identity = await resolveRequestNetworkIdentity(
      '203.0.113.42',
      secrets,
      new Date('2026-04-08T06:00:00.000Z'),
    );
    expect(identity?.keyVersion).toBe(2);

    delete (env as unknown as Record<string, unknown>).NETWORK_IDENTITY_HMAC_SECRET_PREVIOUS;
    const rotated = resolveNetworkIdentitySecrets(env)!;
    const after = await deriveStableNetworkIdentityDigest(rotated, '203.0.113.42', 'ip');
    expect(after).toEqual([expect.objectContaining({ keyVersion: 2 })]);
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
