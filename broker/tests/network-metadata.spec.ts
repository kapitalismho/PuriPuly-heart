import type { Context } from 'hono';
import { describe, expect, it } from 'vitest';

import { extractRequestNetworkMetadata } from '../src/abuse-controls';
import type { BrokerEnv } from '../src/contract';
import { createTestBrokerEnv } from './test-support/sqlite-d1';

function createContextWithRequest(request: Request): Context<BrokerEnv> {
  return {
    req: {
      raw: request,
      header: (name: string) => request.headers.get(name) ?? undefined,
    },
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

    const metadata = await extractRequestNetworkMetadata(
      createContextWithRequest(request),
      env.BROKER_DB,
    );

    expect(metadata.asn).toBeNull();
  });

  it('uses ASN from Cloudflare request metadata supplied by the test harness', async () => {
    const env = createTestBrokerEnv();
    const request = requestWithCloudflareMetadata({ asn: 24940 });

    const metadata = await extractRequestNetworkMetadata(
      createContextWithRequest(request),
      env.BROKER_DB,
    );

    expect(metadata.asn).toBe(24940);
  });

  it('does not derive a risk label from HTTP or TLS metadata', async () => {
    const env = createTestBrokerEnv();
    const request = requestWithCloudflareMetadata({
      httpProtocol: 'HTTP/1.1',
      tlsVersion: 'TLSv1.0',
      tlsCipher: 'legacy-cipher',
    });

    const metadata = await extractRequestNetworkMetadata(
      createContextWithRequest(request),
      env.BROKER_DB,
    );

    expect(metadata.riskLabel).toBeNull();
  });
});
