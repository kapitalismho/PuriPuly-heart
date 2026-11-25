import { describe, expect, it } from 'vitest';

import broker from '../../broker/src/index';
import relay from '../src/index';

const services = [
  {
    name: 'Broker',
    origins: ['https://puripuly-broker.salee.sarl', 'https://puripuly-heart-broker.kapitalismho.workers.dev'],
    purpose: /authenticates PuriPuly Heart installations.*issues the credentials/,
    fetch: broker.fetch,
  },
  {
    name: 'Agent Relay',
    origins: ['https://puripuly-relay.salee.sarl', 'https://puripuly-heart-agent-relay.kapitalismho.workers.dev'],
    purpose: /sends requests from the Puri assistant.*streams replies/,
    fetch: relay.fetch,
  },
];

const noBindings = new Proxy({}, {
  get(_target, key) { throw new Error(`The service page read an environment binding: ${String(key)}`); },
});

for (const service of services) {
  describe(`${service.name} public service page`, () => {
    it.each(service.origins)('serves static HTML and a bodyless HEAD at %s', async (origin) => {
      const response = await service.fetch(new Request(`${origin}/`), noBindings);
      expect(response.status).toBe(200);
      expect(response.headers.get('Content-Type')).toBe('text/html; charset=utf-8');
      expect(response.headers.get('Content-Security-Policy')).toContain("default-src 'none'");
      expect(response.headers.has('Set-Cookie')).toBe(false);
      const html = await response.text();
      expect(html).toContain(`<title>PuriPuly Heart — ${service.name}</title>`);
      expect(html).toContain(`<h1>${service.name}</h1>`);
      expect(html).toMatch(service.purpose);
      expect(html).toContain('an open-source desktop app for real-time speech recognition, translation, and captions');
      expect([...html.matchAll(/<a href="([^"]+)">([^<]+)<\/a>/g)].map((match) => match.slice(1))).toEqual([
        ['https://github.com/kapitalismho/PuriPuly-heart', 'Official repository'],
        ['https://github.com/kapitalismho/PuriPuly-heart/blob/dev/README.md#usage', 'Documentation'],
        ['https://github.com/kapitalismho/PuriPuly-heart/blob/dev/PRIVACY.md', 'Privacy policy'],
        ['https://github.com/kapitalismho/PuriPuly-heart/issues', 'Support'],
      ]);
      expect(html).not.toMatch(/<(?:script|form|iframe|img|link)\b/i);

      const head = await service.fetch(new Request(`${origin}/`, { method: 'HEAD' }), noBindings);
      expect(head.status).toBe(response.status);
      expect([...head.headers]).toEqual([...response.headers]);
      expect(head.body).toBeNull();
    });

    it('preserves not-found responses for other paths, HTTPS methods and plain HTTP', async () => {
      const [origin] = service.origins;
      const unknown = await service.fetch(new Request(`${origin}/not-an-endpoint`), noBindings);
      expect(unknown.status).toBe(404);
      const missingBody = await unknown.text();
      const http = origin.replace('https:', 'http:');
      const cases = [
        new Request(`${origin}/`, { method: 'POST' }),
        new Request(`${http}/`, { headers: { 'X-Forwarded-Proto': 'https' } }),
        new Request(`${http}/`, { method: 'HEAD' }),
      ];
      for (const request of cases) {
        const response = await service.fetch(request, noBindings);
        expect(response.status).toBe(404);
        expect(response.headers.get('Content-Type')).toBe(unknown.headers.get('Content-Type'));
        expect(response.headers.has('Location')).toBe(false);
        if (request.method !== 'HEAD') {
          expect(await response.text()).toBe(missingBody);
        }
      }
    });
  });
}
