import { preflightResponse } from '../../../test-support/worker-deployment-port.mjs';

const networkFetch = globalThis.fetch;
const origins = ['https://puripuly-relay.salee.sarl', 'https://puripuly-heart-agent-relay.kapitalismho.workers.dev'];
const storage = `https://api.cloudflare.com/client/v4/accounts/${process.env.CLOUDFLARE_ACCOUNT_ID}/r2/buckets/puripuly-heart-support-reports/`;

globalThis.fetch = async (input, init) => {
  const url = new URL(input instanceof Request ? input.url : String(input));
  if (origins.includes(url.origin)) {
    const headers = new Headers(init?.headers);
    headers.set('X-Deployment-Probe-Origin', url.origin);
    return networkFetch(`${process.env.LOCAL_HTTP_URL}${url.pathname}${url.search}`, { ...init, headers });
  }
  if (url.href.startsWith(storage)) {
    return networkFetch(`${process.env.LOCAL_HTTP_URL}/cloudflare/${url.href.slice(storage.length)}`, init);
  }
  const response = preflightResponse(input, init, {
    worker: 'puripuly-heart-agent-relay', hostname: 'puripuly-relay.salee.sarl', mode: process.env.SCENARIO_MODE,
  });
  if (response) return response;
  throw new Error(`Unexpected external request: ${url.origin}`);
};
