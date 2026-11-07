const networkFetch = globalThis.fetch;
const origin = process.env.AGENT_RELAY_CANONICAL_WORKERS_DEV_URL;
const storage = `https://api.cloudflare.com/client/v4/accounts/${process.env.CLOUDFLARE_ACCOUNT_ID}/r2/buckets/puripuly-heart-support-reports/`;

globalThis.fetch = (input, init) => {
  const url = new URL(input instanceof Request ? input.url : String(input));
  if (url.origin === origin) {
    return networkFetch(`${process.env.LOCAL_HTTP_URL}${url.pathname}${url.search}`, init);
  }
  if (url.href.startsWith(storage)) {
    return networkFetch(`${process.env.LOCAL_HTTP_URL}/cloudflare/${url.href.slice(storage.length)}`, init);
  }
  throw new Error(`Unexpected external request: ${url.origin}`);
};
