import assert from 'node:assert/strict';

const account = 'a'.repeat(32);
const zone = 'b'.repeat(32);

export function preflightResponse(input, init, { worker, hostname, mode }) {
  const url = new URL(input instanceof Request ? input.url : String(input));
  if (url.origin !== 'https://api.cloudflare.com') return;
  const base = `/client/v4/accounts/${account}/workers`;
  const domain = { id: 'synthetic-domain', hostname, service: worker, zone_id: zone, environment: 'production' };
  const record = { name: hostname, type: 'AAAA', content: '100::', proxied: true,
    meta: { read_only: true, origin_worker_id: 'synthetic-worker-origin' } };
  let result;
  let resultInfo;
  if (url.href === `https://api.cloudflare.com${base}/subdomain`) {
    result = { subdomain: mode === 'wrong-account' ? 'other-account' : 'kapitalismho' };
  } else if (url.href === `https://api.cloudflare.com${base}/scripts/${worker}/subdomain`) {
    if (mode === 'missing-worker') return new Response('', { status: 404 });
    result = { enabled: true, previews_enabled: false };
  } else if (url.href === `https://api.cloudflare.com/client/v4/zones?name=salee.sarl&account.id=${account}&match=all`) {
    result = [{ id: zone, name: 'salee.sarl', status: mode === 'inactive-zone' ? 'pending' : 'active',
      account: { id: mode === 'foreign-zone' ? 'c'.repeat(32) : account } }];
  } else if (url.href === `https://api.cloudflare.com${base}/domains?hostname=${hostname}`) {
    if (mode === 'domain-api-error') return Response.json({ success: false, errors: [{ message: 'private-api-sentinel' }] });
    result = mode === 'foreign-domain' ? [{ ...domain, service: 'other-worker' }]
      : ['managed-domain', 'extra-dns', 'missing-managed-dns'].includes(mode) ? [domain] : [];
  } else if (url.href === `https://api.cloudflare.com${base}/domains?service=${worker}`) {
    result = mode === 'extra-domain' ? [{ ...domain, hostname: 'keep.salee.sarl' }]
      : ['managed-domain', 'extra-dns', 'missing-managed-dns'].includes(mode) ? [domain] : [];
    if (mode === 'partial-domain-list') resultInfo = { count: 0, page: 1, total_pages: 2, total_count: 20 };
  } else if (url.href === `https://api.cloudflare.com/client/v4/zones/${zone}/dns_records?name.exact=${hostname}`) {
    const conflict = { ...record, type: 'CNAME', content: 'another.example', meta: {} };
    result = mode === 'dns-conflict' ? [conflict] : mode === 'extra-dns' ? [record, conflict]
      : ['managed-domain', 'orphaned-managed-dns'].includes(mode) ? [record] : [];
  } else {
    return;
  }
  assert.equal(init?.method, 'GET');
  assert.equal(init?.redirect, 'error');
  assert(init?.signal instanceof AbortSignal);
  assert.equal(new Headers(init.headers).get('authorization'), 'Bearer synthetic-cloudflare-token');
  return Response.json({ success: true, result, ...(resultInfo ? { result_info: resultInfo } : {}) });
}
