import { createHash, createHmac, randomBytes } from 'node:crypto';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { deploymentOrigin } from './prepare-deployment.mjs';

export async function smokeDeployment({ origin, secret, modelEnabled, supportEnabled, allowLoopback = false, fetcher = fetch }) {
  const local = allowLoopback && /^http:\/\/127\.0\.0\.1:[1-9][0-9]*$/.test(origin ?? '');
  const target = local ? new URL(origin).origin : deploymentOrigin(origin);
  if (typeof secret !== 'string' || !secret.trim() || Buffer.byteLength(secret, 'utf8') < 32
    || typeof modelEnabled !== 'boolean' || typeof supportEnabled !== 'boolean') {
    throw new Error('Smoke requires a broker-matching token secret and explicit service flags.');
  }
  const now = Math.floor(Date.now() / 1000);
  const encoded = [{ alg: 'HS256', typ: 'JWT' }, { iss: 'puripuly-heart-broker', aud: 'puripuly-agent-relay',
    sub: createHash('sha256').update(randomBytes(32)).digest('hex'), iat: now, exp: now + 60 }]
    .map((value) => Buffer.from(JSON.stringify(value)).toString('base64url')).join('.');
  const token = `${encoded}.${createHmac('sha256', secret).update(encoded).digest('base64url')}`;
  let probes = 0;
  async function probe(path, options, status, code) {
    let response;
    let value;
    try {
      response = await fetcher(`${target}${path}`, { ...options, redirect: 'error', signal: AbortSignal.timeout(10000) });
      const reader = response.body?.getReader();
      if (!reader) throw new Error('Missing body.');
      const chunks = [];
      let bytes = 0;
      try {
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          bytes += value.byteLength;
          if (bytes > 4096) throw new Error('Oversized response.');
          chunks.push(value);
        }
      } finally {
        await reader.cancel().catch(() => {});
        reader.releaseLock();
      }
      value = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    } catch {
      throw new Error(`Relay smoke request failed for ${path}.`);
    }
    if (response.status !== status || value?.error?.code !== code || response.headers.get('cache-control') !== 'no-store') {
      throw new Error(`Relay smoke rejection mismatch for ${path}; expected ${status} ${code}.`);
    }
    probes++;
  }
  await probe('/deployment-smoke-not-an-endpoint', { method: 'GET' }, 404, 'not_found');
  for (const path of ['/v1/classify', '/v1/respond', '/v1/support']) {
    await probe(path, { method: 'GET' }, 405, 'method_not_allowed');
    await probe(path, { method: 'POST', headers: { 'content-type': 'application/json' }, body: '{}' }, 401, 'service_authentication_failed');
    await probe(path, { method: 'POST', headers: { authorization: 'Bearer invalid', 'content-type': 'application/json' }, body: '{}' }, 401, 'service_authentication_failed');
    const support = path === '/v1/support';
    const enabled = support ? supportEnabled : modelEnabled;
    await probe(path, { method: 'POST', headers: { authorization: `Bearer ${token}`, 'content-type': 'application/json',
      ...(support ? { 'x-agent-protocol': '0' } : {}) }, body: '{"agent_protocol":0}' },
    enabled ? 400 : 503, enabled ? 'unsupported_agent_protocol' : support ? 'support_unavailable' : 'service_shutdown');
  }
  return { status: 'rejection_smoke_passed', probes, model_requests: 0, support_uploads: 0 };
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  try {
    if (process.argv.length !== 2 || !['true', 'false'].includes(process.env.AGENT_SERVICE_ENABLED)
      || !['true', 'false'].includes(process.env.SUPPORT_REPORTS_ENABLED)) throw new Error('Use explicit deployment flags for smoke.');
    console.log(JSON.stringify(await smokeDeployment({ origin: process.env.AGENT_RELAY_CANONICAL_WORKERS_DEV_URL,
      secret: process.env.PURI_TOKEN_SECRET_PRODUCTION, modelEnabled: process.env.AGENT_SERVICE_ENABLED === 'true',
      supportEnabled: process.env.SUPPORT_REPORTS_ENABLED === 'true' })));
  } catch (error) {
    console.error(error instanceof Error ? error.message : 'Relay deployment smoke failed.');
    process.exitCode = 1;
  }
}
