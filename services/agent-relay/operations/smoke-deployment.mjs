import { createHash, createHmac, randomBytes } from 'node:crypto';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { setTimeout as delay } from 'node:timers/promises';
import { relayDeployment, requireLegacyOrigin } from '../../worker-deployment.mjs';
import { deploymentOrigin } from './prepare-deployment.mjs';

const readinessAttempts = 12;
const readinessDelayMs = 5000;
const notReadyStatuses = new Set([502, 503, 504, 525]);
const notReadyCodes = new Set([
  'EAI_AGAIN', 'ENOTFOUND', 'ECONNREFUSED', 'ECONNRESET', 'ETIMEDOUT', 'UND_ERR_CONNECT_TIMEOUT', 'UND_ERR_SOCKET',
  'ERR_SSL_TLSV1_ALERT_INTERNAL_ERROR', 'ERR_SSL_TLSV1_UNRECOGNIZED_NAME', 'ERR_SSL_SSLV3_ALERT_HANDSHAKE_FAILURE',
]);

async function requestProbe(target, path, options, fetcher, waitForReadiness) {
  const attempts = waitForReadiness ? readinessAttempts : 1;
  for (let attempt = 1; attempt <= attempts; attempt++) {
    let response;
    let pending;
    try {
      response = await fetcher(`${target}${path}`, {
        ...options, redirect: 'manual', signal: AbortSignal.timeout(waitForReadiness ? 5000 : 10000),
      });
    } catch (error) {
      if (!waitForReadiness || (error?.name !== 'TimeoutError' && !notReadyCodes.has(error?.cause?.code))) {
        throw new Error(`Relay smoke request failed for ${path}.`);
      }
      pending = 'connection not ready';
    }
    if (response) {
      if (response.status >= 300 && response.status < 400) {
        await response.body?.cancel().catch(() => {});
        throw new Error(`Relay smoke rejected redirect for ${path} (HTTP ${response.status}).`);
      }
      if (!waitForReadiness || !notReadyStatuses.has(response.status)) return response;
      pending = `HTTP ${response.status}`;
      await response.body?.cancel().catch(() => {});
    }
    if (attempt === attempts) throw new Error(`Relay smoke readiness did not settle for ${path} after ${attempts} attempts (${pending}).`);
    await delay(readinessDelayMs);
  }
}

export async function smokeDeployment({ origin, secret, modelEnabled, supportEnabled, allowLoopback = false, fetcher = fetch }) {
  const local = allowLoopback && /^http:\/\/127\.0\.0\.1:[1-9][0-9]*$/.test(origin ?? '');
  const target = local ? new URL(origin).origin : deploymentOrigin(origin);
  if (typeof secret !== 'string' || !secret.trim() || Buffer.byteLength(secret, 'utf8') < 32
    || typeof modelEnabled !== 'boolean' || typeof supportEnabled !== 'boolean') {
    throw new Error('Smoke requires a broker-matching token secret and explicit service flags.');
  }
  let probes = 0;
  async function probe(path, options, status, code, waitForReadiness = false) {
    const response = await requestProbe(target, path, options, fetcher, waitForReadiness);
    let value;
    try {
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
  await probe('/deployment-smoke-not-an-endpoint', { method: 'GET' }, 404, 'not_found', target === relayDeployment.origin);
  const now = Math.floor(Date.now() / 1000);
  const encoded = [{ alg: 'HS256', typ: 'JWT' }, { iss: 'puripuly-heart-broker', aud: 'puripuly-agent-relay',
    sub: createHash('sha256').update(randomBytes(32)).digest('hex'), iat: now, exp: now + 60 }]
    .map((value) => Buffer.from(JSON.stringify(value)).toString('base64url')).join('.');
  const token = `${encoded}.${createHmac('sha256', secret).update(encoded).digest('base64url')}`;
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
    const legacyOrigin = requireLegacyOrigin(process.env, relayDeployment);
    const origins = [relayDeployment.origin, legacyOrigin];
    const checks = await Promise.allSettled(origins.map(async (origin) => {
      const result = await smokeDeployment({ origin,
        secret: process.env.PURI_TOKEN_SECRET_PRODUCTION, modelEnabled: process.env.AGENT_SERVICE_ENABLED === 'true',
        supportEnabled: process.env.SUPPORT_REPORTS_ENABLED === 'true' });
      console.log(JSON.stringify({ origin, ...result }));
    }));
    for (const [index, result] of checks.entries()) {
      if (result.status === 'rejected') {
        console.error(`${origins[index]}: ${result.reason instanceof Error ? result.reason.message : 'Relay deployment smoke failed.'}`);
        process.exitCode = 1;
      }
    }
  } catch (error) {
    console.error(error instanceof Error ? error.message : 'Relay deployment smoke failed.');
    process.exitCode = 1;
  }
}
