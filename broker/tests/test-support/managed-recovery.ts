import worker from '../../src/index';
import {
  buildManagedOperationId,
  buildManagedOperationResumeToken,
  createManagedOperation,
  getManagedOperation,
  hashManagedOperationResumeToken,
  markAttemptUnknown,
  providerKeyNameForOperationAttempt,
  recordAttemptCredential,
  saveOperationIssuanceContext,
  startManagedOperationAttempt,
} from '../../src/managed-operation';
import type { TestBrokerEnv } from './sqlite-d1';
export interface Deferred {
  promise: Promise<void>;
  resolve: () => void;
}
export interface ControlledProviderKey {
  name: string;
  hash: string;
  limit: number;
  disabled: boolean;
  expiresAt?: string;
}

export function deferred(): Deferred {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => { resolve = done; });
  return { promise, resolve };
}

export function controlledProvider() {
  const keys = new Map<string, ControlledProviderKey>();
  const calls: Array<{ method: string; url: string; body: Record<string, unknown> | null }> = [];
  let creates = 0;
  let failCleanup = false;
  let malformedList = false;
  let cleanupGate: Deferred | null = null;
  let cleanupEntered: Deferred | null = null;
  let createGate: Deferred | null = null;
  let createEntered: Deferred | null = null;
  let gateAfterCreate = false;
  const fetchImpl: typeof fetch = async (input, init) => {
    const url = new URL(String(input));
    const method = init?.method ?? 'GET';
    const body = init?.body ? JSON.parse(String(init.body)) as Record<string, unknown> : null;
    calls.push({ method, url: url.toString(), body });
    if (url.hostname === 'discord.test') return new Response(null, { status: 204 });
    if (url.hostname !== 'openrouter.ai') throw new Error(`Uncontrolled network boundary: ${url.origin}`);
    if (url.pathname.includes('/guardrails/') && method === 'POST') return Response.json({ assigned_count: 1 });
    if (url.pathname === '/api/v1/keys' && method === 'POST') {
      const gate = createGate;
      const entered = createEntered;
      const afterCreate = gateAfterCreate;
      createGate = null;
      if (gate && !afterCreate) { entered?.resolve(); await gate.promise; }
      const hash = `controlled-key-${++creates}`;
      const key = { name: String(body?.name), hash, limit: Number(body?.limit), disabled: false };
      keys.set(hash, key);
      const rawKey = `sk-controlled-${creates}`;
      if (gate && afterCreate) { entered?.resolve(); await gate.promise; }
      return Response.json({ key: rawKey, data: key }, { status: 201 });
    }
    if (url.pathname === '/api/v1/keys' && method === 'GET') {
      if (malformedList) return Response.json({ data: {} });
      const offset = Number(url.searchParams.get('offset') ?? 0);
      const visible = [...keys.values()].filter((key) =>
        (!key.disabled || url.searchParams.get('include_disabled') === 'true') &&
        (!key.expiresAt || Date.parse(key.expiresAt) > Date.now() || url.searchParams.get('include_expired') === 'true'));
      return Response.json({ data: visible.slice(offset, offset + 100) });
    }
    const hash = decodeURIComponent(url.pathname.split('/').at(-1) ?? '');
    const key = keys.get(hash);
    if (method === 'GET') return key ? Response.json({ data: key }) : Response.json({ error: { message: 'not found' } }, { status: 404 });
    if (method === 'PATCH' || method === 'DELETE') {
      cleanupEntered?.resolve();
      await cleanupGate?.promise;
      if (failCleanup) return Response.json({ error: { message: 'controlled unavailable' } }, { status: 503 });
      if (!key) return Response.json({ error: { message: 'not found' } }, { status: 404 });
      if (method === 'PATCH') {
        if (body?.disabled !== true || Object.hasOwn(body, 'limit')) throw new Error('Unexpected provider allowance mutation');
        key.disabled = true;
        return Response.json({ data: key });
      }
      keys.delete(hash);
      return new Response(null, { status: 204 });
    }
    throw new Error(`Unexpected controlled provider request: ${method} ${url}`);
  };
  return {
    keys, calls, fetchImpl,
    get creates() { return creates; },
    set failCleanup(value: boolean) { failCleanup = value; },
    set malformedList(value: boolean) { malformedList = value; },
    gateCreate(afterCreate = false) {
      createGate = deferred();
      createEntered = deferred();
      gateAfterCreate = afterCreate;
      return { entered: createEntered.promise, release: createGate.resolve };
    },
    gateCleanup() {
      cleanupGate = deferred();
      cleanupEntered = deferred();
      return { entered: cleanupEntered.promise, release: cleanupGate.resolve };
    },
  };
}

export async function seedRecoverableOperation(env: TestBrokerEnv, source: 'discord' | 'qq', now = new Date(), installationId: string | null = `controlled-${source}`) {
  const operationId = buildManagedOperationId();
  const resumeToken = buildManagedOperationResumeToken();
  const subjectRef = source === 'discord' ? `ph-discord-user-v1_${operationId.slice(-24)}` : `ph-qq-subject-v1_${operationId.slice(-24)}`;
  const { operation } = await createManagedOperation(env.BROKER_DB, {
    operationId, resumeTokenHash: await hashManagedOperationResumeToken(resumeToken),
    issueSource: source, subjectRef, installationId, devicePublicKey: source === 'discord' ? `device-${operationId}` : null, now,
  });
  if (source === 'discord') await saveOperationIssuanceContext(env.BROKER_DB, operationId, { hardwareHash: `hardware-${operationId}`, hardwareHashSaltVersion: 7, appVersion: '1.0.0' }, now);
  const started = await startManagedOperationAttempt(env.BROKER_DB, operation, now);
  if (!started.ok) throw new Error('Controlled operation admission failed');
  const hash = `orphan-${operationId}`;
  await recordAttemptCredential(env.BROKER_DB, operationId, 1, hash, now);
  await markAttemptUnknown(env.BROKER_DB, operationId, 1, now);
  return {
    operationId, resumeToken, subjectRef, installationId, hash,
    keyName: providerKeyNameForOperationAttempt(operationId, source, 1),
    operation: (await getManagedOperation(env.BROKER_DB, operationId))!,
    authority: { operation_id: operationId, resume_token: resumeToken, installation_id: installationId },
  };
}

export async function workerPost(env: TestBrokerEnv, path: string, body: object) {
  return worker.fetch(new Request(`http://broker.test${path}`, {
    method: 'POST', headers: { 'content-type': 'application/json', 'cf-connecting-ip': '203.0.113.95' }, body: JSON.stringify(body),
  }), env);
}

export const RESUME_PATH = '/v1/providers/openrouter/managed-operation/resume';
export const STATUS_PATH = '/v1/providers/openrouter/managed-operation/status';
export const ACK_PATH = '/v1/providers/openrouter/managed-key-delivery/ack';
