import { DatabaseSync } from 'node:sqlite';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { appTools } from '../src/app-tools';
import relay from '../src/index';
import { requestDeadlineMs, requestLeaseMs } from '../src/lifetime';
import { maxSupportBytes } from '../src/support';
import { accountingDatabase, installationHash, signToken, tokenSecret } from './fixtures';

const ticket = 'PP-0123456789ABCDEF';
const bytes = new Uint8Array([31, 139, 8, 0, 1, 2, 3]);
let sqlite: DatabaseSync;
let db: D1Database;
let objects: Map<string, { bytes: Uint8Array; customMetadata: Record<string, string> }>;
let bucket: R2Bucket;
let rejectStorage: boolean;

function request(body: BodyInit = bytes, headers: Record<string, string> = {}, sub = installationHash): Request {
  return new Request('https://relay.example/v1/support', {
    method: 'POST', headers: { authorization: `Bearer ${signToken({ sub })}`, 'content-type': 'application/gzip',
      'x-agent-protocol': '3', 'x-support-ticket': ticket, 'x-app-version': '0.0.0-probe',
      'x-diagnostic-codes': '["unclassified_failure","stt.auth"]', ...headers }, body,
    ...((body instanceof ReadableStream) ? { duplex: 'half' } : {}),
  });
}

function env() {
  return { PURI_TOKEN_SECRET: tokenSecret, AGENT_USAGE_DB: db, SUPPORT_REPORTS: bucket,
    SUPPORT_REPORTS_ENABLED: 'true', AGENT_SERVICE_ENABLED: 'false',
    DISCORD_OPERATIONS_WEBHOOK_URL: 'https://discord.example/synthetic-hook' };
}

function reports(): number {
  const row = sqlite.prepare('SELECT COALESCE(SUM(reports), 0) AS n FROM support_daily').get();
  return Number(row?.n);
}

beforeEach(() => {
  ({ sqlite, db } = accountingDatabase());
  objects = new Map();
  rejectStorage = false;
  bucket = {
    head: vi.fn(async (key: string) => objects.get(key) ?? null),
    put: vi.fn(async (key: string, body: Blob, options: R2PutOptions) => {
      if (rejectStorage) {
        rejectStorage = false;
        return null;
      }
      const received = new Uint8Array(await new Response(body).arrayBuffer());
      if (options.onlyIf && 'etagDoesNotMatch' in options.onlyIf
        && options.onlyIf.etagDoesNotMatch === '*' && objects.has(key)) return null;
      const object = { bytes: received, customMetadata: options.customMetadata ?? {} };
      objects.set(key, object);
      return object;
    }),
  } as unknown as R2Bucket;
  vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(null, { status: 204 }));
});

afterEach(() => {
  sqlite.close();
  vi.restoreAllMocks();
  vi.useRealTimers();
});

describe('user-authorized opaque support storage', () => {
  it('stores exact bytes with a receipt while model service is stopped, unconfigured and exhausted', async () => {
    const day = new Date().toISOString().slice(0, 10);
    sqlite.prepare('INSERT INTO usage_daily (installation_hash, utc_date, decisions_calls, responses_calls) VALUES (?, ?, 50, 50)')
      .run(installationHash, day);
    const seed = sqlite.prepare('INSERT INTO prompt_turns (installation_hash, turn_id, utc_date) VALUES (?, ?, ?)');
    for (let index = 0; index < 30; index++) {
      seed.run(installationHash, index.toString(16).padStart(32, '0'), day);
    }
    const turnsBefore = sqlite.prepare('SELECT * FROM prompt_turns').all();
    const before = sqlite.prepare('SELECT * FROM usage_daily').all();
    const response = await relay.fetch(request(), env());
    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({ ticket });
    expect(response.headers.get('cache-control')).toBe('no-store');
    expect(objects.get(`support/${ticket}.tar.gz`)?.bytes).toEqual(bytes);
    expect(reports()).toBe(1);
    expect(sqlite.prepare('SELECT * FROM usage_daily').all()).toEqual(before);
    expect(sqlite.prepare('SELECT * FROM prompt_turns').all()).toEqual(turnsBefore);
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(fetch).toHaveBeenCalledWith('https://discord.example/synthetic-hook', expect.objectContaining({ redirect: 'manual' }));
    const sent = JSON.parse(vi.mocked(fetch).mock.calls[0][1]?.body as string);
    expect(sent).toEqual({ allowed_mentions: { parse: [] }, embeds: [{ fields: [
      { name: 'Ticket', value: ticket }, { name: 'App version', value: '0.0.0-probe' },
      { name: 'Compressed bytes', value: '7' }, { name: 'Diagnostic codes', value: 'unclassified_failure, stt.auth' },
    ] }] });
  });

  it('returns the first receipt for same-installation ticket retries without overwriting or charging again', async () => {
    expect((await relay.fetch(request(), env())).status).toBe(200);
    expect((await relay.fetch(request(), env())).status).toBe(200);
    sqlite.exec('UPDATE support_daily SET reports = 3');
    expect((await relay.fetch(request(), env())).status).toBe(200);
    expect(reports()).toBe(3);
    expect(bucket.put).toHaveBeenCalledTimes(1);
    expect(fetch).toHaveBeenCalledTimes(1);
    const changed = await relay.fetch(request(new Uint8Array([1, 2, 3])), env());
    expect(changed.status).toBe(200);
    expect(await changed.json()).toEqual({ ticket });
    expect(objects.get(`support/${ticket}.tar.gz`)?.bytes).toEqual(bytes);
    expect(reports()).toBe(3);
    const other = await relay.fetch(request(bytes, {}, 'b'.repeat(64)), env());
    expect(other.status).toBe(409);
    expect(await other.text()).not.toContain(ticket);
    expect(bucket.put).toHaveBeenCalledTimes(1);
  });

  it('admits three sequential reports and resets on the next server UTC date', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-10-30T23:59:59Z'));
    const statuses = [];
    for (const n of [1, 2, 3, 4]) {
      statuses.push((await relay.fetch(request(bytes,
        { 'x-support-ticket': `PP-${String(n).padStart(16, '0')}` }), env())).status);
    }
    expect(statuses).toEqual([200, 200, 200, 429]);
    expect(reports()).toBe(3);
    expect(objects.size).toBe(3);
    vi.setSystemTime(new Date('2026-10-31T00:00:00Z'));
    expect((await relay.fetch(request(), env())).status).toBe(200);
    expect(reports()).toBe(4);
    expect(sqlite.prepare('SELECT utc_date, reports FROM support_daily ORDER BY utc_date').all())
      .toEqual([{ utc_date: '2026-10-30', reports: 3 }, { utc_date: '2026-10-31', reports: 1 }]);
  });

  it('compensates definitive create-only rejection without overwriting or notifying', async () => {
    rejectStorage = true;
    expect((await relay.fetch(request(), env())).status).toBe(409);
    expect(reports()).toBe(0);
    expect(objects.size).toBe(0);
    expect(fetch).not.toHaveBeenCalled();
    expect((await relay.fetch(request(), env())).status).toBe(200);
    expect(reports()).toBe(1);
  });

  it('serializes concurrent different-content uploads and returns the stored receipt on retry', async () => {
    const storing = Promise.withResolvers<void>();
    const finish = Promise.withResolvers<void>();
    const put = vi.mocked(bucket.put).getMockImplementation()!;
    vi.mocked(bucket.put).mockImplementationOnce(async (...args) => {
      storing.resolve();
      await finish.promise;
      return put(...args);
    });
    const first = relay.fetch(request(), env());
    await storing.promise;
    const upload = request(new Uint8Array([4, 5, 6]));
    const busy = await relay.fetch(upload, env());
    expect(busy.status).toBe(429);
    expect(await busy.json()).toMatchObject({ error: { code: 'service_busy', retry_after_ms: 1000 } });
    expect(busy.headers.get('retry-after')).toBe('1');
    expect(upload.bodyUsed).toBe(false);
    finish.resolve();
    expect((await first).status).toBe(200);
    expect((await relay.fetch(request(), env())).status).toBe(200);
    expect(reports()).toBe(1);
    expect(objects.size).toBe(1);
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(sqlite.prepare('SELECT * FROM request_slots').all()).toEqual([]);
  });

  it('retains a conservative slot on uncertain storage and never logs private failures', async () => {
    const logs = [vi.spyOn(console, 'log'), vi.spyOn(console, 'warn'), vi.spyOn(console, 'error')];
    vi.mocked(bucket.put).mockRejectedValueOnce(new Error('synthetic private content'));
    const response = await relay.fetch(request(), env());
    expect(response.status).toBe(503);
    expect(await response.text()).toContain('may consume a daily slot');
    expect(reports()).toBe(1);
    expect(fetch).not.toHaveBeenCalled();
    for (const log of logs) expect(log).not.toHaveBeenCalled();
  });

  it('preserves stored success when notification rejects or returns a failure', async () => {
    vi.mocked(fetch).mockRejectedValueOnce(new Error('synthetic webhook failure'));
    expect((await relay.fetch(request(), env())).status).toBe(200);
    vi.mocked(fetch).mockResolvedValueOnce(new Response(null, { status: 500 }));
    expect((await relay.fetch(request(bytes, { 'x-support-ticket': 'PP-1111111111111111' }), env())).status).toBe(200);
    expect(reports()).toBe(2);
    expect(objects.size).toBe(2);
  });

  it('stores fragmented uploads using actual bytes rather than the declared length', async () => {
    const body = new ReadableStream<Uint8Array>({ start(controller) {
      controller.enqueue(bytes.subarray(0, 1));
      controller.enqueue(bytes.subarray(1, 4));
      controller.enqueue(bytes.subarray(4));
      controller.close();
    } });
    expect((await relay.fetch(request(body, { 'content-length': '1' }), env())).status).toBe(200);
    expect(objects.get(`support/${ticket}.tar.gz`)?.bytes).toEqual(bytes);
    expect(reports()).toBe(1);
  });

  it('enforces inclusive compressed-byte limit from the stream rather than declared length', async () => {
    const full = new Uint8Array(maxSupportBytes);
    expect((await relay.fetch(request(full), env())).status).toBe(200);
    expect(objects.get(`support/${ticket}.tar.gz`)?.bytes.byteLength).toBe(maxSupportBytes);
    const cancel = vi.fn();
    const stream = new ReadableStream<Uint8Array>({ start(controller) {
      controller.enqueue(full);
      controller.enqueue(new Uint8Array([0]));
    }, cancel });
    expect((await relay.fetch(request(stream, { 'content-length': '1', 'x-support-ticket': 'PP-1111111111111111' }), env())).status).toBe(413);
    expect(cancel).toHaveBeenCalledTimes(1);
    expect(reports()).toBe(1);
    expect(bucket.put).toHaveBeenCalledTimes(1);
    const early = request(bytes, { 'content-length': String(maxSupportBytes + 1) });
    expect((await relay.fetch(early, env())).status).toBe(413);
    expect(early.bodyUsed).toBe(false);
  });

  it('rejects unsafe metadata, unsupported protocol, coding and empty bodies before accounting', async () => {
    const invalid: Record<string, string>[] = [
      { 'x-agent-protocol': '1' }, { 'x-support-ticket': 'PP-0123456789abcdef' }, { 'x-support-ticket': '../private' },
      { 'x-app-version': 'a'.repeat(65) }, { 'x-app-version': '@everyone' },
      { 'x-diagnostic-codes': '["private message"]' }, { 'x-diagnostic-codes': JSON.stringify(['a'.repeat(97)]) },
      { 'x-diagnostic-codes': JSON.stringify(Array(11).fill('stt.auth')) }, { 'x-diagnostic-codes': '{}' },
      { 'content-type': 'application/json' }, { 'content-encoding': 'gzip' },
    ];
    for (const headers of invalid) expect((await relay.fetch(request(bytes, headers), env())).status).toBe(400);
    expect((await relay.fetch(request(new Uint8Array()), env())).status).toBe(400);
    expect(reports()).toBe(0);
    expect(bucket.head).not.toHaveBeenCalled();
    expect(fetch).not.toHaveBeenCalled();
  });

  it('fails closed for storage and accounting outages before any notification', async () => {
    vi.mocked(bucket.head).mockRejectedValueOnce(new Error('synthetic private bucket failure'));
    expect((await relay.fetch(request(), env())).status).toBe(503);
    expect(reports()).toBe(0);
    expect(bucket.put).not.toHaveBeenCalled();
    const prepare = vi.spyOn(db, 'prepare').mockImplementationOnce(() => { throw new Error('synthetic private D1 failure'); });
    expect((await relay.fetch(request(), env())).status).toBe(503);
    expect(prepare).toHaveBeenCalledTimes(1);
    expect(bucket.put).not.toHaveBeenCalled();
    expect(fetch).not.toHaveBeenCalled();
  });

  it('holds a single installation upload slot before reading any report bytes without blocking other installations', async () => {
    const reading = Promise.withResolvers<void>();
    let controller!: ReadableStreamDefaultController<Uint8Array>;
    const body = new ReadableStream<Uint8Array>({
      start(value) { controller = value; },
      pull() { reading.resolve(); },
    }, { highWaterMark: 0 });
    const pending = relay.fetch(request(body), env());
    await reading.promise;
    const rejectedBody = request();
    const rejected = await relay.fetch(rejectedBody, env());
    expect(rejected.status).toBe(429);
    expect(await rejected.json()).toMatchObject({ error: { code: 'service_busy' } });
    expect(rejectedBody.bodyUsed).toBe(false);
    expect(reports()).toBe(0);
    expect((await relay.fetch(request(bytes, { 'x-support-ticket': 'PP-1111111111111111' }, 'b'.repeat(64)), env())).status).toBe(200);
    controller.enqueue(bytes);
    controller.close();
    expect((await pending).status).toBe(200);
    expect(reports()).toBe(2);
    expect(sqlite.prepare('SELECT * FROM request_slots').all()).toEqual([]);
  });

  it('cancels a stalled upload reader before lease expiry and releases its known-terminated slot in finally', async () => {
    vi.useFakeTimers();
    const started = new Date('2026-11-01T12:00:00Z');
    vi.setSystemTime(started);
    const reading = Promise.withResolvers<void>();
    const cancel = vi.fn();
    const body = new ReadableStream<Uint8Array>({ pull() { reading.resolve(); }, cancel }, { highWaterMark: 0 });
    const pending = relay.fetch(request(body), env());
    await reading.promise;
    await vi.advanceTimersByTimeAsync(requestDeadlineMs - 1);
    expect(cancel).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(1);
    expect((await pending).status).toBe(400);
    expect(cancel).toHaveBeenCalledTimes(1);
    expect(Date.now()).toBeLessThan(started.getTime() + requestLeaseMs);
    expect(sqlite.prepare('SELECT * FROM request_slots').all()).toEqual([]);
    expect(reports()).toBe(0);
    expect(bucket.head).not.toHaveBeenCalled();
    expect((await relay.fetch(request(), env())).status).toBe(200);
  });

  it('retains an uncertain timed-out storage lease until expiry and accepts a later retry without resetting daily accounting', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-11-01T12:00:00Z'));
    const storing = Promise.withResolvers<void>();
    const finish = Promise.withResolvers<R2Object | null>();
    vi.mocked<(...args: Parameters<R2Bucket['put']>) => Promise<R2Object | null>>(bucket.put).mockImplementationOnce(() => {
      storing.resolve();
      return finish.promise;
    });
    const pending = relay.fetch(request(), env());
    await storing.promise;
    await vi.advanceTimersByTimeAsync(requestDeadlineMs);
    expect((await pending).status).toBe(503);
    expect(sqlite.prepare('SELECT COUNT(*) AS count FROM request_slots').get()).toEqual({ count: 1 });
    expect((await relay.fetch(request(), env())).status).toBe(429);
    expect(reports()).toBe(1);
    await vi.advanceTimersByTimeAsync(requestLeaseMs - requestDeadlineMs);
    expect((await relay.fetch(request(), env())).status).toBe(200);
    finish.resolve(null);
    await finish.promise;
    expect(reports()).toBe(2);
    expect(sqlite.prepare('SELECT * FROM request_slots').all()).toEqual([]);
  });

  it('does not let a stale support finally release a replacement ownership ID', async () => {
    const reading = Promise.withResolvers<void>();
    let controller!: ReadableStreamDefaultController<Uint8Array>;
    const body = new ReadableStream<Uint8Array>({
      start(value) { controller = value; }, pull() { reading.resolve(); },
    }, { highWaterMark: 0 });
    const pending = relay.fetch(request(body), env());
    await reading.promise;
    sqlite.prepare('UPDATE request_slots SET owner = ?, expires_at = ? WHERE installation_hash = ? AND slot = 2')
      .run('replacement-owner', Date.now() + requestLeaseMs, installationHash);
    controller.enqueue(bytes);
    controller.close();
    expect((await pending).status).toBe(200);
    expect(sqlite.prepare('SELECT owner FROM request_slots WHERE installation_hash = ? AND slot = 2').get(installationHash))
      .toEqual({ owner: 'replacement-owner' });
    expect((await relay.fetch(request(), env())).status).toBe(429);
  });

  it('publishes a bounded report proposal without a send tool', () => {
    const tool = appTools.find((value) => value.name === 'propose_app');
    expect(tool).toMatchObject({ type: 'function', parameters: { type: 'object', additionalProperties: false,
      properties: { summary: { type: 'string', minLength: 1, maxLength: 2000 } } } });
    const properties = tool?.parameters.properties;
    expect(properties && 'action' in properties ? properties.action?.enum : []).toContain('report');
    expect(appTools.filter((value) => /report|support|send/.test(value.name))).toHaveLength(0);
  });
});
