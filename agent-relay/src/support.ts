import { beforeAbort, requestDeadlineMs, requestLeaseMs, requestLifetime } from './lifetime';

export type SupportEnv = {
  AGENT_USAGE_DB?: D1Database;
  SUPPORT_REPORTS?: R2Bucket;
  SUPPORT_REPORTS_ENABLED?: string;
  DISCORD_OPERATIONS_WEBHOOK_URL?: string;
};

export const maxSupportBytes = 25_000_000;

type Details = { ticket: string; version: string; codes: string[] };
type Bundle = { chunks: Uint8Array<ArrayBuffer>[]; size: number };

function error(status: number, code: string, message: string): Response {
  const busy = code === 'service_busy';
  return Response.json({ error: { code, message, ...(busy ? { retry_after_ms: 1000 } : {}) } }, {
    status, headers: { 'cache-control': 'no-store', ...(busy ? { 'retry-after': '1' } : {}) },
  });
}

function receipt(ticket: string): Response {
  return Response.json({ ticket }, { headers: { 'cache-control': 'no-store' } });
}

function details(request: Request): Details | null {
  const ticket = request.headers.get('x-support-ticket') ?? '';
  const version = request.headers.get('x-app-version') ?? '';
  const rawCodes = request.headers.get('x-diagnostic-codes') ?? '';
  if (!/^PP-[0-9A-F]{16}$/.test(ticket) || !/^[A-Za-z0-9.+_-]{1,64}$/.test(version)
    || rawCodes.length > 1000) return null;
  try {
    const codes: unknown = JSON.parse(rawCodes);
    if (!Array.isArray(codes) || codes.length > 10 || !codes.every((code) =>
      typeof code === 'string' && /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}$/.test(code))) return null;
    return { ticket, version, codes };
  } catch {
    return null;
  }
}

async function readBundle(request: Request, signal: AbortSignal): Promise<Bundle | Response> {
  const declared = request.headers.get('content-length');
  if (declared !== null && /^\d+$/.test(declared) && Number(declared) > maxSupportBytes) {
    return error(413, 'support_too_large', 'Support reports must be at most 25000000 compressed bytes.');
  }
  if (!request.body) return error(400, 'invalid_support_report', 'Send a nonempty gzip support bundle.');
  const reader = request.body.getReader();
  const cancel = () => { void reader.cancel().catch(() => {}); };
  signal.addEventListener('abort', cancel, { once: true });
  if (signal.aborted) cancel();
  const chunks: Uint8Array<ArrayBuffer>[] = [];
  let size = 0;
  try {
    while (true) {
      const { done, value } = await beforeAbort(signal, () => reader.read());
      if (done) break;
      size += value.byteLength;
      if (size > maxSupportBytes) {
        await reader.cancel().catch(() => {});
        return error(413, 'support_too_large', 'Support reports must be at most 25000000 compressed bytes.');
      }
      if (value.byteLength) chunks.push(value);
    }
  } catch {
    return error(400, 'invalid_support_report', 'The support upload could not be read. Retry the same report.');
  } finally {
    signal.removeEventListener('abort', cancel);
    reader.releaseLock();
  }
  if (!size) return error(400, 'invalid_support_report', 'Send a nonempty gzip support bundle.');
  return { chunks, size };
}

async function notify(url: string | undefined, report: Details, size: number): Promise<void> {
  if (!url) return;
  try {
    await fetch(url, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        embeds: [{ fields: [
          { name: 'Ticket', value: report.ticket },
          { name: 'App version', value: report.version },
          { name: 'Compressed bytes', value: String(size) },
          { name: 'Diagnostic codes', value: report.codes.join(', ') || 'none' },
        ] }],
        allowed_mentions: { parse: [] },
      }),
      signal: AbortSignal.timeout(5000),
      redirect: 'manual',
    });
  } catch {}
}

export async function receiveSupport(request: Request, env: SupportEnv, installation: string): Promise<Response> {
  if (env.SUPPORT_REPORTS_ENABLED !== 'true' || !env.SUPPORT_REPORTS || !env.AGENT_USAGE_DB) {
    return error(503, 'support_unavailable', 'Support reports are unavailable. No report was accepted or stored.');
  }
  if (request.headers.get('x-agent-protocol') !== '3') {
    return error(400, 'unsupported_agent_protocol', 'Update the app to use supported agent protocol 3.');
  }
  const report = details(request);
  if (!report || request.headers.get('content-type') !== 'application/gzip' || request.headers.has('content-encoding')) {
    return error(400, 'invalid_support_report', 'Use the supported gzip report and safe report headers.');
  }
  const key = `support/${report.ticket}.tar.gz`;
  const conflict = () => error(409, 'support_ticket_conflict', 'This report could not be accepted. Retry your report.');
  const unavailable = () => error(503, 'support_unavailable', 'The report outcome is uncertain. Retry the same report. An interrupted upload may consume a daily slot.');
  const bucket = env.SUPPORT_REPORTS;
  const db = env.AGENT_USAGE_DB;
  const timestamp = Date.now();
  const owner = crypto.randomUUID();
  try {
    const claimed = await db.prepare(`
      INSERT INTO request_slots (installation_hash, slot, owner, turn_id, classification, expires_at)
      VALUES (?, 2, ?, NULL, 0, ?)
      ON CONFLICT(installation_hash, slot) DO UPDATE SET owner = excluded.owner, expires_at = excluded.expires_at
      WHERE request_slots.expires_at <= ?
      RETURNING owner
    `).bind(installation, owner, timestamp + requestLeaseMs, timestamp).first<{ owner: string }>();
    if (!claimed) return error(429, 'service_busy', 'A support upload is already running. Wait for it to finish before trying again.');
    if (claimed.owner !== owner) return unavailable();
  } catch {
    return unavailable();
  }
  const { abort, release } = requestLifetime(timestamp + requestDeadlineMs, request.signal);
  let storagePending = false;
  try {
    const bundle = await readBundle(request, abort.signal);
    if (bundle instanceof Response) return bundle;
    try {
      const existing = await beforeAbort(abort.signal, async () => {
        storagePending = true;
        try { return await bucket.head(key); }
        finally { storagePending = false; }
      });
      if (existing) return existing.customMetadata?.installation === installation ? receipt(report.ticket) : conflict();
    } catch {
      return unavailable();
    }
    const day = new Date().toISOString().slice(0, 10);
    try {
      const slot = await beforeAbort(abort.signal, () => db.prepare(`
        INSERT INTO support_daily (installation_hash, utc_date, reports) VALUES (?, ?, 1)
        ON CONFLICT(installation_hash, utc_date) DO UPDATE SET reports = support_daily.reports + 1
        WHERE support_daily.reports < 3 RETURNING reports
      `).bind(installation, day).first<{ reports: number }>());
      if (!slot) return error(429, 'support_allowance_exhausted', 'Three support reports are allowed per installation per UTC day. Retry on the next UTC day.');
    } catch {
      return unavailable();
    }
    let stored: R2Object | null;
    try {
      stored = await beforeAbort(abort.signal, async () => {
        storagePending = true;
        try {
          return await bucket.put(key, new Blob(bundle.chunks, { type: 'application/gzip' }), {
            onlyIf: { etagDoesNotMatch: '*' },
            httpMetadata: { contentType: 'application/gzip' },
            customMetadata: { installation },
          });
        } finally {
          storagePending = false;
        }
      });
    } catch {
      return unavailable();
    }
    if (!stored) {
      try {
        const compensated = await beforeAbort(abort.signal, () => db.prepare(`
          UPDATE support_daily SET reports = reports - 1
          WHERE installation_hash = ? AND utc_date = ? AND reports > 0
        `).bind(installation, day).run());
        if (!compensated.success) return unavailable();
        const existing = await beforeAbort(abort.signal, async () => {
          storagePending = true;
          try { return await bucket.head(key); }
          finally { storagePending = false; }
        });
        return existing?.customMetadata?.installation === installation ? receipt(report.ticket) : conflict();
      } catch {
        return unavailable();
      }
    }
    await beforeAbort(abort.signal, () => notify(env.DISCORD_OPERATIONS_WEBHOOK_URL, report, bundle.size)).catch(() => {});
    return receipt(report.ticket);
  } finally {
    release();
    if (!storagePending) {
      try {
        await db.prepare('DELETE FROM request_slots WHERE installation_hash = ? AND slot = 2 AND owner = ?')
          .bind(installation, owner).run();
      } catch {}
    }
  }
}
