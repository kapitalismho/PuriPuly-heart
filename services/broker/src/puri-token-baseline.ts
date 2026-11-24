import type { Context, Next } from 'hono';

import type { BrokerEnv } from './contract';

const DAY_MS = 24 * 60 * 60_000;
const WINDOW_DAYS = 7;
const LATENCY_BOUNDS_MS = [25, 50, 100, 200, 400, 800, 1600, 3200, 6400];
const VERSION_ID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/iu;

function activeStart(start: string, now: number): boolean {
  if (!/^\d{4}-\d{2}-\d{2}$/u.test(start) || start.startsWith('0000')) return false;
  const midnight = new Date(`${start}T00:00:00.000Z`);
  const midnightMs = midnight.getTime();
  return now >= midnightMs && now < midnightMs + WINDOW_DAYS * DAY_MS &&
    midnight.toISOString().startsWith(start);
}

function latencyBound(elapsedMs: number): number {
  for (const bound of LATENCY_BOUNDS_MS) {
    if (elapsedMs <= bound) return bound;
  }
  return -1;
}

function utcDay(timestamp: number): string {
  return new Date(timestamp).toISOString().slice(0, 10);
}

export async function capturePuriTokenBaseline(c: Context<BrokerEnv>, next: Next): Promise<void> {
  const start = c.env?.TOKEN_BASELINE_START_UTC;
  if (typeof start !== 'string' || start === '') return next();
  const startedAt = performance.now();
  const receivedAt = Date.now();
  const versionId = c.env.CF_VERSION_METADATA?.id;
  if (typeof versionId !== 'string' || !VERSION_ID.test(versionId) || !activeStart(start, receivedAt)) {
    return next();
  }

  const colo = (c.req.raw as Request).cf?.colo;
  c.set('puriTokenBaselineCohort', 'unknown');
  await next();

  try {
    const upperMs = latencyBound(performance.now() - startedAt);
    const context = c.executionCtx;
    if (typeof context.waitUntil !== 'function') return;
    const status = c.res.status;
    const cohort = status >= 200 && status < 300 ? c.get('puriTokenBaselineCohort') ?? 'unknown' : 'unknown';
    const ingressColo = typeof colo === 'string' && /^[A-Z]{3}$/u.test(colo) ? colo : 'unknown';
    const day = utcDay(receivedAt);
    const db = c.env.BROKER_DB;
    let registered = false;
    context.waitUntil(Promise.resolve().then(async () => {
      if (!registered) return;
      await db.prepare(
        `INSERT INTO puri_token_baseline (day_utc, version_id, ingress_colo, cohort, status, upper_ms, requests)
         VALUES (?, ?, ?, ?, ?, ?, 1)
         ON CONFLICT (day_utc, version_id, ingress_colo, cohort, status, upper_ms)
         DO UPDATE SET requests = puri_token_baseline.requests + 1`,
      ).bind(day, versionId.toLowerCase(), ingressColo, cohort, status, upperMs).run();
    }).catch(() => {}));
    registered = true;
  } catch {}
}

export async function expirePuriTokenBaseline(db: D1Database, now: Date): Promise<void> {
  const oldest = utcDay(now.getTime() - (WINDOW_DAYS - 1) * DAY_MS);
  try {
    await db.prepare('DELETE FROM puri_token_baseline WHERE day_utc < ?').bind(oldest).run();
  } catch (error) {
    if (!(error instanceof Error && /\bno such table: (?:main\.)?puri_token_baseline(?:$|:)/u.test(error.message))) {
      throw error;
    }
  }
}
