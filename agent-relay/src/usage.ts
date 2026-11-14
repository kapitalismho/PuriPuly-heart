import { requestDeadlineMs, requestLeaseMs } from './lifetime';
import { isRecord } from './type-guards';

export const defaultRates = {
  as_of: '2026-10-29',
  input: 0.10,
  cache_read: 0.01,
  cache_write: 0.125,
  responses_output: 0.50,
  decisions_output: 0,
};

const terraRates = {
  as_of: '2026-11-11',
  input: 2,
  cache_read: 0.20,
  cache_write: 2.50,
  responses_output: 12,
  decisions_output: 0,
};

export type ResponseModel = 'gpt-5.6-terra' | 'gpt-6-luna';
export const terraDailyTokens = 10_000_000;

type Rates = typeof defaultRates;
type Usage = { uncached: number; read: number; write: number; write12h: number; output: number; missing: number };

export const turnLifetimeMs = 24 * 60 * 60 * 1000;
export const maxTurnAdmissions = 15;

export type UsageEnv = { AGENT_USAGE_DB?: D1Database; AGENT_USAGE_RATES?: string };
export type Accounting = {
  db: D1Database;
  installation: string;
  day: string;
  rates: Rates;
  model: ResponseModel;
  owner: string;
  finalAdmission: boolean;
  deadline: number;
  protocol?: number;
};
type AdmissionError = 'service_allowance_exhausted' | 'service_turn_exhausted' | 'service_busy';
export type Admission = { accounting: Accounting } | { error: AdmissionError };

function usageRates(env: UsageEnv): Rates {
  if (env.AGENT_USAGE_RATES === undefined) return defaultRates;
  const configured: unknown = JSON.parse(env.AGENT_USAGE_RATES);
  if (!isRecord(configured) || Object.keys(configured).length !== Object.keys(defaultRates).length
    || typeof configured.as_of !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(configured.as_of)
    || !['input', 'cache_read', 'cache_write', 'responses_output', 'decisions_output'].every((key) =>
      typeof configured[key] === 'number' && Number.isFinite(configured[key]) && configured[key] >= 0)) {
    throw new Error('accounting_unavailable');
  }
  return configured as Rates;
}

export async function checkAllowance(
  env: UsageEnv, installation: string, turnId: string, classification: boolean, now = new Date(),
): Promise<Admission> {
  if (!env.AGENT_USAGE_DB) throw new Error('accounting_unavailable');
  const rates = usageRates(env);
  const db = env.AGENT_USAGE_DB;
  const day = now.toISOString().slice(0, 10);
  const timestamp = now.getTime();
  const owner = crypto.randomUUID();
  const claimed = db.prepare(`
    INSERT INTO request_slots (installation_hash, slot, owner, turn_id, classification, expires_at)
    SELECT ?, available.slot, ?, ?, ?, ? FROM (SELECT 0 AS slot UNION ALL SELECT 1) AS available
    WHERE NOT EXISTS (SELECT 1 FROM request_slots
      WHERE installation_hash = ? AND slot = available.slot AND expires_at > ?)
      AND (? = 1 OR NOT EXISTS (SELECT 1 FROM request_slots
        WHERE installation_hash = ? AND slot < 2 AND turn_id = ? AND classification = 0 AND expires_at > ?))
      AND (EXISTS (SELECT 1 FROM prompt_turns WHERE installation_hash = ? AND turn_id = ?
        AND expires_at > ? AND model_admissions < ? AND (? = 0 OR classifier_admitted = 0))
        OR (NOT EXISTS (SELECT 1 FROM prompt_turns WHERE installation_hash = ? AND turn_id = ?)
          AND (SELECT COUNT(*) FROM prompt_turns WHERE installation_hash = ? AND utc_date = ?) < 30))
    ORDER BY available.slot LIMIT 1
    ON CONFLICT(installation_hash, slot) DO UPDATE SET owner = excluded.owner, turn_id = excluded.turn_id,
      classification = excluded.classification, expires_at = excluded.expires_at
    RETURNING owner
  `).bind(installation, owner, turnId, classification ? 1 : 0, timestamp + requestLeaseMs, installation, timestamp,
    classification ? 1 : 0, installation, turnId, timestamp, installation, turnId, timestamp, maxTurnAdmissions,
    classification ? 1 : 0, installation, turnId, installation, day);
  const admitted = db.prepare(`
    INSERT INTO prompt_turns (installation_hash, turn_id, utc_date, expires_at, model_admissions, classifier_admitted, response_model)
    SELECT ?, ?, ?, ?, 1, ?, CASE WHEN ? = 1 THEN NULL
      WHEN COALESCE((SELECT tokens FROM terra_daily WHERE utc_date = ?), 0) < ? THEN 'gpt-5.6-terra'
      ELSE 'gpt-6-luna' END
    WHERE EXISTS (SELECT 1 FROM request_slots WHERE installation_hash = ? AND slot < 2 AND owner = ?)
    ON CONFLICT(installation_hash, turn_id) DO UPDATE SET
      model_admissions = prompt_turns.model_admissions + 1,
      classifier_admitted = prompt_turns.classifier_admitted + excluded.classifier_admitted,
      response_model = COALESCE(prompt_turns.response_model, excluded.response_model)
    WHERE prompt_turns.expires_at > ? AND prompt_turns.model_admissions < ?
      AND (? = 0 OR prompt_turns.classifier_admitted = 0)
    RETURNING turn_id, model_admissions, expires_at, response_model
  `).bind(installation, turnId, day, timestamp + turnLifetimeMs, classification ? 1 : 0,
    classification ? 1 : 0, day, terraDailyTokens, installation, owner, timestamp, maxTurnAdmissions, classification ? 1 : 0);
  const results = await db.batch([claimed, admitted]);
  if (results.length !== 2 || results.some((result) => !result.success)) throw new Error('accounting_unavailable');
  const row = results[1].results[0];
  const claim = results[0].results[0];
  if (results[1].results.length === 0) {
    if (results[0].results.length !== 0) throw new Error('accounting_unavailable');
    const reason = await db.prepare(`
      SELECT CASE
        WHEN EXISTS (SELECT 1 FROM prompt_turns WHERE installation_hash = ? AND turn_id = ?
          AND (expires_at <= ? OR model_admissions >= ? OR (? = 1 AND classifier_admitted = 1)))
          THEN 'service_turn_exhausted'
        WHEN NOT EXISTS (SELECT 1 FROM prompt_turns WHERE installation_hash = ? AND turn_id = ?)
          AND (SELECT COUNT(*) FROM prompt_turns WHERE installation_hash = ? AND utc_date = ?) >= 30
          THEN 'service_allowance_exhausted'
        ELSE 'service_busy' END AS code
    `).bind(installation, turnId, timestamp, maxTurnAdmissions, classification ? 1 : 0,
      installation, turnId, installation, day).first<{ code: AdmissionError }>();
    if (!reason || !['service_turn_exhausted', 'service_allowance_exhausted', 'service_busy'].includes(reason.code)) {
      throw new Error('accounting_unavailable');
    }
    return { error: reason.code };
  }
  if (!isRecord(row) || row.turn_id !== turnId || !Number.isSafeInteger(row.model_admissions)
    || (row.model_admissions as number) < 1 || (row.model_admissions as number) > maxTurnAdmissions
    || !Number.isSafeInteger(row.expires_at) || (row.expires_at as number) <= timestamp
    || (!classification && !['gpt-5.6-terra', 'gpt-6-luna'].includes(row.response_model as string))
    || !isRecord(claim) || claim.owner !== owner) throw new Error('accounting_unavailable');
  const model = classification ? 'gpt-6-luna' : row.response_model as ResponseModel;
  return { accounting: { db, installation, day, rates: model === 'gpt-5.6-terra' ? terraRates : rates,
    model, owner, finalAdmission: row.model_admissions === maxTurnAdmissions,
    deadline: Math.min(timestamp + requestDeadlineMs, row.expires_at as number) } };
}

export function usageRecorder(accounting: Accounting, classification: boolean, context?: Pick<ExecutionContext, 'waitUntil'>) {
  let recorded: Promise<void> | undefined;
  return (result: unknown): Promise<void> => {
    if (!recorded) {
      recorded = recordUsage(accounting, result, classification);
      context?.waitUntil(recorded);
    }
    return recorded;
  };
}

function responseUsage(result: unknown): Usage {
  let usage: Usage = { uncached: 0, read: 0, write: 0, write12h: 0, output: 0, missing: 1 };
  const returned = isRecord(result) ? result.usage : undefined;
  if (isRecord(returned)) {
    const details = returned.input_tokens_details;
    const read = isRecord(details) ? details.cached_tokens ?? 0 : 0;
    const write = isRecord(details) ? details.cache_write_tokens ?? 0 : 0;
    const writeDetails = isRecord(details) ? details.cache_write_tokens_details : undefined;
    const write12h = isRecord(writeDetails) ? writeDetails.ttl_12h_tokens ?? 0 : 0;
    const input = returned.input_tokens;
    const output = returned.output_tokens;
    if ([input, output, read, write, write12h].every((value) => Number.isSafeInteger(value) && (value as number) >= 0)
      && (read as number) + (write as number) <= (input as number) && (write12h as number) <= (write as number)) {
      usage = { uncached: (input as number) - (read as number) - (write as number),
        read: read as number, write: write as number, write12h: write12h as number, output: output as number, missing: 0 };
    }
  }
  return usage;
}

export async function recordUsage(accounting: Accounting, result: unknown, classification: boolean): Promise<void> {
  const usage = responseUsage(result);
  const { rates } = accounting;
  const cost = (usage.uncached * rates.input + usage.read * rates.cache_read
    + (usage.write + (accounting.model === 'gpt-5.6-terra' ? usage.write12h : 0)) * rates.cache_write
    + usage.output * (classification ? rates.decisions_output : rates.responses_output)) / 1_000_000;
  if (!Number.isFinite(cost)) throw new Error('accounting_unavailable');
  const statements = [accounting.db.prepare(`
    INSERT INTO usage_daily (installation_hash, utc_date, uncached_input_tokens, cache_read_tokens, cache_write_tokens,
      decisions_output_tokens, responses_output_tokens, decisions_calls, responses_calls, missing_usage_count, cost_usd)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(installation_hash, utc_date) DO UPDATE SET
      uncached_input_tokens = uncached_input_tokens + excluded.uncached_input_tokens,
      cache_read_tokens = cache_read_tokens + excluded.cache_read_tokens,
      cache_write_tokens = cache_write_tokens + excluded.cache_write_tokens,
      decisions_output_tokens = decisions_output_tokens + excluded.decisions_output_tokens,
      responses_output_tokens = responses_output_tokens + excluded.responses_output_tokens,
      decisions_calls = decisions_calls + excluded.decisions_calls,
      responses_calls = responses_calls + excluded.responses_calls,
      missing_usage_count = missing_usage_count + excluded.missing_usage_count,
      cost_usd = cost_usd + excluded.cost_usd
  `).bind(accounting.installation, accounting.day, usage.uncached, usage.read, usage.write,
    classification ? usage.output : 0, classification ? 0 : usage.output,
    classification ? 1 : 0, classification ? 0 : 1, usage.missing, cost)];
  if (!classification && accounting.model === 'gpt-5.6-terra' && !usage.missing) {
    statements.push(accounting.db.prepare(`
      INSERT INTO terra_daily (utc_date, tokens) VALUES (?, ?)
      ON CONFLICT(utc_date) DO UPDATE SET tokens = tokens + excluded.tokens
    `).bind(accounting.day, usage.uncached + usage.read + usage.write + usage.output));
  }
  if (accounting.protocol !== undefined) {
    statements.push(accounting.db.prepare(`
      INSERT INTO protocol_daily (installation_hash, utc_date, agent_protocol, requests) VALUES (?, ?, ?, 1)
      ON CONFLICT(installation_hash, utc_date, agent_protocol) DO UPDATE SET requests = requests + 1
    `).bind(accounting.installation, accounting.day, accounting.protocol));
  }
  statements.push(accounting.db.prepare('DELETE FROM request_slots WHERE installation_hash = ? AND slot < 2 AND owner = ?')
    .bind(accounting.installation, accounting.owner));
  const written = await accounting.db.batch(statements);
  if (written.length !== statements.length || written.some((result) => !result.success)) throw new Error('accounting_unavailable');
}

export type PhotoPolicy = {
  maxBodyBytes: number;
  maxOutputTokens: number;
  concurrency: number;
  allowance: number;
  periodMs: number;
  deadlineMs: number;
};

export function photoPolicy(raw?: string): PhotoPolicy {
  const defaults: PhotoPolicy = {
    maxBodyBytes: 4 * 1024 * 1024,
    maxOutputTokens: 8192,
    concurrency: 1,
    allowance: 25,
    periodMs: 86_400_000,
    deadlineMs: 240_000,
  };
  if (raw === undefined) return defaults;
  const configured: unknown = JSON.parse(raw);
  if (!isRecord(configured) || Object.keys(configured).some((key) => !(key in defaults))) {
    throw new Error('photo_policy_invalid');
  }
  const policy = { ...defaults, ...configured };
  if (Object.values(policy).some((value) => !Number.isSafeInteger(value) || value < 1)
    || policy.maxBodyBytes > 20 * 1024 * 1024 || policy.maxOutputTokens > 32768
    || policy.concurrency > 8 || policy.deadlineMs > 300_000
    || policy.periodMs < 60_000) throw new Error('photo_policy_invalid');
  return policy;
}

export async function photoAllowanceUsed(
  env: UsageEnv, installation: string, policy: PhotoPolicy, now = Date.now(),
): Promise<{ used: number; resetAfter: number }> {
  if (!env.AGENT_USAGE_DB) throw new Error('accounting_unavailable');
  const period = Math.floor(now / policy.periodMs) * policy.periodMs;
  const row = await env.AGENT_USAGE_DB.prepare(
    'SELECT COUNT(*) AS used FROM photo_admissions WHERE installation_hash = ? AND period_start = ?',
  ).bind(installation, period).first<{ used: number }>();
  if (!row || !Number.isSafeInteger(row.used)) throw new Error('accounting_unavailable');
  return { used: row.used, resetAfter: Math.ceil((period + policy.periodMs - now) / 1000) };
}

export async function checkPhotoAllowance(
  env: UsageEnv, installation: string, requestId: string, policy: PhotoPolicy, now = Date.now(),
): Promise<{ settle: (result: unknown) => Promise<void>; deadline: number } | { error: string; retryAfter?: number }> {
  if (!env.AGENT_USAGE_DB) throw new Error('accounting_unavailable');
  const db = env.AGENT_USAGE_DB;
  const rates = usageRates(env);
  const owner = crypto.randomUUID();
  const period = Math.floor(now / policy.periodMs) * policy.periodMs;
  const admitted = await db.prepare(`
    INSERT INTO photo_admissions (installation_hash, request_id, period_start, owner, expires_at)
    SELECT ?, ?, ?, ?, ?
    WHERE NOT EXISTS (SELECT 1 FROM photo_admissions WHERE installation_hash = ? AND request_id = ?)
      AND (SELECT COUNT(*) FROM photo_admissions WHERE installation_hash = ? AND period_start = ?) < ?
      AND (SELECT COUNT(*) FROM photo_admissions
        WHERE installation_hash = ? AND settled = 0 AND expires_at > ?) < ?
    ON CONFLICT(installation_hash, request_id) DO NOTHING
    RETURNING owner
  `).bind(installation, requestId, period, owner, now + policy.deadlineMs + 60_000,
    installation, requestId, installation, period, policy.allowance,
    installation, now, policy.concurrency).all<{ owner: string }>();
  if (!admitted.success) throw new Error('accounting_unavailable');
  if (!admitted.results.length) {
    const reason = await db.prepare(`
      SELECT CASE
        WHEN EXISTS (SELECT 1 FROM photo_admissions WHERE installation_hash = ? AND request_id = ?)
          THEN 'photo_request_already_admitted'
        WHEN (SELECT COUNT(*) FROM photo_admissions WHERE installation_hash = ? AND period_start = ?) >= ?
          THEN 'service_allowance_exhausted'
        ELSE 'service_busy' END AS code
    `).bind(installation, requestId, installation, period, policy.allowance).first<{ code: string }>();
    if (!reason) throw new Error('accounting_unavailable');
    return reason.code === 'service_allowance_exhausted'
      ? { error: reason.code, retryAfter: Math.ceil((period + policy.periodMs - now) / 1000) }
      : { error: reason.code };
  }
  if (admitted.results.length !== 1 || admitted.results[0].owner !== owner) {
    throw new Error('accounting_unavailable');
  }
  let settlement: Promise<void> | undefined;
  return {
    deadline: now + policy.deadlineMs,
    settle(result: unknown) {
      if (!settlement) settlement = (async () => {
        const usage = responseUsage(result);
        const cost = (usage.uncached * rates.input + usage.read * rates.cache_read
          + usage.write * rates.cache_write + usage.output * rates.responses_output) / 1_000_000;
        if (!Number.isFinite(cost)) throw new Error('accounting_unavailable');
        const written = await db.prepare(`
          UPDATE photo_admissions SET settled = 1, uncached_input_tokens = ?, cache_read_tokens = ?,
            cache_write_tokens = ?, output_tokens = ?, missing_usage = ?, cost_usd = ?
          WHERE installation_hash = ? AND request_id = ? AND owner = ? AND settled = 0
          RETURNING owner
        `).bind(usage.uncached, usage.read, usage.write, usage.output, usage.missing, cost,
          installation, requestId, owner).all<{ owner: string }>();
        if (!written.success || written.results.length !== 1 || written.results[0].owner !== owner) {
          throw new Error('accounting_unavailable');
        }
      })();
      return settlement;
    },
  };
}
