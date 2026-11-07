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

type Rates = typeof defaultRates;
type Usage = { uncached: number; read: number; write: number; output: number; missing: number };

export const turnLifetimeMs = 24 * 60 * 60 * 1000;
export const maxTurnAdmissions = 15;

export type UsageEnv = { AGENT_USAGE_DB?: D1Database; AGENT_USAGE_RATES?: string };
export type Accounting = {
  db: D1Database;
  installation: string;
  day: string;
  rates: Rates;
  owner: string;
  finalAdmission: boolean;
  deadline: number;
};
type AdmissionError = 'service_allowance_exhausted' | 'service_turn_exhausted' | 'service_busy';
export type Admission = { accounting: Accounting } | { error: AdmissionError };

export async function checkAllowance(
  env: UsageEnv, installation: string, turnId: string, classification: boolean, now = new Date(),
): Promise<Admission> {
  if (!env.AGENT_USAGE_DB) throw new Error('accounting_unavailable');
  let rates = defaultRates;
  if (env.AGENT_USAGE_RATES !== undefined) {
    const configured: unknown = JSON.parse(env.AGENT_USAGE_RATES);
    if (!isRecord(configured) || Object.keys(configured).length !== Object.keys(defaultRates).length
      || typeof configured.as_of !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(configured.as_of)
      || !['input', 'cache_read', 'cache_write', 'responses_output', 'decisions_output'].every((key) =>
        typeof configured[key] === 'number' && Number.isFinite(configured[key]) && configured[key] >= 0)) {
      throw new Error('accounting_unavailable');
    }
    rates = configured as Rates;
  }
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
          AND (SELECT COUNT(*) FROM prompt_turns WHERE installation_hash = ? AND utc_date = ?) < 50))
    ORDER BY available.slot LIMIT 1
    ON CONFLICT(installation_hash, slot) DO UPDATE SET owner = excluded.owner, turn_id = excluded.turn_id,
      classification = excluded.classification, expires_at = excluded.expires_at
    RETURNING owner
  `).bind(installation, owner, turnId, classification ? 1 : 0, timestamp + requestLeaseMs, installation, timestamp,
    classification ? 1 : 0, installation, turnId, timestamp, installation, turnId, timestamp, maxTurnAdmissions,
    classification ? 1 : 0, installation, turnId, installation, day);
  const admitted = db.prepare(`
    INSERT INTO prompt_turns (installation_hash, turn_id, utc_date, expires_at, model_admissions, classifier_admitted)
    SELECT ?, ?, ?, ?, 1, ?
    WHERE EXISTS (SELECT 1 FROM request_slots WHERE installation_hash = ? AND slot < 2 AND owner = ?)
    ON CONFLICT(installation_hash, turn_id) DO UPDATE SET
      model_admissions = prompt_turns.model_admissions + 1,
      classifier_admitted = prompt_turns.classifier_admitted + excluded.classifier_admitted
    WHERE prompt_turns.expires_at > ? AND prompt_turns.model_admissions < ?
      AND (? = 0 OR prompt_turns.classifier_admitted = 0)
    RETURNING turn_id, model_admissions, expires_at
  `).bind(installation, turnId, day, timestamp + turnLifetimeMs, classification ? 1 : 0,
    installation, owner, timestamp, maxTurnAdmissions, classification ? 1 : 0);
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
          AND (SELECT COUNT(*) FROM prompt_turns WHERE installation_hash = ? AND utc_date = ?) >= 50
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
    || !isRecord(claim) || claim.owner !== owner) throw new Error('accounting_unavailable');
  return { accounting: { db, installation, day, rates, owner, finalAdmission: row.model_admissions === maxTurnAdmissions,
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

export async function recordUsage(accounting: Accounting, result: unknown, classification: boolean): Promise<void> {
  let usage: Usage = { uncached: 0, read: 0, write: 0, output: 0, missing: 1 };
  const returned = isRecord(result) ? result.usage : undefined;
  if (isRecord(returned)) {
    const details = returned.input_tokens_details;
    const read = isRecord(details) ? details.cached_tokens ?? 0 : 0;
    const write = isRecord(details) ? details.cache_write_tokens ?? 0 : 0;
    const input = returned.input_tokens;
    const output = returned.output_tokens;
    if ([input, output, read, write].every((value) => Number.isSafeInteger(value) && (value as number) >= 0)
      && (read as number) + (write as number) <= (input as number)) {
      usage = { uncached: (input as number) - (read as number) - (write as number),
        read: read as number, write: write as number, output: output as number, missing: 0 };
    }
  }
  const { rates } = accounting;
  const cost = (usage.uncached * rates.input + usage.read * rates.cache_read + usage.write * rates.cache_write
    + usage.output * (classification ? rates.decisions_output : rates.responses_output)) / 1_000_000;
  if (!Number.isFinite(cost)) throw new Error('accounting_unavailable');
  const written = await accounting.db.batch([accounting.db.prepare(`
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
    classification ? 1 : 0, classification ? 0 : 1, usage.missing, cost),
  accounting.db.prepare('DELETE FROM request_slots WHERE installation_hash = ? AND slot < 2 AND owner = ?')
    .bind(accounting.installation, accounting.owner)]);
  if (written.length !== 2 || written.some((result) => !result.success)) throw new Error('accounting_unavailable');
}
