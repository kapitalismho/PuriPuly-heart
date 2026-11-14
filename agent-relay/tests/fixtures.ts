import { createHmac } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { DatabaseSync } from 'node:sqlite';
import { vi } from 'vitest';

export const tokenSecret = 'puri-330-relay-synthetic-secret-32-bytes';
export const installationHash = createHmac('sha256', 'synthetic-separate-identity-secret').update('synthetic-installation').digest('hex');
export const turnId = '0123456789abcdef0123456789abcdef';

export function signToken(claims: Record<string, unknown> = {}, header: Record<string, unknown> = { alg: 'HS256', typ: 'JWT' }, secret = tokenSecret): string {
  const now = Math.floor(Date.now() / 1000);
  const encoded = [header, { iss: 'puripuly-heart-broker', aud: 'puripuly-agent-relay', sub: installationHash, iat: now, exp: now + 3600, ...claims }]
    .map((value) => Buffer.from(JSON.stringify(value)).toString('base64url')).join('.');
  return `${encoded}.${createHmac('sha256', secret).update(encoded).digest('base64url')}`;
}

export function authHeaders(): Record<string, string> {
  return { authorization: `Bearer ${signToken()}`, 'content-type': 'application/json' };
}

export function usageDatabase() {
  const admission = vi.fn().mockImplementation(async (turn_id: string) =>
    ({ turn_id, model_admissions: 1, expires_at: Date.now() + 86_400_000, response_model: 'gpt-5.6-terra' }));
  const denial = vi.fn().mockResolvedValue({ code: 'service_allowance_exhausted' });
  const writes: unknown[][] = [];
  const protocols: unknown[][] = [];
  const releases: unknown[][] = [];
  const run = vi.fn().mockResolvedValue({ success: true, results: [] });
  const prepare = vi.fn((sql: string) => ({
    bind: (...values: unknown[]) => ({
      sql, values,
      first: () => sql.includes('INSERT INTO prompt_turns') ? admission(values[1]) : denial(),
      run: () => {
        if (sql.includes('INSERT INTO usage_daily')) {
          writes.push(values);
          return run();
        }
        if (sql.includes('INSERT INTO protocol_daily')) {
          protocols.push(values);
          return Promise.resolve({ success: true, results: [] });
        }
        if (sql.includes('DELETE FROM request_slots')) releases.push(values);
        return Promise.resolve({ success: true, results: [] });
      },
    }),
  }));
  const batch = vi.fn(async (statements: { sql: string; values: unknown[]; first(): Promise<unknown>; run(): Promise<unknown> }[]) => {
    if (statements[0].sql.includes('INSERT INTO request_slots')) {
      const row = await statements[1].first();
      return [{ success: true, results: row === null ? [] : [{ owner: statements[0].values[1] }] },
        { success: true, results: row === null ? [] : [row] }];
    }
    const results = [];
    for (const statement of statements) results.push(await statement.run());
    return results;
  });
  return { db: { prepare, batch } as unknown as D1Database, admission, denial, writes, protocols, releases, run, prepare, batch };
}

export const relayMigrations = ['0001_usage_daily.sql', '0002_support_daily.sql', '0003_prompt_turns.sql', '0004_request_admission.sql', '0005_protocol_daily.sql', '0006_photo_admission.sql', '0007_terra_routing.sql'];

export function accountingDatabase(migrations = relayMigrations) {
  const sqlite = new DatabaseSync(':memory:');
  for (const migration of migrations) {
    sqlite.exec(readFileSync(new URL(`../migrations/${migration}`, import.meta.url), 'utf8'));
  }
  type Statement = { sql: string; values: (string | number | null)[] };
  const prepare = (sql: string) => ({ bind: (...values: Statement['values']) => ({
    sql, values,
    first: async () => sqlite.prepare(sql).get(...values) ?? null,
    all: async () => ({ success: true, results: sqlite.prepare(sql).all(...values) }),
    run: async () => {
      sqlite.prepare(sql).run(...values);
      return { success: true, results: [] };
    },
  }) });
  const batch = async (statements: Statement[]) => {
    sqlite.exec('BEGIN');
    try {
      const results = statements.map(({ sql, values }) => ({
        success: true, results: sqlite.prepare(sql).all(...values),
      }));
      sqlite.exec('COMMIT');
      return results;
    } catch (error) {
      sqlite.exec('ROLLBACK');
      throw error;
    }
  };
  return { sqlite, db: { prepare, batch } as unknown as D1Database };
}

export function testEnv() {
  return { AGENT_SERVICE_ENABLED: 'true', OPENAI_API_KEY: 'synthetic-test-key', PURI_TOKEN_SECRET: tokenSecret,
    AGENT_USAGE_DB: usageDatabase().db };
}

export function responseEvents(response: Record<string, unknown>, events: Record<string, unknown>[] = []): Response {
  const type = response.status === 'incomplete' ? 'response.incomplete' : 'response.completed';
  return new Response([...events, { type, response }].map((event) => `data: ${JSON.stringify(event)}\n\n`).join(''), {
    headers: { 'content-type': 'text/event-stream' },
  });
}

export async function relayEvents(response: Response): Promise<Record<string, unknown>[]> {
  return (await response.text()).split('\n\n').filter(Boolean).map((frame) => JSON.parse(frame.slice(6)));
}
