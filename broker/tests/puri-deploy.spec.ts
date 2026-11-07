import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { DatabaseSync } from 'node:sqlite';

import { afterEach, describe, expect, it } from 'vitest';
import { applyBrokerMigrations, readBrokerMigrationSql } from './test-support/migrations';

const script = fileURLToPath(new URL('../scripts/manage-puri-secrets.mjs', import.meta.url));
const tempDirs: string[] = [];
const token = 'puri-deploy-token-000000000000000000';
const identity = 'puri-deploy-identity-000000000000000';

afterEach(() => {
  for (const dir of tempDirs.splice(0)) rmSync(dir, { recursive: true, force: true });
});

function run(args: string[], values: Record<string, string | undefined> = {}) {
  const dir = mkdtempSync(join(tmpdir(), 'puri-broker-deploy-'));
  tempDirs.push(dir);
  const log = join(dir, 'calls.jsonl');
  const boundary = join(dir, 'pnpm-boundary.mjs');
  writeFileSync(boundary, `
import childProcess from 'node:child_process';
import { syncBuiltinESMExports } from 'node:module';
import { appendFileSync } from 'node:fs';
childProcess.spawnSync = (command, args, options) => {
  if (command !== 'pnpm') throw new Error('Unexpected subprocess');
  appendFileSync(process.env.PURI_TEST_LOG, JSON.stringify({ args, input: options.input }) + '\\n');
  return { status: 0, stdout: '', stderr: '' };
};
syncBuiltinESMExports();
`);
  const result = spawnSync(process.execPath, ['--import', pathToFileURL(boundary).href, script, ...args], {
    encoding: 'utf8', env: {
      ...process.env, PURI_TOKEN_SECRET_PRODUCTION: '', PURI_IDENTITY_HMAC_SECRET_PRODUCTION: '',
      PURI_TEST_LOG: log, ...values,
    },
  });
  let calls: Array<{ args: string[]; input: string }> = [];
  try { calls = readFileSync(log, 'utf8').trim().split('\n').filter(Boolean).map((line) => JSON.parse(line)); } catch {}
  return { ...result, calls };
}

describe('optional Puri deployment secret contract', () => {
  it('adds Puri state with both destructive migrations held and preserves tuned policy and legacy tables', () => {
    const db = new DatabaseSync(':memory:');
    try {
      applyBrokerMigrations(db, { through: '0024_allow_unattributed_request_events.sql' });
      db.exec(readBrokerMigrationSql('0026_managed_attempt_cleanup_ownership.sql'));
      db.exec(`UPDATE broker_config SET value = json_set(value, '$.trialChallenge.maxRequests', 7) WHERE key = 'abuse_controls'`);
      const tablesBefore = db.prepare("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name").all();
      db.exec(readBrokerMigrationSql('0028_add_puri_identity_proofs.sql'));
      expect(db.prepare("SELECT name FROM sqlite_master WHERE type = 'table' AND name <> 'puri_proof_nonces' ORDER BY name").all()).toEqual(tablesBefore);
      expect(db.prepare("SELECT name FROM pragma_table_info('installations') WHERE name = 'puri_bound_at'").get()).toEqual({ name: 'puri_bound_at' });
      const row = db.prepare("SELECT value FROM broker_config WHERE key = 'abuse_controls'").get()!;
      expect(JSON.parse(String(row.value))).toMatchObject({
        trialChallenge: { maxRequests: 7 },
        puriTokenIp: { endpoint: 'POST /v1/puri/token', scope: 'ip', maxRequests: 30, windowMinutes: 15 },
      });
    } finally {
      db.close();
    }
  });

  it('keeps unprovisioned deployments valid and validates UTF-8 length and distinct secrets without cloud calls', () => {
    expect(run(['--validate']).status).toBe(0);
    for (const values of [
      { PURI_TOKEN_SECRET_PRODUCTION: token },
      { PURI_TOKEN_SECRET_PRODUCTION: token, PURI_IDENTITY_HMAC_SECRET_PRODUCTION: 'short' },
      { PURI_TOKEN_SECRET_PRODUCTION: token, PURI_IDENTITY_HMAC_SECRET_PRODUCTION: token },
    ]) {
      const result = run(['--validate'], values);
      expect(result.status).not.toBe(0);
      expect(result.calls).toEqual([]);
      expect(result.stdout + result.stderr).not.toContain(token);
    }
    expect(run(['--validate'], { PURI_TOKEN_SECRET_PRODUCTION: '한'.repeat(11), PURI_IDENTITY_HMAC_SECRET_PRODUCTION: identity }).status).toBe(0);
  });

  it('uploads the configured pair only through stdin', () => {
    const result = run(['--config', 'synthetic.jsonc'], {
      PURI_TOKEN_SECRET_PRODUCTION: token, PURI_IDENTITY_HMAC_SECRET_PRODUCTION: identity,
    });
    expect(result.status).toBe(0);
    expect(result.calls).toEqual([{
      args: ['exec', 'wrangler', 'secret', 'bulk', '--config', 'synthetic.jsonc'],
      input: JSON.stringify({ PURI_TOKEN_SECRET: token, PURI_IDENTITY_HMAC_SECRET: identity }),
    }]);
    expect(result.stdout + result.stderr).toBe('');
  });

  it('preserves existing Worker secrets without any remote command when both inputs are absent', () => {
    const result = run(['--config', 'synthetic.jsonc']);
    expect(result.status).toBe(0);
    expect(result.calls).toEqual([]);
    expect(result.stdout + result.stderr).toBe('');
  });

  it('rejects partial, short, or identical supplied pairs before any remote command', () => {
    for (const values of [
      { PURI_TOKEN_SECRET_PRODUCTION: token },
      { PURI_IDENTITY_HMAC_SECRET_PRODUCTION: identity },
      { PURI_TOKEN_SECRET_PRODUCTION: token, PURI_IDENTITY_HMAC_SECRET_PRODUCTION: 'short' },
      { PURI_TOKEN_SECRET_PRODUCTION: 'short', PURI_IDENTITY_HMAC_SECRET_PRODUCTION: identity },
      { PURI_TOKEN_SECRET_PRODUCTION: token, PURI_IDENTITY_HMAC_SECRET_PRODUCTION: token },
    ]) {
      const result = run(['--config', 'synthetic.jsonc'], values);
      expect(result.status).not.toBe(0);
      expect(result.calls).toEqual([]);
      expect(result.stdout + result.stderr).not.toContain(token);
      expect(result.stdout + result.stderr).not.toContain(identity);
    }
  });
});
