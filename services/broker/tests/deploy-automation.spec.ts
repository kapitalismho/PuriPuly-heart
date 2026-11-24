import { execFileSync, spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, relative, sep } from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import { fileURLToPath } from 'node:url';

import { afterEach, describe, expect, it } from 'vitest';

import { applyBrokerMigrations } from './test-support/migrations';

const renderWranglerConfigScript = new URL(
  '../scripts/render-production-wrangler-config.mjs',
  import.meta.url,
);
const renderFingerprintBootstrapScript = new URL(
  '../scripts/render-fingerprint-bootstrap-sql.mjs',
  import.meta.url,
);
const checkedInWranglerConfig = new URL('../wrangler.jsonc', import.meta.url);
const brokerDatabaseId = '22222222-2222-4222-8222-222222222222';
const agentUsageDatabaseId = '11111111-1111-4111-8111-111111111111';
const abuseControlsWorkflow = new URL(
  '../../../.github/workflows/maintenance-broker-abuse-controls.yml',
  import.meta.url,
);
const cleanupWorkflow = new URL(
  '../../../.github/workflows/maintenance-broker-d1-cleanup.yml',
  import.meta.url,
);

const bashExecutable = process.platform === 'win32'
  ? join(
      dirname(dirname(execFileSync('where.exe', ['git'], { encoding: 'utf8' }).trim().split(/\r?\n/u)[0]!)),
      'bin',
      'bash.exe',
    )
  : 'bash';
const tempDirs: string[] = [];

afterEach(() => {
  for (const tempDir of tempDirs.splice(0)) {
    rmSync(tempDir, { force: true, recursive: true });
  }
});

describe('broker direct deploy automation', () => {
  it('renders distinct database IDs for both bindings while preserving the canonical worker name', () => {
    const outputPath = join(createTempDir(), 'wrangler.production.jsonc');
    runNodeScript(renderWranglerConfigScript, productionRendererArgs(outputPath));

    const renderedConfig = readFileSync(outputPath, 'utf8');
    const config = JSON.parse(renderedConfig);
    expect(config.name).toBe('puripuly-heart-broker');
    expect(config.placement).toEqual({ mode: 'off' });
    expect(config.version_metadata).toEqual({ binding: 'CF_VERSION_METADATA' });
    expect(config.vars.TOKEN_BASELINE_START_UTC).toBe('');
    expect(config.triggers.crons).toEqual(['* * * * *']);
    expect(config.d1_databases).toEqual([
      { binding: 'BROKER_DB', database_name: 'puripuly-heart-broker', database_id: brokerDatabaseId },
      { binding: 'AGENT_USAGE_DB', database_name: 'puripuly-heart-agent-usage', database_id: agentUsageDatabaseId, migrations_dir: '../agent-relay/migrations' },
    ]);
    expect(renderedConfig).not.toContain('REQUIRED_AT_DEPLOY_TIME');
  });

  it.each([true, false])('renders a real UTC start date with explicit off placement present: %s', (explicitOff) => {
    const tempDir = createTempDir();
    const sourcePath = join(tempDir, 'wrangler.source.jsonc');
    const outputPath = join(tempDir, 'wrangler.production.jsonc');
    const source = JSON.parse(readFileSync(checkedInWranglerConfig, 'utf8'));
    if (!explicitOff) delete source.placement;
    writeFileSync(sourcePath, JSON.stringify(source, null, 2));
    runNodeScript(renderWranglerConfigScript, [
      ...productionRendererArgs(outputPath),
      '--source', sourcePath,
      '--token-baseline-start-utc', '2032-02-29',
    ]);
    const config = JSON.parse(readFileSync(outputPath, 'utf8'));
    expect(config.placement).toEqual(explicitOff ? { mode: 'off' } : undefined);
    expect(config.vars.TOKEN_BASELINE_START_UTC).toBe('2032-02-29');
    expect(config.version_metadata).toEqual({ binding: 'CF_VERSION_METADATA' });
  });

  it.each(['2026-02-29', '2032-02-30', '2026-04-31', '2026-13-01', '0000-01-01', '2032-2-29', '2032-02-29T00:00:00Z', '2032-02-29\n', ' '])(
    'rejects a baseline start that is not a real exact UTC date: %j',
    (start) => {
      const outputPath = join(createTempDir(), 'wrangler.production.jsonc');
      expect(() => runNodeScript(renderWranglerConfigScript, [
        ...productionRendererArgs(outputPath),
        '--token-baseline-start-utc', start,
      ])).toThrow(/exact UTC calendar date/u);
      expect(existsSync(outputPath)).toBe(false);
    },
  );

  it.each([
    { mode: 'smart' },
    { mode: 'off', hint: 'aws:ap-northeast-1' },
    { mode: 'targeted', host: 'backend.example.com:443' },
    { mode: 'of' },
    {},
    null,
  ])('rejects non-off placement %j only when a baseline is requested', (placement) => {
    const tempDir = createTempDir();
    const sourcePath = join(tempDir, 'wrangler.source.jsonc');
    const rejectedPath = join(tempDir, 'wrangler.rejected.jsonc');
    const outputPath = join(tempDir, 'wrangler.production.jsonc');
    const source = JSON.parse(readFileSync(checkedInWranglerConfig, 'utf8'));
    source.placement = placement;
    source.vars.TOKEN_BASELINE_START_UTC = '2032-02-29';
    source.vars.UNRELATED_SETTING = 'unchanged';
    writeFileSync(sourcePath, JSON.stringify(source, null, 2));
    expect(() => runNodeScript(renderWranglerConfigScript, [
      ...productionRendererArgs(rejectedPath),
      '--source', sourcePath,
      '--token-baseline-start-utc', '2032-02-29',
    ])).toThrow(/requires placement/u);
    expect(existsSync(rejectedPath)).toBe(false);
    runNodeScript(renderWranglerConfigScript, [
      ...productionRendererArgs(outputPath),
      '--source', sourcePath,
      '--token-baseline-start-utc', '',
    ]);
    const config = JSON.parse(readFileSync(outputPath, 'utf8'));
    expect(config.placement).toEqual(placement);
    expect(config.vars).toMatchObject({ TOKEN_BASELINE_START_UTC: '', UNRELATED_SETTING: 'unchanged' });
  });

  it('substitutes database IDs by binding even when their order changes', () => {
    const tempDir = createTempDir();
    const sourcePath = join(tempDir, 'wrangler.reordered.jsonc');
    const outputPath = join(tempDir, 'wrangler.production.jsonc');
    const source = JSON.parse(readFileSync(checkedInWranglerConfig, 'utf8'));
    source.d1_databases.reverse();
    writeFileSync(sourcePath, JSON.stringify(source, null, 2));
    runNodeScript(renderWranglerConfigScript, [
      ...productionRendererArgs(outputPath),
      '--source', sourcePath,
    ]);
    expect(JSON.parse(readFileSync(outputPath, 'utf8')).d1_databases).toEqual([
      { binding: 'AGENT_USAGE_DB', database_name: 'puripuly-heart-agent-usage', database_id: agentUsageDatabaseId, migrations_dir: '../agent-relay/migrations' },
      { binding: 'BROKER_DB', database_name: 'puripuly-heart-broker', database_id: brokerDatabaseId },
    ]);
  });

  it('fails config rendering if the checked-in worker name stops being canonical', () => {
    const tempDir = createTempDir();
    const sourcePath = join(tempDir, 'wrangler.noncanonical.jsonc');
    const outputPath = join(tempDir, 'wrangler.production.jsonc');
    writeFileSync(sourcePath, readFileSync(checkedInWranglerConfig, 'utf8').replace(
      '"name": "puripuly-heart-broker"',
      '"name": "puripuly-heart-broker-preview"',
    ));
    expect(() => runNodeScript(renderWranglerConfigScript, [
      ...productionRendererArgs(outputPath),
      '--source', sourcePath,
    ])).toThrow(/canonical worker name/i);
    expect(existsSync(outputPath)).toBe(false);
  });

  it.each(['database-id', 'agent-usage-database-id'])('requires a real non-placeholder UUID for --%s', (argument) => {
    const tempDir = createTempDir();
    for (const [index, value] of [undefined, '', ' \t ', 'REQUIRED_AT_DEPLOY_TIME', 'not-a-database-uuid', '00000000-0000-0000-0000-000000000000'].entries()) {
      const outputPath = join(tempDir, `rejected.${index}.jsonc`);
      const args = productionRendererArgs(outputPath);
      args.splice(args.indexOf(`--${argument}`), 2);
      if (value !== undefined) args.push(`--${argument}`, value);
      expect(() => runNodeScript(renderWranglerConfigScript, args)).toThrow();
      expect(existsSync(outputPath)).toBe(false);
    }
  });

  it('rejects a Relay database UUID equal to Broker D1 without regard to case', () => {
    const outputPath = join(createTempDir(), 'wrangler.production.jsonc');
    expect(() => runNodeScript(renderWranglerConfigScript, [
      ...productionRendererArgs(outputPath),
      '--database-id', 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa',
      '--agent-usage-database-id', 'AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA',
    ])).toThrow(/must differ/i);
    expect(existsSync(outputPath)).toBe(false);
  });

  it.each(['BROKER_DB', 'AGENT_USAGE_DB'])('rejects missing, duplicated, or already substituted %s bindings', (binding) => {
    const tempDir = createTempDir();
    const original = JSON.parse(readFileSync(checkedInWranglerConfig, 'utf8')) as {
      d1_databases: Array<{ binding: string; database_id: string }>;
    };
    const entry = original.d1_databases.find((database) => database.binding === binding)!;
    const invalidBindings = [
      original.d1_databases.filter((database) => database.binding !== binding),
      [...original.d1_databases, entry],
      original.d1_databases.map((database) => database.binding === binding ? { ...database, database_id: brokerDatabaseId } : database),
    ];
    for (const [index, databases] of invalidBindings.entries()) {
      const sourcePath = join(tempDir, `wrangler.invalid.${index}.jsonc`);
      const outputPath = join(tempDir, `wrangler.rejected.${index}.jsonc`);
      writeFileSync(sourcePath, JSON.stringify({ ...original, d1_databases: databases }, null, 2));
      expect(() => runNodeScript(renderWranglerConfigScript, [
        ...productionRendererArgs(outputPath),
        '--source', sourcePath,
      ])).toThrow(new RegExp(binding, 'u'));
      expect(existsSync(outputPath)).toBe(false);
    }
  });

  it('renders versioned network identity vars for first deploy and rotation while rejecting mismatched pairs', () => {
    const firstDeployPath = join(createTempDir(), 'wrangler.production.jsonc');
    runNodeScript(renderWranglerConfigScript, productionRendererArgs(firstDeployPath));
    const firstDeployConfig = readFileSync(firstDeployPath, 'utf8');
    expect(firstDeployConfig).toContain('"NETWORK_IDENTITY_HMAC_KEY_VERSION"');
    expect(firstDeployConfig).not.toContain('NETWORK_IDENTITY_HMAC_KEY_VERSION_PREVIOUS');

    const rotationPath = join(createTempDir(), 'wrangler.production.jsonc');
    runNodeScript(renderWranglerConfigScript, [
      ...productionRendererArgs(rotationPath),
      '--network-identity-hmac-key-version', '2',
      '--network-identity-hmac-key-version-previous', '1',
    ]);
    const rotationConfig = readFileSync(rotationPath, 'utf8');
    expect(rotationConfig).toContain('"NETWORK_IDENTITY_HMAC_KEY_VERSION"');
    expect(rotationConfig).toContain('"NETWORK_IDENTITY_HMAC_KEY_VERSION_PREVIOUS"');

    const rejectionDir = createTempDir();
    const rejectionCases = [
      {},
      { current: '0' },
      { current: 'v1' },
      { previous: '1' },
      { current: '2', previous: '2' },
      { current: '2', previous: '0' },
    ];
    for (const [index, versions] of rejectionCases.entries()) {
      const args = productionRendererArgs(join(rejectionDir, `wrangler.rejected.${index}.jsonc`));
      args.splice(args.indexOf('--network-identity-hmac-key-version'), 2);
      if (versions.current) args.push('--network-identity-hmac-key-version', versions.current);
      if (versions.previous) args.push('--network-identity-hmac-key-version-previous', versions.previous);
      expect(() => runNodeScript(renderWranglerConfigScript, args)).toThrow();
    }
  });

  it('renders guarded fingerprint bootstrap SQL that replaces only the placeholder salt', () => {
    const tempDir = createTempDir();
    const outputPath = join(tempDir, 'fingerprint-bootstrap.sql');
    const bootstrapSalt = 'deploy-bootstrap-salt-01';

    runNodeScript(renderFingerprintBootstrapScript, [
      '--out',
      outputPath,
      '--salt',
      bootstrapSalt,
    ]);

    const renderedSql = readFileSync(outputPath, 'utf8');
    const db = new DatabaseSync(':memory:');

    try {
      applyBrokerMigrations(db);
      db.exec(renderedSql);

      const row = db
        .prepare('SELECT value FROM broker_config WHERE key = ?')
        .get('fingerprint_salt') as { value: string };

      expect(JSON.parse(row.value)).toEqual({
        current: {
          version: 1,
          salt: bootstrapSalt,
        },
        previous: null,
        rotated_at: null,
      });
    } finally {
      db.close();
    }
  });

  it('leaves the fingerprint salt unchanged when the placeholder has already been replaced', () => {
    const tempDir = createTempDir();
    const outputPath = join(tempDir, 'fingerprint-bootstrap.sql');

    runNodeScript(renderFingerprintBootstrapScript, [
      '--out',
      outputPath,
      '--salt',
      'deploy-bootstrap-salt-02',
    ]);

    const renderedSql = readFileSync(outputPath, 'utf8');
    const db = new DatabaseSync(':memory:');

    try {
      applyBrokerMigrations(db);
      db.prepare('UPDATE broker_config SET value = ? WHERE key = ?').run(
        JSON.stringify({
          current: {
            version: 1,
            salt: 'already-bootstrapped',
          },
          previous: null,
          rotated_at: null,
        }),
        'fingerprint_salt',
      );

      db.exec(renderedSql);

      const row = db
        .prepare('SELECT value FROM broker_config WHERE key = ?')
        .get('fingerprint_salt') as { value: string };

      expect(JSON.parse(row.value)).toEqual({
        current: {
          version: 1,
          salt: 'already-bootstrapped',
        },
        previous: null,
        rotated_at: null,
      });
    } finally {
      db.close();
    }
  });

  it.skipIf(!bashAvailable)('renders the cleanup workflow with both database IDs and configured identity versions', () => {
    const script = extractWorkflowRunBlock(readFileSync(cleanupWorkflow, 'utf8'), 'Render production Wrangler config');
    const scenario = createDailyCapScenario({
      NETWORK_IDENTITY_HMAC_KEY_VERSION_PRODUCTION: '9',
      NETWORK_IDENTITY_HMAC_KEY_VERSION_PREVIOUS_PRODUCTION: '8',
    });
    const result = runBashScript(script, scenario.tempDir, scenario.env);
    expect(result.status, result.stderr).toBe(0);
    const config = JSON.parse(readFileSync(join(scenario.tempDir, 'wrangler.production.jsonc'), 'utf8'));
    expect(config.d1_databases).toEqual([
      { binding: 'BROKER_DB', database_name: 'puripuly-heart-broker', database_id: brokerDatabaseId },
      { binding: 'AGENT_USAGE_DB', database_name: 'puripuly-heart-agent-usage', database_id: agentUsageDatabaseId, migrations_dir: '../agent-relay/migrations' },
    ]);
    expect(config.vars).toMatchObject({
      NETWORK_IDENTITY_HMAC_KEY_VERSION: '9',
      NETWORK_IDENTITY_HMAC_KEY_VERSION_PREVIOUS: '8',
    });
    expect(existsSync(scenario.callLogPath)).toBe(false);
  });

  it.skipIf(!bashAvailable)('renders configured daily-cap maintenance and changes only the cap before reading it back', () => {
    const script = dailyCapWorkflowScript();
    for (const [current, previous, cap] of [
      ['7', undefined, '1'],
      ['12', '11', '100000'],
      ['9', ' \t ', '450'],
    ]) {
      const scenario = createDailyCapScenario({
        NETWORK_IDENTITY_HMAC_KEY_VERSION_PRODUCTION: current,
        NETWORK_IDENTITY_HMAC_KEY_VERSION_PREVIOUS_PRODUCTION: previous,
        TARGET_MAX_COUNT: cap,
      });
      const result = runBashScript(script, scenario.tempDir, scenario.env);
      expect(result.status, result.stderr).toBe(0);
      const config = readFileSync(join(scenario.tempDir, 'wrangler.production.jsonc'), 'utf8');
      expect(config).toContain('"name": "puripuly-heart-broker"');
      expect(config).toContain('"binding": "BROKER_DB"');
      expect(config).toContain('"database_name": "puripuly-heart-broker"');
      expect(config).toContain(`"database_id": "${brokerDatabaseId}"`);
      expect(config).toContain(`"database_id": "${agentUsageDatabaseId}"`);
      expect(config).not.toContain('REQUIRED_AT_DEPLOY_TIME');
      expect(config).toContain(`"NETWORK_IDENTITY_HMAC_KEY_VERSION": "${current}"`);
      if (previous?.trim()) {
        expect(config).toContain(`"NETWORK_IDENTITY_HMAC_KEY_VERSION_PREVIOUS": "${previous}"`);
      } else {
        expect(config).not.toContain('"NETWORK_IDENTITY_HMAC_KEY_VERSION_PREVIOUS"');
      }

      const db = new DatabaseSync(scenario.databasePath);
      try {
        const rows = db.prepare('SELECT key, value, updated_at FROM broker_config ORDER BY key').all();
        expect(rows).toHaveLength(2);
        expect(rows[0].key).toBe('abuse_controls');
        expect(JSON.parse(String(rows[0].value))).toEqual({
          newActiveEntitlementsPerDay: { maxCount: Number(cap), windowDays: 1 },
          preserved: { enabled: true },
        });
        expect(rows[0].updated_at).not.toBe('2000-01-01 00:00:00');
        expect(rows[1]).toEqual({
          key: 'fingerprint_salt',
          value: '{"preserved":"salt"}',
          updated_at: '2000-01-01 00:00:00',
        });
        expect(db.prepare('SELECT value FROM maintenance_sentinel').all()).toEqual([{ value: 'untouched' }]);
      } finally {
        db.close();
      }
      const calls = readFileSync(scenario.callLogPath, 'utf8').trim().split('\n').map(
        (line) => JSON.parse(line) as { args: string[]; results: unknown[] },
      );
      expect(calls).toHaveLength(3);
      expect(calls[0].args).toContain('--command');
      expect(calls[0].results).toEqual([
        { max_count: 333, window_days: 1, updated_at: '2000-01-01 00:00:00' },
      ]);
      expect(calls[1].args).toContain('--file');
      expect(calls[2].args).toContain('--command');
      expect(calls[2].results).toEqual([
        expect.objectContaining({ max_count: Number(cap), window_days: 1 }),
      ]);
    }
  }, 15000);

  it.skipIf(!bashAvailable)('rejects unauthorized or invalid daily-cap inputs before rendering or D1 access', () => {
    const script = dailyCapWorkflowScript();
    const rejections: Array<Record<string, string | undefined>> = [
      { CONFIRM_UPDATE: undefined },
      { CONFIRM_UPDATE: 'update broker daily auth cap ' },
      ...[undefined, '', '0', '-1', '100001', '18446744073709551617', '9'.repeat(1000), '1.5', 'abc', '01'].map(
        (TARGET_MAX_COUNT) => ({ TARGET_MAX_COUNT }),
      ),
      ...['CLOUDFLARE_API_TOKEN', 'CLOUDFLARE_ACCOUNT_ID', 'BROKER_D1_DATABASE_ID_PRODUCTION', 'AGENT_USAGE_D1_DATABASE_ID_PRODUCTION'].flatMap(
        (name) => [{ [name]: undefined }, { [name]: ' \t ' }],
      ),
    ];
    for (const overrides of rejections) {
      const scenario = createDailyCapScenario(overrides);
      const result = runBashScript(script, scenario.tempDir, scenario.env);
      expect(result.status, JSON.stringify(overrides)).toBe(1);
      expect(existsSync(join(scenario.tempDir, 'wrangler.production.jsonc'))).toBe(false);
      expect(existsSync(scenario.callLogPath)).toBe(false);
    }
  }, 15000);

  it.skipIf(!bashAvailable)('rejects missing or invalid configured identity versions before daily-cap D1 access', () => {
    const script = dailyCapWorkflowScript();
    const rejections: Array<Record<string, string | undefined>> = [
      ...[undefined, '', ' ', '0', 'v7', '9007199254740992'].map(
        (NETWORK_IDENTITY_HMAC_KEY_VERSION_PRODUCTION) => ({ NETWORK_IDENTITY_HMAC_KEY_VERSION_PRODUCTION }),
      ),
      ...['0', 'v6', '7'].map(
        (NETWORK_IDENTITY_HMAC_KEY_VERSION_PREVIOUS_PRODUCTION) => ({ NETWORK_IDENTITY_HMAC_KEY_VERSION_PREVIOUS_PRODUCTION }),
      ),
    ];
    for (const overrides of rejections) {
      const scenario = createDailyCapScenario(overrides);
      const result = runBashScript(script, scenario.tempDir, scenario.env);
      expect(result.status, JSON.stringify(overrides)).toBe(1);
      expect(existsSync(join(scenario.tempDir, 'wrangler.production.jsonc'))).toBe(false);
      expect(existsSync(scenario.callLogPath)).toBe(false);
    }
  }, 15000);

  it.skipIf(!bashAvailable)('rejects a missing row, mismatched cap, or wrong daily window in the actual maintenance readback', () => {
    const workflow = readFileSync(abuseControlsWorkflow, 'utf8');
    const render = extractWorkflowRunBlock(workflow, 'Render production Wrangler config');
    const verify = extractWorkflowRunBlock(workflow, 'Verify daily auth cap runtime config');
    for (const sql of [
      "DELETE FROM broker_config WHERE key = 'abuse_controls'",
      "UPDATE broker_config SET value = json_set(value, '$.newActiveEntitlementsPerDay.maxCount', 444) WHERE key = 'abuse_controls'",
      "UPDATE broker_config SET value = json_set(value, '$.newActiveEntitlementsPerDay.windowDays', 2) WHERE key = 'abuse_controls'",
    ]) {
      const scenario = createDailyCapScenario({ TARGET_MAX_COUNT: '333' });
      const db = new DatabaseSync(scenario.databasePath);
      try {
        db.exec(sql);
      } finally {
        db.close();
      }
      expect(runBashScript(render, scenario.tempDir, scenario.env).status).toBe(0);
      const result = runBashScript(verify, scenario.tempDir, scenario.env);
      expect(result.status, sql).toBe(1);
      expect(readFileSync(scenario.callLogPath, 'utf8').trim().split('\n')).toHaveLength(1);
    }
  }, 15000);
});

const bashAvailable = (() => {
  try {
    return spawnSync(bashExecutable, ['--version']).status === 0;
  } catch {
    return false;
  }
})();

function extractWorkflowRunBlock(workflow: string, stepName: string): string {
  const stepIndex = workflow.indexOf(`- name: ${stepName}`);
  expect(stepIndex).toBeGreaterThanOrEqual(0);

  const runMatch = workflow.slice(stepIndex).match(/\n        run: (\||>-)\n/u);
  expect(runMatch).not.toBeNull();
  const runIndex = stepIndex + runMatch!.index! + runMatch![0].length;

  const body: string[] = [];
  for (const line of workflow.slice(runIndex).split('\n')) {
    if (line !== '' && !line.startsWith('          ')) {
      break;
    }
    body.push(line.startsWith('          ') ? line.slice(10) : line);
  }
  while (body.length > 0 && body[body.length - 1] === '') {
    body.pop();
  }
  return `${body.join(runMatch![1] === '>-' ? ' ' : '\n')}\n`;
}

function dailyCapWorkflowScript(): string {
  const workflow = readFileSync(abuseControlsWorkflow, 'utf8');
  let previousIndex = -1;
  return [
    'Guard production runtime-config update inputs',
    'Render production Wrangler config',
    'Inspect current daily auth cap',
    'Update daily auth cap runtime config',
    'Verify daily auth cap runtime config',
  ].map((name) => {
    const index = workflow.indexOf(`- name: ${name}`);
    expect(index).toBeGreaterThan(previousIndex);
    previousIndex = index;
    return extractWorkflowRunBlock(workflow, name);
  }).join('\n');
}

function createDailyCapScenario(overrides: Record<string, string | undefined>) {
  const tempDir = createRepoTempDir();
  mkdirSync(join(tempDir, 'scripts'));
  writeFileSync(
    join(tempDir, 'scripts', 'render-production-wrangler-config.mjs'),
    readFileSync(renderWranglerConfigScript),
  );
  writeFileSync(join(tempDir, 'wrangler.jsonc'), readFileSync(checkedInWranglerConfig));
  const databasePath = join(tempDir, 'synthetic-d1.sqlite');
  const callLogPath = join(tempDir, 'd1-calls.jsonl');
  const db = new DatabaseSync(databasePath);
  try {
    db.exec(`
      CREATE TABLE broker_config (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL);
      INSERT INTO broker_config VALUES
        ('abuse_controls', '{"newActiveEntitlementsPerDay":{"maxCount":333,"windowDays":1},"preserved":{"enabled":true}}', '2000-01-01 00:00:00'),
        ('fingerprint_salt', '{"preserved":"salt"}', '2000-01-01 00:00:00');
      CREATE TABLE maintenance_sentinel (value TEXT NOT NULL);
      INSERT INTO maintenance_sentinel VALUES ('untouched');
    `);
  } finally {
    db.close();
  }
  writeFileSync(join(tempDir, 'bash-env.sh'), 'pnpm() { node ./synthetic-d1.mjs "$@"; }\n');
  writeFileSync(join(tempDir, 'synthetic-d1.mjs'), `
import assert from 'node:assert/strict';
import { appendFileSync, readFileSync } from 'node:fs';
import { DatabaseSync } from 'node:sqlite';
const args = process.argv.slice(2);
assert.deepEqual(args.slice(0, 5), ['exec', 'wrangler', 'd1', 'execute', 'puripuly-heart-broker']);
assert(args.includes('--remote'));
assert(args.includes('--json'));
assert.equal(args[args.indexOf('--config') + 1], 'wrangler.production.jsonc');
const config = readFileSync('wrangler.production.jsonc', 'utf8');
assert(config.includes('"database_id": "${brokerDatabaseId}"'));
assert(config.includes('"database_id": "${agentUsageDatabaseId}"'));
const db = new DatabaseSync('synthetic-d1.sqlite');
let results = [];
try {
  if (args.includes('--file')) {
    assert(args.includes('--yes'));
    db.exec(readFileSync(args[args.indexOf('--file') + 1], 'utf8'));
  } else {
    assert(args.includes('--command'));
    results = db.prepare(args[args.indexOf('--command') + 1]).all();
  }
} finally {
  db.close();
}
appendFileSync('d1-calls.jsonl', JSON.stringify({ args, results }) + '\\n');
process.stdout.write(JSON.stringify([{ success: true, results }]));
`);
  return {
    tempDir,
    databasePath,
    callLogPath,
    env: {
      BASH_ENV: './bash-env.sh',
      CONFIRM_UPDATE: 'update broker daily auth cap',
      TARGET_MAX_COUNT: '450',
      CLOUDFLARE_API_TOKEN: 'daily-cap-synthetic-token',
      CLOUDFLARE_ACCOUNT_ID: 'daily-cap-synthetic-account',
      BROKER_D1_DATABASE_ID_PRODUCTION: brokerDatabaseId,
      AGENT_USAGE_D1_DATABASE_ID_PRODUCTION: agentUsageDatabaseId,
      NETWORK_IDENTITY_HMAC_KEY_VERSION_PRODUCTION: '7',
      NETWORK_IDENTITY_HMAC_KEY_VERSION_PREVIOUS_PRODUCTION: undefined,
      ...overrides,
    },
  };
}
function runBashScript(
  script: string,
  cwd: string,
  overrides: Record<string, string | undefined> = {},
): { status: number | null; stderr: string } {
  const env: NodeJS.ProcessEnv = { ...process.env };
  const assignments: string[] = [];
  for (const [key, value] of Object.entries(overrides)) {
    if (value === undefined) {
      delete env[key];
      assignments.push(`unset ${key}`);
    } else {
      env[key] = value;
      assignments.push(`${key}='${value.replace(/'/g, `'"'"'`)}'`);
    }
  }
  const bashCwd = relative(process.cwd(), cwd).split(sep).join('/');
  const quotedBashCwd = bashCwd.replace(/'/g, `'"'"'`);
  const sourceBashEnv = overrides.BASH_ENV === undefined ? '' : 'source "$BASH_ENV"\n';
  const nodePath = process.execPath.split(sep).join('/').replace(/'/g, `'"'"'`);
  const runtimeBindings = `node() { '${nodePath}' "$@"; }\n` +
    (process.platform === 'win32' ? 'python3() { python "$@"; }\n' : '');
  const scriptPath = join(cwd, 'workflow-step-scenario.sh');
  writeFileSync(
    scriptPath,
    `cd '${quotedBashCwd}'\n${runtimeBindings}${assignments.join('\n')}\n${sourceBashEnv}${script}`,
    'utf8',
  );
  const bashPath = relative(process.cwd(), scriptPath).split(sep).join('/');
  const result = spawnSync(
    bashExecutable,
    ['--noprofile', '--norc', '-e', '-o', 'pipefail', bashPath],
    {
      encoding: 'utf8',
      env,
    },
  );
  return { status: result.status, stderr: result.stderr };
}

function createRepoTempDir(): string {
  const tempDir = new URL(
    `./.tmp-deploy-guard-${Date.now().toString(36)}-${Math.floor(Math.random() * 0xffffff).toString(36)}/`,
    import.meta.url,
  );
  rmSync(tempDir, { force: true, recursive: true });
  mkdirSync(fileURLToPath(tempDir), { recursive: true });
  tempDirs.push(fileURLToPath(tempDir));
  return fileURLToPath(tempDir);
}


function createTempDir(): string {
  const tempDir = mkdtempSync(join(tmpdir(), 'broker-direct-deploy-'));
  tempDirs.push(tempDir);
  return tempDir;
}

function productionRendererArgs(outputPath: string): string[] {
  return [
    '--source', fileURLToPath(checkedInWranglerConfig),
    '--out', outputPath,
    '--database-id', brokerDatabaseId,
    '--agent-usage-database-id', agentUsageDatabaseId,
    '--network-identity-hmac-key-version', '1',
  ];
}

function runNodeScript(scriptUrl: URL, args: string[]): string {
  return execFileSync(process.execPath, [fileURLToPath(scriptUrl), ...args], {
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'pipe'],
  });
}

