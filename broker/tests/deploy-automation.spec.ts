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
const abuseControlsWorkflow = new URL(
  '../../.github/workflows/maintenance-broker-abuse-controls.yml',
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
  it('renders a deploy-time wrangler config with the production database_id while preserving the canonical worker name', () => {
    const tempDir = createTempDir();
    const outputPath = join(tempDir, 'wrangler.production.jsonc');

    runNodeScript(renderWranglerConfigScript, [
      '--source',
      fileURLToPath(checkedInWranglerConfig),
      '--out',
      outputPath,
      '--database-id',
      'production-d1-database-id',
      '--network-identity-hmac-key-version',
      '1',
    ]);

    const renderedConfig = readFileSync(outputPath, 'utf8');
    expect(renderedConfig).toContain('"name": "puripuly-heart-broker"');
    expect(renderedConfig).toContain('"database_id": "production-d1-database-id"');
    expect(renderedConfig).not.toContain('REQUIRED_AT_DEPLOY_TIME');
  });

  it('fails config rendering if the checked-in worker name stops being canonical', () => {
    const tempDir = createTempDir();
    const sourcePath = join(tempDir, 'wrangler.noncanonical.jsonc');
    const outputPath = join(tempDir, 'wrangler.production.jsonc');

    writeFileSync(
      sourcePath,
      readFileSync(checkedInWranglerConfig, 'utf8').replace(
        '"name": "puripuly-heart-broker"',
        '"name": "puripuly-heart-broker-preview"',
      ),
      'utf8',
    );

    expect(() =>
      runNodeScript(renderWranglerConfigScript, [
        '--source',
        sourcePath,
        '--out',
        outputPath,
        '--database-id',
        'production-d1-database-id',
        '--network-identity-hmac-key-version',
        '1',
      ]),
    ).toThrow(/canonical worker name/i);
  });

  it('renders versioned network identity vars for first deploy and rotation while rejecting mismatched pairs', () => {
    const firstDeployDir = createTempDir();
    const firstDeployPath = join(firstDeployDir, 'wrangler.production.jsonc');
    runNodeScript(renderWranglerConfigScript, [
      '--source',
      fileURLToPath(checkedInWranglerConfig),
      '--out',
      firstDeployPath,
      '--database-id',
      'production-d1-database-id',
      '--network-identity-hmac-key-version',
      '1',
    ]);
    const firstDeployConfig = readFileSync(firstDeployPath, 'utf8');
    expect(firstDeployConfig).toContain('"NETWORK_IDENTITY_HMAC_KEY_VERSION"');
    expect(firstDeployConfig).not.toContain('NETWORK_IDENTITY_HMAC_KEY_VERSION_PREVIOUS');

    const rotationDir = createTempDir();
    const rotationPath = join(rotationDir, 'wrangler.production.jsonc');
    runNodeScript(renderWranglerConfigScript, [
      '--source',
      fileURLToPath(checkedInWranglerConfig),
      '--out',
      rotationPath,
      '--database-id',
      'production-d1-database-id',
      '--network-identity-hmac-key-version',
      '2',
      '--network-identity-hmac-key-version-previous',
      '1',
    ]);
    const rotationConfig = readFileSync(rotationPath, 'utf8');
    expect(rotationConfig).toContain('"NETWORK_IDENTITY_HMAC_KEY_VERSION"');
    expect(rotationConfig).toContain('"NETWORK_IDENTITY_HMAC_KEY_VERSION_PREVIOUS"');

    const rejectionDir = createTempDir();
    const rejectionCases: string[][] = [
      ['--database-id', 'production-d1-database-id'],
      [
        '--database-id',
        'production-d1-database-id',
        '--network-identity-hmac-key-version',
        '0',
      ],
      [
        '--database-id',
        'production-d1-database-id',
        '--network-identity-hmac-key-version',
        'v1',
      ],
      [
        '--database-id',
        'production-d1-database-id',
        '--network-identity-hmac-key-version-previous',
        '1',
      ],
      [
        '--database-id',
        'production-d1-database-id',
        '--network-identity-hmac-key-version',
        '2',
        '--network-identity-hmac-key-version-previous',
        '2',
      ],
      [
        '--database-id',
        'production-d1-database-id',
        '--network-identity-hmac-key-version',
        '2',
        '--network-identity-hmac-key-version-previous',
        '0',
      ],
    ];
    for (const [caseIndex, rejectionArgs] of rejectionCases.entries()) {
      expect(() =>
        runNodeScript(renderWranglerConfigScript, [
          '--source',
          fileURLToPath(checkedInWranglerConfig),
          '--out',
          join(rejectionDir, `wrangler.rejected.${caseIndex}.jsonc`),
          ...rejectionArgs,
        ]),
      ).toThrow();
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
      expect(config).toContain('"database_id": "daily-cap-synthetic-database"');
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
      ...['CLOUDFLARE_API_TOKEN', 'CLOUDFLARE_ACCOUNT_ID', 'BROKER_D1_DATABASE_ID_PRODUCTION'].flatMap(
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
assert(config.includes('"database_id": "daily-cap-synthetic-database"'));
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
      BROKER_D1_DATABASE_ID_PRODUCTION: 'daily-cap-synthetic-database',
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

function runNodeScript(scriptUrl: URL, args: string[]): string {
  return execFileSync(process.execPath, [fileURLToPath(scriptUrl), ...args], {
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'pipe'],
  });
}

