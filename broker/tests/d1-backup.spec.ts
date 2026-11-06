import { execFileSync, spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { chmodSync, copyFileSync, existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { DatabaseSync } from 'node:sqlite';

import { afterEach, describe, expect, it } from 'vitest';

const workflow = readFileSync(new URL('../../.github/workflows/deploy-broker-direct.yml', import.meta.url), 'utf8');
const verifier = new URL('../scripts/verify-persisted-d1-backup.sh', import.meta.url);
const tempDirs: string[] = [];
const sql = "CREATE TABLE installations (id TEXT PRIMARY KEY, hardware_hash TEXT);\n" +
  "INSERT INTO installations VALUES ('synthetic-installation-366', 'synthetic-hardware-366');\n" +
  "CREATE TABLE broker_config (key TEXT PRIMARY KEY, value TEXT);\n" +
  "INSERT INTO broker_config VALUES ('synthetic-config-366', '{\"enabled\":true}');\n";
const signedDownloadUrl = 'https://synthetic.invalid/d1/export.sql?signature=synthetic-capability-366';
const exporterDiagnostic = 'synthetic-sensitive-diagnostic-366';
const exporterDiagnostics = `${signedDownloadUrl}\n${sql}${exporterDiagnostic}`;
const deletionConfirmation = 'delete retired translation telemetry from production D1';
const stepNames = [
  'Export pre-migration production D1 backup',
  'Upload pre-migration D1 backup',
  'Verify persisted pre-migration D1 backup',
  'Apply staged pre-backfill D1 migrations',
  'Apply explicitly authorized legacy telemetry deletion',
];
const steps = workflow.split(/(?=^      - name: )/mu).filter(
  (block) => stepNames.includes(block.match(/^      - name: (.+)/u)?.[1] ?? ''),
);

afterEach(() => {
  for (const directory of tempDirs.splice(0)) rmSync(directory, { force: true, recursive: true });
});

describe('protected pre-migration D1 backup', () => {
  it('persists only protected bytes before the earliest mutation and recovers equivalent SQLite data from that persisted object', () => {
    const scenario = createScenario();
    const result = runBackupFlow(scenario);
    expect(result.status, result.stderr).toBe(0);
    expect(scenario.events).toEqual(['export', 'persist', 'migration', 'deletion']);
    const persisted = readFileSync(scenario.persistedPath);
    expect(persisted.equals(readFileSync(scenario.outputs.d1_backup!.backup_path!))).toBe(true);
    expect(persisted.includes(Buffer.from('synthetic-installation-366'))).toBe(false);
    expect(persisted.includes(Buffer.from('synthetic-hardware-366'))).toBe(false);
    expect(persisted.includes(Buffer.from('synthetic-config-366'))).toBe(false);
    expect(persisted.subarray(0, 22).toString()).toBe('age-encryption.org/v1\n');
    const recovered = execFileSync('age', ['--decrypt', '--identity', scenario.identity, scenario.persistedPath], { encoding: 'utf8' });
    expect(sqliteRecords(recovered)).toEqual(sqliteRecords(sql));
    expect(readFileSync(scenario.permissions, 'utf8').trim()).toBe('700 600');
    expect(readdirSync(join(scenario.cwd, '.deploy-direct/staged-migrations-legacy-telemetry-deletion'))).toEqual(['0025_drop_legacy_translation_telemetry.sql']);
    expect(readFileSync(scenario.pnpmLog, 'utf8').trim().split('\n')).toEqual([
      'exec wrangler d1 export puripuly-heart-broker --remote --config synthetic.jsonc --output ' + readFileSync(scenario.exportPath, 'utf8').trim(),
      'exec wrangler d1 migrations apply puripuly-heart-broker --remote --config pre-backfill.jsonc',
      'exec wrangler types --config legacy-deletion.jsonc',
      'exec wrangler d1 migrations apply puripuly-heart-broker --remote --config legacy-deletion.jsonc',
    ]);
    expectNoPlaintext(scenario);
    const unauthorized = createScenario();
    const denied = spawnSync('age', ['--decrypt', '--identity', unauthorized.identity, scenario.persistedPath], { encoding: 'utf8' });
    expect(denied.status).not.toBe(0);
    expect(denied.stdout).toBe('');
    const cleanup = runShell(extractRun(workflow.split(/(?=^      - name: )/mu).find((block) => block.startsWith('      - name: Clean up local D1 backup files'))!), scenario);
    expect(cleanup.status, cleanup.stderr).toBe(0);
    expect(existsSync(scenario.outputs.d1_backup!.backup_path!)).toBe(false);
    expect(readFileSync(scenario.persistedPath).equals(persisted)).toBe(true);
  });

  it.each([
    ['success', 0],
    ['export-failure', 23],
  ])('keeps exporter stdout, stderr and debug-file diagnostics private on %s', (mode, status) => {
    const scenario = createScenario();
    const result = runBackupFlow(scenario, { mode });
    expect(result.status, result.stderr).toBe(status);
    expect(scenario.publicOutput.stdout).toBe('');
    for (const sensitive of [signedDownloadUrl, 'synthetic-capability-366', sql, exporterDiagnostic]) {
      expect(scenario.publicOutput.stderr).not.toContain(sensitive);
    }
    expect(readFileSync(scenario.debugPermissions, 'utf8').trim()).toBe('700 600');
    expect(existsSync(scenario.env.DEFAULT_WRANGLER_LOG_PATH!)).toBe(false);
    expectNoPlaintext(scenario);
    if (status === 0) {
      expect(scenario.publicOutput.stderr).toBe('');
      expect(scenario.events).toEqual(['export', 'persist', 'migration', 'deletion']);
      expect(readdirSync(join(scenario.root, 'local-artifact'))).toEqual(['backup.sql.age']);
      const recovered = execFileSync('age', ['--decrypt', '--identity', scenario.identity, scenario.persistedPath], { encoding: 'utf8' });
      expect(sqliteRecords(recovered)).toEqual(sqliteRecords(sql));
      expect(recovered).not.toContain(signedDownloadUrl);
      expect(recovered).not.toContain(exporterDiagnostic);
    } else {
      expect(scenario.publicOutput.stderr).toMatch(/D1 export failed \(exit status 23\)/u);
      expect(scenario.events).toEqual(['export']);
      expect(scenario.outputs.d1_backup).toBeUndefined();
      expect(existsSync(scenario.persistedPath)).toBe(false);
      expect(readdirSync(scenario.runnerTemp)).toEqual([]);
    }
  });

  it.each([
    undefined,
    '',
    '   ',
    'not-an-age-recipient',
    'age1' + 'q'.repeat(58),
    'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAITest',
  ])('rejects unusable required public recipient %j before export', (recipient) => {
    const scenario = createScenario();
    const result = runBackupFlow(scenario, { recipient });
    expect(result.status).not.toBe(0);
    expect(scenario.events).toEqual([]);
    expect(existsSync(scenario.pnpmLog)).toBe(false);
    expect(scenario.outputs.d1_backup).toBeUndefined();
    expectNoPlaintext(scenario);
  });

  it.each([
    ['empty', 1],
    ['export-failure', 23],
    ['protection-failure', 17],
    ['empty-protection', 1],
  ])('stops %s before persistence or any mutation and removes plaintext and partial ciphertext', (mode, status) => {
    const scenario = createScenario();
    const result = runBackupFlow(scenario, { mode });
    expect(result.status, result.stderr).toBe(status);
    expect(scenario.events).toEqual(['export']);
    expect(scenario.outputs.d1_backup).toBeUndefined();
    expect(readdirSync(scenario.runnerTemp)).toEqual([]);
    expectNoPlaintext(scenario);
  });

  it.each(['HUP', 'INT', 'TERM'])('cleans plaintext and withholds protected output on %s', (signal) => {
    const scenario = createScenario();
    const result = runBackupFlow(scenario, { mode: 'signal', signal });
    expect(result.status).not.toBe(0);
    expect(scenario.events).toEqual(['export']);
    expect(scenario.outputs.d1_backup).toBeUndefined();
    expect(readdirSync(scenario.runnerTemp)).toEqual([]);
    expectNoPlaintext(scenario);
  });

  it('stops artifact persistence failure before the earliest migration and deletion', () => {
    const scenario = createScenario();
    const result = runBackupFlow(scenario, { persistenceFailure: true });
    expect(result.status).toBe(71);
    expect(scenario.events).toEqual(['export']);
    expect(existsSync(scenario.persistedPath)).toBe(false);
    expectNoPlaintext(scenario);
  });

  it.each([
    { D1_BACKUP_UPLOAD_OUTCOME: 'failure' },
    { D1_BACKUP_UPLOAD_OUTCOME: 'skipped' },
    { D1_BACKUP_ARTIFACT_ID: '' },
    { D1_BACKUP_ARTIFACT_ID: '0' },
    { D1_BACKUP_ARTIFACT_DIGEST: '' },
    { D1_BACKUP_ARTIFACT_DIGEST: 'invalid' },
    { D1_BACKUP_SHA256: '' },
  ])('rejects incomplete persistence or protection evidence %j before the earliest migration and deletion', (evidence) => {
    const scenario = createScenario();
    const result = runBackupFlow(scenario, { evidence });
    expect(result.status).not.toBe(0);
    expect(scenario.events).toEqual(['export', 'persist']);
    expectNoPlaintext(scenario);
    const deletion = runDeletion(scenario, evidence);
    expect(deletion.status).not.toBe(0);
    expect(existsSync(join(scenario.cwd, '.deploy-direct'))).toBe(false);
  });

  it.each(['missing', 'tampered', 'arbitrary', 'replacement-path', 'symlink'])('rejects %s backup before the earliest migration and authorized deletion', (replacement) => {
    const scenario = createScenario();
    const result = runBackupFlow(scenario, { replacement });
    expect(result.status).not.toBe(0);
    expect(scenario.events).toEqual(['export', 'persist']);
    expectNoPlaintext(scenario);
    const deletion = runDeletion(scenario);
    expect(deletion.status).not.toBe(0);
    expect(existsSync(join(scenario.cwd, '.deploy-direct'))).toBe(false);
  });

  it.each([
    { enabled: undefined, confirmation: undefined, status: 0 },
    { enabled: 'false', confirmation: undefined, status: 0 },
    { enabled: 'true', confirmation: 'wrong confirmation', status: 1 },
  ])('preserves separate deletion authorization %j without staging or remote calls', ({ enabled, confirmation, status }) => {
    const scenario = createScenario();
    const result = runDeletion(scenario, {
      DELETE_LEGACY_TRANSLATION_TELEMETRY: enabled,
      CONFIRM_LEGACY_TRANSLATION_TELEMETRY_DELETION: confirmation,
    });
    expect(result.status, result.stderr).toBe(status);
    expect(existsSync(join(scenario.cwd, '.deploy-direct'))).toBe(false);
    expect(existsSync(scenario.pnpmLog)).toBe(false);
  });
});

function createScenario() {
  const root = mkdtempSync(join(tmpdir(), 'broker-d1-backup-'));
  tempDirs.push(root);
  const cwd = join(root, 'broker');
  const runnerTemp = join(root, 'runner');
  const bin = join(root, 'bin');
  const operator = join(root, 'operator');
  for (const directory of [cwd, runnerTemp, bin, operator, join(cwd, 'scripts'), join(cwd, 'migrations')]) mkdirSync(directory, { recursive: true });
  copyFileSync(verifier, join(cwd, 'scripts/verify-persisted-d1-backup.sh'));
  writeFileSync(join(cwd, 'migrations/0025_drop_legacy_translation_telemetry.sql'), 'DROP TABLE IF EXISTS telemetry_active_days;\n');
  const identity = join(operator, 'identity');
  execFileSync('age-keygen', ['--output', identity], { stdio: ['ignore', 'ignore', 'pipe'] });
  const recipient = execFileSync('age-keygen', ['-y', identity], { encoding: 'utf8' }).trim();
  const realAge = execFileSync('bash', ['-c', 'command -v age'], { encoding: 'utf8' }).trim();
  const seed = join(root, 'synthetic.sql');
  writeFileSync(seed, sql);
  const pnpmLog = join(root, 'pnpm.log');
  const permissions = join(root, 'permissions');
  const exportPath = join(root, 'export-path');
  const debugPath = join(root, 'debug-path');
  const debugPermissions = join(root, 'debug-permissions');
  writeFileSync(join(bin, 'pnpm'), '#!/bin/bash\nset -eu\nprintf "%s\\n" "$*" >> "$PNPM_LOG"\n' +
    'if [[ "$*" == "exec wrangler d1 export "* ]]; then\n' +
    '  while [[ "$1" != --output ]]; do shift; done\n' +
    '  cp "$SYNTHETIC_SQL" "$2"\n' +
    '  printf "%s\\n" "$2" > "$EXPORT_PATH"\n' +
    '  printf "%s %s\\n" "$(stat -c %a "$(dirname "$2")")" "$(stat -c %a "$2")" > "$PERMISSIONS"\n' +
    '  printf "%s\\n" "$SYNTHETIC_DIAGNOSTICS"\n' +
    '  printf "%s\\n" "$SYNTHETIC_DIAGNOSTICS" >&2\n' +
    '  debug_path="${WRANGLER_LOG_PATH:-$DEFAULT_WRANGLER_LOG_PATH}"\n' +
    '  mkdir -p "$(dirname "$debug_path")"\n' +
    '  printf "%s\\n" "$SYNTHETIC_DIAGNOSTICS" > "$debug_path"\n' +
    '  printf "%s\\n" "$debug_path" > "$EXPORT_DEBUG_PATH"\n' +
    '  printf "%s %s\\n" "$(stat -c %a "$(dirname "$debug_path")")" "$(stat -c %a "$debug_path")" > "$DEBUG_PERMISSIONS"\n' +
    '  case "$MODE" in\n' +
    '    empty) : > "$2" ;;\n' +
    '    export-failure) exit 23 ;;\n' +
    '    signal) kill -s "$TEST_SIGNAL" "$BACKUP_SHELL_PID" ;;\n' +
    '  esac\n' +
    'elif [[ "$*" != "exec wrangler types "* && "$*" != "exec wrangler d1 migrations apply "* ]]; then\n' +
    '  exit 99\n' +
    'fi\n');
  writeFileSync(join(bin, 'age'), '#!/bin/bash\nset -eu\n' +
    'if [[ "${@: -1}" != /dev/null ]]; then\n' +
    '  case "$MODE" in\n' +
    '    protection-failure|empty-protection)\n' +
    '      while [[ "$1" != --output ]]; do shift; done\n' +
    '      : > "$2"\n' +
    '      if [[ "$MODE" == protection-failure ]]; then printf partial > "$2"; exit 17; fi\n' +
    '      exit 0 ;;\n' +
    '  esac\n' +
    'fi\nexec "$REAL_AGE" "$@"\n');
  chmodSync(join(bin, 'pnpm'), 0o700);
  chmodSync(join(bin, 'age'), 0o700);
  return {
    root, cwd, runnerTemp, identity, recipient, pnpmLog, permissions, exportPath, debugPath, debugPermissions,
    persistedPath: join(root, 'local-artifact/backup.sql.age'),
    events: [] as string[],
    outputs: {} as Record<string, Record<string, string>>,
    outcomes: {} as Record<string, string>,
    publicOutput: { stdout: '', stderr: '' },
    env: {
      ...process.env,
      PATH: `${bin}:${process.env.PATH}`,
      RUNNER_TEMP: runnerTemp,
      GITHUB_RUN_ID: 'synthetic-366',
      GITHUB_RUN_ATTEMPT: '1',
      SYNTHETIC_SQL: seed,
      SYNTHETIC_DIAGNOSTICS: exporterDiagnostics,
      DEFAULT_WRANGLER_LOG_PATH: join(root, 'default-wrangler-logs/export.log'),
      EXPORT_DEBUG_PATH: debugPath,
      DEBUG_PERMISSIONS: debugPermissions,
      PNPM_LOG: pnpmLog,
      PERMISSIONS: permissions,
      EXPORT_PATH: exportPath,
      REAL_AGE: realAge,
      MODE: 'success',
      TEST_SIGNAL: 'TERM',
      CLOUDFLARE_API_TOKEN: 'synthetic-not-a-credential',
      CLOUDFLARE_ACCOUNT_ID: 'synthetic-not-an-account',
      DELETE_LEGACY_TRANSLATION_TELEMETRY: 'true',
      CONFIRM_LEGACY_TRANSLATION_TELEMETRY_DELETION: deletionConfirmation,
    } as NodeJS.ProcessEnv,
  };
}

interface Scenario {
  root: string;
  cwd: string;
  runnerTemp: string;
  identity: string;
  recipient: string;
  pnpmLog: string;
  permissions: string;
  exportPath: string;
  debugPath: string;
  debugPermissions: string;
  persistedPath: string;
  events: string[];
  outputs: Record<string, Record<string, string>>;
  outcomes: Record<string, string>;
  env: NodeJS.ProcessEnv;
  publicOutput: { stdout: string; stderr: string };
}
type Options = {
  recipient?: string;
  mode?: string;
  signal?: string;
  persistenceFailure?: boolean;
  evidence?: Record<string, string | undefined>;
  replacement?: string;
};

function runBackupFlow(scenario: Scenario, options: Options = {}) {
  scenario.env.MODE = options.mode ?? 'success';
  scenario.env.TEST_SIGNAL = options.signal ?? 'TERM';
  const recipient = Object.hasOwn(options, 'recipient') ? options.recipient : scenario.recipient;
  for (const block of steps) {
    const name = block.match(/^      - name: (.+)/u)![1]!;
    const id = block.match(/^        id: (.+)/mu)?.[1];
    if (name === 'Upload pre-migration D1 backup') {
      if (options.persistenceFailure) return { status: 71, stderr: 'synthetic artifact persistence failure' };
      const selected = resolveExpression(block.match(/^          path: (.+)/mu)![1]!, scenario, recipient);
      if (!selected || !existsSync(selected)) return { status: 72, stderr: 'artifact input missing' };
      mkdirSync(join(scenario.root, 'local-artifact'));
      copyFileSync(selected, scenario.persistedPath);
      scenario.outputs[id!] = { 'artifact-id': '366', 'artifact-digest': createHash('sha256').update(readFileSync(scenario.persistedPath)).digest('hex') };
      scenario.outcomes[id!] = 'success';
      scenario.events.push('persist');
      replaceBackup(scenario, options.replacement);
      continue;
    }
    const outputFile = join(scenario.root, 'github-output');
    writeFileSync(outputFile, '');
    const env = resolveStepEnv(block, scenario, recipient);
    if (name.includes('Verify persisted') || name.includes('authorized legacy')) Object.assign(env, options.evidence);
    const result = runShell(resolveExpression(extractRun(block), scenario, recipient), scenario, { ...env, GITHUB_OUTPUT: outputFile });
    if (name.startsWith('Export') && existsSync(scenario.pnpmLog)) scenario.events.push('export');
    if (result.status !== 0) return result;
    if (id) {
      scenario.outputs[id] = Object.fromEntries(readFileSync(outputFile, 'utf8').trim().split('\n').filter(Boolean).map((line) => {
        const index = line.indexOf('=');
        return [line.slice(0, index), line.slice(index + 1)];
      }));
      scenario.outcomes[id] = 'success';
    }
    if (name.startsWith('Apply staged')) scenario.events.push('migration');
    if (name.startsWith('Apply explicitly')) scenario.events.push('deletion');
  }
  return { status: 0, stderr: '' };
}

function replaceBackup(scenario: Scenario, replacement?: string) {
  if (!replacement) return;
  const path = scenario.outputs.d1_backup!.backup_path!;
  if (replacement === 'missing') rmSync(path);
  if (replacement === 'tampered') {
    const bytes = readFileSync(path);
    bytes[bytes.length - 1] = bytes[bytes.length - 1]! ^ 1;
    writeFileSync(path, bytes);
  }
  if (replacement === 'arbitrary') writeFileSync(path, 'arbitrary nonempty replacement\n');
  if (replacement === 'replacement-path') {
    const other = join(scenario.runnerTemp, 'arbitrary.sql.age');
    copyFileSync(path, other);
    scenario.outputs.d1_backup!.backup_path = other;
  }
  if (replacement === 'symlink') {
    rmSync(path);
    symlinkSync(scenario.persistedPath, path);
  }
}

function runDeletion(scenario: Scenario, overrides: Record<string, string | undefined> = {}) {
  const block = steps.find((step) => step.startsWith('      - name: Apply explicitly'))!;
  return runShell(extractRun(block), scenario, {
    ...resolveStepEnv(block, scenario, scenario.recipient),
    GITHUB_OUTPUT: join(scenario.root, 'deletion-output'),
    ...overrides,
  });
}

function resolveStepEnv(block: string, scenario: Scenario, recipient?: string) {
  const env: Record<string, string | undefined> = {};
  for (const match of block.matchAll(/^          ([A-Z_0-9]+): (.+)$/gmu)) {
    env[match[1]!] = resolveExpression(match[2]!, scenario, recipient);
  }
  return env;
}

function resolveExpression(value: string, scenario: Scenario, recipient?: string): string {
  return value.replace(/\$\{\{ (.*?) \}\}/gu, (_, expression: string) => {
    if (expression === 'vars.BROKER_D1_BACKUP_AGE_RECIPIENT_PRODUCTION') return recipient ?? '';
    if (expression === 'steps.render.outputs.config_path') return 'synthetic.jsonc';
    if (expression === 'steps.render.outputs.pre_backfill_config_path') return 'pre-backfill.jsonc';
    if (expression === 'steps.render.outputs.legacy_deletion_config_path') return 'legacy-deletion.jsonc';
    if (expression === 'github.event.inputs.delete_legacy_translation_telemetry') return scenario.env.DELETE_LEGACY_TRANSLATION_TELEMETRY ?? '';
    if (expression === 'github.event.inputs.confirm_legacy_translation_telemetry_deletion') return scenario.env.CONFIRM_LEGACY_TRANSLATION_TELEMETRY_DELETION ?? '';
    if (expression.startsWith('secrets.')) return 'synthetic-not-a-credential';
    const output = expression.match(/^steps\.([^.]+)\.outputs\.(.+)$/u);
    if (output) return scenario.outputs[output[1]!]?.[output[2]!] ?? '';
    const outcome = expression.match(/^steps\.([^.]+)\.outcome$/u);
    if (outcome) return scenario.outcomes[outcome[1]!] ?? '';
    throw new Error(`Unexpected workflow expression: ${expression}`);
  });
}

function extractRun(block: string) {
  const multiline = block.match(/^        run: (\||>-)\n((?:          .*\n|\n)*)/mu);
  if (multiline) {
    const body = multiline[2]!.split('\n').map((line) => line.slice(10));
    return body.join(multiline[1] === '>-' ? ' ' : '\n');
  }
  const inline = block.match(/^        run: (.+)$/mu);
  if (!inline) throw new Error('Workflow step has no run block');
  return inline[1]!;
}

function runShell(script: string, scenario: Scenario, overrides: NodeJS.ProcessEnv = {}) {
  const result = spawnSync('bash', ['--noprofile', '--norc', '-e', '-o', 'pipefail', '-c', `export BACKUP_SHELL_PID=$$\n${script}`], {
    cwd: scenario.cwd,
    env: { ...scenario.env, ...overrides },
    encoding: 'utf8',
    timeout: 20000,
  });
  if (result.error) throw result.error;
  scenario.publicOutput.stdout += result.stdout;
  scenario.publicOutput.stderr += result.stderr;
  return { status: result.status, stderr: result.stderr };
}

function expectNoPlaintext(scenario: Scenario) {
  expect(readdirSync(scenario.runnerTemp).filter((name) => name.includes('plaintext') || name.endsWith('.sql'))).toEqual([]);
  if (existsSync(scenario.exportPath)) expect(existsSync(readFileSync(scenario.exportPath, 'utf8').trim())).toBe(false);
  if (existsSync(scenario.debugPath)) expect(existsSync(readFileSync(scenario.debugPath, 'utf8').trim())).toBe(false);
}

function sqliteRecords(contents: string) {
  const db = new DatabaseSync(':memory:');
  try {
    db.exec(contents);
    return {
      installations: db.prepare('SELECT * FROM installations ORDER BY id').all(),
      config: db.prepare('SELECT * FROM broker_config ORDER BY key').all(),
    };
  } finally {
    db.close();
  }
}
