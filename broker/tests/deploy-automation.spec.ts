import { execFileSync, spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, relative, sep } from 'node:path';
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
const deployWorkflow = new URL(
  '../../.github/workflows/deploy-broker-direct.yml',
  import.meta.url,
);
const abuseControlsWorkflow = new URL(
  '../../.github/workflows/maintenance-broker-abuse-controls.yml',
  import.meta.url,
);

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
      expect(renderedSql).not.toContain('__BOOTSTRAP_REQUIRED__');
      expect(renderedSql).toContain(bootstrapSalt);
      expect(renderedSql).not.toContain('CREATE TEMP TABLE');
      expect(renderedSql).toContain("json_extract(value, '$.current.salt') = '__BOOTSTRAP' || '_REQUIRED__'");

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
  it.skipIf(!bashAvailable)('keeps telemetry deletion and referral retirement out of both ordinary deploy phases while preserving deferred privacy migrations', () => {
    const workflow = readFileSync(deployWorkflow, 'utf8');
    const stageScript = extractWorkflowRunBlock(
      workflow,
      'Stage ordinary broker deployment migrations',
    );
    const tempDir = createRepoTempDir();
    const migrationsDir = join(tempDir, 'migrations');
    mkdirSync(migrationsDir);
    const migrationNames = [
      '0020_network_identity_hmac.sql',
      '0021_network_identity_purge.sql',
      '0022_managed_operation_issuance_context.sql',
      '0023_backfill_operation_route_rate_limits.sql',
      '0024_allow_unattributed_request_events.sql',
      '0025_drop_legacy_translation_telemetry.sql',
      '0026_managed_attempt_cleanup_ownership.sql',
      '0027_remove_managed_referrals.sql',
    ];
    for (const migrationName of migrationNames) {
      writeFileSync(join(migrationsDir, migrationName), `SELECT '${migrationName}';\n`);
    }

    const result = runBashScript(stageScript, tempDir);
    expect(result).toEqual({ status: 0, stderr: '' });
    expect(
      readdirSync(join(tempDir, '.deploy-direct', 'staged-migrations-pre-backfill')).sort(),
    ).toEqual([
      '0020_network_identity_hmac.sql',
      '0022_managed_operation_issuance_context.sql',
      '0023_backfill_operation_route_rate_limits.sql',
      '0026_managed_attempt_cleanup_ownership.sql',
    ]);
    expect(
      readdirSync(join(tempDir, '.deploy-direct', 'staged-migrations-post-backfill')).sort(),
    ).toEqual([
      '0020_network_identity_hmac.sql',
      '0021_network_identity_purge.sql',
      '0022_managed_operation_issuance_context.sql',
      '0023_backfill_operation_route_rate_limits.sql',
      '0024_allow_unattributed_request_events.sql',
      '0026_managed_attempt_cleanup_ownership.sql',
    ]);
  });

  it.skipIf(!bashAvailable)('accepts repeat deploys after both deferred network identity migrations are already applied', () => {
    const workflow = readFileSync(deployWorkflow, 'utf8');
    const verifyScript = extractWorkflowRunBlock(
      workflow,
      'Verify deferred network identity migrations are consistent',
    ).replaceAll('${{ steps.render.outputs.config_path }}', 'wrangler.production.jsonc');
    const requiredMigrations = [
      '0022_managed_operation_issuance_context.sql',
      '0023_backfill_operation_route_rate_limits.sql',
    ];
    const deferredMigrations = [
      '0021_network_identity_purge.sql',
      '0024_allow_unattributed_request_events.sql',
    ];
    const scenarios = [
      { migrations: requiredMigrations, status: 0 },
      { migrations: [...requiredMigrations, ...deferredMigrations], status: 0 },
      { migrations: [...requiredMigrations, deferredMigrations[0]], status: 1 },
      { migrations: [...requiredMigrations, deferredMigrations[1]], status: 1 },
      { migrations: [requiredMigrations[0]], status: 1 },
    ];

    for (const scenario of scenarios) {
      const tempDir = createRepoTempDir();
      const bashEnvPath = join(tempDir, 'bash-env.sh');
      writeFileSync(
        bashEnvPath,
        'pnpm() { printf \'%s\' "$JOURNAL_JSON"; }\n',
      );
      const journalJson = JSON.stringify([
        { results: scenario.migrations.map((name) => ({ name })) },
      ]);

      const result = runBashScript(verifyScript, tempDir, {
        BASH_ENV: './bash-env.sh',
        JOURNAL_JSON: journalJson,
      });

      expect(result.status, scenario.migrations.join(',')).toBe(scenario.status);
    }
  });
  it.skipIf(!bashAvailable)('admits a fresh schema or completed privacy history and rejects incomplete existing databases before deployment', () => {
    const workflow = readFileSync(deployWorkflow, 'utf8');
    const script = extractWorkflowRunBlock(workflow, 'Require completed privacy migration before referral-free deployment')
      .replaceAll('${{ steps.render.outputs.config_path }}', 'wrangler.production.jsonc');
    const scenarios = [
      { tables: [], migrations: [], status: 0, firstInstall: true },
      { tables: ['installations', 'broker_config'], migrations: ['0021_network_identity_purge.sql', '0024_allow_unattributed_request_events.sql'], status: 0, firstInstall: false },
      { tables: ['installations', 'broker_config'], migrations: ['0021_network_identity_purge.sql'], status: 1, firstInstall: false },
      { tables: ['installations', 'broker_config'], migrations: [], status: 1, firstInstall: false },
      { tables: ['broker_config'], migrations: [], status: 1, firstInstall: false },
    ];
    for (const scenario of scenarios) {
      const tempDir = createRepoTempDir();
      writeFileSync(join(tempDir, 'bash-env.sh'), `pnpm() { case \"$*\" in *sqlite_master*) printf '%s' \"$SCHEMA_JSON\" ;; *) printf '%s' \"$JOURNAL_JSON\" ;; esac; }\n`);
      const outputPath = join(tempDir, 'github-output');
      const result = runBashScript(script, tempDir, {
        BASH_ENV: './bash-env.sh',
        GITHUB_OUTPUT: outputPath,
        SCHEMA_JSON: JSON.stringify([{ results: scenario.tables.map((name) => ({ name })) }]),
        JOURNAL_JSON: JSON.stringify([{ results: scenario.migrations.map((name) => ({ name })) }]),
      });
      expect(result.status, JSON.stringify(scenario)).toBe(scenario.status);
      if (scenario.status === 0) expect(readFileSync(outputPath, 'utf8')).toContain(`first_install=${scenario.firstInstall}`);
    }
    const prerequisite = workflow.indexOf('Require completed privacy migration before referral-free deployment');
    expect(prerequisite).toBeLessThan(workflow.indexOf('Export pre-migration production D1 backup'));
    expect(prerequisite).toBeLessThan(workflow.indexOf('pnpm exec wrangler deploy'));
    expect(workflow).toContain("if: steps.privacy_prerequisite.outputs.first_install == 'true'");
  });

  it.skipIf(!bashAvailable)('requires the dedicated deletion inputs and a usable backup before staging and applying only migration 0025', () => {
    const workflow = readFileSync(deployWorkflow, 'utf8');
    const guardScript = extractWorkflowRunBlock(
      workflow,
      'Guard production deploy ref and confirmation',
    );
    for (const scenario of [
      {
        env: {
          DELETE_LEGACY_TRANSLATION_TELEMETRY: undefined,
          CONFIRM_LEGACY_TRANSLATION_TELEMETRY_DELETION: undefined,
        },
        status: 0,
      },
      {
        env: {
          DELETE_LEGACY_TRANSLATION_TELEMETRY: 'false',
          CONFIRM_LEGACY_TRANSLATION_TELEMETRY_DELETION: undefined,
        },
        status: 0,
      },
      {
        env: {
          DELETE_LEGACY_TRANSLATION_TELEMETRY: 'true',
          CONFIRM_LEGACY_TRANSLATION_TELEMETRY_DELETION: 'deploy puripuly-heart-broker from dev',
        },
        status: 1,
      },
      {
        env: {
          DELETE_LEGACY_TRANSLATION_TELEMETRY: 'true',
          CONFIRM_LEGACY_TRANSLATION_TELEMETRY_DELETION:
            'delete retired translation telemetry from production D1',
        },
        status: 0,
      },
    ]) {
      expect(runGuardScript(guardScript, scenario.env).status).toBe(scenario.status);
    }

    const applyScript = extractWorkflowRunBlock(
      workflow,
      'Apply explicitly authorized legacy telemetry deletion',
    );
    const scenarios: Array<{
      deletion: string | undefined;
      confirmation: string | undefined;
      status: number;
      applied: boolean;
      usableBackup?: boolean;
    }> = [
      { deletion: undefined, confirmation: undefined, status: 0, applied: false },
      { deletion: 'false', confirmation: undefined, status: 0, applied: false },
      { deletion: 'true', confirmation: 'wrong confirmation', status: 1, applied: false },
      {
        deletion: 'true',
        confirmation: 'delete retired translation telemetry from production D1',
        status: 0,
        applied: true,
      },
      {
        deletion: 'true',
        confirmation: 'delete retired translation telemetry from production D1',
        status: 1,
        applied: false,
        usableBackup: false,
      },
    ];

    for (const scenario of scenarios) {
      const tempDir = createRepoTempDir();
      const migrationsDir = join(tempDir, 'migrations');
      mkdirSync(migrationsDir);
      writeFileSync(
        join(migrationsDir, '0025_drop_legacy_translation_telemetry.sql'),
        'DROP TABLE IF EXISTS telemetry_active_days;\n',
      );
      const backupPath = join(tempDir, 'pre-migration-backup.sql');
      const configPath = join(tempDir, 'legacy-deletion.jsonc');
      const outputPath = join(tempDir, 'github-output.txt');
      const pnpmLogPath = join(tempDir, 'pnpm.log');
      const bashEnvPath = join(tempDir, 'bash-env.sh');
      if (scenario.usableBackup !== false) {
        writeFileSync(backupPath, 'production backup\n');
      }
      writeFileSync(configPath, '{}\n');
      writeFileSync(outputPath, '');
      writeFileSync(
        bashEnvPath,
        'pnpm() { printf \'%s\\n\' "$*" >> "$PNPM_LOG"; }\n',
      );

      const result = runBashScript(applyScript, tempDir, {
        DELETE_LEGACY_TRANSLATION_TELEMETRY: scenario.deletion,
        CONFIRM_LEGACY_TRANSLATION_TELEMETRY_DELETION: scenario.confirmation,
        LEGACY_DELETION_CONFIG_PATH: 'legacy-deletion.jsonc',
        D1_BACKUP_PATH: 'pre-migration-backup.sql',
        GITHUB_OUTPUT: 'github-output.txt',
        PNPM_LOG: 'pnpm.log',
        BASH_ENV: './bash-env.sh',
      });
      expect(result.status).toBe(scenario.status);

      const stagedDir = join(
        tempDir,
        '.deploy-direct',
        'staged-migrations-legacy-telemetry-deletion',
      );
      if (scenario.applied) {
        expect(readdirSync(stagedDir)).toEqual([
          '0025_drop_legacy_translation_telemetry.sql',
        ]);
        expect(readFileSync(pnpmLogPath, 'utf8').split('\n').filter(Boolean)).toEqual([
          'exec wrangler types --config legacy-deletion.jsonc',
          'exec wrangler d1 migrations apply puripuly-heart-broker --remote --config legacy-deletion.jsonc',
        ]);
        expect(readFileSync(outputPath, 'utf8')).toBe('applied=true\n');
      } else {
        expect(existsSync(stagedDir)).toBe(false);
        expect(existsSync(pnpmLogPath)).toBe(false);
        if (scenario.status === 0) {
          expect(readFileSync(outputPath, 'utf8')).toBe('applied=false\n');
        }
      }
    }

    const backupUploadIndex = workflow.indexOf('Upload pre-migration D1 backup');
    const workerDeployIndex = workflow.indexOf('Deploy canonical broker worker');
    const healthCheckIndex = workflow.indexOf('Wait for deployed workers.dev healthz');
    const smokeCheckIndex = workflow.indexOf(
      'Run deployed broker production QQ issuance smoke test',
    );
    const deletionIndex = workflow.indexOf(
      'Apply explicitly authorized legacy telemetry deletion',
    );
    for (const prerequisiteIndex of [
      backupUploadIndex,
      workerDeployIndex,
      healthCheckIndex,
      smokeCheckIndex,
      deletionIndex,
    ]) {
      expect(prerequisiteIndex).toBeGreaterThanOrEqual(0);
    }
    expect(backupUploadIndex).toBeLessThan(deletionIndex);
    expect(workerDeployIndex).toBeLessThan(deletionIndex);
    expect(healthCheckIndex).toBeLessThan(deletionIndex);
    expect(smokeCheckIndex).toBeLessThan(deletionIndex);
  });


  it.skipIf(!bashAvailable)('guards network identity key versions and the previous pair before migrations', () => {
    const workflow = readFileSync(deployWorkflow, 'utf8');
    const guardScript = extractWorkflowRunBlock(
      workflow,
      'Guard production deploy ref and confirmation',
    );
    const scenarios: Array<{
      name: string;
      env: Record<string, string | undefined>;
      expectStatus: number;
      expectStderrContains?: string;
    }> = [
      {
        name: 'first deploy v1 without previous',
        env: {
          NETWORK_IDENTITY_HMAC_KEY_VERSION_PRODUCTION: '1',
          NETWORK_IDENTITY_HMAC_KEY_VERSION_PREVIOUS_PRODUCTION: undefined,
          NETWORK_IDENTITY_HMAC_SECRET_PREVIOUS_PRODUCTION: undefined,
        },
        expectStatus: 0,
      },
      {
        name: 'rotation v2 with previous v1',
        env: {
          NETWORK_IDENTITY_HMAC_KEY_VERSION_PRODUCTION: '2',
          NETWORK_IDENTITY_HMAC_KEY_VERSION_PREVIOUS_PRODUCTION: '1',
          NETWORK_IDENTITY_HMAC_SECRET_PREVIOUS_PRODUCTION: 'guard-scenario-previous-secret',
        },
        expectStatus: 0,
      },
      {
        name: 'missing current version is rejected without an implicit default',
        env: {
          NETWORK_IDENTITY_HMAC_KEY_VERSION_PRODUCTION: undefined,
          NETWORK_IDENTITY_HMAC_KEY_VERSION_PREVIOUS_PRODUCTION: undefined,
          NETWORK_IDENTITY_HMAC_SECRET_PREVIOUS_PRODUCTION: undefined,
        },
        expectStatus: 1,
        expectStderrContains:
          'NETWORK_IDENTITY_HMAC_KEY_VERSION_PRODUCTION must be a positive integer.',
      },
      {
        name: 'zero current version is rejected',
        env: { NETWORK_IDENTITY_HMAC_KEY_VERSION_PRODUCTION: '0' },
        expectStatus: 1,
        expectStderrContains:
          'NETWORK_IDENTITY_HMAC_KEY_VERSION_PRODUCTION must be a positive integer.',
      },
      {
        name: 'non-numeric current version is rejected',
        env: { NETWORK_IDENTITY_HMAC_KEY_VERSION_PRODUCTION: 'v1' },
        expectStatus: 1,
        expectStderrContains:
          'NETWORK_IDENTITY_HMAC_KEY_VERSION_PRODUCTION must be a positive integer.',
      },
      {
        name: 'zero-padded current version is rejected',
        env: { NETWORK_IDENTITY_HMAC_KEY_VERSION_PRODUCTION: '01' },
        expectStatus: 1,
        expectStderrContains:
          'NETWORK_IDENTITY_HMAC_KEY_VERSION_PRODUCTION must be a positive integer.',
      },
      {
        name: 'previous version without previous secret is rejected',
        env: {
          NETWORK_IDENTITY_HMAC_KEY_VERSION_PRODUCTION: '2',
          NETWORK_IDENTITY_HMAC_KEY_VERSION_PREVIOUS_PRODUCTION: '1',
          NETWORK_IDENTITY_HMAC_SECRET_PREVIOUS_PRODUCTION: undefined,
        },
        expectStatus: 1,
        expectStderrContains:
          'NETWORK_IDENTITY_HMAC_SECRET_PREVIOUS_PRODUCTION is required when NETWORK_IDENTITY_HMAC_KEY_VERSION_PREVIOUS_PRODUCTION is set.',
      },
      {
        name: 'whitespace-only previous secret does not satisfy the pair',
        env: {
          NETWORK_IDENTITY_HMAC_KEY_VERSION_PRODUCTION: '2',
          NETWORK_IDENTITY_HMAC_KEY_VERSION_PREVIOUS_PRODUCTION: '1',
          NETWORK_IDENTITY_HMAC_SECRET_PREVIOUS_PRODUCTION: '   ',
        },
        expectStatus: 1,
        expectStderrContains:
          'NETWORK_IDENTITY_HMAC_SECRET_PREVIOUS_PRODUCTION is required when NETWORK_IDENTITY_HMAC_KEY_VERSION_PREVIOUS_PRODUCTION is set.',
      },
      {
        name: 'previous secret without previous version is rejected',
        env: {
          NETWORK_IDENTITY_HMAC_KEY_VERSION_PRODUCTION: '2',
          NETWORK_IDENTITY_HMAC_KEY_VERSION_PREVIOUS_PRODUCTION: undefined,
          NETWORK_IDENTITY_HMAC_SECRET_PREVIOUS_PRODUCTION: 'guard-scenario-previous-secret',
        },
        expectStatus: 1,
        expectStderrContains:
          'NETWORK_IDENTITY_HMAC_KEY_VERSION_PREVIOUS_PRODUCTION is required when NETWORK_IDENTITY_HMAC_SECRET_PREVIOUS_PRODUCTION is set.',
      },
      {
        name: 'non-numeric previous version is rejected',
        env: {
          NETWORK_IDENTITY_HMAC_KEY_VERSION_PRODUCTION: '2',
          NETWORK_IDENTITY_HMAC_KEY_VERSION_PREVIOUS_PRODUCTION: 'v1',
          NETWORK_IDENTITY_HMAC_SECRET_PREVIOUS_PRODUCTION: 'guard-scenario-previous-secret',
        },
        expectStatus: 1,
        expectStderrContains:
          'NETWORK_IDENTITY_HMAC_KEY_VERSION_PREVIOUS_PRODUCTION must be a positive integer when set.',
      },
      {
        name: 'previous version equal to current is rejected',
        env: {
          NETWORK_IDENTITY_HMAC_KEY_VERSION_PRODUCTION: '2',
          NETWORK_IDENTITY_HMAC_KEY_VERSION_PREVIOUS_PRODUCTION: '2',
          NETWORK_IDENTITY_HMAC_SECRET_PREVIOUS_PRODUCTION: 'guard-scenario-previous-secret',
        },
        expectStatus: 1,
        expectStderrContains:
          'NETWORK_IDENTITY_HMAC_KEY_VERSION_PREVIOUS_PRODUCTION must differ from NETWORK_IDENTITY_HMAC_KEY_VERSION_PRODUCTION.',
      },
    ];

    for (const scenario of scenarios) {
      const result = runGuardScript(guardScript, scenario.env);
      expect(result.status, scenario.name).toBe(scenario.expectStatus);
      if (scenario.expectStderrContains !== undefined) {
        expect(result.stderr, scenario.name).toContain(scenario.expectStderrContains);
      }
      expect(result.stderr, scenario.name).not.toContain('guard-scenario-current-secret');
      expect(result.stderr, scenario.name).not.toContain('guard-scenario-previous-secret');
    }
  }, 15000);

  it('keeps daily-cap maintenance manually authorized in production with configured identity versions', () => {
    const workflow = readFileSync(abuseControlsWorkflow, 'utf8');

    expect(workflow).toContain('workflow_dispatch:');
    expect(workflow).not.toMatch(/^\s*(push|schedule|pull_request):/mu);
    expect(workflow).toContain('environment: production');
    for (const name of [
      'NETWORK_IDENTITY_HMAC_KEY_VERSION_PRODUCTION',
      'NETWORK_IDENTITY_HMAC_KEY_VERSION_PREVIOUS_PRODUCTION',
    ]) {
      expect(workflow).toContain(`${name}: \${{ vars.${name} }}`);
    }
    expect(workflow).not.toMatch(/wrangler (deploy|secret|d1 migrations)/u);
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
    return spawnSync('bash', ['--version']).status === 0;
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
function guardScenarioEnv(overrides: Record<string, string | undefined>): Record<string, string> {
  const env: Record<string, string> = {
    GITHUB_REF: 'refs/heads/dev',
    CONFIRM_PRODUCTION_DEPLOY: 'deploy puripuly-heart-broker from dev',
    DELETE_LEGACY_TRANSLATION_TELEMETRY: 'false',
    BROKER_CANONICAL_WORKERS_DEV_URL: 'https://puripuly-heart-broker.example.workers.dev',
    BROKER_DEPLOY_SMOKE_DISALLOWED_MODEL_PRODUCTION: 'guard-scenario-disallowed-model',
    CLOUDFLARE_API_TOKEN: 'guard-scenario-api-token',
    CLOUDFLARE_ACCOUNT_ID: 'guard-scenario-account-id',
    BROKER_D1_DATABASE_ID_PRODUCTION: 'guard-scenario-database-id',
    OPENROUTER_MANAGED_API_KEY_PRODUCTION: 'guard-scenario-managed-key',
    OPENROUTER_MANAGEMENT_API_KEY_PRODUCTION: 'guard-scenario-management-key',
    OPENROUTER_MANAGED_GUARDRAIL_ID_PRODUCTION: 'guard-scenario-guardrail-id',
    OPENROUTER_MANAGED_USER_HMAC_SECRET_PRODUCTION: 'guard-scenario-user-hmac-secret',
    QQ_AUTH_HMAC_PSK_PRODUCTION: 'guard-scenario-qq-psk',
    TELEMETRY_SUBJECT_HMAC_SECRET_PRODUCTION: 'guard-scenario-telemetry-secret',
    DISCORD_CLIENT_ID_PRODUCTION: 'guard-scenario-discord-client-id',
    DISCORD_CLIENT_SECRET_PRODUCTION: 'guard-scenario-discord-client-secret',
    DISCORD_REDIRECT_URI_ALLOWLIST_PRODUCTION:
      'http://127.0.0.1:62187/discord/callback, http://127.0.0.1:62188/discord/callback, http://127.0.0.1:62189/discord/callback',
    DISCORD_USER_REF_SECRET_PRODUCTION: 'guard-scenario-discord-user-ref-secret',
    NETWORK_IDENTITY_HMAC_SECRET_PRODUCTION: 'guard-scenario-current-secret',
    NETWORK_IDENTITY_HMAC_KEY_VERSION_PRODUCTION: '1',
    DISCORD_OPERATIONS_WEBHOOK_URL_PRODUCTION: 'https://discord.example/webhook',
  };
  for (const [key, value] of Object.entries(overrides)) {
    if (value === undefined) {
      delete env[key];
    } else {
      env[key] = value;
    }
  }
  return env;
}
function runGuardScript(
  guardScript: string,
  envOverrides: Record<string, string | undefined>,
): { status: number; stderr: string } {
  const tempDir = createRepoTempDir();
  const scriptPath = join(tempDir, 'guard-scenario.sh');
  const assignments = Object.entries(guardScenarioEnv(envOverrides)).map(
    ([key, value]) => `${key}='${value.replace(/'/g, `'"'"'`)}'`,
  );
  writeFileSync(scriptPath, `${assignments.join('\n')}\n${guardScript}`, 'utf8');
  const bashPath = relative(process.cwd(), scriptPath).split(sep).join('/');
  try {
    execFileSync('bash', [bashPath], {
      stdio: ['ignore', 'ignore', 'pipe'],
    });
    return { status: 0, stderr: '' };
  } catch (error) {
    const execError = error as { status?: number; stderr?: string | Buffer };
    if (execError.status === undefined) {
      throw error;
    }
    return { status: execError.status, stderr: String(execError.stderr ?? '') };
  }
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
  const scriptPath = join(cwd, 'workflow-step-scenario.sh');
  writeFileSync(
    scriptPath,
    `cd '${quotedBashCwd}'\n${assignments.join('\n')}\n${sourceBashEnv}${script}`,
    'utf8',
  );
  const bashPath = relative(process.cwd(), scriptPath).split(sep).join('/');
  const result = spawnSync(
    'bash',
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
  });
}

