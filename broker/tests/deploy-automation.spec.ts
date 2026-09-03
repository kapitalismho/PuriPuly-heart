import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
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
const deploySmokeSpec = new URL(
  './deploy-smoke/canonical-production.spec.ts',
  import.meta.url,
);
const brokerReadme = new URL('../README.md', import.meta.url);
const dailySummaryV2Finalizer = new URL(
  '../deploy/finalize-daily-summary-v2.sql',
  import.meta.url,
);
const appActiveDayFinalizer = new URL(
  '../deploy/finalize-app-active-day.sql',
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
      ]),
    ).toThrow(/canonical worker name/i);
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

  it('ships a manual direct-deploy workflow that renders config, applies remote D1 changes, syncs the transitional and child-key management secrets, deploys the canonical worker, and runs production QQ issuance smoke', () => {
    const workflow = readFileSync(deployWorkflow, 'utf8');
    const smokeSpec = readFileSync(deploySmokeSpec, 'utf8');
    const readme = readFileSync(brokerReadme, 'utf8');
    const deployJobEnvBlock = extractBetween(
      workflow,
      '    env:\n',
      '    steps:\n',
    );
    const productionSecretNames = [
      'CLOUDFLARE_API_TOKEN',
      'CLOUDFLARE_ACCOUNT_ID',
      'BROKER_D1_DATABASE_ID_PRODUCTION',
      'OPENROUTER_MANAGED_API_KEY_PRODUCTION',
      'OPENROUTER_MANAGEMENT_API_KEY_PRODUCTION',
      'OPENROUTER_MANAGED_GUARDRAIL_ID_PRODUCTION',
      'OPENROUTER_MANAGED_USER_HMAC_SECRET_PRODUCTION',
      'QQ_AUTH_HMAC_PSK_PRODUCTION',
      'TELEMETRY_SUBJECT_HMAC_SECRET_PRODUCTION',
      'DISCORD_CLIENT_ID_PRODUCTION',
      'DISCORD_CLIENT_SECRET_PRODUCTION',
      'DISCORD_REDIRECT_URI_ALLOWLIST_PRODUCTION',
      'DISCORD_USER_REF_SECRET_PRODUCTION',
      'NETWORK_IDENTITY_HMAC_SECRET_PRODUCTION',
      'NETWORK_IDENTITY_HMAC_SECRET_PREVIOUS_PRODUCTION',
      'DISCORD_OPERATIONS_WEBHOOK_URL_PRODUCTION',
    ];
    const liveInputValidationIndex = smokeSpec.indexOf(
      'const liveInputs = readLiveDeploySmokeInputs',
    );
    const healthzProbeIndex = smokeSpec.indexOf(
      "url: new URL('/healthz'",
    );
    const managedUserHmacBlankCheckIndex = workflow.indexOf(
      'OPENROUTER_MANAGED_USER_HMAC_SECRET_PRODUCTION is required and must not be blank.',
    );
    const cloudflareApiTokenBlankCheckIndex = workflow.indexOf(
      'CLOUDFLARE_API_TOKEN is required and must not be blank.',
    );
    const cloudflareAccountIdBlankCheckIndex = workflow.indexOf(
      'CLOUDFLARE_ACCOUNT_ID is required and must not be blank.',
    );
    const brokerD1DatabaseIdBlankCheckIndex = workflow.indexOf(
      'BROKER_D1_DATABASE_ID_PRODUCTION is required and must not be blank.',
    );
    const managedApiKeyBlankCheckIndex = workflow.indexOf(
      'OPENROUTER_MANAGED_API_KEY_PRODUCTION is required and must not be blank.',
    );
    const managementApiKeyBlankCheckIndex = workflow.indexOf(
      'OPENROUTER_MANAGEMENT_API_KEY_PRODUCTION is required and must not be blank.',
    );
    const managedGuardrailIdBlankCheckIndex = workflow.indexOf(
      'OPENROUTER_MANAGED_GUARDRAIL_ID_PRODUCTION is required and must not be blank.',
    );
    const discordWebhookBlankCheckIndex = workflow.indexOf(
      'DISCORD_OPERATIONS_WEBHOOK_URL_PRODUCTION is required and must not be blank.',
    );
    const remoteD1MigrationIndex = workflow.indexOf(
      'wrangler d1 migrations apply',
    );
    const remoteD1BackupIndex = workflow.indexOf('wrangler d1 export');
    const remoteD1BackupUploadIndex = workflow.indexOf(
      'actions/upload-artifact@v7',
    );
    const openRouterGuardrailPatchIndex = workflow.indexOf(
      'PATCH "$guardrail_url"',
    );
    const firstSecretSyncIndex = workflow.indexOf(
      'wrangler secret put OPENROUTER_MANAGED_API_KEY',
    );
    const managedUserHmacSyncIndex = workflow.indexOf(
      'wrangler secret put OPENROUTER_MANAGED_USER_HMAC_SECRET',
    );
    const discordImmediateWebhookSyncIndex = workflow.indexOf(
      'wrangler secret put DISCORD_IMMEDIATE_ALERT_WEBHOOK_URL',
    );
    const discordDailyWebhookSyncIndex = workflow.indexOf(
      'wrangler secret put DISCORD_DAILY_REPORT_WEBHOOK_URL',
    );
    const qqAuthHmacPskBlankCheckIndex = workflow.indexOf(
      'QQ_AUTH_HMAC_PSK_PRODUCTION is required and must not be blank.',
    );
    const qqAuthHmacPskSyncIndex = workflow.indexOf(
      'wrangler secret put QQ_AUTH_HMAC_PSK',
    );
    const telemetrySubjectHmacSecretBlankCheckIndex = workflow.indexOf(
      'TELEMETRY_SUBJECT_HMAC_SECRET_PRODUCTION is required and must not be blank.',
    );
    const telemetrySubjectHmacSecretSyncIndex = workflow.indexOf(
      'wrangler secret put TELEMETRY_SUBJECT_HMAC_SECRET',
    );
    const networkIdentityHmacBlankCheckIndex = workflow.indexOf(
      'NETWORK_IDENTITY_HMAC_SECRET_PRODUCTION is required and must not be blank.',
    );
    const networkIdentityHmacSyncIndex = workflow.indexOf(
      'wrangler secret put NETWORK_IDENTITY_HMAC_SECRET ',
    );
    const networkIdentityPreviousSyncIndex = workflow.indexOf(
      'wrangler secret put NETWORK_IDENTITY_HMAC_SECRET_PREVIOUS',
    );
    const networkIdentityPreviousDeleteIndex = workflow.indexOf(
      'wrangler secret delete NETWORK_IDENTITY_HMAC_SECRET_PREVIOUS',
    );
    const stagedMigrationRenderIndex = workflow.indexOf(
      'staged-migrations-through-0020',
    );
    const stagedMigrationApplyIndex = workflow.indexOf(
      'Apply remote D1 migrations through 0020',
    );
    const networkIdentityBackfillAwaitIndex = workflow.indexOf(
      'Await network identity backfill until keyed_only',
    );
    const networkIdentityPurgeApplyIndex = workflow.indexOf(
      'wrangler d1 migrations apply',
      remoteD1MigrationIndex + 1,
    );
    const networkIdentityPurgeVerifyIndex = workflow.indexOf(
      'Verify legacy network identity columns are gone',
    );

    expect(workflow).toContain('workflow_dispatch:');
    expect(workflow).not.toContain('\npush:');
    expect(workflow).toContain('confirm_production_deploy');
    expect(workflow).toContain('environment: production');
    expect(deployJobEnvBlock).toContain("NODE_VERSION: '22'");
    expect(deployJobEnvBlock).toContain('BROKER_CANONICAL_WORKER_NAME: puripuly-heart-broker');
    expect(deployJobEnvBlock).toContain(
      'BROKER_CANONICAL_WORKERS_DEV_URL: ${{ vars.BROKER_CANONICAL_WORKERS_DEV_URL }}',
    );
    expect(deployJobEnvBlock).toContain(
      'BROKER_DEPLOY_SMOKE_DISALLOWED_MODEL_PRODUCTION: ${{ vars.BROKER_DEPLOY_SMOKE_DISALLOWED_MODEL_PRODUCTION }}',
    );
    expect(deployJobEnvBlock).not.toContain('secrets.');
    for (const productionSecretName of productionSecretNames) {
      expect(deployJobEnvBlock).not.toContain(`${productionSecretName}:`);
      expect(workflow).toContain(
        `${productionSecretName}: \${{ secrets.${productionSecretName} }}`,
      );
    }
    expect(workflow).toContain('BROKER_D1_DATABASE_ID_PRODUCTION');
    expect(workflow).toContain('OPENROUTER_MANAGED_API_KEY_PRODUCTION');
    expect(workflow).toContain('OPENROUTER_MANAGEMENT_API_KEY_PRODUCTION');
    expect(workflow).toContain('OPENROUTER_MANAGED_GUARDRAIL_ID_PRODUCTION');
    expect(workflow).toContain('OPENROUTER_MANAGED_USER_HMAC_SECRET_PRODUCTION');
    expect(workflow).toContain('QQ_AUTH_HMAC_PSK_PRODUCTION');
    expect(workflow).toContain('QQ_AUTH_HMAC_PSK');
    expect(workflow).toContain('TELEMETRY_SUBJECT_HMAC_SECRET_PRODUCTION');
    expect(workflow).toContain('TELEMETRY_SUBJECT_HMAC_SECRET');
    expect(workflow).toContain('NETWORK_IDENTITY_HMAC_SECRET_PRODUCTION');
    expect(workflow).toContain('NETWORK_IDENTITY_HMAC_SECRET_PREVIOUS_PRODUCTION');
    expect(workflow).toContain('NETWORK_IDENTITY_HMAC_SECRET');
    expect(workflow).toContain('staged-migrations-through-0020');
    expect(workflow).toContain('0020_network_identity_hmac.sql');
    expect(workflow).toContain('0021_network_identity_purge.sql');
    expect(workflow).toContain('staged_config_path');
    expect(workflow).toContain('migrations_dir');
    expect(workflow).toContain('network_identity_migration');
    expect(workflow).toContain('keyed_only');
    expect(workflow).toContain('pragma_table_info');
    expect(workflow).toContain('attempt_ip_hash');
    expect(workflow).toContain('Apply remote D1 migrations through 0020');
    expect(workflow).toContain('Await network identity backfill until keyed_only');
    expect(workflow).toContain('Apply network identity purge migration 0021');
    expect(workflow).toContain('Verify legacy network identity columns are gone');
    expect(workflow).toMatch(
      /wrangler secret put NETWORK_IDENTITY_HMAC_SECRET --config/u,
    );
    expect(workflow).toMatch(
      /wrangler secret put NETWORK_IDENTITY_HMAC_SECRET_PREVIOUS --config/u,
    );
    expect(workflow).toMatch(
      /wrangler secret delete NETWORK_IDENTITY_HMAC_SECRET_PREVIOUS --config/u,
    );
    expect(workflow).toContain('DISCORD_OPERATIONS_WEBHOOK_URL_PRODUCTION');
    expect(workflow).toContain('BROKER_DEPLOY_SMOKE_DISALLOWED_MODEL_PRODUCTION');
    expect(workflow).toContain('BROKER_CANONICAL_WORKERS_DEV_URL');
    expect(workflow).toContain(
      'BROKER_DEPLOY_SMOKE_DISALLOWED_MODEL_PRODUCTION is required',
    );
    expect(workflow).toContain('must differ from the managed allowlisted models.');
    expect(workflow).toContain('ref: refs/heads/dev');
    expect(workflow).toContain('render-production-wrangler-config.mjs');
    expect(workflow).toContain('render-fingerprint-bootstrap-sql.mjs');
    expect(workflow).toContain("working-directory: broker");
    expect(workflow).toContain("deploy_dir='.deploy-direct'");
    expect(workflow).toContain("config_path='wrangler.production.jsonc'");
    expect(workflow).toContain('fingerprint-bootstrap.sql');
    expect(workflow).toMatch(/wrangler types --config/u);
    expect(workflow).toContain('BROKER_CANONICAL_WORKERS_DEV_URL is required');
    expect(workflow).toContain('refs/heads/dev');
    expect(workflow).toContain("broker/src/trial-policy.ts");
    expect(workflow).toContain('MANAGED_TRIAL_ALLOWED_MODELS was not found');
    expect(workflow).toContain('https://openrouter.ai/api/v1/guardrails/');
    expect(workflow).toContain('PATCH "$guardrail_url"');
    expect(workflow).toContain('allowed_models');
    expect(workflow).toContain('allowed_providers');
    expect(workflow).toContain('ignored_providers');
    expect(workflow).toContain('enforce_zdr');
    expect(workflow).toContain('must be cleared (null or [])');
    expect(workflow).toContain('GET guardrail');
    expect(workflow).toMatch(
      /wrangler d1 migrations apply\s+puripuly-heart-broker\s+--remote\s+--config/u,
    );
    expect(workflow).toMatch(
      /wrangler d1 export puripuly-heart-broker\s+\\\s+--remote --config/u,
    );
    expect(workflow).toContain('$RUNNER_TEMP/puripuly-heart-broker-pre-migration-');
    expect(workflow).toContain('if-no-files-found: error');
    expect(workflow).toMatch(
      /wrangler d1 execute\s+puripuly-heart-broker\s+--remote\s+--config/u,
    );
    expect(workflow).toContain("json_extract(value, '$.current.salt')");
    expect(workflow).toMatch(
      /wrangler secret put OPENROUTER_MANAGED_API_KEY --config/u,
    );
    expect(workflow).toMatch(
      /wrangler secret put OPENROUTER_MANAGEMENT_API_KEY --config/u,
    );
    expect(workflow).toMatch(
      /wrangler secret put OPENROUTER_MANAGED_GUARDRAIL_ID --config/u,
    );
    expect(workflow).toMatch(
      /wrangler secret put OPENROUTER_MANAGED_USER_HMAC_SECRET --config/u,
    );
    expect(workflow).toMatch(/wrangler secret put QQ_AUTH_HMAC_PSK --config/u);
    expect(workflow).toMatch(
      /wrangler secret put TELEMETRY_SUBJECT_HMAC_SECRET --config/u,
    );
    expect(workflow).toMatch(
      /wrangler secret put DISCORD_IMMEDIATE_ALERT_WEBHOOK_URL --config/u,
    );
    expect(workflow).toMatch(
      /wrangler secret put DISCORD_DAILY_REPORT_WEBHOOK_URL --config/u,
    );
    for (const requiredDeployBlankCheckIndex of [
      cloudflareApiTokenBlankCheckIndex,
      cloudflareAccountIdBlankCheckIndex,
      brokerD1DatabaseIdBlankCheckIndex,
      managedApiKeyBlankCheckIndex,
      managementApiKeyBlankCheckIndex,
      managedGuardrailIdBlankCheckIndex,
    ]) {
      expect(requiredDeployBlankCheckIndex).toBeGreaterThanOrEqual(0);
      expect(requiredDeployBlankCheckIndex).toBeLessThan(remoteD1MigrationIndex);
      expect(requiredDeployBlankCheckIndex).toBeLessThan(openRouterGuardrailPatchIndex);
      expect(requiredDeployBlankCheckIndex).toBeLessThan(firstSecretSyncIndex);
    }
    expect(managedUserHmacBlankCheckIndex).toBeGreaterThanOrEqual(0);
    expect(discordWebhookBlankCheckIndex).toBeGreaterThanOrEqual(0);
    expect(remoteD1MigrationIndex).toBeGreaterThanOrEqual(0);
    expect(remoteD1BackupIndex).toBeGreaterThanOrEqual(0);
    expect(remoteD1BackupUploadIndex).toBeGreaterThanOrEqual(0);
    expect(remoteD1BackupIndex).toBeLessThan(remoteD1BackupUploadIndex);
    expect(remoteD1BackupUploadIndex).toBeLessThan(remoteD1MigrationIndex);
    expect(managedUserHmacBlankCheckIndex).toBeLessThan(remoteD1MigrationIndex);
    expect(managedUserHmacSyncIndex).toBeGreaterThanOrEqual(0);
    expect(managedUserHmacBlankCheckIndex).toBeLessThan(managedUserHmacSyncIndex);
    expect(qqAuthHmacPskBlankCheckIndex).toBeGreaterThanOrEqual(0);
    expect(qqAuthHmacPskBlankCheckIndex).toBeLessThan(remoteD1MigrationIndex);
    expect(qqAuthHmacPskSyncIndex).toBeGreaterThanOrEqual(0);
    expect(qqAuthHmacPskBlankCheckIndex).toBeLessThan(qqAuthHmacPskSyncIndex);
    expect(telemetrySubjectHmacSecretBlankCheckIndex).toBeGreaterThanOrEqual(0);
    expect(telemetrySubjectHmacSecretBlankCheckIndex).toBeLessThan(remoteD1MigrationIndex);
    expect(telemetrySubjectHmacSecretSyncIndex).toBeGreaterThanOrEqual(0);
    expect(telemetrySubjectHmacSecretBlankCheckIndex).toBeLessThan(
      telemetrySubjectHmacSecretSyncIndex,
    );
    expect(discordImmediateWebhookSyncIndex).toBeGreaterThanOrEqual(0);
    expect(discordDailyWebhookSyncIndex).toBeGreaterThanOrEqual(0);
    expect(discordWebhookBlankCheckIndex).toBeLessThan(discordImmediateWebhookSyncIndex);
    expect(discordWebhookBlankCheckIndex).toBeLessThan(discordDailyWebhookSyncIndex);
    expect(networkIdentityHmacBlankCheckIndex).toBeGreaterThanOrEqual(0);
    expect(networkIdentityHmacBlankCheckIndex).toBeLessThan(remoteD1MigrationIndex);
    expect(networkIdentityHmacSyncIndex).toBeGreaterThanOrEqual(0);
    expect(networkIdentityHmacBlankCheckIndex).toBeLessThan(networkIdentityHmacSyncIndex);
    expect(networkIdentityPreviousSyncIndex).toBeGreaterThanOrEqual(0);
    expect(networkIdentityPreviousDeleteIndex).toBeGreaterThanOrEqual(0);
    expect(stagedMigrationRenderIndex).toBeGreaterThanOrEqual(0);
    expect(stagedMigrationRenderIndex).toBeLessThan(remoteD1MigrationIndex);
    expect(stagedMigrationApplyIndex).toBeGreaterThanOrEqual(0);
    expect(stagedMigrationApplyIndex).toBeLessThan(firstSecretSyncIndex);
    expect(networkIdentityHmacSyncIndex).toBeLessThan(
      workflow.indexOf('pnpm exec wrangler deploy'),
    );
    expect(networkIdentityBackfillAwaitIndex).toBeGreaterThanOrEqual(0);
    expect(workflow.indexOf('pnpm exec wrangler deploy')).toBeLessThan(
      networkIdentityBackfillAwaitIndex,
    );
    expect(networkIdentityPurgeApplyIndex).toBeGreaterThanOrEqual(0);
    expect(networkIdentityBackfillAwaitIndex).toBeLessThan(networkIdentityPurgeApplyIndex);
    expect(networkIdentityPurgeVerifyIndex).toBeGreaterThanOrEqual(0);
    expect(networkIdentityPurgeApplyIndex).toBeLessThan(networkIdentityPurgeVerifyIndex);
    expect(networkIdentityPurgeVerifyIndex).toBeLessThan(
      workflow.indexOf('deploy/finalize-daily-summary-v2.sql'),
    );
    expect(networkIdentityPurgeVerifyIndex).toBeLessThan(
      workflow.indexOf('broker/tests/deploy-smoke/canonical-production.spec.ts'),
    );
    expect(workflow).toMatch(/wrangler deploy --config/u);
    expect(workflow).toContain(
      'broker/tests/deploy-smoke/canonical-production.spec.ts',
    );
    expect(workflow).toContain('BROKER_DEPLOY_SMOKE_DISALLOWED_MODEL');
    expect(workflow).toContain('curl --fail');
    expect(workflow).toContain('timeout-minutes: 10');
    expect(workflow).toContain('BROKER_DEPLOY_SMOKE_QQ_AUTH_HMAC_PSK');
    expect(workflow).toContain("BROKER_DEPLOY_SMOKE_RUN: 'true'");
    expect(workflow).toContain(
      'BROKER_DEPLOY_SMOKE_QQ_AUTH_HMAC_PSK: ${{ secrets.QQ_AUTH_HMAC_PSK_PRODUCTION }}',
    );
    expect(workflow).toContain('app / public traffic');
    expect(workflow).toContain('transitional runtime compatibility');
    expect(workflow).toContain('managed child-key creation and cleanup');
    expect(workflow).toContain('QQ production issuance');
    expect(workflow).toContain('TELEMETRY_SUBJECT_HMAC_SECRET');
    expect(workflow).toContain('assign the canonical production guardrail');
    expect(workflow).toContain('positive Qwen/DeepSeek/Gemini routing');
    expect(smokeSpec).not.toContain("process.env.CI === 'true'");
    expect(smokeSpec).not.toContain('smokeBaseUrl ||');
    expect(smokeSpec).toContain('BROKER_DEPLOY_SMOKE_RUN');
    expect(smokeSpec).toContain(
      "process.env.BROKER_DEPLOY_SMOKE_RUN === 'true'",
    );
    expect(liveInputValidationIndex).toBeGreaterThanOrEqual(0);
    expect(healthzProbeIndex).toBeGreaterThanOrEqual(0);
    expect(liveInputValidationIndex).toBeLessThan(healthzProbeIndex);
    expect(smokeSpec).toContain('/api/v1/key');
    expect(smokeSpec).toContain('/api/v1/chat/completions');
    expect(smokeSpec).toContain('BROKER_DEPLOY_SMOKE_DISALLOWED_MODEL');
    expect(smokeSpec).toContain('BROKER_DEPLOY_SMOKE_QQ_AUTH_HMAC_PSK');
    expect(smokeSpec).toContain('/v1/auth/qq/assert');
    expect(smokeSpec).toContain('deploy-smoke-qq-');
    expect(smokeSpec).toContain('ph-qq-subject-v1_');
    expect(smokeSpec).toContain("expect(qqAssertion.body.status).toBe('issued')");
    expect(smokeSpec).not.toContain("expect(qqAssertion.body.status).toBe('verified')");
    expect(smokeSpec).toContain('assertQqIssuedResponse');
    expect(smokeSpec).toContain('qqIssuedKey.openrouterApiKey');
    expect(smokeSpec).toContain('reads issued child-key metadata');
    expect(smokeSpec).toContain('recognizes model-routing failures as guardrail enforcement');
    expect(smokeSpec).toContain('assertSuccessfulChatCompletionResponse');
    expect(smokeSpec).toContain('assertManagedOpenRouterUserId');
    expect(smokeSpec).toContain('issue.body.openrouter_user_id');
    expect(smokeSpec).toContain('MANAGED_OPENROUTER_USER_ID_PATTERN');
    expect(smokeSpec).toContain('MANAGED_OPENROUTER_USER_ID_PATTERN.test(value)');
    expect(smokeSpec).not.toContain(
      'expect(value).toMatch(MANAGED_OPENROUTER_USER_ID_PATTERN)',
    );
    expect(smokeSpec).toContain('ph-or-user-v');
    expect(smokeSpec).toContain('MANAGED_TRIAL_ALLOWED_MODELS');
    expect(smokeSpec).toContain('google/gemma-4-31b-it');
    expect(smokeSpec).toContain('deepseek/deepseek-v4-flash-0731');
    expect(smokeSpec).toContain('deepseek/deepseek-v4-flash');
    expect(smokeSpec).toContain('MANAGED_TRIAL_ALLOWED_MODELS');
    expect(smokeSpec).toContain('must differ from the managed allowlisted models');
    expect(readme).toContain('per-installation OpenRouter child key');
    expect(readme).toContain('not the shared worker secret');
    expect(readme).toContain('BROKER_DEPLOY_SMOKE_DISALLOWED_MODEL_PRODUCTION');
    expect(readme).toContain('OPENROUTER_MANAGED_API_KEY_PRODUCTION` remains transitional');
    expect(readme).toContain('reconciles the production OpenRouter guardrail');
    expect(readme).toContain('OPENROUTER_MANAGED_USER_HMAC_SECRET_PRODUCTION');
    expect(readme).toContain('OPENROUTER_MANAGED_USER_HMAC_SECRET');
    expect(readme).toContain('QQ_AUTH_HMAC_PSK_PRODUCTION');
    expect(readme).toContain('QQ_AUTH_HMAC_PSK');
    expect(readme).toContain('TELEMETRY_SUBJECT_HMAC_SECRET_PRODUCTION');
    expect(readme).toContain('TELEMETRY_SUBJECT_HMAC_SECRET');
    expect(readme).toContain('POST /v1/auth/qq/assert');
    expect(readme).toContain('POST /v1/telemetry/app-active-day');
    expect(readme).toContain('production issuance-capable when runtime issuance configuration is present');
    expect(readme).toContain('issuance-disabled verification-only behavior');
    expect(readme).toContain('bounded retryable/internal error envelope');
    expect(readme).toContain('qq_managed_entitlements');
    expect(readme).toContain('broker_issue_success_events');
    expect(readme).toContain('issue_source');
    expect(readme).toContain('subject_ref');
    expect(readme).not.toContain('test-only `POST /v1/auth/qq/assert`');
    expect(readme).not.toContain('test-only QQ Bot assertion evidence');
    expect(readme).toContain('0008_add_qq_auth_assertions.sql');
    expect(readme).toContain('qq_auth_assertions');
    expect(readme).toContain('qqAuthAssertIp');
    expect(readme).toContain('DISCORD_CLIENT_ID_PRODUCTION');
    expect(readme).toContain('DISCORD_CLIENT_SECRET_PRODUCTION');
    expect(readme).toContain('DISCORD_REDIRECT_URI_ALLOWLIST_PRODUCTION');
    expect(readme).toContain('DISCORD_USER_REF_SECRET_PRODUCTION');
    expect(readme).toContain('DISCORD_CLIENT_ID');
    expect(readme).toContain('DISCORD_CLIENT_SECRET');
    expect(readme).toContain('DISCORD_REDIRECT_URI_ALLOWLIST');
    expect(readme).toContain('DISCORD_USER_REF_SECRET');
    expect(readme).toContain('DISCORD_OPERATIONS_WEBHOOK_URL_PRODUCTION');
    expect(readme).toContain('DISCORD_IMMEDIATE_ALERT_WEBHOOK_URL');
    expect(readme).toContain('DISCORD_DAILY_REPORT_WEBHOOK_URL');
    expect(readme).toContain('puripuly_daily_summary.v2');
    expect(workflow).toContain('deploy/finalize-daily-summary-v2.sql');
    expect(workflow).toContain(
      "json_type(value, '$.dailyReport.includeZeroActivity') AS legacy_type",
    );
    expect(workflow.indexOf('pnpm exec wrangler deploy')).toBeLessThan(
      workflow.indexOf('deploy/finalize-daily-summary-v2.sql'),
    );
    expect(workflow.indexOf('deploy/finalize-daily-summary-v2.sql')).toBeLessThan(
      workflow.indexOf('broker/tests/deploy-smoke/canonical-production.spec.ts'),
    );
    expect(readFileSync(dailySummaryV2Finalizer, 'utf8')).toContain(
      "json_remove(value, '$.dailyReport.includeZeroActivity')",
    );
    expect(workflow).toContain('deploy/finalize-app-active-day.sql');
    expect(workflow).toContain(
      "json_type(value, '$.telemetryTranslationSuccessDayIp') AS legacy_type",
    );
    expect(workflow.indexOf('pnpm exec wrangler deploy')).toBeLessThan(
      workflow.indexOf('deploy/finalize-app-active-day.sql'),
    );
    expect(workflow.indexOf('deploy/finalize-app-active-day.sql')).toBeLessThan(
      workflow.indexOf('broker/tests/deploy-smoke/canonical-production.spec.ts'),
    );
    expect(readFileSync(appActiveDayFinalizer, 'utf8')).toContain(
      "json_remove(value, '$.telemetryTranslationSuccessDayIp')",
    );
    expect(readme).toContain('three-month expiry');
    expect(readme).not.toContain('six-month expiry');
    expect(readme).toContain('optional `openrouter_user_id`');
    expect(readme).toContain('google/gemma-4-31b-it');
    expect(readme).toContain('deepseek/deepseek-v4-flash-0731');
    expect(readme).toContain('deepseek/deepseek-v4-flash');
  });

  it('ships a manual production workflow that updates only the broker daily auth cap runtime config', () => {
    const workflow = readFileSync(abuseControlsWorkflow, 'utf8');

    expect(workflow).toContain('workflow_dispatch:');
    expect(workflow).not.toContain('\npush:');
    expect(workflow).toContain('environment: production');
    expect(workflow).toContain('max_count');
    expect(workflow).toContain('default: "1000"');
    expect(workflow).toContain('confirm_update');
    expect(workflow).toContain('update broker daily auth cap');
    expect(workflow).toContain('CLOUDFLARE_API_TOKEN');
    expect(workflow).toContain('CLOUDFLARE_ACCOUNT_ID');
    expect(workflow).toContain('BROKER_D1_DATABASE_ID_PRODUCTION');
    expect(workflow).toContain('render-production-wrangler-config.mjs');
    expect(workflow).toContain('wrangler.production.jsonc');
    expect(workflow).toContain('wrangler d1 execute');
    expect(workflow).toContain('puripuly-heart-broker --remote --config');
    expect(workflow).toContain("json_set(value, '$.newActiveEntitlementsPerDay.maxCount'");
    expect(workflow).toContain("json_extract(value, '$.newActiveEntitlementsPerDay.maxCount')");
    expect(workflow).toContain('Daily auth cap verification failed');
    expect(workflow).not.toContain('wrangler deploy');
    expect(workflow).not.toContain('wrangler d1 migrations apply');
    expect(workflow).not.toContain('wrangler secret put');
  });
});

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

function extractBetween(source: string, startMarker: string, endMarker: string): string {
  const startIndex = source.indexOf(startMarker);
  expect(startIndex).toBeGreaterThanOrEqual(0);

  const contentStartIndex = startIndex + startMarker.length;
  const endIndex = source.indexOf(endMarker, contentStartIndex);
  expect(endIndex).toBeGreaterThanOrEqual(0);

  return source.slice(contentStartIndex, endIndex);
}
