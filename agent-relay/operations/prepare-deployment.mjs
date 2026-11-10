import { execFileSync } from 'node:child_process';
import { mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import { isAbsolute, relative, resolve, sep } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { configureSupportStorage } from './support-storage.mjs';

const root = fileURLToPath(new URL('../', import.meta.url));
export const workerName = 'puripuly-heart-agent-relay';
export const confirmation = `deploy ${workerName} from dev`;
const uuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const requiredColumns = {
  usage_daily: ['installation_hash', 'utc_date', 'uncached_input_tokens', 'cache_read_tokens', 'cache_write_tokens',
    'decisions_output_tokens', 'responses_output_tokens', 'decisions_calls', 'responses_calls', 'missing_usage_count', 'cost_usd'],
  support_daily: ['installation_hash', 'utc_date', 'reports'],
  prompt_turns: ['installation_hash', 'turn_id', 'utc_date', 'expires_at', 'model_admissions', 'classifier_admitted'],
  request_slots: ['installation_hash', 'slot', 'owner', 'turn_id', 'classification', 'expires_at'],
  protocol_daily: ['installation_hash', 'utc_date', 'agent_protocol', 'requests'],
  photo_admissions: ['installation_hash', 'request_id', 'period_start', 'owner', 'expires_at', 'settled',
    'uncached_input_tokens', 'cache_read_tokens', 'cache_write_tokens', 'output_tokens', 'missing_usage', 'cost_usd'],
};

function required(env, name) {
  if (typeof env[name] !== 'string' || !env[name].trim()) throw new Error(`${name} is required.`);
  return env[name];
}

export function deploymentOrigin(value) {
  if (!/^https:\/\/puripuly-heart-agent-relay\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.workers\.dev\/?$/.test(value ?? '')) {
    throw new Error('AGENT_RELAY_CANONICAL_WORKERS_DEV_URL must be the canonical HTTPS workers.dev origin.');
  }
  return new URL(value).origin;
}

export function guardDeployment(env, head) {
  if (env.GITHUB_EVENT_NAME !== 'workflow_dispatch' || env.GITHUB_REF !== 'refs/heads/dev') {
    throw new Error('Deploy only by workflow_dispatch from refs/heads/dev.');
  }
  if (env.CONFIRM_PRODUCTION_DEPLOY !== confirmation) throw new Error('Production deployment confirmation does not match.');
  const reviewed = env.REVIEWED_COMMIT;
  if (!/^[0-9a-f]{40}$/.test(reviewed ?? '') || reviewed !== env.GITHUB_SHA || reviewed !== head) {
    throw new Error('Reviewed commit, dispatch commit and checked-out HEAD must be the same full lowercase SHA.');
  }
  for (const name of ['AGENT_SERVICE_ENABLED', 'SUPPORT_REPORTS_ENABLED']) {
    if (!['true', 'false'].includes(env[name])) throw new Error(`${name} must explicitly be true or false.`);
  }
  if (!['', undefined, 'true', 'false'].includes(env.PHOTO_SERVICE_ENABLED)) throw new Error('PHOTO_SERVICE_ENABLED must be true or false when set.');
  if (![undefined, '', 'false', 'true'].includes(env.APPLY_MIGRATIONS)) throw new Error('APPLY_MIGRATIONS must be empty, false or true.');
  deploymentOrigin(env.AGENT_RELAY_CANONICAL_WORKERS_DEV_URL);
}

export async function prepareDeployment({ env, directory, fetcher = fetch }) {
  const account = required(env, 'CLOUDFLARE_ACCOUNT_ID');
  if (!/^[a-f0-9]{32}$/i.test(account)) throw new Error('CLOUDFLARE_ACCOUNT_ID must be a real account ID.');
  required(env, 'CLOUDFLARE_API_TOKEN');
  const database = required(env, 'AGENT_USAGE_D1_DATABASE_ID_PRODUCTION');
  const brokerDatabase = required(env, 'BROKER_D1_DATABASE_ID_PRODUCTION');
  if (!uuid.test(database) || !uuid.test(brokerDatabase) || database.toLowerCase() === brokerDatabase.toLowerCase()
    || /^0{8}-0{4}-0{4}-0{4}-0{12}$/.test(database)) {
    throw new Error('A real dedicated relay D1 UUID, distinct from broker D1, is required.');
  }
  for (const name of ['AGENT_SERVICE_ENABLED', 'SUPPORT_REPORTS_ENABLED']) {
    if (!['true', 'false'].includes(env[name])) throw new Error(`${name} must explicitly be true or false.`);
  }
  if (!['', undefined, 'true', 'false'].includes(env.PHOTO_SERVICE_ENABLED)) throw new Error('PHOTO_SERVICE_ENABLED must be true or false when set.');
  deploymentOrigin(env.AGENT_RELAY_CANONICAL_WORKERS_DEV_URL);
  const token = required(env, 'PURI_TOKEN_SECRET_PRODUCTION');
  if (Buffer.byteLength(token, 'utf8') < 32) throw new Error('PURI_TOKEN_SECRET_PRODUCTION must contain at least 32 UTF-8 bytes and match broker.');
  const config = JSON.parse(await readFile(resolve(root, 'wrangler.jsonc'), 'utf8'));
  if (config.name !== workerName || config.d1_databases?.length !== 1
    || config.d1_databases[0].binding !== 'AGENT_USAGE_DB'
    || config.d1_databases[0].database_id !== 'REQUIRED_AT_DEPLOY_TIME'
    || config.r2_buckets?.length !== 1 || config.r2_buckets[0].binding !== 'SUPPORT_REPORTS') {
    throw new Error('Checked-in relay configuration must keep the canonical Worker and dedicated storage bindings.');
  }
  const destination = resolve(directory);
  const within = relative(resolve(root, '..'), destination);
  if (within !== '..' && !within.startsWith(`..${sep}`) && !isAbsolute(within)) throw new Error('Deployment output must be outside the checkout.');
  if (env.SUPPORT_REPORTS_ENABLED === 'true') {
    await configureSupportStorage({ mode: 'verify', bucket: config.r2_buckets[0].bucket_name,
      account, token: env.CLOUDFLARE_API_TOKEN, fetcher });
  } else {
    delete config.r2_buckets;
  }
  config.account_id = account;
  config.workers_dev = true;
  config.preview_urls = false;
  config.observability = { enabled: false };
  config.main = resolve(root, config.main);
  config.d1_databases[0].database_id = database;
  config.d1_databases[0].migrations_dir = resolve(root, 'migrations');
  config.vars.AGENT_SERVICE_ENABLED = env.AGENT_SERVICE_ENABLED;
  config.vars.SUPPORT_REPORTS_ENABLED = env.SUPPORT_REPORTS_ENABLED;
  config.vars.PHOTO_SERVICE_ENABLED = env.PHOTO_SERVICE_ENABLED || 'false';
  if (env.PHOTO_POLICY) config.vars.PHOTO_POLICY = env.PHOTO_POLICY;
  config.secrets.required = ['PURI_TOKEN_SECRET', ...(
    env.AGENT_SERVICE_ENABLED === 'true' || env.PHOTO_SERVICE_ENABLED === 'true' ? ['OPENAI_API_KEY'] : []
  )];
  await mkdir(destination, { recursive: true, mode: 0o700 });
  await writeFile(resolve(destination, 'wrangler.json'), `${JSON.stringify(config, null, 2)}\n`, { flag: 'wx', mode: 0o600 });
}

export async function verifyDeploymentSchema({ payload, initializedOnly = false }) {
  if (!Array.isArray(payload) || payload.length !== 1 || payload[0]?.success !== true || !Array.isArray(payload[0].results)) {
    throw new Error('Expected exactly one successful Wrangler schema result.');
  }
  const applied = new Set();
  const columns = new Map();
  for (const row of payload[0].results) {
    if (!row || !['migration', 'column'].includes(row.kind) || typeof row.name !== 'string' || !row.name
      || typeof row.table_name !== 'string' || (row.kind === 'migration' ? row.table_name !== '' : !row.table_name)) {
      throw new Error('Malformed Wrangler schema row.');
    }
    if (row.kind === 'migration') {
      applied.add(row.name);
    } else {
      if (!columns.has(row.table_name)) columns.set(row.table_name, new Set());
      columns.get(row.table_name).add(row.name);
    }
  }
  const migrations = initializedOnly ? ['0001_usage_daily.sql']
    : (await readdir(resolve(root, 'migrations'))).filter((name) => name.endsWith('.sql')).sort();
  for (const name of migrations) {
    if (!applied.has(name)) throw new Error(`Relay database is missing migration ${name}.`);
  }
  for (const [table, names] of Object.entries(requiredColumns)) {
    if (initializedOnly && table !== 'usage_daily') continue;
    for (const name of names) {
      if (!columns.get(table)?.has(name)) throw new Error(`Relay database is missing required column ${table}.${name}.`);
    }
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  try {
    const [mode, target, ...extra] = process.argv.slice(2);
    if (extra.length || !['guard', 'prepare', 'initialized', 'schema'].includes(mode)
      || (mode === 'guard' ? target !== undefined : !target)) {
      throw new Error('Use guard | prepare <external directory> | initialized <schema-json-file> | schema <schema-json-file>.');
    }
    if (mode === 'guard') guardDeployment(process.env, execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim());
    if (mode === 'prepare') await prepareDeployment({ env: process.env, directory: target });
    if (mode === 'initialized' || mode === 'schema') {
      await verifyDeploymentSchema({ payload: JSON.parse(await readFile(target, 'utf8')), initializedOnly: mode === 'initialized' });
    }
    console.log(`Relay deployment ${mode} completed.`);
  } catch (error) {
    console.error(error instanceof Error ? error.message : 'Relay deployment preparation failed.');
    process.exitCode = 1;
  }
}
