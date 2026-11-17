import { readFile, writeFile } from 'node:fs/promises';
import { resolve, dirname } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { isDeepStrictEqual } from 'node:util';

const root = fileURLToPath(new URL('../', import.meta.url));
const policy = JSON.parse(await readFile(new URL('../support-lifecycle.json', import.meta.url), 'utf8'));
const desired = policy.rules[0];

function validateRules(rules) {
  if (!Array.isArray(rules)) throw new Error('Missing lifecycle rules.');
  for (const rule of rules) {
    if (rule.id === desired.id || rule.enabled !== true || !rule.deleteObjectsTransition) continue;
    const prefix = rule.conditions?.prefix;
    if (typeof prefix !== 'string' || prefix.startsWith('support/') || 'support/'.startsWith(prefix)) {
      throw new Error('Another expiration rule overlaps support/. Resolve it explicitly before enabling reception.');
    }
  }
}

export async function configureSupportStorage({ mode, bucket, account, token, enabledConfig, fetcher = fetch }) {
  if (!['verify', 'configure'].includes(mode) || !/^[a-z0-9][a-z0-9.-]{1,62}[a-z0-9]$/.test(bucket ?? '')
    || !/^[a-fA-F0-9]{32}$/.test(account ?? '') || !token?.trim()) {
    throw new Error('Use verify|configure <bucket> [--enabled-config <path>] with CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN.');
  }
  const base = `https://api.cloudflare.com/client/v4/accounts/${account}/r2/buckets/${encodeURIComponent(bucket)}`;
  async function api(path, body) {
    const response = await fetcher(`${base}/${path}`, {
      method: body === undefined ? 'GET' : 'PUT',
      headers: { authorization: `Bearer ${token}`, 'content-type': 'application/json' },
      ...(body === undefined ? {} : { body: JSON.stringify(body) }),
      redirect: 'error',
    });
    const value = await response.json();
    if (!response.ok || value.success !== true) throw new Error(`Cloudflare storage operation failed (${response.status}).`);
    return value.result;
  }
  async function privateAccess() {
    const managed = await api('domains/managed');
    const custom = await api('domains/custom');
    if (managed?.enabled !== false || !Array.isArray(custom?.domains) || custom.domains.length !== 0) {
      throw new Error('Support bucket must have r2.dev disabled and no custom domains before reception can be enabled.');
    }
  }
  await privateAccess();
  const before = await api('lifecycle');
  validateRules(before?.rules);
  const unrelated = before.rules.filter((rule) => rule.id !== desired.id);
  if (mode === 'configure') await api('lifecycle', { rules: [...unrelated, desired] });
  const after = await api('lifecycle');
  validateRules(after?.rules);
  const retention = after.rules.filter((rule) => rule.id === desired.id);
  if (retention.length !== 1 || !isDeepStrictEqual(retention[0], desired)
    || !isDeepStrictEqual(after.rules.filter((rule) => rule.id !== desired.id), unrelated)) {
    throw new Error('Lifecycle readback did not preserve unrelated rules and exact 30-day support/ expiration.');
  }
  await privateAccess();
  if (enabledConfig) {
    const source = resolve(root, 'wrangler.jsonc');
    const destination = resolve(enabledConfig);
    if (source === destination || dirname(destination) !== resolve(root)) {
      throw new Error('Enabled config must be a separate file inside agent-relay so Wrangler paths remain correct.');
    }
    const config = JSON.parse(await readFile(source, 'utf8'));
    const binding = config.r2_buckets.find((entry) => entry.binding === 'SUPPORT_REPORTS');
    if (!binding) throw new Error('SUPPORT_REPORTS binding is missing.');
    binding.bucket_name = bucket;
    config.account_id = account;
    config.vars.SUPPORT_REPORTS_ENABLED = 'true';
    await writeFile(destination, `${JSON.stringify(config, null, 2)}\n`, { flag: 'wx' });
  }
  return { bucket, private: true, retention_days: 30, prefix: 'support/', enabled_config_created: Boolean(enabledConfig) };
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  try {
    const [mode, bucket, option, enabledConfig, ...extra] = process.argv.slice(2);
    if (extra.length || (option !== undefined && (option !== '--enabled-config' || !enabledConfig))) {
      throw new Error('Use verify|configure <bucket> [--enabled-config <path>].');
    }
    console.log(JSON.stringify(await configureSupportStorage({ mode, bucket, enabledConfig,
      account: process.env.CLOUDFLARE_ACCOUNT_ID, token: process.env.CLOUDFLARE_API_TOKEN })));
  } catch (error) {
    console.error(error instanceof Error ? error.message : 'Support storage configuration failed.');
    process.exitCode = 1;
  }
}
