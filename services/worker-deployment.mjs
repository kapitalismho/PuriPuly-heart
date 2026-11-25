import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const zone = 'salee.sarl';
const accountSubdomain = 'kapitalismho';
const cloudflareId = /^(?!0{32}$)[0-9a-f]{32}$/i;

function target(workerName, label, legacyVariable) {
  const hostname = `${label}.${zone}`;
  return {
    workerName, legacyVariable,
    origin: `https://${hostname}`,
    legacyOrigin: `https://${workerName}.${accountSubdomain}.workers.dev`,
    route: { pattern: hostname, zone_name: zone, custom_domain: true },
  };
}

export const brokerDeployment = target('puripuly-heart-broker', 'puripuly-broker', 'BROKER_CANONICAL_WORKERS_DEV_URL');
export const relayDeployment = target('puripuly-heart-agent-relay', 'puripuly-relay', 'AGENT_RELAY_CANONICAL_WORKERS_DEV_URL');

export function isExactOrigin(value, origin) {
  return value === origin || value === `${origin}/`;
}

export function requireLegacyOrigin(env, deployment) {
  if (!isExactOrigin(env[deployment.legacyVariable], deployment.legacyOrigin)) {
    throw new Error(`${deployment.legacyVariable} must be ${deployment.legacyOrigin} (optional trailing slash).`);
  }
  return deployment.legacyOrigin;
}

export function requireAccountId(value) {
  if (typeof value !== 'string' || !cloudflareId.test(value)) {
    throw new Error('CLOUDFLARE_ACCOUNT_ID must be a real account ID.');
  }
  return value;
}

export async function verifyWorkerTarget(env, deployment, fetcher = fetch) {
  requireLegacyOrigin(env, deployment);
  const account = requireAccountId(env.CLOUDFLARE_ACCOUNT_ID);
  const token = env.CLOUDFLARE_API_TOKEN;
  if (!token?.trim()) throw new Error('CLOUDFLARE_API_TOKEN is required.');
  async function read(path, failure) {
    try {
      const response = await fetcher(`https://api.cloudflare.com/client/v4/${path}`, {
        method: 'GET', headers: { authorization: `Bearer ${token}` },
        redirect: 'error', signal: AbortSignal.timeout(10000),
      });
      if (response.ok) {
        const payload = await response.json();
        if (payload?.success === true && payload.result != null) return payload;
      }
    } catch {
      throw new Error(failure);
    }
    throw new Error(failure);
  }
  async function list(path, failure) {
    const { result, result_info: info } = await read(path, failure);
    if (!Array.isArray(result) || (info !== undefined && (
      !info || typeof info !== 'object' || Array.isArray(info)
      || ('page' in info && info.page !== 1)
      || ('total_pages' in info && info.total_pages !== 0 && info.total_pages !== 1)
      || ('count' in info && info.count !== result.length)
    ))) throw new Error(failure);
    return result;
  }
  const workers = `accounts/${account}/workers`;
  const subdomainFailure = `Cloudflare account must own the ${accountSubdomain}.workers.dev subdomain.`;
  const { result: { subdomain } } = await read(`${workers}/subdomain`, subdomainFailure);
  if (subdomain !== accountSubdomain) throw new Error(subdomainFailure);
  const workerFailure = `Cloudflare account must already contain ${deployment.workerName}.`;
  const { result: availability } = await read(`${workers}/scripts/${deployment.workerName}/subdomain`, workerFailure);
  if (typeof availability.enabled !== 'boolean') throw new Error(workerFailure);

  const zoneFailure = `Expected one active ${zone} zone in the production Cloudflare account.`;
  const zones = await list(`zones?name=${zone}&account.id=${account}&match=all`, zoneFailure);
  const activeZone = zones[0];
  if (zones.length !== 1 || activeZone?.name !== zone || activeZone.status !== 'active'
    || !cloudflareId.test(activeZone.id ?? '') || activeZone.account?.id !== account.toLowerCase()) {
    throw new Error(zoneFailure);
  }

  const hostname = deployment.route.pattern;
  const domainFailure = `Custom Domains for ${hostname} and ${deployment.workerName} must be empty or matched to each other.`;
  const domains = `${workers}/domains`;
  const [atHost, atWorker] = await Promise.all([
    list(`${domains}?hostname=${hostname}`, domainFailure),
    list(`${domains}?service=${deployment.workerName}`, domainFailure),
  ]);
  const matchingDomain = (domain) => typeof domain?.id === 'string' && domain.id.length > 0
    && domain.hostname === hostname && domain.service === deployment.workerName
    && domain.zone_id === activeZone.id && domain.environment === 'production';
  const attached = atHost.length === 1 && atWorker.length === 1
    && matchingDomain(atHost[0]) && matchingDomain(atWorker[0]) && atHost[0].id === atWorker[0].id;
  if (!attached && (atHost.length || atWorker.length)) throw new Error(domainFailure);

  const dnsFailure = `DNS for ${hostname} must be empty for first deployment or contain only its Cloudflare-managed Worker record.`;
  const records = await list(`zones/${activeZone.id}/dns_records?name.exact=${hostname}`, dnsFailure);
  const record = records[0];
  if (attached
    ? records.length !== 1 || record?.name !== hostname || record.type !== 'AAAA' || record.content !== '100::'
      || record.proxied !== true || record.meta?.read_only !== true
      || typeof record.meta.origin_worker_id !== 'string' || !record.meta.origin_worker_id
    : records.length !== 0) throw new Error(dnsFailure);
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  try {
    const service = process.argv[2];
    const deployment = service === 'broker' ? brokerDeployment : service === 'relay' ? relayDeployment : null;
    if (process.argv.length !== 3 || !deployment) throw new Error('Use broker | relay to verify the production Cloudflare target.');
    await verifyWorkerTarget(process.env, deployment);
  } catch (error) {
    console.error(error instanceof Error ? error.message : 'Cloudflare deployment target verification failed.');
    process.exitCode = 1;
  }
}
