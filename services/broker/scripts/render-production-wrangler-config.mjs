import { readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { brokerDeployment, requireAccountId } from '../../worker-deployment.mjs';

const CANONICAL_WORKER_NAME = brokerDeployment.workerName;
const DATABASE_ID_PLACEHOLDER = 'REQUIRED_AT_DEPLOY_TIME';
const DATABASE_ID_PLACEHOLDERS = {
  BROKER_DB: DATABASE_ID_PLACEHOLDER,
  AGENT_USAGE_DB: `AGENT_USAGE_${DATABASE_ID_PLACEHOLDER}`,
};
const DATABASE_ID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/iu;
const NIL_DATABASE_ID = '00000000-0000-0000-0000-000000000000';

await main();

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const sourcePath = resolve(
    args.source ?? fileURLToPath(new URL('../wrangler.jsonc', import.meta.url)),
  );
  const outputPath = resolve(requiredArg(args, 'out'));
  const databaseId = requiredArg(args, 'database-id');
  const agentUsageDatabaseId = requiredArg(args, 'agent-usage-database-id');
  assertDatabaseId(databaseId, '--database-id');
  assertDatabaseId(agentUsageDatabaseId, '--agent-usage-database-id');
  if (databaseId.toLowerCase() === agentUsageDatabaseId.toLowerCase()) {
    throw new Error('--agent-usage-database-id must differ from the Broker --database-id.');
  }
  const keyVersion = requiredArg(args, 'network-identity-hmac-key-version');
  const keyVersionPrevious = args['network-identity-hmac-key-version-previous'];
  assertPositiveInteger(keyVersion, '--network-identity-hmac-key-version must be a positive integer.');
  const previousVersions =
    keyVersionPrevious === undefined || keyVersionPrevious === ''
      ? []
      : [keyVersionPrevious];
  for (const previous of previousVersions) {
    assertPositiveInteger(
      previous,
      '--network-identity-hmac-key-version-previous must be a positive integer when set.',
    );
    if (previous === keyVersion) {
      throw new Error(
        '--network-identity-hmac-key-version-previous must differ from --network-identity-hmac-key-version.',
      );
    }
  }
  const sourceText = await readFile(sourcePath, 'utf8');
  const nameMatch = sourceText.match(/"name"\s*:\s*"([^"]+)"/u);

  if (!nameMatch) {
    throw new Error('wrangler config is missing a worker name field');
  }

  if (nameMatch[1] !== CANONICAL_WORKER_NAME) {
    throw new Error(
      `wrangler config must keep the canonical worker name ${CANONICAL_WORKER_NAME}`,
    );
  }

  const versionEntries = [
    `    "NETWORK_IDENTITY_HMAC_KEY_VERSION": ${JSON.stringify(keyVersion)}`,
    ...previousVersions.map(
      (previous) => `    "NETWORK_IDENTITY_HMAC_KEY_VERSION_PREVIOUS": ${JSON.stringify(previous)}`,
    ),
  ];
  const renderedConfig = injectVersionVars(sourceText, versionEntries);

  const withBrokerDatabaseId = replaceDatabaseId(renderedConfig, 'BROKER_DB', databaseId);
  const withDatabaseIds = replaceDatabaseId(withBrokerDatabaseId, 'AGENT_USAGE_DB', agentUsageDatabaseId);
  if (withDatabaseIds.includes(DATABASE_ID_PLACEHOLDER)) {
    throw new Error('wrangler config has unexpected remaining deploy-time placeholders.');
  }

  const accountId = args['account-id'];
  const output = accountId === undefined ? withDatabaseIds : addProductionRouting(withDatabaseIds, accountId);

  await writeFile(outputPath, output, 'utf8');
  process.stdout.write(`${outputPath}\n`);
}

function addProductionRouting(config, accountId) {
  requireAccountId(accountId);
  if (/"(?:account_id|route|routes|workers_dev)"\s*:/u.test(config)) {
    throw new Error('Broker production routing must come from the deployment renderer.');
  }
  return config.replace(/"name"\s*:/u, [
    `"account_id": ${JSON.stringify(accountId)},`,
    '"workers_dev": true,',
    `"routes": ${JSON.stringify([brokerDeployment.route])},`,
    '"name":',
  ].join('\n  '));
}

function injectVersionVars(sourceText, versionEntries) {
  const lines = sourceText.split('\n');
  const openIndex = lines.findIndex((line) => /^\s*"vars"\s*:\s*\{\s*$/u.test(line));
  if (openIndex === -1) {
    const trimmedEnd = sourceText.replace(/\s+$/u, '');
    if (!trimmedEnd.endsWith('}')) {
      throw new Error('wrangler config does not end with a top-level object');
    }
    const withoutClose = trimmedEnd.slice(0, -1).replace(/\s+$/u, '');
    const separator = withoutClose.endsWith('{') ? '\n' : ',\n';
    return `${withoutClose}${separator}  "vars": {\n${versionEntries.join(',\n')}\n  }\n}\n`;
  }
  let depth = 0;
  let closeIndex = -1;
  for (let index = openIndex; index < lines.length; index += 1) {
    depth += (lines[index].match(/\{/gu) ?? []).length;
    depth -= (lines[index].match(/\}/gu) ?? []).length;
    if (depth === 0) {
      closeIndex = index;
      break;
    }
  }
  if (closeIndex === -1) {
    throw new Error('wrangler config has an unterminated vars block');
  }
  const kept = [];
  for (let index = openIndex + 1; index < closeIndex; index += 1) {
    if (/NETWORK_IDENTITY_HMAC_KEY_VERSION/u.test(lines[index])) {
      continue;
    }
    kept.push(lines[index].replace(/,\s*$/u, ''));
  }
  const body = [...kept, ...versionEntries].join(',\n');
  return [...lines.slice(0, openIndex + 1), body, ...lines.slice(closeIndex)].join('\n');
}

function replaceDatabaseId(sourceText, binding, databaseId) {
  const bindingPattern = new RegExp(`\\{[^{}]*"binding"\\s*:\\s*"${binding}"[^{}]*\\}`, 'gu');
  const bindings = [...sourceText.matchAll(bindingPattern)];
  const placeholder = DATABASE_ID_PLACEHOLDERS[binding];
  const placeholderPattern = new RegExp(
    `"database_id"\\s*:\\s*"${placeholder}"`,
    'gu',
  );
  if (bindings.length !== 1 || (bindings[0][0].match(placeholderPattern) ?? []).length !== 1) {
    throw new Error(`expected exactly one ${binding} binding with a ${placeholder} database_id.`);
  }
  return sourceText.replace(bindingPattern, (entry) => entry.replace(
    placeholderPattern,
    `"database_id": ${JSON.stringify(databaseId)}`,
  ));
}

function assertDatabaseId(value, argument) {
  if (!DATABASE_ID_PATTERN.test(value) || value.toLowerCase() === NIL_DATABASE_ID) {
    throw new Error(`${argument} must be a real D1 database UUID.`);
  }
}

function assertPositiveInteger(value, message) {
  if (!/^[1-9][0-9]*$/u.test(value) || !Number.isSafeInteger(Number(value))) {
    throw new Error(message);
  }
}

function parseArgs(argv) {
  const args = {};

  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];

    if (!token?.startsWith('--')) {
      throw new Error(`unexpected argument: ${token ?? '<missing>'}`);
    }

    const key = token.slice(2);
    const value = argv[index + 1];

    if (!value || value.startsWith('--')) {
      throw new Error(`missing value for --${key}`);
    }

    args[key] = value;
    index += 1;
  }

  return args;
}

function requiredArg(args, key) {
  const value = args[key];

  if (!value) {
    throw new Error(`missing required --${key} argument`);
  }

  return value;
}
