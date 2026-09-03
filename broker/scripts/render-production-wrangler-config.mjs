import { readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const CANONICAL_WORKER_NAME = 'puripuly-heart-broker';
const DATABASE_ID_PLACEHOLDER = 'REQUIRED_AT_DEPLOY_TIME';

await main();

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const sourcePath = resolve(
    args.source ?? fileURLToPath(new URL('../wrangler.jsonc', import.meta.url)),
  );
  const outputPath = resolve(requiredArg(args, 'out'));
  const databaseId = requiredArg(args, 'database-id');
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

  const databaseIdPlaceholderPattern = new RegExp(
    `"database_id"\\s*:\\s*"${DATABASE_ID_PLACEHOLDER}"`,
    'gu',
  );
  const placeholderMatches = sourceText.match(databaseIdPlaceholderPattern) ?? [];

  if (placeholderMatches.length !== 1) {
    throw new Error(
      `expected exactly one ${DATABASE_ID_PLACEHOLDER} database_id placeholder`,
    );
  }

  const withDatabaseId = renderedConfig.replace(
    databaseIdPlaceholderPattern,
    `"database_id": ${JSON.stringify(databaseId)}`,
  );
  if (withDatabaseId === renderedConfig) {
    throw new Error(
      `expected exactly one ${DATABASE_ID_PLACEHOLDER} database_id placeholder`,
    );
  }

  await writeFile(outputPath, withDatabaseId, 'utf8');
  process.stdout.write(`${outputPath}\n`);
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
