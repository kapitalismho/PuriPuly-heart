import { readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const BOOTSTRAP_PLACEHOLDER = '__BOOTSTRAP_REQUIRED__';

await main();

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const templatePath = resolve(
    args.template ??
      fileURLToPath(
        new URL('../deploy/fingerprint-bootstrap.template.sql', import.meta.url),
      ),
  );
  const outputPath = resolve(requiredArg(args, 'out'));
  const salt = requiredArg(args, 'salt');

  if (!salt.trim()) {
    throw new Error('fingerprint bootstrap salt must not be blank');
  }

  if (salt === BOOTSTRAP_PLACEHOLDER) {
    throw new Error('fingerprint bootstrap salt must replace the bootstrap placeholder');
  }

  const template = await readFile(templatePath, 'utf8');
  const placeholderPattern = new RegExp(BOOTSTRAP_PLACEHOLDER, 'gu');
  const placeholderCount = template.match(placeholderPattern)?.length ?? 0;

  if (placeholderCount < 1) {
    throw new Error('fingerprint bootstrap SQL template must contain the bootstrap placeholder');
  }

  const escapedSalt = escapeSqlStringLiteral(salt);
  const renderedSql = template.replace(placeholderPattern, escapedSalt);

  if (renderedSql.includes(BOOTSTRAP_PLACEHOLDER)) {
    throw new Error('fingerprint bootstrap SQL still contains the bootstrap placeholder');
  }

  await writeFile(outputPath, renderedSql, 'utf8');
  process.stdout.write(`${outputPath}\n`);
}

function escapeSqlStringLiteral(value) {
  return value.replace(/'/gu, "''");
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
