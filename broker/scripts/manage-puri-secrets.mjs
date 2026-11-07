import { spawnSync } from 'node:child_process';

const names = ['PURI_TOKEN_SECRET', 'PURI_IDENTITY_HMAC_SECRET'];
const values = names.map((name) => process.env[`${name}_PRODUCTION`] ?? '');
const configured = values.some((value) => value.length > 0);
if (configured && (values.some((value) => Buffer.byteLength(value, 'utf8') < 32) || values[0] === values[1])) {
  throw new Error('Puri requires two distinct secrets of at least 32 UTF-8 bytes, or neither secret');
}

const args = process.argv.slice(2);
if (args.length === 1 && args[0] === '--validate') {
  process.exit(0);
}
if (args.length !== 2 || args[0] !== '--config' || !args[1]) {
  throw new Error('Expected --validate or --config <wrangler-config>');
}
const config = args[1];
function wrangler(command, input) {
  const result = spawnSync('pnpm', ['exec', 'wrangler', ...command, '--config', config], {
    encoding: 'utf8', input, env: { ...process.env, CI: 'true' },
  });
  if (result.error || result.status !== 0) {
    throw new Error('Puri Worker secret synchronization failed');
  }
  return result.stdout;
}

if (configured) {
  wrangler(['secret', 'bulk'], JSON.stringify(Object.fromEntries(names.map((name, index) => [name, values[index]]))));
}
