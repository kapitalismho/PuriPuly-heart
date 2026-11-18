import assert from 'node:assert/strict';
import { appendFileSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { DatabaseSync } from 'node:sqlite';

const root = process.env.SCENARIO_ROOT;
const mode = process.env.SCENARIO_MODE;
const args = process.argv.slice(2);
const event = (value) => appendFileSync(join(root, 'events.jsonl'), `${JSON.stringify(value)}\n`);
const forbidden = () => { event({ event: 'forbidden', args }); throw new Error('Unexpected Cloudflare command or mutation.'); };
function allowFlags(start, valued, switches) {
  for (let index = start; index < args.length; index++) {
    if (valued.includes(args[index])) {
      if (!args[++index] || args[index].startsWith('--')) forbidden();
    } else if (!switches.includes(args[index])) forbidden();
  }
}
if (args[0] !== 'exec' || args[1] !== 'wrangler' || !args.includes('--config')) forbidden();
const configPath = args[args.indexOf('--config') + 1];
const config = JSON.parse(readFileSync(configPath, 'utf8'));
assert.equal(config.name, 'puripuly-heart-agent-relay');
assert.equal(config.account_id, process.env.CLOUDFLARE_ACCOUNT_ID);
assert.equal(config.d1_databases.length, 1);
assert.equal(config.d1_databases[0].binding, 'AGENT_USAGE_DB');
assert.equal(config.d1_databases[0].database_id, process.env.AGENT_USAGE_D1_DATABASE_ID_PRODUCTION);
const databasePath = join(root, 'synthetic.sqlite');

if (args[2] === 'd1' && args[3] === 'execute') {
  allowFlags(5, ['--config', '--command'], ['--remote', '--json']);
  assert.equal(args[4], 'AGENT_USAGE_DB');
  assert(args.includes('--remote') && args.includes('--json'));
  const sql = args[args.indexOf('--command') + 1];
  if (!args.includes('--command') || !/^SELECT\s/iu.test(sql) || args.includes('--file')) forbidden();
  const previous = readFileSync(join(root, 'events.jsonl'), { encoding: 'utf8', flag: 'a+' });
  event({ event: 'read' });
  if (mode === 'read-failure' || (mode === 'final-read-failure' && previous.includes('"event":"migrate"'))) process.exit(32);
  if (mode === 'malformed-read') { process.stdout.write('{'); process.exit(0); }
  const database = new DatabaseSync(databasePath, { readOnly: true });
  try {
    const results = database.prepare(sql).all();
    if (mode === 'malformed-rows') results.push({ kind: 'column', name: 42, table_name: 'usage_daily' });
    process.stdout.write(JSON.stringify([{ success: mode !== 'unsuccessful-read', results }]));
  } finally { database.close(); }
} else if (args[2] === 'd1' && args[3] === 'migrations' && args[4] === 'apply') {
  allowFlags(6, ['--config'], ['--remote']);
  if (process.env.APPLY_MIGRATIONS !== 'true') forbidden();
  assert.equal(args[5], 'AGENT_USAGE_DB');
  assert(args.includes('--remote'));
  const directory = resolve(dirname(configPath), config.d1_databases[0].migrations_dir);
  const database = new DatabaseSync(databasePath);
  try {
    const applied = new Set(database.prepare('SELECT name FROM d1_migrations').all().map((row) => row.name));
    const pending = readdirSync(directory).filter((name) => name.endsWith('.sql') && !applied.has(name)).sort();
    event({ event: 'migrate', pending });
    for (const name of pending) {
      database.exec('BEGIN');
      try {
        database.exec(readFileSync(join(directory, name), 'utf8'));
        if (mode === 'migration-failure') throw new Error('Injected migration failure.');
        database.prepare('INSERT INTO d1_migrations (name) VALUES (?)').run(name);
        database.exec('COMMIT');
      } catch {
        database.exec('ROLLBACK');
        event({ event: 'migration-failed' });
        process.exitCode = 37;
        break;
      }
    }
  } finally { database.close(); }
} else if (args[2] === 'deploy') {
  allowFlags(3, ['--config', '--tag', '--message'], ['--keep-vars', '--experimental-provision=false', '--experimental-auto-create=false', '--no-autoconfig']);
  for (const flag of ['--experimental-provision=false', '--experimental-auto-create=false', '--no-autoconfig']) assert(args.includes(flag));
  assert.equal(args[args.indexOf('--tag') + 1], process.env.GITHUB_SHA);
  assert(args[args.indexOf('--message') + 1].includes(process.env.GITHUB_SHA));
  event({ event: 'deploy' });
  if (mode === 'deploy-failure') process.exit(41);
  const path = join(root, 'worker.json');
  const worker = JSON.parse(readFileSync(path, 'utf8'));
  for (const name of config.secrets.required) assert.equal(typeof worker.secrets[name], 'string');
  worker.vars = { ...(args.includes('--keep-vars') ? worker.vars : {}), ...config.vars };
  worker.reportsBound = Boolean(config.r2_buckets?.some((binding) => binding.binding === 'SUPPORT_REPORTS'));
  worker.deployments++;
  writeFileSync(path, JSON.stringify(worker));
} else forbidden();
