import { spawnSync } from 'node:child_process';
import { createHmac, generateKeyPairSync, randomBytes, randomUUID, sign, timingSafeEqual } from 'node:crypto';
import { appendFileSync } from 'node:fs';
import { copyFile, mkdir, mkdtemp, readdir, readFile, rm, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const broker = fileURLToPath(new URL('../', import.meta.url));
const relay = join(broker, '..', 'agent-relay');
const require = createRequire(import.meta.url);
const wrangler = require.resolve('wrangler');
const appVersion = '1.2.3';
const endpoint = '/v1/puri/token';
const expectedState = { installations: 18, bound: 18, enrollments: 4, nonces: 76, flood: 80, installation: 77 };
const expectedErrors = new Map([
  ['bad-signature', [401, 'puri_proof_invalid']], ['replay', [401, 'puri_proof_invalid']],
  ['binding-mismatch', [409, 'installation_binding_mismatch']], ['installation-limit', [429, 'installation_rate_limited']],
]);
const d1Fields = ['calls', 'waitMs', 'statements', 'meteredStatements', 'sqlMs'];

function numbers(value, fields) {
  return Object.fromEntries(fields.map((field) => [field, Number.isFinite(value?.[field]) ? value[field] : null]));
}

async function readJournal(path) {
  let text;
  try {
    text = await readFile(path, 'utf8');
  } catch (error) {
    return { samples: [], stage: 'setup', journalIncomplete: error.code !== 'ENOENT' };
  }
  const lines = text.split('\n');
  let journalIncomplete = lines.pop() !== '';
  const samples = [];
  let stage = 'setup';
  let brokerMigrations, elapsedMs, state;
  for (const line of lines) {
    let entry;
    try { entry = JSON.parse(line); }
    catch { journalIncomplete = true; break; }
    if (entry.event === 'start') {
      samples.push({ attempt: entry.attempt, kind: entry.kind, status: null, outcome: 'interrupted', matched: null, elapsedMs: null });
    } else if (entry.event === 'finish' && samples.at(-1)?.attempt === entry.sample.attempt) {
      samples[samples.length - 1] = entry.sample;
    } else if (entry.event === 'workload') {
      stage = 'workload';
      brokerMigrations = entry.brokerMigrations;
    } else if (entry.event === 'shutdown') {
      stage = 'shutdown';
      elapsedMs = entry.elapsedMs;
    } else if (entry.event === 'snapshot') {
      stage = 'snapshot';
    } else if (entry.event === 'done') {
      stage = 'done';
      state = entry.state;
    } else {
      journalIncomplete = true;
      break;
    }
  }
  return { samples, stage, brokerMigrations, elapsedMs, state, journalIncomplete };
}

function check(condition, message) {
  if (!condition) throw new Error(message);
}

function node(args, cwd, label) {
  const result = spawnSync(process.execPath, args, { cwd, encoding: 'utf8', timeout: 30_000, maxBuffer: 2_000_000 });
  check(result.status === 0, `${label} failed (${result.error?.code ?? result.signal ?? result.status})`);
  return result.stdout;
}

async function configure(root, timed) {
  const source = JSON.parse(await readFile(join(broker, 'wrangler.jsonc'), 'utf8'));
  const migrations = join(root, 'broker-migrations');
  await mkdir(migrations);
  const names = (await readdir(join(broker, 'migrations'))).filter((name) => name.endsWith('.sql') && !/^002[57]_/.test(name));
  await Promise.all(names.map((name) => copyFile(join(broker, 'migrations', name), join(migrations, name))));
  const main = join(broker, timed ? 'scripts/issue-499-broker-d1-timing.mjs' : source.main);
  const config = {
    name: 'issue-499-broker-local', main, compatibility_date: source.compatibility_date,
    placement: { mode: 'off' }, workers_dev: false, preview_urls: false, send_metrics: false,
    vars: { NETWORK_IDENTITY_HMAC_KEY_VERSION: '1' },
    d1_databases: [
      { binding: 'BROKER_DB', database_name: 'issue-499-local-broker', database_id: randomUUID(), migrations_dir: migrations, remote: false },
      { binding: 'AGENT_USAGE_DB', database_name: 'issue-499-local-relay', database_id: randomUUID(),
        migrations_dir: join(relay, 'migrations'), remote: false },
    ],
  };
  await writeFile(join(root, 'local-off.json'), JSON.stringify(config, null, 2));
  return { main, brokerMigrations: names.length };
}

function device(index) {
  const keys = generateKeyPairSync('ed25519');
  return { id: `019a3544-3380-7000-8000-${index.toString(16).padStart(12, '0')}`,
    publicKey: keys.publicKey.export({ format: 'jwk' }).x, privateKey: keys.privateKey };
}

function proof({ id, publicKey, privateKey }, corrupt = false) {
  const signedAt = new Date().toISOString();
  const nonce = randomBytes(32).toString('base64url');
  const message = ['puripuly-heart:puri-token:v1', 'POST', endpoint, 'puripuly-agent-relay', id, publicKey,
    appVersion, signedAt, nonce].join('\n');
  const signature = sign(null, Buffer.from(message), privateKey);
  if (corrupt) signature[0] ^= 1;
  return JSON.stringify({ installation_id: id, device_public_key: publicKey, app_version: appVersion,
    signed_at: signedAt, nonce, signature_alg: 'ed25519', signature: signature.toString('base64url') });
}

function validToken(payload, id, secrets, receivedAt) {
  try {
    const parts = payload.access_token.split('.');
    const [header, claims, signature] = parts;
    const decode = (value) => JSON.parse(Buffer.from(value, 'base64url').toString('utf8'));
    const { alg, typ } = decode(header);
    const { iss, aud, sub, iat, exp } = decode(claims);
    const hmac = (key, text) => createHmac('sha256', key).update(text);
    const actual = Buffer.from(signature, 'base64url');
    const want = hmac(secrets.PURI_TOKEN_SECRET, `${header}.${claims}`).digest();
    return parts.length === 3 && alg === 'HS256' && typ === 'JWT' && iss === 'puripuly-heart-broker' && aud === 'puripuly-agent-relay' &&
      sub === hmac(secrets.PURI_IDENTITY_HMAC_SECRET, id).digest('hex') && actual.length === want.length && timingSafeEqual(actual, want) &&
      Number.isInteger(iat) && Math.abs(iat * 1000 - receivedAt) < 60_000 && exp - iat === 3600 && payload.expires_in === 3600 &&
      payload.expires_at === new Date(exp * 1000).toISOString() && payload.token_type === 'Bearer';
  } catch {
    return false;
  }
}

async function workload(port, cohort, secrets, timed, record) {
  let attempts = 0;
  const origin = `http://127.0.0.1:${port}`;
  const submit = async (kind, identity, body = proof(identity)) => {
    const attempt = ++attempts;
    record({ event: 'start', attempt, kind });
    const signal = AbortSignal.timeout(4000);
    const start = performance.now();
    let status, subcode, d1, matched, elapsedMs;
    let outcome = 'transport-error';
    try {
      const response = await fetch(origin + endpoint, { method: 'POST', redirect: 'error', signal,
        headers: { 'content-type': 'application/json', 'cf-connecting-ip': '192.0.2.10' }, body });
      status = response.status;
      const bytes = await response.text();
      elapsedMs = performance.now() - start;
      outcome = 'completed';
      const payload = JSON.parse(bytes);
      const observation = response.headers.get('x-issue-499-d1');
      if (observation) d1 = numbers(JSON.parse(observation), d1Fields);
      const expected = expectedErrors.get(kind);
      subcode = [...expectedErrors.values()].some(([, value]) => value === payload.error?.subcode) ? payload.error.subcode : undefined;
      const timingMatched = timed ? d1 && Object.values(d1).every(Number.isFinite) : observation === null;
      matched = Boolean(timingMatched) && response.headers.get('cache-control') === 'no-store' && (expected
        ? status === expected[0] && subcode === expected[1] && !Object.hasOwn(payload, 'access_token')
          && (status !== 429 || Number.isFinite(payload.error.retry_after_ms) && payload.error.retry_after_ms > 0)
        : status === 200 && validToken(payload, identity.id, secrets, Date.now()));
    } catch {
      if (outcome !== 'completed' && signal.aborted) outcome = 'timed-out';
      matched = false;
    }
    record({ event: 'finish', sample: { attempt, kind, status: status ?? null, outcome, subcode, matched,
      elapsedMs: +(elapsedMs ?? performance.now() - start).toFixed(3), d1 } });
    return body;
  };
  const started = performance.now();
  for (let i = 0; i < 3; i++) await submit('warmup', cohort[13]);
  let issuedProof;
  for (let pass = 0; pass < 5; pass++) {
    for (const identity of cohort.slice(0, 12)) {
      const body = await submit('renewal', identity);
      if (identity === cohort[0]) issuedProof = body;
    }
    if (pass < 4) await submit('enrollment', device(100 + pass));
  }
  await submit('bad-signature', cohort[0], proof(cohort[0], true));
  await submit('replay', cohort[0], issuedProof);
  await submit('binding-mismatch', { ...device(200), id: cohort[0].id });
  if (900_000 - Date.now() % 900_000 >= 40_000) {
    for (let i = 0; i < 8; i++) await submit('quota-setup', cohort[12]);
    await submit('installation-limit', cohort[12]);
  }
  await submit('recovery', cohort[0]);
  return +(performance.now() - started).toFixed(3);
}

async function runLocal(root, timed, journal) {
  const record = (entry) => appendFileSync(journal, `${JSON.stringify(entry)}\n`, { mode: 0o600 });
  const secrets = Object.fromEntries(['PURI_TOKEN_SECRET', 'PURI_IDENTITY_HMAC_SECRET', 'NETWORK_IDENTITY_HMAC_SECRET']
    .map((name) => [name, randomBytes(32).toString('hex')]));
  const { version } = require('wrangler/package.json');
  const updateCache = join(root, 'update-check');
  await mkdir(updateCache);
  await writeFile(join(updateCache, 'wrangler-latest.json'), JSON.stringify({ latest: version, lastUpdate: Date.now() }));
  const { main, brokerMigrations } = await configure(root, timed);
  const cohort = Array.from({ length: 14 }, (_, i) => device(i + 1));
  const seed = join(root, 'seed.sql');
  await writeFile(seed, 'INSERT INTO installations (installation_id, device_public_key, app_version, puri_bound_at) VALUES\n' +
    cohort.map(({ id, publicKey }) => `('${id}','${publicKey}','${appVersion}','2026-04-08T00:00:00.000Z')`).join(',\n') + ';\n');
  await writeFile(join(root, '.dev.vars'), Object.entries(secrets).map(([name, value]) => `${name}=${value}`).join('\n') + '\n', { mode: 0o600 });
  const config = join(root, 'local-off.json');
  const state = join(root, 'd1');
  const d1 = (command, binding, ...options) => node([wrangler, 'd1', ...command, binding, '--config', config,
    '--local', '--persist-to', state, ...options], root, `${binding} local ${command.join(' ')}`);
  d1(['migrations', 'apply'], 'BROKER_DB');
  d1(['migrations', 'apply'], 'AGENT_USAGE_DB');
  d1(['execute'], 'BROKER_DB', '--file', seed, '--json');
  const { unstable_dev } = require('wrangler');
  const worker = await unstable_dev(main, { config, ip: '127.0.0.1', port: 0,
    inspectorPort: 0, local: true, persistTo: state, logLevel: 'error',
    experimental: { disableExperimentalWarning: true, disableDevRegistry: true, forceLocal: true, enableContainers: false } });
  try {
    check(worker.address === '127.0.0.1' && worker.port > 0, 'Worker did not bind a private loopback listener');
    record({ event: 'workload', brokerMigrations });
    const elapsedMs = await workload(worker.port, cohort, secrets, timed, record);
    record({ event: 'shutdown', elapsedMs });
  } finally {
    await worker.stop();
  }
  record({ event: 'snapshot' });
  const snapshot = 'SELECT (SELECT count(*) FROM installations) AS installations, ' +
    '(SELECT count(*) FROM installations WHERE puri_bound_at IS NOT NULL) AS bound, ' +
    '(SELECT count(*) FROM puri_enrollment_admissions) AS enrollments, (SELECT count(*) FROM puri_proof_nonces) AS nonces, ' +
    "(SELECT sum(request_count) FROM puri_token_rate_limits WHERE policy='ip_flood') AS flood, " +
    "(SELECT sum(request_count) FROM puri_token_rate_limits WHERE policy='installation') AS installation";
  const row = JSON.parse(d1(['execute'], 'BROKER_DB', '--command', snapshot, '--json'))[0]?.results?.[0];
  check(row, 'D1 snapshot returned no row');
  record({ event: 'done', state: numbers(row, Object.keys(expectedState)) });
}

const args = process.argv.slice(2);
const journalName = 'attempts.jsonl';
if (args[0] === '--isolated-child' && process.env.ISSUE_499_LOCAL_ROOT === process.cwd()) {
  await runLocal(process.cwd(), args[1] === '--d1-timing', join(process.cwd(), journalName));
} else {
  check(args.length === 0 || args.length === 1 && args[0] === '--d1-timing', 'Usage: node issue-499-broker-probe.mjs [--d1-timing]');
  const root = await mkdtemp(join(tmpdir(), 'issue-499-broker-'));
  const env = Object.fromEntries(Object.entries(process.env).filter(([name]) => /^(path|pathext|systemroot|windir|comspec)$/i.test(name)));
  Object.assign(env, { HOME: root, USERPROFILE: root, APPDATA: root, LOCALAPPDATA: root, XDG_CONFIG_HOME: root, XDG_CACHE_HOME: root,
    TMP: root, TEMP: root, TMPDIR: root, CI: 'true', WRANGLER_SEND_METRICS: 'false', WRANGLER_LOG_PATH: join(root, 'wrangler.log'),
    CLOUDFLARE_LOAD_DEV_VARS_FROM_DOT_ENV: 'false', CLOUDFLARE_INCLUDE_PROCESS_ENV: 'false', ISSUE_499_LOCAL_ROOT: root });
  const started = performance.now();
  let report;
  let runnerFailure = null;
  let cleanupFailure;
  try {
    const child = spawnSync(process.execPath, [fileURLToPath(import.meta.url), '--isolated-child', ...args], {
      cwd: root, env, encoding: 'utf8', timeout: 120_000, maxBuffer: 2_000_000,
    });
    if (child.status !== 0) runnerFailure = child.error?.code ?? child.signal ?? `exit-${child.status}`;
    report = await readJournal(join(root, journalName));
  } finally {
    try { await rm(root, { recursive: true, force: true, maxRetries: 5, retryDelay: 100 }); }
    catch (error) { cleanupFailure = error.code ?? 'unknown'; }
  }
  if (report.stage !== 'done') runnerFailure ??= 'missing-final-snapshot';
  const workerStopped = report.stage === 'snapshot' || report.stage === 'done';
  const cleanup = cleanupFailure ? `private local state removal failed (${cleanupFailure})`
    : workerStopped ? 'stopped and removed private local state' : 'removed private local state; Worker shutdown unconfirmed';
  const counts = { planned: 80, attempted: report.samples.length, completed: 0, timedOut: 0, transportErrors: 0, interrupted: 0 };
  for (const sample of report.samples) {
    if (sample.outcome === 'completed') counts.completed++;
    else if (sample.outcome === 'timed-out') counts.timedOut++;
    else if (sample.outcome === 'transport-error') counts.transportErrors++;
    else counts.interrupted++;
  }
  const passed = !runnerFailure && !report.journalIncomplete && !cleanupFailure &&
    counts.attempted === counts.planned && report.samples.every(({ matched }) => matched) &&
    Object.entries(expectedState).every(([name, value]) => report.state[name] === value);
  console.log(JSON.stringify({ timed: args.length === 1, ...report, counts, runnerFailure, passed, cleanup,
    totalMs: +(performance.now() - started).toFixed(3) }));
  if (!passed) process.exitCode = 1;
}
