import { createHash, createHmac } from 'node:crypto';
import { readdir, readFile, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { dirname, extname, relative, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { isDeepStrictEqual } from 'node:util';

const require = createRequire(import.meta.url);
const { Miniflare, Response, convertV4MiniflareOptions } = createRequire(require.resolve('wrangler/package.json'))('miniflare');
const migrationsDefault = fileURLToPath(new URL('../migrations/', import.meta.url));
const tokenSecret = 'puri-context-export-synthetic-token-secret-only';
const installation = createHash('sha256').update('puri-context-export-synthetic-installation').digest('hex');
const dummyKey = 'synthetic-context-export-no-provider-access';
const staticFields = new Set(['model', 'instructions', 'tools', 'questions', 'reasoning',
  'prompt_cache_options', 'store', 'stream', 'parallel_tool_calls', 'max_output_tokens', 'include']);
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');

function signedToken() {
  const now = Math.floor(Date.now() / 1000);
  const encoded = [{ alg: 'HS256', typ: 'JWT' }, { iss: 'puripuly-heart-broker', aud: 'puripuly-agent-relay',
    sub: installation, iat: now, exp: now + 3600 }]
    .map((value) => Buffer.from(JSON.stringify(value)).toString('base64url')).join('.');
  return `${encoded}.${createHmac('sha256', tokenSecret).update(encoded).digest('base64url')}`;
}

async function emittedModules(entry) {
  const root = dirname(entry);
  const modules = [];
  const hashes = {};
  async function visit(directory) {
    for (const item of (await readdir(directory, { withFileTypes: true })).sort((a, b) => a.name.localeCompare(b.name))) {
      const path = resolve(directory, item.name);
      if (item.isSymbolicLink()) throw new Error('Emitted module directories must not contain symbolic links.');
      if (item.isDirectory()) {
        await visit(path);
        continue;
      }
      const type = ({ '.js': 'ESModule', '.mjs': 'ESModule', '.md': 'Text', '.txt': 'Text', '.wasm': 'CompiledWasm' })[extname(path)];
      if (!type || item.name === 'README.md') continue;
      const bytes = await readFile(path);
      const contents = type === 'CompiledWasm' ? bytes : new TextDecoder('utf-8', { fatal: true, ignoreBOM: true }).decode(bytes);
      hashes[relative(root, path).replaceAll('\\', '/')] = sha256(bytes);
      modules.push({ type, path, contents });
    }
  }
  await visit(root);
  const index = modules.findIndex((module) => module.path === entry);
  if (index < 0 || modules[index].type !== 'ESModule') throw new Error('--emitted must identify an existing Wrangler-emitted JavaScript entry.');
  modules.unshift(...modules.splice(index, 1));
  return { root, modules, hashes };
}

export function extractContext(upstream, input) {
  if (!upstream || typeof upstream !== 'object' || Array.isArray(upstream)
    || !Array.isArray(upstream.input) || upstream.input.length < input.length
    || !isDeepStrictEqual(upstream.input.slice(upstream.input.length - input.length), input)) {
    throw new Error('The emitted Worker changed client input instead of prepending a static context.');
  }
  const context = { input_prefix: upstream.input.slice(0, upstream.input.length - input.length) };
  for (const [key, value] of Object.entries(upstream)) {
    if (['input', 'safety_identifier', 'tool_choice'].includes(key)) continue;
    if (!staticFields.has(key)) throw new Error(`Unrecognized outbound static field ${key}; extend the manifest explicitly.`);
    context[key] = value;
  }
  return context;
}

function samples() {
  const user = (content) => ({ role: 'user', content });
  const configuration = (effort) => ({ type: 'configuration_update', reasoning: { effort } });
  return [
    { id: 'classify', input: [user('Synthetic context export classification.')] },
    { id: 'classify', input: [user('Different synthetic classification input.')] },
    { id: 'work', route: 'work', effort: 'low', allow_tools: true,
      input: [configuration('low'), user('Synthetic app help.')] },
    { id: 'work', route: 'work', effort: 'high', allow_tools: false,
      input: [configuration('low'), user('Synthetic prior app help.'), configuration('high'), user('Synthetic current app help.')] },
    { id: 'work', route: 'unclear', effort: 'xhigh', allow_tools: true,
      input: [configuration('xhigh'), user('Synthetic unclear app request.')] },
    ...['social', 'out_of_scope'].flatMap((route) => [true, false].map((allow_tools) => ({
      id: route, route, effort: 'low', allow_tools, input: [user(`Synthetic ${route} context export.`)],
    }))),
  ];
}

export async function exportRelayContext({ emitted, relayBuild, migrations = migrationsDefault,
  compatibilityDate, compatibilityFlags = [], protocol = 6 }) {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(compatibilityDate ?? '')
    || !Array.isArray(compatibilityFlags) || compatibilityFlags.some((flag) => typeof flag !== 'string')) {
    throw new Error('Supply the measured Worker compatibility date and flags explicitly.');
  }
  const entry = resolve(emitted);
  const { root, modules, hashes } = await emittedModules(entry);
  const build = hashes[relative(root, entry).replaceAll('\\', '/')];
  if (relayBuild !== undefined && relayBuild !== build) throw new Error('--relay-build does not match SHA256 of the emitted entry module.');
  const migrationHashes = {};
  const migrationSql = [];
  for (const item of (await readdir(migrations, { withFileTypes: true })).sort((a, b) => a.name.localeCompare(b.name))) {
    if (!item.isFile() || !item.name.endsWith('.sql')) continue;
    const bytes = await readFile(resolve(migrations, item.name));
    migrationHashes[item.name] = sha256(bytes);
    migrationSql.push(bytes.toString('utf8'));
  }
  if (migrationSql.length === 0) throw new Error('The current relay D1 migrations are required.');
  let pending;
  const denied = [];
  const runtime = new Miniflare(convertV4MiniflareOptions({
    host: '127.0.0.1', port: 0, inspectorPort: 0, cf: false,
    workers: [{ name: 'puri-context-export', modules, modulesRoot: root,
      compatibilityDate, compatibilityFlags,
      bindings: { OPENAI_API_KEY: dummyKey, PURI_TOKEN_SECRET: tokenSecret, AGENT_SERVICE_ENABLED: 'true' },
      d1Databases: ['AGENT_USAGE_DB'],
      async outboundService(request) {
        const expected = `https://api.openai.com/v1/${pending?.sample.id === 'classify' ? 'decisions' : 'responses'}`;
        if (!pending || request.url !== expected || request.method !== 'POST'
          || request.headers.get('authorization') !== `Bearer ${dummyKey}`) {
          denied.push('unexpected outbound request');
          throw new Error('All external network access is disabled in the context exporter.');
        }
        pending.bodies.push(await request.json());
        const usage = { input_tokens: 0, output_tokens: 0 };
        if (pending.sample.id === 'classify') return Response.json({ answers: [], usage });
        const response = { id: 'controlled-context-export', object: 'response', status: 'completed', output: [], usage };
        return new Response(`data: ${JSON.stringify({ type: 'response.completed', response })}\n\n`, {
          headers: { 'content-type': 'text/event-stream' },
        });
      },
    }],
  }));
  const contexts = {};
  const captures = [];
  try {
    await runtime.ready;
    const database = await runtime.getD1Database('AGENT_USAGE_DB');
    for (const sql of migrationSql) {
      const statements = sql.split(';').map((statement) => statement.trim()).filter(Boolean);
      await database.batch(statements.map((statement) => database.prepare(statement)));
    }
    for (const [number, sample] of samples().entries()) {
      pending = { sample, bodies: [] };
      const classification = sample.id === 'classify';
      const request = { agent_protocol: protocol, turn_id: (number + 1).toString(16).padStart(32, '0'), input: sample.input,
        ...(classification ? {} : { route: sample.route, effort: sample.effort, allow_tools: sample.allow_tools }) };
      const response = await runtime.dispatchFetch(`http://context-export.invalid/v1/${classification ? 'classify' : 'respond'}`, {
        method: 'POST', headers: { authorization: `Bearer ${signedToken()}`, 'content-type': 'application/json' },
        body: JSON.stringify(request),
      });
      const body = await response.text();
      if (!response.ok || (!classification && !body.includes('"type":"completed"')) || pending.bodies.length !== 1 || denied.length) {
        throw new Error(`Controlled ${sample.route ?? sample.id} capture failed (${response.status}); no manifest was exported.`);
      }
      const upstream = pending.bodies[0];
      if (upstream.safety_identifier !== installation) throw new Error('The emitted Worker did not preserve the synthetic identity boundary.');
      if (!classification && upstream.tool_choice !== (sample.allow_tools && sample.id === 'work' ? 'auto' : 'none')) {
        throw new Error('The emitted Worker tool choice does not follow the per-call route/allow_tools boundary.');
      }
      const context = extractContext(upstream, sample.input);
      if (contexts[sample.id] && !isDeepStrictEqual(contexts[sample.id], context)) {
        throw new Error(`The emitted ${sample.id} context is not static across valid client requests.`);
      }
      contexts[sample.id] = context;
      captures.push({ upstream_context_id: sample.id, route: sample.route ?? 'classify',
        ...(classification ? {} : { allow_tools: sample.allow_tools, effort: sample.effort }) });
      pending = undefined;
    }
  } finally {
    await runtime.dispose();
  }
  return {
    format: 'puri-agent-relay-context-1', agent_protocol: protocol, relay_build: build, bundle_hashes: hashes, contexts,
    export_provenance: {
      mode: 'controlled_no_network', diagnostic_only: true, external_model_calls: 0,
      source: 'observed_emitted_worker_outbound_requests', entry_module: relative(root, entry).replaceAll('\\', '/'),
      compatibility_date: compatibilityDate, compatibility_flags: compatibilityFlags,
      migrations: migrationHashes, captures,
      tool_choice: 'Per-call route/allow_tools, deliberately excluded from static contexts.',
    },
  };
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  try {
    const { values } = parseArgs({ options: {
      emitted: { type: 'string' }, output: { type: 'string' }, 'relay-build': { type: 'string' },
      migrations: { type: 'string' }, 'compatibility-date': { type: 'string' },
      'compatibility-flag': { type: 'string', multiple: true },
      protocol: { type: 'string', default: '6' },
    } });
    if (!values.emitted || !values.output) throw new Error('Use --emitted <Wrangler index.js> --output <new manifest.json> --compatibility-date <date> [--compatibility-flag <flag>] [--relay-build <expected entry SHA256>] [--migrations <directory>].');
    const manifest = await exportRelayContext({ emitted: values.emitted, relayBuild: values['relay-build'],
      migrations: values.migrations, compatibilityDate: values['compatibility-date'], compatibilityFlags: values['compatibility-flag'],
      protocol: Number(values.protocol) });
    await writeFile(values.output, `${JSON.stringify(manifest, null, 2)}\n`, { flag: 'wx' });
    console.log(JSON.stringify({ status: 'controlled_context_exported', diagnostic_only: true,
      external_model_calls: 0, relay_build: manifest.relay_build, output: values.output }));
  } catch (error) {
    console.error(error instanceof Error ? error.message : 'Controlled context export failed.');
    process.exitCode = 1;
  }
}
