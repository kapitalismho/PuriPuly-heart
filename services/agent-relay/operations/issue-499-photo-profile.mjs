import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { createHash, createHmac } from 'node:crypto';
import { readFile, readdir, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { setTimeout as wait } from 'node:timers/promises';
import { fileURLToPath } from 'node:url';
import { build } from '../../../node_modules/.pnpm/node_modules/esbuild/lib/main.js';
import { TraceMap, decodedMappings } from '../../../node_modules/.pnpm/node_modules/@jridgewell/trace-mapping/dist/trace-mapping.mjs';
import { convertV4MiniflareOptions, Miniflare, Response as LocalResponse } from '../../../node_modules/.pnpm/node_modules/miniflare/dist/src/index.js';

const relayRoot = fileURLToPath(new URL('../', import.meta.url));
const root = path.resolve(relayRoot, '../..');
const output = path.resolve(process.argv[2] ?? '');
const batches = Number(process.argv[3] ?? 3);
const count = Number(process.argv[4] ?? 24);
assert(process.argv[2] && Number.isInteger(batches) && batches > 0 && Number.isInteger(count) && count > 0 && count <= 48,
  'Usage: node issue-499-photo-profile.mjs <fixture directory> [batches=3] [requests-per-cohort=24]');

const getJSON = async (filename) => JSON.parse(await readFile(filename, 'utf8'));
const sha256 = (value) => createHash('sha256').update(value).digest('hex');
const now = () => Number(process.hrtime.bigint() / 1000n);
const save = (name, value) => writeFile(path.join(output, name), JSON.stringify(value, null, 2) + '\n');
const deltaType = 'response.output_text.delta';
const answerId = 'synthetic-answer';
const model = 'gpt-6-luna';
const localKey = 'probe-only-no-provider-credential';
const tokenKey = 'issue-499-local-token-key-not-a-credential-32';
const intervalUs = 101;
const tailDelayMs = 30;
const cliTimeoutMs = 12000;
const config = await getJSON(path.join(relayRoot, 'wrangler.jsonc'));
const contract = await getJSON(path.join(relayRoot, 'src/photo-contract.json'));
const fixture = await getJSON(path.join(output, 'app-fixture.json'));
const prepared = await getJSON(path.join(output, 'app-request.json'));
assert.equal(prepared.photo_protocol, 2);
assert.equal(prepared.target_language, 'ko');
assert.deepEqual(prepared.image_size, [1280, 720]);
assert.equal(sha256(Buffer.from(prepared.image_base64, 'base64')), fixture.jpeg_sha256);
const boundary = '/9j/' + 'A'.repeat(3_999_996);
const scenarios = {
  prepared: { base64: prepared.image_base64, blocks: 12, fragment: 48 },
  boundary: { base64: boundary, blocks: 12, fragment: 48 },
  dense_stream: { base64: prepared.image_base64, blocks: 72, fragment: 12 },
};

function sign(subject) {
  const iat = Math.floor(Date.now() / 1000);
  const token = [{ alg: 'HS256', typ: 'JWT' }, { iss: 'puripuly-heart-broker', aud: 'puripuly-agent-relay', sub: subject, iat, exp: iat + 3600 }]
    .map((value) => Buffer.from(JSON.stringify(value)).toString('base64url')).join('.');
  return `${token}.${createHmac('sha256', tokenKey).update(token).digest('base64url')}`;
}

function frames(scenario, incomplete) {
  const blocks = Array.from({ length: scenario.blocks }, (_, index) => {
    const left = 50 + index % 6 * 190;
    const top = 50 + Math.floor(index / 6) * 50;
    return { bbox: [left, top, left + 120, top + 34], source: `Synthetic label ${index + 1}`, translation: `합성 표지판 ${index + 1}` };
  });
  const first = '{"blocks":[' + JSON.stringify(blocks[0]);
  const rest = blocks.slice(1).map((block) => ',' + JSON.stringify(block)).join('') + ']}';
  const partial = ',{"bbox":[240,50,360,84],"source":"unfinished';
  const selectedTail = incomplete ? partial : rest;
  const message = (id, phase, text) => ({ id, type: 'message', role: 'assistant', phase, content: [{ type: 'output_text', text }] });
  const item = (type, output_index, item) => ({ type: `response.output_item.${type}`, output_index, item });
  const firstFrames = [
    item('added', 0, message('synthetic-commentary', 'commentary', '')),
    { type: deltaType, item_id: 'synthetic-commentary', output_index: 0, content_index: 0, delta: 'Ignore synthetic commentary' },
    item('done', 0, message('synthetic-commentary', 'commentary', 'Ignore synthetic commentary')),
    item('added', 1, message(answerId, 'final_answer', '')),
  ];
  const toDeltas = (text) => text.match(new RegExp(`.{1,${scenario.fragment}}`, 'gu'))
    .map((delta) => ({ type: deltaType, delta, item_id: answerId, output_index: 1, content_index: 0 }));
  firstFrames.push(...toDeltas(first));
  const status = incomplete ? 'incomplete' : 'completed';
  const terminal = {
    object: 'response', status, model,
    ...(incomplete ? { incomplete_details: { reason: 'max_output_tokens' } } : {}),
    output: [message('synthetic-commentary', 'commentary', 'Ignore synthetic commentary'), message(answerId, 'final_answer', first + selectedTail)],
    instructions: contract.instructions.replaceAll('${targetName}', 'Korean'), tools: [],
    usage: { input_tokens: 1502, input_tokens_details: { cached_tokens: 0, cache_write_tokens: 1499 }, output_tokens: 2556 },
  };
  const tailFrames = [...toDeltas(selectedTail), item('done', 1, terminal.output[1]), { type: `response.${status}`, response: terminal }];
  const encode = (items) => new TextEncoder().encode(items.map((event) => `data: ${JSON.stringify(event)}\n\n`).join(''));
  return { first, rest: selectedTail, blocks, terminal, firstFrames, tailFrames, firstWire: encode(firstFrames), tailWire: encode(tailFrames) };
}

const entry = path.join(output, 'issue-499-bundle.mjs');
await build({
  absWorkingDir: relayRoot, entryPoints: [config.main], bundle: true, format: 'esm', target: 'es2022', outfile: entry, sourcemap: 'external',
  plugins: [{ name: 'wrangler-raw-text-imports', setup(builder) {
    builder.onResolve({ filter: /\.md\?raw$/ }, ({ path: requested, resolveDir }) => ({
      path: path.resolve(resolveDir, requested.slice(0, -4)), namespace: 'relay-raw-text',
    }));
    builder.onLoad({ filter: /.*/, namespace: 'relay-raw-text' }, async ({ path: filename }) => ({ contents: await readFile(filename, 'utf8'), loader: 'text' }));
  } }],
});
const sourceMap = new TraceMap(await getJSON(`${entry}.map`));
const mappings = decodedMappings(sourceMap);
function originals(generatedLine) {
  return [...new Set((mappings[generatedLine - 1] ?? []).filter((piece) => piece.length >= 4)
    .map(([, file, line]) => `${path.basename(sourceMap.sources[file])}:${line + 1}`))];
}

let active;
const mf = new Miniflare(convertV4MiniflareOptions({
  name: 'issue-499-photo-profile', modules: true, scriptPath: entry, modulesRoot: output,
  compatibilityDate: config.compatibility_date, compatibilityFlags: config.compatibility_flags,
  host: '127.0.0.1', port: 0, inspectorHost: '127.0.0.1', inspectorPort: 0,
  cf: false, telemetry: { enabled: false }, logRequests: false, d1Persist: false,
  bindings: { ...config.vars, PHOTO_SERVICE_ENABLED: 'true', PHOTO_OPENAI_API_KEY: localKey, PURI_TOKEN_SECRET: tokenKey },
  d1Databases: { AGENT_USAGE_DB: 'issue-499-local-volatile-db' },
  async outboundService(request) {
    assert(active, 'unexpected egress outside an admitted probe');
    const job = active;
    const record = job.record;
    record.calls++;
    record.upstreamStart = now();
    try {
      assert.equal(record.calls, 1);
      assert.equal(request.url, 'https://api.openai.com/v1/responses');
      assert.equal(request.method, 'POST');
      assert.equal(request.headers.get('authorization'), `Bearer ${localKey}`);
      const wire = await request.text();
      const body = JSON.parse(wire);
      record.modelRequestBytes = Buffer.byteLength(wire);
      assert.deepEqual(body, {
        model, store: false, stream: true, reasoning: { effort: 'none' }, safety_identifier: job.subject,
        instructions: contract.instructions.replaceAll('${targetName}', 'Korean'),
        input: [{ type: 'message', role: 'user', content: [
          { type: 'input_image', image_url: `data:image/jpeg;base64,${job.base64}` },
          { type: 'input_text', text: 'Target language: Korean.\nThe image is 1280x720 pixels.' },
        ] }],
        text: { format: { type: 'json_schema', name: 'blocks', schema: contract.schema, strict: true } },
        max_output_tokens: 8192,
      });
    } catch (error) {
      job.error = error.message.slice(0, 180);
      return new LocalResponse(null, { status: 500 });
    }
    const { firstWire, tailWire } = job.frames;
    const stream = new ReadableStream({
      start(controller) {
        record.firstSend = now();
        controller.enqueue(firstWire);
        job.tail.then((send) => {
          if (send) {
            record.tailSend = now();
            controller.enqueue(tailWire);
          }
          controller.close();
        }).catch(() => {});
      },
    });
    return new LocalResponse(stream, { headers: { 'content-type': 'text/event-stream' } });
  },
}));

async function connectProfiler() {
  const inspector = await mf.getInspectorURL();
  const listURL = new URL('/json/list', inspector.href.replace(/^ws:/, 'http:'));
  const list = await (await fetch(listURL, { signal: AbortSignal.timeout(cliTimeoutMs) })).json();
  const target = list.find((item) => /issue-499-photo-profile/.test(`${item.title} ${item.id}`));
  assert(target?.webSocketDebuggerUrl, `No Relay isolate at inspector; titles=${list.map((item) => item.title).join(',')}`);
  const socket = new WebSocket(target.webSocketDebuggerUrl);
  const pending = new Map();
  socket.addEventListener('message', ({ data }) => {
    const message = JSON.parse(data);
    if (message.id === undefined) return;
    const request = pending.get(message.id);
    if (!request) return;
    pending.delete(message.id);
    clearTimeout(request.timer);
    if (message.error) request.reject(new Error(message.error.message));
    else request.resolve(message.result);
  });
  await new Promise((resolve, reject) => {
    socket.addEventListener('open', resolve, { once: true });
    socket.addEventListener('error', () => reject(new Error('Could not connect to CDP')), { once: true });
  });
  let id = 0;
  const call = (method, params = {}) => new Promise((resolve, reject) => {
    const requestId = ++id;
    const timer = setTimeout(() => {
      pending.delete(requestId);
      reject(new Error(`CDP timeout: ${method}`));
    }, cliTimeoutMs);
    pending.set(requestId, { resolve, reject, timer });
    socket.send(JSON.stringify({ id: requestId, method, params }));
  });
  await call('Profiler.enable');
  await call('Profiler.setSamplingInterval', { interval: intervalUs });
  return { call, close: () => socket.close() };
}

function initiate(body, subject) {
  return fetch(new URL('/v1/photo', relayURL), {
    method: 'POST', body,
    headers: { authorization: `Bearer ${sign(subject)}`, 'content-type': 'application/json' },
    signal: AbortSignal.timeout(cliTimeoutMs),
  });
}

async function probe(name, index, subject, incomplete, expectedReplay = false) {
  const scenario = scenarios[name];
  const expected = frames(scenario, incomplete);
  const id = index.toString(16).padStart(32, '0');
  const body = JSON.stringify({ ...prepared, request_id: id, image_base64: scenario.base64 });
  const gate = Promise.withResolvers();
  const record = {
    id, incomplete, status: null, requestBytes: Buffer.byteLength(body), calls: 0,
    upstreamFirstBytes: expected.firstWire.length, upstreamTailBytes: expected.tailWire.length,
    upstreamFirstEvents: expected.firstFrames.length, upstreamTailEvents: expected.tailFrames.length,
  };
  const job = { record, frames: expected, subject, tail: gate.promise, base64: scenario.base64 };
  active = job;
  record.start = now();
  try {
    const response = await initiate(body, subject);
    record.headers = now();
    record.status = response.status;
    if (expectedReplay) {
      assert.equal(response.status, 409);
      assert.deepEqual(await response.json(), { error: { code: 'photo_request_already_admitted' } });
      assert.equal(record.calls, 0);
      record.outcome = 'duplicate_rejected';
      return record;
    }
    assert(!job.error, job.error);
    assert.equal(response.status, 200);
    assert.equal(record.calls, 1);
    assert.match(response.headers.get('content-type'), /^text\/event-stream/);
    const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
    let pending = '';
    let selected = '';
    let terminal;
    let deltas = 0;
    let downstreamBytes = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      downstreamBytes += Buffer.byteLength(value);
      pending += value;
      for (;;) {
        const end = pending.indexOf('\n\n');
        if (end === -1) break;
        const line = pending.slice(0, end);
        pending = pending.slice(end + 2);
        assert(line.startsWith('data: '));
        const event = JSON.parse(line.slice(6));
        if (event.type === 'text_delta') {
          assert.equal(terminal, undefined);
          record.firstDelta ??= now();
          deltas++;
          selected += event.text;
          assert((expected.first + expected.rest).startsWith(selected), 'unexpected selected output');
          if (!record.firstBlock && selected === expected.first) {
            assert.deepEqual(JSON.parse(selected + ']}').blocks, [expected.blocks[0]]);
            record.firstBlock = now();
            await wait(tailDelayMs);
            gate.resolve(true);
          }
        } else {
          assert.equal(event.type, 'completed', JSON.stringify(event));
          assert.equal(terminal, undefined);
          terminal = event.response;
          record.terminal = now();
        }
      }
    }
    assert.equal(pending, '');
    assert(record.firstDelta <= record.firstBlock && record.firstBlock < record.tailSend && record.tailSend < record.terminal);
    assert.equal(selected, expected.first + expected.rest);
    assert(!selected.includes('Ignore synthetic commentary'));
    const { instructions, tools, ...expectedTerminal } = expected.terminal;
    assert.deepEqual(terminal, expectedTerminal);
    if (incomplete) {
      assert.throws(() => JSON.parse(selected), SyntaxError);
      record.blocks = 1;
    } else {
      assert.deepEqual(JSON.parse(selected).blocks, expected.blocks);
      record.blocks = expected.blocks.length;
    }
    record.deltas = deltas;
    record.downstreamBytes = downstreamBytes;
    record.outcome = terminal.status;
  } catch (error) {
    record.error = job.error ?? error.message.slice(0, 240);
    record.outcome = 'probe_error';
  } finally {
    record.done = now();
    gate.resolve(false);
    active = undefined;
  }
  return record;
}

function samplerSummary(profile, alignment, records) {
  assert(profile.samples?.length > 0 && profile.samples.length === profile.timeDeltas.length);
  const byId = new Map(profile.nodes.map((node) => [node.id, node]));
  const parent = new Map();
  for (const node of profile.nodes) for (const child of node.children ?? []) parent.set(child, node.id);
  const scopes = {}, stacks = {}, sourceTicks = {};
  let at = alignment.start;
  let cursor = 0;
  for (const [index, sampled] of profile.samples.entries()) {
    at += profile.timeDeltas[index];
    while (cursor < records.length && at > records[cursor].done) cursor++;
    const r = records[cursor];
    const scope = !r || at < r.start ? 'outside_request'
      : at < r.upstreamStart ? 'request_through_upstream_entry'
      : at < r.firstSend ? 'controlled_upstream_read_and_validate'
      : at < r.firstBlock ? 'first_block_stream'
      : at < r.tailSend ? 'controlled_tail_gate'
      : 'later_stream_and_settlement';
    const names = [];
    for (let id = sampled; id; id = parent.get(id)) {
      const { functionName, lineNumber, url } = byId.get(id).callFrame;
      const source = url.endsWith('issue-499-bundle.mjs') ? originals(lineNumber + 1)[0] : '';
      const name = functionName || '(anonymous)';
      if (name !== '(root)') names.push(source ? `${name} ${source}` : name);
    }
    const leaf = names[0] ?? '(root)';
    const group = leaf === '(idle)' ? 'idle' : leaf === '(program)' ? 'program' : leaf === '(garbage collector)' ? 'gc' : 'other';
    scopes[scope] ??= { idle: 0, program: 0, gc: 0, other: 0 };
    scopes[scope][group]++;
    if (group === 'other') {
      const key = names.slice(0, 6).join(' <- ');
      (stacks[scope] ??= {})[key] ??= 0;
      stacks[scope][key]++;
    }
  }
  for (const node of profile.nodes) {
    if (!node.callFrame.url.endsWith('issue-499-bundle.mjs')) continue;
    for (const { line, ticks } of node.positionTicks ?? []) {
      const key = `${node.callFrame.functionName || '(anonymous)'} ${originals(line).join(',') || `generated:${line}`}`;
      sourceTicks[key] = (sourceTicks[key] ?? 0) + ticks;
    }
  }
  const descending = (object) => Object.fromEntries(Object.entries(object).sort(([, a], [, b]) => b - a));
  const sortedDeltas = [...profile.timeDeltas].sort((a, b) => a - b);
  return {
    totalSamples: profile.samples.length, sampleDeltaP50Us: sortedDeltas[Math.floor(sortedDeltas.length / 2)],
    samplesByPhase: scopes, stacksByPhase: Object.fromEntries(Object.entries(stacks).map(([scope, entries]) => [scope, descending(entries)])),
    sourceTicks: descending(sourceTicks),
  };
}

let relayURL;
let inspector;
const warmups = [];
const runs = [];
const exclusions = [];
let accounting;
try {
  relayURL = await mf.ready;
  assert.equal(relayURL.hostname, '127.0.0.1');
  const db = await mf.getD1Database('AGENT_USAGE_DB');
  const migrationDir = path.join(relayRoot, config.d1_databases[0].migrations_dir);
  const migrations = (await readdir(migrationDir)).filter((name) => name.endsWith('.sql')).sort();
  for (const migration of migrations) {
    const statements = (await readFile(path.join(migrationDir, migration), 'utf8')).split(';').map((sql) => sql.trim()).filter(Boolean);
    await db.batch(statements.map((sql) => db.prepare(sql)));
  }
  const subject = (seed) => sha256(`issue-499-only:${seed}`);
  const names = Object.keys(scenarios);
  for (const [i, name] of names.entries()) {
    for (let n = 1; n <= 4; n++) warmups.push(await probe(name, n, subject(`warm-${i}`), false));
  }
  await save('warmups.json', warmups);
  assert(warmups.every((r) => r.outcome === 'completed'), 'Warm-up failed; see warmups.json');
  inspector = await connectProfiler();
  for (let batch = 0; batch < batches; batch++) {
    for (let i = 0; i < names.length; i++) {
      const name = names[(batch + i) % names.length];
      const setup = subject(`${batch}-${name}`);
      const records = [];
      const before = now();
      await inspector.call('Profiler.start');
      const after = now();
      for (let n = 1; n <= count; n++) {
        const record = await probe(name, n, setup, n % 6 === 0);
        records.push(record);
        if (record.outcome === 'probe_error') break;
      }
      const { profile } = await inspector.call('Profiler.stop');
      const direct = profile.startTime >= before && profile.startTime <= after;
      const alignment = { start: direct ? profile.startTime : Math.round((before + after) / 2),
        rawStart: profile.startTime, before, after, mode: direct ? 'shared_monotonic_clock' : 'start_rpc_midpoint' };
      const runName = `${batch + 1}-${name}`;
      await save(`${runName}.cpuprofile`, profile);
      const run = { runName, name, records, alignment, ...samplerSummary(profile, alignment, records) };
      await save(`${runName}.json`, run);
      runs.push(run);
      console.log(JSON.stringify({ runName, outcomes: run.records.map((r) => r.outcome), phaseSamples: run.samplesByPhase, alignment }));
      assert.equal(records.length, count, `Interrupted cohort: ${runName}`);
      assert(records.every((r) => r.outcome === (r.incomplete ? 'incomplete' : 'completed')), `Unexpected outcome: ${runName}`);
    }
  }
  const replay = await probe('prepared', 1, subject('0-prepared'), false, true);
  exclusions.push(replay);
  assert.equal(replay.outcome, 'duplicate_rejected');
  for (const [expected, image_base64] of [[400, boundary.slice(0, -1) + '!'], [413, boundary + 'A'.repeat(196608)]]) {
    let status;
    try {
      const response = await initiate(JSON.stringify({ ...prepared, image_base64 }), subject(`reject-${expected}`));
      status = response.status;
      const body = await response.json();
      exclusions.push({ status, error: body.error?.code });
      assert.equal(status, expected);
      assert.deepEqual(body, { error: { code: 'invalid_photo_request' } });
    } catch (error) {
      if (expected !== 413 || error.cause?.code !== 'ECONNRESET') throw error;
      exclusions.push({ expected, status: status ?? null, transportError: 'ECONNRESET', errorBodyVerified: false });
    }
  }
  const { results: [row] } = await db.prepare(`SELECT COUNT(*) AS admissions, SUM(settled) AS settled,
    SUM(missing_usage) AS missing, SUM(output_tokens) AS outputTokens, SUM(requested_at IS NOT NULL) AS timestamped
    FROM photo_admissions`).all();
  const admissions = warmups.length + runs.reduce((sum, run) => sum + run.records.length, 0);
  assert.deepEqual(row, { admissions, settled: admissions, missing: 0, outputTokens: 2556 * admissions, timestamped: admissions });
  accounting = row;
  await save('receipt.json', {
    revision: execFileSync('git', ['rev-parse', 'HEAD'], { cwd: root, encoding: 'utf8' }).trim(),
    runtime: { node: process.version, os: `${os.platform()} ${os.release()} ${os.arch()}`, cpu: os.cpus()[0].model,
      bundleSha256: sha256(await readFile(entry)), compatibilityDate: config.compatibility_date, compatibilityFlags: config.compatibility_flags },
    workload: { model, instructionsSha256: sha256(contract.instructions.replaceAll('${targetName}', 'Korean')), schemaSha256: sha256(JSON.stringify(contract.schema)),
      protocol: 2, target: 'Korean', dimensions: prepared.image_size, maxOutputTokens: 8192, reasoning: 'none', tailDelayMs, intervalUs, batches, count, migrations },
    fixture, accounting, exclusions, runs: runs.map(({ runName }) => runName),
  });
  console.log(JSON.stringify({ accounting, exclusions }));
} finally {
  inspector?.close();
  await mf.dispose();
}
