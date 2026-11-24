import worker from '../src/index.ts';

export default {
  async fetch(request, env, ctx) {
    if (request.method !== 'POST' || new URL(request.url).pathname !== '/v1/puri/token') {
      return new Response(null, { status: 404 });
    }
    const raw = env.BROKER_DB;
    const observation = { calls: 0, waitMs: 0, statements: 0, meteredStatements: 0, sqlMs: 0 };
    const originals = new WeakMap();
    const record = async (execute, statements, resultHasMetadata = true) => {
      observation.calls++;
      observation.statements += statements;
      const started = performance.now();
      let result;
      try { result = await execute(); }
      finally { observation.waitMs += performance.now() - started; }
      if (resultHasMetadata) {
        for (const { meta } of Array.isArray(result) ? result : [result]) {
          const sqlMs = meta?.timings?.sql_duration_ms ?? meta?.duration;
          if (typeof sqlMs === 'number') {
            observation.meteredStatements++;
            observation.sqlMs += sqlMs;
          }
        }
      }
      return result;
    };
    const wrap = (rawStatement) => {
      const wrapped = {
        bind: (...values) => wrap(rawStatement.bind(...values)),
        all: () => record(() => rawStatement.all(), 1),
        first: (...args) => record(() => rawStatement.first(...args), 1, false),
        run: () => record(() => rawStatement.run(), 1),
      };
      originals.set(wrapped, rawStatement);
      return wrapped;
    };
    const response = await worker.fetch(request, {
      ...env,
      BROKER_DB: {
        prepare: (sql) => wrap(raw.prepare(sql)),
        batch: (statements) => record(() => raw.batch(statements.map((statement) => originals.get(statement))), statements.length),
      },
    }, ctx);
    response.headers.set('x-issue-499-d1', JSON.stringify(observation));
    return response;
  },
};
