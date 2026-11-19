import { readFileSync } from 'node:fs';

import { describe, expect, it } from 'vitest';

describe('broker hosting assumptions', () => {
  it('binds the worker to the broker entrypoint and native D1 config without unsupported location fields', () => {
    const config = JSON.parse(readFileSync(new URL('../wrangler.jsonc', import.meta.url), 'utf8'));

    expect(config.main).toBe('src/index.ts');
    expect(config.d1_databases).toEqual([
      { binding: 'BROKER_DB', database_name: 'puripuly-heart-broker', database_id: 'REQUIRED_AT_DEPLOY_TIME' },
      { binding: 'AGENT_USAGE_DB', database_name: 'puripuly-heart-agent-usage', database_id: 'AGENT_USAGE_REQUIRED_AT_DEPLOY_TIME', migrations_dir: '../agent-relay/migrations' },
    ]);
    expect(config.triggers.crons).toEqual(['* * * * *']);
    expect(config).not.toHaveProperty('location_hint');
  });
});
