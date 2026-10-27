import { readFileSync } from 'node:fs';

import { describe, expect, it } from 'vitest';

import { REQUIRED_BINDINGS } from '../src/contract';

describe('broker hosting assumptions', () => {
  it('binds the worker to the broker entrypoint and native D1 config without unsupported location fields', () => {
    const wranglerConfig = readFileSync(
      new URL('../wrangler.jsonc', import.meta.url),
      'utf8',
    );

    expect(wranglerConfig).toContain('"main": "src/index.ts"');
    expect(wranglerConfig).toContain(`"binding": "${REQUIRED_BINDINGS.d1}"`);
    expect(wranglerConfig).toMatch(/"crons"\s*:\s*\[\s*"\* \* \* \* \*"\s*\]/u);
    expect(wranglerConfig).not.toContain('"location_hint":');
  });

  it('pins wrangler in package metadata for reproducible deploy tooling', () => {
    const packageJson = JSON.parse(
      readFileSync(new URL('../package.json', import.meta.url), 'utf8'),
    ) as {
      scripts: Record<string, string>;
      devDependencies?: Record<string, string>;
    };

    expect(packageJson.scripts).toMatchObject({
      dev: 'wrangler dev --config wrangler.jsonc',
      deploy: 'wrangler deploy --config wrangler.jsonc',
      'verify:config': 'wrangler types --config wrangler.jsonc',
    });
    expect(packageJson.devDependencies?.wrangler).toBeDefined();
  });
});
