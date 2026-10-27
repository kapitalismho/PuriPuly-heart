import { describe, expect, it } from 'vitest';

import app from '../src/index';

describe('broker non-proxy boundary', () => {
  it('does not expose a translation proxy endpoint', async () => {
    const response = await app.request('http://broker.test/v1/translate', {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
      },
      body: JSON.stringify({ text: 'hello world' }),
    });

    expect(response.status).toBe(404);
  });
});
