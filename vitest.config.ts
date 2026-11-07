import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'node',
    include: ['broker/tests/**/*.spec.ts', 'agent-relay/tests/**/*.spec.ts'],
  },
});
