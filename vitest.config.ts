import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'node',
    include: ['services/broker/tests/**/*.spec.ts', 'services/agent-relay/tests/**/*.spec.ts'],
  },
});
