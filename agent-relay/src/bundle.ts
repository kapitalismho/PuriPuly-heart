import { actionTools } from './action-tools';

export const tools = [
  ...actionTools,
  {
    type: 'function',
    name: 'propose_report',
    description: 'Offer the user a support-report send action with a concise problem summary. Returns immediately without uploading or waiting for a click. Only the user can authorize sending the Puri conversation and current app log.',
    strict: true,
    parameters: {
      type: 'object',
      properties: {
        summary: { type: 'string', minLength: 1, maxLength: 2000, pattern: '\\S' },
      },
      required: ['summary'],
      additionalProperties: false,
    },
  },
];
