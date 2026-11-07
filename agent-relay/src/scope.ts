import { effortInstructions, routeInstructions } from './instructions';

export const questions = [
  {
    type: 'choice',
    name: 'route',
    instructions: routeInstructions,
    choices: [
      { value: 'work', description: 'Advance app help, including recognition/translation model or provider choices, setup and troubleshooting, without requiring the app name or a prior task.' },
      { value: 'social', description: 'A brief greeting, thanks, farewell or completed-task acknowledgement, without a remaining request or diagnostic information.' },
      { value: 'out_of_scope', description: 'Engaging would require a role beyond app help or a brief social acknowledgement, even without an explicit task request.' },
      { value: 'unclear', description: 'App help is plausible, but the intended task or referent is uncertain.' },
    ],
  },
  {
    type: 'choice',
    name: 'effort',
    instructions: effortInstructions,
    choices: [
      { value: 'low', description: 'App facts, a clear single action, or a brief social/refusal reply.' },
      { value: 'high', description: 'Setup, recommendation, comparison, ordinary troubleshooting, or unclear intent.' },
      { value: 'xhigh', description: 'Complex diagnosis combining evidence or an unresolved problem after a relevant repair.' },
    ],
  },
];
