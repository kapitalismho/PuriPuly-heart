import type { Contract } from './contracts';

const choices = (...values: string[]) => values.map((value) => ({ value }));

export function classifierQuestions(contract: Contract) {
  return [
    { type: 'choice', name: 'route', instructions: 'Classify current_message by the Route section.', choices: choices('work', 'social', 'out_of_scope', 'unclear') },
    { type: 'choice', name: 'effort', instructions: 'Choose the reply effort by the Effort section.', choices: choices('low', 'high', 'xhigh') },
    ...contract.contextModules.map(({ name }) => ({
      type: 'choice',
      name: `context_${name}`,
      instructions: `Decide whether the ${name} module is needed by the Context modules section.`,
      choices: choices('include', 'omit', 'unclear'),
    })),
    { type: 'choice', name: 'context_relation', instructions: 'Classify the task relation by the Task relation section.', choices: choices('new', 'continue', 'unclear') },
  ];
}
