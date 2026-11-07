import scopeSource from '../prompts/agent/scope.md?raw';
import systemSource from '../prompts/agent/system.md?raw';
import guideSource from '../prompts/agent/guide.md?raw';
import costsSource from '../prompts/agent/references/keys-and-costs.md?raw';
import benchmarksSource from '../prompts/agent/references/benchmark.md?raw';
import type { Contract } from './contracts';

const scope = scopeSource.replaceAll('\r\n', '\n').trim();
const modulesMarker = '{{context_modules}}';
if (scope.split(modulesMarker).length !== 2) throw new Error('Missing prompt placeholder: context modules');
const system = systemSource.replaceAll('\r\n', '\n');
const guide = guideSource.replaceAll('\r\n', '\n').trim();
if (!guide) throw new Error('Empty prompt: guide');
const appContextMarker = '{{app_context}}';
if (guide.split(appContextMarker).length !== 2) throw new Error('Missing prompt placeholder: app context');
const costs = costsSource.replaceAll('\r\n', '\n').trim();
if (!costs) throw new Error('Empty reference: costs');
const benchmarks = benchmarksSource.replaceAll('\r\n', '\n').trim();
if (!benchmarks) throw new Error('Empty reference: benchmarks');

function section(source: string, heading: string): string {
  const marker = `${heading}\n`;
  const start = source.indexOf(`\n${marker}`);
  if (start === -1) throw new Error(`Missing prompt section: ${heading}`);
  const bodyStart = start + marker.length + 1;
  const end = source.indexOf('\n## ', bodyStart);
  const text = source.slice(bodyStart, end === -1 ? undefined : end).trim();
  if (!text) throw new Error(`Empty prompt section: ${heading}`);
  return text;
}

const identityAndVoice = section(system, '## Identity and voice');

export function classifierInstructions(contract: Contract): string {
  return scope.replace(modulesMarker,
    () => contract.contextModules.map(({ name, description }) => `- ${name}: ${description}`).join('\n'));
}

export function workInstructions(contract: Contract): string {
  const appContext = contract.appContext.replaceAll('\r\n', '\n').trim();
  if (!appContext) throw new Error('Empty contract prompt: app context');
  const contractGuide = guide.replace(appContextMarker, () => appContext);
  return `${identityAndVoice}\n\n${section(system, '## Work and unclear requests')}\n\n${contractGuide}\n\n${costs}\n\n${benchmarks}`;
}

export const socialInstructions = `${identityAndVoice}\n\n${section(system, '## Social acknowledgements')}`;
export const outOfScopeInstructions = `${identityAndVoice}\n\n${section(system, '## Out-of-scope responses')}`;
