import scopeSource from '../prompts/agent/scope.md?raw';
import scopeReferencesSource from '../prompts/agent/scope-references.md?raw';
import systemSource from '../prompts/agent/system.md?raw';
import guideSource from '../prompts/agent/guide.md?raw';
import costsSource from '../prompts/agent/references/keys-and-costs.md?raw';
import benchmarksSource from '../prompts/agent/references/benchmark.md?raw';
import setupSource from '../prompts/agent/references/provider-setup.md?raw';
import photoSource from '../prompts/agent/references/vr-photo.md?raw';
import { type Contract, type ReferenceName, referenceNames } from './contracts';

function prompt(source: string, name: string): string {
  const text = source.replaceAll('\r\n', '\n').trim();
  if (!text) throw new Error(`Empty prompt: ${name}`);
  return text;
}

function requirePlaceholder(source: string, marker: string) {
  if (source.split(marker).length !== 2) throw new Error(`Expected exactly one prompt placeholder: ${marker}`);
}

const scope = prompt(scopeSource, 'scope');
const scopeReferences = prompt(scopeReferencesSource, 'scope references');
const modulesMarker = '{{context_modules}}';
requirePlaceholder(scope, modulesMarker);
const system = systemSource.replaceAll('\r\n', '\n');
const guide = prompt(guideSource, 'guide');
const appContextMarker = '{{app_context}}';
const setupControlsMarker = '{{setup_controls}}';
requirePlaceholder(guide, appContextMarker);
requirePlaceholder(guide, setupControlsMarker);
const referenceTexts = {
  costs: prompt(costsSource, 'costs reference'),
  benchmarks: prompt(benchmarksSource, 'benchmarks reference'),
  setup: prompt(setupSource, 'setup reference'),
  photo: prompt(photoSource, 'photo reference'),
};
const photoAllowanceMarker = '{{photo_daily_allowance}}';
requirePlaceholder(referenceTexts.photo, photoAllowanceMarker);

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
const work = section(system, '## Work and unclear requests');
const executionBoundaryMarker = '{{execution_boundary}}';
requirePlaceholder(work, executionBoundaryMarker);

export function classifierInstructions(contract: Contract): string {
  const rendered = scope.replace(modulesMarker,
    () => contract.contextModules.map(({ name, description }) => `- ${name}: ${description}`).join('\n'));
  return contract.selectReferences ? `${rendered}\n\n${scopeReferences}` : rendered;
}

export function referenceInstructions(photoAllowance: number, omitted: readonly ReferenceName[] = []): string {
  return referenceNames.filter((name) => !omitted.includes(name))
    .map((name) => name === 'photo'
      ? referenceTexts.photo.replace(photoAllowanceMarker, () => String(photoAllowance))
      : referenceTexts[name])
    .join('\n\n');
}

export function workInstructions(contract: Contract, photoAllowance: number): string {
  const appContext = contract.appContext.replaceAll('\r\n', '\n').trim();
  if (!appContext) throw new Error('Empty contract prompt: app context');
  const contractGuide = guide.replace(appContextMarker, () => appContext)
    .replace(setupControlsMarker, () => contract.setupControls);
  const contractWork = work.replace(executionBoundaryMarker, () => contract.executionBoundary);
  const prefix = `${identityAndVoice}\n\n${contractWork}\n\n${contractGuide}`;
  return contract.selectReferences ? prefix : `${prefix}\n\n${referenceInstructions(photoAllowance)}`;
}

export const socialInstructions = `${identityAndVoice}\n\n${section(system, '## Social acknowledgements')}`;
export const outOfScopeInstructions = `${identityAndVoice}\n\n${section(system, '## Out-of-scope responses')}`;
