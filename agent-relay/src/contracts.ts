import appContextSource from '../prompts/agent/app-context.md?raw';
import { appTools } from './app-tools';
import { appTools as appToolsV6 } from './app-tools-v6';
import { contextModules } from './context-modules';

export type Contract = {
  tools: readonly ({ name: string } & Record<string, unknown>)[];
  contextModules: readonly { name: string; description: string }[];
  appContext: string;
  setupControls: string;
  executionBoundary: string;
};

export const currentProtocol = 7;

export const contracts: ReadonlyMap<number, Contract> = new Map([
  [6, {
    tools: appToolsV6, contextModules, appContext: appContextSource,
    setupControls: 'Use retained facts before another read. Explicit execute_app actions ui.accounts and ui.provider_setup stage account or provider controls; reads never do. ui.provider_setup also returns setup metadata. Staging is not display, user click, verification or readiness.',
    executionBoundary: '- A repair request authorizes relevant reversible local fixes within tool permissions. When the target and safe next action are clear, do it rather than ask permission again. Batch at most four tool calls; they run in order. Independent reads may share a batch but are not one atomic snapshot. Approval, required user action, pending work or a failure stops later effects. Match claims to the terminal operation result, configuration and time; saved settings, running components and actual output are different facts.',
  }],
  [7, {
    tools: appTools, contextModules, appContext: appContextSource,
    setupControls: 'Use retained facts before another read. Explicit open_app views accounts and provider_setup stage account or provider controls; reads never do. provider_setup also returns setup metadata. Staging is not display, user click, verification or readiness.',
    executionBoundary: '- A repair request authorizes relevant reversible local fixes within tool permissions. When the target and safe next action are clear, do it rather than ask permission again. Batch at most four tool calls; they run in order. Independent reads may share a batch but are not one atomic snapshot. Within the current batch, approval, required user action, pending work or a failure stops subsequent potentially dependent effects; independent reads still run and every call gets a result. Reconsider after those results with the normal tools available on your next response. Resolve pending or uncertain work by its retained operation identity, not a fresh submission. Match claims to the terminal operation result, configuration and time; saved settings, running components and actual output are different facts.',
  }],
]);

export const supportedProtocols = [...contracts.keys()].sort((a, b) => a - b);

export function contractFor(protocol: unknown): Contract | undefined {
  return typeof protocol === 'number' ? contracts.get(protocol) : undefined;
}
