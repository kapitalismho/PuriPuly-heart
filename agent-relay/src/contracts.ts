import appContextSource from '../prompts/agent/app-context.md?raw';
import { appTools } from './app-tools';
import { contextModules } from './context-modules';

export type Contract = {
  tools: readonly ({ name: string } & Record<string, unknown>)[];
  contextModules: readonly { name: string; description: string }[];
  appContext: string;
  setupControls: string;
};

export const currentProtocol = 6;

export const contracts: ReadonlyMap<number, Contract> = new Map([
  [6, {
    tools: appTools, contextModules, appContext: appContextSource,
    setupControls: 'Use retained facts before another read. Explicit execute_app actions ui.accounts and ui.provider_setup stage account or provider controls; reads never do. ui.provider_setup also returns setup metadata. Staging is not display, user click, verification or readiness.',
  }],
]);

export const supportedProtocols = [...contracts.keys()].sort((a, b) => a - b);

export function contractFor(protocol: unknown): Contract | undefined {
  return typeof protocol === 'number' ? contracts.get(protocol) : undefined;
}
