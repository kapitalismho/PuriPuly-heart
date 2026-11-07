import appContextSource from '../prompts/agent/app-context.md?raw';
import { tools } from './bundle';
import { contextModules } from './context-modules';
import { protocol5 } from './contracts/protocol-5';

export type Contract = {
  tools: readonly ({ name: string } & Record<string, unknown>)[];
  contextModules: readonly { name: string; description: string }[];
  appContext: string;
};

export const currentProtocol = 6;

export const contracts: ReadonlyMap<number, Contract> = new Map<number, Contract>([
  [5, protocol5],
  [currentProtocol, { tools, contextModules, appContext: appContextSource }],
]);

export const supportedProtocols = [...contracts.keys()].sort((a, b) => a - b);

export function contractFor(protocol: unknown): Contract | undefined {
  return typeof protocol === 'number' ? contracts.get(protocol) : undefined;
}
