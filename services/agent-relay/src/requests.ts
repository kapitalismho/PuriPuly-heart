import { Contract, contractFor } from './contracts';
import { isRecord } from './type-guards';

export type RespondRequest = {
  agent_protocol: number;
  turn_id: string;
  input: Record<string, unknown>[];
  effort: 'low' | 'medium' | 'high';
  allow_tools: boolean;
  route: 'work' | 'social' | 'out_of_scope' | 'unclear';
};

export type ClassifyRequest = {
  agent_protocol: number;
  turn_id: string;
  input: { role: 'user'; content: string }[];
};

function only(value: Record<string, unknown>, keys: string[]): boolean {
  return Object.keys(value).every((key) => keys.includes(key));
}

function effort(value: unknown): value is RespondRequest['effort'] {
  return value === 'low' || value === 'medium' || value === 'high';
}

function optionalString(value: unknown): boolean {
  return value === undefined || value === null || typeof value === 'string';
}

function status(value: unknown): boolean {
  return value === undefined || value === null || ['completed', 'incomplete', 'in_progress'].includes(value as string);
}

function contentPart(value: unknown, role: unknown): boolean {
  if (!isRecord(value)) return false;
  if (value.type === 'input_text') return only(value, ['type', 'text']) && typeof value.text === 'string';
  if (role !== 'assistant') return false;
  if (value.type === 'refusal') return only(value, ['type', 'refusal']) && typeof value.refusal === 'string';
  return value.type === 'output_text'
    && only(value, ['type', 'text', 'annotations', 'logprobs']) && typeof value.text === 'string'
    && (value.annotations === undefined || Array.isArray(value.annotations))
    && (value.logprobs === undefined || value.logprobs === null || Array.isArray(value.logprobs));
}

function inputItem(value: Record<string, unknown>, contract: Contract): boolean {
  if (value.type === undefined || value.type === 'message') {
    return only(value, ['type', 'role', 'content', 'id', 'status', 'phase'])
      && ['user', 'assistant'].includes(value.role as string)
      && optionalString(value.id) && status(value.status)
      && (value.phase === undefined || (value.role === 'assistant'
        && (value.phase === null || ['commentary', 'final_answer'].includes(value.phase as string))))
      && (typeof value.content === 'string'
        || (Array.isArray(value.content) && value.content.every((part) => contentPart(part, value.role))));
  }
  if (value.type === 'reasoning') {
    return only(value, ['id', 'type', 'summary', 'content', 'encrypted_content', 'status'])
      && optionalString(value.id) && optionalString(value.encrypted_content) && status(value.status)
      && Array.isArray(value.summary) && value.summary.every((part) => isRecord(part)
        && only(part, ['type', 'text']) && part.type === 'summary_text' && typeof part.text === 'string')
      && (value.content === undefined || value.content === null || (Array.isArray(value.content)
        && value.content.every((part) => isRecord(part) && only(part, ['type', 'text'])
          && part.type === 'reasoning_text' && typeof part.text === 'string')));
  }
  if (value.type === 'function_call') {
    return only(value, ['id', 'type', 'call_id', 'name', 'arguments', 'status', 'namespace'])
      && optionalString(value.id) && status(value.status)
      && (value.namespace === undefined || value.namespace === null)
      && typeof value.call_id === 'string' && value.call_id.length > 0
      && contract.tools.some((tool) => tool.name === value.name) && typeof value.arguments === 'string';
  }
  return value.type === 'function_call_output'
    && only(value, ['id', 'type', 'call_id', 'output', 'status'])
    && optionalString(value.id) && status(value.status)
    && typeof value.call_id === 'string' && value.call_id.length > 0 && typeof value.output === 'string';
}

export function isClassifyRequest(value: unknown): value is ClassifyRequest {
  return isRecord(value) && only(value, ['agent_protocol', 'turn_id', 'input'])
    && contractFor(value.agent_protocol) !== undefined && typeof value.turn_id === 'string' && /^[0-9a-f]{32}$/.test(value.turn_id)
    && Array.isArray(value.input) && value.input.length === 1
    && value.input.every((item: unknown) => isRecord(item) && only(item, ['role', 'content'])
      && item.role === 'user' && typeof item.content === 'string');
}

export function isRespondRequest(value: unknown): value is RespondRequest {
  if (!isRecord(value) || !only(value, ['agent_protocol', 'turn_id', 'input', 'effort', 'route', 'allow_tools'])) return false;
  const contract = contractFor(value.agent_protocol);
  if (!contract || typeof value.turn_id !== 'string' || !/^[0-9a-f]{32}$/.test(value.turn_id)
    || !effort(value.effort)
    || !['work', 'social', 'out_of_scope', 'unclear'].includes(value.route as string)
    || typeof value.allow_tools !== 'boolean' || !Array.isArray(value.input)) return false;
  const work = value.route === 'work' || value.route === 'unclear';
  let selectedEffort: RespondRequest['effort'] | undefined;
  for (const item of value.input) {
    if (!isRecord(item)) return false;
    if (item.type === 'configuration_update') {
      if (!work || !only(item, ['type', 'reasoning']) || !isRecord(item.reasoning)
        || !only(item.reasoning, ['effort']) || !effort(item.reasoning.effort)) return false;
      selectedEffort = item.reasoning.effort;
    } else if (!inputItem(item, contract)) return false;
  }
  if (work) return selectedEffort === value.effort;
  return value.route !== 'out_of_scope' || (value.input.length === 1
    && only(value.input[0], ['role', 'content']) && value.input[0].role === 'user'
    && typeof value.input[0].content === 'string');
}

export const maxClassifyBytes = 64 * 1024;
export const maxRespondBytes = 1024 * 1024;

export async function readRequestJson(request: Request, maximum: number): Promise<unknown> {
  const declared = request.headers.get('content-length');
  if (declared !== null && /^\d+$/.test(declared) && Number(declared) > maximum) {
    throw new Error('request_too_large');
  }
  if (!request.body) throw new Error('invalid_request');
  const reader = request.body.getReader();
  const decoder = new TextDecoder();
  let size = 0;
  let text = '';
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > maximum) {
        await reader.cancel().catch(() => {});
        throw new Error('request_too_large');
      }
      text += decoder.decode(value, { stream: true });
    }
    text += decoder.decode();
  } finally {
    reader.releaseLock();
  }
  return JSON.parse(text);
}
