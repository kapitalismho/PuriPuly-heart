import { isRecord } from './type-guards';

type Item = { id: string; kind: string; phase: unknown; done: boolean; pending: string[] };
type Identity = [number, string];

function invalid(): never { throw new Error('upstream_invalid_response'); }
function phase(item: Record<string, unknown>): unknown {
  const value = item.phase ?? null;
  if (value !== null && value !== 'commentary' && value !== 'final_answer') invalid();
  return value;
}
function identity(index: unknown, id: unknown): Identity {
  if (typeof index !== 'number' || !Number.isSafeInteger(index) || index < 0 || typeof id !== 'string' || !id) invalid();
  return [index, id];
}

export class PhotoResponseOutput {
  private items = new Map<number, Item>();
  private selected: number | undefined;
  private anonymous: string[] = [];
  private anonymousIdentity: Identity | undefined;
  private anonymousSeen = false;

  accept(event: Record<string, unknown>): string {
    if (event.type === 'response.output_item.added' || event.type === 'response.output_item.done') {
      const item = event.item;
      if (!isRecord(item) || typeof item.type !== 'string') invalid();
      const [index, id] = identity(event.output_index, item.id);
      let itemPhase = item.type === 'message' ? phase(item) : null;
      if (item.type === 'message' && item.role !== 'assistant') invalid();
      if (event.type === 'response.output_item.added') {
        if (this.anonymousSeen || this.items.has(index)
          || [...this.items.values()].some((value) => value.id === id)
          || (this.items.size > 0 && index <= Math.max(...this.items.keys()))) invalid();
        this.items.set(index, { id, kind: item.type, phase: itemPhase, done: false, pending: [] });
      } else {
        const previous = this.items.get(index);
        if (!previous || previous.done || previous.id !== id || previous.kind !== item.type
          || (previous.phase !== null && itemPhase !== null && previous.phase !== itemPhase)) invalid();
        itemPhase ??= previous.phase;
        previous.phase = itemPhase;
        previous.done = true;
      }
      if (itemPhase === 'final_answer') {
        if (this.selected !== undefined && this.selected !== index) invalid();
        this.selected = index;
        const pending = this.items.get(index)!.pending;
        const text = pending.join('');
        pending.length = 0;
        return text;
      }
      return '';
    }
    if (event.type !== 'response.output_text.delta') return '';
    if (typeof event.delta !== 'string') invalid();
    if ('content_index' in event && event.content_index !== 0) invalid();
    const key = 'output_index' in event || 'item_id' in event ? identity(event.output_index, event.item_id) : undefined;
    if (!this.items.size) {
      if (this.anonymousSeen && (key?.[0] !== this.anonymousIdentity?.[0] || key?.[1] !== this.anonymousIdentity?.[1])) invalid();
      this.anonymousSeen = true;
      this.anonymousIdentity = key;
      this.anonymous.push(event.delta);
      return '';
    }
    if (!key) invalid();
    const item = this.items.get(key[0]);
    if (!item || item.id !== key[1] || item.kind !== 'message' || item.done) invalid();
    if (item.phase === 'final_answer') return event.delta;
    if (item.phase === null) item.pending.push(event.delta);
    return '';
  }

  finish(response: Record<string, unknown>): string {
    const output = response.output;
    if (output != null && !Array.isArray(output)) invalid();
    const messages: [number, Record<string, unknown>][] = [];
    for (const [index, item] of (output ?? []).entries()) {
      if (!isRecord(item)) invalid();
      if (item.type === 'message') {
        if (item.role !== 'assistant') invalid();
        phase(item);
        messages.push([index, item]);
      }
    }
    if (this.items.size && output?.length) {
      const known = [...this.items].filter(([, item]) => item.kind === 'message').map(([index]) => index);
      if (known.length !== messages.length || messages.some(([index]) => !known.includes(index))) invalid();
      for (const [index, terminal] of messages) {
        const item = this.items.get(index)!;
        const itemPhase = phase(terminal);
        if (item.id !== terminal.id || (item.phase !== null && itemPhase !== null && item.phase !== itemPhase)) invalid();
        item.phase = itemPhase ?? item.phase;
      }
    }
    if (this.items.size) {
      let candidates = [...this.items].filter(([, item]) => item.phase === 'final_answer');
      if (!candidates.length) {
        candidates = [...this.items].filter(([, item]) => item.kind === 'message' && item.phase === null);
        if ([...this.items.values()].filter((item) => item.kind === 'message').length !== 1) invalid();
      }
      if (candidates.length !== 1 || (this.selected !== undefined && candidates[0][0] !== this.selected)) invalid();
      const pending = candidates[0][1].pending;
      const text = pending.join('');
      pending.length = 0;
      return text;
    }
    if (messages.length) {
      if (messages.length !== 1 || phase(messages[0][1]) === 'commentary') invalid();
      if (this.anonymousIdentity && (this.anonymousIdentity[0] !== messages[0][0] || this.anonymousIdentity[1] !== messages[0][1].id)) invalid();
    }
    const text = this.anonymous.join('');
    this.anonymous.length = 0;
    return text;
  }
}
