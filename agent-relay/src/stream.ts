import { isRecord } from './type-guards';

type Settlement = (result: unknown) => Promise<void>;

async function* upstreamEvents(reader: ReadableStreamDefaultReader<Uint8Array>): AsyncGenerator<unknown> {
  const decoder = new TextDecoder();
  let fragments: string[] = [];
  let data: string[] = [];
  while (true) {
    const { value, done } = await reader.read();
    const chunk = decoder.decode(value, { stream: !done });
    let offset = 0;
    let boundary: number;
    while ((boundary = chunk.indexOf('\n', offset)) !== -1) {
      let line = chunk.slice(offset, boundary);
      offset = boundary + 1;
      if (fragments.length) {
        fragments.push(line);
        line = fragments.join('');
        fragments = [];
      }
      if (line.endsWith('\r')) line = line.slice(0, -1);
      if (line.startsWith('data:')) data.push(line.slice(line.charCodeAt(5) === 32 ? 6 : 5));
      else if (!line && data.length) {
        yield JSON.parse(data.join('\n'));
        data = [];
      }
    }
    if (offset < chunk.length) fragments.push(chunk.slice(offset));
    if (done) {
      if (fragments.length || data.length) throw new Error('upstream_invalid_response');
      return;
    }
  }
}

export function responseStream(
  upstream: Response,
  abort: AbortController,
  requestSignal: AbortSignal,
  settle: Settlement,
  release: () => void,
  context?: Pick<ExecutionContext, 'waitUntil'>,
): Response {
  const reader = upstream.body!.getReader();
  const encoder = new TextEncoder();
  let cancelled = false;
  let downstreamCancelled = false;
  let completed = false;
  let result: unknown;
  let pump: Promise<void>;
  const interrupt = () => { void reader.cancel().catch(() => {}); };
  const stop = () => {
    cancelled = true;
    abort.abort();
    void reader.cancel().catch(() => {});
  };
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      const emit = (event: Record<string, unknown>) => {
        if (!cancelled) controller.enqueue(encoder.encode(`data: ${JSON.stringify(event)}\n\n`));
      };
      requestSignal.addEventListener('abort', stop, { once: true });
      abort.signal.addEventListener('abort', interrupt, { once: true });
      if (abort.signal.aborted) interrupt();
      if (requestSignal.aborted) stop();
      pump = (async () => {
        let code = 'upstream_unavailable';
        try {
          for await (const event of upstreamEvents(reader)) {
            if (cancelled) break;
            if (!isRecord(event)) throw new Error('upstream_invalid_response');
            if (event.type === 'response.output_text.delta') {
              if (typeof event.delta !== 'string') throw new Error('upstream_invalid_response');
              emit({ type: 'text_delta', text: event.delta });
            } else if (['response.completed', 'response.incomplete', 'response.failed'].includes(event.type as string)) {
              result = event.response;
              if (!isRecord(result) || result.object !== 'response' || !Array.isArray(result.output)) {
                throw new Error('upstream_invalid_response');
              }
              abort.abort();
              await reader.cancel().catch(() => {});
              if (event.type === 'response.failed') throw new Error('upstream_unavailable');
              code = 'service_shutdown';
              await settle(result);
              emit({ type: 'completed', response: result });
              completed = true;
              return;
            } else if (event.type === 'error') {
              throw new Error('upstream_unavailable');
            }
          }
        } catch (error) {
          if (error instanceof Error && error.message === 'upstream_invalid_response') code = error.message;
        } finally {
          abort.abort();
          await reader.cancel().catch(() => {});
          reader.releaseLock();
          requestSignal.removeEventListener('abort', stop);
          abort.signal.removeEventListener('abort', interrupt);
          release();
          try {
            await settle(result);
          } catch {
            code = 'service_shutdown';
          }
          if (!cancelled && !completed) emit({ type: 'error', error: { code } });
          if (!downstreamCancelled) controller.close();
        }
      })();
      context?.waitUntil(pump);
    },
    async cancel() {
      downstreamCancelled = true;
      stop();
      await pump;
    },
  });
  return new Response(body, { headers: { 'content-type': 'text/event-stream', 'cache-control': 'no-store' } });
}
