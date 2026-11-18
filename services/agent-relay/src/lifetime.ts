export const requestDeadlineMs = 300_000;
export const requestLeaseMs = 360_000;

export function requestLifetime(deadline: number, source: AbortSignal) {
  const abort = new AbortController();
  const cancel = () => abort.abort();
  const timer = setTimeout(cancel, Math.max(0, deadline - Date.now()));
  source.addEventListener('abort', cancel, { once: true });
  if (source.aborted || deadline <= Date.now()) cancel();
  return { abort, release() {
    clearTimeout(timer);
    source.removeEventListener('abort', cancel);
  } };
}

export function beforeAbort<T>(signal: AbortSignal, operation: () => Promise<T>): Promise<T> {
  if (signal.aborted) return Promise.reject(new Error('request_aborted'));
  const { promise, resolve, reject } = Promise.withResolvers<T>();
  const cancelled = () => reject(new Error('request_aborted'));
  signal.addEventListener('abort', cancelled, { once: true });
  try {
    operation().then((value) => {
      signal.removeEventListener('abort', cancelled);
      resolve(value);
    }, (error: unknown) => {
      signal.removeEventListener('abort', cancelled);
      reject(error);
    });
  } catch (error) {
    signal.removeEventListener('abort', cancelled);
    reject(error);
  }
  return promise;
}
