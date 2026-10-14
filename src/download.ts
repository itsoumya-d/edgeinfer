/** Per-transfer options; covers response headers and the entire response body. */
export interface DownloadOptions {
  signal?: AbortSignal;
  /** Default 120 seconds per model/tokenizer transfer, not an inference timeout. */
  downloadTimeoutMs?: number;
}

export async function withDownloadDeadline<T>(operation: (signal: AbortSignal) => Promise<T>, options: DownloadOptions = {}): Promise<T> {
  const timeout = options.downloadTimeoutMs ?? 120000;
  if (!Number.isFinite(timeout) || timeout <= 0 || timeout > 2147483647) {
    throw new RangeError('downloadTimeoutMs must be between 0 and 2147483647 milliseconds');
  }
  const controller = new AbortController();
  const abort = () => controller.abort(options.signal?.reason);
  if (options.signal?.aborted) abort();
  else options.signal?.addEventListener('abort', abort, { once: true });
  const timer = setTimeout(() => controller.abort(new DOMException('Download deadline exceeded', 'TimeoutError')), timeout);
  let onAbort: () => void = () => {};
  try {
    controller.signal.throwIfAborted();
    const cancelled = new Promise<never>((_, reject) => {
      onAbort = () => reject(controller.signal.reason);
      controller.signal.addEventListener('abort', onAbort, { once: true });
    });
    return await Promise.race([operation(controller.signal), cancelled]);
  } finally {
    clearTimeout(timer);
    controller.signal.removeEventListener('abort', onAbort);
    options.signal?.removeEventListener('abort', abort);
  }
}
