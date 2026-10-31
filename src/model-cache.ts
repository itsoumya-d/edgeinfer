// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Soumya Debnath

import { EventEmitter } from './events';
import { withDownloadDeadline, DownloadOptions, checkDownloadAbort } from './download';

export class ModelCache extends EventEmitter {
  private cacheName: string;

  constructor(cacheName: string = 'edgeinfer-models') {
    super();
    this.cacheName = cacheName;
  }

  async getModel(url: string, forceDownload: boolean = false, options: DownloadOptions = {}): Promise<ArrayBuffer> {
    return withDownloadDeadline(async signal => {
      const cache = typeof caches === 'undefined' ? undefined : await caches.open(this.cacheName);
      checkDownloadAbort(signal);
      const cached = !forceDownload && cache ? await cache.match(url) : undefined;
      checkDownloadAbort(signal);
      if (cached) {
        const buffer = await cached.arrayBuffer();
        checkDownloadAbort(signal);
        return buffer;
      }
      this.emit('downloadStart', url);
      checkDownloadAbort(signal);
      const response = await fetch(url, { signal });
      if (!response.ok) {
        await response.body?.cancel();
        throw new Error(`EdgeInfer: failed to fetch model from ${url} - HTTP ${response.status} ${response.statusText}`);
      }
      if (response.status === 206 || response.headers.has('content-range')) {
        await response.body?.cancel();
        throw new Error('EdgeInfer: partial model responses are not supported');
      }
      // Read completely before committing, so aborted/truncated bodies cannot be cached.
      const buffer = await response.arrayBuffer();
      checkDownloadAbort(signal);
      if (cache) await cache.put(url, new Response(buffer, { headers: response.headers }));
      checkDownloadAbort(signal);
      this.emit('downloadComplete', url);
      return buffer;
    }, options);
  }
}
