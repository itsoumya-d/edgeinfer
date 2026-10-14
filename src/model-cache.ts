// Copyright (c) 2024-2026 Soumya Debnath. All Rights Reserved.
// Licensed under the Business Source License 1.1 (BSL 1.1).
// See LICENSE file for details. Production use requires a paid license.
// Contact: soumyadebnath1619@gmail.com

import { EventEmitter } from './events';
import { withDownloadDeadline, DownloadOptions } from './download';

export class ModelCache extends EventEmitter {
  private cacheName: string;

  constructor(cacheName: string = 'edgeinfer-models') {
    super();
    this.cacheName = cacheName;
  }

  async getModel(url: string, forceDownload: boolean = false, options: DownloadOptions = {}): Promise<ArrayBuffer> {
    return withDownloadDeadline(async signal => {
      const cache = typeof caches === 'undefined' ? undefined : await caches.open(this.cacheName);
      signal.throwIfAborted();
      const cached = !forceDownload && cache ? await cache.match(url) : undefined;
      signal.throwIfAborted();
      if (cached) {
        const buffer = await cached.arrayBuffer();
        signal.throwIfAborted();
        return buffer;
      }
      this.emit('downloadStart', url);
      signal.throwIfAborted();
      const response = await fetch(url, { signal });
      if (!response.ok) {
        await response.body?.cancel();
        throw new Error(`EdgeInfer: failed to fetch model from ${url} - HTTP ${response.status} ${response.statusText}`);
      }
      // Read completely before committing, so aborted/truncated bodies cannot be cached.
      const buffer = await response.arrayBuffer();
      signal.throwIfAborted();
      if (cache) await cache.put(url, new Response(buffer, { headers: response.headers }));
      signal.throwIfAborted();
      this.emit('downloadComplete', url);
      return buffer;
    }, options);
  }
}
