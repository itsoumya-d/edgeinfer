// Copyright (c) 2024-2026 Soumya Debnath. All Rights Reserved.
// Licensed under the Business Source License 1.1 (BSL 1.1).
// See LICENSE file for details. Production use requires a paid license.
// Contact: soumyadebnath1619@gmail.com

import { EventEmitter } from './events';

export class ModelCache extends EventEmitter {
  private cacheName: string;

  constructor(cacheName: string = 'edgeinfer-models') {
    super();
    this.cacheName = cacheName;
  }

  async getModel(url: string, forceDownload: boolean = false): Promise<ArrayBuffer> {
    if (typeof caches === 'undefined') {
      // No Cache API (Node.js, or a browser without it): plain fetch.
      // The HTTP status MUST still be checked here. Without this, a 404/500
      // error page body is handed straight to the ONNX parser, which then
      // fails with an opaque "protobuf parsing failed" instead of telling the
      // caller the model URL was wrong.
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(
          `EdgeInfer: failed to fetch model from ${url} - HTTP ${response.status} ${response.statusText}`
        );
      }
      return response.arrayBuffer();
    }

    const cache = await caches.open(this.cacheName);
    const cachedResponse = await cache.match(url);

    if (cachedResponse && !forceDownload) {
      return cachedResponse.arrayBuffer();
    }

    this.emit('downloadStart', url);
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(
        `EdgeInfer: failed to fetch model from ${url} - HTTP ${response.status} ${response.statusText}`
      );
    }

    await cache.put(url, response.clone());
    this.emit('downloadComplete', url);

    return response.arrayBuffer();
  }
}
