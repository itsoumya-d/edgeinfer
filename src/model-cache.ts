// Copyright (c) 2024-2026 Soumya Debnath. All Rights Reserved.
// Licensed under the Business Source License 1.1 (BSL 1.1).
// See LICENSE file for details. Production use requires a paid license.
// Contact: soumyadebnath1661@gmail.com | +91 7031648617

import { EventEmitter } from './events';

export class ModelCache extends EventEmitter {
  private cacheName: string;

  constructor(cacheName: string = 'edgeinfer-models') {
    super();
    this.cacheName = cacheName;
  }

  async getModel(url: string, forceDownload: boolean = false): Promise<ArrayBuffer> {
    if (typeof caches === 'undefined') {
      const response = await fetch(url);
      return response.arrayBuffer();
    }

    const cache = await caches.open(this.cacheName);
    const cachedResponse = await cache.match(url);

    if (cachedResponse && !forceDownload) {
      return cachedResponse.arrayBuffer();
    }

    this.emit('downloadStart', url);
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Failed to fetch model from ${url}`);
    
    await cache.put(url, response.clone());
    this.emit('downloadComplete', url);
    
    return response.arrayBuffer();
  }
}
