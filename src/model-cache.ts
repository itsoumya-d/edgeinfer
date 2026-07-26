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
