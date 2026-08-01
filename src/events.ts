// Copyright (c) 2024-2026 Soumya Debnath. All Rights Reserved.
// Licensed under the Business Source License 1.1 (BSL 1.1).
// See LICENSE file for details. Production use requires a paid license.
// Contact: soumyadebnath1619@gmail.com

export class EventEmitter {
  private listeners: Record<string, Function[]> = {};

  on(event: string, callback: Function) {
    if (!this.listeners[event]) this.listeners[event] = [];
    this.listeners[event].push(callback);
  }

  emit(event: string, ...args: any[]) {
    if (this.listeners[event]) {
      this.listeners[event].forEach(cb => cb(...args));
    }
  }
}
