// Copyright (c) 2024-2026 Soumya Debnath. All Rights Reserved.
// Licensed under the Business Source License 1.1 (BSL 1.1).
// See LICENSE file for details. Production use requires a paid license.
// Contact: soumyadebnath1661@gmail.com | +91 7031648617

import * as ort from 'onnxruntime-web';

export class RuntimeManager {
  static async getBestExecutionProvider(): Promise<string> {
    if (typeof navigator !== 'undefined' && (navigator as any).gpu) {
      return 'webgpu';
    }
    return 'wasm';
  }

  static async createSession(
    modelBuffer: ArrayBuffer,
    providers?: string[]
  ): Promise<ort.InferenceSession> {
    const eps = providers || [await this.getBestExecutionProvider()];
    return await ort.InferenceSession.create(modelBuffer, { executionProviders: eps });
  }
}
